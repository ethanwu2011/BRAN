"""Local, deterministic preprocessing helpers for clinical study inputs.

These helpers deliberately operate only on caller-provided in-memory values.
They do not read data, write data, or resolve identities across sources.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import struct
import unicodedata

import numpy as np


_CBC_FIELD_COUNT = 9
_ADMISSION_WINDOW_MINUTES = 24 * 60
_SNAPSHOT_WINDOW_MINUTES = 60
_ALLOWED_SOURCES = frozenset({"mimic", "eicu", "nwicu", "sicdb", "nhanes", "zigong"})
_SPLIT_NAMESPACE = b"bran-clinical-split-v1\x00"

# Canonical position for each field in a returned CBCSnapshot array.
CBC_FIELDS = ("hct", "hemoglobin", "mch", "mchc", "mcv", "plt", "rbc", "rdw", "wbc")


@dataclass(frozen=True, repr=False)
class CBCSnapshot:
    """A contemporaneous, admission-day CBC representation.

    ``values`` and ``selected_offsets_minutes`` contain NaN where no value was
    selected. The nine positions correspond to ``CBC_FIELDS``.
    """

    values: np.ndarray
    observed: np.ndarray
    conflicts: np.ndarray
    anchor_minutes: float
    selected_offsets_minutes: np.ndarray


def _readonly_copy(values: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _require_1d_array(value: object, error: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 1:
        raise ValueError(error)
    return value


def _require_numeric_array(value: object, error: str) -> np.ndarray:
    array = _require_1d_array(value, error)
    if array.dtype.kind not in "iuf":
        raise ValueError(error)
    return array


def _validated_snapshot_inputs(
    field_indices: object,
    offsets_minutes: object,
    values: object,
    observed: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fields = _require_1d_array(field_indices, "field_indices must be a one-dimensional integer array")
    if fields.dtype.kind not in "iu":
        raise ValueError("field_indices must be a one-dimensional integer array")
    offsets = _require_numeric_array(offsets_minutes, "offsets_minutes must be a one-dimensional numeric array")
    measurements = _require_numeric_array(values, "values must be a one-dimensional numeric array")
    mask = _require_1d_array(observed, "observed must be a one-dimensional boolean array")
    if mask.dtype != np.dtype(bool):
        raise ValueError("observed must be a one-dimensional boolean array")
    if not (len(fields) == len(offsets) == len(measurements) == len(mask)):
        raise ValueError("snapshot inputs must have aligned lengths")
    if np.any(fields < 0) or np.any(fields >= _CBC_FIELD_COUNT):
        raise ValueError("field_indices must be between 0 and 8")
    return (
        fields.astype(np.intp, copy=False),
        offsets.astype(np.float64, copy=False),
        measurements.astype(np.float64, copy=False),
        mask,
    )


def select_cbc_snapshot(
    field_indices: np.ndarray,
    offsets_minutes: np.ndarray,
    values: np.ndarray,
    observed: np.ndarray,
) -> CBCSnapshot:
    """Select an admission-day, near-contemporaneous CBC snapshot.

    The anchor is the earliest eligible measurement of any canonical CBC field.
    Each field is then selected independently from the first observed,
    finite value within the next hour (capped at 24 hours from admission).
    Conflicting values at that field's earliest selected timestamp are marked
    missing rather than being ordered, averaged, or replaced by later values.
    """
    fields, offsets, measurements, mask = _validated_snapshot_inputs(
        field_indices, offsets_minutes, values, observed
    )
    result_values = np.full(_CBC_FIELD_COUNT, np.nan, dtype=np.float64)
    result_observed = np.zeros(_CBC_FIELD_COUNT, dtype=bool)
    result_conflicts = np.zeros(_CBC_FIELD_COUNT, dtype=bool)
    result_offsets = np.full(_CBC_FIELD_COUNT, np.nan, dtype=np.float64)

    eligible = (
        mask
        & np.isfinite(offsets)
        & np.isfinite(measurements)
        & (offsets >= 0.0)
        & (offsets <= _ADMISSION_WINDOW_MINUTES)
    )
    if not np.any(eligible):
        return CBCSnapshot(
            _readonly_copy(result_values, dtype=np.float64),
            _readonly_copy(result_observed, dtype=bool),
            _readonly_copy(result_conflicts, dtype=bool),
            float("nan"),
            _readonly_copy(result_offsets, dtype=np.float64),
        )

    anchor = float(np.min(offsets[eligible]))
    cutoff = min(anchor + _SNAPSHOT_WINDOW_MINUTES, float(_ADMISSION_WINDOW_MINUTES))
    in_snapshot = eligible & (offsets >= anchor) & (offsets <= cutoff)
    for field in range(_CBC_FIELD_COUNT):
        candidates = in_snapshot & (fields == field)
        if not np.any(candidates):
            continue
        earliest = float(np.min(offsets[candidates]))
        tied_values = measurements[candidates & (offsets == earliest)]
        # A nonempty finite array is guaranteed by ``eligible``.  Exact equality
        # preserves duplicate same-value results while detecting genuine ties.
        if not np.all(tied_values == tied_values[0]):
            result_conflicts[field] = True
            continue
        result_values[field] = tied_values[0]
        result_observed[field] = True
        result_offsets[field] = earliest

    return CBCSnapshot(
        _readonly_copy(result_values, dtype=np.float64),
        _readonly_copy(result_observed, dtype=bool),
        _readonly_copy(result_conflicts, dtype=bool),
        anchor,
        _readonly_copy(result_offsets, dtype=np.float64),
    )


def _split_message(source: str, person_key: str) -> bytes:
    source_bytes = source.encode("utf-8")
    person_bytes = person_key.encode("utf-8")
    return (
        _SPLIT_NAMESPACE
        + struct.pack(">I", len(source_bytes))
        + source_bytes
        + struct.pack(">I", len(person_bytes))
        + person_bytes
    )


def _valid_person_key(person_key: object) -> bool:
    if not isinstance(person_key, str) or not person_key or person_key != person_key.strip():
        return False
    if any(unicodedata.category(character) == "Cc" for character in person_key):
        return False
    try:
        person_key.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def assign_person_split(source: str, person_key: str, *, salt: bytes) -> str:
    """Assign a source-namespaced person to a deterministic 80/10/10 split."""
    if not isinstance(source, str) or source not in _ALLOWED_SOURCES:
        raise ValueError("source must be one of the supported source names")
    if not _valid_person_key(person_key):
        raise ValueError("person_key must be a nonempty normalized string")
    if not isinstance(salt, bytes) or len(salt) < 16:
        raise ValueError("salt must be bytes with length at least 16")

    digest = hmac.new(salt, _split_message(source, person_key), hashlib.sha256).digest()
    bucket = (int.from_bytes(digest, byteorder="big", signed=False) * 100) >> 256
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"
