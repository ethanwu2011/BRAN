"""Closed, private MIMIC linked-snapshot packing contract.

This module is intentionally source-free: callers bind and canonicalize
original MIMIC measurements before constructing :class:`LinkedSnapshot`.
Only aggregate counts are returned publicly; encounter identifiers and row
bindings remain in the private map returned alongside the private arrays.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import hmac
import math
from numbers import Real
import struct

import numpy as np

from bran_clinical_chemistry_semantics_v1 import CHEMISTRY_FIELDS
from bran_clinical_semantics_v1 import AgeObservation, CBC_FIELDS
from bran_joint_lab_cache_v1 import AGE_KINDS, pack_records
from bran_nhanes_joint_labs_v1 import JointRecord


FIELDS = CBC_FIELDS + CHEMISTRY_FIELDS
LANDMARK_MINUTES = 1440.0
_SNAPSHOT_WINDOW_MINUTES = 60.0
_FIELD_COUNT = len(FIELDS)
_CBC_COUNT = len(CBC_FIELDS)
_SOURCE_HASH_KEYS = ("patients", "admissions", "labs", "dictionary")
_PRIVATE_ARRAY_KEYS = (
    "anchor_minutes",
    "cutoff_minutes",
    "field_minutes",
    "available_minutes",
    "conflicts",
    "row_binding",
)
_BASE_ARRAY_KEYS = (
    "values",
    "observed",
    "provenance",
    "calibration_code",
    "age_triplet",
    "age_kind",
    "adult_qualified",
    "person_group",
    "split",
    "person_weight",
    "cycle_group",
)
_ARRAY_KEYS = _BASE_ARRAY_KEYS + _PRIVATE_ARRAY_KEYS
_MAP_SCHEMA = "bran-mimic-linked-snapshot-map-v1"
_ROW_BINDING_DOMAIN = b"bran-mimic-linked-snapshot-row-binding-v1\x00"


@dataclass(frozen=True, repr=False)
class LinkedSnapshot:
    """A canonical day-one CBC/chemistry snapshot for one MIMIC episode."""

    person: str
    episode: str
    values: np.ndarray
    observed: np.ndarray
    conflicts: np.ndarray
    anchor_minutes: float
    field_minutes: np.ndarray
    available_minutes: np.ndarray
    age: AgeObservation


class _PrivateMap(dict):
    """A dict-compatible private map whose default representation is closed."""

    def __repr__(self) -> str:  # pragma: no cover - exercised by callers
        return "<private linked-snapshot map>"

    __str__ = __repr__


def _raise(message: str) -> None:
    # Keep all failures static.  In particular, never interpolate IDs or
    # caller-provided values into an exception.
    raise ValueError(message)


def _validate_salt(salt: object) -> None:
    if not isinstance(salt, bytes) or len(salt) != 32:
        _raise("invalid linked-snapshot salt")


def _validate_source_sha256(source_sha256: object) -> dict[str, str]:
    if not isinstance(source_sha256, Mapping):
        _raise("invalid linked-snapshot source hashes")
    try:
        if set(source_sha256) != set(_SOURCE_HASH_KEYS):
            _raise("invalid linked-snapshot source hashes")
    except (TypeError, ValueError):
        _raise("invalid linked-snapshot source hashes")
    result: dict[str, str] = {}
    for key in _SOURCE_HASH_KEYS:
        value = source_sha256[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or not value.isascii()
            or any(character not in "0123456789abcdef" for character in value)
        ):
            _raise("invalid linked-snapshot source hashes")
        result[key] = value
    return result


def _validate_numeric_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or not value.isdigit()
        or (len(value) > 1 and value[0] == "0")
    ):
        _raise("invalid linked-snapshot identifier")
    return value


def _as_finite_or_nonfinite_float(value: object, message: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        _raise(message)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        _raise(message)
    return result


def _validate_age(age: object) -> AgeObservation:
    if not isinstance(age, AgeObservation):
        _raise("invalid linked-snapshot age")
    if not isinstance(age.kind, str) or not isinstance(age.reference, str):
        _raise("invalid linked-snapshot age")
    if age.kind not in AGE_KINDS:
        _raise("invalid linked-snapshot age")

    reported = _as_finite_or_nonfinite_float(age.reported_years, "invalid linked-snapshot age")
    lower = _as_finite_or_nonfinite_float(age.lower_years, "invalid linked-snapshot age")
    upper = _as_finite_or_nonfinite_float(age.upper_years, "invalid linked-snapshot age")

    # These are the only MIMIC age states admitted by decode_age.  Missing
    # ages retain the sentinel reference; censored ages never acquire an
    # invented scalar estimate.
    if age.kind == "missing_or_invalid":
        if age.reference != "unknown" or not (math.isnan(reported) and math.isnan(lower) and math.isnan(upper)):
            _raise("invalid linked-snapshot age")
    elif age.kind == "year_derived":
        if age.reference != "hospital_admission":
            _raise("invalid linked-snapshot age")
        if not (math.isfinite(reported) and math.isfinite(lower) and math.isfinite(upper)):
            _raise("invalid linked-snapshot age")
        if reported < 0 or not reported.is_integer() or lower < 0 or upper < 0:
            _raise("invalid linked-snapshot age")
        if lower != max(0.0, reported - 1.0) or upper != reported + 1.0:
            _raise("invalid linked-snapshot age")
    elif age.kind == "topcoded":
        if age.reference != "hospital_admission":
            _raise("invalid linked-snapshot age")
        if not math.isnan(reported) or not math.isfinite(lower) or lower < 0 or upper != math.inf:
            _raise("invalid linked-snapshot age")
    else:
        _raise("invalid linked-snapshot age")
    return age


def _array_1d(value: object, *, name: str, dtype_kinds: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 1 or len(value) != _FIELD_COUNT:
        _raise("invalid linked-snapshot arrays")
    if value.dtype.kind not in dtype_kinds:
        _raise("invalid linked-snapshot arrays")
    return np.array(value, copy=True)


def _normalize_snapshot(record: object) -> LinkedSnapshot:
    # Keep the input contract closed: a subclass carrying an outcome or other
    # unreviewed field is not admitted as a snapshot record.
    if type(record) is not LinkedSnapshot:
        _raise("invalid linked-snapshot record")
    if set(vars(record)) != {
        "person",
        "episode",
        "values",
        "observed",
        "conflicts",
        "anchor_minutes",
        "field_minutes",
        "available_minutes",
        "age",
    }:
        _raise("invalid linked-snapshot record")
    person = _validate_numeric_id(record.person)
    episode = _validate_numeric_id(record.episode)

    values = _array_1d(record.values, name="values", dtype_kinds="iuf").astype(np.float64, copy=False)
    observed_input = _array_1d(record.observed, name="observed", dtype_kinds="b")
    conflicts_input = _array_1d(record.conflicts, name="conflicts", dtype_kinds="b")
    if observed_input.dtype != np.dtype(bool) or conflicts_input.dtype != np.dtype(bool):
        _raise("invalid linked-snapshot arrays")
    observed = observed_input.astype(bool, copy=False)
    conflicts = conflicts_input.astype(bool, copy=False)
    if np.any(observed & conflicts):
        _raise("invalid linked-snapshot conflicts")
    if int(observed[:_CBC_COUNT].sum()) < 2:
        _raise("linked snapshot requires two CBC fields")

    field_minutes = _array_1d(record.field_minutes, name="field_minutes", dtype_kinds="iuf").astype(np.float64, copy=False)
    available_minutes = _array_1d(record.available_minutes, name="available_minutes", dtype_kinds="iuf").astype(np.float64, copy=False)

    anchor = _as_finite_or_nonfinite_float(record.anchor_minutes, "invalid linked-snapshot timing")
    if not math.isfinite(anchor) or anchor < 0.0 or anchor > LANDMARK_MINUTES:
        _raise("invalid linked-snapshot timing")

    # Only observed values/times are admitted.  Every absent value/time is
    # canonicalized to one NaN placeholder before packing and binding.
    if np.any(observed & (~np.isfinite(values) | (values < 0.0))):
        _raise("invalid linked-snapshot observed value")
    values[~observed] = np.nan
    field_minutes[~observed] = np.nan
    available_minutes[~observed] = np.nan

    if np.any(observed & ~np.isfinite(field_minutes)) or np.any(observed & ~np.isfinite(available_minutes)):
        _raise("invalid linked-snapshot timing")
    cutoff = min(anchor + _SNAPSHOT_WINDOW_MINUTES, LANDMARK_MINUTES)
    if np.any(observed & (field_minutes < anchor)) or np.any(observed & (field_minutes > cutoff)):
        _raise("invalid linked-snapshot timing")
    if np.any(observed & (available_minutes < field_minutes)) or np.any(observed & (available_minutes > LANDMARK_MINUTES)):
        _raise("invalid linked-snapshot timing")

    observed_cbc_times = field_minutes[:_CBC_COUNT][observed[:_CBC_COUNT]]
    if len(observed_cbc_times) < 2:
        _raise("invalid linked-snapshot anchor")
    # A conflicting earliest CBC tie can be masked by snapshot selection, so
    # its charttime is intentionally absent.  A CBC conflict flag is the only
    # permitted evidence for an anchor earlier than the surviving selected
    # CBC fields; without it, the anchor must equal the surviving minimum.
    if not np.any(conflicts[:_CBC_COUNT]) and float(np.min(observed_cbc_times)) != anchor:
        _raise("invalid linked-snapshot anchor")

    age = _validate_age(record.age)
    return LinkedSnapshot(
        person,
        episode,
        values,
        observed.astype(bool, copy=True),
        conflicts.astype(bool, copy=True),
        anchor,
        field_minutes,
        available_minutes,
        age,
    )


def _readonly_copy(value: object, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _pack_text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack(">I", len(encoded)) + encoded


def _pack_float(value: object) -> bytes:
    number = float(value)
    if math.isnan(number):
        return b"N"
    if number == math.inf:
        return b"P"
    if number == -math.inf:
        return b"M"
    return b"F" + struct.pack(">d", number)


def _row_binding(record: LinkedSnapshot, source_sha256: Mapping[str, str], salt: bytes) -> str:
    payload = bytearray(_ROW_BINDING_DOMAIN)
    for key in _SOURCE_HASH_KEYS:
        payload.extend(_pack_text(key))
        payload.extend(_pack_text(source_sha256[key]))
    payload.extend(_pack_text(record.person))
    payload.extend(_pack_text(record.episode))
    payload.extend(_pack_float(LANDMARK_MINUTES))
    payload.extend(_pack_float(record.anchor_minutes))
    for field, observed, conflict, field_minute, available in zip(
        record.values,
        record.observed,
        record.conflicts,
        record.field_minutes,
        record.available_minutes,
    ):
        payload.extend(_pack_float(field))
        payload.extend(b"1" if bool(observed) else b"0")
        payload.extend(b"1" if bool(conflict) else b"0")
        payload.extend(_pack_float(field_minute))
        payload.extend(_pack_float(available))
    payload.extend(_pack_text(record.age.kind))
    payload.extend(_pack_text(record.age.reference))
    payload.extend(_pack_float(record.age.reported_years))
    payload.extend(_pack_float(record.age.lower_years))
    payload.extend(_pack_float(record.age.upper_years))
    return hmac.new(salt, bytes(payload), hashlib.sha256).hexdigest()


def _materialize_records(records: object) -> list[object]:
    if isinstance(records, (str, bytes, bytearray, Mapping)):
        _raise("invalid linked-snapshot records")
    try:
        materialized = list(records)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        _raise("invalid linked-snapshot records")
    if not materialized:
        _raise("no usable linked snapshots")
    return materialized


def _canonical_records(records: object) -> list[LinkedSnapshot]:
    materialized = _materialize_records(records)
    normalized: list[LinkedSnapshot] = []
    episodes: set[str] = set()
    for record in materialized:
        item = _normalize_snapshot(record)
        if item.episode in episodes:
            _raise("duplicate linked-snapshot episode")
        episodes.add(item.episode)
        normalized.append(item)
    normalized.sort(key=lambda item: (item.episode, item.person))
    return normalized


def _pack_arrays(records: list[LinkedSnapshot], salt: bytes) -> tuple[dict[str, np.ndarray], dict]:
    old_records = [
        JointRecord(
            item.person,
            np.array(item.values, dtype=np.float64, copy=True),
            np.array(item.observed, dtype=bool, copy=True),
            np.zeros(_FIELD_COUNT, dtype=np.uint8),
            item.age,
            "",
        )
        for item in records
    ]
    arrays, summary = pack_records("mimic", old_records, salt)
    arrays = {key: _readonly_copy(value) for key, value in arrays.items()}
    arrays.update(
        {
            "anchor_minutes": _readonly_copy([item.anchor_minutes for item in records], dtype=np.float64),
            "cutoff_minutes": _readonly_copy(np.full(len(records), LANDMARK_MINUTES), dtype=np.float64),
            "field_minutes": _readonly_copy(np.stack([item.field_minutes for item in records]), dtype=np.float64),
            "available_minutes": _readonly_copy(np.stack([item.available_minutes for item in records]), dtype=np.float64),
            "conflicts": _readonly_copy(np.stack([item.conflicts for item in records]), dtype=bool),
        }
    )
    return arrays, summary


def pack_linked_snapshots(
    records: Iterable[LinkedSnapshot],
    salt: bytes,
    source_sha256: Mapping[str, str],
) -> tuple[dict[str, np.ndarray], Mapping[str, object], dict]:
    """Pack canonical MIMIC snapshots and return private arrays/map plus counts."""

    _validate_salt(salt)
    source_hashes = _validate_source_sha256(source_sha256)
    canonical = _canonical_records(records)
    arrays, summary = _pack_arrays(canonical, salt)
    bindings = [_row_binding(item, source_hashes, salt) for item in canonical]
    arrays["row_binding"] = _readonly_copy(bindings, dtype="U64")
    rows = [
        {"person": item.person, "episode": item.episode, "row_binding": binding}
        for item, binding in zip(canonical, bindings)
    ]
    private_map: Mapping[str, object] = _PrivateMap(
        {
            "schema": _MAP_SCHEMA,
            "source_sha256": dict(source_hashes),
            "rows": rows,
        }
    )
    return arrays, private_map, summary


def _require_array(value: object, *, dtype: np.dtype | str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        _raise("invalid linked-snapshot packed arrays")
    if shape is not None and value.shape != shape:
        _raise("invalid linked-snapshot packed arrays")
    return value


def _validate_packed_array_schema(arrays: object) -> tuple[dict[str, np.ndarray], int]:
    if not isinstance(arrays, Mapping):
        _raise("invalid linked-snapshot packed arrays")
    try:
        if set(arrays) != set(_ARRAY_KEYS):
            _raise("invalid linked-snapshot packed arrays")
    except (TypeError, ValueError):
        _raise("invalid linked-snapshot packed arrays")
    values = arrays["values"]
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype(np.float64) or values.ndim != 2 or values.shape[1] != _FIELD_COUNT:
        _raise("invalid linked-snapshot packed arrays")
    n = values.shape[0]
    if n < 1:
        _raise("invalid linked-snapshot packed arrays")
    shapes = {
        "observed": (n, _FIELD_COUNT),
        "provenance": (n, _FIELD_COUNT),
        "calibration_code": (n, _FIELD_COUNT),
        "age_triplet": (n, 3),
        "age_kind": (n,),
        "adult_qualified": (n,),
        "person_group": (n,),
        "split": (n,),
        "person_weight": (n,),
        "cycle_group": (n,),
        "anchor_minutes": (n,),
        "cutoff_minutes": (n,),
        "field_minutes": (n, _FIELD_COUNT),
        "available_minutes": (n, _FIELD_COUNT),
        "conflicts": (n, _FIELD_COUNT),
        "row_binding": (n,),
    }
    dtypes = {
        "observed": np.bool_,
        "provenance": np.uint8,
        "calibration_code": np.uint8,
        "age_triplet": np.float64,
        "age_kind": np.uint8,
        "adult_qualified": np.bool_,
        "person_group": np.int64,
        "split": np.uint8,
        "person_weight": np.float64,
        "cycle_group": np.int8,
        "anchor_minutes": np.float64,
        "cutoff_minutes": np.float64,
        "field_minutes": np.float64,
        "available_minutes": np.float64,
        "conflicts": np.bool_,
        "row_binding": np.dtype("U64"),
    }
    for key in _BASE_ARRAY_KEYS[1:] + _PRIVATE_ARRAY_KEYS:
        _require_array(arrays[key], dtype=dtypes[key], shape=shapes[key])
    return {key: arrays[key] for key in _ARRAY_KEYS}, n


def _age_from_packed(arrays: Mapping[str, np.ndarray], index: int) -> AgeObservation:
    kind_index = int(arrays["age_kind"][index])
    if kind_index < 0 or kind_index >= len(AGE_KINDS):
        _raise("invalid linked-snapshot packed age")
    kind = AGE_KINDS[kind_index]
    reference = "unknown" if kind == "missing_or_invalid" else "hospital_admission"
    triplet = arrays["age_triplet"][index]
    age = AgeObservation(float(triplet[0]), float(triplet[1]), float(triplet[2]), kind, reference)
    return _validate_age(age)


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype.kind == "f" or right.dtype.kind == "f":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _validate_private_map(private_map: object, source_hashes: Mapping[str, str], n: int) -> list[dict[str, str]]:
    if not isinstance(private_map, Mapping):
        _raise("invalid linked-snapshot private map")
    try:
        if set(private_map) != {"schema", "source_sha256", "rows"}:
            _raise("invalid linked-snapshot private map")
    except (TypeError, ValueError):
        _raise("invalid linked-snapshot private map")
    if private_map["schema"] != _MAP_SCHEMA:
        _raise("invalid linked-snapshot private map")
    mapped_hashes = _validate_source_sha256(private_map["source_sha256"])
    if mapped_hashes != dict(source_hashes):
        _raise("invalid linked-snapshot source hashes")
    rows = private_map["rows"]
    if not isinstance(rows, list) or len(rows) != n:
        _raise("invalid linked-snapshot private map")
    result: list[dict[str, str]] = []
    episodes: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            _raise("invalid linked-snapshot private map")
        try:
            if set(row) != {"person", "episode", "row_binding"}:
                _raise("invalid linked-snapshot private map")
        except (TypeError, ValueError):
            _raise("invalid linked-snapshot private map")
        person = _validate_numeric_id(row["person"])
        episode = _validate_numeric_id(row["episode"])
        binding = row["row_binding"]
        if (
            not isinstance(binding, str)
            or len(binding) != 64
            or not binding.isascii()
            or any(character not in "0123456789abcdef" for character in binding)
        ):
            _raise("invalid linked-snapshot private map")
        if episode in episodes:
            _raise("duplicate linked-snapshot episode")
        episodes.add(episode)
        result.append({"person": person, "episode": episode, "row_binding": binding})
    if [row["episode"] for row in result] != sorted(row["episode"] for row in result):
        _raise("invalid linked-snapshot row order")
    return result


def validate_linked_snapshot_pack(
    arrays: Mapping[str, np.ndarray],
    private_map: Mapping[str, object],
    salt: bytes,
    source_sha256: Mapping[str, str],
) -> None:
    """Fail closed if a packed snapshot or its private linkage has changed."""

    _validate_salt(salt)
    source_hashes = _validate_source_sha256(source_sha256)
    packed, n = _validate_packed_array_schema(arrays)
    rows = _validate_private_map(private_map, source_hashes, n)

    snapshots: list[LinkedSnapshot] = []
    for index, row in enumerate(rows):
        snapshot = LinkedSnapshot(
            row["person"],
            row["episode"],
            np.array(packed["values"][index], copy=True),
            np.array(packed["observed"][index], copy=True),
            np.array(packed["conflicts"][index], copy=True),
            float(packed["anchor_minutes"][index]),
            np.array(packed["field_minutes"][index], copy=True),
            np.array(packed["available_minutes"][index], copy=True),
            _age_from_packed(packed, index),
        )
        normalized = _normalize_snapshot(snapshot)
        if (
            not _arrays_equal(normalized.values, packed["values"][index])
            or not _arrays_equal(normalized.observed, packed["observed"][index])
            or not _arrays_equal(normalized.conflicts, packed["conflicts"][index])
            or normalized.anchor_minutes != float(packed["anchor_minutes"][index])
            or not _arrays_equal(normalized.field_minutes, packed["field_minutes"][index])
            or not _arrays_equal(normalized.available_minutes, packed["available_minutes"][index])
        ):
            _raise("noncanonical linked-snapshot arrays")
        snapshots.append(normalized)

    # Repack reconstructed rows through the one canonical path.  This checks
    # split/group/age/provenance/measurement arrays as well as private timing
    # arrays and row order, with NaN values compared explicitly below.
    expected_arrays, expected_map, _ = pack_linked_snapshots(snapshots, salt, source_hashes)
    for key in _ARRAY_KEYS:
        if not _arrays_equal(packed[key], expected_arrays[key]):
            _raise("linked-snapshot pack mismatch")

    if not isinstance(expected_map, Mapping):
        _raise("linked-snapshot private map mismatch")
    if set(private_map) != set(expected_map):
        _raise("linked-snapshot private map mismatch")
    if private_map["schema"] != expected_map["schema"] or private_map["source_sha256"] != expected_map["source_sha256"]:
        _raise("linked-snapshot private map mismatch")
    expected_rows = expected_map["rows"]
    if not isinstance(expected_rows, list) or rows != expected_rows or private_map["rows"] != expected_rows:
        _raise("linked-snapshot private map mismatch")
