"""Private, source-free MIMIC selected-state bridge for one fold-0 frame.

The caller supplies the already authenticated packed arrays and private row map
from :mod:`bran_mimic_linked_snapshot_contract_v1`, plus an already loaded
frozen anchored model and its fold transform.  This module validates that pack
before projecting the 21-field original-unit frame through the existing V5
state bridge.  It does not read source files, select disease/outcome cohorts,
apply splits, load checkpoints, or emit public counts/results.

The linked-snapshot pack itself requires at least two observed CBC fields per
row.  This bridge therefore preserves that existing source restriction; it is
not a broader cohort admission or a source qualification step.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

import bran_joint_lab_cache_v1 as _cache
import bran_mimic_linked_snapshot_contract_v1 as _pack
import bran_nhanes_v5_state as _state
from bran_multisource_age_v2 import INTERVAL, RIGHT_CENSORED, UNKNOWN


ERROR = "bran_mimic_selected_state_v3_contract_failed"
_PACKED_AGE_KINDS = tuple(_cache.AGE_KINDS)
_EXPECTED_PACKED_AGE_KINDS = {"missing_or_invalid", "year_derived", "topcoded"}


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError(ERROR)


def _readonly(value: object, *, dtype: np.dtype | str | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, repr=False)
class PrivateMimicSelectedStateV3:
    """Caller-local state frame; every array is copied and read-only.

    ``row_binding`` and ``person_group`` are local joining keys, not public
    identifiers.  ``state`` and ``available`` are the clinical route from the
    selected fold-0 anchored frame.  ``clinical``/``clinical_mask`` are the
    canonical 59-slot original-unit frame consumed by that route.  No age,
    disease, outcome, split, or source identifiers are returned.
    """

    row_binding: np.ndarray
    person_group: np.ndarray
    state: np.ndarray
    available: np.ndarray
    clinical: np.ndarray
    clinical_mask: np.ndarray

    def __repr__(self) -> str:
        return "<PrivateMimicSelectedStateV3>"

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)


def _typed_age_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Translate packed MIMIC age states to the typed AgeBatch contract.

    A MIMIC ``year_derived`` value is retained as an interval, a topcoded age
    retains only its finite lower bound as right-censored, and missing age is
    unknown.  No scalar age is fabricated for either censored or missing rows.
    """

    triplet = arrays["age_triplet"]
    packed_kind = arrays["age_kind"]
    rows = triplet.shape[0]
    value = np.full(rows, np.nan, dtype=np.float64)
    lower = np.full(rows, np.nan, dtype=np.float64)
    upper = np.full(rows, np.nan, dtype=np.float64)
    kind = np.full(rows, UNKNOWN, dtype=np.int64)

    for packed_index, kind_name in enumerate(_PACKED_AGE_KINDS):
        rows_for_kind = packed_kind == packed_index
        if not np.any(rows_for_kind):
            continue
        _require(kind_name in _EXPECTED_PACKED_AGE_KINDS)
        if kind_name == "year_derived":
            lower[rows_for_kind] = triplet[rows_for_kind, 1]
            upper[rows_for_kind] = triplet[rows_for_kind, 2]
            kind[rows_for_kind] = INTERVAL
        elif kind_name == "topcoded":
            lower[rows_for_kind] = triplet[rows_for_kind, 1]
            kind[rows_for_kind] = RIGHT_CENSORED
        # missing_or_invalid remains all-NaN and UNKNOWN.

    return {
        "age_value": value,
        "age_lower": lower,
        "age_upper": upper,
        "age_kind": kind,
    }


def _inference_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    age = _typed_age_arrays(arrays)
    # Only the authenticated physiological values/masks and typed age reach
    # the state bridge.  Split, group, cycle, adult, timing, and provenance
    # metadata are deliberately not model inputs.
    return {
        "values": np.array(arrays["values"], dtype=np.float64, copy=True),
        "observed": np.array(arrays["observed"], dtype=bool, copy=True),
        **age,
    }


def infer_selected_state(
    arrays: Mapping[str, np.ndarray],
    private_map: Mapping[str, object],
    salt: bytes,
    source_sha256: Mapping[str, str],
    model: object,
    transform: object,
    expected_transform_sha256: str,
    *,
    selected_rows: np.ndarray | None = None,
) -> PrivateMimicSelectedStateV3:
    """Infer a caller-selected fold-0 clinical state from an authenticated pack.

    Source-pack validation is deliberately the first operation.  The delegated
    ``bran_nhanes_v5_state.infer`` then enforces the anchored MLP, exact CBC
    indices, fold ``0``, frozen/eval model, eligible registry, and transform
    hash contract without using a legacy scalar-age abstention rule.
    """

    try:
        _pack.validate_linked_snapshot_pack(arrays, private_map, salt, source_sha256)
        _require(isinstance(arrays, Mapping))
        n = arrays["values"].shape[0]
        if selected_rows is None:
            rows = np.arange(n, dtype=np.int64)
        else:
            _require(type(selected_rows) is np.ndarray and selected_rows.dtype == np.dtype(np.int64)
                     and selected_rows.ndim == 1 and len(selected_rows) > 0)
            _require(np.all((selected_rows >= 0) & (selected_rows < n))
                     and len(np.unique(selected_rows)) == len(selected_rows))
            rows = selected_rows
        inference = _inference_arrays(arrays)
        # Validate the complete authenticated pack before subsetting. The caller
        # can encode the fixed index pool without spending inference on repeat
        # admissions. Selection metadata is never an encoder input.
        inference = {key: value[rows] for key, value in inference.items()}
        n = len(rows)
        result = _state.infer(inference, model, transform, expected_transform_sha256)

        _require(type(result.state) is np.ndarray and result.state.shape == (n, 192))
        _require(type(result.available) is np.ndarray and result.available.shape == (n,))
        _require(result.available.dtype == np.dtype(bool) and np.isfinite(result.state).all())
        _require(type(result.clinical) is np.ndarray and result.clinical.shape == (n, 59))
        _require(type(result.clinical_mask) is np.ndarray and result.clinical_mask.shape == (n, 59))
        _require(result.clinical.dtype == np.dtype(np.float64))
        _require(result.clinical_mask.dtype == np.dtype(bool))
        _require(np.isfinite(result.clinical).all())

        row_binding = arrays["row_binding"][rows]
        person_group = arrays["person_group"][rows]
        _require(row_binding.dtype == np.dtype("U64") and row_binding.shape == (n,))
        _require(person_group.dtype == np.dtype(np.int64) and person_group.shape == (n,))
        return PrivateMimicSelectedStateV3(
            _readonly(row_binding),
            _readonly(person_group),
            _readonly(result.state),
            _readonly(result.available),
            _readonly(result.clinical),
            _readonly(result.clinical_mask),
        )
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


# Keep the conventional inference name available without creating another
# implementation or a second model path.
infer = infer_selected_state


__all__ = ["ERROR", "PrivateMimicSelectedStateV3", "infer", "infer_selected_state"]
