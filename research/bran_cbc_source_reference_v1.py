"""Assemble source-authenticated CBC reference ranges without I/O.

The caller supplies already selected source rows and remains responsible for
authenticating the source, same-unit range contract, and person/episode/index
identity.  This module only checks that the selected finite measurements replay
the supplied raw target array in float32 and returns private, read-only arrays.
Unknown or unusable reference ranges abstain as ``reference_valid=False``;
neither bounds nor measurements are imputed or clipped.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from numbers import Real
from typing import Any

import numpy as np

from bran_clinical_semantics_v1 import CBC_FIELDS, canonicalize_cbc


ROW_KEYS = (
    "person_id",
    "field",
    "value_as_number",
    "unit_source_value",
    "range_low",
    "range_high",
)
_ROW_KEY_SET = frozenset(ROW_KEYS)
_ERROR = "cbc_source_reference_contract_failed"


def _fail() -> None:
    raise ValueError(_ERROR)


def _finite_float(value: Any) -> float | None:
    """Return an already numeric finite value, or None for missing/nonfinite."""

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        _fail()
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        _fail()
    return number if math.isfinite(number) else None


def _float32(value: Any) -> np.float32:
    try:
        number = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        _fail()
    if number.ndim != 0:
        _fail()
    result = np.float32(number.item())
    if not bool(np.isfinite(result)):
        _fail()
    return result


def _validate_ids(patient_ids: Any) -> tuple[str, ...]:
    if isinstance(patient_ids, (str, bytes, Mapping)):
        _fail()
    try:
        ids = tuple(patient_ids)
    except TypeError:
        _fail()
    if not ids or any(
        not isinstance(person_id, str)
        or not person_id
        or not person_id.strip()
        for person_id in ids
    ):
        _fail()
    if len(set(ids)) != len(ids):
        _fail()
    return ids


def _validate_arrays(
    patient_ids: tuple[str, ...],
    target: Any,
    observed: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(target, np.ndarray) or target.ndim != 2 or target.shape[1] != len(CBC_FIELDS):
        _fail()
    if target.dtype.kind not in "fiu" or target.dtype.kind == "b":
        _fail()
    if not isinstance(observed, np.ndarray) or observed.dtype != np.dtype(bool):
        _fail()
    if observed.shape != target.shape or target.shape[0] != len(patient_ids):
        _fail()
    target_float = np.asarray(target, dtype=np.float64)
    if np.any(np.isinf(target_float)):
        _fail()
    if not np.isfinite(target_float[observed]).all():
        _fail()
    return target_float, observed.copy()


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    value = row[key]
    # A mapping may expose a scalar array or another container as a value; the
    # numeric validators below reject it without serializing its contents.
    return value


def _assemble(
    patient_ids: Any,
    target_N9: Any,
    observed_N9: Any,
    rows: Iterable[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = _validate_ids(patient_ids)
    target, observed = _validate_arrays(ids, target_N9, observed_N9)
    person_index = {person_id: index for index, person_id in enumerate(ids)}
    field_index = {field: index for index, field in enumerate(CBC_FIELDS)}
    lower = np.full(target.shape, np.nan, dtype=np.float64)
    upper = np.full(target.shape, np.nan, dtype=np.float64)
    reference_valid = np.zeros(target.shape, dtype=np.bool_)
    seen = np.zeros(target.shape, dtype=np.bool_)

    try:
        iterator = iter(rows)
    except TypeError:
        _fail()
    for row in iterator:
        if not isinstance(row, Mapping) or frozenset(row) != _ROW_KEY_SET:
            _fail()
        person_id = _row_value(row, "person_id")
        field = _row_value(row, "field")
        if person_id not in person_index or field not in field_index:
            _fail()
        i, j = person_index[person_id], field_index[field]

        raw_value = _finite_float(_row_value(row, "value_as_number"))
        # Missing/nonfinite source targets are ignored by the source loader;
        # the final finite-support check still requires every observed target
        # cell to have exactly one finite source row.
        if raw_value is None:
            continue
        if seen[i, j] or not observed[i, j]:
            _fail()
        seen[i, j] = True

        raw32 = _float32(raw_value)
        if float(raw32) != target[i, j]:
            _fail()

        unit = _row_value(row, "unit_source_value")
        canonical = canonicalize_cbc(field, raw_value, unit, provenance=1)
        if canonical.observed and float(_float32(canonical.value)) != target[i, j]:
            # A recognized conversion that changes the model's existing raw
            # scale is a source/target contradiction, not a reason to rewrite it.
            _fail()
        if not canonical.observed:
            continue

        low = _finite_float(_row_value(row, "range_low"))
        high = _finite_float(_row_value(row, "range_high"))
        if low is None or high is None or low < 0.0 or high < 0.0 or not low < high:
            continue
        lower_obs = canonicalize_cbc(field, low, unit, provenance=1)
        upper_obs = canonicalize_cbc(field, high, unit, provenance=1)
        if not lower_obs.observed or not upper_obs.observed:
            continue
        canonical_low = float(lower_obs.value)
        canonical_high = float(upper_obs.value)
        if (
            not math.isfinite(canonical_low)
            or not math.isfinite(canonical_high)
            or canonical_low < 0.0
            or not canonical_low < canonical_high
        ):
            continue
        lower[i, j] = canonical_low
        upper[i, j] = canonical_high
        reference_valid[i, j] = True

    if not np.array_equal(seen, observed):
        _fail()
    lower.setflags(write=False)
    upper.setflags(write=False)
    reference_valid.setflags(write=False)
    return lower, upper, reference_valid


def assemble(
    patient_ids: Any,
    target_N9: Any,
    observed_N9: Any,
    rows: Iterable[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return read-only canonical lower/upper bounds and validity masks.

    All failures use one generic contract error so no supplied identifiers or
    row contents can enter an exception string.  The function performs no
    source authentication, I/O, normalization fitting, or clinical labeling.
    """

    try:
        return _assemble(patient_ids, target_N9, observed_N9, rows)
    except ValueError as error:
        if str(error) == _ERROR:
            raise
        raise ValueError(_ERROR) from None
    except Exception:
        raise ValueError(_ERROR) from None


__all__ = ["ROW_KEYS", "assemble"]
