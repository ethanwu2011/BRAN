"""Disclosure-safe aggregate diagnostics for frozen V2 CBC calibration.

This pure module accepts only already target-erased native predictions and a
``CBCCalibrationV2`` result.  It never loads a source, invokes a model, writes
an artifact, or returns row-level arrays, bootstrap draws, intervals, or radii.
The output is descriptive only and cannot tune or select a model.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NoReturn

import numpy as np

from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_missingness_stress_metrics_v1 import safe_coverage
from bran_multisource_calibration_v2 import CBCCalibrationV2
from bran_native_cbc_calibration_metrics_v1 import _finite_bootstrap_interval, _weighted_draws


_ERROR = "multisource CBC calibration metric inputs invalid"
_N_FIELDS = 9
_ARMS = ("mlp", "token")
_PATTERNS = (
    "single_target_hidden",
    "whole_cbc_hidden",
    "red_cell_hidden",
    "single_target_no_retina",
    "whole_cbc_no_retina",
    "red_cell_no_retina",
)
_DRAWS = 1000
_MINIMUM = 20


def _fail() -> NoReturn:
    raise ValueError(_ERROR)


def _mapping(value: Any, keys: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != frozenset(keys):
        _fail()
    return value


def _matrix(value: Any, rows: int | None = None) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1] != _N_FIELDS:
        _fail()
    if rows is not None and value.shape[0] != rows:
        _fail()
    if value.dtype.kind not in "fiu" or value.dtype.kind == "b":
        _fail()
    try:
        return value.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError):
        _fail()


def _mask(value: Any, rows: int) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(bool) or value.shape != (rows, _N_FIELDS):
        _fail()
    return value


def _vector(value: Any, rows: int, *, roles: bool = False) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.ndim != 1 or value.shape != (rows,)
            or value.dtype.kind not in "iu" or value.dtype.kind == "b"):
        _fail()
    result = value.astype(np.int64, copy=False)
    if roles:
        if bool(np.any((result != 0) & (result != 1))):
            _fail()
    elif bool(np.any(result < 0)) or bool(np.any(result > 4)):
        _fail()
    return result


def _radius(value: Any) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.shape != (_N_FIELDS,)
            or value.dtype.kind not in "fiu" or value.dtype.kind == "b"):
        _fail()
    try:
        return value.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError):
        _fail()


def _validate_split(folds: np.ndarray, roles: np.ndarray) -> None:
    try:
        if set(int(item) for item in np.unique(folds)) != set(range(5)):
            _fail()
        for fold in range(5):
            total = int(np.count_nonzero(folds == fold))
            calibration = int(np.count_nonzero((folds == fold) & (roles == 0)))
            scoring = int(np.count_nonzero((folds == fold) & (roles == 1)))
            if total < 40 or calibration != total // 2 or scoring != total - total // 2:
                _fail()
    except (TypeError, ValueError):
        _fail()


def _bootstrap_counts(folds: np.ndarray, roles: np.ndarray) -> np.ndarray:
    """Deterministic role-1, within-fold resampling counts; never released."""

    rng = np.random.default_rng(94701)
    counts = np.zeros((_DRAWS, folds.shape[0]), dtype=np.int64)
    for fold in range(5):
        indices = np.flatnonzero((folds == fold) & (roles == 1))
        # Split validation guarantees nonempty scoring roles in every fold.
        for draw in range(_DRAWS):
            counts[draw, indices] = np.bincount(
                rng.choice(indices.shape[0], indices.shape[0], replace=True),
                minlength=indices.shape[0],
            )
    return counts


def _ci(values: np.ndarray, indices: np.ndarray, counts: np.ndarray) -> list[float] | None:
    """Return a CI only if the established helper has >=900 finite draws."""

    try:
        return _finite_bootstrap_interval(_weighted_draws(values, indices, counts))
    except ValueError:
        return None


def _safe_coverage(mask: np.ndarray) -> dict[str, object]:
    try:
        return safe_coverage(mask)
    except ValueError:
        _fail()


def _calibration_arrays(
    calibration: Any,
    pattern: str,
    predictions: Mapping[str, np.ndarray],
    observed: np.ndarray,
    folds: np.ndarray,
    roles: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    if not isinstance(calibration, CBCCalibrationV2):
        _fail()
    try:
        radius_by_fold = calibration.radii[pattern]
        lower_by_arm = calibration.lower[pattern]
        upper_by_arm = calibration.upper[pattern]
        if set(radius_by_fold) != set(range(5)) or set(lower_by_arm) != set(_ARMS) or set(upper_by_arm) != set(_ARMS):
            _fail()
    except (AttributeError, KeyError, TypeError):
        _fail()
    lower: dict[str, np.ndarray] = {}
    upper: dict[str, np.ndarray] = {}
    expected = np.zeros(observed.shape, dtype=bool)
    for arm in _ARMS:
        lo = _matrix(lower_by_arm[arm], observed.shape[0])
        hi = _matrix(upper_by_arm[arm], observed.shape[0])
        try:
            if bool(np.any(np.isinf(lo))) or bool(np.any(np.isinf(hi))):
                _fail()
        except TypeError:
            _fail()
        finite = np.isfinite(lo) & np.isfinite(hi)
        if not np.array_equal(np.isfinite(lo), np.isfinite(hi)) or bool(np.any(finite & (lo > hi))):
            _fail()
        arm_expected = np.zeros(observed.shape, dtype=bool)
        for fold in range(5):
            radii = _radius(radius_by_fold[fold][arm])
            if bool(np.any(np.isinf(radii))) or bool(np.any(np.isfinite(radii) & (radii < 0.0))):
                _fail()
            rows = ((folds == fold) & (roles == 1))[:, None]
            available = rows & observed & np.isfinite(radii)[None, :]
            arm_expected |= available
            # Ensure calibration has not been replaced with arbitrary bounds.
            if not np.allclose(lo[available], (predictions[arm] - radii[None, :])[available], rtol=0.0, atol=0.0):
                _fail()
            if not np.allclose(hi[available], (predictions[arm] + radii[None, :])[available], rtol=0.0, atol=0.0):
                _fail()
        if not np.array_equal(finite, arm_expected):
            _fail()
        if arm == _ARMS[0]:
            expected = arm_expected
        elif not np.array_equal(expected, arm_expected):
            _fail()
        lower[arm], upper[arm] = lo, hi
    return lower, upper, expected


def _coverage(errors: np.ndarray, lower: np.ndarray, upper: np.ndarray,
              target: np.ndarray, indices: np.ndarray, counts: np.ndarray) -> dict[str, object]:
    hits = (target >= lower) & (target <= upper)
    hit_count = int(np.count_nonzero(hits[indices]))
    miss_count = int(indices.shape[0] - hit_count)
    if 1 <= hit_count <= 19 or 1 <= miss_count <= 19:
        return {"coverage_status": "withheld_small_complement"}
    values = np.zeros(target.shape[0], dtype=np.float64)
    values[indices] = hits[indices].astype(np.float64)
    ci = _ci(values, indices, counts)
    if ci is None:
        return {"coverage_status": "unsupported"}
    return {"coverage_status": "released", "coverage": float(np.mean(hits[indices])), "coverage_ci95": ci}


def _group(
    target: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    lower: Mapping[str, np.ndarray],
    upper: Mapping[str, np.ndarray],
    field: int,
    rows: np.ndarray,
    counts: np.ndarray,
) -> dict[str, object]:
    indices = np.flatnonzero(rows)
    if indices.shape[0] < _MINIMUM:
        return {"status": "unsupported"}
    arms: dict[str, dict[str, object]] = {}
    for arm in _ARMS:
        residual = predictions[arm][indices, field] - target[indices, field]
        absolute = np.abs(residual)
        width = upper[arm][indices, field] - lower[arm][indices, field]
        if not bool(np.isfinite(residual).all()) or not bool(np.isfinite(width).all()) or bool(np.any(width < 0.0)):
            _fail()
        full_absolute = np.zeros(target.shape[0], dtype=np.float64)
        full_absolute[indices] = absolute
        ci = _ci(full_absolute, indices, counts)
        if ci is None:
            return {"status": "unsupported"}
        arms[arm] = {
            "mae": float(np.mean(absolute)),
            "bias": float(np.mean(residual)),
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "mean_interval_width": float(np.mean(width)),
            "mae_ci95": ci,
            **_coverage(
                absolute,
                lower[arm][:, field],
                upper[arm][:, field],
                target[:, field],
                indices,
                counts,
            ),
        }
    return {"status": "supported", "arms": arms}


def summarize(
    target: Any,
    intended_by_pattern: Any,
    observed_by_pattern: Any,
    predictions_by_pattern_by_arm: Any,
    calibration: Any,
    folds: Any,
    roles: Any,
) -> dict[str, dict[str, dict[str, object]]]:
    """Return fixed, aggregate-only scoring-role calibration diagnostics.

    ``intended_by_pattern`` captures genuinely observed targets selected for
    erasure.  ``observed_by_pattern`` is the post-abstention scoring mask and
    must be its subset.  No role-0 truth participates in any returned metric.
    """

    truth = _matrix(target)
    rows = truth.shape[0]
    fold_values = _vector(folds, rows)
    role_values = _vector(roles, rows, roles=True)
    _validate_split(fold_values, role_values)
    intended_map = _mapping(intended_by_pattern, _PATTERNS)
    observed_map = _mapping(observed_by_pattern, _PATTERNS)
    prediction_map = _mapping(predictions_by_pattern_by_arm, _PATTERNS)
    if not isinstance(calibration, CBCCalibrationV2):
        _fail()
    try:
        if (set(calibration.radii) != set(_PATTERNS) or set(calibration.lower) != set(_PATTERNS)
                or set(calibration.upper) != set(_PATTERNS)):
            _fail()
    except (AttributeError, TypeError):
        _fail()
    counts = _bootstrap_counts(fold_values, role_values)
    scoring_rows = (role_values == 1)[:, None]
    result: dict[str, dict[str, dict[str, object]]] = {}
    for pattern in _PATTERNS:
        intended = _mask(intended_map[pattern], rows)
        observed = _mask(observed_map[pattern], rows)
        arms_in = _mapping(prediction_map[pattern], _ARMS)
        predictions = {arm: _matrix(arms_in[arm], rows) for arm in _ARMS}
        try:
            if bool(np.any(observed & ~intended)) or bool(np.any(intended & ~np.isfinite(truth))):
                _fail()
            for arm in _ARMS:
                if bool(np.any(observed & ~np.isfinite(predictions[arm]))):
                    _fail()
        except TypeError:
            _fail()
        lower, upper, calibrated = _calibration_arrays(
            calibration, pattern, predictions, observed, fold_values, role_values
        )
        pattern_result: dict[str, dict[str, object]] = {}
        intended_scoring = intended & scoring_rows
        observed_scoring = observed & scoring_rows
        for field, field_index in zip(CBC_FIELDS, range(_N_FIELDS)):
            group_rows = calibrated[:, field_index]
            groups: dict[str, dict[str, object]] = {
                "overall": _group(truth, predictions, lower, upper, field_index, group_rows, counts)
            }
            if field == "hemoglobin":
                low = group_rows & (truth[:, field_index] < 12.0)
                rest = group_rows & (truth[:, field_index] >= 12.0)
                # Complementary suppression prevents recovering a small rest
                # stratum from the aggregate and low-stratum release.
                groups["hb_below_12_research_stratum"] = (
                    _group(truth, predictions, lower, upper, field_index, low, counts)
                    if int(np.count_nonzero(low)) >= _MINIMUM and int(np.count_nonzero(rest)) >= _MINIMUM
                    else {"status": "unsupported"}
                )
            pattern_result[field] = {
                "prediction_coverage_among_intended": _safe_coverage(observed_scoring[:, field_index][intended_scoring[:, field_index]]),
                "calibrated_coverage_among_observed": _safe_coverage(calibrated[:, field_index][observed_scoring[:, field_index]]),
                "groups": groups,
            }
        result[pattern] = pattern_result
    validate_result(result)
    return result


def _finite(value: Any, *, nonnegative: bool = False) -> bool:
    return type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(float(value)) and (not nonnegative or float(value) >= 0.0)


def _ci_valid(value: Any, *, lower: float = -math.inf) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_finite(item) for item in value) and lower <= value[0] <= value[1]


def _coverage_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    status = value.get("status")
    if status == "withheld":
        return set(value) == {"status"}
    return (status == "released" and set(value) == {"status", "supported", "total"}
            and type(value["supported"]) is int and type(value["total"]) is int
            and value["total"] >= _MINIMUM and 0 <= value["supported"] <= value["total"]
            and (value["supported"] == 0 or value["supported"] >= _MINIMUM)
            and (value["total"] - value["supported"] == 0 or value["total"] - value["supported"] >= _MINIMUM))


def _group_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if set(value) == {"status"}:
        return value["status"] == "unsupported"
    if set(value) != {"status", "arms"} or value["status"] != "supported" or set(value["arms"]) != set(_ARMS):
        return False
    for arm in _ARMS:
        item = value["arms"][arm]
        base = {"mae", "bias", "rmse", "mean_interval_width", "mae_ci95", "coverage_status"}
        if not isinstance(item, dict) or not base.issubset(item):
            return False
        if not (_finite(item["mae"], nonnegative=True) and _finite(item["bias"])
                and _finite(item["rmse"], nonnegative=True) and _finite(item["mean_interval_width"], nonnegative=True)
                and _ci_valid(item["mae_ci95"], lower=0.0)):
            return False
        if item["coverage_status"] == "released":
            if set(item) != base | {"coverage", "coverage_ci95"} or not _finite(item["coverage"], nonnegative=True) or item["coverage"] > 1.0 or not _ci_valid(item["coverage_ci95"], lower=0.0) or item["coverage_ci95"][1] > 1.0:
                return False
        elif item["coverage_status"] in ("withheld_small_complement", "unsupported"):
            if set(item) != base:
                return False
        else:
            return False
    return True


def validate_result(value: Any) -> None:
    """Fail closed unless a result has the fixed, aggregate-only schema."""

    if not isinstance(value, dict) or set(value) != set(_PATTERNS):
        _fail()
    for pattern in _PATTERNS:
        fields = value[pattern]
        if not isinstance(fields, dict) or set(fields) != set(CBC_FIELDS):
            _fail()
        for field in CBC_FIELDS:
            cell = fields[field]
            expected_groups = {"overall", "hb_below_12_research_stratum"} if field == "hemoglobin" else {"overall"}
            if not isinstance(cell, dict) or set(cell) != {"prediction_coverage_among_intended", "calibrated_coverage_among_observed", "groups"}:
                _fail()
            if not (_coverage_valid(cell["prediction_coverage_among_intended"])
                    and _coverage_valid(cell["calibrated_coverage_among_observed"])
                    and isinstance(cell["groups"], dict) and set(cell["groups"]) == expected_groups
                    and all(_group_valid(group) for group in cell["groups"].values())):
                _fail()
