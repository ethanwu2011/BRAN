"""Array-only uncertainty metrics for a frozen native CBC-head comparison.

Inputs are caller-owned scoring-half arrays.  This module performs no fitting,
model inference, normalization, file I/O, or patient-level release.  Returned
objects contain only aggregate field/group summaries; calibration residuals
and bootstrap draws remain private to the caller.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NoReturn

import numpy as np

from bran_clinical_semantics_v1 import CBC_FIELDS


_ERROR = "bran_native_cbc_calibration_metrics_failed"
_ARMS = ("native", "generative")
_GROUPS = ("overall", "low", "middle", "high")
_TAIL_GROUPS = ("low", "middle", "high")
_N_FIELDS = len(CBC_FIELDS)
_BOOTSTRAP_DRAWS = 1000
_MIN_FINITE_DRAWS = 900
_MIN_GROUP = 20


def _fail() -> NoReturn:
    raise ValueError(_ERROR)


def _is_real_scalar(value: Any) -> bool:
    return type(value) in (int, float) and not isinstance(value, bool)


def _finite_scalar(value: Any) -> bool:
    if not _is_real_scalar(value):
        return False
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _as_matrix(value: Any, *, rows: int | None = None) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        _fail()
    if value.ndim != 2 or value.shape[1] != _N_FIELDS:
        _fail()
    if rows is not None and value.shape[0] != rows:
        _fail()
    if value.dtype.kind not in "fiu" or value.dtype.kind == "b":
        _fail()
    try:
        matrix = value.astype(np.float64, copy=False)
        if bool(np.isinf(matrix).any()):
            _fail()
    except Exception:
        _fail()
    return matrix


def _as_observed(value: Any, rows: int | None = None) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(bool):
        _fail()
    if value.ndim != 2 or value.shape[1] != _N_FIELDS:
        _fail()
    if rows is not None and value.shape[0] != rows:
        _fail()
    return value


def _validate_observed_finite(matrix: np.ndarray, observed: np.ndarray) -> None:
    try:
        if bool(np.any(observed & ~np.isfinite(matrix))):
            _fail()
    except Exception:
        _fail()


def _validate_alpha_minimum(alpha: Any, minimum: Any) -> tuple[float, int]:
    if not _is_real_scalar(alpha):
        _fail()
    try:
        alpha_value = float(alpha)
    except Exception:
        _fail()
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        _fail()
    if type(minimum) is not int or minimum <= 0:
        _fail()
    return alpha_value, minimum


def fit_radii(
    truth_N9: Any,
    predictions_N9: Any,
    observed_bool_N9: Any,
    *,
    alpha: Any = 0.1,
    minimum: Any = _MIN_GROUP,
) -> np.ndarray:
    """Fit conservative one-based absolute-residual ranks per CBC field."""

    alpha_value, minimum_value = _validate_alpha_minimum(alpha, minimum)
    observed = _as_observed(observed_bool_N9)
    truth = _as_matrix(truth_N9, rows=observed.shape[0])
    predictions = _as_matrix(predictions_N9, rows=observed.shape[0])
    _validate_observed_finite(truth, observed)
    _validate_observed_finite(predictions, observed)

    radii = np.full(_N_FIELDS, np.nan, dtype=np.float64)
    for field in range(_N_FIELDS):
        mask = observed[:, field]
        count = int(np.count_nonzero(mask))
        if count < minimum_value:
            continue
        residuals = np.abs(predictions[mask, field] - truth[mask, field])
        if not bool(np.isfinite(residuals).all()):
            _fail()
        rank = int(math.ceil((count + 1) * (1.0 - alpha_value)))
        if rank > count:
            continue
        radii[field] = float(np.sort(residuals)[rank - 1])
    radii.setflags(write=False)
    return radii


def _validate_folds(folds: Any, rows: int) -> np.ndarray:
    if not isinstance(folds, np.ndarray):
        _fail()
    if folds.ndim != 1 or folds.shape[0] != rows or folds.dtype.kind not in "iu":
        _fail()
    try:
        fold_values = folds.astype(np.int64, copy=False)
        if bool(np.any(fold_values < 0)) or bool(np.any(fold_values > 4)):
            _fail()
    except Exception:
        _fail()
    return fold_values


def _validate_counts(counts: Any, folds: np.ndarray) -> np.ndarray:
    rows = folds.shape[0]
    if not isinstance(counts, np.ndarray):
        _fail()
    if (
        counts.ndim != 2
        or counts.shape != (_BOOTSTRAP_DRAWS, rows)
        or counts.dtype.kind not in "iu"
    ):
        _fail()
    try:
        if bool(np.any(counts < 0)):
            _fail()
        if (
            counts.dtype.kind == "u"
            and bool(np.any(counts > np.iinfo(np.int64).max))
        ):
            _fail()
        counts_i64 = counts.astype(np.int64, copy=False)
        for fold in range(5):
            fold_mask = folds == fold
            expected = int(np.count_nonzero(fold_mask))
            if not bool(np.all(counts_i64[:, fold_mask].sum(axis=1) == expected)):
                _fail()
    except Exception:
        _fail()
    return counts_i64


def _validate_groups(groups: Any, observed: np.ndarray) -> dict[str, np.ndarray]:
    if not isinstance(groups, Mapping) or frozenset(groups) != frozenset(_TAIL_GROUPS):
        _fail()
    result: dict[str, np.ndarray] = {}
    for name in _TAIL_GROUPS:
        try:
            mask = groups[name]
        except Exception:
            _fail()
        if (
            not isinstance(mask, np.ndarray)
            or mask.dtype != np.dtype(bool)
            or mask.shape != observed.shape
        ):
            _fail()
        result[name] = mask
    try:
        membership = sum(result[name].astype(np.int8) for name in _TAIL_GROUPS)
        if bool(np.any(membership > 1)) or bool(np.any(observed & (membership != 1))):
            _fail()
    except Exception:
        _fail()
    return result


def _validate_arm_matrices(
    predictions: Any,
    radii: Any,
    observed: np.ndarray,
    rows: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if (
        not isinstance(predictions, Mapping)
        or frozenset(predictions) != frozenset(_ARMS)
        or not isinstance(radii, Mapping)
        or frozenset(radii) != frozenset(_ARMS)
    ):
        _fail()
    prediction_arrays: dict[str, np.ndarray] = {}
    radius_arrays: dict[str, np.ndarray] = {}
    for arm in _ARMS:
        try:
            prediction = _as_matrix(predictions[arm], rows=rows)
            radius = _as_matrix(radii[arm], rows=rows)
        except Exception:
            _fail()
        _validate_observed_finite(prediction, observed)
        _validate_observed_finite(radius, observed)
        try:
            finite_radius = np.isfinite(radius)
            if bool(np.any(radius[finite_radius] < 0.0)):
                _fail()
            if bool(np.any(observed & ~finite_radius)):
                _fail()
        except Exception:
            _fail()
        prediction_arrays[arm] = prediction
        radius_arrays[arm] = radius
    return prediction_arrays, radius_arrays


def _finite_bootstrap_interval(draws: np.ndarray) -> list[float]:
    finite_draws = draws[np.isfinite(draws)]
    if finite_draws.shape[0] < _MIN_FINITE_DRAWS:
        _fail()
    try:
        interval = np.quantile(finite_draws, [0.025, 0.975])
        if not bool(np.isfinite(interval).all()):
            _fail()
        return [float(interval[0]), float(interval[1])]
    except Exception:
        _fail()


def _weighted_draws(
    values: np.ndarray,
    indices: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    try:
        weights = counts[:, indices].astype(np.float64, copy=False)
        denominators = weights.sum(axis=1)
        numerators = weights @ values[indices]
        return np.divide(
            numerators,
            denominators,
            out=np.full(_BOOTSTRAP_DRAWS, np.nan, dtype=np.float64),
            where=denominators > 0,
        )
    except Exception:
        _fail()


def _coverage_fields(
    errors: np.ndarray,
    radius: np.ndarray,
    indices: np.ndarray,
    counts: np.ndarray,
) -> tuple[str, dict[str, Any]]:
    coverage_by_row = errors <= radius
    hits = coverage_by_row[indices]
    hit_count = int(np.count_nonzero(hits))
    miss_count = int(indices.shape[0] - hit_count)
    if 1 <= hit_count <= 19 or 1 <= miss_count <= 19:
        return "withheld_small_complement", {"coverage_status": "withheld_small_complement"}
    coverage_draws = _weighted_draws(
        coverage_by_row.astype(np.float64),
        indices,
        counts,
    )
    return "released", {
        "coverage_status": "released",
        "coverage": float(np.mean(hits)),
        "coverage_ci95": _finite_bootstrap_interval(coverage_draws),
    }


def _summarize_group(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    radii: dict[str, np.ndarray],
    indices: np.ndarray,
    field: int,
    counts: np.ndarray,
) -> dict[str, Any]:
    if indices.shape[0] < _MIN_GROUP:
        return {"status": "unsupported"}
    arms: dict[str, dict[str, Any]] = {}
    errors_by_arm: dict[str, np.ndarray] = {}
    for arm in _ARMS:
        errors = np.abs(predictions[arm][indices, field] - truth[indices, field])
        radius = radii[arm][indices, field]
        if not bool(np.isfinite(errors).all()) or not bool(np.isfinite(radius).all()):
            _fail()
        if bool(np.any(radius < 0.0)):
            _fail()
        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        width = float(np.mean(2.0 * radius))
        error_by_row = np.abs(predictions[arm][:, field] - truth[:, field])
        mae_draws = _weighted_draws(error_by_row, indices, counts)
        coverage_status, coverage = _coverage_fields(
            error_by_row,
            radii[arm][:, field],
            indices,
            counts,
        )
        arms[arm] = {
            "mae": mae,
            "rmse": rmse,
            "mean_interval_width": width,
            "mae_ci95": _finite_bootstrap_interval(mae_draws),
            **coverage,
        }
        errors_by_arm[arm] = error_by_row

    delta_by_row = errors_by_arm["generative"] - errors_by_arm["native"]
    delta_draws = _weighted_draws(delta_by_row, indices, counts)
    contrast = {
        "generative_minus_native": {
            "mae_delta": float(arms["generative"]["mae"] - arms["native"]["mae"]),
            "ci95": _finite_bootstrap_interval(delta_draws),
        }
    }
    return {"status": "supported", "arms": arms, "contrast": contrast}


def summarize(
    truth_N9: Any,
    predictions: Any,
    radii: Any,
    observed_bool_N9: Any,
    groups: Any,
    folds_Nints: Any,
    counts_1000xN: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return closed aggregate CBC field/group metrics for scoring-half rows."""

    observed = _as_observed(observed_bool_N9)
    rows = observed.shape[0]
    truth = _as_matrix(truth_N9, rows=rows)
    _validate_observed_finite(truth, observed)
    prediction_arrays, radius_arrays = _validate_arm_matrices(
        predictions,
        radii,
        observed,
        rows,
    )
    group_arrays = _validate_groups(groups, observed)
    fold_values = _validate_folds(folds_Nints, rows)
    counts = _validate_counts(counts_1000xN, fold_values)

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for field, field_index in zip(CBC_FIELDS, range(_N_FIELDS)):
        observed_indices = np.flatnonzero(observed[:, field_index])
        tail_indices = {
            name: np.flatnonzero(
                observed[:, field_index] & group_arrays[name][:, field_index]
            )
            for name in _TAIL_GROUPS
        }
        tails_supported = all(
            indices.shape[0] >= _MIN_GROUP for indices in tail_indices.values()
        )
        field_result: dict[str, dict[str, Any]] = {}
        field_result["overall"] = _summarize_group(
            truth,
            prediction_arrays,
            radius_arrays,
            observed_indices,
            field_index,
            counts,
        )
        for name in _TAIL_GROUPS:
            field_result[name] = (
                _summarize_group(
                    truth,
                    prediction_arrays,
                    radius_arrays,
                    tail_indices[name],
                    field_index,
                    counts,
                )
                if tails_supported
                else {"status": "unsupported"}
            )
        result[field] = field_result
    return result


def _validate_ci(value: Any, *, lower: float, upper: float) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(_finite_scalar(entry) for entry in value)
        or not lower <= float(value[0]) <= float(value[1]) <= upper
    ):
        _fail()


def _validate_arm_result(arm: Any) -> None:
    base_keys = {
        "mae",
        "rmse",
        "mean_interval_width",
        "mae_ci95",
        "coverage_status",
    }
    if not isinstance(arm, dict) or not base_keys.issubset(arm):
        _fail()
    coverage_status = arm.get("coverage_status")
    if coverage_status == "released":
        if frozenset(arm) != base_keys | {"coverage", "coverage_ci95"}:
            _fail()
        if not _finite_scalar(arm["coverage"]) or not 0.0 <= float(arm["coverage"]) <= 1.0:
            _fail()
        _validate_ci(arm["coverage_ci95"], lower=0.0, upper=1.0)
    elif coverage_status == "withheld_small_complement":
        if frozenset(arm) != base_keys:
            _fail()
    else:
        _fail()
    for key in ("mae", "rmse", "mean_interval_width"):
        if not _finite_scalar(arm[key]) or float(arm[key]) < 0.0:
            _fail()
    _validate_ci(arm["mae_ci95"], lower=0.0, upper=math.inf)


def _validate_cell(cell: Any) -> None:
    if not isinstance(cell, dict):
        _fail()
    if frozenset(cell) == {"status"} and cell.get("status") == "unsupported":
        return
    if frozenset(cell) != {"status", "arms", "contrast"}:
        _fail()
    if cell.get("status") != "supported":
        _fail()
    arms = cell.get("arms")
    if not isinstance(arms, dict) or frozenset(arms) != frozenset(_ARMS):
        _fail()
    for arm in _ARMS:
        _validate_arm_result(arms[arm])
    contrast = cell.get("contrast")
    if (
        not isinstance(contrast, dict)
        or frozenset(contrast) != {"generative_minus_native"}
        or not isinstance(contrast["generative_minus_native"], dict)
        or frozenset(contrast["generative_minus_native"]) != {"mae_delta", "ci95"}
    ):
        _fail()
    delta = contrast["generative_minus_native"]
    if not _finite_scalar(delta["mae_delta"]):
        _fail()
    _validate_ci(delta["ci95"], lower=-math.inf, upper=math.inf)
    expected_delta = arms["generative"]["mae"] - arms["native"]["mae"]
    if not math.isclose(
        float(delta["mae_delta"]),
        float(expected_delta),
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        _fail()


def validate_result(output: Any) -> None:
    """Deeply validate a closed aggregate summary without exposing its inputs."""

    if not isinstance(output, dict) or frozenset(output) != frozenset(CBC_FIELDS):
        _fail()
    for field in CBC_FIELDS:
        field_result = output[field]
        if (
            not isinstance(field_result, dict)
            or frozenset(field_result) != frozenset(_GROUPS)
        ):
            _fail()
        for group in _GROUPS:
            _validate_cell(field_result[group])
