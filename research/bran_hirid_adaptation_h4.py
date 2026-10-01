"""Closed HiRID H4 fixed-encoder adaptation evaluator.

This module is deliberately an in-memory numerical kernel.  It opens no source,
does not print, and returns only a small JSON-safe aggregate release object.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


ERROR = "hirid_h4_adaptation_contract_failed"
SCHEMA = "bran-hirid-h4-adaptation-evaluation-v1"
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 92344
MIN_VALID_DRAWS = 900
RIDGE_ALPHA = 100.0
COUNT_STEP = 20
METHODS = ("native", "fit_median", "native_offset", "raw_ridge", "state_ridge")
METRICS = ("mae", "signed_bias", "rmse", "interval_coverage", "interval_mean_width")
CONTRASTS = (
    "state_ridge_minus_raw_ridge_mae",
    "state_ridge_minus_native_offset_mae",
    "state_ridge_minus_native_mae",
)
_STATUSES = ("ready", "abstain_no_target", "abstain_no_physiology", "target_conflict")
_ROLE_LABELS = ((0, "fit"), (1, "calibration"), (2, "evaluation"))
_COMPLETED_STUDY_FLAGS = {
    "fixed_encoder": True,
    "readout_adaptation": True,
    "reused_source_external_study": True,
    "source_fitting": True,
    "fixed_fit_intervals": True,
    "split_conformal_calibration_performed": True,
    "nominal_interval_level": 0.9,
    "repeat_person_independence_established": False,
    "clinical_interval_calibration_established": False,
    "patient_level_output_emitted": False,
}
_UNSUPPORTED_STUDY_FLAGS = {
    **_COMPLETED_STUDY_FLAGS,
    "readout_adaptation": False,
    "source_fitting": False,
    "fixed_fit_intervals": False,
    "split_conformal_calibration_performed": False,
}


class _Invalid(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _Invalid


def _finite(value: object) -> float:
    _require(type(value) in (int, float))
    result = float(value)
    _require(math.isfinite(result))
    return result


def _privacy_flags(suppressed: bool) -> dict[str, Any]:
    return {
        "count_suppression_step": COUNT_STEP,
        "counts_and_coverage_suppressed": suppressed,
    }


def _validate_inputs(states: object, raw_values: object, raw_mask: object,
                     native: object, truth: object, statuses: object,
                     roles: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = (states, raw_values, raw_mask, native, truth, statuses, roles)
    _require(all(type(value) is np.ndarray for value in arrays))
    state = states
    raw = raw_values
    mask = raw_mask
    nat = native
    target = truth
    status = statuses
    role = roles
    _require(state.dtype.kind == "f" and state.ndim == 2 and state.shape[1:] == (192,))
    n_rows = state.shape[0]
    _require(raw.dtype.kind == "f" and raw.shape == (n_rows, 7))
    _require(mask.dtype == np.bool_ and mask.shape == (n_rows, 7))
    _require(nat.dtype.kind == "f" and nat.shape == (n_rows,))
    _require(target.dtype.kind == "f" and target.shape == (n_rows,))
    _require(status.dtype.kind == "U" and status.shape == (n_rows,))
    _require(role.dtype.kind in "iu" and role.shape == (n_rows,))
    _require(bool(np.isin(status, _STATUSES).all()))
    _require(bool(np.isin(role, (0, 1, 2)).all()))
    ready = status == "ready"
    target_present = ready | (status == "abstain_no_physiology")
    _require(bool(np.array_equal(np.any(mask, axis=1), ready)))
    _require(bool(np.isfinite(state[ready]).all()) and bool(np.isnan(state[~ready]).all()))
    _require(bool(np.isfinite(nat[ready]).all()) and bool(np.isnan(nat[~ready]).all()))
    _require(bool(np.isfinite(target[target_present]).all()) and bool((target[target_present] > 0.0).all())
             and bool(np.isnan(target[~target_present]).all()))
    # Values hidden by the observation flag are intentionally unconstrained;
    # observed values must be usable by the raw-input comparator.
    _require(bool(np.isfinite(raw[mask]).all()))
    return state, raw, mask, nat, target, status, role


def _standardize(values: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values[fit], axis=0)
    scale = np.std(values[fit], axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    standardized = (values - mean) / scale
    _require(bool(np.isfinite(standardized).all()))
    return standardized, mean, scale


def _ridge_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Fixed alpha=100 Ridge with an unpenalized intercept, solved by SVD."""
    _require(x.ndim == 2 and y.ndim == 1 and x.shape[0] == y.shape[0] and x.shape[0] > 0)
    x_mean = np.mean(x, axis=0)
    y_mean = float(np.mean(y))
    centered = x - x_mean
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    coefficient = vt.T @ ((singular / (singular * singular + RIDGE_ALPHA)) * (u.T @ (y - y_mean)))
    intercept = y_mean - float(x_mean @ coefficient)
    _require(bool(np.isfinite(coefficient).all()) and math.isfinite(intercept))
    return coefficient, intercept


def _ridge_predict(x: np.ndarray, fitted: tuple[np.ndarray, float]) -> np.ndarray:
    coefficient, intercept = fitted
    prediction = x @ coefficient + intercept
    _require(bool(np.isfinite(prediction).all()))
    return prediction


def _fit_predictions(states: np.ndarray, raw_values: np.ndarray, raw_mask: np.ndarray,
                     native: np.ndarray, truth: np.ndarray, statuses: np.ndarray,
                     roles: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Fit only on role-0 ready rows and return private, in-memory predictions.

    This helper exists for synthetic isolation tests.  It must not be serialized
    or used as a public reporting interface.
    """
    ready = statuses == "ready"
    fit = ready & (roles == 0)
    _require(int(np.count_nonzero(fit)) >= 300)
    fit_truth = truth[fit]
    median = float(np.median(fit_truth))
    offset = float(np.median(fit_truth - native[fit]))

    fill = np.zeros(7, dtype=np.float64)
    for column in range(7):
        observed_fit = fit & raw_mask[:, column]
        if np.any(observed_fit):
            fill[column] = float(np.median(raw_values[observed_fit, column]))
    raw_filled = np.where(raw_mask, raw_values, fill[None, :])
    raw_features = np.concatenate((raw_filled, raw_mask.astype(np.float64)), axis=1)
    raw_standardized, _raw_mean, _raw_scale = _standardize(raw_features, fit)
    state_standardized, _state_mean, _state_scale = _standardize(states[ready], fit[ready])
    # Keep state normalization on ready rows only: non-ready state payloads are
    # contractually NaN and must never be allowed to participate.
    state_all = np.full_like(states, np.nan, dtype=np.float64)
    state_all[ready] = state_standardized
    raw_model = _ridge_fit(raw_standardized[fit], fit_truth)
    state_model = _ridge_fit(state_all[fit], fit_truth)
    prediction = {method: np.full(native.shape, np.nan, dtype=np.float64) for method in METHODS}
    prediction["native"][ready] = native[ready]
    prediction["fit_median"][ready] = median
    prediction["native_offset"][ready] = native[ready] + offset
    prediction["raw_ridge"][ready] = _ridge_predict(raw_standardized[ready], raw_model)
    prediction["state_ridge"][ready] = _ridge_predict(state_all[ready], state_model)
    # Non-ready predictions are deliberately never inspected or released.
    return prediction, {"fit_median": median, "native_offset": offset}


def _conformal_radii(prediction: Mapping[str, np.ndarray], truth: np.ndarray,
                      ready_calibration: np.ndarray) -> dict[str, float]:
    n_cal = int(np.count_nonzero(ready_calibration))
    _require(n_cal >= 100)
    rank = int(math.ceil(0.9 * (n_cal + 1))) - 1
    _require(0 <= rank < n_cal)
    radii: dict[str, float] = {}
    for method in METHODS:
        residuals = np.sort(np.abs(prediction[method][ready_calibration] - truth[ready_calibration]))
        radius = float(residuals[rank])
        _require(math.isfinite(radius) and radius >= 0.0)
        radii[method] = radius
    return radii


def _point_values(truth: np.ndarray, prediction: Mapping[str, np.ndarray],
                  radii: Mapping[str, float]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    _require(truth.size > 0)
    methods: dict[str, dict[str, float]] = {}
    mae: dict[str, float] = {}
    for method in METHODS:
        error = prediction[method] - truth
        method_mae = float(np.mean(np.abs(error)))
        metrics = {
            "mae": method_mae,
            "signed_bias": float(np.mean(error)),
            "rmse": float(np.sqrt(np.mean(error * error))),
            "interval_coverage": float(np.mean(np.abs(error) <= radii[method])),
            "interval_mean_width": float(2.0 * radii[method]),
        }
        _require(all(math.isfinite(value) for value in metrics.values()))
        methods[method] = metrics
        mae[method] = method_mae
    contrasts = {
        "state_ridge_minus_raw_ridge_mae": mae["state_ridge"] - mae["raw_ridge"],
        "state_ridge_minus_native_offset_mae": mae["state_ridge"] - mae["native_offset"],
        "state_ridge_minus_native_mae": mae["state_ridge"] - mae["native"],
    }
    _require(all(math.isfinite(value) for value in contrasts.values()))
    return methods, contrasts


def _bootstrap_result(test_indices: np.ndarray, statuses: np.ndarray, truth: np.ndarray,
                      prediction: Mapping[str, np.ndarray], radii: Mapping[str, float],
                      *, low_hb: bool) -> tuple[dict[str, dict[str, float]], dict[str, float]] | None:
    ready_test = statuses[test_indices] == "ready"
    low = ready_test & (truth[test_indices] < 12.0)
    domain = low if low_hb else ready_test
    complement = ready_test & ~low
    if low_hb:
        allowed = int(np.count_nonzero(domain)) >= 20 and (int(np.count_nonzero(complement)) == 0 or int(np.count_nonzero(complement)) >= 20)
    else:
        allowed = int(np.count_nonzero(domain)) >= 100
    if not allowed:
        return None
    flat_keys = tuple((method, metric) for method in METHODS for metric in METRICS) + tuple(("contrast", name) for name in CONTRASTS)
    draws = np.full((BOOTSTRAP_DRAWS, len(flat_keys)), np.nan, dtype=np.float64)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    n_test = test_indices.size
    for draw_number in range(BOOTSTRAP_DRAWS):
        local = generator.integers(0, n_test, size=n_test, dtype=np.intp)
        retained = test_indices[local]
        retained_ready = statuses[retained] == "ready"
        retained_low = retained_ready & (truth[retained] < 12.0)
        retained_domain = retained_low if low_hb else retained_ready
        retained_complement = retained_ready & ~retained_low
        if low_hb:
            valid = (int(np.count_nonzero(retained_domain)) >= 20 and
                     (int(np.count_nonzero(retained_complement)) == 0 or int(np.count_nonzero(retained_complement)) >= 20))
        else:
            valid = int(np.count_nonzero(retained_domain)) >= 100
        if not valid:
            continue
        point, contrasts = _point_values(
            truth[retained[retained_domain]],
            {method: prediction[method][retained[retained_domain]] for method in METHODS}, radii,
        )
        column = 0
        for method in METHODS:
            for metric in METRICS:
                draws[draw_number, column] = point[method][metric]
                column += 1
        for contrast in CONTRASTS:
            draws[draw_number, column] = contrasts[contrast]
            column += 1
    if int(np.count_nonzero(np.isfinite(draws[:, 0]))) < MIN_VALID_DRAWS:
        return None
    method_ci: dict[str, dict[str, float]] = {method: {} for method in METHODS}
    contrast_ci: dict[str, float] = {}
    for column, (method, metric) in enumerate(flat_keys):
        values = draws[:, column]
        _require(int(np.count_nonzero(np.isfinite(values))) >= MIN_VALID_DRAWS)
        low_ci, high_ci = np.percentile(values[np.isfinite(values)], (2.5, 97.5))
        _require(math.isfinite(float(low_ci)) and math.isfinite(float(high_ci)))
        if method == "contrast":
            contrast_ci[metric] = [float(low_ci), float(high_ci)]
        else:
            method_ci[method][metric] = [float(low_ci), float(high_ci)]
    return method_ci, contrast_ci


def _released_group(test_indices: np.ndarray, statuses: np.ndarray, truth: np.ndarray,
                    prediction: Mapping[str, np.ndarray], radii: Mapping[str, float],
                    *, low_hb: bool) -> dict[str, Any]:
    bootstrap = _bootstrap_result(test_indices, statuses, truth, prediction, radii, low_hb=low_hb)
    if bootstrap is None:
        return {"status": "suppressed", "methods": None, "paired_contrasts": None}
    ready = statuses[test_indices] == "ready"
    domain = ready & (truth[test_indices] < 12.0) if low_hb else ready
    point, contrasts = _point_values(
        truth[test_indices[domain]],
        {method: prediction[method][test_indices[domain]] for method in METHODS}, radii,
    )
    method_ci, contrast_ci = bootstrap
    methods = {
        method: {metric: {"estimate": point[method][metric], "ci95": method_ci[method][metric]}
                 for metric in METRICS}
        for method in METHODS
    }
    paired = {name: {"estimate": contrasts[name], "ci95": contrast_ci[name]} for name in CONTRASTS}
    return {"status": "released", "methods": methods, "paired_contrasts": paired}


def _counts_and_coverage(statuses: np.ndarray, roles: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], bool]:
    cells = {(label, status): int(np.count_nonzero((roles == value) & (statuses == status)))
             for value, label in _ROLE_LABELS for status in _STATUSES}
    safe = all(count == 0 or count >= COUNT_STEP for count in cells.values())
    if not safe:
        return {"status": "suppressed"}, {"status": "suppressed"}, True
    table_roles: dict[str, Any] = {}
    coverage_roles: dict[str, Any] = {}
    for value, label in _ROLE_LABELS:
        role_rows = roles == value
        denominator = int(np.count_nonzero(role_rows))
        category = {status: (cells[label, status] // COUNT_STEP) * COUNT_STEP for status in _STATUSES}
        table_roles[label] = {
            "denominator_count_lower_bound": (denominator // COUNT_STEP) * COUNT_STEP,
            "statuses": category,
        }
        ready = cells[label, "ready"]
        eligible = ready + cells[label, "abstain_no_physiology"]
        coverage_roles[label] = {
            "overall": ready / denominator if denominator else None,
            "target_eligible": ready / eligible if eligible else None,
        }
    return {"status": "released", "roles": table_roles}, {"status": "released", "by_role": coverage_roles}, False


def evaluate(states: np.ndarray, raw_values: np.ndarray, raw_mask: np.ndarray, native: np.ndarray,
             truth: np.ndarray, statuses: np.ndarray, roles: np.ndarray) -> dict[str, Any]:
    """Evaluate the five prespecified H4 methods from local arrays only."""
    try:
        state, raw, mask, nat, target, status, role = _validate_inputs(
            states, raw_values, raw_mask, native, truth, statuses, roles,
        )
        ready = status == "ready"
        fit_ready = ready & (role == 0)
        cal_ready = ready & (role == 1)
        test_ready = ready & (role == 2)
        supported = (int(np.count_nonzero(fit_ready)) >= 300 and int(np.count_nonzero(cal_ready)) >= 100 and
                     int(np.count_nonzero(test_ready)) >= 100)
        counts, coverage, counts_suppressed = _counts_and_coverage(status, role)
        if not supported:
            result = {
                "schema": SCHEMA,
                "status": "unsupported",
                "reason": "minimum_ready_support_not_met",
                "study_flags": dict(_UNSUPPORTED_STUDY_FLAGS),
                "privacy_flags": _privacy_flags(counts_suppressed),
            }
            validate_result(result)
            return result
        prediction, _private_controls = _fit_predictions(state, raw, mask, nat, target, status, role)
        radii = _conformal_radii(prediction, target, cal_ready)
        test_indices = np.flatnonzero(role == 2)
        result = {
            "schema": SCHEMA,
            "status": "completed",
            "study_flags": dict(_COMPLETED_STUDY_FLAGS),
            "privacy_flags": _privacy_flags(counts_suppressed),
            "partition_status_counts": counts,
            "prediction_coverage": coverage,
            "overall": _released_group(test_indices, status, target, prediction, radii, low_hb=False),
            "low_hb": _released_group(test_indices, status, target, prediction, radii, low_hb=True),
        }
        validate_result(result)
        return result
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def _validate_json(value: Any) -> None:
    if value is None or type(value) in (bool, str, int):
        return
    if type(value) is float:
        _require(math.isfinite(value))
        return
    if type(value) is list:
        for item in value:
            _validate_json(item)
        return
    if type(value) is dict:
        _require(all(type(key) is str for key in value))
        for item in value.values():
            _validate_json(item)
        return
    raise _Invalid


def _validate_ci(value: Any, *, nonnegative: bool) -> None:
    _require(type(value) is dict and set(value) == {"estimate", "ci95"})
    estimate = _finite(value["estimate"])
    _require(type(value["ci95"]) is list and len(value["ci95"]) == 2)
    low, high = (_finite(item) for item in value["ci95"])
    _require(low <= high)
    if nonnegative:
        _require(estimate >= 0.0 and low >= 0.0)


def _validate_group(value: Any) -> None:
    _require(type(value) is dict and set(value) == {"status", "methods", "paired_contrasts"})
    if value["status"] == "suppressed":
        _require(value["methods"] is None and value["paired_contrasts"] is None)
        return
    _require(value["status"] == "released" and type(value["methods"]) is dict and type(value["paired_contrasts"]) is dict)
    _require(set(value["methods"]) == set(METHODS) and set(value["paired_contrasts"]) == set(CONTRASTS))
    for method in METHODS:
        metrics = value["methods"][method]
        _require(type(metrics) is dict and set(metrics) == set(METRICS))
        for metric in METRICS:
            _validate_ci(metrics[metric], nonnegative=metric in {"mae", "rmse", "interval_coverage", "interval_mean_width"})
        _require(value["methods"][method]["rmse"]["estimate"] + 1e-10 >= value["methods"][method]["mae"]["estimate"])
        _require(0.0 <= value["methods"][method]["interval_coverage"]["estimate"] <= 1.0)
        _require(0.0 <= value["methods"][method]["interval_coverage"]["ci95"][0] <= 1.0)
        _require(0.0 <= value["methods"][method]["interval_coverage"]["ci95"][1] <= 1.0)
    for contrast in CONTRASTS:
        _validate_ci(value["paired_contrasts"][contrast], nonnegative=False)
    mae = {method: value["methods"][method]["mae"]["estimate"] for method in METHODS}
    expected = {
        "state_ridge_minus_raw_ridge_mae": mae["state_ridge"] - mae["raw_ridge"],
        "state_ridge_minus_native_offset_mae": mae["state_ridge"] - mae["native_offset"],
        "state_ridge_minus_native_mae": mae["state_ridge"] - mae["native"],
    }
    for name, number in expected.items():
        _require(abs(value["paired_contrasts"][name]["estimate"] - number) <= 1e-10)


def _validate_counts(value: Any, coverage: Any, suppressed: bool) -> None:
    if suppressed:
        _require(value == {"status": "suppressed"} and coverage == {"status": "suppressed"})
        return
    _require(type(value) is dict and set(value) == {"status", "roles"} and value["status"] == "released")
    _require(type(coverage) is dict and set(coverage) == {"status", "by_role"} and coverage["status"] == "released")
    _require(set(value["roles"]) == {label for _number, label in _ROLE_LABELS})
    _require(set(coverage["by_role"]) == {label for _number, label in _ROLE_LABELS})
    for _number, label in _ROLE_LABELS:
        row = value["roles"][label]
        _require(type(row) is dict and set(row) == {"denominator_count_lower_bound", "statuses"})
        denominator = row["denominator_count_lower_bound"]
        _require(type(denominator) is int and denominator >= 0 and denominator % COUNT_STEP == 0)
        categories = row["statuses"]
        _require(type(categories) is dict and set(categories) == set(_STATUSES))
        _require(all(type(count) is int and count >= 0 and count % COUNT_STEP == 0 for count in categories.values()))
        _require(sum(categories.values()) <= denominator)
        _require(0 <= denominator - sum(categories.values()) <= 3 * COUNT_STEP)
        required_ready = {"fit": 300, "calibration": 100, "evaluation": 100}[label]
        _require(categories["ready"] >= required_ready)
        released = coverage["by_role"][label]
        _require(type(released) is dict and set(released) == {"overall", "target_eligible"})
        for number in released.values():
            _require(number is None or (type(number) is float and math.isfinite(number) and 0.0 <= number <= 1.0))
        ready = categories["ready"]
        no_physiology = categories["abstain_no_physiology"]
        if denominator == 0:
            _require(released["overall"] is None)
        elif ready == 0:
            _require(released["overall"] == 0.0)
        else:
            _require(released["overall"] is not None)
        if ready == 0 and no_physiology == 0:
            _require(released["target_eligible"] is None)
        elif ready == 0:
            _require(released["target_eligible"] == 0.0)
        elif no_physiology == 0:
            _require(released["target_eligible"] == 1.0)
        else:
            _require(released["target_eligible"] is not None)


def _validate_study_flags(value: Any, expected_flags: Mapping[str, Any]) -> None:
    _require(type(value) is dict and set(value) == set(expected_flags))
    for key, expected in expected_flags.items():
        actual = value[key]
        if type(expected) is bool:
            _require(type(actual) is bool and actual is expected)
        elif type(expected) is float:
            _require(type(actual) is float and actual == expected)
        else:
            _require(type(actual) is type(expected) and actual == expected)


def validate_result(result: Any) -> None:
    """Validate the closed public JSON schema, including suppression contracts."""
    try:
        _require(type(result) is dict and result.get("schema") == SCHEMA and type(result.get("status")) is str)
        _validate_json(result)
        if result["status"] == "unsupported":
            _require(set(result) == {"schema", "status", "reason", "study_flags", "privacy_flags"})
            _require(result["reason"] == "minimum_ready_support_not_met")
            _validate_study_flags(result["study_flags"], _UNSUPPORTED_STUDY_FLAGS)
            _require(type(result["privacy_flags"]) is dict and set(result["privacy_flags"]) == {"count_suppression_step", "counts_and_coverage_suppressed"})
            _require(type(result["privacy_flags"]["count_suppression_step"]) is int and result["privacy_flags"]["count_suppression_step"] == COUNT_STEP and type(result["privacy_flags"]["counts_and_coverage_suppressed"]) is bool)
            return
        _require(result["status"] == "completed")
        _require(set(result) == {"schema", "status", "study_flags", "privacy_flags", "partition_status_counts", "prediction_coverage", "overall", "low_hb"})
        _validate_study_flags(result["study_flags"], _COMPLETED_STUDY_FLAGS)
        privacy = result["privacy_flags"]
        _require(type(privacy) is dict and set(privacy) == {"count_suppression_step", "counts_and_coverage_suppressed"})
        _require(type(privacy["count_suppression_step"]) is int and privacy["count_suppression_step"] == COUNT_STEP and type(privacy["counts_and_coverage_suppressed"]) is bool)
        _validate_counts(result["partition_status_counts"], result["prediction_coverage"], privacy["counts_and_coverage_suppressed"])
        _validate_group(result["overall"])
        _validate_group(result["low_hb"])
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None
