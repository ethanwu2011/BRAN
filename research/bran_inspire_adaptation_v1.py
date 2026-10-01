"""Pure fixed three-arm INSPIRE readout adaptation.

This module consumes already-selected, already-authenticated arrays.  It does
not read source files, choose a cohort, load a representation, or emit a
patient-level result.  All preprocessing statistics, categorical vocabulary,
logistic fits, and test probabilities remain in the caller-owned private sink;
the returned value is a closed aggregate report.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from bran_mimic_clinical_outcomes_v3 import (
    _auc_plan,
    _calibrated_probabilities,
    _calibration_offset,
    _losses,
    _weighted_auc,
)


ERROR = "bran_inspire_adaptation_v1_contract_failed"
SCHEMA = "bran-inspire-adaptation-v1"
PRIVATE_SCHEMA = "bran-inspire-adaptation-v1-private"
LAB_WIDTH = 18
STATE_WIDTH = 192
CONTEXT_INPUT_WIDTH = 4
AGE_INTERVAL = 1
AGE_UNKNOWN = 3
ROLE_NAMES = ("fit", "calibration", "test")
ARM_NAMES = ("context", "raw_context", "state_context")
CONTRAST_NAMES = (
    "state_context_minus_raw_context",
    "state_context_minus_context",
)

# This is intentionally an ordinary dictionary: synthetic tests may replace a
# draw count without changing the production default.  No real-run knob is
# exposed by the source runner.
POLICY = {
    "lab_width": LAB_WIDTH,
    "state_width": STATE_WIDTH,
    "context_input_width": CONTEXT_INPUT_WIDTH,
    "logistic_c": 1.0,
    "logistic_solver": "lbfgs",
    "logistic_max_iter": 2000,
    "class_balancing": False,
    "calibration": "calibration_intercept_root_on_clipped_fit_logits",
    # These values are pinned to the audited MIMIC calibration helpers used
    # below (_calibration_offset/_calibrated_probabilities/_losses).
    "logit_clip": 30.0,
    "probability_clip": 1e-7,
    "minimum_fit_rows": 200,
    "minimum_calibration_rows": 100,
    "minimum_test_rows": 100,
    "minimum_class_support_per_role": 20,
    "bootstrap_draws": 1000,
    "minimum_valid_auroc_draws": 900,
    "bootstrap_seed": 98621,
    "familywise_alpha": 0.05,
    "multiplicity_comparisons": 2,
    "superiority_decision": "primary_adjusted_auroc_lower_bound_only",
    "fixed_fit_uncertainty_only": True,
}

_ZERO_ALLOWED_COLUMNS = frozenset((1, 2, 3, 7))
_AGE_ONEHOT_CODES = (0, 1, 2, 3)
_SEX_LEVELS = ("M", "F", "unknown")
_ASA_LEVELS = tuple(str(value) for value in range(1, 7)) + ("unknown",)
_EMERGENCY_LEVELS = ("0", "1", "unknown")
_BASE_KEYS = {
    "schema",
    "status",
    "policy",
    "patient_level_output_emitted",
    "clinical_utility_established",
    "external_adaptation_performance_established",
}
_CLOSED_STATUSES = frozenset(("unsupported_outcome_support", "unsupported_bootstrap"))
_MAX_LOGLOSS = max(
    -math.log(float(POLICY["probability_clip"])),
    -math.log1p(-(1.0 - float(POLICY["probability_clip"]))),
)


def _fail() -> None:
    raise ValueError(ERROR) from None


def _require(condition: bool) -> None:
    if not condition:
        _fail()


def _policy() -> dict[str, object]:
    return dict(POLICY)


def _closed(status: str) -> dict[str, object]:
    _require(status in _CLOSED_STATUSES)
    return {
        "schema": SCHEMA,
        "status": status,
        "policy": _policy(),
        "patient_level_output_emitted": False,
        "clinical_utility_established": False,
        "external_adaptation_performance_established": False,
    }


def _readonly_copy(value: object, dtype: np.dtype | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _numeric(value: object, *, dtype: np.dtype, ndim: int, shape: tuple[int, ...] | None = None) -> np.ndarray:
    _require(type(value) is np.ndarray and value.dtype == dtype and value.ndim == ndim)
    if shape is not None:
        _require(value.shape == shape)
    return value


def _validate_inputs(
    values: object,
    observed: object,
    age_lower: object,
    age_upper: object,
    age_kind: object,
    context: object,
    state: object,
    role: object,
    outcome: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labs = _numeric(values, dtype=np.dtype(np.float64), ndim=2)
    _require(labs.shape[1] == LAB_WIDTH and labs.shape[0] > 0)
    rows = labs.shape[0]
    _require(type(observed) is np.ndarray and observed.dtype == np.dtype(bool)
             and observed.shape == labs.shape)
    mask = np.array(observed, dtype=bool, copy=True)
    _require(bool(np.isfinite(labs[mask]).all()))
    if mask.any():
        for column in range(LAB_WIDTH):
            active = mask[:, column]
            if not active.any():
                continue
            column_values = labs[active, column]
            if column in _ZERO_ALLOWED_COLUMNS:
                _require(bool(np.all(column_values >= 0.0)))
            else:
                _require(bool(np.all(column_values > 0.0)))

    lower = _numeric(age_lower, dtype=np.dtype(np.float64), ndim=1, shape=(rows,))
    upper = _numeric(age_upper, dtype=np.dtype(np.float64), ndim=1, shape=(rows,))
    _require(type(age_kind) is np.ndarray and age_kind.ndim == 1 and age_kind.shape == (rows,)
             and age_kind.dtype.kind in "iu" and age_kind.dtype.kind != "b")
    kind = np.array(age_kind, dtype=np.int64, copy=True)
    _require(bool(np.all((kind == AGE_INTERVAL) | (kind == AGE_UNKNOWN))))
    interval = kind == AGE_INTERVAL
    if interval.any():
        _require(bool(np.isfinite(lower[interval]).all() and np.isfinite(upper[interval]).all()))
        _require(bool(np.all(lower[interval] >= 0.0) and np.all(upper[interval] >= 0.0)
                     and np.all(lower[interval] <= upper[interval])))

    _require(type(context) is np.ndarray and context.dtype.kind == "U"
             and context.shape == (rows, CONTEXT_INPUT_WIDTH))
    _require(type(state) is np.ndarray and state.dtype == np.dtype(np.float64)
             and state.shape == (rows, STATE_WIDTH) and bool(np.isfinite(state).all()))
    _require(type(role) is np.ndarray and role.shape == (rows,)
             and role.dtype.kind in "iu" and role.dtype.kind != "b")
    _require(bool(np.all(np.isin(role, (0, 1, 2)))))
    role_values = np.array(role, dtype=np.int8, copy=True)
    _require(type(outcome) is np.ndarray and outcome.shape == (rows,)
             and outcome.dtype.kind in "iu" and outcome.dtype.kind != "b")
    _require(bool(np.all(np.isin(outcome, (0, 1)))))
    labels = np.array(outcome, dtype=np.int8, copy=True)
    # The source runner admits only physiology-eligible rows to this stage.
    _require(bool(np.all(mask.any(axis=1))))
    return (
        np.array(labs, dtype=np.float64, copy=True),
        mask,
        np.array(lower, dtype=np.float64, copy=True),
        np.array(upper, dtype=np.float64, copy=True),
        kind,
        np.array(context, copy=True),
        np.array(state, dtype=np.float64, copy=True),
        role_values,
        labels,
    )


def _support_ok(role: np.ndarray, labels: np.ndarray) -> bool:
    minimum_rows = (
        int(POLICY["minimum_fit_rows"]),
        int(POLICY["minimum_calibration_rows"]),
        int(POLICY["minimum_test_rows"]),
    )
    minimum_class = int(POLICY["minimum_class_support_per_role"])
    return all(
        int(np.sum(role == role_number)) >= minimum_rows[role_number]
        and all(int(np.sum((role == role_number) & (labels == label))) >= minimum_class
                 for label in (0, 1))
        for role_number in range(3)
    )


def _department_vocabulary(context: np.ndarray, fit: np.ndarray) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for value in context[fit, 3].tolist():
        category = str(value)
        if category == "other":
            continue
        counts[category] = counts.get(category, 0) + 1
    retained = [
        (category, count)
        for category, count in counts.items()
        if count >= int(POLICY["minimum_class_support_per_role"])
    ]
    retained.sort(key=lambda item: (-item[1], item[0]))
    return tuple(category for category, _count in retained[:20])


def _age_parameters(lower: np.ndarray, upper: np.ndarray, kind: np.ndarray, fit: np.ndarray) -> dict[str, object]:
    interval_fit = fit & (kind == AGE_INTERVAL)
    if interval_fit.any():
        lower_median = float(np.median(lower[interval_fit]))
        upper_median = float(np.median(upper[interval_fit]))
    else:
        lower_median = 0.0
        upper_median = 0.0
    fit_lower = np.full(int(fit.sum()), lower_median, dtype=np.float64)
    fit_upper = np.full(int(fit.sum()), upper_median, dtype=np.float64)
    fit_interval = kind[fit] == AGE_INTERVAL
    fit_lower[fit_interval] = lower[fit][fit_interval]
    fit_upper[fit_interval] = upper[fit][fit_interval]
    lower_mean = float(np.mean(fit_lower))
    upper_mean = float(np.mean(fit_upper))
    lower_scale = float(np.std(fit_lower))
    upper_scale = float(np.std(fit_upper))
    if not math.isfinite(lower_scale) or lower_scale <= 0.0:
        lower_scale = 1.0
    if not math.isfinite(upper_scale) or upper_scale <= 0.0:
        upper_scale = 1.0
    _require(all(math.isfinite(value) for value in (
        lower_median, upper_median, lower_mean, upper_mean, lower_scale, upper_scale
    )))
    return {
        "lower_median": lower_median,
        "upper_median": upper_median,
        "lower_mean": lower_mean,
        "upper_mean": upper_mean,
        "lower_scale": lower_scale,
        "upper_scale": upper_scale,
    }


def _age_design(lower: np.ndarray, upper: np.ndarray, kind: np.ndarray, parameters: Mapping[str, object]) -> np.ndarray:
    lower_fill = np.full(lower.shape, float(parameters["lower_median"]), dtype=np.float64)
    upper_fill = np.full(upper.shape, float(parameters["upper_median"]), dtype=np.float64)
    interval = kind == AGE_INTERVAL
    lower_fill[interval] = lower[interval]
    upper_fill[interval] = upper[interval]
    numeric = np.column_stack((
        (lower_fill - float(parameters["lower_mean"])) / float(parameters["lower_scale"]),
        (upper_fill - float(parameters["upper_mean"])) / float(parameters["upper_scale"]),
    ))
    states = np.column_stack([(kind == code).astype(np.float64) for code in _AGE_ONEHOT_CODES])
    result = np.column_stack((numeric, states))
    _require(result.shape == (len(kind), 6) and bool(np.isfinite(result).all()))
    return result


def _categorical_one_hot(values: np.ndarray, levels: tuple[str, ...], normalize) -> np.ndarray:
    normalized = np.asarray([normalize(str(value)) for value in values.tolist()], dtype=object)
    result = np.column_stack([(normalized == level).astype(np.float64) for level in levels])
    _require(result.shape == (len(values), len(levels)))
    return result


def _context_design(
    context: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    kind: np.ndarray,
    age_parameters: Mapping[str, object],
    departments: tuple[str, ...],
) -> np.ndarray:
    def sex(value: str) -> str:
        return value if value in ("M", "F") else "unknown"

    def asa(value: str) -> str:
        return value if value in _ASA_LEVELS[:-1] else "unknown"

    def emergency(value: str) -> str:
        return value if value in ("0", "1") else "unknown"

    def department(value: str) -> str:
        return value if value in departments else "other"

    result = np.column_stack((
        _age_design(lower, upper, kind, age_parameters),
        _categorical_one_hot(context[:, 0], _SEX_LEVELS, sex),
        _categorical_one_hot(context[:, 1], _ASA_LEVELS, asa),
        _categorical_one_hot(context[:, 2], _EMERGENCY_LEVELS, emergency),
        _categorical_one_hot(context[:, 3], departments + ("other",), department),
    ))
    _require(result.ndim == 2 and bool(np.isfinite(result).all()))
    return result


def _numeric_parameters(values: np.ndarray, observed: np.ndarray, fit: np.ndarray) -> dict[str, np.ndarray]:
    median = np.zeros(LAB_WIDTH, dtype=np.float64)
    fit_imputed = np.zeros((int(fit.sum()), LAB_WIDTH), dtype=np.float64)
    fit_rows = np.flatnonzero(fit)
    for column in range(LAB_WIDTH):
        available = observed[fit_rows, column]
        if available.any():
            median[column] = float(np.median(values[fit_rows[available], column]))
        fit_imputed[:, column] = median[column]
        fit_imputed[available, column] = values[fit_rows[available], column]
    mean = np.mean(fit_imputed, axis=0)
    scale = np.std(fit_imputed, axis=0)
    scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0
    _require(bool(np.isfinite(mean).all() and np.isfinite(scale).all()))
    return {
        "median": median,
        "mean": np.asarray(mean, dtype=np.float64),
        "scale": np.asarray(scale, dtype=np.float64),
    }


def _numeric_design(values: np.ndarray, observed: np.ndarray, parameters: Mapping[str, np.ndarray]) -> np.ndarray:
    imputed = np.empty(values.shape, dtype=np.float64)
    imputed[:] = np.asarray(parameters["median"], dtype=np.float64)[None, :]
    imputed[observed] = values[observed]
    standardized = (imputed - parameters["mean"]) / parameters["scale"]
    result = np.column_stack((standardized, observed.astype(np.float64)))
    _require(result.shape == (values.shape[0], LAB_WIDTH * 2) and bool(np.isfinite(result).all()))
    return result


def _state_parameters(state: np.ndarray, fit: np.ndarray) -> dict[str, np.ndarray]:
    fit_state = state[fit]
    mean = np.mean(fit_state, axis=0)
    scale = np.std(fit_state, axis=0)
    scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0
    _require(bool(np.isfinite(mean).all() and np.isfinite(scale).all()))
    return {
        "mean": np.asarray(mean, dtype=np.float64),
        "scale": np.asarray(scale, dtype=np.float64),
    }


def _state_design(state: np.ndarray, parameters: Mapping[str, np.ndarray]) -> np.ndarray:
    result = (state - parameters["mean"]) / parameters["scale"]
    _require(result.shape == state.shape and bool(np.isfinite(result).all()))
    return result


def _fit_arm(design: np.ndarray, labels: np.ndarray, roles: np.ndarray) -> tuple[object, float, np.ndarray]:
    fit = roles == 0
    calibration = roles == 1
    test = roles == 2
    _require(bool(fit.any() and calibration.any() and test.any()))
    model = LogisticRegression(
        C=float(POLICY["logistic_c"]),
        solver=str(POLICY["logistic_solver"]),
        max_iter=int(POLICY["logistic_max_iter"]),
        class_weight=None,
    )
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(design[fit], labels[fit])
    _require(hasattr(model, "n_iter_") and np.all(np.asarray(model.n_iter_) < int(POLICY["logistic_max_iter"])))
    calibration_logits = np.asarray(model.decision_function(design[calibration]), dtype=np.float64)
    test_logits = np.asarray(model.decision_function(design[test]), dtype=np.float64)
    _require(calibration_logits.shape == (int(calibration.sum()),)
             and test_logits.shape == (int(test.sum()),)
             and bool(np.isfinite(calibration_logits).all())
             and bool(np.isfinite(test_logits).all()))
    offset = float(_calibration_offset(calibration_logits, labels[calibration].astype(np.float64)))
    probabilities = np.asarray(_calibrated_probabilities(test_logits, offset), dtype=np.float64)
    _require(probabilities.shape == (int(test.sum()),) and bool(np.isfinite(probabilities).all()))
    return model, offset, probabilities


def _point_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    plan = _auc_plan(labels, probabilities)
    weights = np.ones(len(labels), dtype=np.int64)
    logloss, brier = _losses(labels, probabilities)
    result = {
        "auroc": float(_weighted_auc(plan, weights)),
        "logloss": float(np.mean(logloss)),
        "brier": float(np.mean(brier)),
    }
    _require(all(math.isfinite(value) for value in result.values()))
    return result


def _bootstrap(
    labels: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray] | None:
    rows = len(labels)
    plans = {name: _auc_plan(labels, probability) for name, probability in predictions.items()}
    losses = {}
    briers = {}
    for name, probability in predictions.items():
        losses[name], briers[name] = _losses(labels, probability)
    draws_count = int(POLICY["bootstrap_draws"])
    _require(draws_count > 0)
    draws = {
        name: {
            "auroc": np.empty(draws_count, dtype=np.float64),
            "logloss": np.empty(draws_count, dtype=np.float64),
            "brier": np.empty(draws_count, dtype=np.float64),
        }
        for name in predictions
    }
    valid = np.zeros(draws_count, dtype=bool)
    rng = np.random.default_rng(int(POLICY["bootstrap_seed"]))
    for draw in range(draws_count):
        sampled = rng.integers(0, rows, size=rows)
        weights = np.bincount(sampled, minlength=rows)
        sampled_labels = labels[sampled]
        if not np.any(sampled_labels == 0.0) or not np.any(sampled_labels == 1.0):
            continue
        valid[draw] = True
        for name in predictions:
            draws[name]["auroc"][draw] = float(_weighted_auc(plans[name], weights))
            draws[name]["logloss"][draw] = float(np.dot(losses[name], weights) / rows)
            draws[name]["brier"][draw] = float(np.dot(briers[name], weights) / rows)
    _require(int(valid.sum()) <= draws_count)
    if int(valid.sum()) < int(POLICY["minimum_valid_auroc_draws"]):
        return None
    return draws, valid


def _bootstrap_valid_count(labels: np.ndarray) -> int:
    """Count valid paired resamples without fitting any arm."""

    draws_count = int(POLICY["bootstrap_draws"])
    _require(draws_count > 0 and labels.ndim == 1 and len(labels) > 0)
    rng = np.random.default_rng(int(POLICY["bootstrap_seed"]))
    valid = 0
    for _draw in range(draws_count):
        sampled = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[sampled]
        if np.any(sampled_labels == 0.0) and np.any(sampled_labels == 1.0):
            valid += 1
    return valid


def _interval(values: np.ndarray, adjusted: bool = False) -> list[float]:
    tail = (float(POLICY["familywise_alpha"]) / float(POLICY["multiplicity_comparisons"]) / 2.0
            if adjusted else 0.025)
    result = np.quantile(values, (tail, 1.0 - tail))
    return [float(result[0]), float(result[1])]


def _arm_report(
    labels: np.ndarray,
    probability: np.ndarray,
    draws: Mapping[str, Mapping[str, np.ndarray]],
    valid: np.ndarray,
) -> dict[str, object]:
    point = _point_metrics(labels, probability)
    return {
        "status": "available",
        **point,
        "auroc_marginal_95ci": _interval(draws["auroc"][valid]),
        "logloss_marginal_95ci": _interval(draws["logloss"][valid]),
        "brier_marginal_95ci": _interval(draws["brier"][valid]),
    }


def _contrast_report(
    plus: str,
    minus: str,
    arm_reports: Mapping[str, dict[str, object]],
    draws: Mapping[str, Mapping[str, np.ndarray]],
    valid: np.ndarray,
    *,
    primary: bool,
) -> dict[str, object]:
    plus_report = arm_reports[plus]
    minus_report = arm_reports[minus]
    auc_difference = draws[plus]["auroc"][valid] - draws[minus]["auroc"][valid]
    logloss_improvement = draws[minus]["logloss"][valid] - draws[plus]["logloss"][valid]
    brier_improvement = draws[minus]["brier"][valid] - draws[plus]["brier"][valid]
    result = {
        "status": "available",
        "auroc_difference": float(plus_report["auroc"] - minus_report["auroc"]),
        "logloss_improvement": float(minus_report["logloss"] - plus_report["logloss"]),
        "brier_improvement": float(minus_report["brier"] - plus_report["brier"]),
        "auroc_marginal_95ci": _interval(auc_difference),
        "logloss_improvement_marginal_95ci": _interval(logloss_improvement),
        "brier_improvement_marginal_95ci": _interval(brier_improvement),
        "auroc_adjusted_95ci": _interval(auc_difference, adjusted=True),
        "logloss_improvement_adjusted_95ci": _interval(logloss_improvement, adjusted=True),
        "brier_improvement_adjusted_95ci": _interval(brier_improvement, adjusted=True),
        "superiority": bool(primary and _interval(auc_difference, adjusted=True)[0] > 0.0),
    }
    return result


def _private_sink(private_sink: object) -> dict[str, object] | None:
    if private_sink is None:
        return None
    _require(type(private_sink) is dict and not private_sink)
    private_sink["schema"] = PRIVATE_SCHEMA
    return private_sink


def _nonevent_event_lower_bounds(role: np.ndarray, labels: np.ndarray) -> list[list[int]]:
    return [
        [
            int(np.sum((role == role_number) & (labels == label))) // 20 * 20
            for label in (0, 1)
        ]
        for role_number in range(3)
    ]


def _evaluate(
    values: object,
    observed: object,
    age_lower: object,
    age_upper: object,
    age_kind: object,
    context: object,
    state: object,
    role: object,
    outcome: object,
    private_sink: object,
) -> dict[str, object]:
    values, observed, age_lower, age_upper, age_kind, context, state, role, outcome = _validate_inputs(
        values, observed, age_lower, age_upper, age_kind, context, state, role, outcome
    )
    sink = _private_sink(private_sink)
    if not _support_ok(role, outcome):
        if sink is not None:
            sink["status"] = "unsupported_outcome_support"
        return _closed("unsupported_outcome_support")

    fit = role == 0
    age_parameters = _age_parameters(age_lower, age_upper, age_kind, fit)
    departments = _department_vocabulary(context, fit)
    context_design = _context_design(context, age_lower, age_upper, age_kind, age_parameters, departments)
    raw_parameters = _numeric_parameters(values, observed, fit)
    raw_design = _numeric_design(values, observed, raw_parameters)
    state_parameters = _state_parameters(state, fit)
    state_design = _state_design(state, state_parameters)
    designs = {
        "context": context_design,
        "raw_context": np.column_stack((context_design, raw_design)),
        "state_context": np.column_stack((context_design, state_design)),
    }
    _require(all(design.shape[0] == len(role) and bool(np.isfinite(design).all())
                 for design in designs.values()))

    predictions: dict[str, np.ndarray] = {}
    fitted = {}
    for arm in ARM_NAMES:
        model, offset, probability = _fit_arm(designs[arm], outcome.astype(np.float64), role)
        predictions[arm] = probability
        fitted[arm] = (model, offset, int(designs[arm].shape[1]))

    test = role == 2
    test_labels = outcome[test].astype(np.float64)
    bootstrap = _bootstrap(test_labels, predictions)
    if bootstrap is None:
        if sink is not None:
            sink["status"] = "unsupported_bootstrap"
        return _closed("unsupported_bootstrap")
    draws, valid = bootstrap

    arm_reports = {
        arm: _arm_report(test_labels, predictions[arm], draws[arm], valid)
        for arm in ARM_NAMES
    }
    contrast_reports = {
        "state_context_minus_raw_context": _contrast_report(
            "state_context", "raw_context", arm_reports, draws, valid, primary=True
        ),
        "state_context_minus_context": _contrast_report(
            "state_context", "context", arm_reports, draws, valid, primary=False
        ),
    }
    result = {
        "schema": SCHEMA,
        "status": "ok",
        "policy": _policy(),
        "arms": arm_reports,
        "contrasts": contrast_reports,
        "primary_superiority": bool(contrast_reports["state_context_minus_raw_context"]["superiority"]),
        "support": {
            "status": "released_lower_bounds",
            "roles": list(ROLE_NAMES),
            "nonevent_event_lower_bounds": _nonevent_event_lower_bounds(role, outcome),
        },
        "bootstrap": {
            "requested_draws": int(POLICY["bootstrap_draws"]),
            "accepted_auroc_draws": int(valid.sum()),
        },
        "patient_level_output_emitted": False,
        "clinical_utility_established": False,
        "external_adaptation_performance_established": False,
    }
    if sink is not None:
        sink["status"] = "ok"
        sink["transforms"] = {
            "age": {key: float(value) for key, value in age_parameters.items()},
            "raw": {key: _readonly_copy(value, np.float64) for key, value in raw_parameters.items()},
            "state": {key: _readonly_copy(value, np.float64) for key, value in state_parameters.items()},
        }
        sink["vocabulary"] = {"departments": tuple(departments)}
        sink["arms"] = {
            arm: {
                "model": fitted[arm][0],
                "calibration_offset": float(fitted[arm][1]),
                "feature_width": fitted[arm][2],
                "test_predictions": _readonly_copy(predictions[arm], np.float64),
            }
            for arm in ARM_NAMES
        }
        sink["test_row_count"] = int(test.sum())
    _require(validate_report(result))
    return result


def evaluate(
    values: np.ndarray,
    observed: np.ndarray,
    age_lower: np.ndarray,
    age_upper: np.ndarray,
    age_kind: np.ndarray,
    context: np.ndarray,
    state: np.ndarray,
    role: np.ndarray,
    outcome: np.ndarray,
    *,
    private_sink: dict[str, object] | None = None,
) -> dict[str, object]:
    """Fit the fixed three-arm adaptation recipe and return closed aggregates."""

    try:
        return _evaluate(
            values, observed, age_lower, age_upper, age_kind,
            context, state, role, outcome, private_sink,
        )
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def _replay_designs(
    values: np.ndarray,
    observed: np.ndarray,
    age_lower: np.ndarray,
    age_upper: np.ndarray,
    age_kind: np.ndarray,
    context: np.ndarray,
    state: np.ndarray,
    role: np.ndarray,
    transforms: Mapping[str, object],
    vocabulary: Mapping[str, object],
) -> dict[str, np.ndarray]:
    age_parameters = transforms["age"]
    raw_parameters = transforms["raw"]
    state_parameters = transforms["state"]
    departments = vocabulary["departments"]
    _require(type(age_parameters) is dict and type(raw_parameters) is dict
             and type(state_parameters) is dict and type(departments) is tuple)
    context_design = _context_design(
        context, age_lower, age_upper, age_kind, age_parameters, departments
    )
    raw_design = _numeric_design(values, observed, raw_parameters)
    state_design = _state_design(state, state_parameters)
    designs = {
        "context": context_design,
        "raw_context": np.column_stack((context_design, raw_design)),
        "state_context": np.column_stack((context_design, state_design)),
    }
    _require(all(design.shape[0] == len(role) and bool(np.isfinite(design).all())
                 for design in designs.values()))
    return designs


def replay(
    values: np.ndarray,
    observed: np.ndarray,
    age_lower: np.ndarray,
    age_upper: np.ndarray,
    age_kind: np.ndarray,
    context: np.ndarray,
    state: np.ndarray,
    role: np.ndarray,
    outcome: np.ndarray,
    *,
    private_sink: dict[str, object],
) -> bool:
    """Verify stored fit predictions on the same arrays without refitting."""

    try:
        values, observed, age_lower, age_upper, age_kind, context, state, role, outcome = _validate_inputs(
            values, observed, age_lower, age_upper, age_kind, context, state, role, outcome
        )
        _require(type(private_sink) is dict and private_sink.get("schema") == PRIVATE_SCHEMA
                 and type(private_sink.get("status")) is str)
        sink_status = private_sink["status"]
        if sink_status == "unsupported_outcome_support":
            return not _support_ok(role, outcome)
        if sink_status == "unsupported_bootstrap":
            _require(_support_ok(role, outcome))
            test_labels = outcome[role == 2].astype(np.float64)
            return _bootstrap_valid_count(test_labels) < int(POLICY["minimum_valid_auroc_draws"])
        _require(sink_status == "ok")
        transforms = private_sink.get("transforms")
        vocabulary = private_sink.get("vocabulary")
        arms = private_sink.get("arms")
        _require(type(transforms) is dict and type(vocabulary) is dict and type(arms) is dict
                 and set(arms) == set(ARM_NAMES))
        designs = _replay_designs(
            values, observed, age_lower, age_upper, age_kind, context, state, role,
            transforms, vocabulary,
        )
        test = role == 2
        _require(int(private_sink.get("test_row_count", -1)) == int(test.sum()))
        for arm in ARM_NAMES:
            stored = arms[arm]
            _require(type(stored) is dict and type(stored.get("feature_width")) is int
                     and stored["feature_width"] == designs[arm].shape[1])
            model = stored.get("model")
            offset = stored.get("calibration_offset")
            _require(isinstance(model, LogisticRegression) and type(offset) is float and math.isfinite(offset))
            with threadpool_limits(limits=1):
                logits = np.asarray(model.decision_function(designs[arm][test]), dtype=np.float64)
            probabilities = np.asarray(_calibrated_probabilities(logits, offset), dtype=np.float64)
            expected = stored.get("test_predictions")
            _require(type(expected) is np.ndarray and expected.dtype == np.dtype(np.float64)
                     and expected.shape == probabilities.shape)
            if not np.array_equal(probabilities, expected, equal_nan=False):
                return False
        return True
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except ValueError as error:
        if str(error) == ERROR:
            raise
        raise ValueError(ERROR) from None
    except Exception:
        raise ValueError(ERROR) from None


def validate_report(value: object) -> bool:
    """Validate the closed aggregate schema without exposing row-level data."""

    try:
        _require(type(value) is dict and value.get("schema") == SCHEMA
                 and value.get("status") in {"ok", *_CLOSED_STATUSES}
                 and value.get("policy") == _policy()
                 and value.get("patient_level_output_emitted") is False
                 and value.get("clinical_utility_established") is False
                 and value.get("external_adaptation_performance_established") is False)
        if value["status"] in _CLOSED_STATUSES:
            _require(set(value) == _BASE_KEYS)
            return True
        _require(set(value) == _BASE_KEYS | {
            "arms", "contrasts", "primary_superiority", "support", "bootstrap"
        })
        arms = value["arms"]
        _require(type(arms) is dict and set(arms) == set(ARM_NAMES))

        def interval(value_: object, low: float, high: float) -> None:
            _require(type(value_) is list and len(value_) == 2
                     and all(type(item) is float and math.isfinite(item) for item in value_)
                     and low <= value_[0] <= value_[1] <= high)

        for arm in ARM_NAMES:
            item = arms[arm]
            _require(type(item) is dict and set(item) == {
                "status", "auroc", "logloss", "brier",
                "auroc_marginal_95ci", "logloss_marginal_95ci", "brier_marginal_95ci",
            } and item["status"] == "available")
            _require(all(type(item[metric]) is float and math.isfinite(item[metric])
                         for metric in ("auroc", "logloss", "brier")))
            _require(0.0 <= item["auroc"] <= 1.0
                     and 0.0 <= item["brier"] <= 1.0
                     and 0.0 <= item["logloss"] <= _MAX_LOGLOSS)
            interval(item["auroc_marginal_95ci"], 0.0, 1.0)
            interval(item["logloss_marginal_95ci"], 0.0, _MAX_LOGLOSS)
            interval(item["brier_marginal_95ci"], 0.0, 1.0)
        contrasts = value["contrasts"]
        _require(type(contrasts) is dict and set(contrasts) == set(CONTRAST_NAMES))
        contrast_pairs = {
            "state_context_minus_raw_context": ("state_context", "raw_context"),
            "state_context_minus_context": ("state_context", "context"),
        }
        for name in CONTRAST_NAMES:
            item = contrasts[name]
            _require(type(item) is dict and set(item) == {
                "status", "auroc_difference", "logloss_improvement", "brier_improvement",
                "auroc_marginal_95ci", "logloss_improvement_marginal_95ci", "brier_improvement_marginal_95ci",
                "auroc_adjusted_95ci", "logloss_improvement_adjusted_95ci", "brier_improvement_adjusted_95ci",
                "superiority",
            } and item["status"] == "available" and type(item["superiority"]) is bool)
            _require(all(type(item[key]) is float and math.isfinite(item[key])
                         for key in ("auroc_difference", "logloss_improvement", "brier_improvement")))
            plus, minus = contrast_pairs[name]
            _require(item["auroc_difference"] == arms[plus]["auroc"] - arms[minus]["auroc"]
                     and item["logloss_improvement"] == arms[minus]["logloss"] - arms[plus]["logloss"]
                     and item["brier_improvement"] == arms[minus]["brier"] - arms[plus]["brier"])
            _require(-1.0 <= item["auroc_difference"] <= 1.0
                     and -_MAX_LOGLOSS <= item["logloss_improvement"] <= _MAX_LOGLOSS
                     and -1.0 <= item["brier_improvement"] <= 1.0)
            interval(item["auroc_marginal_95ci"], -1.0, 1.0)
            interval(item["auroc_adjusted_95ci"], -1.0, 1.0)
            interval(item["logloss_improvement_marginal_95ci"], -_MAX_LOGLOSS, _MAX_LOGLOSS)
            interval(item["logloss_improvement_adjusted_95ci"], -_MAX_LOGLOSS, _MAX_LOGLOSS)
            interval(item["brier_improvement_marginal_95ci"], -1.0, 1.0)
            interval(item["brier_improvement_adjusted_95ci"], -1.0, 1.0)
            _require(item["auroc_adjusted_95ci"][0] <= item["auroc_marginal_95ci"][0]
                     <= item["auroc_marginal_95ci"][1] <= item["auroc_adjusted_95ci"][1])
            _require(item["logloss_improvement_adjusted_95ci"][0]
                     <= item["logloss_improvement_marginal_95ci"][0]
                     <= item["logloss_improvement_marginal_95ci"][1]
                     <= item["logloss_improvement_adjusted_95ci"][1])
            _require(item["brier_improvement_adjusted_95ci"][0]
                     <= item["brier_improvement_marginal_95ci"][0]
                     <= item["brier_improvement_marginal_95ci"][1]
                     <= item["brier_improvement_adjusted_95ci"][1])
            expected_superiority = (
                name == "state_context_minus_raw_context"
                and item["auroc_adjusted_95ci"][0] > 0.0
            )
            _require(item["superiority"] is expected_superiority)
        support = value["support"]
        _require(type(support) is dict and set(support) == {
            "status", "roles", "nonevent_event_lower_bounds"
        } and support["status"] == "released_lower_bounds"
                 and support["roles"] == list(ROLE_NAMES))
        bounds = support["nonevent_event_lower_bounds"]
        _require(type(bounds) is list and len(bounds) == 3 and all(
            type(row) is list and len(row) == 2 and all(type(item) is int and item >= 20 and item % 20 == 0 for item in row)
            for row in bounds
        ))
        bootstrap = value["bootstrap"]
        _require(type(bootstrap) is dict and set(bootstrap) == {
            "requested_draws", "accepted_auroc_draws"
        } and type(bootstrap["requested_draws"]) is int
                 and bootstrap["requested_draws"] == int(POLICY["bootstrap_draws"])
                 and type(bootstrap["accepted_auroc_draws"]) is int
                 and int(POLICY["minimum_valid_auroc_draws"]) <= bootstrap["accepted_auroc_draws"] <= bootstrap["requested_draws"])
        primary = contrasts["state_context_minus_raw_context"]
        _require(type(value["primary_superiority"]) is bool
                 and value["primary_superiority"] is (primary["auroc_adjusted_95ci"][0] > 0.0)
                 and value["primary_superiority"] is primary["superiority"])
        return True
    except Exception:
        return False


__all__ = [
    "ARM_NAMES",
    "CONTRAST_NAMES",
    "ERROR",
    "LAB_WIDTH",
    "POLICY",
    "PRIVATE_SCHEMA",
    "SCHEMA",
    "STATE_WIDTH",
    "evaluate",
    "replay",
    "validate_report",
]
