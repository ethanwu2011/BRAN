"""Pure fixed-readout outcome utility for the MIMIC V3 clinical panel.

The caller supplies an authenticated, already-preprocessed context design, the
discovery-standardized 192-D state control, private person keys, fixed role
membership, outcome labels, and (optionally) outcome-blind structure labels.
This module performs no source access, cohort selection, representation fit,
group selection, or row-level output.  It returns only closed aggregate
metrics; model predictions, participant weights, and bootstrap draws never
cross the function boundary.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
import math
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


ERROR = "bran_mimic_clinical_outcomes_v3_contract_failed"
SCHEMA = "bran-mimic-clinical-outcomes-v3"
PRIVATE_SCHEMA = "bran-mimic-clinical-outcomes-v3-private"
CONTEXT_WIDTH = 59
STATE_WIDTH = 192
GROUP_MAX = 6
ROLE_NAMES = ("discovery", "characterization_development", "test")
ARM_NAMES = (
    "context",
    "context_bran_groups",
    "context_raw_groups",
    "context_state",
    "context_state_bran_groups",
)
METRICS = ("auroc", "logloss", "brier")
CONTRASTS = MappingProxyType({
    "bran_groups_minus_context": ("context_bran_groups", "context"),
    "raw_groups_minus_context": ("context_raw_groups", "context"),
    "bran_groups_minus_raw_groups": ("context_bran_groups", "context_raw_groups"),
    "state_minus_context": ("context_state", "context"),
    "state_bran_groups_minus_state": ("context_state_bran_groups", "context_state"),
})

POLICY = MappingProxyType({
    "context_width": CONTEXT_WIDTH,
    "state_width": STATE_WIDTH,
    "logistic_c": 1.0,
    "logistic_solver": "lbfgs",
    "logistic_max_iter": 2000,
    "class_balancing": False,
    "calibration": "development_intercept_root_on_clipped_discovery_log_odds",
    "logit_clip": 30.0,
    "probability_clip": 1e-7,
    "bootstrap_draws": 5000,
    "minimum_valid_auroc_draws": 4950,
    "bootstrap_seed": 98581,
    "familywise_alpha": 0.05,
    "multiplicity_comparisons": 15,
    "minimum_class_support_per_role": 20,
    "fixed_fit_uncertainty_only": True,
})

_BASE_KEYS = {
    "schema", "status", "policy", "patient_level_output_emitted",
    "clinical_utility_established", "novel_subtype_claim", "external_validation_claim",
}
_FULL_KEYS = _BASE_KEYS | {"arms", "contrasts", "bootstrap"}
_CLOSED_STATUSES = {
    "unsupported_outcome_support", "unsupported_bootstrap",
}
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
        "novel_subtype_claim": False,
        "external_validation_claim": False,
    }


def _number(value: object, low: float, high: float) -> bool:
    return (
        type(value) in (int, float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and low <= float(value) <= high
    )


def _interval(value: object, low: float, high: float) -> bool:
    return (
        type(value) is list
        and len(value) == 2
        and all(_number(item, low, high) for item in value)
        and value[0] <= value[1]
    )


def _validate_role_vector(roles: object, rows: int) -> np.ndarray:
    """Validate the fixed 0/1/2 role vector used by the panel."""
    if type(roles) is np.ndarray:
        _require(roles.dtype == np.dtype(np.uint8) and roles.shape == (rows,)
                 and np.isin(roles, (0, 1, 2)).all())
        result = np.array(roles, dtype=np.uint8, copy=True)
    elif isinstance(roles, Mapping):
        _require(set(roles) == set(ROLE_NAMES))
        result = np.empty(rows, dtype=np.uint8)
        seen = np.zeros(rows, dtype=bool)
        for role, name in enumerate(ROLE_NAMES):
            values = roles[name]
            _require(type(values) is np.ndarray and values.dtype == np.dtype(np.int64)
                     and values.ndim == 1 and len(values) > 0)
            _require(np.all((values >= 0) & (values < rows))
                     and len(np.unique(values)) == len(values))
            _require(not seen[values].any())
            result[values] = role
            seen[values] = True
        _require(seen.all())
    else:
        _fail()
    _require(np.all(np.bincount(result, minlength=3) > 0))
    return result


def _validate_group_branch(labels: object, k: object, rows: int) -> tuple[np.ndarray | None, int | None]:
    """Return a usable K>=2 branch, or ``(None, None)`` when unavailable."""
    if labels is None and k is None:
        return None, None
    _require(labels is not None and type(k) is int and 1 <= k <= GROUP_MAX)
    _require(type(labels) is np.ndarray and labels.dtype == np.dtype(np.int64)
             and labels.shape == (rows,) and np.all((labels >= 0) & (labels < k)))
    copied = np.array(labels, dtype=np.int64, copy=True)
    if k == 1:
        _require(np.all(copied == 0))
        return None, None
    return copied, k


def _validate_inputs(
    context: object,
    state: object,
    person_group: object,
    roles: object,
    outcomes: object,
    bran_groups: object,
    bran_k: object,
    raw_groups: object,
    raw_k: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray | None, int | None], tuple[np.ndarray | None, int | None]]:
    _require(type(context) is np.ndarray and context.ndim == 2
             and context.shape[1] == CONTEXT_WIDTH and context.dtype.kind == "f")
    rows = context.shape[0]
    _require(rows > 0 and np.isfinite(context).all())
    _require(type(state) is np.ndarray and state.ndim == 2
             and state.shape == (rows, STATE_WIDTH) and state.dtype.kind == "f"
             and np.isfinite(state).all())
    _require(type(person_group) is np.ndarray and person_group.dtype == np.dtype(np.int64)
             and person_group.shape == (rows,) and np.all(person_group >= 0)
             and len(np.unique(person_group)) == rows)
    _require(type(outcomes) is np.ndarray and outcomes.dtype == np.dtype(np.int8)
             and outcomes.shape == (rows,) and np.isin(outcomes, (-1, 0, 1)).all())
    role_vector = _validate_role_vector(roles, rows)
    bran = _validate_group_branch(bran_groups, bran_k, rows)
    raw = _validate_group_branch(raw_groups, raw_k, rows)
    return (
        np.array(context, dtype=np.float64, copy=True),
        np.array(state, dtype=np.float64, copy=True),
        np.array(person_group, dtype=np.int64, copy=True),
        role_vector,
        np.array(outcomes, dtype=np.int8, copy=True),
        bran,
        raw,
    )


def _one_hot(labels: np.ndarray, k: int) -> np.ndarray:
    return (labels[:, None] == np.arange(1, k, dtype=np.int64)[None, :]).astype(np.float64)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    _require(np.isfinite(value).all())
    result = np.empty_like(value, dtype=np.float64)
    positive = value >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    _require(np.isfinite(result).all() and ((result > 0.0) & (result < 1.0)).all())
    return result


def _calibrated_probabilities(logits: np.ndarray, offset: float) -> np.ndarray:
    """Apply identical clipping before and after the development offset."""
    clipped = np.clip(np.asarray(logits, dtype=np.float64),
                      -float(POLICY["logit_clip"]), float(POLICY["logit_clip"]))
    calibrated = np.clip(clipped + float(offset),
                         -float(POLICY["logit_clip"]), float(POLICY["logit_clip"]))
    return _sigmoid(calibrated)


def _calibration_offset(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit only an intercept on development logits by a fixed bisection root."""
    clipped = np.clip(np.asarray(logits, dtype=np.float64),
                      -float(POLICY["logit_clip"]), float(POLICY["logit_clip"]))
    y = np.asarray(labels, dtype=np.float64)
    _require(clipped.ndim == 1 and y.shape == clipped.shape and np.isfinite(clipped).all()
             and np.isin(y, (0.0, 1.0)).all())
    lower, upper = -60.0, 60.0

    def score(offset: float) -> float:
        return float(np.sum(_calibrated_probabilities(clipped, offset) - y))

    low_score, high_score = score(lower), score(upper)
    _require(low_score <= 0.0 <= high_score)
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if score(midpoint) > 0.0:
            upper = midpoint
        else:
            lower = midpoint
    result = float((lower + upper) / 2.0)
    _require(math.isfinite(result))
    return result


def _fit_arm(
    x: np.ndarray,
    labels: np.ndarray,
    role: np.ndarray,
    *,
    arm_name: str | None = None,
    private_sink: dict[str, object] | None = None,
) -> np.ndarray:
    """Fit discovery logistic, development intercept calibration, test once."""
    discovery, development, test = (role == index for index in range(3))
    _require(discovery.any() and development.any() and test.any())
    model = LogisticRegression(
        C=float(POLICY["logistic_c"]),
        solver=str(POLICY["logistic_solver"]),
        max_iter=int(POLICY["logistic_max_iter"]),
        class_weight=None,
    )
    model.fit(x[discovery], labels[discovery])
    development_logits = np.asarray(model.decision_function(x[development]), dtype=np.float64)
    test_logits = np.asarray(model.decision_function(x[test]), dtype=np.float64)
    _require(development_logits.shape == (int(development.sum()),)
             and test_logits.shape == (int(test.sum()),)
             and np.isfinite(development_logits).all() and np.isfinite(test_logits).all())
    offset = _calibration_offset(development_logits, labels[development])
    probabilities = _calibrated_probabilities(test_logits, offset)
    _require(probabilities.shape == (int(test.sum()),) and np.isfinite(probabilities).all())
    if private_sink is not None:
        _require(type(private_sink) is dict and type(arm_name) is str and arm_name in ARM_NAMES)
        arms = private_sink.get("arms")
        _require(type(arms) is dict)
        _require(arm_name not in arms)
        private_sink["arms"][arm_name] = {
            "model": model,
            "calibration_offset": float(offset),
            "feature_width": int(x.shape[1]),
            "test_predictions": np.array(probabilities, dtype=np.float64, copy=True),
        }
    return np.asarray(probabilities, dtype=np.float64)


def _auc_plan(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(probabilities, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(probabilities[order]) != 0.0) + 1]
    return order, starts, labels[order].astype(np.float64)


def _weighted_auc(plan: tuple[np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray) -> float:
    order, starts, ordered_labels = plan
    ordered_weights = weights[order].astype(np.float64)
    positives = np.add.reduceat(ordered_weights * ordered_labels, starts)
    negatives = np.add.reduceat(ordered_weights * (1.0 - ordered_labels), starts)
    denominator = float(positives.sum() * negatives.sum())
    if denominator <= 0.0:
        return float("nan")
    return float(np.sum(positives * (np.cumsum(negatives) - 0.5 * negatives)) / denominator)


def _losses(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(probabilities, float(POLICY["probability_clip"]), 1.0 - float(POLICY["probability_clip"]))
    with np.errstate(divide="raise", invalid="raise"):
        logloss = -(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))
    brier = (labels - probabilities) ** 2
    _require(np.isfinite(logloss).all() and np.isfinite(brier).all())
    return logloss, brier


def _point_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    plan = _auc_plan(labels, probabilities)
    weights = np.ones(len(labels), dtype=np.int64)
    logloss, brier = _losses(labels, probabilities)
    values = {
        "auroc": _weighted_auc(plan, weights),
        "logloss": float(logloss.mean()),
        "brier": float(brier.mean()),
    }
    _require(all(math.isfinite(value) for value in values.values()))
    return values


def _prepare_private_sink(
    private_sink: object,
    design_binding: object,
) -> dict[str, object] | None:
    if private_sink is None:
        return None
    _require(type(private_sink) is dict and not private_sink)
    _require(design_binding is not None)
    private_sink["schema"] = PRIVATE_SCHEMA
    private_sink["design_binding"] = design_binding
    private_sink["arms"] = {}
    return private_sink


def _bootstrap(
    labels: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray] | None:
    """Stream paired participant resamples, retaining only private metric draws."""
    _require(type(predictions) is dict and set(predictions) and set(predictions).issubset(set(ARM_NAMES)))
    rows = len(labels)
    plans = {}
    losses = {}
    briers = {}
    for arm in predictions:
        probability = predictions[arm]
        _require(type(probability) is np.ndarray and probability.shape == (rows,)
                 and probability.dtype == np.dtype(np.float64) and np.isfinite(probability).all())
        plans[arm] = _auc_plan(labels, probability)
        losses[arm], briers[arm] = _losses(labels, probability)
    draws = {
        arm: {metric: np.empty(int(POLICY["bootstrap_draws"]), dtype=np.float64) for metric in METRICS}
        for arm in predictions
    }
    valid = np.zeros(int(POLICY["bootstrap_draws"]), dtype=bool)
    rng = np.random.default_rng(int(POLICY["bootstrap_seed"]))
    for draw in range(int(POLICY["bootstrap_draws"])):
        sampled = rng.integers(0, rows, size=rows)
        weights = np.bincount(sampled, minlength=rows)
        sampled_labels = labels[sampled]
        if not np.any(sampled_labels == 0) or not np.any(sampled_labels == 1):
            continue
        valid[draw] = True
        for arm in predictions:
            draws[arm]["auroc"][draw] = _weighted_auc(plans[arm], weights)
            draws[arm]["logloss"][draw] = float(np.dot(losses[arm], weights) / rows)
            draws[arm]["brier"][draw] = float(np.dot(briers[arm], weights) / rows)
    if int(valid.sum()) < int(POLICY["minimum_valid_auroc_draws"]):
        return None
    return draws, valid


def _quantile_interval(values: np.ndarray, adjusted: bool = False) -> list[float]:
    if adjusted:
        tail = float(POLICY["familywise_alpha"]) / float(POLICY["multiplicity_comparisons"]) / 2.0
    else:
        tail = 0.025
    result = np.quantile(values, (tail, 1.0 - tail))
    return [float(result[0]), float(result[1])]


def _available_arm(
    point: dict[str, float],
    draws: dict[str, np.ndarray],
    valid: np.ndarray,
) -> dict[str, object]:
    result: dict[str, object] = {"status": "available", **point}
    for metric in METRICS:
        result[metric + "_marginal_95ci"] = _quantile_interval(draws[metric][valid])
    return result


def _contrast_report(
    plus: str,
    minus: str,
    arm_reports: Mapping[str, dict[str, object]],
    draws: Mapping[str, dict[str, np.ndarray]],
    valid: np.ndarray,
) -> dict[str, object]:
    if arm_reports[plus]["status"] != "available" or arm_reports[minus]["status"] != "available":
        return {"status": "unavailable"}
    report: dict[str, object] = {"status": "available"}
    report["auroc_difference"] = float(arm_reports[plus]["auroc"] - arm_reports[minus]["auroc"])
    auc_difference = draws[plus]["auroc"][valid] - draws[minus]["auroc"][valid]
    report["auroc_marginal_95ci"] = _quantile_interval(auc_difference)
    report["auroc_adjusted_95ci"] = _quantile_interval(auc_difference, adjusted=True)
    # Loss benefits use the direction named in the protocol: comparator loss
    # minus added-arm loss, so a positive adjusted lower bound is favorable.
    for metric in ("logloss", "brier"):
        key = metric + "_improvement"
        report[key] = float(arm_reports[minus][metric] - arm_reports[plus][metric])
        improvement = draws[minus][metric][valid] - draws[plus][metric][valid]
        report[key + "_marginal_95ci"] = _quantile_interval(improvement)
        report[key + "_adjusted_95ci"] = _quantile_interval(improvement, adjusted=True)
    report["utility_gate"] = bool(
        report["logloss_improvement_adjusted_95ci"][0] > 0.0
        and report["auroc_difference"] >= 0.0
    )
    return report


def _evaluate(
    context: object,
    state: object,
    person_group: object,
    roles: object,
    outcomes: object,
    bran_groups: object,
    bran_k: object,
    raw_groups: object,
    raw_k: object,
    private_sink: object,
    design_binding: object,
) -> dict[str, object]:
    context, state, _person, role, outcomes, bran, raw = _validate_inputs(
        context, state, person_group, roles, outcomes, bran_groups, bran_k, raw_groups, raw_k
    )
    sink = _prepare_private_sink(private_sink, design_binding)
    known = outcomes >= 0
    if not np.any(known):
        if sink is not None:
            sink["status"] = "unsupported_outcome_support"
        return _closed("unsupported_outcome_support")
    role_known = role[known]
    labels = outcomes[known].astype(np.float64)
    if not all(
        int(np.sum((role_known == role_number) & (labels == label))) >= int(POLICY["minimum_class_support_per_role"])
        for role_number in range(3) for label in (0.0, 1.0)
    ):
        if sink is not None:
            sink["status"] = "unsupported_outcome_support"
        return _closed("unsupported_outcome_support")
    context_known = context[known]
    state_known = state[known]
    bran_known = None if bran[0] is None else (bran[0][known], bran[1])
    raw_known = None if raw[0] is None else (raw[0][known], raw[1])

    designs: dict[str, np.ndarray] = {"context": context_known,
                                      "context_state": np.column_stack((context_known, state_known))}
    if bran_known is not None:
        bran_labels, k = bran_known
        _require(k is not None)
        bran_one = _one_hot(bran_labels, k)
        designs["context_bran_groups"] = np.column_stack((context_known, bran_one))
        designs["context_state_bran_groups"] = np.column_stack((context_known, state_known, bran_one))
    if raw_known is not None:
        raw_labels, k = raw_known
        _require(k is not None)
        designs["context_raw_groups"] = np.column_stack((context_known, _one_hot(raw_labels, k)))

    test = role_known == 2
    test_labels = labels[test].astype(np.float64)
    predictions: dict[str, np.ndarray] = {}
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        for arm, design in designs.items():
            predictions[arm] = _fit_arm(design, labels, role_known, arm_name=arm, private_sink=sink)
    _require(set(predictions) in (set(ARM_NAMES), {"context", "context_state"},
                                  {"context", "context_state", "context_bran_groups", "context_state_bran_groups"},
                                  {"context", "context_state", "context_raw_groups"},
                                  {"context", "context_state", "context_bran_groups", "context_state_bran_groups", "context_raw_groups"}))
    # Missing structure branches are explicit unavailable arms; no branch is
    # substituted with another arm or silently removed from the contrast plan.
    bootstrap = _bootstrap(test_labels, predictions)
    if bootstrap is None:
        if sink is not None:
            sink["status"] = "unsupported_bootstrap"
        return _closed("unsupported_bootstrap")
    draws, valid = bootstrap
    arm_reports: dict[str, dict[str, object]] = {}
    for arm in ARM_NAMES:
        if arm in predictions:
            arm_reports[arm] = _available_arm(_point_metrics(test_labels, predictions[arm]), draws[arm], valid)
        else:
            arm_reports[arm] = {"status": "unavailable"}
    contrast_reports = {
        name: _contrast_report(plus, minus, arm_reports, draws, valid)
        for name, (plus, minus) in CONTRASTS.items()
    }
    status = "ok" if all(item["status"] == "available" for item in arm_reports.values()) else "group_branch_unavailable"
    result = {
        "schema": SCHEMA,
        "status": status,
        "policy": _policy(),
        "arms": arm_reports,
        "contrasts": contrast_reports,
        "bootstrap": {
            "requested_draws": int(POLICY["bootstrap_draws"]),
            "accepted_auroc_draws": int(valid.sum()),
        },
        "patient_level_output_emitted": False,
        "clinical_utility_established": False,
        "novel_subtype_claim": False,
        "external_validation_claim": False,
    }
    if sink is not None:
        sink["status"] = status
    _require(validate_report(result))
    return result


def evaluate(
    context: np.ndarray,
    state: np.ndarray,
    person_group: np.ndarray,
    roles: np.ndarray | Mapping[str, np.ndarray],
    outcomes: np.ndarray,
    bran_groups: np.ndarray | None = None,
    bran_k: int | None = None,
    raw_groups: np.ndarray | None = None,
    raw_k: int | None = None,
    *,
    private_sink: dict[str, object] | None = None,
    design_binding: object | None = None,
) -> dict[str, object]:
    """Evaluate the fixed V3 readout recipe and return only a closed report."""
    try:
        return _evaluate(context, state, person_group, roles, outcomes,
                         bran_groups, bran_k, raw_groups, raw_k,
                         private_sink, design_binding)
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


evaluate_outcomes = evaluate


def validate_report(value: object) -> bool:
    """Validate that a report contains only the closed aggregate schema."""
    try:
        _require(type(value) is dict and value.get("schema") == SCHEMA
                 and type(value.get("status")) is str
                 and value["status"] in _CLOSED_STATUSES | {"ok", "group_branch_unavailable"})
        _require(type(value.get("policy")) is dict and value["policy"] == _policy())
        _require(value.get("patient_level_output_emitted") is False
                 and value.get("clinical_utility_established") is False
                 and value.get("novel_subtype_claim") is False
                 and value.get("external_validation_claim") is False)
        status = value["status"]
        if status in _CLOSED_STATUSES:
            _require(set(value) == _BASE_KEYS)
            return True
        _require(set(value) == _FULL_KEYS)
        arms = value["arms"]
        _require(type(arms) is dict and set(arms) == set(ARM_NAMES))
        _require(arms["context"]["status"] == "available"
                 and arms["context_state"]["status"] == "available")
        all_available = all(item["status"] == "available" for item in arms.values())
        _require((status == "ok") is all_available)
        for item in arms.values():
            _require(type(item) is dict and item.get("status") in {"available", "unavailable"})
            if item["status"] == "unavailable":
                _require(set(item) == {"status"})
                continue
            _require(set(item) == {"status", *METRICS, *(metric + "_marginal_95ci" for metric in METRICS)})
            for metric in METRICS:
                high = _MAX_LOGLOSS if metric == "logloss" else 1.0
                _require(_number(item[metric], 0.0, high)
                         and _interval(item[metric + "_marginal_95ci"], 0.0, high))
        contrasts = value["contrasts"]
        _require(type(contrasts) is dict and set(contrasts) == set(CONTRASTS))
        for name, (plus, minus) in CONTRASTS.items():
            item = contrasts[name]
            _require(type(item) is dict and item.get("status") in {"available", "unavailable"})
            expected_available = arms[plus]["status"] == "available" and arms[minus]["status"] == "available"
            _require((item["status"] == "available") is expected_available)
            if item["status"] == "unavailable":
                _require(set(item) == {"status"})
                continue
            _require(set(item) == {
                "status", "auroc_difference", "logloss_improvement", "brier_improvement",
                "auroc_marginal_95ci", "logloss_improvement_marginal_95ci", "brier_improvement_marginal_95ci",
                "auroc_adjusted_95ci", "logloss_improvement_adjusted_95ci", "brier_improvement_adjusted_95ci", "utility_gate",
            })
            _require(_number(item["auroc_difference"], -1.0, 1.0)
                     and _interval(item["auroc_marginal_95ci"], -1.0, 1.0)
                     and _interval(item["auroc_adjusted_95ci"], -1.0, 1.0)
                     and item["auroc_difference"] == arms[plus]["auroc"] - arms[minus]["auroc"])
            for metric in ("logloss", "brier"):
                high = _MAX_LOGLOSS if metric == "logloss" else 1.0
                difference = item[metric + "_improvement"]
                _require(_number(difference, -high, high))
                _require(difference == arms[minus][metric] - arms[plus][metric])
                _require(_interval(item[metric + "_improvement_marginal_95ci"], -high, high))
                _require(_interval(item[metric + "_improvement_adjusted_95ci"], -high, high))
                _require(item[metric + "_improvement_adjusted_95ci"][0]
                         <= item[metric + "_improvement_marginal_95ci"][0]
                         <= item[metric + "_improvement_marginal_95ci"][1]
                         <= item[metric + "_improvement_adjusted_95ci"][1])
            _require(item["auroc_adjusted_95ci"][0] <= item["auroc_marginal_95ci"][0]
                     <= item["auroc_marginal_95ci"][1] <= item["auroc_adjusted_95ci"][1])
            _require(type(item["utility_gate"]) is bool
                     and item["utility_gate"] is (item["logloss_improvement_adjusted_95ci"][0] > 0.0
                                                   and item["auroc_difference"] >= 0.0))
        bootstrap = value["bootstrap"]
        _require(type(bootstrap) is dict and set(bootstrap) == {"requested_draws", "accepted_auroc_draws"}
                 and type(bootstrap["requested_draws"]) is int
                 and bootstrap["requested_draws"] == int(POLICY["bootstrap_draws"])
                 and type(bootstrap["accepted_auroc_draws"]) is int
                 and int(POLICY["minimum_valid_auroc_draws"]) <= bootstrap["accepted_auroc_draws"] <= bootstrap["requested_draws"])
        _require(status == "group_branch_unavailable"
                 or all(item["status"] == "available" for item in arms.values()))
    except Exception:
        return False
    return True


__all__ = [
    "ARM_NAMES", "CONTEXT_WIDTH", "CONTRASTS", "ERROR", "METRICS", "POLICY", "PRIVATE_SCHEMA",
    "ROLE_NAMES", "SCHEMA", "STATE_WIDTH", "evaluate", "evaluate_outcomes",
    "validate_report",
]
