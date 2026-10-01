"""Frozen, local-only BRAN CGM added-value regression comparison kernel.

This module deliberately accepts already-created local design matrices.  It
does not create representations, read files, retain estimators, or return row
level outputs.  In particular, outcome values must not be used to create the
three input designs before calling this API.
"""
from __future__ import annotations

from collections.abc import Mapping
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


SCHEMA_VERSION = "bran_cgm_added_value_kernel_v1"
DISCOVERY_FOLDS = (0, 1, 2)
VALIDATION_FOLD = 3
REPLICATION_FOLD = 4
RIDGE_ALPHAS = (0.1, 10.0, 1000.0)
EXTRA_TREES_SEED = 94501
BOOTSTRAP_SEED = 94501
BOOTSTRAP_RESAMPLES = 500
ARM_NAMES = (
    "severity_ridge",
    "severity_extra_trees",
    "severity_state_ridge",
    "severity_state_extra_trees",
    "raw_clinical_ridge",
    "raw_clinical_extra_trees",
    "mean",
)
CONTRAST_NAMES = (
    "primary_severity_state_extra_trees_minus_severity_extra_trees",
    "secondary_severity_state_ridge_minus_severity_ridge",
    "secondary_severity_state_extra_trees_minus_raw_clinical_extra_trees",
)
CONTRAST_ARMS = {
    "primary_severity_state_extra_trees_minus_severity_extra_trees": ("severity_state_extra_trees", "severity_extra_trees"),
    "secondary_severity_state_ridge_minus_severity_ridge": ("severity_state_ridge", "severity_ridge"),
    "secondary_severity_state_extra_trees_minus_raw_clinical_extra_trees": ("severity_state_extra_trees", "raw_clinical_extra_trees"),
}


class CGMAddedValueKernelError(ValueError):
    """Raised for an invalid local-only input contract."""


def _strict_bool(value: object, name: str, rows: int) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (rows,) or result.dtype != np.dtype(bool):
        raise CGMAddedValueKernelError(f"invalid_{name}")
    return result


def _matrix(value: object, name: str, rows: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] < 1 or (rows is not None and raw.shape[0] != rows):
        raise CGMAddedValueKernelError(f"invalid_{name}")
    if raw.dtype.kind not in "fiu" or raw.dtype.kind == "b":
        raise CGMAddedValueKernelError(f"invalid_{name}")
    result = raw.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise CGMAddedValueKernelError(f"nonfinite_{name}")
    return result


def _target(value: object, rows: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (rows,) or raw.dtype.kind != "f":
        raise CGMAddedValueKernelError("invalid_y")
    result = raw.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise CGMAddedValueKernelError("nonfinite_y")
    return result


def _outer_folds(value: object, rows: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (rows,) or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise CGMAddedValueKernelError("invalid_outer")
    result = raw.astype(np.int64, copy=False)
    if np.any((result < 0) | (result > 4)):
        raise CGMAddedValueKernelError("invalid_outer")
    return result


def _unique_ids(value: object, rows: int) -> None:
    if value is None:
        return
    raw = np.asarray(value, dtype=object)
    if raw.shape != (rows,):
        raise CGMAddedValueKernelError("invalid_ids")
    identifiers = raw.tolist()
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise CGMAddedValueKernelError("invalid_ids")
    if len(set(identifiers)) != rows:
        raise CGMAddedValueKernelError("nonunique_ids")


def _support_flags(eligible: np.ndarray, outer: np.ndarray) -> dict[str, bool]:
    return {
        "discovery_at_least_80": bool(np.sum(eligible & np.isin(outer, DISCOVERY_FOLDS)) >= 80),
        "validation_at_least_40": bool(np.sum(eligible & (outer == VALIDATION_FOLD)) >= 40),
        "replication_at_least_40": bool(np.sum(eligible & (outer == REPLICATION_FOLD)) >= 40),
        "each_discovery_inner_fold_at_least_10": bool(
            all(np.sum(eligible & (outer == fold)) >= 10 for fold in DISCOVERY_FOLDS)
        ),
    }


def _closed_status(status: str, support: dict[str, bool]) -> dict[str, object]:
    """A count-free suppressed result for inadequate support or numeric failure."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "clinical_claim": False,
        "novel_claim": False,
        "external_claim": False,
        "support": support,
    }


def _fit_ridge_predict(
    x_scaler_fit: np.ndarray,
    x_target_fit: np.ndarray,
    y_target_fit: np.ndarray,
    x_score: np.ndarray,
    alpha: float,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        scaler = StandardScaler().fit(x_scaler_fit)
        model = Ridge(alpha=alpha).fit(scaler.transform(x_target_fit), y_target_fit)
        prediction = model.predict(scaler.transform(x_score))
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != (len(x_score),) or not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite_prediction")
    return prediction


def _select_ridge_alpha(x: np.ndarray, y: np.ndarray, outer: np.ndarray) -> float:
    """Choose from inner discovery folds only, using count-weighted MSE."""
    scores: dict[float, float] = {}
    for alpha in RIDGE_ALPHAS:
        squared_error = 0.0
        observation_count = 0
        for held_out in DISCOVERY_FOLDS:
            score_rows = outer == held_out
            fit_rows = np.isin(outer, DISCOVERY_FOLDS) & ~score_rows
            prediction = _fit_ridge_predict(x[fit_rows], x[fit_rows], y[fit_rows], x[score_rows], alpha)
            with np.errstate(over="ignore", invalid="ignore"):
                error = prediction - y[score_rows]
                contribution = float(error @ error)
            if not np.isfinite(contribution):
                raise FloatingPointError("nonfinite_inner_score")
            squared_error += contribution
            observation_count += int(score_rows.sum())
        score = squared_error / observation_count
        if not np.isfinite(score):
            raise FloatingPointError("nonfinite_inner_score")
        scores[alpha] = score
    # Larger alpha is intentionally chosen for exact ties.
    return max(scores.items(), key=lambda item: (-item[1], item[0]))[0]


def _fit_predictions(
    x: np.ndarray, y: np.ndarray, discovery_outer: np.ndarray, score_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    alpha = _select_ridge_alpha(x, y, discovery_outer)
    ridge = _fit_ridge_predict(x, x, y, score_x, alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        forest = ExtraTreesRegressor(
            n_estimators=128,
            max_depth=8,
            min_samples_leaf=5,
            max_features=1.0,
            n_jobs=1,
            random_state=EXTRA_TREES_SEED,
        ).fit(x, y)
        trees = np.asarray(forest.predict(score_x), dtype=np.float64)
    if trees.shape != (len(score_x),) or not np.isfinite(trees).all():
        raise FloatingPointError("nonfinite_prediction")
    return ridge, trees


def _metrics(y: np.ndarray, prediction: np.ndarray, iqr: float) -> dict[str, float | None]:
    with np.errstate(over="ignore", invalid="ignore"):
        error = prediction - y
        mse = float(np.mean(error * error))
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(mse))
        variance_sum = float(np.sum((y - np.mean(y)) ** 2))
        r2 = None if variance_sum == 0.0 else float(1.0 - np.sum(error * error) / variance_sum)
        normalized_mse = float(mse / (iqr * iqr))
    values = (mae, rmse, normalized_mse) + (() if r2 is None else (r2,))
    if not all(np.isfinite(value) for value in values):
        raise FloatingPointError("nonfinite_metric")
    return {
        "mae_mg_dl": mae,
        "rmse_mg_dl": rmse,
        "normalized_mse": normalized_mse,
        "r2": r2,
    }


def _bootstrap_contrast(
    y: np.ndarray, first: np.ndarray, second: np.ndarray, iqr: float, draws: np.ndarray
) -> dict[str, object]:
    # This is first minus second; negative values favor the first arm.
    with np.errstate(over="ignore", invalid="ignore"):
        errors = (first - y) ** 2 - (second - y) ** 2
        point = float(np.mean(errors) / (iqr * iqr))
        samples = np.mean(errors[draws], axis=1) / (iqr * iqr)
    if not np.isfinite(point) or not np.isfinite(samples).all():
        raise FloatingPointError("nonfinite_bootstrap")
    low, high = np.percentile(samples, (2.5, 97.5))
    return {
        "normalized_mse_difference": point,
        "bootstrap_marginal_95ci": [float(low), float(high)],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }


def _evaluation_report(
    y: np.ndarray, predictions: dict[str, np.ndarray], iqr: float
) -> tuple[dict[str, dict[str, float | None]], dict[str, dict[str, object]]]:
    arms = {name: _metrics(y, prediction, iqr) for name, prediction in predictions.items()}
    # Reinitializing per evaluation split makes all contrasts within that split
    # use exactly the same paired bootstrap draws, fixed at the frozen seed.
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(y), size=(BOOTSTRAP_RESAMPLES, len(y)))
    contrasts = {
        CONTRAST_NAMES[0]: _bootstrap_contrast(
            y, predictions["severity_state_extra_trees"], predictions["severity_extra_trees"], iqr, draws
        ),
        CONTRAST_NAMES[1]: _bootstrap_contrast(
            y, predictions["severity_state_ridge"], predictions["severity_ridge"], iqr, draws
        ),
        CONTRAST_NAMES[2]: _bootstrap_contrast(
            y, predictions["severity_state_extra_trees"], predictions["raw_clinical_extra_trees"], iqr, draws
        ),
    }
    # Publish the point estimate from the already-published arm metrics so the
    # closed report has one exact aggregate identity even under large-value
    # floating-point cancellation.  Bootstrap samples remain paired errors.
    for contrast_name, (first_arm, second_arm) in CONTRAST_ARMS.items():
        contrasts[contrast_name]["normalized_mse_difference"] = float(
            arms[first_arm]["normalized_mse"] - arms[second_arm]["normalized_mse"]
        )
    return arms, contrasts


def fit_and_evaluate(
    designs: dict[str, np.ndarray],
    y: np.ndarray,
    observed: np.ndarray,
    outer: np.ndarray,
    membership: np.ndarray,
    ids: np.ndarray | None = None,
) -> dict[str, object]:
    """Fit frozen heads locally and return only aggregate, count-free metrics.

    The discovery folds are 0--2, validation is 3, and replication is 4.
    Only rows that are both ``observed`` and source-defined ``membership`` are
    used for any fit or score.  The input matrices are features only; callers
    are responsible for ensuring CGM outcome values never informed their
    construction.
    """
    if not isinstance(designs, Mapping) or set(designs) != {"severity", "severity_state", "raw_clinical"}:
        raise CGMAddedValueKernelError("invalid_designs")
    severity = _matrix(designs["severity"], "severity")
    rows = len(severity)
    if rows == 0:
        raise CGMAddedValueKernelError("invalid_severity")
    matrices = {
        "severity": severity,
        "severity_state": _matrix(designs["severity_state"], "severity_state", rows),
        "raw_clinical": _matrix(designs["raw_clinical"], "raw_clinical", rows),
    }
    target = _target(y, rows)
    observed_mask = _strict_bool(observed, "observed", rows)
    member_mask = _strict_bool(membership, "membership", rows)
    outer_fold = _outer_folds(outer, rows)
    _unique_ids(ids, rows)

    eligible = observed_mask & member_mask
    support = _support_flags(eligible, outer_fold)
    if not all(support.values()):
        return _closed_status("unsupported_target_support", support)

    discovery_rows = eligible & np.isin(outer_fold, DISCOVERY_FOLDS)
    try:
        discovery_y = target[discovery_rows]
        discovery_outer = outer_fold[discovery_rows]
        with np.errstate(over="ignore", invalid="ignore"):
            iqr = float(np.percentile(discovery_y, 75) - np.percentile(discovery_y, 25))
        if not np.isfinite(iqr):
            raise FloatingPointError("nonfinite_iqr")
        iqr = max(iqr, 1e-6)

        validation_rows = eligible & (outer_fold == VALIDATION_FOLD)
        replication_rows = eligible & (outer_fold == REPLICATION_FOLD)
        validation_count = int(np.sum(validation_rows))
        combined_predictions: dict[str, np.ndarray] = {}
        # Each design has exactly one final discovery fit.  Concatenating the
        # two held-out feature sets is prediction-only and never exposes their
        # targets to a fit or tuning decision.
        for design_name, arm_prefix in (
            ("severity", "severity"),
            ("severity_state", "severity_state"),
            ("raw_clinical", "raw_clinical"),
        ):
            score_x = np.concatenate((matrices[design_name][validation_rows], matrices[design_name][replication_rows]), axis=0)
            ridge, trees = _fit_predictions(
                matrices[design_name][discovery_rows], discovery_y, discovery_outer, score_x
            )
            combined_predictions[f"{arm_prefix}_ridge"] = ridge
            combined_predictions[f"{arm_prefix}_extra_trees"] = trees
        combined_predictions["mean"] = np.full(
            validation_count + int(np.sum(replication_rows)), float(np.mean(discovery_y)), dtype=np.float64
        )
        arm_by_split: dict[str, dict[str, dict[str, float | None]]] = {}
        contrast_by_split: dict[str, dict[str, dict[str, object]]] = {}
        for split_name, split_rows, start, stop in (
            ("validation", validation_rows, 0, validation_count),
            ("replication", replication_rows, validation_count, None),
        ):
            score_y = target[split_rows]
            predictions = {name: prediction[start:stop] for name, prediction in combined_predictions.items()}
            arm_by_split[split_name], contrast_by_split[split_name] = _evaluation_report(score_y, predictions, iqr)
    except (ConvergenceWarning, FloatingPointError, ValueError, OverflowError):
        return _closed_status("numerical_failure", support)

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "clinical_claim": False,
        "novel_claim": False,
        "external_claim": False,
        "support": support,
        "arms": {name: {split: arm_by_split[split][name] for split in ("validation", "replication")} for name in ARM_NAMES},
        "contrasts": {
            name: {split: contrast_by_split[split][name] for split in ("validation", "replication")}
            for name in CONTRAST_NAMES
        },
    }
    validate_report(report)
    return report


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)) and bool(np.isfinite(value))


def _require_exact_keys(value: object, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise CGMAddedValueKernelError("invalid_report")
    return value


def validate_report(report: object) -> None:
    """Reject anything outside the frozen, aggregate-only report schema."""
    base_keys = {"schema_version", "status", "clinical_claim", "novel_claim", "external_claim", "support"}
    if not isinstance(report, Mapping):
        raise CGMAddedValueKernelError("invalid_report")
    status = report.get("status")
    expected = base_keys | ({"arms", "contrasts"} if status == "ok" else set())
    root = _require_exact_keys(report, expected)
    if root["schema_version"] != SCHEMA_VERSION or status not in {"ok", "unsupported_target_support", "numerical_failure"}:
        raise CGMAddedValueKernelError("invalid_report")
    if root["clinical_claim"] is not False or root["novel_claim"] is not False or root["external_claim"] is not False:
        raise CGMAddedValueKernelError("invalid_report")
    support = _require_exact_keys(root["support"], {
        "discovery_at_least_80", "validation_at_least_40", "replication_at_least_40", "each_discovery_inner_fold_at_least_10",
    })
    if any(type(flag) is not bool for flag in support.values()):
        raise CGMAddedValueKernelError("invalid_report")
    if status in {"ok", "numerical_failure"} and not all(support.values()):
        raise CGMAddedValueKernelError("invalid_report")
    if status == "unsupported_target_support" and all(support.values()):
        raise CGMAddedValueKernelError("invalid_report")
    if status != "ok":
        return
    arms = _require_exact_keys(root["arms"], set(ARM_NAMES))
    for arm in arms.values():
        splits = _require_exact_keys(arm, {"validation", "replication"})
        for metric in splits.values():
            values = _require_exact_keys(metric, {"mae_mg_dl", "rmse_mg_dl", "normalized_mse", "r2"})
            if not all(_finite_number(values[key]) and float(values[key]) >= 0.0 for key in ("mae_mg_dl", "rmse_mg_dl", "normalized_mse")):
                raise CGMAddedValueKernelError("invalid_report")
            if values["r2"] is not None and (not _finite_number(values["r2"]) or float(values["r2"]) > 1.0):
                raise CGMAddedValueKernelError("invalid_report")
            if float(values["mae_mg_dl"]) > float(values["rmse_mg_dl"]) + 1e-12 * max(1.0, abs(float(values["rmse_mg_dl"]))):
                raise CGMAddedValueKernelError("invalid_report")
    contrasts = _require_exact_keys(root["contrasts"], set(CONTRAST_NAMES))
    for contrast_name, contrast in contrasts.items():
        first_arm, second_arm = CONTRAST_ARMS[contrast_name]
        splits = _require_exact_keys(contrast, {"validation", "replication"})
        for split_name, result in splits.items():
            values = _require_exact_keys(result, {"normalized_mse_difference", "bootstrap_marginal_95ci", "bootstrap_resamples"})
            ci = values["bootstrap_marginal_95ci"]
            if not _finite_number(values["normalized_mse_difference"]) or not isinstance(ci, list) or len(ci) != 2:
                raise CGMAddedValueKernelError("invalid_report")
            if not all(_finite_number(bound) for bound in ci) or float(ci[0]) > float(ci[1]):
                raise CGMAddedValueKernelError("invalid_report")
            if type(values["bootstrap_resamples"]) is not int or values["bootstrap_resamples"] != BOOTSTRAP_RESAMPLES:
                raise CGMAddedValueKernelError("invalid_report")
            expected_difference = float(arms[first_arm][split_name]["normalized_mse"]) - float(
                arms[second_arm][split_name]["normalized_mse"]
            )
            reported_difference = float(values["normalized_mse_difference"])
            if reported_difference != expected_difference:
                raise CGMAddedValueKernelError("invalid_report")
