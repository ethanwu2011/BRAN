"""Frozen local-only CGM utility comparison for two locked disease groups.

The caller supplies feature-only, in-memory designs.  ``severity_group`` is
strictly the severity design with one trailing 0/1 locked-group indicator;
this module neither constructs group membership nor reads any external data.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from bran_cgm_added_value_kernel_v1 import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DISCOVERY_FOLDS,
    REPLICATION_FOLD,
    VALIDATION_FOLD,
    CGMAddedValueKernelError as SubgroupUtilityKernelError,
    _bootstrap_contrast,
    _finite_number,
    _fit_predictions,
    _matrix,
    _metrics,
    _outer_folds,
    _require_exact_keys,
    _strict_bool,
    _support_flags,
    _target,
    _unique_ids,
)


SCHEMA_VERSION = "bran_subgroup_utility_kernel_v1"
ARM_NAMES = (
    "severity_ridge",
    "severity_extra_trees",
    "severity_group_ridge",
    "severity_group_extra_trees",
    "mean",
)
CONTRAST_NAMES = (
    "severity_group_extra_trees_minus_severity_extra_trees",
    "severity_group_ridge_minus_severity_ridge",
)
CONTRAST_ARMS = {
    "severity_group_extra_trees_minus_severity_extra_trees": ("severity_group_extra_trees", "severity_extra_trees"),
    "severity_group_ridge_minus_severity_ridge": ("severity_group_ridge", "severity_ridge"),
}


def _closed_status(status: str, support: dict[str, bool]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "clinical_claim": False,
        "novel_claim": False,
        "external_claim": False,
        "support": support,
    }


def _group_designs(designs: object) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(designs, Mapping) or set(designs) != {"severity", "severity_group"}:
        raise SubgroupUtilityKernelError("invalid_designs")
    severity = _matrix(designs["severity"], "severity")
    if len(severity) == 0:
        raise SubgroupUtilityKernelError("invalid_severity")
    severity_group = _matrix(designs["severity_group"], "severity_group", len(severity))
    if severity_group.shape[1] != severity.shape[1] + 1:
        raise SubgroupUtilityKernelError("invalid_severity_group_columns")
    if not np.array_equal(severity_group[:, :-1], severity):
        raise SubgroupUtilityKernelError("severity_group_not_severity_plus_indicator")
    indicator = severity_group[:, -1]
    if not np.isin(indicator, (0.0, 1.0)).all() or not (np.any(indicator == 0.0) and np.any(indicator == 1.0)):
        raise SubgroupUtilityKernelError("invalid_group_indicator")
    return severity, severity_group


def _evaluation_report(
    y: np.ndarray, predictions: dict[str, np.ndarray], iqr: float
) -> tuple[dict[str, dict[str, float | None]], dict[str, dict[str, object]]]:
    arms = {name: _metrics(y, prediction, iqr) for name, prediction in predictions.items()}
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(y), size=(BOOTSTRAP_RESAMPLES, len(y)))
    contrasts = {
        name: _bootstrap_contrast(y, predictions[first], predictions[second], iqr, draws)
        for name, (first, second) in CONTRAST_ARMS.items()
    }
    # Keep the published contrast point exactly identical to the corresponding
    # published arm metrics; bootstrap intervals are still paired patient draws.
    for name, (first, second) in CONTRAST_ARMS.items():
        contrasts[name]["normalized_mse_difference"] = float(
            arms[first]["normalized_mse"] - arms[second]["normalized_mse"]
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
    """Fit frozen subgroup utility heads and return aggregate-only results."""
    severity, severity_group = _group_designs(designs)
    rows = len(severity)
    target = _target(y, rows)
    observed_mask = _strict_bool(observed, "observed", rows)
    membership_mask = _strict_bool(membership, "membership", rows)
    outer_fold = _outer_folds(outer, rows)
    _unique_ids(ids, rows)

    eligible = observed_mask & membership_mask
    support = _support_flags(eligible, outer_fold)
    if not all(support.values()):
        return _closed_status("unsupported_target_support", support)

    discovery_rows = eligible & np.isin(outer_fold, DISCOVERY_FOLDS)
    validation_rows = eligible & (outer_fold == VALIDATION_FOLD)
    replication_rows = eligible & (outer_fold == REPLICATION_FOLD)
    try:
        discovery_y = target[discovery_rows]
        discovery_outer = outer_fold[discovery_rows]
        with np.errstate(over="ignore", invalid="ignore"):
            iqr = float(np.percentile(discovery_y, 75) - np.percentile(discovery_y, 25))
        if not np.isfinite(iqr):
            raise FloatingPointError("nonfinite_iqr")
        iqr = max(iqr, 1e-6)

        validation_count = int(np.sum(validation_rows))
        combined_predictions: dict[str, np.ndarray] = {}
        for design, prefix in ((severity, "severity"), (severity_group, "severity_group")):
            score_x = np.concatenate((design[validation_rows], design[replication_rows]), axis=0)
            ridge, trees = _fit_predictions(design[discovery_rows], discovery_y, discovery_outer, score_x)
            combined_predictions[f"{prefix}_ridge"] = ridge
            combined_predictions[f"{prefix}_extra_trees"] = trees
        combined_predictions["mean"] = np.full(
            validation_count + int(np.sum(replication_rows)), float(np.mean(discovery_y)), dtype=np.float64
        )

        arms_by_split: dict[str, dict[str, dict[str, float | None]]] = {}
        contrasts_by_split: dict[str, dict[str, dict[str, object]]] = {}
        for split_name, split_rows, start, stop in (
            ("validation", validation_rows, 0, validation_count),
            ("replication", replication_rows, validation_count, None),
        ):
            predictions = {name: values[start:stop] for name, values in combined_predictions.items()}
            arms_by_split[split_name], contrasts_by_split[split_name] = _evaluation_report(target[split_rows], predictions, iqr)
    except (ConvergenceWarning, FloatingPointError, ValueError, OverflowError):
        return _closed_status("numerical_failure", support)

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "clinical_claim": False,
        "novel_claim": False,
        "external_claim": False,
        "support": support,
        "arms": {name: {split: arms_by_split[split][name] for split in ("validation", "replication")} for name in ARM_NAMES},
        "contrasts": {
            name: {split: contrasts_by_split[split][name] for split in ("validation", "replication")}
            for name in CONTRAST_NAMES
        },
    }
    validate_report(report)
    return report


def validate_report(report: object) -> None:
    """Validate the exact aggregate-only subgroup utility report schema."""
    base = {"schema_version", "status", "clinical_claim", "novel_claim", "external_claim", "support"}
    if not isinstance(report, Mapping):
        raise SubgroupUtilityKernelError("invalid_report")
    status = report.get("status")
    root = _require_exact_keys(report, base | ({"arms", "contrasts"} if status == "ok" else set()))
    if root["schema_version"] != SCHEMA_VERSION or status not in {"ok", "unsupported_target_support", "numerical_failure"}:
        raise SubgroupUtilityKernelError("invalid_report")
    if root["clinical_claim"] is not False or root["novel_claim"] is not False or root["external_claim"] is not False:
        raise SubgroupUtilityKernelError("invalid_report")
    support = _require_exact_keys(root["support"], {
        "discovery_at_least_80", "validation_at_least_40", "replication_at_least_40", "each_discovery_inner_fold_at_least_10",
    })
    if any(type(flag) is not bool for flag in support.values()):
        raise SubgroupUtilityKernelError("invalid_report")
    if status in {"ok", "numerical_failure"} and not all(support.values()):
        raise SubgroupUtilityKernelError("invalid_report")
    if status == "unsupported_target_support" and all(support.values()):
        raise SubgroupUtilityKernelError("invalid_report")
    if status != "ok":
        return

    arms = _require_exact_keys(root["arms"], set(ARM_NAMES))
    for arm in arms.values():
        for metrics in _require_exact_keys(arm, {"validation", "replication"}).values():
            values = _require_exact_keys(metrics, {"mae_mg_dl", "rmse_mg_dl", "normalized_mse", "r2"})
            if not all(_finite_number(values[key]) and float(values[key]) >= 0.0 for key in ("mae_mg_dl", "rmse_mg_dl", "normalized_mse")):
                raise SubgroupUtilityKernelError("invalid_report")
            if values["r2"] is not None and (not _finite_number(values["r2"]) or float(values["r2"]) > 1.0):
                raise SubgroupUtilityKernelError("invalid_report")
            if float(values["mae_mg_dl"]) > float(values["rmse_mg_dl"]) + 1e-12 * max(1.0, abs(float(values["rmse_mg_dl"]))):
                raise SubgroupUtilityKernelError("invalid_report")

    contrasts = _require_exact_keys(root["contrasts"], set(CONTRAST_NAMES))
    for name, contrast in contrasts.items():
        first, second = CONTRAST_ARMS[name]
        for split, result in _require_exact_keys(contrast, {"validation", "replication"}).items():
            values = _require_exact_keys(result, {"normalized_mse_difference", "bootstrap_marginal_95ci", "bootstrap_resamples"})
            ci = values["bootstrap_marginal_95ci"]
            if not _finite_number(values["normalized_mse_difference"]) or not isinstance(ci, list) or len(ci) != 2:
                raise SubgroupUtilityKernelError("invalid_report")
            if not all(_finite_number(bound) for bound in ci) or float(ci[0]) > float(ci[1]):
                raise SubgroupUtilityKernelError("invalid_report")
            if type(values["bootstrap_resamples"]) is not int or values["bootstrap_resamples"] != BOOTSTRAP_RESAMPLES:
                raise SubgroupUtilityKernelError("invalid_report")
            expected = float(arms[first][split]["normalized_mse"]) - float(arms[second][split]["normalized_mse"])
            if float(values["normalized_mse_difference"]) != expected:
                raise SubgroupUtilityKernelError("invalid_report")
