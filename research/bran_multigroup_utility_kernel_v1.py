"""Closed CGM utility evaluator for a reference-coded K=2..4 group design."""
from __future__ import annotations

import numpy as np
from sklearn.exceptions import ConvergenceWarning

import bran_subgroup_utility_kernel_v1 as base


SCHEMA_VERSION = "bran_multigroup_utility_kernel_v1"
ARM_NAMES = base.ARM_NAMES
CONTRAST_NAMES = base.CONTRAST_NAMES
CONTRAST_ARMS = base.CONTRAST_ARMS


def _group_designs(designs):
    if not isinstance(designs, dict) or set(designs) != {"severity", "severity_group"}:
        raise base.SubgroupUtilityKernelError("invalid_designs")
    severity = base._matrix(designs["severity"], "severity")
    grouped = base._matrix(designs["severity_group"], "severity_group", len(severity))
    added = grouped.shape[1] - severity.shape[1]
    if len(severity) == 0 or added not in (1, 2, 3) or not np.array_equal(grouped[:, :severity.shape[1]], severity):
        raise base.SubgroupUtilityKernelError("invalid_multigroup_design")
    onehot = grouped[:, severity.shape[1]:]
    if (not np.isin(onehot, (0.0, 1.0)).all() or np.any(onehot.sum(axis=1) > 1.0)
            or not all(np.any(onehot[:, column] == 1.0) for column in range(added))):
        raise base.SubgroupUtilityKernelError("invalid_multigroup_onehot")
    return severity, grouped


def _closed_status(status, support):
    return {"schema_version": SCHEMA_VERSION, "status": status, "clinical_claim": False,
            "novel_claim": False, "external_claim": False, "support": support}


def fit_and_evaluate(designs, y, observed, outer, membership, ids=None):
    """Fit frozen heads with severity plus reference-coded one-hot groups."""
    severity, severity_group = _group_designs(designs)
    rows = len(severity); target = base._target(y, rows)
    observed_mask = base._strict_bool(observed, "observed", rows)
    membership_mask = base._strict_bool(membership, "membership", rows)
    outer_fold = base._outer_folds(outer, rows); base._unique_ids(ids, rows)
    eligible = observed_mask & membership_mask; support = base._support_flags(eligible, outer_fold)
    if not all(support.values()): return _closed_status("unsupported_target_support", support)
    discovery = eligible & np.isin(outer_fold, base.DISCOVERY_FOLDS)
    validation = eligible & (outer_fold == base.VALIDATION_FOLD)
    replication = eligible & (outer_fold == base.REPLICATION_FOLD)
    try:
        discovery_y = target[discovery]; discovery_outer = outer_fold[discovery]
        iqr = max(float(np.percentile(discovery_y, 75) - np.percentile(discovery_y, 25)), 1e-6)
        if not np.isfinite(iqr): raise FloatingPointError("nonfinite_iqr")
        validation_count = int(validation.sum()); predictions = {}
        for design, prefix in ((severity, "severity"), (severity_group, "severity_group")):
            score = np.concatenate((design[validation], design[replication]))
            ridge, trees = base._fit_predictions(design[discovery], discovery_y, discovery_outer, score)
            predictions[prefix + "_ridge"] = ridge; predictions[prefix + "_extra_trees"] = trees
        predictions["mean"] = np.full(validation_count + int(replication.sum()), float(discovery_y.mean()))
        arms, contrasts = {}, {}
        for name, rows_mask, start, stop in (("validation", validation, 0, validation_count),
                                             ("replication", replication, validation_count, None)):
            split_predictions = {key: value[start:stop] for key, value in predictions.items()}
            arms[name], contrasts[name] = base._evaluation_report(target[rows_mask], split_predictions, iqr)
    except (ConvergenceWarning, FloatingPointError, ValueError, OverflowError):
        return _closed_status("numerical_failure", support)
    report = {"schema_version": SCHEMA_VERSION, "status": "ok", "clinical_claim": False,
              "novel_claim": False, "external_claim": False, "support": support,
              "arms": {arm: {split: arms[split][arm] for split in ("validation", "replication")} for arm in ARM_NAMES},
              "contrasts": {name: {split: contrasts[split][name] for split in ("validation", "replication")} for name in CONTRAST_NAMES}}
    validate_report(report); return report


def validate_report(report):
    if not isinstance(report, dict): raise base.SubgroupUtilityKernelError("invalid_report")
    basic = {"schema_version", "status", "clinical_claim", "novel_claim", "external_claim", "support"}
    status = report.get("status"); expected = basic | ({"arms", "contrasts"} if status == "ok" else set())
    if set(report) != expected or report.get("schema_version") != SCHEMA_VERSION or status not in {"ok", "unsupported_target_support", "numerical_failure"}:
        raise base.SubgroupUtilityKernelError("invalid_report")
    if any(report[key] is not False for key in ("clinical_claim", "novel_claim", "external_claim")):
        raise base.SubgroupUtilityKernelError("invalid_report")
    support = report["support"]
    if not isinstance(support, dict) or set(support) != {"discovery_at_least_80", "validation_at_least_40", "replication_at_least_40", "each_discovery_inner_fold_at_least_10"} or any(type(x) is not bool for x in support.values()):
        raise base.SubgroupUtilityKernelError("invalid_report")
    if status == "unsupported_target_support":
        if all(support.values()): raise base.SubgroupUtilityKernelError("invalid_report")
        return
    if not all(support.values()): raise base.SubgroupUtilityKernelError("invalid_report")
    if status == "numerical_failure": return
    if set(report["arms"]) != set(ARM_NAMES) or set(report["contrasts"]) != set(CONTRAST_NAMES):
        raise base.SubgroupUtilityKernelError("invalid_report")
    for arm in report["arms"].values():
        if set(arm) != {"validation", "replication"}: raise base.SubgroupUtilityKernelError("invalid_report")
        for metrics in arm.values():
            if set(metrics) != {"mae_mg_dl", "rmse_mg_dl", "normalized_mse", "r2"} or not all(base._finite_number(metrics[key]) and float(metrics[key]) >= 0 for key in ("mae_mg_dl", "rmse_mg_dl", "normalized_mse")):
                raise base.SubgroupUtilityKernelError("invalid_report")
            if metrics["r2"] is not None and (not base._finite_number(metrics["r2"]) or float(metrics["r2"]) > 1.0):
                raise base.SubgroupUtilityKernelError("invalid_report")
            if float(metrics["mae_mg_dl"]) > float(metrics["rmse_mg_dl"]) + 1e-12 * max(1.0, abs(float(metrics["rmse_mg_dl"]))):
                raise base.SubgroupUtilityKernelError("invalid_report")
    for name, contrast in report["contrasts"].items():
        if set(contrast) != {"validation", "replication"}: raise base.SubgroupUtilityKernelError("invalid_report")
        first, second = CONTRAST_ARMS[name]
        for split, values in contrast.items():
            if set(values) != {"normalized_mse_difference", "bootstrap_marginal_95ci", "bootstrap_resamples"}:
                raise base.SubgroupUtilityKernelError("invalid_report")
            ci = values["bootstrap_marginal_95ci"]
            if (not base._finite_number(values["normalized_mse_difference"]) or not isinstance(ci, list) or len(ci) != 2
                    or not all(base._finite_number(bound) for bound in ci) or float(ci[0]) > float(ci[1])
                    or type(values["bootstrap_resamples"]) is not int or values["bootstrap_resamples"] != base.BOOTSTRAP_RESAMPLES):
                raise base.SubgroupUtilityKernelError("invalid_report")
            expected = float(report["arms"][first][split]["normalized_mse"]) - float(report["arms"][second][split]["normalized_mse"])
            if float(values["normalized_mse_difference"]) != expected:
                raise base.SubgroupUtilityKernelError("invalid_report")
