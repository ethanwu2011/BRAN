"""Row-free contracts for the information-matched frozen-FM benchmark.

This module deliberately has no project-data imports.  It accepts local arrays
only from a caller that has already authenticated them, never serializes those
arrays, and returns closed aggregate metrics only.  The production runner is
responsible for binding its callbacks to the prespecified fixed-probe helper.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ENDPOINTS = 26
OUTER_FOLDS = 5
BOOTSTRAP_DRAWS = 1000
MINIMUM_VALID_DRAWS = 900
BOOTSTRAP_SEED = 91501
FAMILY_ALPHA = .05
LOCK_PATH = "/private/tmp/bran_retinal_extraction_v1.lock"

BRAN_ARMS = ("bran_combined_state", "bran_clinical_state", "bran_retinal_state")
FM_BASES = ("retfound_green", "visionfm_last4", "dinov3_generic")
# Both budgets remain visible: representation plus age and representation plus
# the identical eligible clinical values/masks plus age used by BRAN.
FM_AGE_ARMS = tuple(name + "_age" for name in FM_BASES)
FM_CLINICAL_AGE_ARMS = tuple(name + "_clinical_age" for name in FM_BASES)
LABRADOR_ARMS = ("labrador_clinical_age", "labrador_clinical_retinal_age")
FM_ARMS = FM_AGE_ARMS + FM_CLINICAL_AGE_ARMS
CONTROL_ARMS = ("blood_age", "raw_clinical_age", "raw_clinical_retinal_age")
ARMS = BRAN_ARMS + FM_ARMS + LABRADOR_ARMS + CONTROL_ARMS
# These contrasts are declared before scores.  The raw controls remain visible,
# but are not substituted into this predeclared clinical+FM family.
PRIMARY_COMPARATORS = ("raw_clinical_age",) + FM_CLINICAL_AGE_ARMS + ("labrador_clinical_retinal_age",)

FLAGS = {
    "patient_level_output_emitted": False,
    "bootstrap_draws_emitted": False,
    "encoder_training": False,
    "retinal_extraction": False,
    "external_source_scores": False,
    "clinical_superiority_automatic_flag": False,
    "native_heads_called_same_readout_budget": False,
    "official_test_used": False,
}


class ContractError(ValueError):
    """Raised for a closed benchmark contract violation."""


def require(ok: bool, message: str = "information_matched_fm_contract_rejected") -> None:
    if not ok:
        raise ContractError(message)


def canonical_sha256(value: Any) -> str:
    """Digest row-free manifest structures deterministically."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_source_binding(binding: Mapping[str, Any]) -> None:
    """Validate independently generated, row-free source-byte receipts.

    The receipt has no historical AUROC reference.  Its input digest is shared
    by every arm, which prevents a different clinical/mask/age context from
    being silently attached to a frozen FM representation.
    """
    expected = {
        "schema", "endpoint_names", "outer_fold_sha256", "inner_fold_sha256",
        "row_order_sha256", "shared_clinical_values_sha256", "shared_clinical_masks_sha256",
        "shared_eligible_masks_sha256", "shared_age_sha256", "shared_input_sha256",
        "labrador_input_sha256", "bran_checkpoints_sha256", "fm_cache_manifest_sha256", "representation_sha256", "arm_input_sha256", "arm_provenance",
        "source_files_sha256", "transitive_code_sha256", "canonical_source_sha256", "parent_receipts_sha256",
        "all_source_bytes_authenticated", "retinal_encoders_frozen",
        "retinal_extraction_performed", "historical_score_equality_required",
    }
    require(type(binding) is dict and set(binding) == expected, "source_binding_schema_invalid")
    require(binding["schema"] == "bran-information-matched-fm-source-binding-v2")
    names = binding["endpoint_names"]
    require(isinstance(names, list) and len(names) == ENDPOINTS and len(set(names)) == ENDPOINTS
            and all(isinstance(x, str) and x for x in names), "endpoint_binding_invalid")
    for key in ("outer_fold_sha256", "row_order_sha256", "shared_clinical_values_sha256",
                "shared_clinical_masks_sha256", "shared_eligible_masks_sha256", "shared_age_sha256",
                "shared_input_sha256", "labrador_input_sha256", "bran_checkpoints_sha256", "fm_cache_manifest_sha256"):
        require(_sha(binding[key]), "source_binding_digest_invalid")
    require(isinstance(binding["inner_fold_sha256"], list) and len(binding["inner_fold_sha256"]) == OUTER_FOLDS
            and all(_sha(x) for x in binding["inner_fold_sha256"]), "inner_fold_binding_invalid")
    require(isinstance(binding["representation_sha256"], dict)
            and set(binding["representation_sha256"]) == set(ARMS), "representation_binding_invalid")
    require(all(_sha(x) for x in binding["representation_sha256"].values()))
    require(isinstance(binding["arm_input_sha256"], dict) and set(binding["arm_input_sha256"]) == set(ARMS)
            and all(value == binding["shared_input_sha256"] for value in binding["arm_input_sha256"].values()),
            "arm_input_binding_invalid")
    provenance = binding["arm_provenance"]
    require(isinstance(provenance, dict) and set(provenance) == set(ARMS)
            and all(isinstance(row, dict) and set(row) == {"upstream_supervision", "training_exposure"}
                    and all(isinstance(value, str) and value for value in row.values()) for row in provenance.values()),
            "arm_provenance_invalid")
    require(isinstance(binding["source_files_sha256"], dict) and binding["source_files_sha256"]
            and all(isinstance(k, str) and k and _sha(v) for k, v in binding["source_files_sha256"].items()),
            "source_bytes_binding_invalid")
    for key in ("transitive_code_sha256", "canonical_source_sha256", "parent_receipts_sha256"):
        require(isinstance(binding[key], dict) and binding[key]
                and all(isinstance(name, str) and name and _sha(digest)
                        for name, digest in binding[key].items()),
                "transitive_source_binding_invalid")
    require(binding["all_source_bytes_authenticated"] is True
            and binding["retinal_encoders_frozen"] is True
            and binding["retinal_extraction_performed"] is False
            and binding["historical_score_equality_required"] is False,
            "source_binding_policy_invalid")


def matched_input_digest(values_sha: str, masks_sha: str, eligible_sha: str, age_sha: str) -> str:
    """One common clinical-values/masks/eligibility/age binding for every arm."""
    for item in (values_sha, masks_sha, eligible_sha, age_sha):
        require(_sha(item), "matched_input_digest_invalid")
    return canonical_sha256({"clinical_values": values_sha, "clinical_masks": masks_sha,
                             "eligible_masks": eligible_sha, "age": age_sha})


def validate_runtime_context(context: Mapping[str, Any], binding: Mapping[str, Any]) -> tuple[str, ...]:
    """Check only shape and identity relations; caller keeps arrays local."""
    validate_source_binding(binding)
    required = {"folds", "labels", "observed", "matched_support", "inner_folds", "design_factory", "input_sha256",
                "outer_fold_sha256", "row_order_sha256"}
    require(type(context) is dict and set(context) == required, "runtime_context_schema_invalid")
    require(context["input_sha256"] == binding["shared_input_sha256"], "runtime_input_digest_mismatch")
    require(context["outer_fold_sha256"] == binding["outer_fold_sha256"]
            and context["row_order_sha256"] == binding["row_order_sha256"], "runtime_identity_digest_mismatch")
    folds = np.asarray(context["folds"])
    n = len(folds)
    require(folds.shape == (n,) and folds.dtype.kind in "iu" and set(folds.tolist()) == set(range(OUTER_FOLDS)),
            "outer_fold_context_invalid")
    names = tuple(binding["endpoint_names"])
    require(set(context["labels"]) == set(context["observed"]) == set(context["matched_support"]) == set(names), "endpoint_context_invalid")
    require(isinstance(context["inner_folds"], Sequence) and len(context["inner_folds"]) == OUTER_FOLDS,
            "inner_fold_context_invalid")
    require(callable(context["design_factory"]), "fold_design_factory_invalid")
    for endpoint in names:
        y = np.asarray(context["labels"][endpoint]); observed = np.asarray(context["observed"][endpoint])
        support = np.asarray(context["matched_support"][endpoint])
        require(y.shape == observed.shape == (n,) and observed.dtype == np.dtype(bool)
                and np.isfinite(y[observed]).all() and np.isin(y[observed], (0, 1)).all(),
                "label_context_invalid")
        require(support.shape == (n,) and support.dtype == np.dtype(bool) and np.all(support <= observed),
                "matched_support_invalid")
    for fold in range(OUTER_FOLDS):
        train = np.flatnonzero(folds != fold)
        inner = np.asarray(context["inner_folds"][fold])
        require(inner.shape == (len(train),) and inner.dtype.kind in "iu" and set(inner.tolist()) == set(range(OUTER_FOLDS)),
                "inner_fold_assignment_invalid")
    return names


def _validate_fold_designs(designs: Mapping[str, Any], n: int) -> None:
    require(isinstance(designs, Mapping) and set(designs) == set(ARMS), "fold_design_arm_invalid")
    for design in designs.values():
        value = np.asarray(design)
        require(value.ndim == 2 and value.shape[0] == n and value.shape[1] > 0 and np.isfinite(value).all(),
                "design_values_invalid")


def fit_all_readouts(context: Mapping[str, Any], binding: Mapping[str, Any],
                     fit_predict_fixed: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                     progress: Callable[[int, str, int, int], None] | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Use the supplied fixed C=1 probe identically for all arms."""
    names = validate_runtime_context(context, binding)
    folds = np.asarray(context["folds"]); n = len(folds)
    predictions = {endpoint: {arm: np.full(n, np.nan) for arm in ARMS} for endpoint in names}
    completed = 0; budget = OUTER_FOLDS * len(ARMS) * len(names)
    for fold in range(OUTER_FOLDS):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        designs = context["design_factory"](fold)
        _validate_fold_designs(designs, n)
        for arm in ARMS:
            design = np.asarray(designs[arm], float)
            for endpoint in names:
                y = np.asarray(context["labels"][endpoint])
                matched = np.asarray(context["matched_support"][endpoint], bool)
                prediction = np.asarray(fit_predict_fixed(design[train], y[train], matched[train], design[test]), float)
                require(prediction.shape == (len(test),) and np.isfinite(prediction).all()
                        and np.all((prediction >= 0) & (prediction <= 1)), "readout_prediction_invalid")
                predictions[endpoint][arm][test] = prediction
                completed += 1
            if progress is not None:
                progress(fold, arm, completed, budget)
    return predictions


def _auc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    y, p, w = np.asarray(y), np.asarray(p), np.asarray(w, float)
    positive, negative = y == 1, y == 0
    wp, wn = float(w[positive].sum()), float(w[negative].sum())
    require(wp > 0 and wn > 0, "auc_class_support_invalid")
    greater = (p[positive, None] > p[None, negative]).astype(float)
    ties = (p[positive, None] == p[None, negative]).astype(float)
    return float(((greater + .5 * ties) * w[positive, None] * w[None, negative]).sum() / (wp * wn))


def fold_weighted_auc(y: np.ndarray, p: np.ndarray, mask: np.ndarray, folds: np.ndarray,
                      counts: np.ndarray | None = None) -> float:
    """Local synthetic default; production supplies the historical helper."""
    y, p, mask, folds = np.asarray(y), np.asarray(p), np.asarray(mask, bool), np.asarray(folds)
    values, weights = [], []
    for fold in range(OUTER_FOLDS):
        index = mask & (folds == fold)
        w = np.ones(index.sum()) if counts is None else np.asarray(counts)[index]
        values.append(_auc(y[index], p[index], w)); weights.append(float(w.sum()))
    return float(np.average(values, weights=weights))


def bootstrap_fold_weighted_auc(y: np.ndarray, p: np.ndarray, mask: np.ndarray, folds: np.ndarray,
                                counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts)
    require(counts.ndim == 2 and counts.shape[0] == BOOTSTRAP_DRAWS and counts.shape[1] == len(folds),
            "bootstrap_count_shape_invalid")
    draws = np.full(BOOTSTRAP_DRAWS, np.nan)
    for draw in range(BOOTSTRAP_DRAWS):
        try:
            draws[draw] = fold_weighted_auc(y, p, mask, folds, counts[draw])
        except ContractError:
            pass
    return draws


def validate_bootstrap_counts(folds: np.ndarray, counts: np.ndarray) -> None:
    folds, counts = np.asarray(folds), np.asarray(counts)
    require(counts.shape == (BOOTSTRAP_DRAWS, len(folds)) and counts.dtype.kind in "iu" and np.all(counts >= 0),
            "bootstrap_counts_invalid")
    for fold in range(OUTER_FOLDS):
        require(np.all(counts[:, folds == fold].sum(axis=1) == np.sum(folds == fold)), "bootstrap_not_within_fold")


def _metric(point: float, draws: np.ndarray, delta: bool = False) -> dict[str, Any]:
    good = np.isfinite(draws)
    require(good.sum() >= MINIMUM_VALID_DRAWS, "insufficient_valid_paired_bootstrap_draws")
    return {"delta" if delta else "auroc": float(point),
            "ci95": [float(x) for x in np.percentile(draws[good], [2.5, 97.5])]}


def summarize(predictions: Mapping[str, Mapping[str, np.ndarray]], labels: Mapping[str, np.ndarray],
              observed: Mapping[str, np.ndarray], folds: np.ndarray, counts: np.ndarray,
              endpoint_names: Sequence[str], *, auc_fn: Callable[..., float] = fold_weighted_auc,
              bootstrap_auc_fn: Callable[..., np.ndarray] = bootstrap_fold_weighted_auc) -> dict[str, Any]:
    """Closed absolute and paired endpoint/macro AUROC estimates on shared draws."""
    names = tuple(endpoint_names); folds = np.asarray(folds); validate_bootstrap_counts(folds, counts)
    require(len(names) == ENDPOINTS and len(set(names)) == ENDPOINTS
            and set(predictions) == set(labels) == set(observed) == set(names), "metric_endpoint_schema_invalid")
    endpoint_results: dict[str, Any] = {}; arm_points = {a: [] for a in ARMS}; arm_draws = {a: [] for a in ARMS}
    for endpoint in names:
        y, mask = np.asarray(labels[endpoint]), np.asarray(observed[endpoint], bool)
        require(y.shape == mask.shape == folds.shape and np.isin(y[mask], (0, 1)).all())
        for fold in range(OUTER_FOLDS):
            require(set(np.unique(y[mask & (folds == fold)])) == {0, 1}, "endpoint_fold_class_missing")
        require(all(np.sum(mask & (y == value)) >= 20 for value in (0, 1)), "endpoint_class_count_below_20")
        require(set(predictions[endpoint]) == set(ARMS), "metric_arm_schema_invalid")
        points, draws = {}, {}
        for arm in ARMS:
            p = np.asarray(predictions[endpoint][arm], float)
            require(p.shape == folds.shape and np.isfinite(p[mask]).all() and np.all((p[mask] >= 0) & (p[mask] <= 1)),
                    "prediction_support_invalid")
            points[arm] = float(auc_fn(y, p, mask, folds))
            draws[arm] = np.asarray(bootstrap_auc_fn(y, p, mask, folds, counts), float)
            require(draws[arm].shape == (BOOTSTRAP_DRAWS,), "bootstrap_metric_shape_invalid")
        common = np.logical_and.reduce([np.isfinite(draws[a]) for a in ARMS])
        require(common.sum() >= MINIMUM_VALID_DRAWS, "endpoint_shared_draws_below_900")
        for arm in ARMS:
            draws[arm][~common] = np.nan
            arm_points[arm].append(points[arm]); arm_draws[arm].append(draws[arm])
        endpoint_results[endpoint] = {
            "arms": {a: _metric(points[a], draws[a]) for a in ARMS},
            "paired_differences": {a: _metric(points[BRAN_ARMS[0]] - points[a], draws[BRAN_ARMS[0]] - draws[a], True)
                                   for a in ARMS[1:]},
        }
    macro_draws = {a: np.mean(np.stack(arm_draws[a]), axis=0) for a in ARMS}
    common = np.logical_and.reduce([np.isfinite(macro_draws[a]) for a in ARMS])
    require(common.sum() >= MINIMUM_VALID_DRAWS, "macro_shared_draws_below_900")
    macro_points = {a: float(np.mean(arm_points[a])) for a in ARMS}
    family_lower = {arm: float(np.percentile((macro_draws[BRAN_ARMS[0]] - macro_draws[arm])[common],
                                             100 * FAMILY_ALPHA / len(PRIMARY_COMPARATORS)))
                    for arm in PRIMARY_COMPARATORS}
    result = {
        "schema": "bran-information-matched-fm-metrics-v2",
        "endpoints": endpoint_results,
        "macro": {
            "arms": {a: _metric(macro_points[a], macro_draws[a][common]) for a in ARMS},
            "paired_differences": {a: _metric(macro_points[BRAN_ARMS[0]] - macro_points[a],
                                                (macro_draws[BRAN_ARMS[0]] - macro_draws[a])[common], True)
                                   for a in ARMS[1:]},
        },
        "primary_comparisons": list(PRIMARY_COMPARATORS),
        "primary_family": {"one_sided_bonferroni_alpha": FAMILY_ALPHA,
                           "one_sided_bonferroni_lower": family_lower,
                           "automatic_superiority_flag": False},
        "inference": "paired_fixed_fit_within_fold_participant_bootstrap_1000_shared_draws",
        "endpoint_evidence": "exploratory_marginal_not_multiplicity_adjusted",
        **FLAGS,
    }
    validate_result(result, names)
    return result


def validate_result(result: Mapping[str, Any], endpoint_names: Sequence[str]) -> None:
    keys = {"schema", "endpoints", "macro", "primary_comparisons", "primary_family", "inference", "endpoint_evidence"} | set(FLAGS)
    require(type(result) is dict and set(result) == keys and result["schema"] == "bran-information-matched-fm-metrics-v2")
    require(result["primary_comparisons"] == list(PRIMARY_COMPARATORS)
            and result["inference"] == "paired_fixed_fit_within_fold_participant_bootstrap_1000_shared_draws"
            and result["endpoint_evidence"] == "exploratory_marginal_not_multiplicity_adjusted")
    require(all(result[key] is value for key, value in FLAGS.items()))
    family = result["primary_family"]
    require(type(family) is dict and set(family) == {"one_sided_bonferroni_alpha", "one_sided_bonferroni_lower", "automatic_superiority_flag"}
            and family["one_sided_bonferroni_alpha"] == FAMILY_ALPHA and family["automatic_superiority_flag"] is False
            and isinstance(family["one_sided_bonferroni_lower"], dict)
            and set(family["one_sided_bonferroni_lower"]) == set(PRIMARY_COMPARATORS)
            and all(math.isfinite(value) and -1 <= value <= 1 for value in family["one_sided_bonferroni_lower"].values()),
            "primary_family_invalid")
    require(set(result["endpoints"]) == set(endpoint_names) and len(endpoint_names) == ENDPOINTS)
    for row in list(result["endpoints"].values()) + [result["macro"]]:
        require(set(row) == {"arms", "paired_differences"} and set(row["arms"]) == set(ARMS)
                and set(row["paired_differences"]) == set(ARMS[1:]), "result_row_schema_invalid")
        for arm, metric in row["arms"].items():
            require(set(metric) == {"auroc", "ci95"} and math.isfinite(metric["auroc"]) and 0 <= metric["auroc"] <= 1
                    and isinstance(metric["ci95"], list) and len(metric["ci95"]) == 2
                    and all(math.isfinite(x) and 0 <= x <= 1 for x in metric["ci95"]) and metric["ci95"][0] <= metric["ci95"][1],
                    "absolute_metric_invalid")
        for arm, metric in row["paired_differences"].items():
            require(set(metric) == {"delta", "ci95"} and math.isfinite(metric["delta"]) and -1 <= metric["delta"] <= 1
                    and abs(metric["delta"] - (row["arms"][BRAN_ARMS[0]]["auroc"] - row["arms"][arm]["auroc"])) < 1e-12
                    and isinstance(metric["ci95"], list) and len(metric["ci95"]) == 2
                    and all(math.isfinite(x) and -1 <= x <= 1 for x in metric["ci95"]) and metric["ci95"][0] <= metric["ci95"][1],
                    "paired_metric_invalid")
