"""Versioned grouped-readout sensitivity for the universal disease benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from patient_atlas_disease_group_readout import GroupedDiseaseReadoutResult
from patient_atlas_disease_universal import (
    ATLAS_ARMS,
    BreadthEndpoint,
    breadth_endpoints,
)


POLICY_NAME = "PATIENT_ATLAS_V5_DISEASE_UNIVERSAL_GROUPED_PROTOCOL_V2.json"
POLICY_SCHEMA_VERSION = "patient-atlas-v5-disease-universal-grouped-protocol-v2"
_REQUIRED_BINDINGS = {
    "disease_registry",
    "target_source_policy",
    "fold_policy",
    "fold_audit",
    "representation_protocol",
    "v1_universal_result",
    "target_loader_implementation",
    "fold_implementation",
    "target_safety_implementation",
    "group_readout_implementation",
    "representation_implementation",
    "universal_v1_implementation",
    "universal_v2_implementation",
    "universal_v2_runner",
}


class DiseaseUniversalV2IntegrityError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DiseaseUniversalV2IntegrityError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=_strict_object)
    if not isinstance(value, Mapping):
        raise DiseaseUniversalV2IntegrityError("V2 universal protocol must be an object")
    return value


def structured_feature_groups() -> tuple[str, ...]:
    groups = (
        ("eye_evidence",) * 32
        + ("clinical_evidence",) * 32
        + ("eye_evidence",) * 160
        + ("clinical_evidence",) * 64
        + ("age",)
    )
    if len(groups) != 289:
        raise RuntimeError("V2 grouped readout width differs")
    return groups


def validate_universal_v2_protocol(
    project_root: str | Path,
    policy_path: str | Path | None = None,
) -> tuple[Mapping[str, Any], str, Mapping[str, Any], tuple[BreadthEndpoint, ...]]:
    root = Path(project_root).resolve()
    canonical = (root / POLICY_NAME).resolve()
    path = canonical if policy_path is None else Path(policy_path).resolve()
    if path != canonical:
        raise DiseaseUniversalV2IntegrityError("V2 universal protocol is not canonical")
    policy = _load_json(path)
    if (
        policy.get("schema_version") != POLICY_SCHEMA_VERSION
        or policy.get("status")
        != "frozen_after_v1_global_penalty_result_before_any_grouped_readout_score"
    ):
        raise DiseaseUniversalV2IntegrityError("V2 universal protocol is not frozen")
    bindings = policy.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != _REQUIRED_BINDINGS:
        raise DiseaseUniversalV2IntegrityError("V2 universal bindings differ")
    for label, raw in bindings.items():
        if not isinstance(raw, Mapping) or set(raw) != {"file", "sha256"}:
            raise DiseaseUniversalV2IntegrityError(f"malformed V2 binding: {label}")
        name, digest = raw.get("file"), raw.get("sha256")
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise DiseaseUniversalV2IntegrityError(f"invalid V2 binding path: {label}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise DiseaseUniversalV2IntegrityError(f"invalid V2 binding hash: {label}")
        if _sha256((root / name).resolve()) != digest:
            raise DiseaseUniversalV2IntegrityError(f"V2 binding hash differs: {label}")

    parent = _load_json(root / str(bindings["v1_universal_result"]["file"]))
    if (
        parent.get("status") != "universal_mask_internal_disease_breadth_attempt_complete"
        or parent.get("decision", {}).get("both_beats_eye_with_multiplicity_control")
        is not True
        or parent.get("decision", {}).get(
            "both_beats_clinical_with_multiplicity_control"
        )
        is not False
        or parent.get("privacy", {}).get("aggregate_only") is not True
    ):
        raise DiseaseUniversalV2IntegrityError("V1 parent result differs")
    registry = _load_json(root / str(bindings["disease_registry"]["file"]))
    scope = policy.get("scope", {})
    target_columns = scope.get("target_columns")
    if (
        scope.get("lane") != "universal_mask_breadth_grouped_readout_sensitivity"
        or scope.get("universal_exclusion_profile") != "circularity18_universal"
        or scope.get("official_train_validation_only") is not True
        or scope.get("official_test_inputs_or_targets_allowed") is not False
        or not isinstance(target_columns, list)
        or target_columns != parent.get("scope", {}).get("endpoint_columns")
        or scope.get("availability_arms") != list(ATLAS_ARMS)
    ):
        raise DiseaseUniversalV2IntegrityError("V2 scope differs")
    endpoints = breadth_endpoints(registry, target_columns)
    if policy.get("outer_fold_assignment_sha256") != parent.get(
        "outer_fold_assignment_sha256"
    ):
        raise DiseaseUniversalV2IntegrityError("V2 outer folds differ from V1")
    representation = policy.get("representation", {})
    if representation != {
        "architecture_and_training_identical_to_v1": True,
        "v1_model_state_sha256_must_match_per_fold": True,
        "vector_width_including_age": 289,
        "eye_owned_coordinate_ranges": ["0:32", "64:224"],
        "clinical_owned_coordinate_ranges": ["32:64", "224:288"],
        "age_coordinate_range": "288:289",
        "target_values_received_by_representation_fit": False,
        "same_fold_model_transforms_outer_train_and_outer_test": True,
        "cross_fold_coordinates_pooled": False,
    }:
        raise DiseaseUniversalV2IntegrityError("V2 representation contract differs")
    readout = policy.get("readout", {})
    if readout != {
        "task": "block_regularized_binary_logistic",
        "inner_folds": 5,
        "inner_assignment_identical_to_v1": True,
        "penalty_groups": ["eye_evidence", "clinical_evidence"],
        "age_penalty": 0.0,
        "penalty_grid": [
            0.001,
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
            1000.0,
            10000.0,
            100000.0,
        ],
        "both_arm_penalty_grid": "full Cartesian product, 81 combinations",
        "inactive_missing_modality_block": "fixed at maximum penalty and excluded from grid",
        "selection_loss": "inner_validation_log_loss",
        "exact_tie_break": "larger penalties in sorted group order",
        "maximum_penalty_combinations": 81,
    }:
        raise DiseaseUniversalV2IntegrityError("V2 grouped readout contract differs")
    if policy.get("aggregation") != {
        "identical_to_v1": True,
        "endpoint_score": "cross_fitted_mean_log_loss",
        "within_organ_family": "equal_weight_mean_across_endpoints",
        "across_organ_families": "equal_weight_mean",
        "paired_contrasts": ["eye_minus_both", "clinical_minus_both"],
        "multiplicity": "one_sided_centered_patient_bootstrap_max_statistic",
        "bootstrap_samples": 2000,
        "bootstrap_seed": 8675309,
        "confidence_level": 0.95,
    }:
        raise DiseaseUniversalV2IntegrityError("V2 aggregation differs")
    if policy.get("privacy") != {
        "patient_derived_processing": "local_only",
        "patient_ids_targets_predictions_losses_embeddings_and_coordinates_serialized": False,
        "aggregate_only_result": True,
        "small_cell_threshold": 10,
    }:
        raise DiseaseUniversalV2IntegrityError("V2 privacy contract differs")
    return policy, _sha256(path), registry, endpoints


def summarize_grouped_readout_results(
    results: Mapping[str, Mapping[str, Sequence[GroupedDiseaseReadoutResult]]],
) -> dict[str, Any]:
    if set(results) != set(ATLAS_ARMS):
        raise DiseaseUniversalV2IntegrityError("V2 result arms differ")
    summary: dict[str, Any] = {}
    for arm in ATLAS_ARMS:
        arm_summary: dict[str, Any] = {}
        for target, fold_results in sorted(results[arm].items()):
            values = tuple(fold_results)
            if not values:
                raise DiseaseUniversalV2IntegrityError("V2 target has no fold results")
            penalty_selections: dict[str, int] = {}
            for value in values:
                key = json.dumps(
                    dict(sorted(value.selected_penalties.items())),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                penalty_selections[key] = penalty_selections.get(key, 0) + 1
            weights = np.asarray(
                [value.test_eligible_count for value in values], dtype=np.float64
            )
            metrics: dict[str, float | None] = {}
            for metric in values[0].metrics:
                available = [
                    (index, float(value.metrics[metric]))
                    for index, value in enumerate(values)
                    if value.metrics[metric] is not None
                ]
                if not available:
                    metrics[metric] = None
                else:
                    indices = np.asarray([index for index, _ in available], dtype=np.int64)
                    metrics[metric] = float(
                        np.average(
                            np.asarray([item for _, item in available]),
                            weights=weights[indices],
                        )
                    )
            arm_summary[target] = {
                "fold_count": len(values),
                "test_eligible_count": int(weights.sum()),
                "selected_group_penalty_counts": dict(sorted(penalty_selections.items())),
                "test_count_weighted_fold_metrics": metrics,
                "contains_coefficients_predictions_or_losses": False,
            }
        summary[arm] = arm_summary
    return summary


__all__ = [
    "POLICY_NAME",
    "POLICY_SCHEMA_VERSION",
    "DiseaseUniversalV2IntegrityError",
    "structured_feature_groups",
    "summarize_grouped_readout_results",
    "validate_universal_v2_protocol",
]
