"""Data-agnostic kernel for the V5 universal-mask disease breadth run."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from patient_atlas_disease_folds import make_disease_fold_map
from patient_atlas_disease_readout import DiseaseReadoutResult
from patient_atlas_disease_targets import DiseaseTargetMatrix
from patient_atlas_preprocessing import hash_json


POLICY_NAME = "PATIENT_ATLAS_V5_DISEASE_UNIVERSAL_PROTOCOL_V1.json"
POLICY_SCHEMA_VERSION = "patient-atlas-v5-disease-universal-protocol-v1"
ATLAS_ARMS = ("both_atlas", "eye_atlas", "blood_clinical_atlas")
_REQUIRED_BINDINGS = {
    "disease_registry",
    "target_source_policy",
    "target_availability_audit",
    "fold_policy",
    "fold_audit",
    "representation_protocol",
    "target_loader_implementation",
    "fold_implementation",
    "target_safety_implementation",
    "readout_implementation",
    "representation_implementation",
    "universal_implementation",
    "universal_runner",
}


class DiseaseUniversalIntegrityError(ValueError):
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
            raise DiseaseUniversalIntegrityError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=_strict_object)
    if not isinstance(value, Mapping):
        raise DiseaseUniversalIntegrityError("universal disease contract must be an object")
    return value


@dataclass(frozen=True)
class BreadthEndpoint:
    endpoint_id: str
    target_column: str
    organ_family: str
    label_quality: str


def breadth_endpoints(
    registry: Mapping[str, Any],
    target_columns: Sequence[str],
) -> tuple[BreadthEndpoint, ...]:
    required = tuple(str(value) for value in target_columns)
    if not required or len(required) != len(set(required)):
        raise DiseaseUniversalIntegrityError("breadth target columns must be unique")
    by_column: dict[str, BreadthEndpoint] = {}
    ordered: list[BreadthEndpoint] = []
    for raw in registry.get("breadth_outcomes", []):
        column = str(raw.get("target_column", ""))
        if column not in required:
            continue
        endpoint = BreadthEndpoint(
            endpoint_id=str(raw.get("id", "")),
            target_column=column,
            organ_family=str(raw.get("organ_family", "")),
            label_quality=str(raw.get("label_quality", "")),
        )
        if (
            not endpoint.endpoint_id
            or not endpoint.organ_family
            or not endpoint.label_quality
            or column in by_column
        ):
            raise DiseaseUniversalIntegrityError("breadth endpoint registry is malformed")
        by_column[column] = endpoint
        ordered.append(endpoint)
    if set(by_column) != set(required):
        raise DiseaseUniversalIntegrityError("breadth targets and registry differ")
    return tuple(ordered)


def validate_universal_protocol(
    project_root: str | Path,
    policy_path: str | Path | None = None,
) -> tuple[Mapping[str, Any], str, Mapping[str, Any], tuple[BreadthEndpoint, ...]]:
    root = Path(project_root).resolve()
    canonical = (root / POLICY_NAME).resolve()
    path = canonical if policy_path is None else Path(policy_path).resolve()
    if path != canonical:
        raise DiseaseUniversalIntegrityError(
            "universal disease protocol must be the canonical project artifact"
        )
    policy = _load_json(path)
    if (
        policy.get("schema_version") != POLICY_SCHEMA_VERSION
        or policy.get("status") != "frozen_before_any_universal_mask_disease_model_score"
    ):
        raise DiseaseUniversalIntegrityError("universal disease protocol is not frozen")
    bindings = policy.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != _REQUIRED_BINDINGS:
        raise DiseaseUniversalIntegrityError("universal disease bindings differ")
    for label, raw_binding in bindings.items():
        if not isinstance(raw_binding, Mapping) or set(raw_binding) != {"file", "sha256"}:
            raise DiseaseUniversalIntegrityError(f"malformed universal binding: {label}")
        name = raw_binding.get("file")
        digest = raw_binding.get("sha256")
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise DiseaseUniversalIntegrityError(f"invalid universal binding path: {label}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise DiseaseUniversalIntegrityError(f"invalid universal binding hash: {label}")
        if _sha256((root / name).resolve()) != digest:
            raise DiseaseUniversalIntegrityError(f"universal binding hash differs: {label}")

    registry = _load_json(root / str(bindings["disease_registry"]["file"]))
    scope = policy.get("scope", {})
    target_columns = scope.get("target_columns")
    if (
        scope.get("lane") != "universal_mask_breadth"
        or scope.get("official_train_validation_only") is not True
        or scope.get("official_test_inputs_or_targets_allowed") is not False
        or scope.get("universal_exclusion_profile") != "circularity18_universal"
        or not isinstance(target_columns, list)
        or len(target_columns) != 17
        or len(target_columns) != len(set(target_columns))
        or scope.get("blocked_target_columns") != ["albuminuria"]
    ):
        raise DiseaseUniversalIntegrityError("universal disease scope differs")
    endpoints = breadth_endpoints(registry, target_columns)

    fold_audit = _load_json(root / str(bindings["fold_audit"]["file"]))
    passing = set(
        fold_audit.get("execution_decision", {}).get(
            "fold_gate_passing_target_columns", []
        )
    )
    if (
        fold_audit.get("status")
        != "canonical_fold_map_frozen_and_endpoint_eligibility_complete"
        or not set(target_columns).issubset(passing)
        or policy.get("outer_fold_assignment_sha256")
        != fold_audit.get("fold_audit", {}).get("assignment_sha256")
    ):
        raise DiseaseUniversalIntegrityError("universal disease fold binding differs")

    representation = policy.get("representation", {})
    if representation != {
        "architecture": "retinal_capacity_expanded_structured_probabilistic_patient_atlas_v5",
        "latent_partition": {
            "shared": 32,
            "eye_private": 160,
            "clinical_private": 64,
            "total": 256,
        },
        "outcome_coordinates_excluding_age": 288,
        "availability_arms": list(ATLAS_ARMS),
        "identical_age_context_appended_to_every_readout": True,
        "interaction_enabled": False,
        "external_blood_anchor_enabled": False,
        "beta": 0.01,
        "group_shrinkage_rate": 1e-06,
        "target_values_received_by_representation_fit": False,
        "same_fold_model_transforms_outer_train_and_outer_test": True,
        "cross_fold_coordinates_pooled": False,
    }:
        raise DiseaseUniversalIntegrityError("universal representation contract differs")
    training = policy.get("training", {})
    if training != {
        "fit_fraction": 0.7,
        "validation_fraction": 0.15,
        "calibration_fraction": 0.15,
        "batch_size": 96,
        "maximum_steps": 6000,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "validation_interval": 250,
        "early_stopping_patience": 8,
        "base_training_seed": 20260828,
    }:
        raise DiseaseUniversalIntegrityError("universal training contract differs")
    readout = policy.get("readout", {})
    if readout != {
        "task": "binary_logistic",
        "inner_folds": 5,
        "inner_assignment": "same deterministic multilabel algorithm rebuilt within each outer-training set",
        "inner_seed": 1701,
        "inner_salt_rule": "patient-atlas-v5-disease-inner-v1:<outer-fold>",
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
        "selection_loss": "inner_validation_log_loss",
        "exact_tie_break": "larger_penalty",
    }:
        raise DiseaseUniversalIntegrityError("universal readout contract differs")
    aggregation = policy.get("aggregation", {})
    if aggregation != {
        "endpoint_score": "cross_fitted_mean_log_loss",
        "within_organ_family": "equal_weight_mean_across_endpoints",
        "across_organ_families": "equal_weight_mean",
        "paired_contrasts": ["eye_minus_both", "clinical_minus_both"],
        "positive_contrast_favors_both": True,
        "multiplicity": "one_sided_centered_patient_bootstrap_max_statistic",
        "bootstrap_samples": 2000,
        "bootstrap_seed": 8675309,
        "confidence_level": 0.95,
    }:
        raise DiseaseUniversalIntegrityError("universal aggregation contract differs")
    privacy = policy.get("privacy", {})
    if privacy != {
        "patient_derived_processing": "local_only",
        "patient_ids_targets_predictions_losses_embeddings_and_coordinates_serialized": False,
        "aggregate_only_result": True,
        "small_cell_threshold": 10,
    }:
        raise DiseaseUniversalIntegrityError("universal privacy contract differs")
    return policy, _sha256(path), registry, endpoints


def subset_targets(
    targets: DiseaseTargetMatrix,
    indices: np.ndarray,
    patient_ids: Sequence[str],
) -> DiseaseTargetMatrix:
    indices = np.asarray(indices, dtype=np.int64)
    ids = tuple(str(value) for value in patient_ids)
    if indices.ndim != 1 or len(indices) != len(ids):
        raise DiseaseUniversalIntegrityError("target subset indices and IDs differ")
    return DiseaseTargetMatrix(
        columns=targets.columns,
        tasks=targets.tasks,
        execution_statuses=targets.execution_statuses,
        values=np.asarray(targets.values[indices], dtype=np.float64),
        observed_mask=np.asarray(targets.observed_mask[indices], dtype=bool),
        patient_id_order_sha256=hash_json(list(ids)),
        source_policy_sha256=targets.source_policy_sha256,
        source_hashes=targets.source_hashes,
    )


def make_inner_fold_ids(
    *,
    outer_fold: int,
    patient_ids: Sequence[str],
    site_ids: Sequence[str],
    targets: DiseaseTargetMatrix,
    outer_fold_policy: Mapping[str, Any],
) -> tuple[np.ndarray, str]:
    if outer_fold < 0:
        raise DiseaseUniversalIntegrityError("outer fold must be nonnegative")
    inner_policy = {
        "split": {
            "n_folds": 5,
            "seed": 1701,
            "salt": f"patient-atlas-v5-disease-inner-v1:{outer_fold}",
        },
        "balancing": dict(outer_fold_policy["balancing"]),
    }
    fold_map = make_disease_fold_map(
        patient_ids=patient_ids,
        site_ids=site_ids,
        targets=targets,
        policy=inner_policy,
    )
    return fold_map.assignments_for(patient_ids), fold_map.assignment_sha256


def _family_score(
    losses: np.ndarray,
    endpoints: Sequence[BreadthEndpoint],
) -> tuple[float, dict[str, float], dict[str, float]]:
    matrix = np.asarray(losses, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(endpoints):
        raise DiseaseUniversalIntegrityError("breadth losses have the wrong shape")
    endpoint_scores: dict[str, float] = {}
    by_family: dict[str, list[float]] = {}
    for index, endpoint in enumerate(endpoints):
        observed = np.isfinite(matrix[:, index])
        if not bool(observed.any()):
            raise DiseaseUniversalIntegrityError(
                f"breadth endpoint has no scored rows: {endpoint.target_column}"
            )
        score = float(np.mean(matrix[observed, index]))
        endpoint_scores[endpoint.target_column] = score
        by_family.setdefault(endpoint.organ_family, []).append(score)
    family_scores = {
        family: float(np.mean(values)) for family, values in sorted(by_family.items())
    }
    return float(np.mean(list(family_scores.values()))), endpoint_scores, family_scores


def summarize_breadth_losses(
    *,
    losses_by_arm: Mapping[str, np.ndarray],
    endpoints: Sequence[BreadthEndpoint],
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    if set(losses_by_arm) != set(ATLAS_ARMS):
        raise DiseaseUniversalIntegrityError("breadth loss arms differ")
    shapes = {np.asarray(value).shape for value in losses_by_arm.values()}
    if len(shapes) != 1:
        raise DiseaseUniversalIntegrityError("breadth arm losses are not aligned")
    shape = next(iter(shapes))
    if len(shape) != 2 or shape[1] != len(endpoints):
        raise DiseaseUniversalIntegrityError("breadth arm loss shape differs")
    if bootstrap_samples < 100 or not 0.5 < confidence_level < 1.0:
        raise DiseaseUniversalIntegrityError("breadth bootstrap configuration is invalid")
    point: dict[str, float] = {}
    arms: dict[str, Any] = {}
    for arm in ATLAS_ARMS:
        score, endpoint_scores, family_scores = _family_score(
            np.asarray(losses_by_arm[arm]), endpoints
        )
        point[arm] = score
        arms[arm] = {
            "organ_family_balanced_mean_log_loss": score,
            "per_organ_family_mean_log_loss": family_scores,
            "per_endpoint_mean_log_loss": endpoint_scores,
        }

    generator = np.random.default_rng(int(bootstrap_seed))
    bootstrap = {
        arm: np.empty(bootstrap_samples, dtype=np.float64) for arm in ATLAS_ARMS
    }
    for iteration in range(bootstrap_samples):
        indices = generator.integers(0, shape[0], size=shape[0])
        for arm in ATLAS_ARMS:
            bootstrap[arm][iteration] = _family_score(
                np.asarray(losses_by_arm[arm])[indices], endpoints
            )[0]
    alpha = 1.0 - float(confidence_level)
    for arm in ATLAS_ARMS:
        arms[arm]["patient_bootstrap_confidence_interval"] = [
            float(np.quantile(bootstrap[arm], alpha / 2.0)),
            float(np.quantile(bootstrap[arm], 1.0 - alpha / 2.0)),
        ]

    comparators = {
        "eye_minus_both": "eye_atlas",
        "clinical_minus_both": "blood_clinical_atlas",
    }
    contrast_points = {
        name: point[arm] - point["both_atlas"] for name, arm in comparators.items()
    }
    contrast_samples = {
        name: bootstrap[arm] - bootstrap["both_atlas"]
        for name, arm in comparators.items()
    }
    centered_max = np.max(
        np.column_stack(
            [
                contrast_samples[name] - contrast_points[name]
                for name in comparators
            ]
        ),
        axis=1,
    )
    critical = float(np.quantile(centered_max, 1.0 - alpha))
    contrasts: dict[str, Any] = {}
    for name in comparators:
        value = float(contrast_points[name])
        samples = contrast_samples[name]
        lower = float(2.0 * value - np.quantile(samples, 1.0 - alpha / 2.0))
        upper = float(2.0 * value - np.quantile(samples, alpha / 2.0))
        simultaneous_lower = value - critical
        contrasts[name] = {
            "organ_family_balanced_log_loss_difference": value,
            "confidence_interval": [lower, upper],
            "simultaneous_one_sided_lower_confidence_bound": simultaneous_lower,
            "positive_favors_both_atlas": True,
            "multiplicity_control": "centered patient-bootstrap max statistic over two contrasts",
            "superiority_passed": bool(simultaneous_lower > 0.0),
        }
    return {
        "primary_estimand": "equal-organ-family mean of endpoint cross-fitted log loss",
        "endpoint_count": len(endpoints),
        "organ_family_count": len({endpoint.organ_family for endpoint in endpoints}),
        "arms": arms,
        "paired_contrasts": contrasts,
        "paired_information_gate_passed": bool(
            all(value["superiority_passed"] for value in contrasts.values())
        ),
        "bootstrap": {
            "unit": "patient",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
        },
        "contains_patient_losses": False,
    }


def summarize_readout_results(
    results: Mapping[str, Mapping[str, Sequence[DiseaseReadoutResult]]],
) -> dict[str, Any]:
    if set(results) != set(ATLAS_ARMS):
        raise DiseaseUniversalIntegrityError("readout result arms differ")
    summary: dict[str, Any] = {}
    for arm in ATLAS_ARMS:
        arm_summary: dict[str, Any] = {}
        for target, fold_results in sorted(results[arm].items()):
            values = tuple(fold_results)
            if not values:
                raise DiseaseUniversalIntegrityError("target has no fold readout results")
            task = values[0].task
            if any(value.task != task for value in values):
                raise DiseaseUniversalIntegrityError("target task changes across folds")
            penalty_counts: dict[str, int] = {}
            for value in values:
                key = str(value.selected_penalty)
                penalty_counts[key] = penalty_counts.get(key, 0) + 1
            metric_names = tuple(values[0].metrics)
            metrics: dict[str, float | None] = {}
            weights = np.asarray(
                [value.test_eligible_count for value in values], dtype=np.float64
            )
            for metric in metric_names:
                available = [
                    (index, float(value.metrics[metric]))
                    for index, value in enumerate(values)
                    if value.metrics[metric] is not None
                ]
                if not available:
                    metrics[metric] = None
                else:
                    indices = np.asarray([index for index, _ in available], dtype=np.int64)
                    metric_values = np.asarray([value for _, value in available])
                    metrics[metric] = float(
                        np.average(metric_values, weights=weights[indices])
                    )
            arm_summary[target] = {
                "fold_count": len(values),
                "test_eligible_count": int(weights.sum()),
                "selected_penalty_counts": dict(sorted(penalty_counts.items())),
                "test_count_weighted_fold_metrics": metrics,
                "secondary_metric_pooling": "test-count-weighted mean of fold metrics; not pooled predictions",
            }
        summary[arm] = arm_summary
    return summary


__all__ = [
    "ATLAS_ARMS",
    "BreadthEndpoint",
    "DiseaseUniversalIntegrityError",
    "POLICY_NAME",
    "POLICY_SCHEMA_VERSION",
    "breadth_endpoints",
    "make_inner_fold_ids",
    "subset_targets",
    "summarize_breadth_losses",
    "summarize_readout_results",
    "validate_universal_protocol",
]
