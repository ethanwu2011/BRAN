"""Nested target-safe tuned-concatenation comparator for Patient Atlas V5."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from eval_soft_patient_atlas import CONCAT_EYE_DIMENSIONS
from patient_atlas_robust_group_readout import (
    GroupedDiseaseReadoutResult,
    evaluate_nested_grouped_logistic_readout,
)
from patient_atlas_disease_readout import DiseaseReadoutConfig, DiseaseReadoutError
from patient_atlas_disease_universal import BreadthEndpoint, _family_score, breadth_endpoints


POLICY_NAME = "PATIENT_ATLAS_V5_TUNED_CONCAT_PROTOCOL_V3.json"
POLICY_SCHEMA_VERSION = "patient-atlas-v5-tuned-concat-protocol-v3"
RUN_SCHEMA_VERSION = "patient-atlas-v5-tuned-concat-run-v3"
TUNED_EYE_DIMENSIONS = tuple(int(value) for value in CONCAT_EYE_DIMENSIONS)


class TunedConcatIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class TunedConcatReadoutResult:
    selected_eye_dimension: int
    selected_readout: GroupedDiseaseReadoutResult
    candidate_inner_log_losses: Mapping[int, float]
    optimizer_fallback_fit_count_by_dimension: Mapping[int, int]

    def aggregate_only_payload(self) -> dict[str, Any]:
        return {
            "selected_eye_dimension": self.selected_eye_dimension,
            "candidate_inner_log_losses": {
                str(key): value
                for key, value in sorted(self.candidate_inner_log_losses.items())
            },
            "selected_readout": self.selected_readout.aggregate_only_payload(),
            "optimizer_fallback_fit_count_by_dimension": {
                str(key): value
                for key, value in sorted(
                    self.optimizer_fallback_fit_count_by_dimension.items()
                )
            },
            "contains_patient_ids_predictions_losses_or_coordinates": False,
        }


def evaluate_nested_tuned_concat_logistic_readout(
    *,
    x_train_by_dimension: Mapping[int, np.ndarray],
    x_test_by_dimension: Mapping[int, np.ndarray],
    feature_groups_by_dimension: Mapping[int, Sequence[str]],
    y_train: np.ndarray,
    train_eligible: np.ndarray,
    inner_fold_ids: np.ndarray,
    y_test: np.ndarray,
    test_eligible: np.ndarray,
    test_patient_id_hash: str,
    config: DiseaseReadoutConfig,
    maximum_penalty_combinations: int,
    _private_test_primary_loss_out: np.ndarray | None = None,
) -> TunedConcatReadoutResult:
    """Jointly select eye dimension and group penalties on inner log loss.

    Candidate outer-test predictions are computed only to reuse the authenticated
    grouped-readout kernel; selection reads exclusively each candidate's inner
    outer-training loss.  Tests enforce invariance to outer-test labels/features.
    """

    expected = set(TUNED_EYE_DIMENSIONS)
    if (
        set(x_train_by_dimension) != expected
        or set(x_test_by_dimension) != expected
        or set(feature_groups_by_dimension) != expected
    ):
        raise TunedConcatIntegrityError("tuned concat dimension registry differs")
    candidate_results: dict[int, GroupedDiseaseReadoutResult] = {}
    candidate_sinks: dict[int, np.ndarray] = {}
    for dimension in TUNED_EYE_DIMENSIONS:
        sink = np.empty(len(np.asarray(y_test)), dtype=np.float64)
        candidate_results[dimension] = evaluate_nested_grouped_logistic_readout(
            x_train=np.asarray(x_train_by_dimension[dimension]),
            y_train=y_train,
            train_eligible=train_eligible,
            inner_fold_ids=inner_fold_ids,
            x_test=np.asarray(x_test_by_dimension[dimension]),
            y_test=y_test,
            test_eligible=test_eligible,
            feature_groups=feature_groups_by_dimension[dimension],
            unpenalized_groups=("age",),
            test_patient_id_hash=test_patient_id_hash,
            config=config,
            maximum_penalty_combinations=maximum_penalty_combinations,
            _private_test_primary_loss_out=sink,
        )
        candidate_sinks[dimension] = sink
    selected_dimension = min(
        TUNED_EYE_DIMENSIONS,
        key=lambda dimension: (
            candidate_results[dimension].inner_primary_loss,
            dimension,
        ),
    )
    if _private_test_primary_loss_out is not None:
        private = np.asarray(_private_test_primary_loss_out)
        if (
            private.shape != np.asarray(y_test).shape
            or private.dtype.kind != "f"
            or not private.flags.writeable
        ):
            raise DiseaseReadoutError("tuned concat private loss sink is malformed")
        private[:] = candidate_sinks[selected_dimension]
    return TunedConcatReadoutResult(
        selected_eye_dimension=selected_dimension,
        selected_readout=candidate_results[selected_dimension],
        candidate_inner_log_losses=MappingProxyType(
            {
                dimension: float(result.inner_primary_loss)
                for dimension, result in candidate_results.items()
            }
        ),
        optimizer_fallback_fit_count_by_dimension=MappingProxyType(
            {
                dimension: result.optimizer_fallback_fit_count
                for dimension, result in candidate_results.items()
            }
        ),
    )


def summarize_tuned_concat_losses(
    *,
    losses: np.ndarray,
    endpoints: Sequence[BreadthEndpoint],
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(endpoints):
        raise TunedConcatIntegrityError("tuned concat loss matrix differs")
    score, endpoint_scores, family_scores = _family_score(values, endpoints)
    generator = np.random.default_rng(bootstrap_seed)
    draws = np.empty(bootstrap_samples, dtype=np.float64)
    for iteration in range(bootstrap_samples):
        indices = generator.integers(0, len(values), size=len(values))
        draws[iteration] = _family_score(values[indices], endpoints)[0]
    alpha = 1.0 - confidence_level
    return {
        "primary_estimand": "equal-organ-family mean of endpoint cross-fitted log loss",
        "organ_family_balanced_mean_log_loss": score,
        "per_organ_family_mean_log_loss": family_scores,
        "per_endpoint_mean_log_loss": endpoint_scores,
        "patient_bootstrap_confidence_interval": [
            float(np.quantile(draws, alpha / 2.0)),
            float(np.quantile(draws, 1.0 - alpha / 2.0)),
        ],
        "bootstrap": {
            "unit": "patient",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
        },
        "contains_patient_losses": False,
    }


def summarize_tuned_concat_selections(
    results: Mapping[str, Sequence[TunedConcatReadoutResult]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for endpoint, values in sorted(results.items()):
        counts = {str(dimension): 0 for dimension in TUNED_EYE_DIMENSIONS}
        penalty_counts: dict[str, int] = {}
        fallback_counts = {str(dimension): 0 for dimension in TUNED_EYE_DIMENSIONS}
        for value in values:
            counts[str(value.selected_eye_dimension)] += 1
            key = json.dumps(
                dict(sorted(value.selected_readout.selected_penalties.items())),
                sort_keys=True,
                separators=(",", ":"),
            )
            penalty_counts[key] = penalty_counts.get(key, 0) + 1
            for dimension, count in value.optimizer_fallback_fit_count_by_dimension.items():
                fallback_counts[str(dimension)] += int(count)
        output[endpoint] = {
            "fold_count": len(values),
            "selected_eye_dimension_counts": counts,
            "selected_group_penalty_counts": dict(sorted(penalty_counts.items())),
            "optimizer_fallback_fit_counts_by_dimension": fallback_counts,
            "contains_coefficients_predictions_losses_or_coordinates": False,
        }
    return output


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
            raise TunedConcatIntegrityError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=_strict_object)
    if not isinstance(value, Mapping):
        raise TunedConcatIntegrityError("tuned concat protocol must be an object")
    return value


def validate_tuned_concat_protocol(
    project_root: str | Path,
    policy_path: str | Path | None = None,
) -> tuple[Mapping[str, Any], str, Mapping[str, Any], tuple[BreadthEndpoint, ...]]:
    root = Path(project_root).resolve()
    canonical = (root / POLICY_NAME).resolve()
    path = canonical if policy_path is None else Path(policy_path).resolve()
    if path != canonical:
        raise TunedConcatIntegrityError("tuned concat protocol is not canonical")
    policy = _load_json(path)
    if (
        policy.get("schema_version") != POLICY_SCHEMA_VERSION
        or policy.get("status")
        != "technical_retry_frozen_after_attempt2_nonconvergence_before_attempt3_scoring"
    ):
        raise TunedConcatIntegrityError("tuned concat protocol is not frozen")
    bindings = policy.get("bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise TunedConcatIntegrityError("tuned concat bindings are absent")
    for label, raw in bindings.items():
        if not isinstance(raw, Mapping) or set(raw) != {"file", "sha256"}:
            raise TunedConcatIntegrityError(f"malformed tuned concat binding: {label}")
        name, digest = raw["file"], raw["sha256"]
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or _sha256(root / name) != digest
        ):
            raise TunedConcatIntegrityError(f"tuned concat binding differs: {label}")
    parent = _load_json(root / bindings["compact_local_comparator_result"]["file"])
    endpoints = tuple(str(value) for value in policy.get("scope", {}).get("target_columns", ()))
    if (
        parent.get("decision", {}).get("compact_plain_concat_reference_complete")
        is not True
        or parent.get("decision", {}).get(
            "full_tuned_plain_concat_registry_comparison_complete"
        )
        is not False
        or parent.get("scope", {}).get("endpoint_count") != len(endpoints)
    ):
        raise TunedConcatIntegrityError("tuned concat parent differs")
    if policy.get("tuning") != {
        "eye_pca_dimensions": list(TUNED_EYE_DIMENSIONS),
        "dimension_and_eye_clinical_penalties_selected_jointly_by_inner_log_loss": True,
        "dimension_tie_breaker": "prefer smaller eye dimension",
        "pca_fit_scope": "outer representation-fit phase only",
        "target_defining_fields_physically_erased_before_preprocessing": True,
        "outer_test_used_for_selection_preprocessing_or_fit": False,
    }:
        raise TunedConcatIntegrityError("tuned concat registry differs")
    if policy.get("readout") != {
        "implementation": "block_regularized_binary_logistic",
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
        "penalty_groups": ["eye", "clinical"],
        "age_penalty": 0.0,
        "inner_folds_and_assignments_identical_to_v2": True,
        "maximum_penalty_combinations": 81,
        "logistic_max_iterations": 300,
        "logistic_tolerance": 1e-07,
        "newton_nonconvergence_fallback": "L-BFGS-B on the identical penalized likelihood objective",
    }:
        raise TunedConcatIntegrityError("tuned concat readout differs")
    if policy.get("aggregation") != {
        "primary_estimand": "equal-organ-family mean of endpoint cross-fitted log loss",
        "bootstrap_samples": 2000,
        "bootstrap_seed": 8675309,
        "confidence_level": 0.95,
        "comparison_to_v2_both": "descriptive because V2 patient losses were not serialized",
    }:
        raise TunedConcatIntegrityError("tuned concat aggregation differs")
    scope = policy.get("scope", {})
    if (
        scope.get("official_train_validation_only") is not True
        or scope.get("official_test_inputs_or_targets_allowed") is not False
        or scope.get("universal_exclusion_profile") != "circularity18_universal"
        or len(endpoints) != 17
    ):
        raise TunedConcatIntegrityError("tuned concat scope differs")
    if policy.get("privacy") != {
        "patient_derived_processing": "local_only",
        "patient_ids_targets_predictions_losses_embeddings_or_coordinates_serialized": False,
        "aggregate_only_result": True,
        "small_cell_threshold": 10,
    }:
        raise TunedConcatIntegrityError("tuned concat privacy differs")
    registry = _load_json(root / bindings["disease_registry"]["file"])
    return policy, _sha256(path), registry, breadth_endpoints(registry, endpoints)


__all__ = [
    "POLICY_NAME",
    "POLICY_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "TUNED_EYE_DIMENSIONS",
    "TunedConcatReadoutResult",
    "evaluate_nested_tuned_concat_logistic_readout",
    "summarize_tuned_concat_losses",
    "summarize_tuned_concat_selections",
    "validate_tuned_concat_protocol",
]
