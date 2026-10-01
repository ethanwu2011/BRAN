"""Protocol validation and aggregate summaries for frozen public comparators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from patient_atlas_disease_group_readout import GroupedDiseaseReadoutResult
from patient_atlas_disease_universal import BreadthEndpoint, _family_score, breadth_endpoints
from patient_atlas_foundation_adapters import require_artifact


POLICY_NAME = "PATIENT_ATLAS_V5_FOUNDATION_COMPARATORS_PROTOCOL_V1.json"
POLICY_SCHEMA_VERSION = "patient-atlas-v5-foundation-comparators-protocol-v1"
FOUNDATION_COMPARATORS = ("dinov3_generic", "labrador")
_REQUIRED_BINDINGS = {
    "disease_registry",
    "target_source_policy",
    "fold_policy",
    "fold_audit",
    "unit_reconciliation",
    "v2_grouped_result",
    "target_loader_implementation",
    "fold_implementation",
    "target_safety_implementation",
    "foundation_adapter_implementation",
    "labrador_worker_implementation",
    "foundation_readout_implementation",
    "foundation_comparator_implementation",
    "foundation_comparator_runner",
}


class FoundationComparatorIntegrityError(ValueError):
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
            raise FoundationComparatorIntegrityError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=_strict_object)
    if not isinstance(value, Mapping):
        raise FoundationComparatorIntegrityError("foundation protocol must be an object")
    return value


def validate_foundation_comparator_protocol(
    project_root: str | Path,
    *,
    comparator: str,
    policy_path: str | Path | None = None,
) -> tuple[Mapping[str, Any], str, Mapping[str, Any], tuple[BreadthEndpoint, ...]]:
    root = Path(project_root).resolve()
    canonical = (root / POLICY_NAME).resolve()
    path = canonical if policy_path is None else Path(policy_path).resolve()
    if path != canonical or comparator not in FOUNDATION_COMPARATORS:
        raise FoundationComparatorIntegrityError("foundation protocol or comparator is not canonical")
    policy = _load_json(path)
    if (
        policy.get("schema_version") != POLICY_SCHEMA_VERSION
        or policy.get("status")
        != "frozen_before_new_dinov3_or_labrador_patient_encoding_or_scoring"
    ):
        raise FoundationComparatorIntegrityError("foundation protocol is not frozen")
    bindings = policy.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != _REQUIRED_BINDINGS:
        raise FoundationComparatorIntegrityError("foundation protocol bindings differ")
    for label, raw in bindings.items():
        if not isinstance(raw, Mapping) or set(raw) != {"file", "sha256"}:
            raise FoundationComparatorIntegrityError(f"malformed foundation binding: {label}")
        name, digest = raw.get("file"), raw.get("sha256")
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise FoundationComparatorIntegrityError(f"invalid foundation binding path: {label}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise FoundationComparatorIntegrityError(f"invalid foundation binding hash: {label}")
        if _sha256((root / name).resolve()) != digest:
            raise FoundationComparatorIntegrityError(f"foundation binding hash differs: {label}")

    parent = _load_json(root / str(bindings["v2_grouped_result"]["file"]))
    registry = _load_json(root / str(bindings["disease_registry"]["file"]))
    scope = policy.get("scope", {})
    targets = scope.get("target_columns")
    if (
        parent.get("decision", {}).get("paired_information_gate_passed") is not True
        or parent.get("privacy", {}).get("aggregate_only") is not True
        or scope.get("official_train_validation_only") is not True
        or scope.get("official_test_images_or_targets_allowed") is not False
        or scope.get("universal_exclusion_profile") != "circularity18_universal"
        or scope.get("comparators") != list(FOUNDATION_COMPARATORS)
        or targets != parent.get("scope", {}).get("endpoint_columns")
        or policy.get("outer_fold_assignment_sha256")
        != parent.get("outer_fold_assignment_sha256")
    ):
        raise FoundationComparatorIntegrityError("foundation comparator scope differs")
    if policy.get("readout") != {
        "task": "binary",
        "implementation": "standardized_intercept_unpenalized_ridge_logistic",
        "optimizer": "scipy_lbfgsb_then_newtoncg_same_objective_v2",
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
        "inner_folds_and_assignments_identical_to_v2": True,
        "age_appended_and_unpenalized": True,
        "outer_test_used_for_selection_or_standardization": False,
    }:
        raise FoundationComparatorIntegrityError("foundation readout policy differs")
    if policy.get("representation") != {
        "encoder_weights_frozen": True,
        "outcomes_received_by_encoder": False,
        "outer_test_used_for_encoder_or_pooling_fit": False,
        "dinov3_preprocessing": "RGB full-frame bicubic resize to 256 then released ImageNet normalization; no augmentation",
        "dinov3_patient_pooling": "unweighted mean of valid image embeddings",
        "labrador_preprocessing": "physical universal-target and unauthenticated-unit masking, exact released codebook mapping, MIMIC-IV eCDF transform, observed-token left packing",
        "labrador_patient_pooling": "unweighted mean of final hidden states over observed tokens; padding excluded",
        "patient_embedding_file_allowed": False,
    }:
        raise FoundationComparatorIntegrityError("foundation representation policy differs")
    if policy.get("uncertainty") != {
        "bootstrap_unit": "patient",
        "bootstrap_samples": 2000,
        "bootstrap_seed": 314159,
        "confidence_level": 0.95,
        "paired_contrast_to_v2_available": False,
    }:
        raise FoundationComparatorIntegrityError("foundation uncertainty policy differs")
    if policy.get("privacy") != {
        "patient_derived_processing": "local_only",
        "patient_embeddings_written_to_disk": False,
        "patient_ids_targets_predictions_losses_embeddings_or_coordinates_serialized": False,
        "aggregate_only_result": True,
        "small_cell_threshold": 10,
    }:
        raise FoundationComparatorIntegrityError("foundation privacy policy differs")
    comparator_policy = policy.get("comparator_artifacts", {}).get(comparator)
    if not isinstance(comparator_policy, Mapping):
        raise FoundationComparatorIntegrityError("selected foundation artifact policy is absent")
    if comparator == "dinov3_generic":
        if set(comparator_policy) != {
            "checkpoint",
            "checkpoint_sha256",
            "embedding_dimension",
            "input_size",
            "interpolation",
            "mean",
            "std",
            "patient_pooling",
            "pretraining_overlap_status",
        }:
            raise FoundationComparatorIntegrityError("DINOv3 artifact policy differs")
        if (
            comparator_policy.get("embedding_dimension") != 384
            or comparator_policy.get("input_size") != 256
            or comparator_policy.get("interpolation") != "bicubic_full_frame_resize"
            or comparator_policy.get("mean") != [0.485, 0.456, 0.406]
            or comparator_policy.get("std") != [0.229, 0.224, 0.225]
            or comparator_policy.get("patient_pooling")
            != "unweighted_mean_of_valid_images"
            or comparator_policy.get("pretraining_overlap_status")
            != "unresolved_not_proven_absent"
        ):
            raise FoundationComparatorIntegrityError("DINOv3 preprocessing differs")
        require_artifact(
            comparator_policy["checkpoint"], comparator_policy["checkpoint_sha256"]
        )
    else:
        required = {
            "model_root",
            "saved_model_sha256",
            "variables_data_sha256",
            "variables_index_sha256",
            "codebook",
            "codebook_sha256",
            "ecdf",
            "ecdf_sha256",
            "tensorflow_python",
            "embedding_dimension",
            "sequence_length",
            "tokenization",
            "patient_pooling",
            "unit_policy",
        }
        if set(comparator_policy) != required:
            raise FoundationComparatorIntegrityError("Labrador artifact policy differs")
        if (
            comparator_policy.get("embedding_dimension") != 1024
            or comparator_policy.get("sequence_length") != 64
            or comparator_policy.get("tokenization")
            != "exact_mimic_itemid_map_then_released_mimic4_ecdf_searchsorted_left"
            or comparator_policy.get("patient_pooling")
            != "mean_final_hidden_state_over_observed_tokens"
            or comparator_policy.get("unit_policy")
            != "use_only_policy_visible_canonical-unit-authorized_labs"
        ):
            raise FoundationComparatorIntegrityError("Labrador preprocessing differs")
        model_root = Path(str(comparator_policy["model_root"]))
        require_artifact(
            model_root / "saved_model.pb", comparator_policy["saved_model_sha256"]
        )
        require_artifact(
            model_root / "variables" / "variables.data-00000-of-00001",
            comparator_policy["variables_data_sha256"],
        )
        require_artifact(
            model_root / "variables" / "variables.index",
            comparator_policy["variables_index_sha256"],
        )
        require_artifact(comparator_policy["codebook"], comparator_policy["codebook_sha256"])
        require_artifact(comparator_policy["ecdf"], comparator_policy["ecdf_sha256"])
        if not Path(str(comparator_policy["tensorflow_python"])).is_file():
            raise FoundationComparatorIntegrityError("Labrador TensorFlow runtime is absent")
    endpoints = breadth_endpoints(registry, targets)
    return policy, _sha256(path), registry, endpoints


def summarize_foundation_losses(
    *,
    comparator: str,
    losses: np.ndarray,
    endpoints: Sequence[BreadthEndpoint],
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    if comparator not in FOUNDATION_COMPARATORS:
        raise FoundationComparatorIntegrityError("unknown foundation comparator")
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(endpoints):
        raise FoundationComparatorIntegrityError("foundation losses are malformed")
    score, endpoint_scores, family_scores = _family_score(values, endpoints)
    generator = np.random.default_rng(bootstrap_seed)
    samples = np.empty(bootstrap_samples, dtype=np.float64)
    for iteration in range(bootstrap_samples):
        indices = generator.integers(0, len(values), size=len(values))
        samples[iteration] = _family_score(values[indices], endpoints)[0]
    alpha = 1.0 - confidence_level
    return {
        "comparator": comparator,
        "primary_estimand": "equal-organ-family mean of endpoint cross-fitted log loss",
        "organ_family_balanced_mean_log_loss": score,
        "patient_bootstrap_confidence_interval": [
            float(np.quantile(samples, alpha / 2.0)),
            float(np.quantile(samples, 1.0 - alpha / 2.0)),
        ],
        "per_organ_family_mean_log_loss": family_scores,
        "per_endpoint_mean_log_loss": endpoint_scores,
        "bootstrap": {
            "unit": "patient",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
        },
        "contains_patient_losses": False,
    }


def summarize_foundation_readouts(
    results: Mapping[str, Sequence[GroupedDiseaseReadoutResult]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for target, raw in sorted(results.items()):
        values = tuple(raw)
        penalties: dict[str, int] = {}
        for value in values:
            key = json.dumps(
                dict(sorted(value.selected_penalties.items())),
                sort_keys=True,
                separators=(",", ":"),
            )
            penalties[key] = penalties.get(key, 0) + 1
        output[target] = {
            "fold_count": len(values),
            "test_eligible_count": int(sum(value.test_eligible_count for value in values)),
            "selected_penalty_counts": dict(sorted(penalties.items())),
            "contains_coefficients_predictions_or_losses": False,
        }
    return output


__all__ = [
    "FOUNDATION_COMPARATORS",
    "FoundationComparatorIntegrityError",
    "POLICY_NAME",
    "summarize_foundation_losses",
    "summarize_foundation_readouts",
    "validate_foundation_comparator_protocol",
]
