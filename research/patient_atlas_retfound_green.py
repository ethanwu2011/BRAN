"""Frozen RETFound-Green adapter, protocol validation, and aggregate summaries.

The checkpoint is used only as a non-commercial internal benchmark.  Patient
images and embeddings remain local and no patient-level value is serialized.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from patient_atlas_disease_group_readout import GroupedDiseaseReadoutResult
from patient_atlas_disease_universal import BreadthEndpoint, _family_score, breadth_endpoints


POLICY_NAME = "PATIENT_ATLAS_V5_RETFOUND_GREEN_PROTOCOL_V1.json"
POLICY_SCHEMA_VERSION = "patient-atlas-v5-retfound-green-protocol-v1"
MODEL_ID = "vit_small_patch14_reg4_dinov2"
INPUT_SIZE = 392
EMBEDDING_DIMENSION = 384
MEAN = np.asarray((0.5, 0.5, 0.5), dtype=np.float32)
STD = np.asarray((0.5, 0.5, 0.5), dtype=np.float32)
_REQUIRED_BINDINGS = {
    "disease_registry",
    "target_source_policy",
    "fold_policy",
    "fold_audit",
    "v2_grouped_result",
    "tuned_concat_result",
    "target_loader_implementation",
    "fold_implementation",
    "target_safety_implementation",
    "cfp_selection_implementation",
    "adapter_implementation",
    "foundation_readout_implementation",
    "runner_implementation",
}


class RETFoundGreenIntegrityError(ValueError):
    """Raised when the frozen comparator contract is not exactly satisfied."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RETFoundGreenIntegrityError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=_strict_object)
    if not isinstance(value, Mapping):
        raise RETFoundGreenIntegrityError("RETFound-Green protocol must be an object")
    return value


def require_checkpoint(path: str | Path, expected_sha256: str) -> Path:
    artifact = Path(path).expanduser().absolute()
    if not artifact.is_file():
        raise RETFoundGreenIntegrityError("RETFound-Green checkpoint is absent")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(value not in "0123456789abcdef" for value in expected_sha256)
        or sha256_file(artifact) != expected_sha256
    ):
        raise RETFoundGreenIntegrityError("RETFound-Green checkpoint hash differs")
    if artifact.stat().st_size > 128 * 1024 * 1024:
        raise RETFoundGreenIntegrityError("RETFound-Green checkpoint exceeds size cap")
    return artifact


def normalize_pixels(images: np.ndarray) -> np.ndarray:
    """Normalize already resized RGB uint8 images using the released constants."""

    values = np.asarray(images)
    if (
        values.ndim != 4
        or values.shape[1:] != (INPUT_SIZE, INPUT_SIZE, 3)
        or values.dtype != np.uint8
    ):
        raise RETFoundGreenIntegrityError(
            "RETFound-Green pixels must be uint8 [images,392,392,3] RGB"
        )
    normalized = values.astype(np.float32) / np.float32(255.0)
    normalized = (normalized - MEAN[None, None, None]) / STD[None, None, None]
    normalized = np.ascontiguousarray(normalized.transpose(0, 3, 1, 2))
    if not np.isfinite(normalized).all():
        raise RETFoundGreenIntegrityError("RETFound-Green preprocessing is non-finite")
    return normalized


def load_model(
    *, checkpoint_path: str | Path, checkpoint_sha256: str, device: str = "cpu"
):
    """Load the authenticated state dict without enabling arbitrary pickle objects."""

    artifact = require_checkpoint(checkpoint_path, checkpoint_sha256)
    try:
        import timm
        import torch
    except ImportError as error:  # pragma: no cover - runtime-specific
        raise RETFoundGreenIntegrityError("RETFound-Green runtime is unavailable") from error
    if device not in {"cpu", "mps", "cuda"}:
        raise RETFoundGreenIntegrityError("RETFound-Green device is invalid")
    previous_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        try:
            state = torch.load(str(artifact), map_location="cpu", weights_only=True)
        except Exception as error:
            raise RETFoundGreenIntegrityError(
                "RETFound-Green weights-only checkpoint load failed"
            ) from error
        if not isinstance(state, Mapping) or len(state) != 175:
            raise RETFoundGreenIntegrityError("RETFound-Green state schema differs")
        if any(not isinstance(key, str) for key in state):
            raise RETFoundGreenIntegrityError("RETFound-Green state key differs")
        if any(not isinstance(value, torch.Tensor) for value in state.values()):
            raise RETFoundGreenIntegrityError("RETFound-Green state contains non-tensors")
        model = timm.create_model(
            MODEL_ID,
            pretrained=False,
            img_size=(INPUT_SIZE, INPUT_SIZE),
            num_classes=0,
        )
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RETFoundGreenIntegrityError("RETFound-Green state does not load strictly")
        model.global_pool = "avg"
    finally:
        if previous_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_offline
    if int(getattr(model, "num_features", -1)) != EMBEDDING_DIMENSION:
        raise RETFoundGreenIntegrityError("RETFound-Green embedding dimension differs")
    model = model.to(torch.device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode_pixels(model: Any, images: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    """Encode one deterministic image batch with no gradient state."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime-specific
        raise RETFoundGreenIntegrityError("PyTorch is unavailable") from error
    batch = torch.from_numpy(normalize_pixels(images)).to(torch.device(device))
    with torch.inference_mode():
        encoded = model(batch)
    if isinstance(encoded, (tuple, list)):
        encoded = encoded[0]
    values = np.asarray(encoded.detach().cpu().numpy(), dtype=np.float32)
    if values.shape != (len(images), EMBEDDING_DIMENSION):
        raise RETFoundGreenIntegrityError("RETFound-Green returned an unexpected shape")
    if not np.isfinite(values).all():
        raise RETFoundGreenIntegrityError("RETFound-Green returned non-finite values")
    return values


def validate_protocol(
    project_root: str | Path,
    policy_path: str | Path | None = None,
) -> tuple[Mapping[str, Any], str, Mapping[str, Any], tuple[BreadthEndpoint, ...]]:
    root = Path(project_root).resolve()
    canonical = (root / POLICY_NAME).resolve()
    path = canonical if policy_path is None else Path(policy_path).resolve()
    if path != canonical:
        raise RETFoundGreenIntegrityError("RETFound-Green protocol is not canonical")
    policy = _load_json(path)
    if (
        policy.get("schema_version") != POLICY_SCHEMA_VERSION
        or policy.get("status")
        != "frozen_after_checkpoint_acquisition_before_patient_encoding_or_scoring"
    ):
        raise RETFoundGreenIntegrityError("RETFound-Green protocol is not frozen")
    bindings = policy.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != _REQUIRED_BINDINGS:
        raise RETFoundGreenIntegrityError("RETFound-Green bindings differ")
    for label, raw in bindings.items():
        if not isinstance(raw, Mapping) or set(raw) != {"file", "sha256"}:
            raise RETFoundGreenIntegrityError(f"malformed binding: {label}")
        name, digest = raw.get("file"), raw.get("sha256")
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise RETFoundGreenIntegrityError(f"invalid binding path: {label}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RETFoundGreenIntegrityError(f"invalid binding hash: {label}")
        if sha256_file(root / name) != digest:
            raise RETFoundGreenIntegrityError(f"binding hash differs: {label}")
    parent = _load_json(root / str(bindings["v2_grouped_result"]["file"]))
    tuned = _load_json(root / str(bindings["tuned_concat_result"]["file"]))
    registry = _load_json(root / str(bindings["disease_registry"]["file"]))
    scope = policy.get("scope", {})
    targets = scope.get("target_columns")
    if (
        parent.get("decision", {}).get("paired_information_gate_passed") is not True
        or parent.get("privacy", {}).get("aggregate_only") is not True
        or tuned.get("status") != "full_nested_tuned_concat_registry_complete"
        or tuned.get("privacy", {}).get("aggregate_only") is not True
        or scope.get("official_train_validation_only") is not True
        or scope.get("official_test_images_or_targets_allowed") is not False
        or scope.get("universal_exclusion_profile") != "circularity18_universal"
        or targets != parent.get("scope", {}).get("endpoint_columns")
        or policy.get("outer_fold_assignment_sha256")
        != parent.get("outer_fold_assignment_sha256")
    ):
        raise RETFoundGreenIntegrityError("RETFound-Green scope differs")
    artifact = policy.get("artifact")
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "release_url",
        "release_tag",
        "model_id",
        "embedding_dimension",
        "input_size",
        "mean",
        "std",
        "license",
        "license_scope",
        "pretraining_overlap_status",
    }:
        raise RETFoundGreenIntegrityError("RETFound-Green artifact contract differs")
    if (
        artifact.get("checkpoint_size_bytes") != 87396676
        or artifact.get("model_id") != MODEL_ID
        or artifact.get("embedding_dimension") != EMBEDDING_DIMENSION
        or artifact.get("input_size") != INPUT_SIZE
        or artifact.get("mean") != [0.5, 0.5, 0.5]
        or artifact.get("std") != [0.5, 0.5, 0.5]
        or artifact.get("license") != "Justin's Custom Non-Commercial Research Licence (CNCRL)"
        or artifact.get("license_scope") != "benchmark_only_noncommercial_research"
        or artifact.get("pretraining_overlap_status") != "unresolved_not_proven_absent"
    ):
        raise RETFoundGreenIntegrityError("RETFound-Green artifact metadata differs")
    checkpoint = require_checkpoint(
        str(artifact["checkpoint"]), str(artifact["checkpoint_sha256"])
    )
    if checkpoint.stat().st_size != int(artifact["checkpoint_size_bytes"]):
        raise RETFoundGreenIntegrityError("RETFound-Green checkpoint size differs")
    if policy.get("representation") != {
        "encoder_weights_frozen": True,
        "outcomes_received_by_encoder": False,
        "outer_test_used_for_encoder_or_pooling_fit": False,
        "preprocessing": "RGB full-frame bilinear resize to 392 with antialiasing, float scale, then channelwise mean/std 0.5; no augmentation",
        "patient_pooling": "unweighted mean of valid image embeddings",
        "patient_embedding_file_allowed": False,
    }:
        raise RETFoundGreenIntegrityError("RETFound-Green representation policy differs")
    if policy.get("readout") != {
        "task": "binary",
        "implementation": "standardized_intercept_unpenalized_ridge_logistic",
        "optimizer": "scipy_lbfgsb_then_newtoncg_same_objective_v2",
        "penalty_grid": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
        "inner_folds_identical_to_v2": True,
        "age_appended_and_unpenalized": True,
        "outer_test_used_for_selection_or_standardization": False,
    }:
        raise RETFoundGreenIntegrityError("RETFound-Green readout policy differs")
    if policy.get("uncertainty") != {
        "bootstrap_unit": "patient",
        "bootstrap_samples": 2000,
        "bootstrap_seed": 314159,
        "confidence_level": 0.95,
        "paired_contrast_to_v2_available": False,
    }:
        raise RETFoundGreenIntegrityError("RETFound-Green uncertainty policy differs")
    if policy.get("privacy") != {
        "patient_derived_processing": "local_only",
        "patient_embeddings_written_to_disk": False,
        "patient_ids_targets_predictions_losses_embeddings_or_coordinates_serialized": False,
        "aggregate_only_result": True,
        "small_cell_threshold": 10,
    }:
        raise RETFoundGreenIntegrityError("RETFound-Green privacy policy differs")
    return policy, sha256_file(path), registry, breadth_endpoints(registry, targets)


def summarize_losses(
    *,
    losses: np.ndarray,
    endpoints: Sequence[BreadthEndpoint],
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(endpoints):
        raise RETFoundGreenIntegrityError("RETFound-Green losses are malformed")
    score, endpoint_scores, family_scores = _family_score(values, endpoints)
    generator = np.random.default_rng(bootstrap_seed)
    samples = np.empty(bootstrap_samples, dtype=np.float64)
    for iteration in range(bootstrap_samples):
        indices = generator.integers(0, len(values), size=len(values))
        samples[iteration] = _family_score(values[indices], endpoints)[0]
    alpha = 1.0 - confidence_level
    return {
        "comparator": "retfound_green",
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


def summarize_readouts(
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
    "EMBEDDING_DIMENSION",
    "INPUT_SIZE",
    "MEAN",
    "MODEL_ID",
    "POLICY_NAME",
    "RETFoundGreenIntegrityError",
    "STD",
    "encode_pixels",
    "load_model",
    "normalize_pixels",
    "require_checkpoint",
    "sha256_file",
    "summarize_losses",
    "summarize_readouts",
    "validate_protocol",
]
