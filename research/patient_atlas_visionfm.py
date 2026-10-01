"""Frozen VisionFM-fundus adapter, protocol validation, and aggregate summaries."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any, Mapping, Sequence

import numpy as np

from patient_atlas_disease_group_readout import GroupedDiseaseReadoutResult
from patient_atlas_disease_universal import BreadthEndpoint, _family_score, breadth_endpoints


POLICY_NAME = "PATIENT_ATLAS_V5_VISIONFM_PROTOCOL_V1.json"
POLICY_SCHEMA_VERSION = "patient-atlas-v5-visionfm-protocol-v1"
INPUT_SIZE = 224
PATCH_SIZE = 16
BACKBONE_DIMENSION = 768
LAST_BLOCK_COUNT = 4
EMBEDDING_DIMENSION = BACKBONE_DIMENSION * LAST_BLOCK_COUNT
MEAN = np.asarray(
    (0.423737496137619, 0.2609460651874542, 0.128403902053833),
    dtype=np.float32,
)
STD = np.asarray(
    (0.29482534527778625, 0.20167365670204163, 0.13668020069599152),
    dtype=np.float32,
)
VISION_TRANSFORMER_SHA256 = "df057d0d4921088e6a2035fe170a6104d27bd45ab491929aaafb1707c3b1e834"
TENSOR_CHECKPOINT_SHA256 = "d7ea7463c11724801814a9caf8322645ede503dec83a6e03c44082de07c38071"
TENSOR_CHECKPOINT_SIZE = 343208760
TENSOR_COUNT = 150
PARAMETER_COUNT = 85798656
_REQUIRED_BINDINGS = {
    "disease_registry",
    "target_source_policy",
    "fold_policy",
    "fold_audit",
    "v2_grouped_result",
    "tuned_concat_result",
    "retfound_green_milestone",
    "target_loader_implementation",
    "fold_implementation",
    "cfp_selection_implementation",
    "teacher_extraction_implementation",
    "adapter_implementation",
    "foundation_readout_implementation",
    "runner_implementation",
}


class VisionFMIntegrityError(ValueError):
    """Raised when the frozen VisionFM comparator contract is violated."""


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
            raise VisionFMIntegrityError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=_strict_object)
    if not isinstance(value, Mapping):
        raise VisionFMIntegrityError("VisionFM protocol must be an object")
    return value


def require_artifact(
    path: str | Path, expected_sha256: str, expected_size: int
) -> Path:
    artifact = Path(path).expanduser().absolute()
    if (
        not artifact.is_file()
        or artifact.stat().st_size != expected_size
        or sha256_file(artifact) != expected_sha256
    ):
        raise VisionFMIntegrityError("VisionFM artifact authentication failed")
    return artifact


def normalize_pixels(images: np.ndarray) -> np.ndarray:
    values = np.asarray(images)
    if (
        values.ndim != 4
        or values.shape[1:] != (INPUT_SIZE, INPUT_SIZE, 3)
        or values.dtype != np.uint8
    ):
        raise VisionFMIntegrityError(
            "VisionFM pixels must be uint8 [images,224,224,3] RGB"
        )
    normalized = values.astype(np.float32) / np.float32(255.0)
    normalized = (normalized - MEAN[None, None, None]) / STD[None, None, None]
    normalized = np.ascontiguousarray(normalized.transpose(0, 3, 1, 2))
    if not np.isfinite(normalized).all():
        raise VisionFMIntegrityError("VisionFM preprocessing is non-finite")
    return normalized


def _load_architecture(path: Path):
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime-specific
        raise VisionFMIntegrityError("VisionFM PyTorch runtime is unavailable") from error
    if sha256_file(path) != VISION_TRANSFORMER_SHA256:
        raise VisionFMIntegrityError("VisionFM architecture hash differs")
    module_name = "patient_atlas_pinned_visionfm_transformer"
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise VisionFMIntegrityError("VisionFM architecture cannot be imported")
    module = importlib.util.module_from_spec(specification)
    stub = types.ModuleType("utils")
    stub.trunc_normal_ = torch.nn.init.trunc_normal_
    previous = sys.modules.get("utils")
    sys.modules["utils"] = stub
    try:
        specification.loader.exec_module(module)
    except Exception as error:
        raise VisionFMIntegrityError("VisionFM architecture import failed") from error
    finally:
        if previous is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous
    return module


def load_model(
    *,
    checkpoint_path: str | Path,
    architecture_path: str | Path,
    device: str = "cpu",
):
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - runtime-specific
        raise VisionFMIntegrityError("VisionFM runtime is unavailable") from error
    if device not in {"cpu", "mps", "cuda"}:
        raise VisionFMIntegrityError("VisionFM device is invalid")
    checkpoint = require_artifact(
        checkpoint_path, TENSOR_CHECKPOINT_SHA256, TENSOR_CHECKPOINT_SIZE
    )
    architecture = Path(architecture_path).expanduser().absolute()
    module = _load_architecture(architecture)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = tuple(handle.keys())
        metadata = handle.metadata() or {}
    if (
        len(keys) != TENSOR_COUNT
        or len(set(keys)) != TENSOR_COUNT
        or metadata.get("checkpoint_key") != "teacher"
        or metadata.get("state_scope") != "backbone"
        or metadata.get("source_checkpoint_sha256")
        != "9a3176a0308066d6e9c9cf40fc03bfd5fa4580dd3dc62ddb2b5b8db4f2e4f350"
    ):
        raise VisionFMIntegrityError("VisionFM tensor checkpoint schema differs")
    state = load_file(str(checkpoint), device="cpu")
    if (
        len(state) != TENSOR_COUNT
        or sum(value.numel() for value in state.values()) != PARAMETER_COUNT
        or any(not torch.isfinite(value).all() for value in state.values())
    ):
        raise VisionFMIntegrityError("VisionFM tensor state differs")
    model = module.vit_base(
        img_size=[INPUT_SIZE],
        patch_size=PATCH_SIZE,
        num_classes=0,
        use_mean_pooling=False,
    )
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise VisionFMIntegrityError("VisionFM state does not load strictly")
    if (
        int(getattr(model, "embed_dim", -1)) != BACKBONE_DIMENSION
        or len(getattr(model, "blocks", ())) != 12
        or tuple(model.pos_embed.shape) != (1, 197, BACKBONE_DIMENSION)
    ):
        raise VisionFMIntegrityError("VisionFM architecture properties differ")
    model = model.to(torch.device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode_pixels(model: Any, images: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime-specific
        raise VisionFMIntegrityError("PyTorch is unavailable") from error
    batch = torch.from_numpy(normalize_pixels(images)).to(torch.device(device))
    with torch.inference_mode():
        layers = model.get_intermediate_layers(batch, LAST_BLOCK_COUNT)
        encoded = torch.cat([layer[:, 0] for layer in layers], dim=-1)
    values = np.asarray(encoded.detach().cpu().numpy(), dtype=np.float32)
    if values.shape != (len(images), EMBEDDING_DIMENSION):
        raise VisionFMIntegrityError("VisionFM returned an unexpected shape")
    if not np.isfinite(values).all():
        raise VisionFMIntegrityError("VisionFM returned non-finite values")
    return values


def validate_protocol(
    project_root: str | Path,
    policy_path: str | Path | None = None,
) -> tuple[Mapping[str, Any], str, Mapping[str, Any], tuple[BreadthEndpoint, ...]]:
    root = Path(project_root).resolve()
    canonical = (root / POLICY_NAME).resolve()
    path = canonical if policy_path is None else Path(policy_path).resolve()
    if path != canonical:
        raise VisionFMIntegrityError("VisionFM protocol is not canonical")
    policy = _load_json(path)
    if (
        policy.get("schema_version") != POLICY_SCHEMA_VERSION
        or policy.get("status")
        != "frozen_after_checkpoint_acquisition_before_patient_encoding_or_scoring"
    ):
        raise VisionFMIntegrityError("VisionFM protocol is not frozen")
    bindings = policy.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != _REQUIRED_BINDINGS:
        raise VisionFMIntegrityError("VisionFM bindings differ")
    for label, raw in bindings.items():
        if not isinstance(raw, Mapping) or set(raw) != {"file", "sha256"}:
            raise VisionFMIntegrityError(f"malformed VisionFM binding: {label}")
        name, digest = raw.get("file"), raw.get("sha256")
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise VisionFMIntegrityError(f"invalid VisionFM binding path: {label}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise VisionFMIntegrityError(f"invalid VisionFM binding hash: {label}")
        if sha256_file(root / name) != digest:
            raise VisionFMIntegrityError(f"VisionFM binding hash differs: {label}")
    parent = _load_json(root / str(bindings["v2_grouped_result"]["file"]))
    tuned = _load_json(root / str(bindings["tuned_concat_result"]["file"]))
    green = _load_json(root / str(bindings["retfound_green_milestone"]["file"]))
    registry = _load_json(root / str(bindings["disease_registry"]["file"]))
    scope = policy.get("scope", {})
    targets = scope.get("target_columns")
    if (
        parent.get("decision", {}).get("paired_information_gate_passed") is not True
        or parent.get("privacy", {}).get("aggregate_only") is not True
        or tuned.get("status") != "full_nested_tuned_concat_registry_complete"
        or green.get("decision", {}).get("retfound_green_comparator_complete") is not True
        or scope.get("official_train_validation_only") is not True
        or scope.get("official_test_images_or_targets_allowed") is not False
        or scope.get("universal_exclusion_profile") != "circularity18_universal"
        or targets != parent.get("scope", {}).get("endpoint_columns")
        or policy.get("outer_fold_assignment_sha256")
        != parent.get("outer_fold_assignment_sha256")
    ):
        raise VisionFMIntegrityError("VisionFM scope differs")
    artifact = policy.get("artifact")
    required_artifact = {
        "source_checkpoint",
        "source_checkpoint_sha256",
        "source_checkpoint_size_bytes",
        "tensor_checkpoint",
        "tensor_checkpoint_sha256",
        "tensor_checkpoint_size_bytes",
        "architecture_file",
        "architecture_sha256",
        "repository_commit",
        "download_url",
        "license",
        "license_scope",
        "pretraining_overlap_status",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != required_artifact:
        raise VisionFMIntegrityError("VisionFM artifact contract differs")
    if (
        artifact.get("source_checkpoint_sha256")
        != "9a3176a0308066d6e9c9cf40fc03bfd5fa4580dd3dc62ddb2b5b8db4f2e4f350"
        or artifact.get("source_checkpoint_size_bytes") != 1507468320
        or artifact.get("tensor_checkpoint_sha256") != TENSOR_CHECKPOINT_SHA256
        or artifact.get("tensor_checkpoint_size_bytes") != TENSOR_CHECKPOINT_SIZE
        or artifact.get("architecture_sha256") != VISION_TRANSFORMER_SHA256
        or artifact.get("repository_commit") != "648ab0895f8e80e2c937f658bd2f1629ea6d27e7"
        or artifact.get("license") != "CC BY-NC 4.0"
        or artifact.get("license_scope") != "benchmark_only_noncommercial_research"
        or artifact.get("pretraining_overlap_status") != "unresolved_not_proven_absent"
    ):
        raise VisionFMIntegrityError("VisionFM artifact metadata differs")
    require_artifact(
        str(artifact["tensor_checkpoint"]),
        TENSOR_CHECKPOINT_SHA256,
        TENSOR_CHECKPOINT_SIZE,
    )
    if sha256_file(str(artifact["architecture_file"])) != VISION_TRANSFORMER_SHA256:
        raise VisionFMIntegrityError("VisionFM architecture artifact differs")
    if policy.get("representation") != {
        "encoder_weights_frozen": True,
        "outcomes_received_by_encoder": False,
        "outer_test_used_for_encoder_or_pooling_fit": False,
        "architecture": "vit_base_patch16_224",
        "preprocessing": "RGB full-frame bilinear resize to 224, float scale, then released Fundus channel means and standard deviations; no augmentation",
        "image_vector": "concatenated CLS tokens from the final four transformer blocks",
        "image_vector_dimension": EMBEDDING_DIMENSION,
        "patient_pooling": "unweighted mean of valid image vectors",
        "patient_embedding_file_allowed": False,
    }:
        raise VisionFMIntegrityError("VisionFM representation policy differs")
    if policy.get("readout") != {
        "task": "binary",
        "implementation": "standardized_intercept_unpenalized_ridge_logistic",
        "optimizer": "scipy_lbfgsb_then_newtoncg_same_objective_v2",
        "penalty_grid": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
        "inner_folds_identical_to_v2": True,
        "age_appended_and_unpenalized": True,
        "outer_test_used_for_selection_or_standardization": False,
    }:
        raise VisionFMIntegrityError("VisionFM readout policy differs")
    if policy.get("uncertainty") != {
        "bootstrap_unit": "patient",
        "bootstrap_samples": 2000,
        "bootstrap_seed": 314159,
        "confidence_level": 0.95,
        "paired_contrast_to_v2_available": False,
    }:
        raise VisionFMIntegrityError("VisionFM uncertainty policy differs")
    if policy.get("privacy") != {
        "patient_derived_processing": "local_only",
        "patient_embeddings_written_to_disk": False,
        "patient_ids_targets_predictions_losses_embeddings_or_coordinates_serialized": False,
        "aggregate_only_result": True,
        "small_cell_threshold": 10,
    }:
        raise VisionFMIntegrityError("VisionFM privacy policy differs")
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
        raise VisionFMIntegrityError("VisionFM losses are malformed")
    score, endpoint_scores, family_scores = _family_score(values, endpoints)
    generator = np.random.default_rng(bootstrap_seed)
    samples = np.empty(bootstrap_samples, dtype=np.float64)
    for iteration in range(bootstrap_samples):
        indices = generator.integers(0, len(values), size=len(values))
        samples[iteration] = _family_score(values[indices], endpoints)[0]
    alpha = 1.0 - confidence_level
    return {
        "comparator": "visionfm_fundus",
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
        fallback_count = 0
        for value in values:
            key = json.dumps(
                dict(sorted(value.selected_penalties.items())),
                sort_keys=True,
                separators=(",", ":"),
            )
            penalties[key] = penalties.get(key, 0) + 1
            fallback_count += int(getattr(value, "optimizer_fallback_used", False))
        output[target] = {
            "fold_count": len(values),
            "test_eligible_count": int(sum(value.test_eligible_count for value in values)),
            "selected_penalty_counts": dict(sorted(penalties.items())),
            "optimizer_fallback_count": fallback_count,
            "contains_coefficients_predictions_or_losses": False,
        }
    return output


__all__ = [
    "BACKBONE_DIMENSION",
    "EMBEDDING_DIMENSION",
    "INPUT_SIZE",
    "LAST_BLOCK_COUNT",
    "MEAN",
    "POLICY_NAME",
    "STD",
    "VisionFMIntegrityError",
    "encode_pixels",
    "load_model",
    "normalize_pixels",
    "sha256_file",
    "summarize_losses",
    "summarize_readouts",
    "validate_protocol",
]
