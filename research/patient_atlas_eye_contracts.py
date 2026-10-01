"""Fail-closed authentication for the frozen Patient Atlas eye tower.

Only public source-tag metadata and model artifacts are inspected.  The module
does not open retinal images or AI-READI rows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from patient_atlas_contracts import hash_json, hash_state_dict


EXPECTED_SCHEMA_VERSION = "external-eye-tower-v1"
EXPECTED_ARTIFACT_ID = "adapted_dinov3_s16_fundus_v1"
EXPECTED_CHECKPOINT_KEY = "student"
EXPECTED_FILE_SHA256 = "6d3c427940ccf528c9ad0ba4bbf4aa5d71a575ae9d3df41d8621799c4d05e5ff"
EXPECTED_STATE_SHA256 = "3ddf148f6876fb5c28c6610b3040b646c4ebac5c47499e987f9b63170be83d34"
EXPECTED_CONFIG_SHA256 = "c86380aae96dc156b7c845b87279636ead9b6316daa31e649a1bb2b8ccad259b"
EXPECTED_METADATA_SHA256 = "266f517e194976f8bf2f3ebe6eaee45be37d91b326771e9a522d3b230743550e"
EXPECTED_CACHE_BUILDER_SHA256 = "00f306f43ba32d8f94f5e416ba927453570eccac38b615f37c6ecce0f41204e5"
EXPECTED_ADAPTATION_SCRIPT_SHA256 = "950f1611867f31cd9e837f498b82cbff8f1bb03a1aa63d0e450cd4dab21c94ee"
EXPECTED_STATE_HASH_ALGORITHM = (
    "sha256 over state entries sorted by name; for each entry append UTF-8 name, "
    "UTF-8 torch dtype, int64 shape bytes, then contiguous CPU tensor bytes"
)
EXPECTED_SOURCE_COUNTS = {
    "brset": 16266,
    "drunified": 92501,
    "jsiec": 1994,
    "odir": 4512,
}
EXPECTED_TRAINING_CONFIG = {
    "epochs": 8,
    "bs": 96,
    "lr": 0.0001,
    "img": 128,
    "limit": 0,
    "out": "dinov3_fundus",
}


@dataclass(frozen=True)
class EyeSourceProvenance:
    source_counts: Mapping[str, int]
    total_images: int
    successfully_decoded_images: int
    metadata_sha256: str
    cache_builder_sha256: str
    adaptation_script_sha256: str
    per_file_identity_manifest_available: bool


@dataclass(frozen=True)
class ExternalEyeTowerArtifact:
    tower: nn.Module
    contract: Mapping[str, Any]
    checkpoint_file_sha256: str
    state_sha256: str
    source_provenance: EyeSourceProvenance


def _strict_json(path: str | Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    value = json.loads(Path(path).read_bytes(), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise TypeError("eye tower contract must contain a JSON object")
    pinned = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "artifact_id": EXPECTED_ARTIFACT_ID,
        "checkpoint_key": EXPECTED_CHECKPOINT_KEY,
        "file_sha256": EXPECTED_FILE_SHA256,
        "state_sha256": EXPECTED_STATE_SHA256,
        "state_hash_algorithm": EXPECTED_STATE_HASH_ALGORITHM,
    }
    for key, expected in pinned.items():
        if value.get(key) != expected:
            raise ValueError(f"eye tower contract mismatch for {key}")
    if value.get("payload_keys") != ["cfg", "student"]:
        raise ValueError("eye tower payload-key contract changed")
    return value


def authenticate_eye_source_provenance(
    contract: Mapping[str, Any],
    *,
    cache_metadata_path: str | Path,
    cache_builder_path: str | Path,
    adaptation_script_path: str | Path,
) -> EyeSourceProvenance:
    """Authenticate source tags/counts without opening any retinal image."""

    provenance = contract.get("source_cache_provenance", {})
    paths_and_hashes = (
        (cache_metadata_path, "metadata_filename", "metadata_file_sha256", EXPECTED_METADATA_SHA256),
        (cache_builder_path, "cache_builder_filename", "cache_builder_sha256", EXPECTED_CACHE_BUILDER_SHA256),
        (adaptation_script_path, "adaptation_script_filename", "adaptation_script_sha256", EXPECTED_ADAPTATION_SCRIPT_SHA256),
    )
    observed_hashes: list[str] = []
    authenticated_bytes: list[bytes] = []
    for path_value, name_key, hash_key, expected_hash in paths_and_hashes:
        path = Path(path_value)
        if path.name != provenance.get(name_key):
            raise ValueError(f"eye provenance filename mismatch for {name_key}")
        if provenance.get(hash_key) != expected_hash:
            raise ValueError(f"eye provenance contract mismatch for {hash_key}")
        # Read each artifact once so the authenticated bytes are the bytes used
        # below.  In particular, never hash the NPY path and then reopen that
        # path for deserialization: a path swap between those operations could
        # otherwise authenticate one file and load another.
        artifact_bytes = path.read_bytes()
        observed = hashlib.sha256(artifact_bytes).hexdigest()
        if observed != expected_hash:
            raise ValueError(f"eye provenance file hash mismatch for {path.name}")
        observed_hashes.append(observed)
        authenticated_bytes.append(artifact_bytes)

    # The NPY contains only public-corpus source tags and technical decode bits.
    # It is deserialized only after exact byte authentication.
    metadata = np.load(io.BytesIO(authenticated_bytes[0]), allow_pickle=True)
    total = int(provenance.get("total_images", -1))
    if metadata.shape != (2, total):
        raise ValueError("eye cache metadata has the wrong shape")
    source_counts = dict(sorted(Counter(map(str, metadata[0])).items()))
    declared_counts = dict(sorted(provenance.get("source_counts", {}).items()))
    if source_counts != EXPECTED_SOURCE_COUNTS or source_counts != declared_counts:
        raise ValueError("eye cache source-tag counts do not match the contract")
    decode_counts = Counter(map(str, metadata[1]))
    successfully_decoded = int(decode_counts.get("1", 0))
    if set(decode_counts) != {"1"} or successfully_decoded != total:
        raise ValueError("eye cache contains failed or unknown decode states")
    if any("aireadi" in source.lower() for source in source_counts):
        raise ValueError("AI-READI source tags are forbidden in eye-tower adaptation")
    if provenance.get("ai_readi_source_tag_count") != 0:
        raise ValueError("contract must declare zero AI-READI source tags")
    if provenance.get("per_file_identity_manifest_available") is not False:
        raise ValueError("historical per-file identity limitation must remain explicit")
    return EyeSourceProvenance(
        source_counts=source_counts,
        total_images=total,
        successfully_decoded_images=successfully_decoded,
        metadata_sha256=observed_hashes[0],
        cache_builder_sha256=observed_hashes[1],
        adaptation_script_sha256=observed_hashes[2],
        per_file_identity_manifest_available=False,
    )


def load_external_eye_tower(
    checkpoint_path: str | Path,
    contract_path: str | Path,
    *,
    cache_metadata_path: str | Path,
    cache_builder_path: str | Path,
    adaptation_script_path: str | Path,
) -> ExternalEyeTowerArtifact:
    """Authenticate the checkpoint, its exact architecture, and source metadata."""

    contract = _strict_json(contract_path)
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.name != contract.get("source_filename"):
        raise ValueError("eye checkpoint filename does not match its contract")
    checkpoint_bytes = checkpoint_path.read_bytes()
    file_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    if file_hash != EXPECTED_FILE_SHA256:
        raise ValueError("eye checkpoint file hash mismatch")
    payload = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
    )
    if not isinstance(payload, dict) or set(payload) != {"cfg", "student"}:
        raise ValueError("eye checkpoint payload keys do not match the contract")
    config = payload["cfg"]
    if config != EXPECTED_TRAINING_CONFIG or hash_json(config) != EXPECTED_CONFIG_SHA256:
        raise ValueError("eye adaptation configuration mismatch")
    declared_adaptation = contract.get("adaptation", {})
    declared_training_values = {
        "epochs": declared_adaptation.get("epochs"),
        "bs": declared_adaptation.get("batch_size"),
        "lr": declared_adaptation.get("learning_rate"),
        "img": declared_adaptation.get("training_resolution_pixels"),
        "limit": declared_adaptation.get("limit"),
        "out": config.get("out"),
    }
    if declared_training_values != EXPECTED_TRAINING_CONFIG:
        raise ValueError("eye adaptation declarations do not match the checkpoint")
    if declared_adaptation.get("checkpoint_config_sha256") != EXPECTED_CONFIG_SHA256:
        raise ValueError("eye config hash contract mismatch")
    state = payload[EXPECTED_CHECKPOINT_KEY]
    if not isinstance(state, Mapping) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise TypeError("eye checkpoint state must be a tensor mapping")
    state_hash = hash_state_dict(state)
    if state_hash != EXPECTED_STATE_SHA256:
        raise ValueError("eye checkpoint state hash mismatch")

    architecture = contract.get("architecture", {})
    if architecture.get("timm_model") != "vit_small_patch16_dinov3.lvd1689m":
        raise ValueError("unexpected eye architecture")
    if len(state) != architecture.get("state_tensor_count"):
        raise ValueError("eye state tensor count mismatch")
    if sum(tensor.numel() for tensor in state.values()) != architecture.get(
        "state_parameter_count"
    ):
        raise ValueError("eye state parameter count mismatch")

    try:
        import timm
    except ImportError as exc:  # pragma: no cover - dependency error is explicit
        raise RuntimeError("timm is required to instantiate the authenticated eye tower") from exc
    tower = timm.create_model(
        architecture["timm_model"],
        pretrained=False,
        num_classes=0,
        img_size=int(contract["inference"]["resolution_pixels"]),
    )
    if (
        int(getattr(tower, "embed_dim", -1)) != architecture.get("embedding_width")
        or int(getattr(tower, "num_prefix_tokens", -1))
        != architecture.get("prefix_token_count")
        or getattr(tower, "global_pool", None) != architecture.get("global_pool")
        or tuple(getattr(tower.patch_embed, "patch_size", ()))
        != (architecture.get("patch_size"),) * 2
    ):
        raise ValueError("installed eye architecture violates the inference pooling contract")
    tower.load_state_dict(state, strict=True)
    tower.requires_grad_(False).eval()
    source_provenance = authenticate_eye_source_provenance(
        contract,
        cache_metadata_path=cache_metadata_path,
        cache_builder_path=cache_builder_path,
        adaptation_script_path=adaptation_script_path,
    )
    return ExternalEyeTowerArtifact(
        tower=tower,
        contract=contract,
        checkpoint_file_sha256=file_hash,
        state_sha256=state_hash,
        source_provenance=source_provenance,
    )


__all__ = [
    "ExternalEyeTowerArtifact",
    "EyeSourceProvenance",
    "authenticate_eye_source_provenance",
    "load_external_eye_tower",
]
