"""Fail-closed artifact contracts for Patient Atlas v1.

This module authenticates model weights and schemas only.  It never reads
patient rows.  The external checkpoint is hashed before deserialization, so an
untrusted or accidentally substituted pickle is rejected before ``torch.load``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


EXPECTED_ARTIFACT_ID = "external_denoise30_v1"
EXPECTED_SCHEMA_VERSION = "external-blood-tower-v1"
EXPECTED_CHECKPOINT_KEY = "denoise30_external"
EXPECTED_FILE_SHA256 = "7bf6dfd3ffe0438b51679f253e48647268292c9f2b9848e6c36caae6380da141"
EXPECTED_STATE_SHA256 = "990328931a9c16bef0b305aa6c82ff9c46046bbb5d58d6d0aced28488cad2b2b"
EXPECTED_COLUMNS_SHA256 = "5652f63bbacdc0cd007cc6735acfa1705461eb9400bacd5a45e6886a9b4ddbb7"
EXPECTED_STATE_HASH_ALGORITHM = (
    "sha256 over state entries sorted by name; for each entry append UTF-8 name, "
    "UTF-8 torch dtype, int64 shape bytes, then contiguous CPU tensor bytes"
)
EXPECTED_ANCHOR_INELIGIBLE = (
    "a_g_ratio",
    "buncreatinineratio",
    "c_peptide",
    "insulin",
    "vit_bmi_vsorres",
    "vit_diabp_vsorres",
    "vit_height_vsorres",
    "vit_hip_vsorres",
    "vit_pulse_vsorres",
    "vit_pulse_vsorres_2",
    "vit_sysbp_vsorres",
    "vit_waist_vsorres",
    "vit_weight_vsorres",
    "vit_whr_vsorres",
)


def hash_json(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def hash_state_dict(state: Mapping[str, torch.Tensor]) -> str:
    """Canonical tensor-state hash used by the clinical project."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class ExternalDenoise30Tower(nn.Module):
    """Checkpoint-defining 59-field external blood tower."""

    def __init__(self, num_features: int = 59, latent_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(2 * num_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_features),
        )
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def encode(self, values: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        return self.enc(torch.cat([values * observed_mask, observed_mask], dim=1))


@dataclass(frozen=True)
class ExternalBloodTowerArtifact:
    tower: ExternalDenoise30Tower
    columns: tuple[str, ...]
    anchor_eligible_mask: torch.Tensor
    contract: Mapping[str, Any]
    checkpoint_file_sha256: str
    state_sha256: str


def _load_contract(path: str | Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    contract = json.loads(
        Path(path).read_bytes(), object_pairs_hook=reject_duplicates
    )
    if not isinstance(contract, dict):
        raise TypeError("blood tower contract must contain a JSON object")
    pinned = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "artifact_id": EXPECTED_ARTIFACT_ID,
        "checkpoint_key": EXPECTED_CHECKPOINT_KEY,
        "file_sha256": EXPECTED_FILE_SHA256,
        "state_sha256": EXPECTED_STATE_SHA256,
        "state_hash_algorithm": EXPECTED_STATE_HASH_ALGORITHM,
        "ordered_columns_sha256": EXPECTED_COLUMNS_SHA256,
    }
    for field, expected in pinned.items():
        if contract.get(field) != expected:
            raise ValueError(f"blood tower contract mismatch for {field}")
    if contract.get("source_filename") != "blood_tower_external.pt":
        raise ValueError("unexpected external tower source filename")
    return contract


def validate_ordered_columns(
    actual_columns: Sequence[str], expected_columns: Sequence[str]
) -> tuple[str, ...]:
    actual = tuple(map(str, actual_columns))
    expected = tuple(map(str, expected_columns))
    if actual != expected:
        if set(actual) == set(expected):
            raise ValueError("blood feature columns are reordered")
        missing = sorted(set(expected) - set(actual))
        unknown = sorted(set(actual) - set(expected))
        raise ValueError(
            f"blood feature schema mismatch; missing={missing}, unknown={unknown}"
        )
    if hash_json(list(actual)) != EXPECTED_COLUMNS_SHA256:
        raise ValueError("blood feature order hash mismatch")
    return actual


def _validate_state_shapes(
    state: Mapping[str, torch.Tensor], architecture: Mapping[str, Any]
) -> None:
    n = int(architecture["num_features"])
    h = int(architecture["hidden_width"])
    d = int(architecture["latent_width"])
    expected_shapes = {
        "enc.0.weight": (h, 2 * n),
        "enc.0.bias": (h,),
        "enc.2.weight": (d, h),
        "enc.2.bias": (d,),
        "dec.0.weight": (h, d),
        "dec.0.bias": (h,),
        "dec.2.weight": (n, h),
        "dec.2.bias": (n,),
        "proj.0.weight": (d, d),
        "proj.0.bias": (d,),
        "proj.2.weight": (d, d),
        "proj.2.bias": (d,),
    }
    if set(state) != set(expected_shapes):
        raise ValueError("external tower state keys do not match the contract")
    for name, shape in expected_shapes.items():
        if tuple(state[name].shape) != shape:
            raise ValueError(f"external tower tensor has wrong shape: {name}")


def _validated_anchor_mask(contract: Mapping[str, Any]) -> torch.Tensor:
    columns = tuple(map(str, contract["ordered_columns"]))
    coverage = contract.get("anchor_pretraining_coverage", {})
    mask_values = coverage.get("anchor_eligible_mask")
    if not isinstance(mask_values, list) or len(mask_values) != len(columns):
        raise ValueError("anchor eligibility mask is missing or has the wrong width")
    if any(type(value) is not bool for value in mask_values):
        raise TypeError("anchor eligibility mask must contain only booleans")
    mask = torch.tensor(mask_values, dtype=torch.bool)
    declared_untrained = tuple(map(str, coverage.get("never_observed_external_pretraining", ())))
    observed_untrained = tuple(
        column for column, eligible in zip(columns, mask_values) if not eligible
    )
    if (
        observed_untrained != declared_untrained
        or observed_untrained != EXPECTED_ANCHOR_INELIGIBLE
    ):
        raise ValueError("anchor pretraining coverage declaration is inconsistent")
    return mask


def load_external_blood_tower(
    checkpoint_path: str | Path,
    contract_path: str | Path,
) -> ExternalBloodTowerArtifact:
    """Authenticate and load the only blood tower allowed for Atlas v1."""
    checkpoint_path = Path(checkpoint_path)
    contract = _load_contract(contract_path)
    if checkpoint_path.name != contract["source_filename"]:
        raise ValueError("checkpoint filename does not match the pinned contract")
    checkpoint_bytes = checkpoint_path.read_bytes()
    file_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    if file_hash != EXPECTED_FILE_SHA256:
        raise ValueError("external blood tower file hash mismatch")

    # Deserialization occurs only after the complete file is authenticated.
    payload = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
    )
    allowed_payload_keys = {
        EXPECTED_CHECKPOINT_KEY,
        "cols",
        "n_train",
        "corpora",
    }
    if set(payload) != allowed_payload_keys:
        raise ValueError("external blood tower payload keys do not match the contract")
    if "scarf" in payload or "denoise30" in payload:
        raise ValueError("legacy or target-adapted blood tower is forbidden")

    columns = validate_ordered_columns(payload["cols"], contract["ordered_columns"])
    provenance = contract["training_provenance"]
    if int(provenance.get("paired_ai_readi_rows_used", -1)) != 0:
        raise ValueError("target-cohort rows are forbidden in the external tower")
    if int(payload["n_train"]) != int(provenance["reported_n_train"]):
        raise ValueError("external tower training-count provenance mismatch")
    if tuple(map(str, payload["corpora"])) != tuple(map(str, provenance["corpora"])):
        raise ValueError("external tower corpus provenance mismatch")

    state = payload[EXPECTED_CHECKPOINT_KEY]
    if not isinstance(state, Mapping) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise TypeError("external tower state is not a tensor mapping")
    _validate_state_shapes(state, contract["architecture"])
    state_hash = hash_state_dict(state)
    if state_hash != EXPECTED_STATE_SHA256:
        raise ValueError("external blood tower state hash mismatch")

    architecture = contract["architecture"]
    tower = ExternalDenoise30Tower(
        num_features=int(architecture["num_features"]),
        latent_dim=int(architecture["latent_width"]),
        hidden_dim=int(architecture["hidden_width"]),
    )
    tower.load_state_dict(state, strict=True)
    tower.requires_grad_(False).eval()
    return ExternalBloodTowerArtifact(
        tower=tower,
        columns=columns,
        anchor_eligible_mask=_validated_anchor_mask(contract),
        contract=contract,
        checkpoint_file_sha256=file_hash,
        state_sha256=state_hash,
    )


__all__ = [
    "EXPECTED_ARTIFACT_ID",
    "EXPECTED_CHECKPOINT_KEY",
    "ExternalBloodTowerArtifact",
    "ExternalDenoise30Tower",
    "hash_json",
    "hash_state_dict",
    "load_external_blood_tower",
    "sha256_file",
    "validate_ordered_columns",
]
