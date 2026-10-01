"""Canonical calibrated vector export for Patient Atlas v4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from patient_atlas_stage2 import precision_temperature_state
from soft_patient_atlas import GaussianState
from soft_patient_atlas_v4 import CapacityExpandedSoftPatientAtlas


AVAILABILITY_ARMS = ("both", "eye_only", "blood_only")


@dataclass(frozen=True)
class CapacityExpandedVectorOutput:
    mean: torch.Tensor
    log_variance_sidecar: torch.Tensor
    calibrated_fused_state: GaussianState
    eye_present: torch.Tensor
    clinical_present: torch.Tensor


def encode_capacity_expanded_vector(
    model: CapacityExpandedSoftPatientAtlas,
    *,
    eye_embeddings: torch.Tensor,
    eye_visible_mask: torch.Tensor,
    blood_values: torch.Tensor,
    blood_visible_mask: torch.Tensor,
    blood_eligible_mask: torch.Tensor,
    demographics: torch.Tensor,
    demographic_mask: torch.Tensor,
    precision_temperatures: Mapping[str, float],
    eye_device_ids: torch.Tensor | None = None,
    eye_laterality_ids: torch.Tensor | None = None,
    eye_quality: torch.Tensor | None = None,
) -> CapacityExpandedVectorOutput:
    """Encode one v4 batch without appending masks or uncertainty."""

    if set(precision_temperatures) != set(AVAILABILITY_ARMS):
        raise ValueError("precision temperatures must cover all availability arms")
    output = model.infer_evidence(
        eye_embeddings=eye_embeddings,
        eye_visible_mask=eye_visible_mask,
        blood_values=blood_values,
        blood_visible_mask=blood_visible_mask,
        blood_eligible_mask=blood_eligible_mask,
        demographics=demographics,
        demographic_mask=demographic_mask,
        eye_device_ids=eye_device_ids,
        eye_laterality_ids=eye_laterality_ids,
        eye_quality=eye_quality,
        enable_interaction=False,
    )
    eye_state = model._state_from_factor(
        output.eye_evidence,
        min_log_variance=model.config.min_log_variance,
        max_log_variance=model.config.max_log_variance,
    )
    clinical_state = model._state_from_factor(
        output.blood_evidence,
        min_log_variance=model.config.min_log_variance,
        max_log_variance=model.config.max_log_variance,
    )
    eye_state = precision_temperature_state(
        eye_state, float(precision_temperatures["eye_only"])
    )
    clinical_state = precision_temperature_state(
        clinical_state, float(precision_temperatures["blood_only"])
    )
    eye_present = output.eye_evidence.present
    clinical_present = output.blood_evidence.present
    both = eye_present & clinical_present
    temperature = torch.ones(
        eye_present.shape[0],
        dtype=output.physiology.mean.dtype,
        device=output.physiology.mean.device,
    )
    temperature = torch.where(
        eye_present & ~clinical_present,
        torch.full_like(temperature, float(precision_temperatures["eye_only"])),
        temperature,
    )
    temperature = torch.where(
        clinical_present & ~eye_present,
        torch.full_like(temperature, float(precision_temperatures["blood_only"])),
        temperature,
    )
    temperature = torch.where(
        both,
        torch.full_like(temperature, float(precision_temperatures["both"])),
        temperature,
    )
    fused = precision_temperature_state(output.physiology, temperature)
    p = model.partition
    mean = torch.cat(
        [
            eye_state.mean[:, p.shared_slice],
            clinical_state.mean[:, p.shared_slice],
            eye_state.mean[:, p.eye_private_slice],
            clinical_state.mean[:, p.clinical_private_slice],
            output.demographic_context,
        ],
        dim=1,
    )
    log_variance = torch.cat(
        [
            eye_state.log_variance[:, p.shared_slice],
            clinical_state.log_variance[:, p.shared_slice],
            eye_state.log_variance[:, p.eye_private_slice],
            clinical_state.log_variance[:, p.clinical_private_slice],
        ],
        dim=1,
    )
    if mean.shape[1] != 225 or log_variance.shape[1] != 224:
        raise RuntimeError("v4 calibrated vector dimensions differ")
    if not bool(torch.isfinite(mean).all() and torch.isfinite(log_variance).all()):
        raise ValueError("v4 calibrated vector contains non-finite values")
    return CapacityExpandedVectorOutput(
        mean=mean,
        log_variance_sidecar=log_variance,
        calibrated_fused_state=fused,
        eye_present=eye_present,
        clinical_present=clinical_present,
    )


__all__ = [
    "AVAILABILITY_ARMS",
    "CapacityExpandedVectorOutput",
    "encode_capacity_expanded_vector",
]
