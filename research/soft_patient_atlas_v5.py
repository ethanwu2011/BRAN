"""Retinal-capacity-expanded structured Patient Atlas v5 model kernel."""

from __future__ import annotations

import torch
import torch.nn as nn

from soft_patient_atlas import PatientAtlasConfig
from soft_patient_atlas_v2 import LatentPartition, StructuredSoftPatientAtlas


V5_PARTITION = LatentPartition(
    shared_dim=32,
    eye_private_dim=160,
    clinical_private_dim=64,
)


class RetinalCapacityExpandedSoftPatientAtlas(StructuredSoftPatientAtlas):
    """The frozen 256-factor v5 development architecture."""

    def __init__(
        self,
        config: PatientAtlasConfig,
        blood_tower: nn.Module,
        *,
        blood_anchor_eligible_mask: torch.Tensor,
    ) -> None:
        if config.latent_dim != 256:
            raise ValueError("retinal-capacity-expanded v5 requires latent_dim=256")
        if config.hidden_dim < 192:
            raise ValueError("retinal-capacity-expanded v5 requires hidden_dim at least 192")
        super().__init__(
            config,
            blood_tower,
            blood_anchor_eligible_mask=blood_anchor_eligible_mask,
            partition=V5_PARTITION,
        )
        if self.structured_vector_dimension != 289:
            raise RuntimeError("retinal-capacity-expanded v5 vector width differs")
        if self.structured_uncertainty_dimension != 288:
            raise RuntimeError("retinal-capacity-expanded v5 uncertainty width differs")

    @property
    def eye_factor_capacity(self) -> int:
        return self.partition.shared_dim + self.partition.eye_private_dim

    @property
    def clinical_factor_capacity(self) -> int:
        return self.partition.shared_dim + self.partition.clinical_private_dim


__all__ = ["RetinalCapacityExpandedSoftPatientAtlas", "V5_PARTITION"]
