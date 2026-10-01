"""Capacity-expanded structured Patient Atlas v4 model kernel.

V4 keeps the probabilistic shared/private semantics of v3 while expanding the
retinal and clinical private subspaces.  It does not append a raw/PCA bypass.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from soft_patient_atlas import PatientAtlasConfig
from soft_patient_atlas_v2 import LatentPartition, StructuredSoftPatientAtlas


V4_PARTITION = LatentPartition(
    shared_dim=32,
    eye_private_dim=96,
    clinical_private_dim=64,
)


class CapacityExpandedSoftPatientAtlas(StructuredSoftPatientAtlas):
    """The frozen 192-factor v4 development architecture."""

    def __init__(
        self,
        config: PatientAtlasConfig,
        blood_tower: nn.Module,
        *,
        blood_anchor_eligible_mask: torch.Tensor,
    ) -> None:
        if config.latent_dim != 192:
            raise ValueError("capacity-expanded v4 requires latent_dim=192")
        if config.hidden_dim < 128:
            raise ValueError("capacity-expanded v4 requires hidden_dim at least 128")
        super().__init__(
            config,
            blood_tower,
            blood_anchor_eligible_mask=blood_anchor_eligible_mask,
            partition=V4_PARTITION,
        )
        if self.structured_vector_dimension != 225:
            raise RuntimeError("capacity-expanded v4 vector width differs")
        if self.structured_uncertainty_dimension != 224:
            raise RuntimeError("capacity-expanded v4 uncertainty width differs")

    @property
    def eye_factor_capacity(self) -> int:
        return self.partition.shared_dim + self.partition.eye_private_dim

    @property
    def clinical_factor_capacity(self) -> int:
        return self.partition.shared_dim + self.partition.clinical_private_dim


__all__ = ["CapacityExpandedSoftPatientAtlas", "V4_PARTITION"]
