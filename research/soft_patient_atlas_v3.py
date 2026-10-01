"""Capacity-preserving structured Patient Atlas v3 model kernel.

V3 uses 32 shared, 32 eye-private, and 32 clinical-private factors. Each
modality therefore retains 64 permitted factors, matching the unstructured v1
capacity while keeping exact shared/private supports and probabilistic priors.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from soft_patient_atlas import PatientAtlasConfig
from soft_patient_atlas_v2 import LatentPartition, StructuredSoftPatientAtlas


V3_PARTITION = LatentPartition(
    shared_dim=32,
    eye_private_dim=32,
    clinical_private_dim=32,
)


class CapacityPreservingSoftPatientAtlas(StructuredSoftPatientAtlas):
    """The one frozen 96-factor v3 architecture."""

    def __init__(
        self,
        config: PatientAtlasConfig,
        blood_tower: nn.Module,
        *,
        blood_anchor_eligible_mask: torch.Tensor,
    ) -> None:
        if config.latent_dim != 96:
            raise ValueError("capacity-preserving v3 requires latent_dim=96")
        super().__init__(
            config,
            blood_tower,
            blood_anchor_eligible_mask=blood_anchor_eligible_mask,
            partition=V3_PARTITION,
        )
        if self.structured_vector_dimension != 129:
            raise RuntimeError("capacity-preserving v3 vector width differs")
        if self.structured_uncertainty_dimension != 128:
            raise RuntimeError("capacity-preserving v3 uncertainty width differs")

    @property
    def per_modality_factor_dimension(self) -> int:
        return self.partition.shared_dim + self.partition.eye_private_dim


__all__ = ["CapacityPreservingSoftPatientAtlas", "V3_PARTITION"]
