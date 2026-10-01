"""Protected-retinal-subspace probabilistic Patient Atlas V6 kernel.

V6 keeps V5's 32/160/64 shared/private partition.  The first 64
eye-private recognition coordinates are a fold-fit, outcome-free linear
retinal subspace.  They live inside the Gaussian evidence state (and therefore
obey prior reversion and uncertainty semantics); they are not coordinates
appended to the canonical vector.  The other 96 eye-private coordinates remain
nonlinear and learned.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from soft_patient_atlas import EvidenceFactor, EyeEvidenceEncoder, PatientAtlasConfig
from soft_patient_atlas_v5 import RetinalCapacityExpandedSoftPatientAtlas, V5_PARTITION


V6_PROTECTED_RETINAL_DIM = 64
V6_PROTECTED_RETINAL_PRECISION = 1.0
V6_PROTECTED_RETINAL_DECODER_AMPLITUDE = 2.0


class ProtectedRetinalEvidenceEncoder(EyeEvidenceEncoder):
    """V5 set encoder with an exact linear evidence block inside eye-private z."""

    def __init__(
        self,
        config: PatientAtlasConfig,
        retinal_components: torch.Tensor,
    ) -> None:
        super().__init__(config)
        components = torch.as_tensor(retinal_components).detach().clone()
        expected = (config.eye_dim, V6_PROTECTED_RETINAL_DIM)
        if components.shape != expected or not components.is_floating_point():
            raise TypeError(
                f"retinal_components must be floating with shape {expected}"
            )
        if not bool(torch.isfinite(components).all()):
            raise ValueError("retinal_components contain non-finite values")
        components = components.to(dtype=torch.float32)
        gram = components.T @ components
        identity = torch.eye(
            V6_PROTECTED_RETINAL_DIM, dtype=components.dtype
        )
        if not torch.allclose(gram, identity, atol=1e-5, rtol=1e-5):
            raise ValueError("retinal_components must have orthonormal columns")
        self.register_buffer("retinal_components", components)

    @property
    def protected_slice(self) -> slice:
        start = V5_PARTITION.shared_dim
        return slice(start, start + V6_PROTECTED_RETINAL_DIM)

    def forward(
        self,
        eye_embeddings: torch.Tensor,
        visible_mask: torch.Tensor,
        *,
        factor_support: torch.Tensor,
        demographics: torch.Tensor,
        device_ids: torch.Tensor,
        laterality_ids: torch.Tensor,
        quality: torch.Tensor,
    ) -> EvidenceFactor:
        learned = super().forward(
            eye_embeddings,
            visible_mask,
            factor_support=factor_support,
            demographics=demographics,
            device_ids=device_ids,
            laterality_ids=laterality_ids,
            quality=quality,
        )
        safe = torch.where(
            visible_mask[..., None], eye_embeddings, torch.zeros_like(eye_embeddings)
        )
        count = visible_mask.sum(dim=1, keepdim=True)
        mean = safe.sum(dim=1) / count.clamp(min=1).to(dtype=eye_embeddings.dtype)
        protected = mean @ self.retinal_components.to(dtype=eye_embeddings.dtype)
        protected = torch.where(
            learned.present[:, None], protected, torch.zeros_like(protected)
        )
        protected_precision = learned.present[:, None].to(
            dtype=eye_embeddings.dtype
        ).expand(-1, V6_PROTECTED_RETINAL_DIM)
        protected_precision = (
            protected_precision * V6_PROTECTED_RETINAL_PRECISION
        )

        center = learned.center.clone()
        precision = learned.precision_increment.clone()
        center[:, self.protected_slice] = protected
        precision[:, self.protected_slice] = protected_precision
        return EvidenceFactor(
            center=center,
            precision_increment=precision,
            reliability=learned.reliability,
            present=learned.present,
            summary=learned.summary,
        )


class ProtectedRetinalSubspaceSoftPatientAtlas(
    RetinalCapacityExpandedSoftPatientAtlas
):
    """The single precommitted 256-factor V6 development architecture."""

    def __init__(
        self,
        config: PatientAtlasConfig,
        blood_tower: nn.Module,
        *,
        blood_anchor_eligible_mask: torch.Tensor,
        retinal_components: torch.Tensor,
    ) -> None:
        super().__init__(
            config,
            blood_tower,
            blood_anchor_eligible_mask=blood_anchor_eligible_mask,
        )
        self.eye_encoder = ProtectedRetinalEvidenceEncoder(
            config, retinal_components
        )
        protected = self.eye_encoder.protected_slice
        with torch.no_grad():
            components = self.eye_encoder.retinal_components.to(
                dtype=self.decoder.eye_raw_direction.dtype
            )
            self.decoder.eye_raw_direction[:, protected].copy_(components)
            raw_amplitude = torch.log(
                torch.expm1(
                    torch.tensor(
                        V6_PROTECTED_RETINAL_DECODER_AMPLITUDE,
                        dtype=self.decoder.eye_raw_amplitude.dtype,
                    )
                )
            )
            self.decoder.eye_raw_amplitude[protected].fill_(raw_amplitude)
            self.decoder.renormalize_loading_directions_()
        if self.structured_vector_dimension != 289:
            raise RuntimeError("protected-retinal V6 vector width differs")

    @property
    def protected_retinal_slice(self) -> slice:
        return self.eye_encoder.protected_slice

    @property
    def learned_nonlinear_eye_private_dimension(self) -> int:
        return V5_PARTITION.eye_private_dim - V6_PROTECTED_RETINAL_DIM


__all__ = [
    "ProtectedRetinalEvidenceEncoder",
    "ProtectedRetinalSubspaceSoftPatientAtlas",
    "V6_PROTECTED_RETINAL_DECODER_AMPLITUDE",
    "V6_PROTECTED_RETINAL_DIM",
    "V6_PROTECTED_RETINAL_PRECISION",
]
