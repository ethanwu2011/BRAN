"""Structured shared/private extension of the frozen Patient Atlas v1 kernel.

V1 allowed every factor to become softly eye-, clinical-, or jointly relevant.
That flexibility collapsed to a clinically dominated solution in the completed
development refit.  V2 keeps the same probabilistic evidence fusion, typed
decoders, masking, calibration interfaces, and 64-dimensional fused state, but
hard-partitions capacity into shared, eye-private, and clinical-private blocks.

This module contains no patient data, split access, outcome logic, or filesystem
operations.  The retired v1 official test is prohibited by the separate v2
protocol and data runner.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from soft_patient_atlas import (
    EvidenceFactor,
    GaussianState,
    PatientAtlasConfig,
    PatientStateOutput,
    SoftGroupFactorDecoder,
    SoftPatientAtlas,
)


@dataclass(frozen=True)
class LatentPartition:
    """Named, contiguous latent blocks whose widths sum to ``latent_dim``."""

    shared_dim: int = 32
    eye_private_dim: int = 16
    clinical_private_dim: int = 16

    def validate(self, latent_dim: int) -> None:
        values = (self.shared_dim, self.eye_private_dim, self.clinical_private_dim)
        if any(not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("every structured latent block must have positive width")
        if sum(values) != latent_dim:
            raise ValueError("structured latent block widths must sum to latent_dim")

    @property
    def shared_slice(self) -> slice:
        return slice(0, self.shared_dim)

    @property
    def eye_private_slice(self) -> slice:
        start = self.shared_dim
        return slice(start, start + self.eye_private_dim)

    @property
    def clinical_private_slice(self) -> slice:
        start = self.shared_dim + self.eye_private_dim
        return slice(start, start + self.clinical_private_dim)

    def support_masks(
        self, *, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total = self.shared_dim + self.eye_private_dim + self.clinical_private_dim
        eye = torch.zeros(total, dtype=torch.bool, device=device)
        clinical = torch.zeros_like(eye)
        eye[self.shared_slice] = True
        eye[self.eye_private_slice] = True
        clinical[self.shared_slice] = True
        clinical[self.clinical_private_slice] = True
        return eye, clinical

    @property
    def roles(self) -> tuple[str, ...]:
        return (
            ("shared",) * self.shared_dim
            + ("eye_private",) * self.eye_private_dim
            + ("clinical_private",) * self.clinical_private_dim
        )


@dataclass
class StructuredEvidenceVector:
    """Screening vector plus its non-exported uncertainty sidecar."""

    mean: torch.Tensor
    log_variance_sidecar: torch.Tensor
    fused_state: GaussianState


class StructuredGroupFactorDecoder(SoftGroupFactorDecoder):
    """V1 typed decoder with exact shared/private loading support."""

    def __init__(
        self, config: PatientAtlasConfig, partition: LatentPartition
    ) -> None:
        partition.validate(config.latent_dim)
        super().__init__(config)
        eye_support, clinical_support = partition.support_masks()
        self.register_buffer("eye_support_mask", eye_support)
        self.register_buffer("clinical_support_mask", clinical_support)
        self.partition = partition

    @property
    def eye_amplitude(self) -> torch.Tensor:
        return F.softplus(self.eye_raw_amplitude) * self.eye_support_mask.to(
            dtype=self.eye_raw_amplitude.dtype
        )

    @property
    def blood_amplitude(self) -> torch.Tensor:
        return F.softplus(self.blood_raw_amplitude) * self.clinical_support_mask.to(
            dtype=self.blood_raw_amplitude.dtype
        )


class StructuredSoftPatientAtlas(SoftPatientAtlas):
    """One probabilistic state with shared and view-private latent capacity."""

    def __init__(
        self,
        config: PatientAtlasConfig,
        blood_tower: nn.Module,
        *,
        blood_anchor_eligible_mask: torch.Tensor,
        partition: LatentPartition = LatentPartition(),
    ) -> None:
        partition.validate(config.latent_dim)
        super().__init__(
            config,
            blood_tower,
            blood_anchor_eligible_mask=blood_anchor_eligible_mask,
        )
        self.partition = partition
        self.decoder = StructuredGroupFactorDecoder(config, partition)

    @staticmethod
    def _state_from_factor(
        factor: EvidenceFactor,
        *,
        min_log_variance: float,
        max_log_variance: float,
    ) -> GaussianState:
        """Fuse one recognition potential with the standard-normal prior."""

        if (factor.precision_increment < 0).any():
            raise ValueError("evidence precision increments must be nonnegative")
        precision = 1.0 + factor.precision_increment
        mean = factor.natural_parameter / precision
        log_variance = (-precision.log()).clamp(
            min=min_log_variance, max=max_log_variance
        )
        return GaussianState(mean=mean, log_variance=log_variance)

    def structured_vector(
        self, output: PatientStateOutput
    ) -> StructuredEvidenceVector:
        """Return the fixed v2 vector without availability or uncertainty shortcuts.

        Layout is ``[eye shared | clinical shared | eye private | clinical
        private | demographics]``.  A missing view contributes exact prior mean
        zero and log variance zero (variance one) to every block it owns.
        """

        latent = self.config.latent_dim
        if (
            output.eye_evidence.center.shape[-1] != latent
            or output.blood_evidence.center.shape[-1] != latent
            or output.physiology.mean.shape[-1] != latent
        ):
            raise ValueError("PatientStateOutput latent width differs from v2 config")
        eye = self._state_from_factor(
            output.eye_evidence,
            min_log_variance=self.config.min_log_variance,
            max_log_variance=self.config.max_log_variance,
        )
        clinical = self._state_from_factor(
            output.blood_evidence,
            min_log_variance=self.config.min_log_variance,
            max_log_variance=self.config.max_log_variance,
        )
        p = self.partition
        mean = torch.cat(
            [
                eye.mean[:, p.shared_slice],
                clinical.mean[:, p.shared_slice],
                eye.mean[:, p.eye_private_slice],
                clinical.mean[:, p.clinical_private_slice],
                output.demographic_context,
            ],
            dim=-1,
        )
        uncertainty = torch.cat(
            [
                eye.log_variance[:, p.shared_slice],
                clinical.log_variance[:, p.shared_slice],
                eye.log_variance[:, p.eye_private_slice],
                clinical.log_variance[:, p.clinical_private_slice],
            ],
            dim=-1,
        )
        return StructuredEvidenceVector(
            mean=mean,
            log_variance_sidecar=uncertainty,
            fused_state=output.physiology,
        )

    @property
    def structured_vector_dimension(self) -> int:
        p = self.partition
        return (
            2 * p.shared_dim
            + p.eye_private_dim
            + p.clinical_private_dim
            + self.config.demographic_dim
        )

    @property
    def structured_uncertainty_dimension(self) -> int:
        p = self.partition
        return 2 * p.shared_dim + p.eye_private_dim + p.clinical_private_dim

    def factor_role_profile(self) -> tuple[str, ...]:
        """Return structural roles; this does not name biological axes."""

        return self.partition.roles


__all__ = [
    "LatentPartition",
    "StructuredEvidenceVector",
    "StructuredGroupFactorDecoder",
    "StructuredSoftPatientAtlas",
]
