"""Softly shared probabilistic patient atlas.

The model maps any visible subset of retinal embeddings and blood/clinical
measurements into one fixed Gaussian patient state.  Each modality emits an
approximate Gaussian recognition potential; potentials are fused in
natural-parameter space so the prior is counted once and an absent modality
contributes exactly zero.

This module deliberately contains no patient data, outcome definitions, fold
logic, or filesystem access.  Those belong to the authenticated training and
evaluation pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PatientAtlasConfig:
    eye_dim: int
    blood_anchor_dim: int
    num_continuous: int
    num_binary: int
    demographic_dim: int = 1
    num_devices: int = 5
    num_lateralities: int = 3
    latent_dim: int = 64
    hidden_dim: int = 64
    interaction_rank: int = 8
    student_df: float = 5.0
    evidence_center_bound: float = 5.0
    group_shrinkage_rate: float = 1e-3
    binary_loading_metric_weight: float = 0.25
    min_log_variance: float = -10.0
    max_log_variance: float = 4.0

    def __post_init__(self) -> None:
        integers = {
            "eye_dim": self.eye_dim,
            "blood_anchor_dim": self.blood_anchor_dim,
            "num_continuous": self.num_continuous,
            "num_binary": self.num_binary,
            "num_devices": self.num_devices,
            "num_lateralities": self.num_lateralities,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "interaction_rank": self.interaction_rank,
        }
        for name, value in integers.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.demographic_dim, int) or self.demographic_dim < 0:
            raise ValueError("demographic_dim must be a nonnegative integer")
        if self.student_df <= 2:
            raise ValueError("student_df must exceed 2 so predictive variance exists")
        if self.evidence_center_bound <= 0:
            raise ValueError("evidence_center_bound must be positive")
        if self.group_shrinkage_rate <= 0:
            raise ValueError("group_shrinkage_rate must be positive")
        if self.binary_loading_metric_weight <= 0:
            raise ValueError("binary_loading_metric_weight must be positive")
        if self.min_log_variance >= self.max_log_variance:
            raise ValueError("min_log_variance must be smaller than max_log_variance")

    @property
    def num_blood_features(self) -> int:
        return self.num_continuous + self.num_binary


@dataclass
class GaussianState:
    mean: torch.Tensor
    log_variance: torch.Tensor

    @property
    def variance(self) -> torch.Tensor:
        return self.log_variance.exp()

    @property
    def precision(self) -> torch.Tensor:
        return (-self.log_variance).exp()


@dataclass
class EvidenceFactor:
    center: torch.Tensor
    precision_increment: torch.Tensor
    reliability: torch.Tensor
    present: torch.Tensor
    summary: torch.Tensor

    @property
    def natural_parameter(self) -> torch.Tensor:
        return self.precision_increment * self.center


@dataclass
class EyePrediction:
    mean: torch.Tensor
    log_variance: torch.Tensor


@dataclass
class ClinicalPrediction:
    continuous_location: torch.Tensor
    continuous_log_scale: torch.Tensor
    binary_logits: torch.Tensor
    blood_anchor_mean: torch.Tensor
    blood_anchor_log_variance: torch.Tensor


@dataclass
class PatientStateOutput:
    physiology: GaussianState
    demographic_context: torch.Tensor
    demographic_mask: torch.Tensor
    eye: EyePrediction
    clinical: ClinicalPrediction
    eye_evidence: EvidenceFactor
    blood_evidence: EvidenceFactor
    interaction_precision: torch.Tensor
    interaction_natural_parameter: torch.Tensor
    blood_eligible_mask: torch.Tensor
    availability: Dict[str, torch.Tensor]
    abstain: torch.Tensor

    @property
    def vector(self) -> torch.Tensor:
        """Default fixed vector; uncertainty and availability remain sidecars."""
        return torch.cat([self.physiology.mean, self.demographic_context], dim=-1)

    @property
    def z(self) -> torch.Tensor:
        """Compatibility alias for existing fold-evaluation code."""
        return self.physiology.mean


@dataclass
class PatientEvidenceOutput:
    """Patient state before observable decoders are evaluated.

    This is the deployment path for vector extraction. It is mathematically
    identical to the evidence portion of :class:`PatientStateOutput` but avoids
    allocating retinal and clinical reconstruction tensors.
    """

    physiology: GaussianState
    demographic_context: torch.Tensor
    demographic_mask: torch.Tensor
    eye_evidence: EvidenceFactor
    blood_evidence: EvidenceFactor
    interaction_precision: torch.Tensor
    interaction_natural_parameter: torch.Tensor
    blood_eligible_mask: torch.Tensor
    availability: Dict[str, torch.Tensor]
    abstain: torch.Tensor

    @property
    def vector(self) -> torch.Tensor:
        return torch.cat([self.physiology.mean, self.demographic_context], dim=-1)

    @property
    def z(self) -> torch.Tensor:
        return self.physiology.mean


class _ZeroAtOriginMLP(nn.Module):
    """Bias-free MLP, guaranteeing f(0)=0."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.first = nn.Linear(d_in, d_hidden, bias=False)
        self.second = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(F.gelu(self.first(x)))


class EyeEvidenceEncoder(nn.Module):
    """Permutation-invariant retinal-set evidence encoder.

    Device, image count, and quality affect reliability only.  They cannot enter
    the evidence-centre head and therefore cannot directly move physiological
    coordinates.
    """

    def __init__(self, config: PatientAtlasConfig) -> None:
        super().__init__()
        c = config
        self.config = c
        self.value_projection = nn.Sequential(
            # Inputs are already whitened inside the outer training fold.
            # Per-image LayerNorm would erase the embedding's mean and radial
            # components, making otherwise recoverable patient factors
            # unidentifiable.  Keep the slot as Identity so checkpoint/module
            # indexing remains stable while preserving all frozen-tower signal.
            nn.Identity(),
            nn.Linear(c.eye_dim, c.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(c.hidden_dim, c.hidden_dim, bias=False),
        )
        statistic_dim = 2 * c.hidden_dim
        self.center_head = _ZeroAtOriginMLP(
            statistic_dim + c.demographic_dim, c.hidden_dim, c.latent_dim
        )
        self.precision_head = nn.Sequential(
            nn.Linear(statistic_dim, c.hidden_dim),
            nn.GELU(),
            nn.Linear(c.hidden_dim, c.latent_dim),
        )
        reliability_dim = 2 + c.num_devices
        self.reliability_head = nn.Sequential(
            nn.Linear(reliability_dim, max(8, c.hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(8, c.hidden_dim // 2), 1),
        )

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
        batch, n_images, _ = eye_embeddings.shape
        dtype = eye_embeddings.dtype
        safe_eye = torch.where(
            visible_mask[..., None], eye_embeddings, torch.zeros_like(eye_embeddings)
        )
        # Laterality is a decoder query, not physiological evidence in v1.
        # It remains in the signature so an authenticated future schema can be
        # introduced without changing the patient-state API.
        del laterality_ids
        token = self.value_projection(safe_eye)
        token = torch.where(visible_mask[..., None], token, torch.zeros_like(token))

        count = visible_mask.sum(dim=1, keepdim=True)
        denom = count.clamp(min=1).to(dtype=dtype)
        mean = token.sum(dim=1) / denom
        second = token.square().sum(dim=1) / denom
        variance = (second - mean.square()).clamp(min=0)
        # sqrt has an infinite derivative at zero.  Clamp before taking the
        # root, then define singleton/empty-set dispersion to be exactly zero.
        dispersion = variance.clamp(min=1e-8).sqrt()
        dispersion = torch.where(count > 1, dispersion, torch.zeros_like(dispersion))
        present = count[:, 0] > 0
        mean = torch.where(present[:, None], mean, torch.zeros_like(mean))
        dispersion = torch.where(
            present[:, None], dispersion, torch.zeros_like(dispersion)
        )
        statistics = torch.cat([mean, dispersion], dim=-1)
        center = self.config.evidence_center_bound * torch.tanh(
            self.center_head(torch.cat([statistics, demographics], dim=-1))
        )
        center = torch.where(present[:, None], center, torch.zeros_like(center))

        safe_quality = torch.where(visible_mask, quality, torch.zeros_like(quality))
        mean_quality = safe_quality.sum(dim=1, keepdim=True) / denom
        safe_devices = torch.where(visible_mask, device_ids, torch.zeros_like(device_ids))
        device_hist = F.one_hot(
            safe_devices, num_classes=self.config.num_devices
        ).to(dtype=dtype)
        device_hist = (device_hist * visible_mask[..., None]).sum(dim=1) / denom
        log_count = torch.log1p(count.to(dtype=dtype)) / torch.log(
            torch.tensor(33.0, dtype=dtype, device=eye_embeddings.device)
        )
        reliability_features = torch.cat(
            [log_count, mean_quality, device_hist], dim=-1
        )
        reliability = torch.sigmoid(self.reliability_head(reliability_features))
        reliability = torch.where(
            present[:, None], reliability, torch.zeros_like(reliability)
        )
        raw_precision = F.softplus(self.precision_head(statistics))
        precision = raw_precision * reliability * factor_support[None, :]
        precision = torch.where(
            present[:, None], precision, torch.zeros_like(precision)
        )
        return EvidenceFactor(center, precision, reliability, present, mean)


class BloodEvidenceEncoder(nn.Module):
    """Value-bearing analyte-set evidence encoder.

    Presence without a value can affect precision, but it cannot move the
    evidence centre: the signal path is bias-free and exactly zero for a
    standardized value of zero.  The frozen-tower anchor is residualized against
    the same mask with all values erased before it enters the mean path.
    """

    def __init__(self, config: PatientAtlasConfig) -> None:
        super().__init__()
        c = config
        self.config = c
        self.value_direction = nn.Embedding(c.num_blood_features, c.hidden_dim)
        nn.init.normal_(self.value_direction.weight, std=0.02)
        self.anchor_projection = nn.Linear(
            c.blood_anchor_dim, c.hidden_dim, bias=False
        )
        self.center_head = _ZeroAtOriginMLP(
            c.hidden_dim + c.demographic_dim, c.hidden_dim, c.latent_dim
        )
        self.precision_head = nn.Sequential(
            nn.Linear(2 * c.hidden_dim, c.hidden_dim),
            nn.GELU(),
            nn.Linear(c.hidden_dim, c.latent_dim),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(3, max(8, c.hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(8, c.hidden_dim // 2), 1),
        )

    def forward(
        self,
        values: torch.Tensor,
        visible_mask: torch.Tensor,
        *,
        anchor_residual: torch.Tensor,
        factor_support: torch.Tensor,
        demographics: torch.Tensor,
        eligible_mask: torch.Tensor,
    ) -> EvidenceFactor:
        dtype = values.dtype
        batch, n_features = values.shape
        ids = torch.arange(n_features, device=values.device)
        directions = self.value_direction(ids)[None, :, :]
        continuous = values[:, : self.config.num_continuous]
        binary = values[:, self.config.num_continuous :]
        typed_values = torch.cat([continuous, binary.mul(2.0).sub(1.0)], dim=1)
        safe_values = torch.where(
            visible_mask, typed_values, torch.zeros_like(typed_values)
        )
        contributions = safe_values[..., None] * directions
        contributions = torch.where(
            visible_mask[..., None], contributions, torch.zeros_like(contributions)
        )
        count = visible_mask.sum(dim=1, keepdim=True)
        scale = count.clamp(min=1).to(dtype=dtype).sqrt()
        signed_signal = contributions.sum(dim=1) / scale
        magnitude_signal = contributions.abs().sum(dim=1) / scale
        combined = signed_signal + self.anchor_projection(anchor_residual)
        present = count[:, 0] > 0
        combined = torch.where(
            present[:, None], combined, torch.zeros_like(combined)
        )
        center = self.config.evidence_center_bound * torch.tanh(
            self.center_head(torch.cat([combined, demographics], dim=-1))
        )
        center = torch.where(present[:, None], center, torch.zeros_like(center))

        continuous_count = visible_mask[:, : self.config.num_continuous].sum(
            dim=1, keepdim=True
        )
        binary_count = visible_mask[:, self.config.num_continuous :].sum(
            dim=1, keepdim=True
        )
        eligible_count = eligible_mask.sum(dim=1, keepdim=True).clamp(min=1)
        eligible_continuous = eligible_mask[:, : self.config.num_continuous].sum(
            dim=1, keepdim=True
        ).clamp(min=1)
        eligible_binary = eligible_mask[:, self.config.num_continuous :].sum(
            dim=1, keepdim=True
        ).clamp(min=1)
        coverage = torch.cat(
            [
                count.to(dtype=dtype) / eligible_count.to(dtype=dtype),
                continuous_count.to(dtype=dtype)
                / eligible_continuous.to(dtype=dtype),
                binary_count.to(dtype=dtype) / eligible_binary.to(dtype=dtype),
            ],
            dim=-1,
        )
        reliability = torch.sigmoid(self.reliability_head(coverage))
        reliability = torch.where(
            present[:, None], reliability, torch.zeros_like(reliability)
        )
        precision_input = torch.cat([combined, magnitude_signal], dim=-1)
        raw_precision = F.softplus(self.precision_head(precision_input))
        precision = raw_precision * reliability * factor_support[None, :]
        precision = torch.where(
            present[:, None], precision, torch.zeros_like(precision)
        )
        return EvidenceFactor(center, precision, reliability, present, combined)


class LowRankInteractionEvidence(nn.Module):
    """Manually unlocked, quality-gated, both-present correction."""

    def __init__(self, config: PatientAtlasConfig) -> None:
        super().__init__()
        c = config
        self.eye_projection = nn.Linear(
            c.hidden_dim, c.interaction_rank, bias=False
        )
        self.blood_projection = nn.Linear(
            c.hidden_dim, c.interaction_rank, bias=False
        )
        self.to_precision_logit = nn.Linear(
            c.interaction_rank, c.latent_dim, bias=True
        )
        self.to_center = nn.Linear(c.interaction_rank, c.latent_dim, bias=False)
        nn.init.zeros_(self.to_precision_logit.weight)
        nn.init.zeros_(self.to_precision_logit.bias)
        self.raw_gate = nn.Parameter(torch.zeros(c.latent_dim))
        self.register_buffer("unlocked", torch.tensor(False, dtype=torch.bool))
        self.center_bound = 3.0

    @torch.no_grad()
    def unlock(self, initial_gate: float = 1e-2) -> None:
        """Move off the zero-gradient PSD boundary for the Stage-3 challenge."""
        if not 0 < initial_gate <= 0.1:
            raise ValueError("initial_gate must be in (0, 0.1]")
        self.raw_gate.fill_(initial_gate)
        self.unlocked.fill_(True)

    @torch.no_grad()
    def lock(self) -> None:
        """Return exactly to the additive model."""
        self.raw_gate.zero_()
        self.unlocked.fill_(False)

    def forward(
        self,
        eye_summary: torch.Tensor,
        blood_summary: torch.Tensor,
        both_present: torch.Tensor,
        eye_reliability: torch.Tensor,
        blood_reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        interaction = self.eye_projection(eye_summary) * self.blood_projection(
            blood_summary
        )
        reliability = torch.minimum(eye_reliability, blood_reliability)
        reliability = torch.where(
            both_present[:, None], reliability, torch.zeros_like(reliability)
        )
        precision = (
            self.unlocked.to(dtype=interaction.dtype)
            * self.raw_gate.square()[None, :]
            * F.softplus(self.to_precision_logit(interaction))
            * reliability
        )
        center = self.center_bound * torch.tanh(self.to_center(interaction))
        precision = torch.where(
            both_present[:, None], precision, torch.zeros_like(precision)
        )
        natural = precision * center
        natural = torch.where(
            both_present[:, None], natural, torch.zeros_like(natural)
        )
        return precision, natural


class SoftGroupFactorDecoder(nn.Module):
    """Typed decoders with one scale-identified amplitude per factor and view.

    Each loading column is an L2-normalized direction multiplied by exactly one
    nonnegative amplitude.  This removes the scale exchange that made the older
    loading-plus-ARD-gate formulation unidentified.  Clinical directions use
    a fixed likelihood-metric norm; the optional anchor has a separate
    direction and therefore cannot distort clinical factor amplitudes.
    """

    def __init__(self, config: PatientAtlasConfig) -> None:
        super().__init__()
        c = config
        self.config = c
        self.eye_raw_direction = nn.Parameter(
            torch.empty(c.eye_dim, c.latent_dim)
        )
        self.clinical_raw_direction = nn.Parameter(
            torch.empty(c.num_blood_features, c.latent_dim)
        )
        self.anchor_raw_direction = nn.Parameter(
            torch.empty(c.blood_anchor_dim, c.latent_dim)
        )
        nn.init.normal_(self.eye_raw_direction, std=0.02)
        nn.init.normal_(self.clinical_raw_direction, std=0.02)
        nn.init.normal_(self.anchor_raw_direction, std=0.02)
        initial_raw_amplitude = float(torch.log(torch.expm1(torch.tensor(0.25))))
        self.eye_raw_amplitude = nn.Parameter(
            torch.full((c.latent_dim,), initial_raw_amplitude)
        )
        self.blood_raw_amplitude = nn.Parameter(
            torch.full((c.latent_dim,), initial_raw_amplitude)
        )

        self.eye_intercept = nn.Parameter(torch.zeros(c.eye_dim))
        self.continuous_intercept = nn.Parameter(torch.zeros(c.num_continuous))
        self.binary_intercept = nn.Parameter(torch.zeros(c.num_binary))
        self.anchor_intercept = nn.Parameter(torch.zeros(c.blood_anchor_dim))
        self.eye_log_noise = nn.Parameter(torch.zeros(c.eye_dim))
        self.eye_device_log_noise = nn.Embedding(c.num_devices, c.eye_dim)
        self.eye_laterality_mean = nn.Embedding(c.num_lateralities, c.eye_dim)
        nn.init.zeros_(self.eye_device_log_noise.weight)
        nn.init.zeros_(self.eye_laterality_mean.weight)
        self.continuous_log_scale = nn.Parameter(torch.zeros(c.num_continuous))
        self.anchor_log_noise = nn.Parameter(torch.zeros(c.blood_anchor_dim))

        if c.demographic_dim:
            self.eye_demographic = nn.Linear(c.demographic_dim, c.eye_dim, bias=False)
            self.continuous_demographic = nn.Linear(
                c.demographic_dim, c.num_continuous, bias=False
            )
            self.binary_demographic = nn.Linear(
                c.demographic_dim, c.num_binary, bias=False
            )
            self.anchor_demographic = nn.Linear(
                c.demographic_dim, c.blood_anchor_dim, bias=False
            )
        else:
            self.eye_demographic = None
            self.continuous_demographic = None
            self.binary_demographic = None
            self.anchor_demographic = None
        self.renormalize_loading_directions_()

    @property
    def eye_amplitude(self) -> torch.Tensor:
        return F.softplus(self.eye_raw_amplitude)

    @property
    def blood_amplitude(self) -> torch.Tensor:
        return F.softplus(self.blood_raw_amplitude)

    @property
    def eye_factor_support(self) -> torch.Tensor:
        squared = self.eye_amplitude.square()
        return squared / (1.0 + squared)

    @property
    def blood_factor_support(self) -> torch.Tensor:
        squared = self.blood_amplitude.square()
        return squared / (1.0 + squared)

    @property
    def eye_loading(self) -> torch.Tensor:
        direction = F.normalize(self.eye_raw_direction, p=2, dim=0, eps=1e-8)
        return direction * self.eye_amplitude[None, :]

    @property
    def blood_loading(self) -> torch.Tensor:
        raw = self.clinical_raw_direction
        weights = torch.ones(
            raw.shape[0], dtype=raw.dtype, device=raw.device
        )
        weights[self.config.num_continuous :] = (
            self.config.binary_loading_metric_weight
        )
        norm = (raw.square() * weights[:, None]).sum(dim=0).clamp(min=1e-12).sqrt()
        direction = raw / norm[None, :]
        return direction * self.blood_amplitude[None, :]

    @property
    def anchor_loading(self) -> torch.Tensor:
        direction = F.normalize(self.anchor_raw_direction, p=2, dim=0, eps=1e-8)
        return direction * self.blood_amplitude[None, :]

    def _split_blood_loading(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nc = self.config.num_continuous
        loading = self.blood_loading
        return loading[:nc], loading[nc:], self.anchor_loading

    @torch.no_grad()
    def renormalize_loading_directions_(self) -> None:
        """Project raw directions to their unit manifolds after optimizer steps."""
        self.eye_raw_direction.copy_(
            F.normalize(self.eye_raw_direction, p=2, dim=0, eps=1e-8)
        )
        raw = self.clinical_raw_direction
        weights = torch.ones(raw.shape[0], dtype=raw.dtype, device=raw.device)
        weights[self.config.num_continuous :] = (
            self.config.binary_loading_metric_weight
        )
        norm = (raw.square() * weights[:, None]).sum(dim=0).clamp(min=1e-12).sqrt()
        raw.div_(norm[None, :])
        self.anchor_raw_direction.copy_(
            F.normalize(self.anchor_raw_direction, p=2, dim=0, eps=1e-8)
        )

    @staticmethod
    def _latent_variance(
        state_variance: torch.Tensor, effective_loading: torch.Tensor
    ) -> torch.Tensor:
        return state_variance @ effective_loading.square().T

    def group_shrinkage_regularizer(self) -> torch.Tensor:
        """MAP penalty for fixed-rate exponential priors on view amplitudes.

        A fixed-rate prior is intentional: a learned half-Cauchy hierarchy under
        joint MAP has a zero-scale singularity.  The rate is selected using only
        outer-training validation data.
        """
        amplitudes = torch.cat([self.eye_amplitude, self.blood_amplitude])
        return self.config.group_shrinkage_rate * amplitudes.sum()

    def orthogonality_regularizer(self) -> torch.Tensor:
        """Weakly separate active loading directions without forcing factor use."""
        clinical_loading = self.blood_loading.clone()
        clinical_loading[self.config.num_continuous :] = (
            clinical_loading[self.config.num_continuous :]
            * self.config.binary_loading_metric_weight**0.5
        )
        penalties = []
        for loading in (self.eye_loading, clinical_loading):
            gram = loading.T @ loading
            off_diagonal = gram - torch.diag_embed(torch.diagonal(gram))
            penalties.append(off_diagonal.square().mean())
        return torch.stack(penalties).mean()

    def forward(
        self,
        state: GaussianState,
        demographics: torch.Tensor,
        *,
        eye_device_ids: torch.Tensor,
        eye_laterality_ids: torch.Tensor,
    ) -> tuple[EyePrediction, ClinicalPrediction]:
        """Approximate observable posterior-predictive distributions."""
        return self._decode(
            state.mean,
            state.variance,
            demographics,
            eye_device_ids=eye_device_ids,
            eye_laterality_ids=eye_laterality_ids,
        )

    def conditional(
        self,
        z: torch.Tensor,
        demographics: torch.Tensor,
        *,
        eye_device_ids: torch.Tensor,
        eye_laterality_ids: torch.Tensor,
    ) -> tuple[EyePrediction, ClinicalPrediction]:
        """Exact conditional likelihood parameters p(x | z, context)."""
        return self._decode(
            z,
            torch.zeros_like(z),
            demographics,
            eye_device_ids=eye_device_ids,
            eye_laterality_ids=eye_laterality_ids,
        )

    def _decode(
        self,
        state_mean: torch.Tensor,
        state_variance: torch.Tensor,
        demographics: torch.Tensor,
        *,
        eye_device_ids: torch.Tensor,
        eye_laterality_ids: torch.Tensor,
    ) -> tuple[EyePrediction, ClinicalPrediction]:
        eye_loading = self.eye_loading
        continuous_loading, binary_loading, anchor_loading = (
            self._split_blood_loading()
        )

        patient_eye_mean = F.linear(state_mean, eye_loading, self.eye_intercept)
        continuous_location = F.linear(
            state_mean, continuous_loading, self.continuous_intercept
        )
        binary_location = F.linear(
            state_mean, binary_loading, self.binary_intercept
        )
        anchor_mean = F.linear(state_mean, anchor_loading, self.anchor_intercept)
        if self.config.demographic_dim:
            patient_eye_mean = patient_eye_mean + self.eye_demographic(demographics)
            continuous_location = continuous_location + self.continuous_demographic(
                demographics
            )
            binary_location = binary_location + self.binary_demographic(demographics)
            anchor_mean = anchor_mean + self.anchor_demographic(demographics)

        eye_mean = patient_eye_mean[:, None, :] + self.eye_laterality_mean(
            eye_laterality_ids
        )
        latent_eye_variance = self._latent_variance(
            state_variance, eye_loading
        )[:, None, :]
        eye_log_std = self.eye_log_noise[None, None, :] + self.eye_device_log_noise(
            eye_device_ids
        )
        eye_noise_variance = eye_log_std.clamp(min=-8.0, max=4.0).mul(2).exp()
        eye_variance = latent_eye_variance + eye_noise_variance

        latent_continuous_variance = self._latent_variance(
            state_variance, continuous_loading
        )
        base_student_scale_sq = self.continuous_log_scale.mul(2).exp()[None, :]
        student_variance_multiplier = self.config.student_df / (
            self.config.student_df - 2.0
        )
        total_continuous_variance = (
            base_student_scale_sq * student_variance_multiplier
            + latent_continuous_variance
        )
        predictive_student_scale_sq = (
            total_continuous_variance / student_variance_multiplier
        )

        binary_latent_variance = self._latent_variance(
            state_variance, binary_loading
        )
        binary_logits = binary_location / torch.sqrt(
            1.0 + torch.pi * binary_latent_variance / 8.0
        )

        anchor_variance = self.anchor_log_noise.mul(2).exp()[None, :]
        anchor_variance = anchor_variance + self._latent_variance(
            state_variance, anchor_loading
        )

        eye = EyePrediction(
            eye_mean,
            eye_variance.clamp(min=1e-8).log().clamp(
                min=self.config.min_log_variance,
                max=self.config.max_log_variance,
            ),
        )
        clinical = ClinicalPrediction(
            continuous_location=continuous_location,
            continuous_log_scale=0.5
            * predictive_student_scale_sq.clamp(min=1e-8).log(),
            binary_logits=binary_logits,
            blood_anchor_mean=anchor_mean,
            blood_anchor_log_variance=anchor_variance.clamp(min=1e-8).log(),
        )
        return eye, clinical


class SoftPatientAtlas(nn.Module):
    """One posterior patient space with soft view relevance and missingness."""

    def __init__(
        self,
        config: PatientAtlasConfig,
        blood_tower: nn.Module,
        *,
        blood_anchor_eligible_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        if not hasattr(blood_tower, "encode"):
            raise TypeError("blood_tower must expose encode(values, observed_mask)")
        if (
            blood_anchor_eligible_mask.shape != (config.num_blood_features,)
            or blood_anchor_eligible_mask.dtype != torch.bool
        ):
            raise TypeError(
                "blood_anchor_eligible_mask must be boolean with shape [features]"
            )
        self.config = config
        self.blood_tower = blood_tower
        self.blood_tower.requires_grad_(False)
        self.blood_tower.eval()
        self.register_buffer(
            "blood_anchor_eligible_mask",
            blood_anchor_eligible_mask.detach().clone(),
        )

        self.eye_encoder = EyeEvidenceEncoder(config)
        self.blood_encoder = BloodEvidenceEncoder(config)
        self.interaction = LowRankInteractionEvidence(config)
        self.decoder = SoftGroupFactorDecoder(config)

    def train(self, mode: bool = True) -> "SoftPatientAtlas":
        super().train(mode)
        self.blood_tower.eval()
        return self

    def optimizer_parameter_groups(self, weight_decay: float) -> list[dict[str, object]]:
        """AdamW groups preserving unit directions and the explicit amplitude prior."""
        if weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        geometry_parameters = [
            self.decoder.eye_raw_direction,
            self.decoder.clinical_raw_direction,
            self.decoder.anchor_raw_direction,
            self.decoder.eye_raw_amplitude,
            self.decoder.blood_raw_amplitude,
        ]
        geometry_ids = {id(parameter) for parameter in geometry_parameters}
        regular = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in geometry_ids
        ]
        return [
            {"params": regular, "weight_decay": float(weight_decay)},
            {"params": geometry_parameters, "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def project_parameters_(self) -> None:
        """Restore unit loading directions after every optimizer update."""
        self.decoder.renormalize_loading_directions_()

    def unlock_interaction(self, initial_gate: float = 1e-2) -> None:
        self.interaction.unlock(initial_gate)

    def lock_interaction(self) -> None:
        self.interaction.lock()

    def _infer_evidence_with_metadata(
        self,
        *,
        eye_embeddings: torch.Tensor,
        eye_visible_mask: torch.Tensor,
        blood_values: torch.Tensor,
        blood_visible_mask: torch.Tensor,
        blood_eligible_mask: torch.Tensor,
        demographics: torch.Tensor,
        demographic_mask: torch.Tensor,
        eye_device_ids: torch.Tensor | None = None,
        eye_laterality_ids: torch.Tensor | None = None,
        eye_quality: torch.Tensor | None = None,
        enable_interaction: bool = True,
    ) -> tuple[PatientEvidenceOutput, torch.Tensor, torch.Tensor]:
        self._validate_inputs(
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
        )
        batch, n_images, _ = eye_embeddings.shape
        device = eye_embeddings.device
        if eye_device_ids is None:
            eye_device_ids = torch.zeros(
                batch, n_images, dtype=torch.long, device=device
            )
        if eye_laterality_ids is None:
            eye_laterality_ids = torch.full(
                (batch, n_images),
                self.config.num_lateralities - 1,
                dtype=torch.long,
                device=device,
            )
        if eye_quality is None:
            eye_quality = torch.ones(
                batch, n_images, dtype=eye_embeddings.dtype, device=device
            )

        safe_eye_device_ids = torch.where(
            eye_visible_mask, eye_device_ids, torch.zeros_like(eye_device_ids)
        )
        safe_eye_laterality_ids = torch.where(
            eye_visible_mask,
            eye_laterality_ids,
            torch.full_like(
                eye_laterality_ids, self.config.num_lateralities - 1
            ),
        )

        safe_demographics = torch.where(
            demographic_mask, demographics, torch.zeros_like(demographics)
        )
        safe_blood = torch.where(
            blood_visible_mask, blood_values, torch.zeros_like(blood_values)
        )
        anchor_residual = self._residualized_blood_anchor(
            safe_blood, blood_visible_mask
        )
        eye_evidence = self.eye_encoder(
            eye_embeddings,
            eye_visible_mask,
            factor_support=self.decoder.eye_factor_support,
            demographics=safe_demographics,
            device_ids=safe_eye_device_ids,
            laterality_ids=safe_eye_laterality_ids,
            quality=eye_quality,
        )
        blood_evidence = self.blood_encoder(
            safe_blood,
            blood_visible_mask,
            anchor_residual=anchor_residual,
            factor_support=self.decoder.blood_factor_support,
            demographics=safe_demographics,
            eligible_mask=blood_eligible_mask,
        )

        both_present = eye_evidence.present & blood_evidence.present
        if enable_interaction:
            interaction_precision, interaction_natural = self.interaction(
                eye_evidence.summary,
                blood_evidence.summary,
                both_present,
                eye_evidence.reliability,
                blood_evidence.reliability,
            )
        else:
            shape = (batch, self.config.latent_dim)
            interaction_precision = torch.zeros(
                shape, dtype=eye_embeddings.dtype, device=device
            )
            interaction_natural = torch.zeros_like(interaction_precision)
        state = self.fuse_evidence(
            eye_evidence,
            blood_evidence,
            interaction_precision=interaction_precision,
            interaction_natural_parameter=interaction_natural,
            min_log_variance=self.config.min_log_variance,
            max_log_variance=self.config.max_log_variance,
        )
        availability = {
            "eye_present": eye_evidence.present,
            "blood_clinical_present": blood_evidence.present,
            "eye_count": eye_visible_mask.sum(dim=1),
            "blood_clinical_count": blood_visible_mask.sum(dim=1),
            "blood_clinical_eligible_count": blood_eligible_mask.sum(dim=1),
        }
        abstain = ~(eye_evidence.present | blood_evidence.present)
        evidence = PatientEvidenceOutput(
            physiology=state,
            demographic_context=safe_demographics,
            demographic_mask=demographic_mask,
            eye_evidence=eye_evidence,
            blood_evidence=blood_evidence,
            interaction_precision=interaction_precision,
            interaction_natural_parameter=interaction_natural,
            blood_eligible_mask=blood_eligible_mask,
            availability=availability,
            abstain=abstain,
        )
        return evidence, safe_eye_device_ids, safe_eye_laterality_ids

    def infer_evidence(
        self,
        *,
        eye_embeddings: torch.Tensor,
        eye_visible_mask: torch.Tensor,
        blood_values: torch.Tensor,
        blood_visible_mask: torch.Tensor,
        blood_eligible_mask: torch.Tensor,
        demographics: torch.Tensor,
        demographic_mask: torch.Tensor,
        eye_device_ids: torch.Tensor | None = None,
        eye_laterality_ids: torch.Tensor | None = None,
        eye_quality: torch.Tensor | None = None,
        enable_interaction: bool = True,
    ) -> PatientEvidenceOutput:
        """Infer the probabilistic state without running observable decoders."""

        evidence, _, _ = self._infer_evidence_with_metadata(
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
            enable_interaction=enable_interaction,
        )
        return evidence

    def forward(
        self,
        *,
        eye_embeddings: torch.Tensor,
        eye_visible_mask: torch.Tensor,
        blood_values: torch.Tensor,
        blood_visible_mask: torch.Tensor,
        blood_eligible_mask: torch.Tensor,
        demographics: torch.Tensor,
        demographic_mask: torch.Tensor,
        eye_device_ids: torch.Tensor | None = None,
        eye_laterality_ids: torch.Tensor | None = None,
        eye_quality: torch.Tensor | None = None,
        enable_interaction: bool = True,
    ) -> PatientStateOutput:
        evidence, safe_eye_device_ids, safe_eye_laterality_ids = (
            self._infer_evidence_with_metadata(
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
                enable_interaction=enable_interaction,
            )
        )
        eye_prediction, clinical_prediction = self.decoder(
            evidence.physiology,
            evidence.demographic_context,
            eye_device_ids=safe_eye_device_ids,
            eye_laterality_ids=safe_eye_laterality_ids,
        )
        return PatientStateOutput(
            physiology=evidence.physiology,
            demographic_context=evidence.demographic_context,
            demographic_mask=evidence.demographic_mask,
            eye=eye_prediction,
            clinical=clinical_prediction,
            eye_evidence=evidence.eye_evidence,
            blood_evidence=evidence.blood_evidence,
            interaction_precision=evidence.interaction_precision,
            interaction_natural_parameter=evidence.interaction_natural_parameter,
            blood_eligible_mask=evidence.blood_eligible_mask,
            availability=evidence.availability,
            abstain=evidence.abstain,
        )

    @staticmethod
    def fuse_evidence(
        eye: EvidenceFactor,
        blood: EvidenceFactor,
        *,
        interaction_precision: torch.Tensor,
        interaction_natural_parameter: torch.Tensor,
        min_log_variance: float = -10.0,
        max_log_variance: float = 4.0,
    ) -> GaussianState:
        if eye.center.shape != blood.center.shape:
            raise ValueError("eye and blood evidence widths must match")
        expected = eye.center.shape
        tensors = (
            eye.precision_increment,
            blood.precision_increment,
            interaction_precision,
            interaction_natural_parameter,
        )
        if any(tensor.shape != expected for tensor in tensors):
            raise ValueError("all evidence tensors must share [batch, latent] shape")
        if (eye.precision_increment < 0).any() or (
            blood.precision_increment < 0
        ).any() or (interaction_precision < 0).any():
            raise ValueError("precision increments must be nonnegative")
        prior_precision = torch.ones_like(eye.center)
        precision = (
            prior_precision
            + eye.precision_increment
            + blood.precision_increment
            + interaction_precision
        )
        natural = (
            eye.natural_parameter
            + blood.natural_parameter
            + interaction_natural_parameter
        )
        mean = natural / precision
        log_variance = -precision.log()
        log_variance = log_variance.clamp(
            min=min_log_variance, max=max_log_variance
        )
        return GaussianState(mean, log_variance)

    def _validated_eye_target_metadata(
        self,
        *,
        target_eye_mask: torch.Tensor,
        target_eye_device_ids: torch.Tensor,
        target_eye_laterality_ids: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_eye_mask.ndim != 2 or target_eye_mask.dtype != torch.bool:
            raise TypeError("target_eye_mask must be boolean with shape [batch, images]")
        batch, n_images = target_eye_mask.shape
        metadata = (
            ("target_eye_device_ids", target_eye_device_ids, self.config.num_devices, 0),
            (
                "target_eye_laterality_ids",
                target_eye_laterality_ids,
                self.config.num_lateralities,
                self.config.num_lateralities - 1,
            ),
        )
        safe: list[torch.Tensor] = []
        for name, values, upper, default in metadata:
            if values.shape != (batch, n_images) or values.dtype != torch.long:
                raise TypeError(f"{name} must be long with shape [batch, images]")
            if values.device != reference.device:
                raise ValueError(f"{name} must share the model device")
            visible = values[target_eye_mask]
            if visible.numel() and (
                int(visible.min()) < 0 or int(visible.max()) >= upper
            ):
                raise ValueError(f"{name} contains an out-of-range target id")
            safe.append(
                torch.where(
                    target_eye_mask,
                    values,
                    torch.full_like(values, default),
                )
            )
        return safe[0], safe[1]

    def reconstruction_loss(
        self,
        output: PatientStateOutput,
        *,
        target_eye_embeddings: torch.Tensor,
        target_eye_mask: torch.Tensor,
        target_blood_values: torch.Tensor,
        target_blood_mask: torch.Tensor,
        target_blood_eligible_mask: torch.Tensor,
        target_eye_device_ids: torch.Tensor,
        target_eye_laterality_ids: torch.Tensor,
        target_blood_anchor: torch.Tensor | None = None,
        anchor_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        safe_device_ids, safe_laterality_ids = self._validated_eye_target_metadata(
            target_eye_mask=target_eye_mask,
            target_eye_device_ids=target_eye_device_ids,
            target_eye_laterality_ids=target_eye_laterality_ids,
            reference=output.physiology.mean,
        )
        eye_prediction, clinical_prediction = self.decoder(
            output.physiology,
            output.demographic_context,
            eye_device_ids=safe_device_ids,
            eye_laterality_ids=safe_laterality_ids,
        )
        return self._prediction_reconstruction_loss(
            eye_prediction=eye_prediction,
            clinical_prediction=clinical_prediction,
            state_reference=output.physiology.mean,
            blood_eligible_mask=output.blood_eligible_mask,
            target_eye_embeddings=target_eye_embeddings,
            target_eye_mask=target_eye_mask,
            target_blood_values=target_blood_values,
            target_blood_mask=target_blood_mask,
            target_blood_eligible_mask=target_blood_eligible_mask,
            target_blood_anchor=target_blood_anchor,
            anchor_weight=anchor_weight,
        )

    def _prediction_reconstruction_loss(
        self,
        *,
        eye_prediction: EyePrediction,
        clinical_prediction: ClinicalPrediction,
        state_reference: torch.Tensor,
        blood_eligible_mask: torch.Tensor,
        target_eye_embeddings: torch.Tensor,
        target_eye_mask: torch.Tensor,
        target_blood_values: torch.Tensor,
        target_blood_mask: torch.Tensor,
        target_blood_eligible_mask: torch.Tensor,
        target_blood_anchor: torch.Tensor | None = None,
        anchor_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Typed, patient- and view-normalized posterior-predictive losses.

        Target masks are complete observation masks and must remain separate
        from encoder corruption masks.  This function scores predictions; the
        training objective uses ``variational_loss`` below.
        """
        if not 0.0 <= float(anchor_weight) <= 0.25:
            raise ValueError("anchor_weight must lie in [0, 0.25]")
        if target_eye_embeddings.ndim != 3:
            raise ValueError("target_eye_embeddings must have shape [batch, images, dim]")
        if target_eye_mask.shape != target_eye_embeddings.shape[:2]:
            raise ValueError("target_eye_mask has the wrong shape")
        if target_blood_values.shape != target_blood_mask.shape:
            raise ValueError("target blood values and mask must match")
        if target_blood_eligible_mask.shape != target_blood_values.shape:
            raise ValueError("target_blood_eligible_mask has the wrong shape")
        if target_blood_values.shape[1] != self.config.num_blood_features:
            raise ValueError("target blood width does not match config")
        if (
            target_eye_mask.dtype != torch.bool
            or target_blood_mask.dtype != torch.bool
            or target_blood_eligible_mask.dtype != torch.bool
        ):
            raise TypeError("target masks must be boolean")
        if not torch.equal(target_blood_eligible_mask, blood_eligible_mask):
            raise ValueError("target and encoder blood policy masks differ")
        if bool((target_blood_mask & ~target_blood_eligible_mask).any()):
            raise ValueError("target blood mask must be policy eligible")
        target_values = (target_eye_embeddings, target_blood_values)
        if any(value.dtype != state_reference.dtype for value in target_values):
            raise TypeError("targets must share the model floating dtype")
        if any(value.device != state_reference.device for value in target_values):
            raise ValueError("targets must share the model device")
        if not bool(torch.isfinite(target_eye_embeddings[target_eye_mask]).all()):
            raise ValueError("visible eye targets must be finite")
        if not bool(torch.isfinite(target_blood_values[target_blood_mask]).all()):
            raise ValueError("visible blood targets must be finite")
        nc = self.config.num_continuous
        binary_target = target_blood_values[:, nc:]
        binary_mask = target_blood_mask[:, nc:]
        visible_binary = binary_target[binary_mask]
        if visible_binary.numel() and not bool(
            ((visible_binary == 0) | (visible_binary == 1)).all()
        ):
            raise ValueError("visible binary targets must be exactly 0 or 1")

        if eye_prediction.mean.shape != target_eye_embeddings.shape:
            raise ValueError("eye prediction and target shapes must match")
        safe_target_eye = torch.where(
            target_eye_mask[..., None],
            target_eye_embeddings,
            torch.zeros_like(target_eye_embeddings),
        )
        eye_mean = eye_prediction.mean
        eye_log_var = eye_prediction.log_variance
        eye_nll = 0.5 * (
            eye_log_var
            + (safe_target_eye - eye_mean).square() / eye_log_var.exp()
        )
        eye_item_nll = eye_nll.mean(dim=-1)
        eye_loss = self._masked_patient_mean(eye_item_nll, target_eye_mask)
        eye_patient, eye_present = self._masked_patient_values(
            eye_item_nll, target_eye_mask
        )

        continuous_target = target_blood_values[:, :nc]
        continuous_mask = target_blood_mask[:, :nc]
        safe_continuous_target = torch.where(
            continuous_mask, continuous_target, torch.zeros_like(continuous_target)
        )
        continuous_dist = torch.distributions.StudentT(
            df=torch.tensor(
                self.config.student_df,
                dtype=continuous_target.dtype,
                device=continuous_target.device,
            ),
            loc=clinical_prediction.continuous_location,
            scale=clinical_prediction.continuous_log_scale.exp(),
        )
        continuous_nll = -continuous_dist.log_prob(safe_continuous_target)
        continuous_loss = self._masked_patient_mean(
            continuous_nll, continuous_mask
        )

        safe_binary_target = torch.where(
            binary_mask, binary_target, torch.zeros_like(binary_target)
        )
        binary_nll = F.binary_cross_entropy_with_logits(
            clinical_prediction.binary_logits,
            safe_binary_target,
            reduction="none",
        )
        binary_loss = self._masked_patient_mean(binary_nll, binary_mask)

        blood_nll = torch.cat([continuous_nll, binary_nll], dim=-1)
        blood_patient, blood_present = self._masked_patient_values(
            blood_nll, target_blood_mask
        )
        anchor_present = (
            target_blood_mask & self.blood_anchor_eligible_mask[None, :]
        ).any(dim=1)
        view_count = eye_present.to(eye_patient.dtype) + blood_present.to(
            eye_patient.dtype
        )
        eligible_patient = view_count > 0
        if bool(eligible_patient.any()):
            reconstruction = (
                (eye_patient + blood_patient) / view_count.clamp(min=1.0)
            )[eligible_patient].mean()
        else:
            reconstruction = state_reference.sum() * 0.0

        anchor_loss = torch.zeros((), device=target_blood_values.device)
        if anchor_weight and target_blood_anchor is None:
            raise ValueError("target_blood_anchor is required when anchor_weight > 0")
        if target_blood_anchor is not None and anchor_weight:
            if target_blood_anchor.shape != clinical_prediction.blood_anchor_mean.shape:
                raise ValueError("target_blood_anchor has the wrong shape")
            if (
                target_blood_anchor.device != state_reference.device
                or target_blood_anchor.dtype != state_reference.dtype
            ):
                raise TypeError("target_blood_anchor must share model device and dtype")
            if not bool(torch.isfinite(target_blood_anchor[anchor_present]).all()):
                raise ValueError("eligible blood anchor targets must be finite")
            safe_anchor = torch.where(
                anchor_present[:, None],
                target_blood_anchor,
                torch.zeros_like(target_blood_anchor),
            )
            anchor_nll = 0.5 * (
                clinical_prediction.blood_anchor_log_variance
                + (safe_anchor - clinical_prediction.blood_anchor_mean).square()
                / clinical_prediction.blood_anchor_log_variance.exp()
            )
            if bool(anchor_present.any()):
                anchor_loss = anchor_nll.mean(dim=-1)[anchor_present].mean()
        total = reconstruction + anchor_weight * anchor_loss
        return {
            "total": total,
            "reconstruction": reconstruction,
            "eye": eye_loss,
            "blood": self._masked_patient_mean(blood_nll, target_blood_mask),
            "continuous": continuous_loss,
            "binary": binary_loss,
            "anchor": anchor_loss,
        }

    def variational_loss(
        self,
        output: PatientStateOutput,
        *,
        target_eye_embeddings: torch.Tensor,
        target_eye_mask: torch.Tensor,
        target_blood_values: torch.Tensor,
        target_blood_mask: torch.Tensor,
        target_blood_eligible_mask: torch.Tensor,
        target_eye_device_ids: torch.Tensor,
        target_eye_laterality_ids: torch.Tensor,
        target_blood_anchor: torch.Tensor | None = None,
        anchor_weight: float = 0.0,
        sample_count: int = 1,
        beta: float = 1.0,
        orthogonality_weight: float = 1e-3,
        generator: torch.Generator | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Balanced generalized variational objective with a conditional decoder.

        Reconstruction is evaluated under ``p(x | z, context)`` at
        reparameterized samples.  Posterior uncertainty is never inserted into
        the conditional observation noise, avoiding the common mistake of
        optimizing a posterior-predictive score as though it were a conditional
        likelihood.  Per-patient, per-view normalization makes this a tempered
        composite-likelihood objective rather than the literal ELBO of all
        independent coordinates; ``beta`` is therefore a validated balance
        hyperparameter, not a claim of exact Bayesian posterior scaling.
        """
        if not isinstance(sample_count, int) or sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")
        if beta < 0 or orthogonality_weight < 0:
            raise ValueError("beta and orthogonality_weight must be nonnegative")
        batch = output.physiology.mean.shape[0]
        if target_eye_embeddings.ndim != 3 or target_eye_embeddings.shape[0] != batch:
            raise ValueError("target_eye_embeddings has the wrong batch shape")
        n_images = target_eye_embeddings.shape[1]
        if target_eye_mask.shape != (batch, n_images):
            raise ValueError("target_eye_mask has the wrong shape")
        safe_device_ids, safe_laterality_ids = self._validated_eye_target_metadata(
            target_eye_mask=target_eye_mask,
            target_eye_device_ids=target_eye_device_ids,
            target_eye_laterality_ids=target_eye_laterality_ids,
            reference=output.physiology.mean,
        )

        mean = output.physiology.mean
        standard_deviation = output.physiology.log_variance.mul(0.5).exp()
        epsilon = torch.randn(
            (sample_count, *mean.shape),
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        sampled_z = mean[None, :, :] + standard_deviation[None, :, :] * epsilon

        def repeat_samples(tensor: torch.Tensor) -> torch.Tensor:
            expanded = tensor.unsqueeze(0).expand(sample_count, *tensor.shape)
            return expanded.reshape(sample_count * batch, *tensor.shape[1:])

        z_flat = sampled_z.reshape(sample_count * batch, self.config.latent_dim)
        demographics = repeat_samples(output.demographic_context)
        device_ids = repeat_samples(safe_device_ids)
        laterality_ids = repeat_samples(safe_laterality_ids)
        eye_prediction, clinical_prediction = self.decoder.conditional(
            z_flat,
            demographics,
            eye_device_ids=device_ids,
            eye_laterality_ids=laterality_ids,
        )
        repeated_anchor = (
            None
            if target_blood_anchor is None
            else repeat_samples(target_blood_anchor)
        )
        likelihood = self._prediction_reconstruction_loss(
            eye_prediction=eye_prediction,
            clinical_prediction=clinical_prediction,
            state_reference=z_flat,
            blood_eligible_mask=repeat_samples(output.blood_eligible_mask),
            target_eye_embeddings=repeat_samples(target_eye_embeddings),
            target_eye_mask=repeat_samples(target_eye_mask),
            target_blood_values=repeat_samples(target_blood_values),
            target_blood_mask=repeat_samples(target_blood_mask),
            target_blood_eligible_mask=repeat_samples(target_blood_eligible_mask),
            target_blood_anchor=repeated_anchor,
            anchor_weight=anchor_weight,
        )
        kl = self.kl_to_standard_normal(output.physiology)
        group = self.group_shrinkage_regularizer()
        orthogonality = self.orthogonality_regularizer()
        total = (
            likelihood["total"]
            + float(beta) * kl
            + group
            + float(orthogonality_weight) * orthogonality
        )
        result = {f"likelihood_{name}": value for name, value in likelihood.items()}
        result.update(
            {
                "total": total,
                "kl": kl,
                "group_shrinkage": group,
                "orthogonality": orthogonality,
            }
        )
        return result

    def group_shrinkage_regularizer(self) -> torch.Tensor:
        """Return the sole model-level loading shrinkage penalty."""
        return self.decoder.group_shrinkage_regularizer()

    def orthogonality_regularizer(self) -> torch.Tensor:
        return self.decoder.orthogonality_regularizer()

    @staticmethod
    def kl_to_standard_normal(state: GaussianState) -> torch.Tensor:
        """Analytic KL averaged across patients and latent coordinates."""
        per_factor = (
            state.variance
            + state.mean.square()
            - 1.0
            - state.log_variance
        )
        return 0.5 * per_factor.mean(dim=-1).mean()

    def factor_amplitude_profile(self) -> torch.Tensor:
        """Return descriptive eye/blood amplitude shares with shape [latent, 2].

        This is a model diagnostic, not the release interpretation statistic.
        Named clinical relevance must use held-out likelihood deviance reduction
        or expected Fisher information with uncertainty.
        """
        eye_energy = self.decoder.eye_amplitude.square()
        blood_energy = self.decoder.blood_amplitude.square()
        total = (eye_energy + blood_energy).clamp(min=1e-12)
        return torch.stack([eye_energy / total, blood_energy / total], dim=-1)

    @staticmethod
    def _masked_patient_values(
        values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe = torch.where(mask, values, torch.zeros_like(values))
        count = mask.sum(dim=1).clamp(min=1).to(dtype=values.dtype)
        per_patient = safe.sum(dim=1) / count
        eligible = mask.any(dim=1)
        per_patient = torch.where(
            eligible, per_patient, torch.zeros_like(per_patient)
        )
        return per_patient, eligible

    @staticmethod
    def _masked_patient_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        per_patient, eligible = SoftPatientAtlas._masked_patient_values(values, mask)
        if not bool(eligible.any()):
            return values.sum() * 0.0
        return per_patient[eligible].mean()

    def _residualized_blood_anchor(
        self, safe_values: torch.Tensor, visible_mask: torch.Tensor
    ) -> torch.Tensor:
        anchor_mask = visible_mask & self.blood_anchor_eligible_mask[None, :]
        anchor_values = torch.where(
            anchor_mask, safe_values, torch.zeros_like(safe_values)
        )
        float_mask = anchor_mask.to(dtype=safe_values.dtype)
        with torch.no_grad():
            observed = self.blood_tower.encode(anchor_values, float_mask)
            mask_only = self.blood_tower.encode(torch.zeros_like(safe_values), float_mask)
        if observed.shape != (
            len(safe_values),
            self.config.blood_anchor_dim,
        ):
            raise ValueError(
                "blood_tower returned the wrong width: expected "
                f"{self.config.blood_anchor_dim}, got {tuple(observed.shape)}"
            )
        return observed - mask_only

    def _validate_inputs(
        self,
        *,
        eye_embeddings: torch.Tensor,
        eye_visible_mask: torch.Tensor,
        blood_values: torch.Tensor,
        blood_visible_mask: torch.Tensor,
        blood_eligible_mask: torch.Tensor,
        demographics: torch.Tensor,
        demographic_mask: torch.Tensor,
        eye_device_ids: torch.Tensor | None,
        eye_laterality_ids: torch.Tensor | None,
        eye_quality: torch.Tensor | None,
    ) -> None:
        c = self.config
        if eye_embeddings.ndim != 3 or eye_embeddings.shape[2] != c.eye_dim:
            raise ValueError("eye_embeddings must have shape [batch, images, eye_dim]")
        batch, n_images, _ = eye_embeddings.shape
        if eye_visible_mask.shape != (batch, n_images):
            raise ValueError("eye_visible_mask has the wrong shape")
        if blood_values.shape != (batch, c.num_blood_features):
            raise ValueError("blood_values has the wrong shape")
        if blood_visible_mask.shape != blood_values.shape:
            raise ValueError("blood_visible_mask has the wrong shape")
        if blood_eligible_mask.shape != blood_values.shape:
            raise ValueError("blood_eligible_mask has the wrong shape")
        if demographics.shape != (batch, c.demographic_dim):
            raise ValueError("demographics has the wrong shape")
        if demographic_mask.shape != demographics.shape:
            raise ValueError("demographic_mask has the wrong shape")
        for name, mask in (
            ("eye_visible_mask", eye_visible_mask),
            ("blood_visible_mask", blood_visible_mask),
            ("blood_eligible_mask", blood_eligible_mask),
            ("demographic_mask", demographic_mask),
        ):
            if mask.dtype != torch.bool:
                raise TypeError(f"{name} must be boolean")
        value_tensors = (
            ("eye_embeddings", eye_embeddings),
            ("blood_values", blood_values),
            ("demographics", demographics),
        )
        for name, tensor in value_tensors:
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must be floating point")
            if tensor.dtype != eye_embeddings.dtype:
                raise TypeError("all floating patient values must share one dtype")
        model_dtype = next(self.parameters()).dtype
        if eye_embeddings.dtype != model_dtype:
            raise TypeError(
                f"input dtype {eye_embeddings.dtype} does not match model dtype {model_dtype}"
            )
        tensors = [
            tensor
            for _, tensor in value_tensors
        ] + [
            eye_visible_mask,
            blood_visible_mask,
            blood_eligible_mask,
            demographic_mask,
        ]
        if any(tensor.device != eye_embeddings.device for tensor in tensors):
            raise ValueError("all patient values must share one device")
        if bool((blood_visible_mask & ~blood_eligible_mask).any()):
            raise ValueError("blood_visible_mask must be a subset of eligible fields")
        if batch > 1 and not torch.equal(
            blood_eligible_mask, blood_eligible_mask[:1].expand_as(blood_eligible_mask)
        ):
            raise ValueError(
                "blood_eligible_mask is an artifact policy and must be constant "
                "across patients in a batch"
            )
        if not torch.isfinite(eye_embeddings[eye_visible_mask]).all():
            raise ValueError("visible eye embeddings must be finite")
        if not torch.isfinite(blood_values[blood_visible_mask]).all():
            raise ValueError("visible blood values must be finite")
        binary_values = blood_values[:, c.num_continuous :]
        binary_visible = blood_visible_mask[:, c.num_continuous :]
        visible_binary_values = binary_values[binary_visible]
        if visible_binary_values.numel() and not bool(
            ((visible_binary_values == 0) | (visible_binary_values == 1)).all()
        ):
            raise ValueError("visible binary condition values must be exactly 0 or 1")
        if not torch.isfinite(demographics[demographic_mask]).all():
            raise ValueError("visible demographics must be finite")

        integer_metadata = (
            ("eye_device_ids", eye_device_ids, c.num_devices),
            ("eye_laterality_ids", eye_laterality_ids, c.num_lateralities),
        )
        for name, values, upper in integer_metadata:
            if values is None:
                continue
            if values.shape != (batch, n_images) or values.dtype != torch.long:
                raise TypeError(f"{name} must be long with shape [batch, images]")
            if values.device != eye_embeddings.device:
                raise ValueError(f"{name} must share the input device")
            visible_values = values[eye_visible_mask]
            if visible_values.numel() and (
                int(visible_values.min()) < 0 or int(visible_values.max()) >= upper
            ):
                raise ValueError(f"{name} contains an out-of-range visible id")
        if eye_quality is not None:
            if eye_quality.shape != (batch, n_images):
                raise ValueError("eye_quality has the wrong shape")
            if eye_quality.device != eye_embeddings.device:
                raise ValueError("eye_quality must share the input device")
            if not eye_quality.is_floating_point() or eye_quality.dtype != eye_embeddings.dtype:
                raise TypeError("eye_quality must share the floating input dtype")
            if not torch.isfinite(eye_quality[eye_visible_mask]).all():
                raise ValueError("visible eye quality must be finite")


__all__ = [
    "ClinicalPrediction",
    "EvidenceFactor",
    "EyePrediction",
    "GaussianState",
    "PatientAtlasConfig",
    "PatientEvidenceOutput",
    "PatientStateOutput",
    "SoftPatientAtlas",
]
