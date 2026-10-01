"""Conservative, aggregate-first interpretation tools for Patient Atlas v1.

The module deliberately separates four different questions that are often
collapsed into one misleading "factor importance" number:

* held-out likelihood/deviance and expected-Fisher relevance;
* exact modality-level natural evidence and non-additive deletion effects;
* signed-permutation-identifiable axes versus unresolved subspaces; and
* the fail-closed gate that permits a cautious factor name.

No function writes patient rows or selects factors from outcomes.  A caller is
responsible for running held-out calculations inside the fold that produced
the model.  Procrustes/rotation is supported nowhere in the axis-naming path;
principal angles are the only rotation-invariant subspace diagnostic here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from soft_patient_atlas import (
    GaussianState,
    PatientStateOutput,
    SoftGroupFactorDecoder,
    SoftPatientAtlas,
)


_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
_HELD_OUT_SPLITS = {
    "outer_validation",
    "outer_test",
    "external_validation",
    "synthetic_validation",
    "weight_heldout_calibration",
}
_PROFILE_LABELS = {
    "shared",
    "eye_dominant",
    "blood_clinical_dominant",
    "weakly_mixed",
    "inactive",
}


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or _HEX_256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256 digest")


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FactorRelevanceReport:
    """Held-out, likelihood-specific factor relevance.

    Deviance reduction is ``2 * (NLL_factor_ablated - NLL_full)`` and can be
    negative on held-out data.  Expected Fisher information is nonnegative and
    uses the likelihood's native metric.  Every statistic is patient-balanced
    and then observation/dimension-balanced within its declared block.
    """

    split_role: str
    eye_deviance_reduction: torch.Tensor
    continuous_deviance_reduction: torch.Tensor
    binary_deviance_reduction: torch.Tensor
    blood_clinical_deviance_reduction: torch.Tensor
    eye_expected_fisher: torch.Tensor
    continuous_expected_fisher: torch.Tensor
    binary_expected_fisher: torch.Tensor
    blood_clinical_expected_fisher: torch.Tensor
    eye_image_count: int
    continuous_value_count: int
    binary_value_count: int
    patient_count: int
    predictive_uncertainty_integrated: bool

    @property
    def latent_dim(self) -> int:
        return int(self.eye_deviance_reduction.numel())


@dataclass(frozen=True)
class NaturalEvidenceContributions:
    """Exact fusion terms plus exact one-observation deletion effects.

    The three modality/interaction terms add exactly to ``full_*``.  The
    per-observation tensors are leave-one-out differences from the full
    coalition.  They include nonlinear re-encoding and interaction changes;
    consequently they are *not* Shapley values and must not be summed as an
    additive decomposition.
    """

    full_natural_parameter: torch.Tensor
    full_precision_increment: torch.Tensor
    eye_modality_natural_parameter: torch.Tensor
    blood_modality_natural_parameter: torch.Tensor
    interaction_natural_parameter: torch.Tensor
    eye_modality_precision_increment: torch.Tensor
    blood_modality_precision_increment: torch.Tensor
    interaction_precision_increment: torch.Tensor
    eye_observation_deletion_delta_natural: torch.Tensor
    blood_observation_deletion_delta_natural: torch.Tensor
    eye_observation_deletion_delta_precision: torch.Tensor
    blood_observation_deletion_delta_precision: torch.Tensor
    eye_visible_mask: torch.Tensor
    blood_visible_mask: torch.Tensor
    observation_effects_are_additive: bool = False
    observation_effect_semantics: str = "full_coalition_leave_one_out"


@dataclass(frozen=True)
class SignedPermutationMatch:
    """Axis assignments, with unresolved axes fail-closed to ``-1``/sign ``0``."""

    matched_candidate_index: torch.Tensor
    sign: torch.Tensor
    congruence: torch.Tensor
    assignment_margin: torch.Tensor
    identifiable: torch.Tensor
    min_congruence: float
    min_assignment_margin: float


@dataclass(frozen=True)
class PrincipalAngleReport:
    """Rotation-invariant comparison of one explicitly declared factor block."""

    reference_factor_indices: tuple[int, ...]
    candidate_factor_indices: tuple[int, ...]
    angles_degrees: tuple[float, ...]
    max_angle_degrees: float
    mean_angle_degrees: float
    chordal_distance: float
    reference_rank: int
    candidate_rank: int


@dataclass(frozen=True)
class FactorProfileThresholds:
    """Frozen thresholds for descriptive view profiles, not factor naming."""

    min_amplitude: float
    min_expected_fisher: float
    shared_minimum_share: float

    def __post_init__(self) -> None:
        if self.min_amplitude < 0 or self.min_expected_fisher < 0:
            raise ValueError("profile activity thresholds must be nonnegative")
        if not 0 < self.shared_minimum_share <= 0.5:
            raise ValueError("shared_minimum_share must lie in (0, 0.5]")


@dataclass(frozen=True)
class FactorModalityProfile:
    factor_index: int
    profile: str
    eye_amplitude: float
    blood_amplitude: float
    eye_expected_fisher: float
    blood_expected_fisher: float
    eye_combined_share: float
    blood_combined_share: float
    identifiable_axis: bool
    axis_interpretation_allowed: bool


@dataclass(frozen=True)
class FactorNamingThresholds:
    """Thresholds calibrated before interpretation on synthetic/null matches."""

    min_loading_congruence: float
    min_sign_agreement: float
    min_assignment_margin: float
    min_matched_replicates: int
    calibration_artifact_hash: str

    def __post_init__(self) -> None:
        for name, value in (
            ("min_loading_congruence", self.min_loading_congruence),
            ("min_sign_agreement", self.min_sign_agreement),
            ("min_assignment_margin", self.min_assignment_margin),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not isinstance(self.min_matched_replicates, int) or self.min_matched_replicates < 2:
            raise ValueError("min_matched_replicates must be an integer >= 2")
        _require_sha256("calibration_artifact_hash", self.calibration_artifact_hash)


@dataclass(frozen=True)
class InterpretationProvenance:
    """Aggregate-only provenance required by the naming artifact."""

    model_checkpoint_hash: str
    preprocessing_hash: str
    feature_schema_hash: str
    mask_policy_hash: str
    fold_manifest_hash: str
    source_code_hash: str
    discovery_partition: str
    validation_partition: str

    def __post_init__(self) -> None:
        for name in (
            "model_checkpoint_hash",
            "preprocessing_hash",
            "feature_schema_hash",
            "mask_policy_hash",
            "fold_manifest_hash",
            "source_code_hash",
        ):
            _require_sha256(name, getattr(self, name))
        if not self.discovery_partition or not self.validation_partition:
            raise ValueError("discovery and validation partitions must be named")
        if self.discovery_partition == self.validation_partition:
            raise ValueError("discovery and validation partitions must be separated")


@dataclass(frozen=True)
class FactorStabilityRecord:
    """Aggregate evidence for deciding whether an individual axis may be named."""

    factor_index: int
    identifiable_axis: bool
    matched_replicates: int
    min_loading_congruence: float
    sign_agreement: float
    min_assignment_margin: float
    bootstrap_loading_groups_significant: bool
    confound_audits_passed: bool
    eye_recoverable: bool
    blood_recoverable: bool
    pairing_null_passed: bool
    unresolved_block_id: str | None = None
    alignment_method: str = "signed_permutation"

    def __post_init__(self) -> None:
        if not isinstance(self.factor_index, int) or self.factor_index < 0:
            raise ValueError("factor_index must be a nonnegative integer")
        if not isinstance(self.matched_replicates, int) or self.matched_replicates < 0:
            raise ValueError("matched_replicates must be a nonnegative integer")
        for name in (
            "min_loading_congruence",
            "sign_agreement",
            "min_assignment_margin",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.alignment_method != "signed_permutation":
            raise ValueError("individual axes may use signed-permutation alignment only")


def _validate_factor_tensors(tensors: Sequence[torch.Tensor]) -> int:
    if not tensors:
        raise ValueError("at least one factor tensor is required")
    width = tensors[0].numel()
    reference = tensors[0]
    if reference.ndim != 1 or not reference.is_floating_point():
        raise TypeError("factor statistics must be one-dimensional floating tensors")
    for tensor in tensors:
        if tensor.ndim != 1 or tensor.numel() != width:
            raise ValueError("all factor statistics must have the same one-dimensional shape")
        if tensor.device != reference.device or tensor.dtype != reference.dtype:
            raise TypeError("all factor statistics must share dtype and device")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("factor statistics must be finite")
    return width


def _masked_patient_average(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Patient-balanced mean; returns NaN when the block has no observations."""

    if values.shape != mask.shape or mask.dtype != torch.bool:
        raise TypeError("values and boolean mask must have the same shape")
    safe = torch.where(mask, values, torch.zeros_like(values))
    count = mask.sum(dim=1)
    present = count > 0
    if not bool(present.any()):
        return torch.full((), float("nan"), dtype=values.dtype, device=values.device)
    per_patient = safe.sum(dim=1) / count.clamp(min=1).to(values.dtype)
    return per_patient[present].mean()


def _validate_relevance_inputs(
    decoder: SoftGroupFactorDecoder,
    latent_mean: torch.Tensor,
    demographics: torch.Tensor,
    target_eye_embeddings: torch.Tensor,
    target_eye_mask: torch.Tensor,
    target_blood_values: torch.Tensor,
    target_blood_mask: torch.Tensor,
    target_blood_eligible_mask: torch.Tensor,
    eye_device_ids: torch.Tensor,
    eye_laterality_ids: torch.Tensor,
    split_role: str,
    latent_log_variance: torch.Tensor | None,
) -> None:
    c = decoder.config
    if split_role not in _HELD_OUT_SPLITS:
        raise ValueError("factor relevance may be computed only on a declared held-out split")
    if latent_mean.ndim != 2 or latent_mean.shape[1] != c.latent_dim:
        raise ValueError("latent_mean must have shape [patients, latent_dim]")
    batch = latent_mean.shape[0]
    if demographics.shape != (batch, c.demographic_dim):
        raise ValueError("demographics has the wrong shape")
    if target_eye_embeddings.ndim != 3 or target_eye_embeddings.shape[:1] != (batch,):
        raise ValueError("target_eye_embeddings has the wrong batch shape")
    if target_eye_embeddings.shape[2] != c.eye_dim:
        raise ValueError("target_eye_embeddings has the wrong feature width")
    eye_shape = target_eye_embeddings.shape[:2]
    if target_eye_mask.shape != eye_shape or target_eye_mask.dtype != torch.bool:
        raise TypeError("target_eye_mask must be boolean with shape [patients, images]")
    if eye_device_ids.shape != eye_shape or eye_device_ids.dtype != torch.long:
        raise TypeError("eye_device_ids must be long with shape [patients, images]")
    if eye_laterality_ids.shape != eye_shape or eye_laterality_ids.dtype != torch.long:
        raise TypeError("eye_laterality_ids must be long with shape [patients, images]")
    if target_blood_values.shape != (batch, c.num_blood_features):
        raise ValueError("target_blood_values has the wrong shape")
    if target_blood_mask.shape != target_blood_values.shape or target_blood_mask.dtype != torch.bool:
        raise TypeError("target_blood_mask must be boolean and match target_blood_values")
    if (
        target_blood_eligible_mask.shape != target_blood_values.shape
        or target_blood_eligible_mask.dtype != torch.bool
    ):
        raise TypeError(
            "target_blood_eligible_mask must be boolean and match target_blood_values"
        )
    if bool((target_blood_mask & ~target_blood_eligible_mask).any()):
        raise ValueError("target_blood_mask must be a subset of policy-eligible fields")
    if batch > 1 and not torch.equal(
        target_blood_eligible_mask,
        target_blood_eligible_mask[:1].expand_as(target_blood_eligible_mask),
    ):
        raise ValueError(
            "target_blood_eligible_mask is an artifact policy and must be constant"
        )
    floating = (latent_mean, demographics, target_eye_embeddings, target_blood_values)
    if any(tensor.dtype != latent_mean.dtype or tensor.device != latent_mean.device for tensor in floating):
        raise TypeError("all floating relevance inputs must share dtype and device")
    metadata = (
        target_eye_mask,
        target_blood_mask,
        target_blood_eligible_mask,
        eye_device_ids,
        eye_laterality_ids,
    )
    if any(tensor.device != latent_mean.device for tensor in metadata):
        raise ValueError("all relevance inputs must share one device")
    if not bool(torch.isfinite(latent_mean).all()) or not bool(
        torch.isfinite(demographics).all()
    ):
        raise ValueError("latent means and demographic context must be finite")
    if latent_log_variance is not None:
        if latent_log_variance.shape != latent_mean.shape:
            raise ValueError("latent_log_variance must match latent_mean")
        if (
            latent_log_variance.dtype != latent_mean.dtype
            or latent_log_variance.device != latent_mean.device
        ):
            raise TypeError("latent_log_variance must share latent_mean dtype and device")
        if not bool(torch.isfinite(latent_log_variance).all()):
            raise ValueError("latent_log_variance must be finite")
    if not bool(torch.isfinite(target_eye_embeddings[target_eye_mask]).all()):
        raise ValueError("visible eye targets must be finite")
    if not bool(torch.isfinite(target_blood_values[target_blood_mask]).all()):
        raise ValueError("visible blood targets must be finite")
    binary = target_blood_values[:, c.num_continuous :]
    binary_mask = target_blood_mask[:, c.num_continuous :]
    visible_binary = binary[binary_mask]
    if visible_binary.numel() and not bool(((visible_binary == 0) | (visible_binary == 1)).all()):
        raise ValueError("visible binary targets must be exactly 0 or 1")
    visible_devices = eye_device_ids[target_eye_mask]
    if visible_devices.numel() and (
        int(visible_devices.min()) < 0 or int(visible_devices.max()) >= c.num_devices
    ):
        raise ValueError("eye_device_ids contains an out-of-range visible id")
    visible_lateralities = eye_laterality_ids[target_eye_mask]
    if visible_lateralities.numel() and (
        int(visible_lateralities.min()) < 0
        or int(visible_lateralities.max()) >= c.num_lateralities
    ):
        raise ValueError("eye_laterality_ids contains an out-of-range visible id")


@torch.no_grad()
def heldout_factor_relevance(
    decoder: SoftGroupFactorDecoder,
    *,
    latent_mean: torch.Tensor,
    demographics: torch.Tensor,
    target_eye_embeddings: torch.Tensor,
    target_eye_mask: torch.Tensor,
    target_blood_values: torch.Tensor,
    target_blood_mask: torch.Tensor,
    target_blood_eligible_mask: torch.Tensor,
    eye_device_ids: torch.Tensor,
    eye_laterality_ids: torch.Tensor,
    split_role: str,
    latent_log_variance: torch.Tensor | None = None,
) -> FactorRelevanceReport:
    """Compute conditional held-out deviance reduction and expected Fisher.

    Factor ablation sets only ``z_k`` to its residual-prior mean of zero and
    leaves all context and other factor coordinates unchanged.  This is a
    decoder relevance diagnostic, not a causal intervention.
    """

    _validate_relevance_inputs(
        decoder,
        latent_mean,
        demographics,
        target_eye_embeddings,
        target_eye_mask,
        target_blood_values,
        target_blood_mask,
        target_blood_eligible_mask,
        eye_device_ids,
        eye_laterality_ids,
        split_role,
        latent_log_variance,
    )
    c = decoder.config
    nc = c.num_continuous
    safe_device_ids = torch.where(
        target_eye_mask, eye_device_ids, torch.zeros_like(eye_device_ids)
    )
    safe_laterality_ids = torch.where(
        target_eye_mask,
        eye_laterality_ids,
        torch.full_like(eye_laterality_ids, c.num_lateralities - 1),
    )
    safe_eye = torch.where(
        target_eye_mask[..., None], target_eye_embeddings, torch.zeros_like(target_eye_embeddings)
    )
    continuous_target = target_blood_values[:, :nc]
    continuous_mask = target_blood_mask[:, :nc]
    binary_target = target_blood_values[:, nc:]
    binary_mask = target_blood_mask[:, nc:]
    safe_continuous = torch.where(continuous_mask, continuous_target, torch.zeros_like(continuous_target))
    safe_binary = torch.where(binary_mask, binary_target, torch.zeros_like(binary_target))

    conditional_eye, conditional_clinical = decoder.conditional(
        latent_mean,
        demographics,
        eye_device_ids=safe_device_ids,
        eye_laterality_ids=safe_laterality_ids,
    )
    if latent_log_variance is None:
        full_eye, full_clinical = conditional_eye, conditional_clinical
    else:
        full_eye, full_clinical = decoder(
            GaussianState(latent_mean, latent_log_variance),
            demographics,
            eye_device_ids=safe_device_ids,
            eye_laterality_ids=safe_laterality_ids,
        )
    full_eye_nll = 0.5 * (
        full_eye.log_variance
        + (safe_eye - full_eye.mean).square() / full_eye.log_variance.exp()
    )
    student_df = torch.tensor(c.student_df, dtype=latent_mean.dtype, device=latent_mean.device)
    full_continuous_nll = -torch.distributions.StudentT(
        df=student_df,
        loc=full_clinical.continuous_location,
        scale=full_clinical.continuous_log_scale.exp(),
    ).log_prob(safe_continuous)
    full_binary_nll = F.binary_cross_entropy_with_logits(
        full_clinical.binary_logits, safe_binary, reduction="none"
    )

    eye_deviance: list[torch.Tensor] = []
    continuous_deviance: list[torch.Tensor] = []
    binary_deviance: list[torch.Tensor] = []
    blood_deviance: list[torch.Tensor] = []
    eye_fisher: list[torch.Tensor] = []
    continuous_fisher: list[torch.Tensor] = []
    binary_fisher: list[torch.Tensor] = []
    blood_fisher: list[torch.Tensor] = []
    continuous_loading, binary_loading, _ = decoder._split_blood_loading()
    binary_probability = torch.sigmoid(conditional_clinical.binary_logits)
    student_location_constant = (c.student_df + 1.0) / (c.student_df + 3.0)

    for factor in range(c.latent_dim):
        ablated_z = latent_mean.clone()
        ablated_z[:, factor] = 0.0
        if latent_log_variance is None:
            ablated_eye, ablated_clinical = decoder.conditional(
                ablated_z,
                demographics,
                eye_device_ids=safe_device_ids,
                eye_laterality_ids=safe_laterality_ids,
            )
        else:
            ablated_log_variance = latent_log_variance.clone()
            ablated_log_variance[:, factor] = -torch.inf
            ablated_eye, ablated_clinical = decoder(
                GaussianState(ablated_z, ablated_log_variance),
                demographics,
                eye_device_ids=safe_device_ids,
                eye_laterality_ids=safe_laterality_ids,
            )
        ablated_eye_nll = 0.5 * (
            ablated_eye.log_variance
            + (safe_eye - ablated_eye.mean).square() / ablated_eye.log_variance.exp()
        )
        ablated_continuous_nll = -torch.distributions.StudentT(
            df=student_df,
            loc=ablated_clinical.continuous_location,
            scale=ablated_clinical.continuous_log_scale.exp(),
        ).log_prob(safe_continuous)
        ablated_binary_nll = F.binary_cross_entropy_with_logits(
            ablated_clinical.binary_logits, safe_binary, reduction="none"
        )

        eye_delta = 2.0 * (ablated_eye_nll - full_eye_nll).mean(dim=-1)
        continuous_delta = 2.0 * (ablated_continuous_nll - full_continuous_nll)
        binary_delta = 2.0 * (ablated_binary_nll - full_binary_nll)
        eye_deviance.append(_masked_patient_average(eye_delta, target_eye_mask))
        continuous_deviance.append(_masked_patient_average(continuous_delta, continuous_mask))
        binary_deviance.append(_masked_patient_average(binary_delta, binary_mask))
        blood_deviance.append(
            _masked_patient_average(
                torch.cat([continuous_delta, binary_delta], dim=1),
                target_blood_mask,
            )
        )

        eye_information = (
            decoder.eye_loading[:, factor].square()[None, None, :]
            / conditional_eye.log_variance.exp()
        ).mean(dim=-1)
        continuous_information = (
            student_location_constant
            * continuous_loading[:, factor].square()[None, :]
            / conditional_clinical.continuous_log_scale.mul(2.0).exp()
        )
        binary_information = (
            binary_probability
            * (1.0 - binary_probability)
            * binary_loading[:, factor].square()[None, :]
        )
        eye_fisher.append(_masked_patient_average(eye_information, target_eye_mask))
        continuous_fisher.append(
            _masked_patient_average(continuous_information, continuous_mask)
        )
        binary_fisher.append(_masked_patient_average(binary_information, binary_mask))
        blood_fisher.append(
            _masked_patient_average(
                torch.cat([continuous_information, binary_information], dim=1),
                target_blood_mask,
            )
        )

    return FactorRelevanceReport(
        split_role=split_role,
        eye_deviance_reduction=torch.stack(eye_deviance),
        continuous_deviance_reduction=torch.stack(continuous_deviance),
        binary_deviance_reduction=torch.stack(binary_deviance),
        blood_clinical_deviance_reduction=torch.stack(blood_deviance),
        eye_expected_fisher=torch.stack(eye_fisher),
        continuous_expected_fisher=torch.stack(continuous_fisher),
        binary_expected_fisher=torch.stack(binary_fisher),
        blood_clinical_expected_fisher=torch.stack(blood_fisher),
        eye_image_count=int(target_eye_mask.sum().item()),
        continuous_value_count=int(continuous_mask.sum().item()),
        binary_value_count=int(binary_mask.sum().item()),
        patient_count=int(latent_mean.shape[0]),
        predictive_uncertainty_integrated=latent_log_variance is not None,
    )


def likelihood_normalized_loading_signatures(
    decoder: SoftGroupFactorDecoder,
) -> torch.Tensor:
    """Return reference-Fisher loading signatures with shape ``[factor, field]``.

    Eye uses the base-device Gaussian noise, continuous clinical fields use the
    Student-t location Fisher metric, and binary fields use Bernoulli Fisher at
    the reference probability 0.5.  The frozen anchor is intentionally omitted
    because it is a distillation target rather than an independent view.
    """

    c = decoder.config
    continuous_loading, binary_loading, _ = decoder._split_blood_loading()
    eye = decoder.eye_loading / decoder.eye_log_noise.exp()[:, None]
    student_factor = math.sqrt((c.student_df + 1.0) / (c.student_df + 3.0))
    continuous = (
        continuous_loading
        / decoder.continuous_log_scale.exp()[:, None]
        * student_factor
    )
    binary = binary_loading * 0.5
    signature = torch.cat([eye, continuous, binary], dim=0).T
    return F.normalize(signature, p=2, dim=1, eps=1e-12)


def _natural_terms(output: PatientStateOutput) -> tuple[torch.Tensor, torch.Tensor]:
    natural = (
        output.eye_evidence.natural_parameter
        + output.blood_evidence.natural_parameter
        + output.interaction_natural_parameter
    )
    precision = (
        output.eye_evidence.precision_increment
        + output.blood_evidence.precision_increment
        + output.interaction_precision
    )
    return natural, precision


@torch.no_grad()
def natural_evidence_contributions(
    model: SoftPatientAtlas,
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
    max_observations: int = 512,
) -> NaturalEvidenceContributions:
    """Re-run exact single-observation deletions for one in-memory batch.

    This routine never serializes its row-level tensors.  It is intended for a
    controlled per-patient explanation path or synthetic tests, not aggregate
    factor naming.  Deletion effects include all induced recognition and
    interaction changes and therefore need not sum to the total evidence.
    """

    if not isinstance(max_observations, int) or max_observations <= 0:
        raise ValueError("max_observations must be a positive integer")
    n_eye = eye_visible_mask.shape[1]
    n_blood = blood_visible_mask.shape[1]
    if n_eye + n_blood > max_observations:
        raise ValueError("requested deletion explanation exceeds max_observations")
    base_inputs: dict[str, object] = {
        "eye_embeddings": eye_embeddings,
        "eye_visible_mask": eye_visible_mask,
        "blood_values": blood_values,
        "blood_visible_mask": blood_visible_mask,
        "blood_eligible_mask": blood_eligible_mask,
        "demographics": demographics,
        "demographic_mask": demographic_mask,
        "eye_device_ids": eye_device_ids,
        "eye_laterality_ids": eye_laterality_ids,
        "eye_quality": eye_quality,
        "enable_interaction": enable_interaction,
    }
    was_training = model.training
    model.eval()
    try:
        full = model(**base_inputs)
        full_natural, full_precision = _natural_terms(full)
        eye_natural_effects: list[torch.Tensor] = []
        eye_precision_effects: list[torch.Tensor] = []
        for index in range(n_eye):
            present = eye_visible_mask[:, index]
            if not bool(present.any()):
                eye_natural_effects.append(torch.zeros_like(full_natural))
                eye_precision_effects.append(torch.zeros_like(full_precision))
                continue
            reduced_mask = eye_visible_mask.clone()
            reduced_mask[:, index] = False
            reduced = model(**{**base_inputs, "eye_visible_mask": reduced_mask})
            reduced_natural, reduced_precision = _natural_terms(reduced)
            eye_natural_effects.append(
                torch.where(present[:, None], full_natural - reduced_natural, torch.zeros_like(full_natural))
            )
            eye_precision_effects.append(
                torch.where(present[:, None], full_precision - reduced_precision, torch.zeros_like(full_precision))
            )

        blood_natural_effects: list[torch.Tensor] = []
        blood_precision_effects: list[torch.Tensor] = []
        for index in range(n_blood):
            present = blood_visible_mask[:, index]
            if not bool(present.any()):
                blood_natural_effects.append(torch.zeros_like(full_natural))
                blood_precision_effects.append(torch.zeros_like(full_precision))
                continue
            reduced_mask = blood_visible_mask.clone()
            reduced_mask[:, index] = False
            reduced = model(**{**base_inputs, "blood_visible_mask": reduced_mask})
            reduced_natural, reduced_precision = _natural_terms(reduced)
            blood_natural_effects.append(
                torch.where(present[:, None], full_natural - reduced_natural, torch.zeros_like(full_natural))
            )
            blood_precision_effects.append(
                torch.where(present[:, None], full_precision - reduced_precision, torch.zeros_like(full_precision))
            )
    finally:
        model.train(was_training)

    return NaturalEvidenceContributions(
        full_natural_parameter=full_natural,
        full_precision_increment=full_precision,
        eye_modality_natural_parameter=full.eye_evidence.natural_parameter,
        blood_modality_natural_parameter=full.blood_evidence.natural_parameter,
        interaction_natural_parameter=full.interaction_natural_parameter,
        eye_modality_precision_increment=full.eye_evidence.precision_increment,
        blood_modality_precision_increment=full.blood_evidence.precision_increment,
        interaction_precision_increment=full.interaction_precision,
        eye_observation_deletion_delta_natural=torch.stack(eye_natural_effects, dim=1),
        blood_observation_deletion_delta_natural=torch.stack(blood_natural_effects, dim=1),
        eye_observation_deletion_delta_precision=torch.stack(eye_precision_effects, dim=1),
        blood_observation_deletion_delta_precision=torch.stack(blood_precision_effects, dim=1),
        eye_visible_mask=eye_visible_mask.detach().clone(),
        blood_visible_mask=blood_visible_mask.detach().clone(),
    )


def match_signed_permutation_axes(
    reference_signatures: torch.Tensor,
    candidate_signatures: torch.Tensor,
    *,
    min_congruence: float,
    min_assignment_margin: float,
) -> SignedPermutationMatch:
    """Hungarian-match axes by absolute cosine, exposing only identifiable ones.

    A match must clear both an absolute congruence threshold and the smaller of
    its reference-row and candidate-column gaps to the next-best match.  Axes
    that fail either gate receive index ``-1`` and sign ``0``; the Hungarian
    assignment is not treated as an interpretation for them.
    """

    if reference_signatures.ndim != 2 or candidate_signatures.ndim != 2:
        raise ValueError("loading signatures must have shape [factor, feature]")
    if reference_signatures.shape != candidate_signatures.shape:
        raise ValueError("reference and candidate signatures must have identical shape")
    if reference_signatures.shape[0] < 1 or reference_signatures.shape[1] < 1:
        raise ValueError("loading signatures cannot be empty")
    if not 0 <= min_congruence <= 1 or not 0 <= min_assignment_margin <= 1:
        raise ValueError("matching thresholds must lie in [0, 1]")
    if reference_signatures.dtype != candidate_signatures.dtype:
        raise TypeError("loading signatures must share a floating dtype")
    if reference_signatures.device != candidate_signatures.device:
        raise ValueError("loading signatures must share one device")
    if not reference_signatures.is_floating_point():
        raise TypeError("loading signatures must be floating point")
    if not bool(torch.isfinite(reference_signatures).all()) or not bool(torch.isfinite(candidate_signatures).all()):
        raise ValueError("loading signatures must be finite")
    reference_norm = reference_signatures.norm(dim=1)
    candidate_norm = candidate_signatures.norm(dim=1)
    if bool((reference_norm <= 1e-12).any()) or bool((candidate_norm <= 1e-12).any()):
        raise ValueError("zero loading signatures cannot identify an axis")
    reference = F.normalize(reference_signatures, p=2, dim=1)
    candidate = F.normalize(candidate_signatures, p=2, dim=1)
    signed_congruence = reference @ candidate.T
    absolute = signed_congruence.abs()
    row_indices, column_indices = linear_sum_assignment(
        -absolute.detach().cpu().double().numpy()
    )
    factor_count = reference.shape[0]
    assignment = torch.full(
        (factor_count,), -1, dtype=torch.long, device=reference.device
    )
    assignment[
        torch.as_tensor(row_indices, dtype=torch.long, device=reference.device)
    ] = torch.as_tensor(column_indices, dtype=torch.long, device=reference.device)
    rows = torch.arange(factor_count, device=reference.device)
    assigned_signed = signed_congruence[rows, assignment]
    congruence = assigned_signed.abs()
    raw_sign = torch.where(
        assigned_signed >= 0,
        torch.ones_like(assigned_signed),
        -torch.ones_like(assigned_signed),
    )

    if factor_count == 1:
        margin = congruence.clone()
    else:
        row_competitor = absolute.clone()
        row_competitor[rows, assignment] = -1.0
        row_gap = congruence - row_competitor.max(dim=1).values
        column_competitor = absolute.clone()
        column_competitor[rows, assignment] = -1.0
        column_gap = torch.empty_like(congruence)
        for row, column in enumerate(assignment.tolist()):
            column_gap[row] = congruence[row] - column_competitor[:, column].max()
        margin = torch.minimum(row_gap, column_gap).clamp(min=0.0)
    identifiable = (congruence >= min_congruence) & (margin >= min_assignment_margin)
    matched = torch.where(identifiable, assignment, torch.full_like(assignment, -1))
    sign = torch.where(identifiable, raw_sign, torch.zeros_like(raw_sign))
    return SignedPermutationMatch(
        matched_candidate_index=matched,
        sign=sign,
        congruence=congruence,
        assignment_margin=margin,
        identifiable=identifiable,
        min_congruence=float(min_congruence),
        min_assignment_margin=float(min_assignment_margin),
    )


def align_identifiable_axes(
    candidate_values: torch.Tensor,
    match: SignedPermutationMatch,
    *,
    factor_axis: int = -1,
) -> torch.Tensor:
    """Apply only the admitted sign/permutation; unresolved coordinates are NaN."""

    if candidate_values.ndim == 0:
        raise ValueError("candidate_values must contain a factor axis")
    factor_axis = factor_axis % candidate_values.ndim
    factor_count = match.matched_candidate_index.numel()
    if candidate_values.shape[factor_axis] != factor_count:
        raise ValueError("candidate_values factor axis does not match the alignment")
    if not candidate_values.is_floating_point():
        raise TypeError("candidate_values must be floating point so unresolved axes can be NaN")
    moved = candidate_values.movedim(factor_axis, -1)
    aligned = torch.full_like(moved, float("nan"))
    for reference_index in range(factor_count):
        candidate_index = int(match.matched_candidate_index[reference_index])
        if candidate_index >= 0:
            aligned[..., reference_index] = (
                moved[..., candidate_index] * match.sign[reference_index]
            )
    return aligned.movedim(-1, factor_axis)


def principal_angle_subspace_report(
    reference_signatures: torch.Tensor,
    candidate_signatures: torch.Tensor,
    *,
    reference_factor_indices: Sequence[int],
    candidate_factor_indices: Sequence[int],
) -> PrincipalAngleReport:
    """Compare declared unresolved blocks without producing rotated axes."""

    if reference_signatures.ndim != 2 or candidate_signatures.ndim != 2:
        raise ValueError("loading signatures must have shape [factor, feature]")
    if reference_signatures.shape[1] != candidate_signatures.shape[1]:
        raise ValueError("subspaces must live in the same loading-signature feature space")
    reference_indices = tuple(int(index) for index in reference_factor_indices)
    candidate_indices = tuple(int(index) for index in candidate_factor_indices)
    if not reference_indices or not candidate_indices:
        raise ValueError("each unresolved subspace block must contain at least one factor")
    if len(set(reference_indices)) != len(reference_indices) or len(set(candidate_indices)) != len(candidate_indices):
        raise ValueError("subspace factor indices must be unique")
    if min(reference_indices) < 0 or max(reference_indices) >= reference_signatures.shape[0]:
        raise IndexError("reference factor index is out of range")
    if min(candidate_indices) < 0 or max(candidate_indices) >= candidate_signatures.shape[0]:
        raise IndexError("candidate factor index is out of range")
    if reference_signatures.dtype != candidate_signatures.dtype or reference_signatures.device != candidate_signatures.device:
        raise TypeError("subspace signatures must share dtype and device")
    if not reference_signatures.is_floating_point():
        raise TypeError("subspace signatures must be floating point")
    if not bool(torch.isfinite(reference_signatures).all()) or not bool(
        torch.isfinite(candidate_signatures).all()
    ):
        raise ValueError("subspace signatures must be finite")
    reference_block = reference_signatures[list(reference_indices)].T
    candidate_block = candidate_signatures[list(candidate_indices)].T
    reference_basis = torch.linalg.svd(reference_block, full_matrices=False).U
    candidate_basis = torch.linalg.svd(candidate_block, full_matrices=False).U
    reference_rank = int(torch.linalg.matrix_rank(reference_block).item())
    candidate_rank = int(torch.linalg.matrix_rank(candidate_block).item())
    if reference_rank == 0 or candidate_rank == 0:
        raise ValueError("a zero-rank block has no interpretable subspace")
    reference_basis = reference_basis[:, :reference_rank]
    candidate_basis = candidate_basis[:, :candidate_rank]
    singular_values = torch.linalg.svdvals(reference_basis.T @ candidate_basis).clamp(0.0, 1.0)
    angles = torch.acos(singular_values)
    # Unequal-rank blocks have unmatched orthogonal directions at pi/2.
    if reference_rank != candidate_rank:
        missing = abs(reference_rank - candidate_rank)
        angles = torch.cat(
            [angles, torch.full((missing,), math.pi / 2, dtype=angles.dtype, device=angles.device)]
        )
    degrees = torch.rad2deg(angles)
    chordal = torch.sqrt(torch.sin(angles).square().sum())
    degree_values = tuple(float(value) for value in degrees.detach().cpu())
    return PrincipalAngleReport(
        reference_factor_indices=reference_indices,
        candidate_factor_indices=candidate_indices,
        angles_degrees=degree_values,
        max_angle_degrees=max(degree_values),
        mean_angle_degrees=sum(degree_values) / len(degree_values),
        chordal_distance=float(chordal.detach().cpu()),
        reference_rank=reference_rank,
        candidate_rank=candidate_rank,
    )


def classify_factor_modality_profiles(
    *,
    eye_amplitude: torch.Tensor,
    blood_amplitude: torch.Tensor,
    eye_expected_fisher: torch.Tensor,
    blood_expected_fisher: torch.Tensor,
    identifiable_axis: torch.Tensor,
    thresholds: FactorProfileThresholds,
) -> tuple[FactorModalityProfile, ...]:
    """Describe view dominance using both amplitudes and held-out relevance.

    The combined share is the normalized geometric mean of amplitude share and
    expected-Fisher share.  Threshold gates remain separate, so a large loading
    without held-out information (or vice versa) cannot declare a view active.
    An unresolved factor can receive a descriptive profile but can never become
    eligible for individual-axis interpretation.
    """

    latent_dim = _validate_factor_tensors(
        [eye_amplitude, blood_amplitude, eye_expected_fisher, blood_expected_fisher]
    )
    if identifiable_axis.shape != (latent_dim,) or identifiable_axis.dtype != torch.bool:
        raise TypeError("identifiable_axis must be boolean with shape [latent_dim]")
    if identifiable_axis.device != eye_amplitude.device:
        raise ValueError("identifiable_axis must share the factor-statistic device")
    nonnegative = (eye_amplitude, blood_amplitude, eye_expected_fisher, blood_expected_fisher)
    if any(bool((tensor < 0).any()) for tensor in nonnegative):
        raise ValueError("amplitudes and expected Fisher relevance must be nonnegative")

    amplitude_total = (eye_amplitude + blood_amplitude).clamp(min=1e-12)
    relevance_total = (eye_expected_fisher + blood_expected_fisher).clamp(min=1e-12)
    eye_score = torch.sqrt(
        (eye_amplitude / amplitude_total) * (eye_expected_fisher / relevance_total)
    )
    blood_score = torch.sqrt(
        (blood_amplitude / amplitude_total) * (blood_expected_fisher / relevance_total)
    )
    score_total = (eye_score + blood_score).clamp(min=1e-12)
    eye_share = eye_score / score_total
    blood_share = blood_score / score_total

    result: list[FactorModalityProfile] = []
    for factor in range(latent_dim):
        eye_active = bool(
            eye_amplitude[factor] >= thresholds.min_amplitude
            and eye_expected_fisher[factor] >= thresholds.min_expected_fisher
        )
        blood_active = bool(
            blood_amplitude[factor] >= thresholds.min_amplitude
            and blood_expected_fisher[factor] >= thresholds.min_expected_fisher
        )
        if eye_active and blood_active:
            if (
                float(eye_share[factor]) >= thresholds.shared_minimum_share
                and float(blood_share[factor]) >= thresholds.shared_minimum_share
            ):
                profile = "shared"
            elif float(eye_share[factor]) > float(blood_share[factor]):
                profile = "eye_dominant"
            else:
                profile = "blood_clinical_dominant"
        elif eye_active:
            profile = "eye_dominant"
        elif blood_active:
            profile = "blood_clinical_dominant"
        elif float(eye_score[factor] + blood_score[factor]) > 0:
            profile = "weakly_mixed"
        else:
            profile = "inactive"
        stable = bool(identifiable_axis[factor])
        result.append(
            FactorModalityProfile(
                factor_index=factor,
                profile=profile,
                eye_amplitude=float(eye_amplitude[factor]),
                blood_amplitude=float(blood_amplitude[factor]),
                eye_expected_fisher=float(eye_expected_fisher[factor]),
                blood_expected_fisher=float(blood_expected_fisher[factor]),
                eye_combined_share=float(eye_share[factor]),
                blood_combined_share=float(blood_share[factor]),
                identifiable_axis=stable,
                axis_interpretation_allowed=stable and profile != "inactive",
            )
        )
    return tuple(result)


def _naming_failures(
    profile: FactorModalityProfile,
    stability: FactorStabilityRecord,
    thresholds: FactorNamingThresholds,
) -> list[str]:
    failures: list[str] = []
    if profile.factor_index != stability.factor_index:
        failures.append("factor_index_mismatch")
    if profile.profile not in _PROFILE_LABELS:
        failures.append("unknown_modality_profile")
    if profile.profile == "inactive":
        failures.append("inactive_factor")
    if not profile.identifiable_axis or not profile.axis_interpretation_allowed:
        failures.append("profile_not_axis_identifiable")
    if not stability.identifiable_axis:
        failures.append("stability_not_axis_identifiable")
    if stability.unresolved_block_id is not None:
        failures.append("unresolved_subspace_block")
    if stability.alignment_method != "signed_permutation":
        failures.append("non_permutation_axis_alignment")
    if stability.matched_replicates < thresholds.min_matched_replicates:
        failures.append("too_few_matched_replicates")
    if stability.min_loading_congruence < thresholds.min_loading_congruence:
        failures.append("loading_congruence_below_threshold")
    if stability.sign_agreement < thresholds.min_sign_agreement:
        failures.append("sign_agreement_below_threshold")
    if stability.min_assignment_margin < thresholds.min_assignment_margin:
        failures.append("assignment_margin_below_threshold")
    if not stability.bootstrap_loading_groups_significant:
        failures.append("bootstrap_loading_groups_not_significant")
    if not stability.confound_audits_passed:
        failures.append("confound_audits_failed")
    if profile.profile == "shared":
        if not stability.eye_recoverable or not stability.blood_recoverable:
            failures.append("shared_factor_not_recoverable_from_each_view")
        if not stability.pairing_null_passed:
            failures.append("shared_factor_pairing_null_failed")
    elif profile.profile == "eye_dominant" and not stability.eye_recoverable:
        failures.append("eye_dominant_factor_not_eye_recoverable")
    elif profile.profile == "blood_clinical_dominant" and not stability.blood_recoverable:
        failures.append("blood_dominant_factor_not_blood_recoverable")
    return failures


def build_factor_naming_artifact(
    *,
    proposed_names: Mapping[int, str],
    profiles: Sequence[FactorModalityProfile],
    stability_records: Sequence[FactorStabilityRecord],
    thresholds: FactorNamingThresholds,
    provenance: InterpretationProvenance,
) -> dict[str, object]:
    """Build an aggregate, fail-closed artifact containing approved axis names.

    Any proposed name that fails the stable-factor contract aborts the entire
    artifact.  Unstable factors should instead remain numbered in a separate
    aggregate report; their rejected narrative labels are intentionally not
    persisted here.
    """

    profile_by_index = {profile.factor_index: profile for profile in profiles}
    stability_by_index = {record.factor_index: record for record in stability_records}
    if len(profile_by_index) != len(profiles) or len(stability_by_index) != len(stability_records):
        raise ValueError("factor profiles and stability records must have unique indices")
    approved: list[dict[str, object]] = []
    forbidden_name_terms = ("causal", "mechanism", "etiologic", "disease subtype")
    for factor_index, proposed_name in sorted(proposed_names.items()):
        if not isinstance(factor_index, int) or factor_index < 0:
            raise ValueError("proposed-name keys must be nonnegative factor indices")
        if factor_index not in profile_by_index or factor_index not in stability_by_index:
            raise ValueError(f"factor {factor_index} lacks profile or stability evidence")
        if not isinstance(proposed_name, str) or not proposed_name.strip():
            raise ValueError(f"factor {factor_index} has an empty proposed name")
        normalized_name = proposed_name.strip()
        lower_name = normalized_name.lower()
        if re.search(r"\baxis\b", lower_name) is None:
            raise ValueError("named factors must be described cautiously as an axis")
        if any(term in lower_name for term in forbidden_name_terms):
            raise ValueError("causal, mechanistic, or subtype names are prohibited")
        profile = profile_by_index[factor_index]
        stability = stability_by_index[factor_index]
        failures = _naming_failures(profile, stability, thresholds)
        if failures:
            raise ValueError(
                f"factor {factor_index} cannot be named: {', '.join(failures)}"
            )
        approved.append(
            {
                "factor_index": factor_index,
                "approved_name": normalized_name,
                "modality_profile": profile.profile,
                "eye_combined_share": profile.eye_combined_share,
                "blood_combined_share": profile.blood_combined_share,
                "matched_replicates": stability.matched_replicates,
                "min_loading_congruence": stability.min_loading_congruence,
                "sign_agreement": stability.sign_agreement,
                "min_assignment_margin": stability.min_assignment_margin,
                "bootstrap_loading_groups_significant": True,
                "confound_audits_passed": True,
                "eye_recoverable": stability.eye_recoverable,
                "blood_recoverable": stability.blood_recoverable,
                "pairing_null_passed": stability.pairing_null_passed,
                "unresolved_block_id": None,
                "alignment_method": "signed_permutation",
            }
        )

    payload: dict[str, object] = {
        "schema_version": "patient-atlas-factor-naming-v1",
        "axis_alignment": "signed_permutation_only",
        "rotation_used_for_axis_naming": False,
        "unresolved_blocks_named": False,
        "thresholds": asdict(thresholds),
        "provenance": asdict(provenance),
        "named_axes": approved,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    validate_factor_naming_artifact(payload)
    return payload


def validate_factor_naming_artifact(artifact: Mapping[str, object]) -> None:
    """Validate the immutable safety and digest fields of a naming artifact."""

    required = {
        "schema_version",
        "axis_alignment",
        "rotation_used_for_axis_naming",
        "unresolved_blocks_named",
        "thresholds",
        "provenance",
        "named_axes",
        "artifact_sha256",
    }
    if set(artifact) != required:
        raise ValueError("factor naming artifact has missing or unexpected fields")
    if artifact["schema_version"] != "patient-atlas-factor-naming-v1":
        raise ValueError("factor naming artifact schema mismatch")
    if artifact["axis_alignment"] != "signed_permutation_only":
        raise ValueError("axis naming must use signed permutation only")
    if artifact["rotation_used_for_axis_naming"] is not False:
        raise ValueError("rotations may not be used for axis naming")
    if artifact["unresolved_blocks_named"] is not False:
        raise ValueError("unresolved subspace blocks may not be named as axes")
    thresholds_payload = artifact["thresholds"]
    provenance_payload = artifact["provenance"]
    if not isinstance(thresholds_payload, Mapping) or not isinstance(
        provenance_payload, Mapping
    ):
        raise TypeError("artifact thresholds and provenance must be mappings")
    try:
        thresholds = FactorNamingThresholds(**dict(thresholds_payload))
        InterpretationProvenance(**dict(provenance_payload))
    except TypeError as error:
        raise ValueError("artifact thresholds or provenance schema mismatch") from error
    named_axes = artifact["named_axes"]
    if not isinstance(named_axes, list):
        raise TypeError("named_axes must be a list")
    named_axis_fields = {
        "factor_index",
        "approved_name",
        "modality_profile",
        "eye_combined_share",
        "blood_combined_share",
        "matched_replicates",
        "min_loading_congruence",
        "sign_agreement",
        "min_assignment_margin",
        "bootstrap_loading_groups_significant",
        "confound_audits_passed",
        "eye_recoverable",
        "blood_recoverable",
        "pairing_null_passed",
        "unresolved_block_id",
        "alignment_method",
    }
    seen_indices: set[int] = set()
    for entry in named_axes:
        if not isinstance(entry, Mapping) or set(entry) != named_axis_fields:
            raise ValueError("named-axis entry schema mismatch")
        factor_index = entry["factor_index"]
        if not isinstance(factor_index, int) or isinstance(factor_index, bool) or factor_index < 0:
            raise ValueError("named-axis factor indices must be nonnegative integers")
        if factor_index in seen_indices:
            raise ValueError("named-axis factor indices must be unique")
        seen_indices.add(factor_index)
        name = entry["approved_name"]
        if not isinstance(name, str) or re.search(r"\baxis\b", name.lower()) is None:
            raise ValueError("approved factor names must use cautious axis language")
        profile = entry["modality_profile"]
        if profile not in _PROFILE_LABELS or profile == "inactive":
            raise ValueError("named axis has an invalid modality profile")
        for share_name in ("eye_combined_share", "blood_combined_share"):
            share = entry[share_name]
            if not isinstance(share, (float, int)) or isinstance(share, bool):
                raise TypeError("modality shares must be numeric")
            if not math.isfinite(float(share)) or not 0 <= float(share) <= 1:
                raise ValueError("modality shares must lie in [0, 1]")
        if not math.isclose(
            float(entry["eye_combined_share"]) + float(entry["blood_combined_share"]),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise ValueError("named-axis modality shares must sum to one")
        if entry["alignment_method"] != "signed_permutation":
            raise ValueError("named axes must use signed-permutation alignment")
        if entry["unresolved_block_id"] is not None:
            raise ValueError("an unresolved subspace axis cannot be named")
        for gate in (
            "bootstrap_loading_groups_significant",
            "confound_audits_passed",
        ):
            if entry[gate] is not True:
                raise ValueError(f"named-axis gate failed: {gate}")
        if not isinstance(entry["matched_replicates"], int) or isinstance(
            entry["matched_replicates"], bool
        ):
            raise TypeError("matched_replicates must be an integer")
        if entry["matched_replicates"] < thresholds.min_matched_replicates:
            raise ValueError("named axis has too few matched replicates")
        numeric_gates = (
            ("min_loading_congruence", thresholds.min_loading_congruence),
            ("sign_agreement", thresholds.min_sign_agreement),
            ("min_assignment_margin", thresholds.min_assignment_margin),
        )
        for gate, minimum in numeric_gates:
            value = entry[gate]
            if not isinstance(value, (float, int)) or isinstance(value, bool):
                raise TypeError(f"{gate} must be numeric")
            if not math.isfinite(float(value)) or not minimum <= float(value) <= 1:
                raise ValueError(f"named-axis gate failed: {gate}")
        for recoverability in (
            "eye_recoverable",
            "blood_recoverable",
            "pairing_null_passed",
        ):
            if not isinstance(entry[recoverability], bool):
                raise TypeError(f"{recoverability} must be boolean")
        if profile == "shared" and not (
            entry["eye_recoverable"]
            and entry["blood_recoverable"]
            and entry["pairing_null_passed"]
        ):
            raise ValueError("shared named axes must pass both-view and pairing-null gates")
        if profile == "eye_dominant" and not entry["eye_recoverable"]:
            raise ValueError("eye-dominant named axes must be eye recoverable")
        if profile == "blood_clinical_dominant" and not entry["blood_recoverable"]:
            raise ValueError("blood-dominant named axes must be blood recoverable")
    claimed = artifact["artifact_sha256"]
    _require_sha256("artifact_sha256", claimed)  # type: ignore[arg-type]
    unhashed = dict(artifact)
    del unhashed["artifact_sha256"]
    if _canonical_sha256(unhashed) != claimed:
        raise ValueError("factor naming artifact hash mismatch")


__all__ = [
    "FactorModalityProfile",
    "FactorNamingThresholds",
    "FactorProfileThresholds",
    "FactorRelevanceReport",
    "FactorStabilityRecord",
    "InterpretationProvenance",
    "NaturalEvidenceContributions",
    "PrincipalAngleReport",
    "SignedPermutationMatch",
    "align_identifiable_axes",
    "build_factor_naming_artifact",
    "classify_factor_modality_profiles",
    "heldout_factor_relevance",
    "likelihood_normalized_loading_signatures",
    "match_signed_permutation_axes",
    "natural_evidence_contributions",
    "principal_angle_subspace_report",
    "validate_factor_naming_artifact",
]
