"""Synthetic/data-agnostic Stage-2 training utilities for the patient atlas.

This module deliberately knows nothing about file formats, outcomes, or real
patients.  Its responsibilities are limited to deterministic ID splitting,
policy-safe observation corruption, and one optimizer step against the public
``SoftPatientAtlas`` training API.  Callers must construct tensors only after
all authenticated transforms and feature-policy checks have succeeded.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn


MODE_BOTH = 0
MODE_EYE_ONLY = 1
MODE_BLOOD_ONLY = 2
MODE_NAMES = ("both", "eye_only", "blood_only")


@dataclass(frozen=True)
class Stage2TrainingConfig:
    """Frozen Stage-2 defaults from ``PATIENT_ATLAS_SPEC.md``."""

    fit_fraction: float = 0.70
    validation_fraction: float = 0.15
    calibration_fraction: float = 0.15
    both_probability: float = 0.40
    eye_only_probability: float = 0.30
    blood_only_probability: float = 0.30
    batch_size: int = 96
    max_steps: int = 6_000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    validation_interval: int = 250
    early_stopping_patience: int = 8
    kl_warmup_fraction: float = 0.20
    final_kl_weight: float = 1.0
    minimum_item_keep_probability: float = 0.50
    maximum_item_keep_probability: float = 1.00
    sample_count: int = 1
    orthogonality_weight: float = 1e-3
    anchor_weight: float = 0.0
    interaction_enabled: bool = False
    split_salt: str = "soft-patient-atlas-stage2-v1"

    def __post_init__(self) -> None:
        split_total = (
            self.fit_fraction
            + self.validation_fraction
            + self.calibration_fraction
        )
        mode_total = (
            self.both_probability
            + self.eye_only_probability
            + self.blood_only_probability
        )
        if not math.isclose(split_total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("fit/validation/calibration fractions must sum to one")
        if not math.isclose(mode_total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("input-mode probabilities must sum to one")
        probabilities = (
            self.fit_fraction,
            self.validation_fraction,
            self.calibration_fraction,
            self.both_probability,
            self.eye_only_probability,
            self.blood_only_probability,
        )
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("fractions and probabilities must lie in [0, 1]")
        if not (
            0.0
            <= self.minimum_item_keep_probability
            <= self.maximum_item_keep_probability
            <= 1.0
        ):
            raise ValueError("item keep-probability bounds must lie in [0, 1]")
        positive_integers = {
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "validation_interval": self.validation_interval,
            "early_stopping_patience": self.early_stopping_patience,
            "sample_count": self.sample_count,
        }
        for name, value in positive_integers.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0.0 < self.kl_warmup_fraction <= 1.0:
            raise ValueError("kl_warmup_fraction must lie in (0, 1]")
        if self.final_kl_weight <= 0:
            raise ValueError("final_kl_weight must be positive")
        if self.orthogonality_weight < 0:
            raise ValueError("orthogonality_weight must be nonnegative")
        if not 0.0 <= self.anchor_weight <= 0.25:
            raise ValueError("anchor_weight must lie in [0, 0.25]")
        if not self.split_salt:
            raise ValueError("split_salt must not be empty")


@dataclass(frozen=True)
class PatientIdSplit:
    """In-memory ID membership; never include this object in released metrics."""

    fit: tuple[str, ...]
    validation: tuple[str, ...]
    calibration: tuple[str, ...]

    @property
    def counts(self) -> Mapping[str, int]:
        return {
            "fit": len(self.fit),
            "validation": len(self.validation),
            "calibration": len(self.calibration),
        }


@dataclass(frozen=True)
class AtlasTrainingBatch:
    """A genuinely observed batch before encoder corruption.

    ``eye_observed_mask`` and ``blood_observed_mask`` describe collected data.
    The blood policy is applied again inside :func:`corrupt_observations`, so a
    caller cannot reveal a forbidden field merely by setting its observed bit.
    """

    eye_embeddings: torch.Tensor
    eye_observed_mask: torch.Tensor
    blood_values: torch.Tensor
    blood_observed_mask: torch.Tensor
    blood_eligible_mask: torch.Tensor
    demographics: torch.Tensor
    demographic_mask: torch.Tensor
    eye_device_ids: torch.Tensor
    eye_laterality_ids: torch.Tensor
    eye_quality: torch.Tensor | None = None
    target_blood_anchor: torch.Tensor | None = None


@dataclass(frozen=True)
class CorruptedAtlasBatch:
    """Policy-safe batch with target set ``A`` separate from visible set ``O``."""

    encoder_eye_embeddings: torch.Tensor
    encoder_eye_visible_mask: torch.Tensor
    encoder_blood_values: torch.Tensor
    encoder_blood_visible_mask: torch.Tensor
    blood_eligible_mask: torch.Tensor
    demographics: torch.Tensor
    demographic_mask: torch.Tensor
    encoder_eye_device_ids: torch.Tensor
    encoder_eye_laterality_ids: torch.Tensor
    encoder_eye_quality: torch.Tensor | None
    target_eye_embeddings: torch.Tensor
    target_eye_mask: torch.Tensor
    target_blood_values: torch.Tensor
    target_blood_mask: torch.Tensor
    target_eye_device_ids: torch.Tensor
    target_eye_laterality_ids: torch.Tensor
    target_blood_anchor: torch.Tensor | None
    mode_codes: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.encoder_eye_embeddings.shape[0])


def _stable_digest(salt: str, site: str, patient_id: str) -> bytes:
    payload = "\x1f".join((salt, site, patient_id)).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _largest_remainder_counts(size: int, fractions: Sequence[float]) -> list[int]:
    raw = [size * fraction for fraction in fractions]
    counts = [math.floor(value) for value in raw]
    remaining = size - sum(counts)
    order = sorted(
        range(len(fractions)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def assert_disjoint_split(
    split: PatientIdSplit,
    *,
    expected_patient_ids: Sequence[str] | None = None,
) -> None:
    """Fail closed on duplicate, overlapping, missing, or unexpected IDs."""

    memberships = {
        "fit": split.fit,
        "validation": split.validation,
        "calibration": split.calibration,
    }
    sets: dict[str, set[str]] = {}
    for name, members in memberships.items():
        if len(members) != len(set(members)):
            raise ValueError(f"duplicate patient ID inside {name} split")
        sets[name] = set(members)
    names = tuple(sets)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            if sets[left_name] & sets[right_name]:
                raise ValueError(
                    f"patient identity crosses {left_name} and {right_name} splits"
                )
    if expected_patient_ids is not None:
        expected = tuple(str(value) for value in expected_patient_ids)
        if len(expected) != len(set(expected)):
            raise ValueError("expected patient IDs contain duplicates")
        union = set().union(*sets.values())
        if union != set(expected):
            raise ValueError("split membership does not equal the expected patient set")


def deterministic_patient_split(
    patient_ids: Sequence[str],
    site_ids: Sequence[str],
    config: Stage2TrainingConfig = Stage2TrainingConfig(),
) -> PatientIdSplit:
    """Assign deterministic 70/15/15 membership independently within each site.

    Target labels are neither accepted nor inspected.  Sorting by a salted
    SHA-256 digest makes the result independent of source-row order.  Hamilton
    allocation makes the three per-site counts sum exactly to each stratum.
    """

    if len(patient_ids) != len(site_ids):
        raise ValueError("patient_ids and site_ids must have identical length")
    normalized_ids = tuple(str(value) for value in patient_ids)
    normalized_sites = tuple(str(value) for value in site_ids)
    if not normalized_ids:
        raise ValueError("at least one patient ID is required")
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("patient IDs must be globally unique")
    if any(not value for value in normalized_ids):
        raise ValueError("patient IDs must not be empty")
    if any(not value for value in normalized_sites):
        raise ValueError("site IDs must not be empty")

    by_site: dict[str, list[str]] = {}
    for patient_id, site in zip(normalized_ids, normalized_sites):
        by_site.setdefault(site, []).append(patient_id)

    assignments: list[list[str]] = [[], [], []]
    fractions = (
        config.fit_fraction,
        config.validation_fraction,
        config.calibration_fraction,
    )
    for site in sorted(by_site):
        ordered = sorted(
            by_site[site],
            key=lambda patient_id: (
                _stable_digest(config.split_salt, site, patient_id), patient_id
            ),
        )
        counts = _largest_remainder_counts(len(ordered), fractions)
        start = 0
        for destination, count in zip(assignments, counts):
            destination.extend(ordered[start : start + count])
            start += count

    split = PatientIdSplit(*(tuple(sorted(values)) for values in assignments))
    assert_disjoint_split(split, expected_patient_ids=normalized_ids)
    return split


def kl_warmup_beta(step: int, config: Stage2TrainingConfig) -> float:
    """Linear 0-to-1 KL warmup over the first 20% of configured steps."""

    if not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    warmup_steps = max(1, math.ceil(config.max_steps * config.kl_warmup_fraction))
    progress = min(1.0, float(step) / float(warmup_steps))
    return config.final_kl_weight * progress


def _validate_source_batch(batch: AtlasTrainingBatch) -> None:
    eye = batch.eye_embeddings
    blood = batch.blood_values
    if eye.ndim != 3 or blood.ndim != 2:
        raise ValueError("eye and blood values must have [B,N,D] and [B,F] shapes")
    size, images, _ = eye.shape
    if blood.shape[0] != size:
        raise ValueError("eye and blood batch sizes differ")
    expected_shapes = (
        ("eye_observed_mask", batch.eye_observed_mask, (size, images)),
        ("blood_observed_mask", batch.blood_observed_mask, blood.shape),
        ("blood_eligible_mask", batch.blood_eligible_mask, blood.shape),
        ("demographic_mask", batch.demographic_mask, batch.demographics.shape),
    )
    for name, tensor, shape in expected_shapes:
        if tensor.shape != shape or tensor.dtype != torch.bool:
            raise TypeError(f"{name} must be boolean with shape {tuple(shape)}")
    if batch.demographics.ndim != 2 or batch.demographics.shape[0] != size:
        raise ValueError("demographics must have shape [B,D]")
    floating = (eye, blood, batch.demographics)
    if any(not tensor.is_floating_point() for tensor in floating):
        raise TypeError("eye, blood, and demographic values must be floating point")
    if any(tensor.dtype != eye.dtype or tensor.device != eye.device for tensor in floating):
        raise TypeError("all floating values must share one dtype and device")
    masks = (
        batch.eye_observed_mask,
        batch.blood_observed_mask,
        batch.blood_eligible_mask,
        batch.demographic_mask,
    )
    if any(tensor.device != eye.device for tensor in masks):
        raise ValueError("all masks must share the value device")
    if size > 1 and not torch.equal(
        batch.blood_eligible_mask,
        batch.blood_eligible_mask[:1].expand_as(batch.blood_eligible_mask),
    ):
        raise ValueError("blood eligibility is an artifact policy, not patient state")

    integer_metadata = (
        ("eye_device_ids", batch.eye_device_ids),
        ("eye_laterality_ids", batch.eye_laterality_ids),
    )
    for name, values in integer_metadata:
        if values is None or (
            values.shape != (size, images)
            or values.dtype != torch.long
            or values.device != eye.device
        ):
            raise TypeError(f"{name} must be long with shape [B,N] on the value device")
    if batch.eye_quality is not None and (
        batch.eye_quality.shape != (size, images)
        or batch.eye_quality.dtype != eye.dtype
        or batch.eye_quality.device != eye.device
    ):
        raise TypeError("eye_quality must be floating [B,N] on the value device")
    if batch.target_blood_anchor is not None and (
        batch.target_blood_anchor.ndim != 2
        or batch.target_blood_anchor.shape[0] != size
        or batch.target_blood_anchor.dtype != eye.dtype
        or batch.target_blood_anchor.device != eye.device
    ):
        raise TypeError("target_blood_anchor must be floating [B,A] on the value device")


def _mode_codes(
    size: int,
    device: torch.device,
    config: Stage2TrainingConfig,
    modes: Sequence[str] | torch.Tensor | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if modes is None:
        probabilities = torch.tensor(
            [
                config.both_probability,
                config.eye_only_probability,
                config.blood_only_probability,
            ],
            dtype=torch.float64,
            device=device,
        )
        return torch.multinomial(
            probabilities, size, replacement=True, generator=generator
        ).to(dtype=torch.long)
    if isinstance(modes, torch.Tensor):
        codes = modes.to(device=device)
        if codes.shape != (size,) or codes.dtype != torch.long:
            raise TypeError("mode tensor must be long with shape [B]")
    else:
        if len(modes) != size:
            raise ValueError("mode sequence must contain one name per patient")
        lookup = {name: index for index, name in enumerate(MODE_NAMES)}
        try:
            codes = torch.tensor(
                [lookup[str(name)] for name in modes],
                dtype=torch.long,
                device=device,
            )
        except KeyError as error:
            raise ValueError(f"unknown input mode: {error.args[0]}") from error
    if codes.numel() and (int(codes.min()) < MODE_BOTH or int(codes.max()) > MODE_BLOOD_ONLY):
        raise ValueError("mode codes must be 0 (both), 1 (eye), or 2 (blood)")
    return codes


def _random_partial_keep_mask(
    shape: torch.Size,
    *,
    device: torch.device,
    config: Stage2TrainingConfig,
    generator: torch.Generator | None,
) -> torch.Tensor:
    size = shape[0]
    lower = config.minimum_item_keep_probability
    upper = config.maximum_item_keep_probability
    patient_probability = lower + (upper - lower) * torch.rand(
        (size, 1), device=device, generator=generator
    )
    return torch.rand(shape, device=device, generator=generator) < patient_probability


def corrupt_observations(
    batch: AtlasTrainingBatch,
    config: Stage2TrainingConfig = Stage2TrainingConfig(),
    *,
    generator: torch.Generator | None = None,
    modes: Sequence[str] | torch.Tensor | None = None,
    eye_partial_keep_mask: torch.Tensor | None = None,
    blood_partial_keep_mask: torch.Tensor | None = None,
) -> CorruptedAtlasBatch:
    """Create encoder-visible ``O`` while preserving eligible targets ``A``.

    Explicit partial masks can carry empirical availability patterns.  When
    omitted, each patient receives a graded random keep probability followed by
    item-wise deletion.  Every value behind a false target or visibility mask is
    physically replaced by zero before being returned.
    """

    _validate_source_batch(batch)
    device = batch.eye_embeddings.device
    size = batch.eye_embeddings.shape[0]
    codes = _mode_codes(size, device, config, modes, generator)

    target_eye_mask = batch.eye_observed_mask.clone()
    target_blood_mask = batch.blood_observed_mask & batch.blood_eligible_mask

    if eye_partial_keep_mask is None:
        eye_keep = _random_partial_keep_mask(
            target_eye_mask.shape,
            device=device,
            config=config,
            generator=generator,
        )
    else:
        if (
            eye_partial_keep_mask.shape != target_eye_mask.shape
            or eye_partial_keep_mask.dtype != torch.bool
            or eye_partial_keep_mask.device != device
        ):
            raise TypeError("eye_partial_keep_mask must be boolean [B,N] on the value device")
        eye_keep = eye_partial_keep_mask
    if blood_partial_keep_mask is None:
        blood_keep = _random_partial_keep_mask(
            target_blood_mask.shape,
            device=device,
            config=config,
            generator=generator,
        )
    else:
        if (
            blood_partial_keep_mask.shape != target_blood_mask.shape
            or blood_partial_keep_mask.dtype != torch.bool
            or blood_partial_keep_mask.device != device
        ):
            raise TypeError(
                "blood_partial_keep_mask must be boolean [B,F] on the value device"
            )
        blood_keep = blood_partial_keep_mask

    eye_allowed = codes != MODE_BLOOD_ONLY
    blood_allowed = codes != MODE_EYE_ONLY
    eye_visible = target_eye_mask & eye_keep & eye_allowed[:, None]
    blood_visible = target_blood_mask & blood_keep & blood_allowed[:, None]

    target_eye = torch.where(
        target_eye_mask[..., None],
        batch.eye_embeddings,
        torch.zeros_like(batch.eye_embeddings),
    )
    target_blood = torch.where(
        target_blood_mask,
        batch.blood_values,
        torch.zeros_like(batch.blood_values),
    )
    encoder_eye = torch.where(
        eye_visible[..., None], target_eye, torch.zeros_like(target_eye)
    )
    encoder_blood = torch.where(
        blood_visible, target_blood, torch.zeros_like(target_blood)
    )
    safe_demographics = torch.where(
        batch.demographic_mask,
        batch.demographics,
        torch.zeros_like(batch.demographics),
    )

    def sanitize_integer_metadata(
        values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.where(mask, values, torch.zeros_like(values))

    target_device = sanitize_integer_metadata(
        batch.eye_device_ids, target_eye_mask
    )
    target_laterality = sanitize_integer_metadata(
        batch.eye_laterality_ids, target_eye_mask
    )
    encoder_device = sanitize_integer_metadata(batch.eye_device_ids, eye_visible)
    encoder_laterality = sanitize_integer_metadata(
        batch.eye_laterality_ids, eye_visible
    )
    encoder_quality = (
        None
        if batch.eye_quality is None
        else torch.where(
            eye_visible, batch.eye_quality, torch.zeros_like(batch.eye_quality)
        )
    )
    target_anchor = None
    if batch.target_blood_anchor is not None:
        blood_present = target_blood_mask.any(dim=1, keepdim=True)
        target_anchor = torch.where(
            blood_present,
            batch.target_blood_anchor,
            torch.zeros_like(batch.target_blood_anchor),
        )

    if bool((eye_visible & ~target_eye_mask).any()):
        raise RuntimeError("internal error: eye O is not a subset of A")
    if bool((blood_visible & ~target_blood_mask).any()):
        raise RuntimeError("internal error: blood O is not a subset of A")
    if bool((target_blood_mask & ~batch.blood_eligible_mask).any()):
        raise RuntimeError("internal error: policy-forbidden target survived")

    return CorruptedAtlasBatch(
        encoder_eye_embeddings=encoder_eye,
        encoder_eye_visible_mask=eye_visible,
        encoder_blood_values=encoder_blood,
        encoder_blood_visible_mask=blood_visible,
        blood_eligible_mask=batch.blood_eligible_mask.clone(),
        demographics=safe_demographics,
        demographic_mask=batch.demographic_mask.clone(),
        encoder_eye_device_ids=encoder_device,
        encoder_eye_laterality_ids=encoder_laterality,
        encoder_eye_quality=encoder_quality,
        target_eye_embeddings=target_eye,
        target_eye_mask=target_eye_mask,
        target_blood_values=target_blood,
        target_blood_mask=target_blood_mask,
        target_eye_device_ids=target_device,
        target_eye_laterality_ids=target_laterality,
        target_blood_anchor=target_anchor,
        mode_codes=codes,
    )


def build_stage2_optimizer(
    model: nn.Module,
    config: Stage2TrainingConfig = Stage2TrainingConfig(),
) -> torch.optim.AdamW:
    """Construct AdamW exclusively through the model's parameter-group API."""

    group_builder = getattr(model, "optimizer_parameter_groups", None)
    if not callable(group_builder):
        raise TypeError("model must expose optimizer_parameter_groups(weight_decay)")
    if not config.interaction_enabled:
        locker = getattr(model, "lock_interaction", None)
        if not callable(locker):
            raise TypeError("model must expose lock_interaction() for Stage 2")
        locker()
    groups = group_builder(config.weight_decay)
    return torch.optim.AdamW(groups, lr=config.learning_rate)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: CorruptedAtlasBatch,
    config: Stage2TrainingConfig,
    *,
    step: int,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Run one additive Stage-2 update and return aggregate scalar metrics."""

    beta = kl_warmup_beta(step, config)
    projector = getattr(model, "project_parameters_", None)
    if not callable(projector):
        raise TypeError("model must expose project_parameters_()")
    if not config.interaction_enabled:
        locker = getattr(model, "lock_interaction", None)
        if not callable(locker):
            raise TypeError("model must expose lock_interaction() for Stage 2")
        locker()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        eye_embeddings=batch.encoder_eye_embeddings,
        eye_visible_mask=batch.encoder_eye_visible_mask,
        blood_values=batch.encoder_blood_values,
        blood_visible_mask=batch.encoder_blood_visible_mask,
        blood_eligible_mask=batch.blood_eligible_mask,
        demographics=batch.demographics,
        demographic_mask=batch.demographic_mask,
        eye_device_ids=batch.encoder_eye_device_ids,
        eye_laterality_ids=batch.encoder_eye_laterality_ids,
        eye_quality=batch.encoder_eye_quality,
        enable_interaction=config.interaction_enabled,
    )
    objective = getattr(model, "variational_loss", None)
    if not callable(objective):
        raise TypeError("model must expose variational_loss(output, ...)")
    losses = objective(
        output,
        target_eye_embeddings=batch.target_eye_embeddings,
        target_eye_mask=batch.target_eye_mask,
        target_blood_values=batch.target_blood_values,
        target_blood_mask=batch.target_blood_mask,
        target_blood_eligible_mask=batch.blood_eligible_mask,
        target_eye_device_ids=batch.target_eye_device_ids,
        target_eye_laterality_ids=batch.target_eye_laterality_ids,
        target_blood_anchor=batch.target_blood_anchor,
        anchor_weight=config.anchor_weight,
        sample_count=config.sample_count,
        beta=beta,
        orthogonality_weight=config.orthogonality_weight,
        generator=generator,
    )
    if "total" not in losses:
        raise KeyError("model.variational_loss must return a 'total' scalar")
    for name, value in losses.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            raise TypeError(f"variational metric {name!r} is not an aggregate scalar")
        if not bool(torch.isfinite(value)):
            raise FloatingPointError(f"nonfinite variational metric: {name}")
    total = losses["total"]
    total.backward()
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable, config.gradient_clip_norm
    )
    if not bool(torch.isfinite(gradient_norm)):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("nonfinite gradient norm")
    optimizer.step()
    projector()

    metrics: dict[str, float] = {
        "kl_beta": float(beta),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "batch_size": float(batch.batch_size),
    }
    for name, value in losses.items():
        metrics[f"loss_{name}"] = float(value.detach().cpu())
    for code, name in enumerate(MODE_NAMES):
        metrics[f"mode_{name}_count"] = float((batch.mode_codes == code).sum().item())
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("nonfinite aggregate training metric")
    return metrics


__all__ = [
    "AtlasTrainingBatch",
    "CorruptedAtlasBatch",
    "MODE_BLOOD_ONLY",
    "MODE_BOTH",
    "MODE_EYE_ONLY",
    "MODE_NAMES",
    "PatientIdSplit",
    "Stage2TrainingConfig",
    "assert_disjoint_split",
    "build_stage2_optimizer",
    "corrupt_observations",
    "deterministic_patient_split",
    "kl_warmup_beta",
    "train_step",
]
