"""Outcome-free Stage-2 fitting and observable calibration for Patient Atlas.

The orchestration in this module is deliberately data-source agnostic.  It
accepts already authenticated tensors and identifiers, never accepts an
outcome, and releases only aggregate metrics plus hashes of identifier sets.
Model weights are learned from the representation-fit partition, selected on
the validation partition, and frozen before the calibration partition is
opened.

Calibration is an overlay rather than a mutation of the fitted atlas:

* an availability-pattern precision temperature divides *both* the posterior
  evidence precision increment and natural parameter by the same value;
* observable predictive scale multipliers operate only in decoder space; and
* split-conformal residual quantiles are estimated on identities disjoint from
  the identities used to fit the first two calibration layers.

The selected state hash uses the same canonical state-dictionary hashing
routine as the authenticated checkpoint implementation.  Callers can put
``Stage2FitResult.checkpoint_parameters()`` directly into the safe checkpoint
``calibration_parameters`` field after supplying their fold preprocessor and
remaining provenance contracts.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, is_dataclass, replace
import hashlib
import math
from statistics import NormalDist
from typing import Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from patient_atlas_contracts import hash_state_dict
from patient_atlas_preprocessing import hash_json
from soft_patient_atlas import (
    ClinicalPrediction,
    EyePrediction,
    GaussianState,
    PatientStateOutput,
)
from train_soft_patient_atlas import (
    AtlasTrainingBatch,
    CorruptedAtlasBatch,
    MODE_NAMES,
    PatientIdSplit,
    Stage2TrainingConfig,
    _validate_source_batch,
    assert_disjoint_split,
    build_stage2_optimizer,
    corrupt_observations,
    deterministic_patient_split,
    train_step,
)


PRIMARY_PATTERNS = ("both", "eye_only", "blood_only")
_PATTERN_TO_MODE = {name: name for name in PRIMARY_PATTERNS}


def _require_finite_positive_grid(values: Sequence[float], label: str) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if not grid or any(not math.isfinite(value) or value <= 0 for value in grid):
        raise ValueError(f"{label} must be a nonempty finite positive grid")
    if len(grid) != len(set(grid)):
        raise ValueError(f"{label} must not contain duplicates")
    return grid


@dataclass(frozen=True)
class Stage2OrchestrationConfig:
    """Selection and calibration settings outside the optimizer kernel."""

    beta_candidates: tuple[float, ...] = (0.1, 0.3, 1.0)
    group_shrinkage_rate_candidates: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    grid_protocol: str = "initial_v1"
    seed: int = 1701
    score_tie_tolerance: float = 1e-6
    calibration_parameter_fraction: float = 0.50
    precision_temperature_grid: tuple[float, ...] = (
        0.50,
        0.67,
        0.80,
        1.00,
        1.25,
        1.50,
        2.00,
    )
    decoder_scale_grid: tuple[float, ...] = (
        0.50,
        0.67,
        0.80,
        1.00,
        1.25,
        1.50,
        2.00,
    )
    binary_temperature_grid: tuple[float, ...] = (
        0.50,
        0.67,
        0.80,
        1.00,
        1.25,
        1.50,
        2.00,
    )
    conformal_alpha: float = 0.10
    minimum_conformal_patients: int = 20
    calibration_split_salt: str = "soft-patient-atlas-calibration-v1"

    def __post_init__(self) -> None:
        betas = _require_finite_positive_grid(self.beta_candidates, "beta_candidates")
        shrinkage = _require_finite_positive_grid(
            self.group_shrinkage_rate_candidates,
            "group_shrinkage_rate_candidates",
        )
        approved_grids = {
            "initial_v1": ({0.1, 0.3, 1.0}, {1e-4, 1e-3, 1e-2}),
            "lower_bound_expansion_v1": (
                {0.03, 0.1, 0.3},
                {1e-5, 1e-4, 1e-3},
            ),
            "lower_bound_expansion_v2": (
                {0.01, 0.03, 0.1},
                {1e-6, 1e-5, 1e-4},
            ),
            "frozen_selected_v1": ({0.01}, {1e-6}),
        }
        if self.grid_protocol not in approved_grids:
            raise ValueError(
                "grid_protocol must be one of initial_v1, "
                "lower_bound_expansion_v1, lower_bound_expansion_v2, "
                "or frozen_selected_v1"
            )
        expected_betas, expected_shrinkage = approved_grids[self.grid_protocol]
        if set(betas) != expected_betas or set(shrinkage) != expected_shrinkage:
            raise ValueError(
                "Stage-2 candidate grids do not match the selected grid_protocol"
            )
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.score_tie_tolerance < 0 or not math.isfinite(self.score_tie_tolerance):
            raise ValueError("score_tie_tolerance must be finite and nonnegative")
        if not 0.0 < self.calibration_parameter_fraction < 1.0:
            raise ValueError("calibration_parameter_fraction must lie in (0, 1)")
        _require_finite_positive_grid(
            self.precision_temperature_grid, "precision_temperature_grid"
        )
        _require_finite_positive_grid(self.decoder_scale_grid, "decoder_scale_grid")
        _require_finite_positive_grid(
            self.binary_temperature_grid, "binary_temperature_grid"
        )
        if not 0.0 < self.conformal_alpha < 1.0:
            raise ValueError("conformal_alpha must lie in (0, 1)")
        if (
            not isinstance(self.minimum_conformal_patients, int)
            or self.minimum_conformal_patients <= 0
        ):
            raise ValueError("minimum_conformal_patients must be positive")
        if not self.calibration_split_salt:
            raise ValueError("calibration_split_salt must not be empty")


@dataclass(frozen=True)
class AtlasCohort:
    """Authenticated outer-training tensors with non-releasable row identities."""

    patient_ids: tuple[str, ...]
    site_ids: tuple[str, ...]
    observations: AtlasTrainingBatch

    def __post_init__(self) -> None:
        normalized_ids = tuple(str(value) for value in self.patient_ids)
        normalized_sites = tuple(str(value) for value in self.site_ids)
        if normalized_ids != self.patient_ids or normalized_sites != self.site_ids:
            raise TypeError("patient_ids and site_ids must already be string tuples")
        if not normalized_ids or len(normalized_ids) != len(normalized_sites):
            raise ValueError("patient_ids and site_ids must be nonempty and aligned")
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("patient IDs must be unique")
        if any(not value for value in (*normalized_ids, *normalized_sites)):
            raise ValueError("patient and site identifiers must not be empty")
        _validate_source_batch(self.observations)
        if self.observations.eye_embeddings.shape[0] != len(normalized_ids):
            raise ValueError("identifier count does not match observation rows")

    def indices_for(self, patient_ids: Sequence[str]) -> tuple[int, ...]:
        lookup = {patient_id: index for index, patient_id in enumerate(self.patient_ids)}
        try:
            indices = tuple(lookup[str(patient_id)] for patient_id in patient_ids)
        except KeyError as error:
            raise ValueError(f"split contains an unknown patient identity: {error.args[0]}") from error
        if len(indices) != len(set(indices)):
            raise ValueError("requested patient identities are duplicated")
        return indices

    def take(self, indices: Sequence[int]) -> AtlasTrainingBatch:
        if not indices:
            raise ValueError("cannot construct an empty tensor partition")
        batch = self.observations
        index = torch.tensor(
            tuple(int(value) for value in indices),
            dtype=torch.long,
            device=batch.eye_embeddings.device,
        )

        def select(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.index_select(0, index)

        return AtlasTrainingBatch(
            eye_embeddings=select(batch.eye_embeddings),  # type: ignore[arg-type]
            eye_observed_mask=select(batch.eye_observed_mask),  # type: ignore[arg-type]
            blood_values=select(batch.blood_values),  # type: ignore[arg-type]
            blood_observed_mask=select(batch.blood_observed_mask),  # type: ignore[arg-type]
            blood_eligible_mask=select(batch.blood_eligible_mask),  # type: ignore[arg-type]
            demographics=select(batch.demographics),  # type: ignore[arg-type]
            demographic_mask=select(batch.demographic_mask),  # type: ignore[arg-type]
            eye_device_ids=select(batch.eye_device_ids),  # type: ignore[arg-type]
            eye_laterality_ids=select(batch.eye_laterality_ids),  # type: ignore[arg-type]
            eye_quality=select(batch.eye_quality),
            target_blood_anchor=select(batch.target_blood_anchor),
        )


def _identifier_hash(values: Sequence[str]) -> str:
    normalized = tuple(sorted(str(value) for value in values))
    if len(normalized) != len(set(normalized)):
        raise ValueError("identifier set contains duplicates")
    return hash_json(list(normalized))


@dataclass(frozen=True)
class SplitProvenance:
    fit_count: int
    validation_count: int
    calibration_count: int
    fit_id_sha256: str
    validation_id_sha256: str
    calibration_id_sha256: str
    manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_split_provenance(split: PatientIdSplit) -> SplitProvenance:
    assert_disjoint_split(split)
    payload = {
        "fit": _identifier_hash(split.fit),
        "validation": _identifier_hash(split.validation),
        "calibration": _identifier_hash(split.calibration),
        "counts": dict(split.counts),
    }
    return SplitProvenance(
        fit_count=len(split.fit),
        validation_count=len(split.validation),
        calibration_count=len(split.calibration),
        fit_id_sha256=payload["fit"],  # type: ignore[arg-type]
        validation_id_sha256=payload["validation"],  # type: ignore[arg-type]
        calibration_id_sha256=payload["calibration"],  # type: ignore[arg-type]
        manifest_sha256=hash_json(payload),
    )


@dataclass(frozen=True)
class ValidationRecord:
    beta: float
    group_shrinkage_rate: float
    step: int
    balanced_proper_score: float
    calibration_discrepancy: float
    evaluated_patient_patterns: int


@dataclass(frozen=True)
class CandidateSummary:
    beta: float
    group_shrinkage_rate: float
    best_step: int
    stopped_step: int
    validation_checks: int
    best_balanced_proper_score: float
    best_calibration_discrepancy: float
    best_state_sha256: str


@dataclass(frozen=True)
class Stage2SelectionArtifact:
    schema_version: str
    split_provenance: SplitProvenance
    initializer_state_sha256: str
    selected_state_sha256: str
    selected_beta: float
    selected_group_shrinkage_rate: float
    selected_step: int
    hyperparameter_endpoint_status: Mapping[str, str]
    selected_model_config_sha256: str
    candidates: tuple[CandidateSummary, ...]
    history: tuple[ValidationRecord, ...]
    training_config_sha256: str
    orchestration_config_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split_provenance": self.split_provenance.to_dict(),
            "initializer_state_sha256": self.initializer_state_sha256,
            "selected_state_sha256": self.selected_state_sha256,
            "selected_beta": self.selected_beta,
            "selected_group_shrinkage_rate": self.selected_group_shrinkage_rate,
            "selected_step": self.selected_step,
            "hyperparameter_endpoint_status": dict(
                sorted(self.hyperparameter_endpoint_status.items())
            ),
            "selected_model_config_sha256": self.selected_model_config_sha256,
            "candidates": [asdict(value) for value in self.candidates],
            "history": [asdict(value) for value in self.history],
            "training_config_sha256": self.training_config_sha256,
            "orchestration_config_sha256": self.orchestration_config_sha256,
        }


@dataclass(frozen=True)
class DecoderScaleParameters:
    eye_standard_deviation_multiplier: float = 1.0
    continuous_scale_multiplier: float = 1.0
    binary_logit_temperature: float = 1.0
    anchor_standard_deviation_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value <= 0
            for value in asdict(self).values()
        ):
            raise ValueError("decoder calibration scales must be finite and positive")


@dataclass(frozen=True)
class ConformalFamilySidecar:
    status: str
    patient_count: int
    target_coverage: float
    residual_quantile: float | None
    raw_reference_quantile: float
    raw_calibration_patient_coverage: float | None
    conformal_calibration_patient_coverage: float | None
    raw_mean_half_width: float | None
    conformal_mean_half_width: float | None
    sharpness_inflation: float | None


@dataclass(frozen=True)
class CalibrationArtifact:
    schema_version: str
    model_state_sha256: str
    model_config_sha256: str
    calibration_id_sha256: str
    parameter_fit_id_sha256: str
    conformal_fit_id_sha256: str
    parameter_fit_count: int
    conformal_fit_count: int
    precision_temperatures: Mapping[str, float]
    precision_temperature_counts: Mapping[str, int]
    decoder_scales: DecoderScaleParameters
    decoder_scale_counts: Mapping[str, int]
    conformal: Mapping[str, Mapping[str, ConformalFamilySidecar]]
    calibration_config_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_state_sha256": self.model_state_sha256,
            "model_config_sha256": self.model_config_sha256,
            "calibration_id_sha256": self.calibration_id_sha256,
            "parameter_fit_id_sha256": self.parameter_fit_id_sha256,
            "conformal_fit_id_sha256": self.conformal_fit_id_sha256,
            "parameter_fit_count": self.parameter_fit_count,
            "conformal_fit_count": self.conformal_fit_count,
            "precision_temperatures": dict(sorted(self.precision_temperatures.items())),
            "precision_temperature_counts": dict(
                sorted(self.precision_temperature_counts.items())
            ),
            "decoder_scales": asdict(self.decoder_scales),
            "decoder_scale_counts": dict(sorted(self.decoder_scale_counts.items())),
            "conformal": {
                pattern: {
                    family: asdict(sidecar)
                    for family, sidecar in sorted(families.items())
                }
                for pattern, families in sorted(self.conformal.items())
            },
            "calibration_config_sha256": self.calibration_config_sha256,
        }


@dataclass(frozen=True)
class Stage2FitResult:
    """Frozen model plus release-safe selection/calibration sidecars."""

    model: nn.Module
    selection: Stage2SelectionArtifact
    calibration: CalibrationArtifact

    def checkpoint_parameters(self) -> dict[str, object]:
        """Compact, history-free payload accepted by safe checkpoint metadata."""

        selection_value = self.selection.to_dict()
        return {
            "schema_version": "soft-patient-atlas-stage2-checkpoint-parameters-v1",
            "stage2_selection": {
                "selected_beta": self.selection.selected_beta,
                "selected_group_shrinkage_rate": (
                    self.selection.selected_group_shrinkage_rate
                ),
                "selected_step": self.selection.selected_step,
                "hyperparameter_endpoint_status": dict(
                    sorted(self.selection.hyperparameter_endpoint_status.items())
                ),
                "selected_state_sha256": self.selection.selected_state_sha256,
                "selected_model_config_sha256": (
                    self.selection.selected_model_config_sha256
                ),
                "selection_artifact_sha256": hash_json(selection_value),
                "training_config_sha256": self.selection.training_config_sha256,
                "orchestration_config_sha256": (
                    self.selection.orchestration_config_sha256
                ),
            },
            "observable_calibration": self.calibration.to_dict(),
        }


@dataclass(frozen=True)
class Stage2FixedRefitArtifact:
    """Outcome-free provenance for a fixed-recipe post-selection refit."""

    schema_version: str
    split_provenance: SplitProvenance
    initializer_state_sha256: str
    refit_state_sha256: str
    fixed_beta: float
    fixed_group_shrinkage_rate: float
    fixed_step_count: int
    model_config_sha256: str
    validation_balanced_proper_score: float
    validation_calibration_discrepancy: float
    validation_evaluated_patient_patterns: int
    training_config_sha256: str
    orchestration_config_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "soft-patient-atlas-stage2-fixed-refit-v1":
            raise ValueError("unknown fixed-refit artifact schema")
        for name in (
            "initializer_state_sha256",
            "refit_state_sha256",
            "model_config_sha256",
            "training_config_sha256",
            "orchestration_config_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if (
            not math.isfinite(self.fixed_beta)
            or self.fixed_beta <= 0.0
            or not math.isfinite(self.fixed_group_shrinkage_rate)
            or self.fixed_group_shrinkage_rate <= 0.0
        ):
            raise ValueError("fixed refit beta and shrinkage must be finite and positive")
        if not isinstance(self.fixed_step_count, int) or self.fixed_step_count <= 0:
            raise ValueError("fixed refit step count must be positive")
        if not math.isfinite(self.validation_balanced_proper_score):
            raise ValueError("fixed refit validation score must be finite")
        if not math.isfinite(self.validation_calibration_discrepancy):
            raise ValueError("fixed refit calibration discrepancy must be finite")
        if self.validation_evaluated_patient_patterns <= 0:
            raise ValueError("fixed refit validation must evaluate patient patterns")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split_provenance": self.split_provenance.to_dict(),
            "initializer_state_sha256": self.initializer_state_sha256,
            "refit_state_sha256": self.refit_state_sha256,
            "fixed_beta": self.fixed_beta,
            "fixed_group_shrinkage_rate": self.fixed_group_shrinkage_rate,
            "fixed_step_count": self.fixed_step_count,
            "model_config_sha256": self.model_config_sha256,
            "validation_balanced_proper_score": self.validation_balanced_proper_score,
            "validation_calibration_discrepancy": (
                self.validation_calibration_discrepancy
            ),
            "validation_evaluated_patient_patterns": (
                self.validation_evaluated_patient_patterns
            ),
            "training_config_sha256": self.training_config_sha256,
            "orchestration_config_sha256": self.orchestration_config_sha256,
            "outcome_values_received": False,
            "validation_used_for_step_or_hyperparameter_selection": False,
        }


@dataclass(frozen=True)
class Stage2FixedRefitResult:
    """Fixed post-selection model plus validation and calibration provenance."""

    model: nn.Module
    refit: Stage2FixedRefitArtifact
    calibration: CalibrationArtifact

    def checkpoint_parameters(self) -> dict[str, object]:
        refit_value = self.refit.to_dict()
        return {
            "schema_version": "soft-patient-atlas-stage2-checkpoint-parameters-v1",
            "stage2_selection": {
                "selected_beta": self.refit.fixed_beta,
                "selected_group_shrinkage_rate": (
                    self.refit.fixed_group_shrinkage_rate
                ),
                "selected_step": self.refit.fixed_step_count,
                "hyperparameter_endpoint_status": {
                    "beta": "frozen_selected_value",
                    "group_shrinkage_rate": "frozen_selected_value",
                    "overall": "fixed_post_selection_refit_no_selection",
                },
                "selected_state_sha256": self.refit.refit_state_sha256,
                "selected_model_config_sha256": self.refit.model_config_sha256,
                "selection_artifact_sha256": hash_json(refit_value),
                "training_config_sha256": self.refit.training_config_sha256,
                "orchestration_config_sha256": (
                    self.refit.orchestration_config_sha256
                ),
                "refit_mode": "fixed_post_selection_no_validation_selection",
            },
            "observable_calibration": self.calibration.to_dict(),
        }


def precision_temperature_state(
    state: GaussianState, temperature: float | torch.Tensor
) -> GaussianState:
    """Scale evidence natural and precision increments together by ``1/T``."""

    if isinstance(temperature, torch.Tensor):
        value = temperature.to(dtype=state.mean.dtype, device=state.mean.device)
        if value.ndim == 1:
            value = value[:, None]
        if value.shape not in (torch.Size([]), torch.Size([1]), state.mean.shape[:1] + (1,)):
            raise ValueError("temperature tensor must be scalar or have shape [batch,1]")
        if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
            raise ValueError("temperature must be finite and positive")
    else:
        if not math.isfinite(float(temperature)) or float(temperature) <= 0:
            raise ValueError("temperature must be finite and positive")
        value = torch.as_tensor(
            float(temperature), dtype=state.mean.dtype, device=state.mean.device
        )
    precision = state.precision
    if bool((precision < 1.0 - 1e-6).any()):
        raise ValueError("atlas posterior precision cannot be below prior precision")
    evidence_precision = (precision - 1.0).clamp(min=0.0)
    natural = state.mean * precision
    calibrated_precision = 1.0 + evidence_precision / value
    calibrated_natural = natural / value
    return GaussianState(
        mean=calibrated_natural / calibrated_precision,
        log_variance=-calibrated_precision.log(),
    )


def apply_decoder_scales(
    eye: EyePrediction,
    clinical: ClinicalPrediction,
    scales: DecoderScaleParameters,
) -> tuple[EyePrediction, ClinicalPrediction]:
    """Apply an immutable observable-space calibration overlay."""

    eye_prediction = EyePrediction(
        mean=eye.mean,
        log_variance=eye.log_variance
        + 2.0 * math.log(scales.eye_standard_deviation_multiplier),
    )
    clinical_prediction = ClinicalPrediction(
        continuous_location=clinical.continuous_location,
        continuous_log_scale=clinical.continuous_log_scale
        + math.log(scales.continuous_scale_multiplier),
        binary_logits=clinical.binary_logits / scales.binary_logit_temperature,
        blood_anchor_mean=clinical.blood_anchor_mean,
        blood_anchor_log_variance=clinical.blood_anchor_log_variance
        + 2.0 * math.log(scales.anchor_standard_deviation_multiplier),
    )
    return eye_prediction, clinical_prediction


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return hash_state_dict(dict(state))


def _model_state_hash(model: nn.Module) -> str:
    return hash_state_dict(model.state_dict())


def _model_config_payload(model: nn.Module) -> dict[str, object]:
    config = getattr(model, "config", None)
    if is_dataclass(config) and not isinstance(config, type):
        value = asdict(config)
    elif isinstance(config, Mapping):
        value = dict(config)
    else:
        raise TypeError("atlas model must expose a dataclass or mapping config")
    if not value:
        raise ValueError("atlas model config must not be empty")
    return value


def _configure_group_shrinkage_rate(model: nn.Module, rate: float) -> None:
    """Set the fixed MAP rate consistently on every config-bearing submodule."""

    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("group-shrinkage rate must be finite and positive")
    configured = 0
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is None or not hasattr(config, "group_shrinkage_rate"):
            continue
        if is_dataclass(config) and not isinstance(config, type):
            updated = replace(config, group_shrinkage_rate=float(rate))
        elif isinstance(config, Mapping):
            updated = dict(config)
            updated["group_shrinkage_rate"] = float(rate)
        else:
            raise TypeError("config with group_shrinkage_rate must be replaceable")
        setattr(module, "config", updated)
        configured += 1
    if configured == 0:
        raise TypeError("model exposes no configurable group_shrinkage_rate")
    observed = float(_model_config_payload(model).get("group_shrinkage_rate", -1.0))
    if not math.isclose(observed, rate, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("failed to set the model group-shrinkage rate")


def _slice_batch(batch: AtlasTrainingBatch, indices: torch.Tensor) -> AtlasTrainingBatch:
    def select(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value.index_select(0, indices)

    return AtlasTrainingBatch(
        eye_embeddings=select(batch.eye_embeddings),  # type: ignore[arg-type]
        eye_observed_mask=select(batch.eye_observed_mask),  # type: ignore[arg-type]
        blood_values=select(batch.blood_values),  # type: ignore[arg-type]
        blood_observed_mask=select(batch.blood_observed_mask),  # type: ignore[arg-type]
        blood_eligible_mask=select(batch.blood_eligible_mask),  # type: ignore[arg-type]
        demographics=select(batch.demographics),  # type: ignore[arg-type]
        demographic_mask=select(batch.demographic_mask),  # type: ignore[arg-type]
        eye_device_ids=select(batch.eye_device_ids),  # type: ignore[arg-type]
        eye_laterality_ids=select(batch.eye_laterality_ids),  # type: ignore[arg-type]
        eye_quality=select(batch.eye_quality),
        target_blood_anchor=select(batch.target_blood_anchor),
    )


def _forward_corrupted(model: nn.Module, batch: CorruptedAtlasBatch) -> PatientStateOutput:
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
        enable_interaction=False,
    )
    if not isinstance(output, PatientStateOutput):
        raise TypeError("Stage-2 orchestration requires PatientStateOutput")
    return output


@dataclass(frozen=True)
class _PredictionCase:
    requested_mode: str
    batch: CorruptedAtlasBatch
    output: PatientStateOutput
    pattern_codes: torch.Tensor


def _actual_pattern_codes(output: PatientStateOutput) -> torch.Tensor:
    eye = output.availability["eye_present"].reshape(-1).to(dtype=torch.bool)
    blood = output.availability["blood_clinical_present"].reshape(-1).to(dtype=torch.bool)
    codes = torch.full_like(eye, 3, dtype=torch.long)
    codes[eye & blood] = 0
    codes[eye & ~blood] = 1
    codes[~eye & blood] = 2
    return codes


def _complete_cases(model: nn.Module, source: AtlasTrainingBatch) -> tuple[_PredictionCase, ...]:
    eye_keep = torch.ones_like(source.eye_observed_mask)
    blood_keep = torch.ones_like(source.blood_observed_mask)
    cases: list[_PredictionCase] = []
    model.eval()
    with torch.no_grad():
        for mode in MODE_NAMES:
            batch = corrupt_observations(
                source,
                modes=(mode,) * source.eye_embeddings.shape[0],
                eye_partial_keep_mask=eye_keep,
                blood_partial_keep_mask=blood_keep,
            )
            output = _forward_corrupted(model, batch)
            cases.append(
                _PredictionCase(
                    requested_mode=mode,
                    batch=batch,
                    output=output,
                    pattern_codes=_actual_pattern_codes(output),
                )
            )
    return tuple(cases)


def _canonical_pattern_case(
    cases: Sequence[_PredictionCase], pattern: str
) -> tuple[_PredictionCase, torch.Tensor]:
    if pattern not in PRIMARY_PATTERNS:
        raise ValueError(f"unknown availability pattern: {pattern}")
    requested_mode = _PATTERN_TO_MODE[pattern]
    case = next(value for value in cases if value.requested_mode == requested_mode)
    code = PRIMARY_PATTERNS.index(pattern)
    return case, case.pattern_codes == code


def _temperature_vector(
    case: _PredictionCase, temperatures: Mapping[str, float]
) -> torch.Tensor:
    result = torch.ones(
        (case.pattern_codes.shape[0], 1),
        dtype=case.output.physiology.mean.dtype,
        device=case.output.physiology.mean.device,
    )
    for code, pattern in enumerate(PRIMARY_PATTERNS):
        result[case.pattern_codes == code] = float(temperatures.get(pattern, 1.0))
    return result


def _decode_case(
    model: nn.Module,
    case: _PredictionCase,
    temperatures: Mapping[str, float],
    scales: DecoderScaleParameters,
) -> tuple[EyePrediction, ClinicalPrediction]:
    state = precision_temperature_state(
        case.output.physiology, _temperature_vector(case, temperatures)
    )
    decoder = getattr(model, "decoder", None)
    if decoder is None or not callable(decoder):
        raise TypeError("model must expose a callable decoder")
    eye, clinical = decoder(
        state,
        case.output.demographic_context,
        eye_device_ids=case.batch.target_eye_device_ids,
        eye_laterality_ids=case.batch.target_eye_laterality_ids,
    )
    return apply_decoder_scales(eye, clinical, scales)


def _per_patient_proper_score(
    model: nn.Module,
    case: _PredictionCase,
    temperatures: Mapping[str, float],
    scales: DecoderScaleParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    eye, clinical = _decode_case(model, case, temperatures, scales)
    batch = case.batch
    eye_mask = batch.target_eye_mask
    safe_eye = torch.where(
        eye_mask[..., None], batch.target_eye_embeddings, torch.zeros_like(batch.target_eye_embeddings)
    )
    eye_nll = 0.5 * (
        math.log(2.0 * math.pi)
        + eye.log_variance
        + (safe_eye - eye.mean).square() / eye.log_variance.exp()
    ).mean(dim=-1)
    eye_count = eye_mask.sum(dim=1).clamp(min=1).to(dtype=eye_nll.dtype)
    eye_patient = torch.where(
        eye_mask, eye_nll, torch.zeros_like(eye_nll)
    ).sum(dim=1) / eye_count
    eye_present = eye_mask.any(dim=1)

    nc = int(getattr(model, "config").num_continuous)
    blood_mask = batch.target_blood_mask
    continuous_mask = blood_mask[:, :nc]
    binary_mask = blood_mask[:, nc:]
    continuous_target = torch.where(
        continuous_mask,
        batch.target_blood_values[:, :nc],
        torch.zeros_like(batch.target_blood_values[:, :nc]),
    )
    continuous_dist = torch.distributions.StudentT(
        df=torch.as_tensor(
            float(getattr(model, "config").student_df),
            dtype=continuous_target.dtype,
            device=continuous_target.device,
        ),
        loc=clinical.continuous_location,
        scale=clinical.continuous_log_scale.exp(),
    )
    continuous_nll = -continuous_dist.log_prob(continuous_target)
    binary_target = torch.where(
        binary_mask,
        batch.target_blood_values[:, nc:],
        torch.zeros_like(batch.target_blood_values[:, nc:]),
    )
    binary_nll = F.binary_cross_entropy_with_logits(
        clinical.binary_logits, binary_target, reduction="none"
    )
    blood_nll = torch.cat([continuous_nll, binary_nll], dim=1)
    blood_count = blood_mask.sum(dim=1).clamp(min=1).to(dtype=blood_nll.dtype)
    blood_patient = torch.where(
        blood_mask, blood_nll, torch.zeros_like(blood_nll)
    ).sum(dim=1) / blood_count
    blood_present = blood_mask.any(dim=1)

    view_count = eye_present.to(eye_patient.dtype) + blood_present.to(eye_patient.dtype)
    eligible = view_count > 0
    score = (eye_patient + blood_patient) / view_count.clamp(min=1.0)
    return score, eligible


def _balanced_proper_score(
    model: nn.Module,
    cases: Sequence[_PredictionCase],
    temperatures: Mapping[str, float] | None = None,
    scales: DecoderScaleParameters = DecoderScaleParameters(),
) -> tuple[float, int]:
    temperatures = {} if temperatures is None else temperatures
    mode_scores: list[torch.Tensor] = []
    count = 0
    with torch.no_grad():
        for case in cases:
            score, eligible = _per_patient_proper_score(
                model, case, temperatures, scales
            )
            if bool(eligible.any()):
                mode_scores.append(score[eligible].mean())
                count += int(eligible.sum().item())
    if not mode_scores:
        raise ValueError("no validation patient has an observable reconstruction target")
    result = torch.stack(mode_scores).mean()
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("nonfinite posterior-predictive proper score")
    return float(result.cpu()), count


def _calibration_discrepancy(
    model: nn.Module,
    cases: Sequence[_PredictionCase],
    temperatures: Mapping[str, float] | None = None,
    scales: DecoderScaleParameters = DecoderScaleParameters(),
) -> float:
    temperatures = {} if temperatures is None else temperatures
    eye_ratios: list[torch.Tensor] = []
    continuous_ratios: list[torch.Tensor] = []
    binary_probabilities: list[torch.Tensor] = []
    binary_targets: list[torch.Tensor] = []
    nc = int(getattr(model, "config").num_continuous)
    df = float(getattr(model, "config").student_df)
    with torch.no_grad():
        for case in cases:
            eye, clinical = _decode_case(model, case, temperatures, scales)
            eye_mask = case.batch.target_eye_mask[..., None].expand_as(eye.mean)
            if bool(eye_mask.any()):
                ratio = (
                    (case.batch.target_eye_embeddings - eye.mean).square()
                    / eye.log_variance.exp()
                )
                eye_ratios.append(ratio[eye_mask])
            continuous_mask = case.batch.target_blood_mask[:, :nc]
            if bool(continuous_mask.any()):
                variance = clinical.continuous_log_scale.mul(2).exp() * df / (df - 2.0)
                ratio = (
                    (case.batch.target_blood_values[:, :nc] - clinical.continuous_location).square()
                    / variance
                )
                continuous_ratios.append(ratio[continuous_mask])
            binary_mask = case.batch.target_blood_mask[:, nc:]
            if bool(binary_mask.any()):
                binary_probabilities.append(clinical.binary_logits.sigmoid()[binary_mask])
                binary_targets.append(case.batch.target_blood_values[:, nc:][binary_mask])

    components: list[float] = []
    for ratios in (eye_ratios, continuous_ratios):
        if ratios:
            value = torch.cat(ratios).mean()
            components.append(abs(float(value.cpu()) - 1.0))
    if binary_probabilities:
        probabilities = torch.cat(binary_probabilities)
        targets = torch.cat(binary_targets)
        bin_error = 0.0
        total = float(len(probabilities))
        for index in range(5):
            lower = index / 5.0
            upper = (index + 1) / 5.0
            mask = (probabilities >= lower) & (
                probabilities <= upper if index == 4 else probabilities < upper
            )
            if bool(mask.any()):
                error = abs(
                    float(targets[mask].mean().cpu())
                    - float(probabilities[mask].mean().cpu())
                )
                bin_error += float(mask.sum().item()) / total * error
        components.append(bin_error)
    if not components:
        raise ValueError("no observable family is available for calibration checking")
    result = sum(components) / len(components)
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite calibration discrepancy")
    return result


def _is_better(
    score: float,
    discrepancy: float,
    best_score: float,
    best_discrepancy: float,
    tolerance: float,
) -> bool:
    if score < best_score - tolerance:
        return True
    return abs(score - best_score) <= tolerance and discrepancy < best_discrepancy - tolerance


def _candidate_is_better(
    candidate: CandidateSummary,
    incumbent: CandidateSummary,
    tolerance: float,
) -> bool:
    if _is_better(
        candidate.best_balanced_proper_score,
        candidate.best_calibration_discrepancy,
        incumbent.best_balanced_proper_score,
        incumbent.best_calibration_discrepancy,
        tolerance,
    ):
        return True
    score_tied = abs(
        candidate.best_balanced_proper_score
        - incumbent.best_balanced_proper_score
    ) <= tolerance
    calibration_tied = abs(
        candidate.best_calibration_discrepancy
        - incumbent.best_calibration_discrepancy
    ) <= tolerance
    if not (score_tied and calibration_tied):
        return False
    # A fully tied validation result has no empirical preference.  Use the
    # prespecified central pair as a deterministic tertiary rule, reducing
    # needless endpoint expansion without consulting calibration identities.
    candidate_distance = abs(math.log(candidate.beta / 0.3)) + abs(
        math.log(candidate.group_shrinkage_rate / 1e-3)
    )
    incumbent_distance = abs(math.log(incumbent.beta / 0.3)) + abs(
        math.log(incumbent.group_shrinkage_rate / 1e-3)
    )
    return candidate_distance < incumbent_distance


def _endpoint_status(
    beta: float,
    group_shrinkage_rate: float,
    config: Stage2OrchestrationConfig,
) -> dict[str, str]:
    def position(value: float, grid: Sequence[float]) -> str:
        ordered = sorted(float(candidate) for candidate in grid)
        if value == ordered[0]:
            return "lower_endpoint"
        if value == ordered[-1]:
            return "upper_endpoint"
        return "interior"

    if config.grid_protocol == "frozen_selected_v1":
        return {
            "beta": "frozen_selected_value",
            "group_shrinkage_rate": "frozen_selected_value",
            "overall": "frozen_no_hyperparameter_selection",
        }
    beta_status = position(beta, config.beta_candidates)
    shrinkage_status = position(
        group_shrinkage_rate, config.group_shrinkage_rate_candidates
    )
    endpoint_selected = "endpoint" in beta_status or "endpoint" in shrinkage_status
    if endpoint_selected and config.grid_protocol == "lower_bound_expansion_v2":
        overall = "final_expansion_endpoint_design_finding"
    elif endpoint_selected:
        overall = "requires_grid_expansion_before_outer_test"
    elif config.grid_protocol.startswith("lower_bound_expansion_"):
        overall = "expanded_grid_interior"
    else:
        overall = "initial_grid_interior"
    return {
        "beta": beta_status,
        "group_shrinkage_rate": shrinkage_status,
        "overall": overall,
    }


def _fit_one_beta(
    initializer: nn.Module,
    fit_batch: AtlasTrainingBatch,
    validation_batch: AtlasTrainingBatch,
    training: Stage2TrainingConfig,
    orchestration: Stage2OrchestrationConfig,
    beta: float,
    group_shrinkage_rate: float,
) -> tuple[dict[str, torch.Tensor], CandidateSummary, tuple[ValidationRecord, ...]]:
    model = copy.deepcopy(initializer)
    _configure_group_shrinkage_rate(model, group_shrinkage_rate)
    candidate_training = replace(training, final_kl_weight=float(beta))
    optimizer = build_stage2_optimizer(model, candidate_training)
    device = fit_batch.eye_embeddings.device
    order_generator = torch.Generator(device=device).manual_seed(orchestration.seed)
    stochastic_generator = torch.Generator(device=device).manual_seed(
        orchestration.seed + 1
    )
    patient_count = fit_batch.eye_embeddings.shape[0]
    order = torch.randperm(patient_count, generator=order_generator, device=device)
    cursor = 0

    def next_training_batch() -> AtlasTrainingBatch:
        nonlocal order, cursor
        wanted = min(candidate_training.batch_size, patient_count)
        if cursor + wanted > patient_count:
            order = torch.randperm(
                patient_count, generator=order_generator, device=device
            )
            cursor = 0
        selected = order[cursor : cursor + wanted]
        cursor += wanted
        return _slice_batch(fit_batch, selected)

    validation_cases = _complete_cases(model, validation_batch)
    best_score, evaluated = _balanced_proper_score(model, validation_cases)
    best_discrepancy = _calibration_discrepancy(model, validation_cases)
    best_step = 0
    best_state = _clone_state(model)
    history: list[ValidationRecord] = [
        ValidationRecord(
            beta,
            group_shrinkage_rate,
            0,
            best_score,
            best_discrepancy,
            evaluated,
        )
    ]
    stale_checks = 0
    stopped_step = 0
    last_validation_step = 0

    for step in range(1, candidate_training.max_steps + 1):
        source = next_training_batch()
        corrupted = corrupt_observations(
            source, candidate_training, generator=stochastic_generator
        )
        train_step(
            model,
            optimizer,
            corrupted,
            candidate_training,
            step=step,
            generator=stochastic_generator,
        )
        stopped_step = step
        should_validate = (
            step % candidate_training.validation_interval == 0
            or step == candidate_training.max_steps
        )
        if not should_validate:
            continue
        last_validation_step = step
        validation_cases = _complete_cases(model, validation_batch)
        score, evaluated = _balanced_proper_score(model, validation_cases)
        discrepancy = _calibration_discrepancy(model, validation_cases)
        history.append(
            ValidationRecord(
                beta,
                group_shrinkage_rate,
                step,
                score,
                discrepancy,
                evaluated,
            )
        )
        if _is_better(
            score,
            discrepancy,
            best_score,
            best_discrepancy,
            orchestration.score_tie_tolerance,
        ):
            best_score = score
            best_discrepancy = discrepancy
            best_step = step
            best_state = _clone_state(model)
            stale_checks = 0
        else:
            stale_checks += 1
            if stale_checks >= candidate_training.early_stopping_patience:
                break

    if stopped_step and last_validation_step != stopped_step:
        raise RuntimeError("the final trained state was not validation checked")
    model.load_state_dict(best_state, strict=True)
    restored_hash = _model_state_hash(model)
    best_hash = _state_hash(best_state)
    if restored_hash != best_hash:
        raise RuntimeError("failed to restore the exact best validation checkpoint")
    summary = CandidateSummary(
        beta=float(beta),
        group_shrinkage_rate=float(group_shrinkage_rate),
        best_step=best_step,
        stopped_step=stopped_step,
        validation_checks=len(history),
        best_balanced_proper_score=best_score,
        best_calibration_discrepancy=best_discrepancy,
        best_state_sha256=best_hash,
    )
    return best_state, summary, tuple(history)


def _split_calibration_indices(
    patient_ids: Sequence[str], config: Stage2OrchestrationConfig
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    size = len(patient_ids)
    if size < 2:
        raise ValueError("calibration partition needs at least two patients")
    ordered = sorted(
        range(size),
        key=lambda index: (
            hashlib.sha256(
                f"{config.calibration_split_salt}\x1f{patient_ids[index]}".encode("utf-8")
            ).digest(),
            patient_ids[index],
        ),
    )
    parameter_count = round(size * config.calibration_parameter_fraction)
    parameter_count = min(size - 1, max(1, parameter_count))
    parameter = tuple(sorted(ordered[:parameter_count]))
    conformal = tuple(sorted(ordered[parameter_count:]))
    if set(parameter) & set(conformal) or set(parameter) | set(conformal) != set(range(size)):
        raise RuntimeError("internal calibration split is not a partition")
    return parameter, conformal


def _pattern_score(
    model: nn.Module,
    cases: Sequence[_PredictionCase],
    pattern: str,
    temperatures: Mapping[str, float],
    scales: DecoderScaleParameters,
) -> tuple[float, int]:
    case, pattern_mask = _canonical_pattern_case(cases, pattern)
    score, eligible = _per_patient_proper_score(model, case, temperatures, scales)
    include = eligible & pattern_mask
    if not bool(include.any()):
        return float("inf"), 0
    return float(score[include].mean().cpu()), int(include.sum().item())


def _choose_grid_value(
    values: Sequence[float], objective: Callable[[float], float]
) -> float:
    candidates: list[tuple[float, float, float]] = []
    for value in values:
        score = float(objective(float(value)))
        if not math.isfinite(score):
            continue
        candidates.append((score, abs(math.log(float(value))), float(value)))
    if not candidates:
        return 1.0
    return min(candidates)[2]


def _fit_precision_temperatures(
    model: nn.Module,
    cases: Sequence[_PredictionCase],
    config: Stage2OrchestrationConfig,
) -> tuple[dict[str, float], dict[str, int]]:
    temperatures: dict[str, float] = {}
    counts: dict[str, int] = {}
    for pattern in PRIMARY_PATTERNS:
        _, count = _pattern_score(
            model, cases, pattern, {pattern: 1.0}, DecoderScaleParameters()
        )
        counts[pattern] = count
        if count == 0:
            temperatures[pattern] = 1.0
            continue

        def objective(value: float) -> float:
            score, _ = _pattern_score(
                model,
                cases,
                pattern,
                {**temperatures, pattern: value},
                DecoderScaleParameters(),
            )
            return score

        temperatures[pattern] = _choose_grid_value(
            config.precision_temperature_grid, objective
        )
    return temperatures, counts


def _family_patient_scores(
    model: nn.Module,
    case: _PredictionCase,
    temperatures: Mapping[str, float],
    scales: DecoderScaleParameters,
    family: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    eye, clinical = _decode_case(model, case, temperatures, scales)
    batch = case.batch
    nc = int(getattr(model, "config").num_continuous)
    if family == "eye":
        mask = batch.target_eye_mask
        target = torch.where(
            mask[..., None], batch.target_eye_embeddings, torch.zeros_like(batch.target_eye_embeddings)
        )
        nll = 0.5 * (
            math.log(2.0 * math.pi)
            + eye.log_variance
            + (target - eye.mean).square() / eye.log_variance.exp()
        ).mean(dim=-1)
    elif family == "continuous":
        mask = batch.target_blood_mask[:, :nc]
        target = torch.where(
            mask,
            batch.target_blood_values[:, :nc],
            torch.zeros_like(batch.target_blood_values[:, :nc]),
        )
        distribution = torch.distributions.StudentT(
            df=torch.as_tensor(
                float(getattr(model, "config").student_df),
                dtype=target.dtype,
                device=target.device,
            ),
            loc=clinical.continuous_location,
            scale=clinical.continuous_log_scale.exp(),
        )
        nll = -distribution.log_prob(target)
    elif family == "binary":
        mask = batch.target_blood_mask[:, nc:]
        target = torch.where(
            mask,
            batch.target_blood_values[:, nc:],
            torch.zeros_like(batch.target_blood_values[:, nc:]),
        )
        nll = F.binary_cross_entropy_with_logits(
            clinical.binary_logits, target, reduction="none"
        )
    elif family == "anchor":
        if batch.target_blood_anchor is None:
            empty = torch.zeros(
                batch.batch_size,
                dtype=batch.target_blood_values.dtype,
                device=batch.target_blood_values.device,
            )
            return empty, torch.zeros_like(empty, dtype=torch.bool)
        anchor_mask = (
            batch.target_blood_mask
            & getattr(model, "blood_anchor_eligible_mask")[None, :]
        ).any(dim=1)
        mask = anchor_mask[:, None].expand_as(batch.target_blood_anchor)
        target = torch.where(
            mask, batch.target_blood_anchor, torch.zeros_like(batch.target_blood_anchor)
        )
        nll = 0.5 * (
            math.log(2.0 * math.pi)
            + clinical.blood_anchor_log_variance
            + (target - clinical.blood_anchor_mean).square()
            / clinical.blood_anchor_log_variance.exp()
        )
    else:
        raise ValueError(f"unknown decoder family: {family}")
    count = mask.sum(dim=tuple(range(1, mask.ndim))).clamp(min=1).to(dtype=nll.dtype)
    score = torch.where(mask, nll, torch.zeros_like(nll)).sum(
        dim=tuple(range(1, nll.ndim))
    ) / count
    return score, mask.reshape(mask.shape[0], -1).any(dim=1)


def _family_score(
    model: nn.Module,
    cases: Sequence[_PredictionCase],
    temperatures: Mapping[str, float],
    scales: DecoderScaleParameters,
    family: str,
) -> tuple[float, int]:
    values: list[torch.Tensor] = []
    count = 0
    for pattern in PRIMARY_PATTERNS:
        case, pattern_mask = _canonical_pattern_case(cases, pattern)
        score, eligible = _family_patient_scores(
            model, case, temperatures, scales, family
        )
        include = eligible & pattern_mask
        if bool(include.any()):
            values.append(score[include])
            count += int(include.sum().item())
    if not values:
        return float("inf"), 0
    return float(torch.cat(values).mean().cpu()), count


def _fit_decoder_scales(
    model: nn.Module,
    cases: Sequence[_PredictionCase],
    temperatures: Mapping[str, float],
    config: Stage2OrchestrationConfig,
) -> tuple[DecoderScaleParameters, dict[str, int]]:
    scales = DecoderScaleParameters()
    counts: dict[str, int] = {}
    fields = (
        ("eye", "eye_standard_deviation_multiplier", config.decoder_scale_grid),
        ("continuous", "continuous_scale_multiplier", config.decoder_scale_grid),
        ("binary", "binary_logit_temperature", config.binary_temperature_grid),
        ("anchor", "anchor_standard_deviation_multiplier", config.decoder_scale_grid),
    )
    for family, field, grid in fields:
        _, count = _family_score(model, cases, temperatures, scales, family)
        counts[family] = count
        if count == 0:
            continue

        def objective(value: float) -> float:
            candidate = replace(scales, **{field: value})
            score, _ = _family_score(
                model, cases, temperatures, candidate, family
            )
            return score

        chosen = _choose_grid_value(grid, objective)
        scales = replace(scales, **{field: chosen})
    return scales, counts


def _finite_sample_quantile(scores: torch.Tensor, alpha: float) -> float:
    if scores.ndim != 1 or not len(scores):
        raise ValueError("conformal scores must be a nonempty vector")
    ordered = scores.sort().values
    rank = math.ceil((len(ordered) + 1) * (1.0 - alpha))
    index = min(len(ordered), max(1, rank)) - 1
    return float(ordered[index].cpu())


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Stable continued fraction used by the regularized incomplete beta."""

    maximum_iterations = 300
    epsilon = 3e-14
    floor = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        even = 2 * iteration
        coefficient = iteration * (b - iteration) * x / (
            (qam + even) * (a + even)
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c
        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + even) * (qap + even))
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if a <= 0 or b <= 0 or not 0.0 <= x <= 1.0:
        raise ValueError("invalid regularized incomplete-beta arguments")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_cdf(value: float, degrees_of_freedom: float) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("Student-t degrees of freedom must be positive")
    if value == 0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_incomplete_beta(
        degrees_of_freedom / 2.0, 0.5, x
    )
    return 1.0 - tail if value > 0 else tail


def _student_t_quantile(probability: float, degrees_of_freedom: float) -> float:
    if not 0.5 < probability < 1.0:
        raise ValueError("only upper-half Student-t quantiles are supported")
    lower = 0.0
    upper = 1.0
    while _student_t_cdf(upper, degrees_of_freedom) < probability:
        upper *= 2.0
        if upper > 1e6:
            raise ArithmeticError("failed to bracket Student-t quantile")
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if _student_t_cdf(midpoint, degrees_of_freedom) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _conformal_scores_and_widths(
    model: nn.Module,
    case: _PredictionCase,
    pattern_mask: torch.Tensor,
    temperatures: Mapping[str, float],
    scales: DecoderScaleParameters,
    family: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    eye, clinical = _decode_case(model, case, temperatures, scales)
    batch = case.batch
    nc = int(getattr(model, "config").num_continuous)
    df = float(getattr(model, "config").student_df)
    if family == "eye":
        mask = batch.target_eye_mask[..., None].expand_as(eye.mean)
        residual = (batch.target_eye_embeddings - eye.mean).abs()
        standard_deviation = eye.log_variance.mul(0.5).exp()
    elif family == "continuous":
        mask = batch.target_blood_mask[:, :nc]
        residual = (
            batch.target_blood_values[:, :nc] - clinical.continuous_location
        ).abs()
        standard_deviation = (
            clinical.continuous_log_scale.exp() * math.sqrt(df / (df - 2.0))
        )
    elif family == "anchor":
        if batch.target_blood_anchor is None:
            empty = torch.empty(
                0,
                dtype=batch.target_blood_values.dtype,
                device=batch.target_blood_values.device,
            )
            return empty, empty
        present = (
            batch.target_blood_mask
            & getattr(model, "blood_anchor_eligible_mask")[None, :]
        ).any(dim=1)
        mask = present[:, None].expand_as(batch.target_blood_anchor)
        residual = (batch.target_blood_anchor - clinical.blood_anchor_mean).abs()
        standard_deviation = clinical.blood_anchor_log_variance.mul(0.5).exp()
    else:
        raise ValueError(f"conformal residuals are unsupported for family: {family}")
    patient_mask = mask.reshape(mask.shape[0], -1).any(dim=1) & pattern_mask
    if not bool(patient_mask.any()):
        empty = torch.empty(0, dtype=residual.dtype, device=residual.device)
        return empty, empty
    safe_score = torch.where(
        mask,
        residual / standard_deviation.clamp(min=1e-8),
        torch.full_like(residual, float("-inf")),
    )
    scores = safe_score.reshape(mask.shape[0], -1).max(dim=1).values[patient_mask]
    safe_width = torch.where(mask, standard_deviation, torch.zeros_like(standard_deviation))
    counts = mask.reshape(mask.shape[0], -1).sum(dim=1).clamp(min=1).to(residual.dtype)
    widths = safe_width.reshape(mask.shape[0], -1).sum(dim=1) / counts
    return scores, widths[patient_mask]


def _fit_conformal_sidecars(
    model: nn.Module,
    cases: Sequence[_PredictionCase],
    temperatures: Mapping[str, float],
    scales: DecoderScaleParameters,
    config: Stage2OrchestrationConfig,
) -> dict[str, dict[str, ConformalFamilySidecar]]:
    target_coverage = 1.0 - config.conformal_alpha
    gaussian_reference = NormalDist().inv_cdf(
        1.0 - config.conformal_alpha / 2.0
    )
    df = float(getattr(model, "config").student_df)
    # Continuous residuals above are standardized by predictive standard
    # deviation, not by the Student-t scale, hence the variance correction.
    student_reference = _student_t_quantile(
        1.0 - config.conformal_alpha / 2.0, df
    ) / math.sqrt(df / (df - 2.0))
    result: dict[str, dict[str, ConformalFamilySidecar]] = {}
    for pattern in PRIMARY_PATTERNS:
        case, pattern_mask = _canonical_pattern_case(cases, pattern)
        families: dict[str, ConformalFamilySidecar] = {}
        for family in ("eye", "continuous", "anchor"):
            raw_reference = (
                student_reference if family == "continuous" else gaussian_reference
            )
            scores, widths = _conformal_scores_and_widths(
                model,
                case,
                pattern_mask,
                temperatures,
                scales,
                family,
            )
            count = len(scores)
            if count < config.minimum_conformal_patients:
                families[family] = ConformalFamilySidecar(
                    status="insufficient_patients",
                    patient_count=count,
                    target_coverage=target_coverage,
                    residual_quantile=None,
                    raw_reference_quantile=raw_reference,
                    raw_calibration_patient_coverage=None,
                    conformal_calibration_patient_coverage=None,
                    raw_mean_half_width=None,
                    conformal_mean_half_width=None,
                    sharpness_inflation=None,
                )
                continue
            quantile = _finite_sample_quantile(scores, config.conformal_alpha)
            raw_coverage = float((scores <= raw_reference).to(torch.float64).mean().cpu())
            conformal_coverage = float((scores <= quantile).to(torch.float64).mean().cpu())
            mean_width = float(widths.mean().cpu())
            raw_half_width = raw_reference * mean_width
            conformal_half_width = quantile * mean_width
            families[family] = ConformalFamilySidecar(
                status="fitted",
                patient_count=count,
                target_coverage=target_coverage,
                residual_quantile=quantile,
                raw_reference_quantile=raw_reference,
                raw_calibration_patient_coverage=raw_coverage,
                conformal_calibration_patient_coverage=conformal_coverage,
                raw_mean_half_width=raw_half_width,
                conformal_mean_half_width=conformal_half_width,
                sharpness_inflation=(
                    conformal_half_width / raw_half_width
                    if raw_half_width > 0
                    else None
                ),
            )
        result[pattern] = families
    return result


def _fit_calibration(
    model: nn.Module,
    calibration_batch: AtlasTrainingBatch,
    calibration_ids: Sequence[str],
    config: Stage2OrchestrationConfig,
) -> CalibrationArtifact:
    parameter_indices, conformal_indices = _split_calibration_indices(
        calibration_ids, config
    )
    parameter_ids = tuple(calibration_ids[index] for index in parameter_indices)
    conformal_ids = tuple(calibration_ids[index] for index in conformal_indices)
    if set(parameter_ids) & set(conformal_ids):
        raise RuntimeError("calibration parameter and conformal identities overlap")
    model.requires_grad_(False)
    model.eval()
    frozen_hash = _model_state_hash(model)
    parameter_batch = _slice_batch(
        calibration_batch,
        torch.tensor(
            parameter_indices,
            dtype=torch.long,
            device=calibration_batch.eye_embeddings.device,
        ),
    )
    conformal_batch = _slice_batch(
        calibration_batch,
        torch.tensor(
            conformal_indices,
            dtype=torch.long,
            device=calibration_batch.eye_embeddings.device,
        ),
    )
    parameter_cases = _complete_cases(model, parameter_batch)
    temperatures, temperature_counts = _fit_precision_temperatures(
        model, parameter_cases, config
    )
    scales, scale_counts = _fit_decoder_scales(
        model, parameter_cases, temperatures, config
    )
    conformal_cases = _complete_cases(model, conformal_batch)
    conformal = _fit_conformal_sidecars(
        model, conformal_cases, temperatures, scales, config
    )
    if _model_state_hash(model) != frozen_hash:
        raise RuntimeError("calibration mutated frozen atlas parameters or buffers")
    return CalibrationArtifact(
        schema_version="soft-patient-atlas-calibration-v1",
        model_state_sha256=frozen_hash,
        model_config_sha256=hash_json(_model_config_payload(model)),
        calibration_id_sha256=_identifier_hash(calibration_ids),
        parameter_fit_id_sha256=_identifier_hash(parameter_ids),
        conformal_fit_id_sha256=_identifier_hash(conformal_ids),
        parameter_fit_count=len(parameter_ids),
        conformal_fit_count=len(conformal_ids),
        precision_temperatures=temperatures,
        precision_temperature_counts=temperature_counts,
        decoder_scales=scales,
        decoder_scale_counts=scale_counts,
        conformal=conformal,
        calibration_config_sha256=hash_json(asdict(config)),
    )


def fit_stage2_atlas(
    model_factory: Callable[[], nn.Module],
    cohort: AtlasCohort,
    training: Stage2TrainingConfig = Stage2TrainingConfig(),
    orchestration: Stage2OrchestrationConfig = Stage2OrchestrationConfig(),
    *,
    split: PatientIdSplit | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> Stage2FitResult:
    """Fit/select/freeze/calibrate one outer-fold additive atlas.

    The API intentionally has no outcome argument.  The returned history has
    aggregate scalar records only; raw identifiers remain confined to the
    input cohort and are represented in artifacts solely by counts and hashes.
    """

    if not callable(model_factory):
        raise TypeError("model_factory must be callable")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    if training.interaction_enabled:
        raise ValueError("Stage-2 orchestration is additive; interaction must be disabled")
    if split is None:
        split = deterministic_patient_split(
            cohort.patient_ids, cohort.site_ids, training
        )
    assert_disjoint_split(split, expected_patient_ids=cohort.patient_ids)
    if not split.fit or not split.validation or len(split.calibration) < 2:
        raise ValueError(
            "Stage-2 requires nonempty fit/validation and at least two calibration patients"
        )
    split_provenance = build_split_provenance(split)
    fit_batch = cohort.take(cohort.indices_for(split.fit))
    validation_batch = cohort.take(cohort.indices_for(split.validation))
    calibration_batch = cohort.take(cohort.indices_for(split.calibration))

    initializer = model_factory()
    if not isinstance(initializer, nn.Module):
        raise TypeError("model_factory must return torch.nn.Module")
    initializer_state = _clone_state(initializer)
    initializer_hash = _state_hash(initializer_state)
    summaries: list[CandidateSummary] = []
    history: list[ValidationRecord] = []
    candidate_states: dict[tuple[float, float], dict[str, torch.Tensor]] = {}
    if progress_callback is not None:
        progress_callback(
            {
                "event": "grid_started",
                "candidate_count": len(orchestration.beta_candidates)
                * len(orchestration.group_shrinkage_rate_candidates),
            }
        )
    for beta in orchestration.beta_candidates:
        for group_shrinkage_rate in orchestration.group_shrinkage_rate_candidates:
            state, summary, records = _fit_one_beta(
                initializer,
                fit_batch,
                validation_batch,
                training,
                orchestration,
                beta,
                group_shrinkage_rate,
            )
            candidate_states[(float(beta), float(group_shrinkage_rate))] = state
            summaries.append(summary)
            history.extend(records)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "candidate_completed",
                        "beta": summary.beta,
                        "group_shrinkage_rate": summary.group_shrinkage_rate,
                        "best_step": summary.best_step,
                        "stopped_step": summary.stopped_step,
                        "validation_checks": summary.validation_checks,
                        "best_balanced_proper_score": summary.best_balanced_proper_score,
                        "best_calibration_discrepancy": summary.best_calibration_discrepancy,
                        "best_state_sha256": summary.best_state_sha256,
                    }
                )

    selected = summaries[0]
    for candidate in summaries[1:]:
        if _candidate_is_better(
            candidate, selected, orchestration.score_tie_tolerance
        ):
            selected = candidate
    selected_model = copy.deepcopy(initializer)
    _configure_group_shrinkage_rate(
        selected_model, selected.group_shrinkage_rate
    )
    selected_state = candidate_states[
        (selected.beta, selected.group_shrinkage_rate)
    ]
    selected_model.load_state_dict(selected_state, strict=True)
    selected_hash = _model_state_hash(selected_model)
    if selected_hash != selected.best_state_sha256:
        raise RuntimeError("selected model is not the exact best candidate checkpoint")

    selection = Stage2SelectionArtifact(
        schema_version="soft-patient-atlas-stage2-selection-v1",
        split_provenance=split_provenance,
        initializer_state_sha256=initializer_hash,
        selected_state_sha256=selected_hash,
        selected_beta=selected.beta,
        selected_group_shrinkage_rate=selected.group_shrinkage_rate,
        selected_step=selected.best_step,
        hyperparameter_endpoint_status=_endpoint_status(
            selected.beta, selected.group_shrinkage_rate, orchestration
        ),
        selected_model_config_sha256=hash_json(
            _model_config_payload(selected_model)
        ),
        candidates=tuple(summaries),
        history=tuple(history),
        training_config_sha256=hash_json(asdict(training)),
        orchestration_config_sha256=hash_json(asdict(orchestration)),
    )
    if progress_callback is not None:
        progress_callback(
            {
                "event": "selection_completed",
                "selected_beta": selection.selected_beta,
                "selected_group_shrinkage_rate": selection.selected_group_shrinkage_rate,
                "selected_step": selection.selected_step,
                "selected_state_sha256": selection.selected_state_sha256,
                "hyperparameter_endpoint_status": dict(
                    selection.hyperparameter_endpoint_status
                ),
            }
        )
    calibration = _fit_calibration(
        selected_model,
        calibration_batch,
        split.calibration,
        orchestration,
    )
    if calibration.model_state_sha256 != selection.selected_state_sha256:
        raise RuntimeError("calibration model hash does not match selected checkpoint")
    if calibration.model_config_sha256 != selection.selected_model_config_sha256:
        raise RuntimeError("calibration model config does not match selected hyperparameters")
    if progress_callback is not None:
        progress_callback(
            {
                "event": "calibration_completed",
                "model_state_sha256": calibration.model_state_sha256,
                "calibration_config_sha256": calibration.calibration_config_sha256,
            }
        )
    return Stage2FitResult(selected_model, selection, calibration)


def fit_fixed_stage2_atlas(
    model_factory: Callable[[], nn.Module],
    cohort: AtlasCohort,
    training: Stage2TrainingConfig,
    orchestration: Stage2OrchestrationConfig,
    *,
    split: PatientIdSplit,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> Stage2FixedRefitResult:
    """Train one frozen recipe for exactly ``max_steps`` without model selection.

    This is the post-selection refit path. Validation is opened only after the
    final optimizer step and therefore cannot select a step, architecture, beta,
    shrinkage rate, or retry. Calibration remains disjoint and is opened last.
    """

    if not callable(model_factory):
        raise TypeError("model_factory must be callable")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    if training.interaction_enabled:
        raise ValueError("fixed Stage-2 refit is additive; interaction must be disabled")
    if orchestration.grid_protocol != "frozen_selected_v1":
        raise ValueError("fixed Stage-2 refit requires frozen_selected_v1")
    if (
        len(orchestration.beta_candidates) != 1
        or len(orchestration.group_shrinkage_rate_candidates) != 1
    ):
        raise ValueError("fixed Stage-2 refit requires exactly one frozen recipe")
    beta = float(orchestration.beta_candidates[0])
    shrinkage = float(orchestration.group_shrinkage_rate_candidates[0])
    if not math.isclose(training.final_kl_weight, beta, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("training KL endpoint differs from frozen beta")
    assert_disjoint_split(split, expected_patient_ids=cohort.patient_ids)
    if not split.fit or not split.validation or len(split.calibration) < 2:
        raise ValueError(
            "fixed Stage-2 refit requires nonempty fit/validation and calibration"
        )

    provenance = build_split_provenance(split)
    fit_batch = cohort.take(cohort.indices_for(split.fit))
    validation_batch = cohort.take(cohort.indices_for(split.validation))
    calibration_batch = cohort.take(cohort.indices_for(split.calibration))
    model = model_factory()
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return torch.nn.Module")
    initializer_hash = _state_hash(_clone_state(model))
    _configure_group_shrinkage_rate(model, shrinkage)
    optimizer = build_stage2_optimizer(model, training)
    device = fit_batch.eye_embeddings.device
    order_generator = torch.Generator(device=device).manual_seed(orchestration.seed)
    stochastic_generator = torch.Generator(device=device).manual_seed(
        orchestration.seed + 1
    )
    patient_count = int(fit_batch.eye_embeddings.shape[0])
    order = torch.randperm(patient_count, generator=order_generator, device=device)
    cursor = 0

    def next_training_batch() -> AtlasTrainingBatch:
        nonlocal order, cursor
        wanted = min(training.batch_size, patient_count)
        if cursor + wanted > patient_count:
            order = torch.randperm(
                patient_count, generator=order_generator, device=device
            )
            cursor = 0
        selected = order[cursor : cursor + wanted]
        cursor += wanted
        return _slice_batch(fit_batch, selected)

    if progress_callback is not None:
        progress_callback(
            {
                "event": "fixed_refit_started",
                "fit_count": len(split.fit),
                "validation_count": len(split.validation),
                "calibration_count": len(split.calibration),
                "fixed_beta": beta,
                "fixed_group_shrinkage_rate": shrinkage,
                "fixed_step_count": training.max_steps,
            }
        )
    for step in range(1, training.max_steps + 1):
        source = next_training_batch()
        corrupted = corrupt_observations(
            source, training, generator=stochastic_generator
        )
        metrics = train_step(
            model,
            optimizer,
            corrupted,
            training,
            step=step,
            generator=stochastic_generator,
        )
        if progress_callback is not None and (
            step % training.validation_interval == 0 or step == training.max_steps
        ):
            progress_callback(
                {
                    "event": "fixed_refit_training_checkpoint",
                    "step": step,
                    **metrics,
                }
            )

    model.requires_grad_(False)
    model.eval()
    validation_cases = _complete_cases(model, validation_batch)
    validation_score, evaluated = _balanced_proper_score(model, validation_cases)
    validation_discrepancy = _calibration_discrepancy(model, validation_cases)
    refit_hash = _model_state_hash(model)
    model_config_hash = hash_json(_model_config_payload(model))
    refit = Stage2FixedRefitArtifact(
        schema_version="soft-patient-atlas-stage2-fixed-refit-v1",
        split_provenance=provenance,
        initializer_state_sha256=initializer_hash,
        refit_state_sha256=refit_hash,
        fixed_beta=beta,
        fixed_group_shrinkage_rate=shrinkage,
        fixed_step_count=training.max_steps,
        model_config_sha256=model_config_hash,
        validation_balanced_proper_score=validation_score,
        validation_calibration_discrepancy=validation_discrepancy,
        validation_evaluated_patient_patterns=evaluated,
        training_config_sha256=hash_json(asdict(training)),
        orchestration_config_sha256=hash_json(asdict(orchestration)),
    )
    if progress_callback is not None:
        progress_callback(
            {
                "event": "fixed_refit_validation_completed",
                "validation_balanced_proper_score": validation_score,
                "validation_calibration_discrepancy": validation_discrepancy,
                "validation_evaluated_patient_patterns": evaluated,
                "refit_state_sha256": refit_hash,
            }
        )
    calibration = _fit_calibration(
        model, calibration_batch, split.calibration, orchestration
    )
    if calibration.model_state_sha256 != refit.refit_state_sha256:
        raise RuntimeError("fixed-refit calibration model hash differs")
    if calibration.model_config_sha256 != refit.model_config_sha256:
        raise RuntimeError("fixed-refit calibration model config differs")
    if progress_callback is not None:
        progress_callback(
            {
                "event": "fixed_refit_calibration_completed",
                "model_state_sha256": calibration.model_state_sha256,
                "calibration_config_sha256": calibration.calibration_config_sha256,
            }
        )
    return Stage2FixedRefitResult(model, refit, calibration)


__all__ = [
    "AtlasCohort",
    "CalibrationArtifact",
    "CandidateSummary",
    "ConformalFamilySidecar",
    "DecoderScaleParameters",
    "PRIMARY_PATTERNS",
    "SplitProvenance",
    "Stage2FitResult",
    "Stage2FixedRefitArtifact",
    "Stage2FixedRefitResult",
    "Stage2OrchestrationConfig",
    "Stage2SelectionArtifact",
    "ValidationRecord",
    "apply_decoder_scales",
    "build_split_provenance",
    "fit_stage2_atlas",
    "fit_fixed_stage2_atlas",
    "precision_temperature_state",
]
