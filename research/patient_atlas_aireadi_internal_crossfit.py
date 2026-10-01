"""Retrospective, leakage-safe AI-READI cross-fitting for Patient Atlas v3.

This module is the local patient-data boundary.  It uses only AI-READI's
reusable official train/validation pool, refits preprocessing and the complete
outcome-free Atlas inside each outer fold, and releases aggregate metrics only.
The retired official test is never loaded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from eval_soft_patient_atlas import (
    ARM_BLOOD,
    ARM_BOTH,
    ARM_CONCAT,
    ARM_DEMOGRAPHICS,
    ARM_EYE,
    ARM_EYE_TOWER,
    ARM_RAW_BLOOD,
    BASE_STRATUM,
    CONCAT_EYE_DIMENSIONS,
    EvaluationCohort,
    FeatureView,
    FoldCoordinates,
    FoldRepresentationProvenance,
    MissingnessStratum,
    NestedEvaluationConfig,
    RepresentationFitRequest,
    assert_aggregate_only_payload,
    evaluate_nested_patient_atlas,
    load_target_manifest,
)
from patient_atlas_development_validation import (
    FunctionalTargets,
    load_index_visit_functional_targets,
)
from patient_atlas_preprocessing import (
    SPLIT_PROVENANCE_SCHEMA_VERSION,
    FoldPreprocessor,
    FoldSplitProvenance,
    PreprocessingSchemaContract,
    fit_outer_fold_preprocessor,
    hash_identifier_set,
    hash_json,
    load_preprocessing_schema_contract,
    policy_mask_hash,
)
from patient_atlas_prospective_policy import apply_prospective_policy
from patient_atlas_real_data import ExploratoryRawCohort, load_exploratory_raw_cohort
from patient_atlas_stage2 import AtlasCohort, Stage2OrchestrationConfig, fit_stage2_atlas
from patient_atlas_v3_vector import encode_capacity_preserving_vector
from run_patient_atlas_v3_stage2 import _build_model
from soft_patient_atlas_v3 import CapacityPreservingSoftPatientAtlas
from train_soft_patient_atlas import AtlasTrainingBatch, PatientIdSplit, Stage2TrainingConfig


PROTOCOL_SCHEMA_VERSION = "patient-atlas-v3-aireadi-internal-crossfit-protocol-v1"
RUN_SCHEMA_VERSION = "patient-atlas-v3-aireadi-internal-crossfit-run-v1"

FEATURE_BLOOD_VALUES = "clinical_values"
FEATURE_BLOOD_OBSERVED = "clinical_observed_mask"
FEATURE_BLOOD_ELIGIBLE = "clinical_policy_eligible_mask"
FEATURE_EYE = "eye_embeddings"
FEATURE_EYE_OBSERVED = "eye_observed_mask"
FEATURE_EYE_DEVICE = "eye_device_ids"
FEATURE_EYE_LATERALITY = "eye_laterality_ids"
FEATURE_EYE_QUALITY = "eye_quality"

REQUIRED_FEATURES = (
    FEATURE_BLOOD_VALUES,
    FEATURE_BLOOD_OBSERVED,
    FEATURE_BLOOD_ELIGIBLE,
    FEATURE_EYE,
    FEATURE_EYE_OBSERVED,
    FEATURE_EYE_DEVICE,
    FEATURE_EYE_LATERALITY,
    FEATURE_EYE_QUALITY,
)


def _sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(encoded)


def _redacted_error(error: BaseException, patient_ids: Sequence[str]) -> str:
    message = str(error)
    for patient_id in patient_ids:
        message = message.replace(str(patient_id), "<redacted-patient-id>")
    return message[:2_000]


def validate_internal_crossfit_protocol(
    project_root: str | Path, protocol_path: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = Path(protocol_path).resolve()
    path.relative_to(root)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("internal cross-fit protocol must be a JSON object")
    if value.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("internal cross-fit protocol schema differs")
    if value.get("status") != "frozen_before_internal_crossfit_target_loading":
        raise ValueError("internal cross-fit protocol is not frozen before targets")
    cohort = value.get("cohort", {})
    if (
        cohort.get("included_recommended_splits") != ["train", "val"]
        or cohort.get("retired_official_test_loaded") is not False
        or cohort.get("retired_official_test_reuse_allowed") is not False
    ):
        raise ValueError("internal cross-fit protocol must exclude the retired test")
    representation = value.get("representation", {})
    evaluation = value.get("evaluation", {})
    execution = value.get("execution_policy", {})
    if (
        representation.get("architecture")
        != "capacity_preserving_structured_probabilistic_patient_atlas_v3"
        or representation.get("interaction_enabled") is not False
        or representation.get("external_blood_anchor_enabled") is not False
        or float(representation.get("beta", -1.0)) != 0.01
        or float(representation.get("group_shrinkage_rate", -1.0)) != 1e-6
    ):
        raise ValueError("internal cross-fit representation settings differ")
    expected_arms = [
        ARM_BOTH,
        ARM_EYE,
        ARM_BLOOD,
        ARM_CONCAT,
        ARM_DEMOGRAPHICS,
        ARM_EYE_TOWER,
        ARM_RAW_BLOOD,
    ]
    if (
        int(evaluation.get("outer_folds", 0)) != 5
        or int(evaluation.get("inner_readout_folds", 0)) != 5
        or evaluation.get("split_seeds") != [1701, 2718, 3141]
        or evaluation.get("enabled_arms") != expected_arms
        or evaluation.get("cross_fold_latent_coordinates_pooled") is not False
        or evaluation.get("same_fold_model_transforms_outer_train_and_outer_test")
        is not True
    ):
        raise ValueError("internal cross-fit evaluation settings differ")
    if (
        execution.get("patient_derived_processing") != "local_only"
        or execution.get("terminal_and_saved_result") != "aggregate_and_hash_only"
        or execution.get("outcome_driven_architecture_or_hyperparameter_retry_allowed")
        is not False
        or execution.get("result_path_must_be_new") is not True
    ):
        raise ValueError("internal cross-fit execution policy differs")
    return value


def _fold_training_seed(base_seed: int, fold_key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{fold_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _feature_policy(view: FeatureView) -> np.ndarray:
    if set(view.features) != set(REQUIRED_FEATURES):
        raise ValueError("AI-READI feature view does not match the frozen adapter schema")
    policy = np.asarray(view.features[FEATURE_BLOOD_ELIGIBLE])
    if policy.shape != (view.size, 59) or policy.dtype != np.bool_:
        raise TypeError("clinical policy feature must be boolean [patients,59]")
    if view.size > 1 and not np.array_equal(
        policy, np.broadcast_to(policy[:1], policy.shape)
    ):
        raise ValueError("clinical policy is artifact state, not patient state")
    return policy[0].copy()


def _view_to_training_batch(
    view: FeatureView,
    preprocessor: FoldPreprocessor,
    *,
    ordered_feature_names: Sequence[str],
) -> AtlasTrainingBatch:
    policy = _feature_policy(view)
    blood = preprocessor.transform_blood(
        np.asarray(view.features[FEATURE_BLOOD_VALUES]),
        np.asarray(view.features[FEATURE_BLOOD_OBSERVED]),
        ordered_feature_names=ordered_feature_names,
        policy_eligible_mask=policy,
    )
    age = preprocessor.transform_age(
        np.asarray(view.demographics),
        np.asarray(view.demographic_mask),
        require_observed=True,
    )
    eye = preprocessor.transform_eye(
        np.asarray(view.features[FEATURE_EYE]),
        np.asarray(view.features[FEATURE_EYE_OBSERVED]),
    )
    eligible = np.broadcast_to(policy[None, :], blood.values.shape).copy()
    return AtlasTrainingBatch(
        eye_embeddings=torch.from_numpy(eye.values),
        eye_observed_mask=torch.from_numpy(eye.observed_mask),
        blood_values=torch.from_numpy(blood.values),
        blood_observed_mask=torch.from_numpy(blood.observed_mask),
        blood_eligible_mask=torch.from_numpy(eligible),
        demographics=torch.from_numpy(age.values),
        demographic_mask=torch.from_numpy(age.observed_mask),
        eye_device_ids=torch.from_numpy(
            np.asarray(view.features[FEATURE_EYE_DEVICE]).copy()
        ),
        eye_laterality_ids=torch.from_numpy(
            np.asarray(view.features[FEATURE_EYE_LATERALITY]).copy()
        ),
        eye_quality=torch.from_numpy(
            np.asarray(view.features[FEATURE_EYE_QUALITY], dtype=np.float32).copy()
        ),
        target_blood_anchor=None,
    )


def _concatenate_batches(batches: Sequence[AtlasTrainingBatch]) -> AtlasTrainingBatch:
    if not batches:
        raise ValueError("at least one training batch is required")

    def combine(name: str) -> torch.Tensor:
        return torch.cat([getattr(batch, name) for batch in batches], dim=0)

    qualities = [batch.eye_quality for batch in batches]
    if any(value is None for value in qualities):
        if not all(value is None for value in qualities):
            raise ValueError("eye quality is inconsistently present across phases")
        quality = None
    else:
        quality = torch.cat([value for value in qualities if value is not None], dim=0)
    return AtlasTrainingBatch(
        eye_embeddings=combine("eye_embeddings"),
        eye_observed_mask=combine("eye_observed_mask"),
        blood_values=combine("blood_values"),
        blood_observed_mask=combine("blood_observed_mask"),
        blood_eligible_mask=combine("blood_eligible_mask"),
        demographics=combine("demographics"),
        demographic_mask=combine("demographic_mask"),
        eye_device_ids=combine("eye_device_ids"),
        eye_laterality_ids=combine("eye_laterality_ids"),
        eye_quality=quality,
        target_blood_anchor=None,
    )


def _patient_eye_summary(batch: AtlasTrainingBatch) -> tuple[np.ndarray, np.ndarray]:
    eye = batch.eye_embeddings.detach().cpu().numpy().astype(np.float64, copy=False)
    mask = batch.eye_observed_mask.detach().cpu().numpy().astype(bool, copy=False)
    counts = mask.sum(axis=1)
    present = counts > 0
    summary = np.zeros((eye.shape[0], eye.shape[2]), dtype=np.float64)
    if bool(present.any()):
        summary[present] = (eye[present] * mask[present, :, None]).sum(axis=1) / counts[
            present, None
        ]
    return summary, present


@dataclass(frozen=True)
class _EyePCA:
    center: np.ndarray
    components: Mapping[int, np.ndarray]

    def transform(
        self, summary: np.ndarray, present: np.ndarray, dimension: int
    ) -> np.ndarray:
        if dimension not in self.components:
            raise ValueError("unknown eye PCA candidate")
        centered = np.asarray(summary, dtype=np.float64) - self.center
        values = centered @ self.components[dimension]
        values = np.asarray(values, dtype=np.float64)
        values[~np.asarray(present, dtype=bool)] = 0.0
        return values


def _fit_eye_pca(batch: AtlasTrainingBatch) -> _EyePCA:
    summary, present = _patient_eye_summary(batch)
    fitted = summary[present]
    if fitted.shape[0] <= max(CONCAT_EYE_DIMENSIONS[:-1]):
        raise ValueError("outer-fold fit set has too few eye patients for PCA-64")
    center = fitted.mean(axis=0)
    _, _, right = np.linalg.svd(fitted - center, full_matrices=False)
    components: dict[int, np.ndarray] = {
        dimension: right[:dimension].T.copy()
        for dimension in CONCAT_EYE_DIMENSIONS
        if dimension < summary.shape[1]
    }
    components[summary.shape[1]] = np.eye(summary.shape[1], dtype=np.float64)
    return _EyePCA(center=center, components=MappingProxyType(components))


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _preprocessor_provenance(
    request: RepresentationFitRequest,
    *,
    training_seed: int,
    training: Stage2TrainingConfig,
    source_hashes: Mapping[str, str],
) -> FoldSplitProvenance:
    if (
        request.expected_outer_test_patient_id_hash is None
        or request.expected_outer_test_patient_count is None
    ):
        raise ValueError("real fold fitting requires hashed outer-test provenance")
    outer_train_ids = (
        request.fit.patient_ids
        + request.validation.patient_ids
        + request.calibration.patient_ids
    )
    role_hashes = {
        "outer_train": hash_identifier_set(
            outer_train_ids, label="outer_train_patient_ids"
        ),
        "representation_fit": hash_identifier_set(
            request.fit.patient_ids, label="representation_fit_patient_ids"
        ),
        "validation": hash_identifier_set(
            request.validation.patient_ids, label="validation_patient_ids"
        ),
        "calibration": hash_identifier_set(
            request.calibration.patient_ids, label="calibration_patient_ids"
        ),
        "outer_test": request.expected_outer_test_patient_id_hash,
    }
    role_counts = {
        "outer_train": len(outer_train_ids),
        "representation_fit": request.fit.size,
        "validation": request.validation.size,
        "calibration": request.calibration.size,
        "outer_test": request.expected_outer_test_patient_count,
    }
    return FoldSplitProvenance(
        schema_version=SPLIT_PROVENANCE_SCHEMA_VERSION,
        outer_fold_id=request.fold_key,
        outer_train_patient_id_hash=role_hashes["outer_train"],
        representation_fit_patient_id_hash=role_hashes["representation_fit"],
        validation_patient_id_hash=role_hashes["validation"],
        calibration_patient_id_hash=role_hashes["calibration"],
        outer_test_patient_id_hash=role_hashes["outer_test"],
        split_manifest_hash=hash_json(
            {
                "outer_fold_id": request.fold_key,
                "role_hashes": role_hashes,
                "role_counts": role_counts,
            }
        ),
        role_counts=role_counts,
        split_seed=training_seed,
        training_seed=training_seed,
        training_config_hash=hash_json(asdict(training)),
        source_hashes=dict(sorted(source_hashes.items())),
    )


class FittedAIReadIFoldRepresentation:
    """One immutable v3 model/basis used for both train and test transforms."""

    def __init__(
        self,
        *,
        request: RepresentationFitRequest,
        model: CapacityPreservingSoftPatientAtlas,
        preprocessor: FoldPreprocessor,
        eye_pca: _EyePCA,
        precision_temperatures: Mapping[str, float],
        ordered_feature_names: Sequence[str],
        model_token: str,
    ) -> None:
        self.provenance: FoldRepresentationProvenance = request.expected_provenance(
            model_token
        )
        self.model = model.eval()
        self.preprocessor = preprocessor
        self.eye_pca = eye_pca
        self.precision_temperatures = dict(precision_temperatures)
        self.ordered_feature_names = tuple(ordered_feature_names)
        pca_hash = _array_digest(
            eye_pca.center,
            *(eye_pca.components[value] for value in sorted(eye_pca.components)),
        )
        self.atlas_basis_token = hash_json(
            {
                "model_state": model_token,
                "preprocessor": preprocessor.bundle_sha256,
                "vector": "v3-modality-evidence-128",
            }
        )
        self.pca_basis_token = pca_hash
        self._batch_cache: dict[str, AtlasTrainingBatch] = {}
        self._baseline_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._atlas_cache: dict[tuple[str, str], np.ndarray] = {}

    @staticmethod
    def _view_token(view: FeatureView) -> str:
        return hash_identifier_set(view.patient_ids, label="transform_patient_ids")

    def _batch(self, view: FeatureView) -> tuple[str, AtlasTrainingBatch]:
        token = self._view_token(view)
        if token not in self._batch_cache:
            self._batch_cache[token] = _view_to_training_batch(
                view,
                self.preprocessor,
                ordered_feature_names=self.ordered_feature_names,
            )
        return token, self._batch_cache[token]

    def _baselines(
        self, view: FeatureView
    ) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        token, batch = self._batch(view)
        if token not in self._baseline_cache:
            eye_summary, eye_present = _patient_eye_summary(batch)
            clinical_visible = (
                batch.blood_observed_mask & batch.blood_eligible_mask
            ).detach().cpu().numpy()
            raw_clinical = np.concatenate(
                [
                    batch.blood_values.detach().cpu().numpy().astype(np.float64),
                    clinical_visible.astype(np.float64),
                ],
                axis=1,
            )
            self._baseline_cache[token] = (
                eye_summary,
                eye_present,
                raw_clinical,
            )
        eye_summary, eye_present, raw_clinical = self._baseline_cache[token]
        return token, eye_summary, eye_present, raw_clinical

    def _atlas(self, view: FeatureView, arm: str, *, batch_size: int = 128) -> np.ndarray:
        token, batch = self._batch(view)
        cache_key = (token, arm)
        if cache_key in self._atlas_cache:
            return self._atlas_cache[cache_key]
        if arm not in (ARM_BOTH, ARM_EYE, ARM_BLOOD):
            raise ValueError("unknown Atlas availability arm")
        size = int(batch.eye_embeddings.shape[0])
        chunks: list[np.ndarray] = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, size, batch_size):
                stop = min(start + batch_size, size)
                eye_observed = batch.eye_observed_mask[start:stop]
                clinical_observed = (
                    batch.blood_observed_mask[start:stop]
                    & batch.blood_eligible_mask[start:stop]
                )
                output = encode_capacity_preserving_vector(
                    self.model,
                    eye_embeddings=batch.eye_embeddings[start:stop],
                    eye_visible_mask=(
                        eye_observed
                        if arm in (ARM_BOTH, ARM_EYE)
                        else torch.zeros_like(eye_observed)
                    ),
                    blood_values=batch.blood_values[start:stop],
                    blood_visible_mask=(
                        clinical_observed
                        if arm in (ARM_BOTH, ARM_BLOOD)
                        else torch.zeros_like(clinical_observed)
                    ),
                    blood_eligible_mask=batch.blood_eligible_mask[start:stop],
                    demographics=batch.demographics[start:stop],
                    demographic_mask=batch.demographic_mask[start:stop],
                    precision_temperatures=self.precision_temperatures,
                    eye_device_ids=batch.eye_device_ids[start:stop],
                    eye_laterality_ids=batch.eye_laterality_ids[start:stop],
                    eye_quality=(
                        None
                        if batch.eye_quality is None
                        else batch.eye_quality[start:stop]
                    ),
                )
                chunks.append(
                    output.mean[:, :128].detach().cpu().numpy().astype(np.float64)
                )
        values = np.concatenate(chunks, axis=0)
        if values.shape != (size, 128) or not np.isfinite(values).all():
            raise ValueError("fold Atlas emitted malformed outcome coordinates")
        self._atlas_cache[cache_key] = values
        return values

    def transform(
        self,
        view: FeatureView,
        *,
        arm: str,
        stratum: str,
        missingness: MissingnessStratum | None = None,
        concat_eye_dimension: int | None = None,
    ) -> FoldCoordinates:
        if stratum != BASE_STRATUM or missingness is not None:
            raise ValueError("v1 internal cross-fit has no missingness stratum enabled")
        if arm in (ARM_BOTH, ARM_EYE, ARM_BLOOD):
            values = self._atlas(view, arm)
            groups = (
                ("eye_atlas",) * 32
                + ("clinical_atlas",) * 32
                + ("eye_atlas",) * 32
                + ("clinical_atlas",) * 32
            )
            basis = self.atlas_basis_token
        elif arm == ARM_EYE_TOWER:
            _, eye_summary, eye_present, _ = self._baselines(view)
            values = self.eye_pca.transform(eye_summary, eye_present, 384)
            groups = ("eye_tower",) * values.shape[1]
            basis = f"eye-tower:{self.pca_basis_token}"
        elif arm == ARM_RAW_BLOOD:
            _, _, _, values = self._baselines(view)
            groups = ("raw_clinical",) * values.shape[1]
            basis = f"raw-clinical:{self.preprocessor.bundle_sha256}"
        elif arm == ARM_CONCAT:
            if concat_eye_dimension not in CONCAT_EYE_DIMENSIONS:
                raise ValueError("unknown concat eye dimension")
            _, eye_summary, eye_present, clinical = self._baselines(view)
            eye = self.eye_pca.transform(
                eye_summary, eye_present, int(concat_eye_dimension)
            )
            values = np.concatenate([eye, clinical], axis=1)
            groups = ("eye",) * eye.shape[1] + ("blood",) * clinical.shape[1]
            basis = f"concat:{self.pca_basis_token}:{concat_eye_dimension}"
        else:
            raise ValueError(f"unsupported internal cross-fit arm {arm!r}")
        return FoldCoordinates(
            patient_ids=view.patient_ids,
            values=np.asarray(values, dtype=np.float64),
            penalty_groups=groups,
            fold_key=self.provenance.fold_key,
            model_token=self.provenance.model_token,
            basis_token=basis,
            arm=arm,
            stratum=stratum,
            concat_eye_dimension=concat_eye_dimension,
        )


class AIReadIV3FoldRepresentationFactory:
    """Fit one complete outcome-free v3 pipeline for each evaluator outer fold."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        ordered_feature_names: Sequence[str],
        source_policy_sha256: str,
        source_hashes: Mapping[str, str],
        base_training_seed: int = 20260826,
        training: Stage2TrainingConfig = Stage2TrainingConfig(),
        orchestration: Stage2OrchestrationConfig = Stage2OrchestrationConfig(
            beta_candidates=(0.01,),
            group_shrinkage_rate_candidates=(1e-6,),
            grid_protocol="frozen_selected_v1",
        ),
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.ordered_feature_names = tuple(ordered_feature_names)
        self.source_policy_sha256 = str(source_policy_sha256)
        self.source_hashes = {
            **dict(source_hashes),
            "source_policy": self.source_policy_sha256,
        }
        self.base_training_seed = int(base_training_seed)
        self.training = training
        self.orchestration = orchestration
        self.progress_callback = progress_callback
        self.schemas: PreprocessingSchemaContract = load_preprocessing_schema_contract(
            self.root / "PATIENT_ATLAS_FEATURE_REGISTRY.json",
            self.root / "PATIENT_ATLAS_CONTEXT_SCHEMA.json",
            self.root / "PATIENT_ATLAS_EYE_REGISTRY.json",
        )
        if self.schemas.ordered_feature_names != self.ordered_feature_names:
            raise ValueError("factory feature order differs from authenticated registry")
        self.fold_summaries: list[dict[str, Any]] = []

    def _progress(self, fold_key: str, event: Mapping[str, object]) -> None:
        if self.progress_callback is not None:
            self.progress_callback({"fold_key": fold_key, **dict(event)})

    def fit(self, request: RepresentationFitRequest) -> FittedAIReadIFoldRepresentation:
        started = time.perf_counter()
        policy = _feature_policy(request.fit)
        for view in (request.validation, request.calibration):
            if not np.array_equal(_feature_policy(view), policy):
                raise ValueError("clinical policy differs across outer-training phases")
        training_seed = _fold_training_seed(self.base_training_seed, request.fold_key)
        provenance = _preprocessor_provenance(
            request,
            training_seed=training_seed,
            training=self.training,
            source_hashes=self.source_hashes,
        )
        schema_hashes = {
            "feature": self.schemas.feature_schema_hash,
            "context": self.schemas.context_schema_hash,
            "eye": self.schemas.eye_schema_hash,
        }
        preprocessor = fit_outer_fold_preprocessor(
            patient_ids=request.fit.patient_ids,
            blood_values=np.asarray(request.fit.features[FEATURE_BLOOD_VALUES]),
            blood_observed_mask=np.asarray(
                request.fit.features[FEATURE_BLOOD_OBSERVED]
            ),
            ordered_feature_names=self.ordered_feature_names,
            policy_eligible_mask=policy,
            expected_policy_mask_hash=policy_mask_hash(
                self.schemas.ordered_features_hash, policy
            ),
            ages=np.asarray(request.fit.demographics),
            age_observed_mask=np.asarray(request.fit.demographic_mask),
            eye_embeddings=np.asarray(request.fit.features[FEATURE_EYE]),
            eye_observed_mask=np.asarray(request.fit.features[FEATURE_EYE_OBSERVED]),
            schemas=self.schemas,
            provenance=provenance,
            expected_schema_hashes=schema_hashes,
            fit_scope="exploratory",
            source_policy_sha256=self.source_policy_sha256,
        )
        phase_views = (request.fit, request.validation, request.calibration)
        phase_batches = tuple(
            _view_to_training_batch(
                view,
                preprocessor,
                ordered_feature_names=self.ordered_feature_names,
            )
            for view in phase_views
        )
        eye_pca = _fit_eye_pca(phase_batches[0])
        observations = _concatenate_batches(phase_batches)
        patient_ids = tuple(
            patient_id for view in phase_views for patient_id in view.patient_ids
        )
        site_ids = tuple(site_id for view in phase_views for site_id in view.site_ids)
        split = PatientIdSplit(
            fit=request.fit.patient_ids,
            validation=request.validation.patient_ids,
            calibration=request.calibration.patient_ids,
        )
        atlas_cohort = AtlasCohort(patient_ids, site_ids, observations)

        def model_factory() -> CapacityPreservingSoftPatientAtlas:
            torch.manual_seed(training_seed)
            return _build_model(root=self.root, group_shrinkage_rate=1e-6)

        orchestration = Stage2OrchestrationConfig(
            **{**asdict(self.orchestration), "seed": training_seed}
        )
        self._progress(
            request.fold_key,
            {
                "event": "outer_fold_fit_started",
                "outer_train_count": len(patient_ids),
                "fit_count": request.fit.size,
                "validation_count": request.validation.size,
                "calibration_count": request.calibration.size,
            },
        )
        result = fit_stage2_atlas(
            model_factory,
            atlas_cohort,
            self.training,
            orchestration,
            split=split,
            progress_callback=lambda event: self._progress(request.fold_key, event),
        )
        if not isinstance(result.model, CapacityPreservingSoftPatientAtlas):
            raise TypeError("fold fitting returned the wrong Patient Atlas model")
        summary = {
            "fold_key": request.fold_key,
            "outer_train_count": len(patient_ids),
            "outer_test_count": request.expected_outer_test_patient_count,
            "fit_count": request.fit.size,
            "validation_count": request.validation.size,
            "calibration_count": request.calibration.size,
            "selected_step": result.selection.selected_step,
            "best_balanced_proper_score": result.selection.candidates[
                0
            ].best_balanced_proper_score,
            "selected_state_sha256": result.selection.selected_state_sha256,
            "preprocessor_bundle_sha256": preprocessor.bundle_sha256,
            "precision_temperatures": dict(
                sorted(result.calibration.precision_temperatures.items())
            ),
            "runtime_seconds": time.perf_counter() - started,
        }
        self.fold_summaries.append(summary)
        self._progress(request.fold_key, {"event": "outer_fold_fit_completed", **summary})
        return FittedAIReadIFoldRepresentation(
            request=request,
            model=result.model,
            preprocessor=preprocessor,
            eye_pca=eye_pca,
            precision_temperatures=result.calibration.precision_temperatures,
            ordered_feature_names=self.ordered_feature_names,
            model_token=result.selection.selected_state_sha256,
        )


def _evaluation_features(cohort: ExploratoryRawCohort) -> Mapping[str, np.ndarray]:
    policy = np.broadcast_to(
        cohort.blood_eligible_mask[None, :], cohort.blood_values.shape
    ).copy()
    return MappingProxyType(
        {
            FEATURE_BLOOD_VALUES: cohort.blood_values,
            FEATURE_BLOOD_OBSERVED: cohort.blood_observed_mask,
            FEATURE_BLOOD_ELIGIBLE: policy,
            FEATURE_EYE: cohort.eye_embeddings,
            FEATURE_EYE_OBSERVED: cohort.eye_observed_mask,
            FEATURE_EYE_DEVICE: cohort.eye_device_ids,
            FEATURE_EYE_LATERALITY: cohort.eye_laterality_ids,
            FEATURE_EYE_QUALITY: cohort.eye_quality,
        }
    )


def build_internal_evaluation_cohort(
    cohort: ExploratoryRawCohort, targets: FunctionalTargets
) -> EvaluationCohort:
    if tuple(targets.target_ids) == () or targets.values.shape[0] != len(
        cohort.patient_ids
    ):
        raise ValueError("functional targets do not align to the AI-READI cohort")
    eye_present = cohort.eye_observed_mask.any(axis=1)
    clinical_present = (
        cohort.blood_observed_mask & cohort.blood_eligible_mask[None, :]
    ).any(axis=1)
    return EvaluationCohort(
        patient_ids=cohort.patient_ids,
        site_ids=cohort.site_ids,
        features=_evaluation_features(cohort),
        demographics=cohort.ages[:, None].astype(np.float64, copy=False),
        demographic_mask=cohort.age_observed_mask[:, None].copy(),
        targets=np.asarray(targets.values, dtype=np.float64),
        target_ids=tuple(targets.target_ids),
        target_mask=np.asarray(targets.observed_mask, dtype=bool),
        primary_patient_mask=np.asarray(eye_present & clinical_present, dtype=bool),
    )


def _config_from_protocol(
    protocol: Mapping[str, Any], *, noninferiority_margin: float | None
) -> NestedEvaluationConfig:
    evaluation = protocol["evaluation"]
    return NestedEvaluationConfig(
        outer_folds=int(evaluation["outer_folds"]),
        inner_folds=int(evaluation["inner_readout_folds"]),
        split_seeds=tuple(int(value) for value in evaluation["split_seeds"]),
        ridge_grid=tuple(float(value) for value in evaluation["ridge_grid"]),
        bootstrap_samples=int(evaluation["bootstrap_samples"]),
        bootstrap_seed=int(evaluation["bootstrap_seed"]),
        confidence_level=float(evaluation["confidence_level"]),
        noninferiority_margin=noninferiority_margin,
        noninferiority_margin_provenance=None,
        missingness_strata=(),
        enabled_arms=tuple(str(value) for value in evaluation["enabled_arms"]),
    )


def _training_from_protocol(protocol: Mapping[str, Any]) -> Stage2TrainingConfig:
    value = protocol["representation"]["training"]
    phases = protocol["representation"]["representation_phases_within_outer_train"]
    return Stage2TrainingConfig(
        fit_fraction=float(phases["fit"]),
        validation_fraction=float(phases["validation"]),
        calibration_fraction=float(phases["calibration"]),
        batch_size=int(value["batch_size"]),
        max_steps=int(value["maximum_steps"]),
        learning_rate=float(value["learning_rate"]),
        weight_decay=float(value["weight_decay"]),
        validation_interval=int(value["validation_interval"]),
        early_stopping_patience=int(value["early_stopping_patience"]),
    )


def run_aireadi_internal_crossfit(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    clinical_project_root: str | Path,
    protocol_path: str | Path,
    target_manifest_path: str | Path,
    functional_source_policy_path: str | Path,
    output_path: str | Path,
    failure_path: str | Path,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    target_manifest_path = Path(target_manifest_path).resolve()
    functional_source_policy_path = Path(functional_source_policy_path).resolve()
    output_path = Path(output_path).resolve()
    failure_path = Path(failure_path).resolve()
    if output_path.exists() or failure_path.exists():
        raise FileExistsError("internal cross-fit output and failure paths must be new")
    protocol = validate_internal_crossfit_protocol(root, protocol_path)
    manifest = load_target_manifest(target_manifest_path)
    if manifest.noninferiority_margin is not None:
        raise ValueError("internal v1 requires the frozen null concat margin")

    started = time.perf_counter()
    patient_ids: tuple[str, ...] = ()
    targets_loaded = False
    try:
        cohort = load_exploratory_raw_cohort(
            project_root=root,
            dataset_root=dataset_root,
            clinical_project_root=clinical_project_root,
        )
        cohort = apply_prospective_policy(cohort, project_root=root)
        if set(cohort.split_labels) - {"train", "val"}:
            raise PermissionError("retired official test entered the internal cohort")
        patient_ids = cohort.patient_ids
        targets = load_index_visit_functional_targets(
            dataset_root=dataset_root,
            patient_ids=patient_ids,
            manifest_path=target_manifest_path,
            source_policy_path=functional_source_policy_path,
        )
        targets_loaded = True
        evaluation_cohort = build_internal_evaluation_cohort(cohort, targets)
        training = _training_from_protocol(protocol)

        def safe_progress(event: Mapping[str, Any]) -> None:
            if progress_callback is not None:
                progress_callback(dict(event))

        factory = AIReadIV3FoldRepresentationFactory(
            project_root=root,
            ordered_feature_names=cohort.feature_names,
            source_policy_sha256=cohort.source_policy_sha256,
            source_hashes=cohort.source_hashes,
            base_training_seed=int(
                protocol["representation"]["training"]["base_training_seed"]
            ),
            training=training,
            progress_callback=safe_progress,
        )
        report = evaluate_nested_patient_atlas(
            evaluation_cohort,
            manifest,
            factory,
            _config_from_protocol(
                protocol, noninferiority_margin=manifest.noninferiority_margin
            ),
        ).to_dict()
        report.update(
            {
                "run_schema_version": RUN_SCHEMA_VERSION,
                "scope": "aireadi_recommended_train_validation_retrospective_internal_crossfit",
                "claim_role": protocol["claim_role"],
                "runtime_seconds": time.perf_counter() - started,
                "protocol": {
                    "file": protocol_path.name,
                    "file_sha256": _sha256(protocol_path),
                    "status": protocol["status"],
                },
                "data_scope": {
                    "recommended_train_validation_only": True,
                    "recommended_split_counts": {
                        str(key): int(value)
                        for key, value in sorted(
                            {
                                label: cohort.split_labels.count(label)
                                for label in set(cohort.split_labels)
                            }.items()
                        )
                    },
                    "retired_official_test_inputs_loaded": False,
                    "retired_official_test_targets_loaded": False,
                    "functional_targets_loaded": True,
                },
                "fold_representation_fits": factory.fold_summaries,
                "implementation_hashes": {
                    name: _sha256(root / name)
                    for name in (
                        "patient_atlas_aireadi_internal_crossfit.py",
                        "eval_soft_patient_atlas.py",
                        "soft_patient_atlas_v3.py",
                        "patient_atlas_stage2.py",
                        "patient_atlas_preprocessing.py",
                        "patient_atlas_real_data.py",
                        "PATIENT_ATLAS_TARGET_MANIFEST.json",
                        "PATIENT_ATLAS_FUNCTIONAL_SOURCE_POLICY.json",
                    )
                },
                "claim_limitations": {
                    "retrospective_internal_crossfit": True,
                    "researcher_adaptivity_to_aireadi_history_remains": True,
                    "external_validation_completed": False,
                    "clinical_benefit_claim_allowed": False,
                    "retired_official_test_reused": False,
                    "frozen_blood_tower_arm_evaluated": False,
                    "missingness_stress_test_evaluated_in_this_run": False,
                },
            }
        )
        assert_aggregate_only_payload(report, forbidden_patient_ids=patient_ids)
        _write_exclusive_json(output_path, report)
        return report
    except BaseException as error:
        failure = {
            "schema_version": "patient-atlas-v3-aireadi-internal-crossfit-failure-v1",
            "terminal": True,
            "error_type": type(error).__name__,
            "error_message": _redacted_error(error, patient_ids),
            "runtime_seconds": time.perf_counter() - started,
            "targets_loaded": targets_loaded,
            "retired_official_test_inputs_loaded": False,
            "retired_official_test_targets_loaded": False,
            "infrastructure_retry_allowed": True,
            "outcome_driven_retuning_allowed": False,
            "patient_rows_emitted": False,
            "patient_identifiers_emitted": False,
            "patient_vectors_emitted": False,
            "per_patient_predictions_emitted": False,
        }
        _write_exclusive_json(failure_path, failure)
        raise


__all__ = [
    "AIReadIV3FoldRepresentationFactory",
    "FittedAIReadIFoldRepresentation",
    "PROTOCOL_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "build_internal_evaluation_cohort",
    "run_aireadi_internal_crossfit",
    "validate_internal_crossfit_protocol",
]
