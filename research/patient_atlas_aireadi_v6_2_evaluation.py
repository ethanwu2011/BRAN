"""Fold-safe AI-READI representation fitting for Patient Atlas V6.2."""

from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

import numpy as np
import torch

from eval_soft_patient_atlas import RepresentationFitRequest
from patient_atlas_aireadi_internal_crossfit import (
    _concatenate_batches,
    _feature_policy,
    _fit_eye_pca,
    _fold_training_seed,
    _preprocessor_provenance,
    _view_to_training_batch,
)
from patient_atlas_aireadi_v5_evaluation import (
    AIReadIV5FoldRepresentationFactory,
    FittedAIReadIV5FoldRepresentation,
)
from patient_atlas_preprocessing import (
    fit_outer_fold_preprocessor,
    hash_json,
    policy_mask_hash,
)
from patient_atlas_stage2 import AtlasCohort, Stage2OrchestrationConfig, fit_stage2_atlas
from run_patient_atlas_v5_stage2 import capacity_health
from run_patient_atlas_v6_2_stage2 import (
    _build_model,
    heldout_recovery_health,
)
from soft_patient_atlas_v6_2 import (
    RetinallyRecoverableSoftPatientAtlas,
    V6_2_RECOVERY_DIM,
)
from train_soft_patient_atlas import PatientIdSplit


class FittedAIReadIV62FoldRepresentation(FittedAIReadIV5FoldRepresentation):
    """One fold-specific V6.2 state and its target-free local baselines."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.atlas_basis_token = hash_json(
            {
                "model_state": self.provenance.model_token,
                "preprocessor": self.preprocessor.bundle_sha256,
                "vector": "v6.2-retinally-recoverable-evidence-288",
            }
        )


class AIReadIV62FoldRepresentationFactory(AIReadIV5FoldRepresentationFactory):
    """Refit V6.2 and its retinal recovery basis inside every outer fold."""

    def fit(
        self, request: RepresentationFitRequest
    ) -> FittedAIReadIV62FoldRepresentation:
        started = time.perf_counter()
        policy = _feature_policy(request.fit)
        for view in (request.validation, request.calibration):
            if not np.array_equal(_feature_policy(view), policy):
                raise ValueError("clinical policy differs across V6.2 phases")
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
            blood_values=np.asarray(request.fit.features["clinical_values"]),
            blood_observed_mask=np.asarray(
                request.fit.features["clinical_observed_mask"]
            ),
            ordered_feature_names=self.ordered_feature_names,
            policy_eligible_mask=policy,
            expected_policy_mask_hash=policy_mask_hash(
                self.schemas.ordered_features_hash, policy
            ),
            ages=np.asarray(request.fit.demographics),
            age_observed_mask=np.asarray(request.fit.demographic_mask),
            eye_embeddings=np.asarray(request.fit.features["eye_embeddings"]),
            eye_observed_mask=np.asarray(request.fit.features["eye_observed_mask"]),
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
        components = torch.tensor(
            eye_pca.components[V6_2_RECOVERY_DIM],
            dtype=phase_batches[0].eye_embeddings.dtype,
        )
        observations = _concatenate_batches(phase_batches)
        patient_ids = tuple(
            patient_id for view in phase_views for patient_id in view.patient_ids
        )
        site_ids = tuple(
            site_id for view in phase_views for site_id in view.site_ids
        )
        split = PatientIdSplit(
            fit=request.fit.patient_ids,
            validation=request.validation.patient_ids,
            calibration=request.calibration.patient_ids,
        )
        atlas_cohort = AtlasCohort(patient_ids, site_ids, observations)

        def model_factory() -> RetinallyRecoverableSoftPatientAtlas:
            torch.manual_seed(training_seed)
            return _build_model(
                root=self.root,
                retinal_components=components,
            )

        orchestration = Stage2OrchestrationConfig(
            **{**asdict(self.orchestration), "seed": training_seed}
        )
        self._progress(
            request.fold_key,
            {
                "event": "v6_2_outer_fold_fit_started",
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
        if not isinstance(result.model, RetinallyRecoverableSoftPatientAtlas):
            raise TypeError("outer-fold fitting returned the wrong V6.2 model")
        recovery = heldout_recovery_health(result.model, phase_batches[2])
        capacity = capacity_health(result.model, phase_batches[2])
        if not recovery["all_gates_pass"] or not capacity["all_gates_pass"]:
            raise RuntimeError("outer-fold V6.2 outcome-free integrity gate failed")
        summary = {
            "fold_key": request.fold_key,
            "architecture": "retinally_recoverable_v6_2_32_160_64",
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
            "heldout_retinal_recovery_explained_variance": recovery[
                "explained_variance"
            ],
            "retinal_recovery_gate_passed": True,
            "capacity_and_prior_reversion_gates_passed": True,
            "runtime_seconds": time.perf_counter() - started,
        }
        self.fold_summaries.append(summary)
        self._progress(
            request.fold_key,
            {"event": "v6_2_outer_fold_fit_completed", **summary},
        )
        return FittedAIReadIV62FoldRepresentation(
            request=request,
            model=result.model,
            preprocessor=preprocessor,
            eye_pca=eye_pca,
            precision_temperatures=result.calibration.precision_temperatures,
            ordered_feature_names=self.ordered_feature_names,
            model_token=result.selection.selected_state_sha256,
        )


__all__ = [
    "AIReadIV62FoldRepresentationFactory",
    "FittedAIReadIV62FoldRepresentation",
]
