"""Leakage-safe adaptive AI-READI internal evaluation for Patient Atlas v4.

Patient-derived processing stays local. Each outer fold fits preprocessing,
the complete outcome-free v4 representation, calibration, and readouts using
outer-training rows only. Only disclosure-safe aggregate results are written.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from eval_soft_patient_atlas import (
    ARM_BLOOD,
    ARM_BOTH,
    ARM_EYE,
    BASE_STRATUM,
    FeatureView,
    FoldCoordinates,
    MissingnessStratum,
    RepresentationFitRequest,
    assert_aggregate_only_payload,
    evaluate_nested_patient_atlas,
    load_target_manifest,
)
from patient_atlas_aireadi_internal_crossfit import (
    AIReadIV3FoldRepresentationFactory,
    FittedAIReadIFoldRepresentation,
    _concatenate_batches,
    _config_from_protocol,
    _feature_policy,
    _fit_eye_pca,
    _fold_training_seed,
    _preprocessor_provenance,
    _redacted_error,
    _sha256,
    _training_from_protocol,
    _view_to_training_batch,
    _write_exclusive_json,
    build_internal_evaluation_cohort,
)
from patient_atlas_development_validation import load_index_visit_functional_targets
from patient_atlas_preprocessing import (
    fit_outer_fold_preprocessor,
    hash_json,
    policy_mask_hash,
)
from patient_atlas_prospective_policy import apply_prospective_policy
from patient_atlas_real_data import load_exploratory_raw_cohort
from patient_atlas_stage2 import AtlasCohort, Stage2OrchestrationConfig, fit_stage2_atlas
from patient_atlas_v4_vector import encode_capacity_expanded_vector
from run_patient_atlas_v4_stage2 import _build_model
from soft_patient_atlas_v4 import CapacityExpandedSoftPatientAtlas
from train_soft_patient_atlas import PatientIdSplit


PROTOCOL_SCHEMA_VERSION = "patient-atlas-v4-aireadi-internal-evaluation-protocol-v1"
RUN_SCHEMA_VERSION = "patient-atlas-v4-aireadi-internal-evaluation-run-v1"
EXPECTED_GRID = (
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
)
EXPECTED_ARMS = (
    "both_atlas",
    "eye_atlas",
    "blood_clinical_atlas",
    "tuned_complete_case_concat",
    "demographics_only",
    "eye_tower_only",
    "raw_blood_clinical_only",
)


def validate_v4_internal_evaluation_protocol(
    project_root: str | Path, protocol_path: str | Path
) -> dict[str, Any]:
    """Validate all bindings before any patient or target loader is called."""

    root = Path(project_root).resolve()
    path = Path(protocol_path).resolve()
    if path != (root / "PATIENT_ATLAS_V4_AIREADI_INTERNAL_EVALUATION_PROTOCOL_V1.json").resolve():
        raise ValueError("v4 internal evaluation requires the canonical protocol")
    protocol = json.loads(path.read_text())
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("v4 internal evaluation protocol schema differs")
    if (
        protocol.get("status")
        != "frozen_after_outcome_free_v4_gate_before_v4_screening_target_rescoring"
        or protocol.get("role")
        != "adaptive_retrospective_internal_development_evaluation_not_confirmatory"
    ):
        raise ValueError("v4 internal evaluation protocol role/status differs")
    parents = protocol.get("parents", {})
    for parent_name in ("development_protocol", "outcome_free_gate"):
        parent = parents.get(parent_name, {})
        parent_path = root / str(parent.get("file", ""))
        if not parent_path.is_file() or _sha256(parent_path) != parent.get("sha256"):
            raise ValueError(f"v4 {parent_name} binding differs")
    gate_parent = parents["outcome_free_gate"]
    gate = json.loads((root / gate_parent["file"]).read_text())
    if (
        gate.get("decision") != gate_parent.get("required_decision")
        or gate.get("outcome_free_gate_pass") is not True
        or gate.get("outcome_free_acceptance", {}).get("all_gates_pass") is not True
        or gate.get("screening_targets_loaded") is not False
    ):
        raise ValueError("v4 outcome-free gate did not unlock target evaluation")
    cohort = protocol.get("cohort", {})
    if (
        cohort.get("included_recommended_splits") != ["train", "val"]
        or cohort.get("retired_official_test_loaded") is not False
        or cohort.get("retired_official_test_reuse_allowed") is not False
    ):
        raise ValueError("v4 internal evaluation cohort differs")
    representation = protocol.get("representation", {})
    if (
        representation.get("architecture")
        != "capacity_expanded_structured_probabilistic_patient_atlas_v4"
        or representation.get("latent_partition")
        != {"shared": 32, "eye_private": 96, "clinical_private": 64, "total": 192}
        or int(representation.get("default_vector_dimension", 0)) != 225
        or int(representation.get("outcome_coordinate_dimension_excluding_age", 0))
        != 224
        or representation.get("interaction_enabled") is not False
        or representation.get("external_blood_anchor_enabled") is not False
        or representation.get("raw_or_pca_bypass_appended") is not False
        or float(representation.get("beta", -1.0)) != 0.01
        or float(representation.get("group_shrinkage_rate", -1.0)) != 1e-6
    ):
        raise ValueError("v4 internal evaluation representation differs")
    evaluation = protocol.get("evaluation", {})
    if (
        int(evaluation.get("outer_folds", 0)) != 5
        or int(evaluation.get("inner_readout_folds", 0)) != 5
        or evaluation.get("split_seeds") != [1701, 2718, 3141]
        or tuple(float(value) for value in evaluation.get("ridge_grid", ()))
        != EXPECTED_GRID
        or tuple(evaluation.get("enabled_arms", ())) != EXPECTED_ARMS
        or evaluation.get("additional_ridge_grid_expansion_after_this_run") is not False
        or evaluation.get("cross_fold_latent_coordinates_pooled") is not False
        or evaluation.get("same_fold_model_transforms_outer_train_and_outer_test")
        is not True
        or evaluation.get("target_values_enter_representation_fitting") is not False
    ):
        raise ValueError("v4 internal evaluation design differs")
    limits = protocol.get("claim_limits", {})
    if (
        limits.get("may_be_described_as_confirmatory") is not False
        or limits.get("may_be_described_as_external_validation") is not False
        or limits.get("clinical_benefit_claim_allowed") is not False
        or limits.get("retired_official_test_reuse_allowed") is not False
    ):
        raise ValueError("v4 internal evaluation claim limits differ")
    execution = protocol.get("execution_policy", {})
    if (
        execution.get("patient_derived_processing") != "local_only"
        or execution.get("terminal_and_saved_result") != "aggregate_and_hash_only"
        or execution.get("result_path_must_be_new") is not True
    ):
        raise ValueError("v4 internal evaluation execution policy differs")
    return protocol


class FittedAIReadIV4FoldRepresentation(FittedAIReadIFoldRepresentation):
    """One fold-specific 224-coordinate v4 representation and its baselines."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.atlas_basis_token = hash_json(
            {
                "model_state": self.provenance.model_token,
                "preprocessor": self.preprocessor.bundle_sha256,
                "vector": "v4-modality-evidence-224",
            }
        )

    def _atlas(self, view: FeatureView, arm: str, *, batch_size: int = 128) -> np.ndarray:
        token, batch = self._batch(view)
        cache_key = (token, arm)
        if cache_key in self._atlas_cache:
            return self._atlas_cache[cache_key]
        if arm not in (ARM_BOTH, ARM_EYE, ARM_BLOOD):
            raise ValueError("unknown v4 Atlas availability arm")
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
                output = encode_capacity_expanded_vector(
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
                        None if batch.eye_quality is None else batch.eye_quality[start:stop]
                    ),
                )
                chunks.append(
                    output.mean[:, :224].detach().cpu().numpy().astype(np.float64)
                )
        values = np.concatenate(chunks, axis=0)
        if values.shape != (size, 224) or not np.isfinite(values).all():
            raise ValueError("fold v4 Atlas emitted malformed outcome coordinates")
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
        if arm not in (ARM_BOTH, ARM_EYE, ARM_BLOOD):
            return super().transform(
                view,
                arm=arm,
                stratum=stratum,
                missingness=missingness,
                concat_eye_dimension=concat_eye_dimension,
            )
        if stratum != BASE_STRATUM or missingness is not None:
            raise ValueError("v4 internal evaluation has no missingness stratum enabled")
        values = self._atlas(view, arm)
        groups = (
            ("eye_atlas",) * 32
            + ("clinical_atlas",) * 32
            + ("eye_atlas",) * 96
            + ("clinical_atlas",) * 64
        )
        return FoldCoordinates(
            patient_ids=view.patient_ids,
            values=values,
            penalty_groups=groups,
            fold_key=self.provenance.fold_key,
            model_token=self.provenance.model_token,
            basis_token=self.atlas_basis_token,
            arm=arm,
            stratum=stratum,
            concat_eye_dimension=concat_eye_dimension,
        )


class AIReadIV4FoldRepresentationFactory(AIReadIV3FoldRepresentationFactory):
    """Refit the complete outcome-free v4 pipeline inside each outer fold."""

    def fit(self, request: RepresentationFitRequest) -> FittedAIReadIV4FoldRepresentation:
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

        def model_factory() -> CapacityExpandedSoftPatientAtlas:
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
        if not isinstance(result.model, CapacityExpandedSoftPatientAtlas):
            raise TypeError("fold fitting returned the wrong v4 Patient Atlas model")
        summary = {
            "fold_key": request.fold_key,
            "architecture": "capacity_expanded_v4_32_96_64",
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
        return FittedAIReadIV4FoldRepresentation(
            request=request,
            model=result.model,
            preprocessor=preprocessor,
            eye_pca=eye_pca,
            precision_temperatures=result.calibration.precision_temperatures,
            ordered_feature_names=self.ordered_feature_names,
            model_token=result.selection.selected_state_sha256,
        )


def _v4_decision(report: Mapping[str, Any]) -> dict[str, Any]:
    contrasts = report["paired_contrasts"]
    requirements = {
        "both_significantly_beats_eye_only_v4": bool(
            contrasts["eye_minus_both"]["holm_superiority_rejected_null"]
        ),
        "both_significantly_beats_clinical_only_v4": bool(
            contrasts["blood_minus_both"]["holm_superiority_rejected_null"]
        ),
        "both_significantly_beats_regularized_retinal_tower": bool(
            contrasts["eye_tower_minus_both"]["simultaneous_superiority_passed"]
        ),
    }
    return {
        "required_screening_gate_pass": all(requirements.values()),
        "requirements": requirements,
        "raw_clinical_source_superiority": bool(
            contrasts["raw_blood_minus_both"]["simultaneous_superiority_passed"]
        ),
        "concat_comparison_role": "secondary_descriptive_no_noninferiority_margin",
        "lower_primary_loss_is_better": True,
    }


def run_aireadi_v4_internal_evaluation(
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
        raise FileExistsError("v4 internal evaluation output/failure paths must be new")
    protocol = validate_v4_internal_evaluation_protocol(root, protocol_path)
    manifest = load_target_manifest(target_manifest_path)
    if manifest.noninferiority_margin is not None:
        raise ValueError("v4 internal evaluation requires the unchanged null concat margin")

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
            raise PermissionError("retired official test entered v4 internal evaluation")
        patient_ids = cohort.patient_ids
        targets = load_index_visit_functional_targets(
            dataset_root=dataset_root,
            patient_ids=patient_ids,
            manifest_path=target_manifest_path,
            source_policy_path=functional_source_policy_path,
        )
        targets_loaded = True
        evaluation_cohort = build_internal_evaluation_cohort(cohort, targets)
        factory = AIReadIV4FoldRepresentationFactory(
            project_root=root,
            ordered_feature_names=cohort.feature_names,
            source_policy_sha256=cohort.source_policy_sha256,
            source_hashes=cohort.source_hashes,
            base_training_seed=int(
                protocol["representation"]["training"]["base_training_seed"]
            ),
            training=_training_from_protocol(protocol),
            progress_callback=progress_callback,
        )
        report = evaluate_nested_patient_atlas(
            evaluation_cohort,
            manifest,
            factory,
            _config_from_protocol(protocol, noninferiority_margin=None),
        ).to_dict()
        report.update(
            {
                "run_schema_version": RUN_SCHEMA_VERSION,
                "scope": "aireadi_v4_adaptive_retrospective_internal_evaluation",
                "role": protocol["role"],
                "runtime_seconds": time.perf_counter() - started,
                "parents": protocol["parents"],
                "v4_decision": _v4_decision(report),
                "protocol": {
                    "file": protocol_path.name,
                    "file_sha256": _sha256(protocol_path),
                    "status": protocol["status"],
                },
                "data_scope": {
                    "recommended_train_validation_only": True,
                    "retired_official_test_inputs_loaded": False,
                    "retired_official_test_targets_loaded": False,
                },
                "fold_representation_fits": factory.fold_summaries,
                "implementation_hashes": {
                    name: _sha256(root / name)
                    for name in (
                        "patient_atlas_aireadi_v4_evaluation.py",
                        "patient_atlas_aireadi_internal_crossfit.py",
                        "eval_soft_patient_atlas.py",
                        "soft_patient_atlas_v4.py",
                        "patient_atlas_v4_vector.py",
                        "run_patient_atlas_v4_stage2.py",
                        "patient_atlas_stage2.py",
                        "patient_atlas_preprocessing.py",
                        "PATIENT_ATLAS_TARGET_MANIFEST.json",
                    )
                },
                "claim_limitations": {
                    "adaptive_retrospective_internal_development": True,
                    "may_replace_prior_v3_results": False,
                    "external_validation_completed": False,
                    "clinical_benefit_claim_allowed": False,
                    "retired_official_test_reused": False,
                    "missingness_stress_test_evaluated_in_this_run": False,
                },
            }
        )
        assert_aggregate_only_payload(report, forbidden_patient_ids=patient_ids)
        _write_exclusive_json(output_path, report)
        return report
    except BaseException as error:
        _write_exclusive_json(
            failure_path,
            {
                "schema_version": "patient-atlas-v4-aireadi-internal-evaluation-failure-v1",
                "terminal": True,
                "error_type": type(error).__name__,
                "error_message": _redacted_error(error, patient_ids),
                "runtime_seconds": time.perf_counter() - started,
                "targets_loaded": targets_loaded,
                "retired_official_test_inputs_loaded": False,
                "retired_official_test_targets_loaded": False,
                "infrastructure_retry_allowed": True,
                "outcome_driven_architecture_or_grid_change_allowed": False,
                "patient_rows_emitted": False,
                "patient_identifiers_emitted": False,
                "patient_vectors_emitted": False,
                "per_patient_predictions_emitted": False,
            },
        )
        raise


__all__ = [
    "AIReadIV4FoldRepresentationFactory",
    "FittedAIReadIV4FoldRepresentation",
    "PROTOCOL_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "run_aireadi_v4_internal_evaluation",
    "validate_v4_internal_evaluation_protocol",
]
