"""Fit the single frozen protected-retinal Patient Atlas V6 candidate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from interpret_soft_patient_atlas import heldout_factor_relevance
from patient_atlas_aireadi_internal_crossfit import _fit_eye_pca
from patient_atlas_checkpoint import (
    CheckpointExpectations,
    build_checkpoint_metadata,
    save_atlas_checkpoint,
)
from patient_atlas_interpretability_report import DEFAULT_PROFILE_THRESHOLDS
from patient_atlas_preprocessing import FoldPreprocessor, hash_json
from patient_atlas_prospective_policy import EXPECTED_FALSE_INDICES, apply_prospective_policy
from patient_atlas_real_data import ZeroBloodAnchor, load_exploratory_raw_cohort
from patient_atlas_stage2 import AtlasCohort, fit_stage2_atlas
from run_patient_atlas_exploratory_stage2 import (
    _redacted_error,
    _selected_model_config,
    _sha256,
    _write_exclusive_json,
)
from run_patient_atlas_v5_stage2 import (
    TRAINING_SEED,
    V5_OUTCOME_FREE_THRESHOLD,
    _forward_view,
    _orchestration,
    capacity_health,
)
from soft_patient_atlas import PatientAtlasConfig
from soft_patient_atlas_v5 import V5_PARTITION
from soft_patient_atlas_v6 import (
    ProtectedRetinalSubspaceSoftPatientAtlas,
    V6_PROTECTED_RETINAL_DECODER_AMPLITUDE,
    V6_PROTECTED_RETINAL_DIM,
    V6_PROTECTED_RETINAL_PRECISION,
)
from train_soft_patient_atlas import (
    AtlasTrainingBatch,
    Stage2TrainingConfig,
    deterministic_patient_split,
)


SCHEMA_VERSION = "patient-atlas-v6-stage2-run-v1"
PROTOCOL_NAME = "PATIENT_ATLAS_V6_DEVELOPMENT_PROTOCOL_V1.json"
PROTOCOL_SCHEMA_VERSION = "patient-atlas-v6-development-protocol-v1"


def _retinal_basis_sha256(components: torch.Tensor | np.ndarray) -> str:
    value = np.ascontiguousarray(
        torch.as_tensor(components).detach().cpu().numpy().astype(np.float32)
    )
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def validate_v6_protocol(root: Path, path: Path) -> dict[str, Any]:
    canonical = (root / PROTOCOL_NAME).resolve()
    if path.resolve() != canonical:
        raise ValueError("V6 protocol must be the canonical project artifact")
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or value.get("status")
        != "frozen_after_value_paired_activation_before_v6_patient_training_or_target_scoring"
    ):
        raise ValueError("V6 protocol is not frozen at the required boundary")
    bindings = value.get("bindings", {})
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("V6 protocol bindings are absent")
    for label, raw in bindings.items():
        if not isinstance(raw, dict) or set(raw) != {"file", "sha256"}:
            raise ValueError(f"malformed V6 protocol binding: {label}")
        source = root / str(raw["file"])
        if not source.is_file() or _sha256(source) != raw["sha256"]:
            raise ValueError(f"V6 protocol binding differs: {label}")
    activation = json.loads(
        (root / bindings["value_paired_milestone"]["file"]).read_text()
    )
    if (
        activation.get("decision", {}).get(
            "single_outcome_free_v6_branch_activated"
        )
        is not True
        or activation.get("decision", {}).get("patient_target_scoring_allowed_now")
        is not False
    ):
        raise ValueError("V6 activation milestone differs")
    expected_architecture = {
        "name": "protected_retinal_subspace_probabilistic_patient_atlas_v6",
        "latent_partition": {
            "shared": 32,
            "eye_private": 160,
            "clinical_private": 64,
            "total": 256,
        },
        "protected_retinal_dimension": 64,
        "learned_nonlinear_eye_private_dimension": 96,
        "protected_retinal_precision_increment": 1.0,
        "protected_decoder_initial_amplitude": 2.0,
        "default_vector_dimension": 289,
        "uncertainty_sidecar_dimension": 288,
        "interaction_enabled": False,
        "external_blood_anchor_enabled": False,
        "raw_or_pca_coordinates_appended_outside_latent_state": False,
    }
    if value.get("architecture") != expected_architecture:
        raise ValueError("V6 architecture protocol differs")
    expected_training = {
        "beta": 0.01,
        "group_shrinkage_rate": 1e-6,
        "batch_size": 96,
        "maximum_steps": 6000,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "validation_interval": 250,
        "early_stopping_patience": 8,
        "fit_validation_calibration_fractions": [0.7, 0.15, 0.15],
        "base_training_seed": TRAINING_SEED,
    }
    if value.get("training") != expected_training:
        raise ValueError("V6 training protocol differs")
    if value.get("outcome_free_gate") != {
        "balanced_proper_score_maximum": V5_OUTCOME_FREE_THRESHOLD,
        "all_v5_capacity_and_prior_reversion_gates_required": True,
        "all_64_protected_retinal_factors_decoder_active_required": True,
        "protected_center_recovery_maximum_absolute_error": 1e-6,
        "protected_missing_view_prior_reversion_required": True,
        "screening_or_functional_targets_loaded": False,
        "target_evaluation_unlocked_only_if_all_gates_pass": True,
    }:
        raise ValueError("V6 outcome-free gate differs")
    return value


def _build_model(
    *,
    root: Path,
    group_shrinkage_rate: float,
    retinal_components: torch.Tensor,
) -> ProtectedRetinalSubspaceSoftPatientAtlas:
    config = PatientAtlasConfig(
        eye_dim=384,
        blood_anchor_dim=64,
        num_continuous=48,
        num_binary=11,
        demographic_dim=1,
        num_devices=5,
        num_lateralities=3,
        latent_dim=256,
        hidden_dim=192,
        interaction_rank=8,
        group_shrinkage_rate=float(group_shrinkage_rate),
    )
    contract = json.loads((root / "EXTERNAL_BLOOD_TOWER_CONTRACT.json").read_text())
    anchor_mask = torch.tensor(
        contract["anchor_pretraining_coverage"]["anchor_eligible_mask"],
        dtype=torch.bool,
    )
    model = ProtectedRetinalSubspaceSoftPatientAtlas(
        config,
        ZeroBloodAnchor.build(64),
        blood_anchor_eligible_mask=anchor_mask,
        retinal_components=retinal_components,
    )
    model.lock_interaction()
    return model


def protected_retinal_health(
    model: ProtectedRetinalSubspaceSoftPatientAtlas,
    batch: AtlasTrainingBatch,
) -> dict[str, Any]:
    protected = model.protected_retinal_slice
    with torch.inference_mode():
        output = _forward_view(model, batch, use_eye=True, use_clinical=True)
        mask = batch.eye_observed_mask
        count = mask.sum(dim=1, keepdim=True)
        present = count[:, 0] > 0
        safe = torch.where(
            mask[..., None], batch.eye_embeddings, torch.zeros_like(batch.eye_embeddings)
        )
        mean = safe.sum(dim=1) / count.clamp(min=1).to(
            dtype=batch.eye_embeddings.dtype
        )
        expected = mean @ model.eye_encoder.retinal_components
        observed = output.eye_evidence.center[:, protected]
        recovery_error = (
            float((observed[present] - expected[present]).abs().max().item())
            if bool(present.any())
            else math.inf
        )
        precision = output.eye_evidence.precision_increment[:, protected]
        present_precision_error = (
            float((precision[present] - 1.0).abs().max().item())
            if bool(present.any())
            else math.inf
        )
        missing = _forward_view(model, batch, use_eye=False, use_clinical=True)
        missing_vector = model.structured_vector(missing)
        missing_prior = bool(
            torch.equal(
                missing_vector.mean[:, 64:128],
                torch.zeros_like(missing_vector.mean[:, 64:128]),
            )
            and torch.equal(
                missing_vector.log_variance_sidecar[:, 64:128],
                torch.zeros_like(
                    missing_vector.log_variance_sidecar[:, 64:128]
                ),
            )
        )
        clinical_visible = batch.blood_observed_mask & batch.blood_eligible_mask
        relevance = heldout_factor_relevance(
            model.decoder,
            latent_mean=output.physiology.mean,
            latent_log_variance=output.physiology.log_variance,
            demographics=batch.demographics,
            target_eye_embeddings=batch.eye_embeddings,
            target_eye_mask=batch.eye_observed_mask,
            target_blood_values=batch.blood_values,
            target_blood_mask=clinical_visible,
            target_blood_eligible_mask=batch.blood_eligible_mask,
            eye_device_ids=batch.eye_device_ids,
            eye_laterality_ids=batch.eye_laterality_ids,
            split_role="weight_heldout_calibration",
        )
        threshold = float(DEFAULT_PROFILE_THRESHOLDS.min_expected_fisher)
        active = int(
            (relevance.eye_expected_fisher[protected] >= threshold).sum().item()
        )
        components = model.eye_encoder.retinal_components
        gram_error = float(
            (
                components.T @ components
                - torch.eye(
                    V6_PROTECTED_RETINAL_DIM,
                    dtype=components.dtype,
                    device=components.device,
                )
            )
            .abs()
            .max()
            .item()
        )
    gates = {
        "protected_center_recovery": recovery_error <= 1e-6,
        "protected_precision_exact_when_present": present_precision_error <= 1e-7,
        "protected_missing_view_exact_prior": missing_prior,
        "all_64_protected_factors_decoder_active": active == 64,
        "protected_basis_orthonormal": gram_error <= 1e-5,
    }
    return {
        "evaluation_role": "stage2_calibration_patients_held_out_from_weight_fitting",
        "protected_dimension": 64,
        "learned_nonlinear_eye_private_dimension": 96,
        "center_recovery_maximum_absolute_error": recovery_error,
        "present_precision_maximum_absolute_error": present_precision_error,
        "decoder_active_protected_factors": active,
        "decoder_relevance_threshold": threshold,
        "basis_orthonormality_maximum_absolute_error": gram_error,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def _promotion_decision(
    balanced_proper_score: float,
    *,
    capacity_gates_pass: bool,
    protected_gates_pass: bool,
) -> tuple[bool, str]:
    if not math.isfinite(float(balanced_proper_score)):
        raise ValueError("V6 score must be finite")
    if type(capacity_gates_pass) is not bool or type(protected_gates_pass) is not bool:
        raise TypeError("V6 gate inputs must be boolean")
    passed = (
        float(balanced_proper_score) <= V5_OUTCOME_FREE_THRESHOLD
        and capacity_gates_pass
        and protected_gates_pass
    )
    return passed, (
        "unlock_frozen_v6_nested_internal_target_evaluation"
        if passed
        else "stop_v6_before_target_evaluation"
    )


def run_v6_stage2(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    clinical_project_root: str | Path,
    preprocessor_path: str | Path,
    protocol_path: str | Path,
    checkpoint_output: str | Path,
    summary_output: str | Path,
    failure_output: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    preprocessor_path = Path(preprocessor_path).resolve()
    protocol_path = Path(protocol_path).resolve()
    checkpoint_output = Path(checkpoint_output).resolve()
    summary_output = Path(summary_output).resolve()
    failure_output = Path(failure_output).resolve()
    if any(path.exists() for path in (checkpoint_output, summary_output, failure_output)):
        raise FileExistsError("V6 Stage-2 output/failure paths must all be new")
    protocol = validate_v6_protocol(root, protocol_path)
    cohort = apply_prospective_policy(
        load_exploratory_raw_cohort(
            project_root=root,
            dataset_root=dataset_root,
            clinical_project_root=clinical_project_root,
        ),
        project_root=root,
    )
    if "test" in set(cohort.split_labels):
        raise PermissionError("retired official test entered the V6 development cohort")
    preprocessor = FoldPreprocessor.load(preprocessor_path)
    if (
        preprocessor.fit_scope != "exploratory"
        or preprocessor.source_policy_sha256 != cohort.source_policy_sha256
    ):
        raise ValueError("V6 preprocessor scope or source policy differs")
    official_train = cohort.indices_for_split("train")
    patient_ids = tuple(cohort.patient_ids[index] for index in official_train)
    site_ids = tuple(cohort.site_ids[index] for index in official_train)
    training_batch = cohort.to_training_batch(preprocessor, indices=official_train)
    training = Stage2TrainingConfig()
    split = deterministic_patient_split(patient_ids, site_ids, training)
    if (
        preprocessor.provenance.representation_fit_patient_id_hash
        != hash_json(list(split.fit))
        or preprocessor.provenance.validation_patient_id_hash
        != hash_json(list(split.validation))
        or preprocessor.provenance.calibration_patient_id_hash
        != hash_json(list(split.calibration))
    ):
        raise ValueError("V6 split does not match preprocessor provenance")
    atlas_cohort = AtlasCohort(patient_ids, site_ids, training_batch)
    fit_batch = atlas_cohort.take(atlas_cohort.indices_for(split.fit))
    eye_pca = _fit_eye_pca(fit_batch)
    components = torch.tensor(
        eye_pca.components[V6_PROTECTED_RETINAL_DIM],
        dtype=training_batch.eye_embeddings.dtype,
    )
    basis_sha256 = _retinal_basis_sha256(components)

    def model_factory() -> ProtectedRetinalSubspaceSoftPatientAtlas:
        torch.manual_seed(TRAINING_SEED)
        return _build_model(
            root=root,
            group_shrinkage_rate=1e-6,
            retinal_components=components,
        )

    def progress(event: Mapping[str, object]) -> None:
        print(
            json.dumps(
                {
                    **dict(event),
                    "model_family": "protected_retinal_subspace_v6",
                    "patient_details_emitted": False,
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )

    started = time.perf_counter()
    try:
        result = fit_stage2_atlas(
            model_factory,
            atlas_cohort,
            training,
            _orchestration(),
            split=split,
            progress_callback=progress,
        )
        if not isinstance(result.model, ProtectedRetinalSubspaceSoftPatientAtlas):
            raise TypeError("V6 orchestration returned the wrong model family")
        calibration_batch = atlas_cohort.take(
            atlas_cohort.indices_for(split.calibration)
        )
        capacity = capacity_health(result.model, calibration_batch)
        protected_health = protected_retinal_health(
            result.model, calibration_batch
        )
        selected_score = float(
            result.selection.candidates[0].best_balanced_proper_score
        )
        gate_pass, decision = _promotion_decision(
            selected_score,
            capacity_gates_pass=bool(capacity["all_gates_pass"]),
            protected_gates_pass=bool(protected_health["all_gates_pass"]),
        )
        eye_contract = json.loads(
            (root / "EXTERNAL_EYE_TOWER_CONTRACT.json").read_text()
        )
        code_names = (
            "soft_patient_atlas.py",
            "soft_patient_atlas_v2.py",
            "soft_patient_atlas_v5.py",
            "soft_patient_atlas_v6.py",
            "patient_atlas_v6_vector.py",
            "train_soft_patient_atlas.py",
            "patient_atlas_stage2.py",
            "interpret_soft_patient_atlas.py",
            "patient_atlas_preprocessing.py",
            "patient_atlas_real_data.py",
            "run_patient_atlas_v6_stage2.py",
            PROTOCOL_NAME,
            "PATIENT_ATLAS_V6_ARCHITECTURE_BRANCH_POLICY_V1.json",
            "PATIENT_ATLAS_V6_VALUE_PAIRED_MILESTONE_V1.json",
            "PATIENT_ATLAS_SOURCE_POLICY.json",
            "patient_atlas_prospective_policy.py",
            "PATIENT_ATLAS_CONTINUOUS_VALIDITY_POLICY_V1.json",
        )
        code_hashes = {name: _sha256(root / name) for name in code_names}
        metadata = build_checkpoint_metadata(
            model_config=_selected_model_config(result.model),
            patient_state_schema={
                "fit_scope": "adaptive_v6_official_train_only",
                "fused_latent_dimension": 256,
                "paired_evidence_vector_name": "protected_retinal_subspace_patient_atlas_v6_289",
                "paired_evidence_vector_dimension": 289,
                "paired_evidence_vector_order": [
                    "eye_shared_mean[0:32]",
                    "clinical_shared_mean[0:32]",
                    "eye_private_mean[0:160; first 64 protected retinal]",
                    "clinical_private_mean[0:64]",
                    "standardized_age",
                ],
                "log_variance_sidecar_dimension": 288,
                "availability_sidecar_not_in_default_vector": True,
                "raw_or_pca_coordinates_appended": False,
            },
            factor_capacity_and_ordering={
                "factor_capacity": 256,
                "shared_dimension": 32,
                "eye_private_dimension": 160,
                "protected_retinal_dimension": 64,
                "learned_nonlinear_eye_private_dimension": 96,
                "clinical_private_dimension": 64,
                "eye_permitted_factor_dimension": 192,
                "clinical_permitted_factor_dimension": 96,
                "roles": list(V5_PARTITION.roles),
                "forbidden_cross_private_loadings_exactly_zero": True,
                "axis_naming": "protected retinal basis loadings; all other axes fail closed unless signed-permutation stable",
            },
            preprocessor=preprocessor,
            tower_keys_and_state_hashes={
                "eye_tower_file": eye_contract["file_sha256"],
                "eye_tower_state": eye_contract["state_sha256"],
                "protected_retinal_basis": basis_sha256,
                "blood_anchor": hash_json(
                    {
                        "type": "exact_zero_anchor",
                        "width": 64,
                        "source_policy": cohort.source_policy_sha256,
                    }
                ),
            },
            initializer_hash=result.selection.initializer_state_sha256,
            paired_correction_gate_result={
                "enabled": False,
                "status": "locked_additive_v6",
            },
            calibration_parameters=result.checkpoint_parameters(),
            code_config_source_hashes=code_hashes,
            seed=TRAINING_SEED,
            creation_time_utc=datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        )
        saved = save_atlas_checkpoint(
            checkpoint_output, result.model, metadata, preprocessor=preprocessor
        )
        expectations = CheckpointExpectations.from_metadata(
            metadata, state_sha256=saved.state_sha256
        )
    except BaseException as error:
        _write_exclusive_json(
            failure_output,
            {
                "schema_version": "patient-atlas-v6-stage2-failure-v1",
                "terminal_for_attempt": True,
                "same_protocol_technical_retry_allowed": True,
                "error_type": type(error).__name__,
                "error_message": _redacted_error(error, patient_ids),
                "elapsed_seconds": time.perf_counter() - started,
                "screening_targets_loaded": False,
                "functional_outcomes_loaded": False,
                "official_test_inputs_loaded": False,
                "official_test_targets_loaded": False,
                "patient_rows_identifiers_vectors_or_predictions_emitted": False,
            },
        )
        raise
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scope": "adaptive_v6_ai_readi_recommended_train_inputs_only",
        "decision": decision,
        "outcome_free_gate_pass": gate_pass,
        "runtime_seconds": time.perf_counter() - started,
        "screening_targets_loaded": False,
        "functional_outcomes_loaded": False,
        "official_test_inputs_loaded": False,
        "official_test_targets_loaded": False,
        "model_family": "protected_retinal_subspace_probabilistic_patient_atlas_v6",
        "latent_partition": {
            "shared": 32,
            "eye_private": 160,
            "clinical_private": 64,
        },
        "eye_private_partition": {
            "protected_retinal": 64,
            "learned_nonlinear": 96,
        },
        "vector_dimension": 289,
        "uncertainty_sidecar_dimension": 288,
        "raw_or_pca_coordinates_appended": False,
        "protected_retinal_coordinates_inside_probabilistic_state": True,
        "interaction_enabled": False,
        "external_blood_anchor_enabled": False,
        "retinal_pixel_decoder": "deferred",
        "patient_count": len(patient_ids),
        "split_counts": split.counts,
        "source_policy_sha256": cohort.source_policy_sha256,
        "protected_retinal_basis": {
            "fit_role": "representation_fit_only",
            "dimension": 64,
            "sha256": basis_sha256,
            "patient_balanced_input_summary": True,
        },
        "prospective_policy": {
            "applied": True,
            "eligible_clinical_fields": int(cohort.blood_eligible_mask.sum()),
            "masked_unit_conflict_indices": list(EXPECTED_FALSE_INDICES),
        },
        "protocol": {
            "file": protocol_path.name,
            "file_sha256": _sha256(protocol_path),
            "status": protocol["status"],
        },
        "preprocessor": {
            "file_sha256": _sha256(preprocessor_path),
            "bundle_sha256": preprocessor.bundle_sha256,
        },
        "selection": result.selection.to_dict(),
        "calibration": result.calibration.to_dict(),
        "outcome_free_acceptance": {
            "v6_balanced_proper_score": selected_score,
            "frozen_maximum": V5_OUTCOME_FREE_THRESHOLD,
            "lower_is_better": True,
            "score_gate_pass": selected_score <= V5_OUTCOME_FREE_THRESHOLD,
            "v5_capacity_gate_pass": bool(capacity["all_gates_pass"]),
            "protected_retinal_gate_pass": bool(
                protected_health["all_gates_pass"]
            ),
            "all_gates_pass": gate_pass,
        },
        "capacity_interpretability": capacity,
        "protected_retinal_health": protected_health,
        "checkpoint": {
            "file": checkpoint_output.name,
            "file_sha256": saved.file_sha256,
            "state_sha256": saved.state_sha256,
            "metadata_sha256": saved.metadata_sha256,
            "expectations": asdict(expectations),
        },
        "code_hashes": code_hashes,
        "claim_limitations": {
            "adaptive_internal_target_evaluation_unlocked": gate_pass,
            "screening_superiority_evaluated": False,
            "external_validation_completed": False,
            "clinical_benefit_claim_allowed": False,
        },
        "patient_rows_identifiers_targets_predictions_vectors_or_embeddings_emitted": False,
    }
    _write_exclusive_json(summary_output, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clinical-project-root", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--failure-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run_v6_stage2(
            project_root=args.project_root,
            dataset_root=args.dataset_root,
            clinical_project_root=args.clinical_project_root,
            preprocessor_path=args.preprocessor,
            protocol_path=args.protocol,
            checkpoint_output=args.checkpoint_output,
            summary_output=args.summary_output,
            failure_output=args.failure_output,
        )
    except BaseException:
        print(
            json.dumps(
                {
                    "event": "patient_atlas_v6_stage2_failed",
                    "details_emitted": False,
                    "failure_artifact_written": args.failure_output.is_file(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1
    print(
        json.dumps(
            {
                "event": "patient_atlas_v6_stage2_completed",
                "decision": summary["decision"],
                "outcome_free_gate_pass": summary["outcome_free_gate_pass"],
                "balanced_proper_score": summary["outcome_free_acceptance"][
                    "v6_balanced_proper_score"
                ],
                "patient_details_emitted": False,
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROTOCOL_NAME",
    "PROTOCOL_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "_build_model",
    "_promotion_decision",
    "_retinal_basis_sha256",
    "protected_retinal_health",
    "run_v6_stage2",
    "validate_v6_protocol",
]
