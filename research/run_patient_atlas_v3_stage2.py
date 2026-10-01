"""Fit the one frozen capacity-preserving Patient Atlas v3 candidate locally.

Only official-train inputs enter representation fitting. Functional outcomes
and the retired v1 official test are never loaded. Terminal output and written
artifacts are aggregate/hash-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from interpret_soft_patient_atlas import heldout_factor_relevance
from patient_atlas_checkpoint import (
    CheckpointExpectations,
    build_checkpoint_metadata,
    save_atlas_checkpoint,
)
from patient_atlas_interpretability_report import DEFAULT_PROFILE_THRESHOLDS
from patient_atlas_preprocessing import FoldPreprocessor, hash_json
from patient_atlas_prospective_policy import EXPECTED_FALSE_INDICES, apply_prospective_policy
from patient_atlas_real_data import ZeroBloodAnchor, load_exploratory_raw_cohort
from patient_atlas_stage2 import AtlasCohort, Stage2OrchestrationConfig, fit_stage2_atlas
from patient_atlas_v3_protocol import V1_OUTCOME_FREE_THRESHOLD, validate_v3_protocol
from run_patient_atlas_exploratory_stage2 import (
    _redacted_error,
    _selected_model_config,
    _sha256,
    _write_exclusive_json,
)
from soft_patient_atlas import PatientAtlasConfig
from soft_patient_atlas_v3 import CapacityPreservingSoftPatientAtlas, V3_PARTITION
from train_soft_patient_atlas import AtlasTrainingBatch, Stage2TrainingConfig, deterministic_patient_split


SCHEMA_VERSION = "patient-atlas-v3-stage2-run-v1"
TRAINING_SEED = 20260826


def _orchestration() -> Stage2OrchestrationConfig:
    """Keep every training constant fixed; architecture is the sole change."""

    return Stage2OrchestrationConfig(
        beta_candidates=(0.01,),
        group_shrinkage_rate_candidates=(1e-6,),
        grid_protocol="frozen_selected_v1",
    )


def _promotion_decision(
    balanced_proper_score: float, *, capacity_gates_pass: bool
) -> tuple[bool, str]:
    """Apply the frozen outcome-free threshold without tolerance or retuning."""

    score = float(balanced_proper_score)
    if not math.isfinite(score) or type(capacity_gates_pass) is not bool:
        raise ValueError("v3 promotion inputs are malformed")
    passed = score <= V1_OUTCOME_FREE_THRESHOLD and capacity_gates_pass
    return (
        passed,
        (
            "promote_for_new_untouched_validation"
            if passed
            else "reject_exact_v3_candidate"
        ),
    )


def _build_model(
    *, root: Path, group_shrinkage_rate: float
) -> CapacityPreservingSoftPatientAtlas:
    config = PatientAtlasConfig(
        eye_dim=384,
        blood_anchor_dim=64,
        num_continuous=48,
        num_binary=11,
        demographic_dim=1,
        num_devices=5,
        num_lateralities=3,
        latent_dim=96,
        hidden_dim=64,
        interaction_rank=8,
        group_shrinkage_rate=float(group_shrinkage_rate),
    )
    contract = json.loads((root / "EXTERNAL_BLOOD_TOWER_CONTRACT.json").read_text())
    anchor_mask = torch.tensor(
        contract["anchor_pretraining_coverage"]["anchor_eligible_mask"],
        dtype=torch.bool,
    )
    model = CapacityPreservingSoftPatientAtlas(
        config,
        ZeroBloodAnchor.build(64),
        blood_anchor_eligible_mask=anchor_mask,
    )
    model.lock_interaction()
    return model


def _forward_view(
    model: CapacityPreservingSoftPatientAtlas,
    batch: AtlasTrainingBatch,
    *,
    use_eye: bool,
    use_clinical: bool,
):
    eye_visible = batch.eye_observed_mask
    clinical_visible = batch.blood_observed_mask & batch.blood_eligible_mask
    return model(
        eye_embeddings=batch.eye_embeddings,
        eye_visible_mask=(
            eye_visible if use_eye else torch.zeros_like(eye_visible)
        ),
        blood_values=batch.blood_values,
        blood_visible_mask=(
            clinical_visible
            if use_clinical
            else torch.zeros_like(clinical_visible)
        ),
        blood_eligible_mask=batch.blood_eligible_mask,
        demographics=batch.demographics,
        demographic_mask=batch.demographic_mask,
        eye_device_ids=batch.eye_device_ids,
        eye_laterality_ids=batch.eye_laterality_ids,
        eye_quality=batch.eye_quality,
        enable_interaction=False,
    )


def capacity_health(
    model: CapacityPreservingSoftPatientAtlas,
    batch: AtlasTrainingBatch,
) -> dict[str, Any]:
    """Audit aggregate relevance on rows held out from weight fitting."""

    clinical_visible = batch.blood_observed_mask & batch.blood_eligible_mask
    with torch.inference_mode():
        both = _forward_view(model, batch, use_eye=True, use_clinical=True)
        relevance = heldout_factor_relevance(
            model.decoder,
            latent_mean=both.physiology.mean,
            latent_log_variance=both.physiology.log_variance,
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
        eye_only = model.structured_vector(
            _forward_view(model, batch, use_eye=True, use_clinical=False)
        )
        clinical_only = model.structured_vector(
            _forward_view(model, batch, use_eye=False, use_clinical=True)
        )

    p = model.partition
    threshold = float(DEFAULT_PROFILE_THRESHOLDS.min_expected_fisher)
    eye_fisher = relevance.eye_expected_fisher.detach().cpu()
    clinical_fisher = relevance.blood_clinical_expected_fisher.detach().cpu()
    shared_active = int(
        (
            (eye_fisher[p.shared_slice] >= threshold)
            & (clinical_fisher[p.shared_slice] >= threshold)
        ).sum()
    )
    eye_private_active = int((eye_fisher[p.eye_private_slice] >= threshold).sum())
    clinical_private_active = int(
        (clinical_fisher[p.clinical_private_slice] >= threshold).sum()
    )
    forbidden_eye = model.decoder.eye_amplitude[p.clinical_private_slice].detach()
    forbidden_clinical = model.decoder.blood_amplitude[p.eye_private_slice].detach()
    forbidden_zero = bool(
        torch.equal(forbidden_eye, torch.zeros_like(forbidden_eye))
        and torch.equal(forbidden_clinical, torch.zeros_like(forbidden_clinical))
        and torch.equal(
            eye_fisher[p.clinical_private_slice],
            torch.zeros_like(eye_fisher[p.clinical_private_slice]),
        )
        and torch.equal(
            clinical_fisher[p.eye_private_slice],
            torch.zeros_like(clinical_fisher[p.eye_private_slice]),
        )
    )
    missing_eye_prior = bool(
        torch.equal(
            clinical_only.mean[:, 64:96],
            torch.zeros_like(clinical_only.mean[:, 64:96]),
        )
        and torch.equal(
            clinical_only.log_variance_sidecar[:, 64:96],
            torch.zeros_like(clinical_only.log_variance_sidecar[:, 64:96]),
        )
    )
    missing_clinical_prior = bool(
        torch.equal(
            eye_only.mean[:, 96:128],
            torch.zeros_like(eye_only.mean[:, 96:128]),
        )
        and torch.equal(
            eye_only.log_variance_sidecar[:, 96:128],
            torch.zeros_like(eye_only.log_variance_sidecar[:, 96:128]),
        )
    )
    gates = {
        "at_least_8_shared_active_in_both_views": shared_active >= 8,
        "at_least_4_eye_private_active": eye_private_active >= 4,
        "at_least_4_clinical_private_active": clinical_private_active >= 4,
        "forbidden_cross_private_relevance_exactly_zero": forbidden_zero,
        "missing_eye_private_returns_exact_standard_normal_prior": missing_eye_prior,
        "missing_clinical_private_returns_exact_standard_normal_prior": missing_clinical_prior,
        "individual_axes_named": 0,
    }
    pass_keys = tuple(key for key in gates if key != "individual_axes_named")
    return {
        "evaluation_role": "stage2_calibration_patients_held_out_from_weight_fitting",
        "relevance_statistic": "likelihood expected Fisher information",
        "nonzero_threshold": threshold,
        "shared_active_in_both_views": shared_active,
        "shared_capacity": p.shared_dim,
        "eye_private_active": eye_private_active,
        "eye_private_capacity": p.eye_private_dim,
        "clinical_private_active": clinical_private_active,
        "clinical_private_capacity": p.clinical_private_dim,
        "heldout_observation_counts": {
            "patients": relevance.patient_count,
            "eye_images": relevance.eye_image_count,
            "continuous_values": relevance.continuous_value_count,
            "binary_values": relevance.binary_value_count,
        },
        "gates": gates,
        "all_gates_pass": all(gates[key] is True for key in pass_keys)
        and gates["individual_axes_named"] == 0,
        "axis_stability": {
            "independent_real_refit_replicates_available": 1,
            "individual_axes_identifiable": 0,
            "individual_axes_named": 0,
            "unresolved_blocks_reported_without_names": True,
        },
    }


def run_v3_stage2(
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
        raise FileExistsError("v3 Stage-2 output/failure paths must all be new")

    protocol = validate_v3_protocol(root, protocol_path)
    cohort = load_exploratory_raw_cohort(
        project_root=root,
        dataset_root=dataset_root,
        clinical_project_root=clinical_project_root,
    )
    cohort = apply_prospective_policy(cohort, project_root=root)
    if "test" in set(cohort.split_labels):
        raise PermissionError("retired official test entered the v3 exploratory cohort")
    preprocessor = FoldPreprocessor.load(preprocessor_path)
    if preprocessor.fit_scope != "exploratory":
        raise ValueError("v3 Stage-2 requires the exploratory fold preprocessor")
    if preprocessor.source_policy_sha256 != cohort.source_policy_sha256:
        raise ValueError("v3 preprocessor and cohort source policies differ")

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
        raise ValueError("v3 split does not match preprocessor provenance")
    atlas_cohort = AtlasCohort(patient_ids, site_ids, training_batch)

    def model_factory() -> CapacityPreservingSoftPatientAtlas:
        torch.manual_seed(TRAINING_SEED)
        return _build_model(root=root, group_shrinkage_rate=1e-6)

    def progress(event: Mapping[str, object]) -> None:
        safe = dict(event)
        safe["model_family"] = "capacity_preserving_structured_v3"
        print(json.dumps(safe, sort_keys=True, allow_nan=False), flush=True)

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
        if not isinstance(result.model, CapacityPreservingSoftPatientAtlas):
            raise TypeError("v3 orchestration returned the wrong model family")
        calibration_batch = atlas_cohort.take(
            atlas_cohort.indices_for(split.calibration)
        )
        capacity = capacity_health(result.model, calibration_batch)
        if len(result.selection.candidates) != 1:
            raise RuntimeError("v3 Stage-2 produced more than one candidate")
        selected_score = float(
            result.selection.candidates[0].best_balanced_proper_score
        )
        outcome_free_pass = selected_score <= V1_OUTCOME_FREE_THRESHOLD
        engineering_pass, decision = _promotion_decision(
            selected_score,
            capacity_gates_pass=bool(capacity["all_gates_pass"]),
        )

        eye_contract = json.loads((root / "EXTERNAL_EYE_TOWER_CONTRACT.json").read_text())
        code_hashes = {
            filename: _sha256(root / filename)
            for filename in (
                "soft_patient_atlas.py",
                "soft_patient_atlas_v2.py",
                "soft_patient_atlas_v3.py",
                "train_soft_patient_atlas.py",
                "patient_atlas_stage2.py",
                "interpret_soft_patient_atlas.py",
                "patient_atlas_preprocessing.py",
                "patient_atlas_real_data.py",
                "patient_atlas_v3_protocol.py",
                "run_patient_atlas_v3_stage2.py",
                "PATIENT_ATLAS_V3_PROTOCOL_V1.json",
                "PATIENT_ATLAS_SOURCE_POLICY.json",
                "patient_atlas_prospective_policy.py",
                "PATIENT_ATLAS_CONTINUOUS_VALIDITY_POLICY_V1.json",
            )
        }
        selected_model_config = _selected_model_config(result.model)
        metadata = build_checkpoint_metadata(
            model_config=selected_model_config,
            patient_state_schema={
                "fit_scope": "exploratory_v3_official_train_only",
                "fused_latent_dimension": 96,
                "paired_evidence_vector_name": (
                    "capacity_preserving_paired_evidence_atlas_129"
                ),
                "paired_evidence_vector_dimension": 129,
                "paired_evidence_vector_order": [
                    "eye_shared_mean[0:32]",
                    "clinical_shared_mean[0:32]",
                    "eye_private_mean[0:32]",
                    "clinical_private_mean[0:32]",
                    "standardized_age",
                ],
                "log_variance_sidecar_dimension": 128,
                "availability_sidecar_not_in_default_vector": True,
            },
            factor_capacity_and_ordering={
                "factor_capacity": 96,
                "shared_dimension": 32,
                "eye_private_dimension": 32,
                "clinical_private_dimension": 32,
                "per_modality_permitted_factor_dimension": 64,
                "roles": list(V3_PARTITION.roles),
                "forbidden_cross_private_loadings_exactly_zero": True,
                "axis_naming": (
                    "fail closed unless stable under signed-permutation matching"
                ),
            },
            preprocessor=preprocessor,
            tower_keys_and_state_hashes={
                "eye_tower_file": eye_contract["file_sha256"],
                "eye_tower_state": eye_contract["state_sha256"],
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
                "status": "locked_capacity_preserving_additive_v3",
            },
            calibration_parameters=result.checkpoint_parameters(),
            code_config_source_hashes=code_hashes,
            seed=TRAINING_SEED,
            creation_time_utc=datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        )
        checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        saved = save_atlas_checkpoint(
            checkpoint_output,
            result.model,
            metadata,
            preprocessor=preprocessor,
        )
        expectations = CheckpointExpectations.from_metadata(
            metadata, state_sha256=saved.state_sha256
        )
    except BaseException as error:
        failure = {
            "schema_version": "patient-atlas-v3-stage2-failure-v1",
            "terminal": True,
            "rerun_with_new_paths_allowed": True,
            "scientific_settings_change_allowed": False,
            "error_type": type(error).__name__,
            "error_message": _redacted_error(error, patient_ids),
            "elapsed_seconds": time.perf_counter() - started,
            "official_validation_functional_outcomes_loaded": False,
            "official_test_inputs_loaded": False,
            "official_test_targets_loaded": False,
            "patient_rows_emitted": False,
            "patient_identifiers_emitted": False,
            "patient_vectors_emitted": False,
            "per_patient_predictions_emitted": False,
        }
        _write_exclusive_json(failure_output, failure)
        raise

    summary = {
        "schema_version": SCHEMA_VERSION,
        "scope": "v3_official_train_inputs_only",
        "decision": decision,
        "engineering_promotion_gate_pass": engineering_pass,
        "runtime_seconds": time.perf_counter() - started,
        "official_validation_functional_outcomes_loaded": False,
        "functional_outcomes_loaded": False,
        "official_test_inputs_loaded": False,
        "official_test_targets_loaded": False,
        "v1_official_test_permanently_retired": True,
        "model_family": "capacity_preserving_structured_probabilistic_atlas",
        "latent_partition": {
            "shared": V3_PARTITION.shared_dim,
            "eye_private": V3_PARTITION.eye_private_dim,
            "clinical_private": V3_PARTITION.clinical_private_dim,
        },
        "per_modality_permitted_factor_dimension": 64,
        "vector_dimension": 129,
        "uncertainty_sidecar_dimension": 128,
        "external_blood_anchor_enabled": False,
        "interaction_enabled": False,
        "retinal_pixel_decoder": "deferred",
        "patient_count": len(patient_ids),
        "split_counts": split.counts,
        "source_policy_sha256": cohort.source_policy_sha256,
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
            "v3_balanced_proper_score": selected_score,
            "authenticated_v1_threshold": V1_OUTCOME_FREE_THRESHOLD,
            "lower_is_better": True,
            "pass": outcome_free_pass,
        },
        "capacity_interpretability": capacity,
        "checkpoint": {
            "file": checkpoint_output.name,
            "file_sha256": saved.file_sha256,
            "state_sha256": saved.state_sha256,
            "metadata_sha256": saved.metadata_sha256,
            "expectations": asdict(expectations),
        },
        "code_hashes": code_hashes,
        "claim_limitations": {
            "new_untouched_cohort_evaluated": False,
            "functional_screening_evaluated": False,
            "v3_superiority_claim_allowed": False,
            "clinical_benefit_claim_allowed": False,
            "retinal_pixel_generation_evaluated": False,
        },
        "patient_rows_emitted": False,
        "patient_identifiers_emitted": False,
        "patient_vectors_emitted": False,
        "per_patient_predictions_emitted": False,
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
        summary = run_v3_stage2(
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
                    "event": "patient_atlas_v3_stage2_failed",
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
                "event": "patient_atlas_v3_stage2_completed",
                "decision": summary["decision"],
                "engineering_promotion_gate_pass": summary[
                    "engineering_promotion_gate_pass"
                ],
                "balanced_proper_score": summary["outcome_free_acceptance"][
                    "v3_balanced_proper_score"
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
    "SCHEMA_VERSION",
    "TRAINING_SEED",
    "capacity_health",
    "run_v3_stage2",
]
