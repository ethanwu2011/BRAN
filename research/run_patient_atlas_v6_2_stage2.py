"""Run the one outcome-free retinal-recoverability V6.2 candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from patient_atlas_aireadi_internal_crossfit import _fit_eye_pca
from patient_atlas_preprocessing import FoldPreprocessor, hash_json
from patient_atlas_prospective_policy import EXPECTED_FALSE_INDICES, apply_prospective_policy
from patient_atlas_real_data import ZeroBloodAnchor, load_exploratory_raw_cohort
from patient_atlas_stage2 import AtlasCohort, fit_stage2_atlas
from run_patient_atlas_exploratory_stage2 import (
    _redacted_error,
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
from run_patient_atlas_v6_stage2 import _retinal_basis_sha256
from soft_patient_atlas import PatientAtlasConfig
from soft_patient_atlas_v6_2 import (
    RetinallyRecoverableSoftPatientAtlas,
    V6_2_RECOVERY_DIM,
    V6_2_RECOVERY_WEIGHT,
)
from train_soft_patient_atlas import Stage2TrainingConfig, deterministic_patient_split


SCHEMA_VERSION = "patient-atlas-v6-2-stage2-run-v1"
PROTOCOL_NAME = "PATIENT_ATLAS_V6_2_DEVELOPMENT_PROTOCOL_V1.json"
PROTOCOL_SCHEMA_VERSION = "patient-atlas-v6-2-development-protocol-v1"
RECOVERY_R2_MINIMUM = 0.5


def validate_v6_2_protocol(root: Path, path: Path) -> dict[str, Any]:
    if path.resolve() != (root / PROTOCOL_NAME).resolve():
        raise ValueError("V6.2 protocol must be canonical")
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or value.get("status")
        != "frozen_after_protected_subspace_stop_before_v6_2_patient_training_or_target_scoring"
    ):
        raise ValueError("V6.2 protocol is not frozen")
    bindings = value.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("V6.2 bindings are absent")
    for label, raw in bindings.items():
        if not isinstance(raw, dict) or set(raw) != {"file", "sha256"}:
            raise ValueError(f"malformed V6.2 binding: {label}")
        source = root / str(raw["file"])
        if not source.is_file() or _sha256(source) != raw["sha256"]:
            raise ValueError(f"V6.2 binding differs: {label}")
    failed = json.loads(
        (root / bindings["v6_1_stage2_result"]["file"]).read_text()
    )
    if (
        failed.get("outcome_free_gate_pass") is not False
        or failed.get("decision")
        != "stop_protected_subspace_branch_before_target_evaluation"
        or failed.get("screening_targets_loaded") is not False
    ):
        raise ValueError("V6.2 stopped parent differs")
    if value.get("architecture") != {
        "name": "retinally_recoverable_probabilistic_patient_atlas_v6_2",
        "base_latent_partition": {
            "shared": 32,
            "eye_private": 160,
            "clinical_private": 64,
            "total": 256,
        },
        "hard_overwritten_latent_coordinates": 0,
        "retinal_recovery_input_dimension": 160,
        "retinal_recovery_output_dimension": 64,
        "retinal_recovery_head": "bias-free linear",
        "retinal_recovery_loss_weight": 0.05,
        "default_vector_dimension": 289,
        "uncertainty_sidecar_dimension": 288,
        "recovery_output_appended_to_vector": False,
        "interaction_enabled": False,
        "external_blood_anchor_enabled": False,
    }:
        raise ValueError("V6.2 architecture differs")
    if value.get("training") != {
        "beta": 0.01,
        "group_shrinkage_rate": 1e-6,
        "retinal_recovery_loss_weight": 0.05,
        "batch_size": 96,
        "maximum_steps": 6000,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "validation_interval": 250,
        "early_stopping_patience": 8,
        "fit_validation_calibration_fractions": [0.7, 0.15, 0.15],
        "base_training_seed": TRAINING_SEED,
    }:
        raise ValueError("V6.2 training differs")
    if value.get("outcome_free_gate") != {
        "balanced_proper_score_maximum": V5_OUTCOME_FREE_THRESHOLD,
        "heldout_retinal_recovery_explained_variance_minimum": 0.5,
        "all_v5_capacity_and_prior_reversion_gates_required": True,
        "screening_or_functional_targets_loaded": False,
        "target_evaluation_unlocked_only_if_all_gates_pass": True,
    }:
        raise ValueError("V6.2 gate differs")
    return value


def _build_model(
    *, root: Path, retinal_components: torch.Tensor
) -> RetinallyRecoverableSoftPatientAtlas:
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
        group_shrinkage_rate=1e-6,
    )
    contract = json.loads((root / "EXTERNAL_BLOOD_TOWER_CONTRACT.json").read_text())
    anchor_mask = torch.tensor(
        contract["anchor_pretraining_coverage"]["anchor_eligible_mask"],
        dtype=torch.bool,
    )
    model = RetinallyRecoverableSoftPatientAtlas(
        config,
        ZeroBloodAnchor.build(64),
        blood_anchor_eligible_mask=anchor_mask,
        retinal_recovery_components=retinal_components,
    )
    model.lock_interaction()
    return model


def heldout_recovery_health(
    model: RetinallyRecoverableSoftPatientAtlas, batch
) -> dict[str, Any]:
    with torch.inference_mode():
        output = _forward_view(model, batch, use_eye=True, use_clinical=True)
        mask = batch.eye_observed_mask
        present = mask.any(dim=1) & output.eye_evidence.present
        safe = torch.where(
            mask[..., None], batch.eye_embeddings, torch.zeros_like(batch.eye_embeddings)
        )
        count = mask.sum(dim=1, keepdim=True)
        mean = safe.sum(dim=1) / count.clamp(min=1).to(
            dtype=batch.eye_embeddings.dtype
        )
        target = mean @ model.retinal_recovery_components
        prediction = model.retinal_recovery_prediction(output)
        if int(present.sum().item()) < 10:
            r2 = -math.inf
            mse = math.inf
        else:
            target_visible = target[present]
            prediction_visible = prediction[present]
            residual = (prediction_visible - target_visible).square().sum()
            centered = target_visible - target_visible.mean(dim=0, keepdim=True)
            total = centered.square().sum()
            r2 = float((1.0 - residual / total.clamp(min=1e-12)).item())
            mse = float(
                (prediction_visible - target_visible).square().mean().item()
            )
        missing = model.structured_vector(
            _forward_view(model, batch, use_eye=False, use_clinical=True)
        )
        missing_prior = bool(
            torch.equal(missing.mean[:, 0:32], torch.zeros_like(missing.mean[:, 0:32]))
            and torch.equal(
                missing.mean[:, 64:224], torch.zeros_like(missing.mean[:, 64:224])
            )
            and torch.equal(
                missing.log_variance_sidecar[:, 0:32],
                torch.zeros_like(missing.log_variance_sidecar[:, 0:32]),
            )
            and torch.equal(
                missing.log_variance_sidecar[:, 64:224],
                torch.zeros_like(missing.log_variance_sidecar[:, 64:224]),
            )
        )
        rank = int(torch.linalg.matrix_rank(model.retinal_recovery_head.weight).item())
        gram = model.retinal_recovery_components.T @ model.retinal_recovery_components
        basis_error = float(
            (
                gram
                - torch.eye(
                    V6_2_RECOVERY_DIM,
                    dtype=gram.dtype,
                    device=gram.device,
                )
            )
            .abs()
            .max()
            .item()
        )
    gates = {
        "heldout_recovery_r2": r2 >= RECOVERY_R2_MINIMUM,
        "missing_eye_exact_prior": missing_prior,
        "recovery_head_full_row_rank": rank == V6_2_RECOVERY_DIM,
        "recovery_basis_orthonormal": basis_error <= 1e-5,
    }
    return {
        "evaluation_role": "stage2_calibration_patients_held_out_from_weight_fitting",
        "target": "64 fold-fit retinal principal coordinates",
        "explained_variance": r2,
        "mean_squared_error": mse,
        "minimum_explained_variance": RECOVERY_R2_MINIMUM,
        "recovery_head_rank": rank,
        "basis_orthonormality_maximum_absolute_error": basis_error,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def _promotion_decision(
    score: float, *, capacity_pass: bool, recovery_pass: bool
) -> tuple[bool, str]:
    if not math.isfinite(float(score)):
        raise ValueError("V6.2 score must be finite")
    if type(capacity_pass) is not bool or type(recovery_pass) is not bool:
        raise TypeError("V6.2 gate inputs must be boolean")
    passed = (
        score <= V5_OUTCOME_FREE_THRESHOLD and capacity_pass and recovery_pass
    )
    return passed, (
        "unlock_frozen_v6_2_nested_internal_target_evaluation"
        if passed
        else "stop_v6_2_before_target_evaluation"
    )


def run_v6_2_stage2(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    clinical_project_root: str | Path,
    preprocessor_path: str | Path,
    protocol_path: str | Path,
    summary_output: str | Path,
    failure_output: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    preprocessor_path = Path(preprocessor_path).resolve()
    protocol_path = Path(protocol_path).resolve()
    summary_output = Path(summary_output).resolve()
    failure_output = Path(failure_output).resolve()
    if summary_output.exists() or failure_output.exists():
        raise FileExistsError("V6.2 outputs must be new")
    protocol = validate_v6_2_protocol(root, protocol_path)
    cohort = apply_prospective_policy(
        load_exploratory_raw_cohort(
            project_root=root,
            dataset_root=dataset_root,
            clinical_project_root=clinical_project_root,
        ),
        project_root=root,
    )
    if "test" in set(cohort.split_labels):
        raise PermissionError("retired official test entered V6.2 development")
    preprocessor = FoldPreprocessor.load(preprocessor_path)
    if (
        preprocessor.fit_scope != "exploratory"
        or preprocessor.source_policy_sha256 != cohort.source_policy_sha256
    ):
        raise ValueError("V6.2 preprocessor differs")
    official_train = cohort.indices_for_split("train")
    patient_ids = tuple(cohort.patient_ids[index] for index in official_train)
    site_ids = tuple(cohort.site_ids[index] for index in official_train)
    batch = cohort.to_training_batch(preprocessor, indices=official_train)
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
        raise ValueError("V6.2 split differs")
    atlas = AtlasCohort(patient_ids, site_ids, batch)
    fit_batch = atlas.take(atlas.indices_for(split.fit))
    pca = _fit_eye_pca(fit_batch)
    components = torch.tensor(
        pca.components[V6_2_RECOVERY_DIM], dtype=batch.eye_embeddings.dtype
    )
    basis_sha256 = _retinal_basis_sha256(components)

    def factory() -> RetinallyRecoverableSoftPatientAtlas:
        torch.manual_seed(TRAINING_SEED)
        return _build_model(root=root, retinal_components=components)

    def progress(event: Mapping[str, object]) -> None:
        print(
            json.dumps(
                {
                    **dict(event),
                    "model_family": "retinally_recoverable_v6_2",
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
            factory,
            atlas,
            training,
            _orchestration(),
            split=split,
            progress_callback=progress,
        )
        if not isinstance(result.model, RetinallyRecoverableSoftPatientAtlas):
            raise TypeError("V6.2 orchestration returned the wrong model")
        calibration_batch = atlas.take(atlas.indices_for(split.calibration))
        capacity = capacity_health(result.model, calibration_batch)
        recovery = heldout_recovery_health(result.model, calibration_batch)
        score = float(result.selection.candidates[0].best_balanced_proper_score)
        passed, decision = _promotion_decision(
            score,
            capacity_pass=bool(capacity["all_gates_pass"]),
            recovery_pass=bool(recovery["all_gates_pass"]),
        )
    except BaseException as error:
        _write_exclusive_json(
            failure_output,
            {
                "schema_version": "patient-atlas-v6-2-stage2-failure-v1",
                "terminal_for_attempt": True,
                "same_protocol_technical_retry_allowed": True,
                "error_type": type(error).__name__,
                "error_message": _redacted_error(error, patient_ids),
                "elapsed_seconds": time.perf_counter() - started,
                "screening_targets_loaded": False,
                "functional_outcomes_loaded": False,
                "official_test_inputs_or_targets_loaded": False,
                "patient_details_emitted": False,
            },
        )
        raise
    code_names = (
        "soft_patient_atlas_v6_2.py",
        "patient_atlas_v6_2_vector.py",
        "run_patient_atlas_v6_2_stage2.py",
        PROTOCOL_NAME,
        "PATIENT_ATLAS_V6_2_RECOVERABILITY_POLICY_V1.json",
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "outcome_free_v6_2_stage2_complete",
        "decision": decision,
        "outcome_free_gate_pass": passed,
        "runtime_seconds": time.perf_counter() - started,
        "screening_targets_loaded": False,
        "functional_outcomes_loaded": False,
        "official_test_inputs_loaded": False,
        "official_test_targets_loaded": False,
        "model_family": "retinally_recoverable_probabilistic_patient_atlas_v6_2",
        "latent_partition": {
            "shared": 32,
            "eye_private": 160,
            "clinical_private": 64,
        },
        "retinal_recovery": {
            "input_dimension": 160,
            "output_dimension": 64,
            "loss_weight": V6_2_RECOVERY_WEIGHT,
            "output_appended_to_vector": False,
            "basis_sha256": basis_sha256,
            "basis_fit_role": "representation_fit_only",
        },
        "vector_dimension": 289,
        "uncertainty_sidecar_dimension": 288,
        "patient_count": len(patient_ids),
        "split_counts": split.counts,
        "prospective_policy": {
            "applied": True,
            "eligible_clinical_fields": int(cohort.blood_eligible_mask.sum()),
            "masked_unit_conflict_indices": list(EXPECTED_FALSE_INDICES),
        },
        "protocol": {
            "file": protocol_path.name,
            "sha256": _sha256(protocol_path),
            "status": protocol["status"],
        },
        "selection": result.selection.to_dict(),
        "calibration": result.calibration.to_dict(),
        "outcome_free_acceptance": {
            "v6_2_balanced_proper_score": score,
            "frozen_maximum": V5_OUTCOME_FREE_THRESHOLD,
            "score_gate_pass": score <= V5_OUTCOME_FREE_THRESHOLD,
            "v5_capacity_gate_pass": bool(capacity["all_gates_pass"]),
            "retinal_recovery_gate_pass": bool(recovery["all_gates_pass"]),
            "all_gates_pass": passed,
        },
        "capacity_interpretability": capacity,
        "retinal_recovery_health": recovery,
        "selected_state_sha256": result.selection.selected_state_sha256,
        "initializer_state_sha256": result.selection.initializer_state_sha256,
        "code_hashes": {name: _sha256(root / name) for name in code_names},
        "claim_limits": {
            "adaptive_internal_target_evaluation_unlocked": passed,
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
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--failure-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run_v6_2_stage2(
            project_root=args.project_root,
            dataset_root=args.dataset_root,
            clinical_project_root=args.clinical_project_root,
            preprocessor_path=args.preprocessor,
            protocol_path=args.protocol,
            summary_output=args.summary_output,
            failure_output=args.failure_output,
        )
    except BaseException:
        print(
            json.dumps(
                {
                    "event": "patient_atlas_v6_2_stage2_failed",
                    "failure_artifact_written": args.failure_output.is_file(),
                    "patient_details_emitted": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1
    print(
        json.dumps(
            {
                "event": "patient_atlas_v6_2_stage2_completed",
                "decision": summary["decision"],
                "outcome_free_gate_pass": summary["outcome_free_gate_pass"],
                "balanced_proper_score": summary["outcome_free_acceptance"][
                    "v6_2_balanced_proper_score"
                ],
                "retinal_recovery_explained_variance": summary[
                    "retinal_recovery_health"
                ]["explained_variance"],
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
    "RECOVERY_R2_MINIMUM",
    "SCHEMA_VERSION",
    "_build_model",
    "_promotion_decision",
    "heldout_recovery_health",
    "run_v6_2_stage2",
    "validate_v6_2_protocol",
]
