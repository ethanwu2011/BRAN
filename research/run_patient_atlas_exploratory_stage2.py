"""Run a prespecified Patient Atlas Stage-2 protocol.

This command uses official AI-READI train inputs only, with the existing
authenticated exploratory preprocessor and exact-zero blood anchor. It never
loads functional outcomes or the official test split. Progress and final files
contain aggregate metrics/hashes only. Failed attempts are preserved; a later
rerun must use new output paths so the evidence cannot be overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from patient_atlas_checkpoint import (
    CheckpointExpectations,
    build_checkpoint_metadata,
    save_atlas_checkpoint,
)
from patient_atlas_preprocessing import FoldPreprocessor, hash_json
from patient_atlas_real_data import ZeroBloodAnchor, load_exploratory_raw_cohort
from patient_atlas_prospective_policy import (
    EXPECTED_FALSE_INDICES,
    apply_prospective_policy,
)
from patient_atlas_stage2 import (
    AtlasCohort,
    Stage2OrchestrationConfig,
    fit_stage2_atlas,
)
from soft_patient_atlas import PatientAtlasConfig, SoftPatientAtlas
from train_soft_patient_atlas import (
    Stage2TrainingConfig,
    deterministic_patient_split,
)


SCHEMA_VERSION = "patient-atlas-exploratory-stage2-run-v2"
TRAINING_SEED = 20260825


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("xb") as handle:
        handle.write(encoded.encode("utf-8"))


def _redacted_error(error: BaseException, patient_ids: Sequence[str]) -> str:
    message = str(error)
    if any(patient_id and patient_id in message for patient_id in patient_ids):
        return "[REDACTED: exception contained a patient identifier]"
    return message


def _selected_model_config(model: torch.nn.Module) -> dict[str, Any]:
    """Return the frozen post-selection config used by checkpoint validation."""

    config = getattr(model, "config", None)
    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError("selected Atlas model must expose a dataclass config")
    return asdict(config)


def _lower_bound_expansion_config() -> Stage2OrchestrationConfig:
    """Return the one approved follow-up grid after the v1 lower-endpoint win."""

    return Stage2OrchestrationConfig(
        beta_candidates=(0.03, 0.1, 0.3),
        group_shrinkage_rate_candidates=(1e-5, 1e-4, 1e-3),
        grid_protocol="lower_bound_expansion_v1",
    )


def _lower_bound_expansion_v2_config() -> Stage2OrchestrationConfig:
    """Return the final geometric expansion after the v1 edge selection."""

    return Stage2OrchestrationConfig(
        beta_candidates=(0.01, 0.03, 0.1),
        group_shrinkage_rate_candidates=(1e-6, 1e-5, 1e-4),
        grid_protocol="lower_bound_expansion_v2",
    )


def _frozen_selected_config() -> Stage2OrchestrationConfig:
    """Return the already-selected singleton for the masked prospective refit."""

    return Stage2OrchestrationConfig(
        beta_candidates=(0.01,),
        group_shrinkage_rate_candidates=(1e-6,),
        grid_protocol="frozen_selected_v1",
    )


def _grid_config(protocol: str) -> Stage2OrchestrationConfig:
    if protocol == "lower_bound_expansion_v1":
        return _lower_bound_expansion_config()
    if protocol == "lower_bound_expansion_v2":
        return _lower_bound_expansion_v2_config()
    if protocol == "frozen_selected_v1":
        return _frozen_selected_config()
    raise ValueError("unsupported exploratory Stage-2 grid protocol")


def run_exploratory_stage2(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    clinical_project_root: str | Path,
    preprocessor_path: str | Path,
    checkpoint_output: str | Path,
    summary_output: str | Path,
    failure_output: str | Path,
    grid_protocol: str = "lower_bound_expansion_v1",
    prospective_policy: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    preprocessor_path = Path(preprocessor_path).resolve()
    checkpoint_output = Path(checkpoint_output).resolve()
    summary_output = Path(summary_output).resolve()
    failure_output = Path(failure_output).resolve()
    if checkpoint_output.exists() or summary_output.exists() or failure_output.exists():
        raise FileExistsError("Stage-2 output/failure paths must all be new")
    if prospective_policy != (grid_protocol == "frozen_selected_v1"):
        raise ValueError(
            "the prospective policy and frozen_selected_v1 protocol must be used together"
        )

    cohort = load_exploratory_raw_cohort(
        project_root=project_root,
        dataset_root=dataset_root,
        clinical_project_root=clinical_project_root,
    )
    if prospective_policy:
        cohort = apply_prospective_policy(cohort, project_root=project_root)
    preprocessor = FoldPreprocessor.load(preprocessor_path)
    if preprocessor.fit_scope != "exploratory":
        raise ValueError("Stage-2 requires an explicitly exploratory preprocessor")
    if preprocessor.source_policy_sha256 != cohort.source_policy_sha256:
        raise ValueError("preprocessor and cohort source policies differ")

    official_train = cohort.indices_for_split("train")
    patient_ids = tuple(cohort.patient_ids[index] for index in official_train)
    site_ids = tuple(cohort.site_ids[index] for index in official_train)
    training_batch = cohort.to_training_batch(
        preprocessor, indices=official_train
    )
    training = Stage2TrainingConfig()
    orchestration = _grid_config(grid_protocol)
    split = deterministic_patient_split(patient_ids, site_ids, training)
    if (
        preprocessor.provenance.representation_fit_patient_id_hash
        != hash_json(list(split.fit))
        or preprocessor.provenance.validation_patient_id_hash
        != hash_json(list(split.validation))
        or preprocessor.provenance.calibration_patient_id_hash
        != hash_json(list(split.calibration))
    ):
        raise ValueError("Stage-2 split does not match the frozen preprocessor provenance")
    atlas_cohort = AtlasCohort(patient_ids, site_ids, training_batch)

    model_config = PatientAtlasConfig(
        eye_dim=384,
        blood_anchor_dim=64,
        num_continuous=48,
        num_binary=11,
        demographic_dim=1,
        num_devices=5,
        num_lateralities=3,
        latent_dim=64,
        hidden_dim=64,
        interaction_rank=8,
    )
    contract = json.loads(
        (project_root / "EXTERNAL_BLOOD_TOWER_CONTRACT.json").read_text()
    )
    anchor_mask = torch.tensor(
        contract["anchor_pretraining_coverage"]["anchor_eligible_mask"],
        dtype=torch.bool,
    )

    def model_factory() -> SoftPatientAtlas:
        torch.manual_seed(TRAINING_SEED)
        model = SoftPatientAtlas(
            model_config,
            ZeroBloodAnchor.build(64),
            blood_anchor_eligible_mask=anchor_mask,
        )
        model.lock_interaction()
        return model

    def progress(event: Mapping[str, object]) -> None:
        print(json.dumps(dict(event), sort_keys=True, allow_nan=False), flush=True)

    started = time.perf_counter()
    try:
        result = fit_stage2_atlas(
            model_factory,
            atlas_cohort,
            training,
            orchestration,
            split=split,
            progress_callback=progress,
        )
    except BaseException as error:
        failure = {
            "schema_version": "patient-atlas-exploratory-stage2-failure-v2",
            "terminal": True,
            "rerun_with_new_paths_allowed": True,
            "output_overwrite_allowed": False,
            "error_type": type(error).__name__,
            "error_message": _redacted_error(error, patient_ids),
            "elapsed_seconds": time.perf_counter() - started,
            "source_policy_sha256": cohort.source_policy_sha256,
            "preprocessor_bundle_sha256": preprocessor.bundle_sha256,
            "official_test_targets_loaded": False,
            "patient_rows_emitted": False,
            "patient_identifiers_emitted": False,
        }
        _write_exclusive_json(failure_output, failure)
        raise

    try:
        eye_contract = json.loads(
            (project_root / "EXTERNAL_EYE_TOWER_CONTRACT.json").read_text()
        )
        code_hashes = {
            filename: _sha256(project_root / filename)
            for filename in (
                "soft_patient_atlas.py",
                "train_soft_patient_atlas.py",
                "patient_atlas_stage2.py",
                "patient_atlas_preprocessing.py",
                "patient_atlas_real_data.py",
                "run_patient_atlas_exploratory_stage2.py",
                "PATIENT_ATLAS_SOURCE_POLICY.json",
                "patient_atlas_prospective_policy.py",
                "PATIENT_ATLAS_CONTINUOUS_VALIDITY_POLICY_V1.json",
                "PATIENT_ATLAS_OFFICIAL_UNIT_RECONCILIATION_V1.json",
            )
        }
        # Stage-2 replaces group_shrinkage_rate on the selected model config.
        # Checkpoint metadata must describe that frozen selected config, not the
        # initializer used to create the 3x3 candidate grid.
        selected_model_config = _selected_model_config(result.model)
        metadata = build_checkpoint_metadata(
            model_config=selected_model_config,
            patient_state_schema={
                "fit_scope": "exploratory",
                "latent_dimension": 64,
                "default_vector_dimension": 65,
                "default_vector_order": [
                    "physiology_mean[0:64]",
                    "standardized_age",
                ],
                "log_variance_sidecar_dimension": 64,
                "availability_sidecar_not_in_default_vector": True,
            },
            factor_capacity_and_ordering={
                "factor_capacity": 64,
                "view_assignment": "soft eye and blood amplitudes; no fixed shared/private counts",
                "axis_naming": "fail closed unless stable under signed-permutation matching",
                "unresolved_blocks": "report as subspaces",
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
                "status": "locked_primary_additive_model",
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
        post_failure = {
            "schema_version": "patient-atlas-exploratory-stage2-failure-v2",
            "terminal": True,
            "rerun_with_new_paths_allowed": True,
            "output_overwrite_allowed": False,
            "failure_stage": "post_training_checkpoint_or_metadata",
            "error_type": type(error).__name__,
            "error_message": _redacted_error(error, patient_ids),
            "elapsed_seconds": time.perf_counter() - started,
            "source_policy_sha256": cohort.source_policy_sha256,
            "preprocessor_bundle_sha256": preprocessor.bundle_sha256,
            "grid_completed": True,
            "selection_completed": True,
            "calibration_completed": True,
            "selected_beta": result.selection.selected_beta,
            "selected_group_shrinkage_rate": (
                result.selection.selected_group_shrinkage_rate
            ),
            "selected_state_sha256": result.selection.selected_state_sha256,
            "official_test_targets_loaded": False,
            "patient_rows_emitted": False,
            "patient_identifiers_emitted": False,
        }
        _write_exclusive_json(failure_output, post_failure)
        raise
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scope": "exploratory_train_validation_only",
        "runtime_seconds": time.perf_counter() - started,
        "official_test_targets_loaded": False,
        "external_blood_anchor_enabled": False,
        "interaction_enabled": False,
        "patient_count": len(patient_ids),
        "split_counts": split.counts,
        "source_policy_sha256": cohort.source_policy_sha256,
        "prospective_policy": {
            "applied": bool(prospective_policy),
            "eligible_clinical_fields": int(cohort.blood_eligible_mask.sum()),
            "masked_unit_conflict_indices": (
                list(EXPECTED_FALSE_INDICES) if prospective_policy else []
            ),
        },
        "preprocessor": {
            "file_sha256": _sha256(preprocessor_path),
            "bundle_sha256": preprocessor.bundle_sha256,
            "normalization_hashes": dict(
                sorted(preprocessor.normalization_hashes.items())
            ),
        },
        "selection": result.selection.to_dict(),
        "calibration": result.calibration.to_dict(),
        "checkpoint": {
            "file": checkpoint_output.name,
            "file_sha256": saved.file_sha256,
            "state_sha256": saved.state_sha256,
            "metadata_sha256": saved.metadata_sha256,
            "expectations": asdict(expectations),
        },
        "code_hashes": code_hashes,
        "patient_rows_emitted": False,
        "patient_identifiers_emitted": False,
        "patient_vectors_emitted": False,
        "per_patient_predictions_emitted": False,
    }
    _write_exclusive_json(summary_output, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clinical-project-root", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--failure-output", type=Path, required=True)
    parser.add_argument(
        "--grid-protocol",
        choices=(
            "lower_bound_expansion_v1",
            "lower_bound_expansion_v2",
            "frozen_selected_v1",
        ),
        default="lower_bound_expansion_v1",
    )
    parser.add_argument(
        "--prospective-policy",
        action="store_true",
        help="apply the authenticated five-field unit-conflict mask",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = run_exploratory_stage2(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        clinical_project_root=args.clinical_project_root,
        preprocessor_path=args.preprocessor,
        checkpoint_output=args.checkpoint_output,
        summary_output=args.summary_output,
        failure_output=args.failure_output,
        grid_protocol=args.grid_protocol,
        prospective_policy=args.prospective_policy,
    )
    print(
        json.dumps(
            {
                "event": "stage2_completed",
                "runtime_seconds": summary["runtime_seconds"],
                "selected_beta": summary["selection"]["selected_beta"],
                "selected_group_shrinkage_rate": summary["selection"][
                    "selected_group_shrinkage_rate"
                ],
                "selected_step": summary["selection"]["selected_step"],
                "checkpoint": summary["checkpoint"],
                "summary_output": str(args.summary_output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "_lower_bound_expansion_config",
    "_lower_bound_expansion_v2_config",
    "_frozen_selected_config",
    "run_exploratory_stage2",
]
