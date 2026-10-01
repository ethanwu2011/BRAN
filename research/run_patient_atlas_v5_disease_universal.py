"""Run the first target-safe V5 universal-mask disease breadth evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from eval_soft_patient_atlas import (
    ARM_BLOOD,
    ARM_BOTH,
    ARM_EYE,
    BASE_STRATUM,
    EvaluationCohort,
    RepresentationFitRequest,
    _partition_outer_train,
    _patient_hash,
    _patient_json_hash,
    assert_aggregate_only_payload,
    assert_same_fold_basis,
)
from patient_atlas_aireadi_internal_crossfit import (
    _evaluation_features,
    _training_from_protocol,
)
from patient_atlas_aireadi_v5_evaluation import (
    AIReadIV5FoldRepresentationFactory,
    validate_v5_internal_evaluation_protocol,
)
from patient_atlas_disease_folds import (
    make_disease_fold_map,
    validate_disease_fold_policy,
)
from patient_atlas_disease_readout import (
    DiseaseReadoutConfig,
    evaluate_nested_readout,
)
from patient_atlas_disease_targets import load_development_disease_targets
from patient_atlas_disease_universal import (
    ATLAS_ARMS,
    make_inner_fold_ids,
    subset_targets,
    summarize_breadth_losses,
    summarize_readout_results,
    validate_universal_protocol,
)
from patient_atlas_disease_utility import TargetSafeFoldRepresentationFactory
from patient_atlas_prospective_policy import apply_prospective_policy
from patient_atlas_real_data import load_exploratory_raw_cohort
from run_patient_atlas_exploratory_stage2 import _redacted_error, _write_exclusive_json
from run_patient_atlas_v5_disease_target_audit import _contains_exact_patient_identity


SCHEMA_VERSION = "patient-atlas-v5-disease-universal-run-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _design_with_age(coordinates: np.ndarray, demographics: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64)
    age = np.asarray(demographics, dtype=np.float64)
    if values.ndim != 2 or age.ndim != 2 or age.shape != (values.shape[0], 1):
        raise ValueError("disease readout coordinates and age context differ")
    result = np.concatenate([values, age], axis=1)
    if not np.isfinite(result).all():
        raise ValueError("disease readout design is non-finite")
    return result


def _safe_progress_payload(event: Mapping[str, Any]) -> dict[str, Any] | None:
    name = str(event.get("event", ""))
    if name not in {"outer_fold_fit_started", "outer_fold_fit_completed"}:
        return None
    allowed = {
        "event",
        "fold_key",
        "outer_train_count",
        "outer_test_count",
        "fit_count",
        "validation_count",
        "calibration_count",
        "selected_step",
        "best_balanced_proper_score",
        "runtime_seconds",
    }
    return {key: event[key] for key in sorted(allowed & set(event))}


def run_universal_disease_benchmark(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    clinical_project_root: str | Path,
    output_path: str | Path,
    failure_path: str | Path,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_path).resolve()
    failure = Path(failure_path).resolve()
    if output.exists() or failure.exists():
        raise FileExistsError("universal disease outputs must be new")
    patient_ids: tuple[str, ...] = ()
    targets_loaded = False
    disease_scores_computed = False
    started = time.perf_counter()
    try:
        protocol, protocol_sha256, registry, endpoints = validate_universal_protocol(root)
        fold_policy, fold_policy_sha256 = validate_disease_fold_policy(root)
        if fold_policy_sha256 != protocol["bindings"]["fold_policy"]["sha256"]:
            raise ValueError("universal and canonical fold policies differ")
        representation_path = root / protocol["bindings"]["representation_protocol"]["file"]
        representation_protocol = validate_v5_internal_evaluation_protocol(
            root, representation_path
        )

        raw_cohort = load_exploratory_raw_cohort(
            project_root=root,
            dataset_root=dataset_root,
            clinical_project_root=clinical_project_root,
        )
        if set(raw_cohort.split_labels) - {"train", "val"}:
            raise PermissionError("official test patient entered universal disease run")
        patient_ids = raw_cohort.patient_ids
        targets, _ = load_development_disease_targets(
            project_root=root,
            dataset_root=dataset_root,
            cohort=raw_cohort,
        )
        targets_loaded = True
        feature_cohort = apply_prospective_policy(raw_cohort, project_root=root)
        if feature_cohort.patient_ids != raw_cohort.patient_ids:
            raise ValueError("prospective clinical policy changed patient order")

        outer_map = make_disease_fold_map(
            patient_ids=raw_cohort.patient_ids,
            site_ids=raw_cohort.site_ids,
            targets=targets,
            policy=fold_policy,
        )
        if outer_map.assignment_sha256 != protocol["outer_fold_assignment_sha256"]:
            raise ValueError("rebuilt universal outer-fold assignment differs")
        outer_assignment = outer_map.assignments_for(raw_cohort.patient_ids)

        evaluation_cohort = EvaluationCohort(
            patient_ids=feature_cohort.patient_ids,
            site_ids=feature_cohort.site_ids,
            features=_evaluation_features(feature_cohort),
            demographics=feature_cohort.ages[:, None].astype(np.float64, copy=False),
            demographic_mask=feature_cohort.age_observed_mask[:, None].copy(),
            targets=np.asarray(targets.values, dtype=np.float64),
            target_ids=targets.columns,
            target_mask=np.asarray(targets.observed_mask, dtype=bool),
            primary_patient_mask=np.ones(len(patient_ids), dtype=bool),
        )

        def representation_progress(event: Mapping[str, Any]) -> None:
            safe = _safe_progress_payload(event)
            if safe is not None and progress_callback is not None:
                progress_callback(safe)

        base_factory = AIReadIV5FoldRepresentationFactory(
            project_root=root,
            ordered_feature_names=feature_cohort.feature_names,
            source_policy_sha256=feature_cohort.source_policy_sha256,
            source_hashes=feature_cohort.source_hashes,
            base_training_seed=int(protocol["training"]["base_training_seed"]),
            training=_training_from_protocol(representation_protocol),
            progress_callback=representation_progress,
        )
        safe_factory = TargetSafeFoldRepresentationFactory(
            base_factory,
            registry=registry,
            registry_sha256=protocol["bindings"]["disease_registry"]["sha256"],
            profile_id=str(protocol["scope"]["universal_exclusion_profile"]),
            ordered_feature_names=feature_cohort.feature_names,
        )
        readout_config = DiseaseReadoutConfig(
            penalty_grid=tuple(float(value) for value in protocol["readout"]["penalty_grid"]),
            minimum_disclosable_cell_count=10,
        )
        endpoint_indices = [targets.columns.index(endpoint.target_column) for endpoint in endpoints]
        losses_by_arm = {
            arm: np.full(
                (len(patient_ids), len(endpoints)), np.nan, dtype=np.float64
            )
            for arm in ATLAS_ARMS
        }
        readout_results: dict[str, dict[str, list[Any]]] = {
            arm: {endpoint.target_column: [] for endpoint in endpoints}
            for arm in ATLAS_ARMS
        }
        fold_records: list[dict[str, Any]] = []

        for outer_fold in range(outer_map.n_folds):
            test_indices = np.flatnonzero(outer_assignment == outer_fold)
            train_indices = np.flatnonzero(outer_assignment != outer_fold)
            train_view = evaluation_cohort.feature_view(train_indices)
            test_view = evaluation_cohort.feature_view(test_indices)
            fold_key = f"disease-universal-seed-1701-outer-{outer_fold}"
            fit_rel, validation_rel, calibration_rel = _partition_outer_train(
                train_view.patient_ids,
                train_view.site_ids,
                fractions=(0.7, 0.15, 0.15),
                salt=f"patient-atlas-v5-disease-representation-phases-v1:{fold_key}",
            )
            request = RepresentationFitRequest(
                fold_key=fold_key,
                fit=evaluation_cohort.feature_view(train_indices[fit_rel]),
                validation=evaluation_cohort.feature_view(train_indices[validation_rel]),
                calibration=evaluation_cohort.feature_view(train_indices[calibration_rel]),
                expected_outer_train_patient_hash=_patient_hash(train_view.patient_ids),
                expected_outer_test_patient_id_hash=_patient_json_hash(test_view.patient_ids),
                expected_outer_test_patient_count=test_view.size,
            )
            fitted = safe_factory.fit(request)
            inner_targets = subset_targets(targets, train_indices, train_view.patient_ids)
            inner_fold_ids, inner_assignment_sha256 = make_inner_fold_ids(
                outer_fold=outer_fold,
                patient_ids=train_view.patient_ids,
                site_ids=train_view.site_ids,
                targets=inner_targets,
                outer_fold_policy=fold_policy,
            )

            designs_train: dict[str, np.ndarray] = {}
            designs_test: dict[str, np.ndarray] = {}
            basis_token: str | None = None
            for arm in ATLAS_ARMS:
                train_coordinates = fitted.transform(
                    train_view, arm=arm, stratum=BASE_STRATUM
                )
                test_coordinates = fitted.transform(
                    test_view, arm=arm, stratum=BASE_STRATUM
                )
                assert_same_fold_basis(train_coordinates, test_coordinates)
                if basis_token is None:
                    basis_token = train_coordinates.basis_token
                elif train_coordinates.basis_token != basis_token:
                    raise ValueError("universal availability arms do not share one basis")
                designs_train[arm] = _design_with_age(
                    train_coordinates.values, train_view.demographics
                )
                designs_test[arm] = _design_with_age(
                    test_coordinates.values, test_view.demographics
                )

            for endpoint_position, target_index in enumerate(endpoint_indices):
                target = endpoints[endpoint_position].target_column
                train_y = targets.values[train_indices, target_index]
                test_y = targets.values[test_indices, target_index]
                train_eligible = targets.observed_mask[train_indices, target_index]
                test_eligible = targets.observed_mask[test_indices, target_index]
                for arm in ATLAS_ARMS:
                    private_loss = np.empty(len(test_indices), dtype=np.float64)
                    result = evaluate_nested_readout(
                        task="binary",
                        x_train=designs_train[arm],
                        y_train=train_y,
                        train_eligible=train_eligible,
                        inner_fold_ids=inner_fold_ids,
                        x_test=designs_test[arm],
                        y_test=test_y,
                        test_eligible=test_eligible,
                        test_patient_id_hash=_patient_json_hash(test_view.patient_ids),
                        config=readout_config,
                        _private_test_primary_loss_out=private_loss,
                    )
                    losses_by_arm[arm][test_indices, endpoint_position] = private_loss
                    readout_results[arm][target].append(result)
            disease_scores_computed = True
            fold_records.append(
                {
                    "fold": outer_fold,
                    "fold_key": fold_key,
                    "outer_train_count": len(train_indices),
                    "outer_test_count": len(test_indices),
                    "outer_test_patient_set_sha256": _patient_json_hash(
                        test_view.patient_ids
                    ),
                    "inner_assignment_sha256": inner_assignment_sha256,
                    "endpoint_count_scored": len(endpoints),
                    "availability_arm_count_scored": len(ATLAS_ARMS),
                    "contains_patient_rows_predictions_losses_or_coordinates": False,
                }
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "universal_disease_outer_fold_scored",
                        "fold": outer_fold,
                        "outer_test_count": len(test_indices),
                        "endpoint_count": len(endpoints),
                        "arm_count": len(ATLAS_ARMS),
                    }
                )
            del fitted, designs_train, designs_test, inner_targets
            gc.collect()

        for endpoint_position, target_index in enumerate(endpoint_indices):
            expected = targets.observed_mask[:, target_index]
            for arm in ATLAS_ARMS:
                if not np.array_equal(
                    np.isfinite(losses_by_arm[arm][:, endpoint_position]), expected
                ):
                    raise RuntimeError("universal cross-fitted loss coverage differs from targets")
        breadth = summarize_breadth_losses(
            losses_by_arm=losses_by_arm,
            endpoints=endpoints,
            bootstrap_samples=int(protocol["aggregation"]["bootstrap_samples"]),
            bootstrap_seed=int(protocol["aggregation"]["bootstrap_seed"]),
            confidence_level=float(protocol["aggregation"]["confidence_level"]),
        )
        readout_summary = summarize_readout_results(readout_results)
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "universal_mask_internal_disease_breadth_attempt_complete",
            "runtime_seconds": time.perf_counter() - started,
            "scope": {
                "cohort": "AI-READI official train plus validation only",
                "patient_count": len(patient_ids),
                "endpoint_count": len(endpoints),
                "endpoint_columns": [endpoint.target_column for endpoint in endpoints],
                "blocked_breadth_endpoint_columns": protocol["scope"][
                    "blocked_target_columns"
                ],
                "availability_arms": list(ATLAS_ARMS),
                "universal_exclusion_profile": protocol["scope"][
                    "universal_exclusion_profile"
                ],
            },
            "outer_fold_assignment_sha256": outer_map.assignment_sha256,
            "outer_folds": fold_records,
            "representation_fits": base_factory.fold_summaries,
            "target_safety_artifact_contracts": safe_factory.artifact_contracts,
            "primary_breadth_result": breadth,
            "readout_summary": readout_summary,
            "decision": {
                "paired_information_gate_passed": breadth[
                    "paired_information_gate_passed"
                ],
                "both_beats_eye_with_multiplicity_control": breadth[
                    "paired_contrasts"
                ]["eye_minus_both"]["superiority_passed"],
                "both_beats_clinical_with_multiplicity_control": breadth[
                    "paired_contrasts"
                ]["clinical_minus_both"]["superiority_passed"],
                "mandatory_foundation_comparator_breadth_complete": False,
                "plain_concat_comparison_complete": False,
                "full_foundation_model_breadth_claim_allowed": False,
                "canonical_deployment_embedding_changed_by_this_evaluation": False,
                "next_required_gate": (
                    "add frozen plain-concat and mandatory retinal/clinical comparators on the identical folds"
                ),
            },
            "protocol": {
                "file": protocol["bindings"]["universal_runner"]["file"],
                "protocol_file": "PATIENT_ATLAS_V5_DISEASE_UNIVERSAL_PROTOCOL_V1.json",
                "protocol_sha256": protocol_sha256,
                "fold_policy_sha256": fold_policy_sha256,
            },
            "privacy": {
                "patient_derived_processing": "local_only",
                "patient_rows_identifiers_targets_predictions_losses_embeddings_or_coordinates_emitted": False,
                "aggregate_only": True,
                "small_cells_suppressed": True,
                "official_test_inputs_or_targets_loaded": False,
            },
            "claim_limits": {
                "retrospective_internal_development_only": True,
                "individual_endpoints_exploratory": True,
                "incident_risk_claim_allowed": False,
                "diagnostic_replacement_claim_allowed": False,
                "external_validation_completed": False,
                "clinical_benefit_claim_allowed": False,
            },
        }
        assert_aggregate_only_payload(report, forbidden_patient_ids=patient_ids)
        if _contains_exact_patient_identity(report, patient_ids):
            raise RuntimeError("universal disease report contains a patient identity")
        json.dumps(report, sort_keys=True, allow_nan=False)
        _write_exclusive_json(output, report)
        return report
    except BaseException as error:
        _write_exclusive_json(
            failure,
            {
                "schema_version": "patient-atlas-v5-disease-universal-failure-v1",
                "terminal_for_attempt": True,
                "same_protocol_technical_retry_allowed": True,
                "error_type": type(error).__name__,
                "error_message": _redacted_error(error, patient_ids),
                "runtime_seconds": time.perf_counter() - started,
                "targets_loaded": targets_loaded,
                "disease_scores_computed": disease_scores_computed,
                "official_test_inputs_or_targets_loaded": False,
                "patient_rows_identifiers_targets_predictions_losses_embeddings_or_coordinates_emitted": False,
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clinical-project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    def emit(event: Mapping[str, Any]) -> None:
        print(json.dumps(dict(event), sort_keys=True, allow_nan=False), flush=True)

    try:
        report = run_universal_disease_benchmark(
            project_root=args.project_root,
            dataset_root=args.dataset_root,
            clinical_project_root=args.clinical_project_root,
            output_path=args.output,
            failure_path=args.failure,
            progress_callback=emit,
        )
    except BaseException:
        print(
            json.dumps(
                {
                    "event": "patient_atlas_v5_disease_universal_failed",
                    "details_emitted": False,
                    "failure_artifact_written": args.failure.is_file(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1
    print(
        json.dumps(
            {
                "event": "patient_atlas_v5_disease_universal_completed",
                "status": report["status"],
                "patient_details_emitted": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "run_universal_disease_benchmark"]
