"""Run the frozen nested target-safe V6.2 disease-breadth evaluation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from eval_soft_patient_atlas import (
    ARM_BLOOD,
    ARM_BOTH,
    ARM_CONCAT,
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
    validate_v5_internal_evaluation_protocol,
)
from patient_atlas_aireadi_v6_2_evaluation import (
    AIReadIV62FoldRepresentationFactory,
)
from patient_atlas_disease_folds import make_disease_fold_map, validate_disease_fold_policy
from patient_atlas_disease_group_readout import (
    GroupedDiseaseReadoutResult,
    evaluate_nested_grouped_logistic_readout,
)
from patient_atlas_disease_readout import DiseaseReadoutConfig
from patient_atlas_disease_targets import load_development_disease_targets
from patient_atlas_disease_universal import (
    ATLAS_ARMS,
    make_inner_fold_ids,
    subset_targets,
    summarize_breadth_losses,
)
from patient_atlas_disease_universal_v2 import (
    structured_feature_groups,
    summarize_grouped_readout_results,
    validate_universal_v2_protocol,
)
from patient_atlas_disease_utility import TargetSafeFoldRepresentationFactory
from patient_atlas_prospective_policy import apply_prospective_policy
from patient_atlas_real_data import load_exploratory_raw_cohort
from patient_atlas_tuned_concat import (
    TUNED_EYE_DIMENSIONS,
    TunedConcatReadoutResult,
    evaluate_nested_tuned_concat_logistic_readout,
    summarize_tuned_concat_selections,
    validate_tuned_concat_protocol,
)
from patient_atlas_v6_2_screening import (
    NONINFERIORITY_MARGIN,
    summarize_v6_2_vs_concat,
)
from run_patient_atlas_exploratory_stage2 import (
    _redacted_error,
    _sha256,
    _write_exclusive_json,
)
from run_patient_atlas_v5_disease_target_audit import _contains_exact_patient_identity
from run_patient_atlas_v5_disease_universal import _design_with_age


SCHEMA_VERSION = "patient-atlas-v6-2-screening-run-v1"
PROTOCOL_NAME = "PATIENT_ATLAS_V6_2_SCREENING_PROTOCOL_V1.json"
PROTOCOL_SCHEMA_VERSION = "patient-atlas-v6-2-screening-protocol-v1"


def validate_v6_2_screening_protocol(
    root: Path, path: Path | None = None
) -> tuple[Mapping[str, Any], str, Mapping[str, Any], tuple[Any, ...]]:
    canonical = (root / PROTOCOL_NAME).resolve()
    protocol_path = canonical if path is None else path.resolve()
    if protocol_path != canonical:
        raise ValueError("V6.2 screening protocol must be canonical")
    protocol = json.loads(protocol_path.read_text())
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or protocol.get("status")
        != "frozen_after_v6_2_outcome_free_gate_before_any_v6_2_screening_score"
    ):
        raise ValueError("V6.2 screening protocol is not frozen")
    bindings = protocol.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("V6.2 screening bindings are absent")
    for label, raw in bindings.items():
        if not isinstance(raw, dict) or set(raw) != {"file", "sha256"}:
            raise ValueError(f"malformed V6.2 screening binding: {label}")
        source = root / str(raw["file"])
        if not source.is_file() or _sha256(source) != raw["sha256"]:
            raise ValueError(f"V6.2 screening binding differs: {label}")
    milestone = json.loads(
        (root / bindings["outcome_free_milestone"]["file"]).read_text()
    )
    if (
        milestone.get("decision", {}).get(
            "nested_internal_target_evaluation_unlocked"
        )
        is not True
        or milestone.get("outcome_free_result", {}).get(
            "screening_or_functional_targets_loaded"
        )
        is not False
    ):
        raise ValueError("V6.2 outcome-free milestone differs")
    v2, v2_sha, registry, endpoints = validate_universal_v2_protocol(root)
    tuned, tuned_sha, tuned_registry, tuned_endpoints = validate_tuned_concat_protocol(
        root
    )
    if (
        v2_sha != bindings["v2_grouped_protocol"]["sha256"]
        or tuned_sha != bindings["tuned_concat_protocol"]["sha256"]
        or registry != tuned_registry
        or [endpoint.target_column for endpoint in endpoints]
        != [endpoint.target_column for endpoint in tuned_endpoints]
    ):
        raise ValueError("V6.2 screening parents differ")
    if protocol.get("representation") != {
        "architecture": "retinally_recoverable_probabilistic_patient_atlas_v6_2",
        "latent_partition": {
            "shared": 32,
            "eye_private": 160,
            "clinical_private": 64,
            "total": 256,
        },
        "retinal_recovery_dimension": 64,
        "retinal_recovery_loss_weight": 0.05,
        "canonical_vector_dimension": 289,
        "outcome_coordinate_dimension_excluding_age": 288,
        "all_five_fold_recovery_and_capacity_gates_must_pass_before_any_target_readout": True,
        "same_fold_model_transforms_outer_train_and_outer_test": True,
        "cross_fold_latent_coordinates_pooled": False,
        "interaction_enabled": False,
        "external_blood_anchor_enabled": False,
    }:
        raise ValueError("V6.2 screening representation differs")
    if protocol.get("evaluation") != {
        "outer_folds": 5,
        "inner_readout_folds": 5,
        "outer_fold_assignment_sha256": "c10632ba7b94cbd5571b56ae6889c6c37ea8fe0c7e71f6c8ed9dda6ebb8a0f48",
        "availability_arms": list(ATLAS_ARMS),
        "locked_fusion_reference": "full tuned concat with nested eye dimension and eye/clinical penalties",
        "tuned_eye_dimensions": list(TUNED_EYE_DIMENSIONS),
        "same_target_masks_and_patient_folds_for_all_arms": True,
        "target_defining_fields_physically_erased_before_preprocessing": True,
        "age_appended_and_unpenalized": True,
        "outer_test_used_for_representation_or_readout_selection": False,
    }:
        raise ValueError("V6.2 screening evaluation differs")
    if protocol.get("acceptance") != {
        "both_view_must_simultaneously_beat_eye_only_and_clinical_only": True,
        "single_view_multiplicity": "centered patient-bootstrap max statistic over two contrasts",
        "v6_2_minus_concat_noninferiority_margin": NONINFERIORITY_MARGIN,
        "paired_one_sided_confidence_level": 0.95,
        "v6_2_point_estimate_must_not_be_worse_than_concat": True,
        "promotion_requires_single_view_gate_and_locked_concat_target_recovered": True,
    }:
        raise ValueError("V6.2 screening acceptance differs")
    if protocol.get("aggregation") != {
        "bootstrap_unit": "patient",
        "bootstrap_samples": 10000,
        "bootstrap_seed_single_view": 20260901,
        "bootstrap_seed_concat": 20260902,
        "confidence_level": 0.95,
        "patient_losses_or_bootstrap_draws_serialized": False,
    }:
        raise ValueError("V6.2 screening aggregation differs")
    return protocol, _sha256(protocol_path), registry, endpoints


def run_v6_2_screening(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    clinical_project_root: str | Path,
    output_path: str | Path,
    failure_path: str | Path,
) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_path).resolve()
    failure = Path(failure_path).resolve()
    if output.exists() or failure.exists():
        raise FileExistsError("V6.2 screening outputs must be new")
    patient_ids: tuple[str, ...] = ()
    targets_loaded = False
    scores_computed = False
    started = time.perf_counter()
    try:
        protocol, protocol_sha256, registry, endpoints = (
            validate_v6_2_screening_protocol(root)
        )
        v2_protocol = json.loads(
            (root / protocol["bindings"]["v2_grouped_protocol"]["file"]).read_text()
        )
        tuned_protocol = json.loads(
            (root / protocol["bindings"]["tuned_concat_protocol"]["file"]).read_text()
        )
        tuned_parent = json.loads(
            (root / protocol["bindings"]["tuned_concat_result"]["file"]).read_text()
        )
        representation_protocol = validate_v5_internal_evaluation_protocol(
            root,
            root / protocol["bindings"]["training_template_protocol"]["file"],
        )
        fold_policy, fold_policy_sha256 = validate_disease_fold_policy(root)
        raw_cohort = load_exploratory_raw_cohort(
            project_root=root,
            dataset_root=dataset_root,
            clinical_project_root=clinical_project_root,
        )
        if set(raw_cohort.split_labels) - {"train", "val"}:
            raise PermissionError("official test patient entered V6.2 screening")
        patient_ids = raw_cohort.patient_ids
        targets, _ = load_development_disease_targets(
            project_root=root,
            dataset_root=dataset_root,
            cohort=raw_cohort,
        )
        targets_loaded = True
        feature_cohort = apply_prospective_policy(raw_cohort, project_root=root)
        outer_map = make_disease_fold_map(
            patient_ids=raw_cohort.patient_ids,
            site_ids=raw_cohort.site_ids,
            targets=targets,
            policy=fold_policy,
        )
        if outer_map.assignment_sha256 != protocol["evaluation"][
            "outer_fold_assignment_sha256"
        ]:
            raise ValueError("V6.2 outer folds differ")
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
        base_factory = AIReadIV62FoldRepresentationFactory(
            project_root=root,
            ordered_feature_names=feature_cohort.feature_names,
            source_policy_sha256=feature_cohort.source_policy_sha256,
            source_hashes=feature_cohort.source_hashes,
            base_training_seed=int(
                representation_protocol["representation"]["training"][
                    "base_training_seed"
                ]
            ),
            training=_training_from_protocol(representation_protocol),
        )
        safe_factory = TargetSafeFoldRepresentationFactory(
            base_factory,
            registry=registry,
            registry_sha256=v2_protocol["bindings"]["disease_registry"]["sha256"],
            profile_id="circularity18_universal",
            ordered_feature_names=feature_cohort.feature_names,
        )

        # Complete and gate every target-free representation fit before any readout.
        fitted_folds: list[tuple[int, np.ndarray, np.ndarray, Any]] = []
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
                validation=evaluation_cohort.feature_view(
                    train_indices[validation_rel]
                ),
                calibration=evaluation_cohort.feature_view(
                    train_indices[calibration_rel]
                ),
                expected_outer_train_patient_hash=_patient_hash(
                    train_view.patient_ids
                ),
                expected_outer_test_patient_id_hash=_patient_json_hash(
                    test_view.patient_ids
                ),
                expected_outer_test_patient_count=test_view.size,
            )
            fitted = safe_factory.fit(request)
            summary = base_factory.fold_summaries[-1]
            if (
                summary["retinal_recovery_gate_passed"] is not True
                or summary["capacity_and_prior_reversion_gates_passed"] is not True
            ):
                raise RuntimeError("V6.2 fold gate failed before target readout")
            fitted_folds.append((outer_fold, train_indices, test_indices, fitted))
            print(
                json.dumps(
                    {
                        "event": "v6_2_representation_fold_gated",
                        "fold": outer_fold,
                        "target_scores_computed": False,
                        "patient_details_emitted": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if len(fitted_folds) != 5 or any(
            not item["retinal_recovery_gate_passed"]
            for item in base_factory.fold_summaries
        ):
            raise RuntimeError("not all V6.2 representation folds passed")

        endpoint_indices = [
            targets.columns.index(endpoint.target_column) for endpoint in endpoints
        ]
        losses_by_arm = {
            arm: np.full((len(patient_ids), len(endpoints)), np.nan, dtype=np.float64)
            for arm in ATLAS_ARMS
        }
        concat_losses = np.full_like(losses_by_arm[ARM_BOTH], np.nan)
        v6_results: dict[str, dict[str, list[GroupedDiseaseReadoutResult]]] = {
            arm: {endpoint.target_column: [] for endpoint in endpoints}
            for arm in ATLAS_ARMS
        }
        concat_results: dict[str, list[TunedConcatReadoutResult]] = {
            endpoint.target_column: [] for endpoint in endpoints
        }
        v6_config = DiseaseReadoutConfig(
            penalty_grid=tuple(
                float(value) for value in v2_protocol["readout"]["penalty_grid"]
            ),
            minimum_disclosable_cell_count=10,
        )
        concat_config = DiseaseReadoutConfig(
            penalty_grid=tuple(
                float(value) for value in tuned_protocol["readout"]["penalty_grid"]
            ),
            logistic_max_iterations=int(
                tuned_protocol["readout"]["logistic_max_iterations"]
            ),
            logistic_tolerance=float(tuned_protocol["readout"]["logistic_tolerance"]),
            minimum_disclosable_cell_count=10,
        )
        groups = structured_feature_groups()
        fold_records: list[dict[str, Any]] = []

        for outer_fold, train_indices, test_indices, fitted in fitted_folds:
            train_view = evaluation_cohort.feature_view(train_indices)
            test_view = evaluation_cohort.feature_view(test_indices)
            inner_targets = subset_targets(targets, train_indices, train_view.patient_ids)
            inner_folds, inner_hash = make_inner_fold_ids(
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
                    raise ValueError("V6.2 availability arms differ in basis")
                designs_train[arm] = _design_with_age(
                    train_coordinates.values, train_view.demographics
                )
                designs_test[arm] = _design_with_age(
                    test_coordinates.values, test_view.demographics
                )
            concat_train: dict[int, np.ndarray] = {}
            concat_test: dict[int, np.ndarray] = {}
            concat_groups: dict[int, tuple[str, ...]] = {}
            for dimension in TUNED_EYE_DIMENSIONS:
                train_coordinates = fitted.transform(
                    train_view,
                    arm=ARM_CONCAT,
                    stratum=BASE_STRATUM,
                    concat_eye_dimension=dimension,
                )
                test_coordinates = fitted.transform(
                    test_view,
                    arm=ARM_CONCAT,
                    stratum=BASE_STRATUM,
                    concat_eye_dimension=dimension,
                )
                assert_same_fold_basis(train_coordinates, test_coordinates)
                expected_width = dimension + 118
                if (
                    train_coordinates.values.shape[1] != expected_width
                    or test_coordinates.values.shape[1] != expected_width
                ):
                    raise ValueError("V6.2 concat coordinate width differs")
                concat_train[dimension] = _design_with_age(
                    train_coordinates.values, train_view.demographics
                )
                concat_test[dimension] = _design_with_age(
                    test_coordinates.values, test_view.demographics
                )
                concat_groups[dimension] = (
                    ("eye",) * dimension
                    + ("clinical",) * 118
                    + ("age",)
                )

            for endpoint_position, target_index in enumerate(endpoint_indices):
                target = endpoints[endpoint_position].target_column
                train_y = targets.values[train_indices, target_index]
                test_y = targets.values[test_indices, target_index]
                train_eligible = targets.observed_mask[train_indices, target_index]
                test_eligible = targets.observed_mask[test_indices, target_index]
                for arm in ATLAS_ARMS:
                    sink = np.empty(len(test_indices), dtype=np.float64)
                    result = evaluate_nested_grouped_logistic_readout(
                        x_train=designs_train[arm],
                        y_train=train_y,
                        train_eligible=train_eligible,
                        inner_fold_ids=inner_folds,
                        x_test=designs_test[arm],
                        y_test=test_y,
                        test_eligible=test_eligible,
                        feature_groups=groups,
                        unpenalized_groups=("age",),
                        test_patient_id_hash=_patient_json_hash(
                            test_view.patient_ids
                        ),
                        config=v6_config,
                        maximum_penalty_combinations=int(
                            v2_protocol["readout"]["maximum_penalty_combinations"]
                        ),
                        _private_test_primary_loss_out=sink,
                    )
                    losses_by_arm[arm][test_indices, endpoint_position] = sink
                    v6_results[arm][target].append(result)
                concat_sink = np.empty(len(test_indices), dtype=np.float64)
                concat_result = evaluate_nested_tuned_concat_logistic_readout(
                    x_train_by_dimension=concat_train,
                    x_test_by_dimension=concat_test,
                    feature_groups_by_dimension=concat_groups,
                    y_train=train_y,
                    train_eligible=train_eligible,
                    inner_fold_ids=inner_folds,
                    y_test=test_y,
                    test_eligible=test_eligible,
                    test_patient_id_hash=_patient_json_hash(test_view.patient_ids),
                    config=concat_config,
                    maximum_penalty_combinations=int(
                        tuned_protocol["readout"]["maximum_penalty_combinations"]
                    ),
                    _private_test_primary_loss_out=concat_sink,
                )
                concat_losses[test_indices, endpoint_position] = concat_sink
                concat_results[target].append(concat_result)
            scores_computed = True
            fold_records.append(
                {
                    "fold": outer_fold,
                    "fold_key": base_factory.fold_summaries[outer_fold]["fold_key"],
                    "outer_train_count": len(train_indices),
                    "outer_test_count": len(test_indices),
                    "outer_test_patient_set_sha256": _patient_json_hash(
                        test_view.patient_ids
                    ),
                    "inner_assignment_sha256": inner_hash,
                    "all_representation_gates_passed_before_target_readout": True,
                    "endpoint_count_scored": len(endpoints),
                    "contains_patient_rows_predictions_losses_or_coordinates": False,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "v6_2_screening_fold_scored",
                        "fold": outer_fold,
                        "patient_details_emitted": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del fitted, designs_train, designs_test, concat_train, concat_test
            gc.collect()

        for endpoint_position, target_index in enumerate(endpoint_indices):
            expected = targets.observed_mask[:, target_index]
            for arm in ATLAS_ARMS:
                if not np.array_equal(
                    np.isfinite(losses_by_arm[arm][:, endpoint_position]), expected
                ):
                    raise RuntimeError("V6.2 arm loss coverage differs")
            if not np.array_equal(
                np.isfinite(concat_losses[:, endpoint_position]), expected
            ):
                raise RuntimeError("V6.2 concat loss coverage differs")
        breadth = summarize_breadth_losses(
            losses_by_arm=losses_by_arm,
            endpoints=endpoints,
            bootstrap_samples=int(protocol["aggregation"]["bootstrap_samples"]),
            bootstrap_seed=int(
                protocol["aggregation"]["bootstrap_seed_single_view"]
            ),
            confidence_level=float(protocol["aggregation"]["confidence_level"]),
        )
        fusion = summarize_v6_2_vs_concat(
            v6_2_both_losses=losses_by_arm[ARM_BOTH],
            tuned_concat_losses=concat_losses,
            endpoints=endpoints,
            bootstrap_samples=int(protocol["aggregation"]["bootstrap_samples"]),
            bootstrap_seed=int(protocol["aggregation"]["bootstrap_seed_concat"]),
            confidence_level=float(protocol["aggregation"]["confidence_level"]),
        )
        expected_concat = float(
            tuned_parent["comparator_result"][
                "organ_family_balanced_mean_log_loss"
            ]
        )
        concat_delta = fusion["tuned_concat_score"] - expected_concat
        if abs(concat_delta) > 1e-8:
            raise RuntimeError("V6.2 run failed tuned-concat parent reproduction")
        promoted = bool(
            breadth["paired_information_gate_passed"]
            and fusion["locked_concat_target_recovered"]
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "frozen_v6_2_nested_internal_screening_complete",
            "runtime_seconds": time.perf_counter() - started,
            "scope": {
                "patient_count": len(patient_ids),
                "endpoint_count": len(endpoints),
                "universal_exclusion_profile": "circularity18_universal",
                "official_train_validation_only": True,
                "official_test_inputs_or_targets_loaded": False,
            },
            "outer_fold_assignment_sha256": outer_map.assignment_sha256,
            "outer_folds": fold_records,
            "representation_fits": base_factory.fold_summaries,
            "target_safety_artifact_contracts": safe_factory.artifact_contracts,
            "v6_2_availability_breadth": breadth,
            "v6_2_vs_locked_concat": fusion,
            "tuned_concat_parent_reproduction": {
                "parent_score": expected_concat,
                "rerun_score": fusion["tuned_concat_score"],
                "rerun_minus_parent": concat_delta,
                "gate_passed": True,
            },
            "readout_summary": {
                "v6_2": summarize_grouped_readout_results(v6_results),
                "tuned_concat": summarize_tuned_concat_selections(
                    concat_results
                ),
            },
            "decision": {
                "both_view_simultaneously_beats_both_single_views": breadth[
                    "paired_information_gate_passed"
                ],
                "locked_concat_target_recovered": fusion[
                    "locked_concat_target_recovered"
                ],
                "v6_2_internal_promotion_passed": promoted,
                "freeze_v6_2_as_best_valid_internal_embedding": promoted,
                "external_validation_complete": False,
                "clinical_deployment_ready": False,
            },
            "protocol": {
                "file": PROTOCOL_NAME,
                "sha256": protocol_sha256,
                "fold_policy_sha256": fold_policy_sha256,
            },
            "privacy": {
                "patient_derived_processing": "local_only",
                "aggregate_only": True,
                "patient_rows_identifiers_targets_predictions_losses_embeddings_or_coordinates_emitted": False,
                "official_test_inputs_or_targets_loaded": False,
            },
            "claim_limits": {
                "adaptive_internal_development_only": True,
                "external_validation_completed": False,
                "clinical_benefit_claim_allowed": False,
                "diagnostic_or_deployment_claim_allowed": False,
            },
        }
        assert_aggregate_only_payload(report, forbidden_patient_ids=patient_ids)
        if _contains_exact_patient_identity(report, patient_ids):
            raise RuntimeError("V6.2 screening report contains a patient identity")
        json.dumps(report, sort_keys=True, allow_nan=False)
        _write_exclusive_json(output, report)
        return report
    except BaseException as error:
        _write_exclusive_json(
            failure,
            {
                "schema_version": "patient-atlas-v6-2-screening-failure-v1",
                "terminal_for_attempt": True,
                "same_protocol_technical_retry_allowed": True,
                "error_type": type(error).__name__,
                "error_message": _redacted_error(error, patient_ids),
                "elapsed_seconds": time.perf_counter() - started,
                "targets_loaded": targets_loaded,
                "target_scores_computed": scores_computed,
                "official_test_inputs_or_targets_loaded": False,
                "patient_details_emitted": False,
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
    try:
        report = run_v6_2_screening(
            project_root=args.project_root,
            dataset_root=args.dataset_root,
            clinical_project_root=args.clinical_project_root,
            output_path=args.output,
            failure_path=args.failure,
        )
    except BaseException:
        print(
            json.dumps(
                {
                    "event": "patient_atlas_v6_2_screening_failed",
                    "failure_artifact_written": args.failure.is_file(),
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
                "event": "patient_atlas_v6_2_screening_completed",
                "promotion_passed": report["decision"][
                    "v6_2_internal_promotion_passed"
                ],
                "patient_details_emitted": False,
            },
            sort_keys=True,
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
    "run_v6_2_screening",
    "validate_v6_2_screening_protocol",
]
