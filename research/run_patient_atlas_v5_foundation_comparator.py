"""Run one frozen public foundation comparator on the canonical disease folds."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import io
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from eval_soft_patient_atlas import (
    EvaluationCohort,
    _patient_json_hash,
    assert_aggregate_only_payload,
)
from patient_atlas_aireadi_internal_crossfit import _evaluation_features
from patient_atlas_disease_folds import make_disease_fold_map, validate_disease_fold_policy
from patient_atlas_disease_readout import DiseaseReadoutConfig
from patient_atlas_disease_targets import load_development_disease_targets
from patient_atlas_disease_universal import make_inner_fold_ids, subset_targets
from patient_atlas_disease_utility import target_safe_feature_view
from patient_atlas_foundation_adapters import (
    DINO_EMBEDDING_DIMENSION,
    DINO_INPUT_SIZE,
    LABRADOR_EMBEDDING_DIMENSION,
    build_labrador_inputs,
    encode_dinov3_pixels,
    encode_labrador_in_memory,
    load_dinov3_model,
)
from patient_atlas_foundation_comparators import (
    FOUNDATION_COMPARATORS,
    summarize_foundation_losses,
    summarize_foundation_readouts,
    validate_foundation_comparator_protocol,
)
from patient_atlas_foundation_readout import (
    FOUNDATION_READOUT_OPTIMIZER,
    evaluate_nested_foundation_logistic_readout,
)
from patient_atlas_prospective_policy import apply_prospective_policy
from patient_atlas_real_data import _normalized_retinal_path, load_exploratory_raw_cohort
from run_patient_atlas_exploratory_stage2 import _redacted_error, _write_exclusive_json
from run_patient_atlas_v5_disease_target_audit import _contains_exact_patient_identity


SCHEMA_VERSION = "patient-atlas-v5-foundation-comparator-run-v1"


def _safe_count(value: int) -> int | str:
    value = int(value)
    return value if value == 0 or value >= 10 else "<10"


def _enumerate_selected_cfp(
    *, dataset_root: Path, patient_ids: Sequence[str]
) -> tuple[list[Path], np.ndarray]:
    manifest = pd.read_csv(
        dataset_root / "retinal_photography" / "manifest.tsv",
        sep="\t",
        usecols=("person_id", "filepath"),
    )
    manifest["normalized_path"] = manifest["filepath"].astype(str).map(
        _normalized_retinal_path
    )
    manifest = manifest.loc[
        manifest["normalized_path"].str.contains(
            r"(?:^|/)cfp(?:/|$)", regex=True, na=False
        )
    ].copy()
    manifest["basename"] = manifest["normalized_path"].map(
        lambda value: Path(value).name
    )
    if not manifest["basename"].is_unique:
        raise ValueError("CFP manifest basenames are not unique")
    by_basename = manifest.set_index("basename", verify_integrity=True)
    pattern = str(dataset_root / "retinal_photography" / "cfp" / "**" / "*.dcm")
    physical = [Path(value) for value in sorted(glob.glob(pattern, recursive=True))]
    physical_basenames = [value.name for value in physical]
    if (
        len(physical_basenames) != len(set(physical_basenames))
        or set(physical_basenames) != set(by_basename.index.astype(str))
    ):
        raise ValueError("physical CFP files and manifest differ")
    patient_row = {str(value): index for index, value in enumerate(patient_ids)}
    selected_paths: list[Path] = []
    selected_rows: list[int] = []
    for path in physical:
        person = str(by_basename.loc[path.name, "person_id"])
        row = patient_row.get(person)
        if row is not None:
            selected_paths.append(path)
            selected_rows.append(row)
    if not selected_paths:
        raise ValueError("no allowed train/validation CFP images were found")
    return selected_paths, np.asarray(selected_rows, dtype=np.int64)


def _decode_cfp(path: Path) -> np.ndarray:
    import pydicom
    from pydicom.encaps import generate_pixel_data_frame
    from PIL import Image

    dataset = pydicom.dcmread(str(path))
    if not dataset.file_meta.TransferSyntaxUID.is_encapsulated:
        pixels = dataset.pixel_array
        image = Image.fromarray(
            pixels if pixels.ndim == 3 else np.stack([pixels] * 3, axis=-1)
        ).convert("RGB")
    else:
        image = Image.open(io.BytesIO(next(generate_pixel_data_frame(dataset.PixelData))))
        if image.format == "JPEG2000":
            reduction = 0
            while min(image.size) / 2 ** (reduction + 1) >= DINO_INPUT_SIZE:
                reduction += 1
            image.reduce = reduction
            image.load()
        else:
            image.draft("RGB", (DINO_INPUT_SIZE, DINO_INPUT_SIZE))
        image = image.convert("RGB")
    resized = image.resize(
        (DINO_INPUT_SIZE, DINO_INPUT_SIZE), Image.Resampling.BICUBIC
    )
    values = np.asarray(resized, dtype=np.uint8)
    if values.shape != (DINO_INPUT_SIZE, DINO_INPUT_SIZE, 3):
        raise ValueError("decoded CFP shape differs")
    return values


def _encode_dinov3_cohort(
    *,
    dataset_root: Path,
    patient_ids: Sequence[str],
    artifact: Mapping[str, Any],
    batch_size: int = 32,
    decode_threads: int = 8,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    paths, patient_rows = _enumerate_selected_cfp(
        dataset_root=dataset_root, patient_ids=patient_ids
    )
    model = load_dinov3_model(
        checkpoint_path=artifact["checkpoint"],
        checkpoint_sha256=artifact["checkpoint_sha256"],
        device="cpu",
    )
    totals = np.zeros(
        (len(patient_ids), DINO_EMBEDDING_DIMENSION), dtype=np.float64
    )
    counts = np.zeros(len(patient_ids), dtype=np.int64)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=decode_threads) as executor:
        for start in range(0, len(paths), batch_size):
            stop = min(start + batch_size, len(paths))
            try:
                pixels = np.stack(
                    list(executor.map(_decode_cfp, paths[start:stop])), axis=0
                )
            except Exception as error:
                raise ValueError("one selected CFP image could not be decoded") from error
            embedded = encode_dinov3_pixels(model, pixels, device="cpu")
            rows = patient_rows[start:stop]
            np.add.at(totals, rows, embedded.astype(np.float64))
            np.add.at(counts, rows, 1)
            if stop == len(paths) or stop % (batch_size * 100) == 0:
                print(
                    json.dumps(
                        {
                            "event": "dinov3_encoding_progress",
                            "images_completed": stop,
                            "images_total": len(paths),
                            "patient_details_emitted": False,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    present = counts > 0
    pooled = np.zeros_like(totals, dtype=np.float32)
    pooled[present] = (totals[present] / counts[present, None]).astype(np.float32)
    if not np.isfinite(pooled).all():
        raise ValueError("DINOv3 patient pooling is non-finite")
    return pooled, {
        "embedding_dimension": DINO_EMBEDDING_DIMENSION,
        "selected_image_count": len(paths),
        "patients_with_retinal_evidence_count": _safe_count(int(present.sum())),
        "patients_without_retinal_evidence_count": _safe_count(int((~present).sum())),
        "patient_pooling": "unweighted_mean_of_valid_images",
        "runtime_seconds": time.perf_counter() - started,
        "patient_embeddings_written_to_disk": False,
    }


def _encode_labrador_cohort(
    *,
    project_root: Path,
    safe_view: Any,
    ordered_feature_names: Sequence[str],
    artifact: Mapping[str, Any],
) -> tuple[np.ndarray, Mapping[str, Any]]:
    started = time.perf_counter()
    inputs = build_labrador_inputs(
        clinical_values=np.asarray(safe_view.features["clinical_values"]),
        clinical_observed_mask=np.asarray(
            safe_view.features["clinical_observed_mask"]
        ),
        clinical_eligible_mask=np.asarray(
            safe_view.features["clinical_policy_eligible_mask"]
        ),
        ordered_feature_names=ordered_feature_names,
        codebook_path=artifact["codebook"],
        codebook_sha256=artifact["codebook_sha256"],
        ecdf_path=artifact["ecdf"],
        ecdf_sha256=artifact["ecdf_sha256"],
    )
    embedded = encode_labrador_in_memory(
        inputs,
        python_executable=artifact["tensorflow_python"],
        worker_path=project_root / "patient_atlas_labrador_worker.py",
        model_root=artifact["model_root"],
        saved_model_sha256=artifact["saved_model_sha256"],
        variables_data_sha256=artifact["variables_data_sha256"],
        variables_index_sha256=artifact["variables_index_sha256"],
    )
    token_counts = (inputs.categorical > 0).sum(axis=1)
    return embedded, {
        "embedding_dimension": LABRADOR_EMBEDDING_DIMENSION,
        "mapped_feature_count": len(inputs.mapped_features),
        "mapped_feature_names": list(inputs.mapped_features),
        "patients_with_visible_mapped_labs_count": _safe_count(
            int((token_counts > 0).sum())
        ),
        "patients_without_visible_mapped_labs_count": _safe_count(
            int((token_counts == 0).sum())
        ),
        "sequence_length": int(inputs.categorical.shape[1]),
        "patient_pooling": "mean_hidden_state_over_observed_lab_tokens",
        "runtime_seconds": time.perf_counter() - started,
        "patient_embeddings_written_to_disk": False,
    }


def _with_age(embedding: np.ndarray, age: np.ndarray) -> np.ndarray:
    values = np.concatenate(
        [np.asarray(embedding, dtype=np.float64), np.asarray(age, dtype=np.float64)],
        axis=1,
    )
    if not np.isfinite(values).all():
        raise ValueError("foundation comparator design is non-finite")
    return values


def run_foundation_comparator(
    *,
    comparator: str,
    project_root: str | Path,
    dataset_root: str | Path,
    clinical_project_root: str | Path,
    output_path: str | Path,
    failure_path: str | Path,
) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    dataset = Path(dataset_root).resolve()
    output = Path(output_path).resolve()
    failure = Path(failure_path).resolve()
    if output.exists() or failure.exists():
        raise FileExistsError("foundation comparator outputs must be new")
    patient_ids: tuple[str, ...] = ()
    targets_loaded = False
    representation_encoded = False
    scores_computed = False
    started = time.perf_counter()
    try:
        protocol, protocol_sha256, registry, endpoints = (
            validate_foundation_comparator_protocol(root, comparator=comparator)
        )
        fold_policy, fold_policy_sha256 = validate_disease_fold_policy(root)
        raw_cohort = load_exploratory_raw_cohort(
            project_root=root,
            dataset_root=dataset,
            clinical_project_root=clinical_project_root,
        )
        if set(raw_cohort.split_labels) - {"train", "val"}:
            raise PermissionError("official test patient entered foundation run")
        patient_ids = raw_cohort.patient_ids
        targets, _ = load_development_disease_targets(
            project_root=root, dataset_root=dataset, cohort=raw_cohort
        )
        targets_loaded = True
        feature_cohort = apply_prospective_policy(raw_cohort, project_root=root)
        outer_map = make_disease_fold_map(
            patient_ids=patient_ids,
            site_ids=raw_cohort.site_ids,
            targets=targets,
            policy=fold_policy,
        )
        if outer_map.assignment_sha256 != protocol["outer_fold_assignment_sha256"]:
            raise ValueError("foundation outer folds differ")
        outer_assignment = outer_map.assignments_for(patient_ids)
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
        all_rows = np.arange(len(patient_ids), dtype=np.int64)
        safe_view = target_safe_feature_view(
            evaluation_cohort.feature_view(all_rows),
            registry=registry,
            profile_id="circularity18_universal",
            ordered_feature_names=feature_cohort.feature_names,
        )
        artifact = protocol["comparator_artifacts"][comparator]
        if comparator == "dinov3_generic":
            embedding, representation_summary = _encode_dinov3_cohort(
                dataset_root=dataset,
                patient_ids=patient_ids,
                artifact=artifact,
            )
            group_name = "retinal_foundation"
        elif comparator == "labrador":
            embedding, representation_summary = _encode_labrador_cohort(
                project_root=root,
                safe_view=safe_view,
                ordered_feature_names=feature_cohort.feature_names,
                artifact=artifact,
            )
            group_name = "clinical_foundation"
        else:  # pragma: no cover - validated above
            raise ValueError("unknown foundation comparator")
        representation_encoded = True
        design = _with_age(embedding, evaluation_cohort.demographics)
        feature_groups = (group_name,) * embedding.shape[1] + ("age",)
        endpoint_indices = [
            targets.columns.index(endpoint.target_column) for endpoint in endpoints
        ]
        losses = np.full((len(patient_ids), len(endpoints)), np.nan, dtype=np.float64)
        results: dict[str, list[Any]] = {
            endpoint.target_column: [] for endpoint in endpoints
        }
        fold_records: list[dict[str, Any]] = []
        config = DiseaseReadoutConfig(
            penalty_grid=tuple(float(value) for value in protocol["readout"]["penalty_grid"]),
            minimum_disclosable_cell_count=10,
        )
        for outer_fold in range(outer_map.n_folds):
            test_indices = np.flatnonzero(outer_assignment == outer_fold)
            train_indices = np.flatnonzero(outer_assignment != outer_fold)
            train_ids = tuple(patient_ids[index] for index in train_indices)
            test_ids = tuple(patient_ids[index] for index in test_indices)
            inner_targets = subset_targets(targets, train_indices, train_ids)
            inner_folds, inner_hash = make_inner_fold_ids(
                outer_fold=outer_fold,
                patient_ids=train_ids,
                site_ids=tuple(raw_cohort.site_ids[index] for index in train_indices),
                targets=inner_targets,
                outer_fold_policy=fold_policy,
            )
            for endpoint_position, target_index in enumerate(endpoint_indices):
                endpoint = endpoints[endpoint_position]
                sink = np.empty(len(test_indices), dtype=np.float64)
                result = evaluate_nested_foundation_logistic_readout(
                    x_train=design[train_indices],
                    y_train=targets.values[train_indices, target_index],
                    train_eligible=targets.observed_mask[train_indices, target_index],
                    inner_fold_ids=inner_folds,
                    x_test=design[test_indices],
                    y_test=targets.values[test_indices, target_index],
                    test_eligible=targets.observed_mask[test_indices, target_index],
                    feature_groups=feature_groups,
                    unpenalized_groups=("age",),
                    test_patient_id_hash=_patient_json_hash(test_ids),
                    config=config,
                    _private_test_primary_loss_out=sink,
                )
                losses[test_indices, endpoint_position] = sink
                results[endpoint.target_column].append(result)
            scores_computed = True
            fold_records.append(
                {
                    "fold": outer_fold,
                    "outer_train_count": len(train_indices),
                    "outer_test_count": len(test_indices),
                    "outer_test_patient_set_sha256": _patient_json_hash(test_ids),
                    "inner_assignment_sha256": inner_hash,
                    "representation_is_one_shared_frozen_encoder": True,
                    "contains_patient_rows_predictions_losses_or_coordinates": False,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "foundation_scoring_fold_completed",
                        "comparator": comparator,
                        "fold": outer_fold,
                        "patient_details_emitted": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        summary = summarize_foundation_losses(
            comparator=comparator,
            losses=losses,
            endpoints=endpoints,
            bootstrap_samples=2000,
            bootstrap_seed=int(protocol["uncertainty"]["bootstrap_seed"]),
            confidence_level=0.95,
        )
        readout_summary = summarize_foundation_readouts(results)
        v2_result = json.loads(
            (root / protocol["bindings"]["v2_grouped_result"]["file"]).read_text()
        )
        v2_score = float(
            v2_result["primary_breadth_result"]["arms"]["both_atlas"]
            ["organ_family_balanced_mean_log_loss"]
        )
        comparator_score = float(summary["organ_family_balanced_mean_log_loss"])
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "foundation_comparator_internal_disease_breadth_complete",
            "comparator": comparator,
            "runtime_seconds": time.perf_counter() - started,
            "scope": {
                "patient_count": len(patient_ids),
                "endpoint_count": len(endpoints),
                "universal_exclusion_profile": "circularity18_universal",
                "official_train_validation_only": True,
            },
            "representation_summary": representation_summary,
            "outer_fold_assignment_sha256": outer_map.assignment_sha256,
            "outer_folds": fold_records,
            "comparator_result": summary,
            "readout_summary": readout_summary,
            "descriptive_comparison_to_v2_both": {
                "v2_both_score": v2_score,
                "foundation_comparator_score": comparator_score,
                "comparator_minus_v2_both_score": comparator_score - v2_score,
                "positive_favors_v2_both": True,
                "paired_inference_available": False,
                "reason": "V2 patient losses were deliberately not serialized",
            },
            "decision": {
                "selected_foundation_comparator_complete": True,
                "mandatory_external_foundation_comparator_suite_complete": False,
                "full_foundation_model_breadth_claim_allowed": False,
                "canonical_patient_atlas_embedding_changed": False,
            },
            "protocol": {
                "file": "PATIENT_ATLAS_V5_FOUNDATION_COMPARATORS_PROTOCOL_V1.json",
                "sha256": protocol_sha256,
                "fold_policy_sha256": fold_policy_sha256,
                "optimizer": FOUNDATION_READOUT_OPTIMIZER,
            },
            "privacy": {
                "patient_derived_processing": "local_only",
                "aggregate_only": True,
                "official_test_images_or_targets_loaded": False,
                "patient_embeddings_written_to_disk": False,
                "patient_rows_identifiers_targets_predictions_losses_embeddings_or_coordinates_emitted": False,
            },
            "claim_limits": {
                "internal_same_cohort_comparison_only": True,
                "pretraining_overlap_resolved": comparator != "dinov3_generic",
                "paired_superiority_inference_vs_v2_available": False,
                "external_validation_completed": False,
                "clinical_benefit_claim_allowed": False,
            },
        }
        assert_aggregate_only_payload(report, forbidden_patient_ids=patient_ids)
        if _contains_exact_patient_identity(report, patient_ids):
            raise RuntimeError("foundation report contains a patient identity")
        json.dumps(report, sort_keys=True, allow_nan=False)
        _write_exclusive_json(output, report)
        return report
    except BaseException as error:
        _write_exclusive_json(
            failure,
            {
                "schema_version": "patient-atlas-v5-foundation-comparator-failure-v1",
                "comparator": comparator,
                "terminal_for_attempt": True,
                "same_protocol_technical_retry_allowed": True,
                "error_type": type(error).__name__,
                "error_message": _redacted_error(error, patient_ids),
                "runtime_seconds": time.perf_counter() - started,
                "targets_loaded": targets_loaded,
                "representation_encoded": representation_encoded,
                "scores_computed": scores_computed,
                "official_test_images_or_targets_loaded": False,
                "patient_rows_identifiers_targets_predictions_losses_embeddings_or_coordinates_emitted": False,
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparator", choices=FOUNDATION_COMPARATORS, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clinical-project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_foundation_comparator(
            comparator=args.comparator,
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
                    "event": "patient_atlas_v5_foundation_comparator_failed",
                    "comparator": args.comparator,
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
                "event": "patient_atlas_v5_foundation_comparator_completed",
                "comparator": args.comparator,
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


__all__ = ["SCHEMA_VERSION", "run_foundation_comparator"]
