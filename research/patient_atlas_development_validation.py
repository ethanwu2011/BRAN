"""Local-only official-validation readout for the frozen exploratory atlas.

This module never writes patient vectors, labels, predictions, or identifiers.
It extracts the five frozen non-input functional targets at the authenticated
index visit, encodes one frozen checkpoint under three availability patterns,
and reports aggregate paired validation evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from eval_soft_patient_atlas import (
    TargetManifest,
    _basic_bootstrap_interval,
    _fit_ridge,
    _holm_bootstrap_lower_bounds,
    _percentile_interval,
    _predict_ridge,
    _select_probe,
    _target_aggregate,
    _target_family_score,
    assert_aggregate_only_payload,
    make_canonical_fold_map,
)
from patient_atlas_real_data import _index_linked_rows, _load_visits
from patient_atlas_preflight import PreflightStage, require_preflight
from patient_atlas_stage2 import precision_temperature_state
from train_soft_patient_atlas import AtlasTrainingBatch


PRIMARY_ARMS = ("both", "eye_only", "blood_only")
RIDGE_GRID = (
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1_000.0,
    10_000.0,
)
PAIRED_RIDGE_GRID = (
    0.0,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1_000.0,
    10_000.0,
    100_000.0,
    1_000_000.0,
    10_000_000.0,
)


def _group_subset_candidates(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    groups: tuple[str, ...],
) -> tuple[
    dict[int, tuple[np.ndarray, tuple[str, ...]]],
    dict[int, np.ndarray],
]:
    """Enumerate full and exact group-drop readouts, preferring full on ties."""

    train = np.asarray(x_train, dtype=np.float64)
    validation = np.asarray(x_validation, dtype=np.float64)
    if (
        train.ndim != 2
        or validation.ndim != 2
        or train.shape[1] != validation.shape[1]
        or len(groups) != train.shape[1]
        or not np.isfinite(train).all()
        or not np.isfinite(validation).all()
    ):
        raise ValueError("group-subset readout coordinates are malformed")
    unique_groups = tuple(sorted(set(groups)))
    if len(unique_groups) < 2:
        raise ValueError("group-subset tuning requires at least two groups")
    train_candidates: dict[int, tuple[np.ndarray, tuple[str, ...]]] = {}
    validation_candidates: dict[int, np.ndarray] = {}
    candidate_id = 0
    for keep_count in range(len(unique_groups), 0, -1):
        for kept in itertools.combinations(unique_groups, keep_count):
            kept_set = set(kept)
            column_mask = np.asarray(
                [group in kept_set for group in groups], dtype=np.bool_
            )
            selected_groups = tuple(
                group for group, selected in zip(groups, column_mask) if selected
            )
            train_candidates[candidate_id] = (
                train[:, column_mask],
                selected_groups,
            )
            validation_candidates[candidate_id] = validation[:, column_mask]
            candidate_id += 1
    return train_candidates, validation_candidates


@dataclass(frozen=True, repr=False)
class FunctionalTargets:
    """Sensitive in-memory target matrix aligned to a supplied patient order."""

    target_ids: tuple[str, ...]
    values: np.ndarray
    observed_mask: np.ndarray

    def __post_init__(self) -> None:
        if not self.target_ids or len(self.target_ids) != len(set(self.target_ids)):
            raise ValueError("functional target IDs must be nonempty and unique")
        if self.values.ndim != 2 or self.values.shape[1] != len(self.target_ids):
            raise ValueError("functional target values have the wrong shape")
        if self.observed_mask.shape != self.values.shape:
            raise ValueError("functional target mask is misaligned")
        if self.observed_mask.dtype != np.bool_:
            raise TypeError("functional target mask must be boolean")
        if not np.isfinite(self.values[self.observed_mask]).all():
            raise ValueError("observed functional target values must be finite")

    def __repr__(self) -> str:
        return (
            "FunctionalTargets(aggregate_only=True, "
            f"patients={self.values.shape[0]}, targets={self.values.shape[1]})"
        )

    @property
    def observed_counts(self) -> Mapping[str, int]:
        return {
            target_id: int(self.observed_mask[:, index].sum())
            for index, target_id in enumerate(self.target_ids)
        }


def _target_source_columns(manifest_path: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    payload = json.loads(manifest_path.read_text())
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("target manifest has no executable target declarations")
    result: list[tuple[str, tuple[str, ...]]] = []
    for target in targets:
        target_id = str(target.get("id", ""))
        columns = target.get("columns")
        transform = target.get("transform")
        if not target_id or not isinstance(columns, list) or not columns:
            raise ValueError("target manifest contains a malformed target declaration")
        if transform != "identity":
            raise ValueError("development validation supports identity targets only")
        sources: list[str] = []
        for column in columns:
            column = str(column)
            if not column.startswith("fun_"):
                raise ValueError("functional target columns must use the fun_ namespace")
            sources.append(column.removeprefix("fun_"))
        result.append((target_id, tuple(sources)))
    return tuple(result)


def _load_functional_source_policy(
    source_policy_path: Path,
    manifest_path: Path,
    declarations: tuple[tuple[str, tuple[str, ...]], ...],
) -> Mapping[str, Any]:
    raw_manifest = manifest_path.read_bytes()
    policy = json.loads(source_policy_path.read_text())
    if policy.get("schema_version") != "patient-atlas-functional-source-policy-v1":
        raise ValueError("unsupported functional source policy")
    if policy.get("scope") != "official_train_and_validation_only":
        raise ValueError("functional source policy does not seal the official test")
    if policy.get("target_manifest_sha256") != hashlib.sha256(raw_manifest).hexdigest():
        raise ValueError("functional source policy is not bound to the target manifest")
    repeat = policy.get("within_source_repeat_rule")
    if not isinstance(repeat, Mapping) or repeat.get("aggregation") != "arithmetic_mean":
        raise ValueError("functional source repeat aggregation is not frozen")
    rules = policy.get("target_rules")
    if not isinstance(rules, Mapping) or set(rules) != {
        target_id for target_id, _ in declarations
    }:
        raise ValueError("functional source policy target set differs from the manifest")
    for target_id, sources in declarations:
        rule = rules[target_id]
        if not isinstance(rule, Mapping) or tuple(rule.get("sources", ())) != sources:
            raise ValueError(f"functional source mapping differs for {target_id}")
    return policy


def _load_confirmatory_functional_source_policy(
    source_policy_path: Path,
    manifest_path: Path,
    declarations: tuple[tuple[str, tuple[str, ...]], ...],
) -> Mapping[str, Any]:
    authorization = json.loads(source_policy_path.read_text())
    if (
        authorization.get("schema_version")
        != "patient-atlas-confirmatory-functional-source-policy-v1"
        or authorization.get("status") != "frozen_before_official_test_access"
        or authorization.get("scope")
        != "official_test_once_after_confirmatory_preflight"
        or authorization.get("authorized_split") != "test"
        or authorization.get("official_test_accessed_when_frozen") is not False
    ):
        raise ValueError("confirmatory functional authorization is not frozen")
    root = source_policy_path.parent
    base_binding = authorization.get("base_functional_source_policy", {})
    target_binding = authorization.get("target_manifest", {})
    base_path = root / str(base_binding.get("file", ""))
    if (
        base_path.name != base_binding.get("file")
        or hashlib.sha256(base_path.read_bytes()).hexdigest()
        != base_binding.get("sha256")
        or manifest_path.name != target_binding.get("file")
        or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        != target_binding.get("sha256")
    ):
        raise ValueError("confirmatory functional authorization binding differs")
    return _load_functional_source_policy(base_path, manifest_path, declarations)


def _load_index_visit_functional_targets(
    *,
    dataset_root: str | Path,
    patient_ids: Sequence[str],
    manifest_path: str | Path,
    source_policy_path: str | Path,
    allowed_splits: frozenset[str],
) -> FunctionalTargets:
    """Load the five frozen targets at the same authenticated study visit.

    Split authorization is established by one of the public wrappers below.
    """

    dataset_root = Path(dataset_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    source_policy_path = Path(source_policy_path).resolve()
    normalized_ids = tuple(str(value) for value in patient_ids)
    if not normalized_ids or len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("target patient identities must be nonempty and unique")
    declarations = _target_source_columns(manifest_path)
    source_policy = (
        _load_confirmatory_functional_source_policy(
            source_policy_path, manifest_path, declarations
        )
        if allowed_splits == frozenset(("test",))
        else _load_functional_source_policy(
            source_policy_path, manifest_path, declarations
        )
    )
    required_sources = {
        source for _, sources in declarations for source in sources
    }

    participants = pd.read_csv(
        dataset_root / "participants.tsv",
        sep="\t",
        usecols=("person_id", "study_visit_date", "recommended_split"),
    )
    participants["person_id"] = participants["person_id"].astype(str)
    if not participants["person_id"].is_unique:
        raise ValueError("participant identities are not unique")
    available = set(participants["person_id"])
    missing_count = len(set(normalized_ids) - available)
    if missing_count:
        raise ValueError(f"{missing_count} target patient identities are absent")
    participants = (
        participants.set_index("person_id", verify_integrity=True)
        .loc[list(normalized_ids)]
        .reset_index()
    )
    observed_splits = set(participants["recommended_split"].astype(str))
    if not observed_splits or not observed_splits.issubset(allowed_splits):
        if allowed_splits == frozenset(("train", "val")) and "test" in observed_splits:
            raise ValueError("functional target loader refuses the official test split")
        raise ValueError("functional target identities cross the authorized split")
    participants["study_visit_date"] = pd.to_datetime(
        participants["study_visit_date"], errors="coerce", format="mixed"
    )
    if not bool(participants["study_visit_date"].notna().all()):
        raise ValueError("functional target cohort has an invalid study visit date")

    measurements = pd.read_csv(
        dataset_root / "clinical_data" / "measurement.csv",
        usecols=(
            "person_id",
            "measurement_date",
            "visit_occurrence_id",
            "measurement_source_value",
            "value_as_number",
        ),
        low_memory=False,
    )
    measurements["source"] = (
        measurements["measurement_source_value"]
        .astype(str)
        .str.split(",", n=1)
        .str[0]
        .str.strip()
    )
    measurements = measurements.loc[
        measurements["source"].isin(required_sources)
    ].copy()
    measurements = _index_linked_rows(
        measurements,
        participants,
        _load_visits(dataset_root),
        source_date_column="measurement_date",
    )
    measurements["numeric"] = pd.to_numeric(
        measurements["value_as_number"], errors="coerce"
    )
    measurements = measurements.loc[np.isfinite(measurements["numeric"])].copy()
    grouped = measurements.groupby(["person_id", "source"], sort=False)["numeric"]
    sizes = grouped.size()
    repeated = sizes[sizes > 1]
    if len(repeated):
        timing = measurements.groupby(["person_id", "source"], sort=False).agg(
            measurement_dates=("measurement_date", "nunique"),
            visit_ids=("visit_occurrence_id", "nunique"),
        )
        timing = timing.loc[repeated.index]
        if bool(
            (timing["measurement_dates"] > 1).any()
            or (timing["visit_ids"] > 1).any()
        ):
            raise ValueError("repeated functional records cross date or visit boundaries")
    if source_policy["within_source_repeat_rule"]["aggregation"] != "arithmetic_mean":
        raise RuntimeError("functional repeat policy changed after validation")
    source_values = grouped.mean().unstack("source")

    values = np.full((len(normalized_ids), len(declarations)), np.nan, dtype=np.float64)
    row_lookup = {patient_id: index for index, patient_id in enumerate(normalized_ids)}
    for target_index, (target_id, sources) in enumerate(declarations):
        available_sources = [source for source in sources if source in source_values.columns]
        if len(available_sources) != len(sources):
            raise ValueError(f"raw source schema is missing a field for {target_id}")
        target = source_values.loc[:, list(sources)].mean(axis=1, skipna=True)
        target.loc[source_values.loc[:, list(sources)].notna().sum(axis=1) == 0] = np.nan
        if target_id == "moca_total":
            target = target.where((target >= 0.0) & (target <= 30.0))
        for patient_id, value in target.items():
            if np.isfinite(value):
                values[row_lookup[str(patient_id)], target_index] = float(value)
    mask = np.isfinite(values)
    return FunctionalTargets(
        target_ids=tuple(target_id for target_id, _ in declarations),
        values=values,
        observed_mask=mask,
    )


def load_index_visit_functional_targets(
    *,
    dataset_root: str | Path,
    patient_ids: Sequence[str],
    manifest_path: str | Path,
    source_policy_path: str | Path,
) -> FunctionalTargets:
    """Load official train/validation targets while refusing test identities."""

    return _load_index_visit_functional_targets(
        dataset_root=dataset_root,
        patient_ids=patient_ids,
        manifest_path=manifest_path,
        source_policy_path=source_policy_path,
        allowed_splits=frozenset(("train", "val")),
    )


def load_confirmatory_index_visit_functional_targets(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    patient_ids: Sequence[str],
    manifest_path: str | Path,
    source_policy_path: str | Path,
) -> FunctionalTargets:
    """Load official test targets only after confirmatory row-free preflight."""

    require_preflight(project_root, PreflightStage.CONFIRMATORY)
    return _load_index_visit_functional_targets(
        dataset_root=dataset_root,
        patient_ids=patient_ids,
        manifest_path=manifest_path,
        source_policy_path=source_policy_path,
        allowed_splits=frozenset(("test",)),
    )


def _slice_tensor(value: torch.Tensor | None, start: int, stop: int):
    return None if value is None else value[start:stop]


def encode_availability_vectors(
    model: torch.nn.Module,
    batch: AtlasTrainingBatch,
    precision_temperatures: Mapping[str, float],
    *,
    batch_size: int = 128,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Encode three views in one immutable basis; vectors remain in memory only."""

    if set(precision_temperatures) != set(PRIMARY_ARMS):
        raise ValueError("precision calibration must cover all three availability arms")
    size = int(batch.eye_embeddings.shape[0])
    if size <= 0 or batch_size <= 0:
        raise ValueError("encoding batch and batch_size must be nonempty")
    eye_present = batch.eye_observed_mask.any(dim=1).cpu().numpy().astype(bool)
    blood_present = (
        batch.blood_observed_mask & batch.blood_eligible_mask
    ).any(dim=1).cpu().numpy().astype(bool)
    availability = {
        "both": eye_present & blood_present,
        "eye_only": eye_present,
        "blood_only": blood_present,
    }
    chunks: dict[str, list[np.ndarray]] = {arm: [] for arm in PRIMARY_ARMS}
    model.eval()
    with torch.inference_mode():
        for start in range(0, size, batch_size):
            stop = min(start + batch_size, size)
            eye_embeddings = batch.eye_embeddings[start:stop]
            eye_observed = batch.eye_observed_mask[start:stop]
            blood_values = batch.blood_values[start:stop]
            blood_observed = (
                batch.blood_observed_mask[start:stop]
                & batch.blood_eligible_mask[start:stop]
            )
            for arm in PRIMARY_ARMS:
                eye_visible = (
                    eye_observed
                    if arm in ("both", "eye_only")
                    else torch.zeros_like(eye_observed)
                )
                blood_visible = (
                    blood_observed
                    if arm in ("both", "blood_only")
                    else torch.zeros_like(blood_observed)
                )
                output = model(
                    eye_embeddings=eye_embeddings,
                    eye_visible_mask=eye_visible,
                    blood_values=blood_values,
                    blood_visible_mask=blood_visible,
                    blood_eligible_mask=batch.blood_eligible_mask[start:stop],
                    demographics=batch.demographics[start:stop],
                    demographic_mask=batch.demographic_mask[start:stop],
                    eye_device_ids=batch.eye_device_ids[start:stop],
                    eye_laterality_ids=batch.eye_laterality_ids[start:stop],
                    eye_quality=_slice_tensor(batch.eye_quality, start, stop),
                    enable_interaction=False,
                )
                state = precision_temperature_state(
                    output.physiology, float(precision_temperatures[arm])
                )
                vector = torch.cat(
                    [state.mean, batch.demographics[start:stop]], dim=1
                )
                chunks[arm].append(vector.cpu().numpy().astype(np.float64, copy=False))
    vectors = {arm: np.concatenate(chunks[arm], axis=0) for arm in PRIMARY_ARMS}
    expected_width = int(getattr(model, "config").latent_dim) + int(
        getattr(model, "config").demographic_dim
    )
    for arm, vector in vectors.items():
        if vector.shape != (size, expected_width) or not np.isfinite(vector).all():
            raise ValueError(f"{arm} vector matrix is malformed")
    return vectors, availability


def paired_evidence_vectors_from_unimodal(
    unimodal_vectors: Mapping[str, np.ndarray],
    *,
    latent_dim: int,
) -> dict[str, np.ndarray]:
    """Build one non-destructive vector from matched eye/blood factor axes."""

    if set(unimodal_vectors) != set(PRIMARY_ARMS):
        raise ValueError("unimodal vector mapping must cover all availability arms")
    eye = np.asarray(unimodal_vectors["eye_only"], dtype=np.float64)
    blood = np.asarray(unimodal_vectors["blood_only"], dtype=np.float64)
    both = np.asarray(unimodal_vectors["both"], dtype=np.float64)
    if (
        eye.ndim != 2
        or eye.shape != blood.shape
        or eye.shape != both.shape
        or eye.shape[1] != latent_dim + 1
    ):
        raise ValueError("unimodal vectors must have shape [patients,latent+age]")
    if not (
        np.allclose(eye[:, -1], blood[:, -1], atol=0.0, rtol=0.0)
        and np.allclose(eye[:, -1], both[:, -1], atol=0.0, rtol=0.0)
    ):
        raise ValueError("all availability arms must carry identical age context")
    zero = np.zeros_like(eye[:, :latent_dim])
    age = eye[:, -1:]
    return {
        "both": np.concatenate(
            [eye[:, :latent_dim], blood[:, :latent_dim], age], axis=1
        ),
        "eye_only": np.concatenate(
            [eye[:, :latent_dim], zero, age], axis=1
        ),
        "blood_only": np.concatenate(
            [zero, blood[:, :latent_dim], age], axis=1
        ),
    }


def encode_paired_evidence_vectors(
    model: torch.nn.Module,
    batch: AtlasTrainingBatch,
    precision_temperatures: Mapping[str, float],
    *,
    batch_size: int = 128,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Encode the 129-D paired-evidence vector without retraining the atlas."""

    unimodal, availability = encode_availability_vectors(
        model,
        batch,
        precision_temperatures,
        batch_size=batch_size,
    )
    latent_dim = int(getattr(model, "config").latent_dim)
    vectors = paired_evidence_vectors_from_unimodal(
        unimodal, latent_dim=latent_dim
    )
    expected_width = 2 * latent_dim + 1
    if any(
        values.shape != (batch.eye_embeddings.shape[0], expected_width)
        or not np.isfinite(values).all()
        for values in vectors.values()
    ):
        raise ValueError("paired-evidence vector matrix is malformed")
    return vectors, availability


def _bootstrap_arm_reports(
    losses: Mapping[str, np.ndarray],
    manifest: TargetManifest,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    size = next(iter(losses.values())).shape[0]
    if bootstrap_samples < 100:
        raise ValueError("development bootstrap requires at least 100 samples")
    if any(value.shape != (size, len(manifest.targets)) for value in losses.values()):
        raise ValueError("arm loss matrices are misaligned")
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, size, size=(bootstrap_samples, size))
    samples = {
        arm: np.asarray(
            [_target_family_score(value[index], manifest) for index in indices],
            dtype=np.float64,
        )
        for arm, value in losses.items()
    }
    arm_report: dict[str, Any] = {}
    for arm, value in losses.items():
        point = _target_family_score(value, manifest)
        lower, upper = _percentile_interval(samples[arm], 0.95)
        arm_report[arm] = {
            "family_balanced_normalized_loss": float(point),
            "confidence_interval": [lower, upper],
            "per_target_normalized_loss": _target_aggregate(value, manifest),
        }

    contrast_samples = {
        "eye_minus_both": samples["eye_only"] - samples["both"],
        "blood_minus_both": samples["blood_only"] - samples["both"],
    }
    points = {
        "eye_minus_both": (
            arm_report["eye_only"]["family_balanced_normalized_loss"]
            - arm_report["both"]["family_balanced_normalized_loss"]
        ),
        "blood_minus_both": (
            arm_report["blood_only"]["family_balanced_normalized_loss"]
            - arm_report["both"]["family_balanced_normalized_loss"]
        ),
    }
    holm = _holm_bootstrap_lower_bounds(
        points, contrast_samples, alpha=0.05
    )
    contrast_report: dict[str, Any] = {}
    for name, point in points.items():
        lower, upper = _basic_bootstrap_interval(
            float(point), contrast_samples[name], 0.95
        )
        contrast_report[name] = {
            "loss_difference": float(point),
            "confidence_interval": [lower, upper],
            "positive_favors_both": True,
            "paired_inference": "centered patient-cluster basic bootstrap",
            **holm[name],
        }
    return arm_report, contrast_report


def evaluate_official_validation(
    *,
    train_patient_ids: Sequence[str],
    train_site_ids: Sequence[str],
    validation_patient_ids: Sequence[str],
    train_vectors: Mapping[str, np.ndarray],
    validation_vectors: Mapping[str, np.ndarray],
    train_availability: Mapping[str, np.ndarray],
    validation_availability: Mapping[str, np.ndarray],
    train_targets: FunctionalTargets,
    validation_targets: FunctionalTargets,
    manifest: TargetManifest,
    ridge_grid: tuple[float, ...] = RIDGE_GRID,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 8_675_309,
    coordinate_groups: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Fit readouts on official train and score once on official validation."""

    if tuple(train_targets.target_ids) != manifest.target_ids or tuple(
        validation_targets.target_ids
    ) != manifest.target_ids:
        raise ValueError("functional targets do not match the frozen target manifest")
    n_train = len(train_patient_ids)
    n_validation = len(validation_patient_ids)
    if len(train_site_ids) != n_train or n_train == 0 or n_validation == 0:
        raise ValueError("development train/validation identities are malformed")
    if set(map(str, train_patient_ids)) & set(map(str, validation_patient_ids)):
        raise ValueError("official train and validation identities overlap")
    if any(
        train_vectors[arm].shape[0] != n_train
        or validation_vectors[arm].shape[0] != n_validation
        for arm in PRIMARY_ARMS
    ):
        raise ValueError("development vectors are not aligned to their split")
    if coordinate_groups is None:
        coordinate_groups = {
            arm: ("atlas",) * (train_vectors[arm].shape[1] - 1)
            + ("demographic",)
            for arm in PRIMARY_ARMS
        }
    if set(coordinate_groups) != set(PRIMARY_ARMS):
        raise ValueError("coordinate groups must cover all availability arms")
    for arm in PRIMARY_ARMS:
        if (
            len(coordinate_groups[arm]) != train_vectors[arm].shape[1]
            or len(set(coordinate_groups[arm])) < 2
        ):
            raise ValueError(f"coordinate groups are malformed for {arm}")
    common_train = np.asarray(train_availability["both"], dtype=bool)
    common_validation = np.asarray(validation_availability["both"], dtype=bool)
    if common_train.shape != (n_train,) or common_validation.shape != (n_validation,):
        raise ValueError("complete-case availability masks are malformed")

    fold_map = make_canonical_fold_map(
        train_patient_ids,
        train_site_ids,
        n_folds=5,
        seed=1701,
        salt="patient-atlas-official-train-readout-v1",
    )
    inner_folds = fold_map.assignments_for(train_patient_ids)
    losses = {
        arm: np.full((n_validation, len(manifest.targets)), np.nan, dtype=np.float64)
        for arm in PRIMARY_ARMS
    }
    tuning: dict[str, dict[str, Any]] = {arm: {} for arm in PRIMARY_ARMS}
    train_counts: dict[str, int] = {}
    validation_counts: dict[str, int] = {}

    for target_index, target in enumerate(manifest.targets):
        eligible_train = common_train & train_targets.observed_mask[:, target_index]
        eligible_validation = (
            common_validation & validation_targets.observed_mask[:, target_index]
        )
        train_count = int(eligible_train.sum())
        validation_count = int(eligible_validation.sum())
        train_counts[target.id] = train_count
        validation_counts[target.id] = validation_count
        if train_count < 50 or validation_count < 20:
            raise ValueError(f"target {target.id} lacks a reliable development sample")
        if train_count + validation_count < manifest.minimum_complete_target_patients:
            raise ValueError(
                f"target {target.id} is below the frozen complete-case minimum"
            )
        y_train = train_targets.values[:, target_index]
        baseline_mean = float(y_train[eligible_train].mean())
        baseline_loss = float(
            np.mean((y_train[eligible_train] - baseline_mean) ** 2)
        )
        if not math.isfinite(baseline_loss) or baseline_loss <= 1e-12:
            raise ValueError(f"target {target.id} has degenerate training variance")

        for arm in PRIMARY_ARMS:
            x_train = np.asarray(train_vectors[arm], dtype=np.float64)
            x_validation = np.asarray(validation_vectors[arm], dtype=np.float64)
            groups = tuple(coordinate_groups[arm])
            train_candidates, validation_candidates = _group_subset_candidates(
                x_train, x_validation, groups
            )
            choice = _select_probe(
                train_candidates,
                y_train,
                eligible_train,
                inner_folds,
                ridge_grid=ridge_grid,
                max_tuning_combinations=20_000,
            )
            candidate_id = choice.concat_eye_dimension
            if candidate_id is None or candidate_id not in train_candidates:
                raise RuntimeError("group-subset readout selection is invalid")
            selected_x_train, selected_groups = train_candidates[candidate_id]
            selected_x_validation = validation_candidates[candidate_id]
            model = _fit_ridge(
                selected_x_train[eligible_train],
                y_train[eligible_train],
                selected_groups,
                choice.penalty_mapping(),
            )
            prediction = _predict_ridge(
                model, selected_x_validation[eligible_validation]
            )
            squared = (
                prediction - validation_targets.values[eligible_validation, target_index]
            ) ** 2
            losses[arm][eligible_validation, target_index] = squared / baseline_loss
            selected_penalties = choice.penalty_mapping()
            retained = tuple(sorted(set(selected_groups)))
            dropped = tuple(sorted(set(groups) - set(retained)))
            positive_lower_endpoint = bool(
                ridge_grid[0] > 0
                and any(value == ridge_grid[0] for value in selected_penalties.values())
            )
            upper_endpoint = bool(
                any(value == ridge_grid[-1] for value in selected_penalties.values())
            )
            tuning[arm][target.id] = {
                "penalties": {
                    name: float(value)
                    for name, value in sorted(selected_penalties.items())
                },
                "retained_groups": list(retained),
                "dropped_groups": list(dropped),
                "unregularized_groups": sorted(
                    name for name, value in selected_penalties.items() if value == 0.0
                ),
                "inner_normalized_loss": float(choice.inner_loss / baseline_loss),
                "positive_lower_grid_endpoint": positive_lower_endpoint,
                "upper_grid_endpoint": upper_endpoint,
                "grid_endpoint": positive_lower_endpoint or upper_endpoint,
            }

    arm_report, contrast_report = _bootstrap_arm_reports(
        losses,
        manifest,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    superiority = bool(
        contrast_report["eye_minus_both"]["holm_superiority_rejected_null"]
        and contrast_report["blood_minus_both"]["holm_superiority_rejected_null"]
    )
    endpoint_count = sum(
        int(details["grid_endpoint"])
        for arm in tuning.values()
        for details in arm.values()
    )
    tuning_total = len(PRIMARY_ARMS) * len(manifest.targets)
    report: dict[str, Any] = {
        "schema_version": "patient-atlas-official-development-validation-v2",
        "scope": "official_train_readouts_official_validation_scoring",
        "confirmatory_test_split_loaded": False,
        "representation_refit_for_outcomes": False,
        "same_complete_case_patients_all_arms": True,
        "identical_demographic_context_all_arms": True,
        "train_patient_count": n_train,
        "validation_patient_count": n_validation,
        "complete_case_train_count": int(common_train.sum()),
        "complete_case_validation_count": int(common_validation.sum()),
        "target_train_counts": train_counts,
        "target_validation_counts": validation_counts,
        "arm_results": arm_report,
        "primary_contrasts": contrast_report,
        "acceptance": {
            "both_beats_each_single_modality_with_holm_fwer_0_05": superiority,
            "point_estimate_beats_eye_only": bool(
                contrast_report["eye_minus_both"]["loss_difference"] > 0.0
            ),
            "point_estimate_beats_blood_only": bool(
                contrast_report["blood_minus_both"]["loss_difference"] > 0.0
            ),
        },
        "readout_tuning": {
            "ridge_grid": list(ridge_grid),
            "selection": tuning,
            "endpoint_selection_count": endpoint_count,
            "selection_count": tuning_total,
            "requires_future_grid_review": bool(endpoint_count / tuning_total > 0.10),
        },
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "unit": "official-validation patient",
        },
        "claim_limitations": {
            "development_not_confirmatory": True,
            "external_validation_completed": False,
            "clinical_benefit_claim_allowed": False,
            "concat_noninferiority_evaluable": False,
        },
        "patient_rows_emitted": False,
        "patient_identifiers_emitted": False,
        "patient_vectors_emitted": False,
        "per_patient_predictions_emitted": False,
    }
    assert_aggregate_only_payload(
        report,
        forbidden_patient_ids=tuple(map(str, train_patient_ids))
        + tuple(map(str, validation_patient_ids)),
    )
    return report


__all__ = [
    "FunctionalTargets",
    "PAIRED_RIDGE_GRID",
    "PRIMARY_ARMS",
    "RIDGE_GRID",
    "encode_availability_vectors",
    "encode_paired_evidence_vectors",
    "evaluate_official_validation",
    "load_confirmatory_index_visit_functional_targets",
    "load_index_visit_functional_targets",
    "paired_evidence_vectors_from_unimodal",
]
