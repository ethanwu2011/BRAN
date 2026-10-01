"""Fail-closed exploratory AI-READI loader for the Patient Atlas.

Patient-derived rows remain local and in memory. The public summary contains
only disclosure-safe aggregate counts and hashes. Functional target values are
not selected by this module, and the official test split is rejected.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from patient_atlas_preflight import PreflightStage, require_preflight
from patient_atlas_preprocessing import FoldPreprocessor
from patient_atlas_raw_audit import (
    CONDITION_SOURCES,
    SMALL_CELL_THRESHOLD,
    map_measurement_features,
)


SOURCE_POLICY_SCHEMA = "patient-atlas-source-policy-v1"
ALLOWED_EXPLORATORY_SPLITS = ("train", "val")
BP_FEATURES = frozenset(("vit_sysbp_vsorres", "vit_diabp_vsorres"))
STUDY_GROUP_DIABETES = {
    "healthy": 0.0,
    "pre_diabetes_lifestyle_controlled": 0.0,
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": 1.0,
    "insulin_dependent": 1.0,
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_count(value: int) -> int | str:
    value = int(value)
    if value == 0 or value >= SMALL_CELL_THRESHOLD:
        return value
    return f"<{SMALL_CELL_THRESHOLD}"


def _safe_value_counts(values: Sequence[str]) -> dict[str, int | str]:
    counts = pd.Series(tuple(map(str, values)), dtype="string").value_counts().sort_index()
    return {str(key): _safe_count(int(value)) for key, value in counts.items()}


def _normalized_retinal_path(value: str) -> str:
    path = str(value).replace("\\", "/").lstrip("./")
    marker = "retinal_photography/"
    position = path.find(marker)
    return path[position:] if position >= 0 else path


def _read_policy(project_root: Path) -> tuple[dict[str, Any], str]:
    policy_path = project_root / "PATIENT_ATLAS_SOURCE_POLICY.json"
    raw = policy_path.read_bytes()
    policy = json.loads(raw)
    if not isinstance(policy, dict) or policy.get("schema_version") != SOURCE_POLICY_SCHEMA:
        raise ValueError("unsupported Patient Atlas source policy")
    if policy.get("scope") != "exploratory_train_validation_only":
        raise ValueError("source policy is not restricted to exploratory train/validation")
    return policy, hashlib.sha256(raw).hexdigest()


def _verify_source_hashes(
    policy: Mapping[str, Any],
    *,
    dataset_root: Path,
    clinical_project_root: Path,
) -> dict[str, str]:
    expected = policy.get("source_hashes")
    if not isinstance(expected, Mapping):
        raise ValueError("source policy does not contain source hashes")
    paths = {
        "participants_tsv": dataset_root / "participants.tsv",
        "participants_json": dataset_root / "participants.json",
        "measurement_csv": dataset_root / "clinical_data" / "measurement.csv",
        "observation_csv": dataset_root / "clinical_data" / "observation.csv",
        "visit_occurrence_csv": dataset_root
        / "clinical_data"
        / "visit_occurrence.csv",
        "retinal_manifest_tsv": dataset_root
        / "retinal_photography"
        / "manifest.tsv",
        "eye_embedding_npy": clinical_project_root
        / "data"
        / "local_emb"
        / "aireadi_emb_ours.npy",
        "eye_embedding_metadata_parquet": clinical_project_root
        / "data"
        / "aireadi_emb_ours_meta.parquet",
    }
    if set(expected) != set(paths):
        raise ValueError("source policy hash names do not match the loader contract")
    observed: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required source artifact is missing: {name}")
        observed[name] = _sha256(path)
        if observed[name] != expected[name]:
            raise ValueError(f"source artifact hash mismatch: {name}")
    return observed


def _load_registry(project_root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    value = json.loads((project_root / "PATIENT_ATLAS_FEATURE_REGISTRY.json").read_text())
    features = value.get("features")
    if not isinstance(features, list) or len(features) != 59:
        raise ValueError("feature registry must contain exactly 59 fields")
    if [feature.get("index") for feature in features] != list(range(59)):
        raise ValueError("feature registry is not in canonical order")
    names = tuple(str(feature["name"]) for feature in features)
    types = tuple(str(feature["type"]) for feature in features)
    if types != ("continuous",) * 48 + ("binary",) * 11:
        raise ValueError("feature registry must contain 48 continuous then 11 binary fields")
    return names, types


def _load_visits(dataset_root: Path) -> pd.DataFrame:
    visits = pd.read_csv(
        dataset_root / "clinical_data" / "visit_occurrence.csv",
        usecols=("visit_occurrence_id", "visit_start_date"),
        low_memory=False,
    )
    if not visits["visit_occurrence_id"].is_unique:
        raise ValueError("visit_occurrence_id is not unique")
    visits["visit_start_date"] = pd.to_datetime(
        visits["visit_start_date"], errors="coerce", format="mixed"
    )
    return visits


def _index_linked_rows(
    rows: pd.DataFrame,
    participants: pd.DataFrame,
    visits: pd.DataFrame,
    *,
    source_date_column: str,
) -> pd.DataFrame:
    selected_ids = set(participants["person_id"])
    rows = rows.loc[rows["person_id"].astype(str).isin(selected_ids)].copy()
    rows["person_id"] = rows["person_id"].astype(str)
    rows[source_date_column] = pd.to_datetime(
        rows[source_date_column], errors="coerce", format="mixed"
    )
    rows = rows.merge(
        participants[["person_id", "study_visit_date"]],
        on="person_id",
        how="inner",
        validate="many_to_one",
    )
    rows = rows.merge(
        visits,
        on="visit_occurrence_id",
        how="left",
        validate="many_to_one",
    )
    at_index = rows[source_date_column].eq(rows["study_visit_date"]) | rows[
        "visit_start_date"
    ].eq(rows["study_visit_date"])
    return rows.loc[at_index].copy()


def _assemble_continuous(
    *,
    dataset_root: Path,
    participants: pd.DataFrame,
    visits: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    columns = (
        "person_id",
        "measurement_date",
        "visit_occurrence_id",
        "measurement_source_value",
        "value_as_number",
    )
    measurements = pd.read_csv(
        dataset_root / "clinical_data" / "measurement.csv",
        usecols=columns,
        low_memory=False,
    )
    continuous_names = set(feature_names[:48])
    measurements["feature"] = map_measurement_features(
        measurements["measurement_source_value"], continuous_names
    )
    measurements = measurements.loc[measurements["feature"].notna()]
    measurements = _index_linked_rows(
        measurements,
        participants,
        visits,
        source_date_column="measurement_date",
    )
    measurements["numeric"] = pd.to_numeric(
        measurements["value_as_number"], errors="coerce"
    )
    measurements = measurements.loc[np.isfinite(measurements["numeric"])]

    grouped = measurements.groupby(["person_id", "feature"], sort=False)["numeric"]
    sizes = grouped.size()
    illegal = sizes.loc[
        (sizes > 1)
        & ~sizes.index.get_level_values("feature").isin(BP_FEATURES)
    ]
    if len(illegal):
        raise ValueError("unexpected duplicate non-BP index-visit measurement")
    aggregated = grouped.mean().reset_index()

    n = len(participants)
    values = np.zeros((n, 59), dtype=np.float32)
    observed = np.zeros((n, 59), dtype=bool)
    patient_index = {value: index for index, value in enumerate(participants["person_id"])}
    feature_index = {value: index for index, value in enumerate(feature_names)}
    for row in aggregated.itertuples(index=False):
        i = patient_index[str(row.person_id)]
        j = feature_index[str(row.feature)]
        values[i, j] = float(row.numeric)
        observed[i, j] = True
    return values, observed


def _assemble_conditions(
    *,
    dataset_root: Path,
    participants: pd.DataFrame,
    visits: pd.DataFrame,
    feature_names: tuple[str, ...],
    values: np.ndarray,
    observed: np.ndarray,
) -> None:
    columns = (
        "person_id",
        "observation_date",
        "visit_occurrence_id",
        "observation_source_value",
        "value_as_number",
    )
    observations = pd.read_csv(
        dataset_root / "clinical_data" / "observation.csv",
        usecols=columns,
        low_memory=False,
    )
    observations["source"] = (
        observations["observation_source_value"]
        .fillna("")
        .astype(str)
        .str.split(",", n=1)
        .str[0]
        .str.strip()
    )
    source_to_condition = {source: name for name, source in CONDITION_SOURCES.items()}
    observations = observations.loc[observations["source"].isin(source_to_condition)]
    observations["condition"] = observations["source"].map(source_to_condition)
    observations = _index_linked_rows(
        observations,
        participants,
        visits,
        source_date_column="observation_date",
    )
    observations["numeric"] = pd.to_numeric(
        observations["value_as_number"], errors="coerce"
    )
    observations = observations.loc[observations["numeric"].isin((0.0, 1.0))]
    grouped = observations.groupby(["person_id", "condition"], sort=False)["numeric"]
    sizes = grouped.size()
    if bool((sizes > 1).any()) or bool((grouped.nunique() > 1).any()):
        raise ValueError("duplicate or conflicting index-linked condition record")
    aggregated = grouped.first().reset_index()

    patient_index = {value: index for index, value in enumerate(participants["person_id"])}
    feature_index = {value: index for index, value in enumerate(feature_names)}
    for row in aggregated.itertuples(index=False):
        i = patient_index[str(row.person_id)]
        j = feature_index[str(row.condition)]
        values[i, j] = float(row.numeric)
        observed[i, j] = True

    unknown = set(participants["study_group"]) - set(STUDY_GROUP_DIABETES)
    if unknown:
        raise ValueError("unrecognized study_group value")
    diabetes_index = feature_index["diabetes"]
    diabetes = participants["study_group"].map(STUDY_GROUP_DIABETES).to_numpy(
        dtype=np.float32
    )
    values[:, diabetes_index] = diabetes
    observed[:, diabetes_index] = True


def _assemble_eye(
    *,
    dataset_root: Path,
    embedding_path: Path,
    participants: pd.DataFrame,
    policy: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eye_policy = policy.get("eye_alignment", {})
    device_order = tuple(map(str, eye_policy.get("device_order", ())))
    laterality_order = tuple(map(str, eye_policy.get("laterality_order", ())))
    if len(device_order) != 5 or device_order[-1] != "unknown":
        raise ValueError("source policy device order is invalid")
    if laterality_order != ("L", "R", "unknown"):
        raise ValueError("source policy laterality order is invalid")
    device_index = {value: index for index, value in enumerate(device_order)}
    laterality_index = {value: index for index, value in enumerate(laterality_order)}

    manifest = pd.read_csv(
        dataset_root / "retinal_photography" / "manifest.tsv",
        sep="\t",
        usecols=("person_id", "laterality", "filepath"),
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
    manifest_by_basename = manifest.set_index("basename", verify_integrity=True)

    pattern = str(
        dataset_root / "retinal_photography" / "cfp" / "**" / "*.dcm"
    )
    physical_paths = [Path(value) for value in sorted(glob.glob(pattern, recursive=True))]
    basenames = [path.name for path in physical_paths]
    if len(basenames) != len(set(basenames)):
        raise ValueError("physical CFP basenames are not unique")
    if set(basenames) != set(manifest_by_basename.index.astype(str)):
        raise ValueError("physical CFP files and authoritative manifest differ")

    embeddings = np.load(embedding_path, mmap_mode="r")
    expected_shape = tuple(eye_policy.get("embedding_shape", ()))
    if tuple(embeddings.shape) != expected_shape or expected_shape != (50315, 384):
        raise ValueError("eye embedding shape differs from the source policy")
    if embeddings.dtype != np.float32:
        raise TypeError("eye embeddings must be float32")

    selected_ids = set(participants["person_id"])
    records: list[tuple[int, str, int, int]] = []
    for embedding_row, path in enumerate(physical_paths):
        metadata = manifest_by_basename.loc[path.name]
        patient_id = str(metadata["person_id"])
        if patient_id not in selected_ids:
            continue
        normalized = _normalized_retinal_path(str(path))
        parts = normalized.split("/")
        try:
            device = parts[parts.index("cfp") + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError("cannot derive device from authenticated CFP path") from exc
        if device not in device_index:
            device = "unknown"
        laterality = str(metadata["laterality"])
        if laterality not in laterality_index:
            laterality = "unknown"
        records.append(
            (
                embedding_row,
                patient_id,
                device_index[device],
                laterality_index[laterality],
            )
        )

    by_patient: dict[str, list[tuple[int, int, int]]] = {
        patient_id: [] for patient_id in participants["person_id"]
    }
    for embedding_row, patient_id, device_id, laterality_id in records:
        by_patient[patient_id].append((embedding_row, device_id, laterality_id))
    maximum_images = max((len(rows) for rows in by_patient.values()), default=0)
    if maximum_images <= 0:
        raise ValueError("selected exploratory cohort has no CFP observations")

    n = len(participants)
    eye = np.zeros((n, maximum_images, 384), dtype=np.float32)
    mask = np.zeros((n, maximum_images), dtype=bool)
    devices = np.zeros((n, maximum_images), dtype=np.int64)
    lateralities = np.full(
        (n, maximum_images), laterality_index["unknown"], dtype=np.int64
    )
    quality = np.zeros((n, maximum_images), dtype=np.float32)
    for patient_row, patient_id in enumerate(participants["person_id"]):
        for image_row, (embedding_row, device_id, laterality_id) in enumerate(
            by_patient[patient_id]
        ):
            vector = np.asarray(embeddings[embedding_row], dtype=np.float32)
            if not np.isfinite(vector).all() or not bool(np.abs(vector).sum() > 0):
                raise ValueError("visible eye embedding is nonfinite or zero")
            eye[patient_row, image_row] = vector
            mask[patient_row, image_row] = True
            devices[patient_row, image_row] = device_id
            lateralities[patient_row, image_row] = laterality_id
            quality[patient_row, image_row] = 1.0
    return eye, mask, devices, lateralities, quality


@dataclass(frozen=True, repr=False)
class ExploratoryRawCohort:
    """Sensitive in-memory cohort; its repr and public summary are aggregate-only."""

    patient_ids: tuple[str, ...]
    site_ids: tuple[str, ...]
    split_labels: tuple[str, ...]
    feature_names: tuple[str, ...]
    blood_values: np.ndarray
    blood_observed_mask: np.ndarray
    blood_eligible_mask: np.ndarray
    ages: np.ndarray
    age_observed_mask: np.ndarray
    eye_embeddings: np.ndarray
    eye_observed_mask: np.ndarray
    eye_device_ids: np.ndarray
    eye_laterality_ids: np.ndarray
    eye_quality: np.ndarray
    source_policy_sha256: str
    source_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        n = len(self.patient_ids)
        if n == 0 or len(set(self.patient_ids)) != n:
            raise ValueError("cohort patient identities must be nonempty and unique")
        if len(self.site_ids) != n or len(self.split_labels) != n:
            raise ValueError("cohort identity metadata is misaligned")
        if set(self.split_labels) - set(ALLOWED_EXPLORATORY_SPLITS):
            raise ValueError("exploratory cohort contains a sealed split")
        if len(self.feature_names) != 59:
            raise ValueError("cohort must contain the canonical 59 clinical fields")
        expected = (n, 59)
        if self.blood_values.shape != expected or self.blood_observed_mask.shape != expected:
            raise ValueError("clinical arrays have the wrong shape")
        if self.blood_eligible_mask.shape != (59,) or self.blood_eligible_mask.dtype != np.bool_:
            raise TypeError("blood eligibility mask must be boolean with width 59")
        if self.eye_embeddings.ndim != 3 or self.eye_embeddings.shape[0] != n or self.eye_embeddings.shape[2] != 384:
            raise ValueError("eye array must have shape [patients,images,384]")
        if self.eye_observed_mask.shape != self.eye_embeddings.shape[:2]:
            raise ValueError("eye observation mask is misaligned")
        if self.eye_device_ids.shape != self.eye_observed_mask.shape or self.eye_laterality_ids.shape != self.eye_observed_mask.shape or self.eye_quality.shape != self.eye_observed_mask.shape:
            raise ValueError("eye metadata arrays are misaligned")
        if self.ages.shape != (n,) or self.age_observed_mask.shape != (n,):
            raise ValueError("age arrays are misaligned")
        if len(self.source_policy_sha256) != 64:
            raise ValueError("source policy hash is invalid")

    def __repr__(self) -> str:
        return (
            "ExploratoryRawCohort(aggregate_only=True, "
            f"patients={len(self.patient_ids)}, images={int(self.eye_observed_mask.sum())})"
        )

    def indices_for_split(self, split: str) -> np.ndarray:
        if split not in ALLOWED_EXPLORATORY_SPLITS:
            raise ValueError("only train and val are available in exploratory scope")
        return np.flatnonzero(np.asarray(self.split_labels) == split)

    def aggregate_summary(self) -> dict[str, Any]:
        eye_present = self.eye_observed_mask.any(axis=1)
        blood_present = self.blood_observed_mask.any(axis=1)
        continuous = self.blood_observed_mask[:, :48]
        binary = self.blood_observed_mask[:, 48:]
        visible_device_ids = self.eye_device_ids[self.eye_observed_mask]
        visible_lateralities = self.eye_laterality_ids[self.eye_observed_mask]
        return {
            "scope": "exploratory_train_validation_only",
            "patients": _safe_count(len(self.patient_ids)),
            "split_counts": _safe_value_counts(self.split_labels),
            "site_counts": _safe_value_counts(self.site_ids),
            "eye_images": _safe_count(int(self.eye_observed_mask.sum())),
            "patients_with_eye": _safe_count(int(eye_present.sum())),
            "patients_with_clinical": _safe_count(int(blood_present.sum())),
            "patients_with_both": _safe_count(int((eye_present & blood_present).sum())),
            "continuous_observed_cells": _safe_count(int(continuous.sum())),
            "binary_observed_cells": _safe_count(int(binary.sum())),
            "visible_device_id_counts": _safe_value_counts(
                tuple(map(str, visible_device_ids.tolist()))
            ),
            "visible_laterality_id_counts": _safe_value_counts(
                tuple(map(str, visible_lateralities.tolist()))
            ),
            "age_missing": _safe_count(int((~self.age_observed_mask).sum())),
            "feature_count": len(self.feature_names),
            "eye_dimension": int(self.eye_embeddings.shape[2]),
            "source_policy_sha256": self.source_policy_sha256,
            "source_hashes": dict(sorted(self.source_hashes.items())),
            "patient_rows_emitted": False,
            "patient_identifiers_emitted": False,
            "functional_target_values_loaded": False,
        }

    def to_training_batch(
        self,
        preprocessor: FoldPreprocessor,
        *,
        indices: Sequence[int] | np.ndarray | None = None,
    ):
        """Transform selected rows and return the outcome-free torch batch API."""

        import torch

        from train_soft_patient_atlas import AtlasTrainingBatch

        index = (
            np.arange(len(self.patient_ids), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if index.ndim != 1 or len(index) == 0 or index.min() < 0 or index.max() >= len(self.patient_ids):
            raise ValueError("training-batch indices are invalid")
        blood = preprocessor.transform_blood(
            self.blood_values[index],
            self.blood_observed_mask[index],
            ordered_feature_names=self.feature_names,
            policy_eligible_mask=self.blood_eligible_mask,
        )
        age = preprocessor.transform_age(
            self.ages[index], self.age_observed_mask[index], require_observed=True
        )
        eye = preprocessor.transform_eye(
            self.eye_embeddings[index], self.eye_observed_mask[index]
        )
        eligible = np.broadcast_to(
            self.blood_eligible_mask[None, :], blood.values.shape
        ).copy()
        return AtlasTrainingBatch(
            eye_embeddings=torch.from_numpy(eye.values),
            eye_observed_mask=torch.from_numpy(eye.observed_mask),
            blood_values=torch.from_numpy(blood.values),
            blood_observed_mask=torch.from_numpy(blood.observed_mask),
            blood_eligible_mask=torch.from_numpy(eligible),
            demographics=torch.from_numpy(age.values),
            demographic_mask=torch.from_numpy(age.observed_mask),
            eye_device_ids=torch.from_numpy(self.eye_device_ids[index].copy()),
            eye_laterality_ids=torch.from_numpy(
                self.eye_laterality_ids[index].copy()
            ),
            eye_quality=torch.from_numpy(self.eye_quality[index].copy()),
            target_blood_anchor=None,
        )


def load_exploratory_raw_cohort(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    clinical_project_root: str | Path,
) -> ExploratoryRawCohort:
    """Load only official train/validation inputs after all row-free gates pass."""

    project_root = Path(project_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    clinical_project_root = Path(clinical_project_root).resolve()
    require_preflight(project_root, PreflightStage.EXPLORATORY)
    policy, source_policy_sha256 = _read_policy(project_root)
    source_hashes = _verify_source_hashes(
        policy,
        dataset_root=dataset_root,
        clinical_project_root=clinical_project_root,
    )
    feature_names, _ = _load_registry(project_root)

    participants = pd.read_csv(
        dataset_root / "participants.tsv",
        sep="\t",
        usecols=(
            "person_id",
            "clinical_site",
            "study_group",
            "age",
            "study_visit_date",
            "recommended_split",
        ),
    )
    participants["person_id"] = participants["person_id"].astype(str)
    if not participants["person_id"].is_unique:
        raise ValueError("participant identities are not unique")
    participants = participants.loc[
        participants["recommended_split"].isin(ALLOWED_EXPLORATORY_SPLITS)
    ].copy()
    participants["study_visit_date"] = pd.to_datetime(
        participants["study_visit_date"], errors="coerce", format="mixed"
    )
    participants["age"] = pd.to_numeric(participants["age"], errors="coerce")
    if not bool(
        participants["age"].notna().all()
        and np.isfinite(participants["age"]).all()
        and (participants["age"] >= 0).all()
        and participants["study_visit_date"].notna().all()
    ):
        raise ValueError("exploratory cohort requires valid observed age and study visit date")

    visits = _load_visits(dataset_root)
    blood_values, blood_mask = _assemble_continuous(
        dataset_root=dataset_root,
        participants=participants,
        visits=visits,
        feature_names=feature_names,
    )
    _assemble_conditions(
        dataset_root=dataset_root,
        participants=participants,
        visits=visits,
        feature_names=feature_names,
        values=blood_values,
        observed=blood_mask,
    )
    embedding_path = (
        clinical_project_root / "data" / "local_emb" / "aireadi_emb_ours.npy"
    )
    eye, eye_mask, devices, lateralities, quality = _assemble_eye(
        dataset_root=dataset_root,
        embedding_path=embedding_path,
        participants=participants,
        policy=policy,
    )

    age_mask = np.isfinite(participants["age"].to_numpy(dtype=float))
    eligible = age_mask & (blood_mask.any(axis=1) | eye_mask.any(axis=1))
    if not bool(eligible.any()):
        raise ValueError("no eligible exploratory patients remain")
    participants = participants.loc[eligible].reset_index(drop=True)
    blood_values = blood_values[eligible]
    blood_mask = blood_mask[eligible]
    eye = eye[eligible]
    eye_mask = eye_mask[eligible]
    devices = devices[eligible]
    lateralities = lateralities[eligible]
    quality = quality[eligible]

    return ExploratoryRawCohort(
        patient_ids=tuple(participants["person_id"]),
        site_ids=tuple(participants["clinical_site"].astype(str)),
        split_labels=tuple(participants["recommended_split"].astype(str)),
        feature_names=feature_names,
        blood_values=blood_values,
        blood_observed_mask=blood_mask,
        blood_eligible_mask=np.ones(59, dtype=bool),
        ages=participants["age"].to_numpy(dtype=np.float64),
        age_observed_mask=np.ones(len(participants), dtype=bool),
        eye_embeddings=eye,
        eye_observed_mask=eye_mask,
        eye_device_ids=devices,
        eye_laterality_ids=lateralities,
        eye_quality=quality,
        source_policy_sha256=source_policy_sha256,
        source_hashes=source_hashes,
    )


class ZeroBloodAnchor:
    """Factory namespace for the exact-zero primary exploratory anchor."""

    @staticmethod
    def build(width: int = 64):
        import torch
        import torch.nn as nn

        if not isinstance(width, int) or width <= 0:
            raise ValueError("zero-anchor width must be positive")

        class _ZeroBloodAnchor(nn.Module):
            def __init__(self, output_width: int) -> None:
                super().__init__()
                self.output_width = output_width

            def encode(self, values: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
                if values.ndim != 2 or observed_mask.shape != values.shape:
                    raise ValueError("zero-anchor inputs must have aligned [batch,features] shape")
                return values.new_zeros((values.shape[0], self.output_width))

        return _ZeroBloodAnchor(width)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clinical-project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cohort = load_exploratory_raw_cohort(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        clinical_project_root=args.clinical_project_root,
    )
    summary = {
        "schema_version": "patient-atlas-exploratory-cohort-summary-v1",
        "loader_sha256": _sha256(Path(__file__).resolve()),
        **cohort.aggregate_summary(),
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with args.output.open("xb") as handle:
        handle.write(encoded.encode("utf-8"))
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_EXPLORATORY_SPLITS",
    "ExploratoryRawCohort",
    "ZeroBloodAnchor",
    "load_exploratory_raw_cohort",
    "main",
]
