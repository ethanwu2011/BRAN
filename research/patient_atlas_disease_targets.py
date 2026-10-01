"""Local-only, fail-closed disease targets for Patient Atlas V5.

The source/derivation policy is frozen independently of target values. This
module aligns targets to an already authenticated train/validation cohort and
exposes only small-cell-suppressed aggregate summaries outside local memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from patient_atlas_preprocessing import hash_json
from patient_atlas_raw_audit import SMALL_CELL_THRESHOLD
from patient_atlas_real_data import (
    ExploratoryRawCohort,
    _index_linked_rows,
    _load_visits,
)


POLICY_NAME = "PATIENT_ATLAS_V5_DISEASE_TARGET_SOURCE_POLICY_V1.json"
POLICY_SCHEMA_VERSION = "patient-atlas-v5-disease-target-source-policy-v1"
TARGET_COLUMNS = (
    "raw_hba1c",
    "raw_ldl_cholesterol",
    "any_dysglycemia",
    "any_dm",
    "htn_measured",
    "uacr",
    "albuminuria",
    "fun_moca_total_score",
    "moca_lt26",
    "mhoccur_rnl",
    "mhoccur_ca",
    "mhoccur_mi",
    "mhoccur_strk",
    "mhoccur_circ",
    "mhoccur_cvdot",
    "mhoccur_clsh",
    "mhoccur_plm",
    "mhoccur_cns",
    "mhoccur_cogn",
    "mhoccur_ear",
    "mhoccur_oa",
    "mhoccur_fall",
)
OBSERVATION_TARGET_COLUMNS = TARGET_COLUMNS[9:]
BLOCKED_TARGET_COLUMNS = ("uacr", "albuminuria")
_STUDY_GROUP_SEVERITY = {
    "healthy": 0,
    "pre_diabetes_lifestyle_controlled": 1,
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": 2,
    "insulin_dependent": 3,
}
_REQUIRED_BINDINGS = {
    "disease_utility_registry",
    "base_source_policy",
    "feature_registry",
    "official_unit_reconciliation",
    "functional_target_manifest",
    "functional_source_policy",
    "target_loader_implementation",
    "target_audit_runner",
}
_DATASET_HASH_PATHS = {
    "participants_tsv": "participants.tsv",
    "measurement_csv": "clinical_data/measurement.csv",
    "observation_csv": "clinical_data/observation.csv",
    "visit_occurrence_csv": "clinical_data/visit_occurrence.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=_strict_object)
    if not isinstance(value, Mapping):
        raise TypeError("disease target contract must be a JSON object")
    return value


def _safe_count(value: int) -> int | str:
    value = int(value)
    if value == 0 or value >= SMALL_CELL_THRESHOLD:
        return value
    return f"<{SMALL_CELL_THRESHOLD}"


def validate_disease_target_source_policy(
    project_root: str | Path,
    policy_path: str | Path | None = None,
) -> tuple[Mapping[str, Any], str]:
    root = Path(project_root).resolve()
    path = (root / POLICY_NAME).resolve() if policy_path is None else Path(policy_path).resolve()
    if path != (root / POLICY_NAME).resolve():
        raise ValueError("disease target policy must be the canonical project artifact")
    policy = _load_json(path)
    if (
        policy.get("schema_version") != POLICY_SCHEMA_VERSION
        or policy.get("status") != "frozen_before_disease_target_value_access"
    ):
        raise ValueError("disease target policy is not frozen")
    scope = policy.get("scope", {})
    if (
        scope.get("allowed_recommended_splits") != ["train", "val"]
        or scope.get("official_test_targets_allowed") is not False
        or scope.get("one_row_per_patient") is not True
        or scope.get("patient_level_values_may_be_serialized") is not False
        or scope.get("aggregate_only_reporting") is not True
    ):
        raise ValueError("disease target scope differs")

    bindings = policy.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != _REQUIRED_BINDINGS:
        raise ValueError("disease target bindings differ")
    for label, raw_binding in bindings.items():
        if not isinstance(raw_binding, Mapping) or set(raw_binding) != {"file", "sha256"}:
            raise ValueError(f"disease target binding is malformed: {label}")
        name = raw_binding.get("file")
        digest = raw_binding.get("sha256")
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"disease target binding path is invalid: {label}")
        bound = (root / name).resolve()
        if not isinstance(digest, str) or len(digest) != 64 or _sha256(bound) != digest:
            raise ValueError(f"disease target binding hash differs: {label}")

    declarations = policy.get("targets")
    if not isinstance(declarations, list):
        raise ValueError("disease target declarations are missing")
    columns = tuple(str(item.get("column", "")) for item in declarations)
    if columns != TARGET_COLUMNS or len(columns) != len(set(columns)):
        raise ValueError("disease target columns or order differ")
    for item in declarations:
        column = str(item.get("column"))
        expected_task = (
            "continuous"
            if column
            in {"raw_hba1c", "raw_ldl_cholesterol", "uacr", "fun_moca_total_score"}
            else "binary"
        )
        expected_status = (
            "blocked_by_unauthenticated_urine_units"
            if column in BLOCKED_TARGET_COLUMNS
            else "eligible_for_audit"
        )
        if item.get("task") != expected_task or item.get("execution_status") != expected_status:
            raise ValueError(f"disease target task/status differs: {column}")
    gates = policy.get("eligibility_gates", {})
    if gates != {
        "binary": {
            "minimum_total_events": 50,
            "minimum_total_nonevents": 50,
            "minimum_each_class_per_outer_test_fold": 10,
        },
        "continuous": {
            "minimum_total_observed": 200,
            "minimum_observed_per_outer_test_fold": 20,
            "nonzero_outer_training_variance_required": True,
        },
        "minimum_disclosable_cell_count": 10,
        "gate_fail_action": (
            "omit endpoint with prespecified reason; do not merge outcomes or change thresholds"
        ),
    }:
        raise ValueError("disease target eligibility gates differ")
    privacy = policy.get("privacy", {})
    if (
        privacy.get("patient_derived_processing") != "local_only"
        or privacy.get("patient_rows_identifiers_targets_or_predictions_emitted") is not False
        or privacy.get("small_cells_suppressed_below") != SMALL_CELL_THRESHOLD
    ):
        raise ValueError("disease target privacy contract differs")
    return policy, _sha256(path)


def _verify_dataset_hashes(dataset_root: Path, policy: Mapping[str, Any]) -> dict[str, str]:
    expected = policy.get("dataset_source_hashes")
    if not isinstance(expected, Mapping) or set(expected) != set(_DATASET_HASH_PATHS):
        raise ValueError("disease target dataset hash contract differs")
    observed: dict[str, str] = {}
    for label, relative in _DATASET_HASH_PATHS.items():
        digest = _sha256(dataset_root / relative)
        if digest != expected[label]:
            raise ValueError(f"disease target source hash differs: {label}")
        observed[label] = digest
    return observed


@dataclass(frozen=True, repr=False)
class DiseaseTargetMatrix:
    columns: tuple[str, ...]
    tasks: tuple[str, ...]
    execution_statuses: tuple[str, ...]
    values: np.ndarray
    observed_mask: np.ndarray
    patient_id_order_sha256: str
    source_policy_sha256: str
    source_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.columns != TARGET_COLUMNS or len(self.tasks) != len(self.columns):
            raise ValueError("disease target schema differs")
        if len(self.execution_statuses) != len(self.columns):
            raise ValueError("disease target status schema differs")
        if self.values.ndim != 2 or self.values.shape[1] != len(self.columns):
            raise ValueError("disease target matrix has the wrong shape")
        if self.observed_mask.shape != self.values.shape or self.observed_mask.dtype != np.bool_:
            raise TypeError("disease target observation mask is malformed")
        if not np.isfinite(self.values).all() or np.any(self.values[~self.observed_mask] != 0):
            raise ValueError("disease target missing values must be finite physical zeroes")
        for index, task in enumerate(self.tasks):
            observed = self.values[self.observed_mask[:, index], index]
            if task == "binary" and not set(np.unique(observed)).issubset({0.0, 1.0}):
                raise ValueError(f"binary disease target is not zero/one: {self.columns[index]}")
        for digest in (self.patient_id_order_sha256, self.source_policy_sha256):
            if len(digest) != 64:
                raise ValueError("disease target provenance hash is invalid")

    def __repr__(self) -> str:
        return (
            "DiseaseTargetMatrix(patient_data_redacted=True, "
            f"patients={self.values.shape[0]}, targets={self.values.shape[1]})"
        )

    def column(self, name: str) -> tuple[np.ndarray, np.ndarray, str, str]:
        try:
            index = self.columns.index(str(name))
        except ValueError as error:
            raise KeyError(name) from error
        return (
            self.values[:, index],
            self.observed_mask[:, index],
            self.tasks[index],
            self.execution_statuses[index],
        )

    def aggregate_summary(self, policy: Mapping[str, Any]) -> dict[str, Any]:
        binary_gate = policy["eligibility_gates"]["binary"]
        continuous_gate = policy["eligibility_gates"]["continuous"]
        targets: dict[str, Any] = {}
        for index, column in enumerate(self.columns):
            observed = self.observed_mask[:, index]
            count = int(observed.sum())
            status = self.execution_statuses[index]
            record: dict[str, Any] = {
                "task": self.tasks[index],
                "source_execution_status": status,
                "observed_count": _safe_count(count),
            }
            if status != "eligible_for_audit":
                record.update({"total_gate_pass": False, "gate_reason": status})
            elif self.tasks[index] == "binary":
                events = int(self.values[observed, index].sum())
                nonevents = count - events
                passed = (
                    events >= int(binary_gate["minimum_total_events"])
                    and nonevents >= int(binary_gate["minimum_total_nonevents"])
                )
                record.update(
                    {
                        "event_count": _safe_count(events),
                        "nonevent_count": _safe_count(nonevents),
                        "total_gate_pass": bool(passed),
                        "gate_reason": "passed" if passed else "insufficient_total_class_count",
                    }
                )
            else:
                passed = count >= int(continuous_gate["minimum_total_observed"])
                record.update(
                    {
                        "total_gate_pass": bool(passed),
                        "gate_reason": "passed" if passed else "insufficient_total_observed",
                    }
                )
            targets[column] = record
        return {
            "schema_version": "patient-atlas-v5-disease-target-audit-v1",
            "patient_count": int(self.values.shape[0]),
            "target_count": len(self.columns),
            "patient_id_order_sha256": self.patient_id_order_sha256,
            "source_policy_sha256": self.source_policy_sha256,
            "source_hashes": dict(sorted(self.source_hashes.items())),
            "targets": targets,
            "fold_specific_gates_evaluated": False,
            "official_test_targets_loaded": False,
            "patient_rows_emitted": False,
            "patient_identifiers_emitted": False,
            "target_values_emitted": False,
        }


def _derive_measured_hypertension(
    systolic: np.ndarray,
    systolic_observed: np.ndarray,
    diastolic: np.ndarray,
    diastolic_observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    systolic = np.asarray(systolic, dtype=np.float64)
    diastolic = np.asarray(diastolic, dtype=np.float64)
    systolic_observed = np.asarray(systolic_observed, dtype=bool)
    diastolic_observed = np.asarray(diastolic_observed, dtype=bool)
    if not (
        systolic.shape
        == systolic_observed.shape
        == diastolic.shape
        == diastolic_observed.shape
    ):
        raise ValueError("blood-pressure target inputs are misaligned")
    positive = (systolic_observed & (systolic >= 130.0)) | (
        diastolic_observed & (diastolic >= 80.0)
    )
    negative = (
        systolic_observed
        & diastolic_observed
        & (systolic < 130.0)
        & (diastolic < 80.0)
    )
    observed = positive | negative
    return positive.astype(np.float64), observed


def _load_binary_observations(
    *,
    dataset_root: Path,
    participants: pd.DataFrame,
    visits: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    rows = pd.read_csv(
        dataset_root / "clinical_data" / "observation.csv",
        usecols=(
            "person_id",
            "observation_date",
            "visit_occurrence_id",
            "observation_source_value",
            "value_as_number",
        ),
        low_memory=False,
    )
    rows["source"] = (
        rows["observation_source_value"]
        .fillna("")
        .astype(str)
        .str.split(",", n=1)
        .str[0]
        .str.strip()
    )
    rows = rows.loc[rows["source"].isin(OBSERVATION_TARGET_COLUMNS)].copy()
    rows = _index_linked_rows(
        rows,
        participants,
        visits,
        source_date_column="observation_date",
    )
    rows["numeric"] = pd.to_numeric(rows["value_as_number"], errors="coerce")
    rows = rows.loc[rows["numeric"].isin((0.0, 1.0))]
    grouped = rows.groupby(["person_id", "source"], sort=False)["numeric"]
    if bool((grouped.size() > 1).any()) or bool((grouped.nunique() > 1).any()):
        raise ValueError("disease observation target is duplicated or conflicting")
    aggregated = grouped.first().reset_index()
    values = np.zeros((len(participants), len(OBSERVATION_TARGET_COLUMNS)), dtype=np.float64)
    observed = np.zeros_like(values, dtype=bool)
    patient_index = {str(value): index for index, value in enumerate(participants["person_id"])}
    target_index = {value: index for index, value in enumerate(OBSERVATION_TARGET_COLUMNS)}
    for row in aggregated.itertuples(index=False):
        i = patient_index[str(row.person_id)]
        j = target_index[str(row.source)]
        values[i, j] = float(row.numeric)
        observed[i, j] = True
    return values, observed


def _load_moca_total(
    *,
    dataset_root: Path,
    participants: pd.DataFrame,
    visits: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    rows = pd.read_csv(
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
    rows["source"] = (
        rows["measurement_source_value"]
        .fillna("")
        .astype(str)
        .str.split(",", n=1)
        .str[0]
        .str.strip()
    )
    rows = rows.loc[rows["source"] == "moca_total_score"].copy()
    rows = _index_linked_rows(
        rows,
        participants,
        visits,
        source_date_column="measurement_date",
    )
    rows["numeric"] = pd.to_numeric(rows["value_as_number"], errors="coerce")
    rows = rows.loc[np.isfinite(rows["numeric"])].copy()
    grouped = rows.groupby("person_id", sort=False)
    timing = grouped.agg(
        measurement_dates=("measurement_date", "nunique"),
        visit_ids=("visit_occurrence_id", "nunique"),
    )
    if bool((timing["measurement_dates"] > 1).any() or (timing["visit_ids"] > 1).any()):
        raise ValueError("repeated MoCA targets cross date or visit boundaries")
    aggregated = grouped["numeric"].mean()
    values = np.zeros(len(participants), dtype=np.float64)
    observed = np.zeros(len(participants), dtype=bool)
    patient_index = {str(value): index for index, value in enumerate(participants["person_id"])}
    for patient_id, value in aggregated.items():
        if np.isfinite(value) and 0.0 <= float(value) <= 30.0:
            index = patient_index[str(patient_id)]
            values[index] = float(value)
            observed[index] = True
    return values, observed


def load_development_disease_targets(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    cohort: ExploratoryRawCohort,
) -> tuple[DiseaseTargetMatrix, Mapping[str, Any]]:
    """Load aligned train/validation targets; refuse every official-test identity."""

    if not isinstance(cohort, ExploratoryRawCohort):
        raise TypeError("disease targets require an authenticated exploratory cohort")
    root = Path(project_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    policy, policy_sha256 = validate_disease_target_source_policy(root)
    source_hashes = _verify_dataset_hashes(dataset_root, policy)
    for label in source_hashes:
        if cohort.source_hashes.get(label) != source_hashes[label]:
            raise ValueError(f"disease target and cohort source differ: {label}")

    participants = pd.read_csv(
        dataset_root / "participants.tsv",
        sep="\t",
        usecols=("person_id", "study_group", "study_visit_date", "recommended_split"),
    )
    participants["person_id"] = participants["person_id"].astype(str)
    if not participants["person_id"].is_unique:
        raise ValueError("disease target participant identities are duplicated")
    available = set(participants["person_id"])
    if set(cohort.patient_ids) - available:
        raise ValueError("disease target source lacks cohort identities")
    participants = (
        participants.set_index("person_id", verify_integrity=True)
        .loc[list(cohort.patient_ids)]
        .reset_index()
    )
    if set(participants["recommended_split"].astype(str)) - {"train", "val"}:
        raise ValueError("disease target loader refuses official-test identities")
    participants["study_visit_date"] = pd.to_datetime(
        participants["study_visit_date"], errors="coerce", format="mixed"
    )
    if not bool(participants["study_visit_date"].notna().all()):
        raise ValueError("disease target visit dates are invalid")

    declarations = policy["targets"]
    tasks = tuple(str(item["task"]) for item in declarations)
    statuses = tuple(str(item["execution_status"]) for item in declarations)
    values = np.zeros((len(cohort.patient_ids), len(TARGET_COLUMNS)), dtype=np.float64)
    observed = np.zeros_like(values, dtype=bool)
    column_index = {column: index for index, column in enumerate(TARGET_COLUMNS)}

    def assign(column: str, data: np.ndarray, mask: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if data.shape != (len(cohort.patient_ids),) or mask.shape != data.shape:
            raise ValueError(f"disease target assignment is misaligned: {column}")
        index = column_index[column]
        values[mask, index] = data[mask]
        observed[:, index] = mask

    feature_index = {name: index for index, name in enumerate(cohort.feature_names)}
    for column, feature in (
        ("raw_hba1c", "hba1c"),
        ("raw_ldl_cholesterol", "ldl_cholesterol"),
    ):
        index = feature_index[feature]
        assign(column, cohort.blood_values[:, index], cohort.blood_observed_mask[:, index])

    severity = participants["study_group"].map(_STUDY_GROUP_SEVERITY)
    if severity.isna().any():
        raise ValueError("disease target study-group vocabulary differs")
    severity_values = severity.to_numpy(dtype=np.int64)
    assign("any_dysglycemia", (severity_values >= 1).astype(float), np.ones(len(severity), bool))
    assign("any_dm", (severity_values >= 2).astype(float), np.ones(len(severity), bool))

    systolic_index = feature_index["vit_sysbp_vsorres"]
    diastolic_index = feature_index["vit_diabp_vsorres"]
    htn, htn_observed = _derive_measured_hypertension(
        cohort.blood_values[:, systolic_index],
        cohort.blood_observed_mask[:, systolic_index],
        cohort.blood_values[:, diastolic_index],
        cohort.blood_observed_mask[:, diastolic_index],
    )
    assign("htn_measured", htn, htn_observed)

    visits = _load_visits(dataset_root)
    observation_values, observation_masks = _load_binary_observations(
        dataset_root=dataset_root,
        participants=participants,
        visits=visits,
    )
    for source_index, column in enumerate(OBSERVATION_TARGET_COLUMNS):
        assign(column, observation_values[:, source_index], observation_masks[:, source_index])

    moca, moca_observed = _load_moca_total(
        dataset_root=dataset_root,
        participants=participants,
        visits=visits,
    )
    assign("fun_moca_total_score", moca, moca_observed)
    assign("moca_lt26", (moca < 26.0).astype(float), moca_observed)

    targets = DiseaseTargetMatrix(
        columns=TARGET_COLUMNS,
        tasks=tasks,
        execution_statuses=statuses,
        values=values,
        observed_mask=observed,
        patient_id_order_sha256=hash_json(list(cohort.patient_ids)),
        source_policy_sha256=policy_sha256,
        source_hashes=MappingProxyType(dict(sorted(source_hashes.items()))),
    )
    return targets, policy


__all__ = [
    "BLOCKED_TARGET_COLUMNS",
    "DiseaseTargetMatrix",
    "OBSERVATION_TARGET_COLUMNS",
    "POLICY_NAME",
    "POLICY_SCHEMA_VERSION",
    "TARGET_COLUMNS",
    "load_development_disease_targets",
    "validate_disease_target_source_policy",
]
