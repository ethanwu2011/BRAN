"""Deterministic, privacy-gated outer folds for the V5 disease benchmark.

Patient identities and target values remain in local memory.  The only
serializable fold artifact is an aggregate audit containing cryptographic set
hashes, fold sizes, and small-cell-suppressed eligibility counts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from patient_atlas_disease_targets import (
    BLOCKED_TARGET_COLUMNS,
    DiseaseTargetMatrix,
    TARGET_COLUMNS,
)
from patient_atlas_raw_audit import SMALL_CELL_THRESHOLD


POLICY_NAME = "PATIENT_ATLAS_V5_DISEASE_FOLD_POLICY_V1.json"
POLICY_SCHEMA_VERSION = "patient-atlas-v5-disease-fold-policy-v1"
FOLD_AUDIT_SCHEMA_VERSION = "patient-atlas-v5-disease-fold-audit-v1"
_REQUIRED_BINDINGS = {
    "disease_utility_registry",
    "target_source_policy",
    "target_availability_audit",
    "target_loader_implementation",
    "fold_implementation",
    "fold_audit_runner",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        raise TypeError("disease fold contract must be a JSON object")
    return value


def _stable_digest(*parts: object) -> bytes:
    payload = "\x1f".join(str(value) for value in parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _safe_count(value: int) -> int | str:
    value = int(value)
    if value == 0 or value >= SMALL_CELL_THRESHOLD:
        return value
    return f"<{SMALL_CELL_THRESHOLD}"


def _largest_remainder_capacities(size: int, n_folds: int, *, seed: int, salt: str) -> np.ndarray:
    base = size // n_folds
    capacities = np.full(n_folds, base, dtype=np.int64)
    remainder = size - base * n_folds
    order = sorted(
        range(n_folds),
        key=lambda fold: (_stable_digest(salt, seed, "capacity", fold), fold),
    )
    for fold in order[:remainder]:
        capacities[fold] += 1
    return capacities


@dataclass(frozen=True, repr=False)
class DiseaseFoldMap:
    """Restricted in-memory patient-to-fold assignment."""

    patient_ids: tuple[str, ...]
    fold_ids: tuple[int, ...]
    n_folds: int
    seed: int
    salt: str
    balancing_label_count: int

    def __post_init__(self) -> None:
        if not self.patient_ids or len(self.patient_ids) != len(self.fold_ids):
            raise ValueError("disease fold IDs and assignments must be aligned")
        if len(self.patient_ids) != len(set(self.patient_ids)):
            raise ValueError("disease fold map must have one row per patient")
        if self.n_folds < 2 or set(self.fold_ids) != set(range(self.n_folds)):
            raise ValueError("disease fold map must populate every fold")
        if any(value < 0 or value >= self.n_folds for value in self.fold_ids):
            raise ValueError("disease fold assignment is outside its declared range")
        if self.balancing_label_count < 1:
            raise ValueError("disease fold map has no balancing labels")

    def __repr__(self) -> str:
        return (
            "DiseaseFoldMap(patient_data_redacted=True, "
            f"patients={len(self.patient_ids)}, folds={self.n_folds})"
        )

    def as_mapping(self) -> Mapping[str, int]:
        return MappingProxyType(dict(zip(self.patient_ids, self.fold_ids)))

    def assignments_for(self, patient_ids: Sequence[str]) -> np.ndarray:
        normalized = tuple(str(value) for value in patient_ids)
        if len(normalized) != len(set(normalized)):
            raise ValueError("disease fold evaluation rows must be unique")
        if set(normalized) != set(self.patient_ids):
            raise ValueError("disease fold map and patient set differ")
        mapping = self.as_mapping()
        return np.asarray([mapping[value] for value in normalized], dtype=np.int64)

    @property
    def patient_set_sha256(self) -> str:
        return _hash_json(sorted(self.patient_ids))

    @property
    def assignment_sha256(self) -> str:
        return _hash_json(
            [[patient, int(fold)] for patient, fold in zip(self.patient_ids, self.fold_ids)]
        )


def validate_disease_fold_policy(
    project_root: str | Path,
    policy_path: str | Path | None = None,
) -> tuple[Mapping[str, Any], str]:
    root = Path(project_root).resolve()
    canonical = (root / POLICY_NAME).resolve()
    path = canonical if policy_path is None else Path(policy_path).resolve()
    if path != canonical:
        raise ValueError("disease fold policy must be the canonical project artifact")
    policy = _load_json(path)
    if (
        policy.get("schema_version") != POLICY_SCHEMA_VERSION
        or policy.get("status") != "frozen_before_fold_assignment_and_disease_scores"
    ):
        raise ValueError("disease fold policy is not frozen")

    split = policy.get("split", {})
    if split != {
        "n_folds": 5,
        "seed": 1701,
        "salt": "patient-atlas-v5-disease-outer-v1",
        "patient_level": True,
        "created_before_endpoint_specific_filtering": True,
        "same_assignment_for_all_arms_endpoints_and_availability_patterns": True,
        "official_test_patients_allowed": False,
    }:
        raise ValueError("disease fold split contract differs")

    balancing = policy.get("balancing", {})
    binary = balancing.get("binary_target_columns")
    continuous = balancing.get("continuous_target_columns")
    if (
        balancing.get("algorithm")
        != "deterministic_greedy_multilabel_rarest_first_v1"
        or balancing.get("site_one_hot") is not True
        or balancing.get("binary_labels") != ["event", "nonevent", "missing"]
        or balancing.get("continuous_labels")
        != ["observed", "missing", "five_quantile_bins"]
        or balancing.get("continuous_quantile_bins") != 5
        or not isinstance(binary, list)
        or not isinstance(continuous, list)
        or len(binary) != len(set(binary))
        or len(continuous) != len(set(continuous))
        or set(binary) & set(continuous)
        or set(binary) | set(continuous) != set(TARGET_COLUMNS) - set(BLOCKED_TARGET_COLUMNS)
    ):
        raise ValueError("disease fold balancing contract differs")

    bindings = policy.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != _REQUIRED_BINDINGS:
        raise ValueError("disease fold bindings differ")
    for label, raw_binding in bindings.items():
        if not isinstance(raw_binding, Mapping) or set(raw_binding) != {"file", "sha256"}:
            raise ValueError(f"disease fold binding is malformed: {label}")
        name = raw_binding.get("file")
        digest = raw_binding.get("sha256")
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"disease fold binding path is invalid: {label}")
        bound = (root / name).resolve()
        if not isinstance(digest, str) or len(digest) != 64 or _sha256(bound) != digest:
            raise ValueError(f"disease fold binding hash differs: {label}")

    availability = _load_json(
        root / str(bindings["target_availability_audit"]["file"])
    )
    passing = availability.get("execution_decision", {}).get(
        "total_gate_passing_target_columns"
    )
    if (
        availability.get("status")
        != "source_and_total_eligibility_audit_complete_fold_freeze_required"
        or not isinstance(passing, list)
        or set(passing) != set(binary) | set(continuous)
        or availability.get("privacy", {}).get("patient_identifiers_emitted") is not False
    ):
        raise ValueError("disease target availability audit is incompatible")

    gates = policy.get("eligibility_gates", {})
    if gates != {
        "binary_minimum_each_class_per_outer_test_fold": 10,
        "continuous_minimum_observed_per_outer_test_fold": 20,
        "continuous_nonzero_outer_training_variance_required": True,
        "failure_action": "omit_endpoint_without_changing_folds_or_thresholds",
    }:
        raise ValueError("disease fold eligibility gates differ")
    privacy = policy.get("privacy", {})
    if privacy != {
        "patient_derived_processing": "local_only",
        "patient_ids_targets_assignments_and_fold_rows_serialized": False,
        "aggregate_counts_small_cell_suppressed_below": 10,
        "fold_identity_represented_only_by_sha256_set_hashes": True,
    }:
        raise ValueError("disease fold privacy contract differs")
    return policy, _sha256(path)


def _append_label(
    columns: list[np.ndarray],
    names: list[str],
    name: str,
    values: np.ndarray,
) -> None:
    label = np.asarray(values, dtype=bool)
    if label.ndim != 1:
        raise ValueError("disease balancing label must be one-dimensional")
    if bool(label.any()):
        columns.append(label)
        names.append(str(name))


def _balancing_matrix(
    *,
    patient_ids: Sequence[str],
    site_ids: Sequence[str],
    targets: DiseaseTargetMatrix,
    policy: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[str, ...]]:
    n = len(patient_ids)
    if len(site_ids) != n or targets.values.shape[0] != n:
        raise ValueError("disease fold inputs are not aligned")
    columns: list[np.ndarray] = []
    names: list[str] = []
    sites = np.asarray(tuple(str(value) for value in site_ids), dtype=object)
    if any(not value for value in sites):
        raise ValueError("disease fold site is empty")
    for site in sorted(set(sites.tolist())):
        _append_label(columns, names, f"site::{site}", sites == site)

    balancing = policy["balancing"]
    for column in balancing["binary_target_columns"]:
        values, observed, task, status = targets.column(str(column))
        if task != "binary" or status != "eligible_for_audit":
            raise ValueError(f"binary disease balancing target is ineligible: {column}")
        _append_label(columns, names, f"{column}::event", observed & (values == 1.0))
        _append_label(columns, names, f"{column}::nonevent", observed & (values == 0.0))
        _append_label(columns, names, f"{column}::missing", ~observed)

    quantile_count = int(balancing["continuous_quantile_bins"])
    for column in balancing["continuous_target_columns"]:
        values, observed, task, status = targets.column(str(column))
        if task != "continuous" or status != "eligible_for_audit":
            raise ValueError(f"continuous disease balancing target is ineligible: {column}")
        _append_label(columns, names, f"{column}::observed", observed)
        _append_label(columns, names, f"{column}::missing", ~observed)
        observed_values = values[observed]
        if observed_values.size < quantile_count:
            raise ValueError(f"continuous disease balancing target is too sparse: {column}")
        boundaries = np.unique(
            np.quantile(
                observed_values,
                np.arange(1, quantile_count, dtype=np.float64) / quantile_count,
                method="linear",
            )
        )
        bins = np.searchsorted(boundaries, values, side="right")
        for bin_index in range(len(boundaries) + 1):
            _append_label(
                columns,
                names,
                f"{column}::quantile-{bin_index}",
                observed & (bins == bin_index),
            )
    if not columns:
        raise ValueError("disease fold balancing matrix is empty")
    matrix = np.column_stack(columns).astype(bool, copy=False)
    if matrix.shape[0] != n or bool((matrix.sum(axis=1) == 0).any()):
        raise ValueError("every disease fold patient must have a balancing stratum")
    return matrix, tuple(names)


def make_disease_fold_map(
    *,
    patient_ids: Sequence[str],
    site_ids: Sequence[str],
    targets: DiseaseTargetMatrix,
    policy: Mapping[str, Any],
) -> DiseaseFoldMap:
    """Create one deterministic site/outcome-balanced fold map before filtering."""

    normalized_ids = tuple(str(value) for value in patient_ids)
    if not normalized_ids or len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("disease fold patients must be nonempty and unique")
    if targets.patient_id_order_sha256 != _hash_json(list(normalized_ids)):
        raise ValueError("disease target order and fold patient order differ")
    split = policy["split"]
    n_folds = int(split["n_folds"])
    seed = int(split["seed"])
    salt = str(split["salt"])
    if len(normalized_ids) < n_folds:
        raise ValueError("disease fold patient count is smaller than fold count")
    labels, label_names = _balancing_matrix(
        patient_ids=normalized_ids,
        site_ids=site_ids,
        targets=targets,
        policy=policy,
    )
    support = labels.sum(axis=0).astype(np.float64)
    if bool((support <= 0).any()):
        raise ValueError("disease balancing label has zero support")
    capacities = _largest_remainder_capacities(
        len(normalized_ids), n_folds, seed=seed, salt=salt
    )
    desired = np.outer(capacities / float(len(normalized_ids)), support)
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    fold_label_counts = np.zeros((n_folds, labels.shape[1]), dtype=np.float64)
    assignment = np.full(len(normalized_ids), -1, dtype=np.int64)

    inverse_support = 1.0 / support
    priority = labels @ inverse_support
    burden = labels.sum(axis=1)
    order = sorted(
        range(len(normalized_ids)),
        key=lambda index: (
            -float(priority[index]),
            -int(burden[index]),
            _stable_digest(salt, seed, "patient-order", normalized_ids[index]),
            normalized_ids[index],
        ),
    )
    for patient_index in order:
        active = labels[patient_index]
        candidates = np.flatnonzero(fold_sizes < capacities)
        if candidates.size == 0:
            raise RuntimeError("disease fold capacity was exhausted early")
        scored: list[tuple[float, bytes, int]] = []
        for fold in candidates.tolist():
            normalized_deficit = (
                desired[fold, active] - fold_label_counts[fold, active]
            ) / np.maximum(desired[fold, active], 1.0)
            size_need = (capacities[fold] - fold_sizes[fold]) / max(capacities[fold], 1)
            score = float(normalized_deficit.sum() + 0.05 * size_need)
            scored.append(
                (
                    -score,
                    _stable_digest(
                        salt,
                        seed,
                        "fold-tie",
                        normalized_ids[patient_index],
                        fold,
                    ),
                    int(fold),
                )
            )
        _, _, chosen = min(scored)
        assignment[patient_index] = chosen
        fold_sizes[chosen] += 1
        fold_label_counts[chosen] += labels[patient_index]
    if bool((assignment < 0).any()) or not np.array_equal(fold_sizes, capacities):
        raise RuntimeError("disease fold assignment did not meet exact capacities")

    mapping = dict(zip(normalized_ids, assignment.tolist()))
    ordered_ids = tuple(sorted(mapping))
    return DiseaseFoldMap(
        patient_ids=ordered_ids,
        fold_ids=tuple(int(mapping[value]) for value in ordered_ids),
        n_folds=n_folds,
        seed=seed,
        salt=salt,
        balancing_label_count=len(label_names),
    )


def audit_disease_fold_eligibility(
    *,
    fold_map: DiseaseFoldMap,
    patient_ids: Sequence[str],
    site_ids: Sequence[str],
    targets: DiseaseTargetMatrix,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an identifier-free, small-cell-suppressed fold audit."""

    normalized_ids = tuple(str(value) for value in patient_ids)
    assignments = fold_map.assignments_for(normalized_ids)
    if len(site_ids) != len(normalized_ids) or targets.values.shape[0] != len(normalized_ids):
        raise ValueError("disease fold audit inputs are not aligned")
    gates = policy["eligibility_gates"]
    binary_minimum = int(gates["binary_minimum_each_class_per_outer_test_fold"])
    continuous_minimum = int(gates["continuous_minimum_observed_per_outer_test_fold"])
    sites = np.asarray(tuple(str(value) for value in site_ids), dtype=object)

    fold_records: list[dict[str, Any]] = []
    for fold in range(fold_map.n_folds):
        selected = assignments == fold
        ids = [normalized_ids[index] for index in np.flatnonzero(selected)]
        site_counts = {
            str(site): _safe_count(int(np.sum(sites[selected] == site)))
            for site in sorted(set(sites.tolist()))
        }
        fold_records.append(
            {
                "fold": fold,
                "patient_count": int(selected.sum()),
                "patient_set_sha256": _hash_json(sorted(ids)),
                "site_counts": site_counts,
            }
        )

    target_records: dict[str, Any] = {}
    passing: list[str] = []
    for index, column in enumerate(targets.columns):
        task = targets.tasks[index]
        status = targets.execution_statuses[index]
        record: dict[str, Any] = {
            "task": task,
            "source_execution_status": status,
            "folds": [],
        }
        if status != "eligible_for_audit":
            record.update({"fold_gate_pass": False, "gate_reason": status})
            target_records[column] = record
            continue
        observed = targets.observed_mask[:, index]
        values = targets.values[:, index]
        target_pass = True
        for fold in range(fold_map.n_folds):
            test = assignments == fold
            train = ~test
            test_observed = test & observed
            fold_record: dict[str, Any] = {
                "fold": fold,
                "observed_count": _safe_count(int(test_observed.sum())),
            }
            if task == "binary":
                events = int(values[test_observed].sum())
                nonevents = int(test_observed.sum()) - events
                passed = events >= binary_minimum and nonevents >= binary_minimum
                fold_record.update(
                    {
                        "event_count": _safe_count(events),
                        "nonevent_count": _safe_count(nonevents),
                        "gate_pass": bool(passed),
                    }
                )
            elif task == "continuous":
                train_values = values[train & observed]
                nonzero_variance = bool(
                    train_values.size >= 2
                    and np.isfinite(np.var(train_values, ddof=0))
                    and np.var(train_values, ddof=0) > 0.0
                )
                passed = int(test_observed.sum()) >= continuous_minimum and nonzero_variance
                fold_record.update(
                    {
                        "outer_training_nonzero_variance": nonzero_variance,
                        "gate_pass": bool(passed),
                    }
                )
            else:
                raise ValueError(f"unknown disease target task: {task}")
            target_pass = target_pass and bool(passed)
            record["folds"].append(fold_record)
        record.update(
            {
                "fold_gate_pass": bool(target_pass),
                "gate_reason": "passed" if target_pass else "per_fold_eligibility_failed",
            }
        )
        if target_pass:
            passing.append(column)
        target_records[column] = record

    return {
        "schema_version": FOLD_AUDIT_SCHEMA_VERSION,
        "patient_count": len(normalized_ids),
        "n_folds": fold_map.n_folds,
        "seed": fold_map.seed,
        "salt": fold_map.salt,
        "balancing_label_count": fold_map.balancing_label_count,
        "patient_set_sha256": fold_map.patient_set_sha256,
        "assignment_sha256": fold_map.assignment_sha256,
        "folds": fold_records,
        "targets": target_records,
        "fold_gate_passing_target_columns": sorted(passing),
        "fold_map_created_before_endpoint_specific_filtering": True,
        "same_fold_map_for_all_arms_endpoints_and_patterns": True,
        "official_test_targets_loaded": False,
        "patient_ids_target_values_assignments_or_rows_emitted": False,
    }


__all__ = [
    "DiseaseFoldMap",
    "FOLD_AUDIT_SCHEMA_VERSION",
    "POLICY_NAME",
    "POLICY_SCHEMA_VERSION",
    "audit_disease_fold_eligibility",
    "make_disease_fold_map",
    "validate_disease_fold_policy",
]
