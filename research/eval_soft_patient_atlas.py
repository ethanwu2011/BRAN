"""Leakage-safe nested evaluation for the soft patient atlas.

This module is deliberately data-source agnostic.  It accepts patient-level
feature arrays and synthetic or controlled-environment target arrays, but the
representation callback never receives a target.  Every outer fold owns one
fitted representation object; probes are tuned inside that fold and only
outer-test *losses* are aggregated across fold-specific coordinate bases.

Each declared missingness pattern is transformed once for outer train and test.
The corrupted outer-test state is scored both by the frozen both-present probe
and by a separately inner-tuned probe trained on the identically corrupted
outer-training state.  Family-wise patient-cluster bootstrap bounds decide
whether the tested pattern remains within its prespecified oracle margin.

The public report is aggregate-only.  Patient identifiers, predictions,
targets, and latent coordinates exist only in local memory while ``evaluate``
runs and are not fields of :class:`NestedEvaluationReport`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


ARM_BOTH = "both_atlas"
ARM_EYE = "eye_atlas"
ARM_BLOOD = "blood_clinical_atlas"
ARM_CONCAT = "tuned_complete_case_concat"
ARM_DEMOGRAPHICS = "demographics_only"
ARM_EYE_TOWER = "eye_tower_only"
ARM_RAW_BLOOD = "raw_blood_clinical_only"
ARM_FROZEN_BLOOD_TOWER = "frozen_blood_tower_only"
PRIMARY_ARMS = (
    ARM_BOTH,
    ARM_EYE,
    ARM_BLOOD,
    ARM_CONCAT,
    ARM_DEMOGRAPHICS,
    ARM_EYE_TOWER,
    ARM_RAW_BLOOD,
    ARM_FROZEN_BLOOD_TOWER,
)
CONCAT_EYE_DIMENSIONS = (16, 32, 64, 384)
BASE_STRATUM = "both_present"


def _as_read_only(array: np.ndarray, *, copy: bool = True) -> np.ndarray:
    value = np.array(array, copy=copy)
    value.setflags(write=False)
    return value


def _patient_hash(patient_ids: Sequence[str]) -> str:
    normalized = sorted(str(value) for value in patient_ids)
    payload = "\n".join(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _patient_json_hash(patient_ids: Sequence[str]) -> str:
    """Canonical identifier-set hash used by fold preprocessing provenance."""

    normalized = sorted(str(value) for value in patient_ids)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_digest(*parts: object) -> bytes:
    payload = "\x1f".join(str(value) for value in parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _largest_remainder_counts(size: int, fractions: Sequence[float]) -> list[int]:
    raw = [size * float(fraction) for fraction in fractions]
    counts = [math.floor(value) for value in raw]
    remaining = size - sum(counts)
    order = sorted(
        range(len(raw)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


@dataclass(frozen=True)
class CanonicalFoldMap:
    """Restricted in-memory patient-to-fold map.

    This object may contain identifiers and therefore must never be embedded in
    an aggregate report or deployment bundle.
    """

    patient_ids: tuple[str, ...]
    fold_ids: tuple[int, ...]
    n_folds: int
    seed: int
    salt: str

    def __post_init__(self) -> None:
        if len(self.patient_ids) != len(self.fold_ids) or not self.patient_ids:
            raise ValueError("fold map IDs and assignments must be nonempty and aligned")
        if len(self.patient_ids) != len(set(self.patient_ids)):
            raise ValueError("canonical fold map must have one entry per patient")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least two")
        if any(fold < 0 or fold >= self.n_folds for fold in self.fold_ids):
            raise ValueError("fold assignment lies outside [0, n_folds)")
        if set(self.fold_ids) != set(range(self.n_folds)):
            raise ValueError("every canonical fold must contain at least one patient")

    def as_mapping(self) -> Mapping[str, int]:
        return MappingProxyType(dict(zip(self.patient_ids, self.fold_ids)))

    def assignments_for(self, patient_ids: Sequence[str]) -> np.ndarray:
        mapping = self.as_mapping()
        normalized = tuple(str(value) for value in patient_ids)
        if len(normalized) != len(set(normalized)):
            raise ValueError("evaluation rows must be unique at patient level")
        if set(normalized) != set(self.patient_ids):
            raise ValueError("fold map and evaluation patient set differ")
        return np.asarray([mapping[value] for value in normalized], dtype=np.int64)


def make_canonical_fold_map(
    patient_ids: Sequence[str],
    site_ids: Sequence[str],
    *,
    n_folds: int,
    seed: int,
    salt: str = "soft-patient-atlas-outer-v1",
) -> CanonicalFoldMap:
    """Create a deterministic, site-balanced patient fold map without labels.

    Repeated rows are allowed at map-construction time so long as each patient
    has one unambiguous site; all repeated visits then receive the same fold.
    The evaluator itself consumes one patient-level row per patient.
    """

    if len(patient_ids) != len(site_ids) or not patient_ids:
        raise ValueError("patient_ids and site_ids must be nonempty and aligned")
    if not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("n_folds must be an integer of at least two")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not salt:
        raise ValueError("fold salt must not be empty")

    patient_site: dict[str, str] = {}
    for raw_patient, raw_site in zip(patient_ids, site_ids):
        patient = str(raw_patient)
        site = str(raw_site)
        if not patient or not site:
            raise ValueError("patient and site identifiers must not be empty")
        if patient in patient_site and patient_site[patient] != site:
            raise ValueError("one patient appears under multiple site strata")
        patient_site[patient] = site
    if len(patient_site) < n_folds:
        raise ValueError("number of unique patients is smaller than n_folds")

    by_site: dict[str, list[str]] = {}
    for patient, site in patient_site.items():
        by_site.setdefault(site, []).append(patient)

    assignment: dict[str, int] = {}
    for site in sorted(by_site):
        ordered = sorted(
            by_site[site],
            key=lambda patient: (
                _stable_digest(salt, seed, site, patient),
                patient,
            ),
        )
        offset = int.from_bytes(_stable_digest(salt, seed, site)[:4], "big") % n_folds
        for position, patient in enumerate(ordered):
            assignment[patient] = (position + offset) % n_folds

    # Very small site strata can leave a global fold empty.  Rebalance only in
    # that case, deterministically, without consulting outcomes.
    counts = [sum(value == fold for value in assignment.values()) for fold in range(n_folds)]
    for empty_fold in [fold for fold, count in enumerate(counts) if count == 0]:
        donor = max(range(n_folds), key=lambda fold: (counts[fold], -fold))
        if counts[donor] <= 1:
            raise ValueError("cannot populate every fold without emptying another fold")
        candidates = sorted(
            (patient for patient, fold in assignment.items() if fold == donor),
            key=lambda patient: (_stable_digest(salt, seed, "rebalance", patient), patient),
        )
        assignment[candidates[0]] = empty_fold
        counts[donor] -= 1
        counts[empty_fold] += 1

    ordered_ids = tuple(sorted(assignment))
    return CanonicalFoldMap(
        patient_ids=ordered_ids,
        fold_ids=tuple(assignment[value] for value in ordered_ids),
        n_folds=n_folds,
        seed=seed,
        salt=salt,
    )


@dataclass(frozen=True)
class TargetDefinition:
    id: str
    family: str
    family_weight: float
    within_family_weight: float
    kind: str = "continuous"

    def __post_init__(self) -> None:
        if not self.id or not self.family:
            raise ValueError("target ID and family must not be empty")
        if self.kind != "continuous":
            raise ValueError("confirmatory primary targets must be continuous")
        if self.family_weight <= 0 or self.within_family_weight <= 0:
            raise ValueError("target-family weights must be positive")


@dataclass(frozen=True)
class TargetManifest:
    schema_version: str
    targets: tuple[TargetDefinition, ...]
    minimum_complete_target_patients: int
    noninferiority_margin: float | None

    def __post_init__(self) -> None:
        if not self.schema_version or not self.targets:
            raise ValueError("target manifest must have a schema and at least one target")
        ids = [target.id for target in self.targets]
        if len(ids) != len(set(ids)):
            raise ValueError("target IDs must be unique")
        if self.minimum_complete_target_patients <= 0:
            raise ValueError("minimum target-patient count must be positive")
        if self.noninferiority_margin is not None and self.noninferiority_margin < 0:
            raise ValueError("noninferiority margin must be nonnegative or null")

        family_weights: dict[str, float] = {}
        within_totals: dict[str, float] = {}
        for target in self.targets:
            previous = family_weights.setdefault(target.family, target.family_weight)
            if not math.isclose(previous, target.family_weight, abs_tol=1e-12):
                raise ValueError("all targets in a family must declare one family weight")
            within_totals[target.family] = (
                within_totals.get(target.family, 0.0) + target.within_family_weight
            )
        if not math.isclose(sum(family_weights.values()), 1.0, abs_tol=1e-12):
            raise ValueError("family weights must sum to one")
        for family, total in within_totals.items():
            if not math.isclose(total, 1.0, abs_tol=1e-12):
                raise ValueError(f"within-family weights for {family!r} must sum to one")

    @property
    def target_ids(self) -> tuple[str, ...]:
        return tuple(target.id for target in self.targets)


def load_target_manifest(path: str | Path) -> TargetManifest:
    """Load and validate the frozen executable target manifest."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    families = payload.get("families")
    targets = payload.get("targets")
    eligibility = payload.get("eligibility")
    noninferiority = payload.get("noninferiority")
    if not isinstance(families, dict) or not isinstance(targets, list):
        raise ValueError("target manifest families/targets are malformed")
    if not isinstance(eligibility, dict) or not isinstance(noninferiority, dict):
        raise ValueError("target manifest eligibility/noninferiority is malformed")
    definitions: list[TargetDefinition] = []
    for target in targets:
        family = str(target["family"])
        if family not in families:
            raise ValueError(f"target references unknown family {family!r}")
        definitions.append(
            TargetDefinition(
                id=str(target["id"]),
                family=family,
                family_weight=float(families[family]["weight"]),
                within_family_weight=float(target["within_family_weight"]),
                kind=str(target["kind"]),
            )
        )
    return TargetManifest(
        schema_version=str(payload["schema_version"]),
        targets=tuple(definitions),
        minimum_complete_target_patients=int(
            eligibility["minimum_complete_target_patients"]
        ),
        noninferiority_margin=(
            None
            if noninferiority.get("margin") is None
            else float(noninferiority["margin"])
        ),
    )


@dataclass(frozen=True)
class EvaluationCohort:
    """Patient-level arrays held only inside the controlled evaluation process."""

    patient_ids: tuple[str, ...]
    site_ids: tuple[str, ...]
    features: Mapping[str, np.ndarray]
    demographics: np.ndarray
    demographic_mask: np.ndarray
    targets: np.ndarray
    target_ids: tuple[str, ...]
    target_mask: np.ndarray
    primary_patient_mask: np.ndarray

    def __post_init__(self) -> None:
        size = len(self.patient_ids)
        if size == 0 or len(self.site_ids) != size:
            raise ValueError("cohort patient/site IDs must be nonempty and aligned")
        if len(self.patient_ids) != len(set(self.patient_ids)):
            raise ValueError("nested evaluation requires one row per patient")
        if any(not str(value) for value in self.patient_ids + self.site_ids):
            raise ValueError("patient/site IDs must not be empty")
        if self.demographics.ndim != 2 or self.demographics.shape[0] != size:
            raise ValueError("demographics must have shape [patients, fields]")
        if self.demographic_mask.shape != self.demographics.shape:
            raise ValueError("demographic mask shape differs from demographics")
        if self.demographic_mask.dtype != np.bool_:
            raise TypeError("demographic mask must be boolean")
        if not bool(self.demographic_mask.all()):
            raise ValueError("confirmatory V1 rejects missing required demographics")
        if not np.isfinite(self.demographics).all():
            raise ValueError("observed demographics must be finite")
        if self.targets.ndim != 2 or self.targets.shape[0] != size:
            raise ValueError("targets must have shape [patients, targets]")
        if len(self.target_ids) != self.targets.shape[1] or len(self.target_ids) != len(
            set(self.target_ids)
        ):
            raise ValueError("target_ids must uniquely name every target column in order")
        if self.target_mask.shape != self.targets.shape or self.target_mask.dtype != np.bool_:
            raise TypeError("target mask must be boolean and match targets")
        if self.primary_patient_mask.shape != (size,) or self.primary_patient_mask.dtype != np.bool_:
            raise TypeError("primary_patient_mask must be boolean [patients]")
        if not bool(self.primary_patient_mask.any()):
            raise ValueError("primary evaluation set is empty")
        if not isinstance(self.features, Mapping) or not self.features:
            raise ValueError("at least one representation input feature is required")
        for name, values in self.features.items():
            if (
                not str(name)
                or not isinstance(values, np.ndarray)
                or values.ndim < 1
                or values.shape[0] != size
            ):
                raise ValueError("every named feature array must start with the patient axis")

    @property
    def size(self) -> int:
        return len(self.patient_ids)

    def feature_view(self, indices: np.ndarray) -> "FeatureView":
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("feature indices must be one-dimensional")
        if len(indices) and (indices.min() < 0 or indices.max() >= self.size):
            raise IndexError("feature-view index outside cohort")
        feature_subset = {
            str(name): _as_read_only(values[indices], copy=True)
            for name, values in self.features.items()
        }
        return FeatureView(
            patient_ids=tuple(self.patient_ids[index] for index in indices),
            site_ids=tuple(self.site_ids[index] for index in indices),
            features=MappingProxyType(feature_subset),
            demographics=_as_read_only(self.demographics[indices], copy=True),
            demographic_mask=_as_read_only(self.demographic_mask[indices], copy=True),
        )


@dataclass(frozen=True)
class FeatureView:
    """Outcome-free input passed to a representation fit/transform callback."""

    patient_ids: tuple[str, ...]
    site_ids: tuple[str, ...]
    features: Mapping[str, np.ndarray]
    demographics: np.ndarray
    demographic_mask: np.ndarray

    @property
    def size(self) -> int:
        return len(self.patient_ids)


@dataclass(frozen=True)
class FoldRepresentationProvenance:
    schema_version: str
    fold_key: str
    model_token: str
    outer_train_patient_hash: str
    fit_patient_hash: str
    validation_patient_hash: str
    calibration_patient_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "soft-patient-atlas-eval-fold-v1":
            raise ValueError("unknown evaluation-fold representation schema")
        text_fields = (
            self.fold_key,
            self.model_token,
            self.outer_train_patient_hash,
            self.fit_patient_hash,
            self.validation_patient_hash,
            self.calibration_patient_hash,
        )
        if any(not value for value in text_fields):
            raise ValueError("representation provenance fields must not be empty")


@dataclass(frozen=True)
class RepresentationFitRequest:
    """The only data a representation factory receives for one outer fold."""

    fold_key: str
    fit: FeatureView
    validation: FeatureView
    calibration: FeatureView
    expected_outer_train_patient_hash: str
    expected_outer_test_patient_id_hash: str | None = None
    expected_outer_test_patient_count: int | None = None

    def __post_init__(self) -> None:
        supplied = (
            self.expected_outer_test_patient_id_hash is not None,
            self.expected_outer_test_patient_count is not None,
        )
        if supplied[0] != supplied[1]:
            raise ValueError("outer-test hash and count must be supplied together")
        if supplied[0]:
            digest = str(self.expected_outer_test_patient_id_hash)
            if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
                raise ValueError("outer-test patient hash must be lowercase SHA-256")
            if not isinstance(self.expected_outer_test_patient_count, int) or (
                self.expected_outer_test_patient_count <= 0
            ):
                raise ValueError("outer-test patient count must be positive")

    def expected_provenance(self, model_token: str) -> FoldRepresentationProvenance:
        return FoldRepresentationProvenance(
            schema_version="soft-patient-atlas-eval-fold-v1",
            fold_key=self.fold_key,
            model_token=str(model_token),
            outer_train_patient_hash=self.expected_outer_train_patient_hash,
            fit_patient_hash=_patient_hash(self.fit.patient_ids),
            validation_patient_hash=_patient_hash(self.validation.patient_ids),
            calibration_patient_hash=_patient_hash(self.calibration.patient_ids),
        )


@dataclass(frozen=True)
class MissingnessStratum:
    """Prespecified observation pattern evaluated with the frozen both-arm probe."""

    name: str
    evidence_class: str
    pattern: str
    blood_deletion_fraction: float | None = None
    retinal_image_count: int | None = None
    graceful_degradation_margin: float | None = None
    evaluation_available: bool = True
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name == BASE_STRATUM:
            raise ValueError("missingness stratum name must be nonempty and non-reserved")
        allowed = {
            "natural_partial",
            "controlled_deletion",
            "genuinely_incomplete_cohort",
        }
        if self.evidence_class not in allowed:
            raise ValueError("unknown missingness evidence class")
        if not self.pattern:
            raise ValueError("missingness pattern must not be empty")
        if self.blood_deletion_fraction is not None and not (
            0.0 <= self.blood_deletion_fraction <= 1.0
        ):
            raise ValueError("blood deletion fraction must lie in [0,1]")
        if self.retinal_image_count is not None and self.retinal_image_count < 0:
            raise ValueError("retinal image count must be nonnegative")
        if self.graceful_degradation_margin is not None and (
            not np.isfinite(self.graceful_degradation_margin)
            or self.graceful_degradation_margin < 0.0
        ):
            raise ValueError("graceful-degradation margin must be finite and nonnegative")
        if self.evaluation_available and self.unavailable_reason is not None:
            raise ValueError("available missingness strata cannot declare an unavailable reason")
        if not self.evaluation_available and not self.unavailable_reason:
            raise ValueError("unavailable missingness strata require a reason")


@dataclass(frozen=True)
class FoldCoordinates:
    """Coordinates emitted by exactly one fitted outer-fold representation."""

    patient_ids: tuple[str, ...]
    values: np.ndarray
    penalty_groups: tuple[str, ...]
    fold_key: str
    model_token: str
    basis_token: str
    arm: str
    stratum: str
    concat_eye_dimension: int | None = None


class FittedFoldRepresentation(Protocol):
    provenance: FoldRepresentationProvenance

    def transform(
        self,
        view: FeatureView,
        *,
        arm: str,
        stratum: str,
        missingness: MissingnessStratum | None = None,
        concat_eye_dimension: int | None = None,
    ) -> FoldCoordinates: ...


class FoldRepresentationFactory(Protocol):
    def fit(self, request: RepresentationFitRequest) -> FittedFoldRepresentation: ...


def _validate_provenance(
    provenance: FoldRepresentationProvenance,
    request: RepresentationFitRequest,
) -> None:
    if not isinstance(provenance, FoldRepresentationProvenance):
        raise TypeError("fitted representation provenance has the wrong contract type")
    expected = request.expected_provenance(provenance.model_token)
    if provenance != expected:
        raise ValueError(
            "representation provenance does not match the exact outer-training phases"
        )
    phase_ids = (
        set(request.fit.patient_ids),
        set(request.validation.patient_ids),
        set(request.calibration.patient_ids),
    )
    if any(left & right for i, left in enumerate(phase_ids) for right in phase_ids[i + 1 :]):
        raise ValueError("representation fit/validation/calibration identities overlap")
    phase_union = set().union(*phase_ids)
    if _patient_hash(tuple(phase_union)) != request.expected_outer_train_patient_hash:
        raise ValueError("representation phases do not cover exactly the outer-training set")


def _validate_coordinates(
    coordinates: FoldCoordinates,
    *,
    view: FeatureView,
    provenance: FoldRepresentationProvenance,
    arm: str,
    stratum: str,
    concat_eye_dimension: int | None,
) -> None:
    if not isinstance(coordinates, FoldCoordinates):
        raise TypeError("representation transform must return FoldCoordinates")
    if coordinates.patient_ids != view.patient_ids:
        raise ValueError("representation transform changed patient order or identity")
    if coordinates.fold_key != provenance.fold_key:
        raise ValueError("coordinates came from a different outer fold")
    if coordinates.model_token != provenance.model_token:
        raise ValueError("coordinates came from a different fitted model")
    if coordinates.arm != arm or coordinates.stratum != stratum:
        raise ValueError("coordinate arm/stratum provenance mismatch")
    if coordinates.concat_eye_dimension != concat_eye_dimension:
        raise ValueError("concat candidate provenance mismatch")
    values = np.asarray(coordinates.values)
    if values.ndim != 2 or values.shape[0] != view.size or values.shape[1] == 0:
        raise ValueError("coordinates must have finite shape [patients, positive dimension]")
    if not np.isfinite(values).all():
        raise ValueError("coordinates contain non-finite values")
    if len(coordinates.penalty_groups) != values.shape[1]:
        raise ValueError("one penalty-group label is required per coordinate")
    if any(not value for value in coordinates.penalty_groups):
        raise ValueError("coordinate penalty groups must not be empty")
    if not coordinates.basis_token:
        raise ValueError("coordinate basis token must not be empty")


def assert_same_fold_basis(
    train: FoldCoordinates,
    test: FoldCoordinates,
) -> None:
    """Reject cross-model, cross-fold, or train/test-specific coordinate bases."""

    identity = (
        "fold_key",
        "model_token",
        "basis_token",
        "arm",
        "stratum",
        "concat_eye_dimension",
        "penalty_groups",
    )
    mismatched = [name for name in identity if getattr(train, name) != getattr(test, name)]
    if mismatched:
        raise ValueError(
            "train/test coordinates do not share one fitted fold basis: "
            + ", ".join(mismatched)
        )
    if train.values.shape[1] != test.values.shape[1]:
        raise ValueError("train/test coordinate dimensions differ")


@dataclass(frozen=True)
class NestedEvaluationConfig:
    outer_folds: int = 5
    inner_folds: int = 5
    split_seeds: tuple[int, ...] = (1701, 2718, 3141)
    ridge_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    bootstrap_samples: int = 2_000
    confidence_level: float = 0.95
    bootstrap_seed: int = 8675309
    phase_fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
    max_tuning_combinations: int = 20_000
    noninferiority_margin: float | None = None
    noninferiority_margin_provenance: str | None = None
    missingness_strata: tuple[MissingnessStratum, ...] = ()
    outer_salt: str = "soft-patient-atlas-outer-v1"
    inner_salt: str = "soft-patient-atlas-inner-v1"
    phase_salt: str = "soft-patient-atlas-representation-phases-v1"
    enabled_arms: tuple[str, ...] = PRIMARY_ARMS

    def __post_init__(self) -> None:
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("outer_folds and inner_folds must be at least two")
        if not self.split_seeds or len(self.split_seeds) != len(set(self.split_seeds)):
            raise ValueError("split seeds must be nonempty and unique")
        if any(not isinstance(seed, int) for seed in self.split_seeds):
            raise TypeError("split seeds must be integers")
        if not self.ridge_grid or any(value <= 0 for value in self.ridge_grid):
            raise ValueError("ridge grid must contain positive values")
        if tuple(sorted(set(self.ridge_grid))) != self.ridge_grid:
            raise ValueError("ridge grid must be strictly increasing and unique")
        if self.bootstrap_samples < 100:
            raise ValueError("at least 100 patient-cluster bootstrap samples are required")
        if not math.isclose(self.confidence_level, 0.95, abs_tol=1e-12):
            raise ValueError("confirmatory confidence level is frozen at 0.95")
        if not math.isclose(sum(self.phase_fractions), 1.0, abs_tol=1e-12):
            raise ValueError("representation phase fractions must sum to one")
        if any(value <= 0 for value in self.phase_fractions):
            raise ValueError("every representation phase fraction must be positive")
        if self.max_tuning_combinations <= 0:
            raise ValueError("max_tuning_combinations must be positive")
        if self.noninferiority_margin is not None:
            if self.noninferiority_margin < 0:
                raise ValueError("noninferiority margin must be nonnegative")
            if not self.noninferiority_margin_provenance:
                raise ValueError(
                    "a noninferiority margin requires frozen blinded-simulation provenance"
                )
        elif self.noninferiority_margin_provenance is not None:
            raise ValueError("margin provenance is invalid while the margin remains null")
        names = [stratum.name for stratum in self.missingness_strata]
        if len(names) != len(set(names)):
            raise ValueError("missingness stratum names must be unique")
        if any(not value for value in (self.outer_salt, self.inner_salt, self.phase_salt)):
            raise ValueError("split salts must not be empty")
        if not self.enabled_arms or len(self.enabled_arms) != len(set(self.enabled_arms)):
            raise ValueError("enabled_arms must be nonempty and unique")
        unknown_arms = set(self.enabled_arms) - set(PRIMARY_ARMS)
        if unknown_arms:
            raise ValueError(f"enabled_arms contains unknown arms: {sorted(unknown_arms)}")
        required = {ARM_BOTH, ARM_EYE, ARM_BLOOD}
        if not required.issubset(set(self.enabled_arms)):
            raise ValueError("enabled_arms must include both Atlas and both single-modality arms")
        if self.noninferiority_margin is not None and ARM_CONCAT not in self.enabled_arms:
            raise ValueError("concat must be enabled when noninferiority is evaluated")


@dataclass(frozen=True)
class NestedEvaluationReport:
    """Aggregate-only nested evaluation result."""

    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        # JSON round-trip gives callers a detached object and also rejects
        # accidental NumPy arrays or custom row-level containers.
        return json.loads(
            json.dumps(dict(self.payload), sort_keys=True, allow_nan=False)
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            dict(self.payload), sort_keys=True, indent=indent, allow_nan=False
        )


def assert_aggregate_only_payload(
    payload: Mapping[str, Any],
    *,
    forbidden_patient_ids: Sequence[str] = (),
) -> None:
    """Fail closed if a proposed released report contains row-level material."""

    forbidden_keys = {
        "patient_id",
        "patient_ids",
        "row_id",
        "row_ids",
        "predictions",
        "coordinates",
        "outcomes",
        "target_values",
        "fold_manifest",
    }
    forbidden_values = set(str(value) for value in forbidden_patient_ids)

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).lower() in forbidden_keys:
                    raise ValueError(f"aggregate report contains forbidden key {key!r}")
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, np.ndarray):
            raise TypeError("aggregate report must not contain NumPy arrays")
        elif isinstance(value, str) and value in forbidden_values:
            raise ValueError("aggregate report exposes a patient identifier")
        elif not isinstance(value, (str, int, float, bool, type(None))):
            raise TypeError(f"aggregate report contains unsupported value type {type(value)!r}")

    visit(payload)


def _partition_outer_train(
    patient_ids: Sequence[str],
    site_ids: Sequence[str],
    *,
    fractions: tuple[float, float, float],
    salt: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically partition outer training into fit/validation/calibration."""

    if len(patient_ids) != len(site_ids) or not patient_ids:
        raise ValueError("outer-training IDs and sites must be nonempty and aligned")
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("outer-training phase partition requires unique patients")
    by_site: dict[str, list[int]] = {}
    for index, site in enumerate(site_ids):
        by_site.setdefault(str(site), []).append(index)
    phases: list[list[int]] = [[], [], []]
    for site in sorted(by_site):
        ordered = sorted(
            by_site[site],
            key=lambda index: (
                _stable_digest(salt, site, patient_ids[index]),
                patient_ids[index],
            ),
        )
        counts = _largest_remainder_counts(len(ordered), fractions)
        start = 0
        for destination, count in zip(phases, counts):
            destination.extend(ordered[start : start + count])
            start += count
    if any(not phase for phase in phases):
        raise ValueError(
            "outer-training fold is too small for nonempty fit/validation/calibration phases"
        )
    arrays = tuple(np.asarray(sorted(phase), dtype=np.int64) for phase in phases)
    union = np.concatenate(arrays)
    if len(np.unique(union)) != len(patient_ids) or set(union.tolist()) != set(
        range(len(patient_ids))
    ):
        raise RuntimeError("internal representation phase partition error")
    return arrays  # type: ignore[return-value]


@dataclass(frozen=True)
class _RobustDemographicTransform:
    center: np.ndarray
    scale: np.ndarray

    def apply(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if values.ndim != 2 or values.shape[1] != len(self.center):
            raise ValueError("demographic dimension differs from fitted transform")
        if mask.shape != values.shape or not bool(mask.all()):
            raise ValueError("required demographic context is missing")
        if not np.isfinite(values).all():
            raise ValueError("demographic context contains non-finite values")
        return (np.asarray(values, dtype=np.float64) - self.center) / self.scale


def _fit_demographic_transform(view: FeatureView) -> _RobustDemographicTransform:
    values = np.asarray(view.demographics, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not bool(view.demographic_mask.all()):
        raise ValueError("representation-fit demographics must be observed")
    if not np.isfinite(values).all():
        raise ValueError("representation-fit demographics must be finite")
    center = np.median(values, axis=0)
    q25 = np.quantile(values, 0.25, axis=0)
    q75 = np.quantile(values, 0.75, axis=0)
    scale = (q75 - q25) / 1.349
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return _RobustDemographicTransform(center=center, scale=scale)


def _with_identical_demographics(
    coordinates: FoldCoordinates,
    transformed_demographics: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if transformed_demographics.ndim != 2 or transformed_demographics.shape[0] != len(
        coordinates.patient_ids
    ):
        raise ValueError("demographics and coordinates are not row aligned")
    values = np.concatenate(
        [np.asarray(coordinates.values, dtype=np.float64), transformed_demographics],
        axis=1,
    )
    groups = coordinates.penalty_groups + (
        ("demographic",) * transformed_demographics.shape[1]
    )
    return values, groups


def _demographics_only(
    transformed_demographics: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    return (
        np.asarray(transformed_demographics, dtype=np.float64),
        ("demographic",) * transformed_demographics.shape[1],
    )


def _fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    groups: tuple[str, ...],
    penalties: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] < 2:
        raise ValueError("ridge fit requires aligned two-dimensional X and vector y")
    if len(groups) != x.shape[1] or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge inputs/groups are malformed or non-finite")
    if set(groups) != set(penalties):
        raise ValueError("ridge penalties do not cover exactly the feature groups")
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    standardized = (x - center) / scale
    y_center = float(y.mean())
    penalty_diagonal = np.asarray([penalties[group] for group in groups], dtype=np.float64)
    gram = standardized.T @ standardized
    rhs = standardized.T @ (y - y_center)
    system = gram + np.diag(penalty_diagonal)
    try:
        coefficient = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.lstsq(system, rhs, rcond=None)[0]
    if not np.isfinite(coefficient).all():
        raise ValueError("ridge solution is non-finite")
    return coefficient, center, scale, y_center


def _predict_ridge(
    model: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    x: np.ndarray,
) -> np.ndarray:
    coefficient, center, scale, y_center = model
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(coefficient):
        raise ValueError("ridge prediction dimension mismatch")
    prediction = y_center + ((values - center) / scale) @ coefficient
    if not np.isfinite(prediction).all():
        raise ValueError("ridge predictions are non-finite")
    return prediction


@dataclass(frozen=True)
class _ProbeChoice:
    concat_eye_dimension: int | None
    penalties: tuple[tuple[str, float], ...]
    inner_loss: float

    def penalty_mapping(self) -> Mapping[str, float]:
        return dict(self.penalties)


def _penalty_combinations(
    groups: tuple[str, ...],
    ridge_grid: tuple[float, ...],
) -> tuple[tuple[tuple[str, float], ...], ...]:
    unique_groups = tuple(sorted(set(groups)))
    return tuple(
        tuple(zip(unique_groups, values))
        for values in itertools.product(ridge_grid, repeat=len(unique_groups))
    )


def _select_probe(
    candidates: Mapping[int | None, tuple[np.ndarray, tuple[str, ...]]],
    y: np.ndarray,
    eligible: np.ndarray,
    inner_fold_ids: np.ndarray,
    *,
    ridge_grid: tuple[float, ...],
    max_tuning_combinations: int,
) -> _ProbeChoice:
    """Select representation candidate and block penalties using inner folds only."""

    if not candidates:
        raise ValueError("at least one probe candidate is required")
    if y.ndim != 1 or eligible.shape != y.shape or inner_fold_ids.shape != y.shape:
        raise ValueError("inner-probe labels, masks, and fold IDs must align")
    if eligible.dtype != np.bool_:
        raise TypeError("inner-probe eligibility must be boolean")
    expected_folds = tuple(sorted(set(int(value) for value in inner_fold_ids)))
    if len(expected_folds) < 2:
        raise ValueError("inner probe selection requires at least two populated folds")

    tuning: list[tuple[int | None, tuple[tuple[str, float], ...], np.ndarray, tuple[str, ...]]] = []
    for candidate, (x, groups) in candidates.items():
        if x.ndim != 2 or x.shape[0] != len(y) or len(groups) != x.shape[1]:
            raise ValueError("probe candidate coordinates are malformed")
        for penalties in _penalty_combinations(groups, ridge_grid):
            tuning.append((candidate, penalties, x, groups))
    if len(tuning) > max_tuning_combinations:
        raise ValueError("probe tuning grid exceeds the configured safety limit")

    best: _ProbeChoice | None = None
    best_order: tuple[float, int, tuple[float, ...]] | None = None
    candidate_rank = {
        candidate: index
        for index, candidate in enumerate(
            sorted(candidates, key=lambda value: (-1 if value is None else int(value)))
        )
    }
    for candidate, penalty_items, x, groups in tuning:
        penalties = dict(penalty_items)
        squared_errors: list[np.ndarray] = []
        for fold in expected_folds:
            validation = eligible & (inner_fold_ids == fold)
            training = eligible & (inner_fold_ids != fold)
            if int(training.sum()) < 2 or int(validation.sum()) < 1:
                raise ValueError("target has insufficient patients in an inner fold")
            model = _fit_ridge(x[training], y[training], groups, penalties)
            prediction = _predict_ridge(model, x[validation])
            squared_errors.append((prediction - y[validation]) ** 2)
        inner_loss = float(np.concatenate(squared_errors).mean())
        tie_order = (
            inner_loss,
            candidate_rank[candidate],
            tuple(value for _, value in penalty_items),
        )
        if best_order is None or tie_order < best_order:
            best_order = tie_order
            best = _ProbeChoice(candidate, penalty_items, inner_loss)
    if best is None:
        raise RuntimeError("internal probe-selection failure")
    return best


def _target_family_score(losses: np.ndarray, manifest: TargetManifest) -> float:
    """Family-balanced mean loss with target-specific eligible-patient means."""

    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(manifest.targets):
        raise ValueError("loss matrix must have shape [patients, manifest targets]")
    target_means: list[float] = []
    for column in range(values.shape[1]):
        observed = np.isfinite(values[:, column])
        if not bool(observed.any()):
            raise ValueError("a primary target has no evaluable loss")
        target_means.append(float(values[observed, column].mean()))
    family_scores: dict[str, float] = {}
    family_weights: dict[str, float] = {}
    for target, target_mean in zip(manifest.targets, target_means):
        family_scores[target.family] = family_scores.get(target.family, 0.0) + (
            target.within_family_weight * target_mean
        )
        family_weights[target.family] = target.family_weight
    return float(
        sum(family_weights[family] * score for family, score in family_scores.items())
    )


def _target_aggregate(losses: np.ndarray, manifest: TargetManifest) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, target in enumerate(manifest.targets):
        observed = np.isfinite(losses[:, index])
        if not bool(observed.any()):
            raise ValueError("a primary target has no evaluable loss")
        result[target.id] = float(losses[observed, index].mean())
    return result


def _percentile_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def _basic_bootstrap_interval(
    point: float,
    samples: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    """Centered/basic interval for a patient-level bootstrap estimator."""

    alpha = 1.0 - confidence_level
    centered_error = np.asarray(samples, dtype=np.float64) - float(point)
    lower = point - float(np.quantile(centered_error, 1.0 - alpha / 2.0))
    upper = point - float(np.quantile(centered_error, alpha / 2.0))
    return lower, upper


def _basic_one_sided_lower(
    point: float,
    samples: np.ndarray,
    *,
    alpha: float,
) -> float:
    centered_error = np.asarray(samples, dtype=np.float64) - float(point)
    return float(point - np.quantile(centered_error, 1.0 - alpha))


def _centered_bootstrap_null_tail(
    point: float,
    samples: np.ndarray,
    *,
    null_value: float = 0.0,
) -> float:
    """Approximate one-sided null tail from the *centered* cluster bootstrap.

    Unlike the fraction of ordinary bootstrap estimates crossing zero, this
    shifts the resampling distribution to the null before evaluating the
    observed effect.  It is labelled as a bootstrap approximation, not an
    exact randomization p-value.
    """

    centered_null = np.asarray(samples, dtype=np.float64) - float(point) + null_value
    return float(
        (1 + np.count_nonzero(centered_null >= point)) / (len(centered_null) + 1)
    )


def _holm_bootstrap_lower_bounds(
    points: Mapping[str, float],
    samples: Mapping[str, np.ndarray],
    *,
    alpha: float,
) -> dict[str, dict[str, float | bool | int]]:
    """Holm step-down inference using centered patient-cluster bootstraps."""

    if set(points) != set(samples) or not points:
        raise ValueError("Holm point estimates and bootstrap samples must align")
    tail = {
        name: _centered_bootstrap_null_tail(points[name], samples[name])
        for name in points
    }
    ordered = sorted(points, key=lambda name: (tail[name], name))
    result: dict[str, dict[str, float | bool | int]] = {}
    running_adjusted = 0.0
    continue_rejection = True
    family_size = len(ordered)
    for rank, name in enumerate(ordered):
        remaining = family_size - rank
        local_alpha = alpha / remaining
        lower = _basic_one_sided_lower(
            points[name], samples[name], alpha=local_alpha
        )
        running_adjusted = max(running_adjusted, remaining * tail[name])
        rejected = bool(continue_rejection and lower > 0.0)
        if not rejected:
            continue_rejection = False
        result[name] = {
            "holm_rank": rank + 1,
            "holm_local_alpha": float(local_alpha),
            "centered_cluster_bootstrap_null_tail_probability": tail[name],
            "holm_adjusted_bootstrap_tail_probability": min(1.0, running_adjusted),
            "holm_stepdown_lower_confidence_bound": lower,
            "holm_superiority_rejected_null": rejected,
        }
    return result


def _simultaneous_max_stat_lower_bounds(
    points: Mapping[str, float],
    samples: Mapping[str, np.ndarray],
    *,
    alpha: float,
) -> dict[str, float]:
    """One-sided family-wise lower bounds from centered bootstrap max errors."""

    if set(points) != set(samples) or not points:
        return {}
    lengths = {len(value) for value in samples.values()}
    if len(lengths) != 1:
        raise ValueError("max-stat bootstrap samples must have identical lengths")
    errors = np.column_stack(
        [np.asarray(samples[name], dtype=np.float64) - points[name] for name in points]
    )
    critical_value = float(np.quantile(errors.max(axis=1), 1.0 - alpha))
    return {name: float(points[name] - critical_value) for name in points}


def _bootstrap_aggregates(
    arm_losses: Mapping[str, np.ndarray],
    missingness_atlas_losses: Mapping[str, np.ndarray],
    missingness_oracle_losses: Mapping[str, np.ndarray],
    manifest: TargetManifest,
    config: NestedEvaluationConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Patient-cluster bootstrap after repeat-level losses are averaged per patient."""

    enabled_arms = tuple(config.enabled_arms)
    if set(enabled_arms) != set(arm_losses):
        raise ValueError("configured enabled arms differ from supplied arm losses")
    size = next(iter(arm_losses.values())).shape[0]
    if any(values.shape != (size, len(manifest.targets)) for values in arm_losses.values()):
        raise ValueError("arm loss arrays do not share patient/target shape")
    if set(missingness_atlas_losses) != set(missingness_oracle_losses):
        raise ValueError("missingness atlas/oracle pattern sets differ")
    for collection in (missingness_atlas_losses, missingness_oracle_losses):
        if any(
            values.shape != (size, len(manifest.targets))
            for values in collection.values()
        ):
            raise ValueError("missingness loss arrays do not share patient/target shape")

    rng = np.random.default_rng(config.bootstrap_seed)
    bootstrap_indices = rng.integers(0, size, size=(config.bootstrap_samples, size))
    arm_bootstrap: dict[str, np.ndarray] = {
        arm: np.empty(config.bootstrap_samples, dtype=np.float64) for arm in arm_losses
    }
    missing_atlas_bootstrap: dict[str, np.ndarray] = {
        name: np.empty(config.bootstrap_samples, dtype=np.float64)
        for name in missingness_atlas_losses
    }
    missing_oracle_bootstrap: dict[str, np.ndarray] = {
        name: np.empty(config.bootstrap_samples, dtype=np.float64)
        for name in missingness_oracle_losses
    }
    for iteration, indices in enumerate(bootstrap_indices):
        for arm, values in arm_losses.items():
            arm_bootstrap[arm][iteration] = _target_family_score(values[indices], manifest)
        for name, values in missingness_atlas_losses.items():
            missing_atlas_bootstrap[name][iteration] = _target_family_score(
                values[indices], manifest
            )
        for name, values in missingness_oracle_losses.items():
            missing_oracle_bootstrap[name][iteration] = _target_family_score(
                values[indices], manifest
            )

    arm_report: dict[str, Any] = {}
    for arm, values in arm_losses.items():
        score = _target_family_score(values, manifest)
        lower, upper = _percentile_interval(arm_bootstrap[arm], config.confidence_level)
        arm_report[arm] = {
            "primary_family_balanced_normalized_loss": score,
            "confidence_interval": [lower, upper],
            "per_target_normalized_loss": _target_aggregate(values, manifest),
        }

    all_contrast_specs = {
        "eye_minus_both": ARM_EYE,
        "blood_minus_both": ARM_BLOOD,
        "concat_minus_both": ARM_CONCAT,
        "eye_tower_minus_both": ARM_EYE_TOWER,
        "raw_blood_minus_both": ARM_RAW_BLOOD,
        "frozen_blood_tower_minus_both": ARM_FROZEN_BLOOD_TOWER,
    }
    contrast_specs = {
        name: comparator
        for name, comparator in all_contrast_specs.items()
        if comparator in arm_losses
    }
    contrast_report: dict[str, Any] = {}
    bootstrap_contrasts: dict[str, np.ndarray] = {}
    for name, comparator in contrast_specs.items():
        point = arm_report[comparator]["primary_family_balanced_normalized_loss"] - arm_report[
            ARM_BOTH
        ]["primary_family_balanced_normalized_loss"]
        samples = arm_bootstrap[comparator] - arm_bootstrap[ARM_BOTH]
        bootstrap_contrasts[name] = samples
        lower, upper = _basic_bootstrap_interval(
            point, samples, config.confidence_level
        )
        contrast_report[name] = {
            "loss_difference": float(point),
            "confidence_interval": [lower, upper],
            "one_sided_lower_confidence_bound": _basic_one_sided_lower(
                point,
                samples,
                alpha=1.0 - config.confidence_level,
            ),
            "positive_favors_both_atlas": True,
            "paired_inference": "centered patient-cluster basic bootstrap",
        }

    primary_names = ("eye_minus_both", "blood_minus_both")
    holm = _holm_bootstrap_lower_bounds(
        {name: contrast_report[name]["loss_difference"] for name in primary_names},
        {name: bootstrap_contrasts[name] for name in primary_names},
        alpha=1.0 - config.confidence_level,
    )
    for name, inference in holm.items():
        contrast_report[name].update(inference)

    standalone_names = tuple(
        name
        for name in (
            "eye_tower_minus_both",
            "raw_blood_minus_both",
            "frozen_blood_tower_minus_both",
        )
        if name in contrast_report
    )
    standalone_bounds = (
        _simultaneous_max_stat_lower_bounds(
            {
                name: contrast_report[name]["loss_difference"]
                for name in standalone_names
            },
            {name: bootstrap_contrasts[name] for name in standalone_names},
            alpha=1.0 - config.confidence_level,
        )
        if standalone_names
        else {}
    )
    for name, lower_bound in standalone_bounds.items():
        contrast_report[name]["simultaneous_max_stat_lower_confidence_bound"] = lower_bound
        contrast_report[name]["simultaneous_superiority_passed"] = bool(lower_bound > 0.0)

    stratum_by_name = {stratum.name: stratum for stratum in config.missingness_strata}
    missing_report: dict[str, Any] = {}
    missing_effect_points: dict[str, float] = {}
    missing_effect_samples: dict[str, np.ndarray] = {}
    for name, values in missingness_atlas_losses.items():
        oracle_values = missingness_oracle_losses[name]
        atlas_score = _target_family_score(values, manifest)
        oracle_score = _target_family_score(oracle_values, manifest)
        atlas_lower, atlas_upper = _percentile_interval(
            missing_atlas_bootstrap[name], config.confidence_level
        )
        oracle_lower, oracle_upper = _percentile_interval(
            missing_oracle_bootstrap[name], config.confidence_level
        )
        degradation_samples = missing_atlas_bootstrap[name] - arm_bootstrap[ARM_BOTH]
        degradation = atlas_score - arm_report[ARM_BOTH][
            "primary_family_balanced_normalized_loss"
        ]
        deg_lower, deg_upper = _basic_bootstrap_interval(
            degradation, degradation_samples, config.confidence_level
        )
        oracle_minus_atlas = oracle_score - atlas_score
        oracle_minus_atlas_samples = (
            missing_oracle_bootstrap[name] - missing_atlas_bootstrap[name]
        )
        local_lower = _basic_one_sided_lower(
            oracle_minus_atlas,
            oracle_minus_atlas_samples,
            alpha=1.0 - config.confidence_level,
        )
        pair_lower, pair_upper = _basic_bootstrap_interval(
            oracle_minus_atlas,
            oracle_minus_atlas_samples,
            config.confidence_level,
        )
        missing_effect_points[name] = oracle_minus_atlas
        missing_effect_samples[name] = oracle_minus_atlas_samples
        missing_report[name] = {
            "atlas_with_frozen_both_present_probe": {
                "primary_family_balanced_normalized_loss": atlas_score,
                "confidence_interval": [atlas_lower, atlas_upper],
                "per_target_normalized_loss": _target_aggregate(values, manifest),
            },
            "separately_inner_tuned_available_input_oracle": {
                "primary_family_balanced_normalized_loss": oracle_score,
                "confidence_interval": [oracle_lower, oracle_upper],
                "per_target_normalized_loss": _target_aggregate(
                    oracle_values, manifest
                ),
            },
            "degradation_from_both_present": float(degradation),
            "degradation_confidence_interval": [deg_lower, deg_upper],
            "oracle_minus_atlas_loss_difference": float(oracle_minus_atlas),
            "oracle_minus_atlas_confidence_interval": [pair_lower, pair_upper],
            "local_one_sided_lower_confidence_bound": local_lower,
            "positive_favors_atlas": True,
            "oracle_fit_scope": "identically corrupted outer-training only",
        }

    max_stat_bounds = _simultaneous_max_stat_lower_bounds(
        missing_effect_points,
        missing_effect_samples,
        alpha=1.0 - config.confidence_level,
    )
    for name, bound in max_stat_bounds.items():
        missing_report[name]["simultaneous_max_stat_lower_confidence_bound"] = bound

    for name, stratum in stratum_by_name.items():
        if name not in missing_report:
            missing_report[name] = {
                "status": "unsupported",
                "supported": False,
                "unsupported_reason": stratum.unavailable_reason
                or "evaluation unavailable",
            }
            continue
        margin = stratum.graceful_degradation_margin
        bound = missing_report[name]["simultaneous_max_stat_lower_confidence_bound"]
        if margin is None:
            supported = False
            reason = "graceful-degradation margin unresolved"
        elif bound > -margin:
            supported = True
            reason = None
        else:
            supported = False
            reason = "simultaneous lower bound crossed the prespecified margin"
        missing_report[name].update(
            {
                "graceful_degradation_margin": margin,
                "support_rule": "oracle_minus_atlas simultaneous lower bound > -margin",
                "status": "supported" if supported else "unsupported",
                "supported": supported,
                "unsupported_reason": reason,
            }
        )

    evidence_classes: dict[str, list[str]] = {}
    for stratum in config.missingness_strata:
        evidence_classes.setdefault(stratum.evidence_class, []).append(stratum.name)
    hierarchy = {
        evidence_class: {
            "pattern_count": len(names),
            "all_patterns_supported": bool(
                all(missing_report[name]["supported"] for name in names)
            ),
        }
        for evidence_class, names in evidence_classes.items()
    }
    missing_family_passed = bool(
        config.missingness_strata
        and all(value["all_patterns_supported"] for value in hierarchy.values())
    )
    inference_summary = {
        "standalone_information_source_gate_passed": bool(
            standalone_names
            and all(
                contrast_report[name]["simultaneous_superiority_passed"]
                for name in standalone_names
            )
        ),
        "standalone_information_source_contrasts_evaluated": list(
            standalone_names
        ),
        "missingness_family_gate_passed": missing_family_passed,
        "missingness_hierarchy": hierarchy,
        "missingness_family_control": (
            "one-sided centered patient-cluster bootstrap max-statistic; "
            "patterns nested within evidence class"
        ),
    }
    return arm_report, contrast_report, missing_report, inference_summary


def _average_repeats(losses: np.ndarray) -> np.ndarray:
    if losses.ndim != 3:
        raise ValueError("repeat losses must have shape [repeats, patients, targets]")
    observed = np.isfinite(losses)
    count = observed.sum(axis=0)
    total = np.where(observed, losses, 0.0).sum(axis=0)
    averaged = np.full(count.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=averaged, where=count > 0)
    return averaged


def _transform_pair(
    fitted: FittedFoldRepresentation,
    train_view: FeatureView,
    test_view: FeatureView,
    *,
    arm: str,
    stratum: str,
    missingness: MissingnessStratum | None = None,
    concat_eye_dimension: int | None = None,
) -> tuple[FoldCoordinates, FoldCoordinates]:
    provenance = fitted.provenance
    train = fitted.transform(
        train_view,
        arm=arm,
        stratum=stratum,
        missingness=missingness,
        concat_eye_dimension=concat_eye_dimension,
    )
    test = fitted.transform(
        test_view,
        arm=arm,
        stratum=stratum,
        missingness=missingness,
        concat_eye_dimension=concat_eye_dimension,
    )
    for coordinates, view in ((train, train_view), (test, test_view)):
        _validate_coordinates(
            coordinates,
            view=view,
            provenance=provenance,
            arm=arm,
            stratum=stratum,
            concat_eye_dimension=concat_eye_dimension,
        )
    assert_same_fold_basis(train, test)
    return train, test


def evaluate_nested_patient_atlas(
    cohort: EvaluationCohort,
    manifest: TargetManifest,
    representation_factory: FoldRepresentationFactory,
    config: NestedEvaluationConfig = NestedEvaluationConfig(),
) -> NestedEvaluationReport:
    """Run strict repeated nested evaluation and return aggregate results only.

    The representation factory is invoked once per outer fold and receives only
    its disjoint fit, validation, and calibration feature views.  It never
    receives target values, target masks, outer-test features, or outer-test
    identities.  The returned fold object must transform both outer train and
    outer test under one provenance-tagged model/basis.
    """

    if cohort.targets.shape[1] != len(manifest.targets):
        raise ValueError("cohort target columns do not match the target manifest")
    if cohort.target_ids != manifest.target_ids:
        raise ValueError("cohort target order does not exactly match the frozen manifest")
    if config.noninferiority_margin != manifest.noninferiority_margin:
        raise ValueError(
            "evaluation noninferiority margin must exactly match the frozen target manifest"
        )
    observed_targets = cohort.target_mask & cohort.primary_patient_mask[:, None]
    if not np.isfinite(cohort.targets[cohort.target_mask]).all():
        raise ValueError("observed target values must be finite")
    target_counts = observed_targets.sum(axis=0)
    for target, count in zip(manifest.targets, target_counts):
        if int(count) < manifest.minimum_complete_target_patients:
            raise ValueError(
                f"target {target.id!r} has {int(count)} eligible patients, below the frozen minimum"
            )

    repeats = len(config.split_seeds)
    target_count = len(manifest.targets)
    enabled_arms = tuple(config.enabled_arms)
    arm_losses = {
        arm: np.full((repeats, cohort.size, target_count), np.nan, dtype=np.float64)
        for arm in enabled_arms
    }
    evaluated_strata = tuple(
        stratum for stratum in config.missingness_strata if stratum.evaluation_available
    )
    missing_atlas_losses = {
        stratum.name: np.full(
            (repeats, cohort.size, target_count), np.nan, dtype=np.float64
        )
        for stratum in evaluated_strata
    }
    missing_oracle_losses = {
        stratum.name: np.full(
            (repeats, cohort.size, target_count), np.nan, dtype=np.float64
        )
        for stratum in evaluated_strata
    }
    selection_counts: dict[str, dict[str, dict[str, int]]] = {
        arm: {} for arm in enabled_arms
    }
    selection_totals = {arm: 0 for arm in enabled_arms}
    endpoint_selections = {arm: 0 for arm in enabled_arms}
    oracle_selection_counts: dict[str, dict[str, dict[str, int]]] = {
        stratum.name: {} for stratum in evaluated_strata
    }
    oracle_selection_totals = {stratum.name: 0 for stratum in evaluated_strata}
    oracle_endpoint_selections = {stratum.name: 0 for stratum in evaluated_strata}
    concat_dimension_counts = {str(value): 0 for value in CONCAT_EYE_DIMENSIONS}
    outer_fold_test_counts: list[int] = []
    seed_score_comparators = {
        "eye_minus_both": ARM_EYE,
        "blood_minus_both": ARM_BLOOD,
        "concat_minus_both": ARM_CONCAT,
    }
    seed_scores: dict[str, list[float]] = {
        name: []
        for name, comparator in seed_score_comparators.items()
        if comparator in enabled_arms
    }

    for repeat_index, split_seed in enumerate(config.split_seeds):
        fold_map = make_canonical_fold_map(
            cohort.patient_ids,
            cohort.site_ids,
            n_folds=config.outer_folds,
            seed=split_seed,
            salt=config.outer_salt,
        )
        outer_assignment = fold_map.assignments_for(cohort.patient_ids)

        for outer_fold in range(config.outer_folds):
            test_indices = np.flatnonzero(outer_assignment == outer_fold)
            train_indices = np.flatnonzero(outer_assignment != outer_fold)
            if not len(test_indices) or not len(train_indices):
                raise ValueError("canonical outer fold has an empty train or test partition")
            outer_fold_test_counts.append(int(len(test_indices)))
            train_view = cohort.feature_view(train_indices)
            test_view = cohort.feature_view(test_indices)
            fold_key = f"seed-{split_seed}-outer-{outer_fold}"

            fit_rel, validation_rel, calibration_rel = _partition_outer_train(
                train_view.patient_ids,
                train_view.site_ids,
                fractions=config.phase_fractions,
                salt=f"{config.phase_salt}:{fold_key}",
            )
            request = RepresentationFitRequest(
                fold_key=fold_key,
                fit=cohort.feature_view(train_indices[fit_rel]),
                validation=cohort.feature_view(train_indices[validation_rel]),
                calibration=cohort.feature_view(train_indices[calibration_rel]),
                expected_outer_train_patient_hash=_patient_hash(train_view.patient_ids),
                expected_outer_test_patient_id_hash=_patient_json_hash(
                    test_view.patient_ids
                ),
                expected_outer_test_patient_count=test_view.size,
            )
            fitted = representation_factory.fit(request)
            if not hasattr(fitted, "provenance"):
                raise TypeError("fitted representation lacks required provenance")
            _validate_provenance(fitted.provenance, request)

            demographic_transform = _fit_demographic_transform(request.fit)
            demographic_train = demographic_transform.apply(
                train_view.demographics, train_view.demographic_mask
            )
            demographic_test = demographic_transform.apply(
                test_view.demographics, test_view.demographic_mask
            )

            design_train: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {}
            design_test: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {}
            atlas_coordinates: dict[str, tuple[FoldCoordinates, FoldCoordinates]] = {}
            for arm in (ARM_BOTH, ARM_EYE, ARM_BLOOD):
                train_coordinates, test_coordinates = _transform_pair(
                    fitted,
                    train_view,
                    test_view,
                    arm=arm,
                    stratum=BASE_STRATUM,
                )
                atlas_coordinates[arm] = (train_coordinates, test_coordinates)
                design_train[arm] = _with_identical_demographics(
                    train_coordinates, demographic_train
                )
                design_test[arm] = _with_identical_demographics(
                    test_coordinates, demographic_test
                )

            both_train_coordinates, both_test_coordinates = atlas_coordinates[ARM_BOTH]
            for arm in (ARM_EYE, ARM_BLOOD):
                arm_train, arm_test = atlas_coordinates[arm]
                if (
                    arm_train.basis_token != both_train_coordinates.basis_token
                    or arm_test.basis_token != both_test_coordinates.basis_token
                    or arm_train.values.shape[1] != both_train_coordinates.values.shape[1]
                    or arm_train.penalty_groups != both_train_coordinates.penalty_groups
                ):
                    raise ValueError(
                        "both/eye/blood atlas states do not inhabit one fold coordinate system"
                    )

            for arm in (
                ARM_EYE_TOWER,
                ARM_RAW_BLOOD,
                ARM_FROZEN_BLOOD_TOWER,
            ):
                if arm not in enabled_arms:
                    continue
                train_coordinates, test_coordinates = _transform_pair(
                    fitted,
                    train_view,
                    test_view,
                    arm=arm,
                    stratum=BASE_STRATUM,
                )
                design_train[arm] = _with_identical_demographics(
                    train_coordinates, demographic_train
                )
                design_test[arm] = _with_identical_demographics(
                    test_coordinates, demographic_test
                )

            concat_train: dict[int, tuple[np.ndarray, tuple[str, ...]]] = {}
            concat_test: dict[int, tuple[np.ndarray, tuple[str, ...]]] = {}
            if ARM_CONCAT in enabled_arms:
                for eye_dimension in CONCAT_EYE_DIMENSIONS:
                    train_coordinates, test_coordinates = _transform_pair(
                        fitted,
                        train_view,
                        test_view,
                        arm=ARM_CONCAT,
                        stratum=BASE_STRATUM,
                        concat_eye_dimension=eye_dimension,
                    )
                    if not {"eye", "blood"}.issubset(
                        set(train_coordinates.penalty_groups)
                    ):
                        raise ValueError(
                            "concat representation must identify separate eye and blood penalty blocks"
                        )
                    concat_train[eye_dimension] = _with_identical_demographics(
                        train_coordinates, demographic_train
                    )
                    concat_test[eye_dimension] = _with_identical_demographics(
                        test_coordinates, demographic_test
                    )
            if ARM_DEMOGRAPHICS in enabled_arms:
                design_train[ARM_DEMOGRAPHICS] = _demographics_only(
                    demographic_train
                )
                design_test[ARM_DEMOGRAPHICS] = _demographics_only(
                    demographic_test
                )

            missing_design_train: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {}
            missing_design_test: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {}
            for stratum in evaluated_strata:
                missing_train_coordinates, missing_test_coordinates = _transform_pair(
                    fitted,
                    train_view,
                    test_view,
                    arm=ARM_BOTH,
                    stratum=stratum.name,
                    missingness=stratum,
                )
                if (
                    missing_train_coordinates.basis_token
                    != both_train_coordinates.basis_token
                    or missing_train_coordinates.penalty_groups
                    != both_train_coordinates.penalty_groups
                    or missing_train_coordinates.values.shape[1]
                    != both_train_coordinates.values.shape[1]
                ):
                    raise ValueError(
                        "missingness transform left the both-atlas coordinate system"
                    )
                missing_design_train[stratum.name] = _with_identical_demographics(
                    missing_train_coordinates, demographic_train
                )
                missing_design_test[stratum.name] = _with_identical_demographics(
                    missing_test_coordinates, demographic_test
                )

            inner_seed = int.from_bytes(
                _stable_digest(config.inner_salt, split_seed, outer_fold)[:4], "big"
            )
            inner_map = make_canonical_fold_map(
                train_view.patient_ids,
                train_view.site_ids,
                n_folds=config.inner_folds,
                seed=inner_seed,
                salt=config.inner_salt,
            )
            inner_fold_ids = inner_map.assignments_for(train_view.patient_ids)

            for target_index, target in enumerate(manifest.targets):
                y_train = np.asarray(cohort.targets[train_indices, target_index], dtype=np.float64)
                y_test = np.asarray(cohort.targets[test_indices, target_index], dtype=np.float64)
                train_eligible = (
                    cohort.primary_patient_mask[train_indices]
                    & cohort.target_mask[train_indices, target_index]
                )
                test_eligible = (
                    cohort.primary_patient_mask[test_indices]
                    & cohort.target_mask[test_indices, target_index]
                )
                if int(train_eligible.sum()) < config.inner_folds * 2:
                    raise ValueError(
                        f"target {target.id!r} has insufficient outer-training observations"
                    )
                if int(test_eligible.sum()) < 1:
                    raise ValueError(
                        f"target {target.id!r} has no eligible patient in one outer test fold"
                    )
                target_center = float(y_train[train_eligible].mean())
                intercept_loss = float(
                    np.mean((y_train[train_eligible] - target_center) ** 2)
                )
                if not np.isfinite(intercept_loss) or intercept_loss <= 1e-12:
                    raise ValueError(
                        f"target {target.id!r} has zero/non-finite outer-training intercept loss"
                    )

                fitted_both_probe: tuple[
                    tuple[np.ndarray, np.ndarray, np.ndarray, float],
                    tuple[str, ...],
                ] | None = None
                for arm in enabled_arms:
                    if arm == ARM_CONCAT:
                        candidates: Mapping[
                            int | None, tuple[np.ndarray, tuple[str, ...]]
                        ] = concat_train
                    else:
                        candidates = {None: design_train[arm]}
                    choice = _select_probe(
                        candidates,
                        y_train,
                        train_eligible,
                        inner_fold_ids,
                        ridge_grid=config.ridge_grid,
                        max_tuning_combinations=config.max_tuning_combinations,
                    )
                    chosen_train, chosen_groups = candidates[choice.concat_eye_dimension]
                    if arm == ARM_CONCAT:
                        if choice.concat_eye_dimension is None:
                            raise RuntimeError("concat tuning failed to choose an eye dimension")
                        chosen_test, test_groups = concat_test[choice.concat_eye_dimension]
                        concat_dimension_counts[str(choice.concat_eye_dimension)] += 1
                    else:
                        chosen_test, test_groups = design_test[arm]
                    if test_groups != chosen_groups:
                        raise ValueError("outer train/test probe penalty groups differ")
                    ridge_model = _fit_ridge(
                        chosen_train[train_eligible],
                        y_train[train_eligible],
                        chosen_groups,
                        choice.penalty_mapping(),
                    )
                    prediction = _predict_ridge(ridge_model, chosen_test)
                    global_test = test_indices[test_eligible]
                    arm_losses[arm][repeat_index, global_test, target_index] = (
                        (prediction[test_eligible] - y_test[test_eligible]) ** 2
                    ) / intercept_loss
                    if arm == ARM_BOTH:
                        fitted_both_probe = (ridge_model, chosen_groups)

                    selection_totals[arm] += 1
                    selected_endpoint = False
                    for group, penalty in choice.penalties:
                        text_penalty = format(penalty, ".12g")
                        group_counts = selection_counts[arm].setdefault(group, {})
                        group_counts[text_penalty] = group_counts.get(text_penalty, 0) + 1
                        if penalty in (config.ridge_grid[0], config.ridge_grid[-1]):
                            selected_endpoint = True
                    if selected_endpoint:
                        endpoint_selections[arm] += 1

                if fitted_both_probe is None:
                    raise RuntimeError("both-atlas probe was not fitted")
                both_ridge_model, both_groups = fitted_both_probe
                for stratum in evaluated_strata:
                    missing_train_x, missing_train_groups = missing_design_train[
                        stratum.name
                    ]
                    missing_x, missing_groups = missing_design_test[stratum.name]
                    if missing_groups != both_groups:
                        raise ValueError(
                            "missingness design cannot be scored by the frozen both-atlas probe"
                        )
                    frozen_prediction = _predict_ridge(both_ridge_model, missing_x)
                    global_test = test_indices[test_eligible]
                    missing_atlas_losses[stratum.name][
                        repeat_index, global_test, target_index
                    ] = (
                        (frozen_prediction[test_eligible] - y_test[test_eligible]) ** 2
                    ) / intercept_loss

                    oracle_choice = _select_probe(
                        {None: (missing_train_x, missing_train_groups)},
                        y_train,
                        train_eligible,
                        inner_fold_ids,
                        ridge_grid=config.ridge_grid,
                        max_tuning_combinations=config.max_tuning_combinations,
                    )
                    oracle_model = _fit_ridge(
                        missing_train_x[train_eligible],
                        y_train[train_eligible],
                        missing_train_groups,
                        oracle_choice.penalty_mapping(),
                    )
                    oracle_prediction = _predict_ridge(oracle_model, missing_x)
                    missing_oracle_losses[stratum.name][
                        repeat_index, global_test, target_index
                    ] = (
                        (oracle_prediction[test_eligible] - y_test[test_eligible]) ** 2
                    ) / intercept_loss

                    oracle_selection_totals[stratum.name] += 1
                    oracle_selected_endpoint = False
                    for group, penalty in oracle_choice.penalties:
                        text_penalty = format(penalty, ".12g")
                        group_counts = oracle_selection_counts[stratum.name].setdefault(
                            group, {}
                        )
                        group_counts[text_penalty] = (
                            group_counts.get(text_penalty, 0) + 1
                        )
                        if penalty in (config.ridge_grid[0], config.ridge_grid[-1]):
                            oracle_selected_endpoint = True
                    if oracle_selected_endpoint:
                        oracle_endpoint_selections[stratum.name] += 1

        expected = observed_targets
        for arm, values in arm_losses.items():
            if not np.array_equal(np.isfinite(values[repeat_index]), expected):
                raise RuntimeError(
                    f"arm {arm!r} did not produce exactly one eligible OOF loss per patient"
                )
        for collection_name, collection in (
            ("frozen-probe", missing_atlas_losses),
            ("available-input oracle", missing_oracle_losses),
        ):
            for name, values in collection.items():
                if not np.array_equal(np.isfinite(values[repeat_index]), expected):
                    raise RuntimeError(
                        f"missingness {collection_name} stratum {name!r} has "
                        "incomplete/misaligned OOF losses"
                    )
        both_seed_score = _target_family_score(arm_losses[ARM_BOTH][repeat_index], manifest)
        for contrast, comparator in seed_score_comparators.items():
            if comparator not in enabled_arms:
                continue
            seed_scores[contrast].append(
                _target_family_score(arm_losses[comparator][repeat_index], manifest)
                - both_seed_score
            )

    averaged_arm_losses = {arm: _average_repeats(values) for arm, values in arm_losses.items()}
    averaged_missing_atlas_losses = {
        name: _average_repeats(values) for name, values in missing_atlas_losses.items()
    }
    averaged_missing_oracle_losses = {
        name: _average_repeats(values) for name, values in missing_oracle_losses.items()
    }
    arm_report, contrast_report, missing_report, inference_summary = _bootstrap_aggregates(
        averaged_arm_losses,
        averaged_missing_atlas_losses,
        averaged_missing_oracle_losses,
        manifest,
        config,
    )
    for name, values in seed_scores.items():
        contrast_report[name]["split_seed_loss_differences"] = [float(value) for value in values]
        contrast_report[name]["same_favorable_sign_every_split_seed"] = bool(
            all(value > 0.0 for value in values)
        )

    tuning_report: dict[str, Any] = {}
    for arm in enabled_arms:
        total = selection_totals[arm]
        endpoint_fraction = endpoint_selections[arm] / total if total else 1.0
        tuning_report[arm] = {
            "selection_count": total,
            "penalty_selection_counts": selection_counts[arm],
            "grid_endpoint_selection_fraction": float(endpoint_fraction),
            "requires_grid_expansion": bool(endpoint_fraction > 0.10),
        }
    if ARM_CONCAT in tuning_report:
        tuning_report[ARM_CONCAT][
            "eye_dimension_selection_counts"
        ] = concat_dimension_counts
    arm_tuning_grid_passed = not any(
        value["requires_grid_expansion"] for value in tuning_report.values()
    )
    oracle_tuning_report: dict[str, Any] = {}
    for stratum in evaluated_strata:
        total = oracle_selection_totals[stratum.name]
        endpoint_fraction = (
            oracle_endpoint_selections[stratum.name] / total if total else 1.0
        )
        oracle_tuning_report[stratum.name] = {
            "selection_count": total,
            "penalty_selection_counts": oracle_selection_counts[stratum.name],
            "grid_endpoint_selection_fraction": float(endpoint_fraction),
            "requires_grid_expansion": bool(endpoint_fraction > 0.10),
            "fit_scope": "identically corrupted outer-training only",
        }
    oracle_tuning_grid_passed = not any(
        value["requires_grid_expansion"] for value in oracle_tuning_report.values()
    )
    tuning_grid_passed = arm_tuning_grid_passed and oracle_tuning_grid_passed

    superiority_passed = bool(
        contrast_report["eye_minus_both"]["holm_superiority_rejected_null"]
        and contrast_report["blood_minus_both"]["holm_superiority_rejected_null"]
        and contrast_report["eye_minus_both"]["same_favorable_sign_every_split_seed"]
        and contrast_report["blood_minus_both"]["same_favorable_sign_every_split_seed"]
    )
    if ARM_CONCAT not in enabled_arms:
        noninferiority = {
            "status": "not_evaluable_concat_arm_disabled",
            "margin": config.noninferiority_margin,
            "passed": False,
            "fail_closed": True,
        }
    elif config.noninferiority_margin is None:
        noninferiority = {
            "status": "not_evaluable_margin_unresolved",
            "margin": None,
            "passed": False,
            "fail_closed": True,
        }
    else:
        lower_bound = contrast_report["concat_minus_both"][
            "one_sided_lower_confidence_bound"
        ]
        noninferiority = {
            "status": "evaluated",
            "margin": config.noninferiority_margin,
            "margin_provenance": config.noninferiority_margin_provenance,
            "one_sided_lower_confidence_bound": lower_bound,
            "passed": bool(lower_bound > -config.noninferiority_margin),
            "fail_closed": True,
        }

    missingness_metadata = {
        stratum.name: {
            "evidence_class": stratum.evidence_class,
            "pattern": stratum.pattern,
            "blood_deletion_fraction": stratum.blood_deletion_fraction,
            "retinal_image_count": stratum.retinal_image_count,
            "graceful_degradation_margin": stratum.graceful_degradation_margin,
            "evaluation_available": stratum.evaluation_available,
        }
        for stratum in config.missingness_strata
    }
    for name, metadata in missingness_metadata.items():
        missing_report[name].update(metadata)

    family_weights: dict[str, float] = {}
    for target in manifest.targets:
        family_weights[target.family] = target.family_weight
    payload: dict[str, Any] = {
        "schema_version": "soft-patient-atlas-nested-evaluation-v2",
        "evaluation": {
            "n_patients": cohort.size,
            "n_primary_patients": int(cohort.primary_patient_mask.sum()),
            "outer_folds": config.outer_folds,
            "inner_folds": config.inner_folds,
            "split_seed_count": len(config.split_seeds),
            "enabled_arms": list(enabled_arms),
            "outer_test_count_min": min(outer_fold_test_counts),
            "outer_test_count_max": max(outer_fold_test_counts),
            "patient_level": True,
            "site_stratified": True,
            "canonical_fold_map_created_before_target_filtering": True,
            "representation_fit_received_outer_train_only": True,
            "same_fold_model_transformed_train_and_test": True,
            "cross_fold_coordinates_pooled": False,
            "outer_test_predictions_aggregated": True,
            "identical_demographic_transform_and_values_all_arms": True,
            "repeats_averaged_per_patient_before_bootstrap": True,
            "bootstrap_unit": "patient",
            "bootstrap_samples": config.bootstrap_samples,
            "confidence_level": config.confidence_level,
            "paired_contrast_inference": "centered patient-cluster basic bootstrap",
            "primary_multiplicity_bound": "Holm step-down one-sided lower bounds",
            "missingness_oracle_fit_received_corrupted_outer_train_only": True,
        },
        "primary_estimand": {
            "name": "family_balanced_outer_test_normalized_squared_loss",
            "target_manifest_schema": manifest.schema_version,
            "target_count": len(manifest.targets),
            "target_eligible_counts": {
                target.id: int(count)
                for target, count in zip(manifest.targets, target_counts)
            },
            "family_weights": family_weights,
            "normalization": "outer-training intercept-only mean squared loss",
            "primary_contrast_multiplicity": "Holm one-sided family-wise alpha 0.05",
        },
        "arms": arm_report,
        "paired_contrasts": contrast_report,
        "missingness_strata": missing_report,
        "inner_tuning": {
            "arms": tuning_report,
            "available_input_oracles": oracle_tuning_report,
        },
        "acceptance": {
            "superiority_to_each_single_modality_passed": superiority_passed,
            "combined_atlas_exceeds_standalone_information_sources_passed": (
                inference_summary["standalone_information_source_gate_passed"]
            ),
            "tested_pattern_missingness_gate_passed": inference_summary[
                "missingness_family_gate_passed"
            ],
            "missingness_hierarchy": inference_summary["missingness_hierarchy"],
            "missingness_family_control": inference_summary[
                "missingness_family_control"
            ],
            "tuning_grid_endpoint_gate_passed": tuning_grid_passed,
            "noninferiority_to_tuned_concat": noninferiority,
            "confirmatory_acceptance_passed": bool(
                superiority_passed
                and inference_summary["standalone_information_source_gate_passed"]
                and inference_summary["missingness_family_gate_passed"]
                and tuning_grid_passed
                and noninferiority["passed"]
            ),
            "deployment_or_clinical_benefit_claim_allowed": False,
            "external_validation_status": "not_evaluated",
        },
        "privacy": {
            "aggregate_only": True,
            "contains_patient_identifiers": False,
            "contains_row_level_predictions": False,
            "contains_latent_coordinates": False,
        },
    }
    assert_aggregate_only_payload(payload, forbidden_patient_ids=cohort.patient_ids)
    return NestedEvaluationReport(payload=MappingProxyType(payload))


def require_confirmatory_acceptance(report: NestedEvaluationReport) -> None:
    """Fail closed unless all currently evaluable confirmatory gates pass."""

    acceptance = report.payload.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("evaluation report lacks an acceptance block")
    noninferiority = acceptance.get("noninferiority_to_tuned_concat")
    if not isinstance(noninferiority, Mapping):
        raise ValueError("evaluation report lacks noninferiority status")
    if noninferiority.get("status") != "evaluated":
        raise RuntimeError("noninferiority margin is unresolved; acceptance is fail-closed")
    if not bool(acceptance.get("confirmatory_acceptance_passed", False)):
        raise RuntimeError("the prespecified confirmatory acceptance gates did not pass")
