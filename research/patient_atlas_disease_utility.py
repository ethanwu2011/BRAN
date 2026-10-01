"""Target-safety kernel for Patient Atlas V5 disease-utility evaluation.

The functions here are intentionally outcome-value agnostic.  They construct and
verify exclusion masks, physically erase forbidden clinical inputs, group economical
refits, and authenticate fold artifact metadata.  Patient-derived arrays may be passed
only inside the local evaluation process and are never serialized by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from eval_soft_patient_atlas import (
    FeatureView,
    FoldCoordinates,
    MissingnessStratum,
    RepresentationFitRequest,
)
from patient_atlas_disease_utility_registry import validate_registry_files


ARTIFACT_CONTRACT_SCHEMA = "patient-atlas-v5-disease-fold-artifact-contract-v1"
DEFAULT_REGISTRY_NAME = "PATIENT_ATLAS_V5_DISEASE_UTILITY_BENCHMARK_REGISTRY_V1.json"
CLINICAL_VALUES_KEY = "clinical_values"
CLINICAL_OBSERVED_KEY = "clinical_observed_mask"
CLINICAL_ELIGIBLE_KEY = "clinical_policy_eligible_mask"


class DiseaseUtilityIntegrityError(ValueError):
    """Raised when an exclusion or artifact boundary is not target-safe."""


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DiseaseUtilityIntegrityError(f"{path} must contain a JSON object")
    return value


@dataclass(frozen=True)
class DiseaseUtilityPlan:
    registry_sha256: str
    universal_profile_id: str
    profile_to_core_endpoints: Mapping[str, tuple[str, ...]]
    outer_folds: int

    @property
    def target_specific_profile_count(self) -> int:
        return len(self.profile_to_core_endpoints)

    @property
    def total_profile_count(self) -> int:
        return 1 + self.target_specific_profile_count

    @property
    def universal_refit_count(self) -> int:
        return self.outer_folds

    @property
    def full_refit_count(self) -> int:
        return self.total_profile_count * self.outer_folds

    def aggregate_only_summary(self) -> dict[str, Any]:
        return {
            "schema_version": "patient-atlas-v5-disease-utility-refit-plan-v1",
            "registry_sha256": self.registry_sha256,
            "outer_folds": self.outer_folds,
            "universal_profile_id": self.universal_profile_id,
            "universal_refit_count": self.universal_refit_count,
            "target_specific_profile_count": self.target_specific_profile_count,
            "total_profile_count": self.total_profile_count,
            "full_refit_count": self.full_refit_count,
            "profile_endpoint_counts": {
                profile: len(endpoints)
                for profile, endpoints in sorted(self.profile_to_core_endpoints.items())
            },
        }


def load_disease_utility_registry(
    project_root: str | Path,
    *,
    registry_name: str = DEFAULT_REGISTRY_NAME,
) -> tuple[Mapping[str, Any], str]:
    root = Path(project_root).resolve()
    registry_path = (root / registry_name).resolve()
    if registry_path.parent != root or registry_path.name != registry_name:
        raise DiseaseUtilityIntegrityError("registry must be the canonical project-root artifact")
    validate_registry_files(registry_path)
    registry = _load_mapping(registry_path)
    digest = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    return registry, digest


def build_refit_plan(
    registry: Mapping[str, Any],
    registry_sha256: str,
) -> DiseaseUtilityPlan:
    protocol = registry.get("fair_comparison_protocol", {})
    outer_folds = int(protocol.get("split", {}).get("outer_folds", 0))
    if outer_folds < 2:
        raise DiseaseUtilityIntegrityError("outer-fold count is missing or invalid")
    grouped: dict[str, list[str]] = {}
    for endpoint in registry.get("core_endpoints", []):
        profile = str(endpoint.get("v5_exclusion_profile", ""))
        endpoint_id = str(endpoint.get("id", ""))
        if not profile or not endpoint_id:
            raise DiseaseUtilityIntegrityError("core endpoint lacks id or exclusion profile")
        grouped.setdefault(profile, []).append(endpoint_id)
    frozen = MappingProxyType(
        {
            profile: tuple(sorted(endpoint_ids))
            for profile, endpoint_ids in sorted(grouped.items())
        }
    )
    return DiseaseUtilityPlan(
        registry_sha256=registry_sha256,
        universal_profile_id="circularity18_universal",
        profile_to_core_endpoints=frozen,
        outer_folds=outer_folds,
    )


def _profile_map(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for profile in registry.get("feature_exclusion_profiles", []):
        profile_id = str(profile.get("id", ""))
        if not profile_id or profile_id in result:
            raise DiseaseUtilityIntegrityError("exclusion profile ids are missing or duplicated")
        result[profile_id] = profile
    return result


def exclusion_mask(
    registry: Mapping[str, Any],
    *,
    profile_id: str,
    ordered_feature_names: Sequence[str],
) -> np.ndarray:
    names = tuple(str(value) for value in ordered_feature_names)
    if not names or len(names) != len(set(names)):
        raise DiseaseUtilityIntegrityError("ordered feature names must be nonempty and unique")
    profiles = _profile_map(registry)
    if profile_id not in profiles:
        raise DiseaseUtilityIntegrityError(f"unknown exclusion profile {profile_id!r}")
    target_excluded = tuple(str(value) for value in profiles[profile_id].get("fields", ()))
    global_excluded = tuple(
        str(value)
        for value in registry.get("global_clinical_policy", {}).get(
            "globally_excluded_fields", ()
        )
    )
    if len(target_excluded) != len(set(target_excluded)):
        raise DiseaseUtilityIntegrityError("exclusion profile contains duplicate fields")
    if len(global_excluded) != len(set(global_excluded)):
        raise DiseaseUtilityIntegrityError("global clinical policy contains duplicate fields")
    excluded = set(target_excluded) | set(global_excluded)
    missing = sorted(excluded - set(names))
    if missing:
        raise DiseaseUtilityIntegrityError(f"excluded fields are absent from feature order: {missing}")
    return np.asarray([name not in excluded for name in names], dtype=bool)


def apply_exclusion_profile_to_features(
    features: Mapping[str, np.ndarray],
    *,
    registry: Mapping[str, Any],
    profile_id: str,
    ordered_feature_names: Sequence[str],
) -> Mapping[str, np.ndarray]:
    """Return a copy whose forbidden values are zero and both masks are false.

    Physical erasure occurs before any preprocessing or frozen-tower inference.  The
    input policy must be artifact state (identical for every patient), not a patient
    availability feature.
    """

    required = {CLINICAL_VALUES_KEY, CLINICAL_OBSERVED_KEY, CLINICAL_ELIGIBLE_KEY}
    missing_keys = required - set(features)
    if missing_keys:
        raise DiseaseUtilityIntegrityError(
            f"feature mapping lacks target-safety arrays: {sorted(missing_keys)}"
        )
    values = np.asarray(features[CLINICAL_VALUES_KEY])
    observed = np.asarray(features[CLINICAL_OBSERVED_KEY])
    eligible = np.asarray(features[CLINICAL_ELIGIBLE_KEY])
    if values.ndim != 2 or observed.shape != values.shape or observed.dtype != np.bool_:
        raise DiseaseUtilityIntegrityError("clinical values/observed mask must align as [patients,fields]")
    if values.shape[1] != len(tuple(ordered_feature_names)):
        raise DiseaseUtilityIntegrityError("clinical matrix width differs from ordered feature names")
    if eligible.ndim == 1:
        if eligible.shape != (values.shape[1],) or eligible.dtype != np.bool_:
            raise DiseaseUtilityIntegrityError("one-dimensional clinical policy has wrong shape/type")
        eligible = np.broadcast_to(eligible[None, :], values.shape)
    elif eligible.shape != values.shape or eligible.dtype != np.bool_:
        raise DiseaseUtilityIntegrityError("clinical policy must be boolean [fields] or [patients,fields]")
    if values.shape[0] > 1 and not np.array_equal(
        eligible, np.broadcast_to(eligible[:1], eligible.shape)
    ):
        raise DiseaseUtilityIntegrityError("clinical policy may not vary by patient")

    keep = exclusion_mask(
        registry,
        profile_id=profile_id,
        ordered_feature_names=ordered_feature_names,
    )
    safe_values = np.asarray(values).copy()
    safe_observed = observed.copy()
    safe_eligible = np.asarray(eligible).copy()
    safe_values[:, ~keep] = 0
    safe_observed[:, ~keep] = False
    safe_eligible[:, ~keep] = False
    safe_observed &= safe_eligible
    safe_values = np.where(safe_observed, safe_values, 0)

    result = {str(name): np.asarray(array).copy() for name, array in features.items()}
    result[CLINICAL_VALUES_KEY] = safe_values
    result[CLINICAL_OBSERVED_KEY] = safe_observed
    result[CLINICAL_ELIGIBLE_KEY] = safe_eligible
    return MappingProxyType(result)


def assert_exclusion_applied(
    features: Mapping[str, np.ndarray],
    *,
    registry: Mapping[str, Any],
    profile_id: str,
    ordered_feature_names: Sequence[str],
) -> None:
    keep = exclusion_mask(
        registry,
        profile_id=profile_id,
        ordered_feature_names=ordered_feature_names,
    )
    values = np.asarray(features[CLINICAL_VALUES_KEY])
    observed = np.asarray(features[CLINICAL_OBSERVED_KEY])
    eligible = np.asarray(features[CLINICAL_ELIGIBLE_KEY])
    if eligible.ndim == 1:
        eligible = np.broadcast_to(eligible[None, :], values.shape)
    if values.ndim != 2 or observed.shape != values.shape or eligible.shape != values.shape:
        raise DiseaseUtilityIntegrityError("target-safe feature arrays have incompatible shapes")
    if np.any(values[:, ~keep] != 0):
        raise DiseaseUtilityIntegrityError("forbidden clinical values were not physically erased")
    if np.any(observed[:, ~keep]) or np.any(eligible[:, ~keep]):
        raise DiseaseUtilityIntegrityError("forbidden clinical masks remain visible or eligible")


def _read_only_copy(values: np.ndarray) -> np.ndarray:
    copied = np.asarray(values).copy()
    copied.setflags(write=False)
    return copied


def target_safe_feature_view(
    view: FeatureView,
    *,
    registry: Mapping[str, Any],
    profile_id: str,
    ordered_feature_names: Sequence[str],
) -> FeatureView:
    """Copy a representation view and erase every forbidden clinical field."""

    safe = apply_exclusion_profile_to_features(
        view.features,
        registry=registry,
        profile_id=profile_id,
        ordered_feature_names=ordered_feature_names,
    )
    assert_exclusion_applied(
        safe,
        registry=registry,
        profile_id=profile_id,
        ordered_feature_names=ordered_feature_names,
    )
    features = MappingProxyType(
        {str(name): _read_only_copy(values) for name, values in safe.items()}
    )
    return FeatureView(
        patient_ids=view.patient_ids,
        site_ids=view.site_ids,
        features=features,
        demographics=_read_only_copy(view.demographics),
        demographic_mask=_read_only_copy(view.demographic_mask),
    )


class _TargetSafeFittedFoldRepresentation:
    """Sanitize every later transform view before delegating to one fold model."""

    def __init__(
        self,
        fitted: Any,
        *,
        registry: Mapping[str, Any],
        profile_id: str,
        ordered_feature_names: Sequence[str],
    ) -> None:
        if not hasattr(fitted, "provenance"):
            raise DiseaseUtilityIntegrityError("fitted representation lacks provenance")
        self._fitted = fitted
        self.provenance = fitted.provenance
        self._registry = registry
        self._profile_id = profile_id
        self._ordered_feature_names = tuple(map(str, ordered_feature_names))

    def transform(
        self,
        view: FeatureView,
        *,
        arm: str,
        stratum: str,
        missingness: MissingnessStratum | None = None,
        concat_eye_dimension: int | None = None,
    ) -> FoldCoordinates:
        safe_view = target_safe_feature_view(
            view,
            registry=self._registry,
            profile_id=self._profile_id,
            ordered_feature_names=self._ordered_feature_names,
        )
        return self._fitted.transform(
            safe_view,
            arm=arm,
            stratum=stratum,
            missingness=missingness,
            concat_eye_dimension=concat_eye_dimension,
        )


class TargetSafeFoldRepresentationFactory:
    """Wrap any fold factory with one authenticated target-exclusion policy.

    The wrapped factory sees only sanitized fit/validation/calibration views. Its
    fitted object subsequently sees only sanitized train/test transform views. One
    aggregate-only artifact contract is retained per outer fold; patient identities
    and predictions are never retained here.
    """

    def __init__(
        self,
        base_factory: Any,
        *,
        registry: Mapping[str, Any],
        registry_sha256: str,
        profile_id: str,
        ordered_feature_names: Sequence[str],
    ) -> None:
        self.base_factory = base_factory
        self.registry = registry
        self.registry_sha256 = str(registry_sha256)
        self.profile_id = str(profile_id)
        self.ordered_feature_names = tuple(map(str, ordered_feature_names))
        exclusion_mask(
            registry,
            profile_id=self.profile_id,
            ordered_feature_names=self.ordered_feature_names,
        )
        self.artifact_contracts: list[dict[str, Any]] = []

    def fit(self, request: RepresentationFitRequest) -> _TargetSafeFittedFoldRepresentation:
        if request.expected_outer_test_patient_id_hash is None:
            raise DiseaseUtilityIntegrityError(
                "disease evaluation requires a committed outer-test identity hash"
            )
        safe_request = RepresentationFitRequest(
            fold_key=request.fold_key,
            fit=target_safe_feature_view(
                request.fit,
                registry=self.registry,
                profile_id=self.profile_id,
                ordered_feature_names=self.ordered_feature_names,
            ),
            validation=target_safe_feature_view(
                request.validation,
                registry=self.registry,
                profile_id=self.profile_id,
                ordered_feature_names=self.ordered_feature_names,
            ),
            calibration=target_safe_feature_view(
                request.calibration,
                registry=self.registry,
                profile_id=self.profile_id,
                ordered_feature_names=self.ordered_feature_names,
            ),
            expected_outer_train_patient_hash=request.expected_outer_train_patient_hash,
            expected_outer_test_patient_id_hash=request.expected_outer_test_patient_id_hash,
            expected_outer_test_patient_count=request.expected_outer_test_patient_count,
        )
        fitted = self.base_factory.fit(safe_request)
        if not hasattr(fitted, "provenance"):
            raise DiseaseUtilityIntegrityError("wrapped factory returned no provenance")
        provenance = fitted.provenance
        expected = safe_request.expected_provenance(str(provenance.model_token))
        if provenance != expected:
            raise DiseaseUtilityIntegrityError(
                "wrapped representation provenance differs from the target-safe request"
            )
        contract = make_artifact_contract(
            registry=self.registry,
            registry_sha256=self.registry_sha256,
            profile_id=self.profile_id,
            ordered_feature_names=self.ordered_feature_names,
            outer_fold_id=request.fold_key,
            outer_train_patient_id_hash=request.expected_outer_train_patient_hash,
            outer_test_patient_id_hash=request.expected_outer_test_patient_id_hash,
            model_state_sha256=str(provenance.model_token),
        )
        validate_artifact_contract(
            contract,
            registry=self.registry,
            registry_sha256=self.registry_sha256,
            profile_id=self.profile_id,
            ordered_feature_names=self.ordered_feature_names,
        )
        self.artifact_contracts.append(contract)
        return _TargetSafeFittedFoldRepresentation(
            fitted,
            registry=self.registry,
            profile_id=self.profile_id,
            ordered_feature_names=self.ordered_feature_names,
        )


def make_artifact_contract(
    *,
    registry: Mapping[str, Any],
    registry_sha256: str,
    profile_id: str,
    ordered_feature_names: Sequence[str],
    outer_fold_id: str,
    outer_train_patient_id_hash: str,
    outer_test_patient_id_hash: str,
    model_state_sha256: str,
) -> dict[str, Any]:
    profiles = _profile_map(registry)
    if profile_id not in profiles:
        raise DiseaseUtilityIntegrityError(f"unknown exclusion profile {profile_id!r}")
    target_fields = tuple(str(value) for value in profiles[profile_id].get("fields", ()))
    global_fields = tuple(
        str(value)
        for value in registry.get("global_clinical_policy", {}).get(
            "globally_excluded_fields", ()
        )
    )
    keep = exclusion_mask(
        registry,
        profile_id=profile_id,
        ordered_feature_names=ordered_feature_names,
    )
    policy_payload = {
        "ordered_feature_names_sha256": canonical_hash(list(map(str, ordered_feature_names))),
        "global_excluded_fields": list(global_fields),
        "target_excluded_fields": list(target_fields),
        "effective_excluded_fields": [
            str(name)
            for name, is_kept in zip(ordered_feature_names, keep)
            if not bool(is_kept)
        ],
        "keep_mask": keep.tolist(),
    }
    policy_hash = canonical_hash(policy_payload)
    for label, digest in (
        ("registry_sha256", registry_sha256),
        ("outer_train_patient_id_hash", outer_train_patient_id_hash),
        ("outer_test_patient_id_hash", outer_test_patient_id_hash),
        ("model_state_sha256", model_state_sha256),
    ):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise DiseaseUtilityIntegrityError(f"{label} must be lowercase SHA-256")
    if not outer_fold_id:
        raise DiseaseUtilityIntegrityError("outer_fold_id must not be empty")
    return {
        "schema_version": ARTIFACT_CONTRACT_SCHEMA,
        "registry_sha256": registry_sha256,
        "profile_id": profile_id,
        "policy": policy_payload,
        "policy_hashes": {
            "encoder_input": policy_hash,
            "reconstruction_target": policy_hash,
            "normalization_fit": policy_hash,
            "frozen_tower_input": policy_hash,
        },
        "outer_fold_id": outer_fold_id,
        "outer_train_patient_id_hash": outer_train_patient_id_hash,
        "outer_test_patient_id_hash": outer_test_patient_id_hash,
        "model_state_sha256": model_state_sha256,
        "target_values_received_by_representation_fit": False,
        "patient_rows_serialized": False,
        "patient_predictions_serialized": False,
    }


def validate_artifact_contract(
    contract: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    registry_sha256: str,
    profile_id: str,
    ordered_feature_names: Sequence[str],
) -> None:
    if contract.get("schema_version") != ARTIFACT_CONTRACT_SCHEMA:
        raise DiseaseUtilityIntegrityError("artifact contract schema differs")
    if contract.get("registry_sha256") != registry_sha256:
        raise DiseaseUtilityIntegrityError("artifact registry binding differs")
    if contract.get("profile_id") != profile_id:
        raise DiseaseUtilityIntegrityError("artifact exclusion profile differs")
    if contract.get("target_values_received_by_representation_fit") is not False:
        raise DiseaseUtilityIntegrityError("representation fitting received target values")
    if contract.get("patient_rows_serialized") is not False:
        raise DiseaseUtilityIntegrityError("artifact contract permits patient-row serialization")
    if contract.get("patient_predictions_serialized") is not False:
        raise DiseaseUtilityIntegrityError("artifact contract permits prediction serialization")

    profiles = _profile_map(registry)
    expected_target_fields = [str(value) for value in profiles[profile_id].get("fields", ())]
    expected_global_fields = [
        str(value)
        for value in registry.get("global_clinical_policy", {}).get(
            "globally_excluded_fields", ()
        )
    ]
    keep = exclusion_mask(
        registry,
        profile_id=profile_id,
        ordered_feature_names=ordered_feature_names,
    )
    expected_policy = {
        "ordered_feature_names_sha256": canonical_hash(list(map(str, ordered_feature_names))),
        "global_excluded_fields": expected_global_fields,
        "target_excluded_fields": expected_target_fields,
        "effective_excluded_fields": [
            str(name)
            for name, is_kept in zip(ordered_feature_names, keep)
            if not bool(is_kept)
        ],
        "keep_mask": keep.tolist(),
    }
    if contract.get("policy") != expected_policy:
        raise DiseaseUtilityIntegrityError("artifact target-safety policy differs")
    expected_hash = canonical_hash(expected_policy)
    hashes = contract.get("policy_hashes", {})
    required_hashes = {
        "encoder_input",
        "reconstruction_target",
        "normalization_fit",
        "frozen_tower_input",
    }
    if set(hashes) != required_hashes or any(
        hashes[name] != expected_hash for name in required_hashes
    ):
        raise DiseaseUtilityIntegrityError(
            "encoder, reconstruction, normalization, and tower policies are not identical"
        )


__all__ = [
    "ARTIFACT_CONTRACT_SCHEMA",
    "DiseaseUtilityIntegrityError",
    "DiseaseUtilityPlan",
    "TargetSafeFoldRepresentationFactory",
    "apply_exclusion_profile_to_features",
    "assert_exclusion_applied",
    "build_refit_plan",
    "canonical_hash",
    "exclusion_mask",
    "load_disease_utility_registry",
    "make_artifact_contract",
    "target_safe_feature_view",
    "validate_artifact_contract",
]
