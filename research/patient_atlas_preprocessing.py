"""Fold-local, outcome-free preprocessing for Patient Atlas v1.

This module is deliberately data-source agnostic.  It accepts in-memory arrays
only after a caller has authenticated the source schemas and assigned an outer
fold.  A fitted artifact contains aggregate transform parameters and hashes of
the exact identity partitions; it never serializes patient identifiers or
patient rows.

The public fitter is fail closed in three important ways:

* the supplied patient IDs must equal the representation-fit membership hash;
* ordered clinical features and the policy mask must match the authenticated
  schemas exactly; and
* a confirmatory/production real-data fit is refused while the feature registry
  declares unresolved blockers. An exploratory fit must carry the authenticated
  nonconfirmatory source-policy hash; synthetic tests opt in separately.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from patient_atlas_contracts import EXPECTED_COLUMNS_SHA256


PREPROCESSOR_SCHEMA_VERSION = "soft-patient-atlas-preprocessor-v1"
SPLIT_PROVENANCE_SCHEMA_VERSION = "patient-atlas-fold-provenance-v1"
FEATURE_SCHEMA_VERSION = "patient-atlas-feature-registry-v1"
CONTEXT_SCHEMA_VERSION = "patient-atlas-context-v1"
EYE_SCHEMA_VERSION = "patient-atlas-eye-registry-v1"
CONTINUOUS_POLICY_ID = "outer_fold_robust_continuous_v1"
BINARY_POLICY_ID = "binary_identity_v1"
AGE_POLICY_ID = "outer-fold-age-median-iqr-v1"
EYE_POLICY_ID = "outer-fold-zca-whitening-v1"
EYE_WEIGHTING_POLICY_ID = "equal-patient-uniform-visible-images-v1"


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical finite JSON encoding used by every Atlas hash in this module."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_loads(raw: bytes | str) -> Any:
    return json.loads(raw, object_pairs_hook=_strict_object)


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys mismatch; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _normalize_identifiers(
    identifiers: Sequence[str], *, label: str
) -> tuple[str, ...]:
    values = tuple(str(value) for value in identifiers)
    if not values:
        raise ValueError(f"{label} must not be empty")
    if any(not value for value in values):
        raise ValueError(f"{label} contains an empty identifier")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate identifiers")
    return tuple(sorted(values))


def hash_identifier_set(identifiers: Sequence[str], *, label: str) -> str:
    """Hash a set of identifiers without retaining it in the artifact."""

    normalized = _normalize_identifiers(identifiers, label=label)
    return hash_json(list(normalized))


@dataclass(frozen=True)
class FoldSplitProvenance:
    """Identifier-free hashes of one exact outer-fold partition."""

    schema_version: str
    outer_fold_id: str
    outer_train_patient_id_hash: str
    representation_fit_patient_id_hash: str
    validation_patient_id_hash: str
    calibration_patient_id_hash: str
    outer_test_patient_id_hash: str
    split_manifest_hash: str
    role_counts: Mapping[str, int]
    split_seed: int
    training_seed: int
    training_config_hash: str
    source_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != SPLIT_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("unsupported split-provenance schema")
        if not self.outer_fold_id:
            raise ValueError("outer_fold_id must not be empty")
        for name in (
            "outer_train_patient_id_hash",
            "representation_fit_patient_id_hash",
            "validation_patient_id_hash",
            "calibration_patient_id_hash",
            "outer_test_patient_id_hash",
            "split_manifest_hash",
            "training_config_hash",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        expected_roles = {
            "outer_train",
            "representation_fit",
            "validation",
            "calibration",
            "outer_test",
        }
        if set(self.role_counts) != expected_roles:
            raise ValueError("split provenance role_counts are incomplete")
        if any(
            not isinstance(value, int) or value <= 0
            for value in self.role_counts.values()
        ):
            raise ValueError("every split role count must be a positive integer")
        if self.role_counts["outer_train"] != (
            self.role_counts["representation_fit"]
            + self.role_counts["validation"]
            + self.role_counts["calibration"]
        ):
            raise ValueError("outer-training role counts do not partition exactly")
        if not isinstance(self.split_seed, int) or self.split_seed < 0:
            raise ValueError("split_seed must be a nonnegative integer")
        if not isinstance(self.training_seed, int) or self.training_seed < 0:
            raise ValueError("training_seed must be a nonnegative integer")
        if not self.source_hashes or any(
            not isinstance(key, str) or not key or not _is_sha256(value)
            for key, value in self.source_hashes.items()
        ):
            raise ValueError("source_hashes must be a nonempty name-to-SHA256 mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "outer_fold_id": self.outer_fold_id,
            "outer_train_patient_id_hash": self.outer_train_patient_id_hash,
            "representation_fit_patient_id_hash": (
                self.representation_fit_patient_id_hash
            ),
            "validation_patient_id_hash": self.validation_patient_id_hash,
            "calibration_patient_id_hash": self.calibration_patient_id_hash,
            "outer_test_patient_id_hash": self.outer_test_patient_id_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "role_counts": dict(sorted(self.role_counts.items())),
            "split_seed": self.split_seed,
            "training_seed": self.training_seed,
            "training_config_hash": self.training_config_hash,
            "source_hashes": dict(sorted(self.source_hashes.items())),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FoldSplitProvenance":
        expected = {
            "schema_version",
            "outer_fold_id",
            "outer_train_patient_id_hash",
            "representation_fit_patient_id_hash",
            "validation_patient_id_hash",
            "calibration_patient_id_hash",
            "outer_test_patient_id_hash",
            "split_manifest_hash",
            "role_counts",
            "split_seed",
            "training_seed",
            "training_config_hash",
            "source_hashes",
        }
        _require_exact_keys(value, expected, "split provenance")
        return cls(**dict(value))


def build_fold_split_provenance(
    *,
    outer_fold_id: str,
    outer_train_patient_ids: Sequence[str],
    representation_fit_patient_ids: Sequence[str],
    validation_patient_ids: Sequence[str],
    calibration_patient_ids: Sequence[str],
    outer_test_patient_ids: Sequence[str],
    split_seed: int,
    training_seed: int,
    training_config: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> FoldSplitProvenance:
    """Validate a complete partition and retain only exact aggregate hashes."""

    roles = {
        "outer_train": _normalize_identifiers(
            outer_train_patient_ids, label="outer_train_patient_ids"
        ),
        "representation_fit": _normalize_identifiers(
            representation_fit_patient_ids,
            label="representation_fit_patient_ids",
        ),
        "validation": _normalize_identifiers(
            validation_patient_ids, label="validation_patient_ids"
        ),
        "calibration": _normalize_identifiers(
            calibration_patient_ids, label="calibration_patient_ids"
        ),
        "outer_test": _normalize_identifiers(
            outer_test_patient_ids, label="outer_test_patient_ids"
        ),
    }
    fit = set(roles["representation_fit"])
    validation = set(roles["validation"])
    calibration = set(roles["calibration"])
    outer_train = set(roles["outer_train"])
    outer_test = set(roles["outer_test"])
    if fit & validation or fit & calibration or validation & calibration:
        raise ValueError("fit, validation, and calibration identities overlap")
    if fit | validation | calibration != outer_train:
        raise ValueError("fit/validation/calibration do not equal outer training")
    if outer_train & outer_test:
        raise ValueError("outer-train and outer-test identities overlap")
    if not isinstance(training_config, Mapping) or not training_config:
        raise ValueError("training_config must be a nonempty mapping")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("source_hashes must be a nonempty mapping")

    manifest_payload = {
        "outer_fold_id": str(outer_fold_id),
        "roles": {name: list(values) for name, values in sorted(roles.items())},
    }
    return FoldSplitProvenance(
        schema_version=SPLIT_PROVENANCE_SCHEMA_VERSION,
        outer_fold_id=str(outer_fold_id),
        outer_train_patient_id_hash=hash_json(list(roles["outer_train"])),
        representation_fit_patient_id_hash=hash_json(
            list(roles["representation_fit"])
        ),
        validation_patient_id_hash=hash_json(list(roles["validation"])),
        calibration_patient_id_hash=hash_json(list(roles["calibration"])),
        outer_test_patient_id_hash=hash_json(list(roles["outer_test"])),
        split_manifest_hash=hash_json(manifest_payload),
        role_counts={name: len(values) for name, values in roles.items()},
        split_seed=split_seed,
        training_seed=training_seed,
        training_config_hash=hash_json(dict(training_config)),
        source_hashes=dict(sorted((str(k), str(v)) for k, v in source_hashes.items())),
    )


@dataclass(frozen=True)
class PreprocessingSchemaContract:
    """Authenticated schema facts needed by fold preprocessing."""

    feature_schema_version: str
    feature_schema_hash: str
    ordered_feature_names: tuple[str, ...]
    feature_types: tuple[str, ...]
    ordered_features_hash: str
    continuous_count: int
    binary_count: int
    context_schema_version: str
    context_schema_hash: str
    eye_schema_version: str
    eye_schema_hash: str
    eye_dimension: int
    real_training_ready: bool
    blocking_gap_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("unsupported clinical feature schema")
        if self.context_schema_version != CONTEXT_SCHEMA_VERSION:
            raise ValueError("unsupported demographic context schema")
        if self.eye_schema_version != EYE_SCHEMA_VERSION:
            raise ValueError("unsupported eye registry schema")
        for name in ("feature_schema_hash", "context_schema_hash", "eye_schema_hash"):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.continuous_count <= 0 or self.binary_count <= 0:
            raise ValueError("continuous and binary feature counts must be positive")
        if len(self.ordered_feature_names) != self.continuous_count + self.binary_count:
            raise ValueError("ordered feature count does not match type counts")
        if len(self.feature_types) != len(self.ordered_feature_names):
            raise ValueError("feature type vector has the wrong width")
        expected_types = (
            ("continuous",) * self.continuous_count
            + ("binary",) * self.binary_count
        )
        if self.feature_types != expected_types:
            raise ValueError("clinical features are not continuous-then-binary")
        if len(set(self.ordered_feature_names)) != len(self.ordered_feature_names):
            raise ValueError("ordered clinical features contain duplicates")
        if hash_json(list(self.ordered_feature_names)) != self.ordered_features_hash:
            raise ValueError("ordered feature hash is inconsistent")
        if self.eye_dimension <= 0:
            raise ValueError("eye_dimension must be positive")
        if not isinstance(self.real_training_ready, bool):
            raise TypeError("real_training_ready must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_schema_version": self.feature_schema_version,
            "feature_schema_hash": self.feature_schema_hash,
            "ordered_feature_names": list(self.ordered_feature_names),
            "feature_types": list(self.feature_types),
            "ordered_features_hash": self.ordered_features_hash,
            "continuous_count": self.continuous_count,
            "binary_count": self.binary_count,
            "context_schema_version": self.context_schema_version,
            "context_schema_hash": self.context_schema_hash,
            "eye_schema_version": self.eye_schema_version,
            "eye_schema_hash": self.eye_schema_hash,
            "eye_dimension": self.eye_dimension,
            "real_training_ready": self.real_training_ready,
            "blocking_gap_ids": list(self.blocking_gap_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PreprocessingSchemaContract":
        expected = {
            "feature_schema_version",
            "feature_schema_hash",
            "ordered_feature_names",
            "feature_types",
            "ordered_features_hash",
            "continuous_count",
            "binary_count",
            "context_schema_version",
            "context_schema_hash",
            "eye_schema_version",
            "eye_schema_hash",
            "eye_dimension",
            "real_training_ready",
            "blocking_gap_ids",
        }
        _require_exact_keys(value, expected, "preprocessing schema contract")
        normalized = dict(value)
        normalized["ordered_feature_names"] = tuple(value["ordered_feature_names"])
        normalized["feature_types"] = tuple(value["feature_types"])
        normalized["blocking_gap_ids"] = tuple(value["blocking_gap_ids"])
        return cls(**normalized)


def _load_schema(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = _strict_json_loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"schema at {path} must be a JSON object")
    return value


def load_preprocessing_schema_contract(
    feature_registry_path: str | Path,
    context_schema_path: str | Path,
    eye_registry_path: str | Path,
) -> PreprocessingSchemaContract:
    """Authenticate the three schema registries used by the fold fitter."""

    feature = _load_schema(feature_registry_path)
    context = _load_schema(context_schema_path)
    eye = _load_schema(eye_registry_path)
    if feature.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("unexpected feature-registry schema version")
    if context.get("schema_version") != CONTEXT_SCHEMA_VERSION:
        raise ValueError("unexpected context schema version")
    if eye.get("schema_version") != EYE_SCHEMA_VERSION:
        raise ValueError("unexpected eye-registry schema version")

    features = feature.get("features")
    if not isinstance(features, list) or len(features) != feature.get("feature_count"):
        raise ValueError("feature registry count is inconsistent")
    indices = [entry.get("index") for entry in features]
    if indices != list(range(len(features))):
        raise ValueError("feature registry indices are not contiguous and ordered")
    names = tuple(str(entry.get("name")) for entry in features)
    types = tuple(str(entry.get("type")) for entry in features)
    ordered_hash = feature.get("ordered_columns_sha256")
    if (
        not _is_sha256(ordered_hash)
        or hash_json(list(names)) != ordered_hash
        or ordered_hash != EXPECTED_COLUMNS_SHA256
    ):
        raise ValueError("feature registry ordered-column hash is inconsistent")
    slices = feature.get("type_slices", {})
    continuous_end = int(slices.get("vitals_continuous", [0, -1])[1])
    binary_start, binary_end = map(
        int, slices.get("conditions_binary", [-1, -1])
    )
    if binary_start != continuous_end or binary_end != len(features):
        raise ValueError("feature registry type slices are inconsistent")
    if continuous_end != 48 or binary_end - binary_start != 11:
        raise ValueError("v1 clinical feature type widths must be 48 continuous and 11 binary")
    policies = feature.get("normalization_policies", {})
    if CONTINUOUS_POLICY_ID not in policies or BINARY_POLICY_ID not in policies:
        raise ValueError("required clinical normalization policies are missing")

    fields = context.get("fields")
    if not isinstance(fields, list) or len(fields) != 1 or fields[0].get("name") != "age":
        raise ValueError("v1 context schema must contain age only")
    age_transform = fields[0].get("outer_fold_transform", {})
    if age_transform.get("center") != "median" or age_transform.get("scale") != "IQR/1.349":
        raise ValueError("context age transform does not match v1")
    eye_dimension = eye.get("embedding", {}).get("dimension")
    if not isinstance(eye_dimension, int) or eye_dimension != 384:
        raise ValueError("eye registry embedding dimension is invalid")
    blockers = feature.get("blocking_gaps", ())
    if not isinstance(blockers, list) or any(
        not isinstance(entry, Mapping) or not entry.get("id") for entry in blockers
    ):
        raise ValueError("feature registry blocking_gaps are malformed")

    return PreprocessingSchemaContract(
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_hash=hash_json(feature),
        ordered_feature_names=names,
        feature_types=types,
        ordered_features_hash=ordered_hash,
        continuous_count=continuous_end,
        binary_count=binary_end - binary_start,
        context_schema_version=CONTEXT_SCHEMA_VERSION,
        context_schema_hash=hash_json(context),
        eye_schema_version=EYE_SCHEMA_VERSION,
        eye_schema_hash=hash_json(eye),
        eye_dimension=eye_dimension,
        real_training_ready=feature.get("real_training_ready"),
        blocking_gap_ids=tuple(str(entry["id"]) for entry in blockers),
    )


def policy_mask_hash(
    ordered_features_hash: str, policy_eligible_mask: Sequence[bool]
) -> str:
    values = tuple(policy_eligible_mask)
    if any(type(value) not in (bool, np.bool_) for value in values):
        raise TypeError("policy_eligible_mask must contain only booleans")
    return hash_json(
        {
            "ordered_features_hash": ordered_features_hash,
            "policy_eligible_mask": [bool(value) for value in values],
        }
    )


@dataclass(frozen=True)
class TransformedArray:
    values: np.ndarray
    observed_mask: np.ndarray


@dataclass(frozen=True)
class FoldPreprocessor:
    """Serializable fold transform containing aggregate parameters only."""

    schema_version: str
    fit_scope: str
    source_policy_sha256: str | None
    schemas: PreprocessingSchemaContract
    provenance: FoldSplitProvenance
    policy_eligible_mask: tuple[bool, ...]
    policy_mask_hash: str
    blood_log_flags: tuple[bool, ...]
    blood_medians: tuple[float, ...]
    blood_scales: tuple[float, ...]
    blood_fitted: tuple[bool, ...]
    blood_clip: tuple[float, float]
    age_median: float
    age_scale: float
    eye_mean: tuple[float, ...]
    eye_whitening: tuple[tuple[float, ...], ...]
    eye_eigenvalue_floor: float
    eye_weighting_policy: str
    eye_fit_patient_count: int
    eye_fit_image_count: int
    normalization_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != PREPROCESSOR_SCHEMA_VERSION:
            raise ValueError("unsupported preprocessor schema")
        if self.fit_scope not in {"real", "exploratory", "synthetic"}:
            raise ValueError("fit_scope must be real, exploratory, or synthetic")
        if self.fit_scope == "synthetic":
            if self.source_policy_sha256 is not None:
                raise ValueError("synthetic preprocessing cannot claim a source policy")
        elif not _is_sha256(self.source_policy_sha256):
            raise ValueError("real/exploratory preprocessing requires a source-policy SHA-256")
        width = len(self.schemas.ordered_feature_names)
        if len(self.policy_eligible_mask) != width or any(
            type(value) is not bool for value in self.policy_eligible_mask
        ):
            raise TypeError("policy_eligible_mask has the wrong boolean width")
        if self.policy_mask_hash != policy_mask_hash(
            self.schemas.ordered_features_hash, self.policy_eligible_mask
        ):
            raise ValueError("policy mask hash is inconsistent")
        continuous = self.schemas.continuous_count
        arrays = (
            self.blood_log_flags,
            self.blood_medians,
            self.blood_scales,
            self.blood_fitted,
        )
        if any(len(values) != continuous for values in arrays):
            raise ValueError("blood normalization vectors have the wrong width")
        if any(type(value) is not bool for value in self.blood_log_flags):
            raise TypeError("blood_log_flags must contain booleans")
        if any(type(value) is not bool for value in self.blood_fitted):
            raise TypeError("blood_fitted must contain booleans")
        numeric = (
            *self.blood_medians,
            *self.blood_scales,
            *self.blood_clip,
            self.age_median,
            self.age_scale,
            *self.eye_mean,
            self.eye_eigenvalue_floor,
        )
        if not all(np.isfinite(float(value)) for value in numeric):
            raise ValueError("preprocessor contains nonfinite statistics")
        if any(float(value) <= 0 for value in self.blood_scales):
            raise ValueError("blood scales must be positive")
        if self.blood_clip[0] >= self.blood_clip[1]:
            raise ValueError("blood clipping interval is invalid")
        if self.age_scale <= 0:
            raise ValueError("age scale must be positive")
        eye_dimension = self.schemas.eye_dimension
        if len(self.eye_mean) != eye_dimension or len(self.eye_whitening) != eye_dimension:
            raise ValueError("eye transform has the wrong dimension")
        if any(len(row) != eye_dimension for row in self.eye_whitening):
            raise ValueError("eye whitening matrix is not square")
        whitening = np.asarray(self.eye_whitening, dtype=np.float64)
        if not np.isfinite(whitening).all():
            raise ValueError("eye whitening matrix contains nonfinite values")
        if self.eye_weighting_policy != EYE_WEIGHTING_POLICY_ID:
            raise ValueError("unsupported eye-moment weighting policy")
        if (
            self.eye_eigenvalue_floor <= 0
            or self.eye_fit_patient_count < 2
            or self.eye_fit_patient_count
            > self.provenance.role_counts["representation_fit"]
            or self.eye_fit_image_count < self.eye_fit_patient_count
        ):
            raise ValueError("eye whitening fit provenance is invalid")
        expected_hashes = self._computed_normalization_hashes()
        if dict(self.normalization_hashes) != expected_hashes:
            raise ValueError("normalization/whitening hashes are inconsistent")

    def _blood_payload(self) -> dict[str, Any]:
        return {
            "policy_id": CONTINUOUS_POLICY_ID,
            "binary_policy_id": BINARY_POLICY_ID,
            "ordered_features_hash": self.schemas.ordered_features_hash,
            "policy_mask_hash": self.policy_mask_hash,
            "source_policy_sha256": self.source_policy_sha256,
            "log_flags": list(self.blood_log_flags),
            "medians": list(self.blood_medians),
            "scales": list(self.blood_scales),
            "fitted": list(self.blood_fitted),
            "clip": list(self.blood_clip),
            "fit_patient_id_hash": (
                self.provenance.representation_fit_patient_id_hash
            ),
        }

    def _age_payload(self) -> dict[str, Any]:
        return {
            "policy_id": AGE_POLICY_ID,
            "context_schema_hash": self.schemas.context_schema_hash,
            "source_policy_sha256": self.source_policy_sha256,
            "median": self.age_median,
            "scale": self.age_scale,
            "fit_patient_id_hash": (
                self.provenance.representation_fit_patient_id_hash
            ),
        }

    def _eye_payload(self) -> dict[str, Any]:
        return {
            "policy_id": EYE_POLICY_ID,
            "eye_schema_hash": self.schemas.eye_schema_hash,
            "source_policy_sha256": self.source_policy_sha256,
            "mean": list(self.eye_mean),
            "whitening": [list(row) for row in self.eye_whitening],
            "eigenvalue_floor": self.eye_eigenvalue_floor,
            "weighting_policy": self.eye_weighting_policy,
            "fit_patient_count": self.eye_fit_patient_count,
            "fit_image_count": self.eye_fit_image_count,
            "fit_patient_id_hash": (
                self.provenance.representation_fit_patient_id_hash
            ),
        }

    def _computed_normalization_hashes(self) -> dict[str, str]:
        return {
            "blood": hash_json(self._blood_payload()),
            "age": hash_json(self._age_payload()),
            "eye": hash_json(self._eye_payload()),
        }

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fit_scope": self.fit_scope,
            "source_policy_sha256": self.source_policy_sha256,
            "schemas": self.schemas.to_dict(),
            "provenance": self.provenance.to_dict(),
            "policy_eligible_mask": list(self.policy_eligible_mask),
            "policy_mask_hash": self.policy_mask_hash,
            "blood": {
                "log_flags": list(self.blood_log_flags),
                "medians": list(self.blood_medians),
                "scales": list(self.blood_scales),
                "fitted": list(self.blood_fitted),
                "clip": list(self.blood_clip),
            },
            "age": {"median": self.age_median, "scale": self.age_scale},
            "eye": {
                "mean": list(self.eye_mean),
                "whitening": [list(row) for row in self.eye_whitening],
                "eigenvalue_floor": self.eye_eigenvalue_floor,
                "weighting_policy": self.eye_weighting_policy,
                "fit_patient_count": self.eye_fit_patient_count,
                "fit_image_count": self.eye_fit_image_count,
            },
            "normalization_hashes": dict(sorted(self.normalization_hashes.items())),
        }

    @property
    def bundle_sha256(self) -> str:
        return hash_json(self._payload_dict())

    def to_serialized_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        return {
            "payload": payload,
            "payload_sha256": hash_json(payload),
        }

    def save(self, path: str | Path) -> str:
        """Create a canonical JSON artifact without overwriting an existing file."""

        raw = canonical_json_bytes(self.to_serialized_dict())
        with Path(path).open("xb") as handle:
            handle.write(raw)
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "FoldPreprocessor":
        value = _strict_json_loads(Path(path).read_bytes())
        if not isinstance(value, Mapping):
            raise TypeError("preprocessor bundle must be a JSON object")
        _require_exact_keys(value, {"payload", "payload_sha256"}, "preprocessor bundle")
        payload = value["payload"]
        if not isinstance(payload, Mapping) or hash_json(payload) != value["payload_sha256"]:
            raise ValueError("preprocessor payload hash mismatch")
        expected = {
            "schema_version",
            "fit_scope",
            "source_policy_sha256",
            "schemas",
            "provenance",
            "policy_eligible_mask",
            "policy_mask_hash",
            "blood",
            "age",
            "eye",
            "normalization_hashes",
        }
        _require_exact_keys(payload, expected, "preprocessor payload")
        blood = payload["blood"]
        age = payload["age"]
        eye = payload["eye"]
        if not all(isinstance(value, Mapping) for value in (blood, age, eye)):
            raise TypeError("preprocessor transform sections must be mappings")
        _require_exact_keys(
            blood, {"log_flags", "medians", "scales", "fitted", "clip"}, "blood transform"
        )
        _require_exact_keys(age, {"median", "scale"}, "age transform")
        _require_exact_keys(
            eye,
            {
                "mean",
                "whitening",
                "eigenvalue_floor",
                "weighting_policy",
                "fit_patient_count",
                "fit_image_count",
            },
            "eye transform",
        )
        return cls(
            schema_version=payload["schema_version"],
            fit_scope=payload["fit_scope"],
            source_policy_sha256=payload["source_policy_sha256"],
            schemas=PreprocessingSchemaContract.from_dict(payload["schemas"]),
            provenance=FoldSplitProvenance.from_dict(payload["provenance"]),
            policy_eligible_mask=tuple(payload["policy_eligible_mask"]),
            policy_mask_hash=payload["policy_mask_hash"],
            blood_log_flags=tuple(blood["log_flags"]),
            blood_medians=tuple(float(value) for value in blood["medians"]),
            blood_scales=tuple(float(value) for value in blood["scales"]),
            blood_fitted=tuple(blood["fitted"]),
            blood_clip=tuple(float(value) for value in blood["clip"]),
            age_median=float(age["median"]),
            age_scale=float(age["scale"]),
            eye_mean=tuple(float(value) for value in eye["mean"]),
            eye_whitening=tuple(
                tuple(float(value) for value in row) for row in eye["whitening"]
            ),
            eye_eigenvalue_floor=float(eye["eigenvalue_floor"]),
            eye_weighting_policy=str(eye["weighting_policy"]),
            eye_fit_patient_count=int(eye["fit_patient_count"]),
            eye_fit_image_count=int(eye["fit_image_count"]),
            normalization_hashes=dict(payload["normalization_hashes"]),
        )

    def _validate_runtime_blood_contract(
        self,
        ordered_feature_names: Sequence[str],
        policy_eligible_mask: Sequence[bool],
    ) -> np.ndarray:
        names = tuple(str(value) for value in ordered_feature_names)
        if names != self.schemas.ordered_feature_names:
            if set(names) == set(self.schemas.ordered_feature_names):
                raise ValueError("clinical feature columns are reordered")
            raise ValueError("clinical feature schema has missing or unknown columns")
        policy = np.asarray(policy_eligible_mask)
        if policy.shape != (len(names),) or policy.dtype != np.bool_:
            raise TypeError("policy_eligible_mask must be boolean with shape [features]")
        if tuple(bool(value) for value in policy) != self.policy_eligible_mask:
            raise ValueError("runtime blood policy mask does not match the fitted artifact")
        return policy

    def transform_blood(
        self,
        values: np.ndarray,
        observed_mask: np.ndarray,
        *,
        ordered_feature_names: Sequence[str],
        policy_eligible_mask: Sequence[bool],
    ) -> TransformedArray:
        """Robustly transform continuous fields and preserve binary 0/1 values."""

        policy = self._validate_runtime_blood_contract(
            ordered_feature_names, policy_eligible_mask
        )
        raw = _numeric_array(values, label="blood values")
        observed = _boolean_array(observed_mask, raw.shape, label="blood observed mask")
        if raw.ndim != 2 or raw.shape[1] != len(self.schemas.ordered_feature_names):
            raise ValueError("blood values must have shape [patients, features]")
        effective = observed & policy[None, :] & np.isfinite(raw)
        output = np.zeros(raw.shape, dtype=np.float32)
        nc = self.schemas.continuous_count
        for index in range(nc):
            valid = effective[:, index]
            column = raw[:, index].copy()
            if self.blood_log_flags[index]:
                valid &= column > 0
                column[valid] = np.log(column[valid])
            effective[:, index] = valid
            if valid.any() and not self.blood_fitted[index]:
                raise ValueError(
                    "a continuous feature visible at transform had no fit-set observations: "
                    f"{self.schemas.ordered_feature_names[index]}"
                )
            if valid.any():
                standardized = (
                    column[valid] - self.blood_medians[index]
                ) / self.blood_scales[index]
                output[valid, index] = np.clip(
                    standardized, self.blood_clip[0], self.blood_clip[1]
                ).astype(np.float32)

        binary = raw[:, nc:]
        binary_valid = effective[:, nc:] & ((binary == 0) | (binary == 1))
        effective[:, nc:] = binary_valid
        output[:, nc:] = np.where(binary_valid, binary, 0).astype(np.float32)
        output[~effective] = 0.0
        return TransformedArray(output, effective)

    def transform_age(
        self,
        ages: np.ndarray,
        observed_mask: np.ndarray,
        *,
        require_observed: bool = True,
    ) -> TransformedArray:
        raw = _numeric_array(ages, label="age")
        if raw.ndim == 1:
            raw = raw[:, None]
        if raw.ndim != 2 or raw.shape[1] != 1:
            raise ValueError("age must have shape [patients] or [patients,1]")
        mask = np.asarray(observed_mask)
        if mask.ndim == 1:
            mask = mask[:, None]
        observed = _boolean_array(mask, raw.shape, label="age observed mask")
        valid = observed & np.isfinite(raw) & (raw >= 0)
        if require_observed and not bool(valid.all()):
            raise ValueError("confirmatory v1 requires finite observed nonnegative age")
        output = np.zeros(raw.shape, dtype=np.float32)
        output[valid] = ((raw[valid] - self.age_median) / self.age_scale).astype(
            np.float32
        )
        return TransformedArray(output, valid)

    def transform_eye(
        self, eye_embeddings: np.ndarray, observed_mask: np.ndarray
    ) -> TransformedArray:
        raw = _numeric_array(eye_embeddings, label="eye embeddings")
        if raw.ndim != 3 or raw.shape[2] != self.schemas.eye_dimension:
            raise ValueError("eye embeddings must have shape [patients,images,eye_dim]")
        observed = _boolean_array(
            observed_mask, raw.shape[:2], label="eye observed mask"
        )
        visible = raw[observed]
        if visible.size and not np.isfinite(visible).all():
            raise ValueError("visible eye embeddings must be finite")
        output = np.zeros(raw.shape, dtype=np.float32)
        if len(visible):
            mean = np.asarray(self.eye_mean, dtype=np.float64)
            whitening = np.asarray(self.eye_whitening, dtype=np.float64)
            output[observed] = ((visible - mean) @ whitening).astype(np.float32)
        return TransformedArray(output, observed.copy())


def _numeric_array(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{label} must be a numeric array")
    return array.astype(np.float64, copy=False)


def _boolean_array(value: np.ndarray, shape: tuple[int, ...], *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.bool_:
        raise TypeError(f"{label} must be boolean with shape {shape}")
    return array


def fit_outer_fold_preprocessor(
    *,
    patient_ids: Sequence[str],
    blood_values: np.ndarray,
    blood_observed_mask: np.ndarray,
    ordered_feature_names: Sequence[str],
    policy_eligible_mask: Sequence[bool],
    expected_policy_mask_hash: str,
    ages: np.ndarray,
    age_observed_mask: np.ndarray,
    eye_embeddings: np.ndarray,
    eye_observed_mask: np.ndarray,
    schemas: PreprocessingSchemaContract,
    provenance: FoldSplitProvenance,
    expected_schema_hashes: Mapping[str, str],
    fit_scope: str = "real",
    source_policy_sha256: str | None = None,
) -> FoldPreprocessor:
    """Fit all transforms on exactly the representation-fit patients.

    ``fit_scope='exploratory'`` is a real-data but explicitly nonconfirmatory
    path bound to a reviewed source-policy hash. ``fit_scope='synthetic'`` is a
    separate test-only escape hatch. Both labels are serialized.
    """

    if fit_scope not in {"real", "exploratory", "synthetic"}:
        raise ValueError("fit_scope must be real, exploratory, or synthetic")
    observed_schema_hashes = {
        "feature": schemas.feature_schema_hash,
        "context": schemas.context_schema_hash,
        "eye": schemas.eye_schema_hash,
    }
    if not isinstance(expected_schema_hashes, Mapping) or set(expected_schema_hashes) != set(
        observed_schema_hashes
    ):
        raise ValueError("expected_schema_hashes must contain exactly feature, context, and eye")
    if any(not _is_sha256(value) for value in expected_schema_hashes.values()):
        raise ValueError("expected_schema_hashes contains an invalid SHA-256 digest")
    if dict(expected_schema_hashes) != observed_schema_hashes:
        raise ValueError("preprocessing schema hash does not match the authenticated release")
    if fit_scope == "real" and not schemas.real_training_ready:
        gaps = ", ".join(schemas.blocking_gap_ids)
        raise RuntimeError(
            "feature registry is not ready for real Atlas training; unresolved: "
            + gaps
        )
    if fit_scope == "synthetic":
        if source_policy_sha256 is not None:
            raise ValueError("synthetic preprocessing cannot claim a source policy")
    elif not _is_sha256(source_policy_sha256):
        raise ValueError("real/exploratory preprocessing requires a source-policy SHA-256")
    if fit_scope == "real" and (
        schemas.ordered_features_hash != EXPECTED_COLUMNS_SHA256
        or schemas.continuous_count != 48
        or schemas.binary_count != 11
        or schemas.eye_dimension != 384
    ):
        raise ValueError("real v1 preprocessing schemas do not match production dimensions")
    normalized_ids = _normalize_identifiers(patient_ids, label="patient_ids")
    if hash_json(list(normalized_ids)) != provenance.representation_fit_patient_id_hash:
        raise ValueError("fit patient membership does not match fold provenance")
    if len(normalized_ids) != provenance.role_counts["representation_fit"]:
        raise ValueError("fit patient count does not match fold provenance")
    size = len(patient_ids)

    names = tuple(str(value) for value in ordered_feature_names)
    if names != schemas.ordered_feature_names:
        if set(names) == set(schemas.ordered_feature_names):
            raise ValueError("clinical feature columns are reordered")
        raise ValueError("clinical feature schema has missing or unknown columns")
    policy = np.asarray(policy_eligible_mask)
    if policy.shape != (len(names),) or policy.dtype != np.bool_:
        raise TypeError("policy_eligible_mask must be boolean with shape [features]")
    if not bool(policy.any()):
        raise ValueError("policy_eligible_mask cannot forbid every clinical feature")
    mask_hash = policy_mask_hash(schemas.ordered_features_hash, policy)
    if not _is_sha256(expected_policy_mask_hash):
        raise ValueError("expected_policy_mask_hash must be a lowercase SHA-256 digest")
    if mask_hash != expected_policy_mask_hash:
        raise ValueError("clinical policy mask does not match its authenticated hash")

    blood = _numeric_array(blood_values, label="blood values")
    if blood.shape != (size, len(names)):
        raise ValueError("blood values have the wrong shape")
    blood_mask = _boolean_array(
        blood_observed_mask, blood.shape, label="blood observed mask"
    )
    effective_blood = blood_mask & policy[None, :] & np.isfinite(blood)
    log_flags: list[bool] = []
    medians: list[float] = []
    scales: list[float] = []
    fitted: list[bool] = []
    for index in range(schemas.continuous_count):
        observed = blood[effective_blood[:, index], index].astype(np.float64)
        is_fitted = bool(len(observed))
        use_log = bool(
            len(observed) > 50
            and bool((observed > 0).all())
            and observed.max() / max(float(np.median(observed)), 1e-9) > 20
        )
        transformed = np.log(observed) if use_log else observed
        median = float(np.median(transformed)) if len(transformed) else 0.0
        if len(transformed):
            q25, q75 = np.quantile(transformed, [0.25, 0.75])
        else:
            q25, q75 = 0.0, 0.0
        scale = float((q75 - q25) / 1.349)
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        log_flags.append(use_log)
        medians.append(median)
        scales.append(scale)
        fitted.append(is_fitted)

    age = _numeric_array(ages, label="age")
    if age.ndim == 1:
        age = age[:, None]
    if age.shape != (size, 1):
        raise ValueError("age must have shape [patients] or [patients,1]")
    raw_age_mask = np.asarray(age_observed_mask)
    if raw_age_mask.ndim == 1:
        raw_age_mask = raw_age_mask[:, None]
    age_mask = _boolean_array(raw_age_mask, age.shape, label="age observed mask")
    if not bool(age_mask.all()) or not np.isfinite(age).all() or bool((age < 0).any()):
        raise ValueError("fit-set age must be finite, observed, and nonnegative")
    age_values = age[:, 0]
    age_median = float(np.median(age_values))
    age_q25, age_q75 = np.quantile(age_values, [0.25, 0.75])
    age_scale = float((age_q75 - age_q25) / 1.349)
    if not np.isfinite(age_scale) or age_scale <= 1e-8:
        age_scale = 1.0

    eye = _numeric_array(eye_embeddings, label="eye embeddings")
    if eye.shape[:1] != (size,) or eye.ndim != 3 or eye.shape[2] != schemas.eye_dimension:
        raise ValueError("eye embeddings have the wrong shape")
    eye_mask = _boolean_array(
        eye_observed_mask, eye.shape[:2], label="eye observed mask"
    )
    visible_eye = eye[eye_mask]
    image_counts = eye_mask.sum(axis=1, dtype=np.int64)
    contributing_patients = image_counts > 0
    eye_fit_patient_count = int(contributing_patients.sum())
    if eye_fit_patient_count < 2:
        raise ValueError("at least two fit-set patients with visible eye embeddings are required")
    if not np.isfinite(visible_eye).all():
        raise ValueError("visible fit-set eye embeddings must be finite")
    # Match the Stage-2 patient-averaged eye likelihood: every contributing
    # patient has total weight one, divided uniformly over that patient's
    # visible images.  Repeating an entire image set therefore cannot make a
    # heavily imaged patient or device protocol dominate the fold moments.
    patient_inverse_counts = np.zeros(size, dtype=np.float64)
    patient_inverse_counts[contributing_patients] = 1.0 / image_counts[
        contributing_patients
    ]
    image_weights = eye_mask.astype(np.float64) * patient_inverse_counts[:, None]
    total_patient_weight = float(eye_fit_patient_count)
    safe_eye = np.where(eye_mask[..., None], eye, 0.0)
    eye_mean = (
        safe_eye * image_weights[..., None]
    ).sum(axis=(0, 1), dtype=np.float64) / total_patient_weight
    centered = np.where(eye_mask[..., None], eye - eye_mean, 0.0)
    covariance = np.einsum(
        "bni,bnj,bn->ij",
        centered,
        centered,
        image_weights,
        optimize=True,
    ) / total_patient_weight
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = max(float(eigenvalues.max()), 0.0)
    eigenvalue_floor = max(1e-8, maximum * 1e-6) if maximum > 0 else 1.0
    regularized = np.maximum(eigenvalues, eigenvalue_floor)
    whitening = (eigenvectors * np.power(regularized, -0.5)[None, :]) @ eigenvectors.T
    whitening = 0.5 * (whitening + whitening.T)

    temporary = object.__new__(FoldPreprocessor)
    # Compute the transform hashes from the same immutable payloads that the
    # completed dataclass will expose.  object.__setattr__ is used only during
    # construction of this frozen aggregate-statistics object.
    values = {
        "schema_version": PREPROCESSOR_SCHEMA_VERSION,
        "fit_scope": fit_scope,
        "source_policy_sha256": source_policy_sha256,
        "schemas": schemas,
        "provenance": provenance,
        "policy_eligible_mask": tuple(bool(value) for value in policy),
        "policy_mask_hash": mask_hash,
        "blood_log_flags": tuple(log_flags),
        "blood_medians": tuple(medians),
        "blood_scales": tuple(scales),
        "blood_fitted": tuple(fitted),
        "blood_clip": (-10.0, 10.0),
        "age_median": age_median,
        "age_scale": age_scale,
        "eye_mean": tuple(float(value) for value in eye_mean),
        "eye_whitening": tuple(
            tuple(float(value) for value in row) for row in whitening
        ),
        "eye_eigenvalue_floor": eigenvalue_floor,
        "eye_weighting_policy": EYE_WEIGHTING_POLICY_ID,
        "eye_fit_patient_count": eye_fit_patient_count,
        "eye_fit_image_count": int(len(visible_eye)),
    }
    for name, value in values.items():
        object.__setattr__(temporary, name, value)
    object.__setattr__(temporary, "normalization_hashes", {})
    hashes = temporary._computed_normalization_hashes()
    return FoldPreprocessor(**values, normalization_hashes=hashes)


__all__ = [
    "EYE_WEIGHTING_POLICY_ID",
    "FoldPreprocessor",
    "FoldSplitProvenance",
    "PreprocessingSchemaContract",
    "TransformedArray",
    "build_fold_split_provenance",
    "canonical_json_bytes",
    "fit_outer_fold_preprocessor",
    "hash_identifier_set",
    "hash_json",
    "load_preprocessing_schema_contract",
    "policy_mask_hash",
]
