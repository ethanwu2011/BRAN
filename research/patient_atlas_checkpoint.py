"""Safe fold-checkpoint and provenance bundle for Patient Atlas v1.

The format is a ZIP container of canonical JSON plus non-pickled ``.npy``
tensors.  The complete file is read once, every member and tensor is hashed,
and metadata expectations are checked before a model is mutated.  Loading
therefore never executes pickle code and fails closed on schema, fold,
preprocessing, feature-order, policy, tower, factor-order, or calibration
mismatches.

Only code/config/schema hashes and aggregate transform parameters belong in
this checkpoint.  Optimizer state, patient identifiers, outcomes, source paths,
and fold histories are structurally absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping
import zipfile

import numpy as np
import torch
import torch.nn as nn

from patient_atlas_contracts import hash_state_dict
from patient_atlas_preprocessing import (
    FoldPreprocessor,
    canonical_json_bytes,
    hash_json,
)


CHECKPOINT_SCHEMA_VERSION = "soft-patient-atlas-fold-v1"
CHECKPOINT_MODEL_TYPE = "soft_group_factor_evidence_fusion"
MAX_CHECKPOINT_BYTES = 1 << 30
MAX_CHECKPOINT_MEMBERS = 2_048
MAX_MANIFEST_BYTES = 2 << 20
MAX_TOTAL_UNCOMPRESSED_BYTES = 64 << 20
_MANIFEST_MEMBER = "manifest.json"
_METADATA_FIELDS = {
    "schema_version",
    "model_type",
    "model_config",
    "patient_state_schema",
    "factor_capacity_and_ordering",
    "demographic_context_schema",
    "typed_feature_schema_and_order",
    "mask_policy_and_hash",
    "normalization_and_whitening_hashes",
    "eye_preprocessing_contract",
    "preprocessor_bundle_sha256",
    "tower_keys_and_state_hashes",
    "fold_manifest_and_training_id_hash",
    "initializer_hash",
    "paired_correction_gate_result",
    "calibration_parameters",
    "code_config_source_hashes",
    "seed",
    "creation_time_utc",
}
_MANIFEST_FIELDS = {
    "schema_version",
    "model_type",
    "metadata",
    "metadata_sha256",
    "state_sha256",
    "tensor_entries",
}
_TENSOR_ENTRY_FIELDS = {"name", "member", "dtype", "shape", "npy_sha256"}
_BANNED_METADATA_KEYS = {
    "patient_id",
    "patient_ids",
    "pid",
    "pids",
    "outcome",
    "outcomes",
    "labels",
    "optimizer",
    "optimizer_state",
    "filesystem_path",
    "source_path",
    "fold_history",
}
_DTYPE_BY_NAME = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_loads(raw: bytes) -> Any:
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


def _validate_json_tree(value: Any, *, path: tuple[str, ...] = ()) -> None:
    """Reject nonfinite/non-JSON content and row-level metadata key names."""

    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, str):
        if value.startswith(("/", "file://")) or (
            len(value) >= 3
            and value[0].isalpha()
            and value[1] == ":"
            and value[2] in {"\\", "/"}
        ):
            raise ValueError(f"filesystem paths are forbidden at {'.'.join(path)}")
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"nonfinite metadata value at {'.'.join(path)}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(item, path=(*path, str(index)))
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError("metadata mapping keys must be nonempty strings")
            if key.lower() in _BANNED_METADATA_KEYS:
                raise ValueError(f"row-level or restricted metadata key is forbidden: {key}")
            _validate_json_tree(item, path=(*path, key))
        return
    raise TypeError(f"metadata is not canonical JSON at {'.'.join(path)}")


@dataclass(frozen=True)
class AtlasCheckpointMetadata:
    schema_version: str
    model_type: str
    model_config: Mapping[str, Any]
    patient_state_schema: Mapping[str, Any]
    factor_capacity_and_ordering: Mapping[str, Any]
    demographic_context_schema: Mapping[str, Any]
    typed_feature_schema_and_order: Mapping[str, Any]
    mask_policy_and_hash: Mapping[str, Any]
    normalization_and_whitening_hashes: Mapping[str, str]
    eye_preprocessing_contract: Mapping[str, Any]
    preprocessor_bundle_sha256: str
    tower_keys_and_state_hashes: Mapping[str, str]
    fold_manifest_and_training_id_hash: Mapping[str, Any]
    initializer_hash: str
    paired_correction_gate_result: Mapping[str, Any]
    calibration_parameters: Mapping[str, Any]
    code_config_source_hashes: Mapping[str, str]
    seed: int
    creation_time_utc: str

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported Atlas checkpoint schema")
        if self.model_type != CHECKPOINT_MODEL_TYPE:
            raise ValueError("unexpected Atlas checkpoint model type")
        _validate_json_tree(self.to_dict())
        required_mappings = (
            self.model_config,
            self.patient_state_schema,
            self.factor_capacity_and_ordering,
            self.demographic_context_schema,
            self.typed_feature_schema_and_order,
            self.mask_policy_and_hash,
            self.normalization_and_whitening_hashes,
            self.eye_preprocessing_contract,
            self.tower_keys_and_state_hashes,
            self.fold_manifest_and_training_id_hash,
            self.paired_correction_gate_result,
            self.calibration_parameters,
            self.code_config_source_hashes,
        )
        if any(not isinstance(value, Mapping) or not value for value in required_mappings):
            raise ValueError("checkpoint metadata mappings must be nonempty")
        hashes = (
            self.preprocessor_bundle_sha256,
            self.initializer_hash,
            *self.normalization_and_whitening_hashes.values(),
            *self.tower_keys_and_state_hashes.values(),
            *self.code_config_source_hashes.values(),
        )
        if any(not _is_sha256(value) for value in hashes):
            raise ValueError("checkpoint metadata contains an invalid SHA-256 digest")
        if set(self.normalization_and_whitening_hashes) != {"age", "blood", "eye"}:
            raise ValueError("normalization hashes must contain exactly age, blood, and eye")
        if set(self.eye_preprocessing_contract) != {
            "weighting_policy",
            "fit_patient_count",
            "fit_image_count",
            "whitening_hash",
        }:
            raise ValueError("eye preprocessing contract is incomplete")
        if (
            not isinstance(self.eye_preprocessing_contract["weighting_policy"], str)
            or not self.eye_preprocessing_contract["weighting_policy"]
            or not isinstance(self.eye_preprocessing_contract["fit_patient_count"], int)
            or self.eye_preprocessing_contract["fit_patient_count"] < 2
            or not isinstance(self.eye_preprocessing_contract["fit_image_count"], int)
            or self.eye_preprocessing_contract["fit_image_count"]
            < self.eye_preprocessing_contract["fit_patient_count"]
            or self.eye_preprocessing_contract["whitening_hash"]
            != self.normalization_and_whitening_hashes["eye"]
        ):
            raise ValueError("eye preprocessing contract is inconsistent")
        required_context = {"schema_version", "schema_hash"}
        if not required_context <= set(self.demographic_context_schema):
            raise ValueError("demographic context provenance is incomplete")
        required_features = {
            "schema_version",
            "schema_hash",
            "ordered_feature_names",
            "feature_types",
            "ordered_features_hash",
        }
        if not required_features <= set(self.typed_feature_schema_and_order):
            raise ValueError("typed feature provenance is incomplete")
        required_policy = {"policy_eligible_mask", "policy_mask_hash"}
        if not required_policy <= set(self.mask_policy_and_hash):
            raise ValueError("mask policy provenance is incomplete")
        required_fold = {
            "outer_fold_id",
            "split_manifest_hash",
            "representation_fit_patient_id_hash",
            "training_config_hash",
            "split_seed",
            "training_seed",
        }
        if not required_fold <= set(self.fold_manifest_and_training_id_hash):
            raise ValueError("fold/training provenance is incomplete")
        explicit_hashes = (
            self.demographic_context_schema["schema_hash"],
            self.typed_feature_schema_and_order["schema_hash"],
            self.typed_feature_schema_and_order["ordered_features_hash"],
            self.mask_policy_and_hash["policy_mask_hash"],
            self.fold_manifest_and_training_id_hash["split_manifest_hash"],
            self.fold_manifest_and_training_id_hash[
                "representation_fit_patient_id_hash"
            ],
            self.fold_manifest_and_training_id_hash["training_config_hash"],
        )
        if any(not _is_sha256(value) for value in explicit_hashes):
            raise ValueError("checkpoint contract contains an invalid provenance hash")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.creation_time_utc, str) or not self.creation_time_utc.endswith("Z"):
            raise ValueError("creation_time_utc must be an explicit UTC timestamp ending in Z")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_type": self.model_type,
            "model_config": dict(self.model_config),
            "patient_state_schema": dict(self.patient_state_schema),
            "factor_capacity_and_ordering": dict(self.factor_capacity_and_ordering),
            "demographic_context_schema": dict(self.demographic_context_schema),
            "typed_feature_schema_and_order": dict(self.typed_feature_schema_and_order),
            "mask_policy_and_hash": dict(self.mask_policy_and_hash),
            "normalization_and_whitening_hashes": dict(
                sorted(self.normalization_and_whitening_hashes.items())
            ),
            "eye_preprocessing_contract": dict(self.eye_preprocessing_contract),
            "preprocessor_bundle_sha256": self.preprocessor_bundle_sha256,
            "tower_keys_and_state_hashes": dict(
                sorted(self.tower_keys_and_state_hashes.items())
            ),
            "fold_manifest_and_training_id_hash": dict(
                self.fold_manifest_and_training_id_hash
            ),
            "initializer_hash": self.initializer_hash,
            "paired_correction_gate_result": dict(self.paired_correction_gate_result),
            "calibration_parameters": dict(self.calibration_parameters),
            "code_config_source_hashes": dict(
                sorted(self.code_config_source_hashes.items())
            ),
            "seed": self.seed,
            "creation_time_utc": self.creation_time_utc,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AtlasCheckpointMetadata":
        _require_exact_keys(value, _METADATA_FIELDS, "checkpoint metadata")
        return cls(**dict(value))


def build_checkpoint_metadata(
    *,
    model_config: Mapping[str, Any],
    patient_state_schema: Mapping[str, Any],
    factor_capacity_and_ordering: Mapping[str, Any],
    preprocessor: FoldPreprocessor,
    tower_keys_and_state_hashes: Mapping[str, str],
    initializer_hash: str,
    paired_correction_gate_result: Mapping[str, Any],
    calibration_parameters: Mapping[str, Any],
    code_config_source_hashes: Mapping[str, str],
    seed: int,
    creation_time_utc: str,
) -> AtlasCheckpointMetadata:
    """Build metadata by deriving every preprocessing/schema field in one place."""

    schemas = preprocessor.schemas
    provenance = preprocessor.provenance
    return AtlasCheckpointMetadata(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        model_type=CHECKPOINT_MODEL_TYPE,
        model_config=dict(model_config),
        patient_state_schema=dict(patient_state_schema),
        factor_capacity_and_ordering=dict(factor_capacity_and_ordering),
        demographic_context_schema={
            "schema_version": schemas.context_schema_version,
            "schema_hash": schemas.context_schema_hash,
        },
        typed_feature_schema_and_order={
            "schema_version": schemas.feature_schema_version,
            "schema_hash": schemas.feature_schema_hash,
            "ordered_feature_names": list(schemas.ordered_feature_names),
            "feature_types": list(schemas.feature_types),
            "ordered_features_hash": schemas.ordered_features_hash,
        },
        mask_policy_and_hash={
            "policy_eligible_mask": list(preprocessor.policy_eligible_mask),
            "policy_mask_hash": preprocessor.policy_mask_hash,
        },
        normalization_and_whitening_hashes=dict(preprocessor.normalization_hashes),
        eye_preprocessing_contract={
            "weighting_policy": preprocessor.eye_weighting_policy,
            "fit_patient_count": preprocessor.eye_fit_patient_count,
            "fit_image_count": preprocessor.eye_fit_image_count,
            "whitening_hash": preprocessor.normalization_hashes["eye"],
        },
        preprocessor_bundle_sha256=preprocessor.bundle_sha256,
        tower_keys_and_state_hashes=dict(tower_keys_and_state_hashes),
        fold_manifest_and_training_id_hash={
            "outer_fold_id": provenance.outer_fold_id,
            "split_manifest_hash": provenance.split_manifest_hash,
            "outer_train_patient_id_hash": provenance.outer_train_patient_id_hash,
            "representation_fit_patient_id_hash": (
                provenance.representation_fit_patient_id_hash
            ),
            "validation_patient_id_hash": provenance.validation_patient_id_hash,
            "calibration_patient_id_hash": provenance.calibration_patient_id_hash,
            "outer_test_patient_id_hash": provenance.outer_test_patient_id_hash,
            "role_counts": dict(provenance.role_counts),
            "training_config_hash": provenance.training_config_hash,
            "split_seed": provenance.split_seed,
            "training_seed": provenance.training_seed,
            "source_hashes": dict(provenance.source_hashes),
        },
        initializer_hash=initializer_hash,
        paired_correction_gate_result=dict(paired_correction_gate_result),
        calibration_parameters=dict(calibration_parameters),
        code_config_source_hashes=dict(code_config_source_hashes),
        seed=seed,
        creation_time_utc=creation_time_utc,
    )


@dataclass(frozen=True)
class CheckpointExpectations:
    """Out-of-band digests that must match before loading any tensor."""

    metadata_sha256: str
    state_sha256: str
    preprocessor_bundle_sha256: str
    model_config_sha256: str
    factor_ordering_sha256: str
    context_schema_sha256: str
    typed_feature_order_sha256: str
    policy_sha256: str
    fold_manifest_sha256: str
    training_id_sha256: str
    tower_provenance_sha256: str
    calibration_sha256: str
    eye_preprocessing_contract_sha256: str

    def __post_init__(self) -> None:
        if any(not _is_sha256(value) for value in self.__dict__.values()):
            raise ValueError("all checkpoint expectations must be SHA-256 digests")

    @classmethod
    def from_metadata(
        cls, metadata: AtlasCheckpointMetadata, *, state_sha256: str
    ) -> "CheckpointExpectations":
        value = metadata.to_dict()
        return cls(
            metadata_sha256=hash_json(value),
            state_sha256=state_sha256,
            preprocessor_bundle_sha256=metadata.preprocessor_bundle_sha256,
            model_config_sha256=hash_json(value["model_config"]),
            factor_ordering_sha256=hash_json(
                value["factor_capacity_and_ordering"]
            ),
            context_schema_sha256=hash_json(value["demographic_context_schema"]),
            typed_feature_order_sha256=hash_json(
                value["typed_feature_schema_and_order"]
            ),
            policy_sha256=hash_json(value["mask_policy_and_hash"]),
            fold_manifest_sha256=value["fold_manifest_and_training_id_hash"][
                "split_manifest_hash"
            ],
            training_id_sha256=value["fold_manifest_and_training_id_hash"][
                "representation_fit_patient_id_hash"
            ],
            tower_provenance_sha256=hash_json(value["tower_keys_and_state_hashes"]),
            calibration_sha256=hash_json(value["calibration_parameters"]),
            eye_preprocessing_contract_sha256=hash_json(
                value["eye_preprocessing_contract"]
            ),
        )


@dataclass(frozen=True)
class SavedCheckpoint:
    file_sha256: str
    state_sha256: str
    metadata_sha256: str


@dataclass(frozen=True)
class LoadedCheckpoint:
    metadata: AtlasCheckpointMetadata
    file_sha256: str
    state_sha256: str


def _model_config_dict(model: nn.Module) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if is_dataclass(config) and not isinstance(config, type):
        value = asdict(config)
    elif isinstance(config, Mapping):
        value = dict(config)
    else:
        raise TypeError(
            "checkpointed models must expose a dataclass or mapping in model.config"
        )
    if not value:
        raise ValueError("model.config must not be empty")
    _validate_json_tree(value, path=("model", "config"))
    return value


def _validate_model_and_preprocessor_contract(
    model: nn.Module,
    preprocessor: FoldPreprocessor,
    metadata: AtlasCheckpointMetadata,
) -> None:
    if not isinstance(preprocessor, FoldPreprocessor):
        raise TypeError("an explicit validated FoldPreprocessor is required")
    if hash_json(_model_config_dict(model)) != hash_json(metadata.model_config):
        raise ValueError("target model config does not match checkpoint metadata")
    if preprocessor.bundle_sha256 != metadata.preprocessor_bundle_sha256:
        raise ValueError("preprocessor bundle does not match checkpoint metadata")
    if dict(preprocessor.normalization_hashes) != dict(
        metadata.normalization_and_whitening_hashes
    ):
        raise ValueError("normalization/whitening hashes do not match checkpoint metadata")
    expected_eye_contract = {
        "weighting_policy": preprocessor.eye_weighting_policy,
        "fit_patient_count": preprocessor.eye_fit_patient_count,
        "fit_image_count": preprocessor.eye_fit_image_count,
        "whitening_hash": preprocessor.normalization_hashes["eye"],
    }
    if dict(metadata.eye_preprocessing_contract) != expected_eye_contract:
        raise ValueError("eye preprocessing contract does not match the checkpoint preprocessor")
    schemas = preprocessor.schemas
    if metadata.demographic_context_schema.get("schema_hash") != schemas.context_schema_hash:
        raise ValueError("context schema does not match the checkpoint preprocessor")
    if metadata.typed_feature_schema_and_order.get("schema_hash") != schemas.feature_schema_hash:
        raise ValueError("feature schema does not match the checkpoint preprocessor")
    if (
        metadata.typed_feature_schema_and_order.get("ordered_features_hash")
        != schemas.ordered_features_hash
    ):
        raise ValueError("ordered feature schema does not match the checkpoint preprocessor")
    if metadata.mask_policy_and_hash.get("policy_mask_hash") != preprocessor.policy_mask_hash:
        raise ValueError("policy mask does not match the checkpoint preprocessor")
    provenance = preprocessor.provenance
    fold = metadata.fold_manifest_and_training_id_hash
    if fold.get("split_manifest_hash") != provenance.split_manifest_hash:
        raise ValueError("fold manifest does not match the checkpoint preprocessor")
    if (
        fold.get("representation_fit_patient_id_hash")
        != provenance.representation_fit_patient_id_hash
    ):
        raise ValueError("training membership does not match the checkpoint preprocessor")


def _tensor_npy_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    if str(tensor.dtype) not in _DTYPE_BY_NAME:
        raise TypeError(f"unsupported checkpoint tensor dtype: {tensor.dtype}")
    array = tensor.numpy()
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _fixed_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info


def save_atlas_checkpoint(
    path: str | Path,
    model: nn.Module,
    metadata: AtlasCheckpointMetadata,
    *,
    preprocessor: FoldPreprocessor,
) -> SavedCheckpoint:
    """Serialize a deterministic, non-pickled checkpoint without overwrite."""

    _validate_model_and_preprocessor_contract(model, preprocessor, metadata)
    state = model.state_dict()
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("model state_dict must be a nonempty tensor mapping")
    state_sha256 = hash_state_dict(state)
    tensor_payloads: dict[str, bytes] = {}
    entries: list[dict[str, Any]] = []
    for index, name in enumerate(sorted(state)):
        tensor = state[name].detach().cpu().contiguous()
        member = f"tensors/{index:05d}.npy"
        raw = _tensor_npy_bytes(tensor)
        tensor_payloads[member] = raw
        entries.append(
            {
                "name": name,
                "member": member,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "npy_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    metadata_dict = metadata.to_dict()
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": CHECKPOINT_MODEL_TYPE,
        "metadata": metadata_dict,
        "metadata_sha256": hash_json(metadata_dict),
        "state_sha256": state_sha256,
        "tensor_entries": entries,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    if len(entries) + 1 > MAX_CHECKPOINT_MEMBERS:
        raise ValueError("checkpoint exceeds the ZIP member-count cap")
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ValueError("checkpoint manifest exceeds the size cap")
    if len(manifest_bytes) + sum(map(len, tensor_payloads.values())) > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise ValueError("checkpoint exceeds the aggregate uncompressed-byte cap")
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w") as archive:
        archive.writestr(_fixed_zip_info(_MANIFEST_MEMBER), manifest_bytes)
        for member in sorted(tensor_payloads):
            archive.writestr(_fixed_zip_info(member), tensor_payloads[member])
    raw_checkpoint = output.getvalue()
    if len(raw_checkpoint) > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint exceeds the maximum permitted size")
    with Path(path).open("xb") as handle:
        handle.write(raw_checkpoint)
    return SavedCheckpoint(
        file_sha256=hashlib.sha256(raw_checkpoint).hexdigest(),
        state_sha256=state_sha256,
        metadata_sha256=manifest["metadata_sha256"],
    )


def _validate_expectations(
    metadata: AtlasCheckpointMetadata,
    state_sha256: str,
    expectations: CheckpointExpectations,
) -> None:
    observed = CheckpointExpectations.from_metadata(
        metadata, state_sha256=state_sha256
    )
    for name in expectations.__dict__:
        if getattr(observed, name) != getattr(expectations, name):
            raise ValueError(f"checkpoint expectation mismatch for {name}")


def load_atlas_checkpoint(
    path: str | Path,
    model: nn.Module,
    expectations: CheckpointExpectations,
    *,
    preprocessor: FoldPreprocessor,
) -> LoadedCheckpoint:
    """Authenticate one byte snapshot, then strictly load it into ``model``."""

    if not isinstance(expectations, CheckpointExpectations):
        raise TypeError("explicit CheckpointExpectations are required")
    if not isinstance(preprocessor, FoldPreprocessor):
        raise TypeError("an explicit validated FoldPreprocessor is required")
    checkpoint_bytes = Path(path).read_bytes()
    if not checkpoint_bytes or len(checkpoint_bytes) > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint size is invalid")
    file_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    try:
        archive = zipfile.ZipFile(io.BytesIO(checkpoint_bytes), mode="r")
    except zipfile.BadZipFile as error:
        raise ValueError("checkpoint is not a valid ZIP container") from error
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_CHECKPOINT_MEMBERS:
            raise ValueError("checkpoint exceeds the ZIP member-count cap")
        if sum(info.file_size for info in infos) > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError("checkpoint exceeds the aggregate uncompressed-byte cap")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("checkpoint contains duplicate ZIP members")
        if _MANIFEST_MEMBER not in names:
            raise ValueError("checkpoint manifest is missing")
        manifest_info = archive.getinfo(_MANIFEST_MEMBER)
        if manifest_info.file_size > MAX_MANIFEST_BYTES:
            raise ValueError("checkpoint manifest exceeds the size cap")
        if any(
            name.startswith("/") or ".." in Path(name).parts or name.endswith("/")
            for name in names
        ):
            raise ValueError("checkpoint contains an unsafe ZIP member name")
        if any(info.file_size > MAX_CHECKPOINT_BYTES for info in infos):
            raise ValueError("checkpoint member exceeds the maximum size")
        manifest = _strict_json_loads(archive.read(_MANIFEST_MEMBER))
        if not isinstance(manifest, Mapping):
            raise TypeError("checkpoint manifest must be a JSON object")
        _require_exact_keys(manifest, _MANIFEST_FIELDS, "checkpoint manifest")
        if manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("checkpoint schema version mismatch")
        if manifest["model_type"] != CHECKPOINT_MODEL_TYPE:
            raise ValueError("checkpoint model type mismatch")
        metadata_value = manifest["metadata"]
        if not isinstance(metadata_value, Mapping):
            raise TypeError("checkpoint metadata must be a JSON object")
        if hash_json(metadata_value) != manifest["metadata_sha256"]:
            raise ValueError("checkpoint metadata hash mismatch")
        metadata = AtlasCheckpointMetadata.from_dict(metadata_value)
        if not _is_sha256(manifest["state_sha256"]):
            raise ValueError("checkpoint state hash is malformed")
        _validate_expectations(metadata, manifest["state_sha256"], expectations)
        _validate_model_and_preprocessor_contract(model, preprocessor, metadata)

        entries = manifest["tensor_entries"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("checkpoint tensor manifest is empty")
        model_state = model.state_dict()
        if len(entries) != len(model_state):
            raise ValueError("checkpoint tensor-entry count does not match the target model")
        tensor_names: set[str] = set()
        tensor_members: set[str] = set()
        validated_entries: list[Mapping[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise TypeError("checkpoint tensor entry must be a mapping")
            _require_exact_keys(entry, _TENSOR_ENTRY_FIELDS, "tensor entry")
            name = entry["name"]
            member = entry["member"]
            if not isinstance(name, str) or not name or name in tensor_names:
                raise ValueError("checkpoint tensor names are invalid or duplicated")
            if (
                not isinstance(member, str)
                or not member.startswith("tensors/")
                or member in tensor_members
            ):
                raise ValueError("checkpoint tensor members are invalid or duplicated")
            tensor_names.add(name)
            tensor_members.add(member)
            if member not in names:
                raise ValueError(f"checkpoint tensor member is missing: {member}")
            if not _is_sha256(entry["npy_sha256"]):
                raise ValueError(f"checkpoint tensor hash is malformed: {name}")
            if entry["dtype"] not in _DTYPE_BY_NAME:
                raise TypeError(f"unsupported checkpoint tensor dtype: {entry['dtype']}")
            shape = entry["shape"]
            if not isinstance(shape, list) or any(
                not isinstance(dimension, int) or dimension < 0 for dimension in shape
            ):
                raise ValueError(f"checkpoint tensor shape declaration is malformed: {name}")
            validated_entries.append(entry)

        # Complete all structural checks against the target model before
        # archive.read() is called for even one tensor payload.
        if tensor_names != set(model_state):
            raise ValueError("checkpoint tensor-entry names do not match the target model")
        expected_members = {_MANIFEST_MEMBER, *tensor_members}
        if set(names) != expected_members:
            raise ValueError("checkpoint contains undeclared ZIP members")

        state: dict[str, torch.Tensor] = {}
        for entry in validated_entries:
            name = entry["name"]
            member = entry["member"]
            raw = archive.read(member)
            if hashlib.sha256(raw).hexdigest() != entry["npy_sha256"]:
                raise ValueError(f"checkpoint tensor hash mismatch: {name}")
            dtype = _DTYPE_BY_NAME.get(entry["dtype"])
            if dtype is None:
                raise TypeError(f"unsupported checkpoint tensor dtype: {entry['dtype']}")
            try:
                array = np.load(io.BytesIO(raw), allow_pickle=False)
            except (ValueError, OSError) as error:
                raise ValueError(f"invalid non-pickled tensor payload: {name}") from error
            if array.dtype.hasobject:
                raise TypeError("object arrays are forbidden in checkpoint tensors")
            if list(array.shape) != entry["shape"]:
                raise ValueError(f"checkpoint tensor shape declaration mismatch: {name}")
            tensor = torch.from_numpy(np.array(array, copy=True))
            if tensor.dtype != dtype:
                raise TypeError(f"checkpoint tensor dtype declaration mismatch: {name}")
            state[name] = tensor

        if hash_state_dict(state) != manifest["state_sha256"]:
            raise ValueError("checkpoint canonical state hash mismatch")
        for name, target in model_state.items():
            source = state[name]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f"checkpoint state contract mismatch for tensor: {name}")
        model.load_state_dict(state, strict=True)
    return LoadedCheckpoint(
        metadata=metadata,
        file_sha256=file_sha256,
        state_sha256=manifest["state_sha256"],
    )


__all__ = [
    "AtlasCheckpointMetadata",
    "CheckpointExpectations",
    "LoadedCheckpoint",
    "MAX_CHECKPOINT_MEMBERS",
    "MAX_MANIFEST_BYTES",
    "MAX_TOTAL_UNCOMPRESSED_BYTES",
    "SavedCheckpoint",
    "build_checkpoint_metadata",
    "load_atlas_checkpoint",
    "save_atlas_checkpoint",
]
