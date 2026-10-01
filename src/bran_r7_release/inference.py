"""Bounded checkpoint loading and local BRAN R7 inference helpers.

Inputs to :func:`infer` are already standardized in the caller's bound fold
coordinate frame. No source, field-map, or transform fitting is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any
import zipfile

import torch
from torch import Tensor

from bran_multisource_age_v2 import validate_normalized_age
from bran_multisource_model_v2 import erase_cbc_for_completion
from bran_robust_clinical_r7 import BRANRobustClinicalR7


CHECKPOINT_SCHEMA = "bran-r7-source-checkpoint-v1"
FIELD_MAP_SCHEMA = "bran-r7-field-map-binding-v1"
TRANSFORM_SCHEMA = "bran-r7-fold-transform-binding-v1"
BINDING_SCHEMA = "bran-r7-checkpoint-binding-v1"
ERROR = "bran_r7_inference_contract_failed"
# These caps are intentionally well above the retained architecture's state
# (56 tensors and fewer than 0.5 million scalar parameters for the MLP arm).
# They bound metadata and storage before restricted deserialization begins.
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 256
MAX_STATE_TENSORS = 128
MAX_STATE_ELEMENTS = 5_000_000
MAX_STATE_KEY_BYTES = 256
_TRANSFORM_ARRAYS = (
    ("clinical_median", (59,), "float"),
    ("clinical_iqr", (59,), "float"),
    ("retinal_mean", (384,), "float"),
    ("retinal_scale", (384,), "float"),
    ("eligible", (59,), "bool"),
)


def _fail() -> None:
    raise ValueError(ERROR) from None


def _canonical_hash(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        _fail()
    return hashlib.sha256(encoded).hexdigest()


def _hex_digest(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _indices(values: Any, *, count: int, upper: int) -> tuple[int, ...]:
    if type(values) not in (tuple, list):
        _fail()
    try:
        result = tuple(values)
        unique_count = len(set(result))
    except Exception:
        _fail()
    if (len(result) != count or unique_count != count
            or any(type(item) is not int or not 0 <= item < upper for item in result)):
        _fail()
    return result


def make_field_map_binding(feature_names: tuple[str, ...] | list[str],
                           eligible_indices: tuple[int, ...] | list[int],
                           cbc_indices: tuple[int, ...] | list[int]) -> dict[str, Any]:
    """Build a digest binding without retaining or exporting feature names."""
    if (type(feature_names) not in (tuple, list) or len(feature_names) != 59
            or any(type(name) is not str or not name.strip() for name in feature_names)
            or len(set(feature_names)) != 59):
        _fail()
    eligible = _indices(eligible_indices, count=43, upper=48)
    cbc = _indices(cbc_indices, count=9, upper=48)
    if not set(cbc).issubset(eligible):
        _fail()
    payload = {
        "feature_names": list(feature_names),
        "eligible_indices": list(eligible),
        "cbc_indices": list(cbc),
    }
    return {
        "schema": FIELD_MAP_SCHEMA,
        "sha256": _canonical_hash(payload),
        "eligible_indices": list(eligible),
        "cbc_indices": list(cbc),
    }


def _validate_field_map(binding: Any, config: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {"schema", "sha256", "eligible_indices", "cbc_indices"}
    if type(binding) is not dict or set(binding) != expected_keys or binding.get("schema") != FIELD_MAP_SCHEMA:
        _fail()
    if not _hex_digest(binding.get("sha256")):
        _fail()
    eligible = _indices(binding.get("eligible_indices"), count=43, upper=48)
    cbc = _indices(binding.get("cbc_indices"), count=9, upper=48)
    if (not set(cbc).issubset(eligible)
            or config.get("eligible_indices") != list(eligible)
            or config.get("cbc_indices") != list(cbc)):
        _fail()
    return {
        "schema": FIELD_MAP_SCHEMA,
        "sha256": binding["sha256"],
        "eligible_indices": list(eligible),
        "cbc_indices": list(cbc),
    }


def _array_parts(value: Any, shape: tuple[int, ...], kind: str) -> tuple[str, tuple[int, ...], bytes, list[Any]]:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        dtype_name = str(tensor.dtype).removeprefix("torch.")
        actual_shape = tuple(tensor.shape)
        if kind == "float" and not tensor.is_floating_point():
            _fail()
        if kind == "bool" and tensor.dtype != torch.bool:
            _fail()
        items = tensor.reshape(-1).tolist()
        raw_tensor = tensor.view(torch.uint8).reshape(-1)
        chunks = [bytes(chunk.tolist()) for chunk in raw_tensor.split(1_000_000)]
        raw = b"".join(chunks)
    else:
        try:
            dtype_name = str(value.dtype)
            actual_shape = tuple(value.shape)
            dtype_kind = value.dtype.kind
            items = value.reshape(-1).tolist()
            raw = value.tobytes()
        except Exception:
            _fail()
        if kind == "float" and dtype_kind != "f":
            _fail()
        if kind == "bool" and dtype_kind != "b":
            _fail()
    if actual_shape != shape:
        _fail()
    if kind == "float" and any(not math.isfinite(float(item)) for item in items):
        _fail()
    return dtype_name, actual_shape, raw, items


def make_transform_binding(transform: Any) -> dict[str, Any]:
    """Fingerprint a supplied FoldTransformV2-compatible object without fitting it."""
    try:
        heldout = transform.heldout_fold
        fold_hash = transform.fold_identity_sha256
        train_hash = transform.training_indices_sha256
        age_mean = transform.age_mean
        age_scale = transform.age_scale
    except Exception:
        _fail()
    if (type(heldout) is not int or heldout not in range(5)
            or not _hex_digest(fold_hash) or not _hex_digest(train_hash)
            or type(age_mean) not in (int, float) or type(age_scale) not in (int, float)
            or not math.isfinite(float(age_mean)) or not math.isfinite(float(age_scale))
            or float(age_scale) <= 0):
        _fail()

    eligible_items: list[Any] | None = None
    records: dict[str, tuple[str, tuple[int, ...], bytes, list[Any]]] = {}
    for name, shape, kind in _TRANSFORM_ARRAYS:
        try:
            value = getattr(transform, name)
        except Exception:
            _fail()
        dtype_name, actual_shape, raw, items = _array_parts(value, shape, kind)
        if name in ("clinical_iqr", "retinal_scale") and any(float(item) <= 0 for item in items):
            _fail()
        if name == "eligible":
            eligible_items = items
        records[name] = (dtype_name, actual_shape, raw, items)

    # Match the existing V3 transform fingerprint field order and byte
    # representation so callers can compare the same frozen transform object.
    digest = hashlib.sha256()
    for name in ("clinical_median", "clinical_iqr", "retinal_mean", "retinal_scale"):
        dtype_name, actual_shape, raw, _ = records[name]
        digest.update(name.encode("utf-8"))
        digest.update(dtype_name.encode("utf-8"))
        digest.update(str(actual_shape).encode("utf-8"))
        digest.update(raw)
    for name, value in (("age_mean", age_mean), ("age_scale", age_scale)):
        digest.update(name.encode("utf-8"))
        digest.update(repr(value).encode("utf-8"))
    dtype_name, actual_shape, raw, _ = records["eligible"]
    digest.update(b"eligible")
    digest.update(dtype_name.encode("utf-8"))
    digest.update(str(actual_shape).encode("utf-8"))
    digest.update(raw)
    for name, value in (("heldout_fold", heldout), ("fold_identity_sha256", fold_hash),
                        ("training_indices_sha256", train_hash)):
        digest.update(name.encode("utf-8"))
        digest.update(repr(value).encode("utf-8"))
    if eligible_items is None:
        _fail()
    eligible_indices = [index for index, value in enumerate(eligible_items) if bool(value)]
    if len(eligible_indices) != 43 or any(index >= 48 for index in eligible_indices):
        _fail()
    return {
        "schema": TRANSFORM_SCHEMA,
        "sha256": digest.hexdigest(),
        "heldout_fold": heldout,
        "fold_identity_sha256": fold_hash,
        "training_indices_sha256": train_hash,
        "eligible_indices": eligible_indices,
    }


def _config_hash(config: dict[str, Any]) -> str:
    return _canonical_hash(config)


def _valid_binding(binding: Any) -> bool:
    return (type(binding) is dict
            and set(binding) == {
                "schema", "config_sha256", "state_dict_sha256",
                "field_map_sha256", "transform_sha256",
            }
            and binding.get("schema") == BINDING_SCHEMA
            and all(_hex_digest(binding.get(key)) for key in (
                "config_sha256", "state_dict_sha256", "field_map_sha256", "transform_sha256")))


def _state_hash(state_dict: Any) -> str:
    if (not isinstance(state_dict, dict) or not state_dict
            or len(state_dict) > MAX_STATE_TENSORS
            or any(type(name) is not str or len(name.encode("utf-8")) > MAX_STATE_KEY_BYTES
                   for name in state_dict)):
        _fail()
    total_elements = 0
    for value in state_dict.values():
        if not isinstance(value, Tensor) or value.layout != torch.strided:
            _fail()
        total_elements += value.numel()
        if total_elements > MAX_STATE_ELEMENTS:
            _fail()
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        tensor = value.detach().cpu().contiguous()
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
            _fail()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        raw_tensor = tensor.view(torch.uint8).reshape(-1)
        for chunk in raw_tensor.split(1_000_000):
            digest.update(bytes(chunk.tolist()))
    return digest.hexdigest()


def _check_checkpoint_archive(stream: Any) -> None:
    """Check PyTorch's public ZIP serialization metadata without extracting it."""
    try:
        stream.seek(0)
        with zipfile.ZipFile(stream, "r") as archive:
            members = archive.infolist()
            if not members or len(members) > MAX_ARCHIVE_MEMBERS:
                _fail()
            names: set[str] = set()
            total_size = 0
            for member in members:
                if (type(member.filename) is not str
                        or len(member.filename.encode("utf-8")) > 512
                        or member.filename in names
                        or member.file_size < 0 or member.compress_size < 0
                        or member.file_size > MAX_ARCHIVE_MEMBER_BYTES
                        or member.flag_bits & 0x1):
                    _fail()
                names.add(member.filename)
                total_size += member.file_size
                if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    _fail()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail()


def _checkpoint_binding(config: dict[str, Any], state_dict: dict[str, Tensor],
                        field_map: dict[str, Any], transform: dict[str, Any]) -> dict[str, str]:
    return {
        "schema": BINDING_SCHEMA,
        "config_sha256": _config_hash(config),
        "state_dict_sha256": _state_hash(state_dict),
        "field_map_sha256": field_map["sha256"],
        "transform_sha256": transform["sha256"],
    }


@dataclass(frozen=True, repr=False)
class LoadedR7Checkpoint:
    """A validated model and the external bindings used to load it."""

    model: BRANRobustClinicalR7
    binding: dict[str, str]
    field_map_binding: dict[str, Any]
    transform_binding: dict[str, Any]
    input_transform: Any | None


@dataclass(frozen=True, repr=False)
class R7InferenceResult:
    """Detached outputs. Abstained screening and CBC estimates are NaN."""

    state_mean: Tensor
    state_logvar: Tensor
    abstained: Tensor
    screening_probability: Tensor
    cbc_standardized: Tensor
    cbc_original_units: Tensor | None
    completion_target_mask: Tensor | None


def _checkpoint_payload(model: BRANRobustClinicalR7, field_map_binding: dict[str, Any],
                        input_transform: Any) -> dict[str, Any]:
    if type(model) is not BRANRobustClinicalR7:
        _fail()
    config = model.export_config()
    field_map = _validate_field_map(field_map_binding, config)
    transform = make_transform_binding(input_transform)
    if sorted(field_map["eligible_indices"]) != transform["eligible_indices"]:
        _fail()
    state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    binding = _checkpoint_binding(config, state_dict, field_map, transform)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "config": config,
        "state_dict": state_dict,
        "field_map_binding": field_map,
        "transform_binding": transform,
        "binding": binding,
    }


def _validate_loaded(loaded: LoadedR7Checkpoint) -> None:
    model = loaded.model
    binding = loaded.binding
    if (type(model) is not BRANRobustClinicalR7 or type(binding) is not dict
            or set(binding) != {"schema", "config_sha256", "state_dict_sha256", "field_map_sha256", "transform_sha256"}
            or binding.get("schema") != BINDING_SCHEMA
            or any(not _hex_digest(binding.get(key)) for key in
                   ("config_sha256", "state_dict_sha256", "field_map_sha256", "transform_sha256"))):
        _fail()
    config = model.export_config()
    field_map = _validate_field_map(loaded.field_map_binding, config)
    transform = loaded.transform_binding
    if (type(transform) is not dict or set(transform) != {
            "schema", "sha256", "heldout_fold", "fold_identity_sha256",
            "training_indices_sha256", "eligible_indices"}
            or transform.get("schema") != TRANSFORM_SCHEMA
            or not _hex_digest(transform.get("sha256"))
            or type(transform.get("heldout_fold")) is not int
            or transform["heldout_fold"] not in range(5)
            or not _hex_digest(transform.get("fold_identity_sha256"))
            or not _hex_digest(transform.get("training_indices_sha256"))
            or transform.get("eligible_indices") != sorted(field_map["eligible_indices"])):
        _fail()
    if (binding["config_sha256"] != _config_hash(config)
            or binding["field_map_sha256"] != field_map["sha256"]
            or binding["transform_sha256"] != transform["sha256"]
            or binding["state_dict_sha256"] != _state_hash(model.state_dict())):
        _fail()
    if loaded.input_transform is not None and make_transform_binding(loaded.input_transform) != transform:
        _fail()


def save_checkpoint(path: str | os.PathLike[str], model: BRANRobustClinicalR7,
                    *, field_map_binding: dict[str, Any], input_transform: Any) -> dict[str, str]:
    """Save a tensor-only checkpoint without overwriting an existing path."""
    created = False
    try:
        target = Path(path)
        if target.is_symlink() or target.exists() or not target.parent.is_dir():
            _fail()
        payload = _checkpoint_payload(model, field_map_binding, input_transform)
        with target.open("xb") as stream:
            created = True
            os.chmod(target, 0o600)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        return dict(payload["binding"])
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        try:
            if created and "target" in locals() and target.is_file() and not target.is_symlink():
                target.unlink()
        except Exception:
            pass
        _fail()


def load_checkpoint(path: str | os.PathLike[str], *, expected_binding: dict[str, str],
                    expected_field_map_binding: dict[str, Any],
                    expected_transform_binding: dict[str, Any],
                    input_transform: Any | None = None,
                    device: str | torch.device = "cpu") -> LoadedR7Checkpoint:
    """Load with PyTorch's restricted unpickler and require caller-pinned bindings."""
    try:
        source = Path(path)
        if source.is_symlink() or type(expected_binding) is not dict or not _valid_binding(expected_binding):
            _fail()
        if type(expected_transform_binding) is not dict or set(expected_transform_binding) != {
                "schema", "sha256", "heldout_fold", "fold_identity_sha256",
                "training_indices_sha256", "eligible_indices"}:
            _fail()
        if (expected_transform_binding.get("schema") != TRANSFORM_SCHEMA
                or not _hex_digest(expected_transform_binding.get("sha256"))
                or type(expected_transform_binding.get("heldout_fold")) is not int
                or expected_transform_binding["heldout_fold"] not in range(5)
                or not _hex_digest(expected_transform_binding.get("fold_identity_sha256"))
                or not _hex_digest(expected_transform_binding.get("training_indices_sha256"))
                or type(expected_transform_binding.get("eligible_indices")) is not list
                or expected_transform_binding.get("eligible_indices") != sorted(set(expected_transform_binding["eligible_indices"]))
                or len(expected_transform_binding["eligible_indices"]) != 43
                or any(type(index) is not int or not 0 <= index < 48
                       for index in expected_transform_binding["eligible_indices"])):
            _fail()
        if input_transform is not None and make_transform_binding(input_transform) != expected_transform_binding:
            _fail()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(source, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
                    or metadata.st_size <= 0 or metadata.st_size > MAX_CHECKPOINT_BYTES):
                _fail()
            _check_checkpoint_archive(stream)
            stream.seek(0)
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        if (type(payload) is not dict or set(payload) != {
                "schema", "config", "state_dict", "field_map_binding", "transform_binding", "binding"}
                or payload.get("schema") != CHECKPOINT_SCHEMA
                or type(payload.get("config")) is not dict
                or type(payload.get("state_dict")) is not dict):
            _fail()
        model = BRANRobustClinicalR7.from_config(payload["config"])
        field_map = _validate_field_map(payload["field_map_binding"], payload["config"])
        expected_field_map = _validate_field_map(expected_field_map_binding, payload["config"])
        if field_map != expected_field_map:
            _fail()
        if expected_transform_binding["eligible_indices"] != sorted(field_map["eligible_indices"]):
            _fail()
        saved_transform = payload["transform_binding"]
        if type(saved_transform) is not dict or set(saved_transform) != set(expected_transform_binding):
            _fail()
        if saved_transform != expected_transform_binding:
            _fail()
        binding = _checkpoint_binding(payload["config"], payload["state_dict"], field_map, saved_transform)
        if binding != payload.get("binding") or binding != expected_binding:
            _fail()
        expected_state = model.state_dict()
        if (set(payload["state_dict"]) != set(expected_state)
                or any(not isinstance(payload["state_dict"][key], Tensor)
                       or payload["state_dict"][key].shape != expected_state[key].shape
                       or payload["state_dict"][key].dtype != expected_state[key].dtype
                       for key in expected_state)):
            _fail()
        model.load_state_dict(payload["state_dict"], strict=True)
        parsed_device = torch.device(device)
        model.to(parsed_device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return LoadedR7Checkpoint(
            model=model,
            binding=dict(binding),
            field_map_binding=dict(field_map),
            transform_binding=dict(saved_transform),
            input_transform=input_transform,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail()


def _transform_vector(transform: Any, name: str, indices: tuple[int, ...], *, device: torch.device,
                      dtype: torch.dtype) -> Tensor:
    try:
        values = getattr(transform, name)
        if isinstance(values, Tensor):
            result = values.detach().to(device=device, dtype=dtype)
        else:
            result = torch.tensor(values.tolist(), device=device, dtype=dtype)
        if result.shape != (59,) or not bool(torch.isfinite(result).all()):
            _fail()
        return result[list(indices)]
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail()


@torch.no_grad()
def infer(loaded: LoadedR7Checkpoint, clinical_standardized: Tensor,
          clinical_observed: Tensor, retinal_standardized: Tensor,
          retinal_visible: Tensor, normalized_age: Tensor, *,
          expected_binding: dict[str, str],
          clinical_eligible: Tensor | None = None,
          erase_cbc_targets: bool = False) -> R7InferenceResult:
    """Infer native heads from already fold-standardized tensors.

    Set ``erase_cbc_targets`` for target-completion inference. The returned
    original-unit CBC values are absent unless the same bound transform object
    was supplied during checkpoint loading.
    """
    if (type(loaded) is not LoadedR7Checkpoint or type(erase_cbc_targets) is not bool
            or type(expected_binding) is not dict or not _valid_binding(expected_binding)):
        _fail()
    model = loaded.model
    if loaded.binding != expected_binding:
        _fail()
    _validate_loaded(loaded)
    if (type(model) is not BRANRobustClinicalR7 or model.training
            or any(parameter.requires_grad for parameter in model.parameters())):
        _fail()
    try:
        parameter = next(model.parameters())
        device, dtype = parameter.device, parameter.dtype
    except Exception:
        _fail()
    if (not isinstance(clinical_standardized, Tensor) or not clinical_standardized.is_floating_point()
            or clinical_standardized.ndim != 2 or clinical_standardized.shape[1] != 59
            or clinical_standardized.shape[0] == 0
            or clinical_standardized.device != device or clinical_standardized.dtype != dtype):
        _fail()
    batch = clinical_standardized.shape[0]
    if (not isinstance(clinical_observed, Tensor) or clinical_observed.dtype != torch.bool
            or clinical_observed.shape != clinical_standardized.shape or clinical_observed.device != device):
        _fail()
    if (clinical_eligible is not None and
            (not isinstance(clinical_eligible, Tensor) or clinical_eligible.dtype != torch.bool
             or clinical_eligible.shape != clinical_standardized.shape or clinical_eligible.device != device)):
        _fail()
    if (not isinstance(retinal_standardized, Tensor) or not retinal_standardized.is_floating_point()
            or retinal_standardized.ndim != 3 or retinal_standardized.shape[0] != batch
            or retinal_standardized.shape[1] == 0 or retinal_standardized.shape[2] != 384
            or retinal_standardized.device != device or retinal_standardized.dtype != dtype):
        _fail()
    if (not isinstance(retinal_visible, Tensor) or retinal_visible.dtype != torch.bool
            or retinal_visible.shape != retinal_standardized.shape[:2] or retinal_visible.device != device):
        _fail()
    if (not isinstance(normalized_age, Tensor) or normalized_age.shape != (batch, 7)
            or normalized_age.device != device or normalized_age.dtype != dtype):
        _fail()
    try:
        validate_normalized_age(normalized_age)
    except Exception:
        _fail()
    effective = clinical_observed & model.eligible_slots[None, :]
    if clinical_eligible is not None:
        effective = effective & clinical_eligible
    if not bool(torch.isfinite(clinical_standardized[effective]).all()):
        _fail()
    if bool(retinal_visible.any()) and not bool(torch.isfinite(retinal_standardized[retinal_visible]).all()):
        _fail()

    target_mask = None
    clinical_values = clinical_standardized
    observed = clinical_observed
    if erase_cbc_targets:
        target_indices = tuple(model.cbc_indices)
        target_mask = observed[:, list(target_indices)].clone()
        if clinical_eligible is not None:
            target_mask &= clinical_eligible[:, list(target_indices)]
        clinical_values, observed = erase_cbc_for_completion(
            clinical_values, observed, target_indices)
    state = model.encode(clinical_values, observed, retinal_standardized,
                         retinal_visible, normalized_age, clinical_eligible)
    if (not bool(torch.isfinite(state.mean).all())
            or not bool(torch.isfinite(state.logvar).all())):
        _fail()
    logits = model.screening_joint_head(state.mean)
    screening = torch.sigmoid(logits)
    cbc = model.cbc_joint_head(state.mean)
    if (logits.shape != (batch, 26) or cbc.shape != (batch, 9)
            or not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(cbc).all())):
        _fail()
    active = ~state.abstain
    nan_screen = torch.full_like(screening, float("nan"))
    nan_cbc = torch.full_like(cbc, float("nan"))
    screening = torch.where(active[:, None], screening, nan_screen)
    cbc = torch.where(active[:, None], cbc, nan_cbc)
    original = None
    if loaded.input_transform is not None:
        if make_transform_binding(loaded.input_transform) != loaded.transform_binding:
            _fail()
        indices = tuple(model.cbc_indices)
        median = _transform_vector(loaded.input_transform, "clinical_median", indices,
                                   device=device, dtype=dtype)
        scale = _transform_vector(loaded.input_transform, "clinical_iqr", indices,
                                  device=device, dtype=dtype)
        original = cbc * scale[None, :] + median[None, :]
        original = torch.where(active[:, None], original, nan_cbc)
        if not bool(torch.isfinite(original[active]).all()):
            _fail()
    return R7InferenceResult(
        state_mean=state.mean.detach().clone(),
        state_logvar=state.logvar.detach().clone(),
        abstained=state.abstain.detach().clone(),
        screening_probability=screening.detach().clone(),
        cbc_standardized=cbc.detach().clone(),
        cbc_original_units=None if original is None else original.detach().clone(),
        completion_target_mask=None if target_mask is None else target_mask.detach().clone(),
    )
