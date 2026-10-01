"""Local-only serialization and inference for the compact BRAN research state.

This is an artifact boundary, not a trainer, cohort runner, or clinical
service.  It stores only model tensors, architecture metadata, an outer-fold
``FoldTransform`` normalization state, and a caller-supplied canonical field
schema.  It never writes inputs, posterior values, samples, or predictions.

The input retinal contract is one already-pooled 384-dimensional frozen
feature token per row, matching the exploratory diagnostic's FoldTransform.
There is no raw-pixel decoder.  Returned uncertainty is model uncertainty in
normalized coordinates; it has not been calibrated for clinical use.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from bran_patient_state_prototype_v1 import BRANPatientStatePrototypeV1, PatientStateConfig, PosteriorState
from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2, CLINICAL_ANCHOR_WEIGHT, V2_EXTENSION_NAME
from run_bran_overnight_diagnostic_v1 import FoldTransform


ARTIFACT_SCHEMA = "bran-research-state-io-v1"
MODEL_V1 = "bran_patient_state_prototype_v1"
MODEL_V2 = "bran_clinical_anchor_v2"
_MODEL_KINDS = frozenset((MODEL_V1, MODEL_V2))
_CONTINUOUS, _BINARY, _RETINAL, _STATE = 48, 11, 384, 192
_INELIGIBLE_CONTINUOUS = (8, 9, 20, 35, 36)
_DEFAULT_CONFIG = PatientStateConfig().to_dict()
_DEFAULT_V2_CONFIG = {
    "base_patient_state_config": _DEFAULT_CONFIG,
    "extension": {
        "name": V2_EXTENSION_NAME,
        "weight": CLINICAL_ANCHOR_WEIGHT,
        "target": "visible_eligible_normalized_continuous_clinical_fields_only",
        "head": "linear(shared_plus_clinical_private_to_48)",
        "pretrained": False,
        "downstream_raw_feature_bypass": False,
    },
}


class ResearchStateIOError(ValueError):
    """Fail-closed artifact or inference-contract failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_bindings() -> dict[str, str]:
    """Bind an artifact to precisely the local V1/V2 source it can load."""
    root = Path(__file__).resolve().parent
    names = ("bran_patient_state_prototype_v1.py", "bran_patient_state_anchor_v2.py",
             "run_bran_overnight_diagnostic_v1.py", "bran_research_state_io_v1.py",
             "PATIENT_ATLAS_FEATURE_REGISTRY.json")
    return {name: _sha256(root / name) for name in names}


def _basic_string_tuple(value: Sequence[str], name: str, expected: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != expected or any(not isinstance(x, str) or not x for x in value):
        raise ResearchStateIOError(f"invalid_{name}")
    return tuple(value)


def canonical_registry() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read only the row-free registry and require its immutable 59-field order."""
    path = Path(__file__).resolve().with_name("PATIENT_ATLAS_FEATURE_REGISTRY.json")
    try:
        value = json.loads(path.read_text())
        features = value["features"]
        if not isinstance(features, list) or len(features) != 59:
            raise TypeError
        if [item.get("index") for item in features] != list(range(59)):
            raise TypeError
        names = tuple(item["name"] for item in features)
        types = tuple(item["type"] for item in features)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ResearchStateIOError("canonical_registry_invalid") from error
    return _basic_string_tuple(names, "registry_names", 59), _basic_string_tuple(types, "registry_types", 59)


def validate_canonical_schema(field_names: Sequence[str], field_types: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Require exact names and order from the row-free canonical registry."""
    names = _basic_string_tuple(field_names, "field_names", _CONTINUOUS + _BINARY)
    types = _basic_string_tuple(field_types, "field_types", _CONTINUOUS + _BINARY)
    expected_names, expected_types = canonical_registry()
    if names != expected_names or types != expected_types:
        raise ResearchStateIOError("noncanonical_59_field_schema")
    return names, types


def _model_kind(model: BRANPatientStatePrototypeV1) -> str:
    if type(model) is BRANClinicalAnchorV2:
        return MODEL_V2
    if type(model) is BRANPatientStatePrototypeV1:
        return MODEL_V1
    raise ResearchStateIOError("unsupported_model_class")


def _validate_config(kind: str, config: Mapping[str, Any]) -> None:
    """Reject a mismatched or unsupported architecture before weight loading."""
    # Do this structural comparison before constructing a model.  An artifact
    # cannot ask this loader to allocate arbitrary architecture dimensions.
    if kind == MODEL_V1 and dict(config) == _DEFAULT_CONFIG:
        return
    if kind == MODEL_V2 and dict(config) == _DEFAULT_V2_CONFIG:
        return
    if kind not in _MODEL_KINDS:
        raise ResearchStateIOError("unknown_model_kind")
    raise ResearchStateIOError("nondefault_model_config")


def _tensor(value: Any, shape: tuple[int, ...], name: str) -> Tensor:
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise ResearchStateIOError(f"invalid_{name}")
    value = value.detach().cpu().to(torch.float64)
    if not torch.isfinite(value).all():
        raise ResearchStateIOError(f"nonfinite_{name}")
    return value


def _transform_payload(transform: FoldTransform) -> dict[str, Tensor]:
    """Extract only FoldTransform's numeric normalizer state, never training rows."""
    cmed = _tensor(torch.as_tensor(transform.clinical_median), (59,), "clinical_median")
    ciqr = _tensor(torch.as_tensor(transform.clinical_iqr), (59,), "clinical_iqr")
    rmean = _tensor(torch.as_tensor(transform.retinal_mean), (384,), "retinal_mean")
    rscale = _tensor(torch.as_tensor(transform.retinal_scale), (384,), "retinal_scale")
    age_mean = _tensor(torch.as_tensor([transform.age_mean], dtype=torch.float64), (1,), "age_mean")
    age_scale = _tensor(torch.as_tensor([transform.age_scale], dtype=torch.float64), (1,), "age_scale")
    if torch.any(ciqr <= 0) or torch.any(rscale <= 0) or age_scale.item() <= 0:
        raise ResearchStateIOError("nonpositive_normalization_scale")
    return {"clinical_median": cmed, "clinical_iqr": ciqr, "retinal_mean": rmean,
            "retinal_scale": rscale, "age_mean": age_mean, "age_scale": age_scale}


def _restore_transform(payload: Mapping[str, Any]) -> FoldTransform:
    required = {"clinical_median", "clinical_iqr", "retinal_mean", "retinal_scale", "age_mean", "age_scale"}
    if set(payload) != required:
        raise ResearchStateIOError("normalization_keys_invalid")
    values = {
        "clinical_median": _tensor(payload["clinical_median"], (59,), "clinical_median").numpy(),
        "clinical_iqr": _tensor(payload["clinical_iqr"], (59,), "clinical_iqr").numpy(),
        "retinal_mean": _tensor(payload["retinal_mean"], (384,), "retinal_mean").numpy(),
        "retinal_scale": _tensor(payload["retinal_scale"], (384,), "retinal_scale").numpy(),
        "age_mean": float(_tensor(payload["age_mean"], (1,), "age_mean")[0]),
        "age_scale": float(_tensor(payload["age_scale"], (1,), "age_scale")[0]),
    }
    if np.any(values["clinical_iqr"] <= 0) or np.any(values["retinal_scale"] <= 0) or values["age_scale"] <= 0:
        raise ResearchStateIOError("nonpositive_normalization_scale")
    result = FoldTransform.__new__(FoldTransform)
    for key, value in values.items():
        setattr(result, key, value)
    return result


def _eligibility(value: Sequence[bool]) -> Tensor:
    array = np.asarray(value)
    expected = np.zeros(59, dtype=bool)
    expected[:48] = True
    expected[list(_INELIGIBLE_CONTINUOUS)] = False
    if array.shape != (59,) or array.dtype.kind != "b" or not np.array_equal(array, expected):
        raise ResearchStateIOError("invalid_clinical_eligibility")
    return torch.from_numpy(array.astype(np.bool_, copy=True))


def save_research_state_artifact(
    path: str | Path,
    model: BRANPatientStatePrototypeV1,
    fold_transform: FoldTransform,
    field_names: Sequence[str],
    field_types: Sequence[str],
    clinical_eligibility: Sequence[bool],
) -> str:
    """Create one new owner-readable checkpoint and return its SHA-256.

    ``path`` is exclusive: an existing artifact is never overwritten.  The
    payload contains basic metadata and tensors only; inputs and inference
    outputs are intentionally not accepted by this API.
    """
    target = Path(path).resolve()
    if target.exists():
        raise ResearchStateIOError("artifact_path_exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    names, types = validate_canonical_schema(field_names, field_types)
    eligible = _eligibility(clinical_eligibility)
    kind = _model_kind(model)
    config = model.export_config()
    _validate_config(kind, config)
    state = {str(k): v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if not all(value.is_floating_point() and torch.isfinite(value).all() for value in state.values()):
        raise ResearchStateIOError("nonfinite_model_weights")
    payload: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "model_kind": kind,
        "model_config": config,
        "model_state_dict": state,
        "field_names": list(names),
        "field_types": list(types),
        "clinical_eligibility": eligible,
        "fold_transform": _transform_payload(fold_transform),
        "source_sha256": _source_bindings(),
    }
    try:
        fd = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ResearchStateIOError("artifact_path_exists") from error
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
        os.chmod(target, 0o600)
    except BaseException:
        # A failed exclusive write is not a usable checkpoint; leave the named
        # path rather than silently replace it on a later attempt.
        raise
    return _sha256(target)


@dataclass
class ResearchStateArtifact:
    """An in-memory trusted artifact with raw-to-normalized masked inference."""

    model: BRANPatientStatePrototypeV1
    fold_transform: FoldTransform
    field_names: tuple[str, ...]
    field_types: tuple[str, ...]
    clinical_eligibility: Tensor
    checkpoint_sha256: str
    model_kind: str

    def infer(
        self,
        clinical_values: np.ndarray | Tensor,
        clinical_observed_mask: np.ndarray | Tensor,
        retinal_features: np.ndarray | Tensor,
        retinal_visible_mask: np.ndarray | Tensor,
        age: np.ndarray | Tensor,
        *,
        samples: int = 0,
    ) -> dict[str, Tensor | PosteriorState]:
        """Infer from raw units, returning only normalized in-memory results.

        Clinical and retinal values masked by availability or eligibility are
        physically zeroed before FoldTransform arithmetic. ``samples`` adds
        coherent joint clinical posterior-predictive draws; it does not imply
        calibration, and no result is serialized by this method.
        """
        if samples < 0:
            raise ResearchStateIOError("samples_must_be_nonnegative")
        cv = np.asarray(clinical_values, dtype=np.float64)
        cm = np.asarray(clinical_observed_mask, dtype=bool)
        rf = np.asarray(retinal_features, dtype=np.float64)
        rm = np.asarray(retinal_visible_mask, dtype=bool)
        age_array = np.asarray(age, dtype=np.float64)
        if cv.ndim != 2 or cv.shape[1] != 59 or cm.shape != cv.shape:
            raise ResearchStateIOError("clinical_input_shape_invalid")
        if rf.shape != (len(cv), _RETINAL) or rm.shape != (len(cv),):
            raise ResearchStateIOError("retinal_input_shape_invalid")
        if age_array.shape not in ((len(cv),), (len(cv), 1)) or not np.isfinite(age_array).all():
            raise ResearchStateIOError("age_input_invalid")
        # FoldTransform itself uses safe-zero, but establish validity before
        # passing values across the artifact boundary as an explicit contract.
        allowed = self.clinical_eligibility.cpu().numpy()[None, :]
        clean_mask = cm & allowed & np.isfinite(cv)
        safe_clinical = np.zeros_like(cv)
        safe_clinical[clean_mask] = cv[clean_mask]
        safe_retinal = np.zeros_like(rf)
        retinal_valid = rm & np.isfinite(rf).all(axis=1)
        safe_retinal[retinal_valid] = rf[retinal_valid]
        normalized_c, normalized_cm, normalized_r, normalized_age = self.fold_transform.apply(
            safe_clinical, clean_mask, np.broadcast_to(allowed, cv.shape), safe_retinal, retinal_valid, age_array.reshape(-1)
        )
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            ct = torch.as_tensor(normalized_c, dtype=torch.float32, device=device)
            cmt = torch.as_tensor(normalized_cm, dtype=torch.bool, device=device)
            rt = torch.as_tensor(normalized_r[:, None, :], dtype=torch.float32, device=device)
            rmt = torch.as_tensor(retinal_valid[:, None], dtype=torch.bool, device=device)
            at = torch.as_tensor(normalized_age, dtype=torch.float32, device=device)
            state = self.model.encode(ct, cmt, rt, rmt, at, clinical_eligible_mask=self.clinical_eligibility.to(device)[None].expand_as(cmt))
            mean = self.model.predictive_mean(state, at)
            result: dict[str, Tensor | PosteriorState] = {
                "posterior": state,
                "posterior_mean": state.mean,
                "posterior_logvar": state.logvar,
                "abstain": state.abstain,
                "clinical_eligible_field_mask": self.clinical_eligibility[:48],
                "clinical_plugin_mean_normalized": self._mask_untrained_continuous(mean["continuous"]),
            }
            if samples:
                draws = self.model.sample_clinical(state, at, samples=samples)
                result["clinical_conditional_means_normalized"] = self._mask_untrained_continuous(draws["continuous_mean"])
                result["clinical_posterior_predictive_samples_normalized"] = self._mask_untrained_continuous(draws["continuous"])
        return result

    def _mask_untrained_continuous(self, value: Tensor) -> Tensor:
        """Never expose the five excluded or all-untrained binary channels."""
        allowed = self.clinical_eligibility[:48].to(value.device)
        return value.masked_fill(~allowed.reshape((1,) * (value.ndim - 1) + (-1,)), float("nan"))


def load_research_state_artifact(path: str | Path, expected_sha256: str) -> ResearchStateArtifact:
    """Load a SHA-bound local artifact with PyTorch's safe tensor-only loader."""
    source = Path(path).resolve()
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or not source.is_file() or _sha256(source) != expected_sha256:
        raise ResearchStateIOError("checkpoint_sha256_mismatch")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ResearchStateIOError("checkpoint_load_failed") from error
    required = {"schema", "model_kind", "model_config", "model_state_dict", "field_names", "field_types", "clinical_eligibility", "fold_transform", "source_sha256"}
    if not isinstance(payload, dict) or set(payload) != required or payload["schema"] != ARTIFACT_SCHEMA:
        raise ResearchStateIOError("checkpoint_schema_invalid")
    kind = payload["model_kind"]
    if kind not in _MODEL_KINDS or not isinstance(payload["model_config"], dict):
        raise ResearchStateIOError("checkpoint_model_metadata_invalid")
    bindings = payload["source_sha256"]
    if bindings != _source_bindings():
        raise ResearchStateIOError("checkpoint_source_binding_mismatch")
    names, types = validate_canonical_schema(payload["field_names"], payload["field_types"])
    eligible = _eligibility(payload["clinical_eligibility"].numpy() if isinstance(payload["clinical_eligibility"], Tensor) else ())
    _validate_config(kind, payload["model_config"])
    model: BRANPatientStatePrototypeV1
    model = BRANPatientStatePrototypeV1.from_config(payload["model_config"]) if kind == MODEL_V1 else BRANClinicalAnchorV2.from_config(payload["model_config"])
    if not isinstance(payload["model_state_dict"], dict):
        raise ResearchStateIOError("checkpoint_weights_invalid")
    if not payload["model_state_dict"] or not all(isinstance(v, Tensor) and v.is_floating_point() and torch.isfinite(v).all()
                                                  for v in payload["model_state_dict"].values()):
        raise ResearchStateIOError("checkpoint_weights_invalid")
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (RuntimeError, TypeError) as error:
        raise ResearchStateIOError("checkpoint_weights_invalid") from error
    return ResearchStateArtifact(model.eval(), _restore_transform(payload["fold_transform"]), names, types, eligible, expected_sha256, kind)
