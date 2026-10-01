"""Immutable loader for the two retained historical multisource-mask roles.

The caller must already hold the legacy job lock and FD-quiet boundary.  This
module reads the legacy protocol metadata and invokes legacy authenticated
terminal loaders; it has no training path and never serializes models, arrays,
or patient-derived outputs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

import run_bran_multisource_mask_v1 as old


_INVALID = "historical reference roles invalid"
_ROLES = ("initial", "continued")


@dataclass(frozen=True, repr=False)
class HistoricalReferenceContextV2:
    """Authenticated legacy metadata, recursively frozen and private to caller."""

    protocol_pin: str
    audit_pin: str
    protocol: Mapping[str, Any]
    result: Mapping[str, Any]
    manifest: Mapping[str, Any]
    role_metadata: Mapping[str, Any]


def _invalid() -> None:
    raise ValueError(_INVALID)


def _require(value: bool) -> None:
    if not value:
        _invalid()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            _invalid()
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    _invalid()


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _same(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pin(value: Any) -> str:
    try:
        valid = old.valid_sha(value)
    except (AttributeError, TypeError):
        valid = False
    _require(bool(valid))
    return value


def _read_protocol() -> dict[str, Any]:
    path = getattr(old, "PROTOCOL", None)
    _require(isinstance(path, Path) and path.is_file() and not path.is_symlink())
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        _invalid()
    _require(isinstance(value, dict))
    return value


def _authentication(protocol: Mapping[str, Any], result: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        auth = protocol["native_source"]["source"]["authentication"]
    except (KeyError, TypeError):
        _invalid()
    _require(isinstance(auth, Mapping) and result.get("fold_authentication") == auth)
    try:
        outer, inner = auth["outer_fold_sha256"], auth["inner_fold_sha256"]
    except (KeyError, TypeError):
        _invalid()
    _pin(outer)
    _require(isinstance(inner, (list, tuple)) and len(inner) == 5)
    for item in inner:
        _pin(item)
    return auth


def _validate_authenticated(protocol: Mapping[str, Any], result: Mapping[str, Any], manifest: Mapping[str, Any],
                            protocol_pin: str) -> None:
    _require(protocol.get("schema") == "bran-multisource-mask-protocol-v1"
             and protocol.get("status") == "frozen_before_execution")
    _require(result.get("schema") == "bran-multisource-mask-aggregate-v1" and result.get("status") == "completed")
    _require(manifest.get("protocol_sha256") == protocol_pin)
    _authentication(protocol, result)
    try:
        native_checkpoints = protocol["native_source"]["checkpoint_sha256"]
        retained_checkpoints = manifest["checkpoint_sha256"]
        weights = protocol["parameters"]["combined_weights"]
    except (KeyError, TypeError):
        _invalid()
    _require(isinstance(native_checkpoints, Mapping) and isinstance(retained_checkpoints, Mapping)
             and set(native_checkpoints) == set(retained_checkpoints) == {"fold" + str(fold) for fold in range(5)})
    _require(isinstance(weights, Mapping) and set(weights) == {"continued", "student"}
             and weights["continued"] == 0.0 and weights["student"] == 0.5)
    for fold in range(5):
        _pin(native_checkpoints["fold" + str(fold)])
        _pin(retained_checkpoints["fold" + str(fold)])


def _safe_metadata(protocol_pin: str, audit_pin: str, protocol: Mapping[str, Any], manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    auth = _authentication(protocol, {"fold_authentication": protocol["native_source"]["source"]["authentication"]})
    return MappingProxyType({
        "schema": "bran-multisource-historical-reference-roles-v2",
        "historical_protocol_sha256": protocol_pin,
        "historical_audit_sha256": audit_pin,
        "roles": MappingProxyType({
            "initial": "immutable historical per-fold native checkpoint loaded by legacy origin.load_initial",
            "continued": "immutable historical per-fold continued state loaded into a deepcopy of that initial model",
        }),
        "excluded_role": "historical student is intentionally not returned",
        "serving_checkpoint_role": "retained full-frame serving checkpoint is not a held-out fold and is not returned",
        "fold_authentication": MappingProxyType({"outer_fold_sha256": auth["outer_fold_sha256"],
                                                    "inner_fold_sha256": tuple(auth["inner_fold_sha256"])}),
        "continued_checkpoint_sha256": MappingProxyType({key: manifest["checkpoint_sha256"][key]
                                                            for key in sorted(manifest["checkpoint_sha256"])}),
        "patient_level_output_emitted": False,
        "training_performed": False,
    })


def authenticate_references(protocol_pin: str, audit_pin: str) -> HistoricalReferenceContextV2:
    """Authenticate immutable V1 terminal roles using caller-supplied pins only."""

    protocol_pin, audit_pin = _pin(protocol_pin), _pin(audit_pin)
    protocol = _read_protocol()
    # These legacy calls verify protocol/result/manifest/audit byte bindings,
    # all private checkpoint hashes, normalizer bundles, and final replay.
    try:
        result, manifest = old.authenticate_terminal(protocol, protocol_pin)
        audited = old.authenticate_audit(protocol, protocol_pin, audit_pin)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError):
        _invalid()
    _require(isinstance(result, dict) and isinstance(manifest, dict) and isinstance(audited, dict)
             and _same(result, audited))
    _validate_authenticated(protocol, result, manifest, protocol_pin)
    metadata = _safe_metadata(protocol_pin, audit_pin, protocol, manifest)
    return HistoricalReferenceContextV2(protocol_pin, audit_pin, _freeze(copy.deepcopy(protocol)),
                                        _freeze(copy.deepcopy(result)), _freeze(copy.deepcopy(manifest)), metadata)


def role_description(context: HistoricalReferenceContextV2) -> Mapping[str, Any]:
    """Return safe, serializable role text and hashes only; no model or arrays."""

    _require(isinstance(context, HistoricalReferenceContextV2))
    return _thaw(context.role_metadata)


def _transform_for_fold(context: HistoricalReferenceContextV2, fold: int, transform: Any) -> None:
    _require(isinstance(fold, int) and not isinstance(fold, bool) and 0 <= fold < 5)
    _require(transform is not None and getattr(transform, "heldout_fold", fold) == fold)
    # FoldTransformV2's hashes use a different serialization from the legacy
    # historical frame.  Comparing them would create a false identity claim;
    # reject that contract rather than silently substituting either hash.
    _require(not hasattr(transform, "fold_identity_sha256") and not hasattr(transform, "training_indices_sha256"))
    # The exact historical frame is instead verified by legacy load_initial and
    # read_checkpoint, then by equality of every saved normalizer below.
    normalizers = getattr(old.origin, "NORMALIZERS", ())
    _require(isinstance(normalizers, tuple) and len(normalizers) > 0)
    for key in normalizers:
        _require(isinstance(key, str) and hasattr(transform, key))


def fold_models(context: HistoricalReferenceContextV2, fold: int, historical_transform: Any) -> dict[str, Any]:
    """Load only initial and continued historical per-fold models, frozen for eval."""

    _require(isinstance(context, HistoricalReferenceContextV2))
    _transform_for_fold(context, fold, historical_transform)
    protocol, manifest = _thaw(context.protocol), _thaw(context.manifest)
    key = "fold" + str(fold)
    try:
        initial = old.origin.load_initial(fold, historical_transform, protocol)
        bundle = old.read_checkpoint(protocol, context.protocol_pin, fold, manifest["checkpoint_sha256"][key])
    except (ValueError, TypeError, KeyError, OSError, RuntimeError):
        _invalid()
    _require(isinstance(bundle, dict) and bundle.get("fold") == fold
             and bundle.get("protocol_sha256") == context.protocol_pin
             and bundle.get("initial_checkpoint_sha256") == protocol["native_source"]["checkpoint_sha256"][key]
             and isinstance(bundle.get("continued"), dict))
    for normalizer in old.origin.NORMALIZERS:
        try:
            same = np.array_equal(bundle[normalizer], getattr(historical_transform, normalizer))
        except (KeyError, TypeError):
            same = False
        _require(bool(same))
    try:
        continued = copy.deepcopy(initial)
        continued.load_state_dict(bundle["continued"], strict=True)
        for model in (initial, continued):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
    except (AttributeError, RuntimeError, TypeError):
        _invalid()
    return {"initial": initial, "continued": continued}
