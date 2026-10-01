"""Authenticate V3 retinal receipts before local, no-fit person pooling."""

from __future__ import annotations

import importlib
import os
import stat
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

import bran_authenticated_retinal_input_v1 as _v1
import bran_authenticated_retinal_input_v2 as _v2


_ERROR = "bran_authenticated_retinal_input_v3_failed"

# Reuse only pure V1/V2 parsing, digest, array-validation, and no-fit pooling
# helpers.  Receipt schemas and the runner import are V3-specific below.
_digest_file = _v2._digest_file
_read_json = _v2._read_json
_validate_patient_ids = _v2._validate_patient_ids
_validate_folds = _v2._validate_folds
_validate_selection = _v2._validate_selection
_validate_inventory_binding = _v2._validate_inventory_binding
_validate_features = _v2._validate_features
_close_mmap = _v2._close_mmap
pool_arrays = _v2.pool_arrays

_PROTOCOL_KEYS = frozenset({
    "schema", "status", "parameters", "technical_parameters", "origin_protocol",
    "selection", "preflight_protocol_sha256", "preflight_aggregate_sha256",
    "preflight_audit_sha256", "v1_private_sha256", "previous_protocol_sha256",
    "previous_failure_sha256", "previous_inventory_sha256", "code_sha256",
})
_AGGREGATE_KEYS = frozenset({
    "schema", "status", "protocol_sha256", "selection", "inventory_sha256",
    "inventory_file_sha256", "output_sha256", "dimension", "dtype",
    "historical_features_allclose", "preflight_audit_sha256",
    "patient_level_output_emitted", "official_test_images_encoded",
    "historical_production_proven", "clinical_benefit_claim", "model_promoted",
    "all_rows_authenticated_before_decode", "all_rows_written_once",
    "original_failure_cause_established", "v1_partial_features_reused",
    "scientific_parameters_changed", "gpu_preflight_passed",
})
_AUDIT_KEYS = frozenset({
    "schema", "status", "protocol_sha256", "aggregate_sha256",
    "inventory_file_sha256", "output_sha256", "selection",
    "preflight_audit_sha256", "gpu_preflight_passed",
    "independent_generator_replay_passed", "patient_level_output_emitted",
})
_MANIFEST_KEYS = frozenset({"protocol_sha256", "artifact_sha256"})
_FLAGS = {**_v2._AGGREGATE_FLAGS, "gpu_preflight_passed": True}


def _fail() -> NoReturn:
    raise ValueError(_ERROR) from None


def _runner_module() -> Any:
    try:
        return importlib.import_module("run_bran_retinal_extraction_v3")
    except Exception:
        _fail()


def _path_attr(runner: Any, name: str) -> Path:
    try:
        return Path(getattr(runner, name))
    except Exception:
        _fail()


def _require_digest(value: Any) -> str:
    if not _v1._is_digest(value):
        _fail()
    return value


def _require_regular(path: Path) -> None:
    try:
        if not path.is_file() or path.is_symlink():
            _fail()
    except Exception:
        _fail()


def _require_private_regular(path: Path) -> None:
    try:
        value = os.lstat(path)
        if not stat.S_ISREG(value.st_mode) or stat.S_IMODE(value.st_mode) != 0o600:
            _fail()
    except Exception:
        _fail()


def _require_private_dir(path: Path) -> None:
    try:
        value = os.lstat(path)
        if not stat.S_ISDIR(value.st_mode) or stat.S_IMODE(value.st_mode) != 0o700:
            _fail()
    except Exception:
        _fail()


def _require_no_failure_files(*directories: Path) -> None:
    try:
        if any((directory / "failure.json").exists() for directory in directories):
            _fail()
    except Exception:
        _fail()


def _validate_protocol(protocol: Any, pin: str) -> dict[str, Any]:
    if not isinstance(protocol, dict) or frozenset(protocol) != _PROTOCOL_KEYS:
        _fail()
    if (protocol.get("schema") != "bran-retinal-extraction-protocol-v3"
            or protocol.get("status") != "frozen_before_execution"
            or protocol.get("selection") != protocol.get("origin_protocol", {}).get("selection")):
        _fail()
    for key in ("preflight_protocol_sha256", "preflight_aggregate_sha256",
                "preflight_audit_sha256", "previous_protocol_sha256",
                "previous_failure_sha256", "previous_inventory_sha256"):
        _require_digest(protocol.get(key))
    origin = protocol.get("origin_protocol")
    if not isinstance(origin, dict) or origin.get("schema") != "bran-retinal-extraction-protocol-v1":
        _fail()
    try:
        _require_digest(origin["external_sha256"]["source_policy"])
    except Exception:
        _fail()
    _validate_selection(protocol["selection"])
    _require_digest(pin)
    return protocol


def _validate_manifest(value: Any, pin: str, digest: str) -> None:
    if not isinstance(value, dict) or frozenset(value) != _MANIFEST_KEYS:
        _fail()
    if value.get("protocol_sha256") != pin or value.get("artifact_sha256") != digest:
        _fail()


def _validate_aggregate(value: Any, protocol: dict[str, Any], pin: str) -> None:
    if not isinstance(value, dict) or frozenset(value) != _AGGREGATE_KEYS:
        _fail()
    if (value.get("schema") != "bran-retinal-extraction-aggregate-v3"
            or value.get("status") != "completed" or value.get("protocol_sha256") != pin
            or value.get("selection") != protocol["selection"]
            or value.get("preflight_audit_sha256") != protocol["preflight_audit_sha256"]
            or value.get("dimension") != 384 or value.get("dtype") != "float32"
            or type(value.get("historical_features_allclose")) is not bool):
        _fail()
    for key in ("inventory_sha256", "inventory_file_sha256", "output_sha256"):
        _require_digest(value.get(key))
    for key, expected in _FLAGS.items():
        if type(value.get(key)) is not bool or value[key] is not expected:
            _fail()


def _validate_audit(value: Any, aggregate_digest: str, protocol: dict[str, Any], pin: str) -> None:
    if not isinstance(value, dict) or frozenset(value) != _AUDIT_KEYS:
        _fail()
    if (value.get("schema") != "bran-retinal-extraction-audit-v3"
            or value.get("status") != "authenticated" or value.get("protocol_sha256") != pin
            or value.get("aggregate_sha256") != aggregate_digest
            or value.get("selection") != protocol["selection"]
            or value.get("preflight_audit_sha256") != protocol["preflight_audit_sha256"]
            or value.get("gpu_preflight_passed") is not True
            or value.get("independent_generator_replay_passed") is not True
            or value.get("patient_level_output_emitted") is not False):
        _fail()
    for key in ("inventory_file_sha256", "output_sha256"):
        _require_digest(value.get(key))


def _rehash(expected: dict[Path, str]) -> None:
    for path, digest in expected.items():
        if _digest_file(path) != digest:
            _fail()


def _load_impl(*, protocol_pin: str, audit_pin: str, patient_ids: Any, folds: Any,
               source_policy_sha256: str) -> tuple[np.ndarray, np.ndarray]:
    if not _v1._is_digest(protocol_pin) or not _v1._is_digest(audit_pin) or not _v1._is_digest(source_policy_sha256):
        _fail()
    ids = _validate_patient_ids(patient_ids)
    fold_values = _validate_folds(folds, len(ids))
    patient_digest = _v1._digest_json(ids)
    fold_digest = _v1._digest_json(fold_values)
    runner = _runner_module()
    protocol_path = _path_attr(runner, "PROTOCOL")
    audit_dir = _path_attr(runner, "AUDIT")
    output_dir = _path_attr(runner, "OUT")
    private_dir = _path_attr(runner, "PRIVATE")
    audit_path, audit_manifest = audit_dir / "audit.json", audit_dir / "audit.manifest.json"
    aggregate_path, aggregate_manifest = output_dir / "aggregate.json", output_dir / "aggregate.manifest.json"
    inventory_path, features_path = private_dir / "inventory.json", private_dir / "features.npy"
    _require_no_failure_files(output_dir, audit_dir, private_dir)
    for path in (protocol_path, audit_path, audit_manifest, aggregate_path, aggregate_manifest, features_path):
        _require_regular(path)
    _require_private_dir(private_dir)
    _require_private_regular(inventory_path)
    protocol, _, protocol_digest = _read_json(protocol_path, dict)
    if protocol_digest != protocol_pin:
        _fail()
    _validate_protocol(protocol, protocol_pin)
    try:
        if not callable(runner.validate_protocol):
            _fail()
        runner.validate_protocol(protocol, full=False)
    except Exception:
        _fail()
    selection = _validate_selection(protocol["selection"])
    if selection["patient_order_sha256"] != patient_digest or selection["fold_order_sha256"] != fold_digest:
        _fail()
    if protocol["origin_protocol"]["external_sha256"]["source_policy"] != source_policy_sha256:
        _fail()
    audit, _, audit_digest = _read_json(audit_path, dict)
    if audit_digest != audit_pin:
        _fail()
    aggregate, _, aggregate_digest = _read_json(aggregate_path, dict)
    try:
        if not callable(runner.validate_result):
            _fail()
        runner.validate_result(aggregate, protocol, protocol_pin)
    except Exception:
        _fail()
    _validate_aggregate(aggregate, protocol, protocol_pin)
    _validate_audit(audit, aggregate_digest, protocol, protocol_pin)
    if audit["inventory_file_sha256"] != aggregate["inventory_file_sha256"] or audit["output_sha256"] != aggregate["output_sha256"]:
        _fail()
    aggregate_m, _, aggregate_m_digest = _read_json(aggregate_manifest, dict)
    audit_m, _, audit_m_digest = _read_json(audit_manifest, dict)
    _validate_manifest(aggregate_m, protocol_pin, aggregate_digest)
    _validate_manifest(audit_m, protocol_pin, audit_digest)
    records, _, inventory_digest = _read_json(inventory_path, list)
    _validate_inventory_binding(records, inventory_digest, aggregate, audit, selection, ids)
    if aggregate["inventory_file_sha256"] != inventory_digest or audit["inventory_file_sha256"] != inventory_digest:
        _fail()
    output_digest, features = _validate_features(features_path, len(records))
    try:
        if output_digest != aggregate["output_sha256"] or output_digest != audit["output_sha256"]:
            _fail()
        pooled, present = pool_arrays(records, features, ids)
    finally:
        _close_mmap(features)
    _require_no_failure_files(output_dir, audit_dir, private_dir)
    try:
        runner.validate_protocol(protocol, full=False)
    except Exception:
        _fail()
    _require_no_failure_files(output_dir, audit_dir, private_dir)
    _require_private_dir(private_dir)
    _require_private_regular(inventory_path)
    _require_private_regular(features_path)
    # The second source/code validation may take time. Bind the terminal files
    # after that call too, rather than accepting a mutation during validation.
    _rehash({protocol_path: protocol_digest, audit_path: audit_digest, audit_manifest: audit_m_digest,
             aggregate_path: aggregate_digest, aggregate_manifest: aggregate_m_digest,
             inventory_path: inventory_digest, features_path: output_digest})
    _require_no_failure_files(output_dir, audit_dir, private_dir)
    if (not isinstance(pooled, np.ndarray) or not isinstance(present, np.ndarray)
            or pooled.shape != (len(ids), 384) or pooled.dtype != np.dtype(np.float64)
            or present.shape != (len(ids),) or present.dtype != np.dtype(bool)
            or pooled.flags.writeable or present.flags.writeable or not np.isfinite(pooled).all()):
        _fail()
    return pooled, present


def load_pooled_features(*, protocol_pin: str, audit_pin: str, patient_ids: Any,
                         folds: Any, source_policy_sha256: str) -> tuple[np.ndarray, np.ndarray]:
    """Authenticate completed V3 artifacts and return readonly no-fit pools."""
    try:
        return _load_impl(protocol_pin=protocol_pin, audit_pin=audit_pin, patient_ids=patient_ids,
                          folds=folds, source_policy_sha256=source_policy_sha256)
    except Exception:
        _fail()
