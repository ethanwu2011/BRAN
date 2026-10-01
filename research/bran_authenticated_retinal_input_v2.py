"""Authenticate V2 retinal extraction artifacts before private pooling.

The consumer is intentionally local-only.  It authenticates the versioned
runner receipts, inventory, and feature NPY, then returns only readonly
per-person pools and a readonly presence mask.  It never fits, normalizes,
publishes, or prints patient-level output.
"""

from __future__ import annotations

import importlib
import os
import stat
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

import bran_authenticated_retinal_input_v1 as _v1
import bran_retinal_extraction_kernel_v1 as kernel


_ERROR = "bran_authenticated_retinal_input_v2_failed"
_ZERO_SHA256 = "0" * 64

# Reuse only the V1 module's pure, row-free helpers.  In particular, this
# module does not call or monkeypatch V1 receipt validators or its loader.
_digest_bytes = _v1._digest_bytes
_digest_file = _v1._digest_file
_read_json = _v1._read_json
_validate_patient_ids = _v1._validate_patient_ids
_validate_folds = _v1._validate_folds
_validate_selection = _v1._validate_selection
_validate_inventory_binding = _v1._validate_inventory_binding
pool_arrays = _v1.pool_arrays

_PROTOCOL_KEYS = frozenset(
    {
        "schema",
        "status",
        "parameters",
        "technical_parameters",
        "origin_protocol",
        "selection",
        "preflight_protocol_sha256",
        "preflight_aggregate_sha256",
        "preflight_audit_sha256",
        "v1_private_sha256",
        "code_sha256",
    }
)
_AGGREGATE_KEYS = frozenset(
    {
        "schema",
        "status",
        "protocol_sha256",
        "selection",
        "inventory_sha256",
        "inventory_file_sha256",
        "output_sha256",
        "dimension",
        "dtype",
        "historical_features_allclose",
        "preflight_audit_sha256",
        "patient_level_output_emitted",
        "official_test_images_encoded",
        "historical_production_proven",
        "clinical_benefit_claim",
        "model_promoted",
        "all_rows_authenticated_before_decode",
        "all_rows_written_once",
        "original_failure_cause_established",
        "v1_partial_features_reused",
        "scientific_parameters_changed",
    }
)
_AUDIT_KEYS = frozenset(
    {
        "schema",
        "status",
        "protocol_sha256",
        "aggregate_sha256",
        "inventory_file_sha256",
        "output_sha256",
        "selection",
        "preflight_audit_sha256",
        "independent_generator_replay_passed",
        "patient_level_output_emitted",
    }
)
_MANIFEST_KEYS = frozenset({"protocol_sha256", "artifact_sha256"})
_AGGREGATE_FLAGS = {
    "patient_level_output_emitted": False,
    "official_test_images_encoded": False,
    "historical_production_proven": False,
    "clinical_benefit_claim": False,
    "model_promoted": False,
    "all_rows_authenticated_before_decode": True,
    "all_rows_written_once": True,
    "original_failure_cause_established": False,
    "v1_partial_features_reused": False,
    "scientific_parameters_changed": False,
}


def _fail() -> NoReturn:
    raise ValueError(_ERROR) from None


def _runner_module() -> Any:
    try:
        return importlib.import_module("run_bran_retinal_extraction_v2")
    except Exception:
        _fail()


def _path_attr(runner: Any, name: str) -> Path:
    try:
        value = Path(getattr(runner, name))
    except Exception:
        _fail()
    return value


def _require_digest(value: Any) -> str:
    try:
        if not _v1._is_digest(value):
            _fail()
    except Exception:
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
        file_stat = os.lstat(path)
        if not stat.S_ISREG(file_stat.st_mode) or stat.S_IMODE(file_stat.st_mode) != 0o600:
            _fail()
    except Exception:
        _fail()


def _require_no_failure_files(output_dir: Path, audit_dir: Path, private_dir: Path) -> None:
    try:
        if any(
            (directory / "failure.json").exists()
            for directory in (output_dir, audit_dir, private_dir)
        ):
            _fail()
    except Exception:
        _fail()


def _validate_protocol(protocol: Any, protocol_pin: str) -> tuple[dict[str, Any], str]:
    if not isinstance(protocol, dict) or frozenset(protocol) != _PROTOCOL_KEYS:
        _fail()
    if (
        protocol.get("schema") != "bran-retinal-extraction-protocol-v2"
        or protocol.get("status") != "frozen_before_execution"
        or protocol.get("selection") != protocol.get("origin_protocol", {}).get("selection")
    ):
        _fail()
    for key in (
        "preflight_protocol_sha256",
        "preflight_aggregate_sha256",
        "preflight_audit_sha256",
    ):
        _require_digest(protocol.get(key))
    if not isinstance(protocol.get("origin_protocol"), dict):
        _fail()
    origin = protocol["origin_protocol"]
    if origin.get("schema") != "bran-retinal-extraction-protocol-v1":
        _fail()
    try:
        pinned_policy = origin["external_sha256"]["source_policy"]
    except Exception:
        _fail()
    _require_digest(pinned_policy)
    _validate_selection(protocol.get("selection"))
    return protocol, pinned_policy


def _validate_manifest(manifest: Any, protocol_pin: str, artifact_digest: str) -> None:
    if not isinstance(manifest, dict) or frozenset(manifest) != _MANIFEST_KEYS:
        _fail()
    if (
        manifest.get("protocol_sha256") != protocol_pin
        or manifest.get("artifact_sha256") != artifact_digest
    ):
        _fail()


def _validate_aggregate(
    aggregate: Any,
    protocol: dict[str, Any],
    protocol_pin: str,
) -> None:
    if not isinstance(aggregate, dict) or frozenset(aggregate) != _AGGREGATE_KEYS:
        _fail()
    if (
        aggregate.get("schema") != "bran-retinal-extraction-aggregate-v2"
        or aggregate.get("status") != "completed"
        or aggregate.get("protocol_sha256") != protocol_pin
        or aggregate.get("selection") != protocol["selection"]
        or aggregate.get("preflight_audit_sha256")
        != protocol["preflight_audit_sha256"]
        or aggregate.get("dimension") != 384
        or aggregate.get("dtype") != "float32"
        or type(aggregate.get("historical_features_allclose")) is not bool
    ):
        _fail()
    for key in ("inventory_sha256", "inventory_file_sha256", "output_sha256"):
        _require_digest(aggregate.get(key))
    for key, expected in _AGGREGATE_FLAGS.items():
        if type(aggregate.get(key)) is not bool or aggregate[key] is not expected:
            _fail()


def _validate_audit(
    audit: Any,
    aggregate_digest: str,
    protocol: dict[str, Any],
    protocol_pin: str,
) -> None:
    if not isinstance(audit, dict) or frozenset(audit) != _AUDIT_KEYS:
        _fail()
    if (
        audit.get("schema") != "bran-retinal-extraction-audit-v2"
        or audit.get("status") != "authenticated"
        or audit.get("protocol_sha256") != protocol_pin
        or audit.get("selection") != protocol["selection"]
        or audit.get("preflight_audit_sha256")
        != protocol["preflight_audit_sha256"]
        or audit.get("patient_level_output_emitted") is not False
        or type(audit.get("patient_level_output_emitted")) is not bool
        or audit.get("independent_generator_replay_passed") is not True
        or type(audit.get("independent_generator_replay_passed")) is not bool
    ):
        _fail()
    for key in (
        "aggregate_sha256",
        "inventory_file_sha256",
        "output_sha256",
    ):
        _require_digest(audit.get(key))
    if audit["aggregate_sha256"] != aggregate_digest:
        _fail()


def _validate_features(path: Path, expected_rows: int) -> tuple[str, np.memmap]:
    """Validate all feature bytes and values, returning an open map to pool."""

    mapped: np.memmap | None = None
    keep_open = False
    try:
        file_stat = os.stat(path)
        if not os.path.isfile(path) or os.path.islink(path):
            _fail()
        if (file_stat.st_mode & 0o777) != 0o600:
            _fail()
        loaded = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(loaded, np.memmap):
            _fail()
        mapped = loaded
        if (
            mapped.shape != (expected_rows, 384)
            or mapped.dtype != np.dtype(np.float32)
            or not bool(mapped.flags.c_contiguous)
        ):
            _fail()
        data_offset = int(mapped.offset)
        expected_size = data_offset + expected_rows * 384 * np.dtype(np.float32).itemsize
        if int(file_stat.st_size) != expected_size:
            _fail()
        rows_per_chunk = max(
            1,
            min(
                expected_rows,
                1024 * 1024 // (384 * np.dtype(np.float32).itemsize),
            ),
        )
        for start in range(0, expected_rows, rows_per_chunk):
            stop = min(expected_rows, start + rows_per_chunk)
            block = np.asarray(mapped[start:stop])
            if not bool(np.isfinite(block).all()):
                _fail()
            if bool(np.any(np.sum(np.abs(block), axis=1, dtype=np.float64) == 0.0)):
                _fail()
        digest = _digest_file(path)
        keep_open = True
        return digest, mapped
    except ValueError:
        _fail()
    except Exception:
        _fail()
    finally:
        if mapped is not None and not keep_open:
            _close_mmap(mapped)
    # The function either returns an open map or raises; the caller owns the
    # explicit close after pooling.
    _fail()


def _close_mmap(mapped: Any) -> None:
    try:
        mmap = getattr(mapped, "_mmap", None)
        if mmap is not None:
            mmap.close()
    except Exception:
        # Do not report successful consumption when explicit cleanup failed.
        _fail()


def _rehash(expected: dict[Path, str]) -> None:
    for path, digest in expected.items():
        if _digest_file(path) != digest:
            _fail()


def _load_impl(
    *,
    protocol_pin: str,
    audit_pin: str,
    patient_ids: Any,
    folds: Any,
    source_policy_sha256: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not _v1._is_digest(protocol_pin) or not _v1._is_digest(audit_pin):
        _fail()
    if not _v1._is_digest(source_policy_sha256):
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
    audit_path = audit_dir / "audit.json"
    audit_manifest_path = audit_dir / "audit.manifest.json"
    aggregate_path = output_dir / "aggregate.json"
    aggregate_manifest_path = output_dir / "aggregate.manifest.json"
    inventory_path = private_dir / "inventory.json"
    features_path = private_dir / "features.npy"

    _require_no_failure_files(output_dir, audit_dir, private_dir)
    for path in (
        protocol_path,
        audit_path,
        audit_manifest_path,
        aggregate_path,
        aggregate_manifest_path,
        features_path,
    ):
        _require_regular(path)
    _require_private_regular(inventory_path)

    protocol, protocol_bytes, protocol_digest = _read_json(protocol_path, dict)
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
    if (
        selection["patient_order_sha256"] != patient_digest
        or selection["fold_order_sha256"] != fold_digest
    ):
        _fail()
    try:
        pinned_policy = protocol["origin_protocol"]["external_sha256"]["source_policy"]
    except Exception:
        _fail()
    if pinned_policy != source_policy_sha256:
        _fail()

    audit, audit_bytes, audit_digest = _read_json(audit_path, dict)
    if audit_digest != audit_pin:
        _fail()
    aggregate, aggregate_bytes, aggregate_digest = _read_json(aggregate_path, dict)
    try:
        if not callable(runner.validate_result):
            _fail()
        runner.validate_result(aggregate, protocol, protocol_pin)
    except Exception:
        _fail()
    _validate_aggregate(aggregate, protocol, protocol_pin)
    _validate_audit(audit, aggregate_digest, protocol, protocol_pin)
    if (
        audit["aggregate_sha256"] != aggregate_digest
        or audit["inventory_file_sha256"] != aggregate["inventory_file_sha256"]
        or audit["output_sha256"] != aggregate["output_sha256"]
    ):
        _fail()

    aggregate_manifest, aggregate_manifest_bytes, aggregate_manifest_digest = _read_json(
        aggregate_manifest_path, dict
    )
    audit_manifest, audit_manifest_bytes, audit_manifest_digest = _read_json(
        audit_manifest_path, dict
    )
    _validate_manifest(aggregate_manifest, protocol_pin, aggregate_digest)
    _validate_manifest(audit_manifest, protocol_pin, audit_digest)

    records, inventory_bytes, inventory_digest = _read_json(inventory_path, list)
    _validate_inventory_binding(
        records,
        inventory_digest,
        aggregate,
        audit,
        selection,
        ids,
    )

    if (
        aggregate["inventory_file_sha256"] != inventory_digest
        or audit["inventory_file_sha256"] != inventory_digest
    ):
        _fail()

    output_digest, features = _validate_features(features_path, len(records))
    try:
        if output_digest != aggregate["output_sha256"] or output_digest != audit["output_sha256"]:
            _fail()
        pooled, present = pool_arrays(records, features, ids)
    finally:
        _close_mmap(features)
        features = None

    _require_no_failure_files(output_dir, audit_dir, private_dir)
    _rehash(
        {
            protocol_path: protocol_digest,
            audit_path: audit_digest,
            audit_manifest_path: audit_manifest_digest,
            aggregate_path: aggregate_digest,
            aggregate_manifest_path: aggregate_manifest_digest,
            inventory_path: inventory_digest,
            features_path: output_digest,
        }
    )
    # Artifact hashes alone do not revalidate runtime/source/code dependencies.
    # Check the immutable runner contract again after private computation.
    try:
        runner.validate_protocol(protocol, full=False)
    except Exception:
        _fail()
    _require_no_failure_files(output_dir, audit_dir, private_dir)
    if (
        not isinstance(pooled, np.ndarray)
        or not isinstance(present, np.ndarray)
        or pooled.shape != (len(ids), 384)
        or pooled.dtype != np.dtype(np.float64)
        or present.shape != (len(ids),)
        or present.dtype != np.dtype(bool)
        or pooled.flags.writeable
        or present.flags.writeable
        or not bool(np.isfinite(pooled).all())
    ):
        _fail()
    return pooled, present


def load_pooled_features(
    *,
    protocol_pin: str,
    audit_pin: str,
    patient_ids: Any,
    folds: Any,
    source_policy_sha256: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Authenticate V2 receipts, then return readonly private pools."""

    try:
        return _load_impl(
            protocol_pin=protocol_pin,
            audit_pin=audit_pin,
            patient_ids=patient_ids,
            folds=folds,
            source_policy_sha256=source_policy_sha256,
        )
    except ValueError:
        _fail()
    except Exception:
        _fail()
