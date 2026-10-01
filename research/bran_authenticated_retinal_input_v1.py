"""Authenticate a completed retinal extraction before private BRAN pooling.

The reader is intentionally local-only: it consumes the runner's private
protocol, receipts, inventory, and feature NPY, and returns only a per-person
pooled array plus a presence mask to its caller.  It does not fit, normalize,
publish, or emit patient-level output.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

import bran_retinal_extraction_kernel_v1 as kernel


_ERROR = "bran_authenticated_retinal_input_failed"
_ZERO_SHA256 = "0" * 64
_DIGEST_HEX = frozenset("0123456789abcdef")
_SELECTION_KEYS = frozenset(
    {
        "patient_order_sha256",
        "fold_order_sha256",
        "selection_sha256",
        "rows",
        "people",
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
        "independent_generator_replay_passed",
        "patient_level_output_emitted",
    }
)
_AGGREGATE_FLAGS = {
    "patient_level_output_emitted": False,
    "official_test_images_encoded": False,
    "historical_production_proven": False,
    "clinical_benefit_claim": False,
    "model_promoted": False,
    "all_rows_authenticated_before_decode": True,
    "all_rows_written_once": True,
}
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
        *tuple(_AGGREGATE_FLAGS),
    }
)


def _fail() -> NoReturn:
    raise ValueError(_ERROR)


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _DIGEST_HEX for character in value)
    )


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except Exception:
        _fail()
    return digest.hexdigest()


def _parse_json_bytes(data: bytes, want: type[Any]) -> Any:
    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail()
            result[key] = value
        return result

    try:
        parsed = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=no_duplicate_keys,
            parse_constant=lambda _: _fail(),
        )
    except Exception:
        _fail()
    if not isinstance(parsed, want):
        _fail()
    return parsed


def _read_json(path: Path, want: type[Any]) -> tuple[Any, bytes, str]:
    try:
        data = path.read_bytes()
    except Exception:
        _fail()
    return _parse_json_bytes(data, want), data, _digest_bytes(data)


def _digest_json(value: Any) -> str:
    try:
        data = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except Exception:
        _fail()
    return _digest_bytes(data)


def _runner_module() -> Any:
    try:
        return importlib.import_module("run_bran_retinal_extraction_v1")
    except Exception:
        _fail()


def _path_attr(runner: Any, name: str) -> Path:
    try:
        value = Path(getattr(runner, name))
    except Exception:
        _fail()
    return value


def _validate_patient_ids(patient_ids: Any) -> list[str]:
    try:
        if isinstance(patient_ids, np.ndarray):
            if patient_ids.ndim != 1:
                _fail()
            values = patient_ids.tolist()
        elif isinstance(patient_ids, (list, tuple)):
            values = list(patient_ids)
        else:
            _fail()
    except Exception:
        _fail()
    if not values or any(type(value) is not str or not value for value in values):
        _fail()
    if len(set(values)) != len(values):
        _fail()
    return values


def _validate_folds(folds: Any, expected_rows: int) -> list[Any]:
    try:
        if isinstance(folds, (list, tuple)) and any(isinstance(x, (bool, np.bool_)) for x in folds):
            _fail()
        values_array = np.asarray(folds)
        if values_array.ndim != 1 or values_array.shape[0] != expected_rows:
            _fail()
        values = values_array.tolist()
    except Exception:
        _fail()
    if any(type(value) is not int or value < 0 or value > 4 for value in values):
        _fail()
    # This also rejects NaN/Inf and non-JSON fold objects without exposing them.
    _digest_json(values)
    return values


def _validate_selection(selection: Any) -> dict[str, Any]:
    if not isinstance(selection, dict) or frozenset(selection) != _SELECTION_KEYS:
        _fail()
    for key in (
        "patient_order_sha256",
        "fold_order_sha256",
        "selection_sha256",
    ):
        if not _is_digest(selection.get(key)):
            _fail()
    if (
        type(selection.get("rows")) is not int
        or selection["rows"] <= 0
        or type(selection.get("people")) is not int
        or selection["people"] <= 0
    ):
        _fail()
    return selection


def _validate_aggregate(aggregate: Any, protocol: dict[str, Any], protocol_pin: str) -> None:
    if not isinstance(aggregate, dict) or frozenset(aggregate) != _AGGREGATE_KEYS:
        _fail()
    if (
        aggregate.get("schema") != "bran-retinal-extraction-aggregate-v1"
        or aggregate.get("status") != "completed"
        or aggregate.get("protocol_sha256") != protocol_pin
        or aggregate.get("selection") != protocol["selection"]
        or aggregate.get("dimension") != 384
        or aggregate.get("dtype") != "float32"
        or type(aggregate.get("historical_features_allclose")) is not bool
    ):
        _fail()
    for key in ("inventory_sha256", "inventory_file_sha256", "output_sha256"):
        if not _is_digest(aggregate.get(key)):
            _fail()
    for key, expected in _AGGREGATE_FLAGS.items():
        if type(aggregate.get(key)) is not bool or aggregate[key] is not expected:
            _fail()


def _validate_audit(
    audit: Any,
    protocol: dict[str, Any],
    protocol_pin: str,
) -> None:
    if not isinstance(audit, dict) or frozenset(audit) != _AUDIT_KEYS:
        _fail()
    if (
        audit.get("schema") != "bran-retinal-extraction-audit-v1"
        or audit.get("status") != "authenticated"
        or audit.get("protocol_sha256") != protocol_pin
        or audit.get("selection") != protocol["selection"]
        or audit.get("independent_generator_replay_passed") is not True
        or type(audit.get("independent_generator_replay_passed")) is not bool
        or audit.get("patient_level_output_emitted") is not False
        or type(audit.get("patient_level_output_emitted")) is not bool
    ):
        _fail()
    for key in (
        "aggregate_sha256",
        "inventory_file_sha256",
        "output_sha256",
    ):
        if not _is_digest(audit.get(key)):
            _fail()


def _validate_inventory_binding(
    records: Any,
    inventory_digest: str,
    aggregate: dict[str, Any],
    audit: dict[str, Any],
    selection: dict[str, Any],
    patient_ids: list[str],
) -> None:
    if not isinstance(records, list) or not records:
        _fail()
    try:
        kernel.validate_inventory(records)
    except Exception:
        _fail()
    if (
        inventory_digest != aggregate["inventory_file_sha256"]
        or inventory_digest != audit["inventory_file_sha256"]
        or selection["rows"] != len(records)
    ):
        _fail()
    try:
        semantic_digest = kernel.inventory_sha256(records)
    except Exception:
        _fail()
    if semantic_digest != aggregate["inventory_sha256"]:
        _fail()
    people = {record["person_id"] for record in records}
    if selection["people"] != len(patient_ids) or not people.issubset(set(patient_ids)):
        _fail()
    selected_records = []
    for record in records:
        selected = dict(record)
        selected["source_sha256"] = _ZERO_SHA256
        selected_records.append(selected)
    try:
        selected_digest = kernel.inventory_sha256(selected_records)
    except Exception:
        _fail()
    if selected_digest != selection["selection_sha256"]:
        _fail()


def pool_arrays(
    records: Any,
    features: Any,
    patient_ids: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool image rows by expected patient order using float64 accumulation."""

    ids = _validate_patient_ids(patient_ids)
    if not isinstance(features, np.ndarray):
        _fail()
    try:
        kernel.validate_inventory(records)
    except Exception:
        _fail()
    if (
        features.shape != (len(records), 384)
        or features.dtype != np.dtype(np.float32)
    ):
        _fail()
    try:
        if not bool(np.isfinite(features).all()):
            _fail()
        if bool(np.any(np.sum(np.abs(features), axis=1, dtype=np.float64) == 0.0)):
            _fail()
    except Exception:
        _fail()

    positions = {patient_id: index for index, patient_id in enumerate(ids)}
    pooled = np.zeros((len(ids), 384), dtype=np.float64)
    counts = np.zeros(len(ids), dtype=np.int64)
    for row, record in enumerate(records):
        try:
            position = positions[record["person_id"]]
            pooled[position] += np.asarray(features[row], dtype=np.float64)
            counts[position] += 1
        except Exception:
            _fail()
    present = counts > 0
    observed = present
    if bool(np.any(observed)):
        pooled[observed] /= counts[observed, None]
    pooled[~observed] = 0.0
    if not bool(np.isfinite(pooled).all()):
        _fail()
    pooled.setflags(write=False)
    present = np.asarray(present, dtype=bool)
    present.setflags(write=False)
    return pooled, present


def load_pooled_features(
    *,
    protocol_pin: str,
    audit_pin: str,
    patient_ids: Any,
    folds: Any,
    source_policy_sha256: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Authenticate private extraction receipts, then return readonly pools."""

    if not _is_digest(protocol_pin) or not _is_digest(audit_pin):
        _fail()
    if not _is_digest(source_policy_sha256):
        _fail()
    ids = _validate_patient_ids(patient_ids)
    fold_values = _validate_folds(folds, len(ids))
    patient_digest = _digest_json(ids)
    fold_digest = _digest_json(fold_values)

    runner = _runner_module()
    protocol_path = _path_attr(runner, "PROTOCOL")
    audit_dir = _path_attr(runner, "AUDIT")
    output_dir = _path_attr(runner, "OUT")
    private_dir = _path_attr(runner, "PRIVATE")
    audit_path = audit_dir / "audit.json"
    aggregate_path = output_dir / "aggregate.json"
    inventory_path = private_dir / "inventory.json"
    features_path = private_dir / "features.npy"

    try:
        if (output_dir / "failure.json").exists() or (audit_dir / "failure.json").exists():
            _fail()
    except Exception:
        _fail()

    protocol, _, protocol_digest = _read_json(protocol_path, dict)
    if protocol_digest != protocol_pin:
        _fail()
    try:
        if not callable(runner.validate_protocol):
            _fail()
        runner.validate_protocol(protocol, full=False)
    except Exception:
        _fail()
    selection = _validate_selection(protocol.get("selection"))
    if (
        selection["patient_order_sha256"] != patient_digest
        or selection["fold_order_sha256"] != fold_digest
    ):
        _fail()
    try:
        external = protocol["external_sha256"]
        pinned_policy = external["source_policy"]
    except Exception:
        _fail()
    if not _is_digest(pinned_policy) or pinned_policy != source_policy_sha256:
        _fail()

    audit, _, audit_digest = _read_json(audit_path, dict)
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
    _validate_audit(audit, protocol, protocol_pin)
    if audit["aggregate_sha256"] != aggregate_digest:
        _fail()

    try:
        inventory_stat = os.lstat(inventory_path)
        if (
            not stat.S_ISREG(inventory_stat.st_mode)
            or stat.S_IMODE(inventory_stat.st_mode) != 0o600
        ):
            _fail()
    except Exception:
        _fail()
    records, inventory_bytes, inventory_digest = _read_json(inventory_path, list)
    _validate_inventory_binding(
        records,
        inventory_digest,
        aggregate,
        audit,
        selection,
        ids,
    )

    try:
        validated_output = kernel.validate_output(
            features_path,
            len(records),
            width=384,
        )
    except Exception:
        _fail()
    if (
        not isinstance(validated_output, dict)
        or validated_output.get("rows") != len(records)
        or validated_output.get("dimension") != 384
        or validated_output.get("dtype") != "float32"
        or validated_output.get("output_sha256") != aggregate["output_sha256"]
        or validated_output.get("output_sha256") != audit["output_sha256"]
    ):
        _fail()

    try:
        features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    except Exception:
        _fail()
    pooled, present = pool_arrays(records, features, ids)

    if (
        _digest_file(inventory_path) != inventory_digest
        or _digest_file(features_path) != validated_output["output_sha256"]
    ):
        _fail()
    # Keep local references alive through the byte-integrity check, but never
    # return the mapped source array.
    del features, inventory_bytes, aggregate_bytes
    return pooled, present
