"""Versioned local-only learning curves for four already-extracted BRAN FMs.

This runner deliberately has no extraction or encoder-training path.  It accepts
only a separately materialized, row-free cache manifest and reads the pinned
embedding arrays locally while the descriptor streams are silenced.  The
historical named-FM aggregate is used as a full-budget canary; every endpoint is
replayed internally, while the released result remains macro-only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import exclusive_json, sha
import run_bran_learning_curve_v1 as lc
import run_bran_blood_learning_curve_v1 as blood
import bran_named_fm_auc_v1 as probe


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "BRAN_FM_LEARNING_CURVE_PROTOCOL_V1.json"
OUT = ROOT / "BRAN_FM_LEARNING_CURVE_V1"
AUDIT = ROOT / "BRAN_FM_LEARNING_CURVE_AUDIT_V1"
DEFAULT_CACHE_MANIFEST = ROOT / "private_artifacts" / "bran_named_fm_cache_v1" / "manifest.json"
LOCK_PATH = Path("/private/tmp/bran_fm_learning_curve_v1.lock")

BUDGETS = tuple(lc.BUDGETS)
FRACTIONS = tuple(lc.FRACTIONS)
ARMS = tuple(probe.WIDTHS)
WIDTHS = dict(probe.WIDTHS)
PATIENT_COUNT = 1928
ENDPOINT_COUNT = 26
OUTER_FOLD_COUNT = 5
INNER_FOLD_COUNT = 5
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 91501
SUBSET_SEED_BASE = 97001
REPLAY_TOLERANCE = 1e-8

# The named-FM release is the approved aggregate source used by the completed
# attempt-2 benchmark.  These are source pins, not new evidence.
NAMED_PROTOCOL_SHA = "7da7d9781d1f9d0f00df4ab12923eb3024c94d63b2c0539df35a5fa63904dd1e"
NAMED_SUCCESS_SHA = "d5af27264ba109117c9e767fd1344dd37d05ff1a461376acf4c1a09982978144"

# The native learning-curve source supplies the canonical cohort, folds and
# existing nested standardized-logistic readout.  The blood curve is not run.
LC_PROTOCOL_SHA = "066dead55b897feb8cf1dae29c8d870117b1e437c92808efc332b2de28b121e2"
LC_AGGREGATE_SHA = "6f0e0bc152259d741d03a1ab17d253262fcb0dcb2135378816f27d9cdefade78"
LC_AUDIT_SHA = "b88e635217fa890a048feca57e52fe58de4381af1102c7ef13cc711ba5e3ab92"

SCHEMA = "bran-fm-learning-curve-v1"
PROTOCOL_SCHEMA = "bran-fm-learning-curve-protocol-v1"
CACHE_SCHEMA = "bran-named-fm-embedding-cache-v1"

PARAMETERS = {
    "arms": list(ARMS),
    "dimensions": WIDTHS,
    "outer_folds": OUTER_FOLD_COUNT,
    "inner_folds": INNER_FOLD_COUNT,
    "fractions": list(FRACTIONS),
    "budgets": list(BUDGETS),
    "subset_seed_base": SUBSET_SEED_BASE,
    "head": "existing nested StandardScaler + logistic; C=.01,.1,1,10; max_iter500",
    "head_C_grid": [0.01, 0.1, 1.0, 10.0],
    "head_selection": "inner AUROC then log loss then smaller C",
    "incomplete_inner_support": "fixed StandardScaler + logistic C=1 max_iter5000; no tuning",
    "sparse_training_support": "Laplace prevalence if observed training<20 or one class; .5 if none",
    "nonconvergence_fallback": False,
    "normalizers_fit_on_training_subset_only": True,
    "age_in_every_arm": True,
    "preextracted_fm_cache_only": True,
    "encoder_extraction": False,
    "encoder_training": False,
    "checkpoint_write": False,
    "reference_replay": "all 26 AUROCs for all four FMs plus macro on all observed endpoint rows",
    "comparison_mask": "observed endpoint AND retinal present AND any eligible clinical observed",
    "reference_replay_mask": "all observed endpoint rows, as approved named-FM aggregate",
    "bootstrap_draws": BOOTSTRAP_DRAWS,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "minimum_release_per_class": 20,
    "minimum_valid_draws": 900,
    "fixed_fit_intervals": True,
    "patient_level_output_permitted": False,
    "automatic_promotion": False,
    "adaptive_development": True,
    "official_test_used": False,
    "extrapolation": False,
}

PRIVACY = {
    "patient_processing_local_only": True,
    "patient_rows_ids_images_predictions_embeddings_draws_serialized": False,
    "cache_manifest_row_free_required": True,
    "extraction_or_encoder_training": False,
    "official_test_loaded": False,
    "hosted_inference_used": False,
    "automatic_promotion": False,
    "clinical_use_claim": False,
}

PATHS = {
    "output": "BRAN_FM_LEARNING_CURVE_V1",
    "audit": "BRAN_FM_LEARNING_CURVE_AUDIT_V1",
    "progress": "BRAN_FM_LEARNING_CURVE_V1/progress.json",
    "lock": str(LOCK_PATH),
}

CODE = (
    "run_bran_fm_learning_curve_v1.py",
    "test_bran_fm_learning_curve_v1.py",
    "BRAN_FM_LEARNING_CURVE_DESIGN_V1.md",
    "run_bran_blood_learning_curve_v1.py",
    "run_bran_learning_curve_v1.py",
    "bran_learning_curve_sampling_v1.py",
    "bran_named_fm_auc_v1.py",
    "bran_named_fm_reference_v1.py",
    "bran_named_fm_experiment_v1_attempt2.py",
    "bran_named_fm_extraction_v1_attempt2.py",
    "build_bran_named_fm_results_v1.py",
    "run_bran_overnight_diagnostic_v1.py",
)

PHASES = {
    "protocol",
    "cache",
    "context",
    "inner_folds",
    "fm_heads",
    "reference_replay",
    "aggregate",
    "writing",
    "audit",
    "completed",
}

CACHE_KEYS = {
    "schema",
    "status",
    "variant_order",
    "patient_count",
    "dimensions",
    "embedding_files",
    "artifact_files",
    "current_inputs_sha256",
    "row_order_sha256",
    "outer_fold_sha256",
    "inner_fold_sha256",
    "patient_level_output_emitted",
    "encoder_training",
}
CACHE_FILE_KEYS = {"path", "sha256", "dtype", "shape", "array_key"}
ARTIFACT_FILE_KEYS = {"path", "sha256"}


class FMCacheUnavailable(RuntimeError):
    """A required frozen local cache is absent or cannot be authenticated."""


def require(condition: Any, message: str = "fm_learning_curve_contract_failed") -> None:
    if not condition:
        raise ValueError(message)


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _require_sha(value: Any, message: str = "invalid_sha256") -> str:
    require(_valid_sha(value), message)
    return str(value)


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and np.isfinite(value)


def _exact_keys(value: Mapping[str, Any], keys: set[str], message: str) -> None:
    require(isinstance(value, Mapping) and set(value) == keys, message)


def _json_load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FMCacheUnavailable("cache_manifest_unreadable") from exc
    require(isinstance(value, Mapping), "cache_manifest_not_mapping")
    return value


def _resolve_path(value: Any, *, label: str) -> Path:
    require(isinstance(value, str) and value, label + "_path_invalid")
    path = Path(value).expanduser().resolve()
    require(path.is_file(), label + "_missing")
    return path


def _stat_mode(path: Path, expected: int | None = None) -> None:
    if expected is not None:
        require(path.stat().st_mode & 0o777 == expected, "cache_file_mode_invalid")


def _array_digest(h: "hashlib._Hash", name: str, value: Any) -> None:
    array = np.asarray(value)
    require(array.dtype != np.dtype("O"), "current_input_object_array")
    h.update(name.encode("utf-8")); h.update(b"\0")
    h.update(array.dtype.str.encode("ascii")); h.update(b"\0")
    h.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")); h.update(b"\0")
    h.update(np.ascontiguousarray(array).tobytes(order="C")); h.update(b"\0")


def current_inputs_sha256(
    clinical: np.ndarray,
    clinical_mask: np.ndarray,
    eligible: np.ndarray,
    retinal: np.ndarray,
    retinal_present: np.ndarray,
    ages: np.ndarray,
    names: Sequence[str],
) -> str:
    """Digest current non-label inputs without returning or serializing them."""

    h = hashlib.sha256()
    for name, value in (
        ("clinical", clinical),
        ("clinical_mask", clinical_mask),
        ("eligible", eligible),
        ("retinal", retinal),
        ("retinal_present", retinal_present),
        ("ages", ages),
    ):
        _array_digest(h, name, value)
    h.update(json.dumps([str(x) for x in names], separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def row_order_sha256(patient_ids: Sequence[Any]) -> str:
    """Digest canonical patient order locally; the digest itself is row-free."""

    h = hashlib.sha256()
    for value in patient_ids:
        encoded = str(value).encode("utf-8")
        h.update(len(encoded).to_bytes(8, "little", signed=False)); h.update(encoded)
    return h.hexdigest()


def _load_cache_manifest(path: Path) -> tuple[Mapping[str, Any], str]:
    path = Path(path).expanduser().resolve()
    require(path.is_file(), "cache_manifest_missing")
    digest = sha(path)
    value = _json_load(path)
    _validate_cache_manifest(value)
    return value, digest


def _validate_cache_manifest(value: Mapping[str, Any]) -> None:
    _exact_keys(value, CACHE_KEYS, "cache_manifest_schema_invalid")
    require(value["schema"] == CACHE_SCHEMA and value["status"] == "frozen_local_only", "cache_manifest_identity_invalid")
    require(value["variant_order"] == list(ARMS), "cache_variant_order_invalid")
    require(type(value["patient_count"]) is int and value["patient_count"] == PATIENT_COUNT, "cache_patient_count_invalid")
    require(value["dimensions"] == WIDTHS, "cache_dimensions_invalid")
    require(value["patient_level_output_emitted"] is False and value["encoder_training"] is False, "cache_privacy_invalid")
    for key in ("current_inputs_sha256", "row_order_sha256", "outer_fold_sha256"):
        _require_sha(value[key], "cache_" + key + "_invalid")
    require(isinstance(value["inner_fold_sha256"], list) and len(value["inner_fold_sha256"]) == INNER_FOLD_COUNT, "cache_inner_folds_invalid")
    for digest in value["inner_fold_sha256"]:
        _require_sha(digest, "cache_inner_fold_sha256_invalid")
    require(isinstance(value["embedding_files"], Mapping) and set(value["embedding_files"]) == set(ARMS), "cache_embedding_files_invalid")
    require(isinstance(value["artifact_files"], Mapping) and set(value["artifact_files"]) == set(ARMS), "cache_artifact_files_invalid")
    for arm in ARMS:
        item = value["embedding_files"][arm]
        _exact_keys(item, CACHE_FILE_KEYS, "cache_embedding_file_schema_invalid")
        _require_sha(item["sha256"], "cache_embedding_sha256_invalid")
        require(item["dtype"] == "float32" and item["shape"] == [PATIENT_COUNT, WIDTHS[arm]], "cache_embedding_shape_invalid")
        require(item["array_key"] in (None, "", "embedding", arm), "cache_embedding_key_invalid")
        require(isinstance(item["path"], str) and item["path"], "cache_embedding_path_invalid")
        artifact = value["artifact_files"][arm]
        _exact_keys(artifact, ARTIFACT_FILE_KEYS, "cache_artifact_file_schema_invalid")
        _require_sha(artifact["sha256"], "cache_artifact_sha256_invalid")
        require(isinstance(artifact["path"], str) and artifact["path"], "cache_artifact_path_invalid")


def _cache_binding(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Authenticate a row-free cache manifest and return its frozen binding."""

    manifest_path = Path(path).expanduser().resolve()
    value, manifest_sha = _load_cache_manifest(manifest_path)
    def normalize_files(items: Mapping[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for arm in ARMS:
            item = dict(items[arm])
            item["path"] = str((manifest_path.parent / item["path"]).resolve()) if not Path(item["path"]).expanduser().is_absolute() else str(Path(item["path"]).expanduser().resolve())
            normalized[arm] = item
        return normalized

    result = {
        "schema": value["schema"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "variant_order": list(value["variant_order"]),
        "patient_count": value["patient_count"],
        "dimensions": dict(value["dimensions"]),
        "embedding_files": normalize_files(value["embedding_files"]),
        "artifact_files": normalize_files(value["artifact_files"]),
        "current_inputs_sha256": value["current_inputs_sha256"],
        "row_order_sha256": value["row_order_sha256"],
        "outer_fold_sha256": value["outer_fold_sha256"],
        "inner_fold_sha256": list(value["inner_fold_sha256"]),
        "patient_level_output_emitted": False,
        "encoder_training": False,
    }
    # Resolve all files now so preparation cannot freeze a dangling binding.
    _verify_cache_files(result, verify_embeddings=False)
    return result


def _verify_cache_files(binding: Mapping[str, Any], *, verify_embeddings: bool) -> None:
    for arm in ARMS:
        embedding = binding["embedding_files"][arm]
        embedding_path = _resolve_path(embedding["path"], label="cache_embedding")
        require(sha(embedding_path) == embedding["sha256"], "cache_embedding_changed")
        artifact = binding["artifact_files"][arm]
        artifact_path = _resolve_path(artifact["path"], label="cache_artifact")
        require(sha(artifact_path) == artifact["sha256"], "cache_artifact_changed")
        if verify_embeddings:
            _load_embedding_array(embedding, arm)


def _load_embedding_array(spec: Mapping[str, Any], arm: str) -> np.ndarray:
    path = _resolve_path(spec["path"], label="cache_embedding")
    try:
        loaded = np.load(path, mmap_mode="r", allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            key = spec["array_key"] or "embedding"
            require(set(loaded.files) == {key}, "cache_npz_keys_invalid")
            array = loaded[key]
            loaded.close()
        else:
            array = loaded
    except Exception as exc:
        raise FMCacheUnavailable("cache_embedding_unreadable") from exc
    require(array.dtype == np.dtype("float32") and array.shape == (PATIENT_COUNT, WIDTHS[arm]), "cache_embedding_shape_changed")
    require(np.isfinite(np.asarray(array)).all(), "cache_embedding_nonfinite")
    # Preserve an immutable view for the caller; no copy is needed for heads.
    return np.asarray(array)


def load_fm_cache(
    binding: Mapping[str, Any] | str | Path,
    *,
    expected_current_inputs_sha256: str | None = None,
    expected_row_order_sha256: str | None = None,
    expected_outer_fold_sha256: str | None = None,
    expected_inner_fold_sha256: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load only authenticated pre-extracted arrays; never extract or train.

    ``binding`` is normally the protocol's ``cache`` mapping.  A manifest path
    is accepted for synthetic/unit use, but it is still validated against every
    artifact and identity field before any array is returned.
    """

    if isinstance(binding, (str, Path)):
        binding = _cache_binding(Path(binding))
    require(isinstance(binding, Mapping), "cache_binding_invalid")
    needed = {
        "schema", "manifest_path", "manifest_sha256", "variant_order", "patient_count", "dimensions",
        "embedding_files", "artifact_files", "current_inputs_sha256", "row_order_sha256",
        "outer_fold_sha256", "inner_fold_sha256", "patient_level_output_emitted", "encoder_training",
    }
    _exact_keys(binding, needed, "cache_binding_schema_invalid")
    require(binding["schema"] == CACHE_SCHEMA and binding["variant_order"] == list(ARMS), "cache_binding_identity_invalid")
    manifest_path = _resolve_path(binding["manifest_path"], label="cache_manifest")
    require(sha(manifest_path) == binding["manifest_sha256"], "cache_manifest_changed")
    manifest, _ = _load_cache_manifest(manifest_path)
    reconstructed = _cache_binding(manifest_path)
    require(reconstructed == dict(binding), "cache_binding_manifest_mismatch")
    if expected_current_inputs_sha256 is not None:
        require(binding["current_inputs_sha256"] == expected_current_inputs_sha256, "cache_current_inputs_mismatch")
    if expected_row_order_sha256 is not None:
        require(binding["row_order_sha256"] == expected_row_order_sha256, "cache_row_order_mismatch")
    if expected_outer_fold_sha256 is not None:
        require(binding["outer_fold_sha256"] == expected_outer_fold_sha256, "cache_outer_fold_mismatch")
    if expected_inner_fold_sha256 is not None:
        require(list(binding["inner_fold_sha256"]) == list(expected_inner_fold_sha256), "cache_inner_fold_mismatch")
    _verify_cache_files(binding, verify_embeddings=True)
    arrays = {arm: _load_embedding_array(binding["embedding_files"][arm], arm) for arm in ARMS}
    # Keep the manifest variable intentionally consumed: parsing is part of the
    # current-input/cache identity check, even though values are not returned.
    require(manifest["schema"] == CACHE_SCHEMA, "cache_manifest_replay_failed")
    return arrays


def _approved_reference(root: Path = ROOT) -> dict[str, Any]:
    """Read the approved aggregate source and retain only aggregate metrics."""

    try:
        import build_bran_named_fm_results_v1 as release

        require(sha(root / release.experiment.PROTOCOL) == NAMED_PROTOCOL_SHA, "named_fm_protocol_changed")
        require(sha(root / release.experiment.PATHS["success"]) == NAMED_SUCCESS_SHA, "named_fm_success_changed")
        report, summary = release.source(root, external=False)
    except Exception as exc:
        if isinstance(exc, (ValueError, KeyError, OSError, TypeError, RuntimeError)):
            raise
        raise ValueError("approved_named_fm_source_unavailable") from exc
    result = report["result"]
    endpoint_results = result["endpoint_results"]
    require(len(endpoint_results) == ENDPOINT_COUNT, "approved_named_fm_endpoint_count_changed")
    require(set(endpoint_results) == set(report["scope"]["eligible_source_codes"]), "approved_named_fm_endpoint_identity_changed")
    endpoint_reference = {
        arm: {endpoint: float(endpoint_results[endpoint]["arms"][arm]["auroc"]) for endpoint in endpoint_results}
        for arm in ARMS
    }
    macro_reference = {arm: float(summary["macro_auroc"][arm]) for arm in ARMS}
    for arm in ARMS:
        require(all(_finite(value) and 0 <= value <= 1 for value in endpoint_reference[arm].values()), "approved_named_fm_metric_invalid")
        require(abs(np.mean(list(endpoint_reference[arm].values())) - macro_reference[arm]) < 1e-10, "approved_named_fm_macro_changed")
    return {
        "protocol_sha256": NAMED_PROTOCOL_SHA,
        "success_sha256": NAMED_SUCCESS_SHA,
        "endpoint_names": list(endpoint_results),
        "endpoint_auroc": endpoint_reference,
        "macro_auroc": macro_reference,
    }


def _learning_binding(root: Path = ROOT) -> dict[str, Any]:
    require(sha(lc.PROTOCOL) == LC_PROTOCOL_SHA, "learning_source_protocol_changed")
    require(sha(lc.OUT / "aggregate.json") == LC_AGGREGATE_SHA, "learning_source_aggregate_changed")
    require(sha(lc.AUDIT / "audit.json") == LC_AUDIT_SHA, "learning_source_audit_changed")
    native_protocol = json.loads(lc.PROTOCOL.read_text(encoding="utf-8"))
    lc.validate_protocol(native_protocol)
    aggregate = json.loads((lc.OUT / "aggregate.json").read_text(encoding="utf-8"))
    lc.validate_result(aggregate, native_protocol)
    source = native_protocol["native_source"]["source"]
    endpoints = list(source["endpoint_names"])
    require(len(endpoints) == ENDPOINT_COUNT and len(set(endpoints)) == ENDPOINT_COUNT, "learning_endpoint_identity_changed")
    authentication = source["authentication"]
    require(_valid_sha(authentication["outer_fold_sha256"]), "learning_outer_fold_hash_invalid")
    require(isinstance(authentication["inner_fold_sha256"], list) and len(authentication["inner_fold_sha256"]) == INNER_FOLD_COUNT, "learning_inner_fold_hash_invalid")
    for digest in authentication["inner_fold_sha256"]:
        _require_sha(digest, "learning_inner_fold_hash_invalid")
    return {
        "protocol_sha256": LC_PROTOCOL_SHA,
        "aggregate_sha256": LC_AGGREGATE_SHA,
        "audit_sha256": LC_AUDIT_SHA,
        "endpoint_names": endpoints,
        "outer_fold_sha256": authentication["outer_fold_sha256"],
        "inner_fold_sha256": list(authentication["inner_fold_sha256"]),
    }


def _runtime() -> dict[str, str]:
    try:
        import sklearn

        sklearn_version = str(sklearn.__version__)
    except Exception:
        sklearn_version = "unavailable"
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scikit_learn": sklearn_version,
        "platform": platform.platform(),
        "device": "cpu",
        "threads": "2",
    }


def _code_hashes(root: Path = ROOT) -> dict[str, str]:
    result = {}
    for name in CODE:
        path = root / name
        require(path.is_file(), "code_component_missing")
        result[name] = sha(path)
    return result


def prepare(root: Path = ROOT, cache_manifest: Path | None = None) -> dict[str, Any]:
    """Build a frozen protocol, or fail explicitly when the FM cache is absent."""

    root = Path(root).resolve()
    cache_path = DEFAULT_CACHE_MANIFEST if cache_manifest is None else Path(cache_manifest).expanduser().resolve()
    if not cache_path.is_file():
        raise FMCacheUnavailable("preexecution_fm_cache_missing")
    learning = _learning_binding(root)
    reference = _approved_reference(root)
    require(reference["endpoint_names"] == learning["endpoint_names"], "approved_endpoint_order_differs")
    cache = _cache_binding(cache_path, root=root)
    require(cache["patient_count"] == PATIENT_COUNT and cache["dimensions"] == WIDTHS, "cache_scope_changed")
    require(cache["outer_fold_sha256"] == learning["outer_fold_sha256"], "cache_outer_fold_differs")
    require(cache["inner_fold_sha256"] == learning["inner_fold_sha256"], "cache_inner_fold_differs")
    return {
        "schema": PROTOCOL_SCHEMA,
        "status": "frozen_before_execution",
        "parameters": PARAMETERS,
        "privacy": PRIVACY,
        "paths": PATHS,
        "learning_source": learning,
        "approved_reference": reference,
        "cache": cache,
        "runtime": _runtime(),
        "code_sha256": _code_hashes(root),
    }


def validate_protocol(p: Mapping[str, Any], root: Path = ROOT, *, verify_cache: bool = True) -> Mapping[str, Any]:
    root = Path(root).resolve()
    expected_keys = {"schema", "status", "parameters", "privacy", "paths", "learning_source", "approved_reference", "cache", "runtime", "code_sha256"}
    _exact_keys(p, expected_keys, "fm_learning_curve_protocol_schema_invalid")
    require(p["schema"] == PROTOCOL_SCHEMA and p["status"] == "frozen_before_execution", "fm_learning_curve_protocol_identity_invalid")
    require(p["parameters"] == PARAMETERS and p["privacy"] == PRIVACY and p["paths"] == PATHS, "fm_learning_curve_protocol_parameters_changed")
    require(p["code_sha256"] == _code_hashes(root), "fm_learning_curve_code_changed")
    learning = _learning_binding(root)
    require(p["learning_source"] == learning, "fm_learning_curve_learning_source_changed")
    reference = _approved_reference(root)
    require(p["approved_reference"] == reference, "fm_learning_curve_reference_changed")
    cache = p["cache"]
    cache_keys = {"schema", "manifest_path", "manifest_sha256", "variant_order", "patient_count", "dimensions", "embedding_files", "artifact_files", "current_inputs_sha256", "row_order_sha256", "outer_fold_sha256", "inner_fold_sha256", "patient_level_output_emitted", "encoder_training"}
    _exact_keys(cache, cache_keys, "fm_learning_curve_cache_binding_schema_invalid")
    require(cache["schema"] == CACHE_SCHEMA and cache["variant_order"] == list(ARMS), "fm_learning_curve_cache_binding_invalid")
    require(cache["outer_fold_sha256"] == learning["outer_fold_sha256"] and cache["inner_fold_sha256"] == learning["inner_fold_sha256"], "fm_learning_curve_cache_fold_binding_changed")
    if verify_cache:
        reconstructed = _cache_binding(Path(cache["manifest_path"]), root=root)
        require(reconstructed == dict(cache), "fm_learning_curve_cache_manifest_changed")
    return p


def check_endpoint_replay(actual: Mapping[str, Any], reference: Mapping[str, Any], tolerance: float = REPLAY_TOLERANCE) -> None:
    """Require every FM/end-point canary, not only a matching macro."""

    require(set(actual) == set(reference) and set(actual) == set(ARMS), "fm_endpoint_replay_keys_invalid")
    for arm in ARMS:
        require(isinstance(actual[arm], Mapping) and isinstance(reference[arm], Mapping), "fm_endpoint_replay_shape_invalid")
        require(set(actual[arm]) == set(reference[arm]) and len(actual[arm]) == ENDPOINT_COUNT, "fm_endpoint_replay_endpoint_keys_invalid")
        for endpoint in reference[arm]:
            require(_finite(actual[arm][endpoint]) and 0 <= actual[arm][endpoint] <= 1, "fm_endpoint_replay_metric_invalid")
            require(abs(float(actual[arm][endpoint]) - float(reference[arm][endpoint])) <= tolerance, "fm_endpoint_replay_failed")


def check_macro_replay(actual: Mapping[str, float], reference: Mapping[str, float], tolerance: float = REPLAY_TOLERANCE) -> None:
    require(set(actual) == set(reference) == set(ARMS), "fm_macro_replay_keys_invalid")
    for arm in ARMS:
        require(_finite(actual[arm]) and 0 <= actual[arm] <= 1 and abs(float(actual[arm]) - float(reference[arm])) <= tolerance, "fm_macro_replay_failed")


def _support_mask(rm: np.ndarray, c: np.ndarray, cm: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    rm = np.asarray(rm, bool); c = np.asarray(c, float); cm = np.asarray(cm, bool); eligible = np.asarray(eligible, bool)
    require(rm.shape == (len(c),) and c.shape == cm.shape == eligible.shape and c.ndim == 2, "native_support_shape_invalid")
    return rm & (cm & eligible & np.isfinite(c)).any(axis=1)


def common_evaluation_mask(observed: np.ndarray, native_common: np.ndarray, labels: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, bool); native_common = np.asarray(native_common, bool); labels = np.asarray(labels, float)
    require(observed.shape == native_common.shape == labels.shape, "native_support_endpoint_shape_invalid")
    return observed & native_common


def _subset_hashes(folds: np.ndarray) -> tuple[dict[str, str], dict[str, list[int]]]:
    from bran_learning_curve_sampling_v1 import nested_training_subsets

    folds = np.asarray(folds)
    require(folds.shape == (PATIENT_COUNT,) and folds.dtype.kind in "iu" and set(np.unique(folds)) == set(range(OUTER_FOLD_COUNT)), "outer_fold_array_invalid")
    hashes: dict[str, str] = {}
    counts: dict[str, list[int]] = {budget: [] for budget in BUDGETS}
    for fold in range(OUTER_FOLD_COUNT):
        train = np.flatnonzero(folds != fold); test = np.flatnonzero(folds == fold)
        subsets = nested_training_subsets(train, test, PATIENT_COUNT, seed=SUBSET_SEED_BASE + fold)
        for budget in BUDGETS:
            selected = np.asarray(subsets[budget], dtype=np.intp)
            hashes[f"{fold}_{budget}"] = hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest()
            counts[budget].append(int(len(selected)))
    return hashes, counts


def _source_context() -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    source = lc.old.native.source
    return source, *source.io.load_context()


def _validate_context(p: Mapping[str, Any], source: Any, ctx: Mapping[str, Any], folds: np.ndarray, c0: np.ndarray,
                     cm0: np.ndarray, eligible: np.ndarray, r0: np.ndarray, rm: np.ndarray, ages: np.ndarray,
                     names: Sequence[str]) -> tuple[list[str], list[str]]:
    learning = p["learning_source"]
    endpoints = list(learning["endpoint_names"])
    require(len(folds) == PATIENT_COUNT and set(np.unique(folds)) == set(range(OUTER_FOLD_COUNT)), "cohort_fold_identity_invalid")
    require(c0.shape == cm0.shape == eligible.shape == (PATIENT_COUNT, 59), "clinical_context_shape_invalid")
    require(r0.shape == (PATIENT_COUNT, 384) and rm.shape == (PATIENT_COUNT,) and np.asarray(rm).dtype == bool, "retinal_context_shape_invalid")
    require(np.asarray(ages).shape == (PATIENT_COUNT,) and np.isfinite(ages).all(), "age_context_invalid")
    require(tuple(endpoints) == tuple(p["approved_reference"]["endpoint_names"]), "endpoint_order_changed")
    require(set(endpoints) <= set(ctx["labels_by_source"]) and set(endpoints) <= set(ctx["observed_by_source"]), "endpoint_payload_missing")
    for endpoint in endpoints:
        y = np.asarray(ctx["labels_by_source"][endpoint]); observed = np.asarray(ctx["observed_by_source"][endpoint], bool)
        require(y.shape == observed.shape == (PATIENT_COUNT,) and np.isfinite(y[observed]).all() and np.isin(y[observed], [0, 1]).all(), "endpoint_payload_invalid")
    row_ids = getattr(ctx.get("raw_cohort"), "patient_ids", None) if isinstance(ctx, Mapping) else None
    require(row_ids is not None and len(row_ids) == PATIENT_COUNT, "row_order_identity_unavailable")
    input_digest = current_inputs_sha256(c0, cm0, eligible, r0, rm, ages, names)
    order_digest = row_order_sha256(row_ids)
    cache = p["cache"]
    require(input_digest == cache["current_inputs_sha256"], "current_inputs_identity_changed")
    require(order_digest == cache["row_order_sha256"], "row_order_identity_changed")
    return endpoints, [str(x) for x in row_ids]


def _inner_assignments(p: Mapping[str, Any], source: Any, ctx: Mapping[str, Any], folds: np.ndarray) -> list[np.ndarray]:
    result = []
    expected = p["learning_source"]["inner_fold_sha256"]
    for fold in range(OUTER_FOLD_COUNT):
        train = np.flatnonzero(folds != fold)
        inner, digest = source.base._inner_context(ctx, train, fold)
        require(digest == expected[fold] and np.asarray(inner).shape == (len(train),), "inner_fold_identity_changed")
        require(set(np.unique(inner)) == set(range(INNER_FOLD_COUNT)), "inner_fold_values_invalid")
        result.append(np.asarray(inner))
    return result


def _fit_curve(p: Mapping[str, Any], source: Any, ctx: Mapping[str, Any], folds: np.ndarray, c0: np.ndarray,
               cm0: np.ndarray, eligible: np.ndarray, r0: np.ndarray, rm: np.ndarray, ages: np.ndarray,
               names: Sequence[str], endpoints: Sequence[str], embeddings: Mapping[str, np.ndarray],
               inner_assignments: Sequence[np.ndarray]) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    from bran_learning_curve_sampling_v1 import nested_training_subsets

    n = len(folds)
    predictions = {budget: {arm: {endpoint: np.full(n, np.nan) for endpoint in endpoints} for arm in ARMS} for budget in BUDGETS}
    subset_hashes, training_counts = _subset_hashes(folds)
    routes: set[str] = set()
    native_common = _support_mask(rm, c0, cm0, eligible)
    for fold in range(OUTER_FOLD_COUNT):
        outer_train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        subsets = nested_training_subsets(outer_train, test, n, seed=SUBSET_SEED_BASE + fold)
        for budget in BUDGETS:
            selected = np.asarray(subsets[budget], dtype=np.intp)
            inner_selected = np.asarray(inner_assignments[fold])[np.searchsorted(outer_train, selected)]
            lc.old.native.source.base._atomic_progress(OUT / "progress.json", "fm_heads", fold)
            for arm in ARMS:
                x = np.c_[np.asarray(embeddings[arm], float), np.asarray(ages, float)]
                require(x.shape == (n, WIDTHS[arm] + 1) and np.isfinite(x).all(), "fm_design_invalid")
                for endpoint in endpoints:
                    y = np.asarray(ctx["labels_by_source"][endpoint])
                    observed = np.asarray(ctx["observed_by_source"][endpoint], bool)
                    prediction, route = blood.fit_predict(x, y, observed, selected, test, inner_selected)
                    routes.add(route)
                    require(np.asarray(prediction).shape == (len(test),) and np.isfinite(prediction).all() and np.all((prediction >= 0) & (prediction <= 1)), "fm_prediction_invalid")
                    if budget == "100":
                        require(route == "nested", "full_budget_head_not_nested")
                    predictions[budget][arm][endpoint][test] = prediction
    # Full-budget canary uses the approved all-observed endpoint mask.
    actual_reference = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for endpoint in endpoints:
            y = np.asarray(ctx["labels_by_source"][endpoint]); observed = np.asarray(ctx["observed_by_source"][endpoint], bool)
            actual_reference[arm][endpoint] = source.base.fold_weighted_auc(y, predictions["100"][arm][endpoint], observed, folds)
    check_endpoint_replay(actual_reference, p["approved_reference"]["endpoint_auroc"])
    check_macro_replay({arm: float(np.mean(list(actual_reference[arm].values()))) for arm in ARMS}, p["approved_reference"]["macro_auroc"])
    counts = source.ev.paired_counts(folds, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    curves: dict[str, Any] = {}
    macro_draws: dict[str, np.ndarray] = {}
    for budget, fraction in zip(BUDGETS, FRACTIONS):
        points: dict[str, list[float]] = {arm: [] for arm in ARMS}
        draws: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
        for endpoint in endpoints:
            y = np.asarray(ctx["labels_by_source"][endpoint]); observed = np.asarray(ctx["observed_by_source"][endpoint], bool)
            mask = common_evaluation_mask(observed, native_common, y)
            require(np.sum(mask & (y == 0)) >= 20 and np.sum(mask & (y == 1)) >= 20, "native_common_support_below_20")
            for arm in ARMS:
                pred = predictions[budget][arm][endpoint]
                points[arm].append(source.base.fold_weighted_auc(y, pred, mask, folds))
                draws[arm].append(source.base._weighted_auc_draws(y, pred, mask, folds, counts))
        macro_draws[budget] = {arm: np.mean(np.stack(draws[arm], axis=0), axis=0) for arm in ARMS}
        for arm in ARMS:
            require(np.isfinite(macro_draws[budget][arm]).sum() >= PARAMETERS["minimum_valid_draws"], "native_common_bootstrap_support_below_900")
        curves[budget] = {
            "n": float(np.mean(training_counts[budget])),
            "fraction": fraction,
            "arms": {
                arm: {"auroc": float(np.mean(points[arm])), "ci95": lc.helper.interval(macro_draws[budget][arm])}
                for arm in ARMS
            },
        }
    full_minus_smaller = {
        budget: {
            arm: {
                "delta": curves["100"]["arms"][arm]["auroc"] - curves[budget]["arms"][arm]["auroc"],
                "ci95": lc.helper.interval(macro_draws["100"][arm] - macro_draws[budget][arm]),
            }
            for arm in ARMS
        }
        for budget in BUDGETS[:-1]
    }
    flags = {
        "any_incomplete_inner_support_fallback": "fixed_c1" in routes,
        "any_sparse_training_fallback": "prevalence" in routes,
        "any_nonconvergence_fallback": False,
    }
    return curves, subset_hashes, {"full_minus_smaller": full_minus_smaller, "flags": flags}


def _source_hashes(p: Mapping[str, Any]) -> dict[str, Any]:
    cache = p["cache"]
    return {
        "learning_protocol_sha256": p["learning_source"]["protocol_sha256"],
        "learning_aggregate_sha256": p["learning_source"]["aggregate_sha256"],
        "learning_audit_sha256": p["learning_source"]["audit_sha256"],
        "approved_named_fm_protocol_sha256": p["approved_reference"]["protocol_sha256"],
        "approved_named_fm_success_sha256": p["approved_reference"]["success_sha256"],
        "cache_manifest_sha256": cache["manifest_sha256"],
        "cache_current_inputs_sha256": cache["current_inputs_sha256"],
        "cache_row_order_sha256": cache["row_order_sha256"],
        "cache_outer_fold_sha256": cache["outer_fold_sha256"],
        "cache_inner_fold_sha256": list(cache["inner_fold_sha256"]),
        "cache_embedding_sha256": {arm: cache["embedding_files"][arm]["sha256"] for arm in ARMS},
        "cache_artifact_sha256": {arm: cache["artifact_files"][arm]["sha256"] for arm in ARMS},
    }


def validate_result(result: Mapping[str, Any], p: Mapping[str, Any]) -> None:
    keys = {
        "schema", "status", "protocol_sha256", "curve", "full_minus_smaller", "paired_people", "outcomes",
        "full_budget_original_reference_replayed", "same_native_evaluation_support",
        "any_incomplete_inner_support_fallback", "any_sparse_training_fallback", "any_nonconvergence_fallback",
        "patient_level_output_emitted", "automatic_promotion", "source_hashes", "subset_hashes",
    }
    _exact_keys(result, keys, "fm_learning_curve_result_schema_invalid")
    require(result["schema"] == SCHEMA and result["status"] == "completed", "fm_learning_curve_result_identity_invalid")
    require(result["protocol_sha256"] == sha(PROTOCOL) and result["paired_people"] == PATIENT_COUNT and result["outcomes"] == ENDPOINT_COUNT, "fm_learning_curve_result_scope_invalid")
    require(result["full_budget_original_reference_replayed"] is True and result["same_native_evaluation_support"] is True, "fm_learning_curve_result_replay_invalid")
    require(result["patient_level_output_emitted"] is False and result["automatic_promotion"] is False and result["any_nonconvergence_fallback"] is False, "fm_learning_curve_result_privacy_invalid")
    for key in ("any_incomplete_inner_support_fallback", "any_sparse_training_fallback"):
        require(type(result[key]) is bool, "fm_learning_curve_fallback_flag_invalid")
    require(result["source_hashes"] == _source_hashes(p), "fm_learning_curve_source_hashes_invalid")
    expected_subsets = {f"{fold}_{budget}" for fold in range(OUTER_FOLD_COUNT) for budget in BUDGETS}
    require(isinstance(result["subset_hashes"], Mapping) and set(result["subset_hashes"]) == expected_subsets, "fm_learning_curve_subset_hashes_invalid")
    for digest in result["subset_hashes"].values():
        _require_sha(digest, "fm_learning_curve_subset_hash_invalid")
    require(set(result["curve"]) == set(BUDGETS) and set(result["full_minus_smaller"]) == set(BUDGETS[:-1]), "fm_learning_curve_budget_keys_invalid")
    for budget, fraction in zip(BUDGETS, FRACTIONS):
        row = result["curve"][budget]
        _exact_keys(row, {"n", "fraction", "arms"}, "fm_learning_curve_curve_row_invalid")
        require(_finite(row["n"]) and 20 <= row["n"] <= PATIENT_COUNT and row["fraction"] == fraction, "fm_learning_curve_curve_count_invalid")
        require(set(row["arms"]) == set(ARMS), "fm_learning_curve_curve_arm_keys_invalid")
        for metric in row["arms"].values():
            _exact_keys(metric, {"auroc", "ci95"}, "fm_learning_curve_metric_schema_invalid")
            require(_finite(metric["auroc"]) and 0 <= metric["auroc"] <= 1, "fm_learning_curve_auroc_invalid")
            require(isinstance(metric["ci95"], list) and len(metric["ci95"]) == 2 and all(_finite(x) and 0 <= x <= 1 for x in metric["ci95"]) and metric["ci95"][0] <= metric["ci95"][1], "fm_learning_curve_ci_invalid")
    for budget, arm_rows in result["full_minus_smaller"].items():
        require(set(arm_rows) == set(ARMS), "fm_learning_curve_contrast_arm_keys_invalid")
        for arm, metric in arm_rows.items():
            _exact_keys(metric, {"delta", "ci95"}, "fm_learning_curve_contrast_schema_invalid")
            expected = result["curve"]["100"]["arms"][arm]["auroc"] - result["curve"][budget]["arms"][arm]["auroc"]
            require(_finite(metric["delta"]) and -1 <= metric["delta"] <= 1 and abs(metric["delta"] - expected) < 1e-10, "fm_learning_curve_contrast_invalid")
            require(isinstance(metric["ci95"], list) and len(metric["ci95"]) == 2 and all(_finite(x) and -1 <= x <= 1 for x in metric["ci95"]) and metric["ci95"][0] <= metric["ci95"][1], "fm_learning_curve_contrast_ci_invalid")


def run(p: Mapping[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """Execute only fixed heads over an authenticated existing cache."""

    root = Path(root).resolve()
    validate_protocol(p, root, verify_cache=True)
    require(not OUT.exists() and not AUDIT.exists(), "fm_learning_curve_output_already_exists")
    fd = None
    owned = False
    phase = "protocol"
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_WRONLY, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OUT.mkdir(mode=0o700)
        owned = True
        lc.old.native.source.base._atomic_progress(OUT / "progress.json", "cache")
        source, ctx, folds, c0, cm0, eligible, r0, rm, ages, names = _source_context()
        endpoints, _ = _validate_context(p, source, ctx, np.asarray(folds), np.asarray(c0), np.asarray(cm0), np.asarray(eligible), np.asarray(r0), np.asarray(rm), np.asarray(ages), names)
        lc.old.native.source.base._atomic_progress(OUT / "progress.json", "inner_folds")
        inner = _inner_assignments(p, source, ctx, np.asarray(folds))
        cache = load_fm_cache(
            p["cache"],
            expected_current_inputs_sha256=p["cache"]["current_inputs_sha256"],
            expected_row_order_sha256=p["cache"]["row_order_sha256"],
            expected_outer_fold_sha256=p["learning_source"]["outer_fold_sha256"],
            expected_inner_fold_sha256=p["learning_source"]["inner_fold_sha256"],
        )
        curves, subset_hashes, extra = _fit_curve(p, source, ctx, np.asarray(folds), np.asarray(c0), np.asarray(cm0), np.asarray(eligible), np.asarray(r0), np.asarray(rm), np.asarray(ages), names, endpoints, cache, inner)
        result = {
            "schema": SCHEMA,
            "status": "completed",
            "protocol_sha256": sha(PROTOCOL),
            "curve": curves,
            "full_minus_smaller": extra["full_minus_smaller"],
            "paired_people": PATIENT_COUNT,
            "outcomes": ENDPOINT_COUNT,
            "full_budget_original_reference_replayed": True,
            "same_native_evaluation_support": True,
            **extra["flags"],
            "patient_level_output_emitted": False,
            "automatic_promotion": False,
            "source_hashes": _source_hashes(p),
            "subset_hashes": subset_hashes,
        }
        phase = "writing"
        validate_result(result, p)
        exclusive_json(OUT / "aggregate.json", result)
        exclusive_json(OUT / "manifest.json", {
            "schema": "bran-fm-learning-curve-manifest-v1",
            "protocol_sha256": sha(PROTOCOL),
            "aggregate_sha256": sha(OUT / "aggregate.json"),
            "code_sha256": p["code_sha256"],
            "source_hashes": _source_hashes(p),
            "subset_hashes": subset_hashes,
            "patient_level_output_emitted": False,
            "checkpoint_write": False,
        })
        lc.old.native.source.base._atomic_completed(OUT / "progress.json")
        return result
    except Exception as exc:
        if owned and not (OUT / "aggregate.json").exists() and not (OUT / "failure.json").exists():
            exclusive_json(OUT / "failure.json", {
                "schema": "bran-fm-learning-curve-failure-v1",
                "status": "execution_failed",
                "phase": phase,
                "error_class": type(exc).__name__ if type(exc) in (ValueError, TypeError, KeyError, RuntimeError, OSError, ImportError) else "other_execution_error",
                "patient_level_output_emitted": False,
                "checkpoint_write": False,
            })
        raise
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _validate_failure(path: Path) -> dict[str, Any]:
    value = _json_load(path)
    _exact_keys(value, {"schema", "status", "phase", "error_class", "patient_level_output_emitted", "checkpoint_write"}, "fm_failure_schema_invalid")
    require(value["schema"] == "bran-fm-learning-curve-failure-v1" and value["status"] == "execution_failed" and value["phase"] in PHASES, "fm_failure_identity_invalid")
    require(isinstance(value["error_class"], str) and value["error_class"].isidentifier() and len(value["error_class"]) < 90, "fm_failure_error_class_invalid")
    require(value["patient_level_output_emitted"] is False and value["checkpoint_write"] is False, "fm_failure_privacy_invalid")
    return dict(value)


def audit(root: Path = ROOT) -> dict[str, Any]:
    """Authenticate the exclusive terminal and independently rebuild split pins."""

    root = Path(root).resolve()
    p = json.loads((root / PROTOCOL.name).read_text(encoding="utf-8"))
    validate_protocol(p, root, verify_cache=True)
    out = root / PATHS["output"]
    require(out.is_dir(), "fm_audit_output_missing")
    failure = out / "failure.json"
    aggregate = out / "aggregate.json"
    manifest = out / "manifest.json"
    require(not (aggregate.exists() and failure.exists()), "fm_audit_terminal_not_exclusive")
    if failure.exists():
        value = _validate_failure(failure)
        return {"schema": "bran-fm-learning-curve-audit-v1", "status": "authenticated_execution_failure", "failure_sha256": sha(failure), "phase": value["phase"], "error_class": value["error_class"], "patient_level_output_emitted": False}
    require(aggregate.is_file() and manifest.is_file() and (out / "progress.json").is_file(), "fm_audit_terminal_artifacts_missing")
    require({path.name for path in out.iterdir()} == {"aggregate.json", "manifest.json", "progress.json"}, "fm_audit_output_not_exclusive")
    require(json.loads((out / "progress.json").read_text(encoding="utf-8")) == {"status": "completed", "phase": "completed"}, "fm_audit_progress_invalid")
    result = json.loads(aggregate.read_text(encoding="utf-8")); validate_result(result, p)
    m = _json_load(manifest)
    _exact_keys(m, {"schema", "protocol_sha256", "aggregate_sha256", "code_sha256", "source_hashes", "subset_hashes", "patient_level_output_emitted", "checkpoint_write"}, "fm_audit_manifest_schema_invalid")
    require(m["schema"] == "bran-fm-learning-curve-manifest-v1" and m["protocol_sha256"] == sha(root / PROTOCOL.name) and m["aggregate_sha256"] == sha(aggregate), "fm_audit_manifest_identity_invalid")
    require(m["code_sha256"] == p["code_sha256"] and m["source_hashes"] == result["source_hashes"] and m["subset_hashes"] == result["subset_hashes"], "fm_audit_manifest_bindings_invalid")
    require(m["patient_level_output_emitted"] is False and m["checkpoint_write"] is False, "fm_audit_manifest_privacy_invalid")
    source, ctx, folds, *_ = _source_context()
    _, reconstructed_counts = _subset_hashes(np.asarray(folds))
    expected_subsets, _ = _subset_hashes(np.asarray(folds))
    require(result["subset_hashes"] == expected_subsets, "fm_audit_subset_identity_invalid")
    for budget in BUDGETS:
        require(result["curve"][budget]["n"] == float(np.mean(reconstructed_counts[budget])), "fm_audit_training_count_invalid")
    _inner_assignments(p, source, ctx, np.asarray(folds))
    require(result["source_hashes"] == _source_hashes(p), "fm_audit_source_hashes_invalid")
    return {
        "schema": "bran-fm-learning-curve-audit-v1",
        "status": "authenticated",
        "protocol_sha256": sha(root / PROTOCOL.name),
        "aggregate_sha256": sha(aggregate),
        "manifest_sha256": sha(manifest),
        "code_hashes_verified": True,
        "source_hashes_verified": True,
        "subset_identities_verified": True,
        "inner_fold_identities_verified": True,
        "paired_people": PATIENT_COUNT,
        "outcomes": ENDPOINT_COUNT,
        "full_budget_original_reference_replayed": True,
        "patient_level_output_emitted": False,
        "checkpoint_write": False,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--audit", action="store_true")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--cache-manifest", type=Path)
    args = parser.parse_args(argv)
    operation = "prepare" if args.prepare else "run" if args.run else "audit"
    ok = False
    status = "failed"
    owned = False
    target = AUDIT if args.audit else OUT
    start = time.monotonic()
    os.umask(0o077)
    with _quiet():
        try:
            if args.prepare:
                require(not PROTOCOL.exists() and not OUT.exists() and not AUDIT.exists(), "fm_learning_curve_existing_artifact")
                prepared = prepare(ROOT, args.cache_manifest)
                exclusive_json(PROTOCOL, prepared)
                status = "protocol_prepared"
            elif args.run:
                require(isinstance(args.protocol_sha256, str) and sha(PROTOCOL) == args.protocol_sha256, "fm_learning_curve_protocol_sha_invalid")
                p = json.loads(PROTOCOL.read_text(encoding="utf-8")); validate_protocol(p, ROOT, verify_cache=True)
                run(p, ROOT)
                status = "completed"
            else:
                require(isinstance(args.protocol_sha256, str) and sha(PROTOCOL) == args.protocol_sha256, "fm_learning_curve_protocol_sha_invalid")
                AUDIT.mkdir(mode=0o700)
                owned = True
                receipt = audit(ROOT)
                exclusive_json(AUDIT / "audit.json", receipt)
                status = "audited"
            ok = True
        except FMCacheUnavailable:
            status = "blocked_cache_missing"
        except Exception as exc:
            if owned and not (target / "failure.json").exists():
                exclusive_json(target / "failure.json", {"schema": "bran-fm-learning-curve-failure-v1", "status": "execution_failed", "phase": "protocol", "error_class": type(exc).__name__ if type(exc) in (ValueError, TypeError, KeyError, RuntimeError, OSError, ImportError) else "other_execution_error", "patient_level_output_emitted": False, "checkpoint_write": False})
    print(json.dumps({"operation": operation, "status": status, "patient_level_output_emitted": False, "checkpoint_write": False, "elapsed_seconds": round(time.monotonic() - start, 1)}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
