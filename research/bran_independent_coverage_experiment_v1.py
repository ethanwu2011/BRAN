"""Frozen, local-only BRAN CGM/ECG coverage preflight for one diabetes subgroup.

This driver has no model fitting, image-pixel decode, note selection, or hosted input.
The established canonical loader may read its locally cached structured vectors; this
driver never opens or decodes image pixels.  Manifest identifiers and masks remain
local transient variables; terminal artifacts are strictly aggregate-only or fully
suppressed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from datetime import datetime, timezone
from typing import Any

import numpy as np

import bran_innovation_experiment_v1_attempt2 as prior
from bran_independent_coverage_io_v1 import (
    CELL_FLOOR,
    MODALITY_FILES,
    SOURCE_COLUMNS,
    parse_manifest,
    validate_coverage,
)
from bran_independent_coverage_v1 import summarize_coverage


ROOT = Path(__file__).resolve().parent
SCHEMA = "bran-independent-coverage-experiment-v1"
PROTOCOL = "BRAN_INDEPENDENT_COVERAGE_PROTOCOL_V1.json"
OUTDIR = "validation_results/BRAN_INDEPENDENT_COVERAGE_V1"
PATHS = {key: OUTDIR + "/" + name for key, name in {
    "success": "SUCCESS.json", "failure": "FAILURE.json", "progress": "progress.json", "lock": "run.lock"
}.items()}
PRIOR_PROTOCOL = "BRAN_INNOVATION_READOUT_PROTOCOL_V1_ATTEMPT2.json"
PRIOR_PROTOCOL_SHA = "961ec1d2283ac6ea8275ff275903f63df6563444604dc89b8a9051e3625743e5"
EXPECTED_PRIOR_COMPONENTS = 99
COVERAGE_FILES = {
    "bran_independent_coverage_v1.py",
    "test_bran_independent_coverage_v1.py",
    "bran_independent_coverage_io_v1.py",
    "test_bran_independent_coverage_io_v1.py",
}
NEW_FILES = COVERAGE_FILES | {
    "bran_independent_coverage_experiment_v1.py",
    "test_bran_independent_coverage_experiment_v1.py",
}
INPUT_CONTRACT = {
    "id_column": "person_id",
    "manifest_files": MODALITY_FILES,
    "source_columns": SOURCE_COLUMNS,
    "source_values_local_only": True,
}
PARAMETERS = {
    "canonical_patient_count": 1928,
    "canonical_endpoint_count": 26,
    "subgroup": "observed_mhterm_dm2_equals_1",
    "outer_folds": 5,
    "inner_folds_authenticated": 5,
    "cell_floor": CELL_FLOOR,
    "coverage": "manifest_finite_value_preflight_only_no_clinical_thresholds",
}
PRIVACY = {
    "patient_processing_local_only": True,
    "manifest_ids_masks_and_values_serialized": False,
    "patient_rows_ids_notes_images_embeddings_predictions_or_models_serialized": False,
    "hosted_inference_used": False,
    "pixel_decodes_or_training_performed": False,
    "population_wide_coverage_emitted": False,
}
PHASES = {"protocol", "validated", "context", "subgroup_gate", "manifest", "aggregate", "writing", "completed"}
require = prior.base.require
exact_keys = prior.base.exact_keys
write_x = prior.base.write_x


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_paths(data_roots: Any) -> dict[str, Path]:
    require(isinstance(data_roots, dict) and set(data_roots) >= {"dataset_root", "clinical_project_root"})
    dataset = Path(data_roots["dataset_root"])
    return {modality: dataset / relative for modality, relative in MODALITY_FILES.items()}


def _manifest_hashes(data_roots: Any) -> dict[str, str]:
    return {modality: sha(path) for modality, path in _manifest_paths(data_roots).items()}


def _valid_hashes(value: Any, names: set[str] | tuple[str, ...]) -> bool:
    if not isinstance(value, dict) or set(value) != set(names):
        return False
    return all(type(digest) is str and len(digest) == 64 and set(digest) <= set("0123456789abcdef") for digest in value.values())


def _prior_protocol(root: Path) -> dict[str, Any]:
    require(sha(root / PRIOR_PROTOCOL) == PRIOR_PROTOCOL_SHA)
    previous = prior.validate_protocol(root)
    require(isinstance(previous, dict) and len(previous.get("expected_hashes", {})) == EXPECTED_PRIOR_COMPONENTS)
    return previous


def _expected_names(previous: dict[str, Any]) -> set[str]:
    names = set(previous["expected_hashes"])
    require(len(names) == EXPECTED_PRIOR_COMPONENTS and not (names & NEW_FILES))
    return names | NEW_FILES


def freeze(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    require(not (root / PROTOCOL).exists())
    previous = _prior_protocol(root)
    names = _expected_names(previous)
    require(not any((root / value).exists() for value in PATHS.values()))
    manifest_hashes = _manifest_hashes(previous["data_roots"])
    require(_valid_hashes(manifest_hashes, set(MODALITY_FILES)))
    protocol = {
        "schema": SCHEMA,
        "status": "frozen_before_execution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": PARAMETERS,
        "privacy": PRIVACY,
        "paths": PATHS,
        "authentication": previous["authentication"],
        "data_roots": previous["data_roots"],
        "prior_protocol_sha256": PRIOR_PROTOCOL_SHA,
        "input_contract": INPUT_CONTRACT,
        "manifest_hashes": manifest_hashes,
        "expected_hashes": {name: sha(root / name) for name in sorted(names)},
    }
    write_x(root / PROTOCOL, protocol)
    validate_protocol(root)
    return {"status": "frozen", "protocol_sha256": sha(root / PROTOCOL), "components": len(names)}


def validate_protocol(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    previous = _prior_protocol(root)
    protocol = json.loads((root / PROTOCOL).read_text())
    exact_keys(protocol, {
        "schema", "status", "created_utc", "parameters", "privacy", "paths", "authentication", "data_roots",
        "prior_protocol_sha256", "input_contract", "manifest_hashes", "expected_hashes",
    })
    require(protocol["schema"] == SCHEMA and protocol["status"] == "frozen_before_execution")
    require(protocol["parameters"] == PARAMETERS and protocol["privacy"] == PRIVACY and protocol["paths"] == PATHS)
    require(protocol["authentication"] == previous["authentication"] and protocol["data_roots"] == previous["data_roots"])
    require(protocol["prior_protocol_sha256"] == PRIOR_PROTOCOL_SHA and protocol["input_contract"] == INPUT_CONTRACT)
    require(_valid_hashes(protocol["manifest_hashes"], set(MODALITY_FILES)))
    require(protocol["manifest_hashes"] == _manifest_hashes(protocol["data_roots"]))
    names = _expected_names(previous)
    exact_keys(protocol["expected_hashes"], names)
    for name, digest in protocol["expected_hashes"].items():
        path = (root / name).resolve()
        require(path.parent == root.resolve() and type(digest) is str and len(digest) == 64 and sha(path) == digest)
    return protocol


def _safe_cell(value: Any) -> bool:
    return type(value) is int and (value == 0 or value >= CELL_FLOOR)


def _subgroup_gate(canonical_fold_sizes: tuple[int, ...], subgroup_fold_sizes: tuple[int, ...]) -> bool:
    if len(canonical_fold_sizes) != 5 or len(subgroup_fold_sizes) != 5:
        return False
    if any(type(value) is not int or value < 0 for value in (*canonical_fold_sizes, *subgroup_fold_sizes)):
        return False
    complements = tuple(canonical - subgroup for canonical, subgroup in zip(canonical_fold_sizes, subgroup_fold_sizes, strict=True))
    return (
        all(subgroup <= canonical for canonical, subgroup in zip(canonical_fold_sizes, subgroup_fold_sizes, strict=True))
        and all(size >= CELL_FLOOR for size in subgroup_fold_sizes)
        and _safe_cell(sum(subgroup_fold_sizes))
        and _safe_cell(sum(complements))
        and all(_safe_cell(size) for size in complements)
    )


def _aggregate_analysis(coverage: Any, subgroup_count: int, fold_sizes: tuple[int, ...]) -> dict[str, Any]:
    require(validate_coverage(coverage, subgroup_count, fold_sizes) is coverage)
    return {
        "subgroup": "observed_mhterm_dm2_equals_1",
        "subgroup_count": subgroup_count,
        "fold_sizes": list(fold_sizes),
        "coverage": coverage,
    }


def _validate_analysis(value: Any, canonical_fold_sizes: tuple[int, ...]) -> bool:
    if not isinstance(value, dict) or set(value) != {"subgroup", "subgroup_count", "fold_sizes", "coverage"}:
        return False
    if value["subgroup"] != "observed_mhterm_dm2_equals_1" or type(value["subgroup_count"]) is not int:
        return False
    if not isinstance(value["fold_sizes"], list) or len(value["fold_sizes"]) != 5:
        return False
    fold_sizes = tuple(value["fold_sizes"])
    basic_gate = (
        all(type(size) is int and size >= CELL_FLOOR for size in fold_sizes)
        and _safe_cell(value["subgroup_count"])
        and sum(fold_sizes) == value["subgroup_count"]
    )
    return basic_gate and _subgroup_gate(canonical_fold_sizes, fold_sizes) and validate_coverage(value["coverage"], value["subgroup_count"], fold_sizes) is value["coverage"]


def validate_report(report: Any, protocol: dict[str, Any], protocol_sha256: str, canonical_fold_sizes: tuple[int, ...] | None = None) -> bool:
    """Validate either the safe aggregate or a fully count-suppressed terminal report."""
    if not isinstance(report, dict):
        return False
    common = {
        "schema", "status", "protocol_sha256", "code_hashes", "source_hashes", "support_receipt_sha256",
        "fold_hashes", "parameters", "privacy", "manifest_hashes",
    }
    auth = protocol["authentication"]
    if report.get("status") == "completed_aggregate_only":
        if set(report) != common | {"scope", "analysis"}:
            return False
        scope = report["scope"]
        if not isinstance(scope, dict) or set(scope) != {"canonical_patient_count", "canonical_endpoint_count", "canonical_fold_sizes", "official_test_loaded"}:
            return False
        if scope["canonical_patient_count"] != 1928 or scope["canonical_endpoint_count"] != 26 or scope["official_test_loaded"] is not False:
            return False
        if not isinstance(scope["canonical_fold_sizes"], list) or len(scope["canonical_fold_sizes"]) != 5:
            return False
        frozen_canonical_folds = tuple(scope["canonical_fold_sizes"])
        if any(type(size) is not int or size < CELL_FLOOR for size in frozen_canonical_folds) or sum(frozen_canonical_folds) != 1928:
            return False
        if canonical_fold_sizes is not None and tuple(canonical_fold_sizes) != frozen_canonical_folds:
            return False
        analysis_ok = _validate_analysis(report["analysis"], frozen_canonical_folds)
    elif report.get("status") == "suppressed_subgroup_small_cell":
        if set(report) != common | {"analysis"} or report["analysis"] != {"status": "suppressed_subgroup_small_cell", "coverage": None}:
            return False
        analysis_ok = True
    else:
        return False
    return bool(
        analysis_ok
        and report["schema"] == SCHEMA
        and report["protocol_sha256"] == protocol_sha256
        and report["code_hashes"] == protocol["expected_hashes"]
        and report["source_hashes"] == auth["canonical_source_hashes"]
        and report["support_receipt_sha256"] == auth["support_receipt_sha256"]
        and report["fold_hashes"] == {"outer": auth["outer_fold_sha256"], "inner": auth["inner_fold_sha256"]}
        and report["parameters"] == PARAMETERS
        and report["privacy"] == PRIVACY
        and report["manifest_hashes"] == protocol["manifest_hashes"]
    )


def _local_canonical_ids(values: Any) -> tuple[str, ...]:
    try:
        identifiers = tuple(str(value).strip() for value in values)
    except TypeError as error:
        raise RuntimeError("canonical identifier contract failed") from error
    require(len(identifiers) == 1928 and len(set(identifiers)) == 1928)
    require(all(identifier and identifier.isascii() and identifier.isdecimal() and (identifier == "0" or not identifier.startswith("0")) for identifier in identifiers))
    return identifiers


def _write_progress(path: Path, phase: str) -> None:
    require(phase in PHASES)
    temporary = path.with_suffix(".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps({"status": "running", "phase": phase}, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _failure(root: Path, protocol: dict[str, Any], paths: dict[str, Path], phase: str, error: Exception) -> dict[str, Any]:
    frames = [
        {"file": Path(item.filename).name, "line": item.lineno}
        for item in traceback.extract_tb(error.__traceback__)
        if Path(item.filename).resolve().parent == root.resolve() and Path(item.filename).name in protocol.get("expected_hashes", {})
    ][:8]
    failure = {
        "schema": SCHEMA,
        "status": "failed",
        "phase": phase,
        "error_class": type(error).__name__,
        "bound_code_frames": frames,
        "exception_text_serialized": False,
        "patient_content_serialized": False,
    }
    if not paths["success"].exists() and not paths["failure"].exists():
        write_x(paths["failure"], failure)
    return {"status": "execution_failed", "phase": phase, "error_class": type(error).__name__, "exception_contents_emitted": False}


def run(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    protocol: dict[str, Any] = {}
    fd = None
    quiet = None
    phase = "protocol"
    paths = {key: root / value for key, value in PATHS.items()}
    try:
        protocol = validate_protocol(root)
        require(not any(paths[key].exists() for key in ("success", "failure", "progress")))
        fd = prior.base.paired.base._acquire_lock(paths["lock"])
        _write_progress(paths["progress"], "validated")
        quiet = prior.base.paired.base._quiet_sensitive_block()
        quiet.__enter__()
        phase = "context"
        from patient_atlas_v6_2_expanded_endpoint_evaluation import (
            EXACT_INNER_FOLD_ASSIGNMENT_SHA256,
            EXACT_OUTER_FOLD_HASH,
            FROZEN_SUPPORT_RECEIPT_NAME,
            load_eligible_support_receipt,
            validate_support_against_observed,
        )
        from run_patient_atlas_v6_2_expanded_endpoint_evaluation import _load_actual_v6_2_context

        support = load_eligible_support_receipt(root / FROZEN_SUPPORT_RECEIPT_NAME, project_root=root)
        context = _load_actual_v6_2_context(root=root, support=support, **{key: Path(value) for key, value in protocol["data_roots"].items()})
        outer = np.asarray(context["outer_assignment"])
        auth = protocol["authentication"]
        require(len(outer) == 1928 and outer.dtype.kind in "iu" and set(outer.tolist()) == set(range(5)))
        require(dict(context["source_hashes"]) == auth["canonical_source_hashes"] and support.receipt_sha256 == auth["support_receipt_sha256"])
        require(len(support.eligible_sources) == 26)
        require(auth["outer_fold_sha256"] == EXACT_OUTER_FOLD_HASH and len(auth["inner_fold_sha256"]) == 5)
        validate_support_against_observed(support, context["labels_by_source"], context["observed_by_source"], outer)
        for fold in range(5):
            _, digest = prior.base.paired.base._inner_context(context, np.flatnonzero(outer != fold), fold)
            require(digest == auth["inner_fold_sha256"][fold] == EXACT_INNER_FOLD_ASSIGNMENT_SHA256[fold])
        canonical_ids = _local_canonical_ids(context["raw_cohort"].patient_ids)
        observed = context["observed_by_source"]["mhterm_dm2"]
        labels = context["labels_by_source"]["mhterm_dm2"]
        require(isinstance(observed, np.ndarray) and observed.dtype == np.dtype(bool) and observed.shape == outer.shape)
        require(isinstance(labels, np.ndarray) and labels.shape == outer.shape and labels.dtype.kind in "biuf")
        require(np.all(np.isfinite(labels[observed])) and np.all((labels[observed] == 0) | (labels[observed] == 1)))
        selected = observed & (labels == 1)
        subset_ids = tuple(identifier for identifier, keep in zip(canonical_ids, selected, strict=True) if bool(keep))
        subset_outer = tuple(int(fold) for fold, keep in zip(outer.tolist(), selected, strict=True) if bool(keep))
        canonical_fold_sizes = tuple(int(np.sum(outer == fold)) for fold in range(5))
        subgroup_fold_sizes = tuple(subset_outer.count(fold) for fold in range(5))
        phase = "subgroup_gate"
        if not _subgroup_gate(canonical_fold_sizes, subgroup_fold_sizes):
            report = _report_base(protocol, root) | {"status": "suppressed_subgroup_small_cell", "analysis": {"status": "suppressed_subgroup_small_cell", "coverage": None}}
        else:
            phase = "manifest"
            before = _manifest_hashes(protocol["data_roots"])
            require(before == protocol["manifest_hashes"])
            source_ids: dict[str, tuple[str, ...]] = {}
            finite_masks: dict[str, dict[str, tuple[bool, ...]]] = {}
            for modality, path in _manifest_paths(protocol["data_roots"]).items():
                with path.open("r", encoding="utf-8", newline="") as handle:
                    source_ids[modality], finite_masks[modality] = parse_manifest(handle, modality)
            phase = "aggregate"
            coverage = summarize_coverage(subset_ids, subset_outer, source_ids, finite_masks)
            require(_manifest_hashes(protocol["data_roots"]) == before == protocol["manifest_hashes"])
            report = _report_base(protocol, root) | {
                "status": "completed_aggregate_only",
                "scope": {"canonical_patient_count": 1928, "canonical_endpoint_count": 26, "canonical_fold_sizes": list(canonical_fold_sizes), "official_test_loaded": False},
                "analysis": _aggregate_analysis(coverage, len(subset_ids), subgroup_fold_sizes),
            }
        phase = "writing"
        protocol = validate_protocol(root)
        require(validate_report(report, protocol, sha(root / PROTOCOL), canonical_fold_sizes))
        write_x(paths["success"], report)
        _write_progress(paths["progress"], "completed")
        return {"status": report["status"], "artifact_sha256": sha(paths["success"])}
    except Exception as error:
        return _failure(root, protocol, paths, phase, error) if fd is not None else {"status": "execution_failed", "phase": phase, "error_class": type(error).__name__, "exception_contents_emitted": False}
    finally:
        if quiet is not None:
            quiet.__exit__(None, None, None)
        if fd is not None:
            os.close(fd)
            try:
                paths["lock"].unlink()
            except FileNotFoundError:
                pass


def _report_base(protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    auth = protocol["authentication"]
    return {
        "schema": SCHEMA,
        "protocol_sha256": sha(root / PROTOCOL),
        "code_hashes": protocol["expected_hashes"],
        "source_hashes": auth["canonical_source_hashes"],
        "support_receipt_sha256": auth["support_receipt_sha256"],
        "fold_hashes": {"outer": auth["outer_fold_sha256"], "inner": auth["inner_fold_sha256"]},
        "parameters": PARAMETERS,
        "privacy": PRIVACY,
        "manifest_hashes": protocol["manifest_hashes"],
    }


def audit(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    protocol = validate_protocol(root)
    paths = {key: root / value for key, value in PATHS.items()}
    require(not (paths["success"].exists() and paths["failure"].exists()))
    if paths["failure"].exists():
        failure = json.loads(paths["failure"].read_text())
        exact_keys(failure, {"schema", "status", "phase", "error_class", "bound_code_frames", "exception_text_serialized", "patient_content_serialized"})
        require(failure["schema"] == SCHEMA and failure["status"] == "failed" and failure["phase"] in PHASES)
        require(failure["exception_text_serialized"] is False and failure["patient_content_serialized"] is False)
        require(type(failure["error_class"]) is str and failure["error_class"].isidentifier() and len(failure["error_class"]) < 90)
        require(type(failure["bound_code_frames"]) is list and len(failure["bound_code_frames"]) <= 8)
        for frame in failure["bound_code_frames"]:
            exact_keys(frame, {"file", "line"})
            require(frame["file"] in protocol["expected_hashes"] and type(frame["line"]) is int and 0 < frame["line"] < 100000)
        return {"status": "authenticated_execution_failure", "artifact_sha256": sha(paths["failure"]), "phase": failure["phase"], "error_class": failure["error_class"]}
    if paths["success"].exists():
        report = json.loads(paths["success"].read_text())
        require(validate_report(report, protocol, sha(root / PROTOCOL)))
        result = {
            "status": "authenticated_success",
            "artifact_sha256": sha(paths["success"]),
            "protocol_sha256": sha(root / PROTOCOL),
            "terminal_status": report["status"],
            "components": len(protocol["expected_hashes"]),
            "analysis": report["analysis"],
        }
        if report["status"] == "completed_aggregate_only":
            result["scope"] = report["scope"]
            result["folds_authenticated"] = True
        return result
    result = {"status": "no_terminal_artifact", "lock_exists": paths["lock"].exists()}
    if paths["progress"].exists():
        progress = json.loads(paths["progress"].read_text())
        require(isinstance(progress, dict) and set(progress) == {"status", "phase"} and progress["status"] in {"running", "completed"} and progress["phase"] in PHASES)
        result["progress"] = progress
    return result


if __name__ == "__main__":
    try:
        print(json.dumps({"freeze": freeze, "run": run, "audit": audit}[sys.argv[1]](), sort_keys=True, allow_nan=False))
    except Exception as error:
        print(json.dumps({"status": "blocked_without_disclosure", "error_class": type(error).__name__, "contents_emitted": False}))
        raise SystemExit(1)
