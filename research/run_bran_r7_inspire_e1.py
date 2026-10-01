"""Exclusive local R7 fold-0 external adaptation on the frozen INSPIRE frame.

This runner is a small versioned envelope around the already audited, pure
INSPIRE input/state and three-arm adaptation kernels.  It authenticates the
old V5 terminal and the R7/P1 checkpoint manifests before any source rows are
loaded, then changes only the 192-coordinate state.  No model fitting or
cohort construction happens in this module.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path
import pickle
import sys

import numpy as np

import bran_inspire_adaptation_v1 as kernel
import bran_inspire_v5_state_v1 as state_kernel
import run_bran_inspire_adaptation_v1 as base
import run_bran_r7_fixed_state_p1 as p1
import run_bran_robust_clinical_r7 as r7


ROOT = Path(__file__).resolve().parent
ATTEMPT = 1
OUT = ROOT / f"BRAN_R7_INSPIRE_E1_ATTEMPT{ATTEMPT}"
PRIVATE = ROOT / "private_artifacts" / f"bran_r7_inspire_e1_attempt{ATTEMPT}"
LOCK = base.LOCK
ERROR = "bran_r7_inspire_e1_failed"
FRAME = "R7_attempt1_fold0"
PROTOCOL_SCHEMA = "bran-r7-inspire-e1-protocol-v1"
AGGREGATE_SCHEMA = "bran-r7-inspire-e1-aggregate-v1"
PRIVATE_FILES = ("states.npz", "readouts.pkl")
PHASES = (
    "authentication",
    "source_load",
    "state_inference",
    "checkpoint_replay",
    "fixed_readouts_and_bootstrap",
    "private_write_and_replay",
    "post_authentication",
    "completed",
)

# The design and launcher are root-owned protocol files.  Hashing this closed
# list occurs before source arrays are materialized.
CODE = (
    "BRAN_R7_INSPIRE_E1_DESIGN.md",
    "run_bran_r7_inspire_e1.py",
    "test_run_bran_r7_inspire_e1.py",
    "run_bran_r7_inspire_e1_attempt1.sh",
    "bran_inspire_adaptation_v1.py",
    "test_bran_inspire_adaptation_v1.py",
    "bran_inspire_v5_state_v1.py",
    "test_bran_inspire_v5_state_v1.py",
    "bran_inspire_input_adapter_v1.py",
    "test_bran_inspire_input_adapter_v1.py",
    "run_bran_inspire_adaptation_v1.py",
    "run_bran_r7_fixed_state_p1.py",
    "run_bran_robust_clinical_r7.py",
    "bran_robust_clinical_r7.py",
    "bran_multisource_batches_v2.py",
    "bran_multisource_profiles_v3.py",
    "bran_multisource_inference_v2.py",
    "bran_multisource_age_v2.py",
    "bran_clinical_semantics_v1.py",
    "bran_knhanes_input_kernel_v1.py",
)


def require(ok: bool) -> None:
    if not ok:
        raise ValueError(ERROR) from None


def paths() -> tuple[Path, Path]:
    return OUT, PRIVATE


def _sha(path: Path) -> str:
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _write_json(path: Path, value: dict) -> None:
    require(not path.exists() and not path.is_symlink())
    temporary = path.with_name(path.name + ".next")
    require(not temporary.exists() and not temporary.is_symlink())
    with temporary.open("x", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(value, handle, sort_keys=True, separators=(",", ":"),
                  allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _replace_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".next")
    require(not temporary.exists() and not temporary.is_symlink())
    with temporary.open("x", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(value, handle, sort_keys=True, separators=(",", ":"),
                  allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
            and path.stat().st_size <= 5_000_000)

    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result)
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=unique,
                          parse_constant=lambda _value: require(False))
    require(type(value) is dict)
    return value


def code_hashes() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in sorted(set(CODE))}


def runtime_versions() -> dict[str, str]:
    import joblib
    import scipy
    import sklearn
    import torch
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "torch": torch.__version__,
        "executable": sys.executable,
    }


def progress(out: Path, state: dict[str, object], phase: str) -> None:
    require(phase in PHASES)
    state["phase"] = phase
    _replace_json(out / "progress.json", {
        "phase": phase,
        "pid": int(os.getpid()),
        "patient_level_output_emitted": False,
    }) if (out / "progress.json").exists() else _write_json(out / "progress.json", {
        "phase": phase,
        "pid": int(os.getpid()),
        "patient_level_output_emitted": False,
    })


def _valid_binding(binding: object, role: str, fold: int) -> None:
    require(type(binding) is dict and binding.get("role") == role
            and binding.get("fold") == fold)
    for key in ("outer_fold_sha256", "inner_fold_sha256", "transform_sha256"):
        require(_hex(binding.get(key)))


def _valid_manifest(manifest: object, role: str) -> None:
    require(type(manifest) is dict and manifest.get("component_role") == role
            and type(manifest.get("folds")) is list and len(manifest["folds"]) == 5)
    for fold, item in enumerate(manifest["folds"]):
        require(type(item) is dict and set(item) == {"fold", "checkpoint_sha256", "binding"}
                and item["fold"] == fold and _hex(item["checkpoint_sha256"])
                and isinstance(item["binding"], dict))
        _valid_binding(item["binding"], role, fold)


def _base_audit_shape(value: object) -> None:
    require(type(value) is dict
            and value.get("status") in {"authenticated", "aggregate_terminal_authenticated"}
            and value.get("patient_level_output_emitted") is False)
    for key in ("protocol_sha256", "aggregate_sha256"):
        require(_hex(value.get(key)))


def make_bindings(base_receipt: dict, base_audit: dict, p1_protocol: dict,
                  p1_receipt: dict, r7_protocol: dict, r7_items: dict,
                  r7_receipt: dict) -> dict:
    """Validate and retain only public source/checkpoint provenance."""
    try:
        require(type(base_receipt) is dict and base_receipt.get("inspire_encoder_exposed") is False)
        selected = base_receipt.get("selected_frame")
        require(type(selected) is dict and selected.get("fold") == 0
                and _hex(selected.get("checkpoint_sha256")))
        require(type(base_receipt.get("checkpoint_binding")) is dict)
        _base_audit_shape(base_audit)
        require(type(p1_receipt) is dict and type(r7_receipt) is dict)
        require(type(p1_protocol) is dict and type(r7_protocol) is dict)
        require(p1_protocol.get("schema") == "bran-r7-fixed-state-p1-protocol"
                and p1_protocol.get("status") == "frozen_before_readouts")
        require(r7_protocol.get("stage") == "fit")
        v5_source = p1_protocol.get("v5_source_binding")
        r7_source = p1_protocol.get("r7_source_binding")
        require(type(v5_source) is dict and v5_source == r7_source
                and r7_protocol.get("source_binding") == r7_source)
        v5_manifest = p1_protocol.get("v5_checkpoint_manifest")
        r7_manifest = p1_protocol.get("r7_checkpoint_manifest")
        _valid_manifest(v5_manifest, "M")
        _valid_manifest(r7_manifest, "R")
        v5 = v5_manifest["folds"][0]
        r7_fold = r7_manifest["folds"][0]
        require(v5["checkpoint_sha256"] == selected["checkpoint_sha256"]
                and v5["binding"] == base_receipt["checkpoint_binding"])
        require(_hex(r7_fold["binding"].get("initial_checkpoint_sha256"))
                and r7_fold["binding"]["initial_checkpoint_sha256"] == v5["checkpoint_sha256"]
                and r7_fold["binding"]["transform_sha256"] == v5["binding"]["transform_sha256"])
        require(type(r7_items) is dict and ("R", 0) in r7_items)
        item = r7_items[("R", 0)]
        require(type(item) is dict and item.get("fold") == 0 and item.get("role") == "R"
                and item.get("binding") == r7_fold["binding"]
                and item.get("checkpoint_sha256") == r7_fold["checkpoint_sha256"]
                and item.get("checkpoint_reload_exact") is True)
        return {
            "schema": "bran-r7-inspire-e1-bindings-v1",
            "base_receipt": base_receipt,
            "base_audit": base_audit,
            "p1_receipt": p1_receipt,
            "r7_receipt": r7_receipt,
            "source_binding": v5_source,
            "v5_fold0": v5,
            "r7_fold0": r7_fold,
            "frame": FRAME,
        }
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def authenticate_inputs() -> dict:
    """Authenticate old V5/source and R7/P1 receipts without loading rows."""
    try:
        # ``audit`` is intentionally retained alongside the receipt check: it
        # authenticates the completed V5 terminal and its aggregate closure.
        base_audit = base.audit(ATTEMPT)
        base_out, _base_private = base.paths(ATTEMPT)
        base_protocol = base.read(base_out / "protocol.json")
        require(_sha(base_out / "protocol.json") == base_audit["protocol_sha256"]
                and type(base_protocol) is dict
                and base_protocol.get("schema") == "bran-inspire-adaptation-protocol-v1")
        # ``base.audit`` already performed the expensive source/archive
        # admission and authenticated this exact public protocol.  Reuse its
        # hash-bound inputs rather than immediately repeating that scan.
        base_receipt = base_protocol.get("inputs")
        require(type(base_receipt) is dict)
        p1_receipt = p1.authenticate(ATTEMPT)
        p1_protocol = p1._read(p1.paths(ATTEMPT) / "protocol.json")
        r7_protocol, _r7_aggregate, r7_items, r7_receipt = r7.authenticate("fit", ATTEMPT)
        return make_bindings(base_receipt, base_audit, p1_protocol, p1_receipt,
                             r7_protocol, r7_items, r7_receipt)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def _load_r7_checkpoint(bindings: dict):
    """Load only the exact authenticated R7 fold-0 checkpoint."""
    try:
        from bran_robust_clinical_r7 import BRANRobustClinicalR7
        fold = bindings["r7_fold0"]
        _out, private = r7.paths("fit", ATTEMPT)
        model, transform = r7.load_checkpoint(
            private / "fold0_R.pt", fold["checkpoint_sha256"], fold["binding"])
        require(type(model) is BRANRobustClinicalR7)
        require(model.training is False and not any(parameter.requires_grad
                                                     for parameter in model.parameters()))
        from bran_multisource_batches_v2 import transform_hash
        require(transform_hash(transform) == fold["binding"]["transform_sha256"])
        return model, transform
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def _arrays_unchanged(frame: object, snapshots: dict[str, np.ndarray]) -> None:
    for name, before in snapshots.items():
        after = getattr(frame, name)
        require(np.array_equal(after, before, equal_nan=True))


def infer_r7(frame: object, bindings: dict, *, loader=None):
    """Run the existing model-agnostic INSPIRE state adapter on R7."""
    try:
        names = ("values", "observed", "age_lower", "age_upper", "age_kind")
        snapshots = {name: np.array(getattr(frame, name), copy=True) for name in names}
        if loader is None:
            loader = _load_r7_checkpoint
        require(callable(loader))
        model, transform = loader(bindings)
        expected = bindings["r7_fold0"]["binding"]["transform_sha256"]
        result = state_kernel.infer(
            frame.values, frame.observed, frame.age_lower, frame.age_upper,
            frame.age_kind, model, transform, expected,
        )
        n = len(frame.values)
        require(hasattr(result, "state") and hasattr(result, "available")
                and type(result.state) is np.ndarray and result.state.dtype == np.dtype(np.float32)
                and result.state.shape == (n, kernel.STATE_WIDTH)
                and bool(np.isfinite(result.state).all())
                and type(result.available) is np.ndarray
                and result.available.dtype == np.dtype(bool)
                and result.available.shape == (n,)
                and np.array_equal(result.available, frame.observed.any(axis=1)))
        _arrays_unchanged(frame, snapshots)
        return result
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def _historical_report(bindings: dict) -> dict:
    out, _private = base.paths(ATTEMPT)
    path = out / "aggregate.json"
    require(_sha(path) == bindings["base_audit"]["aggregate_sha256"])
    result = base.read(path)
    validated = base.validate_result(result)
    require(validated is None or validated is True)
    report = result.get("report")
    require(kernel.validate_report(report) is True)
    return report


def _post_authentication(bindings: dict, frame: object) -> None:
    """Recheck the frozen frame/receipts without rescanning the source archive."""
    base_out, _base_private = base.paths(ATTEMPT)
    protocol = base.read(base_out / "protocol.json")
    require(_sha(base_out / "protocol.json") == bindings["base_audit"]["protocol_sha256"]
            and protocol.get("inputs") == bindings["base_receipt"])
    completed = base.read(base_out / "completed.json")
    require(completed.get("status") == "completed"
            and completed.get("protocol_sha256") == bindings["base_audit"]["protocol_sha256"]
            and completed.get("aggregate_sha256") == bindings["base_audit"]["aggregate_sha256"]
            and completed.get("patient_level_output_emitted") is False)
    # This is a private cached-frame replay, not a second source admission.
    current = base.load_source(bindings["base_receipt"])
    for item in fields(frame):
        before, after = getattr(frame, item.name), getattr(current, item.name)
        if before.dtype.kind == "f":
            require(np.array_equal(before, after, equal_nan=True))
        else:
            require(np.array_equal(before, after))
    p1_receipt = p1.authenticate(ATTEMPT)
    p1_protocol = p1._read(p1.paths(ATTEMPT) / "protocol.json")
    r7_protocol, _r7_aggregate, r7_items, r7_receipt = r7.authenticate("fit", ATTEMPT)
    refreshed = make_bindings(bindings["base_receipt"], bindings["base_audit"],
                              p1_protocol, p1_receipt, r7_protocol, r7_items,
                              r7_receipt)
    require(refreshed == bindings)


def compare_historical(new_report: dict, historical_report: dict) -> None:
    """Require unchanged context/blood arms, support and bootstrap population."""
    require(kernel.validate_report(new_report) is True
            and kernel.validate_report(historical_report) is True
            and new_report.get("status") == historical_report.get("status") == "ok")
    require(new_report["arms"]["context"] == historical_report["arms"]["context"]
            and new_report["arms"]["raw_context"] == historical_report["arms"]["raw_context"])
    require(new_report["support"] == historical_report["support"]
            and new_report["bootstrap"] == historical_report["bootstrap"])


def _rebuild_report(args: tuple, readouts: dict) -> dict:
    """Rebuild the closed report from saved predictions, without refitting."""
    try:
        role, outcome = args[-2], args[-1]
        labels = outcome[role == 2].astype(np.float64)
        arms = readouts.get("arms")
        require(type(arms) is dict and set(arms) == set(kernel.ARM_NAMES))
        predictions = {}
        for arm in kernel.ARM_NAMES:
            probability = arms[arm].get("test_predictions")
            require(type(probability) is np.ndarray and probability.dtype == np.dtype(np.float64)
                    and probability.shape == labels.shape and bool(np.isfinite(probability).all()))
            predictions[arm] = probability
        bootstrap = kernel._bootstrap(labels, predictions)
        require(bootstrap is not None)
        draws, valid = bootstrap
        reports = {
            arm: kernel._arm_report(labels, predictions[arm], draws[arm], valid)
            for arm in kernel.ARM_NAMES
        }
        contrasts = {
            "state_context_minus_raw_context": kernel._contrast_report(
                "state_context", "raw_context", reports, draws, valid, primary=True),
            "state_context_minus_context": kernel._contrast_report(
                "state_context", "context", reports, draws, valid, primary=False),
        }
        primary = contrasts["state_context_minus_raw_context"]
        return {
            "schema": kernel.SCHEMA,
            "status": "ok",
            "policy": dict(kernel.POLICY),
            "arms": reports,
            "contrasts": contrasts,
            "primary_superiority": bool(primary["superiority"]),
            "support": {
                "status": "released_lower_bounds",
                "roles": list(kernel.ROLE_NAMES),
                "nonevent_event_lower_bounds": kernel._nonevent_event_lower_bounds(role, outcome),
            },
            "bootstrap": {
                "requested_draws": int(kernel.POLICY["bootstrap_draws"]),
                "accepted_auroc_draws": int(valid.sum()),
            },
            "patient_level_output_emitted": False,
            "clinical_utility_established": False,
            "external_adaptation_performance_established": False,
        }
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def _private_write(path: Path, writer) -> None:
    require(not path.exists() and not path.is_symlink())
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _private_file(path: Path, pin: str) -> None:
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
            and path.stat().st_mode & 0o777 == 0o600
            and path.parent.is_dir() and not path.parent.is_symlink()
            and path.parent.stat().st_mode & 0o777 == 0o700 and _sha(path) == pin)


def _load_private_state(path: Path):
    try:
        with np.load(path, allow_pickle=False) as saved:
            require(set(saved.files) == {"state", "available"})
            raw_state = np.asarray(saved["state"])
            raw_available = np.asarray(saved["available"])
            require(raw_state.dtype == np.dtype(np.float32)
                    and raw_available.dtype == np.dtype(bool))
            state = np.array(raw_state, copy=True)
            available = np.array(raw_available, copy=True)
        require(state.ndim == 2 and state.shape[1] == kernel.STATE_WIDTH
                and available.shape == (len(state),) and bool(np.isfinite(state).all()))
        state.setflags(write=False)
        available.setflags(write=False)
        return state_kernel.PrivateInspireState(state, available)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def _load_private_readouts(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
        require(type(value) is dict)
        return value
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def validate_result(value: object) -> bool:
    try:
        require(type(value) is dict and set(value) == {
            "schema", "status", "frame", "report", "private_sha256", "provenance",
            "prior_external_test_reused", "encoder_exposed", "encoder_updated",
            "external_model_selection", "candidate_promoted", "patient_level_output_emitted",
            "clinical_use", "same_source_roles_and_input_eligibility",
            "same_context_raw_arms_as_v5", "same_support_and_bootstrap_as_v5",
        })
        require(value["schema"] == AGGREGATE_SCHEMA and value["status"] == "completed"
                and value["frame"] == FRAME)
        require(value["report"] is not None and kernel.validate_report(value["report"]) is True)
        require(value["prior_external_test_reused"] is True
                and all(value[key] is False for key in (
                    "encoder_exposed", "encoder_updated", "external_model_selection",
                    "candidate_promoted", "patient_level_output_emitted", "clinical_use"))
                and value["same_source_roles_and_input_eligibility"] is True
                and value["same_context_raw_arms_as_v5"] is True
                and value["same_support_and_bootstrap_as_v5"] is True)
        pins = value["private_sha256"]
        require(type(pins) is dict and set(pins) == set(PRIVATE_FILES)
                and all(_hex(pins[key]) for key in PRIVATE_FILES))
        provenance = value["provenance"]
        require(type(provenance) is dict and set(provenance) == {
            "code_sha256", "source_binding", "base_source_protocol_sha256",
            "base_source_aggregate_sha256", "base_v5_aggregate_sha256",
            "p1_receipt", "r7_receipt", "v5_checkpoint_sha256",
            "r7_checkpoint_sha256", "shared_transform_sha256",
        })
        require(type(provenance["code_sha256"]) is dict
                and all(_hex(item) for item in provenance["code_sha256"].values()))
        for key in ("base_source_protocol_sha256", "base_source_aggregate_sha256",
                    "base_v5_aggregate_sha256", "v5_checkpoint_sha256",
                    "r7_checkpoint_sha256", "shared_transform_sha256"):
            require(_hex(provenance[key]))
        require(type(provenance["source_binding"]) is dict
                and type(provenance["p1_receipt"]) is dict
                and type(provenance["r7_receipt"]) is dict)
        return True
    except Exception:
        return False


def _protocol(bindings: dict, hashes: dict[str, str]) -> dict:
    base_receipt = bindings["base_receipt"]
    base_audit = bindings["base_audit"]
    v5 = bindings["v5_fold0"]
    r7_fold = bindings["r7_fold0"]
    return {
        "schema": PROTOCOL_SCHEMA,
        "status": "frozen_before_inference",
        "frame": FRAME,
        "parameters": dict(kernel.POLICY),
        "bindings": bindings,
        "code_sha256": hashes,
        "runtime": runtime_versions(),
        "base_source_protocol_sha256": base_receipt["source_protocol_sha256"],
        "base_source_aggregate_sha256": base_receipt["source_aggregate_sha256"],
        "base_v5_aggregate_sha256": base_audit["aggregate_sha256"],
        "v5_checkpoint_sha256": v5["checkpoint_sha256"],
        "r7_checkpoint_sha256": r7_fold["checkpoint_sha256"],
        "shared_transform_sha256": v5["binding"]["transform_sha256"],
        "prior_external_test_reused": True,
        "encoder_exposed": False,
        "encoder_updated": False,
        "external_model_selection": False,
        "candidate_promoted": False,
        "patient_level_output_emitted": False,
    }


def _result(report: dict, bindings: dict, hashes: dict[str, str], pins: dict[str, str]) -> dict:
    base_receipt = bindings["base_receipt"]
    base_audit = bindings["base_audit"]
    v5 = bindings["v5_fold0"]
    r7_fold = bindings["r7_fold0"]
    return {
        "schema": AGGREGATE_SCHEMA,
        "status": "completed",
        "frame": FRAME,
        "report": report,
        "private_sha256": pins,
        "provenance": {
            "code_sha256": hashes,
            "source_binding": bindings["source_binding"],
            "base_source_protocol_sha256": base_receipt["source_protocol_sha256"],
            "base_source_aggregate_sha256": base_receipt["source_aggregate_sha256"],
            "base_v5_aggregate_sha256": base_audit["aggregate_sha256"],
            "p1_receipt": bindings["p1_receipt"],
            "r7_receipt": bindings["r7_receipt"],
            "v5_checkpoint_sha256": v5["checkpoint_sha256"],
            "r7_checkpoint_sha256": r7_fold["checkpoint_sha256"],
            "shared_transform_sha256": v5["binding"]["transform_sha256"],
        },
        "prior_external_test_reused": True,
        "encoder_exposed": False,
        "encoder_updated": False,
        "external_model_selection": False,
        "candidate_promoted": False,
        "patient_level_output_emitted": False,
        "clinical_use": False,
        "same_source_roles_and_input_eligibility": True,
        "same_context_raw_arms_as_v5": True,
        "same_support_and_bootstrap_as_v5": True,
    }


def _replay_saved(frame: object, saved_state: object, readouts: dict):
    args = base.evaluation_arguments(frame, saved_state)
    require(kernel.replay(*args, private_sink=readouts) is True)
    return args


def _audit_unlocked() -> dict:
    """Audit the terminal and independently replay its saved private readout."""
    out, private = paths()
    require(out.is_dir() and not out.is_symlink() and not (out / "failure.json").exists())
    require({item.name for item in out.iterdir()} == {
        "protocol.json", "aggregate.json", "progress.json", "completed.json"})
    require(private.is_dir() and not private.is_symlink()
            and private.stat().st_mode & 0o777 == 0o700
            and {item.name for item in private.iterdir()} == set(PRIVATE_FILES))
    bindings = authenticate_inputs()
    hashes = code_hashes()
    protocol = _read_json(out / "protocol.json")
    require(protocol == _protocol(bindings, hashes))
    result = _read_json(out / "aggregate.json")
    require(validate_result(result))
    pins = result["private_sha256"]
    for name, pin in pins.items():
        _private_file(private / name, pin)
    frame = base.load_source(bindings["base_receipt"])
    saved_state = _load_private_state(private / "states.npz")
    readouts = _load_private_readouts(private / "readouts.pkl")
    require(readouts.get("report") == result["report"])
    require(result == _result(readouts["report"], bindings, hashes, pins))
    fresh = infer_r7(frame, bindings)
    require(np.array_equal(saved_state.state, fresh.state)
            and np.array_equal(saved_state.available, fresh.available))
    args = _replay_saved(frame, saved_state, readouts)
    require(_rebuild_report(args, readouts) == result["report"])
    for name, pin in pins.items():
        _private_file(private / name, pin)
    compare_historical(result["report"], _historical_report(bindings))
    progress_record = _read_json(out / "progress.json")
    require(set(progress_record) == {"phase", "pid", "patient_level_output_emitted"}
            and progress_record["phase"] == "completed"
            and type(progress_record["pid"]) is int and progress_record["pid"] > 0
            and progress_record["patient_level_output_emitted"] is False)
    completed = _read_json(out / "completed.json")
    require(completed == {
        "status": "authenticated_completed",
        "protocol_sha256": _sha(out / "protocol.json"),
        "aggregate_sha256": _sha(out / "aggregate.json"),
        "patient_level_output_emitted": False,
    })
    return {"status": "aggregate_terminal_authenticated",
            "protocol_sha256": completed["protocol_sha256"],
            "aggregate_sha256": completed["aggregate_sha256"],
            "patient_level_output_emitted": False}


def audit() -> dict:
    """Audit the terminal under the shared lock and FD-quiet boundary."""
    try:
        with base._quiet():
            with LOCK.open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_SH)
                return _audit_unlocked()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def run() -> dict:
    out, private = paths()
    state = {"phase": "authentication"}
    require(not out.exists() and not out.is_symlink()
            and not private.exists() and not private.is_symlink())
    require(private.parent.is_dir() and not private.parent.is_symlink())
    with LOCK.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "not_started_shared_lock_busy", "patient_level_output_emitted": False}
        out.mkdir(mode=0o700)
        private.mkdir(mode=0o700)
        try:
            with base._quiet():
                progress(out, state, "authentication")
                bindings = authenticate_inputs()
                hashes = code_hashes()
                protocol = _protocol(bindings, hashes)
                _write_json(out / "protocol.json", protocol)

                progress(out, state, "source_load")
                frame = base.load_source(bindings["base_receipt"])
                historical = _historical_report(bindings)

                progress(out, state, "state_inference")
                fresh = infer_r7(frame, bindings)

                progress(out, state, "checkpoint_replay")
                replay = infer_r7(frame, bindings)
                require(np.array_equal(fresh.state, replay.state)
                        and np.array_equal(fresh.available, replay.available))
                del replay

                progress(out, state, "fixed_readouts_and_bootstrap")
                args = base.evaluation_arguments(frame, fresh)
                sink: dict[str, object] = {}
                report = kernel.evaluate(*args, private_sink=sink)
                require(kernel.validate_report(report) is True)
                compare_historical(report, historical)
                sink["report"] = report

                progress(out, state, "private_write_and_replay")
                _private_write(private / "states.npz", lambda handle: np.savez_compressed(
                    handle, state=fresh.state, available=fresh.available))
                _private_write(private / "readouts.pkl", lambda handle: pickle.dump(
                    sink, handle, protocol=5))
                pins = {name: _sha(private / name) for name in PRIVATE_FILES}
                for name, pin in pins.items():
                    _private_file(private / name, pin)
                saved_state = _load_private_state(private / "states.npz")
                saved_readouts = _load_private_readouts(private / "readouts.pkl")
                _replay_saved(frame, saved_state, saved_readouts)
                for name, pin in pins.items():
                    _private_file(private / name, pin)

                progress(out, state, "post_authentication")
                _post_authentication(bindings, frame)
                require(code_hashes() == hashes)
                result = _result(report, bindings, hashes, pins)
                require(validate_result(result))
                _write_json(out / "aggregate.json", result)
                progress(out, state, "completed")
                completed = {
                    "status": "authenticated_completed",
                    "protocol_sha256": _sha(out / "protocol.json"),
                    "aggregate_sha256": _sha(out / "aggregate.json"),
                    "patient_level_output_emitted": False,
                }
                _write_json(out / "completed.json", completed)
            return {"status": "completed", "patient_level_output_emitted": False}
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            with base._quiet():
                if not (out / "completed.json").exists():
                    _write_json(out / "failure.json", {
                        "status": "technical_failure",
                        "phase": state["phase"],
                        "error": ERROR,
                        "patient_level_output_emitted": False,
                    })
            return {"status": "failed", "phase": state["phase"],
                    "error": ERROR, "patient_level_output_emitted": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    try:
        with base._quiet():
            result = audit() if args.audit_only else run()
    except Exception:
        result = {"status": "not_completed", "error": ERROR,
                  "patient_level_output_emitted": False}
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get("status") in {
        "completed", "aggregate_terminal_authenticated", "not_started_shared_lock_busy"
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
