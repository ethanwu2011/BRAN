"""Quiet, immutable H6 R7 replication of the frozen H4 adaptation recipe.

H6 owns only authentication and orchestration.  The H4 numerical evaluator and
state extractor remain unchanged; private arrays exist only in memory and are
never serialized or printed.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
import run_bran_hirid_h3 as h3
import bran_hirid_source_h3 as source


ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "BRAN_HIRID_R7_H6_DESIGN.md"
CODE = (
    "BRAN_HIRID_R7_H6_DESIGN.md",
    "run_bran_hirid_r7_h6.py",
    "test_run_bran_hirid_r7_h6.py",
    "run_bran_hirid_r7_h6_attempt1.sh",
)
H4_PROTOCOL = "d7a810ecda72608fca344e7dc856249458c93a0756f78fcbf3bb196aed40674c"
H4_AGGREGATE = "1e731b80ee5634552c01357dade01a4a90c64d5cc361b8a6ffdd8635bed5e2fe"
H5_PROTOCOL = "81e1d1b3ab3335a83bb0dee082f0afaddee4627196a2279d9ff63216386d8701"
H5_AGGREGATE = "e7dab6358ca4e56086af321c670f33d956df3faec5396244453f00e0c0e0fbf2"
R7_CHECKPOINT = "82d9cc71794dc116811214b647f396e4a43525f3094c7c0ca04b59e36ce0c08f"
H5_AUDIT_RECEIPT = ROOT / "BRAN_HIRID_R7_H5_REPLAY.json"
SCHEMA = "bran-hirid-r7-h6-aggregate-v1"
PROTOCOL_SCHEMA = "bran-hirid-r7-h6-protocol-v1"
ERROR = "bran_hirid_r7_h6_execution_failed"
PHASES = (
    "authentication", "source_authentication", "source_selection",
    "state_inference", "checkpoint_replay", "readout_fitting_and_bootstrap",
    "aggregate_replay", "post_authentication", "completed", "unsupported",
)
FIELDS = (
    "potassium", "sodium", "chloride", "creatinine", "bilirubin_total",
    "albumin", "glucose",
)
TERMINAL_NAMES = ("completed.json", "unsupported.json")


def require(ok: bool) -> None:
    if not ok:
        raise ValueError(ERROR) from None


def sha(path: Path) -> str:
    return source._hash_file(path)


def paths(attempt: int = 1) -> Path:
    require(type(attempt) is int and 1 <= attempt <= 99)
    return ROOT / f"BRAN_HIRID_R7_H6_ATTEMPT{attempt}"


def _read(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 5_000_000)

    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result)
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                       parse_constant=lambda _value: require(False))
    require(type(value) is dict)
    return value


def _safe_code_name(name: Any) -> str:
    require(type(name) is str and name and not Path(name).is_absolute())
    path = Path(name)
    require(path.name == name and ".." not in path.parts)
    return name


def _code_closure(code: Any) -> dict[str, str]:
    require(type(code) is dict and bool(code))
    current = {}
    for name, pin in sorted(code.items()):
        name = _safe_code_name(name)
        require(type(pin) is str and len(pin) == 64 and set(pin) <= set("0123456789abcdef"))
        current[name] = sha(ROOT / name)
        require(current[name] == pin)
    return current


def _terminal(out: Path) -> tuple[str, dict[str, Any]]:
    found = [name for name in TERMINAL_NAMES if (out / name).exists()]
    require(len(found) == 1)
    return found[0], _read(out / found[0])


def authenticate_h4() -> dict[str, Any]:
    """Authenticate the closed H4 terminal without rerunning H4."""
    import run_bran_hirid_h4 as h4

    out = h4.paths(1)
    require(out.is_dir() and not out.is_symlink() and not (out / "failed.json").exists())
    terminal_name, done = _terminal(out)
    require({item.name for item in out.iterdir()} ==
            {"launch.json", "protocol.json", "aggregate.json", "progress.json", terminal_name})
    require(all(item.is_file() and not item.is_symlink() and item.stat().st_nlink == 1
                for item in out.iterdir()))
    launch = _read(out / "launch.json")
    protocol = _read(out / "protocol.json")
    aggregate = _read(out / "aggregate.json")
    progress_value = _read(out / "progress.json")
    pin = sha(out / "protocol.json")
    require(pin == H4_PROTOCOL and sha(out / "aggregate.json") == H4_AGGREGATE)
    require(type(protocol) is dict and set(protocol) == {
        "schema", "launch", "source_receipt_sha256", "admission_role_binding_sha256",
        "frozen_before_fitting",
    } and protocol["schema"] == "bran-hirid-h4-protocol-v1"
            and protocol["launch"] == launch and protocol["frozen_before_fitting"] is True)
    require(type(protocol["source_receipt_sha256"]) is str and len(protocol["source_receipt_sha256"]) == 64)
    role_pin = protocol["admission_role_binding_sha256"]
    require(type(role_pin) is str and len(role_pin) == 64 and set(role_pin) <= set("0123456789abcdef"))
    require(type(done) is dict and type(done.get("status")) is str
            and done.get("status") + ".json" == terminal_name
            and done.get("status") == aggregate.get("report", {}).get("status")
            and done.get("protocol_sha256") == H4_PROTOCOL
            and done.get("aggregate_sha256") == H4_AGGREGATE
            and done.get("patient_level_output_emitted") is False)
    require(type(progress_value) is dict and progress_value.get("phase") == done["status"]
            and progress_value.get("patient_level_output_emitted") is False)
    if "elapsed_seconds" in done:
        require(type(done["elapsed_seconds"]) in (int, float)
                and not isinstance(done["elapsed_seconds"], bool)
                and done["elapsed_seconds"] >= 0)
    h4.validate_aggregate(aggregate)
    require(type(aggregate["report"]) is dict)
    code = _code_closure(launch.get("code_sha256"))
    require(launch.get("patient_level_output_emitted") is False)
    return {
        "launch": launch,
        "protocol": protocol,
        "aggregate": aggregate,
        "report": aggregate["report"],
        "role_pin": role_pin,
        "source_receipt_sha256": protocol["source_receipt_sha256"],
        "code_sha256": code,
    }


def _validate_h5_receipt(receipt: Any, protocol_pin: str, aggregate_pin: str) -> None:
    require(type(receipt) is dict and set(receipt) == {
        "status", "protocol_sha256", "aggregate_sha256",
        "independent_source_inference_replay", "patient_level_output_emitted",
    } and receipt["status"] == "aggregate_terminal_authenticated"
            and receipt["protocol_sha256"] == protocol_pin
            and receipt["aggregate_sha256"] == aggregate_pin
            and receipt["independent_source_inference_replay"] is True
            and receipt["patient_level_output_emitted"] is False)


def authenticate_h5(receipt_path: Path = H5_AUDIT_RECEIPT) -> dict[str, Any]:
    """Authenticate H5's closed terminal and explicit independent receipt."""
    import run_bran_hirid_r7_h5 as h5

    out = h5.paths(1)
    require(out.is_dir() and not out.is_symlink() and not (out / "failure.json").exists())
    require({item.name for item in out.iterdir()} ==
            {"protocol.json", "aggregate.json", "progress.json", "completed.json"})
    require(all(item.is_file() and not item.is_symlink() and item.stat().st_nlink == 1
                for item in out.iterdir()))
    protocol_path, aggregate_path = out / "protocol.json", out / "aggregate.json"
    protocol, aggregate, done, progress = (_read(path) for path in (
        protocol_path, aggregate_path, out / "completed.json", out / "progress.json"))
    require(sha(protocol_path) == H5_PROTOCOL and sha(aggregate_path) == H5_AGGREGATE)
    require(type(protocol) is dict and set(protocol) == {
        "schema", "binding", "source", "frozen_before_source_selection_and_inference",
        "patient_level_output_emitted",
    } and protocol["schema"] == "bran-hirid-r7-h5-protocol-v1"
            and protocol["frozen_before_source_selection_and_inference"] is True
            and protocol["patient_level_output_emitted"] is False)
    require(type(done) is dict and done == {
        "status": "authenticated_completed", "protocol_sha256": H5_PROTOCOL,
        "aggregate_sha256": H5_AGGREGATE, "patient_level_output_emitted": False,
    })
    require(type(progress) is dict and progress.get("phase") == "completed"
            and progress.get("patient_level_output_emitted") is False)
    h5.validate_aggregate(aggregate)
    binding = protocol["binding"]
    require(type(binding) is dict and binding.get("patient_level_output_emitted") is False)
    code = _code_closure(binding.get("code_sha256"))
    _validate_h5_receipt(_read(receipt_path), H5_PROTOCOL, H5_AGGREGATE)
    receipt_sha = sha(receipt_path)
    # H5 authentication supplies the R7 binding and the exact H3 source receipt.
    authenticated_binding, source_receipt = h5.authenticate()
    require(authenticated_binding == binding and source_receipt == protocol["source"])
    entry = binding.get("R7")
    require(type(entry) is dict and entry.get("fold") == 0
            and entry.get("checkpoint_sha256") == R7_CHECKPOINT)
    return {
        "binding": binding,
        "source": source_receipt,
        "aggregate": aggregate,
        "report": aggregate["report"],
        "receipt_sha256": receipt_sha,
        "code_sha256": code,
    }


def authenticate(receipt_path: Path = H5_AUDIT_RECEIPT) -> dict[str, Any]:
    h4 = authenticate_h4()
    h5 = authenticate_h5(receipt_path)
    require(h4["source_receipt_sha256"] == source.content_hash(h5["source"])
            and h4["launch"].get("approval_sha256") == h5["binding"].get("approval_sha256"))
    # Keep the H6 protocol a binding, not a second copy of the historical
    # aggregates.  The validated H4 report is returned in a runtime-only
    # companion field for exact comparison during calculation/audit.
    own_code = {name: sha(ROOT / name) for name in CODE}
    binding = {
        "plan_sha256": own_code[PLAN.name],
        "code_sha256": own_code,
        "h4": {
            "protocol_sha256": H4_PROTOCOL,
            "aggregate_sha256": H4_AGGREGATE,
            "launch": h4["launch"],
            "role_pin": h4["role_pin"],
            "source_receipt_sha256": h4["source_receipt_sha256"],
            "code_sha256": h4["code_sha256"],
        },
        "h5": {
            "protocol_sha256": H5_PROTOCOL,
            "aggregate_sha256": H5_AGGREGATE,
            "binding": h5["binding"],
            "receipt_sha256": h5["receipt_sha256"],
            "code_sha256": h5["code_sha256"],
        },
        "source": h5["source"],
        "h4_role_pin": h4["role_pin"],
        "h5_receipt_sha256": h5["receipt_sha256"],
        "patient_level_output_emitted": False,
    }
    return {"binding": binding, "h4_report": h4["report"]}


def load_source(receipt: dict[str, Any], h4_launch: dict[str, Any], callback):
    import run_bran_hirid_h4 as h4
    selections, roles, role_pin = h4.load_source(receipt, h4_launch, callback)
    return selections, roles, role_pin


def extract_r7(selections, binding: dict[str, Any]):
    """Extract R7 private arrays with the unchanged H4 adapter/state boundary."""
    import torch
    import run_bran_robust_clinical_r7 as r7
    from bran_hirid_hb_evaluation_h2 import _selection_statuses
    from bran_hirid_state_h4 import extract
    from bran_hirid_v5_input_adapter_v1 import prepare_inputs
    from bran_knhanes_input_kernel_v1 import CANONICAL_INDEX
    from bran_robust_clinical_r7 import BRANRobustClinicalR7

    torch.set_num_threads(2)
    item = binding["R7"]
    _, private = r7.paths("fit", 1)
    model, transform = r7.load_checkpoint(private / "fold0_R.pt",
                                          item["checkpoint_sha256"], item["binding"])
    require(type(model) is BRANRobustClinicalR7)
    batch = prepare_inputs(selections)
    state = extract(batch, model, transform, item["binding"]["transform_sha256"])
    statuses, truth = _selection_statuses(selections)
    require(np.array_equal(state.available, statuses == "ready"))
    columns = [CANONICAL_INDEX[name] for name in FIELDS]
    raw_mask = batch.clinical_mask[:, columns]
    require(np.array_equal(raw_mask.any(axis=1), statuses == "ready"))
    raw_values = np.where(raw_mask, batch.clinical_values[:, columns], np.nan)
    return state.states, raw_values, raw_mask, state.native_hb, truth, statuses


def _same_arrays(first, second) -> None:
    require(len(first) == len(second))
    for left, right in zip(first, second):
        require(isinstance(left, np.ndarray) and isinstance(right, np.ndarray))
        if left.dtype.kind in "fc" or right.dtype.kind in "fc":
            require(np.array_equal(left, right, equal_nan=True))
        else:
            require(np.array_equal(left, right))


def _compare_h4_raw(h6_report: dict[str, Any], h4_report: dict[str, Any]) -> None:
    require(h6_report["status"] == h4_report["status"]
            and h6_report["privacy_flags"] == h4_report["privacy_flags"])
    if h4_report["status"] == "unsupported":
        require(h6_report == h4_report)
        return
    require(h6_report["partition_status_counts"] == h4_report["partition_status_counts"]
            and h6_report["prediction_coverage"] == h4_report["prediction_coverage"]
            and h6_report["study_flags"] == h4_report["study_flags"])
    for scope in ("overall", "low_hb"):
        historical, current = h4_report[scope], h6_report[scope]
        require(historical["status"] == current["status"])
        if historical["status"] == "suppressed":
            require(current == historical)
        else:
            require(current["methods"]["fit_median"] == historical["methods"]["fit_median"]
                    and current["methods"]["raw_ridge"] == historical["methods"]["raw_ridge"])


def calculate(selections, roles, binding: dict[str, Any], h4_report: dict[str, Any], callback):
    import bran_hirid_adaptation_h4 as h4metrics

    callback("state_inference")
    arrays = extract_r7(selections, binding)
    callback("checkpoint_replay")
    other = extract_r7(selections, binding)
    _same_arrays(arrays, other)
    callback("readout_fitting_and_bootstrap")
    report = h4metrics.evaluate(*arrays, roles)
    h4metrics.validate_result(report)
    _compare_h4_raw(report, h4_report)
    callback("aggregate_replay")
    replay = h4metrics.evaluate(*other, roles)
    require(report == replay)
    return report


def progress(out: Path, state: dict[str, Any], phase: str) -> None:
    require(phase in PHASES)
    state["phase"] = phase
    temporary = out / "progress.next.json"
    payload = {"phase": phase, "patient_level_output_emitted": False}
    h3.write(temporary, payload)
    os.replace(temporary, out / "progress.json")


def validate_aggregate(value: Any) -> None:
    import bran_hirid_adaptation_h4 as h4metrics

    require(type(value) is dict and set(value) == {
        "schema", "protocol_sha256", "report", "historical_h4_raw_replay_equal",
        "r7_state_replay_equal", "aggregate_replay_equal", "encoder_updated",
        "external_model_selection", "patient_level_output_emitted",
    } and value["schema"] == SCHEMA)
    pin = value["protocol_sha256"]
    require(type(pin) is str and len(pin) == 64 and set(pin) <= set("0123456789abcdef"))
    for key in ("historical_h4_raw_replay_equal", "r7_state_replay_equal", "aggregate_replay_equal"):
        require(value[key] is True)
    for key in ("encoder_updated", "external_model_selection", "patient_level_output_emitted"):
        require(value[key] is False)
    h4metrics.validate_result(value["report"])


def run(attempt: int = 1, h5_audit_receipt: Path = H5_AUDIT_RECEIPT):
    out = paths(attempt)
    require(not out.exists() and not out.is_symlink())
    state = {"phase": "authentication"}
    with h3.LOCK.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "not_started_shared_lock_busy", "patient_level_output_emitted": False}
        out.mkdir(mode=0o700)
        try:
            with _quiet():
                progress(out, state, "authentication")
                auth = authenticate(h5_audit_receipt)
                binding = auth["binding"]
                h4_report = auth["h4_report"]
                protocol = {
                    "schema": PROTOCOL_SCHEMA,
                    "binding": binding,
                    "source": binding["source"],
                    "frozen_before_source_selection_and_fitting": True,
                    "patient_level_output_emitted": False,
                }
                h3.write(out / "protocol.json", protocol)
                protocol_pin = sha(out / "protocol.json")
                progress(out, state, "source_authentication")
                source.recheck_source(binding["source"])
                progress(out, state, "source_selection")
                selections, roles, role_pin = load_source(
                    binding["source"], binding["h4"]["launch"],
                    lambda phase, *_counts: progress(out, state, phase)
                )
                require(role_pin == binding["h4_role_pin"])
                report = calculate(selections, roles, binding["h5"]["binding"],
                                   h4_report, lambda phase: progress(out, state, phase))
                progress(out, state, "post_authentication")
                auth_after = authenticate(h5_audit_receipt)
                require(auth_after["binding"] == binding and auth_after["h4_report"] == h4_report)
                require(sha(out / "protocol.json") == protocol_pin)
                source.recheck_source(binding["source"])
                aggregate = {
                    "schema": SCHEMA,
                    "protocol_sha256": protocol_pin,
                    "report": report,
                    "historical_h4_raw_replay_equal": True,
                    "r7_state_replay_equal": True,
                    "aggregate_replay_equal": True,
                    "encoder_updated": False,
                    "external_model_selection": False,
                    "patient_level_output_emitted": False,
                }
                validate_aggregate(aggregate)
                h3.write(out / "aggregate.json", aggregate)
                terminal_status = "unsupported" if report["status"] == "unsupported" else "completed"
                progress(out, state, terminal_status)
                h3.write(out / f"{terminal_status}.json", {
                    "status": terminal_status,
                    "protocol_sha256": protocol_pin,
                    "aggregate_sha256": sha(out / "aggregate.json"),
                    "patient_level_output_emitted": False,
                })
            return {"status": terminal_status + "_pending_separate_audit",
                    "patient_level_output_emitted": False}
        except BaseException:
            require(not (out / "completed.json").exists() and not (out / "unsupported.json").exists())
            h3.write(out / "failure.json", {
                "status": "technical_failure", "phase": state["phase"],
                "error_code": ERROR, "patient_level_output_emitted": False,
            })
            return {"status": "technical_failure", "phase": state["phase"],
                    "patient_level_output_emitted": False}


def _audit(attempt: int = 1, h5_audit_receipt: Path = H5_AUDIT_RECEIPT):
    with h3.LOCK.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with _quiet():
            out = paths(attempt)
            terminal_name, terminal = _terminal(out)
            require({item.name for item in out.iterdir()} ==
                    {"protocol.json", "aggregate.json", "progress.json", terminal_name})
            protocol = _read(out / "protocol.json")
            aggregate = _read(out / "aggregate.json")
            pin = sha(out / "protocol.json")
            require(type(protocol) is dict and set(protocol) == {
                "schema", "binding", "source", "frozen_before_source_selection_and_fitting",
                "patient_level_output_emitted",
            } and protocol["schema"] == PROTOCOL_SCHEMA
                    and protocol["frozen_before_source_selection_and_fitting"] is True
                    and protocol["patient_level_output_emitted"] is False)
            require(type(terminal) is dict and type(terminal.get("status")) is str
                    and terminal.get("status") + ".json" == terminal_name
                    and terminal.get("status") == aggregate.get("report", {}).get("status")
                    and terminal.get("protocol_sha256") == pin
                    and terminal.get("aggregate_sha256") == sha(out / "aggregate.json")
                    and terminal.get("patient_level_output_emitted") is False)
            progress_value = _read(out / "progress.json")
            require(progress_value == {"phase": terminal["status"],
                                       "patient_level_output_emitted": False})
            require(aggregate.get("protocol_sha256") == pin)
            validate_aggregate(aggregate)
            auth = authenticate(h5_audit_receipt)
            binding = auth["binding"]
            h4_report = auth["h4_report"]
            require(binding == protocol["binding"] and binding["source"] == protocol["source"])
            source.recheck_source(binding["source"])
            selections, roles, role_pin = load_source(
                binding["source"], binding["h4"]["launch"], lambda *_args: None
            )
            require(role_pin == binding["h4_role_pin"])
            report = calculate(selections, roles, binding["h5"]["binding"],
                               h4_report, lambda *_args: None)
            require(report == aggregate["report"])
            source.recheck_source(binding["source"])
            auth_after = authenticate(h5_audit_receipt)
            require(auth_after == auth)
            terminal_after_name, terminal_after = _terminal(out)
            require(terminal_after_name == terminal_name
                    and _read(out / "protocol.json") == protocol
                    and _read(out / "aggregate.json") == aggregate
                    and terminal_after == terminal)
            return {"status": "aggregate_terminal_authenticated",
                    "protocol_sha256": pin,
                    "aggregate_sha256": terminal["aggregate_sha256"],
                    "independent_source_fit_calibration_evaluation_replay": True,
                    "patient_level_output_emitted": False}


def audit(attempt: int = 1, h5_audit_receipt: Path = H5_AUDIT_RECEIPT):
    try:
        return _audit(attempt, h5_audit_receipt)
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ValueError(ERROR) from None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--h5-audit-receipt", type=Path, default=H5_AUDIT_RECEIPT)
    args = parser.parse_args()
    try:
        value = audit(args.attempt, args.h5_audit_receipt) if args.audit else run(args.attempt, args.h5_audit_receipt)
        print(json.dumps(value, sort_keys=True))
        return 0 if value["status"] in (
            "completed_pending_separate_audit", "unsupported_pending_separate_audit",
            "aggregate_terminal_authenticated",
        ) else 1
    except BaseException:
        print(json.dumps({"status": "closed_failure", "patient_level_output_emitted": False}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
