"""FD-quiet, locked NWICU V2 binding CLI; no training or source-table parsing."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import run_bran_nwicu_cohort_adult_v1 as legacy
import bran_multisource_nwicu_v2 as nwicu


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "BRAN_MULTISOURCE_NWICU_BINDING_V2_ATTEMPT1"
CODE = ("bran_multisource_nwicu_v2.py", "run_bran_multisource_nwicu_binding_v2.py",
        "bran_multisource_clinical_v2.py", "bran_multisource_age_v2.py",
        "bran_nwicu_cohort_adult_v1.py", "run_bran_nwicu_cohort_adult_v1.py")
ERROR = "nwicu v2 binding rejected"
_PHASES = frozenset(("authentication", "private_load", "conversion", "finalization"))


def _fail() -> None:
    raise ValueError(ERROR) from None


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def failure_payload() -> dict[str, object]:
    """Static, row-free failure result used for every authentication/cache error."""
    return {"status": "failed", "patient_level_output_emitted": False,
            "training_started": False, "arrays_emitted": False}


def _code_hashes() -> dict[str, str]:
    return {name: legacy.safe.sha(ROOT / name) for name in CODE}


def _result(protocol_sha256: str, audit_sha256: str, state: dict[str, object]) -> dict[str, object]:
    # Reauthenticate the frozen old audit and source receipt before private cache access.
    state["phase"] = "authentication"
    before = _code_hashes()
    legacy.verify_audit(protocol_sha256, audit_sha256)
    receipt = legacy.source_receipt()
    state["phase"] = "private_load"
    adapted = legacy.load_private(receipt)
    # A private load cannot race a changed audit, receipt, or bridge dependency.
    state["phase"] = "authentication"
    legacy.verify_audit(protocol_sha256, audit_sha256)
    if legacy.source_receipt() != receipt or _code_hashes() != before:
        _fail()
    state["phase"] = "conversion"
    pool = nwicu.convert(adapted)
    summary = nwicu.safe_summary(pool)
    if summary["status"] != "supported_training_pool" or summary["prospective_v2_training_admission"] is not True:
        _fail()
    state["phase"] = "finalization"
    return {
        "schema": "bran-multisource-nwicu-binding-v2", "status": "completed_aggregate_only",
        "source_receipt_sha256": _digest(receipt), "legacy_protocol_sha256": protocol_sha256,
        "legacy_audit_sha256": audit_sha256,
        "code_sha256": before,
        "eligibility_sha256": nwicu.eligibility_hash(pool), "safe_summary": summary,
        "new_prospective_training_admission": True,
        "legacy_qualification_training_permitted": False,
        "patient_level_output_emitted": False, "training_started": False,
    }


def run(protocol_sha256: str, audit_sha256: str, state: dict[str, object]) -> dict[str, object]:
    if not isinstance(protocol_sha256, str) or not isinstance(audit_sha256, str):
        _fail()
    legacy.safe.absent(OUT)
    OUT.mkdir()
    state["owned"] = OUT
    item = _result(protocol_sha256, audit_sha256, state)
    legacy.safe.write_json(OUT / "aggregate.json", item)
    legacy.safe.write_json(OUT / "manifest.json", {
        "aggregate_sha256": legacy.safe.sha(OUT / "aggregate.json"),
        "patient_level_output_emitted": False,
    })
    state["owned"] = None
    return item


def _failure_terminal(path: Path, phase: object) -> None:
    if phase not in _PHASES:
        phase = "authentication"
    legacy.safe.write_json(path / "failure.json", {"status": "failed", "phase": phase,
                                                     "reason": "authentication_or_binding_failed"})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--audit-sha256", required=True)
    args = parser.parse_args(argv)
    answer = failure_payload()
    state: dict[str, object] = {"owned": None, "phase": "authentication"}
    with legacy.safe.quiet():
        try:
            with legacy.LOCK.open("a") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                item = run(args.protocol_sha256, args.audit_sha256, state)
                answer = {"status": "completed", "aggregate_sha256": legacy.safe.sha(OUT / "aggregate.json"),
                          "patient_level_output_emitted": item["patient_level_output_emitted"],
                          "training_started": False, "arrays_emitted": False}
        except Exception:
            if state["owned"] is not None:
                try:
                    _failure_terminal(state["owned"], state["phase"])
                except Exception:
                    pass
            answer = failure_payload()
    print(json.dumps(answer, sort_keys=True))
    return int(answer["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
