"""Exclusive FD-quiet runner and aggregate-only verifier for R7 fixed-state P1."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json

ROOT = Path(__file__).resolve().parent
ATTEMPT = 1
ERROR = "r7_fixed_state_p1_runner_failed"
PHASES = ("authentication", "source_loading", "state_inference", "fixed_readout",
          "aggregate_bootstrap", "post_authentication", "completed")
CODE = ("BRAN_R7_FIXED_STATE_REUSE_P1_DESIGN.md", "bran_r7_fixed_state_p1.py",
        "run_bran_r7_fixed_state_p1.py", "test_bran_r7_fixed_state_p1.py",
        "test_run_bran_r7_fixed_state_p1.py", "run_bran_r7_fixed_state_p1_attempt1.sh",
        "bran_v5_state_routes.py", "bran_robust_clinical_r7.py",
        "run_bran_robust_clinical_r7.py", "audit_bran_robust_clinical_r7_v2.py")
_ROUTE_METRIC_DEPS = ("bran_v5_state_routes.py", "bran_v5_residual_training.py",
    "bran_multisource_inference_v2.py", "bran_multisource_age_v2.py",
    "bran_multisource_profiles_v3.py", "bran_multisource_outcome_metrics_v2.py",
    "bran_external_cbc_evaluation_v1.py", "run_bran_overnight_diagnostic_v1.py",
    "bran_missingness_stress_metrics_v1.py", "bran_multisource_batches_v2.py",
    "bran_multisource_outcomes_v2.py", "bran_clinical_semantics_v1.py")
_R7_GUARD_DEPS = ("audit_bran_robust_clinical_r7_v2.py", "run_bran_robust_clinical_r7.py",
    "bran_robust_clinical_r7.py", "audit_bran_source_pattern_v6.py",
    "bran_multisource_protocol_v2.py", "bran_research_state_io_v1.py",
    "run_bran_multisource_fit_v2.py")


def require(ok: bool) -> None:
    if not ok:
        raise ValueError(ERROR) from None


def paths(attempt: int = ATTEMPT) -> Path:
    require(type(attempt) is int and attempt == ATTEMPT)
    return ROOT / "BRAN_R7_FIXED_STATE_P1_ATTEMPT1"


def code_hashes() -> dict[str, str]:
    # Parent.CODE and the guarded R7 runner's own closure are versioned inputs,
    # not transitive assumptions hidden behind their public receipts.
    import run_bran_v5_cbc_uncertainty as parent
    import run_bran_robust_clinical_r7 as r7
    names = set(CODE) | set(parent.CODE) | set(r7.code_hashes()) | set(_ROUTE_METRIC_DEPS) | set(_R7_GUARD_DEPS)
    return {name: sha(ROOT / name) for name in sorted(names)}


def checkpoint_manifest(components: dict, component_role: str, source_binding: dict) -> dict:
    """Return a five-fold public binding manifest without checkpoint bytes."""
    try:
        require(component_role in ("M", "R") and type(components) is dict and type(source_binding) is dict)
        rows = []
        for fold in range(5):
            item = components[(component_role, fold)]
            binding, pin = item["binding"], item["checkpoint_sha256"]
            require(type(binding) is dict and binding["role"] == component_role and binding["fold"] == fold
                    and binding["outer_fold_sha256"] == source_binding["outer_fold_sha256"]
                    and binding["inner_fold_sha256"] == source_binding["inner_fold_sha256"][fold]
                    and type(pin) is str and len(pin) == 64)
            rows.append({"fold": fold, "checkpoint_sha256": pin, "binding": binding})
        return {"component_role": component_role, "folds": rows}
    except Exception:
        raise ValueError(ERROR) from None


def validate_checkpoint_manifest(value: object, component_role: str, source_binding: dict) -> None:
    try:
        require(type(value) is dict and value.get("component_role") == component_role
                and type(value.get("folds")) is list and len(value["folds"]) == 5)
        require([row.get("fold") if type(row) is dict else None for row in value["folds"]] == list(range(5)))
        copied = {(component_role, row["fold"]): {"binding": row["binding"],
                  "checkpoint_sha256": row["checkpoint_sha256"]} for row in value["folds"]}
        require(checkpoint_manifest(copied, component_role, source_binding) == value)
    except Exception:
        raise ValueError(ERROR) from None


def require_checkpoint_manifest_match(components: dict, component_role: str,
                                      source_binding: dict, expected: object) -> None:
    """Bind a post-authentication component map to its pre-inference manifest."""
    try:
        validate_checkpoint_manifest(expected, component_role, source_binding)
        require(checkpoint_manifest(components, component_role, source_binding) == expected)
    except Exception:
        raise ValueError(ERROR) from None


def _read(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 5_000_000)
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result)
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=lambda _: require(False))
    require(type(value) is dict)
    return value


def progress(out: Path, state: dict, phase: str, fold=None, role=None) -> None:
    require(phase in PHASES and (fold is None or type(fold) is int and fold in range(5))
            and role in (None, "V5", "R7"))
    state["phase"] = phase
    write_json(out / "progress.next.json", {"phase": phase, "fold": fold, "role": role,
        "pid": os.getpid(), "patient_level_output_emitted": False})
    os.replace(out / "progress.next.json", out / "progress.json")


def _receipt_terminal(protocol: Path, aggregate: Path, completed: Path) -> None:
    terminal = _read(completed)
    require(set(terminal) == {"status", "protocol_sha256", "aggregate_sha256", "patient_level_output_emitted"}
            and terminal["status"] == "authenticated_completed"
            and terminal["protocol_sha256"] == sha(protocol) and terminal["aggregate_sha256"] == sha(aggregate)
            and terminal["patient_level_output_emitted"] is False)


def authenticate(attempt: int = ATTEMPT) -> dict:
    """Audit only the bounded public terminal; never loads data or checkpoints."""
    try:
        import bran_r7_fixed_state_p1 as evaluation
        out = paths(attempt)
        require(out.is_dir() and not out.is_symlink() and not (out / "failure.json").exists())
        require({item.name for item in out.iterdir()} == {"protocol.json", "progress.json", "aggregate.json", "completed.json"})
        require(all(item.is_file() and not item.is_symlink() and item.stat().st_nlink == 1 for item in out.iterdir()))
        protocol, aggregate = _read(out / "protocol.json"), _read(out / "aggregate.json")
        require(set(protocol) == {"schema", "status", "parameters", "code_sha256", "v5_source_binding",
                "r7_source_binding", "v5_baseline_record_sha256", "v5_checkpoint_manifest",
                "r7_checkpoint_manifest", "r7_fit_receipt",
                "patient_level_output_emitted", "candidate_promoted"})
        require(protocol["schema"] == "bran-r7-fixed-state-p1-protocol"
                and protocol["status"] == "frozen_before_readouts" and protocol["parameters"] == evaluation.PARAMETERS
                and protocol["code_sha256"] == code_hashes() and protocol["v5_source_binding"] == protocol["r7_source_binding"]
                and type(protocol["v5_baseline_record_sha256"]) is str and len(protocol["v5_baseline_record_sha256"]) == 64
                and type(protocol["r7_fit_receipt"]) is dict and set(protocol["r7_fit_receipt"]) ==
                    {"protocol_sha256", "aggregate_sha256", "terminal_sha256"}
                and all(type(x) is str and len(x) == 64 for x in protocol["r7_fit_receipt"].values())
                and protocol["patient_level_output_emitted"] is False and protocol["candidate_promoted"] is False)
        validate_checkpoint_manifest(protocol["v5_checkpoint_manifest"], "M", protocol["v5_source_binding"])
        validate_checkpoint_manifest(protocol["r7_checkpoint_manifest"], "R", protocol["r7_source_binding"])
        evaluation.validate_result(aggregate)
        item = _read(out / "progress.json")
        require(set(item) == {"phase", "fold", "role", "pid", "patient_level_output_emitted"}
                and item["phase"] == "completed" and item["fold"] is None and item["role"] is None
                and type(item["pid"]) is int and item["pid"] > 0 and item["patient_level_output_emitted"] is False)
        _receipt_terminal(out / "protocol.json", out / "aggregate.json", out / "completed.json")
        return {"status": "aggregate_terminal_authenticated", "protocol_sha256": sha(out / "protocol.json"),
                "aggregate_sha256": sha(out / "aggregate.json"), "source_rows_read": False,
                "checkpoint_bytes_reloaded_here": False, "patient_level_output_emitted": False}
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def run(attempt: int, state: dict) -> None:
    import run_bran_v5_cbc_uncertainty as parent
    import run_bran_context_preservation_v5 as v5
    import run_bran_robust_clinical_r7 as r7
    from audit_bran_robust_clinical_r7_v2 import authenticate as authenticate_r7
    from bran_multisource_binding_v3 import load_bound_sources
    import bran_r7_fixed_state_p1 as evaluation

    require(type(attempt) is int and attempt == ATTEMPT)
    out = paths(attempt)
    require(not out.exists() and not out.is_symlink())
    out.mkdir(); state["out"] = out
    progress(out, state, "authentication")
    baseline, v5_protocol, _, v5_components = parent.small_authentication()
    baseline_pin = sha(parent.BASELINE)
    r7_protocol, _, r7_components, r7_receipt = authenticate_r7("fit", 1)
    progress(out, state, "source_loading")
    sources = load_bound_sources()
    source_receipt = sources.receipt()
    require(v5_protocol["source_binding"] == r7_protocol["source_binding"] == source_receipt)
    v5_manifest = checkpoint_manifest(v5_components, "M", source_receipt)
    r7_manifest = checkpoint_manifest(r7_components, "R", source_receipt)
    protocol = {"schema": "bran-r7-fixed-state-p1-protocol", "status": "frozen_before_readouts",
        "parameters": evaluation.PARAMETERS, "code_sha256": code_hashes(),
        "v5_source_binding": source_receipt, "r7_source_binding": r7_protocol["source_binding"],
        "v5_baseline_record_sha256": baseline_pin, "v5_checkpoint_manifest": v5_manifest,
        "r7_checkpoint_manifest": r7_manifest, "r7_fit_receipt": r7_receipt,
        "patient_level_output_emitted": False, "candidate_promoted": False}
    write_json(out / "protocol.json", protocol); protocol_pin = sha(out / "protocol.json")
    _, v5_private = v5.paths("fit", 2)
    _, r7_private = r7.paths("fit", 1)

    def provider(role: str, fold: int):
        require(role in ("V5", "R7") and type(fold) is int and fold in range(5))
        if role == "V5":
            item = v5_components[("M", fold)]
            require({"fold": fold, "checkpoint_sha256": item["checkpoint_sha256"], "binding": item["binding"]}
                    == v5_manifest["folds"][fold])
            return v5.oldfit.load_checkpoint(v5_private / f"fold{fold}_M.pt", item["checkpoint_sha256"], item["binding"])
        item = r7_components[("R", fold)]
        binding = item["binding"]
        require({"fold": fold, "checkpoint_sha256": item["checkpoint_sha256"], "binding": binding}
                == r7_manifest["folds"][fold])
        return r7.load_checkpoint(r7_private / f"fold{fold}_R.pt", item["checkpoint_sha256"], binding)

    result = evaluation.evaluate(sources.paired, provider, lambda p, f=None, r=None: progress(out, state, p, f, r))
    progress(out, state, "post_authentication")
    post_baseline, post_v5_protocol, _, post_v5_components = parent.small_authentication()
    post_r7_protocol, _, post_r7_components, post_r7_receipt = authenticate_r7("fit", 1)
    require(post_baseline == baseline and sha(parent.BASELINE) == baseline_pin
            and post_v5_protocol["source_binding"] == source_receipt
            )
    require_checkpoint_manifest_match(post_v5_components, "M", source_receipt, v5_manifest)
    require(post_r7_protocol["source_binding"] == source_receipt and post_r7_receipt == r7_receipt)
    require_checkpoint_manifest_match(post_r7_components, "R", source_receipt, r7_manifest)
    require(
            load_bound_sources().receipt() == source_receipt and code_hashes() == protocol["code_sha256"]
            and sha(out / "protocol.json") == protocol_pin and result["encoder_updated"] is False
            and result["patient_level_output_emitted"] is False)
    write_json(out / "aggregate.json", result); progress(out, state, "completed")
    completed = out / "completed.json"
    write_json(completed, {"status": "authenticated_completed", "protocol_sha256": protocol_pin,
        "aggregate_sha256": sha(out / "aggregate.json"), "patient_level_output_emitted": False})
    try:
        authenticate(attempt)
    except Exception:
        # A just-created terminal that fails its own bounded audit is not a
        # success receipt.  Remove only that fresh file so main records the
        # closed failure terminal for this exclusive attempt directory.
        completed.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args(); state = {"out": None, "phase": "authentication"}; ok = False
    with quiet():
        try:
            with LOCK.open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                import torch
                torch.set_num_threads(2); run(args.attempt, state); ok = True
        except Exception as exc:
            out = state["out"]
            if out is not None and not (out / "completed.json").exists():
                from run_bran_source_pattern_v6 import safe_site
                write_json(out / "failure.json", {"status": "technical_failure", "phase": state["phase"],
                    "safe_code_site": safe_site(exc), "patient_level_output_emitted": False,
                    "candidate_promoted": False})
    print(json.dumps({"status": "completed" if ok else "not_completed", "phase": state["phase"],
                      "patient_level_output_emitted": False, "candidate_promoted": False}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
