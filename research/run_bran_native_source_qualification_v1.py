"""Read-only local qualification of existing clinical caches for native BRAN.

No fitting, predictions, raw event scan or row-level artifact is produced.
All cache access is invoked inside the FD-quiet CLI boundary.
"""
import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from bran_joint_lab_cache_v1 import coarse_count
from run_bran_source_linkage_audit_v1 import exclusive_json, sha
import run_bran_joint_lab_comparison_v1 as old

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "BRAN_NATIVE_SOURCE_QUALIFICATION_PROTOCOL_V1.json"
OUT = ROOT / "BRAN_NATIVE_SOURCE_QUALIFICATION_V1"
AUDIT = ROOT / "BRAN_NATIVE_SOURCE_QUALIFICATION_AUDIT_V1"
TASKS = {"mimic": "whole_cbc", "nhanes": "whole_cbc", "eicu": "partial_cbc"}
CODE = ("run_bran_native_source_qualification_v1.py", "test_run_bran_native_source_qualification_v1.py",
        "bran_native_external_source_v1.py", "test_bran_native_external_source_v1.py",
        "bran_external_native_age_v1.py", "test_bran_external_native_age_v1.py",
        "bran_joint_lab_pretraining_v1.py", "bran_joint_lab_task_contract_v1.py",
        "bran_external_cbc_pretraining_v1.py", "run_bran_joint_lab_comparison_v1.py",
        "audit_bran_joint_lab_cache_v1.py", "bran_clinical_dictionary_binding_v1.py",
        "run_bran_source_linkage_audit_v1.py", "bran_joint_lab_cache_v1.py",
        "BRAN_NATIVE_SOURCE_QUALIFICATION_DESIGN_V1.md")
POLICY = {"split": 0, "cbc_minimum_observed": 2, "whole_cbc_requires_chemistry": True,
          "age_kinds": ["reported_year", "year_derived"], "minimum_age_lower_bound": 18,
          "count_lower_bound_rounding": 20, "minimum_distinct_people_to_release": 20,
          "one_counted_task_per_source": TASKS, "training_steps": 0,
          "original_events_rescanned": False, "patient_level_output_emitted": False}
FLAGS = {"training_started": False, "model_benefit_established": False,
         "patient_level_output_emitted": False, "clinical_use": False,
         "cross_source_identity_resolved": False, "source_caches_unchanged": True}


def require(ok):
    if not ok:
        raise ValueError("native_source_qualification_contract_failed")


def prepare():
    receipts = {s: old.source_receipt(s) for s in TASKS}
    code_names = set(CODE)
    for source in TASKS:
        module = importlib.import_module(old.SOURCE_SPECS[source][0])
        protocol = json.loads(module.PROTOCOL.read_text())
        code_names.update(protocol["code_sha256"])
    return {"schema": "bran-native-source-qualification-protocol-v1", "status": "frozen_before_execution",
            "policy": POLICY, "sources": receipts, "runtime": old.runtime(),
            "code_sha256": {n: sha(ROOT / n) for n in sorted(code_names)}}


def validate_protocol(p):
    require(p == prepare())


def load_one(source, receipt):
    """Private local arrays, never return this function's result to hosted tools."""
    from audit_bran_joint_lab_cache_v1 import check_arrays
    from bran_native_external_source_v1 import adapt_source
    require(source in TASKS and old.source_receipt(source) == receipt)
    module = importlib.import_module(old.SOURCE_SPECS[source][0])
    cache = module.PRIVATE / "observations.npz"
    with np.load(cache, allow_pickle=False) as handle:
        arrays = {k: handle[k] for k in handle.files}
    aggregate = json.loads((module.PUBLIC / "aggregate.json").read_text())
    module.validate_aggregate(aggregate)
    if source == "eicu":
        module.audit_private_arrays(arrays, aggregate["base_pool_aggregate"]["counts_lower_bounds_20"])
    else:
        check_arrays(arrays, source, aggregate["counts_lower_bounds_20"])
    adapted = adapt_source(source, arrays)
    require(sha(cache) == receipt["private_cache_sha256"])
    return adapted


def summarize_one(source, arrays):
    """One prespecified task count per source; no exclusion/age-cell breakdown."""
    task = TASKS[source]
    take = arrays[task + "_eligible"]
    people = len(np.unique(arrays["person_group"][take]))
    snapshots = int(take.sum())
    supported = people >= POLICY["minimum_distinct_people_to_release"]
    return {"counted_task": task, "status": "qualified_pool" if supported else "suppressed_insufficient_support",
            "eligible_training_snapshots_lower_bound": coarse_count(snapshots) if supported else None,
            "eligible_source_local_people_lower_bound": coarse_count(people) if supported else None}


def run(p):
    sources = {}
    for source in TASKS:
        arrays = load_one(source, p["sources"][source])
        sources[source] = summarize_one(source, arrays)
        del arrays
    return {"schema": "bran-native-source-qualification-aggregate-v1", "status": "completed",
            "sources": sources, "source_receipts": p["sources"], "training_steps": 0, **FLAGS}


def validate_result(a, p):
    require(type(a) is dict and set(a) == {"schema", "status", "sources", "source_receipts", "training_steps"} | set(FLAGS))
    require(a["schema"] == "bran-native-source-qualification-aggregate-v1" and a["status"] == "completed")
    require(type(a["training_steps"]) is int and a["training_steps"] == 0)
    require(all(a[k] is v for k, v in FLAGS.items()))
    require(a["source_receipts"] == p["sources"] and type(a["sources"]) is dict and set(a["sources"]) == set(TASKS))
    for source, cell in a["sources"].items():
        require(type(cell) is dict and set(cell) == {"counted_task", "status", "eligible_training_snapshots_lower_bound", "eligible_source_local_people_lower_bound"})
        require(cell["counted_task"] == TASKS[source])
        snapshots, people = cell["eligible_training_snapshots_lower_bound"], cell["eligible_source_local_people_lower_bound"]
        if cell["status"] == "qualified_pool":
            require(all(type(v) is int and v >= 20 and v % 20 == 0 for v in (snapshots, people)))
            require(snapshots >= people)
        else:
            require(cell["status"] == "suppressed_insufficient_support" and snapshots is None and people is None)


def audit(p, pin):
    require(not (OUT / "failure.json").exists())
    a = json.loads((OUT / "aggregate.json").read_text())
    manifest = json.loads((OUT / "manifest.json").read_text())
    require(set(manifest) == {"protocol_sha256", "aggregate_sha256", "elapsed_seconds", "patient_level_output_emitted"})
    require(manifest["protocol_sha256"] == pin and manifest["aggregate_sha256"] == sha(OUT / "aggregate.json"))
    require(manifest["patient_level_output_emitted"] is False)
    require(type(manifest["elapsed_seconds"]) in (int, float) and np.isfinite(manifest["elapsed_seconds"]) and manifest["elapsed_seconds"] >= 0)
    validate_result(a, p)
    # Recompute only this small cache qualification, not prior training/experiments.
    require(run(p) == a)
    validate_protocol(p)
    return {"schema": "bran-native-source-qualification-audit-v1", "status": "authenticated",
            "protocol_sha256": pin, "aggregate_sha256": sha(OUT / "aggregate.json"),
            "manifest_sha256": sha(OUT / "manifest.json"), "counts_recomputed_locally": True,
            "patient_level_output_emitted": False}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for operation in ("prepare", "run", "audit"):
        group.add_argument("--" + operation, action="store_true")
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    ok, owned, phase, start = False, False, "protocol", time.monotonic()
    destination = AUDIT if args.audit else OUT
    with _quiet():
        try:
            if args.prepare:
                require(not OUT.exists() and not AUDIT.exists())
                exclusive_json(PROTOCOL, prepare())
            else:
                require(args.protocol_sha256 and sha(PROTOCOL) == args.protocol_sha256)
                p = json.loads(PROTOCOL.read_text())
                validate_protocol(p)
                destination.mkdir()
                owned = True
                if args.run:
                    phase = "source_qualification"
                    a = run(p)
                    phase = "terminal_validation"
                    validate_result(a, p)
                    validate_protocol(p)
                    exclusive_json(OUT / "aggregate.json", a)
                    exclusive_json(OUT / "manifest.json", {"protocol_sha256": args.protocol_sha256,
                        "aggregate_sha256": sha(OUT / "aggregate.json"), "elapsed_seconds": round(time.monotonic() - start, 1),
                        "patient_level_output_emitted": False})
                else:
                    phase = "audit"
                    exclusive_json(AUDIT / "audit.json", audit(p, args.protocol_sha256))
            ok = True
        except Exception as error:
            if owned:
                exclusive_json(destination / "failure.json", {"status": "execution_failed", "phase": phase,
                    "error_class": type(error).__name__ if type(error) in (ValueError, TypeError, KeyError, RuntimeError, OSError, ImportError) else "other",
                    "patient_level_output_emitted": False})
    print(json.dumps({"operation": "prepare" if args.prepare else "audit" if args.audit else "run",
                      "status": "completed" if ok else "execution_failed", "patient_level_output_emitted": False}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
