"""Frozen native-head input-removal diagnostic; patient work stays FD-silenced."""
import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import exclusive_json, sha
import bran_missingness_stress_v1 as masking
import run_bran_raw_teacher_distillation_v1 as prior

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "BRAN_MISSINGNESS_STRESS_PROTOCOL_V1.json"
OUT = ROOT / "BRAN_MISSINGNESS_STRESS_V1"
AUDIT = ROOT / "BRAN_MISSINGNESS_STRESS_AUDIT_V1"
PARAMETERS = {
    "patterns": list(masking.PATTERNS), "mask_seed": masking.MASK_SEED,
    "bootstrap_seed": 91501, "bootstrap_draws": 1000, "minimum_valid_draws": 900,
    "minimum_release_support": 20, "training_steps": 0, "state_width": 192,
    "age_remains_available": True, "head_refit": False,
    "mask_realizations": 1, "replay_tolerance": 1e-10,
    "clinical_use": False, "model_promoted": False,
    "official_test_used": False, "adaptive_development": True,
    "patient_level_output_emitted": False,
}
CODE = (
    "run_bran_missingness_stress_v1.py", "test_run_bran_missingness_stress_v1.py",
    "bran_missingness_stress_v1.py", "test_bran_missingness_stress_v1.py",
    "bran_missingness_stress_metrics_v1.py", "test_bran_missingness_stress_metrics_v1.py",
    "BRAN_MISSINGNESS_STRESS_DESIGN_V1.md", "run_bran_raw_teacher_distillation_v1.py",
)


def require(ok, code="missingness_stress_contract_failed"):
    if not ok:
        raise ValueError(code)


def prepare():
    source = prior.prepare()
    return {
        "schema": "bran-missingness-stress-protocol-v1", "status": "frozen_before_execution",
        "parameters": PARAMETERS, "native_source": source["native_source"],
        "code_sha256": {**source["code_sha256"], **prior.native.source.io.code_closure(CODE)},
        "runtime": prior.native.source.io.runtime(),
    }


def validate_protocol(p):
    require(p == prepare(), "missingness_stress_protocol_changed")


def parameter_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def replay(predictions, labels, observed, folds, names):
    reference = json.loads((prior.native.OUT / "aggregate.json").read_text())
    common = (np.isfinite(predictions["available"]).all(1)
              & np.isfinite(predictions["no_retina"]).all(1)
              & np.isfinite(predictions["all_clinical_hidden"]).all(1))
    for j, name in enumerate(names):
        for pattern, route in (("available", "both"), ("no_retina", "clinical"),
                               ("all_clinical_hidden", "retinal")):
            value = prior.native.source.base.fold_weighted_auc(
                labels[:, j], predictions[pattern][:, j], observed[:, j] & common, folds)
            expected = reference["results"]["endpoints"][name]["arms"]["native_" + route]["auroc"]
            require(np.isfinite(value) and abs(value - expected) <= 1e-10,
                    "missingness_stress_baseline_replay_failed")


def run(p):
    import torch
    import bran_missingness_stress_metrics_v1 as metrics
    from bran_clinical_semantics_v1 import CBC_FIELDS

    torch.set_num_threads(2)
    source = prior.native.source
    ctx, folds, c0, cm0, eligible, r0, rm, ages, field_names = source.io.load_context()
    require(len(folds) == 1928 and set(np.unique(folds)) == set(range(5)))
    names = p["native_source"]["source"]["endpoint_names"]
    slots = tuple(field_names.index(field) for field in CBC_FIELDS)
    labels = np.column_stack([ctx["labels_by_source"][name] for name in names])
    observed = np.column_stack([ctx["observed_by_source"][name] for name in names]).astype(bool)
    predictions = {pattern: np.full((len(folds), 26), np.nan) for pattern in masking.PATTERNS}
    authentication = p["native_source"]["source"]["authentication"]
    for fold in range(5):
        source.base._atomic_progress(OUT / "progress.json", "frozen_missingness_inference", fold)
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        _, identity = source.base._inner_context(ctx, train, fold)
        require(identity == authentication["inner_fold_sha256"][fold], "missingness_inner_fold_changed")
        transform = source.base.FoldTransform(c0, cm0, eligible, r0, rm, ages, train)
        c, cm, r, age = transform.apply(c0, cm0, eligible, r0, rm, ages)
        model = prior.load_initial(fold, transform, p)
        original = parameter_digest(model)
        for pattern in masking.PATTERNS:
            # Mask the full authenticated ordering before test slicing, keeping
            # each person's simulated mask identical across folds/severities.
            x = masking.remove_inputs(c, cm, r, rm, slots, pattern)
            masking.assert_no_input_leak(x, cm, rm, slots, pattern)
            result = prior.native.kernel.predict_native(
                model, x.clinical[test], x.clinical_mask[test], x.retinal[test],
                x.retinal_mask[test], age[test])["both"]
            require(np.array_equal(np.isfinite(result).all(1), x.available[test]),
                    "missingness_abstention_contract_failed")
            predictions[pattern][test] = result
        require(parameter_digest(model) == original, "missingness_inference_changed_parameters")
    replay(predictions, labels, observed, folds, names)
    source.base._atomic_progress(OUT / "progress.json", "aggregate_bootstrap")
    counts = source.ev.paired_counts(folds, draws=1000, seed=91501)
    results = metrics.summarize(predictions, labels, observed, folds, names, counts)
    return {
        "schema": "bran-missingness-stress-aggregate-v1", "status": "completed",
        "results": results, "paired_people": 1928, "recorded_conditions": 26,
        "baseline_three_routes_reproduced": True, "original_parameters_unchanged": True,
        "fold_authentication": authentication, "training_steps": 0,
        "state_width": 192, "clinical_use": False, "model_promoted": False,
        "official_test_used": False, "adaptive_development": True,
        "patient_level_output_emitted": False,
    }


def validate_result(a, p):
    import bran_missingness_stress_metrics_v1 as metrics
    keys = {"schema", "status", "results", "paired_people", "recorded_conditions",
            "baseline_three_routes_reproduced", "original_parameters_unchanged",
            "fold_authentication", "training_steps", "state_width", "clinical_use",
            "model_promoted", "official_test_used", "adaptive_development", "patient_level_output_emitted"}
    require(type(a) is dict and set(a) == keys)
    require(a["schema"] == "bran-missingness-stress-aggregate-v1" and a["status"] == "completed")
    for key, value in (("paired_people", 1928), ("recorded_conditions", 26),
                       ("state_width", 192), ("training_steps", 0)):
        require(type(a[key]) is int and a[key] == value)
    require(all(a[k] is True for k in ("baseline_three_routes_reproduced", "original_parameters_unchanged", "adaptive_development")))
    require(all(a[k] is False for k in ("clinical_use", "model_promoted", "official_test_used", "patient_level_output_emitted")))
    require(a["fold_authentication"] == p["native_source"]["source"]["authentication"])
    metrics.validate_result(a["results"], p["native_source"]["source"]["endpoint_names"])


def audit(p, pin):
    require(not (OUT / "failure.json").exists(), "missingness_terminal_conflict")
    a = json.loads((OUT / "aggregate.json").read_text())
    m = json.loads((OUT / "manifest.json").read_text())
    require(set(m) == {"protocol_sha256", "aggregate_sha256", "elapsed_seconds", "patient_level_output_emitted"})
    require(m["protocol_sha256"] == pin and m["aggregate_sha256"] == sha(OUT / "aggregate.json"))
    require(m["patient_level_output_emitted"] is False and type(m["elapsed_seconds"]) in (float, int)
            and np.isfinite(m["elapsed_seconds"]) and m["elapsed_seconds"] >= 0)
    validate_protocol(p); validate_result(a, p)
    return {"schema": "bran-missingness-stress-audit-v1", "status": "authenticated",
            "protocol_sha256": pin, "aggregate_sha256": sha(OUT / "aggregate.json"),
            "manifest_sha256": sha(OUT / "manifest.json"),
            "patient_level_output_emitted": False, "training_steps": 0}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for operation in ("prepare", "run", "audit"):
        group.add_argument("--" + operation, action="store_true")
    parser.add_argument("--protocol-sha256"); args = parser.parse_args()
    ok, owned, phase = False, False, "protocol"
    destination = AUDIT if args.audit else OUT
    start = time.monotonic()
    with _quiet():
        try:
            if args.prepare:
                require(not OUT.exists() and not AUDIT.exists())
                exclusive_json(PROTOCOL, prepare()); ok = True
            else:
                require(args.protocol_sha256 and sha(PROTOCOL) == args.protocol_sha256)
                p = json.loads(PROTOCOL.read_text()); validate_protocol(p)
                destination.mkdir(); owned = True
                if args.run:
                    with open("/private/tmp/bran_missingness_stress_v1.lock", "a") as lock:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        phase = "local_inference_and_aggregation"; a = run(p)
                        phase = "terminal_validation"; validate_protocol(p); validate_result(a, p)
                        exclusive_json(OUT / "aggregate.json", a)
                        exclusive_json(OUT / "manifest.json", {
                            "protocol_sha256": args.protocol_sha256,
                            "aggregate_sha256": sha(OUT / "aggregate.json"),
                            "elapsed_seconds": round(time.monotonic() - start, 1),
                            "patient_level_output_emitted": False})
                        prior.native.source.base._atomic_completed(OUT / "progress.json")
                else:
                    phase = "audit"; result = audit(p, args.protocol_sha256)
                    exclusive_json(AUDIT / "audit.json", result)
                ok = True
        except Exception as error:
            if owned:
                exclusive_json(destination / "failure.json", {
                    "status": "execution_failed", "phase": phase,
                    "error_class": type(error).__name__ if type(error) in (ValueError, TypeError, KeyError, RuntimeError, OSError, ImportError) else "other",
                    "patient_level_output_emitted": False})
    print(json.dumps({"operation": "prepare" if args.prepare else "audit" if args.audit else "run",
                      "status": "completed" if ok else "execution_failed", "patient_level_output_emitted": False}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
