"""One native external-task rehearsal comparison; private work is FD-quiet."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import exclusive_json, sha
import run_bran_raw_teacher_distillation_v1 as origin
import run_bran_native_source_qualification_v1 as qualification
import bran_distillation_metrics_v1 as metrics
import bran_missingness_stress_metrics_v1 as stress_metrics
import bran_missingness_stress_v1 as masking

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "BRAN_NATIVE_REHEARSAL_PROTOCOL_V1.json"
OUT = ROOT / "BRAN_NATIVE_REHEARSAL_V1"
AUDIT = ROOT / "BRAN_NATIVE_REHEARSAL_AUDIT_V1"
PRIVATE = ROOT / "private_artifacts/bran_native_rehearsal_v1"
QUAL_PIN = "3e6b2f101d897526dd71dd8a599867bc2477cb3e04c15399c844aedea11633be"
QUAL_AUDIT = "c7d2f9033d4d39247b269f603a382794aecd13749eaf12a155b49d7e645e1180"
PATTERNS = ("single_target_hidden", "whole_cbc_hidden")
FLAGS = {"patient_level_output_emitted": False, "clinical_use": False, "automatic_promotion": False,
         "official_test_used": False, "adaptive_development": True, "new_subtype_claim": False,
         "reload_predictions_equal": True, "baseline_replay_passed": True}
PARAMETERS = {"state_width": 192, "steps_per_arm_fold": 1500, "batch_size": 96, "seed_base": 94101,
    "learning_rate": .0001, "weight_decay": .0001, "gradient_clip_norm": 5., "torch_threads": 2,
    "native_heads_retained": True, "external_weights": {"continued": 0., "student": .5},
    "student_definition": "native external-task rehearsal; no distillation or external disease labels",
    "common_losses": {"generative": 1., "screening": 1., "whole_cbc": .5, "preservation": .1},
    "preservation_teacher": "frozen initial native model, same visible clinical mask and age",
    "external_tasks": "alternate wholeCBC/partialCBC; source then person then episode balanced",
    "normalization": "recipient outer-training fold only, including scalar age",
    "screening_empty_input": "exclude abstained states in both new training arms",
    "paired_screening_routes": ["both", "both", "both", "clinical", "retinal"],
    "mask_seed": 93711, "bootstrap_draws": 1000, "bootstrap_seed": 91501,
    "minimum_release_support": 20, "minimum_valid_draws": 900,
    "cbc_patterns": list(PATTERNS), "missingness_patterns": list(masking.PATTERNS),
    "tail_quantiles": [.1, .9], "tail_small_cell_policy": "suppress all three tails if any tail has fewer than20",
    "full_screening": "common initial nonabstaining three-route scope; all26 conditions",
    "baseline_replay_tolerance": 1e-10, "automatic_promotion": False}
CODE = ("run_bran_native_rehearsal_v1.py", "test_run_bran_native_rehearsal_v1.py",
        "bran_native_rehearsal_kernel_v1.py", "test_bran_native_rehearsal_kernel_v1.py",
        "bran_native_rehearsal_batches_v1.py", "test_bran_native_rehearsal_batches_v1.py",
        "BRAN_NATIVE_REHEARSAL_DESIGN_V1.md", "bran_missingness_stress_v1.py",
        "bran_missingness_stress_metrics_v1.py", "test_bran_missingness_stress_metrics_v1.py")


def require(ok):
    if not ok: raise ValueError("native_rehearsal_contract_failed")


def prepare():
    base = origin.prepare()
    require(sha(qualification.PROTOCOL) == QUAL_PIN and sha(qualification.AUDIT / "audit.json") == QUAL_AUDIT)
    require(not (qualification.OUT / "failure.json").exists() and not (qualification.AUDIT / "failure.json").exists())
    qp = json.loads(qualification.PROTOCOL.read_text()); qualification.validate_protocol(qp)
    qa = json.loads((qualification.OUT / "aggregate.json").read_text()); qualification.validate_result(qa, qp)
    audit = json.loads((qualification.AUDIT / "audit.json").read_text())
    require(audit["aggregate_sha256"] == sha(qualification.OUT / "aggregate.json")
            and audit["manifest_sha256"] == sha(qualification.OUT / "manifest.json")
            and audit["protocol_sha256"] == QUAL_PIN and audit["status"] == "authenticated")
    require(all(cell["status"] == "qualified_pool" for cell in qa["sources"].values()))
    code = {**base["code_sha256"], **qp["code_sha256"], **{n: sha(ROOT / n) for n in CODE}}
    return {"schema": "bran-native-rehearsal-protocol-v1", "status": "frozen_before_execution",
            "parameters": PARAMETERS, "native_source": base["native_source"],
            "native_aggregate_sha256": base["native_aggregate_sha256"],
            "native_audit_sha256": base["native_audit_sha256"], "raw_reference_sha256": base["raw_reference_sha256"],
            "qualification_protocol_sha256": QUAL_PIN, "qualification_audit_sha256": QUAL_AUDIT,
            "sources": qp["sources"], "joint_registry_indices": qualification.old.joint_units(),
            "code_sha256": code, "runtime": base["runtime"]}


def validate_protocol(p): require(p == prepare())


def cbc_prediction(model, c, cm, r, rm, age, slots, transform, pattern):
    import torch
    require(pattern in PATTERNS)
    source = origin.native.source
    output = np.full((len(c), 9), np.nan)
    support = np.zeros((len(c), 9), bool)
    targets = range(9) if pattern == "single_target_hidden" else (0,)
    with torch.no_grad():
        for j in targets:
            hidden, hm = source.mask_inputs(c, cm, slots, pattern, slots[j])
            masked_slots = (slots[j],) if pattern == "single_target_hidden" else slots
            require(not hidden[:, masked_slots].any() and not hm[:, masked_slots].any())
            z = source.lineage._state_routes(model, hidden, hm, r, rm, age)["both"]
            pred = model.cbc_joint_head(torch.tensor(z, dtype=torch.float32)).numpy()
            pred = pred * transform.clinical_iqr[list(slots)] + transform.clinical_median[list(slots)]
            valid = hm.any(1) | rm
            require(np.isfinite(pred[valid]).all())
            if pattern == "whole_cbc_hidden":
                output[valid] = pred[valid]; support[:] = valid[:, None]
            else:
                output[valid, j] = pred[valid, j]; support[:, j] = valid
    return output, support


def safe_tail_groups(observed, groups):
    result = {k: v.copy() for k, v in groups.items()}
    for j in range(9):
        if any(np.count_nonzero(observed[:, j] & result[g][:, j]) < 20 for g in ("low", "middle", "high")):
            for g in ("low", "middle", "high"): result[g][:, j] = False
    return result


def screen_masks(pred, observed, endpoints, labels, folds):
    masks = origin.common_masks(pred, observed, endpoints)
    for e in endpoints:
        valid = masks[e]; keep = np.zeros_like(valid)
        for fold in range(5):
            local = valid & (folds == fold)
            if np.any(local & (labels[e] == 0)) and np.any(local & (labels[e] == 1)): keep |= local
        masks[e] = keep
    return masks


def decisions(screen, completion, missingness):
    primary = metrics.decisions(screen, completion["whole_cbc_hidden"])
    checks = {"primary_screening_and_priority_whole_cbc": primary["advancement_supported"]}
    for pattern in PATTERNS:
        fields = metrics.CBC_FIELDS if pattern == "whole_cbc_hidden" else ("hemoglobin", "plt", "wbc")
        checks[pattern + "_point_nonworse"] = all(
            completion[pattern][field]["overall"]["status"] == "supported" and
            all(completion[pattern][field]["overall"]["contrasts"][ref]["delta"] <= 0 for ref in ("initial", "continued"))
            for field in fields)
    checks["missingness_point_nonworse"] = all(
        missingness[version][pattern]["macro"] is not None and missingness["student"][pattern]["macro"] is not None
        and missingness["student"][pattern]["macro"]["masked"]["auroc"] >= missingness[version][pattern]["macro"]["masked"]["auroc"]
        for version in ("initial", "continued") for pattern in masking.PATTERNS)
    return {"checks": checks, "advancement_supported": bool(all(checks.values())), "automatic_promotion": False}


def validate_result(a, p):
    require(type(a) is dict and set(a) == {"schema", "status", "screening", "completion", "missingness", "decisions",
            "state_width", "paired_people", "recorded_conditions", "fold_authentication"} | set(FLAGS))
    require(a["schema"] == "bran-native-rehearsal-aggregate-v1" and a["status"] == "completed")
    require(all(a[k] is v for k, v in FLAGS.items()))
    for k, v in (("state_width", 192), ("paired_people", 1928), ("recorded_conditions", 26)):
        require(type(a[k]) is int and a[k] == v)
    require(a["fold_authentication"] == p["native_source"]["source"]["authentication"])
    require(type(a["completion"]) is dict and set(a["completion"]) == set(PATTERNS))
    names = p["native_source"]["source"]["endpoint_names"]
    for pattern in PATTERNS:
        metrics.validate(a["screening"], a["completion"][pattern], metrics.decisions(a["screening"], a["completion"][pattern]), names)
    require(type(a["missingness"]) is dict and set(a["missingness"]) == set(metrics.VERSIONS))
    for item in a["missingness"].values(): stress_metrics.validate_result(item, names)
    require(type(a["decisions"]) is dict and set(a["decisions"]) == {"checks", "advancement_supported", "automatic_promotion"})
    require(type(a["decisions"]["checks"]) is dict and all(type(v) is bool for v in a["decisions"]["checks"].values()))
    require(type(a["decisions"]["advancement_supported"]) is bool and a["decisions"]["automatic_promotion"] is False)
    require(a["decisions"] == decisions(a["screening"], a["completion"], a["missingness"]))


def validate_bundle(bundle, original, p, fold):
    import torch
    require(type(bundle) is dict and set(bundle) == set(origin.NORMALIZERS) |
        {"continued", "student", "protocol_sha256", "initial_checkpoint_sha256", "fold", "endpoint_names", "cbc_fields"})
    require(bundle["protocol_sha256"] == sha(PROTOCOL) and type(bundle["fold"]) is int and bundle["fold"] == fold)
    require(bundle["initial_checkpoint_sha256"] == p["native_source"]["checkpoint_sha256"]["fold" + str(fold)])
    require(bundle["endpoint_names"] == p["native_source"]["source"]["endpoint_names"] and bundle["cbc_fields"] == list(metrics.CBC_FIELDS))
    for key in origin.NORMALIZERS: require(np.array_equal(bundle[key], original[key]))
    expected = original["candidate"]
    for version in ("continued", "student"):
        require(set(bundle[version]) == set(expected))
        for key, value in bundle[version].items():
            require(isinstance(value, torch.Tensor) and value.shape == expected[key].shape
                    and value.dtype == expected[key].dtype and bool(torch.isfinite(value).all()))


def run(p):
    import torch
    from bran_matched_screening_kernel_v1 import fit_predict, late_fusion_average
    from bran_native_rehearsal_batches_v1 import prepare_sources
    from bran_native_rehearsal_kernel_v1 import adapt
    source = origin.native.source; torch.set_num_threads(2)
    ctx, folds, c0, cm0, eligible, r0, rm, ages, names = source.io.load_context()
    require(len(folds) == 1928 and set(np.unique(folds)) == set(range(5)))
    endpoints = p["native_source"]["source"]["endpoint_names"]
    slots = tuple(names.index(f) for f in metrics.CBC_FIELDS)
    require(all(names[j] == field for field, j in p["joint_registry_indices"].items()))
    labels = np.column_stack([ctx["labels_by_source"][e] for e in endpoints])
    lm = np.column_stack([ctx["observed_by_source"][e] for e in endpoints]).astype(bool)
    target = c0[:, slots].copy(); observed = (cm0 & eligible)[:, slots].copy()
    pred = {e: {a: np.full(len(folds), np.nan) for a in metrics.S_ARMS} for e in endpoints}
    cpred = {pattern: {a: np.full(target.shape, np.nan) for a in metrics.C_ARMS} for pattern in PATTERNS}
    cobs = {pattern: observed.copy() for pattern in PATTERNS}
    groups = {pattern: {g: np.zeros(target.shape, bool) for g in metrics.GROUPS} for pattern in PATTERNS}
    stress = {v: {pattern: np.full(labels.shape, np.nan) for pattern in masking.PATTERNS} for v in metrics.VERSIONS}
    cache = []; old_raw = origin.previous.baseline()
    old_native = json.loads((origin.native.OUT / "aggregate.json").read_text())
    old_cbc = json.loads((source.OUT / "aggregate.json").read_text())["retained_cbc_head_whole_panel"]
    for fold in range(5):
        source.base._atomic_progress(OUT / "progress.json", "baseline_replay", fold)
        tr, te = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        transform = source.base.FoldTransform(c0, cm0, eligible, r0, rm, ages, tr)
        c, cm, r, age = transform.apply(c0, cm0, eligible, r0, rm, ages)
        initial = origin.load_initial(fold, transform, p)
        inner, identity = source.base._inner_context(ctx, tr, fold)
        require(identity == p["native_source"]["source"]["authentication"]["inner_fold_sha256"][fold])
        full_inner = np.full(len(folds), -1, int); full_inner[tr] = inner
        ip = origin.native.kernel.predict_native(initial, c, cm, r, rm, age)
        raw = {"raw_clinical": np.c_[c, cm, age], "raw_retinal": np.c_[r, rm, age], "raw_concat": np.c_[c, cm, r, rm, age]}
        for j, e in enumerate(endpoints):
            for route in metrics.ROUTES: pred[e]["initial_" + route][te] = ip[route][te, j]
            for arm in raw:
                pred[e][arm][te], _ = fit_predict(raw[arm], labels[:, j], lm[:, j], tr, te, full_inner,
                                                family="extra_trees", seed=92381 + fold)
            pred[e]["late_average"][te] = late_fusion_average(pred[e]["raw_clinical"][te], pred[e]["raw_retinal"][te])
        for pattern in PATTERNS:
            initial_cbc, support = cbc_prediction(initial, c, cm, r, rm, age, slots, transform, pattern)
            cpred[pattern]["initial"][te] = initial_cbc[te]; cobs[pattern][te] &= support[te]
            for j in range(9):
                hc, hm = source.mask_inputs(c, cm, slots, pattern, slots[j])
                valid = observed[:, j] & support[:, j]
                for arm, x in {"raw_clinical": np.c_[hc, hm, age], "raw_concat": np.c_[hc, hm, r, rm, age]}.items():
                    cpred[pattern][arm][te, j] = source.ev.fixed_cbc_probe(x, target[:, j], valid, tr, te)
                train_y = target[tr, j][valid[tr]]; require(len(train_y) >= 20)
                groups[pattern]["overall"][te, j] = True
                for group, mask in origin.previous.tail_masks(train_y, target[te, j]).items():
                    groups[pattern][group][te, j] = mask
        cache.append((tr, te, transform, initial, c, cm, r, age, ip))
    masks = screen_masks(pred, ctx["observed_by_source"], endpoints, ctx["labels_by_source"], folds)
    for e in endpoints:
        for route in metrics.ROUTES:
            point = source.base.fold_weighted_auc(ctx["labels_by_source"][e], pred[e]["initial_" + route], masks[e], folds)
            require(abs(point - old_native["results"]["endpoints"][e]["arms"]["native_" + route]["auroc"]) <= 1e-10)
        for arm in ("raw_clinical", "raw_retinal", "raw_concat", "late_average"):
            point = source.base.fold_weighted_auc(ctx["labels_by_source"][e], pred[e][arm], ctx["observed_by_source"][e], folds)
            require(abs(point - old_raw["endpoints"][e]["arms"][arm]["auroc"]) <= 1e-10)
    for j, field in enumerate(metrics.CBC_FIELDS):
        valid = cobs["whole_cbc_hidden"][:, j]
        errors = cpred["whole_cbc_hidden"]["initial"][valid, j] - target[valid, j]
        require(old_cbc[field]["status"] == "complete" and np.isclose(np.abs(errors).mean(), old_cbc[field]["mae"], rtol=1e-10, atol=1e-10)
                and np.isclose(np.square(errors).mean(), old_cbc[field]["mse"], rtol=1e-10, atol=1e-10))
    exclusive_json(OUT / "baseline_replay.json", {"status": "passed", "native_screen_all26_three_routes": True,
        "raw_screen_all26_four_arms": True, "native_wholeCBC_all9": True, "patient_level_output_emitted": False})
    external = {s: qualification.load_one(s, p["sources"][s]) for s in qualification.TASKS}
    checkpoints = {}
    for fold, (tr, te, transform, initial, c, cm, r, age, ip) in enumerate(cache):
        source.base._atomic_progress(OUT / "progress.json", "external_batch_preparation", fold)
        prepared = prepare_sources(external, tuple(names), transform.clinical_median, transform.clinical_iqr,
                                   transform.age_mean, transform.age_scale)
        models = {"initial": initial}
        for version, weight in (("continued", 0.), ("student", .5)):
            source.base._atomic_progress(OUT / "progress.json", "adapt_" + version, fold)
            def progress(step):
                source.base._atomic_progress(OUT / "progress.json", "adapt_" + version + "_step" + str(step), fold)
            models[version] = adapt(initial, c, cm, r, rm, age, labels, lm, tr, slots, prepared,
                seed=94101 + fold, external_weight=weight, steps=1500, batch_size=96, progress=progress)
        path = PRIVATE / ("fold" + str(fold) + ".pt")
        bundle = {v: models[v].state_dict() for v in ("continued", "student")}
        bundle.update({k: getattr(transform, k) for k in origin.NORMALIZERS})
        bundle.update(protocol_sha256=sha(PROTOCOL), initial_checkpoint_sha256=p["native_source"]["checkpoint_sha256"]["fold" + str(fold)],
                      fold=fold, endpoint_names=endpoints, cbc_fields=list(metrics.CBC_FIELDS))
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle: torch.save(bundle, handle)
        checkpoints["fold" + str(fold)] = sha(path)
        reloaded = torch.load(path, map_location="cpu", weights_only=False)
        original = torch.load(source.PRIVATE / ("fold" + str(fold) + ".pt"), map_location="cpu", weights_only=False)
        validate_bundle(reloaded, original, p, fold)
        for version, model in models.items():
            source.base._atomic_progress(OUT / "progress.json", "frozen_inference_" + version, fold)
            pp = origin.native.kernel.predict_native(model, c, cm, r, rm, age)
            for route in metrics.ROUTES:
                require(np.array_equal(np.isnan(pp[route]), np.isnan(ip[route])))
                for j, e in enumerate(endpoints): pred[e][version + "_" + route][te] = pp[route][te, j]
            for pattern in PATTERNS:
                cpred[pattern][version][te] = cbc_prediction(model, c, cm, r, rm, age, slots, transform, pattern)[0][te]
            for pattern in masking.PATTERNS:
                x = masking.remove_inputs(c, cm, r, rm, slots, pattern)
                masking.assert_no_input_leak(x, cm, rm, slots, pattern)
                value = origin.native.kernel.predict_native(model, x.clinical[te], x.clinical_mask[te],
                    x.retinal[te], x.retinal_mask[te], age[te])["both"]
                require(np.array_equal(np.isfinite(value).all(1), x.available[te]))
                stress[version][pattern][te] = value
            if version != "initial":
                require(all(torch.equal(model.state_dict()[k], v) for k, v in reloaded[version].items()))
                model.load_state_dict(reloaded[version], strict=True)
                replay = origin.native.kernel.predict_native(model, c, cm, r, rm, age)
                require(all(np.array_equal(pp[route], replay[route], equal_nan=True) for route in metrics.ROUTES))
                for pattern in PATTERNS:
                    require(np.array_equal(cbc_prediction(model, c, cm, r, rm, age, slots, transform, pattern)[0][te],
                                           cpred[pattern][version][te], equal_nan=True))
        del prepared, models
    for e in endpoints:
        for arm in metrics.S_ARMS: pred[e][arm][~masks[e]] = np.nan
    source.base._atomic_progress(OUT / "progress.json", "aggregate_bootstrap")
    counts = source.ev.paired_counts(folds, draws=1000, seed=91501)
    screen = metrics.screening(pred, ctx["labels_by_source"], masks, folds, endpoints, counts)
    completion = {pattern: metrics.completion(target, cobs[pattern], cpred[pattern],
                  safe_tail_groups(cobs[pattern], groups[pattern]), counts) for pattern in PATTERNS}
    missingness = {v: stress_metrics.summarize(stress[v], labels, lm, folds, endpoints, counts) for v in metrics.VERSIONS}
    result = {"schema": "bran-native-rehearsal-aggregate-v1", "status": "completed",
        "screening": screen, "completion": completion, "missingness": missingness,
        "decisions": decisions(screen, completion, missingness), "state_width": 192,
        "paired_people": 1928, "recorded_conditions": 26,
        "fold_authentication": p["native_source"]["source"]["authentication"], **FLAGS}
    return result, checkpoints


def audit(p, pin):
    import torch
    from patient_atlas_v6_2_expanded_endpoint_evaluation import EXACT_OUTER_FOLD_HASH, EXACT_INNER_FOLD_ASSIGNMENT_SHA256
    require(not (OUT / "failure.json").exists())
    a = json.loads((OUT / "aggregate.json").read_text()); validate_result(a, p)
    m = json.loads((OUT / "manifest.json").read_text())
    require(set(m) == {"protocol_sha256", "aggregate_sha256", "checkpoint_sha256", "baseline_replay_sha256", "elapsed_seconds", "patient_level_output_emitted"})
    require(m["protocol_sha256"] == pin and m["aggregate_sha256"] == sha(OUT / "aggregate.json") and m["patient_level_output_emitted"] is False)
    require(metrics.finite(m["elapsed_seconds"]) and m["elapsed_seconds"] >= 0
            and set(m["checkpoint_sha256"]) == {"fold" + str(f) for f in range(5)})
    require(m["baseline_replay_sha256"] == sha(OUT / "baseline_replay.json"))
    require(json.loads((OUT / "baseline_replay.json").read_text()) == {"status": "passed",
        "native_screen_all26_three_routes": True, "raw_screen_all26_four_arms": True,
        "native_wholeCBC_all9": True, "patient_level_output_emitted": False})
    for fold in range(5):
        key = "fold" + str(fold); path = PRIVATE / (key + ".pt")
        require(path.stat().st_mode & 0o777 == 0o600 and sha(path) == m["checkpoint_sha256"][key])
        original = origin.native.source.PRIVATE / (key + ".pt")
        require(sha(original) == p["native_source"]["checkpoint_sha256"][key])
        validate_bundle(torch.load(path, map_location="cpu", weights_only=False), torch.load(original, map_location="cpu", weights_only=False), p, fold)
        require(sha(path) == m["checkpoint_sha256"][key])
    auth = p["native_source"]["source"]["authentication"]
    require(auth["outer_fold_sha256"] == EXACT_OUTER_FOLD_HASH and auth["inner_fold_sha256"] == list(EXACT_INNER_FOLD_ASSIGNMENT_SHA256))
    validate_protocol(p)
    return {"schema": "bran-native-rehearsal-audit-v1", "status": "authenticated", "protocol_sha256": pin,
            "aggregate_sha256": sha(OUT / "aggregate.json"), "manifest_sha256": sha(OUT / "manifest.json"),
            "checkpoint_sha256": m["checkpoint_sha256"], "fold_authentication": auth, "patient_level_output_emitted": False}


def main():
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    for operation in ("prepare", "run", "audit"): group.add_argument("--" + operation, action="store_true")
    parser.add_argument("--protocol-sha256"); args = parser.parse_args()
    ok, owned, phase, start = False, False, "protocol", time.monotonic()
    destination = AUDIT if args.audit else OUT
    with _quiet():
        try:
            if args.prepare:
                require(not OUT.exists() and not AUDIT.exists() and not PRIVATE.exists())
                exclusive_json(PROTOCOL, prepare())
            else:
                require(args.protocol_sha256 and sha(PROTOCOL) == args.protocol_sha256)
                p = json.loads(PROTOCOL.read_text()); validate_protocol(p)
                destination.mkdir(); owned = True
                if args.run:
                    PRIVATE.mkdir(mode=0o700)
                    with open("/private/tmp/bran_native_rehearsal_v1.lock", "a") as lock:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        phase = "training_comparison"; a, checkpoints = run(p)
                        phase = "terminal_validation"; validate_result(a, p); validate_protocol(p)
                        exclusive_json(OUT / "aggregate.json", a)
                        exclusive_json(OUT / "manifest.json", {"protocol_sha256": args.protocol_sha256,
                            "aggregate_sha256": sha(OUT / "aggregate.json"), "checkpoint_sha256": checkpoints,
                            "baseline_replay_sha256": sha(OUT / "baseline_replay.json"),
                            "elapsed_seconds": round(time.monotonic() - start, 1), "patient_level_output_emitted": False})
                        origin.native.source.base._atomic_completed(OUT / "progress.json")
                else:
                    phase = "audit"; exclusive_json(AUDIT / "audit.json", audit(p, args.protocol_sha256))
            ok = True
        except Exception as error:
            if owned:
                allowed = p["code_sha256"] if "p" in locals() else {}
                frames = [{"file": Path(f.filename).name, "line": f.lineno}
                          for f in traceback.extract_tb(error.__traceback__)
                          if Path(f.filename).parent == ROOT and Path(f.filename).name in allowed]
                exclusive_json(destination / "failure.json", {"status": "execution_failed", "phase": phase,
                    "error_class": type(error).__name__ if type(error) in (ValueError, TypeError, KeyError, RuntimeError, OSError, ImportError) else "other",
                    "code_frames": frames,
                    "patient_level_output_emitted": False})
    print(json.dumps({"operation": "prepare" if args.prepare else "audit" if args.audit else "run",
                      "status": "completed" if ok else "execution_failed", "patient_level_output_emitted": False}))
    return 0 if ok else 1


if __name__ == "__main__": raise SystemExit(main())
