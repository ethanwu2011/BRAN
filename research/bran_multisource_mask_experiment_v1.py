"""Local experiment evaluation; caller owns authentication and FD suppression.

This module is not a public data interface. No patient array is serialized or
returned. The model provider fits only on the training path, or loads already
authenticated checkpoints on the replay path. Reference readouts are refit
deterministically on both paths; candidate training is not repeated by audit.
"""
import numpy as np
import hashlib
import json

import run_bran_native_rehearsal_v1 as old
import bran_multisource_mask_contract_v1 as contract
import bran_native_calibration_split_v1 as splitting
import bran_supervised_mask_uncertainty_v1 as uncertainty

origin, metrics, masking = old.origin, old.metrics, old.masking
PATTERNS = old.PATTERNS
EVAL_PATTERNS = contract.EVALPATTERNS
NO_RETINA_PATTERNS = {"single_target_hidden": "single_target_no_retina",
                      "whole_cbc_hidden": "whole_cbc_no_retina"}
BASELINE = {"status": "passed", "native_screen_all26_three_routes": True,
            "raw_screen_all26_four_arms": True, "native_wholeCBC_all9": True,
            "patient_level_output_emitted": False}


def require(ok):
    if not ok:
        raise ValueError("supervised_mask_experiment_contract_failed")


def cbc_context(r, rm, pattern):
    """Erase both retinal payload and availability before no-eye completion."""
    require(pattern in EVAL_PATTERNS)
    if pattern in NO_RETINA_PATTERNS.values():
        base = next(key for key, value in NO_RETINA_PATTERNS.items() if value == pattern)
        return np.zeros_like(r), np.zeros_like(rm), base
    return r, rm, pattern


def cbc_prediction(model, c, cm, r, rm, age, slots, transform, pattern):
    visible_r, visible_rm, base = cbc_context(r, rm, pattern)
    return old.cbc_prediction(model, c, cm, visible_r, visible_rm, age, slots, transform, base)


def evaluate(p, pin, *, provider, baseline_ready, calibration_ready, progress):
    """Evaluate original and continued native models; aggregate-only return.

    provider(fold, transform, initial, training_args, registry_names) returns dictionaries of
    continued/student models and independently reloaded copies. training_args
    are caller-local arrays and must never enter a log or hosted context.
    """
    import torch
    from bran_matched_screening_kernel_v1 import fit_predict, late_fusion_average

    source = origin.native.source
    torch.set_num_threads(2)
    ctx, folds, c0, cm0, eligible, r0, rm, ages, names = source.io.load_context()
    require(len(folds) == 1928 and set(np.unique(folds)) == set(range(5)))
    endpoints = p["native_source"]["source"]["endpoint_names"]
    slots = tuple(names.index(f) for f in metrics.CBC_FIELDS)
    require(len(set(slots)) == 9 and all(0 <= j < 48 for j in slots))
    labels = np.column_stack([ctx["labels_by_source"][e] for e in endpoints])
    lm = np.column_stack([ctx["observed_by_source"][e] for e in endpoints]).astype(bool)
    target, observed = c0[:, slots].copy(), (cm0 & eligible)[:, slots].copy()
    pred = {e: {a: np.full(len(folds), np.nan) for a in metrics.S_ARMS} for e in endpoints}
    cpred = {pattern: {a: np.full(target.shape, np.nan) for a in metrics.C_ARMS} for pattern in EVAL_PATTERNS}
    cobs = {pattern: observed.copy() for pattern in EVAL_PATTERNS}
    groups = {pattern: {g: np.zeros(target.shape, bool) for g in metrics.GROUPS} for pattern in EVAL_PATTERNS}
    stress = {v: {pattern: np.full(labels.shape, np.nan) for pattern in masking.PATTERNS}
              for v in metrics.VERSIONS}
    cache = []
    raw_reference = origin.previous.baseline()
    native_reference = json.loads((origin.native.OUT / "aggregate.json").read_text())
    cbc_reference = json.loads((source.OUT / "aggregate.json").read_text())["retained_cbc_head_whole_panel"]
    for fold in range(5):
        progress("baseline_replay", fold)
        tr, te = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        transform = source.base.FoldTransform(c0, cm0, eligible, r0, rm, ages, tr)
        c, cm, r, age = transform.apply(c0, cm0, eligible, r0, rm, ages)
        initial = origin.load_initial(fold, transform, p)
        inner, identity = source.base._inner_context(ctx, tr, fold)
        require(identity == p["native_source"]["source"]["authentication"]["inner_fold_sha256"][fold])
        full_inner = np.full(len(folds), -1, int)
        full_inner[tr] = inner
        ip = origin.native.kernel.predict_native(initial, c, cm, r, rm, age)
        raw = {"raw_clinical": np.c_[c, cm, age], "raw_retinal": np.c_[r, rm, age],
               "raw_concat": np.c_[c, cm, r, rm, age]}
        for j, endpoint in enumerate(endpoints):
            for route in metrics.ROUTES:
                pred[endpoint]["initial_" + route][te] = ip[route][te, j]
            for arm in raw:
                pred[endpoint][arm][te], _ = fit_predict(raw[arm], labels[:, j], lm[:, j], tr, te,
                    full_inner, family="extra_trees", seed=92381 + fold)
            pred[endpoint]["late_average"][te] = late_fusion_average(
                pred[endpoint]["raw_clinical"][te], pred[endpoint]["raw_retinal"][te])
        for pattern in EVAL_PATTERNS:
            initial_cbc, support = cbc_prediction(initial, c, cm, r, rm, age, slots, transform, pattern)
            cpred[pattern]["initial"][te] = initial_cbc[te]
            cobs[pattern][te] &= support[te]
            visible_r, visible_rm, base_pattern = cbc_context(r, rm, pattern)
            for j in range(9):
                hc, hm = source.mask_inputs(c, cm, slots, base_pattern, slots[j])
                valid = observed[:, j] & support[:, j]
                for arm, x in {"raw_clinical": np.c_[hc, hm, age],
                               "raw_concat": np.c_[hc, hm, visible_r, visible_rm, age]}.items():
                    cpred[pattern][arm][te, j] = source.ev.fixed_cbc_probe(x, target[:, j], valid, tr, te)
                train_y = target[tr, j][valid[tr]]
                require(len(train_y) >= 20)
                groups[pattern]["overall"][te, j] = True
                for group, mask in origin.previous.tail_masks(train_y, target[te, j]).items():
                    groups[pattern][group][te, j] = mask
        cache.append((tr, te, transform, initial, c, cm, r, age, ip))
    masks = old.screen_masks(pred, ctx["observed_by_source"], endpoints, ctx["labels_by_source"], folds)
    for endpoint in endpoints:
        for route in metrics.ROUTES:
            value = source.base.fold_weighted_auc(ctx["labels_by_source"][endpoint],
                pred[endpoint]["initial_" + route], masks[endpoint], folds)
            require(abs(value - native_reference["results"]["endpoints"][endpoint]["arms"]["native_" + route]["auroc"]) <= 1e-10)
        for arm in ("raw_clinical", "raw_retinal", "raw_concat", "late_average"):
            value = source.base.fold_weighted_auc(ctx["labels_by_source"][endpoint], pred[endpoint][arm],
                ctx["observed_by_source"][endpoint], folds)
            require(abs(value - raw_reference["endpoints"][endpoint]["arms"][arm]["auroc"]) <= 1e-10)
    for j, field in enumerate(metrics.CBC_FIELDS):
        valid = cobs["whole_cbc_hidden"][:, j]
        errors = cpred["whole_cbc_hidden"]["initial"][valid, j] - target[valid, j]
        require(cbc_reference[field]["status"] == "complete"
                and np.isclose(np.abs(errors).mean(), cbc_reference[field]["mae"], rtol=1e-10, atol=1e-10)
                and np.isclose(np.square(errors).mean(), cbc_reference[field]["mse"], rtol=1e-10, atol=1e-10))
    baseline_ready(dict(BASELINE))
    for fold, (tr, te, transform, initial, c, cm, r, age, ip) in enumerate(cache):
        models, reloaded = provider(fold, transform, initial, (c, cm, r, rm, age, labels, lm, tr, slots), tuple(names))
        require(set(models) == set(reloaded) == {"continued", "student"})
        models = {"initial": initial, **models}
        for version, model in models.items():
            progress("frozen_inference_" + version, fold)
            pp = origin.native.kernel.predict_native(model, c, cm, r, rm, age)
            for route in metrics.ROUTES:
                require(np.array_equal(np.isnan(pp[route]), np.isnan(ip[route])))
                for j, endpoint in enumerate(endpoints):
                    pred[endpoint][version + "_" + route][te] = pp[route][te, j]
            for pattern in EVAL_PATTERNS:
                values, support = cbc_prediction(model, c, cm, r, rm, age, slots, transform, pattern)
                require(np.array_equal(support[te] & observed[te], cobs[pattern][te]))
                cpred[pattern][version][te] = values[te]
            for pattern in masking.PATTERNS:
                x = masking.remove_inputs(c, cm, r, rm, slots, pattern)
                masking.assert_no_input_leak(x, cm, rm, slots, pattern)
                values = origin.native.kernel.predict_native(model, x.clinical[te], x.clinical_mask[te],
                    x.retinal[te], x.retinal_mask[te], age[te])["both"]
                require(np.array_equal(np.isfinite(values).all(1), x.available[te]))
                stress[version][pattern][te] = values
            if version != "initial":
                replay = origin.native.kernel.predict_native(reloaded[version], c, cm, r, rm, age)
                require(all(np.array_equal(pp[route], replay[route], equal_nan=True) for route in metrics.ROUTES))
                for pattern in EVAL_PATTERNS:
                    values, _ = cbc_prediction(reloaded[version], c, cm, r, rm, age, slots, transform, pattern)
                    require(np.array_equal(values[te], cpred[pattern][version][te], equal_nan=True))
                for pattern in masking.PATTERNS:
                    x = masking.remove_inputs(c, cm, r, rm, slots, pattern)
                    values = origin.native.kernel.predict_native(reloaded[version], x.clinical[te], x.clinical_mask[te],
                        x.retinal[te], x.retinal_mask[te], age[te])["both"]
                    require(np.array_equal(values, stress[version][pattern][te], equal_nan=True))
    progress("aggregate_bootstrap")
    for endpoint in endpoints:
        for arm in metrics.S_ARMS:
            pred[endpoint][arm][~masks[endpoint]] = np.nan
    counts = source.ev.paired_counts(folds, draws=1000, seed=91501)
    screen = metrics.screening(pred, ctx["labels_by_source"], masks, folds, endpoints, counts)
    all_completion = {pattern: metrics.completion(target, cobs[pattern], cpred[pattern],
        old.safe_tail_groups(cobs[pattern], groups[pattern]), counts) for pattern in EVAL_PATTERNS}
    completion = {pattern: all_completion[pattern] for pattern in PATTERNS}
    completion_no_retina = {pattern: all_completion[NO_RETINA_PATTERNS[pattern]] for pattern in PATTERNS}
    missingness = {v: old.stress_metrics.summarize(stress[v], labels, lm, folds, endpoints, counts)
                   for v in metrics.VERSIONS}
    progress("calibration_scoring")
    ids = list(map(str, ctx["raw_cohort"].patient_ids))
    roles = splitting.make_roles(ids, folds)
    score_counts = source.ev.paired_counts(folds[roles == 1], draws=1000, seed=94701)
    calibrated, radii = uncertainty.evaluate(target, cobs,
        {pattern: {v: cpred[pattern][v] for v in metrics.VERSIONS} for pattern in EVAL_PATTERNS},
        {pattern: {g: groups[pattern][g] for g in ("low", "middle", "high")} for pattern in EVAL_PATTERNS},
        folds, roles, score_counts)
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    split_auth = {"patient_order_sha256": digest(ids), "roles_sha256": digest(roles.tolist())}
    calibration_ready(radii, split_auth)
    result = {"schema": "bran-multisource-mask-aggregate-v1", "status": "completed",
        "protocol_sha256": pin, "screening": screen, "completion": completion,
        "completion_no_retina": completion_no_retina, "missingness": missingness,
        "calibrated_completion": {"split_authentication": split_auth, "patterns": calibrated},
        "decisions": contract.decisions(screen, completion, missingness, completion_no_retina), "state_width": 192,
        "paired_people": 1928, "recorded_conditions": 26,
        "fold_authentication": p["native_source"]["source"]["authentication"], **contract.FLAGS}
    contract.validate_result(result, p, pin)
    return result
