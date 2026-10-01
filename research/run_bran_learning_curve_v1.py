"""Local, subset-refitted paired BRAN learning curve; closed aggregates only."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import run_bran_raw_teacher_distillation_v1 as old
import bran_distillation_metrics_v1 as helper
from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import sha, exclusive_json

require = helper.require
ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'BRAN_LEARNING_CURVE_PROTOCOL_V1.json'
OUT = ROOT / 'BRAN_LEARNING_CURVE_V1'
AUDIT = ROOT / 'BRAN_LEARNING_CURVE_AUDIT_V1'
PRIVATE = ROOT / 'private_artifacts/bran_learning_curve_v1'
BUDGETS = ('025', '050', '075', '100')
FRACTIONS = (.25, .5, .75, 1.)
ARMS = ('bran_both', 'bran_clinical', 'bran_retinal', 'raw_average')
PARAMETERS = {'fractions': list(FRACTIONS), 'outer_folds': 5, 'one_nested_draw_per_fold': True,
    'subset_seed_base': 97001, 'paired_seed_base': 1701, 'joint_seed_base': 92401,
    'paired_steps': 1500, 'joint_steps': 1500, 'batch_size': 96, 'retinal_backbone_fixed': True,
    'smaller_size_full_cohort_checkpoint_reuse': False, 'full_size_checkpoint_reused_with_replay': True,
    'normalizers_and_paired_training_subset_only': True, 'raw_tree_seed_base': 92381,
    'raw_tree': '128trees_depth6_leaf10_maxfeatures0.7_unweighted_threads2',
    'raw_prior_fallback': 'Laplace prevalence if observed training<20 or only one class; 0.5 if none',
    'arms': list(ARMS), 'bootstrap_draws': 1000, 'bootstrap_seed': 91501,
    'minimum_release_per_class': 20, 'minimum_valid_draws': 900, 'prediction_replay_tolerance': 1e-10,
    'uncertainty': 'marginal_fixed_fit_evaluation_sampling_only_not_training_subset_or_optimization_variation',
    'fixed_step_budget_all_sizes': True, 'hyperparameter_selection': False, 'extrapolation_permitted': False,
    'adaptive_development': True, 'official_test_used': False, 'automatic_promotion': False,
    'patient_level_output_permitted': False}
CODE = ('run_bran_learning_curve_v1.py', 'bran_learning_curve_sampling_v1.py',
    'test_bran_learning_curve_sampling_v1.py', 'test_run_bran_learning_curve_v1.py',
    'build_bran_learning_curve_v1.py', 'test_build_bran_learning_curve_v1.py',
    'BRAN_LEARNING_CURVE_DESIGN_V1.md', 'bran_external_cbc_fit_kernel_v1.py', 'bran_screening_joint_kernel_v1.py')


def prepare():
    src = old.prepare()
    return {'schema': 'bran-learning-curve-protocol-v1', 'status': 'frozen_before_execution',
        'parameters': PARAMETERS, 'native_source': src['native_source'],
        'code_sha256': {**src['code_sha256'], **old.native.source.io.code_closure(CODE)}, 'runtime': old.native.source.io.runtime()}


def validate_protocol(p): require(p == prepare(), 'learning_curve_protocol_changed')


def trained_model(c, cm, r, rm, age, labels, lm, train, slots, fold):
    """Only train is passed to both stages. No checkpoint is accepted here."""
    from bran_external_cbc_fit_kernel_v1 import paired_train
    from bran_screening_joint_kernel_v1 import adapt
    initial = paired_train(c, cm, r, rm, age, train, seed=1701 + fold, steps=1500, batch_size=96)
    model = adapt(initial, c, cm, r, rm, age, labels, lm, train, slots,
        seed=92401 + fold, steps=1500, batch_size=96, candidate=True)
    model.eval(); return model


def load_bundle(path, p, fold, budget):
    import torch
    from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
    from bran_patient_state_prototype_v1 import PatientStateConfig
    require(path.stat().st_mode & 0o777 == 0o600)
    b = torch.load(path, map_location='cpu', weights_only=False)
    require(set(b) == {'schema', 'fold', 'budget', 'protocol_sha256', 'train_count', 'subset_sha256', 'state_dict', 'normalizers'})
    require(b['schema'] == 'bran-learning-curve-checkpoint-v1' and b['fold'] == fold and b['budget'] == budget)
    require(b['protocol_sha256'] == sha(PROTOCOL) and type(b['train_count']) is int and b['train_count'] >= 20)
    require(type(b['subset_sha256']) is str and len(b['subset_sha256']) == 64)
    require(set(b['normalizers']) == set(old.NORMALIZERS))
    model = BRANClinicalAnchorV2(PatientStateConfig())
    model.screening_joint_head = torch.nn.Linear(192, 26); model.cbc_joint_head = torch.nn.Linear(192, 9)
    model.load_state_dict(b['state_dict'], strict=True); model.eval()
    require(all(torch.isfinite(v).all() for v in model.state_dict().values()))
    return model, b


def summarize(pred, labels, masks, folds, names, training_counts, counts):
    points = {}; draws = {}; endpoint_result = {}
    for budget in BUDGETS:
        endpoint_result[budget] = {}; ep_points = {a: [] for a in ARMS}; ep_draws = {a: [] for a in ARMS}
        for e in names:
            y, valid = labels[e], masks[e]
            if sum(valid & (y == 0)) < 20 or sum(valid & (y == 1)) < 20:
                endpoint_result[budget][e] = {'status': 'unsupported'}; continue
            require(all(np.isfinite(pred[budget][e][a][valid]).all() for a in ARMS))
            ps = {a: old.native.source.base.fold_weighted_auc(y, pred[budget][e][a], valid, folds) for a in ARMS}
            ds = {a: old.native.source.base._weighted_auc_draws(y, pred[budget][e][a], valid, folds, counts) for a in ARMS}
            endpoint_result[budget][e] = {'status': 'supported', 'arms': {a: {'auroc': float(ps[a]), 'ci95': helper.interval(ds[a])} for a in ARMS}}
            for a in ARMS: ep_points[a].append(ps[a]); ep_draws[a].append(ds[a])
        if all(cell['status'] == 'supported' for cell in endpoint_result[budget].values()):
            points[budget] = {a: float(np.mean(ep_points[a])) for a in ARMS}
            draws[budget] = {a: np.mean(ep_draws[a], axis=0) for a in ARMS}
    complete = len(points) == 4; curve = {}
    for budget, frac in zip(BUDGETS, FRACTIONS):
        ns = training_counts[budget]
        curve[budget] = {'fraction': frac, 'mean_training_participants': float(np.mean(ns)),
            'minimum_training_participants': int(min(ns)), 'maximum_training_participants': int(max(ns)),
            'arms': {a: {'auroc': points[budget][a], 'ci95': helper.interval(draws[budget][a])} for a in ARMS} if complete else None,
            'endpoints': endpoint_result[budget]}
    contrasts = None
    if complete:
        contrasts = {b: {a: {'delta': points['100'][a] - points[b][a], 'ci95': helper.interval(draws['100'][a] - draws[b][a])} for a in ARMS} for b in BUDGETS[:-1]}
    return {'complete_panel': complete, 'curve': curve, 'full_minus_smaller': contrasts}


def validate_metrics(s, names):
    require(set(s) == {'complete_panel', 'curve', 'full_minus_smaller'} and type(s['complete_panel']) is bool)
    require(set(s['curve']) == set(BUDGETS) and len(names) == 26)
    def metric(v, key):
        require(set(v) == {key, 'ci95'} and helper.finite(v[key]))
        require((-1 if key == 'delta' else 0) <= v[key] <= 1)
        require(type(v['ci95']) is list and len(v['ci95']) == 2 and all(helper.finite(x) for x in v['ci95']))
        require((-1 if key == 'delta' else 0) <= v['ci95'][0] <= v['ci95'][1] <= 1)
    for budget, frac in zip(BUDGETS, FRACTIONS):
        row = s['curve'][budget]
        require(set(row) == {'fraction', 'mean_training_participants', 'minimum_training_participants', 'maximum_training_participants', 'arms', 'endpoints'})
        require(row['fraction'] == frac and helper.finite(row['mean_training_participants']))
        require(type(row['minimum_training_participants']) is int and type(row['maximum_training_participants']) is int)
        require(20 <= row['minimum_training_participants'] <= row['mean_training_participants'] <= row['maximum_training_participants'] < 1928)
        require(set(row['endpoints']) == set(names))
        for cell in row['endpoints'].values():
            if cell == {'status': 'unsupported'}: continue
            require(set(cell) == {'status', 'arms'} and cell['status'] == 'supported' and set(cell['arms']) == set(ARMS))
            for v in cell['arms'].values(): metric(v, 'auroc')
        if s['complete_panel']:
            require(set(row['arms']) == set(ARMS) and all(x['status'] == 'supported' for x in row['endpoints'].values()))
            for a, v in row['arms'].items():
                metric(v, 'auroc'); require(abs(v['auroc'] - np.mean([x['arms'][a]['auroc'] for x in row['endpoints'].values()])) < 1e-10)
        else: require(row['arms'] is None)
    expected_complete = all(x['status'] == 'supported' for row in s['curve'].values() for x in row['endpoints'].values())
    require(s['complete_panel'] == expected_complete)
    require(np.isclose(s['curve']['100']['mean_training_participants'], 1928 * .8))
    for left, right in zip(BUDGETS[:-1], BUDGETS[1:]):
        require(s['curve'][left]['mean_training_participants'] < s['curve'][right]['mean_training_participants'])
    if s['complete_panel']:
        require(set(s['full_minus_smaller']) == set(BUDGETS[:-1]))
        for b, arms in s['full_minus_smaller'].items():
            require(set(arms) == set(ARMS))
            for a, v in arms.items():
                metric(v, 'delta'); require(abs(v['delta'] - (s['curve']['100']['arms'][a]['auroc'] - s['curve'][b]['arms'][a]['auroc'])) < 1e-10)
    else: require(s['full_minus_smaller'] is None)


def run(p):
    import torch
    from bran_learning_curve_sampling_v1 import nested_training_subsets, constant_or_tree_screen
    torch.set_num_threads(2); source = old.native.source
    ctx, folds, c0, cm0, eligible, r0, rm, ages, names = source.io.load_context()
    require(len(folds) == 1928 and set(np.unique(folds)) == set(range(5)))
    endpoints = p['native_source']['source']['endpoint_names']; slots = tuple(names.index(f) for f in helper.CBC_FIELDS)
    labels = np.column_stack([ctx['labels_by_source'][e] for e in endpoints]); lm = np.column_stack([ctx['observed_by_source'][e] for e in endpoints]).astype(bool)
    pred = {b: {e: {a: np.full(len(folds), np.nan) for a in ARMS} for e in endpoints} for b in BUDGETS}
    training_counts = {b: [] for b in BUDGETS}; cache = []; hashes = {}
    for fold in range(5):
        source.base._atomic_progress(OUT / 'progress.json', 'full_budget_replay', fold)
        tr, te = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        subsets = nested_training_subsets(tr, te, len(folds), seed=97001 + fold)
        for b in BUDGETS: training_counts[b].append(len(subsets[b]))
        transform = source.base.FoldTransform(c0, cm0, eligible, r0, rm, ages, tr)
        c, cm, r, age = transform.apply(c0, cm0, eligible, r0, rm, ages)
        model = old.load_initial(fold, transform, p)
        _, identity = source.base._inner_context(ctx, tr, fold)
        require(identity == p['native_source']['source']['authentication']['inner_fold_sha256'][fold])
        native = old.native.kernel.predict_native(model, c, cm, r, rm, age)
        for j, e in enumerate(endpoints):
            for route in helper.ROUTES: pred['100'][e]['bran_' + route][te] = native[route][te, j]
            a = constant_or_tree_screen(np.c_[c, cm, age], labels[:, j], lm[:, j], tr, te, 92381 + fold)
            b = constant_or_tree_screen(np.c_[r, rm, age], labels[:, j], lm[:, j], tr, te, 92381 + fold)
            pred['100'][e]['raw_average'][te] = .5 * (a + b)
        cache.append((tr, te, subsets))
    masks = {}
    old_native = json.loads((old.native.OUT / 'aggregate.json').read_text())
    raw = old.previous.baseline()
    for e in endpoints:
        masks[e] = np.asarray(ctx['observed_by_source'][e], bool).copy()
        for route in helper.ROUTES: masks[e] &= np.isfinite(pred['100'][e]['bran_' + route])
        for route in helper.ROUTES:
            point = source.base.fold_weighted_auc(ctx['labels_by_source'][e], pred['100'][e]['bran_' + route], masks[e], folds)
            require(abs(point - old_native['results']['endpoints'][e]['arms']['native_' + route]['auroc']) <= 1e-10)
        point = source.base.fold_weighted_auc(ctx['labels_by_source'][e], pred['100'][e]['raw_average'], ctx['observed_by_source'][e], folds)
        require(abs(point - raw['endpoints'][e]['arms']['late_average']['auroc']) <= 1e-10)
    exclusive_json(OUT / 'baseline_replay.json', {'status': 'passed', 'all26_native_routes_and_raw_average': True, 'patient_level_output_emitted': False})
    for budget in BUDGETS[:-1]:
        for fold, (_, te, subsets) in enumerate(cache):
            source.base._atomic_progress(OUT / 'progress.json', 'fit_fraction_' + budget, fold)
            tr = subsets[budget]
            transform = source.base.FoldTransform(c0, cm0, eligible, r0, rm, ages, tr)
            c, cm, r, age = transform.apply(c0, cm0, eligible, r0, rm, ages)
            model = trained_model(c, cm, r, rm, age, labels, lm, tr, slots, fold)
            native = old.native.kernel.predict_native(model, c, cm, r, rm, age)
            for j, e in enumerate(endpoints):
                for route in helper.ROUTES: pred[budget][e]['bran_' + route][te] = native[route][te, j]
                a = constant_or_tree_screen(np.c_[c, cm, age], labels[:, j], lm[:, j], tr, te, 92381 + fold)
                b = constant_or_tree_screen(np.c_[r, rm, age], labels[:, j], lm[:, j], tr, te, 92381 + fold)
                pred[budget][e]['raw_average'][te] = .5 * (a + b)
            name = 'fold' + str(fold) + '_' + budget; path = PRIVATE / (name + '.pt')
            bundle = {'schema': 'bran-learning-curve-checkpoint-v1', 'fold': fold, 'budget': budget,
                'protocol_sha256': sha(PROTOCOL), 'train_count': len(tr), 'subset_sha256': hashlib.sha256(tr.astype('<i8').tobytes()).hexdigest(),
                'state_dict': model.state_dict(), 'normalizers': {k: getattr(transform, k) for k in old.NORMALIZERS}}
            with path.open('xb') as handle: torch.save(bundle, handle)
            os.chmod(path, 0o600); hashes[name] = sha(path)
            loaded, b = load_bundle(path, p, fold, budget)
            for k in old.NORMALIZERS: require(np.array_equal(b['normalizers'][k], getattr(transform, k)))
            output = old.native.kernel.predict_native(loaded, c, cm, r, rm, age)
            for route in helper.ROUTES: require(np.array_equal(output[route][te], native[route][te], equal_nan=True))
            require(sha(path) == hashes[name])
    source.base._atomic_progress(OUT / 'progress.json', 'paired_uncertainty')
    counts = source.ev.paired_counts(folds, draws=1000, seed=91501)
    results = summarize(pred, ctx['labels_by_source'], masks, folds, endpoints, training_counts, counts)
    return {'schema': 'bran-learning-curve-aggregate-v1', 'status': 'completed', 'paired_people': 1928,
        'recorded_conditions': 26, 'results': results, 'smaller_sizes_refit_from_scratch': True,
        'original_checkpoint_unchanged': True, 'full_budget_replay_passed': True, 'new_checkpoint_reload_equal': True,
        'evaluation_sampling_uncertainty_only': True, 'adaptive_development': True, 'official_test_used': False,
        'automatic_promotion': False, 'patient_level_output_emitted': False}, hashes


def validate_result(a, p):
    require(set(a) == {'schema', 'status', 'paired_people', 'recorded_conditions', 'results', 'smaller_sizes_refit_from_scratch',
        'original_checkpoint_unchanged', 'full_budget_replay_passed', 'new_checkpoint_reload_equal',
        'evaluation_sampling_uncertainty_only', 'adaptive_development', 'official_test_used', 'automatic_promotion', 'patient_level_output_emitted'})
    require(a['schema'] == 'bran-learning-curve-aggregate-v1' and a['status'] == 'completed')
    require(type(a['paired_people']) is int and a['paired_people'] == 1928 and type(a['recorded_conditions']) is int and a['recorded_conditions'] == 26)
    require(all(a[k] is True for k in ('smaller_sizes_refit_from_scratch','original_checkpoint_unchanged','full_budget_replay_passed',
        'new_checkpoint_reload_equal','evaluation_sampling_uncertainty_only','adaptive_development')))
    require(all(a[k] is False for k in ('official_test_used','automatic_promotion','patient_level_output_emitted')))
    validate_metrics(a['results'], p['native_source']['source']['endpoint_names'])


def audit(p, pin):
    from patient_atlas_v6_2_expanded_endpoint_evaluation import EXACT_OUTER_FOLD_HASH, EXACT_INNER_FOLD_ASSIGNMENT_SHA256
    from bran_learning_curve_sampling_v1 import nested_training_subsets
    require(not (OUT / 'failure.json').exists())
    a = json.loads((OUT / 'aggregate.json').read_text()); validate_result(a, p)
    m = json.loads((OUT / 'manifest.json').read_text())
    require(set(m) == {'protocol_sha256', 'aggregate_sha256', 'checkpoint_sha256', 'baseline_replay_sha256', 'elapsed_seconds', 'patient_level_output_emitted'})
    require(m['protocol_sha256'] == pin and m['aggregate_sha256'] == sha(OUT / 'aggregate.json') and m['patient_level_output_emitted'] is False)
    require(helper.finite(m['elapsed_seconds']) and m['elapsed_seconds'] >= 0)
    require(m['baseline_replay_sha256'] == sha(OUT / 'baseline_replay.json'))
    require(json.loads((OUT / 'baseline_replay.json').read_text()) == {'status': 'passed', 'all26_native_routes_and_raw_average': True, 'patient_level_output_emitted': False})
    expected = {'fold' + str(f) + '_' + b for f in range(5) for b in BUDGETS[:-1]}
    require(set(m['checkpoint_sha256']) == expected and PRIVATE.stat().st_mode & 0o777 == 0o700)
    _, folds, c0, cm0, eligible, r0, rm, ages, _ = old.native.source.io.load_context()
    training_counts = {b: [] for b in BUDGETS}
    subsets = {}
    for fold in range(5):
        tr, te = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        subsets[fold] = nested_training_subsets(tr, te, len(folds), seed=97001 + fold)
        for b in BUDGETS: training_counts[b].append(len(subsets[fold][b]))
        require(sha(old.native.source.PRIVATE / ('fold' + str(fold) + '.pt')) == p['native_source']['checkpoint_sha256']['fold' + str(fold)])
    for budget in BUDGETS[:-1]:
        for fold in range(5):
            name = 'fold' + str(fold) + '_' + budget; path = PRIVATE / (name + '.pt')
            require(sha(path) == m['checkpoint_sha256'][name]); _, bundle = load_bundle(path, p, fold, budget)
            tr = subsets[fold][budget]
            require(bundle['train_count'] == len(tr) and bundle['subset_sha256'] == hashlib.sha256(tr.astype('<i8').tobytes()).hexdigest())
            transform = old.native.source.base.FoldTransform(c0, cm0, eligible, r0, rm, ages, tr)
            for key in old.NORMALIZERS: require(np.array_equal(bundle['normalizers'][key], getattr(transform, key)))
            require(sha(path) == m['checkpoint_sha256'][name])
    for b, sizes in training_counts.items():
        row = a['results']['curve'][b]
        require(row['mean_training_participants'] == float(np.mean(sizes)) and row['minimum_training_participants'] == min(sizes) and row['maximum_training_participants'] == max(sizes))
    auth = p['native_source']['source']['authentication']
    require(auth['outer_fold_sha256'] == EXACT_OUTER_FOLD_HASH and auth['inner_fold_sha256'] == list(EXACT_INNER_FOLD_ASSIGNMENT_SHA256))
    return {'schema': 'bran-learning-curve-audit-v1', 'status': 'authenticated', 'protocol_sha256': pin,
        'aggregate_sha256': sha(OUT / 'aggregate.json'), 'manifest_sha256': sha(OUT / 'manifest.json'),
        'checkpoint_sha256': m['checkpoint_sha256'], 'outer_fold_sha256': auth['outer_fold_sha256'], 'inner_fold_sha256': auth['inner_fold_sha256'],
        'subset_identities_and_subset_only_normalizers_verified': True,
        'full_budget_replay_passed': True, 'patient_level_output_emitted': False, 'automatic_promotion': False}


def main():
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'run', 'audit'): group.add_argument('--' + name, action='store_true')
    parser.add_argument('--protocol-sha256'); args = parser.parse_args()
    os.umask(0o077); ok = False; owned = False; phase = 'protocol'; target = AUDIT if args.audit else OUT; start = time.monotonic()
    with _quiet():
        try:
            if args.prepare:
                require(not OUT.exists() and not AUDIT.exists() and not PRIVATE.exists()); exclusive_json(PROTOCOL, prepare()); ok = True
            else:
                require(args.protocol_sha256 and sha(PROTOCOL) == args.protocol_sha256)
                p = json.loads(PROTOCOL.read_text()); validate_protocol(p); target.mkdir(); owned = True
                if args.run:
                    PRIVATE.mkdir(mode=0o700)
                    with open('/private/tmp/bran_learning_curve_v1.lock', 'a') as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); phase = 'experiment'
                        a, hashes = run(p); phase = 'terminal_validation'; validate_protocol(p); validate_result(a, p)
                        exclusive_json(OUT / 'aggregate.json', a)
                        exclusive_json(OUT / 'manifest.json', {'protocol_sha256': args.protocol_sha256, 'aggregate_sha256': sha(OUT / 'aggregate.json'),
                            'checkpoint_sha256': hashes, 'baseline_replay_sha256': sha(OUT / 'baseline_replay.json'),
                            'elapsed_seconds': round(time.monotonic() - start, 1), 'patient_level_output_emitted': False})
                        old.native.source.base._atomic_completed(OUT / 'progress.json')
                else: phase = 'audit'; exclusive_json(AUDIT / 'audit.json', audit(p, args.protocol_sha256))
                ok = True
        except Exception as error:
            if owned: exclusive_json(target / 'failure.json', {'status': 'execution_failed', 'phase': phase,
                'error_class': type(error).__name__ if type(error) in (ValueError,TypeError,RuntimeError,OSError,KeyError) else 'other_execution_error', 'patient_level_output_emitted': False})
    print(json.dumps({'operation': 'prepare' if args.prepare else ('audit' if args.audit else 'run'), 'status': 'completed' if ok else 'failed', 'patient_level_output_emitted': False}))
    return 0 if ok else 1


if __name__ == '__main__': raise SystemExit(main())
