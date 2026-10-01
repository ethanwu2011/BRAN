"""Local blood-plus-age baseline at the frozen BRAN training budgets; no encoder fit."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import warnings
import numpy as np
from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import sha, exclusive_json
import run_bran_learning_curve_v1 as lc
import build_bran_screening_finish_figure_v1 as benchmark

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'BRAN_BLOOD_LEARNING_CURVE_V1'
AUDIT = ROOT / 'BRAN_BLOOD_LEARNING_CURVE_AUDIT_V1'
PROTOCOL = ROOT / 'BRAN_BLOOD_LEARNING_CURVE_PROTOCOL_V1.json'
LC_PIN = '066dead55b897feb8cf1dae29c8d870117b1e437c92808efc332b2de28b121e2'
LC_AGG = '6f0e0bc152259d741d03a1ab17d253262fcb0dcb2135378816f27d9cdefade78'
LC_AUDIT = 'b88e635217fa890a048feca57e52fe58de4381af1102c7ef13cc711ba5e3ab92'
BLOOD = tuple(i for i in range(38) if i not in (8, 9, 20, 35, 36))
CODE = ('run_bran_blood_learning_curve_v1.py', 'test_bran_blood_learning_curve_v1.py',
        'BRAN_BLOOD_LEARNING_CURVE_DESIGN_V1.md')
PARAMETERS = {'blood_slots': list(BLOOD), 'age_included': True, 'vitals_or_retina_in_design': False,
    'fractions': [.25, .5, .75, 1.], 'subset_seed_base': 97001,
    'head': 'original nested StandardScaler + logistic; C=.01,.1,1,10; max_iter500',
    'inner_assignment': 'original five inner folds restricted to each nested training subset',
    'incomplete_inner_support': 'fixed StandardScaler + logistic C=1 max_iter5000; no tuning',
    'sparse_training_support': 'Laplace prevalence if observed training<20 or oneclass; .5 if none',
    'nonconvergence_fallback': False, 'normalizers_fit_on_training_subset_only': True,
    'full_budget_original_blood_reference_replay': True,
    'comparison_mask': 'observed endpoint AND retinal evidence AND any eligible clinical evidence; same native routes',
    'reference_replay_mask': 'all observed endpoint rows, as original blood reference',
    'bootstrap_draws': 1000, 'bootstrap_seed': 91501, 'minimum_release_per_class': 20,
    'minimum_valid_draws': 900, 'encoder_training': False, 'checkpoint_write': False,
    'patient_level_output_permitted': False, 'automatic_promotion': False,
    'adaptive_development': True, 'official_test_used': False, 'extrapolation': False}


def require(x):
    if not x: raise ValueError('blood_curve_contract_failed')


def prepare():
    require(sha(lc.PROTOCOL) == LC_PIN and sha(lc.OUT / 'aggregate.json') == LC_AGG)
    require(sha(lc.AUDIT / 'audit.json') == LC_AUDIT)
    p = json.loads(lc.PROTOCOL.read_text()); lc.validate_protocol(p)
    lc.validate_result(json.loads((lc.OUT / 'aggregate.json').read_text()), p)
    ref = benchmark.source_screening()
    receipt = ref['source']
    evidence_path = ROOT / receipt['package'] / 'evidence.json'
    require(sha(evidence_path) == receipt['evidence_sha256'])
    endpoint_reference = {r['source_code']: r['innovation_arms']['raw_blood']['auroc']
        for r in json.loads(evidence_path.read_text())['screening']['endpoint_rows']}
    require(set(endpoint_reference) == set(p['native_source']['source']['endpoint_names']))
    return {'schema': 'bran-blood-learning-curve-protocol-v1', 'status': 'frozen_before_execution',
        'parameters': PARAMETERS, 'learning_protocol_sha256': LC_PIN,
        'learning_aggregate_sha256': LC_AGG, 'learning_audit_sha256': LC_AUDIT,
        'benchmark_receipt': ref['source'],
        'reference_full_budget_auroc': ref['screening']['innovation_macro_auroc']['raw_blood'],
        'reference_endpoint_auroc': endpoint_reference,
        'code_sha256': {**p['code_sha256'], **{name: sha(ROOT / name) for name in CODE}},
        'outer_fold_sha256': p['native_source']['source']['authentication']['outer_fold_sha256'],
        'inner_fold_sha256': p['native_source']['source']['authentication']['inner_fold_sha256']}


def blood_design(c, cm, age):
    require(c.ndim == 2 and c.shape == cm.shape and c.shape[1] == 59 and cm.dtype == bool)
    require(age.shape == (len(c),) and np.isfinite(age).all())
    clean = np.where(cm[:, BLOOD], c[:, BLOOD], 0.)
    require(np.isfinite(clean).all())
    return np.c_[clean, cm[:, BLOOD].astype(float), age]


def check_endpoint_replay(actual, reference):
    require(set(actual) == set(reference) and len(actual) == 26)
    require(all(lc.helper.finite(actual[e]) and abs(actual[e]-reference[e]) < 1e-8 for e in reference))


def fit_predict(x, y, observed, train, test, inner):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    require(len(set(train)) == len(train) and not np.intersect1d(train, test).size)
    require(x.ndim == 2 and np.isfinite(x).all() and inner.shape == train.shape)
    require(observed.dtype == bool and y.shape == observed.shape == (len(x),))
    tr_y, tr_m = y[train], observed[train]
    require(np.isfinite(tr_y[tr_m]).all() and np.isin(tr_y[tr_m], [0, 1]).all())
    if tr_m.sum() < 20 or len(np.unique(tr_y[tr_m])) < 2:
        value = (np.sum(tr_y[tr_m] == 1)+1)/(tr_m.sum()+2)
        return np.full(len(test), value), 'prevalence'
    complete = all(np.sum(tr_m & (inner == f)) >= 2 and np.sum(tr_m & (inner != f)) >= 2
        and len(np.unique(tr_y[tr_m & (inner == f)])) == 2
        and len(np.unique(tr_y[tr_m & (inner != f)])) == 2 for f in range(5))
    if complete:
        pred, _ = lc.old.native.source.base._fit_predict_nested(x[train], tr_y, tr_m, inner, x[test])
        return pred, 'nested'
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., solver='lbfgs', max_iter=5000))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning); model.fit(x[train][tr_m], tr_y[tr_m])
    require(not any(issubclass(w.category, ConvergenceWarning) for w in caught))
    return model.predict_proba(x[test])[:, 1], 'fixed_c1'


def validate_result(a, p):
    require(p == json.loads(PROTOCOL.read_text()) and sha(lc.OUT / 'aggregate.json') == LC_AGG)
    require(set(a) == {'schema', 'status', 'protocol_sha256', 'curve', 'full_minus_smaller',
        'paired_people', 'outcomes', 'full_budget_original_reference_replayed',
        'same_native_evaluation_support', 'any_incomplete_inner_support_fallback',
        'any_sparse_training_fallback', 'patient_level_output_emitted', 'automatic_promotion'})
    require(a['schema'] == 'bran-blood-learning-curve-v1' and a['status'] == 'completed')
    require(a['protocol_sha256'] == sha(PROTOCOL) and a['paired_people'] == 1928 and a['outcomes'] == 26)
    require(a['full_budget_original_reference_replayed'] is True and a['same_native_evaluation_support'] is True)
    require(a['patient_level_output_emitted'] is False and a['automatic_promotion'] is False)
    for flag in ('any_incomplete_inner_support_fallback', 'any_sparse_training_fallback'): require(type(a[flag]) is bool)
    require(set(a['curve']) == set(lc.BUDGETS) and set(a['full_minus_smaller']) == set(lc.BUDGETS[:-1]))
    old_result = json.loads((lc.OUT / 'aggregate.json').read_text())
    for budget in lc.BUDGETS:
        row = a['curve'][budget]
        require(set(row) == {'n', 'fraction', 'auroc', 'ci95'})
        require(row['n'] == old_result['results']['curve'][budget]['mean_training_participants'])
        require(row['fraction'] == old_result['results']['curve'][budget]['fraction'])
        require(lc.helper.finite(row['auroc']) and 0 <= row['auroc'] <= 1)
        require(type(row['ci95']) is list and len(row['ci95']) == 2
            and all(lc.helper.finite(x) and 0 <= x <= 1 for x in row['ci95']) and row['ci95'][0] <= row['ci95'][1])
    for budget, row in a['full_minus_smaller'].items():
        require(set(row) == {'delta', 'ci95'} and lc.helper.finite(row['delta']))
        require(abs(row['delta']-(a['curve']['100']['auroc']-a['curve'][budget]['auroc'])) < 1e-10)
        require(type(row['ci95']) is list and len(row['ci95']) == 2
            and all(lc.helper.finite(x) and -1 <= x <= 1 for x in row['ci95']) and row['ci95'][0] <= row['ci95'][1])


def run(p):
    from bran_learning_curve_sampling_v1 import nested_training_subsets
    source = lc.old.native.source
    ctx, folds, c0, cm0, eligible, r0, rm, ages, names = source.io.load_context()
    lp = json.loads(lc.PROTOCOL.read_text())
    endpoints = lp['native_source']['source']['endpoint_names']
    require(len(folds) == 1928 and len(BLOOD) == 33 and len(endpoints) == 26)
    predictions = {budget: {e: np.full(len(folds), np.nan) for e in endpoints} for budget in lc.BUDGETS}
    ns = {b: [] for b in lc.BUDGETS}; subset_hashes = {}; flags = set()
    for fold in range(5):
        outer_train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        subsets = nested_training_subsets(outer_train, test, len(folds), seed=97001+fold)
        inner, digest = source.base._inner_context(ctx, outer_train, fold)
        require(digest == p['inner_fold_sha256'][fold])
        for budget in lc.BUDGETS:
            source.base._atomic_progress(OUT / 'progress.json', 'blood_heads_' + budget, fold)
            train = subsets[budget]; ns[budget].append(len(train))
            subset_hashes[str(fold)+'_'+budget] = hashlib.sha256(train.astype('<i8').tobytes()).hexdigest()
            transform = source.base.FoldTransform(c0, cm0, eligible, r0, rm, ages, train)
            c, cm, _, age = transform.apply(c0, cm0, eligible, r0, rm, ages)
            x = blood_design(c, cm, age)
            inner_selected = inner[np.searchsorted(outer_train, train)]
            for e in endpoints:
                pred, route = fit_predict(x, np.asarray(ctx['labels_by_source'][e]), np.asarray(ctx['observed_by_source'][e], bool), train, test, inner_selected)
                flags.add(route); require(np.isfinite(pred).all() and np.all((pred >= 0) & (pred <= 1)))
                if budget == '100': require(route == 'nested')
                predictions[budget][e][test] = pred
    source.base._atomic_progress(OUT / 'progress.json', 'aggregate_bootstrap')
    original_points = {e: source.base.fold_weighted_auc(ctx['labels_by_source'][e], predictions['100'][e], ctx['observed_by_source'][e], folds) for e in endpoints}
    check_endpoint_replay(original_points, p['reference_endpoint_auroc'])
    require(abs(float(np.mean(list(original_points.values())))-p['reference_full_budget_auroc']) < 1e-8)
    counts = source.ev.paired_counts(folds, draws=1000, seed=91501)
    physiology = rm & (cm0 & eligible & np.isfinite(c0)).any(axis=1)
    curve = {}; draws = {}
    for budget, fraction in zip(lc.BUDGETS, lc.FRACTIONS):
        ep_points = []; ep_draws = []
        for e in endpoints:
            y = np.asarray(ctx['labels_by_source'][e]); mask = np.asarray(ctx['observed_by_source'][e], bool) & physiology
            require(sum(mask & (y == 0)) >= 20 and sum(mask & (y == 1)) >= 20)
            ep_points.append(source.base.fold_weighted_auc(y, predictions[budget][e], mask, folds))
            ep_draws.append(source.base._weighted_auc_draws(y, predictions[budget][e], mask, folds, counts))
        ds = np.mean(ep_draws, axis=0); draws[budget] = ds
        curve[budget] = {'n': float(np.mean(ns[budget])), 'fraction': fraction,
                         'auroc': float(np.mean(ep_points)), 'ci95': lc.helper.interval(ds)}
    result = {'schema': 'bran-blood-learning-curve-v1', 'status': 'completed', 'protocol_sha256': sha(PROTOCOL),
        'curve': curve, 'full_minus_smaller': {b: {'delta': curve['100']['auroc']-curve[b]['auroc'],
             'ci95': lc.helper.interval(draws['100']-draws[b])} for b in lc.BUDGETS[:-1]},
        'paired_people': 1928, 'outcomes': 26, 'full_budget_original_reference_replayed': True,
        'same_native_evaluation_support': True, 'any_incomplete_inner_support_fallback': 'fixed_c1' in flags,
        'any_sparse_training_fallback': 'prevalence' in flags, 'patient_level_output_emitted': False, 'automatic_promotion': False}
    validate_result(result, p)
    return result, subset_hashes


def audit(p):
    """Authenticate the exclusive macro artifact and independently reconstruct split identities."""
    from bran_learning_curve_sampling_v1 import nested_training_subsets
    require(p == prepare() and not (OUT / 'failure.json').exists())
    require({v.name for v in OUT.iterdir()} == {'aggregate.json', 'manifest.json', 'progress.json'})
    require(json.loads((OUT / 'progress.json').read_text()) == {'status': 'completed', 'phase': 'completed'})
    a = json.loads((OUT / 'aggregate.json').read_text()); validate_result(a, p)
    m = json.loads((OUT / 'manifest.json').read_text())
    require(set(m) == {'protocol_sha256', 'aggregate_sha256', 'subset_sha256', 'elapsed_seconds', 'patient_level_output_emitted'})
    require(m['protocol_sha256'] == sha(PROTOCOL) and m['aggregate_sha256'] == sha(OUT / 'aggregate.json'))
    require(m['patient_level_output_emitted'] is False and lc.helper.finite(m['elapsed_seconds']) and m['elapsed_seconds'] >= 0)
    source = lc.old.native.source
    ctx, folds, *_ = source.io.load_context()
    require(len(folds) == 1928 and set(np.unique(folds)) == set(range(5)))
    expected = {}; ns = {b: [] for b in lc.BUDGETS}
    for fold in range(5):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        _, digest = source.base._inner_context(ctx, train, fold)
        require(digest == p['inner_fold_sha256'][fold])
        subsets = nested_training_subsets(train, test, len(folds), seed=97001+fold)
        for b, selected in subsets.items():
            expected[str(fold)+'_'+b] = hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest()
            ns[b].append(len(selected))
    require(m['subset_sha256'] == expected)
    for b in lc.BUDGETS: require(a['curve'][b]['n'] == float(np.mean(ns[b])))
    require(p == prepare())
    return {'schema': 'bran-blood-learning-curve-audit-v1', 'status': 'authenticated',
        'protocol_sha256': sha(PROTOCOL), 'aggregate_sha256': sha(OUT / 'aggregate.json'),
        'manifest_sha256': sha(OUT / 'manifest.json'), 'outer_fold_sha256': p['outer_fold_sha256'],
        'inner_fold_sha256': p['inner_fold_sha256'], 'subset_identities_verified': True,
        'paired_people': 1928, 'outcomes': 26, 'patient_level_output_emitted': False,
        'automatic_promotion': False, 'full_budget_original_reference_replayed': True}


def main():
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'run', 'audit'): group.add_argument('--'+name, action='store_true')
    parser.add_argument('--protocol-sha256'); args = parser.parse_args()
    ok = False; owned = False; phase = 'protocol'; start = time.monotonic()
    target = AUDIT if args.audit else OUT
    os.umask(0o077)
    with _quiet():
        try:
            if args.prepare:
                require(not OUT.exists() and not AUDIT.exists()); exclusive_json(PROTOCOL, prepare()); ok = True
            elif args.audit:
                require(args.protocol_sha256 == sha(PROTOCOL)); p = json.loads(PROTOCOL.read_text()); require(p == prepare())
                AUDIT.mkdir(); owned = True; phase = 'audit'; exclusive_json(AUDIT / 'audit.json', audit(p)); ok = True
            else:
                require(args.protocol_sha256 == sha(PROTOCOL)); p = json.loads(PROTOCOL.read_text()); require(p == prepare())
                with open('/private/tmp/bran_blood_learning_curve_v1.lock', 'a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    OUT.mkdir(); owned = True; phase = 'blood_readouts'; a, subsets = run(p)
                    phase = 'terminal_authentication'; require(p == prepare()); validate_result(a, p)
                    exclusive_json(OUT / 'aggregate.json', a)
                    exclusive_json(OUT / 'manifest.json', {'protocol_sha256': sha(PROTOCOL), 'aggregate_sha256': sha(OUT / 'aggregate.json'),
                        'subset_sha256': subsets, 'elapsed_seconds': round(time.monotonic()-start, 1), 'patient_level_output_emitted': False})
                    source = lc.old.native.source; source.base._atomic_completed(OUT / 'progress.json'); ok = True
        except Exception as e:
            if owned: exclusive_json(target / 'failure.json', {'status': 'execution_failed', 'phase': phase,
                'error_class': type(e).__name__ if type(e) in (ValueError, TypeError, KeyError, RuntimeError, OSError) else 'other', 'patient_level_output_emitted': False})
    print(json.dumps({'operation': 'prepare' if args.prepare else ('audit' if args.audit else 'run'),
        'status': 'completed' if ok else 'failed', 'patient_level_output_emitted': False}))
    return 0 if ok else 1


if __name__ == '__main__': raise SystemExit(main())
