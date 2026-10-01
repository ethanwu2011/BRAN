"""Frozen local head-only diagnostic; only closed aggregates leave FD silence."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import sha, exclusive_json
import bran_september_push_io_v1 as io
import run_bran_overnight_diagnostic_v1 as base
import run_bran_anchor_ablation_v2 as lineage
import bran_external_cbc_evaluation_v1 as ev
from bran_clinical_semantics_v1 import CBC_FIELDS

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'BRAN_MODALITY_UNION_PROTOCOL_V1.json'
OUT = ROOT / 'BRAN_MODALITY_UNION_V1'
AUDIT = ROOT / 'BRAN_MODALITY_UNION_AUDIT_V1'
S_ARMS = ('joint', 'clinical', 'retinal', 'raw_clinical', 'raw_retinal', 'raw_concat', 'late_average', 'union')
C_ARMS = ('joint', 'raw_clinical', 'raw_concat', 'union')
GROUPS = ('overall', 'low', 'middle', 'high')
BASELINE_MAP = {'joint': 'bran_both', 'clinical': 'bran_clinical', 'retinal': 'bran_retinal',
                'raw_clinical': 'raw_clinical', 'raw_retinal': 'raw_retinal',
                'raw_concat': 'raw_concat', 'late_average': 'late_average'}
BASELINE_PINS = {
    'BRAN_MATCHED_SCREENING_V1/aggregate.json': 'dee3e35af3c91fbf370c0ff7b29134fca25673ca1005859fa2d549649821092a',
    'BRAN_MATCHED_SCREENING_AUDIT_V1/audit.json': '64fceef06ae7b7214152a61200efb5c528a3a6d0c55cf34bf170e577bd8a6b6b'}
PARAMETERS = {
    'state_dim': 384, 'readout_dim': 387, 'outer_folds': 5, 'encoder_fit_steps': 0,
    'screening_recipe': 'existing matched ExtraTrees128 depth6 leaf10 max_features0.7 threads2',
    'screening_seed': 92381, 'screening_fitted_arms': 7,
    'cbc_recipe': 'StandardScaler and Ridge alpha1; outer-training observed targets',
    'cbc_pattern': 'all9CBCvalues_and_flags_hidden_before_encoding',
    'bootstrap_draws': 1000, 'bootstrap_seed': 91501, 'minimum_release_support': 20,
    'minimum_valid_cbc_resamples': 900, 'baseline_tolerance': 1e-10,
    'automatic_promotion': False, 'patient_level_output_permitted': False,
    'official_test_used': False, 'clinical_use_permitted': False,
    'scope': 'adaptive internal latent-union diagnostic; unequal dimension; not a calibrated joint posterior',
}
CODE = ('run_bran_modality_union_v1.py', 'bran_modality_union_v1.py',
        'test_bran_modality_union_v1.py', 'test_run_bran_modality_union_v1.py',
        'BRAN_MODALITY_UNION_DESIGN_V1.md', 'bran_research_bundle_api_v1.py',
        'bran_matched_screening_kernel_v1.py', 'run_bran_matched_screening_v1.py',
        'bran_september_push_io_v1.py', 'bran_external_cbc_evaluation_v1.py')


def require(ok, code='modality_union_contract_failed'):
    if not ok:
        raise ValueError(code)


def baseline():
    for name, digest in BASELINE_PINS.items():
        require(sha(ROOT / name) == digest, 'baseline_identity_changed')
    require(not (ROOT / 'BRAN_MATCHED_SCREENING_V1/failure.json').exists())
    a = json.loads((ROOT / 'BRAN_MATCHED_SCREENING_V1/aggregate.json').read_text())
    audit = json.loads((ROOT / 'BRAN_MATCHED_SCREENING_AUDIT_V1/audit.json').read_text())
    require(a['status'] == 'completed' and a['patient_level_output_emitted'] is False)
    require(audit['status'] == 'authenticated' and audit['aggregate_sha256'] == BASELINE_PINS['BRAN_MATCHED_SCREENING_V1/aggregate.json'])
    return a['results']['extra_trees']


def prepare():
    baseline()
    return {'schema': 'bran-modality-union-protocol-v1', 'status': 'frozen_before_execution',
            'parameters': PARAMETERS, 'baseline_sha256': BASELINE_PINS,
            'source': io.source_receipt(), 'code_sha256': io.code_closure(CODE), 'runtime': io.runtime()}


def validate_protocol(p):
    require(p == prepare(), 'protocol_changed')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def interval(draws):
    draws = np.asarray(draws, float)
    valid = draws[np.isfinite(draws)]
    require(len(valid) >= 900, 'bootstrap_support_failed')
    return [float(np.quantile(valid, .025)), float(np.quantile(valid, .975))]


def screen_summary(pred, labels, masks, folds, names, counts):
    endpoints = {}; macro_points = {a: [] for a in S_ARMS}; macro_draws = {a: [] for a in S_ARMS}
    for name in names:
        y = np.asarray(labels[name]); observed = np.asarray(masks[name], bool)
        if (sum(observed & (y == 0)) < 20 or sum(observed & (y == 1)) < 20 or
                any(not np.isfinite(pred[name][a][observed]).all() for a in S_ARMS)):
            endpoints[name] = {'status': 'unsupported'}; continue
        points = {a: base.fold_weighted_auc(y, pred[name][a], observed, folds) for a in S_ARMS}
        draws = {a: base._weighted_auc_draws(y, pred[name][a], observed, folds, counts) for a in S_ARMS}
        endpoints[name] = {'status': 'supported', 'arms': {a: {'auroc': float(points[a]), 'ci95': interval(draws[a])} for a in S_ARMS},
                           'contrasts': {a: {'delta': float(points['union'] - points[a]), 'ci95': interval(draws['union'] - draws[a])} for a in S_ARMS[:-1]}}
        for a in S_ARMS:
            macro_points[a].append(points[a]); macro_draws[a].append(draws[a])
    complete = all(x['status'] == 'supported' for x in endpoints.values())
    macro = None
    if complete:
        point = {a: float(np.mean(macro_points[a])) for a in S_ARMS}
        draw = {a: np.mean(macro_draws[a], axis=0) for a in S_ARMS}
        macro = {'arms': {a: {'auroc': point[a], 'ci95': interval(draw[a])} for a in S_ARMS},
                 'contrasts': {a: {'delta': point['union'] - point[a], 'ci95': interval(draw['union'] - draw[a])} for a in S_ARMS[:-1]}}
    return {'complete_panel': complete, 'endpoints': endpoints, 'macro': macro}


def cbc_summary(target, observed, pred, group_masks, counts):
    result = {}
    for j, field in enumerate(CBC_FIELDS):
        result[field] = {}
        for group in GROUPS:
            valid = observed[:, j] & group_masks[group][:, j] & np.isfinite(target[:, j])
            if sum(valid) < 20:
                result[field][group] = {'status': 'unsupported'}; continue
            require(all(np.isfinite(pred[a][valid, j]).all() for a in C_ARMS), 'nonfinite_cbc_prediction')
            weights = counts[:, valid].astype(float); denominator = weights.sum(1)
            errors = {a: np.abs(pred[a][valid, j] - target[valid, j]) for a in C_ARMS}
            draws = {a: np.divide(weights @ errors[a], denominator, out=np.full(len(counts), np.nan), where=denominator > 0) for a in C_ARMS}
            arms = {a: {'mae': float(errors[a].mean()), 'mse': float(np.square(errors[a]).mean()), 'ci95': interval(draws[a])} for a in C_ARMS}
            result[field][group] = {'status': 'supported', 'arms': arms,
                'contrasts': {a: {'delta': arms['union']['mae'] - arms[a]['mae'], 'ci95': interval(draws['union'] - draws[a])} for a in C_ARMS[:-1]}}
    return result


def decisions(screen, cbc):
    s = screen['macro']; hb = cbc['hemoglobin']['overall']
    return {
        'screen_union_over_joint': bool(s is not None and s['contrasts']['joint']['ci95'][0] > 0),
        'screen_union_over_all_references': bool(s is not None and all(c['ci95'][0] > 0 for c in s['contrasts'].values())),
        'cbc_hb_union_over_joint': bool(hb['status'] == 'supported' and hb['contrasts']['joint']['ci95'][1] < 0),
        'cbc_hb_union_over_all_references': bool(hb['status'] == 'supported' and all(c['ci95'][1] < 0 for c in hb['contrasts'].values())),
        'automatic_promotion': False,
    }


def validate_cell(cell, arms, metric):
    require(set(cell) == {'arms', 'contrasts'} and set(cell['arms']) == set(arms) and set(cell['contrasts']) == set(arms[:-1]))
    def ci(c, low, high):
        require(type(c) is list and len(c) == 2 and all(finite(x) for x in c) and low <= c[0] <= c[1] <= high)
    for v in cell['arms'].values():
        require(set(v) == ({'auroc', 'ci95'} if metric == 'auroc' else {'mae', 'mse', 'ci95'}))
        require(finite(v[metric]) and 0 <= v[metric] <= (1 if metric == 'auroc' else float('inf')))
        if metric == 'mae': require(finite(v['mse']) and v['mse'] + 1e-10 >= v['mae'] ** 2)
        ci(v['ci95'], 0, 1 if metric == 'auroc' else float('inf'))
    for a, v in cell['contrasts'].items():
        require(set(v) == {'delta', 'ci95'} and finite(v['delta']))
        require(abs(v['delta'] - (cell['arms']['union'][metric] - cell['arms'][a][metric])) < 1e-10)
        ci(v['ci95'], -1 if metric == 'auroc' else -float('inf'), 1 if metric == 'auroc' else float('inf'))


def validate_result(a, p):
    require(set(a) == {'schema', 'status', 'paired_people', 'recorded_conditions', 'screening', 'completion', 'decisions', 'baseline_replay', 'patient_level_output_emitted', 'encoder_refitted', 'official_test_used', 'adaptive_development'})
    require(a['schema'] == 'bran-modality-union-aggregate-v1' and a['status'] == 'completed' and type(a['paired_people']) is int and a['paired_people'] == 1928 and type(a['recorded_conditions']) is int and a['recorded_conditions'] == 26)
    require(a['adaptive_development'] is True and all(a[k] is False for k in ('patient_level_output_emitted', 'encoder_refitted', 'official_test_used')))
    require(a['baseline_replay'] == {'all26_endpoints_all7_reference_arms': True, 'absolute_tolerance': 1e-10})
    s = a['screening']; require(set(s) == {'endpoints', 'complete_panel', 'macro'} and type(s['complete_panel']) is bool)
    require(set(s['endpoints']) == set(p['source']['endpoint_names']))
    for cell in s['endpoints'].values():
        if cell == {'status': 'unsupported'}: continue
        require(set(cell) == {'status', 'arms', 'contrasts'} and cell['status'] == 'supported')
        validate_cell({k: v for k, v in cell.items() if k != 'status'}, S_ARMS, 'auroc')
    require(s['complete_panel'] == all(c['status'] == 'supported' for c in s['endpoints'].values()))
    if s['complete_panel']:
        validate_cell(s['macro'], S_ARMS, 'auroc')
        for arm in S_ARMS:
            require(abs(s['macro']['arms'][arm]['auroc'] - np.mean([c['arms'][arm]['auroc'] for c in s['endpoints'].values()])) < 1e-10)
    else: require(s['macro'] is None)
    require(set(a['completion']) == set(CBC_FIELDS))
    for groups in a['completion'].values():
        require(set(groups) == set(GROUPS))
        for cell in groups.values():
            if cell == {'status': 'unsupported'}: continue
            require(set(cell) == {'status', 'arms', 'contrasts'} and cell['status'] == 'supported')
            validate_cell({k: v for k, v in cell.items() if k != 'status'}, C_ARMS, 'mae')
    require(all(type(v) is bool for v in a['decisions'].values()) and a['decisions'] == decisions(s, a['completion']))


def full_batch_infer(model, c, cm, r, rm, age, route):
    from bran_research_bundle_api_v1 import encode
    return encode(model, c, cm, r, rm, age, route, batch_size=max(1, len(c)))


def tail_masks(train_y, values):
    train_y = np.asarray(train_y, float); values = np.asarray(values, float)
    require(train_y.ndim == values.ndim == 1 and len(train_y) >= 20 and np.isfinite(train_y).all())
    lo, hi = np.quantile(train_y, [.1, .9])
    if not lo < hi:
        return {g: np.zeros(len(values), bool) for g in ('low', 'middle', 'high')}
    return {'low': values <= lo, 'high': values >= hi, 'middle': (values > lo) & (values < hi)}


def run(p):
    import torch
    from bran_matched_screening_kernel_v1 import fit_predict, late_fusion_average
    from bran_modality_union_v1 import make_designs
    torch.set_num_threads(2)
    ctx, folds, c0, cm0, eligible, r0, rm, ages, names = io.load_context()
    require(len(folds) == 1928 and set(np.unique(folds)) == set(range(5)))
    endpoints = p['source']['endpoint_names']; old = baseline()
    pred = {e: {a: np.full(len(folds), np.nan) for a in S_ARMS} for e in endpoints}
    cache = []
    indices = np.array([names.index(f) for f in CBC_FIELDS])
    target = c0[:, indices].copy(); observed = (cm0 & eligible)[:, indices].copy()
    score_observed = observed.copy()
    cpred = {a: np.full(target.shape, np.nan) for a in C_ARMS}
    groups = {g: np.zeros(target.shape, bool) for g in GROUPS}
    for fold in range(5):
        base._atomic_progress(OUT / 'progress.json', 'baseline_screening', fold)
        tr, te = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        transform = base.FoldTransform(c0, cm0, eligible, r0, rm, ages, tr)
        c, cm, r, age = transform.apply(c0, cm0, eligible, r0, rm, ages)
        model = io.load_control(fold, transform, p['source'])
        designs, abstain = make_designs(model, c, cm, r, rm, age, infer=full_batch_infer)
        require(not abstain.any(), 'screening_complete_comparison_requires_physiology')
        designs.update({'raw_clinical': np.c_[c, cm, age], 'raw_retinal': np.c_[r, rm, age], 'raw_concat': np.c_[c, cm, r, rm, age]})
        inner, identity = base._inner_context(ctx, tr, fold)
        require(identity == p['source']['authentication']['inner_fold_sha256'][fold], 'inner_identity_changed')
        full_inner = np.full(len(folds), -1, int); full_inner[tr] = inner
        for e in endpoints:
            for a in S_ARMS[:6]:
                values, _ = fit_predict(designs[a], ctx['labels_by_source'][e], ctx['observed_by_source'][e], tr, te, full_inner, family='extra_trees', seed=92381 + fold)
                pred[e][a][te] = values
            pred[e]['late_average'][te] = late_fusion_average(pred[e]['raw_clinical'][te], pred[e]['raw_retinal'][te])
        # Cache only the new design in RAM until all historical points replay.
        cache.append((tr, te, full_inner, designs['union']))
        base._atomic_progress(OUT / 'progress.json', 'whole_cbc_fixed_readouts', fold)
        hidden_c, hidden_m, _, _ = ev.whole_cbc_inputs(c, cm, names)
        masked, abstain = make_designs(model, hidden_c, hidden_m, r, rm, age, infer=full_batch_infer)
        raw = {'raw_clinical': np.c_[hidden_c, hidden_m, age], 'raw_concat': np.c_[hidden_c, hidden_m, r, rm, age]}
        cdesigns = {a: masked[a] if a in ('joint', 'union') else raw[a] for a in C_ARMS}
        score_observed[te] &= ~abstain[te, None]
        for j in range(9):
            fit_obs = observed[:, j] & ~abstain
            for a in C_ARMS:
                cpred[a][te, j] = ev.fixed_cbc_probe(cdesigns[a], target[:, j], fit_obs, tr, te)
            train_y = target[tr, j][fit_obs[tr]]
            require(len(train_y) >= 20 and np.isfinite(train_y).all())
            groups['overall'][te, j] = True
            for group, mask in tail_masks(train_y, target[te, j]).items():
                groups[group][te, j] = mask
    for e in endpoints:
        for a, old_a in BASELINE_MAP.items():
            point = base.fold_weighted_auc(ctx['labels_by_source'][e], pred[e][a], ctx['observed_by_source'][e], folds)
            require(abs(point - old['endpoints'][e]['arms'][old_a]['auroc']) <= 1e-10, 'baseline_replay_failed')
    for fold, (tr, te, full_inner, x) in enumerate(cache):
        base._atomic_progress(OUT / 'progress.json', 'new_union_screening', fold)
        for e in endpoints:
            pred[e]['union'][te], _ = fit_predict(x, ctx['labels_by_source'][e], ctx['observed_by_source'][e], tr, te, full_inner, family='extra_trees', seed=92381 + fold)
    base._atomic_progress(OUT / 'progress.json', 'paired_aggregate_uncertainty')
    counts = ev.paired_counts(folds, draws=1000, seed=91501)
    screening = screen_summary(pred, ctx['labels_by_source'], ctx['observed_by_source'], folds, endpoints, counts)
    completion = cbc_summary(target, score_observed, cpred, groups, counts)
    return {'schema': 'bran-modality-union-aggregate-v1', 'status': 'completed', 'paired_people': 1928, 'recorded_conditions': 26,
            'screening': screening, 'completion': completion, 'decisions': decisions(screening, completion),
            'baseline_replay': {'all26_endpoints_all7_reference_arms': True, 'absolute_tolerance': 1e-10},
            'patient_level_output_emitted': False, 'encoder_refitted': False, 'official_test_used': False, 'adaptive_development': True}


def audit(p, digest):
    from patient_atlas_v6_2_expanded_endpoint_evaluation import EXACT_OUTER_FOLD_HASH, EXACT_INNER_FOLD_ASSIGNMENT_SHA256
    require(not (OUT / 'failure.json').exists() and (OUT / 'aggregate.json').is_file() and (OUT / 'manifest.json').is_file())
    a = json.loads((OUT / 'aggregate.json').read_text()); validate_result(a, p)
    m = json.loads((OUT / 'manifest.json').read_text())
    require(set(m) == {'protocol_sha256', 'aggregate_sha256', 'elapsed_seconds', 'patient_level_output_emitted'})
    require(m['protocol_sha256'] == digest and m['aggregate_sha256'] == sha(OUT / 'aggregate.json') and m['patient_level_output_emitted'] is False and finite(m['elapsed_seconds']) and m['elapsed_seconds'] >= 0)
    auth = p['source']['authentication']
    require(auth['outer_fold_sha256'] == EXACT_OUTER_FOLD_HASH and auth['inner_fold_sha256'] == list(EXACT_INNER_FOLD_ASSIGNMENT_SHA256))
    return {'schema': 'bran-modality-union-audit-v1', 'status': 'authenticated', 'protocol_sha256': digest,
            'aggregate_sha256': m['aggregate_sha256'], 'manifest_sha256': sha(OUT / 'manifest.json'),
            'outer_fold_sha256': auth['outer_fold_sha256'], 'inner_fold_sha256': auth['inner_fold_sha256'],
            'checkpoint_sha256': p['source']['checkpoint_sha256'], 'paired_people': 1928, 'recorded_conditions': 26,
            'baseline_replay_passed': True, 'patient_level_output_emitted': False, 'automatic_promotion': False}


def main():
    parser = argparse.ArgumentParser(); g = parser.add_mutually_exclusive_group(required=True)
    for mode in ('prepare', 'run', 'audit'): g.add_argument('--' + mode, action='store_true')
    parser.add_argument('--protocol-sha256'); args = parser.parse_args()
    ok = False; owned = False; phase = 'protocol'; target = AUDIT if args.audit else OUT
    start = time.monotonic(); os.umask(0o077)
    with _quiet():
        try:
            if args.prepare:
                require(not OUT.exists() and not AUDIT.exists()); exclusive_json(PROTOCOL, prepare()); ok = True
            else:
                require(args.protocol_sha256 and sha(PROTOCOL) == args.protocol_sha256)
                p = json.loads(PROTOCOL.read_text()); validate_protocol(p)
                target.mkdir(); owned = True
                if args.run:
                    with open('/private/tmp/bran_modality_union_v1.lock', 'a') as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        phase = 'experiment'; a = run(p); phase = 'terminal_validation'
                        validate_protocol(p); validate_result(a, p)
                        exclusive_json(OUT / 'aggregate.json', a)
                        exclusive_json(OUT / 'manifest.json', {'protocol_sha256': args.protocol_sha256, 'aggregate_sha256': sha(OUT / 'aggregate.json'), 'elapsed_seconds': round(time.monotonic() - start, 1), 'patient_level_output_emitted': False})
                        base._atomic_completed(OUT / 'progress.json')
                else:
                    phase = 'audit'; exclusive_json(AUDIT / 'audit.json', audit(p, args.protocol_sha256))
                ok = True
        except Exception as error:
            if owned:
                exclusive_json(target / 'failure.json', {'status': 'execution_failed', 'phase': phase,
                    'error_class': type(error).__name__ if type(error) in (ValueError, TypeError, KeyError, RuntimeError, OSError) else 'other_execution_error',
                    'patient_level_output_emitted': False})
    mode = 'prepare' if args.prepare else ('audit' if args.audit else 'run')
    print(json.dumps({'operation': mode, 'status': 'completed' if ok else 'failed', 'patient_level_output_emitted': False}))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
