"""Closed aggregate summaries; patient arrays are caller-local, never serialized."""
import math
import numpy as np
from bran_clinical_semantics_v1 import CBC_FIELDS
import run_bran_overnight_diagnostic_v1 as base

VERSIONS = ('initial', 'continued', 'student')
ROUTES = ('both', 'clinical', 'retinal')
S_ARMS = tuple(v + '_' + r for v in VERSIONS for r in ROUTES) + ('raw_clinical', 'raw_retinal', 'raw_concat', 'late_average')
S_REFS = tuple(a for a in S_ARMS if a != 'student_both')
C_ARMS = VERSIONS + ('raw_clinical', 'raw_concat')
C_REFS = tuple(a for a in C_ARMS if a != 'student')
GROUPS = ('overall', 'low', 'middle', 'high')


def require(ok, code='distillation_aggregate_contract_failed'):
    if not ok: raise ValueError(code)


def finite(v): return type(v) in (int, float) and math.isfinite(v)


def interval(draws):
    values = np.asarray(draws, float); values = values[np.isfinite(values)]
    require(len(values) >= 900, 'distillation_bootstrap_support_failed')
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def screening(pred, labels, masks, folds, names, counts):
    result = {}; points = {a: [] for a in S_ARMS}; draws = {a: [] for a in S_ARMS}
    for name in names:
        y = np.asarray(labels[name]); valid = np.asarray(masks[name], bool)
        if sum(valid & (y == 0)) < 20 or sum(valid & (y == 1)) < 20:
            result[name] = {'status': 'unsupported'}; continue
        require(all(np.isfinite(pred[name][a][valid]).all() for a in S_ARMS))
        ps = {a: base.fold_weighted_auc(y, pred[name][a], valid, folds) for a in S_ARMS}
        ds = {a: base._weighted_auc_draws(y, pred[name][a], valid, folds, counts) for a in S_ARMS}
        result[name] = {'status': 'supported', 'arms': {a: {'auroc': float(ps[a]), 'ci95': interval(ds[a])} for a in S_ARMS},
                       'contrasts': {a: {'delta': float(ps['student_both'] - ps[a]), 'ci95': interval(ds['student_both'] - ds[a])} for a in S_REFS}}
        for a in S_ARMS: points[a].append(ps[a]); draws[a].append(ds[a])
    complete = all(c['status'] == 'supported' for c in result.values()); macro = None
    if complete:
        ps = {a: float(np.mean(points[a])) for a in S_ARMS}; ds = {a: np.mean(draws[a], axis=0) for a in S_ARMS}
        macro = {'arms': {a: {'auroc': ps[a], 'ci95': interval(ds[a])} for a in S_ARMS},
                 'contrasts': {a: {'delta': ps['student_both'] - ps[a], 'ci95': interval(ds['student_both'] - ds[a])} for a in S_REFS}}
    return {'complete_panel': complete, 'endpoints': result, 'macro': macro}


def completion(target, observed, predictions, groups, counts):
    result = {}
    for j, field in enumerate(CBC_FIELDS):
        result[field] = {}
        for group in GROUPS:
            valid = observed[:, j] & groups[group][:, j] & np.isfinite(target[:, j])
            if sum(valid) < 20: result[field][group] = {'status': 'unsupported'}; continue
            require(all(np.isfinite(predictions[a][valid, j]).all() for a in C_ARMS))
            weights = counts[:, valid].astype(float); den = weights.sum(1)
            errors = {a: np.abs(predictions[a][valid, j] - target[valid, j]) for a in C_ARMS}
            ds = {a: np.divide(weights @ errors[a], den, out=np.full(len(counts), np.nan), where=den > 0) for a in C_ARMS}
            arms = {a: {'mae': float(errors[a].mean()), 'mse': float(np.square(errors[a]).mean()), 'ci95': interval(ds[a])} for a in C_ARMS}
            result[field][group] = {'status': 'supported', 'arms': arms,
                'contrasts': {a: {'delta': arms['student']['mae'] - arms[a]['mae'], 'ci95': interval(ds['student'] - ds[a])} for a in C_REFS}}
    return result


def decisions(screen, cbc):
    macro = screen['macro']
    checks = {'screen_over_' + ref: bool(macro is not None and macro['contrasts'][ref]['ci95'][0] > 0)
              for ref in ('initial_both', 'continued_both', 'raw_clinical', 'late_average', 'student_clinical', 'student_retinal')}
    checks['screen_over_all_references'] = bool(macro is not None and all(x['ci95'][0] > 0 for x in macro['contrasts'].values()))
    checks['priority_cbc_point_nonworse'] = all(cbc[f]['overall']['status'] == 'supported' and
        all(cbc[f]['overall']['contrasts'][r]['delta'] <= 0 for r in ('initial', 'continued')) for f in ('hemoglobin', 'plt', 'wbc'))
    checks['low_hb_point_nonworse'] = bool(cbc['hemoglobin']['low']['status'] == 'supported' and
        all(cbc['hemoglobin']['low']['contrasts'][r]['delta'] <= 0 for r in ('initial', 'continued')))
    checks['advancement_supported'] = all(checks.values()); checks['automatic_promotion'] = False
    return checks


def validate_cell(cell, arms, refs, candidate, key):
    require(set(cell) == {'arms', 'contrasts'} and set(cell['arms']) == set(arms) and set(cell['contrasts']) == set(refs))
    def ci(v, lo, hi):
        require(type(v) is list and len(v) == 2 and all(finite(x) for x in v) and lo <= v[0] <= v[1] <= hi)
    for x in cell['arms'].values():
        require(set(x) == ({'auroc', 'ci95'} if key == 'auroc' else {'mae', 'mse', 'ci95'}))
        require(finite(x[key]) and 0 <= x[key] <= (1 if key == 'auroc' else math.inf))
        if key == 'mae': require(finite(x['mse']) and x['mse'] + 1e-10 >= x['mae'] ** 2)
        ci(x['ci95'], 0, 1 if key == 'auroc' else math.inf)
    for ref, x in cell['contrasts'].items():
        require(set(x) == {'delta', 'ci95'} and finite(x['delta']))
        require(abs(x['delta'] - (cell['arms'][candidate][key] - cell['arms'][ref][key])) < 1e-10)
        ci(x['ci95'], -1 if key == 'auroc' else -math.inf, 1 if key == 'auroc' else math.inf)


def validate(screen, cbc, flags, names):
    require(set(screen) == {'complete_panel', 'endpoints', 'macro'} and type(screen['complete_panel']) is bool)
    require(set(screen['endpoints']) == set(names) and len(names) == 26)
    for x in screen['endpoints'].values():
        if x == {'status': 'unsupported'}: continue
        require(set(x) == {'status', 'arms', 'contrasts'} and x['status'] == 'supported')
        validate_cell({k: v for k, v in x.items() if k != 'status'}, S_ARMS, S_REFS, 'student_both', 'auroc')
    complete = all(x['status'] == 'supported' for x in screen['endpoints'].values())
    require(screen['complete_panel'] == complete)
    if complete:
        validate_cell(screen['macro'], S_ARMS, S_REFS, 'student_both', 'auroc')
        for arm in S_ARMS:
            require(abs(screen['macro']['arms'][arm]['auroc'] - np.mean([x['arms'][arm]['auroc'] for x in screen['endpoints'].values()])) < 1e-10)
    else: require(screen['macro'] is None)
    require(set(cbc) == set(CBC_FIELDS))
    for field in cbc.values():
        require(set(field) == set(GROUPS))
        for x in field.values():
            if x == {'status': 'unsupported'}: continue
            require(set(x) == {'status', 'arms', 'contrasts'} and x['status'] == 'supported')
            validate_cell({k: v for k, v in x.items() if k != 'status'}, C_ARMS, C_REFS, 'student', 'mae')
    require(all(type(v) is bool for v in flags.values()) and flags == decisions(screen, cbc))
