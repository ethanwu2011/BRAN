"""Local-only V6 joint stress estimand and closed research-lead decision.

Bootstrap draws/predictions remain private. Public cells enforce support and
complement suppression. This does not change the historical promotion gate.
"""
import numpy as np
import bran_multisource_outcome_metrics_v2 as metrics
from bran_missingness_stress_metrics_v1 import safe_coverage
import run_bran_overnight_diagnostic_v1 as auc
from bran_clinical_semantics_v1 import CBC_FIELDS

ROLES = ('V5', 'C', 'S')
PATTERNS = ('clinical_drop25','clinical_drop50','clinical_drop75','no_retina','all_clinical_hidden')
CONTRASTS = {'S_minus_V5': ('S','V5'), 'S_minus_C': ('S','C')}
ERROR = 'source_pattern_metrics_v6_invalid'
PRIORITY_FIELDS = ('hemoglobin', 'plt', 'wbc')
PROTECTED_COMPLETION = tuple([('whole_cbc_hidden', f) for f in CBC_FIELDS] +
    [(p, f) for p in ('single_target_hidden', 'whole_cbc_no_retina', 'single_target_no_retina')
     for f in PRIORITY_FIELDS])
COMPLETION_CHECK_KEYS = frozenset(p+'__'+f for p,f in PROTECTED_COMPLETION) | {
    'whole_cbc_hidden__historical_low_hb'}


def require(ok):
    if not ok: raise ValueError(ERROR) from None


def stress_summary(predictions, labels, observed, folds, names, counts):
    """One paired bootstrap across five patterns; same endpoints and people.

    An endpoint is included only when it passes across ALL five patterns. Within
    a pattern, all three arms share its exact eligible rows. The total estimand
    equally averages pattern macro-AUROCs. Missing pattern support fails closed.
    """
    try:
        metrics.validate_counts(counts,folds)
        n = len(folds)
        require(labels.shape == observed.shape == (n,26) and observed.dtype == bool
                and len(names)==len(set(names))==26 and np.isfinite(labels[observed]).all()
                and np.isin(labels[observed],[0,1]).all())
        require(set(predictions)==set(PATTERNS))
        coverage, row_masks = {}, {}
        for pattern in PATTERNS:
            arms = predictions[pattern]
            require(set(arms)==set(ROLES))
            available = {}
            for role in ROLES:
                v = arms[role]
                require(type(v) is np.ndarray and v.shape == labels.shape and v.dtype.kind=='f')
                mask = np.isfinite(v).all(1)
                require(np.all(mask | np.isnan(v).all(1)) and np.all((v[mask]>=0)&(v[mask]<=1)))
                available[role] = mask
            common = np.logical_and.reduce(list(available.values()))
            coverage[pattern] = {'arms':{k:safe_coverage(v) for k,v in available.items()},
                                 'matched':safe_coverage(common)}
            row_masks[pattern] = []
            for j in range(26):
                valid = common & observed[:,j]
                for fold in range(5):
                    local = valid & (folds==fold)
                    if not np.any(local & (labels[:,j]==0)) or not np.any(local & (labels[:,j]==1)):
                        valid[folds==fold] = False
                row_masks[pattern].append(valid)
        endpoints = [j for j in range(26) if all(
            np.count_nonzero(row_masks[p][j] & (labels[:,j]==0))>=20 and
            np.count_nonzero(row_masks[p][j] & (labels[:,j]==1))>=20 for p in PATTERNS)]
        # A locked 26-endpoint primary cannot silently shrink to a favorable subset.
        if len(endpoints)!=26:
            return {'status':'unsupported','complete_26_panel':False,'coverage':coverage,
                    'summary':None,'patterns':{},'endpoint_count':len(endpoints)}
        points = {r:[] for r in ROLES}; draws = {r:[] for r in ROLES}; by_pattern = {}
        for pattern in PATTERNS:
            pp = {r:[] for r in ROLES}; dd = {r:[] for r in ROLES}
            for j in endpoints:
                for role in ROLES:
                    p, valid = predictions[pattern][role][:,j], row_masks[pattern][j]
                    pp[role].append(auc.fold_weighted_auc(labels[:,j],p,valid,folds))
                    dd[role].append(auc._weighted_auc_draws(labels[:,j],p,valid,folds,counts))
            m = {r:float(np.mean(pp[r])) for r in ROLES}
            d = {r:np.mean(dd[r],axis=0) for r in ROLES}
            by_pattern[pattern] = metrics._cell(m,d,CONTRASTS,'auroc')
            for r in ROLES: points[r].append(m[r]); draws[r].append(d[r])
        summary = metrics._cell({r:float(np.mean(points[r])) for r in ROLES},
                                {r:np.mean(draws[r],axis=0) for r in ROLES},CONTRASTS,'auroc')
        return {'status':summary['status'],'complete_26_panel':True,'summary':summary,
                'patterns':by_pattern,'coverage':coverage,'endpoint_count':26}
    except Exception:
        raise ValueError(ERROR) from None


def protected_completion_checks(completion, low_tail):
    """Historical protected scope, prospectively compared against V5 only.

    All-nine whole-CBC with retina; priority-three single/no-retina contexts;
    historical low-Hb with whole-CBC hidden. Other cells remain descriptive.
    """
    checks = {}
    for pattern, field in PROTECTED_COMPLETION:
        value = completion.get(pattern,{}).get(field,{}).get('groups',{}).get('overall',{})
        checks[pattern+'__'+field] = (value.get('status')=='supported'
            and value['arms']['S']['mae'] <= value['arms']['V5']['mae'])
    for pattern in ('whole_cbc_hidden',):
        v = low_tail.get(pattern,{})
        checks[pattern+'__historical_low_hb'] = (v.get('status')=='supported'
            and v['arms']['S']['mae'] <= v['arms']['V5']['mae'])
    return checks


def decide(stress, available_screening, completion_checks, authenticated):
    require(type(authenticated) is bool and type(completion_checks) is dict
            and set(completion_checks)==COMPLETION_CHECK_KEYS
            and all(type(v) is bool for v in completion_checks.values()))
    summary = stress.get('summary') or {}
    stress_ok = (stress.get('complete_26_panel') is True and summary.get('status')=='supported'
        and all(summary['contrasts'][key]['ci95'][0]>0 for key in CONTRASTS))
    cell = available_screening.get('macro') or {}
    screen_ok = (available_screening.get('complete_26_panel') is True and cell.get('status')=='supported'
        and cell['arms']['S']['auroc'] >= cell['arms']['V5']['auroc'])
    checks = {'stress_improves_both_references':bool(stress_ok),'available_screening_nonworse':bool(screen_ok),
              'protected_completion_nonworse':bool(all(completion_checks.values())),
              'authenticated':authenticated}
    eligible = all(checks.values())
    return {'checks':checks, 'completion_checks':completion_checks, 'research_lead_eligible':eligible,
        'fallback':'V5' if not eligible else None, 'candidate_promoted':False,
        'historical_gate_changed':False, 'scientific_goal_achieved':False}
