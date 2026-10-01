"""Aggregate-only U1 measurement-interface metrics; arrays never leave locally.

The 25-condition panel excludes the assay-defined elevated-A1C endpoint.
Bonferroni percentile intervals control the two prespecified macro contrasts;
per-condition intervals are explicitly exploratory, not adjusted discoveries.
"""
import numpy as np

import run_bran_overnight_diagnostic_v1 as auc
from bran_multisource_outcome_metrics_v2 import validate_counts, completion
from bran_missingness_stress_metrics_v1 import safe_coverage
from bran_clinical_semantics_v1 import CBC_FIELDS

ARMS = ('median', 'ridge', 'native_no_retina', 'observed_cbc',
        'native_with_retina', 'noncbc_only')
CONTRASTS = {'native_minus_median': ('native_no_retina', 'median'),
             'native_minus_ridge': ('native_no_retina', 'ridge')}


def require(ok):
    if not ok:
        raise ValueError('completion_utility_u1_metrics_contract_failed') from None


def interval(draws, *, family_size=1):
    require(draws.shape == (1000,) and family_size in (1, 2))
    finite = draws[np.isfinite(draws)]
    if len(finite) < 900:
        return None
    alpha = .05 / family_size
    return [float(v) for v in np.quantile(finite, [alpha/2, 1-alpha/2])]


def cell(points, draws, *, simultaneous=False):
    cis = {arm: interval(draws[arm]) for arm in ARMS}
    differences = {key: draws[left] - draws[right]
                   for key, (left, right) in CONTRASTS.items()}
    dc = {key: interval(value, family_size=2 if simultaneous else 1)
          for key, value in differences.items()}
    if any(v is None for v in (*cis.values(), *dc.values())):
        return {'status': 'unsupported'}
    return {'status': 'supported',
            'arms': {arm: {'auroc': float(points[arm]), 'ci95': cis[arm]} for arm in ARMS},
            'contrasts': {key: {
                'delta_auroc': float(points[left] - points[right]),
                'interval': dc[key],
                'interval_kind': 'family95_bonferroni_percentile' if simultaneous else 'marginal95_percentile'}
                for key, (left, right) in CONTRASTS.items()}}


def summarize(result, target, labels, labelmask, folds, names, counts):
    validate_counts(counts, folds)
    n = len(folds)
    require(len(names) == len(set(names)) == 25 and 'mh_a1c' not in names)
    require(labels.shape == labelmask.shape == (n, 25) and labelmask.dtype == bool)
    require(np.isfinite(labels[labelmask]).all() and np.isin(labels[labelmask], [0, 1]).all())
    predictions = result['predictions']
    cohort = result['cohort_mask']
    require(cohort.shape == (n,) and cohort.dtype == bool and set(predictions) == set(ARMS))
    for value in predictions.values():
        require(value.shape == (n, 25) and value.dtype.kind == 'f' and not np.isinf(value).any())
        require(np.isnan(value[~cohort]).all())
        finite = np.isfinite(value)
        require(np.all((value[finite] >= 0) & (value[finite] <= 1)))
    require(result['endpoint_fold_support'].shape == (5, 25)
            and result['endpoint_fold_support'].dtype == bool)
    endpoints = {}
    macro_points = {arm: [] for arm in ARMS}
    macro_draws = {arm: [] for arm in ARMS}
    for j, name in enumerate(names):
        require(type(name) is str and name)
        # Never silently drop an unsupported outer fold to improve a panel.
        if not result['endpoint_fold_support'][:, j].all():
            endpoints[name] = {'status': 'unsupported'}
            continue
        valid = cohort & labelmask[:, j]
        require(all(np.isfinite(value[valid, j]).all() for value in predictions.values()))
        y = labels[:, j]
        if (np.count_nonzero(valid & (y == 0)) < 20
                or np.count_nonzero(valid & (y == 1)) < 20):
            endpoints[name] = {'status': 'unsupported'}
            continue
        # The five-fold weighted estimand requires both classes in every fold.
        if any(not (np.any(valid & (folds == f) & (y == 0))
                    and np.any(valid & (folds == f) & (y == 1))) for f in range(5)):
            endpoints[name] = {'status': 'unsupported'}
            continue
        points = {arm: auc.fold_weighted_auc(y, predictions[arm][:, j], valid, folds) for arm in ARMS}
        draws = {arm: auc._weighted_auc_draws(y, predictions[arm][:, j], valid, folds, counts) for arm in ARMS}
        endpoints[name] = cell(points, draws)
        if endpoints[name]['status'] == 'supported':
            for arm in ARMS:
                macro_points[arm].append(points[arm])
                macro_draws[arm].append(draws[arm])
    complete = all(v['status'] == 'supported' for v in endpoints.values())
    macro = (cell({arm: float(np.mean(macro_points[arm])) for arm in ARMS},
                  {arm: np.mean(macro_draws[arm], axis=0) for arm in ARMS}, simultaneous=True)
             if complete else {'status': 'unsupported'})
    observed = np.repeat(cohort[:, None], 9, axis=1)
    require(target.shape == observed.shape and np.isfinite(target[observed]).all())
    mp = result['measurement_predictions']
    require(set(mp) == {'median', 'ridge', 'native_no_retina', 'native_with_retina'})
    masked_mp = {key: np.where(observed, value, np.nan) for key, value in mp.items()}
    measurements = completion(masked_mp, target, observed, folds, counts,
        contrasts=CONTRASTS, low_hb=True)
    return {'schema': 'bran-v5-completion-utility-u1-aggregate',
            'status': 'completed', 'complete_prespecified_25_panel': complete,
            'screening_macro': macro, 'screening_endpoints': endpoints,
            'measurement_completion': measurements,
            'cohort_coverage': safe_coverage(cohort),
            'cohort_rule': 'all_nine_cbc_originally_observed_and_noncbc_physiology_present',
            'screening_metric': 'mean_of_25_participant_weighted_outer_fold_aurocs',
            'primary_family': list(CONTRASTS),
            'primary_family_interval': 'two_comparisons_bonferroni_percentile_family95',
            'endpoint_intervals': 'exploratory_marginal95_not_multiplicity_adjusted',
            'bootstrap': {'draws': 1000, 'minimum_valid': 900, 'seed': 99221,
                          'sampling': 'paired_participant_within_outer_fold',
                          'uncertainty_scope': 'fixed_fits_not_model_selection'},
            'replacement_provenance': {
                'median': 'training_median_not_observed', 'ridge': 'estimated_not_observed',
                'native_no_retina': 'estimated_no_retina_not_observed',
                'native_with_retina': 'estimated_with_retina_not_observed',
                'observed_cbc': 'genuinely_observed_contextual_reference',
                'noncbc_only': 'separate_readout_no_cbc_value_or_flag'},
            'limitations': ['reused_development_not_external_validation',
                'selected_complete_cbc_population_not_natural_missingness',
                'no_new_information_claim_from_imputation',
                'with_retina_secondary_not_information_matched_to_clinical_imputers',
                'no_interval_calibration_or_blood_draw_replacement_claim',
                'assay_defined_mh_a1c_excluded_native26_result_unchanged'],
            'encoder_updated': False, 'checkpoint_inference_replayed': True,
            'readout_fit_replayed': True, 'historical_gates_changed': False,
            'candidate_promoted': False, 'patient_level_output_emitted': False}


def validate_aggregate(value, names):
    """Closed field sets and support-aware scalar-only release contract."""
    top = {'schema', 'status', 'complete_prespecified_25_panel', 'screening_macro',
           'screening_endpoints', 'measurement_completion', 'cohort_coverage',
           'cohort_rule', 'screening_metric', 'primary_family', 'primary_family_interval',
           'endpoint_intervals', 'bootstrap', 'replacement_provenance', 'limitations',
           'encoder_updated', 'checkpoint_inference_replayed', 'readout_fit_replayed',
           'historical_gates_changed', 'candidate_promoted', 'patient_level_output_emitted'}
    require(set(value) == top and value['schema'] == 'bran-v5-completion-utility-u1-aggregate')
    require(value['status'] == 'completed' and set(value['screening_endpoints']) == set(names)
            and len(names) == 25 and 'mh_a1c' not in names)
    for flag in ('encoder_updated', 'historical_gates_changed', 'candidate_promoted', 'patient_level_output_emitted'):
        require(value[flag] is False)
    for flag in ('checkpoint_inference_replayed', 'readout_fit_replayed'):
        require(value[flag] is True)
    for item in [value['screening_macro'], *value['screening_endpoints'].values()]:
        if item['status'] == 'unsupported':
            require(set(item) == {'status'})
            continue
        require(item['status'] == 'supported' and set(item) == {'status', 'arms', 'contrasts'})
        require(set(item['arms']) == set(ARMS) and set(item['contrasts']) == set(CONTRASTS))
        for row in item['arms'].values():
            require(set(row) == {'auroc', 'ci95'} and 0 <= row['auroc'] <= 1)
            require(len(row['ci95']) == 2 and 0 <= row['ci95'][0] <= row['ci95'][1] <= 1)
        for row in item['contrasts'].values():
            require(set(row) == {'delta_auroc', 'interval', 'interval_kind'})
            require(len(row['interval']) == 2 and -1 <= row['interval'][0] <= row['interval'][1] <= 1)
            require(-1 <= row['delta_auroc'] <= 1)
    require(value['complete_prespecified_25_panel'] is
            all(v['status'] == 'supported' for v in value['screening_endpoints'].values()))
    require(not value['complete_prespecified_25_panel'] or value['screening_macro']['status'] == 'supported')
    def coverage(item):
        if item.get('status') == 'withheld':
            require(set(item) == {'status'})
            return
        require(set(item) == {'status', 'supported', 'total'} and item['status'] == 'released')
        s, n = item['supported'], item['total']
        require(type(s) is int and type(n) is int and n >= 20 and 0 <= s <= n)
        require((s == 0 or s >= 20) and (n-s == 0 or n-s >= 20))
    coverage(value['cohort_coverage'])
    require(set(value['measurement_completion']) == set(CBC_FIELDS))
    measurement_arms = {'median', 'ridge', 'native_no_retina', 'native_with_retina'}
    for field, item in value['measurement_completion'].items():
        require(set(item) == {'prediction_coverage_among_observed', 'groups'})
        coverage(item['prediction_coverage_among_observed'])
        require(set(item['groups']) == ({'overall', 'hb_below_12_research_stratum'}
                                        if field == 'hemoglobin' else {'overall'}))
        for group in item['groups'].values():
            if group['status'] == 'unsupported':
                require(set(group) == {'status'})
                continue
            require(set(group) == {'status', 'arms', 'contrasts'} and group['status'] == 'supported')
            require(set(group['arms']) == measurement_arms and set(group['contrasts']) == set(CONTRASTS))
            for row in group['arms'].values():
                require(set(row) == {'mae', 'ci95', 'bias', 'rmse'})
                require(row['mae'] >= 0 and row['rmse'] >= 0 and len(row['ci95']) == 2
                        and 0 <= row['ci95'][0] <= row['ci95'][1])
            for row in group['contrasts'].values():
                require(set(row) == {'delta', 'ci95'} and len(row['ci95']) == 2
                        and row['ci95'][0] <= row['ci95'][1])
    require(set(value['bootstrap']) == {'draws', 'minimum_valid', 'seed', 'sampling', 'uncertainty_scope'})
    require(value['bootstrap']['draws'] == 1000 and value['bootstrap']['minimum_valid'] == 900
            and value['bootstrap']['seed'] == 99221)
    require(set(value['replacement_provenance']) == set(ARMS)
            and value['primary_family'] == list(CONTRASTS))
    # Reject arrays or unexpected deeply nested patient records anywhere.
    def scalar_tree(x):
        require(type(x) in (dict, list, str, bool, int, float, type(None)))
        if type(x) is dict:
            require(all(type(k) is str for k in x))
            for v in x.values(): scalar_tree(v)
        elif type(x) is list:
            require(len(x) <= 25)
            for v in x: scalar_tree(v)
        elif type(x) is float:
            require(np.isfinite(x))
    scalar_tree(value)
