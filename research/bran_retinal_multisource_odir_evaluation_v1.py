"""Private ODIR patient readouts for an exploratory source-domain comparison.

This module performs no source discovery, image handling, encoder fitting, or
clinical validation.  Fitted parameters and patient-level predictions remain
private; ``summarize`` returns only a closed aggregate.
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from bran_retinal_adaptation_evaluation_v1 import _auc_plan, _weighted_auc


ARMS = ('base', 'source_control', 'multisource_candidate')
ENDPOINTS = ('N', 'D', 'G', 'C', 'A', 'H', 'M', 'O')
PRIMARY = ('N', 'D', 'O')
MIN_CLASS = 20
_PARTITIONS = ('train', 'validation', 'test')
_ERROR = 'invalid ODIR multisource evaluation input'


def _require(value):
    if not value:
        raise ValueError(_ERROR)


def _coarse(value):
    value = int(value)
    return 0 if value == 0 else ('<20' if value < 20 else (value // 20) * 20)


def _validate_patient_data(patient_data):
    _require(type(patient_data) is dict and set(patient_data) == {'labels', 'observed', 'split', 'features'})
    labels, observed, split, features = (patient_data[key] for key in ('labels', 'observed', 'split', 'features'))
    _require(type(labels) is np.ndarray and labels.ndim == 2 and labels.shape[1] == len(ENDPOINTS)
             and np.issubdtype(labels.dtype, np.floating) and len(labels) > 0)
    _require(type(observed) is np.ndarray and observed.dtype == bool and observed.shape == labels.shape)
    _require(np.all(np.isfinite(labels[observed])) and np.all(np.isin(labels[observed], (0.0, 1.0))))
    _require(np.all(~np.isnan(labels) | ~observed))
    _require(np.all(np.isnan(labels) | np.isin(labels, (0.0, 1.0))))
    _require(type(split) is np.ndarray and split.dtype == np.uint8 and split.shape == (len(labels),)
             and np.all(np.isin(split, (0, 1, 2))))
    _require(type(features) is dict and set(features) == set(ARMS))
    width = features['base'].shape[1] if (type(features['base']) is np.ndarray
                                          and features['base'].ndim == 2) else 0
    _require(width > 0 and all(type(value) is np.ndarray and value.shape == (len(labels), width)
                               and np.issubdtype(value.dtype, np.floating) and np.all(np.isfinite(value))
                               for value in features.values()))
    return labels, observed, split, features


def _support(labels, observed, split, partition):
    rows = split == partition
    return np.asarray([np.sum(rows & observed[:, endpoint] & (labels[:, endpoint] == target))
                       for endpoint in range(len(ENDPOINTS)) for target in (0.0, 1.0)], dtype=np.int64).reshape(-1, 2)


def fit_readouts(patient_data):
    """Fit fixed-C, balanced, train-only ODIR readouts; return private state."""
    labels, observed, split, features = _validate_patient_data(patient_data)
    train = split == 0
    usable = np.all(_support(labels, observed, split, 0) >= MIN_CLASS, axis=1)
    predictions = np.full((len(ARMS), len(labels), len(ENDPOINTS)), np.nan, dtype=np.float64)
    heads = {}
    for arm_index, arm in enumerate(ARMS):
        try:
            scaler = StandardScaler().fit(features[arm][train])
            standardized = scaler.transform(features[arm])
        except Exception:
            raise ValueError(_ERROR) from None
        arm_heads = {'mean': scaler.mean_.copy(), 'scale': scaler.scale_.copy(), 'endpoints': {}}
        for endpoint in np.flatnonzero(usable):
            rows = train & observed[:, endpoint]
            try:
                model = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=2000)
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    model.fit(standardized[rows], labels[rows, endpoint])
                predictions[arm_index, :, endpoint] = model.predict_proba(standardized)[:, 1]
            except Exception:
                raise ValueError(_ERROR) from None
            arm_heads['endpoints'][ENDPOINTS[endpoint]] = {
                'coefficient': model.coef_.copy(), 'intercept': model.intercept_.copy(),
            }
        heads[arm] = arm_heads
    return {'predictions': predictions, 'heads': heads}


def _interval(samples, draws):
    finite = samples[np.isfinite(samples)]
    _require(len(finite) >= .95 * draws)
    return [float(value) for value in np.quantile(finite, (.025, .975))]


def _estimates(points, samples, draws):
    return {
        'auroc': {
            arm: {'estimate': float(points[index]), 'ci95': _interval(samples[:, index], draws)}
            for index, arm in enumerate(ARMS)
        },
        'candidate_delta': {
            arm: {'estimate': float(points[2] - points[index]),
                  'ci95': _interval(samples[:, 2] - samples[:, index], draws)}
            for index, arm in enumerate(ARMS[:2])
        },
    }


def _count_summary(counts, endpoint):
    result = {}
    for partition, values in counts.items():
        negative, positive = (int(value) for value in values[endpoint])
        result[partition] = ({'negative': _coarse(negative), 'positive': _coarse(positive)}
                             if negative >= MIN_CLASS and positive >= MIN_CLASS else None)
    return result


def summarize(patient_data, readout, *, draws=2000, seed=74194):
    """Return a disclosure-safe, paired patient-bootstrap ODIR summary."""
    _require(type(draws) is int and not isinstance(draws, bool) and draws >= 20)
    _require(type(seed) is int and not isinstance(seed, bool) and seed >= 0)
    labels, observed, split, _ = _validate_patient_data(patient_data)
    _require(type(readout) is dict and set(readout) == {'predictions', 'heads'})
    prediction = readout['predictions']
    _require(type(prediction) is np.ndarray and prediction.shape == (len(ARMS), len(labels), len(ENDPOINTS))
             and np.issubdtype(prediction.dtype, np.floating))
    counts = {name: _support(labels, observed, split, code) for code, name in enumerate(_PARTITIONS)}
    reportable = np.all(counts['train'] >= MIN_CLASS, axis=1) & np.all(counts['test'] >= MIN_CLASS, axis=1)
    test_rows = np.flatnonzero(split == 2)
    _require(len(test_rows) > 0)
    y, known, score = labels[test_rows], observed[test_rows], prediction[:, test_rows]
    tested = np.flatnonzero(reportable)
    _require(np.all(np.isfinite(score[:, :, tested])) and np.all((score[:, :, tested] >= 0.0)
                                                                  & (score[:, :, tested] <= 1.0)))
    plans = {
        (arm, endpoint): (np.flatnonzero(known[:, endpoint]),
                          _auc_plan(y[known[:, endpoint], endpoint], score[arm, known[:, endpoint], endpoint]))
        for arm in range(len(ARMS)) for endpoint in tested
    }

    def scores(rows):
        weights = np.bincount(rows, minlength=len(test_rows))
        result = np.full((len(ARMS), len(ENDPOINTS)), np.nan, dtype=np.float64)
        for endpoint in tested:
            for arm in range(len(ARMS)):
                positions, plan = plans[arm, endpoint]
                result[arm, endpoint] = _weighted_auc(plan, weights[positions])
        return result

    point = scores(np.arange(len(test_rows)))
    rng = np.random.default_rng(seed)
    bootstrap = np.stack([scores(rng.integers(0, len(test_rows), len(test_rows))) for _ in range(draws)])
    endpoints = {
        name: {
            'status': 'evaluated' if reportable[index] else 'insufficient_support',
            'counts_coarsened': _count_summary(counts, index),
            'estimates': _estimates(point[:, index], bootstrap[:, :, index], draws) if reportable[index] else None,
        }
        for index, name in enumerate(ENDPOINTS)
    }
    primary_indices = [ENDPOINTS.index(endpoint) for endpoint in PRIMARY]
    macro_supported = bool(np.all(reportable[primary_indices])
                           and np.all(counts['validation'][primary_indices] >= MIN_CLASS))
    primary_macro = (_estimates(point[:, primary_indices].mean(axis=1),
                                bootstrap[:, :, primary_indices].mean(axis=2), draws)
                     if macro_supported else None)
    result = {
        'schema': 'bran-retinal-multisource-odir-evaluation-v1',
        'analysis': 'exploratory_source_domain_only',
        'arms': list(ARMS),
        'primary_named_endpoints': list(PRIMARY),
        'bootstrap': {'unit': 'patient', 'draws': draws, 'seed': seed, 'percentile': 95,
                      'conditional_on_fixed_training': True, 'draws_released': False},
        'endpoints': endpoints,
        'primary_macro': primary_macro,
        'systemic_disease_validation_established': False,
        'eligible_for_unified_model_advancement': False,
        'unified_model_promoted': False,
        'patient_level_output_emitted': False,
    }
    validate_summary(result)
    return result


def _valid_estimates(value):
    if type(value) is not dict or set(value) != {'auroc', 'candidate_delta'}:
        return False
    if type(value['auroc']) is not dict or type(value['candidate_delta']) is not dict:
        return False
    if set(value['auroc']) != set(ARMS) or set(value['candidate_delta']) != set(ARMS[:2]):
        return False
    for group_name, group in value.items():
        for estimate in group.values():
            if type(estimate) is not dict or set(estimate) != {'estimate', 'ci95'}:
                return False
            lower, upper = ((0.0, 1.0) if group_name == 'auroc' else (-1.0, 1.0))
            if (type(estimate['estimate']) is not float or not np.isfinite(estimate['estimate'])
                    or not lower <= estimate['estimate'] <= upper):
                return False
            ci = estimate['ci95']
            if (type(ci) is not list or len(ci) != 2
                    or any(type(item) is not float or not np.isfinite(item) or not lower <= item <= upper for item in ci)
                    or ci[0] > ci[1]):
                return False
    candidate = value['auroc'][ARMS[2]]['estimate']
    for arm in ARMS[:2]:
        if abs(value['candidate_delta'][arm]['estimate'] - (candidate - value['auroc'][arm]['estimate'])) > 1e-12:
            return False
    return True


def validate_summary(summary):
    """Reject non-public, malformed, or schema-expanded aggregate summaries."""
    required = {'schema', 'analysis', 'arms', 'primary_named_endpoints', 'bootstrap', 'endpoints', 'primary_macro',
                'systemic_disease_validation_established', 'eligible_for_unified_model_advancement',
                'unified_model_promoted', 'patient_level_output_emitted'}
    _require(type(summary) is dict and set(summary) == required
             and summary['schema'] == 'bran-retinal-multisource-odir-evaluation-v1'
             and summary['analysis'] == 'exploratory_source_domain_only'
             and summary['arms'] == list(ARMS) and summary['primary_named_endpoints'] == list(PRIMARY))
    bootstrap = summary['bootstrap']
    _require(type(bootstrap) is dict and set(bootstrap) == {'unit', 'draws', 'seed', 'percentile',
                                                            'conditional_on_fixed_training', 'draws_released'}
             and bootstrap['unit'] == 'patient' and type(bootstrap['draws']) is int and bootstrap['draws'] >= 20
             and type(bootstrap['seed']) is int and bootstrap['seed'] >= 0 and bootstrap['percentile'] == 95
             and bootstrap['conditional_on_fixed_training'] is True and bootstrap['draws_released'] is False)
    _require(type(summary['endpoints']) is dict and set(summary['endpoints']) == set(ENDPOINTS))
    for name in ENDPOINTS:
        endpoint = summary['endpoints'][name]
        _require(type(endpoint) is dict and set(endpoint) == {'status', 'counts_coarsened', 'estimates'}
                 and endpoint['status'] in ('evaluated', 'insufficient_support')
                 and type(endpoint['counts_coarsened']) is dict and set(endpoint['counts_coarsened']) == set(_PARTITIONS))
        for count in endpoint['counts_coarsened'].values():
            _require(count is None or (type(count) is dict and set(count) == {'negative', 'positive'}
                                       and all(type(number) is int and number >= MIN_CLASS and number % 20 == 0
                                               for number in count.values())))
        train_supported = endpoint['counts_coarsened']['train'] is not None
        test_supported = endpoint['counts_coarsened']['test'] is not None
        _require((endpoint['status'] == 'evaluated') == (train_supported and test_supported))
        _require((endpoint['estimates'] is None) == (endpoint['status'] == 'insufficient_support'))
        _require(endpoint['estimates'] is None or _valid_estimates(endpoint['estimates']))
    macro_expected = all(summary['endpoints'][name]['status'] == 'evaluated'
                         and summary['endpoints'][name]['counts_coarsened']['validation'] is not None
                         for name in PRIMARY)
    _require((summary['primary_macro'] is not None) == macro_expected)
    _require(summary['primary_macro'] is None or _valid_estimates(summary['primary_macro']))
    _require(summary['systemic_disease_validation_established'] is False
             and summary['eligible_for_unified_model_advancement'] is False
             and summary['unified_model_promoted'] is False
             and summary['patient_level_output_emitted'] is False)
