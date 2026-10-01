"""Pure private coordinator for prospective BRSET and ODIR retinal readouts."""
from __future__ import annotations

import copy

import numpy as np

import bran_retinal_adaptation_evaluation_v1 as brset
import bran_retinal_multisource_odir_evaluation_v1 as odir


ARMS = ('base', 'source_control', 'multisource_candidate')
_OLD_ARMS = ('base', 'masked_control', 'supervised_candidate')
_TO_OLD = dict(zip(ARMS, _OLD_ARMS, strict=True))
_TO_NEW = dict(zip(_OLD_ARMS, ARMS, strict=True))
_ERROR = 'invalid retinal multisource evaluation input'
_PARTITIONS = ('train', 'validation', 'test')


def _fail():
    raise ValueError(_ERROR) from None


def _require(value):
    if not value:
        _fail()


def _features(image_features, lengths):
    _require(type(image_features) is dict and set(image_features) == {'brset', 'odir'})
    result, width = {}, None
    for source in ('brset', 'odir'):
        values = image_features[source]
        _require(type(values) is dict and set(values) == set(ARMS))
        for arm in ARMS:
            value = values[arm]
            _require(type(value) is np.ndarray and value.ndim == 2 and value.shape[0] == lengths[source]
                     and np.issubdtype(value.dtype, np.floating) and np.all(np.isfinite(value)))
            if width is None:
                width = value.shape[1]
            _require(width is not None and width > 0 and value.shape[1] == width)
        result[source] = {arm: np.array(values[arm], copy=True, order='C') for arm in ARMS}
    return result


def _odir_patient_data(spec, features):
    _require(type(spec) is dict and set(spec) == {'image_groups', 'group_split', 'labels', 'observed', 'usable'})
    groups, split, labels, observed, usable = (spec[key] for key in ('image_groups', 'group_split', 'labels', 'observed', 'usable'))
    _require(type(groups) is np.ndarray and groups.dtype == np.int64 and groups.ndim == 1 and len(groups) > 0)
    _require(type(split) is np.ndarray and split.dtype == np.uint8 and split.ndim == 1 and len(split) > 0
             and np.all(np.isin(split, (0, 1, 2))))
    _require(np.all(groups >= 0) and int(groups.max()) < len(split))
    represented = np.zeros(len(split), dtype=bool)
    represented[groups] = True
    _require(np.all(represented))
    sizes = np.zeros(len(split), dtype=np.int64)
    np.add.at(sizes, groups, 1)
    _require(np.all((sizes >= 1) & (sizes <= 2)))
    _require(type(labels) is np.ndarray and labels.ndim == 2 and labels.shape == (len(split), len(odir.ENDPOINTS))
             and np.issubdtype(labels.dtype, np.floating))
    _require(type(observed) is np.ndarray and observed.dtype == bool and observed.shape == labels.shape)
    _require(type(usable) is np.ndarray and usable.dtype == bool and usable.shape == (len(odir.ENDPOINTS),))
    _require(np.all(np.isfinite(labels[observed])) and np.all(np.isin(labels[observed], (0.0, 1.0))))
    _require(np.all(np.isnan(labels) | np.isin(labels, (0.0, 1.0))))
    pooled = {}
    for arm in ARMS:
        value = features[arm]
        sums = np.zeros((len(split), value.shape[1]), dtype=value.dtype)
        np.add.at(sums, groups, value)
        pooled[arm] = sums / sizes[:, None]
    # ``usable`` belongs to the prospective optimizer policy.  It cannot erase
    # known evaluation labels in validation/test (or alter train support here).
    patient_labels = np.array(labels, copy=True, order='C')
    patient_observed = np.array(observed, copy=True, order='C')
    patient_labels[~patient_observed] = np.nan
    return {'labels': patient_labels, 'observed': patient_observed, 'split': np.array(split, copy=True),
            'features': pooled}


def _brset_patient_data(arrays, features):
    _require(type(arrays) is dict and {'groups', 'split', 'labels', 'observed', 'ages'} <= set(arrays))
    values = brset.group_patients(arrays['groups'], arrays['split'], arrays['labels'], arrays['observed'], arrays['ages'],
                                  {_TO_OLD[arm]: features[arm] for arm in ARMS})
    return values


def _rename_estimates(value):
    if value is None:
        return None
    return {
        'auroc': {_TO_NEW[arm]: copy.deepcopy(estimate) for arm, estimate in value['auroc'].items()},
        'candidate_delta': {_TO_NEW[arm]: copy.deepcopy(estimate) for arm, estimate in value['candidate_delta'].items()},
    }


def _coarse_pair(values):
    negative, positive = (int(value) for value in values)
    if negative < brset.MIN_CLASS or positive < brset.MIN_CLASS:
        return None
    return {'negative': (negative // 20) * 20, 'positive': (positive // 20) * 20}


def _rename_brset(summary, patient_data):
    counts = {name: brset.support(patient_data['labels'], patient_data['observed'], patient_data['split'], code)
              for code, name in enumerate(_PARTITIONS)}
    endpoints = {}
    for index, (name, endpoint) in enumerate(summary['endpoints'].items()):
        endpoints[name] = {
            'status': endpoint['status'],
            'counts_coarsened': {part: _coarse_pair(values[index]) for part, values in counts.items()},
            'estimates': _rename_estimates(endpoint['estimates']),
        }
    return {
        'schema': 'bran-retinal-multisource-brset-evaluation-v1',
        'analysis': summary['analysis'],
        'arms': list(ARMS),
        'primary_named_endpoints': list(summary['primary_named_endpoints']),
        'bootstrap': copy.deepcopy(summary['bootstrap']),
        'endpoints': endpoints,
        'primary_macro': _rename_estimates(summary['primary_macro']),
        'eligible_for_separate_unified_refit_review': summary['eligible_for_separate_unified_refit_review'],
        'unified_model_promoted': False,
        'clinical_validation_established': False,
        'patient_level_output_emitted': False,
    }


def _private_readout(readout, *, legacy_brset=False):
    heads = ({_TO_NEW[arm]: value for arm, value in readout['heads'].items()}
             if legacy_brset else readout['heads'])
    return {'heads': heads, 'predictions': readout['predictions']}


def evaluate(brset_arrays, odir_spec, image_features, *, draws=2000):
    """Fit private patient readouts and return a closed two-source aggregate.

    The ODIR result is exploratory and cannot change the BRSET-only refit gate.
    """
    try:
        _require(type(draws) is int and not isinstance(draws, bool) and draws >= 20)
        _require(type(brset_arrays) is dict and type(odir_spec) is dict)
        brset_length = len(brset_arrays['labels'])
        odir_length = len(odir_spec['image_groups'])
        features = _features(image_features, {'brset': brset_length, 'odir': odir_length})
        brset_data = _brset_patient_data(brset_arrays, features['brset'])
        odir_data = _odir_patient_data(odir_spec, features['odir'])
        image_readout = brset.fit_readouts(brset_data, include_age=False)
        secondary_readout = brset.fit_readouts(brset_data, include_age=True)
        image_summary = _rename_brset(brset.summarize(brset_data, image_readout, draws=draws, seed=74195), brset_data)
        secondary_summary = _rename_brset(brset.summarize(brset_data, secondary_readout, draws=draws, seed=74195), brset_data)
        odir_readout = odir.fit_readouts(odir_data)
        odir_summary = odir.summarize(odir_data, odir_readout, draws=draws, seed=74194)
    except Exception:
        _fail()
    result = {
        'schema': 'bran-retinal-multisource-evaluation-v1',
        'arms': list(ARMS),
        'brset': {'image_only_primary': image_summary, 'age_adjusted_secondary': secondary_summary},
        'odir': odir_summary,
        'eligible_for_separate_unified_refit_review': image_summary['eligible_for_separate_unified_refit_review'],
        'unified_model_promoted': False,
        'clinical_validation_established': False,
        'patient_level_output_emitted': False,
    }
    validate_summary(result)
    return result, {'brset': {'image_only_primary': _private_readout(image_readout, legacy_brset=True),
                              'age_adjusted_secondary': _private_readout(secondary_readout, legacy_brset=True)},
                    'odir': _private_readout(odir_readout)}


def _valid_estimates(value):
    if type(value) is not dict or set(value) != {'auroc', 'candidate_delta'}:
        return False
    if type(value['auroc']) is not dict or type(value['candidate_delta']) is not dict:
        return False
    if set(value['auroc']) != set(ARMS) or set(value['candidate_delta']) != set(ARMS[:2]):
        return False
    for group_name, group in value.items():
        bounds = (0.0, 1.0) if group_name == 'auroc' else (-1.0, 1.0)
        for estimate in group.values():
            if type(estimate) is not dict or set(estimate) != {'estimate', 'ci95'}:
                return False
            point, interval = estimate['estimate'], estimate['ci95']
            if (type(point) is not float or not np.isfinite(point) or not bounds[0] <= point <= bounds[1]
                    or type(interval) is not list or len(interval) != 2
                    or any(type(item) is not float or not np.isfinite(item) or not bounds[0] <= item <= bounds[1]
                           for item in interval) or interval[0] > interval[1]):
                return False
    candidate = value['auroc']['multisource_candidate']['estimate']
    return all(abs(value['candidate_delta'][arm]['estimate'] -
                   (candidate - value['auroc'][arm]['estimate'])) <= 1e-12 for arm in ARMS[:2])


def _valid_brset(summary):
    required = {'schema', 'analysis', 'arms', 'primary_named_endpoints', 'bootstrap', 'endpoints', 'primary_macro',
                'eligible_for_separate_unified_refit_review', 'unified_model_promoted',
                'clinical_validation_established', 'patient_level_output_emitted'}
    if (type(summary) is not dict or set(summary) != required
            or summary['schema'] != 'bran-retinal-multisource-brset-evaluation-v1'
            or summary['analysis'] not in ('image_only_primary', 'age_adjusted_secondary')
            or summary['arms'] != list(ARMS) or summary['primary_named_endpoints'] != list(brset.PRIMARY)):
        return False
    bootstrap = summary['bootstrap']
    if (type(bootstrap) is not dict or set(bootstrap) != {'unit', 'draws', 'seed', 'percentile',
                                                          'conditional_on_fixed_training', 'draws_released'}
            or bootstrap['unit'] != 'patient' or type(bootstrap['draws']) is not int or bootstrap['draws'] < 20
            or type(bootstrap['seed']) is not int or bootstrap['seed'] != 74195 or bootstrap['percentile'] != 95
            or bootstrap['conditional_on_fixed_training'] is not True or bootstrap['draws_released'] is not False):
        return False
    if type(summary['endpoints']) is not dict or set(summary['endpoints']) != set(brset.ENDPOINTS):
        return False
    for name in brset.ENDPOINTS:
        endpoint = summary['endpoints'][name]
        if (type(endpoint) is not dict or set(endpoint) != {'status', 'counts_coarsened', 'estimates'}
                or endpoint['status'] not in ('evaluated', 'insufficient_support')
                or type(endpoint['counts_coarsened']) is not dict
                or set(endpoint['counts_coarsened']) != set(_PARTITIONS)):
            return False
        for count in endpoint['counts_coarsened'].values():
            if count is not None and (type(count) is not dict or set(count) != {'negative', 'positive'}
                                      or any(type(number) is not int or number < 20 or number % 20 != 0
                                             for number in count.values())):
                return False
        expected = endpoint['counts_coarsened']['train'] is not None and endpoint['counts_coarsened']['test'] is not None
        if ((endpoint['status'] == 'evaluated') != expected
                or (endpoint['estimates'] is None) != (endpoint['status'] == 'insufficient_support')
                or (endpoint['estimates'] is not None and not _valid_estimates(endpoint['estimates']))):
            return False
    macro_expected = all(summary['endpoints'][name]['status'] == 'evaluated'
                         and summary['endpoints'][name]['counts_coarsened']['validation'] is not None
                         for name in brset.PRIMARY)
    macro = summary['primary_macro']
    if ((macro is not None) != macro_expected or (macro is not None and not _valid_estimates(macro))
            or type(summary['eligible_for_separate_unified_refit_review']) is not bool):
        return False
    expected_gate = False
    if macro is not None:
        for arm in ARMS:
            mean_point = sum(summary['endpoints'][name]['estimates']['auroc'][arm]['estimate']
                             for name in brset.PRIMARY) / len(brset.PRIMARY)
            if abs(macro['auroc'][arm]['estimate'] - mean_point) > 1e-12:
                return False
        expected_gate = (all(macro['candidate_delta'][arm]['ci95'][0] > 0.0 for arm in ARMS[:2])
                         and all(summary['endpoints'][name]['estimates']['auroc']['multisource_candidate']['estimate']
                                 - summary['endpoints'][name]['estimates']['auroc'][arm]['estimate'] >= -.02
                                 for name in brset.PRIMARY for arm in ARMS[:2]))
    if summary['analysis'] == 'age_adjusted_secondary':
        expected_gate = False
    return (summary['eligible_for_separate_unified_refit_review'] == expected_gate
            and summary['unified_model_promoted'] is False and summary['clinical_validation_established'] is False
            and summary['patient_level_output_emitted'] is False)


def validate_summary(summary):
    """Validate the entire closed public aggregate; no private state is accepted."""
    required = {'schema', 'arms', 'brset', 'odir', 'eligible_for_separate_unified_refit_review',
                'unified_model_promoted', 'clinical_validation_established', 'patient_level_output_emitted'}
    _require(type(summary) is dict and set(summary) == required
             and summary['schema'] == 'bran-retinal-multisource-evaluation-v1' and summary['arms'] == list(ARMS)
             and type(summary['brset']) is dict and set(summary['brset']) == {'image_only_primary', 'age_adjusted_secondary'})
    _require(_valid_brset(summary['brset']['image_only_primary']) and _valid_brset(summary['brset']['age_adjusted_secondary']))
    _require(summary['brset']['image_only_primary']['analysis'] == 'image_only_primary'
             and summary['brset']['age_adjusted_secondary']['analysis'] == 'age_adjusted_secondary')
    try:
        odir.validate_summary(summary['odir'])
    except Exception:
        _fail()
    brset_draws = summary['brset']['image_only_primary']['bootstrap']['draws']
    _require(type(summary['eligible_for_separate_unified_refit_review']) is bool
             and summary['eligible_for_separate_unified_refit_review']
             == summary['brset']['image_only_primary']['eligible_for_separate_unified_refit_review']
             and brset_draws == summary['brset']['age_adjusted_secondary']['bootstrap']['draws']
             and brset_draws == summary['odir']['bootstrap']['draws']
             and summary['odir']['bootstrap']['seed'] == 74194
             and summary['unified_model_promoted'] is False and summary['clinical_validation_established'] is False
             and summary['patient_level_output_emitted'] is False)
