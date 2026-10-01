"""Private, array-only NWICU adult-cohort qualification.

The caller must separately authenticate an actual local NWICU release and its
receipt.  This adapter neither opens sources nor establishes scalar ages,
training permission, model performance, or a whole-CBC route.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from bran_clinical_semantics_v1 import CBC_FIELDS


ERROR = 'nwicu adult cohort qualification rejected'
KEYS = ('values', 'observed', 'provenance', 'age_triplet', 'age_kind',
        'adult_qualified', 'person_group', 'split', 'person_weight')
_AGE_UNRESOLVED = 'source_age_unresolved'
_AGE_KINDS = ('missing_or_invalid', _AGE_UNRESOLVED, 'reported_year', 'year_derived',
              'topcoded', 'rounded', 'rounded_topcoded')


def _fail():
    raise ValueError(ERROR) from None


def _require(value):
    if not value:
        _fail()


def _coarse(value):
    return None if value < 20 else value // 20 * 20


def _expected_summary(observed, groups, split):
    n = int(observed.shape[0])
    people = int(groups.max()) + 1
    return {
        'snapshots': _coarse(n),
        'source_local_people': _coarse(people),
        'adult_qualified_snapshots': None,
        'field_observed_snapshots': {
            field: _coarse(int(observed[:, index].sum()))
            for index, field in enumerate(CBC_FIELDS)
        },
        'split_snapshots': {
            name: _coarse(int((split == index).sum()))
            for index, name in enumerate(('train', 'validation', 'test'))
        },
    }


def validate_cache(arrays, expected_summary=None):
    """Validate the unchanged private v1 NWICU observation-cache interface.

    Returns only ``True``; patient-level arrays remain caller-private.
    """
    try:
        _require(type(arrays) is dict and set(arrays) == set(KEYS))
        values = arrays['values']
        observed = arrays['observed']
        provenance = arrays['provenance']
        ages = arrays['age_triplet']
        age_kind = arrays['age_kind']
        adult = arrays['adult_qualified']
        groups = arrays['person_group']
        split = arrays['split']
        weights = arrays['person_weight']
        _require(type(values) is np.ndarray and values.dtype == np.dtype(np.float64)
                 and values.ndim == 2 and values.shape[1] == len(CBC_FIELDS)
                 and values.shape[0] > 0
                 and type(observed) is np.ndarray and observed.dtype == np.dtype(bool)
                 and observed.shape == values.shape
                 and type(provenance) is np.ndarray and provenance.dtype == np.dtype(np.uint8)
                 and provenance.shape == values.shape
                 and np.array_equal(provenance, observed.astype(np.uint8))
                 and np.all(observed.sum(axis=1) >= 2)
                 and np.isfinite(values[observed]).all() and np.all(values[observed] > 0)
                 and np.isnan(values[~observed]).all())
        n = values.shape[0]
        _require(type(ages) is np.ndarray and ages.dtype == np.dtype(np.float64)
                 and ages.shape == (n, 3) and np.isnan(ages).all()
                 and type(age_kind) is np.ndarray and age_kind.dtype == np.dtype(np.uint8)
                 and age_kind.shape == (n,) and np.all(age_kind == _AGE_KINDS.index(_AGE_UNRESOLVED))
                 and type(adult) is np.ndarray and adult.dtype == np.dtype(bool)
                 and adult.shape == (n,) and not adult.any()
                 and type(groups) is np.ndarray and groups.dtype == np.dtype(np.int64)
                 and groups.shape == (n,) and np.all(groups >= 0)
                 and type(split) is np.ndarray and split.dtype == np.dtype(np.uint8)
                 and split.shape == (n,) and np.all(split <= 2)
                 and type(weights) is np.ndarray and weights.dtype == np.dtype(np.float64)
                 and weights.shape == (n,) and np.isfinite(weights).all() and np.all(weights > 0))
        _require(int(groups.max()) < n)
        unique_groups, first, counts = np.unique(groups, return_index=True, return_counts=True)
        people = len(unique_groups)
        _require(np.array_equal(unique_groups, np.arange(people, dtype=np.int64)))
        expected_weights = 1.0 / counts[groups]
        _require(np.array_equal(weights, expected_weights))
        _require(np.array_equal(split, split[first][groups]))
        summary = _expected_summary(observed, groups, split)
        if expected_summary is not None:
            _require(type(expected_summary) is dict and expected_summary == summary)
        return True
    except Exception:
        _fail()


def _readonly(value, dtype):
    copied = np.array(value, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, repr=False)
class AdaptedNWICUCohort:
    original_cache: object
    cohort_adult: np.ndarray
    partial_cbc_eligible: np.ndarray
    native_age_eligible: np.ndarray


def adapt(arrays):
    """Copy a validated cache and expose only the age-free partial-CBC route."""
    try:
        validate_cache(arrays)
        copied = {
            'values': _readonly(arrays['values'], np.float64),
            'observed': _readonly(arrays['observed'], bool),
            'provenance': _readonly(arrays['provenance'], np.uint8),
            'age_triplet': _readonly(arrays['age_triplet'], np.float64),
            'age_kind': _readonly(arrays['age_kind'], np.uint8),
            'adult_qualified': _readonly(arrays['adult_qualified'], bool),
            'person_group': _readonly(arrays['person_group'], np.int64),
            'split': _readonly(arrays['split'], np.uint8),
            'person_weight': _readonly(arrays['person_weight'], np.float64),
        }
        n = copied['values'].shape[0]
        cohort_adult = _readonly(np.ones(n, dtype=bool), bool)
        partial = _readonly(copied['split'] == 0, bool)
        no_age = _readonly(np.zeros(n, dtype=bool), bool)
        return AdaptedNWICUCohort(MappingProxyType(copied), cohort_adult, partial, no_age)
    except Exception:
        _fail()


def summarize(adapted):
    """Return coarsened age-free partial-CBC support only, never rows or ages."""
    try:
        _require(type(adapted) is AdaptedNWICUCohort)
        cache = adapted.original_cache
        _require(type(cache) is MappingProxyType)
        validate_cache(dict(cache))
        n = cache['values'].shape[0]
        _require(all(type(mask) is np.ndarray and mask.dtype == np.dtype(bool)
                     and mask.shape == (n,) and not mask.flags.writeable
                     for mask in (adapted.cohort_adult, adapted.partial_cbc_eligible,
                                  adapted.native_age_eligible))
                 and adapted.cohort_adult.all()
                 and np.array_equal(adapted.partial_cbc_eligible, cache['split'] == 0)
                 and not adapted.native_age_eligible.any())
        training = adapted.partial_cbc_eligible
        people = int(np.unique(cache['person_group'][training]).size)
        if people < 20:
            return {'status': 'suppressed_insufficient_support',
                    'eligible_training_snapshots_lower_bound': None,
                    'eligible_training_people_lower_bound': None}
        return {'status': 'qualified_age_free_pool',
                'eligible_training_snapshots_lower_bound': _coarse(int(training.sum())),
                'eligible_training_people_lower_bound': _coarse(people)}
    except Exception:
        _fail()
