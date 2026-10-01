"""Private array-only sampler for a prospective age-free CBC comparison.

Callers, not this module, authenticate source files and decide whether any
prospective fit is authorized.  No scalar age, retinal input, or pairing is
accepted or returned here.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

import bran_joint_lab_pretraining_v1 as frozen
from bran_joint_lab_task_contract_v1 import CBC_WIDTH, JOINT_WIDTH, REGISTRY_WIDTH, project_joint_labs_to_registry
import bran_nwicu_cohort_adult_v1 as nwicu


ERROR = 'age-free source sampler rejected'
_THREE = ('mimic', 'nhanes', 'eicu')
_ARMS = ('control', 'candidate')
_MODES = ('whole_cbc', 'partial_cbc')


def _fail():
    raise ValueError(ERROR) from None


def _require(value):
    if not value:
        _fail()


def _readonly(array, dtype):
    copied = np.array(array, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, repr=False)
class _Mode:
    sampler: object
    sources: tuple


@dataclass(frozen=True, repr=False)
class Prepared:
    control_whole: _Mode
    control_partial: _Mode
    candidate_partial: _Mode
    registry_fields: tuple
    medians: np.ndarray
    iqrs: np.ndarray


def _eligible(observed, groups, split, adult, mode):
    cbc = observed[:, :CBC_WIDTH].sum(axis=1)
    task = cbc >= 2 if mode == 'partial_cbc' else ((cbc >= 2) & observed[:, CBC_WIDTH:].any(axis=1))
    selected = task & (split == 0) & adult
    _require(np.unique(groups[selected]).size >= 20)
    return selected


def _mode(items, mode, registry_fields, medians, iqrs):
    pools, sources = [], []
    for name, values, observed, provenance, groups, split, adult in items:
        projected_values, projected_mask = project_joint_labs_to_registry(
            values, observed, provenance, registry_fields, medians, iqrs,
        )
        _eligible(observed, groups, split, adult, mode)
        prepared = frozen._prepare_mode(name, projected_values, projected_mask,
                                        observed, groups, split, adult, mode)
        _require(prepared is not None)
        pool, source = prepared
        pools.append(pool)
        sources.append(source)
    return _Mode(frozen.HierarchicalEpisodeSampler(tuple(pools)), tuple(sources))


def _nwicu_items(adapted):
    _require(type(adapted) is nwicu.AdaptedNWICUCohort
             and type(adapted.original_cache) is MappingProxyType)
    cache = adapted.original_cache
    _require(set(cache) == set(nwicu.KEYS))
    for value in cache.values():
        _require(type(value) is np.ndarray and not value.flags.writeable)
    nwicu.validate_cache(dict(cache))
    n = cache['values'].shape[0]
    masks = (adapted.cohort_adult, adapted.partial_cbc_eligible, adapted.native_age_eligible)
    _require(all(type(mask) is np.ndarray and mask.dtype == np.dtype(bool)
                     and mask.shape == (n,) and not mask.flags.writeable for mask in masks)
             and adapted.cohort_adult.all()
             and np.array_equal(adapted.partial_cbc_eligible, cache['split'] == 0)
             and not adapted.native_age_eligible.any())
    values = np.full((n, JOINT_WIDTH), np.nan, dtype=np.float64)
    observed = np.zeros((n, JOINT_WIDTH), dtype=bool)
    values[:, :CBC_WIDTH] = cache['values']
    observed[:, :CBC_WIDTH] = cache['observed']
    provenance = observed.astype(np.uint8)
    # This is the sole NWICU qualification: documented cohort adulthood plus
    # train split partial-CBC eligibility; native scalar-age eligibility stays false.
    adult = adapted.cohort_adult & adapted.partial_cbc_eligible
    return ('nwicu', values, observed, provenance, cache['person_group'], cache['split'], adult)


def _three_items(three_sources):
    _require(type(three_sources) is dict and set(three_sources) == set(_THREE))
    result = []
    for name in _THREE:
        _require(type(three_sources[name]) is dict)
        try:
            values, observed, provenance, groups, split = frozen._validate_source_arrays(name, three_sources[name])
            adult = three_sources[name]['adult_qualified']
            _require(type(adult) is np.ndarray and adult.dtype == np.dtype(bool)
                     and adult.shape == (values.shape[0],))
            if name == 'eicu':
                _require(not observed[:, CBC_WIDTH:].any())
            result.append((name, values, observed, provenance, groups, split, adult))
        except Exception:
            _fail()
    return tuple(result)


def prepare(three_sources, nwicu_adapted, registry_fields, medians, iqrs):
    """Prepare exact three-source control and four-source age-free candidate."""
    try:
        _require(type(registry_fields) is tuple and len(registry_fields) == REGISTRY_WIDTH
                 and type(medians) is np.ndarray and medians.shape == (REGISTRY_WIDTH,)
                 and medians.dtype.kind == 'f' and type(iqrs) is np.ndarray
                 and iqrs.shape == (REGISTRY_WIDTH,) and iqrs.dtype.kind == 'f')
        # _mode deliberately reuses the frozen projection/eligibility helpers.
        registry = tuple(registry_fields)
        local_medians = _readonly(medians, np.float64)
        local_iqrs = _readonly(iqrs, np.float64)
        three = _three_items(three_sources)
        nw = _nwicu_items(nwicu_adapted)
        # eICU's authenticated cache is CBC-only: exact whole-CBC support is
        # MIMIC/NHANES, while partial-CBC retains all three sources.
        control_whole = _mode(three[:2], 'whole_cbc', registry, local_medians, local_iqrs)
        control_partial = _mode(three, 'partial_cbc', registry, local_medians, local_iqrs)
        candidate_partial = _mode((*three, nw), 'partial_cbc', registry, local_medians, local_iqrs)
        return Prepared(control_whole, control_partial, candidate_partial,
                        registry, local_medians, local_iqrs)
    except Exception:
        _fail()


def sample(prepared, arm, mode, batch_size, rng):
    """Sample private 59-slot values/masks and 21-field observation truth."""
    try:
        _require(type(prepared) is Prepared and arm in _ARMS and mode in _MODES
                 and type(batch_size) is int and not isinstance(batch_size, bool) and batch_size > 0
                 and type(rng) is np.random.Generator)
        if mode == 'whole_cbc':
            chosen = prepared.control_whole
        elif arm == 'control':
            chosen = prepared.control_partial
        else:
            chosen = prepared.candidate_partial
        values, mask, observed = frozen._sample_batch(chosen, batch_size, rng)
        return _readonly(values, np.float64), _readonly(mask, bool), _readonly(observed, bool)
    except Exception:
        _fail()
