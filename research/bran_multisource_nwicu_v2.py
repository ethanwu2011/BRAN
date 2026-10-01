"""Local-only NWICU partial-CBC bridge for the prospective multisource V2 study.

This module neither opens a cache nor qualifies source assays.  A caller must
first authenticate the frozen NWICU adult-cohort receipt and provide its private
in-memory adapted cohort.  It intentionally makes no scalar-age, chemistry, or
whole-CBC claim.
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np

import bran_nwicu_cohort_adult_v1 as legacy
from bran_clinical_chemistry_semantics_v1 import CHEMISTRY_FIELDS
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_multisource_clinical_v2 import ClinicalPoolV2, eligibility_hash as _eligibility_hash


ERROR = "nwicu multisource binding rejected"
FIELDS = CBC_FIELDS + CHEMISTRY_FIELDS


def _fail() -> None:
    raise ValueError(ERROR) from None


def _readonly(value: np.ndarray, dtype) -> np.ndarray:
    copy = np.array(value, dtype=dtype, copy=True)
    copy.setflags(write=False)
    return copy


def validate_pool(pool: object) -> None:
    """Validate the new partial-CBC-only container without exposing arrays."""
    try:
        if not isinstance(pool, ClinicalPoolV2) or pool.source != "nwicu":
            _fail()
        arrays = (pool.values, pool.observed, pool.person_group, pool.age_value,
                  pool.age_lower, pool.age_upper, pool.age_kind, pool.original_indices)
        if any(not isinstance(value, np.ndarray) or value.flags.writeable for value in arrays):
            _fail()
        n = pool.values.shape[0]
        if (n <= 0 or pool.values.dtype != np.float64 or pool.values.shape != (n, len(FIELDS))
                or pool.observed.dtype != np.bool_ or pool.observed.shape != pool.values.shape
                or pool.person_group.dtype != np.int64 or pool.person_group.shape != (n,)
                or pool.original_indices.dtype != np.int64 or pool.original_indices.shape != (n,)
                or len(np.unique(pool.original_indices)) != n or np.any(pool.original_indices < 0)):
            _fail()
        if (not np.isfinite(pool.values[:, :len(CBC_FIELDS)][pool.observed[:, :len(CBC_FIELDS)]]).all()
                or np.any(pool.values[:, :len(CBC_FIELDS)][pool.observed[:, :len(CBC_FIELDS)]] <= 0)
                or not np.isnan(pool.values[~pool.observed]).all()
                or np.any(pool.observed[:, len(CBC_FIELDS):])
                or not np.isnan(pool.values[:, len(CBC_FIELDS):]).all()
                or np.any(pool.observed.sum(axis=1) < 2)):
            _fail()
        for age in (pool.age_value, pool.age_lower, pool.age_upper):
            if age.dtype != np.float64 or age.shape != (n,) or not np.isnan(age).all():
                _fail()
        if pool.age_kind.dtype != np.int64 or pool.age_kind.shape != (n,) or not np.all(pool.age_kind == 3):
            _fail()
        if np.any(pool.person_group < 0):
            _fail()
    except Exception:
        _fail()


def convert(adapted: object) -> ClinicalPoolV2:
    """Create a split-0, 21-slot pool from the authenticated legacy CBC cache.

    ``legacy.adapt`` is deliberately preserved: the old qualification remains
    frozen with ``training_permitted=False``.  This conversion is a separately
    declared prospective V2-admission decision and does not mutate its arrays.
    """
    try:
        if type(adapted) is not legacy.AdaptedNWICUCohort:
            _fail()
        cache = adapted.original_cache
        if not isinstance(cache, MappingProxyType):
            _fail()
        legacy.validate_cache(dict(cache))
        n = cache["values"].shape[0]
        if (any(not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (n,)
                for mask in (adapted.cohort_adult, adapted.partial_cbc_eligible, adapted.native_age_eligible))
                or not adapted.cohort_adult.all() or adapted.native_age_eligible.any()
                or not np.array_equal(adapted.partial_cbc_eligible, cache["split"] == 0)):
            _fail()
        take = adapted.partial_cbc_eligible
        if not take.any():
            _fail()
        count = int(take.sum())
        values = np.full((count, len(FIELDS)), np.nan, dtype=np.float64)
        observed = np.zeros((count, len(FIELDS)), dtype=bool)
        values[:, :len(CBC_FIELDS)] = cache["values"][take]
        observed[:, :len(CBC_FIELDS)] = cache["observed"][take]
        unknown = np.full(count, np.nan, dtype=np.float64)
        pool = ClinicalPoolV2("nwicu", _readonly(values, np.float64), _readonly(observed, bool),
                              _readonly(cache["person_group"][take], np.int64), _readonly(unknown, np.float64),
                              _readonly(unknown, np.float64), _readonly(unknown, np.float64),
                              _readonly(np.full(count, 3, dtype=np.int64), np.int64),
                              _readonly(np.flatnonzero(take), np.int64))
        validate_pool(pool)
        return pool
    except Exception:
        _fail()


def safe_summary(pool: ClinicalPoolV2) -> dict[str, object]:
    """Closed aggregate only; never returns rows, ages, or observed values."""
    validate_pool(pool)
    groups = int(np.unique(pool.person_group).size)
    rows = int(pool.values.shape[0])
    support = groups >= 20
    return {
        "status": "supported_training_pool" if support else "insufficient_support",
        "source": "nwicu", "source_local_people_lower_bound_20": groups // 20 * 20 if support else None,
        "training_examples_lower_bound_20": rows // 20 * 20 if support else None,
        "canonical_fields": list(FIELDS), "canonical_width": len(FIELDS),
        "partial_cbc_only": True, "chemistry_qualified": False,
        "age_representation": "unknown", "prospective_v2_training_admission": support,
        "legacy_qualification_training_permitted": False,
        "patient_level_output_emitted": False,
    }


def eligibility_hash(pool: ClinicalPoolV2) -> str:
    validate_pool(pool)
    return _eligibility_hash(pool)
