"""Authenticated private clinical sources for a new, age-free BRAN comparison.

No I/O on import. Caller must hold the shared compute lock. Public entry points
silence file descriptors themselves and return private objects, never row logs.
This source loader does not freeze or launch a training experiment.
"""
from dataclasses import dataclass
import importlib
import os
from pathlib import Path

import run_bran_joint_lab_comparison_v1 as three
import run_bran_nwicu_cohort_adult_v1 as nwicu

safe = nwicu.safe
ERROR = 'age-free clinical source authentication rejected'
NWICU_PROTOCOL = 'f2f3930300500a5141c3cf2bd2f274c50134aebf3bb9608bcf63d5db65f75d59'
NWICU_AGGREGATE = '7f5852eb0c08b2bb69fcac712cfc3b90a7436cc9e9902bdae49e12e125a76b87'
NWICU_AUDIT = '44a508ca93ce856729ce738ead76a20bf66f55bced77551a14c91b07c66354f4'
POLICY = {
    'control_partial_sources': ['mimic', 'nhanes', 'eicu'],
    'candidate_partial_sources': ['mimic', 'nhanes', 'eicu', 'nwicu'],
    'whole_sources_both_arms': ['mimic', 'nhanes'],
    'training_partition': 0, 'external_scalar_age_used': False,
    'nwicu_eligibility': 'audited release-cohort adulthood; partial CBC only',
    'source_permission_created': False, 'cross_source_people_deduplicated': False,
    'source_patients_paired_with_retina': False, 'imputed_truth_permitted': False,
    'patient_level_output_permitted': False, 'training_started': False,
}


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def _inventory_three(source):
    module, _, _, audit_path, _ = three.SOURCE_SPECS[source]
    r = importlib.import_module(module)
    expected = ('aggregate.json', 'manifest.json', 'progress.json') if source == 'mimic' else (
        'aggregate.json', 'manifest.json')
    safe.inventory(r.PUBLIC, expected)
    if audit_path:
        safe.inventory((three.ROOT / audit_path).parent, ('audit.json',))
    cache = r.PRIVATE / 'observations.npz'
    safe.regular(cache, private=True)
    require(os.lstat(cache).st_nlink == 1)


def describe(*, verify_raw=True):
    """Authenticate original observation caches and the separate NWICU policy."""
    with safe.quiet():
        try:
            require(type(verify_raw) is bool)
            sources = {}
            for name in ('mimic', 'nhanes', 'eicu'):
                _inventory_three(name)
                sources[name] = three.source_receipt(name, verify_raw=verify_raw)
            nwicu.verify_audit(NWICU_PROTOCOL, NWICU_AUDIT)
            require(safe.sha(nwicu.OUT / 'aggregate.json') == NWICU_AGGREGATE)
            p, a = nwicu.payload(NWICU_PROTOCOL)
            require(a['qualification']['status'] == 'qualified_age_free_pool')
            return {'schema': 'bran-agefree-clinical-sources-v1',
                'sources': sources, 'joint_registry_indices': three.joint_units(),
                'nwicu': {'protocol_sha256': NWICU_PROTOCOL, 'aggregate_sha256': NWICU_AGGREGATE,
                    'audit_sha256': NWICU_AUDIT, 'source': p['source']},
                'policy': {key: value.copy() if isinstance(value, list) else value
                           for key, value in POLICY.items()}}
        except Exception:
            raise ValueError(ERROR) from None


@dataclass(frozen=True, repr=False)
class PrivateClinicalSources:
    three_sources: dict
    nwicu_adapted: object


def load(expected):
    """Authenticate before/after private cache decoding; never return scalar ages."""
    with safe.quiet():
        try:
            require(type(expected) is dict and safe.equal(expected, describe(verify_raw=False)))
            original = three.load_external({'sources': expected['sources']})
            require(set(original) == {'mimic', 'nhanes', 'eicu'})
            adapted = nwicu.load_private(expected['nwicu']['source'])
            require(safe.equal(expected, describe(verify_raw=False)))
            return PrivateClinicalSources(original, adapted)
        except Exception:
            raise ValueError(ERROR) from None
