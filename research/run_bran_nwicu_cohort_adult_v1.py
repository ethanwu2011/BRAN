"""Local cache-only NWICU adult-cohort qualification; never a model fit.

All real source processing must occur through the FD-quiet locked CLI or a
caller providing that same boundary. Only closed aggregate receipts may leave.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re

import numpy as np

import run_bran_observed_cbc_pool_v1 as pool
import run_bran_retinal_group_readiness_v1 as safe
import bran_nwicu_cohort_adult_v1 as kernel

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'BRAN_NWICU_COHORT_ADULT_PROTOCOL_V1.json'
OUT = ROOT / 'BRAN_NWICU_COHORT_ADULT_V1'
AUDIT = ROOT / 'BRAN_NWICU_COHORT_ADULT_AUDIT_V1'
LOCK = safe.LOCK
POOL_PIN = '397292c03fa89e496ec3295b19ca9200b6d93e3d80717251d07f0003748880f6'
POOL_AGGREGATE_PIN = '0cb1d57dce7fef3c442a7d5d0cd4c826a56c667e0ad605e8ff94a5d716793f4c'
RELEASE_ROOT = Path(pool.INPUTS['nwicu_patients']).parents[2]
RELEASE_FILES = {
    'nwicu_patients': ('data/nw_hosp/patients.csv.gz', 'c9d922c04ef969896bab52f0509bd115801f5acf8113240803f870e3679caad9'),
    'nwicu_admissions': ('data/nw_hosp/admissions.csv.gz', '79377f74e34f283ea6c2f95848932abfb6b3aa0ea71dc5f5ecfce03265535957'),
    'nwicu_labs': ('data/nw_hosp/labevents.csv.gz', '858aaefa18da3f6c8b73e3b0de27eb7718cb2377dd025426f0d67be5be507e68'),
}
LICENSE_PIN = '26794be6ea7916cd4b4017b45b9c34c3c31f645a03a9cd6c591c529b92435042'
CODE = tuple(pool.CODE) + (
    'run_bran_nwicu_cohort_adult_v1.py', 'test_run_bran_nwicu_cohort_adult_v1.py',
    'bran_nwicu_cohort_adult_v1.py', 'test_bran_nwicu_cohort_adult_v1.py',
    'run_bran_retinal_group_readiness_v1.py', 'bran_retinal_group_readiness_v1.py',
    'BRAN_NWICU_COHORT_ADULT_DESIGN_V1.md')
POLICY = {
    'source': 'nwicu', 'version': '0.1.0',
    'documentation_url': 'https://physionet.org/content/nwicu-northwestern-icu/0.1.0/',
    'adulthood_basis': 'provider_excluded_under18_at_earliest_admission',
    'task': 'age_free_partial_cbc', 'source_split': 'existing_train_only',
    'minimum_observed_cbc': 2, 'minimum_release_people': 20,
    'count_rounding': 20, 'scalar_age_inference_permitted': False,
    'whole_cbc_from_chemistry_qualified': False, 'original_age_fields_unchanged': True,
    'training_permitted': False, 'retinal_pairing_claimed': False,
    'official_download_independently_verified': False,
}
FLAGS = {'patient_level_output_emitted': False, 'training_started': False,
         'model_improvement_established': False, 'clinical_use': False,
         'source_cache_unchanged': True, 'old_age_policy_unchanged': True,
         'exact_age_resolved': False, 'new_access_permission_created': False}
ERROR = 'NWICU cohort qualification rejected'


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def read_json(path):
    safe.regular(path)
    return json.loads(path.read_text())


def checksum_entries(path):
    safe.regular(path)
    require(path.stat().st_size <= 100000)
    entries = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r'([a-f0-9]{64})\s+\*?(?:\./)?([^\s]+)', line)
        require(match is not None)
        digest, name = match.groups()
        require(name not in entries and not Path(name).is_absolute() and '..' not in Path(name).parts)
        entries[name] = digest
    return entries


def source_receipt():
    """Hash only original tables: no event/identifier rows are parsed."""
    require(safe.sha(pool.PROTOCOL) == POOL_PIN)
    old = read_json(pool.PROTOCOL); pool.validate_protocol(old)
    parent = pool.PUBLIC / 'nwicu'
    safe.inventory(parent, ('aggregate.json', 'manifest.json'))
    require(safe.sha(parent / 'aggregate.json') == POOL_AGGREGATE_PIN)
    aggregate = read_json(parent / 'aggregate.json'); pool.validate_aggregate(aggregate)
    require(aggregate['source'] == 'nwicu')
    manifest = read_json(parent / 'manifest.json')
    salt = pool.PRIVATE / 'split_salt.bin'
    safe.regular(salt, private=True)
    require(safe.equal(manifest, {'protocol_sha256': POOL_PIN,
        'aggregate_sha256': POOL_AGGREGATE_PIN, 'split_salt_sha256': safe.sha(salt),
        'patient_level_output_emitted': False, 'training_started': False}))
    cache = pool.PRIVATE / 'nwicu.npz'
    safe.regular(cache, private=True)
    require(os.lstat(cache).st_nlink == 1 and safe.sha(cache) == aggregate['private_cache_sha256'])
    checksum = RELEASE_ROOT / 'SHA256SUMS.txt'
    license_path = RELEASE_ROOT / 'LICENSE.txt'
    safe.regular(license_path)
    require(safe.sha(license_path) == LICENSE_PIN)
    listed = checksum_entries(checksum)
    require(listed.get('LICENSE.txt') == LICENSE_PIN)
    for key, (relative, expected) in RELEASE_FILES.items():
        path = RELEASE_ROOT / relative
        require(Path(pool.INPUTS[key]) == path and listed.get(relative) == expected
                and old['source_files'][key]['sha256'] == expected)
        safe.regular(path)
        require(safe.sha(path) == expected)
    return {'pool_protocol_sha256': POOL_PIN, 'pool_aggregate_sha256': POOL_AGGREGATE_PIN,
        'pool_manifest_sha256': safe.sha(parent / 'manifest.json'),
        'private_cache_sha256': aggregate['private_cache_sha256'],
        'release_checksum_sha256': safe.sha(checksum), 'license_sha256': LICENSE_PIN,
        'local_release_checksums_matched': True,
        'official_download_independently_verified': False}


def expected_protocol():
    return {'schema': 'bran-nwicu-cohort-adult-protocol-v1',
        'status': 'frozen_before_execution', 'source': source_receipt(), 'policy': POLICY,
        'code_sha256': {name: safe.sha(ROOT / name) for name in sorted(set(CODE))},
        'runtime': safe.runtime(), 'patient_level_output_permitted': False}


def prepare():
    safe.absent(PROTOCOL, OUT, AUDIT)
    value = expected_protocol()
    with os.fdopen(os.open(PROTOCOL, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    return safe.sha(PROTOCOL)


def load_protocol(pin):
    require(pool.valid_sha(pin))
    safe.regular(PROTOCOL, private=True)
    require(safe.sha(PROTOCOL) == pin and os.lstat(PROTOCOL).st_nlink == 1)
    value = read_json(PROTOCOL)
    require(safe.equal(value, expected_protocol()) and safe.sha(PROTOCOL) == pin)
    return value


def load_private(receipt):
    """Returns PRIVATE arrays only, never print or pass to hosted models."""
    require(safe.equal(source_receipt(), receipt))
    cache = pool.PRIVATE / 'nwicu.npz'
    with np.load(cache, allow_pickle=False) as handle:
        require(set(handle.files) == set(kernel.KEYS))
        arrays = {key: handle[key].copy() for key in kernel.KEYS}
    aggregate = read_json(pool.PUBLIC / 'nwicu/aggregate.json')
    kernel.validate_cache(arrays, aggregate['counts_lower_bounds_20'])
    value = kernel.adapt(arrays)
    require(safe.sha(cache) == receipt['private_cache_sha256'])
    return value


def result(value, p, pin):
    item = {'schema': 'bran-nwicu-cohort-adult-result-v1', 'status': 'completed',
        'protocol_sha256': pin, 'source_receipt': p['source'], 'policy': POLICY,
        'qualification': kernel.summarize(value), **FLAGS}
    validate_result(item, p, pin)
    return item


def validate_result(item, p, pin):
    require(type(item) is dict and set(item) == {'schema', 'status', 'protocol_sha256',
        'source_receipt', 'policy', 'qualification'} | set(FLAGS))
    require(item['schema'] == 'bran-nwicu-cohort-adult-result-v1' and item['status'] == 'completed'
        and item['protocol_sha256'] == pin and safe.equal(item['source_receipt'], p['source'])
        and safe.equal(item['policy'], POLICY) and all(item[k] is v for k, v in FLAGS.items()))
    q = item['qualification']
    require(type(q) is dict and set(q) == {'status', 'eligible_training_snapshots_lower_bound',
                                         'eligible_training_people_lower_bound'})
    values = (q['eligible_training_snapshots_lower_bound'], q['eligible_training_people_lower_bound'])
    if q['status'] == 'qualified_age_free_pool':
        require(all(type(v) is int and v >= 20 and v % 20 == 0 for v in values) and values[0] >= values[1])
    else:
        require(q['status'] == 'suppressed_insufficient_support' and values == (None, None))


def run(pin, state):
    p = load_protocol(pin); safe.absent(OUT, AUDIT)
    OUT.mkdir(); state['owned'] = OUT; state['phase'] = 'qualification'
    item = result(load_private(p['source']), p, pin)
    require(safe.equal(load_protocol(pin), p))
    safe.write_json(OUT / 'aggregate.json', item)
    safe.write_json(OUT / 'manifest.json', {'protocol_sha256': pin,
        'aggregate_sha256': safe.sha(OUT / 'aggregate.json'), 'patient_level_output_emitted': False})
    state['owned'] = None


def payload(pin):
    p = load_protocol(pin)
    safe.inventory(OUT, ('aggregate.json', 'manifest.json'))
    require(not os.path.lexists(AUDIT / 'failure.json'))
    require(safe.equal(read_json(OUT / 'manifest.json'), {'protocol_sha256': pin,
        'aggregate_sha256': safe.sha(OUT / 'aggregate.json'), 'patient_level_output_emitted': False}))
    item = read_json(OUT / 'aggregate.json'); validate_result(item, p, pin)
    return p, item


def audit(pin, state):
    p, expected = payload(pin); safe.absent(AUDIT)
    AUDIT.mkdir(); state['owned'] = AUDIT; state['phase'] = 'audit'
    require(safe.equal(result(load_private(p['source']), p, pin), expected)
            and safe.equal(load_protocol(pin), p))
    safe.write_json(AUDIT / 'audit.json', {'schema': 'bran-nwicu-cohort-adult-audit-v1',
        'status': 'authenticated', 'protocol_sha256': pin,
        'aggregate_sha256': safe.sha(OUT / 'aggregate.json'),
        'manifest_sha256': safe.sha(OUT / 'manifest.json'),
        'cache_semantics_and_counts_replayed': True, **FLAGS})
    safe.write_json(AUDIT / 'manifest.json', {'protocol_sha256': pin,
        'audit_sha256': safe.sha(AUDIT / 'audit.json'), 'patient_level_output_emitted': False})
    state['owned'] = None


def verify_audit(pin, audit_pin):
    payload(pin); safe.inventory(AUDIT, ('audit.json', 'manifest.json'))
    require(pool.valid_sha(audit_pin) and safe.sha(AUDIT / 'audit.json') == audit_pin)
    require(safe.equal(read_json(AUDIT / 'audit.json'), {
        'schema': 'bran-nwicu-cohort-adult-audit-v1', 'status': 'authenticated',
        'protocol_sha256': pin, 'aggregate_sha256': safe.sha(OUT / 'aggregate.json'),
        'manifest_sha256': safe.sha(OUT / 'manifest.json'),
        'cache_semantics_and_counts_replayed': True, **FLAGS}))
    require(safe.equal(read_json(AUDIT / 'manifest.json'), {'protocol_sha256': pin,
        'audit_sha256': audit_pin, 'patient_level_output_emitted': False}))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'audit', 'verify'))
    parser.add_argument('--protocol-sha256'); parser.add_argument('--audit-sha256')
    args = parser.parse_args(argv)
    state = {'owned': None, 'phase': 'authentication'}
    answer = {'status': 'failed', 'patient_level_output_emitted': False, 'training_started': False}
    with safe.quiet():
        try:
            with LOCK.open('a') as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                pin = args.protocol_sha256
                if args.action == 'prepare': pin = prepare()
                elif args.action == 'run': run(pin, state)
                elif args.action == 'audit': audit(pin, state)
                elif args.audit_sha256: verify_audit(pin, args.audit_sha256)
                else: payload(pin)
                answer = {'status': 'completed', 'action': args.action, 'protocol_sha256': pin,
                    'patient_level_output_emitted': False, 'training_started': False}
                if args.action != 'prepare': answer['aggregate_sha256'] = safe.sha(OUT / 'aggregate.json')
                if args.action == 'audit' or args.audit_sha256:
                    answer['audit_sha256'] = safe.sha(AUDIT / 'audit.json')
        except Exception:
            answer['phase'] = state['phase']
            if state['owned'] is not None:
                try: safe.write_json(state['owned'] / 'failure.json', answer)
                except Exception: pass
    print(json.dumps(answer, sort_keys=True))
    return int(answer['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
