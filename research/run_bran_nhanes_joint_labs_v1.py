"""Frozen local-only NHANES CBC + chemistry materialization; no model fit."""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from pandas.io.sas.sas_xport import XportReader

from bran_clinical_dictionary_binding_v1 import _quiet
from bran_clinical_semantics_v1 import NHANES_CBC_CODES
from bran_nhanes_chemistry_binding_v1 import NHANES_BIOPRO_RULES, inspect_nhanes_biopro_header, codebook_binding_sha256
from bran_nhanes_chemistry_observation_v2 import ASSAY_POLICY
from bran_nhanes_source_reader_v1 import iter_xport_projection
from bran_nhanes_joint_labs_v1 import joint_cycle_records
from bran_joint_lab_cache_v1 import pack_records, safe_payload, validate_aggregate, valid_sha
from run_bran_source_linkage_audit_v1 import sha, exclusive_json
import hashlib

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'BRAN_NHANES_JOINT_LABS_PROTOCOL_V1.json'
PUBLIC = ROOT / 'BRAN_NHANES_JOINT_LABS_V1'
PRIVATE = ROOT / 'private_artifacts' / 'bran_nhanes_joint_labs_v1'
PREVIOUS_PROTOCOL = ROOT / 'BRAN_OBSERVED_CBC_POOL_PROTOCOL_V1.json'
PREVIOUS_SHA = '397292c03fa89e496ec3295b19ca9200b6d93e3d80717251d07f0003748880f6'
SALT = ROOT / 'private_artifacts' / 'bran_observed_cbc_pool_v1' / 'split_salt.bin'
INPUTS = {f'{kind}_{c.lower()}': f'/Users/ethanwu/nhanes-oculomics/{kind.upper()}_{c}.xpt' for c in ('D', 'E') for kind in ('demo', 'cbc', 'biopro')}
CODE = ('run_bran_nhanes_joint_labs_v1.py', 'bran_nhanes_joint_labs_v1.py', 'bran_joint_lab_cache_v1.py',
        'bran_nhanes_chemistry_observation_v2.py', 'bran_nhanes_chemistry_binding_v1.py', 'bran_nhanes_header_extension_v1.py',
        'bran_clinical_chemistry_semantics_v1.py', 'bran_nhanes_source_reader_v1.py', 'bran_clinical_semantics_v1.py',
        'bran_clinical_snapshot_v1.py', 'bran_clinical_dictionary_binding_v1.py', 'run_bran_source_linkage_audit_v1.py',
        'bran_clinical_source_reader_v1.py', 'bran_six_source_schema_preflight_v1.py',
        'test_bran_nhanes_chemistry_binding_v1.py', 'test_bran_nhanes_chemistry_observation_v2.py',
        'test_bran_nhanes_joint_labs_v1.py', 'test_bran_joint_lab_cache_v1.py', 'test_run_bran_nhanes_joint_labs_v1.py')
POLICY = {'max_rows_per_table': 2000000, 'minimum_observed_CBC_fields': 2,
          'join': 'unique source SEQN within authenticated D/E cycle; union chemistry is not a CBC anchor; no positional or cache-row join',
          'time': 'same examination cycle, not guaranteed same draw or pretreatment',
          'split': 'reuse existing source-local person HMAC salt, 80/10/10; no episode random split',
          'age': 'preserve reported/lower/upper/kind; adult qualification lower bound >=18',
          'truth': 'original measured CBC and BIOPRO only, with explicit cycle-specific creatinine calibration; no imputed truth',
          'value_gate': 'CBC strictly positive, chemistry nonnegative; finite; exact bound units; no upper clipping or disease threshold',
          'assay': ASSAY_POLICY, 'training_permitted': False, 'patient_level_output_permitted': False,
          'counts': 'all counts lower-bound multiples of20, counts below20 suppressed; no exact small counts'}


def runtime():
    return {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__,
            'pandas_xport_reader_sha256': hashlib.sha256(inspect.getsource(XportReader).encode()).hexdigest()}


def prepare():
    if sha(PREVIOUS_PROTOCOL) != PREVIOUS_SHA: raise ValueError('previous protocol mismatch')
    previous = json.loads(PREVIOUS_PROTOCOL.read_text())
    for key, path in INPUTS.items():
        if not key.startswith('biopro') and sha(path) != previous['source_files']['nhanes_' + key]['sha256']:
            raise ValueError('previous source binding mismatch')
    for cycle in ('D', 'E'):
        h = inspect_nhanes_biopro_header(cycle, INPUTS['biopro_' + cycle.lower()])
        if not all(h['whitelist_columns_present'].values()): raise ValueError('required header not present')
    return {'schema': 'bran-nhanes-joint-labs-protocol-v1', 'policy': POLICY, 'previous_protocol_sha256': PREVIOUS_SHA,
            'code_sha256': {n: sha(ROOT / n) for n in CODE}, 'source_files': {k: {'path': p, 'sha256': sha(p)} for k, p in INPUTS.items()},
            'split_salt_sha256': sha(SALT), 'codebook_binding_sha256': codebook_binding_sha256(), 'runtime': runtime()}


def validate_protocol(p):
    if set(p) != {'schema', 'policy', 'previous_protocol_sha256', 'code_sha256', 'source_files', 'split_salt_sha256', 'codebook_binding_sha256', 'runtime'} or p['schema'] != 'bran-nhanes-joint-labs-protocol-v1' or p['policy'] != POLICY or p['runtime'] != runtime():
        raise ValueError('protocol schema or policy mismatch')
    if p['previous_protocol_sha256'] != PREVIOUS_SHA or sha(PREVIOUS_PROTOCOL) != PREVIOUS_SHA or p['codebook_binding_sha256'] != codebook_binding_sha256(): raise ValueError('prior binding mismatch')
    if set(p['code_sha256']) != set(CODE) or set(p['source_files']) != set(INPUTS): raise ValueError('protocol binding keys mismatch')
    for name, h in p['code_sha256'].items():
        if sha(ROOT / name) != h: raise ValueError('code binding mismatch')
    for key, spec in p['source_files'].items():
        if set(spec) != {'path', 'sha256'} or spec['path'] != INPUTS[key] or not valid_sha(spec['sha256']): raise ValueError('source binding mismatch')
    if p['split_salt_sha256'] != sha(SALT): raise ValueError('split salt mismatch')


def records():
    limit = POLICY['max_rows_per_table']
    for cycle in ('D', 'E'):
        suffix = cycle.lower()
        demos = iter_xport_projection(INPUTS['demo_' + suffix], ('SEQN', 'SDDSRVYR', 'RIDAGEYR'), max_rows=limit)
        cbc = iter_xport_projection(INPUTS['cbc_' + suffix], ('SEQN',) + tuple(NHANES_CBC_CODES), max_rows=limit)
        biopro = iter_xport_projection(INPUTS['biopro_' + suffix], ('SEQN',) + tuple(r.column for r in NHANES_BIOPRO_RULES[cycle].values()), max_rows=limit)
        yield from joint_cycle_records(cycle, demos, cbc, biopro)


def run(p):
    validate_protocol(p)
    PUBLIC.mkdir()  # Exclusive attempt, never overwrite terminal evidence.
    phase = 'source_authentication'
    try:
        for key, path in INPUTS.items():
            if sha(path) != p['source_files'][key]['sha256']: raise ValueError('input changed')
        PRIVATE.mkdir(parents=True, mode=0o700)
        os.chmod(PRIVATE, 0o700)
        phase = 'observation_join_and_pack'
        arrays, summary = pack_records('nhanes', records(), SALT.read_bytes())
        phase = 'private_cache_write'
        cache = PRIVATE / 'observations.npz'
        fd = os.open(cache, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as handle: np.savez_compressed(handle, **arrays)
        phase = 'post_scan_authentication'
        for key, path in INPUTS.items():
            if sha(path) != p['source_files'][key]['sha256']: raise ValueError('input changed during scan')
        validate_protocol(p)
        phase = 'aggregate_commit'
        payload = safe_payload('nhanes', summary, sha(cache))
        validate_aggregate(payload)
        exclusive_json(PUBLIC / 'aggregate.json', payload)
        exclusive_json(PUBLIC / 'manifest.json', {'protocol_sha256': sha(PROTOCOL), 'aggregate_sha256': sha(PUBLIC / 'aggregate.json'),
            'split_salt_sha256': sha(SALT), 'patient_level_output_emitted': False, 'training_started': False})
        return True
    except Exception:
        exclusive_json(PUBLIC / 'failure.json', {'status': 'joint_lab_observation_scan_failed', 'phase': phase,
                       'patient_level_output_emitted': False, 'training_started': False})
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-protocol', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare_protocol == args.run: parser.error('choose exactly one operation')
    ok = False
    with _quiet():
        try:
            if args.prepare_protocol: exclusive_json(PROTOCOL, prepare()); ok = True
            else: ok = run(json.loads(PROTOCOL.read_text()))
        except Exception: pass
    print(json.dumps({'status': ('protocol_prepared' if args.prepare_protocol else 'joint_labs_completed') if ok else 'joint_labs_failed',
                      'patient_level_output_emitted': False, 'training_started': False}))
    return 0 if ok else 1


if __name__ == '__main__': raise SystemExit(main())
