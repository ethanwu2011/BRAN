"""Local-only BRSET metadata qualification; never emits private records."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

import numpy as np
import bran_retinal_group_readiness_v1 as kernel

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/Users/ethanwu/brazilian-ophthalmological-1.0.1/labels_brset.csv')
CHECKSUM = SOURCE.parent / 'SHA256SUMS.txt'
EXPECTED_SOURCE = '357354fbe543fd30901406f44c939bb52d457f3c8b701ba70a72735338826b20'
EXPECTED_CHECKSUM = '5b1c6dd5b1157ddb5e130c50b93349f99a9623929c406ecfb474ed8bd34431be'
PROTOCOL = ROOT / 'BRAN_RETINAL_GROUP_READINESS_PROTOCOL_V1.json'
OUT = ROOT / 'BRAN_RETINAL_GROUP_READINESS_V1'
AUDIT = ROOT / 'BRAN_RETINAL_GROUP_READINESS_AUDIT_V1'
PRIVATE = ROOT / 'private_artifacts/bran_retinal_group_readiness_v1'
LOCK = Path('/private/tmp/bran_retinal_extraction_v1.lock')
FILES = ('run_bran_retinal_group_readiness_v1.py', 'bran_retinal_group_readiness_v1.py',
         'test_run_bran_retinal_group_readiness_v1.py', 'test_bran_retinal_group_readiness_v1.py',
         'BRAN_RETINAL_GROUP_READINESS_DESIGN_V1.md')
FIELDS = ('image_id', 'patient_id', 'patient_age', 'exam_eye', *kernel.ENDPOINTS)
ARRAYS = ('image_ids', 'patient_ids', 'ages', 'adult_eligible', 'split', 'labels', 'observed')
CODEBOOK = 'https://physionet.org/content/brazilian-ophthalmological/1.0.1/'
SCHEMA = 'bran-retinal-group-readiness-receipt-v1'


def require(condition):
    if not condition:
        raise ValueError('retinal metadata receipt validation failed')


@contextmanager
def quiet():
    sys.stdout.flush(); sys.stderr.flush()
    out, err = os.dup(1), os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1); os.dup2(null, 2)
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(out, 1); os.dup2(err, 2)
        os.close(null); os.close(out); os.close(err)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def plain(value):
    return {k: plain(v) for k, v in value.items()} if isinstance(value, Mapping) else value


def equal(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)


def regular(path, private=False):
    p = Path(path)
    require(p.is_file() and not p.is_symlink())
    if private:
        require(stat.S_IMODE(p.stat().st_mode) == 0o600)


def inventory(path, names, private=False):
    require(path.is_dir() and not path.is_symlink())
    require({p.name for p in path.iterdir()} == set(names))
    if private:
        require(stat.S_IMODE(path.stat().st_mode) == 0o700)
    for name in names:
        regular(path / name, private=private)


def absent(*paths):
    require(all(not p.exists() and not p.is_symlink() for p in paths))


def source_receipt():
    # Local source parent may resolve onto the external drive; individual files
    # must still be regular, and exact bytes are bound before and after parsing.
    regular(SOURCE); regular(CHECKSUM)
    require(sha(SOURCE) == EXPECTED_SOURCE and sha(CHECKSUM) == EXPECTED_CHECKSUM)
    matches = []
    with CHECKSUM.open() as f:
        for line in f:
            m = re.fullmatch(r'([0-9a-fA-F]{64})\s+\*?(?:\./)?labels_brset\.csv', line.rstrip('\r\n'))
            if m:
                matches.append(m[1].lower())
    require(matches == [EXPECTED_SOURCE])
    with SOURCE.open(encoding='utf-8-sig', newline='') as f:
        header = next(csv.reader(f, strict=True))
    require(len(header) == len(set(header)) and all(type(x) is str and x for x in header))
    require(set(FIELDS).issubset(header))
    return {'labels_sha256': EXPECTED_SOURCE, 'local_checksum_sha256': EXPECTED_CHECKSUM,
            'local_checksum_match': True, 'official_release_independently_authenticated': False,
            'codebook_url': CODEBOOK, 'canonical_illumination_header': 'illumination' in header,
            'legacy_illumination_header': any(x in header for x in ('Illuminaton', 'ilumination'))}


def runtime():
    return {'python': sys.version, 'python_executable_sha256': sha(Path(sys.executable).resolve()),
            'numpy_version': np.__version__}


def expected_protocol(salt_hash):
    return {'schema': SCHEMA, 'source': source_receipt(),
            'code_sha256': {name: sha(ROOT / name) for name in FILES}, 'runtime': runtime(),
            'salt_sha256': salt_hash, 'split_namespace': 'bran-retinal-group-readiness-v1\\x00',
            'split_percentages': [80, 10, 10], 'minimum_age_years': 18,
            'minimum_label_class_patients': 20, 'maximum_rows': 50000,
            'patient_level_output_permitted': False, 'training_permitted': False}


def load_protocol(pin):
    regular(PROTOCOL); regular(PRIVATE / 'split_salt.bin', private=True)
    require(type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin) is not None and sha(PROTOCOL) == pin)
    require(len((PRIVATE / 'split_salt.bin').read_bytes()) == 32)
    p = json.loads(PROTOCOL.read_text())
    require(equal(p, expected_protocol(sha(PRIVATE / 'split_salt.bin'))))
    return p


def prepare():
    absent(PROTOCOL, OUT, AUDIT, PRIVATE)
    # Hash/schema checks before secret generation; no patient rows inspected.
    p = expected_protocol('pending')
    PRIVATE.mkdir(mode=0o700)
    with (PRIVATE / 'split_salt.bin').open('xb') as f:
        os.fchmod(f.fileno(), 0o600)
        f.write(os.urandom(32))
    p['salt_sha256'] = sha(PRIVATE / 'split_salt.bin')
    write_json(PROTOCOL, p)
    return sha(PROTOCOL)


def compute():
    source_receipt()
    rows = []
    with SOURCE.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, strict=True)
        header = next(reader)
        indexes = [header.index(field) for field in FIELDS]
        for row in reader:
            require(len(row) == len(header) and len(rows) < 50000)
            rows.append({field: row[i] for field, i in zip(FIELDS, indexes)})
    result = kernel.assess(rows, salt=(PRIVATE / 'split_salt.bin').read_bytes())
    kernel.validate_summary(result.safe_summary)
    source_receipt()
    return result


def save_private(result):
    with (PRIVATE / 'assignment.npz').open('xb') as f:
        os.fchmod(f.fileno(), 0o600)
        np.savez_compressed(f, **{name: np.asarray(getattr(result, name)) for name in ARRAYS})


def compare_private(result):
    regular(PRIVATE / 'assignment.npz', private=True)
    with np.load(PRIVATE / 'assignment.npz', allow_pickle=False) as stored:
        require(set(stored.files) == set(ARRAYS))
        for name in ARRAYS:
            a, b = stored[name], np.asarray(getattr(result, name))
            require(a.dtype == b.dtype and a.shape == b.shape)
            require(np.array_equal(a, b, equal_nan=True) if a.dtype.kind == 'f' else np.array_equal(a, b))


def manifest(pin):
    return {'schema': SCHEMA, 'protocol_sha256': pin, 'aggregate_sha256': sha(OUT / 'aggregate.json'),
            'assignment_sha256': sha(PRIVATE / 'assignment.npz'),
            'salt_sha256': sha(PRIVATE / 'split_salt.bin'), 'patient_level_output_emitted': False}


def aggregate(p, pin, result):
    return {'schema': SCHEMA, 'status': 'metadata_assessed', 'protocol_sha256': pin,
            'source': p['source'], 'assessment': plain(result.safe_summary),
            'new_encoder_trained': False, 'unified_bran_improvement_established': False}


def authenticate(pin):
    p = load_protocol(pin)
    # A failed replay is not overridden by a still-matching terminal hash.
    require(not (AUDIT / 'failure.json').exists())
    inventory(PRIVATE, ('split_salt.bin', 'assignment.npz'), private=True)
    inventory(OUT, ('aggregate.json', 'manifest.json'))
    m = json.loads((OUT / 'manifest.json').read_text())
    require(equal(m, manifest(pin)))
    a = json.loads((OUT / 'aggregate.json').read_text())
    kernel.validate_summary(a['assessment'])
    # Closed keys and fixed receipts, not a permissive substring privacy scan.
    require(equal(a, {'schema': SCHEMA, 'status': 'metadata_assessed', 'protocol_sha256': pin,
        'source': p['source'], 'assessment': a['assessment'], 'new_encoder_trained': False,
        'unified_bran_improvement_established': False}))
    return p, a


def execute(pin, state):
    p = load_protocol(pin)
    inventory(PRIVATE, ('split_salt.bin',), private=True)
    absent(OUT, AUDIT)
    OUT.mkdir(); state['owned'] = OUT
    state['phase'] = 'metadata_assessment'
    result = compute()
    state['phase'] = 'private_receipt'
    save_private(result); compare_private(result)
    load_protocol(pin)
    state['phase'] = 'publication'
    write_json(OUT / 'aggregate.json', aggregate(p, pin, result))
    write_json(OUT / 'manifest.json', manifest(pin))
    authenticate(pin)


def audit(pin, state):
    p, a = authenticate(pin)
    absent(AUDIT)
    AUDIT.mkdir(); state['owned'] = AUDIT
    state['phase'] = 'independent_metadata_replay'
    result = compute(); compare_private(result)
    require(equal(a, aggregate(p, pin, result)))
    authenticate(pin)
    value = audit_value(pin)
    write_json(AUDIT / 'audit.json', value)
    write_json(AUDIT / 'manifest.json', {'protocol_sha256': pin, 'audit_sha256': sha(AUDIT / 'audit.json')})
    authenticate_audit(pin, sha(AUDIT / 'audit.json'))


def audit_value(pin):
    return {'schema': SCHEMA, 'status': 'authenticated', 'protocol_sha256': pin,
            'terminal_manifest_sha256': sha(OUT / 'manifest.json'),
            'source_metadata_replayed': True, 'all_private_arrays_replayed': True,
            'split_salt_reused': True, 'model_training_repeated': False,
            'patient_level_output_emitted': False}


def authenticate_audit(pin, audit_pin):
    authenticate(pin)
    inventory(AUDIT, ('audit.json', 'manifest.json'))
    require(sha(AUDIT / 'audit.json') == audit_pin)
    require(equal(json.loads((AUDIT / 'audit.json').read_text()), audit_value(pin)))
    require(equal(json.loads((AUDIT / 'manifest.json').read_text()),
                  {'protocol_sha256': pin, 'audit_sha256': audit_pin}))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'audit', 'verify'))
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--audit-sha256')
    args = parser.parse_args(argv)
    state = {'phase': 'authentication', 'owned': None}
    response = {'status': 'failed', 'patient_level_output_emitted': False}
    with quiet():
        try:
            with LOCK.open('a') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                pin = args.protocol_sha256
                if args.action == 'prepare':
                    pin = prepare()
                elif args.action == 'run':
                    execute(pin, state)
                elif args.action == 'audit':
                    audit(pin, state)
                elif args.audit_sha256:
                    authenticate_audit(pin, args.audit_sha256)
                else:
                    authenticate(pin)
                response = {'status': 'complete', 'action': args.action,
                            'protocol_sha256': pin, 'patient_level_output_emitted': False}
                if args.action != 'prepare':
                    response['aggregate_sha256'] = sha(OUT / 'aggregate.json')
                if args.action == 'audit' or args.audit_sha256:
                    response['audit_sha256'] = sha(AUDIT / 'audit.json')
        except Exception:
            response['phase'] = state['phase']
            if state['owned'] is not None:
                try:
                    write_json(state['owned'] / 'failure.json', response)
                except Exception:
                    pass
    print(json.dumps(response, sort_keys=True))
    return 0 if response['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
