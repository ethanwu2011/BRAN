"""Conditional local paired extraction, not a unified BRAN refit or promotion.

Source admission and a positive authenticated source-stage result are required
before preparation. No action on import; all CLI work is silent under one lock.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

import numpy as np

import bran_multisource_paired_source_v1 as source
import bran_retinal_multisource_encoder_binding_v1 as binding
import bran_retinal_extraction_kernel_v1 as kernel
from bran_authenticated_retinal_input_v1 import pool_arrays

r = binding.run.r
ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'BRAN_MULTISOURCE_PAIRED_EXTRACTION_PROTOCOL_V1.json'
OUT = ROOT / 'BRAN_MULTISOURCE_PAIRED_EXTRACTION_V1'
AUDIT = ROOT / 'BRAN_MULTISOURCE_PAIRED_EXTRACTION_AUDIT_V1'
PRIVATE = ROOT / 'private_artifacts/bran_multisource_paired_extraction_v1'
LOCK = binding.run.LOCK
SCHEMA = 'bran-multisource-paired-extraction-v1'
ERROR = 'paired extraction validation failed'
PRIVATE_FILES = ('inventory.json', 'cohort.json', 'encoder_contract.json', 'features.npy', 'pooled.npz')
FILES = (
    'run_bran_multisource_paired_extraction_v1.py', 'test_run_bran_multisource_paired_extraction_v1.py',
    'BRAN_MULTISOURCE_PAIRED_EXTRACTION_DESIGN_V1.md',
    'bran_multisource_paired_source_v1.py', 'test_bran_multisource_paired_source_v1.py',
    'bran_retinal_multisource_encoder_binding_v1.py', 'test_bran_retinal_multisource_encoder_binding_v1.py',
    'bran_retinal_extraction_kernel_v1.py', 'bran_authenticated_retinal_input_v1.py',
)


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def read_json(path):
    r.regular(path)
    return json.loads(path.read_text())


def parameters():
    return {'batch_size': 16, 'width': 384, 'image_size': 224, 'device': 'mps',
            'inference_audit_images': 32, 'inference_atol': 1e-5, 'inference_rtol': 1e-5,
            'pooling': 'mean_actual_images_within_original_patient', 'resume_supported': False}


def expected_protocol(source_protocol_sha256, source_audit_sha256):
    # Gate before paired cohort/source reads, output creation or inference loading.
    contract = binding.authenticate_contract(source_protocol_sha256, source_audit_sha256)
    return {'schema': SCHEMA, 'encoder': contract, 'paired_source': source.authenticate_binding(),
            'code_sha256': {name: r.sha(ROOT / name) for name in FILES},
            'runtime': binding.run.ref.runtime(), 'parameters': parameters(),
            'patient_level_output_permitted': False, 'unified_refit_permitted': False}


def prepare(source_protocol_sha256, source_audit_sha256):
    value = expected_protocol(source_protocol_sha256, source_audit_sha256)
    r.absent(PROTOCOL, OUT, AUDIT, PRIVATE)
    r.write_json(PROTOCOL, value)
    return r.sha(PROTOCOL)


def load_protocol(pin):
    binding.pin(pin)
    value = read_json(PROTOCOL)
    require(r.sha(PROTOCOL) == pin)
    expected = expected_protocol(value['encoder']['protocol_sha256'], value['encoder']['audit_sha256'])
    require(r.equal(value, expected))
    return value


def write_private(path, writer):
    # Exclusive, mode-correct from creation; never truncate/overwrite prior evidence.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        writer(handle)
        handle.flush()
        os.fsync(handle.fileno())


def write_private_json(path, value):
    write_private(path, lambda handle: handle.write(
        json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8')))


def cohort_value(paired):
    # Private only. Exact original ordering, folds, selected source rows, missingness.
    return {'patient_ids': list(paired.private_patient_ids), 'folds': paired.private_folds.tolist(),
            'retinal_present': paired.private_retinal_present.tolist(),
            'selection': list(paired.private_selection)}


def check_source(paired, protocol):
    require(paired.binding == protocol['paired_source'])


def progress(state, phase):
    require(phase in ('authentication', 'extraction', 'pooling', 'publication', 'audit'))
    state['phase'] = phase
    if state.get('owned') is not None:
        temporary = state['owned'] / 'progress.tmp'
        r.absent(temporary)
        r.write_json(temporary, {'phase': phase, 'patient_level_output_emitted': False})
        os.replace(temporary, state['owned'] / 'progress.json')


def extract(protocol, state):
    contract = protocol['encoder']
    candidate = binding.load_candidate(contract['protocol_sha256'], contract['audit_sha256'])
    require(r.equal(candidate.contract, contract))
    with source.QualifiedPairedSource() as paired:
        check_source(paired, protocol)
        records = [dict(record) for record in paired.private_records]
        PRIVATE.parent.mkdir(exist_ok=True)
        PRIVATE.mkdir(mode=0o700)
        write_private_json(PRIVATE / 'inventory.json', records)
        write_private_json(PRIVATE / 'cohort.json', cohort_value(paired))
        write_private_json(PRIVATE / 'encoder_contract.json', contract)
        progress(state, 'extraction')
        kernel.extract_batches(records, PRIVATE / 'features.npy', paired.read, candidate.encode,
                               batch_size=parameters()['batch_size'], width=parameters()['width'])
        progress(state, 'pooling')
        features = np.load(PRIVATE / 'features.npy', mmap_mode='r', allow_pickle=False)
        try:
            pooled, present = pool_arrays(records, features, list(paired.private_patient_ids))
            require(np.array_equal(present, paired.private_retinal_present))
            write_private(PRIVATE / 'pooled.npz', lambda handle: np.savez(handle, features=pooled, present=present))
        finally:
            features._mmap.close()
    # The context rechecks original source/cohort identities on exit.
    return {name: r.sha(PRIVATE / name) for name in PRIVATE_FILES}


def authenticate_private(hashes):
    require(type(hashes) is dict and set(hashes) == set(PRIVATE_FILES))
    r.inventory(PRIVATE, PRIVATE_FILES, private=True)
    for name in PRIVATE_FILES:
        binding.pin(hashes[name])
        require(os.lstat(PRIVATE / name).st_nlink == 1 and r.sha(PRIVATE / name) == hashes[name])


def replay_private(protocol, hashes):
    authenticate_private(hashes)  # All hashes before any private deserialization.
    records = read_json(PRIVATE / 'inventory.json')
    cohort = read_json(PRIVATE / 'cohort.json')
    require(r.equal(read_json(PRIVATE / 'encoder_contract.json'), protocol['encoder']))
    with source.QualifiedPairedSource() as paired:
        check_source(paired, protocol)
        require(records == list(paired.private_records) and r.equal(cohort, cohort_value(paired)))
        kernel.validate_output(PRIVATE / 'features.npy', expected_rows=len(records), width=parameters()['width'])
        features = np.load(PRIVATE / 'features.npy', mmap_mode='r', allow_pickle=False)
        try:
            pooled, present = pool_arrays(records, features, list(paired.private_patient_ids))
            with np.load(PRIVATE / 'pooled.npz', allow_pickle=False) as stored:
                require(set(stored.files) == {'features', 'present'}
                        and stored['features'].dtype == np.float64 and stored['present'].dtype == bool
                        and np.array_equal(stored['features'], pooled)
                        and np.array_equal(stored['present'], present)
                        and np.array_equal(present, paired.private_retinal_present))
        finally:
            features._mmap.close()
        counts = (len(records), len(paired.private_patient_ids))
    authenticate_private(hashes)
    return counts


def floor_count(value):
    return None if value < 20 else int(value // 20 * 20)


def result_value(pin, counts):
    return {'schema': SCHEMA, 'status': 'extraction_completed', 'protocol_sha256': pin,
            'image_count_floor20': floor_count(counts[0]), 'patient_count_floor20': floor_count(counts[1]),
            'dimension': parameters()['width'], 'dtype': 'float32',
            'patient_pool_dtype': 'float64', 'original_cohort_and_folds_preserved': True,
            'natural_retinal_availability_preserved': True, 'all_selected_images_encoded_once': True,
            'all_rows_authenticated_before_decode': True, 'full_patient_pooling_replayed': True,
            'encoder_training_performed': False, 'unified_refit_performed': False,
            'old_coordinate_compatibility_established': False, 'unified_model_promoted': False,
            'clinical_benefit_established': False, 'patient_level_output_emitted': False,
            'patient_processing_local_only': True, 'patient_images_saved_or_displayed': False}


def marker(pin):
    return {'schema': SCHEMA, 'protocol_sha256': pin, 'manifest_sha256': r.sha(OUT / 'manifest.json'),
            'status': 'completed', 'patient_level_output_emitted': False}


def payload(pin, *, terminal=True):
    protocol = load_protocol(pin)
    require(not (AUDIT / 'failure.json').exists())
    r.inventory(OUT, ('results.json', 'manifest.json', 'progress.json', *(('success.json',) if terminal else ())))
    require(r.equal(read_json(OUT / 'progress.json'), {'phase': 'publication', 'patient_level_output_emitted': False}))
    manifest = read_json(OUT / 'manifest.json')
    require(type(manifest) is dict and set(manifest) == {
        'schema', 'protocol_sha256', 'results_sha256', 'private_sha256', 'patient_level_output_emitted'}
        and manifest['schema'] == SCHEMA and manifest['protocol_sha256'] == pin
        and manifest['patient_level_output_emitted'] is False
        and manifest['results_sha256'] == r.sha(OUT / 'results.json'))
    counts = replay_private(protocol, manifest['private_sha256'])
    value = result_value(pin, counts)
    require(r.equal(read_json(OUT / 'results.json'), value))
    if terminal:
        require(r.equal(read_json(OUT / 'success.json'), marker(pin)))
    return value


def run(pin, state):
    protocol = load_protocol(pin)
    r.absent(OUT, AUDIT, PRIVATE)
    OUT.mkdir()
    state['owned'] = OUT
    hashes = extract(protocol, state)
    counts = replay_private(protocol, hashes)
    progress(state, 'publication')
    r.write_json(OUT / 'results.json', result_value(pin, counts))
    r.write_json(OUT / 'manifest.json', {'schema': SCHEMA, 'protocol_sha256': pin,
        'results_sha256': r.sha(OUT / 'results.json'), 'private_sha256': hashes,
        'patient_level_output_emitted': False})
    payload(pin, terminal=False)
    r.write_json(OUT / 'success.json', marker(pin))
    state['owned'] = None


def inference_audit(protocol):
    contract, p = protocol['encoder'], parameters()
    candidate = binding.load_candidate(contract['protocol_sha256'], contract['audit_sha256'])
    require(r.equal(candidate.contract, contract))
    with source.QualifiedPairedSource() as paired:
        check_source(paired, protocol)
        require(paired.reauthenticate_all_bytes() is True)
        records = [dict(record) for record in paired.private_records]
        require(len(records) >= p['inference_audit_images'])
        indices = np.linspace(0, len(records) - 1, p['inference_audit_images'], dtype=np.int64)
        require(len(np.unique(indices)) == p['inference_audit_images'])
        stored = np.load(PRIVATE / 'features.npy', mmap_mode='r', allow_pickle=False)
        try:
            for start in range(0, len(indices), p['batch_size']):
                selected = indices[start:start + p['batch_size']]
                images = np.stack([paired.read(records[index]) for index in selected])
                actual = candidate.encode(images)
                require(np.allclose(actual, stored[selected], atol=p['inference_atol'], rtol=p['inference_rtol']))
        finally:
            stored._mmap.close()


def audit_value(pin):
    return {'schema': SCHEMA, 'status': 'authenticated', 'protocol_sha256': pin,
            'terminal_manifest_sha256': r.sha(OUT / 'manifest.json'),
            'all_selected_raw_bytes_reauthenticated': True,
            'inference_images_replayed': parameters()['inference_audit_images'],
            'all_inference_replayed': False, 'full_patient_pooling_replayed': True,
            'original_cohort_and_folds_preserved': True, 'encoder_training_repeated': False,
            'unified_refit_performed': False, 'unified_model_promoted': False,
            'patient_level_output_emitted': False}


def audit(pin, state):
    payload(pin)
    r.absent(AUDIT)
    AUDIT.mkdir()
    state['owned'] = AUDIT
    progress(state, 'audit')
    inference_audit(load_protocol(pin))
    payload(pin)
    progress(state, 'publication')
    r.write_json(AUDIT / 'audit.json', audit_value(pin))
    r.write_json(AUDIT / 'success.json', {'audit_sha256': r.sha(AUDIT / 'audit.json'), 'protocol_sha256': pin})
    state['owned'] = None


def authenticate_audit(pin, audit_pin):
    payload(pin)
    r.inventory(AUDIT, ('audit.json', 'success.json', 'progress.json'))
    binding.pin(audit_pin)
    require(r.sha(AUDIT / 'audit.json') == audit_pin and r.equal(read_json(AUDIT / 'audit.json'), audit_value(pin))
            and r.equal(read_json(AUDIT / 'success.json'), {'audit_sha256': audit_pin, 'protocol_sha256': pin})
            and r.equal(read_json(AUDIT / 'progress.json'), {'phase': 'publication', 'patient_level_output_emitted': False}))


def load_private_refit_input(pin, audit_pin, *, patient_ids, folds, selection):
    """Private readonly inputs for a future refit; caller must hold the shared lock.

    Exact original cohort identity and completed extraction audit are mandatory.
    This does not select fit membership or grant permission to train on test rows.
    """
    with r.quiet():
        try:
            authenticate_audit(pin, audit_pin)
            manifest = read_json(OUT / 'manifest.json')
            authenticate_private(manifest['private_sha256'])
            cohort = read_json(PRIVATE / 'cohort.json')
            require(type(patient_ids) in (list, tuple) and all(type(item) is str for item in patient_ids)
                    and list(patient_ids) == cohort['patient_ids'])
            require(type(folds) is np.ndarray and folds.dtype == np.int64
                    and type(selection) in (list, tuple) and all(type(item) is int for item in selection)
                    and np.array_equal(folds, np.asarray(cohort['folds'], dtype=np.int64))
                    and list(selection) == cohort['selection'])
            with np.load(PRIVATE / 'pooled.npz', allow_pickle=False) as stored:
                features, present = stored['features'].copy(), stored['present'].copy()
            features.setflags(write=False); present.setflags(write=False)
            contract = read_json(PRIVATE / 'encoder_contract.json')
            binding.validate_contract(contract)
            authenticate_private(manifest['private_sha256'])
            return features, present, contract
        except Exception:
            raise ValueError(ERROR) from None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'verify', 'audit'))
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--source-protocol-sha256')
    parser.add_argument('--source-audit-sha256')
    parser.add_argument('--audit-sha256')
    args = parser.parse_args(argv)
    state = {'phase': 'authentication', 'owned': None}
    answer = {'status': 'failed', 'patient_level_output_emitted': False}
    with r.quiet():
        try:
            with LOCK.open('a') as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                pin = args.protocol_sha256
                if args.action == 'prepare':
                    pin = prepare(args.source_protocol_sha256, args.source_audit_sha256)
                elif args.action == 'run':
                    run(pin, state)
                elif args.action == 'audit':
                    audit(pin, state)
                elif args.audit_sha256:
                    authenticate_audit(pin, args.audit_sha256)
                else:
                    payload(pin)
                answer = {'status': 'complete', 'action': args.action, 'protocol_sha256': pin,
                          'patient_level_output_emitted': False}
                if args.action != 'prepare':
                    answer['results_sha256'] = r.sha(OUT / 'results.json')
                if args.action == 'audit' or args.audit_sha256:
                    answer['audit_sha256'] = r.sha(AUDIT / 'audit.json')
        except Exception:
            answer['phase'] = state['phase']
            if state['owned'] is not None:
                try:
                    r.write_json(state['owned'] / 'failure.json', answer)
                except Exception:
                    pass
    print(json.dumps(answer, sort_keys=True))
    return int(answer['status'] != 'complete')


if __name__ == '__main__':
    raise SystemExit(main())
