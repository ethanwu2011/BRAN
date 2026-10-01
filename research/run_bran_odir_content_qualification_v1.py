"""Prospective ODIR pixel/overlap qualification; FD-quiet, local-only, no training."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

import bran_odir_content_decode_v1 as decode
import bran_odir_content_screen_v1 as screen
import run_bran_odir_metadata_qualification_v2 as metadata
import run_bran_retinal_content_admission_v1 as reference

r = reference.r
ROOT = r.ROOT
PROTOCOL = ROOT / 'BRAN_ODIR_CONTENT_QUALIFICATION_PROTOCOL_V1.json'
OUT = ROOT / 'BRAN_ODIR_CONTENT_QUALIFICATION_V1'
AUDIT = ROOT / 'BRAN_ODIR_CONTENT_QUALIFICATION_AUDIT_V1'
PRIVATE = ROOT / 'private_artifacts/bran_odir_content_qualification_v1'
LOCK = r.LOCK
META_PY = Path('/Users/ethanwu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3')
META_PIN = '6dc4ed24e0aaa4e24949dae23c63002f2e7f3b6f86bebb0c26a302caca462647'
META_AUDIT = '5f519e1304d0ef35aff9276ca88e6f2bc7516928bb8a9e5d5a6e082098bd56bf'
REF_PIN = 'ee76c14b6b59ad090772508feafa9bf5e19a4258ee3d47e404d51e5d873ad7e6'
REF_AUDIT = 'e993efec14fe922a9e0108acc598b75b88d2bc64c31bb411bc7812e908ad252d'
SCHEMA = 'bran-odir-content-qualification-v1'
FILES = ('run_bran_odir_content_qualification_v1.py', 'bran_odir_content_decode_v1.py',
         'bran_odir_content_screen_v1.py', 'test_run_bran_odir_content_qualification_v1.py',
         'test_bran_odir_content_decode_v1.py', 'test_bran_odir_content_screen_v1.py',
         'BRAN_ODIR_CONTENT_QUALIFICATION_DESIGN_V1.md')
PHASES = ('authentication', 'selection', 'image_content', 'overlap_screen', 'audit_bytes', 'publication')
PRIVATE_FILES = ('source_locator.json', 'selection.json', 'fingerprints.npz', 'flags.npz')
HASH_KEYS = ('decoded_sha256', 'phash', 'dhash')


def upstream():
    # Original runtime checks stay intact. Child does not acquire the parent's lock.
    code = ('import run_bran_odir_metadata_qualification_v2 as m; '
            f'm.authenticate_audit({META_PIN!r},{META_AUDIT!r}); print("authenticated")')
    child = subprocess.run([str(META_PY), '-c', code], cwd=ROOT, capture_output=True, timeout=60)
    r.require(child.returncode == 0 and child.stdout == b'authenticated\n' and not child.stderr)
    reference.authenticate_audit(REF_PIN, REF_AUDIT)
    return {
        'metadata_protocol_sha256': META_PIN, 'metadata_audit_sha256': META_AUDIT,
        'reference_protocol_sha256': REF_PIN, 'reference_audit_sha256': REF_AUDIT,
        'metadata_python_sha256': r.sha(META_PY.resolve()),
        'metadata_records_sha256': r.sha(metadata.PRIVATE / 'grouped_records.json'),
        'reference_fingerprints_sha256': {name: r.sha(reference.PRIVATE / f'{name}_hashes.npz')
                                        for name in ('ai', 'brset')},
    }


def select_records(grouped):
    """Select only uniquely linked adult eyes; no invented counterpart or identity."""
    records = grouped['grouped']['records']
    r.require(type(records) is list)
    result = []
    seen_people = set()
    for row in records:
        patient = row['patient_id']
        r.require(type(patient) is str and patient and patient not in seen_people)
        seen_people.add(patient)
        if not set(row['reasons']).issubset({'missing_member_link', 'ambiguous_member_link'}):
            continue
        r.require(type(row['age']) in (int, float) and 18 <= row['age'] <= 130)
        for eye in ('left', 'right'):
            candidates = row[eye]['member_candidates']
            if len(candidates) == 1:
                r.require(type(candidates[0]) is str and candidates[0])
                result.append({'patient_id': patient, 'eye': eye, 'member': candidates[0]})
    r.require(result and len({row['member'] for row in result}) == len(result))
    return result


def selected():
    rows = select_records(json.loads((metadata.PRIVATE / 'grouped_records.json').read_text()))
    aggregate = json.loads((metadata.OUT / 'aggregate.json').read_text())
    expected = aggregate['qualification']['identified_record_qualification']['counts_rounded_down20']
    r.require(len(rows) // 20 * 20 == expected['uniquely_linked_adult_eyes'])
    r.require(len({row['patient_id'] for row in rows}) >= 20)
    return rows


def source_receipt():
    chosen = json.loads((PRIVATE / 'source_locator.json').read_text())
    _, _, receipt = metadata.old.source(chosen)
    expected = json.loads(metadata.PROTOCOL.read_text())['source']
    r.require(r.equal(receipt, expected))
    return receipt


def template():
    return {'schema': SCHEMA, 'status': 'frozen_before_image_content_qualification',
        'upstream': upstream(), 'code_sha256': {name: r.sha(ROOT / name) for name in FILES},
        'runtime': reference.runtime(), 'source': source_receipt(),
        'selection_sha256': r.sha(PRIVATE / 'selection.json'),
        'source_locator_sha256': r.sha(PRIVATE / 'source_locator.json'),
        'parameters': {'hash_size': 128, 'secondary_size': 224, 'dual_hamming_radius': 4,
            'workers': 4, 'window': 128, 'replay_images': 32, 'max_image_bytes': decode.LIMIT,
            'max_image_pixels': decode.MAX_PIXELS, 'cross_person_whole_person_quarantine': True,
            'reference_whole_person_quarantine': True, 'within_direct_means_cross_person': True,
            'same_person_near_match_retained': True, 'same_person_exact_keep_first': True},
        'training_admitted': False, 'reference_raw_images_rescanned': False}


def protocol(pin):
    r.regular(PROTOCOL)
    r.require(type(pin) is str and len(pin) == 64 and r.sha(PROTOCOL) == pin)
    for name in ('selection.json', 'source_locator.json'):
        r.regular(PRIVATE / name, private=True)
    value = json.loads(PROTOCOL.read_text())
    r.require(r.equal(value, template()))
    r.require(r.equal(json.loads((PRIVATE / 'selection.json').read_text()), selected()))
    r.require(r.equal(json.loads((PRIVATE / 'source_locator.json').read_text()),
                      json.loads((metadata.PRIVATE / 'source_locator.json').read_text())))
    return value


def progress(state, phase, done=0, total=0):
    r.require(phase in PHASES and type(done) is int and type(total) is int and 0 <= done <= total)
    state['phase'] = phase
    if state['owned'] is not None:
        value = {'phase': phase, 'completed_rounded_down20': done // 20 * 20,
                 'total_rounded_down20': total // 20 * 20}
        path = state['owned'] / 'progress.tmp'
        r.write_json(path, value)
        os.replace(path, state['owned'] / 'progress.json')


def validate_progress(value, terminal=False):
    r.require(type(value) is dict and set(value) == {
        'phase', 'completed_rounded_down20', 'total_rounded_down20'})
    r.require(type(value['phase']) is str and value['phase'] in PHASES)
    for key in ('completed_rounded_down20', 'total_rounded_down20'):
        r.require(type(value[key]) is int and value[key] >= 0 and value[key] % 20 == 0)
    r.require(value['completed_rounded_down20'] <= value['total_rounded_down20'])
    if terminal:
        r.require(value == {'phase': 'publication', 'completed_rounded_down20': 0, 'total_rounded_down20': 0})


def references():
    return {name: {key: value[key] for key in HASH_KEYS} for name, value in (
        ('ai_readi', reference.load_npz(reference.PRIVATE / 'ai_hashes.npz')),
        ('brset', reference.load_npz(reference.PRIVATE / 'brset_hashes.npz')))}


def screened(selection, fingerprints):
    return screen.screen([row['patient_id'] for row in selection],
                         {key: fingerprints[key] for key in HASH_KEYS}, references())


def result(pin, summary):
    screen.validate_summary(summary)
    return {'schema': SCHEMA, 'status': 'content_qualification_completed', 'protocol_sha256': pin,
        'content_screen': summary, 'selected_odir_image_bytes_authenticated': True,
        'reference_raw_images_rescanned': False, 'original_release_authenticated': False,
        'official_split_authenticated': False, 'reuse_terms_authenticated': False,
        'training_admitted': False, 'patient_level_output_emitted': False}


def manifest(pin):
    return {'protocol_sha256': pin, 'aggregate_sha256': r.sha(OUT / 'aggregate.json'),
            'private_artifact_sha256': {name: r.sha(PRIVATE / name) for name in PRIVATE_FILES}}


def authenticate(pin, replay=False):
    protocol(pin)
    r.inventory(OUT, ('aggregate.json', 'manifest.json', 'progress.json'))
    r.inventory(PRIVATE, PRIVATE_FILES, private=True)
    r.require(not (AUDIT / 'failure.json').exists())
    validate_progress(json.loads((OUT / 'progress.json').read_text()), terminal=True)
    r.require(r.equal(json.loads((OUT / 'manifest.json').read_text()), manifest(pin)))
    selection = json.loads((PRIVATE / 'selection.json').read_text())
    hashes = reference.load_npz(PRIVATE / 'fingerprints.npz')
    flags = reference.load_npz(PRIVATE / 'flags.npz')
    decode.validate_hashes(hashes, len(selection)); screen.validate_flags(flags)
    r.require(all(len(value) == len(selection) for value in flags.values()))
    value = json.loads((OUT / 'aggregate.json').read_text())
    r.require(r.equal(value, result(pin, value['content_screen'])))
    if replay:
        expected, summary = screened(selection, hashes)
        r.require(all(np.array_equal(flags[key], expected[key]) for key in flags)
                  and r.equal(value, result(pin, summary)))
    return selection, hashes, value


def audit_value(pin):
    return {'schema': SCHEMA, 'status': 'authenticated', 'protocol_sha256': pin,
        'terminal_manifest_sha256': r.sha(OUT / 'manifest.json'),
        'all_selected_odir_raw_bytes_reauthenticated': True, 'deterministic_pixel_replay_passed': True,
        'all_overlap_and_retention_flags_replayed': True, 'cached_reference_fingerprints_authenticated': True,
        'reference_raw_images_rescanned': False, 'cross_study_identity_proven': False,
        'training_admitted': False, 'patient_level_output_emitted': False}


def authenticate_audit(pin, audit_pin):
    authenticate(pin)
    r.inventory(AUDIT, ('audit.json', 'manifest.json', 'progress.json'))
    validate_progress(json.loads((AUDIT / 'progress.json').read_text()), terminal=True)
    r.require(r.sha(AUDIT / 'audit.json') == audit_pin and
              r.equal(json.loads((AUDIT / 'audit.json').read_text()), audit_value(pin)))
    r.require(r.equal(json.loads((AUDIT / 'manifest.json').read_text()),
                      {'protocol_sha256': pin, 'audit_sha256': audit_pin}))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'audit', 'verify'))
    parser.add_argument('--protocol-sha256'); parser.add_argument('--audit-sha256')
    args = parser.parse_args(argv)
    state = {'phase': 'authentication', 'owned': None}
    answer = {'status': 'failed', 'action': args.action, 'patient_level_output_emitted': False}
    with r.quiet():
        try:
            with LOCK.open('a') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                pin = args.protocol_sha256
                if args.action == 'prepare':
                    upstream(); r.absent(PROTOCOL, OUT, AUDIT, PRIVATE)
                    progress(state, 'selection')
                    rows = selected()
                    chosen = json.loads((metadata.PRIVATE / 'source_locator.json').read_text())
                    PRIVATE.mkdir(mode=0o700)
                    reference.private_json(PRIVATE / 'selection.json', rows)
                    reference.private_json(PRIVATE / 'source_locator.json', chosen)
                    r.write_json(PROTOCOL, template()); pin = r.sha(PROTOCOL); protocol(pin)
                elif args.action == 'run':
                    protocol(pin); r.absent(OUT, AUDIT)
                    r.inventory(PRIVATE, ('selection.json', 'source_locator.json'), private=True)
                    OUT.mkdir(); state['owned'] = OUT
                    rows = json.loads((PRIVATE / 'selection.json').read_text())
                    chosen = json.loads((PRIVATE / 'source_locator.json').read_text())
                    progress(state, 'image_content')
                    with decode.PrivateArchive(chosen['archive']) as archive:
                        values = archive.map(lambda i, data: decode.fingerprint(data),
                            [row['member'] for row in rows],
                            lambda done, total: progress(state, 'image_content', done, total))
                    hashes = decode.stack_fingerprints(values); del values
                    decode.validate_hashes(hashes, len(rows))
                    reference.private_npz(PRIVATE / 'fingerprints.npz', hashes)
                    progress(state, 'overlap_screen')
                    flags, summary = screened(rows, hashes)
                    reference.private_npz(PRIVATE / 'flags.npz', flags)
                    protocol(pin); progress(state, 'publication')
                    r.write_json(OUT / 'aggregate.json', result(pin, summary))
                    r.write_json(OUT / 'manifest.json', manifest(pin)); authenticate(pin)
                elif args.action == 'audit':
                    rows, hashes, original = authenticate(pin, replay=True)
                    r.absent(AUDIT); AUDIT.mkdir(); state['owned'] = AUDIT
                    chosen = json.loads((PRIVATE / 'source_locator.json').read_text())
                    replay_rows = frozenset(reference.sampled(len(rows)))
                    def check(index, data):
                        r.require(hashlib.sha256(data).digest() == bytes(hashes['raw_sha256'][index]))
                        if index in replay_rows:
                            actual = decode.fingerprint(data)
                            r.require(all(np.array_equal(actual[key], hashes[key][index]) for key in hashes))
                        return True
                    with decode.PrivateArchive(chosen['archive']) as archive:
                        archive.map(check, [row['member'] for row in rows],
                                    lambda done, total: progress(state, 'audit_bytes', done, total))
                    r.require(r.equal(authenticate(pin, replay=True)[2], original))
                    progress(state, 'publication')
                    r.write_json(AUDIT / 'audit.json', audit_value(pin))
                    r.write_json(AUDIT / 'manifest.json', {'protocol_sha256': pin,
                                                          'audit_sha256': r.sha(AUDIT / 'audit.json')})
                    authenticate_audit(pin, r.sha(AUDIT / 'audit.json'))
                elif args.audit_sha256:
                    authenticate_audit(pin, args.audit_sha256)
                else:
                    authenticate(pin)
                answer.update(status='complete', protocol_sha256=pin, training_admitted=False)
                if args.action != 'prepare': answer['aggregate_sha256'] = r.sha(OUT / 'aggregate.json')
                if args.action == 'audit' or args.audit_sha256:
                    answer['audit_sha256'] = r.sha(AUDIT / 'audit.json')
        except Exception:
            answer['phase'] = state['phase']
            if state['owned'] is not None:
                try: r.write_json(state['owned'] / 'failure.json', answer)
                except Exception: pass
    print(json.dumps(answer, sort_keys=True))
    return int(answer['status'] != 'complete')


if __name__ == '__main__':
    raise SystemExit(main())
