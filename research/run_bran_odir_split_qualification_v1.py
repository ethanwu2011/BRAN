"""Local prospective source-patient partition/support qualification, no training."""
import argparse
import fcntl
import json

import numpy as np

import bran_odir_split_support_v1 as kernel
import run_bran_odir_content_qualification_v1 as content

r = content.r
ROOT = r.ROOT
PROTOCOL = ROOT / 'BRAN_ODIR_SPLIT_QUALIFICATION_PROTOCOL_V1.json'
OUT = ROOT / 'BRAN_ODIR_SPLIT_QUALIFICATION_V1'
AUDIT = ROOT / 'BRAN_ODIR_SPLIT_QUALIFICATION_AUDIT_V1'
PRIVATE = ROOT / 'private_artifacts/bran_odir_split_qualification_v1'
LOCK = r.LOCK
CONTENT_PIN = 'afd17d7e3b1daaa5a305e238c7361032faf5cb52d997ecf57e25c2705081f297'
CONTENT_AUDIT = '458e81841509463214413a783d44fff03bf3aa36bae3bb5cae67d9af386af84e'
SCHEMA = 'bran-odir-split-qualification-v1'
FILES = ('run_bran_odir_split_qualification_v1.py', 'bran_odir_split_support_v1.py',
         'test_run_bran_odir_split_qualification_v1.py', 'test_bran_odir_split_support_v1.py',
         'BRAN_ODIR_SPLIT_QUALIFICATION_DESIGN_V1.md')


def upstream():
    content.authenticate_audit(CONTENT_PIN, CONTENT_AUDIT)
    return {'content_protocol_sha256': CONTENT_PIN, 'content_audit_sha256': CONTENT_AUDIT,
            'content_manifest_sha256': r.sha(content.OUT / 'manifest.json')}


def group_inputs(selection, flags, records):
    """One retained source patient per row; preserve unknown labels, not zero fill."""
    content.screen.validate_flags(flags)
    retained = flags['retained_representative']
    r.require(len(retained) == len(selection))
    record_map = {row['patient_id']: row for row in records}
    r.require(len(record_map) == len(records))
    counts = {}
    for index, row in enumerate(selection):
        if retained[index]: counts[row['patient_id']] = counts.get(row['patient_id'], 0) + 1
    r.require(counts)
    patients = list(counts)
    value = {'patient_ids': np.asarray(patients, dtype=np.str_),
             'labels': np.asarray([record_map[patient]['labels'] for patient in patients], dtype=np.float64),
             'eye_counts': np.asarray([counts[patient] for patient in patients], dtype=np.int64)}
    validate_inputs(value)
    return value


def inputs():
    selection = json.loads((content.PRIVATE / 'selection.json').read_text())
    flags = content.reference.load_npz(content.PRIVATE / 'flags.npz')
    records = json.loads((content.metadata.PRIVATE / 'grouped_records.json').read_text())['grouped']['records']
    value = group_inputs(selection, flags, records)
    counts = json.loads((content.OUT / 'aggregate.json').read_text())['content_screen']['counts_rounded_down20']
    r.require(len(value['patient_ids']) // 20 * 20 == counts['retained_people']
              and int(value['eye_counts'].sum()) // 20 * 20 == counts['retained_images'])
    return value


def validate_inputs(value):
    r.require(type(value) is dict and set(value) == {'patient_ids', 'labels', 'eye_counts'})
    r.require(type(value['patient_ids']) is np.ndarray and value['patient_ids'].ndim == 1
              and value['patient_ids'].dtype.kind == 'U')
    kernel._validate_inputs(value['patient_ids'].tolist(), value['labels'], value['eye_counts'])


def evaluate(value):
    validate_inputs(value)
    return kernel.qualify(value['patient_ids'].tolist(), value['labels'], value['eye_counts'])


def template():
    return {'schema': SCHEMA, 'status': 'frozen_before_source_partition_qualification',
        'upstream': upstream(), 'code_sha256': {name: r.sha(ROOT / name) for name in FILES},
        'runtime': content.reference.runtime(), 'private_inputs_sha256': r.sha(PRIVATE / 'inputs.npz'),
        'parameters': {'assignment': 'sha256_prefix8_big_endian_mod10000',
            'prefix': 'bran-odir-source-split-v1\u0000', 'thresholds': [8000, 9000],
            'minimum_positive_and_negative_per_split': 20, 'label_order': list(kernel.LABELS),
            'require_train_validation_test_support': True, 'partition_retry_permitted': False,
            'group_unit': 'source_patient', 'unknown_is_negative': False},
        'official_split_authenticated': False, 'training_admitted': False}


def protocol(pin):
    r.regular(PROTOCOL); r.regular(PRIVATE / 'inputs.npz', private=True)
    r.require(type(pin) is str and len(pin) == 64 and r.sha(PROTOCOL) == pin)
    value = json.loads(PROTOCOL.read_text()); r.require(r.equal(value, template()))
    return value


def result(pin, aggregate):
    kernel.validate_aggregate(aggregate)
    return {'schema': SCHEMA, 'status': 'split_qualification_completed', 'protocol_sha256': pin,
            'qualification': aggregate, 'source_local_patient_partition_fixed': True,
            'official_split_authenticated': False, 'original_release_authenticated': False,
            'reuse_terms_authenticated': False, 'training_admitted': False,
            'patient_level_output_emitted': False}


def manifest(pin):
    return {'protocol_sha256': pin, 'aggregate_sha256': r.sha(OUT / 'aggregate.json'),
            'private_assignment_sha256': r.sha(PRIVATE / 'assignment.npz')}


def authenticate(pin):
    protocol(pin)
    r.inventory(OUT, ('aggregate.json', 'manifest.json'))
    r.inventory(PRIVATE, ('inputs.npz', 'assignment.npz'), private=True)
    r.require(not (AUDIT / 'failure.json').exists())
    r.require(r.equal(json.loads((OUT / 'manifest.json').read_text()), manifest(pin)))
    split, aggregate = evaluate(content.reference.load_npz(PRIVATE / 'inputs.npz'))
    actual = content.reference.load_npz(PRIVATE / 'assignment.npz')
    r.require(set(actual) == {'split'} and actual['split'].dtype == np.uint8
              and np.array_equal(split, actual['split']))
    value = result(pin, aggregate)
    r.require(r.equal(json.loads((OUT / 'aggregate.json').read_text()), value))
    return value


def audit_value(pin):
    return {'schema': SCHEMA, 'status': 'authenticated', 'protocol_sha256': pin,
            'terminal_manifest_sha256': r.sha(OUT / 'manifest.json'),
            'retained_source_group_inputs_rebuilt': True, 'assignment_and_support_replayed': True,
            'official_split_authenticated': False, 'training_admitted': False,
            'patient_level_output_emitted': False}


def authenticate_audit(pin, audit_pin):
    authenticate(pin); r.inventory(AUDIT, ('audit.json', 'manifest.json'))
    r.require(r.sha(AUDIT / 'audit.json') == audit_pin and
              r.equal(json.loads((AUDIT / 'audit.json').read_text()), audit_value(pin)))
    r.require(r.equal(json.loads((AUDIT / 'manifest.json').read_text()),
                      {'protocol_sha256': pin, 'audit_sha256': audit_pin}))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'audit', 'verify'))
    parser.add_argument('--protocol-sha256'); parser.add_argument('--audit-sha256')
    args = parser.parse_args(argv)
    answer = {'status': 'failed', 'action': args.action, 'patient_level_output_emitted': False}
    owned = None
    with r.quiet():
        try:
            with LOCK.open('a') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                pin = args.protocol_sha256
                if args.action == 'prepare':
                    upstream(); r.absent(PROTOCOL, OUT, AUDIT, PRIVATE)
                    value = inputs(); validate_inputs(value)
                    PRIVATE.mkdir(mode=0o700); content.reference.private_npz(PRIVATE / 'inputs.npz', value)
                    r.write_json(PROTOCOL, template()); pin = r.sha(PROTOCOL); protocol(pin)
                elif args.action == 'run':
                    protocol(pin); r.absent(OUT, AUDIT)
                    r.inventory(PRIVATE, ('inputs.npz',), private=True)
                    OUT.mkdir(); owned = OUT
                    split, aggregate = evaluate(content.reference.load_npz(PRIVATE / 'inputs.npz'))
                    content.reference.private_npz(PRIVATE / 'assignment.npz', {'split': split})
                    protocol(pin); r.write_json(OUT / 'aggregate.json', result(pin, aggregate))
                    r.write_json(OUT / 'manifest.json', manifest(pin)); authenticate(pin)
                elif args.action == 'audit':
                    original = authenticate(pin); r.absent(AUDIT); AUDIT.mkdir(); owned = AUDIT
                    rebuilt = inputs(); actual = content.reference.load_npz(PRIVATE / 'inputs.npz')
                    r.require(set(rebuilt) == set(actual) and all(np.array_equal(rebuilt[key], actual[key],
                              equal_nan=rebuilt[key].dtype.kind == 'f') for key in rebuilt))
                    split, aggregate = evaluate(rebuilt)
                    r.require(r.equal(original, result(pin, aggregate)) and r.equal(authenticate(pin), original))
                    r.write_json(AUDIT / 'audit.json', audit_value(pin))
                    r.write_json(AUDIT / 'manifest.json', {'protocol_sha256': pin,
                                                          'audit_sha256': r.sha(AUDIT / 'audit.json')})
                    authenticate_audit(pin, r.sha(AUDIT / 'audit.json'))
                elif args.audit_sha256: authenticate_audit(pin, args.audit_sha256)
                else: authenticate(pin)
                answer.update(status='complete', protocol_sha256=pin, training_admitted=False)
                if args.action != 'prepare': answer['aggregate_sha256'] = r.sha(OUT / 'aggregate.json')
                if args.action == 'audit' or args.audit_sha256:
                    answer['audit_sha256'] = r.sha(AUDIT / 'audit.json')
        except Exception:
            if owned is not None:
                try: r.write_json(owned / 'failure.json', answer)
                except Exception: pass
    print(json.dumps(answer, sort_keys=True))
    return int(answer['status'] != 'complete')


if __name__ == '__main__': raise SystemExit(main())
