"""Source-bound, FD-quiet ODIR qualification; no hosted patient data."""
import argparse
import csv
from dataclasses import asdict
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import zipfile

import bran_odir_schema_discovery_v1 as locator
import bran_odir_metadata_kernel_v1 as k
import run_bran_retinal_group_readiness_v1 as r

ROOT = r.ROOT
PROTOCOL = ROOT / 'BRAN_ODIR_METADATA_QUALIFICATION_PROTOCOL_V1.json'
OUT = ROOT / 'BRAN_ODIR_METADATA_QUALIFICATION_V1'
AUDIT = ROOT / 'BRAN_ODIR_METADATA_QUALIFICATION_AUDIT_V1'
PRIVATE = ROOT / 'private_artifacts/bran_odir_metadata_qualification_v1'
LOCK = r.LOCK
SCHEMA = 'bran-odir-metadata-qualification-v1'
LIMIT = 64 * 1024 * 1024
FIELDS = ('ID', 'Patient Age', 'Patient Sex', 'Left-Fundus', 'Right-Fundus', *tuple('NDGCAHMO'))
IMAGES = ('.jpg', '.jpeg', '.png')
FILES = ('run_bran_odir_metadata_qualification_v1.py', 'bran_odir_metadata_kernel_v1.py',
         'test_run_bran_odir_metadata_qualification_v1.py', 'test_bran_odir_metadata_kernel_v1.py',
         'BRAN_ODIR_METADATA_QUALIFICATION_DESIGN_V1.md', 'bran_odir_schema_discovery_v1.py',
         'run_bran_retinal_group_readiness_v1.py', 'bran_retinal_group_readiness_v1.py')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def directory_entries(archive):
    infos = archive.infolist()
    r.require(len(infos) <= 1000000)
    names = [x.filename for x in infos]
    r.require(len(names) == len(set(names)) and all(name and '\\' not in name and '\x00' not in name
              and not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts for name in names))
    entries = sorted((x.filename, x.file_size, x.CRC) for x in infos)
    members = sorted(x.filename for x in infos if not x.is_dir() and PurePosixPath(x.filename).suffix.lower() in IMAGES)
    return entries, members


def source(chosen):
    r.require(type(chosen) is dict and set(chosen) == {'archive', 'csv_member'})
    path = Path(chosen['archive']); r.regular(path)
    before = path.stat()
    with zipfile.ZipFile(path) as archive:
        entries, members = directory_entries(archive)
        info = archive.getinfo(chosen['csv_member'])
        r.require(not info.is_dir() and info.file_size <= LIMIT and PurePosixPath(info.filename).suffix.lower() == '.csv')
        with archive.open(info) as stream: raw = stream.read(LIMIT + 1)
    after = path.stat()
    r.require(len(raw) <= LIMIT and (before.st_size, before.st_mtime_ns, before.st_ino)
              == (after.st_size, after.st_mtime_ns, after.st_ino))
    header = next(csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline=''), strict=True))
    r.require(len(header) == len(set(header)) and set(FIELDS).issubset(header))
    receipt = {'csv_bytes_sha256': digest(raw),
               'zip_directory_metadata_sha256': digest(json.dumps(entries, separators=(',', ':')).encode()),
               'csv_bytes_authenticated': True, 'image_payload_authenticated': False,
               'original_release_authenticated': False}
    return raw, members, receipt


def choose_source():
    found = []
    for path in locator.discover():
        with zipfile.ZipFile(path) as archive:
            directory_entries(archive)
            for info in archive.infolist():
                if info.is_dir() or PurePosixPath(info.filename).suffix.lower() != '.csv': continue
                with archive.open(info) as stream: first = stream.readline(65537)
                if all(locator.header_profile(first)['expected_fields_present'].values()):
                    found.append({'archive': str(path), 'csv_member': info.filename})
    r.require(len(found) == 1)
    return found[0]


def private_json(path, value):
    with Path(path).open('x') as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def template(receipt):
    return {'schema': SCHEMA, 'status': 'frozen_before_record_qualification',
            'source': receipt, 'private_locator_sha256': r.sha(PRIVATE / 'source_locator.json'),
            'code_sha256': {name: r.sha(ROOT / name) for name in FILES}, 'runtime': r.runtime(),
            'adult_age_minimum': 18, 'counts': 'rounded_down20_zero_not_proven_absence',
            'training_admitted': False, 'official_split_authenticated': False}


def protocol(pin):
    r.regular(PROTOCOL); r.regular(PRIVATE / 'source_locator.json', private=True)
    r.require(type(pin) is str and len(pin) == 64 and r.sha(PROTOCOL) == pin)
    value = json.loads(PROTOCOL.read_text())
    r.require(r.equal(value, template(value['source'])))
    return value


def evaluate(pin):
    p = protocol(pin)
    chosen = json.loads((PRIVATE / 'source_locator.json').read_text())
    raw, members, receipt = source(chosen)
    r.require(r.equal(receipt, p['source']))
    reader = csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline=''), strict=True)
    header = next(reader); indexes = [header.index(field) for field in FIELDS]
    rows = []
    for row in reader:
        r.require(len(row) == len(header) and len(rows) < 100000)
        rows.append({key: row[index] for key, index in zip(FIELDS, indexes)})
    private, aggregate = k.qualify(rows, members)
    k.validate_aggregate(aggregate)
    r.require(aggregate['counts_rounded_down20']['grouped_patients'] >= 20)
    return asdict(private), aggregate


def result(pin, aggregate):
    k.validate_aggregate(aggregate)
    return {'schema': SCHEMA, 'status': 'metadata_qualification_completed', 'protocol_sha256': pin,
            'qualification': aggregate, 'private_records_sha256': r.sha(PRIVATE / 'grouped_records.json'),
            'image_payload_authenticated': False, 'official_split_authenticated': False,
            'training_admitted': False, 'patient_level_output_emitted': False}


def authenticate(pin):
    protocol(pin)
    r.inventory(OUT, ('aggregate.json', 'manifest.json'))
    r.inventory(PRIVATE, ('source_locator.json', 'grouped_records.json'), private=True)
    value = json.loads((OUT / 'aggregate.json').read_text())
    r.require(r.equal(value, result(pin, value['qualification'])))
    r.require(r.equal(json.loads((OUT / 'manifest.json').read_text()),
              {'protocol_sha256': pin, 'aggregate_sha256': r.sha(OUT / 'aggregate.json')}))
    return value


def audit_value(pin):
    return {'schema': SCHEMA, 'status': 'authenticated', 'protocol_sha256': pin,
            'aggregate_sha256': r.sha(OUT / 'aggregate.json'), 'record_qualification_replayed': True,
            'csv_bytes_authenticated': True, 'zip_directory_metadata_authenticated': True,
            'image_payload_authenticated': False, 'original_release_authenticated': False,
            'training_admitted': False, 'patient_level_output_emitted': False}


def authenticate_audit(pin, audit_pin):
    authenticate(pin); r.inventory(AUDIT, ('audit.json', 'manifest.json'))
    r.require(r.sha(AUDIT / 'audit.json') == audit_pin and r.equal(
              json.loads((AUDIT / 'audit.json').read_text()), audit_value(pin)))
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
                    r.absent(PROTOCOL, OUT, AUDIT, PRIVATE)
                    chosen = choose_source(); _, _, receipt = source(chosen)
                    PRIVATE.mkdir(mode=0o700)
                    private_json(PRIVATE / 'source_locator.json', chosen)
                    r.write_json(PROTOCOL, template(receipt)); pin = r.sha(PROTOCOL)
                elif args.action == 'run':
                    protocol(pin); r.absent(OUT, AUDIT, PRIVATE / 'grouped_records.json')
                    OUT.mkdir(); owned = OUT
                    private, aggregate = evaluate(pin); protocol(pin)
                    private_json(PRIVATE / 'grouped_records.json', private)
                    r.write_json(OUT / 'aggregate.json', result(pin, aggregate))
                    r.write_json(OUT / 'manifest.json', {'protocol_sha256': pin,
                                 'aggregate_sha256': r.sha(OUT / 'aggregate.json')})
                    authenticate(pin)
                elif args.action == 'audit':
                    original = authenticate(pin); r.absent(AUDIT); AUDIT.mkdir(); owned = AUDIT
                    private, aggregate = evaluate(pin)
                    r.require(r.equal(json.loads((PRIVATE / 'grouped_records.json').read_text()), private)
                              and r.equal(result(pin, aggregate), original) and r.equal(authenticate(pin), original))
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
