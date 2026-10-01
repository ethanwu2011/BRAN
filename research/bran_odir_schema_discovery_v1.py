"""Read-only, closed header discovery. Never emit names, headers or records."""
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import zipfile

import run_bran_retinal_group_readiness_v1 as r

ROOTS = ('/Volumes/Extreme/cwm-data', '/Users/ethanwu/Downloads')
OUT = r.ROOT / 'BRAN_ODIR_SCHEMA_DISCOVERY_V1'
FIELDS = {'patient_id': ('ID',), 'age': ('Patient Age',), 'sex': ('Patient Sex',),
          'left_image': ('Left-Fundus',), 'right_image': ('Right-Fundus',),
          'left_diagnostic_keywords': ('Left-Diagnostic Keywords',),
          'right_diagnostic_keywords': ('Right-Diagnostic Keywords',),
          'eight_patient_label_columns': tuple('NDGCAHMO')}


def header_profile(raw):
    r.require(type(raw) is bytes and len(raw) <= 65536)
    header = next(csv.reader([raw.decode('utf-8-sig')], strict=True))
    r.require(len(header) == len(set(header)))
    value = {'expected_fields_present': {key: set(fields).issubset(header) for key, fields in FIELDS.items()},
             'record_rows_parsed': False, 'patient_level_output_emitted': False,
             'original_release_authenticated': False, 'training_admitted': False}
    validate_profile(value)
    return value


def validate_profile(value):
    r.require(type(value) is dict and set(value) == {'expected_fields_present', 'record_rows_parsed',
              'patient_level_output_emitted', 'original_release_authenticated', 'training_admitted'})
    flags = value['expected_fields_present']
    r.require(type(flags) is dict and set(flags) == set(FIELDS) and all(type(v) is bool for v in flags.values()))
    r.require(all(value[key] is False for key in set(value) - {'expected_fields_present'}))


def discover():
    candidates = []
    for root in ROOTS:
        if not Path(root).is_dir(): continue
        result = subprocess.run(['rg', '--files', '--hidden', '-g', '*.zip', root], capture_output=True)
        r.require(result.returncode in (0, 1))
        candidates.extend(Path(raw) for raw in result.stdout.decode().splitlines() if 'odir' in raw.lower())
    return sorted(set(candidates))


def evaluate():
    profiles = []
    for path in discover():
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir() or Path(info.filename).suffix.lower() != '.csv': continue
                with archive.open(info) as stream:
                    profiles.append(header_profile(stream.readline(65537)))
    # This is structural discovery, not an authenticated original-file binding.
    return {'schema': 'bran-odir-header-discovery-v1', 'status': 'completed',
            'csv_header_profiles': profiles, 'patient_level_output_emitted': False,
            'training_admitted': False, 'row_linkage_checked': False}


def main():
    answer = {'status': 'failed', 'patient_level_output_emitted': False}
    with r.quiet():
        try:
            with r.LOCK.open('a') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                r.absent(OUT)
                result = evaluate()
                for profile in result['csv_header_profiles']: validate_profile(profile)
                OUT.mkdir()
                r.write_json(OUT / 'aggregate.json', result)
                r.write_json(OUT / 'manifest.json', {'aggregate_sha256': r.sha(OUT / 'aggregate.json'),
                             'code_sha256': r.sha(Path(__file__)),
                             'test_sha256': r.sha(r.ROOT / 'test_bran_odir_schema_discovery_v1.py'),
                             'source_authenticated': False})
                answer = result
        except Exception: pass
    print(json.dumps(answer, sort_keys=True))
    return int(answer['status'] != 'completed')


if __name__ == '__main__': raise SystemExit(main())
