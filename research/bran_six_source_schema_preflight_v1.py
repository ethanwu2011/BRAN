"""Local CSV-header/presence audit, never a patient-row or training loader.

This establishes structural prerequisites only. No schema result certifies
patient identity, unit conversions, contemporaneity, observations or permission.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
OUT = 'BRAN_SIX_SOURCE_SCHEMA_PREFLIGHT_V1'
MAX_HEADER = 65536
# Paths are taken from the existing extractors; missing paths are reported, not
# searched recursively. Column names below are public schema names, not values.
SOURCES = {
    'mimic': {
        'root': '/Users/ethanwu/mimiciv-3.1/hosp',
        'tables': {
            'patients.csv.gz': {'person_key': ['subject_id'], 'age': ['anchor_age'], 'age_year': ['anchor_year']},
            'admissions.csv.gz': {'person_key': ['subject_id'], 'encounter_key': ['hadm_id'], 'index_time': ['admittime']},
            'labevents.csv.gz': {'person_key': ['subject_id'], 'encounter_key': ['hadm_id'], 'test_code': ['itemid'], 'time': ['charttime'], 'numeric_value': ['valuenum'], 'unit': ['valueuom']},
            'd_labitems.csv.gz': {'test_code': ['itemid'], 'test_name': ['label'], 'specimen': ['fluid']},
        },
    },
    'eicu': {
        'root': '/Users/ethanwu/eicu-crd-2.0',
        'tables': {
            'patient.csv.gz': {'stay_key': ['patientunitstayid'], 'person_key': ['uniquepid'], 'encounter_key': ['patienthealthsystemstayid'], 'age': ['age']},
            'lab.csv.gz': {'stay_key': ['patientunitstayid'], 'test_name': ['labname'], 'numeric_value': ['labresult'], 'time_offset': ['labresultoffset'], 'unit': ['labmeasurenamesystem']},
        },
    },
    'nwicu': {
        'root': '/Users/ethanwu/nwicu-northwestern-icu-0.1.0/data/nw_hosp',
        'tables': {
            'patients.csv.gz': {'person_key': ['subject_id'], 'age_candidate': ['anchor_age', 'age']},
            'admissions.csv.gz': {'person_key': ['subject_id'], 'encounter_key': ['hadm_id'], 'index_time': ['admittime']},
            'labevents.csv.gz': {'person_key': ['subject_id'], 'encounter_key': ['hadm_id'], 'test_code': ['itemid'], 'time': ['charttime'], 'numeric_value': ['valuenum'], 'unit': ['valueuom']},
            'd_labitems.csv.gz': {'test_code': ['itemid'], 'test_name': ['label'], 'specimen': ['fluid']},
        },
    },
    'sicdb': {
        'root': '/Users/ethanwu/sicdb-1.0.8',
        'tables': {
            'cases.csv.gz': {'case_key': ['CaseID'], 'person_key': ['PatientID'], 'age': ['AgeOnAdmission']},
            'laboratory.csv.gz': {'case_key': ['CaseID'], 'test_code': ['LaboratoryID'], 'numeric_value': ['LaboratoryValue'], 'time_offset': ['Offset']},
            'd_references.csv.gz': {'test_code': ['ReferenceGlobalID'], 'test_name': ['ReferenceValue'], 'unit': ['ReferenceUnit']},
        },
    },
    'zigong': {
        'root': '/tmp/zigong/DataTables',
        'tables': {'dtLab.csv': {'local_record_key': ['INP_NO'], 'test_name': ['Item'], 'numeric_value': ['LabValue']}},
    },
    'nhanes': {
        'root': '/Users/ethanwu/nhanes-oculomics',
        'tables': {'CBC_D.xpt': {}, 'CBC_E.xpt': {}, 'DEMO_D.xpt': {}, 'DEMO_E.xpt': {}},
    },
}


def check_header(line, expected_roles):
    if not isinstance(line, str) or not line or len(line) >= MAX_HEADER:
        raise ValueError('header_invalid')
    fields = next(csv.reader([line.removeprefix('\ufeff').rstrip('\r\n')]))
    normalized = [s.strip().casefold() for s in fields]
    if (not normalized or len(set(normalized)) != len(normalized)
            or any(not re.fullmatch(r'[a-z_][a-z0-9_]*', s) for s in normalized)):
        raise ValueError('header_invalid')
    roles = {role: any(name.casefold() in normalized for name in alternatives)
             for role, alternatives in expected_roles.items()}
    return {'status': 'header_checked', 'expected_role_columns_present': roles,
            'all_expected_role_columns_present': all(roles.values()),
            'patient_rows_read': False}


def inspect_table(path, roles):
    path = Path(path)
    if not path.is_file():
        return {'status': 'not_present_at_declared_path', 'patient_rows_read': False}
    if path.suffix.lower() == '.xpt':
        return {'status': 'present_xpt_header_not_parsed', 'patient_rows_read': False}
    try:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8', newline='') as handle:
            line = handle.readline(MAX_HEADER)
        return check_header(line, roles)
    except Exception:
        # No row, unknown header name, path, or library exception is emitted.
        return {'status': 'header_unreadable_or_invalid', 'patient_rows_read': False}


def summarize_source(tables):
    checked = [v for v in tables.values() if v['status'] == 'header_checked']
    return {
        'tables': tables,
        'all_requested_headers_match': (len(checked) == len(tables)
                                        and all(v['all_expected_role_columns_present'] for v in checked)),
        'readiness_scope': 'presence_and_declared_header_columns_only',
        'ready_for_real_data_training': False,
        'actual_unit_values_checked': False,
        'person_linkage_or_timing_validated': False,
    }


def build(root=ROOT):
    root = Path(root)
    out = root / OUT
    if out.exists():
        raise ValueError('output_already_exists')
    # Consume only the prior closed metadata release, never its source tables.
    prior = root / 'BRAN_DATA_EXPANSION_READINESS_V1'
    manifest = json.loads((prior / 'manifest.json').read_text())
    raw = (prior / 'metadata.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['files']['metadata.json']:
        raise ValueError('prior_metadata_changed')
    metadata = json.loads(raw)
    if metadata['schema'] != 'bran-data-expansion-metadata-v1':
        raise ValueError('prior_schema_invalid')
    result = {
        'schema': 'bran-six-source-schema-preflight-v1',
        'status': 'completed_headers_only_no_training',
        'prior_metadata_sha256': hashlib.sha256(raw).hexdigest(),
        'sources': {name: summarize_source({table: inspect_table(Path(spec['root']) / table, roles)
                    for table, roles in spec['tables'].items()}) for name, spec in SOURCES.items()},
        'all_six_sources_in_scope': True,
        'patient_rows_images_predictions_or_embeddings_read': False,
        'legacy_caches_modified': False,
        'model_training_launched': False,
    }
    out.mkdir()
    report = out / 'schema.json'
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    bindings = {'schema': 'bran-six-source-schema-manifest-v1',
                'files': {'schema.json': hashlib.sha256(report.read_bytes()).hexdigest()},
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'prior_manifest_sha256': hashlib.sha256((prior / 'manifest.json').read_bytes()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(bindings, indent=2, sort_keys=True) + '\n')
    return {'status': result['status'], 'sources': {k: {
        'all_requested_headers_match': v['all_requested_headers_match'],
        'table_statuses': {t: r['status'] for t, r in v['tables'].items()}}
        for k, v in result['sources'].items()}}


if __name__ == '__main__':
    try:
        print(json.dumps(build(), sort_keys=True))
    except Exception:
        print('{"status":"schema_preflight_failed_without_disclosure"}')
        sys.exit(1)
