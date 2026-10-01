"""Non-patient ICD vocabulary binding for prespecified prior-disease cohorts.

Only d_icd_diagnoses (code/version/title) is read here, never diagnoses_icd rows.
The combined diabetes family explicitly retains ICD9's unspecified-type caveat.
"""
import argparse
import json
from pathlib import Path
import re

from bran_clinical_source_reader_v1 import iter_projected_csv
from bran_clinical_dictionary_binding_v1 import _quiet
from bran_joint_lab_cache_v1 import valid_sha
from bran_mimic_prior_disease_v1 import FAMILIES, canonical_icd
from run_bran_source_linkage_audit_v1 import sha, exclusive_json

ROOT = Path(__file__).resolve().parent
DICTIONARY = Path('/Users/ethanwu/mimiciv-3.1/hosp/d_icd_diagnoses.csv.gz')
OUT = ROOT / 'BRAN_MIMIC_PRIOR_DISEASE_DICTIONARY_V1.json'
AUDIT = ROOT / 'BRAN_MIMIC_PRIOR_DISEASE_DICTIONARY_AUDIT_V1.json'
CODE = ('bran_mimic_prior_disease_dictionary_v1.py', 'test_bran_mimic_prior_disease_dictionary_v1.py',
        'bran_mimic_prior_disease_v1.py', 'test_bran_mimic_prior_disease_v1.py',
        'bran_clinical_source_reader_v1.py', 'bran_cbc_event_adapter_v1.py',
        'bran_clinical_snapshot_v1.py', 'bran_clinical_semantics_v1.py',
        'bran_clinical_chemistry_semantics_v1.py', 'bran_joint_lab_cache_v1.py',
        'bran_clinical_dictionary_binding_v1.py', 'run_bran_source_linkage_audit_v1.py')
POLICY = {
    'diabetes': 'ICD9_250xx_last_digit0_or2_typeII_or_unspecified_OR_ICD10_E11',
    'heart_failure': 'ICD9_428_family_OR_ICD10_I50_family',
    'chronic_kidney_disease': 'ICD9_585_family_OR_ICD10_N18_family',
    'dictionary_intersection_required': True,
    'retrospective_administrative_phenotype': True,
    'clinically_adjudicated_diagnosis': False,
    'absence_of_code_proves_absence_of_disease': False,
}


def require(ok):
    if not ok:
        raise ValueError('mimic_prior_disease_dictionary_contract_failed')


def family_for_code(version, code):
    require(canonical_icd(version, code) == (version, code))
    if (version == '9' and re.fullmatch(r'250[0-9][02]', code)) or (version == '10' and code.startswith('E11')):
        return FAMILIES[0]
    if (version == '9' and re.fullmatch(r'428[0-9]{0,2}', code)) or (version == '10' and code.startswith('I50')):
        return FAMILIES[1]
    if (version == '9' and re.fullmatch(r'585[0-9]{0,2}', code)) or (version == '10' and code.startswith('N18')):
        return FAMILIES[2]
    return None


def title_agrees(version, family, title):
    require(type(title) is str and title.strip() and not any(ord(c) < 32 for c in title))
    text = title.casefold()
    if family == FAMILIES[0]:
        return ('diabetes' in text and 'type ii or unspecified type' in text) if version == '9' else 'type 2 diabetes mellitus' in text
    if family == FAMILIES[1]:
        return 'failure' in text
    if family == FAMILIES[2]:
        return ('kidney' in text or 'renal' in text) and ('disease' in text or 'failure' in text)
    return False


def bind_dictionary(rows):
    """Return only codes whose prespecified family agrees with source vocabulary."""
    seen, approved = set(), {}
    iterator = iter(rows)
    try:
        for row in iterator:
            require(type(row) is dict and set(row) == {'icd_code', 'icd_version', 'long_title'})
            key = canonical_icd(row['icd_version'], row['icd_code'])
            require(key is not None and key not in seen)
            seen.add(key)
            family = family_for_code(*key)
            if family is not None:
                require(title_agrees(key[0], family, row['long_title']))
                approved[key] = family
    finally:
        close = getattr(iterator, 'close', None)
        if close is not None:
            close()
    require(set(approved.values()) == set(FAMILIES))
    # Both coding eras must be explicitly represented in every family.
    require(all(any(v == version and f == family for (v, _), f in approved.items())
                for family in FAMILIES for version in ('9', '10')))
    return approved


def build_receipt():
    before = sha(DICTIONARY)
    approved = bind_dictionary(iter_projected_csv(DICTIONARY,
        ('icd_code', 'icd_version', 'long_title'), max_rows=250000))
    require(sha(DICTIONARY) == before)
    result = {'schema': 'bran-mimic-prior-disease-dictionary-v1', 'status': 'nonpatient_vocabulary_bound',
        'dictionary_sha256': before, 'code_sha256': {name: sha(ROOT / name) for name in CODE},
        'policy': POLICY, 'family_order': list(FAMILIES),
        'approved_codes': [{'icd_version': version, 'icd_code': code, 'family': family}
                           for (version, code), family in sorted(approved.items())],
        'nonpatient_code_counts': {family: {version: sum(v == version and f == family for (v, _), f in approved.items())
                                           for version in ('9', '10')} for family in FAMILIES},
        'patient_source_read': False, 'patient_counts_computed': False,
        'disease_cohort_coverage_established': False}
    validate_receipt(result)
    return result


def validate_receipt(result):
    require(type(result) is dict and set(result) == {'schema', 'status', 'dictionary_sha256',
        'code_sha256', 'policy', 'family_order', 'approved_codes', 'nonpatient_code_counts',
        'patient_source_read', 'patient_counts_computed', 'disease_cohort_coverage_established'})
    require(result['schema'] == 'bran-mimic-prior-disease-dictionary-v1'
            and result['status'] == 'nonpatient_vocabulary_bound' and valid_sha(result['dictionary_sha256']))
    require(result['family_order'] == list(FAMILIES) and result['policy'] == POLICY)
    for key in ('patient_source_read', 'patient_counts_computed', 'disease_cohort_coverage_established'):
        require(result[key] is False)
    require(type(result['code_sha256']) is dict and set(result['code_sha256']) == set(CODE)
            and all(valid_sha(value) for value in result['code_sha256'].values()))
    require(type(result['approved_codes']) is list and len(result['approved_codes']) > 0)
    approved = {}
    for row in result['approved_codes']:
        require(type(row) is dict and set(row) == {'icd_version', 'icd_code', 'family'})
        key = (row['icd_version'], row['icd_code'])
        require(key not in approved and canonical_icd(*key) == key)
        require(row['family'] in FAMILIES and family_for_code(*key) == row['family'])
        approved[key] = row['family']
    require(list(approved) == sorted(approved))
    counts = result['nonpatient_code_counts']
    require(type(counts) is dict and set(counts) == set(FAMILIES))
    for family in FAMILIES:
        require(type(counts[family]) is dict and set(counts[family]) == {'9', '10'})
        for version in ('9', '10'):
            value = counts[family][version]
            require(type(value) is int and value > 0
                    and value == sum(v == version and f == family for (v, _), f in approved.items()))


def authenticate_receipt():
    """Full vocabulary/code replay; no patient records involved."""
    original = OUT.read_bytes()
    result = json.loads(original); validate_receipt(result)
    require(result == build_receipt())
    require(OUT.read_bytes() == original)
    return {(row['icd_version'], row['icd_code']): row['family'] for row in result['approved_codes']}


def validate_audit(result, receipt_pin):
    require(type(result) is dict and set(result) == {'schema', 'status', 'receipt_sha256',
        'dictionary_sha256', 'full_vocabulary_replay_equal', 'patient_source_read'})
    require(result['schema'] == 'bran-mimic-prior-disease-dictionary-audit-v1'
            and result['status'] == 'passed' and result['receipt_sha256'] == receipt_pin)
    require(valid_sha(receipt_pin) and valid_sha(result['dictionary_sha256']))
    require(result['full_vocabulary_replay_equal'] is True and result['patient_source_read'] is False)


def authenticate_audit():
    """Authenticate the independent replay and unchanged receipt/source/code."""
    original = AUDIT.read_bytes()
    receipt_pin = sha(OUT)
    approved = authenticate_receipt()
    result = json.loads(original); validate_audit(result, receipt_pin)
    receipt = json.loads(OUT.read_bytes())
    require(result['dictionary_sha256'] == receipt['dictionary_sha256'] == sha(DICTIONARY))
    require(sha(OUT) == receipt_pin and AUDIT.read_bytes() == original)
    return approved


def main(argv=None):
    parser = argparse.ArgumentParser(); modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--build', action='store_true'); modes.add_argument('--audit', action='store_true')
    args = parser.parse_args(argv)
    result = {'status': 'nonpatient_dictionary_operation_failed', 'patient_source_read': False}
    with _quiet():
        try:
            if args.build:
                require(not OUT.exists() and not AUDIT.exists())
                receipt = build_receipt()
                exclusive_json(OUT, receipt)
                result = {'status': 'nonpatient_dictionary_bound', 'patient_source_read': False,
                    'receipt_sha256': sha(OUT), 'nonpatient_code_counts': receipt['nonpatient_code_counts']}
            else:
                require(not AUDIT.exists())
                receipt_pin = sha(OUT)
                authenticate_receipt()
                require(sha(OUT) == receipt_pin)
                dictionary_pin = json.loads(OUT.read_bytes())['dictionary_sha256']
                require(sha(DICTIONARY) == dictionary_pin)
                audit = {'schema': 'bran-mimic-prior-disease-dictionary-audit-v1', 'status': 'passed',
                    'receipt_sha256': receipt_pin, 'dictionary_sha256': dictionary_pin,
                    'full_vocabulary_replay_equal': True, 'patient_source_read': False}
                validate_audit(audit, receipt_pin); exclusive_json(AUDIT, audit)
                result = {'status': 'nonpatient_dictionary_audited', 'patient_source_read': False,
                          'receipt_sha256': receipt_pin, 'audit_sha256': sha(AUDIT)}
        except Exception:
            pass
    print(json.dumps(result))
    return 1 if result['status'] == 'nonpatient_dictionary_operation_failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
