"""New private source-key/time cache; no model fit or subtype claim.

Prepare/run/audit are heavy local operations and must wait for the shared heavy
job slot. All source access is inside the FD-silenced CLI. No old cache is joined
by row order or changed. Real execution requires a frozen protocol hash.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from bran_clinical_semantics_v1 import current_v2_age_scalar
from bran_clinical_source_reader_v1 import iter_projected_csv
from bran_joint_lab_cache_v1 import coarse_count, valid_sha
from bran_mimic_linked_snapshot_contract_v1 import pack_linked_snapshots, validate_linked_snapshot_pack
import bran_mimic_landmark_source_v1 as source
import bran_mimic_prior_disease_v1 as disease
import bran_mimic_prior_disease_dictionary_v1 as disease_dictionary
import run_bran_mimic_joint_labs_v1 as parent
from run_bran_source_linkage_audit_v1 import sha, exclusive_json
from run_bran_cbc_reference_preflight_v1 import publish

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'BRAN_MIMIC_LANDMARK_LINKAGE_PROTOCOL_V1.json'
OUT = ROOT / 'BRAN_MIMIC_LANDMARK_LINKAGE_V1'
AUDIT = ROOT / 'BRAN_MIMIC_LANDMARK_LINKAGE_AUDIT_V1'
PRIVATE = ROOT / 'private_artifacts/bran_mimic_landmark_linkage_v1'
LOCK = Path('/private/tmp/bran_retinal_extraction_v1.lock')
SALT = parent.SALT
INPUTS = {key: parent.INPUTS['mimic_' + key] for key in ('patients', 'admissions', 'labs', 'dictionary')}
PHENOTYPE_INPUTS = {
    'diagnoses': Path('/Users/ethanwu/mimiciv-3.1/hosp/diagnoses_icd.csv.gz'),
    'dictionary': disease_dictionary.DICTIONARY,
}
CODE = tuple(sorted(set(parent.CODE) | set(disease_dictionary.CODE) | {
    'bran_mimic_linked_snapshot_contract_v1.py', 'test_bran_mimic_linked_snapshot_contract_v1.py',
    'bran_mimic_landmark_source_v1.py', 'test_bran_mimic_landmark_source_v1.py',
    'run_bran_mimic_landmark_linkage_v1.py', 'test_run_bran_mimic_landmark_linkage_v1.py',
    'run_bran_cbc_reference_preflight_v1.py', 'BRAN_MIMIC_LANDMARK_LINKAGE_DESIGN_V1.md',
}))
FILES = ('observations.npz', 'row_map.json', 'outcomes.npz', 'prior_diseases.npz')
PHASES = ('source_authentication', 'source_reconstruction', 'private_write', 'source_replay', 'publication')
POLICY = {
    'source': 'original_MIMIC_IV_3_1_hospital_admissions_not_ICU_only',
    'landmark_minutes': 1440., 'snapshot_window_minutes': 60., 'minimum_observed_CBC': 2,
    'availability': 'charttime_and_storetime_required_0_le_chart_le_store_le_landmark',
    'snapshot': 'earliest_available_CBC_anchor_then_earliest_fields_in_next60min_capped_at24h',
    'duplicate_ties': 'conflicting_earliest_values_missing_identical_ties_latest_eligible_storetime',
    'mortality_endpoint': 'recorded_index_admission_death_strictly_after_24h_before_or_at_discharge',
    'mortality_negative': 'expire_flag0_empty_deathtime_discharge_strictly_after24h',
    'missing_or_inconsistent_followup': 'label_minus1_not_survival_no_proxy_death_time',
    'postdischarge_mortality': 'not_ascertained_alive_discharge_is_competing_end_of_admission',
    'phenotypes': 'three_dictionary_bound_prior_admission_recorded_disease_families_not_adjudicated_subtypes',
    'disease_family_order': list(disease.FAMILIES),
    'prior_disease_time': 'valid_prior_discharge_strictly_before_index_admission_no_current_or_future_episode_codes',
    'prior_disease_meaning': 'retrospective_administrative_history_not_proven_real_time_clinician_knowledge',
    'no_code_meaning': 'absence_of_qualifying_history_not_absence_of_disease',
    'split': 'same_source_person_HMAC80_10_10_salt_repeated_admissions_grouped',
    'prediction_frame': 'not_selected_here_no_embedding_or_efficacy_computation',
    'disclosure': 'whole_artifact_hashes_only_counts_coarsened20_both_outcome_classes_require20people',
    'max_metadata_rows': 2000000, 'max_lab_rows_per_pass': 200000000, 'max_diagnosis_rows': 20000000,
    'audit': 'full_independent_source_reread_reconstruction_and_exact_private_array_replay',
}
FLAGS = {'patient_level_output_emitted': False, 'training_started': False,
         'clinical_outcome_utility_established': False, 'clinical_use': False,
         'cross_source_identity_resolved': False, 'old_cache_retrofit': False,
         'source_observations_rebuilt': True, 'prior_disease_membership_constructed': True,
         'disease_specific_counts_released': False, 'subtyping_performed': False}


def require(ok):
    if not ok:
        raise ValueError('mimic_landmark_linkage_contract_failed')


def protocol_template():
    p = json.loads(parent.PROTOCOL.read_text())
    parent.validate_protocol(p)
    disease_dictionary.authenticate_audit()
    return {'schema': 'bran-mimic-landmark-linkage-protocol-v1', 'policy': POLICY,
        'parent_protocol_sha256': sha(parent.PROTOCOL),
        'code_sha256': {name: sha(ROOT / name) for name in CODE},
        'source_sha256': {key: p['source_files']['mimic_' + key]['sha256'] for key in INPUTS},
        'phenotype_source_sha256': {key: sha(path) for key, path in PHENOTYPE_INPUTS.items()},
        'disease_dictionary_receipt_sha256': sha(disease_dictionary.OUT),
        'disease_dictionary_audit_sha256': sha(disease_dictionary.AUDIT),
        'split_salt_sha256': sha(SALT),
        'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__}}


def validate_protocol(p):
    require(p == protocol_template())


def phase(state, name, destination):
    require(name in PHASES)
    state['phase'] = name
    value = {'phase': name, 'patient_level_output_emitted': False, 'training_started': False}
    temp = destination / 'progress.tmp'
    temp.write_text(json.dumps(value) + '\n')
    os.replace(temp, destination / 'progress.json')


def source_hashes(p):
    require(set(p['source_sha256']) == set(INPUTS))
    for key, path in INPUTS.items():
        require(sha(path) == p['source_sha256'][key])
    require(set(p['phenotype_source_sha256']) == set(PHENOTYPE_INPUTS))
    for key, path in PHENOTYPE_INPUTS.items():
        require(sha(path) == p['phenotype_source_sha256'][key])


def reconstruct(p):
    """Private arrays only; execute locally with source hashes checked around it."""
    def rows(key, columns):
        limit = POLICY['max_lab_rows_per_pass'] if key == 'labs' else POLICY['max_metadata_rows']
        return iter_projected_csv(INPUTS[key], columns, max_rows=limit)
    metadata = source.predictor_metadata(rows('patients', source.PATIENT_COLUMNS),
                                         rows('admissions', source.ADMISSION_COLUMNS))
    records = list(source.linked_snapshots(lambda: rows('labs', source.LAB_COLUMNS),
                                          metadata, parent.mapping()))
    salt = SALT.read_bytes()
    arrays, rowmap, _ = pack_linked_snapshots(records, salt, p['source_sha256'])
    # Eligibility is computed after predictor construction, not used to select labs.
    scalar_ok = {record.episode: current_v2_age_scalar(record.age) is not None for record in records}
    qualified = np.array([scalar_ok[row['episode']] for row in rowmap['rows']], bool)
    qualified.setflags(write=False)
    outcomes = source.join_landmark_outcomes(arrays, rowmap, salt, p['source_sha256'],
                                             rows('admissions', source.OUTCOME_COLUMNS))
    outcomes['current_scalar_age_qualified'] = qualified
    # Cohorts are separate outputs; neither codes nor outcomes enter predictors.
    prior = disease.assemble_prior_membership(rowmap['rows'],
        rows('admissions', ('subject_id', 'hadm_id', 'admittime', 'dischtime')),
        iter_projected_csv(PHENOTYPE_INPUTS['diagnoses'],
            ('subject_id', 'hadm_id', 'icd_code', 'icd_version'), max_rows=POLICY['max_diagnosis_rows']),
        disease_dictionary.authenticate_audit())
    return arrays, rowmap, outcomes, prior


def summarize(arrays, outcomes, hashes):
    eligible = outcomes['current_scalar_age_qualified'] & (outcomes['label'] >= 0)
    groups = arrays['person_group']
    counts = [len(np.unique(groups[eligible & (outcomes['label'] == label)])) for label in (0, 1)]
    supported = min(counts) >= 20
    return {'schema': 'bran-mimic-landmark-linkage-aggregate-v1', 'status': 'completed',
        'private_sha256': hashes, 'outcome_support': 'supported' if supported else 'suppressed_insufficient_support',
        'age_qualified_later_outcome_snapshots_lower_bound': coarse_count(int(eligible.sum())) if supported else None,
        'age_qualified_later_outcome_people_lower_bound': coarse_count(len(np.unique(groups[eligible]))) if supported else None,
        'surviving_discharge_people_lower_bound': coarse_count(counts[0]) if supported else None,
        'later_in_hospital_death_people_lower_bound': coarse_count(counts[1]) if supported else None,
        **FLAGS}


def validate_result(a):
    names = ('age_qualified_later_outcome_snapshots_lower_bound', 'age_qualified_later_outcome_people_lower_bound',
             'surviving_discharge_people_lower_bound', 'later_in_hospital_death_people_lower_bound')
    require(type(a) is dict and set(a) == {'schema', 'status', 'private_sha256', 'outcome_support', *names, *FLAGS})
    require(a['schema'] == 'bran-mimic-landmark-linkage-aggregate-v1' and a['status'] == 'completed')
    require(all(a[key] is value for key, value in FLAGS.items()))
    require(type(a['private_sha256']) is dict and set(a['private_sha256']) == set(FILES)
            and all(valid_sha(value) for value in a['private_sha256'].values()))
    require(a['outcome_support'] in ('supported', 'suppressed_insufficient_support'))
    for name in names:
        value = a[name]
        require((type(value) is int and value >= 20 and value % 20 == 0)
                if a['outcome_support'] == 'supported' else value is None)
    if a['outcome_support'] == 'supported':
        require(a[names[0]] >= a[names[1]] >= max(a[names[2]], a[names[3]]))


def write_private(name, value):
    fd = os.open(PRIVATE / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        if name.endswith('.npz'):
            np.savez_compressed(handle, **value)
        else:
            handle.write((json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode())
        handle.flush(); os.fsync(handle.fileno())


def equal_arrays(first, second):
    require(set(first) == set(second))
    for key in first:
        a, b = first[key], second[key]
        require(a.dtype == b.dtype and a.shape == b.shape)
        require(np.array_equal(a, b, equal_nan=True) if a.dtype.kind == 'f' else np.array_equal(a, b))


def compute(p, state):
    phase(state, 'source_authentication', OUT); source_hashes(p)
    phase(state, 'source_reconstruction', OUT)
    arrays, rowmap, outcomes, prior = reconstruct(p)
    phase(state, 'private_write', OUT); PRIVATE.mkdir(mode=0o700)
    for name, value in zip(FILES, (arrays, rowmap, outcomes, prior)):
        write_private(name, value)
    source_hashes(p); validate_protocol(p)
    result = summarize(arrays, outcomes, {name: sha(PRIVATE / name) for name in FILES})
    validate_result(result)
    return result


def authenticate_terminal(p, pin):
    require(valid_sha(pin) and sha(PROTOCOL) == pin)
    require(json.loads(PROTOCOL.read_text()) == p); validate_protocol(p)
    require(not (OUT / 'failure.json').exists())
    result = json.loads((OUT / 'aggregate.json').read_text()); validate_result(result)
    manifest = json.loads((OUT / 'aggregate.manifest.json').read_text())
    require(manifest == {'protocol_sha256': pin, 'artifact_sha256': sha(OUT / 'aggregate.json')})
    require(PRIVATE.is_dir() and not PRIVATE.is_symlink() and PRIVATE.stat().st_mode & 0o777 == 0o700)
    for name in FILES:
        path = PRIVATE / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_mode & 0o777 == 0o600)
        require(sha(path) == result['private_sha256'][name])
    return result


def audit(p, pin, state):
    original = authenticate_terminal(p, pin)
    phase(state, 'source_authentication', AUDIT); source_hashes(p)
    phase(state, 'source_replay', AUDIT)
    arrays, rowmap, outcomes, prior = reconstruct(p)
    stored_map = json.loads((PRIVATE / 'row_map.json').read_text())
    require(rowmap == stored_map)
    with np.load(PRIVATE / 'observations.npz', allow_pickle=False) as handle:
        stored_arrays = {key: handle[key] for key in handle.files}
    validate_linked_snapshot_pack(stored_arrays, stored_map, SALT.read_bytes(), p['source_sha256'])
    equal_arrays(arrays, stored_arrays)
    with np.load(PRIVATE / 'outcomes.npz', allow_pickle=False) as handle:
        equal_arrays(outcomes, {key: handle[key] for key in handle.files})
    with np.load(PRIVATE / 'prior_diseases.npz', allow_pickle=False) as handle:
        equal_arrays(prior, {key: handle[key] for key in handle.files})
    source_hashes(p); validate_protocol(p)
    require(authenticate_terminal(p, pin) == original)
    require(summarize(arrays, outcomes, original['private_sha256']) == original)
    result = {'schema': 'bran-mimic-landmark-linkage-audit-v1', 'status': 'passed',
            'protocol_sha256': pin, 'aggregate_sha256': sha(OUT / 'aggregate.json'),
            'source_replay_equal': True, 'patient_level_output_emitted': False,
            'training_started': False, 'clinical_outcome_utility_established': False}
    validate_audit(result, pin, sha(OUT / 'aggregate.json'))
    return result


def validate_audit(a, pin, aggregate_pin):
    require(valid_sha(pin) and valid_sha(aggregate_pin))
    require(type(a) is dict and a == {
        'schema': 'bran-mimic-landmark-linkage-audit-v1', 'status': 'passed',
        'protocol_sha256': pin, 'aggregate_sha256': aggregate_pin,
        'source_replay_equal': True, 'patient_level_output_emitted': False,
        'training_started': False, 'clinical_outcome_utility_established': False})
    require(a['source_replay_equal'] is True and a['patient_level_output_emitted'] is False
            and a['training_started'] is False and a['clinical_outcome_utility_established'] is False)


def main(argv=None):
    parser = argparse.ArgumentParser(); operations = parser.add_mutually_exclusive_group(required=True)
    for mode in ('prepare', 'run', 'audit'):
        operations.add_argument('--' + mode, action='store_true')
    parser.add_argument('--protocol-sha256'); args = parser.parse_args(argv)
    ok, owned = False, False
    dest = AUDIT if args.audit else OUT
    state = {'phase': 'source_authentication'}
    with _quiet():
        try:
            # Same exclusion lock as retinal extraction; no concurrent heavy run.
            with open(LOCK, 'a') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.prepare:
                    require(not any(path.exists() for path in (PROTOCOL, OUT, AUDIT, PRIVATE)))
                    p = protocol_template(); source_hashes(p)
                    exclusive_json(PROTOCOL, p)
                else:
                    require(args.protocol_sha256 and sha(PROTOCOL) == args.protocol_sha256)
                    p = json.loads(PROTOCOL.read_text()); validate_protocol(p)
                    dest.mkdir(); owned = True
                    if args.audit:
                        result = audit(p, args.protocol_sha256, state); name = 'audit.json'
                    else:
                        require(not PRIVATE.exists() and not AUDIT.exists())
                        result = compute(p, state); name = 'aggregate.json'
                    phase(state, 'publication', dest)
                    publish(dest, name, result, args.protocol_sha256)
            ok = True
        except Exception:
            try:
                if owned and not any((dest / name).exists() for name in ('aggregate.json', 'audit.json')):
                    require(state['phase'] in PHASES)
                    exclusive_json(dest / 'failure.json', {'status': 'execution_failed', 'phase': state['phase'],
                        'patient_level_output_emitted': False, 'training_started': False})
            except Exception:
                pass  # Never emit source-bearing exception text after leaving _quiet.
    print(json.dumps({'status': 'completed' if ok else 'execution_failed', 'patient_level_output_emitted': False}))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
