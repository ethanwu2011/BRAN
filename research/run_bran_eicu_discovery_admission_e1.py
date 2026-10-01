"""FD-quiet one-physiology eICU qualification. No model or cluster fitting."""
import fcntl
import json
import math
import os
from pathlib import Path
import numpy as np

from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json
from run_bran_source_pattern_v6 import safe_site
from audit_bran_source_pattern_v6 import read as read_json_closed
import bran_eicu_discovery_bridge_v1 as bridge
from bran_clinical_source_reader_v1 import iter_projected_csv, build_episode_links
import bran_mimic_prior_disease_dictionary_v1 as dictionary

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'BRAN_EICU_DISCOVERY_ADMISSION_E1_ATTEMPT1'
PRIVATE = ROOT / 'private_artifacts' / 'bran_eicu_discovery_admission_e1_attempt1'
INPUTS = {name: Path('/Users/ethanwu/eicu-crd-2.0') / (name+'.csv.gz')
          for name in ('patient', 'lab', 'diagnosis')}
AGE_KINDS = ('missing_or_invalid', 'reported_year', 'topcoded')
PHASES = ('authentication', 'source_hashes', 'metadata', 'labs', 'diagnoses',
          'private_artifact', 'post_authentication', 'completed')
ERROR = 'eicu_discovery_admission_e1_failed'


def require(ok):
    if not ok:
        raise ValueError(ERROR) from None


def code_hashes():
    names = ('bran_eicu_discovery_bridge_v1.py', 'test_bran_eicu_discovery_bridge_v1.py',
             'run_bran_eicu_discovery_admission_e1.py', 'test_bran_eicu_admission_e1.py',
             'BRAN_EICU_DISCOVERY_ADMISSION_E1_DESIGN.md',
             'bran_eicu_cbc_adapter_v2.py', 'bran_cbc_event_adapter_v1.py',
             'bran_cbc_mcl_units_v3.py', 'bran_clinical_semantics_v1.py',
             'bran_clinical_source_reader_v1.py', 'bran_mimic_prior_disease_v1.py',
             'run_bran_multisource_retinal_features_v2.py', 'run_bran_source_pattern_v6.py',
             'audit_bran_source_pattern_v6.py')
    return {**{n: sha(ROOT/n) for n in names},
            **{n: sha(ROOT/n) for n in dictionary.CODE}}


def count(value):
    require(type(value) is int and value >= 0)
    return value//20*20 if value >= 20 else None


def progress(phase):
    require(phase in PHASES)
    write_json(OUT/'progress.next.json', {'phase': phase, 'pid': os.getpid(),
               'patient_level_output_emitted': False})
    os.replace(OUT/'progress.next.json', OUT/'progress.json')


def update_panel(panels, event):
    """Latest observed specimen; equal-time disagreement is missing, not average."""
    panel = panels.setdefault(event.episode_key, {})
    old = panel.get(event.field_index)
    if old is None or event.offset_minutes > old[0]:
        panel[event.field_index] = (event.offset_minutes, event.canonical_value, True)
    elif event.offset_minutes == old[0] and event.canonical_value != old[1]:
        panel[event.field_index] = (old[0], old[1], False)


def summary(arrays):
    observed = arrays['observed']
    require(observed.ndim == 2 and observed.shape[1] == 9 and observed.dtype == bool)
    n = len(observed)
    require(arrays['membership'].shape == (n, 3) and arrays['outcome'].shape == (n,))
    return {'source_local_people_lower_bound_20': count(n),
            'at_least_two_cbc_people_lower_bound_20': count(int((observed.sum(1)>=2).sum())),
            'unknown_age_people_lower_bound_20': count(int((arrays['age_kind']==0).sum())),
            'recorded_membership_people_lower_bound_20': {
                family: count(int(arrays['membership'][:, j].sum()))
                for j, family in enumerate(dictionary.FAMILIES)},
            'recorded_membership_known_outcome_people_lower_bound_20': {
                family: count(int((arrays['membership'][:, j] & (arrays['outcome']>=0)).sum()))
                for j, family in enumerate(dictionary.FAMILIES)}}


def validate_arrays(arrays):
    expected = {'values', 'observed', 'person', 'episode', 'hospital_stay', 'hospital',
                'age_triplet', 'age_kind', 'membership', 'outcome'}
    require(type(arrays) is dict and set(arrays) == expected)
    n = len(arrays['person'])
    for key in ('person', 'episode', 'hospital_stay', 'hospital'):
        require(arrays[key].shape == (n,) and arrays[key].dtype.kind == 'U')
    require(len(set(arrays['person'])) == n and len(set(arrays['episode'])) == n)
    require(arrays['values'].shape == arrays['observed'].shape == (n, 9)
            and arrays['values'].dtype == np.float64 and arrays['observed'].dtype == bool
            and np.isfinite(arrays['values']).all()
            and (arrays['values'][arrays['observed']]>0).all()
            and (arrays['values'][~arrays['observed']]==0).all()
            and (arrays['observed'].sum(1)>=1).all())
    require(arrays['age_triplet'].shape == (n, 3) and arrays['age_triplet'].dtype == np.float64
            and arrays['age_kind'].shape == (n,) and arrays['age_kind'].dtype == np.uint8
            and np.isin(arrays['age_kind'], range(3)).all())
    require(arrays['membership'].shape == (n, 3) and arrays['membership'].dtype == bool
            and arrays['outcome'].shape == (n,) and arrays['outcome'].dtype == np.int8
            and np.isin(arrays['outcome'], (-1, 0, 1)).all())


def read_closed(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1)
    return read_json_closed(path)


def authenticate():
    require(OUT.is_dir() and not OUT.is_symlink()
            and set(p.name for p in OUT.iterdir()) ==
                {'protocol.json', 'aggregate.json', 'completed.json', 'progress.json'})
    p, a, t = (read_closed(OUT/n) for n in ('protocol.json', 'aggregate.json', 'completed.json'))
    require(set(p) == {'schema', 'status', 'code_sha256', 'source_sha256',
                      'vocabulary_receipt_sha256', 'vocabulary_audit_sha256',
                      'max_rows', 'minimum_observed_fields', 'patient_level_output_emitted'})
    require(p['schema'] == 'bran-eicu-discovery-admission-e1-protocol'
            and p['status'] == 'frozen_before_source_rows' and p['code_sha256'] == code_hashes()
            and p['max_rows'] == {'patient': 500000, 'lab': 100000000, 'diagnosis': 10000000}
            and p['minimum_observed_fields'] == 1 and p['patient_level_output_emitted'] is False
            and set(p['source_sha256']) == set(INPUTS))
    require(p['vocabulary_receipt_sha256'] == sha(dictionary.OUT)
            and p['vocabulary_audit_sha256'] == sha(dictionary.AUDIT))
    dictionary.authenticate_audit()
    require(all(sha(path) == p['source_sha256'][key] for key, path in INPUTS.items()))
    require(set(a) == {'schema', 'status', 'summary', 'private_sha256',
            'model_inference_performed', 'clustering_performed', 'training_exposure_changed',
            'clinically_adjudicated_membership', 'patient_level_output_emitted'})
    require(a['schema'] == 'bran-eicu-discovery-admission-e1' and a['status'] == 'completed'
            and all(a[k] is False for k in ('model_inference_performed', 'clustering_performed',
                'training_exposure_changed', 'clinically_adjudicated_membership',
                'patient_level_output_emitted')))
    require(PRIVATE.is_dir() and not PRIVATE.is_symlink() and PRIVATE.stat().st_mode&0o777 == 0o700
            and set(p.name for p in PRIVATE.iterdir()) == {'cohort.npz'})
    path = PRIVATE/'cohort.npz'
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
            and path.stat().st_mode&0o777 == 0o600 and sha(path) == a['private_sha256'])
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    validate_arrays(arrays)
    require(summary(arrays) == a['summary'] and sha(path) == a['private_sha256'])
    require(set(t) == {'status', 'protocol_sha256', 'aggregate_sha256', 'patient_level_output_emitted'}
            and t['status'] == 'authenticated_completed' and t['patient_level_output_emitted'] is False
            and t['protocol_sha256'] == sha(OUT/'protocol.json')
            and t['aggregate_sha256'] == sha(OUT/'aggregate.json'))
    q = read_closed(OUT/'progress.json')
    require(set(q) == {'phase', 'pid', 'patient_level_output_emitted'} and q['phase'] == 'completed'
            and type(q['pid']) is int and q['pid'] > 0 and q['patient_level_output_emitted'] is False)
    return a, t


def run():
    require(not OUT.exists() and not OUT.is_symlink() and not PRIVATE.exists() and not PRIVATE.is_symlink())
    OUT.mkdir(); progress('authentication')
    vocabulary = bridge.DiseaseVocabulary(dictionary.authenticate_audit())
    progress('source_hashes')
    protocol = {'schema': 'bran-eicu-discovery-admission-e1-protocol',
        'status': 'frozen_before_source_rows', 'code_sha256': code_hashes(),
        'source_sha256': {k: sha(v) for k, v in INPUTS.items()},
        'vocabulary_receipt_sha256': sha(dictionary.OUT), 'vocabulary_audit_sha256': sha(dictionary.AUDIT),
        'max_rows': {'patient': 500000, 'lab': 100000000, 'diagnosis': 10000000},
        'minimum_observed_fields': 1, 'patient_level_output_emitted': False}
    write_json(OUT/'protocol.json', protocol)
    progress('metadata')
    rows = list(iter_projected_csv(INPUTS['patient'], bridge.PATIENT_COLUMNS,
                                  max_rows=protocol['max_rows']['patient']))
    links = build_episode_links('eicu', rows, [])
    episodes = {r['patientunitstayid']: bridge.parse_patient(r) for r in rows}
    require(len(episodes) == len(rows)); del rows
    progress('labs'); panels = {}
    for row in iter_projected_csv(INPUTS['lab'], bridge.LAB_COLUMNS, max_rows=protocol['max_rows']['lab']):
        event = bridge.observed_cbc_before_landmark(row, links)
        if event is not None:
            update_panel(panels, event)
    counts = {key: sum(v[2] for v in panel.values()) for key, panel in panels.items()}
    selected = bridge.select_one_per_person(episodes.values(), counts)
    index = {e.episode: j for j, e in enumerate(selected)}
    n = len(selected)
    values = np.zeros((n, 9), np.float64); observed = np.zeros((n, 9), bool)
    for i, episode in enumerate(selected):
        for field, (_time, value, valid) in panels[episode.episode].items():
            if valid: values[i, field] = value; observed[i, field] = True
    del panels, counts
    progress('diagnoses'); membership = np.zeros((n, 3), bool)
    for row in iter_projected_csv(INPUTS['diagnosis'], bridge.DIAGNOSIS_COLUMNS,
                                  max_rows=protocol['max_rows']['diagnosis']):
        i = index.get(row['patientunitstayid'])
        if i is None: continue
        for family in bridge.recorded_membership(row, vocabulary, selected[i]):
            membership[i, dictionary.FAMILIES.index(family)] = True
    arrays = {name: np.asarray([getattr(e, name) for e in selected], dtype='U128')
              for name in ('person', 'episode', 'hospital_stay', 'hospital')}
    arrays.update(values=values, observed=observed, membership=membership,
        age_triplet=np.asarray([(e.age.reported_years, e.age.lower_years, e.age.upper_years)
                                for e in selected], np.float64).reshape(n, 3),
        age_kind=np.asarray([AGE_KINDS.index(e.age.kind) for e in selected], np.uint8),
        outcome=np.asarray([e.outcome for e in selected], np.int8))
    validate_arrays(arrays); progress('private_artifact')
    require(PRIVATE.parent.is_dir() and not PRIVATE.parent.is_symlink())
    PRIVATE.mkdir(mode=0o700)
    path = PRIVATE/'cohort.npz'
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'wb') as handle: np.savez_compressed(handle, **arrays)
    progress('post_authentication')
    require(protocol['code_sha256'] == code_hashes()
            and all(sha(path) == protocol['source_sha256'][key] for key, path in INPUTS.items()))
    dictionary.authenticate_audit()
    aggregate = {'schema': 'bran-eicu-discovery-admission-e1', 'status': 'completed',
        'summary': summary(arrays), 'private_sha256': sha(path),
        'model_inference_performed': False, 'clustering_performed': False,
        'training_exposure_changed': False, 'clinically_adjudicated_membership': False,
        'patient_level_output_emitted': False}
    write_json(OUT/'aggregate.json', aggregate); progress('completed')
    write_json(OUT/'completed.json', {'status': 'authenticated_completed',
        'protocol_sha256': sha(OUT/'protocol.json'), 'aggregate_sha256': sha(OUT/'aggregate.json'),
        'patient_level_output_emitted': False})


def main():
    ok = False
    with quiet():
        try:
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                run(); authenticate(); ok = True
        except Exception as exc:
            if OUT.is_dir() and not (OUT/'completed.json').exists():
                write_json(OUT/'failure.json', {'status': 'technical_failure',
                    'safe_code_site': safe_site(exc), 'patient_level_output_emitted': False})
    print(json.dumps({'status': 'completed' if ok else 'not_completed',
                      'patient_level_output_emitted': False}))
    return 0 if ok else 1


if __name__ == '__main__': raise SystemExit(main())
