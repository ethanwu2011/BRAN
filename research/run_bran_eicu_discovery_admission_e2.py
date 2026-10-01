"""FD-quiet broader eICU admission; no model, clustering, or public private data."""
import fcntl
import json
import os
from pathlib import Path
import numpy as np

import run_bran_eicu_discovery_admission_e1 as e1
import bran_eicu_discovery_bridge_v1 as bridge
import bran_eicu_chemistry_adapter_e2 as chemistry
from bran_clinical_source_reader_v1 import iter_projected_csv, build_episode_links
from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json
from run_bran_source_pattern_v6 import safe_site

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'BRAN_EICU_DISCOVERY_ADMISSION_E2_ATTEMPT1'
PRIVATE = ROOT/'private_artifacts'/'bran_eicu_discovery_admission_e2_attempt1'
INPUTS = {**e1.INPUTS, 'apache': Path('/Users/ethanwu/eicu-crd-2.0/apachePredVar.csv.gz')}
LIMITS = {'patient': 500000, 'lab': 100000000, 'diagnosis': 10000000, 'apache': 1000000}
APACHE_COLUMNS = ('patientunitstayid', 'diabetes')
PHASES = ('authentication', 'source_hashes', 'metadata', 'labs', 'diagnoses', 'apache',
          'private_artifact', 'post_authentication', 'completed')
ERROR = 'eicu_discovery_admission_e2_failed'
FALSE_FLAGS = ('model_inference_performed', 'clustering_performed', 'training_exposure_changed',
               'clinically_adjudicated_membership', 'patient_level_output_emitted')


def require(ok):
    if not ok: raise ValueError(ERROR) from None


def code_hashes():
    names = ('bran_eicu_chemistry_adapter_e2.py', 'test_bran_eicu_chemistry_adapter_e2.py',
             'run_bran_eicu_discovery_admission_e2.py', 'test_bran_eicu_admission_e2.py',
             'BRAN_EICU_DISCOVERY_ADMISSION_E2_DESIGN.md',
             'bran_clinical_chemistry_semantics_v1.py')
    return {**e1.code_hashes(), **{n: sha(ROOT/n) for n in names}}


def progress(phase):
    require(phase in PHASES)
    write_json(OUT/'progress.next.json', {'phase': phase, 'pid': os.getpid(),
                                        'patient_level_output_emitted': False})
    os.replace(OUT/'progress.next.json', OUT/'progress.json')


def e1_binding():
    e1.authenticate()
    return {'protocol_sha256': sha(e1.OUT/'protocol.json'),
            'aggregate_sha256': sha(e1.OUT/'aggregate.json')}


def baseline_people():
    with np.load(e1.PRIVATE/'cohort.npz', allow_pickle=False) as data:
        return set(data['person'].tolist())


def update_apache(flags, row):
    """Only 0/1 carries meaning. Contradictory known records remain unknown."""
    require(type(row) is dict and set(row) == set(APACHE_COLUMNS))
    episode = bridge.key(row['patientunitstayid'])
    payload = row['diabetes']
    require(type(payload) is str)
    flag = {'0': 0, '1': 1}.get(payload, -1)
    if flag < 0: return
    previous = flags.get(episode)
    if previous is None: flags[episode] = flag
    elif previous != flag: flags[episode] = -1


def validate_arrays(a):
    names = {'values', 'observed', 'person', 'episode', 'hospital_stay', 'hospital',
             'age_triplet', 'age_kind', 'membership', 'outcome', 'apache_diabetes_indicator'}
    require(type(a) is dict and set(a) == names)
    n = len(a['person']); width = len(chemistry.FIELDS)
    require(width == 21 and len(set(chemistry.FIELDS)) == width)
    for name in ('person', 'episode', 'hospital_stay', 'hospital'):
        require(a[name].shape == (n,) and a[name].dtype.kind == 'U')
        require(all(type(v) is str and bridge.key(v) == v for v in a[name].tolist()))
    require(len(set(a['person'])) == n and len(set(a['episode'])) == n)
    v, o = a['values'], a['observed']
    require(v.shape == o.shape == (n, width) and v.dtype == np.float64 and o.dtype == bool)
    require(np.isfinite(v).all() and (v[~o] == 0).all() and (v[o] >= 0).all()
            and (v[:, :9][o[:, :9]] > 0).all() and (o.sum(1) >= 1).all())
    require(not o[:, chemistry.FIELDS.index('protein_total')].any())
    ages, kinds = a['age_triplet'], a['age_kind']
    require(ages.shape == (n, 3) and ages.dtype == np.float64 and kinds.shape == (n,)
            and kinds.dtype == np.uint8 and np.isin(kinds, (0, 1, 2)).all())
    require(np.isnan(ages[kinds == 0]).all())
    reported = ages[kinds == 1]
    require(np.isfinite(reported).all() and (reported[:, 0] >= 18).all()
            and (reported[:, 0] == reported[:, 1]).all()
            and (reported[:, 2] == reported[:, 0]+1).all())
    censored = ages[kinds == 2]
    require(np.isnan(censored[:, 0]).all() and (censored[:, 1] == 90).all()
            and np.isposinf(censored[:, 2]).all())
    require(a['membership'].shape == (n, 3) and a['membership'].dtype == bool)
    for name in ('outcome', 'apache_diabetes_indicator'):
        require(a[name].shape == (n,) and a[name].dtype == np.int8
                and np.isin(a[name], (-1, 0, 1)).all())


def summary(a, old_people):
    people = set(a['person'].tolist())
    require(old_people <= people)
    c = e1.count; o = a['observed']; known = a['outcome'] >= 0
    return {'source_local_people_lower_bound_20': c(len(people)),
        'added_vs_e1_people_lower_bound_20': c(len(people-old_people)),
        'all_e1_people_retained': True,
        'chemistry_only_people_lower_bound_20': c(int((~o[:, :9].any(1)).sum())),
        'unknown_age_people_lower_bound_20': c(int((a['age_kind'] == 0).sum())),
        'observed_field_people_lower_bound_20': {field: c(int(o[:, j].sum()))
            for j, field in enumerate(chemistry.FIELDS)},
        'recorded_membership_people_lower_bound_20': {family: c(int(a['membership'][:, j].sum()))
            for j, family in enumerate(e1.dictionary.FAMILIES)},
        'recorded_membership_known_outcome_people_lower_bound_20': {
            family: c(int((a['membership'][:, j] & known).sum()))
            for j, family in enumerate(e1.dictionary.FAMILIES)},
        'apache_diabetes_indicator_positive_people_lower_bound_20':
            c(int((a['apache_diabetes_indicator'] == 1).sum())),
        'apache_diabetes_indicator_positive_known_outcome_people_lower_bound_20':
            c(int(((a['apache_diabetes_indicator'] == 1) & known).sum()))}


def authenticate(*, _pending_terminal=None):
    expected = {'protocol.json', 'aggregate.json', 'progress.json'}
    if _pending_terminal is None: expected.add('completed.json')
    require(OUT.is_dir() and not OUT.is_symlink() and set(p.name for p in OUT.iterdir()) == expected)
    p, a = (e1.read_closed(OUT/n) for n in ('protocol.json', 'aggregate.json'))
    t = _pending_terminal if _pending_terminal is not None else e1.read_closed(OUT/'completed.json')
    require(set(p) == {'schema', 'status', 'code_sha256', 'source_sha256', 'e1_binding',
                      'fields', 'max_rows', 'minimum_observed_fields', 'patient_level_output_emitted'})
    require(p['schema'] == 'bran-eicu-discovery-admission-e2-protocol'
            and p['status'] == 'frozen_before_source_rows' and p['code_sha256'] == code_hashes()
            and p['fields'] == list(chemistry.FIELDS) and p['max_rows'] == LIMITS
            and p['minimum_observed_fields'] == 1 and p['patient_level_output_emitted'] is False
            and p['e1_binding'] == e1_binding() and set(p['source_sha256']) == set(INPUTS))
    require(all(sha(path) == p['source_sha256'][key] for key, path in INPUTS.items()))
    require(set(a) == {'schema', 'status', 'summary', 'private_sha256', *FALSE_FLAGS}
            and a['schema'] == 'bran-eicu-discovery-admission-e2' and a['status'] == 'completed'
            and all(a[k] is False for k in FALSE_FLAGS))
    require(PRIVATE.is_dir() and not PRIVATE.is_symlink() and PRIVATE.stat().st_mode&0o777 == 0o700
            and set(p.name for p in PRIVATE.iterdir()) == {'cohort.npz'})
    path = PRIVATE/'cohort.npz'
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
            and path.stat().st_mode&0o777 == 0o600 and sha(path) == a['private_sha256'])
    with np.load(path, allow_pickle=False) as data:
        arrays = {n: data[n] for n in data.files}
    validate_arrays(arrays)
    require(a['summary'] == summary(arrays, baseline_people()) and sha(path) == a['private_sha256'])
    require(set(t) == {'status', 'protocol_sha256', 'aggregate_sha256', 'patient_level_output_emitted'}
            and t['status'] == 'authenticated_completed' and t['patient_level_output_emitted'] is False
            and t['protocol_sha256'] == sha(OUT/'protocol.json')
            and t['aggregate_sha256'] == sha(OUT/'aggregate.json'))
    q = e1.read_closed(OUT/'progress.json')
    require(set(q) == {'phase', 'pid', 'patient_level_output_emitted'} and q['phase'] == 'completed'
            and type(q['pid']) is int and q['pid'] > 0 and q['patient_level_output_emitted'] is False)
    return a, t


def run(state=None):
    require(not OUT.exists() and not OUT.is_symlink() and not PRIVATE.exists() and not PRIVATE.is_symlink())
    OUT.mkdir()
    if state is not None: state['owned'] = True
    progress('authentication')
    binding = e1_binding(); old_people = baseline_people()
    vocabulary = bridge.DiseaseVocabulary(e1.dictionary.authenticate_audit())
    progress('source_hashes')
    p = {'schema': 'bran-eicu-discovery-admission-e2-protocol', 'status': 'frozen_before_source_rows',
         'code_sha256': code_hashes(), 'source_sha256': {k: sha(v) for k, v in INPUTS.items()},
         'e1_binding': binding, 'fields': list(chemistry.FIELDS), 'max_rows': dict(LIMITS),
         'minimum_observed_fields': 1, 'patient_level_output_emitted': False}
    write_json(OUT/'protocol.json', p); progress('metadata')
    rows = list(iter_projected_csv(INPUTS['patient'], bridge.PATIENT_COLUMNS, max_rows=LIMITS['patient']))
    links = build_episode_links('eicu', rows, [])
    episodes = {r['patientunitstayid']: bridge.parse_patient(r) for r in rows}
    require(len(episodes) == len(rows)); del rows
    panels = {}; progress('labs')
    for row in iter_projected_csv(INPUTS['lab'], bridge.LAB_COLUMNS, max_rows=LIMITS['lab']):
        event = bridge.observed_cbc_before_landmark(row, links)
        if event is None: event = chemistry.observed_chemistry_before_landmark(row, links)
        if event is not None: e1.update_panel(panels, event)
    counts = {key: sum(v[2] for v in panel.values()) for key, panel in panels.items()}
    selected = bridge.select_one_per_person(episodes.values(), counts)
    index = {e.episode: j for j, e in enumerate(selected)}; n = len(selected)
    values = np.zeros((n, len(chemistry.FIELDS)), np.float64); observed = np.zeros_like(values, dtype=bool)
    for i, episode in enumerate(selected):
        for field, (_time, value, valid) in panels[episode.episode].items():
            if valid: values[i, field] = value; observed[i, field] = True
    del panels, counts
    progress('diagnoses'); membership = np.zeros((n, 3), bool)
    for row in iter_projected_csv(INPUTS['diagnosis'], bridge.DIAGNOSIS_COLUMNS, max_rows=LIMITS['diagnosis']):
        i = index.get(row['patientunitstayid'])
        if i is None: continue
        for family in bridge.recorded_membership(row, vocabulary, selected[i]):
            membership[i, e1.dictionary.FAMILIES.index(family)] = True
    progress('apache'); flags = {}
    for row in iter_projected_csv(INPUTS['apache'], APACHE_COLUMNS, max_rows=LIMITS['apache']):
        if row['patientunitstayid'] in index: update_apache(flags, row)
    arrays = {name: np.asarray([getattr(e, name) for e in selected], dtype='U128')
              for name in ('person', 'episode', 'hospital_stay', 'hospital')}
    arrays.update(values=values, observed=observed, membership=membership,
        apache_diabetes_indicator=np.asarray([flags.get(e.episode, -1) for e in selected], np.int8),
        age_triplet=np.asarray([(e.age.reported_years, e.age.lower_years, e.age.upper_years)
                               for e in selected], np.float64).reshape(n, 3),
        age_kind=np.asarray([e1.AGE_KINDS.index(e.age.kind) for e in selected], np.uint8),
        outcome=np.asarray([e.outcome for e in selected], np.int8))
    validate_arrays(arrays); safe_summary = summary(arrays, old_people)
    progress('private_artifact')
    require(PRIVATE.parent.is_dir() and not PRIVATE.parent.is_symlink()); PRIVATE.mkdir(mode=0o700)
    path = PRIVATE/'cohort.npz'; descriptor = os.open(path, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'wb') as handle: np.savez_compressed(handle, **arrays)
    progress('post_authentication')
    require(p['code_sha256'] == code_hashes() and binding == e1_binding()
            and all(sha(v) == p['source_sha256'][k] for k, v in INPUTS.items()))
    aggregate = {'schema': 'bran-eicu-discovery-admission-e2', 'status': 'completed',
                 'summary': safe_summary, 'private_sha256': sha(path), **{k: False for k in FALSE_FLAGS}}
    write_json(OUT/'aggregate.json', aggregate); progress('completed')
    terminal = {'status': 'authenticated_completed',
        'protocol_sha256': sha(OUT/'protocol.json'), 'aggregate_sha256': sha(OUT/'aggregate.json'),
        'patient_level_output_emitted': False}
    authenticate(_pending_terminal=terminal)
    write_json(OUT/'completed.json', terminal)


def main():
    ok = False; state = {'owned': False}
    with quiet():
        try:
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                run(state); authenticate(); ok = True
        except Exception as exc:
            if state['owned'] and OUT.is_dir():
                if (OUT/'completed.json').is_file():
                    os.replace(OUT/'completed.json', OUT/'rejected_completed.json')
                write_json(OUT/'failure.json', {'status': 'technical_failure',
                    'safe_code_site': safe_site(exc), 'patient_level_output_emitted': False})
    print(json.dumps({'status': 'completed' if ok else 'not_completed', 'patient_level_output_emitted': False}))
    return 0 if ok else 1


if __name__ == '__main__': raise SystemExit(main())
