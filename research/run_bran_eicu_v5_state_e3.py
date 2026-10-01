"""Quiet eICU E2-to-V5-fold0 inference with full checkpoint replay; no fitting."""
import fcntl
import json
import os
from pathlib import Path
import numpy as np
import torch

import run_bran_eicu_discovery_admission_e2 as admission
import run_bran_knhanes_v5_external as provider
import bran_eicu_v5_state_e3 as state_bridge
from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json
from run_bran_source_pattern_v6 import safe_site

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'BRAN_EICU_V5_STATE_E3_ATTEMPT1'
PRIVATE = ROOT/'private_artifacts'/'bran_eicu_v5_state_e3_attempt1'
DECISION = ROOT/'BRAN_NATURE_WEEK_MODEL_DECISION_2026-09-21.json'
DECISION_PIN = '13bdd6dce2f2ccbb51008d0c90ea40df88f03805ff451017b97bf359e1d490d3'
ERROR = 'eicu_v5_state_e3_failed'
PHASES = ('authentication', 'inference', 'checkpoint_replay', 'private_artifact',
          'post_authentication', 'completed')
FALSE_FLAGS = ('patient_level_output_emitted', 'encoder_fitted', 'clustering_performed',
               'novel_subtype_claim', 'clinical_utility_established', 'external_validation_established')


def require(ok):
    if not ok: raise ValueError(ERROR) from None


def code_hashes():
    own = ('bran_eicu_v5_state_e3.py', 'test_bran_eicu_v5_state_e3.py',
           'run_bran_eicu_v5_state_e3.py', 'test_run_bran_eicu_v5_state_e3.py',
           'BRAN_EICU_V5_STATE_E3_DESIGN.md', 'bran_nhanes_v5_state.py',
           'bran_joint_lab_task_contract_v1.py', 'bran_joint_lab_cache_v1.py',
           'bran_multisource_profiles_v3.py')
    return {**admission.code_hashes(), **{name: sha(ROOT/name) for name in set(provider.CODE)|set(own)}}


def source_receipt():
    a, t = admission.authenticate()
    return {'protocol_sha256': t['protocol_sha256'], 'aggregate_sha256': t['aggregate_sha256'],
            'terminal_sha256': sha(admission.OUT/'completed.json'), 'private_sha256': a['private_sha256']}


def model_receipt():
    require(sha(DECISION) == DECISION_PIN)
    d = admission.e1.read_closed(DECISION); m = provider.model_binding()
    selected = d['single_downstream_coordinate_frame']
    require(d['research_lead'] == 'V5_M_attempt2' and d['coordinate_frames_may_be_pooled'] is False
            and selected['fold'] == 0 and selected['checkpoint_sha256'] == m['checkpoint_sha256']
            and selected['receipt_sha256'] == m['component_sha256']
            and m['binding']['role'] == 'M' and m['binding']['fold'] == 0)
    return {'decision_sha256': DECISION_PIN, 'provider': m}


def load_cohort(source):
    path = admission.PRIVATE/'cohort.npz'
    require(sha(path) == source['private_sha256'])
    with np.load(path, allow_pickle=False) as h:
        a = {name: h[name] for name in h.files}
    admission.validate_arrays(a)
    require(sha(path) == source['private_sha256'])
    return a


def phase(name):
    require(name in PHASES)
    write_json(OUT/'progress.next.json', {'phase': name, 'pid': os.getpid(), 'patient_level_output_emitted': False})
    os.replace(OUT/'progress.next.json', OUT/'progress.json')


def infer(cohort, receipt):
    m = receipt['provider']; model, transform = provider.provider(m)
    return state_bridge.infer(cohort['values'], cohort['observed'], cohort['age_triplet'],
        cohort['age_kind'], model, transform, m['binding']['transform_sha256'], batch_size=256)


def packed(result):
    return {'parent_row': np.arange(len(result.state), dtype=np.int64),
            'state': result.state, 'available': result.available,
            'clinical': result.clinical, 'clinical_mask': result.clinical_mask}


def validate_arrays(a, cohort):
    n = len(cohort['person'])
    require(set(a) == {'parent_row', 'state', 'available', 'clinical', 'clinical_mask'})
    contract = {'parent_row': ((n,), np.int64), 'state': ((n, 192), np.float32),
                'available': ((n,), bool), 'clinical': ((n, 59), np.float64),
                'clinical_mask': ((n, 59), bool)}
    for key, (shape, dtype) in contract.items():
        require(type(a[key]) is np.ndarray and a[key].shape == shape and a[key].dtype == np.dtype(dtype))
    require(np.array_equal(a['parent_row'], np.arange(n, dtype=np.int64))
            and np.isfinite(a['state']).all() and np.isfinite(a['clinical']).all()
            and (a['clinical'][~a['clinical_mask']] == 0).all()
            and np.array_equal(a['available'], a['clinical_mask'].any(1))
            and a['available'].all())
    # Rebuild only the original-unit projection: no encoder or outcome access.
    from bran_joint_lab_task_contract_v1 import project_joint_labs_to_registry
    from bran_knhanes_input_kernel_v1 import CANONICAL_NAMES, ADMITTED_CANONICAL_INDICES
    c, cm = project_joint_labs_to_registry(np.where(cohort['observed'], cohort['values'], np.nan), cohort['observed'],
        cohort['observed'].astype(np.uint8), CANONICAL_NAMES, np.zeros(59), np.ones(59))
    keep = np.zeros(59, bool); keep[list(ADMITTED_CANONICAL_INDICES)] = True
    cm = cm.copy(); cm[:, ~keep] = False; c = c.copy(); c[~cm] = 0
    require(np.array_equal(cm, a['clinical_mask']) and np.array_equal(c, a['clinical']))


def summary(a):
    return {'source_local_encoded_people_lower_bound_20': admission.e1.count(len(a['state'])),
            'state_width': 192, 'all_admitted_people_encoded': True,
            'coordinate_frame': 'V5_M_attempt2_fold0', 'checkpoint_replay_exact': True}


def authenticate(*, _pending_terminal=None):
    expected = {'protocol.json', 'aggregate.json', 'progress.json'}
    if _pending_terminal is None: expected.add('completed.json')
    require(OUT.is_dir() and not OUT.is_symlink() and {p.name for p in OUT.iterdir()} == expected)
    p, a = (admission.e1.read_closed(OUT/n) for n in ('protocol.json', 'aggregate.json'))
    t = _pending_terminal if _pending_terminal is not None else admission.e1.read_closed(OUT/'completed.json')
    require(set(p) == {'schema', 'status', 'source', 'model', 'code_sha256', 'batch_size', *FALSE_FLAGS}
            and p['schema'] == 'bran-eicu-v5-state-e3-protocol' and p['status'] == 'frozen_before_inference'
            and p['source'] == source_receipt() and p['model'] == model_receipt()
            and p['code_sha256'] == code_hashes() and p['batch_size'] == 256
            and all(p[k] is False for k in FALSE_FLAGS))
    require(set(a) == {'schema', 'status', 'summary', 'private_sha256', *FALSE_FLAGS}
            and a['schema'] == 'bran-eicu-v5-state-e3' and a['status'] == 'completed'
            and all(a[k] is False for k in FALSE_FLAGS))
    require(PRIVATE.is_dir() and not PRIVATE.is_symlink() and PRIVATE.stat().st_mode&0o777 == 0o700
            and {f.name for f in PRIVATE.iterdir()} == {'state.npz'})
    path = PRIVATE/'state.npz'
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
            and path.stat().st_mode&0o777 == 0o600 and sha(path) == a['private_sha256'])
    with np.load(path, allow_pickle=False) as h: arrays = {n: h[n] for n in h.files}
    validate_arrays(arrays, load_cohort(p['source']))
    require(a['summary'] == summary(arrays) and sha(path) == a['private_sha256'])
    require(set(t) == {'status', 'protocol_sha256', 'aggregate_sha256', 'patient_level_output_emitted'}
            and t['status'] == 'authenticated_completed' and t['patient_level_output_emitted'] is False
            and t['protocol_sha256'] == sha(OUT/'protocol.json') and t['aggregate_sha256'] == sha(OUT/'aggregate.json'))
    q = admission.e1.read_closed(OUT/'progress.json')
    require(set(q) == {'phase', 'pid', 'patient_level_output_emitted'} and q['phase'] == 'completed'
            and type(q['pid']) is int and q['pid'] > 0 and q['patient_level_output_emitted'] is False)
    return a, t


def run(state):
    require(not OUT.exists() and not OUT.is_symlink() and not PRIVATE.exists() and not PRIVATE.is_symlink())
    OUT.mkdir(); state['owned'] = True; phase('authentication')
    source, model = source_receipt(), model_receipt()
    p = {'schema': 'bran-eicu-v5-state-e3-protocol', 'status': 'frozen_before_inference',
         'source': source, 'model': model, 'code_sha256': code_hashes(), 'batch_size': 256,
         **{k: False for k in FALSE_FLAGS}}
    write_json(OUT/'protocol.json', p)
    cohort = load_cohort(source); phase('inference')
    arrays = packed(infer(cohort, model)); validate_arrays(arrays, cohort)
    phase('checkpoint_replay'); replay = packed(infer(cohort, model))
    require(all(np.array_equal(arrays[k], replay[k]) for k in arrays)); del replay
    phase('private_artifact')
    require(PRIVATE.parent.is_dir() and not PRIVATE.parent.is_symlink()); PRIVATE.mkdir(mode=0o700)
    path = PRIVATE/'state.npz'; descriptor = os.open(path, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'wb') as handle: np.savez_compressed(handle, **arrays)
    phase('post_authentication')
    require(source == source_receipt() and model == model_receipt() and p['code_sha256'] == code_hashes())
    a = {'schema': 'bran-eicu-v5-state-e3', 'status': 'completed', 'summary': summary(arrays),
         'private_sha256': sha(path), **{k: False for k in FALSE_FLAGS}}
    write_json(OUT/'aggregate.json', a); phase('completed')
    terminal = {'status': 'authenticated_completed', 'protocol_sha256': sha(OUT/'protocol.json'),
                'aggregate_sha256': sha(OUT/'aggregate.json'), 'patient_level_output_emitted': False}
    authenticate(_pending_terminal=terminal); write_json(OUT/'completed.json', terminal)


def main():
    ok = False; state = {'owned': False}
    with quiet():
        try:
            torch.set_num_threads(1)
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
