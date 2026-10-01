"""Fixed 100-update local correctness/throughput pilot; no heldout scoring.

Pilot models are discarded. Only closed scalar/boolean receipts are emitted.
"""
import argparse
import fcntl
import json
from pathlib import Path
import time

import torch

from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json
from bran_multisource_binding_v3 import load_bound_sources, bind_fold
from bran_multisource_pilot_core_v3 import run_pilot
from run_bran_multisource_fit_v2 import CODE as LEGACY_CODE

ROOT = Path(__file__).resolve().parent
PREFLIGHT = ROOT/'BRAN_MULTISOURCE_WARMSTART_PREFLIGHT_V3_ATTEMPT3'
CODE = tuple(sorted(set(LEGACY_CODE) | {
    'run_bran_multisource_pilot_v3.py', 'bran_multisource_binding_v3.py',
    'bran_multisource_pilot_core_v3.py', 'bran_multisource_continuation_v3.py',
    'bran_multisource_warmstart_v3.py', 'bran_multisource_native_transform_v3.py',
    'bran_multisource_anchored_model_v3.py', 'bran_patient_state_anchor_v2.py',
    'bran_multisource_reference_roles_v2.py', 'bran_multisource_reference_predictions_v2.py'}))


def require(ok):
    if not ok: raise ValueError('multisource_pilot_failed')


def code_hashes():
    return {name: sha(ROOT/name) for name in CODE}


def authenticate_preflight():
    require(not (PREFLIGHT/'failure.json').exists())
    manifest = json.loads((PREFLIGHT/'manifest.json').read_text())
    aggregate = json.loads((PREFLIGHT/'aggregate.json').read_text())
    require(manifest['aggregate_sha256'] == sha(PREFLIGHT/'aggregate.json')
        and manifest['protocol_sha256'] == sha(PREFLIGHT/'protocol.json')
        and all(sha(ROOT/name) == digest for name, digest in manifest['code_sha256'].items())
        and aggregate['status'] == 'reported_age_replay_passed'
        and len(aggregate['folds']) == 5 and aggregate['patient_level_output_emitted'] is False)
    return sha(PREFLIGHT/'manifest.json')


def execute(out, state):
    state['phase'] = 'source_binding'
    code, preflight = code_hashes(), authenticate_preflight()
    sources = load_bound_sources()
    source_receipt = sources.receipt()
    protocol = {'schema': 'bran-multisource-pilot-v3', 'steps_per_role': 100,
        'roles': ['C', 'M'], 'fold': 0, 'seed': 95101, 'paired_batch': 96, 'source_batch': 128,
        'learning_rate': 5e-5, 'weight_decay': 1e-4, 'clip': 5.,
        'code_sha256': code, 'source_binding': source_receipt,
        'preflight_manifest_sha256': preflight, 'heldout_evaluation_permitted': False,
        'pilot_checkpoint_retention_permitted': False, 'parameter_tuning_permitted': False,
        'source_data_must_remain_local': True}
    write_json(out/'protocol.json', protocol)
    state['phase'] = 'fold_zero_binding'
    bound = bind_fold(sources, 0)
    state['phase'] = 'fixed_training_pilot'
    torch.set_num_threads(2)
    result = run_pilot(bound.model, bound.teacher, bound.paired_factory, bound.source_factory,
        steps=100, age_mean=bound.transform.age_mean, age_scale=bound.transform.age_scale,
        seed=bound.seed, cbc_indices=bound.cbc_indices, state_scale=bound.state_scale)
    del result['control_model'], result['multisource_model']
    require(set(result) == {'elapsed_seconds', 'control_updates', 'candidate_updates',
        'source_generative_supported', 'source_cbc_supported', 'paired_batches_equivalent',
        'paired_mask_streams_equivalent'})
    require(result['control_updates'] == result['candidate_updates'] == 100
        and all(result[k] is True for k in ('source_generative_supported', 'source_cbc_supported',
             'paired_batches_equivalent', 'paired_mask_streams_equivalent')))
    state['phase'] = 'post_pilot_authentication'
    require(code_hashes() == code and authenticate_preflight() == preflight)
    require(load_bound_sources().receipt() == source_receipt)
    aggregate = {**result, 'schema': 'bran-multisource-pilot-v3', 'status': 'pilot_passed',
        'full_30000_update_kernel_seconds_projection': result['elapsed_seconds']*150,
        'projection_excludes_source_loading_checkpoints_and_evaluation': True,
        'pilot_models_discarded': True, 'heldout_scoring_performed': False,
        'candidate_promoted': False, 'scientific_goal_achieved': False,
        'patient_level_output_emitted': False}
    write_json(out/'aggregate.json', aggregate)
    write_json(out/'manifest.json', {'aggregate_sha256': sha(out/'aggregate.json'),
        'protocol_sha256': sha(out/'protocol.json'), 'code_sha256': code,
        'patient_level_output_emitted': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    require(type(args.attempt) is int and 1 <= args.attempt <= 99)
    out = ROOT/f'BRAN_MULTISOURCE_PILOT_V3_ATTEMPT{args.attempt}'
    require(not out.exists() and not out.is_symlink())
    state, ok = {'phase': 'lock_acquisition'}, False
    with quiet():
        try:
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                out.mkdir()
                execute(out, state)
                ok = True
        except Exception:
            if out.is_dir() and not (out/'aggregate.json').exists():
                write_json(out/'failure.json', {'status': 'technical_pilot_failure',
                    'phase': state['phase'], 'patient_level_output_emitted': False,
                    'candidate_promoted': False})
    print(json.dumps({'status': 'pilot_passed' if ok else 'pilot_not_passed',
        'phase': state['phase'], 'patient_level_output_emitted': False}))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
