"""Quiet, locked diagnostic after failed V2 advancement; never promotes a model."""
import argparse
import fcntl
import json
import os
from pathlib import Path

import numpy as np

from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json
from bran_multisource_protocol_v2 import digest
import bran_native_calibration_split_v1 as splitting
import run_bran_multisource_advancement_v2 as advancement
import bran_multisource_calibration_evaluation_v2 as evaluation

ROOT = Path(__file__).resolve().parent
ADVANCEMENT_PIN = '00e59359da138cbf7b6c20c34e5238dbb7a434530bda1deb18d15bf3fdbcca9e'
ADVANCEMENT_AGGREGATE_PIN = 'b1cc6cdc82e445cca1d4caf3d67050d4a6a8a2e57c2453a5240f07e2ecb394ae'
CODE = ('run_bran_multisource_calibration_v2.py', 'bran_multisource_calibration_evaluation_v2.py',
        'bran_multisource_calibration_v2.py', 'bran_multisource_calibration_metrics_v2.py',
        'bran_native_calibration_split_v1.py', 'bran_native_cbc_calibration_metrics_v1.py')
PHASES = ('calibration_inference', 'interval_fitting', 'scoring_aggregate')


def require(ok):
    if not ok: raise ValueError('multisource_calibration_runner_failed')


def paths(attempt):
    require(type(attempt) is int and 1 <= attempt <= 99)
    return ROOT / ('BRAN_MULTISOURCE_CALIBRATION_V2_ATTEMPT'+str(attempt))


def code_hashes(): return {name: sha(ROOT / name) for name in CODE}


def source_context():
    p, paired, components, references = advancement.authenticate(1, ADVANCEMENT_PIN)
    out = advancement.paths(1)
    require(sha(out / 'aggregate.json') == ADVANCEMENT_AGGREGATE_PIN)
    manifest = advancement.profiles.read(out / 'manifest.json')
    require(manifest == {'protocol_sha256': ADVANCEMENT_PIN, 'aggregate_sha256': ADVANCEMENT_AGGREGATE_PIN,
                        'patient_level_output_emitted': False})
    audit = advancement.profiles.read(out / 'audit.json')
    require(audit == {'status': 'historical_advancement_comparison_replayed',
        'protocol_sha256': ADVANCEMENT_PIN, 'aggregate_sha256': ADVANCEMENT_AGGREGATE_PIN,
        'all_aggregates_replayed': True, 'raw_reference_heads_refit': True,
        'candidate_training_repeated': False, 'historical_gate_definitions_changed': False,
        'candidate_promoted': False, 'scientific_goal_achieved': False, 'patient_level_output_emitted': False})
    a = advancement.profiles.read(out / 'aggregate.json')
    require(a['eligibility'] == {'mlp': False, 'token': False} and a['candidate_promoted'] is False
            and a['source_binding'] == p['source_binding'])
    # Reuse the authenticated legacy order only to construct the unchanged
    # patient-level split; IDs and all arrays stay inside this quiet call.
    source = advancement.references.experiment.origin.native.source
    aligned = advancement.references._align(paired, source, tuple(paired.endpoint_names), references)
    ctx, folds = aligned[:2]
    ids = list(map(str, ctx['raw_cohort'].patient_ids))
    roles = splitting.make_roles(ids, folds)
    split_auth = {'patient_order_sha256': digest(ids), 'roles_sha256': digest(roles.tolist())}
    require(split_auth == dict(references.result['calibrated_completion']['split_authentication']))
    binding = {'advancement_protocol_sha256': ADVANCEMENT_PIN,
        'advancement_aggregate_sha256': ADVANCEMENT_AGGREGATE_PIN,
        'advancement_manifest_sha256': sha(out / 'manifest.json'), 'advancement_audit_sha256': sha(out / 'audit.json'),
        'inherited_source_binding': p['source_binding'], 'split_authentication': split_auth,
        'outer_fold_sha256': paired.receipt['outer_fold_sha256'],
        'inner_fold_sha256': list(paired.receipt['inner_fold_sha256'])}
    return p, paired, components, roles, binding


def spec(binding):
    return {'schema': 'bran-multisource-cbc-calibration-protocol-v2', 'status': 'frozen_before_diagnostic',
        'source_binding': binding, 'code_sha256': code_hashes(), 'arms': list(evaluation.ARMS),
        'patterns': list(evaluation.PATTERNS), 'final_stage_c_steps': 1500,
        'alpha': .1, 'minimum_calibration': 20, 'minimum_calibration_or_scoring_role_rows_per_fold': 20,
        'split_salt': splitting.SALT, 'calibration_fraction': .5,
        'split_scope': 'same-checkpoint heldout fold only, outcome-blind half calibration and half scoring',
        'bootstrap_draws': 1000, 'bootstrap_seed': 94701, 'minimum_valid_draws': 900,
        'low_hb_definition': 'observed Hb <12 g/dL research stratum, not a clinical diagnosis',
        'coverage_small_complement_rule': 'withhold when hits or misses are 1 through 19',
        'parameter_tuning': False, 'model_selection': False, 'encoder_or_task_head_training': False,
        'diagnostic_only_after_failed_advancement': True, 'automatic_promotion': False,
        'clinical_use': False, 'patient_level_output_permitted': False}


def prepare(attempt, state):
    out = paths(attempt); require(not out.exists() and not out.is_symlink())
    _, _, _, _, binding = source_context()
    out.mkdir(); state['owned'] = out
    write_json(out / 'protocol.json', spec(binding)); state['owned'] = None
    return sha(out / 'protocol.json')


def authenticate(attempt, pin):
    out = paths(attempt)
    require(out.is_dir() and not out.is_symlink() and sha(out / 'protocol.json') == pin
            and not (out / 'failure.json').exists() and not (out / 'audit_failure.json').exists())
    saved = advancement.profiles.read(out / 'protocol.json')
    _, paired, components, roles, binding = source_context()
    require(saved == spec(binding))
    return saved, paired, components, roles


def progress(out, state, event):
    require(set(event) <= {'phase', 'arm', 'fold'} and event['phase'] in PHASES)
    if 'fold' in event: require(type(event['fold']) is int and event['fold'] in range(5))
    if 'arm' in event: require(event['arm'] in evaluation.ARMS)
    state['phase'] = event['phase']; temporary = out / 'progress.tmp'
    write_json(temporary, {**event, 'pid': os.getpid(), 'patient_level_output_emitted': False})
    os.replace(temporary, out / 'progress.json')


def compute(attempt, pin, state):
    saved, paired, components, roles = authenticate(attempt, pin)
    _, private = advancement.profiles.fit.paths(advancement.profiles.FIT_ATTEMPT)
    def provider(arm, fold):
        require(arm in evaluation.ARMS and type(fold) is int and fold in range(5))
        name = 'fold'+str(fold)+'_'+arm; item = components[name+'.json']
        require(item['binding']['arm'] == arm and item['binding']['fold'] == fold
                and item['binding']['inner_fold_sha256'] == saved['source_binding']['inner_fold_sha256'][fold])
        checkpoint = 'stage_C_step_1500.pt'
        return advancement.profiles.fit.load_checkpoint(private / name / checkpoint,
            expected_sha256=item['checkpoint_manifest'][checkpoint]['sha256'],
            binding=item['binding'], stage='C', steps=1500)
    result = evaluation.evaluate(paired, roles, provider, progress=lambda event: progress(paths(attempt), state, event))
    result['protocol_sha256'] = pin; result['source_binding'] = saved['source_binding']
    authenticate(attempt, pin)
    return result


def run(attempt, pin, state):
    out = paths(attempt); require(not any((out / name).exists() for name in ('aggregate.json', 'manifest.json')))
    state.update(owned=out, phase='diagnostic_authentication')
    result = compute(attempt, pin, state)
    write_json(out / 'aggregate.json', result)
    write_json(out / 'manifest.json', {'protocol_sha256': pin, 'aggregate_sha256': sha(out / 'aggregate.json'),
                                    'patient_level_output_emitted': False})
    state['owned'] = None


def audit(attempt, pin, state):
    out = paths(attempt); require(not any((out / name).exists() for name in ('audit.json', 'audit_failure.json')))
    state.update(owned=out, phase='diagnostic_replay', failure_name='audit_failure.json')
    value = advancement.profiles.read(out / 'aggregate.json'); aggregate_pin = sha(out / 'aggregate.json')
    require(advancement.profiles.read(out / 'manifest.json') == {'protocol_sha256': pin,
        'aggregate_sha256': aggregate_pin, 'patient_level_output_emitted': False})
    replay = compute(attempt, pin, state)
    require(value == replay and sha(out / 'aggregate.json') == aggregate_pin)
    write_json(out / 'audit.json', {'status': 'diagnostic_calibration_replayed_not_promoted',
        'protocol_sha256': pin, 'aggregate_sha256': aggregate_pin,
        'all_aggregates_replayed': True, 'split_authentication': value['source_binding']['split_authentication'],
        'radii_refit_only_on_same_calibration_roles': True, 'encoder_or_task_head_training': False,
        'failed_advancement_decisions_unchanged': True, 'candidate_promoted': False,
        'scientific_goal_achieved': False, 'patient_level_output_emitted': False})
    state['owned'] = None


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('prepare', 'run', 'audit'))
    parser.add_argument('--attempt', type=int, default=1); parser.add_argument('--protocol-sha256')
    args = parser.parse_args(); state = {'owned': None, 'phase': 'diagnostic_authentication', 'failure_name': 'failure.json'}
    answer = {'status': 'failed', 'patient_level_output_emitted': False, 'candidate_promoted': False}
    with quiet():
        try:
            with LOCK.open('a') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.action == 'prepare': pin = prepare(args.attempt, state)
                elif args.action == 'run': pin = args.protocol_sha256; run(args.attempt, pin, state)
                else: pin = args.protocol_sha256; audit(args.attempt, pin, state)
                answer.update(status=args.action+'_completed', protocol_sha256=pin)
        except Exception:
            if state['owned'] is not None:
                try:
                    write_json(state['owned'] / state['failure_name'], {'status': 'technical_failure',
                        'phase': state['phase'], 'reason': 'calibration_diagnostic_contract_failed',
                        'patient_level_output_emitted': False, 'candidate_promoted': False})
                except Exception: pass
    print(json.dumps(answer)); return int(answer['status'] == 'failed')


if __name__ == '__main__': raise SystemExit(main())
