"""Frozen V5 residual CBC fit/evaluation lifecycle; private checkpoints only."""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import run_bran_v5_residual_pilot as pilot
import run_bran_v5_cbc_uncertainty as source
import run_bran_context_preservation_v5 as v5
import bran_v5_residual_training as training
from bran_v5_residual_cbc import ResidualCBC
from bran_multisource_batches_v2 import tensor, transform_hash
from bran_multisource_outcomes_v2 import original_age, subset_age
from bran_multisource_profiles_v3 import _validate_provider, _unchanged
from bran_multisource_inference_v2 import infer_native
from bran_clinical_semantics_v1 import CBC_FIELDS
from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json

ROOT = Path(__file__).resolve().parent
FIT_CODE = tuple(sorted(set(pilot.CODE) | {
    'run_bran_v5_residual_cbc.py', 'test_run_bran_v5_residual_cbc.py'}))
PARAMETERS = {
    'role': 'M', 'folds': list(range(5)), 'updates_per_fold': 1500,
    'batch_size': 96, 'seed_base': 96101, 'learning_rate': 1e-4,
    'weight_decay': 1e-4, 'gradient_clip': 5, 'residual_penalty': .1,
    'state_dimension': 192, 'residual_hidden': 128, 'output_dimension': 9,
    'patterns': list(training.PATTERNS),
    'encoder_frozen': True, 'native_screening_frozen': True, 'native_cbc_frozen': True,
    'sampler': 'uniform_proper_training_rows_with_replacement',
    'schedule': 'six_context_cycle_independent_single_target_counters',
    'new_source_learning': False, 'protected_external_used': False,
    'patient_level_output_permitted': False, 'automatic_promotion': False,
}
EVALUATION = {
    'fullfold_bootstrap_draws': 1000, 'fullfold_bootstrap_seed': 96199,
    'minimum_valid_draws': 900, 'interval_level': .9,
    'interval_bootstrap_seed': 94701,
    'calibration_split': 'inherited_outcome_blind_half_within_each_outer_fold',
    'primary': 'whole_cbc_hidden_equal_nine_field_fold_iqr_mae_R_minus_V5_ci_upper_below_zero',
    'protected_point_nonincrease': ['single_target_hidden', 'single_target_no_retina', 'whole_cbc_no_retina'],
    'low_hb_research_stratum': 'observed hemoglobin below 12 g/dL; not a universal clinical definition',
    'historical_gates_changed': False, 'automatic_promotion': False,
}


def require(ok):
    if not ok:
        raise ValueError('v5_residual_lifecycle_contract_failed')


def paths(stage, attempt):
    require(stage in ('fit', 'evaluate') and type(attempt) is int and 1 <= attempt <= 99)
    name = f'BRAN_V5_RESIDUAL_CBC_{stage.upper()}_ATTEMPT{attempt}'
    return ROOT/name, ROOT/'private_artifacts'/name.lower()


def code_hashes(stage):
    names = set(FIT_CODE)
    if stage == 'evaluate':
        import bran_v5_residual_evaluation as evaluator
        names.update(evaluator.CODE_DEPENDENCIES)
        names.update(('bran_v5_residual_evaluation.py', 'test_bran_v5_residual_evaluation.py'))
    return {name: sha(ROOT/name) for name in sorted(names)}


def authenticate_pilot():
    out = ROOT/'BRAN_V5_RESIDUAL_CBC_PILOT_ATTEMPT1'
    require(not (out/'failure.json').exists())
    p = json.loads((out/'protocol.json').read_text())
    a = json.loads((out/'aggregate.json').read_text())
    m = json.loads((out/'manifest.json').read_text())
    t = json.loads((out/'completed.json').read_text())
    require(p == pilot.protocol(p['inherited_source_authentication']))
    require(t['status'] == 'completed' and a['status'] == 'pilot_completed')
    require(a['protocol_sha256'] == m['protocol_sha256'] == t['protocol_sha256'] == sha(out/'protocol.json'))
    require(m['aggregate_sha256'] == t['aggregate_sha256'] == sha(out/'aggregate.json'))
    pilot.safe_receipt(a['training'])
    require(a['initial_native_predictions_exact'] and a['screening_predictions_unchanged']
            and a['heldout_people_excluded'] and not a['heldout_inference_performed']
            and not a['pilot_checkpoint_retained'] and not a['patient_level_output_emitted'])
    return {'protocol_sha256': m['protocol_sha256'], 'aggregate_sha256': m['aggregate_sha256']}


def fit_protocol(evidence):
    return {'schema': 'bran-v5-residual-cbc-fit-protocol-v1', 'status': 'frozen_before_training',
            'parameters': PARAMETERS, 'evaluation_plan': EVALUATION,
            'plan_sha256': sha(pilot.PLAN), 'pilot': authenticate_pilot(),
            'code_sha256': code_hashes('fit'), 'source_binding': evidence}


def validate_receipt(receipt):
    value = asdict(receipt) if hasattr(receipt, '__dataclass_fields__') else dict(receipt)
    # Validate every field through the already frozen pilot release boundary,
    # with its numerical budget translated solely for this structural check.
    require(set(value) == {'attempted_updates', 'optimizer_updates', 'empty_updates',
                          'elapsed_seconds', 'encoder_unchanged', 'baseline_head_unchanged',
                          'residual_parameters_changed', 'sampling_schedule_sha256'})
    require(value['attempted_updates'] == 1500
            and type(value['optimizer_updates']) is int and 0 < value['optimizer_updates'] <= 1500
            and type(value['empty_updates']) is int
            and value['empty_updates'] == 1500-value['optimizer_updates'])
    structural = {**value, 'attempted_updates': 100, 'optimizer_updates': 100, 'empty_updates': 0}
    pilot.safe_receipt(structural)
    return value


def head_binding(fold, item, transform, protocol_pin):
    require(type(fold) is int and fold in range(5)
            and item['binding']['fold'] == fold and item['binding']['role'] == 'M'
            and transform.heldout_fold == fold)
    return {'fold': fold, 'role': 'M', 'teacher_checkpoint_sha256': item['checkpoint_sha256'],
            'teacher_binding': item['binding'], 'transform_sha256': transform_hash(transform),
            'fit_protocol_sha256': protocol_pin, 'seed': 96101+fold,
            'architecture': '192-128-GELU-9-additive-to-frozen-native-head',
            'updates': 1500}


def _bound(head, teacher):
    require(isinstance(head, ResidualCBC) and not head.training)
    require(all(torch.equal(t, head.baseline.state_dict()[k]) for k, t in teacher.cbc_joint_head.state_dict().items()))
    require(all(not t.is_floating_point() or bool(torch.isfinite(t).all()) for t in head.state_dict().values()))


def teacher_state_hash(teacher):
    """Private weight fingerprint supplements the file/fold binding at reload."""
    digest = hashlib.sha256()
    for key, tensor_value in sorted(teacher.state_dict().items()):
        value = tensor_value.detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def save_head(path, head, teacher, binding):
    _bound(head, teacher)
    require(not path.exists() and not path.is_symlink())
    payload = {'schema': 'bran-v5-residual-cbc-checkpoint-v1', 'binding': binding,
               'teacher_state_sha256': teacher_state_hash(teacher),
               'residual_state_dict': {k: t.detach().cpu().clone() for k, t in head.residual.state_dict().items()}}
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(), 0o600)
        torch.save(payload, stream)
        stream.flush(); os.fsync(stream.fileno())
    return sha(path)


def load_head(path, expected_sha, teacher, binding):
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
            and path.stat().st_mode & 0o777 == 0o600 and sha(path) == expected_sha)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(type(payload) is dict and set(payload) == {'schema', 'binding', 'residual_state_dict', 'teacher_state_sha256'}
            and payload['schema'] == 'bran-v5-residual-cbc-checkpoint-v1'
            and payload['binding'] == binding and payload['teacher_state_sha256'] == teacher_state_hash(teacher))
    require(binding['role'] == 'M' and type(binding['fold']) is int and binding['fold'] in range(5)
            and binding['updates'] == 1500 and binding['seed'] == 96101+binding['fold'])
    head = ResidualCBC(teacher.cbc_joint_head, binding['seed'])
    head.residual.load_state_dict(payload['residual_state_dict'], strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    _bound(head, teacher)
    require(sha(path) == expected_sha)
    return head


def progress(out, state, phase, fold=None):
    require(phase in ('source_authentication', 'fit', 'checkpoint_replay', 'post_authentication',
                     'inference', 'interval_fitting', 'aggregate_bootstrap'))
    require(fold is None or type(fold) is int and fold in range(5))
    state['phase'] = phase
    value = {'phase': phase, 'pid': os.getpid(), 'patient_level_output_emitted': False}
    if fold is not None: value['fold'] = fold
    temporary = out/'progress.next.json'
    write_json(temporary, value); os.replace(temporary, out/'progress.json')


def create_attempt(stage, attempt, state):
    out, private = paths(stage, attempt)
    require(not out.exists() and not out.is_symlink() and not private.exists() and not private.is_symlink())
    out.mkdir(); state['owned'] = out
    if stage == 'fit':
        require(private.parent.is_dir() and not private.parent.is_symlink())
        private.mkdir(mode=0o700)
    return out, private


def finish(out, aggregate, protocol_pin, state):
    write_json(out/'aggregate.json', aggregate)
    manifest = {'protocol_sha256': protocol_pin, 'aggregate_sha256': sha(out/'aggregate.json'),
                'patient_level_output_emitted': False}
    write_json(out/'manifest.json', manifest)
    write_json(out/'completed.json', {'status': 'completed', **manifest,
                                      'candidate_promoted': False, 'scientific_goal_achieved': False})
    state['owned'] = None


def fit(attempt, state):
    out, private = create_attempt('fit', attempt, state)
    start = time.monotonic(); torch.set_num_threads(2)
    progress(out, state, 'source_authentication')
    paired, _, components, evidence = source.source_context()
    frozen = fit_protocol(evidence)
    write_json(out/'protocol.json', frozen); pin = sha(out/'protocol.json')
    slots = tuple(paired.names.index(field) for field in CBC_FIELDS)
    age = original_age(paired)
    _, teacher_private = v5.paths('fit', 2)
    receipts = {}
    for fold in range(5):
        progress(out, state, 'fit', fold)
        item = components[('M', fold)]
        teacher, transform = v5.oldfit.load_checkpoint(
            teacher_private/f'fold{fold}_M.pt', item['checkpoint_sha256'], item['binding'])
        before, th, grads = _validate_provider(teacher, transform, fold, slots, paired.transforms[fold])
        rows = np.flatnonzero(paired.folds != fold)
        require(len(rows) >= 100 and set(np.unique(paired.folds[rows])) == set(range(5))-{fold})
        c, cm = transform.clinical(paired.c, paired.cm)
        r, rm = transform.retinal(paired.r, paired.rm)
        args = (tensor(c[rows]), tensor(cm[rows], torch.bool), tensor(r[rows]), tensor(rm[rows], torch.bool),
                subset_age(age, rows), transform.age_mean, transform.age_scale)
        probe_rows = rows[:96]
        probe = (tensor(c[probe_rows]), tensor(cm[probe_rows], torch.bool), tensor(r[probe_rows]),
                 tensor(rm[probe_rows], torch.bool), subset_age(age, probe_rows), transform.age_mean, transform.age_scale)
        screening = infer_native(teacher, *probe).screening_probability
        head, receipt = training.train_residual(teacher, *args, slots,
            tensor(paired.folds[rows], torch.int64), fold, updates=1500, batch_size=96)
        safe = validate_receipt(receipt)
        require(_unchanged(before, th, grads, teacher, transform)
                and pilot.same_values(screening, infer_native(teacher, *probe).screening_probability))
        bound = head_binding(fold, item, transform, pin)
        checkpoint = private/f'fold{fold}_residual.pt'
        checkpoint_pin = save_head(checkpoint, head, teacher, bound)
        progress(out, state, 'checkpoint_replay', fold)
        restored = load_head(checkpoint, checkpoint_pin, teacher, bound)
        for pattern in training.PATTERNS:
            first = training.residual_completion_predictions(teacher, head, *probe, pattern, slots)
            second = training.residual_completion_predictions(teacher, restored, *probe, pattern, slots)
            require(pilot.same_values(first.cbc_standardized, second.cbc_standardized)
                    and torch.equal(first.targetmask, second.targetmask)
                    and torch.equal(first.scoring_target_mask, second.scoring_target_mask)
                    and torch.equal(first.abstained, second.abstained))
        row = {'fold': fold, 'binding': bound, 'checkpoint_sha256': checkpoint_pin,
               'training': safe, 'checkpoint_replay_exact': True, 'screening_unchanged': True,
               'heldout_inference_performed': False, 'patient_level_output_emitted': False}
        name = f'fold{fold}.json'
        write_json(out/name, row); receipts[name] = sha(out/name)
    progress(out, state, 'post_authentication')
    _, _, _, replay_evidence = source.source_context()
    require(evidence == replay_evidence and frozen == fit_protocol(evidence)
            and pin == sha(out/'protocol.json'))
    result = {'schema': 'bran-v5-residual-cbc-fit-v1', 'status': 'completed',
              'protocol_sha256': pin, 'component_sha256': receipts, 'updates_total': 7500,
              'five_checkpoints_replayed': True, 'screening_unchanged': True,
              'heldout_inference_performed': False, 'candidate_promoted': False,
              'patient_level_output_emitted': False, 'elapsed_seconds': time.monotonic()-start}
    finish(out, result, pin, state)


def authenticate_fit(attempt):
    out, private = paths('fit', attempt)
    require(not (out/'failure.json').exists() and private.is_dir() and not private.is_symlink()
            and private.stat().st_mode & 0o777 == 0o700)
    p = json.loads((out/'protocol.json').read_text())
    a = json.loads((out/'aggregate.json').read_text())
    m = json.loads((out/'manifest.json').read_text())
    t = json.loads((out/'completed.json').read_text())
    require(p == fit_protocol(p['source_binding']) and t['status'] == a['status'] == 'completed')
    require(a['protocol_sha256'] == m['protocol_sha256'] == t['protocol_sha256'] == sha(out/'protocol.json'))
    require(m['aggregate_sha256'] == t['aggregate_sha256'] == sha(out/'aggregate.json'))
    require(a['updates_total'] == 7500 and a['five_checkpoints_replayed'] and a['screening_unchanged']
            and not a['patient_level_output_emitted'] and not a['heldout_inference_performed'])
    require(set(a['component_sha256']) == {f'fold{f}.json' for f in range(5)})
    rows = {}
    for fold in range(5):
        name = f'fold{fold}.json'
        require(sha(out/name) == a['component_sha256'][name])
        item = json.loads((out/name).read_text())
        require(item['fold'] == item['binding']['fold'] == fold
                and item['binding']['fit_protocol_sha256'] == a['protocol_sha256']
                and item['checkpoint_replay_exact'] and item['screening_unchanged']
                and not item['heldout_inference_performed'] and not item['patient_level_output_emitted'])
        validate_receipt(item['training'])
        require(sha(private/f'fold{fold}_residual.pt') == item['checkpoint_sha256'])
        rows[fold] = item
    return p, a, rows


def evaluate(attempt, fit_attempt, state):
    import bran_v5_residual_evaluation as evaluator
    out, _ = create_attempt('evaluate', attempt, state)
    start = time.monotonic(); torch.set_num_threads(2)
    progress(out, state, 'source_authentication')
    fp, fa, heads = authenticate_fit(fit_attempt)
    paired, roles, components, evidence = source.source_context()
    require(fp['source_binding'] == evidence)
    fit_out, fit_private = paths('fit', fit_attempt)
    frozen = {'schema': 'bran-v5-residual-cbc-evaluation-protocol-v1', 'status': 'frozen_before_inference',
              'parameters': EVALUATION, 'fit_protocol_sha256': fa['protocol_sha256'],
              'fit_aggregate_sha256': sha(fit_out/'aggregate.json'),
              'source_binding': evidence, 'code_sha256': code_hashes('evaluate')}
    write_json(out/'protocol.json', frozen); pin = sha(out/'protocol.json')
    _, teacher_private = v5.paths('fit', 2)

    def provider(fold):
        require(type(fold) is int and fold in range(5))
        item = components[('M', fold)]
        teacher, transform = v5.oldfit.load_checkpoint(
            teacher_private/f'fold{fold}_M.pt', item['checkpoint_sha256'], item['binding'])
        expected = head_binding(fold, item, transform, fa['protocol_sha256'])
        require(heads[fold]['binding'] == expected)
        head = load_head(fit_private/f'fold{fold}_residual.pt', heads[fold]['checkpoint_sha256'], teacher, expected)
        return teacher, transform, head

    def notify(event):
        require(type(event) is dict and set(event) <= {'phase', 'fold'} and 'phase' in event)
        progress(out, state, event['phase'], event.get('fold'))

    result = evaluator.evaluate(paired, roles, provider, progress=notify)
    evaluator.validate_result(result)
    progress(out, state, 'post_authentication')
    _, _, _, replay_evidence = source.source_context()
    require(replay_evidence == evidence and code_hashes('evaluate') == frozen['code_sha256']
            and authenticate_fit(fit_attempt)[:2] == (fp, fa) and sha(out/'protocol.json') == pin)
    aggregate = {'schema': 'bran-v5-residual-cbc-evaluation-v1', 'status': 'completed',
                 'protocol_sha256': pin, 'source_binding': evidence, 'evaluation': result,
                 'patient_level_output_emitted': False, 'candidate_promoted': False,
                 'elapsed_seconds': time.monotonic()-start}
    finish(out, aggregate, pin, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('fit', 'evaluate'))
    parser.add_argument('--attempt', required=True, type=int)
    parser.add_argument('--fit-attempt', default=1, type=int)
    args = parser.parse_args()
    state = {'phase': 'authentication', 'owned': None}; ok = False
    with quiet():
        try:
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.stage == 'fit': fit(args.attempt, state)
                else: evaluate(args.attempt, args.fit_attempt, state)
                ok = True
        except Exception:
            if state['owned'] is not None and not (state['owned']/'completed.json').exists():
                try:
                    write_json(state['owned']/'failure.json', {'status': 'technical_failure',
                               'phase': state['phase'], 'patient_level_output_emitted': False})
                except Exception:
                    pass
    print(json.dumps({'status': 'completed' if ok else 'not_completed', 'stage': args.stage,
                      'phase': state['phase'], 'patient_level_output_emitted': False}))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
