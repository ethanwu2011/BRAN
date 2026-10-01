"""Disposable, training-only V5 residual CBC pilot; local, quiet and locked.

No held-out inference, checkpoint publication, performance claims or selection.
The original V5 artifacts are immutable. A full fit requires a separate runner.
"""
import argparse
from dataclasses import asdict, is_dataclass
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json
from bran_multisource_batches_v2 import tensor
from bran_multisource_outcomes_v2 import original_age, subset_age
from bran_multisource_profiles_v3 import _validate_provider, _unchanged
from bran_multisource_inference_v2 import completion_predictions, infer_native
from bran_clinical_semantics_v1 import CBC_FIELDS
import run_bran_v5_cbc_uncertainty as source
import run_bran_context_preservation_v5 as v5
import bran_v5_residual_training as training
from bran_v5_residual_cbc import ResidualCBC

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'BRAN_V5_RESIDUAL_CBC_NEXT_EXPERIMENT.md'
CODE = tuple(sorted(set(source.CODE) | {
    'run_bran_v5_cbc_uncertainty.py', 'bran_v5_residual_cbc.py',
    'bran_v5_residual_training.py', 'test_bran_v5_residual_cbc.py',
    'test_bran_v5_residual_training.py', 'run_bran_v5_residual_pilot.py',
    'test_run_bran_v5_residual_pilot.py'}))
PARAMETERS = {
    'role': 'M', 'heldout_fold': 0, 'updates': 100, 'batch_size': 96,
    'seed': 96101, 'learning_rate': 1e-4, 'weight_decay': 1e-4,
    'gradient_clip': 5, 'residual_width': 128,
    'residual_penalty': .1, 'patterns': list(training.PATTERNS),
    'sampler': 'uniform_proper_training_rows_with_replacement',
    'context_schedule': 'six_context_cycle; independent_all_nine_single_target_cycles',
    'pilot_discarded': True, 'heldout_inference': False,
    'new_source_training': False, 'encoder_training': False,
    'patient_level_output_permitted': False, 'candidate_promoted': False,
}


def require(ok):
    if not ok:
        raise ValueError('v5_residual_pilot_contract_failed')


def protocol(evidence):
    return {'schema': 'bran-v5-residual-cbc-pilot-v1',
            'status': 'frozen_before_training', 'parameters': PARAMETERS,
            'plan_sha256': sha(PLAN),
            'code_sha256': {name: sha(ROOT / name) for name in CODE},
            'inherited_source_authentication': evidence}


def same_values(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(torch.isnan(left), torch.isnan(right))
            and torch.equal(torch.nan_to_num(left), torch.nan_to_num(right)))


def safe_receipt(receipt):
    value = asdict(receipt) if is_dataclass(receipt) else dict(receipt)
    allowed = {'attempted_updates', 'optimizer_updates', 'empty_updates',
               'elapsed_seconds', 'encoder_unchanged', 'baseline_head_unchanged',
               'residual_parameters_changed', 'sampling_schedule_sha256'}
    require(set(value) == allowed)
    require(value['attempted_updates'] == 100)
    require(type(value['optimizer_updates']) is int and 0 < value['optimizer_updates'] <= 100)
    require(type(value['empty_updates']) is int
            and value['empty_updates'] == 100 - value['optimizer_updates'])
    require(type(value['elapsed_seconds']) in (int, float)
            and np.isfinite(value['elapsed_seconds']) and value['elapsed_seconds'] >= 0)
    require(all(value[key] is True for key in
                ('encoder_unchanged', 'baseline_head_unchanged', 'residual_parameters_changed')))
    pin = value['sampling_schedule_sha256']
    require(type(pin) is str and len(pin) == 64 and set(pin) <= set('0123456789abcdef'))
    return value


def progress(out, state, phase):
    require(phase in ('source_authentication', 'initial_replay', 'pilot_training',
                     'immutability_replay', 'post_authentication'))
    state['phase'] = phase
    temporary = out / 'progress.next.json'
    write_json(temporary, {'phase': phase, 'pid': os.getpid(),
                          'patient_level_output_emitted': False})
    os.replace(temporary, out / 'progress.json')


def run(attempt, state):
    require(type(attempt) is int and 1 <= attempt <= 99)
    out = ROOT / f'BRAN_V5_RESIDUAL_CBC_PILOT_ATTEMPT{attempt}'
    require(not out.exists() and not out.is_symlink())
    out.mkdir()
    state['owned'] = out
    start = time.monotonic()
    torch.set_num_threads(2)
    progress(out, state, 'source_authentication')
    paired, _, components, evidence = source.source_context()
    frozen = protocol(evidence)
    write_json(out / 'protocol.json', frozen)
    pin = sha(out / 'protocol.json')
    fold = PARAMETERS['heldout_fold']
    component = components[('M', fold)]
    _, private = v5.paths('fit', 2)
    model, transform = v5.oldfit.load_checkpoint(
        private / f'fold{fold}_M.pt', component['checkpoint_sha256'], component['binding'])
    slots = tuple(paired.names.index(field) for field in CBC_FIELDS)
    before, transform_before, grads = _validate_provider(
        model, transform, fold, slots, paired.transforms[fold])
    rows = np.flatnonzero(paired.folds != fold)
    require(len(rows) >= 100 and set(np.unique(paired.folds[rows])) == {1, 2, 3, 4})
    clinical, cm = transform.clinical(paired.c, paired.cm)
    retinal, rm = transform.retinal(paired.r, paired.rm)
    age = original_age(paired)
    args = (tensor(clinical[rows]), tensor(cm[rows], torch.bool),
            tensor(retinal[rows]), tensor(rm[rows], torch.bool), subset_age(age, rows),
            transform.age_mean, transform.age_scale)
    # Every probe uses proper-training people only; no held-out target inspection.
    probe_rows = rows[:96]
    probe = (tensor(clinical[probe_rows]), tensor(cm[probe_rows], torch.bool),
             tensor(retinal[probe_rows]), tensor(rm[probe_rows], torch.bool),
             subset_age(age, probe_rows), transform.age_mean, transform.age_scale)
    progress(out, state, 'initial_replay')
    initial = ResidualCBC(model.cbc_joint_head, PARAMETERS['seed']).eval()
    screen_before = infer_native(model, *probe).screening_probability
    for pattern in training.PATTERNS:
        expected = completion_predictions(model, *probe, pattern, slots)
        actual = training.residual_completion_predictions(model, initial, *probe, pattern, slots)
        require(same_values(expected.cbc_standardized, actual.cbc_standardized))
        require(torch.equal(expected.targetmask, actual.targetmask)
                and torch.equal(expected.scoring_target_mask, actual.scoring_target_mask)
                and torch.equal(expected.abstained, actual.abstained))
    progress(out, state, 'pilot_training')
    head, receipt = training.train_residual(
        model, *args, slots, tensor(paired.folds[rows], torch.int64), fold,
        updates=PARAMETERS['updates'], batch_size=PARAMETERS['batch_size'])
    safe = safe_receipt(receipt)
    progress(out, state, 'immutability_replay')
    require(_unchanged(before, transform_before, grads, model, transform))
    require(same_values(screen_before, infer_native(model, *probe).screening_probability))
    require(all(torch.equal(value, head.baseline.state_dict()[key])
                for key, value in model.cbc_joint_head.state_dict().items()))
    require(any(not torch.equal(value, initial.residual.state_dict()[key])
                for key, value in head.residual.state_dict().items()))
    del head, initial  # Deliberately no checkpoint or patient-state serialization.
    progress(out, state, 'post_authentication')
    _, _, _, again = source.source_context()
    require(again == evidence and protocol(evidence) == frozen
            and sha(out / 'protocol.json') == pin)
    result = {'schema': 'bran-v5-residual-cbc-pilot-aggregate-v1',
              'status': 'pilot_completed', 'protocol_sha256': pin, 'training': safe,
              'initial_native_predictions_exact': True, 'screening_predictions_unchanged': True,
              'heldout_people_excluded': True, 'heldout_inference_performed': False,
              'pilot_checkpoint_retained': False, 'patient_level_output_emitted': False,
              'scientific_improvement_established': False, 'candidate_promoted': False,
              'projected_five_fold_fit_compute_seconds': safe['elapsed_seconds'] * 75,
              'projection_excludes_authentication_evaluation_bootstrap': True,
              'elapsed_seconds': time.monotonic() - start}
    write_json(out / 'aggregate.json', result)
    manifest = {'protocol_sha256': pin, 'aggregate_sha256': sha(out / 'aggregate.json'),
                'patient_level_output_emitted': False}
    write_json(out / 'manifest.json', manifest)
    write_json(out / 'completed.json', {'status': 'completed', **manifest,
                                      'candidate_promoted': False})
    state['owned'] = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True, type=int)
    args = parser.parse_args()
    state = {'owned': None, 'phase': 'authentication'}
    ok = False
    with quiet():
        try:
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                run(args.attempt, state)
                ok = True
        except Exception:
            out = state['owned']
            if out is not None and not (out / 'completed.json').exists():
                try:
                    write_json(out / 'failure.json', {'status': 'technical_failure',
                               'phase': state['phase'], 'patient_level_output_emitted': False})
                except Exception:
                    pass
    print(json.dumps({'status': 'completed' if ok else 'not_completed',
                      'phase': state['phase'], 'patient_level_output_emitted': False}))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
