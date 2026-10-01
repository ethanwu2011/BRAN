"""Quiet local-only R7 native screening head interpretation; no fitting."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json

ROOT = Path(__file__).resolve().parent
ERROR = 'bran_r7_head_blocks_i1_failed'
CODE = ('BRAN_R7_HEAD_BLOCKS_I1_DESIGN.md', 'bran_r7_head_blocks_i1.py',
        'test_bran_r7_head_blocks_i1.py', 'run_bran_r7_head_blocks_i1.py',
        'test_run_bran_r7_head_blocks_i1.py', 'run_bran_r7_head_blocks_i1_attempt1.sh')
PHASES = ('authentication', 'source_loading', 'state_inference', 'aggregate_summary',
          'post_authentication', 'completed')


def require(ok):
    if not ok:
        raise ValueError(ERROR) from None


def paths(attempt=1):
    require(type(attempt) is int and attempt == 1)
    return ROOT / 'BRAN_R7_HEAD_BLOCKS_I1_ATTEMPT1'


def read(path):
    from run_bran_r7_fixed_state_p1 import _read
    return _read(path)


def authenticate():
    import audit_bran_robust_clinical_r7_v2 as r7audit
    import run_bran_r7_fixed_state_p1 as p1
    fit, _, components, receipt = r7audit.authenticate('fit', 1)
    p1receipt = p1.authenticate(1)
    pp = read(p1.paths(1) / 'protocol.json')
    require(pp['r7_fit_receipt'] == receipt
            and pp['r7_source_binding'] == fit['source_binding'])
    p1.require_checkpoint_manifest_match(components, 'R', fit['source_binding'],
                                        pp['r7_checkpoint_manifest'])
    for name, pin in pp['code_sha256'].items():
        require(sha(ROOT / name) == pin)
    return {'source_binding': fit['source_binding'], 'checkpoint_manifest': pp['r7_checkpoint_manifest'],
            'r7_fit_receipt': receipt, 'p1_receipt': p1receipt,
            'code_sha256': {name: sha(ROOT/name) for name in sorted(set(CODE) | set(pp['code_sha256']))},
            'patient_level_output_emitted': False}


def sources(binding):
    from bran_multisource_binding_v3 import load_bound_sources
    value = load_bound_sources()
    require(value.receipt() == binding['source_binding'])
    return value


def evaluate(bound, binding, callback):
    import numpy as np
    import torch
    import bran_r7_fixed_state_p1 as p1
    import bran_r7_head_blocks_i1 as kernel
    import run_bran_robust_clinical_r7 as r7
    from bran_clinical_semantics_v1 import CBC_FIELDS
    from bran_multisource_batches_v2 import tensor
    from bran_multisource_outcomes_v2 import original_age
    from bran_v5_state_routes import state_routes
    torch.set_num_threads(2)
    paired = bound.paired
    folds = np.asarray(paired.folds)
    p1.validate_evaluation_inputs(paired, lambda *args: None)
    age = original_age(paired)
    slots = tuple(paired.names.index(field) for field in CBC_FIELDS)
    contributions = np.full((len(folds), 26, 3), np.nan, dtype=np.float64)
    availability = np.zeros(len(folds), dtype=bool)
    _, private = r7.paths('fit', 1)
    for fold in range(5):
        callback('state_inference', fold)
        item = binding['checkpoint_manifest']['folds'][fold]
        require(item['fold'] == fold)
        def load():
            return r7.load_checkpoint(private/f'fold{fold}_R.pt', item['checkpoint_sha256'], item['binding'])
        model, transform = load()
        before, pin, gradients = p1._validate_r7(model, transform, fold, slots, paired.transforms[fold])
        c, cm = transform.clinical(paired.c, paired.cm)
        r, rm = transform.retinal(paired.r, paired.rm)
        args = (tensor(c), tensor(cm, torch.bool), tensor(r), tensor(rm, torch.bool),
                age, transform.age_mean, transform.age_scale)
        state = state_routes(model, *args)
        require(p1._role_unchanged(before, pin, gradients, model, transform))
        replay, replay_transform = load()
        replay_before, replay_pin, replay_gradients = p1._validate_r7(
            replay, replay_transform, fold, slots, paired.transforms[fold])
        state_again = state_routes(replay, *args)
        require(replay is not model and p1.replay_equal(state, state_again)
                and p1._role_unchanged(replay_before, replay_pin, replay_gradients, replay, replay_transform))
        head = model.screening_joint_head
        require(type(head) is torch.nn.Linear and head.in_features == 192 and head.out_features == 26
                and head.bias is not None)
        x = state.states['both'].detach().cpu().numpy().astype(np.float64)
        available = state.available['both'].detach().cpu().numpy()
        w = head.weight.detach().cpu().numpy().astype(np.float64)
        b = head.bias.detach().cpu().numpy().astype(np.float64)
        unfolded = kernel.unfold(x, w,
            model.retinal_prior.weight.detach().cpu().numpy().astype(np.float64),
            model.clinical_prior.weight.detach().cpu().numpy().astype(np.float64))
        parts = kernel.decompose(unfolded.states, available, folds != fold, unfolded.weight, b)
        with torch.no_grad():
            actual = head(state.states['both']).detach().cpu().numpy().astype(np.float64)
        reconstructed = parts.reference_logit[None, :] + parts.blocks.sum(axis=2)
        require(np.allclose(reconstructed[available], actual[available], rtol=1e-5, atol=1e-5))
        heldout = folds == fold
        contributions[heldout] = parts.blocks[heldout]
        availability[heldout] = available[heldout]
        del model, replay, state, state_again, parts, unfolded, actual, reconstructed, x, w, b
    callback('aggregate_summary', None)
    result = kernel.summarize(contributions, paired.labels, paired.labelmask,
                              availability, folds, tuple(paired.endpoint_names))
    require(result == kernel.summarize(contributions, paired.labels, paired.labelmask,
                                      availability, folds, tuple(paired.endpoint_names)))
    kernel.validate_result(result)
    return result


def progress(out, state, phase, fold=None):
    require(phase in PHASES and (fold is None or type(fold) is int and fold in range(5)))
    state['phase'] = phase
    write_json(out/'progress.next.json', {'phase': phase, 'fold': fold,
        'patient_level_output_emitted': False})
    os.replace(out/'progress.next.json', out/'progress.json')


def validate_aggregate(result):
    import bran_r7_head_blocks_i1 as kernel
    require(type(result) is dict and set(result) == {'schema','protocol_sha256','report',
        'state_replay_equal','aggregate_replay_equal','native_head_identity_checked',
        'encoder_updated','patient_level_output_emitted'}
        and result['schema'] == 'bran-r7-head-blocks-i1-aggregate-v1'
        and type(result['protocol_sha256']) is str and len(result['protocol_sha256']) == 64
        and result['state_replay_equal'] is True and result['aggregate_replay_equal'] is True
        and result['native_head_identity_checked'] is True and result['encoder_updated'] is False
        and result['patient_level_output_emitted'] is False)
    kernel.validate_result(result['report'])


def run(attempt=1):
    out = paths(attempt)
    require(not out.exists() and not out.is_symlink())
    with LOCK.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        out.mkdir(); state = {'phase': 'authentication'}
        try:
            with quiet():
                cb = lambda phase, fold=None: progress(out, state, phase, fold)
                cb('authentication'); binding = authenticate()
                protocol = {'schema': 'bran-r7-head-blocks-i1-protocol-v1', 'binding': binding,
                    'frozen_before_inference': True, 'patient_level_output_emitted': False}
                write_json(out/'protocol.json', protocol); protocol_pin = sha(out/'protocol.json')
                cb('source_loading'); bound = sources(binding)
                report = evaluate(bound, binding, cb)
                cb('post_authentication')
                require(authenticate() == binding and sources(binding).receipt() == binding['source_binding']
                        and sha(out/'protocol.json') == protocol_pin)
                result = {'schema': 'bran-r7-head-blocks-i1-aggregate-v1', 'protocol_sha256': protocol_pin,
                    'report': report, 'state_replay_equal': True, 'aggregate_replay_equal': True,
                    'native_head_identity_checked': True, 'encoder_updated': False,
                    'patient_level_output_emitted': False}
                validate_aggregate(result); write_json(out/'aggregate.json', result); cb('completed')
                write_json(out/'completed.json', {'status': 'authenticated_completed',
                    'protocol_sha256': protocol_pin, 'aggregate_sha256': sha(out/'aggregate.json'),
                    'patient_level_output_emitted': False})
            return {'status': 'completed_pending_separate_audit', 'patient_level_output_emitted': False}
        except BaseException:
            require(not (out/'completed.json').exists())
            write_json(out/'failure.json', {'status': 'technical_failure', 'phase': state['phase'],
                'error_code': ERROR, 'patient_level_output_emitted': False})
            return {'status': 'technical_failure', 'phase': state['phase'], 'patient_level_output_emitted': False}


def audit(attempt=1, replay=True):
    with LOCK.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        with quiet():
            out = paths(attempt)
            require(out.is_dir() and not out.is_symlink()
                and {p.name for p in out.iterdir()} == {'protocol.json','aggregate.json','completed.json','progress.json'}
                and all(p.is_file() and not p.is_symlink() and p.stat().st_nlink == 1 for p in out.iterdir()))
            protocol, result, done = (read(out/name) for name in ('protocol.json','aggregate.json','completed.json'))
            pin = sha(out/'protocol.json')
            require(set(protocol) == {'schema','binding','frozen_before_inference','patient_level_output_emitted'}
                and protocol['schema'] == 'bran-r7-head-blocks-i1-protocol-v1'
                and protocol['frozen_before_inference'] is True and protocol['patient_level_output_emitted'] is False
                and done == {'status':'authenticated_completed','protocol_sha256':pin,
                    'aggregate_sha256':sha(out/'aggregate.json'),'patient_level_output_emitted':False}
                and result['protocol_sha256'] == pin
                and read(out/'progress.json') == {'phase':'completed','fold':None,'patient_level_output_emitted':False})
            validate_aggregate(result)
            require(authenticate() == protocol['binding'])
            bound = sources(protocol['binding'])
            if replay:
                require(evaluate(bound, protocol['binding'], lambda *args: None) == result['report'])
                require(authenticate() == protocol['binding'])
                sources(protocol['binding'])
            return {'status':'aggregate_terminal_authenticated','protocol_sha256':pin,
                'aggregate_sha256':done['aggregate_sha256'],'independent_source_inference_replay':replay,
                'patient_level_output_emitted':False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--audit',action='store_true'); args = parser.parse_args()
    try:
        with quiet():
            result = audit(args.attempt) if args.audit else run(args.attempt)
        print(json.dumps(result,sort_keys=True))
        return 0 if result['status'] in ('completed_pending_separate_audit','aggregate_terminal_authenticated') else 1
    except BaseException:
        print(json.dumps({'status':'closed_failure','patient_level_output_emitted':False})); return 1


if __name__ == '__main__':
    raise SystemExit(main())
