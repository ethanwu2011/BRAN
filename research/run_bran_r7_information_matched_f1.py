"""Quiet local R7 application of the sealed 14-arm FM benchmark recipe.

No patient arrays, vectors, predictions, probes or draws are serialized. The
legacy adapter supplies authenticated cache/raw designs, never an old BRAN
state. All new inference uses the already selected five R7 checkpoints.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits

import bran_information_matched_fm_v2 as bench
import run_bran_information_matched_fm_v2 as old
import run_bran_v5_information_matched_fm as bridge
import run_bran_v5_cbc_uncertainty as source
import run_bran_robust_clinical_r7 as r7
import run_bran_r7_fixed_state_p1 as p1
import bran_r7_fixed_state_p1 as reuse
from audit_bran_robust_clinical_r7_v2 import authenticate as authenticate_fit
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_multisource_batches_v2 import tensor
from bran_multisource_outcomes_v2 import original_age
from bran_multisource_profiles_v3 import _unchanged
from bran_v5_state_routes import ROUTES, state_routes
from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json

ROOT = Path(__file__).resolve().parent
ERROR = 'bran_r7_information_matched_f1_failed'
PLAN = 'BRAN_R7_INFORMATION_MATCHED_F1_DESIGN.md'
CODE = (PLAN, 'run_bran_r7_information_matched_f1.py',
        'test_run_bran_r7_information_matched_f1.py',
        'run_bran_r7_information_matched_f1_attempt1.sh')
P1_PINS = {'protocol.json': '8a8809cc15cbfbfe07bd8966cbc67b1da5bcc2cb6c83b88a30a4f96873b3511f',
           'aggregate.json': '2774699906b6d7d03847ccae42b278924e47344504128212c6427aca390002f2'}
PARAMETERS = {**bridge.PARAMETERS,
    'bran_representation': 'frozen_R7_R_target_free_state_routes',
    'reference_checkpoint_family': 'R7_robust_attempt1_selected_before_F1',
    'availability_cross_fold_invariant_required': True,
    'checkpoint_reload_state_replay_required': True}
FLAGS = {**bench.FLAGS, 'candidate_promoted': False,
         'historical_score_equality_required': False,
         'r7_state_replayed_exactly': True, 'inherited_input_frame_matched': True}
PHASES = ('authentication', 'source_loading', 'state_inference', 'fixed_readouts',
          'aggregate_bootstrap', 'post_authentication', 'completed')


def require(ok):
    if not ok:
        raise ValueError(ERROR) from None


def paths(attempt=1):
    require(type(attempt) is int and attempt == 1)
    return ROOT / 'BRAN_R7_INFORMATION_MATCHED_F1_ATTEMPT1'


def code_hashes():
    names = set(CODE) | set(bridge.CODE) | set(p1.code_hashes())
    return {name: sha(ROOT / name) for name in sorted(names)}


def r7_evidence():
    for name, pin in P1_PINS.items():
        require(sha(p1.paths() / name) == pin)
    p1_receipt = p1.authenticate()
    protocol, _, components, receipt = authenticate_fit('fit', 1)
    original = p1._read(p1.paths() / 'protocol.json')
    require(protocol['source_binding'] == original['r7_source_binding']
            and receipt == original['r7_fit_receipt'])
    manifest = p1.checkpoint_manifest(components, 'R', protocol['source_binding'])
    require(manifest == original['r7_checkpoint_manifest'])
    return {'fit_receipt': receipt, 'p1_receipt': p1_receipt,
            'source_binding': protocol['source_binding'], 'checkpoint_manifest': manifest}


def decorate_binding(old_binding, evidence):
    bench.validate_source_binding(old_binding)
    manifest = evidence['checkpoint_manifest']
    p1.validate_checkpoint_manifest(manifest, 'R', evidence['source_binding'])
    value = copy.deepcopy(old_binding)
    pins = {f'r7_r_fold{row["fold"]}': row['checkpoint_sha256'] for row in manifest['folds']}
    composite = bench.canonical_sha256(pins)
    value['bran_checkpoints_sha256'] = composite
    for arm in bench.BRAN_ARMS:
        value['representation_sha256'][arm] = composite
        value['arm_provenance'][arm] = {
            'upstream_supervision': 'R7 fixed outer-fold supervised multitask encoder; no encoder fitting in F1',
            'training_exposure': 'multisource V5 ancestry and paired R7 continuation; inherited outer-training transforms'}
    value['source_files_sha256'].update(pins)
    value['parent_receipts_sha256']['r7_fit_p1_binding'] = bench.canonical_sha256(evidence)
    bench.validate_source_binding(value)
    return value


def common_availability(reference, current):
    current = np.asarray(current)
    require(current.dtype == np.dtype(bool) and current.ndim == 1)
    if reference is not None:
        require(np.array_equal(reference, current))
    return current.copy()


def sanitize_inputs(c, cm, eligible, r, rm):
    visible = np.asarray(cm, bool) & np.asarray(eligible, bool) & np.isfinite(c)
    require(np.isfinite(np.asarray(r)[np.asarray(rm, bool)]).all())
    return np.where(visible, c, 0.), visible, np.where(np.asarray(rm)[:, None], r, 0.)


def build_live_adapter(callback):
    callback('authentication')
    evidence = r7_evidence()
    old_binding, parts = old._live_components()
    callback('source_loading')
    paired, _, _, source_receipt = source.source_context()
    require(evidence['source_binding'] == source_receipt['source_binding'])
    _, folds, c0, cm0, eligible, r0, rm, ages, labels, observed = bridge._aligned(
        parts, paired, source_receipt, old_binding)
    binding = decorate_binding(old_binding, evidence)
    fm, old_source = parts['fm'], parts['source']
    arrays = fm.load_fm_cache(parts['cache_binding'],
        expected_current_inputs_sha256=parts['current_input_sha256'],
        expected_row_order_sha256=old_binding['row_order_sha256'],
        expected_outer_fold_sha256=old_binding['outer_fold_sha256'],
        expected_inner_fold_sha256=old_binding['inner_fold_sha256'])
    slots = tuple(paired.names.index(field) for field in CBC_FIELDS)
    age = original_age(paired)
    _, private = r7.paths('fit', 1)
    legacy, states = {}, {}
    available = None
    for fold in range(5):
        callback('state_inference', fold)
        train = np.flatnonzero(folds != fold)
        transform0 = old_source.base.FoldTransform(c0, cm0, eligible, r0, rm, ages, train)
        c, cm, r, age_value = transform0.apply(c0, cm0, eligible, r0, rm, ages)
        c, cm, r = sanitize_inputs(c, cm, eligible, r, rm)
        item = evidence['checkpoint_manifest']['folds'][fold]
        model, transform = r7.load_checkpoint(private / f'fold{fold}_R.pt',
                                              item['checkpoint_sha256'], item['binding'])
        before, transform_pin, gradients = reuse._validate_r7(
            model, transform, fold, slots, paired.transforms[fold])
        vc, vcm = transform.clinical(paired.c, paired.cm)
        vr, vrm = transform.retinal(paired.r, paired.rm)
        bridge._frame_match(c, cm, r, rm, vc, vcm, vr, vrm)
        args = (tensor(vc), tensor(vcm, torch.bool), tensor(vr), tensor(vrm, torch.bool),
                age, transform.age_mean, transform.age_scale)
        routed = state_routes(model, *args)
        require(_unchanged(before, transform_pin, gradients, model, transform))
        reloaded, replay_transform = r7.load_checkpoint(private / f'fold{fold}_R.pt',
                                                       item['checkpoint_sha256'], item['binding'])
        rb, rt, rg = reuse._validate_r7(reloaded, replay_transform, fold, slots, paired.transforms[fold])
        replay = state_routes(reloaded, *args)
        require(reuse.replay_equal(routed, replay)
                and _unchanged(rb, rt, rg, reloaded, replay_transform))
        available = common_availability(available, np.logical_and.reduce(
            [routed.available[key].numpy() for key in ROUTES]))
        states[fold] = {key: routed.states[key].numpy() for key in ROUTES}
        legacy[fold] = (c, cm, r, age_value)
    common = fm._support_mask(rm, c0, cm0, eligible) & available
    matched = {name: fm.common_evaluation_mask(observed[name], common, labels[name])
               for name in binding['endpoint_names']}

    def design_factory(fold):
        require(type(fold) is int and fold in range(5))
        c, cm, r, age_value = legacy[fold]
        designs = {'bran_combined_state': states[fold]['both'],
            'bran_clinical_state': states[fold]['clinical'], 'bran_retinal_state': states[fold]['retinal'],
            'blood_age': parts['blood_design'](c, cm, age_value),
            'raw_clinical_age': np.c_[c, cm, age_value],
            'raw_clinical_retinal_age': np.c_[c, cm, r, rm, age_value]}
        for base in fm.ARMS:
            if base == 'labrador':
                designs['labrador_clinical_age'] = np.c_[arrays[base], age_value]
                designs['labrador_clinical_retinal_age'] = np.c_[arrays[base], c, cm, r, rm, age_value]
            else:
                designs[base + '_age'] = np.c_[arrays[base], age_value]
                designs[base + '_clinical_age'] = np.c_[arrays[base], c, cm, age_value]
        require(set(designs) == set(bench.ARMS))
        return designs

    def reauthenticate():
        old._metadata_reauthenticate(old_binding)
        require(r7_evidence() == evidence)
        _, _, _, fresh = source.source_context()
        require(fresh == source_receipt)

    context = {'folds': folds, 'labels': labels, 'observed': observed,
        'matched_support': matched, 'inner_folds': parts['inner'],
        'design_factory': design_factory, 'input_sha256': binding['shared_input_sha256'],
        'outer_fold_sha256': binding['outer_fold_sha256'], 'row_order_sha256': binding['row_order_sha256']}
    bench.validate_runtime_context(context, binding)
    provenance = {'r7': evidence, 'legacy_metadata_sha256': bench.canonical_sha256(old_binding),
                  'source_context_sha256': bench.canonical_sha256(source_receipt)}
    return binding, context, provenance, reauthenticate, old_source.ev.paired_counts, old_source.base


def make_protocol(binding, provenance):
    return {'schema': 'bran-r7-information-matched-f1-protocol-v1',
        'status': 'frozen_before_readouts', 'parameters': PARAMETERS,
        'source_binding': binding, 'provenance': provenance,
        'code_sha256': code_hashes(), 'privacy': FLAGS}


def validate_protocol(p):
    require(type(p) is dict and set(p) == {'schema', 'status', 'parameters',
            'source_binding', 'provenance', 'code_sha256', 'privacy'}
            and p['schema'] == 'bran-r7-information-matched-f1-protocol-v1'
            and p['status'] == 'frozen_before_readouts' and p['parameters'] == PARAMETERS
            and p['code_sha256'] == code_hashes() and p['privacy'] == FLAGS)
    bench.validate_source_binding(p['source_binding'])
    require(type(p['provenance']) is dict and set(p['provenance']) == {
        'r7', 'legacy_metadata_sha256', 'source_context_sha256'})
    evidence = p['provenance']['r7']
    require(type(evidence) is dict and set(evidence) == {
        'fit_receipt', 'p1_receipt', 'source_binding', 'checkpoint_manifest'})
    p1.validate_checkpoint_manifest(evidence['checkpoint_manifest'], 'R', evidence['source_binding'])
    require(evidence['source_binding']['outer_fold_sha256'] == p['source_binding']['outer_fold_sha256']
            and list(evidence['source_binding']['inner_fold_sha256']) == p['source_binding']['inner_fold_sha256'])
    require(all(bench._sha(p['provenance'][key]) for key in
                ('legacy_metadata_sha256', 'source_context_sha256')))
    pins = {f'r7_r_fold{row["fold"]}': row['checkpoint_sha256']
            for row in evidence['checkpoint_manifest']['folds']}
    b = p['source_binding']; composite = bench.canonical_sha256(pins)
    require(all(b['source_files_sha256'].get(key) == pin for key, pin in pins.items())
            and b['bran_checkpoints_sha256'] == composite
            and all(b['representation_sha256'][arm] == composite for arm in bench.BRAN_ARMS)
            and b['parent_receipts_sha256'].get('r7_fit_p1_binding') == bench.canonical_sha256(evidence))


def calculate(binding, context, counts_fn, base, callback):
    def progress(fold, arm, completed, budget):
        callback('fixed_readouts', fold, arm, completed, budget)
    with threadpool_limits(limits=2):
        predictions = bench.fit_all_readouts(context, binding, reuse.fixed_probe, progress)
        callback('aggregate_bootstrap')
        counts = counts_fn(context['folds'], draws=1000, seed=91501)
        result = bench.summarize(predictions, context['labels'], context['matched_support'],
            context['folds'], counts, binding['endpoint_names'],
            auc_fn=base.fold_weighted_auc, bootstrap_auc_fn=base._weighted_auc_draws)
    bench.validate_result(result, binding['endpoint_names'])
    return result


def progress_row(phase, fold=None, arm=None, completed=None, budget=None):
    require(phase in PHASES and (fold is None or type(fold) is int and fold in range(5))
            and arm in (None, *bench.ARMS))
    if completed is not None:
        require(type(completed) is int and type(budget) is int
                and budget == 1820 and 0 <= completed <= budget)
    else:
        require(budget is None)
    if phase == 'fixed_readouts':
        require(fold is not None and arm is not None and completed is not None)
    else:
        require(arm is None and completed is None)
        require((fold is not None) == (phase == 'state_inference'))
    return {'phase': phase, 'fold': fold, 'arm': arm,
           'completed_readout_calls': completed, 'readout_call_budget': budget,
           'patient_level_output_emitted': False}


def progress(out, state, phase, fold=None, arm=None, completed=None, budget=None):
    row = progress_row(phase, fold, arm, completed, budget)
    state.update(row)
    write_json(out / 'progress.next.json', row)
    os.replace(out / 'progress.next.json', out / 'progress.json')


def authenticate_terminal(out):
    require(out.is_dir() and not out.is_symlink() and not (out / 'failure.json').exists())
    require({p.name for p in out.iterdir()} == {'protocol.json', 'aggregate.json', 'completed.json', 'progress.json'})
    p, a, t = (p1._read(out / n) for n in ('protocol.json', 'aggregate.json', 'completed.json'))
    validate_protocol(p)
    require(set(a) == {'schema', 'status', 'protocol_sha256', 'source_binding_sha256', 'results'} | set(FLAGS)
        and a['schema'] == 'bran-r7-information-matched-f1-aggregate-v1' and a['status'] == 'completed'
        and a['protocol_sha256'] == sha(out / 'protocol.json')
        and a['source_binding_sha256'] == bench.canonical_sha256(p['source_binding'])
        and all(a[k] is v for k, v in FLAGS.items()))
    bench.validate_result(a['results'], p['source_binding']['endpoint_names'])
    require(t == {'status': 'authenticated_completed', 'protocol_sha256': sha(out / 'protocol.json'),
        'aggregate_sha256': sha(out / 'aggregate.json'), 'patient_level_output_emitted': False})
    require(p1._read(out / 'progress.json') == progress_row('completed'))
    return p, a


def run(attempt=1):
    out = paths(attempt)
    with quiet(), LOCK.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not out.exists() and not out.is_symlink())
        out.mkdir(mode=0o700)
        state = {'phase': 'authentication', 'fold': None, 'arm': None}
        callback = lambda *args: progress(out, state, *args)
        try:
            torch.set_num_threads(2)
            binding, context, evidence, recheck, counts_fn, base = build_live_adapter(callback)
            protocol = make_protocol(binding, evidence)
            validate_protocol(protocol)
            write_json(out / 'protocol.json', protocol)
            pin = sha(out / 'protocol.json')
            result = calculate(binding, context, counts_fn, base, callback)
            callback('post_authentication'); recheck()
            require(protocol == p1._read(out / 'protocol.json') and pin == sha(out / 'protocol.json'))
            validate_protocol(protocol)
            write_json(out / 'aggregate.json', {'schema': 'bran-r7-information-matched-f1-aggregate-v1',
                'status': 'completed', 'protocol_sha256': pin,
                'source_binding_sha256': bench.canonical_sha256(binding), 'results': result, **FLAGS})
            callback('completed')
            write_json(out / 'completed.json', {'status': 'authenticated_completed', 'protocol_sha256': pin,
                'aggregate_sha256': sha(out / 'aggregate.json'), 'patient_level_output_emitted': False})
            authenticate_terminal(out)
        except BaseException as exc:
            # Only this invocation's newly created success marker can be withdrawn.
            (out / 'completed.json').unlink(missing_ok=True)
            site = {'module': 'unclassified', 'line': None}
            trace = exc.__traceback__
            while trace is not None:
                module = Path(trace.tb_frame.f_code.co_filename)
                if module.parent == ROOT and module.suffix == '.py':
                    site = {'module': module.name, 'line': trace.tb_lineno}
                trace = trace.tb_next
            write_json(out / 'failure.json', {'status': 'technical_failure',
                'phase': state['phase'], 'fold': state['fold'], 'arm': state['arm'],
                'code_site': site, 'patient_level_output_emitted': False})
            raise ValueError(ERROR) from None


def audit(attempt=1, replay=False):
    out = paths(attempt)
    with quiet(), LOCK.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        p, a = authenticate_terminal(out)
        if replay:
            callback = lambda *args: None
            binding, context, evidence, recheck, counts_fn, base = build_live_adapter(callback)
            require(make_protocol(binding, evidence) == p)
            result = calculate(binding, context, counts_fn, base, callback)
            require(result == a['results']); recheck()
        require(authenticate_terminal(out) == (p, a))
        receipt = {'status': 'aggregate_terminal_authenticated',
            'protocol_sha256': sha(out / 'protocol.json'), 'aggregate_sha256': sha(out / 'aggregate.json'),
            'independent_source_state_readout_bootstrap_replay': replay,
            'patient_level_output_emitted': False}
        if replay:
            destination = ROOT / 'BRAN_R7_INFORMATION_MATCHED_F1_REPLAY.json'
            if destination.exists():
                require(p1._read(destination) == receipt)
            else:
                write_json(destination, receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, default=1)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--audit', action='store_true')
    mode.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    try:
        if args.run:
            run(args.attempt)
            result = {'status': 'completed_pending_independent_replay', 'patient_level_output_emitted': False}
        else:
            result = audit(args.attempt, args.replay)
        print(json.dumps(result, sort_keys=True)); return 0
    except BaseException:
        print(json.dumps({'status': 'closed_failure', 'patient_level_output_emitted': False})); return 1


if __name__ == '__main__':
    raise SystemExit(main())
