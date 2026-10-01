"""One local V4 recipe: pilot, five M fits, then unchanged native evaluation.

No raw logs. All patient computation is under shared lock and FD-level silence.
V3 paired-only checkpoints are authenticated and reused, never refitted here.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import time

import torch

from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json
import run_bran_multisource_fit_v3 as oldfit
import run_bran_multisource_evaluation_v3 as oldeval
from bran_multisource_binding_v3 import bind_fold, load_bound_sources
from bran_multisource_batches_v2 import transform_hash
from bran_multisource_fit_v2 import _exposure
from bran_multisource_protocol_v2 import digest
import bran_multisource_profiles_v3 as profiles
import bran_multisource_comparison_v3 as comparison
import bran_multisource_reference_predictions_v2 as references

ROOT = Path(__file__).resolve().parent
DIAGNOSTIC = ROOT/'BRAN_SOURCE_DIAGNOSTIC_V1_ATTEMPT1'
DIAGNOSTIC_PIN = '2611b36e9e58095c6b137eb3384aa1e5f5ed8df3e8184b05bdac09c3d3673ed7'
CODE = tuple(sorted(set(oldeval.CODE) | {
    'BRAN_MULTISOURCE_GRADIENT_BALANCED_V4_PLAN.md',
    'run_bran_multisource_gradient_balanced_v4.py',
    'test_run_bran_multisource_gradient_balanced_v4.py',
    'bran_multisource_continuation_v4.py', 'test_bran_multisource_continuation_v4.py',
    'bran_multisource_fit_v4.py', 'test_bran_multisource_fit_v4.py'}))
PARAMETERS = {**oldfit.PARAMETERS, 'roles': ['M'], 'control_reused': 'V3_C_attempt1',
    'only_change': 'cap_weighted_source_generative_global_gradient_at_native_paired_gradient_norm',
    'cap_norm_ratio': 1., 'norm_accumulation_dtype': 'float64',
    'native_reference_loss': 'screening_plus_0.5_CBC',
    'source_CBC_gradient_not_rescaled': True, 'pilot_updates': 100,
    'pilot_models_discarded': True, 'total_new_full_fit_updates': 15000}


def require(ok):
    if not ok: raise ValueError('gradient_balanced_v4_contract_failed')


def paths(stage, attempt):
    require(stage in ('pilot', 'fit', 'evaluate') and type(attempt) is int and 1 <= attempt <= 99)
    return (ROOT/f'BRAN_GRADIENT_BALANCED_V4_{stage.upper()}_ATTEMPT{attempt}',
            ROOT/'private_artifacts'/f'bran_gradient_balanced_v4_fit_attempt{attempt}')


def code_hashes():
    return {name: sha(ROOT/name) for name in CODE}


def auth_diagnostic():
    terminal = json.loads((DIAGNOSTIC/'completed.json').read_text())
    manifest = json.loads((DIAGNOSTIC/'manifest.json').read_text())
    aggregate = json.loads((DIAGNOSTIC/'aggregate.json').read_text())
    require(not (DIAGNOSTIC/'failure.json').exists()
        and terminal['status'] == 'authenticated'
        and terminal['aggregate_sha256'] == manifest['aggregate_sha256'] == DIAGNOSTIC_PIN
        and sha(DIAGNOSTIC/'aggregate.json') == DIAGNOSTIC_PIN
        and sha(DIAGNOSTIC/'manifest.json') == terminal['manifest_sha256']
        and sha(DIAGNOSTIC/'protocol.json') == terminal['protocol_sha256'] == manifest['protocol_sha256']
        and aggregate['five_fold_initial_models_unchanged'] is True
        and aggregate['heldout_scoring_performed'] is False
        and aggregate['protected_sources_used'] is False)
    require(all(sha(ROOT/name) == pin for name, pin in manifest['code_sha256'].items()))
    return sha(DIAGNOSTIC/'completed.json')


def auth_terminal(out):
    require(not (out/'failure.json').exists())
    terminal = json.loads((out/'completed.json').read_text())
    aggregate = json.loads((out/'aggregate.json').read_text())
    protocol = json.loads((out/'protocol.json').read_text())
    require(terminal['status'] == 'authenticated'
        and terminal['aggregate_sha256'] == sha(out/'aggregate.json')
        and terminal['protocol_sha256'] == sha(out/'protocol.json')
        and protocol['code_sha256'] == code_hashes()
        and protocol['parameters'] == PARAMETERS
        and aggregate['patient_level_output_emitted'] is False
        and aggregate['candidate_promoted'] is False)
    return protocol, aggregate, terminal


def progress(out, state, phase, fold=None, updates=None):
    require(phase in ('authentication', 'pilot_training', 'training', 'checkpoint_replay',
        'post_fit_authentication', 'reference_loading', 'reference_inference', 'reference_replay',
        'reference_bootstrap', 'candidate_inference', 'advancement_bootstrap',
        'candidate_aggregate_bootstrap', 'native_profiles_completed', 'result_authentication', 'completed'))
    require(fold is None or type(fold) is int and fold in range(5))
    require(updates is None or type(updates) is int and 0 <= updates <= 3000)
    state['phase'] = phase
    tmp = out/'progress.next.json'
    write_json(tmp, {'phase': phase, 'fold': fold, 'updates_completed': updates,
        'pid': os.getpid(), 'patient_level_output_emitted': False})
    os.replace(tmp, out/'progress.json')


def freeze(out, stage, binding, extra=None):
    protocol = {'schema': 'bran-gradient-balanced-v4-protocol', 'stage': stage,
        'status': 'frozen_before_stage', 'parameters': PARAMETERS, 'evaluation': oldfit.EVALUATION,
        'code_sha256': code_hashes(), 'source_binding': binding,
        'diagnostic_terminal_sha256': auth_diagnostic(), 'adaptive_development': True,
        'source_weights_or_sources_selected_by_external_performance': False,
        'patient_level_output_emitted': False, 'candidate_promoted': False,
        'dependencies': extra or {}}
    write_json(out/'protocol.json', protocol)
    return protocol, sha(out/'protocol.json')


def finish(out, protocol, pin, value, start, state):
    require(code_hashes() == protocol['code_sha256'] and sha(out/'protocol.json') == pin)
    require(value['patient_level_output_emitted'] is False and value['candidate_promoted'] is False)
    json.dumps(value, allow_nan=False)
    write_json(out/'aggregate.json', value)
    require(json.loads((out/'aggregate.json').read_text()) == value and not (out/'failure.json').exists())
    progress(out, state, 'completed')
    write_json(out/'completed.json', {'status': 'authenticated',
        'protocol_sha256': pin, 'aggregate_sha256': sha(out/'aggregate.json'),
        'elapsed_seconds': time.monotonic()-start, 'patient_level_output_emitted': False})


def pilot(out, state):
    from bran_multisource_continuation_v4 import train_step_v4
    start = time.monotonic()
    sources, _, receipt = oldeval.authenticate_fits(1)
    protocol, pin = freeze(out, 'pilot', sources.receipt(), {'V3_fit': receipt})
    bound = bind_fold(sources, 0)
    model = copy.deepcopy(bound.model).train()
    for p in model.parameters(): p.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    paired, source = bound.paired_factory(), bound.source_factory()
    counters = {'cap_applied_updates': 0, 'cap_contract_checks': 0,
                'source_generative_nonzero_updates': 0, 'source_cbc_nonzero_updates': 0}
    teacher = {key: value.clone() for key, value in bound.teacher.state_dict().items()}
    loop_start = time.monotonic()
    progress(out, state, 'pilot_training', 0, 0)
    for step in range(100):
        result = train_step_v4(model, bound.teacher, optimizer, paired.sample(), source.sample(),
            step, bound.transform.age_mean, bound.transform.age_scale, bound.seed,
            bound.cbc_indices, paired.positive_weights, bound.state_scale, source_enabled=True)
        require(result['optimizer_updated'] and result['cap_contract_satisfied'])
        counters['cap_applied_updates'] += int(result['source_generative_cap_applied'])
        counters['cap_contract_checks'] += 1
        counters['source_generative_nonzero_updates'] += int(result['source_generative_gradient_nonzero'])
        counters['source_cbc_nonzero_updates'] += int(result['source_cbc_gradient_nonzero'])
    loop_seconds = time.monotonic()-loop_start
    require(counters['cap_applied_updates'] > 0 and counters['source_generative_nonzero_updates'] > 0
        and counters['source_cbc_nonzero_updates'] > 0
        and all(torch.equal(v, bound.teacher.state_dict()[k]) for k, v in teacher.items())
        and all(torch.isfinite(v).all() for v in model.parameters()))
    del model, optimizer
    require(load_bound_sources().receipt() == protocol['source_binding'])
    finish(out, protocol, pin, {'schema': 'bran-gradient-balanced-v4-pilot', 'status': 'pilot_passed',
        'updates': 100, 'algorithm_update_counters': counters, 'loop_seconds': loop_seconds,
        'training_only': True, 'pilot_model_discarded': True, 'heldout_scoring_performed': False,
        'patient_level_output_emitted': False, 'candidate_promoted': False}, start, state)


def fit(out, private, attempt, state):
    from bran_multisource_fit_v4 import fit_one_v4
    start = time.monotonic()
    pilot_out, _ = paths('pilot', attempt)
    pp, pa, pt = auth_terminal(pilot_out)
    require(pp['stage'] == 'pilot' and pa['status'] == 'pilot_passed')
    sources, controls, old_receipt = oldeval.authenticate_fits(1)
    require(pp['source_binding'] == sources.receipt())
    protocol, pin = freeze(out, 'fit', sources.receipt(),
        {'pilot_terminal_sha256': sha(pilot_out/'completed.json'), 'V3_fit': old_receipt})
    private.mkdir(mode=0o700)
    components = {}
    for fold in range(5):
        bound = bind_fold(sources, fold)
        progress(out, state, 'training', fold, 0)
        result = fit_one_v4(bound.model, bound.teacher, bound.paired_factory, bound.source_factory,
            bound.state_scale, bound.transform.age_mean, bound.transform.age_scale,
            bound.seed, bound.cbc_indices, 'M',
            progress=lambda n: progress(out, state, 'training', fold, n))
        control = controls[('C', fold)]
        require(result['paired_input_digest'] == control['paired_input_digest']
            and result['paired_completion_mask_digest'] == control['paired_mask_digest'])
        binding = {'protocol_sha256': pin, 'fold': fold, 'role': 'M', 'training_recipe': 'gradient_balanced_v4',
            'initial_checkpoint_sha256': sources.context.protocol['native_source']['checkpoint_sha256'][f'fold{fold}'],
            'transform_sha256': transform_hash(bound.transform),
            'model_config_sha256': digest(result['model'].export_config()),
            'clinical_field_order_sha256': digest(sources.paired.names),
            'outer_fold_sha256': sources.receipt()['outer_fold_sha256'],
            'inner_fold_sha256': sources.receipt()['inner_fold_sha256'][fold]}
        progress(out, state, 'checkpoint_replay', fold, 3000)
        path = private/f'fold{fold}_M.pt'
        checkpoint_pin = oldfit.save_checkpoint(path, result, bound, binding)
        restored, transform = oldfit.load_checkpoint(path, checkpoint_pin, binding)
        require(all(torch.equal(v, restored.state_dict()[k]) for k, v in result['model'].state_dict().items()))
        exposure = _exposure(result['source_sampler'].private_sampler,
            result['paired_sampler'].private_sampler, sources.paired, sources.pools)
        require(all(exposure['per_source'][p.source]['source_local_people_lower_bound_20'] is not None
                    for p in sources.pools))
        counters = result['algorithm_update_counters']
        require(counters['cap_contract_checks'] == 3000 and counters['cap_applied_updates'] > 0
            and counters['source_generative_nonzero_updates'] > 0 and counters['source_cbc_nonzero_updates'] > 0)
        item = {'status': 'fit_completed_pending_evaluation', 'binding': binding,
            'checkpoint_sha256': checkpoint_pin, 'checkpoint_reload_exact': True,
            'updates_completed': result['updates'], 'runtime_seconds': result['elapsed_seconds'],
            'source_loss_supported': result['source_gradient_supported'], 'exposure': exposure,
            'algorithm_update_counters': counters, 'paired_input_digest': result['paired_input_digest'],
            'paired_mask_digest': result['paired_completion_mask_digest'],
            'matched_C_component_sha256': old_receipt['component_sha256'][f'fold{fold}_C.json'],
            'candidate_promoted': False, 'patient_level_output_emitted': False}
        write_json(out/f'fold{fold}_M.json', item)
        components[f'fold{fold}_M.json'] = sha(out/f'fold{fold}_M.json')
        del result, restored
    progress(out, state, 'post_fit_authentication')
    require(load_bound_sources().receipt() == protocol['source_binding'])
    auth_terminal(pilot_out)
    finish(out, protocol, pin, {'schema': 'bran-gradient-balanced-v4-fits',
        'status': 'fits_completed_pending_evaluation', 'component_sha256': components,
        'five_final_checkpoints_reloaded': True, 'paired_inputs_masks_match_reused_C': True,
        'all_seven_qualified_sources_sampled': True, 'historical_models_unchanged': True,
        'protected_sources_used': False, 'patient_level_output_emitted': False,
        'candidate_promoted': False}, start, state)


def evaluate(out, attempt, state):
    start = time.monotonic()
    fit_out, private = paths('fit', attempt)
    fp, fa, ft = auth_terminal(fit_out)
    require(fp['stage'] == 'fit' and fa['status'] == 'fits_completed_pending_evaluation')
    sources, controls, old_receipt = oldeval.authenticate_fits(1)
    require(fp['source_binding'] == sources.receipt() and fp['dependencies']['V3_fit'] == old_receipt)
    require(set(fa['component_sha256']) == {f'fold{f}_M.json' for f in range(5)})
    components = {}
    for fold in range(5):
        name = f'fold{fold}_M.json'
        require(sha(fit_out/name) == fa['component_sha256'][name])
        item = json.loads((fit_out/name).read_text())
        b = item['binding']
        require(b['protocol_sha256'] == ft['protocol_sha256'] and b['fold'] == fold
            and b['training_recipe'] == 'gradient_balanced_v4' and b['role'] == 'M'
            and item['updates_completed'] == 3000 and item['checkpoint_reload_exact'] is True
            and b['outer_fold_sha256'] == sources.receipt()['outer_fold_sha256']
            and b['inner_fold_sha256'] == sources.receipt()['inner_fold_sha256'][fold]
            and item['paired_input_digest'] == controls[('C', fold)]['paired_input_digest']
            and item['paired_mask_digest'] == controls[('C', fold)]['paired_mask_digest'])
        components[fold] = item
        oldfit.load_checkpoint(private/f'fold{fold}_M.pt', item['checkpoint_sha256'], b)
    protocol, pin = freeze(out, 'evaluate', sources.receipt(),
        {'fit_terminal_sha256': sha(fit_out/'completed.json'), 'V3_fit': old_receipt,
         'comparison_M': 'new_gradient_balanced_v4', 'comparison_C': 'unchanged_reused_V3_paired_only'})
    _, old_private = oldfit.paths(1)
    def provider(role, fold):
        require(role in ('C', 'M') and type(fold) is int and fold in range(5))
        item = controls[('C', fold)] if role == 'C' else components[fold]
        folder = old_private if role == 'C' else private
        return oldfit.load_checkpoint(folder/f'fold{fold}_{role}.pt', item['checkpoint_sha256'], item['binding'])
    def report(event):
        require(set(event) <= {'phase', 'fold', 'role'})
        progress(out, state, event['phase'], event.get('fold'))
    native = profiles.evaluate(sources.paired, provider, progress=report)
    require(native['reload_predictions_equal'] is True and native['all_empty_physiology_abstained'] is True
        and native['patient_level_output_emitted'] is False and native['candidate_promoted'] is False)
    write_json(out/'native_profiles.json', native)
    progress(out, state, 'native_profiles_completed')
    reference = references.build(sources.paired, sources.context, progress=report)
    value = comparison.evaluate(sources.paired, reference, provider, progress=report)
    oldeval.validate_result(value)
    progress(out, state, 'result_authentication')
    auth_terminal(fit_out)
    require(load_bound_sources().receipt() == protocol['source_binding'])
    finish(out, protocol, pin, {'schema': 'bran-gradient-balanced-v4-evaluation', 'status': 'completed',
        'comparison': value, 'native_profiles_sha256': sha(out/'native_profiles.json'),
        'roles': {'M': 'gradient_balanced_v4', 'C': 'reused_V3_paired_only', 'I': 'unchanged_initial'},
        'promotion_eligible': value['promotion_eligible'], 'candidate_promoted': False,
        'historical_gate_definitions_changed': False, 'protected_sources_used': False,
        'independent_candidate_aggregate_recomputation': False,
        'patient_level_output_emitted': False, 'scientific_goal_achieved': False}, start, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('pilot', 'fit', 'evaluate'), required=True)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    out, private = paths(args.stage, args.attempt)
    state, owned, ok = {'phase': 'authentication'}, False, False
    with quiet():
        try:
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                out.mkdir(exist_ok=False); owned = True
                torch.set_num_threads(2)
                progress(out, state, 'authentication')
                if args.stage == 'pilot': pilot(out, state)
                elif args.stage == 'fit': fit(out, private, args.attempt, state)
                else: evaluate(out, args.attempt, state)
                ok = True
        except Exception:
            if owned and not (out/'completed.json').exists():
                write_json(out/'failure.json', {'status': 'technical_failure', 'phase': state['phase'],
                    'patient_level_output_emitted': False, 'candidate_promoted': False,
                    'completed_components_preserved': True})
    print(json.dumps({'status': 'completed' if ok else 'not_completed', 'stage': args.stage,
        'phase': state['phase'], 'patient_level_output_emitted': False, 'candidate_promoted': False}))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__': main()
