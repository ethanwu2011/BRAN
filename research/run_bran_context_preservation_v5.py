"""Fixed V5 local queue: diagnostic, two-arm pilot, fits, unchanged evaluation.

Patient-derived computation is FD-quiet under the shared exclusive lock. Public
receipts contain only closed aggregates and content bindings. V1 is not changed.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import torch

from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json
import run_bran_multisource_fit_v3 as oldfit
import run_bran_multisource_evaluation_v3 as oldeval
import run_bran_multisource_gradient_balanced_v4 as prior
from bran_multisource_binding_v3 import bind_fold, load_bound_sources
from bran_multisource_batches_v2 import transform_hash
from bran_multisource_fit_v2 import _exposure
from bran_multisource_protocol_v2 import digest
import bran_multisource_profiles_v3 as profiles
import bran_multisource_comparison_v3 as comparison
import bran_multisource_reference_predictions_v2 as references

ROOT = Path(__file__).resolve().parent
ROLES = {'C': 'paired_values_with_source_masks_only', 'M': 'paired_and_source_values_with_source_masks'}
CODE = tuple(sorted(set(oldeval.CODE) | set(prior.CODE) | {
    'BRAN_CONTEXT_PRESERVATION_V5_PLAN.md', 'bran_context_diagnostic_v5.py', 'run_bran_context_v5_attempt1.sh',
    'run_bran_multisource_source_diagnostic_v1.py',
    'run_bran_context_preservation_v5.py', 'test_run_bran_context_preservation_v5.py',
    'test_bran_context_v5_privacy.py',
    'bran_multisource_preservation_v5.py', 'test_bran_multisource_preservation_v5.py',
    'bran_multisource_continuation_v5.py', 'test_bran_multisource_continuation_v5.py',
    'bran_multisource_fit_v5.py', 'test_bran_multisource_fit_v5.py'}))
PARAMETERS = {**oldfit.PARAMETERS, 'roles': ['C', 'M'], 'role_semantics': ROLES,
    'paired_loss': {**oldfit.PARAMETERS['paired_loss'], 'prediction_preservation': 1.},
    'source_generative_gradient_cap': 'V4_unchanged',
    'prediction_preservation_coefficient': 1., 'prediction_preservation_CBC_weight': .5,
    'prediction_preservation_views': ['natural', 'source_availability_intersection'],
    'prediction_preservation_age': 'reported_only', 'state_preservation_coefficient': .1,
    'pilot_updates_per_arm': 100, 'pilot_models_discarded': True,
    'total_new_full_fit_updates': 30000,
    'context_diagnostic': {'folds': 5, 'repeats': 4, 'paired_batch': 512, 'source_batch': 128,
        'seed_base': 96101, 'first_step': 330, 'repeat_step_stride': 54,
        'minimum_people_per_batch': 20, 'minimum_batches': 20, 'training_updates': 0}}
EVALUATION = {**oldfit.EVALUATION, 'roles': {**ROLES, 'I': 'unchanged_native_per_fold'}}
COUNTERS = {'cap_applied_updates': 'source_generative_cap_applied',
    'cap_contract_checks': 'cap_contract_satisfied',
    'source_generative_nonzero_updates': 'source_generative_gradient_nonzero',
    'source_cbc_nonzero_updates': 'source_cbc_gradient_nonzero',
    'preservation_supported_updates': 'prediction_preservation_supported'}


def require(ok):
    if not ok: raise ValueError('context_preservation_v5_contract_failed')


def paths(stage, attempt):
    require(stage in ('diagnostic', 'pilot', 'fit', 'evaluate') and type(attempt) is int and 1 <= attempt <= 99)
    return ROOT/f'BRAN_CONTEXT_V5_{stage.upper()}_ATTEMPT{attempt}', ROOT/'private_artifacts'/f'bran_context_v5_fit_attempt{attempt}'


def code_hashes(): return {name: sha(ROOT/name) for name in CODE}


def auth_terminal(out):
    require(not (out/'failure.json').exists())
    terminal = json.loads((out/'completed.json').read_text())
    protocol = json.loads((out/'protocol.json').read_text())
    aggregate = json.loads((out/'aggregate.json').read_text())
    require(terminal['status'] == 'authenticated'
        and terminal['aggregate_sha256'] == sha(out/'aggregate.json')
        and terminal['protocol_sha256'] == sha(out/'protocol.json')
        and protocol['code_sha256'] == code_hashes() and protocol['parameters'] == PARAMETERS
        and aggregate['patient_level_output_emitted'] is False
        and aggregate['candidate_promoted'] is False)
    return protocol, aggregate, terminal


def progress(out, state, phase, fold=None, updates=None, role=None):
    require(phase in ('authentication', 'context_diagnostic', 'pilot_training', 'training',
        'checkpoint_replay', 'post_fit_authentication', 'reference_loading', 'reference_inference',
        'reference_replay', 'reference_bootstrap', 'candidate_inference', 'advancement_bootstrap',
        'candidate_aggregate_bootstrap', 'native_profiles_completed', 'result_authentication', 'completed'))
    require(fold is None or type(fold) is int and fold in range(5))
    require(updates is None or type(updates) is int and 0 <= updates <= 3000)
    require(role is None or role in ROLES)
    state['phase'] = phase
    tmp = out/'progress.next.json'
    write_json(tmp, {'phase': phase, 'fold': fold, 'role': role, 'updates_completed': updates,
        'pid': os.getpid(), 'patient_level_output_emitted': False})
    os.replace(tmp, out/'progress.json')


def freeze(out, stage, binding, dependencies):
    protocol = {'schema': 'bran-context-preservation-v5-protocol', 'stage': stage,
        'status': 'frozen_before_stage', 'parameters': PARAMETERS, 'evaluation': EVALUATION,
        'code_sha256': code_hashes(), 'source_binding': binding, 'dependencies': dependencies,
        'historical_diagnostic_terminal_sha256': prior.auth_diagnostic(),
        'adaptive_development': True, 'source_selection_by_external_performance': False,
        'patient_level_output_emitted': False, 'candidate_promoted': False}
    write_json(out/'protocol.json', protocol)
    return protocol, sha(out/'protocol.json')


def finish(out, protocol, pin, value, start, state):
    require(code_hashes() == protocol['code_sha256'] and sha(out/'protocol.json') == pin)
    require(value['patient_level_output_emitted'] is False and value['candidate_promoted'] is False)
    json.dumps(value, allow_nan=False)
    write_json(out/'aggregate.json', value)
    require(json.loads((out/'aggregate.json').read_text()) == value and not (out/'failure.json').exists())
    progress(out, state, 'completed')
    write_json(out/'completed.json', {'status': 'authenticated', 'protocol_sha256': pin,
        'aggregate_sha256': sha(out/'aggregate.json'), 'elapsed_seconds': time.monotonic()-start,
        'patient_level_output_emitted': False})


def check_counters(counters, role, updates):
    require(set(counters) == set(COUNTERS) and role in ROLES)
    require(all(type(n) is int and 0 <= n <= updates for n in counters.values()))
    require(counters['cap_contract_checks'] == updates and counters['preservation_supported_updates'] > 0)
    if role == 'M':
        require(counters['source_generative_nonzero_updates'] > 0 and counters['source_cbc_nonzero_updates'] > 0)
    else:
        require(all(counters[k] == 0 for k in ('cap_applied_updates', 'source_generative_nonzero_updates', 'source_cbc_nonzero_updates')))


def diagnostic(out, state):
    from bran_context_diagnostic_v5 import execute
    start = time.monotonic()
    sources, _, receipt = oldeval.authenticate_fits(1)
    protocol, pin = freeze(out, 'diagnostic', sources.receipt(), {'V3_fit': receipt})
    summary = execute(sources, lambda f: progress(out, state, 'context_diagnostic', f))
    require(load_bound_sources().receipt() == protocol['source_binding'])
    finish(out, protocol, pin, {'schema': 'bran-context-v5-diagnostic', 'status': 'completed',
        'summary': summary, 'training_only': True, 'training_updates': 0,
        'five_fold_initial_models_unchanged': True, 'batch_repeats_are_not_unique_people': True,
        'paired_views_use_identical_supported_people': True, 'not_a_covariate_matched_cohort_comparison': True,
        'protected_sources_used': False, 'heldout_scoring_performed': False,
        'patient_level_output_emitted': False, 'candidate_promoted': False}, start, state)


def pilot(out, attempt, state):
    from bran_multisource_continuation_v5 import train_step_v5
    from bran_multisource_fit_v3 import _add_paired_input
    start = time.monotonic()
    diagnostic_out, _ = paths('diagnostic', attempt)
    dp, da, _ = auth_terminal(diagnostic_out)
    require(dp['stage'] == 'diagnostic' and da['status'] == 'completed')
    sources, _, receipt = oldeval.authenticate_fits(1)
    require(dp['source_binding'] == sources.receipt())
    protocol, pin = freeze(out, 'pilot', sources.receipt(),
        {'diagnostic_terminal_sha256': sha(diagnostic_out/'completed.json'), 'V3_fit': receipt})
    arms = {}
    for role in ROLES:
        bound = bind_fold(sources, 0)
        model = copy.deepcopy(bound.model).train()
        for p in model.parameters(): p.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        paired, source = bound.paired_factory(), bound.source_factory()
        counters = dict.fromkeys(COUNTERS, 0)
        hashes = {key: hashlib.sha256() for key in ('input', 'mask', 'bridge')}
        teacher = {k: v.clone() for k, v in bound.teacher.state_dict().items()}
        loop = time.monotonic()
        progress(out, state, 'pilot_training', 0, 0, role)
        for step in range(100):
            pb, sb = paired.sample(), source.sample()
            _add_paired_input(hashes['input'], pb, paired.positive_weights)
            result = train_step_v5(model, bound.teacher, optimizer, pb, sb, step,
                bound.transform.age_mean, bound.transform.age_scale, bound.seed,
                bound.cbc_indices, paired.positive_weights, bound.state_scale, source_enabled=role == 'M')
            require(result['optimizer_updated'] and result['cap_contract_satisfied'])
            for key, field in COUNTERS.items(): counters[key] += int(result[field])
            hashes['mask'].update(result['mask_hashes']['paired'].encode())
            hashes['mask'].update(result['mask_hashes']['paired_completion'].encode())
            hashes['bridge'].update(result['bridge_mask_digest'].encode())
        check_counters(counters, role, 100)
        require(all(torch.equal(v, bound.teacher.state_dict()[k]) for k, v in teacher.items())
            and all(torch.isfinite(v).all() for v in model.parameters()))
        arms[role] = {'updates': 100, 'algorithm_update_counters': counters,
            'loop_seconds': time.monotonic()-loop, 'digests': {k: h.hexdigest() for k, h in hashes.items()}}
        del model, optimizer
    require(arms['C']['digests'] == arms['M']['digests'])
    require(load_bound_sources().receipt() == protocol['source_binding'])
    finish(out, protocol, pin, {'schema': 'bran-context-v5-pilot', 'status': 'pilot_passed',
        'arms': arms, 'training_only': True, 'pilot_models_discarded': True,
        'heldout_scoring_performed': False, 'patient_level_output_emitted': False,
        'candidate_promoted': False}, start, state)


def fit(out, private, attempt, state):
    from bran_multisource_fit_v5 import fit_one_v5
    start = time.monotonic()
    pilot_out, _ = paths('pilot', attempt)
    pp, pa, _ = auth_terminal(pilot_out)
    require(pp['stage'] == 'pilot' and pa['status'] == 'pilot_passed')
    sources, old_components, receipt = oldeval.authenticate_fits(1)
    require(pp['source_binding'] == sources.receipt())
    protocol, pin = freeze(out, 'fit', sources.receipt(),
        {'pilot_terminal_sha256': sha(pilot_out/'completed.json'), 'V3_fit': receipt})
    private.mkdir(mode=0o700)
    components = {}
    for fold in range(5):
        matched = None
        for role in ROLES:
            bound = bind_fold(sources, fold)
            result = fit_one_v5(bound.model, bound.teacher, bound.paired_factory, bound.source_factory,
                bound.state_scale, bound.transform.age_mean, bound.transform.age_scale, bound.seed,
                bound.cbc_indices, role,
                progress=lambda n: progress(out, state, 'training', fold, n, role))
            trace = {k: result[k] for k in ('paired_input_digest', 'paired_completion_mask_digest', 'bridge_mask_digest')}
            require(result['updates'] == 3000 and result['source_gradient_supported'] == (role == 'M'))
            require(trace['paired_input_digest'] == old_components[('C', fold)]['paired_input_digest']
                and trace['paired_completion_mask_digest'] == old_components[('C', fold)]['paired_mask_digest'])
            if matched is not None: require(matched == trace)
            matched = trace
            binding = {'protocol_sha256': pin, 'fold': fold, 'role': role,
                'training_recipe': 'context_preservation_v5',
                'initial_checkpoint_sha256': sources.context.protocol['native_source']['checkpoint_sha256'][f'fold{fold}'],
                'transform_sha256': transform_hash(bound.transform),
                'model_config_sha256': digest(result['model'].export_config()),
                'clinical_field_order_sha256': digest(sources.paired.names),
                'outer_fold_sha256': sources.receipt()['outer_fold_sha256'],
                'inner_fold_sha256': sources.receipt()['inner_fold_sha256'][fold]}
            progress(out, state, 'checkpoint_replay', fold, 3000, role)
            path = private/f'fold{fold}_{role}.pt'
            checkpoint_pin = oldfit.save_checkpoint(path, result, bound, binding)
            restored, _ = oldfit.load_checkpoint(path, checkpoint_pin, binding)
            require(all(torch.equal(v, restored.state_dict()[k]) for k, v in result['model'].state_dict().items()))
            exposure = _exposure(result['source_sampler'].private_sampler,
                result['paired_sampler'].private_sampler, sources.paired, sources.pools)
            require(all(exposure['per_source'][p.source]['source_local_people_lower_bound_20'] is not None for p in sources.pools))
            check_counters(result['algorithm_update_counters'], role, 3000)
            item = {'status': 'fit_completed_pending_evaluation', 'binding': binding,
                'checkpoint_sha256': checkpoint_pin, 'checkpoint_reload_exact': True,
                'updates_completed': 3000, 'runtime_seconds': result['elapsed_seconds'],
                'source_loss_supported': result['source_gradient_supported'], 'exposure': exposure,
                'source_exposure_semantics': 'availability_masks_only' if role == 'C' else 'measurement_values_and_masks',
                'algorithm_update_counters': result['algorithm_update_counters'], **trace,
                'candidate_promoted': False, 'patient_level_output_emitted': False}
            name = f'fold{fold}_{role}.json'
            write_json(out/name, item); components[name] = sha(out/name)
            del result, restored
    progress(out, state, 'post_fit_authentication')
    require(load_bound_sources().receipt() == protocol['source_binding'])
    auth_terminal(pilot_out)
    finish(out, protocol, pin, {'schema': 'bran-context-v5-fits', 'status': 'fits_completed_pending_evaluation',
        'component_sha256': components, 'ten_final_checkpoints_reloaded': True,
        'paired_inputs_masks_and_bridge_match_C_M': True, 'all_seven_qualified_sources_sampled': True,
        'source_exposure_in_C_is_masks_only': True, 'historical_models_unchanged': True,
        'protected_sources_used': False, 'patient_level_output_emitted': False,
        'candidate_promoted': False}, start, state)


def evaluate(out, attempt, state):
    start = time.monotonic()
    fit_out, private = paths('fit', attempt)
    fp, fa, ft = auth_terminal(fit_out)
    require(fp['stage'] == 'fit' and fa['status'] == 'fits_completed_pending_evaluation')
    sources, _, receipt = oldeval.authenticate_fits(1)
    require(fp['source_binding'] == sources.receipt() and fp['dependencies']['V3_fit'] == receipt)
    require(set(fa['component_sha256']) == {f'fold{f}_{r}.json' for f in range(5) for r in ROLES})
    components = {}
    for fold in range(5):
        for role in ROLES:
            name = f'fold{fold}_{role}.json'
            require(sha(fit_out/name) == fa['component_sha256'][name])
            item = json.loads((fit_out/name).read_text()); b = item['binding']
            require(b['protocol_sha256'] == ft['protocol_sha256'] and b['fold'] == fold and b['role'] == role
                and b['training_recipe'] == 'context_preservation_v5' and item['updates_completed'] == 3000
                and item['checkpoint_reload_exact'] is True
                and b['outer_fold_sha256'] == sources.receipt()['outer_fold_sha256']
                and b['inner_fold_sha256'] == sources.receipt()['inner_fold_sha256'][fold])
            check_counters(item['algorithm_update_counters'], role, 3000)
            oldfit.load_checkpoint(private/f'fold{fold}_{role}.pt', item['checkpoint_sha256'], b)
            components[(role, fold)] = item
        require(all(components[('C', fold)][k] == components[('M', fold)][k] for k in
            ('paired_input_digest', 'paired_completion_mask_digest', 'bridge_mask_digest')))
    protocol, pin = freeze(out, 'evaluate', sources.receipt(),
        {'fit_terminal_sha256': sha(fit_out/'completed.json'), 'V3_fit': receipt, 'roles': ROLES})
    def provider(role, fold):
        require(role in ROLES and type(fold) is int and fold in range(5))
        item = components[(role, fold)]
        return oldfit.load_checkpoint(private/f'fold{fold}_{role}.pt', item['checkpoint_sha256'], item['binding'])
    def report(event):
        require(set(event) <= {'phase', 'fold', 'role'})
        progress(out, state, event['phase'], event.get('fold'), role=event.get('role'))
    native = profiles.evaluate(sources.paired, provider, progress=report)
    require(native['reload_predictions_equal'] and native['all_empty_physiology_abstained']
        and native['patient_level_output_emitted'] is False and native['candidate_promoted'] is False)
    write_json(out/'native_profiles.json', native)
    progress(out, state, 'native_profiles_completed')
    reference = references.build(sources.paired, sources.context, progress=report)
    value = comparison.evaluate(sources.paired, reference, provider, progress=report)
    oldeval.validate_result(value)
    progress(out, state, 'result_authentication')
    auth_terminal(fit_out)
    require(load_bound_sources().receipt() == protocol['source_binding'])
    finish(out, protocol, pin, {'schema': 'bran-context-v5-evaluation', 'status': 'completed',
        'comparison': value, 'native_profiles_sha256': sha(out/'native_profiles.json'),
        'roles': {**ROLES, 'I': 'unchanged_initial'}, 'promotion_eligible': value['promotion_eligible'],
        'candidate_promoted': False, 'historical_gate_definitions_changed': False,
        'protected_sources_used': False, 'independent_candidate_aggregate_recomputation': False,
        'patient_level_output_emitted': False, 'scientific_goal_achieved': False}, start, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('diagnostic', 'pilot', 'fit', 'evaluate'), required=True)
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
                if args.stage == 'diagnostic': diagnostic(out, state)
                elif args.stage == 'pilot': pilot(out, args.attempt, state)
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
