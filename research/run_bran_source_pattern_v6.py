"""Exclusive, FD-quiet V6 pilot/fit lifecycle. No evaluation or promotion here."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import time

from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json

ROOT = Path(__file__).resolve().parent
PLAN = 'BRAN_NATURE_WEEK_EXECUTION_2026-09-21.md'
ROLES = {'C': 'V5_M_continuation_control', 'S': 'source_pattern_true_target_supervision'}
PARAMETERS = {
    'parent': 'V5_M_attempt2_corresponding_fold', 'teacher': 'frozen_parent_V5_M',
    'roles': ROLES, 'folds': 5, 'pilot_updates_per_arm': 100,
    'pilot_fold': 0, 'pilot_models_discarded': True, 'updates_per_arm_fold': 3000,
    'paired_batch': 96, 'source_batch': 128, 'seed_base': 95101,
    'learning_rate': 5e-5, 'weight_decay': 1e-4, 'gradient_clip': 5,
    'bridge_screening_coefficient': .5, 'bridge_cbc_coefficient': .25,
    'bridge_cbc_targets': 'original_six_pattern_designated_observed_targets_only',
    'source_values_enabled_in_both_arms': True, 'retinal_backbone_frozen': True,
    'state_scale': 'V5_proper_training_std_clamped_min1',
    'source_streams_matched': True, 'all_other_V5_losses_unchanged': True,
    'protected_sources_used': False, 'automatic_promotion': False,
}
EVALUATION = {
    'stress_patterns': ['clinical_drop25', 'clinical_drop50', 'clinical_drop75',
                        'no_retina', 'all_clinical_hidden'],
    'stress_weighting': 'equal_pattern_equal_supported_endpoint_macro_AUROC',
    'primary_contrasts': ['S_minus_V5', 'S_minus_C'],
    'draws': 1000, 'bootstrap_seed': 98551, 'minimum_valid': 900,
    'bootstrap': 'participant_paired_within_fold', 'ci': .95,
    'gate_stress': 'both_paired_lower_bounds_strictly_positive',
    'gate_screen': 'available_native_macro_point_S_ge_V5',
    'gate_completion': 'historical_protected_point_checks_against_V5_tolerance_zero',
    'unsupported_required_gate': 'fail', 'fallback': 'V5_not_control',
    'historical_gate': 'unchanged_and_reported_separately',
    'external_model_selection': False, 'clinical_deployment': False,
}
TRACES = ('paired_input_digest', 'paired_completion_mask_digest', 'bridge_mask_digest',
          'source_values_digest', 'source_availability_digest')
PHASES = ('authentication', 'source_loading', 'parent_binding', 'pilot_training',
          'training', 'checkpoint_replay', 'post_authentication', 'completed')


def require(ok):
    if not ok:
        raise ValueError('source_pattern_v6_runner_contract_failed') from None


def code_hashes():
    import run_bran_v5_cbc_uncertainty as parent
    names = set(parent.CODE) | {PLAN, 'run_bran_source_pattern_v6.py',
        'bran_source_pattern_supervision_v6.py', 'bran_multisource_continuation_v6.py',
        'bran_multisource_fit_v6.py', 'test_bran_source_pattern_supervision_v6.py',
        'test_bran_source_pattern_lifecycle_v6.py', 'run_bran_source_pattern_v6_attempt1.sh'}
    return {name: sha(ROOT/name) for name in sorted(names)}


def paths(stage, attempt):
    require(stage in ('pilot', 'fit') and type(attempt) is int and 1 <= attempt <= 99)
    return (ROOT/f'BRAN_SOURCE_PATTERN_V6_{stage.upper()}_ATTEMPT{attempt}',
            ROOT/'private_artifacts'/f'bran_source_pattern_v6_{stage}_attempt{attempt}')


def progress(out, state, phase, fold=None, role=None, updates=0):
    require(phase in PHASES and (fold is None or type(fold) is int and fold in range(5)))
    require(role is None or role in ROLES)
    require(type(updates) is int and 0 <= updates <= 3000)
    state['phase'] = phase
    write_json(out/'progress.next.json', {'phase': phase, 'fold': fold, 'role': role,
        'updates_completed': updates, 'pid': os.getpid(), 'patient_level_output_emitted': False})
    os.replace(out/'progress.next.json', out/'progress.json')


def check_counters(value, role, budget):
    keys = {'cap_applied_updates', 'cap_contract_checks', 'source_generative_nonzero_updates',
        'source_cbc_nonzero_updates', 'preservation_supported_updates',
        'bridge_screen_supervised_updates', 'bridge_cbc_supervised_updates'}
    require(type(value) is dict and set(value) == keys and role in ROLES)
    require(all(type(n) is int and 0 <= n <= budget for n in value.values()))
    require(value['cap_contract_checks'] == budget)
    require(all(value[k] > 0 for k in ('source_generative_nonzero_updates',
                                      'source_cbc_nonzero_updates', 'preservation_supported_updates')))
    for key in ('bridge_screen_supervised_updates', 'bridge_cbc_supervised_updates'):
        require(value[key] > 0 if role == 'S' else value[key] == 0)


def authenticate(stage, attempt):
    out, private = paths(stage, attempt)
    require(out.is_dir() and not out.is_symlink() and not (out/'failure.json').exists())
    p, a, t = (json.loads((out/n).read_text()) for n in ('protocol.json', 'aggregate.json', 'completed.json'))
    require(t['status'] == 'authenticated_completed' and t['protocol_sha256'] == sha(out/'protocol.json')
        and t['aggregate_sha256'] == sha(out/'aggregate.json') and p['stage'] == stage
        and p['parameters'] == PARAMETERS and p['evaluation'] == EVALUATION
        and p['code_sha256'] == code_hashes() and a['status'] == 'completed'
        and a['patient_level_output_emitted'] is False and a['candidate_promoted'] is False)
    expected = {f'fold{f}_{r}.json' for f in ([0] if stage == 'pilot' else range(5)) for r in ROLES}
    require(set(a['component_sha256']) == expected)
    components = {}
    for fold in ([0] if stage == 'pilot' else range(5)):
        matched = None
        for role in ROLES:
            name = f'fold{fold}_{role}.json'
            require(sha(out/name) == a['component_sha256'][name])
            item = json.loads((out/name).read_text())
            budget = 100 if stage == 'pilot' else 3000
            require(item['updates_completed'] == budget and item['role'] == role and item['fold'] == fold
                    and item['patient_level_output_emitted'] is False)
            check_counters(item['algorithm_update_counters'], role, budget)
            traces = {k: item[k] for k in TRACES}
            require(all(type(v) is str and len(v) == 64 and set(v) <= set('0123456789abcdef') for v in traces.values()))
            if matched is not None:
                require(matched == traces)
            matched = traces
            if stage == 'fit':
                b = item['binding']
                require(b['protocol_sha256'] == t['protocol_sha256'] and b['fold'] == fold and b['role'] == role
                    and b['outer_fold_sha256'] == p['source_binding']['outer_fold_sha256']
                    and b['inner_fold_sha256'] == p['source_binding']['inner_fold_sha256'][fold]
                    and b['initial_checkpoint_sha256'] == p['parent_checkpoints'][fold]['checkpoint_sha256']
                    and item['checkpoint_reload_exact'] is True)
                load_checkpoint(private/f'fold{fold}_{role}.pt', item['checkpoint_sha256'], b)
            components[(role, fold)] = item
    return p, a, components, {'protocol_sha256': t['protocol_sha256'],
                             'aggregate_sha256': t['aggregate_sha256'],
                             'terminal_sha256': sha(out/'completed.json')}


def load_checkpoint(path, pin, binding):
    import torch
    from bran_multisource_anchored_model_v3 import BRANMultisourceAnchoredModelV3
    from bran_multisource_protocol_v2 import digest
    from run_bran_multisource_fit_v3 import _restore_transform
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
            and path.stat().st_mode & 0o777 == 0o600 and sha(path) == pin)
    p = torch.load(path, map_location='cpu', weights_only=True)
    require(set(p) == {'schema', 'config', 'binding', 'updates', 'state_dict', 'optimizer_state', 'input_transform'}
        and p['schema'] == 'bran-multisource-continuation-checkpoint-v3'
        and p['binding'] == binding and p['updates'] == 3000
        and binding['role'] in ROLES and type(binding['fold']) is int and binding['fold'] in range(5)
        and binding['training_recipe'] == 'source_pattern_supervision_v6')
    model = BRANMultisourceAnchoredModelV3.from_config(p['config'])
    require(digest(model.export_config()) == binding['model_config_sha256'])
    require(all(not v.is_floating_point() or bool(torch.isfinite(v).all()) for v in p['state_dict'].values()))
    model.load_state_dict(p['state_dict'], strict=True)
    transform = _restore_transform(p['input_transform'], binding['transform_sha256'])
    require(transform.heldout_fold == binding['fold'] and sha(path) == pin)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, transform


def bind_v5(sources, components, fold):
    """Keep factories/normalizers, replace native initialization AND scale by V5."""
    import numpy as np
    import torch
    from bran_multisource_binding_v3 import bind_fold
    from bran_multisource_batches_v2 import tensor, transform_hash
    from bran_multisource_outcomes_v2 import original_age, subset_age
    from bran_multisource_age_v2 import normalize_age
    import run_bran_context_preservation_v5 as v5
    bound = bind_fold(sources, fold)
    item = components[('M', fold)]
    _, private = v5.paths('fit', 2)
    model, transform = v5.oldfit.load_checkpoint(private/f'fold{fold}_M.pt', item['checkpoint_sha256'], item['binding'])
    require(transform_hash(transform) == transform_hash(bound.transform))
    bound.model = model
    bound.teacher = copy.deepcopy(model)
    c, cm = transform.clinical(sources.paired.c, sources.paired.cm)
    r, rm = transform.retinal(sources.paired.r, sources.paired.rm)
    age, rows = original_age(sources.paired), np.flatnonzero(sources.paired.folds != fold)
    states = []
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            idx = rows[start:start+256]
            a = normalize_age(subset_age(age, idx), transform.age_mean, transform.age_scale)
            state = model.encode(tensor(c[idx]), tensor(cm[idx], torch.bool), tensor(r[idx,None]),
                                 tensor(rm[idx,None], torch.bool), a)
            states.append(state.mean[~state.abstain])
    joined = torch.cat(states)
    require(len(joined) >= 20)
    bound.state_scale = joined.std(dim=0, correction=0).clamp_min(1).detach()
    require(bool(torch.isfinite(bound.state_scale).all()))
    return bound


def run(stage, attempt, state):
    import torch
    import run_bran_v5_cbc_uncertainty as parent
    from bran_multisource_binding_v3 import load_bound_sources
    from bran_multisource_batches_v2 import transform_hash
    from bran_multisource_protocol_v2 import digest
    from bran_multisource_fit_v2 import _exposure
    from bran_multisource_fit_v6 import fit_one_v6
    from run_bran_multisource_fit_v3 import save_checkpoint
    out, private = paths(stage, attempt)
    require(not out.exists() and not out.is_symlink() and not private.exists() and not private.is_symlink())
    out.mkdir(); state['out'] = out
    progress(out, state, 'authentication')
    dependency = None if stage == 'pilot' else authenticate('pilot', attempt)
    baseline, vp, _, components = parent.small_authentication()
    progress(out, state, 'source_loading')
    sources = load_bound_sources()
    require(vp['source_binding'] == sources.receipt())
    if dependency is not None:
        require(dependency[0]['source_binding'] == sources.receipt())
    protocol = {'schema': 'bran-source-pattern-v6-protocol', 'stage': stage,
        'status': 'frozen_before_training', 'parameters': PARAMETERS, 'evaluation': EVALUATION,
        'code_sha256': code_hashes(), 'source_binding': sources.receipt(),
        'baseline_record_sha256': sha(parent.BASELINE), 'parent_checkpoints': baseline['fold_checkpoints'],
        'pilot_dependency': None if dependency is None else dependency[3],
        'patient_level_output_emitted': False, 'candidate_promoted': False}
    write_json(out/'protocol.json', protocol); pin = sha(out/'protocol.json')
    budget, folds = (100, [0]) if stage == 'pilot' else (3000, range(5))
    if stage == 'fit':
        private.mkdir(mode=0o700)
    receipts, runtime = {}, 0.
    for fold in folds:
        progress(out, state, 'parent_binding', fold)
        bound = bind_v5(sources, components, fold)
        matched = None
        for role in ROLES:
            phase = 'pilot_training' if stage == 'pilot' else 'training'
            progress(out, state, phase, fold, role)
            result = fit_one_v6(bound.model, bound.teacher, bound.paired_factory, bound.source_factory,
                bound.state_scale, bound.transform.age_mean, bound.transform.age_scale, bound.seed,
                bound.cbc_indices, role, lambda n: progress(out, state, phase, fold, role, n), updates=budget)
            traces = {k: result[k] for k in TRACES}
            if matched is not None:
                require(matched == traces)
            matched = traces
            check_counters(result['algorithm_update_counters'], role, budget)
            runtime += result['elapsed_seconds']
            item = {'role': role, 'fold': fold, 'updates_completed': budget,
                'runtime_seconds': result['elapsed_seconds'], 'patient_level_output_emitted': False,
                'algorithm_update_counters': result['algorithm_update_counters'], **traces}
            if stage == 'fit':
                b = {'protocol_sha256': pin, 'fold': fold, 'role': role,
                    'training_recipe': 'source_pattern_supervision_v6',
                    'initial_checkpoint_sha256': components[('M',fold)]['checkpoint_sha256'],
                    'transform_sha256': transform_hash(bound.transform),
                    'model_config_sha256': digest(result['model'].export_config()),
                    'clinical_field_order_sha256': digest(sources.paired.names),
                    'outer_fold_sha256': sources.receipt()['outer_fold_sha256'],
                    'inner_fold_sha256': sources.receipt()['inner_fold_sha256'][fold]}
                progress(out, state, 'checkpoint_replay', fold, role, budget)
                checkpoint = private/f'fold{fold}_{role}.pt'
                checkpoint_pin = save_checkpoint(checkpoint, result, bound, b)
                restored, _ = load_checkpoint(checkpoint, checkpoint_pin, b)
                require(all(torch.equal(v, restored.state_dict()[k]) for k,v in result['model'].state_dict().items()))
                item.update(binding=b, checkpoint_sha256=checkpoint_pin, checkpoint_reload_exact=True,
                    exposure=_exposure(result['source_sampler'].private_sampler,
                        result['paired_sampler'].private_sampler, sources.paired, sources.pools))
                del restored
            name = f'fold{fold}_{role}.json'; write_json(out/name, item); receipts[name] = sha(out/name)
            del result
    progress(out, state, 'post_authentication')
    require(parent.small_authentication()[0] == baseline and load_bound_sources().receipt() == protocol['source_binding'])
    require(code_hashes() == protocol['code_sha256'] and sha(out/'protocol.json') == pin)
    value = {'schema': 'bran-source-pattern-v6-training', 'status': 'completed', 'stage': stage,
        'component_sha256': receipts, 'total_training_loop_seconds': runtime,
        'projected_full_training_loop_seconds': runtime*150 if stage == 'pilot' else None,
        'prediction_evaluation_performed': False, 'pilot_models_discarded': stage == 'pilot',
        'ten_final_checkpoints_reloaded': stage == 'fit', 'matched_all_streams': True,
        'protected_sources_used': False, 'patient_level_output_emitted': False,
        'candidate_promoted': False, 'scientific_goal_achieved': False}
    write_json(out/'aggregate.json', value)
    progress(out, state, 'completed')
    require(not (out/'failure.json').exists())
    write_json(out/'completed.json', {'status': 'authenticated_completed',
        'protocol_sha256': pin, 'aggregate_sha256': sha(out/'aggregate.json'),
        'patient_level_output_emitted': False})


def safe_site(exc):
    """Code location only; never exception text, locals or source lines."""
    tb, result = exc.__traceback__, None
    while tb is not None:
        name = Path(tb.tb_frame.f_code.co_filename).name
        if name.startswith(('bran_', 'run_bran_')) and name.endswith('.py'):
            result = {'file': name, 'line': tb.tb_lineno}
        tb = tb.tb_next
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('pilot', 'fit'), required=True)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args(); state = {'out': None, 'phase': 'authentication'}; ok = False
    with quiet():
        try:
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                import torch
                torch.set_num_threads(2)
                run(args.stage, args.attempt, state); ok = True
        except Exception as exc:
            out = state['out']
            if out is not None and not (out/'completed.json').exists():
                write_json(out/'failure.json', {'status': 'technical_failure', 'phase': state['phase'],
                    'safe_code_site': safe_site(exc), 'completed_components_preserved': True,
                    'patient_level_output_emitted': False, 'candidate_promoted': False})
    print(json.dumps({'status': 'completed' if ok else 'not_completed', 'stage': args.stage,
        'phase': state['phase'], 'patient_level_output_emitted': False, 'candidate_promoted': False}))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
