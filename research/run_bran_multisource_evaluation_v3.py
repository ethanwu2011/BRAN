"""Exclusive local V3 evaluation; original gates and role labels remain explicit.

No training/promotion/retuning. Receipts contain only disclosure-safe aggregates.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json
import run_bran_multisource_fit_v3 as fit
from bran_multisource_binding_v3 import load_bound_sources
import bran_multisource_reference_predictions_v2 as references
import bran_multisource_comparison_v3 as comparison
import bran_multisource_profiles_v3 as profiles
from run_bran_multisource_advancement_v2 import CODE as LEGACY_EVAL_CODE

ROOT = Path(__file__).resolve().parent
CODE = tuple(sorted(set(fit.CODE) | set(LEGACY_EVAL_CODE) | {
    'run_bran_multisource_evaluation_v3.py', 'bran_multisource_comparison_v3.py',
    'bran_multisource_profiles_v3.py',
    'bran_multisource_inference_v2.py', 'bran_multisource_outcome_metrics_v2.py',
    'bran_multisource_outcomes_v2.py'}))
PHASES = ('fit_authentication', 'reference_loading', 'reference_inference',
    'reference_replay', 'reference_bootstrap', 'candidate_inference',
    'advancement_bootstrap', 'candidate_aggregate_bootstrap', 'native_profiles_completed',
    'result_authentication', 'evaluation_completed')


def require(ok):
    if not ok: raise ValueError('multisource_evaluation_v3_failed')


def code_hashes():
    return {name: sha(ROOT/name) for name in CODE}


def authenticate_fits(attempt):
    out, private = fit.paths(attempt)
    require(not (out/'failure.json').exists() and not private.is_symlink()
        and private.is_dir() and private.stat().st_mode & 0o777 == 0o700)
    p = json.loads((out/'protocol.json').read_text())
    a = json.loads((out/'aggregate.json').read_text())
    m = json.loads((out/'manifest.json').read_text())
    pin = sha(out/'protocol.json')
    require(p['schema'] == 'bran-multisource-continuation-protocol-v3'
        and p['parameters'] == fit.PARAMETERS and p['evaluation'] == fit.EVALUATION
        and p['code_sha256'] == fit.code_hashes()
        and a['status'] == 'fits_completed_pending_evaluation'
        and a['protocol_sha256'] == pin and a['patient_level_output_emitted'] is False
        and a['candidate_promoted'] is False and a['ten_final_checkpoints_reloaded'] is True
        and a['paired_role_inputs_and_masks_identical'] is True and a['five_folds_completed'] is True
        and m == {'aggregate_sha256': sha(out/'aggregate.json'), 'protocol_sha256': pin,
                  'component_sha256': a['component_sha256'], 'patient_level_output_emitted': False})
    require(set(a['component_sha256']) == {f'fold{f}_{r}.json' for f in range(5) for r in fit.ROLES})
    sources = load_bound_sources()
    require(sources.receipt() == p['source_binding'])
    components = {}
    source_influence = []
    for fold in range(5):
        traces = []
        model_pair = {}
        for role in fit.ROLES:
            name = f'fold{fold}_{role}'
            require(sha(out/(name+'.json')) == a['component_sha256'][name+'.json'])
            item = json.loads((out/(name+'.json')).read_text())
            b = item['binding']
            require(b['protocol_sha256'] == pin and b['fold'] == fold and b['role'] == role
                and b['outer_fold_sha256'] == p['source_binding']['outer_fold_sha256']
                and b['inner_fold_sha256'] == p['source_binding']['inner_fold_sha256'][fold]
                and item['updates_completed'] == 3000 and item['checkpoint_reload_exact'] is True
                and item['source_loss_supported'] is (role == 'M')
                and item['patient_level_output_emitted'] is False and item['candidate_promoted'] is False)
            loaded, _ = fit.load_checkpoint(private/(name+'.pt'), item['checkpoint_sha256'], b)
            model_pair[role] = loaded
            components[(role, fold)] = item
            traces.append((item['paired_input_digest'], item['paired_mask_digest']))
        require(traces[0] == traces[1])
        import torch
        control = model_pair['C'].state_dict()
        multisource = model_pair['M'].state_dict()
        encoder_prefixes = ('clinical_encoder.', 'clinical_residual.', 'retinal_projection.',
                            'retinal_pool.', 'shared_posterior.', 'retinal_delta.', 'clinical_delta.')
        changed = any(not torch.equal(control[name], multisource[name])
                      for name in control if name.startswith(encoder_prefixes))
        source_influence.append({'fold': fold, 'encoder_differs_between_matched_arms': changed})
    return sources, components, {'fit_attempt': attempt, 'protocol_sha256': pin,
        'aggregate_sha256': sha(out/'aggregate.json'), 'manifest_sha256': sha(out/'manifest.json'),
        'component_sha256': a['component_sha256'], 'source_binding': p['source_binding'],
        'source_influence_diagnostic': source_influence,
        'parameter_difference_is_not_evidence_of_better_performance': True}


def progress(out, state, event):
    require(set(event) <= {'phase', 'role', 'fold'} and event['phase'] in PHASES)
    require('role' not in event or event['role'] in ('C', 'M'))
    require('fold' not in event or type(event['fold']) is int and event['fold'] in range(5))
    state['phase'] = event['phase']
    tmp = out/'progress.next.json'
    write_json(tmp, {**event, 'pid': os.getpid(), 'patient_level_output_emitted': False})
    os.replace(tmp, out/'progress.json')


def validate_result(value):
    require(type(value) is dict and set(value) == {
        'schema', 'status', 'historical_comparisons', 'matched_m_vs_c',
        'blood_completion_profiles', 'historical_replay', 'eligibility',
        'promotion_eligible', 'candidate_promoted', 'historical_gate_definitions_changed',
        'named_fm_benchmark_complete', 'external_validation_complete', 'subtyping_established',
        'scientific_goal_achieved', 'patient_level_output_emitted'})
    require(value['schema'] == 'bran-multisource-comparison-v3'
        and value['patient_level_output_emitted'] is False
        and value['candidate_promoted'] is False
        and value['historical_gate_definitions_changed'] is False
        and value['scientific_goal_achieved'] is False
        and set(value['historical_comparisons']) == {'C', 'M'})
    for role, item in value['historical_comparisons'].items():
        require(item['candidate_role'] == role)
        require(item['decisions'] == comparison.gates.contract.decisions(
            item['screening'], item['completion'], item['missingness'], item['completion_no_retina']))
    item = value['matched_m_vs_c']
    require(item['candidate_role'] == 'M' and item['prospective_matched_comparison'] is True)
    require(item['decisions'] == comparison.gates.contract.decisions(
        item['screening'], item['completion'], item['missingness'], item['completion_no_retina']))
    eligible = bool(value['historical_comparisons']['M']['decisions']['advancement_supported']
                    and item['decisions']['advancement_supported'])
    require(value['promotion_eligible'] is eligible and value['eligibility'] == {'C': False, 'M': eligible})
    profiles = value['blood_completion_profiles']
    require(type(profiles) is dict and set(profiles) == {
        'descriptive_only', 'low_hb_below_12_research_stratum_only', 'contexts',
        'evaluated_contexts', 'other_two_training_contexts_profile_pending'}
        and profiles['descriptive_only'] is True
        and set(profiles['contexts']) == set(comparison.gates.contract.EVALPATTERNS))
    # This serializability check also rejects NaN/Inf and accidental tensors or
    # patient-array containers. Closed imported metrics enforce cell suppression.
    json.dumps(value, sort_keys=True, allow_nan=False)


def execute(out, fit_attempt, state):
    start = time.perf_counter()
    progress(out, state, {'phase': 'fit_authentication'})
    code = code_hashes()
    sources, components, binding = authenticate_fits(fit_attempt)
    protocol = {'schema': 'bran-multisource-evaluation-protocol-v3',
        'status': 'frozen_before_candidate_scoring', 'code_sha256': code,
        'fit_binding': binding, 'evaluation': fit.EVALUATION,
        'historical_replay_tolerance': 1e-10, 'historical_gates_unchanged': True,
        'completion_gate_contexts': list(comparison.gates.contract.EVALPATTERNS),
        'native_profile_completion_contexts': list(profiles.PARAMETERS['completion_patterns']),
        'native_profile_age_scenarios': list(profiles.AGE_SCENARIOS),
        'native_profiles_descriptive_only_not_gate_redefinitions': True,
        'raw_reference_readouts_replayed_under_their_original_recipe': True,
        'encoder_or_native_head_fitting': False, 'automatic_promotion': False,
        'patient_level_output_permitted': False}
    write_json(out/'protocol.json', protocol)
    pin = sha(out/'protocol.json')
    _, private = fit.paths(fit_attempt)

    def provider(role, fold):
        require(role in fit.ROLES and type(fold) is int and fold in range(5))
        item = components[(role, fold)]
        return fit.load_checkpoint(private/f'fold{fold}_{role}.pt',
                                   item['checkpoint_sha256'], item['binding'])

    native = profiles.evaluate(sources.paired, provider,
        progress=lambda event: progress(out, state, event))
    require(native['schema'] == 'bran-multisource-native-profiles-v3'
        and native['reload_predictions_equal'] is True and native['all_empty_physiology_abstained'] is True
        and native['patient_level_output_emitted'] is False and native['candidate_promoted'] is False)
    require(code_hashes() == code and sha(out/'protocol.json') == pin)
    write_json(out/'native_profiles.json', native)
    write_json(out/'native_profiles_manifest.json', {'native_profiles_sha256': sha(out/'native_profiles.json'),
        'protocol_sha256': pin, 'fit_binding': binding, 'code_sha256': code,
        'status': 'native_profiles_replayed_not_advancement', 'patient_level_output_emitted': False})
    progress(out, state, {'phase': 'native_profiles_completed'})
    reference = references.build(sources.paired, sources.context,
        progress=lambda event: progress(out, state, event))
    value = comparison.evaluate(sources.paired, reference, provider,
                                progress=lambda event: progress(out, state, event))
    progress(out, state, {'phase': 'result_authentication'})
    require(code_hashes() == code and sha(out/'protocol.json') == pin)
    _, _, rebound = authenticate_fits(fit_attempt)
    require(rebound == binding)
    validate_result(value)
    write_json(out/'aggregate.json', {**value, 'protocol_sha256': pin, 'fit_binding': binding})
    write_json(out/'manifest.json', {'aggregate_sha256': sha(out/'aggregate.json'),
        'protocol_sha256': pin, 'elapsed_seconds': time.perf_counter()-start,
        'native_profiles_manifest_sha256': sha(out/'native_profiles_manifest.json'),
        'patient_level_output_emitted': False})
    write_json(out/'audit.json', {'status': 'artifact_bindings_and_gate_decisions_authenticated',
        'aggregate_sha256': sha(out/'aggregate.json'), 'protocol_sha256': pin,
        'historical_metric_replay_passed': True, 'ten_candidate_checkpoints_reloaded': True,
        'source_and_code_bindings_reauthenticated': True,
        'native_route_completion_stress_age_predictions_replayed': True,
        'independent_candidate_aggregate_replay_performed': False,
        'candidate_promoted': False, 'patient_level_output_emitted': False})
    progress(out, state, {'phase': 'evaluation_completed'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--fit-attempt', type=int, default=1)
    args = parser.parse_args()
    require(type(args.attempt) is int and 1 <= args.attempt <= 99)
    out = ROOT/f'BRAN_MULTISOURCE_EVALUATION_V3_ATTEMPT{args.attempt}'
    require(not out.exists() and not out.is_symlink())
    state, ok = {'phase': 'lock_acquisition'}, False
    with quiet():
        try:
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                out.mkdir()
                execute(out, args.fit_attempt, state)
                ok = True
        except Exception:
            if out.is_dir() and not (out/'aggregate.json').exists():
                write_json(out/'failure.json', {'status': 'technical_evaluation_failure',
                    'phase': state['phase'], 'patient_level_output_emitted': False,
                    'candidate_promoted': False})
    print(json.dumps({'status': 'evaluation_completed_pending_author_review' if ok else 'evaluation_not_completed',
        'phase': state['phase'], 'patient_level_output_emitted': False, 'candidate_promoted': False}))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
