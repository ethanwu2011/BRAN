"""Quiet, sequential candidate profiling after all V2 fits authenticate.

Run prepare only after the fitting wrapper's terminal audit. This job does not
replace the historical-reference advancement test, choose an encoder, train
readouts, calibrate intervals or run external validation.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path

from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json
import run_bran_multisource_fit_v2 as fit
import bran_multisource_outcomes_v2 as outcomes
from bran_multisource_protocol_v2 import digest

ROOT = Path(__file__).resolve().parent
FIT_ATTEMPT = 1
FIT_PIN = '98ad5e161324ec30dc7463912d6e59d7aa8c9b36c4976f40a2d93e907a52cc30'
CODE = ('run_bran_multisource_outcomes_v2.py', 'bran_multisource_outcomes_v2.py',
        'bran_multisource_outcome_metrics_v2.py', 'bran_multisource_inference_v2.py',
        'bran_missingness_stress_v1.py', 'bran_missingness_stress_metrics_v1.py',
        'bran_external_cbc_evaluation_v1.py', 'run_bran_overnight_diagnostic_v1.py')


def require(ok):
    if not ok: raise ValueError('multisource_outcome_runner_failed')


def paths(attempt):
    require(type(attempt) is int and 1 <= attempt <= 99)
    return ROOT / ('BRAN_MULTISOURCE_OUTCOMES_V2_ATTEMPT'+str(attempt))


def code_hashes(): return {name: sha(ROOT/name) for name in CODE}


def read(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1)
    return json.loads(path.read_text())


def authenticated_fit():
    out, private = fit.paths(FIT_ATTEMPT)
    require(not (out/'failure.json').exists() and not (out/'audit_failure.json').exists())
    p, sources = fit.authenticate(FIT_ATTEMPT, FIT_PIN)
    a, m, audit = (read(out/name) for name in ('aggregate.json','manifest.json','audit.json'))
    require(a['status'] == 'completed_fits_pending_evaluation' and a['protocol_sha256'] == FIT_PIN
            and a['protocol_content_sha256'] == digest(p)
            and a['paired_arm_input_and_mask_traces_identical'] is True
            and a['five_folds_completed'] is True and tuple(a['arms']) == outcomes.ARMS
            and a['candidate_promoted'] is False and a['scientific_goal_achieved'] is False
            and a['patient_level_output_emitted'] is False)
    require(m == {'protocol_sha256':FIT_PIN, 'aggregate_sha256':sha(out/'aggregate.json'),
                  'component_sha256':a['component_sha256'], 'patient_level_output_emitted':False})
    require(audit == {'status':'checkpoints_authenticated_pending_outcome_evaluation',
                     'protocol_sha256':FIT_PIN, 'aggregate_sha256':sha(out/'aggregate.json'),
                     'ten_final_checkpoints_reloaded':True, 'candidate_promoted':False,
                     'scientific_goal_achieved':False, 'patient_level_output_emitted':False})
    require(set(a['component_sha256']) == {'fold'+str(f)+'_'+arm+'.json' for f in range(5) for arm in outcomes.ARMS})
    components = {}
    for name, pin in a['component_sha256'].items():
        require(sha(out/name) == pin); item = read(out/name)
        require(item['status'] == 'fit_completed_not_evaluated' and item['checkpoint_reload_exact'] is True
                and item['binding']['protocol_sha256'] == digest(p)
                and item['binding']['outer_folds_sha256'] == p['outer_folds_sha256']
                and item['candidate_promoted'] is False and item['patient_level_output_emitted'] is False)
        components[name] = item
    for fold in range(5):
        require(all(components['fold'+str(fold)+'_mlp.json'][key] == components['fold'+str(fold)+'_token.json'][key]
                    for key in ('input_sequence_sha256','mask_sequence_sha256')))
    pins = {'fit_protocol_sha256':FIT_PIN, 'fit_aggregate_sha256':sha(out/'aggregate.json'),
            'fit_manifest_sha256':sha(out/'manifest.json'), 'fit_audit_sha256':sha(out/'audit.json'),
            'component_sha256':a['component_sha256']}
    return p, sources.paired, components, pins


def spec(p, pins):
    require(p['parameters']['stage_c_steps'] == 1500)
    return {'schema':'bran-multisource-native-outcomes-protocol-v2', 'status':'frozen_before_outcome_evaluation',
            'fit_binding':pins, 'code_sha256':code_hashes(),
            'outer_folds_sha256':p['outer_folds_sha256'], 'inner_folds_sha256':p['inner_folds_sha256'],
            'transform_sha256':p['transform_sha256'], 'arms':list(outcomes.ARMS),
            'final_stage_c_steps':p['parameters']['stage_c_steps'],
            'completion_patterns':p['parameters']['completion_patterns'],
            'masking_patterns':list(outcomes.masking.PATTERNS), 'age_scenarios':list(outcomes.AGE_SCENARIOS),
            'bootstrap_draws':1000, 'bootstrap_seed':91501, 'minimum_valid_draws':900,
            'low_hb_definition':'observed hemoglobin <12 g/dL, research stratum, not demographic clinical diagnosis',
            'purpose':'native candidate profiles before historical-reference advancement and uniform FM benchmark',
            'clinical_use':False, 'automatic_promotion':False, 'patient_level_output_permitted':False}


def prepare(attempt, state):
    out = paths(attempt); require(not out.exists() and not out.is_symlink())
    # Do not create a failure directory merely because training is still active.
    p, _, _, pins = authenticated_fit()
    out.mkdir(); state['owned'] = out
    write_json(out/'protocol.json', spec(p, pins)); state['owned'] = None
    return sha(out/'protocol.json')


def authenticate(attempt, pin):
    out = paths(attempt)
    require(out.is_dir() and not out.is_symlink() and sha(out/'protocol.json') == pin
            and not (out/'failure.json').exists() and not (out/'audit_failure.json').exists())
    saved = read(out/'protocol.json'); p, paired, components, pins = authenticated_fit()
    require(saved == spec(p, pins))
    return saved, paired, components


def progress(out, state, event):
    require(set(event) <= {'phase','arm','fold'}
            and event['phase'] in ('candidate_inference','candidate_aggregate_bootstrap'))
    state['phase'] = event['phase']
    temporary = out/'progress.tmp'
    write_json(temporary, {**event, 'pid':os.getpid(), 'patient_level_output_emitted':False})
    os.replace(temporary, out/'progress.json')


def compute(attempt, pin, state):
    out = paths(attempt); saved, paired, components = authenticate(attempt, pin)
    _, private = fit.paths(FIT_ATTEMPT)
    def provider(arm, fold):
        require(arm in outcomes.ARMS and fold in range(5))
        name = 'fold'+str(fold)+'_'+arm; item = components[name+'.json']
        require(item['binding']['arm'] == arm and item['binding']['fold'] == fold
                and item['binding']['inner_fold_sha256'] == saved['inner_folds_sha256'][fold])
        steps = saved['final_stage_c_steps']; require(steps == 1500)
        checkpoint = 'stage_C_step_'+str(steps)+'.pt'
        return fit.load_checkpoint(private/name/checkpoint,
                    expected_sha256=item['checkpoint_manifest'][checkpoint]['sha256'],
                    binding=item['binding'],stage='C',steps=steps)
    result = outcomes.evaluate(paired, provider, progress=lambda event:progress(out,state,event))
    result['protocol_sha256'] = pin; result['fit_binding'] = saved['fit_binding']
    authenticate(attempt, pin)
    return result


def run(attempt, pin, state):
    out = paths(attempt)
    require(not any((out/name).exists() for name in ('aggregate.json','manifest.json')))
    state.update(owned=out,phase='outcome_authentication')
    result = compute(attempt,pin,state)
    write_json(out/'aggregate.json',result)
    write_json(out/'manifest.json',{'protocol_sha256':pin, 'aggregate_sha256':sha(out/'aggregate.json'),
                                  'patient_level_output_emitted':False})
    state['owned'] = None


def audit(attempt, pin, state):
    out = paths(attempt)
    require(not any((out/name).exists() for name in ('audit.json','audit_failure.json')))
    state.update(owned=out,phase='outcome_replay',failure_name='audit_failure.json')
    result = read(out/'aggregate.json'); aggregate_pin = sha(out/'aggregate.json')
    require(read(out/'manifest.json') == {'protocol_sha256':pin,'aggregate_sha256':aggregate_pin,
                                       'patient_level_output_emitted':False})
    replay = compute(attempt,pin,state)
    require(result == replay and sha(out/'aggregate.json') == aggregate_pin)
    write_json(out/'audit.json',{'status':'native_profiles_replayed_not_promoted', 'protocol_sha256':pin,
               'aggregate_sha256':aggregate_pin, 'all_native_aggregates_replayed':True,
               'candidate_training_repeated':False, 'historical_reference_gates_evaluated':False,
               'candidate_promoted':False, 'scientific_goal_achieved':False,'patient_level_output_emitted':False})
    state['owned'] = None


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action',choices=('prepare','run','audit'))
    parser.add_argument('--attempt',type=int,default=1); parser.add_argument('--protocol-sha256')
    args = parser.parse_args(); state = {'owned':None,'phase':'outcome_authentication','failure_name':'failure.json'}
    answer = {'status':'failed','candidate_promoted':False,'patient_level_output_emitted':False}
    with quiet():
        try:
            with LOCK.open('a') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                if args.action == 'prepare': pin = prepare(args.attempt,state)
                elif args.action == 'run': pin = args.protocol_sha256; run(args.attempt,pin,state)
                else: pin = args.protocol_sha256; audit(args.attempt,pin,state)
                answer.update(status=args.action+'_completed',protocol_sha256=pin)
        except Exception:
            if state['owned'] is not None:
                try: write_json(state['owned']/state['failure_name'],{'status':'technical_failure',
                    'phase':state['phase'],'reason':'candidate_profile_contract_failed',
                    'candidate_promoted':False,'patient_level_output_emitted':False})
                except Exception: pass
    print(json.dumps(answer)); return int(answer['status']=='failed')


if __name__ == '__main__': raise SystemExit(main())
