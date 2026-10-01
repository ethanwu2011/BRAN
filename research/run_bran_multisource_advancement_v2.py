"""Single-job V2 advancement comparison using immutable historical references.

Prepare only after native candidate profile audit. Historical recipe/roles and
gates are fixed here before candidate outcomes; no encoder/readout selection,
scientific retuning, external evaluation or automatic promotion occurs.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path

from run_bran_multisource_retinal_features_v2 import LOCK,quiet,sha,write_json
import run_bran_multisource_outcomes_v2 as profiles
import bran_multisource_reference_roles_v2 as roles
import bran_multisource_reference_predictions_v2 as references
import bran_multisource_comparison_v2 as comparison

ROOT=Path(__file__).resolve().parent
REFERENCE_PROTOCOL_PIN='ce793a9f902b27b6f5c09f48c8b659ee98c5c5d251d61436014e505e1642d5da'
REFERENCE_AUDIT_PIN='4191cdb3b6965ac65d6c0eb43d6613079668b0065178801713953304def0020c'
CODE=('run_bran_multisource_advancement_v2.py','bran_multisource_reference_roles_v2.py',
      'bran_multisource_reference_predictions_v2.py','bran_multisource_comparison_v2.py',
      'bran_multisource_advancement_v2.py','bran_multisource_mask_contract_v1.py',
      'bran_distillation_metrics_v1.py','run_bran_native_rehearsal_v1.py',
      'run_bran_multisource_mask_v1.py','bran_multisource_mask_experiment_v1.py',
      'run_bran_retinal_input_bridge_v3.py','run_bran_raw_teacher_distillation_v1.py',
      'run_bran_native_screening_v1.py','bran_native_screening_kernel_v1.py',
      'run_bran_screening_joint_v1.py','run_bran_modality_union_v1.py',
      'bran_september_push_io_v1.py','run_bran_overnight_diagnostic_v1.py',
      'bran_matched_screening_kernel_v1.py','bran_external_cbc_evaluation_v1.py',
      'bran_missingness_stress_v1.py','bran_missingness_stress_metrics_v1.py')
PHASES=('reference_loading','reference_inference','reference_replay','reference_bootstrap',
        'candidate_inference','advancement_bootstrap')


def require(ok):
    if not ok:raise ValueError('multisource_advancement_runner_failed')


def paths(attempt):
    require(type(attempt) is int and 1<=attempt<=99)
    return ROOT/('BRAN_MULTISOURCE_ADVANCEMENT_V2_ATTEMPT'+str(attempt))


def code_hashes():return {name:sha(ROOT/name) for name in CODE}


def source_context():
    # This is allowed only after the sole fit and native-profile jobs finish.
    out=profiles.paths(1);pin=sha(out/'protocol.json')
    p,paired,components=profiles.authenticate(1,pin)
    aggregate_pin=sha(out/'aggregate.json')
    require(profiles.read(out/'manifest.json')=={'protocol_sha256':pin,
        'aggregate_sha256':aggregate_pin,'patient_level_output_emitted':False})
    require(profiles.read(out/'audit.json')=={'status':'native_profiles_replayed_not_promoted',
        'protocol_sha256':pin,'aggregate_sha256':aggregate_pin,'all_native_aggregates_replayed':True,
        'candidate_training_repeated':False,'historical_reference_gates_evaluated':False,
        'candidate_promoted':False,'scientific_goal_achieved':False,'patient_level_output_emitted':False})
    prior=profiles.read(out/'aggregate.json')
    require(prior['protocol_sha256']==pin and prior['fit_binding']==p['fit_binding']
            and prior['historical_reference_gates_evaluated'] is False and prior['candidate_promoted'] is False)
    reference=roles.authenticate_references(REFERENCE_PROTOCOL_PIN,REFERENCE_AUDIT_PIN)
    metadata=roles.role_description(reference)
    require(metadata['fold_authentication']['outer_fold_sha256']==p['outer_folds_sha256']
        and metadata['fold_authentication']['inner_fold_sha256']==p['inner_folds_sha256'])
    binding={'profile_protocol_sha256':pin,'profile_aggregate_sha256':aggregate_pin,
        'profile_manifest_sha256':sha(out/'manifest.json'),'profile_audit_sha256':sha(out/'audit.json'),
        'fit_binding':p['fit_binding'],'historical_roles':metadata,
        # The legacy authenticator already verifies this entire inherited
        # closure. Record it here as well as this runner's direct dependencies.
        'historical_code_sha256':dict(reference.protocol['code_sha256'])}
    return p,paired,components,reference,binding


def spec(profile,binding):
    require(profile['final_stage_c_steps']==1500)
    return {'schema':'bran-multisource-advancement-protocol-v2','status':'frozen_before_comparison',
        'source_binding':binding,'code_sha256':code_hashes(),'arms':['mlp','token'],
        'final_stage_c_steps':1500,'historical_gate_callable':'bran_multisource_mask_contract_v1.decisions',
        'role_mapping':dict(comparison.gates.ROLE_DESCRIPTION),'bootstrap_draws':1000,'bootstrap_seed':91501,
        'baseline_replay_tolerance':1e-10,'minimum_valid_draws':900,
        'raw_reference_heads_refit':True,'candidate_encoder_training':False,
        'historical_gate_definitions_changed':False,'automatic_promotion':False,'clinical_use':False,
        'patient_level_output_permitted':False,'named_fm_benchmark_is_separate':True}


def prepare(attempt,state):
    out=paths(attempt);require(not out.exists() and not out.is_symlink())
    p,_,_,_,binding=source_context()
    out.mkdir();state['owned']=out
    write_json(out/'protocol.json',spec(p,binding));state['owned']=None
    return sha(out/'protocol.json')


def authenticate(attempt,pin):
    out=paths(attempt)
    require(out.is_dir() and not out.is_symlink() and sha(out/'protocol.json')==pin
        and not (out/'failure.json').exists() and not (out/'audit_failure.json').exists())
    value=profiles.read(out/'protocol.json')
    p,paired,components,reference,binding=source_context()
    require(value==spec(p,binding))
    return value,paired,components,reference


def progress(out,state,event):
    require(set(event)<={'phase','arm','fold'} and event['phase'] in PHASES)
    if 'fold' in event:require(type(event['fold']) is int and event['fold'] in range(5))
    if 'arm' in event:require(event['arm'] in ('mlp','token'))
    state['phase']=event['phase'];temporary=out/'progress.tmp'
    write_json(temporary,{**event,'pid':os.getpid(),'patient_level_output_emitted':False})
    os.replace(temporary,out/'progress.json')


def compute(attempt,pin,state):
    out=paths(attempt);saved,paired,components,reference=authenticate(attempt,pin)
    def notify(event):progress(out,state,event)
    reference_predictions=references.build(paired,reference,progress=notify)
    _,private=profiles.fit.paths(profiles.FIT_ATTEMPT)
    def provider(arm,fold):
        require(arm in ('mlp','token') and type(fold) is int and fold in range(5))
        name='fold'+str(fold)+'_'+arm;item=components[name+'.json'];steps=saved['final_stage_c_steps']
        require(item['binding']['arm']==arm and item['binding']['fold']==fold)
        checkpoint='stage_C_step_'+str(steps)+'.pt'
        return profiles.fit.load_checkpoint(private/name/checkpoint,
            expected_sha256=item['checkpoint_manifest'][checkpoint]['sha256'],
            binding=item['binding'],stage='C',steps=steps)
    result=comparison.evaluate(paired,reference_predictions,provider,progress=notify)
    result['protocol_sha256']=pin;result['source_binding']=saved['source_binding']
    authenticate(attempt,pin)
    return result


def run(attempt,pin,state):
    out=paths(attempt);require(not any((out/name).exists() for name in ('aggregate.json','manifest.json')))
    state.update(owned=out,phase='comparison_authentication')
    result=compute(attempt,pin,state)
    write_json(out/'aggregate.json',result)
    write_json(out/'manifest.json',{'protocol_sha256':pin,'aggregate_sha256':sha(out/'aggregate.json'),
                                 'patient_level_output_emitted':False})
    state['owned']=None


def audit(attempt,pin,state):
    out=paths(attempt);require(not any((out/name).exists() for name in ('audit.json','audit_failure.json')))
    state.update(owned=out,phase='comparison_replay',failure_name='audit_failure.json')
    original=profiles.read(out/'aggregate.json');aggregate_pin=sha(out/'aggregate.json')
    require(profiles.read(out/'manifest.json')=={'protocol_sha256':pin,'aggregate_sha256':aggregate_pin,
                                              'patient_level_output_emitted':False})
    replay=compute(attempt,pin,state)
    require(replay==original and sha(out/'aggregate.json')==aggregate_pin)
    write_json(out/'audit.json',{'status':'historical_advancement_comparison_replayed',
        'protocol_sha256':pin,'aggregate_sha256':aggregate_pin,'all_aggregates_replayed':True,
        'raw_reference_heads_refit':True,'candidate_training_repeated':False,
        'historical_gate_definitions_changed':False,'candidate_promoted':False,
        'scientific_goal_achieved':False,'patient_level_output_emitted':False})
    state['owned']=None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run','audit'))
    parser.add_argument('--attempt',type=int,default=1);parser.add_argument('--protocol-sha256')
    args=parser.parse_args();state={'owned':None,'phase':'comparison_authentication','failure_name':'failure.json'}
    answer={'status':'failed','patient_level_output_emitted':False,'candidate_promoted':False}
    with quiet():
        try:
            with LOCK.open('a') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                if args.action=='prepare':pin=prepare(args.attempt,state)
                elif args.action=='run':pin=args.protocol_sha256;run(args.attempt,pin,state)
                else:pin=args.protocol_sha256;audit(args.attempt,pin,state)
                answer.update(status=args.action+'_completed',protocol_sha256=pin)
        except Exception:
            if state['owned'] is not None:
                try:write_json(state['owned']/state['failure_name'],{'status':'technical_failure',
                    'phase':state['phase'],'reason':'historical_comparison_contract_failed',
                    'patient_level_output_emitted':False,'candidate_promoted':False})
                except Exception:pass
    print(json.dumps(answer));return int(answer['status']=='failed')


if __name__=='__main__':raise SystemExit(main())
