"""Quiet native/historical R7 evaluations with explicit generic-role adapter."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from run_bran_multisource_retinal_features_v2 import LOCK,quiet,sha,write_json
import run_bran_robust_clinical_r7 as train

ROOT=Path(__file__).resolve().parent
PHASES=('authentication','source_loading','inference','aggregate_bootstrap','historical_gate','post_authentication','completed')
ERROR='robust_clinical_r7_evaluation_failed'


def require(ok):
    if not ok:raise ValueError(ERROR) from None


def paths(stage,attempt):
    require(stage in ('native','historical') and type(attempt) is int and 1<=attempt<=99)
    return ROOT/f'BRAN_ROBUST_CLINICAL_R7_{stage.upper()}_EVALUATE_ATTEMPT{attempt}'


def progress(out,state,phase,fold=None,role=None):
    require(phase in PHASES and role in (None,'V5','C','S')
        and (fold is None or type(fold) is int and fold in range(5)))
    state['phase']=phase
    write_json(out/'progress.next.json',{'phase':phase,'fold':fold,'role':role,
        'role_adapter':train.ROLE_ADAPTER,'pid':os.getpid(),'patient_level_output_emitted':False})
    os.replace(out/'progress.next.json',out/'progress.json')


def authenticate(stage,attempt,fit_attempt):
    import bran_source_pattern_metrics_v6 as gates
    from audit_bran_source_pattern_v6 import read,validate_native
    out=paths(stage,attempt)
    require(out.is_dir() and not out.is_symlink() and not (out/'failure.json').exists())
    p,a,t=(read(out/n) for n in ('protocol.json','aggregate.json','completed.json'))
    fp,fa,components,pins=train.authenticate('fit',fit_attempt)
    require(set(t)=={'status','protocol_sha256','aggregate_sha256','patient_level_output_emitted'}
        and t['status']=='authenticated_completed' and t['patient_level_output_emitted'] is False
        and t['protocol_sha256']==sha(out/'protocol.json') and t['aggregate_sha256']==sha(out/'aggregate.json'))
    require(p['schema']=='bran-robust-clinical-r7-evaluation-protocol' and p['stage']==stage
        and p['parameters']==train.EVALUATION and p['code_sha256']==train.code_hashes()
        and p['source_binding']==fp['source_binding'] and p['fit_receipt']==pins
        and p['role_adapter']==train.ROLE_ADAPTER and p['protected_sources_used'] is False
        and p['automatic_promotion'] is False and p['patient_level_output_emitted'] is False)
    require(set(a)=={'schema','status','stage','role_adapter','result','research_lead_decision',
        'historical_role_adapter','historical_gate_changed','candidate_promoted','scientific_goal_achieved',
        'patient_level_output_emitted'} and a['schema']=='bran-robust-clinical-r7-evaluation-terminal'
        and a['status']=='completed' and a['stage']==stage and a['role_adapter']==train.ROLE_ADAPTER
        and a['patient_level_output_emitted'] is False and a['candidate_promoted'] is False
        and a['scientific_goal_achieved'] is False and a['historical_gate_changed'] is False)
    if stage=='native':
        checks=validate_native(a['result'])
        require(a['research_lead_decision']==gates.decide(a['result']['stress_primary'],a['result']['screening']['both'],checks,True)
                and a['historical_role_adapter'] is None)
    else:
        import run_bran_multisource_evaluation_v3 as oldeval
        oldeval.validate_result(a['result'])
        require(a['research_lead_decision'] is None
            and a['historical_role_adapter']=={'C':'archived_V6_control_C','M':'R7_robust_input_R'})
    return a,{'status':'authenticated_completed','protocol_sha256':t['protocol_sha256'],
              'aggregate_sha256':t['aggregate_sha256'],'patient_level_output_emitted':False}


def run(stage,attempt,fit_attempt,state):
    import run_bran_v5_cbc_uncertainty as parent
    import run_bran_context_preservation_v5 as v5
    from bran_multisource_binding_v3 import load_bound_sources
    import bran_source_pattern_evaluation_v6 as evaluation
    import bran_source_pattern_metrics_v6 as gates
    from audit_bran_source_pattern_v6 import validate_native
    out=paths(stage,attempt);require(not out.exists() and not out.is_symlink())
    out.mkdir();state['out']=out;progress(out,state,'authentication')
    fp,fa,components,fit_receipt=train.authenticate('fit',fit_attempt)
    cp,controls,control_receipt=train.archived('fit')
    baseline,vp,_,parents=parent.small_authentication()
    progress(out,state,'source_loading');sources=load_bound_sources()
    require(fp['source_binding']==cp['source_binding']==vp['source_binding']==sources.receipt())
    protocol={'schema':'bran-robust-clinical-r7-evaluation-protocol','stage':stage,
        'status':'frozen_before_evaluation','parameters':train.EVALUATION,'code_sha256':train.code_hashes(),
        'source_binding':sources.receipt(),'fit_receipt':fit_receipt,'control_receipt':control_receipt,
        'baseline_record_sha256':sha(parent.BASELINE),'role_adapter':train.ROLE_ADAPTER,
        'patient_level_output_emitted':False,'protected_sources_used':False,'automatic_promotion':False}
    write_json(out/'protocol.json',protocol);pin=sha(out/'protocol.json')
    _,private=train.paths('fit',fit_attempt);_,parent_private=v5.paths('fit',2)
    _,control_private=train.control.paths('fit',1)
    def provider(role,fold):
        require(role in train.ROLE_ADAPTER and type(fold) is int and fold in range(5))
        if role=='V5':
            item=parents[('M',fold)]
            return v5.oldfit.load_checkpoint(parent_private/f'fold{fold}_M.pt',item['checkpoint_sha256'],item['binding'])
        if role=='C':
            item=controls[('C',fold)]
            return train.control.load_checkpoint(control_private/f'fold{fold}_C.pt',item['checkpoint_sha256'],item['binding'])
        item=components[('R',fold)]
        return train.load_checkpoint(private/f'fold{fold}_R.pt',item['checkpoint_sha256'],item['binding'])
    decision=None
    if stage=='native':
        value=evaluation.evaluate(sources.paired,provider,lambda p,f,r:progress(out,state,p,f,r))
        checks=validate_native(value)
        decision=gates.decide(value['stress_primary'],value['screening']['both'],checks,True)
    else:
        import bran_multisource_reference_predictions_v2 as references
        import bran_multisource_comparison_v3 as comparison
        import run_bran_multisource_evaluation_v3 as oldeval
        progress(out,state,'historical_gate')
        reference=references.build(sources.paired,sources.context)
        value=comparison.evaluate(sources.paired,reference,lambda role,fold:provider({'C':'C','M':'S'}[role],fold))
        oldeval.validate_result(value)
    progress(out,state,'post_authentication')
    require(parent.small_authentication()[0]==baseline and train.authenticate('fit',fit_attempt)[3]==fit_receipt
        and train.archived('fit')[2]==control_receipt and load_bound_sources().receipt()==protocol['source_binding']
        and train.code_hashes()==protocol['code_sha256'] and sha(out/'protocol.json')==pin)
    aggregate={'schema':'bran-robust-clinical-r7-evaluation-terminal','status':'completed','stage':stage,
        'role_adapter':train.ROLE_ADAPTER,'result':value,'research_lead_decision':decision,
        'historical_role_adapter':{'C':'archived_V6_control_C','M':'R7_robust_input_R'} if stage=='historical' else None,
        'historical_gate_changed':False,'candidate_promoted':False,'scientific_goal_achieved':False,
        'patient_level_output_emitted':False}
    write_json(out/'aggregate.json',aggregate);progress(out,state,'completed')
    require(not (out/'failure.json').exists())
    write_json(out/'completed.json',{'status':'authenticated_completed','protocol_sha256':pin,
        'aggregate_sha256':sha(out/'aggregate.json'),'patient_level_output_emitted':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('native','historical'),required=True)
    p.add_argument('--attempt',type=int,required=True);p.add_argument('--fit-attempt',type=int,required=True)
    args=p.parse_args();state={'out':None,'phase':'authentication'};ok=False
    with quiet():
        try:
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                import torch
                torch.set_num_threads(2);run(args.stage,args.attempt,args.fit_attempt,state);ok=True
        except Exception as exc:
            out=state['out']
            if out is not None and not (out/'completed.json').exists():
                write_json(out/'failure.json',{'status':'technical_failure','phase':state['phase'],
                    'safe_code_site':train.control.safe_site(exc),'patient_level_output_emitted':False,
                    'candidate_promoted':False})
    print(json.dumps({'status':'completed' if ok else 'not_completed','stage':args.stage,
                     'phase':state['phase'],'patient_level_output_emitted':False}))
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
