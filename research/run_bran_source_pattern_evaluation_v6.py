"""Authenticated local V6 native evaluation and unchanged historical gate.

Two stages are separate: native decision inputs first; historical gate second.
No automatic model promotion, external evaluation, or file overwrite.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from run_bran_multisource_retinal_features_v2 import LOCK,quiet,sha,write_json
import run_bran_source_pattern_v6 as train

ROOT=Path(__file__).resolve().parent
PHASES=('authentication','source_loading','inference','aggregate_bootstrap','historical_gate',
        'post_authentication','completed')


def require(ok):
    if not ok:raise ValueError('source_pattern_evaluation_v6_runner_failed') from None


def code_hashes():
    names={'bran_source_pattern_metrics_v6.py','bran_source_pattern_evaluation_v6.py',
        'run_bran_source_pattern_evaluation_v6.py','test_bran_source_pattern_metrics_v6.py',
        'test_bran_source_pattern_evaluation_v6.py'}
    return {**train.code_hashes(),**{n:sha(ROOT/n) for n in sorted(names)}}


def progress(out,state,phase,fold=None,role=None):
    require(phase in PHASES and (fold is None or type(fold) is int and fold in range(5)))
    require(role is None or role in ('V5','C','S'))
    state['phase']=phase
    write_json(out/'progress.next.json',{'phase':phase,'fold':fold,'role':role,'pid':os.getpid(),
                                       'patient_level_output_emitted':False})
    os.replace(out/'progress.next.json',out/'progress.json')


def run(attempt,fit_attempt,stage,state):
    import torch
    import run_bran_v5_cbc_uncertainty as parent
    import run_bran_context_preservation_v5 as v5
    from bran_multisource_binding_v3 import load_bound_sources
    import bran_source_pattern_evaluation_v6 as evaluation
    import bran_source_pattern_metrics_v6 as gates
    require(type(attempt) is int and 1<=attempt<=99 and stage in ('native','historical'))
    out=ROOT/f'BRAN_SOURCE_PATTERN_V6_{stage.upper()}_EVALUATE_ATTEMPT{attempt}'
    require(not out.exists() and not out.is_symlink())
    out.mkdir();state['out']=out;progress(out,state,'authentication')
    fp,fa,components,fit_receipt=train.authenticate('fit',fit_attempt)
    baseline,vp,_,parents=parent.small_authentication()
    progress(out,state,'source_loading')
    sources=load_bound_sources()
    require(fp['source_binding']==vp['source_binding']==sources.receipt())
    protocol={'schema':'bran-source-pattern-v6-evaluation-protocol','stage':stage,
        'status':'frozen_before_evaluation','parameters':train.EVALUATION,'code_sha256':code_hashes(),
        'source_binding':sources.receipt(),'fit_receipt':fit_receipt,
        'baseline_record_sha256':sha(parent.BASELINE),'patient_level_output_emitted':False,
        'protected_sources_used':False,'automatic_promotion':False}
    write_json(out/'protocol.json',protocol);pin=sha(out/'protocol.json')
    _,private=train.paths('fit',fit_attempt);_,parent_private=v5.paths('fit',2)
    def provider(role,fold):
        require(role in ('V5','C','S') and type(fold) is int and fold in range(5))
        if role=='V5':
            item=parents[('M',fold)]
            return v5.oldfit.load_checkpoint(parent_private/f'fold{fold}_M.pt',item['checkpoint_sha256'],item['binding'])
        item=components[(role,fold)]
        return train.load_checkpoint(private/f'fold{fold}_{role}.pt',item['checkpoint_sha256'],item['binding'])
    if stage=='native':
        value=evaluation.evaluate(sources.paired,provider,lambda p,f,r:progress(out,state,p,f,r))
        evaluation.validate_result(value)
    else:
        import bran_multisource_reference_predictions_v2 as references
        import bran_multisource_comparison_v3 as comparison
        import run_bran_multisource_evaluation_v3 as oldeval
        progress(out,state,'historical_gate')
        reference=references.build(sources.paired,sources.context)
        # Explicit adapter to historical C/M metric roles, not a redefinition.
        value=comparison.evaluate(sources.paired,reference,
            lambda role,fold:provider({'C':'C','M':'S'}[role],fold))
        oldeval.validate_result(value)
    progress(out,state,'post_authentication')
    require(parent.small_authentication()[0]==baseline and train.authenticate('fit',fit_attempt)[3]==fit_receipt)
    require(load_bound_sources().receipt()==protocol['source_binding'] and code_hashes()==protocol['code_sha256']
            and sha(out/'protocol.json')==pin)
    decision=None
    if stage=='native':
        checks=gates.protected_completion_checks(value['completion'],value['historical_low_hb'])
        decision=gates.decide(value['stress_primary'],value['screening']['both'],checks,True)
    result={'schema':'bran-source-pattern-v6-evaluation-terminal','status':'completed','stage':stage,
        'result':value,'research_lead_decision':decision,
        'historical_role_adapter':{'C':'V6_control','M':'V6_source_pattern_candidate'} if stage=='historical' else None,
        'historical_gate_changed':False,'candidate_promoted':False,'scientific_goal_achieved':False,
        'patient_level_output_emitted':False}
    write_json(out/'aggregate.json',result)
    progress(out,state,'completed')
    require(not (out/'failure.json').exists())
    write_json(out/'completed.json',{'status':'authenticated_completed','protocol_sha256':pin,
        'aggregate_sha256':sha(out/'aggregate.json'),'patient_level_output_emitted':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,required=True)
    p.add_argument('--fit-attempt',type=int,required=True)
    p.add_argument('--stage',choices=('native','historical'),required=True)
    args=p.parse_args();state={'out':None,'phase':'authentication'};ok=False
    with quiet():
        try:
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                import torch
                torch.set_num_threads(2)
                run(args.attempt,args.fit_attempt,args.stage,state);ok=True
        except Exception as exc:
            out=state['out']
            if out is not None and not (out/'completed.json').exists():
                write_json(out/'failure.json',{'status':'technical_failure','phase':state['phase'],
                    'safe_code_site':train.safe_site(exc),'patient_level_output_emitted':False,
                    'candidate_promoted':False})
    print(json.dumps({'status':'completed' if ok else 'not_completed','stage':args.stage,
                      'phase':state['phase'],'patient_level_output_emitted':False}))
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
