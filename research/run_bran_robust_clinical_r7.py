"""Exclusive, FD-quiet R7 pilot/fit with authenticated archived matched control."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json
import run_bran_source_pattern_v6 as control

ROOT=Path(__file__).resolve().parent
PLAN='BRAN_ROBUST_CLINICAL_R7_DESIGN.md'
PARAMETERS={**control.PARAMETERS, 'roles':{'C':'authenticated_archived_ordinary_continuation',
    'R':'robust_clinical_scaled_asinh3'}, 'bridge_screening_coefficient':0.,
    'bridge_cbc_coefficient':0., 'input_map':'3_asinh_z_div3_encoder_only',
    'teacher_and_state_preservation_unchanged':True,'control_attempt':1,
    'control_retrained':False,'pilot_control_reproduced':True}
ROLE_ADAPTER={'V5':'retained_V5_M','C':'archived_V6_control_C','S':'R7_robust_input_R'}
EVALUATION={**control.EVALUATION,'role_adapter':ROLE_ADAPTER}
TRACES=control.TRACES
PHASES=control.PHASES
ERROR='robust_clinical_r7_runner_failed'


def require(ok):
    if not ok:raise ValueError(ERROR) from None


def code_hashes():
    names=(PLAN,'bran_robust_clinical_r7.py','run_bran_robust_clinical_r7.py',
        'run_bran_robust_clinical_evaluation_r7.py','test_bran_robust_clinical_r7.py',
        'test_bran_robust_clinical_lifecycle_r7.py','run_bran_robust_clinical_r7_attempt1.sh',
        'bran_source_pattern_metrics_v6.py','bran_source_pattern_evaluation_v6.py',
        'audit_bran_source_pattern_v6.py')
    return {**control.code_hashes(),**{n:sha(ROOT/n) for n in names}}


def paths(stage,attempt):
    require(stage in ('pilot','fit') and type(attempt) is int and 1<=attempt<=99)
    return ROOT/f'BRAN_ROBUST_CLINICAL_R7_{stage.upper()}_ATTEMPT{attempt}',ROOT/'private_artifacts'/f'bran_robust_clinical_r7_{stage}_attempt{attempt}'


def progress(out,state,phase,fold=None,role=None,updates=0):
    require(phase in PHASES and (fold is None or type(fold) is int and fold in range(5)))
    require(role in (None,'C','R') and type(updates) is int and 0<=updates<=3000)
    state['phase']=phase
    write_json(out/'progress.next.json',{'phase':phase,'fold':fold,'role':role,
        'updates_completed':updates,'pid':os.getpid(),'patient_level_output_emitted':False})
    os.replace(out/'progress.next.json',out/'progress.json')


def archived(stage):
    p,a,items,pins=control.authenticate(stage,1)
    require(p['parameters']==control.PARAMETERS and p['parameters']['updates_per_arm_fold']==3000
        and p['parameters']['learning_rate']==5e-5 and p['parameters']['paired_batch']==96
        and p['parameters']['source_batch']==128 and p['parameters']['all_other_V5_losses_unchanged'] is True)
    return p,items,pins


def matched_traces(current,reference):
    values={k:current[k] for k in TRACES}
    require(all(type(v) is str and len(v)==64 and set(v)<=set('0123456789abcdef') for v in values.values()))
    require(values=={k:reference[k] for k in TRACES})
    return values


def load_checkpoint(path,pin,binding):
    import torch
    from bran_robust_clinical_r7 import BRANRobustClinicalR7
    from bran_multisource_protocol_v2 import digest
    from run_bran_multisource_fit_v3 import _restore_transform
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink==1
        and path.stat().st_mode&0o777==0o600 and sha(path)==pin)
    p=torch.load(path,map_location='cpu',weights_only=True)
    require(set(p)=={'schema','config','binding','updates','state_dict','optimizer_state','input_transform'}
        and p['schema']=='bran-multisource-continuation-checkpoint-v3' and p['binding']==binding
        and p['updates']==3000 and binding['training_recipe']=='robust_clinical_r7'
        and binding['role']=='R' and type(binding['fold']) is int and binding['fold'] in range(5))
    model=BRANRobustClinicalR7.from_config(p['config'])
    require(digest(model.export_config())==binding['model_config_sha256'])
    require(all(not v.is_floating_point() or bool(torch.isfinite(v).all()) for v in p['state_dict'].values()))
    model.load_state_dict(p['state_dict'],strict=True)
    transform=_restore_transform(p['input_transform'],binding['transform_sha256'])
    require(transform.heldout_fold==binding['fold'] and sha(path)==pin)
    model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model,transform


def authenticate(stage,attempt):
    out,private=paths(stage,attempt)
    require(out.is_dir() and not out.is_symlink() and not (out/'failure.json').exists())
    from audit_bran_source_pattern_v6 import read
    p,a,t=(read(out/n) for n in ('protocol.json','aggregate.json','completed.json'))
    require(t['status']=='authenticated_completed' and t['protocol_sha256']==sha(out/'protocol.json')
        and t['aggregate_sha256']==sha(out/'aggregate.json') and p['stage']==stage
        and p['parameters']==PARAMETERS and p['evaluation']==EVALUATION and p['code_sha256']==code_hashes()
        and a['status']=='completed' and a['patient_level_output_emitted'] is False
        and a['candidate_promoted'] is False and a['scientific_goal_achieved'] is False)
    ap,prior,pins=archived(stage)
    require(p['control_receipt']==pins and p['source_binding']==ap['source_binding']
        and p['parent_checkpoints']==ap['parent_checkpoints'])
    folds=[0] if stage=='pilot' else range(5)
    expected={f'fold{f}_{r}.json' for f in folds for r in (('C','R') if stage=='pilot' else ('R',))}
    require(set(a['component_sha256'])==expected)
    items={}
    for name in sorted(expected):
        require(sha(out/name)==a['component_sha256'][name]);item=read(out/name)
        role,fold=item['role'],item['fold'];budget=100 if stage=='pilot' else 3000
        require(name==f'fold{fold}_{role}.json' and item['updates_completed']==budget
            and item['patient_level_output_emitted'] is False)
        control.check_counters(item['algorithm_update_counters'],'C',budget)
        matched_traces(item,prior[('C',fold)])
        if stage=='fit':
            b=item['binding']
            require(b['protocol_sha256']==t['protocol_sha256'] and b['fold']==fold and b['role']=='R'
                and b['outer_fold_sha256']==p['source_binding']['outer_fold_sha256']
                and b['inner_fold_sha256']==p['source_binding']['inner_fold_sha256'][fold]
                and b['initial_checkpoint_sha256']==p['parent_checkpoints'][fold]['checkpoint_sha256']
                and b['archived_control_checkpoint_sha256']==prior[('C',fold)]['checkpoint_sha256']
                and item['checkpoint_reload_exact'] is True)
            load_checkpoint(private/f'fold{fold}_R.pt',item['checkpoint_sha256'],b)
        items[(role,fold)]=item
    return p,a,items,{'protocol_sha256':t['protocol_sha256'],'aggregate_sha256':t['aggregate_sha256'],
                      'terminal_sha256':sha(out/'completed.json')}


def run(stage,attempt,state):
    import torch
    import run_bran_v5_cbc_uncertainty as parent
    from bran_robust_clinical_r7 import BRANRobustClinicalR7
    from bran_multisource_binding_v3 import load_bound_sources
    from bran_multisource_batches_v2 import transform_hash
    from bran_multisource_protocol_v2 import digest
    from bran_multisource_fit_v2 import _exposure
    from bran_multisource_fit_v6 import fit_one_v6
    from run_bran_multisource_fit_v3 import save_checkpoint
    out,private=paths(stage,attempt)
    require(not out.exists() and not out.is_symlink() and not private.exists() and not private.is_symlink())
    out.mkdir();state['out']=out;progress(out,state,'authentication')
    dependency=None if stage=='pilot' else authenticate('pilot',attempt)
    archived_protocol,archived_items,control_pin=archived(stage)
    baseline,vp,_,parents=parent.small_authentication()
    progress(out,state,'source_loading');sources=load_bound_sources()
    require(sources.receipt()==vp['source_binding']==archived_protocol['source_binding'])
    if dependency is not None:require(dependency[0]['source_binding']==sources.receipt())
    protocol={'schema':'bran-robust-clinical-r7-protocol','stage':stage,'status':'frozen_before_training',
        'parameters':PARAMETERS,'evaluation':EVALUATION,'code_sha256':code_hashes(),
        'source_binding':sources.receipt(),'baseline_record_sha256':sha(parent.BASELINE),
        'parent_checkpoints':baseline['fold_checkpoints'],'control_receipt':control_pin,
        'pilot_dependency':None if dependency is None else dependency[3],
        'patient_level_output_emitted':False,'candidate_promoted':False}
    require(protocol['parent_checkpoints']==archived_protocol['parent_checkpoints'])
    write_json(out/'protocol.json',protocol);pin=sha(out/'protocol.json')
    budget,folds=(100,[0]) if stage=='pilot' else (3000,range(5))
    if stage=='fit':
        require(private.parent.is_dir() and not private.parent.is_symlink())
        private.mkdir(mode=0o700)
    receipts={};runtime=0.;candidate_runtime=0.
    for fold in folds:
        progress(out,state,'parent_binding',fold);bound=control.bind_v5(sources,parents,fold)
        for role in (('C','R') if stage=='pilot' else ('R',)):
            initial=bound.model if role=='C' else BRANRobustClinicalR7.from_parent(bound.model)
            phase='pilot_training' if stage=='pilot' else 'training'
            progress(out,state,phase,fold,role)
            result=fit_one_v6(initial,bound.teacher,bound.paired_factory,bound.source_factory,
                bound.state_scale,bound.transform.age_mean,bound.transform.age_scale,bound.seed,
                bound.cbc_indices,'C',lambda n:progress(out,state,phase,fold,role,n),updates=budget)
            traces=matched_traces(result,archived_items[('C',fold)])
            control.check_counters(result['algorithm_update_counters'],'C',budget)
            runtime+=result['elapsed_seconds']
            if role=='R':candidate_runtime+=result['elapsed_seconds']
            item={'role':role,'fold':fold,'updates_completed':budget,'runtime_seconds':result['elapsed_seconds'],
                'algorithm_update_counters':result['algorithm_update_counters'],**traces,
                'patient_level_output_emitted':False,'archived_control_streams_matched':True}
            if stage=='fit':
                b={'protocol_sha256':pin,'fold':fold,'role':'R','training_recipe':'robust_clinical_r7',
                    'initial_checkpoint_sha256':parents[('M',fold)]['checkpoint_sha256'],
                    'archived_control_checkpoint_sha256':archived_items[('C',fold)]['checkpoint_sha256'],
                    'transform_sha256':transform_hash(bound.transform),
                    'model_config_sha256':digest(result['model'].export_config()),
                    'clinical_field_order_sha256':digest(sources.paired.names),
                    'outer_fold_sha256':sources.receipt()['outer_fold_sha256'],
                    'inner_fold_sha256':sources.receipt()['inner_fold_sha256'][fold]}
                progress(out,state,'checkpoint_replay',fold,role,budget)
                path=private/f'fold{fold}_R.pt';checkpoint_pin=save_checkpoint(path,result,bound,b)
                restored,_=load_checkpoint(path,checkpoint_pin,b)
                require(all(torch.equal(v,restored.state_dict()[k]) for k,v in result['model'].state_dict().items()))
                item.update(binding=b,checkpoint_sha256=checkpoint_pin,checkpoint_reload_exact=True,
                    exposure=_exposure(result['source_sampler'].private_sampler,result['paired_sampler'].private_sampler,
                                       sources.paired,sources.pools))
                del restored
            name=f'fold{fold}_{role}.json';write_json(out/name,item);receipts[name]=sha(out/name)
            del result,initial
    progress(out,state,'post_authentication')
    require(parent.small_authentication()[0]==baseline and archived(stage)[2]==control_pin
        and load_bound_sources().receipt()==protocol['source_binding'] and code_hashes()==protocol['code_sha256']
        and sha(out/'protocol.json')==pin)
    aggregate={'schema':'bran-robust-clinical-r7-training','status':'completed','stage':stage,
        'component_sha256':receipts,'total_training_loop_seconds':runtime,
        'projected_candidate_fit_seconds':candidate_runtime*150 if stage=='pilot' else None,
        'pilot_models_discarded':stage=='pilot','control_retrained':False,'matched_all_streams':True,
        'patient_level_output_emitted':False,'candidate_promoted':False,'scientific_goal_achieved':False}
    write_json(out/'aggregate.json',aggregate);progress(out,state,'completed')
    require(not (out/'failure.json').exists())
    write_json(out/'completed.json',{'status':'authenticated_completed','protocol_sha256':pin,
        'aggregate_sha256':sha(out/'aggregate.json'),'patient_level_output_emitted':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('pilot','fit'),required=True)
    p.add_argument('--attempt',type=int,required=True);args=p.parse_args()
    state={'out':None,'phase':'authentication'};ok=False
    with quiet():
        try:
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                import torch
                torch.set_num_threads(2);run(args.stage,args.attempt,state);ok=True
        except Exception as exc:
            out=state['out']
            if out is not None and not (out/'completed.json').exists():
                write_json(out/'failure.json',{'status':'technical_failure','phase':state['phase'],
                    'safe_code_site':control.safe_site(exc),'patient_level_output_emitted':False,
                    'completed_components_preserved':True,'candidate_promoted':False})
    print(json.dumps({'status':'completed' if ok else 'not_completed','stage':args.stage,
        'phase':state['phase'],'patient_level_output_emitted':False,'candidate_promoted':False}))
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
