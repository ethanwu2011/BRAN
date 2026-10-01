"""Q1 frozen pilot/fit/evaluate lifecycle. Private checkpoints, aggregate-only logs."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import run_bran_v5_cbc_uncertainty as source
import run_bran_context_preservation_v5 as v5
import bran_v5_quantile_training as training
import bran_v5_quantile_evaluation as evaluation
from bran_v5_quantile_cbc import QuantileCBC
from bran_multisource_batches_v2 import tensor, transform_hash
from bran_multisource_outcomes_v2 import original_age, subset_age
from bran_multisource_profiles_v3 import _validate_provider, _unchanged
from bran_multisource_inference_v2 import infer_native, completion_predictions
from bran_clinical_semantics_v1 import CBC_FIELDS
from run_bran_v5_residual_cbc import teacher_state_hash
from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json

ROOT=Path(__file__).resolve().parent
PLAN='BRAN_V5_QUANTILE_COMPLETION_PLAN_2026-09-20.md'
CODE=tuple(sorted(set(source.CODE)|{
    PLAN,'bran_v5_quantile_cbc.py','bran_v5_quantile_training.py','bran_v5_quantile_evaluation.py',
    'test_bran_v5_quantile_cbc.py','test_bran_v5_quantile_training.py','test_bran_v5_quantile_evaluation.py',
    'run_bran_v5_quantile.py','test_run_bran_v5_quantile.py','run_bran_v5_quantile_attempt1.sh',
    'bran_v5_residual_training.py','bran_v5_residual_cbc.py','run_bran_v5_residual_cbc.py',
    'run_bran_v5_residual_pilot.py','run_bran_v5_cbc_uncertainty.py'}))
PARAMETERS={'role':'M','seed_base':98301,'architecture':'192-128-GELU-27-noncrossing-quantiles',
    'pilot_updates':100,'updates_per_fold':1500,'batch_size':96,'learning_rate':1e-4,'weight_decay':1e-4,
    'gradient_clip':5,'quantiles':[.05,.5,.95],'patterns':list(training.PATTERNS),
    'loss':'equal_supported_field_equal_quantile_pinball','fixed_native_center':True,
    'encoder_frozen':True,'draws':1000,'bootstrap_seed':98351,'minimum_valid':900,
    'calibration':'separate_existing_within_fold_half_rank_ceil((n+1)*.9);n>=20;nonnegative_CQR',
    'primary':'equal_18_whole_context_field_fold_iqr_interval_score_Q1_minus_native',
    'coverage_guard':.85,'automatic_promotion':False,'protected_external_used':False}
PHASES=('source_authentication','initial_replay','fit','checkpoint_replay','post_authentication',
        'inference','aggregate_bootstrap','aggregate_replay')

def require(ok):
    if not ok: raise ValueError('q1_runner_contract_failed')

def paths(stage,attempt):
    require(stage in ('pilot','fit','evaluate') and type(attempt)==int and 1<=attempt<=99)
    name=f'BRAN_V5_QUANTILE_{stage.upper()}_ATTEMPT{attempt}'
    return ROOT/name,ROOT/'private_artifacts'/name.lower()

def code_hashes(): return {n:sha(ROOT/n) for n in CODE}

def protocol(stage,evidence,dependency=None):
    require(stage in ('pilot','fit','evaluate'))
    return {'schema':'bran-v5-quantile-protocol-v1','stage':stage,'status':'frozen_before_execution',
            'parameters':PARAMETERS,'source_binding':evidence,'code_sha256':code_hashes(),
            'dependency':dependency,'patient_level_output_emitted':False}

def receipt(value,updates):
    d=asdict(value) if hasattr(value,'__dataclass_fields__') else dict(value)
    require(set(d)=={'attempted_updates','optimizer_updates','empty_updates','elapsed_seconds',
        'encoder_unchanged','baseline_head_unchanged','quantile_parameters_changed','sampling_schedule_sha256'})
    require(d['attempted_updates']==updates and type(d['optimizer_updates'])==int
            and 0<d['optimizer_updates']<=updates and type(d['empty_updates'])==int
            and d['empty_updates']==updates-d['optimizer_updates'])
    require(type(d['elapsed_seconds']) in (int,float) and np.isfinite(d['elapsed_seconds']) and d['elapsed_seconds']>=0)
    require(all(d[k] is True for k in ('encoder_unchanged','baseline_head_unchanged','quantile_parameters_changed')))
    require(type(d['sampling_schedule_sha256'])==str and len(d['sampling_schedule_sha256'])==64
            and set(d['sampling_schedule_sha256'])<=set('0123456789abcdef'))
    return d

def binding(fold,item,transform,pin,updates):
    require(fold in range(5) and item['binding']['fold']==fold and item['binding']['role']=='M'
            and transform.heldout_fold==fold and updates in (100,1500))
    return {'fold':fold,'role':'M','teacher_checkpoint_sha256':item['checkpoint_sha256'],
            'teacher_binding':item['binding'],'transform_sha256':transform_hash(transform),
            'protocol_sha256':pin,'seed':98301+fold,'updates':updates,'architecture':PARAMETERS['architecture']}

def bound(head,teacher):
    require(isinstance(head,QuantileCBC) and not head.training)
    require(all(torch.equal(v,head.baseline.state_dict()[k]) for k,v in teacher.cbc_joint_head.state_dict().items()))
    require(all(torch.isfinite(v).all() for v in head.state_dict().values()))

def save_head(path,head,teacher,bind):
    bound(head,teacher); require(not path.exists() and not path.is_symlink())
    payload={'schema':'bran-v5-quantile-checkpoint-v1','binding':bind,'teacher_state_sha256':teacher_state_hash(teacher),
             'quantile_state_dict':{k:v.detach().cpu().clone() for k,v in head.quantiles.state_dict().items()}}
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(),0o600); torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
    return sha(path)

def load_head(path,pin,teacher,bind):
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink==1
            and path.stat().st_mode&0o777==0o600 and sha(path)==pin)
    p=torch.load(path,map_location='cpu',weights_only=True)
    require(type(p)==dict and set(p)=={'schema','binding','teacher_state_sha256','quantile_state_dict'}
            and p['schema']=='bran-v5-quantile-checkpoint-v1' and p['binding']==bind
            and p['teacher_state_sha256']==teacher_state_hash(teacher))
    require(bind['role']=='M' and type(bind['fold'])==int and bind['fold'] in range(5)
            and bind['seed']==98301+bind['fold'] and bind['updates'] in (100,1500)
            and bind['architecture']==PARAMETERS['architecture'])
    head=QuantileCBC(teacher.cbc_joint_head,bind['seed']); head.quantiles.load_state_dict(p['quantile_state_dict'],strict=True)
    head.eval()
    for parameter in head.parameters(): parameter.requires_grad_(False)
    bound(head,teacher); require(sha(path)==pin); return head

def progress(out,state,phase,fold=None):
    require(phase in PHASES and (fold is None or type(fold)==int and fold in range(5)))
    state['phase']=phase; item={'phase':phase,'pid':os.getpid(),'patient_level_output_emitted':False}
    if fold is not None:item['fold']=fold
    write_json(out/'progress.next.json',item); os.replace(out/'progress.next.json',out/'progress.json')

def create(stage,attempt,state):
    out,private=paths(stage,attempt)
    require(not out.exists() and not out.is_symlink() and not private.exists() and not private.is_symlink())
    out.mkdir(); state['out']=out
    if stage!='evaluate':
        require(private.parent.is_dir() and not private.parent.is_symlink()); private.mkdir(mode=0o700)
    return out,private

def finish(out,result,pin,state):
    write_json(out/'aggregate.json',result)
    manifest={'protocol_sha256':pin,'aggregate_sha256':sha(out/'aggregate.json'),'patient_level_output_emitted':False}
    write_json(out/'manifest.json',manifest)
    write_json(out/'completed.json',{'status':'authenticated_completed',**manifest,
        'manifest_sha256':sha(out/'manifest.json'),'candidate_promoted':False,'scientific_goal_achieved':False})
    state['out']=None

def authenticate(stage,attempt):
    out,private=paths(stage,attempt)
    require((out/'completed.json').is_file() and not (out/'failure.json').exists())
    p=json.loads((out/'protocol.json').read_text()); a=json.loads((out/'aggregate.json').read_text())
    m=json.loads((out/'manifest.json').read_text()); t=json.loads((out/'completed.json').read_text())
    require(p==protocol(stage,p['source_binding'],p['dependency']) and a['status']=='completed'
            and t['status']=='authenticated_completed')
    require(a['protocol_sha256']==m['protocol_sha256']==t['protocol_sha256']==sha(out/'protocol.json'))
    require(m['aggregate_sha256']==t['aggregate_sha256']==sha(out/'aggregate.json')
            and t['manifest_sha256']==sha(out/'manifest.json') and not a['patient_level_output_emitted'])
    pins={'protocol_sha256':sha(out/'protocol.json'),'aggregate_sha256':sha(out/'aggregate.json')}
    components={}
    if stage!='evaluate':
        require(private.is_dir() and not private.is_symlink() and private.stat().st_mode&0o777==0o700)
        folds=[0] if stage=='pilot' else list(range(5)); updates=100 if stage=='pilot' else 1500
        require(set(a['component_sha256'])=={f'fold{f}.json' for f in folds}
                and a['updates_total']==updates*len(folds) and not a['heldout_inference_performed']
                and a['screening_unchanged'] and a['checkpoint_replay_exact'])
        for fold in folds:
            name=f'fold{fold}.json'; require(sha(out/name)==a['component_sha256'][name])
            item=json.loads((out/name).read_text()); receipt(item['training'],updates)
            require(item['fold']==item['binding']['fold']==fold and item['binding']['protocol_sha256']==pins['protocol_sha256']
                    and item['checkpoint_replay_exact'] and item['screening_unchanged'] and not item['heldout_inference_performed'])
            require(sha(private/f'fold{fold}_quantile.pt')==item['checkpoint_sha256']); components[fold]=item
    else: evaluation.validate_result(a['result'])
    return p,a,components,pins

def fit(stage,attempt,dependency_attempt,state):
    require(stage in ('pilot','fit')); started=time.monotonic(); torch.set_num_threads(1)
    out,private=create(stage,attempt,state); progress(out,state,'source_authentication')
    dep=None if stage=='pilot' else authenticate('pilot',dependency_attempt)[3]
    paired,_,components,evidence=source.source_context()
    if stage=='fit': require(authenticate('pilot',dependency_attempt)[0]['source_binding']==evidence)
    frozen=protocol(stage,evidence,dep); write_json(out/'protocol.json',frozen); pin=sha(out/'protocol.json')
    slots=tuple(paired.names.index(f) for f in CBC_FIELDS); age=original_age(paired)
    _,teacher_private=v5.paths('fit',2); component_pins={}; total_seconds=0.
    folds=[0] if stage=='pilot' else list(range(5)); updates=100 if stage=='pilot' else 1500
    for fold in folds:
        progress(out,state,'fit',fold); item=components[('M',fold)]
        teacher,transform=v5.oldfit.load_checkpoint(teacher_private/f'fold{fold}_M.pt',item['checkpoint_sha256'],item['binding'])
        before,th,grads=_validate_provider(teacher,transform,fold,slots,paired.transforms[fold])
        rows=np.flatnonzero(paired.folds!=fold)
        require(len(rows)>=100 and set(np.unique(paired.folds[rows]))==set(range(5))-{fold})
        c,cm=transform.clinical(paired.c,paired.cm); r,rm=transform.retinal(paired.r,paired.rm)
        def args(ii):return (tensor(c[ii]),tensor(cm[ii],torch.bool),tensor(r[ii]),tensor(rm[ii],torch.bool),
                            subset_age(age,ii),transform.age_mean,transform.age_scale)
        probe=args(rows[:96]); screen=infer_native(teacher,*probe).screening_probability
        initial=QuantileCBC(teacher.cbc_joint_head,98301+fold).eval()
        for pattern in training.PATTERNS:
            q=training.predict_quantiles(teacher,initial,*probe,pattern,slots)
            native=completion_predictions(teacher,*probe,pattern,slots)
            require(np.array_equal(q.median.numpy(),native.cbc_standardized.numpy(),equal_nan=True)
                    and torch.equal(q.scoringmask,native.scoring_target_mask))
        head,rec=training.train_quantile(teacher,*args(rows),slots,tensor(paired.folds[rows],torch.int64),fold,
                                       updates=updates,batch_size=96)
        safe=receipt(rec,updates); total_seconds+=safe['elapsed_seconds']
        require(_unchanged(before,th,grads,teacher,transform)
                and np.array_equal(screen.numpy(),infer_native(teacher,*probe).screening_probability.numpy(),equal_nan=True))
        bind=binding(fold,item,transform,pin,updates); checkpoint=private/f'fold{fold}_quantile.pt'
        checkpoint_pin=save_head(checkpoint,head,teacher,bind); progress(out,state,'checkpoint_replay',fold)
        restored=load_head(checkpoint,checkpoint_pin,teacher,bind)
        for pattern in training.PATTERNS:
            one=training.predict_quantiles(teacher,head,*probe,pattern,slots)
            two=training.predict_quantiles(teacher,restored,*probe,pattern,slots)
            require(all(np.array_equal(getattr(one,k).numpy(),getattr(two,k).numpy(),equal_nan=True)
                        for k in ('q05','median','q95','scoringmask','abstained')))
        value={'fold':fold,'binding':bind,'checkpoint_sha256':checkpoint_pin,'training':safe,
            'checkpoint_replay_exact':True,'screening_unchanged':True,'heldout_inference_performed':False,
            'patient_level_output_emitted':False}
        name=f'fold{fold}.json'; write_json(out/name,value); component_pins[name]=sha(out/name)
    progress(out,state,'post_authentication')
    require(source.source_context()[3]==evidence and protocol(stage,evidence,dep)==frozen and sha(out/'protocol.json')==pin)
    aggregate={'schema':'bran-v5-quantile-fit-v1','stage':stage,'status':'completed','protocol_sha256':pin,
        'component_sha256':component_pins,'updates_total':updates*len(folds),'checkpoint_replay_exact':True,
        'screening_unchanged':True,'heldout_inference_performed':False,'patient_level_output_emitted':False,
        'candidate_promoted':False,'elapsed_seconds':time.monotonic()-started}
    if stage=='pilot':aggregate['projected_fit_compute_seconds']=total_seconds*75
    finish(out,aggregate,pin,state)

def evaluate(attempt,fit_attempt,state):
    started=time.monotonic(); out,_=create('evaluate',attempt,state); progress(out,state,'source_authentication')
    fp,fa,heads,dep=authenticate('fit',fit_attempt)
    paired,roles,components,evidence=source.source_context(); require(fp['source_binding']==evidence)
    frozen=protocol('evaluate',evidence,dep); write_json(out/'protocol.json',frozen); pin=sha(out/'protocol.json')
    _,teacher_private=v5.paths('fit',2); _,private=paths('fit',fit_attempt)
    def provider(fold):
        item=components[('M',fold)]
        teacher,transform=v5.oldfit.load_checkpoint(teacher_private/f'fold{fold}_M.pt',item['checkpoint_sha256'],item['binding'])
        bind=binding(fold,item,transform,fa['protocol_sha256'],1500); require(heads[fold]['binding']==bind)
        head=load_head(private/f'fold{fold}_quantile.pt',heads[fold]['checkpoint_sha256'],teacher,bind)
        return teacher,transform,head
    result=evaluation.evaluate(paired,roles,provider,lambda p,f:progress(out,state,p,f))
    progress(out,state,'post_authentication')
    require(source.source_context()[3]==evidence and protocol('evaluate',evidence,dep)==frozen
            and authenticate('fit',fit_attempt)[:2]==(fp,fa) and sha(out/'protocol.json')==pin)
    aggregate={'schema':'bran-v5-quantile-evaluation-aggregate-v1','status':'completed','protocol_sha256':pin,
               'result':result,'patient_level_output_emitted':False,'candidate_promoted':False,
               'full_aggregate_replayed_exactly':True,'checkpoint_replay_exact':True,
               'elapsed_seconds':time.monotonic()-started}
    finish(out,aggregate,pin,state)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',required=True,choices=('pilot','fit','evaluate'))
    parser.add_argument('--attempt',required=True,type=int); parser.add_argument('--dependency-attempt',default=1,type=int)
    args=parser.parse_args(); state={'out':None,'phase':'source_authentication'}; ok=False
    with quiet():
        try:
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if args.stage=='evaluate':evaluate(args.attempt,args.dependency_attempt,state)
                else:fit(args.stage,args.attempt,args.dependency_attempt,state)
                ok=True
        except Exception:
            if state['out'] is not None and not (state['out']/'completed.json').exists():
                write_json(state['out']/'failure.json',{'status':'technical_failure','phase':state['phase'],
                    'patient_level_output_emitted':False,'candidate_promoted':False})
    print(json.dumps({'status':'completed' if ok else 'not_completed','stage':args.stage,'phase':state['phase'],
                      'patient_level_output_emitted':False,'candidate_promoted':False}))
    return 0 if ok else 1

if __name__=='__main__':raise SystemExit(main())
