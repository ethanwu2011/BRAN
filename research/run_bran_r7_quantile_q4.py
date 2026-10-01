"""Quiet R7-bound Q4 pilot, attachment fitting and independent evaluation."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import bran_r7_quantile_q4 as kernel
import bran_r7_fixed_state_p1 as p1
import run_bran_r7_head_blocks_i1 as r7source
import run_bran_r7_completion_utility_u2 as r7provider
import run_bran_v5_cbc_uncertainty as calibration
import run_bran_v5_quantile as q1
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_multisource_batches_v2 import tensor
from bran_multisource_inference_v2 import completion_predictions,infer_native
from bran_multisource_outcomes_v2 import original_age,subset_age
from bran_multisource_protocol_v2 import digest
from bran_multisource_calibration_metrics_v2 import _validate_split
from bran_v5_quantile_cbc import QuantileCBC
from bran_v5_quantile_training import PATTERNS
from run_bran_v5_residual_cbc import teacher_state_hash
from run_bran_multisource_retinal_features_v2 import LOCK,quiet,sha,write_json

ROOT=Path(__file__).resolve().parent
DESIGN=ROOT/'BRAN_R7_QUANTILE_Q4_DESIGN_2026-09-28.md'
OWN_CODE=('run_bran_r7_quantile_q4.py','bran_r7_quantile_q4.py',
    'test_bran_r7_quantile_q4.py','test_bran_r7_quantile_q4_runner.py',
    'run_bran_r7_quantile_q4_attempt1.sh','BRAN_R7_QUANTILE_Q4_DESIGN_2026-09-28.md')
PARAMETERS={**q1.PARAMETERS,'role':'R','encoder_version':'R7',
    'primary':'equal_18_whole_context_field_fold_iqr_interval_score_Q4_minus_R7_native',
    'recipe':'unchanged_Q1_not_V5_checkpoint_or_calibration_weights'}
PHASES=('authentication','source_loading','initial_replay','fit','checkpoint_replay',
        'post_authentication','inference','aggregate_bootstrap','aggregate_replay',
        'completed','independent_replay')
ERROR='r7_quantile_q4_contract_failed'


def require(ok):
    if not ok:raise ValueError(ERROR) from None


def read(path):return json.loads(path.read_text())


def canonical(value):return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def paths(stage):
    require(stage in ('pilot','fit','evaluate'))
    name=f'BRAN_R7_QUANTILE_Q4_{stage.upper()}_ATTEMPT1'
    return ROOT/name,ROOT/'private_artifacts'/name.lower()


def replay_path():return ROOT/'BRAN_R7_QUANTILE_Q4_REPLAY.json'


def code_hashes():
    inherited={**r7source.authenticate()['code_sha256'],**calibration.code_hashes(),**q1.code_hashes()}
    inherited.update({name:sha(ROOT/name) for name in OWN_CODE})
    inherited['run_bran_r7_completion_utility_u2.py']=sha(ROOT/'run_bran_r7_completion_utility_u2.py')
    return inherited


def source_context():
    b=r7source.authenticate()
    paired,roles,_,evidence=calibration.source_context()
    require(b['source_binding']==evidence['source_binding'])
    _validate_split(paired.folds,roles)
    return paired,roles,{'R7_binding':b,'calibration_binding':evidence,
        'fold_sha256':digest(paired.folds.tolist()),'roles_sha256':digest(roles.tolist())}


def protocol(stage,evidence,dependency):
    return canonical({'schema':'bran-r7-quantile-q4-protocol-v1','stage':stage,
        'parameters':PARAMETERS,'binding':evidence,'dependency':dependency,
        'code_sha256':code_hashes(),'design_sha256':sha(DESIGN),
        'patient_level_output_permitted':False,'frozen_before_execution':True})


def load_teacher(evidence,fold):
    return r7provider.provider({'R7_binding':evidence['R7_binding']})(fold)


def head_binding(fold,item,transform,protocol_pin,updates):
    require(item['fold']==fold and item['binding']['fold']==fold and item['binding']['role']=='R')
    require(transform.heldout_fold==fold and updates in (100,1500))
    return {'fold':fold,'role':'R','encoder_version':'R7','seed':98301+fold,'updates':updates,
        'teacher_checkpoint_sha256':item['checkpoint_sha256'],'teacher_binding':item['binding'],
        'transform_sha256':q1.transform_hash(transform),'protocol_sha256':protocol_pin,
        'architecture':PARAMETERS['architecture']}


def save_head(path,head,teacher,binding):
    q1.bound(head,teacher);require(not path.exists() and not path.is_symlink())
    payload={'schema':'bran-r7-q4-attachment-v1','binding':binding,
        'teacher_state_sha256':teacher_state_hash(teacher),
        'quantile_state_dict':{k:v.detach().cpu().clone() for k,v in head.quantiles.state_dict().items()}}
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(),0o600);torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    return sha(path)


def load_head(path,pin,teacher,binding):
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink==1
        and path.stat().st_mode&0o777==0o600 and sha(path)==pin)
    payload=torch.load(path,map_location='cpu',weights_only=True)
    require(set(payload)=={'schema','binding','teacher_state_sha256','quantile_state_dict'}
        and payload['schema']=='bran-r7-q4-attachment-v1' and payload['binding']==binding
        and payload['teacher_state_sha256']==teacher_state_hash(teacher))
    require(binding['role']=='R' and binding['encoder_version']=='R7'
        and type(binding['fold']) is int and binding['fold'] in range(5)
        and binding['seed']==98301+binding['fold'] and binding['updates'] in (100,1500)
        and binding['architecture']==PARAMETERS['architecture'])
    head=QuantileCBC(teacher.cbc_joint_head,binding['seed'])
    head.quantiles.load_state_dict(payload['quantile_state_dict'],strict=True);head.eval()
    for parameter in head.parameters():parameter.requires_grad_(False)
    q1.bound(head,teacher);require(sha(path)==pin)
    return head


def progress(out,phase,fold=None):
    require(phase in PHASES and (fold is None or type(fold) is int and fold in range(5)))
    write_json(out/'progress.next.json',{'phase':phase,'fold':fold,'pid':os.getpid(),
                                       'patient_level_output_emitted':False})
    os.replace(out/'progress.next.json',out/'progress.json')


def terminal(out):
    return {'status':'Q4_completed','protocol_sha256':sha(out/'protocol.json'),
        'aggregate_sha256':sha(out/'aggregate.json'),'patient_level_output_emitted':False,
        'candidate_promoted':False}


def authenticate(stage,with_replay=False):
    out,private=paths(stage)
    require(out.is_dir() and not out.is_symlink())
    names={'protocol.json','aggregate.json','completed.json','progress.json'}
    folds=[0] if stage=='pilot' else list(range(5))
    if stage!='evaluate':names|={f'fold{f}.json' for f in folds}
    require({x.name for x in out.iterdir()}==names)
    p=read(out/'protocol.json');a=read(out/'aggregate.json');t=read(out/'completed.json')
    require(p==protocol(stage,p['binding'],p['dependency']) and t==terminal(out))
    require(p['binding']['R7_binding']==r7source.authenticate())
    components={}
    if stage=='evaluate':
        kernel.validate_result(a)
        if with_replay:
            require(read(replay_path())=={**t,'status':'Q4_independent_full_replay_passed'}
                and read(out/'progress.json')['phase']=='independent_replay')
    else:
        updates=100 if stage=='pilot' else 1500
        require(private.is_dir() and not private.is_symlink() and private.stat().st_mode&0o777==0o700)
        require(set(a)=={'schema','status','component_sha256','updates_total','training_seconds',
            'projected_fit_seconds_training_only','heldout_inference_performed','encoder_unchanged',
            'native_heads_unchanged','checkpoint_replay_exact','patient_level_output_emitted'})
        require(a['schema']=='bran-r7-q4-fit-aggregate-v1' and a['status']=='completed'
            and set(a['component_sha256'])=={f'fold{f}.json' for f in folds}
            and a['updates_total']==updates*len(folds)
            and a['heldout_inference_performed'] is False and a['patient_level_output_emitted'] is False
            and all(a[k] is True for k in ('encoder_unchanged','native_heads_unchanged','checkpoint_replay_exact')))
        for fold in folds:
            name=f'fold{fold}.json';require(sha(out/name)==a['component_sha256'][name])
            item=read(out/name);q1.receipt(item['training'],updates)
            teacher,transform=load_teacher(p['binding'],fold)
            original=p['binding']['R7_binding']['checkpoint_manifest']['folds'][fold]
            expected=head_binding(fold,original,transform,t['protocol_sha256'],updates)
            require(item['binding']==expected and item['fold']==fold and item['checkpoint_replay_exact'] is True
                and item['screening_unchanged'] is True and item['heldout_inference_performed'] is False)
            load_head(private/f'fold{fold}.pt',item['checkpoint_sha256'],teacher,expected)
            components[fold]=item
    return p,a,components,t


def fit(stage,paired,evidence,out,private,pin):
    folds=[0] if stage=='pilot' else list(range(5));updates=100 if stage=='pilot' else 1500
    slots=tuple(paired.names.index(f) for f in CBC_FIELDS);age=original_age(paired)
    components={};seconds=0.
    for fold in folds:
        progress(out,'initial_replay',fold)
        teacher,t=load_teacher(evidence,fold)
        before,tp,grads=p1._validate_r7(teacher,t,fold,slots,paired.transforms[fold])
        rows=np.flatnonzero(paired.folds!=fold);require(len(rows)>=96)
        c,cm=t.clinical(paired.c,paired.cm);r,rm=t.retinal(paired.r,paired.rm)
        args=(tensor(c[rows]),tensor(cm[rows],torch.bool),tensor(r[rows]),tensor(rm[rows],torch.bool),
              subset_age(age,rows),t.age_mean,t.age_scale)
        probe=tuple(x[:96] if isinstance(x,torch.Tensor) else x for x in args[:4])+(
            subset_age(age,rows[:96]),t.age_mean,t.age_scale)
        initial=QuantileCBC(teacher.cbc_joint_head,98301+fold)
        screen=infer_native(teacher,*probe).screening_probability.clone()
        for pattern in PATTERNS:
            q=kernel.predict_quantiles(teacher,initial,*probe,pattern,slots)
            ref=completion_predictions(teacher,*probe,pattern,slots)
            require(torch.allclose(q.median,ref.cbc_standardized,rtol=0,atol=0,equal_nan=True))
        progress(out,'fit',fold)
        head,rec=kernel.train_quantile(teacher,*args,slots,torch.as_tensor(paired.folds[rows]),fold,
                                       updates=updates,batch_size=96)
        safe=q1.receipt(rec,updates);seconds+=safe['elapsed_seconds']
        require(p1._role_unchanged(before,tp,grads,teacher,t)
            and np.array_equal(screen.numpy(),infer_native(teacher,*probe).screening_probability.numpy(),equal_nan=True))
        original=evidence['R7_binding']['checkpoint_manifest']['folds'][fold]
        bind=head_binding(fold,original,t,pin,updates)
        checkpoint=save_head(private/f'fold{fold}.pt',head,teacher,bind)
        restored=load_head(private/f'fold{fold}.pt',checkpoint,teacher,bind)
        progress(out,'checkpoint_replay',fold)
        for pattern in PATTERNS:
            q=kernel.predict_quantiles(teacher,head,*probe,pattern,slots)
            rq=kernel.predict_quantiles(teacher,restored,*probe,pattern,slots)
            for key in ('q05','median','q95','scoringmask','abstained'):
                require(np.array_equal(getattr(q,key).numpy(),getattr(rq,key).numpy(),equal_nan=True))
        require(p1._role_unchanged(before,tp,grads,teacher,t))
        name=f'fold{fold}.json'
        write_json(out/name,{'fold':fold,'binding':bind,'checkpoint_sha256':checkpoint,'training':safe,
            'checkpoint_replay_exact':True,'screening_unchanged':True,'heldout_inference_performed':False})
        components[name]=sha(out/name)
    return {'schema':'bran-r7-q4-fit-aggregate-v1','status':'completed','component_sha256':components,
        'updates_total':updates*len(folds),'training_seconds':seconds,
        'projected_fit_seconds_training_only':seconds*75 if stage=='pilot' else None,
        'heldout_inference_performed':False,'encoder_unchanged':True,'native_heads_unchanged':True,
        'checkpoint_replay_exact':True,'patient_level_output_emitted':False}


def run(stage,replay=False):
    torch.set_num_threads(2)
    out,private=paths(stage)
    with LOCK.open('a+b') as lock,quiet():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not out.is_symlink() and not private.is_symlink())
        if replay:
            require(stage=='evaluate' and not replay_path().exists())
            authenticate(stage)
        else:
            require(not out.exists() and not private.exists())
            out.mkdir(mode=0o700)
            if stage!='evaluate':
                require(private.parent.is_dir() and not private.parent.is_symlink())
                private.mkdir(mode=0o700)
        try:
            progress(out,'authentication')
            dep=None if stage=='pilot' else authenticate('pilot' if stage=='fit' else 'fit')[3]
            progress(out,'source_loading')
            paired,roles,evidence=source_context();p=protocol(stage,evidence,dep)
            if replay:require(read(out/'protocol.json')==p)
            else:write_json(out/'protocol.json',p)
            if stage=='evaluate':
                _,_,components,_=authenticate('fit');_,fitprivate=paths('fit')
                def provider(fold):
                    teacher,t=load_teacher(evidence,fold);item=components[fold]
                    head=load_head(fitprivate/f'fold{fold}.pt',item['checkpoint_sha256'],teacher,item['binding'])
                    return teacher,t,head
                result=kernel.evaluate(paired,roles,provider,lambda phase,fold:progress(out,phase,fold))
                kernel.validate_result(result)
            else:result=fit(stage,paired,evidence,out,private,sha(out/'protocol.json'))
            progress(out,'post_authentication')
            _,_,again=source_context();require(again==evidence and p==protocol(stage,evidence,dep))
            if replay:
                require(read(out/'aggregate.json')==result and read(out/'completed.json')==terminal(out))
                write_json(replay_path(),{**terminal(out),'status':'Q4_independent_full_replay_passed'})
                progress(out,'independent_replay')
            else:
                write_json(out/'aggregate.json',result);write_json(out/'completed.json',terminal(out))
                progress(out,'completed')
        except BaseException as exc:
            locations=[{'file':Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(exc.__traceback__)
                       if Path(x.filename).name in set(OWN_CODE)|set(q1.CODE)|set(r7source.CODE)]
            write_json(out/('replay_failure.json' if replay else 'failure.json'),
                {'status':'closed_Q4_failure','own_code_locations':locations,'patient_level_output_emitted':False})
            raise
    return {'status':'Q4_replayed' if replay else 'Q4_completed','stage':stage,'patient_level_output_emitted':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('pilot','fit','evaluate'))
    parser.add_argument('--replay',action='store_true');parser.add_argument('--authenticate',action='store_true')
    args=parser.parse_args()
    try:
        with quiet():result=authenticate(args.stage,args.replay)[3] if args.authenticate else run(args.stage,args.replay)
        print(json.dumps(result,sort_keys=True))
    except BaseException:
        print(json.dumps({'status':'closed_Q4_failure','stage':args.stage,'patient_level_output_emitted':False}))
        raise SystemExit(1) from None
