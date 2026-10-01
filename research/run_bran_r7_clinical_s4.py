"""Local-only R7 inference and fixed S1 clinical analysis on the full pool."""
import argparse
import fcntl
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch

import run_bran_multisource_clinical_s1 as old
import run_bran_r7_fixed_state_p1 as p1
import bran_broad_clinical_state_v1 as bridge
import bran_r7_clinical_s4 as panel
from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'BRAN_R7_CLINICAL_S4_ATTEMPT1'
PRIVATE=ROOT/'private_artifacts'/'bran_r7_clinical_s4_attempt1'
ERROR='r7_clinical_s4_failed'
CODE=('BRAN_R7_CLINICAL_S4_DESIGN.md','run_bran_r7_clinical_s4.py',
      'test_run_bran_r7_clinical_s4.py','bran_r7_clinical_s4.py',
      'test_bran_r7_clinical_s4.py','run_bran_r7_clinical_s4_attempt1.sh')
PHASES=('authentication','source_assembly','v5_full_pool_replay','r7_inference',
        'r7_reload_replay','bran_structure','raw_structure','outcome_utility',
        'private_write','object_replay','post_authentication','completed')
FLAGS={**old.panel.FLAGS,'candidate_promoted':False}


def require(ok):
    if not ok:raise ValueError(ERROR) from None


def bindings():
    _,terminal=old.authenticate()
    sources=old.read(old.OUT/'protocol.json')['sources']
    receipt=p1.authenticate(1);pp=p1._read(p1.paths()/'protocol.json')
    v=pp['v5_checkpoint_manifest']['folds'][0]
    r=pp['r7_checkpoint_manifest']['folds'][0]
    require(v['fold']==r['fold']==0 and r['binding']['initial_checkpoint_sha256']==v['checkpoint_sha256']
        and r['binding']['transform_sha256']==v['binding']['transform_sha256'])
    for s in sources.values():
        require(s['model']['provider']['checkpoint_sha256']==v['checkpoint_sha256']
                and s['model']['provider']['binding']==v['binding'])
    names=set(CODE)|set(old.code_hashes())|set(pp['code_sha256'])
    return {'s1_terminal':terminal,'sources':sources,'p1_receipt':receipt,'V5':v,'R7':r,
            'code_sha256':{n:sha(ROOT/n) for n in sorted(names)}}


def protocol(b):
    import sklearn,scipy,sys
    return {'schema':'bran-r7-clinical-s4-protocol','status':'frozen_before_inference_and_clustering',
        'binding':b,'families':list(old.panel.FAMILIES),'frame':'R7_attempt1_fold0',
        'same_s1_population_roles_and_targets':True,'raw_clustering_refitted':False,
        'historical_partitions_previously_evaluated':True,
        'runtime':{'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,
                   'joblib':joblib.__version__,'python':sys.version.split()[0]},**FLAGS}


def checkpoint(role,b):
    if role=='R7':
        import run_bran_robust_clinical_r7 as r7
        from bran_robust_clinical_r7 import BRANRobustClinicalR7
        _,private=r7.paths('fit',1);entry=b[role]
        model,transform=r7.load_checkpoint(private/'fold0_R.pt',entry['checkpoint_sha256'],entry['binding'])
        require(type(model) is BRANRobustClinicalR7)
    else:
        require(role=='V5')
        model,transform=old.eicu.provider.provider(b['sources']['mimic']['model']['provider'])
    require(not model.training and not any(p.requires_grad for p in model.parameters()))
    return model,transform


def infer(frame,role,b):
    model,transform=checkpoint(role,b)
    result=np.empty_like(frame['state'])
    # Preserve each archived source's batch boundaries for exact floating-point
    # replay; neither outcomes nor disease memberships enter this operation.
    for source in (0,1):
        rows=frame['source']==source
        value=bridge.infer(frame['values'][rows],frame['observed'][rows],
            **{k:frame[k][rows] for k in ('age_value','age_lower','age_upper','age_kind')},
            model=model,transform=transform,expected_transform_sha256=b[role]['binding']['transform_sha256'],
            batch_size=256)
        require(value.available.all() and value.state.shape==result[rows].shape)
        result[rows]=value.state
    return result


def replace_state(frame,state):
    require(state.shape==frame['state'].shape and state.dtype==np.float32 and np.isfinite(state).all())
    result={**frame,'state':state}
    old.panel.check_frame(result)
    for key in frame:
        if key!='state':require(result[key] is frame[key])
    return result


def phase(state,name,family=None):
    require(name in PHASES and family in (None,*old.panel.FAMILIES));state['phase']=name
    write_json(OUT/'progress.next.json',{'phase':name,'family':family,'pid':os.getpid(),
        'patient_level_output_emitted':False})
    os.replace(OUT/'progress.next.json',OUT/'progress.json')


def private_write(path,writer):
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as h:writer(h);h.flush();os.fsync(h.fileno())


def read_reference(family):
    component=old.read(old.OUT/(family+'.json'));old.validate_component(component,family)
    path=old.PRIVATE/(family+'.joblib');old.regular(path,0o600)
    require(sha(path)==component['objects_sha256'])
    objects=joblib.load(path);require(sha(path)==component['objects_sha256'])
    return component,objects


def validate_component(c,family):
    old.validate_component(c,family)


def authenticate(*,_pending=None):
    names={'protocol.json','aggregate.json','progress.json',*(f+'.json' for f in old.panel.FAMILIES)}
    if _pending is None:names.add('completed.json')
    require(OUT.is_dir() and not OUT.is_symlink() and {x.name for x in OUT.iterdir()}==names)
    p=old.read(OUT/'protocol.json');require(p==protocol(bindings()))
    require(PRIVATE.is_dir() and not PRIVATE.is_symlink() and PRIVATE.stat().st_mode&0o777==0o700
        and {x.name for x in PRIVATE.iterdir()}=={'state.npz',*(f+'.joblib' for f in old.panel.FAMILIES)})
    components={};objects={}
    for family in old.panel.FAMILIES:
        c=old.read(OUT/(family+'.json'));validate_component(c,family)
        path=PRIVATE/(family+'.joblib');old.regular(path,0o600)
        require(sha(path)==c['objects_sha256']);objects[family]=sha(path)
        components[family]=sha(OUT/(family+'.json'))
    state_path=PRIVATE/'state.npz';old.regular(state_path,0o600)
    a=old.read(OUT/'aggregate.json')
    require(a=={'schema':'bran-r7-clinical-s4','status':'completed','components_sha256':components,
        'private_objects_sha256':objects,'private_state_sha256':sha(state_path),
        'v5_full_pool_replay_exact':True,'r7_reload_replay_exact':True,
        'same_s1_population_roles_and_targets':True,'raw_clustering_refitted':False,**FLAGS})
    q=old.read(OUT/'progress.json')
    require(set(q)=={'phase','family','pid','patient_level_output_emitted'} and q['phase']=='completed'
        and q['family'] is None and type(q['pid']) is int and q['pid']>0 and q['patient_level_output_emitted'] is False)
    t=_pending if _pending is not None else old.read(OUT/'completed.json')
    require(t=={'status':'authenticated_completed','protocol_sha256':sha(OUT/'protocol.json'),
        'aggregate_sha256':sha(OUT/'aggregate.json'),'patient_level_output_emitted':False})
    return a,t


def run(state):
    require(not OUT.exists() and not OUT.is_symlink() and not PRIVATE.exists() and not PRIVATE.is_symlink())
    OUT.mkdir(mode=0o700);state['owned']=True;phase(state,'authentication')
    b=bindings();p=protocol(b);write_json(OUT/'protocol.json',p)
    phase(state,'source_assembly');frame=old.load_frame(b['sources'])
    phase(state,'v5_full_pool_replay');replay=infer(frame,'V5',b)
    require(np.array_equal(replay,frame['state']));del replay
    phase(state,'r7_inference');fresh=infer(frame,'R7',b)
    phase(state,'r7_reload_replay');replay=infer(frame,'R7',b)
    require(np.array_equal(fresh,replay));del replay
    frame=replace_state(frame,fresh)
    require(PRIVATE.parent.is_dir() and not PRIVATE.parent.is_symlink());PRIVATE.mkdir(mode=0o700)
    private_write(PRIVATE/'state.npz',lambda h:np.savez_compressed(h,state=fresh))
    with np.load(PRIVATE/'state.npz',allow_pickle=False) as h:
        require(h.files==['state'] and np.array_equal(h['state'],fresh))
    components={};objects={}
    for family in old.panel.FAMILIES:
        component,reference=read_reference(family)
        result=panel.fit_family(frame,family,component,reference,progress=lambda n:phase(state,n,family))
        panel.validate_report(result.aggregate);phase(state,'private_write',family)
        path=PRIVATE/(family+'.joblib');private_write(path,lambda h:joblib.dump(result.objects,h,compress=3))
        pin=sha(path);phase(state,'object_replay',family);old.regular(path,0o600)
        restored=joblib.load(path);require(sha(path)==pin)
        if result.objects:
            before=panel.replay(frame,result.objects);after=panel.replay(frame,restored)
            require(set(before)==set(after) and all(np.array_equal(before[k],after[k]) for k in before))
            del before,after
        else:require(restored=={})
        c={'result':result.aggregate,'objects_sha256':pin,'object_replay_exact':True,'patient_level_output_emitted':False}
        validate_component(c,family);write_json(OUT/(family+'.json'),c)
        objects[family]=pin;components[family]=sha(OUT/(family+'.json'));del result,restored,reference
    phase(state,'post_authentication');require(p==protocol(bindings()))
    a={'schema':'bran-r7-clinical-s4','status':'completed','components_sha256':components,
        'private_objects_sha256':objects,'private_state_sha256':sha(PRIVATE/'state.npz'),
        'v5_full_pool_replay_exact':True,'r7_reload_replay_exact':True,
        'same_s1_population_roles_and_targets':True,'raw_clustering_refitted':False,**FLAGS}
    write_json(OUT/'aggregate.json',a);phase(state,'completed')
    t={'status':'authenticated_completed','protocol_sha256':sha(OUT/'protocol.json'),
        'aggregate_sha256':sha(OUT/'aggregate.json'),'patient_level_output_emitted':False}
    authenticate(_pending=t);write_json(OUT/'completed.json',t)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    state={'owned':False,'phase':'authentication'};ok=False;receipt=None
    with quiet():
        try:
            torch.set_num_threads(1)
            if args.audit_only:_,receipt=authenticate();ok=True
            else:
                with LOCK.open('a+b') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(state);authenticate();ok=True
        except Exception as exc:
            if state['owned']:
                if (OUT/'completed.json').is_file():os.replace(OUT/'completed.json',OUT/'rejected_completed.json')
                from diagnose_bran_agefree_reference_failure_v1 import safe_trace
                write_json(OUT/'failure.json',{'status':'technical_failure','phase':state['phase'],
                    'safe_exception_chain':safe_trace(exc,ROOT,set(CODE)|set(old.code_hashes())),
                    'patient_level_output_emitted':False})
    print(json.dumps(receipt or {'status':'completed' if ok else 'not_completed','phase':state['phase'],
                                'patient_level_output_emitted':False}))
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
