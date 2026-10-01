"""One locked, quiet two-source clinical study; no encoder fitting."""
import fcntl
import json
import os
from pathlib import Path
import joblib
import numpy as np
import torch
import bran_multisource_clinical_panel_s1 as panel
import bran_multisource_clinical_inputs_s1 as inputs
import run_bran_mimic_broad_state_m2 as mimic
import run_bran_eicu_v5_state_e3 as eicu
from run_bran_multisource_retinal_features_v2 import LOCK,quiet,sha,write_json
from diagnose_bran_agefree_reference_failure_v1 import safe_trace

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'BRAN_MULTISOURCE_CLINICAL_S1_ATTEMPT1'
PRIVATE=ROOT/'private_artifacts'/'bran_multisource_clinical_s1_attempt1'
ERROR='multisource_clinical_s1_failed'
PHASES=('authentication','source_assembly','bran_structure','raw_structure','outcome_utility',
        'private_write','object_replay','post_authentication','completed')
read=mimic.admission._read_json


def require(ok):
    if not ok:raise ValueError(ERROR) from None


def code_hashes():
    own=('BRAN_MULTISOURCE_CLINICAL_S1_DESIGN.md','bran_multisource_clinical_design_s1.py',
        'test_bran_multisource_clinical_design_s1.py','bran_multisource_clinical_panel_s1.py',
        'test_bran_multisource_clinical_panel_s1.py','bran_multisource_clinical_inputs_s1.py',
        'test_bran_multisource_clinical_inputs_s1.py','run_bran_multisource_clinical_s1.py',
        'test_run_bran_multisource_clinical_s1.py','bran_r1_hf_structure_kernel_v1.py',
        'bran_mimic_clinical_outcomes_v3.py','bran_mimic_clinical_panel_v3.py',
        'diagnose_bran_agefree_reference_failure_v1.py')
    return {**mimic.code_hashes(),**eicu.code_hashes(),**{n:sha(ROOT/n) for n in own}}


def sources():
    result={}
    for key,runner in (('mimic',mimic),('eicu',eicu)):
        a,t=runner.authenticate();p=read(runner.OUT/'protocol.json')
        result[key]={'terminal':t,'terminal_sha256':sha(runner.OUT/'completed.json'),
            'private_sha256':a['private_sha256'],'model':p['model'],'source':p['source']}
    require(result['mimic']['model']==result['eicu']['model'])
    return result


def load_frame(receipts):
    cohorts={};states={}
    for key,runner in (('mimic',mimic),('eicu',eicu)):
        cohorts[key]=runner.load_cohort(receipts[key]['source'])
        path=runner.PRIVATE/'state.npz';require(sha(path)==receipts[key]['private_sha256'])
        with np.load(path,allow_pickle=False) as h:states[key]={n:h[n] for n in h.files}
        runner.validate_arrays(states[key],cohorts[key]);require(sha(path)==receipts[key]['private_sha256'])
    f=inputs.assemble(cohorts['mimic'],states['mimic'],cohorts['eicu'],states['eicu'])
    panel.check_frame(f);return f


def protocol(receipts):
    import sklearn,scipy,sys
    return {'schema':'bran-multisource-clinical-s1-protocol','sources':receipts,'code_sha256':code_hashes(),
        'status':'frozen_before_clustering','families':list(panel.FAMILIES),
        'runtime':{'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,
                   'joblib':joblib.__version__,'python':sys.version.split()[0]},**panel.FLAGS}


def phase(name,family=None):
    require(name in PHASES and (family is None or family in panel.FAMILIES))
    write_json(OUT/'progress.next.json',{'phase':name,'family':family,'pid':os.getpid(),
        'patient_level_output_emitted':False})
    os.replace(OUT/'progress.next.json',OUT/'progress.json')


def regular(path,mode=None):
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink==1)
    if mode is not None:require(path.stat().st_mode&0o777==mode)


def validate_component(a,family):
    require(set(a)=={'result','objects_sha256','object_replay_exact','patient_level_output_emitted'}
        and a['result']['family']==family and a['object_replay_exact'] is True
        and a['patient_level_output_emitted'] is False)
    panel.validate_report(a['result']);require(mimic.admission._valid_hex(a['objects_sha256']))


def authenticate(*,_pending_terminal=None):
    names={'protocol.json','aggregate.json','progress.json',*(f+'.json' for f in panel.FAMILIES)}
    if _pending_terminal is None:names.add('completed.json')
    require(OUT.is_dir() and not OUT.is_symlink() and {p.name for p in OUT.iterdir()}==names)
    p=read(OUT/'protocol.json');require(p==protocol(sources()))
    require(PRIVATE.is_dir() and not PRIVATE.is_symlink() and PRIVATE.stat().st_mode&0o777==0o700
        and {f.name for f in PRIVATE.iterdir()}=={f+'.joblib' for f in panel.FAMILIES})
    components={};objects={}
    for family in panel.FAMILIES:
        a=read(OUT/(family+'.json'));validate_component(a,family)
        path=PRIVATE/(family+'.joblib');regular(path,0o600)
        require(sha(path)==a['objects_sha256']);objects[family]=sha(path);components[family]=sha(OUT/(family+'.json'))
    a=read(OUT/'aggregate.json')
    require(a=={'schema':'bran-multisource-clinical-s1','status':'completed',
        'components_sha256':components,'private_objects_sha256':objects,**panel.FLAGS})
    q=read(OUT/'progress.json')
    require(set(q)=={'phase','family','pid','patient_level_output_emitted'} and q['phase']=='completed'
        and q['family'] is None and type(q['pid']) is int and q['pid']>0 and q['patient_level_output_emitted'] is False)
    t=_pending_terminal if _pending_terminal is not None else read(OUT/'completed.json')
    require(t=={'status':'authenticated_completed','protocol_sha256':sha(OUT/'protocol.json'),
        'aggregate_sha256':sha(OUT/'aggregate.json'),'patient_level_output_emitted':False})
    return a,t


def run(owned):
    require(not OUT.exists() and not OUT.is_symlink() and not PRIVATE.exists() and not PRIVATE.is_symlink())
    OUT.mkdir(mode=0o700);owned['owned']=True;phase('authentication')
    p=protocol(sources());write_json(OUT/'protocol.json',p)
    phase('source_assembly');frame=load_frame(p['sources'])
    require(PRIVATE.parent.is_dir() and not PRIVATE.parent.is_symlink());PRIVATE.mkdir(mode=0o700)
    components={};objects={}
    for family in panel.FAMILIES:
        result=panel.fit_family(frame,family,progress=lambda name:phase(name,family))
        panel.validate_report(result.aggregate);phase('private_write',family)
        path=PRIVATE/(family+'.joblib');fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,'wb') as h:joblib.dump(result.objects,h,compress=3);h.flush();os.fsync(h.fileno())
        pin=sha(path);phase('object_replay',family);regular(path,0o600)
        restored=joblib.load(path);require(sha(path)==pin)
        if result.objects:
            before=panel.replay(frame,result.objects);after=panel.replay(frame,restored)
            require(set(before)==set(after) and all(np.array_equal(before[k],after[k]) for k in before))
            del before,after
        else:require(restored=={})
        component={'result':result.aggregate,'objects_sha256':pin,'object_replay_exact':True,'patient_level_output_emitted':False}
        validate_component(component,family);write_json(OUT/(family+'.json'),component)
        objects[family]=pin;components[family]=sha(OUT/(family+'.json'));del result,restored
    phase('post_authentication');require(p==protocol(sources()))
    a={'schema':'bran-multisource-clinical-s1','status':'completed','components_sha256':components,
       'private_objects_sha256':objects,**panel.FLAGS}
    write_json(OUT/'aggregate.json',a);phase('completed')
    t={'status':'authenticated_completed','protocol_sha256':sha(OUT/'protocol.json'),
       'aggregate_sha256':sha(OUT/'aggregate.json'),'patient_level_output_emitted':False}
    authenticate(_pending_terminal=t);write_json(OUT/'completed.json',t)


def main():
    ok=False;owned={'owned':False}
    with quiet():
        try:
            torch.set_num_threads(1)
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(owned);authenticate();ok=True
        except Exception as exc:
            if owned['owned'] and OUT.is_dir():
                if (OUT/'completed.json').is_file():os.replace(OUT/'completed.json',OUT/'rejected_completed.json')
                write_json(OUT/'failure.json',{'status':'technical_failure',
                    'safe_exception_chain':safe_trace(exc,ROOT,set(code_hashes())),
                    'patient_level_output_emitted':False})
    print(json.dumps({'status':'completed' if ok else 'not_completed','patient_level_output_emitted':False}))
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
