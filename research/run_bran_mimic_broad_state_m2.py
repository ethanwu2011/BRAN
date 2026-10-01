"""Private M1-to-V5 fold-0 inference; no labels enter the encoder."""
import fcntl
import json
import os
from pathlib import Path
import numpy as np
import torch
import run_bran_mimic_broad_admission_m1 as admission
import run_bran_eicu_v5_state_e3 as reference
import bran_broad_clinical_state_v1 as bridge
from bran_mimic_selected_state_v3 import _typed_age_arrays
from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json
from run_bran_source_pattern_v6 import safe_site

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'BRAN_MIMIC_BROAD_STATE_M2_ATTEMPT1'
PRIVATE = ROOT/'private_artifacts'/'bran_mimic_broad_state_m2_attempt1'
ERROR = 'mimic_broad_state_m2_contract_failed'
FALSE_FLAGS = reference.FALSE_FLAGS


def require(ok):
    if not ok: raise ValueError(ERROR) from None


def code_hashes():
    own = ('run_bran_mimic_broad_state_m2.py','test_bran_mimic_broad_state_m2.py',
           'bran_broad_clinical_state_v1.py','test_bran_broad_clinical_state_v1.py',
           'BRAN_MIMIC_BROAD_STATE_M2_DESIGN.md','bran_mimic_selected_state_v3.py')
    return {**reference.code_hashes(),**admission.code_hashes(),
            **{n:sha(ROOT/n) for n in own}}


def source_receipt():
    a,t = admission.authenticate()
    return {'protocol_sha256':t['protocol_sha256'],'aggregate_sha256':t['aggregate_sha256'],
            'terminal_sha256':sha(admission.OUT/'completed.json'),
            'private_sha256':a['private_sha256']['cohort.npz']}


def model_receipt(): return reference.model_receipt()


def load_cohort(source):
    path = admission.PRIVATE/'cohort.npz'
    require(sha(path) == source['private_sha256'])
    a = admission._load_private(path)
    require(sha(path) == source['private_sha256'])
    return a


def infer(cohort,receipt):
    binding=receipt['provider']; model,transform=reference.provider.provider(binding)
    ages=_typed_age_arrays(cohort)
    return bridge.infer(cohort['values'],cohort['observed'],**ages,model=model,
        transform=transform,expected_transform_sha256=binding['binding']['transform_sha256'],batch_size=256)


def packed(result): return reference.packed(result)


def validate_arrays(arrays,cohort):
    try: reference.validate_arrays(arrays,cohort)
    except Exception: raise ValueError(ERROR) from None


def phase(name):
    require(name in reference.PHASES)
    write_json(OUT/'progress.next.json',{'phase':name,'pid':os.getpid(),'patient_level_output_emitted':False})
    os.replace(OUT/'progress.next.json',OUT/'progress.json')


def summary(arrays):
    return {'source_local_encoded_people_lower_bound_20':admission._coarse(len(arrays['state'])),
            'state_width':192,'all_admitted_people_encoded':True,
            'coordinate_frame':'V5_M_attempt2_fold0','checkpoint_replay_exact':True}


def authenticate(*,_pending_terminal=None):
    names={'protocol.json','aggregate.json','progress.json'}
    if _pending_terminal is None: names.add('completed.json')
    require(OUT.is_dir() and not OUT.is_symlink() and {p.name for p in OUT.iterdir()}==names)
    p,a=(admission._read_json(OUT/n) for n in ('protocol.json','aggregate.json'))
    t=_pending_terminal if _pending_terminal is not None else admission._read_json(OUT/'completed.json')
    require(set(p)=={'schema','status','source','model','code_sha256','batch_size',*FALSE_FLAGS}
        and p['schema']=='bran-mimic-broad-state-m2-protocol' and p['status']=='frozen_before_inference'
        and p['source']==source_receipt() and p['model']==model_receipt()
        and p['code_sha256']==code_hashes() and p['batch_size']==256
        and all(p[k] is False for k in FALSE_FLAGS))
    require(set(a)=={'schema','status','summary','private_sha256',*FALSE_FLAGS}
        and a['schema']=='bran-mimic-broad-state-m2' and a['status']=='completed'
        and all(a[k] is False for k in FALSE_FLAGS))
    require(PRIVATE.is_dir() and not PRIVATE.is_symlink() and PRIVATE.stat().st_mode&0o777==0o700
        and {f.name for f in PRIVATE.iterdir()}=={'state.npz'})
    path=PRIVATE/'state.npz'
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink==1
        and path.stat().st_mode&0o777==0o600 and sha(path)==a['private_sha256'])
    with np.load(path,allow_pickle=False) as h: arrays={k:h[k] for k in h.files}
    validate_arrays(arrays,load_cohort(p['source']))
    require(a['summary']==summary(arrays) and sha(path)==a['private_sha256'])
    require(set(t)=={'status','protocol_sha256','aggregate_sha256','patient_level_output_emitted'}
        and t['status']=='authenticated_completed' and t['patient_level_output_emitted'] is False
        and t['protocol_sha256']==sha(OUT/'protocol.json') and t['aggregate_sha256']==sha(OUT/'aggregate.json'))
    q=admission._read_json(OUT/'progress.json')
    require(set(q)=={'phase','pid','patient_level_output_emitted'} and q['phase']=='completed'
        and type(q['pid']) is int and q['pid']>0 and q['patient_level_output_emitted'] is False)
    return a,t


def run(state):
    require(not OUT.exists() and not OUT.is_symlink() and not PRIVATE.exists() and not PRIVATE.is_symlink())
    OUT.mkdir(mode=0o700);state['owned']=True;phase('authentication')
    source,model=source_receipt(),model_receipt()
    p={'schema':'bran-mimic-broad-state-m2-protocol','status':'frozen_before_inference',
        'source':source,'model':model,'code_sha256':code_hashes(),'batch_size':256,
        **{k:False for k in FALSE_FLAGS}}
    write_json(OUT/'protocol.json',p)
    cohort=load_cohort(source);phase('inference')
    arrays=packed(infer(cohort,model));validate_arrays(arrays,cohort)
    phase('checkpoint_replay');replay=packed(infer(cohort,model))
    require(all(np.array_equal(arrays[k],replay[k]) for k in arrays));del replay
    phase('private_artifact')
    require(PRIVATE.parent.is_dir() and not PRIVATE.parent.is_symlink());PRIVATE.mkdir(mode=0o700)
    path=PRIVATE/'state.npz';fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as h:
        np.savez_compressed(h,**arrays);h.flush();os.fsync(h.fileno())
    phase('post_authentication')
    require(source==source_receipt() and model==model_receipt() and p['code_sha256']==code_hashes())
    a={'schema':'bran-mimic-broad-state-m2','status':'completed','summary':summary(arrays),
        'private_sha256':sha(path),**{k:False for k in FALSE_FLAGS}}
    write_json(OUT/'aggregate.json',a);phase('completed')
    terminal={'status':'authenticated_completed','protocol_sha256':sha(OUT/'protocol.json'),
        'aggregate_sha256':sha(OUT/'aggregate.json'),'patient_level_output_emitted':False}
    authenticate(_pending_terminal=terminal);write_json(OUT/'completed.json',terminal)


def main():
    ok=False;state={'owned':False}
    with quiet():
        try:
            torch.set_num_threads(1)
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                run(state);authenticate();ok=True
        except Exception as exc:
            if state['owned'] and OUT.is_dir():
                if (OUT/'completed.json').is_file():os.replace(OUT/'completed.json',OUT/'rejected_completed.json')
                write_json(OUT/'failure.json',{'status':'technical_failure','safe_code_site':safe_site(exc),
                    'patient_level_output_emitted':False})
    print(json.dumps({'status':'completed' if ok else 'not_completed','patient_level_output_emitted':False}))
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
