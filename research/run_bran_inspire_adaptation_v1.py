"""Exclusive local INSPIRE adaptation, with frozen V5, source and private replay.

Only closed disclosure-safe aggregates leave the FD-quiet boundary. No encoder
fitting, source qualification rescue, or external-result selection is allowed.
"""
import argparse
from dataclasses import fields
import fcntl
import json
import os
from pathlib import Path
import pickle
import platform
import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
import run_bran_inspire_source_qualification_v1 as source_run
import bran_inspire_source_contract_v1 as source
import bran_inspire_v5_state_v1 as state_kernel

ROOT=Path(__file__).resolve().parent
LOCK=source_run.LOCK
ERROR='bran_inspire_adaptation_v1_failed'
DECISION_PIN='13bdd6dce2f2ccbb51008d0c90ea40df88f03805ff451017b97bf359e1d490d3'
SOURCE_PROTOCOL_PIN='7877f4628cddf970103dc72a8130c94a31f0582cf74bf9375e4f929f23bb97d3'
SOURCE_AGGREGATE_PIN='4ebd23fe14f40488fb33fc01f5e6519de01182990166f2ed9ea285efde8647e1'
TRAINING_SOURCES=('aireadi','brset','eicu','mimiciii','mimiciv','nhanes_exposed','nwicu')
CODE=('run_bran_inspire_adaptation_v1.py','test_run_bran_inspire_adaptation_v1.py',
    'run_bran_inspire_adaptation_v1_attempt1.sh','bran_inspire_adaptation_v1.py',
    'test_bran_inspire_adaptation_v1.py','bran_inspire_input_adapter_v1.py',
    'test_bran_inspire_input_adapter_v1.py','bran_inspire_v5_state_v1.py',
    'test_bran_inspire_v5_state_v1.py','bran_mimic_clinical_outcomes_v3.py',
    'bran_clinical_dictionary_binding_v1.py')
PHASES=('authentication','source_load','state_inference','checkpoint_replay',
    'fixed_readouts_and_bootstrap','private_write_and_replay','post_authentication','completed')
sha=source_run.sha
write=source_run.write
read=source_run.read


def require(ok):
    if not ok:raise ValueError(ERROR)


def paths(attempt):
    require(type(attempt) is int and 1<=attempt<=99)
    return ROOT/f'BRAN_INSPIRE_ADAPTATION_V1_ATTEMPT{attempt}', ROOT/'private_artifacts'/f'bran_inspire_adaptation_v1_attempt{attempt}'


def authenticate_inputs():
    """Authenticate receipts and bytes only; no source arrays materialized."""
    import run_bran_v5_cbc_uncertainty as v5
    from bran_multisource_protocol_v2 import validate_receipts
    import sklearn
    import torch
    admitted=source_run.audit(1)
    require(admitted['protocol_sha256']==SOURCE_PROTOCOL_PIN
        and admitted['aggregate_sha256']==SOURCE_AGGREGATE_PIN and admitted['role_support_met'] is True)
    source_out,_=source_run.paths(1)
    aggregate=read(source_out/'aggregate.json')
    decision_path=ROOT/'BRAN_NATURE_WEEK_MODEL_DECISION_2026-09-21.json'
    require(sha(decision_path)==DECISION_PIN)
    decision=read(decision_path)
    require(decision['research_lead']=='V5_M_attempt2' and decision['coordinate_frames_may_be_pooled'] is False)
    base,fp,_,components=v5.small_authentication()
    selected=decision['single_downstream_coordinate_frame']
    require(selected=={k:base['fold_checkpoints'][0][k] for k in
        ('fold','checkpoint_path','checkpoint_sha256','receipt_sha256')} and selected['fold']==0)
    require(components[('M',0)]['checkpoint_sha256']==selected['checkpoint_sha256'])
    roles=fp['source_binding']['source_roles'];validate_receipts(roles)
    require(tuple(sorted(k for k,v in roles.items() if v['disposition']=='training'))==TRAINING_SOURCES)
    require(roles['inspire']['disposition']!='training' and fp['source_binding']['protected_sources_used'] is False)
    code={n:sha(ROOT/n) for n in sorted(set(CODE)|set(fp['code_sha256']))}
    require(all(code[n]==h for n,h in fp['code_sha256'].items()))
    return {'source_protocol_sha256':SOURCE_PROTOCOL_PIN,'source_aggregate_sha256':SOURCE_AGGREGATE_PIN,
        'source_private_sha256':aggregate['private_sha256'],'model_decision_sha256':DECISION_PIN,
        'selected_frame':selected,'checkpoint_binding':components[('M',0)]['binding'],
        'training_sources':list(TRAINING_SOURCES),'inspire_encoder_exposed':False,
        'code_sha256':code,'environment':{'python':platform.python_version(),'numpy':np.__version__,
            'sklearn':sklearn.__version__,'torch':torch.__version__}}


def private_file(path,pin):
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink==1
        and path.stat().st_mode&0o777==0o600 and not path.parent.is_symlink()
        and path.parent.stat().st_mode&0o777==0o700 and sha(path)==pin)


def load_source(receipt):
    out,private=source_run.paths(1);path=private/'frame.npz'
    private_file(path,receipt['source_private_sha256'])
    names=[f.name for f in fields(source.PrivateInspireFrame)]
    with np.load(path,allow_pickle=False) as saved:
        require(set(saved.files)==set(names));arrays={k:saved[k] for k in names}
    private_file(path,receipt['source_private_sha256'])
    f=source.PrivateInspireFrame(**arrays)
    n=len(f.person)
    require(n>0 and f.person.dtype==np.dtype('U24') and len(np.unique(f.person))==n
        and len(np.unique(f.operation))==n and f.values.shape==(n,18) and f.observed.shape==(n,18)
        and f.observed.dtype==bool and f.context.shape==(n,4) and f.context.dtype==np.dtype('U64')
        and np.isin(f.outcome,[-1,0,1]).all() and np.isin(f.role,[0,1,2]).all())
    require(source.safe_support(f)==read(out/'aggregate.json')['support'])
    return f


def infer(f,receipt):
    import run_bran_context_preservation_v5 as v5
    selected=receipt['selected_frame']
    model,transform=v5.oldfit.load_checkpoint(ROOT/selected['checkpoint_path'],
        selected['checkpoint_sha256'],receipt['checkpoint_binding'])
    return state_kernel.infer(f.values,f.observed,f.age_lower,f.age_upper,f.age_kind,
        model,transform,receipt['checkpoint_binding']['transform_sha256'])


def evaluation_arguments(f,state):
    require(state.state.shape==(len(f.person),192)
        and np.array_equal(state.available,f.observed.any(axis=1)))
    keep=f.chronology_eligible & (f.outcome>=0) & state.available
    return (f.values[keep],f.observed[keep],f.age_lower[keep],f.age_upper[keep],f.age_kind[keep],
        f.context[keep],state.state[keep].astype(np.float64),f.role[keep],f.outcome[keep])


def validate_result(result):
    import bran_inspire_adaptation_v1 as kernel
    require(set(result)=={'schema','report','source_replay_equal','checkpoint_replay_equal','readout_replay_equal',
        'private_sha256','encoder_updated','external_encoder_selection','patient_level_output_emitted','clinical_use'})
    require(result['schema']=='bran-inspire-adaptation-aggregate-v1')
    require(all(result[k] is True for k in ('source_replay_equal','checkpoint_replay_equal','readout_replay_equal')))
    require(all(result[k] is False for k in ('encoder_updated','external_encoder_selection','patient_level_output_emitted','clinical_use')))
    require(set(result['private_sha256'])=={'states.npz','readouts.pkl'})
    require(all(type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v)
        for v in result['private_sha256'].values()))
    require(kernel.validate_report(result['report']) is True)


def audit(attempt):
    out,private=paths(attempt)
    require(out.is_dir() and private.is_dir() and not out.is_symlink() and not private.is_symlink()
        and not private.parent.is_symlink())
    require({p.name for p in out.iterdir()}=={'protocol.json','aggregate.json','completed.json'})
    require({p.name for p in private.iterdir()}=={'states.npz','readouts.pkl'})
    require(all(p.is_file() and not p.is_symlink() and p.stat().st_nlink==1 for p in out.iterdir()))
    result=read(out/'aggregate.json');protocol=read(out/'protocol.json');terminal=read(out/'completed.json')
    require(terminal=={'status':'completed','protocol_sha256':sha(out/'protocol.json'),
        'aggregate_sha256':sha(out/'aggregate.json'),'patient_level_output_emitted':False})
    require(protocol=={'schema':'bran-inspire-adaptation-protocol-v1','inputs':authenticate_inputs(),
        'scientific_design_sha256':sha(ROOT/'BRAN_INSPIRE_ADAPTATION_V1_DESIGN.md'),
        'task':'protected_encoder_source_external_adaptation','patient_level_output_emitted':False})
    validate_result(result)
    for name,pin in result['private_sha256'].items():private_file(private/name,pin)
    return {'status':'authenticated','protocol_sha256':terminal['protocol_sha256'],
        'aggregate_sha256':terminal['aggregate_sha256'],'patient_level_output_emitted':False}


def run(attempt):
    import bran_inspire_adaptation_v1 as kernel
    out,private=paths(attempt)
    require(not out.exists() and not out.is_symlink() and not private.exists() and not private.is_symlink())
    require(private.parent.is_dir() and not private.parent.is_symlink())
    phase='authentication'
    def progress(name):
        nonlocal phase
        require(name in PHASES);phase=name
        print(json.dumps({'phase':name,'status':'started','patient_level_output_emitted':False}),flush=True)
    with LOCK.open('a') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'not_started_shared_lock_busy'}
        out.mkdir(mode=0o700);private.mkdir(mode=0o700)
        try:
            progress('authentication')
            with _quiet():
                receipt=authenticate_inputs()
                protocol={'schema':'bran-inspire-adaptation-protocol-v1','inputs':receipt,
                    'scientific_design_sha256':sha(ROOT/'BRAN_INSPIRE_ADAPTATION_V1_DESIGN.md'),
                    'task':'protected_encoder_source_external_adaptation','patient_level_output_emitted':False}
                write(out/'protocol.json',protocol);protocol_pin=sha(out/'protocol.json')
            progress('source_load')
            with _quiet():f=load_source(receipt)
            progress('state_inference')
            with _quiet():state=infer(f,receipt)
            progress('checkpoint_replay')
            with _quiet():
                second=infer(f,receipt)
                require(np.array_equal(state.state,second.state) and np.array_equal(state.available,second.available))
                del second
                args=evaluation_arguments(f,state)
            progress('fixed_readouts_and_bootstrap')
            with _quiet():
                sink={};report=kernel.evaluate(*args,private_sink=sink)
                require(kernel.validate_report(report) is True)
            progress('private_write_and_replay')
            with _quiet():
                with (private/'states.npz').open('xb') as handle:
                    os.fchmod(handle.fileno(),0o600);np.savez_compressed(handle,state=state.state,available=state.available)
                with (private/'readouts.pkl').open('xb') as handle:
                    os.fchmod(handle.fileno(),0o600);pickle.dump(sink,handle,protocol=5)
                pins={n:sha(private/n) for n in ('states.npz','readouts.pkl')}
                private_file(private/'states.npz',pins['states.npz'])
                with np.load(private/'states.npz',allow_pickle=False) as saved:
                    require(set(saved.files)=={'state','available'}
                        and np.array_equal(saved['state'],state.state) and np.array_equal(saved['available'],state.available))
                private_file(private/'states.npz',pins['states.npz'])
                private_file(private/'readouts.pkl',pins['readouts.pkl'])
                with (private/'readouts.pkl').open('rb') as handle:restored=pickle.load(handle)
                private_file(private/'readouts.pkl',pins['readouts.pkl'])
                require(kernel.replay(*args,private_sink=restored) is True)
            progress('post_authentication')
            with _quiet():
                require(authenticate_inputs()==receipt and sha(out/'protocol.json')==protocol_pin)
                source_replay=load_source(receipt)
                require(all(np.array_equal(getattr(f,k.name),getattr(source_replay,k.name),equal_nan=True)
                    if getattr(f,k.name).dtype.kind=='f' else np.array_equal(getattr(f,k.name),getattr(source_replay,k.name))
                    for k in fields(source.PrivateInspireFrame)))
                result={'schema':'bran-inspire-adaptation-aggregate-v1','report':report,
                    'source_replay_equal':True,'checkpoint_replay_equal':True,'readout_replay_equal':True,
                    'private_sha256':pins,'encoder_updated':False,'external_encoder_selection':False,
                    'patient_level_output_emitted':False,'clinical_use':False}
                validate_result(result);write(out/'aggregate.json',result)
                write(out/'completed.json',{'status':'completed','protocol_sha256':protocol_pin,
                    'aggregate_sha256':sha(out/'aggregate.json'),'patient_level_output_emitted':False})
            return {'status':'completed','encoder_updated':False,'patient_level_output_emitted':False}
        except Exception:
            with _quiet():
                if not (out/'completed.json').exists():
                    write(out/'failure.json',{'status':'failed','phase':phase,'error':ERROR,
                        'private_artifacts_admitted':False,'patient_level_output_emitted':False})
            return {'status':'failed','phase':phase,'error':ERROR,'patient_level_output_emitted':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,required=True);p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    try:
        if a.audit_only:
            with _quiet():result=audit(a.attempt)
        else:result=run(a.attempt)
    except Exception:result={'status':'not_completed','error':ERROR,'patient_level_output_emitted':False}
    print(json.dumps(result,sort_keys=True),flush=True)
    return 0 if result['status'] in ('completed','authenticated') else 1


if __name__=='__main__':raise SystemExit(main())
