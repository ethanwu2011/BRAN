"""Exclusive quiet local U2 runner and full-source independent replay."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import traceback
import bran_r7_completion_utility_u2 as kernel
import run_bran_r7_head_blocks_i1 as parent
from run_bran_multisource_retinal_features_v2 import LOCK,quiet,sha,write_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'BRAN_R7_COMPLETION_UTILITY_U2_ATTEMPT1'
REPLAY=ROOT/'BRAN_R7_COMPLETION_UTILITY_U2_REPLAY.json'
DESIGN=ROOT/'BRAN_R7_COMPLETION_UTILITY_U2_DESIGN.md'
CODE=('BRAN_R7_COMPLETION_UTILITY_U2_DESIGN.md','bran_r7_completion_utility_u2.py',
    'run_bran_r7_completion_utility_u2.py','test_bran_r7_completion_utility_u2.py',
    'run_bran_r7_completion_utility_u2_attempt1.sh','bran_v5_completion_utility_u1.py',
    'bran_v5_completion_utility_readout_u1.py','bran_v5_completion_utility_metrics_u1.py',
    'test_bran_v5_completion_utility_readout_u1.py','test_bran_v5_completion_utility_metrics_u1.py')
PHASES=('authentication','source_loading','native_inference','checkpoint_replay','fixed_readouts',
        'readout_replay','aggregate_bootstrap','post_authentication','completed','independent_replay')
ERROR='r7_completion_utility_u2_runner_failed'


def require(ok):
    if not ok:raise ValueError(ERROR) from None


def progress(phase,fold=None):
    require(phase in PHASES and (fold is None or type(fold) is int and fold in range(5)))
    write_json(OUT/'progress.next.json',{'phase':phase,'fold':fold,'pid':os.getpid(),
                                      'patient_level_output_emitted':False})
    os.replace(OUT/'progress.next.json',OUT/'progress.json')


def binding():
    inherited=parent.authenticate()
    code=dict(inherited['code_sha256'])
    code.update({name:sha(ROOT/name) for name in CODE})
    return {'R7_binding':inherited,'code_sha256':code,'design_sha256':sha(DESIGN)}


def make_protocol(b,paired):
    from bran_multisource_protocol_v2 import digest
    names=[x for x in paired.endpoint_names if x!='mh_a1c']
    require(len(names)==25 and len(set(names))==25)
    protocol={'schema':'bran-r7-completion-utility-u2-protocol-v1','binding':b,
        'endpoint_names':names,'fold_sha256':digest(paired.folds.tolist()),
        'readout_parameters':kernel.readout.PARAMETERS,
        'bootstrap_draws':1000,'bootstrap_seed':99221,'minimum_valid_draws':900,
        'primary_contrasts':kernel.metrics.CONTRASTS,'network_training_steps':0,
        'historical_gates_changed':False,'protected_external_data_used':False,
        'patient_level_output_permitted':False}
    # Historical pure kernels use tuples; durable JSON comparisons use lists.
    return json.loads(json.dumps(protocol,sort_keys=True,allow_nan=False))


def provider(b):
    import run_bran_robust_clinical_r7 as r7
    manifest=b['R7_binding']['checkpoint_manifest']
    require(manifest['component_role']=='R' and len(manifest['folds'])==5)
    _,private=r7.paths('fit',1)
    def load(fold):
        require(type(fold) is int and fold in range(5))
        item=manifest['folds'][fold]
        require(item['fold']==fold)
        return r7.load_checkpoint(private/f'fold{fold}_R.pt',item['checkpoint_sha256'],item['binding'])
    return load


def terminal(p,a):
    return {'status':'U2_completed','protocol_sha256':sha(OUT/'protocol.json'),
            'aggregate_sha256':sha(OUT/'aggregate.json'),'patient_level_output_emitted':False}


def run(replay=False):
    import torch
    torch.set_num_threads(2)
    with LOCK.open('a+b') as lock,quiet():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not OUT.is_symlink())
        if replay:
            require(OUT.is_dir() and {x.name for x in OUT.iterdir()}==
                    {'protocol.json','aggregate.json','completed.json','progress.json'} and not REPLAY.exists())
        else:
            require(not OUT.exists() and not REPLAY.exists())
            OUT.mkdir(mode=0o700)
        try:
            progress('authentication')
            b=binding()
            progress('source_loading')
            source=parent.sources(b['R7_binding']);paired=source.paired
            p=make_protocol(b,paired)
            if replay:require(parent.read(OUT/'protocol.json')==p)
            else:write_json(OUT/'protocol.json',p)
            a=kernel.evaluate(paired,provider(b),progress)
            kernel.validate_aggregate(a,p['endpoint_names'])
            progress('post_authentication')
            require(binding()==b and parent.sources(b['R7_binding']).receipt()==source.receipt()
                    and make_protocol(b,paired)==p)
            if replay:require(parent.read(OUT/'aggregate.json')==a)
            else:write_json(OUT/'aggregate.json',a)
            t=terminal(p,a)
            if replay:
                require(parent.read(OUT/'completed.json')==t)
                write_json(REPLAY,{**t,'status':'U2_independent_full_replay_passed'})
                progress('independent_replay')
            else:
                write_json(OUT/'completed.json',t)
                progress('completed')
        except BaseException as exc:
            safe=[{'file':Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(exc.__traceback__)
                  if Path(x.filename).name in set(CODE)|set(parent.CODE)]
            write_json(OUT/('replay_failure.json' if replay else 'failure.json'),
                {'status':'closed_U2_failure','own_code_locations':safe,'patient_level_output_emitted':False})
            raise
    return {'status':'U2_replayed' if replay else 'U2_completed','patient_level_output_emitted':False}


def authenticate():
    require(OUT.is_dir() and not OUT.is_symlink() and {p.name for p in OUT.iterdir()}==
            {'protocol.json','aggregate.json','completed.json','progress.json'})
    p=parent.read(OUT/'protocol.json');a=parent.read(OUT/'aggregate.json')
    kernel.validate_aggregate(a,p['endpoint_names'])
    require(binding()==p['binding'])
    t=terminal(p,a)
    require(parent.read(OUT/'completed.json')==t and
            parent.read(REPLAY)=={**t,'status':'U2_independent_full_replay_passed'})
    require(parent.read(OUT/'progress.json')['phase']=='independent_replay')
    return a,{**t,'replay_sha256':sha(REPLAY)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true')
    parser.add_argument('--authenticate',action='store_true');args=parser.parse_args()
    try:
        with quiet():result=authenticate()[1] if args.authenticate else run(args.replay)
        print(json.dumps(result,sort_keys=True))
    except BaseException:
        print(json.dumps({'status':'closed_U2_failure','patient_level_output_emitted':False}))
        raise SystemExit(1) from None
