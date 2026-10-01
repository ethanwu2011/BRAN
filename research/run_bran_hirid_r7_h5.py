"""Quiet, frozen H5 native R7-versus-V5 HiRID transport; no fitting."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
from bran_clinical_dictionary_binding_v1 import _quiet
import run_bran_hirid_h3 as h3
import bran_hirid_source_h3 as source

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'BRAN_HIRID_R7_H5_DESIGN.md'
H3_PROTOCOL = 'cab732d6466dfa041cce707d8318437c724d01828ed77f9b7fd595a2cdaabbbd'
H3_AGGREGATE = '352a39236086507a39797f5f0759428ee512f5b88c180af3f97f93c59832d52c'
H3_REPORT = '1377723c9e801f24937ed15dbf82c2495983b3c9b09668d4393d853cca6c5eff'
R7_CHECKPOINT = '82d9cc71794dc116811214b647f396e4a43525f3094c7c0ca04b59e36ce0c08f'
CODE = ('BRAN_HIRID_R7_H5_DESIGN.md', 'run_bran_hirid_r7_h5.py',
        'bran_hirid_r7_h5.py', 'test_bran_hirid_r7_h5.py',
        'test_run_bran_hirid_r7_h5.py', 'run_bran_hirid_r7_h5_attempt1.sh',
        'audit_bran_robust_clinical_r7_v2.py', 'run_bran_r7_fixed_state_p1.py')
PHASES = ('authentication', 'source_authentication', 'source_selection',
          'manifest_preflight', 'native_inference', 'checkpoint_replay',
          'aggregate_bootstrap', 'historical_v5_replay', 'aggregate_replay',
          'post_authentication', 'completed')
ERROR = 'bran_hirid_r7_h5_execution_failed'


def require(ok):
    if not ok:
        raise ValueError(ERROR) from None


def sha(path):
    return source._hash_file(path)


def paths(attempt=1):
    require(type(attempt) is int and 1 <= attempt <= 99)
    return ROOT / f'BRAN_HIRID_R7_H5_ATTEMPT{attempt}'


def authenticate():
    import audit_bran_robust_clinical_r7_v2 as r7audit
    import run_bran_r7_fixed_state_p1 as p1
    parent = h3.paths(2)
    require(sha(parent/'protocol.json') == H3_PROTOCOL and sha(parent/'aggregate.json') == H3_AGGREGATE
            and not (parent/'failed.json').exists())
    prior = h3.read(parent/'protocol.json')
    done = h3.read(parent/'completed.json')
    require(done['status'] == 'completed' and done['protocol_sha256'] == H3_PROTOCOL
            and done['aggregate_sha256'] == H3_AGGREGATE and done['patient_level_output_emitted'] is False)
    h3.validate_aggregate(h3.read(parent/'aggregate.json'))
    report = ROOT/'BRAN_HIRID_H3_REPORT_ATTEMPT2_2026-09-23'/'manifest.json'
    require(sha(report) == H3_REPORT)
    audit = h3.read(report)['audit']
    require(audit['independent_source_inference_aggregate_replay'] is True
            and audit['protocol_sha256'] == H3_PROTOCOL and audit['aggregate_sha256'] == H3_AGGREGATE
            and audit['patient_level_output_emitted'] is False)
    v5 = h3.authenticate_model()
    require(v5 == prior['launch']['model'] and h3.approval() == prior['launch']['approval_sha256'])
    fit, _, components, fit_receipt = r7audit.authenticate('fit', 1)
    p1_receipt = p1.authenticate(1)
    pp = p1._read(p1.paths(1)/'protocol.json')
    entry = pp['r7_checkpoint_manifest']['folds'][0]
    component = components[('R', 0)]
    require(entry['fold'] == 0 and entry['checkpoint_sha256'] == R7_CHECKPOINT
            and entry['binding'] == component['binding']
            and entry['checkpoint_sha256'] == component['checkpoint_sha256']
            and pp['r7_fit_receipt'] == fit_receipt
            and fit['source_binding']['protected_sources_used'] is False
            and fit['source_binding']['source_roles']['hirid']['disposition'] != 'training'
            and entry['binding']['initial_checkpoint_sha256'] == v5['checkpoint_sha256']
            and entry['binding']['transform_sha256'] == v5['binding']['transform_sha256'])
    names = set(CODE) | set(v5['code_sha256']) | set(pp['code_sha256']) | set(fit['code_sha256'])
    for name, pin in {**v5['code_sha256'], **pp['code_sha256'], **fit['code_sha256']}.items():
        require(sha(ROOT/name) == pin)
    return {'parent_protocol_sha256': H3_PROTOCOL, 'parent_aggregate_sha256': H3_AGGREGATE,
            'parent_report_manifest_sha256': H3_REPORT, 'V5': v5, 'R7': entry,
            'r7_fit_receipt': fit_receipt, 'p1_receipt': p1_receipt,
            'approval_sha256': h3.approval(), 'plan_sha256': sha(PLAN),
            'code_sha256': {name: sha(ROOT/name) for name in sorted(names)},
            'patient_level_output_emitted': False}, prior['source']


def infer_r7(selections, binding):
    import torch
    import run_bran_robust_clinical_r7 as r7
    from bran_robust_clinical_r7 import BRANRobustClinicalR7
    from bran_hirid_v5_input_adapter_v1 import prepare_inputs
    from bran_hirid_v5_inference_v1 import infer
    from bran_knhanes_input_kernel_v1 import CANONICAL_INDEX
    torch.set_num_threads(2)
    _, private = r7.paths('fit', 1)
    item = binding['R7']
    model, transform = r7.load_checkpoint(private/'fold0_R.pt', item['checkpoint_sha256'], item['binding'])
    require(type(model) is BRANRobustClinicalR7)
    result = infer(prepare_inputs(selections), model, transform, item['binding']['transform_sha256'])
    median = float(transform.clinical_median[CANONICAL_INDEX['hemoglobin']])
    require(np.isfinite(median) and median > 0)
    return result, median


def calculate(selections, binding, callback):
    import bran_hirid_r7_h5 as metrics
    callback('native_inference')
    v5, median = h3.model_inference(selections, binding['V5'])
    r7, r7median = infer_r7(selections, binding)
    require(median == r7median)
    callback('checkpoint_replay')
    v5b, medianb = h3.model_inference(selections, binding['V5'])
    r7b, r7medianb = infer_r7(selections, binding)
    require(median == medianb == r7medianb)
    for one, two in ((v5,v5b),(r7,r7b)):
        require(np.array_equal(one.available,two.available)
                and np.array_equal(one.native_hemoglobin,two.native_hemoglobin,equal_nan=True))
    callback('aggregate_bootstrap')
    result = metrics.summarize(selections, r7, v5, median)
    metrics.validate_result(result)
    callback('historical_v5_replay')
    require(result['V5'] == h3.read(h3.paths(2)/'aggregate.json')['report'])
    callback('aggregate_replay')
    require(result == metrics.summarize(selections, r7b, v5b, medianb))
    return result


def progress(out, state, phase, n=0, total=0):
    require(phase in PHASES and type(n) is int and type(total) is int and 0 <= n <= total)
    state['phase'] = phase
    item = {'phase': phase, 'partitions_completed': n, 'partitions_total': total,
            'elapsed_seconds': round(time.monotonic()-state['start'],1), 'patient_level_output_emitted':False}
    h3.write(out/'progress.next.json',item)
    os.replace(out/'progress.next.json',out/'progress.json')


def validate_aggregate(value):
    import bran_hirid_r7_h5 as metrics
    require(type(value) is dict and set(value) == {'schema','protocol_sha256','report',
        'checkpoint_replay_equal','aggregate_replay_equal','historical_v5_replay_equal',
        'encoder_updated','external_model_selection','patient_level_output_emitted'})
    require(value['schema']=='bran-hirid-r7-h5-aggregate-v1'
            and type(value['protocol_sha256']) is str and len(value['protocol_sha256'])==64
            and set(value['protocol_sha256']) <= set('0123456789abcdef'))
    for key in ('checkpoint_replay_equal','aggregate_replay_equal','historical_v5_replay_equal'):
        require(value[key] is True)
    for key in ('encoder_updated','external_model_selection','patient_level_output_emitted'):
        require(value[key] is False)
    metrics.validate_result(value['report'])


def run(attempt=1):
    out=paths(attempt)
    require(not out.exists() and not out.is_symlink())
    state={'phase':'authentication','start':time.monotonic()}
    with h3.LOCK.open('a+b') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return {'status':'not_started_shared_lock_busy','patient_level_output_emitted':False}
        out.mkdir(mode=0o700)
        try:
            with _quiet():
                cb=lambda phase,n=0,total=0:progress(out,state,phase,n,total)
                cb('authentication')
                binding, receipt=authenticate()
                protocol={'schema':'bran-hirid-r7-h5-protocol-v1','binding':binding,'source':receipt,
                          'frozen_before_source_selection_and_inference':True,'patient_level_output_emitted':False}
                h3.write(out/'protocol.json',protocol);pin=sha(out/'protocol.json')
                cb('source_authentication');source.recheck_source(receipt)
                cb('source_selection')
                selections=h3.select_source(receipt,binding['approval_sha256'],cb)
                result=calculate(selections,binding,cb)
                cb('post_authentication')
                require(authenticate()==(binding,receipt) and sha(out/'protocol.json')==pin)
                source.recheck_source(receipt)
                aggregate={'schema':'bran-hirid-r7-h5-aggregate-v1','protocol_sha256':pin,'report':result,
                           'checkpoint_replay_equal':True,'aggregate_replay_equal':True,'historical_v5_replay_equal':True,
                           'encoder_updated':False,'external_model_selection':False,'patient_level_output_emitted':False}
                validate_aggregate(aggregate);h3.write(out/'aggregate.json',aggregate)
                cb('completed')
                h3.write(out/'completed.json',{'status':'authenticated_completed','protocol_sha256':pin,
                    'aggregate_sha256':sha(out/'aggregate.json'),'patient_level_output_emitted':False})
            return {'status':'completed_pending_separate_audit','patient_level_output_emitted':False}
        except BaseException:
            require(not (out/'completed.json').exists())
            h3.write(out/'failure.json',{'status':'technical_failure','phase':state['phase'],
                                       'error_code':ERROR,'patient_level_output_emitted':False})
            return {'status':'technical_failure','phase':state['phase'],'patient_level_output_emitted':False}


def audit(attempt=1, replay=True):
    with h3.LOCK.open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with _quiet():
            out=paths(attempt)
            require(out.is_dir() and not out.is_symlink() and not (out/'failure.json').exists()
                    and {p.name for p in out.iterdir()}=={'protocol.json','progress.json','aggregate.json','completed.json'})
            require(all(p.is_file() and not p.is_symlink() and p.stat().st_nlink==1 for p in out.iterdir()))
            protocol,result,done=(h3.read(out/n) for n in ('protocol.json','aggregate.json','completed.json'))
            pin=sha(out/'protocol.json')
            require(set(protocol)=={'schema','binding','source','frozen_before_source_selection_and_inference',
                                    'patient_level_output_emitted'}
                and done=={'status':'authenticated_completed','protocol_sha256':pin,
                'aggregate_sha256':sha(out/'aggregate.json'),'patient_level_output_emitted':False}
                and result['protocol_sha256']==pin
                and protocol['schema']=='bran-hirid-r7-h5-protocol-v1'
                and protocol['frozen_before_source_selection_and_inference'] is True
                and protocol['patient_level_output_emitted'] is False)
            final_progress=h3.read(out/'progress.json')
            require(set(final_progress)=={'phase','partitions_completed','partitions_total','elapsed_seconds',
                                          'patient_level_output_emitted'}
                    and final_progress['phase']=='completed'
                    and final_progress['partitions_completed']==final_progress['partitions_total']==0
                    and type(final_progress['elapsed_seconds']) in (float,int)
                    and final_progress['elapsed_seconds']>=0
                    and final_progress['patient_level_output_emitted'] is False)
            validate_aggregate(result)
            require(authenticate()==(protocol['binding'],protocol['source']))
            source.recheck_source(protocol['source'])
            if replay:
                selections=h3.select_source(protocol['source'],protocol['binding']['approval_sha256'])
                require(calculate(selections,protocol['binding'],lambda *a:None)==result['report'])
            return {'status':'aggregate_terminal_authenticated','protocol_sha256':pin,
                    'aggregate_sha256':done['aggregate_sha256'],'independent_source_inference_replay':replay,
                    'patient_level_output_emitted':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--audit',action='store_true');parser.add_argument('--terminal-only',action='store_true')
    args=parser.parse_args()
    try:
        require(not args.terminal_only or args.audit)
        with _quiet(): value=audit(args.attempt,not args.terminal_only) if args.audit else run(args.attempt)
        print(json.dumps(value,sort_keys=True))
        return 0 if value['status'] in ('completed_pending_separate_audit','aggregate_terminal_authenticated') else 1
    except BaseException:
        print(json.dumps({'status':'closed_failure','patient_level_output_emitted':False}));return 1


if __name__=='__main__':raise SystemExit(main())
