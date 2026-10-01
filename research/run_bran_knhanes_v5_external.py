"""V5 phase-9 external Hb run: admitted local source, quiet execution, no rows out."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import run_bran_v5_cbc_uncertainty as model_source
import run_bran_context_preservation_v5 as v5
from bran_multisource_batches_v2 import transform_hash
from bran_knhanes_input_kernel_v1 import CANONICAL_NAMES, ADMITTED_CANONICAL_INDICES
from bran_knhanes_phase9_adapter import prepare
import build_bran_knhanes_phase9_binding_v1 as phase9
from bran_knhanes_v5_inference import infer
import bran_knhanes_v5_evaluation as prediction
import bran_knhanes_v5_metrics as metrics
from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json

ROOT=Path(__file__).resolve().parent
PLAN=ROOT/'BRAN_KNHANES_V5_EXTERNAL_PLAN_2026-09-20.md'
SOURCE_BINDING_SHA256='314e33a8149c3e29cf4cade7510d2b19bb29ea605f00c64213cd3a4108f71db8'
CODE=tuple(sorted(set(model_source.CODE)|{
    'run_bran_knhanes_v5_external.py','test_run_bran_knhanes_v5_external.py',
    'test_bran_knhanes_v5_integration.py',
    'BRAN_KNHANES_V5_EXTERNAL_PLAN_2026-09-20.md',
    'bran_knhanes_phase9_source.py','test_bran_knhanes_phase9_source.py',
    'bran_knhanes_phase9_adapter.py','test_bran_knhanes_phase9_adapter.py',
    'bran_knhanes_grouped_folds_v1.py','test_bran_knhanes_grouped_folds_v1.py',
    'bran_knhanes_input_kernel_v1.py','test_bran_knhanes_input_kernel_v1.py',
    'bran_knhanes_target_v1.py','test_bran_knhanes_target_v1.py',
    'bran_knhanes_survey_v1.py','test_bran_knhanes_survey_v1.py',
    'bran_knhanes_v5_inference.py','test_bran_knhanes_v5_inference.py',
    'bran_knhanes_v5_evaluation.py','test_bran_knhanes_v5_evaluation.py',
    'bran_knhanes_v5_metrics.py','test_bran_knhanes_v5_metrics.py',
    'build_bran_knhanes_phase9_binding_v1.py','audit_knhanes_metadata_v1.py',
    'bran_v5_state_routes.py','bran_v5_residual_training.py'}))
PHASES=('source_admission','protocol_freeze','local_source_read','native_inference',
        'fixed_readout_fit','aggregate_metrics','checkpoint_replay','post_authentication','completed')
FLAGS={'patient_level_output_emitted':False,'encoder_fitted':False,'model_promoted':False,
       'individual_prediction_intervals_validated':False,'clinical_safety_established':False,
       'paired_multimodal_external_validation':False,'source_is_external_adaptation':True,
       'native_head_unchanged_direct_transport':True,'single_frozen_v5_fold0_frame':True}
PARAMETERS={'years':[2022,2023],'checkpoint_role':'M','checkpoint_fold':0,
    'routes':'clinical_only_whole_cbc_hidden','arms':list(prediction.ARMS),
    'primary':'survey_weighted_paired_state_minus_raw_mae_g_dl',
    'folds':5,'group':'year_namespaced_PSU_household_contained',
    'tree':{'n_estimators':256,'min_samples_leaf':5,'max_features':1.,'bootstrap':False,
            'n_jobs':1,'seed_base':97201,'sample_weighted_fit':False},
    'normalizers':'frozen_original_V5_fold0','weight':'wt_itvex_divided_by_2',
    'interval':'stratified_PSU_Taylor_domain_Student_t_fixed_prediction',
    'model_selection':False,'raw_source_rows_persisted':False}


def require(ok):
    if not ok:raise ValueError('knhanes_v5_external_contract_failed')


def paths(attempt):
    require(type(attempt)is int and 1<=attempt<=99)
    return ROOT/f'BRAN_KNHANES_V5_EXTERNAL_ATTEMPT{attempt}'


def code_hashes():return {name:sha(ROOT/name) for name in CODE}


def source_binding(admitted):
    r=admitted.receipt
    require(r['binding_file_sha256']==SOURCE_BINDING_SHA256
        and sha(ROOT/'BRAN_KNHANES_PHASE9_BINDING_V1/binding.json')==SOURCE_BINDING_SHA256
        and r['metadata_sha256']==phase9.META_PIN and r['guide_sha256']==phase9.GUIDE_PIN
        and r['model_units_sha256']==phase9.UNITS_PIN
        and admitted.source_root.resolve()==phase9.SOURCE.resolve())
    # MappingProxyType and tuples are internal immutability mechanisms, not
    # JSON artifact types. Canonicalize once so reload equality is meaningful.
    return json.loads(json.dumps(dict(r),sort_keys=True,allow_nan=False))


def model_binding():
    base,_,fit,components=model_source.small_authentication()
    component=components[('M',0)];b=component['binding']
    require(b['fold']==0 and b['role']=='M'
            and b['clinical_field_order_sha256']==v5.digest(CANONICAL_NAMES))
    return {'baseline_sha256':sha(model_source.BASELINE),'fit_aggregate_sha256':base['fit_aggregate_sha256'],
        'component_sha256':fit['component_sha256']['fold0_M.json'],
        'checkpoint_sha256':component['checkpoint_sha256'],'binding':b}


def provider(binding):
    _,private=v5.paths('fit',2)
    model,transform=v5.oldfit.load_checkpoint(private/'fold0_M.pt',binding['checkpoint_sha256'],binding['binding'])
    require(transform_hash(transform)==binding['binding']['transform_sha256'])
    return model,transform


def make_protocol(source_receipt,model_receipt):
    return {'schema':'bran-knhanes-v5-external-protocol-v1','status':'frozen_before_source_rows',
        'parameters':PARAMETERS,'source':source_receipt,'model':model_receipt,
        'code_sha256':code_hashes(),'privacy':FLAGS}


def validate_protocol(p):
    require(type(p)is dict and set(p)=={'schema','status','parameters','source','model','code_sha256','privacy'}
        and p['schema']=='bran-knhanes-v5-external-protocol-v1' and p['status']=='frozen_before_source_rows'
        and p['parameters']==PARAMETERS and p['privacy']==FLAGS and p['code_sha256']==code_hashes())


def validate_result(a,pin):
    require(type(a)is dict and set(a)=={'schema','status','protocol_sha256','results',*FLAGS}
        and a['schema']=='bran-knhanes-v5-external-aggregate-v1' and a['status']=='completed'
        and a['protocol_sha256']==pin and len(pin)==64 and all(a[k]is v for k,v in FLAGS.items()))
    metrics.validate_result(a['results'])


def phase(out,state,name):
    require(name in PHASES);state['phase']=name
    temp=out/'progress.next.json'
    write_json(temp,{'phase':name,'patient_level_output_emitted':False,'encoder_fitted':False})
    os.replace(temp,out/'progress.json')


def same_inference(a,b):
    return (np.array_equal(a.states,b.states,equal_nan=True)
        and np.array_equal(a.native_hemoglobin,b.native_hemoglobin,equal_nan=True)
        and np.array_equal(a.available,b.available) and dict(a.provider_flags)==dict(b.provider_flags))


def execute(attempt,admission_path,state):
    # No output attempt or SAS decode is created merely because authority is absent.
    import bran_knhanes_phase9_source as source
    admitted=source.authenticate_sources(admission_path)
    receipt=source_binding(admitted);m=model_binding();out=paths(attempt)
    require(not out.exists() and not out.is_symlink());out.mkdir();state['owned']=out
    phase(out,state,'protocol_freeze');p=make_protocol(receipt,m)
    validate_protocol(p);write_json(out/'protocol.json',p);pin=sha(out/'protocol.json')
    phase(out,state,'local_source_read')
    data=prepare(source.load_tables(admitted,fd_quiet=True,exclusive_lock_held=True),admitted.binding)
    phase(out,state,'native_inference');model,transform=provider(m)
    x=infer(data,model,transform,m['binding']['transform_sha256'])
    phase(out,state,'fixed_readout_fit')
    fitted=prediction.evaluate(data.grouped_folds,x.states,data.inputs.clinical_values,
        data.inputs.clinical_mask,data.inputs.ages,data.hemoglobin,data.target_masks.eligible,
        x.available,x.native_hemoglobin,CANONICAL_NAMES,tuple(sorted(ADMITTED_CANONICAL_INDICES)),x.provider_flags)
    phase(out,state,'aggregate_metrics')
    r=metrics.summarize(fitted,data.hemoglobin,fitted.support,data.design_weights,data.years,
        data.psu_groups,data.stratum_groups,data.folds,data.target_masks.eligible,data.target_masks.low)
    metrics.validate_result(r)
    phase(out,state,'checkpoint_replay');reloaded,reloaded_transform=provider(m)
    replay=infer(data,reloaded,reloaded_transform,m['binding']['transform_sha256'])
    require(same_inference(x,replay))
    phase(out,state,'post_authentication')
    require(source_binding(source.authenticate_sources(admission_path))==receipt and model_binding()==m)
    require(make_protocol(receipt,m)==p and sha(out/'protocol.json')==pin)
    a={'schema':'bran-knhanes-v5-external-aggregate-v1','status':'completed',
       'protocol_sha256':pin,'results':r,**FLAGS}
    validate_result(a,pin);write_json(out/'aggregate.json',a)
    manifest={'protocol_sha256':pin,'aggregate_sha256':sha(out/'aggregate.json'),
        'checkpoint_sha256':m['checkpoint_sha256'],'source_receipt_sha256':v5.digest(receipt),
        'patient_level_output_emitted':False}
    write_json(out/'manifest.json',manifest)
    require(not (out/'failure.json').exists())
    write_json(out/'completed.json',{'status':'authenticated','protocol_sha256':pin,
        'aggregate_sha256':manifest['aggregate_sha256'],'checkpoint_reload_exact':True,
        'source_bytes_postauthenticated':True,'code_postauthenticated':True,
        'patient_level_output_emitted':False,'model_promoted':False})
    phase(out,state,'completed');state['owned']=None


def safe_site(exc):
    # Code location only; no exception message, local variables or data reprs.
    found=None;tb=exc.__traceback__
    while tb:
        path=Path(tb.tb_frame.f_code.co_filename)
        if path.parent==ROOT and path.name in CODE:found={'module':path.name,'line':tb.tb_lineno}
        tb=tb.tb_next
    return found


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,required=True)
    parser.add_argument('--admission',type=Path,required=True);args=parser.parse_args()
    state={'owned':None,'phase':'source_admission'};ok=False
    with quiet():
        try:
            with LOCK.open('a') as lock,threadpool_limits(limits=1):
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                execute(args.attempt,args.admission,state);ok=True
        except Exception as exc:
            if state['owned'] is not None and not (state['owned']/'completed.json').exists():
                write_json(state['owned']/'failure.json',{'status':'technical_failure','phase':state['phase'],
                    'code_site':safe_site(exc),'patient_level_output_emitted':False,'model_promoted':False})
    print(json.dumps({'status':'completed' if ok else 'not_completed','phase':state['phase'],
        'patient_level_output_emitted':False,'model_promoted':False}))
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
