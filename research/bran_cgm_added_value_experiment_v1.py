"""Authenticated local manifest-CGM added-value study; closed aggregate outputs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
import bran_disease_structure_experiment_v1 as prior
import bran_cgm_target_io_v1 as target_io
import bran_cgm_added_value_evaluation_v1 as evaluator

ROOT=Path(__file__).resolve().parent
SCHEMA='bran-cgm-added-value-v1'
PROTOCOL='BRAN_CGM_ADDED_VALUE_PROTOCOL_V1.json'
OUTDIR='validation_results/BRAN_CGM_ADDED_VALUE_V1'
PATHS={k:OUTDIR+'/'+v for k,v in {'success':'SUCCESS.json','failure':'FAILURE.json','progress':'progress.json','lock':'run.lock'}.items()}
PRIOR_PROTOCOL_SHA='40917b38c7a293d61a83267aa9078cd2eb0dbfabaafff198f8eb21db503a768c'
PRIOR_SUCCESS_SHA='61839b757748a8cf6b4a3f4aab56c8d82bf85b08186a9f70737a39b2d8c7e3c8'
NEW_FILES={prior.PROTOCOL,'BRAN_CGM_ADDED_VALUE_DESIGN_V1.md',
    'bran_cgm_target_io_v1.py','test_bran_cgm_target_io_v1.py',
    'bran_cgm_added_value_kernel_v1.py','test_bran_cgm_added_value_kernel_v1.py',
    'bran_cgm_added_value_evaluation_v1.py','test_bran_cgm_added_value_evaluation_v1.py',
    'bran_cgm_added_value_experiment_v1.py','test_bran_cgm_added_value_experiment_v1.py'}
PARAMETERS=evaluator.PARAMETERS
PRIVACY={'patient_processing_local_only':True,'patient_rows_ids_targets_predictions_states_weights_draws_serialized':False,
    'exact_target_denominators_or_site_labels_serialized':False,'raw_cgm_traces_loaded':False,
    'clinical_subtype_external_validation_or_causal_claimed':False,'default_promotion':False}
PHASES={'protocol','validated','context','manifest','training','heads','writing','completed'}
require=prior.require; exact_keys=prior.exact_keys; write_x=prior.write_x; sha=prior.sha

def _prior(root):
    require(sha(root/prior.PROTOCOL)==PRIOR_PROTOCOL_SHA and sha(root/prior.PATHS['success'])==PRIOR_SUCCESS_SHA)
    p=prior.validate_protocol(root); a=prior.audit(root)
    require(len(p['expected_hashes'])==123 and a['status']=='authenticated_success' and a['artifact_sha256']==PRIOR_SUCCESS_SHA)
    require(a['result']['diagnostics']=={'status':'no_discrete_groups','selected_k':1})
    return p

def _names(p):
    names=set(p['expected_hashes']); require(len(names)==123 and not names&NEW_FILES); return names|NEW_FILES

def freeze(root=ROOT):
    root=Path(root); require(not(root/PROTOCOL).exists() and not any((root/v).exists() for v in PATHS.values()))
    old=_prior(root); names=_names(old)
    p={'schema':SCHEMA,'status':'frozen_before_execution','created_utc':datetime.now(timezone.utc).isoformat(),
       'paths':PATHS,'parameters':PARAMETERS,'target_contract':target_io.TARGET_CONTRACT,'privacy':PRIVACY,
       'authentication':old['authentication'],'data_roots':old['data_roots'],'manifest_hashes':old['manifest_hashes'],
       'runtime':prior.runtime(),'prior_protocol_sha256':PRIOR_PROTOCOL_SHA,'prior_success_sha256':PRIOR_SUCCESS_SHA,
       'expected_hashes':{n:sha(root/n) for n in sorted(names)}}
    write_x(root/PROTOCOL,p); validate_protocol(root)
    return {'status':'frozen','protocol_sha256':sha(root/PROTOCOL),'components':len(names)}

def validate_protocol(root=ROOT):
    root=Path(root); old=_prior(root); p=json.loads((root/PROTOCOL).read_text())
    exact_keys(p,{'schema','status','created_utc','paths','parameters','target_contract','privacy','authentication','data_roots','manifest_hashes','runtime','prior_protocol_sha256','prior_success_sha256','expected_hashes'})
    require(p['schema']==SCHEMA and p['status']=='frozen_before_execution' and p['paths']==PATHS and p['parameters']==PARAMETERS and p['target_contract']==target_io.TARGET_CONTRACT and p['privacy']==PRIVACY and p['runtime']==prior.runtime())
    require(p['prior_protocol_sha256']==PRIOR_PROTOCOL_SHA and p['prior_success_sha256']==PRIOR_SUCCESS_SHA and all(p[k]==old[k] for k in ('authentication','data_roots','manifest_hashes')))
    exact_keys(p['expected_hashes'],_names(old))
    for name,digest in p['expected_hashes'].items():
        path=(root/name).resolve(); require(path.parent==root.resolve() and type(digest)is str and len(digest)==64 and sha(path)==digest)
    return p

def _progress(path,phase):
    require(phase in PHASES); path.write_text(json.dumps({'status':'completed' if phase=='completed' else 'running','phase':phase})+'\n')

def _target_path(p): return Path(p['data_roots']['dataset_root'])/target_io.TARGET_CONTRACT['source']

def validate_report(r,p,digest):
    if not isinstance(r,dict) or set(r)!={'schema','status','protocol_sha256','code_hashes','source_hashes','support_receipt_sha256','fold_hashes','manifest_hashes','runtime','parameters','target_contract','privacy','scope','result'}: return False
    a=p['authentication']
    if not(r['schema']==SCHEMA and r['status']=='completed_cgm_added_value' and r['protocol_sha256']==digest and r['code_hashes']==p['expected_hashes'] and r['source_hashes']==a['canonical_source_hashes'] and r['support_receipt_sha256']==a['support_receipt_sha256'] and r['fold_hashes']=={'outer':a['outer_fold_sha256'],'inner':a['inner_fold_sha256']} and r['manifest_hashes']==p['manifest_hashes'] and r['runtime']==p['runtime'] and r['parameters']==PARAMETERS and r['target_contract']==target_io.TARGET_CONTRACT and r['privacy']==PRIVACY): return False
    if r['scope']!={'canonical_patient_count':1928,'predefined_disease_patient_count':719,'source_count':26,'official_test_loaded':False}: return False
    return evaluator.validate_result(r['result'])

def run(root=ROOT):
    root=Path(root); p={}; fd=None; quiet=None; phase='protocol'; paths={k:root/v for k,v in PATHS.items()}
    try:
        p=validate_protocol(root); require(not any(paths[k].exists() for k in ('success','failure','progress')))
        base=prior.prior.coverage.original.prior.base.paired.base
        fd=base._acquire_lock(paths['lock']); _progress(paths['progress'],'validated'); quiet=base._quiet_sensitive_block(); quiet.__enter__()
        phase='context'; _progress(paths['progress'],phase); context,outer,ids,membership=prior._context(root,p)
        phase='manifest'; _progress(paths['progress'],phase); source=_target_path(p); require(sha(source)==p['manifest_hashes']['cgm'])
        with source.open('r',encoding='utf-8',newline='') as handle: targets=target_io.parse_manifest(handle)
        y,observed=target_io.align(targets,ids); require(sha(source)==p['manifest_hashes']['cgm'])
        import run_bran_anchor_ablation_v2 as paired
        c,cm,eligible,r,rm,names=paired._actual_arrays(root,context); eligible[:,48:]=False
        def progress(value):
            nonlocal phase
            require(value in {'training','heads'}); phase=value; _progress(paths['progress'],phase)
        raw=context['raw_cohort']
        result=evaluator.evaluate(c,cm,eligible,r,rm,names,np.asarray(raw.ages),outer,ids,membership,tuple(raw.site_ids),y,observed,progress=progress)
        a=p['authentication']; report={'schema':SCHEMA,'status':'completed_cgm_added_value','protocol_sha256':sha(root/PROTOCOL),'code_hashes':p['expected_hashes'],
            'source_hashes':a['canonical_source_hashes'],'support_receipt_sha256':a['support_receipt_sha256'],
            'fold_hashes':{'outer':a['outer_fold_sha256'],'inner':a['inner_fold_sha256']},'manifest_hashes':p['manifest_hashes'],'runtime':p['runtime'],
            'parameters':PARAMETERS,'target_contract':target_io.TARGET_CONTRACT,'privacy':PRIVACY,
            'scope':{'canonical_patient_count':1928,'predefined_disease_patient_count':719,'source_count':26,'official_test_loaded':False},'result':result}
        phase='writing'; p=validate_protocol(root); require(sha(source)==p['manifest_hashes']['cgm'] and validate_report(report,p,sha(root/PROTOCOL)))
        write_x(paths['success'],report); _progress(paths['progress'],'completed'); return {'status':'completed_cgm_added_value','artifact_sha256':sha(paths['success'])}
    except Exception as error:
        frames=[{'file':Path(f.filename).name,'line':f.lineno} for f in traceback.extract_tb(error.__traceback__) if Path(f.filename).resolve().parent==root.resolve() and Path(f.filename).name in p.get('expected_hashes',{})][:8]
        failure={'schema':SCHEMA,'status':'failed','phase':phase,'error_class':type(error).__name__,'bound_code_frames':frames,'exception_text_serialized':False,'patient_content_serialized':False}
        if fd is not None and not paths['success'].exists() and not paths['failure'].exists(): write_x(paths['failure'],failure)
        return {'status':'execution_failed','phase':phase,'error_class':type(error).__name__,'contents_emitted':False}
    finally:
        if quiet is not None: quiet.__exit__(None,None,None)
        if fd is not None:
            os.close(fd)
            try: paths['lock'].unlink()
            except FileNotFoundError: pass

def audit(root=ROOT):
    root=Path(root); p=validate_protocol(root); paths={k:root/v for k,v in PATHS.items()}; require(not(paths['success'].exists() and paths['failure'].exists()))
    if paths['failure'].exists():
        f=json.loads(paths['failure'].read_text()); exact_keys(f,{'schema','status','phase','error_class','bound_code_frames','exception_text_serialized','patient_content_serialized'})
        require(f['schema']==SCHEMA and f['status']=='failed' and f['phase'] in PHASES and f['exception_text_serialized'] is False and f['patient_content_serialized'] is False and type(f['error_class'])is str and f['error_class'].isidentifier() and type(f['bound_code_frames'])is list and len(f['bound_code_frames'])<=8)
        for frame in f['bound_code_frames']:
            exact_keys(frame,{'file','line'}); require(frame['file'] in p['expected_hashes'] and type(frame['line'])is int and 0<frame['line']<100000)
        return {'status':'authenticated_execution_failure','artifact_sha256':sha(paths['failure']),'phase':f['phase'],'error_class':f['error_class'],'bound_code_frames':f['bound_code_frames']}
    if paths['success'].exists():
        r=json.loads(paths['success'].read_text()); require(validate_report(r,p,sha(root/PROTOCOL)))
        return {'status':'authenticated_success','artifact_sha256':sha(paths['success']),'protocol_sha256':sha(root/PROTOCOL),'components':len(p['expected_hashes']),'scope':r['scope'],'result':r['result'],'privacy':r['privacy'],'folds_authenticated':True}
    out={'status':'no_terminal_artifact','lock_exists':paths['lock'].exists()}
    if paths['progress'].exists():
        q=json.loads(paths['progress'].read_text()); exact_keys(q,{'status','phase'}); require(q['status'] in {'running','completed'} and q['phase'] in PHASES); out['progress']=q
    return out

if __name__=='__main__':
    try: print(json.dumps({'freeze':freeze,'run':run,'audit':audit}[sys.argv[1]](),sort_keys=True,allow_nan=False))
    except Exception as error:
        print(json.dumps({'status':'blocked_without_disclosure','error_class':type(error).__name__,'contents_emitted':False})); raise SystemExit(1)
