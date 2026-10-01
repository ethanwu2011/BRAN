"""Local-only authenticated discovery/stability stage; never exports patient states."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import numpy as np

import bran_disease_structure_feasibility_v1 as prior
import bran_disease_structure_evaluation_v1 as evaluator

ROOT = Path(__file__).resolve().parent
SCHEMA = 'bran-disease-structure-discovery-v1'
PROTOCOL = 'BRAN_DISEASE_STRUCTURE_DISCOVERY_PROTOCOL_V1.json'
OUTDIR = 'validation_results/BRAN_DISEASE_STRUCTURE_DISCOVERY_V1'
PATHS = {k: OUTDIR+'/'+v for k,v in {'success':'SUCCESS.json','failure':'FAILURE.json','progress':'progress.json','lock':'run.lock'}.items()}
PRIOR_PROTOCOL_SHA = '6c57d50c582bc85d3b86b38161c379c09f1b9055440d34e670bc8935f9407002'
PRIOR_SUCCESS_SHA = 'd9bc8e50d1c727a0a0b1234a7c459d96dd87ec06434c8637966295ea385e2e36'
PRIOR_COMPONENTS = 113
NEW_FILES = {prior.PROTOCOL, 'BRAN_DISEASE_STRUCTURE_DISCOVERY_DESIGN_V1.md',
    'bran_disease_structure_192_v1.py', 'test_bran_disease_structure_192_v1.py',
    'bran_disease_structure_diagnostics_v1.py', 'test_bran_disease_structure_diagnostics_v1.py',
    'bran_disease_structure_evaluation_v1.py', 'test_bran_disease_structure_evaluation_v1.py',
    'bran_disease_structure_experiment_v1.py', 'test_bran_disease_structure_experiment_v1.py'}
PARAMETERS = {'discovery_folds':[0,1,2], 'validation_fold':3, 'replication_fold':4,
    'encoder_steps':1500, 'encoder_seed':1701, 'encoder_fit':'all_discovery_patients',
    'disease':'observed_mhterm_dm2_equals_1', 'state_width':192, 'diagnosis_history_inputs':False,
    'pca_components':8, 'pca_whiten':True, 'pca_solver':'full', 'mixture_k':[1,2,3,4],
    'mixture_covariance':'diag', 'mixture_regularization':0.0001, 'mixture_initializations':5,
    'mixture_iterations':500, 'structure_seed':93501, 'selection':'smallest_k_within_one_se_of_best_validation_density',
    'minimum_discovery':80, 'minimum_validation':40, 'minimum_replication':40, 'minimum_group':20,
    'bootstrap_count':50, 'minimum_bootstrap_converged':45, 'bootstrap_refits':'scaler_pca_locked_k_not_encoder',
    'bootstrap_quantiles':[0.025,0.5,0.975], 'independent_characterization':False,
    'inference':'descriptive_fixed_encoder_internal_reused_development_no_promotion'}
PRIVACY = {'patient_processing_local_only':True, 'patient_rows_ids_states_assignments_weights_draws_serialized':False,
    'group_counts_serialized':False, 'clinical_characterization_values_loaded':False,
    'clinical_subtype_or_external_validation_claimed':False, 'default_promotion':False}
PHASES = {'protocol','validated','context','training','structure','writing','completed'}
require=prior.require; exact_keys=prior.exact_keys; sha=prior.sha; write_x=prior.write_x

def runtime():
    return {'python':platform.python_version(), **{name:version(name) for name in ('numpy','scipy','scikit-learn','torch')}}

def _prior(root):
    require(sha(root/prior.PROTOCOL)==PRIOR_PROTOCOL_SHA and sha(root/prior.PATHS['success'])==PRIOR_SUCCESS_SHA)
    p=prior.validate_protocol(root); a=prior.audit(root)
    require(len(p['expected_hashes'])==PRIOR_COMPONENTS and a['status']=='authenticated_success' and a['artifact_sha256']==PRIOR_SUCCESS_SHA)
    require(all(v is True for mod in a['field_support'].values() for field in mod.values() for v in field.values()))
    return p

def _names(p):
    names=set(p['expected_hashes']); require(len(names)==PRIOR_COMPONENTS and not names&NEW_FILES)
    return names|NEW_FILES

def freeze(root=ROOT):
    root=Path(root); require(not(root/PROTOCOL).exists() and not any((root/v).exists() for v in PATHS.values()))
    old=_prior(root); names=_names(old)
    p={'schema':SCHEMA,'status':'frozen_before_execution','created_utc':datetime.now(timezone.utc).isoformat(),
       'paths':PATHS,'parameters':PARAMETERS,'privacy':PRIVACY,'authentication':old['authentication'],
       'data_roots':old['data_roots'],'manifest_hashes':old['manifest_hashes'],'runtime':runtime(),
       'prior_protocol_sha256':PRIOR_PROTOCOL_SHA,'prior_success_sha256':PRIOR_SUCCESS_SHA,
       'expected_hashes':{n:sha(root/n) for n in sorted(names)}}
    write_x(root/PROTOCOL,p); validate_protocol(root)
    return {'status':'frozen','protocol_sha256':sha(root/PROTOCOL),'components':len(names)}

def validate_protocol(root=ROOT):
    root=Path(root); old=_prior(root); p=json.loads((root/PROTOCOL).read_text())
    exact_keys(p,{'schema','status','created_utc','paths','parameters','privacy','authentication','data_roots','manifest_hashes','runtime','prior_protocol_sha256','prior_success_sha256','expected_hashes'})
    require(p['schema']==SCHEMA and p['status']=='frozen_before_execution' and p['paths']==PATHS and p['parameters']==PARAMETERS and p['privacy']==PRIVACY and p['runtime']==runtime())
    require(p['prior_protocol_sha256']==PRIOR_PROTOCOL_SHA and p['prior_success_sha256']==PRIOR_SUCCESS_SHA)
    require(all(p[k]==old[k] for k in ('authentication','data_roots','manifest_hashes')))
    exact_keys(p['expected_hashes'],_names(old))
    for name,digest in p['expected_hashes'].items():
        path=(root/name).resolve(); require(path.parent==root.resolve() and type(digest)is str and len(digest)==64 and sha(path)==digest)
    return p

def _progress(path,phase):
    require(phase in PHASES)
    path.write_text(json.dumps({'status':'completed' if phase=='completed' else 'running','phase':phase})+'\n')

def _context(root,p):
    from patient_atlas_v6_2_expanded_endpoint_evaluation import EXACT_INNER_FOLD_ASSIGNMENT_SHA256,EXACT_OUTER_FOLD_HASH,FROZEN_SUPPORT_RECEIPT_NAME,load_eligible_support_receipt,validate_support_against_observed
    from run_patient_atlas_v6_2_expanded_endpoint_evaluation import _load_actual_v6_2_context
    support=load_eligible_support_receipt(root/FROZEN_SUPPORT_RECEIPT_NAME,project_root=root)
    context=_load_actual_v6_2_context(root=root,support=support,**{k:Path(v) for k,v in p['data_roots'].items()})
    outer=np.asarray(context['outer_assignment']); a=p['authentication']
    require(outer.shape==(1928,) and outer.dtype.kind in 'iu' and set(outer.tolist())==set(range(5)))
    require(len(support.eligible_sources)==26 and dict(context['source_hashes'])==a['canonical_source_hashes'] and support.receipt_sha256==a['support_receipt_sha256'] and a['outer_fold_sha256']==EXACT_OUTER_FOLD_HASH)
    validate_support_against_observed(support,context['labels_by_source'],context['observed_by_source'],outer)
    for f in range(5):
        _,digest=prior.coverage.original.prior.base.paired.base._inner_context(context,np.flatnonzero(outer!=f),f)
        require(digest==a['inner_fold_sha256'][f]==EXACT_INNER_FOLD_ASSIGNMENT_SHA256[f])
    ids=prior.coverage.original._local_canonical_ids(context['raw_cohort'].patient_ids)
    observed=context['observed_by_source']['mhterm_dm2']; labels=context['labels_by_source']['mhterm_dm2']
    require(isinstance(observed,np.ndarray) and observed.dtype==np.dtype(bool) and observed.shape==outer.shape)
    require(isinstance(labels,np.ndarray) and labels.shape==outer.shape and labels.dtype.kind in 'biuf' and np.all(np.isfinite(labels[observed])) and np.all((labels[observed]==0)|(labels[observed]==1)))
    member=observed&(labels==1); require(int(member.sum())==719)
    require([int(np.sum(member&(outer==f))) for f in range(5)]==[151,137,139,141,151])
    return context,outer,ids,member

def validate_report(r,p,digest):
    if not isinstance(r,dict) or set(r)!={'schema','status','protocol_sha256','code_hashes','source_hashes','support_receipt_sha256','fold_hashes','manifest_hashes','runtime','parameters','privacy','scope','result'}: return False
    a=p['authentication']
    if not(r['schema']==SCHEMA and r['status']=='completed_structure_diagnostics' and r['protocol_sha256']==digest and r['code_hashes']==p['expected_hashes'] and r['source_hashes']==a['canonical_source_hashes'] and r['support_receipt_sha256']==a['support_receipt_sha256'] and r['fold_hashes']=={'outer':a['outer_fold_sha256'],'inner':a['inner_fold_sha256']} and r['manifest_hashes']==p['manifest_hashes'] and r['runtime']==p['runtime'] and r['parameters']==PARAMETERS and r['privacy']==PRIVACY): return False
    if r['scope']!={'patient_count':1928,'disease_patient_count':719,'source_count':26,'official_test_loaded':False}: return False
    return evaluator.validate_result(r['result'])

def run(root=ROOT):
    root=Path(root); p={}; fd=None; quiet=None; phase='protocol'; paths={k:root/v for k,v in PATHS.items()}
    try:
        p=validate_protocol(root); require(not any(paths[k].exists() for k in ('success','failure','progress')))
        base=prior.coverage.original.prior.base.paired.base
        fd=base._acquire_lock(paths['lock']); _progress(paths['progress'],'validated')
        quiet=base._quiet_sensitive_block(); quiet.__enter__(); phase='context'
        context,outer,ids,member=_context(root,p)
        import run_bran_anchor_ablation_v2 as paired
        c,cm,eligible,r,rm,names=paired._actual_arrays(root,context); eligible[:,48:]=False
        def progress(value):
            nonlocal phase
            require(value in {'training','structure'}); phase=value; _progress(paths['progress'],phase)
        result=evaluator.evaluate(c,cm,eligible,r,rm,names,np.asarray(context['raw_cohort'].ages),outer,ids,member,progress=progress)
        a=p['authentication']; report={'schema':SCHEMA,'status':'completed_structure_diagnostics','protocol_sha256':sha(root/PROTOCOL),
            'code_hashes':p['expected_hashes'],'source_hashes':a['canonical_source_hashes'],'support_receipt_sha256':a['support_receipt_sha256'],
            'fold_hashes':{'outer':a['outer_fold_sha256'],'inner':a['inner_fold_sha256']},'manifest_hashes':p['manifest_hashes'],
            'runtime':p['runtime'],'parameters':PARAMETERS,'privacy':PRIVACY,
            'scope':{'patient_count':1928,'disease_patient_count':719,'source_count':26,'official_test_loaded':False},'result':result}
        phase='writing'; p=validate_protocol(root); require(validate_report(report,p,sha(root/PROTOCOL))); write_x(paths['success'],report)
        _progress(paths['progress'],'completed'); return {'status':'completed_structure_diagnostics','artifact_sha256':sha(paths['success'])}
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
    root=Path(root); p=validate_protocol(root); paths={k:root/v for k,v in PATHS.items()}
    require(not(paths['success'].exists() and paths['failure'].exists()))
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
