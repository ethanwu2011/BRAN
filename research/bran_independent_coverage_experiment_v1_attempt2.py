"""Attempt2 technical cardinality correction for frozen BRAN coverage preflight.

Only duplicate-identical local finite-presence masks are collapsed.  Cohort, subgroup,
folds, thresholds, source mapping, and aggregate disclosure rules are unchanged.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import traceback
from datetime import datetime, timezone
from typing import Any
import numpy as np

import bran_independent_coverage_experiment_v1 as original
from bran_independent_coverage_io_v1_attempt2 import DUPLICATE_POLICY, parse_manifest
from bran_independent_coverage_v1 import summarize_coverage

ROOT=Path(__file__).resolve().parent
SCHEMA='bran-independent-coverage-experiment-v1-attempt2'
PROTOCOL='BRAN_INDEPENDENT_COVERAGE_PROTOCOL_V1_ATTEMPT2.json'
OUTDIR='validation_results/BRAN_INDEPENDENT_COVERAGE_V1_ATTEMPT2'
PATHS={k:OUTDIR+'/'+v for k,v in {'success':'SUCCESS.json','failure':'FAILURE.json','progress':'progress.json','lock':'run.lock'}.items()}
ORIGINAL_PROTOCOL=original.PROTOCOL
ORIGINAL_PROTOCOL_SHA='797f933b85291dedd42b91c19c82033980304984565183a054d377ecfd12ef0f'
ORIGINAL_FAILURE=original.PATHS['failure']
ORIGINAL_FAILURE_SHA='02b428eb8e663f89f7d32eac4b2bf64f05580b428a70a98c75e4330b3ffba262'
ORIGINAL_COMPONENTS=105
NEW_FILES={'bran_independent_coverage_io_v1_attempt2.py','bran_independent_coverage_experiment_v1_attempt2.py','test_bran_independent_coverage_attempt2.py','diagnose_bran_coverage_duplicates_v1.py',ORIGINAL_PROTOCOL}
PARAMETERS=original.PARAMETERS
PRIVACY=original.PRIVACY
PHASES=original.PHASES
INPUT_CONTRACT=original.INPUT_CONTRACT|{'repeated_id_policy':DUPLICATE_POLICY}
require=original.require
exact_keys=original.exact_keys
write_x=original.write_x
sha=original.sha

def _original(root: Path) -> dict[str,Any]:
    require(sha(root/ORIGINAL_PROTOCOL)==ORIGINAL_PROTOCOL_SHA)
    require(sha(root/ORIGINAL_FAILURE)==ORIGINAL_FAILURE_SHA)
    old=original.validate_protocol(root)
    require(len(old.get('expected_hashes',{}))==ORIGINAL_COMPONENTS)
    audited=original.audit(root)
    require(audited['status']=='authenticated_execution_failure' and audited['artifact_sha256']==ORIGINAL_FAILURE_SHA)
    return old

def _names(old: dict[str,Any]) -> set[str]:
    names=set(old['expected_hashes']);require(len(names)==ORIGINAL_COMPONENTS and not(names&NEW_FILES));return names|NEW_FILES

def freeze(root: Path=ROOT) -> dict[str,Any]:
    root=Path(root);require(not(root/PROTOCOL).exists());old=_original(root);names=_names(old)
    require(not any((root/v).exists() for v in PATHS.values()))
    p={'schema':SCHEMA,'status':'frozen_before_execution','created_utc':datetime.now(timezone.utc).isoformat(),
       'parameters':PARAMETERS,'privacy':PRIVACY,'paths':PATHS,'authentication':old['authentication'],'data_roots':old['data_roots'],
       'original_protocol_sha256':ORIGINAL_PROTOCOL_SHA,'original_failure_sha256':ORIGINAL_FAILURE_SHA,
       'input_contract':INPUT_CONTRACT,'manifest_hashes':original._manifest_hashes(old['data_roots']),
       'expected_hashes':{n:sha(root/n) for n in sorted(names)}}
    write_x(root/PROTOCOL,p);validate_protocol(root)
    return {'status':'frozen','protocol_sha256':sha(root/PROTOCOL),'components':len(names)}

def validate_protocol(root: Path=ROOT) -> dict[str,Any]:
    root=Path(root);old=_original(root);p=json.loads((root/PROTOCOL).read_text())
    exact_keys(p,{'schema','status','created_utc','parameters','privacy','paths','authentication','data_roots','original_protocol_sha256','original_failure_sha256','input_contract','manifest_hashes','expected_hashes'})
    require(p['schema']==SCHEMA and p['status']=='frozen_before_execution' and p['parameters']==PARAMETERS and p['privacy']==PRIVACY and p['paths']==PATHS)
    require(p['authentication']==old['authentication'] and p['data_roots']==old['data_roots'] and p['input_contract']==INPUT_CONTRACT)
    require(p['original_protocol_sha256']==ORIGINAL_PROTOCOL_SHA and p['original_failure_sha256']==ORIGINAL_FAILURE_SHA)
    require(p['manifest_hashes']==original._manifest_hashes(p['data_roots']))
    names=_names(old);exact_keys(p['expected_hashes'],names)
    for n,h in p['expected_hashes'].items():
        path=(root/n).resolve();require(path.parent==root.resolve() and type(h) is str and len(h)==64 and sha(path)==h)
    return p

def validate_report(report: Any,p: dict[str,Any],digest: str,canonical_fold_sizes: tuple[int,...]|None=None) -> bool:
    if not isinstance(report,dict):return False
    common={'schema','status','protocol_sha256','code_hashes','source_hashes','support_receipt_sha256','fold_hashes','parameters','privacy','manifest_hashes'}
    auth=p['authentication']
    if report.get('status')=='completed_aggregate_only':
        if set(report)!=common|{'scope','analysis'}:return False
        scope=report['scope']
        if not isinstance(scope,dict) or set(scope)!={'canonical_patient_count','canonical_endpoint_count','canonical_fold_sizes','official_test_loaded'}:return False
        folds=tuple(scope['canonical_fold_sizes']) if isinstance(scope['canonical_fold_sizes'],list) and len(scope['canonical_fold_sizes'])==5 else ()
        if scope['canonical_patient_count']!=1928 or scope['canonical_endpoint_count']!=26 or scope['official_test_loaded'] is not False or any(type(x)is not int or x<10 for x in folds) or sum(folds)!=1928:return False
        if canonical_fold_sizes is not None and folds!=tuple(canonical_fold_sizes):return False
        analysis_ok=original._validate_analysis(report['analysis'],folds)
    elif report.get('status')=='suppressed_subgroup_small_cell':
        if set(report)!=common|{'analysis'} or report['analysis']!={'status':'suppressed_subgroup_small_cell','coverage':None}:return False
        analysis_ok=True
    else:return False
    return bool(analysis_ok and report['schema']==SCHEMA and report['protocol_sha256']==digest and report['code_hashes']==p['expected_hashes'] and report['source_hashes']==auth['canonical_source_hashes'] and report['support_receipt_sha256']==auth['support_receipt_sha256'] and report['fold_hashes']=={'outer':auth['outer_fold_sha256'],'inner':auth['inner_fold_sha256']} and report['parameters']==PARAMETERS and report['privacy']==PRIVACY and report['manifest_hashes']==p['manifest_hashes'])

def _base_report(p:dict[str,Any],root:Path)->dict[str,Any]:
    a=p['authentication'];return {'schema':SCHEMA,'protocol_sha256':sha(root/PROTOCOL),'code_hashes':p['expected_hashes'],'source_hashes':a['canonical_source_hashes'],'support_receipt_sha256':a['support_receipt_sha256'],'fold_hashes':{'outer':a['outer_fold_sha256'],'inner':a['inner_fold_sha256']},'parameters':PARAMETERS,'privacy':PRIVACY,'manifest_hashes':p['manifest_hashes']}

def run(root:Path=ROOT)->dict[str,Any]:
    root=Path(root);p={};fd=None;quiet=None;phase='protocol';paths={k:root/v for k,v in PATHS.items()}
    try:
        p=validate_protocol(root);require(not any(paths[k].exists() for k in ('success','failure','progress')));fd=original.prior.base.paired.base._acquire_lock(paths['lock']);original._write_progress(paths['progress'],'validated');quiet=original.prior.base.paired.base._quiet_sensitive_block();quiet.__enter__();phase='context'
        from patient_atlas_v6_2_expanded_endpoint_evaluation import EXACT_INNER_FOLD_ASSIGNMENT_SHA256,EXACT_OUTER_FOLD_HASH,FROZEN_SUPPORT_RECEIPT_NAME,load_eligible_support_receipt,validate_support_against_observed
        from run_patient_atlas_v6_2_expanded_endpoint_evaluation import _load_actual_v6_2_context
        support=load_eligible_support_receipt(root/FROZEN_SUPPORT_RECEIPT_NAME,project_root=root);context=_load_actual_v6_2_context(root=root,support=support,**{k:Path(v) for k,v in p['data_roots'].items()});outer=np.asarray(context['outer_assignment']);a=p['authentication']
        require(len(outer)==1928 and outer.dtype.kind in 'iu' and set(outer.tolist())==set(range(5)) and len(support.eligible_sources)==26)
        require(dict(context['source_hashes'])==a['canonical_source_hashes'] and support.receipt_sha256==a['support_receipt_sha256'] and a['outer_fold_sha256']==EXACT_OUTER_FOLD_HASH and len(a['inner_fold_sha256'])==5)
        validate_support_against_observed(support,context['labels_by_source'],context['observed_by_source'],outer)
        for fold in range(5):
            _,digest=original.prior.base.paired.base._inner_context(context,np.flatnonzero(outer!=fold),fold);require(digest==a['inner_fold_sha256'][fold]==EXACT_INNER_FOLD_ASSIGNMENT_SHA256[fold])
        ids=original._local_canonical_ids(context['raw_cohort'].patient_ids);observed=context['observed_by_source']['mhterm_dm2'];labels=context['labels_by_source']['mhterm_dm2']
        require(isinstance(observed,np.ndarray) and observed.dtype==np.dtype(bool) and observed.shape==outer.shape and isinstance(labels,np.ndarray) and labels.shape==outer.shape and labels.dtype.kind in 'biuf' and np.all(np.isfinite(labels[observed])) and np.all((labels[observed]==0)|(labels[observed]==1)))
        selected=observed&(labels==1);subset_ids=tuple(x for x,keep in zip(ids,selected,strict=True) if bool(keep));subset_outer=tuple(int(x) for x,keep in zip(outer.tolist(),selected,strict=True) if bool(keep));canonical_folds=tuple(int(np.sum(outer==f)) for f in range(5));subgroup_folds=tuple(subset_outer.count(f) for f in range(5));phase='subgroup_gate'
        if not original._subgroup_gate(canonical_folds,subgroup_folds):report=_base_report(p,root)|{'status':'suppressed_subgroup_small_cell','analysis':{'status':'suppressed_subgroup_small_cell','coverage':None}}
        else:
            phase='manifest';before=original._manifest_hashes(p['data_roots']);require(before==p['manifest_hashes']);source_ids={};masks={}
            for modality,path in original._manifest_paths(p['data_roots']).items():
                with path.open('r',encoding='utf-8',newline='') as handle:source_ids[modality],masks[modality]=parse_manifest(handle,modality)
            phase='aggregate';coverage=summarize_coverage(subset_ids,subset_outer,source_ids,masks);require(original._manifest_hashes(p['data_roots'])==before==p['manifest_hashes'])
            report=_base_report(p,root)|{'status':'completed_aggregate_only','scope':{'canonical_patient_count':1928,'canonical_endpoint_count':26,'canonical_fold_sizes':list(canonical_folds),'official_test_loaded':False},'analysis':original._aggregate_analysis(coverage,len(subset_ids),subgroup_folds)}
        phase='writing';p=validate_protocol(root);require(validate_report(report,p,sha(root/PROTOCOL),canonical_folds));write_x(paths['success'],report);original._write_progress(paths['progress'],'completed');return {'status':report['status'],'artifact_sha256':sha(paths['success'])}
    except Exception as error:
        frames=[{'file':Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(error.__traceback__) if Path(x.filename).resolve().parent==root.resolve() and Path(x.filename).name in p.get('expected_hashes',{})][:8]
        failure={'schema':SCHEMA,'status':'failed','phase':phase,'error_class':type(error).__name__,'bound_code_frames':frames,'exception_text_serialized':False,'patient_content_serialized':False}
        if fd is not None and not paths['success'].exists() and not paths['failure'].exists():write_x(paths['failure'],failure)
        return {'status':'execution_failed','phase':phase,'error_class':type(error).__name__,'exception_contents_emitted':False}
    finally:
        if quiet is not None:quiet.__exit__(None,None,None)
        if fd is not None:
            os.close(fd)
            try:paths['lock'].unlink()
            except FileNotFoundError:pass

def audit(root:Path=ROOT)->dict[str,Any]:
    root=Path(root);p=validate_protocol(root);paths={k:root/v for k,v in PATHS.items()};require(not(paths['success'].exists() and paths['failure'].exists()))
    if paths['failure'].exists():
        f=json.loads(paths['failure'].read_text());exact_keys(f,{'schema','status','phase','error_class','bound_code_frames','exception_text_serialized','patient_content_serialized'});require(f['schema']==SCHEMA and f['status']=='failed' and f['phase'] in PHASES and f['exception_text_serialized'] is False and f['patient_content_serialized'] is False);require(type(f['error_class']) is str and f['error_class'].isidentifier() and len(f['error_class'])<90 and type(f['bound_code_frames']) is list and len(f['bound_code_frames'])<=8)
        for frame in f['bound_code_frames']:
            exact_keys(frame,{'file','line'});require(frame['file'] in p['expected_hashes'] and type(frame['line']) is int and 0<frame['line']<100000)
        return {'status':'authenticated_execution_failure','artifact_sha256':sha(paths['failure']),'phase':f['phase'],'error_class':f['error_class']}
    if paths['success'].exists():
        r=json.loads(paths['success'].read_text());require(validate_report(r,p,sha(root/PROTOCOL)));out={'status':'authenticated_success','artifact_sha256':sha(paths['success']),'protocol_sha256':sha(root/PROTOCOL),'components':len(p['expected_hashes']),'analysis':r['analysis']}
        if r['status']=='completed_aggregate_only':out|={'scope':r['scope'],'folds_authenticated':True}
        return out
    return {'status':'no_terminal_artifact','lock_exists':paths['lock'].exists()}

if __name__=='__main__':
    try:print(json.dumps({'freeze':freeze,'run':run,'audit':audit}[sys.argv[1]](),sort_keys=True,allow_nan=False))
    except Exception as error:print(json.dumps({'status':'blocked_without_disclosure','error_class':type(error).__name__,'contents_emitted':False}));raise SystemExit(1)
