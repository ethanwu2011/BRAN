"""Count-free local feasibility gate for a future BRAN disease-structure protocol.

The emitted booleans are fixed design-threshold checks only; they do not establish
clinical validity, stability, efficacy, or a novel subtype.
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

import bran_independent_coverage_experiment_v1_attempt2 as coverage
from bran_independent_coverage_io_v1 import MODALITY_FILES, SOURCE_COLUMNS
from bran_independent_coverage_io_v1_attempt2 import parse_manifest

ROOT=Path(__file__).resolve().parent
SCHEMA='bran-disease-structure-feasibility-v1'
PROTOCOL='BRAN_DISEASE_STRUCTURE_FEASIBILITY_PROTOCOL_V1.json'
OUTDIR='validation_results/BRAN_DISEASE_STRUCTURE_FEASIBILITY_V1'
PATHS={k:OUTDIR+'/'+v for k,v in {'success':'SUCCESS.json','failure':'FAILURE.json','progress':'progress.json','lock':'run.lock'}.items()}
COVERAGE_PROTOCOL=coverage.PROTOCOL
COVERAGE_PROTOCOL_SHA='d1d41f6fefa75a5ccb27e0a18ab17784b25fc33769173b835227fb3dac97cfc5'
COVERAGE_SUCCESS=coverage.PATHS['success']
COVERAGE_SUCCESS_SHA='96f7e50c54b5a1363fbdaa8a218a17d4602c2ffc2edbb3a6c5cf964d871edcbe'
COVERAGE_COMPONENTS=110
NEW_FILES={'bran_disease_structure_feasibility_v1.py','test_bran_disease_structure_feasibility_v1.py',COVERAGE_PROTOCOL}
THRESHOLDS={'discovery_support_at_least_80':80,'validation_support_at_least_40':40,'replication_support_at_least_40':40}
PRIVACY={'patient_processing_local_only':True,'patient_ids_rows_arrays_values_or_counts_serialized':False,'clinical_validity_stability_or_subtype_claimed':False}
PHASES={'protocol','validated','context','manifest','aggregate','writing','completed'}
require=coverage.require;exact_keys=coverage.exact_keys;write_x=coverage.write_x;sha=coverage.sha

def _prior(root:Path)->dict[str,Any]:
    require(sha(root/COVERAGE_PROTOCOL)==COVERAGE_PROTOCOL_SHA and sha(root/COVERAGE_SUCCESS)==COVERAGE_SUCCESS_SHA)
    p=coverage.validate_protocol(root);require(len(p.get('expected_hashes',{}))==COVERAGE_COMPONENTS)
    a=coverage.audit(root);require(a['status']=='authenticated_success' and a['artifact_sha256']==COVERAGE_SUCCESS_SHA)
    return p

def _names(p:dict[str,Any])->set[str]:
    names=set(p['expected_hashes']);require(len(names)==COVERAGE_COMPONENTS and not(names&NEW_FILES));return names|NEW_FILES

def freeze(root:Path=ROOT)->dict[str,Any]:
    root=Path(root);require(not(root/PROTOCOL).exists());old=_prior(root);names=_names(old);require(not any((root/v).exists() for v in PATHS.values()))
    p={'schema':SCHEMA,'status':'frozen_before_execution','created_utc':datetime.now(timezone.utc).isoformat(),'paths':PATHS,'thresholds':THRESHOLDS,'privacy':PRIVACY,'authentication':old['authentication'],'data_roots':old['data_roots'],'coverage_protocol_sha256':COVERAGE_PROTOCOL_SHA,'coverage_success_sha256':COVERAGE_SUCCESS_SHA,'manifest_hashes':old['manifest_hashes'],'expected_hashes':{n:sha(root/n) for n in sorted(names)}}
    write_x(root/PROTOCOL,p);validate_protocol(root);return {'status':'frozen','protocol_sha256':sha(root/PROTOCOL),'components':len(names)}

def validate_protocol(root:Path=ROOT)->dict[str,Any]:
    root=Path(root);old=_prior(root);p=json.loads((root/PROTOCOL).read_text());exact_keys(p,{'schema','status','created_utc','paths','thresholds','privacy','authentication','data_roots','coverage_protocol_sha256','coverage_success_sha256','manifest_hashes','expected_hashes'})
    require(p['schema']==SCHEMA and p['status']=='frozen_before_execution' and p['paths']==PATHS and p['thresholds']==THRESHOLDS and p['privacy']==PRIVACY and p['authentication']==old['authentication'] and p['data_roots']==old['data_roots'] and p['manifest_hashes']==old['manifest_hashes'] and p['coverage_protocol_sha256']==COVERAGE_PROTOCOL_SHA and p['coverage_success_sha256']==COVERAGE_SUCCESS_SHA and p['manifest_hashes']==coverage.original._manifest_hashes(p['data_roots']))
    names=_names(old);exact_keys(p['expected_hashes'],names)
    for n,h in p['expected_hashes'].items():
        path=(root/n).resolve();require(path.parent==root.resolve() and type(h)is str and len(h)==64 and sha(path)==h)
    return p

def support_booleans(subgroup_ids:Any,subgroup_outer:Any,source_ids:dict[str,tuple[str,...]],masks:dict[str,dict[str,tuple[bool,...]]])->dict[str,dict[str,dict[str,bool]]]:
    """Local threshold gate; accepts only IDs, outer folds, and finite-mask booleans."""
    require(not isinstance(subgroup_ids,(str,bytes)) and not isinstance(subgroup_outer,(str,bytes)))
    ids=tuple(subgroup_ids);outer=tuple(subgroup_outer);require(len(ids)==len(outer) and len(set(ids))==len(ids) and all(type(identifier)is str and bool(identifier) for identifier in ids) and all(type(f)is int and 0<=f<5 for f in outer))
    result={}
    for modality,fields in SOURCE_COLUMNS.items():
        require(set(source_ids)==set(SOURCE_COLUMNS) and set(masks)==set(SOURCE_COLUMNS) and set(masks[modality])==set(fields))
        require(not isinstance(source_ids[modality],(str,bytes)))
        positions={identifier:index for index,identifier in enumerate(source_ids[modality])};require(len(positions)==len(source_ids[modality]) and all(type(identifier)is str and bool(identifier) for identifier in source_ids[modality]))
        result[modality]={}
        for field in fields:
            values=masks[modality][field];require(len(values)==len(source_ids[modality]) and all(type(v)is bool for v in values))
            paired=[(fold,values[positions[identifier]]) for identifier,fold in zip(ids,outer,strict=True) if identifier in positions]
            result[modality][field]={'discovery_support_at_least_80':sum(ok for fold,ok in paired if fold in (0,1,2))>=80,'validation_support_at_least_40':sum(ok for fold,ok in paired if fold==3)>=40,'replication_support_at_least_40':sum(ok for fold,ok in paired if fold==4)>=40}
    return result

def validate_report(r:Any,p:dict[str,Any],digest:str)->bool:
    if not isinstance(r,dict) or set(r)!={'schema','status','protocol_sha256','code_hashes','source_hashes','support_receipt_sha256','fold_hashes','manifest_hashes','thresholds','privacy','field_support'}:return False
    a=p['authentication'];fields=r['field_support']
    if r['schema']!=SCHEMA or r['status']!='completed_count_free_feasibility' or r['protocol_sha256']!=digest or r['code_hashes']!=p['expected_hashes'] or r['source_hashes']!=a['canonical_source_hashes'] or r['support_receipt_sha256']!=a['support_receipt_sha256'] or r['fold_hashes']!={'outer':a['outer_fold_sha256'],'inner':a['inner_fold_sha256']} or r['manifest_hashes']!=p['manifest_hashes'] or r['thresholds']!=THRESHOLDS or r['privacy']!=PRIVACY:return False
    if not isinstance(fields,dict) or set(fields)!=set(SOURCE_COLUMNS):return False
    for modality,names in SOURCE_COLUMNS.items():
        if not isinstance(fields[modality],dict) or set(fields[modality])!=set(names):return False
        for flags in fields[modality].values():
            if not isinstance(flags,dict) or set(flags)!=set(THRESHOLDS) or any(type(v)is not bool for v in flags.values()):return False
    return True

def run(root:Path=ROOT)->dict[str,Any]:
    root=Path(root);p={};fd=None;quiet=None;phase='protocol';paths={k:root/v for k,v in PATHS.items()}
    try:
        p=validate_protocol(root);require(not any(paths[k].exists() for k in ('success','failure','progress')));fd=coverage.original.prior.base.paired.base._acquire_lock(paths['lock']);coverage.original._write_progress(paths['progress'],'validated');quiet=coverage.original.prior.base.paired.base._quiet_sensitive_block();quiet.__enter__();phase='context'
        from patient_atlas_v6_2_expanded_endpoint_evaluation import EXACT_INNER_FOLD_ASSIGNMENT_SHA256,EXACT_OUTER_FOLD_HASH,FROZEN_SUPPORT_RECEIPT_NAME,load_eligible_support_receipt,validate_support_against_observed
        from run_patient_atlas_v6_2_expanded_endpoint_evaluation import _load_actual_v6_2_context
        support=load_eligible_support_receipt(root/FROZEN_SUPPORT_RECEIPT_NAME,project_root=root);context=_load_actual_v6_2_context(root=root,support=support,**{k:Path(v) for k,v in p['data_roots'].items()});outer=np.asarray(context['outer_assignment']);a=p['authentication'];require(outer.shape==(1928,) and outer.dtype.kind in 'iu' and set(outer.tolist())==set(range(5)) and len(support.eligible_sources)==26 and dict(context['source_hashes'])==a['canonical_source_hashes'] and support.receipt_sha256==a['support_receipt_sha256'] and a['outer_fold_sha256']==EXACT_OUTER_FOLD_HASH)
        validate_support_against_observed(support,context['labels_by_source'],context['observed_by_source'],outer)
        for f in range(5):_,d=coverage.original.prior.base.paired.base._inner_context(context,np.flatnonzero(outer!=f),f);require(d==a['inner_fold_sha256'][f]==EXACT_INNER_FOLD_ASSIGNMENT_SHA256[f])
        ids=coverage.original._local_canonical_ids(context['raw_cohort'].patient_ids);observed=context['observed_by_source']['mhterm_dm2'];labels=context['labels_by_source']['mhterm_dm2'];require(isinstance(observed,np.ndarray) and observed.dtype==np.dtype(bool) and observed.shape==outer.shape and isinstance(labels,np.ndarray) and labels.shape==outer.shape and labels.dtype.kind in 'biuf' and np.all(np.isfinite(labels[observed])) and np.all((labels[observed]==0)|(labels[observed]==1)))
        selected=observed&(labels==1);sub_ids=tuple(x for x,k in zip(ids,selected,strict=True) if bool(k));sub_outer=tuple(int(x) for x,k in zip(outer.tolist(),selected,strict=True) if bool(k));require(len(sub_ids)==719);phase='manifest';before=coverage.original._manifest_hashes(p['data_roots']);require(before==p['manifest_hashes']);source_ids={};masks={}
        for modality,path in coverage.original._manifest_paths(p['data_roots']).items():
            with path.open('r',encoding='utf-8',newline='') as h:source_ids[modality],masks[modality]=parse_manifest(h,modality)
        phase='aggregate';flags=support_booleans(sub_ids,sub_outer,source_ids,masks);require(coverage.original._manifest_hashes(p['data_roots'])==before==p['manifest_hashes']);r={'schema':SCHEMA,'status':'completed_count_free_feasibility','protocol_sha256':sha(root/PROTOCOL),'code_hashes':p['expected_hashes'],'source_hashes':a['canonical_source_hashes'],'support_receipt_sha256':a['support_receipt_sha256'],'fold_hashes':{'outer':a['outer_fold_sha256'],'inner':a['inner_fold_sha256']},'manifest_hashes':p['manifest_hashes'],'thresholds':THRESHOLDS,'privacy':PRIVACY,'field_support':flags};phase='writing';p=validate_protocol(root);require(validate_report(r,p,sha(root/PROTOCOL)));write_x(paths['success'],r);coverage.original._write_progress(paths['progress'],'completed');return {'status':'completed_count_free_feasibility','artifact_sha256':sha(paths['success'])}
    except Exception as e:
        frames=[{'file':Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__) if Path(x.filename).resolve().parent==root.resolve() and Path(x.filename).name in p.get('expected_hashes',{})][:8];f={'schema':SCHEMA,'status':'failed','phase':phase,'error_class':type(e).__name__,'bound_code_frames':frames,'exception_text_serialized':False,'patient_content_serialized':False}
        if fd is not None and not paths['success'].exists() and not paths['failure'].exists():write_x(paths['failure'],f)
        return {'status':'execution_failed','phase':phase,'error_class':type(e).__name__,'exception_contents_emitted':False}
    finally:
        if quiet is not None:quiet.__exit__(None,None,None)
        if fd is not None:
            os.close(fd)
            try:paths['lock'].unlink()
            except FileNotFoundError:pass

def audit(root:Path=ROOT)->dict[str,Any]:
    root=Path(root);p=validate_protocol(root);paths={k:root/v for k,v in PATHS.items()};require(not(paths['success'].exists() and paths['failure'].exists()))
    if paths['failure'].exists():
        f=json.loads(paths['failure'].read_text());exact_keys(f,{'schema','status','phase','error_class','bound_code_frames','exception_text_serialized','patient_content_serialized'});require(f['schema']==SCHEMA and f['status']=='failed' and f['phase'] in PHASES and f['exception_text_serialized'] is False and f['patient_content_serialized'] is False and type(f['error_class']) is str and f['error_class'].isidentifier() and type(f['bound_code_frames']) is list and len(f['bound_code_frames'])<=8)
        for frame in f['bound_code_frames']:exact_keys(frame,{'file','line'});require(frame['file'] in p['expected_hashes'] and type(frame['line']) is int and 0<frame['line']<100000)
        return {'status':'authenticated_execution_failure','artifact_sha256':sha(paths['failure']),'phase':f['phase'],'error_class':f['error_class']}
    if paths['success'].exists():
        r=json.loads(paths['success'].read_text());require(validate_report(r,p,sha(root/PROTOCOL)));return {'status':'authenticated_success','artifact_sha256':sha(paths['success']),'protocol_sha256':sha(root/PROTOCOL),'components':len(p['expected_hashes']),'field_support':r['field_support']}
    out={'status':'no_terminal_artifact','lock_exists':paths['lock'].exists()}
    if paths['progress'].exists():
        q=json.loads(paths['progress'].read_text());require(isinstance(q,dict) and set(q)=={'status','phase'} and q['status'] in {'running','completed'} and q['phase'] in PHASES);out['progress']=q
    return out

if __name__=='__main__':
    try:print(json.dumps({'freeze':freeze,'run':run,'audit':audit}[sys.argv[1]](),sort_keys=True,allow_nan=False))
    except Exception as e:print(json.dumps({'status':'blocked_without_disclosure','error_class':type(e).__name__,'contents_emitted':False}));raise SystemExit(1)
