"""Authenticate exclusive aggregate artifacts before exposing any content."""
import hashlib
import json
import math
from pathlib import Path
from run_bran_anchor_ablation_v2 import SCHEMA,PARAMETERS,ARM_NAMES,COMPARISONS,CBC_FIELDS,validate_protocol

ROOT=Path(__file__).resolve().parent
SCENARIOS={'retina_plus_other_clinical':'primary_all9_cbc_hidden',
           'retina_absent_other_clinical':'descriptive_robustness','retina_age_only':'descriptive_no_clinical_view'}
CBC_METRICS={m+'_'+e for m in ('bare_plugin_decoder','posterior_predictive_mc_mean','state_age_ridge',
              'raw_available_ridge','age_ridge','median_baseline') for e in ('mae','mse')} | {
              'posterior_predictive_coverage90','posterior_predictive_interval_width90'}

class AuditError(RuntimeError):pass

def require(ok):
    if not ok:raise AuditError('aggregate schema or authentication failed; contents withheld')

def finite(v,lo=0.,hi=None):
    require(type(v) in (int,float) and math.isfinite(v) and v>=lo and (hi is None or v<=hi))

def ci(v,lo,hi):
    require(isinstance(v,list) and len(v)==2)
    for x in v:finite(x,lo,hi)
    require(v[0]<=v[1])

def validate_report(r,p,digest):
    require(set(r)=={'schema_version','status','exploratory','protocol_sha256','code_hashes','source_hashes',
        'support_receipt_sha256','scope','fold_hashes','parameters','screening','clinical_recoverability','cbc',
        'models_oof_rows_ids_or_draws_serialized','patient_rows_or_ids_serialized'})
    require(r['schema_version']==SCHEMA and r['status']=='completed_aggregate_only' and r['exploratory'] is True)
    require(r['models_oof_rows_ids_or_draws_serialized'] is False and r['patient_rows_or_ids_serialized'] is False)
    require(r['protocol_sha256']==digest and r['code_hashes']==p['expected_hashes'] and r['parameters']==PARAMETERS)
    auth=p['authentication']
    require(r['source_hashes']==auth['canonical_source_hashes'] and r['support_receipt_sha256']==auth['support_receipt_sha256'])
    require(r['fold_hashes']=={'outer':auth['outer_fold_sha256'],'inner':auth['inner_fold_sha256']})
    require(r['scope']=={'patient_count':1928,'endpoint_count':26,'official_test_loaded':False})
    s=r['screening'];require(set(s)=={'endpoint_results','macro_26_endpoint_paired_deltas','inference','head_diagnostics'})
    require(s['inference']=='one_shared_fold_stratified_patient_bootstrap_count_matrix_marginal_95pct_fixed_oof_1000')
    endpoints=s['endpoint_results'];require(set(endpoints)==set(p['scope']['eligible_source_codes']) and len(endpoints)==26)
    for e in endpoints.values():
        require(set(e)=={'arms','paired_deltas'} and set(e['arms'])==set(ARM_NAMES) and set(e['paired_deltas'])==set(COMPARISONS))
        for m in e['arms'].values():
            require(set(m)=={'auroc','logloss','ci95'});finite(m['auroc'],0,1);finite(m['logloss']);ci(m['ci95'],0,1)
        for name,d in e['paired_deltas'].items():
            require(set(d)=={'auroc_delta','ci95'});finite(d['auroc_delta'],-1,1);ci(d['ci95'],-1,1)
            left,right=name.split('-');require(abs(d['auroc_delta']-(e['arms'][left]['auroc']-e['arms'][right]['auroc']))<1e-12)
    macro=s['macro_26_endpoint_paired_deltas'];require(set(macro)==set(COMPARISONS))
    for name,d in macro.items():
        require(set(d)=={'mean_26_endpoint_auroc_delta','ci95'});finite(d['mean_26_endpoint_auroc_delta'],-1,1);ci(d['ci95'],-1,1)
        require(abs(d['mean_26_endpoint_auroc_delta']-sum(e['paired_deltas'][name]['auroc_delta'] for e in endpoints.values())/26)<1e-12)
    hd=s['head_diagnostics'];require(set(hd)=={'completed_head_refits','candidates_rejected_nonconvergence','candidates_rejected_incomplete_inner_support'})
    require(hd['completed_head_refits']==1300)
    for value in hd.values():require(type(value) is int and 0<=value<=5200)
    recovery=r['clinical_recoverability'];require(set(recovery)=={'v1','v2'})
    for routes in recovery.values():
        require(set(routes)=={'clinical','both'})
        for m in routes.values():
            if m['status']=='suppressed_eligible_field_below_10_in_any_fold':require(set(m)=={'status'})
            else:
                require(set(m)=={'status','eligible_field_macro_mse'} and m['status']=='scored_normalized_mse');finite(m['eligible_field_macro_mse'])
    cbc=r['cbc'];require(set(cbc)=={'scenarios','v1','v2','v2_minus_v1_point_differences_no_paired_ci'} and cbc['scenarios']==SCENARIOS)
    for ver in ('v1','v2'):
        require(set(cbc[ver])==set(SCENARIOS))
        for fields in cbc[ver].values():
            require(set(fields)==set(CBC_FIELDS))
            for m in fields.values():
                if m['status']=='suppressed_any_outer_fold_below_10':require(set(m)=={'status'});continue
                require(set(m)==CBC_METRICS|{'status'} and m['status']=='scored_normalized_units_only')
                for key in CBC_METRICS:finite(m[key])
                finite(m['posterior_predictive_coverage90'],0,1)
    diffs=cbc['v2_minus_v1_point_differences_no_paired_ci'];require(set(diffs)==set(SCENARIOS))
    for q,fields in diffs.items():
        require(set(fields)==set(CBC_FIELDS))
        for field,m in fields.items():
            a,b=cbc['v1'][q][field],cbc['v2'][q][field]
            if a['status']!='scored_normalized_units_only' or b['status']!='scored_normalized_units_only':
                require(m=={'status':'suppressed'});continue
            require(set(m)==CBC_METRICS|{'status'} and m['status']=='scored_no_paired_ci')
            for key in CBC_METRICS:require(type(m[key]) in (int,float) and math.isfinite(m[key]) and abs(m[key]-(b[key]-a[key]))<1e-12)
    return {'macro_auc':{a:sum(e['arms'][a]['auroc'] for e in endpoints.values())/26 for a in ARM_NAMES},
        'paired_macro_auc_deltas':macro,'clinical_recoverability':recovery,'head_diagnostics':hd,
        'both_above_own_single_views_point_count':{v:sum(e['arms'][v+'both']['auroc']>max(e['arms'][v+'clinical']['auroc'],e['arms'][v+'retinal']['auroc']) for e in endpoints.values()) for v in ('v1','v2')}}

def audit(root=ROOT):
    path=root/'BRAN_ANCHOR_ABLATION_PROTOCOL_V2.json';p=validate_protocol(root,path);b=p['_paths'];success,failure=b['output'],b['failure']
    require(not(success.exists() and failure.exists()))
    if not(success.exists() or failure.exists()):
        out={'status':'no_terminal_artifact','lock_exists':b['lock'].exists()}
        if b['progress'].exists():
            q=json.loads(b['progress'].read_text());require(set(q)<={'status','phase','fold'})
            require(q.get('phase') in {'validated','training','completion','screening','fold_complete','aggregate','completed'})
            require(q.get('status') in {'running','completed'})
            if 'fold' in q:require(type(q['fold']) is int and 0<=q['fold']<5)
            out['progress']=q
        return out
    target=success if success.exists() else failure;raw=target.read_bytes();r=json.loads(raw);sha=hashlib.sha256(raw).hexdigest()
    if failure.exists():
        require(set(r)=={'schema_version','status','phase','error_class','bound_code_frames','exception_text_serialized','patient_rows_or_ids_serialized'})
        require(r['schema_version']==SCHEMA and r['status']=='failed' and r['exception_text_serialized'] is False and r['patient_rows_or_ids_serialized'] is False)
        require(r['phase'] in {'protocol','context','folds','aggregate'})
        require(isinstance(r['error_class'],str) and r['error_class'].isidentifier() and len(r['error_class'])<90)
        require(isinstance(r['bound_code_frames'],list) and len(r['bound_code_frames'])<=8)
        for f in r['bound_code_frames']:
            require(set(f)=={'file','line'} and f['file'] in p['expected_hashes'] and type(f['line']) is int and 1<=f['line']<100000)
        return {'status':'authenticated_execution_failure','artifact_sha256':sha,'phase':r['phase'],'error_class':r['error_class'],'code_frames':r['bound_code_frames']}
    checked=validate_report(r,p,hashlib.sha256(path.read_bytes()).hexdigest())
    return {'status':'authenticated_success','artifact_sha256':sha,'protocol_sha256':r['protocol_sha256'],**checked,
        'claim_limit':'Paired internal development diagnostic; no external validation, familywise superiority, calibrated imputation or new subtype claim.'}

if __name__=='__main__':
    try:print(json.dumps(audit(),indent=2,sort_keys=True,allow_nan=False))
    except Exception as exc:
        print(json.dumps({'status':'audit_failed','error_class':type(exc).__name__,'artifact_contents_emitted':False}));raise SystemExit(1)
