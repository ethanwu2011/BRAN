"""Fail-closed disclosure-safe audit of one frozen conditional-mean experiment."""
import hashlib
import json
import math
from pathlib import Path
from run_bran_frozen_cbc_readout_v1 import SCHEMA, PARAMETERS, CBC_FIELDS, ARMS, SCENARIOS, validate_protocol

ROOT=Path(__file__).resolve().parent
CONTRASTS=tuple('state_nested_ridge_minus_'+r for r in ('state_unscaled_ridge1','decoder_mc','raw_nested_ridge'))
SCENARIO_ROLES=dict(zip(SCENARIOS,('primary','descriptive','descriptive')))
PHASES={'protocol','context','training','completion','fold_complete','aggregate','replay','writing','validated','completed'}
class AuditError(RuntimeError):pass
def require(ok):
    if not ok:raise AuditError('authentication or disclosure schema invalid; contents withheld')
def number(x, nonnegative=False):
    require(type(x) in (int,float) and math.isfinite(x) and (not nonnegative or x>=0))
def interval(x):
    require(isinstance(x,list) and len(x)==2)
    for v in x:number(v)
    require(x[0]<=x[1])
def validate_report(r,p,digest):
    require(set(r)=={'schema_version','status','protocol_sha256','code_hashes','source_hashes','support_receipt_sha256',
        'scope','fold_hashes','parameters','scenarios','cbc','prior_paired_replay_authenticated',
        'encoder_parameters_unchanged_before_after_heads','patient_rows_or_ids_serialized',
        'models_or_oof_predictions_serialized','uncertainty_claim'})
    require(r['schema_version']==SCHEMA and r['status']=='completed_aggregate_only')
    require(r['parameters']==PARAMETERS and r['protocol_sha256']==digest and r['code_hashes']==p['expected_hashes'])
    a=p['authentication']
    require(r['source_hashes']==a['canonical_source_hashes'] and r['support_receipt_sha256']==a['support_receipt_sha256'])
    require(r['fold_hashes']=={'outer':a['outer_fold_sha256'],'inner':a['inner_fold_sha256']})
    require(r['scope']=={'patient_count':1928,'endpoint_count':26,'official_test_loaded':False})
    require(r['prior_paired_replay_authenticated'] is True and r['encoder_parameters_unchanged_before_after_heads'] is True)
    require(r['patient_rows_or_ids_serialized'] is False and r['models_or_oof_predictions_serialized'] is False)
    require(r['uncertainty_claim']=='conditional_mean_only_no_calibration_or_joint_generation_claim')
    require(r['scenarios']==SCENARIO_ROLES and set(r['cbc'])==set(SCENARIOS))
    summary={}
    for q,c in r['cbc'].items():
        require(set(c)=={'field_metrics','paired_mse_contrasts'})
        fields=c['field_metrics'];require(set(fields)==set(CBC_FIELDS) and set(c['paired_mse_contrasts'])==set(CONTRASTS))
        for f,m in fields.items():
            require(set(m)=={'status'}|set(ARMS)|set(CONTRASTS) and m['status']=='scored_normalized_units_only')
            for arm in ARMS:
                v=m[arm];require(set(v)=={'normalized_mae','normalized_mse'})
                number(v['normalized_mae'],True);number(v['normalized_mse'],True)
                require(v['normalized_mae']**2<=v['normalized_mse']+1e-8)
            for name in CONTRASTS:
                v=m[name];require(set(v)=={'normalized_mse_difference','ci95'})
                number(v['normalized_mse_difference']);interval(v['ci95'])
                right=name.split('_minus_',1)[1]
                require(abs(v['normalized_mse_difference']-(m['state_nested_ridge']['normalized_mse']-m[right]['normalized_mse']))<1e-10)
        for name,v in c['paired_mse_contrasts'].items():
            require(set(v)=={'mean_9_normalized_mse_difference','ci95'})
            number(v['mean_9_normalized_mse_difference']);interval(v['ci95'])
            require(abs(v['mean_9_normalized_mse_difference']-sum(m[name]['normalized_mse_difference'] for m in fields.values())/9)<1e-10)
        summary[q]={'macro9_normalized_mse':{arm:sum(m[arm]['normalized_mse'] for m in fields.values())/9 for arm in ARMS},
            'paired_macro_contrasts':c['paired_mse_contrasts'],
            'mse_below_median_point_count':{arm:sum(m[arm]['normalized_mse']<m['median']['normalized_mse'] for m in fields.values()) for arm in ARMS if arm!='median'}}
    return summary
def audit(root=ROOT):
    path=root/'BRAN_FROZEN_CBC_READOUT_PROTOCOL_V1.json';p=validate_protocol(root,path);b=p['_paths']
    success,failure=b['output'],b['failure'];require(not(success.exists() and failure.exists()))
    if not(success.exists() or failure.exists()):
        result={'status':'no_terminal_artifact','lock_exists':b['lock'].exists()}
        if b['progress'].exists():
            q=json.loads(b['progress'].read_text());require(set(q)<={'status','phase','fold'})
            require(q.get('status') in {'running','completed'} and q.get('phase') in PHASES)
            if 'fold' in q:require(type(q['fold']) is int and 0<=q['fold']<5)
            result['progress']=q
        return result
    target=success if success.exists() else failure;raw=target.read_bytes();r=json.loads(raw);sha=hashlib.sha256(raw).hexdigest()
    if failure.exists():
        require(set(r)=={'schema_version','status','phase','error_class','bound_code_frames','exception_text_serialized','patient_rows_or_ids_serialized'})
        require(r['schema_version']==SCHEMA and r['status']=='failed' and r['phase'] in PHASES)
        require(r['exception_text_serialized'] is False and r['patient_rows_or_ids_serialized'] is False)
        require(isinstance(r['error_class'],str) and r['error_class'].isidentifier() and len(r['error_class'])<90)
        require(isinstance(r['bound_code_frames'],list) and len(r['bound_code_frames'])<=8)
        for f in r['bound_code_frames']:
            require(set(f)=={'file','line'} and f['file'] in p['expected_hashes'] and type(f['line']) is int and 1<=f['line']<100000)
        return {'status':'authenticated_execution_failure','artifact_sha256':sha,'phase':r['phase'],
            'error_class':r['error_class'],'code_frames':r['bound_code_frames']}
    summary=validate_report(r,p,hashlib.sha256(path.read_bytes()).hexdigest())
    return {'status':'authenticated_success','artifact_sha256':sha,'protocol_sha256':r['protocol_sha256'],
        'cbc':summary,'prior_replay_verified':True,'encoder_unchanged':True,
        'claim_limit':'Conditional means only. Internal reused development data and fixed-fit marginal intervals; no calibrated generation, screening improvement or external validation.'}
if __name__=='__main__':
    try:print(json.dumps(audit(),indent=2,sort_keys=True,allow_nan=False))
    except Exception as exc:
        print(json.dumps({'status':'audit_failed','error_class':type(exc).__name__,'artifact_contents_emitted':False}));raise SystemExit(1)
