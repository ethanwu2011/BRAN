"""Frozen, local-only execution and closed-schema audit for one bounded readout test."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from datetime import datetime, timezone
import numpy as np
import run_bran_anchor_ablation_v2 as paired
import run_bran_frozen_cbc_readout_v1 as previous
import audit_bran_anchor_ablation_v2 as paired_audit
import audit_bran_frozen_cbc_readout_v1 as cbc_audit
from bran_innovation_readout_v1 import PROFILES,C_GRID,ALPHA_GRID
from bran_innovation_aggregate_v1 import COMPARATORS,FAMILY
from run_bran_innovation_readout_v1 import evaluate,ARMS,CBC_ARMS,CBC_FIELDS,SCENARIOS

ROOT=Path(__file__).resolve().parent
PROTOCOL_NAME='BRAN_INNOVATION_READOUT_PROTOCOL_V1.json'
SCHEMA='bran-innovation-experiment-v1'
OUTDIR='validation_results/BRAN_INNOVATION_READOUT_V1'
PATHS={k:OUTDIR+'/'+v for k,v in {'output':'SUCCESS.json','failure':'FAILURE.json','progress':'progress.json','lock':'run.lock'}.items()}
PRIORS={
 'paired':{'path':'validation_results/BRAN_ANCHOR_ABLATION_V2/SUCCESS.json','sha256':'3e274f9920e272634442cfb90a3e794d4f15c5505a8c30345f98fb375de79c18'},
 'cbc':{'path':'validation_results/BRAN_FROZEN_CBC_READOUT_V1/SUCCESS.json','sha256':'67b2887926a0edec5a23abe866710d0f43eff3ef4430278d90228f4b8ccbe4a7'}}
PRIOR_PROTOCOL_SHA='8ad020eaf5f48abc8c6a771dbe7edfaa64a948b66ffa3f1c12fc6daee57b1b7f'
PARAMETERS={'encoder':'unchanged_v2_outer_train_only','encoder_steps':1500,'encoder_seed':1701,
 'outer_folds':5,'inner_folds':5,'state_dimension':192,'blood_fields':33,'clinical_fields':43,
 'screening_arms':list(ARMS),'cbc_arms':list(CBC_ARMS),'cbc_fields':list(CBC_FIELDS),'cbc_scenarios':list(SCENARIOS),
 'profiles':[list(x) for x in PROFILES],'screening_C':list(C_GRID),'cbc_alphas':list(ALPHA_GRID),
 'bootstrap_draws':1000,'bootstrap_seed':91501,'minimum_valid_draws':950,
 'screening_family':list(FAMILY),'family_lower_percentile':100*.05/3,
 'screening_metric':'observed_count_weighted_outer_fold_auroc_then_equal_endpoint_macro',
 'cbc_metric':'equal_outer_fold_normalized_mse_then_equal_9_field_macro',
 'cbc_primary':'retina_plus_other_clinical_innovation_selected_minus_state_nested',
 'canary_tolerance':1e-8,'head_selection':'inner_head_only_outer_encoder_not_inner_refit',
 'inference':'common_patient_fold_stratified_fixed_fit_reused_development',
 'selected_head_uncertainty':'means_only_no_predictive_intervals'}
NEW_FILES={
 'bran_innovation_experiment_v1.py','test_bran_innovation_experiment_v1.py',
 'bran_innovation_readout_v1.py','test_bran_innovation_readout_v1.py',
 'bran_innovation_aggregate_v1.py','test_bran_innovation_aggregate_v1.py',
 'run_bran_innovation_readout_v1.py','test_run_bran_innovation_readout_v1.py',
 'audit_bran_frozen_cbc_readout_v1.py','BRAN_CONTINUATION_PLAN_2026-09-05.md'}
PHASES={'protocol','validated','context','training','screening','completion','fold_complete','aggregate','replay','writing','completed'}
PRIVACY={'patient_rows_ids_images_predictions_embeddings_states_draws_serialized':False,
         'patient_processing_local_only':True,'official_test_loaded':False,'encoder_weights_serialized':False,
         'image_generation_claimed':False,'new_subtypes_claimed':False,'external_validation_claimed':False}

class ExperimentError(RuntimeError):pass
def require(ok):
 if not ok:raise ExperimentError('authentication_or_contract_failed_contents_withheld')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_x(path,obj):
 path=Path(path);payload=json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n';path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:f.write(payload)
def exact_keys(value,keys):require(isinstance(value,dict) and set(value)==set(keys))
def num(x,low=-math.inf,high=math.inf):require(type(x) in (int,float) and math.isfinite(x) and low<=x<=high)
def interval(x,low=-math.inf,high=math.inf):
 require(type(x) is list and len(x)==2)
 for v in x:num(v,low,high)
 require(x[0]<=x[1])
def prior_protocol(root):
 path=root/'BRAN_FROZEN_CBC_READOUT_PROTOCOL_V1.json';require(sha(path)==PRIOR_PROTOCOL_SHA)
 return previous.validate_protocol(root,path)
def authenticate_priors(root):
 for binding in PRIORS.values():require(sha(root/binding['path'])==binding['sha256'])
 require(paired_audit.audit(root)['status']=='authenticated_success')
 require(cbc_audit.audit(root)['status']=='authenticated_success')

def freeze(root=ROOT):
 root=Path(root);require(not(root/PROTOCOL_NAME).exists());old=prior_protocol(root);authenticate_priors(root)
 names=set(old['expected_hashes'])|NEW_FILES
 p={'schema_version':SCHEMA,'status':'frozen_before_execution','created_utc':datetime.now(timezone.utc).isoformat(),
    'parameters':PARAMETERS,'authentication':old['authentication'],'scope':old['scope'],'data_roots':old['data_roots'],
    'prior_protocol_sha256':PRIOR_PROTOCOL_SHA,'prior_results':PRIORS,'paths':PATHS,
    'expected_hashes':{n:sha(root/n) for n in sorted(names)},'privacy':PRIVACY,
    'claim_limit':'Exploratory repeated-development readout experiment; no new information, encoder, external validation, calibrated generation or disease subtypes.'}
 require(not any((root/v).exists() for v in PATHS.values()));write_x(root/PROTOCOL_NAME,p)
 validate_protocol(root)
 return {'status':'frozen','protocol_sha256':sha(root/PROTOCOL_NAME),'components':len(names)}

def validate_protocol(root=ROOT):
 root=Path(root).resolve();p=json.loads((root/PROTOCOL_NAME).read_text());old=prior_protocol(root)
 exact_keys(p,{'schema_version','status','created_utc','parameters','authentication','scope','data_roots','prior_protocol_sha256','prior_results','paths','expected_hashes','privacy','claim_limit'})
 require(p['schema_version']==SCHEMA and p['status']=='frozen_before_execution' and p['parameters']==PARAMETERS)
 require(p['authentication']==old['authentication'] and p['scope']==old['scope'] and p['data_roots']==old['data_roots'])
 require(p['prior_protocol_sha256']==PRIOR_PROTOCOL_SHA and p['prior_results']==PRIORS and p['paths']==PATHS and p['privacy']==PRIVACY)
 exact_keys(p['expected_hashes'],set(old['expected_hashes'])|NEW_FILES)
 for name,digest in p['expected_hashes'].items():
  path=(root/name).resolve();require(path.parent==root and type(digest) is str and len(digest)==64 and sha(path)==digest)
 for binding in PRIORS.values():require(sha(root/binding['path'])==binding['sha256'])
 for value in PATHS.values():require((root/value).resolve().parent==(root/OUTDIR).resolve() and (root/value).resolve().is_relative_to(root/'validation_results'))
 return p

def validate_result(r,sources):
 exact_keys(r,{'schema','screening','cbc','diagnostics','patient_rows_or_predictions_serialized','selected_head_intervals_claimed'})
 require(r['schema']=='bran-innovation-readout-v1' and r['patient_rows_or_predictions_serialized'] is False and r['selected_head_intervals_claimed'] is False)
 d=r['diagnostics'];exact_keys(d,{'legacy_and_raw_head_refits','innovation_head_refits','innovation_candidates_rejected_nonconvergence','legacy_candidates_rejected_nonconvergence','legacy_candidates_rejected_incomplete_inner_support','completion_head_refits','encoder_weights_unchanged_after_heads'})
 require(d['encoder_weights_unchanged_after_heads'] is True)
 for name,expected in (('legacy_and_raw_head_refits',1040),('innovation_head_refits',260),('completion_head_refits',540)):require(type(d[name]) is int and d[name]==expected)
 for name in ('innovation_candidates_rejected_nonconvergence','legacy_candidates_rejected_nonconvergence','legacy_candidates_rejected_incomplete_inner_support'):require(type(d[name]) is int and 0<=d[name]<=20000)
 s=r['screening'];exact_keys(s,{'endpoint_results','macro_paired_deltas','single_view_family','bootstrap_draws_retained'});require(s['bootstrap_draws_retained'] is False)
 endpoints=s['endpoint_results'];exact_keys(endpoints,sources);require(len(endpoints)==26)
 contrasts={'innovation_selected_minus_'+a for a in COMPARATORS}
 for e in endpoints.values():
  exact_keys(e,{'arms','paired_deltas'});exact_keys(e['arms'],ARMS);exact_keys(e['paired_deltas'],contrasts)
  for m in e['arms'].values():
   exact_keys(m,{'auroc','logloss','ci95'});num(m['auroc'],0,1);num(m['logloss'],0);interval(m['ci95'],0,1)
  for name,m in e['paired_deltas'].items():
   exact_keys(m,{'auroc_difference','ci95'});num(m['auroc_difference'],-1,1);interval(m['ci95'],-1,1)
   right=name.split('_minus_')[1];require(abs(m['auroc_difference']-(e['arms']['innovation_selected']['auroc']-e['arms'][right]['auroc']))<1e-10)
 exact_keys(s['macro_paired_deltas'],contrasts)
 for name,m in s['macro_paired_deltas'].items():
  exact_keys(m,{'mean_endpoint_auroc_difference','ci95'});num(m['mean_endpoint_auroc_difference'],-1,1);interval(m['ci95'],-1,1)
  require(abs(m['mean_endpoint_auroc_difference']-sum(e['paired_deltas'][name]['auroc_difference'] for e in endpoints.values())/26)<1e-10)
 fam=s['single_view_family'];exact_keys(fam,{'comparators','bonferroni_one_sided_95_lower','all_lower_above_zero'})
 require(fam['comparators']==list(FAMILY));exact_keys(fam['bonferroni_one_sided_95_lower'],FAMILY)
 for name,v in fam['bonferroni_one_sided_95_lower'].items():
  num(v,-1,1);require(v<=s['macro_paired_deltas']['innovation_selected_minus_'+name]['ci95'][0]+1e-12)
 require(type(fam['all_lower_above_zero']) is bool and fam['all_lower_above_zero']==all(v>0 for v in fam['bonferroni_one_sided_95_lower'].values()))
 exact_keys(r['cbc'],SCENARIOS);cbc_summary={}
 cc={'innovation_selected_minus_state_nested','innovation_selected_minus_raw_nested'}
 for scenario,c in r['cbc'].items():
  exact_keys(c,{'field_metrics','field_paired_deltas','macro_paired_deltas'});exact_keys(c['field_metrics'],CBC_FIELDS)
  for field,arms in c['field_metrics'].items():
   exact_keys(arms,CBC_ARMS)
   for m in arms.values():
    exact_keys(m,{'normalized_mae','normalized_mse'});num(m['normalized_mae'],0);num(m['normalized_mse'],0)
    require(m['normalized_mae']**2<=m['normalized_mse']+1e-8)
  exact_keys(c['field_paired_deltas'],cc);exact_keys(c['macro_paired_deltas'],cc)
  for name,fields in c['field_paired_deltas'].items():
   exact_keys(fields,CBC_FIELDS);right=name.split('_minus_')[1]
   for field,m in fields.items():
    exact_keys(m,{'normalized_mse_difference','ci95'});num(m['normalized_mse_difference']);interval(m['ci95'])
    arms=c['field_metrics'][field];require(abs(m['normalized_mse_difference']-(arms['innovation_selected']['normalized_mse']-arms[right]['normalized_mse']))<1e-10)
   m=c['macro_paired_deltas'][name];exact_keys(m,{'mean_9_normalized_mse_difference','ci95'});num(m['mean_9_normalized_mse_difference']);interval(m['ci95'])
   require(abs(m['mean_9_normalized_mse_difference']-sum(v['normalized_mse_difference'] for v in fields.values())/9)<1e-10)
  cbc_summary[scenario]={'macro9_normalized_mse':{a:sum(m[a]['normalized_mse'] for m in c['field_metrics'].values())/9 for a in CBC_ARMS},'paired_macro_deltas':c['macro_paired_deltas']}
 return {'screening_macro_auroc':{a:sum(e['arms'][a]['auroc'] for e in endpoints.values())/26 for a in ARMS},
         'screening_paired_macro_deltas':s['macro_paired_deltas'],'single_view_family':fam,'cbc':cbc_summary}

def replay(result,old_screen,old_cbc):
 for source,e in result['screening']['endpoint_results'].items():
  for arm,old in (('v2both_legacy','v2both'),('v2clinical_legacy','v2clinical'),('v2retinal_legacy','v2retinal')):
   for metric in ('auroc','logloss'):require(abs(e['arms'][arm][metric]-old_screen['screening']['endpoint_results'][source]['arms'][old][metric])<=1e-8)
 for scenario,c in result['cbc'].items():
  for field,m in c['field_metrics'].items():
   for arm,old in (('state_nested','state_nested_ridge'),('raw_nested','raw_nested_ridge'),('age_nested','age_nested_ridge'),('median','median')):
    for metric in ('normalized_mae','normalized_mse'):require(abs(m[arm][metric]-old_cbc['cbc'][scenario]['field_metrics'][field][old][metric])<=1e-8)
 return {'screening_3x26_auroc_and_logloss':True,'cbc_4x9x3_mae_and_mse':True,'tolerance':1e-8}

def validate_report(r,p,digest):
 exact_keys(r,{'schema_version','status','protocol_sha256','code_hashes','source_hashes','support_receipt_sha256','fold_hashes','scope','parameters','privacy','replay','result'})
 a=p['authentication'];require(r['schema_version']==SCHEMA and r['status']=='completed_aggregate_only' and r['protocol_sha256']==digest and r['code_hashes']==p['expected_hashes'])
 require(r['source_hashes']==a['canonical_source_hashes'] and r['support_receipt_sha256']==a['support_receipt_sha256'] and r['fold_hashes']=={'outer':a['outer_fold_sha256'],'inner':a['inner_fold_sha256']})
 require(r['scope']=={'patient_count':1928,'endpoint_count':26,'official_test_loaded':False} and r['parameters']==PARAMETERS and r['privacy']==PRIVACY)
 exact_keys(r['replay'],{'screening_3x26_auroc_and_logloss','cbc_4x9x3_mae_and_mse','tolerance'})
 require(r['replay']['screening_3x26_auroc_and_logloss'] is True and r['replay']['cbc_4x9x3_mae_and_mse'] is True and r['replay']['tolerance']==1e-8)
 return validate_result(r['result'],p['scope']['eligible_source_codes'])

def run(root=ROOT):
 root=Path(root);p={};fd=None;quiet=None;phase='protocol';paths={k:root/v for k,v in PATHS.items()}
 try:
  p=validate_protocol(root);authenticate_priors(root)
  require(not paths['output'].exists() and not paths['failure'].exists() and not paths['progress'].exists())
  fd=paired.base._acquire_lock(paths['lock']);paired.base._atomic_progress(paths['progress'],'validated')
  quiet=paired.base._quiet_sensitive_block();quiet.__enter__();phase='context'
  from patient_atlas_v6_2_expanded_endpoint_evaluation import FROZEN_SUPPORT_RECEIPT_NAME,EXACT_OUTER_FOLD_HASH,EXACT_INNER_FOLD_ASSIGNMENT_SHA256,load_eligible_support_receipt,validate_support_against_observed
  from run_patient_atlas_v6_2_expanded_endpoint_evaluation import _load_actual_v6_2_context
  support=load_eligible_support_receipt(root/FROZEN_SUPPORT_RECEIPT_NAME,project_root=root)
  context=_load_actual_v6_2_context(root=root,support=support,**{k:Path(v) for k,v in p['data_roots'].items()})
  outer=np.asarray(context['outer_assignment']);labels=context['labels_by_source'];observed=context['observed_by_source'];a=p['authentication']
  validate_support_against_observed(support,labels,observed,outer)
  require(len(outer)==1928 and outer.dtype.kind in 'iu' and set(outer.tolist())==set(range(5)) and len(support.eligible_sources)==26)
  require(a['outer_fold_sha256']==EXACT_OUTER_FOLD_HASH and a['canonical_source_hashes']==dict(context.get('source_hashes',{})) and a['support_receipt_sha256']==support.receipt_sha256)
  require(set(support.eligible_sources)==set(p['scope']['eligible_source_codes']) and len(a['inner_fold_sha256'])==5)
  c,cm,elig,r,rm,names=paired._actual_arrays(root,context);elig[:,48:]=False;inners=[]
  for f in range(5):
   inner,digest=paired.base._inner_context(context,np.flatnonzero(outer!=f),f)
   require(digest==EXACT_INNER_FOLD_ASSIGNMENT_SHA256[f]==a['inner_fold_sha256'][f]);inners.append(inner)
  def progress(name,fold):
   nonlocal phase
   require(name in PHASES and type(fold) is int and 0<=fold<5);phase=name;paired.base._atomic_progress(paths['progress'],name,fold)
  phase='training'
  result=evaluate(c,cm,elig,r,rm,names,np.asarray(context['raw_cohort'].ages),outer,inners,labels,observed,
                  p['scope']['eligible_source_codes'],progress_callback=progress)
  phase='aggregate';paired.base._atomic_progress(paths['progress'],'aggregate');validate_result(result,p['scope']['eligible_source_codes'])
  phase='replay';paired.base._atomic_progress(paths['progress'],'replay');authenticate_priors(root)
  checked=replay(result,json.loads((root/PRIORS['paired']['path']).read_text()),json.loads((root/PRIORS['cbc']['path']).read_text()))
  report={'schema_version':SCHEMA,'status':'completed_aggregate_only','protocol_sha256':sha(root/PROTOCOL_NAME),
          'code_hashes':p['expected_hashes'],'source_hashes':a['canonical_source_hashes'],'support_receipt_sha256':a['support_receipt_sha256'],
          'fold_hashes':{'outer':a['outer_fold_sha256'],'inner':a['inner_fold_sha256']},
          'scope':{'patient_count':1928,'endpoint_count':26,'official_test_loaded':False},
          'parameters':PARAMETERS,'privacy':PRIVACY,'replay':checked,'result':result}
  phase='writing';validate_protocol(root);validate_report(report,p,sha(root/PROTOCOL_NAME));paired.base._atomic_progress(paths['progress'],'writing')
  write_x(paths['output'],report);paired.base._atomic_completed(paths['progress']);return {'status':'completed_aggregate_only','artifact_sha256':sha(paths['output'])}
 except Exception as exc:
  frames=[{'file':Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(exc.__traceback__) if Path(x.filename).resolve().parent==root.resolve() and Path(x.filename).name in p.get('expected_hashes',{})][:8]
  failure={'schema_version':SCHEMA,'status':'failed','phase':phase,'error_class':type(exc).__name__,'bound_code_frames':frames,'exception_text_serialized':False,'patient_rows_or_ids_serialized':False}
  if fd is not None and not paths['output'].exists() and not paths['failure'].exists():write_x(paths['failure'],failure)
  return {'status':'execution_failed','phase':phase,'error_class':type(exc).__name__,'exception_contents_emitted':False}
 finally:
  if quiet is not None:quiet.__exit__(None,None,None)
  if fd is not None:
   os.close(fd)
   try:paths['lock'].unlink()
   except FileNotFoundError:pass

def audit(root=ROOT):
 root=Path(root);p=validate_protocol(root);paths={k:root/v for k,v in PATHS.items()};success,failure=paths['output'],paths['failure']
 require(not(success.exists() and failure.exists()))
 if not(success.exists() or failure.exists()):
  out={'status':'no_terminal_artifact','lock_exists':paths['lock'].exists()}
  if paths['progress'].exists():
   q=json.loads(paths['progress'].read_text());require(isinstance(q,dict) and set(q)<={'status','phase','fold'} and q.get('status') in {'running','completed'} and q.get('phase') in PHASES)
   if 'fold' in q:require(type(q['fold']) is int and 0<=q['fold']<5)
   out['progress']=q
  return out
 path=success if success.exists() else failure;r=json.loads(path.read_text())
 if failure.exists():
  exact_keys(r,{'schema_version','status','phase','error_class','bound_code_frames','exception_text_serialized','patient_rows_or_ids_serialized'})
  require(r['schema_version']==SCHEMA and r['status']=='failed' and r['phase'] in PHASES and r['exception_text_serialized'] is False and r['patient_rows_or_ids_serialized'] is False)
  require(type(r['error_class']) is str and r['error_class'].isidentifier() and len(r['error_class'])<90 and type(r['bound_code_frames']) is list and len(r['bound_code_frames'])<=8)
  for f in r['bound_code_frames']:
   exact_keys(f,{'file','line'});require(f['file'] in p['expected_hashes'] and type(f['line']) is int and 1<=f['line']<100000)
  return {'status':'authenticated_execution_failure','artifact_sha256':sha(path),'phase':r['phase'],'error_class':r['error_class'],'code_frames':r['bound_code_frames']}
 summary=validate_report(r,p,sha(root/PROTOCOL_NAME))
 return {'status':'authenticated_success','artifact_sha256':sha(path),'protocol_sha256':sha(root/PROTOCOL_NAME),
         'component_count':len(p['expected_hashes']),'scope':r['scope'],'folds_authenticated':True,'replay_authenticated':True,**summary,
         'claim_limit':'Repeated development, fixed-fit intervals; three macro screening comparisons adjusted only. No calibrated generation, external validation or new subtypes.'}

if __name__=='__main__':
 try:print(json.dumps({'freeze':freeze,'run':run,'audit':audit}[sys.argv[1]](),sort_keys=True,allow_nan=False))
 except Exception as exc:
  print(json.dumps({'status':'blocked_without_disclosure','error_class':type(exc).__name__,'contents_emitted':False}));raise SystemExit(1)
