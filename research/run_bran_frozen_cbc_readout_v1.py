"""Aggregate-only fixed CBC conditional-mean readout diagnostic.

No final-fit checkpoint is used: a V2 encoder is freshly recreated per outer
fold, then remains frozen while prespecified local completion heads are fit.
"""
from __future__ import annotations
import hashlib, json, os, traceback
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from sklearn.linear_model import Ridge

import run_bran_overnight_diagnostic_v1 as base
import run_bran_anchor_ablation_v2 as paired
from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
from bran_frozen_cbc_ridge_v1 import fit_frozen_cbc_ridge, FrozenCBCRidgeError

SCHEMA="bran-frozen-cbc-readout-v1"
CBC_FIELDS=("hct","hemoglobin","rbc","mcv","mch","mchc","rdw","plt","wbc")
SCENARIOS=("retina_plus_other_clinical","retina_absent_other_clinical","retina_age_only")
ARMS=("decoder_mc","state_unscaled_ridge1","state_scaled_ridge1","state_nested_ridge","raw_nested_ridge","age_nested_ridge","median")
PARAMETERS={"outer_folds":5,"inner_folds":5,"encoder":"frozen_v2_per_outer_fold_seed_1701_plus_fold","encoder_steps":1500,"state_dim_plus_age":193,"conditional_mean_samples":64,"mc_seed":91501,"ridge_alphas":[.01,.1,1.,10.,100.,1000.,10000.],"bootstrap_samples":1000,"bootstrap_seed":91501,"scenarios":list(SCENARIOS),"arms":list(ARMS),"primary":"scenario_A_state_nested_ridge_minus_state_unscaled_ridge1_mean_9_normalized_mse"}
_EXPECTED_PATIENT_COUNT=1928
class ReadoutError(RuntimeError): pass
def _hash(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def _encoder_hash(model:Any)->str:
 h=hashlib.sha256()
 for k,v in model.state_dict().items(): h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def validate_protocol(root:str|Path,path:str|Path)->Mapping[str,Any]:
 root,path=Path(root).resolve(),Path(path).resolve();p=json.loads(path.read_text())
 if not isinstance(p,dict) or p.get("schema_version")!=SCHEMA or p.get("status")!="frozen_before_execution" or p.get("parameters")!=PARAMETERS:raise ReadoutError("protocol_invalid")
 req={"run_bran_frozen_cbc_readout_v1.py","bran_patient_state_anchor_v2.py","bran_frozen_cbc_ridge_v1.py"}; hashes=p.get("expected_hashes")
 if not isinstance(hashes,dict) or not req<=set(hashes):raise ReadoutError("protocol_hashes_missing")
 for n,d in hashes.items():
  q=(root/n).resolve()
  if q.parent!=root or not isinstance(d,str) or len(d)!=64 or not q.is_file() or _hash(q)!=d:raise ReadoutError("protocol_hash_mismatch")
 paths=p.get("paths",{}); auth=p.get("authentication",{}); prior=p.get("prior_paired_result",{}); areq={"outer_fold_sha256","inner_fold_sha256","support_receipt_sha256","canonical_source_hashes"}
 if set(paths)!={"output","failure","progress","lock"} or not isinstance(auth,dict) or not areq<=set(auth) or not isinstance(prior,dict) or set(prior)!={"path","sha256"}:raise ReadoutError("protocol_binding_invalid")
 prior_path=(root/str(prior["path"])).resolve()
 if not prior_path.is_relative_to(root/"validation_results") or not isinstance(prior["sha256"],str) or len(prior["sha256"])!=64 or not prior_path.is_file() or _hash(prior_path)!=prior["sha256"]:raise ReadoutError("prior_binding_invalid")
 bound_paths={k:(root/str(v)).resolve() for k,v in paths.items()}
 if len(set(bound_paths.values()))!=4 or any(not q.is_relative_to(root/"validation_results") for q in bound_paths.values()) or any(q.parent!=bound_paths["output"].parent for q in bound_paths.values()):raise ReadoutError("output_binding_invalid")
 p["_prior_path"]=prior_path
 p["_paths"]=bound_paths;return p
def _safe_write(path:Path,obj:Mapping[str,Any])->None:
 text=json.dumps(obj,sort_keys=True,allow_nan=False,indent=2)+"\n";path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("x") as f:f.write(text)
def _precheck(cm,idx,folds,inners):
 for f in range(5):
  tr,te=folds!=f,folds==f
  if np.any(cm[te][:,idx].sum(0)<10):raise ReadoutError("outer_cbc_support_below_10")
  inner=inners[f]
  for g in range(5):
   fit=(inner!=g);score=(inner==g)
   if np.any(cm[tr][:,idx][fit].sum(0)<2) or np.any(cm[tr][:,idx][score].sum(0)<10):raise ReadoutError("inner_cbc_support_invalid")
def _scenario_predictions(model,c,cm,r,rm,age,names,tr,te,inner,scenario,seed):
 idx=np.asarray([tuple(names).index(x) for x in CBC_FIELDS]);ic,im,ir,irm,score=paired._cbc_scenario_inputs(c,cm,r,rm,idx,scenario)
 with torch.no_grad():
  st=model.encode(torch.tensor(ic,dtype=torch.float32),torch.tensor(im),torch.tensor(ir[:,None],dtype=torch.float32),torch.tensor(irm[:,None]),torch.tensor(age,dtype=torch.float32))
  torch.manual_seed(seed); dec=model.sample_clinical(st,torch.tensor(age,dtype=torch.float32),samples=64)["continuous_mean"].numpy().mean(0)
 state=np.c_[st.mean.numpy(),age];keep=np.ones(c.shape[1],bool);keep[idx]=False
 raw=np.c_[ic[:,keep],im[:,keep].astype(float),ir,age] if scenario!="retina_age_only" else np.c_[ir,age]
 y,obs=c[:,idx],score[:,idx]
 nested_state=fit_frozen_cbc_ridge(state[tr],y[tr],obs[tr],inner,state[te])
 scaled_one=fit_frozen_cbc_ridge(state[tr],y[tr],obs[tr],inner,state[te],fixed_alpha1=True)
 nested_raw=fit_frozen_cbc_ridge(raw[tr],y[tr],obs[tr],inner,raw[te])
 nested_age=fit_frozen_cbc_ridge(age[tr,None],y[tr],obs[tr],inner,age[te,None])
 unscaled=np.empty((len(te),9));median=np.empty_like(unscaled)
 for j in range(9):
  fit=obs[tr,j];unscaled[:,j]=Ridge(alpha=1.).fit(state[tr][fit],y[tr,j][fit]).predict(state[te]);median[:,j]=np.median(y[tr,j][fit])
 return {"decoder_mc":dec[te][:,idx],"state_unscaled_ridge1":unscaled,"state_scaled_ridge1":scaled_one.predictions,"state_nested_ridge":nested_state.predictions,"raw_nested_ridge":nested_raw.predictions,"age_nested_ridge":nested_age.predictions,"median":median},y[te],obs[te]
def _metric(y,p,m,folds,counts=None):
 point=np.mean([np.mean((p[(folds==f)&m]-y[(folds==f)&m])**2) for f in range(5)])
 if counts is None:return float(point)
 draw=[]
 for w in counts:
  vals=[]
  for f in range(5):
   z=(folds==f)&m;ww=w[z];err=(p[z]-y[z])**2;den=np.sum(ww);vals.append(float(np.sum(ww*err)/den) if den else np.nan)
  draw.append(np.mean(vals))
 return float(point),np.asarray(draw)
def _summary(y,pred,mask,folds,counts):
 if y.shape!=mask.shape or y.shape!=(len(folds),9) or set(pred)!=set(ARMS) or not np.isfinite(y[mask]).all() or any(v.shape!=y.shape or not np.isfinite(v).all() for v in pred.values()):raise ReadoutError("invalid_oof_values")
 out={};draw={}
 for j,field in enumerate(CBC_FIELDS):
  out[field]={"status":"scored_normalized_units_only"};draw[field]={}
  for arm in ARMS:
   m=mask[:,j];mae=np.mean([np.mean(np.abs(pred[arm][(folds==f)&m,j]-y[(folds==f)&m,j])) for f in range(5)])
   mse,d=_metric(y[:,j],pred[arm][:,j],m,folds,counts);out[field][arm]={"normalized_mae":float(mae),"normalized_mse":mse};draw[field][arm]=d
 contrasts={}
 for left,right in (("state_nested_ridge","state_unscaled_ridge1"),("state_nested_ridge","decoder_mc"),("state_nested_ridge","raw_nested_ridge")):
  diffs=np.stack([draw[x][left]-draw[x][right] for x in CBC_FIELDS]);valid=np.isfinite(diffs).all(0)
  if valid.sum()<950:raise ReadoutError("insufficient_bootstrap_valid_draws")
  contrasts[left+"_minus_"+right]={"mean_9_normalized_mse_difference":float(np.mean([out[x][left]["normalized_mse"]-out[x][right]["normalized_mse"] for x in CBC_FIELDS])),"ci95":[float(np.percentile(diffs.mean(0)[valid],2.5)),float(np.percentile(diffs.mean(0)[valid],97.5))]}
  for field in CBC_FIELDS:
   d=draw[field][left]-draw[field][right];ok=np.isfinite(d)
   if ok.sum()<950:raise ReadoutError("insufficient_bootstrap_valid_draws")
   out[field][left+"_minus_"+right]={"normalized_mse_difference":float(out[field][left]["normalized_mse"]-out[field][right]["normalized_mse"]),"ci95":[float(np.percentile(d[ok],2.5)),float(np.percentile(d[ok],97.5))]}
 return out,contrasts
def _validate_prior(prior_path, expected, current):
 if not prior_path.is_file() or _hash(prior_path)!=expected:raise ReadoutError("prior_paired_receipt_hash_mismatch")
 try: prior=json.loads(prior_path.read_text())
 except Exception as e:raise ReadoutError("prior_paired_receipt_invalid") from e
 for q in SCENARIOS:
  for field in CBC_FIELDS:
   old=prior["cbc"]["v2"][q][field];new=current[q]["field_metrics"][field]
   if old.get("status")!="scored_normalized_units_only":raise ReadoutError("prior_paired_receipt_invalid")
   for arm,key in (("decoder_mc","posterior_predictive_mc_mean"),("state_unscaled_ridge1","state_age_ridge")):
    for metric in ("mae","mse"):
     a,b=float(new[arm]["normalized_"+metric]),float(old[key+"_"+metric])
     if not np.isfinite(a) or not np.isfinite(b) or abs(a-b)>1e-8:raise ReadoutError("prior_paired_replay_mismatch")
 return True
def run_bran_frozen_cbc_readout_v1(*,project_root,protocol_path,output_path,failure_path,progress_path,lock_path,dataset_root,clinical_project_root):
 root=Path(project_root).resolve();output,failure,progress,lock=map(lambda x:Path(x).resolve(),(output_path,failure_path,progress_path,lock_path));fd=None;quiet=None;phase="protocol";protocol={}
 try:
  protocol=validate_protocol(root,protocol_path);paths=protocol["_paths"]
  if paths!={"output":output,"failure":failure,"progress":progress,"lock":lock} or len(set(paths.values()))!=4 or any(x.parent!=output.parent for x in paths.values()) or output.exists() or failure.exists():raise ReadoutError("exclusive_paths_invalid")
  fd=base._acquire_lock(lock);base._atomic_progress(progress,"validated");quiet=base._quiet_sensitive_block();quiet.__enter__()
  from patient_atlas_v6_2_expanded_endpoint_evaluation import FROZEN_SUPPORT_RECEIPT_NAME,EXACT_OUTER_FOLD_HASH,EXACT_INNER_FOLD_ASSIGNMENT_SHA256,load_eligible_support_receipt,validate_support_against_observed
  from run_patient_atlas_v6_2_expanded_endpoint_evaluation import _load_actual_v6_2_context
  phase="context";support=load_eligible_support_receipt(root/FROZEN_SUPPORT_RECEIPT_NAME,project_root=root);context=_load_actual_v6_2_context(root=root,dataset_root=dataset_root,clinical_project_root=clinical_project_root,support=support);auth=protocol["authentication"]
  folds=np.asarray(context["outer_assignment"],int);labels,observed=context["labels_by_source"],context["observed_by_source"];validate_support_against_observed(support,labels,observed,folds)
  if auth["outer_fold_sha256"]!=EXACT_OUTER_FOLD_HASH or auth["canonical_source_hashes"]!=dict(context.get("source_hashes",{})) or auth["support_receipt_sha256"]!=getattr(support,"receipt_sha256",None) or len(auth["inner_fold_sha256"])!=5 or len(support.eligible_sources)!=26 or len(folds)!=_EXPECTED_PATIENT_COUNT or set(folds)!={0,1,2,3,4}:raise ReadoutError("authentication_failed")
  c0,cm0,elig,r0,rm,names=paired._actual_arrays(root,context);elig[:,48:]=False;idx=np.asarray([tuple(names).index(x) for x in CBC_FIELDS]);inners=[]
  for f in range(5):
   inner,h=base._inner_context(context,np.flatnonzero(folds!=f),f)
   if h!=EXACT_INNER_FOLD_ASSIGNMENT_SHA256[f] or h!=auth["inner_fold_sha256"][f]:raise ReadoutError("inner_hash_failed")
   inners.append(inner)
  _precheck(cm0,idx,folds,inners);pred={q:{a:np.full((len(folds),9),np.nan) for a in ARMS} for q in SCENARIOS};truth=np.full((len(folds),9),np.nan);mask=np.zeros((len(folds),9),bool);changed=[]
  for f in range(5):
   phase="training";base._atomic_progress(progress,"training",f);tr,te=np.flatnonzero(folds!=f),np.flatnonzero(folds==f);trans=base.FoldTransform(c0,cm0,elig,r0,rm,np.asarray(context["raw_cohort"].ages),tr);c,cm,r,age=trans.apply(c0,cm0,elig,r0,rm,np.asarray(context["raw_cohort"].ages));model=paired._train(BRANClinicalAnchorV2,c,cm,r,rm,age,tr,1701+f);before=_encoder_hash(model);phase="completion";base._atomic_progress(progress,"completion",f)
   for q in SCENARIOS:
    pp,yy,mm=_scenario_predictions(model,c,cm,r,rm,age,names,tr,te,inners[f],q,91501+f)
    for a in ARMS:pred[q][a][te]=pp[a]
    truth[te],mask[te]=yy,mm
   changed.append(before==_encoder_hash(model));base._atomic_progress(progress,"fold_complete",f)
  phase="aggregate";base._atomic_progress(progress,"aggregate");counts=paired._shared_bootstrap_counts(folds);result={};
  for q in SCENARIOS: fields,contrasts=_summary(truth,pred[q],mask,folds,counts);result[q]={"field_metrics":fields,"paired_mse_contrasts":contrasts}
  if not all(changed):raise ReadoutError("encoder_parameters_changed")
  phase="replay";base._atomic_progress(progress,"replay");replay=_validate_prior(protocol["_prior_path"],protocol["prior_paired_result"]["sha256"],result)
  report={"schema_version":SCHEMA,"status":"completed_aggregate_only","protocol_sha256":_hash(Path(protocol_path)),"code_hashes":protocol["expected_hashes"],"source_hashes":dict(context.get("source_hashes",{})),"support_receipt_sha256":getattr(support,"receipt_sha256",None),"scope":{"patient_count":1928,"endpoint_count":26,"official_test_loaded":False},"fold_hashes":{"outer":EXACT_OUTER_FOLD_HASH,"inner":auth["inner_fold_sha256"]},"parameters":PARAMETERS,"scenarios":{"retina_plus_other_clinical":"primary","retina_absent_other_clinical":"descriptive","retina_age_only":"descriptive"},"cbc":result,"prior_paired_replay_authenticated":replay,"encoder_parameters_unchanged_before_after_heads":True,"patient_rows_or_ids_serialized":False,"models_or_oof_predictions_serialized":False,"uncertainty_claim":"conditional_mean_only_no_calibration_or_joint_generation_claim"};phase="writing";base._atomic_progress(progress,"writing");_safe_write(output,report);base._atomic_completed(progress);return report
 except Exception as e:
  allowed={str((root/n).resolve()) for n in protocol.get("expected_hashes",{})};frames=[{"file":Path(x.filename).name,"line":x.lineno} for x in traceback.extract_tb(e.__traceback__) if str(Path(x.filename).resolve()) in allowed][:8];safe={"schema_version":SCHEMA,"status":"failed","phase":phase,"error_class":type(e).__name__,"bound_code_frames":frames,"exception_text_serialized":False,"patient_rows_or_ids_serialized":False}
  if fd is not None and not output.exists() and not failure.exists():_safe_write(failure,safe)
  return safe
 finally:
  if quiet is not None:quiet.__exit__(None,None,None)
  if fd is not None:
   os.close(fd)
   try:lock.unlink()
   except FileNotFoundError:pass
