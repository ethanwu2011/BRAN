"""Fresh paired V1/V2 clinical-preservation development runner (aggregate-only)."""
from __future__ import annotations

import hashlib, json, os, traceback
from pathlib import Path
from typing import Any, Mapping
import numpy as np

import run_bran_overnight_diagnostic_v1 as base
from bran_patient_state_prototype_v1 import BRANPatientStatePrototypeV1, PatientStateConfig
from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2, CLINICAL_ANCHOR_WEIGHT

SCHEMA = "bran-anchor-ablation-v2"
_EXPECTED_PATIENT_COUNT = 1928  # overridden only by isolated synthetic unit tests
ARM_NAMES = ("age", "rawclinical", "rawretina", "rawconcat", "v1clinical", "v1retinal", "v1both", "v2clinical", "v2retinal", "v2both")
COMPARISONS = ("v2both-v1both", "v2clinical-v1clinical", "v2retinal-v1retinal", "v2both-rawclinical", "v2both-rawconcat", "v1both-v1clinical", "v1both-v1retinal", "v2both-v2clinical", "v2both-v2retinal")
CBC_FIELDS = ("hct", "hemoglobin", "rbc", "mcv", "mch", "mchc", "rdw", "plt", "wbc")
ELIGIBLE_CONTINUOUS_INDICES = tuple(i for i in range(48) if i not in (8, 9, 20, 35, 36))
PARAMETERS = {**base.PARAMETERS, "anchor_weight": CLINICAL_ANCHOR_WEIGHT, "arms": list(ARM_NAMES), "comparisons": list(COMPARISONS), "head_refits": 1300}

class AblationError(RuntimeError): pass

def _hash(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def validate_protocol(root: str | Path, path: str | Path) -> Mapping[str, Any]:
    root, path = Path(root).resolve(), Path(path).resolve(); p = json.loads(path.read_text())
    if not isinstance(p, dict) or p.get("schema_version") != SCHEMA or p.get("status") != "frozen_before_execution" or p.get("parameters") != PARAMETERS: raise AblationError("protocol_invalid")
    hashes = p.get("expected_hashes", {}); required = {"run_bran_anchor_ablation_v2.py", "bran_patient_state_prototype_v1.py", "bran_patient_state_anchor_v2.py"}
    if not isinstance(hashes, dict) or not required <= set(hashes): raise AblationError("protocol_hashes_missing")
    for name, digest in hashes.items():
        target = (root / name).resolve()
        if target.parent != root or not target.is_file() or not isinstance(digest, str) or _hash(target) != digest: raise AblationError("protocol_hash_mismatch")
    paths = p.get("paths", {})
    if not isinstance(paths, dict) or set(paths) != {"output", "failure", "progress", "lock"}: raise AblationError("protocol_paths_missing")
    binding=p.get("authentication")
    required_binding={"outer_fold_sha256","inner_fold_sha256","support_receipt_sha256","canonical_source_hashes"}
    if not isinstance(binding,dict) or not required_binding <= set(binding) or set(binding)-required_binding-{"support_receipt"}: raise AblationError("protocol_bindings_missing")
    p["_paths"] = {k: (root / str(v)).resolve() for k,v in paths.items()}; return p

def _write_x(path: Path, obj: Mapping[str, Any]) -> None:
    text = json.dumps(obj, sort_keys=True, allow_nan=False, indent=2) + "\n"; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f: f.write(text)

def _actual_arrays(root,context):
    """Same extraction contract as V1, bound to the canonical local CBC names."""
    from patient_atlas_real_data import _load_registry
    cohort,raw=context['feature_cohort'],context['raw_cohort'];names,types=_load_registry(root)
    if tuple(types)!=('continuous',)*48+('binary',)*11 or not set(CBC_FIELDS)<=set(names[:48]):raise AblationError('clinical_schema_or_cbc_names_differ')
    if len(raw.patient_ids)!=_EXPECTED_PATIENT_COUNT or set(raw.split_labels)-{'train','val'}:raise AblationError('cohort_partition_mismatch')
    eye,mask=np.asarray(cohort.eye_embeddings),np.asarray(cohort.eye_observed_mask,bool)
    mask=mask & np.isfinite(eye).all(axis=2);clean=np.zeros_like(eye);clean[mask]=eye[mask]
    pooled=np.zeros((len(eye),eye.shape[2]),dtype=np.float64);present=mask.any(1)
    pooled[present]=clean[present].sum(1)/mask[present].sum(1,keepdims=True)
    return np.asarray(cohort.blood_values),np.asarray(cohort.blood_observed_mask,bool),np.asarray(cohort.blood_eligible_mask,bool)[None,:].repeat(len(eye),0),pooled,present,names

def _train(cls: type, c: np.ndarray, cm: np.ndarray, r: np.ndarray, rm: np.ndarray, age: np.ndarray, train: np.ndarray, seed: int, *, steps: int = 1500):
    import torch
    torch.set_num_threads(2); torch.manual_seed(seed); rng = np.random.default_rng(seed); model = cls(PatientStateConfig()); opt = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    ct, cmt, rt, rmt, at = torch.tensor(c, dtype=torch.float32), torch.tensor(cm), torch.tensor(r[:,None], dtype=torch.float32), torch.tensor(rm[:,None]), torch.tensor(age, dtype=torch.float32)
    for step in range(steps):
        ix = rng.choice(train, 96, replace=len(train)<96); vc, vr0 = base.masked_route(rng, cm[ix], rm[ix]); vr = torch.tensor(vr0[:,None]); vc = torch.tensor(vc)
        state = model.encode(ct[ix]*vc, vc, rt[ix]*vr[...,None], vr, at[ix])
        kw = dict(kl_weight=.001*min(1.,(step+1)/300), visible_weight=.1)
        if cls is BRANClinicalAnchorV2: kw["clinical_eligible_mask"] = cmt[ix]
        loss = model.objective(state, at[ix], ct[ix], cmt[ix], vc, rt[ix,0], rmt[ix].expand(-1,r.shape[1]), vr.expand(-1,r.shape[1]), **kw)["loss"]
        if not torch.isfinite(loss): raise AblationError("nonfinite_training")
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def _state_routes(model: Any, c: np.ndarray, cm: np.ndarray, r: np.ndarray, rm: np.ndarray, age: np.ndarray) -> Mapping[str,np.ndarray]:
    import torch
    with torch.no_grad():
        ct, rt, at = torch.tensor(c,dtype=torch.float32), torch.tensor(r[:,None],dtype=torch.float32), torch.tensor(age,dtype=torch.float32)
        def get(use_c,use_r):
            return model.encode(ct if use_c else torch.zeros_like(ct), torch.tensor(cm if use_c else np.zeros_like(cm)), rt if use_r else torch.zeros_like(rt), torch.tensor(rm[:,None] if use_r else np.zeros((len(rm),1),bool)), at).mean.numpy()
        return {"clinical":get(True,False),"retinal":get(False,True),"both":get(True,True)}

def _shared_bootstrap_counts(folds):
    rng=np.random.default_rng(91501); n=len(folds); counts=np.zeros((1000,n),np.int16)
    for f in range(5):
        ix=np.flatnonzero(folds==f)
        for d in range(1000): counts[d,ix]=np.bincount(rng.choice(len(ix),len(ix),True),minlength=len(ix))
    return counts

def _bootstrap(y,pred,obs,folds,counts=None):
    if counts is None: counts=_shared_bootstrap_counts(folds)
    point={a:base.fold_weighted_auc(y,p,obs,folds) for a,p in pred.items()}; draws={a:base._weighted_auc_draws(y,p,obs,folds,counts) for a,p in pred.items()}
    if not all(np.isfinite(list(point.values()))): raise AblationError("nonfinite_score")
    metrics={a:{"auroc":float(point[a]),"logloss":float(base.fold_weighted_logloss(y,pred[a],obs,folds)),"ci95":[float(np.nanpercentile(draws[a],2.5)),float(np.nanpercentile(draws[a],97.5))]} for a in pred}
    delta={}
    for name in COMPARISONS:
        left,right=name.split("-"); d=draws[left]-draws[right]; delta[name]={"auroc_delta":float(point[left]-point[right]),"ci95":[float(np.nanpercentile(d,2.5)),float(np.nanpercentile(d,97.5))]}
    return {"arms":metrics,"paired_deltas":delta,"_local_draw_auroc":draws}

def _recoverability(routes, c, cm, age, train, test, store, version, eligible_indices):
    from sklearn.linear_model import Ridge
    cfg=PatientStateConfig(); start=cfg.shared_dim+cfg.retinal_private_dim
    for route, z in routes.items():
        if route not in store[version]: continue
        x=np.c_[z[:,:cfg.shared_dim],z[:,start:], age]
        for j in eligible_indices:
            fit,score=cm[train,j],cm[test,j]
            if score.sum()<10 or fit.sum()<2: store[version][route][j].append(None); continue
            m=Ridge(alpha=1.).fit(x[train][fit],c[train,j][fit]); store[version][route][j].append(float(np.mean((m.predict(x[test])[score]-c[test,j][score])**2)))

def _summarize_recovery(store, eligible_indices):
    out={}
    for ver in store:
        out[ver]={}
        for route,fields in store[ver].items():
            selected=[fields[j] for j in eligible_indices]
            if any(len(v)!=5 or any(x is None for x in v) for v in selected): out[ver][route]={"status":"suppressed_eligible_field_below_10_in_any_fold"}
            else: out[ver][route]={"status":"scored_normalized_mse","eligible_field_macro_mse":float(np.mean([np.mean(v) for v in selected]))}
    return out

def _summarize_cbc(folds):
    out={}
    for field in CBC_FIELDS:
        rows=[f[field] for f in folds]
        if any(row.get("status")!="scored_normalized_units_only" for row in rows): out[field]={"status":"suppressed_any_outer_fold_below_10"}; continue
        keys=[k for k in rows[0] if k!="status"]
        out[field]={"status":"scored_normalized_units_only",**{k:float(np.mean([r[k] for r in rows])) for k in keys}}
    return out

def _cbc_diff(v1,v2):
    result={}
    for field in CBC_FIELDS:
        a,b=v1[field],v2[field]
        if a.get("status")!="scored_normalized_units_only" or b.get("status")!="scored_normalized_units_only": result[field]={"status":"suppressed"}; continue
        result[field]={"status":"scored_no_paired_ci", **{k:float(b[k]-a[k]) for k in b if k.endswith(("_mae","_mse","coverage90","width90"))}}
    return result

def _cbc_scenario_inputs(c, cm, r, rm, indices, scenario):
    """Physical input erasure; return the untouched observed mask for scoring."""
    input_c,input_m=c.copy(),cm.copy(); input_c[:,indices]=0.; input_m[:,indices]=False; input_r,input_rm=r,rm
    if scenario=="retina_absent_other_clinical": input_r=np.zeros_like(r); input_rm=np.zeros_like(rm)
    elif scenario=="retina_age_only": input_c=np.zeros_like(c); input_m=np.zeros_like(cm)
    elif scenario!="retina_plus_other_clinical": raise AblationError("unknown_cbc_scenario")
    return input_c,input_m,input_r,input_rm,cm

def _cbc_scenario(model, c, cm, r, rm, age, names, train, test, *, scenario, seed):
    """Three fixed CBC-hidden availability scenarios; targets are never input masks."""
    from sklearn.linear_model import Ridge
    import torch
    idx=np.asarray([tuple(names).index(field) for field in CBC_FIELDS]); input_c,input_m,input_r,input_rm,score_mask=_cbc_scenario_inputs(c,cm,r,rm,idx,scenario)
    with torch.no_grad():
        st=model.encode(torch.tensor(input_c,dtype=torch.float32),torch.tensor(input_m),torch.tensor(input_r[:,None],dtype=torch.float32),torch.tensor(input_rm[:,None]),torch.tensor(age,dtype=torch.float32)); plugin=model.predictive_mean(st,torch.tensor(age,dtype=torch.float32))["continuous"].numpy(); torch.manual_seed(seed); sample=model.sample_clinical(st,torch.tensor(age,dtype=torch.float32),samples=64); mean=sample["continuous_mean"].numpy().mean(0); draws=sample["continuous"].numpy()
    state_x=np.c_[st.mean.numpy(),age]; keep=np.ones(c.shape[1],bool); keep[idx]=False; raw_x=np.c_[input_c[:,keep],input_r,age] if scenario!="retina_age_only" else np.c_[input_r,age]
    result={}; lo,hi=np.quantile(draws,.05,axis=0),np.quantile(draws,.95,axis=0)
    for field,j in zip(CBC_FIELDS,idx):
        fit,score=score_mask[train,j],score_mask[test,j]
        if fit.sum()<2 or score.sum()<10: result[field]={"status":"suppressed_small_test_support"}; continue
        target=c[test,j][score]; bridge=Ridge(alpha=1.).fit(state_x[train][fit],c[train,j][fit]); raw=Ridge(alpha=1.).fit(raw_x[train][fit],c[train,j][fit]); age_ref=Ridge(alpha=1.).fit(age[train][fit][:,None],c[train,j][fit])
        def err(p,prefix):
            d=p[score]-target; return {prefix+"_mae":float(np.mean(abs(d))),prefix+"_mse":float(np.mean(d**2))}
        result[field]={"status":"scored_normalized_units_only",**err(plugin[test,j],"bare_plugin_decoder"),**err(mean[test,j],"posterior_predictive_mc_mean"),**err(bridge.predict(state_x[test]),"state_age_ridge"),**err(raw.predict(raw_x[test]),"raw_available_ridge"),**err(age_ref.predict(age[test,None]),"age_ridge"),"median_baseline_mae":float(np.mean(abs(target))),"median_baseline_mse":float(np.mean(target**2)),"posterior_predictive_coverage90":float(np.mean((target>=lo[test,j][score])&(target<=hi[test,j][score]))),"posterior_predictive_interval_width90":float(np.mean(hi[test,j][score]-lo[test,j][score]))}
    return result

def run_bran_anchor_ablation_v2(*, project_root, protocol_path, output_path, failure_path, progress_path, lock_path, dataset_root, clinical_project_root):
    root=Path(project_root).resolve(); output,failure,progress,lock=map(lambda p:Path(p).resolve(),(output_path,failure_path,progress_path,lock_path)); locked=False; fd=None; quiet=None; phase="protocol"
    try:
        protocol=validate_protocol(root,protocol_path); paths=protocol["_paths"]
        if paths!={"output":output,"failure":failure,"progress":progress,"lock":lock} or len(set(paths.values()))!=4 or any(p.parent!=output.parent for p in paths.values()) or output.exists() or failure.exists(): raise AblationError("exclusive_bound_paths_required")
        fd=base._acquire_lock(lock); locked=True; base._atomic_progress(progress,"validated"); quiet=base._quiet_sensitive_block(); quiet.__enter__()
        from patient_atlas_v6_2_expanded_endpoint_evaluation import FROZEN_SUPPORT_RECEIPT_NAME, EXACT_OUTER_FOLD_HASH, EXACT_INNER_FOLD_ASSIGNMENT_SHA256, load_eligible_support_receipt, validate_support_against_observed
        from run_patient_atlas_v6_2_expanded_endpoint_evaluation import _load_actual_v6_2_context
        phase="context"; support=load_eligible_support_receipt(root/FROZEN_SUPPORT_RECEIPT_NAME,project_root=root); context=_load_actual_v6_2_context(root=root,dataset_root=dataset_root,clinical_project_root=clinical_project_root,support=support)
        binding=protocol["authentication"]
        if binding["outer_fold_sha256"]!=EXACT_OUTER_FOLD_HASH or binding["canonical_source_hashes"]!=dict(context.get("source_hashes",{})) or binding["support_receipt_sha256"]!=getattr(support,"receipt_sha256",None) or not isinstance(binding["inner_fold_sha256"],list) or len(binding["inner_fold_sha256"])!=5: raise AblationError("authenticated_binding_mismatch")
        folds=np.asarray(context["outer_assignment"],int); labels,observed=context["labels_by_source"],context["observed_by_source"]; validate_support_against_observed(support,labels,observed,folds)
        if len(support.eligible_sources)!=26 or len(folds)!=_EXPECTED_PATIENT_COUNT or set(folds)!={0,1,2,3,4}: raise AblationError("cohort_or_fold_contract_mismatch")
        c0,cm0,elig,r0,rm,names=_actual_arrays(root,context); elig[:,48:]=False; eligible_indices=tuple(np.flatnonzero(elig[0,:48]));
        if eligible_indices!=ELIGIBLE_CONTINUOUS_INDICES or not np.array_equal(elig[:,:48],np.broadcast_to(elig[0,:48],elig[:,:48].shape)): raise AblationError("continuous_eligibility_contract_mismatch")
        inn=[]
        for f in range(5):
            tr=np.flatnonzero(folds!=f); inner,h=base._inner_context(context,tr,f)
            if h!=EXACT_INNER_FOLD_ASSIGNMENT_SHA256[f]: raise AblationError("inner_hash_mismatch")
            if h!=binding["inner_fold_sha256"][f]: raise AblationError("protocol_inner_hash_mismatch")
            inn.append((inner,h))
        preds={s:{a:np.full(len(folds),np.nan) for a in ARM_NAMES} for s in support.eligible_sources}; rec={v:{r:[[] for _ in range(48)] for r in ("clinical","both")} for v in ("v1","v2")}; scenarios=("retina_plus_other_clinical","retina_absent_other_clinical","retina_age_only"); cbc={v:{q:[] for q in scenarios} for v in ("v1","v2")}; head_diag={"completed_head_refits":0,"candidates_rejected_nonconvergence":0,"candidates_rejected_incomplete_inner_support":0}
        phase="folds"
        for f in range(5):
            base._atomic_progress(progress,"training",f); tr,te=np.flatnonzero(folds!=f),np.flatnonzero(folds==f); trans=base.FoldTransform(c0,cm0,elig,r0,rm,np.asarray(context["raw_cohort"].ages),tr); c,cm,r,age=trans.apply(c0,cm0,elig,r0,rm,np.asarray(context["raw_cohort"].ages)); v1=_train(BRANPatientStatePrototypeV1,c,cm,r,rm,age,tr,1701+f); v2=_train(BRANClinicalAnchorV2,c,cm,r,rm,age,tr,1701+f)
            one,two=_state_routes(v1,c,cm,r,rm,age),_state_routes(v2,c,cm,r,rm,age); raw=base.encode_arms(v1,c,cm,r,rm,age); arms={"age":raw["age"],"rawclinical":raw["original_clinical"],"rawretina":raw["original_retinal"],"rawconcat":raw["original_concat"],**{"v1"+k:np.c_[v,age] for k,v in one.items()},**{"v2"+k:np.c_[v,age] for k,v in two.items()}}
            _recoverability(one,c,cm,age,tr,te,rec,"v1",eligible_indices); _recoverability(two,c,cm,age,tr,te,rec,"v2",eligible_indices)
            base._atomic_progress(progress,"completion",f)
            for q in scenarios:
                cbc["v1"][q].append(_cbc_scenario(v1,c,cm,r,rm,age,names,tr,te,scenario=q,seed=91501+f)); cbc["v2"][q].append(_cbc_scenario(v2,c,cm,r,rm,age,names,tr,te,scenario=q,seed=91501+f))
            base._atomic_progress(progress,"screening",f)
            for s in support.eligible_sources:
                y,m=np.asarray(labels[s]),np.asarray(observed[s],bool)
                for a in ARM_NAMES: preds[s][a][te]=base._fit_predict_nested(arms[a][tr],y[tr],m[tr],inn[f][0],arms[a][te],diagnostics=head_diag)[0]
            base._atomic_progress(progress,"fold_complete",f)
        phase="aggregate"; base._atomic_progress(progress,"aggregate"); counts=_shared_bootstrap_counts(folds); endpoint={}; draw_cache={}
        for s in support.eligible_sources:
            item=_bootstrap(np.asarray(labels[s]),preds[s],np.asarray(observed[s],bool),folds,counts); draw_cache[s]=item.pop("_local_draw_auroc"); endpoint[s]=item
        macro={}
        for name in COMPARISONS:
            left,right=name.split("-"); point=float(np.mean([endpoint[s]["paired_deltas"][name]["auroc_delta"] for s in support.eligible_sources])); draws=np.mean([draw_cache[s][left]-draw_cache[s][right] for s in support.eligible_sources],axis=0)
            macro[name]={"mean_26_endpoint_auroc_delta":point,"ci95":[float(np.nanpercentile(draws,2.5)),float(np.nanpercentile(draws,97.5))]}
        cbc_summary={v:{q:_summarize_cbc(cbc[v][q]) for q in scenarios} for v in ("v1","v2")}; cbc_delta={q:_cbc_diff(cbc_summary["v1"][q],cbc_summary["v2"][q]) for q in scenarios}
        report={"schema_version":SCHEMA,"status":"completed_aggregate_only","exploratory":True,"protocol_sha256":_hash(Path(protocol_path)),"code_hashes":protocol["expected_hashes"],"source_hashes":dict(context.get("source_hashes",{})),"support_receipt_sha256":getattr(support,"receipt_sha256",None),"scope":{"patient_count":1928,"endpoint_count":26,"official_test_loaded":False},"fold_hashes":{"outer":EXACT_OUTER_FOLD_HASH,"inner":[h for _,h in inn]},"parameters":PARAMETERS,"screening":{"endpoint_results":endpoint,"macro_26_endpoint_paired_deltas":macro,"head_diagnostics":head_diag,"inference":"one_shared_fold_stratified_patient_bootstrap_count_matrix_marginal_95pct_fixed_oof_1000"},"clinical_recoverability":_summarize_recovery(rec,eligible_indices),"cbc":{"scenarios":{"retina_plus_other_clinical":"primary_all9_cbc_hidden","retina_absent_other_clinical":"descriptive_robustness","retina_age_only":"descriptive_no_clinical_view"},"v1":cbc_summary["v1"],"v2":cbc_summary["v2"],"v2_minus_v1_point_differences_no_paired_ci":cbc_delta},"patient_rows_or_ids_serialized":False,"models_oof_rows_ids_or_draws_serialized":False}
        _write_x(output,report); base._atomic_completed(progress); return report
    except Exception as e:
        allowed={str((root/name).resolve()) for name in protocol.get("expected_hashes",{})} if "protocol" in locals() else set()
        frames=[{"file":Path(f.filename).name,"line":int(f.lineno)} for f in traceback.extract_tb(e.__traceback__) if str(Path(f.filename).resolve()) in allowed][:8]
        safe={"schema_version":SCHEMA,"status":"failed","phase":phase,"error_class":type(e).__name__,"bound_code_frames":frames,"exception_text_serialized":False,"patient_rows_or_ids_serialized":False}
        if locked and not output.exists() and not failure.exists(): _write_x(failure,safe)
        return safe
    finally:
        if quiet is not None: quiet.__exit__(None,None,None)
        if fd is not None:
            os.close(fd)
            try: lock.unlink()
            except FileNotFoundError: pass
