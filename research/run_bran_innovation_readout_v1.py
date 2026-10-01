"""In-memory, aggregate-only innovation screening/completion evaluation."""
import hashlib
import numpy as np
import torch
import run_bran_overnight_diagnostic_v1 as base
import run_bran_anchor_ablation_v2 as paired
from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
from bran_frozen_cbc_ridge_v1 import fit_frozen_cbc_ridge
from run_bran_frozen_cbc_readout_v1 import CBC_FIELDS,SCENARIOS,_precheck
from bran_innovation_readout_v1 import innovation_coordinates,fit_screening,fit_cbc
from bran_innovation_aggregate_v1 import summarize_screening
ARMS=('v2both_legacy','v2clinical_legacy','v2retinal_legacy','innovation_equal','innovation_selected','raw_blood','raw_clinical_masked','raw_retinal','raw_concat_masked','age')
CBC_ARMS=('state_nested','innovation_selected','raw_nested','age_nested','median')
ELIGIBLE=tuple(i for i in range(48) if i not in (8,9,20,35,36))
class InnovationEvaluationError(ValueError):pass
def require(x):
    if not x:raise InnovationEvaluationError('evaluation_contract_invalid')
def weights_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def summarize_cbc(y,pred,mask,outer,counts,minimum_valid_draws):
    require(y.shape==mask.shape==(len(outer),9) and mask.dtype.kind=='b' and np.isfinite(y[mask]).all())
    require(set(pred)==set(CBC_ARMS) and all(p.shape==y.shape and np.isfinite(p).all() for p in pred.values()))
    metrics={};draw={}
    for j,field in enumerate(CBC_FIELDS):
        metrics[field]={};draw[field]={}
        for arm in CBC_ARMS:
            mae=[];mse=[];ds=[]
            for f in range(5):
                select=(outer==f)&mask[:,j];require(select.sum()>=10)
                error=pred[arm][select,j]-y[select,j];square=error**2
                mae.append(np.mean(np.abs(error)));mse.append(np.mean(square))
                w=counts[:,select].astype(float);den=w.sum(1)
                ds.append(np.divide(w@square,den,out=np.full(len(w),np.nan),where=den>0))
            d=np.mean(ds,axis=0);require(np.isfinite(d).sum()>=minimum_valid_draws)
            metrics[field][arm]={'normalized_mae':float(np.mean(mae)),'normalized_mse':float(np.mean(mse))};draw[field][arm]=d
    paired_fields={};macro={}
    for right in ('state_nested','raw_nested'):
        name='innovation_selected_minus_'+right;all_d=[];paired_fields[name]={}
        for field in CBC_FIELDS:
            d=draw[field]['innovation_selected']-draw[field][right];ok=np.isfinite(d)
            require(ok.sum()>=minimum_valid_draws);all_d.append(d)
            paired_fields[name][field]={'normalized_mse_difference':metrics[field]['innovation_selected']['normalized_mse']-metrics[field][right]['normalized_mse'],
                'ci95':[float(np.percentile(d[ok],2.5)),float(np.percentile(d[ok],97.5))]}
        d=np.stack(all_d);ok=np.isfinite(d).all(0);require(ok.sum()>=minimum_valid_draws);a=d[:,ok].mean(0)
        macro[name]={'mean_9_normalized_mse_difference':float(np.mean([paired_fields[name][f]['normalized_mse_difference'] for f in CBC_FIELDS])),
                     'ci95':[float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]}
    return {'field_metrics':metrics,'field_paired_deltas':paired_fields,'macro_paired_deltas':macro}

def evaluate(c0,cm0,elig,r0,rm,names,ages,outer,inner_assignments,labels_by_source,observed_by_source,sources,
             progress_callback=None,steps=1500,bootstrap_counts=None,minimum_valid_draws=950):
    outer=np.asarray(outer);n=len(outer);sources=tuple(sources);cm0=np.asarray(cm0);elig=np.asarray(elig)
    c0=np.asarray(c0,float);r0=np.asarray(r0,float);rm=np.asarray(rm);ages=np.asarray(ages,float)
    require(c0.shape==cm0.shape==elig.shape==(n,59) and cm0.dtype.kind==elig.dtype.kind=='b')
    require(outer.dtype.kind in 'iu' and set(outer.tolist())==set(range(5)) and r0.shape==(n,384) and rm.shape==(n,) and rm.dtype.kind=='b')
    require(ages.shape==(n,) and np.isfinite(ages).all() and len(sources)==len(set(sources))==26)
    require(set(sources)==set(labels_by_source)==set(observed_by_source) and len(inner_assignments)==5)
    require(tuple(np.flatnonzero(elig[0,:48]))==ELIGIBLE and np.array_equal(elig,np.broadcast_to(elig[0],elig.shape)) and not elig[:,48:].any())
    idx=np.asarray([tuple(names).index(x) for x in CBC_FIELDS]);keep43=np.asarray(ELIGIBLE);blood=keep43[keep43<38]
    require(len(blood)==33);safe_mask=cm0&elig&np.isfinite(c0);_precheck(safe_mask,idx,outer,inner_assignments)
    for source in sources:
        y=np.asarray(labels_by_source[source]);m=np.asarray(observed_by_source[source])
        require(y.shape==m.shape==(n,) and m.dtype.kind=='b' and np.isfinite(y[m]).all() and set(np.unique(y[m]))=={0,1})
        for f in range(5):require(np.sum(m&(outer==f)&(y==0))>=10 and np.sum(m&(outer==f)&(y==1))>=10)
    predictions={s:{a:np.full(n,np.nan) for a in ARMS} for s in sources}
    completion={q:{a:np.full((n,9),np.nan) for a in CBC_ARMS} for q in SCENARIOS}
    truth=np.full((n,9),np.nan);mask=np.zeros((n,9),bool)
    diagnostics={'legacy_and_raw_head_refits':0,'innovation_head_refits':0,'innovation_candidates_rejected_nonconvergence':0,
                 'legacy_candidates_rejected_nonconvergence':0,'legacy_candidates_rejected_incomplete_inner_support':0,
                 'completion_head_refits':0,'encoder_weights_unchanged_after_heads':True}
    def progress(phase,f):
        if progress_callback:progress_callback(phase,f)
    for f in range(5):
        progress('training',f);tr,te=np.flatnonzero(outer!=f),np.flatnonzero(outer==f)
        inner=np.asarray(inner_assignments[f]);require(inner.shape==(len(tr),) and inner.dtype.kind in 'iu' and set(inner.tolist())==set(range(5)))
        transform=base.FoldTransform(c0,cm0,elig,r0,rm,ages,tr);c,cm,r,age=transform.apply(c0,cm0,elig,r0,rm,ages)
        model=paired._train(BRANClinicalAnchorV2,c,cm,r,rm,age,tr,1701+f,steps=steps);before=weights_hash(model)
        @torch.no_grad()
        def state(cc,mm,rr,rrm):
            return model.encode(torch.tensor(cc,dtype=torch.float32),torch.tensor(mm),torch.tensor(rr[:,None],dtype=torch.float32),torch.tensor(rrm[:,None]),torch.tensor(age,dtype=torch.float32))
        sb=state(c,cm,r,rm);sc=state(c,cm,np.zeros_like(r),np.zeros_like(rm));sr=state(np.zeros_like(c),np.zeros_like(cm),r,rm)
        ar=model.retinal_prior.weight.detach().numpy();ac=model.clinical_prior.weight.detach().numpy();inv=np.c_[innovation_coordinates(sb.mean.numpy(),ar,ac),age]
        arms={'v2both_legacy':np.c_[sb.mean.numpy(),age],'v2clinical_legacy':np.c_[sc.mean.numpy(),age],
            'v2retinal_legacy':np.c_[sr.mean.numpy(),age],'innovation_equal':inv,'innovation_selected':inv,
            'raw_blood':np.c_[c[:,blood],cm[:,blood].astype(float),age],'raw_clinical_masked':np.c_[c[:,keep43],cm[:,keep43].astype(float),age],
            'raw_retinal':np.c_[r,age],'raw_concat_masked':np.c_[r,c[:,keep43],cm[:,keep43].astype(float),age],'age':age[:,None]}
        progress('screening',f);legacy={}
        for source in sources:
            y=np.asarray(labels_by_source[source]);m=np.asarray(observed_by_source[source])
            for arm,x in arms.items():
                if arm.startswith('innovation'):
                    z=fit_screening(x[tr],y[tr],m[tr],inner,x[te],selected_profiles=(arm=='innovation_selected'))
                    predictions[source][arm][te]=z['predictions'];diagnostics['innovation_head_refits']+=1
                    diagnostics['innovation_candidates_rejected_nonconvergence']+=z['rejected_nonconvergence']
                else:predictions[source][arm][te]=base._fit_predict_nested(x[tr],y[tr],m[tr],inner,x[te],diagnostics=legacy)[0]
        diagnostics['legacy_and_raw_head_refits']+=legacy.get('completed_head_refits',0)
        diagnostics['legacy_candidates_rejected_nonconvergence']+=legacy.get('candidates_rejected_nonconvergence',0)
        diagnostics['legacy_candidates_rejected_incomplete_inner_support']+=legacy.get('candidates_rejected_incomplete_inner_support',0)
        progress('completion',f)
        for scenario in SCENARIOS:
            ic,im,ir,irm,score=paired._cbc_scenario_inputs(c,cm,r,rm,idx,scenario);require(not im[:,idx].any() and np.all(ic[:,idx]==0))
            st=state(ic,im,ir,irm);x=np.c_[st.mean.numpy(),age];innovation=np.c_[innovation_coordinates(st.mean.numpy(),ar,ac),age]
            keep=np.ones(c.shape[1],bool);keep[idx]=False
            raw=np.c_[ic[:,keep],im[:,keep].astype(float),ir,age] if scenario!=SCENARIOS[2] else np.c_[ir,age]
            y,observed=c[:,idx],score[:,idx]
            for arm,design in (('state_nested',x),('raw_nested',raw),('age_nested',age[:,None])):
                completion[scenario][arm][te]=fit_frozen_cbc_ridge(design[tr],y[tr],observed[tr],inner,design[te]).predictions;diagnostics['completion_head_refits']+=9
            completion[scenario]['innovation_selected'][te]=fit_cbc(innovation[tr],y[tr],observed[tr],inner,innovation[te])['predictions'];diagnostics['completion_head_refits']+=9
            completion[scenario]['median'][te]=np.array([np.median(y[tr,j][observed[tr,j]]) for j in range(9)])
            truth[te]=y[te];mask[te]=observed[te]
        require(weights_hash(model)==before);progress('fold_complete',f)
    counts=paired._shared_bootstrap_counts(outer) if bootstrap_counts is None else np.asarray(bootstrap_counts)
    screen=summarize_screening(labels_by_source,predictions,observed_by_source,outer,counts,minimum_valid_draws=minimum_valid_draws)
    cbc={q:summarize_cbc(truth,p,mask,outer,counts,minimum_valid_draws) for q,p in completion.items()}
    return {'schema':'bran-innovation-readout-v1','screening':screen,'cbc':cbc,'diagnostics':diagnostics,
            'patient_rows_or_predictions_serialized':False,'selected_head_intervals_claimed':False}
