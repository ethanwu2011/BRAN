"""Draft in-memory bridge for matched named-FM AUROC probes; no data/model IO.

Not a real-data entry point. Requires a separately frozen extraction/identity
protocol before use on patients. Local prediction arrays must never be emitted.
"""
import numpy as np
import run_bran_overnight_diagnostic_v1 as base

WIDTHS={'retfound_green':384,'visionfm_last4':3072,'dinov3_generic':384,'labrador':1024}
REFERENCE_ARMS=('bran_both','bran_clinical','bran_retinal')
ARMS=REFERENCE_ARMS+tuple(WIDTHS)
class FMProbeError(ValueError):pass
def require(ok):
    if not ok:raise FMProbeError('fm_probe_contract_invalid')

def fit_probe_fold(embedding,age,train,test,inner,labels,observed,*,variant,source_codes):
    """Fit only train labels; released encoders are upstream and must stay frozen."""
    require(variant in WIDTHS);embedding=np.asarray(embedding,float);age=np.asarray(age,float)
    n=len(age);train=np.asarray(train);test=np.asarray(test);inner=np.asarray(inner)
    require(embedding.shape==(n,WIDTHS[variant]) and age.shape==(n,) and np.isfinite(embedding).all() and np.isfinite(age).all())
    require(train.ndim==test.ndim==inner.ndim==1 and train.dtype.kind in 'iu' and test.dtype.kind in 'iu' and inner.dtype.kind in 'iu')
    require(set(train.tolist()).isdisjoint(test.tolist()) and len(set(train.tolist()))==len(train) and len(set(test.tolist()))==len(test))
    require(set(train.tolist())|set(test.tolist())==set(range(n)) and inner.shape==(len(train),) and set(inner.tolist())==set(range(5)))
    sources=tuple(source_codes);require(len(sources)==len(set(sources))==26 and set(sources)<=set(labels) and set(sources)<=set(observed))
    x=np.c_[embedding,age];pred={};diagnostics={}
    for source in sources:
        y=np.asarray(labels[source],float);m=np.asarray(observed[source]);require(y.shape==m.shape==(n,) and m.dtype.kind=='b')
        require(np.isfinite(y[m]).all() and set(np.unique(y[m]))=={0.,1.})
        # The fixed readout fits all standardizers using the relevant training rows.
        pred[source]=base._fit_predict_nested(x[train],y[train],m[train],inner,x[test],diagnostics=diagnostics)[0]
        require(pred[source].shape==(len(test),) and np.isfinite(pred[source]).all())
    require(diagnostics.get('completed_head_refits')==26)
    return pred,diagnostics

def summarize_paired(predictions,labels,observed,outer,counts,*,source_codes,minimum_valid_draws=950):
    """Common paired draws, exact arm coverage, no per-patient or draw return."""
    outer=np.asarray(outer);counts=np.asarray(counts);n=len(outer);sources=tuple(source_codes)
    require(outer.ndim==1 and outer.dtype.kind in 'iu' and set(outer.tolist())==set(range(5)))
    require(counts.ndim==2 and counts.shape[1]==n and counts.dtype.kind in 'iu' and np.all(counts>=0))
    require(type(minimum_valid_draws) is int and 1<=minimum_valid_draws<=len(counts))
    require(len(sources)==len(set(sources))==26 and set(predictions)==set(sources) and set(sources)<=set(labels) and set(sources)<=set(observed))
    for f in range(5):require(np.all(counts[:,outer==f].sum(1)==np.sum(outer==f)))
    endpoints={};draws={}
    for s in sources:
        y=np.asarray(labels[s],float);m=np.asarray(observed[s]);require(y.shape==m.shape==(n,) and m.dtype.kind=='b' and np.isfinite(y[m]).all() and set(np.unique(y[m]))=={0.,1.})
        require(set(predictions[s])==set(ARMS));valid=np.ones(len(counts),bool)
        for f in range(5):
            for label in (0,1):
                sel=m&(outer==f)&(y==label);require(sel.sum()>=10);valid&=counts[:,sel].sum(1)>0
        require(valid.sum()>=minimum_valid_draws);metrics={};d={}
        for arm in ARMS:
            p=np.asarray(predictions[s][arm],float);require(p.shape==(n,) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)))
            v=base._weighted_auc_draws(y,p,m,outer,counts);v[~valid]=np.nan;d[arm]=v
            metrics[arm]={'auroc':base.fold_weighted_auc(y,p,m,outer),'logloss':base.fold_weighted_logloss(y,p,m,outer),
                'ci95':[float(np.nanpercentile(v,2.5)),float(np.nanpercentile(v,97.5))]}
        paired={}
        for right in ARMS[1:]:
            delta=d['bran_both']-d[right];paired['bran_both_minus_'+right]={'auroc_difference':metrics['bran_both']['auroc']-metrics[right]['auroc'],
                'ci95':[float(np.nanpercentile(delta,2.5)),float(np.nanpercentile(delta,97.5))]}
        endpoints[s]={'arms':metrics,'paired_deltas':paired};draws[s]=d
    macro={};lower={}
    for right in ARMS[1:]:
        d=np.stack([draws[s]['bran_both']-draws[s][right] for s in sources]);ok=np.isfinite(d).all(0);require(ok.sum()>=minimum_valid_draws);v=d[:,ok].mean(0)
        macro['bran_both_minus_'+right]={'mean_endpoint_auroc_difference':float(np.mean([endpoints[s]['paired_deltas']['bran_both_minus_'+right]['auroc_difference'] for s in sources])),
            'ci95':[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]}
        if right in WIDTHS:lower[right]=float(np.percentile(v,100*.05/len(WIDTHS)))
    return {'schema':'bran-named-fm-matched-auroc-v1','endpoint_results':endpoints,'macro_paired_deltas':macro,
        'named_fm_macro_family':{'comparators':list(WIDTHS),'bonferroni_one_sided_95_lower':lower,'all_lower_above_zero':all(x>0 for x in lower.values())},
        'inference':'marginal_unadjusted_fixed_fit_common_patient_bootstrap','patient_arrays_or_draws_serialized':False,
        'original_nature_retfound_included':False,'longitudinal_ehr_fm_included':False}
