"""Strict aggregate-only paired screening inference with common patient draws."""
import numpy as np
import run_bran_overnight_diagnostic_v1 as base
COMPARATORS=('v2both_legacy','v2clinical_legacy','v2retinal_legacy','raw_blood','raw_clinical_masked','raw_retinal','raw_concat_masked','innovation_equal')
FAMILY=('v2clinical_legacy','v2retinal_legacy','raw_blood')
ARMS=('innovation_selected',)+COMPARATORS+('age',)
class InnovationAggregateError(ValueError):pass
def require(x):
    if not x:raise InnovationAggregateError('aggregate_contract_invalid')

def summarize_screening(labels_by_source,preds_by_source_arm,observed_by_source,outer,counts,*,minimum_valid_draws=950):
    outer=np.asarray(outer);counts=np.asarray(counts);n=len(outer)
    require(outer.ndim==1 and outer.dtype.kind in 'iu' and set(outer.tolist())==set(range(5)))
    require(counts.ndim==2 and counts.shape[1]==n and counts.dtype.kind in 'iu' and np.all(counts>=0))
    require(type(minimum_valid_draws) is int and 1<=minimum_valid_draws<=len(counts))
    for f in range(5):require(np.all(counts[:,outer==f].sum(1)==np.sum(outer==f)))
    sources=tuple(labels_by_source)
    require(bool(sources) and set(preds_by_source_arm)==set(observed_by_source)==set(sources))
    endpoints={};cache={}
    for source in sources:
        y=np.asarray(labels_by_source[source],float);m=np.asarray(observed_by_source[source])
        require(y.shape==(n,) and m.shape==y.shape and m.dtype.kind=='b' and np.isfinite(y[m]).all())
        require(set(np.unique(y[m]))=={0.,1.} and set(preds_by_source_arm[source])==set(ARMS))
        complete=np.ones(len(counts),bool)
        for f in range(5):
            pos=m&(outer==f)&(y==1);neg=m&(outer==f)&(y==0)
            require(pos.sum()>=10 and neg.sum()>=10)
            complete&=(counts[:,pos].sum(1)>0)&(counts[:,neg].sum(1)>0)
        require(complete.sum()>=minimum_valid_draws)
        metrics={};draws={}
        for arm,p in preds_by_source_arm[source].items():
            p=np.asarray(p,float);require(p.shape==(n,) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)))
            d=base._weighted_auc_draws(y,p,m,outer,counts);d[~complete]=np.nan
            require(np.isfinite(d).sum()>=minimum_valid_draws)
            metrics[arm]={'auroc':base.fold_weighted_auc(y,p,m,outer),'logloss':base.fold_weighted_logloss(y,p,m,outer),
                          'ci95':[float(np.nanpercentile(d,2.5)),float(np.nanpercentile(d,97.5))]}
            draws[arm]=d
        paired={}
        for right in COMPARATORS:
            d=draws['innovation_selected']-draws[right]
            paired['innovation_selected_minus_'+right]={'auroc_difference':metrics['innovation_selected']['auroc']-metrics[right]['auroc'],
                'ci95':[float(np.nanpercentile(d,2.5)),float(np.nanpercentile(d,97.5))]}
        endpoints[source]={'arms':metrics,'paired_deltas':paired};cache[source]=draws
    macro={};family={}
    for right in COMPARATORS:
        d=np.stack([cache[s]['innovation_selected']-cache[s][right] for s in sources])
        ok=np.isfinite(d).all(0);require(ok.sum()>=minimum_valid_draws);a=d[:,ok].mean(0)
        point=float(np.mean([endpoints[s]['paired_deltas']['innovation_selected_minus_'+right]['auroc_difference'] for s in sources]))
        macro['innovation_selected_minus_'+right]={'mean_endpoint_auroc_difference':point,'ci95':[float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]}
        if right in FAMILY:family[right]=float(np.percentile(a,100*.05/len(FAMILY)))
    return {'endpoint_results':endpoints,'macro_paired_deltas':macro,
        'single_view_family':{'comparators':list(FAMILY),'bonferroni_one_sided_95_lower':family,'all_lower_above_zero':all(v>0 for v in family.values())},
        'bootstrap_draws_retained':False}
