"""Recreate the prespecified BRAN references locally; no files or patient output.

This is a component of a new matched benchmark, not a rerun of CBC experiments.
Returned prediction arrays are strictly local and must never be serialized.
"""
import numpy as np
import run_bran_anchor_ablation_v2 as paired
from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
from bran_innovation_readout_v1 import innovation_coordinates, fit_screening
from run_bran_innovation_readout_v1 import weights_hash, ELIGIBLE
from bran_named_fm_auc_v1 import require, REFERENCE_ARMS


def predict_references(c0, cm0, elig, r0, rm, ages, outer, inners, labels, observed,
                       *, sources, progress_callback=None, steps=1500):
    n=len(outer);outer=np.asarray(outer);ages=np.asarray(ages,float)
    c0=np.asarray(c0,float);cm0=np.asarray(cm0);elig=np.asarray(elig)
    r0=np.asarray(r0,float);rm=np.asarray(rm);sources=tuple(sources)
    require(c0.shape==cm0.shape==elig.shape==(n,59) and cm0.dtype.kind==elig.dtype.kind=='b')
    require(r0.shape==(n,384) and rm.shape==(n,) and rm.dtype.kind=='b')
    require(ages.shape==(n,) and np.isfinite(ages).all() and outer.shape==(n,) and outer.dtype.kind in 'iu' and set(outer.tolist())==set(range(5)))
    require(np.isfinite(c0[cm0&elig]).all() and np.isfinite(r0[rm]).all())
    require(tuple(np.flatnonzero(elig[0,:48]))==ELIGIBLE and np.array_equal(elig,np.broadcast_to(elig[0],elig.shape)) and not elig[:,48:].any())
    require(len(sources)==len(set(sources))==26 and set(sources)==set(labels)==set(observed) and len(inners)==5)
    for s in sources:
        y=np.asarray(labels[s]);m=np.asarray(observed[s])
        require(y.shape==m.shape==(n,) and m.dtype.kind=='b' and np.isfinite(y[m]).all() and set(np.unique(y[m]))=={0,1})
        for f in range(5):
            for value in (0,1):require(np.sum(m&(outer==f)&(y==value))>=10)
    pred={s:{a:np.full(n,np.nan) for a in REFERENCE_ARMS} for s in sources}
    diag={'encoder_fits':0,'reference_head_refits':0,'innovation_candidates_rejected_nonconvergence':0,
          'legacy_candidates_rejected_nonconvergence':0,'legacy_candidates_rejected_incomplete_inner_support':0,
          'encoder_weights_unchanged_after_heads':True}
    for f in range(5):
        if progress_callback:progress_callback('reference_training',f)
        tr=np.flatnonzero(outer!=f);te=np.flatnonzero(outer==f);inner=np.asarray(inners[f])
        require(inner.shape==(len(tr),) and inner.dtype.kind in 'iu' and set(inner.tolist())==set(range(5)))
        transform=paired.base.FoldTransform(c0,cm0,elig,r0,rm,ages,tr)
        c,cm,r,age=transform.apply(c0,cm0,elig,r0,rm,ages)
        model=paired._train(BRANClinicalAnchorV2,c,cm,r,rm,age,tr,1701+f,steps=steps)
        before=weights_hash(model);diag['encoder_fits']+=1
        routes=paired._state_routes(model,c,cm,r,rm,age)
        ar=model.retinal_prior.weight.detach().numpy();ac=model.clinical_prior.weight.detach().numpy()
        designs={'bran_both':np.c_[innovation_coordinates(routes['both'],ar,ac),age],
                 'bran_clinical':np.c_[routes['clinical'],age], 'bran_retinal':np.c_[routes['retinal'],age]}
        if progress_callback:progress_callback('reference_heads',f)
        legacy={}
        for s in sources:
            y=np.asarray(labels[s]);m=np.asarray(observed[s]);x=designs['bran_both']
            result=fit_screening(x[tr],y[tr],m[tr],inner,x[te],selected_profiles=True)
            pred[s]['bran_both'][te]=result['predictions'];diag['innovation_candidates_rejected_nonconvergence']+=result['rejected_nonconvergence']
            for a in REFERENCE_ARMS[1:]:
                x=designs[a];pred[s][a][te]=paired.base._fit_predict_nested(x[tr],y[tr],m[tr],inner,x[te],diagnostics=legacy)[0]
            diag['reference_head_refits']+=3
        require(legacy.get('completed_head_refits')==52 and weights_hash(model)==before)
        for key in ('candidates_rejected_nonconvergence','candidates_rejected_incomplete_inner_support'):
            diag['legacy_'+key]+=legacy.get(key,0)
    require(all(np.isfinite(p).all() and np.all((p>=0)&(p<=1)) for d in pred.values() for p in d.values()))
    return pred,diag


def replay_references(pred, labels, observed, outer, previous_result):
    """Point-metric canary against the completed 26-endpoint receipt, no arrays out."""
    endpoints=previous_result['screening']['endpoint_results']
    require(len(pred)==26 and set(pred)==set(endpoints)==set(labels)==set(observed))
    mapping={'bran_both':'innovation_selected','bran_clinical':'v2clinical_legacy','bran_retinal':'v2retinal_legacy'}
    for s,d in pred.items():
        require(set(d)==set(REFERENCE_ARMS))
        for arm,old in mapping.items():
            for metric,fn in (('auroc',paired.base.fold_weighted_auc),('logloss',paired.base.fold_weighted_logloss)):
                point=fn(labels[s],d[arm],observed[s],outer)
                require(np.isfinite(point) and abs(point-endpoints[s]['arms'][old][metric])<=1e-8)
    return {'screening_3x26_auroc_and_logloss':True,'tolerance':1e-8}
