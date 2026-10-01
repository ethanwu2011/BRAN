"""Private array-only fixed probes for the prospective CBC warm-start study.

No ingestion, fitting authorization, plotting, logging or file output. Existing
development folds and source/label eligibility must be authenticated by caller.
"""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_bran_overnight_diagnostic_v1 as base
from bran_clinical_semantics_v1 import CBC_FIELDS

SCREEN_ARMS=('control_clinical','control_retinal','control_both',
             'candidate_clinical','candidate_retinal','candidate_both',
             'raw_clinical','raw_retinal','raw_concat')
CBC_ARMS=('control','candidate','raw')
SCREEN_COMPARISONS=(('candidate_both','control_both'),('candidate_both','candidate_clinical'),
    ('candidate_both','candidate_retinal'),('candidate_both','raw_clinical'),('candidate_both','raw_concat'))


def fixed_screen_probe(x,y,observed,train,test):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float); observed=np.asarray(observed,bool)
    if x.ndim!=2 or y.shape!=(len(x),) or observed.shape!=y.shape or not np.isfinite(x).all():
        raise ValueError('invalid screening probe inputs')
    fit=train[observed[train]]
    if set(np.unique(y[fit]))!={0.,1.}: raise ValueError('screening training support unavailable')
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver='lbfgs',max_iter=5000,random_state=7191))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(x[fit],y[fit])
    return model.predict_proba(x[test])[:,1]


def whole_cbc_inputs(c,observed,registry_fields):
    """Remove every CBC value AND its observation flag before state inference."""
    if c.ndim!=2 or observed.shape!=c.shape or observed.dtype!=bool or len(registry_fields)!=c.shape[1]:
        raise ValueError('invalid completion input shape')
    if len(set(registry_fields))!=len(registry_fields) or not set(CBC_FIELDS)<=set(registry_fields):
        raise ValueError('invalid completion registry')
    indices=np.array([registry_fields.index(f) for f in CBC_FIELDS])
    if np.any(indices>=48): raise ValueError('CBC fields must be continuous')
    values=c.copy(); mask=observed.copy()
    values[:,indices]=0.; mask[:,indices]=False
    keep=np.ones(c.shape[1],bool); keep[indices]=False
    return values,mask,keep,indices


def fixed_cbc_probe(x,target,target_observed,train,test):
    x=np.asarray(x,float); y=np.asarray(target,float); observed=np.asarray(target_observed,bool)
    if x.ndim!=2 or y.shape!=(len(x),) or observed.shape!=y.shape or not np.isfinite(x).all():
        raise ValueError('invalid completion probe inputs')
    fit=train[observed[train] & np.isfinite(y[train])]
    if len(fit)<20: raise ValueError('completion training support unavailable')
    # Both embeddings and raw inputs receive the same train-only scaling and L2.
    model=make_pipeline(StandardScaler(),Ridge(alpha=1.))
    model.fit(x[fit],y[fit])
    return model.predict(x[test])


def paired_counts(folds,*,draws=1000,seed=91501):
    folds=np.asarray(folds)
    if folds.ndim!=1 or set(np.unique(folds))!={0,1,2,3,4} or type(draws) is not int or draws<1:
        raise ValueError('invalid uncertainty fold configuration')
    rng=np.random.default_rng(seed); counts=np.zeros((draws,len(folds)),np.int16)
    for f in range(5):
        ix=np.flatnonzero(folds==f)
        for b in range(draws): counts[b,ix]=np.bincount(rng.choice(len(ix),len(ix),replace=True),minlength=len(ix))
    return counts


def _ci(draws):
    draws=np.asarray(draws,float)
    if draws.ndim!=1 or not np.isfinite(draws).all(): raise ValueError('nonfinite uncertainty draws')
    return [float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]


def summarize_cbc(target,observed,predictions,counts):
    """Disclosure-safe original-unit aggregates; no row residuals or draws leave."""
    if target.shape!=observed.shape or target.shape[1]!=9 or observed.dtype!=bool or set(predictions)!=set(CBC_ARMS):
        raise ValueError('invalid CBC result contract')
    if counts.ndim!=2 or counts.shape[1]!=len(target) or np.any(counts<0): raise ValueError('invalid paired counts')
    result={}
    for j,field in enumerate(CBC_FIELDS):
        valid=observed[:,j] & np.isfinite(target[:,j]); n=int(valid.sum())
        if n<20: raise ValueError('CBC aggregate support below privacy threshold')
        weights=counts[:,valid].astype(float); denominator=weights.sum(1)
        if np.any(denominator<=0): raise ValueError('CBC resampling support unavailable')
        arms={}; errors={}
        for name in CBC_ARMS:
            pred=np.asarray(predictions[name])
            if pred.shape!=target.shape or not np.isfinite(pred[valid,j]).all(): raise ValueError('invalid CBC predictions')
            residual=pred[valid,j]-target[valid,j]; absolute=np.abs(residual)
            arms[name]={'mae':float(absolute.mean()),'mse':float(np.square(residual).mean())}
            errors[name]=absolute
        contrasts={}
        for reference in ('control','raw'):
            delta=errors['candidate']-errors[reference]
            contrasts['candidate-'+reference]={'mae_delta':float(delta.mean()),'ci95':_ci((weights@delta)/denominator)}
        result[field]={'observed_count_lower_bound_20':n//20*20,'arms':arms,'paired_deltas':contrasts}
    return result


def summarize_screening(labels,observed,predictions,folds,counts,endpoint_names):
    if set(labels)!=set(endpoint_names) or set(observed)!=set(endpoint_names) or set(predictions)!=set(endpoint_names):
        raise ValueError('screening endpoint names differ from frozen registry')
    endpoint={}; macro_points={a:[] for a in SCREEN_ARMS}; macro_draws={a:[] for a in SCREEN_ARMS}
    for name in endpoint_names:
        y=np.asarray(labels[name]); obs=np.asarray(observed[name],bool); preds=predictions[name]
        if y.shape!=folds.shape or obs.shape!=y.shape or set(preds)!=set(SCREEN_ARMS): raise ValueError('invalid screening result shape')
        if int(np.count_nonzero(obs & (y==1)))<20 or int(np.count_nonzero(obs & (y==0)))<20: raise ValueError('screening aggregate support below privacy threshold')
        points={}; draws={}
        for arm in SCREEN_ARMS:
            p=np.asarray(preds[arm])
            if p.shape!=y.shape or not np.isfinite(p[obs]).all() or np.any((p[obs]<0)|(p[obs]>1)): raise ValueError('invalid screening probabilities')
            points[arm]=float(base.fold_weighted_auc(y,p,obs,folds))
            draws[arm]=base._weighted_auc_draws(y,p,obs,folds,counts)
            macro_points[arm].append(points[arm]); macro_draws[arm].append(draws[arm])
        contrasts={a+'-'+b:{'auroc_delta':points[a]-points[b],'ci95':_ci(draws[a]-draws[b])} for a,b in SCREEN_COMPARISONS}
        endpoint[name]={'arms':{a:{'auroc':points[a],'ci95':_ci(draws[a])} for a in SCREEN_ARMS},'paired_deltas':contrasts}
    macro={a:float(np.mean(macro_points[a])) for a in SCREEN_ARMS}
    drawmeans={a:np.mean(macro_draws[a],axis=0) for a in SCREEN_ARMS}
    return {'endpoints':endpoint,'macro_auroc':macro,'macro_paired_deltas':{
        a+'-'+b:{'auroc_delta':macro[a]-macro[b],'ci95':_ci(drawmeans[a]-drawmeans[b])} for a,b in SCREEN_COMPARISONS}}
