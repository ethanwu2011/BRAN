"""Portable array-only linear research heads and split-safe residual intervals."""
from __future__ import annotations
from dataclasses import dataclass
import math
import warnings
import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from bran_clinical_semantics_v1 import CBC_FIELDS

_INVALID = "research linear heads inputs invalid"

def _bad(): raise ValueError(_INVALID)

@dataclass(frozen=True, repr=False)
class ScreeningHead:
    mean: np.ndarray; scale: np.ndarray; coef: np.ndarray; intercept: np.ndarray; supported: np.ndarray

@dataclass(frozen=True, repr=False)
class CompletionHead:
    mean: np.ndarray; scale: np.ndarray; coef: np.ndarray; intercept: np.ndarray; supported: np.ndarray

@dataclass(frozen=True, repr=False)
class ResidualCalibration:
    quantile: np.ndarray; valid: np.ndarray

def _indices(value, n):
    x=np.asarray(value)
    if x.ndim!=1 or x.dtype.kind not in 'iu' or len(x)==0 or np.any(x<0) or np.any(x>=n) or len(np.unique(x))!=len(x): _bad()
    return x.astype(int,copy=False)

def validate_splits(train, calibration, test, n):
    a,b,c=(_indices(v,n) for v in (train,calibration,test))
    if np.intersect1d(a,b).size or np.intersect1d(a,c).size or np.intersect1d(b,c).size: _bad()
    return a,b,c

def _x(x):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=193 or not np.isfinite(x).all(): _bad()
    return x

def _binary(y,m,n,cols):
    y=np.asarray(y,float); m=np.asarray(m)
    if y.shape!=(n,cols) or m.shape!=y.shape or m.dtype!=np.dtype(bool) or np.any(m&(~np.isfinite(y)|((y!=0)&(y!=1)))): _bad()
    return y,m

def fit_screening(features, labels, observed, train_indices):
    x=_x(features); y,m=_binary(labels,observed,len(x),26); train=_indices(train_indices,len(x))
    mean=np.zeros((26,193)); scale=np.ones((26,193)); coef=np.zeros((26,193)); intercept=np.zeros(26); supported=np.zeros(26,bool)
    for j in range(26):
        rows=train[m[train,j]]
        if len(rows)<20 or np.sum(y[rows,j]==0)<2 or np.sum(y[rows,j]==1)<2: continue
        scaler=StandardScaler().fit(x[rows]); model=LogisticRegression(C=1.,solver='lbfgs',max_iter=5000)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning); model.fit(scaler.transform(x[rows]),y[rows,j])
        if any(issubclass(item.category,ConvergenceWarning) for item in caught): continue
        mean[j],scale[j],coef[j],intercept[j]=scaler.mean_,scaler.scale_,model.coef_[0],model.intercept_[0]; supported[j]=True
    head=ScreeningHead(mean,scale,coef,intercept,supported); _validate_head(head,26,ScreeningHead); return head

def predict_screening(head, features, abstain):
    x=_x(features); a=np.asarray(abstain)
    if a.shape!=(len(x),) or a.dtype!=np.dtype(bool): _bad()
    _validate_head(head,26,ScreeningHead)
    z=(x[:,None,:]-head.mean[None])/head.scale[None]
    p=expit((z*head.coef[None]).sum(2)+head.intercept[None])
    p[:,~head.supported]=np.nan; p[a]=np.nan; return p

def _continuous(y,m,n,cols):
    y=np.asarray(y,float);m=np.asarray(m)
    if y.shape!=(n,cols) or m.shape!=y.shape or m.dtype!=np.dtype(bool) or np.any(m&~np.isfinite(y)): _bad()
    return y,m

def fit_completion(features, targets, observed, train_indices):
    x=_x(features);y,m=_continuous(targets,observed,len(x),9);train=_indices(train_indices,len(x))
    mean=np.zeros((9,193));scale=np.ones((9,193));coef=np.zeros((9,193));intercept=np.zeros(9);supported=np.zeros(9,bool)
    for j in range(9):
        rows=train[m[train,j]]
        if len(rows)<20: continue
        scaler=StandardScaler().fit(x[rows]); model=Ridge(alpha=1.).fit(scaler.transform(x[rows]),y[rows,j])
        mean[j],scale[j],coef[j],intercept[j]=scaler.mean_,scaler.scale_,model.coef_,model.intercept_;supported[j]=True
    head=CompletionHead(mean,scale,coef,intercept,supported); _validate_head(head,9,CompletionHead); return head

def _validate_head(head, cols, kind):
    if not isinstance(head,kind): _bad()
    fields=(head.mean,head.scale,head.coef,head.intercept,head.supported)
    if (any(not isinstance(v,np.ndarray) for v in fields) or any(v.dtype.kind != 'f' for v in fields[:4]) or head.mean.shape!=(cols,193) or head.scale.shape!=head.mean.shape or head.coef.shape!=head.mean.shape or head.intercept.shape!=(cols,) or head.supported.shape!=(cols,) or head.supported.dtype!=np.dtype(bool) or not np.isfinite(head.mean).all() or not np.isfinite(head.scale).all() or not np.isfinite(head.coef).all() or not np.isfinite(head.intercept).all() or np.any(head.scale<=0)): _bad()

def _completion_mean(head,x):
    _validate_head(head,9,CompletionHead); return ((x[:,None,:]-head.mean[None])/head.scale[None]*head.coef[None]).sum(2)+head.intercept[None]

def calibrate_completion(head, features, targets, observed, train_indices, calibration_indices, abstain):
    """Calibrate only on caller-certified held-out rows.

    The root caller proves encoder/calibration isolation and supplies train rows
    matching this head fit.  Numeric portable heads store no row identifiers.
    """
    x=_x(features);y,m=_continuous(targets,observed,len(x),9);train=_indices(train_indices,len(x));calibration=_indices(calibration_indices,len(x))
    if np.intersect1d(train,calibration).size: _bad()
    a=np.asarray(abstain)
    if a.shape!=(len(x),) or a.dtype!=np.dtype(bool): _bad()
    pred=_completion_mean(head,x);q=np.zeros(9);valid=np.zeros(9,bool)
    for j in range(9):
        rows=calibration[m[calibration,j]&~a[calibration]&head.supported[j]]
        if len(rows)<20: continue
        residual=np.sort(np.abs(pred[rows,j]-y[rows,j])); rank=min(len(rows)-1,math.ceil((len(rows)+1)*.90)-1)
        q[j]=residual[rank];valid[j]=True
    calibration=ResidualCalibration(q,valid)
    if not np.isfinite(calibration.quantile).all() or np.any(calibration.quantile<0): _bad()
    return calibration

def predict_completion(head, calibration, features, abstain):
    x=_x(features);a=np.asarray(abstain)
    if not isinstance(calibration,ResidualCalibration) or a.shape!=(len(x),) or a.dtype!=np.dtype(bool) or not isinstance(calibration.quantile,np.ndarray) or not isinstance(calibration.valid,np.ndarray) or calibration.quantile.shape!=(9,) or calibration.quantile.dtype.kind != 'f' or calibration.valid.shape!=(9,) or calibration.valid.dtype!=np.dtype(bool) or not np.isfinite(calibration.quantile).all() or np.any(calibration.quantile<0): _bad()
    mean=_completion_mean(head,x); active=head.supported&calibration.valid
    mean[:,~active]=np.nan;mean[a]=np.nan; q=calibration.quantile[None]
    return mean, mean-q, mean+q

def screening_metrics(labels, observed, probabilities, test_indices):
    p=np.asarray(probabilities,float);y,m=_binary(labels,observed,len(p),26);test=_indices(test_indices,len(p))
    if p.shape!=(len(y),26) or np.any(np.isinf(p)) or np.any(np.isfinite(p)&((p<0)|(p>1))): _bad()
    out=[]
    for j in range(26):
        rows=test[m[test,j]&np.isfinite(p[test,j])]; cases=np.sum(y[rows,j]==1);controls=np.sum(y[rows,j]==0)
        out.append(None if cases<20 or controls<20 else {"auroc":float(roc_auc_score(y[rows,j],p[rows,j]))})
    return out

def validate_screening_metrics(metrics):
    if not isinstance(metrics,list) or len(metrics)!=26: _bad()
    for item in metrics:
        if item is not None and (not isinstance(item,dict) or set(item)!={"auroc"} or not isinstance(item["auroc"],float) or not math.isfinite(item["auroc"]) or not 0<=item["auroc"]<=1): _bad()

def completion_metrics(targets, observed, mean, lower, upper, test_indices):
    y,m=_continuous(targets,observed,len(mean),9);mean=np.asarray(mean,float);lower=np.asarray(lower,float);upper=np.asarray(upper,float);test=_indices(test_indices,len(y))
    if any(v.shape!=(len(y),9) for v in (mean,lower,upper)) or any(np.isinf(v).any() for v in (mean,lower,upper)): _bad()
    out=[]
    for j in range(9):
        triple=np.isfinite(mean[:,j])&np.isfinite(lower[:,j])&np.isfinite(upper[:,j])
        partial=(np.isfinite(mean[:,j])|np.isfinite(lower[:,j])|np.isfinite(upper[:,j]))&~triple
        if np.any(m[test,j]&partial[test]) or np.any(triple[test]&(lower[test,j]>upper[test,j])): _bad()
        rows=test[m[test,j]&triple[test]]
        if len(rows)<20: out.append(None);continue
        error=mean[rows,j]-y[rows,j]; den=np.sum((y[rows,j]-np.mean(y[rows,j]))**2)
        out.append({"mae":float(np.mean(np.abs(error))),"rmse":float(np.sqrt(np.mean(error**2))),"r2":None if den==0 else float(1-np.sum(error**2)/den),"coverage90":float(np.mean((y[rows,j]>=lower[rows,j])&(y[rows,j]<=upper[rows,j]))),"mean_width":float(np.mean(upper[rows,j]-lower[rows,j]))})
    return out

def validate_completion_metrics(metrics):
    keys={"mae","rmse","r2","coverage90","mean_width"}
    if not isinstance(metrics,list) or len(metrics)!=9: _bad()
    for item in metrics:
        if item is None: continue
        if not isinstance(item,dict) or set(item)!=keys or any(not isinstance(item[key],float) or not math.isfinite(item[key]) for key in keys-{"r2"}): _bad()
        if item["r2"] is not None and (not isinstance(item["r2"],float) or not math.isfinite(item["r2"]) or item["r2"]>1): _bad()
        if item["mae"]<0 or item["rmse"]<0 or item["mean_width"]<0 or not 0<=item["coverage90"]<=1 or item["mae"]>item["rmse"]+1e-12*max(1.,item["rmse"]): _bad()
