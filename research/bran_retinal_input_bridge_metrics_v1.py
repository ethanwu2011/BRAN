"""Closed aggregate old/new-input comparisons; patient arrays stay caller-local."""
import math
import numpy as np
import run_bran_overnight_diagnostic_v1 as base
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_retinal_input_bridge_v1 import VERSIONS, ROUTES, require

ARMS=tuple(version+"_"+route for version in VERSIONS for route in ROUTES)


def ci(draws):
    finite=draws[np.isfinite(draws)]
    require(len(finite)>=900)
    return [float(x) for x in np.quantile(finite,[.025,.975])]


def validate_counts(folds,counts,n):
    require(isinstance(folds,np.ndarray) and folds.shape==(n,) and folds.dtype.kind in "iu"
            and set(folds.tolist())==set(range(5)))
    require(isinstance(counts,np.ndarray) and counts.shape==(1000,n) and counts.dtype.kind in "iu")
    # Bounding every count prevents fixed-width sum overflow before mass checks.
    require((counts>=0).all() and (counts<=n).all())
    for f in range(5):
        ix=folds==f
        require((counts[:,ix].sum(axis=1)==int(ix.sum())).all())


def summarize(screen,labels,observed,folds,names,counts,truth,blood,blood_observed):
    n=len(folds);validate_counts(folds,counts,n)
    require(type(names) in (list,tuple) and len(names)==len(set(names))==26 and all(type(x)is str and x for x in names))
    require(type(screen)is dict and set(screen)==set(ARMS))
    require(isinstance(labels,np.ndarray) and isinstance(observed,np.ndarray)
            and labels.shape==observed.shape==(n,26) and observed.dtype==np.dtype(bool)
            and np.isin(labels[observed],[0,1]).all())
    for x in screen.values():
        require(isinstance(x,np.ndarray) and x.shape==(n,26) and x.dtype.kind=="f" and not np.isinf(x).any()
                and ((x[np.isfinite(x)]>=0)&(x[np.isfinite(x)]<=1)).all())
    require(isinstance(truth,np.ndarray) and isinstance(blood_observed,np.ndarray)
            and truth.shape==blood_observed.shape==(n,9) and blood_observed.dtype==np.dtype(bool)
            and np.isfinite(truth[blood_observed]).all())
    require(type(blood)is dict and set(blood)==set(VERSIONS))
    for x in blood.values():
        require(isinstance(x,np.ndarray) and x.shape==(n,9) and np.isfinite(x[blood_observed]).all())
    points={arm:[] for arm in ARMS};draws={arm:[] for arm in ARMS};endpoints={}
    for j,name in enumerate(names):
        valid=observed[:,j].copy()
        for x in screen.values():valid &= np.isfinite(x[:,j])
        contributing=np.zeros(n,bool)
        for f in range(5):
            at=valid&(folds==f)
            if len(np.unique(labels[at,j]))==2:contributing |= at
        if any(np.count_nonzero(contributing&(labels[:,j]==cls))<20 for cls in (0,1)):
            endpoints[name]={"status":"unsupported"};continue
        ps={arm:base.fold_weighted_auc(labels[:,j],x[:,j],contributing,folds) for arm,x in screen.items()}
        ds={arm:base._weighted_auc_draws(labels[:,j],x[:,j],contributing,folds,counts) for arm,x in screen.items()}
        endpoints[name]=screen_cell(ps,ds)
        for arm in ARMS:points[arm].append(ps[arm]);draws[arm].append(ds[arm])
    macro=screen_cell({a:float(np.mean(points[a])) for a in ARMS},
                      {a:np.mean(draws[a],axis=0) for a in ARMS}) if len(points[ARMS[0]])==26 else None
    completion={}
    for j,field in enumerate(CBC_FIELDS):
        valid=blood_observed[:,j];ix=np.flatnonzero(valid)
        if len(ix)<20:completion[field]={"status":"unsupported"};continue
        weights=counts[:,ix].astype(float);denom=weights.sum(1)
        def bootstrap(values):
            return np.divide(weights@values,denom,out=np.full(1000,np.nan),where=denom>0)
        error={v:blood[v][ix,j]-truth[ix,j] for v in VERSIONS}
        ae={v:np.abs(x) for v,x in error.items()}
        completion[field]={"status":"supported","arms":{v:{"mae":float(ae[v].mean()),
            "rmse":float(np.sqrt(np.square(error[v]).mean())),"ci95":ci(bootstrap(ae[v]))} for v in VERSIONS},
            "contrast":{"mae_delta":float(ae["authenticated"].mean()-ae["historical"].mean()),
                        "ci95":ci(bootstrap(ae["authenticated"]-ae["historical"]))}}
    result={"screening":{"endpoints":endpoints,"macro":macro},"whole_cbc":completion}
    validate(result,names)
    return result


def screen_cell(points,draws):
    return {"status":"supported","arms":{a:{"auroc":float(points[a]),"ci95":ci(draws[a])} for a in ARMS},
        "contrasts":{r:{"auroc_delta":float(points["authenticated_"+r]-points["historical_"+r]),
            "ci95":ci(draws["authenticated_"+r]-draws["historical_"+r])} for r in ROUTES}}


def finite(x):return type(x) in (float,int) and math.isfinite(x)


def check_ci(x,lower,upper):
    require(type(x)is list and len(x)==2 and all(finite(v) for v in x) and lower<=x[0]<=x[1]<=upper)


def validate_screen(cell):
    if cell=={"status":"unsupported"}:return
    require(type(cell)is dict and set(cell)=={"status","arms","contrasts"} and cell["status"]=="supported"
            and type(cell["arms"])is dict and set(cell["arms"])==set(ARMS)
            and type(cell["contrasts"])is dict and set(cell["contrasts"])==set(ROUTES))
    for x in cell["arms"].values():
        require(type(x)is dict and set(x)=={"auroc","ci95"} and finite(x["auroc"]) and 0<=x["auroc"]<=1)
        check_ci(x["ci95"],0,1)
    for route,x in cell["contrasts"].items():
        require(type(x)is dict and set(x)=={"auroc_delta","ci95"} and finite(x["auroc_delta"]))
        expected=cell["arms"]["authenticated_"+route]["auroc"]-cell["arms"]["historical_"+route]["auroc"]
        require(math.isclose(x["auroc_delta"],expected,rel_tol=0,abs_tol=1e-12))
        check_ci(x["ci95"],-1,1)


def validate(result,names):
    require(type(result)is dict and set(result)=={"screening","whole_cbc"})
    screen=result["screening"]
    require(type(screen)is dict and set(screen)=={"endpoints","macro"}
            and type(screen["endpoints"])is dict and set(screen["endpoints"])==set(names))
    for x in screen["endpoints"].values():validate_screen(x)
    all_supported=all(x["status"]=="supported" for x in screen["endpoints"].values())
    require((screen["macro"] is not None)==all_supported)
    if all_supported:
        validate_screen(screen["macro"])
        require(screen["macro"]["status"]=="supported")
        for a in ARMS:
            require(math.isclose(screen["macro"]["arms"][a]["auroc"],
                float(np.mean([x["arms"][a]["auroc"] for x in screen["endpoints"].values()])),abs_tol=1e-12,rel_tol=0))
    require(type(result["whole_cbc"])is dict and set(result["whole_cbc"])==set(CBC_FIELDS))
    for x in result["whole_cbc"].values():
        if x=={"status":"unsupported"}:continue
        require(type(x)is dict and set(x)=={"status","arms","contrast"} and x["status"]=="supported"
                and type(x["arms"])is dict and set(x["arms"])==set(VERSIONS))
        for arm in x["arms"].values():
            require(type(arm)is dict and set(arm)=={"mae","rmse","ci95"}
                    and all(finite(arm[k]) and arm[k]>=0 for k in ("mae","rmse")))
            check_ci(arm["ci95"],0,math.inf)
        d=x["contrast"]
        require(type(d)is dict and set(d)=={"mae_delta","ci95"} and finite(d["mae_delta"]))
        require(math.isclose(d["mae_delta"],x["arms"]["authenticated"]["mae"]-x["arms"]["historical"]["mae"],abs_tol=1e-12,rel_tol=0))
        check_ci(d["ci95"],-math.inf,math.inf)
