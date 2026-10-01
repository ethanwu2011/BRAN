"""In-memory, train-only block-regularized readouts. No data or filesystem IO."""
from __future__ import annotations
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROFILES = ((1.,1.,1.), (0.,1.,1.), (0.,0.,1.), (0.,1.,0.), (.25,.25,1.), (.25,1.,.25))
C_GRID = (.01,.1,1.,10.)
ALPHA_GRID = (.01,.1,1.,10.,100.,1000.,10000.)


class InnovationReadoutError(ValueError):
    pass


def require(condition, code='invalid_contract'):
    if not condition:
        raise InnovationReadoutError(code)


def innovation_coordinates(mean, retinal_loading, clinical_loading):
    """An invertible linear transform; no new information or calibrated covariance."""
    z=np.asarray(mean,dtype=np.float64)
    ar=np.asarray(retinal_loading,dtype=np.float64);ac=np.asarray(clinical_loading,dtype=np.float64)
    require(z.ndim==2 and z.shape[1]==192 and ar.shape==ac.shape==(64,64))
    require(all(np.isfinite(x).all() for x in (z,ar,ac)))
    shared=z[:,:64]
    return np.c_[shared,z[:,64:128]-shared@ar.T,z[:,128:]-shared@ac.T]


def inverse_innovation_coordinates(innovation, retinal_loading, clinical_loading):
    z=np.asarray(innovation,dtype=np.float64)
    ar=np.asarray(retinal_loading,dtype=np.float64);ac=np.asarray(clinical_loading,dtype=np.float64)
    require(z.ndim==2 and z.shape[1]==192 and ar.shape==ac.shape==(64,64))
    require(all(np.isfinite(x).all() for x in (z,ar,ac)))
    return np.c_[z[:,:64],z[:,64:128]+z[:,:64]@ar.T,z[:,128:]+z[:,:64]@ac.T]


def profile_weights(index):
    require(type(index) is int and 0<=index<len(PROFILES))
    return np.r_[np.repeat(np.asarray(PROFILES[index]),64),1.]


def _inputs(x,y,observed,inner,test_x,width):
    x=np.asarray(x,float);test_x=np.asarray(test_x,float);y=np.asarray(y,float)
    observed=np.asarray(observed);inner=np.asarray(inner)
    require(x.ndim==test_x.ndim==2 and x.shape[1]==test_x.shape[1]==193)
    require(np.isfinite(x).all() and np.isfinite(test_x).all())
    require(y.shape==((len(x),) if width==1 else (len(x),width)) and observed.shape==y.shape and observed.dtype.kind=='b')
    require(inner.shape==(len(x),) and inner.dtype.kind in 'iu' and set(inner.tolist())==set(range(5)))
    require(np.isfinite(y[observed]).all(),'nonfinite_observed_target')
    return x,y,observed,inner,test_x


def _fit_logistic(x,y,C):
    head=LogisticRegression(C=C,solver='lbfgs',max_iter=500)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        head.fit(x,y)
    return head, not any(issubclass(w.category,ConvergenceWarning) for w in caught)


def fit_screening(x_train,y,observed,inner_ids,x_test,*,selected_profiles=True):
    x,y,observed,inner,test=_inputs(x_train,y,observed,inner_ids,x_test,1)
    require(set(np.unique(y[observed]))=={0.,1.},'binary_target_required')
    splits=[]
    for f in range(5):
        tr=observed&(inner!=f);va=observed&(inner==f)
        require(tr.sum()>=2 and va.sum()>=2 and len(np.unique(y[tr]))==len(np.unique(y[va]))==2,'incomplete_inner_support')
        scaler=StandardScaler().fit(x[tr])
        splits.append((scaler.transform(x[tr]),y[tr],scaler.transform(x[va]),y[va]))
    choices=[];rejected=0
    for profile in (range(len(PROFILES)) if selected_profiles else (0,)):
        w=profile_weights(profile)
        for penalty in C_GRID:
            auc=[];loss=[];counts=[];converged=True
            for tr,yt,va,yv in splits:
                head,ok=_fit_logistic(tr*w,yt,penalty)
                if not ok:converged=False;break
                p=head.predict_proba(va*w)[:,1]
                require(np.isfinite(p).all(),'nonfinite_prediction')
                auc.append(roc_auc_score(yv,p));loss.append(log_loss(yv,p,labels=[0,1]));counts.append(len(yv))
            if not converged:rejected+=1;continue
            choices.append((-float(np.average(auc,weights=counts)),float(np.average(loss,weights=counts)),
                            penalty,int(np.count_nonzero(PROFILES[profile])),profile))
    require(bool(choices),'no_converged_candidate')
    _,_,penalty,_,profile=min(choices)
    scaler=StandardScaler().fit(x[observed]);w=profile_weights(profile)
    head,ok=_fit_logistic(scaler.transform(x[observed])*w,y[observed],penalty)
    require(ok,'selected_candidate_nonconvergence')
    prediction=head.predict_proba(scaler.transform(test)*w)[:,1]
    require(np.isfinite(prediction).all(),'nonfinite_prediction')
    return {'predictions':prediction,'profile_index':int(profile),'penalty':float(penalty),
            'rejected_nonconvergence':int(rejected)}


def fit_cbc(x_train,y,observed,inner_ids,x_test):
    x,y,observed,inner,test=_inputs(x_train,y,observed,inner_ids,x_test,9)
    splits=[]
    for f in range(5):
        tr=inner!=f;va=inner==f
        require(np.all(observed[tr].sum(0)>=2) and np.all(observed[va].sum(0)>=10),'incomplete_cbc_inner_support')
        scaler=StandardScaler().fit(x[tr])
        splits.append((scaler.transform(x[tr]),y[tr],observed[tr],scaler.transform(x[va]),y[va],observed[va]))
    outer_scale=StandardScaler().fit(x);outer_x=outer_scale.transform(x);outer_test=outer_scale.transform(test)
    out=np.empty((len(test),9));profiles=[];penalties=[]
    for j in range(9):
        choices=[]
        for profile in range(len(PROFILES)):
            w=profile_weights(profile)
            for penalty in ALPHA_GRID:
                total=0.;count=0
                for tr,yt,mt,va,yv,mv in splits:
                    head=Ridge(alpha=penalty).fit(tr[mt[:,j]]*w,yt[mt[:,j],j])
                    prediction=head.predict(va[mv[:,j]]*w)
                    total+=float(np.square(prediction-yv[mv[:,j],j]).sum());count+=int(mv[:,j].sum())
                require(np.isfinite(total) and count>0,'nonfinite_inner_score')
                # Stronger penalty, then fewer blocks, then fixed profile order break exact ties.
                choices.append((total/count,-penalty,int(np.count_nonzero(PROFILES[profile])),profile))
        _,negative_penalty,_,profile=min(choices);penalty=-negative_penalty;w=profile_weights(profile)
        head=Ridge(alpha=penalty).fit(outer_x[observed[:,j]]*w,y[observed[:,j],j])
        out[:,j]=head.predict(outer_test*w);profiles.append(int(profile));penalties.append(float(penalty))
    require(np.isfinite(out).all(),'nonfinite_prediction')
    return {'predictions':out,'profile_indices':profiles,'penalties':penalties}
