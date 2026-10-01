"""Matched fixed probes of sealed historical V5 C/M states; local arrays only.

This isolates source values beyond mask exposure, not all multisource learning.
It never fits an encoder or uses a protected external source.
"""
import warnings
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bran_multisource_batches_v2 import tensor
from bran_multisource_outcomes_v2 import original_age
from bran_multisource_profiles_v3 import _validate_provider,_unchanged
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_v5_state_routes import state_routes
from bran_external_cbc_evaluation_v1 import paired_counts
import bran_multisource_outcome_metrics_v2 as metrics

ERROR='source_value_readout_v6_invalid'
PARAMETERS={'roles':['C','M'],'role_semantics':{'C':'paired_values_source_masks','M':'paired_and_source_values_source_masks'},
    'checkpoint':'sealed_V5_attempt2','routes':['both','clinical','retinal'],
    'readout':'StandardScaler_L2_logistic_C1_lbfgs_max5000_default_class_policy',
    'encoder_updates':0,'folds':5,'endpoints':26,'draws':1000,'seed':98571,'minimum_valid':900,
    'primary':'both_M_minus_C','secondary_routes':['clinical','retinal'],
    'interpretation':'source_values_beyond_mask_exposure_not_paired_only_control'}


def require(ok):
    if not ok:raise ValueError(ERROR) from None


def fixed_probe(x_train,y_train,observed_train,x_test):
    try:
        mask=np.asarray(observed_train,bool);y=np.asarray(y_train)
        require(mask.shape==y.shape==(len(x_train),) and mask.sum()>=20 and set(np.unique(y[mask]))=={0,1})
        require(np.isfinite(x_train[mask]).all() and np.isfinite(x_test).all())
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,penalty='l2',solver='lbfgs',max_iter=5000))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(np.asarray(x_train)[mask],y[mask])
        require(not any(issubclass(w.category,ConvergenceWarning) for w in caught))
        return model.predict_proba(x_test)[:,1]
    except Exception:raise ValueError(ERROR) from None


def evaluate(paired,provider,progress=None):
    try:
        folds=paired.folds;n=len(folds);age=original_age(paired)
        slots=tuple(paired.names.index(f) for f in CBC_FIELDS)
        predictions={route:{role:np.full((n,26),np.nan) for role in ('C','M')} for route in PARAMETERS['routes']}
        for fold in range(5):
            designs={}
            for role in ('C','M'):
                if progress:progress('state_inference',fold,role)
                model,t=provider(role,fold)
                before,pin,grad=_validate_provider(model,t,fold,slots,paired.transforms[fold])
                c,cm=t.clinical(paired.c,paired.cm);r,rm=t.retinal(paired.r,paired.rm)
                args=(tensor(c),tensor(cm,torch.bool),tensor(r),tensor(rm,torch.bool),age,t.age_mean,t.age_scale)
                designs[role]=state_routes(model,*args)
                require(_unchanged(before,pin,grad,model,t))
                replay,rt=provider(role,fold)
                require(replay is not model)
                rb,rpin,rgrad=_validate_provider(replay,rt,fold,slots,paired.transforms[fold])
                rvalue=state_routes(replay,*args)
                require(all(torch.equal(rvalue.states[k],designs[role].states[k]) and
                    torch.equal(rvalue.available[k],designs[role].available[k]) for k in PARAMETERS['routes']))
                require(_unchanged(rb,rpin,rgrad,replay,rt))
                del model,replay,rvalue
            train,test=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
            for route in PARAMETERS['routes']:
                common=np.logical_and.reduce([designs[role].available[route].numpy() for role in ('C','M')])
                for role in ('C','M'):
                    if progress:progress('fixed_readout',fold,role)
                    x=designs[role].states[route].numpy()
                    eligible_test=test[common[test]]
                    for j in range(26):
                        mask=paired.labelmask[:,j]&common
                        predictions[route][role][eligible_test,j]=fixed_probe(
                            x[train],paired.labels[train,j],mask[train],x[eligible_test])
            del designs
        if progress:progress('aggregate_bootstrap',None,None)
        counts=paired_counts(folds,draws=1000,seed=98571)
        result={route:metrics.screening(value,paired.labels,paired.labelmask,folds,
            tuple(paired.endpoint_names),counts,{'M_minus_C':('M','C')}) for route,value in predictions.items()}
        return {'schema':'bran-v5-source-value-fixed-readout','parameters':PARAMETERS,'routes':result,
            'all_state_inference_replayed':True,'encoder_updated':False,'candidate_promoted':False,
            'protected_sources_used':False,'patient_level_output_emitted':False}
    except Exception:raise ValueError(ERROR) from None
