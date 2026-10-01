"""Fixed-encoder bootstrap stability diagnostic for the locked 192-state structure fit.

It makes no clinical-validity, stability, efficacy, or novel-subtype claim.  It has no
clinical-outcome inputs; independent characterization remains a future protocol.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import bran_disease_structure_192_v1 as structure

BOOTSTRAPS=50
MIN_SUCCESSFUL_BOOTSTRAPS=45
QUANTILES=(.025,.5,.975)
STATUS_SUPPORTED='supported'
STATUS_NO_DISCRETE_GROUPS='no_discrete_groups'
STATUS_UNSUPPORTED='unsupported'

@dataclass(repr=False)
class StabilityResult:
    status:str
    _locked:structure.StructureResult|None
    _aggregate:dict[str,Any]|None

def diagnose(discovery_states:Any,validation_states:Any,replication_states:Any,*,locked_model:structure.StructureResult|None=None)->StabilityResult:
    discovery=structure._states(discovery_states,'discovery_states');validation=structure._states(validation_states,'validation_states');replication=structure._states(replication_states,'replication_states')
    if len(discovery)<structure.MIN_DISCOVERY_ROWS or len(validation)<structure.MIN_VALIDATION_ROWS or len(replication)<structure.MIN_VALIDATION_ROWS:
        return StabilityResult(STATUS_UNSUPPORTED,None,None)
    locked=structure.fit_structure(discovery,validation) if locked_model is None else locked_model
    if locked.status!=structure.STATUS_SUPPORTED:return StabilityResult(STATUS_UNSUPPORTED,locked,None)
    k=locked._selected_k
    if type(k) is not int or k not in structure.K_VALUES:return StabilityResult(STATUS_UNSUPPORTED,locked,None)
    if k==1:return StabilityResult(STATUS_NO_DISCRETE_GROUPS,locked,{'status':STATUS_NO_DISCRETE_GROUPS,'selected_k':1})
    original=[locked.predict(values) for values in (discovery,validation,replication)]
    support=bool(np.all(np.bincount(original[2],minlength=k)>=structure.MIN_GROUP_ROWS))
    if not support:return StabilityResult('unsupported_replication_support',locked,{'status':'unsupported_replication_support','selected_k':k,'replication_group_support_at_least_20':False})
    rng=np.random.default_rng(structure.RANDOM_STATE);scores=[[],[],[]];converged=0;failed=0
    for _ in range(BOOTSTRAPS):
        rows=rng.integers(0,len(discovery),size=len(discovery));sample=discovery[rows]
        try:
            scaler=StandardScaler();sample_scaled=scaler.fit_transform(sample)
            # Refit exactly the locked K; there is deliberately no candidate search here.
            pca=PCA(n_components=8,whiten=True,svd_solver='full');sample_r=pca.fit_transform(sample_scaled)
            mixture=GaussianMixture(n_components=k,covariance_type='diag',reg_covar=1e-4,n_init=5,max_iter=500,random_state=structure.RANDOM_STATE).fit(sample_r)
            if not mixture.converged_:failed+=1;continue
            draw_scores=[]
            for index,values in enumerate((discovery,validation,replication)):
                labels=mixture.predict(pca.transform(scaler.transform(values)))
                score=float(adjusted_rand_score(original[index],labels))
                if not math.isfinite(score):raise FloatingPointError('nonfinite bootstrap ARI')
                draw_scores.append(score)
            for index,score in enumerate(draw_scores):scores[index].append(score)
            converged+=1
        except Exception:
            failed+=1
    available=converged>=MIN_SUCCESSFUL_BOOTSTRAPS
    aggregate={'status':STATUS_SUPPORTED,'selected_k':k,'group_count_within_1_to_4':True,'replication_group_support_at_least_20':True,'convergence':{'converged_bootstraps':converged,'failed_bootstraps':failed,'all_bootstraps_converged':failed==0},'fixed_fit_aggregate_stability_available':available,'ari_quantiles':{name:_quantile(values) if available else None for name,values in zip(('discovery','validation','replication'),scores,strict=True)}}
    return StabilityResult(STATUS_SUPPORTED,locked,aggregate if validate_report(aggregate) else None)

def _quantile(values:list[float])->dict[str,float]|None:
    if not values:return None
    q=np.quantile(values,QUANTILES);return {'q025':float(q[0]),'q500':float(q[1]),'q975':float(q[2])}

def aggregate_report(result:Any)->dict[str,Any]|None:
    if not isinstance(result,StabilityResult) or result._aggregate is None:return None
    return result._aggregate if validate_report(result._aggregate) else None

def validate_report(value:Any)->bool:
    if not isinstance(value,dict):return False
    if value.get('status')==STATUS_NO_DISCRETE_GROUPS:return set(value)=={'status','selected_k'} and type(value['selected_k']) is int and value['selected_k']==1
    if value.get('status')=='unsupported_replication_support':return set(value)=={'status','selected_k','replication_group_support_at_least_20'} and type(value['selected_k']) is int and 2<=value['selected_k']<=4 and value['replication_group_support_at_least_20'] is False
    if set(value)!={'status','selected_k','group_count_within_1_to_4','replication_group_support_at_least_20','convergence','fixed_fit_aggregate_stability_available','ari_quantiles'} or value['status']!=STATUS_SUPPORTED:return False
    if type(value['selected_k']) is not int or not 2<=value['selected_k']<=4 or any(type(value[x]) is not bool for x in ('group_count_within_1_to_4','replication_group_support_at_least_20','fixed_fit_aggregate_stability_available')) or value['group_count_within_1_to_4'] is not True or value['replication_group_support_at_least_20'] is not True:return False
    c=value['convergence']
    if not isinstance(c,dict) or set(c)!={'converged_bootstraps','failed_bootstraps','all_bootstraps_converged'} or type(c['converged_bootstraps']) is not int or type(c['failed_bootstraps']) is not int or not(0<=c['converged_bootstraps']<=BOOTSTRAPS and 0<=c['failed_bootstraps']<=BOOTSTRAPS) or c['converged_bootstraps']+c['failed_bootstraps']!=BOOTSTRAPS or type(c['all_bootstraps_converged']) is not bool or c['all_bootstraps_converged']!=(c['failed_bootstraps']==0):return False
    q=value['ari_quantiles']
    if not isinstance(q,dict) or set(q)!={'discovery','validation','replication'}:return False
    for values in q.values():
        if values is None:
            if value['fixed_fit_aggregate_stability_available'] or c['converged_bootstraps']>=MIN_SUCCESSFUL_BOOTSTRAPS:return False
            continue
        if not value['fixed_fit_aggregate_stability_available'] or c['converged_bootstraps']<MIN_SUCCESSFUL_BOOTSTRAPS:return False
        if not isinstance(values,dict) or set(values)!={'q025','q500','q975'} or any(type(x) is not float or not math.isfinite(x) or x<-1 or x>1 for x in values.values()) or not values['q025']<=values['q500']<=values['q975']:return False
    return value['fixed_fit_aggregate_stability_available']==(c['converged_bootstraps']>=MIN_SUCCESSFUL_BOOTSTRAPS)
