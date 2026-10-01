"""Local-only replay of two locked candidates, nuisance and independent CGM tests."""
import math
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import bran_other_disease_panel_evaluation_v1 as parent
import bran_disease_structure_192_v1 as structure
import bran_cgm_added_value_evaluation_v1 as cgm
import bran_subgroup_utility_kernel_v1 as utility

CODES=('mhoccur_clsh','mhoccur_oa')
DESIGNS=('age','technical','metabolic','combined')
SPLITS={'validation':3,'replication':4}
PARAMETERS={
    'candidates':list(CODES),'locked_k':2,'new_k_search':False,'new_stability_bootstraps':False,
    'encoder_width':192,'encoder_steps':1500,'encoder_seed':1701,'discovery_folds':[0,1,2],
    'pca':{'components':8,'whiten':True,'svd_solver':'full'},
    'mixture':{'k':2,'covariance':'diag','reg_covar':0.0001,'n_init':5,'max_iter':500,'seed':93501},
    'replay':{'validation_density_and_all_support_permitted_profiles':True,'atol':1e-10,'rtol':1e-10,
              'both_candidates_before_any_new_head':True},
    'minimum_groups_per_split':20,'nuisance_designs':list(DESIGNS),
    'metabolic_fields':list(cgm.SEVERITY_FIELDS),'technical':'48_clinical_masks_retina_presence_discovery_site_onehot_unknown_flag',
    'combined':'age_five_metabolic_values_plus_technical_no_duplicate_mask_columns',
    'site_vocabulary':'disease_member_discovery_only','clinical_scaling':'canonical_discovery_fold_transform_then_head_discovery_standardization',
    'logistic':{'C':1.0,'solver':'lbfgs','max_iter':4000,'class_weight':None,'metric':'AUROC','tuning':False},
    'cgm_baseline':'same_combined_nuisance_design','group_addition':'one_binary_locked_indicator_0_or_1',
    'ridge_alphas':[0.1,10.0,1000.0],'ridge_selection':'observed_discovery_three_fold_count_weighted_mse_larger_alpha_tie',
    'trees':{'n_estimators':128,'max_depth':8,'min_samples_leaf':5,'max_features':1.0,'n_jobs':1,'seed':94501},
    'cgm_support':[80,40,40],'cgm_inner_support':10,'bootstrap_resamples':500,'bootstrap_seed':94501,
    'normalization':'squared_error_divided_by_observed_discovery_target_iqr_squared_floor_1e-6',
    'primary':'replication_severity_group_extra_trees_minus_severity_extra_trees',
    'inference':'exploratory_marginal_fixed_fit_no_multiplicity_adjusted_success_gate',
    'sex_treatments_adjusted':False,'target_used_for_groups_or_predictors':False,
}

class FixedSubgroupError(ValueError): pass

def require(ok):
    if not ok: raise FixedSubgroupError('fixed_subgroup_contract_failed')

def compare_replay(actual,expected):
    """Strict keys/types; numeric tolerances only for previously published floats."""
    require(type(actual)is type(expected))
    if isinstance(actual,dict):
        require(set(actual)==set(expected))
        for key in expected: compare_replay(actual[key],expected[key])
    elif isinstance(actual,list):
        require(len(actual)==len(expected))
        for a,b in zip(actual,expected,strict=True): compare_replay(a,b)
    elif type(actual)is float:
        require(math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10))
    else: require(actual==expected)

def locked_fit(discovery):
    """Only K=2; no validation/outcome argument and no selection stage."""
    d=np.asarray(discovery,dtype=float)
    require(d.ndim==2 and d.shape[1]==192 and len(d)>=80 and np.isfinite(d).all())
    scaler=StandardScaler(); ds=scaler.fit_transform(d)
    pca=PCA(n_components=8,whiten=True,svd_solver='full'); reduced=pca.fit_transform(ds)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        mix=GaussianMixture(n_components=2,covariance_type='diag',reg_covar=1e-4,
            n_init=5,max_iter=500,random_state=93501).fit(reduced)
    require(bool(mix.converged_))
    return structure.StructureResult(structure.STATUS_SUPPORTED,scaler,pca,mix,(),2)

def group_support(groups,member,outer):
    return {label:bool(all(np.sum(member & rows & (groups==g))>=20 for g in (0,1)))
        for label,rows in (('discovery',outer<3),('validation',outer==3),('replication',outer==4))}

def make_designs(c,mask,rm,age,names,sites,fit):
    n=len(c); require(c.shape==mask.shape==(n,59) and mask.dtype==np.dtype(bool) and not mask[:,48:].any())
    require(rm.shape==age.shape==(n,) and rm.dtype==np.dtype(bool) and len(sites)==n)
    require(all(type(s)is str and bool(s) for s in sites) and len(fit)>0)
    indices=np.array([tuple(names).index(k) for k in cgm.SEVERITY_FIELDS]); require(np.all(indices<48))
    categories=tuple(sorted({sites[i] for i in fit}))
    onehot=np.array([[float(s==t) for t in categories] for s in sites])
    unknown=np.array([s not in categories for s in sites],float)
    technical=np.column_stack((mask[:,:48].astype(float),rm.astype(float),onehot,unknown))
    metabolic=np.column_stack((age,c[:,indices],mask[:,indices].astype(float)))
    combined=np.column_stack((age,c[:,indices],technical))
    designs={'age':age[:,None],'technical':technical,'metabolic':metabolic,'combined':combined}
    require(all(np.isfinite(x).all() for x in designs.values()))
    return designs

def nuisance(designs,groups,member,outer):
    require(set(designs)==set(DESIGNS))
    support=group_support(groups,member,outer)
    require(all(support.values()))
    fit=member&(outer<3); score=member&(outer>=3)
    result={'status':'descriptive_membership_prediction_not_causal','support':support,'splits':{s:{} for s in SPLITS}}
    for name in DESIGNS:
        x=designs[name]
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            scaler=StandardScaler().fit(x[fit])
            model=LogisticRegression(C=1.0,solver='lbfgs',max_iter=4000,class_weight=None)
            model.fit(scaler.transform(x[fit]),groups[fit])
            p=model.predict_proba(scaler.transform(x[score]))[:,1]
        require(np.isfinite(p).all())
        for split,fold in SPLITS.items():
            take=outer[score]==fold
            result['splits'][split][name]={'auroc':float(roc_auc_score(groups[score][take],p[take]))}
    validate_nuisance(result); return result

def validate_nuisance(r):
    require(isinstance(r,dict) and set(r)=={'status','support','splits'})
    require(r['status']=='descriptive_membership_prediction_not_causal' and r['support']=={k:True for k in ('discovery','validation','replication')})
    require(all(type(v)is bool for v in r['support'].values()) and isinstance(r['splits'],dict) and set(r['splits'])==set(SPLITS))
    for side in r['splits'].values():
        require(isinstance(side,dict) and set(side)==set(DESIGNS))
        for item in side.values():
            require(isinstance(item,dict) and set(item)=={'auroc'} and type(item['auroc'])is float
                and math.isfinite(item['auroc']) and 0<=item['auroc']<=1)

def evaluate(c0,cm0,eligible,r0,rm,names,ages,outer,ids,memberships,sites,y,observed,baseline,
             *,steps=1500,progress=None,trainer=None,infer=None,fitter=None,nuisance_evaluator=None,utility_evaluator=None):
    import run_bran_anchor_ablation_v2 as paired
    from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
    from run_bran_frozen_cbc_readout_v1 import _encoder_hash
    parent.validate_export(baseline)
    c0,cm0,eligible,r0,rm,ages,outer,y,observed=map(np.asarray,(c0,cm0,eligible,r0,rm,ages,outer,y,observed))
    n=len(c0); require(c0.shape==cm0.shape==eligible.shape==(n,59) and r0.shape==(n,384))
    require(all(v.shape==(n,) for v in (rm,ages,outer,y,observed)))
    require(all(v.dtype==np.dtype(bool) for v in (cm0,eligible,rm,observed)))
    require(outer.dtype.kind in 'iu' and set(outer.tolist())==set(range(5)))
    require(len(names)==len(set(names))==59 and set(cgm.SEVERITY_FIELDS)<=set(names))
    require(len(ids)==len(set(ids))==n and all(type(i)is str and bool(i) for i in ids))
    require(set(memberships)==set(CODES) and all(v.shape==(n,) and v.dtype==np.dtype(bool) for v in memberships.values()))
    require(np.isfinite(c0[cm0]).all() and np.isfinite(r0[rm]).all() and np.isfinite(ages).all())
    require(y.dtype.kind=='f' and np.isfinite(y).all() and np.all(y[observed]>0))
    require(type(steps)is int and steps>0)
    local_eligible=eligible.copy(); local_eligible[:,48:]=False; fit=np.flatnonzero(outer<3)
    transform=paired.base.FoldTransform(c0,cm0,local_eligible,r0,rm,ages,fit)
    c,cm,r,age=transform.apply(c0,cm0,local_eligible,r0,rm,ages)
    if progress: progress('training')
    model=(paired._train if trainer is None else trainer)(BRANClinicalAnchorV2,c,cm,r,rm,age,fit,1701,steps=steps)
    model.eval(); before=_encoder_hash(model)
    states=np.asarray((parent.old._inference if infer is None else infer)(model,c,cm,r,rm,age),dtype=float)
    require(states.shape==(n,192) and np.isfinite(states).all())
    values={'age_years':ages,'manifest_average_cgm_glucose_mg_dl':y}
    masks={'age_years':np.ones(n,bool),'manifest_average_cgm_glucose_mg_dl':observed}
    for field in parent.FIELDS[1:-1]:
        j=tuple(names).index(field); require(j<48)
        values[field]=c0[:,j]; masks[field]=cm0[:,j]&local_eligible[:,j]
    assignments={}
    if progress: progress('locked_replay')
    # BOTH candidates must replay before any new nuisance/target head is fit.
    for code in CODES:
        member=memberships[code]; old=baseline['candidates'][code]
        require(old['status']=='evaluated' and old['structure']['selected_k']==2)
        require(old['diagnostics']['status']==parent.diagnostics.STATUS_SUPPORTED)
        compare_replay(parent.support_flags(member,outer),old['support'])
        locked=(locked_fit if fitter is None else fitter)(states[member&(outer<3)])
        groups=locked.predict(states); require(groups.shape==(n,) and groups.dtype.kind in 'iu' and np.all((groups==0)|(groups==1)))
        require(all(group_support(groups,member,outer).values()))
        v=locked._pca.transform(locked._scaler.transform(states[member&(outer==3)]))
        density=float(np.mean(locked._mixture.score_samples(v)))
        compare_replay(density,old['structure']['candidates']['2']['validation_mean_log_density'])
        compare_replay(parent.profiles(locked,states,member,outer,values,masks),old['profiles'])
        assignments[code]=groups
    result={'encoder_fits':1,'state_dimension':192,'locked_k':2,'both_candidates_replayed_before_heads':True,
        'encoder_unchanged_after_heads':True,'patient_arrays_serialized':False,'novel_subtype_claimed':False,
        'clinical_utility_established':False,'default_promotion':False,'candidates':{}}
    if progress: progress('heads')
    for code in CODES:
        member=memberships[code]; groups=assignments[code]
        designs=make_designs(c,cm,rm,age,tuple(names),tuple(sites),np.flatnonzero(member&(outer<3)))
        nr=(nuisance if nuisance_evaluator is None else nuisance_evaluator)(designs,groups,member,outer)
        severity=designs['combined']
        ur=(utility.fit_and_evaluate if utility_evaluator is None else utility_evaluator)(
            {'severity':severity,'severity_group':np.column_stack((severity,groups.astype(float)))},
            y,observed,outer,member,ids=ids)
        utility.validate_report(ur); require(ur['status']!='numerical_failure')
        result['candidates'][code]={'nuisance':nr,'cgm_utility':ur}
    require(_encoder_hash(model)==before); validate_result(result); return result

def validate_result(r):
    require(isinstance(r,dict) and set(r)=={'encoder_fits','state_dimension','locked_k','both_candidates_replayed_before_heads',
        'encoder_unchanged_after_heads','patient_arrays_serialized','novel_subtype_claimed','clinical_utility_established','default_promotion','candidates'})
    for name,value in (('encoder_fits',1),('state_dimension',192),('locked_k',2)):
        require(type(r[name])is int and r[name]==value)
    require(r['both_candidates_replayed_before_heads'] is True and r['encoder_unchanged_after_heads'] is True)
    require(all(r[k] is False for k in ('patient_arrays_serialized','novel_subtype_claimed','clinical_utility_established','default_promotion')))
    require(isinstance(r['candidates'],dict) and set(r['candidates'])==set(CODES))
    for item in r['candidates'].values():
        require(isinstance(item,dict) and set(item)=={'nuisance','cgm_utility'})
        validate_nuisance(item['nuisance']); utility.validate_report(item['cgm_utility'])
        require(item['cgm_utility']['status'] in {'ok','unsupported_target_support'})
