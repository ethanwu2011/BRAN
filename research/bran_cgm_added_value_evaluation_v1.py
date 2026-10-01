"""Local fixed192-state/clinical-design evaluator for independent CGM targets."""
import numpy as np
import bran_cgm_added_value_kernel_v1 as kernel
from bran_disease_structure_evaluation_v1 import _inference

SEVERITY_FIELDS=('hba1c','glucose','vit_bmi_vsorres','vit_sysbp_vsorres','vit_diabp_vsorres')
PARAMETERS={'state_width':192,'encoder_steps':1500,'encoder_seed':1701,'discovery_folds':[0,1,2],
    'severity_fields':list(SEVERITY_FIELDS),'severity_context':['age','field_masks','clinical_observed_fraction','retina_available','discovery_site_indicators','unseen_site'],
    'sex_available':False,'cgmtime_metadata_in_predictors':False,'diagnosis_history_in_predictors':False,
    'designs':['severity','severity_state','raw_clinical'],'ridge_alphas':[0.1,10.0,1000.0],
    'trees':128,'tree_max_depth':8,'tree_min_leaf':5,'tree_max_features':1.0,'tree_threads':1,'tree_seed':94501,
    'minimum_support':[80,40,40],'minimum_inner_support':10,'bootstrap_resamples':500,'bootstrap_seed':94501,
    'normalization':'squared_error_divided_by_discovery_target_iqr_squared','target_iqr_floor':0.000001,
    'primary':'replication_severity_state_extra_trees_minus_severity_extra_trees','head_selection':'fixed_representation_discovery_three_fold_only'}

class CGMEvaluationError(ValueError): pass
def require(ok):
    if not ok: raise CGMEvaluationError('cgm_evaluation_contract_failed')

def make_designs(clinical,mask,eligible,retina_present,age,states,names,sites,fit):
    n=len(clinical); indices=np.asarray([tuple(names).index(name) for name in SEVERITY_FIELDS])
    require(np.all(indices<48) and not mask[:,48:].any())
    require(len(sites)==n and all(type(site)is str and bool(site) for site in sites))
    categories=tuple(sorted(set(sites[i] for i in fit))); require(bool(categories))
    one_hot=np.asarray([[float(site==category) for category in categories] for site in sites])
    unseen=np.asarray([site not in categories for site in sites],float)
    denominator=eligible[:,:48].sum(1); require(np.all(denominator>0))
    observed_fraction=mask[:,:48].sum(1)/denominator
    severity=np.column_stack((clinical[:,indices],mask[:,indices].astype(float),age,observed_fraction,retina_present.astype(float),one_hot,unseen))
    require(states.shape==(n,192) and np.isfinite(states).all())
    return {'severity':severity,'severity_state':np.column_stack((severity,states)),
            'raw_clinical':np.column_stack((severity,clinical[:,:48],mask[:,:48].astype(float)))}

def evaluate(c0,cm0,eligible,r0,rm,names,ages,outer,ids,membership,sites,y,observed,*,steps=1500,progress=None,trainer=None,infer=None,head_evaluator=None):
    import run_bran_anchor_ablation_v2 as paired
    from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
    from run_bran_frozen_cbc_readout_v1 import _encoder_hash
    c0,cm0,eligible,r0,rm,ages,outer,membership,y,observed=map(np.asarray,(c0,cm0,eligible,r0,rm,ages,outer,membership,y,observed))
    n=len(c0); require(c0.shape==cm0.shape==eligible.shape==(n,59) and r0.shape==(n,384))
    require(all(v.shape==(n,) for v in (rm,ages,outer,membership,y,observed)))
    require(all(v.dtype==np.dtype(bool) for v in (cm0,eligible,rm,membership,observed)) and outer.dtype.kind in 'iu' and set(outer.tolist())==set(range(5)))
    require(len(names)==len(set(names))==59 and set(SEVERITY_FIELDS)<=set(names))
    require(not isinstance(ids,(str,bytes)) and len(ids)==len(set(ids))==n and all(type(i)is str and bool(i) for i in ids))
    require(np.isfinite(ages).all() and y.dtype.kind=='f' and np.isfinite(y).all() and np.all(y[observed]>0))
    local_eligible=eligible.copy(); local_eligible[:,48:]=False
    fit=np.flatnonzero(outer<3); require(len(fit)>=10)
    transform=paired.base.FoldTransform(c0,cm0,local_eligible,r0,rm,ages,fit)
    c,cm,r,age=transform.apply(c0,cm0,local_eligible,r0,rm,ages)
    require(np.isfinite(c).all() and np.isfinite(r).all())
    if progress: progress('training')
    train=paired._train if trainer is None else trainer
    model=train(BRANClinicalAnchorV2,c,cm,r,rm,age,fit,1701,steps=steps); model.eval(); before=_encoder_hash(model)
    states=(_inference if infer is None else infer)(model,c,cm,r,rm,age)
    designs=make_designs(c,cm,local_eligible,rm,age,np.asarray(states),tuple(names),tuple(sites),fit)
    if progress: progress('heads')
    result=(kernel.fit_and_evaluate if head_evaluator is None else head_evaluator)(designs,y,observed,outer,membership,ids=ids)
    kernel.validate_report(result); require(result['status']!='numerical_failure')
    require(_encoder_hash(model)==before)
    report={'encoder_fits':1,'state_dimension':192,'encoder_unchanged_after_heads':True,
            'target_excluded_from_encoder_and_designs':True,'patient_arrays_serialized':False,'default_promotion':False,'analysis':result}
    require(validate_result(report)); return report

def validate_result(value):
    if not isinstance(value,dict) or set(value)!={'encoder_fits','state_dimension','encoder_unchanged_after_heads','target_excluded_from_encoder_and_designs','patient_arrays_serialized','default_promotion','analysis'}: return False
    if type(value['encoder_fits'])is not int or value['encoder_fits']!=1 or type(value['state_dimension'])is not int or value['state_dimension']!=192: return False
    if value['encoder_unchanged_after_heads'] is not True or value['target_excluded_from_encoder_and_designs'] is not True or value['patient_arrays_serialized'] is not False or value['default_promotion'] is not False: return False
    try: kernel.validate_report(value['analysis'])
    except kernel.CGMAddedValueKernelError: return False
    return value['analysis']['status'] in {'ok','unsupported_target_support'}
