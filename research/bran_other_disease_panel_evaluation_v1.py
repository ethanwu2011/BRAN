"""Local-only multi-condition structure panel. No patient arrays are exported."""
import math
import numpy as np
import bran_disease_structure_evaluation_v1 as old
import bran_disease_structure_192_v1 as structure
import bran_disease_structure_diagnostics_v1 as diagnostics

CODES = ('mhterm_dm1','mhterm_predm','mhoccur_mi','mhoccur_strk','mhoccur_clsh',
    'mhoccur_hbp','mhoccur_pdr','mhoccur_pd','mhoccur_ad','mhoccur_cogn','mhoccur_ms',
    'mhoccur_ra','mhoccur_oa','mhoccur_ca','mhoccur_plm','mhoccur_rnl','mhoccur_obs',
    'mhoccur_glc','mhoccur_amd','mhoccur_crt','mhoccur_rvo','mhoccur_ded')
FIELDS = ('age_years','hba1c','glucose','hemoglobin','vit_bmi_vsorres',
    'vit_sysbp_vsorres','vit_diabp_vsorres','manifest_average_cgm_glucose_mg_dl')
SUPPORT = {'discovery_at_least_80':80,'validation_at_least_40':40,'replication_at_least_40':40}
PARAMETERS = {**{k:v for k,v in __import__('bran_disease_structure_experiment_v1').PARAMETERS.items()
    if k not in {'disease','independent_characterization'}},
    'candidate_codes':list(CODES),'selection_scope':'all_22_prespecified_history_positive_cohorts',
    'encoder_fits':'one_shared_discovery_fit_if_any_cohort_supported_else_zero',
    'profile_fields':list(FIELDS),'profile_folds':[3,4], 'minimum_profile_observed_per_group':20,
    'profiles':'count_free_observed_means_not_adjusted_clinical_validation',
    'cgm_used_for_group_selection':False,'multiplicity':'descriptive_panel_no_significance_or_promotion_claims'}

class PanelContractError(ValueError): pass

def require(ok):
    if not ok: raise PanelContractError('other_disease_panel_contract_failed')

def support_flags(member,outer):
    return {key:bool(np.sum(member & rows)>=threshold) for (key,threshold),rows in
        zip(SUPPORT.items(),(outer<3,outer==3,outer==4),strict=True)}

def profiles(locked,states,member,outer,values,masks):
    """Means only; labels and all denominators stay inside this function."""
    require(set(values)==set(masks)==set(FIELDS))
    k=locked._selected_k
    require(type(k)is int and 2<=k<=4)
    result={'status':'descriptive_not_clinical_validation','fields':{}}
    for field in FIELDS:
        result['fields'][field]={}
        for label,fold in (('validation',3),('replication',4)):
            rows=member&(outer==fold); assigned=locked.predict(states[rows])
            require(assigned.shape==(int(rows.sum()),) and assigned.dtype.kind in 'iu' and np.all((assigned>=0)&(assigned<k)))
            y=np.asarray(values[field])[rows]; mask=np.asarray(masks[field])[rows]
            require(mask.dtype==np.dtype(bool) and y.shape==mask.shape and np.isfinite(y[mask]).all())
            groups=[y[mask&(assigned==g)] for g in range(k)]
            if any(len(v)<20 for v in groups): item={'status':'suppressed_support'}
            else: item={'status':'descriptive','means':[float(np.mean(v)) for v in groups]}
            result['fields'][field][label]=item
    return result

def evaluate(c0,cm0,eligible,r0,rm,names,ages,outer,patient_ids,memberships,cgm,cgm_observed,
             *,steps=1500,progress=None,trainer=None,infer=None,structure_fitter=None,diagnostic=None):
    import run_bran_anchor_ablation_v2 as paired
    from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
    from run_bran_frozen_cbc_readout_v1 import _encoder_hash
    trainer=paired._train if trainer is None else trainer
    infer=old._inference if infer is None else infer
    structure_fitter=structure.fit_structure if structure_fitter is None else structure_fitter
    diagnostic=diagnostics.diagnose if diagnostic is None else diagnostic
    c0,cm0,eligible,r0,rm,ages,outer,cgm,cgm_observed=[np.asarray(v) for v in
        (c0,cm0,eligible,r0,rm,ages,outer,cgm,cgm_observed)]
    n=len(c0); ids=tuple(patient_ids); names=tuple(names)
    require(c0.shape==cm0.shape==eligible.shape==(n,59) and r0.shape==(n,384))
    require(all(v.dtype==np.dtype(bool) for v in (cm0,eligible,rm,cgm_observed)))
    require(rm.shape==ages.shape==outer.shape==cgm.shape==cgm_observed.shape==(n,))
    require(outer.dtype.kind in 'iu' and set(outer.tolist())==set(range(5)))
    require(len(ids)==len(set(ids))==n and all(type(i)is str and i for i in ids))
    require(len(names)==len(set(names))==59 and all(type(i)is str and i for i in names))
    require(set(memberships)==set(CODES) and all(isinstance(v,np.ndarray) and v.dtype==np.dtype(bool) and v.shape==(n,) for v in memberships.values()))
    require(np.isfinite(c0[cm0]).all() and np.isfinite(r0[rm]).all() and np.isfinite(ages).all())
    require(np.isfinite(cgm[cgm_observed]).all() and np.all(cgm[cgm_observed]>0))
    require(type(steps)is int and steps>0)
    result={'candidate_codes':list(CODES),'state_dimension':192,'encoder_fits':0,
        'encoder_unchanged_after_diagnostics':True,'patient_arrays_serialized':False,
        'default_promotion':False,'novel_subtype_claimed':False,'clinical_efficacy_claimed':False,
        'candidates':{code:{'support':support_flags(memberships[code],outer),
            'status':'unsupported_cohort_support'} for code in CODES}}
    supported=[code for code in CODES if all(result['candidates'][code]['support'].values())]
    if not supported:
        validate_export(result); return result
    local_eligible=eligible.copy(); local_eligible[:,48:]=False
    fit=np.flatnonzero(outer<3)
    transform=paired.base.FoldTransform(c0,cm0,local_eligible,r0,rm,ages,fit)
    c,cm,r,age=transform.apply(c0,cm0,local_eligible,r0,rm,ages)
    if progress: progress('training')
    model=trainer(BRANClinicalAnchorV2,c,cm,r,rm,age,fit,1701,steps=steps); model.eval()
    before=_encoder_hash(model)
    states=np.asarray(infer(model,c,cm,r,rm,age),dtype=float)
    require(states.shape==(n,192) and np.isfinite(states).all())
    result['encoder_fits']=1
    # Raw profiles never enter the scaler/PCA/GMM or validation-density search.
    values={'age_years':ages,'manifest_average_cgm_glucose_mg_dl':cgm}
    masks={'age_years':np.ones(n,bool),'manifest_average_cgm_glucose_mg_dl':cgm_observed}
    for field in FIELDS[1:-1]:
        require(field in names); j=names.index(field); require(j<48)
        values[field]=c0[:,j]; masks[field]=cm0[:,j]&local_eligible[:,j]
    if progress: progress('structure')
    for code in supported:
        member=memberships[code]; d=states[member&(outer<3)]
        v=states[member&(outer==3)]; rtest=states[member&(outer==4)]
        locked=structure_fitter(d,v)
        stability=diagnostic(d,v,rtest,locked_model=locked)
        sr=old._closed_structure(locked); dr=old._closed_stability(stability)
        old._validate_structure_and_diagnostics(sr,dr)
        pr={'status':'not_applicable_no_supported_groups'}
        if dr['status']==diagnostics.STATUS_SUPPORTED:
            pr=profiles(locked,states,member,outer,values,masks)
        result['candidates'][code].update(status='evaluated',structure=sr,diagnostics=dr,profiles=pr)
    require(_encoder_hash(model)==before)
    validate_export(result); return result

def _validate_profiles(pr,dr):
    if dr['status']!=diagnostics.STATUS_SUPPORTED:
        require(pr=={'status':'not_applicable_no_supported_groups'}); return
    require(isinstance(pr,dict) and set(pr)=={'status','fields'} and pr['status']=='descriptive_not_clinical_validation')
    require(isinstance(pr['fields'],dict) and set(pr['fields'])==set(FIELDS))
    k=dr['selected_k']
    for field in FIELDS:
        sides=pr['fields'][field]; require(isinstance(sides,dict) and set(sides)=={'validation','replication'})
        for item in sides.values():
            require(isinstance(item,dict))
            if item.get('status')=='suppressed_support': require(set(item)=={'status'})
            else:
                require(set(item)=={'status','means'} and item['status']=='descriptive')
                require(type(item['means'])is list and len(item['means'])==k and
                    all(type(x)is float and math.isfinite(x) for x in item['means']))

def validate_export(r):
    require(isinstance(r,dict) and set(r)=={'candidate_codes','state_dimension','encoder_fits',
        'encoder_unchanged_after_diagnostics','patient_arrays_serialized','default_promotion',
        'novel_subtype_claimed','clinical_efficacy_claimed','candidates'})
    require(type(r['candidate_codes'])is list and r['candidate_codes']==list(CODES))
    require(type(r['state_dimension'])is int and r['state_dimension']==192 and type(r['encoder_fits'])is int and r['encoder_fits'] in (0,1))
    require(r['encoder_unchanged_after_diagnostics'] is True and all(r[k] is False for k in
        ('patient_arrays_serialized','default_promotion','novel_subtype_claimed','clinical_efficacy_claimed')))
    require(isinstance(r['candidates'],dict) and set(r['candidates'])==set(CODES))
    any_supported=False
    for item in r['candidates'].values():
        require(isinstance(item,dict) and {'support','status'}<=set(item))
        support=item['support']; require(isinstance(support,dict) and set(support)==set(SUPPORT) and all(type(v)is bool for v in support.values()))
        if not all(support.values()):
            require(set(item)=={'support','status'} and item['status']=='unsupported_cohort_support'); continue
        any_supported=True
        require(set(item)=={'support','status','structure','diagnostics','profiles'} and item['status']=='evaluated')
        old._validate_structure_and_diagnostics(item['structure'],item['diagnostics'])
        _validate_profiles(item['profiles'],item['diagnostics'])
    require(r['encoder_fits']==int(any_supported) and not old._contains_array(r))

def validate_result(r):
    try: validate_export(r)
    except (PanelContractError,old.DiseaseStructureEvaluationError,TypeError,ValueError): return False
    return True
