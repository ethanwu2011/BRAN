"""Private three-role inference with replay; aggregate-only V6 evaluation.

No file I/O. Caller owns exact source/checkpoint provenance and the quiet lock.
"""
import numpy as np
import torch

from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_multisource_batches_v2 import tensor
from bran_multisource_outcomes_v2 import original_age, subset_age, AGE_SCENARIOS, age_scenario
from bran_multisource_inference_v2 import route_predictions, completion_predictions, infer_native
from bran_multisource_profiles_v3 import _validate_provider, _unchanged
from bran_multisource_protocol_v2 import PARAMETERS
from bran_external_cbc_evaluation_v1 import paired_counts
import bran_missingness_stress_v1 as masking
import bran_multisource_outcome_metrics_v2 as metrics
from bran_missingness_stress_metrics_v1 import safe_coverage
import bran_source_pattern_metrics_v6 as gate

ERROR = 'source_pattern_evaluation_v6_invalid'


def require(ok):
    if not ok: raise ValueError(ERROR) from None


def _infer_fold(model, transform, paired, rows, slots):
    c,cm = transform.clinical(paired.c,paired.cm)
    r,rm = transform.retinal(paired.r,paired.rm)
    age=subset_age(original_age(paired),rows)
    args=(tensor(c[rows]),tensor(cm[rows],torch.bool),tensor(r[rows]),tensor(rm[rows],torch.bool),
          age,transform.age_mean,transform.age_scale)
    routes={k:v.screening_probability.numpy() for k,v in route_predictions(model,*args).items()}
    completion, intended = {}, {}
    for pattern in PARAMETERS['completion_patterns']:
        result=completion_predictions(model,*args,pattern,slots)
        completion[pattern]=(result.cbc_standardized.numpy()*transform.clinical_iqr[list(slots)]
                             +transform.clinical_median[list(slots)])
        intended[pattern]=result.targetmask.numpy()
        require(np.array_equal(result.scoring_target_mask.numpy(),
                               intended[pattern]&np.isfinite(completion[pattern])))
    stress={}
    for pattern in masking.PATTERNS:
        hidden=masking.remove_inputs(c,cm,r,rm,slots,pattern)
        masking.assert_no_input_leak(hidden,cm,rm,slots,pattern)
        result=infer_native(model,tensor(hidden.clinical[rows]),tensor(hidden.clinical_mask[rows],torch.bool),
            tensor(hidden.retinal[rows]),tensor(hidden.retinal_mask[rows],torch.bool),age,
            transform.age_mean,transform.age_scale)
        require(np.array_equal(~result.abstained.numpy(),hidden.available[rows]))
        stress[pattern]=result.screening_probability.numpy()
    ages={}
    for scenario in AGE_SCENARIOS:
        empty=scenario=='all_physiology_hidden'
        result=infer_native(model,args[0],torch.zeros_like(args[1]) if empty else args[1],args[2],
            torch.zeros_like(args[3]) if empty else args[3],age_scenario(age,scenario),
            transform.age_mean,transform.age_scale)
        require(not empty or bool(result.abstained.all()))
        ages[scenario]=result.screening_probability.numpy()
    return {'routes':routes,'completion':completion,'intended':intended,'stress':stress,'ages':ages}


def _equal(a,b):
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(_equal(a[k],b[k]) for k in a)
    return np.array_equal(a,b,equal_nan=True)


def low_hb_cells(paired,completion,slots,counts):
    """Historical training-only deciles, with all three tails support protected."""
    from run_bran_modality_union_v1 import tail_masks
    j=CBC_FIELDS.index('hemoglobin'); target=paired.c[:,slots[j]]
    observed=paired.cm[:,slots[j]]; folds=paired.folds
    result={}
    for pattern in ('whole_cbc_hidden','whole_cbc_no_retina'):
        common=observed & np.logical_and.reduce([np.isfinite(completion[pattern][r][:,j]) for r in gate.ROLES])
        tails={k:np.zeros(len(folds),bool) for k in ('low','middle','high')}
        for fold in range(5):
            training=(folds!=fold)&observed
            # Context support, determined solely by availability, matches the
            # whole-CBC task before computing its training-only cutpoints.
            transform=paired.transforms[fold]
            c,cm=transform.clinical(paired.c,paired.cm); r,rm=transform.retinal(paired.r,paired.rm)
            hidden=masking.remove_inputs(c,cm,r,rm,slots,pattern)
            training &= hidden.available
            require(training.sum()>=20)
            rows=np.flatnonzero(folds==fold)
            for key,mask in tail_masks(target[training],target[rows]).items():tails[key][rows]=mask
        if any(np.count_nonzero(common&mask)<20 for mask in tails.values()):
            result[pattern]={'status':'unsupported'}; continue
        low=common&tails['low']; den=counts[:,low].sum(1)
        error={role:np.abs(completion[pattern][role][low,j]-target[low]) for role in gate.ROLES}
        result[pattern]=metrics._cell({r:float(e.mean()) for r,e in error.items()},
            {r:np.divide(counts[:,low]@e,den,out=np.full(1000,np.nan),where=den>0) for r,e in error.items()},
            gate.CONTRASTS,'mae')
    return result


def evaluate(paired,provider,progress=None):
    """Return only safe aggregate cells. Arrays never leave this local function."""
    try:
        n=len(paired.folds); folds=paired.folds; names=tuple(paired.endpoint_names)
        require(n>=100 and set(np.unique(folds))==set(range(5)) and len(names)==26)
        slots=tuple(paired.names.index(f) for f in CBC_FIELDS)
        routes={p:{r:np.full((n,26),np.nan) for r in gate.ROLES} for p in ('both','clinical','retinal')}
        stress={p:{r:np.full((n,26),np.nan) for r in gate.ROLES} for p in masking.PATTERNS}
        ages={p:{r:np.full((n,26),np.nan) for r in gate.ROLES} for p in AGE_SCENARIOS}
        completion={p:{r:np.full((n,9),np.nan) for r in gate.ROLES} for p in PARAMETERS['completion_patterns']}
        intended={p:np.zeros((n,9),bool) for p in completion}
        for fold in range(5):
            rows=np.flatnonzero(folds==fold)
            for role in gate.ROLES:
                if progress:progress('inference',fold,role)
                model,transform=provider(role,fold)
                before,pin,grad=_validate_provider(model,transform,fold,slots,paired.transforms[fold])
                value=_infer_fold(model,transform,paired,rows,slots)
                require(_unchanged(before,pin,grad,model,transform))
                reloaded,rt=provider(role,fold)
                require(reloaded is not model)
                rb,rpin,rgrad=_validate_provider(reloaded,rt,fold,slots,paired.transforms[fold])
                require(_equal(value,_infer_fold(reloaded,rt,paired,rows,slots)))
                require(_unchanged(rb,rpin,rgrad,reloaded,rt))
                for key,container in (('routes',routes),('stress',stress),('ages',ages),('completion',completion)):
                    for pattern in container:container[pattern][role][rows]=value[key][pattern]
                for pattern in intended:
                    if role=='V5':intended[pattern][rows]=value['intended'][pattern]
                    else:require(np.array_equal(intended[pattern][rows],value['intended'][pattern]))
                del model,reloaded,value
        if progress:progress('aggregate_bootstrap',None,None)
        counts=paired_counts(folds,draws=1000,seed=98551)
        screen={route:metrics.screening(values,paired.labels,paired.labelmask,folds,names,counts,gate.CONTRASTS)
                for route,values in routes.items()}
        stress_primary=gate.stress_summary({p:stress[p] for p in gate.PATTERNS},
            paired.labels,paired.labelmask,folds,names,counts)
        stress_profiles={p:metrics.screening(v,paired.labels,paired.labelmask,folds,names,counts,gate.CONTRASTS)
                         for p,v in stress.items()}
        age_profiles={p:metrics.screening(v,paired.labels,paired.labelmask,folds,names,counts,gate.CONTRASTS)
                      for p,v in ages.items() if p!='all_physiology_hidden'}
        blood={}
        target=paired.c[:,slots].copy()
        for p,values in completion.items():
            observed=intended[p]; truth=np.where(observed,target,np.nan)
            predictions={r:np.where(observed,v,np.nan) for r,v in values.items()}
            blood[p]=metrics.completion(predictions,truth,observed,folds,counts,contrasts=gate.CONTRASTS)
        low=low_hb_cells(paired,completion,slots,counts)
        checks=gate.protected_completion_checks(blood,low)
        decision=gate.decide(stress_primary,screen['both'],checks,False)
        result={'schema':'bran-source-pattern-v6-native-evaluation','screening':screen,
            'stress_primary':stress_primary,'stress_profiles':stress_profiles,'age_profiles':age_profiles,
            'completion':blood,'historical_low_hb':low,'decision_pending_authentication':decision,
            'all_inference_replayed_exactly':True,'all_empty_physiology_abstained':True,
            'patient_level_output_emitted':False,'candidate_promoted':False}
        validate_result(result)
        return result
    except Exception:
        raise ValueError(ERROR) from None


def validate_result(result):
    import json
    require(type(result) is dict and set(result)=={'schema','screening','stress_primary','stress_profiles',
        'age_profiles','completion','historical_low_hb','decision_pending_authentication',
        'all_inference_replayed_exactly','all_empty_physiology_abstained','patient_level_output_emitted','candidate_promoted'})
    require(result['schema']=='bran-source-pattern-v6-native-evaluation'
        and result['patient_level_output_emitted'] is False and result['candidate_promoted'] is False
        and result['all_inference_replayed_exactly'] is True and result['all_empty_physiology_abstained'] is True)
    require(set(result['completion'])==set(PARAMETERS['completion_patterns']))
    require(set(result['screening'])=={'both','clinical','retinal'})
    require(set(result['stress_profiles'])==set(masking.PATTERNS))
    require(set(result['age_profiles'])==set(AGE_SCENARIOS)-{'all_physiology_hidden'})
    json.dumps(result,allow_nan=False)
