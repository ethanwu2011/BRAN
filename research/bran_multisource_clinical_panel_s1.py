"""Source-free pooled clinical analysis; private state, safe aggregate output."""
from dataclasses import dataclass
import hashlib
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score, balanced_accuracy_score
from threadpoolctl import threadpool_limits
import bran_multisource_clinical_design_s1 as design
import bran_r1_hf_structure_kernel_v1 as structure
import bran_mimic_clinical_outcomes_v3 as outcome
import bran_mimic_clinical_panel_v3 as previous

ERROR='multisource_clinical_panel_s1_failed'
FAMILIES=('recorded_diabetes_definition_mismatch','recorded_heart_failure','recorded_ckd')
SOURCES=('mimic_iv_hospital_day1','eicu_icu_day1')
FLAGS={'patient_level_output_emitted':False,'encoder_fitted':False,
       'novel_subtype_claim':False,'clinical_utility_established':False,'external_validation_established':False}


def require(ok):
    if not ok:raise ValueError(ERROR) from None


def check_frame(frame):
    n=len(frame['state'])
    spec={'state':((n,192),np.float32),'values':((n,21),np.float64),'observed':((n,21),bool),
          'age_value':((n,),np.float64),'age_lower':((n,),np.float64),'age_upper':((n,),np.float64),
          'age_kind':((n,),np.int64),'source':((n,),np.uint8),'roles':((n,),np.uint8),
          'membership':((n,3),bool),'outcome':((n,),np.int8),'person_group':((n,),np.int64)}
    require(set(frame)==set(spec))
    for key,(shape,dtype) in spec.items():
        v=frame[key];require(type(v) is np.ndarray and v.shape==shape and v.dtype==np.dtype(dtype))
    require(n>=1 and np.isfinite(frame['state']).all() and frame['observed'].any(1).all()
        and np.isfinite(frame['values'][frame['observed']]).all()
        and np.isin(frame['source'],(0,1)).all() and np.isin(frame['roles'],(0,1,2)).all()
        and np.isin(frame['outcome'],(-1,0,1)).all() and np.unique(frame['person_group']).size==n
        and (frame['person_group']>=0).all())


def build(frame,rows):
    return design.build(**{k:frame[k][rows] for k in
        ('values','observed','age_value','age_lower','age_upper','age_kind','state','roles','source')})


def support(frame,rows):
    # Whole source/role table suppression; never expose one small complement.
    cells=[[int(np.sum((frame['source'][rows]==s)&(frame['roles'][rows]==r))) for r in range(3)] for s in range(2)]
    if min(v for row in cells for v in row)<20:return {'status':'suppressed_source_role_support'}
    return {'status':'released','source_role_people_lower_bounds_20':[[v//20*20 for v in row] for row in cells]}


def nuisance_check(labels,k,x,roles,source):
    test=roles==2;train=roles==0
    if any(np.bincount(labels[(source==s)&test],minlength=k).min()<20 for s in range(2)):
        return {'status':'suppressed_source_group_support'}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        model=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000).fit(x[train],labels[train])
    if any(issubclass(w.category,ConvergenceWarning) for w in caught):return {'status':'nuisance_readout_not_converged'}
    return {'status':'released',
        'test_group_source_adjusted_mutual_information':float(adjusted_mutual_info_score(source[test],labels[test])),
        'test_nuisance_group_balanced_accuracy':float(balanced_accuracy_score(labels[test],model.predict(x[test]))),
        'nuisance_predictors':'missingness_typed_age_source_only','used_for_group_selection':False}


def validate_nuisance(a):
    closed=('suppressed_source_group_support','nuisance_readout_not_converged','not_run_structure_gate_failed')
    if a.get('status') in closed:require(set(a)=={'status'});return
    require(set(a)=={'status','test_group_source_adjusted_mutual_information','test_nuisance_group_balanced_accuracy',
        'nuisance_predictors','used_for_group_selection'} and a['status']=='released'
        and a['nuisance_predictors']=='missingness_typed_age_source_only' and a['used_for_group_selection'] is False)
    for key,lo in (('test_group_source_adjusted_mutual_information',-1),('test_nuisance_group_balanced_accuracy',0)):
        require(type(a[key]) is float and np.isfinite(a[key]) and lo<=a[key]<=1)


@dataclass(repr=False)
class PrivateResult:
    aggregate:dict
    objects:dict
    def __repr__(self):return '<PrivateMultisourceClinicalResult>'
    def __reduce__(self):raise TypeError(ERROR)


def fit_family(frame,family,progress=None):
    try:
        with threadpool_limits(limits=2):return _fit_family(frame,family,progress)
    except (MemoryError,KeyboardInterrupt,SystemExit):raise
    except Exception:raise ValueError(ERROR) from None


def _fit_family(frame,family,progress):
    check_frame(frame);require(family in FAMILIES)
    rows=np.flatnonzero(frame['membership'][:,FAMILIES.index(family)])
    roles=frame['roles'][rows];source=frame['source'][rows]
    base={'schema':'bran-multisource-clinical-family-s1','family':family,'support':support(frame,rows),
        'source_names':list(SOURCES),'outcomes_used_for_group_fitting':False,**FLAGS}
    if any(np.sum(roles==r)<m for r,m in enumerate((80,40,40))):
        return PrivateResult({**base,'status':'unsupported_cohort_roles'}, {})
    x=build(frame,rows);groups={};fits={};branches={}
    for branch,states in (('bran',frame['state'][rows]),('raw',x.raw_padded192)):
        if progress:progress(branch+'_structure')
        fit=structure.fit_structure(*(states[roles==r] for r in range(3)))
        report=fit.aggregate;require(structure.validate_aggregate(report))
        active=fit.status=='supported' and report.get('stability_gate') is True
        fits[branch]=fit;groups[branch]=fit.predict(states) if active else None
        branch_report={'structure':report,'group_arms_admitted':active}
        if active:
            branch_report['source_test_profiles']={SOURCES[s]:previous.profiles(
                frame['values'][rows][(roles==2)&(source==s)],frame['observed'][rows][(roles==2)&(source==s)],
                groups[branch][(roles==2)&(source==s)],frame['outcome'][rows][(roles==2)&(source==s)],fit.selected_k)
                for s in range(2)}
            branch_report['nuisance']=nuisance_check(groups[branch],fit.selected_k,x.nuisance_design,roles,source)
        else:
            branch_report['source_test_profiles']={s:{'status':'not_run_structure_gate_failed'} for s in SOURCES}
            branch_report['nuisance']={'status':'not_run_structure_gate_failed'}
        branches[branch]=branch_report
    if progress:progress('outcome_utility')
    sink={};binding=hashlib.sha256(('S1|'+family+'|raw49_source2_pad8_state192').encode()).hexdigest()
    utility=outcome.evaluate(x.context59,x.state_scaled,frame['person_group'][rows],roles,frame['outcome'][rows],
        bran_groups=groups['bran'],bran_k=fits['bran'].selected_k if groups['bran'] is not None else None,
        raw_groups=groups['raw'],raw_k=fits['raw'].selected_k if groups['raw'] is not None else None,
        private_sink=sink,design_binding=binding)
    keys=('bran_groups_minus_context','bran_groups_minus_raw_groups','state_bran_groups_minus_state')
    incremental=branches['bran']['group_arms_admitted'] and all(
        utility.get('contrasts',{}).get(k,{}).get('utility_gate') is True for k in keys)
    aggregate={**base,'status':'evaluated','branches':branches,'outcome_utility':utility,
        'three_adjusted_increment_checks_passed':bool(incremental),
        'interpretation':'internal_two_source_candidate_heterogeneity_not_validated_subtypes'}
    validate_report(aggregate)
    return PrivateResult(aggregate,{'rows':rows,'fits':fits,'outcome':sink,'binding':binding,'design_stats':x.stats})


def validate_report(a):
    base={'schema','family','support','source_names','outcomes_used_for_group_fitting','status',*FLAGS}
    require(type(a) is dict and a.get('schema')=='bran-multisource-clinical-family-s1'
        and a.get('family') in FAMILIES and a.get('source_names')==list(SOURCES)
        and a.get('outcomes_used_for_group_fitting') is False and all(a.get(k) is v for k,v in FLAGS.items()))
    support_a=a['support'];require(type(support_a) is dict)
    if support_a.get('status')=='suppressed_source_role_support':require(set(support_a)=={'status'})
    else:
        require(set(support_a)=={'status','source_role_people_lower_bounds_20'} and support_a['status']=='released')
        v=support_a['source_role_people_lower_bounds_20'];require(type(v) is list and len(v)==2)
        require(all(type(row) is list and len(row)==3 and all(type(x) is int and x>=20 and x%20==0 for x in row) for row in v))
    if a['status']=='unsupported_cohort_roles':require(set(a)==base);return
    require(a['status']=='evaluated' and set(a)==base|{'branches','outcome_utility',
        'three_adjusted_increment_checks_passed','interpretation'}
        and a['interpretation']=='internal_two_source_candidate_heterogeneity_not_validated_subtypes')
    require(set(a['branches'])=={'bran','raw'})
    for b in a['branches'].values():
        require(set(b)=={'structure','group_arms_admitted','source_test_profiles','nuisance'}
            and structure.validate_aggregate(b['structure']))
        active=b['structure']['status']=='supported' and b['structure'].get('stability_gate') is True
        require(b['group_arms_admitted'] is active and set(b['source_test_profiles'])==set(SOURCES))
        for profile in b['source_test_profiles'].values():
            if active:previous.validate_profiles(profile,b['structure']['selected_k'])
            else:require(profile=={'status':'not_run_structure_gate_failed'})
        validate_nuisance(b['nuisance'])
        if not active:require(b['nuisance']=={'status':'not_run_structure_gate_failed'})
    require(outcome.validate_report(a['outcome_utility']))
    if 'arms' in a['outcome_utility']:
        for branch,names in (('bran',('context_bran_groups','context_state_bran_groups')),
                             ('raw',('context_raw_groups',))):
            for name in names:
                require((a['outcome_utility']['arms'][name]['status']=='available') is
                        a['branches'][branch]['group_arms_admitted'])
    keys=('bran_groups_minus_context','bran_groups_minus_raw_groups','state_bran_groups_minus_state')
    expected=a['branches']['bran']['group_arms_admitted'] and all(
        a['outcome_utility'].get('contrasts',{}).get(k,{}).get('utility_gate') is True for k in keys)
    require(a['three_adjusted_increment_checks_passed'] is bool(expected))


def replay(frame,objects):
    rows=objects['rows'];x=build(frame,rows);require(set(x.stats)==set(objects['design_stats']))
    require(all(np.array_equal(np.asarray(x.stats[k]),np.asarray(v),equal_nan=True) for k,v in objects['design_stats'].items()))
    result={};labels={}
    for branch,states in (('bran',frame['state'][rows]),('raw',x.raw_padded192)):
        fit=objects['fits'][branch]
        if fit._mixture is not None:
            labels[branch]=fit._mixture.predict(fit._pca.transform(fit._scaler.transform(states)))
            result[branch]=labels[branch]
    sink=objects['outcome'];require(sink.get('design_binding')==objects['binding'])
    if sink['arms']:
        test=(frame['roles'][rows]==2)&(frame['outcome'][rows]>=0)
        designs={'context':x.context59[test],'context_state':np.column_stack((x.context59[test],x.state_scaled[test]))}
        for branch,names in (('bran',('context_bran_groups','context_state_bran_groups')),('raw',('context_raw_groups',))):
            if names[0] not in sink['arms']:continue
            one=outcome._one_hot(labels[branch][test],objects['fits'][branch].selected_k)
            designs[names[0]]=np.column_stack((x.context59[test],one))
            if branch=='bran':designs[names[1]]=np.column_stack((x.context59[test],x.state_scaled[test],one))
        require(set(designs)==set(sink['arms']))
        for arm,saved in sink['arms'].items():
            require(designs[arm].shape[1]==saved['feature_width'])
            probability=outcome._calibrated_probabilities(saved['model'].decision_function(designs[arm]),saved['calibration_offset'])
            require(np.array_equal(probability,saved['test_predictions']))
            result['outcome_'+arm]=probability
    return result
