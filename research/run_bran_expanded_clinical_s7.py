"""Quiet S7 membership qualification and eight-family fixed R7 clinical study."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import run_bran_r7_clinical_s4 as parent
import bran_expanded_membership_s7 as membership
import bran_expanded_clinical_panel_s7 as panel
from bran_clinical_source_reader_v1 import iter_projected_csv
from diagnose_bran_agefree_reference_failure_v1 import safe_trace

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'BRAN_EXPANDED_CLINICAL_S7_ATTEMPT1'
PRIVATE=ROOT/'private_artifacts'/'bran_expanded_clinical_s7_attempt1'
ERROR='expanded_clinical_s7_failed'
sha=parent.sha
read=parent.old.read
write_json=parent.write_json
FAMILIES=membership.FAMILIES
CODE=('BRAN_EXPANDED_CLINICAL_S7_DESIGN.md','bran_expanded_membership_s7.py',
    'test_bran_expanded_membership_s7.py','bran_expanded_clinical_panel_s7.py',
    'test_bran_expanded_clinical_panel_s7.py','run_bran_expanded_clinical_s7.py',
    'test_run_bran_expanded_clinical_s7.py','run_bran_expanded_clinical_s7_attempt1.sh')
FLAGS={'patient_level_output_emitted':False,'encoder_fitted':False,'historical_gates_changed':False,
    'novel_subtype_claim':False,'independent_external_validation':False,'automatic_promotion':False,
    'beyond_bmi_obesity_claim':False,'outcomes_used_for_group_selection':False}

def require(ok):
    if not ok:raise ValueError(ERROR) from None

def source_paths():
    m=parent.old.mimic.admission;e=parent.old.eicu.admission
    return {'mimic_diagnoses':m.SOURCE_INPUTS['diagnoses'],
        'dictionary':m.SOURCE_INPUTS['disease_dictionary'],
        'eicu_patient':e.INPUTS['patient'],'eicu_diagnosis':e.INPUTS['diagnosis']}

def dictionary():
    return membership.bind_dictionary(iter_projected_csv(source_paths()['dictionary'],
        ('icd_version','icd_code','long_title'),max_rows=500000))

def code_mapping_digest(mapping):
    import hashlib
    return hashlib.sha256(json.dumps([[v,c,f] for (v,c),f in sorted(mapping.items())],
        separators=(',',':')).encode()).hexdigest()

def bindings():
    a,t=parent.authenticate();p=read(parent.OUT/'protocol.json')
    require(p['frame']=='R7_attempt1_fold0' and tuple(panel.FAMILIES)==FAMILIES)
    m=parent.old.mimic.admission;e=parent.old.eicu.admission
    mp=read(m.OUT/'protocol.json');ep=read(e.OUT/'protocol.json')
    pins={'mimic_diagnoses':mp['dependency_binding']['source_sha256']['diagnoses'],
        'dictionary':mp['dependency_binding']['source_sha256']['disease_dictionary'],
        'eicu_patient':ep['source_sha256']['patient'],'eicu_diagnosis':ep['source_sha256']['diagnosis']}
    require(all(sha(path)==pins[k] for k,path in source_paths().items()))
    mapping=dictionary()
    return {'s4_terminal':t,'s4_terminal_sha256':sha(parent.OUT/'completed.json'),
        's4_private_state_sha256':a['private_state_sha256'],'sources':p['binding']['sources'],
        'r7_checkpoint':p['binding']['R7'],'source_sha256':pins,
        'vocabulary_sha256':code_mapping_digest(mapping),
        'code_sha256':{**p['binding']['code_sha256'],**{n:sha(ROOT/n) for n in CODE}}}

def protocol(b):
    import sys,scipy,sklearn
    return {'schema':'bran-expanded-clinical-s7-protocol','status':'frozen_before_new_membership_and_fitting',
        'binding':b,'families':list(FAMILIES),'frame':'R7_attempt1_fold0','multiplicity_comparisons':40,
        'source_order':['mimic','eicu'],'new_eicu_membership':'dictionary_bound_icd9_only_within_icu_stay',
        'runtime':{'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,
            'sklearn':sklearn.__version__,'joblib':joblib.__version__},**FLAGS}

def load_frame(b):
    f=parent.old.load_frame(b['sources']);cohorts={}
    for name,runner in (('mimic',parent.old.mimic),('eicu',parent.old.eicu)):
        cohorts[name]=runner.load_cohort(b['sources'][name]['source'])
    n_m=len(cohorts['mimic']['person']);n_e=len(cohorts['eicu']['person'])
    require(len(f['state'])==n_m+n_e and (f['source'][:n_m]==0).all() and (f['source'][n_m:]==1).all()
        and np.array_equal(f['membership'],np.vstack([cohorts['mimic']['membership'],cohorts['eicu']['membership']])))
    path=parent.PRIVATE/'state.npz';parent.old.regular(path,0o600)
    require(sha(path)==b['s4_private_state_sha256'])
    with np.load(path,allow_pickle=False) as h:
        require(h.files==['state']);state=h['state']
    require(sha(path)==b['s4_private_state_sha256']);f=parent.replace_state(f,state)
    return f,cohorts

def selected_eicu_episodes(cohort):
    e=parent.old.eicu.admission;wanted=set(cohort['episode']);result={}
    for row in iter_projected_csv(e.INPUTS['patient'],e.bridge.PATIENT_COLUMNS,max_rows=e.LIMITS['patient']):
        key=row['patientunitstayid']
        if key not in wanted:continue
        require(key not in result);result[key]=e.bridge.parse_patient(row)
    require(set(result)==wanted)
    return result

def diagnosis_rows(source):
    if source=='mimic':
        return iter_projected_csv(source_paths()['mimic_diagnoses'],
            ('subject_id','hadm_id','icd_version','icd_code'),max_rows=20_000_000)
    require(source=='eicu')
    return iter_projected_csv(source_paths()['eicu_diagnosis'],
        ('patientunitstayid','diagnosisoffset','icd9code'),max_rows=10_000_000)

def join_membership(cohorts,b):
    require(all(sha(p)==b['source_sha256'][k] for k,p in source_paths().items()))
    mapping=dictionary();require(code_mapping_digest(mapping)==b['vocabulary_sha256'])
    old_map=parent.old.mimic.admission.disease_dictionary.authenticate_audit()
    old_families=tuple(parent.old.mimic.admission.FAMILIES)
    episodes=selected_eicu_episodes(cohorts['eicu'])
    prior_m=membership.mimic_membership(cohorts['mimic'],diagnosis_rows('mimic'),old_map,old_families)
    prior_e=membership.eicu_membership(cohorts['eicu'],episodes,diagnosis_rows('eicu'),old_map,old_families,icd9_only=False)
    require(np.array_equal(prior_m,cohorts['mimic']['membership'])
        and np.array_equal(prior_e,cohorts['eicu']['membership']))
    new_m=membership.mimic_membership(cohorts['mimic'],diagnosis_rows('mimic'),mapping)
    new_e=membership.eicu_membership(cohorts['eicu'],episodes,diagnosis_rows('eicu'),mapping)
    require(all(sha(p)==b['source_sha256'][k] for k,p in source_paths().items()))
    return np.vstack([new_m,new_e])

def attach(frame,new):
    require(new.shape==(len(frame['state']),len(FAMILIES)) and new.dtype==bool)
    result={**frame,'membership':new};panel.check_frame(result)
    require(all(result[k] is frame[k] for k in frame if k!='membership'))
    return result

def phase(state,name,family=None):
    require(name in ('authentication','source_assembly','membership_join','membership_replay',
        'bran_structure','raw_structure','outcome_utility','private_write','object_replay',
        'post_authentication','completed') and family in (None,*FAMILIES))
    state['phase']=name
    write_json(OUT/'progress.next.json',{'phase':name,'family':family,'pid':os.getpid(),
        'patient_level_output_emitted':False})
    os.replace(OUT/'progress.next.json',OUT/'progress.json')

def support_report(frame):
    return {'schema':'bran-expanded-clinical-s7-support','families':membership.support(
        frame['membership'],frame['roles'],frame['source']),
        'source_local_pool_people_lower_bounds_20':[int(np.sum(frame['source']==s))//20*20 for s in range(2)],
        'nonadditive_source_local_counts':True,'full_admitted_pools_retained':True,**FLAGS}

def validate_support(a):
    require(type(a) is dict and set(a)=={'schema','families','source_local_pool_people_lower_bounds_20',
        'nonadditive_source_local_counts','full_admitted_pools_retained',*FLAGS}
        and a['schema']=='bran-expanded-clinical-s7-support' and set(a['families'])==set(FAMILIES)
        and a['nonadditive_source_local_counts'] is True and a['full_admitted_pools_retained'] is True
        and all(a[k] is False for k in FLAGS))
    counts=a['source_local_pool_people_lower_bounds_20']
    require(type(counts) is list and len(counts)==2 and all(type(v) is int and v>=20 and v%20==0 for v in counts))
    for value in a['families'].values():
        require(type(value) is dict and set(value)=={'status','source_role_lower_bounds_20'})
        if value['status']=='suppressed':require(value['source_role_lower_bounds_20'] is None)
        else:
            require(value['status']=='released');cells=value['source_role_lower_bounds_20']
            require(type(cells) is list and len(cells)==2 and all(type(row) is list and len(row)==3
                and all(type(v) is int and v>=20 and v%20==0 for v in row) for row in cells))

def validate_component(c,family):
    require(set(c)=={'result','objects_sha256','object_replay_exact','patient_level_output_emitted'}
        and c['result']['family']==family and c['object_replay_exact'] is True
        and c['patient_level_output_emitted'] is False
        and parent.old.mimic.admission._valid_hex(c['objects_sha256']))
    panel.validate_report(c['result'])

def replay(frame,objects):
    return panel.replay(frame,objects) if objects else {}

def verify_objects(frame,aggregate,objects):
    panel.validate_report(aggregate)
    family=aggregate['family']
    if aggregate['status']=='unsupported_cohort_roles':
        require(objects=={})
        roles=frame['roles'][frame['membership'][:,FAMILIES.index(family)]]
        require(any(np.sum(roles==r)<minimum for r,minimum in enumerate((80,40,40))))
        return {}
    require(objects and objects['family']==family and objects['frame']==aggregate['frame']
        and objects['legacy_recipe']==aggregate['legacy_recipe']
        and objects['family40_uncertainty']==aggregate['family40_uncertainty'])
    for branch in ('bran','raw'):
        require(objects['fits'][branch].aggregate==aggregate['branches'][branch]['structure'])
    return replay(frame,objects)

def authenticate(pending=None):
    names={'protocol.json','aggregate.json','support.json','progress.json',*(f+'.json' for f in FAMILIES)}
    if pending is None:names.add('completed.json')
    require(OUT.is_dir() and not OUT.is_symlink() and {x.name for x in OUT.iterdir()}==names)
    p=read(OUT/'protocol.json');require(p==protocol(bindings()))
    require(PRIVATE.is_dir() and not PRIVATE.is_symlink() and PRIVATE.stat().st_mode&0o777==0o700
        and {x.name for x in PRIVATE.iterdir()}=={'membership.npz',*(f+'.joblib' for f in FAMILIES)})
    a=read(OUT/'aggregate.json');path=PRIVATE/'membership.npz';parent.old.regular(path,0o600)
    require(sha(path)==a['private_membership_sha256'])
    with np.load(path,allow_pickle=False) as h:
        require(h.files==['membership']);joined=h['membership']
    f,cohorts=load_frame(p['binding']);require(np.array_equal(joined,join_membership(cohorts,p['binding'])))
    f=attach(f,joined);support=read(OUT/'support.json');validate_support(support);require(support==support_report(f))
    pins={}
    for family in FAMILIES:
        c=read(OUT/(family+'.json'));validate_component(c,family)
        path=PRIVATE/(family+'.joblib');parent.old.regular(path,0o600);require(sha(path)==c['objects_sha256'])
        obj=joblib.load(path)
        require(set(obj)=={'family','frame','protocol_sha256','aggregate','objects'} and obj['family']==family
            and obj['frame']=='R7_attempt1_fold0' and obj['protocol_sha256']==sha(OUT/'protocol.json')
            and obj['aggregate']==c['result'])
        verify_objects(f,c['result'],obj['objects']);require(sha(path)==c['objects_sha256'])
        pins[family]=sha(OUT/(family+'.json'))
    require(a=={'schema':'bran-expanded-clinical-s7','status':'completed','components_sha256':pins,
        'private_membership_sha256':sha(PRIVATE/'membership.npz'),'support_sha256':sha(OUT/'support.json'),
        'membership_replay_exact':True,'original_three_family_join_replay_exact':True,**FLAGS})
    q=read(OUT/'progress.json');require(set(q)=={'phase','family','pid','patient_level_output_emitted'}
        and q['phase']=='completed' and q['family'] is None and type(q['pid']) is int and q['pid']>0
        and q['patient_level_output_emitted'] is False)
    t=pending if pending is not None else read(OUT/'completed.json')
    require(t=={'status':'authenticated_completed','protocol_sha256':sha(OUT/'protocol.json'),
        'aggregate_sha256':sha(OUT/'aggregate.json'),'patient_level_output_emitted':False})
    return a,t

def run(ownership):
    require(not OUT.exists() and not OUT.is_symlink() and not PRIVATE.exists() and not PRIVATE.is_symlink())
    OUT.mkdir(mode=0o700);ownership['owned']=True;phase(ownership,'authentication')
    b=bindings();p=protocol(b);write_json(OUT/'protocol.json',p)
    phase(ownership,'source_assembly');f,cohorts=load_frame(b)
    phase(ownership,'membership_join');joined=join_membership(cohorts,b)
    phase(ownership,'membership_replay');require(np.array_equal(joined,join_membership(cohorts,b)))
    f=attach(f,joined);del cohorts
    require(PRIVATE.parent.is_dir() and not PRIVATE.parent.is_symlink());PRIVATE.mkdir(mode=0o700)
    parent.private_write(PRIVATE/'membership.npz',lambda h:np.savez_compressed(h,membership=joined))
    support=support_report(f);validate_support(support);write_json(OUT/'support.json',support);pins={}
    for family in FAMILIES:
        result=panel.fit_family(f,family,progress=lambda n:phase(ownership,n,family))
        panel.validate_report(result.aggregate);phase(ownership,'private_write',family)
        obj={'family':family,'frame':'R7_attempt1_fold0','protocol_sha256':sha(OUT/'protocol.json'),
            'aggregate':result.aggregate,'objects':result.objects}
        path=PRIVATE/(family+'.joblib');parent.private_write(path,lambda h:joblib.dump(obj,h,compress=3))
        pin=sha(path);phase(ownership,'object_replay',family);restored=joblib.load(path)
        before=verify_objects(f,result.aggregate,result.objects)
        after=verify_objects(f,restored['aggregate'],restored['objects'])
        require(set(before)==set(after) and all(np.array_equal(before[k],after[k]) for k in before)
            and sha(path)==pin)
        c={'result':result.aggregate,'objects_sha256':pin,'object_replay_exact':True,'patient_level_output_emitted':False}
        validate_component(c,family);write_json(OUT/(family+'.json'),c);pins[family]=sha(OUT/(family+'.json'))
        del result,obj,restored,before,after
    phase(ownership,'post_authentication');require(p==protocol(bindings()))
    write_json(OUT/'aggregate.json',{'schema':'bran-expanded-clinical-s7','status':'completed',
        'components_sha256':pins,'private_membership_sha256':sha(PRIVATE/'membership.npz'),
        'support_sha256':sha(OUT/'support.json'),'membership_replay_exact':True,
        'original_three_family_join_replay_exact':True,**FLAGS})
    phase(ownership,'completed');t={'status':'authenticated_completed','protocol_sha256':sha(OUT/'protocol.json'),
        'aggregate_sha256':sha(OUT/'aggregate.json'),'patient_level_output_emitted':False}
    authenticate(pending=t);write_json(OUT/'completed.json',t)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    ownership={'owned':False,'phase':'authentication'};ok=False;t=None
    with parent.quiet():
        try:
            with threadpool_limits(limits=1),parent.LOCK.open('a+b') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if args.audit_only:_,t=authenticate()
                else:run(ownership)
                ok=True
        except Exception as exc:
            if ownership['owned']:
                if (OUT/'completed.json').exists():os.replace(OUT/'completed.json',OUT/'rejected_completed.json')
                write_json(OUT/'failure.json',{'status':'technical_failure','phase':ownership['phase'],
                    'safe_exception_chain':safe_trace(exc,ROOT,set(CODE)),**FLAGS})
    print(json.dumps(t or {'status':'completed' if ok else 'not_completed','phase':ownership['phase'],
        'patient_level_output_emitted':False}))
    return 0 if ok else 1

if __name__=='__main__':raise SystemExit(main())
