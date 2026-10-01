"""One discovery-trained joint state for the fixed23cohort subtype panel."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import sha,exclusive_json
import bran_september_push_io_v1 as io
import run_bran_overnight_diagnostic_v1 as base
import run_bran_anchor_ablation_v2 as lineage
import bran_cgm_added_value_experiment_v1 as cgm_source
import bran_cgm_target_io_v1 as target_io

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'BRAN_JOINT_SUBTYPING_PROTOCOL_V1.json'
OUT=ROOT/'BRAN_JOINT_SUBTYPING_V1'
PRIVATE=ROOT/'private_artifacts/bran_joint_subtyping_v1'
SOURCE_PROTOCOL='BRAN_FIXED_SUBGROUP_VALIDATION_PROTOCOL_V1.json'
SOURCE_PIN='0d65fbadcca0e972f6e026b9d56afc55ab0f9b8672e2b9c4427fee3f7e929b89'
JOINT_PIN='34ba125921bbf00127c9d5a43f1483a6b18f1b8201eacfee9c4bf26af4d6e021'
PARAMETERS={'initial_steps':1500,'adaptation_steps':1500,'initial_seed':1701,'adaptation_seed':92401,
    'batch_size':96,'discovery_folds':[0,1,2],'validation_fold':3,'replication_fold':4,
    'normalization':'discovery only','state_width':192,'initialization':'fresh V2 discoveryfit, NEVER an outer-fold checkpoint',
    'group_rules':'unchanged prior23cohort support80/40/40; PCA8 diagGMMK1..4 validation1SE;50bootstrap fixedencoder',
    'utility':'CGM manifestaverage beyond severity+technical+26conditionvalues+26indicators; Kminus1onehot',
    'automatic_promotion':False,'novel_subtype_claim':False,'patient_level_output_permitted':False,
    'official_test_used':False,'adaptive_development':True}
CODE=('run_bran_joint_subtyping_v1.py','test_run_bran_joint_subtyping_v1.py','bran_joint_subtyping_kernel_v1.py',
      'test_bran_joint_subtyping_kernel_v1.py','bran_multigroup_utility_kernel_v1.py','test_bran_multigroup_utility_kernel_v1.py',
      'BRAN_JOINT_SUBTYPING_DESIGN_V1.md','bran_cbc_completion_io_v1.py')


def phenotype_receipt():
    if sha(ROOT/SOURCE_PROTOCOL)!=SOURCE_PIN:raise ValueError('phenotype_protocol_changed')
    p=json.loads((ROOT/SOURCE_PROTOCOL).read_text())
    for name,digest in p['expected_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('phenotype_code_changed')
    path=cgm_source._target_path(p)
    if sha(path)!=p['manifest_hashes']['cgm']:raise ValueError('cgm_source_changed')
    return {'protocol_sha256':SOURCE_PIN,'cgm_sha256':p['manifest_hashes']['cgm'],
            'target_contract':target_io.TARGET_CONTRACT,'code_sha256':p['expected_hashes']}


def prepare():
    import run_bran_screening_joint_v1 as joint
    import bran_joint_subtyping_kernel_v1 as kernel
    if sha(joint.PROTOCOL)!=JOINT_PIN:raise ValueError('joint_recipe_changed')
    p=json.loads(joint.PROTOCOL.read_text());joint.validate_protocol(p)
    phenotype=phenotype_receipt()
    code={**p['code_sha256'],**phenotype['code_sha256'],**io.code_closure(CODE)}
    return {'schema':'bran-joint-subtyping-protocol-v1','status':'frozen_before_execution','parameters':PARAMETERS,
        'joint_protocol_sha256':JOINT_PIN,'source':p['source'],'phenotype':phenotype,
        'candidate_codes':list(kernel.CODES),'code_sha256':code,'runtime':io.runtime()}


def validate_protocol(p):
    if p!=prepare():raise ValueError('protocol_changed')


def discovery_indices(folds):
    if folds.ndim!=1 or folds.dtype.kind not in 'iu' or set(folds)!={0,1,2,3,4}:raise ValueError('folds_invalid')
    return np.flatnonzero(folds<3)


def run(p):
    import torch
    from bran_clinical_semantics_v1 import CBC_FIELDS
    from bran_external_cbc_fit_kernel_v1 import paired_train
    from bran_screening_joint_kernel_v1 import adapt
    import bran_joint_subtyping_kernel_v1 as kernel
    ctx,folds,c0,cm0,eligible,r0,rm,ages,names=io.load_context()
    old=json.loads((ROOT/SOURCE_PROTOCOL).read_text())
    source=cgm_source._target_path(old)
    if sha(source)!=p['phenotype']['cgm_sha256']:raise ValueError('cgm_hash_invalid')
    with source.open('r',encoding='utf-8',newline='') as handle:target=target_io.parse_manifest(handle)
    y,ym=target_io.align(target,tuple(ctx['raw_cohort'].patient_ids))
    if sha(source)!=p['phenotype']['cgm_sha256']:raise ValueError('cgm_changed_during_load')
    memberships={e:np.asarray(ctx['observed_by_source'][e],bool)&(np.asarray(ctx['labels_by_source'][e])==1) for e in kernel.CODES}
    endpoints=p['source']['endpoint_names'];labels=np.column_stack([ctx['labels_by_source'][e] for e in endpoints]);lm=np.column_stack([ctx['observed_by_source'][e] for e in endpoints]).astype(bool)
    fit=discovery_indices(folds)
    transform=base.FoldTransform(c0,cm0,eligible,r0,rm,ages,fit)
    c,cm,r,age=transform.apply(c0,cm0,eligible,r0,rm,ages)
    base._atomic_progress(OUT/'progress.json','discovery_initialization')
    initial=paired_train(c,cm,r,rm,age,fit,seed=1701,steps=1500,batch_size=96)
    base._atomic_progress(OUT/'progress.json','discovery_joint_adaptation')
    model=adapt(initial,c,cm,r,rm,age,labels,lm,fit,tuple(names.index(f) for f in CBC_FIELDS),seed=92401,steps=1500,batch_size=96,candidate=True)
    model.eval();states=lineage._state_routes(model,c,cm,r,rm,age)['both']
    base._atomic_progress(OUT/'progress.json','fixed_cohort_structure_and_utility')
    result=kernel.evaluate(states,c0,cm,rm,age,names,tuple(ctx['raw_cohort'].site_ids),folds,memberships,labels,lm,y,ym,
                           normalized_clinical=c,profile_age=ages)
    PRIVATE.mkdir(parents=True,mode=0o700);path=PRIVATE/'discovery.pt'
    bundle={'candidate':model.state_dict(),'initial':initial.state_dict(),
        **{k:getattr(transform,k) for k in ('clinical_median','clinical_iqr','retinal_mean','retinal_scale','age_mean','age_scale')},
        'protocol_sha256':sha(PROTOCOL),'joint_protocol_sha256':JOINT_PIN,'endpoint_names':endpoints,'cbc_fields':list(CBC_FIELDS),
        'training_folds':[0,1,2],'state_width':192}
    with path.open('xb') as handle:torch.save(bundle,handle)
    os.chmod(path,0o600)
    return {'schema':'bran-joint-subtyping-aggregate-v1','status':'completed','result':result,
        'paired_people':1928,'discovery_only_encoder':True,'one_coordinate_frame':True,
        'patient_level_output_emitted':False,'official_test_used':False,'automatic_promotion':False,
        'adaptive_development':True},sha(path)


def validate_result(a,p):
    import bran_joint_subtyping_kernel_v1 as kernel
    if set(a)!={'schema','status','result','paired_people','discovery_only_encoder','one_coordinate_frame',
                'patient_level_output_emitted','official_test_used','automatic_promotion','adaptive_development'}:
        raise ValueError('aggregate_schema_invalid')
    if a['schema']!='bran-joint-subtyping-aggregate-v1' or a['status']!='completed' or a['paired_people']!=1928:
        raise ValueError('aggregate_identity_invalid')
    if any(a[k] is not True for k in ('discovery_only_encoder','one_coordinate_frame','adaptive_development')) or any(a[k] is not False for k in ('patient_level_output_emitted','official_test_used','automatic_promotion')):
        raise ValueError('aggregate_claim_invalid')
    if a['result']['candidate_codes']!=p['candidate_codes']:raise ValueError('cohort_panel_changed')
    kernel.validate_result(a['result'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--protocol-sha256');args=parser.parse_args()
    if args.prepare==args.run:parser.error('choose_one_operation')
    ok=False;owned=False;phase='protocol';start=time.monotonic()
    with _quiet():
        try:
            if args.prepare:exclusive_json(PROTOCOL,prepare());ok=True
            else:
                if not args.protocol_sha256 or sha(PROTOCOL)!=args.protocol_sha256:raise ValueError('protocol_sha_invalid')
                p=json.loads(PROTOCOL.read_text());validate_protocol(p)
                if PRIVATE.exists():raise ValueError('attempt_exists')
                OUT.mkdir();owned=True;phase='discovery_fit_and_evaluation';a,checkpoint=run(p)
                phase='terminal_validation';validate_protocol(p);validate_result(a,p)
                exclusive_json(OUT/'aggregate.json',a)
                exclusive_json(OUT/'manifest.json',{'protocol_sha256':sha(PROTOCOL),'aggregate_sha256':sha(OUT/'aggregate.json'),
                    'checkpoint_sha256':checkpoint,'elapsed_seconds':round(time.monotonic()-start,1),'patient_level_output_emitted':False})
                base._atomic_completed(OUT/'progress.json');ok=True
        except Exception as e:
            if owned:exclusive_json(OUT/'failure.json',{'status':'execution_failed','phase':phase,
                'error_class':type(e).__name__ if type(e) in (ValueError,TypeError,KeyError,RuntimeError,OSError,ImportError) else 'other_execution_error',
                'patient_level_output_emitted':False})
    print(json.dumps({'status':('protocol_prepared' if args.prepare else 'joint_subtyping_completed') if ok else 'execution_failed','patient_level_output_emitted':False}))
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
