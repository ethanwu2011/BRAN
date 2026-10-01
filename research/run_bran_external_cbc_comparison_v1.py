"""Prospectively bound external-CBC warm start vs paired V2 lineage control.

Only --prepare-protocol binds row-free receipts. --run reads patient-derived
arrays locally under OS stdout/stderr suppression and releases closed aggregates.
No old scientific run is called, no clinical image is read, and no private row,
state, prediction or bootstrap draw is exported.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from bran_clinical_semantics_v1 import CBC_FIELDS,CANONICAL_UNITS
from run_bran_source_linkage_audit_v1 import sha,exclusive_json
import run_bran_observed_cbc_pool_v1 as pool
import run_bran_anchor_ablation_v2 as lineage
import run_bran_overnight_diagnostic_v1 as base
import bran_external_cbc_evaluation_v1 as evaluation

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'BRAN_EXTERNAL_CBC_COMPARISON_PROTOCOL_V1.json'
OUT=ROOT/'BRAN_EXTERNAL_CBC_COMPARISON_V1'
PRIVATE=ROOT/'private_artifacts'/'bran_external_cbc_comparison_v1'
SOURCES=('mimic','nhanes','eicu')
UNIT_FILE='PATIENT_ATLAS_OFFICIAL_UNIT_RECONCILIATION_V1.json'
UNIT_SHA='4d428667185a974116b0f13d525bcf86c9837ea86806604b26f6f0c41184e167'
CODE=('run_bran_external_cbc_comparison_v1.py','bran_external_cbc_evaluation_v1.py',
    'bran_external_cbc_fit_kernel_v1.py','bran_external_cbc_pretraining_v1.py',
    'bran_patient_state_anchor_v2.py','bran_patient_state_prototype_v1.py',
    'run_bran_anchor_ablation_v2.py','run_bran_overnight_diagnostic_v1.py',
    'run_bran_observed_cbc_pool_v1.py','bran_clinical_semantics_v1.py',
    'run_bran_source_linkage_audit_v1.py','bran_clinical_dictionary_binding_v1.py',
    'test_bran_external_cbc_pretraining_v1.py','test_bran_external_cbc_fit_kernel_v1.py',
    'test_bran_external_cbc_evaluation_v1.py','test_run_bran_external_cbc_comparison_v1.py')
PARAMETERS={
    'external_steps_per_fold':3000,'paired_steps_per_arm_fold':1500,'batch_size':96,
    'external_learning_rate':.001,'external_weight_decay':0.,
    'paired_learning_rate':.0003,'paired_weight_decay':.0001,
    'seed_base':1701,'external_numpy_seed_base':19001,'outer_folds':5,
    'bootstrap_draws':1000,'bootstrap_seed':91501,'cbc_ridge_alpha':1.,
    'screening_logistic_c':1.,'screening_max_iter':5000,
    'primary':'hemoglobin whole-CBC-hidden original-unit MAE: candidate vs control and raw-feature Ridge',
    'completion_secondary':'all nine CBC fields original-unit MAE/MSE; no incompatible-unit macro average',
    'screening_secondary':'same26 recorded-condition AUROCs; no claim of incident risk or new FM superiority',
    'uncertainty':'paired patient bootstrap within five outer folds; fixed OOF predictions, no refits; secondary contrasts exploratory unadjusted',
    'normalization':'current outer-training-fold medians/IQRs; no external or outer-test fitted normalization',
    'external_age_policy':'age-free clinical pretraining; source lower-bound>=18 for admission, recipient age-input column retained',
    'external_sampling':'uniform source, uniform train adult person, uniform eligible episode; per-row field-mean loss',
    'external_loss':'SmoothL1 hidden +0.1 visible; uniformly hide1..k-1 observed CBC fields; temporary128-to9head discarded',
    'transfer':'clinical_encoder and clinical_residual only, identical initialization seed; preserve recipient age column and nonclinical parameters',
    'source_admission':'authenticated MIMIC/NHANES/eICU original numeric unit-qualified CBC observations; ICU firstday/firsthour, NHANES singlecycleexam; >=2 measuredfields; private train split only',
    'assay_limitation':'eICU exact source CBC lab names and compatible recorded units; heterogeneous hospital assays not assumed interchangeable or separately validated',
    'unresolved_sources':'NWICU age, SICdb unit/method/observation build, Zigong originalfiles; excluded from this fit, not declared unavailable in principle',
    'raw_controls':'same available clinical values+masks, retinal features+availability, age; allCBCvalues+flags removed for completion',
    'state':'one192Dmean shared64+retinalprivate64+clinicalprivate64; observed age appended to fixed downstream probes',
    'release_threshold':20,'official_test_used':False,'patient_level_output_permitted':False,
    'automatic_promotion':False,'clinical_use_permitted':False,
}


def authenticate_units():
    if sha(ROOT/UNIT_FILE)!=UNIT_SHA: raise ValueError('official CBC unit policy differs')
    units=json.loads((ROOT/UNIT_FILE).read_text())
    approved={f['name']:f['canonical_unit'] for f in units['fields'] if f.get('canonical_unit_authorized') is True}
    if any(approved.get(f)!=CANONICAL_UNITS[f] for f in CBC_FIELDS): raise ValueError('CBC original-unit alignment failed')


def source_receipt(source):
    directory=pool.PUBLIC/source
    if (directory/'failure.json').exists(): raise ValueError('source pool has failure state')
    a=json.loads((directory/'aggregate.json').read_text()); m=json.loads((directory/'manifest.json').read_text())
    pool.validate_aggregate(a)
    if a['source']!=source or m['aggregate_sha256']!=sha(directory/'aggregate.json') or m['protocol_sha256']!=sha(pool.PROTOCOL): raise ValueError('source receipt hash mismatch')
    return {'aggregate_sha256':sha(directory/'aggregate.json'),'manifest_sha256':sha(directory/'manifest.json'),
        'private_cache_sha256':a['private_cache_sha256'],'pool_protocol_sha256':sha(pool.PROTOCOL)}


def prepare():
    from patient_atlas_v6_2_expanded_endpoint_evaluation import FROZEN_SUPPORT_RECEIPT_NAME,load_eligible_support_receipt
    authenticate_units()
    pool.validate_protocol(json.loads(pool.PROTOCOL.read_text()))
    old=lineage.validate_protocol(ROOT,ROOT/'BRAN_ANCHOR_ABLATION_PROTOCOL_V2.json')
    support=load_eligible_support_receipt(ROOT/FROZEN_SUPPORT_RECEIPT_NAME,project_root=ROOT)
    if len(support.eligible_sources)!=26: raise ValueError('endpoint support differs')
    return {'schema_version':'bran-external-cbc-comparison-protocol-v1','status':'frozen_before_execution',
        'parameters':PARAMETERS,'sources':{s:source_receipt(s) for s in SOURCES},
        'code_sha256':{n:sha(ROOT/n) for n in CODE},'unit_policy_sha256':UNIT_SHA,
        'lineage_protocol_sha256':sha(ROOT/'BRAN_ANCHOR_ABLATION_PROTOCOL_V2.json'),
        'authentication':old['authentication'],'data_roots':old['data_roots'],
        'endpoint_names':list(support.eligible_sources)}


def validate_protocol(p):
    keys={'schema_version','status','parameters','sources','code_sha256','unit_policy_sha256',
        'lineage_protocol_sha256','authentication','data_roots','endpoint_names'}
    if set(p)!=keys or p['schema_version']!='bran-external-cbc-comparison-protocol-v1' or p['status']!='frozen_before_execution' or p['parameters']!=PARAMETERS: raise ValueError('comparison protocol mismatch')
    if set(p['sources'])!=set(SOURCES) or set(p['code_sha256'])!=set(CODE): raise ValueError('comparison source/code binding mismatch')
    for name,h in p['code_sha256'].items():
        if sha(ROOT/name)!=h: raise ValueError('comparison implementation changed')
    authenticate_units()
    if p['unit_policy_sha256']!=UNIT_SHA or p['lineage_protocol_sha256']!=sha(ROOT/'BRAN_ANCHOR_ABLATION_PROTOCOL_V2.json'): raise ValueError('lineage or unit binding mismatch')
    old=lineage.validate_protocol(ROOT,ROOT/'BRAN_ANCHOR_ABLATION_PROTOCOL_V2.json')
    if old['authentication']!=p['authentication'] or old['data_roots']!=p['data_roots']: raise ValueError('paired evaluation binding mismatch')
    for s in SOURCES:
        if p['sources'][s]!=source_receipt(s): raise ValueError('source pool changed')


def load_external(p):
    """Called only under FD suppression; private values never returned to tools."""
    result={}
    for s in SOURCES:
        path=pool.PRIVATE/(s+'.npz')
        if sha(path)!=p['sources'][s]['private_cache_sha256']: raise ValueError('private source cache mismatch')
        with np.load(path,allow_pickle=False) as archive:
            arrays={k:archive[k] for k in ('values','observed','provenance','person_group','split','adult_qualified')}
        if not np.array_equal(arrays.pop('provenance'),arrays['observed'].astype(np.uint8)):
            raise ValueError('imputed external targets prohibited')
        result[s]=arrays
    return result


def context(p):
    from patient_atlas_v6_2_expanded_endpoint_evaluation import (FROZEN_SUPPORT_RECEIPT_NAME,
        EXACT_OUTER_FOLD_HASH,EXACT_INNER_FOLD_ASSIGNMENT_SHA256,load_eligible_support_receipt,validate_support_against_observed)
    from run_patient_atlas_v6_2_expanded_endpoint_evaluation import _load_actual_v6_2_context
    support=load_eligible_support_receipt(ROOT/FROZEN_SUPPORT_RECEIPT_NAME,project_root=ROOT)
    ctx=_load_actual_v6_2_context(root=ROOT,**p['data_roots'],support=support)
    binding=p['authentication']; folds=np.asarray(ctx['outer_assignment'],int)
    if list(support.eligible_sources)!=p['endpoint_names'] or len(folds)!=1928 or set(folds)!={0,1,2,3,4}: raise ValueError('paired cohort identity mismatch')
    if binding['outer_fold_sha256']!=EXACT_OUTER_FOLD_HASH or binding['canonical_source_hashes']!=dict(ctx['source_hashes']) or binding['support_receipt_sha256']!=support.receipt_sha256:
        raise ValueError('paired source or support identity mismatch')
    validate_support_against_observed(support,ctx['labels_by_source'],ctx['observed_by_source'],folds)
    for f in range(5):
        _,h=base._inner_context(ctx,np.flatnonzero(folds!=f),f)
        if h!=EXACT_INNER_FOLD_ASSIGNMENT_SHA256[f] or h!=binding['inner_fold_sha256'][f]: raise ValueError('inner fold identity mismatch')
    return ctx,folds


def fit_and_score(p):
    import torch
    from bran_external_cbc_fit_kernel_v1 import external_warm_start,paired_train
    ctx,folds=context(p); external=load_external(p)
    c0,cm0,eligible,r0,rm,names=lineage._actual_arrays(ROOT,ctx)
    eligible[:,48:]=False
    if tuple(np.flatnonzero(eligible[0,:48]))!=lineage.ELIGIBLE_CONTINUOUS_INDICES or not np.array_equal(eligible, np.broadcast_to(eligible[0],eligible.shape)):
        raise ValueError('clinical eligibility changed')
    ages=np.asarray(ctx['raw_cohort'].ages)
    cbc_indices=np.array([names.index(f) for f in CBC_FIELDS])
    screening={e:{a:np.full(len(folds),np.nan) for a in evaluation.SCREEN_ARMS} for e in p['endpoint_names']}
    completion={a:np.full((len(folds),9),np.nan) for a in evaluation.CBC_ARMS}
    PRIVATE.mkdir(parents=True,mode=0o700)  # Exclusive model attempt directory.
    for f in range(5):
        base._atomic_progress(OUT/'progress.json','external_pretraining',f)
        tr,te=np.flatnonzero(folds!=f),np.flatnonzero(folds==f)
        transform=base.FoldTransform(c0,cm0,eligible,r0,rm,ages,tr)
        c,cm,r,age=transform.apply(c0,cm0,eligible,r0,rm,ages)
        warm=external_warm_start(external,tuple(names),transform.clinical_median,transform.clinical_iqr,
            seed=1701+f,steps=PARAMETERS['external_steps_per_fold'],batch_size=96)
        base._atomic_progress(OUT/'progress.json','paired_alignment',f)
        control=paired_train(c,cm,r,rm,age,tr,seed=1701+f,steps=PARAMETERS['paired_steps_per_arm_fold'])
        candidate=paired_train(c,cm,r,rm,age,tr,seed=1701+f,steps=PARAMETERS['paired_steps_per_arm_fold'],warm_start=warm)
        models={'control':control,'candidate':candidate}
        arms={'raw_clinical':np.c_[c,cm,age],'raw_retinal':np.c_[r,rm,age],
              'raw_concat':np.c_[c,cm,r,rm,age]}
        for version,model in models.items():
            for route,z in lineage._state_routes(model,c,cm,r,rm,age).items():
                if z.shape!=(len(folds),192): raise ValueError('patient state width changed')
                arms[version+'_'+route]=np.c_[z,age]
        base._atomic_progress(OUT/'progress.json','fixed_screening_probes',f)
        for endpoint in p['endpoint_names']:
            for arm in evaluation.SCREEN_ARMS:
                screening[endpoint][arm][te]=evaluation.fixed_screen_probe(arms[arm],ctx['labels_by_source'][endpoint],ctx['observed_by_source'][endpoint],tr,te)
        hidden,hiddenmask,keep,idx=evaluation.whole_cbc_inputs(c,cm,tuple(names))
        inputs={'raw':np.c_[hidden[:,keep],hiddenmask[:,keep],r,rm,age]}
        for version,model in models.items():
            inputs[version]=np.c_[lineage._state_routes(model,hidden,hiddenmask,r,rm,age)['both'],age]
        for j,slot in enumerate(idx):
            for version in evaluation.CBC_ARMS:
                completion[version][te,j]=evaluation.fixed_cbc_probe(inputs[version],c0[:,slot],cm[:,slot],tr,te)
        # Only model parameters and train-fitted transformations, not patient
        # embeddings/predictions, are saved in this private fold checkpoint.
        checkpoint=PRIVATE/('fold'+str(f)+'.pt')
        with checkpoint.open('xb') as handle:
            torch.save({'control':control.state_dict(),'candidate':candidate.state_dict(),
                'clinical_median':transform.clinical_median,'clinical_iqr':transform.clinical_iqr,
                'retinal_mean':transform.retinal_mean,'retinal_scale':transform.retinal_scale,
                'age_mean':transform.age_mean,'age_scale':transform.age_scale,'protocol_sha256':sha(PROTOCOL)},handle)
        os.chmod(checkpoint,0o600)
    base._atomic_progress(OUT/'progress.json','paired_uncertainty')
    counts=evaluation.paired_counts(folds,draws=1000,seed=91501)
    cbc=evaluation.summarize_cbc(c0[:,cbc_indices],cm0[:,cbc_indices]&eligible[:,cbc_indices],completion,counts)
    screen=evaluation.summarize_screening({e:ctx['labels_by_source'][e] for e in p['endpoint_names']},
        {e:ctx['observed_by_source'][e] for e in p['endpoint_names']},screening,folds,counts,p['endpoint_names'])
    return {'schema_version':'bran-external-cbc-comparison-aggregate-v1','status':'completed',
        'paired_people':1928,'recorded_conditions':26,'cbc_fields':9,'state_width':192,
        'cbc_whole_panel_hidden':cbc,'screening':screen,'patient_level_output_emitted':False,
        'official_test_used':False,'automatic_promotion':False,'clinical_use_permitted':False,
        'external_training_steps':15000,'paired_training_steps_total':15000,
        'scope':'internal_development_frozen_budget_compatible_clinical_warm_start'}


def validate_result(a,p):
    keys={'schema_version','status','paired_people','recorded_conditions','cbc_fields','state_width',
        'cbc_whole_panel_hidden','screening','patient_level_output_emitted','official_test_used',
        'automatic_promotion','clinical_use_permitted','external_training_steps','paired_training_steps_total','scope'}
    if set(a)!=keys or a['schema_version']!='bran-external-cbc-comparison-aggregate-v1' or a['status']!='completed': raise ValueError('aggregate schema invalid')
    for k,v in {'paired_people':1928,'recorded_conditions':26,'cbc_fields':9,'state_width':192,'external_training_steps':15000,'paired_training_steps_total':15000}.items():
        if type(a[k]) is not int or a[k]!=v: raise ValueError('aggregate count invalid')
    if any(a[k] is not False for k in ('patient_level_output_emitted','official_test_used','automatic_promotion','clinical_use_permitted')) or a['scope']!='internal_development_frozen_budget_compatible_clinical_warm_start': raise ValueError('aggregate claim/privacy flags invalid')
    def numeric_dict(d,keys):
        if set(d)!=set(keys): raise ValueError('aggregate metric keys invalid')
        for k,v in d.items():
            if k=='ci95':
                if type(v) is not list or len(v)!=2 or any(type(x) is not float or not np.isfinite(x) for x in v) or v[0]>v[1]: raise ValueError('invalid aggregate interval')
            elif type(v) is not float or not np.isfinite(v): raise ValueError('invalid aggregate numeric value')
            elif k in ('mae','mse') and v<0: raise ValueError('negative aggregate error metric')
            elif k=='auroc' and not 0<=v<=1: raise ValueError('invalid aggregate AUROC')
        if 'auroc' in d and any(x<0 or x>1 for x in d['ci95']): raise ValueError('invalid AUROC interval')
    cbc=a['cbc_whole_panel_hidden']
    if set(cbc)!=set(CBC_FIELDS): raise ValueError('CBC result fields invalid')
    for d in cbc.values():
        if set(d)!={'observed_count_lower_bound_20','arms','paired_deltas'} or type(d['observed_count_lower_bound_20']) is not int or d['observed_count_lower_bound_20']<20 or d['observed_count_lower_bound_20']%20: raise ValueError('CBC privacy counts invalid')
        if set(d['arms'])!=set(evaluation.CBC_ARMS) or set(d['paired_deltas'])!={'candidate-control','candidate-raw'}: raise ValueError('CBC arm keys invalid')
        for v in d['arms'].values(): numeric_dict(v,('mae','mse'))
        for v in d['paired_deltas'].values(): numeric_dict(v,('mae_delta','ci95'))
    screen=a['screening']; contrasts={a+'-'+b for a,b in evaluation.SCREEN_COMPARISONS}
    if set(screen)!={'endpoints','macro_auroc','macro_paired_deltas'} or set(screen['endpoints'])!=set(p['endpoint_names']): raise ValueError('screening result registry invalid')
    numeric_dict(screen['macro_auroc'],evaluation.SCREEN_ARMS)
    if any(x<0 or x>1 for x in screen['macro_auroc'].values()): raise ValueError('invalid macro AUROC')
    if set(screen['macro_paired_deltas'])!=contrasts: raise ValueError('screening contrasts invalid')
    for v in screen['macro_paired_deltas'].values(): numeric_dict(v,('auroc_delta','ci95'))
    for d in screen['endpoints'].values():
        if set(d)!={'arms','paired_deltas'} or set(d['arms'])!=set(evaluation.SCREEN_ARMS) or set(d['paired_deltas'])!=contrasts: raise ValueError('screening arm keys invalid')
        for v in d['arms'].values(): numeric_dict(v,('auroc','ci95'))
        for v in d['paired_deltas'].values(): numeric_dict(v,('auroc_delta','ci95'))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare-protocol',action='store_true'); parser.add_argument('--run',action='store_true'); args=parser.parse_args()
    if args.prepare_protocol==args.run: parser.error('choose exactly one operation')
    ok=False; owned=False; phase='protocol'; started=time.monotonic()
    with _quiet():
        try:
            if args.prepare_protocol: exclusive_json(PROTOCOL,prepare()); ok=True
            else:
                p=json.loads(PROTOCOL.read_text()); validate_protocol(p)
                OUT.mkdir(); owned=True; phase='fit_and_evaluate'
                a=fit_and_score(p); phase='aggregate_validation'; validate_result(a,p)
                exclusive_json(OUT/'aggregate.json',a)
                exclusive_json(OUT/'manifest.json',{'protocol_sha256':sha(PROTOCOL),'aggregate_sha256':sha(OUT/'aggregate.json'),
                    'elapsed_seconds':round(time.monotonic()-started,1),'patient_level_output_emitted':False})
                ok=True
        except Exception as error:
            allowed={ValueError:'ValueError',TypeError:'TypeError',KeyError:'KeyError',RuntimeError:'RuntimeError',
                MemoryError:'MemoryError',ImportError:'ImportError',FileNotFoundError:'FileNotFoundError',OSError:'OSError'}
            if owned: exclusive_json(OUT/'failure.json',{'status':'comparison_execution_failed','phase':phase,
                'error_class':allowed.get(type(error),'other_execution_error'),'patient_level_output_emitted':False})
    print(json.dumps({'status':('comparison_protocol_prepared' if args.prepare_protocol else 'comparison_completed') if ok else 'comparison_execution_failed','patient_level_output_emitted':False}))
    return 0 if ok else 1


if __name__=='__main__': raise SystemExit(main())
