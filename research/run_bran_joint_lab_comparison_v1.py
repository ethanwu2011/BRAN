"""Isolated chemistry-conditioned follow-up. All real processing stays FD-quiet."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

import run_bran_external_cbc_comparison_v1 as prior
from bran_clinical_dictionary_binding_v1 import _quiet
from bran_joint_lab_cache_v1 import FIELDS, coarse_count
from run_bran_source_linkage_audit_v1 import sha, exclusive_json

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT/'BRAN_JOINT_LAB_COMPARISON_PROTOCOL_V1.json'
OUT = ROOT/'BRAN_JOINT_LAB_COMPARISON_V1'
PRIVATE = ROOT/'private_artifacts'/'bran_joint_lab_comparison_v1'
PRIOR_PROTOCOL_SHA = '88f4d929bc037a3febe7f158ab87f0fd17108a814d8918f057e0c6fef8e7a3b0'
PRIOR_AGGREGATE_SHA = 'f64998841b1f5009e2dab7c6beb81caa0a131fd62bbb5b48c42ff91b0f90f854'
PRIOR_AUDIT_SHA = '85cc725e5e911a7914871b0b2b69054ddfc6d72ac830e169b6e8669b51d78824'
SOURCE_SPECS = {
    'mimic': ('run_bran_mimic_joint_labs_v1',
        '96803a0613dd19a2171b1d97b98ab56228767c90794f2068c48e522f96e37246',
        '016fc7e86799a602dbb967ecfb7c7d314d473609f321be1002d715a173bf0eda',
        'BRAN_MIMIC_JOINT_LABS_AUDIT_V1/audit.json',
        '63767a71ff859a04cb183797274b31845a425775ec9c43b821d0a560832b986d'),
    'nhanes': ('run_bran_nhanes_joint_labs_v1',
        'fdbec0e9333a2622aa6522d05e39fbcef62b92680246c409a7574cbf26b7ce45',
        '55f1d7ec79e226a23a879c288b205cc1d587ba1f35c81737aeb9ef85bfdc2949',
        'BRAN_NHANES_JOINT_LABS_AUDIT_V1/audit.json',
        '2d4c3d98dde231cc0dc99dc2937cebc63ad6e43468761bd103b6887f5e37d662'),
    'eicu': ('run_bran_eicu_cbc_cache_v2',
        '11865963ef2eb69137d1cf56b9caeea203dd3330d2c11ada09aa966857567341',
        'a40d3222c99665e59684a65dc13928857162410c89ed1c7365f3a2cc1a135a1b', None, None),
}
PARAMETERS = {
    'external_steps_per_fold':3000,'whole_cbc_steps_per_fold':1500,'partial_cbc_steps_per_fold':1500,
    'paired_steps_per_arm_fold':1500,'batch_size':96,'seed_base':1701,
    'external_learning_rate':0.001,'external_weight_decay':0.,'torch_threads':2,
    'task_schedule':'whole on even zero-based steps; partial on odd',
    'sampling':'uniform eligible source, then person, then task-eligible episode; adult and private train split0 only',
    'whole_eligibility':'at least2observed CBC plus at least1observed chemistry',
    'partial_eligibility':'at least2observed CBC; chemistry optional',
    'loss':'mean within row then mean rows; hidden SmoothL1 plus0.1visible CBC only for partial task',
    'external_age':'disabled; preserve recipient age column on transfer',
    'normalization':'AI-READI outer-training fold only; canonical21to59 mapping',
    'evaluation':'same fixedRidge1/LogisticC1;1000paired bootstrap draws seed91501; all26conditions and9CBCfields',
    'primary':'whole-CBC-hidden hemoglobin MAE candidate-control and candidate-raw',
    'screening_goal':'preserve or improve mean26-condition AUROC; no noninferiority margin invented',
    'control':'same-budget paired control replay; verify against prior preserved control',
    'limitation':'adaptive repeated development; nominal intervals do not adjust experiment selection',
    'automatic_promotion':False,'automatic_third_branch':False,
}
CODE = tuple(dict.fromkeys(prior.CODE + (
    'run_bran_joint_lab_comparison_v1.py','test_run_bran_joint_lab_comparison_v1.py',
    'bran_joint_lab_pretraining_v1.py','test_bran_joint_lab_pretraining_v1.py',
    'bran_joint_lab_task_contract_v1.py','test_bran_joint_lab_task_contract_v1.py',
    'bran_joint_lab_cache_v1.py','bran_clinical_chemistry_semantics_v1.py',
    'audit_bran_joint_lab_cache_v1.py','test_audit_bran_joint_lab_cache_v1.py',
    'audit_bran_joint_lab_comparison_v1.py','test_audit_bran_joint_lab_comparison_v1.py',
    'audit_bran_external_cbc_comparison_v1.py','test_audit_bran_external_cbc_comparison_v1.py',
)))


def old_protocol():
    if sha(prior.PROTOCOL) != PRIOR_PROTOCOL_SHA or sha(prior.OUT/'aggregate.json') != PRIOR_AGGREGATE_SHA:
        raise ValueError('prior comparison identity changed')
    if sha(ROOT/'BRAN_EXTERNAL_CBC_COMPARISON_AUDIT_V1/audit.json') != PRIOR_AUDIT_SHA:
        raise ValueError('prior comparison audit changed')
    p = json.loads(prior.PROTOCOL.read_text()); prior.validate_protocol(p)
    prior.validate_result(json.loads((prior.OUT/'aggregate.json').read_text()),p)
    return p


def joint_units():
    from bran_clinical_semantics_v1 import CANONICAL_UNITS as cbc_units
    from bran_clinical_chemistry_semantics_v1 import CANONICAL_UNITS as chemistry_units
    if sha(ROOT/prior.UNIT_FILE) != prior.UNIT_SHA: raise ValueError('unit registry changed')
    fields = json.loads((ROOT/prior.UNIT_FILE).read_text())['fields']
    expected = {**cbc_units, **chemistry_units}; result = {}
    for name in FIELDS:
        hits = [f for f in fields if f['name'] == name]
        if len(hits)!=1 or hits[0].get('canonical_unit_authorized') is not True or hits[0]['canonical_unit']!=expected[name]:
            raise ValueError('joint canonical unit mismatch')
        result[name] = hits[0]['index']
    if any(type(i) is not int or i not in prior.lineage.ELIGIBLE_CONTINUOUS_INDICES for i in result.values()):
        raise ValueError('joint field is not active in V2')
    return result


def source_receipt(source, *, verify_raw=False):
    module, protocol_pin, aggregate_pin, audit_path, audit_pin = SOURCE_SPECS[source]
    r = importlib.import_module(module)
    if sha(r.PROTOCOL)!=protocol_pin or sha(r.PUBLIC/'aggregate.json')!=aggregate_pin or (r.PUBLIC/'failure.json').exists():
        raise ValueError('new source terminal identity mismatch')
    p=json.loads(r.PROTOCOL.read_text()); r.validate_protocol(p)
    a=json.loads((r.PUBLIC/'aggregate.json').read_text()); r.validate_aggregate(a)
    m=json.loads((r.PUBLIC/'manifest.json').read_text())
    base=a['base_pool_aggregate'] if source=='eicu' else a
    salt=r.PERSISTENT_SALT if source=='eicu' else r.SALT
    if base['source']!=source or m['protocol_sha256']!=protocol_pin or m['aggregate_sha256']!=aggregate_pin or m['split_salt_sha256']!=sha(salt):
        raise ValueError('new source manifest mismatch')
    if audit_path is not None and sha(ROOT/audit_path)!=audit_pin: raise ValueError('source semantic audit changed')
    cache=r.PRIVATE/'observations.npz'
    if cache.stat().st_mode & 0o777 != 0o600 or sha(cache)!=base['private_cache_sha256']:
        raise ValueError('private source cache changed')
    if verify_raw:
        for spec in p['source_files'].values():
            if sha(spec['path'])!=spec['sha256']: raise ValueError('original source bytes changed')
    return {'source_protocol_sha256':protocol_pin,'aggregate_sha256':aggregate_pin,
            'manifest_sha256':sha(r.PUBLIC/'manifest.json'),'private_cache_sha256':base['private_cache_sha256'],
            'split_salt_sha256':sha(salt),'semantic_audit_sha256':audit_pin}


def prepare():
    old=old_protocol()
    return {'schema':'bran-joint-lab-comparison-protocol-v1','status':'frozen_before_execution',
            'parameters':PARAMETERS,'prior_protocol_sha256':PRIOR_PROTOCOL_SHA,'prior_aggregate_sha256':PRIOR_AGGREGATE_SHA,
            'paired_authentication':old['authentication'],'endpoint_names':old['endpoint_names'],
            'unit_registry_sha256':prior.UNIT_SHA,'joint_registry_indices':joint_units(),
            'sources':{s:source_receipt(s,verify_raw=True) for s in SOURCE_SPECS},
            'code_sha256':{n:sha(ROOT/n) for n in CODE},'runtime':runtime()}


def runtime():
    import torch
    import sklearn
    import scipy
    return {'python':sys.version.split()[0],'numpy':np.__version__,'torch':str(torch.__version__),
            'sklearn':sklearn.__version__,'scipy':scipy.__version__,'device':'cpu'}


def validate_protocol(p):
    keys={'schema','status','parameters','prior_protocol_sha256','prior_aggregate_sha256','paired_authentication',
          'endpoint_names','unit_registry_sha256','joint_registry_indices','sources','code_sha256','runtime'}
    old=old_protocol()
    if set(p)!=keys or p['schema']!='bran-joint-lab-comparison-protocol-v1' or p['status']!='frozen_before_execution':
        raise ValueError('joint comparison protocol schema mismatch')
    if (p['parameters']!=PARAMETERS or p['prior_protocol_sha256']!=PRIOR_PROTOCOL_SHA or p['prior_aggregate_sha256']!=PRIOR_AGGREGATE_SHA
        or p['paired_authentication']!=old['authentication'] or p['endpoint_names']!=old['endpoint_names']
        or p['unit_registry_sha256']!=prior.UNIT_SHA or p['joint_registry_indices']!=joint_units()
        or p['code_sha256']!={n:sha(ROOT/n) for n in CODE} or set(p['sources'])!=set(SOURCE_SPECS) or p['runtime']!=runtime()):
        raise ValueError('joint comparison protocol binding mismatch')
    if p['sources']!={s:source_receipt(s) for s in SOURCE_SPECS}: raise ValueError('new source receipt changed')


def load_external(p):
    """Private source arrays; NEVER call or print outside local FD suppression."""
    from audit_bran_joint_lab_cache_v1 import check_arrays
    result={}
    for source,(module,*_) in SOURCE_SPECS.items():
        r=importlib.import_module(module); cache=r.PRIVATE/'observations.npz'
        if sha(cache)!=p['sources'][source]['private_cache_sha256']: raise ValueError('private input changed')
        with np.load(cache,allow_pickle=False) as handle: arrays={k:handle[k] for k in handle.files}
        a=json.loads((r.PUBLIC/'aggregate.json').read_text()); r.validate_aggregate(a)
        if source=='eicu': r.audit_private_arrays(arrays,a['base_pool_aggregate']['counts_lower_bounds_20'])
        else: check_arrays(arrays,source,a['counts_lower_bounds_20'])
        item={k:arrays[k] for k in ('values','observed','provenance','person_group','split','adult_qualified')}
        if source=='eicu':
            n=len(item['values'])
            values=np.full((n,21),np.nan); observed=np.zeros((n,21),bool)
            values[:,:9]=item['values']; observed[:,:9]=item['observed']
            item={**item,'values':values,'observed':observed,'provenance':observed.astype(np.uint8)}
        if sha(cache)!=p['sources'][source]['private_cache_sha256']: raise ValueError('private input changed during load')
        result[source]=item
    return result


def eligibility_counts(sources):
    result={}
    for source,item in sources.items():
        common=(item['split']==0)&item['adult_qualified']&(item['observed'][:,:9].sum(1)>=2)
        result[source]={}
        for mode,take in (('partial_cbc',common),('whole_cbc',common&item['observed'][:,9:].any(1))):
            result[source][mode]={'snapshots':coarse_count(int(take.sum())),
                                  'source_local_people':coarse_count(len(np.unique(item['person_group'][take])))}
    return result


def control_matches(new, reference):
    """Deterministic reference replay check; false is retained, never tuned away."""
    checks=[]
    for arm in ('control_both','control_clinical','control_retinal','raw_clinical','raw_retinal','raw_concat'):
        checks.append(np.isclose(new['screening']['macro_auroc'][arm],reference['screening']['macro_auroc'][arm],rtol=0,atol=1e-10))
        for endpoint in new['screening']['endpoints']:
            for metric in ('auroc','ci95'):
                checks.append(np.allclose(new['screening']['endpoints'][endpoint]['arms'][arm][metric],
                    reference['screening']['endpoints'][endpoint]['arms'][arm][metric],rtol=0,atol=1e-10))
    for field in prior.CBC_FIELDS:
        for arm in ('control','raw'):
            for metric in ('mae','mse'):
                checks.append(np.isclose(new['cbc_whole_panel_hidden'][field]['arms'][arm][metric],reference['cbc_whole_panel_hidden'][field]['arms'][arm][metric],rtol=0,atol=1e-10))
    return bool(all(checks))


def fit_and_score(p):
    import torch
    from bran_joint_lab_pretraining_v1 import external_joint_warm_start
    from bran_external_cbc_fit_kernel_v1 import paired_train
    old=old_protocol(); ctx,folds=prior.context(old); external=load_external(p)
    coverage=eligibility_counts(external)
    base,lineage,evaluation=prior.base,prior.lineage,prior.evaluation
    c0,cm0,eligible,r0,rm,names=lineage._actual_arrays(ROOT,ctx)
    eligible[:,48:]=False
    if tuple(np.flatnonzero(eligible[0,:48]))!=lineage.ELIGIBLE_CONTINUOUS_INDICES or not np.array_equal(eligible,np.broadcast_to(eligible[0],eligible.shape)):
        raise ValueError('clinical eligibility changed')
    if any(names[i]!=name for name,i in p['joint_registry_indices'].items()): raise ValueError('joint registry coordinate mismatch')
    ages=np.asarray(ctx['raw_cohort'].ages)
    cbc_indices=np.array([names.index(f) for f in prior.CBC_FIELDS])
    screening={e:{a:np.full(len(folds),np.nan) for a in evaluation.SCREEN_ARMS} for e in p['endpoint_names']}
    completion={a:np.full((len(folds),9),np.nan) for a in evaluation.CBC_ARMS}
    PRIVATE.mkdir(parents=True,mode=0o700)
    for f in range(5):
        base._atomic_progress(OUT/'progress.json','joint_external_pretraining',f)
        tr,te=np.flatnonzero(folds!=f),np.flatnonzero(folds==f)
        transform=base.FoldTransform(c0,cm0,eligible,r0,rm,ages,tr)
        c,cm,r,age=transform.apply(c0,cm0,eligible,r0,rm,ages)
        warm=external_joint_warm_start(external,tuple(names),transform.clinical_median,transform.clinical_iqr,
                                      seed=1701+f,steps=3000,batch_size=96)
        base._atomic_progress(OUT/'progress.json','paired_alignment',f)
        control=paired_train(c,cm,r,rm,age,tr,seed=1701+f,steps=1500)
        candidate=paired_train(c,cm,r,rm,age,tr,seed=1701+f,steps=1500,warm_start=warm)
        models={'control':control,'candidate':candidate}
        arms={'raw_clinical':np.c_[c,cm,age],'raw_retinal':np.c_[r,rm,age],'raw_concat':np.c_[c,cm,r,rm,age]}
        for version,model in models.items():
            for route,z in lineage._state_routes(model,c,cm,r,rm,age).items():
                if z.shape!=(len(folds),192): raise ValueError('state width changed')
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
        checkpoint=PRIVATE/('fold'+str(f)+'.pt')
        with checkpoint.open('xb') as handle:
            torch.save({'control':control.state_dict(),'candidate':candidate.state_dict(),
                'clinical_median':transform.clinical_median,'clinical_iqr':transform.clinical_iqr,
                'retinal_mean':transform.retinal_mean,'retinal_scale':transform.retinal_scale,
                'age_mean':transform.age_mean,'age_scale':transform.age_scale,'protocol_sha256':sha(PROTOCOL)},handle)
        os.chmod(checkpoint,0o600)
    base._atomic_progress(OUT/'progress.json','paired_uncertainty')
    draws=evaluation.paired_counts(folds,draws=1000,seed=91501)
    cbc=evaluation.summarize_cbc(c0[:,cbc_indices],cm0[:,cbc_indices]&eligible[:,cbc_indices],completion,draws)
    screen=evaluation.summarize_screening({e:ctx['labels_by_source'][e] for e in p['endpoint_names']},
        {e:ctx['observed_by_source'][e] for e in p['endpoint_names']},screening,folds,draws,p['endpoint_names'])
    metrics={'schema_version':'bran-external-cbc-comparison-aggregate-v1','status':'completed',
        'paired_people':1928,'recorded_conditions':26,'cbc_fields':9,'state_width':192,
        'cbc_whole_panel_hidden':cbc,'screening':screen,'patient_level_output_emitted':False,
        'official_test_used':False,'automatic_promotion':False,'clinical_use_permitted':False,
        'external_training_steps':15000,'paired_training_steps_total':15000,
        'scope':'internal_development_frozen_budget_compatible_clinical_warm_start'}
    prior.validate_result(metrics,old)
    reference=json.loads((prior.OUT/'aggregate.json').read_text())
    return {'schema':'bran-joint-lab-comparison-aggregate-v1','status':'completed',
        'metrics':metrics,'training_eligibility_lower_bounds_20':coverage,
        'control_replay_matches_prior':control_matches(metrics,reference),
        'adaptive_development':True,'intervals_adjusted_for_experiment_selection':False,
        'prior_negative_result_preserved':True,'automatic_promotion':False,'patient_level_output_emitted':False}


def validate_result(a,p):
    keys={'schema','status','metrics','training_eligibility_lower_bounds_20','control_replay_matches_prior',
          'adaptive_development','intervals_adjusted_for_experiment_selection','prior_negative_result_preserved',
          'automatic_promotion','patient_level_output_emitted'}
    if set(a)!=keys or a['schema']!='bran-joint-lab-comparison-aggregate-v1' or a['status']!='completed': raise ValueError('result schema mismatch')
    if (type(a['control_replay_matches_prior']) is not bool or a['adaptive_development'] is not True or a['prior_negative_result_preserved'] is not True
        or any(a[k] is not False for k in ('intervals_adjusted_for_experiment_selection','automatic_promotion','patient_level_output_emitted'))):
        raise ValueError('result claim flags invalid')
    prior.validate_result(a['metrics'],p)
    coverage=a['training_eligibility_lower_bounds_20']
    if set(coverage)!=set(SOURCE_SPECS): raise ValueError('coverage sources invalid')
    for tasks in coverage.values():
        if set(tasks)!={'whole_cbc','partial_cbc'}: raise ValueError('coverage tasks invalid')
        for counts in tasks.values():
            if set(counts)!={'snapshots','source_local_people'} or any(v is not None and (type(v) is not int or v<20 or v%20) for v in counts.values()):
                raise ValueError('coverage counts invalid')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare-protocol',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.prepare_protocol==args.run: parser.error('choose exactly one operation')
    ok=False;owned=False;phase='protocol';started=time.monotonic()
    with _quiet():
        try:
            if args.prepare_protocol: exclusive_json(PROTOCOL,prepare());ok=True
            else:
                p=json.loads(PROTOCOL.read_text());validate_protocol(p)
                OUT.mkdir();owned=True;phase='source_authentication'
                for source in SOURCE_SPECS: source_receipt(source,verify_raw=True)
                phase='fit_and_evaluate';a=fit_and_score(p)
                phase='post_fit_authentication';validate_protocol(p)
                for source in SOURCE_SPECS: source_receipt(source,verify_raw=True)
                validate_result(a,p);exclusive_json(OUT/'aggregate.json',a)
                exclusive_json(OUT/'manifest.json',{'protocol_sha256':sha(PROTOCOL),'aggregate_sha256':sha(OUT/'aggregate.json'),
                    'checkpoint_sha256':{'fold'+str(f):sha(PRIVATE/('fold'+str(f)+'.pt')) for f in range(5)},
                    'elapsed_seconds':round(time.monotonic()-started,1),'patient_level_output_emitted':False})
                prior.base._atomic_progress(OUT/'progress.json','completed');ok=True
        except Exception as error:
            allowed={ValueError:'ValueError',TypeError:'TypeError',KeyError:'KeyError',RuntimeError:'RuntimeError',
                     MemoryError:'MemoryError',ImportError:'ImportError',FileNotFoundError:'FileNotFoundError',OSError:'OSError'}
            if owned: exclusive_json(OUT/'failure.json',{'status':'joint_comparison_execution_failed','phase':phase,
                'error_class':allowed.get(type(error),'other_execution_error'),'patient_level_output_emitted':False})
    print(json.dumps({'status':('joint_protocol_prepared' if args.prepare_protocol else 'joint_comparison_completed') if ok else 'joint_comparison_execution_failed','patient_level_output_emitted':False}))
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
