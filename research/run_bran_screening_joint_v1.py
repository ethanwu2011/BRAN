"""One frozen joint-objective comparison. Private arrays never cross FD-quiet boundary."""
import argparse
import json
import math
import os
import time
from pathlib import Path
import numpy as np
from bran_clinical_dictionary_binding_v1 import _quiet
from bran_clinical_semantics_v1 import CBC_FIELDS
from run_bran_source_linkage_audit_v1 import sha, exclusive_json
import bran_september_push_io_v1 as io
import run_bran_overnight_diagnostic_v1 as base
import run_bran_anchor_ablation_v2 as lineage
import bran_external_cbc_evaluation_v1 as ev

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'BRAN_SCREENING_JOINT_PROTOCOL_V1.json'
OUT=ROOT/'BRAN_SCREENING_JOINT_V1'
PRIVATE=ROOT/'private_artifacts/bran_screening_joint_v1'
SCREEN_ARMS=ev.SCREEN_ARMS+('lineage_both',)
SCREEN_COMPARISONS=ev.SCREEN_COMPARISONS+(('candidate_both','lineage_both'),)
CBC_ARMS=('control','candidate','raw','lineage')
PATTERNS=('single_target_hidden','whole_cbc_hidden','all_clinical_hidden')
PRIORITY=('hemoglobin','plt','wbc')
PARAMETERS={'steps_per_arm_fold':1500,'batch_size':96,'seed_base':92401,'learning_rate':.0001,
    'weight_decay':.0001,'gradient_clip_norm':5.,'state_width':192,'screening_outputs':26,
    'candidate_loss_weights':{'disease':1.,'whole_cbc':.5,'clinical_preservation':.1},
    'control_loss_weights':{'disease':0.,'whole_cbc':0.,'clinical_preservation':0.},
    'screening_route_cycle':['both','both','both','clinical','retinal'],
    'disease_positive_weight':'proper-training negative/positive clipped1..10; per-endpoint maskedmean',
    'cbc':'wholeCBC values+indicators removed; normalized observed targets; rowmeanSmoothL1 from retained192to9head',
    'preservation':'frozen initial clinicalonly teacher; same visibleclinical mask+age; train SDmax1; private64 MSE',
    'evaluation':'all26conditions fixed standardizedLogisticC1 max_iter5000; all9CBC fixed standardizedRidge1',
    'mask_patterns':list(PATTERNS),'screen_arms':list(SCREEN_ARMS),'cbc_arms':list(CBC_ARMS),
    'bootstrap_draws':1000,'bootstrap_seed':91501,'minimum_release_support':20,
    'automatic_promotion':False,'clinical_use_permitted':False,'official_test_used':False,
    'adaptive_development':True,'calibrated_uncertainty_claim':False,'novel_subtype_claim':False}
CODE=('run_bran_screening_joint_v1.py','bran_september_push_io_v1.py',
      'bran_screening_joint_kernel_v1.py','test_bran_screening_joint_kernel_v1.py',
      'test_run_bran_screening_joint_v1.py','bran_clinical_preservation_loss_draft_v1.py',
      'test_bran_clinical_preservation_loss_draft_v1.py','BRAN_SCREENING_JOINT_DESIGN_V1.md',
      'audit_bran_external_cbc_comparison_v1.py')


def prepare():
    return {'schema':'bran-screening-joint-protocol-v1','status':'frozen_before_execution',
            'parameters':PARAMETERS,'source':io.source_receipt(),'code_sha256':io.code_closure(CODE),'runtime':io.runtime()}


def validate_protocol(p):
    if p!=prepare():raise ValueError('protocol_changed')


def mask_inputs(c,cm,slots,pattern,target):
    if pattern not in PATTERNS or len(set(slots))!=9 or any(i<0 or i>=48 for i in slots) or target not in slots:
        raise ValueError('mask_contract_invalid')
    if c.ndim!=2 or c.shape[1]!=59 or cm.shape!=c.shape or cm.dtype!=bool:
        raise ValueError('mask_shape_invalid')
    v=c.copy();m=cm.copy()
    hide=[target] if pattern=='single_target_hidden' else list(slots) if pattern=='whole_cbc_hidden' else list(range(59))
    v[:,hide]=0.;m[:,hide]=False
    return v,m


def summarize_screen(pred,labels,observed,folds,counts,names):
    endpoints={};points={a:[] for a in SCREEN_ARMS};draws={a:[] for a in SCREEN_ARMS}
    for name in names:
        y,o=labels[name],np.asarray(observed[name],bool)
        if sum(o&(y==1))<20 or sum(o&(y==0))<20:raise ValueError('screen_support_invalid')
        ps,ds={},{}
        for a in SCREEN_ARMS:
            if not np.isfinite(pred[name][a][o]).all():raise ValueError('screen_predictions_invalid')
            ps[a]=base.fold_weighted_auc(y,pred[name][a],o,folds)
            ds[a]=base._weighted_auc_draws(y,pred[name][a],o,folds,counts)
            points[a].append(ps[a]);draws[a].append(ds[a])
        endpoints[name]={'arms':{a:{'auroc':ps[a],'ci95':ev._ci(ds[a])} for a in SCREEN_ARMS},
            'paired_deltas':{a+'-'+b:{'auroc_delta':ps[a]-ps[b],'ci95':ev._ci(ds[a]-ds[b])} for a,b in SCREEN_COMPARISONS}}
    ps={a:float(np.mean(points[a])) for a in SCREEN_ARMS};ds={a:np.mean(draws[a],axis=0) for a in SCREEN_ARMS}
    return {'endpoints':endpoints,'macro_auroc':ps,'macro_paired_deltas':{
        a+'-'+b:{'auroc_delta':ps[a]-ps[b],'ci95':ev._ci(ds[a]-ds[b])} for a,b in SCREEN_COMPARISONS}}


def summarize_cbc(y,observed,pred,counts):
    result={}
    for j,field in enumerate(CBC_FIELDS):
        valid=observed[:,j]&np.isfinite(y[:,j])
        if valid.sum()<20:
            result[field]={'status':'unsupported'};continue
        if any(not np.isfinite(pred[a][valid,j]).all() for a in CBC_ARMS):raise ValueError('completion_predictions_invalid')
        weights=counts[:,valid].astype(float);denom=weights.sum(1)
        if np.any(denom<=0):raise ValueError('bootstrap_support_invalid')
        errors={a:np.abs(pred[a][valid,j]-y[valid,j]) for a in CBC_ARMS}
        result[field]={'status':'complete','observed_count_lower_bound_20':int(valid.sum()//20*20),
            'arms':{a:{'mae':float(errors[a].mean()),'mse':float(np.square(errors[a]).mean())} for a in CBC_ARMS},
            'paired_deltas':{'candidate-'+a:{'mae_delta':float((errors['candidate']-errors[a]).mean()),
                'ci95':ev._ci(weights@(errors['candidate']-errors[a])/denom)} for a in ('control','raw','lineage')}}
    return result


def decision(screen,cbc):
    deltas=screen['macro_paired_deltas']
    priority=PRIORITY
    if not set(priority)<=set(CBC_FIELDS):raise ValueError('priority_field_contract_changed')
    parts={
        'auc_lower_bound_positive_vs_continued':deltas['candidate_both-control_both']['ci95'][0]>0,
        'auc_lower_bound_positive_vs_raw_clinical':deltas['candidate_both-raw_clinical']['ci95'][0]>0,
        'auc_lower_bound_positive_vs_initial_lineage':deltas['candidate_both-lineage_both']['ci95'][0]>0,
        'auc_point_positive_vs_own_clinical':deltas['candidate_both-candidate_clinical']['auroc_delta']>0,
        'auc_point_positive_vs_own_retinal':deltas['candidate_both-candidate_retinal']['auroc_delta']>0,
        'priority_whole_cbc_point_nonworse':all(cbc['whole_cbc_hidden'][f]['status']=='complete' and
            all(cbc['whole_cbc_hidden'][f]['paired_deltas']['candidate-'+a]['mae_delta']<=0 for a in ('control','lineage')) for f in priority)}
    return {'criteria':parts,'all_pass':all(parts.values()),'automatic_promotion':False,
            'scope':'engineering_advancement_gate_not_multiplicity_adjusted_clinical_efficacy'}


def run(p):
    import torch
    from bran_screening_joint_kernel_v1 import adapt
    ctx,folds,c0,cm0,eligible,r0,rm,ages,names=io.load_context()
    endpoints=p['source']['endpoint_names'];slots=tuple(names.index(f) for f in CBC_FIELDS)
    labels=np.column_stack([ctx['labels_by_source'][e] for e in endpoints]);label_mask=np.column_stack([ctx['observed_by_source'][e] for e in endpoints]).astype(bool)
    pred={e:{a:np.full(len(folds),np.nan) for a in SCREEN_ARMS} for e in endpoints}
    completion={pattern:{a:np.full((len(folds),9),np.nan) for a in CBC_ARMS} for pattern in PATTERNS}
    completion_observed={pattern:(cm0[:,slots]&eligible[:,slots]).copy() for pattern in PATTERNS}
    retained=np.full((len(folds),9),np.nan);checkpoints={}
    PRIVATE.mkdir(parents=True,mode=0o700)
    for f in range(5):
        tr,te=np.flatnonzero(folds!=f),np.flatnonzero(folds==f)
        transform=base.FoldTransform(c0,cm0,eligible,r0,rm,ages,tr)
        c,cm,r,age=transform.apply(c0,cm0,eligible,r0,rm,ages)
        initial=io.load_control(f,transform,p['source'])
        models={'lineage':initial}
        for name,iscandidate in (('control',False),('candidate',True)):
            base._atomic_progress(OUT/'progress.json','adapt_'+name,f)
            models[name]=adapt(initial,c,cm,r,rm,age,labels,label_mask,tr,slots,seed=92401+f,steps=1500,batch_size=96,candidate=iscandidate)
            models[name].eval()
        arms={'raw_clinical':np.c_[c,cm,age],'raw_retinal':np.c_[r,rm,age],'raw_concat':np.c_[c,cm,r,rm,age]}
        for version,model in models.items():
            for route,z in lineage._state_routes(model,c,cm,r,rm,age).items():
                if version!='lineage' or route=='both':arms[version+'_'+route]=np.c_[z,age]
        base._atomic_progress(OUT/'progress.json','fixed_screening',f)
        for e in endpoints:
            for arm in SCREEN_ARMS:
                pred[e][arm][te]=ev.fixed_screen_probe(arms[arm],ctx['labels_by_source'][e],ctx['observed_by_source'][e],tr,te)
        base._atomic_progress(OUT/'progress.json','completion_routes',f)
        for pattern in PATTERNS:
            cached=None
            for j,slot in enumerate(slots):
                if pattern=='single_target_hidden' or cached is None:
                    hidden,hm=mask_inputs(c,cm,slots,pattern,slot)
                    physiology=hm.any(1)|rm
                    inputs={'raw':np.c_[hidden,hm,r,rm,age]}
                    for version,model in models.items():
                        z=lineage._state_routes(model,hidden,hm,r,rm,age)['both']
                        inputs[version]=np.c_[z,age]
                    cached=(inputs,physiology)
                    if pattern=='whole_cbc_hidden':
                        z=inputs['candidate'][:,:192]
                        with torch.no_grad():value=models['candidate'].cbc_joint_head(torch.tensor(z,dtype=torch.float32)).numpy()
                        retained[te]=value[te]*transform.clinical_iqr[list(slots)]+transform.clinical_median[list(slots)]
                inputs,physiology=cached
                completion_observed[pattern][:,j]&=physiology
                target_observed=cm[:,slot]&physiology
                for version in CBC_ARMS:
                    completion[pattern][version][te,j]=ev.fixed_cbc_probe(inputs[version],c0[:,slot],target_observed,tr,te)
        path=PRIVATE/('fold'+str(f)+'.pt')
        bundle={'control':models['control'].state_dict(),'candidate':models['candidate'].state_dict(),
            **{k:getattr(transform,k) for k in ('clinical_median','clinical_iqr','retinal_mean','retinal_scale','age_mean','age_scale')},
            'protocol_sha256':sha(PROTOCOL),'initial_checkpoint_sha256':p['source']['checkpoint_sha256']['fold'+str(f)],
            'endpoint_names':endpoints,'cbc_fields':list(CBC_FIELDS)}
        with path.open('xb') as handle:torch.save(bundle,handle)
        os.chmod(path,0o600);checkpoints['fold'+str(f)]=sha(path)
    base._atomic_progress(OUT/'progress.json','paired_uncertainty')
    counts=ev.paired_counts(folds,draws=1000,seed=91501)
    screen=summarize_screen(pred,ctx['labels_by_source'],ctx['observed_by_source'],folds,counts,endpoints)
    cbc={pat:summarize_cbc(c0[:,slots],completion_observed[pat],completion[pat],counts) for pat in PATTERNS}
    direct={}
    for j,field in enumerate(CBC_FIELDS):
        valid=completion_observed['whole_cbc_hidden'][:,j]&np.isfinite(c0[:,slots[j]])
        if valid.sum()<20:direct[field]={'status':'unsupported'};continue
        errors=retained[valid,j]-c0[valid,slots[j]]
        if not np.isfinite(errors).all():raise ValueError('retained_head_nonfinite')
        direct[field]={'status':'complete','mae':float(np.abs(errors).mean()),'mse':float(np.square(errors).mean())}
    a={'schema':'bran-screening-joint-aggregate-v1','status':'completed','paired_people':1928,'recorded_conditions':26,
       'screening':screen,'completion':cbc,'retained_cbc_head_whole_panel':direct,'research_advancement':decision(screen,cbc),
       'patient_level_output_emitted':False,'automatic_promotion':False,'official_test_used':False,
       'adaptive_development':True,'calibrated_uncertainty_claim':False,'novel_subtype_claim':False}
    return a,checkpoints


def validate_result(a,p):
    keys={'schema','status','paired_people','recorded_conditions','screening','completion','retained_cbc_head_whole_panel',
          'research_advancement','patient_level_output_emitted','automatic_promotion','official_test_used','adaptive_development',
          'calibrated_uncertainty_claim','novel_subtype_claim'}
    if set(a)!=keys or a['schema']!='bran-screening-joint-aggregate-v1' or a['status']!='completed' or a['paired_people']!=1928 or a['recorded_conditions']!=26:
        raise ValueError('aggregate_identity_invalid')
    if a['adaptive_development'] is not True or any(a[k] is not False for k in ('patient_level_output_emitted','automatic_promotion','official_test_used','calibrated_uncertainty_claim','novel_subtype_claim')):
        raise ValueError('aggregate_claim_invalid')
    def number(v,lo=-math.inf,hi=math.inf):
        if type(v) not in (float,int) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError('aggregate_number_invalid')
    def interval(v,lo=-math.inf,hi=math.inf):
        if not isinstance(v,list) or len(v)!=2:raise ValueError('aggregate_interval_invalid')
        for x in v:number(x,lo,hi)
        if v[0]>v[1]:raise ValueError('aggregate_interval_order')
    def contrasts(v,names,key,bound=False):
        if set(v)!=set(names):raise ValueError('aggregate_contrast_keys')
        for row in v.values():
            if set(row)!={key,'ci95'}:raise ValueError('aggregate_contrast_schema')
            number(row[key],-1 if bound else -math.inf,1 if bound else math.inf)
            interval(row['ci95'],-1 if bound else -math.inf,1 if bound else math.inf)
    s=a['screening'];contrast_names=[x+'-'+y for x,y in SCREEN_COMPARISONS]
    if set(s)!={'endpoints','macro_auroc','macro_paired_deltas'} or set(s['endpoints'])!=set(p['source']['endpoint_names']) or set(s['macro_auroc'])!=set(SCREEN_ARMS):raise ValueError('screen_schema_invalid')
    for v in s['macro_auroc'].values():number(v,0,1)
    contrasts(s['macro_paired_deltas'],contrast_names,'auroc_delta',True)
    for row in s['endpoints'].values():
        if set(row)!={'arms','paired_deltas'} or set(row['arms'])!=set(SCREEN_ARMS):raise ValueError('screen_endpoint_keys')
        for v in row['arms'].values():
            if set(v)!={'auroc','ci95'}:raise ValueError('screen_metric_keys')
            number(v['auroc'],0,1);interval(v['ci95'],0,1)
        contrasts(row['paired_deltas'],contrast_names,'auroc_delta',True)
    if set(a['completion'])!=set(PATTERNS) or set(a['retained_cbc_head_whole_panel'])!=set(CBC_FIELDS):raise ValueError('completion_schema_invalid')
    for fields in a['completion'].values():
        if set(fields)!=set(CBC_FIELDS):raise ValueError('completion_field_keys')
        for row in fields.values():
            if row=={'status':'unsupported'}:continue
            if set(row)!={'status','observed_count_lower_bound_20','arms','paired_deltas'} or row['status']!='complete' or set(row['arms'])!=set(CBC_ARMS):raise ValueError('completion_field_schema')
            n=row['observed_count_lower_bound_20']
            if type(n) is not int or n<20 or n%20:raise ValueError('completion_support_privacy')
            for v in row['arms'].values():
                if set(v)!={'mae','mse'}:raise ValueError('completion_metric_schema')
                number(v['mae'],0);number(v['mse'],0)
                if v['mse']+1e-10<v['mae']**2:raise ValueError('completion_moment_invalid')
            contrasts(row['paired_deltas'],['candidate-'+x for x in ('control','raw','lineage')],'mae_delta')
    for row in a['retained_cbc_head_whole_panel'].values():
        if row=={'status':'unsupported'}:continue
        if set(row)!={'status','mae','mse'} or row['status']!='complete':raise ValueError('retained_head_schema')
        number(row['mae'],0);number(row['mse'],0)
    if a['research_advancement']!=decision(s,a['completion']):raise ValueError('advancement_arithmetic_invalid')


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
                OUT.mkdir();owned=True;phase='fit_evaluate';a,checkpoints=run(p)
                phase='terminal_validation';validate_protocol(p);validate_result(a,p)
                exclusive_json(OUT/'aggregate.json',a)
                exclusive_json(OUT/'manifest.json',{'protocol_sha256':sha(PROTOCOL),'aggregate_sha256':sha(OUT/'aggregate.json'),
                    'checkpoint_sha256':checkpoints,'elapsed_seconds':round(time.monotonic()-start,1),'patient_level_output_emitted':False})
                base._atomic_completed(OUT/'progress.json');ok=True
        except Exception as e:
            if owned:exclusive_json(OUT/'failure.json',{'status':'execution_failed','phase':phase,
                'error_class':type(e).__name__ if type(e) in (ValueError,TypeError,KeyError,RuntimeError,OSError,ImportError) else 'other_execution_error',
                'patient_level_output_emitted':False})
    print(json.dumps({'status':('protocol_prepared' if args.prepare else 'joint_comparison_completed') if ok else 'execution_failed','patient_level_output_emitted':False}))
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
