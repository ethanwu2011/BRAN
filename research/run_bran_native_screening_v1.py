"""Secondary retained-head diagnosis; no encoder/head adaptation or promotion."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import sha,exclusive_json
import run_bran_screening_joint_v1 as source
import audit_bran_september_push_v1 as auth
import bran_native_screening_kernel_v1 as kernel

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'BRAN_NATIVE_SCREENING_V1'
PROTOCOL=ROOT/'BRAN_NATIVE_SCREENING_PROTOCOL_V1.json'
SOURCE_AGGREGATE='f1e9e9f769a2e238dde0aff6241bbb0d689ec351dca845588a2ae8dd6305780f'
SOURCE_AUDIT='8c6e7659ee7af9d15a271f950b35fcaf1ed86d1e6578bf4a0055533eb2a16951'
ROUTES=('both','clinical','retinal')
ARMS=tuple('native_'+r for r in ROUTES)+tuple('probe_'+r for r in ROUTES)+('control_both','raw_clinical','raw_concat')
CONTRASTS=(('native_both','probe_both'),('native_both','control_both'),('native_both','raw_clinical'),
           ('native_both','native_clinical'),('native_both','native_retinal'),('native_both','raw_concat'))
PARAMETERS={'secondary_diagnostic':True,'primary_joint_failure_unchanged':True,'encoder_or_native_head_training':False,
    'retained_head':'trained_192_to26_linear_head_sigmoid_no_posthoc_calibration','readout_reference':'standardized_logistic_C1_max5000',
    'outer_folds':5,'arms':list(ARMS),'bootstrap_draws':1000,'bootstrap_seed':91501,'minimum_per_class':20,
    'common_rows':'observed_endpoint_and_all_routes_nonabstaining','macro_requires_all26':True,
    'clinical_use':False,'candidate_promoted':False,'adaptive_development':True,'patient_level_output_permitted':False}
CODE=('run_bran_native_screening_v1.py','test_run_bran_native_screening_v1.py','bran_native_screening_kernel_v1.py',
      'test_bran_native_screening_kernel_v1.py','audit_bran_september_push_v1.py')


def prepare():
    if sha(source.PROTOCOL)!=auth.PINS['joint'] or sha(source.OUT/'aggregate.json')!=SOURCE_AGGREGATE or sha(ROOT/'BRAN_SCREENING_JOINT_AUDIT_V1/audit.json')!=SOURCE_AUDIT:raise ValueError('native_source_changed')
    p=json.loads(source.PROTOCOL.read_text());source.validate_protocol(p)
    if (source.OUT/'failure.json').exists():raise ValueError('native_source_terminal_conflict')
    m=json.loads((source.OUT/'manifest.json').read_text());auth.validate_manifest(m,'joint')
    for name,digest in m['checkpoint_sha256'].items():
        path=source.PRIVATE/(name+'.pt')
        if sha(path)!=digest or path.stat().st_mode&0o777!=0o600:raise ValueError('native_checkpoint_changed')
    return {'schema':'bran-native-screening-protocol-v1','status':'frozen_before_execution','parameters':PARAMETERS,
        'source_protocol_sha256':auth.PINS['joint'],'source_aggregate_sha256':SOURCE_AGGREGATE,'source_audit_sha256':SOURCE_AUDIT,
        'checkpoint_sha256':m['checkpoint_sha256'],'source':p['source'],
        'code_sha256':{**p['code_sha256'],**source.io.code_closure(CODE)},'runtime':source.io.runtime()}


def validate_protocol(p):
    if p!=prepare():raise ValueError('native_protocol_changed')


def summarize(pred,labels,observed,folds,names):
    counts=source.ev.paired_counts(folds,draws=1000,seed=91501)
    points={a:[] for a in ARMS};draws={a:[] for a in ARMS};endpoints={}
    for e in names:
        y=np.asarray(labels[e]);o=np.asarray(observed[e],bool).copy()
        for arm in ARMS:o&=np.isfinite(pred[e][arm])
        if np.sum(o&(y==1))<20 or np.sum(o&(y==0))<20:endpoints[e]={'status':'unsupported'};continue
        ps={a:source.base.fold_weighted_auc(y,pred[e][a],o,folds) for a in ARMS}
        ds={a:source.base._weighted_auc_draws(y,pred[e][a],o,folds,counts) for a in ARMS}
        for a in ARMS:points[a].append(ps[a]);draws[a].append(ds[a])
        endpoints[e]={'status':'complete','arms':{a:{'auroc':ps[a],'ci95':source.ev._ci(ds[a])} for a in ARMS},
            'contrasts':{a+'-'+b:{'auroc_delta':ps[a]-ps[b],'ci95':source.ev._ci(ds[a]-ds[b])} for a,b in CONTRASTS}}
    macro=None
    if all(v['status']=='complete' for v in endpoints.values()):
        ps={a:float(np.mean(points[a])) for a in ARMS};ds={a:np.mean(draws[a],axis=0) for a in ARMS}
        macro={'arms':{a:{'auroc':ps[a],'ci95':source.ev._ci(ds[a])} for a in ARMS},
               'contrasts':{a+'-'+b:{'auroc_delta':ps[a]-ps[b],'ci95':source.ev._ci(ds[a]-ds[b])} for a,b in CONTRASTS}}
    return {'endpoints':endpoints,'macro':macro}


def run(p):
    import torch
    ctx,folds,c0,cm0,eligible,r0,rm,ages,names=source.io.load_context()
    endpoints=p['source']['endpoint_names'];pred={e:{a:np.full(len(folds),np.nan) for a in ARMS} for e in endpoints}
    original_protocol=json.loads(source.PROTOCOL.read_text())
    for f in range(5):
        source.base._atomic_progress(OUT/'progress.json','native_and_fixed_reference',f)
        tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f)
        transform=source.base.FoldTransform(c0,cm0,eligible,r0,rm,ages,tr);c,cm,r,age=transform.apply(c0,cm0,eligible,r0,rm,ages)
        path=source.PRIVATE/('fold'+str(f)+'.pt')
        if sha(path)!=p['checkpoint_sha256']['fold'+str(f)]:raise ValueError('native_checkpoint_invalid')
        b=torch.load(path,map_location='cpu',weights_only=False)
        candidate=auth.validate_joint_bundle(b,original_protocol,f);candidate.load_state_dict(b['candidate'],strict=True);candidate.eval()
        control=auth.validate_joint_bundle(b,original_protocol,f);control.load_state_dict(b['control'],strict=True);control.eval()
        for k in ('clinical_median','clinical_iqr','retinal_mean','retinal_scale','age_mean','age_scale'):
            if not np.array_equal(b[k],getattr(transform,k)):raise ValueError('native_normalizer_invalid')
        native=kernel.predict_native(candidate,c,cm,r,rm,age)
        designs={'probe_'+route:np.c_[z,age] for route,z in source.lineage._state_routes(candidate,c,cm,r,rm,age).items()}
        designs['control_both']=np.c_[source.lineage._state_routes(control,c,cm,r,rm,age)['both'],age]
        designs.update(raw_clinical=np.c_[c,cm,age],raw_concat=np.c_[c,cm,r,rm,age])
        for j,e in enumerate(endpoints):
            for route in ROUTES:pred[e]['native_'+route][te]=native[route][te,j]
            for arm,x in designs.items():pred[e][arm][te]=source.ev.fixed_screen_probe(x,ctx['labels_by_source'][e],ctx['observed_by_source'][e],tr,te)
    return {'schema':'bran-native-screening-aggregate-v1','status':'completed','results':summarize(pred,ctx['labels_by_source'],ctx['observed_by_source'],folds,endpoints),
        'paired_people':1928,'recorded_conditions':26,'primary_joint_failure_unchanged':True,'secondary_diagnostic':True,
        'candidate_promoted':False,'clinical_use':False,'patient_level_output_emitted':False,'official_test_used':False,'adaptive_development':True}


def validate_metric(v,delta=False):
    key='auroc_delta' if delta else 'auroc'
    if not isinstance(v,dict) or set(v)!={key,'ci95'} or type(v[key]) not in (float,int) or not np.isfinite(v[key]) or not (-1 if delta else 0)<=v[key]<=1:raise ValueError('native_metric_invalid')
    if not isinstance(v['ci95'],list) or len(v['ci95'])!=2 or any(type(x) not in (float,int) or not np.isfinite(x) or not (-1 if delta else 0)<=x<=1 for x in v['ci95']) or v['ci95'][0]>v['ci95'][1]:raise ValueError('native_interval_invalid')


def validate_result(a,p):
    keys={'schema','status','results','paired_people','recorded_conditions','primary_joint_failure_unchanged','secondary_diagnostic','candidate_promoted','clinical_use','patient_level_output_emitted','official_test_used','adaptive_development'}
    if set(a)!=keys or a['schema']!='bran-native-screening-aggregate-v1' or a['status']!='completed' or a['paired_people']!=1928 or a['recorded_conditions']!=26:raise ValueError('native_aggregate_invalid')
    if any(a[k] is not True for k in ('primary_joint_failure_unchanged','secondary_diagnostic','adaptive_development')) or any(a[k] is not False for k in ('candidate_promoted','clinical_use','patient_level_output_emitted','official_test_used')):raise ValueError('native_claim_invalid')
    res=a['results']
    if set(res)!={'endpoints','macro'} or set(res['endpoints'])!=set(p['source']['endpoint_names']):raise ValueError('native_results_invalid')
    complete=True;rows=[]
    for row in res['endpoints'].values():
        if row=={'status':'unsupported'}:complete=False;continue
        if set(row)!={'status','arms','contrasts'} or row['status']!='complete':raise ValueError('native_endpoint_invalid')
        rows.append({k:row[k] for k in ('arms','contrasts')})
    if complete:
        if res['macro'] is None:raise ValueError('native_macro_missing')
        rows.append(res['macro'])
    elif res['macro'] is not None:raise ValueError('native_partial_macro_invalid')
    for row in rows:
        if set(row)!={'arms','contrasts'} or set(row['arms'])!=set(ARMS) or set(row['contrasts'])!={a+'-'+b for a,b in CONTRASTS}:raise ValueError('native_arm_invalid')
        for value in row['arms'].values():validate_metric(value)
        for value in row['contrasts'].values():validate_metric(value,True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--protocol-sha256');args=parser.parse_args()
    if args.prepare==args.run:parser.error('choose_one_operation')
    ok=False;owned=False;phase='protocol';start=time.monotonic()
    with _quiet():
        try:
            if args.prepare:exclusive_json(PROTOCOL,prepare());ok=True
            else:
                if not args.protocol_sha256 or sha(PROTOCOL)!=args.protocol_sha256:raise ValueError('native_protocol_hash_invalid')
                p=json.loads(PROTOCOL.read_text());validate_protocol(p);OUT.mkdir();owned=True;phase='inference_and_references';a=run(p)
                phase='terminal_validation';validate_protocol(p);validate_result(a,p);exclusive_json(OUT/'aggregate.json',a)
                exclusive_json(OUT/'manifest.json',{'protocol_sha256':sha(PROTOCOL),'aggregate_sha256':sha(OUT/'aggregate.json'),'elapsed_seconds':round(time.monotonic()-start,1),'patient_level_output_emitted':False})
                source.base._atomic_completed(OUT/'progress.json');ok=True
        except Exception as e:
            if owned:exclusive_json(OUT/'failure.json',{'status':'execution_failed','phase':phase,'error_class':type(e).__name__ if type(e) in (ValueError,TypeError,KeyError,RuntimeError,OSError,ImportError) else 'other_execution_error','patient_level_output_emitted':False})
    print(json.dumps({'status':('protocol_prepared' if args.prepare else 'native_screening_completed') if ok else 'execution_failed','patient_level_output_emitted':False}));return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
