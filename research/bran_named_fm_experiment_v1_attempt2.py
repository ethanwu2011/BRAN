"""Attempt2: retain canonical patients with no retinal evidence using zero vectors.

Attempt1 is preserved: its all-patients-have-retina assumption failed before training.
All cohort/outcome/fold/readout/inference settings remain unchanged.

No model, image, patient array or bootstrap draw is written. Existing results and
their source closure remain immutable. All real-data work is descriptor-silenced.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone
import numpy as np
import bran_innovation_experiment_v1_attempt2 as prior
import bran_named_fm_auc_v1 as probe
from bran_named_fm_reference_v1 import predict_references, replay_references

ROOT=Path(__file__).resolve().parent
FIRST_PROTOCOL='BRAN_NAMED_FM_PROTOCOL_V1.json'
FIRST_PROTOCOL_SHA='fc6aba48f24ba213e3e41352c3ee575acb34c019b76f19004b0302580ed44f2b'
FIRST_FAILURE='validation_results/BRAN_NAMED_FM_V1/FAILURE.json'
FIRST_FAILURE_SHA='97c354e5b73565f4419ee4b40fa0f08caf562bdb30af17d44de18abc4669bdf0'
SCHEMA='bran-named-fm-experiment-v1-attempt2'
PROTOCOL='BRAN_NAMED_FM_PROTOCOL_V1_ATTEMPT2.json'
OUTDIR='validation_results/BRAN_NAMED_FM_V1_ATTEMPT2'
PATHS={k:OUTDIR+'/'+v for k,v in {'success':'SUCCESS.json','failure':'FAILURE.json','progress':'progress.json','lock':'run.lock'}.items()}
PRIOR_PROTOCOL='BRAN_INNOVATION_READOUT_PROTOCOL_V1_ATTEMPT2.json'
PRIOR_PROTOCOL_SHA='961ec1d2283ac6ea8275ff275903f63df6563444604dc89b8a9051e3625743e5'
PRIOR_RESULT='validation_results/BRAN_INNOVATION_READOUT_V1_ATTEMPT2/SUCCESS.json'
PRIOR_RESULT_SHA='5df5b384fd345b6e242399e1a46c9b65aa6827d3550f231d8a559654c5ee3894'
ARTIFACT_PROTOCOLS=('PATIENT_ATLAS_V5_RETFOUND_GREEN_PROTOCOL_V1.json','PATIENT_ATLAS_V5_VISIONFM_PROTOCOL_V1.json','PATIENT_ATLAS_V5_FOUNDATION_COMPARATORS_PROTOCOL_V1.json')
NEW_FILES={'bran_named_fm_experiment_v1.py','test_bran_named_fm_experiment_v1.py',
           'bran_named_fm_reference_v1.py','test_bran_named_fm_reference_v1.py',
           'bran_named_fm_auc_v1.py','test_bran_named_fm_auc_v1.py',
           'bran_named_fm_extraction_v1.py','test_bran_named_fm_extraction_v1.py',
           'patient_atlas_labrador_worker.py',*ARTIFACT_PROTOCOLS,
           'bran_named_fm_experiment_v1_attempt2.py','test_bran_named_fm_experiment_v1_attempt2.py',
           'bran_named_fm_extraction_v1_attempt2.py','test_bran_named_fm_extraction_v1_attempt2.py',
           'diagnose_bran_named_fm_selection_v1.py',FIRST_PROTOCOL}
PARAMETERS={'arms':list(probe.ARMS),'dimensions':probe.WIDTHS,'encoder_steps':1500,'encoder_seed':1701,
 'reference':'innovation_selected_both_and_legacy_v2_single_views_192_plus_age',
 'reference_promotion':False,'all_arms_include_age':True,'outer_folds':5,'inner_folds':5,
 'head_C':[.01,.1,1.,10.],'head_selection':'inner_auroc_then_logloss_then_smaller_C',
 'reference_combined_profiles':[list(x) for x in prior.base.PROFILES],
 'retinal_device':'cpu','torch_threads':2,'retinal_batch_size':16,
 'retinal_pooling':'same_manifest_selected_CFP_unweighted_mean_all_views',
 'retinal_order':'retfound_green_then_visionfm_last4_then_dinov3_generic',
 'labrador_inputs':'observed_eligible_canonical_blood_indices_below_38_exact_released_map',
 'bootstrap_draws':1000,'bootstrap_seed':91501,'minimum_valid_draws':950,
 'metric':'observed_count_weighted_outer_fold_auroc_equal_endpoint_macro',
 'fm_macro_family':list(probe.WIDTHS),'fm_family_lower_percentile':1.25,
 'inference':'fixed_fit_reused_development_marginal_endpoint_intervals_four_FM_macro_Bonferroni',
 'reference_canary_tolerance':1e-8,'no_clinical_completion_refit':True,
 'retinal_missingness':'retain_all_canonical_patients_zero_vector_when_no_retinal_evidence'}
PRIVACY={'patient_processing_local_only':True,'patient_rows_ids_images_predictions_embeddings_states_draws_serialized':False,
 'model_weights_serialized':False,'official_test_loaded':False,'hosted_inference_used':False,
 'original_nature_retfound_included':False,'longitudinal_ehr_fm_included':False,
 'external_validation_claimed':False,'disease_subtypes_claimed':False,'retinal_generation_claimed':False}
PHASES={'validated','context','selection','reference_training','reference_heads','reference_replay',
        'retinal_extraction','labrador_extraction','fm_heads','aggregate','writing','completed','protocol'}
MODULES=('numpy','scipy','pandas','scikit-learn','torch','torchvision','timm','safetensors','pydicom','Pillow')
require=prior.base.require
exact_keys=prior.base.exact_keys
num=prior.base.num
interval=prior.base.interval
write_x=prior.base.write_x

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1<<20):h.update(chunk)
    return h.hexdigest()

def old_protocol(root):
    require(sha(root/FIRST_PROTOCOL)==FIRST_PROTOCOL_SHA and sha(root/FIRST_FAILURE)==FIRST_FAILURE_SHA)
    require(not(root/'validation_results/BRAN_NAMED_FM_V1/SUCCESS.json').exists())
    require(sha(root/PRIOR_PROTOCOL)==PRIOR_PROTOCOL_SHA and sha(root/PRIOR_RESULT)==PRIOR_RESULT_SHA)
    return prior.validate_protocol(root)

def artifacts(root):
    a=[json.loads((root/n).read_text()) for n in ARTIFACT_PROTOCOLS]
    return {'retfound_green':a[0]['artifact'],'visionfm_last4':a[1]['artifact'],**a[2]['comparator_artifacts']}

def authenticate_artifacts(a):
    exact_keys(a,probe.WIDTHS)
    for name in ('retfound_green','dinov3_generic'):
        require(sha(a[name]['checkpoint'])==a[name]['checkpoint_sha256'])
    v=a['visionfm_last4']
    for pathkey,hashkey in (('source_checkpoint','source_checkpoint_sha256'),('tensor_checkpoint','tensor_checkpoint_sha256'),('architecture_file','architecture_sha256')):
        require(sha(v[pathkey])==v[hashkey])
    lab=a['labrador'];r=Path(lab['model_root'])
    for relative,key in (('saved_model.pb','saved_model_sha256'),('variables/variables.data-00000-of-00001','variables_data_sha256'),('variables/variables.index','variables_index_sha256')):
        require(sha(r/relative)==lab[key])
    for k in ('codebook','ecdf'):require(sha(lab[k])==lab[k+'_sha256'])

def runtime_binding(a):
    script='import importlib.metadata as m,json,sys; print(json.dumps({"python":sys.version.split()[0],"packages":{k:m.version(k) for k in ("numpy","tensorflow","tf-keras")}},sort_keys=True))'
    tfexe=Path(a['labrador']['tensorflow_python']).absolute()
    p=subprocess.run([str(tfexe),'-c',script],capture_output=True,timeout=30,check=False)
    require(p.returncode==0 and len(p.stdout)<5000)
    tf=json.loads(p.stdout);exact_keys(tf,{'python','packages'});exact_keys(tf['packages'],{'numpy','tensorflow','tf-keras'})
    return {'main_python':sys.version.split()[0],'main_executable_sha256':sha(sys.executable),
            'packages':{k:metadata.version(k) for k in MODULES},'tensorflow':tf,'tensorflow_executable_sha256':sha(tfexe)}

def closure(root,names):
    """Bind recursive project-local Python imports without importing extra code."""
    found=set(names);pending=list(names)
    while pending:
        name=pending.pop()
        if not name.endswith('.py'):continue
        tree=ast.parse((root/name).read_text())
        for node in ast.walk(tree):
            modules=([a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in modules:
                file=module.split('.')[0]+'.py'
                if (root/file).is_file() and file not in found:found.add(file);pending.append(file)
    return found

def freeze(root=ROOT):
    root=Path(root);require(not(root/PROTOCOL).exists());old=old_protocol(root)
    require(prior.base.audit(root)['status']=='authenticated_success')
    a=artifacts(root);authenticate_artifacts(a)
    names=closure(root,set(old['expected_hashes'])|NEW_FILES|{PRIOR_PROTOCOL})
    require(not any((root/v).exists() for v in PATHS.values()))
    p={'schema':SCHEMA,'status':'frozen_before_execution','created_utc':datetime.now(timezone.utc).isoformat(),
       'parameters':PARAMETERS,'privacy':PRIVACY,'paths':PATHS,'authentication':old['authentication'],
       'scope':old['scope'],'data_roots':old['data_roots'],'artifacts':a,'runtime':runtime_binding(a),
       'prior_protocol_sha256':PRIOR_PROTOCOL_SHA,'prior_result_sha256':PRIOR_RESULT_SHA,
       'expected_hashes':{n:sha(root/n) for n in sorted(names)}}
    write_x(root/PROTOCOL,p);validate_protocol(root)
    return {'status':'frozen','protocol_sha256':sha(root/PROTOCOL),'components':len(names)}

def validate_protocol(root=ROOT,*,check_external=False):
    root=Path(root);old=old_protocol(root);p=json.loads((root/PROTOCOL).read_text())
    exact_keys(p,{'schema','status','created_utc','parameters','privacy','paths','authentication','scope','data_roots','artifacts','runtime','prior_protocol_sha256','prior_result_sha256','expected_hashes'})
    require(p['schema']==SCHEMA and p['status']=='frozen_before_execution' and p['parameters']==PARAMETERS and p['privacy']==PRIVACY and p['paths']==PATHS)
    require(p['prior_protocol_sha256']==PRIOR_PROTOCOL_SHA and p['prior_result_sha256']==PRIOR_RESULT_SHA)
    for key in ('authentication','scope','data_roots'):require(p[key]==old[key])
    require(p['artifacts']==artifacts(root))
    names=closure(root,set(old['expected_hashes'])|NEW_FILES|{PRIOR_PROTOCOL});exact_keys(p['expected_hashes'],names)
    for n,h in p['expected_hashes'].items():require((root/n).resolve().parent==root.resolve() and sha(root/n)==h)
    if check_external:
        authenticate_artifacts(p['artifacts']);require(p['runtime']==runtime_binding(p['artifacts']))
    return p

def validate_result(r,sources):
    exact_keys(r,{'schema','endpoint_results','macro_paired_deltas','named_fm_macro_family','inference','patient_arrays_or_draws_serialized','original_nature_retfound_included','longitudinal_ehr_fm_included'})
    require(r['schema']=='bran-named-fm-matched-auroc-v1' and r['inference']=='marginal_unadjusted_fixed_fit_common_patient_bootstrap')
    for key in ('patient_arrays_or_draws_serialized','original_nature_retfound_included','longitudinal_ehr_fm_included'):require(r[key] is False)
    endpoints=r['endpoint_results'];exact_keys(endpoints,sources);require(len(sources)==26)
    contrasts={'bran_both_minus_'+x for x in probe.ARMS[1:]}
    for e in endpoints.values():
        exact_keys(e,{'arms','paired_deltas'});exact_keys(e['arms'],probe.ARMS);exact_keys(e['paired_deltas'],contrasts)
        for m in e['arms'].values():
            exact_keys(m,{'auroc','logloss','ci95'});num(m['auroc'],0,1);num(m['logloss'],0);interval(m['ci95'],0,1)
        for name,m in e['paired_deltas'].items():
            exact_keys(m,{'auroc_difference','ci95'});num(m['auroc_difference'],-1,1);interval(m['ci95'],-1,1)
            require(abs(m['auroc_difference']-(e['arms']['bran_both']['auroc']-e['arms'][name.removeprefix('bran_both_minus_')]['auroc']))<1e-10)
    exact_keys(r['macro_paired_deltas'],contrasts)
    for name,m in r['macro_paired_deltas'].items():
        exact_keys(m,{'mean_endpoint_auroc_difference','ci95'});num(m['mean_endpoint_auroc_difference'],-1,1);interval(m['ci95'],-1,1)
        require(abs(m['mean_endpoint_auroc_difference']-sum(e['paired_deltas'][name]['auroc_difference'] for e in endpoints.values())/26)<1e-10)
    fam=r['named_fm_macro_family'];exact_keys(fam,{'comparators','bonferroni_one_sided_95_lower','all_lower_above_zero'})
    require(fam['comparators']==list(probe.WIDTHS));exact_keys(fam['bonferroni_one_sided_95_lower'],probe.WIDTHS)
    for a,v in fam['bonferroni_one_sided_95_lower'].items():
        num(v,-1,1);require(v<=r['macro_paired_deltas']['bran_both_minus_'+a]['ci95'][0]+1e-12)
    require(type(fam['all_lower_above_zero']) is bool and fam['all_lower_above_zero']==all(v>0 for v in fam['bonferroni_one_sided_95_lower'].values()))
    return {'macro_auroc':{a:sum(e['arms'][a]['auroc'] for e in endpoints.values())/26 for a in probe.ARMS},
            'macro_paired_deltas':r['macro_paired_deltas'],'named_fm_macro_family':fam}

def validate_report(r,p):
    exact_keys(r,{'schema','status','protocol_sha256','code_hashes','source_hashes','fold_hashes','support_receipt_sha256','scope','parameters','privacy','runtime','selection','diagnostics','replay','result'})
    require(r['schema']==SCHEMA and r['status']=='completed_aggregate_only' and r['code_hashes']==p['expected_hashes'])
    a=p['authentication'];require(r['source_hashes']==a['canonical_source_hashes'] and r['support_receipt_sha256']==a['support_receipt_sha256'])
    require(r['fold_hashes']=={'outer':a['outer_fold_sha256'],'inner':a['inner_fold_sha256']})
    require(r['scope']=={'patient_count':1928,'endpoint_count':26,'official_test_loaded':False})
    for k in ('parameters','privacy','runtime'):require(r[k]==p[k])
    sel=r['selection'];exact_keys(sel,{'ordered_source_sha256','image_count','canonical_view_counts_equal','unchanged_stat_checks','individual_hashes_serialized'})
    require(type(sel['ordered_source_sha256']) is str and len(sel['ordered_source_sha256'])==64 and all(c in '0123456789abcdef' for c in sel['ordered_source_sha256']))
    require(type(sel['image_count']) is int and sel['image_count']>=10 and sel['canonical_view_counts_equal'] is True and sel['unchanged_stat_checks'] is True and sel['individual_hashes_serialized'] is False)
    require(r['replay']=={'screening_3x26_auroc_and_logloss':True,'tolerance':1e-8})
    d=r['diagnostics'];exact_keys(d,{'reference','foundation_models'})
    ref=d['reference'];exact_keys(ref,{'encoder_fits','reference_head_refits','innovation_candidates_rejected_nonconvergence','legacy_candidates_rejected_nonconvergence','legacy_candidates_rejected_incomplete_inner_support','encoder_weights_unchanged_after_heads'})
    require(type(ref['encoder_fits']) is int and ref['encoder_fits']==5 and type(ref['reference_head_refits']) is int and ref['reference_head_refits']==390 and ref['encoder_weights_unchanged_after_heads'] is True)
    for k in ref:
        if 'rejected' in k:require(type(ref[k]) is int and 0<=ref[k]<=10000)
    exact_keys(d['foundation_models'],probe.WIDTHS)
    for fm in d['foundation_models'].values():
        exact_keys(fm,{'completed_head_refits','candidates_rejected_nonconvergence','candidates_rejected_incomplete_inner_support'})
        require(type(fm['completed_head_refits']) is int and fm['completed_head_refits']==130)
        for k in fm:require(type(fm[k]) is int and 0<=fm[k]<=10000)
    return validate_result(r['result'],p['scope']['eligible_source_codes'])

def image_signature(paths,rows,dataset):
    """One joint content digest, no patient/image-level hash leaves this process."""
    h=hashlib.sha256();stats=[]
    for path,row in zip(paths,rows,strict=True):
        path=Path(path);s=path.stat();stats.append((s.st_size,s.st_mtime_ns))
        h.update(json.dumps([str(path.relative_to(dataset)),int(row),s.st_size],separators=(',',':')).encode());h.update(b'\0')
        h.update(bytes.fromhex(sha(path)))
        after=path.stat();require((after.st_size,after.st_mtime_ns)==stats[-1])
    return h.hexdigest(),stats

def run(root=ROOT):
    root=Path(root);p={};fd=None;quiet=None;phase='protocol';paths={k:root/v for k,v in PATHS.items()}
    try:
        p=validate_protocol(root,check_external=True)
        require(not any(paths[k].exists() for k in ('success','failure','progress')))
        fd=prior.base.paired.base._acquire_lock(paths['lock'])
        def progress(name,fold=None,variant=None,batch=None,total=None):
            nonlocal phase
            require(name in PHASES);phase=name;q={'status':'running','phase':name}
            if fold is not None:require(type(fold) is int and 0<=fold<5);q['fold']=fold
            if variant is not None:require(variant in probe.WIDTHS);q['variant']=variant
            if batch is not None:require(type(batch) is int and type(total) is int and 0<=batch<=total);q.update(batch=batch,batches=total)
            temp=paths['progress'].with_suffix('.tmp');temp.parent.mkdir(parents=True,exist_ok=True)
            temp.write_text(json.dumps(q,sort_keys=True)+'\n');os.replace(temp,paths['progress'])
        progress('validated');quiet=prior.base.paired.base._quiet_sensitive_block();quiet.__enter__();progress('context')
        from patient_atlas_v6_2_expanded_endpoint_evaluation import FROZEN_SUPPORT_RECEIPT_NAME,EXACT_OUTER_FOLD_HASH,EXACT_INNER_FOLD_ASSIGNMENT_SHA256,load_eligible_support_receipt,validate_support_against_observed
        from run_patient_atlas_v6_2_expanded_endpoint_evaluation import _load_actual_v6_2_context
        from run_patient_atlas_v5_foundation_comparator import _enumerate_selected_cfp
        from bran_named_fm_extraction_v1_attempt2 import extract_retinal,extract_labrador
        support=load_eligible_support_receipt(root/FROZEN_SUPPORT_RECEIPT_NAME,project_root=root)
        context=_load_actual_v6_2_context(root=root,support=support,**{k:Path(v) for k,v in p['data_roots'].items()})
        outer=np.asarray(context['outer_assignment']);a=p['authentication'];sources=p['scope']['eligible_source_codes']
        require(len(outer)==1928 and set(sources)==set(support.eligible_sources) and support.receipt_sha256==a['support_receipt_sha256'])
        require(a['canonical_source_hashes']==dict(context['source_hashes']))
        require(a['outer_fold_sha256']==EXACT_OUTER_FOLD_HASH)
        validate_support_against_observed(support,context['labels_by_source'],context['observed_by_source'],outer)
        labels={s:context['labels_by_source'][s] for s in sources};observed={s:context['observed_by_source'][s] for s in sources}
        c,cm,elig,r,rm,names=prior.base.paired._actual_arrays(root,context);elig[:,48:]=False
        ages=np.asarray(context['raw_cohort'].ages);inners=[]
        for f in range(5):
            inner,digest=prior.base.paired.base._inner_context(context,np.flatnonzero(outer!=f),f)
            require(digest==a['inner_fold_sha256'][f]==EXACT_INNER_FOLD_ASSIGNMENT_SHA256[f]);inners.append(inner)
        progress('selection');dataset=Path(p['data_roots']['dataset_root']);cohort=context['feature_cohort']
        require(tuple(cohort.patient_ids)==tuple(context['raw_cohort'].patient_ids))
        image_paths,rows=_enumerate_selected_cfp(dataset_root=dataset,patient_ids=cohort.patient_ids)
        require(np.array_equal(np.bincount(rows,minlength=1928),np.asarray(cohort.eye_observed_mask,bool).sum(1)) and np.array_equal(np.bincount(rows,minlength=1928)>0,rm))
        signature,stats=image_signature(image_paths,rows,dataset)
        def unchanged():
            require([(x.stat().st_size,x.stat().st_mtime_ns) for x in image_paths]==stats)
        pred,refdiag=predict_references(c,cm,elig,r,rm,ages,outer,inners,labels,observed,sources=sources,progress_callback=progress)
        progress('reference_replay');old=json.loads((root/PRIOR_RESULT).read_text())
        replay=replay_references(pred,labels,observed,outer,old['result']);fmdiag={}
        for variant in probe.WIDTHS:
            unchanged()
            if variant=='labrador':
                progress('labrador_extraction',variant=variant)
                embedded=extract_labrador(c,cm,elig,names,artifact=p['artifacts'][variant],project_root=root)
            else:
                progress('retinal_extraction',variant=variant)
                embedded=extract_retinal(image_paths,rows,1928,variant=variant,artifact=p['artifacts'][variant],
                    progress_callback=lambda v,b,t:progress('retinal_extraction',variant=v,batch=b,total=t))
            unchanged();diag={'completed_head_refits':0,'candidates_rejected_nonconvergence':0,'candidates_rejected_incomplete_inner_support':0}
            for f in range(5):
                progress('fm_heads',fold=f,variant=variant);tr=np.flatnonzero(outer!=f);te=np.flatnonzero(outer==f)
                local,d=probe.fit_probe_fold(embedded,ages,tr,te,inners[f],labels,observed,variant=variant,source_codes=sources)
                for k in diag:diag[k]+=d.get(k,0)
                for s in sources:
                    if variant not in pred[s]:pred[s][variant]=np.full(1928,np.nan)
                    pred[s][variant][te]=local[s]
            fmdiag[variant]=diag;del embedded
        progress('aggregate');counts=prior.base.paired._shared_bootstrap_counts(outer)
        result=probe.summarize_paired(pred,labels,observed,outer,counts,source_codes=sources)
        selection={'ordered_source_sha256':signature,'image_count':len(image_paths),'canonical_view_counts_equal':True,'unchanged_stat_checks':True,'individual_hashes_serialized':False}
        report={'schema':SCHEMA,'status':'completed_aggregate_only','protocol_sha256':sha(root/PROTOCOL),'code_hashes':p['expected_hashes'],
          'source_hashes':a['canonical_source_hashes'],'fold_hashes':{'outer':a['outer_fold_sha256'],'inner':a['inner_fold_sha256']},
          'support_receipt_sha256':a['support_receipt_sha256'],'scope':{'patient_count':1928,'endpoint_count':26,'official_test_loaded':False},
          'parameters':PARAMETERS,'privacy':PRIVACY,'runtime':p['runtime'],'selection':selection,'diagnostics':{'reference':refdiag,'foundation_models':fmdiag},'replay':replay,'result':result}
        progress('writing');validate_protocol(root,check_external=True);validate_report(report,p);unchanged()
        write_x(paths['success'],report);progress('completed')
        return {'status':'completed_aggregate_only','artifact_sha256':sha(paths['success'])}
    except Exception as exc:
        frames=[{'file':Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(exc.__traceback__) if Path(x.filename).resolve().parent==root.resolve() and Path(x.filename).name in p.get('expected_hashes',{})][:8]
        failure={'schema':SCHEMA,'status':'failed','phase':phase,'error_class':type(exc).__name__,'bound_code_frames':frames,'exception_text_serialized':False,'patient_content_serialized':False}
        if fd is not None and not paths['success'].exists() and not paths['failure'].exists():write_x(paths['failure'],failure)
        return {'status':'execution_failed','phase':phase,'error_class':type(exc).__name__,'exception_contents_emitted':False}
    finally:
        if quiet is not None:quiet.__exit__(None,None,None)
        if fd is not None:
            os.close(fd)
            try:paths['lock'].unlink()
            except FileNotFoundError:pass

def audit(root=ROOT):
    root=Path(root);p=validate_protocol(root);paths={k:root/v for k,v in PATHS.items()}
    require(not(paths['success'].exists() and paths['failure'].exists()))
    if paths['failure'].exists():
        r=json.loads(paths['failure'].read_text());exact_keys(r,{'schema','status','phase','error_class','bound_code_frames','exception_text_serialized','patient_content_serialized'})
        require(r['schema']==SCHEMA and r['status']=='failed' and r['phase'] in PHASES and r['exception_text_serialized'] is False and r['patient_content_serialized'] is False)
        require(type(r['error_class']) is str and r['error_class'].isidentifier() and len(r['error_class'])<90 and type(r['bound_code_frames']) is list and len(r['bound_code_frames'])<=8)
        for f in r['bound_code_frames']:
            exact_keys(f,{'file','line'});require(f['file'] in p['expected_hashes'] and type(f['line']) is int and 0<f['line']<100000)
        return {'status':'authenticated_execution_failure','artifact_sha256':sha(paths['failure']),'phase':r['phase'],'error_class':r['error_class'],'code_frames':r['bound_code_frames']}
    if paths['success'].exists():
        r=json.loads(paths['success'].read_text());require(r['protocol_sha256']==sha(root/PROTOCOL));summary=validate_report(r,p)
        return {'status':'authenticated_success','artifact_sha256':sha(paths['success']),'protocol_sha256':sha(root/PROTOCOL),'components':len(p['expected_hashes']),'scope':r['scope'],'folds_and_replay_authenticated':True,**summary}
    out={'status':'no_terminal_artifact','lock_exists':paths['lock'].exists()}
    if paths['progress'].exists():
        q=json.loads(paths['progress'].read_text());require(isinstance(q,dict) and set(q)<= {'status','phase','fold','variant','batch','batches'} and q.get('status')=='running' and q.get('phase') in PHASES)
        if 'fold' in q:require(type(q['fold']) is int and 0<=q['fold']<5)
        if 'variant' in q:require(q['variant'] in probe.WIDTHS)
        if 'batch' in q or 'batches' in q:require(type(q.get('batch')) is int and type(q.get('batches')) is int and 0<=q['batch']<=q['batches'])
        out['progress']=q
    return out

if __name__=='__main__':
    try:print(json.dumps({'freeze':freeze,'run':run,'audit':audit}[sys.argv[1]](),sort_keys=True,allow_nan=False))
    except Exception as exc:
        print(json.dumps({'status':'blocked_without_disclosure','error_class':type(exc).__name__,'contents_emitted':False}));raise SystemExit(1)

