"""Single-heavy-job, source-authenticated V2 fitting. No outcome selection.

Preparation closes the source roster before performance inspection. Execution
retains every arm/fold and its exposure receipt. A completed fit is not a
successful scientific result; promotion/evaluation are separate prerequisites.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
from collections.abc import Mapping
from types import SimpleNamespace

from run_bran_multisource_retinal_features_v2 import LOCK,quiet,sha,write_json
from bran_multisource_protocol_v2 import (pending_receipts,build_protocol,digest,
    PARAMETERS,SOURCE_POLICY)
from bran_multisource_batches_v2 import transform_hash
from bran_multisource_fit_v2 import fit_one,load_checkpoint

ROOT=Path(__file__).resolve().parent
TRAINING=('aireadi','brset','eicu','mimiciii','mimiciv','nhanes_exposed','nwicu')
NONADMITTED={
    'sicdb':('pending_units','Dictionary/type metadata alone does not qualify an observed-assay cache; the retained audit does not admit hemoglobin.'),
    'zigong':('pending_source_qualification','Archive is inventoried, but original-unit measurements, linkage and altered-age semantics lack an authenticated V2 cache.'),
    'odir':('pending_access','Existing grouping/content/split work is retained; original-release and reuse provenance remain unauthenticated.'),
    'dr_unified':('pending_grouping','Historical source-tag exposure is not an available authenticated original-image source with participant grouping and overlap proof.'),
    'jsiec':('pending_grouping','Historical filename syntax is not authenticated original bytes, participant grouping, overlap clearance or a source-bound reader.')}
CODE=('run_bran_multisource_fit_v2.py','bran_multisource_local_sources_v2.py',
    'bran_multisource_protocol_v2.py','bran_multisource_fit_v2.py','bran_multisource_batches_v2.py',
    'bran_multisource_data_v2.py','bran_multisource_model_v2.py','bran_multisource_age_v2.py',
    'bran_multisource_training_v2.py','bran_multisource_clinical_v2.py',
    'bran_multisource_nwicu_v2.py','run_bran_multisource_paired_binding_v2.py',
    'bran_patient_state_prototype_v1.py','bran_screening_joint_kernel_v1.py',
    'bran_joint_lab_task_contract_v1.py','bran_joint_lab_cache_v1.py','bran_research_state_io_v1.py')


def require(ok):
    if not ok:raise ValueError('multisource_fit_runner_failed')


def paths(attempt):
    require(type(attempt) is int and 1<=attempt<=99)
    return (ROOT/('BRAN_MULTISOURCE_FIT_V2_ATTEMPT'+str(attempt)),
        ROOT/'private_artifacts'/('bran_multisource_fit_v2_attempt'+str(attempt)))


def code_hashes():return {name:sha(ROOT/name) for name in CODE}


def progress(out,event):
    require(set(event)<={'phase','arm','fold','stage','updates_completed'})
    require(event['phase'] in ('source_authentication','training','post_fit_authentication','audit','completed_fits_pending_evaluation'))
    value={**event,'patient_level_output_emitted':False,'pid':os.getpid()}
    temporary=out/'progress.tmp';write_json(temporary,value);os.replace(temporary,out/'progress.json')


def source_closure(evidence):
    require(type(evidence) is dict and tuple(sorted(evidence))==TRAINING)
    receipts=pending_receipts()
    for name,ev in evidence.items():receipts[name]={'disposition':'training','reason':'qualified','evidence':ev}
    for name,(reason,_) in NONADMITTED.items():
        require(SOURCE_POLICY[name][1]=='development')
        receipts[name]={'disposition':'excluded','reason':reason,'evidence':None}
    return receipts


def load_sources():
    from bran_multisource_local_sources_v2 import load_prepared_sources
    private=load_prepared_sources()
    def plain(value):
        return {key:plain(item) for key,item in value.items()} if isinstance(value,Mapping) else value
    require(all(key==pool.source for key,pool in private.pools.items()))
    return SimpleNamespace(paired=private.paired,
        pools=tuple(private.pools[name] for name in sorted(private.pools)),
        source_evidence=plain(private.source_evidence),artifact_pins=plain(private.artifact_pins))


def expected(sources):
    paired=sources.paired
    require(tuple(sorted(p.source for p in sources.pools))==tuple(s for s in TRAINING if s!='aireadi'))
    counts={'aireadi':len(paired.folds),**{p.source:len(p.person_group) for p in sources.pools}}
    protocol=build_protocol(source_closure(sources.source_evidence),
        outer_folds_sha256=paired.receipt['outer_fold_sha256'],
        inner_folds_sha256=paired.receipt['inner_fold_sha256'],
        transform_sha256=[transform_hash(t) for t in paired.transforms],
        code_sha256=code_hashes(),training_examples=counts)
    protocol.update(artifact_pins=sources.artifact_pins,
        source_admission_notes={name:note for name,(_,note) in NONADMITTED.items()},
        status='source_authenticated_frozen_before_training',
        nonadmitted_sources_not_claimed_as_new_training=True,
        protected_sources_not_used_for_model_selection=True)
    return protocol


def prepare(attempt,state):
    out,private=paths(attempt)
    require(not out.exists() and not out.is_symlink() and not private.exists() and not private.is_symlink())
    out.mkdir();state.update(owned=out,phase='source_authentication')
    progress(out,{'phase':state['phase']});sources=load_sources();protocol=expected(sources)
    private.mkdir(mode=0o700)
    write_json(out/'protocol.json',protocol)
    write_json(out/'source_decisions.json',{'training_sources':list(TRAINING),
        'not_admitted_to_this_frozen_experiment':protocol['source_admission_notes'],
        'amsterdamumcdb':'user_deferred_not_searched',
        'reason':'qualification_constraints_not_outcome_or_deadline_selection',
        'actual_new_model_exposure_established':False})
    state['owned']=None;return sha(out/'protocol.json')


def authenticate(attempt,pin,*,load=True):
    out,private=paths(attempt)
    require(not any((out/name).exists() for name in ('failure.json','audit_failure.json'))
        and out.is_dir() and not out.is_symlink() and sha(out/'protocol.json')==pin)
    require(private.is_dir() and not private.is_symlink() and private.stat().st_mode&0o777==0o700)
    protocol=json.loads((out/'protocol.json').read_text())
    require(protocol['code_sha256']==code_hashes())
    sources=load_sources() if load else None
    if sources is not None:require(protocol==expected(sources))
    return protocol,sources


def run(attempt,pin,state):
    out,private=paths(attempt)
    require(not (out/'aggregate.json').exists() and not any(private.iterdir()))
    state.update(owned=out,phase='source_authentication');progress(out,{'phase':state['phase']})
    protocol,sources=authenticate(attempt,pin)
    components={}
    for fold in range(PARAMETERS['fold_count']):
        traces=[]
        for arm in protocol['arms']:
            require(sha(out/'protocol.json')==pin and protocol['code_sha256']==code_hashes())
            state['phase']='training';name='fold'+str(fold)+'_'+arm
            item=fit_one(protocol,sources.paired,sources.pools,arm=arm,fold=fold,
                private_directory=private/name,progress=lambda event:progress(out,event))
            write_json(out/(name+'.json'),item);components[name+'.json']=sha(out/(name+'.json'))
            traces.append((item['input_sequence_sha256'],item['mask_sequence_sha256']))
        require(traces[0]==traces[1])
    state['phase']='post_fit_authentication';progress(out,{'phase':state['phase']})
    authenticate(attempt,pin)
    value={'schema':'bran-multisource-fits-v2','status':'completed_fits_pending_evaluation',
        'protocol_sha256':pin,'protocol_content_sha256':digest(protocol),
        'component_sha256':components,'paired_arm_input_and_mask_traces_identical':True,
        'five_folds_completed':True,'arms':protocol['arms'],'candidate_promoted':False,
        'screening_gate_evaluated':False,'completion_gate_evaluated':False,
        'subtyping_established':False,'scientific_goal_achieved':False,'patient_level_output_emitted':False}
    write_json(out/'aggregate.json',value)
    write_json(out/'manifest.json',{'protocol_sha256':pin,'aggregate_sha256':sha(out/'aggregate.json'),
        'component_sha256':components,'patient_level_output_emitted':False})
    progress(out,{'phase':'completed_fits_pending_evaluation'});state['owned']=None


def audit(attempt,pin,state):
    out,private=paths(attempt);require(not (out/'audit.json').exists() and not (out/'audit_failure.json').exists())
    state.update(owned=out,phase='audit',failure_name='audit_failure.json')
    protocol,_=authenticate(attempt,pin)
    value=json.loads((out/'aggregate.json').read_text())
    components=value['component_sha256']
    require(set(components)=={'fold'+str(f)+'_'+a+'.json' for f in range(5) for a in protocol['arms']})
    require(json.loads((out/'manifest.json').read_text())=={'protocol_sha256':pin,
        'aggregate_sha256':sha(out/'aggregate.json'),'component_sha256':components,'patient_level_output_emitted':False})
    for name,expected_sha in components.items():
        require(sha(out/name)==expected_sha);item=json.loads((out/name).read_text())
        require(item['binding']['protocol_sha256']==digest(protocol))
        checkpoint='stage_C_step_'+str(PARAMETERS['stage_c_steps'])+'.pt'
        model,transform=load_checkpoint(private/name[:-5]/checkpoint,
            expected_sha256=item['checkpoint_manifest'][checkpoint]['sha256'],
            binding=item['binding'],stage='C',steps=PARAMETERS['stage_c_steps'])
        require(transform_hash(transform)==protocol['transform_sha256'][item['binding']['fold']])
        del model,transform
    write_json(out/'audit.json',{'status':'checkpoints_authenticated_pending_outcome_evaluation',
        'protocol_sha256':pin,'aggregate_sha256':sha(out/'aggregate.json'),
        'ten_final_checkpoints_reloaded':True,'candidate_promoted':False,
        'scientific_goal_achieved':False,'patient_level_output_emitted':False})
    state['owned']=None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run','audit'))
    parser.add_argument('--attempt',type=int,default=1);parser.add_argument('--protocol-sha256')
    args=parser.parse_args();state={'owned':None,'phase':'source_authentication','failure_name':'failure.json'}
    answer={'status':'failed','patient_level_output_emitted':False,'candidate_promoted':False}
    with quiet():
        try:
            with LOCK.open('a') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                if args.action=='prepare':pin=prepare(args.attempt,state)
                elif args.action=='run':pin=args.protocol_sha256;run(args.attempt,pin,state)
                else:pin=args.protocol_sha256;audit(args.attempt,pin,state)
                answer.update(status=args.action+'_completed',protocol_sha256=pin)
        except Exception:
            if state['owned'] is not None:
                try:write_json(state['owned']/state['failure_name'],{'status':'technical_failure',
                    'phase':state['phase'],'reason':'source_fit_or_checkpoint_contract_failed',
                    'patient_level_output_emitted':False,'candidate_promoted':False})
                except Exception:pass
    print(json.dumps(answer));return int(answer['status']=='failed')


if __name__=='__main__':raise SystemExit(main())
