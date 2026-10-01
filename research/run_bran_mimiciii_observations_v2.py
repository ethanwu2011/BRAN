"""Private two-pass MIMIC-III lab cache, with frozen source/code/dictionary pins.

No notes, diagnoses, images, or hosted processing. Rows are consumed only under
a local FD-quiet lock. No efficacy or independent MIMIC-IV validation claim.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path

import numpy as np

from run_bran_multisource_retinal_features_v2 import LOCK,quiet,sha,write_json
import run_bran_mimiciii_intake_v2 as intake
from bran_clinical_source_reader_v1 import iter_projected_csv
from bran_mimiciii_metadata_v2 import build_mimiciii_metadata
from bran_mimiciii_observations_v2 import (bind_mimiciii_dictionary,
    build_mimiciii_observations,validate_observations,MimicIIIObservationArrays,FIELDS)
from bran_multisource_clinical_v2 import ClinicalPoolV2,safe_summary,eligibility_hash

ROOT=Path(__file__).resolve().parent
SOURCE=intake.SOURCE
INTAKE=ROOT/'BRAN_MIMICIII_INTAKE_V2_ATTEMPT1/aggregate.json'
TABLES={
    'patients':('PATIENTS.csv.gz',('SUBJECT_ID','DOB'),2000000),
    'admissions':('ADMISSIONS.csv.gz',('SUBJECT_ID','HADM_ID','ADMITTIME'),2000000),
    'dictionary':('D_LABITEMS.csv.gz',('ITEMID','LABEL','FLUID','CATEGORY'),2000000),
    'events':('LABEVENTS.csv.gz',('SUBJECT_ID','HADM_ID','ITEMID','VALUENUM','VALUEUOM','CHARTTIME'),200000000)}
CODE=('run_bran_mimiciii_observations_v2.py','bran_mimiciii_observations_v2.py',
    'bran_mimiciii_metadata_v2.py','run_bran_mimiciii_intake_v2.py',
    'bran_clinical_source_reader_v1.py','bran_clinical_dictionary_binding_v1.py',
    'bran_clinical_semantics_v1.py','bran_clinical_chemistry_semantics_v1.py',
    'bran_joint_lab_online_v1.py','bran_cbc_chemistry_snapshot_v1.py',
    'bran_cbc_event_adapter_v1.py','bran_multisource_clinical_v2.py',
    'bran_multisource_age_v2.py','run_bran_multisource_retinal_features_v2.py')
POLICY={'source':'mimiciii','source_family':'mimic','role':'clinical_development',
    'version':'local_tables_pinned_release_number_unverified','cross_release_identity_resolved':False,
    'independent_of_mimiciv':False,'minimum_observed_cbc':2,'maximum_admission_minutes':1440,
    'context_minutes':60,'original_observations_only':True,'exact_compatible_units_only':True,
    'ties':'earliest_conflict_missing_no_later_replacement','split_percent':[80,10,10],
    'age':'completed_year_interval_or_verified_dob_shift_unknown_adult',
    'notes_diagnoses_medications_inputs':False,'same_physical_draw_proven':False}
PHASES=('source_authentication','dictionary_binding','metadata','CBC_anchor_scan',
        'joint_context_scan','private_cache_write','post_scan_authentication','audit','completed')


def require(ok):
    if not ok:raise ValueError('mimiciii_observation_run_failed')


def paths(attempt):
    require(type(attempt) is int and 1<=attempt<=99)
    return (ROOT/('BRAN_MIMICIII_OBSERVATIONS_V2_ATTEMPT'+str(attempt)),
        ROOT/'private_artifacts'/('bran_mimiciii_observations_v2_attempt'+str(attempt)))


def code_hashes():return {name:sha(ROOT/name) for name in CODE}


def rows(table):
    name,columns,limit=TABLES[table]
    return iter_projected_csv(SOURCE/name,columns,max_rows=limit)


def progress(out,phase):
    require(phase in PHASES)
    value={'phase':phase,'pid':os.getpid(),'training_started':False,'patient_level_output_emitted':False}
    temporary=out/'progress.tmp';write_json(temporary,value);os.replace(temporary,out/'progress.json')


def source_receipt():
    require(INTAKE.is_file() and not INTAKE.is_symlink())
    expected=json.loads(INTAKE.read_text());require(intake.inspect(SOURCE)==expected)
    return expected


def prepare(attempt,state):
    out,private=paths(attempt)
    require(not out.exists() and not out.is_symlink() and not private.exists() and not private.is_symlink())
    out.mkdir();state.update(owned=out,phase='source_authentication');progress(out,state['phase'])
    original=source_receipt();before=code_hashes()
    state['phase']='dictionary_binding';progress(out,state['phase'])
    binding=bind_mimiciii_dictionary(rows('dictionary'))
    require(len(set(binding.code_to_field.values())&set(FIELDS[:9]))>=2)
    private.mkdir(mode=0o700)
    fd=os.open(private/'split_salt.bin',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(os.urandom(32));stream.flush();os.fsync(stream.fileno())
    require(code_hashes()==before and source_receipt()==original)
    protocol={'schema':'bran-mimiciii-observations-protocol-v2','status':'frozen_before_observation_scan',
        'policy':POLICY,'intake_sha256':sha(INTAKE),'source_receipt':original,
        'code_sha256':before,'dictionary_code_to_field':dict(binding.code_to_field),
        'split_salt_sha256':sha(private/'split_salt.bin'),'patient_level_output_emitted':False}
    write_json(out/'protocol.json',protocol);state['owned']=None
    return sha(out/'protocol.json')


def authenticate(attempt,pin):
    out,private=paths(attempt)
    require(not (out/'failure.json').exists() and sha(out/'protocol.json')==pin)
    require(private.is_dir() and not private.is_symlink() and private.stat().st_mode&0o777==0o700)
    protocol=json.loads((out/'protocol.json').read_text())
    require(protocol['schema']=='bran-mimiciii-observations-protocol-v2' and protocol['policy']==POLICY
        and protocol['code_sha256']==code_hashes() and protocol['intake_sha256']==sha(INTAKE)
        and protocol['source_receipt']==source_receipt()
        and protocol['split_salt_sha256']==sha(private/'split_salt.bin'))
    salt=private/'split_salt.bin';require(not salt.is_symlink() and salt.stat().st_mode&0o777==0o600 and salt.stat().st_size==32)
    binding=bind_mimiciii_dictionary(rows('dictionary'))
    require(dict(binding.code_to_field)==protocol['dictionary_code_to_field'])
    return protocol,binding


def training_pool(arrays):
    validate_observations(arrays);take=np.flatnonzero(arrays.split==0)
    require(len(take)>0)
    values=[getattr(arrays,name)[take].copy() for name in ('values','observed','person_group',
        'age_value','age_lower','age_upper','age_kind')]
    for value in values:value.setflags(write=False)
    take.setflags(write=False)
    return ClinicalPoolV2('mimiciii',*values,take)


def run(attempt,pin,state):
    out,private=paths(attempt)
    require(not any((out/name).exists() for name in ('aggregate.json','failure.json'))
            and not (private/'observations.npz').exists())
    state.update(owned=out,phase='source_authentication')
    protocol,binding=authenticate(attempt,pin)
    state['phase']='metadata';progress(out,state['phase'])
    metadata=build_mimiciii_metadata(rows('patients'),rows('admissions'))
    passes=0
    def event_factory():
        nonlocal passes
        require(passes<2);state['phase']=('CBC_anchor_scan','joint_context_scan')[passes]
        progress(out,state['phase']);passes+=1
        return rows('events')
    arrays=build_mimiciii_observations(metadata,binding,event_factory,(private/'split_salt.bin').read_bytes())
    require(passes==2);validate_observations(arrays)
    pool=training_pool(arrays);summary=safe_summary(pool);require(summary['status']=='supported_training_pool')
    state['phase']='private_cache_write';progress(out,state['phase'])
    fd=os.open(private/'observations.npz',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as stream:
        np.savez_compressed(stream,**vars(arrays));stream.flush();os.fsync(stream.fileno())
    state['phase']='post_scan_authentication';progress(out,state['phase'])
    require(authenticate(attempt,pin)[0]==protocol)
    artifact={'schema':'bran-mimiciii-observations-v2','status':'observed_cache_materialized_not_trained',
        'protocol_sha256':pin,'private_cache_sha256':sha(private/'observations.npz'),
        'training_summary':summary,'eligibility_sha256':eligibility_hash(pool),
        'admitted_dictionary_fields':sorted(set(binding.code_to_field.values())),
        'policy':POLICY,'training_started':False,'new_encoder_trained':False,
        'patient_level_output_emitted':False}
    write_json(out/'aggregate.json',artifact)
    write_json(out/'manifest.json',{'aggregate_sha256':sha(out/'aggregate.json'),'protocol_sha256':pin,
        'patient_level_output_emitted':False})
    progress(out,'completed');state['owned']=None


def audit(attempt,pin,state):
    out,private=paths(attempt);require(not (out/'audit.json').exists() and not (out/'audit_failure.json').exists())
    state.update(owned=out,phase='audit',failure_name='audit_failure.json')
    authenticate(attempt,pin);value=json.loads((out/'aggregate.json').read_text())
    require(json.loads((out/'manifest.json').read_text())=={'aggregate_sha256':sha(out/'aggregate.json'),
        'protocol_sha256':pin,'patient_level_output_emitted':False})
    path=private/'observations.npz'
    require(path.is_file() and not path.is_symlink() and path.stat().st_mode&0o777==0o600
            and path.stat().st_nlink==1 and sha(path)==value['private_cache_sha256'])
    with np.load(path,allow_pickle=False) as archive:
        require(set(archive.files)==set(MimicIIIObservationArrays.__dataclass_fields__))
        data={name:archive[name].copy() for name in archive.files}
    for array in data.values():array.setflags(write=False)
    arrays=MimicIIIObservationArrays(**data);pool=training_pool(arrays)
    require(safe_summary(pool)==value['training_summary'] and eligibility_hash(pool)==value['eligibility_sha256'])
    require(sha(path)==value['private_cache_sha256'])
    write_json(out/'audit.json',{'status':'authenticated','protocol_sha256':pin,
        'aggregate_sha256':sha(out/'aggregate.json'),'private_cache_sha256':sha(path),
        'cache_schema_and_training_eligibility_replayed':True,'full_source_event_reextraction_replayed':False,
        'patient_level_output_emitted':False,'training_started':False})
    state['owned']=None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run','audit'))
    parser.add_argument('--attempt',type=int,default=1);parser.add_argument('--protocol-sha256')
    args=parser.parse_args();state={'owned':None,'phase':'source_authentication','failure_name':'failure.json'}
    answer={'status':'failed','patient_level_output_emitted':False,'training_started':False}
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
                    'phase':state['phase'],'reason':'source_or_observation_contract_failed',
                    'patient_level_output_emitted':False,'training_started':False})
                except Exception:pass
    print(json.dumps(answer));return int(answer['status']=='failed')


if __name__=='__main__':raise SystemExit(main())
