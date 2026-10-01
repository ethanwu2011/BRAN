"""Bounded local linkage audit; closed aggregate output, no lab eligibility claim.

--prepare-protocol reads file bytes only to bind hashes. --run processes local
person/episode tables under FD suppression after the protocol is frozen. Neither
mode trains models or writes patient records/identifiers/assignments.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from bran_clinical_dictionary_binding_v1 import _quiet
from bran_clinical_source_reader_v1 import iter_projected_csv, build_episode_links
from bran_nhanes_source_reader_v1 import iter_xport_projection, link_nhanes_cycle

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'BRAN_SOURCE_LINKAGE_AUDIT_PROTOCOL_V1.json'
OUTPUT = ROOT / 'BRAN_SOURCE_LINKAGE_AUDIT_V1'
CODE_FILES = (
    'run_bran_source_linkage_audit_v1.py', 'bran_clinical_source_reader_v1.py',
    'bran_nhanes_source_reader_v1.py', 'bran_clinical_semantics_v1.py',
    'bran_clinical_dictionary_binding_v1.py',
)
INPUTS = {
    'mimic_patients': '/Users/ethanwu/mimiciv-3.1/hosp/patients.csv.gz',
    'mimic_admissions': '/Users/ethanwu/mimiciv-3.1/hosp/admissions.csv.gz',
    'nwicu_patients': '/Users/ethanwu/nwicu-northwestern-icu-0.1.0/data/nw_hosp/patients.csv.gz',
    'nwicu_admissions': '/Users/ethanwu/nwicu-northwestern-icu-0.1.0/data/nw_hosp/admissions.csv.gz',
    'eicu_patients': '/Users/ethanwu/eicu-crd-2.0/patient.csv.gz',
    'sicdb_cases': '/Users/ethanwu/sicdb-1.0.8/cases.csv.gz',
    'nhanes_demo_d': '/Users/ethanwu/nhanes-oculomics/DEMO_D.xpt',
    'nhanes_cbc_d': '/Users/ethanwu/nhanes-oculomics/CBC_D.xpt',
    'nhanes_demo_e': '/Users/ethanwu/nhanes-oculomics/DEMO_E.xpt',
    'nhanes_cbc_e': '/Users/ethanwu/nhanes-oculomics/CBC_E.xpt',
}
MAX_ROWS = 2000000
SOURCE_NAMES = ('mimic','eicu','nwicu','sicdb','nhanes','zigong')
SOURCE_INPUTS = {
    'mimic': ('mimic_patients','mimic_admissions'),
    'eicu': ('eicu_patients',), 'nwicu': ('nwicu_patients','nwicu_admissions'),
    'sicdb': ('sicdb_cases',),
    'nhanes': ('nhanes_demo_d','nhanes_cbc_d','nhanes_demo_e','nhanes_cbc_e'),
}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def exclusive_json(path,payload):
    with path.open('x') as handle:
        json.dump(payload,handle,sort_keys=True,indent=2,allow_nan=False)


def prepare():
    # No CSV/XPORT parse or count during protocol preparation.
    source_files={}
    for key,path in INPUTS.items():
        if not Path(path).is_file():
            source_files[key]={'path':path,'present':False,'sha256':None}
        else:
            source_files[key]={'path':path,'present':True,'sha256':sha(path)}
    return {
        'schema_version':'bran-source-linkage-audit-protocol-v1',
        'source_files':source_files,
        'code_sha256':{name:sha(ROOT/name) for name in CODE_FILES},
        'max_rows_per_table':MAX_ROWS,
        'patient_level_output_permitted':False,
        'small_cell_threshold':20,
        'training_permitted':False,
        'scope':'person/episode and NHANES within-cycle linkage only; not lab coverage, eligibility, model splitting or efficacy',
    }


def validate_protocol(protocol):
    if set(protocol) != {'schema_version','source_files','code_sha256','max_rows_per_table','patient_level_output_permitted','small_cell_threshold','training_permitted','scope'}:
        raise ValueError('protocol schema mismatch')
    if protocol['schema_version']!='bran-source-linkage-audit-protocol-v1' or protocol['max_rows_per_table']!=MAX_ROWS or protocol['patient_level_output_permitted'] is not False or protocol['training_permitted'] is not False or protocol['small_cell_threshold']!=20:
        raise ValueError('protocol policy mismatch')
    if set(protocol['source_files']) != set(INPUTS) or set(protocol['code_sha256']) != set(CODE_FILES):
        raise ValueError('protocol bindings mismatch')
    for name,expected in protocol['code_sha256'].items():
        if sha(ROOT/name)!=expected: raise ValueError('implementation hash mismatch')
    for key,record in protocol['source_files'].items():
        if set(record)!={'path','present','sha256'} or record['path']!=INPUTS[key] or type(record['present']) is not bool:
            raise ValueError('protocol source binding mismatch')
        if record['present'] and (not isinstance(record['sha256'],str) or len(record['sha256'])!=64):
            raise ValueError('protocol source hash missing')


def safe_counts(people,episodes):
    if type(people) is not int or type(episodes) is not int or people<0 or episodes<people:
        raise ValueError('invalid aggregate counts')
    # Complementary suppression: no rare repeat count recoverable by subtraction.
    if people<20 or episodes<20 or 0<episodes-people<20:
        return {'linked_people':None,'linked_episodes':None}
    return {'linked_people':people,'linked_episodes':episodes}


def _failed(status):
    return {'status':status,'linked_people':None,'linked_episodes':None,'training_ready':False}


def _source_links(source):
    if source in ('mimic','nwicu'):
        persons=iter_projected_csv(INPUTS[source+'_patients'],('subject_id',),max_rows=MAX_ROWS)
        admissions=iter_projected_csv(INPUTS[source+'_admissions'],('hadm_id','subject_id'),max_rows=MAX_ROWS)
        links=build_episode_links(source,persons,admissions)
    elif source=='eicu':
        rows=iter_projected_csv(INPUTS['eicu_patients'],('uniquepid','patientunitstayid','patienthealthsystemstayid'),max_rows=MAX_ROWS)
        links=build_episode_links(source,rows,())
    elif source=='sicdb':
        rows=iter_projected_csv(INPUTS['sicdb_cases'],('PatientID','CaseID'),max_rows=MAX_ROWS)
        links=build_episode_links(source,(),rows)
    elif source=='nhanes':
        people=set(); episodes=0
        for cycle in ('D','E'):
            suffix=cycle.lower()
            demos=iter_xport_projection(INPUTS['nhanes_demo_'+suffix],('SEQN','SDDSRVYR','RIDAGEYR'),max_rows=MAX_ROWS)
            cbcs=iter_xport_projection(INPUTS['nhanes_cbc_'+suffix],('SEQN',),max_rows=MAX_ROWS)
            linked=link_nhanes_cycle(cycle,demos,cbcs)
            # Repeated SEQN across cycles remains one person, not two independent
            # people. No cross-cycle record join or age averaging is performed.
            people.update(linked.person_age)
            episodes+=len(linked.person_age)
        return len(people),episodes
    else:
        raise ValueError('unsupported linkage source')
    return len(set(links.episode_to_person.values())),len(links.episode_to_person)


def audit(protocol):
    validate_protocol(protocol)
    results={}
    for source in SOURCE_NAMES:
        if source=='zigong':
            results[source]=_failed('original_source_unresolved')
            continue
        keys=SOURCE_INPUTS[source]
        if any(not protocol['source_files'][key]['present'] for key in keys):
            results[source]=_failed('source_file_unavailable')
            continue
        try:
            for key in keys:
                if sha(INPUTS[key])!=protocol['source_files'][key]['sha256']:
                    raise ValueError('input changed')
            people,episodes=_source_links(source)
            for key in keys:
                if sha(INPUTS[key])!=protocol['source_files'][key]['sha256']:
                    raise ValueError('input changed')
            results[source]={'status':'linkage_validated',**safe_counts(people,episodes),'training_ready':False}
        except Exception:
            # No exception string, identifier, path or patient value is exported.
            results[source]=_failed('linkage_validation_failed')
    return {'schema_version':'bran-source-linkage-audit-aggregate-v1','sources':results,
            'patient_rows_processed_locally':True,'patient_level_output_emitted':False,
            'lab_observations_evaluated':False,'training_started':False,
            'new_unique_training_pool_established':False}


def validate_aggregate(payload):
    if set(payload)!={'schema_version','sources','patient_rows_processed_locally','patient_level_output_emitted','lab_observations_evaluated','training_started','new_unique_training_pool_established'}:
        raise ValueError('invalid aggregate schema')
    if payload['schema_version']!='bran-source-linkage-audit-aggregate-v1' or set(payload['sources'])!=set(SOURCE_NAMES):
        raise ValueError('invalid aggregate schema')
    if payload['patient_rows_processed_locally'] is not True or any(payload[k] is not False for k in ('patient_level_output_emitted','lab_observations_evaluated','training_started','new_unique_training_pool_established')):
        raise ValueError('invalid aggregate privacy or claim flags')
    allowed={'linkage_validated','original_source_unresolved','source_file_unavailable','linkage_validation_failed'}
    for source,result in payload['sources'].items():
        if set(result)!={'status','linked_people','linked_episodes','training_ready'} or result['status'] not in allowed or result['training_ready'] is not False:
            raise ValueError('invalid aggregate source schema')
        p,e=result['linked_people'],result['linked_episodes']
        if (p is None)!=(e is None): raise ValueError('invalid count suppression')
        if p is not None and (safe_counts(p,e)!={'linked_people':p,'linked_episodes':e} or result['status']!='linkage_validated'):
            raise ValueError('invalid disclosed counts')
        if source=='zigong' and result['status']!='original_source_unresolved':
            raise ValueError('unresolved source claim invalid')


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-protocol',action='store_true')
    mode.add_argument('--run',action='store_true')
    args=parser.parse_args()
    ok=False
    own_output=False
    with _quiet():
        try:
            if args.prepare_protocol:
                exclusive_json(PROTOCOL,prepare())
            else:
                protocol=json.loads(PROTOCOL.read_text())
                validate_protocol(protocol)
                OUTPUT.mkdir()
                own_output=True
                payload=audit(protocol)
                validate_aggregate(payload)
                exclusive_json(OUTPUT/'aggregate.json',payload)
                exclusive_json(OUTPUT/'manifest.json',{'schema_version':'bran-source-linkage-audit-manifest-v1',
                    'protocol_sha256':sha(PROTOCOL),'aggregate_sha256':sha(OUTPUT/'aggregate.json'),
                    'patient_level_output_emitted':False,'training_started':False})
            ok=True
        except Exception:
            # Preserve a static technical-failure receipt only for this attempt.
            # If the output directory was already present, never change it.
            if own_output:
                try:
                    exclusive_json(OUTPUT/'failure.json',{'status':'linkage_audit_execution_failed',
                        'patient_level_output_emitted':False,'training_started':False})
                except Exception:
                    pass
    print(json.dumps({'status':('protocol_prepared' if args.prepare_protocol else 'linkage_audit_completed') if ok else 'linkage_audit_execution_failed',
                      'patient_level_output_emitted':False,'training_started':False}))
    return 0 if ok else 1


if __name__=='__main__': raise SystemExit(main())
