"""Local observed-CBC pool, not a model fit or clinical efficacy evaluation.

Original source rows are read only inside OS-output suppression. Private caches
contain measurements/masks and source-local person-group indices, never source
identifiers. Only a deeply closed, count-coarsened receipt leaves that boundary.
Assay and age qualification are retained separately from observation availability.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sys

import numpy as np

from bran_cbc_event_adapter_v1 import OnlineCBCPanels, adapt_cbc_event, _naive_iso_datetime
from bran_clinical_dictionary_binding_v1 import _quiet
from bran_clinical_semantics_v1 import (CBC_FIELDS, CANONICAL_UNITS, EICU_CBC_NAMES,
    NHANES_CBC_CODES, decode_age, nhanes_cbc_observation)
from bran_clinical_snapshot_v1 import assign_person_split
from bran_clinical_source_reader_v1 import iter_projected_csv, build_episode_links, _numeric_key, _opaque_key
from bran_nhanes_source_reader_v1 import iter_xport_projection, link_nhanes_cycle, nhanes_numeric_key
from run_bran_source_linkage_audit_v1 import INPUTS as LINK_INPUTS, sha, exclusive_json

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT/'BRAN_OBSERVED_CBC_POOL_PROTOCOL_V1.json'
PUBLIC = ROOT/'BRAN_OBSERVED_CBC_POOL_V1'
PRIVATE = ROOT/'private_artifacts'/'bran_observed_cbc_pool_v1'
SOURCES = ('nhanes', 'nwicu', 'eicu', 'mimic')
DICTIONARY = ROOT/'BRAN_LOCAL_CBC_DICTIONARY_BINDINGS_V1'/'bindings.json'
DICTIONARY_SHA = '1f8a230ff9ce79fca7cd568f88a1445965be931a895e47c449c51cdab3bd445e'
INPUTS = dict(LINK_INPUTS)
INPUTS.pop('sicdb_cases')
INPUTS.update({
    'mimic_labs':'/Users/ethanwu/mimiciv-3.1/hosp/labevents.csv.gz',
    'nwicu_labs':'/Users/ethanwu/nwicu-northwestern-icu-0.1.0/data/nw_hosp/labevents.csv.gz',
    'eicu_labs':'/Users/ethanwu/eicu-crd-2.0/lab.csv.gz',
})
CODE = ('run_bran_observed_cbc_pool_v1.py','bran_cbc_event_adapter_v1.py',
    'bran_clinical_dictionary_binding_v1.py','bran_clinical_semantics_v1.py',
    'bran_clinical_snapshot_v1.py','bran_clinical_source_reader_v1.py',
    'bran_nhanes_source_reader_v1.py','run_bran_source_linkage_audit_v1.py')
AGE_KINDS = ('missing_or_invalid','source_age_unresolved','reported_year','year_derived',
             'topcoded','rounded','rounded_topcoded')
STATUS = ('observation_pool_materialized','observation_scan_failed')
POLICY = {
    'minimum_observed_fields_per_snapshot':2,
    'max_metadata_rows':2000000,
    'max_lab_rows':200000000,
    'adult_qualification':'finite source-specific lower age bound >=18; unresolved remains unqualified, not fabricated',
    'icu_time_window':'earliest valid positive CBC in [0,1440] minutes; earliest per field in next 60 minutes capped at1440; conflicting first-timestamp ties masked',
    'nhanes_time_window':'one cycle-specific CBC examination; no fabricated ICU timestamp',
    'value_quality':'original numeric observations; finite strictly positive; exact approved units; no inequality coercion or upper cutoff; not clinical plausibility certification',
    'split':'source-local HMAC-person 80/10/10 with persistent private salt; never episode-random',
    'repeated_person_policy':'retain eligible episodes; cache person indices and inverse-episode weights summing to one per source-local person',
    'source_sampling':'not selected here; must be frozen before fitting; no population prevalence claim',
    'age_covariates':'preserve reported value, interval bounds and kind; do not create exact scalar for censored/coarsened/missing age',
    'eicu_assay_status':'exact source lab names plus per-row explicit compatible units; hospital-specific assay equivalence remains provisional',
    'nwicu_age_status':'unresolved; retained observations are not qualified adult training data',
    'nhanes_design':'unweighted observation pretraining candidate pool; no survey population or prevalence inference',
    'small_count_release':'all positive counts floored to multiples of20; values below20 suppressed; no exact totals or rates',
    'training_permitted':False,
    'patient_level_output_permitted':False,
    'retinal_data_read':False,
}


@dataclass(repr=False)
class PoolRecord:
    person: str
    values: np.ndarray
    observed: np.ndarray
    age: object


def source_keys(source):
    if source not in SOURCES: raise ValueError('unsupported source')
    return tuple(k for k in INPUTS if k.startswith(source+'_'))


def prepare():
    if sha(DICTIONARY)!=DICTIONARY_SHA: raise ValueError('dictionary authentication failed')
    return {'schema_version':'bran-observed-cbc-pool-protocol-v1',
        'policy':POLICY,'dictionary_sha256':DICTIONARY_SHA,
        'code_sha256':{n:sha(ROOT/n) for n in CODE},
        'source_files':{k:{'path':v,'sha256':sha(v)} for k,v in INPUTS.items()}}


def validate_protocol(p):
    if set(p)!={'schema_version','policy','dictionary_sha256','code_sha256','source_files'} or p['schema_version']!='bran-observed-cbc-pool-protocol-v1':
        raise ValueError('protocol schema mismatch')
    if p['policy']!=POLICY or p['dictionary_sha256']!=DICTIONARY_SHA or sha(DICTIONARY)!=DICTIONARY_SHA:
        raise ValueError('protocol policy mismatch')
    if set(p['code_sha256'])!=set(CODE) or set(p['source_files'])!=set(INPUTS): raise ValueError('protocol bindings mismatch')
    for n,h in p['code_sha256'].items():
        if sha(ROOT/n)!=h: raise ValueError('implementation authentication failed')
    for k,r in p['source_files'].items():
        if set(r)!={'path','sha256'} or r['path']!=INPUTS[k] or not valid_sha(r['sha256']): raise ValueError('input binding invalid')


def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def coarse_count(n):
    if type(n) is not int or n<0: raise ValueError('invalid count')
    return None if n<20 else n//20*20


def adult_qualified(age):
    return math.isfinite(age.lower_years) and age.lower_years>=18 and age.kind!='source_age_unresolved'


def pack_records(source, records, salt):
    """Private arrays only; no normalizer fit, no person-level result exported."""
    if source not in SOURCES or len(salt)<16: raise ValueError('invalid pool configuration')
    records=list(records)
    if not records: raise ValueError('no usable observations')
    persons=sorted({r.person for r in records})
    indices={p:i for i,p in enumerate(persons)}
    group=np.array([indices[r.person] for r in records],dtype=np.int64)
    splitmap={'train':0,'validation':1,'test':2}
    psplit=np.array([splitmap[assign_person_split(source,p,salt=salt)] for p in persons],dtype=np.uint8)
    counts=np.bincount(group,minlength=len(persons))
    values=np.stack([r.values for r in records]).astype(np.float64)
    observed=np.stack([r.observed for r in records]).astype(bool)
    if values.shape!=(len(records),9) or observed.shape!=values.shape: raise ValueError('invalid pool shape')
    if np.any(observed & (~np.isfinite(values) | (values<=0))) or np.any(observed.sum(1)<2): raise ValueError('invalid observed values')
    values[~observed]=np.nan
    ages=np.array([[r.age.reported_years,r.age.lower_years,r.age.upper_years] for r in records],dtype=np.float64)
    age_kind=np.array([AGE_KINDS.index(r.age.kind) for r in records],dtype=np.uint8)
    qualified=np.array([adult_qualified(r.age) for r in records],dtype=bool)
    arrays={'values':values,'observed':observed,'provenance':observed.astype(np.uint8),
        'age_triplet':ages,'age_kind':age_kind,'adult_qualified':qualified,
        'person_group':group,'split':psplit[group],'person_weight':1./counts[group]}
    summary={
        'snapshots':coarse_count(len(records)),
        'source_local_people':coarse_count(len(persons)),
        'adult_qualified_snapshots':coarse_count(int(qualified.sum())),
        'field_observed_snapshots':{f:coarse_count(int(observed[:,i].sum())) for i,f in enumerate(CBC_FIELDS)},
        'split_snapshots':{name:coarse_count(int((arrays['split']==v).sum())) for name,v in splitmap.items()},
    }
    return arrays,summary


def metadata(source):
    limit=POLICY['max_metadata_rows']
    if source in ('mimic','nwicu'):
        people=list(iter_projected_csv(INPUTS[source+'_patients'],('subject_id','anchor_age','anchor_year'),max_rows=limit))
        episodes=list(iter_projected_csv(INPUTS[source+'_admissions'],('subject_id','hadm_id','admittime'),max_rows=limit))
        links=build_episode_links(source,people,episodes)
        person_metadata={_numeric_key(r['subject_id']):r for r in people}
        admissions={}; ages={}
        for r in episodes:
            key=_numeric_key(r['hadm_id']); p=person_metadata[_numeric_key(r['subject_id'])]
            admissions[key]=r['admittime']
            dt=_naive_iso_datetime(r['admittime'])
            ages[key]=decode_age(source,p['anchor_age'],anchor_year=p['anchor_year'],admission_year=dt.year if dt else None)
        return links,admissions,ages
    if source=='eicu':
        rows=list(iter_projected_csv(INPUTS['eicu_patients'],('uniquepid','patientunitstayid','patienthealthsystemstayid','age'),max_rows=limit))
        links=build_episode_links(source,rows,())
        ages={_opaque_key(r['patientunitstayid']):decode_age(source,r['age']) for r in rows}
        return links,None,ages
    raise ValueError('unsupported metadata source')


def dictionary_mapping(source):
    if source=='eicu': return dict(EICU_CBC_NAMES)
    d=json.loads(DICTIONARY.read_text())
    fields=d['sources'][source]['fields']
    result={code:field for field,record in fields.items() for code in record['dictionary_bound_codes']}
    if not result or any(field not in CBC_FIELDS for field in result.values()): raise ValueError('invalid dictionary map')
    return result


def icu_records(source):
    links,admissions,ages=metadata(source)
    mapping=dictionary_mapping(source)
    if source in ('mimic','nwicu'):
        columns=('subject_id','hadm_id','itemid','valuenum','valueuom','charttime'); code_column='itemid'
    else:
        columns=('patientunitstayid','labname','labresult','labmeasurenamesystem','labresultoffset'); code_column='labname'
    panels=OnlineCBCPanels(source)
    for row in iter_projected_csv(INPUTS[source+'_labs'],columns,max_rows=POLICY['max_lab_rows']):
        if row[code_column] not in mapping: continue
        event=adapt_cbc_event(source,row,links,mapping,admissions)
        if event is not None: panels.add(event)
    for episode,person,snapshot in panels.iterate_snapshots():
        if int(snapshot.observed.sum())>=2:
            yield PoolRecord(person,snapshot.values,snapshot.observed,ages[episode])


def nhanes_records():
    for cycle in ('D','E'):
        suffix=cycle.lower(); limit=POLICY['max_metadata_rows']
        rows=list(iter_xport_projection(INPUTS['nhanes_cbc_'+suffix],('SEQN',)+tuple(NHANES_CBC_CODES),max_rows=limit))
        demos=iter_xport_projection(INPUTS['nhanes_demo_'+suffix],('SEQN','SDDSRVYR','RIDAGEYR'),max_rows=limit)
        links=link_nhanes_cycle(cycle,demos,rows)
        for row in rows:
            person=nhanes_numeric_key(row['SEQN'])
            values=np.full(9,np.nan); observed=np.zeros(9,bool)
            for code,field in NHANES_CBC_CODES.items():
                item=nhanes_cbc_observation(cycle,code,row[code],provenance=1)
                if item.observed and item.value>0:
                    j=CBC_FIELDS.index(field); values[j]=item.value; observed[j]=True
            if int(observed.sum())>=2: yield PoolRecord(person,values,observed,links.person_age[person])


def safe_payload(source, summary, cache_hash):
    return {'schema_version':'bran-observed-cbc-pool-aggregate-v1','source':source,
        'status':'observation_pool_materialized','counts_lower_bounds_20':summary,
        'private_cache_sha256':cache_hash,'patient_level_output_emitted':False,
        'raw_observations_processed_locally':True,'training_started':False,
        'training_ready':False,'new_model_benefit_established':False,
        'age_qualification_available':source!='nwicu',
        'assay_equivalence_provisional':source=='eicu',
        'population_inference_permitted':False,'cross_source_identity_resolved':False}


def validate_aggregate(p):
    expected={'schema_version','source','status','counts_lower_bounds_20','private_cache_sha256',
        'patient_level_output_emitted','raw_observations_processed_locally','training_started','training_ready',
        'new_model_benefit_established','age_qualification_available','assay_equivalence_provisional',
        'population_inference_permitted','cross_source_identity_resolved'}
    if set(p)!=expected or p['schema_version']!='bran-observed-cbc-pool-aggregate-v1' or p['source'] not in SOURCES or p['status']!=STATUS[0] or not valid_sha(p['private_cache_sha256']): raise ValueError('invalid aggregate schema')
    for k in ('patient_level_output_emitted','training_started','training_ready','new_model_benefit_established','population_inference_permitted','cross_source_identity_resolved'):
        if p[k] is not False: raise ValueError('invalid privacy or claim flag')
    if p['raw_observations_processed_locally'] is not True or p['age_qualification_available'] is not (p['source']!='nwicu') or p['assay_equivalence_provisional'] is not (p['source']=='eicu'): raise ValueError('invalid status flag')
    c=p['counts_lower_bounds_20']
    if set(c)!={'snapshots','source_local_people','adult_qualified_snapshots','field_observed_snapshots','split_snapshots'} or set(c['field_observed_snapshots'])!=set(CBC_FIELDS) or set(c['split_snapshots'])!={'train','validation','test'}: raise ValueError('invalid count keys')
    counts=[c[k] for k in ('snapshots','source_local_people','adult_qualified_snapshots')]+list(c['field_observed_snapshots'].values())+list(c['split_snapshots'].values())
    if any(v is not None and (type(v) is not int or v<20 or v%20) for v in counts): raise ValueError('invalid coarsened counts')


def run(source,p):
    validate_protocol(p)
    PUBLIC.mkdir(exist_ok=True)
    out=PUBLIC/source
    out.mkdir()  # Exclusive: never overwrite an existing success/failure attempt.
    phase='source_authentication'
    try:
        for k in source_keys(source):
            if sha(INPUTS[k])!=p['source_files'][k]['sha256']: raise ValueError('source authentication failed')
        PRIVATE.mkdir(parents=True,exist_ok=True,mode=0o700)
        os.chmod(PRIVATE,0o700)
        salt_path=PRIVATE/'split_salt.bin'
        if not salt_path.exists():
            with salt_path.open('xb') as h: h.write(secrets.token_bytes(32))
            os.chmod(salt_path,0o600)
        salt=salt_path.read_bytes()
        if len(salt)!=32: raise ValueError('private split salt invalid')
        phase='observation_scan_and_pack'
        records=nhanes_records() if source=='nhanes' else icu_records(source)
        arrays,summary=pack_records(source,records,salt)
        phase='private_cache_write'
        private_file=PRIVATE/(source+'.npz')
        with private_file.open('xb') as h: np.savez_compressed(h,**arrays)
        os.chmod(private_file,0o600)
        phase='post_scan_authentication'
        for k in source_keys(source):
            if sha(INPUTS[k])!=p['source_files'][k]['sha256']: raise ValueError('source changed during scan')
        phase='aggregate_validation_and_commit'
        payload=safe_payload(source,summary,sha(private_file)); validate_aggregate(payload)
        exclusive_json(out/'aggregate.json',payload)
        exclusive_json(out/'manifest.json',{'protocol_sha256':sha(PROTOCOL),'aggregate_sha256':sha(out/'aggregate.json'),
            'split_salt_sha256':sha(salt_path),'patient_level_output_emitted':False,'training_started':False})
        return True
    except Exception:
        exclusive_json(out/'failure.json',{'status':'observation_scan_failed','phase':phase,'patient_level_output_emitted':False,'training_started':False})
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-protocol',action='store_true')
    parser.add_argument('--source',choices=SOURCES)
    args=parser.parse_args()
    if args.prepare_protocol==bool(args.source): parser.error('choose exactly one operation')
    ok=False
    with _quiet():
        try:
            if args.prepare_protocol: exclusive_json(PROTOCOL,prepare()); ok=True
            else: ok=run(args.source,json.loads(PROTOCOL.read_text()))
        except Exception: pass
    print(json.dumps({'status':('protocol_prepared' if args.prepare_protocol else 'source_observation_pool_completed') if ok else 'observation_pool_execution_failed',
        'source':args.source,'patient_level_output_emitted':False,'training_started':False}))
    return 0 if ok else 1


if __name__=='__main__': raise SystemExit(main())
