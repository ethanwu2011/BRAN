"""Exclusive, FD-quiet local INSPIRE source qualification, no model fitting."""
import argparse
import csv
from dataclasses import fields
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile
import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
import bran_inspire_source_contract_v1 as source

ROOT=Path(__file__).resolve().parent
ARCHIVE=Path('/Users/ethanwu/Downloads/inspire-a-publicly-available-research-dataset-for-perioperative-medicine-1.4.2.zip')
LOCK=Path('/private/tmp/bran_retinal_extraction_v1.lock')
PREFIX='inspire-a-publicly-available-research-dataset-for-perioperative-medicine-1.4.2/'
METADATA_PINS={'schema.csv':'0a58162f9d11a56e1e54112dd1b3737d34f0726c276a5c22604d1a2502101934',
 'parameters.csv':'6d657d12925016888002f4b0428ce374a75c1704d332e5a9c8242eed4f51fdaa'}
MODEL_UNIT_PIN='4d428667185a974116b0f13d525bcf86c9837ea86806604b26f6f0c41184e167'
MEMBERS=tuple(METADATA_PINS)+('operations.csv.gz','labs.csv.gz')
CODE=('bran_inspire_source_contract_v1.py','test_bran_inspire_source_contract_v1.py',
 'run_bran_inspire_source_qualification_v1.py','test_run_bran_inspire_source_qualification_v1.py',
 'run_bran_inspire_source_qualification_v1_attempt1.sh','BRAN_INSPIRE_ADAPTATION_V1_DESIGN.md',
 'bran_clinical_dictionary_binding_v1.py','patient_atlas_official_unit_reconciliation.py')
ERROR='bran_inspire_source_qualification_v1_failed'
PHASES=('authentication','operations','laboratories','private_write','post_authentication','completed')


def require(ok):
    if not ok:raise ValueError(ERROR)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,data):
    with Path(path).open('x',encoding='utf-8') as f:
        os.fchmod(f.fileno(),0o600);json.dump(data,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def paths(attempt):
    require(type(attempt) is int and 1<=attempt<=99)
    return ROOT/f'BRAN_INSPIRE_SOURCE_V1_ATTEMPT{attempt}',ROOT/'private_artifacts'/f'bran_inspire_source_v1_attempt{attempt}'


def authenticate_model_units():
    import patient_atlas_official_unit_reconciliation as units
    path=ROOT/'PATIENT_ATLAS_OFFICIAL_UNIT_RECONCILIATION_V1.json'
    require(not path.is_symlink() and sha(path)==MODEL_UNIT_PIN)
    report=read(path);units.validate_official_unit_reconciliation(report)
    require(sha(path)==MODEL_UNIT_PIN)
    names=('albumin','alkaline_phosphatase','alt_got','ast_got','bun','chloride','creatinine','crp_hs','glucose',
        'hemoglobin','hba1c','hct','plt','potassium','sodium','bilirubin_total','protein_total','wbc')
    expected=('g/dL','IU/L','IU/L','IU/L','mg/dL','mEq/L','mg/dL','mg/L','mg/dL','g/dL','%','%',
        '10^3/uL','mEq/L','mEq/L','mg/dL','g/dL','10^3/uL')
    selected={f['name']:f for f in report['fields'] if f['name'] in names}
    require(set(selected)==set(names))
    for name,unit in zip(names,expected):
        require(selected[name]['canonical_unit_authorized'] is True and selected[name]['canonical_unit']==unit)
    return {'unit_reconciliation_sha256':MODEL_UNIT_PIN,'canonical_units':dict(zip(names,expected)),
        'official_lab_metadata_sha256':report['source_bindings']['clinical_lab_json_sha256']}


def _member_bytes(z,name):
    require(z.namelist().count(PREFIX+name)==1)
    return z.read(PREFIX+name)


def metadata_receipt(z):
    blobs={n:_member_bytes(z,n) for n in METADATA_PINS}
    require(all(hashlib.sha256(v).hexdigest()==METADATA_PINS[n] for n,v in blobs.items()))
    schema=list(csv.DictReader(io.StringIO(blobs['schema.csv'].decode('utf-8-sig'),newline='')))
    table=None;columns={}
    for row in schema:
        if row['Table']:table=row['Table']
        require(table is not None);columns.setdefault(table,[]).append(row['Variable'])
    require(set(source.OP_COLUMNS).issubset(columns['operations']) and set(source.LAB_COLUMNS)==set(columns['labs']))
    parameters=list(csv.DictReader(io.StringIO(blobs['parameters.csv'].decode('utf-8-sig'),newline='')))
    units={}
    for row in parameters:
        if row['Table']=='labs' and row['Label'] in source.SOURCE_FIELDS:
            require(row['Label'] not in units);units[row['Label']]=row['Unit']
    require(units==dict(zip(source.SOURCE_FIELDS,source.SOURCE_UNITS)))
    return {'metadata_sha256':dict(METADATA_PINS),'laboratory_units':units,'columns':columns}


def authenticate_archive(archive):
    require(archive.is_file() and not archive.is_symlink())
    before=(archive.stat().st_size,archive.stat().st_mtime_ns)
    with zipfile.ZipFile(archive) as z:
        metadata=metadata_receipt(z);pins={}
        for name in MEMBERS:
            require(z.namelist().count(PREFIX+name)==1)
            info=z.getinfo(PREFIX+name)
            require(not info.is_dir() and 0<info.file_size<512*1024*1024)
            h=hashlib.sha256()
            with z.open(PREFIX+name) as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            pins[name]=h.hexdigest()
    require(before==(archive.stat().st_size,archive.stat().st_mtime_ns))
    return {'schema':'bran-inspire-source-receipt-v1','source_version':'1.4.2',
        'source_archive_name':archive.name,'member_sha256':pins,'metadata':metadata}


def row_iterator(z,name,allowed,required,max_rows):
    # Values never escape this private local iterator into logs or a hosted tool.
    with (z.open(PREFIX+name) as compressed, gzip.GzipFile(fileobj=compressed) as gz,
          io.TextIOWrapper(gz,encoding='utf-8-sig',newline='') as stream):
        reader=csv.DictReader(stream,strict=True)
        require(reader.fieldnames is not None and len(set(reader.fieldnames))==len(reader.fieldnames)
            and set(reader.fieldnames).issubset(allowed) and set(required).issubset(reader.fieldnames))
        for count,row in enumerate(reader,1):
            require(count<=max_rows and None not in row and all(row[k] is not None for k in required))
            yield {k:row[k] for k in required}


def assemble(archive,receipt):
    with zipfile.ZipFile(archive) as z:
        require(metadata_receipt(z)==receipt['metadata'])
        f=source.select_operations(row_iterator(z,'operations.csv.gz',receipt['metadata']['columns']['operations'],source.OP_COLUMNS,500000))
        f=source.fill_labs(f,row_iterator(z,'labs.csv.gz',receipt['metadata']['columns']['labs'],source.LAB_COLUMNS,50000000))
    return f


def validate_result(result):
    require(set(result)=={'schema','support','private_sha256','source_replay_equal','source_receipt_sha256',
        'patient_level_output_emitted','model_inference_performed','external_performance_established'})
    require(result['schema']=='bran-inspire-source-aggregate-v1' and result['source_replay_equal'] is True)
    require(all(result[k] is False for k in ('patient_level_output_emitted','model_inference_performed','external_performance_established')))
    support=result['support']
    require(set(support)=={'schema','role_support_met','source_task','performance_evaluated','patient_level_output_emitted','clinical_use','flow'})
    require(support['schema']=='bran-inspire-source-support-v1' and type(support['role_support_met']) is bool
        and support['source_task']=='external_adaptation_not_fitted' and support['performance_evaluated'] is False
        and support['patient_level_output_emitted'] is False and support['clinical_use'] is False)
    flow=support['flow']
    if flow['status']=='released_lower_bounds':
        require(set(flow)=={'status','columns','roles','lower_bounds'})
        require(flow['roles']==['fit','calibration','test'] and flow['columns']==['chronology_ineligible','unknown_outcome','empty_physiology','eligible_surviving_discharge','eligible_hospital_death'])
        require(len(flow['lower_bounds'])==3 and all(len(r)==5 and all(type(v) is int and v>=20 and v%20==0 for v in r) for r in flow['lower_bounds']))
    else:require(flow=={'status':'suppressed_complement_support'})
    for k in ('private_sha256','source_receipt_sha256'):
        require(type(result[k]) is str and len(result[k])==64 and all(c in '0123456789abcdef' for c in result[k]))


def read(path):return json.loads(path.read_text())


def validate_protocol(protocol):
    require(set(protocol)=={'schema','source_receipt','code_sha256','scientific_design_sha256',
        'model_units','source_role','encoder_inference','patient_level_output_emitted'})
    require(protocol['schema']=='bran-inspire-source-protocol-v1'
        and protocol['source_role']=='protected_encoder_source_external_adaptation_candidate'
        and protocol['encoder_inference'] is False and protocol['patient_level_output_emitted'] is False)
    require(protocol['code_sha256']=={n:sha(ROOT/n) for n in CODE})
    require(protocol['scientific_design_sha256']==sha(ROOT/'BRAN_INSPIRE_ADAPTATION_V1_DESIGN.md'))


def audit(attempt):
    out,private=paths(attempt)
    require(out.is_dir() and not out.is_symlink() and private.is_dir() and not private.is_symlink())
    require({p.name for p in out.iterdir()}=={'protocol.json','aggregate.json','completed.json'})
    require({p.name for p in private.iterdir()}=={'frame.npz'})
    require(not private.parent.is_symlink())
    require(all(p.is_file() and not p.is_symlink() and p.stat().st_nlink==1 for p in out.iterdir()))
    terminal=read(out/'completed.json');protocol=read(out/'protocol.json');result=read(out/'aggregate.json')
    require(terminal=={'status':'completed','protocol_sha256':sha(out/'protocol.json'),'aggregate_sha256':sha(out/'aggregate.json'),'patient_level_output_emitted':False})
    validate_result(result)
    validate_protocol(protocol)
    require(protocol['source_receipt']==authenticate_archive(ARCHIVE))
    require(protocol['model_units']==authenticate_model_units())
    require(result['source_receipt_sha256']==hashlib.sha256(json.dumps(protocol['source_receipt'],sort_keys=True).encode()).hexdigest())
    f=private/'frame.npz'
    require(not f.is_symlink() and f.stat().st_nlink==1 and f.stat().st_mode&0o777==0o600
        and private.stat().st_mode&0o777==0o700 and sha(f)==result['private_sha256'])
    return {'status':'authenticated','role_support_met':result['support']['role_support_met'],
        'protocol_sha256':terminal['protocol_sha256'],'aggregate_sha256':terminal['aggregate_sha256'],
        'model_inference_performed':False,'patient_level_output_emitted':False}


def run(attempt):
    out,private=paths(attempt)
    require(not out.exists() and not out.is_symlink() and not private.exists() and not private.is_symlink())
    require(private.parent.is_dir() and not private.parent.is_symlink())
    phase='authentication'
    with LOCK.open('a') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'not_started_shared_lock_busy'}
        out.mkdir(mode=0o700);private.mkdir(mode=0o700)
        try:
            with _quiet():
                receipt=authenticate_archive(ARCHIVE)
                protocol={'schema':'bran-inspire-source-protocol-v1','source_receipt':receipt,
                    'code_sha256':{n:sha(ROOT/n) for n in CODE},'scientific_design_sha256':sha(ROOT/'BRAN_INSPIRE_ADAPTATION_V1_DESIGN.md'),
                    'model_units':authenticate_model_units(),
                    'source_role':'protected_encoder_source_external_adaptation_candidate','encoder_inference':False,
                    'patient_level_output_emitted':False}
                validate_protocol(protocol)
                write(out/'protocol.json',protocol)
            print(json.dumps({'phase':'operations','status':'started','patient_level_output_emitted':False}),flush=True)
            phase='operations'
            with _quiet():
                f=assemble(ARCHIVE,receipt)
                # Replay the full source assembly, not only an artifact reload.
                phase='laboratories';replay=assemble(ARCHIVE,receipt)
                require(all(np.array_equal(getattr(f,k.name),getattr(replay,k.name),equal_nan=True)
                    if getattr(f,k.name).dtype.kind=='f' else np.array_equal(getattr(f,k.name),getattr(replay,k.name))
                    for k in fields(source.PrivateInspireFrame)))
                support=source.safe_support(f);phase='private_write'
                arrays={k.name:getattr(f,k.name) for k in fields(source.PrivateInspireFrame)}
                with (private/'frame.npz').open('xb') as handle:
                    os.fchmod(handle.fileno(),0o600);np.savez_compressed(handle,**arrays)
                pin=sha(private/'frame.npz')
                with np.load(private/'frame.npz',allow_pickle=False) as saved:
                    require(set(saved.files)==set(arrays))
                    require(all(np.array_equal(saved[k],v,equal_nan=True) if v.dtype.kind=='f' else np.array_equal(saved[k],v)
                        for k,v in arrays.items()))
                require(pin==sha(private/'frame.npz'));phase='post_authentication'
                require(receipt==authenticate_archive(ARCHIVE))
                require(protocol['model_units']==authenticate_model_units())
                validate_protocol(protocol)
                result={'schema':'bran-inspire-source-aggregate-v1','support':support,'private_sha256':pin,
                    'source_replay_equal':True,'source_receipt_sha256':hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest(),
                    'patient_level_output_emitted':False,'model_inference_performed':False,'external_performance_established':False}
                validate_result(result);write(out/'aggregate.json',result)
                write(out/'completed.json',{'status':'completed','protocol_sha256':sha(out/'protocol.json'),
                    'aggregate_sha256':sha(out/'aggregate.json'),'patient_level_output_emitted':False})
            return {'status':'completed','role_support_met':support['role_support_met'],'model_inference_performed':False,'patient_level_output_emitted':False}
        except Exception:
            with _quiet():
                if not (out/'completed.json').exists():
                    write(out/'failure.json',{'status':'failed','phase':phase if phase in PHASES else 'authentication',
                        'error':ERROR,'private_artifacts_admitted':False,'patient_level_output_emitted':False})
            return {'status':'failed','phase':phase if phase in PHASES else 'authentication','error':ERROR,'patient_level_output_emitted':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,required=True);p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    try:
        if a.audit_only:
            with _quiet():result=audit(a.attempt)
        else:result=run(a.attempt)
    except Exception:result={'status':'not_completed','error':ERROR,'patient_level_output_emitted':False}
    print(json.dumps(result,sort_keys=True),flush=True)
    return 0 if result['status'] in ('completed','authenticated') else 1


if __name__=='__main__':raise SystemExit(main())
