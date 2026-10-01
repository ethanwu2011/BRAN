"""Versioned mixed-record parser correction, retaining failed V1 unchanged."""
import argparse
import csv
import fcntl
import io
import json

import bran_odir_id_adapter_v2 as k
import run_bran_odir_metadata_qualification_v1 as old

r = old.r
ROOT = old.ROOT
PROTOCOL = ROOT / 'BRAN_ODIR_METADATA_QUALIFICATION_PROTOCOL_V2.json'
OUT = ROOT / 'BRAN_ODIR_METADATA_QUALIFICATION_V2'
AUDIT = ROOT / 'BRAN_ODIR_METADATA_QUALIFICATION_AUDIT_V2'
PRIVATE = ROOT / 'private_artifacts/bran_odir_metadata_qualification_v2'
LOCK = old.LOCK
PREVIOUS = 'f183520bf75e49b8e79f5d026303bdc322c65082c6079046f9e7e844eb79cc46'
SCHEMA = 'bran-odir-metadata-qualification-v2'
FILES = ('run_bran_odir_metadata_qualification_v2.py', 'bran_odir_id_adapter_v2.py',
         'test_run_bran_odir_metadata_qualification_v2.py', 'test_bran_odir_id_adapter_v2.py',
         'BRAN_ODIR_METADATA_QUALIFICATION_DESIGN_V2.md')


def prior():
    p = old.protocol(PREVIOUS)
    r.inventory(old.OUT, ('failure.json',))
    r.inventory(old.PRIVATE, ('source_locator.json',), private=True)
    r.require(r.equal(json.loads((old.OUT/'failure.json').read_text()),
              {'status':'failed', 'action':'run', 'patient_level_output_emitted':False}))
    return p


def template():
    p = prior()
    return {'schema':SCHEMA, 'status':'frozen_before_mixed_record_qualification',
            'previous_protocol_sha256':PREVIOUS, 'previous_failure_sha256':r.sha(old.OUT/'failure.json'),
            'source':p['source'], 'private_locator_sha256':r.sha(PRIVATE/'source_locator.json'),
            'code_sha256':{name:r.sha(ROOT/name) for name in FILES}, 'runtime':r.runtime(),
            'training_admitted':False, 'missing_id_rows_ineligible':True}


def protocol(pin):
    r.regular(PROTOCOL); r.regular(PRIVATE/'source_locator.json', private=True)
    r.require(type(pin) is str and len(pin)==64 and r.sha(PROTOCOL)==pin)
    value=json.loads(PROTOCOL.read_text()); r.require(r.equal(value,template()))
    return value


def evaluate(pin):
    p=protocol(pin)
    chosen=json.loads((PRIVATE/'source_locator.json').read_text())
    raw,members,receipt=old.source(chosen); r.require(r.equal(receipt,p['source']))
    reader=csv.reader(io.StringIO(raw.decode('utf-8-sig'),newline=''),strict=True)
    header=next(reader); indexes=[header.index(field) for field in old.FIELDS]
    rows=[]
    for row in reader:
        r.require(len(row)==len(header) and len(rows)<100000)
        rows.append({key:row[index] for key,index in zip(old.FIELDS,indexes)})
    private,aggregate=k.qualify(rows,members); k.validate_aggregate(aggregate)
    r.require(aggregate['identified_record_qualification']['counts_rounded_down20']['grouped_patients']>=20)
    return private,aggregate


def result(pin,aggregate):
    k.validate_aggregate(aggregate)
    return {'schema':SCHEMA,'status':'metadata_qualification_completed','protocol_sha256':pin,
            'qualification':aggregate,'private_records_sha256':r.sha(PRIVATE/'grouped_records.json'),
            'image_payload_authenticated':False,'official_split_authenticated':False,
            'training_admitted':False,'patient_level_output_emitted':False}


def authenticate(pin):
    protocol(pin); r.inventory(OUT,('aggregate.json','manifest.json'))
    r.inventory(PRIVATE,('source_locator.json','grouped_records.json'),private=True)
    value=json.loads((OUT/'aggregate.json').read_text())
    r.require(r.equal(value,result(pin,value['qualification'])))
    r.require(r.equal(json.loads((OUT/'manifest.json').read_text()),
              {'protocol_sha256':pin,'aggregate_sha256':r.sha(OUT/'aggregate.json')}))
    return value


def audit_value(pin):
    return {'schema':SCHEMA,'status':'authenticated','protocol_sha256':pin,
            'aggregate_sha256':r.sha(OUT/'aggregate.json'),'mixed_record_qualification_replayed':True,
            'csv_bytes_authenticated':True,'zip_directory_metadata_authenticated':True,
            'image_payload_authenticated':False,'original_release_authenticated':False,
            'training_admitted':False,'patient_level_output_emitted':False}


def authenticate_audit(pin,audit_pin):
    authenticate(pin); r.inventory(AUDIT,('audit.json','manifest.json'))
    r.require(r.sha(AUDIT/'audit.json')==audit_pin and
              r.equal(json.loads((AUDIT/'audit.json').read_text()),audit_value(pin)))
    r.require(r.equal(json.loads((AUDIT/'manifest.json').read_text()),
              {'protocol_sha256':pin,'audit_sha256':audit_pin}))


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','run','audit','verify'))
    parser.add_argument('--protocol-sha256'); parser.add_argument('--audit-sha256')
    args=parser.parse_args(argv)
    answer={'status':'failed','action':args.action,'patient_level_output_emitted':False}
    owned=None
    with r.quiet():
        try:
            with LOCK.open('a') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                pin=args.protocol_sha256
                if args.action=='prepare':
                    p=prior(); r.absent(PROTOCOL,OUT,AUDIT,PRIVATE)
                    chosen=json.loads((old.PRIVATE/'source_locator.json').read_text())
                    _,_,receipt=old.source(chosen); r.require(r.equal(receipt,p['source']))
                    PRIVATE.mkdir(mode=0o700); old.private_json(PRIVATE/'source_locator.json',chosen)
                    r.write_json(PROTOCOL,template()); pin=r.sha(PROTOCOL)
                elif args.action=='run':
                    protocol(pin); r.absent(OUT,AUDIT,PRIVATE/'grouped_records.json')
                    OUT.mkdir(); owned=OUT
                    private,aggregate=evaluate(pin); protocol(pin)
                    old.private_json(PRIVATE/'grouped_records.json',private)
                    r.write_json(OUT/'aggregate.json',result(pin,aggregate))
                    r.write_json(OUT/'manifest.json',{'protocol_sha256':pin,'aggregate_sha256':r.sha(OUT/'aggregate.json')})
                    authenticate(pin)
                elif args.action=='audit':
                    original=authenticate(pin); r.absent(AUDIT); AUDIT.mkdir(); owned=AUDIT
                    private,aggregate=evaluate(pin)
                    r.require(r.equal(json.loads((PRIVATE/'grouped_records.json').read_text()),private)
                              and r.equal(result(pin,aggregate),original) and r.equal(authenticate(pin),original))
                    r.write_json(AUDIT/'audit.json',audit_value(pin))
                    r.write_json(AUDIT/'manifest.json',{'protocol_sha256':pin,'audit_sha256':r.sha(AUDIT/'audit.json')})
                    authenticate_audit(pin,r.sha(AUDIT/'audit.json'))
                elif args.audit_sha256: authenticate_audit(pin,args.audit_sha256)
                else: authenticate(pin)
                answer.update(status='complete',protocol_sha256=pin,training_admitted=False)
                if args.action!='prepare': answer['aggregate_sha256']=r.sha(OUT/'aggregate.json')
                if args.action=='audit' or args.audit_sha256: answer['audit_sha256']=r.sha(AUDIT/'audit.json')
        except Exception:
            if owned is not None:
                try: r.write_json(owned/'failure.json',answer)
                except Exception: pass
    print(json.dumps(answer,sort_keys=True))
    return int(answer['status']!='complete')


if __name__=='__main__': raise SystemExit(main())
