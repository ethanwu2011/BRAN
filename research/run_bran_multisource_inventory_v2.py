"""Metadata-only source inventory and legacy aggregate authentication.

Never scans patient directories, opens archives/images/arrays, or samples rows.
Checks only named source roots/table files and known aggregate-safe legacy JSON.
An inventory is not a V2 admission receipt and cannot authorize training.
"""
import argparse
import hashlib
import json
from pathlib import Path

from bran_multisource_protocol_v2 import pending_receipts, readiness, SOURCE_POLICY

ROOT = Path(__file__).resolve().parent
KNOWN_LOCATIONS = {
    'mimiciii': (Path('/Volumes/Extreme/mimic-iii'),
        ('PATIENTS.csv.gz','ADMISSIONS.csv.gz','LABEVENTS.csv.gz','D_LABITEMS.csv.gz','LICENSE.txt')),
    'mimiciv': (Path('/Users/ethanwu/mimiciv-3.1'), ()),
    'hirid': (Path('/Volumes/Extreme/hirid-1.1.1'), ('raw_stage',)),
    'inspire': (Path('/Users/ethanwu/Downloads/inspire-a-publicly-available-research-dataset-for-perioperative-medicine-1.4.2.zip'), ()),
    'zigong': (Path('/Users/ethanwu/icu-infection-zigong-fourth-1.1/DataTables.zip'), ()),
    'fd3611': (Path('/Users/ethanwu/Downloads/FD3611.zip'), ()),
    'trihemo_mcv': (Path('/Users/ethanwu/Downloads/TriHemo-MCV.zip'), ()),
}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def verify_legacy_native_metadata(root=ROOT):
    """Authenticate published aggregate/manifest links without opening caches.

    Deliberately not a requalification: no claim that private caches are current
    or that the legacy finite-age eligibility covers V2's missing-age inputs.
    """
    out=root/'BRAN_NATIVE_SOURCE_QUALIFICATION_V1'
    audit_dir=root/'BRAN_NATIVE_SOURCE_QUALIFICATION_AUDIT_V1'
    try:
        if (out/'failure.json').exists() or (audit_dir/'failure.json').exists():
            return {'status':'preserved_failure','v2_admitted':False}
        protocol=root/'BRAN_NATIVE_SOURCE_QUALIFICATION_PROTOCOL_V1.json'
        aggregate=out/'aggregate.json'; manifest=out/'manifest.json'; audit=audit_dir/'audit.json'
        if not all(p.is_file() for p in (protocol,aggregate,manifest,audit)):
            return {'status':'not_available','v2_admitted':False}
        # These are pre-existing, known aggregate-only artifacts; no arrays.
        m=json.loads(manifest.read_text()); a=json.loads(audit.read_text()); r=json.loads(aggregate.read_text())
        valid=(a.get('schema')=='bran-native-source-qualification-audit-v1'
            and a.get('status')=='authenticated' and r.get('status')=='completed'
            and a.get('patient_level_output_emitted') is False
            and r.get('patient_level_output_emitted') is False
            and m.get('patient_level_output_emitted') is False
            and a.get('protocol_sha256')==m.get('protocol_sha256')==sha(protocol)
            and a.get('aggregate_sha256')==m.get('aggregate_sha256')==sha(aggregate)
            and a.get('manifest_sha256')==sha(manifest))
        return {'status':'legacy_artifact_links_authenticated' if valid else 'authentication_failed',
                'v2_admitted':False, 'private_caches_reauthenticated':False,
                'patient_level_output_emitted':False}
    except Exception:
        return {'status':'authentication_failed','v2_admitted':False}


def inventory(root=ROOT, locations=KNOWN_LOCATIONS):
    entries={}
    for source in SOURCE_POLICY:
        if source=='amsterdamumcdb':
            entries[source]={'status':'deferred_not_searched'}
        elif source in locations:
            path,names=locations[source]
            entries[source]={'status':'named_location_checked', 'exists':path.exists(),
                'named_requirements_present':all((path/n).exists() for n in names),
                'contents_opened':False, 'qualified':False}
        else:
            entries[source]={'status':'existing_adapter_review_required','qualified':False}
    receipts=pending_receipts()
    return {'schema':'bran-multisource-inventory-v2', 'status':'metadata_inventory_completed',
        'sources':entries, 'admission_ledger':receipts, 'readiness':readiness(receipts),
        'legacy_native_metadata':verify_legacy_native_metadata(root),
        'new_training_started':False, 'scientific_results_present':False,
        'patient_level_output_emitted':False}


def write_exclusive(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(content,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'BRAN_MULTISOURCE_V2_INTAKE_ATTEMPT1'/'inventory.json')
    args=parser.parse_args()
    try:
        result=inventory(); write_exclusive(args.output,result)
        # Closed console output: never echo contents, arbitrary keys or errors.
        print(json.dumps({'status':result['status'],
            'legacy_native_metadata':result['legacy_native_metadata']['status'],
            'v2_training_ready':result['readiness']['source_ready'],
            'patient_level_output_emitted':False}))
        return 0
    except Exception:
        print('{"status":"inventory_execution_failed","patient_level_output_emitted":false}')
        return 1


if __name__=='__main__': raise SystemExit(main())
