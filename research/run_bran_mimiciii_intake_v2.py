"""Local checksum/header intake; never reads parsed patient rows or fits models.

Hashing verifies supplied bytes against the supplied manifest, not publisher
authenticity. An observed-lab/unit/age adapter and source admission remain needed.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

from run_bran_multisource_retinal_features_v2 import LOCK, quiet
from run_bran_multisource_inventory_v2 import write_exclusive
import fcntl

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/Volumes/Extreme/mimic-iii')
REQUIRED={
    'PATIENTS.csv.gz':{'SUBJECT_ID','DOB'},
    'ADMISSIONS.csv.gz':{'SUBJECT_ID','HADM_ID','ADMITTIME'},
    'LABEVENTS.csv.gz':{'SUBJECT_ID','HADM_ID','ITEMID','CHARTTIME','VALUENUM','VALUEUOM'},
    'D_LABITEMS.csv.gz':{'ITEMID','LABEL','FLUID','CATEGORY'},
}


def require(ok):
    if not ok:raise ValueError('mimiciii_intake_contract_failed')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def inspect(source):
    manifest=source/'SHA256SUMS.txt';manifest_hash=sha(manifest)
    expected={}
    for line in manifest.read_text().splitlines():
        pieces=line.split()
        if len(pieces)==2 and pieces[1] in set(REQUIRED)|{'LICENSE.txt'}:
            require(pieces[1] not in expected and re.fullmatch('[0-9a-f]{64}',pieces[0]) is not None)
            expected[pieces[1]]=pieces[0]
    require(set(expected)==set(REQUIRED)|{'LICENSE.txt'})
    authenticated={}
    for name in sorted(expected):
        path=source/name
        require(path.is_file() and not path.is_symlink())
        stat=path.stat();actual=sha(path);require(actual==expected[name])
        if name in REQUIRED:
            with gzip.open(path,'rt',newline='') as stream:
                header=next(csv.reader([stream.readline(8192)]))
            require(len(header)==len(set(header)) and REQUIRED[name].issubset(set(header)))
        require(path.stat().st_size==stat.st_size and path.stat().st_mtime_ns==stat.st_mtime_ns)
        authenticated[name]={'sha256':actual,'required_header_passed':name in REQUIRED}
    require(sha(manifest)==manifest_hash)
    return {'schema':'bran-mimiciii-intake-v2','status':'local_bytes_and_required_headers_verified',
        'manifest_sha256':manifest_hash,'files':authenticated,
        'publisher_manifest_independently_authenticated':False,
        'release_version_authenticated':False,'lab_units_bound':False,
        'observed_patient_rows_parsed':False,'observation_cache_materialized':False,
        'v2_training_admitted':False,'source_family':'mimic',
        'cross_release_identity_resolved':False,'patient_level_output_emitted':False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'BRAN_MIMICIII_INTAKE_V2_ATTEMPT1'/'aggregate.json')
    args=parser.parse_args();ok=False
    with quiet():
        try:
            require(not args.output.exists())
            with LOCK.open('a') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                result=inspect(SOURCE);write_exclusive(args.output,result);ok=True
        except Exception:pass
    print(json.dumps({'status':'local_bytes_and_required_headers_verified' if ok else 'intake_failed',
        'v2_training_admitted':False,'patient_level_output_emitted':False}))
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
