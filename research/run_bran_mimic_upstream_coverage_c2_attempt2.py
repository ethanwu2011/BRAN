"""Versioned technical correction: normalize string source paths for SHA helper.

Attempt1 and its bound code remain unchanged. No coverage rule or data contract
changes. Import this module only in the dedicated attempt2 process or tests.
"""
import argparse
from pathlib import Path
import run_bran_mimic_upstream_coverage_c2 as original

EXTRA=('run_bran_mimic_upstream_coverage_c2_attempt2.py',
       'test_bran_mimic_upstream_coverage_c2_attempt2.py',
       'run_bran_mimic_upstream_coverage_c2_attempt2.sh',
       'BRAN_MIMIC_UPSTREAM_COVERAGE_C2_ATTEMPT2_TECHNICAL_NOTE.md')
_base_code_hashes=original.code_hashes


def source_hashes(pins):
    import run_bran_mimic_landmark_linkage_v1 as linked
    original.require(set(pins['source_sha256'])==set(linked.INPUTS))
    for key,path in linked.INPUTS.items():
        original.require(original.sha(Path(path))==pins['source_sha256'][key])


def code_hashes():
    return {**_base_code_hashes(),**{name:original.sha(original.ROOT/name) for name in EXTRA}}


def install():
    original.source_hashes=source_hashes
    original.code_hashes=code_hashes


def authenticate(attempt=2):
    original.require(type(attempt) is int and attempt==2)
    install()
    return original.authenticate(attempt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,choices=[2],required=True)
    p.add_argument('--audit-only',action='store_true');p.parse_args()
    install()
    raise SystemExit(original.main())
