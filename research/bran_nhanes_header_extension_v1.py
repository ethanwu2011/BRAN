"""Bounded XPORT header extension; no record counting or observation reading.

The pandas implementation's normal record count can inspect its final data
record. This audit-only subclass disables that method and all observation APIs.
It must never be reused as a data reader or to report participant counts.
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys

from pandas.io.sas.sas_xport import XportReader
from bran_six_source_schema_preflight_v1 import check_header, summarize_source

ROOT = Path(__file__).resolve().parent
PRIOR = 'BRAN_SIX_SOURCE_SCHEMA_PREFLIGHT_V1'
PRIOR_PIN = '60ead462d8c2a4bbc03ec07fbe3250704fc60d7b0103cdf1d89c01f62826bb9f'
OUT = 'BRAN_SIX_SOURCE_SCHEMA_PREFLIGHT_V2'
CBC_ROLES = {'person_key': ['SEQN'], 'hemoglobin': ['LBXHGB'], 'hct': ['LBXHCT'],
             'rbc': ['LBXRBCSI'], 'wbc': ['LBXWBCSI'], 'plt': ['LBXPLTSI'],
             'mcv': ['LBXMCVSI'], 'mch': ['LBXMCHSI'], 'mchc': ['LBXMC'], 'rdw': ['LBXRDW']}
DEMO_ROLES = {'person_key': ['SEQN'], 'age': ['RIDAGEYR'], 'cycle': ['SDDSRVYR']}


class HeaderOnlyXportReader(XportReader):
    def _record_count(self):
        return 0  # Uncomputed sentinel, never a measured number of records.

    def read(self, *args, **kwargs):
        raise RuntimeError('observation_read_forbidden')

    def get_chunk(self, *args, **kwargs):
        raise RuntimeError('observation_read_forbidden')


def inspect_xpt_header(path, roles):
    if not Path(path).is_file():
        return {'status': 'not_present_at_declared_path', 'patient_rows_read': False}
    try:
        with HeaderOnlyXportReader(path) as reader:
            result = check_header(','.join(reader.columns), roles)
        return result
    except Exception:
        return {'status': 'header_unreadable_or_invalid', 'patient_rows_read': False}


def build(root=ROOT):
    root = Path(root)
    out = root / OUT
    if out.exists():
        raise ValueError('output_already_exists')
    manifest_raw = (root / PRIOR / 'manifest.json').read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != PRIOR_PIN:
        raise ValueError('prior_manifest_changed')
    manifest = json.loads(manifest_raw)
    raw = (root / PRIOR / 'schema.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['files']['schema.json']:
        raise ValueError('prior_artifact_changed')
    result = json.loads(raw)
    result['schema'] = 'bran-six-source-schema-preflight-v2'
    result['prior_header_manifest_sha256'] = PRIOR_PIN
    nhanes = Path('/Users/ethanwu/nhanes-oculomics')
    result['sources']['nhanes'] = summarize_source({f'{stem}_{cycle}.xpt': inspect_xpt_header(nhanes / f'{stem}_{cycle}.xpt', roles)
                                                  for cycle in ('D', 'E') for stem, roles in (('CBC', CBC_ROLES), ('DEMO', DEMO_ROLES))})
    out.mkdir()
    report = out / 'schema.json'
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    bindings = {'schema': 'bran-six-source-schema-manifest-v2',
                'files': {'schema.json': hashlib.sha256(report.read_bytes()).hexdigest()},
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'base_header_reader_code_sha256': hashlib.sha256((root / 'bran_six_source_schema_preflight_v1.py').read_bytes()).hexdigest(),
                'pandas_xport_reader_source_sha256': hashlib.sha256(inspect.getsource(XportReader).encode()).hexdigest(),
                'prior_header_manifest_sha256': PRIOR_PIN}
    (out / 'manifest.json').write_text(json.dumps(bindings, indent=2, sort_keys=True) + '\n')
    return {'status': 'completed_header_extension_no_rows_or_training',
            'all_requested_headers_match': {k: v['all_requested_headers_match'] for k, v in result['sources'].items()}}


if __name__ == '__main__':
    try:
        print(json.dumps(build(), sort_keys=True))
    except Exception:
        print('{"status":"header_extension_failed_without_disclosure"}')
        sys.exit(1)
