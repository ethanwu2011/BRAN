"""Local semantic audit of the private cache, exporting only closed flags.

Neither raw arrays nor arbitrary errors may leave FD suppression. This is not
an efficacy test and does not authorize a fit.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from bran_joint_lab_cache_v1 import FIELDS, AGE_KINDS, COUNT_KEYS, SPLITS, coarse_count, validate_aggregate
from run_bran_source_linkage_audit_v1 import sha, exclusive_json

KEYS = {'values','observed','provenance','calibration_code','age_triplet','age_kind','adult_qualified','person_group','split','person_weight','cycle_group'}
ROOT = Path(__file__).resolve().parent


def check_arrays(a, source, summary):
    """Return no arrays or counts; fail closed if persisted semantics changed."""
    if source not in ('nhanes','mimic') or set(a) != KEYS: raise ValueError('invalid cache keys')
    x, m = a['values'], a['observed']
    if x.dtype.kind != 'f' or x.ndim != 2 or x.shape[1] != 21 or x.shape[0] == 0 or m.shape != x.shape or m.dtype != np.bool_: raise ValueError('invalid value or mask array')
    n = x.shape[0]
    if np.any(m & (~np.isfinite(x) | (x < 0))) or not np.isnan(x[~m]).all() or np.any(m[:,:9] & (x[:,:9] <= 0)) or np.any(m[:,:9].sum(1) < 2): raise ValueError('invalid observed truth')
    provenance = a['provenance']
    if provenance.dtype.kind not in 'iu' or not np.array_equal(provenance,m.astype(np.uint8)): raise ValueError('invalid observation provenance')
    for key in ('age_kind','person_group','split','cycle_group'):
        if a[key].shape != (n,) or a[key].dtype.kind not in 'iu': raise ValueError('invalid grouping metadata')
    ages, kinds, adult = a['age_triplet'], a['age_kind'], a['adult_qualified']
    if ages.shape != (n,3) or ages.dtype.kind != 'f' or np.any((kinds < 0) | (kinds >= len(AGE_KINDS))) or adult.shape != (n,) or adult.dtype != np.bool_: raise ValueError('invalid age metadata')
    if not np.array_equal(adult,np.isfinite(ages[:,1]) & (ages[:,1] >= 18) & (kinds != AGE_KINDS.index('source_age_unresolved'))): raise ValueError('invalid adult qualification')
    groups, split, weights = a['person_group'], a['split'], a['person_weight']
    unique, first, frequencies = np.unique(groups,return_index=True,return_counts=True)
    if not np.array_equal(unique,np.arange(len(unique))) or np.any((split < 0) | (split > 2)) or not np.array_equal(split,split[first][groups]): raise ValueError('person split inconsistent')
    if weights.shape != (n,) or weights.dtype.kind != 'f' or not np.allclose(weights,1./frequencies[groups],rtol=0,atol=1e-12): raise ValueError('person weighting inconsistent')
    if source == 'nhanes' and len(unique) != n: raise ValueError('NHANES person duplicated')
    calibration, cycles = a['calibration_code'], a['cycle_group']
    if calibration.shape != x.shape or calibration.dtype.kind not in 'iu' or np.any((calibration < 0) | (calibration > 2)): raise ValueError('invalid calibration metadata')
    expected = np.zeros_like(calibration)
    if source == 'nhanes':
        if np.any((cycles < 0) | (cycles > 1)): raise ValueError('invalid NHANES cycle')
        j = FIELDS.index('creatinine')
        expected[:,j] = np.where(m[:,j],cycles+1,0)
    elif np.any(cycles != -1): raise ValueError('invalid MIMIC cycle sentinel')
    if not np.array_equal(calibration,expected): raise ValueError('calibration identity mismatch')
    chem = m[:,9:].any(1)
    counts = dict(zip(COUNT_KEYS,map(coarse_count,(n,len(unique),int(adult.sum()),int(chem.sum()),int((chem & adult).sum())))))
    counts['field_observed_snapshots'] = {f:coarse_count(int(m[:,j].sum())) for j,f in enumerate(FIELDS)}
    counts['adult_field_observed_snapshots'] = {f:coarse_count(int((m[:,j] & adult).sum())) for j,f in enumerate(FIELDS)}
    counts['split_snapshots'] = {k:coarse_count(int((split == v).sum())) for k,v in SPLITS.items()}
    if counts != summary: raise ValueError('released aggregate differs from private cache')


def audit(source):
    if source == 'nhanes':
        import run_bran_nhanes_joint_labs_v1 as r
        pin = 'fdbec0e9333a2622aa6522d05e39fbcef62b92680246c409a7574cbf26b7ce45'
    elif source == 'mimic':
        import run_bran_mimic_joint_labs_v1 as r
        pin = '96803a0613dd19a2171b1d97b98ab56228767c90794f2068c48e522f96e37246'
    else: raise ValueError('unsupported source')
    if sha(r.PROTOCOL) != pin: raise ValueError('protocol authentication failed')
    r.validate_protocol(json.loads(r.PROTOCOL.read_text()))
    if not (r.PUBLIC/'aggregate.json').is_file() or not (r.PUBLIC/'manifest.json').is_file() or (r.PUBLIC/'failure.json').exists(): raise ValueError('terminal state not exclusive success')
    a = json.loads((r.PUBLIC/'aggregate.json').read_text()); validate_aggregate(a)
    manifest = json.loads((r.PUBLIC/'manifest.json').read_text())
    if set(manifest) != {'protocol_sha256','aggregate_sha256','split_salt_sha256','patient_level_output_emitted','training_started'} or manifest['patient_level_output_emitted'] is not False or manifest['training_started'] is not False: raise ValueError('manifest schema mismatch')
    if manifest['protocol_sha256'] != pin or manifest['aggregate_sha256'] != sha(r.PUBLIC/'aggregate.json') or manifest['split_salt_sha256'] != sha(r.SALT): raise ValueError('manifest authentication failed')
    cache = r.PRIVATE/'observations.npz'
    if a['source'] != source or a['private_cache_sha256'] != sha(cache) or cache.stat().st_mode & 0o777 != 0o600: raise ValueError('cache authentication failed')
    with np.load(cache,allow_pickle=False) as handle:
        if set(handle.files) != KEYS: raise ValueError('private cache keys mismatch')
        check_arrays({key:handle[key] for key in KEYS},source,a['counts_lower_bounds_20'])
    if sha(cache) != a['private_cache_sha256']: raise ValueError('cache changed during audit')
    return {'schema':'bran-joint-lab-cache-audit-v1','source':source,'status':'authenticated',
        'protocol_sha256':pin,'aggregate_sha256':sha(r.PUBLIC/'aggregate.json'),
        'cache_semantics_valid':True,'person_splits_consistent':True,'calibration_identity_valid':True,
        'aggregate_recomputed_locally':True,'patient_level_output_emitted':False,'training_started':False,
        'auditor_sha256':sha(Path(__file__)),'synthetic_test_sha256':sha(ROOT/'test_audit_bran_joint_lab_cache_v1.py')}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',choices=('nhanes','mimic'),required=True); args=p.parse_args()
    ok=False
    with _quiet():
        try:
            receipt=audit(args.source)
            out=ROOT/('BRAN_'+args.source.upper()+'_JOINT_LABS_AUDIT_V1')
            out.mkdir(); exclusive_json(out/'audit.json',receipt); ok=True
        except Exception: pass
    print(json.dumps({'status':'joint_cache_audit_passed' if ok else 'joint_cache_audit_failed','source':args.source,'patient_level_output_emitted':False,'training_started':False}))
    return 0 if ok else 1


if __name__=='__main__': raise SystemExit(main())
