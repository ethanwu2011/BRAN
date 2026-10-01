"""Local-only component authentication for the completed frozen comparison.

Reads only own-run private checkpoint structures and file bytes under FD
suppression. Never emits weights, normalizers, source records or private arrays.
"""
import json
from pathlib import Path
import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import sha,exclusive_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'BRAN_EXTERNAL_CBC_COMPARISON_AUDIT_V1'


def validate_checkpoint(bundle,template,protocol_sha):
    import torch
    keys={'control','candidate','clinical_median','clinical_iqr','retinal_mean','retinal_scale','age_mean','age_scale','protocol_sha256'}
    if not isinstance(bundle,dict) or set(bundle)!=keys or bundle['protocol_sha256']!=protocol_sha:
        raise ValueError('checkpoint schema or identity mismatch')
    for arm in ('control','candidate'):
        state=bundle[arm]
        if set(state)!=set(template): raise ValueError('checkpoint parameter keys mismatch')
        for key,reference in template.items():
            tensor=state[key]
            if not isinstance(tensor,torch.Tensor) or tensor.shape!=reference.shape or tensor.dtype!=reference.dtype or not torch.isfinite(tensor).all():
                raise ValueError('checkpoint parameter validation failed')
    for key,width in (('clinical_median',59),('clinical_iqr',59),('retinal_mean',384),('retinal_scale',384)):
        value=bundle[key]
        if not isinstance(value,np.ndarray) or value.shape!=(width,) or not np.isfinite(value).all(): raise ValueError('checkpoint normalizer invalid')
        if key.endswith(('iqr','scale')) and np.any(value<=0): raise ValueError('checkpoint normalizer invalid')
    if type(bundle['age_mean']) is not float or type(bundle['age_scale']) is not float or not np.isfinite(bundle['age_mean']) or not np.isfinite(bundle['age_scale']) or bundle['age_scale']<=0:
        raise ValueError('checkpoint age normalizer invalid')


def audit():
    import torch
    import run_bran_external_cbc_comparison_v1 as r
    from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
    from patient_atlas_real_data import _read_policy,_verify_source_hashes
    r.pool.validate_protocol(json.loads(r.pool.PROTOCOL.read_text()))
    p=json.loads(r.PROTOCOL.read_text()); r.validate_protocol(p)
    a=json.loads((r.OUT/'aggregate.json').read_text()); m=json.loads((r.OUT/'manifest.json').read_text())
    r.validate_result(a,p)
    if (r.OUT/'failure.json').exists() or m['aggregate_sha256']!=sha(r.OUT/'aggregate.json') or m['protocol_sha256']!=sha(r.PROTOCOL):
        raise ValueError('exclusive terminal artifact mismatch')
    policy,_=_read_policy(ROOT)
    _verify_source_hashes(policy, dataset_root=Path(p['data_roots']['dataset_root']),
        clinical_project_root=Path(p['data_roots']['clinical_project_root']))
    for source in r.SOURCES:
        if sha(r.pool.PRIVATE/(source+'.npz'))!=p['sources'][source]['private_cache_sha256']:
            raise ValueError('external source cache changed')
    template=BRANClinicalAnchorV2().state_dict(); hashes={}
    for f in range(5):
        path=r.PRIVATE/('fold'+str(f)+'.pt'); before=sha(path)
        # Own-run file after authenticated source/implementation/terminal checks;
        # it contains torch state dicts plus NumPy train-fitted normalizers.
        bundle=torch.load(path,map_location='cpu',weights_only=False)
        validate_checkpoint(bundle,template,sha(r.PROTOCOL))
        if sha(path)!=before: raise ValueError('checkpoint changed during authentication')
        hashes['fold'+str(f)]=before
    return {'schema_version':'bran-external-cbc-comparison-audit-v1','status':'authenticated',
        'protocol_sha256':sha(r.PROTOCOL),'aggregate_sha256':sha(r.OUT/'aggregate.json'),
        'checkpoint_sha256':hashes,'checkpoint_count':5,'models_per_checkpoint':2,
        'outer_fold_sha256':p['authentication']['outer_fold_sha256'],
        'inner_fold_sha256':p['authentication']['inner_fold_sha256'],
        'all_model_shapes_and_values_valid':True,'normalizer_contract_valid':True,
        'paired_source_hashes_rechecked':True,'external_cache_hashes_rechecked':True,
        'patient_level_output_emitted':False,'model_weights_or_normalizers_emitted':False,
        'efficacy_recomputed':False,'auditor_sha256':sha(Path(__file__))}


def main():
    ok=False; owned=False
    with _quiet():
        try:
            OUT.mkdir(); owned=True; payload=audit()
            exclusive_json(OUT/'audit.json',payload)
            exclusive_json(OUT/'manifest.json',{'audit_sha256':sha(OUT/'audit.json'),'patient_level_output_emitted':False})
            ok=True
        except Exception:
            if owned: exclusive_json(OUT/'failure.json',{'status':'component_authentication_failed','patient_level_output_emitted':False})
    print(json.dumps({'status':'component_authentication_completed' if ok else 'component_authentication_failed','patient_level_output_emitted':False}))
    return 0 if ok else 1


if __name__=='__main__': raise SystemExit(main())
