"""Terminal/component audits for own-run September protocols; no tensor emission."""
import argparse
import json
import math
from pathlib import Path
from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import sha, exclusive_json
import bran_september_push_io_v1 as io

ROOT=Path(__file__).resolve().parent
PINS={'matched':'2b37788135cc2b9fbd351430b031285bfa0ee130bb1557839147f21574da44d7',
      'joint':'34ba125921bbf00127c9d5a43f1483a6b18f1b8201eacfee9c4bf26af4d6e021'}


def is_sha(v):return type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v)


def validate_manifest(m,kind):
    keys={'protocol_sha256','aggregate_sha256','elapsed_seconds','patient_level_output_emitted'}
    if kind=='joint':keys.add('checkpoint_sha256')
    if set(m)!=keys or m['protocol_sha256']!=PINS[kind] or not is_sha(m['aggregate_sha256']) or m['patient_level_output_emitted'] is not False:
        raise ValueError('terminal_manifest_invalid')
    if type(m['elapsed_seconds']) not in (float,int) or not math.isfinite(m['elapsed_seconds']) or m['elapsed_seconds']<0:
        raise ValueError('terminal_time_invalid')
    if kind=='joint' and (set(m['checkpoint_sha256'])!={'fold'+str(f) for f in range(5)} or not all(is_sha(v) for v in m['checkpoint_sha256'].values())):
        raise ValueError('checkpoint_manifest_invalid')


def validate_joint_bundle(bundle,p,fold):
    import torch
    from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
    from audit_bran_external_cbc_comparison_v1 import validate_checkpoint
    from bran_clinical_semantics_v1 import CBC_FIELDS
    extra={'initial_checkpoint_sha256','endpoint_names','cbc_fields'}
    if (not isinstance(bundle,dict) or not extra<=set(bundle) or
        bundle['initial_checkpoint_sha256']!=p['source']['checkpoint_sha256']['fold'+str(fold)] or
        bundle['endpoint_names']!=p['source']['endpoint_names'] or bundle['cbc_fields']!=list(CBC_FIELDS)):
        raise ValueError('joint_bundle_identity_invalid')
    model=BRANClinicalAnchorV2()
    model.screening_joint_head=torch.nn.Linear(192,26)
    model.cbc_joint_head=torch.nn.Linear(192,9)
    validate_checkpoint({k:v for k,v in bundle.items() if k not in extra},model.state_dict(),PINS['joint'])
    return model


def audit(kind):
    import importlib
    from patient_atlas_real_data import _read_policy,_verify_source_hashes
    from patient_atlas_v6_2_expanded_endpoint_evaluation import EXACT_OUTER_FOLD_HASH,EXACT_INNER_FOLD_ASSIGNMENT_SHA256
    r=importlib.import_module('run_bran_matched_screening_v1' if kind=='matched' else 'run_bran_screening_joint_v1')
    if sha(r.PROTOCOL)!=PINS[kind]:raise ValueError('protocol_identity_invalid')
    p=json.loads(r.PROTOCOL.read_text());r.validate_protocol(p)
    if (r.OUT/'failure.json').exists() or not (r.OUT/'aggregate.json').is_file() or not (r.OUT/'manifest.json').is_file():
        raise ValueError('exclusive_terminal_invalid')
    a=json.loads((r.OUT/'aggregate.json').read_text());r.validate_result(a,p)
    m=json.loads((r.OUT/'manifest.json').read_text());validate_manifest(m,kind)
    if sha(r.OUT/'aggregate.json')!=m['aggregate_sha256']:raise ValueError('aggregate_bytes_invalid')
    auth=p['source']['authentication']
    if auth['outer_fold_sha256']!=EXACT_OUTER_FOLD_HASH or auth['inner_fold_sha256']!=list(EXACT_INNER_FOLD_ASSIGNMENT_SHA256):
        raise ValueError('fold_identity_invalid')
    old=io.joint.old_protocol();policy,_=_read_policy(ROOT)
    _verify_source_hashes(policy,dataset_root=Path(old['data_roots']['dataset_root']),clinical_project_root=Path(old['data_roots']['clinical_project_root']))
    checkpoints={}
    if kind=='joint':
        import torch
        for f in range(5):
            name='fold'+str(f);path=r.PRIVATE/(name+'.pt');digest=m['checkpoint_sha256'][name]
            if sha(path)!=digest or path.stat().st_mode & 0o777 != 0o600:raise ValueError('checkpoint_bytes_invalid')
            bundle=torch.load(path,map_location='cpu',weights_only=False)
            validate_joint_bundle(bundle,p,f)
            # Compare with the authenticated initial train-only normalizer. The
            # runner independently recomputed it; this audit checks the saved replay.
            prior_path=io.joint.PRIVATE/(name+'.pt')
            if sha(prior_path)!=p['source']['checkpoint_sha256'][name]:raise ValueError('initial_checkpoint_changed')
            prior=torch.load(prior_path,map_location='cpu',weights_only=False)
            import numpy as np
            for key in ('clinical_median','clinical_iqr','retinal_mean','retinal_scale','age_mean','age_scale'):
                if not np.array_equal(bundle[key],prior[key]):raise ValueError('normalizer_replay_invalid')
            if sha(path)!=digest:raise ValueError('checkpoint_changed_during_audit')
            checkpoints[name]=digest
    return {'schema':'bran-september-push-audit-v1','status':'authenticated','kind':kind,
        'protocol_sha256':PINS[kind],'aggregate_sha256':m['aggregate_sha256'],'manifest_sha256':sha(r.OUT/'manifest.json'),
        'checkpoint_sha256':checkpoints,'outer_fold_sha256':auth['outer_fold_sha256'],'inner_fold_sha256':auth['inner_fold_sha256'],
        'paired_people':1928,'recorded_conditions':26,'source_hashes_rechecked':True,'patient_level_output_emitted':False,
        'private_checkpoint_shapes_finite_and_normalizers_valid':kind=='joint','efficacy_recomputed':False,
        'auditor_sha256':sha(Path(__file__))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=('matched','joint'),required=True);args=parser.parse_args()
    out=ROOT/('BRAN_'+('MATCHED_SCREENING' if args.kind=='matched' else 'SCREENING_JOINT')+'_AUDIT_V1')
    ok=False;owned=False
    with _quiet():
        try:
            out.mkdir();owned=True;a=audit(args.kind);exclusive_json(out/'audit.json',a)
            exclusive_json(out/'manifest.json',{'audit_sha256':sha(out/'audit.json'),'patient_level_output_emitted':False});ok=True
        except Exception:
            if owned:exclusive_json(out/'failure.json',{'status':'component_audit_failed','patient_level_output_emitted':False})
    print(json.dumps({'status':'component_audit_passed' if ok else 'component_audit_failed','patient_level_output_emitted':False}))
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
