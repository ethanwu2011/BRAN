"""Authenticate AI-READI source/folds and V2 outer-training-only transforms.

No model fitting. Patient arrays remain inside a file-descriptor-quiet local
process; only full-cohort hashes and disclosure-safe aggregate flags are saved.
"""
import argparse
from dataclasses import dataclass
import fcntl
import json
from pathlib import Path

import numpy as np

from run_bran_multisource_retinal_features_v2 import LOCK,quiet,sha,write_json
from bran_multisource_data_v2 import fit_fold_transform
from bran_multisource_batches_v2 import transform_hash

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'BRAN_MULTISOURCE_PAIRED_BINDING_V2_ATTEMPT1'
CODE=('run_bran_multisource_paired_binding_v2.py','bran_multisource_data_v2.py',
      'bran_multisource_batches_v2.py','bran_multisource_age_v2.py',
      'bran_multisource_model_v2.py','bran_multisource_protocol_v2.py')


def require(ok):
    if not ok:raise ValueError('paired_v2_binding_failed')


@dataclass(frozen=True,repr=False)
class PairedSourceV2:
    c:np.ndarray
    cm:np.ndarray
    r:np.ndarray
    rm:np.ndarray
    age_value:np.ndarray
    age_kind:np.ndarray
    folds:np.ndarray
    names:tuple
    eligible_indices:tuple
    labels:np.ndarray
    labelmask:np.ndarray
    endpoint_names:tuple
    transforms:tuple
    receipt:dict


def load_private():
    """Call ONLY within a locked FD-quiet local process. Returns private arrays."""
    import bran_september_push_io_v1 as source
    import bran_multisource_paired_source_v1 as paired
    import bran_authenticated_retinal_input_v3 as inputs
    import run_bran_overnight_diagnostic_v1 as base
    code={name:sha(ROOT/name) for name in CODE}
    initial=source.source_receipt();retinal=paired.authenticate_binding()
    ctx,folds,c,cm,eligible,_,rm,ages,names=source.load_context()
    ids=list(map(str,ctx['raw_cohort'].patient_ids))
    require(len(ids)==len(set(ids)) and len(ids)>=100 and
            tuple(map(str,ctx['feature_cohort'].patient_ids))==tuple(ids))
    require(set(ctx['raw_cohort'].split_labels)<={'train','val'})
    folds=np.asarray(folds,np.int64)
    r,present=inputs.load_pooled_features(protocol_pin=paired.PROTOCOL_SHA256,
        audit_pin=paired.AUDIT_SHA256,patient_ids=ids,folds=folds,
        source_policy_sha256=retinal['source_policy_sha256'])
    require(np.array_equal(present,rm))
    n=len(c);require(c.shape==(n,59) and eligible.shape==c.shape and eligible.dtype==np.bool_)
    require(np.array_equal(eligible,np.broadcast_to(eligible[0],eligible.shape)))
    indices=tuple(map(int,np.flatnonzero(eligible[0])))
    require(len(indices)==43 and all(i<48 for i in indices))
    av=np.asarray(ages,np.float64);ak=np.where(np.isfinite(av)&(av>=0),0,3).astype(np.int64)
    av=av.copy();av[ak==3]=np.nan
    endpoints=tuple(initial['endpoint_names']);require(len(endpoints)==26 and len(set(endpoints))==26)
    labels=np.column_stack([ctx['labels_by_source'][e] for e in endpoints]).astype(np.float64)
    labelmask=np.column_stack([ctx['observed_by_source'][e] for e in endpoints]).astype(bool)
    require(labels.shape==(n,26) and np.isfinite(labels[labelmask]).all()
            and np.isin(labels[labelmask],[0,1]).all())
    transforms=[];inner=[]
    for fold in range(5):
        train=np.flatnonzero(folds!=fold)
        _,identity=base._inner_context(ctx,train,fold)
        require(identity==initial['authentication']['inner_fold_sha256'][fold]);inner.append(identity)
        transforms.append(fit_fold_transform(c,cm,r,present,av,ak,folds,fold,indices))
    require(source.source_receipt()==initial and paired.authenticate_binding()==retinal)
    require(code=={name:sha(ROOT/name) for name in CODE})
    receipt={'schema':'bran-multisource-paired-binding-v2','status':'paired_source_and_five_fold_transforms_authenticated',
        'source_receipt':initial,'retinal_binding':retinal,
        'outer_fold_sha256':initial['authentication']['outer_fold_sha256'],
        'fold_array_sha256':transforms[0].fold_identity_sha256,
        'inner_fold_sha256':inner,'transform_sha256':[transform_hash(t) for t in transforms],
        'code_sha256':code,'people_lower_bound_20':n//20*20,'native_endpoints':len(endpoints),
        'eligible_continuous_inputs':len(indices),'clinical_container_width':59,
        'retinal_feature_width':384,'missing_age_supported':True,
        'outer_training_only_normalizers':True,'heldout_pretraining_permitted':False,
        'official_test_used':False,'training_started':False,'patient_level_output_emitted':False}
    # Never expose source authentication internals with counts not separately
    # disclosure-reviewed. Retain full objects only in the private return value.
    public={k:v for k,v in receipt.items() if k!='source_receipt'}
    public['source_receipt_sha256']=__import__('hashlib').sha256(json.dumps(initial,
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return PairedSourceV2(c,cm,r,present,av,ak,folds,tuple(names),indices,labels,labelmask,
                          endpoints,tuple(transforms),public)


def run():
    require(not OUT.exists());OUT.mkdir()
    try:
        source=load_private();write_json(OUT/'aggregate.json',source.receipt)
        write_json(OUT/'manifest.json',{'aggregate_sha256':sha(OUT/'aggregate.json'),
            'patient_level_output_emitted':False})
    except Exception:
        # Publication is accepted only with a manifest and no failure marker.
        write_json(OUT/'failure.json',{'status':'technical_failure','phase':'paired_source_authentication',
            'patient_level_output_emitted':False,'training_started':False})
        raise ValueError('paired_v2_binding_failed') from None


def main():
    argparse.ArgumentParser().parse_args();ok=False
    with quiet():
        try:
            with LOCK.open('a') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);run();ok=True
        except Exception:pass
    print(json.dumps({'status':'paired_source_and_five_fold_transforms_authenticated' if ok else 'paired_binding_failed',
        'patient_level_output_emitted':False,'training_started':False}))
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
