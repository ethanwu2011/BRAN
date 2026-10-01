"""Local fixed-budget V2 fit orchestration and authenticated private checkpoints.

The source runner must hold the shared real-data lock, silence process file
descriptors, and authenticate receipts before/after calling this module. This
module never logs patient data, losses, or states and never selects a model.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_multisource_age_v2 import AgeBatch
from bran_multisource_model_v2 import BRANMultisourceModelV2
from bran_multisource_protocol_v2 import PARAMETERS,ARMS,readiness,digest,is_hash
from bran_multisource_batches_v2 import (PairedBatchesV2,UnpairedBatchesV2,
    transform_hash,tensor)
from bran_multisource_training_v2 import train_step
from bran_multisource_data_v2 import FoldTransformV2,frozen
from bran_research_state_io_v1 import canonical_registry
from bran_joint_lab_cache_v1 import FIELDS


def require(ok):
    if not ok:raise ValueError('multisource_fit_contract_failed')


class _TrackedChoice:
    """Same RNG draws, with private exposure accounting; never printable rows."""
    def __init__(self,generator):self.generator=generator;self.seen=set()
    def __repr__(self):return '<TrackedChoice private>'
    def choice(self,*args,**kwargs):
        rows=self.generator.choice(*args,**kwargs);self.seen.update(map(int,rows));return rows


def _exposure(unpaired,paired_batches,paired,pools):
    def coarse(value):return int(value)//20*20 if value>=20 else None
    result={}
    for pool in pools:
        rows=np.asarray(sorted(row for source,row in unpaired.sampler.seen_rows if source==pool.source),np.int64)
        people={group for source,group in unpaired.sampler.seen_groups if source==pool.source}
        supported=len(people)>=20
        item={'source_local_people_lower_bound_20':coarse(len(people)),
              'unique_examples_lower_bound_20':coarse(len(rows)) if supported else None}
        if hasattr(pool,'observed'):
            admitted=[j for j,name in enumerate(FIELDS) if paired.names.index(name) in paired.eligible_indices]
            item['unique_observed_measurements_lower_bound_20']=coarse(pool.observed[rows][:,admitted].sum()) if supported else None
        else:item['unique_retinal_images_lower_bound_20']=coarse(len(rows)) if supported else None
        result[pool.source]=item
    seen=np.asarray(sorted(paired_batches.rng.seen),np.int64)
    result['aireadi']={'paired_people_lower_bound_20':coarse(len(seen)),
        'unique_observed_clinical_measurements_lower_bound_20':coarse(paired_batches.cm[seen].sum()) if len(seen)>=20 else None,
        'pooled_retinal_inputs_lower_bound_20':coarse(paired_batches.rm[seen].sum()) if len(seen)>=20 else None,
        'pooled_vectors_not_counted_as_individual_images':True}
    return {'per_source':result,'global_unique_people':None,'cross_source_identity_resolved':False,
        'fold_exposures_must_not_be_summed':True}


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def batch_fingerprint(batch):
    """Private whole-batch fingerprint; caller only releases a full-run digest."""
    h=hashlib.sha256()
    for name,value in (('c',batch.c),('cm',batch.cm),('r',batch.r),('rm',batch.rm),
                       ('age_value',batch.age.value),('age_lower',batch.age.lower),
                       ('age_upper',batch.age.upper),('age_kind',batch.age.kind),
                       ('labels',batch.labels),('labelmask',batch.labelmask)):
        h.update(name.encode())
        if value is None:h.update(b'none');continue
        a=value.detach().cpu().numpy();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def validate_session(protocol,paired,pools,arm,fold):
    require(type(protocol) is dict and protocol.get('schema')=='bran-multisource-protocol-v2')
    require(protocol.get('parameters')==PARAMETERS and protocol.get('arms')==list(ARMS))
    require(type(fold) is int and fold in range(5) and arm in ARMS)
    ready=readiness(protocol['sources']);require(ready['source_ready'])
    require(sorted([p.source for p in pools]+['aireadi'])==ready['training_source_names'])
    count=sum(len(p.person_group) for p in pools)
    require(protocol['stage_a_steps_per_arm_fold']==math.ceil(count/PARAMETERS['stage_a_batch']))
    require(protocol['outer_folds_sha256']==paired.receipt['outer_fold_sha256'])
    require(protocol['inner_folds_sha256']==paired.receipt['inner_fold_sha256'])
    require(protocol['transform_sha256']==[transform_hash(t) for t in paired.transforms])
    require(paired.transforms[fold].heldout_fold==fold)
    names,types=canonical_registry()
    require(tuple(paired.names)==names and types==('continuous',)*48+('binary',)*11)
    require(tuple(paired.eligible_indices)==tuple(map(int,np.flatnonzero(paired.transforms[fold].eligible))))
    return paired.transforms[fold]


def _transform_payload(transform):
    return {name:(torch.tensor(value.copy()) if isinstance(value,np.ndarray) else value)
            for name,value in vars(transform).items()}


def _restore_transform(payload,expected_sha):
    require(type(payload) is dict and set(payload)==set(FoldTransformV2.__dataclass_fields__))
    restored={name:(frozen(value.cpu().numpy()) if isinstance(value,torch.Tensor) else value)
              for name,value in payload.items()}
    result=FoldTransformV2(**restored);require(transform_hash(result)==expected_sha)
    return result


def _save_checkpoint(path,model,optimizer,binding,stage,step,transform):
    """Fresh private file only; a prior checkpoint is never overwritten."""
    path=Path(path);require(not path.exists() and not path.is_symlink())
    bundle={'schema':'bran-multisource-checkpoint-v2','config':model.export_config(),
        'binding':binding,'stage':stage,'steps_completed':step,
        'state_dict':model.state_dict(),'optimizer_state':optimizer.state_dict(),
        'input_transform':_transform_payload(transform)}
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as stream:
        torch.save(bundle,stream);stream.flush();os.fsync(stream.fileno())
    return file_sha(path)


def load_checkpoint(path,*,expected_sha256,binding,stage,steps):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink==1
            and path.stat().st_mode&0o777==0o600 and is_hash(expected_sha256)
            and file_sha(path)==expected_sha256)
    bundle=torch.load(path,map_location='cpu',weights_only=True)
    require(type(bundle) is dict and set(bundle)=={'schema','config','binding','stage','steps_completed','state_dict','optimizer_state','input_transform'})
    require(bundle['schema']=='bran-multisource-checkpoint-v2' and bundle['binding']==binding
            and bundle['stage']==stage and bundle['steps_completed']==steps)
    model=BRANMultisourceModelV2.from_config(bundle['config'])
    require(model.arm==binding['arm'] and digest(model.export_config())==binding['model_config_sha256'])
    names,_=canonical_registry()
    require(binding['clinical_field_order_sha256']==digest(names)
            and model.cbc_indices==tuple(names.index(name) for name in CBC_FIELDS))
    transform=_restore_transform(bundle['input_transform'],binding['transform_sha256'])
    require(transform.heldout_fold==binding['fold'])
    require(all(not value.is_floating_point() or bool(torch.isfinite(value).all()) for value in bundle['state_dict'].values()))
    model.load_state_dict(bundle['state_dict'],strict=True)
    require(file_sha(path)==expected_sha256);model.eval()
    return model,transform


def fit_one(protocol,paired,pools,*,arm,fold,private_directory,progress=None):
    """All three frozen stages for one arm/fold. No outcome evaluation or tuning.

    Stages and every 300 updates get separate durable private checkpoints.
    A failed call does not automatically resume or retry. Its owning runner
    preserves the failure and can authorize a separately versioned correction.
    """
    transform=validate_session(protocol,paired,pools,arm,fold)
    directory=Path(private_directory);require(not directory.exists() and not directory.is_symlink())
    require(directory.parent.is_dir() and directory.parent.stat().st_mode&0o777==0o700)
    directory.mkdir(mode=0o700)
    torch.set_num_threads(2);seed=PARAMETERS['seed_base']+fold
    cbc=tuple(paired.names.index(name) for name in CBC_FIELDS)
    model=BRANMultisourceModelV2(arm,paired.eligible_indices,cbc,seed=seed);model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=PARAMETERS['learning_rate'],weight_decay=PARAMETERS['weight_decay'])
    missing=np.full(len(paired.age_value),np.nan)
    # The current AI-READI source carries reported/unknown age. Compatible future
    # paired adapters must supply actual interval/censoring bounds explicitly.
    lower=getattr(paired,'age_lower',missing);upper=getattr(paired,'age_upper',missing)
    age=AgeBatch(tensor(paired.age_value),tensor(lower),tensor(upper),tensor(paired.age_kind,torch.long))
    paired_batches=PairedBatchesV2(paired.c,paired.cm,paired.r,paired.rm,age,paired.labels,
        paired.labelmask,paired.folds,transform,seed+202)
    paired_batches.rng=_TrackedChoice(paired_batches.rng)
    unpaired=UnpairedBatchesV2(pools,transform,paired.names,seed+404)
    positive_weights=paired_batches.positive_weights()
    binding={'protocol_sha256':digest(protocol),'arm':arm,'fold':fold,
        'transform_sha256':transform_hash(transform),'model_config_sha256':digest(model.export_config()),
        'clinical_field_order_sha256':digest(paired.names),
        'outer_folds_sha256':protocol['outer_folds_sha256'],
        'inner_fold_sha256':protocol['inner_folds_sha256'][fold],
        'representation_dimension':192,'native_screening_outputs':26,'native_cbc_outputs':9}
    stages=(('A',protocol['stage_a_steps_per_arm_fold']),('B',PARAMETERS['stage_b_steps']),('C',PARAMETERS['stage_c_steps']))
    manifests={};input_trace=hashlib.sha256();mask_trace=hashlib.sha256();start=time.perf_counter()
    for stage,steps in stages:
        for step in range(steps):
            if stage=='A':
                primary=unpaired.sample(PARAMETERS['stage_a_batch']);rehearsal=None
            else:
                primary=paired_batches.sample(PARAMETERS['paired_batch'],supervised=(stage=='C'))
                rehearsal=unpaired.sample(PARAMETERS['stage_a_batch'])
            input_trace.update(batch_fingerprint(primary).encode())
            if rehearsal is not None:input_trace.update(batch_fingerprint(rehearsal).encode())
            result=train_step(model,optimizer,stage,primary,rehearsal,step,transform.age_mean,
                transform.age_scale,seed,cbc,positive_weights if stage=='C' else None)
            require(result['optimizer_updated'] is True)
            mask_trace.update(json.dumps(result['mask_hashes'],sort_keys=True).encode())
            # No per-batch loss, mask digest, or support size reaches progress.
            if (step+1)%300==0 or step+1==steps:
                name='stage_'+stage+'_step_'+str(step+1)+'.pt'
                checkpoint_sha=_save_checkpoint(directory/name,model,optimizer,binding,stage,step+1,transform)
                manifests[name]={'sha256':checkpoint_sha,'stage':stage,'steps':step+1}
                if progress is not None:progress({'phase':'training','arm':arm,'fold':fold,'stage':stage,'updates_completed':step+1})
    final_name='stage_C_step_'+str(PARAMETERS['stage_c_steps'])+'.pt'
    restored,restored_transform=load_checkpoint(directory/final_name,expected_sha256=manifests[final_name]['sha256'],
        binding=binding,stage='C',steps=PARAMETERS['stage_c_steps'])
    require(transform_hash(restored_transform)==transform_hash(transform))
    require(all(torch.equal(value,restored.state_dict()[key]) for key,value in model.state_dict().items()))
    return {'status':'fit_completed_not_evaluated','binding':binding,'checkpoint_manifest':manifests,
        'input_sequence_sha256':input_trace.hexdigest(),'mask_sequence_sha256':mask_trace.hexdigest(),
        'exposure':_exposure(unpaired,paired_batches,paired,pools),'runtime_seconds':time.perf_counter()-start,
        'checkpoint_reload_exact':True,'candidate_promoted':False,'scientific_goal_achieved':False,
        'patient_level_output_emitted':False}
