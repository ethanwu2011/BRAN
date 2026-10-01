"""Private, array-only materialization for BRAN V2; no file or network I/O.

All values, indices, tensors, states, and per-batch fingerprints stay local.
Callers authenticate source receipts before constructing these objects.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from bran_multisource_age_v2 import AgeBatch, validate_age
from bran_multisource_clinical_v2 import ClinicalPoolV2
from bran_multisource_data_v2 import FoldTransformV2, SourceGroups, UnpairedSamplerV2
from bran_multisource_training_v2 import MaterializedBatch
from bran_joint_lab_task_contract_v1 import project_joint_labs_to_registry


def require(ok):
    if not ok: raise ValueError('multisource_batch_contract_failed')


def tensor(values, dtype=torch.float32):
    return torch.tensor(np.asarray(values), dtype=dtype)


@dataclass(frozen=True, repr=False)
class RetinalPoolV2:
    source: str
    features: np.ndarray
    person_group: np.ndarray
    age_value: np.ndarray
    age_lower: np.ndarray
    age_upper: np.ndarray
    age_kind: np.ndarray


def typed_age(pool, rows):
    result=AgeBatch(tensor(pool.age_value[rows]),tensor(pool.age_lower[rows]),
                    tensor(pool.age_upper[rows]),tensor(pool.age_kind[rows],torch.long))
    return validate_age(result)


def transform_hash(transform):
    require(isinstance(transform,FoldTransformV2))
    h=hashlib.sha256()
    for name in ('clinical_median','clinical_iqr','retinal_mean','retinal_scale',
                 'age_mean','age_scale','eligible','heldout_fold',
                 'fold_identity_sha256','training_indices_sha256'):
        value=getattr(transform,name)
        h.update(name.encode())
        if isinstance(value,np.ndarray):
            h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
        else:h.update(repr(value).encode())
    return h.hexdigest()


class UnpairedBatchesV2:
    """Sample from authenticated TRAIN-ONLY pools in the recipient coordinate frame."""
    def __repr__(self): return '<UnpairedBatchesV2 private>'

    def __init__(self,pools,transform,registry_names,seed):
        require(isinstance(transform,FoldTransformV2))
        require(type(seed) is int and seed>=0)
        require(type(registry_names) is tuple and len(registry_names)==59)
        self.transform=transform;self.registry=registry_names
        self.pools={};groups=[]
        for pool in pools:
            require(isinstance(pool,(ClinicalPoolV2,RetinalPoolV2)) and pool.source not in self.pools)
            n=len(pool.person_group);require(n>0)
            require(pool.person_group.shape==(n,) and pool.person_group.dtype.kind in 'iu')
            for name in ('age_value','age_lower','age_upper','age_kind'):
                require(getattr(pool,name).shape==(n,))
            typed_age(pool,np.arange(n))
            if isinstance(pool,RetinalPoolV2):
                require(pool.features.shape==(n,384) and pool.features.dtype.kind=='f'
                        and np.isfinite(pool.features).all())
            else:require(len(pool.values)==n and len(pool.observed)==n)
            self.pools[pool.source]=pool
            groups.append(SourceGroups(pool.source,pool.person_group,np.ones(n,bool)))
        self.sampler=UnpairedSamplerV2(groups);self.rng=np.random.default_rng(seed)

    def sample(self,size):
        draws=self.sampler.sample(size,self.rng)
        c=np.zeros((size,59),np.float32);cm=np.zeros((size,59),bool)
        r=np.zeros((size,1,384),np.float32);rm=np.zeros((size,1),bool)
        av=np.full(size,np.nan);al=av.copy();au=av.copy();ak=np.full(size,3,np.int64)
        # Work source-wise, not one Python/torch operation per patient.
        for source,pool in self.pools.items():
            dest=np.asarray([i for i,(s,_) in enumerate(draws) if s==source],dtype=np.int64)
            if not len(dest):continue
            rows=np.asarray([draws[i][1] for i in dest],dtype=np.int64)
            if isinstance(pool,ClinicalPoolV2):
                projected,observed=project_joint_labs_to_registry(pool.values[rows],pool.observed[rows],
                    pool.observed[rows].astype(np.uint8),self.registry,
                    np.zeros(59),np.ones(59))
                c[dest],cm[dest]=self.transform.clinical(projected,observed)
            else:
                r[dest,0],rm[dest,0]=self.transform.retinal(pool.features[rows],np.ones(len(rows),bool))
            av[dest]=pool.age_value[rows];al[dest]=pool.age_lower[rows]
            au[dest]=pool.age_upper[rows];ak[dest]=pool.age_kind[rows]
        require(not np.any(cm.any(1)&rm.any(1)))
        age=AgeBatch(tensor(av),tensor(al),tensor(au),tensor(ak,torch.long))
        return MaterializedBatch(tensor(c),tensor(cm,torch.bool),tensor(r),tensor(rm,torch.bool),validate_age(age))


class PairedBatchesV2:
    """Keep fold membership enforced at the sampling boundary, not in a comment."""
    def __repr__(self):return '<PairedBatchesV2 private>'

    def __init__(self,c,cm,r,rm,age,labels,labelmask,folds,transform,seed):
        require(isinstance(transform,FoldTransformV2) and type(seed) is int and seed>=0)
        n=len(c);require(folds.shape==(n,) and folds.dtype.kind in 'iu')
        require(hashlib.sha256(np.asarray(folds,dtype='<i8').tobytes()).hexdigest()==transform.fold_identity_sha256)
        require(labels.shape==(n,26) and labelmask.shape==labels.shape and labelmask.dtype==np.bool_)
        require(np.isfinite(labels[labelmask]).all() and np.isin(labels[labelmask],[0,1]).all())
        validate_age(age);require(age.value.shape==(n,))
        self.c,self.cm=transform.clinical(c,cm)
        retinal,present=transform.retinal(r,rm)
        self.r,self.rm=retinal[:,None],present[:,None]
        self.age=age;self.labels=labels.copy();self.labelmask=labelmask.copy()
        self.train=np.flatnonzero(folds!=transform.heldout_fold)
        require(hashlib.sha256(np.asarray(self.train,dtype='<i8').tobytes()).hexdigest()==transform.training_indices_sha256)
        self.rng=np.random.default_rng(seed)

    def sample(self,size,*,supervised):
        require(type(size) is int and size>0 and type(supervised) is bool)
        rows=self.rng.choice(self.train,size,replace=len(self.train)<size)
        index=tensor(rows,torch.long)
        age=AgeBatch(*(getattr(self.age,name)[index] for name in ('value','lower','upper','kind')))
        return MaterializedBatch(tensor(self.c[rows]),tensor(self.cm[rows],torch.bool),
            tensor(self.r[rows]),tensor(self.rm[rows],torch.bool),age,
            tensor(self.labels[rows]) if supervised else None,
            tensor(self.labelmask[rows],torch.bool) if supervised else None)

    def positive_weights(self):
        from bran_screening_joint_kernel_v1 import _positive_weights
        return tensor(_positive_weights(self.labels,self.labelmask,self.train))
