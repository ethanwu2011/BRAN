"""Local array-only fold transforms and hierarchical unpaired sampling.

No source files are opened here. Caller must authenticate original-unit inputs,
grouping, source-role receipts and row alignment before invoking these helpers.
Outputs contain private derived arrays and MUST NOT be printed or sent to tools.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

from bran_multisource_protocol_v2 import SOURCE_POLICY


def require(ok):
    if not ok: raise ValueError('bran_multisource_data_contract_failed')


def frozen(values, dtype=None):
    a=np.array(values,dtype=dtype,copy=True); a.setflags(write=False); return a


@dataclass(frozen=True, repr=False)
class FoldTransformV2:
    clinical_median: np.ndarray
    clinical_iqr: np.ndarray
    retinal_mean: np.ndarray
    retinal_scale: np.ndarray
    age_mean: float
    age_scale: float
    eligible: np.ndarray
    heldout_fold: int
    fold_identity_sha256: str
    training_indices_sha256: str

    def clinical(self, values, observed):
        require(type(values) is np.ndarray and values.ndim==2 and values.shape[1]==59 and values.dtype.kind=='f')
        require(type(observed) is np.ndarray and observed.dtype==np.bool_ and observed.shape==values.shape)
        mask=observed & self.eligible[None,:]
        require(np.isfinite(values[mask]).all())
        clean=np.zeros(values.shape,np.float32)
        # Only admitted observed values are ever used in arithmetic.
        rows,cols=np.nonzero(mask)
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            clean[rows,cols]=(values[rows,cols]-self.clinical_median[cols])/self.clinical_iqr[cols]
        require(np.isfinite(clean).all())
        return clean,mask.copy()

    def retinal(self, values, observed):
        require(type(values) is np.ndarray and values.ndim in (2,3) and values.shape[-1]==384 and values.dtype.kind=='f')
        require(type(observed) is np.ndarray and observed.dtype==np.bool_ and observed.shape==values.shape[:-1])
        require(np.isfinite(values[observed]).all())
        clean=np.zeros(values.shape,np.float32)
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            clean[observed]=(values[observed]-self.retinal_mean)/self.retinal_scale
        require(np.isfinite(clean).all())
        return clean,observed.copy()


def fit_fold_transform(clinical, observed, retinal, retinal_observed,
                       age_value, age_kind, folds, heldout_fold, eligible_indices):
    """Fit only AI outer-training rows; never accepts external fit statistics."""
    require(type(clinical) is np.ndarray and clinical.ndim==2 and clinical.shape[1]==59 and clinical.dtype.kind=='f')
    n=len(clinical)
    require(type(observed) is np.ndarray and observed.shape==clinical.shape and observed.dtype==np.bool_)
    require(type(retinal) is np.ndarray and retinal.shape==(n,384) and retinal.dtype.kind=='f')
    require(type(retinal_observed) is np.ndarray and retinal_observed.shape==(n,) and retinal_observed.dtype==np.bool_)
    require(type(folds) is np.ndarray and folds.shape==(n,) and folds.dtype.kind in 'iu')
    require(type(heldout_fold) is int and heldout_fold in range(5) and set(np.unique(folds))==set(range(5)))
    require(type(age_value) is np.ndarray and age_value.shape==(n,) and age_value.dtype.kind=='f')
    require(type(age_kind) is np.ndarray and age_kind.shape==(n,) and age_kind.dtype.kind in 'iu')
    indices=tuple(eligible_indices)
    require(len(indices)==43 and all(type(i) is int and 0<=i<48 for i in indices) and len(set(indices))==43)
    eligible=np.zeros(59,bool); eligible[list(indices)]=True
    train=np.flatnonzero(folds!=heldout_fold)
    c=clinical[train]; cm=observed[train]&eligible[None,:]
    require(np.isfinite(c[cm]).all())
    median=np.zeros(59,np.float64); iqr=np.ones(59,np.float64)
    for j in indices:
        v=c[cm[:,j],j]
        if len(v):
            median[j]=np.median(v); spread=float(np.quantile(v,.75)-np.quantile(v,.25))
            iqr[j]=spread if spread>0 else 1.
    rt=retinal[train][retinal_observed[train]]
    require(len(rt)>0 and np.isfinite(rt).all())
    rmean=rt.mean(0); rscale=rt.std(0); rscale[rscale==0]=1.
    kinds=age_kind[train]
    require(np.isin(kinds,[0,1,2,3]).all())
    ages=age_value[train][kinds==0]
    require(len(ages)>0 and np.isfinite(ages).all() and (ages>=0).all())
    amean=float(ages.mean()); ascale=float(ages.std()) or 1.
    require(np.isfinite(median).all() and np.isfinite(iqr).all() and
            np.isfinite(rmean).all() and np.isfinite(rscale).all() and
            np.isfinite(amean) and np.isfinite(ascale))
    return FoldTransformV2(frozen(median),frozen(iqr),frozen(rmean),frozen(rscale),amean,ascale,
        frozen(eligible),heldout_fold,
        hashlib.sha256(np.asarray(folds,dtype='<i8').tobytes()).hexdigest(),
        hashlib.sha256(np.asarray(train,dtype='<i8').tobytes()).hexdigest())


@dataclass(frozen=True, repr=False)
class SourceGroups:
    """Private source-local integer group identities, not global person IDs."""
    source: str
    group: np.ndarray
    eligible: np.ndarray


class UnpairedSamplerV2:
    """Family -> source-local person/group -> eligible episode, half per modality.

    MIMIC-III and IV share a family. Person choice within that family spans both
    releases, without falsely deduplicating matching integers across releases.
    Caller must first restrict each pool to its authenticated training partition.
    """
    def __init__(self,pools):
        require(type(pools) in (tuple,list) and len(pools)>0)
        self.families={'clinical':{},'retinal':{}}
        self.rows={}; self.group_for_row={}; self.seen_rows=set(); self.seen_groups=set()
        names=[]
        for pool in pools:
            require(isinstance(pool,SourceGroups) and pool.source in SOURCE_POLICY)
            modality,role,family=SOURCE_POLICY[pool.source]
            require(role=='development' and modality in self.families)
            require(type(pool.group) is np.ndarray and pool.group.ndim==1 and pool.group.dtype.kind in 'iu')
            require(type(pool.eligible) is np.ndarray and pool.eligible.shape==pool.group.shape and pool.eligible.dtype==np.bool_)
            require(pool.source not in names); names.append(pool.source)
            use=np.flatnonzero(pool.eligible); require(len(use)>0)
            require((pool.group[use]>=0).all())
            # One stable sort, rather than scanning a large cohort once/person.
            order=use[np.argsort(pool.group[use],kind='stable')]
            groups_sorted=pool.group[order]
            boundaries=np.r_[0,np.flatnonzero(groups_sorted[1:]!=groups_sorted[:-1])+1,len(order)]
            for start,stop in zip(boundaries[:-1],boundaries[1:]):
                key=(pool.source,int(groups_sorted[start])); rows=order[start:stop]
                self.rows[key]=frozen(rows,np.int64)
                self.families[modality].setdefault(family,[]).append(key)
                for row in rows: self.group_for_row[(pool.source,int(row))]=key
        require(all(self.families[m] for m in self.families))
        for modality in self.families:
            self.families[modality]={f:tuple(sorted(g)) for f,g in sorted(self.families[modality].items())}

    def sample(self,batch_size,rng):
        require(type(batch_size) is int and batch_size>0 and batch_size%2==0)
        require(isinstance(rng,np.random.Generator))
        result=[]
        for modality in ('clinical','retinal'):
            families=self.families[modality]; names=tuple(families)
            for _ in range(batch_size//2):
                family=names[int(rng.integers(len(names)))]; groups=families[family]
                key=groups[int(rng.integers(len(groups)))]; rows=self.rows[key]
                row=int(rows[int(rng.integers(len(rows)))]); pair=(key[0],row)
                result.append(pair); self.seen_rows.add(pair); self.seen_groups.add(key)
        return tuple(result)  # private index pairs, never a public progress value

    def safe_exposure(self):
        def count(values):
            n=len(values); return (n//20)*20 if n>=20 else None
        return {'source_local_groups_lower_bound_20':count(self.seen_groups),
            'examples_lower_bound_20':count(self.seen_rows),
            'global_unique_people':None,'cross_source_identity_resolved':False}
