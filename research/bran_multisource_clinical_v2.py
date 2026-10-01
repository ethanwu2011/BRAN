"""Private array bridge from authenticated original-unit legacy clinical pools.

No I/O or model fitting. These inputs/outputs are patient-derived: keep local,
never print, and never send them to hosted tools. Source receipt authentication
belongs to the caller. No imputation and no exact age invented for coarsening.
"""
from dataclasses import dataclass
import hashlib
import numpy as np

from bran_joint_lab_cache_v1 import AGE_KINDS, FIELDS

SOURCES={'mimic':'mimiciv','nhanes':'nhanes_exposed','eicu':'eicu'}


def require(ok):
    if not ok:raise ValueError('multisource_clinical_bridge_failed')


@dataclass(frozen=True,repr=False)
class ClinicalPoolV2:
    source: str
    values: np.ndarray
    observed: np.ndarray
    person_group: np.ndarray
    age_value: np.ndarray
    age_lower: np.ndarray
    age_upper: np.ndarray
    age_kind: np.ndarray
    original_indices: np.ndarray


def bridge(source, arrays):
    require(source in SOURCES and type(arrays) is dict)
    keys=('values','observed','provenance','person_group','split','adult_qualified','age_triplet','age_kind')
    require(all(k in arrays for k in keys))
    n=len(arrays['values']);v=arrays['values'];m=arrays['observed'];p=arrays['provenance']
    require(type(v) is np.ndarray and v.shape==(n,len(FIELDS)) and v.dtype.kind=='f')
    require(type(m) is np.ndarray and m.shape==v.shape and m.dtype==np.bool_)
    require(type(p) is np.ndarray and p.shape==v.shape and p.dtype.kind in 'iu' and np.array_equal(p,m.astype(p.dtype)))
    require(np.isfinite(v[m]).all() and np.isnan(v[~m]).all())
    require((v[:,:9][m[:,:9]]>0).all() and (v[:,9:][m[:,9:]]>=0).all())
    for k in ('person_group','split','age_kind'):
        require(type(arrays[k]) is np.ndarray and arrays[k].shape==(n,) and arrays[k].dtype.kind in 'iu')
    require((arrays['person_group']>=0).all() and np.isin(arrays['split'],[0,1,2]).all())
    require(np.isin(arrays['age_kind'],range(len(AGE_KINDS))).all())
    adult=arrays['adult_qualified'];triplet=arrays['age_triplet']
    require(type(adult) is np.ndarray and adult.shape==(n,) and adult.dtype==np.bool_)
    require(type(triplet) is np.ndarray and triplet.shape==(n,3) and triplet.dtype.kind=='f')
    # Retain original observed/adult/source-train scope; broaden AGE REPRESENTATION,
    # never precision/assay truth or permission to use protected source rows.
    take=np.flatnonzero((arrays['split']==0)&adult&(m[:,:9].sum(1)>=2))
    require(len(take)>0)
    original_kind=arrays['age_kind'][take];a=triplet[take]
    value=np.full(len(take),np.nan);lower=value.copy();upper=value.copy();kind=np.full(len(take),3,np.int64)
    for j,name in enumerate(AGE_KINDS):
        mask=original_kind==j
        if name in ('reported_year','year_derived'):
            require(np.isfinite(a[mask,0]).all() and (a[mask,0]>=0).all())
            value[mask]=a[mask,0];kind[mask]=0
        elif name=='rounded':
            require(np.isfinite(a[mask,1:]).all() and (a[mask,1]>=0).all() and (a[mask,2]>=a[mask,1]).all())
            lower[mask]=a[mask,1];upper[mask]=a[mask,2];kind[mask]=1
        elif name in ('topcoded','rounded_topcoded'):
            require(np.isfinite(a[mask,1]).all() and (a[mask,1]>=0).all())
            lower[mask]=a[mask,1];kind[mask]=2
        # Source-unresolved or missing remains genuinely unknown.
    result=ClinicalPoolV2(SOURCES[source],v[take].copy(),m[take].copy(),arrays['person_group'][take].copy(),
        value,lower,upper,kind,take)
    for item in vars(result).values():
        if isinstance(item,np.ndarray):item.setflags(write=False)
    return result


def safe_summary(pool):
    require(isinstance(pool,ClinicalPoolV2))
    groups=len(np.unique(pool.person_group));rows=len(pool.values)
    supported=groups>=20
    return {'status':'supported_training_pool' if supported else 'insufficient_support',
        'source_local_people_lower_bound_20':groups//20*20 if supported else None,
        'training_examples_lower_bound_20':rows//20*20 if supported else None,
        'canonical_fields':list(FIELDS),'canonical_width':len(FIELDS),
        'exact_units_changed':False,'disease_labels_supplied':False,
        'new_encoder_trained':False,'patient_level_output_emitted':False}


def eligibility_hash(pool):
    h=hashlib.sha256()
    for name in ('original_indices','person_group','age_value','age_lower','age_upper','age_kind'):
        a=np.ascontiguousarray(getattr(pool,name));h.update(name.encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()
