"""Source-free, bounded reuse of the authenticated V5 21-lab state interface.

Only original-unit physiology, masks and typed age are accepted. Source/role/
label qualification and checkpoint provenance remain the caller's responsibility.
"""
from dataclasses import dataclass
import numpy as np
import bran_nhanes_v5_state as existing

ERROR = 'broad_clinical_state_v1_contract_failed'


def require(ok):
    if not ok: raise ValueError(ERROR) from None


@dataclass(frozen=True, repr=False)
class PrivateBroadState:
    state: np.ndarray
    available: np.ndarray
    clinical: np.ndarray
    clinical_mask: np.ndarray

    def __post_init__(self):
        n = len(self.state)
        contract = {'state': ((n,192), np.float32), 'available': ((n,), bool),
                    'clinical': ((n,59), np.float64), 'clinical_mask': ((n,59), bool)}
        for key, (shape, dtype) in contract.items():
            v = getattr(self,key)
            require(type(v) is np.ndarray and v.shape == shape and v.dtype == np.dtype(dtype))
            copied = v.copy(); copied.setflags(write=False); object.__setattr__(self,key,copied)
        require(np.isfinite(self.state).all() and np.isfinite(self.clinical).all()
                and (self.clinical[~self.clinical_mask] == 0).all()
                and np.array_equal(self.available,self.clinical_mask.any(1))
                and (self.state[~self.available] == 0).all())

    def __repr__(self): return '<PrivateBroadClinicalState>'
    def __reduce__(self): raise TypeError(ERROR)


def infer(values, observed, age_value, age_lower, age_upper, age_kind, model,
          transform, expected_transform_sha256, *, batch_size=256):
    """No filling of physiological targets; inactive payloads cannot enter state."""
    try:
        require(type(batch_size) is int and 1 <= batch_size <= 4096)
        require(type(values) is np.ndarray and values.ndim == 2 and values.shape[1] == 21
                and values.dtype == np.float64)
        n = len(values)
        require(type(observed) is np.ndarray and observed.shape == values.shape and observed.dtype == bool)
        age = {'age_value': age_value, 'age_lower': age_lower, 'age_upper': age_upper, 'age_kind': age_kind}
        for key,v in age.items():
            require(type(v) is np.ndarray and v.shape == (n,)
                    and v.dtype == np.dtype(np.int64 if key=='age_kind' else np.float64))
        require(np.isin(age_kind,(0,1,2,3)).all())
        outputs = []
        for first in range(0,max(1,n),batch_size):
            last = min(first+batch_size,n)
            if n:
                inputs = {'values':np.where(observed[first:last],values[first:last],np.nan),
                          'observed':observed[first:last].copy(),
                          **{k:v[first:last].copy() for k,v in age.items()}}
            else:
                # Authenticate even the empty-call model/transform contract.
                inputs = {'values':np.full((1,21),np.nan), 'observed':np.zeros((1,21),bool),
                          'age_value':np.full(1,np.nan), 'age_lower':np.full(1,np.nan),
                          'age_upper':np.full(1,np.nan), 'age_kind':np.full(1,3,np.int64)}
            r = existing.infer(inputs,model,transform,expected_transform_sha256)
            take = slice(None) if n else slice(0,0)
            outputs.append(PrivateBroadState(r.state[take],r.available[take],r.clinical[take],r.clinical_mask[take]))
        return PrivateBroadState(*(np.concatenate([getattr(x,k) for x in outputs])
            for k in ('state','available','clinical','clinical_mask')))
    except Exception:
        raise ValueError(ERROR) from None
