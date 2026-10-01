"""Local V5 fold0 external inference; receives authenticated objects, no I/O."""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import torch

from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_knhanes_phase9_adapter import PrivateKNHANESPhase9Prepared
from bran_knhanes_input_kernel_v1 import CANONICAL_NAMES, ADMITTED_CANONICAL_INDICES
from bran_multisource_age_v2 import AgeBatch, REPORTED
from bran_multisource_batches_v2 import tensor, transform_hash
from bran_multisource_profiles_v3 import _validate_provider, _unchanged
from bran_v5_state_routes import state_routes
from bran_knhanes_v5_evaluation import _STATE_FLAGS

_ERROR='knhanes_v5_inference_contract_failed'


def require(ok):
    if not ok: raise ValueError(_ERROR)


def readonly(value):
    value=np.array(value,copy=True);value.setflags(write=False);return value


@dataclass(frozen=True,repr=False)
class PrivateExternalState:
    states: np.ndarray
    native_hemoglobin: np.ndarray
    available: np.ndarray
    provider_flags: object

    def __repr__(self): return '<PrivateV5ExternalState>'
    def __reduce__(self): raise TypeError(_ERROR)
    def __reduce_ex__(self,protocol): raise TypeError(_ERROR)


def infer(prepared,model,transform,expected_transform_sha256):
    """Caller must authenticate V5-M fold0 checkpoint bytes before this call.

    Only adult observed-target/input-eligible people are encoded. Every other
    person stays in the survey design with no emitted prediction. This changes
    neither teacher weights nor normalizers. No residual correction is used.
    """
    try:
        require(isinstance(prepared,PrivateKNHANESPhase9Prepared))
        require(transform_hash(transform)==expected_transform_sha256)
        slots=tuple(CANONICAL_NAMES.index(field) for field in CBC_FIELDS)
        before,thash,grads=_validate_provider(model,transform,0,slots,transform)
        eligible=np.zeros(59,bool);eligible[list(ADMITTED_CANONICAL_INDICES)]=True
        require(np.array_equal(transform.eligible,eligible))
        x=prepared.inputs;n=len(x.ages)
        require(x.clinical_mask.shape==(n,59) and not x.clinical_mask[:,slots].any()
                and not x.clinical_mask[:,~eligible].any() and not x.retinal_mask.any())
        chosen=x.eligible & prepared.target_masks.eligible
        require(np.all(~chosen | (np.isfinite(x.ages)&(x.ages>=19)&(x.ages<=79))))
        states=np.full((n,192),np.nan);native=np.full(n,np.nan);available=np.zeros(n,bool)
        rows=np.flatnonzero(chosen)
        if len(rows):
            c,cm=transform.clinical(x.clinical_values[rows],x.clinical_mask[rows])
            r=np.zeros((len(rows),384),np.float32);rm=np.zeros(len(rows),bool)
            nan=torch.full((len(rows),),float('nan'))
            age=AgeBatch(tensor(x.ages[rows]),nan,nan,torch.full((len(rows),),REPORTED,dtype=torch.long))
            routes=state_routes(model,tensor(c),tensor(cm,torch.bool),tensor(r),tensor(rm,torch.bool),
                                age,transform.age_mean,transform.age_scale)
            state=routes.states['clinical'];valid=routes.available['clinical'].numpy()
            with torch.no_grad():
                hb=model.cbc_joint_head(state)[:,CBC_FIELDS.index('hemoglobin')].numpy()
            hb_slot=CANONICAL_NAMES.index('hemoglobin')
            hb=hb*transform.clinical_iqr[hb_slot]+transform.clinical_median[hb_slot]
            selected=rows[valid]
            require(np.isfinite(hb[valid]).all())
            states[selected]=state.numpy()[valid];native[selected]=hb[valid];available[selected]=True
        require(_unchanged(before,thash,grads,model,transform))
        return PrivateExternalState(readonly(states),readonly(native),readonly(available),
            MappingProxyType({key:True for key in _STATE_FLAGS}))
    except Exception:
        raise ValueError(_ERROR) from None
