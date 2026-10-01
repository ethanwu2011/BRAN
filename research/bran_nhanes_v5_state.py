"""Private NHANES clinical-only V5 fold0 frame, including censored/missing age."""
from dataclasses import dataclass
import numpy as np
import torch

from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_knhanes_input_kernel_v1 import CANONICAL_NAMES,ADMITTED_CANONICAL_INDICES
from bran_joint_lab_task_contract_v1 import project_joint_labs_to_registry
from bran_multisource_age_v2 import AgeBatch,validate_age
from bran_multisource_batches_v2 import tensor,transform_hash
from bran_multisource_profiles_v3 import _validate_provider,_unchanged
from bran_v5_state_routes import state_routes


def require(ok):
    if not ok:raise ValueError('nhanes_v5_state_contract_failed')


def readonly(x):
    x=np.array(x,copy=True);x.setflags(write=False);return x


@dataclass(frozen=True,repr=False)
class PrivateState:
    state:np.ndarray
    available:np.ndarray
    clinical:np.ndarray
    clinical_mask:np.ndarray

    def __repr__(self):return '<PrivateNHANESV5State>'
    def __reduce__(self):raise TypeError('private_state_serialization_forbidden')


def infer(arrays,model,transform,expected_transform_sha256):
    """No I/O or normalization fit; phenotype/weight/split fields are never read.

    All source CBC/chemistry records can be encoded. Disease/age/outcome domains
    are selected separately by the evaluator, not passed into the state. Missing
    age is supported even though the prospective age40 cohort cannot use it to
    establish membership. Caller authenticates exact V5 checkpoint provenance.
    """
    try:
        require(transform_hash(transform)==expected_transform_sha256)
        slots=tuple(CANONICAL_NAMES.index(f) for f in CBC_FIELDS)
        before,thash,grads=_validate_provider(model,transform,0,slots,transform)
        eligible=np.zeros(59,bool);eligible[list(ADMITTED_CANONICAL_INDICES)]=True
        require(np.array_equal(transform.eligible,eligible))
        c,cm=project_joint_labs_to_registry(arrays['values'],arrays['observed'],
            arrays['observed'].astype(np.uint8),CANONICAL_NAMES,np.zeros(59),np.ones(59))
        c=c.copy();cm=cm.copy();cm[:,~eligible]=False;c[~cm]=0
        values,masks=transform.clinical(c,cm);n=len(c)
        age=validate_age(AgeBatch(tensor(arrays['age_value']),tensor(arrays['age_lower']),
            tensor(arrays['age_upper']),tensor(arrays['age_kind'],torch.long)))
        require(age.value.shape==(n,))
        routes=state_routes(model,tensor(values),tensor(masks,torch.bool),tensor(np.zeros((n,384))),
            tensor(np.zeros(n,bool),torch.bool),age,transform.age_mean,transform.age_scale)
        require(_unchanged(before,thash,grads,model,transform))
        return PrivateState(readonly(routes.states['clinical'].numpy()),
            readonly(routes.available['clinical'].numpy()),readonly(c),readonly(cm))
    except Exception:raise ValueError('nhanes_v5_state_contract_failed') from None
