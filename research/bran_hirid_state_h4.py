"""Local-only frozen H4 state extraction; no source I/O, fitting or output."""
from dataclasses import dataclass

import numpy as np
import torch

import bran_hirid_v5_inference_v1 as native
from bran_multisource_age_v2 import normalize_age
from bran_multisource_batches_v2 import tensor, transform_hash
from bran_multisource_inference_v2 import _clean_for_encode

ERROR = 'bran_hirid_h4_state_contract_failed'


def require(ok):
    if not ok:
        raise ValueError(ERROR) from None


@dataclass(frozen=True, repr=False)
class PrivateH4State:
    states: np.ndarray
    native_hb: np.ndarray
    available: np.ndarray

    def __repr__(self):
        return '<PrivateH4State>'

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)


def extract(batch, model, transform, transform_pin, *, batch_size=256):
    try:
        native._validate_model(model)
        native._validate_transform(transform, model, transform_pin)
        n, batch = native._validate_batch(batch)
        require(type(batch_size) is int and batch_size > 0)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        reference = native.infer(batch, model, transform, transform_pin, batch_size=batch_size)
        values = np.where(batch.clinical_mask, batch.clinical_values, 0.).astype(np.float64)
        clinical, cm = transform.clinical(values, batch.clinical_mask)
        retinal, rm = transform.retinal(np.zeros((n, 384)), np.zeros(n, dtype=bool))
        c, cm = tensor(clinical), tensor(cm, torch.bool)
        r, rm = tensor(retinal), tensor(rm, torch.bool)
        age = normalize_age(batch.ages, transform.age_mean, transform.age_scale)
        states = np.full((n, 192), np.nan, dtype=np.float64)
        predictions = np.full(n, np.nan, dtype=np.float64)
        available = np.zeros(n, dtype=bool)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                clean_c, clean_r = _clean_for_encode(model, c[start:stop], cm[start:stop], r[start:stop], rm[start:stop])
                state = model.encode(clean_c, cm[start:stop], clean_r, rm[start:stop, None], age[start:stop])
                require(state.mean.shape == (stop-start, 192) and state.abstain.shape == (stop-start,))
                active = (~state.abstain).cpu().numpy() & batch.clinical_eligible[start:stop]
                z = state.mean.cpu().numpy()
                standardized = model.cbc_joint_head(state.mean).cpu().numpy()[:, native._HB_POSITION]
                column = native.CANONICAL_INDEX['hemoglobin']
                hb = standardized * transform.clinical_iqr[column] + transform.clinical_median[column]
                require(np.isfinite(z[active]).all() and np.isfinite(hb[active]).all())
                states[start:stop][active] = z[active]
                predictions[start:stop][active] = hb[active]
                available[start:stop] = active
        require(np.array_equal(available, reference.available))
        require(np.array_equal(predictions, reference.native_hemoglobin, equal_nan=True))
        require(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
        native._validate_model(model)
        require(transform_hash(transform) == transform_pin)
        for array in (states, predictions, available):
            array.setflags(write=False)
        return PrivateH4State(states, predictions, available)
    except BaseException:
        raise ValueError(ERROR) from None
