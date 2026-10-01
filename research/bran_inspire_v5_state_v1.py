"""Bounded-memory, local-only INSPIRE inference in the frozen V5 fold-0 frame.

No source I/O, fitting, labels, identifiers, context, or partition inputs.
Caller must authenticate source units, checkpoint and transform before calling.
"""
from dataclasses import dataclass
import numpy as np
import torch

import bran_inspire_input_adapter_v1 as adapter
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_knhanes_input_kernel_v1 import CANONICAL_INDEX, ADMITTED_CANONICAL_INDICES
from bran_multisource_age_v2 import normalize_age
from bran_multisource_batches_v2 import tensor, transform_hash
from bran_multisource_profiles_v3 import _validate_provider, _unchanged
from bran_multisource_inference_v2 import _validate_structure, _validate_visible_values, _clean_for_encode

ERROR = 'bran_inspire_v5_state_v1_failed'


def require(ok):
    if not ok:
        raise ValueError(ERROR)


@dataclass(frozen=True, slots=True, repr=False)
class PrivateInspireState:
    state: np.ndarray
    available: np.ndarray

    def __repr__(self):
        return '<PrivateInspireState>'

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)


def infer(values, observed, age_lower, age_upper, age_kind, model, transform,
          expected_transform_sha256, *, batch_size=256):
    """Encode only physiology and typed age; missing physiology must abstain.

    Adapter construction is chunked so absent retinal tensors are never
    allocated for the entire source cohort. All returned arrays remain private.
    """
    try:
        require(type(batch_size) is int and 1 <= batch_size <= 4096)
        require(type(values) is np.ndarray and values.ndim == 2 and values.shape[1] == 18)
        n = len(values)
        require(type(observed) is np.ndarray and observed.shape == values.shape and observed.dtype == bool)
        for age in (age_lower, age_upper, age_kind):
            require(type(age) is np.ndarray and age.shape == (n,))
        require(transform_hash(transform) == expected_transform_sha256)
        slots = tuple(CANONICAL_INDEX[f] for f in CBC_FIELDS)
        before, thash, grads = _validate_provider(model, transform, 0, slots, transform)
        require(set(model.eligible_indices)==set(ADMITTED_CANONICAL_INDICES))
        eligible = np.zeros(59, bool)
        eligible[list(ADMITTED_CANONICAL_INDICES)] = True
        require(np.array_equal(transform.eligible, eligible))
        state = np.zeros((n, 192), np.float32)
        available = np.zeros(n, bool)
        # Validate the public array contract even for a zero-row call.
        if n == 0:
            adapter.prepare_inputs(values, observed, age_lower, age_upper, age_kind)
        with torch.no_grad():
            for first in range(0, n, batch_size):
                last = min(n, first + batch_size)
                batch = adapter.prepare_inputs(values[first:last], observed[first:last],
                    age_lower[first:last], age_upper[first:last], age_kind[first:last])
                clinical, cm = transform.clinical(
                    np.where(batch.clinical_mask, batch.clinical_values, 0.), batch.clinical_mask)
                c, mask = tensor(clinical), tensor(cm, torch.bool)
                retina = torch.zeros((last-first, 384), dtype=c.dtype)
                rm = torch.zeros(last-first, dtype=torch.bool)
                age = batch.ages
                _validate_structure(model, c, mask, retina, rm, age, batch_size)
                _validate_visible_values(model, c, mask, retina, rm)
                clean, clean_retina = _clean_for_encode(model, c, mask, retina, rm)
                age7 = normalize_age(age, transform.age_mean, transform.age_scale).to(c.dtype)
                posterior = model.encode(clean, mask, clean_retina, rm[:, None], age7)
                require(posterior.mean.shape == (last-first, 192)
                    and posterior.abstain.shape == (last-first,)
                    and posterior.abstain.dtype == torch.bool
                    and bool(torch.isfinite(posterior.mean).all()))
                state[first:last] = posterior.mean.detach().cpu().numpy()
                available[first:last] = ~posterior.abstain.detach().cpu().numpy()
                require(np.array_equal(available[first:last], batch.eligible))
                require(np.all(state[first:last][~available[first:last]] == 0))
        require(_unchanged(before, thash, grads, model, transform))
        state.setflags(write=False)
        available.setflags(write=False)
        return PrivateInspireState(state, available)
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None
