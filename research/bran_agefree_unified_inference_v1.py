"""Private no-fit native inference for source-controlled clinical BRAN fits.

Neither inputs nor returned states/predictions may be printed or exported to a
hosted model. Caller authenticates source/checkpoint bytes and owns FD silence.
"""
from dataclasses import dataclass
import copy

import numpy as np
import torch

import bran_agefree_unified_jobs_v1 as jobs
import bran_native_cbc_decoders_v1 as cbc
import bran_native_screening_kernel_v1 as screen
import bran_missingness_stress_v1 as masking
from bran_raw_teacher_distillation_v1 import _validate_model
from bran_retinal_refit_inputs_v2 import digest

ARMS = ('control', 'candidate')
COMPLETION = {
    'partial_cbc_with_retina': ('single_target_hidden', True),
    'partial_cbc_no_retina': ('single_target_hidden', False),
    'whole_cbc_with_retina': ('whole_cbc_hidden', True),
    'whole_cbc_no_retina': ('whole_cbc_hidden', False),
}
ERROR = 'age-free unified inference rejected'


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, repr=False)
class PrivateInference:
    outputs: dict
    available: dict


def snapshots(fitted):
    require(type(fitted) is jobs.FittedJob and type(fitted.models) is dict
            and set(fitted.models) == set(ARMS)
            and digest(fitted.bundle['metadata']) == fitted._metadata_sha256
            and set(fitted.bundle['arms']) == set(ARMS))
    result = {}
    for arm in ARMS:
        model = fitted.models[arm]
        require(not model.training)
        _validate_model(model)
        saved = fitted.bundle['arms'][arm]
        jobs._same_normalizers(saved['normalizers'], jobs.normalizers(fitted.inputs.transform))
        state = model.state_dict()
        require(set(state) == set(saved['state_dict']))
        for name, value in state.items():
            target = saved['state_dict'][name]
            require(isinstance(target, torch.Tensor) and target.device.type == 'cpu'
                    and value.device.type == 'cpu' and target.dtype == value.dtype
                    and target.shape == value.shape and bool(torch.isfinite(target).all())
                    and torch.equal(value, target))
        result[arm] = copy.deepcopy(state)
    return result


def unchanged(fitted, before):
    after = snapshots(fitted)
    require(all(torch.equal(before[arm][name], value)
                for arm in ARMS for name, value in after[arm].items()))


def arrays(fitted):
    value = fitted.inputs.paired
    return (value.clinical, value.clinical_mask, value.retinal, value.retinal_present, value.age)


def screening(fitted):
    try:
        before = snapshots(fitted)
        outputs = {}
        for arm in ARMS:
            result = screen.predict_native(fitted.models[arm], *arrays(fitted))
            require(set(result) == {'both', 'clinical', 'retinal'})
            outputs[arm] = {name: readonly(value) for name, value in result.items()}
        unchanged(fitted, before)
        return PrivateInference(outputs, {})
    except Exception:
        raise ValueError(ERROR) from None


def completion(fitted, pattern):
    try:
        before = snapshots(fitted)
        require(type(pattern) is str and pattern in COMPLETION)
        native_pattern, use_retina = COMPLETION[pattern]
        c, cm, r, rm, age = arrays(fitted)
        if not use_retina:
            r, rm = np.zeros_like(r), np.zeros_like(rm)
        names = tuple(fitted.bundle['metadata']['registry_names'])
        transform = fitted.inputs.transform
        outputs, available = {}, {}
        for arm in ARMS:
            result, support = cbc.infer(fitted.models[arm], c, cm, r, rm, age,
                names, transform.clinical_median, transform.clinical_iqr,
                pattern=native_pattern, route='both' if use_retina else 'clinical')
            require(set(result) == {'native', 'generative'})
            outputs[arm] = {name: readonly(value) for name, value in result.items()}
            available[arm] = readonly(support)
        require(np.array_equal(available['control'], available['candidate']))
        unchanged(fitted, before)
        return PrivateInference(outputs, available)
    except Exception:
        raise ValueError(ERROR) from None


def missingness(fitted, pattern):
    try:
        before = snapshots(fitted)
        require(type(pattern) is str and pattern in masking.PATTERNS)
        c, cm, r, rm, age = arrays(fitted)
        names = tuple(fitted.bundle['metadata']['registry_names'])
        slots = tuple(names.index(field) for field in cbc.CBC_FIELDS)
        removed = masking.remove_inputs(c, cm, r, rm, slots, pattern)
        masking.assert_no_input_leak(removed, cm, rm, slots, pattern)
        outputs, available = {}, {}
        for arm in ARMS:
            result = screen.predict_native(fitted.models[arm], removed.clinical,
                removed.clinical_mask, removed.retinal, removed.retinal_mask, age)['both']
            require(np.array_equal(np.isfinite(result).all(axis=1), removed.available))
            outputs[arm] = readonly(result)
            available[arm] = readonly(removed.available)
        unchanged(fitted, before)
        return PrivateInference(outputs, available)
    except Exception:
        raise ValueError(ERROR) from None


def structure_states(fitted):
    """One discovery-only coordinate frame per arm, never pooled outer fits."""
    try:
        before = snapshots(fitted)
        require(fitted.bundle['metadata']['job'] == 'structure')
        c, cm, r, rm, age = arrays(fitted)
        outputs, available = {}, {}
        with torch.inference_mode():
            for arm in ARMS:
                states = np.empty((len(age), 192), np.float32)
                support = np.empty(len(age), bool)
                for start in range(0, len(age), 256):
                    stop = min(start + 256, len(age))
                    ct = torch.tensor(c[start:stop], dtype=torch.float32)
                    mt = torch.tensor(cm[start:stop], dtype=torch.bool)
                    rt = torch.tensor(r[start:stop, None], dtype=torch.float32)
                    rmt = torch.tensor(rm[start:stop, None], dtype=torch.bool)
                    at = torch.tensor(age[start:stop], dtype=torch.float32)
                    state = fitted.models[arm].encode(torch.where(mt, ct, 0.), mt,
                        torch.where(rmt[..., None], rt, 0.), rmt, at)
                    require(tuple(state.mean.shape) == (stop - start, 192)
                            and tuple(state.abstain.shape) == (stop - start,))
                    states[start:stop] = state.mean.cpu().numpy()
                    support[start:stop] = (~state.abstain).cpu().numpy()
                require(np.isfinite(states).all())
                outputs[arm], available[arm] = readonly(states), readonly(support)
        require(np.array_equal(available['control'], available['candidate']))
        unchanged(fitted, before)
        return PrivateInference(outputs, available)
    except Exception:
        raise ValueError(ERROR) from None
