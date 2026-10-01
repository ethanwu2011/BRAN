"""Private no-fit inference for freshly restored matched BRAN refit jobs."""
from __future__ import annotations

from dataclasses import dataclass
import copy
from collections.abc import Mapping

import numpy as np

import bran_native_cbc_decoders_v1 as native_cbc
import bran_native_screening_kernel_v1 as native_screening
import bran_retinal_unified_refit_jobs_v1 as jobs
import bran_retinal_unified_refit_training_v1 as training


_ERROR = 'retinal unified refit inference rejected'
_ARMS = ('original', 'candidate')
_COMPLETION = {
    'partial_cbc_with_retina': ('single_target_hidden', True),
    'partial_cbc_no_retina': ('single_target_hidden', False),
    'whole_cbc_with_retina': ('whole_cbc_hidden', True),
    'whole_cbc_no_retina': ('whole_cbc_hidden', False),
}


def _fail():
    raise ValueError(_ERROR) from None


def _require(value):
    if not value:
        _fail()


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(repr=False)
class PrivateScreening:
    outputs: dict


@dataclass(repr=False)
class PrivateCompletion:
    pattern: str
    outputs: dict
    available: dict


@dataclass(repr=False)
class PrivateStructureStates:
    states: dict
    available: dict


def _snapshots(fitted):
    """Reject a non-restored/eval model and copy its authenticated state."""
    try:
        import torch

        _require(type(fitted) is jobs.FittedJob and type(fitted.models) is dict
                 and set(fitted.models) == set(_ARMS) and type(fitted.bundle) is dict
                 and type(fitted.bundle.get('arms')) is dict and set(fitted.bundle['arms']) == set(_ARMS)
                 and type(fitted.bundle.get('metadata')) is dict
                 and type(getattr(fitted, '_metadata_sha256', None)) is str
                 and jobs.handoff.digest(fitted.bundle['metadata']) == fitted._metadata_sha256)
        result = {}
        for arm in _ARMS:
            model = fitted.models[arm]
            _require(not model.training)
            training._validate_model(model)
            saved = fitted.bundle['arms'][arm]
            _require(type(saved) is dict and isinstance(saved.get('state_dict'), Mapping))
            jobs._same_normalizers(saved.get('normalizers'), jobs.normalizers(fitted.inputs, arm))
            current = model.state_dict()
            _require(set(saved['state_dict']) == set(current))
            copied = copy.deepcopy(saved['state_dict'])
            for name, tensor in copied.items():
                _require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu'
                         and tensor.dtype == current[name].dtype and tensor.shape == current[name].shape
                         and bool(torch.isfinite(tensor).all()) and torch.equal(current[name], tensor))
            result[arm] = copied
        return result
    except Exception:
        _fail()


def _unchanged(fitted, snapshots):
    try:
        import torch
        current_saved = _snapshots(fitted)
        for arm in _ARMS:
            current = fitted.models[arm].state_dict()
            _require(set(current) == set(snapshots[arm])
                     and all(torch.equal(value, snapshots[arm][name])
                             and torch.equal(current_saved[arm][name], snapshots[arm][name])
                             for name, value in current.items()))
    except Exception:
        _fail()


def _arrays(fitted, arm):
    values = getattr(fitted.inputs, arm, None)
    present = getattr(fitted.inputs, 'retinal_present', None)
    _require(type(values) is tuple and len(values) == 4 and type(present) is np.ndarray
             and present.dtype == np.dtype(bool))
    clinical, clinical_mask, retinal, age = values
    n = len(present)
    _require(type(clinical) is np.ndarray and clinical.shape == (n, 59)
             and type(clinical_mask) is np.ndarray and clinical_mask.shape == clinical.shape
             and clinical_mask.dtype == np.dtype(bool)
             and type(retinal) is np.ndarray and retinal.shape == (n, 384)
             and type(age) is np.ndarray and age.shape == (n,))
    return clinical, clinical_mask, retinal, present, age


def screening(fitted):
    """Private full/clinical/retinal native screening outputs for both arms."""
    snapshots = _snapshots(fitted)
    try:
        import torch
        outputs = {}
        with torch.inference_mode():
            for arm in _ARMS:
                clinical, mask, retinal, present, age = _arrays(fitted, arm)
                result = native_screening.predict_native(fitted.models[arm], clinical, mask, retinal, present, age)
                _require(set(result) == {'both', 'clinical', 'retinal'})
                outputs[arm] = {route: _readonly(value) for route, value in result.items()}
        _unchanged(fitted, snapshots)
        return PrivateScreening(outputs)
    except Exception:
        _fail()


def completion(fitted, registry_names, pattern):
    """Private native/generative CBC completions for one fixed route/mask mode."""
    snapshots = _snapshots(fitted)
    try:
        import torch
        _require(type(registry_names) is tuple and len(registry_names) == len(set(registry_names)) == 59
                 and all(type(name) is str and name for name in registry_names)
                 and registry_names == tuple(fitted.bundle['metadata'].get('registry_names', ()))
                 and type(pattern) is str and pattern in _COMPLETION)
        native_pattern, use_retina = _COMPLETION[pattern]
        outputs, available = {}, {}
        with torch.inference_mode():
            for arm in _ARMS:
                clinical, mask, retinal, present, age = _arrays(fitted, arm)
                transform = getattr(fitted.inputs, arm + '_transform', None)
                median, iqr = getattr(transform, 'clinical_median', None), getattr(transform, 'clinical_iqr', None)
                if use_retina:
                    route, supplied_retinal, supplied_present = 'both', retinal, present
                else:
                    # The explicit zero copies make no-retina independent of
                    # any caller-retained retinal payload, beyond cbc.infer's
                    # own clinical-route masking.
                    route = 'clinical'
                    supplied_retinal, supplied_present = np.zeros_like(retinal), np.zeros_like(present)
                values, support = native_cbc.infer(
                    fitted.models[arm], clinical, mask, supplied_retinal, supplied_present, age,
                    registry_names, median, iqr, pattern=native_pattern, route=route,
                )
                _require(set(values) == {'native', 'generative'})
                outputs[arm] = {name: _readonly(value) for name, value in values.items()}
                available[arm] = _readonly(support)
        _unchanged(fitted, snapshots)
        return PrivateCompletion(pattern, outputs, available)
    except Exception:
        _fail()


def structure_states(fitted):
    """Private per-person 192-D states from the one discovery-only job."""
    snapshots = _snapshots(fitted)
    try:
        import torch
        _require(type(fitted.bundle.get('metadata')) is dict and fitted.bundle['metadata'].get('job') == 'structure')
        states, available = {}, {}
        with torch.inference_mode():
            for arm in _ARMS:
                model = fitted.models[arm]
                clinical, mask, retinal, present, age = _arrays(fitted, arm)
                n = len(age)
                values = np.empty((n, 192), dtype=np.float32)
                supported = np.empty(n, dtype=bool)
                for start in range(0, n, 256):
                    stop = min(n, start + 256)
                    ct = torch.tensor(clinical[start:stop], dtype=torch.float32)
                    cmt = torch.tensor(mask[start:stop], dtype=torch.bool)
                    rt = torch.tensor(retinal[start:stop, None], dtype=torch.float32)
                    rmt = torch.tensor(present[start:stop, None], dtype=torch.bool)
                    at = torch.tensor(age[start:stop], dtype=torch.float32)
                    clean_c = torch.where(cmt, ct, torch.zeros_like(ct))
                    clean_r = torch.where(rmt[..., None], rt, torch.zeros_like(rt))
                    state = model.encode(clean_c, cmt, clean_r, rmt, at)
                    _require(tuple(state.mean.shape) == (stop - start, 192)
                             and tuple(state.abstain.shape) == (stop - start,))
                    values[start:stop] = state.mean.detach().cpu().numpy()
                    supported[start:stop] = (~state.abstain).detach().cpu().numpy()
                _require(np.isfinite(values).all())
                states[arm], available[arm] = _readonly(values), _readonly(supported)
        _unchanged(fitted, snapshots)
        return PrivateStructureStates(states, available)
    except Exception:
        _fail()
