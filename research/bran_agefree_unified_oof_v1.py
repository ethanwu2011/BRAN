"""Private five-fold inference assembly for the matched clinical-source transfer.

No source access, fitting, persistence, or patient-derived public output. The
provider supplies byte-authenticated, already fitted/restored jobs; this layer
rechecks exact memberships, transforms, budgets, and state before scoring only
that job's held-out fold. Never pool latent coordinates across these jobs.
"""
import hashlib
from dataclasses import dataclass, field

import numpy as np
import torch

import bran_agefree_unified_jobs_v1 as jobs
import bran_agefree_unified_inference_v1 as inference
import bran_missingness_stress_v1 as masking

ERROR = 'age-free refit out-of-fold contract rejected'
COMPLETION = {
    'single_target_hidden': 'partial_cbc_with_retina',
    'whole_cbc_hidden': 'whole_cbc_with_retina',
    'single_target_no_retina': 'partial_cbc_no_retina',
    'whole_cbc_no_retina': 'whole_cbc_no_retina',
}
HEADS = ('native', 'generative')
ROUTES = ('both', 'clinical', 'retinal')


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def private_digest(value):
    """In-memory integrity only; digest and inputs must remain private.

    Distinguishes numeric dtype/shape and scalar types; never pickle/repr rows.
    File authentication belongs to the future execution/audit caller.
    """
    h = hashlib.sha256()

    def add(item):
        if type(item) is np.ndarray:
            require(item.dtype.kind in 'biuf' and not item.dtype.hasobject)
            h.update(b'array'); add(item.dtype.str); add(list(item.shape))
            raw = np.ascontiguousarray(item)
            h.update(str(raw.nbytes).encode() + b':'); h.update(raw.tobytes())
        elif type(item) is dict:
            require(all(type(k) is str for k in item))
            h.update(b'dict'); add(len(item))
            for key in sorted(item):
                add(key); add(item[key])
        elif type(item) in (list, tuple):
            h.update(b'list' if type(item) is list else b'tuple'); add(len(item))
            for child in item:
                add(child)
        elif item is None or type(item) in (str, int, float, bool):
            encoded = jobs.handoff.digest({'type': type(item).__name__, 'value': item}).encode()
            h.update(encoded)
        else:
            require(False)
    add(value)
    return h.hexdigest()


def context_binding(*, data, plan, plan_sha256, inner_folds, registry_names,
                    endpoint_names, protocol_sha256, source_descriptor_sha256):
    require(type(inner_folds) is dict and set(inner_folds) == set(range(5)))
    return private_digest({'data': data, 'plan': plan, 'plan_sha256': plan_sha256,
        'inner_folds': {str(k): v for k, v in inner_folds.items()},
        'registry_names': registry_names, 'endpoint_names': endpoint_names,
        'protocol_sha256': protocol_sha256, 'source_descriptor_sha256': source_descriptor_sha256})


def _seal(value):
    if type(value) is np.ndarray:
        value.setflags(write=False)
    elif type(value) is dict:
        for child in value.values():
            _seal(child)


@dataclass(frozen=True, repr=False)
class PrivateOOF:
    binding: str
    screening: dict
    completion: dict
    completion_available: dict
    stress: dict
    _contents_sha256: str = field(init=False, repr=False)

    def __post_init__(self):
        _seal(self.payload())
        object.__setattr__(self, '_contents_sha256', private_digest(self.payload()))

    def payload(self):
        return {key: getattr(self, key) for key in
                ('binding', 'screening', 'completion', 'completion_available', 'stress')}


def _prediction(value, shape, available, *, probability=False):
    require(type(value) is np.ndarray and value.shape == shape and value.dtype.kind == 'f'
            and np.array_equal(np.isfinite(value), np.broadcast_to(available, shape))
            and np.isnan(value[~np.broadcast_to(available, shape)]).all())
    if probability:
        require(np.all((value[np.isfinite(value)] >= 0) & (value[np.isfinite(value)] <= 1)))


def validate_private(value, *, expected_binding, n):
    """Closed private schema and immutable payload; does not release any data."""
    require(type(value) is PrivateOOF and type(n) is int and n > 0
            and type(expected_binding) is str and len(expected_binding) == 64
            and value.binding == expected_binding
            and value._contents_sha256 == private_digest(value.payload()))
    require(set(value.screening) == set(value.completion) == set(value.stress) == set(jobs.ARMS)
            and set(value.completion_available) == set(COMPLETION))
    for pattern, support in value.completion_available.items():
        require(type(support) is np.ndarray and support.shape == (n, 9) and support.dtype == bool)
    for arm in jobs.ARMS:
        require(type(value.screening[arm]) is dict and set(value.screening[arm]) == set(ROUTES)
                and type(value.completion[arm]) is dict and set(value.completion[arm]) == set(COMPLETION)
                and type(value.stress[arm]) is dict and set(value.stress[arm]) == set(masking.PATTERNS))
        for route, prediction in value.screening[arm].items():
            _prediction(prediction, (n, 26), np.isfinite(prediction).all(1)[:, None], probability=True)
            other = value.screening['control'][route]
            require(np.array_equal(np.isnan(prediction), np.isnan(other)))
        for pattern, heads in value.completion[arm].items():
            require(type(heads) is dict and set(heads) == set(HEADS))
            for prediction in heads.values():
                _prediction(prediction, (n, 9), value.completion_available[pattern])
        for pattern, prediction in value.stress[arm].items():
            _prediction(prediction, (n, 26), np.isfinite(prediction).all(1)[:, None], probability=True)
            require(np.array_equal(np.isnan(prediction), np.isnan(value.stress['control'][pattern])))
        require(np.array_equal(value.stress[arm]['available'], value.screening[arm]['both'], equal_nan=True))


def collect(*, data, plan, plan_sha256, inner_folds, registry_names, endpoint_names,
            protocol_sha256, source_descriptor_sha256, provider, external_steps=3000, paired_steps=1500, joint_steps=1500, batch_size=96):
    """Assemble full OOF predictions from five EXACT held-out jobs, never states.

    provider(fold) may load or fit according to the caller's frozen execution
    mode. This function itself never trains and must not be used as permission
    to train before the required source/reference authentication.
    """
    try:
        args = dict(data=data, plan=plan, plan_sha256=plan_sha256, inner_folds=inner_folds,
            registry_names=registry_names, endpoint_names=endpoint_names, protocol_sha256=protocol_sha256,
            source_descriptor_sha256=source_descriptor_sha256)
        binding = context_binding(**args)
        jobs.membership.authenticate(plan, plan_sha256, patient_ids=data['patient_ids'],
                                     folds=data['folds'], inner_folds=inner_folds)
        n = len(data['folds']); slots = tuple(registry_names.index(x) for x in jobs.CBC_FIELDS)
        screen = {arm: {route: np.full((n, 26), np.nan) for route in ROUTES} for arm in jobs.ARMS}
        blood = {arm: {p: {h: np.full((n, 9), np.nan) for h in HEADS} for p in COMPLETION} for arm in jobs.ARMS}
        support = {p: np.zeros((n, 9), bool) for p in COMPLETION}
        stress = {arm: {p: np.full((n, 26), np.nan) for p in masking.PATTERNS} for arm in jobs.ARMS}
        seen = np.zeros(n, bool)
        for fold in range(5):
            supplied = provider(fold)
            require(context_binding(**args) == binding)
            inference.snapshots(supplied)
            fitted = jobs.restore_job(supplied.bundle, **args, job='outer' + str(fold),
                external_steps=external_steps, paired_steps=paired_steps, joint_steps=joint_steps, batch_size=batch_size)
            te = np.flatnonzero(data['folds'] == fold)
            require(np.array_equal(te, np.flatnonzero(data['folds'] == fold)) and not seen[te].any())
            pp = inference.screening(fitted)
            for arm in jobs.ARMS:
                for route in ROUTES:
                    screen[arm][route][te] = pp.outputs[arm][route][te]
            for pattern, native_name in COMPLETION.items():
                out = inference.completion(fitted, native_name)
                require(np.array_equal(out.available['control'], out.available['candidate']))
                support[pattern][te] = out.available['control'][te]
                for arm in jobs.ARMS:
                    for head in HEADS:
                        blood[arm][pattern][head][te] = out.outputs[arm][head][te]
            snapshot = inference.snapshots(fitted)
            with torch.inference_mode():
                for pattern in masking.PATTERNS:
                    same_masks = None
                    for arm in jobs.ARMS:
                        c, cm, r, rm, age = inference.arrays(fitted)
                        # Draw masks on the complete, fixed patient order, then
                        # score this fold. Do not restart RNG on a fold slice.
                        x = masking.remove_inputs(c, cm, r, rm, slots, pattern)
                        masking.assert_no_input_leak(x, cm, rm, slots, pattern)
                        current_masks = (x.clinical_mask, x.retinal_mask, x.available)
                        if same_masks is not None:
                            require(all(np.array_equal(a, b) for a, b in zip(same_masks, current_masks)))
                        same_masks = current_masks
                        values = inference.screen.predict_native(fitted.models[arm],
                            x.clinical[te], x.clinical_mask[te], x.retinal[te],
                            x.retinal_mask[te], age[te])['both']
                        _prediction(values, (len(te), 26), x.available[te, None], probability=True)
                        stress[arm][pattern][te] = values
            inference.unchanged(fitted, snapshot)
            seen[te] = True
        require(seen.all() and context_binding(**args) == binding)
        result = PrivateOOF(binding, screen, blood, support, stress)
        validate_private(result, expected_binding=binding, n=n)
        return result
    except Exception:
        raise ValueError(ERROR) from None
