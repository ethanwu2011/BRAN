"""Local diagnostic designs that preserve separate clinical and retinal routes.

``union`` is a deterministic concatenation of two route-specific posterior means.
It is not a new joint posterior, calibrated representation, or an input accepted
by the existing 192-coordinate decoders.  Returned arrays remain caller-owned.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

import bran_research_bundle_api_v1 as research_api


_INPUT_ERROR = "modality_union_input_invalid"
_INFERENCE_ERROR = "modality_union_inference_invalid"


def _invalid() -> None:
    raise ValueError(_INPUT_ERROR)


def _as_inputs(clinical, clinical_mask, retinal, retinal_mask, age):
    """Validate and copy inputs, selecting only finite observed payloads."""
    c = np.asarray(clinical)
    cm = np.asarray(clinical_mask)
    r = np.asarray(retinal)
    rm = np.asarray(retinal_mask)
    a = np.asarray(age)
    if (
        c.ndim != 2
        or c.shape[1] != 59
        or c.shape[0] == 0
        or cm.shape != c.shape
        or r.shape != (c.shape[0], 384)
        or rm.shape != (c.shape[0],)
        or a.shape != (c.shape[0],)
        or cm.dtype != bool
        or rm.dtype != bool
        or not np.issubdtype(c.dtype, np.floating)
        or not np.issubdtype(r.dtype, np.floating)
        or not np.issubdtype(a.dtype, np.floating)
        or not np.isfinite(a).all()
    ):
        _invalid()
    if not np.isfinite(c[cm]).all() or not np.isfinite(r[rm]).all():
        _invalid()

    # Do not multiply masks by payloads: hidden NaN/Inf values must never enter
    # a route call, including an injected test inference function.
    clean_c = np.zeros(c.shape, dtype=float)
    clean_r = np.zeros(r.shape, dtype=float)
    clean_c[cm] = c[cm]
    clean_r[rm] = r[rm]
    return clean_c, cm.copy(), clean_r, rm.copy(), a.astype(float, copy=True)


def _route_inputs(c, cm, r, rm, route: str):
    if route == "both":
        return c.copy(), cm.copy(), r.copy(), rm.copy()
    if route == "clinical":
        return c.copy(), cm.copy(), np.zeros_like(r), np.zeros_like(rm)
    if route == "retinal":
        return np.zeros_like(c), np.zeros_like(cm), r.copy(), rm.copy()
    raise AssertionError("internal route contract")


def _infer_route(infer: Callable, model, c, cm, r, rm, age, route: str, expected_abstain):
    try:
        value = infer(model, c, cm, r, rm, age.copy(), route=route)
    except Exception as exc:  # Public API reports only its fixed failure class.
        raise ValueError(_INFERENCE_ERROR) from exc
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(_INFERENCE_ERROR)
    state_age, abstain = value
    x = np.asarray(state_age)
    a = np.asarray(abstain)
    if (
        x.shape != (len(age), 193)
        or not np.issubdtype(x.dtype, np.floating)
        or not np.isfinite(x).all()
        or not np.array_equal(x[:, 192], age)
        or a.shape != (len(age),)
        or a.dtype != bool
        or not np.array_equal(a, expected_abstain)
        or np.any(x[expected_abstain, :192] != 0.0)
    ):
        raise ValueError(_INFERENCE_ERROR)
    return x.astype(float, copy=True), a.copy()


def make_designs(model, c, cm, r, rm, age, *, infer=None):
    """Return route designs and a 387-wide modality-preserving diagnostic union.

    Args are already-normalized clinical and retinal feature arrays compatible
    with :func:`bran_research_bundle_api_v1.encode`.  ``infer`` is an optional
    test seam with that function's call signature and return contract.

    The ``union`` columns are, in order: clinical-route state mean (192),
    retinal-route state mean (192), clinical availability (0.0/1.0), retinal
    availability (0.0/1.0), and the single supplied age column.  It appends no
    raw clinical or retinal values and is incompatible with 192-wide decoders.
    """
    clean_c, clean_cm, clean_r, clean_rm, clean_age = _as_inputs(c, cm, r, rm, age)
    if infer is None:
        infer = research_api.encode
    if not callable(infer):
        _invalid()

    clinical_available = clean_cm.any(axis=1)
    retinal_available = clean_rm.copy()
    route_results = {}
    for route, expected in (
        ("both", ~(clinical_available | retinal_available)),
        ("clinical", ~clinical_available),
        ("retinal", ~retinal_available),
    ):
        rc, rcm, rr, rrm = _route_inputs(clean_c, clean_cm, clean_r, clean_rm, route)
        route_results[route] = _infer_route(
            infer, model, rc, rcm, rr, rrm, clean_age, route, expected
        )

    clinical_state = route_results["clinical"][0][:, :192]
    retinal_state = route_results["retinal"][0][:, :192]
    # A route may validly have a conditional state when its modality is absent,
    # but the union is deliberately modality-preserving rather than imputing it.
    clinical_state = np.where(clinical_available[:, None], clinical_state, 0.0)
    retinal_state = np.where(retinal_available[:, None], retinal_state, 0.0)
    union = np.column_stack(
        (
            clinical_state,
            retinal_state,
            clinical_available.astype(float),
            retinal_available.astype(float),
            clean_age,
        )
    )
    abstain = ~(clinical_available | retinal_available)
    if union.shape != (len(clean_age), 387) or not np.isfinite(union).all():
        raise ValueError(_INFERENCE_ERROR)
    return {
        "joint": route_results["both"][0],
        "clinical": route_results["clinical"][0],
        "retinal": route_results["retinal"][0],
        "union": union,
    }, abstain.copy()


__all__ = ["make_designs"]
