"""Private frozen V5 route states for same-frame fixed-readout comparisons."""
from dataclasses import dataclass
from types import MappingProxyType

import torch

from bran_multisource_age_v2 import normalize_age
from bran_multisource_inference_v2 import _validate_structure, _validate_visible_values, _route_masks, _clean_for_encode
from bran_v5_residual_training import _teacher_ok

ROUTES = ('both', 'clinical', 'retinal')


@dataclass(frozen=True, repr=False)
class FrozenStateRoutes:
    states: object
    available: object


def state_routes(model, clinical, cm, retinal, rm, age, age_mean, age_scale, batch_size=256):
    """Return detached local states; never persist or emit them from this helper.

    Inactive rows have finite zero states and a false availability mask. Callers
    must use matched physiological support for fitting/scoring, not those zeros.
    Every route is computed in the same supplied checkpoint coordinate frame.
    """
    try:
        _teacher_ok(model)
        _validate_structure(model, clinical, cm, retinal, rm, age, batch_size)
        age7 = normalize_age(age, age_mean, age_scale)
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        states, available = {}, {}
        with torch.no_grad():
            for route in ROUTES:
                route_cm, route_rm = _route_masks(cm, rm, route)
                _validate_visible_values(model, clinical, route_cm, retinal, route_rm)
                chunks, masks = [], []
                for first in range(0, len(clinical), batch_size):
                    last = first+batch_size
                    c, r = _clean_for_encode(model, clinical[first:last], route_cm[first:last],
                                              retinal[first:last], route_rm[first:last])
                    posterior = model.encode(c, route_cm[first:last], r, route_rm[first:last, None], age7[first:last])
                    if (posterior.mean.shape != (len(c), 192) or posterior.abstain.shape != (len(c),)
                            or not bool(torch.isfinite(posterior.mean).all())):
                        raise ValueError
                    chunks.append(posterior.mean.detach().clone())
                    masks.append((~posterior.abstain).detach().clone())
                states[route] = torch.cat(chunks)
                available[route] = torch.cat(masks)
                if not bool((states[route][~available[route]] == 0).all()): raise ValueError
        _teacher_ok(model)
        if not all(torch.equal(value, model.state_dict()[key]) for key,value in before.items()): raise ValueError
        return FrozenStateRoutes(MappingProxyType(states), MappingProxyType(available))
    except Exception:
        raise ValueError('v5_state_routes_contract_failed') from None
