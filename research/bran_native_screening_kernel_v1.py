"""No-fit inference for a retained 192-to-26 BRAN screening head."""
from __future__ import annotations

import numpy as np

_INVALID = "native screening kernel inputs invalid"


def _invalid():
    raise ValueError(_INVALID)


def _validate(model, c, cm, r, rm, age, batch_size):
    import torch
    if not isinstance(model, torch.nn.Module): _invalid()
    head = getattr(model, "screening_joint_head", None)
    config = getattr(model, "config", None)
    if (not isinstance(head, torch.nn.Linear) or head.bias is None or tuple(head.weight.shape) != (26, 192)
            or tuple(head.bias.shape) != (26,) or getattr(config, "state_dim", None) != 192
            or not callable(getattr(model, "encode", None)) or not torch.isfinite(head.weight).all()
            or not torch.isfinite(head.bias).all()): _invalid()
    c, cm, r, rm, age = (np.asarray(value) for value in (c, cm, r, rm, age))
    n = len(c)
    if (n <= 0 or c.shape != cm.shape or c.shape != (n, 59) or r.shape != (n, 384)
            or cm.dtype != np.dtype(bool) or rm.shape != age.shape or rm.shape != (n,)
            or rm.dtype != np.dtype(bool) or not np.isfinite(c[cm]).all()
            or not np.isfinite(r[rm]).all() or not np.isfinite(age).all()
            or isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0): _invalid()
    return c, cm, r, rm, age, int(batch_size)


def predict_native(model, c, cm, r, rm, age, batch_size=256):
    """Return caller-local probabilities for full, clinical-only, and retinal-only routes.

    Masked values are physically zeroed before each encode.  A row is NaN only
    when that route's posterior explicitly abstains; this function never fits.
    """
    import torch
    c, cm, r, rm, age, batch_size = _validate(model, c, cm, r, rm, age, batch_size)
    output = {route: np.empty((len(c), 26), dtype=float) for route in ("both", "clinical", "retinal")}
    with torch.no_grad():
        for start in range(0, len(c), batch_size):
            stop = min(len(c), start + batch_size)
            ct = torch.tensor(c[start:stop], dtype=torch.float32)
            cmt = torch.tensor(cm[start:stop], dtype=torch.bool)
            rt = torch.tensor(r[start:stop, None, :], dtype=torch.float32)
            rmt = torch.tensor(rm[start:stop, None], dtype=torch.bool)
            at = torch.tensor(age[start:stop], dtype=torch.float32)
            clean_c = torch.where(cmt, ct, torch.zeros_like(ct))
            clean_r = torch.where(rmt[..., None], rt, torch.zeros_like(rt))
            for route, use_c, use_r in (("both", True, True), ("clinical", True, False), ("retinal", False, True)):
                route_c = clean_c if use_c else torch.zeros_like(clean_c)
                route_cm = cmt if use_c else torch.zeros_like(cmt)
                route_r = clean_r if use_r else torch.zeros_like(clean_r)
                route_rm = rmt if use_r else torch.zeros_like(rmt)
                state = model.encode(route_c, route_cm, route_r, route_rm, at)
                probabilities = torch.sigmoid(model.screening_joint_head(state.mean)).cpu().numpy()
                abstain = state.abstain.cpu().numpy()
                probabilities[abstain] = np.nan
                if (np.any(np.isinf(probabilities)) or np.any(np.isfinite(probabilities) & ((probabilities < 0) | (probabilities > 1)))
                        or np.any(np.isnan(probabilities) & ~abstain[:, None])
                        or not np.array_equal(np.isnan(probabilities).all(axis=1), abstain)):
                    _invalid()
                output[route][start:stop] = probabilities
    return output
