"""Array-only true paired-target supervision under source availability patterns.

No I/O. Donors contribute masks only. Caller owns provenance and privacy.
"""
from dataclasses import replace
import torch

from bran_multisource_preservation_v5 import bridge_masks
from bran_multisource_training_v2 import (
    _completion_masks, _masked_digest, _screening_loss, _validate_batch, _zero,
)
from bran_screening_joint_kernel_v1 import _cbc_loss

ERROR = 'source_pattern_supervision_v6_invalid'


def inputs(batch, clinical, retinal):
    """Erase payloads as well as flags, without exposing supervision masks."""
    return (torch.where(clinical, batch.c, torch.zeros_like(batch.c)),
            torch.where(retinal[..., None], batch.r, torch.zeros_like(batch.r)))


def supervise(model, paired, source, age7, visible_c, visible_r, step, positive_weight):
    """Two genuine-label losses; returns private scalar tensors and counters.

    CBC targets are those designated by the original six-pattern schedule. Extra
    source-pattern hiding does not silently expand the task's target definition.
    """
    try:
        _validate_batch(paired)
        if paired.labels is None or paired.labelmask is None:
            raise ValueError(ERROR)
        if (visible_c.dtype != torch.bool or visible_c.shape != paired.cm.shape
                or visible_r.dtype != torch.bool or visible_r.shape != paired.rm.shape
                or bool((visible_c & ~paired.cm).any()) or bool((visible_r & ~paired.rm).any())):
            raise ValueError(ERROR)
        bc, br = bridge_masks(paired, source, step)
        sc, sr = visible_c & bc, visible_r & br
        c, r = inputs(paired, sc, sr)
        screened = replace(paired, c=c, r=r)
        screen_loss, screen_rows = _screening_loss(model, screened, age7, sc, sr, positive_weight)

        completion_c, completion_r = _completion_masks(paired, model.cbc_indices, step)
        cc, cr = completion_c & bc, completion_r & br
        c, r = inputs(paired, cc, cr)
        state = model.encode(c, cc, r, cr, age7)
        slots = model.cbc_indices
        targets = paired.cm[:, slots] & ~completion_c[:, slots]
        targets = targets & ~state.abstain[:, None]
        cbc_rows = int(targets.any(dim=1).sum())
        cbc_loss = _zero(model)
        if cbc_rows:
            cbc_loss = _cbc_loss(torch, model.cbc_joint_head(state.mean),
                paired.c[:, slots], targets,
                (state.clinical_available | state.retinal_available) & ~state.abstain)
        if not bool(torch.isfinite(screen_loss) & torch.isfinite(cbc_loss)):
            raise ValueError(ERROR)
        return {'screening': screen_loss, 'cbc': cbc_loss,
                'screening_rows': screen_rows, 'cbc_rows': cbc_rows,
                'screen_mask_digest': _masked_digest(sc, sr),
                'cbc_mask_digest': _masked_digest(cc, cr)}
    except Exception:
        raise ValueError(ERROR) from None
