"""Private array-only V5 context-matched prediction preservation."""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from bran_multisource_model_v2 import BRANMultisourceModelV2
from bran_multisource_training_v2 import (
    MaterializedBatch, _completion_masks, _masked_digest, _require_unpaired,
    _validate_batch, _zero,
)


_INVALID = "multisource prediction preservation inputs invalid"


def _invalid() -> None:
    raise ValueError(_INVALID)


def bridge_masks(paired_batch: MaterializedBatch, source_batch: MaterializedBatch,
                 step: int) -> tuple[Tensor, Tensor]:
    """Project donor *availability only* onto paired physiology."""
    try:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            _invalid()
        _validate_batch(paired_batch)
        _validate_batch(source_batch)
        if (paired_batch.c.device != source_batch.c.device or source_batch.c.shape[0] == 0
                or source_batch.labels is not None or source_batch.labelmask is not None):
            _invalid()
        donor = (torch.arange(paired_batch.c.shape[0], device=paired_batch.c.device)
                 + step * paired_batch.c.shape[0]) % source_batch.c.shape[0]
        clinical = paired_batch.cm & source_batch.cm[donor]
        retinal = paired_batch.rm & source_batch.rm[donor].any(dim=1, keepdim=True)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        _invalid()
    return clinical, retinal


def _inputs(batch: MaterializedBatch, clinical: Tensor, retinal: Tensor) -> tuple[Tensor, Tensor]:
    return (torch.where(clinical, batch.c, torch.zeros_like(batch.c)),
            torch.where(retinal[..., None], batch.r, torch.zeros_like(batch.r)))


def _screen_component(model: BRANMultisourceModelV2, teacher: BRANMultisourceModelV2,
                      batch: MaterializedBatch, age7: Tensor, clinical: Tensor,
                      retinal: Tensor) -> tuple[Tensor, int]:
    c, r = _inputs(batch, clinical, retinal)
    student = model.encode(c, clinical, r, retinal, age7)
    with torch.no_grad():
        reference = teacher.encode(c, clinical, r, retinal, age7)
    keep = (~student.abstain & ~reference.abstain & (age7[:, 3] == 1))
    if not bool(keep.any()):
        return _zero(model), 0
    delta = model.screening_joint_head(student.mean[keep]) - teacher.screening_joint_head(reference.mean[keep])
    return delta.square().mean(), int(keep.sum())


def _cbc_component(model: BRANMultisourceModelV2, teacher: BRANMultisourceModelV2,
                   batch: MaterializedBatch, age7: Tensor, clinical: Tensor,
                   retinal: Tensor) -> tuple[Tensor, int]:
    c, r = _inputs(batch, clinical, retinal)
    student = model.encode(c, clinical, r, retinal, age7)
    with torch.no_grad():
        reference = teacher.encode(c, clinical, r, retinal, age7)
    keep = ~student.abstain & ~reference.abstain & (age7[:, 3] == 1)
    cells = batch.cm[:, model.cbc_indices] & ~clinical[:, model.cbc_indices]
    cells = cells & keep[:, None]
    count = int(cells.sum())
    if count == 0:
        return _zero(model), 0
    difference = F.smooth_l1_loss(model.cbc_joint_head(student.mean),
                                  teacher.cbc_joint_head(reference.mean), reduction="none")
    return difference[cells].mean(), count


def preservation(model: BRANMultisourceModelV2, teacher: BRANMultisourceModelV2,
                 paired_batch: MaterializedBatch, source_batch: MaterializedBatch,
                 age7: Tensor, visible_c: Tensor, visible_r: Tensor, step: int) -> dict[str, object]:
    """Match initial-teacher heads in route and donor-availability contexts.

    ``visible_c``/``visible_r`` are the V4 screening-route masks. CBC contexts
    are independently intersected with the fixed completion masks.
    """
    try:
        if (not isinstance(model, BRANMultisourceModelV2) or not isinstance(teacher, BRANMultisourceModelV2)
                or model is teacher or model.cbc_indices != teacher.cbc_indices
                or teacher.training or any(parameter.requires_grad for parameter in teacher.parameters())
                or not isinstance(age7, Tensor) or age7.shape != (paired_batch.c.shape[0], 7)
                or age7.device != paired_batch.c.device or age7.dtype != paired_batch.c.dtype
                or not isinstance(visible_c, Tensor) or visible_c.dtype != torch.bool
                or visible_c.shape != paired_batch.c.shape or not isinstance(visible_r, Tensor)
                or visible_r.dtype != torch.bool or visible_r.shape != paired_batch.rm.shape):
            _invalid()
        _require_unpaired(source_batch, model)
        bridge_c, bridge_r = bridge_masks(paired_batch, source_batch, step)
        completion_c, completion_r = _completion_masks(paired_batch, model.cbc_indices, step)

        route_screen, route_screen_count = _screen_component(model, teacher, paired_batch, age7,
                                                              visible_c, visible_r)
        route_cbc, route_cbc_count = _cbc_component(model, teacher, paired_batch, age7,
                                                     completion_c, completion_r)
        bridge_screen_c, bridge_screen_r = visible_c & bridge_c, visible_r & bridge_r
        bridge_cbc_c, bridge_cbc_r = completion_c & bridge_c, completion_r & bridge_r
        donor_screen, donor_screen_count = _screen_component(model, teacher, paired_batch, age7,
                                                              bridge_screen_c, bridge_screen_r)
        donor_cbc, donor_cbc_count = _cbc_component(model, teacher, paired_batch, age7,
                                                     bridge_cbc_c, bridge_cbc_r)
        route = route_screen + 0.5 * route_cbc
        donor = donor_screen + 0.5 * donor_cbc
        loss = 0.5 * (route + donor)
        if not torch.isfinite(loss):
            _invalid()
    except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
        _invalid()
    return {
        "loss": loss,
        "supported": bool(route_screen_count or route_cbc_count or donor_screen_count or donor_cbc_count),
        "bridge_mask_digest": _masked_digest(bridge_c, bridge_r),
        "route_screen_rows": route_screen_count, "route_cbc_cells": route_cbc_count,
        "bridge_screen_rows": donor_screen_count, "bridge_cbc_cells": donor_cbc_count,
    }
