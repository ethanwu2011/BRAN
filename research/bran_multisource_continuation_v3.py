"""Array-only synthetic V3 continuation update; no loaders, I/O, or logging.

The caller owns normalized arrays, the private teacher scale, and all optimizer
lifecycle decisions.  This is an in-memory training kernel, not a run protocol.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from bran_multisource_model_v2 import BRANMultisourceModelV2
from bran_multisource_training_v2 import (
    MaterializedBatch, _age7, _cbc_completion_loss, _clinical_visible,
    _generator, _generative, _masked_digest, _require_unpaired, _route_masks,
    _screening_loss, _validate_batch, _zero,
)


_INVALID = "multisource continuation inputs invalid"
_LR = 5e-5
_WEIGHT_DECAY = 1e-4
_CLIP = 5.0
_MAX_STEP = 2999


def _invalid() -> None:
    raise ValueError(_INVALID)


def _seed(seed: int, step: int, salt: int) -> int:
    return (seed + 1000003 * step + salt) % (2**63 - 1)


def _indices(values: Iterable[int]) -> tuple[int, ...]:
    try:
        result = tuple(values)
        unique = set(result)
    except (TypeError, ValueError):
        _invalid()
    if (len(result) != 9 or len(unique) != 9
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 or value >= 48 for value in result)):
        _invalid()
    return result


def _valid_optimizer(model: BRANMultisourceModelV2, optimizer: object) -> bool:
    if not isinstance(optimizer, torch.optim.AdamW):
        return False
    model_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group.get("params", [])}
    return (bool(model_ids) and model_ids == optimizer_ids and all(
        float(group.get("lr", float("nan"))) == _LR
        and float(group.get("weight_decay", float("nan"))) == _WEIGHT_DECAY
        for group in optimizer.param_groups
    ))


def _same_unshared_mlp(model: object, teacher: object) -> bool:
    if (not isinstance(model, BRANMultisourceModelV2)
            or not isinstance(teacher, BRANMultisourceModelV2)
            or model is teacher or model.arm != "mlp" or teacher.arm != "mlp"
            or model.config != teacher.config or model.eligible_indices != teacher.eligible_indices
            or model.cbc_indices != teacher.cbc_indices or not model.training or teacher.training
            or any(not parameter.requires_grad for parameter in model.parameters())
            or any(parameter.requires_grad for parameter in teacher.parameters())):
        return False
    model_modules, teacher_modules = dict(model.named_modules()), dict(teacher.named_modules())
    if model_modules.keys() != teacher_modules.keys() or any(
            model_modules[name] is teacher_modules[name] for name in model_modules):
        return False
    model_state, teacher_state = model.state_dict(), teacher.state_dict()
    if model_state.keys() != teacher_state.keys():
        return False
    try:
        return all(left.device.type == "cpu" and right.device.type == "cpu"
                   and left.dtype == right.dtype and left.shape == right.shape
                   and left.data_ptr() != right.data_ptr()
                   for left, right in zip(model_state.values(), teacher_state.values()))
    except (AttributeError, RuntimeError):
        return False


def _visible_inputs(batch: MaterializedBatch, visible_c: Tensor, visible_r: Tensor) -> tuple[Tensor, Tensor]:
    """Erase hidden payloads before preservation encodes without NaN arithmetic."""
    return (torch.where(visible_c, batch.c, torch.zeros_like(batch.c)),
            torch.where(visible_r[..., None], batch.r, torch.zeros_like(batch.r)))


def _state_preservation(model: BRANMultisourceModelV2, teacher: BRANMultisourceModelV2,
                        batch: MaterializedBatch, age7: Tensor, visible_c: Tensor,
                        visible_r: Tensor, state_scale: Tensor) -> tuple[Tensor, int]:
    clinical, retinal = _visible_inputs(batch, visible_c, visible_r)
    student = model.encode(clinical, visible_c, retinal, visible_r, age7)
    # The reported flag is the first fixed kind indicator in age7.
    keep = ~student.abstain & (age7[:, 3] == 1)
    if not bool(keep.any()):
        return _zero(model), 0
    with torch.no_grad():
        reference = teacher.encode(clinical, visible_c, retinal, visible_r, age7)
    value = ((student.mean[keep] - reference.mean[keep]) / state_scale).square().mean()
    return value, int(keep.sum())


def _validate(
    model: object, teacher: object, optimizer: object, paired_batch: object,
    source_batch: object, step: object, age_mean: object, age_scale: object,
    seed: object, cbc_indices: Iterable[int], positive_weight: object,
    state_scale: object, source_enabled: object,
) -> tuple[tuple[int, ...], Tensor]:
    if (isinstance(step, bool) or not isinstance(step, int) or step < 0 or step > _MAX_STEP
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or type(source_enabled) is not bool):
        _invalid()
    try:
        mean, scale = float(age_mean), float(age_scale)
    except (TypeError, ValueError, OverflowError):
        _invalid()
    if not torch.isfinite(torch.tensor(mean)) or not torch.isfinite(torch.tensor(scale)) or scale <= 0:
        _invalid()
    cbc = _indices(cbc_indices)
    if not _same_unshared_mlp(model, teacher) or cbc != model.cbc_indices or not _valid_optimizer(model, optimizer):
        _invalid()
    _validate_batch(paired_batch)
    _validate_batch(source_batch)
    dtype = next(model.parameters()).dtype
    if (paired_batch.c.device.type != "cpu" or source_batch.c.device.type != "cpu"
            or paired_batch.labels is None or paired_batch.labelmask is None
            or source_batch.labels is not None or source_batch.labelmask is not None
            or any(value.dtype != dtype for value in (
                paired_batch.c, paired_batch.r, paired_batch.age.value,
                paired_batch.age.lower, paired_batch.age.upper, source_batch.c,
                source_batch.r, source_batch.age.value, source_batch.age.lower,
                source_batch.age.upper))):
        _invalid()
    _require_unpaired(source_batch, model)
    if (not isinstance(positive_weight, Tensor) or not positive_weight.is_floating_point()
            or positive_weight.shape != (26,) or positive_weight.device != paired_batch.c.device
            or not torch.isfinite(positive_weight).all() or bool((positive_weight <= 0).any())):
        _invalid()
    if (not isinstance(state_scale, Tensor) or not state_scale.is_floating_point()
            or state_scale.shape != (192,) or state_scale.device != paired_batch.c.device
            or state_scale.dtype != next(model.parameters()).dtype or state_scale.requires_grad
            or not torch.isfinite(state_scale).all() or bool((state_scale < 1).any())):
        _invalid()
    return cbc, state_scale


def train_step_v3(
    model: BRANMultisourceModelV2,
    teacher: BRANMultisourceModelV2,
    optimizer: torch.optim.Optimizer,
    paired_batch: MaterializedBatch,
    source_batch: MaterializedBatch,
    step: int,
    age_mean: float,
    age_scale: float,
    seed: int,
    cbc_indices: Iterable[int],
    positive_weight: Tensor,
    state_scale: Tensor,
    *,
    source_enabled: bool,
) -> dict[str, object]:
    """Perform one fixed-recipe C/M continuation update on caller-owned arrays.

    The source batch is always validated as unlabeled and unpaired.  It is not
    evaluated or differentiated when ``source_enabled`` is false, preserving
    C/M paired stochastic streams and preventing source labels from entering
    either arm.
    """
    try:
        cbc, state_scale = _validate(model, teacher, optimizer, paired_batch, source_batch,
                                     step, age_mean, age_scale, seed, cbc_indices,
                                     positive_weight, state_scale, source_enabled)
        with torch.random.fork_rng(devices=[]):
            paired_age = _age7(paired_batch, age_mean, age_scale,
                                _generator(paired_batch.c.device, seed, step, 11))
            paired_mask_generator = _generator(paired_batch.c.device, seed, step, 23)
            visible_c, visible_r = _route_masks(paired_batch, step, paired_mask_generator)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(_seed(seed, step, 101))
                generative, generative_rows = _generative(model, paired_batch, paired_age,
                                                           visible_c, visible_r, step)
            screening, screening_rows = _screening_loss(model, paired_batch, paired_age,
                                                         visible_c, visible_r, positive_weight)
            completion, completion_rows, completion_c, completion_r, _ = _cbc_completion_loss(
                model, paired_batch, paired_age, cbc, step)
            preservation, preservation_rows = _state_preservation(
                model, teacher, paired_batch, paired_age, visible_c, visible_r, state_scale)
            paired_total = generative + screening + 0.5 * completion + 0.1 * preservation

            source_generative = _zero(model)
            source_completion = _zero(model)
            source_generative_rows = source_completion_rows = 0
            source_c = source_r = None
            source_weight = 0.0
            if source_enabled:
                source_age = _age7(source_batch, age_mean, age_scale,
                                   _generator(source_batch.c.device, seed, step, 31))
                source_mask_generator = _generator(source_batch.c.device, seed, step, 37)
                source_c = _clinical_visible(source_batch.cm, source_mask_generator, step)
                source_r = source_batch.rm.clone()
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(_seed(seed, step, 103))
                    source_generative, source_generative_rows = _generative(
                        model, source_batch, source_age, source_c, source_r, step)
                source_completion, source_completion_rows, _, _, _ = _cbc_completion_loss(
                    model, source_batch, source_age, cbc, step)
                source_weight = 0.1 * min(1.0, (step + 1) / 300.0)
            source_total = source_generative + 0.5 * source_completion
            total = paired_total + source_weight * source_total
            supported = bool(generative_rows or screening_rows or completion_rows
                             or preservation_rows or source_generative_rows
                             or source_completion_rows)
            if not bool(torch.isfinite(total)):
                _invalid()
            if supported:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), _CLIP, error_if_nonfinite=True)
                optimizer.step()
    except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
        _invalid()

    hashes = {"paired": _masked_digest(visible_c, visible_r),
              "paired_completion": _masked_digest(completion_c, completion_r)}
    if source_c is not None:
        hashes["source"] = _masked_digest(source_c, source_r)
    return {
        "step": step, "loss": float(total.detach()), "paired_loss": float(paired_total.detach()),
        "generative_loss": float(generative.detach()), "screening_loss": float(screening.detach()),
        "cbc_loss": float(completion.detach()), "state_preservation_loss": float(preservation.detach()),
        "source_loss": float(source_total.detach()),
        "source_generative_loss": float(source_generative.detach()),
        "source_cbc_loss": float(source_completion.detach()), "source_weight": source_weight,
        "generative_supervised": bool(generative_rows), "screening_supervised": bool(screening_rows),
        "cbc_supervised": bool(completion_rows),
        "state_preservation_supervised": bool(preservation_rows),
        "source_generative_supervised": bool(source_generative_rows),
        "source_cbc_supervised": bool(source_completion_rows),
        "source_enabled": source_enabled, "optimizer_updated": supported, "mask_hashes": hashes,
    }
