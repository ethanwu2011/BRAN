"""Array-only V4 continuation kernel with a bounded source-generative gradient.

This correction changes only M-arm source-generative gradient contribution.  It
does not load data, choose sources, write artifacts, or make release decisions.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor

from bran_multisource_continuation_v3 import (
    _CLIP, _LR, _MAX_STEP, _WEIGHT_DECAY, _indices, _invalid, _seed,
    _same_unshared_mlp, _state_preservation, _valid_optimizer,
)
from bran_multisource_model_v2 import BRANMultisourceModelV2
from bran_multisource_training_v2 import (
    MaterializedBatch, _age7, _cbc_completion_loss, _clinical_visible,
    _generator, _generative, _masked_digest, _require_unpaired, _route_masks,
    _screening_loss, _validate_batch, _zero,
)


_INVALID = "multisource continuation v4 inputs invalid"


def _fail() -> None:
    raise ValueError(_INVALID)


def _validate(
    model: object, teacher: object, optimizer: object, paired_batch: object,
    source_batch: object, step: object, age_mean: object, age_scale: object,
    seed: object, cbc_indices: Iterable[int], positive_weight: object,
    state_scale: object, source_enabled: object,
) -> tuple[tuple[int, ...], Tensor]:
    if (isinstance(step, bool) or not isinstance(step, int) or step < 0 or step > _MAX_STEP
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or type(source_enabled) is not bool):
        _fail()
    try:
        mean, scale = float(age_mean), float(age_scale)
    except (TypeError, ValueError, OverflowError):
        _fail()
    if not torch.isfinite(torch.tensor(mean)) or not torch.isfinite(torch.tensor(scale)) or scale <= 0:
        _fail()
    cbc = _indices(cbc_indices)
    if not _same_unshared_mlp(model, teacher) or cbc != model.cbc_indices or not _valid_optimizer(model, optimizer):
        _fail()
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
        _fail()
    _require_unpaired(source_batch, model)
    if (not isinstance(positive_weight, Tensor) or not positive_weight.is_floating_point()
            or positive_weight.shape != (26,) or positive_weight.device != paired_batch.c.device
            or not torch.isfinite(positive_weight).all() or bool((positive_weight <= 0).any())):
        _fail()
    if (not isinstance(state_scale, Tensor) or not state_scale.is_floating_point()
            or state_scale.shape != (192,) or state_scale.device != paired_batch.c.device
            or state_scale.dtype != next(model.parameters()).dtype or state_scale.requires_grad
            or not torch.isfinite(state_scale).all() or bool((state_scale < 1).any())):
        _fail()
    return cbc, state_scale


def _norm64(gradients: Sequence[Tensor | None]) -> float:
    total = 0.0
    for gradient in gradients:
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            _fail()
        total += float(gradient.detach().to(dtype=torch.float64).square().sum())
    value = total ** 0.5
    if not torch.isfinite(torch.tensor(value, dtype=torch.float64)):
        _fail()
    return value


def _cap_alpha(paired_native: Sequence[Tensor | None],
               weighted_source_generative: Sequence[Tensor | None]) -> tuple[float, bool, bool, bool]:
    """Return bounded alpha and private booleans; never expose gradient norms."""
    native_norm = _norm64(paired_native)
    source_norm = _norm64(weighted_source_generative)
    source_nonzero = source_norm > 0.0
    alpha = min(1.0, native_norm / source_norm) if source_nonzero else 1.0
    if not torch.isfinite(torch.tensor(alpha, dtype=torch.float64)) or alpha < 0.0 or alpha > 1.0:
        _fail()
    bounded = alpha * source_norm
    # The relative tolerance preserves the specified one-times-native contract.
    contract = (not source_nonzero) or bounded <= native_norm * (1.0 + 1e-6) + 1e-12
    if not contract:
        _fail()
    return alpha, alpha < 1.0, contract, source_nonzero


def _gradients(loss: Tensor, parameters: tuple[torch.nn.Parameter, ...], *, retain_graph: bool) -> tuple[Tensor | None, ...]:
    if not isinstance(loss, Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
        _fail()
    try:
        return torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    except (RuntimeError, TypeError):
        _fail()


def _set_merged_grads(parameters: tuple[torch.nn.Parameter, ...], paired: Sequence[Tensor | None],
                     source_gen: Sequence[Tensor | None], source_cbc: Sequence[Tensor | None], alpha: float) -> None:
    if not (len(parameters) == len(paired) == len(source_gen) == len(source_cbc)):
        _fail()
    for parameter, paired_grad, source_gen_grad, source_cbc_grad in zip(parameters, paired, source_gen, source_cbc):
        # Preserve AdamW's distinction between unused (None) and a computed zero
        # gradient. Otherwise the correction would also decay unused heads.
        if paired_grad is None and source_gen_grad is None and source_cbc_grad is None:
            parameter.grad = None
            continue
        value = torch.zeros_like(parameter)
        if paired_grad is not None:
            value.add_(paired_grad)
        if source_gen_grad is not None:
            value.add_(source_gen_grad, alpha=alpha)
        if source_cbc_grad is not None:
            value.add_(source_cbc_grad)
        if not torch.isfinite(value).all():
            _fail()
        parameter.grad = value


def train_step_v4(
    model: BRANMultisourceModelV2, teacher: BRANMultisourceModelV2,
    optimizer: torch.optim.Optimizer, paired_batch: MaterializedBatch,
    source_batch: MaterializedBatch, step: int, age_mean: float, age_scale: float,
    seed: int, cbc_indices: Iterable[int], positive_weight: Tensor, state_scale: Tensor,
    *, source_enabled: bool,
) -> dict[str, object]:
    """Run one fixed V4 update; source-off follows the V3 backward path exactly."""
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
            paired_native = screening + 0.5 * completion
            # Keep V3's original operation order so source-off is bit-exact.
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
                _fail()
            cap_applied = False
            cap_contract = True
            source_gen_nonzero = False
            source_cbc_nonzero = False
            if supported:
                optimizer.zero_grad(set_to_none=True)
                if not source_enabled:
                    # This is deliberately the original V3 backward/clip/step path.
                    total.backward()
                else:
                    parameters = tuple(model.parameters())
                    grad_pair = _gradients(paired_total, parameters, retain_graph=True)
                    grad_native = _gradients(paired_native, parameters, retain_graph=True)
                    grad_source_gen = _gradients(source_weight * source_generative, parameters, retain_graph=True)
                    grad_source_cbc = _gradients(source_weight * 0.5 * source_completion,
                                                 parameters, retain_graph=False)
                    alpha, cap_applied, cap_contract, source_gen_nonzero = _cap_alpha(
                        grad_native, grad_source_gen)
                    source_cbc_nonzero = _norm64(grad_source_cbc) > 0.0
                    _set_merged_grads(parameters, grad_pair, grad_source_gen, grad_source_cbc, alpha)
                torch.nn.utils.clip_grad_norm_(model.parameters(), _CLIP, error_if_nonfinite=True)
                optimizer.step()
    except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
        _fail()

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
        "source_generative_cap_applied": cap_applied,
        "cap_contract_satisfied": cap_contract,
        "source_generative_gradient_nonzero": source_gen_nonzero,
        "source_cbc_gradient_nonzero": source_cbc_nonzero,
    }
