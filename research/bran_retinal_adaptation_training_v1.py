"""Pure in-memory matched retinal-adaptation loop for prospective experiments.

Receipts are private audit snapshots, not checkpoints: this V1 module exposes
no resume operation because exact continuation also requires a caller-owned
remaining batch stream contract.  It performs no file I/O, sampling, printing,
or experiment authorization.
"""

from __future__ import annotations

import copy
import math
import random

import numpy as np
import torch

from bran_retinal_supervised_adaptation_kernel_v1 import RetinalSupervisedAdaptationKernel


_MIN_RATIO = 0.1
_WARMUP_FRACTION = 0.1


def _invalid() -> ValueError:
    return ValueError("invalid retinal adaptation training")


def _number(value: object, *, lower: float, upper: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < lower:
        raise _invalid()
    if upper is not None and float(value) > upper:
        raise _invalid()
    return float(value)


def _validate_config(
    label_weight: object, positive_weight: object, steps: object, lr: object,
    weight_decay: object, ema: object, seed: object, checkpoint_callback: object,
) -> tuple[float, int, float, float, float, int]:
    if type(label_weight) not in (int, float) or float(label_weight) not in (0.0, 1.0):
        raise _invalid()
    if not isinstance(positive_weight, torch.Tensor):
        raise _invalid()
    if type(steps) is not int or steps <= 0:
        raise _invalid()
    checked_lr = _number(lr, lower=0.0)
    if checked_lr == 0.0:
        raise _invalid()
    checked_weight_decay = _number(weight_decay, lower=0.0)
    checked_ema = _number(ema, lower=0.0, upper=1.0)
    if type(seed) is not int or seed < 0 or seed >= 2**32:
        raise _invalid()
    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise _invalid()
    return float(label_weight), steps, checked_lr, checked_weight_decay, checked_ema, seed


def learning_rate_at_step(step: int, *, steps: int, lr: float) -> float:
    """Fixed step-indexed 10%-warmup cosine schedule, ending at 10% of base LR."""
    if type(step) is not int or type(steps) is not int or type(lr) not in (int, float):
        raise _invalid()
    if steps <= 0 or step < 0 or step >= steps or not math.isfinite(float(lr)) or float(lr) <= 0.0:
        raise _invalid()
    warmup_steps = max(1, int(math.ceil(steps * _WARMUP_FRACTION)))
    if step < warmup_steps:
        return float(lr) * (step + 1) / warmup_steps
    tail = steps - warmup_steps
    if tail <= 1:
        return float(lr)
    progress = (step - warmup_steps) / (tail - 1)
    scale = _MIN_RATIO + (1.0 - _MIN_RATIO) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(lr) * scale


def _cpu_clone(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def _mps_rng_state() -> torch.Tensor | None:
    available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    getter = getattr(getattr(torch, "mps", None), "get_rng_state", None)
    if not available or not callable(getter):
        return None
    try:
        return getter().detach().cpu().clone()
    except Exception:
        return None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    setter = getattr(getattr(torch, "mps", None), "manual_seed", None)
    if bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()) and callable(setter):
        setter(seed)


def _values_finite(values: list[torch.Tensor]) -> bool:
    """One device-side reduction and one host read, including heterogeneous shapes."""
    if not values:
        raise _invalid()
    reduced = torch.stack([torch.isfinite(value).all() for value in values]).all()
    return bool(reduced.item())


def _receipt(
    kernel: RetinalSupervisedAdaptationKernel,
    optimizer: torch.optim.Optimizer,
    *,
    completed_step: int,
    steps: int,
    lr: float,
    label_weight: float,
    weight_decay: float,
    ema: float,
    seed: int,
) -> dict[str, object]:
    return {
        "schema": "bran-retinal-adaptation-training-receipt-v1",
        "resume_supported": False,
        "training_config": {
            "label_weight": label_weight,
            "lr": lr,
            "weight_decay": weight_decay,
            "ema": ema,
            "seed": seed,
            "steps": steps,
            "optimizer": {"name": "AdamW", "betas": (0.9, 0.95)},
        },
        "kernel_state": _cpu_clone(kernel.state_dict()),
        "optimizer_state": _cpu_clone(optimizer.state_dict()),
        "scheduler": {
            "name": "linear_warmup_cosine",
            "warmup_fraction": _WARMUP_FRACTION,
            "warmup_steps": max(1, int(math.ceil(steps * _WARMUP_FRACTION))),
            "min_ratio": _MIN_RATIO,
            "step": completed_step,
            "steps": steps,
            "last_lr": learning_rate_at_step(completed_step - 1, steps=steps, lr=lr),
        },
        "rng": {
            "python": _cpu_clone(random.getstate()),
            "numpy": _cpu_clone(np.random.get_state()),
            "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
            "torch_mps": _mps_rng_state(),
        },
        "completed_step": completed_step,
    }


def _batch(value: object, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(value, (tuple, list)) or len(value) != 4 or not all(isinstance(item, torch.Tensor) for item in value):
        raise _invalid()
    images, patch_mask, labels, observed = value
    return (
        images.to(device=device),
        patch_mask.to(device=device),
        labels.to(device=device),
        observed.to(device=device),
    )


def train_arm(
    base_encoder: torch.nn.Module,
    batches: object,
    *,
    label_weight: float | int,
    positive_weight: torch.Tensor,
    steps: int,
    lr: float = 1e-4,
    weight_decay: float = 0.04,
    ema: float = 0.996,
    seed: int = 73191,
    device: str | torch.device = "cpu",
    checkpoint_callback=None,
) -> tuple[RetinalSupervisedAdaptationKernel, dict[str, object]]:
    """Train one caller-batched arm and return eval kernel plus private receipt.

    There is intentionally no resume argument or loader.  Resuming a receipt
    without a cryptographically identical remaining caller batch stream would
    be misleading, so V1 records state for audit only.
    """
    weight, planned_steps, base_lr, decay, coefficient, checked_seed = _validate_config(
        label_weight, positive_weight, steps, lr, weight_decay, ema, seed, checkpoint_callback
    )
    try:
        destination = torch.device(device)
        if destination.type == "mps" and not torch.backends.mps.is_available():
            raise _invalid()
        iterator = iter(batches)
        _seed_everything(checked_seed)
        kernel = RetinalSupervisedAdaptationKernel(base_encoder).to(destination)
        kernel.train()
        trainable = [parameter for parameter in kernel.parameters() if parameter.requires_grad]
        if not trainable:
            raise _invalid()
        # Frozen parameters cannot be allowed to enter an audit receipt with a
        # nonfinite value.  This one full-model gate is deliberately outside
        # the step loop to avoid repeated MPS synchronization.
        if not _values_finite(list(kernel.parameters())):
            raise _invalid()
        teacher_named = dict(kernel.teacher.named_parameters())
        teacher_after_ema = [
            teacher_named[name]
            for name, parameter in kernel.student.named_parameters()
            if parameter.requires_grad and name in teacher_named
        ]
        optimizer = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=decay, betas=(0.9, 0.95))
        weights = positive_weight.to(device=destination)
        for step in range(planned_steps):
            current_lr = learning_rate_at_step(step, steps=planned_steps, lr=base_lr)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            try:
                batch = next(iterator)
            except StopIteration:
                raise _invalid() from None
            images, patch_mask, labels, observed = _batch(batch, destination)
            optimizer.zero_grad(set_to_none=True)
            loss = kernel.objective(
                images, patch_mask, labels, observed,
                label_weight=weight, positive_weight=weights,
            )
            if not torch.isfinite(loss).item():
                raise _invalid()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            kernel.update_teacher(coefficient)
            if not _values_finite([*trainable, *teacher_after_ema]):
                raise _invalid()
            completed = step + 1
            if checkpoint_callback is not None and (completed % 32 == 0 or completed == planned_steps):
                checkpoint_callback(_receipt(
                    kernel, optimizer, completed_step=completed, steps=planned_steps, lr=base_lr,
                    label_weight=weight, weight_decay=decay, ema=coefficient, seed=checked_seed,
                ))
        kernel.eval()
        kernel.teacher.eval()
        return kernel, _receipt(
            kernel, optimizer, completed_step=planned_steps, steps=planned_steps, lr=base_lr,
            label_weight=weight, weight_decay=decay, ema=coefficient, seed=checked_seed,
        )
    except Exception:
        raise _invalid() from None
