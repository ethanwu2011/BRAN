"""Typed, local-only age inputs for the synthetic BRAN V2 scaffold.

This module performs no source admission, I/O, or logging.  The contract keeps
years in their original units until a caller explicitly normalizes them.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


REPORTED = 0
INTERVAL = 1
RIGHT_CENSORED = 2
UNKNOWN = 3
AGE_FEATURE_DIM = 7
_INVALID = "age contract invalid"


@dataclass(frozen=True)
class AgeBatch:
    """A batch of original-year age observations; kind codes are fixed above."""

    value: Tensor
    lower: Tensor
    upper: Tensor
    kind: Tensor


def _invalid() -> None:
    raise ValueError(_INVALID)


def _shape(batch: AgeBatch) -> tuple[int, ...]:
    if not isinstance(batch, AgeBatch) or any(not isinstance(x, Tensor) for x in
                                               (batch.value, batch.lower, batch.upper, batch.kind)):
        _invalid()
    if batch.value.ndim != 1 or batch.lower.shape != batch.value.shape or batch.upper.shape != batch.value.shape:
        _invalid()
    if batch.kind.shape != batch.value.shape or batch.kind.device != batch.value.device:
        _invalid()
    if batch.lower.device != batch.value.device or batch.upper.device != batch.value.device:
        _invalid()
    if not (batch.value.is_floating_point() and batch.lower.is_floating_point() and batch.upper.is_floating_point()):
        _invalid()
    if batch.value.dtype != batch.lower.dtype or batch.value.dtype != batch.upper.dtype:
        _invalid()
    return tuple(batch.value.shape)


def validate_age(batch: AgeBatch) -> AgeBatch:
    """Validate only fields active for each declared representation."""

    _shape(batch)
    if batch.kind.dtype == torch.bool or batch.kind.dtype.is_floating_point or batch.kind.dtype.is_complex:
        _invalid()
    kind = batch.kind.to(torch.long)
    if ((kind < REPORTED) | (kind > UNKNOWN)).any():
        _invalid()

    def valid(field: Tensor, active: Tensor) -> bool:
        return bool((torch.isfinite(field[active]) & (field[active] >= 0)).all())

    reported = kind == REPORTED
    interval = kind == INTERVAL
    right = kind == RIGHT_CENSORED
    if not valid(batch.value, reported) or not valid(batch.lower, interval | right) or not valid(batch.upper, interval):
        _invalid()
    if bool((batch.lower[interval] > batch.upper[interval]).any()):
        _invalid()
    return batch


def validate_normalized_age(age: Tensor) -> Tensor:
    """Fail closed on the normalized seven-feature representation used by V2."""

    if not isinstance(age, Tensor) or age.ndim != 2 or age.shape[1] != AGE_FEATURE_DIM or not age.is_floating_point() or not torch.isfinite(age).all():
        _invalid()
    numeric, onehot = age[:, :3], age[:, 3:]
    if not torch.all((onehot == 0) | (onehot == 1)) or not torch.all(onehot.sum(dim=1) == 1):
        _invalid()
    kind = onehot.argmax(dim=1)
    reported, interval, right, unknown = kind == REPORTED, kind == INTERVAL, kind == RIGHT_CENSORED, kind == UNKNOWN
    if (numeric[reported, 1:] != 0).any() or (numeric[interval, 0] != 0).any() or (numeric[right][:, (0, 2)] != 0).any() or (numeric[unknown] != 0).any():
        _invalid()
    if (numeric[interval, 1] > numeric[interval, 2]).any():
        _invalid()
    return age


def normalize_age(batch: AgeBatch, mean: float, scale: float) -> Tensor:
    """Return ``[B, 7]``: standardized active values followed by kind one-hot."""

    validate_age(batch)
    if isinstance(mean, bool) or isinstance(scale, bool):
        _invalid()
    try:
        mean_value, scale_value = float(mean), float(scale)
    except (TypeError, ValueError, OverflowError):
        _invalid()
    if not torch.isfinite(torch.tensor(mean_value)) or not torch.isfinite(torch.tensor(scale_value)) or scale_value <= 0:
        _invalid()
    kind = batch.kind.to(torch.long)
    out = torch.zeros((batch.value.shape[0], AGE_FEATURE_DIM), device=batch.value.device, dtype=batch.value.dtype)
    reported, interval, right = kind == REPORTED, kind == INTERVAL, kind == RIGHT_CENSORED
    out[reported, 0] = (batch.value[reported] - mean_value) / scale_value
    out[interval | right, 1] = (batch.lower[interval | right] - mean_value) / scale_value
    out[interval, 2] = (batch.upper[interval] - mean_value) / scale_value
    out[:, 3:] = torch.nn.functional.one_hot(kind, num_classes=4).to(out.dtype)
    validate_normalized_age(out)
    return out


def augment_reported_age(batch: AgeBatch, generator: torch.Generator) -> AgeBatch:
    """Convert reported ages: 10% unknown, 10% documented five-year bins, 80% kept.

    The caller supplies the generator, making the augmentation deterministic and
    avoiding any global RNG side effects.  Non-reported observations are copied
    unchanged.
    """

    validate_age(batch)
    if not isinstance(generator, torch.Generator):
        _invalid()
    if generator.device != batch.value.device:
        _invalid()
    value, lower, upper, kind = (x.clone() for x in (batch.value, batch.lower, batch.upper, batch.kind))
    reported = kind.to(torch.long) == REPORTED
    draw = torch.rand(kind.shape, generator=generator, device=kind.device)
    unknown = reported & (draw < 0.10)
    interval = reported & (draw >= 0.10) & (draw < 0.20)
    kind[unknown] = UNKNOWN
    value[unknown], lower[unknown], upper[unknown] = float("nan"), float("nan"), float("nan")
    floors = torch.floor(value[interval] / 5.0) * 5.0
    kind[interval] = INTERVAL
    lower[interval], upper[interval], value[interval] = floors, floors + 5.0, float("nan")
    return AgeBatch(value, lower, upper, kind)
