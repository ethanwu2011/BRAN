"""Small, side-effect-free gradient geometry diagnostics for BRAN objectives.

The values returned by :func:`gradient_summary`, including the scalar
per-batch loss values, are private caller-owned diagnostics.  They must not be
released as per-example output; only a separately supported pooled aggregate
may be released.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import torch


def _validate_losses(losses: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    if not isinstance(losses, Mapping):
        raise TypeError("losses must be a mapping")

    task_names: list[str] = []
    for task, loss in losses.items():
        if not isinstance(task, str):
            raise TypeError("loss names must be strings")
        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"loss for task {task!r} must be a torch.Tensor")
        if loss.ndim != 0:
            raise ValueError(f"loss for task {task!r} must be scalar")
        if not loss.requires_grad:
            raise ValueError(f"loss for task {task!r} must be differentiable")
        if not loss.is_floating_point():
            raise TypeError(f"loss for task {task!r} must be a real floating tensor")
        if not bool(torch.isfinite(loss.detach()).item()):
            raise ValueError(f"loss for task {task!r} must be finite")
        task_names.append(task)

    if not task_names:
        raise ValueError("losses must be nonempty")
    return tuple(task_names)


def _validate_groups(
    groups: Mapping[str, tuple[torch.nn.Parameter, ...]],
) -> tuple[tuple[str, tuple[torch.nn.Parameter, ...]], ...]:
    if not isinstance(groups, Mapping):
        raise TypeError("groups must be a mapping")
    if not groups:
        raise ValueError("groups must contain at least one nonempty group")

    validated: list[tuple[str, tuple[torch.nn.Parameter, ...]]] = []
    seen: dict[int, str] = {}
    for group_name, parameters in groups.items():
        if not isinstance(group_name, str):
            raise TypeError("group names must be strings")
        if not isinstance(parameters, tuple):
            raise TypeError(f"parameters for group {group_name!r} must be a tuple")
        if not parameters:
            raise ValueError(f"group {group_name!r} must be nonempty")

        group_seen: set[int] = set()
        for parameter in parameters:
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError(
                    f"group {group_name!r} contains a non-Parameter value"
                )
            parameter_id = id(parameter)
            if parameter_id in group_seen:
                raise ValueError(
                    f"parameter appears more than once in group {group_name!r}"
                )
            if parameter_id in seen:
                raise ValueError(
                    f"parameter is shared by groups {seen[parameter_id]!r} and "
                    f"{group_name!r}"
                )
            group_seen.add(parameter_id)
            seen[parameter_id] = group_name

        validated.append((group_name, parameters))

    return tuple(validated)


def _validate_weights(
    weights: Mapping[str, float], task_names: tuple[str, ...]
) -> dict[str, float]:
    if not isinstance(weights, Mapping):
        raise TypeError("weights must be a mapping")

    expected = set(task_names)
    actual = set(weights)
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        details: list[str] = []
        if missing:
            details.append(f"missing tasks {sorted(missing)!r}")
        if extra:
            details.append(f"unknown tasks {sorted(extra)!r}")
        suffix = "; ".join(details)
        raise ValueError(f"weights must contain exactly one value per loss task{': ' + suffix if suffix else ''}")

    checked: dict[str, float] = {}
    for task in task_names:
        weight = weights[task]
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise TypeError(f"weight for task {task!r} must be a real number")
        weight_float = float(weight)
        if not math.isfinite(weight_float):
            raise ValueError(f"weight for task {task!r} must be finite")
        if weight_float <= 0.0:
            raise ValueError(f"weight for task {task!r} must be positive")
        checked[task] = weight_float
    return checked


def _as_float64_vector(gradient: torch.Tensor) -> torch.Tensor:
    """Return a detached dense gradient as a real float64 vector."""

    if gradient.is_sparse:
        gradient = gradient.to_dense()
    if not bool(torch.isfinite(gradient).all().item()):
        raise ValueError("autograd produced a nonfinite gradient")

    detached = gradient.detach()
    if detached.is_complex():
        # Treat real and imaginary components as adjacent coordinates.  This
        # keeps norms, dots, and weighted sums real and uses binary64 math.
        return torch.view_as_real(detached).reshape(-1).to(dtype=torch.float64)
    return detached.reshape(-1).to(dtype=torch.float64)


def _zero_float64_vector(parameter: torch.nn.Parameter) -> torch.Tensor:
    coordinate_count = parameter.numel() * (2 if parameter.is_complex() else 1)
    return torch.zeros(
        coordinate_count, dtype=torch.float64, device=parameter.device
    )


def _norm_squared(vector: torch.Tensor) -> float:
    # The explicit dtype keeps accumulation in float64 even for half/bfloat16
    # model parameters.
    return float(torch.sum(vector * vector, dtype=torch.float64).item())


def gradient_summary(
    losses: dict[str, torch.Tensor],
    groups: dict[str, tuple[torch.nn.Parameter, ...]],
    weights: dict[str, float],
) -> dict[str, Any]:
    """Summarize per-task gradient geometry without mutating caller state.

    ``loss_values`` are included for a private caller-owned diagnostic only;
    even these returned scalar per-batch values must remain private.  Only a
    separately supported pooled aggregate may be released.  This function
    never calls ``backward``, never writes ``Parameter.grad``, and does not
    return raw gradients.
    """

    task_names = _validate_losses(losses)
    validated_groups = _validate_groups(groups)
    checked_weights = _validate_weights(weights, task_names)

    # Preserve caller insertion order while making a private copy of scalar
    # values.  ``detach`` and ``item`` do not modify the loss or its graph.
    loss_values = {
        task: float(losses[task].detach().item()) for task in task_names
    }

    union_parameters: tuple[torch.nn.Parameter, ...] = tuple(
        parameter
        for _, parameters in validated_groups
        for parameter in parameters
    )
    differentiable_parameters = tuple(
        parameter for parameter in union_parameters if parameter.requires_grad
    )

    # Keep gradients private and detached.  They are retained only for the
    # duration of this call and never appear in the returned structure.
    task_gradients: dict[str, dict[str, tuple[torch.Tensor, ...]]] = {}
    for task in task_names:
        if differentiable_parameters:
            gradients = torch.autograd.grad(
                losses[task],
                differentiable_parameters,
                allow_unused=True,
                retain_graph=True,
                create_graph=False,
            )
            gradients_by_id = {
                id(parameter): gradient
                for parameter, gradient in zip(differentiable_parameters, gradients)
            }
        else:
            gradients_by_id = {}

        per_group: dict[str, tuple[torch.Tensor, ...]] = {}
        for group_name, parameters in validated_groups:
            vectors: list[torch.Tensor] = []
            for parameter in parameters:
                gradient = gradients_by_id.get(id(parameter))
                if gradient is None:
                    # A fresh zero is used only for local computation; the
                    # parameter itself and its existing .grad are untouched.
                    vector = _zero_float64_vector(parameter)
                else:
                    vector = _as_float64_vector(gradient)
                    expected_coordinates = parameter.numel() * (
                        2 if parameter.is_complex() else 1
                    )
                    if vector.numel() != expected_coordinates:
                        raise RuntimeError("autograd returned a gradient with the wrong shape")
                vectors.append(vector)
            per_group[group_name] = tuple(vectors)
        task_gradients[task] = per_group

    result_groups: dict[str, dict[str, Any]] = {}
    for group_name, parameters in validated_groups:
        del parameters  # Group membership is represented by local vectors.

        norm_squared_by_task: dict[str, float] = {}
        norms: dict[str, float] = {}
        weighted_norms: dict[str, float] = {}
        for task in task_names:
            norm_squared = sum(
                _norm_squared(vector)
                for vector in task_gradients[task][group_name]
            )
            norm_squared_by_task[task] = norm_squared
            if not math.isfinite(norm_squared):
                raise ValueError("gradient norm accumulation is nonfinite")
            norm = math.sqrt(max(0.0, norm_squared))
            norms[task] = norm
            weighted_norms[task] = checked_weights[task] * norm
            if not math.isfinite(weighted_norms[task]):
                raise ValueError("weighted gradient norm is nonfinite")

        cosines: dict[str, float | None] = {}
        for first_index, first_task in enumerate(task_names):
            first_vectors = task_gradients[first_task][group_name]
            for second_task in task_names[first_index + 1 :]:
                second_vectors = task_gradients[second_task][group_name]
                first_norm_squared = norm_squared_by_task[first_task]
                second_norm_squared = norm_squared_by_task[second_task]
                pair_key = f"{first_task}|{second_task}"
                if first_norm_squared == 0.0 or second_norm_squared == 0.0:
                    cosines[pair_key] = None
                    continue

                dot = 0.0
                for first_vector, second_vector in zip(first_vectors, second_vectors):
                    dot += float(
                        torch.sum(first_vector * second_vector, dtype=torch.float64).item()
                    )
                cosine = (dot / math.sqrt(first_norm_squared)) / math.sqrt(second_norm_squared)
                if not math.isfinite(cosine):
                    raise ValueError("gradient cosine is nonfinite")
                # Account for a final ulp of roundoff without altering genuine
                # values in the mathematically valid interval.
                cosines[pair_key] = max(-1.0, min(1.0, cosine))

        weighted_sum_squared = 0.0
        for parameter_index in range(len(task_gradients[task_names[0]][group_name])) if task_names else range(0):
            weighted_vector: torch.Tensor | None = None
            for task in task_names:
                vector = task_gradients[task][group_name][parameter_index]
                contribution = vector * checked_weights[task]
                weighted_vector = (
                    contribution
                    if weighted_vector is None
                    else weighted_vector + contribution
                )
            if weighted_vector is not None:
                weighted_sum_squared += _norm_squared(weighted_vector)

        if not math.isfinite(weighted_sum_squared):
            raise ValueError("weighted gradient accumulation is nonfinite")
        result_groups[group_name] = {
            "norms": norms,
            "weighted_norms": weighted_norms,
            "cosines": cosines,
            "weighted_sum_norm": math.sqrt(max(0.0, weighted_sum_squared)),
        }

    return {"loss_values": loss_values, "groups": result_groups}


__all__ = ["gradient_summary"]
