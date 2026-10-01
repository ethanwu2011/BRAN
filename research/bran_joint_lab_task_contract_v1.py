"""Pure task contracts for a future 21-field CBC-plus-chemistry learner.

No data ingestion, source admission, task mixture, loss, optimization, or
performance claim is implemented here.
"""

from __future__ import annotations

import numpy as np

from bran_joint_lab_cache_v1 import FIELDS


REGISTRY_WIDTH = 59
CBC_WIDTH = 9
JOINT_WIDTH = len(FIELDS)
_MODES = frozenset({"partial_cbc", "whole_cbc"})


def _joint_observed(value: object, error: str = "joint observed mask is invalid") -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1] != JOINT_WIDTH or value.dtype != np.dtype(bool):
        raise ValueError(error)
    return value


def _registry_slots(registry_fields: object) -> tuple[int, ...]:
    if not isinstance(registry_fields, tuple) or len(registry_fields) != REGISTRY_WIDTH:
        raise ValueError("registry schema must be a 59-field tuple")
    if any(not isinstance(field, str) for field in registry_fields) or len(set(registry_fields)) != REGISTRY_WIDTH:
        raise ValueError("registry schema must contain unique field names")
    if any(field not in registry_fields for field in FIELDS):
        raise ValueError("registry schema is missing a joint laboratory field")
    slots = tuple(registry_fields.index(field) for field in FIELDS)
    if any(slot >= 48 for slot in slots):
        raise ValueError("joint laboratory fields must occupy registry slots below 48")
    return slots


def _readonly(array: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    copied = np.array(array, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


def project_joint_labs_to_registry(
    values: np.ndarray,
    observed: np.ndarray,
    provenance: np.ndarray,
    registry_fields: tuple[str, ...],
    medians: np.ndarray,
    iqrs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project only original observed joint-lab values into fixed registry slots."""
    if not isinstance(values, np.ndarray) or values.ndim != 2 or values.shape[1] != JOINT_WIDTH or values.dtype.kind != "f":
        raise ValueError("joint values must have shape [rows, 21] and floating dtype")
    mask = _joint_observed(observed)
    if mask.shape != values.shape:
        raise ValueError("joint observed mask is invalid")
    if not isinstance(provenance, np.ndarray) or provenance.shape != values.shape or provenance.dtype.kind not in "iu":
        raise ValueError("joint provenance is invalid")
    if np.any(provenance < 0) or np.any(provenance > 1) or not np.array_equal(provenance, mask.astype(provenance.dtype)):
        raise ValueError("joint provenance must be observed=1 and unobserved=0")
    if np.any(~mask & ~np.isnan(values)):
        raise ValueError("missing joint values must be NaN")
    if np.any(mask & ~np.isfinite(values)):
        raise ValueError("observed joint values must be finite")
    if np.any(mask[:, :CBC_WIDTH] & (values[:, :CBC_WIDTH] <= 0.0)):
        raise ValueError("observed CBC values must be positive")
    if np.any(mask[:, CBC_WIDTH:] & (values[:, CBC_WIDTH:] < 0.0)):
        raise ValueError("observed chemistry values must be nonnegative")
    slots = _registry_slots(registry_fields)
    for scale, label, positive in ((medians, "medians", False), (iqrs, "iqrs", True)):
        if not isinstance(scale, np.ndarray) or scale.shape != (REGISTRY_WIDTH,) or scale.dtype.kind not in "f":
            raise ValueError(label + " must be a finite 59-vector")
        if not np.isfinite(scale).all() or (positive and np.any(scale <= 0.0)):
            raise ValueError(label + " must be a finite 59-vector")
    output = np.zeros((values.shape[0], REGISTRY_WIDTH), dtype=np.float64)
    output_mask = np.zeros((values.shape[0], REGISTRY_WIDTH), dtype=bool)
    for field_index, slot in enumerate(slots):
        active = mask[:, field_index]
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = (values[active, field_index] - medians[slot]) / iqrs[slot]
        if not np.isfinite(normalized).all():
            raise ValueError("joint normalization produced nonfinite values")
        output[active, slot] = normalized
        output_mask[:, slot] = active
    return _readonly(output, np.float64), _readonly(output_mask, bool)


def task_eligibility(observed: np.ndarray, mode: str) -> np.ndarray:
    """Return task eligibility without selecting a task mixture or source rows."""
    mask = _joint_observed(observed)
    if not isinstance(mode, str) or mode not in _MODES:
        raise ValueError("joint task mode is invalid")
    cbc = mask[:, :CBC_WIDTH].sum(axis=1)
    if mode == "partial_cbc":
        return _readonly(cbc >= 2, bool)
    return _readonly((cbc >= 1) & mask[:, CBC_WIDTH:].any(axis=1), bool)


def build_task_masks(observed: np.ndarray, mode: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Build visible/hidden 21-field masks for one explicitly selected task mode."""
    mask = _joint_observed(observed)
    if not isinstance(mode, str) or mode not in _MODES:
        raise ValueError("joint task mode is invalid")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a NumPy Generator")
    eligible = task_eligibility(mask, mode)
    if not np.all(eligible):
        raise ValueError("every row must be eligible for the selected joint task")
    visible = mask.copy()
    hidden = np.zeros_like(mask)
    if mode == "whole_cbc":
        hidden[:, :CBC_WIDTH] = mask[:, :CBC_WIDTH]
        visible[:, :CBC_WIDTH] = False
    else:
        for row in range(mask.shape[0]):
            candidates = np.flatnonzero(mask[row, :CBC_WIDTH])
            hidden_count = int(rng.integers(1, len(candidates)))
            selected = rng.choice(candidates, size=hidden_count, replace=False)
            visible[row, selected] = False
            hidden[row, selected] = True
    if (
        np.any(visible & hidden)
        or not np.array_equal(visible | hidden, mask)
        or np.any(hidden[:, CBC_WIDTH:])
        or np.any(~visible[:, CBC_WIDTH:] & mask[:, CBC_WIDTH:])
    ):
        raise ValueError("joint task masks are invalid")
    return _readonly(visible, bool), _readonly(hidden, bool)


def masked_registry_inputs(
    projected_values: np.ndarray,
    projected_mask: np.ndarray,
    visible: np.ndarray,
    hidden: np.ndarray,
    registry_fields: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Remove hidden CBC targets from projected registry values and indicators."""
    if (
        not isinstance(projected_values, np.ndarray)
        or projected_values.ndim != 2
        or projected_values.shape[1] != REGISTRY_WIDTH
        or projected_values.dtype.kind != "f"
        or not np.isfinite(projected_values).all()
    ):
        raise ValueError("projected registry values are invalid")
    if (
        not isinstance(projected_mask, np.ndarray)
        or projected_mask.shape != projected_values.shape
        or projected_mask.dtype != np.dtype(bool)
    ):
        raise ValueError("projected registry mask is invalid")
    if np.any(~projected_mask & (projected_values != 0.0)):
        raise ValueError("inactive projected registry values must be zero")
    visible_mask = _joint_observed(visible, "joint visible mask is invalid")
    hidden_mask = _joint_observed(hidden, "joint hidden mask is invalid")
    if visible_mask.shape[0] != projected_values.shape[0] or hidden_mask.shape != visible_mask.shape:
        raise ValueError("joint task masks are invalid")
    slots = _registry_slots(registry_fields)
    joint_registry_mask = projected_mask[:, slots]
    non_joint = np.ones(REGISTRY_WIDTH, dtype=bool)
    non_joint[list(slots)] = False
    if np.any(projected_mask[:, non_joint]):
        raise ValueError("projected registry mask contains nonjoint observations")
    if (
        np.any(visible_mask & hidden_mask)
        or not np.array_equal(visible_mask | hidden_mask, joint_registry_mask)
        or np.any(hidden_mask[:, CBC_WIDTH:])
        or np.any(~visible_mask[:, CBC_WIDTH:] & joint_registry_mask[:, CBC_WIDTH:])
    ):
        raise ValueError("joint task masks are invalid")
    input_values = np.array(projected_values, dtype=np.float64, copy=True)
    input_mask = np.array(projected_mask, dtype=bool, copy=True)
    for field_index in range(CBC_WIDTH):
        target = hidden_mask[:, field_index]
        input_values[target, slots[field_index]] = 0.0
        input_mask[target, slots[field_index]] = False
    return _readonly(input_values, np.float64), _readonly(input_mask, bool)
