"""Prospective mask-only tasks for controlled supervised training.

This is deliberately a pure, no-I/O kernel: it accepts only missingness masks
and creates no targets, losses, rows, values, or fitted state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ERROR = "bran_supervised_mask_tasks_v1_invalid_input"
SCREEN_PATTERNS = (
    "clinical_drop25", "clinical_drop50", "clinical_drop75",
    "whole_cbc_hidden", "whole_cbc_no_retina", "all_clinical_hidden",
)


@dataclass(frozen=True, repr=False)
class SupervisedMasks:
    screen_clinical_mask: np.ndarray
    screen_retinal_mask: np.ndarray
    screen_available: np.ndarray
    cbc_clinical_mask: np.ndarray
    cbc_retinal_mask: np.ndarray
    cbc_target_mask: np.ndarray
    cbc_available: np.ndarray
    screen_pattern: str
    cbc_pattern: str


def _fail():
    raise ValueError(ERROR) from None


def _validate(clinical_mask, retinal_mask, cbc_slots, step, rng):
    if (type(clinical_mask) is not np.ndarray or clinical_mask.dtype != np.dtype(bool)
            or clinical_mask.ndim != 2 or clinical_mask.shape[1] != 59):
        _fail()
    n = clinical_mask.shape[0]
    if n < 1 or type(retinal_mask) is not np.ndarray or retinal_mask.dtype != np.dtype(bool) or retinal_mask.shape != (n,):
        _fail()
    if type(cbc_slots) is not tuple or len(cbc_slots) != 9:
        _fail()
    if any(type(slot) is not int or not 0 <= slot <= 47 for slot in cbc_slots) or len(set(cbc_slots)) != 9:
        _fail()
    if type(step) is not int or step < 0 or not isinstance(rng, np.random.Generator):
        _fail()


def _readonly(value):
    value.setflags(write=False)
    return value


def build_masks(clinical_mask, retinal_mask, cbc_slots, *, step, rng):
    """Create fixed-consumption screen and CBC masking tasks.

    True always means an originally observed / still supplied item.  CBC targets
    are a subset of originally observed CBC slots and are never refilled from a
    missing source.  The caller is responsible for erasing payloads at false
    mask locations and filtering downstream losses by the returned availability.
    """
    _validate(clinical_mask, retinal_mask, cbc_slots, step, rng)
    n = clinical_mask.shape[0]
    # These three draws occur on every call, including whole-modality patterns,
    # so schedule changes cannot perturb the Generator's subsequent stream.
    screen_uniform = rng.random((n, 59))
    cbc_uniform = rng.random((n, 9))
    count_uniform = rng.random((n, 1))

    original_clinical = clinical_mask
    original_retinal = retinal_mask
    screen_pattern = SCREEN_PATTERNS[step % len(SCREEN_PATTERNS)]
    screen_clinical = original_clinical.copy()
    screen_retinal = original_retinal.copy()
    if screen_pattern.startswith("clinical_drop"):
        fraction = {"clinical_drop25": .25, "clinical_drop50": .50,
                    "clinical_drop75": .75}[screen_pattern]
        screen_clinical &= screen_uniform >= fraction
    elif screen_pattern == "whole_cbc_hidden":
        screen_clinical[:, cbc_slots] = False
    elif screen_pattern == "whole_cbc_no_retina":
        screen_clinical[:, cbc_slots] = False
        screen_retinal[:] = False
    else:  # all_clinical_hidden
        screen_clinical[:] = False
    screen_available = screen_clinical.any(axis=1) | screen_retinal

    cbc_observed = original_clinical[:, cbc_slots]
    cbc_clinical = original_clinical.copy()
    cbc_retinal = original_retinal.copy()
    cbc_target = np.zeros((n, 9), dtype=bool)
    if step % 2 == 0:
        cbc_pattern = "partial_cbc"
        for row in range(n):
            observed = cbc_observed[row]
            n_observed = int(observed.sum())
            if n_observed >= 2:
                # Select 1..n_observed-1 by a uniform count, then take that
                # many lowest random ranks among *only* observed CBC entries.
                count = 1 + int(count_uniform[row, 0] * (n_observed - 1))
                candidates = np.flatnonzero(observed)
                ranked = candidates[np.argsort(cbc_uniform[row, candidates], kind="stable")]
                cbc_target[row, ranked[:count]] = True
        cbc_clinical[:, cbc_slots] &= ~cbc_target
    else:
        cbc_pattern = "whole_cbc_no_retina"
        cbc_target[:] = cbc_observed
        cbc_clinical[:, cbc_slots] = False
        cbc_retinal[:] = False
    cbc_available = (cbc_clinical.any(axis=1) | cbc_retinal) & cbc_target.any(axis=1)

    return SupervisedMasks(
        _readonly(screen_clinical), _readonly(screen_retinal), _readonly(screen_available),
        _readonly(cbc_clinical), _readonly(cbc_retinal), _readonly(cbc_target),
        _readonly(cbc_available), screen_pattern, cbc_pattern,
    )
