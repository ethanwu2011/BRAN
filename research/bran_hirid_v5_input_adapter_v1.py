"""Pure HiRID V5 input assembly from already-selected episode results.

This adapter does not read a source, authenticate a schema, derive age, choose
a model, normalize values, or invoke inference.  Its model-input fields are
the original-unit 59-slot clinical arrays and zero/false retinal arrays.  The
separate ``clinical_eligible`` field is a caller-facing physiology filter; it
is not a model feature and does not encode target or anchor metadata.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
import torch

from bran_hirid_v5_observation_kernel import (
    EpisodeSelection,
    INPUT_CANDIDATES,
    STATUS,
)
from bran_knhanes_input_kernel_v1 import (
    ADMITTED_CANONICAL_INDICES,
    CANONICAL_INDEX,
    CANONICAL_NAMES,
)
from bran_multisource_age_v2 import AgeBatch, UNKNOWN


ERROR = "hirid_v5_input_adapter_contract_failed"
RETINAL_FEATURE_DIM = 384
CLINICAL_WIDTH = len(CANONICAL_NAMES)

_READY = "ready"
_ALLOWED_FIELDS = frozenset(field for field, _factor in INPUT_CANDIDATES.values())
_ALLOWED_INDICES = tuple(sorted(CANONICAL_INDEX[field] for field in _ALLOWED_FIELDS))
_EXPECTED_FIELDS = frozenset((
    "potassium", "sodium", "chloride", "creatinine", "bilirubin_total",
    "albumin", "glucose",
))


@dataclass(frozen=True, slots=True, repr=False)
class HiRIDV5InputBatch:
    """Readonly model inputs plus a separate clinical-only filter.

    ``clinical_values``, ``clinical_mask``, ``retinal_features`` and
    ``retinal_mask`` are the model payload.  Clinical values remain in the
    canonical original units supplied by the selector; no target or timestamp
    is represented.  ``ages`` returns a fresh, caller-owned all-UNKNOWN typed-
    age batch on each access; mutating one returned batch cannot alter this
    result.
    ``clinical_eligible`` is only a caller-facing filter indicating that a
    ready row has at least one allowed physiological field; it is not part of
    the model payload and must be combined with any external status policy by
    the caller.
    """

    clinical_values: np.ndarray
    clinical_mask: np.ndarray
    retinal_features: np.ndarray
    retinal_mask: np.ndarray
    clinical_eligible: np.ndarray

    def __repr__(self) -> str:
        return "<HiRIDV5InputBatch>"

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)

    @property
    def ages(self) -> AgeBatch:
        return _unknown_ages(self.clinical_values.shape[0])


class _Invalid(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _Invalid


def _finite_real(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise _Invalid
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise _Invalid from None
    if not math.isfinite(number):
        raise _Invalid
    return number


def _validated_selection(selection: object) -> tuple[str, dict[str, float]]:
    # Exact type is intentional: this boundary accepts only selector results,
    # not lookalike objects carrying source-controlled fields.
    _require(type(selection) is EpisodeSelection)
    status = selection.status
    _require(type(status) is str and status in STATUS)
    labs = selection.canonical_labs
    _require(isinstance(labs, Mapping))

    copied: dict[str, float] = {}
    for field, value in labs.items():
        _require(type(field) is str and field in _ALLOWED_FIELDS)
        _require(field not in copied)
        copied[field] = _finite_real(value)
    # A selector result marked ready must contain a usable physiological value.
    # Non-ready results are accepted as abstentions, and their mapping is never
    # copied into model payload even when it contains allowed fields.
    if status == _READY:
        _require(bool(copied))
    return status, copied


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _unknown_ages(n_rows: int) -> AgeBatch:
    # Keep the typed-age bridge deliberately small: all active age values are
    # NaN and the representation is explicitly UNKNOWN for every row.
    nan = torch.full((n_rows,), float("nan"), dtype=torch.float32)
    kind = torch.full((n_rows,), UNKNOWN, dtype=torch.long)
    return AgeBatch(nan.clone(), nan.clone(), nan, kind)


def prepare_inputs(selections: Iterable[EpisodeSelection]) -> HiRIDV5InputBatch:
    """Assemble fixed-width model inputs from exact selector results.

    The selector's target and anchor fields are deliberately never read.  Only
    ready rows with allowed finite physiology populate model slots.  All other
    statuses produce an empty row and therefore no clinical-only eligibility.
    """
    try:
        # Pin this adapter's seven-field contract even if a future candidate
        # mapping grows; and keep every writable slot in the admitted, non-CBC
        # portion of the fixed clinical registry.
        _require(CLINICAL_WIDTH == 59)
        _require(_ALLOWED_FIELDS == _EXPECTED_FIELDS)
        _require(set(_ALLOWED_INDICES).issubset(ADMITTED_CANONICAL_INDICES))
        rows = [_validated_selection(selection) for selection in selections]
        n_rows = len(rows)
        clinical_values = np.full((n_rows, CLINICAL_WIDTH), np.nan, dtype=np.float64)
        clinical_mask = np.zeros((n_rows, CLINICAL_WIDTH), dtype=bool)
        retinal_features = np.zeros((n_rows, RETINAL_FEATURE_DIM), dtype=np.float64)
        retinal_mask = np.zeros(n_rows, dtype=bool)
        clinical_eligible = np.zeros(n_rows, dtype=bool)

        for row_index, (status, labs) in enumerate(rows):
            if status != _READY:
                continue
            for field, value in labs.items():
                canonical_index = CANONICAL_INDEX[field]
                clinical_values[row_index, canonical_index] = value
                clinical_mask[row_index, canonical_index] = True
            clinical_eligible[row_index] = bool(labs)

        # The fixed width and all-hidden slots are structural invariants.  The
        # assertion also prevents a future allowed-field edit from admitting a
        # CBC or history/disabled slot without changing this adapter explicitly.
        _require(set(_ALLOWED_INDICES).issubset(range(CLINICAL_WIDTH)))
        _require(not clinical_mask[:, [CANONICAL_INDEX[field] for field in (
            "hct", "hemoglobin", "mch", "mchc", "mcv", "plt", "rbc", "rdw", "wbc"
        )]].any())

        return HiRIDV5InputBatch(
            _readonly_copy(clinical_values),
            _readonly_copy(clinical_mask),
            _readonly_copy(retinal_features),
            _readonly_copy(retinal_mask),
            _readonly_copy(clinical_eligible),
        )
    except _Invalid:
        raise ValueError(ERROR) from None
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


__all__ = ["ERROR", "HiRIDV5InputBatch", "prepare_inputs"]
