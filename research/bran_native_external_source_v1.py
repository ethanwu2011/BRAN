"""Pure adapter for private native laboratory arrays from three sources.

The adapter performs schema and value validation only.  It does not read
files, authenticate a source, normalize measurements, train a model, or emit
row content.  Its task masks are eligibility gates for a future learner, not
evidence that more data improves performance.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from bran_external_native_age_v1 import AGE_KINDS, eligible_native_age
from bran_joint_lab_pretraining_v1 import _validate_source_arrays


_SOURCES = frozenset({"mimic", "nhanes", "eicu"})
_REQUIRED_FIELDS = frozenset(
    {
        "values",
        "observed",
        "provenance",
        "person_group",
        "split",
        "adult_qualified",
        "age_triplet",
        "age_kind",
    }
)
_CBC_WIDTH = 9
_JOINT_WIDTH = 21


def _readonly(array: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    copied = np.array(array, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


def _validate_lab_arrays(
    values: np.ndarray, observed: np.ndarray, provenance: np.ndarray, width: int
) -> None:
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 2
        or values.shape[1] != width
        or values.dtype.kind != "f"
        or not isinstance(observed, np.ndarray)
        or observed.shape != values.shape
        or observed.dtype != np.dtype(bool)
        or not isinstance(provenance, np.ndarray)
        or provenance.shape != values.shape
        or provenance.dtype.kind not in "iu"
    ):
        raise ValueError("external laboratory arrays are invalid")

    if values.shape[0] <= 0:
        raise ValueError("external laboratory arrays are invalid")
    if (
        (provenance.dtype.kind == "u" and np.any(provenance > 1))
        or (provenance.dtype.kind == "i" and np.any((provenance < 0) | (provenance > 1)))
        or not np.array_equal(provenance, observed.astype(provenance.dtype))
    ):
        raise ValueError("external laboratory provenance is invalid")
    if np.any(~observed & ~np.isnan(values)):
        raise ValueError("missing laboratory values must be NaN")
    if np.any(observed & ~np.isfinite(values)):
        raise ValueError("observed laboratory values must be finite")
    if np.any(observed[:, :_CBC_WIDTH] & (values[:, :_CBC_WIDTH] <= 0.0)):
        raise ValueError("observed CBC values must be positive")
    if width > _CBC_WIDTH and np.any(
        observed[:, _CBC_WIDTH:] & (values[:, _CBC_WIDTH:] < 0.0)
    ):
        raise ValueError("observed chemistry values must be nonnegative")


def _pad_eicu(
    values: np.ndarray, observed: np.ndarray, provenance: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = values.shape[0]
    padded_values = np.full((n, _JOINT_WIDTH), np.nan, dtype=np.float64)
    padded_observed = np.zeros((n, _JOINT_WIDTH), dtype=np.bool_)
    padded_provenance = np.zeros((n, _JOINT_WIDTH), dtype=np.uint8)
    padded_values[:, :_CBC_WIDTH] = values
    padded_observed[:, :_CBC_WIDTH] = observed
    padded_provenance[:, :_CBC_WIDTH] = provenance
    return padded_values, padded_observed, padded_provenance


def adapt_source(source: str, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate and adapt one private MIMIC, NHANES, or eICU array mapping.

    MIMIC and NHANES inputs contain all 21 joint laboratory columns.  eICU
    inputs contain only the nine CBC columns; its twelve chemistry columns
    are padded as missing (``NaN`` values, false observation flags, and zero
    provenance codes).  Age provenance is retained, while scalar age is
    derived through :func:`eligible_native_age` and is ``NaN`` when ineligible.
    Every returned array is a fresh read-only copy.
    """

    if not isinstance(source, str) or source not in _SOURCES:
        raise ValueError("unsupported external source")
    if not isinstance(arrays, Mapping) or not _REQUIRED_FIELDS.issubset(arrays):
        raise ValueError("external source arrays are incomplete")

    values = arrays["values"]
    observed = arrays["observed"]
    provenance = arrays["provenance"]
    expected_width = _CBC_WIDTH if source == "eicu" else _JOINT_WIDTH
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 2
        or values.shape[1] != expected_width
    ):
        raise ValueError("external laboratory arrays have an invalid width")
    _validate_lab_arrays(values, observed, provenance, expected_width)
    n = values.shape[0]

    if source == "eicu":
        joint_values, joint_observed, joint_provenance = _pad_eicu(
            values, observed, provenance
        )
    else:
        joint_values = np.array(values, dtype=np.float64, copy=True)
        joint_observed = np.array(observed, dtype=np.bool_, copy=True)
        joint_provenance = np.array(provenance, dtype=np.uint8, copy=True)

    # The shared validator is intentionally called on exactly its six-key
    # contract subset.  It supplies the source-local group/split checks;
    # laboratory value/provenance checks above remain explicit here.
    _, _, _, normalized_groups, normalized_split = _validate_source_arrays(
        source,
        {
            "values": joint_values,
            "observed": joint_observed,
            "provenance": joint_provenance,
            "person_group": arrays["person_group"],
            "split": arrays["split"],
            "adult_qualified": arrays["adult_qualified"],
        },
    )

    triplet = arrays["age_triplet"]
    kind = arrays["age_kind"]
    adult = arrays["adult_qualified"]
    point_age, native_age_eligible = eligible_native_age(triplet, kind, adult)

    cbc_observed_count = joint_observed[:, :_CBC_WIDTH].sum(axis=1)
    in_training_split = normalized_split == 0
    partial = in_training_split & native_age_eligible & (cbc_observed_count >= 2)
    whole = partial & joint_observed[:, _CBC_WIDTH:].any(axis=1)

    return {
        "values": _readonly(joint_values, np.float64),
        "observed": _readonly(joint_observed, np.bool_),
        "provenance": _readonly(joint_provenance, np.uint8),
        "person_group": _readonly(normalized_groups, np.int64),
        "split": _readonly(normalized_split, np.int64),
        "adult_qualified": _readonly(adult, np.bool_),
        "age_triplet": _readonly(triplet, np.float64),
        "age_kind": _readonly(kind, np.uint8),
        "scalar_age": _readonly(point_age, np.float64),
        "native_age_eligible": _readonly(native_age_eligible, np.bool_),
        "partial_cbc_eligible": _readonly(partial, np.bool_),
        "whole_cbc_eligible": _readonly(whole, np.bool_),
    }


__all__ = ["AGE_KINDS", "adapt_source"]
