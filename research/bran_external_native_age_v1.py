"""Strict bridge from the private age cache to a native scalar-age covariate.

This module is an eligibility gate, not a model fit and not evidence that
more data improves performance.  It performs no normalization or source
authentication; those responsibilities remain with the caller.  Censored,
rounded, unresolved, and missing source ages stay ineligible rather than
being converted to invented point ages.
"""

from __future__ import annotations

import numpy as np


# Keep this closed schema synchronized with the cache and observed-pool
# builders.  Integers in ``age_kind`` are indexes into this tuple.
AGE_KINDS = (
    "missing_or_invalid",
    "source_age_unresolved",
    "reported_year",
    "year_derived",
    "topcoded",
    "rounded",
    "rounded_topcoded",
)

_REPORTED_YEAR = AGE_KINDS.index("reported_year")
_YEAR_DERIVED = AGE_KINDS.index("year_derived")
_SOURCE_AGE_UNRESOLVED = AGE_KINDS.index("source_age_unresolved")


def _validate_inputs(
    age_triplet: np.ndarray, age_kind: np.ndarray, adult: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and copy the three cache arrays without exposing row data."""

    if not isinstance(age_triplet, np.ndarray):
        raise TypeError("age_triplet must be a NumPy array")
    if age_triplet.ndim != 2 or age_triplet.shape[1] != 3:
        raise ValueError("age_triplet must have shape (n, 3)")
    if age_triplet.dtype.kind != "f":
        raise TypeError("age_triplet must have a floating dtype")

    if not isinstance(age_kind, np.ndarray):
        raise TypeError("age_kind must be a NumPy array")
    if age_kind.ndim != 1 or age_kind.shape[0] != age_triplet.shape[0]:
        raise ValueError("age_kind must have shape (n,)")
    if age_kind.dtype.kind not in "iu":
        raise TypeError("age_kind must have an integer dtype")

    if not isinstance(adult, np.ndarray):
        raise TypeError("adult must be a NumPy array")
    if adult.ndim != 1 or adult.shape[0] != age_triplet.shape[0]:
        raise ValueError("adult must have shape (n,)")
    if adult.dtype.kind != "b":
        raise TypeError("adult must have boolean dtype")

    # Validate the integer domain before narrowing to a convenient index type;
    # this avoids wrapping an oversized unsigned value.
    known_kind = (age_kind >= 0) & (age_kind < len(AGE_KINDS))
    if not bool(np.all(known_kind)):
        raise ValueError("age_kind contains an unknown code")

    triplet = age_triplet.astype(np.float64, copy=True)
    kind = age_kind.astype(np.int64, copy=True)
    adult_copy = adult.astype(np.bool_, copy=True)
    return triplet, kind, adult_copy


def eligible_native_age(
    age_triplet: np.ndarray, age_kind: np.ndarray, adult: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return eligible reported/derived scalar ages and an eligibility mask.

    ``age_triplet`` is ordered ``reported, lower, upper``.  Only finite,
    ordered ``reported_year`` and ``year_derived`` intervals with a lower
    bound of at least 18 and a consistent ``adult=True`` flag are admitted.
    A reported or year-derived scalar is not exact chronological age; the
    original interval and source precision remain part of its provenance.
    The returned arrays are fresh, read-only arrays; all other kinds receive
    ``NaN`` and ``False``.  Malformed intervals for candidate kinds raise
    instead of being silently enabled.
    """

    triplet, kind, adult_copy = _validate_inputs(age_triplet, age_kind, adult)
    lower = triplet[:, 1]

    expected_adult = (
        np.isfinite(lower)
        & (lower >= 18.0)
        & (kind != _SOURCE_AGE_UNRESOLVED)
    )
    if not np.array_equal(adult_copy, expected_adult):
        raise ValueError("adult flag is inconsistent with the age cache")

    candidate = (kind == _REPORTED_YEAR) | (kind == _YEAR_DERIVED)
    if bool(np.any(candidate)):
        candidate_triplet = triplet[candidate]
        reported = candidate_triplet[:, 0]
        candidate_lower = candidate_triplet[:, 1]
        upper = candidate_triplet[:, 2]
        finite = np.isfinite(candidate_triplet).all(axis=1)
        ordered = (candidate_lower <= reported) & (reported <= upper)
        if not bool(np.all(finite & ordered)):
            raise ValueError("candidate native-age interval is malformed")

    point_age = np.full(triplet.shape[0], np.nan, dtype=np.float64)
    eligible = np.zeros(triplet.shape[0], dtype=np.bool_)

    admitted = candidate & (lower >= 18.0) & adult_copy
    point_age[admitted] = triplet[admitted, 0]
    eligible[admitted] = True

    # The outputs own their storage and cannot mutate the private inputs or
    # be changed accidentally by a downstream caller.
    point_age.setflags(write=False)
    eligible.setflags(write=False)
    return point_age, eligible


__all__ = ["AGE_KINDS", "eligible_native_age"]
