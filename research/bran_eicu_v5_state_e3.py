"""Pure E2-to-V5 clinical state adapter for the bounded E3 path.

This module accepts only the E2 21-slot physiology and typed-age arrays.  It
performs no source admission, label handling, I/O, normalization fitting, or
serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import bran_eicu_chemistry_adapter_e2 as chemistry
import bran_nhanes_v5_state as v5_state


ERROR = "eicu_v5_state_e3_contract_failed"
FIELDS = tuple(chemistry.FIELDS)
PHYSIOLOGY_WIDTH = 21
STATE_WIDTH = 192
CLINICAL_WIDTH = 59
E2_UNKNOWN = 0
E2_REPORTED = 1
E2_RIGHT_CENSORED = 2


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError(ERROR)


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    copied = np.array(value, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, repr=False)
class PrivateEicuV5StateE3:
    """Private, non-serializable state result with read-only array fields."""

    state: np.ndarray
    available: np.ndarray
    clinical: np.ndarray
    clinical_mask: np.ndarray

    def __post_init__(self) -> None:
        try:
            state = np.asarray(self.state)
            available = np.asarray(self.available)
            clinical = np.asarray(self.clinical)
            clinical_mask = np.asarray(self.clinical_mask)
            _require(
                state.ndim == 2
                and state.shape[1] == STATE_WIDTH
                and state.dtype == np.float32
                and np.isfinite(state).all()
            )
            n = state.shape[0]
            _require(
                available.shape == (n,)
                and available.dtype == np.bool_
                and clinical.shape == (n, CLINICAL_WIDTH)
                and clinical.dtype == np.float64
                and np.isfinite(clinical).all()
                and clinical_mask.shape == (n, CLINICAL_WIDTH)
                and clinical_mask.dtype == np.bool_
                and (clinical[~clinical_mask] == 0).all()
                and np.array_equal(available, clinical_mask.any(axis=1))
                and (state[~available] == 0).all()
            )
            object.__setattr__(self, "state", _readonly_copy(state))
            object.__setattr__(self, "available", _readonly_copy(available))
            object.__setattr__(self, "clinical", _readonly_copy(clinical))
            object.__setattr__(self, "clinical_mask", _readonly_copy(clinical_mask))
        except Exception:
            raise ValueError(ERROR) from None

    def __repr__(self) -> str:
        return "<PrivateEicuV5StateE3>"

    def __reduce__(self):
        raise TypeError("eicu_v5_state_e3_serialization_forbidden")


def _validate_inputs(
    values: object,
    observed: object,
    age_triplet: object,
    age_kind: object,
    batch_size: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    _require(type(values) is np.ndarray and values.ndim == 2)
    _require(
        values.shape[1] == PHYSIOLOGY_WIDTH
        and len(FIELDS) == PHYSIOLOGY_WIDTH
        and len(set(FIELDS)) == PHYSIOLOGY_WIDTH
        and tuple(FIELDS) == tuple(chemistry.FIELDS)
    )
    _require(values.dtype == np.float64)
    _require(type(observed) is np.ndarray and observed.shape == values.shape)
    _require(observed.dtype == np.bool_)
    _require(type(age_triplet) is np.ndarray and age_triplet.shape == (len(values), 3))
    _require(age_triplet.dtype == np.float64)
    _require(type(age_kind) is np.ndarray and age_kind.shape == (len(values),))
    _require(age_kind.dtype == np.uint8 and np.isin(age_kind, (0, 1, 2)).all())
    _require(
        isinstance(batch_size, (int, np.integer))
        and not isinstance(batch_size, (bool, np.bool_))
        and 1 <= int(batch_size) <= 4096
    )

    observed_values = values[observed]
    _require(np.isfinite(observed_values).all())
    _require((values[:, :9][observed[:, :9]] > 0).all())
    _require((values[:, 9:][observed[:, 9:]] >= 0).all())
    _require(not observed[:, FIELDS.index("protein_total")].any())

    unknown = age_kind == E2_UNKNOWN
    reported = age_kind == E2_REPORTED
    censored = age_kind == E2_RIGHT_CENSORED
    _require(np.isnan(age_triplet[unknown]).all())
    reported_values = age_triplet[reported]
    _require(
        np.isfinite(reported_values).all()
        and (reported_values[:, 0] >= 0).all()
        and (reported_values[:, 1] == reported_values[:, 0]).all()
        and (reported_values[:, 2] == reported_values[:, 0] + 1).all()
    )
    censored_values = age_triplet[censored]
    _require(
        np.isnan(censored_values[:, 0]).all()
        and np.isfinite(censored_values[:, 1]).all()
        and (censored_values[:, 1] == 90).all()
        and np.isposinf(censored_values[:, 2]).all()
    )
    return (
        values,
        observed,
        age_triplet,
        age_kind,
        int(batch_size),
    )


def _typed_age_arrays(age_triplet: np.ndarray, age_kind: np.ndarray) -> dict[str, np.ndarray]:
    n = len(age_kind)
    value = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    upper = np.full(n, np.nan, dtype=np.float64)
    internal_kind = np.full(n, 3, dtype=np.int64)
    reported = age_kind == E2_REPORTED
    censored = age_kind == E2_RIGHT_CENSORED
    value[reported] = age_triplet[reported, 0]
    internal_kind[reported] = 0
    lower[censored] = age_triplet[censored, 1]
    internal_kind[censored] = 2
    return {
        "age_value": value,
        "age_lower": lower,
        "age_upper": upper,
        "age_kind": internal_kind,
    }


def _base_arrays(
    values: np.ndarray,
    observed: np.ndarray,
    age_triplet: np.ndarray,
    age_kind: np.ndarray,
) -> dict[str, np.ndarray]:
    sanitized_values = np.array(values, copy=True)
    # The E2 storage contract uses zero for masked slots, while the existing
    # V5 projection contract requires NaN for an unobserved original value.
    sanitized_values[~observed] = np.nan
    arrays = {
        "values": sanitized_values,
        "observed": np.array(observed, copy=True),
    }
    arrays.update(_typed_age_arrays(age_triplet, age_kind))
    return arrays


def _run_chunk(
    values: np.ndarray,
    observed: np.ndarray,
    age_triplet: np.ndarray,
    age_kind: np.ndarray,
    model: object,
    transform: object,
    expected_transform_sha256: object,
) -> PrivateEicuV5StateE3:
    result = v5_state.infer(
        _base_arrays(values, observed, age_triplet, age_kind),
        model,
        transform,
        expected_transform_sha256,
    )
    return PrivateEicuV5StateE3(
        result.state,
        result.available,
        result.clinical,
        result.clinical_mask,
    )


def infer(
    values: np.ndarray,
    observed: np.ndarray,
    age_triplet: np.ndarray,
    age_kind: np.ndarray,
    model: object,
    transform: object,
    expected_transform_sha256: str,
    *,
    batch_size: int = 256,
) -> PrivateEicuV5StateE3:
    """Encode bounded E2 physiology and typed age through the frozen V5 state."""
    try:
        values, observed, age_triplet, age_kind, batch_size = _validate_inputs(
            values,
            observed,
            age_triplet,
            age_kind,
            batch_size,
        )
        n = len(values)

        if n == 0:
            # The V5 route is not required to concatenate an empty torch batch.
            # A one-row masked probe still authenticates model/transform identity
            # through the existing V5 provider and route checks.
            probe_values = np.zeros((1, PHYSIOLOGY_WIDTH), dtype=np.float64)
            probe_observed = np.zeros((1, PHYSIOLOGY_WIDTH), dtype=np.bool_)
            probe_age = np.full((1, 3), np.nan, dtype=np.float64)
            probe_kind = np.zeros(1, dtype=np.uint8)
            _run_chunk(
                probe_values,
                probe_observed,
                probe_age,
                probe_kind,
                model,
                transform,
                expected_transform_sha256,
            )
            return PrivateEicuV5StateE3(
                np.empty((0, STATE_WIDTH), dtype=np.float32),
                np.empty((0,), dtype=np.bool_),
                np.empty((0, CLINICAL_WIDTH), dtype=np.float64),
                np.empty((0, CLINICAL_WIDTH), dtype=np.bool_),
            )

        chunks = []
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            chunks.append(
                _run_chunk(
                    values[start:stop],
                    observed[start:stop],
                    age_triplet[start:stop],
                    age_kind[start:stop],
                    model,
                    transform,
                    expected_transform_sha256,
                )
            )
        return PrivateEicuV5StateE3(
            np.concatenate([chunk.state for chunk in chunks], axis=0),
            np.concatenate([chunk.available for chunk in chunks], axis=0),
            np.concatenate([chunk.clinical for chunk in chunks], axis=0),
            np.concatenate([chunk.clinical_mask for chunk in chunks], axis=0),
        )
    except Exception:
        raise ValueError(ERROR) from None
