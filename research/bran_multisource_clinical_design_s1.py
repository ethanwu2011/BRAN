"""Pure discovery-fitted clinical design for the pooled multisource S1 study.

The caller supplies already authenticated original-unit clinical values and
typed age components.  This module performs no source I/O, outcome handling,
identity handling, model fitting, or clustering.  Only role ``0`` rows fit
imputation, age normalization, and the two independent standardizers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from bran_multisource_age_v2 import INTERVAL, REPORTED, RIGHT_CENSORED, UNKNOWN


ERROR = "bran_multisource_clinical_design_s1_contract_failed"
LAB_COUNT = 21
AGE_FEATURE_COUNT = 7
RAW_FEATURE_COUNT = 49
STATE_FEATURE_COUNT = 192
CONTEXT_FEATURE_COUNT = 59
NUISANCE_FEATURE_COUNT = 30
ROLE_DISCOVERY = 0
ROLE_CHARACTERIZATION_DEVELOPMENT = 1
ROLE_TEST = 2
SOURCE_COUNT = 2
AGE_KINDS = (REPORTED, INTERVAL, RIGHT_CENSORED, UNKNOWN)


def _fail() -> None:
    raise ValueError(ERROR) from None


def _require(condition: bool) -> None:
    if not condition:
        _fail()


def _readonly(value: object, dtype: np.dtype | str | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _array(value: object, dtype: np.dtype | str, shape: tuple[int, ...]) -> np.ndarray:
    _require(type(value) is np.ndarray and value.dtype == np.dtype(dtype) and value.shape == shape)
    return value


def _validate_inputs(
    values: object,
    observed: object,
    age_value: object,
    age_lower: object,
    age_upper: object,
    age_kind: object,
    state: object,
    roles: object,
    source: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    _require(type(values) is np.ndarray and values.dtype == np.dtype(np.float64)
             and values.ndim == 2 and values.shape[1] == LAB_COUNT)
    rows = values.shape[0]
    _require(rows > 0)
    _require(type(observed) is np.ndarray and observed.dtype == np.dtype(bool)
             and observed.shape == values.shape)
    _require(np.isfinite(values[observed]).all())
    _array(age_value, np.float64, (rows,))
    _array(age_lower, np.float64, (rows,))
    _array(age_upper, np.float64, (rows,))
    _array(age_kind, np.int64, (rows,))
    _array(state, state.dtype, (rows, STATE_FEATURE_COUNT))
    _require(state.dtype.kind == "f" and np.isfinite(state).all())
    _array(roles, np.uint8, (rows,))
    _array(source, np.uint8, (rows,))
    _require(np.isin(roles, (ROLE_DISCOVERY, ROLE_CHARACTERIZATION_DEVELOPMENT, ROLE_TEST)).all())
    _require(np.isin(source, (0, 1)).all())
    for role in (ROLE_DISCOVERY, ROLE_CHARACTERIZATION_DEVELOPMENT, ROLE_TEST):
        _require(bool(np.any(roles == role)))
    _require(np.isin(age_kind, AGE_KINDS).all())

    # Inactive typed-age components must be absent rather than silently
    # interpreted as an exact age.  Active components retain original units.
    reported = age_kind == REPORTED
    interval = age_kind == INTERVAL
    censored = age_kind == RIGHT_CENSORED
    unknown = age_kind == UNKNOWN
    _require(np.isfinite(age_value[reported]).all() and (age_value[reported] >= 0).all())
    _require(np.isnan(age_lower[reported]).all() and np.isnan(age_upper[reported]).all())
    _require(np.isnan(age_value[interval]).all())
    _require(np.isfinite(age_lower[interval]).all() and np.isfinite(age_upper[interval]).all())
    _require((age_lower[interval] >= 0).all() and (age_upper[interval] >= age_lower[interval]).all())
    _require(np.isnan(age_value[censored]).all() and np.isfinite(age_lower[censored]).all())
    _require((age_lower[censored] >= 0).all() and np.isnan(age_upper[censored]).all())
    _require(np.isnan(age_value[unknown]).all() and np.isnan(age_lower[unknown]).all()
             and np.isnan(age_upper[unknown]).all())
    return (
        np.array(values, dtype=np.float64, copy=True),
        np.array(observed, dtype=bool, copy=True),
        np.array(age_value, dtype=np.float64, copy=True),
        np.array(age_lower, dtype=np.float64, copy=True),
        np.array(age_upper, dtype=np.float64, copy=True),
        np.array(age_kind, dtype=np.int64, copy=True),
        np.array(state, dtype=np.float64, copy=True),
        rows,
    )


def _age_features(
    age_value: np.ndarray,
    age_lower: np.ndarray,
    age_upper: np.ndarray,
    age_kind: np.ndarray,
    discovery: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    numeric = np.column_stack((age_value, age_lower, age_upper))
    active = np.isfinite(numeric)
    discovery_numeric = numeric[discovery]
    discovery_active = active[discovery]
    finite = discovery_numeric[discovery_active]
    mean = float(np.mean(finite)) if finite.size else 0.0
    scale = float(np.std(finite)) if finite.size else 1.0
    if scale <= 0.0:
        scale = 1.0
    normalized = np.zeros_like(numeric, dtype=np.float64)
    for column in range(3):
        selected = active[:, column]
        normalized[selected, column] = (numeric[selected, column] - mean) / scale
    one_hot = np.eye(4, dtype=np.float64)[age_kind]
    features = np.concatenate((normalized, one_hot), axis=1)
    _require(features.shape == (len(age_kind), AGE_FEATURE_COUNT) and np.isfinite(features).all())
    return features, mean, scale


def _fit_scale(values: np.ndarray, discovery: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        scaler = StandardScaler().fit(values[discovery])
        transformed = np.asarray(scaler.transform(values), dtype=np.float64)
        mean = np.asarray(scaler.mean_, dtype=np.float64)
        scale = np.asarray(scaler.scale_, dtype=np.float64)
    except Exception:
        _fail()
    _require(np.isfinite(transformed).all() and np.isfinite(mean).all()
             and np.isfinite(scale).all() and (scale > 0).all())
    return transformed, _readonly(mean, np.float64), _readonly(scale, np.float64)


@dataclass(frozen=True, slots=True, repr=False)
class PrivateMultisourceClinicalDesignS1:
    """Readonly private design matrices and discovery-fitted replay statistics."""

    imputed_values: np.ndarray
    age_features: np.ndarray
    raw_design: np.ndarray
    raw_scaled: np.ndarray
    raw_padded192: np.ndarray
    context59: np.ndarray
    nuisance_design: np.ndarray
    state_scaled: np.ndarray
    lab_medians: np.ndarray
    age_mean: float
    age_scale: float
    raw_scaler_mean: np.ndarray
    raw_scaler_scale: np.ndarray
    state_scaler_mean: np.ndarray
    state_scaler_scale: np.ndarray

    def __repr__(self) -> str:
        return "<PrivateMultisourceClinicalDesignS1>"

    @property
    def state_design(self) -> np.ndarray:
        return self.state_scaled

    @property
    def raw_clustering_states(self) -> np.ndarray:
        return self.raw_padded192

    @property
    def stats(self) -> dict[str, object]:
        """Discovery-fitted replay statistics; no source or row metadata."""
        return {
            "lab_medians": self.lab_medians,
            "age_mean": self.age_mean,
            "age_scale": self.age_scale,
            "raw_scaler_mean": self.raw_scaler_mean,
            "raw_scaler_scale": self.raw_scaler_scale,
            "state_scaler_mean": self.state_scaler_mean,
            "state_scaler_scale": self.state_scaler_scale,
        }

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)


def build(
    values: np.ndarray,
    observed: np.ndarray,
    age_value: np.ndarray,
    age_lower: np.ndarray,
    age_upper: np.ndarray,
    age_kind: np.ndarray,
    state: np.ndarray,
    roles: np.ndarray,
    source: np.ndarray,
) -> PrivateMultisourceClinicalDesignS1:
    """Fit and transform the fixed S1 clinical/control representations."""
    try:
        (
            lab_values, lab_observed, typed_value, typed_lower, typed_upper,
            typed_kind, state_values, rows,
        ) = _validate_inputs(
            values, observed, age_value, age_lower, age_upper, age_kind,
            state, roles, source,
        )
        discovery = roles == ROLE_DISCOVERY
        age_features, age_mean, age_scale = _age_features(
            typed_value, typed_lower, typed_upper, typed_kind, discovery,
        )

        lab_medians = np.zeros(LAB_COUNT, dtype=np.float64)
        for column in range(LAB_COUNT):
            present = lab_values[discovery, column][lab_observed[discovery, column]]
            if present.size:
                lab_medians[column] = float(np.median(present))
        imputed = np.where(lab_observed, lab_values, lab_medians[None, :])
        missingness = (~lab_observed).astype(np.float64)
        raw_design = np.concatenate((imputed, missingness, age_features), axis=1)
        _require(raw_design.shape == (rows, RAW_FEATURE_COUNT) and np.isfinite(raw_design).all())

        raw_scaled, raw_mean, raw_scale = _fit_scale(raw_design, discovery)
        state_scaled, state_mean, state_scale = _fit_scale(state_values, discovery)
        raw_padded = np.zeros((rows, STATE_FEATURE_COUNT), dtype=np.float64)
        raw_padded[:, :RAW_FEATURE_COUNT] = raw_scaled

        source_one_hot = np.eye(SOURCE_COUNT, dtype=np.float64)[source]
        context = np.concatenate((raw_scaled, source_one_hot, np.zeros((rows, 8), dtype=np.float64)), axis=1)
        nuisance = np.concatenate((missingness, age_features, source_one_hot), axis=1)
        _require(context.shape == (rows, CONTEXT_FEATURE_COUNT)
                 and nuisance.shape == (rows, NUISANCE_FEATURE_COUNT)
                 and np.isfinite(context).all() and np.isfinite(nuisance).all())
        return PrivateMultisourceClinicalDesignS1(
            _readonly(imputed, np.float64),
            _readonly(age_features, np.float64),
            _readonly(raw_design, np.float64),
            _readonly(raw_scaled, np.float64),
            _readonly(raw_padded, np.float64),
            _readonly(context, np.float64),
            _readonly(nuisance, np.float64),
            _readonly(state_scaled, np.float64),
            _readonly(lab_medians, np.float64),
            float(age_mean),
            float(age_scale),
            raw_mean,
            raw_scale,
            state_mean,
            state_scale,
        )
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def fit_transform(*args, **kwargs) -> PrivateMultisourceClinicalDesignS1:
    """Alias retained for callers that use the fixed-frame naming convention."""
    return build(*args, **kwargs)


__all__ = [
    "AGE_FEATURE_COUNT", "AGE_KINDS", "CONTEXT_FEATURE_COUNT", "ERROR", "LAB_COUNT",
    "NUISANCE_FEATURE_COUNT", "PrivateMultisourceClinicalDesignS1", "RAW_FEATURE_COUNT",
    "STATE_FEATURE_COUNT", "build", "fit_transform",
]
