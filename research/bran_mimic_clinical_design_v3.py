"""Pure discovery-fitted raw/state design controls for the MIMIC V3 frame.

The caller supplies authenticated original-unit 21-field values/masks, packed
MIMIC age triplets/kinds, state vectors, and a complete disjoint role partition.
Only discovery rows fit lab medians, typed-age normalization, and the separate
raw/state standardizers.  No outcomes, disease labels, source identifiers,
files, clustering, or model fitting are accepted here.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

import bran_joint_lab_cache_v1 as _cache
from bran_mimic_selected_state_v3 import _typed_age_arrays
from bran_multisource_age_v2 import INTERVAL, RIGHT_CENSORED, UNKNOWN


ERROR = "bran_mimic_clinical_design_v3_contract_failed"
LAB_COUNT = 21
AGE_FEATURE_COUNT = 7
RAW_FEATURE_COUNT = 49
STATE_FEATURE_COUNT = 192
ROLE_NAMES = ("discovery", "characterization_development", "test")
_EXPECTED_PACKED_AGE_KINDS = {"missing_or_invalid", "year_derived", "topcoded"}


def _fail() -> None:
    raise ValueError(ERROR) from None


def _require(condition: bool) -> None:
    if not condition:
        _fail()


def _readonly(value: object, dtype: np.dtype | str | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, repr=False)
class PrivateMimicClinicalDesignV3:
    """Private fitted design and standardized controls.

    ``raw_design`` is the finite 49-column original-unit/imputed design:
    21 imputed lab values, 21 missingness indicators (one when the source
    field is hidden), and seven typed-age features.  ``raw_scaled`` is the
    discovery-fitted logistic design;
    ``raw_padded192`` right-pads that design with zeros for a retained 192-D
    raw-clustering control.  ``state_scaled`` uses an independent discovery
    standardizer fit only on the supplied state rows.
    """

    imputed_values: np.ndarray
    age_features: np.ndarray
    raw_design: np.ndarray
    raw_scaled: np.ndarray
    raw_padded192: np.ndarray
    state_scaled: np.ndarray
    lab_medians: np.ndarray
    age_mean: float
    age_scale: float
    raw_scaler_mean: np.ndarray
    raw_scaler_scale: np.ndarray
    state_scaler_mean: np.ndarray
    state_scaler_scale: np.ndarray

    def __repr__(self) -> str:
        return "<PrivateMimicClinicalDesignV3>"

    @property
    def state_design(self) -> np.ndarray:
        """Discovery-standardized 192-D state control (caller-owned view)."""
        return self.state_scaled

    @property
    def raw_clustering_states(self) -> np.ndarray:
        """Discovery-standardized raw design, zero-padded to 192 columns."""
        return self.raw_padded192

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)


def _validate_role_indices(role_indices: object, rows: int) -> dict[str, np.ndarray]:
    _require(isinstance(role_indices, Mapping) and set(role_indices) == set(ROLE_NAMES))
    result: dict[str, np.ndarray] = {}
    all_rows: list[np.ndarray] = []
    for name in ROLE_NAMES:
        value = role_indices[name]
        _require(type(value) is np.ndarray and value.dtype == np.dtype(np.int64)
                 and value.ndim == 1 and len(value) > 0)
        _require(np.all((value >= 0) & (value < rows)) and len(np.unique(value)) == len(value))
        result[name] = np.array(value, dtype=np.int64, copy=True)
        all_rows.append(result[name])
    combined = np.concatenate(all_rows)
    _require(len(np.unique(combined)) == len(combined) == rows)
    _require(np.array_equal(np.sort(combined), np.arange(rows, dtype=np.int64)))
    return result


def _validate_labs(values: object, observed: object) -> tuple[np.ndarray, np.ndarray, int]:
    _require(type(values) is np.ndarray and values.dtype == np.dtype(np.float64)
             and values.ndim == 2 and values.shape[1] == LAB_COUNT)
    rows = values.shape[0]
    _require(rows > 0)
    _require(type(observed) is np.ndarray and observed.dtype == np.dtype(bool)
             and observed.shape == values.shape)
    _require(np.isfinite(values[observed]).all())
    return np.array(values, dtype=np.float64, copy=True), np.array(observed, dtype=bool, copy=True), rows


def _validate_state(state: object, rows: int) -> np.ndarray:
    _require(type(state) is np.ndarray and state.ndim == 2 and state.shape == (rows, STATE_FEATURE_COUNT)
             and state.dtype.kind == "f")
    _require(np.isfinite(state).all())
    return np.array(state, dtype=np.float64, copy=True)


def _typed_age_features(age_triplet: object, age_kind: object, discovery_rows: np.ndarray) -> tuple[np.ndarray, float, float]:
    _require(type(age_triplet) is np.ndarray and age_triplet.dtype == np.dtype(np.float64)
             and age_triplet.ndim == 2 and age_triplet.shape[1] == 3 and age_triplet.shape[0] > 0)
    rows = age_triplet.shape[0]
    _require(type(age_kind) is np.ndarray and age_kind.ndim == 1 and age_kind.shape == (rows,)
             and age_kind.dtype.kind in "iu")
    packed = np.array(age_kind, dtype=np.int64, copy=True)
    packed_names = tuple(_cache.AGE_KINDS)
    _require(np.all((packed >= 0) & (packed < len(packed_names))))
    for index, name in enumerate(packed_names):
        selected = packed == index
        if not np.any(selected):
            continue
        _require(name in _EXPECTED_PACKED_AGE_KINDS)
        triplet = age_triplet[selected]
        if name == "missing_or_invalid":
            _require(np.isnan(triplet).all())
        elif name == "year_derived":
            _require(np.isfinite(triplet).all())
            _require(np.equal(triplet[:, 0], np.floor(triplet[:, 0])).all())
            _require((triplet[:, 0] >= 0).all() and (triplet[:, 1] >= 0).all() and (triplet[:, 2] >= 0).all())
            _require(np.array_equal(triplet[:, 1], np.maximum(0.0, triplet[:, 0] - 1.0)))
            _require(np.array_equal(triplet[:, 2], triplet[:, 0] + 1.0))
        elif name == "topcoded":
            _require(np.isnan(triplet[:, 0]).all() and np.isfinite(triplet[:, 1]).all())
            _require((triplet[:, 1] >= 0).all() and np.isposinf(triplet[:, 2]).all())

    typed = _typed_age_arrays({"age_triplet": np.array(age_triplet, copy=True), "age_kind": packed})
    numeric = np.column_stack((typed["age_value"], typed["age_lower"], typed["age_upper"]))
    kind = np.asarray(typed["age_kind"], dtype=np.int64)
    _require(np.isin(kind, (INTERVAL, RIGHT_CENSORED, UNKNOWN)).all())
    active = np.isfinite(numeric)
    discovery_numeric = numeric[discovery_rows]
    discovery_active = active[discovery_rows]
    finite_values = discovery_numeric[discovery_active]
    if len(finite_values):
        age_mean = float(np.mean(finite_values))
        age_scale = float(np.std(finite_values))
        if age_scale <= 0.0:
            age_scale = 1.0
    else:
        age_mean, age_scale = 0.0, 1.0
    normalized = np.zeros_like(numeric, dtype=np.float64)
    normalized[active] = (numeric[active] - age_mean) / age_scale
    indicators = np.eye(4, dtype=np.float64)[kind - 0]
    age_features = np.concatenate((normalized, indicators), axis=1)
    _require(age_features.shape == (rows, AGE_FEATURE_COUNT) and np.isfinite(age_features).all())
    return age_features, age_mean, age_scale


def _fit_scale(values: np.ndarray, discovery_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(values[discovery_rows])
    transformed = np.asarray(scaler.transform(values), dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    _require(np.isfinite(transformed).all() and np.isfinite(mean).all() and np.isfinite(scale).all()
             and (scale > 0).all())
    return transformed, _readonly(mean, np.float64), _readonly(scale, np.float64)


def build_clinical_design(
    values: np.ndarray,
    observed: np.ndarray,
    age_triplet: np.ndarray,
    age_kind: np.ndarray,
    state: np.ndarray,
    role_indices: Mapping[str, np.ndarray],
) -> PrivateMimicClinicalDesignV3:
    """Fit discovery-only raw/state controls and transform every supplied row."""
    try:
        lab_values, lab_observed, rows = _validate_labs(values, observed)
        state_values = _validate_state(state, rows)
        roles = _validate_role_indices(role_indices, rows)
        age_features, age_mean, age_scale = _typed_age_features(age_triplet, age_kind, roles["discovery"])

        lab_medians = np.zeros(LAB_COUNT, dtype=np.float64)
        discovery_values = lab_values[roles["discovery"]]
        discovery_observed = lab_observed[roles["discovery"]]
        for column in range(LAB_COUNT):
            present = discovery_values[:, column][discovery_observed[:, column]]
            if len(present):
                lab_medians[column] = float(np.median(present))
        imputed = np.where(lab_observed, lab_values, lab_medians[None, :])
        missingness = (~lab_observed).astype(np.float64)
        raw_design = np.concatenate((imputed, missingness, age_features), axis=1)
        _require(raw_design.shape == (rows, RAW_FEATURE_COUNT) and np.isfinite(raw_design).all())

        raw_scaled, raw_mean, raw_scale = _fit_scale(raw_design, roles["discovery"])
        state_scaled, state_mean, state_scale = _fit_scale(state_values, roles["discovery"])
        raw_padded = np.zeros((rows, STATE_FEATURE_COUNT), dtype=np.float64)
        raw_padded[:, :RAW_FEATURE_COUNT] = raw_scaled
        _require(np.isfinite(raw_padded).all())
        return PrivateMimicClinicalDesignV3(
            _readonly(imputed, np.float64),
            _readonly(age_features, np.float64),
            _readonly(raw_design, np.float64),
            _readonly(raw_scaled, np.float64),
            _readonly(raw_padded, np.float64),
            _readonly(state_scaled, np.float64),
            _readonly(lab_medians, np.float64),
            age_mean,
            age_scale,
            raw_mean,
            raw_scale,
            state_mean,
            state_scale,
        )
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def fit_transform(
    values21: np.ndarray,
    observed21: np.ndarray,
    age_triplet: np.ndarray,
    age_kind: np.ndarray,
    states192: np.ndarray,
    roles: Mapping[str, np.ndarray],
) -> PrivateMimicClinicalDesignV3:
    """Named integration entry point for the fixed logistic/control frame."""
    return build_clinical_design(values21, observed21, age_triplet, age_kind, states192, roles)


__all__ = [
    "AGE_FEATURE_COUNT", "ERROR", "LAB_COUNT", "PrivateMimicClinicalDesignV3",
    "RAW_FEATURE_COUNT", "ROLE_NAMES", "STATE_FEATURE_COUNT", "build_clinical_design",
    "fit_transform",
]
