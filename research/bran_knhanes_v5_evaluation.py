"""Private local four-arm KNHANES Hb prediction kernel.

Callers authenticate the source, fixed fold0 V5 state provenance, native-head
prediction, and grouped-fold assignment.  This module performs no I/O,
aggregation, calibration, source admission, or public release.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_knhanes_grouped_folds_v1 import GroupedKNHANESFolds
from bran_knhanes_input_kernel_v1 import CANONICAL_NAMES, ADMITTED_CANONICAL_INDICES


ARMS = ("native", "state_extratrees", "raw_extratrees", "training_median")
_ERROR = "knhanes_v5_evaluation_contract_failed"
_STATE_FLAGS = {"frozen_fold0_frame", "clinical_only", "whole_cbc_hidden", "state_provider_authenticated"}


def _require(ok: bool) -> None:
    if not ok:
        raise ValueError(_ERROR)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy(); result.setflags(write=False); return result


@dataclass(frozen=True, repr=False)
class PrivateKNHANESPredictions:
    """Per-row predictions for a later local disclosure wrapper only."""

    predictions: Mapping[str, np.ndarray]
    support: np.ndarray
    provider_flags: Mapping[str, bool]

    def __repr__(self) -> str:
        return "<PrivateKNHANESPredictions>"

    def __reduce__(self):
        raise TypeError(_ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(_ERROR)


def _matrix(value: Any, rows: int, columns: int, *, finite_supported: np.ndarray | None = None) -> np.ndarray:
    _require(type(value) is np.ndarray and value.shape == (rows, columns) and value.dtype.kind in "fiu")
    array = value.astype(np.float64, copy=False)
    if finite_supported is None:
        _require(bool(np.isfinite(array).all()))
    else:
        _require(bool(np.isfinite(array[finite_supported]).all()))
    return array


def _vector(value: Any, rows: int, *, finite: bool = False) -> np.ndarray:
    _require(type(value) is np.ndarray and value.shape == (rows,) and value.dtype.kind in "fiu")
    array = value.astype(np.float64, copy=False)
    if finite: _require(bool(np.isfinite(array).all()))
    return array


def _mask(value: Any, rows: int) -> np.ndarray:
    _require(type(value) is np.ndarray and value.shape == (rows,) and value.dtype == np.dtype(bool))
    return value


def _state_contract(value: Any) -> Mapping[str, bool]:
    _require(isinstance(value, Mapping) and set(value) == _STATE_FLAGS and all(type(value[key]) is bool for key in value))
    _require(all(value.values()))
    return MappingProxyType(dict(value))


def _raw_design(values: np.ndarray, observed: np.ndarray, eligible: np.ndarray, ages: np.ndarray,
                train: np.ndarray, cbc_slots: tuple[int, ...]) -> np.ndarray:
    visible = observed & eligible[None, :]
    visible = visible.copy(); visible[:, cbc_slots] = False
    clean = np.where(visible, values, np.nan)
    medians = np.zeros(59, dtype=np.float64)
    for column in range(59):
        present = clean[train, column]
        finite = present[np.isfinite(present)]
        if finite.size:
            medians[column] = float(np.median(finite))
    imputed = np.where(np.isfinite(clean), clean, medians[None, :])
    _require(bool(np.isfinite(imputed).all()))
    return np.c_[imputed, visible.astype(np.float64), ages]


def _regressor(fold: int, factory: Callable[..., Any] | None) -> Any:
    kwargs = {"n_estimators": 256, "min_samples_leaf": 5, "max_features": 1.0,
              "bootstrap": False, "n_jobs": 1, "random_state": 97201 + fold}
    return (ExtraTreesRegressor if factory is None else factory)(**kwargs)


def evaluate(grouped: Any, states: Any, clinical_values: Any, clinical_mask: Any, ages: Any,
             target: Any, target_observed: Any, physiological_available: Any, native_prediction: Any,
             names: Any, eligible_indices: Any, state_provider_flags: Any, *, regressor_factory: Callable[..., Any] | None = None) -> PrivateKNHANESPredictions:
    """Fit train-fold-only state/raw ExtraTrees heads on exactly shared support.

    ``states`` must already be computed from the authenticated frozen fold0,
    clinical-only, whole-CBC-hidden V5 path.  This pure kernel cannot prove
    that provenance, so callers must supply and authenticate the strict flags.
    """

    try:
        _require(isinstance(grouped, GroupedKNHANESFolds))
        folds = grouped.folds; rows = folds.shape[0]
        _require(folds.dtype.kind in "iu" and set(int(item) for item in np.unique(folds)) == set(range(5)))
        _require(isinstance(names, tuple) and names == CANONICAL_NAMES)
        _require(isinstance(eligible_indices, tuple) and len(eligible_indices) == 43
                 and all(type(item) is int for item in eligible_indices)
                 and set(eligible_indices) == set(ADMITTED_CANONICAL_INDICES))
        eligible = np.zeros(59, dtype=bool); eligible[list(eligible_indices)] = True
        cbc_slots = tuple(names.index(field) for field in CBC_FIELDS)
        _require(all(eligible[slot] for slot in cbc_slots))
        support = _mask(target_observed, rows) & _mask(physiological_available, rows)
        _require(bool(support.any()))
        state = _matrix(states, rows, 192, finite_supported=support)
        _require(type(clinical_values) is np.ndarray and clinical_values.shape == (rows, 59)
                 and clinical_values.dtype.kind in "fiu")
        values = clinical_values.astype(np.float64, copy=False)
        _require(type(clinical_mask) is np.ndarray and clinical_mask.shape == (rows, 59) and clinical_mask.dtype == np.dtype(bool))
        visible_non_cbc = clinical_mask & eligible[None, :]
        visible_non_cbc[:, cbc_slots] = False
        _require(bool(np.isfinite(values[visible_non_cbc]).all()))
        age = _vector(ages, rows)
        available = _mask(physiological_available, rows)
        # Keep out-of-domain survey PSUs, including rows with unknown age, in
        # the design. They cannot become train/test model observations.
        _require(bool(np.all(~available | (visible_non_cbc.any(axis=1) & np.isfinite(age) & (age >= 0)))))
        truth = _vector(target, rows)
        native = _vector(native_prediction, rows)
        _require(bool(np.isfinite(truth[support]).all()) and bool(np.isfinite(native[support]).all()))
        flags = _state_contract(state_provider_flags)
        predictions = {arm: np.full(rows, np.nan, dtype=np.float64) for arm in ARMS}
        predictions["native"][support] = native[support]
        for fold in range(5):
            train = (folds != fold) & support
            test = (folds == fold) & support
            _require(int(np.count_nonzero(train)) >= 20 and int(np.count_nonzero(test)) >= 1)
            state_model = _regressor(fold, regressor_factory)
            raw_model = _regressor(fold, regressor_factory)
            raw = _raw_design(values, clinical_mask, eligible, age, train, cbc_slots)
            state_model.fit(state[train], truth[train]); raw_model.fit(raw[train], truth[train])
            state_prediction = np.asarray(state_model.predict(state[test]), dtype=np.float64)
            raw_prediction = np.asarray(raw_model.predict(raw[test]), dtype=np.float64)
            _require(state_prediction.shape == raw_prediction.shape == (int(np.count_nonzero(test)),)
                     and bool(np.isfinite(state_prediction).all()) and bool(np.isfinite(raw_prediction).all()))
            predictions["state_extratrees"][test] = state_prediction
            predictions["raw_extratrees"][test] = raw_prediction
            predictions["training_median"][test] = float(np.median(truth[train]))
        _require(all(bool(np.isfinite(value[support]).all()) and bool(np.isnan(value[~support]).all())
                     for value in predictions.values()))
        return PrivateKNHANESPredictions(MappingProxyType({key: _readonly(value) for key, value in predictions.items()}),
                                         _readonly(support), flags)
    except (TypeError, ValueError, IndexError, KeyError, AttributeError):
        raise ValueError(_ERROR) from None
