"""Private, array-only CBC prediction-interval calibration for BRAN V2.

The caller supplies target-erased native CBC predictions and an authenticated
within-fold role vector.  This module neither loads sources nor runs models;
``observed_by_pattern`` is the final target/scoring support from completion
inference, not a request to derive support from clinical payloads.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, NoReturn

import numpy as np

from bran_native_cbc_calibration_metrics_v1 import fit_radii


_ERROR = "multisource CBC calibration inputs invalid"
_N_FIELDS = 9
_ARMS = ("mlp", "token")
_PATTERNS = (
    "single_target_hidden",
    "whole_cbc_hidden",
    "red_cell_hidden",
    "single_target_no_retina",
    "whole_cbc_no_retina",
    "red_cell_no_retina",
)


@dataclass(frozen=True, repr=False)
class CBCCalibrationV2:
    """Read-only calibration radii and scoring-only prediction intervals.

    ``radii`` is indexed ``pattern -> fold -> arm``.  ``lower`` and ``upper``
    are indexed ``pattern -> arm`` and contain ``[N, 9]`` arrays.  All arrays
    are private copies, write-protected, and NaN outside supported scoring
    rows (including fields without enough calibration support).
    """

    radii: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]]
    lower: Mapping[str, Mapping[str, np.ndarray]]
    upper: Mapping[str, Mapping[str, np.ndarray]]


def _fail() -> NoReturn:
    raise ValueError(_ERROR)


def _matrix(value: Any, rows: int | None = None) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1] != _N_FIELDS:
        _fail()
    if rows is not None and value.shape[0] != rows:
        _fail()
    if value.dtype.kind not in "fiu" or value.dtype.kind == "b":
        _fail()
    try:
        return value.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError):
        _fail()


def _mask(value: Any, rows: int) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.dtype != np.dtype(bool)
            or value.shape != (rows, _N_FIELDS)):
        _fail()
    return value


def _vector(value: Any, rows: int, *, roles: bool = False) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.ndim != 1 or value.shape[0] != rows
            or value.dtype.kind not in "iu" or value.dtype.kind == "b"):
        _fail()
    try:
        result = value.astype(np.int64, copy=False)
        if roles:
            if bool(np.any((result != 0) & (result != 1))):
                _fail()
        elif bool(np.any(result < 0)):
            _fail()
        return result
    except (TypeError, ValueError, OverflowError):
        _fail()


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _exact_mapping(value: Any, keys: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != frozenset(keys):
        _fail()
    return value


def calibrate(
    target: Any,
    observed_by_pattern: Any,
    predictions_by_pattern_by_arm: Any,
    folds: Any,
    roles: Any,
) -> CBCCalibrationV2:
    """Fit fixed 90% CBC radii on calibration roles and bound scoring roles.

    ``roles`` must be the caller-authenticated deterministic split: ``0`` is
    calibration and ``1`` is scoring.  It must retain all five historical
    folds (at least 40 rows each) and their exact floor-half role partitions.
    Inputs outside an observed completion
    target are deliberately ignored, so hidden source payload cannot enter a
    radius or interval.  Conversely, an observed target must have finite truth
    and finite predictions for *both* arms; arm-specific support is rejected.
    """

    truth = _matrix(target)
    rows = truth.shape[0]
    fold_values = _vector(folds, rows)
    role_values = _vector(roles, rows, roles=True)
    observed_map = _exact_mapping(observed_by_pattern, _PATTERNS)
    prediction_map = _exact_mapping(predictions_by_pattern_by_arm, _PATTERNS)

    try:
        if (rows == 0 or bool(np.any(fold_values > 4))
                or set(int(item) for item in np.unique(fold_values)) != set(range(5))):
            _fail()
        # The caller authenticates IDs and their role hash; this enforces the
        # non-negotiable public structure made by the historical split helper.
        for fold in range(5):
            count = int(np.count_nonzero(fold_values == fold))
            calibration_count = int(np.count_nonzero((fold_values == fold) & (role_values == 0)))
            scoring_count = int(np.count_nonzero((fold_values == fold) & (role_values == 1)))
            if count < 40 or calibration_count != count // 2 or scoring_count != count - count // 2:
                _fail()
    except Exception:
        _fail()

    pattern_masks: dict[str, np.ndarray] = {}
    pattern_predictions: dict[str, dict[str, np.ndarray]] = {}
    for pattern in _PATTERNS:
        observed = _mask(observed_map[pattern], rows)
        arms = _exact_mapping(prediction_map[pattern], _ARMS)
        predictions = {arm: _matrix(arms[arm], rows) for arm in _ARMS}
        try:
            # These masks are the completion adapter's final, target-erased
            # support.  Every supported value must be valid for both arms.
            if bool(np.any(observed & ~np.isfinite(truth))):
                _fail()
            finite_mlp = np.isfinite(predictions["mlp"])
            finite_token = np.isfinite(predictions["token"])
            if bool(np.any(observed & ~finite_mlp)) or bool(np.any(observed & ~finite_token)):
                _fail()
            if not np.array_equal(observed & finite_mlp, observed & finite_token):
                _fail()
        except (TypeError, ValueError, FloatingPointError):
            _fail()
        pattern_masks[pattern] = observed
        pattern_predictions[pattern] = predictions

    unique_folds = tuple(int(item) for item in np.unique(fold_values))
    radii_out: dict[str, Mapping[int, Mapping[str, np.ndarray]]] = {}
    lower_out: dict[str, Mapping[str, np.ndarray]] = {}
    upper_out: dict[str, Mapping[str, np.ndarray]] = {}
    for pattern in _PATTERNS:
        observed = pattern_masks[pattern]
        # fit_radii deliberately rejects an infinity anywhere in its matrices.
        # Replace inaccessible payload before passing it over that V1 boundary.
        clean_truth = np.where(observed, truth, 0.0)
        by_fold: dict[int, Mapping[str, np.ndarray]] = {}
        lower_arms: dict[str, np.ndarray] = {}
        upper_arms: dict[str, np.ndarray] = {}
        for arm in _ARMS:
            clean_prediction = np.where(observed, pattern_predictions[pattern][arm], 0.0)
            lower = np.full((rows, _N_FIELDS), np.nan, dtype=np.float64)
            upper = np.full((rows, _N_FIELDS), np.nan, dtype=np.float64)
            # Cache radii first; scoring targets never participate in the fit.
            arm_radii: dict[int, np.ndarray] = {}
            for fold in unique_folds:
                calibration_mask = observed & ((fold_values == fold) & (role_values == 0))[:, None]
                radius = fit_radii(clean_truth, clean_prediction, calibration_mask, alpha=0.1, minimum=20)
                arm_radii[fold] = radius
                scoring = observed & ((fold_values == fold) & (role_values == 1))[:, None]
                usable = scoring & np.isfinite(radius)[None, :]
                lower[usable] = (clean_prediction - radius[None, :])[usable]
                upper[usable] = (clean_prediction + radius[None, :])[usable]
            lower_arms[arm] = _readonly(lower)
            upper_arms[arm] = _readonly(upper)
            # We organize after both arms have run because folds are a shared
            # support contract, never an arm-selected sample.
            for fold, radius in arm_radii.items():
                existing = dict(by_fold.get(fold, {}))
                existing[arm] = _readonly(radius)
                by_fold[fold] = MappingProxyType(existing)
        radii_out[pattern] = MappingProxyType(by_fold)
        lower_out[pattern] = MappingProxyType(lower_arms)
        upper_out[pattern] = MappingProxyType(upper_arms)

    return CBCCalibrationV2(
        radii=MappingProxyType(radii_out),
        lower=MappingProxyType(lower_out),
        upper=MappingProxyType(upper_out),
    )
