"""Train-only nested Ridge completion heads for frozen BRAN state features.

This small helper deliberately does not train or alter an encoder.  It accepts
outer-train state-plus-age features and never accepts test labels.  Every
scaler is fit only on the corresponding inner-training or outer-training rows.
Returned arrays and sklearn objects remain in caller memory; this module has
no filesystem, logging, or command-line interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


CBC_TARGET_DIM = 9
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


class FrozenCBCRidgeError(ValueError):
    """Raised before a potentially invalid nested completion head is fit."""


@dataclass(frozen=True)
class FrozenCBCRidgeResult:
    """In-memory predictions and fitted outer-train-only decoder objects."""

    predictions: np.ndarray
    selected_alpha: tuple[float, ...]
    scalers: tuple[StandardScaler, ...]
    heads: tuple[Ridge, ...]
    inner_weighted_mse: tuple[float | None, ...]
    fixed_alpha1: bool


def _float_matrix(value: np.ndarray, name: str, rows: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or (rows is not None and array.shape[0] != rows) or not array.shape[1] or not np.isfinite(array).all():
        raise FrozenCBCRidgeError(f"invalid_{name}")
    return array


def _valid_labels(target: np.ndarray, observed: np.ndarray) -> np.ndarray:
    if observed.shape != target.shape or observed.dtype.kind != "b":
        raise FrozenCBCRidgeError("invalid_observed_mask")
    # Missing target payload never flows into sklearn.  A nonfinite target is
    # considered unobserved even if a caller's mask was overbroad.
    return observed & np.isfinite(target)


def _validate_inputs(
    x_train: np.ndarray, y_train: np.ndarray, observed_train: np.ndarray,
    inner_fold_ids: np.ndarray, x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = _float_matrix(x_train, "x_train")
    test_x = _float_matrix(x_test, "x_test")
    if test_x.shape[1] != train_x.shape[1]:
        raise FrozenCBCRidgeError("x_feature_dimension_mismatch")
    target = np.asarray(y_train, dtype=np.float64)
    if target.shape != (len(train_x), CBC_TARGET_DIM):
        raise FrozenCBCRidgeError("invalid_y_train")
    valid = _valid_labels(target, np.asarray(observed_train))
    inner = np.asarray(inner_fold_ids)
    if inner.shape != (len(train_x),) or inner.dtype.kind not in "iu" or set(inner.tolist()) != set(range(5)):
        raise FrozenCBCRidgeError("invalid_five_inner_fold_ids")
    return train_x, target, valid, inner.astype(np.int64, copy=False), test_x


def _require_inner_support(valid: np.ndarray, inner: np.ndarray) -> None:
    """Fail closed per target and inner fold before any candidate selection."""
    for fold in range(5):
        fit, score = inner != fold, inner == fold
        if np.any(valid[fit].sum(axis=0) < 2) or np.any(valid[score].sum(axis=0) < 10):
            raise FrozenCBCRidgeError("insufficient_inner_label_support")


def _weighted_inner_mse(
    x: np.ndarray, y: np.ndarray, valid: np.ndarray, inner: np.ndarray, field: int, alpha: float,
) -> float:
    squared_error, support = 0.0, 0
    for fold in range(5):
        fit_rows, score_rows = inner != fold, inner == fold
        # Both scalers and target heads see only the inner-training rows.
        scaler = StandardScaler().fit(x[fit_rows])
        fit_labels = fit_rows & valid[:, field]
        score_labels = score_rows & valid[:, field]
        head = Ridge(alpha=alpha).fit(scaler.transform(x[fit_labels]), y[fit_labels, field])
        residual = head.predict(scaler.transform(x[score_labels])) - y[score_labels, field]
        squared_error += float(residual @ residual)
        support += int(score_labels.sum())
    if support < 1 or not np.isfinite(squared_error):
        raise FrozenCBCRidgeError("nonfinite_inner_score")
    return squared_error / support


def fit_frozen_cbc_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    observed_train: np.ndarray,
    inner_fold_ids: np.ndarray,
    x_test: np.ndarray,
    *,
    fixed_alpha1: bool = False,
) -> FrozenCBCRidgeResult:
    """Fit nine standardized, train-only nested-Ridge target heads.

    For normal operation every candidate is scored by a count-weighted sum of
    all five inner-fold squared errors.  Exact ties choose the larger alpha.
    ``fixed_alpha1=True`` skips only alpha selection but retains the identical
    outer-train StandardScaler policy for a controlled alpha=1 ablation.
    """
    x, y, valid, inner, test_x = _validate_inputs(x_train, y_train, observed_train, inner_fold_ids, x_test)
    _require_inner_support(valid, inner)
    selected: list[float] = []
    scores: list[float | None] = []
    scalers: list[StandardScaler] = []
    heads: list[Ridge] = []
    prediction = np.empty((len(test_x), CBC_TARGET_DIM), dtype=np.float64)
    for field in range(CBC_TARGET_DIM):
        if fixed_alpha1:
            alpha, score = 1.0, None
        else:
            candidate_scores = {alpha: _weighted_inner_mse(x, y, valid, inner, field, alpha) for alpha in RIDGE_ALPHAS}
            alpha, score = max(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
        outer_scaler = StandardScaler().fit(x)
        fit_labels = valid[:, field]
        outer_head = Ridge(alpha=alpha).fit(outer_scaler.transform(x[fit_labels]), y[fit_labels, field])
        prediction[:, field] = outer_head.predict(outer_scaler.transform(test_x))
        if not np.isfinite(prediction[:, field]).all():
            raise FrozenCBCRidgeError("nonfinite_prediction")
        selected.append(float(alpha)); scores.append(None if score is None else float(score))
        scalers.append(outer_scaler); heads.append(outer_head)
    return FrozenCBCRidgeResult(prediction, tuple(selected), tuple(scalers), tuple(heads), tuple(scores), bool(fixed_alpha1))
