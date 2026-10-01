"""Array-only matched screening readouts with leakage-safe outer training.

This module deliberately accepts arrays supplied by its caller and neither reads nor
writes data.  ``fit_predict`` returns only test probabilities and compact numeric
fit diagnostics; it does not retain a fitted estimator.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


_C_GRID = (0.01, 0.1, 1.0, 10.0)
_INVALID_INPUT = "matched screening kernel inputs invalid"
_FAMILY_FAILURE = "matched screening kernel family failure"


def _invalid() -> None:
    raise ValueError(_INVALID_INPUT)


def _as_float_array(value: object, *, ndim: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        _invalid()
    if array.ndim != ndim:
        _invalid()
    return array


def _indices(value: object, n_rows: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        _invalid()
    if len(array) == 0 or np.any(array < 0) or np.any(array >= n_rows):
        _invalid()
    array = array.astype(np.intp, copy=False)
    if len(np.unique(array)) != len(array):
        _invalid()
    return array


def _validate_inputs(x: object, y: object, observed: object, train: object,
                     test: object, inner_fold_ids: object, seed: object):
    features = _as_float_array(x, ndim=2)
    labels = _as_float_array(y, ndim=1)
    mask = np.asarray(observed)
    n_rows = len(features)
    if (features.shape[1] == 0 or labels.shape != (n_rows,) or mask.shape != (n_rows,)
            or mask.dtype.kind != "b" or not np.isfinite(features).all()):
        _invalid()
    # Unobserved label payloads may be NaN; observed labels must be binary.
    if np.any(mask & (~np.isfinite(labels) | ((labels != 0.0) & (labels != 1.0)))):
        _invalid()
    outer_train = _indices(train, n_rows)
    outer_test = _indices(test, n_rows)
    if np.intersect1d(outer_train, outer_test).size:
        _invalid()
    folds = np.asarray(inner_fold_ids)
    if (folds.shape != (n_rows,) or folds.dtype.kind not in "iu" or folds.dtype.kind == "b"
            or np.any(folds[outer_train] < 0)):
        _invalid()
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        _invalid()
    fit = outer_train[mask[outer_train]]
    if len(fit) < 2 or set(np.unique(labels[fit])) != {0.0, 1.0}:
        _invalid()
    return features, labels, mask, outer_train, outer_test, folds, fit, int(seed)


def _fit_logistic(features: np.ndarray, labels: np.ndarray, fit: np.ndarray, c: float,
                  seed: int):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, solver="lbfgs", max_iter=5000, random_state=seed),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features[fit], labels[fit])
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        return None
    return model


def _fit_mlp(features: np.ndarray, labels: np.ndarray, fit: np.ndarray, seed: int):
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32,), alpha=0.01, max_iter=300,
            early_stopping=False, random_state=seed,
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features[fit], labels[fit])
    # A fixed 300-iteration budget defines this family.  It is not a search
    # for a convergence optimum, so retain the fitted final iterate and report
    # whether the budget was exhausted in compact diagnostics.
    budget_exhausted = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return model, budget_exhausted


def _fit_extra_trees(features: np.ndarray, labels: np.ndarray, fit: np.ndarray, seed: int):
    model = ExtraTreesClassifier(
        n_estimators=128, max_depth=6, min_samples_leaf=10, max_features=0.7,
        n_jobs=2, random_state=seed,
    )
    model.fit(features[fit], labels[fit])
    return model


def _select_logistic_c(features: np.ndarray, labels: np.ndarray, observed: np.ndarray,
                       outer_train: np.ndarray, folds: np.ndarray, seed: int):
    """Choose C by the unweighted mean of AUROCs over complete inner folds."""
    train_folds = np.unique(folds[outer_train])
    if len(train_folds) < 2:
        raise ValueError(_FAMILY_FAILURE)
    choices: list[tuple[float, float]] = []
    rejected = 0
    for c in _C_GRID:
        fold_aucs: list[float] = []
        complete = True
        converged = True
        for fold in train_folds:
            inner_fit = outer_train[(folds[outer_train] != fold) & observed[outer_train]]
            inner_validation = outer_train[(folds[outer_train] == fold) & observed[outer_train]]
            if (len(inner_fit) < 2 or len(inner_validation) < 2
                    or set(np.unique(labels[inner_fit])) != {0.0, 1.0}
                    or set(np.unique(labels[inner_validation])) != {0.0, 1.0}):
                complete = False
                break
            model = _fit_logistic(features, labels, inner_fit, c, seed)
            if model is None:
                converged = False
                break
            probabilities = model.predict_proba(features[inner_validation])[:, 1]
            fold_aucs.append(float(roc_auc_score(labels[inner_validation], probabilities)))
        if complete and converged:
            # Smaller C is the deterministic tie-breaker.
            choices.append((-float(np.mean(fold_aucs)), c))
        else:
            rejected += 1
    if not choices:
        raise ValueError(_FAMILY_FAILURE)
    return min(choices)[1], rejected, len(train_folds)


def fit_predict(x, y, observed, train, test, inner_fold_ids, family, seed):
    """Fit one readout exclusively on observed outer-training labels.

    Args use integer row indices for ``train`` and ``test``.  ``inner_fold_ids``
    is an integer array aligned to all rows, though only outer-training entries
    are inspected.  ``family`` is one of ``"logistic"``, ``"mlp"``, or
    ``"extra_trees"``.

    Returns:
        ``(probabilities, diagnostics)`` where probabilities are aligned to
        ``test`` and diagnostics has exactly ``family``, ``selected_C``,
        ``rejected_candidates``, ``n_inner_folds``, and
        ``iteration_budget_exhausted``.
    """
    if family not in {"logistic", "mlp", "extra_trees"}:
        _invalid()
    features, labels, mask, outer_train, outer_test, folds, fit, fit_seed = _validate_inputs(
        x, y, observed, train, test, inner_fold_ids, seed
    )
    n_inner_folds = int(len(np.unique(folds[outer_train])))
    if family == "logistic":
        selected_c, rejected, n_inner_folds = _select_logistic_c(
            features, labels, mask, outer_train, folds, fit_seed
        )
        model = _fit_logistic(features, labels, fit, selected_c, fit_seed)
        if model is None:
            raise ValueError(_FAMILY_FAILURE)
    elif family == "mlp":
        selected_c, rejected = None, 0
        model, iteration_budget_exhausted = _fit_mlp(features, labels, fit, fit_seed)
    else:
        selected_c, rejected = None, 0
        model = _fit_extra_trees(features, labels, fit, fit_seed)
    if family != "mlp":
        iteration_budget_exhausted = False
    probabilities = np.asarray(model.predict_proba(features[outer_test])[:, 1], dtype=float)
    if probabilities.shape != (len(outer_test),) or not np.isfinite(probabilities).all():
        raise ValueError(_FAMILY_FAILURE)
    diagnostics = {
        "family": family,
        "selected_C": selected_c,
        "rejected_candidates": int(rejected),
        "n_inner_folds": int(n_inner_folds),
        "iteration_budget_exhausted": bool(iteration_budget_exhausted),
    }
    return probabilities, diagnostics


def late_fusion_average(p_clinical, p_retinal):
    """Fixed, unfitted equal-weight late fusion of aligned probabilities."""
    clinical = _as_float_array(p_clinical, ndim=1)
    retinal = _as_float_array(p_retinal, ndim=1)
    if clinical.shape != retinal.shape or not np.isfinite(clinical).all() or not np.isfinite(retinal).all():
        _invalid()
    if np.any((clinical < 0.0) | (clinical > 1.0) | (retinal < 0.0) | (retinal > 1.0)):
        _invalid()
    return 0.5 * clinical + 0.5 * retinal
