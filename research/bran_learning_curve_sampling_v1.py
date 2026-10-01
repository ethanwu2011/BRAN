"""Pure nested training subsets and fixed local screening readouts.

This module is intentionally array-only: it has no labels in its sampling API
and never reads or writes files.  Callers use a returned subset for both stages
of a learning-curve fit, preserving identical people across those stages.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


_INVALID = "learning curve sampling inputs invalid"
_FIT_FAILURE = "learning curve screening fit failure"
_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
_KEYS = ("025", "050", "075", "100")
_MAX_RANDOM_STATE = 2**32 - 1


def _bad() -> None:
    raise ValueError(_INVALID)


def _n_rows(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        _bad()
    result = int(value)
    if result < 1:
        _bad()
    return result


def _seed(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        _bad()
    result = int(value)
    if result < 0 or result > _MAX_RANDOM_STATE:
        _bad()
    return result


def _indices(value: object, n_rows: int) -> np.ndarray:
    try:
        index = np.asarray(value)
    except (TypeError, ValueError):
        _bad()
    if (
        index.ndim != 1
        or index.size == 0
        or index.dtype.kind not in "iu"
        or index.dtype.kind == "b"
        or np.any(index < 0)
        or np.any(index >= n_rows)
    ):
        _bad()
    result = index.astype(np.intp, copy=False)
    if np.unique(result).size != result.size:
        _bad()
    return result


def _split(train_indices: object, test_indices: object, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    train = _indices(train_indices, n_rows)
    test = _indices(test_indices, n_rows)
    if np.intersect1d(train, test).size:
        _bad()
    return train, test


def nested_training_subsets(
    train_indices,
    test_indices,
    n_rows,
    fractions=(0.25, 0.5, 0.75, 1.0),
    seed=97001,
) -> dict[str, np.ndarray]:
    """Return frozen, nested, ascending training subsets.

    A single seeded permutation of the canonical ascending training set is
    used.  Therefore each smaller subset is a prefix of the next before its
    returned indices are sorted.  No targets are accepted or considered.
    """
    total = _n_rows(n_rows)
    if not isinstance(fractions, tuple) or fractions != _FRACTIONS:
        _bad()
    fit_seed = _seed(seed)
    train, _ = _split(train_indices, test_indices, total)
    ordered = np.sort(train)
    sizes = tuple(int(np.floor(fraction * ordered.size)) for fraction in _FRACTIONS)
    if sizes[0] < 20:
        _bad()
    # RandomState makes the frozen selection independent of NumPy's global RNG.
    permuted = np.random.RandomState(fit_seed).permutation(ordered)
    subsets = {
        key: np.sort(permuted[:size]).astype(np.intp, copy=False)
        for key, size in zip(_KEYS, sizes)
    }
    # The full level is the canonical supplied training set, not a shuffled copy.
    subsets["100"] = ordered.astype(np.intp, copy=False)
    return subsets


def _features(value: object) -> np.ndarray:
    try:
        features = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        _bad()
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0 or not np.isfinite(features).all():
        _bad()
    return features


def _screen_inputs(x: object, y: object, observed: object, train: object, test: object,
                   seed: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    features = _features(x)
    try:
        raw_target = np.asarray(y)
        mask = np.asarray(observed)
    except (TypeError, ValueError):
        _bad()
    if raw_target.ndim != 1 or raw_target.shape != (len(features),):
        _bad()
    if mask.dtype.kind != "b" or mask.shape != (len(features),):
        _bad()
    train_rows, test_rows = _split(train, test, len(features))
    fit_seed = _seed(seed)
    fit_rows = train_rows[mask[train_rows]]
    if fit_rows.size == 0:
        return features, np.empty(0, dtype=np.float64), fit_rows, test_rows, fit_seed
    # Hidden train targets and all held-out targets are deliberately untouched.
    try:
        fit_target = np.asarray(raw_target[fit_rows], dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        _bad()
    if fit_target.shape != (fit_rows.size,) or not np.isfinite(fit_target).all():
        _bad()
    if np.any((fit_target != 0.0) & (fit_target != 1.0)):
        _bad()
    return features, fit_target, fit_rows, test_rows, fit_seed


def constant_or_tree_screen(x, y, observed, train, test, seed) -> np.ndarray:
    """Return a Laplace-prior constant or the frozen unweighted tree readout.

    Fewer than 20 visible training labels, or a one-class visible fit set, is
    intentionally a constant prediction.  A two-class set of at least 20 uses
    the fixed ExtraTrees configuration with no label-dependent retry or tuning.
    """
    features, fit_y, fit_rows, test_rows, fit_seed = _screen_inputs(
        x, y, observed, train, test, seed
    )
    positives = int(np.sum(fit_y == 1.0))
    if fit_rows.size < 20 or positives == 0 or positives == fit_rows.size:
        probability = (positives + 1.0) / (fit_rows.size + 2.0)
        return np.full(test_rows.size, probability, dtype=np.float64)
    estimator = ExtraTreesClassifier(
        n_estimators=128,
        max_depth=6,
        min_samples_leaf=10,
        max_features=0.7,
        class_weight=None,
        random_state=fit_seed,
        n_jobs=2,
    )
    try:
        with warnings.catch_warnings(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            estimator.fit(features[fit_rows], fit_y)
            prediction = estimator.predict_proba(features[test_rows])[:, 1]
        result = np.asarray(prediction, dtype=np.float64)
    except Exception:
        raise ValueError(_FIT_FAILURE) from None
    if result.shape != (test_rows.size,) or not np.isfinite(result).all():
        raise ValueError(_FIT_FAILURE)
    return result
