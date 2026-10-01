"""Outcome-blind calibration/scoring roles inside each model's held-out fold.

Pure local helper. IDs and arrays must never be emitted by callers. This does
not authenticate a checkpoint or establish exchangeability by itself.
"""
import hashlib
import numpy as np

SALT = "bran-native-cbc-calibration-v1-sep09"


def require(ok):
    if not ok:
        raise ValueError("native_calibration_split_invalid")


def make_roles(patient_ids, folds):
    require(type(patient_ids) in (tuple, list) and len(patient_ids) > 0)
    require(all(type(x) is str and x for x in patient_ids) and len(set(patient_ids)) == len(patient_ids))
    require(not isinstance(folds, (list, tuple)) or not any(isinstance(x, (bool, np.bool_)) for x in folds))
    values = np.asarray(folds)
    require(values.shape == (len(patient_ids),) and np.issubdtype(values.dtype, np.integer)
            and values.dtype != np.bool_ and set(values.tolist()) == set(range(5)))
    roles = np.full(len(values), -1, dtype=np.int8)
    for fold in range(5):
        indices = np.flatnonzero(values == fold)
        require(len(indices) >= 40)
        ordered = sorted(indices.tolist(), key=lambda i: (
            hashlib.sha256((SALT + "\0" + patient_ids[i]).encode()).digest(), patient_ids[i]))
        # Each scoring prediction uses its own outer-fold model. Calibration
        # labels come only from the other half of that SAME held-out fold.
        roles[ordered[:len(ordered)//2]] = 0
        roles[ordered[len(ordered)//2:]] = 1
    require(np.isin(roles, [0, 1]).all())
    roles.setflags(write=False)
    return roles


def partition(patient_ids, folds, fold):
    require(type(fold) is int and 0 <= fold < 5)
    roles = make_roles(patient_ids, folds)
    values = np.asarray(folds)
    fit = np.flatnonzero(values != fold)
    calibration = np.flatnonzero((values == fold) & (roles == 0))
    score = np.flatnonzero((values == fold) & (roles == 1))
    require(not (set(fit) & set(calibration) or set(fit) & set(score) or set(calibration) & set(score)))
    require(len(fit) + len(calibration) + len(score) == len(patient_ids))
    for x in (fit, calibration, score):
        x.setflags(write=False)
    return fit, calibration, score
