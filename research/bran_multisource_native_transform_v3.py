"""Carry authenticated native normalization into V3; never fit new statistics.

Source authentication belongs to the local caller. Arrays returned here remain
private; this module performs no data loading, logging or serialization.
"""
import hashlib

import numpy as np

from bran_multisource_data_v2 import FoldTransformV2, frozen


def inherit_native_transform(native, eligible_indices, folds, heldout_fold):
    def require(ok):
        if not ok:
            raise ValueError('native_transform_bridge_invalid')

    require(type(folds) is np.ndarray and folds.ndim == 1 and folds.dtype.kind in 'iu')
    require(type(heldout_fold) is int and heldout_fold in range(5)
            and set(np.unique(folds)) == set(range(5)))
    try:
        indices = tuple(eligible_indices)
    except TypeError:
        raise ValueError('native_transform_bridge_invalid') from None
    require(len(indices) == 43 and all(type(i) is int and i in range(48) for i in indices))
    require(len(set(indices)) == 43)
    values = {}
    for key, width in (('clinical_median', 59), ('clinical_iqr', 59),
                       ('retinal_mean', 384), ('retinal_scale', 384)):
        value = getattr(native, key, None)
        require(type(value) is np.ndarray and value.shape == (width,)
                and value.dtype.kind == 'f' and bool(np.isfinite(value).all()))
        if key in ('clinical_iqr', 'retinal_scale'):
            require(bool((value > 0).all()))
        values[key] = frozen(value)
    for key in ('age_mean', 'age_scale'):
        value = getattr(native, key, None)
        require(type(value) in (float, int) and np.isfinite(value))
        if key == 'age_scale':
            require(value > 0)
        values[key] = float(value)
    eligible = np.zeros(59, dtype=bool)
    eligible[list(indices)] = True
    train = np.flatnonzero(folds != heldout_fold)
    return FoldTransformV2(**values, eligible=frozen(eligible), heldout_fold=heldout_fold,
        fold_identity_sha256=hashlib.sha256(np.asarray(folds, dtype='<i8').tobytes()).hexdigest(),
        training_indices_sha256=hashlib.sha256(np.asarray(train, dtype='<i8').tobytes()).hexdigest())
