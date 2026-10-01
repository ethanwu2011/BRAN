"""Private multisource-retinal handoff with explicit frozen fit membership.

No file/source authentication or model fitting. Callers authenticate the source
audit, extraction bytes and membership artifact before entering this kernel.
Never insert changed retinal coordinates into old BRAN normalization/weights.
V1 remains untouched for the historical supervised-source experiment.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from bran_authenticated_retinal_input_v1 import pool_arrays
from bran_retinal_extraction_kernel_v1 import inventory_sha256
from bran_retinal_multisource_encoder_binding_v1 import require_encoder_identity
from run_bran_overnight_diagnostic_v1 import FoldTransform


def require(value):
    if not value:
        raise ValueError('multisource retinal refit input rejected') from None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, repr=False)
class RefitInputs:
    """Patient-derived private state, without array-containing default repr."""
    original_transform: object
    candidate_transform: object
    original: tuple
    candidate: tuple
    retinal_present: np.ndarray
    fit_indices: np.ndarray
    outer_test_indices: np.ndarray


def validate_fit_membership(membership, expected_sha256, *, patient_ids, folds, outer_fold, fit_indices):
    """Require the exact prospective membership, not just any outer-train subset.

    ``expected_sha256`` is the canonical private membership hash pinned by the
    caller's prospective protocol. Do not derive it on the fly from supplied
    fitting rows during training. This is a binding check, not authentication
    of the protocol or permission to inspect/emit private membership hashes.
    """
    require(type(membership) is dict and set(membership) == {
        'schema', 'scope', 'outer_fold', 'patient_order_sha256', 'fold_order_sha256',
        'fit_indices_sha256', 'excluded_indices_sha256'})
    excluded = np.setdiff1d(np.arange(len(patient_ids), dtype=np.int64), fit_indices)
    expected = {'schema': 'bran-retinal-unified-fit-membership-v1',
        'scope': 'preprocessing_and_representation_training', 'outer_fold': outer_fold,
        'patient_order_sha256': digest(patient_ids), 'fold_order_sha256': digest(folds.tolist()),
        'fit_indices_sha256': digest(fit_indices.tolist()), 'excluded_indices_sha256': digest(excluded.tolist())}
    require(type(expected_sha256) is str and digest(membership) == expected_sha256
            and json.dumps(membership, sort_keys=True) == json.dumps(expected, sort_keys=True))


def prepare_fold(*, records, image_features, patient_ids, folds, selection,
                 candidate_contract, expected_checkpoint_sha256, original_retinal,
                 retinal_present, clinical, observed, eligible, age, outer_fold,
                 fit_indices, fit_membership, expected_fit_membership_sha256):
    """Pool genuine images, keep natural missingness, fit encoder-specific scales.

    Membership excludes outer-test and any prospectively withheld inner or
    calibration rows from BOTH preprocessing and later representation fitting.
    This function returns that exact fit index vector for the model trainer.
    No labels, outcomes, imputed pairing or historical checkpoint weights enter.
    """
    try:
        require_encoder_identity(candidate_contract, expected_checkpoint_sha256)
        require(type(patient_ids) is list and all(type(x) is str and x for x in patient_ids)
                and len(set(patient_ids)) == len(patient_ids))
        n = len(patient_ids)
        require(type(folds) is np.ndarray and folds.shape == (n,) and folds.dtype.kind in 'iu'
                and set(folds.tolist()) == set(range(5)))
        require(type(outer_fold) is int and outer_fold in range(5))
        require(type(selection) is dict and set(selection) == {
            'patient_order_sha256', 'fold_order_sha256', 'selection_sha256', 'rows', 'people'})
        require(type(selection['rows']) is int and selection['rows'] == len(records)
                and type(selection['people']) is int and selection['people'] == n)
        require(selection['patient_order_sha256'] == digest(patient_ids)
                and selection['fold_order_sha256'] == digest(folds.tolist())
                and selection['selection_sha256'] == inventory_sha256(
                    [{**record, 'source_sha256': '0' * 64} for record in records]))
        require(type(retinal_present) is np.ndarray and retinal_present.shape == (n,)
                and retinal_present.dtype == np.dtype(bool))
        require(type(original_retinal) is np.ndarray and original_retinal.shape == (n, 384)
                and original_retinal.dtype.kind == 'f' and np.isfinite(original_retinal[retinal_present]).all())
        require(type(clinical) is np.ndarray and clinical.shape == (n, 59) and clinical.dtype.kind == 'f')
        require(all(type(x) is np.ndarray and x.shape == clinical.shape and x.dtype == np.dtype(bool)
                    for x in (observed, eligible)))
        require(np.isfinite(clinical[observed & eligible]).all())
        require(type(age) is np.ndarray and age.shape == (n,) and age.dtype.kind == 'f' and np.isfinite(age).all())
        require(type(fit_indices) is np.ndarray and fit_indices.ndim == 1 and len(fit_indices) > 0
                and fit_indices.dtype.kind in 'iu' and np.all((fit_indices >= 0) & (fit_indices < n))
                and np.all(np.diff(fit_indices.astype(np.int64)) > 0)
                and np.all(folds[fit_indices] != outer_fold) and retinal_present[fit_indices].any())
        validate_fit_membership(fit_membership, expected_fit_membership_sha256, patient_ids=patient_ids,
                                folds=folds, outer_fold=outer_fold, fit_indices=fit_indices)
        pooled, present = pool_arrays(records, image_features, patient_ids)
        require(np.array_equal(present, retinal_present))
        transforms, values = [], []
        for retina in (original_retinal, pooled):
            transform = FoldTransform(clinical, observed, eligible, retina, present, age, fit_indices)
            c, cm, retinal, ages = transform.apply(clinical, observed, eligible, retina, present, age)
            require(np.isfinite(c).all() and np.isfinite(retinal).all() and np.isfinite(ages).all()
                    and np.all(retinal[~present] == 0) and np.all(c[~cm] == 0))
            transforms.append(transform)
            values.append(tuple(readonly(x) for x in (c, cm, retinal, ages)))
        for key in ('clinical_median', 'clinical_iqr', 'age_mean', 'age_scale'):
            require(np.array_equal(getattr(transforms[0], key), getattr(transforms[1], key)))
        require(all(np.array_equal(values[0][j], values[1][j]) for j in (0, 1, 3)))
        return RefitInputs(*transforms, *values, readonly(present), readonly(fit_indices),
                           readonly(np.flatnonzero(folds == outer_fold)))
    except Exception:
        raise ValueError('multisource retinal refit input rejected') from None
