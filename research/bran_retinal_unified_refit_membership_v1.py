"""Private prospective refit memberships; no source I/O, labels or model fitting.

The launcher must bind the resulting whole private artifact before training.
Recreating its expected hash from chosen fitting rows is not authentication.
"""
import copy

import numpy as np

import bran_retinal_refit_inputs_v2 as inputs
import bran_native_calibration_split_v1 as calibration

SCHEMA = 'bran-retinal-unified-refit-memberships-v1'
JOBS = tuple('outer' + str(fold) for fold in range(5)) + ('structure',)
ERROR = 'unified refit membership rejected'


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def _membership(patient_ids, folds, fit, outer_fold):
    excluded = np.setdiff1d(np.arange(len(folds), dtype=np.int64), fit)
    return {'schema': 'bran-retinal-unified-fit-membership-v1',
        'scope': 'preprocessing_and_representation_training', 'outer_fold': outer_fold,
        'patient_order_sha256': inputs.digest(patient_ids), 'fold_order_sha256': inputs.digest(folds.tolist()),
        'fit_indices_sha256': inputs.digest(fit.tolist()), 'excluded_indices_sha256': inputs.digest(excluded.tolist())}


def build(patient_ids, folds, inner_folds):
    """One fixed fit per outer fold plus a discovery-only same-coordinate model.

    ``inner_folds`` are the original authenticated five inner-fold vectors,
    supplied here in full-cohort order with -1 on each corresponding outer test.
    They belong to fixed baseline readouts, not tuning the neural training budget.
    Original externally pinned inner identities remain the launcher's obligation.
    """
    require(type(patient_ids) is list and len(patient_ids) > 0
            and all(type(item) is str and item for item in patient_ids)
            and len(set(patient_ids)) == len(patient_ids))
    n = len(patient_ids)
    require(type(folds) is np.ndarray and folds.shape == (n,) and folds.dtype == np.int64
            and set(folds.tolist()) == set(range(5)))
    require(type(inner_folds) is dict and set(inner_folds) == set(range(5)))
    roles = calibration.make_roles(patient_ids, folds)
    fits, private_inner = {}, {}
    for fold in range(5):
        fit, heldout = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        inner = inner_folds[fold]
        require(type(inner) is np.ndarray and inner.shape == (n,) and inner.dtype == np.int64
                and np.all(inner[heldout] == -1) and set(inner[fit].tolist()) == set(range(5)))
        private_inner[str(fold)] = inner.tolist()
        membership = _membership(patient_ids, folds, fit, fold)
        fits['outer' + str(fold)] = {'outer_fold': fold, 'seed_index': fold,
            'fit_indices': fit.tolist(), 'screen_indices': heldout.tolist(),
            'calibration_indices': np.flatnonzero((folds == fold) & (roles == 0)).tolist(),
            'completion_score_indices': np.flatnonzero((folds == fold) & (roles == 1)).tolist(),
            'validation_indices': [], 'replication_indices': [],
            'membership': membership, 'membership_sha256': inputs.digest(membership)}
    fit = np.flatnonzero(folds < 3)
    membership = _membership(patient_ids, folds, fit, 4)
    fits['structure'] = {'outer_fold': 4, 'seed_index': 0,
        'fit_indices': fit.tolist(), 'screen_indices': [], 'calibration_indices': [],
        'completion_score_indices': [], 'validation_indices': np.flatnonzero(folds == 3).tolist(),
        'replication_indices': np.flatnonzero(folds == 4).tolist(),
        'membership': membership, 'membership_sha256': inputs.digest(membership)}
    return {'schema': SCHEMA, 'patient_ids': patient_ids.copy(), 'outer_folds': folds.tolist(),
            'inner_folds': private_inner, 'calibration_roles': roles.tolist(), 'fits': fits,
            'new_outcome_data_read_by_builder': False, 'cross_checkpoint_state_pooling_permitted': False}


def authenticate(value, expected_sha256, *, patient_ids, folds, inner_folds):
    require(type(expected_sha256) is str and inputs.digest(value) == expected_sha256)
    require(inputs.digest(value) == inputs.digest(build(patient_ids, folds, inner_folds)))
    return copy.deepcopy(value)


def selected(value, expected_sha256, job, *, patient_ids, folds, inner_folds):
    """Return a defensive private job only after complete prospective validation."""
    require(type(job) is str and job in JOBS)
    authenticated = authenticate(value, expected_sha256, patient_ids=patient_ids,
                                 folds=folds, inner_folds=inner_folds)
    item = authenticated['fits'][job]
    fit = np.asarray(item['fit_indices'], dtype=np.int64)
    inputs.validate_fit_membership(item['membership'], item['membership_sha256'],
        patient_ids=patient_ids, folds=folds, outer_fold=item['outer_fold'], fit_indices=fit)
    return item
