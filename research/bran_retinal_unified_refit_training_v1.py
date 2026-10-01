"""Fresh, fit-only BRAN refitting after a prospective retinal replacement.

This private-array wrapper deliberately has no source, checkpoint, or file
interface.  Membership authentication and all execution/persistence remain
with its caller.
"""
from __future__ import annotations

import numpy as np

from bran_external_cbc_fit_kernel_v1 import paired_train
from bran_raw_teacher_distillation_v1 import _validate_model
from bran_retinal_refit_inputs_v2 import RefitInputs, validate_fit_membership
from bran_screening_joint_kernel_v1 import adapt


_ERROR = 'retinal unified refit training rejected'


def _fail():
    raise ValueError(_ERROR) from None


def _require(value):
    if not value:
        _fail()


def _indices(inputs, patient_ids, folds, outer_fold, membership, expected):
    _require(type(inputs) is RefitInputs and type(patient_ids) is list
             and patient_ids and all(type(value) is str and value for value in patient_ids)
             and len(set(patient_ids)) == len(patient_ids))
    n = len(patient_ids)
    _require(type(folds) is np.ndarray and folds.dtype == np.dtype(np.int64)
             and folds.shape == (n,) and set(folds.tolist()) == set(range(5))
             and type(outer_fold) is int and outer_fold in range(5)
             and type(inputs.outer_test_indices) is np.ndarray
             and inputs.outer_test_indices.dtype == np.dtype(np.int64)
             and np.array_equal(inputs.outer_test_indices, np.flatnonzero(folds == outer_fold)))
    fit = inputs.fit_indices
    _require(type(fit) is np.ndarray and fit.dtype == np.dtype(np.int64) and fit.ndim == 1
             and len(fit) > 0 and np.all((fit >= 0) & (fit < n))
             and np.all(np.diff(fit) > 0) and np.all(folds[fit] != outer_fold))
    # This is deliberately before inspecting training labels or invoking either
    # trainer: a different inner subset is not a permitted refit.
    validate_fit_membership(membership, expected, patient_ids=patient_ids, folds=folds,
                            outer_fold=outer_fold, fit_indices=fit)
    return fit


def _arm_arrays(inputs, arm, n, fit):
    _require(type(arm) is str and arm in ('original', 'candidate')
             and type(inputs.original) is tuple and type(inputs.candidate) is tuple
             and len(inputs.original) == len(inputs.candidate) == 4
             and type(inputs.retinal_present) is np.ndarray
             and inputs.retinal_present.dtype == np.dtype(bool)
             and inputs.retinal_present.shape == (n,))
    original, candidate = inputs.original, inputs.candidate
    for value in (original[0], candidate[0]):
        _require(type(value) is np.ndarray and value.dtype.kind == 'f' and value.shape == (n, 59))
    for value in (original[1], candidate[1]):
        _require(type(value) is np.ndarray and value.dtype == np.dtype(bool) and value.shape == (n, 59))
    for value in (original[2], candidate[2]):
        _require(type(value) is np.ndarray and value.dtype.kind == 'f' and value.shape == (n, 384))
    for value in (original[3], candidate[3]):
        _require(type(value) is np.ndarray and value.dtype.kind == 'f' and value.shape == (n,))
    # Values outside ``fit`` must not even participate in parity/finite
    # validation; they are excluded from the refit by protocol.
    _require(np.array_equal(original[0][fit], candidate[0][fit])
             and np.array_equal(original[1][fit], candidate[1][fit])
             and np.array_equal(original[3][fit], candidate[3][fit]))
    values = original if arm == 'original' else candidate
    clinical, clinical_mask, retinal, age = (value[fit].copy() for value in values)
    retinal_mask = inputs.retinal_present[fit].copy()
    _require(np.isfinite(age).all() and np.isfinite(clinical[clinical_mask]).all()
             and np.isfinite(retinal[retinal_mask]).all())
    # Copy and physically erase unavailable payloads before either legacy
    # trainer receives them.  The input handoff remains caller-owned.
    return (np.where(clinical_mask, clinical, 0.0).astype(np.float32, copy=False),
            clinical_mask.copy(),
            np.where(retinal_mask[:, None], retinal, 0.0).astype(np.float32, copy=False),
            retinal_mask, age.astype(np.float32, copy=True))


def _labels(labels, observed, n, fit):
    _require(type(labels) is np.ndarray and labels.dtype.kind == 'f' and labels.shape == (n, 26)
             and type(observed) is np.ndarray and observed.dtype == np.dtype(bool)
             and observed.shape == labels.shape)
    labels, observed = labels[fit].copy(), observed[fit].copy()
    _require(not np.any(observed & (~np.isfinite(labels) | ((labels != 0) & (labels != 1)))))
    return np.where(observed, labels, np.nan).astype(np.float32, copy=False), observed.copy()


def _slots(value):
    _require(type(value) is tuple and len(value) == 9
             and all(type(index) is int for index in value) and len(set(value)) == 9
             and all(0 <= index < 48 for index in value))
    return value


def validate_receipt(metadata, *, arm, outer_fold, expected_fit_membership_sha256,
                     seed_index, paired_steps=1500, joint_steps=1500, batch_size=96):
    """Validate a closed, non-row-level receipt for one fresh fitting arm."""
    try:
        _require(type(arm) is str and arm in ('original', 'candidate')
                 and type(outer_fold) is int and outer_fold in range(5)
                 and type(seed_index) is int and 0 <= seed_index <= 5
                 and type(expected_fit_membership_sha256) is str
                 and len(expected_fit_membership_sha256) == 64
                 and all(value in '0123456789abcdef' for value in expected_fit_membership_sha256)
                 and type(paired_steps) is int and paired_steps > 0
                 and type(joint_steps) is int and joint_steps > 0
                 and type(batch_size) is int and batch_size > 0)
        expected = {
            'schema': 'bran-retinal-unified-refit-training-v1', 'arm': arm,
            'paired_seed': 1701 + seed_index, 'joint_seed': 92401 + seed_index,
            'paired_steps': paired_steps, 'joint_steps': joint_steps, 'batch_size': batch_size,
            'fit_membership_sha256': expected_fit_membership_sha256,
            'outer_fold': outer_fold, 'state_width': 192,
            'fresh_paired_initialization': True, 'fresh_joint_adaptation': True,
            'checkpoint_initialized': False,
        }
        _require(type(metadata) is dict and set(metadata) == set(expected)
                 and all(type(metadata[key]) is type(value) and metadata[key] == value
                         for key, value in expected.items()))
        return True
    except Exception:
        _fail()


def train_arm(inputs: RefitInputs, labels: np.ndarray, observed: np.ndarray,
              cbc_slots: tuple[int, ...], *, arm: str, outer_fold: int,
              patient_ids: list[str], folds: np.ndarray, fit_membership: dict,
              expected_fit_membership_sha256: str, seed_index: int,
              paired_steps: int = 1500, joint_steps: int = 1500,
              batch_size: int = 96):
    """Fit a newly initialized native-192 BRAN arm on the frozen fit subset."""
    try:
        import torch

        _require(type(seed_index) is int and 0 <= seed_index <= 5
                 and type(paired_steps) is int and paired_steps > 0
                 and type(joint_steps) is int and joint_steps > 0
                 and type(batch_size) is int and batch_size > 0)
        fit = _indices(inputs, patient_ids, folds, outer_fold, fit_membership,
                       expected_fit_membership_sha256)
        n = len(patient_ids)
        clinical, clinical_mask, retinal, retinal_mask, age = _arm_arrays(inputs, arm, n, fit)
        target, target_mask = _labels(labels, observed, n, fit)
        slots = _slots(cbc_slots)
        # Both existing kernels validate their full supplied arrays.  Supplying
        # a fresh fit-only local universe prevents any excluded row from
        # contributing to normalizers, objective weights, or stochastic draws.
        train = np.arange(len(fit), dtype=np.int64)
        paired_seed, joint_seed = 1701 + seed_index, 92401 + seed_index
        with torch.random.fork_rng(devices=[]):
            model = paired_train(clinical, clinical_mask, retinal, retinal_mask, age, train,
                                 seed=paired_seed, steps=paired_steps, batch_size=batch_size)
            model = adapt(model, clinical, clinical_mask, retinal, retinal_mask, age,
                          target, target_mask, train, slots, seed=joint_seed,
                          steps=joint_steps, batch_size=batch_size, candidate=True)
        _validate_model(model)
        _require(all(value.device.type == 'cpu' and bool(torch.isfinite(value).all())
                     for value in model.state_dict().values()))
        model.eval()
        metadata = {
            'schema': 'bran-retinal-unified-refit-training-v1', 'arm': arm,
            'paired_seed': paired_seed, 'joint_seed': joint_seed,
            'paired_steps': paired_steps, 'joint_steps': joint_steps, 'batch_size': batch_size,
            'fit_membership_sha256': expected_fit_membership_sha256,
            'outer_fold': outer_fold, 'state_width': 192,
            'fresh_paired_initialization': True, 'fresh_joint_adaptation': True,
            'checkpoint_initialized': False,
        }
        validate_receipt(metadata, arm=arm, outer_fold=outer_fold,
                         expected_fit_membership_sha256=expected_fit_membership_sha256,
                         seed_index=seed_index, paired_steps=paired_steps,
                         joint_steps=joint_steps, batch_size=batch_size)
        return model, metadata
    except Exception:
        _fail()
