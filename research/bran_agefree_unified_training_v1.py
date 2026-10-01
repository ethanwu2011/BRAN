"""Private-array three/four-source transfer into one native BRAN representation.

Caller owns source authentication, fit-only normalization, prospective membership,
the shared execution lock and FD silence. This module performs no I/O and emits
no measurements, states or losses. Its receipt is not scientific validation.
"""
from dataclasses import dataclass

import numpy as np

import bran_agefree_source_sampling_v1 as sampling
import bran_joint_lab_pretraining_v1 as external
from bran_external_cbc_fit_kernel_v1 import paired_train
from bran_external_cbc_pretraining_v1 import (
    export_clinical_warm_start, make_cbc_pretrain_head, make_cbc_pretrain_optimizer,
)
from bran_joint_lab_task_contract_v1 import build_task_masks
from bran_raw_teacher_distillation_v1 import _validate_model
from bran_retinal_refit_inputs_v2 import validate_fit_membership
from bran_screening_joint_kernel_v1 import adapt

ERROR = 'age-free unified training rejected'
ARMS = ('control', 'candidate')


def require(value):
    if not value:
        raise ValueError(ERROR) from None


@dataclass(frozen=True, repr=False)
class PairedInputs:
    clinical: np.ndarray
    clinical_mask: np.ndarray
    retinal: np.ndarray
    retinal_present: np.ndarray
    age: np.ndarray
    fit_indices: np.ndarray


def fit_arrays(inputs, labels, observed, *, patient_ids, folds, outer_fold,
               fit_membership, expected_fit_membership_sha256):
    """Authenticate membership, then physically exclude all other patients."""
    require(type(inputs) is PairedInputs and type(patient_ids) is list and patient_ids
            and all(type(x) is str and x for x in patient_ids)
            and len(set(patient_ids)) == len(patient_ids))
    n, fit = len(patient_ids), inputs.fit_indices
    require(type(folds) is np.ndarray and folds.dtype == np.int64 and folds.shape == (n,)
            and set(folds.tolist()) == set(range(5)) and type(outer_fold) is int
            and outer_fold in range(5) and type(fit) is np.ndarray and fit.dtype == np.int64
            and fit.ndim == 1 and len(fit) > 0 and np.all((fit >= 0) & (fit < n))
            and np.all(np.diff(fit) > 0) and np.all(folds[fit] != outer_fold))
    validate_fit_membership(fit_membership, expected_fit_membership_sha256,
        patient_ids=patient_ids, folds=folds, outer_fold=outer_fold, fit_indices=fit)
    arrays = (inputs.clinical, inputs.clinical_mask, inputs.retinal,
              inputs.retinal_present, inputs.age, labels, observed)
    shapes = ((n, 59), (n, 59), (n, 384), (n,), (n,), (n, 26), (n, 26))
    for j, (value, shape) in enumerate(zip(arrays, shapes)):
        require(type(value) is np.ndarray and value.shape == shape)
        require(value.dtype == bool if j in (1, 3, 6) else value.dtype.kind == 'f')
    c, cm, r, rm, age, y, ym = (value[fit].copy() for value in arrays)
    require(np.isfinite(c[cm]).all() and np.isfinite(r[rm]).all()
            and np.isfinite(age).all() and np.isfinite(y[ym]).all()
            and np.isin(y[ym], (0, 1)).all())
    return (np.where(cm, c, 0.).astype(np.float32), cm,
            np.where(rm[:, None], r, 0.).astype(np.float32), rm,
            age.astype(np.float32), np.where(ym, y, np.nan).astype(np.float32), ym)


def settings(arm, seed_index, external_steps, paired_steps, joint_steps, batch_size):
    require(type(arm) is str and arm in ARMS and type(seed_index) is int
            and 0 <= seed_index <= 5 and all(type(x) is int and x > 0
                for x in (external_steps, paired_steps, joint_steps, batch_size))
            and external_steps % 2 == 0)


def warm_start(prepared, registry_fields, *, arm, seed_index,
               external_steps=3000, batch_size=96):
    """Age-disabled clinical learning; separate whole/partial stochastic streams."""
    import torch
    from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
    from bran_patient_state_prototype_v1 import PatientStateConfig

    settings(arm, seed_index, external_steps, 1, 1, batch_size)
    external._registry_slots(registry_fields)
    require(type(prepared) is sampling.Prepared and prepared.registry_fields == registry_fields)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1701 + seed_index)
        model = BRANClinicalAnchorV2(PatientStateConfig())
        head = make_cbc_pretrain_head(model)
        optimizer = make_cbc_pretrain_optimizer(model, head, learning_rate=.001)
        # NWICU changes only the partial source mix. Its random consumption
        # cannot change which whole-CBC patients or masks the other arm sees.
        sample_rng = {mode: np.random.default_rng(seed + seed_index)
                      for mode, seed in (('whole_cbc', 19001), ('partial_cbc', 29001))}
        mask_rng = {mode: np.random.default_rng(seed + seed_index)
                    for mode, seed in (('whole_cbc', 39001), ('partial_cbc', 49001))}
        for step in range(external_steps):
            mode = 'whole_cbc' if step % 2 == 0 else 'partial_cbc'
            values, mask, observed = sampling.sample(prepared, arm, mode, batch_size, sample_rng[mode])
            visible, hidden = build_task_masks(observed, mode, mask_rng[mode])
            external.joint_pretrain_step(model, head, optimizer, values, mask,
                visible, hidden, registry_fields, mode=mode)
        return export_clinical_warm_start(model)


def receipt(*, arm, outer_fold, expected_fit_membership_sha256, seed_index,
            external_steps=3000, paired_steps=1500, joint_steps=1500, batch_size=96):
    settings(arm, seed_index, external_steps, paired_steps, joint_steps, batch_size)
    require(type(outer_fold) is int and outer_fold in range(5)
            and type(expected_fit_membership_sha256) is str
            and len(expected_fit_membership_sha256) == 64
            and all(x in '0123456789abcdef' for x in expected_fit_membership_sha256))
    return {'schema': 'bran-agefree-unified-training-v1', 'arm': arm,
        'external_sources': ['mimic', 'nhanes', 'eicu'] + (['nwicu'] if arm == 'candidate' else []),
        'external_steps': external_steps, 'paired_steps': paired_steps,
        'joint_steps': joint_steps, 'batch_size': batch_size,
        'external_seed': 1701 + seed_index, 'paired_seed': 1701 + seed_index,
        'joint_seed': 92401 + seed_index, 'outer_fold': outer_fold,
        'fit_membership_sha256': expected_fit_membership_sha256, 'state_width': 192,
        'external_scalar_age_used': False, 'nwicu_whole_cbc_used': False,
        'external_clinical_warm_start': True, 'temporary_external_head_retained': False,
        'retinal_inputs_changed': False, 'shared_state_across_tasks': True,
        'retained_checkpoint_initialized': False, 'patient_level_output_emitted': False}


def validate_receipt(value, **kwargs):
    expected = receipt(**kwargs)
    require(type(value) is dict and set(value) == set(expected)
            and all(type(value[k]) is type(v) and value[k] == v for k, v in expected.items()))
    return True


def train_arm(inputs, labels, observed, prepared, registry_fields, *, arm, outer_fold,
              patient_ids, folds, fit_membership, expected_fit_membership_sha256,
              seed_index, external_steps=3000, paired_steps=1500,
              joint_steps=1500, batch_size=96):
    """Transfer clinical weights, learn paired alignment, attach native task heads.

    No external rows, temporary-head outputs or auxiliary states are appended to
    the native192 representation. Scalar-age input weights are initialized from
    the paired seed, not inferred from the age-free external phase.
    """
    try:
        import torch
        configuration = dict(arm=arm, outer_fold=outer_fold,
            expected_fit_membership_sha256=expected_fit_membership_sha256, seed_index=seed_index,
            external_steps=external_steps, paired_steps=paired_steps,
            joint_steps=joint_steps, batch_size=batch_size)
        metadata = receipt(**configuration)
        slots = external._registry_slots(registry_fields)[:9]
        c, cm, r, rm, age, y, ym = fit_arrays(inputs, labels, observed,
            patient_ids=patient_ids, folds=folds, outer_fold=outer_fold,
            fit_membership=fit_membership,
            expected_fit_membership_sha256=expected_fit_membership_sha256)
        train = np.arange(len(c), dtype=np.int64)
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            weights = warm_start(prepared, registry_fields, arm=arm, seed_index=seed_index,
                                 external_steps=external_steps, batch_size=batch_size)
            model = paired_train(c, cm, r, rm, age, train, seed=1701 + seed_index,
                steps=paired_steps, batch_size=batch_size, warm_start=weights)
            model = adapt(model, c, cm, r, rm, age, y, ym, train, slots,
                seed=92401 + seed_index, steps=joint_steps, batch_size=batch_size, candidate=True)
        _validate_model(model)
        require(all(v.device.type == 'cpu' and bool(torch.isfinite(v).all())
                    for v in model.state_dict().values()))
        model.eval()
        validate_receipt(metadata, **configuration)
        return model, metadata
    except Exception:
        raise ValueError(ERROR) from None
