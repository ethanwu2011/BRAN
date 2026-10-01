"""Private, array-only matched refit coordinator and strict checkpoint restore.

The real-data launcher must authenticate source bytes, complete retinal audit,
prospective memberships and protocol before this layer, and own quiet/lock/I/O.
This module does not grant source access, evaluate efficacy, or promote BRAN.
"""
import copy
from dataclasses import dataclass, field

import numpy as np
import torch

import bran_retinal_refit_inputs_v2 as handoff
import bran_retinal_unified_refit_membership_v1 as membership
import bran_retinal_unified_refit_training_v1 as training
from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
from bran_patient_state_prototype_v1 import PatientStateConfig
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_retinal_multisource_encoder_binding_v1 import pin

ARMS = ('original', 'candidate')
NORMALIZERS = ('clinical_median', 'clinical_iqr', 'retinal_mean', 'retinal_scale', 'age_mean', 'age_scale')
DATA_FIELDS = {'records', 'image_features', 'patient_ids', 'folds', 'selection', 'candidate_contract',
    'expected_checkpoint_sha256', 'original_retinal', 'retinal_present', 'clinical', 'observed', 'eligible', 'age'}
ERROR = 'matched unified refit job rejected'


@dataclass(frozen=True, repr=False)
class FittedJob:
    models: dict
    bundle: dict
    inputs: object
    _metadata_sha256: str = field(init=False, repr=False)

    def __post_init__(self):
        # Keeps an accidental later relabelling of an outer-fold model as the
        # discovery-only structure fit from passing the inference interface.
        object.__setattr__(self, '_metadata_sha256', handoff.digest(self.bundle['metadata']))


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def context(data, plan, plan_sha256, job, inner_folds, registry_names, endpoint_names):
    require(type(data) is dict and set(data) == DATA_FIELDS)
    require(type(registry_names) is tuple and len(registry_names) == len(set(registry_names)) == 59
            and all(type(name) is str and name for name in registry_names))
    require(type(endpoint_names) is tuple and len(endpoint_names) == len(set(endpoint_names)) == 26
            and all(type(name) is str and name for name in endpoint_names))
    slots = tuple(registry_names.index(field) for field in CBC_FIELDS)
    require(all(slot < 48 for slot in slots))
    selected = membership.selected(plan, plan_sha256, job, patient_ids=data['patient_ids'],
        folds=data['folds'], inner_folds=inner_folds)
    prepared = handoff.prepare_fold(**data, outer_fold=selected['outer_fold'],
        fit_indices=np.asarray(selected['fit_indices'], dtype=np.int64),
        fit_membership=selected['membership'], expected_fit_membership_sha256=selected['membership_sha256'])
    return selected, prepared, slots


def metadata(protocol_sha256, plan_sha256, job, selected, data, registry_names, endpoint_names):
    pin(protocol_sha256); pin(plan_sha256)
    return {'schema': 'bran-retinal-unified-refit-job-v1', 'protocol_sha256': protocol_sha256,
        'membership_plan_sha256': plan_sha256, 'fit_membership_sha256': selected['membership_sha256'],
        'candidate_contract_sha256': handoff.digest(data['candidate_contract']), 'job': job,
        'registry_names': list(registry_names), 'endpoint_names': list(endpoint_names), 'cbc_fields': list(CBC_FIELDS),
        'state_width': 192, 'shared_state_across_tasks': True, 'fresh_matched_initialization': True,
        'cross_checkpoint_state_pooling_permitted': False, 'clinical_evaluation_completed': False,
        'unified_model_promoted': False, 'patient_level_output_emitted': False}


def normalizers(prepared, arm):
    transform = getattr(prepared, arm + '_transform')
    return {name: copy.deepcopy(getattr(transform, name)) for name in NORMALIZERS}


def _same_normalizers(actual, expected):
    require(type(actual) is dict and set(actual) == set(expected))
    for name, value in expected.items():
        saved = actual[name]
        require(type(saved) is type(value) and np.array_equal(saved, value))
        if isinstance(value, np.ndarray):
            require(saved.dtype == value.dtype and saved.shape == value.shape and np.isfinite(saved).all())
        else:
            require(np.isfinite(saved))


def _new_model():
    # Restore does not perturb later training RNG streams.
    with torch.random.fork_rng(devices=[]):
        model = BRANClinicalAnchorV2(PatientStateConfig())
        model.screening_joint_head = torch.nn.Linear(192, 26)
        model.cbc_joint_head = torch.nn.Linear(192, 9)
    return model


def restore_job(bundle, *, data, plan, plan_sha256, job, inner_folds, registry_names,
                endpoint_names, protocol_sha256, paired_steps=1500, joint_steps=1500, batch_size=96):
    """Restore already byte-authenticated private bundle; no representation refit."""
    try:
        selected, prepared, _ = context(data, plan, plan_sha256, job, inner_folds, registry_names, endpoint_names)
        expected = metadata(protocol_sha256, plan_sha256, job, selected, data, registry_names, endpoint_names)
        require(type(bundle) is dict and set(bundle) == {'metadata', 'arms'}
                and handoff.digest(bundle['metadata']) == handoff.digest(expected)
                and type(bundle['arms']) is dict and set(bundle['arms']) == set(ARMS))
        models = {}
        for arm in ARMS:
            saved = bundle['arms'][arm]
            require(type(saved) is dict and set(saved) == {'state_dict', 'normalizers', 'receipt'})
            training.validate_receipt(saved['receipt'], arm=arm, outer_fold=selected['outer_fold'],
                expected_fit_membership_sha256=selected['membership_sha256'], seed_index=selected['seed_index'],
                paired_steps=paired_steps, joint_steps=joint_steps, batch_size=batch_size)
            _same_normalizers(saved['normalizers'], normalizers(prepared, arm))
            model = _new_model()
            expected_state = model.state_dict()
            require(isinstance(saved['state_dict'], dict) and set(saved['state_dict']) == set(expected_state))
            for key, tensor in saved['state_dict'].items():
                require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu'
                        and tensor.shape == expected_state[key].shape and tensor.dtype == expected_state[key].dtype
                        and bool(torch.isfinite(tensor).all()))
            model.load_state_dict(saved['state_dict'], strict=True)
            model.eval()
            models[arm] = model
        return FittedJob(models, bundle, prepared)
    except Exception:
        raise ValueError(ERROR) from None


def fit_job(*, data, labels, label_observed, plan, plan_sha256, job, inner_folds,
            registry_names, endpoint_names, protocol_sha256, paired_steps=1500,
            joint_steps=1500, batch_size=96):
    """Fit both encoders' patient models with identical memberships and budgets."""
    try:
        selected, prepared, slots = context(data, plan, plan_sha256, job, inner_folds, registry_names, endpoint_names)
        bundle = {'metadata': metadata(protocol_sha256, plan_sha256, job, selected, data, registry_names, endpoint_names),
                  'arms': {}}
        fitted = {}
        for arm in ARMS:
            model, receipt = training.train_arm(prepared, labels, label_observed, slots,
                arm=arm, outer_fold=selected['outer_fold'], patient_ids=data['patient_ids'], folds=data['folds'],
                fit_membership=selected['membership'], expected_fit_membership_sha256=selected['membership_sha256'],
                seed_index=selected['seed_index'], paired_steps=paired_steps, joint_steps=joint_steps, batch_size=batch_size)
            fitted[arm] = model
            bundle['arms'][arm] = {'state_dict': copy.deepcopy(model.state_dict()),
                                  'normalizers': normalizers(prepared, arm), 'receipt': receipt}
        restored = restore_job(bundle, data=data, plan=plan, plan_sha256=plan_sha256, job=job,
            inner_folds=inner_folds, registry_names=registry_names, endpoint_names=endpoint_names,
            protocol_sha256=protocol_sha256, paired_steps=paired_steps, joint_steps=joint_steps, batch_size=batch_size)
        for arm in ARMS:
            require(all(torch.equal(value, restored.models[arm].state_dict()[key])
                        for key, value in fitted[arm].state_dict().items()))
        return restored
    except Exception:
        raise ValueError(ERROR) from None
