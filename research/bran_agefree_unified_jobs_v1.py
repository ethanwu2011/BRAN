"""Private array-only matched age-free clinical job coordination.

Source authentication, artifact I/O, and experiment authorization remain with
the caller.  This module fits or restores only caller-supplied private arrays.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

import bran_agefree_source_sampling_v1 as sampling
import bran_agefree_unified_source_v1 as source
import bran_agefree_unified_training_v1 as training
import bran_retinal_refit_inputs_v2 as handoff
import bran_retinal_unified_refit_membership_v1 as membership
from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2, PatientStateConfig
from bran_clinical_semantics_v1 import CBC_FIELDS
from run_bran_overnight_diagnostic_v1 import FoldTransform


ERROR = 'age-free unified job rejected'
ARMS = ('control', 'candidate')
NORMALIZERS = ('clinical_median', 'clinical_iqr', 'retinal_mean', 'retinal_scale',
               'age_mean', 'age_scale')
DATA_FIELDS = {'patient_ids', 'folds', 'clinical', 'observed', 'eligible', 'retinal',
               'retinal_present', 'age'}


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def _sha(value):
    return type(value) is str and len(value) == 64 and all(item in '0123456789abcdef' for item in value)


def _readonly(value, dtype=None):
    result = np.array(value, copy=True, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, repr=False)
class JobInputs:
    transform: object
    paired: training.PairedInputs


@dataclass(frozen=True, repr=False)
class FittedJob:
    models: dict
    bundle: dict
    inputs: JobInputs
    _metadata_sha256: str = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, '_metadata_sha256', handoff.digest(self.bundle['metadata']))


def _data(data):
    require(type(data) is dict and set(data) == DATA_FIELDS)
    ids, folds = data['patient_ids'], data['folds']
    require(type(ids) is list and ids and all(type(item) is str and item for item in ids)
            and len(set(ids)) == len(ids) and type(folds) is np.ndarray
            and folds.dtype == np.dtype(np.int64) and folds.shape == (len(ids),)
            and set(folds.tolist()) == set(range(5)))
    n = len(ids)
    clinical, observed, eligible = data['clinical'], data['observed'], data['eligible']
    retinal, retinal_present, age = data['retinal'], data['retinal_present'], data['age']
    require(type(clinical) is np.ndarray and clinical.dtype.kind == 'f' and clinical.shape == (n, 59)
            and type(observed) is np.ndarray and observed.dtype == np.dtype(bool)
            and observed.shape == clinical.shape and type(eligible) is np.ndarray
            and eligible.dtype == np.dtype(bool) and eligible.shape == clinical.shape
            and type(retinal) is np.ndarray and retinal.dtype.kind == 'f' and retinal.shape == (n, 384)
            and type(retinal_present) is np.ndarray and retinal_present.dtype == np.dtype(bool)
            and retinal_present.shape == (n,) and type(age) is np.ndarray and age.dtype.kind == 'f'
            and age.shape == (n,) and np.isfinite(age).all()
            and np.isfinite(clinical[observed & eligible]).all()
            and np.isfinite(retinal[retinal_present]).all())


def normalizers(transform):
    try:
        result = {name: copy.deepcopy(getattr(transform, name)) for name in NORMALIZERS}
        for name, value in result.items():
            if isinstance(value, np.ndarray):
                require(value.dtype.kind == 'f' and value.shape == ((59,) if name.startswith('clinical') else (384,))
                        and np.isfinite(value).all())
            else:
                require(type(value) in (float, np.float32, np.float64) and np.isfinite(value))
        return result
    except Exception:
        raise ValueError(ERROR) from None


def _same_normalizers(actual, expected):
    require(type(actual) is dict and set(actual) == set(expected))
    for name, value in expected.items():
        saved = actual[name]
        require(type(saved) is type(value) and np.array_equal(saved, value))
        if isinstance(value, np.ndarray):
            require(saved.dtype == value.dtype and saved.shape == value.shape and np.isfinite(saved).all())
        else:
            require(np.isfinite(saved))


def context(data, plan, plan_sha256, job, inner_folds, registry_names, endpoint_names,
            *, external_sources=None):
    """Bind a prospective job and its train-only transform.

    Supplying ``external_sources`` additionally prepares the new age-free
    sampler; restore callers intentionally omit it.
    """
    try:
        _data(data)
        require(type(registry_names) is tuple and len(registry_names) == len(set(registry_names)) == 59
                and all(type(name) is str and name for name in registry_names)
                and type(endpoint_names) is tuple and len(endpoint_names) == len(set(endpoint_names)) == 26
                and all(type(name) is str and name for name in endpoint_names))
        slots = tuple(registry_names.index(field) for field in CBC_FIELDS)
        require(all(slot < 48 for slot in slots))
        selected = membership.selected(plan, plan_sha256, job, patient_ids=data['patient_ids'],
                                       folds=data['folds'], inner_folds=inner_folds)
        fit = np.asarray(selected['fit_indices'], dtype=np.int64)
        transform = FoldTransform(data['clinical'], data['observed'], data['eligible'], data['retinal'],
                                  data['retinal_present'], data['age'], fit)
        clinical, clinical_mask, retinal, age = transform.apply(
            data['clinical'], data['observed'], data['eligible'], data['retinal'],
            data['retinal_present'], data['age'],
        )
        require(np.isfinite(clinical).all() and np.isfinite(retinal).all() and np.isfinite(age).all()
                and np.all(clinical[~clinical_mask] == 0) and np.all(retinal[~data['retinal_present']] == 0))
        # Retain transform precision for exact historical raw-feature replay.
        # Neural training/inference performs its own explicit float32 conversion.
        paired = training.PairedInputs(_readonly(clinical), _readonly(clinical_mask, bool),
                                       _readonly(retinal), _readonly(data['retinal_present'], bool),
                                       _readonly(age), _readonly(fit, np.int64))
        inputs = JobInputs(transform, paired)
        prepared = None
        if external_sources is not None:
            require(type(external_sources) is source.PrivateClinicalSources)
            prepared = sampling.prepare(external_sources.three_sources, external_sources.nwicu_adapted,
                                        registry_names, transform.clinical_median, transform.clinical_iqr)
            require(prepared.registry_fields == registry_names
                    and np.array_equal(prepared.medians, transform.clinical_median)
                    and np.array_equal(prepared.iqrs, transform.clinical_iqr))
        return selected, inputs, slots, prepared
    except Exception:
        raise ValueError(ERROR) from None


def metadata(protocol_sha256, source_descriptor_sha256, plan_sha256, job, selected,
             registry_names, endpoint_names):
    require(_sha(protocol_sha256) and _sha(source_descriptor_sha256) and _sha(plan_sha256)
            and type(job) is str and type(selected) is dict and _sha(selected.get('membership_sha256')))
    return {'schema': 'bran-agefree-unified-job-v1', 'protocol_sha256': protocol_sha256,
            'source_descriptor_sha256': source_descriptor_sha256, 'membership_plan_sha256': plan_sha256,
            'fit_membership_sha256': selected['membership_sha256'], 'job': job,
            'registry_names': list(registry_names), 'endpoint_names': list(endpoint_names),
            'cbc_fields': list(CBC_FIELDS), 'state_width': 192, 'shared_state_across_tasks': True,
            'agefree_external_clinical_warm_start': True, 'external_scalar_age_used': False,
            'retinal_inputs_changed': False, 'retained_checkpoint_initialized': False,
            'clinical_evaluation_completed': False, 'unified_model_promoted': False,
            'patient_level_output_emitted': False}


def _new_model():
    with torch.random.fork_rng(devices=[]):
        model = BRANClinicalAnchorV2(PatientStateConfig())
        model.screening_joint_head = torch.nn.Linear(192, 26)
        model.cbc_joint_head = torch.nn.Linear(192, 9)
    return model


def _model_state(saved):
    model = _new_model()
    expected = model.state_dict()
    require(isinstance(saved, dict) and set(saved) == set(expected))
    for name, tensor in saved.items():
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu'
                and tensor.shape == expected[name].shape and tensor.dtype == expected[name].dtype
                and bool(torch.isfinite(tensor).all()))
    model.load_state_dict(saved, strict=True)
    training._validate_model(model)
    model.eval()
    return model


def restore_job(bundle, *, data, plan, plan_sha256, job, inner_folds, registry_names,
                endpoint_names, protocol_sha256, source_descriptor_sha256, external_steps=3000,
                paired_steps=1500, joint_steps=1500, batch_size=96):
    """Restore only already-authenticated private state; never prepares sources."""
    try:
        selected, inputs, _slots, prepared = context(data, plan, plan_sha256, job, inner_folds,
                                                      registry_names, endpoint_names)
        require(prepared is None)
        expected = metadata(protocol_sha256, source_descriptor_sha256, plan_sha256, job,
                            selected, registry_names, endpoint_names)
        require(type(bundle) is dict and set(bundle) == {'metadata', 'arms'}
                and handoff.digest(bundle['metadata']) == handoff.digest(expected)
                and type(bundle['arms']) is dict and set(bundle['arms']) == set(ARMS))
        models = {}
        for arm in ARMS:
            saved = bundle['arms'][arm]
            require(type(saved) is dict and set(saved) == {'state_dict', 'normalizers', 'receipt'})
            training.validate_receipt(saved['receipt'], arm=arm, outer_fold=selected['outer_fold'],
                                      expected_fit_membership_sha256=selected['membership_sha256'],
                                      seed_index=selected['seed_index'], external_steps=external_steps,
                                      paired_steps=paired_steps, joint_steps=joint_steps, batch_size=batch_size)
            _same_normalizers(saved['normalizers'], normalizers(inputs.transform))
            models[arm] = _model_state(saved['state_dict'])
        return FittedJob(models, bundle, inputs)
    except Exception:
        raise ValueError(ERROR) from None


def fit_job(*, data, external_sources, labels, label_observed, plan, plan_sha256, job,
            inner_folds, registry_names, endpoint_names, protocol_sha256, source_descriptor_sha256,
            external_steps=3000, paired_steps=1500, joint_steps=1500, batch_size=96):
    """Fit both fresh matched arms, then restore their private bundle exactly."""
    try:
        selected, inputs, slots, prepared = context(data, plan, plan_sha256, job, inner_folds,
                                                     registry_names, endpoint_names,
                                                     external_sources=external_sources)
        bundle = {'metadata': metadata(protocol_sha256, source_descriptor_sha256, plan_sha256, job,
                                       selected, registry_names, endpoint_names), 'arms': {}}
        fitted = {}
        for arm in ARMS:
            model, receipt = training.train_arm(
                inputs.paired, labels, label_observed, prepared, registry_names, arm=arm,
                outer_fold=selected['outer_fold'], patient_ids=data['patient_ids'], folds=data['folds'],
                fit_membership=selected['membership'], expected_fit_membership_sha256=selected['membership_sha256'],
                seed_index=selected['seed_index'], external_steps=external_steps,
                paired_steps=paired_steps, joint_steps=joint_steps, batch_size=batch_size,
            )
            fitted[arm] = model
            bundle['arms'][arm] = {'state_dict': copy.deepcopy(model.state_dict()),
                                   'normalizers': normalizers(inputs.transform), 'receipt': receipt}
        restored = restore_job(bundle, data=data, plan=plan, plan_sha256=plan_sha256, job=job,
                               inner_folds=inner_folds, registry_names=registry_names,
                               endpoint_names=endpoint_names, protocol_sha256=protocol_sha256,
                               source_descriptor_sha256=source_descriptor_sha256,
                               external_steps=external_steps, paired_steps=paired_steps,
                               joint_steps=joint_steps, batch_size=batch_size)
        for arm in ARMS:
            require(all(torch.equal(value, restored.models[arm].state_dict()[name])
                        for name, value in fitted[arm].state_dict().items()))
        return restored
    except Exception:
        raise ValueError(ERROR) from None
