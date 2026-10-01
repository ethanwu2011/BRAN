"""Aggregate-only fixed-cohort structure evaluation for age-free clinical fits."""
from __future__ import annotations

import numpy as np

import bran_agefree_unified_inference_v1 as inference
import bran_agefree_unified_jobs_v1 as jobs
import bran_joint_subtyping_kernel_v1 as joint


ERROR = 'age-free unified structure rejected'
ARMS = ('control', 'candidate')
MISSING_OUTER = 'unsupported_missing_available_outer_fold'


def _fail():
    raise ValueError(ERROR) from None


def _require(value):
    if not value:
        _fail()


def _bound(fitted, clinical, clinical_mask, profile_age, outer_folds, patient_ids,
           plan, plan_sha256, inner_folds):
    """Bind original-unit caller arrays to this discovery-only fitted frame."""
    try:
        _require(type(patient_ids) is list and patient_ids
                 and all(type(value) is str and value for value in patient_ids)
                 and len(set(patient_ids)) == len(patient_ids))
        n = len(patient_ids)
        _require(type(outer_folds) is np.ndarray and outer_folds.dtype == np.dtype(np.int64)
                 and outer_folds.shape == (n,) and set(outer_folds.tolist()) == set(range(5))
                 and type(clinical) is np.ndarray and clinical.dtype.kind == 'f'
                 and clinical.shape == (n, 59)
                 and type(clinical_mask) is np.ndarray and clinical_mask.dtype == np.dtype(bool)
                 and clinical_mask.shape == clinical.shape and np.isfinite(clinical[clinical_mask]).all()
                 and type(profile_age) is np.ndarray and profile_age.dtype.kind == 'f'
                 and profile_age.shape == (n,) and np.isfinite(profile_age).all())
        selected = jobs.membership.selected(plan, plan_sha256, 'structure', patient_ids=patient_ids,
                                            folds=outer_folds, inner_folds=inner_folds)
        metadata = fitted.bundle['metadata']
        paired, transform = fitted.inputs.paired, fitted.inputs.transform
        _require(metadata.get('job') == 'structure'
                 and metadata.get('membership_plan_sha256') == plan_sha256
                 and metadata.get('fit_membership_sha256') == selected['membership_sha256']
                 and np.array_equal(paired.fit_indices, np.asarray(selected['fit_indices'], dtype=np.int64)))
        rows, columns = np.nonzero(clinical_mask)
        expected = ((clinical[rows, columns] - transform.clinical_median[columns])
                    / transform.clinical_iqr[columns])
        _require(np.array_equal(clinical_mask, paired.clinical_mask)
                 and np.allclose(paired.clinical[rows, columns], expected, rtol=1e-6, atol=1e-6)
                 and np.allclose(paired.age, (profile_age - transform.age_mean) / transform.age_scale,
                                 rtol=1e-6, atol=1e-6))
        return selected
    except Exception:
        _fail()


def _validate_inputs(fitted, clinical, clinical_mask, profile_age, site_ids, outer_folds,
                     memberships, clinical_labels, label_observed, cgmmean, cgm_observed,
                     patient_ids, plan, plan_sha256, inner_folds):
    try:
        _require(type(fitted) is jobs.FittedJob and type(fitted.bundle.get('metadata')) is dict
                 and fitted.bundle['metadata'].get('job') == 'structure'
                 and type(fitted.bundle['metadata'].get('registry_names')) is list
                 and len(fitted.bundle['metadata']['registry_names']) == 59
                 and len(set(fitted.bundle['metadata']['registry_names'])) == 59
                 and all(type(name) is str and name for name in fitted.bundle['metadata']['registry_names']))
        selected = _bound(fitted, clinical, clinical_mask, profile_age, outer_folds, patient_ids,
                          plan, plan_sha256, inner_folds)
        n = len(patient_ids)
        _require(type(site_ids) in (tuple, list) and len(site_ids) == n
                 and all(type(value) is str and value for value in site_ids)
                 and type(memberships) is dict and set(memberships) == set(joint.CODES)
                 and all(type(value) is np.ndarray and value.dtype == np.dtype(bool)
                         and value.shape == (n,) for value in memberships.values())
                 and type(clinical_labels) is np.ndarray and clinical_labels.dtype.kind == 'f'
                 and clinical_labels.shape == (n, 26)
                 and type(label_observed) is np.ndarray and label_observed.dtype == np.dtype(bool)
                 and label_observed.shape == clinical_labels.shape
                 and np.all(~label_observed | (np.isfinite(clinical_labels)
                                                & ((clinical_labels == 0) | (clinical_labels == 1))))
                 and type(cgmmean) is np.ndarray and cgmmean.dtype.kind == 'f' and cgmmean.shape == (n,)
                 and type(cgm_observed) is np.ndarray and cgm_observed.dtype == np.dtype(bool)
                 and cgm_observed.shape == (n,)
                 and np.isfinite(cgmmean[cgm_observed]).all() and np.all(cgmmean[cgm_observed] > 0))
        return selected, tuple(fitted.bundle['metadata']['registry_names'])
    except Exception:
        _fail()


def evaluate(fitted, *, clinical59, clinical_mask, profile_age, site_ids, outer_folds,
             memberships, clinical_labels, label_observed, cgmmean, cgm_observed,
             patient_ids, plan, plan_sha256, inner_folds):
    """Evaluate unchanged 23-cohort aggregates in separate control/candidate frames."""
    try:
        snapshots = inference.snapshots(fitted)
        _selected, names = _validate_inputs(
            fitted, clinical59, clinical_mask, profile_age, site_ids, outer_folds,
            memberships, clinical_labels, label_observed, cgmmean, cgm_observed,
            patient_ids, plan, plan_sha256, inner_folds,
        )
        states = inference.structure_states(fitted)
        _require(np.array_equal(states.available['control'], states.available['candidate']))
        keep = states.available['control']
        result = {'schema': 'bran-agefree-unified-structure-v1', 'state_dimension': 192,
                  'coordinate_frames_separate': True, 'patient_arrays_serialized': False,
                  'novel_subtype_claimed': False, 'clinical_utility_established': False, 'arms': {}}
        if set(np.asarray(outer_folds)[keep].tolist()) != set(range(5)):
            for arm in ARMS:
                result['arms'][arm] = {'status': MISSING_OUTER}
        else:
            paired = fitted.inputs.paired
            for arm in ARMS:
                report = joint.evaluate(
                    states.outputs[arm][keep], clinical59[keep], clinical_mask[keep],
                    paired.retinal_present[keep], paired.age[keep], names,
                    tuple(site_ids[index] for index in np.flatnonzero(keep)), outer_folds[keep],
                    {code: memberships[code][keep] for code in joint.CODES},
                    clinical_labels[keep], label_observed[keep], cgmmean[keep], cgm_observed[keep],
                    normalized_clinical=paired.clinical[keep], profile_age=profile_age[keep],
                )
                joint.validate_result(report)
                result['arms'][arm] = {'status': 'evaluated', 'report': report}
        validate_result(result)
        inference.unchanged(fitted, snapshots)
        return result
    except Exception:
        _fail()


def validate_result(value):
    try:
        _require(type(value) is dict and set(value) == {
            'schema', 'state_dimension', 'coordinate_frames_separate', 'patient_arrays_serialized',
            'novel_subtype_claimed', 'clinical_utility_established', 'arms',
        } and value['schema'] == 'bran-agefree-unified-structure-v1'
                 and type(value['state_dimension']) is int and value['state_dimension'] == 192
                 and value['coordinate_frames_separate'] is True
                 and all(value[key] is False for key in
                         ('patient_arrays_serialized', 'novel_subtype_claimed', 'clinical_utility_established'))
                 and type(value['arms']) is dict and set(value['arms']) == set(ARMS)
                 and value['arms']['control'].get('status') == value['arms']['candidate'].get('status'))
        for item in value['arms'].values():
            _require(type(item) is dict and type(item.get('status')) is str)
            if item['status'] == MISSING_OUTER:
                _require(set(item) == {'status'})
            else:
                _require(item['status'] == 'evaluated' and set(item) == {'status', 'report'})
                joint.validate_result(item['report'])
        return True
    except Exception:
        _fail()
