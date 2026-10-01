"""Private fixed-model V5 interval evaluation; aggregate-only return.

No loaders, writes, fitting of network parameters, or patient-level logging.
Caller owns source authentication, FD suppression and the single-heavy-job lock.
"""
from __future__ import annotations

import copy
import numpy as np
import torch

from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_multisource_batches_v2 import tensor
from bran_multisource_inference_v2 import completion_predictions
from bran_multisource_outcomes_v2 import original_age, subset_age
from bran_multisource_profiles_v3 import _validate_provider, _unchanged
from bran_multisource_calibration_v2 import calibrate
import bran_multisource_calibration_metrics_v2 as metrics

ROLES = ('C', 'M')
PATTERNS = tuple(metrics._PATTERNS)
_LEGACY = {'C': 'mlp', 'M': 'token'}
_ERROR = 'v5_cbc_uncertainty_contract_failed'


def require(ok):
    if not ok:
        raise ValueError(_ERROR)


def remap_aggregates(value, *, to_legacy=False):
    """Map metric slot names only; both real V5 arms are anchored MLPs."""
    result = copy.deepcopy(value)
    mapping = _LEGACY if to_legacy else {v: k for k, v in _LEGACY.items()}
    require(set(result) == set(PATTERNS))
    for fields in result.values():
        require(set(fields) == set(CBC_FIELDS))
        for cell in fields.values():
            for group in cell['groups'].values():
                if 'arms' in group:
                    require(set(group['arms']) == set(mapping))
                    group['arms'] = {mapping[key]: item for key, item in group['arms'].items()}
    return result


def validate_patterns(value):
    metrics.validate_result(remap_aggregates(value, to_legacy=True))


def evaluate(paired, roles, provider, *, progress=None):
    """Same-checkpoint held-out calibration/scoring, with exact inference replay."""
    try:
        torch.set_num_threads(2)
        n = len(paired.folds)
        require(type(roles) is np.ndarray and roles.shape == (n,) and roles.dtype.kind in 'iu')
        metrics._validate_split(paired.folds, roles)
        require(callable(provider))
        slots = tuple(paired.names.index(field) for field in CBC_FIELDS)
        eligible = np.zeros(59, bool)
        eligible[list(paired.eligible_indices)] = True
        target = paired.c[:, slots].copy()
        original = (paired.cm & eligible[None, :])[:, slots]
        intended = {pattern: original.copy() for pattern in PATTERNS}
        for pattern in PATTERNS:
            if pattern.startswith('red_cell'):
                intended[pattern][:, (5, 7, 8)] = False
        support = {p: np.zeros((n, 9), bool) for p in PATTERNS}
        predictions = {p: {r: np.full((n, 9), np.nan) for r in ROLES} for p in PATTERNS}
        age = original_age(paired)
        seen = []
        for fold in range(5):
            rows = np.flatnonzero(paired.folds == fold)
            local_age = subset_age(age, rows)
            for role in ROLES:
                if progress:
                    progress({'phase': 'inference', 'fold': fold, 'role': role})
                model, transform = provider(role, fold)
                require(not any(model is previous for previous in seen))
                seen.append(model)
                before, transform_before, grads = _validate_provider(
                    model, transform, fold, slots, paired.transforms[fold])
                c, cm = transform.clinical(paired.c, paired.cm)
                r, rm = transform.retinal(paired.r, paired.rm)
                args = (tensor(c[rows]), tensor(cm[rows], torch.bool), tensor(r[rows]),
                        tensor(rm[rows], torch.bool), local_age, transform.age_mean, transform.age_scale)
                for pattern in PATTERNS:
                    item = completion_predictions(model, *args, pattern, slots)
                    mask = item.scoring_target_mask.numpy()
                    values = item.cbc_standardized.numpy()
                    require(np.array_equal(item.targetmask.numpy(), intended[pattern][rows]))
                    require(not (mask & ~intended[pattern][rows]).any())
                    require(np.array_equal(np.isfinite(values), mask))
                    if role == 'C':
                        support[pattern][rows] = mask
                    else:
                        require(np.array_equal(support[pattern][rows], mask))
                    predictions[pattern][role][rows] = (
                        values * transform.clinical_iqr[list(slots)] + transform.clinical_median[list(slots)])
                require(_unchanged(before, transform_before, grads, model, transform))
                if progress:
                    progress({'phase': 'checkpoint_replay', 'fold': fold, 'role': role})
                replay, rt = provider(role, fold)
                require(not any(replay is previous for previous in seen))
                seen.append(replay)
                rb, rh, rg = _validate_provider(replay, rt, fold, slots, paired.transforms[fold])
                require(rh == transform_before)
                require(before.keys() == rb.keys() and all(torch.equal(before[k], rb[k]) for k in before))
                for pattern in PATTERNS:
                    item = completion_predictions(replay, *args, pattern, slots)
                    require(np.array_equal(item.targetmask.numpy(), intended[pattern][rows]))
                    require(np.array_equal(item.scoring_target_mask.numpy(), support[pattern][rows]))
                    replay_values = (item.cbc_standardized.numpy() * rt.clinical_iqr[list(slots)]
                                     + rt.clinical_median[list(slots)])
                    require(np.array_equal(replay_values, predictions[pattern][role][rows], equal_nan=True))
                require(_unchanged(rb, rh, rg, replay, rt))
        # Legacy names are internal metric slots, NOT architecture declarations.
        internal = {p: {_LEGACY[r]: values for r, values in arms.items()}
                    for p, arms in predictions.items()}
        if progress:
            progress({'phase': 'interval_fitting'})
        intervals = calibrate(target, support, internal, paired.folds, roles)
        if progress:
            progress({'phase': 'aggregate_bootstrap'})
        legacy = metrics.summarize(target, intended, support, internal, intervals, paired.folds, roles)
        result = remap_aggregates(legacy)
        validate_patterns(result)
        return {'schema': 'bran-v5-cbc-uncertainty-v1', 'status': 'completed', 'patterns': result,
                'roles': {'C': 'paired_values_plus_source_masks', 'M': 'additional_source_values_features'},
                'interval_level': .9, 'bootstrap_draws': 1000, 'bootstrap_seed': 94701,
                'minimum_valid_draws': 900, 'minimum_calibration_values': 20,
                'same_checkpoint_heldout_calibration_and_scoring': True,
                'scoring_targets_used_to_fit_intervals': False,
                'checkpoint_predictions_replayed': True, 'model_parameters_unchanged': True,
                'encoder_or_task_head_training': False, 'historical_gates_changed': False,
                'clinical_coverage_guarantee': False, 'external_validation': False,
                'candidate_promoted': False, 'scientific_goal_achieved': False,
                'patient_level_output_emitted': False}
    except Exception:
        raise ValueError(_ERROR) from None
