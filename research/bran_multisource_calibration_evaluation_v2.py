"""Private six-context inference and fixed calibration; aggregate return only.

Caller authenticates sources, deterministic roles, checkpoints and transforms,
and owns the local FD-quiet/one-heavy-job boundary. No training or selection.
"""
import numpy as np
import torch

from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_multisource_batches_v2 import tensor, transform_hash
from bran_multisource_inference_v2 import completion_predictions
from bran_multisource_protocol_v2 import PARAMETERS
from bran_multisource_calibration_v2 import calibrate
import bran_multisource_calibration_metrics_v2 as metrics
import bran_multisource_outcomes_v2 as native

ARMS = ('mlp', 'token')
PATTERNS = tuple(PARAMETERS['completion_patterns'])


def require(ok):
    if not ok: raise ValueError('multisource_calibration_evaluation_failed')


def evaluate(paired, roles, provider, *, progress=None):
    torch.set_num_threads(2)
    n = len(paired.folds)
    require(type(roles) is np.ndarray and roles.shape == (n,) and roles.dtype.kind in 'iu'
            and np.isin(roles, (0, 1)).all())
    for fold in range(5):
        rows = paired.folds == fold
        require(rows.sum() >= 40 and np.count_nonzero(rows & (roles == 0)) == rows.sum() // 2)
    slots = tuple(paired.names.index(field) for field in CBC_FIELDS)
    eligible = np.zeros(59, bool); eligible[list(paired.eligible_indices)] = True
    target = paired.c[:, slots].copy()
    original_observed = (paired.cm & eligible[None, :])[:, slots]
    intended = {pattern: original_observed.copy() for pattern in PATTERNS}
    for pattern in PATTERNS:
        if pattern.startswith('red_cell'):
            intended[pattern][:, (5, 7, 8)] = False
    observed = {pattern: np.zeros((n, 9), bool) for pattern in PATTERNS}
    predictions = {pattern: {arm: np.full((n, 9), np.nan) for arm in ARMS} for pattern in PATTERNS}
    age = native.original_age(paired)
    for fold in range(5):
        rows = np.flatnonzero(paired.folds == fold)
        local_age = native.subset_age(age, rows)
        for arm in ARMS:
            if progress: progress({'phase': 'calibration_inference', 'arm': arm, 'fold': fold})
            model, transform = provider(arm, fold)
            require(model.arm == arm and model.cbc_indices == slots and transform.heldout_fold == fold
                    and transform_hash(transform) == transform_hash(paired.transforms[fold]))
            c, cm = transform.clinical(paired.c, paired.cm)
            r, rm = transform.retinal(paired.r, paired.rm)
            args = (tensor(c[rows]), tensor(cm[rows], torch.bool), tensor(r[rows]), tensor(rm[rows], torch.bool),
                    local_age, transform.age_mean, transform.age_scale)
            for pattern in PATTERNS:
                item = completion_predictions(model, *args, pattern, slots)
                require(np.array_equal(item.targetmask.numpy(), intended[pattern][rows]))
                support = item.scoring_target_mask.numpy()
                require(not (support & ~intended[pattern][rows]).any())
                if arm == 'mlp': observed[pattern][rows] = support
                else: require(np.array_equal(observed[pattern][rows], support))
                values = item.cbc_standardized.numpy()
                require(np.array_equal(np.isfinite(values), support))
                predictions[pattern][arm][rows] = (values * transform.clinical_iqr[list(slots)]
                                                  + transform.clinical_median[list(slots)])
            # Checkpoint audit reloads again; this is not a second encoder fit.
            del model
    if progress: progress({'phase': 'interval_fitting'})
    intervals = calibrate(target, observed, predictions, paired.folds, roles)
    if progress: progress({'phase': 'scoring_aggregate'})
    summaries = metrics.summarize(target, intended, observed, predictions, intervals, paired.folds, roles)
    metrics.validate_result(summaries)
    return {'schema': 'bran-multisource-cbc-calibration-v2',
        'status': 'diagnostic_intervals_completed_not_promoted', 'patterns': summaries,
        'interval_level': .9, 'minimum_observed_calibration': 20,
        'bootstrap_draws': 1000, 'bootstrap_seed': 94701, 'minimum_valid_draws': 900,
        'same_arm_fold_checkpoint_for_all_contexts': True,
        'radii_use_only_same_fold_calibration_roles': True, 'scoring_targets_used_to_fit_radii': False,
        'target_values_and_input_flags_erased': True, 'encoder_or_task_head_training': False,
        'failed_advancement_decisions_unchanged': True, 'candidate_promoted': False,
        'scientific_goal_achieved': False, 'clinical_coverage_guarantee': False,
        'diagnostic_reused_development_data': True, 'patient_level_output_emitted': False}
