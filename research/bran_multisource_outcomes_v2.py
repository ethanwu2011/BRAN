"""Held-out native candidate profiling; caller owns source and checkpoint auth.

All patient arrays stay in this local call and are never returned. This does
not train, pick epochs, calibrate, or promote a model. The same arm/fold model
is used for all routes, stress patterns and target-erased CBC contexts.
"""
import numpy as np
import torch

from bran_multisource_age_v2 import AgeBatch, validate_age
from bran_multisource_batches_v2 import tensor, transform_hash
from bran_multisource_inference_v2 import infer_native, route_predictions, completion_predictions
from bran_multisource_protocol_v2 import PARAMETERS
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_external_cbc_evaluation_v1 import paired_counts
import bran_missingness_stress_v1 as masking
import bran_missingness_stress_metrics_v1 as stress_metrics
import bran_multisource_outcome_metrics_v2 as metrics

ARMS = ('mlp', 'token')
AGE_SCENARIOS = ('age_unknown', 'age_five_year_interval', 'all_physiology_hidden')


def require(ok):
    if not ok: raise ValueError('multisource_outcomes_failed')


def original_age(paired):
    missing = np.full(len(paired.folds), np.nan)
    age = AgeBatch(tensor(paired.age_value), tensor(getattr(paired, 'age_lower', missing)),
                   tensor(getattr(paired, 'age_upper', missing)), tensor(paired.age_kind, torch.long))
    return validate_age(age)


def age_scenario(age, scenario):
    require(scenario in AGE_SCENARIOS)
    values = {key: getattr(age, key).clone() for key in ('value', 'lower', 'upper', 'kind')}
    if scenario == 'age_unknown':
        for key in ('value', 'lower', 'upper'): values[key].fill_(float('nan'))
        values['kind'].fill_(3)
    elif scenario == 'age_five_year_interval':
        reported = values['kind'] == 0
        lower = torch.floor(values['value'][reported] / 5.) * 5.
        values['value'][reported] = float('nan')
        values['lower'][reported] = lower; values['upper'][reported] = lower + 5.
        values['kind'][reported] = 1
    return validate_age(AgeBatch(**values))


def subset_age(age, rows):
    return AgeBatch(*(getattr(age, key)[rows] for key in ('value', 'lower', 'upper', 'kind')))


def evaluate(paired, provider, *, progress=None):
    """provider(arm, fold) -> (authenticated frozen model, saved transform)."""
    torch.set_num_threads(2)
    n = len(paired.folds); require(n >= 100 and len(paired.endpoint_names) == 26)
    slots = tuple(paired.names.index(field) for field in CBC_FIELDS)
    eligible = np.zeros(59, bool); eligible[list(paired.eligible_indices)] = True
    target = paired.c[:, slots].copy(); observed = (paired.cm & eligible[None, :])[:, slots]
    age = original_age(paired)
    screen = {arm+'_'+route: np.full((n, 26), np.nan)
              for arm in ARMS for route in ('both', 'clinical', 'retinal')}
    stress = {arm: {pattern: np.full((n, 26), np.nan) for pattern in masking.PATTERNS} for arm in ARMS}
    ages = {scenario: {arm: np.full((n, 26), np.nan) for arm in ARMS} for scenario in AGE_SCENARIOS}
    cbc = {pattern: {arm: np.full((n, 9), np.nan) for arm in ARMS} for pattern in PARAMETERS['completion_patterns']}
    intended = {pattern: observed.copy() for pattern in cbc}
    for pattern in cbc:
        if pattern.startswith('red_cell'):
            intended[pattern][:, (5, 7, 8)] = False
    for fold in range(5):
        rows = np.flatnonzero(paired.folds == fold)
        require(len(rows) > 0)
        local_age = subset_age(age, rows)
        for arm in ARMS:
            if progress: progress({'phase':'candidate_inference', 'arm':arm, 'fold':fold})
            model, transform = provider(arm, fold)
            require(model.arm == arm and model.cbc_indices == slots
                    and transform.heldout_fold == fold
                    and transform_hash(transform) == transform_hash(paired.transforms[fold]))
            c, cm = transform.clinical(paired.c, paired.cm)
            r, rm = transform.retinal(paired.r, paired.rm)
            args = (model, tensor(c[rows]), tensor(cm[rows], torch.bool),
                    tensor(r[rows]), tensor(rm[rows], torch.bool), local_age,
                    transform.age_mean, transform.age_scale)
            predictions = route_predictions(*args)
            for route, item in predictions.items():
                screen[arm+'_'+route][rows] = item.screening_probability.numpy()
            for pattern in cbc:
                item = completion_predictions(*args, pattern, slots)
                require(np.array_equal(item.targetmask.numpy(), intended[pattern][rows]))
                require(np.array_equal(item.scoring_target_mask.numpy(),
                                       intended[pattern][rows] & np.isfinite(item.cbc_standardized.numpy())))
                cbc[pattern][arm][rows] = (item.cbc_standardized.numpy() * transform.clinical_iqr[list(slots)]
                                         + transform.clinical_median[list(slots)])
            for pattern in masking.PATTERNS:
                hidden = masking.remove_inputs(c, cm, r, rm, slots, pattern)
                masking.assert_no_input_leak(hidden, cm, rm, slots, pattern)
                item = infer_native(model, tensor(hidden.clinical[rows]), tensor(hidden.clinical_mask[rows], torch.bool),
                                    tensor(hidden.retinal[rows]), tensor(hidden.retinal_mask[rows], torch.bool),
                                    local_age, transform.age_mean, transform.age_scale)
                require(np.array_equal(~item.abstained.numpy(), hidden.available[rows]))
                stress[arm][pattern][rows] = item.screening_probability.numpy()
            for scenario in AGE_SCENARIOS:
                no_physiology = scenario == 'all_physiology_hidden'
                item = infer_native(model, args[1], torch.zeros_like(args[2]) if no_physiology else args[2],
                                    args[3], torch.zeros_like(args[4]) if no_physiology else args[4],
                                    age_scenario(local_age, scenario), transform.age_mean, transform.age_scale)
                if no_physiology: require(bool(item.abstained.all()))
                ages[scenario][arm][rows] = item.screening_probability.numpy()
            # Reload the independently authenticated final checkpoint, not a
            # second fit, and check exact held-out output equality on all routes.
            reloaded, replay_transform = provider(arm, fold)
            require(transform_hash(replay_transform) == transform_hash(transform))
            replay = route_predictions(reloaded, *args[1:])
            require(all(torch.equal(torch.isnan(replay[k].screening_probability), torch.isnan(v.screening_probability))
                        and np.array_equal(replay[k].screening_probability.numpy(), v.screening_probability.numpy(), equal_nan=True)
                        for k, v in predictions.items()))
            for pattern in cbc:
                replay_cbc = completion_predictions(reloaded, *args[1:], pattern, slots)
                replay_value = replay_cbc.cbc_standardized.numpy() * transform.clinical_iqr[list(slots)] + transform.clinical_median[list(slots)]
                require(np.array_equal(replay_value, cbc[pattern][arm][rows], equal_nan=True))
            for pattern in masking.PATTERNS:
                hidden = masking.remove_inputs(c, cm, r, rm, slots, pattern)
                item = infer_native(reloaded, tensor(hidden.clinical[rows]), tensor(hidden.clinical_mask[rows], torch.bool),
                                    tensor(hidden.retinal[rows]), tensor(hidden.retinal_mask[rows], torch.bool),
                                    local_age, transform.age_mean, transform.age_scale)
                require(np.array_equal(item.screening_probability.numpy(), stress[arm][pattern][rows], equal_nan=True))
            for scenario in AGE_SCENARIOS:
                no_physiology = scenario == 'all_physiology_hidden'
                item = infer_native(reloaded, args[1], torch.zeros_like(args[2]) if no_physiology else args[2],
                                    args[3], torch.zeros_like(args[4]) if no_physiology else args[4],
                                    age_scenario(local_age, scenario), transform.age_mean, transform.age_scale)
                require(np.array_equal(item.screening_probability.numpy(), ages[scenario][arm][rows], equal_nan=True))
            del model, reloaded
    if progress: progress({'phase':'candidate_aggregate_bootstrap'})
    counts = paired_counts(paired.folds, draws=1000, seed=91501)
    contrasts = {'token_minus_mlp': ('token_both', 'mlp_both')}
    for arm in ARMS:
        for route in ('clinical', 'retinal'):
            contrasts[arm+'_both_minus_'+route] = (arm+'_both', arm+'_'+route)
    result = {'schema':'bran-multisource-native-outcomes-v2', 'status':'native_profiles_complete_not_promoted',
              'screening':metrics.screening(screen, paired.labels, paired.labelmask, paired.folds, paired.endpoint_names, counts, contrasts),
              'completion':{pattern: metrics.completion(predictions, target, intended[pattern], paired.folds, counts,
                            contrasts={'token_minus_mlp':('token','mlp')}) for pattern, predictions in cbc.items()},
              'missingness':{arm: stress_metrics.summarize(stress[arm], paired.labels, paired.labelmask, paired.folds,
                            paired.endpoint_names, counts) for arm in ARMS},
              'age_and_abstention':{scenario:metrics.screening(predictions, paired.labels, paired.labelmask, paired.folds,
                            paired.endpoint_names, counts, {'token_minus_mlp':('token','mlp')}) for scenario, predictions in ages.items()},
              'endpoint_names':list(paired.endpoint_names), 'cbc_fields':list(CBC_FIELDS),
              'bootstrap_draws':1000, 'bootstrap_seed':91501, 'minimum_valid_draws':900,
              'fixed_fit_development_intervals_not_full_model_selection_uncertainty':True,
              'native_scores_are_not_calibrated_disease_probabilities':True,
              'same_arm_fold_model_used_for_all_tasks':True, 'reload_predictions_equal':True,
              'all_empty_physiology_abstained':True, 'historical_reference_gates_evaluated':False,
              'uncertainty_intervals_calibrated':False, 'external_validation_complete':False,
              'subtyping_established':False, 'candidate_promoted':False,
              'scientific_goal_achieved':False, 'patient_level_output_emitted':False}
    for value in result['missingness'].values(): stress_metrics.validate_result(value, paired.endpoint_names)
    return result
