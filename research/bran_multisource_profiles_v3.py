"""Private native descriptive profiles for explicit anchored V3 C/M roles.

All arrays remain local.  This is not advancement evaluation, calibration,
training, checkpoint I/O, external validation, or a promotion decision.
"""
from __future__ import annotations

import numpy as np
import torch

from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_external_cbc_evaluation_v1 import paired_counts
from bran_multisource_anchored_model_v3 import BRANMultisourceAnchoredModelV3
from bran_multisource_batches_v2 import tensor, transform_hash
from bran_multisource_inference_v2 import completion_predictions, infer_native, route_predictions
from bran_multisource_outcomes_v2 import AGE_SCENARIOS, age_scenario, original_age, subset_age
from bran_multisource_protocol_v2 import PARAMETERS
import bran_missingness_stress_metrics_v1 as stress_metrics
import bran_missingness_stress_v1 as masking
import bran_multisource_outcome_metrics_v2 as metrics


_INVALID = "multisource profiles inputs invalid"
_ROLES = ("C", "M")
_ROUTES = ("both", "clinical", "retinal")


def _invalid() -> None:
    raise ValueError(_INVALID)


def _same_state(before: dict[str, torch.Tensor], model: torch.nn.Module) -> bool:
    current = model.state_dict()
    return before.keys() == current.keys() and all(torch.equal(before[name], current[name]) for name in before)


def _validate_provider(model: object, transform: object, fold: int, slots: tuple[int, ...],
                       inherited_transform: object) -> tuple[dict[str, torch.Tensor], str, tuple[bool, ...]]:
    if (not isinstance(model, BRANMultisourceAnchoredModelV3) or model.arm != "mlp"
            or model.training or any(parameter.requires_grad for parameter in model.parameters())
            or tuple(model.cbc_indices) != slots or getattr(transform, "heldout_fold", None) != fold
            or transform_hash(transform) != transform_hash(inherited_transform)):
        _invalid()
    return ({name: value.detach().clone() for name, value in model.state_dict().items()},
            transform_hash(transform), tuple(parameter.requires_grad for parameter in model.parameters()))


def _unchanged(before: dict[str, torch.Tensor], transform_hash_before: str,
               requires_grad: tuple[bool, ...], model: torch.nn.Module, transform: object) -> bool:
    return (not model.training and _same_state(before, model)
            and tuple(parameter.requires_grad for parameter in model.parameters()) == requires_grad
            and transform_hash(transform) == transform_hash_before)


def evaluate(paired, provider, progress=None) -> dict[str, object]:
    """Profile C/M on fixed held-out routes, CBC contexts, stress, and ages.

    ``provider(role, fold)`` is called once for inference and once for an
    independently reloaded replay.  Both models must be frozen anchored MLPs
    bound to the exact inherited fold transform.
    """
    try:
        if not callable(provider) or (progress is not None and not callable(progress)):
            _invalid()
        folds = paired.folds
        count = len(folds)
        names = tuple(paired.endpoint_names)
        if count < 100 or not (len(names) == len(set(names)) == 26) or set(np.unique(folds)) != set(range(5)):
            _invalid()
        slots = tuple(paired.names.index(field) for field in CBC_FIELDS)
        eligible = np.zeros(59, dtype=bool)
        eligible[list(paired.eligible_indices)] = True
        target = paired.c[:, slots].copy()
        observed = (paired.cm & eligible[None, :])[:, slots]
        age = original_age(paired)
        screen = {role + "_" + route: np.full((count, 26), np.nan)
                  for role in _ROLES for route in _ROUTES}
        stress = {role: {pattern: np.full((count, 26), np.nan) for pattern in masking.PATTERNS}
                  for role in _ROLES}
        ages = {scenario: {role: np.full((count, 26), np.nan) for role in _ROLES}
                for scenario in AGE_SCENARIOS}
        completion = {pattern: {role: np.full((count, 9), np.nan) for role in _ROLES}
                      for pattern in PARAMETERS["completion_patterns"]}
        intended = {pattern: observed.copy() for pattern in completion}
        for pattern in intended:
            if pattern.startswith("red_cell"):
                intended[pattern][:, (5, 7, 8)] = False

        seen_models: set[int] = set()
        retained_models = []
        for fold in range(5):
            rows = np.flatnonzero(folds == fold)
            if len(rows) == 0:
                _invalid()
            local_age = subset_age(age, rows)
            for role in _ROLES:
                if progress is not None:
                    progress({"phase": "candidate_inference", "role": role, "fold": int(fold)})
                model, transform = provider(role, fold)
                if id(model) in seen_models:
                    _invalid()
                seen_models.add(id(model)); retained_models.append(model)
                before, transform_before, before_grad = _validate_provider(
                    model, transform, fold, slots, paired.transforms[fold])
                clinical, clinical_mask = transform.clinical(paired.c, paired.cm)
                retinal, retinal_mask = transform.retinal(paired.r, paired.rm)
                args = (tensor(clinical[rows]), tensor(clinical_mask[rows], torch.bool),
                        tensor(retinal[rows]), tensor(retinal_mask[rows], torch.bool), local_age,
                        transform.age_mean, transform.age_scale)
                predicted = route_predictions(model, *args)
                for route, item in predicted.items():
                    screen[role + "_" + route][rows] = item.screening_probability.numpy()
                for pattern in completion:
                    item = completion_predictions(model, *args, pattern, slots)
                    if (not np.array_equal(item.targetmask.numpy(), intended[pattern][rows])
                            or not np.array_equal(item.scoring_target_mask.numpy(),
                                                  intended[pattern][rows] & np.isfinite(item.cbc_standardized.numpy()))):
                        _invalid()
                    completion[pattern][role][rows] = (
                        item.cbc_standardized.numpy() * transform.clinical_iqr[list(slots)]
                        + transform.clinical_median[list(slots)])
                for pattern in masking.PATTERNS:
                    hidden = masking.remove_inputs(clinical, clinical_mask, retinal, retinal_mask, slots, pattern)
                    masking.assert_no_input_leak(hidden, clinical_mask, retinal_mask, slots, pattern)
                    item = infer_native(model, tensor(hidden.clinical[rows]),
                                        tensor(hidden.clinical_mask[rows], torch.bool),
                                        tensor(hidden.retinal[rows]),
                                        tensor(hidden.retinal_mask[rows], torch.bool), local_age,
                                        transform.age_mean, transform.age_scale)
                    if not np.array_equal(~item.abstained.numpy(), hidden.available[rows]):
                        _invalid()
                    stress[role][pattern][rows] = item.screening_probability.numpy()
                for scenario in AGE_SCENARIOS:
                    all_hidden = scenario == "all_physiology_hidden"
                    item = infer_native(model, args[0], torch.zeros_like(args[1]) if all_hidden else args[1],
                                        args[2], torch.zeros_like(args[3]) if all_hidden else args[3],
                                        age_scenario(local_age, scenario), transform.age_mean, transform.age_scale)
                    if all_hidden and not bool(item.abstained.all()):
                        _invalid()
                    ages[scenario][role][rows] = item.screening_probability.numpy()
                if not _unchanged(before, transform_before, before_grad, model, transform):
                    _invalid()

                reloaded, replay_transform = provider(role, fold)
                if id(reloaded) in seen_models:
                    _invalid()
                seen_models.add(id(reloaded)); retained_models.append(reloaded)
                replay_before, replay_hash, replay_grad = _validate_provider(
                    reloaded, replay_transform, fold, slots, paired.transforms[fold])
                if replay_hash != transform_before:
                    _invalid()
                replay = route_predictions(reloaded, *args)
                if not all(torch.equal(torch.isnan(replay[key].screening_probability), torch.isnan(value.screening_probability))
                           and np.array_equal(replay[key].screening_probability.numpy(),
                                              value.screening_probability.numpy(), equal_nan=True)
                           for key, value in predicted.items()):
                    _invalid()
                for pattern in completion:
                    item = completion_predictions(reloaded, *args, pattern, slots)
                    value = item.cbc_standardized.numpy() * transform.clinical_iqr[list(slots)] + transform.clinical_median[list(slots)]
                    if not np.array_equal(value, completion[pattern][role][rows], equal_nan=True):
                        _invalid()
                for pattern in masking.PATTERNS:
                    hidden = masking.remove_inputs(clinical, clinical_mask, retinal, retinal_mask, slots, pattern)
                    item = infer_native(reloaded, tensor(hidden.clinical[rows]),
                                        tensor(hidden.clinical_mask[rows], torch.bool),
                                        tensor(hidden.retinal[rows]),
                                        tensor(hidden.retinal_mask[rows], torch.bool), local_age,
                                        transform.age_mean, transform.age_scale)
                    if not np.array_equal(item.screening_probability.numpy(), stress[role][pattern][rows], equal_nan=True):
                        _invalid()
                for scenario in AGE_SCENARIOS:
                    all_hidden = scenario == "all_physiology_hidden"
                    item = infer_native(reloaded, args[0], torch.zeros_like(args[1]) if all_hidden else args[1],
                                        args[2], torch.zeros_like(args[3]) if all_hidden else args[3],
                                        age_scenario(local_age, scenario), transform.age_mean, transform.age_scale)
                    if not np.array_equal(item.screening_probability.numpy(), ages[scenario][role][rows], equal_nan=True):
                        _invalid()
                if not _unchanged(replay_before, replay_hash, replay_grad, reloaded, replay_transform):
                    _invalid()

        if progress is not None:
            progress({"phase": "candidate_aggregate_bootstrap"})
        counts = paired_counts(folds, draws=1000, seed=91501)
        screening_contrasts = {"M_minus_C": ("M_both", "C_both")}
        for role in _ROLES:
            for route in ("clinical", "retinal"):
                screening_contrasts[role + "_both_minus_" + route] = (role + "_both", role + "_" + route)
        result = {
            "schema": "bran-multisource-native-profiles-v3",
            "status": "native_descriptive_profiles_complete_not_advancement",
            "screening": metrics.screening(screen, paired.labels, paired.labelmask, folds, names,
                                              counts, screening_contrasts),
            "completion": {pattern: metrics.completion(
                {role: np.where(intended[pattern], value, np.nan) for role, value in values.items()},
                target, intended[pattern], folds, counts,
                contrasts={"M_minus_C": ("M", "C")}, low_hb=True)
                for pattern, values in completion.items()},
            "missingness": {role: stress_metrics.summarize(stress[role], paired.labels,
                            paired.labelmask, folds, names, counts) for role in _ROLES},
            "age_and_abstention": {scenario: metrics.screening(
                values, paired.labels, paired.labelmask, folds, names, counts,
                {"M_minus_C": ("M", "C")}) for scenario, values in ages.items()},
            "endpoint_names": list(names), "cbc_fields": list(CBC_FIELDS),
            "completion_patterns": list(PARAMETERS["completion_patterns"]),
            "bootstrap_draws": 1000, "bootstrap_seed": 91501, "minimum_valid_draws": 900,
            "fixed_fit_development_intervals_not_full_model_selection_uncertainty": True,
            "native_scores_are_not_calibrated_disease_probabilities": True,
            "same_role_fold_model_used_for_all_tasks": True, "reload_predictions_equal": True,
            "all_empty_physiology_abstained": True, "historical_reference_gates_evaluated": False,
            "uncertainty_intervals_calibrated": False, "external_validation_complete": False,
            "subtyping_established": False, "candidate_promoted": False,
            "scientific_goal_achieved": False, "patient_level_output_emitted": False,
        }
        for value in result["missingness"].values():
            stress_metrics.validate_result(value, names)
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        _invalid()
    return result


def metrics_profile_age_scenarios() -> tuple[str, ...]:
    """The immutable V2 three-scenario contract, retained without arm aliases."""
    return tuple(AGE_SCENARIOS)
