"""Private C/M advancement comparison using the retained anchored V3 model.

This module performs array-only inference and aggregate calculation.  It has no
loader, checkpoint, training, serialization, selection, or promotion path.
"""
from __future__ import annotations

import numpy as np
import torch

import bran_multisource_advancement_v2 as gates
import bran_multisource_outcomes_v2 as profile
from bran_external_cbc_evaluation_v1 import paired_counts
from bran_missingness_stress_metrics_v1 import summarize
import bran_missingness_stress_v1 as masking
from bran_multisource_anchored_model_v3 import BRANMultisourceAnchoredModelV3
from bran_multisource_batches_v2 import tensor, transform_hash
from bran_multisource_inference_v2 import completion_predictions, infer_native, route_predictions
from bran_multisource_outcome_metrics_v2 import completion as descriptive_completion


_INVALID = "multisource comparison inputs invalid"
_ROLES = ("C", "M")
_ROUTES = ("both", "clinical", "retinal")


def _invalid() -> None:
    raise ValueError(_INVALID)


def _same_state(before: dict[str, torch.Tensor], model: torch.nn.Module) -> bool:
    after = model.state_dict()
    return before.keys() == after.keys() and all(torch.equal(before[name], after[name]) for name in before)


def _view(role: str, screening: dict, completion: dict, missingness: dict,
          *, continued_role: str | None = None) -> gates.RoleViews:
    """Map explicit C/M arrays into the fixed historical metric role names."""
    if role not in _ROLES or (continued_role is not None and continued_role not in _ROLES):
        _invalid()
    try:
        screen = {}
        for endpoint, values in screening.items():
            selected = {}
            for key in gates.old.metrics.S_ARMS:
                if key.startswith("student_"):
                    selected[key] = values[role + key[len("student"):]]
                elif continued_role is not None and key.startswith("continued_"):
                    selected[key] = values[continued_role + key[len("continued"):]]
                else:
                    selected[key] = values[key]
            screen[endpoint] = selected
        complete = {}
        for pattern, values in completion.items():
            selected = {}
            for key in gates.old.metrics.C_ARMS:
                if key == "student":
                    selected[key] = values[role]
                elif key == "continued" and continued_role is not None:
                    selected[key] = values[continued_role]
                else:
                    selected[key] = values[key]
            complete[pattern] = selected
        stress = {}
        for key in gates.old.metrics.VERSIONS:
            if key == "student":
                stress[key] = missingness[role]
            elif key == "continued" and continued_role is not None:
                stress[key] = missingness[continued_role]
            else:
                stress[key] = missingness[key]
        return gates.RoleViews(screen, complete, stress)
    except (KeyError, TypeError, AttributeError):
        _invalid()


def _evaluate_role(role: str, screening: dict, completion: dict, missingness: dict,
                   target: np.ndarray, observed: dict, groups: dict, labels: dict,
                   labelmask: dict, folds: np.ndarray, names: tuple[str, ...], counts: np.ndarray,
                   *, continued_role: str | None = None) -> dict[str, object]:
    """Use the imported historical metrics and unchanged contract decisions."""
    gates.validate_counts(counts, folds)
    if (tuple(screening) != names or not (len(names) == len(set(names)) == 26)
            or not (set(labels) == set(labelmask) == set(names))
            or not (set(observed) == set(groups) == set(gates.contract.EVALPATTERNS))
            or target.shape != (len(folds), 9)):
        _invalid()
    view = _view(role, screening, completion, missingness, continued_role=continued_role)
    for item in gates.old.metrics.VERSIONS:
        gates.old.stress_metrics.validate_result(view.missingness[item], names)
    masks = gates.old.screen_masks(view.screening, labelmask, names, labels, folds)
    screen = gates.old.metrics.screening(view.screening, labels, masks, folds, names, counts)
    summaries = {}
    for pattern in gates.contract.EVALPATTERNS:
        mask = observed[pattern]
        if (mask.shape != target.shape or mask.dtype != bool or not np.isfinite(target[mask]).all()
                or set(groups[pattern]) != set(gates.old.metrics.GROUPS)
                or any(value.shape != target.shape or value.dtype != bool for value in groups[pattern].values())
                or any(value.shape != target.shape or not np.isfinite(value[mask]).all()
                       for value in view.completion[pattern].values())):
            _invalid()
        summaries[pattern] = gates.old.metrics.completion(
            target, mask, view.completion[pattern], gates.old.safe_tail_groups(mask, groups[pattern]), counts)
    paired = {pattern: summaries[pattern] for pattern in gates.contract.PATTERNS}
    no_retina = {pattern: summaries[pattern.replace("_hidden", "_no_retina")]
                 for pattern in gates.contract.PATTERNS}
    decisions = gates.contract.decisions(screen, paired, view.missingness, no_retina)
    for item in summaries.values():
        gates.old.metrics.validate(screen, item, gates.old.metrics.decisions(screen, item), names)
    return {
        "schema": "bran-multisource-advancement-v3", "candidate_role": role,
        "screening": screen, "completion": paired, "completion_no_retina": no_retina,
        "missingness": view.missingness, "decisions": decisions,
        "historical_gate_definitions_changed": False, "candidate_promoted": False,
        "scientific_goal_achieved": False, "patient_level_output_emitted": False,
    }


def _validate_provider(model: object, transform: object, fold: int, slots: tuple[int, ...],
                       expected_transform: object) -> tuple[dict[str, torch.Tensor], str, tuple[bool, ...]]:
    if (not isinstance(model, BRANMultisourceAnchoredModelV3) or model.arm != "mlp"
            or model.training or any(parameter.requires_grad for parameter in model.parameters())
            or tuple(model.cbc_indices) != slots or getattr(transform, "heldout_fold", None) != fold
            or transform_hash(transform) != transform_hash(expected_transform)):
        _invalid()
    return ({name: value.detach().clone() for name, value in model.state_dict().items()},
            transform_hash(transform), tuple(parameter.requires_grad for parameter in model.parameters()))


def _blood_profiles(completion: dict, target: np.ndarray, observed: dict,
                    folds: np.ndarray, counts: np.ndarray) -> dict[str, object]:
    """Four already-inferred descriptive profiles; deliberately not gate inputs."""
    profiles = {}
    for pattern in gates.contract.EVALPATTERNS:
        mask = observed[pattern]
        local_target = target.copy()
        local_target[~mask] = np.nan
        predictions = {}
        for name in ("initial", "C", "M"):
            values = completion[pattern][name].copy()
            values[~mask] = np.nan
            predictions[name] = values
        profiles[pattern] = descriptive_completion(
            predictions, local_target, mask, folds, counts,
            contrasts={"M_minus_C": ("M", "C"), "M_minus_initial": ("M", "initial")},
            low_hb=True)
    return {
        "descriptive_only": True, "low_hb_below_12_research_stratum_only": True,
        "contexts": profiles, "evaluated_contexts": list(gates.contract.EVALPATTERNS),
        "other_two_training_contexts_profile_pending": True,
    }


def evaluate(paired, reference, provider, progress=None) -> dict[str, object]:
    """Aggregate immutable five-fold C/M comparisons without deciding promotion.

    ``provider(role, fold)`` must return a distinct frozen/eval anchored MLP
    plus the exact inherited fold transform.  Progress receives only a small
    role/fold phase dictionary, never arrays, states, or predictions.
    """
    try:
        if not callable(provider) or (progress is not None and not callable(progress)):
            _invalid()
        folds = paired.folds
        names = tuple(paired.endpoint_names)
        count = len(folds)
        if (len(np.unique(folds)) != 5 or set(np.unique(folds)) != set(range(5))
                or tuple(reference.screening) != names or not (len(names) == len(set(names)) == 26)):
            _invalid()
        slots = tuple(paired.names.index(field) for field in gates.old.metrics.CBC_FIELDS)
        if (reference.target.shape != (count, 9)
                or not np.array_equal(reference.target, paired.c[:, slots], equal_nan=True)
                or reference.replay_receipt["historical_point_metrics_replayed"] is not True
                or reference.replay_receipt["patient_level_output_emitted"] is not False):
            _invalid()
        for column, name in enumerate(names):
            if (not np.array_equal(reference.labels[name], paired.labels[:, column], equal_nan=True)
                    or not np.array_equal(reference.labelmask[name], paired.labelmask[:, column])):
                _invalid()
        if set(reference.completion) != set(gates.contract.EVALPATTERNS):
            _invalid()

        screening = {endpoint: {key: value.copy() for key, value in data.items()}
                     for endpoint, data in reference.screening.items()}
        completion = {pattern: {key: value.copy() for key, value in data.items()}
                      for pattern, data in reference.completion.items()}
        for endpoint, data in screening.items():
            if not all(key in data for key in gates.old.metrics.S_ARMS if not key.startswith("student_")):
                _invalid()
            for role in _ROLES:
                for route in _ROUTES:
                    data[role + "_" + route] = np.full(count, np.nan)
        for pattern, data in completion.items():
            if not all(key in data for key in gates.old.metrics.C_ARMS if key != "student"):
                _invalid()
            for role in _ROLES:
                data[role] = np.full((count, 9), np.nan)
        missingness = dict(reference.missingness)
        if not all(key in missingness for key in ("initial", "continued")):
            _invalid()

        age = profile.original_age(paired)
        seen_models: set[int] = set()
        provided_models = []
        for fold in range(5):
            rows = np.flatnonzero(folds == fold)
            local_age = profile.subset_age(age, rows)
            for role in _ROLES:
                if progress is not None:
                    progress({"phase": "candidate_inference", "role": role, "fold": int(fold)})
                model, transform = provider(role, fold)
                identity = id(model)
                if identity in seen_models:
                    _invalid()
                seen_models.add(identity)
                provided_models.append(model)
                before, inherited_hash, before_grad = _validate_provider(
                    model, transform, fold, slots, paired.transforms[fold])
                clinical, clinical_mask = transform.clinical(paired.c, paired.cm)
                retinal, retinal_mask = transform.retinal(paired.r, paired.rm)
                args = (tensor(clinical[rows]), tensor(clinical_mask[rows], torch.bool),
                        tensor(retinal[rows]), tensor(retinal_mask[rows], torch.bool), local_age,
                        transform.age_mean, transform.age_scale)
                predictions = route_predictions(model, *args)
                for route, item in predictions.items():
                    values = item.screening_probability.numpy()
                    for column, endpoint in enumerate(names):
                        if not np.array_equal(np.isfinite(values[:, column]),
                                              np.isfinite(screening[endpoint]["initial_" + route][rows])):
                            _invalid()
                        screening[endpoint][role + "_" + route][rows] = values[:, column]
                for pattern in completion:
                    item = completion_predictions(model, *args, pattern, slots)
                    if not np.array_equal(item.scoring_target_mask.numpy(), reference.observed[pattern][rows]):
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
                    # Allocate lazily so no stress predictions are retained outside this aggregate.
                    missingness.setdefault(role, {})[pattern] = missingness.get(role, {}).get(
                        pattern, np.full((count, 26), np.nan))
                    missingness[role][pattern][rows] = item.screening_probability.numpy()
                if (not _same_state(before, model) or model.training
                        or tuple(parameter.requires_grad for parameter in model.parameters()) != before_grad
                        or transform_hash(transform) != inherited_hash):
                    _invalid()

        counts = paired_counts(folds, draws=1000, seed=91501)
        for role in _ROLES:
            missingness[role] = summarize(missingness[role], paired.labels, paired.labelmask,
                                           folds, names, counts)
        historical = {}
        for role in _ROLES:
            if progress is not None:
                progress({"phase": "advancement_bootstrap", "role": role})
            historical[role] = _evaluate_role(role, screening, completion, missingness,
                                               reference.target, reference.observed, reference.groups,
                                               reference.labels, reference.labelmask, folds, names, counts)
        matched = _evaluate_role("M", screening, completion, missingness, reference.target,
                                 reference.observed, reference.groups, reference.labels,
                                 reference.labelmask, folds, names, counts, continued_role="C")
        macro = matched["screening"]["macro"]
        macro_interval = None if macro is None else macro["contrasts"]["continued_both"]["ci95"]
        blood_profiles = _blood_profiles(completion, reference.target, reference.observed, folds, counts)
        promotion_eligible = bool(historical["M"]["decisions"]["advancement_supported"]
                                  and matched["decisions"]["advancement_supported"])
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        _invalid()

    return {
        "schema": "bran-multisource-comparison-v3",
        "status": "prospective_matched_comparison_completed_pending_audit",
        "historical_comparisons": historical,
        "matched_m_vs_c": {**matched, "prospective_matched_comparison": True,
                             "macro_auc_interval_m_vs_c": macro_interval},
        "blood_completion_profiles": blood_profiles,
        "historical_replay": dict(reference.replay_receipt),
        "eligibility": {"C": False, "M": promotion_eligible},
        "promotion_eligible": promotion_eligible, "candidate_promoted": False,
        "historical_gate_definitions_changed": False, "named_fm_benchmark_complete": False,
        "external_validation_complete": False, "subtyping_established": False,
        "scientific_goal_achieved": False, "patient_level_output_emitted": False,
    }
