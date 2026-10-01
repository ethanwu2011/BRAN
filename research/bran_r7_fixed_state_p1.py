"""Frozen, local-only R7-versus-V5 fixed-state linear readout diagnostic.

This module owns no source or checkpoint I/O.  It accepts only already
authenticated providers, replays every inferred state, and returns aggregate
metrics.  States, predictions, labels, and bootstrap draws remain local to the
caller.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_external_cbc_evaluation_v1 import paired_counts
from bran_multisource_batches_v2 import tensor, transform_hash
from bran_multisource_outcomes_v2 import original_age
from bran_multisource_profiles_v3 import _unchanged, _validate_provider
from bran_robust_clinical_r7 import BRANRobustClinicalR7
from bran_v5_state_routes import ROUTES, state_routes
import bran_multisource_outcome_metrics_v2 as metrics


ERROR = "r7_fixed_state_p1_invalid"
ROLES = ("V5", "R7")
PARAMETERS = {
    "roles": {"V5": "V5_attempt2_old_M_checkpoint", "R7": "R7_attempt1_robust_R_checkpoint"},
    "checkpoint_families": {"V5": "V5-M_attempt2", "R7": "R7_robust_attempt1"},
    "routes": list(ROUTES), "folds": 5, "endpoints": 26,
    "state": "posterior_mean_only_per_checkpoint_fold_coordinate_system",
    "readout": "StandardScaler_L2_logistic_C1_lbfgs_max5000_default_class_weight",
    "encoder_updates": 0, "readout_search": False,
    "fit_population": "outer_fold_training_observed_labels_common_physiological_support",
    "score_population": "outer_fold_test_common_available_participants",
    "draws": 1000, "seed": 98571, "minimum_valid": 900,
    "bootstrap": "matched_participant_within_fold", "ci": 0.95,
    "primary": "both_R7_minus_V5", "secondary_routes": ["clinical", "retinal"],
    "reused_development_and_model_selection_uncertainty_not_removed": True,
    "candidate_promoted": False,
}


def require(ok: bool) -> None:
    if not ok:
        raise ValueError(ERROR) from None


def _invalid() -> None:
    raise ValueError(ERROR) from None


def validate_evaluation_inputs(paired: object, provider: object, progress: object = None) -> None:
    """Validate only array metadata; this intentionally does not inspect rows."""
    require(callable(provider) and (progress is None or callable(progress)))
    folds = np.asarray(getattr(paired, "folds", None))
    names = tuple(getattr(paired, "endpoint_names", ()))
    labels, mask = getattr(paired, "labels", None), getattr(paired, "labelmask", None)
    require(folds.ndim == 1 and len(folds) >= 100 and set(np.unique(folds)) == set(range(5)))
    require(len(names) == len(set(names)) == 26 and all(type(name) is str and name for name in names))
    require(isinstance(labels, np.ndarray) and isinstance(mask, np.ndarray)
            and labels.shape == mask.shape == (len(folds), 26) and mask.dtype == bool)
    require(np.isfinite(labels[mask]).all() and np.isin(labels[mask], (0, 1)).all())
    require(len(getattr(paired, "transforms", ())) == 5)


def common_support(*available: np.ndarray) -> np.ndarray:
    """Return the only support on which a paired state contrast may be fit/scored."""
    try:
        require(len(available) == 2)
        values = tuple(np.asarray(value, dtype=bool) for value in available)
        require(values[0].ndim == values[1].ndim == 1 and values[0].shape == values[1].shape)
        return np.logical_and(values[0], values[1])
    except Exception:
        _invalid()


def fixed_probe(x_train, y_train, observed_train, x_test):
    """The locked C/M recipe, with convergence treated as a terminal failure."""
    try:
        x, y, observed, test = (np.asarray(x_train, dtype=float), np.asarray(y_train),
                                 np.asarray(observed_train, dtype=bool), np.asarray(x_test, dtype=float))
        require(x.ndim == 2 and test.ndim == 2 and x.shape[1] == test.shape[1])
        require(y.shape == observed.shape == (len(x),) and observed.sum() >= 20)
        require(np.isfinite(x[observed]).all() and np.isfinite(test).all())
        require(set(np.unique(y[observed])) == {0, 1})
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=5000))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x[observed], y[observed])
        require(not any(issubclass(item.category, ConvergenceWarning) for item in caught))
        value = model.predict_proba(test)[:, 1]
        require(value.shape == (len(test),) and np.isfinite(value).all()
                and bool(((0.0 <= value) & (value <= 1.0)).all()))
        return value
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _invalid()


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _validate_r7(model: object, transform: object, fold: int, slots: tuple[int, ...], inherited: object):
    require(type(model) is BRANRobustClinicalR7 and model.arm == "mlp" and not model.training)
    require(not any(parameter.requires_grad for parameter in model.parameters()))
    require(tuple(model.cbc_indices) == slots and getattr(transform, "heldout_fold", None) == fold)
    require(transform_hash(transform) == transform_hash(inherited))
    return _snapshot(model), transform_hash(transform), tuple(parameter.requires_grad for parameter in model.parameters())


def _validate_role(role: str, model: object, transform: object, fold: int, slots: tuple[int, ...], inherited: object):
    require(role in ROLES)
    return (_validate_provider(model, transform, fold, slots, inherited) if role == "V5"
            else _validate_r7(model, transform, fold, slots, inherited))


def replay_equal(first, second) -> bool:
    """Exact state-and-abstention replay predicate; no tolerances are permitted."""
    try:
        return all(torch.equal(first.states[route], second.states[route])
                   and torch.equal(first.available[route], second.available[route]) for route in ROUTES)
    except Exception:
        return False


def _role_unchanged(before, transform_pin, gradients, model, transform) -> bool:
    return _unchanged(before, transform_pin, gradients, model, transform)


def evaluate(paired: object, provider: object, progress=None) -> dict[str, object]:
    """Infer and replay fixed V5/R7 states, then fit one locked probe per cell."""
    try:
        validate_evaluation_inputs(paired, provider, progress)
        folds = np.asarray(paired.folds)
        count, age = len(folds), original_age(paired)
        slots = tuple(paired.names.index(field) for field in CBC_FIELDS)
        predictions = {route: {role: np.full((count, 26), np.nan) for role in ROLES} for route in ROUTES}
        for fold in range(5):
            designs = {}
            for role in ROLES:
                if progress is not None:
                    progress("state_inference", fold, role)
                model, transform = provider(role, fold)
                before, pin, grad = _validate_role(role, model, transform, fold, slots, paired.transforms[fold])
                c, cm = transform.clinical(paired.c, paired.cm)
                r, rm = transform.retinal(paired.r, paired.rm)
                args = (tensor(c), tensor(cm, torch.bool), tensor(r), tensor(rm, torch.bool),
                        age, transform.age_mean, transform.age_scale)
                designs[role] = state_routes(model, *args)
                require(_role_unchanged(before, pin, grad, model, transform))
                replay, replay_transform = provider(role, fold)
                replay_before, replay_pin, replay_grad = _validate_role(
                    role, replay, replay_transform, fold, slots, paired.transforms[fold])
                replayed = state_routes(replay, *args)
                require(replay is not model and replay_equal(designs[role], replayed))
                require(_role_unchanged(replay_before, replay_pin, replay_grad, replay, replay_transform))
                del model, replay, replayed
            train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            for route in ROUTES:
                support = common_support(*(designs[role].available[route].detach().cpu().numpy() for role in ROLES))
                eligible_test = test[support[test]]
                for role in ROLES:
                    if progress is not None:
                        progress("fixed_readout", fold, role)
                    x = designs[role].states[route].detach().cpu().numpy()
                    for endpoint in range(26):
                        observed = paired.labelmask[:, endpoint] & support
                        predictions[route][role][eligible_test, endpoint] = fixed_probe(
                            x[train], paired.labels[train, endpoint], observed[train], x[eligible_test])
            del designs
        if progress is not None:
            progress("aggregate_bootstrap", None, None)
        counts = paired_counts(folds, draws=1000, seed=98571)
        routes = {route: metrics.screening(value, paired.labels, paired.labelmask, folds,
                  tuple(paired.endpoint_names), counts, {"R7_minus_V5": ("R7", "V5")})
                  for route, value in predictions.items()}
        result = {"schema": "bran-r7-fixed-state-p1", "parameters": PARAMETERS, "routes": routes,
                  "all_state_inference_replayed": True, "actual_r7_robust_class_required": True,
                  "encoder_updated": False, "candidate_promoted": False,
                  "protected_sources_used": False, "patient_level_output_emitted": False}
        validate_result(result)
        return result
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _invalid()


def _number(value, low=-math.inf, high=math.inf) -> bool:
    return type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(float(value)) and low <= value <= high


def _cell(value: object) -> None:
    if value == {"status": "unsupported"}:
        return
    require(type(value) is dict and set(value) == {"status", "arms", "contrasts"} and value["status"] == "supported")
    require(set(value["arms"]) == set(ROLES) and set(value["contrasts"]) == {"R7_minus_V5"})
    for arm in value["arms"].values():
        require(set(arm) == {"auroc", "ci95"} and _number(arm["auroc"], 0, 1)
                and type(arm["ci95"]) is list and len(arm["ci95"]) == 2
                and all(_number(x, 0, 1) for x in arm["ci95"]) and arm["ci95"][0] <= arm["ci95"][1])
    contrast = value["contrasts"]["R7_minus_V5"]
    require(set(contrast) == {"delta", "ci95"} and _number(contrast["delta"], -1, 1)
            and type(contrast["ci95"]) is list and len(contrast["ci95"]) == 2
            and all(_number(x, -1, 1) for x in contrast["ci95"]) and contrast["ci95"][0] <= contrast["ci95"][1])
    require(abs(contrast["delta"] - (value["arms"]["R7"]["auroc"] - value["arms"]["V5"]["auroc"])) < 1e-9)


def _coverage(value: object) -> None:
    if value == {"status": "withheld"}:
        return
    require(type(value) is dict and set(value) == {"status", "supported", "total"}
            and value["status"] == "released" and type(value["supported"]) is int
            and type(value["total"]) is int and value["total"] >= 20
            and 0 <= value["supported"] <= value["total"]
            and (value["supported"] == 0 or value["supported"] >= 20)
            and (value["total"] - value["supported"] == 0
                 or value["total"] - value["supported"] >= 20))


def validate_result(value: object) -> None:
    """Aggregate-only schema validator; deliberately cannot load sources/checkpoints."""
    try:
        require(type(value) is dict and set(value) == {"schema", "parameters", "routes",
            "all_state_inference_replayed", "actual_r7_robust_class_required", "encoder_updated",
            "candidate_promoted", "protected_sources_used", "patient_level_output_emitted"})
        require(value["schema"] == "bran-r7-fixed-state-p1" and value["parameters"] == PARAMETERS
                and set(value["routes"]) == set(ROUTES))
        require(value["all_state_inference_replayed"] is True and value["actual_r7_robust_class_required"] is True
                and value["encoder_updated"] is False and value["candidate_promoted"] is False
                and value["protected_sources_used"] is False and value["patient_level_output_emitted"] is False)
        for panel in value["routes"].values():
            require(type(panel) is dict and set(panel) == {"complete_26_panel", "macro", "endpoints",
                    "prediction_coverage", "matched_population_coverage"})
            require(type(panel["complete_26_panel"]) is bool and len(panel["endpoints"]) == 26
                    and all(type(name) is str and name and len(name) < 100 and "\n" not in name
                            for name in panel["endpoints"])
                    and set(panel["prediction_coverage"]) == set(ROLES))
            for cell in panel["endpoints"].values():
                _cell(cell)
            require(panel["complete_26_panel"] == all(cell["status"] == "supported" for cell in panel["endpoints"].values()))
            require(panel["macro"] is None or panel["complete_26_panel"])
            if panel["macro"] is not None:
                _cell(panel["macro"])
                for role in ROLES:
                    mean = sum(cell["arms"][role]["auroc"] for cell in panel["endpoints"].values()) / 26
                    require(abs(mean - panel["macro"]["arms"][role]["auroc"]) < 1e-9)
            for item in panel["prediction_coverage"].values():
                _coverage(item)
            _coverage(panel["matched_population_coverage"])
    except Exception:
        _invalid()
