"""Block-regularized nested logistic readout for structured V5 evidence vectors.

The both-modality vector contains separate eye-owned and clinical-owned blocks.
This readout tunes one penalty per nonconstant modality block inside outer
training, so adding a modality does not force the original block to accept the
new block's shrinkage.  Patient-level predictions and losses remain local only.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from patient_atlas_disease_readout import (
    DEFAULT_PENALTY_GRID,
    DiseaseReadoutConfig,
    DiseaseReadoutError,
    _average_precision,
    _auroc,
    _binary_calibration,
    _binary_log_loss,
    _fit_standardizer,
    _hash_json,
    _sigmoid,
    _validate_design,
)


GROUP_READOUT_SCHEMA_VERSION = "patient-atlas-v5-disease-group-readout-result-v1"


@dataclass(frozen=True)
class GroupedDiseaseReadoutResult:
    selected_penalties: Mapping[str, float]
    active_penalty_groups: tuple[str, ...]
    inner_primary_loss: float
    train_eligible_count: int
    test_eligible_count: int
    train_event_count: int
    test_event_count: int
    evaluation_set_sha256: str
    minimum_disclosable_cell_count: int
    metrics: Mapping[str, float | int | None]
    optimizer_fallback_fit_count: int

    def aggregate_only_payload(self) -> dict[str, Any]:
        threshold = self.minimum_disclosable_cell_count
        train_events: int | None = self.train_event_count
        test_events: int | None = self.test_event_count
        if (
            train_events < threshold
            or self.train_eligible_count - train_events < threshold
        ):
            train_events = None
        if test_events < threshold or self.test_eligible_count - test_events < threshold:
            test_events = None
        return {
            "schema_version": GROUP_READOUT_SCHEMA_VERSION,
            "task": "binary",
            "selected_penalties": dict(sorted(self.selected_penalties.items())),
            "active_penalty_groups": list(self.active_penalty_groups),
            "inner_primary_loss": self.inner_primary_loss,
            "train_eligible_count": self.train_eligible_count,
            "test_eligible_count": self.test_eligible_count,
            "train_event_count": train_events,
            "test_event_count": test_events,
            "minimum_disclosable_cell_count": threshold,
            "evaluation_set_sha256": self.evaluation_set_sha256,
            "metrics": dict(self.metrics),
            "optimizer_fallback_fit_count": self.optimizer_fallback_fit_count,
            "contains_patient_ids": False,
            "contains_predictions": False,
            "contains_per_patient_losses": False,
        }


@dataclass(frozen=True)
class _GroupedLogisticModel:
    center: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    optimizer_fallback_used: bool


def _fit_grouped_logistic(
    x: np.ndarray,
    y: np.ndarray,
    feature_penalties: np.ndarray,
    config: DiseaseReadoutConfig,
) -> _GroupedLogisticModel:
    if set(np.unique(y)) != {0.0, 1.0}:
        raise DiseaseReadoutError("grouped logistic training requires both binary classes")
    penalties = np.asarray(feature_penalties, dtype=np.float64)
    if penalties.shape != (x.shape[1],) or not np.isfinite(penalties).all():
        raise DiseaseReadoutError("grouped logistic feature penalties are malformed")
    if bool((penalties < 0.0).any()):
        raise DiseaseReadoutError("grouped logistic penalties must be nonnegative")
    center, scale, standardized = _fit_standardizer(x)
    design = np.column_stack((np.ones(len(x), dtype=np.float64), standardized))
    event_rate = float(
        np.clip(y.mean(), config.probability_clip, 1.0 - config.probability_clip)
    )
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    parameters[0] = math.log(event_rate / (1.0 - event_rate))
    penalty_diagonal = np.concatenate(([0.0], penalties))

    def objective(candidate: np.ndarray) -> float:
        linear = design @ candidate
        return float(
            np.logaddexp(0.0, linear).sum()
            - y @ linear
            + 0.5 * np.sum(penalty_diagonal * candidate**2)
        )

    def gradient(candidate: np.ndarray) -> np.ndarray:
        probability = _sigmoid(design @ candidate)
        return design.T @ (probability - y) + penalty_diagonal * candidate

    converged = False
    used_fallback = False
    for _ in range(config.logistic_max_iterations):
        probability = _sigmoid(design @ parameters)
        gradient_value = gradient(parameters)
        weights = np.clip(probability * (1.0 - probability), 1e-9, None)
        hessian = design.T @ (design * weights[:, None])
        hessian.flat[:: hessian.shape[0] + 1] += penalty_diagonal
        try:
            step = np.linalg.solve(hessian, gradient_value)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient_value, rcond=None)[0]
        if not np.isfinite(step).all():
            raise DiseaseReadoutError("grouped logistic produced a non-finite step")
        current = objective(parameters)
        step_scale = 1.0
        for _ in range(30):
            candidate = parameters - step_scale * step
            if objective(candidate) <= current + 1e-12:
                parameters = candidate
                break
            step_scale *= 0.5
        else:
            raise DiseaseReadoutError("grouped logistic could not find a descent step")
        if float(np.max(np.abs(step_scale * step))) <= config.logistic_tolerance:
            converged = True
            break
    if not converged:
        fallback = minimize(
            objective,
            parameters,
            jac=gradient,
            method="L-BFGS-B",
            options={
                "maxiter": max(1000, 3 * config.logistic_max_iterations),
                "ftol": 1e-12,
                "gtol": max(config.logistic_tolerance, 1e-8),
                "maxls": 50,
            },
        )
        parameters = np.asarray(fallback.x, dtype=np.float64)
        gradient_max = float(np.max(np.abs(gradient(parameters))))
        if (
            not np.isfinite(parameters).all()
            or not np.isfinite(float(fallback.fun))
            or (not bool(fallback.success) and gradient_max > 1e-5)
        ):
            raise DiseaseReadoutError(
                "grouped logistic Newton and L-BFGS fallback did not converge"
            )
        used_fallback = True
    return _GroupedLogisticModel(
        center=center,
        scale=scale,
        coefficient=parameters[1:],
        intercept=float(parameters[0]),
        optimizer_fallback_used=used_fallback,
    )


def _probability(
    model: _GroupedLogisticModel,
    x: np.ndarray,
    config: DiseaseReadoutConfig,
) -> np.ndarray:
    linear = model.intercept + ((x - model.center) / model.scale) @ model.coefficient
    if not np.isfinite(linear).all():
        raise DiseaseReadoutError("grouped logistic prediction is non-finite")
    return np.clip(
        _sigmoid(linear),
        config.probability_clip,
        1.0 - config.probability_clip,
    )


def _feature_penalties(
    groups: np.ndarray,
    penalties: Mapping[str, float],
    unpenalized_groups: frozenset[str],
) -> np.ndarray:
    result = np.empty(len(groups), dtype=np.float64)
    for index, raw_group in enumerate(groups):
        group = str(raw_group)
        result[index] = 0.0 if group in unpenalized_groups else float(penalties[group])
    return result


def evaluate_nested_grouped_logistic_readout(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_eligible: np.ndarray,
    inner_fold_ids: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_eligible: np.ndarray,
    feature_groups: Sequence[str],
    unpenalized_groups: Sequence[str],
    test_patient_id_hash: str,
    config: DiseaseReadoutConfig = DiseaseReadoutConfig(),
    maximum_penalty_combinations: int = 100,
    _private_test_primary_loss_out: np.ndarray | None = None,
) -> GroupedDiseaseReadoutResult:
    train_y = np.asarray(y_train, dtype=np.float64)
    test_y = np.asarray(y_test, dtype=np.float64)
    train_mask = np.asarray(train_eligible)
    test_mask = np.asarray(test_eligible)
    folds = np.asarray(inner_fold_ids)
    train_x = _validate_design("x_train", x_train, len(train_y))
    test_x = _validate_design("x_test", x_test, len(test_y))
    if train_x.shape[1] != test_x.shape[1]:
        raise DiseaseReadoutError("grouped train/test dimensions differ")
    groups = np.asarray(tuple(str(value) for value in feature_groups), dtype=object)
    if groups.shape != (train_x.shape[1],) or any(not value for value in groups):
        raise DiseaseReadoutError("grouped readout feature groups are malformed")
    unpenalized = frozenset(str(value) for value in unpenalized_groups)
    if not unpenalized or not unpenalized.issubset(set(groups.tolist())):
        raise DiseaseReadoutError("grouped readout unpenalized groups differ")
    if train_mask.shape != train_y.shape or train_mask.dtype != np.bool_:
        raise DiseaseReadoutError("grouped train eligibility is malformed")
    if test_mask.shape != test_y.shape or test_mask.dtype != np.bool_:
        raise DiseaseReadoutError("grouped test eligibility is malformed")
    if folds.shape != train_y.shape or not np.issubdtype(folds.dtype, np.integer):
        raise DiseaseReadoutError("grouped inner folds are malformed")
    if len(test_patient_id_hash) != 64:
        raise DiseaseReadoutError("grouped test identity hash is malformed")
    if not bool(train_mask.any()) or not bool(test_mask.any()):
        raise DiseaseReadoutError("grouped readout has an empty train or test target")
    for label, values in (("train", train_y[train_mask]), ("test", test_y[test_mask])):
        if not set(np.unique(values)).issubset({0.0, 1.0}):
            raise DiseaseReadoutError(f"grouped {label} target is not binary")
    if len(np.unique(train_y[train_mask])) != 2:
        raise DiseaseReadoutError("grouped outer training requires both classes")

    penalized_groups = tuple(sorted(set(groups.tolist()) - unpenalized))
    active_groups = tuple(
        group
        for group in penalized_groups
        if bool(
            np.any(
                np.std(train_x[train_mask][:, groups == group], axis=0) > 1e-12
            )
        )
    )
    if not active_groups:
        raise DiseaseReadoutError("grouped readout has no active modality block")
    inactive_groups = set(penalized_groups) - set(active_groups)
    candidates = tuple(itertools.product(config.penalty_grid, repeat=len(active_groups)))
    if len(candidates) > maximum_penalty_combinations:
        raise DiseaseReadoutError("grouped penalty grid exceeds its frozen limit")
    populated_folds = tuple(sorted(int(value) for value in np.unique(folds[train_mask])))
    if len(populated_folds) < 2:
        raise DiseaseReadoutError("grouped readout needs at least two inner folds")

    best_penalties: dict[str, float] | None = None
    fallback_fit_count = 0
    best_loss: float | None = None
    best_order: tuple[float, tuple[float, ...]] | None = None
    for candidate in candidates:
        penalty_map = {
            group: float(value) for group, value in zip(active_groups, candidate)
        }
        penalty_map.update(
            {group: float(config.penalty_grid[-1]) for group in inactive_groups}
        )
        feature_penalty = _feature_penalties(groups, penalty_map, unpenalized)
        total_loss = 0.0
        total_count = 0
        for fold in populated_folds:
            validation = train_mask & (folds == fold)
            training = train_mask & (folds != fold)
            if not bool(validation.any()) or len(np.unique(train_y[training])) != 2:
                raise DiseaseReadoutError("grouped inner fold lacks rows or a binary class")
            model = _fit_grouped_logistic(
                train_x[training], train_y[training], feature_penalty, config
            )
            fallback_fit_count += int(model.optimizer_fallback_used)
            probability = _probability(model, train_x[validation], config)
            count = int(validation.sum())
            total_loss += _binary_log_loss(train_y[validation], probability) * count
            total_count += count
        mean_loss = total_loss / total_count
        order = (
            float(mean_loss),
            tuple(-float(penalty_map[group]) for group in active_groups),
        )
        if best_order is None or order < best_order:
            best_order = order
            best_penalties = penalty_map
            best_loss = float(mean_loss)
    if best_penalties is None or best_loss is None:
        raise RuntimeError("grouped readout selection failed")

    feature_penalty = _feature_penalties(groups, best_penalties, unpenalized)
    model = _fit_grouped_logistic(
        train_x[train_mask], train_y[train_mask], feature_penalty, config
    )
    fallback_fit_count += int(model.optimizer_fallback_used)
    probability = _probability(model, test_x[test_mask], config)
    eligible_test_y = test_y[test_mask]
    calibration_intercept, calibration_slope = _binary_calibration(
        eligible_test_y, probability, config
    )
    metrics: dict[str, float | int | None] = {
        "log_loss": _binary_log_loss(eligible_test_y, probability),
        "brier_score": float(np.mean((probability - eligible_test_y) ** 2)),
        "auroc": _auroc(eligible_test_y, probability),
        "average_precision": _average_precision(eligible_test_y, probability),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }
    if _private_test_primary_loss_out is not None:
        private = np.asarray(_private_test_primary_loss_out)
        if private.shape != test_y.shape or private.dtype.kind != "f" or not private.flags.writeable:
            raise DiseaseReadoutError("grouped private loss sink is malformed")
        private.fill(np.nan)
        private[test_mask] = -(
            eligible_test_y * np.log(probability)
            + (1.0 - eligible_test_y) * np.log1p(-probability)
        )
    result = GroupedDiseaseReadoutResult(
        selected_penalties=MappingProxyType(dict(sorted(best_penalties.items()))),
        active_penalty_groups=active_groups,
        inner_primary_loss=best_loss,
        train_eligible_count=int(train_mask.sum()),
        test_eligible_count=int(test_mask.sum()),
        train_event_count=int(train_y[train_mask].sum()),
        test_event_count=int(test_y[test_mask].sum()),
        evaluation_set_sha256=_hash_json(
            {
                "test_patient_id_hash": test_patient_id_hash,
                "eligible_mask": test_mask.tolist(),
            }
        ),
        minimum_disclosable_cell_count=config.minimum_disclosable_cell_count,
        metrics=MappingProxyType(metrics),
        optimizer_fallback_fit_count=fallback_fit_count,
    )
    payload = result.aggregate_only_payload()
    if any(key in payload for key in ("patient_ids", "predictions", "losses")):
        raise RuntimeError("grouped aggregate payload exposed patient material")
    return result


__all__ = [
    "GROUP_READOUT_SCHEMA_VERSION",
    "GroupedDiseaseReadoutResult",
    "evaluate_nested_grouped_logistic_readout",
]
