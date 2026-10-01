"""Dependency-free nested linear readouts for Patient Atlas disease utility.

This module operates only inside the local evaluation process.  Public results are
aggregate metrics and hashes; fitted coefficients, patient identifiers, predictions,
and per-patient losses are deliberately not exposed or serialized.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


READOUT_SCHEMA_VERSION = "patient-atlas-v5-disease-readout-result-v1"
DEFAULT_PENALTY_GRID = (
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
)
VALID_TASKS = {"binary", "continuous"}


class DiseaseReadoutError(ValueError):
    """Raised when a nested readout cannot be evaluated without leakage."""


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DiseaseReadoutConfig:
    penalty_grid: tuple[float, ...] = DEFAULT_PENALTY_GRID
    logistic_max_iterations: int = 100
    logistic_tolerance: float = 1e-8
    probability_clip: float = 1e-7
    minimum_disclosable_cell_count: int = 10

    def __post_init__(self) -> None:
        if not self.penalty_grid:
            raise DiseaseReadoutError("penalty grid must not be empty")
        if tuple(sorted(set(self.penalty_grid))) != self.penalty_grid:
            raise DiseaseReadoutError("penalty grid must be unique and increasing")
        if any(not np.isfinite(value) or value <= 0.0 for value in self.penalty_grid):
            raise DiseaseReadoutError("all penalties must be finite and positive")
        if self.logistic_max_iterations < 2 or self.logistic_tolerance <= 0.0:
            raise DiseaseReadoutError("logistic optimizer settings are invalid")
        if not 0.0 < self.probability_clip < 0.5:
            raise DiseaseReadoutError("probability clip must lie in (0, 0.5)")
        if self.minimum_disclosable_cell_count < 1:
            raise DiseaseReadoutError("minimum disclosable cell count must be positive")


@dataclass(frozen=True)
class DiseaseReadoutResult:
    task: str
    selected_penalty: float
    inner_primary_loss: float
    train_eligible_count: int
    test_eligible_count: int
    train_event_count: int | None
    test_event_count: int | None
    evaluation_set_sha256: str
    minimum_disclosable_cell_count: int
    metrics: Mapping[str, float | int | None]

    def aggregate_only_payload(self) -> dict[str, Any]:
        threshold = self.minimum_disclosable_cell_count
        train_count: int | None = self.train_eligible_count
        test_count: int | None = self.test_eligible_count
        train_events: int | None = self.train_event_count
        test_events: int | None = self.test_event_count
        if train_count < threshold:
            train_count = None
        if test_count < threshold:
            test_count = None
        if train_events is not None and (
            train_events < threshold or self.train_eligible_count - train_events < threshold
        ):
            train_events = None
        if test_events is not None and (
            test_events < threshold or self.test_eligible_count - test_events < threshold
        ):
            test_events = None
        return {
            "schema_version": READOUT_SCHEMA_VERSION,
            "task": self.task,
            "selected_penalty": self.selected_penalty,
            "inner_primary_loss": self.inner_primary_loss,
            "train_eligible_count": train_count,
            "test_eligible_count": test_count,
            "train_event_count": train_events,
            "test_event_count": test_events,
            "minimum_disclosable_cell_count": threshold,
            "small_cells_suppressed": any(
                raw is not None and disclosed is None
                for raw, disclosed in (
                    (self.train_eligible_count, train_count),
                    (self.test_eligible_count, test_count),
                    (self.train_event_count, train_events),
                    (self.test_event_count, test_events),
                )
            ),
            "evaluation_set_sha256": self.evaluation_set_sha256,
            "metrics": dict(self.metrics),
            "contains_patient_ids": False,
            "contains_predictions": False,
            "contains_per_patient_losses": False,
        }


@dataclass(frozen=True)
class _LinearModel:
    center: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float


def _validate_design(name: str, values: np.ndarray, rows: int) -> np.ndarray:
    design = np.asarray(values, dtype=np.float64)
    if design.ndim != 2 or design.shape[0] != rows or design.shape[1] < 1:
        raise DiseaseReadoutError(f"{name} must have shape [patients, positive features]")
    if not np.isfinite(design).all():
        raise DiseaseReadoutError(f"{name} contains non-finite values")
    return design


def _fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return center, scale, (x - center) / scale


def _fit_ridge(x: np.ndarray, y: np.ndarray, penalty: float) -> _LinearModel:
    center, scale, standardized = _fit_standardizer(x)
    intercept = float(y.mean())
    system = standardized.T @ standardized + penalty * np.eye(x.shape[1])
    rhs = standardized.T @ (y - intercept)
    try:
        coefficient = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.lstsq(system, rhs, rcond=None)[0]
    if not np.isfinite(coefficient).all():
        raise DiseaseReadoutError("ridge fit produced non-finite coefficients")
    return _LinearModel(center, scale, coefficient, intercept)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    penalty: float,
    config: DiseaseReadoutConfig,
) -> _LinearModel:
    if set(np.unique(y)) != {0.0, 1.0}:
        raise DiseaseReadoutError("logistic training requires both binary classes")
    center, scale, standardized = _fit_standardizer(x)
    design = np.column_stack((np.ones(len(x), dtype=np.float64), standardized))
    event_rate = float(np.clip(y.mean(), config.probability_clip, 1.0 - config.probability_clip))
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    parameters[0] = math.log(event_rate / (1.0 - event_rate))
    penalty_diagonal = np.concatenate(([0.0], np.full(x.shape[1], penalty)))

    def objective(candidate: np.ndarray) -> float:
        linear = design @ candidate
        return float(
            np.logaddexp(0.0, linear).sum()
            - y @ linear
            + 0.5 * np.sum(penalty_diagonal * candidate**2)
        )

    converged = False
    for _ in range(config.logistic_max_iterations):
        probability = _sigmoid(design @ parameters)
        gradient = design.T @ (probability - y) + penalty_diagonal * parameters
        weights = np.clip(probability * (1.0 - probability), 1e-9, None)
        hessian = design.T @ (design * weights[:, None])
        hessian.flat[:: hessian.shape[0] + 1] += penalty_diagonal
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if not np.isfinite(step).all():
            raise DiseaseReadoutError("logistic optimizer produced a non-finite step")
        current_objective = objective(parameters)
        step_scale = 1.0
        for _ in range(30):
            candidate = parameters - step_scale * step
            if objective(candidate) <= current_objective + 1e-12:
                parameters = candidate
                break
            step_scale *= 0.5
        else:
            raise DiseaseReadoutError("logistic optimizer could not find a descent step")
        if float(np.max(np.abs(step_scale * step))) <= config.logistic_tolerance:
            converged = True
            break
    if not converged:
        raise DiseaseReadoutError("logistic optimizer did not converge")
    if not np.isfinite(parameters).all():
        raise DiseaseReadoutError("logistic fit produced non-finite coefficients")
    return _LinearModel(center, scale, parameters[1:], float(parameters[0]))


def _linear_prediction(model: _LinearModel, x: np.ndarray) -> np.ndarray:
    prediction = model.intercept + ((x - model.center) / model.scale) @ model.coefficient
    if not np.isfinite(prediction).all():
        raise DiseaseReadoutError("readout prediction is non-finite")
    return prediction


def _binary_probabilities(
    model: _LinearModel,
    x: np.ndarray,
    config: DiseaseReadoutConfig,
) -> np.ndarray:
    return np.clip(
        _sigmoid(_linear_prediction(model, x)),
        config.probability_clip,
        1.0 - config.probability_clip,
    )


def _binary_log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    return float(
        -np.mean(y * np.log(probability) + (1.0 - y) * np.log1p(-probability))
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = y == 1.0
    positive_count = int(positives.sum())
    negative_count = len(y) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = _average_ranks(score)
    value = (
        float(ranks[positives].sum()) - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(value)


def _average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    positive_count = int((y == 1.0).sum())
    if positive_count == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    ordered_y = y[order]
    ordered_score = score[order]
    cumulative_positive = 0.0
    cumulative_count = 0
    area = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and ordered_score[stop] == ordered_score[start]:
            stop += 1
        group_positive = float(ordered_y[start:stop].sum())
        cumulative_positive += group_positive
        cumulative_count += stop - start
        area += (group_positive / positive_count) * (
            cumulative_positive / cumulative_count
        )
        start = stop
    return float(area)


def _binary_calibration(
    y: np.ndarray,
    probability: np.ndarray,
    config: DiseaseReadoutConfig,
) -> tuple[float | None, float | None]:
    if len(np.unique(y)) < 2:
        return None, None
    logit = np.log(probability) - np.log1p(-probability)
    if float(np.std(logit)) <= 1e-12:
        return None, None
    try:
        model = _fit_logistic(logit[:, None], y, 1e-8, config)
    except DiseaseReadoutError:
        return None, None
    slope = float(model.coefficient[0] / model.scale[0])
    intercept = float(model.intercept - model.center[0] * slope)
    return intercept, slope


def _continuous_calibration(y: np.ndarray, prediction: np.ndarray) -> tuple[float | None, float | None]:
    centered = prediction - prediction.mean()
    denominator = float(centered @ centered)
    if denominator <= 1e-12:
        return float(y.mean()), None
    slope = float(centered @ (y - y.mean()) / denominator)
    intercept = float(y.mean() - slope * prediction.mean())
    return intercept, slope


def _fit_and_primary_loss(
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    penalty: float,
    config: DiseaseReadoutConfig,
) -> float:
    if task == "binary":
        model = _fit_logistic(x_train, y_train, penalty, config)
        probability = _binary_probabilities(model, x_validation, config)
        return _binary_log_loss(y_validation, probability)
    model = _fit_ridge(x_train, y_train, penalty)
    prediction = _linear_prediction(model, x_validation)
    scale = float(np.std(y_train))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise DiseaseReadoutError("continuous inner-training target has zero variance")
    return float(np.mean(((prediction - y_validation) / scale) ** 2))


def evaluate_nested_readout(
    *,
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_eligible: np.ndarray,
    inner_fold_ids: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_eligible: np.ndarray,
    test_patient_id_hash: str,
    config: DiseaseReadoutConfig = DiseaseReadoutConfig(),
    _private_test_primary_loss_out: np.ndarray | None = None,
) -> DiseaseReadoutResult:
    """Tune on inner outer-training folds, refit once, and score outer test.

    The returned object is aggregate-only.  Outer-test labels never affect penalty
    selection, feature standardization, model fitting, or target scaling.  The
    optional private loss sink is for local paired inference only; it is filled
    in-place, contains no identifiers, and must never be serialized.
    """

    if task not in VALID_TASKS:
        raise DiseaseReadoutError(f"unknown task {task!r}")
    if len(test_patient_id_hash) != 64 or any(
        value not in "0123456789abcdef" for value in test_patient_id_hash
    ):
        raise DiseaseReadoutError("test patient identity hash must be lowercase SHA-256")
    train_y = np.asarray(y_train, dtype=np.float64)
    test_y = np.asarray(y_test, dtype=np.float64)
    train_mask = np.asarray(train_eligible)
    test_mask = np.asarray(test_eligible)
    folds = np.asarray(inner_fold_ids)
    if train_y.ndim != 1 or test_y.ndim != 1:
        raise DiseaseReadoutError("readout targets must be one-dimensional")
    if train_mask.shape != train_y.shape or train_mask.dtype != np.bool_:
        raise DiseaseReadoutError("train eligibility must be boolean and match targets")
    if test_mask.shape != test_y.shape or test_mask.dtype != np.bool_:
        raise DiseaseReadoutError("test eligibility must be boolean and match targets")
    if folds.shape != train_y.shape or not np.issubdtype(folds.dtype, np.integer):
        raise DiseaseReadoutError("inner fold IDs must be integer and match training targets")
    private_loss = None
    if _private_test_primary_loss_out is not None:
        private_loss = np.asarray(_private_test_primary_loss_out)
        if (
            private_loss.shape != test_y.shape
            or private_loss.dtype.kind != "f"
            or not private_loss.flags.writeable
        ):
            raise DiseaseReadoutError(
                "private test loss sink must be writable floating point and match test targets"
            )
        private_loss.fill(np.nan)
    train_x = _validate_design("x_train", x_train, len(train_y))
    test_x = _validate_design("x_test", x_test, len(test_y))
    if train_x.shape[1] != test_x.shape[1]:
        raise DiseaseReadoutError("train/test design dimensions differ")
    if not bool(train_mask.any()) or not bool(test_mask.any()):
        raise DiseaseReadoutError("train and test each require eligible target rows")
    if not np.isfinite(train_y[train_mask]).all() or not np.isfinite(test_y[test_mask]).all():
        raise DiseaseReadoutError("eligible targets must be finite")
    if task == "binary":
        for label, values in (("train", train_y[train_mask]), ("test", test_y[test_mask])):
            if not set(np.unique(values)).issubset({0.0, 1.0}):
                raise DiseaseReadoutError(f"{label} binary targets are not zero/one")
        if len(np.unique(train_y[train_mask])) != 2:
            raise DiseaseReadoutError("outer training requires both binary classes")

    populated_folds = tuple(sorted(int(value) for value in np.unique(folds[train_mask])))
    if len(populated_folds) < 2:
        raise DiseaseReadoutError("nested selection requires at least two inner folds")
    selected_penalty: float | None = None
    selected_loss: float | None = None
    selected_order: tuple[float, float] | None = None
    for penalty in config.penalty_grid:
        total_loss = 0.0
        total_count = 0
        for fold in populated_folds:
            inner_validation = train_mask & (folds == fold)
            inner_training = train_mask & (folds != fold)
            if not bool(inner_validation.any()) or int(inner_training.sum()) < 2:
                raise DiseaseReadoutError("an inner fold has an empty fit or validation set")
            fold_loss = _fit_and_primary_loss(
                task,
                train_x[inner_training],
                train_y[inner_training],
                train_x[inner_validation],
                train_y[inner_validation],
                float(penalty),
                config,
            )
            count = int(inner_validation.sum())
            total_loss += fold_loss * count
            total_count += count
        mean_loss = total_loss / total_count
        order = (mean_loss, -float(penalty))
        if selected_order is None or order < selected_order:
            selected_order = order
            selected_penalty = float(penalty)
            selected_loss = float(mean_loss)
    if selected_penalty is None or selected_loss is None:
        raise RuntimeError("internal readout-selection failure")

    eligible_train_x = train_x[train_mask]
    eligible_train_y = train_y[train_mask]
    eligible_test_x = test_x[test_mask]
    eligible_test_y = test_y[test_mask]
    if task == "binary":
        model = _fit_logistic(eligible_train_x, eligible_train_y, selected_penalty, config)
        probability = _binary_probabilities(model, eligible_test_x, config)
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
        if private_loss is not None:
            private_loss[test_mask] = -(
                eligible_test_y * np.log(probability)
                + (1.0 - eligible_test_y) * np.log1p(-probability)
            )
        train_event_count: int | None = int(eligible_train_y.sum())
        test_event_count: int | None = int(eligible_test_y.sum())
    else:
        model = _fit_ridge(eligible_train_x, eligible_train_y, selected_penalty)
        prediction = _linear_prediction(model, eligible_test_x)
        target_scale = float(np.std(eligible_train_y))
        if target_scale <= 1e-12:
            raise DiseaseReadoutError("outer-training continuous target has zero variance")
        squared_error = (prediction - eligible_test_y) ** 2
        null_squared_error = (eligible_test_y - float(eligible_train_y.mean())) ** 2
        null_sum = float(null_squared_error.sum())
        calibration_intercept, calibration_slope = _continuous_calibration(
            eligible_test_y, prediction
        )
        metrics = {
            "standardized_mean_squared_error": float(
                np.mean(squared_error / (target_scale**2))
            ),
            "r2_outer_training_intercept": (
                None if null_sum <= 1e-12 else float(1.0 - squared_error.sum() / null_sum)
            ),
            "mean_absolute_error": float(np.mean(np.abs(prediction - eligible_test_y))),
            "calibration_intercept": calibration_intercept,
            "calibration_slope": calibration_slope,
        }
        if private_loss is not None:
            private_loss[test_mask] = squared_error / (target_scale**2)
        train_event_count = None
        test_event_count = None

    if any(
        isinstance(value, float) and not np.isfinite(value)
        for value in metrics.values()
        if value is not None
    ):
        raise DiseaseReadoutError("aggregate readout metrics contain non-finite values")
    result = DiseaseReadoutResult(
        task=task,
        selected_penalty=selected_penalty,
        inner_primary_loss=selected_loss,
        train_eligible_count=int(train_mask.sum()),
        test_eligible_count=int(test_mask.sum()),
        train_event_count=train_event_count,
        test_event_count=test_event_count,
        evaluation_set_sha256=_hash_json(
            {
                "test_patient_id_hash": test_patient_id_hash,
                "eligible_mask": test_mask.tolist(),
            }
        ),
        minimum_disclosable_cell_count=config.minimum_disclosable_cell_count,
        metrics=MappingProxyType(metrics),
    )
    aggregate = result.aggregate_only_payload()
    if any(key in aggregate for key in ("patient_ids", "predictions", "losses")):
        raise RuntimeError("aggregate readout payload exposed patient-level material")
    if private_loss is not None:
        if (
            not np.isfinite(private_loss[test_mask]).all()
            or not np.isnan(private_loss[~test_mask]).all()
        ):
            raise RuntimeError("private readout losses are malformed")
    return result


def proper_score_increment(
    comparator: DiseaseReadoutResult,
    augmented: DiseaseReadoutResult,
) -> Mapping[str, float]:
    """Return comparator-minus-augmented proper scores; positive favors Atlas."""

    if comparator.task != augmented.task:
        raise DiseaseReadoutError("increment results use different tasks")
    if comparator.evaluation_set_sha256 != augmented.evaluation_set_sha256:
        raise DiseaseReadoutError("increment results were not scored on the same eligible rows")
    if comparator.task == "binary":
        names = ("log_loss", "brier_score")
    else:
        names = ("standardized_mean_squared_error",)
    return MappingProxyType(
        {
            name: float(comparator.metrics[name]) - float(augmented.metrics[name])
            for name in names
        }
    )


__all__ = [
    "DEFAULT_PENALTY_GRID",
    "DiseaseReadoutConfig",
    "DiseaseReadoutError",
    "DiseaseReadoutResult",
    "READOUT_SCHEMA_VERSION",
    "evaluate_nested_readout",
    "proper_score_increment",
]
