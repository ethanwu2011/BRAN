"""Memory-efficient nested logistic probe for high-dimensional frozen comparators.

This solves the same standardized, intercept-unpenalized ridge-logistic
objective as :mod:`patient_atlas_disease_group_readout`, but uses deterministic
L-BFGS instead of materialising a dense feature Hessian.  The latter is
prohibitively expensive for Labrador's 1,024-dimensional encoder output.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Sequence

import numpy as np

from patient_atlas_disease_group_readout import GroupedDiseaseReadoutResult
from patient_atlas_disease_readout import (
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


FOUNDATION_READOUT_OPTIMIZER = "scipy_lbfgsb_then_newtoncg_same_objective_v2"


def _fit_parameters(
    standardized: np.ndarray,
    y: np.ndarray,
    penalties: np.ndarray,
    config: DiseaseReadoutConfig,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    try:
        from scipy.optimize import minimize
    except ImportError as error:  # pragma: no cover - runtime-specific
        raise DiseaseReadoutError("SciPy is required for foundation readouts") from error
    n, dimension = standardized.shape
    if penalties.shape != (dimension,) or bool((penalties < 0.0).any()):
        raise DiseaseReadoutError("foundation feature penalties are malformed")
    if initial is None:
        event_rate = float(
            np.clip(y.mean(), config.probability_clip, 1.0 - config.probability_clip)
        )
        candidate = np.zeros(dimension + 1, dtype=np.float64)
        candidate[0] = math.log(event_rate / (1.0 - event_rate))
    else:
        candidate = np.asarray(initial, dtype=np.float64).copy()
        if candidate.shape != (dimension + 1,) or not np.isfinite(candidate).all():
            raise DiseaseReadoutError("foundation optimizer initial state is malformed")

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = parameters[0] + standardized @ parameters[1:]
        residual = _sigmoid(linear) - y
        value = (
            np.logaddexp(0.0, linear).sum()
            - y @ linear
            + 0.5 * np.sum(penalties * parameters[1:] ** 2)
        ) / n
        gradient = np.empty_like(parameters)
        gradient[0] = residual.mean()
        gradient[1:] = (
            standardized.T @ residual + penalties * parameters[1:]
        ) / n
        return float(value), gradient

    result = minimize(
        objective,
        candidate,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": max(200, config.logistic_max_iterations * 5),
            "maxls": 50,
            "ftol": 1e-12,
            "gtol": min(1e-7, config.logistic_tolerance * 10.0),
        },
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    gradient = np.asarray(result.jac, dtype=np.float64)
    if not np.isfinite(parameters).all() or not np.isfinite(gradient).all():
        raise DiseaseReadoutError("foundation L-BFGS fit is non-finite")
    # SciPy can report a line-search warning at an already stationary point.
    # Accept only a strict first-order certificate in that case.
    if bool(result.success) or float(np.max(np.abs(gradient))) <= 5e-6:
        return parameters

    # The objective is convex but Labrador is wide and highly collinear.  Use
    # the L-BFGS state as a deterministic warm start for Newton-CG with an exact
    # Hessian-vector product.  This changes only the numerical solver, not the
    # objective, regularization, standardization, folds, or selection rule.
    def hessian_product(candidate: np.ndarray, direction: np.ndarray) -> np.ndarray:
        linear = candidate[0] + standardized @ candidate[1:]
        probability = _sigmoid(linear)
        weights = probability * (1.0 - probability)
        projected = direction[0] + standardized @ direction[1:]
        output = np.empty_like(direction)
        output[0] = np.mean(weights * projected)
        output[1:] = (
            standardized.T @ (weights * projected)
            + penalties * direction[1:]
        ) / n
        return output

    fallback = minimize(
        objective,
        parameters,
        method="Newton-CG",
        jac=True,
        hessp=hessian_product,
        options={
            "maxiter": max(200, config.logistic_max_iterations * 5),
            "xtol": min(1e-8, config.logistic_tolerance * 10.0),
        },
    )
    fallback_parameters = np.asarray(fallback.x, dtype=np.float64)
    fallback_gradient = np.asarray(fallback.jac, dtype=np.float64)
    if (
        not np.isfinite(fallback_parameters).all()
        or not np.isfinite(fallback_gradient).all()
        or (
            not bool(fallback.success)
            and float(np.max(np.abs(fallback_gradient))) > 5e-6
        )
    ):
        raise DiseaseReadoutError("foundation convex optimizer did not converge")
    return fallback_parameters


def _feature_penalties(
    groups: np.ndarray,
    *,
    active_group: str,
    penalty: float,
    unpenalized: frozenset[str],
) -> np.ndarray:
    result = np.zeros(len(groups), dtype=np.float64)
    for index, group in enumerate(groups):
        name = str(group)
        if name not in unpenalized:
            if name != active_group:
                raise DiseaseReadoutError("foundation readout has multiple penalized groups")
            result[index] = float(penalty)
    return result


def evaluate_nested_foundation_logistic_readout(
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
    _private_test_primary_loss_out: np.ndarray | None = None,
) -> GroupedDiseaseReadoutResult:
    """Nested one-block probe with no dense Hessian and no exposed predictions."""

    train_y = np.asarray(y_train, dtype=np.float64)
    test_y = np.asarray(y_test, dtype=np.float64)
    train_mask = np.asarray(train_eligible)
    test_mask = np.asarray(test_eligible)
    folds = np.asarray(inner_fold_ids)
    train_x = _validate_design("x_train", x_train, len(train_y))
    test_x = _validate_design("x_test", x_test, len(test_y))
    groups = np.asarray(tuple(str(value) for value in feature_groups), dtype=object)
    unpenalized = frozenset(str(value) for value in unpenalized_groups)
    if train_x.shape[1] != test_x.shape[1] or groups.shape != (train_x.shape[1],):
        raise DiseaseReadoutError("foundation readout dimensions differ")
    if not unpenalized or not unpenalized.issubset(set(groups.tolist())):
        raise DiseaseReadoutError("foundation unpenalized groups differ")
    penalized_groups = tuple(sorted(set(groups.tolist()) - unpenalized))
    if len(penalized_groups) != 1:
        raise DiseaseReadoutError("foundation readout requires exactly one modality block")
    active_group = penalized_groups[0]
    if not bool(np.any(np.std(train_x[train_mask][:, groups == active_group], axis=0) > 1e-12)):
        raise DiseaseReadoutError("foundation modality block is constant")
    if (
        train_y.ndim != 1
        or test_y.ndim != 1
        or train_mask.shape != train_y.shape
        or test_mask.shape != test_y.shape
        or train_mask.dtype != np.bool_
        or test_mask.dtype != np.bool_
        or folds.shape != train_y.shape
        or not np.issubdtype(folds.dtype, np.integer)
    ):
        raise DiseaseReadoutError("foundation target or fold arrays are malformed")
    if len(test_patient_id_hash) != 64 or any(
        value not in "0123456789abcdef" for value in test_patient_id_hash
    ):
        raise DiseaseReadoutError("foundation test identity hash is malformed")
    for label, values in (("train", train_y[train_mask]), ("test", test_y[test_mask])):
        if not set(np.unique(values)).issubset({0.0, 1.0}):
            raise DiseaseReadoutError(f"foundation {label} target is not binary")
    if len(np.unique(train_y[train_mask])) != 2:
        raise DiseaseReadoutError("foundation outer training lacks a binary class")
    populated_folds = tuple(sorted(int(value) for value in np.unique(folds[train_mask])))
    if len(populated_folds) < 2:
        raise DiseaseReadoutError("foundation readout needs two inner folds")

    loss_sums = {float(penalty): 0.0 for penalty in config.penalty_grid}
    loss_counts = {float(penalty): 0 for penalty in config.penalty_grid}
    for fold in populated_folds:
        validation = train_mask & (folds == fold)
        training = train_mask & (folds != fold)
        if not bool(validation.any()) or len(np.unique(train_y[training])) != 2:
            raise DiseaseReadoutError("foundation inner fold lacks rows or a binary class")
        center, scale, standardized = _fit_standardizer(train_x[training])
        validation_x = (train_x[validation] - center) / scale
        initial: np.ndarray | None = None
        # Strong-to-weak warm starts reduce compute but do not alter the objective.
        for penalty in reversed(config.penalty_grid):
            feature_penalty = _feature_penalties(
                groups,
                active_group=active_group,
                penalty=float(penalty),
                unpenalized=unpenalized,
            )
            parameters = _fit_parameters(
                standardized,
                train_y[training],
                feature_penalty,
                config,
                initial,
            )
            initial = parameters
            probability = np.clip(
                _sigmoid(parameters[0] + validation_x @ parameters[1:]),
                config.probability_clip,
                1.0 - config.probability_clip,
            )
            count = int(validation.sum())
            loss_sums[float(penalty)] += (
                _binary_log_loss(train_y[validation], probability) * count
            )
            loss_counts[float(penalty)] += count
    candidate_losses = {
        penalty: loss_sums[penalty] / loss_counts[penalty]
        for penalty in loss_sums
    }
    selected_penalty = min(
        candidate_losses,
        key=lambda penalty: (candidate_losses[penalty], -penalty),
    )

    center, scale, standardized = _fit_standardizer(train_x[train_mask])
    final_feature_penalty = _feature_penalties(
        groups,
        active_group=active_group,
        penalty=selected_penalty,
        unpenalized=unpenalized,
    )
    parameters = _fit_parameters(
        standardized,
        train_y[train_mask],
        final_feature_penalty,
        config,
    )
    test_probability = np.clip(
        _sigmoid(
            parameters[0]
            + ((test_x[test_mask] - center) / scale) @ parameters[1:]
        ),
        config.probability_clip,
        1.0 - config.probability_clip,
    )
    eligible_test_y = test_y[test_mask]
    calibration_intercept, calibration_slope = _binary_calibration(
        eligible_test_y, test_probability, config
    )
    metrics = {
        "log_loss": _binary_log_loss(eligible_test_y, test_probability),
        "brier_score": float(np.mean((test_probability - eligible_test_y) ** 2)),
        "auroc": _auroc(eligible_test_y, test_probability),
        "average_precision": _average_precision(eligible_test_y, test_probability),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }
    if _private_test_primary_loss_out is not None:
        private = np.asarray(_private_test_primary_loss_out)
        if (
            private.shape != test_y.shape
            or private.dtype.kind != "f"
            or not private.flags.writeable
        ):
            raise DiseaseReadoutError("foundation private loss sink is malformed")
        private.fill(np.nan)
        private[test_mask] = -(
            eligible_test_y * np.log(test_probability)
            + (1.0 - eligible_test_y) * np.log1p(-test_probability)
        )
    return GroupedDiseaseReadoutResult(
        selected_penalties=MappingProxyType({active_group: float(selected_penalty)}),
        active_penalty_groups=(active_group,),
        inner_primary_loss=float(candidate_losses[selected_penalty]),
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
    )


__all__ = [
    "FOUNDATION_READOUT_OPTIMIZER",
    "evaluate_nested_foundation_logistic_readout",
]
