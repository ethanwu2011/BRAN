"""Held-out, aggregate-only evaluation for the frozen V5 residual CBC head.

This module deliberately accepts arrays and authenticated providers only.  It
does not load checkpoints, fit a model, write a result, or return participant
arrays.  The provider is invoked twice for every held-out fold so a separately
loaded frozen teacher/residual pair must replay both CBC heads exactly.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_multisource_batches_v2 import tensor, transform_hash
from bran_multisource_inference_v2 import completion_predictions
from bran_multisource_outcomes_v2 import original_age, subset_age
from bran_multisource_profiles_v3 import _unchanged, _validate_provider
from bran_v5_residual_cbc import ResidualCBC
from bran_v5_residual_training import PATTERNS, residual_completion_predictions
from bran_multisource_calibration_v2 import calibrate
import bran_multisource_calibration_metrics_v2 as calibration_metrics
import bran_multisource_outcome_metrics_v2 as outcome_metrics


_ERROR = "v5_residual_evaluation_contract_failed"
_ROLES = ("V5", "R")
_CALIBRATION_SLOTS = {"V5": "mlp", "R": "token"}
_DRAWS = 1000
_BOOTSTRAP_SEED = 96199
_MINIMUM = 20
_MIN_VALID_DRAWS = 900
_PROGRESS_PHASES = frozenset(("inference", "checkpoint_replay", "interval_fitting", "aggregate_bootstrap"))
_PROTECTED = ("single_target_hidden", "single_target_no_retina", "whole_cbc_no_retina")

# This is code closure only.  The runner records its own checkpoint/source
# closure independently, so a reporting correction cannot imply a refit.
CODE_DEPENDENCIES = (
    "bran_v5_residual_evaluation.py",
    "bran_clinical_semantics_v1.py",
    "bran_multisource_batches_v2.py",
    "bran_multisource_inference_v2.py",
    "bran_multisource_outcomes_v2.py",
    "bran_multisource_profiles_v3.py",
    "bran_v5_residual_cbc.py",
    "bran_v5_residual_training.py",
    "bran_multisource_calibration_v2.py",
    "bran_multisource_calibration_metrics_v2.py",
    "bran_multisource_outcome_metrics_v2.py",
)


def _require(ok: bool) -> None:
    if not ok:
        raise ValueError(_ERROR)


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _same(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[name], right[name]) for name in left)


def _residual_ok(teacher: torch.nn.Module, residual: object) -> dict[str, torch.Tensor]:
    """Verify the separately serialized residual is bound to this V5 head."""

    _require(isinstance(residual, ResidualCBC) and not residual.training)
    _require(all(not parameter.requires_grad for parameter in residual.parameters()))
    _require(isinstance(residual.baseline, torch.nn.Linear))
    _require(residual.baseline.in_features == 192 and residual.baseline.out_features == 9)
    _require(residual.baseline.weight.device.type == "cpu" and residual.baseline.weight.dtype == torch.float32)
    _require(_same(_snapshot(teacher.cbc_joint_head), _snapshot(residual.baseline)))
    state = _snapshot(residual)
    _require(all(bool(torch.isfinite(value).all()) for value in state.values()))
    return state


def _fullfold_counts(folds: np.ndarray) -> np.ndarray:
    """Fixed within-fold participant resampling counts; never returned."""

    _require(isinstance(folds, np.ndarray) and folds.ndim == 1 and folds.dtype.kind in "iu")
    _require(set(int(value) for value in np.unique(folds)) == set(range(5)))
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    counts = np.zeros((_DRAWS, folds.shape[0]), dtype=np.int64)
    for fold in range(5):
        rows = np.flatnonzero(folds == fold)
        _require(rows.shape[0] > 0)
        for draw in range(_DRAWS):
            counts[draw, rows] = np.bincount(
                rng.choice(rows.shape[0], rows.shape[0], replace=True), minlength=rows.shape[0]
            )
    return counts


def _interval(draws: np.ndarray) -> list[float] | None:
    _require(isinstance(draws, np.ndarray) and draws.shape == (_DRAWS,))
    finite = draws[np.isfinite(draws)]
    if finite.shape[0] < _MIN_VALID_DRAWS:
        return None
    return [float(value) for value in np.quantile(finite, (0.025, 0.975))]


def _standardized_macros(target: np.ndarray, support: Mapping[str, np.ndarray],
                         predictions: Mapping[str, Mapping[str, np.ndarray]], folds: np.ndarray,
                         counts: np.ndarray, fold_iqrs: np.ndarray) -> dict[str, dict[str, object]]:
    """Equal-target-field MAE macros; no original-unit cross-assay averaging."""

    _require(target.shape == (folds.shape[0], 9) and target.dtype.kind in "fiu")
    _require(fold_iqrs.shape == (5, 9) and bool(np.isfinite(fold_iqrs).all())
             and bool((fold_iqrs > 0.0).all()))
    result: dict[str, dict[str, object]] = {}
    scales = fold_iqrs[folds]
    for pattern in PATTERNS:
        mask = support[pattern]
        _require(isinstance(mask, np.ndarray) and mask.dtype == np.dtype(bool) and mask.shape == target.shape)
        indices = tuple(range(9)) if not pattern.startswith("red_cell") else (0, 1, 2, 3, 4, 6)
        targeted_fields = tuple(CBC_FIELDS[index] for index in indices)
        fields: dict[str, dict[str, object]] = {}
        point: dict[str, list[float]] = {role: [] for role in _ROLES}
        draws_by_role: dict[str, list[np.ndarray]] = {role: [] for role in _ROLES}
        for field_index, field in zip(indices, targeted_fields):
            rows = mask[:, field_index]
            if int(np.count_nonzero(rows)) < _MINIMUM:
                fields[field] = {"status": "unsupported"}
                continue
            weights = counts[:, rows].astype(np.float64)
            denominator = weights.sum(axis=1)
            errors: dict[str, np.ndarray] = {}
            field_draws: dict[str, np.ndarray] = {}
            valid = True
            for role in _ROLES:
                values = predictions[pattern][role]
                _require(values.shape == target.shape)
                error = np.abs(values[rows, field_index] - target[rows, field_index]) / scales[rows, field_index]
                if not bool(np.isfinite(error).all()):
                    valid = False
                    break
                errors[role] = error
                field_draws[role] = np.divide(weights @ error, denominator,
                                               out=np.full(_DRAWS, np.nan), where=denominator > 0.0)
                if _interval(field_draws[role]) is None:
                    valid = False
                    break
            if not valid:
                fields[field] = {"status": "unsupported"}
                continue
            delta_draws = field_draws["R"] - field_draws["V5"]
            if _interval(delta_draws) is None:
                fields[field] = {"status": "unsupported"}
                continue
            points = {role: float(np.mean(errors[role])) for role in _ROLES}
            fields[field] = {
                "status": "supported",
                "V5_standardized_mae": points["V5"],
                "R_standardized_mae": points["R"],
                "R_minus_V5_standardized_mae_delta": points["R"] - points["V5"],
            }
            for role in _ROLES:
                point[role].append(points[role])
                draws_by_role[role].append(field_draws[role])
        all_fields = all(fields[field]["status"] == "supported" for field in targeted_fields)
        if not all_fields:
            result[pattern] = {"status": "unsupported", "targeted_fields": list(targeted_fields),
                               "targeted_field_count": len(targeted_fields), "all_targeted_fields_supported": False,
                               "fields": fields, "macro": None}
            continue
        macro_draws = {role: np.mean(draws_by_role[role], axis=0) for role in _ROLES}
        delta_draws = macro_draws["R"] - macro_draws["V5"]
        intervals = {role: _interval(macro_draws[role]) for role in _ROLES}
        delta_interval = _interval(delta_draws)
        if any(value is None for value in intervals.values()) or delta_interval is None:
            # Different fields can lose different bootstrap draws. Their shared
            # macro may therefore have insufficient support despite individually
            # valid fields. This is unsupported evidence, not an execution error.
            result[pattern] = {"status": "unsupported", "targeted_fields": list(targeted_fields),
                               "targeted_field_count": len(targeted_fields), "all_targeted_fields_supported": True,
                               "fields": fields, "macro": None}
            continue
        macro = {
            "V5_standardized_mae": float(np.mean(point["V5"])),
            "V5_standardized_mae_ci95": intervals["V5"],
            "R_standardized_mae": float(np.mean(point["R"])),
            "R_standardized_mae_ci95": intervals["R"],
            "R_minus_V5_standardized_mae_delta": float(np.mean(point["R"]) - np.mean(point["V5"])),
            "R_minus_V5_standardized_mae_delta_ci95": delta_interval,
        }
        result[pattern] = {"status": "supported", "targeted_fields": list(targeted_fields),
                           "targeted_field_count": len(targeted_fields), "all_targeted_fields_supported": True,
                           "fields": fields, "macro": macro}
    return result


def _decision(macros: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    required = ("whole_cbc_hidden",) + _PROTECTED
    all_supported = all(macros[pattern]["status"] == "supported" for pattern in required)

    def entry(pattern: str, *, primary: bool) -> dict[str, object]:
        cell = macros[pattern]
        if cell["status"] != "supported":
            return {"status": "unsupported"}
        macro = cell["macro"]
        _require(isinstance(macro, dict))
        point = float(macro["R_minus_V5_standardized_mae_delta"])
        interval = macro["R_minus_V5_standardized_mae_delta_ci95"]
        _require(isinstance(interval, list) and len(interval) == 2)
        if primary:
            return {"status": "supported", "R_minus_V5_standardized_mae_delta": point,
                    "ci95": list(interval), "upper_ci_below_zero": bool(interval[1] < 0.0)}
        return {"status": "supported", "R_minus_V5_standardized_mae_delta": point,
                "no_point_increase": bool(point <= 0.0)}

    primary = entry("whole_cbc_hidden", primary=True)
    protected = {pattern: entry(pattern, primary=False) for pattern in _PROTECTED}
    positive = bool(all_supported and primary.get("upper_ci_below_zero")
                    and all(item.get("no_point_increase") for item in protected.values()))
    return {
        "all_required_standardized_macros_supported": all_supported,
        "primary_whole_cbc_hidden": primary,
        "protected_no_point_increase": protected,
        "eligible_positive_result": positive,
        "candidate_promoted": False,
        "automatic_promotion": False,
    }


def _remap_calibration(value: Mapping[str, Any], *, to_internal: bool = False) -> dict[str, Any]:
    """Rename metric slots only; both heads use the same anchored V5 teacher."""

    mapping = _CALIBRATION_SLOTS if to_internal else {value: key for key, value in _CALIBRATION_SLOTS.items()}
    result = copy.deepcopy(value)
    _require(isinstance(result, dict) and set(result) == set(PATTERNS))
    for fields in result.values():
        _require(isinstance(fields, dict) and set(fields) == set(CBC_FIELDS))
        for cell in fields.values():
            _require(isinstance(cell, dict) and isinstance(cell.get("groups"), dict))
            for group in cell["groups"].values():
                if "arms" in group:
                    _require(isinstance(group["arms"], dict) and set(group["arms"]) == set(mapping))
                    group["arms"] = {mapping[key]: item for key, item in group["arms"].items()}
    return result


def _finite(value: object, *, nonnegative: bool = False) -> bool:
    return (type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(float(value))
            and (not nonnegative or float(value) >= 0.0))


def _ci(value: object, *, lower: float = -math.inf) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_finite(item) for item in value) and lower <= value[0] <= value[1]


def _coverage(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if set(value) == {"status"}:
        return value["status"] == "withheld"
    return (set(value) == {"status", "supported", "total"} and value["status"] == "released"
            and type(value["supported"]) is int and type(value["total"]) is int and value["total"] >= _MINIMUM
            and 0 <= value["supported"] <= value["total"]
            and (value["supported"] == 0 or value["supported"] >= _MINIMUM)
            and (value["total"] - value["supported"] == 0 or value["total"] - value["supported"] >= _MINIMUM))


def _fullfold_group(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if set(value) == {"status"}:
        return value["status"] == "unsupported"
    if set(value) != {"status", "arms", "contrasts"} or value["status"] != "supported":
        return False
    if not isinstance(value["arms"], dict) or set(value["arms"]) != set(_ROLES):
        return False
    if not isinstance(value["contrasts"], dict) or set(value["contrasts"]) != {"R_minus_V5"}:
        return False
    for item in value["arms"].values():
        if not (isinstance(item, dict) and set(item) == {"mae", "ci95", "bias", "rmse"}
                and _finite(item["mae"], nonnegative=True) and _finite(item["bias"])
                and _finite(item["rmse"], nonnegative=True) and _ci(item["ci95"], lower=0.0)):
            return False
    contrast = value["contrasts"]["R_minus_V5"]
    if not (isinstance(contrast, dict) and set(contrast) == {"delta", "ci95"}
            and _finite(contrast["delta"]) and _ci(contrast["ci95"])):
        return False
    v5, residual = value["arms"]["V5"], value["arms"]["R"]
    return (abs(contrast["delta"] - (residual["mae"] - v5["mae"])) <= 1e-12
            and residual["rmse"] + 1e-12 >= residual["mae"] and v5["rmse"] + 1e-12 >= v5["mae"]
            and abs(residual["bias"]) <= residual["mae"] + 1e-12
            and abs(v5["bias"]) <= v5["mae"] + 1e-12)


def _fullfold_patterns_ok(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != set(PATTERNS):
        return False
    for fields in value.values():
        if not isinstance(fields, dict) or set(fields) != set(CBC_FIELDS):
            return False
        for field, cell in fields.items():
            expected_groups = {"overall", "hb_below_12_research_stratum"} if field == "hemoglobin" else {"overall"}
            if not (isinstance(cell, dict) and set(cell) == {"prediction_coverage_among_observed", "groups"}
                    and _coverage(cell["prediction_coverage_among_observed"])
                    and isinstance(cell["groups"], dict) and set(cell["groups"]) == expected_groups
                    and all(_fullfold_group(group) for group in cell["groups"].values())):
                return False
    return True


def _macro_ok(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != set(PATTERNS):
        return False
    for pattern, item in value.items():
        expected_indices = tuple(range(9)) if not pattern.startswith("red_cell") else (0, 1, 2, 3, 4, 6)
        expected_fields = tuple(CBC_FIELDS[index] for index in expected_indices)
        if not isinstance(item, dict) or set(item) != {"status", "targeted_fields", "targeted_field_count", "all_targeted_fields_supported", "fields", "macro"}:
            return False
        if (item["targeted_fields"] != list(expected_fields) or item["targeted_field_count"] != len(expected_fields)
                or type(item["all_targeted_fields_supported"]) is not bool or not isinstance(item["fields"], dict)
                or set(item["fields"]) != set(expected_fields)):
            return False
        supported = []
        for field in expected_fields:
            cell = item["fields"][field]
            if not isinstance(cell, dict):
                return False
            if set(cell) == {"status"}:
                if cell["status"] != "unsupported":
                    return False
                supported.append(False)
            elif set(cell) == {"status", "V5_standardized_mae", "R_standardized_mae", "R_minus_V5_standardized_mae_delta"}:
                if cell["status"] != "supported" or not (_finite(cell["V5_standardized_mae"], nonnegative=True)
                        and _finite(cell["R_standardized_mae"], nonnegative=True)
                        and _finite(cell["R_minus_V5_standardized_mae_delta"])):
                    return False
                if abs((cell["R_standardized_mae"] - cell["V5_standardized_mae"])
                       - cell["R_minus_V5_standardized_mae_delta"]) > 1e-12:
                    return False
                supported.append(True)
            else:
                return False
        all_supported = all(supported)
        if item["all_targeted_fields_supported"] != all_supported:
            return False
        if item["status"] == "unsupported":
            if item["macro"] is not None: return False
            continue
        if not all_supported: return False
        macro = item["macro"]
        expected = {"V5_standardized_mae", "V5_standardized_mae_ci95", "R_standardized_mae", "R_standardized_mae_ci95", "R_minus_V5_standardized_mae_delta", "R_minus_V5_standardized_mae_delta_ci95"}
        if not (item["status"] == "supported" and isinstance(macro, dict) and set(macro) == expected
                and _finite(macro["V5_standardized_mae"], nonnegative=True)
                and _finite(macro["R_standardized_mae"], nonnegative=True)
                and _finite(macro["R_minus_V5_standardized_mae_delta"])
                and _ci(macro["V5_standardized_mae_ci95"], lower=0.0)
                and _ci(macro["R_standardized_mae_ci95"], lower=0.0)
                and _ci(macro["R_minus_V5_standardized_mae_delta_ci95"])):
            return False
        if abs((macro["R_standardized_mae"] - macro["V5_standardized_mae"])
               - macro["R_minus_V5_standardized_mae_delta"]) > 1e-12:
            return False
        for role, key in (("V5", "V5_standardized_mae"), ("R", "R_standardized_mae")):
            if abs(macro[key] - float(np.mean([item["fields"][field][key] for field in expected_fields]))) > 1e-12:
                return False
    return True


def _decision_ok(value: object, macros: Mapping[str, Any]) -> bool:
    expected = {"all_required_standardized_macros_supported", "primary_whole_cbc_hidden",
                "protected_no_point_increase", "eligible_positive_result", "candidate_promoted", "automatic_promotion"}
    if not isinstance(value, dict) or set(value) != expected or any(type(value[key]) is not bool for key in (
            "all_required_standardized_macros_supported", "eligible_positive_result", "candidate_promoted", "automatic_promotion")):
        return False
    if value["candidate_promoted"] or value["automatic_promotion"]:
        return False
    primary = value["primary_whole_cbc_hidden"]
    protected = value["protected_no_point_increase"]
    if not isinstance(protected, dict) or set(protected) != set(_PROTECTED) or not isinstance(primary, dict):
        return False
    required = ("whole_cbc_hidden",) + _PROTECTED
    all_supported = all(macros[pattern]["status"] == "supported" for pattern in required)
    if value["all_required_standardized_macros_supported"] != all_supported:
        return False
    if macros["whole_cbc_hidden"]["status"] == "supported":
        macro = macros["whole_cbc_hidden"]["macro"]
        if not (set(primary) == {"status", "R_minus_V5_standardized_mae_delta", "ci95", "upper_ci_below_zero"}
                and primary["status"] == "supported" and _finite(primary["R_minus_V5_standardized_mae_delta"])
                and _ci(primary["ci95"]) and type(primary["upper_ci_below_zero"]) is bool
                and primary["R_minus_V5_standardized_mae_delta"] == macro["R_minus_V5_standardized_mae_delta"]
                and primary["ci95"] == macro["R_minus_V5_standardized_mae_delta_ci95"]
                and primary["upper_ci_below_zero"] == (primary["ci95"][1] < 0.0)):
            return False
    elif primary != {"status": "unsupported"}:
        return False
    for pattern in _PROTECTED:
        item = protected[pattern]
        if macros[pattern]["status"] == "supported":
            macro = macros[pattern]["macro"]
            if not (isinstance(item, dict) and set(item) == {"status", "R_minus_V5_standardized_mae_delta", "no_point_increase"}
                    and item["status"] == "supported" and _finite(item["R_minus_V5_standardized_mae_delta"])
                    and item["R_minus_V5_standardized_mae_delta"] == macro["R_minus_V5_standardized_mae_delta"]
                    and type(item["no_point_increase"]) is bool
                    and item["no_point_increase"] == (item["R_minus_V5_standardized_mae_delta"] <= 0.0)):
                return False
        elif item != {"status": "unsupported"}:
            return False
    expected_positive = bool(all_supported and primary.get("upper_ci_below_zero")
                             and all(item.get("no_point_increase") for item in protected.values()))
    return value["eligible_positive_result"] == expected_positive


def _contains_array(value: object) -> bool:
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return True
    if isinstance(value, dict):
        return any(_contains_array(key) or _contains_array(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_array(item) for item in value)
    return False


def validate_result(value: Any) -> None:
    """Reject non-finite, row-level, unknown-key, or arithmetically false output."""

    top = {
        "schema", "status", "fullfold_patterns", "standardized_macros", "calibration_patterns", "decision",
        "interval_level", "bootstrap_draws", "bootstrap_seed", "interval_bootstrap_seed", "minimum_valid_draws",
        "minimum_calibration_values", "fullfold_denominator", "calibration_denominator", "confidence_interval_semantics",
        "separate_half_roles", "interval_radii_exported", "no_automatic_promotion",
        "same_target_masks", "same_support_masks", "checkpoint_predictions_replayed", "teacher_parameters_unchanged",
        "residual_parameters_unchanged", "encoder_and_screening_unchanged", "original_cbc_weights_unchanged",
        "intervals_symmetric_residual", "scoring_targets_used_to_fit_intervals", "all_reused_development_not_external",
        "candidate_promoted", "patient_level_output_emitted",
    }
    try:
        if not isinstance(value, dict) or set(value) != top or _contains_array(value):
            _require(False)
        _require(value["schema"] == "bran-v5-residual-cbc-evaluation-v1" and value["status"] == "completed_not_promoted")
        _require(value["interval_level"] == 0.9 and value["bootstrap_draws"] == _DRAWS
                 and value["bootstrap_seed"] == _BOOTSTRAP_SEED and value["interval_bootstrap_seed"] == 94701
                 and value["minimum_valid_draws"] == _MIN_VALID_DRAWS and value["minimum_calibration_values"] == _MINIMUM)
        _require(value["fullfold_denominator"] == "all_heldout_rows_with_supported_observed_erased_targets")
        _require(value["calibration_denominator"] == "within_heldout_fold_scoring_role_only")
        _require(value["confidence_interval_semantics"] == {
            "fullfold_arm_ci95": "absolute_original_unit_mae",
            "fullfold_R_minus_V5_ci95": "paired_original_unit_mae_difference",
            "standardized_arm_ci95": "absolute_equal_field_standardized_mae",
            "standardized_R_minus_V5_ci95": "paired_equal_field_standardized_mae_difference",
            "calibration_mae_ci95": "absolute_scoring_half_original_unit_mae",
        })
        for key in ("separate_half_roles", "interval_radii_exported", "no_automatic_promotion",
                    "same_target_masks", "same_support_masks", "checkpoint_predictions_replayed",
                    "teacher_parameters_unchanged", "residual_parameters_unchanged", "encoder_and_screening_unchanged",
                    "original_cbc_weights_unchanged", "intervals_symmetric_residual",
                    "scoring_targets_used_to_fit_intervals", "all_reused_development_not_external",
                    "candidate_promoted", "patient_level_output_emitted"):
            _require(type(value[key]) is bool)
        _require(value["separate_half_roles"] and not value["interval_radii_exported"] and value["no_automatic_promotion"]
                 and value["same_target_masks"] and value["same_support_masks"] and value["checkpoint_predictions_replayed"]
                 and value["teacher_parameters_unchanged"] and value["residual_parameters_unchanged"]
                 and value["encoder_and_screening_unchanged"] and value["original_cbc_weights_unchanged"]
                 and value["intervals_symmetric_residual"] and not value["scoring_targets_used_to_fit_intervals"]
                 and value["all_reused_development_not_external"] and not value["candidate_promoted"]
                 and not value["patient_level_output_emitted"])
        _require(_fullfold_patterns_ok(value["fullfold_patterns"]))
        _require(_macro_ok(value["standardized_macros"]))
        calibration_metrics.validate_result(_remap_calibration(value["calibration_patterns"], to_internal=True))
        _require(_decision_ok(value["decision"], value["standardized_macros"]))
    except Exception:
        raise ValueError(_ERROR) from None


def evaluate(paired: object, roles: np.ndarray, provider: object, progress=None) -> dict[str, object]:
    """Evaluate frozen V5 and residual CBC heads on the five held-out folds.

    ``provider(fold)`` returns ``(teacher, inherited_transform, residual)``.
    Role zero/one is an already authenticated, within-held-out-fold
    calibration/scoring split.  Full-fold completion estimates intentionally do
    not use this split; interval diagnostics intentionally do.
    """

    try:
        _require(callable(provider) and (progress is None or callable(progress)))
        folds = paired.folds
        n = len(folds)
        _require(isinstance(roles, np.ndarray) and roles.shape == (n,) and roles.dtype.kind in "iu")
        calibration_metrics._validate_split(folds, roles)
        _require(len(PATTERNS) == 6 and tuple(PATTERNS) == tuple(calibration_metrics._PATTERNS))
        slots = tuple(paired.names.index(field) for field in CBC_FIELDS)
        _require(len(slots) == 9 and len(set(slots)) == 9)
        eligible = np.zeros(59, dtype=bool)
        eligible[list(paired.eligible_indices)] = True
        target = np.asarray(paired.c[:, slots], dtype=np.float64).copy()
        intended = {pattern: (np.asarray(paired.cm, dtype=bool) & eligible[None, :])[:, slots].copy()
                    for pattern in PATTERNS}
        for pattern in PATTERNS:
            if pattern.startswith("red_cell"):
                intended[pattern][:, (5, 7, 8)] = False
            _require(bool(np.isfinite(target[intended[pattern]]).all()))
        age = original_age(paired)
        support = {pattern: np.zeros((n, 9), dtype=bool) for pattern in PATTERNS}
        predictions = {pattern: {role: np.full((n, 9), np.nan, dtype=np.float64) for role in _ROLES}
                       for pattern in PATTERNS}
        fold_iqrs = np.full((5, 9), np.nan, dtype=np.float64)
        retained: list[object] = []

        def emit(phase: str, fold: int) -> None:
            _require(phase in _PROGRESS_PHASES and type(fold) is int and 0 <= fold <= 4)
            if progress is not None:
                progress({"phase": phase, "fold": fold})

        for fold in range(5):
            rows = np.flatnonzero(folds == fold)
            _require(rows.shape[0] > 0)
            local_age = subset_age(age, rows)
            emit("inference", fold)
            teacher, transform, residual = provider(fold)
            _require(not any(teacher is prior or residual is prior for prior in retained))
            retained.extend((teacher, residual))
            teacher_before, transform_before, teacher_grads = _validate_provider(
                teacher, transform, fold, slots, paired.transforms[fold]
            )
            residual_before = _residual_ok(teacher, residual)
            iqr = np.asarray(transform.clinical_iqr[list(slots)], dtype=np.float64)
            median = np.asarray(transform.clinical_median[list(slots)], dtype=np.float64)
            _require(iqr.shape == median.shape == (9,) and bool(np.isfinite(iqr).all())
                     and bool(np.isfinite(median).all()) and bool((iqr > 0.0).all()))
            fold_iqrs[fold] = iqr
            c, cm = transform.clinical(paired.c, paired.cm)
            r, rm = transform.retinal(paired.r, paired.rm)
            args = (tensor(c[rows]), tensor(cm[rows], torch.bool), tensor(r[rows]), tensor(rm[rows], torch.bool),
                    local_age, transform.age_mean, transform.age_scale)
            for pattern in PATTERNS:
                native = completion_predictions(teacher, *args, pattern, slots)
                candidate = residual_completion_predictions(teacher, residual, *args, pattern, slots)
                native_target = native.targetmask.numpy()
                candidate_target = candidate.targetmask.numpy()
                native_support = native.scoring_target_mask.numpy()
                candidate_support = candidate.scoring_target_mask.numpy()
                native_values = native.cbc_standardized.numpy()
                candidate_values = candidate.cbc_standardized.numpy()
                _require(np.array_equal(native_target, intended[pattern][rows])
                         and np.array_equal(candidate_target, intended[pattern][rows])
                         and np.array_equal(native_support, candidate_support)
                         and not bool(np.any(native_support & ~native_target))
                         and np.array_equal(np.isfinite(native_values), native_support)
                         and np.array_equal(np.isfinite(candidate_values), candidate_support))
                support[pattern][rows] = native_support
                predictions[pattern]["V5"][rows] = native_values * iqr + median
                predictions[pattern]["R"][rows] = candidate_values * iqr + median
            _require(_unchanged(teacher_before, transform_before, teacher_grads, teacher, transform)
                     and _same(residual_before, _residual_ok(teacher, residual)))

            emit("checkpoint_replay", fold)
            replay_teacher, replay_transform, replay_residual = provider(fold)
            _require(not any(replay_teacher is prior or replay_residual is prior for prior in retained))
            retained.extend((replay_teacher, replay_residual))
            replay_before, replay_hash, replay_grads = _validate_provider(
                replay_teacher, replay_transform, fold, slots, paired.transforms[fold]
            )
            replay_residual_before = _residual_ok(replay_teacher, replay_residual)
            _require(replay_hash == transform_before and _same(teacher_before, replay_before)
                     and _same(residual_before, replay_residual_before))
            replay_iqr = np.asarray(replay_transform.clinical_iqr[list(slots)], dtype=np.float64)
            replay_median = np.asarray(replay_transform.clinical_median[list(slots)], dtype=np.float64)
            _require(np.array_equal(replay_iqr, iqr) and np.array_equal(replay_median, median))
            rc, rcm = replay_transform.clinical(paired.c, paired.cm)
            rr, rrm = replay_transform.retinal(paired.r, paired.rm)
            replay_args = (tensor(rc[rows]), tensor(rcm[rows], torch.bool), tensor(rr[rows]), tensor(rrm[rows], torch.bool),
                           local_age, replay_transform.age_mean, replay_transform.age_scale)
            for pattern in PATTERNS:
                native = completion_predictions(replay_teacher, *replay_args, pattern, slots)
                candidate = residual_completion_predictions(replay_teacher, replay_residual, *replay_args, pattern, slots)
                _require(np.array_equal(native.targetmask.numpy(), intended[pattern][rows])
                         and np.array_equal(candidate.targetmask.numpy(), intended[pattern][rows])
                         and np.array_equal(native.scoring_target_mask.numpy(), support[pattern][rows])
                         and np.array_equal(candidate.scoring_target_mask.numpy(), support[pattern][rows]))
                replay_v5 = native.cbc_standardized.numpy() * replay_iqr + replay_median
                replay_r = candidate.cbc_standardized.numpy() * replay_iqr + replay_median
                _require(np.array_equal(replay_v5, predictions[pattern]["V5"][rows], equal_nan=True)
                         and np.array_equal(replay_r, predictions[pattern]["R"][rows], equal_nan=True))
            _require(_unchanged(replay_before, replay_hash, replay_grads, replay_teacher, replay_transform)
                     and _same(replay_residual_before, _residual_ok(replay_teacher, replay_residual)))

        # This count matrix is shared by every original-unit and standardized
        # full-fold comparison, preserving the pairing across heads and fields.
        emit("aggregate_bootstrap", 0)
        counts = _fullfold_counts(folds)
        fullfold = {
            pattern: outcome_metrics.completion(
                {role: np.where(support[pattern], predictions[pattern][role], np.nan) for role in _ROLES},
                target, intended[pattern], folds, counts, contrasts={"R_minus_V5": ("R", "V5")}, low_hb=True
            ) for pattern in PATTERNS
        }
        macros = _standardized_macros(target, support, predictions, folds, counts, fold_iqrs)

        emit("interval_fitting", 0)
        internal = {pattern: {_CALIBRATION_SLOTS[role]: values for role, values in predictions[pattern].items()}
                    for pattern in PATTERNS}
        intervals = calibrate(target, support, internal, folds, roles)
        calibration = _remap_calibration(calibration_metrics.summarize(
            target, intended, support, internal, intervals, folds, roles
        ))
        result = {
            "schema": "bran-v5-residual-cbc-evaluation-v1",
            "status": "completed_not_promoted",
            "fullfold_patterns": fullfold,
            "standardized_macros": macros,
            "calibration_patterns": calibration,
            "decision": _decision(macros),
            "interval_level": 0.9,
            "bootstrap_draws": _DRAWS,
            "bootstrap_seed": _BOOTSTRAP_SEED,
            "interval_bootstrap_seed": 94701,
            "minimum_valid_draws": _MIN_VALID_DRAWS,
            "minimum_calibration_values": _MINIMUM,
            "fullfold_denominator": "all_heldout_rows_with_supported_observed_erased_targets",
            "calibration_denominator": "within_heldout_fold_scoring_role_only",
            "confidence_interval_semantics": {
                "fullfold_arm_ci95": "absolute_original_unit_mae",
                "fullfold_R_minus_V5_ci95": "paired_original_unit_mae_difference",
                "standardized_arm_ci95": "absolute_equal_field_standardized_mae",
                "standardized_R_minus_V5_ci95": "paired_equal_field_standardized_mae_difference",
                "calibration_mae_ci95": "absolute_scoring_half_original_unit_mae",
            },
            "separate_half_roles": True,
            "interval_radii_exported": False,
            "no_automatic_promotion": True,
            "same_target_masks": True,
            "same_support_masks": True,
            "checkpoint_predictions_replayed": True,
            "teacher_parameters_unchanged": True,
            "residual_parameters_unchanged": True,
            "encoder_and_screening_unchanged": True,
            "original_cbc_weights_unchanged": True,
            "intervals_symmetric_residual": True,
            "scoring_targets_used_to_fit_intervals": False,
            "all_reused_development_not_external": True,
            "candidate_promoted": False,
            "patient_level_output_emitted": False,
        }
        validate_result(result)
        return result
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(_ERROR) from None
