"""Closed, aggregate-only survey release wrapper for private KNHANES Hb predictions.

This module has no file, source, model, or fit I/O.  It treats predictions as
fixed when forming design-based intervals; model-selection and fitting
uncertainty are deliberately outside its interval interpretation.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from bran_knhanes_survey_v1 import estimate_mae_difference, estimate_mean
from bran_knhanes_v5_evaluation import ARMS, PrivateKNHANESPredictions


_ERROR = "knhanes_v5_metrics_contract_failed"
_SCHEMA = "bran-knhanes-v5-metrics-v1"
_LOWER_BOUND = ">=20"
_ARMS = tuple(ARMS)
_TOP_KEYS = {
    "schema", "fixed_prediction_design_intervals", "model_selection_uncertainty_included",
    "overall", "year_2022", "year_2023", "low_hb",
}


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError(_ERROR)


def _numeric(value: Any, rows: int | None = None) -> np.ndarray:
    _require(type(value) is np.ndarray and value.ndim == 1 and value.dtype.kind in "fiu")
    if rows is not None:
        _require(value.shape == (rows,))
    return value


def _mask(value: Any, rows: int | None = None) -> np.ndarray:
    _require(type(value) is np.ndarray and value.ndim == 1 and value.dtype == np.dtype(bool))
    if rows is not None:
        _require(value.shape == (rows,))
    return value


def _prediction_map(predictions: Any, support: np.ndarray) -> Mapping[str, np.ndarray]:
    if isinstance(predictions, PrivateKNHANESPredictions):
        _require(np.array_equal(predictions.support, support))
        mapping = predictions.predictions
    else:
        mapping = predictions
    _require(isinstance(mapping, Mapping) and set(mapping) == set(_ARMS))
    rows = support.shape[0]
    return {arm: _numeric(mapping[arm], rows) for arm in _ARMS}


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    _require(values.ndim == weights.ndim == 1 and values.size > 0
             and bool(np.isfinite(values).all()) and bool(np.isfinite(weights).all())
             and bool((weights > 0).all()))
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order] / np.sum(weights))
    result = float(ordered_values[np.searchsorted(cumulative, 0.5, side="left")])
    _require(np.isfinite(result))
    return result


def _survey_mean(value: np.ndarray, weight: np.ndarray, strata: np.ndarray,
                 psus: np.ndarray, domain: np.ndarray) -> Any:
    return estimate_mean(value, weight, strata, psus, domain)


def _filled(rows: int, domain: np.ndarray, values: np.ndarray) -> np.ndarray:
    output = np.full(rows, np.nan, dtype=np.float64)
    output[domain] = values
    return output


def _arm_metrics(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray,
                 strata: np.ndarray, psus: np.ndarray, domain: np.ndarray) -> dict[str, Any]:
    rows = target.shape[0]
    error = prediction[domain] - target[domain]
    absolute = np.abs(error)
    mae = _survey_mean(_filled(rows, domain, absolute), weight, strata, psus, domain)
    bias = _survey_mean(_filled(rows, domain, error), weight, strata, psus, domain)
    square = _survey_mean(_filled(rows, domain, error ** 2), weight, strata, psus, domain)
    rmse = float(np.sqrt(square.mean))
    _require(np.isfinite(rmse))
    return {
        "mae": {"estimate": float(mae.mean), "ci95": [float(mae.ci95[0]), float(mae.ci95[1])]},
        "signed_bias": float(bias.mean),
        "rmse": rmse,
        "median_absolute_error": _weighted_median(absolute, weight[domain]),
    }


def _coverage(target_eligible: np.ndarray, domain: np.ndarray, subgroup: np.ndarray,
              weight: np.ndarray) -> dict[str, Any]:
    denominator = target_eligible & subgroup
    supported = int(np.count_nonzero(domain))
    missing = int(np.count_nonzero(denominator & ~domain))
    if supported >= 20 and (missing == 0 or missing >= 20):
        proportion = float(np.sum(weight[domain]) / np.sum(weight[denominator]))
        _require(np.isfinite(proportion) and 0.0 <= proportion <= 1.0)
        return {"status": "released", "supported_count_lower_bound": _LOWER_BOUND,
                "nonsupported_count": "zero_or_at_least_20",
                "weighted_proportion": proportion}
    return {"status": "suppressed"}


def _release_allowed(domain: np.ndarray, folds: np.ndarray, *, overall: bool) -> bool:
    total_floor = 100 if overall else 20
    fold_floor = 20 if overall else 5
    return int(np.count_nonzero(domain)) >= total_floor and all(
        int(np.count_nonzero(domain & (folds == fold))) >= fold_floor for fold in range(5)
    )


def _suppressed() -> dict[str, str]:
    return {"status": "suppressed"}


def _released(target: np.ndarray, prediction_map: Mapping[str, np.ndarray], weight: np.ndarray,
              strata: np.ndarray, psus: np.ndarray, domain: np.ndarray,
              target_eligible: np.ndarray, subgroup: np.ndarray, *, primary: bool) -> dict[str, Any]:
    output: dict[str, Any] = {
        "status": "released",
        "support_count_lower_bound": _LOWER_BOUND,
        "exposure_coverage": _coverage(target_eligible, domain, subgroup, weight),
        "arms": {arm: _arm_metrics(target, prediction_map[arm], weight, strata, psus, domain) for arm in _ARMS},
    }
    if primary:
        paired = estimate_mae_difference(target, prediction_map["state_extratrees"],
                                         prediction_map["raw_extratrees"], weight, strata, psus, domain)
        output["paired_state_minus_raw_mae"] = {
            "estimate": float(paired.mean), "ci95": [float(paired.ci95[0]), float(paired.ci95[1])],
        }
    return output


def _design_contract(weight: np.ndarray, years: np.ndarray, psus: np.ndarray,
                     strata: np.ndarray, folds: np.ndarray) -> None:
    _require(bool(np.isfinite(weight).all()) and bool((weight > 0).all()))
    for values in (years, psus, strata, folds):
        _require(values.dtype.kind in "iu" and bool(np.isfinite(values).all()))
    _require(set(int(item) for item in np.unique(years)) == {2022, 2023})
    _require(set(int(item) for item in np.unique(folds)) == set(range(5)))
    for group in np.unique(psus):
        members = psus == group
        _require(np.unique(years[members]).size == 1 and np.unique(strata[members]).size == 1
                 and np.unique(folds[members]).size == 1)
    # Survey variance must always see the complete design, including PSUs that
    # have no target-domain contribution.  Singleton strata are not repaired.
    for stratum in np.unique(strata):
        _require(np.unique(years[strata == stratum]).size == 1)
        _require(np.unique(psus[strata == stratum]).size >= 2)


def summarize(predictions: Any, target: Any, support: Any, design_weights: Any, years: Any,
              psu_groups: Any, stratum_groups: Any, folds: Any, target_eligible: Any,
              clinical_low: Any) -> dict[str, Any]:
    """Return a closed release object from fixed four-arm local predictions.

    ``support`` is the prediction/exposure support.  Analysis scoring is its
    intersection with ``target_eligible``; the full positive-weight design is
    nevertheless always passed unchanged into Taylor variance calculations.
    """
    try:
        target_array = _numeric(target)
        rows = target_array.shape[0]
        support_mask = _mask(support, rows)
        weight = _numeric(design_weights, rows)
        year = _numeric(years, rows)
        psu = _numeric(psu_groups, rows)
        stratum = _numeric(stratum_groups, rows)
        fold = _numeric(folds, rows)
        eligible = _mask(target_eligible, rows)
        low = _mask(clinical_low, rows)
        _require(bool(np.all(~low | eligible)))
        _design_contract(weight, year, psu, stratum, fold)
        prediction_map = _prediction_map(predictions, support_mask)
        scoring = support_mask & eligible
        _require(bool(np.isfinite(target_array[scoring]).all()) and
                 all(bool(np.isfinite(prediction_map[arm][scoring]).all()) for arm in _ARMS))

        all_people = np.ones(rows, dtype=bool)
        overall = (_released(target_array, prediction_map, weight, stratum.astype(np.int64), psu.astype(np.int64),
                             scoring, eligible, all_people, primary=True)
                   if _release_allowed(scoring, fold, overall=True) else _suppressed())
        if overall["status"] == "suppressed":
            result = {
                "schema": _SCHEMA, "fixed_prediction_design_intervals": True,
                "model_selection_uncertainty_included": False, "overall": overall,
                "year_2022": _suppressed(), "year_2023": _suppressed(), "low_hb": _suppressed(),
            }
            validate_result(result)
            return result

        year_domains = {year_value: scoring & (year == year_value) for year_value in (2022, 2023)}
        year_ok = all(_release_allowed(year_domains[year_value], fold, overall=False) for year_value in (2022, 2023))
        year_results = {
            year_value: (_released(target_array, prediction_map, weight, stratum.astype(np.int64), psu.astype(np.int64),
                                    year_domains[year_value], eligible, year == year_value, primary=False)
                         if year_ok else _suppressed())
            for year_value in (2022, 2023)
        }
        low_domain = scoring & low
        low_complement = scoring & ~low
        low_ok = (_release_allowed(low_domain, fold, overall=False)
                  and (int(np.count_nonzero(low_complement)) == 0 or int(np.count_nonzero(low_complement)) >= 20))
        low_result = (_released(target_array, prediction_map, weight, stratum.astype(np.int64), psu.astype(np.int64),
                                low_domain, eligible, low, primary=False)
                      if low_ok else _suppressed())
        result = {
            "schema": _SCHEMA, "fixed_prediction_design_intervals": True,
            "model_selection_uncertainty_included": False, "overall": overall,
            "year_2022": year_results[2022], "year_2023": year_results[2023], "low_hb": low_result,
        }
        validate_result(result)
        return result
    except (TypeError, ValueError, KeyError, IndexError, AttributeError, FloatingPointError):
        raise ValueError(_ERROR) from None


evaluate = summarize


def _finite_number(value: Any) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value))
    return float(value)


def _ci(value: Any, estimate: float) -> None:
    _require(isinstance(value, list) and len(value) == 2)
    lower, upper = (_finite_number(item) for item in value)
    _require(lower <= estimate <= upper)


def _validate_arm(value: Any) -> None:
    _require(isinstance(value, Mapping) and set(value) == {"mae", "signed_bias", "rmse", "median_absolute_error"})
    mae = value["mae"]
    _require(isinstance(mae, Mapping) and set(mae) == {"estimate", "ci95"})
    mae_estimate = _finite_number(mae["estimate"]); _ci(mae["ci95"], mae_estimate)
    bias = _finite_number(value["signed_bias"])
    rmse = _finite_number(value["rmse"])
    median = _finite_number(value["median_absolute_error"])
    _require(mae_estimate >= 0 and rmse >= mae_estimate - 1e-8 and median >= 0 and abs(bias) <= mae_estimate + 1e-8)


def _validate_coverage(value: Any) -> None:
    _require(isinstance(value, Mapping) and "status" in value)
    if value["status"] == "suppressed":
        _require(set(value) == {"status"})
    else:
        _require(value["status"] == "released" and set(value) == {"status", "supported_count_lower_bound", "nonsupported_count", "weighted_proportion"}
                 and value["supported_count_lower_bound"] == _LOWER_BOUND
                 and value["nonsupported_count"] == "zero_or_at_least_20")
        proportion = _finite_number(value["weighted_proportion"])
        _require(0.0 <= proportion <= 1.0)


def _validate_group(value: Any, *, primary: bool) -> None:
    _require(isinstance(value, Mapping) and "status" in value)
    if value["status"] == "suppressed":
        _require(set(value) == {"status"})
        return
    expected = {"status", "support_count_lower_bound", "exposure_coverage", "arms"}
    if primary:
        expected.add("paired_state_minus_raw_mae")
    _require(value["status"] == "released" and set(value) == expected and value["support_count_lower_bound"] == _LOWER_BOUND)
    _validate_coverage(value["exposure_coverage"])
    arms = value["arms"]
    _require(isinstance(arms, Mapping) and set(arms) == set(_ARMS))
    for arm in _ARMS:
        _validate_arm(arms[arm])
    if primary:
        paired = value["paired_state_minus_raw_mae"]
        _require(isinstance(paired, Mapping) and set(paired) == {"estimate", "ci95"})
        delta = _finite_number(paired["estimate"]); _ci(paired["ci95"], delta)
        expected_delta = (float(arms["state_extratrees"]["mae"]["estimate"])
                          - float(arms["raw_extratrees"]["mae"]["estimate"]))
        _require(abs(delta - expected_delta) <= 1e-8)


def validate_result(result: Any) -> None:
    """Reject malformed, non-finite, row-bearing, or arithmetically inconsistent releases."""
    try:
        _require(isinstance(result, Mapping) and set(result) == _TOP_KEYS
                 and result["schema"] == _SCHEMA
                 and result["fixed_prediction_design_intervals"] is True
                 and result["model_selection_uncertainty_included"] is False)
        _validate_group(result["overall"], primary=True)
        for name in ("year_2022", "year_2023", "low_hb"):
            _validate_group(result[name], primary=False)
        if result["overall"]["status"] == "suppressed":
            _require(all(result[name]["status"] == "suppressed" for name in ("year_2022", "year_2023", "low_hb")))
        year_status = (result["year_2022"]["status"], result["year_2023"]["status"])
        _require(year_status[0] == year_status[1])
    except (TypeError, ValueError, KeyError, IndexError, AttributeError):
        raise ValueError(_ERROR) from None
