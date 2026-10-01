"""Closed, aggregate-only HiRID H2 Hb evaluation for fixed local predictions.

This is deliberately a local-array kernel.  It neither opens a source nor
serializes, fits, calibrates, or emits patient-level material.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

import numpy as np

from bran_hirid_v5_inference_v1 import PrivateHiRIDV5Hb
from bran_hirid_v5_input_adapter_v1 import _validated_selection
from bran_hirid_v5_observation_kernel import EpisodeSelection


ERROR = "hirid_h2_evaluation_contract_failed"
SCHEMA = "bran-hirid-h2-evaluation-v1"
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 92331
MIN_VALID_DRAWS = 900
COUNT_STEP = 20
OVERALL_SCORE_FLOOR = 100
LOW_SCORE_FLOOR = 20
_STATUSES = ("ready", "abstain_no_target", "abstain_no_physiology", "target_conflict")
_TOP_KEYS = frozenset((
    "schema", "direct_transport", "adaptation", "source_fitting", "fixed_fit_intervals",
    "repeat_person_independence_established", "calibrated_measurement_intervals",
    "patient_level_output_emitted", "denominator_counts", "prediction_coverage", "overall", "low_hb",
))


class _Invalid(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _Invalid


def _finite_positive(value: object) -> float:
    _require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise _Invalid from None
    _require(math.isfinite(result) and result > 0.0)
    return result


def _nan(value: object) -> bool:
    try:
        return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isnan(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _selection_statuses(selections: Iterable[EpisodeSelection]) -> tuple[np.ndarray, np.ndarray]:
    """Validate selector contracts without retaining their private payloads."""
    statuses: list[str] = []
    truths: list[float] = []
    for selection in selections:
        # Reuse the adapter's exact-type, status, and physiology-map boundary.
        status, _ignored_labs = _validated_selection(selection)
        _require(type(selection) is EpisodeSelection)
        if status in ("ready", "abstain_no_physiology"):
            truths.append(_finite_positive(selection.target_hb))
        else:
            _require(_nan(selection.target_hb))
            truths.append(float("nan"))
        statuses.append(status)
    return np.asarray(statuses, dtype="<U24"), np.asarray(truths, dtype=np.float64)


def _validated_predictions(predictions: object, statuses: np.ndarray) -> np.ndarray:
    _require(type(predictions) is PrivateHiRIDV5Hb)
    n_rows = statuses.shape[0]
    native = predictions.native_hemoglobin
    available = predictions.available
    _require(type(native) is np.ndarray and native.shape == (n_rows,) and native.dtype.kind == "f")
    _require(type(available) is np.ndarray and available.shape == (n_rows,) and available.dtype == np.bool_)
    ready = statuses == "ready"
    _require(np.array_equal(available, ready))
    _require(bool(np.isfinite(native[ready]).all()))
    _require(bool(np.isnan(native[~ready]).all()))
    return np.asarray(native, dtype=np.float64)


def _bootstrap_draws(rows: int):
    """Yield deterministic full-denominator draws without retaining them."""
    _require(type(rows) is int and rows >= 0)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    for _draw_number in range(BOOTSTRAP_DRAWS):
        if rows == 0:
            yield np.empty(0, dtype=np.intp)
        else:
            yield generator.integers(0, rows, size=rows, dtype=np.intp)


def _interval(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    if finite.size < MIN_VALID_DRAWS:
        return None
    low, high = np.percentile(finite, (2.5, 97.5))
    _require(math.isfinite(float(low)) and math.isfinite(float(high)))
    return [float(low), float(high)]


def _point_metrics(truth: np.ndarray, native: np.ndarray, median: float) -> dict[str, float]:
    error = native - truth
    result = {
        "native_mae": float(np.mean(np.abs(error))),
        "native_signed_bias": float(np.mean(error)),
        "native_rmse": float(np.sqrt(np.mean(error ** 2))),
        "training_median_mae": float(np.mean(np.abs(median - truth))),
    }
    result["paired_native_minus_median_mae"] = result["native_mae"] - result["training_median_mae"]
    _require(all(math.isfinite(value) for value in result.values()))
    return result


def _stratum_allowed(scoring: np.ndarray, stratum: np.ndarray, *, minimum: int,
                     require_complement: bool) -> tuple[np.ndarray, bool]:
    domain = scoring & stratum
    complement = scoring & ~stratum
    supported = int(np.count_nonzero(domain))
    complement_count = int(np.count_nonzero(complement))
    allowed = supported >= minimum and (not require_complement or complement_count == 0 or complement_count >= LOW_SCORE_FLOOR)
    return domain, allowed


def _released_metrics(domain: np.ndarray, truth: np.ndarray, native: np.ndarray, median: float,
                      boot: dict[str, np.ndarray] | None) -> dict[str, Any]:
    if boot is None:
        return {"status": "suppressed", "metrics": None}
    point = _point_metrics(truth[domain], native[domain], median)
    intervals = {key: _interval(value) for key, value in boot.items()}
    if any(value is None for value in intervals.values()):
        return {"status": "suppressed", "metrics": None}
    return {"status": "released", "metrics": {key: {"estimate": point[key], "ci95": intervals[key]} for key in point}}


def _stream_metrics(scoring: np.ndarray, low: np.ndarray, truth: np.ndarray,
                    native: np.ndarray, median: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stream each shared draw once and retain only scalar metric summaries."""
    overall_domain, overall_allowed = _stratum_allowed(
        scoring, np.ones(scoring.size, dtype=bool), minimum=OVERALL_SCORE_FLOOR, require_complement=False,
    )
    low_domain, low_allowed = _stratum_allowed(
        scoring, low, minimum=LOW_SCORE_FLOOR, require_complement=True,
    )
    keys = tuple(_point_metrics(np.array([1.0]), np.array([1.0]), median))
    overall_boot = ({key: np.full(BOOTSTRAP_DRAWS, np.nan, dtype=np.float64) for key in keys}
                    if overall_allowed else None)
    low_boot = ({key: np.full(BOOTSTRAP_DRAWS, np.nan, dtype=np.float64) for key in keys}
                if low_allowed else None)
    rows = scoring.size
    draw_count = 0
    for draw_count, index in enumerate(_bootstrap_draws(rows), start=1):
        _require(type(index) is np.ndarray and index.shape == (rows,) and index.dtype.kind in "iu"
                 and (rows == 0 or bool(((index >= 0) & (index < rows)).all())))
        # Both strata retain membership from this exact full-denominator draw.
        for domain, boot in ((overall_domain, overall_boot), (low_domain, low_boot)):
            if boot is None:
                continue
            retained = index[domain[index]]
            if retained.size:
                values = _point_metrics(truth[retained], native[retained], median)
                for key, value in values.items():
                    boot[key][draw_count - 1] = value
    _require(draw_count == BOOTSTRAP_DRAWS)
    return (_released_metrics(overall_domain, truth, native, median, overall_boot),
            _released_metrics(low_domain, truth, native, median, low_boot))


def _denominators(statuses: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    counts = {status: int(np.count_nonzero(statuses == status)) for status in _STATUSES}
    safe = statuses.size >= COUNT_STEP and all(value == 0 or value >= COUNT_STEP for value in counts.values())
    if not safe:
        return {"status": "suppressed"}, {"status": "suppressed"}
    rounded = {status: (value // COUNT_STEP) * COUNT_STEP for status, value in counts.items()}
    breakdown = {"status": "released", "full_cohort_count_lower_bound": (statuses.size // COUNT_STEP) * COUNT_STEP,
                 "categories": rounded}
    overall = counts["ready"] / statuses.size
    eligible_total = counts["ready"] + counts["abstain_no_physiology"]
    eligible = None if eligible_total == 0 else counts["ready"] / eligible_total
    coverage: dict[str, Any] = {"status": "released", "overall": overall}
    if eligible is not None:
        coverage["target_eligible"] = eligible
    else:
        coverage["target_eligible"] = None
    return breakdown, coverage


def summarize(selections: Iterable[EpisodeSelection], predictions: PrivateHiRIDV5Hb,
              training_median_g_dl: float) -> dict[str, Any]:
    """Return the closed H2 prospective aggregate release object.

    The supplied comparator is a frozen AI-READI fold-0 training Hb median;
    it is never estimated from HiRID inputs.
    """
    try:
        statuses, truth = _selection_statuses(selections)
        median = _finite_positive(training_median_g_dl)
        native = _validated_predictions(predictions, statuses)
        scoring = statuses == "ready"
        # Hidden truth in non-ready records is never read beyond contract
        # validation, and can therefore never affect points or intervals.
        low = scoring & (truth < 12.0)
        breakdown, coverage = _denominators(statuses)
        overall, low_hb = _stream_metrics(scoring, low, truth, native, median)
        result = {
            "schema": SCHEMA,
            "direct_transport": True, "adaptation": False, "source_fitting": False,
            "fixed_fit_intervals": True, "repeat_person_independence_established": False,
            "calibrated_measurement_intervals": False, "patient_level_output_emitted": False,
            "denominator_counts": breakdown, "prediction_coverage": coverage,
            "overall": overall, "low_hb": low_hb,
        }
        validate_result(result)
        return result
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


evaluate = summarize


def _finite_number(value: Any) -> float:
    _require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise _Invalid from None
    _require(math.isfinite(result))
    return result


def _validate_metric(value: Any, *, nonnegative: bool = False) -> None:
    _require(isinstance(value, Mapping) and set(value) == {"estimate", "ci95"})
    estimate = _finite_number(value["estimate"])
    _require(isinstance(value["ci95"], list) and len(value["ci95"]) == 2)
    low, high = (_finite_number(item) for item in value["ci95"])
    _require(low <= high)
    if nonnegative:
        _require(estimate >= 0.0 and low >= 0.0)


def _validate_group(value: Any) -> None:
    _require(isinstance(value, Mapping) and set(value) == {"status", "metrics"})
    if value["status"] == "suppressed":
        _require(value["metrics"] is None)
        return
    _require(value["status"] == "released" and isinstance(value["metrics"], Mapping))
    metrics = value["metrics"]
    expected = {"native_mae", "native_signed_bias", "native_rmse", "training_median_mae", "paired_native_minus_median_mae"}
    _require(set(metrics) == expected)
    for name in expected:
        _validate_metric(metrics[name], nonnegative=name in {"native_mae", "native_rmse", "training_median_mae"})
    _require(abs(metrics["paired_native_minus_median_mae"]["estimate"] -
                 (metrics["native_mae"]["estimate"] - metrics["training_median_mae"]["estimate"])) <= 1e-10)
    _require(metrics["native_rmse"]["estimate"] + 1e-10 >= metrics["native_mae"]["estimate"])


def validate_result(result: Any) -> None:
    """Reject extra fields, raw arrays/draws, non-finite, and inconsistent releases."""
    try:
        _require(isinstance(result, Mapping) and set(result) == _TOP_KEYS and result["schema"] == SCHEMA)
        for name, expected in (("direct_transport", True), ("adaptation", False), ("source_fitting", False),
                               ("fixed_fit_intervals", True), ("repeat_person_independence_established", False),
                               ("calibrated_measurement_intervals", False), ("patient_level_output_emitted", False)):
            _require(result[name] is expected)
        counts = result["denominator_counts"]
        _require(isinstance(counts, Mapping) and "status" in counts)
        if counts["status"] == "suppressed":
            _require(set(counts) == {"status"})
            _require(result["prediction_coverage"] == {"status": "suppressed"})
        else:
            _require(set(counts) == {"status", "full_cohort_count_lower_bound", "categories"} and counts["status"] == "released")
            total = counts["full_cohort_count_lower_bound"]
            _require(type(total) is int and total >= COUNT_STEP and total % COUNT_STEP == 0)
            categories = counts["categories"]
            _require(isinstance(categories, Mapping) and set(categories) == set(_STATUSES))
            for value in categories.values():
                _require(type(value) is int and value >= 0 and value % COUNT_STEP == 0)
            summed_lower = sum(categories.values())
            positive_cells = sum(value > 0 for value in categories.values())
            # Released zero categories are exact zeros; every positive cell
            # can exceed its floored lower bound by at most COUNT_STEP - 1.
            upper_total = summed_lower + positive_cells * (COUNT_STEP - 1)
            _require(summed_lower <= total <= (upper_total // COUNT_STEP) * COUNT_STEP)
            coverage = result["prediction_coverage"]
            _require(isinstance(coverage, Mapping) and set(coverage) == {"status", "overall", "target_eligible"}
                     and coverage["status"] == "released")
            overall = _finite_number(coverage["overall"])
            _require(0.0 <= overall <= 1.0)
            ready = categories["ready"]
            other = sum(categories[name] for name in _STATUSES if name != "ready")
            _require(not (ready == 0 and overall != 0.0))
            _require(not (ready > 0 and overall <= 0.0))
            _require(not (other == 0 and overall != 1.0))
            _require(not (other > 0 and overall >= 1.0))
            eligible = ready + categories["abstain_no_physiology"]
            target_eligible = coverage["target_eligible"]
            if eligible == 0:
                _require(target_eligible is None)
            else:
                target_value = _finite_number(target_eligible)
                _require(0.0 <= target_value <= 1.0)
                no_physiology = categories["abstain_no_physiology"]
                _require(not (ready == 0 and target_value != 0.0))
                _require(not (ready > 0 and target_value <= 0.0))
                _require(not (no_physiology == 0 and target_value != 1.0))
                _require(not (no_physiology > 0 and target_value >= 1.0))
        _validate_group(result["overall"]); _validate_group(result["low_hb"])
    except (TypeError, ValueError, KeyError, IndexError, AttributeError, _Invalid):
        raise ValueError(ERROR) from None


__all__ = ["ERROR", "SCHEMA", "PrivateHiRIDV5Hb", "summarize", "evaluate", "validate_result"]
