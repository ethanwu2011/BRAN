"""Closed, aggregate-only native R7 versus V5 HiRID comparison.

H5 is deliberately a small comparison layer around the frozen H2 evaluator.
It accepts only already validated local selector results and the two private
native-prediction containers.  It performs no fitting, selection, calibration,
inference, source access, serialization, or patient-level output.  The two
H2 reports are returned unchanged; the only new statistic is a paired
R7-minus-V5 native-MAE contrast on the same admissions and the same full
denominator bootstrap draws.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any
from types import MappingProxyType

import numpy as np

import bran_hirid_hb_evaluation_h2 as h2
from bran_hirid_v5_inference_v1 import PrivateHiRIDV5Hb
from bran_hirid_v5_observation_kernel import EpisodeSelection


ERROR = "hirid_r7_h5_comparison_contract_failed"
SCHEMA = "bran-hirid-r7-v5-h5-comparison-v1"
BOOTSTRAP_DRAWS = h2.BOOTSTRAP_DRAWS
BOOTSTRAP_SEED = h2.BOOTSTRAP_SEED
MIN_VALID_DRAWS = h2.MIN_VALID_DRAWS
COUNT_STEP = h2.COUNT_STEP
OVERALL_SCORE_FLOOR = h2.OVERALL_SCORE_FLOOR
LOW_SCORE_FLOOR = h2.LOW_SCORE_FLOOR
PAIR_METRIC = "r7_minus_v5_native_mae"

FLAGS = MappingProxyType({
    # Direct, fixed-readout comparison claims.
    "direct_native_comparison": True,
    "same_h3_admissions": True,
    "same_full_denominator_bootstrap": True,
    "h2_reports_unchanged": True,
    "reused_external_evaluation": True,
    "untouched_external_validation": False,
    "paired_ci_not_subtracted": True,
    # Explicitly negative model-development/privacy claims.
    "source_fitting": False,
    "fit_performed": False,
    "selection_performed": False,
    "calibration_performed": False,
    "patient_level_output_emitted": False,
    "predictions_serialized": False,
    "repeat_person_independence_established": False,
    "model_promotion_claim": False,
})

_BOOTSTRAP_KEYS = frozenset(("draws", "seed", "minimum_valid_draws", "full_denominator"))
_TOP_KEYS = frozenset(("schema", "flags", "bootstrap", "R7", "V5", "paired_r7_minus_v5"))
_PAIR_KEYS = frozenset(("overall", "low_hb"))


class _Invalid(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _Invalid


def _finite(value: object) -> float:
    _require(isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise _Invalid from None
    _require(math.isfinite(result))
    return result


def _bootstrap_draws(rows: int):
    """Yield the fixed H2 full-denominator admission draws exactly."""
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


def _pair_group(domain: np.ndarray, truth: np.ndarray, r7: np.ndarray, v5: np.ndarray,
                *, allowed: bool, h2_released: bool) -> dict[str, Any]:
    """Return one paired MAE group while retaining no draw-level values."""
    if not allowed or not h2_released:
        return {"status": "suppressed", "metrics": None}
    _require(type(domain) is np.ndarray and domain.dtype == np.bool_)
    _require(domain.shape == truth.shape == r7.shape == v5.shape)
    rows = int(truth.size)
    point_domain = truth[domain]
    _require(point_domain.size > 0)
    point = float(np.mean(np.abs(r7[domain] - point_domain)) -
                  np.mean(np.abs(v5[domain] - point_domain)))
    _require(math.isfinite(point))

    draws = np.full(BOOTSTRAP_DRAWS, np.nan, dtype=np.float64)
    count = 0
    for count, index in enumerate(_bootstrap_draws(rows), start=1):
        _require(type(index) is np.ndarray and index.shape == (rows,)
                 and index.dtype.kind in "iu"
                 and (rows == 0 or bool(((index >= 0) & (index < rows)).all())))
        retained = index[domain[index]]
        if retained.size:
            truth_draw = truth[retained]
            draws[count - 1] = float(np.mean(np.abs(r7[retained] - truth_draw)) -
                                     np.mean(np.abs(v5[retained] - truth_draw)))
            _require(math.isfinite(float(draws[count - 1])))
    _require(count == BOOTSTRAP_DRAWS)
    interval = _interval(draws)
    if interval is None:
        return {"status": "suppressed", "metrics": None}
    return {"status": "released", "metrics": {
        PAIR_METRIC: {"estimate": point, "ci95": interval},
    }}


def _validate_pair_metric(value: Any) -> None:
    _require(isinstance(value, Mapping) and set(value) == {"estimate", "ci95"})
    _finite(value["estimate"])
    _require(type(value["ci95"]) is list and len(value["ci95"]) == 2)
    low, high = (_finite(item) for item in value["ci95"])
    _require(low <= high)


def _validate_pair_group(value: Any, r7_group: Mapping[str, Any], v5_group: Mapping[str, Any]) -> None:
    _require(isinstance(value, Mapping) and set(value) == {"status", "metrics"})
    expected_release = r7_group["status"] == "released" and v5_group["status"] == "released"
    if not expected_release:
        _require(value["status"] == "suppressed" and value["metrics"] is None)
        return
    _require(value["status"] == "released" and isinstance(value["metrics"], Mapping)
             and set(value["metrics"]) == {PAIR_METRIC})
    _validate_pair_metric(value["metrics"][PAIR_METRIC])
    expected = (r7_group["metrics"]["native_mae"]["estimate"] -
                v5_group["metrics"]["native_mae"]["estimate"])
    _require(abs(value["metrics"][PAIR_METRIC]["estimate"] - expected) <= 1e-10)


def _validate_h2(report: Any) -> None:
    _require(isinstance(report, Mapping))
    h2.validate_result(report)


def summarize(selections: Iterable[EpisodeSelection], r7_predictions: PrivateHiRIDV5Hb,
              v5_predictions: PrivateHiRIDV5Hb, training_median_g_dl: float) -> dict[str, Any]:
    """Return the closed H5 native R7-minus-V5 comparison aggregate.

    ``selections`` is materialized once so both fixed H2 reports and the
    paired contrast see the exact same ordered H3 admission denominator.
    ``r7_predictions`` and ``v5_predictions`` are private, non-serializable
    local containers; only scalar H2 reports and a scalar paired contrast are
    released.
    """
    try:
        selected = tuple(selections)
        # H2 is the authoritative selector/prediction validation and report
        # implementation.  Keep these nested objects byte-for-byte structurally
        # unchanged rather than reimplementing or rewriting their fields.
        r7_h2 = h2.summarize(selected, r7_predictions, training_median_g_dl)
        v5_h2 = h2.summarize(selected, v5_predictions, training_median_g_dl)
        statuses, truth = h2._selection_statuses(selected)
        r7 = h2._validated_predictions(r7_predictions, statuses)
        v5 = h2._validated_predictions(v5_predictions, statuses)
        scoring = statuses == "ready"
        low = scoring & (truth < 12.0)
        _require(r7.shape == v5.shape == truth.shape == scoring.shape)

        overall_domain, overall_allowed = h2._stratum_allowed(
            scoring, np.ones(scoring.size, dtype=bool), minimum=OVERALL_SCORE_FLOOR,
            require_complement=False,
        )
        low_domain, low_allowed = h2._stratum_allowed(
            scoring, low, minimum=LOW_SCORE_FLOOR, require_complement=True,
        )
        paired = {
            "overall": _pair_group(
                overall_domain, truth, r7, v5, allowed=overall_allowed,
                h2_released=(r7_h2["overall"]["status"] == "released" and
                             v5_h2["overall"]["status"] == "released"),
            ),
            "low_hb": _pair_group(
                low_domain, truth, r7, v5, allowed=low_allowed,
                h2_released=(r7_h2["low_hb"]["status"] == "released" and
                             v5_h2["low_hb"]["status"] == "released"),
            ),
        }
        result = {
            "schema": SCHEMA,
            "flags": dict(FLAGS),
            "bootstrap": {
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "minimum_valid_draws": MIN_VALID_DRAWS,
                "full_denominator": True,
            },
            "R7": r7_h2,
            "V5": v5_h2,
            "paired_r7_minus_v5": paired,
        }
        validate_result(result)
        return result
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


evaluate = summarize


def validate_result(result: Any) -> None:
    """Validate the closed H5 schema, nested unchanged H2 reports, and pair."""
    try:
        _require(isinstance(result, Mapping) and set(result) == _TOP_KEYS and result["schema"] == SCHEMA)
        flags = result["flags"]
        _require(isinstance(flags, Mapping) and set(flags) == set(FLAGS))
        for key, expected in FLAGS.items():
            _require(flags[key] is expected)
        bootstrap = result["bootstrap"]
        _require(isinstance(bootstrap, Mapping) and set(bootstrap) == _BOOTSTRAP_KEYS)
        _require(type(bootstrap["draws"]) is int and bootstrap["draws"] == BOOTSTRAP_DRAWS)
        _require(type(bootstrap["seed"]) is int and bootstrap["seed"] == BOOTSTRAP_SEED)
        _require(type(bootstrap["minimum_valid_draws"]) is int
                 and bootstrap["minimum_valid_draws"] == MIN_VALID_DRAWS)
        _require(bootstrap["full_denominator"] is True)

        r7_h2 = result["R7"]
        v5_h2 = result["V5"]
        _validate_h2(r7_h2)
        _validate_h2(v5_h2)
        _require(r7_h2["schema"] == h2.SCHEMA and v5_h2["schema"] == h2.SCHEMA)
        # The two arms are evaluated on one exact H3 denominator.  A report
        # with independently edited counts/coverage is not a paired result,
        # even if each edited H2 object remains internally well-formed.
        _require(r7_h2["denominator_counts"] == v5_h2["denominator_counts"])
        _require(r7_h2["prediction_coverage"] == v5_h2["prediction_coverage"])
        for name in ("overall", "low_hb"):
            r7_group = r7_h2[name]
            v5_group = v5_h2[name]
            _require(r7_group["status"] == v5_group["status"])
            if r7_group["status"] == "released":
                _require(r7_group["metrics"]["training_median_mae"] ==
                         v5_group["metrics"]["training_median_mae"])
        paired = result["paired_r7_minus_v5"]
        _require(isinstance(paired, Mapping) and set(paired) == _PAIR_KEYS)
        _validate_pair_group(paired["overall"], r7_h2["overall"], v5_h2["overall"])
        _validate_pair_group(paired["low_hb"], r7_h2["low_hb"], v5_h2["low_hb"])
    except (TypeError, ValueError, KeyError, IndexError, AttributeError, _Invalid):
        raise ValueError(ERROR) from None


validate = validate_result


__all__ = [
    "ERROR", "SCHEMA", "FLAGS", "PAIR_METRIC", "BOOTSTRAP_DRAWS", "BOOTSTRAP_SEED",
    "MIN_VALID_DRAWS", "PrivateHiRIDV5Hb", "summarize", "evaluate", "validate", "validate_result",
]
