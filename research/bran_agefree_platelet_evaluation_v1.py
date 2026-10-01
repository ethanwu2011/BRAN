"""Closed, array-only platelet-abnormality summaries for age-free BRAN.

This evaluator consumes caller-owned out-of-fold arrays and already-saved
calibration radii.  It performs neither model fitting nor calibration fitting,
and it does not access files.  Its platelet strata are descriptive error
strata, not diagnoses or a substitute for a blood draw.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

import numpy as np

from bran_clinical_semantics_v1 import CBC_FIELDS, CANONICAL_UNITS
import bran_missingness_stress_metrics_v1 as stress
import bran_native_cbc_calibration_metrics_v1 as native
import bran_supervised_mask_uncertainty_v1 as uncertainty


ERROR = "bran_agefree_platelet_evaluation_v1_failed"
PATTERNS = uncertainty.PATTERNS
VERSIONS = uncertainty.VERSIONS
REFERENCES = uncertainty.REFERENCES
PLATELET_FIELD = "plt"
PLATELET_INDEX = CBC_FIELDS.index(PLATELET_FIELD)
UNIT = CANONICAL_UNITS[PLATELET_FIELD]
STRATA = ("low", "intermediate", "high")
NHLBI_SOURCE = "https://www.nhlbi.nih.gov/health/platelet-disorders/diagnosis"
_MINIMUM = 20


def _fail() -> NoReturn:
    raise ValueError(ERROR) from None


def _matrix(value: Any, rows: int | None = None) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1] != len(CBC_FIELDS):
        _fail()
    if rows is not None and value.shape[0] != rows:
        _fail()
    if value.dtype.kind not in "fiu" or value.dtype.kind == "b":
        _fail()
    try:
        result = value.astype(np.float64, copy=False)
        if bool(np.isinf(result).any()):
            _fail()
    except Exception:
        _fail()
    return result


def _bool_matrix(value: Any, rows: int) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.dtype != np.dtype(bool)
            or value.shape != (rows, len(CBC_FIELDS))):
        _fail()
    return value


def _folds(value: Any, rows: int) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.dtype.kind not in "iu" or value.dtype.kind == "b"
            or value.shape != (rows,)):
        _fail()
    try:
        result = value.astype(np.int64, copy=False)
        if bool(np.any((result < 0) | (result > 4))) or not np.array_equal(np.unique(result), np.arange(5)):
            _fail()
    except Exception:
        _fail()
    return result


def _roles(value: Any, rows: int) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.dtype.kind not in "iu" or value.dtype.kind == "b"
            or value.shape != (rows,)):
        _fail()
    try:
        result = value.astype(np.int64, copy=False)
        if bool(np.any((result != 0) & (result != 1))) or not bool(np.any(result == 1)):
            _fail()
    except Exception:
        _fail()
    return result


def _adult(value: Any, rows: int) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(bool) or value.shape != (rows,):
        _fail()
    return value


def _counts(value: Any, scoring_folds: np.ndarray) -> np.ndarray:
    rows = scoring_folds.shape[0]
    if (not isinstance(value, np.ndarray) or value.dtype.kind not in "iu" or value.dtype.kind == "b"
            or value.shape != (1000, rows)):
        _fail()
    try:
        result = value.astype(np.int64, copy=False)
        if bool(np.any(result < 0)):
            _fail()
        for fold in range(5):
            local = scoring_folds == fold
            if not bool(np.all(result[:, local].sum(axis=1) == int(np.count_nonzero(local)))):
                _fail()
    except Exception:
        _fail()
    return result


def _observed(value: Any, rows: int) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or frozenset(value) != frozenset(PATTERNS):
        _fail()
    return {pattern: _bool_matrix(value[pattern], rows) for pattern in PATTERNS}


def _predictions(value: Any, rows: int, observed: Mapping[str, np.ndarray], target: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    if not isinstance(value, Mapping) or frozenset(value) != frozenset(PATTERNS):
        _fail()
    result: dict[str, dict[str, np.ndarray]] = {}
    for pattern in PATTERNS:
        item = value[pattern]
        if not isinstance(item, Mapping) or frozenset(item) != frozenset(VERSIONS):
            _fail()
        parsed = {version: _matrix(item[version], rows) for version in VERSIONS}
        try:
            if bool(np.any(observed[pattern] & ~np.isfinite(target))):
                _fail()
            for prediction in parsed.values():
                if bool(np.any(observed[pattern] & ~np.isfinite(prediction))):
                    _fail()
        except Exception:
            _fail()
        result[pattern] = parsed
    return result


def _public_cell(cell: dict[str, Any]) -> dict[str, Any]:
    return uncertainty._public_cell(cell)


def _strata(target: np.ndarray, eligible: np.ndarray) -> dict[str, np.ndarray]:
    platelet = target[:, PLATELET_INDEX]
    return {
        "low": eligible & (platelet < 150.0),
        "intermediate": eligible & (platelet >= 150.0) & (platelet <= 450.0),
        "high": eligible & (platelet > 450.0),
    }


def _comparison(
    target: np.ndarray,
    prediction: Mapping[str, np.ndarray],
    radii: Mapping[str, np.ndarray],
    eligible: np.ndarray,
    counts: np.ndarray,
) -> dict[str, Any]:
    overall = np.flatnonzero(eligible)
    groups = _strata(target, eligible)

    def stratum_cell(mask: np.ndarray) -> dict[str, Any]:
        count = int(np.count_nonzero(mask))
        complement = int(np.count_nonzero(eligible) - count)
        # Releasing a stratum when its inverse has 1--19 members would leak a
        # small complementary adult subgroup.  This is separate from interval
        # hit/miss suppression inside the established native summarizer.
        if count < _MINIMUM or 1 <= complement <= _MINIMUM - 1:
            return {"status": "unsupported"}
        return _public_cell(native._summarize_group(
            target, {"native": prediction["reference"], "generative": prediction["candidate"]},
            {"native": radii["reference"], "generative": radii["candidate"]},
            np.flatnonzero(mask), PLATELET_INDEX, counts))

    return {
        "overall": _public_cell(native._summarize_group(
            target, {"native": prediction["reference"], "generative": prediction["candidate"]},
            {"native": radii["reference"], "generative": radii["candidate"]},
            overall, PLATELET_INDEX, counts)),
        "strata": {
            name: stratum_cell(mask)
            for name, mask in groups.items()
        },
    }


def _support(mask: np.ndarray) -> dict[str, object]:
    return stress.safe_coverage(mask)


def evaluate(
    target_N9: Any,
    observed_by_pattern_N9: Any,
    predictions_by_pattern_by_version_N9: Any,
    folds_N: Any,
    roles_N: Any,
    counts_1000xNscore: Any,
    radii: Any,
    *,
    adult_mask_N: Any,
) -> dict[str, Any]:
    """Return a closed descriptive platelet-only aggregate from fixed inputs.

    ``observed_by_pattern_N9`` must already encode an observed, held-out target
    and a usable prediction.  Calibration radii are caller-supplied fixed
    artifacts; this function does not fit or alter them.
    """
    try:
        target = _matrix(target_N9)
        rows = target.shape[0]
        if rows < 1:
            _fail()
        observed = _observed(observed_by_pattern_N9, rows)
        predictions = _predictions(predictions_by_pattern_by_version_N9, rows, observed, target)
        folds = _folds(folds_N, rows)
        roles = _roles(roles_N, rows)
        uncertainty._split_roles(folds, roles)
        adult = _adult(adult_mask_N, rows)
        uncertainty.validate_radii(radii)
        scoring = roles == 1
        score_indices = np.flatnonzero(scoring)
        adult_score_indices = np.flatnonzero(scoring & adult)
        counts = _counts(counts_1000xNscore, folds[score_indices])
        result_patterns: dict[str, Any] = {}
        for pattern_index, pattern in enumerate(PATTERNS):
            prediction_mask = observed[pattern] & scoring[:, None] & adult[:, None]
            expanded = {
                version: np.broadcast_to(radii[pattern_index, folds, version_index], target.shape)
                for version_index, version in enumerate(VERSIONS)
            }
            calibrated_mask = prediction_mask & np.logical_and.reduce(
                tuple(np.isfinite(expanded[version]) for version in VERSIONS)
            )
            comparisons: dict[str, Any] = {}
            for reference in REFERENCES:
                comparisons[reference] = _comparison(
                    target[score_indices],
                    {"reference": predictions[pattern][reference][score_indices],
                     "candidate": predictions[pattern]["student"][score_indices]},
                    {"reference": expanded[reference][score_indices],
                     "candidate": expanded["student"][score_indices]},
                    calibrated_mask[score_indices, PLATELET_INDEX], counts,
                )
            result_patterns[pattern] = {
                "prediction_support": _support(prediction_mask[adult_score_indices, PLATELET_INDEX]),
                "calibrated_support": _support(calibrated_mask[adult_score_indices, PLATELET_INDEX]),
                "comparisons": comparisons,
            }
        result = {
            "schema": "bran-agefree-platelet-abnormality-evaluation-v1",
            "platelet": {"field": PLATELET_FIELD, "unit": UNIT,
                         "strata": {"low": "<150", "intermediate": "150_to_450_inclusive", "high": ">450"},
                         "threshold_source": NHLBI_SOURCE},
            "scoring_population": "adult_role_1_only",
            "patterns": result_patterns,
            "clinical_diagnosis_claimed": False,
            "blood_draw_replacement_claimed": False,
            "automatic_promotion": False,
            "classification_metrics_emitted": False,
            "patient_level_output_emitted": False,
        }
        validate_result(result)
        return result
    except Exception:
        _fail()


def _valid_support(value: Any) -> None:
    if not isinstance(value, dict) or type(value.get("status")) is not str:
        _fail()
    if value["status"] == "withheld":
        if frozenset(value) != {"status"}:
            _fail()
        return
    if value["status"] != "released" or frozenset(value) != {"status", "supported", "total"}:
        _fail()
    if type(value["supported"]) is not int or type(value["total"]) is not int:
        _fail()
    supported, total = value["supported"], value["total"]
    if total < _MINIMUM or not 0 <= supported <= total:
        _fail()
    complement = total - supported
    if not (supported == 0 or supported >= _MINIMUM) or not (complement == 0 or complement >= _MINIMUM):
        _fail()


def _native_cell(value: Any) -> dict[str, Any]:
    if value == {"status": "unsupported"}:
        return value
    if (not isinstance(value, dict) or frozenset(value) != {"status", "arms", "contrast"}
            or value.get("status") != "supported"):
        _fail()
    arms, contrast = value["arms"], value["contrast"]
    if not isinstance(arms, dict) or frozenset(arms) != {"reference", "candidate"}:
        _fail()
    if not isinstance(contrast, dict) or frozenset(contrast) != {"candidate_minus_reference"}:
        _fail()
    return {
        "status": "supported",
        "arms": {"native": arms["reference"], "generative": arms["candidate"]},
        "contrast": {"generative_minus_native": contrast["candidate_minus_reference"]},
    }


def validate_result(result: Any) -> bool:
    """Validate the closed, aggregate-only public platelet schema."""
    try:
        required = {
            "schema", "platelet", "scoring_population", "patterns", "clinical_diagnosis_claimed",
            "blood_draw_replacement_claimed", "automatic_promotion", "classification_metrics_emitted",
            "patient_level_output_emitted",
        }
        if not isinstance(result, dict) or frozenset(result) != required:
            _fail()
        if result["schema"] != "bran-agefree-platelet-abnormality-evaluation-v1":
            _fail()
        if result["scoring_population"] != "adult_role_1_only":
            _fail()
        if result["platelet"] != {
            "field": PLATELET_FIELD, "unit": UNIT,
            "strata": {"low": "<150", "intermediate": "150_to_450_inclusive", "high": ">450"},
            "threshold_source": NHLBI_SOURCE,
        }:
            _fail()
        if any(result[key] is not False for key in (
            "clinical_diagnosis_claimed", "blood_draw_replacement_claimed", "automatic_promotion",
            "classification_metrics_emitted", "patient_level_output_emitted",
        )):
            _fail()
        patterns = result["patterns"]
        if not isinstance(patterns, dict) or frozenset(patterns) != frozenset(PATTERNS):
            _fail()
        for pattern in PATTERNS:
            report = patterns[pattern]
            if not isinstance(report, dict) or frozenset(report) != {
                "prediction_support", "calibrated_support", "comparisons"}:
                _fail()
            _valid_support(report["prediction_support"])
            _valid_support(report["calibrated_support"])
            left, right = report["prediction_support"], report["calibrated_support"]
            if left["status"] == right["status"] == "released":
                if left["total"] != right["total"] or right["supported"] > left["supported"]:
                    _fail()
            comparisons = report["comparisons"]
            if not isinstance(comparisons, dict) or frozenset(comparisons) != frozenset(REFERENCES):
                _fail()
            reference_cells: dict[str, list[dict[str, Any]]] = {reference: [] for reference in REFERENCES}
            for reference in REFERENCES:
                comparison = comparisons[reference]
                if not isinstance(comparison, dict) or frozenset(comparison) != {"overall", "strata"}:
                    _fail()
                native._validate_cell(_native_cell(comparison["overall"]))
                reference_cells[reference].append(comparison["overall"])
                strata = comparison["strata"]
                if not isinstance(strata, dict) or frozenset(strata) != frozenset(STRATA):
                    _fail()
                for stratum in STRATA:
                    native._validate_cell(_native_cell(strata[stratum]))
                    reference_cells[reference].append(strata[stratum])
            for left, right in zip(reference_cells["initial"], reference_cells["continued"]):
                if left["status"] != right["status"]:
                    _fail()
                if left["status"] == "supported" and left["arms"]["candidate"] != right["arms"]["candidate"]:
                    _fail()
        return True
    except Exception:
        _fail()
