"""Closed, disclosure-safe wrapper for the unchanged native screening metric kernel.

This module accepts only fold-held-out route probabilities and aggregate inputs.
It performs no inference, source I/O, checkpoint I/O, fitting, or plotting.
"""
from __future__ import annotations

import copy
import math
from typing import Mapping, Sequence

import numpy as np

import bran_multisource_outcome_metrics_v2 as metrics


ERROR = "bran_r7_modality_atlas_a3_contract_failed"
SCHEMA = "bran-r7-modality-atlas-a3-v1"
STATUS = "descriptive_posthoc_reused_development"
ROUTES = ("both", "clinical", "retinal")
CONTRASTS = {
    "combined_minus_clinical": ("both", "clinical"),
    "combined_minus_retinal": ("both", "retinal"),
}
NAMES = {
    "mh_a1c": "Elevated A1C",
    "mhoccur_amd": "Macular degeneration",
    "mhoccur_ca": "Cancer",
    "mhoccur_circ": "Circulation problems",
    "mhoccur_clsh": "High cholesterol",
    "mhoccur_cns": "Other neurological conditions",
    "mhoccur_cogn": "Mild cognitive impairment",
    "mhoccur_crt": "Cataracts",
    "mhoccur_cvdot": "Other heart issues",
    "mhoccur_ded": "Dry eye",
    "mhoccur_ear": "Hearing impairment",
    "mhoccur_fall": "Falls in prior year",
    "mhoccur_gi": "Digestive problems",
    "mhoccur_glc": "Glaucoma",
    "mhoccur_hbp": "High blood pressure",
    "mhoccur_lbp": "Low blood pressure",
    "mhoccur_mi": "Heart attack",
    "mhoccur_oa": "Osteoporosis",
    "mhoccur_obs": "Obesity",
    "mhoccur_plm": "Pulmonary problems",
    "mhoccur_ra": "Arthritis",
    "mhoccur_rnl": "Kidney problems",
    "mhoccur_strk": "Stroke",
    "mhoccur_ua": "Urinary problems",
    "mhterm_dm2": "Type II diabetes",
    "mhterm_predm": "Prediabetes",
}
ENDPOINT_NAMES = tuple(sorted(NAMES))
DISPLAY_ORDER = tuple(sorted(NAMES, key=NAMES.get))

PARAMETERS = {
    "routes": list(ROUTES),
    "contrasts": {key: list(value) for key, value in CONTRASTS.items()},
    "endpoint_count": 26,
    "native_head": "same_frozen_r7_screening_joint_head_192_to_26",
    "probability_link": "sigmoid",
    "folds": 5,
    "metric": "fold_size_weighted_auroc",
    "bootstrap": "matched_participant_within_fold_weighted",
    "bootstrap_draws": 1000,
    "bootstrap_seed": 98571,
    "minimum_valid_draws": 900,
    "minimum_class_support": 20,
    "confidence_interval": "unadjusted_marginal_95_percentile",
    "scoring_population": "observed_label_intersected_with_all_three_route_availability",
    "clinical_route": "eligible_blood_and_nonblood_measurements_plus_typed_age",
    "typed_age_retained_on_all_routes": True,
    "fitted_readout": False,
    "encoder_updates": 0,
    "reused_development_posthoc_not_confirmatory": True,
}

_BASE_FLAGS = {
    "patient_level_output_emitted": False,
    "arrays_or_bootstrap_draws_emitted": False,
    "p_values_emitted": False,
    "significance_classification_emitted": False,
    "multiplicity_control_claim": False,
    "causal_attribution_claim": False,
    "external_validation_claim": False,
    "encoder_updated": False,
    "head_fitted": False,
    "new_confirmatory_test": False,
    "reused_development_population": True,
}


def _require(ok: bool) -> None:
    if not ok:
        raise ValueError(ERROR) from None


def _number(value: object, low: float, high: float) -> bool:
    return (type(value) in (int, float) and not isinstance(value, bool)
            and math.isfinite(float(value)) and low <= float(value) <= high)


def _safe_coverage(value: object) -> None:
    if value == {"status": "withheld"}:
        return
    _require(type(value) is dict and set(value) == {"status", "supported", "total"}
             and value["status"] == "released"
             and type(value["supported"]) is int and type(value["total"]) is int
             and value["total"] >= 20 and 0 <= value["supported"] <= value["total"]
             and (value["supported"] == 0 or value["supported"] >= 20)
             and (value["total"] - value["supported"] == 0
                  or value["total"] - value["supported"] >= 20))


def _metric_cell(value: object) -> bool:
    if value == {"status": "unsupported"}:
        return False
    _require(type(value) is dict and set(value) == {"status", "arms", "contrasts"}
             and value["status"] == "supported"
             and type(value["arms"]) is dict and set(value["arms"]) == set(ROUTES)
             and type(value["contrasts"]) is dict and set(value["contrasts"]) == set(CONTRASTS))
    for arm in ROUTES:
        metric = value["arms"][arm]
        _require(type(metric) is dict and set(metric) == {"auroc", "ci95"}
                 and _number(metric["auroc"], 0.0, 1.0)
                 and type(metric["ci95"]) is list and len(metric["ci95"]) == 2
                 and all(_number(point, 0.0, 1.0) for point in metric["ci95"])
                 and metric["ci95"][0] <= metric["ci95"][1])
    for name, (left, right) in CONTRASTS.items():
        metric = value["contrasts"][name]
        expected = value["arms"][left]["auroc"] - value["arms"][right]["auroc"]
        _require(type(metric) is dict and set(metric) == {"delta", "ci95"}
                 and _number(metric["delta"], -1.0, 1.0)
                 and math.isclose(metric["delta"], expected, rel_tol=0.0, abs_tol=1e-9)
                 and type(metric["ci95"]) is list and len(metric["ci95"]) == 2
                 and all(_number(point, -1.0, 1.0) for point in metric["ci95"])
                 and metric["ci95"][0] <= metric["ci95"][1])
    return True


def _pattern_table_safe(predictions: Mapping[str, np.ndarray]) -> bool:
    availability = np.stack(
        [np.isfinite(predictions[route]).all(axis=1) for route in ROUTES], axis=1
    )
    code = availability[:, 0].astype(np.uint8) * 4
    code += availability[:, 1].astype(np.uint8) * 2
    code += availability[:, 2].astype(np.uint8)
    cells = np.bincount(code, minlength=8)
    return bool(np.all((cells == 0) | (cells >= 20)))


def summarize(predictions: Mapping[str, np.ndarray], labels: np.ndarray,
              observed: np.ndarray, folds: np.ndarray,
              endpoint_names: Sequence[str], counts: np.ndarray) -> dict[str, object]:
    """Run the existing screening kernel, then apply the joint-cell privacy gate.

    Route and paired metrics remain exactly those of
    ``bran_multisource_outcome_metrics_v2.screening``.  Only release of coverage
    is additionally suppressed when any nonzero three-route availability cell
    has fewer than 20 participants.
    """
    try:
        _require(type(predictions) is dict and set(predictions) == set(ROUTES))
        _require(type(endpoint_names) in (tuple, list)
                 and len(endpoint_names) == 26
                 and all(type(name) is str for name in endpoint_names)
                 and len(set(endpoint_names)) == 26
                 and set(endpoint_names) == set(ENDPOINT_NAMES))
        _require(isinstance(labels, np.ndarray) and labels.ndim == 2
                 and labels.shape[1] == 26 and labels.shape[0] >= 0)
        _require(isinstance(observed, np.ndarray) and observed.dtype == np.dtype(bool)
                 and observed.shape == labels.shape)
        _require(isinstance(folds, np.ndarray) and folds.shape == (len(labels),)
                 and folds.dtype.kind in "iu"
                 and set(np.unique(folds)) == set(range(5)))
        for route in ROUTES:
            value = predictions[route]
            _require(isinstance(value, np.ndarray) and value.shape == labels.shape
                     and value.dtype.kind == "f")
            finite = np.isfinite(value).all(axis=1)
            absent = np.isnan(value).all(axis=1)
            _require(np.all(finite | absent))

        # Support disclosure is evaluated over all eight possible route-mask
        # cells, but the cell counts themselves never leave this function.
        pattern_safe = _pattern_table_safe(predictions)
        screening = metrics.screening(
            predictions, labels, observed, folds, tuple(endpoint_names), counts,
            {name: tuple(pair) for name, pair in CONTRASTS.items()},
        )
        if not pattern_safe:
            screening["prediction_coverage"] = {route: {"status": "withheld"} for route in ROUTES}
            screening["matched_population_coverage"] = {"status": "withheld"}
        result = {
            "schema": SCHEMA,
            "status": STATUS,
            "parameters": copy.deepcopy(PARAMETERS),
            "endpoint_names": list(ENDPOINT_NAMES),
            "screening": screening,
            "flags": {
                **_BASE_FLAGS,
                "joint_availability_coverage_suppressed": not pattern_safe,
            },
        }
        validate_result(result)
        return copy.deepcopy(result)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def validate_result(value: object) -> None:
    """Validate the public A3 envelope and inherited screening summary."""
    try:
        _require(type(value) is dict and set(value) == {
            "schema", "status", "parameters", "endpoint_names", "screening", "flags"
        })
        _require(value["schema"] == SCHEMA and value["status"] == STATUS
                 and value["parameters"] == PARAMETERS
                 and value["endpoint_names"] == list(ENDPOINT_NAMES))
        flags = value["flags"]
        _require(type(flags) is dict and set(flags) == set(_BASE_FLAGS) | {
            "joint_availability_coverage_suppressed"
        })
        for key, expected in _BASE_FLAGS.items():
            _require(flags[key] is expected)
        _require(type(flags["joint_availability_coverage_suppressed"]) is bool)

        screening = value["screening"]
        _require(type(screening) is dict and set(screening) == {
            "complete_26_panel", "macro", "endpoints", "prediction_coverage",
            "matched_population_coverage"
        })
        _require(type(screening["complete_26_panel"]) is bool
                 and type(screening["endpoints"]) is dict
                 and set(screening["endpoints"]) == set(ENDPOINT_NAMES))
        complete = True
        for row in screening["endpoints"].values():
            supported = _metric_cell(row)
            complete = complete and supported
        _require(screening["complete_26_panel"] is complete)
        if complete:
            _require(_metric_cell(screening["macro"]))
            for route in ROUTES:
                expected = float(np.mean([
                    screening["endpoints"][name]["arms"][route]["auroc"]
                    for name in ENDPOINT_NAMES
                ]))
                _require(math.isclose(
                    screening["macro"]["arms"][route]["auroc"], expected,
                    rel_tol=0.0, abs_tol=1e-9,
                ))
        else:
            _require(screening["macro"] is None)

        coverage = screening["prediction_coverage"]
        _require(type(coverage) is dict and set(coverage) == set(ROUTES))
        if flags["joint_availability_coverage_suppressed"]:
            _require(all(item == {"status": "withheld"} for item in coverage.values())
                     and screening["matched_population_coverage"] == {"status": "withheld"})
        else:
            for item in coverage.values():
                _safe_coverage(item)
            _safe_coverage(screening["matched_population_coverage"])
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None
