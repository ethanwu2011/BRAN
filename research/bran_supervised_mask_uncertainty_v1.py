"""Closed, array-only calibrated CBC uncertainty summaries.

This is a deliberately narrow evaluation seam for the prospective
supervised-missingness study.  It accepts caller-owned arrays, fits only
per-fold held-out residual radii, and returns aggregate summaries plus the
small fold/radius array.  It neither reads nor writes files and does not fit
or invoke a model.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, NoReturn

import numpy as np

from bran_clinical_semantics_v1 import CBC_FIELDS
import bran_missingness_stress_metrics_v1 as stress
import bran_native_cbc_calibration_metrics_v1 as native


ERROR = "bran_supervised_mask_uncertainty_v1_failed"
PATTERNS = (
    "single_target_hidden",
    "whole_cbc_hidden",
    "single_target_no_retina",
    "whole_cbc_no_retina",
)
VERSIONS = ("initial", "continued", "student")
REFERENCES = ("initial", "continued")
GROUPS = ("low", "middle", "high")
RADIUS_SHAPE = (len(PATTERNS), 5, len(VERSIONS), len(CBC_FIELDS))
_N_FIELDS = len(CBC_FIELDS)
_MINIMUM = 20


def _fail() -> NoReturn:
    raise ValueError(ERROR) from None


def _matrix(value: Any, rows: int | None = None) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1] != _N_FIELDS:
        _fail()
    if rows is not None and value.shape[0] != rows:
        _fail()
    if value.dtype.kind not in "fiu" or value.dtype.kind == "b":
        _fail()
    try:
        array = value.astype(np.float64, copy=False)
        if bool(np.isinf(array).any()):
            _fail()
    except Exception:
        _fail()
    return array


def _bool_matrix(value: Any, rows: int | None = None) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(bool) or value.ndim != 2:
        _fail()
    if value.shape[1] != _N_FIELDS or (rows is not None and value.shape[0] != rows):
        _fail()
    return value


def _vector(value: Any, rows: int, *, roles: bool = False) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != (rows,) or value.ndim != 1:
        _fail()
    if value.dtype.kind not in "iu" or value.dtype.kind == "b":
        _fail()
    values = value.astype(np.int64, copy=False)
    if roles:
        if bool(np.any((values != 0) & (values != 1))):
            _fail()
    elif bool(np.any(values < 0)) or bool(np.any(values > 4)):
        _fail()
    return values


def _counts(value: Any, scoring_folds: np.ndarray) -> np.ndarray:
    rows = scoring_folds.shape[0]
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (1000, rows)
        or value.ndim != 2
        or value.dtype.kind not in "iu"
    ):
        _fail()
    try:
        if bool(np.any(value < 0)) or (value.dtype.kind == "u" and bool(np.any(value > np.iinfo(np.int64).max))):
            _fail()
        counts = value.astype(np.int64, copy=False)
        for fold in range(5):
            mask = scoring_folds == fold
            if not bool(np.all(counts[:, mask].sum(axis=1) == int(np.count_nonzero(mask)))):
                _fail()
    except Exception:
        _fail()
    return counts


def _split_roles(folds: np.ndarray, roles: np.ndarray) -> None:
    if not np.array_equal(np.unique(folds), np.arange(5, dtype=np.int64)):
        _fail()
    for fold in range(5):
        local = folds == fold
        size = int(np.count_nonzero(local))
        calibration = int(np.count_nonzero(local & (roles == 0)))
        scoring = int(np.count_nonzero(local & (roles == 1)))
        if calibration != size // 2 or scoring != size - size // 2 or calibration < _MINIMUM or scoring < _MINIMUM:
            _fail()


def _groups(value: Any, rows: int, observed: Mapping[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    if not isinstance(value, Mapping) or frozenset(value) != frozenset(PATTERNS):
        _fail()
    result: dict[str, dict[str, np.ndarray]] = {}
    for pattern in PATTERNS:
        group = value[pattern]
        if not isinstance(group, Mapping) or frozenset(group) != frozenset(GROUPS):
            _fail()
        parsed = {name: _bool_matrix(group[name], rows) for name in GROUPS}
        try:
            membership = sum(item.astype(np.int8) for item in parsed.values())
            # Tail strata may be unavailable (for example, a degenerate
            # outer-training distribution).  They remain undefined rather
            # than being silently assigned to a published middle group.
            if bool(np.any(membership > 1)):
                _fail()
        except Exception:
            _fail()
        result[pattern] = parsed
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


def _support(mask: np.ndarray) -> dict[str, dict[str, object]]:
    return {field: stress.safe_coverage(mask[:, index]) for index, field in enumerate(CBC_FIELDS)}


def _public_cell(cell: dict[str, Any]) -> dict[str, Any]:
    if cell == {"status": "unsupported"}:
        return {"status": "unsupported"}
    return {
        "status": "supported",
        "arms": {
            "reference": copy.deepcopy(cell["arms"]["native"]),
            "candidate": copy.deepcopy(cell["arms"]["generative"]),
        },
        "contrast": {
            "candidate_minus_reference": copy.deepcopy(cell["contrast"]["generative_minus_native"]),
        },
    }


def _public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        field: {group: _public_cell(summary[field][group]) for group in ("overall", *GROUPS)}
        for field in CBC_FIELDS
    }


def _native_summary(public: Any) -> dict[str, Any]:
    """Translate the closed public names back for the established validator."""
    if not isinstance(public, dict) or frozenset(public) != frozenset(CBC_FIELDS):
        _fail()
    translated: dict[str, Any] = {}
    for field in CBC_FIELDS:
        groups = public[field]
        if not isinstance(groups, dict) or frozenset(groups) != frozenset(("overall", *GROUPS)):
            _fail()
        translated[field] = {}
        for group in ("overall", *GROUPS):
            cell = groups[group]
            if cell == {"status": "unsupported"}:
                translated[field][group] = cell
                continue
            if not isinstance(cell, dict) or frozenset(cell) != {"status", "arms", "contrast"} or cell.get("status") != "supported":
                _fail()
            arms, contrast = cell["arms"], cell["contrast"]
            if not isinstance(arms, dict) or frozenset(arms) != {"reference", "candidate"}:
                _fail()
            if not isinstance(contrast, dict) or frozenset(contrast) != {"candidate_minus_reference"}:
                _fail()
            translated[field][group] = {
                "status": "supported",
                "arms": {"native": arms["reference"], "generative": arms["candidate"]},
                "contrast": {"generative_minus_native": contrast["candidate_minus_reference"]},
            }
    return translated


def _validate_support(value: Any) -> None:
    if not isinstance(value, dict) or frozenset(value) != frozenset(CBC_FIELDS):
        _fail()
    for field in CBC_FIELDS:
        cell = value[field]
        if not isinstance(cell, dict) or type(cell.get("status")) is not str:
            _fail()
        if cell["status"] == "withheld":
            if frozenset(cell) != {"status"}:
                _fail()
        elif cell["status"] == "released":
            if frozenset(cell) != {"status", "supported", "total"}:
                _fail()
            if type(cell["supported"]) is not int or type(cell["total"]) is not int:
                _fail()
            if cell["total"] < _MINIMUM or not 0 <= cell["supported"] <= cell["total"]:
                _fail()
            complement = cell["total"] - cell["supported"]
            if not (cell["supported"] == 0 or cell["supported"] >= _MINIMUM):
                _fail()
            if not (complement == 0 or complement >= _MINIMUM):
                _fail()
        else:
            _fail()


def validate_radii(radii: Any) -> None:
    """Validate private patient-derived calibration parameters, never patient rows."""
    if not isinstance(radii, np.ndarray) or radii.dtype != np.dtype(np.float64) or radii.shape != RADIUS_SHAPE:
        _fail()
    try:
        if radii.flags.writeable or bool(np.isinf(radii).any()) or bool(np.any(radii[np.isfinite(radii)] < 0.0)):
            _fail()
    except Exception:
        _fail()


def validate_result(result: Any) -> None:
    """Deeply validate only the public aggregate schema returned by evaluate."""
    try:
        if not isinstance(result, dict) or frozenset(result) != frozenset(PATTERNS):
            _fail()
        for pattern in PATTERNS:
            item = result[pattern]
            if not isinstance(item, dict) or frozenset(item) != {"prediction_support", "calibrated_support", "comparisons"}:
                _fail()
            _validate_support(item["prediction_support"])
            _validate_support(item["calibrated_support"])
            for field in CBC_FIELDS:
                prediction_support = item["prediction_support"][field]
                calibrated_support = item["calibrated_support"][field]
                if prediction_support["status"] == calibrated_support["status"] == "released":
                    if (prediction_support["total"] != calibrated_support["total"]
                            or calibrated_support["supported"] > prediction_support["supported"]):
                        _fail()
            comparisons = item["comparisons"]
            if not isinstance(comparisons, dict) or frozenset(comparisons) != frozenset(REFERENCES):
                _fail()
            summaries = {reference: _native_summary(comparisons[reference]) for reference in REFERENCES}
            for summary in summaries.values():
                native.validate_result(summary)
            for field in CBC_FIELDS:
                for group in ("overall", *GROUPS):
                    left, right = comparisons["initial"][field][group], comparisons["continued"][field][group]
                    if left == {"status": "unsupported"} or right == {"status": "unsupported"}:
                        if left != right:
                            _fail()
                    elif left["arms"]["candidate"] != right["arms"]["candidate"]:
                        _fail()
    except Exception:
        _fail()


def evaluate(
    target_N9: Any,
    observed_by_pattern_N9_bool: Any,
    predictions_by_pattern_by_version_N9: Any,
    groups_by_pattern: Any,
    folds_Nint: Any,
    roles_Nint: Any,
    counts_1000xNscore: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    """Fit fold-calibration radii, then summarize only held-out scoring rows."""
    target = _matrix(target_N9)
    rows = target.shape[0]
    if rows < 1:
        _fail()
    observed = _observed(observed_by_pattern_N9_bool, rows)
    predictions = _predictions(predictions_by_pattern_by_version_N9, rows, observed, target)
    groups = _groups(groups_by_pattern, rows, observed)
    folds = _vector(folds_Nint, rows)
    roles = _vector(roles_Nint, rows, roles=True)
    _split_roles(folds, roles)
    scoring_rows = roles == 1
    score_indices = np.flatnonzero(scoring_rows)
    counts = _counts(counts_1000xNscore, folds[score_indices])

    radii = np.full(RADIUS_SHAPE, np.nan, dtype=np.float64)
    for pattern_index, pattern in enumerate(PATTERNS):
        for fold in range(5):
            calibration = (folds == fold) & (roles == 0)
            for version_index, version in enumerate(VERSIONS):
                radii[pattern_index, fold, version_index] = native.fit_radii(
                    target[calibration], predictions[pattern][version][calibration], observed[pattern][calibration],
                    alpha=0.1, minimum=_MINIMUM,
                )

    aggregate: dict[str, Any] = {}
    for pattern_index, pattern in enumerate(PATTERNS):
        prediction_mask = observed[pattern] & scoring_rows[:, None]
        radius_by_version: dict[str, np.ndarray] = {}
        for version_index, version in enumerate(VERSIONS):
            expanded = radii[pattern_index, folds, version_index]
            radius_by_version[version] = np.broadcast_to(expanded, (rows, _N_FIELDS))
        calibrated_mask = prediction_mask & np.logical_and.reduce(
            tuple(np.isfinite(radius_by_version[version]) for version in VERSIONS)
        )
        scoring_groups = {name: groups[pattern][name][score_indices].copy() for name in GROUPS}
        scoring_calibrated = calibrated_mask[score_indices]
        membership = sum(item.astype(np.int8) for item in scoring_groups.values())
        undefined_tail = scoring_calibrated & (membership == 0)
        # The established native summarizer requires each observed cell to
        # have a group.  This local fill is solely to invoke it; all affected
        # published tail cells are suppressed immediately below.
        scoring_groups["middle"][undefined_tail] = True
        comparisons: dict[str, Any] = {}
        for reference in REFERENCES:
            summary = native.summarize(
                target[score_indices],
                {"native": predictions[pattern][reference][score_indices],
                 "generative": predictions[pattern]["student"][score_indices]},
                {"native": radius_by_version[reference][score_indices],
                 "generative": radius_by_version["student"][score_indices]},
                calibrated_mask[score_indices],
                scoring_groups,
                folds[score_indices],
                counts,
            )
            for field_index, field in enumerate(CBC_FIELDS):
                if bool(np.any(undefined_tail[:, field_index])):
                    for group in GROUPS:
                        summary[field][group] = {"status": "unsupported"}
            comparisons[reference] = _public_summary(summary)
        aggregate[pattern] = {
            "prediction_support": _support(prediction_mask[score_indices]),
            "calibrated_support": _support(calibrated_mask[score_indices]),
            "comparisons": comparisons,
        }
    radii.setflags(write=False)
    validate_radii(radii)
    validate_result(aggregate)
    return aggregate, radii
