"""Disclosure-safe aggregate metrics for the frozen BRAN stress diagnostic.

The caller owns all row-level arrays.  This module returns only fixed-schema
coverage and AUROC aggregates; it never stores or serializes patient rows,
predictions, bootstrap draws, or endpoint support counts.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from bran_missingness_stress_v1 import PATTERNS
import run_bran_overnight_diagnostic_v1 as base


_ENDPOINTS = 26
_BOOTSTRAP_DRAWS = 1000
_MIN_SUPPORT = 20
_MIN_FINITE_DRAWS = 900
_INVALID = "missingness_stress_metrics_contract_failed"


def _fail() -> None:
    raise ValueError(_INVALID)


def _names(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        _fail()
    try:
        result = tuple(value)
    except (TypeError, ValueError):
        _fail()
    if (len(result) != _ENDPOINTS or
            any(type(name) is not str or not name for name in result) or
            len(set(result)) != _ENDPOINTS):
        _fail()
    return result


def safe_coverage(mask: np.ndarray) -> dict[str, object]:
    """Release coverage only when both cells satisfy the privacy threshold."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 1 or mask.dtype != np.dtype(bool):
        _fail()
    total = int(mask.shape[0])
    supported = int(mask.sum())
    complement = total - supported
    releasable = (
        total >= _MIN_SUPPORT
        and (supported == 0 or supported >= _MIN_SUPPORT)
        and (complement == 0 or complement >= _MIN_SUPPORT)
    )
    if not releasable:
        return {"status": "withheld"}
    return {"status": "released", "supported": supported, "total": total}


def _finite_interval(draws: np.ndarray) -> list[float] | None:
    values = np.asarray(draws, dtype=float)
    if values.ndim != 1 or values.shape[0] != _BOOTSTRAP_DRAWS:
        _fail()
    finite = values[np.isfinite(values)]
    if finite.shape[0] < _MIN_FINITE_DRAWS:
        return None
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def _validate_inputs(predictions, labels, observed, folds, names, counts):
    endpoint_names = _names(names)
    if not isinstance(labels, np.ndarray) or labels.ndim != 2 or labels.shape[1] != _ENDPOINTS:
        _fail()
    n = labels.shape[0]
    if n < 1 or labels.dtype.kind not in "iufb":
        _fail()
    if (not isinstance(observed, np.ndarray) or observed.shape != labels.shape
            or observed.dtype != np.dtype(bool)):
        _fail()
    observed_values = labels[observed]
    if (not np.isfinite(observed_values).all()
            or np.any((observed_values != 0) & (observed_values != 1))):
        _fail()

    if not isinstance(folds, np.ndarray) or folds.shape != (n,) or folds.dtype.kind not in "iu":
        _fail()
    if not np.array_equal(np.unique(folds), np.arange(5, dtype=folds.dtype)):
        _fail()

    if (not isinstance(counts, np.ndarray) or counts.shape != (_BOOTSTRAP_DRAWS, n)
            or counts.dtype.kind not in "iu" or np.any(counts < 0)):
        _fail()
    for fold in range(5):
        fold_rows = folds == fold
        expected = int(np.count_nonzero(fold_rows))
        if not np.all(counts[:, fold_rows].sum(axis=1) == expected):
            _fail()

    if not isinstance(predictions, Mapping) or set(predictions) != set(PATTERNS):
        _fail()
    checked: dict[str, np.ndarray] = {}
    rowmask: dict[str, np.ndarray] = {}
    for pattern in PATTERNS:
        value = predictions[pattern]
        if not isinstance(value, np.ndarray) or value.shape != (n, _ENDPOINTS):
            _fail()
        if value.dtype.kind not in "iuf":
            _fail()
        finite = np.isfinite(value)
        all_finite = finite.all(axis=1)
        all_nan = np.isnan(value).all(axis=1)
        # Every row is either a fully usable prediction or a fully abstaining
        # row.  Infinities and partially missing endpoint vectors are invalid.
        if not np.all(all_finite | all_nan):
            _fail()
        if np.any(value[all_finite] < 0) or np.any(value[all_finite] > 1):
            _fail()
        checked[pattern] = value
        rowmask[pattern] = all_finite
    return endpoint_names, labels, observed, folds, counts, checked, rowmask


def _endpoint_cell(y: np.ndarray, p_available: np.ndarray, p_masked: np.ndarray,
                   valid: np.ndarray, folds: np.ndarray, counts: np.ndarray):
    fold_valid = np.zeros_like(valid, dtype=bool)
    for fold in range(5):
        in_fold = valid & (folds == fold)
        if np.any(in_fold & (y == 0)) and np.any(in_fold & (y == 1)):
            fold_valid |= folds == fold
    valid = valid & fold_valid
    positives = int(np.count_nonzero(valid & (y == 1)))
    negatives = int(np.count_nonzero(valid & (y == 0)))
    if positives < _MIN_SUPPORT or negatives < _MIN_SUPPORT:
        return {"status": "unsupported"}, None

    available_point = float(base.fold_weighted_auc(y, p_available, valid, folds))
    masked_point = float(base.fold_weighted_auc(y, p_masked, valid, folds))
    available_draws = np.asarray(
        base._weighted_auc_draws(y, p_available, valid, folds, counts), dtype=float
    )
    masked_draws = np.asarray(
        base._weighted_auc_draws(y, p_masked, valid, folds, counts), dtype=float
    )
    delta_draws = masked_draws - available_draws
    available_ci = _finite_interval(available_draws)
    masked_ci = _finite_interval(masked_draws)
    delta_ci = _finite_interval(delta_draws)
    if (not math.isfinite(available_point) or not math.isfinite(masked_point)
            or available_ci is None or masked_ci is None or delta_ci is None):
        return {"status": "unsupported"}, None
    cell = {
        "status": "supported",
        "available": {"auroc": available_point, "ci95": available_ci},
        "masked": {"auroc": masked_point, "ci95": masked_ci},
        "delta": {"auroc_delta": masked_point - available_point, "ci95": delta_ci},
    }
    return cell, (available_draws, masked_draws, delta_draws)


def _macro_cell(points, draws):
    available_points = [item["available"]["auroc"] for item in points]
    masked_points = [item["masked"]["auroc"] for item in points]
    available = np.mean(np.stack([item[0] for item in draws], axis=0), axis=0)
    masked = np.mean(np.stack([item[1] for item in draws], axis=0), axis=0)
    delta = np.mean(np.stack([item[2] for item in draws], axis=0), axis=0)
    available_ci = _finite_interval(available)
    masked_ci = _finite_interval(masked)
    delta_ci = _finite_interval(delta)
    if available_ci is None or masked_ci is None or delta_ci is None:
        return None
    available_point = float(np.mean(available_points))
    masked_point = float(np.mean(masked_points))
    return {
        "available": {"auroc": available_point, "ci95": available_ci},
        "masked": {"auroc": masked_point, "ci95": masked_ci},
        "delta": {"auroc_delta": masked_point - available_point, "ci95": delta_ci},
    }


def summarize(predictions, labels, observed, folds, names, counts):
    """Summarize every fixed stress pattern against available evidence."""
    endpoint_names, y_all, observed, folds, counts, checked, rowmask = _validate_inputs(
        predictions, labels, observed, folds, names, counts
    )
    result: dict[str, dict[str, object]] = {}
    available_rows = rowmask["available"]
    for pattern in PATTERNS:
        common_rows = available_rows & rowmask[pattern]
        endpoints: dict[str, object] = {}
        supported_points = []
        supported_draws = []
        for j, name in enumerate(endpoint_names):
            valid = common_rows & observed[:, j]
            cell, draw_triplet = _endpoint_cell(
                y_all[:, j], checked["available"][:, j], checked[pattern][:, j],
                valid, folds, counts
            )
            endpoints[name] = cell
            if cell.get("status") == "supported":
                supported_points.append(cell)
                supported_draws.append(draw_triplet)
        macro = None
        if len(supported_points) == _ENDPOINTS:
            macro = _macro_cell(supported_points, supported_draws)
        result[pattern] = {
            "coverage": safe_coverage(rowmask[pattern]),
            "endpoints": endpoints,
            "macro": macro,
        }
    return result


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _ci(value, lower: float, upper: float) -> None:
    if (not isinstance(value, list) or len(value) != 2
            or any(not _number(item) for item in value)
            or not lower <= value[0] <= value[1] <= upper):
        _fail()


def _metric(value, *, delta: bool = False) -> None:
    expected = {"auroc_delta", "ci95"} if delta else {"auroc", "ci95"}
    if not isinstance(value, dict) or set(value) != expected:
        _fail()
    key = "auroc_delta" if delta else "auroc"
    if not _number(value[key]) or not (-1 if delta else 0) <= value[key] <= 1:
        _fail()
    _ci(value["ci95"], -1 if delta else 0, 1)


def _cell(cell) -> None:
    if not isinstance(cell, dict) or set(cell) != {"status", "available", "masked", "delta"}:
        _fail()
    if cell["status"] != "supported":
        _fail()
    _metric(cell["available"])
    _metric(cell["masked"])
    _metric(cell["delta"], delta=True)
    if abs(cell["delta"]["auroc_delta"] -
           (cell["masked"]["auroc"] - cell["available"]["auroc"])) >= 1e-10:
        _fail()


def _coverage(value) -> None:
    if not isinstance(value, dict) or "status" not in value:
        _fail()
    if value["status"] == "withheld":
        if set(value) != {"status"}:
            _fail()
        return
    if value["status"] != "released" or set(value) != {"status", "supported", "total"}:
        _fail()
    supported, total = value["supported"], value["total"]
    if (type(supported) is not int or type(total) is not int or total < 0
            or supported < 0 or supported > total):
        _fail()
    complement = total - supported
    if (total < _MIN_SUPPORT
            or (supported != 0 and supported < _MIN_SUPPORT)
            or (complement != 0 and complement < _MIN_SUPPORT)):
        _fail()


def validate_result(results, names) -> None:
    """Validate the closed disclosure-safe result schema in memory."""
    endpoint_names = _names(names)
    if not isinstance(results, dict) or set(results) != set(PATTERNS):
        _fail()
    for pattern in PATTERNS:
        value = results[pattern]
        if not isinstance(value, dict) or set(value) != {"coverage", "endpoints", "macro"}:
            _fail()
        _coverage(value["coverage"])
        endpoints = value["endpoints"]
        if not isinstance(endpoints, dict) or set(endpoints) != set(endpoint_names):
            _fail()
        unsupported = False
        supported_cells = []
        for name in endpoint_names:
            cell = endpoints[name]
            if isinstance(cell, dict) and cell == {"status": "unsupported"}:
                unsupported = True
                continue
            _cell(cell)
            supported_cells.append(cell)
        macro = value["macro"]
        if macro is None:
            continue
        if unsupported:
            _fail()
        if len(supported_cells) != _ENDPOINTS or not isinstance(macro, dict):
            _fail()
        if set(macro) != {"available", "masked", "delta"}:
            _fail()
        _metric(macro["available"])
        _metric(macro["masked"])
        _metric(macro["delta"], delta=True)
        available_point = float(np.mean([cell["available"]["auroc"] for cell in supported_cells]))
        masked_point = float(np.mean([cell["masked"]["auroc"] for cell in supported_cells]))
        if (abs(macro["available"]["auroc"] - available_point) >= 1e-10
                or abs(macro["masked"]["auroc"] - masked_point) >= 1e-10
                or abs(macro["delta"]["auroc_delta"] -
                       (masked_point - available_point)) >= 1e-10):
            _fail()
