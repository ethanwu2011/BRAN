"""Array-only, disclosure-safe native V2 summaries; no model/source I/O.

This first evaluation describes the two frozen candidates. It cannot promote a
candidate: historical reference replay, information-matched FM comparisons and
the unchanged advancement gates are separate prerequisites. Arrays and bootstrap
draws remain inside the caller's quiet local process.
"""
import numpy as np

import run_bran_overnight_diagnostic_v1 as auc
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_missingness_stress_metrics_v1 import safe_coverage


def require(ok):
    if not ok: raise ValueError('multisource_outcome_metrics_failed')


def validate_counts(counts, folds):
    require(type(folds) is np.ndarray and folds.ndim == 1 and folds.dtype.kind in 'iu'
            and set(np.unique(folds)) == set(range(5)))
    require(type(counts) is np.ndarray and counts.shape == (1000, len(folds))
            and counts.dtype.kind in 'iu' and np.all(counts >= 0))
    for fold in range(5):
        rows = folds == fold
        require(np.all(counts[:, rows].sum(1) == rows.sum()))


def interval(draws):
    require(np.shape(draws) == (1000,))
    valid = np.asarray(draws)[np.isfinite(draws)]
    return None if len(valid) < 900 else [float(x) for x in np.quantile(valid, [.025, .975])]


def _cell(points, draws, contrasts, metric):
    ci = {key: interval(draws[key]) for key in points}
    ds = {name: draws[left] - draws[right] for name, (left, right) in contrasts.items()}
    dc = {name: interval(value) for name, value in ds.items()}
    if any(value is None for value in (*ci.values(), *dc.values())):
        return {'status': 'unsupported'}
    return {'status': 'supported',
            'arms': {key: {metric: float(value), 'ci95': ci[key]} for key, value in points.items()},
            'contrasts': {name: {'delta': float(points[left] - points[right]), 'ci95': dc[name]}
                          for name, (left, right) in contrasts.items()}}


def screening(predictions, labels, observed, folds, names, counts, contrasts):
    validate_counts(counts, folds)
    n = len(folds)
    require(labels.shape == observed.shape == (n, 26) and observed.dtype == bool
            and np.isfinite(labels[observed]).all() and np.isin(labels[observed], [0, 1]).all())
    require(len(names) == len(set(names)) == 26 and all(type(x) is str and x for x in names))
    require(type(predictions) is dict and len(predictions) >= 1
            and all(type(key) is str for key in predictions))
    availability = {}
    for name, values in predictions.items():
        require(values.shape == labels.shape and values.dtype.kind == 'f')
        valid = np.isfinite(values).all(1)
        require(np.all(valid | np.isnan(values).all(1))
                and np.all((values[valid] >= 0) & (values[valid] <= 1)))
        availability[name] = valid
    require(all(left in predictions and right in predictions for left, right in contrasts.values()))
    common = np.logical_and.reduce(list(availability.values()))
    result = {}; points = {key: [] for key in predictions}; draws = {key: [] for key in predictions}
    for j, endpoint in enumerate(names):
        valid = observed[:, j] & common
        y = labels[:, j]
        valid = valid.copy()
        for fold in range(5):
            local = valid & (folds == fold)
            if not np.any(local & (y == 0)) or not np.any(local & (y == 1)):
                valid[folds == fold] = False
        if np.count_nonzero(valid & (y == 0)) < 20 or np.count_nonzero(valid & (y == 1)) < 20:
            result[endpoint] = {'status': 'unsupported'}; continue
        ps = {key: auc.fold_weighted_auc(y, value[:, j], valid, folds) for key, value in predictions.items()}
        ds = {key: auc._weighted_auc_draws(y, value[:, j], valid, folds, counts)
              for key, value in predictions.items()}
        result[endpoint] = _cell(ps, ds, contrasts, 'auroc')
        if result[endpoint]['status'] == 'supported':
            for key in predictions: points[key].append(ps[key]); draws[key].append(ds[key])
    complete = all(cell['status'] == 'supported' for cell in result.values())
    macro = _cell({key: np.mean(value) for key, value in points.items()},
                  {key: np.mean(value, axis=0) for key, value in draws.items()}, contrasts, 'auroc') if complete else None
    return {'complete_26_panel': complete, 'macro': macro, 'endpoints': result,
            'prediction_coverage': {key: safe_coverage(value) for key, value in availability.items()},
            'matched_population_coverage': safe_coverage(common)}


def completion(predictions, target, observed, folds, counts, *, contrasts, low_hb=True):
    validate_counts(counts, folds)
    require(target.shape == observed.shape == (len(folds), 9) and observed.dtype == bool
            and np.isfinite(target[observed]).all())
    require(type(predictions) is dict and len(predictions) >= 1)
    require(all(value.shape == target.shape and value.dtype.kind == 'f' and not np.isinf(value).any()
                for value in predictions.values()))
    require(all(np.isnan(value[~observed]).all() for value in predictions.values()))
    require(all(left in predictions and right in predictions for left, right in contrasts.values()))
    result = {}
    for j, field in enumerate(CBC_FIELDS):
        valid = observed[:, j] & np.logical_and.reduce([np.isfinite(value[:, j]) for value in predictions.values()])
        groups = {'overall': valid}
        if low_hb and field == 'hemoglobin':
            low = valid & (target[:, j] < 12.0); other = valid & (target[:, j] >= 12.0)
            # Complementary suppression: do not publish a low stratum that can
            # reveal an unsupported remainder through the overall result.
            groups['hb_below_12_research_stratum'] = low if min(low.sum(), other.sum()) >= 20 else np.zeros_like(low)
        result[field] = {'prediction_coverage_among_observed': safe_coverage(valid[observed[:, j]]), 'groups': {}}
        for group, rows in groups.items():
            if rows.sum() < 20:
                result[field]['groups'][group] = {'status': 'unsupported'}; continue
            errors = {key: value[rows, j] - target[rows, j] for key, value in predictions.items()}
            weights = counts[:, rows].astype(float); den = weights.sum(1)
            ps = {key: np.mean(np.abs(value)) for key, value in errors.items()}
            ds = {key: np.divide(weights @ np.abs(value), den,
                                out=np.full(1000, np.nan), where=den > 0) for key, value in errors.items()}
            cell = _cell(ps, ds, contrasts, 'mae')
            if cell['status'] == 'supported':
                for key, values in errors.items():
                    cell['arms'][key].update(bias=float(values.mean()), rmse=float(np.sqrt(np.square(values).mean())))
            result[field]['groups'][group] = cell
    return result
