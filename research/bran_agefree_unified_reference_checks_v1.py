"""No-I/O canary against pre-existing, authenticated reference point estimates.

The caller must obtain and pin the expected scalars from their actual prior
source artifacts BEFORE reference fitting. Recomputing expectations from the
new predictions defeats this check. No private arrays/hashes are returned.
"""
import numpy as np

import bran_agefree_unified_oof_v1 as oof
import bran_agefree_unified_metrics_v1 as evaluation
from bran_clinical_semantics_v1 import CBC_FIELDS

ERROR = 'age-free refit reference replay rejected'


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def validate_expected(expected, names):
    require(type(names) is tuple and len(names) == len(set(names)) == 26
            and all(type(name) is str and name for name in names)
            and type(expected) is dict and set(expected) == {'screening', 'native_whole_cbc'}
            and type(expected['screening']) is dict
            and set(expected['screening']) == set(evaluation.REFERENCE_SCREEN)
            and type(expected['native_whole_cbc']) is dict
            and set(expected['native_whole_cbc']) == set(CBC_FIELDS))
    for endpoints in expected['screening'].values():
        require(type(endpoints) is dict and set(endpoints) == set(names)
                and all(evaluation.metrics.finite(v) and 0 <= v <= 1 for v in endpoints.values()))
    for cell in expected['native_whole_cbc'].values():
        require(type(cell) is dict and set(cell) == {'mae', 'mse'}
                and all(evaluation.metrics.finite(v) and v >= 0 for v in cell.values())
                and cell['mse'] + 1e-10 >= cell['mae'] ** 2)


def check(reference, *, expected, expected_sha256, expected_binding, labels,
          label_observed, clinical59, observed, eligible, registry_names, folds, endpoint_names):
    """Replay all26 AUROCs for all12 reference routes and whole-CBC all9.

    Native routes use the historical common clinical/retinal evaluation mask;
    raw/blood/FM canaries use all observed endpoint rows, matching their original
    benchmarks. This is distinct from the matched comparison population used
    later for the candidate. No classifier is fit here; no canary is tuned.
    """
    try:
        validate_expected(expected, endpoint_names)
        require(type(expected_sha256) is str and oof.private_digest(expected) == expected_sha256
                and type(reference) is dict and set(reference) == {
                    'binding', 'screening', 'completion', 'completion_available', 'stress'}
                and reference['binding'] == expected_binding
                and type(reference['screening']) is dict
                and set(reference['screening']) == set(evaluation.REFERENCE_SCREEN))
        n = len(folds)
        require(type(folds) is np.ndarray and folds.dtype == np.int64 and folds.shape == (n,)
                and set(folds.tolist()) == set(range(5)))
        evaluation._labels(labels, label_observed, n)
        require(type(clinical59) is np.ndarray and clinical59.dtype.kind == 'f' and clinical59.shape == (n, 59)
                and all(type(x) is np.ndarray and x.dtype == bool and x.shape == (n, 59) for x in (observed, eligible))
                and type(registry_names) is tuple and len(registry_names) == len(set(registry_names)) == 59
                and all(type(name) is str and name for name in registry_names))
        preds = reference['screening']
        for prediction in preds.values():
            oof._prediction(prediction, (n, 26), np.isfinite(prediction).all(1)[:, None], probability=True)
        native_present = np.logical_and.reduce([
            np.isfinite(preds['initial_' + route]).all(1) for route in oof.ROUTES])
        for j, name in enumerate(endpoint_names):
            y = labels[:, j]
            for arm in evaluation.REFERENCE_SCREEN:
                native = arm.startswith('initial_')
                valid = label_observed[:, j].copy()
                if native:
                    valid &= native_present
                require(np.isfinite(preds[arm][valid, j]).all())
                contributing = np.zeros(n, bool)
                for fold in range(5):
                    local = valid & (folds == fold)
                    if np.any(local & (y == 0)) and np.any(local & (y == 1)):
                        contributing |= local
                require(min(np.sum(contributing & (y == 0)), np.sum(contributing & (y == 1))) >= 20)
                actual = evaluation.metrics.base.fold_weighted_auc(y, preds[arm][:, j], valid, folds)
                tolerance = 1e-8 if arm in evaluation.EXTERNAL else 1e-10
                require(np.isfinite(actual) and abs(actual - expected['screening'][arm][name]) <= tolerance)
        slots = tuple(registry_names.index(field) for field in CBC_FIELDS)
        require(all(slot < 48 for slot in slots))
        target = clinical59[:, slots]; truth_mask = (observed & eligible)[:, slots]
        support = reference['completion_available']['whole_cbc_hidden']
        require(type(support) is np.ndarray and support.dtype == bool and support.shape == (n, 9)
                and np.isfinite(target[truth_mask]).all())
        values = reference['completion']['whole_cbc_hidden']['initial_native']
        oof._prediction(values, (n, 9), support)
        for j, field in enumerate(CBC_FIELDS):
            valid = support[:, j] & truth_mask[:, j]
            require(np.sum(valid) >= 20)
            errors = values[valid, j] - target[valid, j]
            for statistic, value in (('mae', np.abs(errors).mean()), ('mse', np.square(errors).mean())):
                require(np.isclose(value, expected['native_whole_cbc'][field][statistic], rtol=1e-10, atol=1e-10))
        return True
    except Exception:
        raise ValueError(ERROR) from None
