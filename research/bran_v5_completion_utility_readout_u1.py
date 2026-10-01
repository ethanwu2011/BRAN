"""Pure fixed-readout kernel for the BRAN V5 completed-CBC utility check.

This module deliberately accepts only in-memory arrays and returns only local
arrays.  It does not load data, write artifacts, or expose row-level output.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


ERROR = "completion_utility_u1_invalid"
QUALIFICATION_ERROR = "completion_utility_u1_qualification_failed"
N_FOLDS = 5
N_ENDPOINTS = 25
N_CBC = 9
N_SLOTS = 59

PARAMETERS = {
    "schema": "bran-v5-completion-utility-u1",
    "local_only": True,
    "folds": N_FOLDS,
    "endpoints": N_ENDPOINTS,
    "clinical_slots": N_SLOTS,
    "cbc_assays": N_CBC,
    "cohort": "complete_original_cbc_and_some_noncbc_physiology",
    "eligible_slots": "admitted_slots_including_all_cbc_assays;_noncbc_subset_derived_internally",
    "noncbc_features": "eligible_values_train_median_fill_plus_masks_plus_fold_age7",
    "cbc_input": "never_in_noncbc_design_or_mask_features",
    "classifier": "StandardScaler_LogisticRegression_C1_lbfgs_max5000",
    "ridge": "StandardScaler_Ridge_alpha1",
    "endpoint_fold_qualification": "at_least_20_observed_training_and_both_classes_present",
    "arms": (
        "median", "ridge", "native_no_retina", "observed_cbc",
        "native_with_retina", "noncbc_only",
    ),
    "native_estimates": "precomputed_outer_heldout_no_refit",
    "arm_provenance": {
        "median": "outer_training_cbc_field_median_replacement",
        "ridge": "outer_training_raw_noncbc_ridge_replacement",
        "native_no_retina": "precomputed_outer_heldout_native_replacement_no_retina",
        "observed_cbc": "original_observed_cbc_reference_only",
        "native_with_retina": "precomputed_outer_heldout_native_replacement_with_retina",
        "noncbc_only": "no_cbc_input",
    },
    "patient_level_output_emitted": False,
}


class QualificationFailure(ValueError):
    """A fixed, non-row-specific signal that frozen study qualification failed."""

    def __init__(self):
        super().__init__(QUALIFICATION_ERROR)


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError(ERROR)


def _qualify(condition: bool) -> None:
    if not condition:
        raise QualificationFailure()


def _bool_vector(value, n: int) -> np.ndarray:
    out = np.asarray(value)
    _require(out.shape == (n,) and out.dtype == np.bool_)
    return out


def _numeric_matrix(value, shape: tuple[int, int]) -> np.ndarray:
    out = np.asarray(value)
    _require(out.shape == shape and out.dtype.kind in "iuf")
    return out.astype(float, copy=False)


def _indices(value, *, allow_empty: bool = False) -> tuple[int, ...]:
    try:
        values = tuple(value)
    except TypeError:
        raise ValueError(ERROR) from None
    _require(allow_empty or bool(values))
    _require(all(isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)) for x in values))
    _require(len(set(values)) == len(values) and all(0 <= x < N_SLOTS for x in values))
    return tuple(int(x) for x in values)


def erase_cbc(c, cm, cbc_slots, eligible_indices):
    """Erase CBC and all non-eligible clinical slots before non-CBC statistics.

    Missing/invalid payloads are zeroed as well, so masked garbage cannot enter
    a later calculation through a value-only path.
    """
    try:
        values = _numeric_matrix(c, (len(np.asarray(c)), N_SLOTS))
        mask = np.asarray(cm)
        _require(mask.shape == values.shape and mask.dtype == np.bool_)
        cbc = _indices(cbc_slots)
        eligible = _indices(eligible_indices)
        _require(set(cbc).issubset(eligible))
        noncbc = tuple(slot for slot in eligible if slot not in set(cbc))
        _require(bool(noncbc))
        # A flagged non-CBC measurement is an observed input, not an invitation
        # to silently reinterpret a non-finite value as missing.  In contrast,
        # payloads behind a false flag are never inspected by downstream fits.
        _require(np.isfinite(values[:, noncbc][mask[:, noncbc]]).all())
        kept = np.zeros_like(mask)
        kept[:, noncbc] = mask[:, noncbc]
        erased = np.zeros_like(values, dtype=float)
        erased[kept] = values[kept]
        return erased, kept
    except Exception:
        raise ValueError(ERROR) from None


def _folds(value, n: int) -> np.ndarray:
    raw = np.asarray(value)
    _require(raw.shape == (n,) and raw.dtype.kind in "iu" and raw.dtype != np.bool_)
    out = raw.astype(np.int64, copy=False)
    _require(np.all((out >= 0) & (out < N_FOLDS)))
    _require(set(np.unique(out)) == set(range(N_FOLDS)))
    return out


def _age_by_fold(value, n: int) -> dict[int, np.ndarray]:
    _require(isinstance(value, Mapping) and set(value) == set(range(N_FOLDS)))
    out = {}
    for fold in range(N_FOLDS):
        age = _numeric_matrix(value[fold], (n, 7))
        _require(np.isfinite(age).all())
        out[fold] = age
    return out


def _labels_and_mask(labels, labelmask, n: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(labels)
    mask = np.asarray(labelmask)
    _require(raw.shape == (n, N_ENDPOINTS) and mask.shape == raw.shape and mask.dtype == np.bool_)
    # Convert only labelled cells, so hidden payload in unlabelled cells is inert.
    out = np.zeros((n, N_ENDPOINTS), dtype=float)
    try:
        out[mask] = np.asarray(raw[mask], dtype=float)
    except Exception:
        raise ValueError(ERROR) from None
    _require(np.isfinite(out[mask]).all() and np.all((out[mask] == 0) | (out[mask] == 1)))
    return out, mask


def _noncbc_design(erased_c: np.ndarray, erased_cm: np.ndarray, eligible: tuple[int, ...],
                   train: np.ndarray, age: np.ndarray) -> np.ndarray:
    values, observed = erased_c[:, eligible], erased_cm[:, eligible]
    fill = np.zeros(len(eligible), dtype=float)
    for field in range(len(eligible)):
        donors = values[train, field][observed[train, field]]
        if donors.size:
            fill[field] = float(np.median(donors))
    filled = np.where(observed, values, fill[None, :])
    return np.hstack((filled, observed.astype(float), age))


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000)
    scaler = StandardScaler()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaler.fit_transform(x), y)
    _require(not any(issubclass(item.category, ConvergenceWarning) for item in caught))
    # Retain the paired scaler privately on the estimator; it is never returned.
    model._bran_u1_scaler = scaler
    return model


def _probability(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(model._bran_u1_scaler.transform(x))[:, 1]


def _ridge_predict(x: np.ndarray, donor: np.ndarray, target: np.ndarray, test: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    model = Ridge(alpha=1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaler.fit_transform(x[donor]), target[donor])
    _require(not any(issubclass(item.category, ConvergenceWarning) for item in caught))
    return model.predict(scaler.transform(x[test]))


def fit_predict(c, cm, age7_by_fold, folds, labels, labelmask, cbc_slots, eligible_indices,
                native_no_retina, native_with_retina, progress=None):
    """Fit only fixed outer-fold readouts and return local prediction arrays.

    Native completion arrays are inputs, never training features or targets.
    The original CBC values are accessed only to train/evaluate the explicit
    complete-CBC reference and to supervise the raw non-CBC ridge completions.
    """
    try:
        raw_c = np.asarray(c)
        _require(raw_c.ndim == 2 and raw_c.shape[1] == N_SLOTS)
        n = raw_c.shape[0]
        values = _numeric_matrix(raw_c, (n, N_SLOTS))
        raw_cm = np.asarray(cm)
        _require(raw_cm.shape == values.shape and raw_cm.dtype == np.bool_)
        cbc = _indices(cbc_slots)
        eligible = _indices(eligible_indices)
        _require(len(cbc) == N_CBC and set(cbc).issubset(eligible))
        noncbc_eligible = tuple(slot for slot in eligible if slot not in set(cbc))
        _require(bool(noncbc_eligible))
        erased_c, erased_cm = erase_cbc(values, raw_cm, cbc, eligible)
        fold_values = _folds(folds, n)
        age_values = _age_by_fold(age7_by_fold, n)
        label_values, label_observed = _labels_and_mask(labels, labelmask, n)
        native_nr = _numeric_matrix(native_no_retina, (n, N_CBC))
        native_wr = _numeric_matrix(native_with_retina, (n, N_CBC))

        original_cbc_observed = raw_cm[:, cbc] & np.isfinite(values[:, cbc])
        original_cbc = np.where(original_cbc_observed, values[:, cbc], np.nan)
        some_noncbc = erased_cm[:, noncbc_eligible].any(axis=1)
        cohort = original_cbc_observed.all(axis=1) & some_noncbc
        _require(np.isfinite(native_nr[cohort]).all() and np.isfinite(native_wr[cohort]).all())

        predictions = {arm: np.full((n, N_ENDPOINTS), np.nan) for arm in PARAMETERS["arms"]}
        measurement_predictions = {
            "median": np.full((n, N_CBC), np.nan),
            "ridge": np.full((n, N_CBC), np.nan),
            "native_no_retina": np.full((n, N_CBC), np.nan),
            "native_with_retina": np.full((n, N_CBC), np.nan),
        }
        support = np.zeros((N_FOLDS, N_ENDPOINTS), dtype=bool)

        for fold in range(N_FOLDS):
            if progress is not None:
                progress("completion_utility_u1", fold, "fixed_readout")
            outer_train = fold_values != fold
            test = (fold_values == fold) & cohort
            design = _noncbc_design(erased_c, erased_cm, noncbc_eligible, outer_train, age_values[fold])
            donor_base = outer_train & some_noncbc
            median = np.empty(N_CBC, dtype=float)
            ridge = np.full((n, N_CBC), np.nan)
            for field in range(N_CBC):
                donor = donor_base & original_cbc_observed[:, field]
                _qualify(int(donor.sum()) >= 20)
                median[field] = float(np.median(original_cbc[donor, field]))
                if test.any():
                    ridge[test, field] = _ridge_predict(design, donor, original_cbc[:, field], test)
            measurement_predictions["median"][test] = median
            measurement_predictions["ridge"][test] = ridge[test]
            measurement_predictions["native_no_retina"][test] = native_nr[test]
            measurement_predictions["native_with_retina"][test] = native_wr[test]

            classifier_train = outer_train & cohort
            _qualify(int(classifier_train.sum()) >= 20)
            x_noncbc = design
            x_train_noncbc = x_noncbc[classifier_train]
            x_test_noncbc = x_noncbc[test]
            x_train_cbc = np.hstack((x_train_noncbc, original_cbc[classifier_train]))
            x_test_by_arm = {
                "median": np.hstack((x_test_noncbc, np.broadcast_to(median, (int(test.sum()), N_CBC)))),
                "ridge": np.hstack((x_test_noncbc, ridge[test])),
                "native_no_retina": np.hstack((x_test_noncbc, native_nr[test])),
                "observed_cbc": np.hstack((x_test_noncbc, original_cbc[test])),
                "native_with_retina": np.hstack((x_test_noncbc, native_wr[test])),
            }
            for endpoint in range(N_ENDPOINTS):
                known_train = classifier_train & label_observed[:, endpoint]
                y = label_values[known_train, endpoint]
                if y.size < 20 or np.unique(y).size != 2:
                    continue
                support[fold, endpoint] = True
                classifier = _fit_logistic(x_train_cbc[label_observed[classifier_train, endpoint]], y)
                noncbc = _fit_logistic(x_train_noncbc[label_observed[classifier_train, endpoint]], y)
                if test.any():
                    for arm, x_test in x_test_by_arm.items():
                        predictions[arm][test, endpoint] = _probability(classifier, x_test)
                    predictions["noncbc_only"][test, endpoint] = _probability(noncbc, x_test_noncbc)

        return {
            "schema": PARAMETERS["schema"],
            "parameters": PARAMETERS,
            "predictions": predictions,
            "cohort_mask": cohort,
            "endpoint_fold_support": support,
            "measurement_predictions": measurement_predictions,
            "diagnostics": {"native_estimates": "precomputed_not_refit", "output": "local_arrays_only"},
        }
    except QualificationFailure:
        raise
    except Exception:
        raise ValueError(ERROR) from None
