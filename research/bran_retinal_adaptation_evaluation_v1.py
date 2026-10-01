"""Prospective private-array retinal readouts and disclosure-safe aggregation.

No source discovery, I/O, fitting of encoders, or clinical-validation claim.
Returned predictions, patient features and readout parameters stay local.
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from bran_retinal_group_readiness_v1 import ENDPOINTS

ARMS = ('base', 'masked_control', 'supervised_candidate')
PRIMARY = ('diabetic_retinopathy', 'drusens', 'increased_cup_disc')
MIN_CLASS = 20


def require(value):
    if not value:
        raise ValueError('invalid retinal comparison input')


def coarse(n):
    return 0 if n == 0 else ('<20' if n < 20 else (int(n) // 20) * 20)


def validate_labels(labels, observed):
    require(type(labels) is np.ndarray and labels.ndim == 2 and labels.shape[1] == 13
            and np.issubdtype(labels.dtype, np.floating))
    require(type(observed) is np.ndarray and observed.dtype == bool and observed.shape == labels.shape)
    require(np.all(np.isfinite(labels[observed])) and np.all(np.isin(labels[observed], (0, 1))))


def group_patients(groups, split, labels, observed, ages, features):
    """Any positive eye => positive; negative only if every eye is known negative."""
    validate_labels(labels, observed)
    n = len(labels)
    require(n > 0 and type(groups) is np.ndarray and groups.shape == (n,)
            and np.issubdtype(groups.dtype, np.integer))
    require(type(split) is np.ndarray and split.shape == (n,) and np.all(np.isin(split, (0, 1, 2))))
    require(type(ages) is np.ndarray and ages.shape == (n,) and np.all(np.isfinite(ages))
            and np.all(ages >= 18))
    require(type(features) is dict and set(features) == set(ARMS))
    d = features['base'].shape[1] if type(features['base']) is np.ndarray and features['base'].ndim == 2 else 0
    require(d > 0 and all(type(x) is np.ndarray and x.shape == (n, d) and np.all(np.isfinite(x))
                         for x in features.values()))
    unique, inverse = np.unique(groups, return_inverse=True)
    result = {'labels': np.full((len(unique), 13), np.nan),
              'observed': np.zeros((len(unique), 13), bool),
              'split': np.empty(len(unique), np.int8), 'ages': np.empty(len(unique)),
              'features': {arm: np.empty((len(unique), d)) for arm in ARMS}}
    for i in range(len(unique)):
        rows = inverse == i
        require(len(np.unique(split[rows])) == 1)
        result['split'][i] = split[rows][0]
        result['ages'][i] = ages[rows].mean()
        positive = np.any(observed[rows] & (labels[rows] == 1), axis=0)
        negative = np.all(observed[rows] & (labels[rows] == 0), axis=0)
        result['labels'][i, positive] = 1
        result['labels'][i, negative] = 0
        result['observed'][i] = positive | negative
        for arm in ARMS:
            result['features'][arm][i] = features[arm][rows].mean(axis=0)
    return result


def support(labels, observed, split, code):
    rows = split == code
    return np.asarray([np.sum(rows & observed[:, j] & (labels[:, j] == value))
                       for j in range(13) for value in (0, 1)]).reshape(13, 2)


def training_label_policy(image_labels, image_observed, image_split, patient_data):
    """Train-only image prevalence weights, restricted to person-supported tasks."""
    validate_labels(image_labels, image_observed)
    require(type(image_split) is np.ndarray and image_split.shape == (len(image_labels),))
    s = support(patient_data['labels'], patient_data['observed'], patient_data['split'], 0)
    usable = np.all(s >= MIN_CLASS, axis=1)
    counts = support(image_labels, image_observed, image_split, 0)
    weight = np.ones(13, np.float32)
    require(np.all(counts[usable] > 0))
    weight[usable] = np.clip(np.sqrt(counts[usable, 0] / counts[usable, 1]), 1, 10)
    return weight, usable


def fit_readouts(patient_data, *, include_age=False):
    """Fixed-C balanced logistic readouts; neither validation nor test used to fit."""
    require(type(include_age) is bool)
    labels, observed, split = (patient_data[k] for k in ('labels', 'observed', 'split'))
    validate_labels(labels, observed)
    train = split == 0
    usable = np.all(support(labels, observed, split, 0) >= MIN_CLASS, axis=1)
    require(train.sum() >= 2 * MIN_CLASS)
    predictions = np.full((3, len(labels), 13), np.nan)
    heads = {}
    for ai, arm in enumerate(ARMS):
        x = patient_data['features'][arm]
        if include_age:
            x = np.column_stack((x, patient_data['ages']))
        require(np.all(np.isfinite(x)))
        scaler = StandardScaler().fit(x[train])
        z = scaler.transform(x)
        heads[arm] = {'mean': scaler.mean_.copy(), 'scale': scaler.scale_.copy(), 'endpoints': {}}
        for j in np.flatnonzero(usable):
            rows = train & observed[:, j]
            model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, class_weight='balanced')
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(z[rows], labels[rows, j])
            predictions[ai, :, j] = model.predict_proba(z)[:, 1]
            heads[arm]['endpoints'][ENDPOINTS[j]] = {
                'coefficient': model.coef_.copy(), 'intercept': model.intercept_.copy(),
            }
    return {'predictions': predictions, 'heads': heads, 'include_age': include_age}


def _auc_plan(labels, scores):
    """Sort/tie-group once; bootstrap only changes integer patient weights."""
    _, bins = np.unique(scores, return_inverse=True)
    return np.asarray(labels == 1), bins


def _weighted_auc(plan, weights):
    positive, bins = plan
    pw = np.bincount(bins, weights=weights * positive)
    nw = np.bincount(bins, weights=weights * ~positive)
    np_, nn_ = pw.sum(), nw.sum()
    if np_ == 0 or nn_ == 0:
        return float('nan')
    # Half-credit for ties, exactly the weighted empirical AUROC estimand.
    return float(np.dot(pw, np.cumsum(nw) - .5 * nw) / (np_ * nn_))


def summarize(patient_data, readout, *, draws=2000, seed=73193):
    """Paired person bootstrap; only >=20 positive and negative cases released.

    These intervals are conditional on the fixed trained encoders and readouts;
    they do not quantify retraining uncertainty or independent clinical validity.
    """
    require(type(draws) is int and draws >= 20 and type(seed) is int and seed >= 0)
    labels, observed, split = (patient_data[k] for k in ('labels', 'observed', 'split'))
    validate_labels(labels, observed)
    prediction = readout['predictions']
    require(type(prediction) is np.ndarray and prediction.shape == (3, len(labels), 13))
    counts = {name: support(labels, observed, split, i)
              for i, name in enumerate(('train', 'validation', 'test'))}
    supported = np.all(counts['train'] >= MIN_CLASS, axis=1) & np.all(counts['test'] >= MIN_CLASS, axis=1)
    tested = np.flatnonzero(supported)
    test_rows = np.flatnonzero(split == 2)
    y, ob = labels[test_rows], observed[test_rows]
    p = prediction[:, test_rows]
    require(len(test_rows) >= 2 * MIN_CLASS and np.all(np.isfinite(p[:, :, tested]))
            and np.all((p[:, :, tested] >= 0) & (p[:, :, tested] <= 1)))
    primary_indices = [ENDPOINTS.index(name) for name in PRIMARY]
    macro_supported = bool(np.all(supported[primary_indices])
                           and np.all(counts['validation'][primary_indices] >= MIN_CLASS))

    plans = {(ai, j): (np.flatnonzero(ob[:, j]), _auc_plan(y[ob[:, j], j], p[ai, ob[:, j], j]))
             for j in tested for ai in range(3)}

    def score(rows):
        values = np.full((3, 13), np.nan)
        weights = np.bincount(rows, minlength=len(y))
        for j in tested:
            for ai in range(3):
                known, plan = plans[ai, j]
                values[ai, j] = _weighted_auc(plan, weights[known])
        return values

    point = score(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    boot = np.stack([score(rng.integers(0, len(y), len(y))) for _ in range(draws)])

    def interval(samples):
        finite = samples[np.isfinite(samples)]
        require(len(finite) >= .95 * draws)
        return [float(v) for v in np.quantile(finite, [.025, .975])]

    def estimates(points, samples):
        return {'auroc': {arm: {'estimate': float(points[ai]), 'ci95': interval(samples[:, ai])}
                          for ai, arm in enumerate(ARMS)},
                'candidate_delta': {arm: {'estimate': float(points[2] - points[ai]),
                                          'ci95': interval(samples[:, 2] - samples[:, ai])}
                                    for ai, arm in enumerate(ARMS[:2])}}

    endpoints = {}
    for j, name in enumerate(ENDPOINTS):
        endpoints[name] = {
            'status': 'evaluated' if supported[j] else 'insufficient_support',
            'counts_coarsened': {part: {'negative': coarse(int(v[j, 0])), 'positive': coarse(int(v[j, 1]))}
                                 for part, v in counts.items()},
            'estimates': estimates(point[:, j], boot[:, :, j]) if supported[j] else None,
        }
    macro = estimates(point[:, primary_indices].mean(axis=1), boot[:, :, primary_indices].mean(axis=2)) if macro_supported else None
    gate = bool(macro is not None and all(x['ci95'][0] > 0 for x in macro['candidate_delta'].values())
                and all(np.all(point[2, primary_indices] - point[ai, primary_indices] >= -.02) for ai in (0, 1)))
    if readout['include_age']:
        gate = False  # Prespecified secondary analysis cannot promote the encoder.
    return {'schema': 'bran-retinal-adaptation-evaluation-v1',
            'analysis': 'age_adjusted_secondary' if readout['include_age'] else 'image_only_primary',
            'arms': list(ARMS), 'primary_named_endpoints': list(PRIMARY),
            'bootstrap': {'unit': 'patient', 'draws': draws, 'seed': seed, 'percentile': 95,
                          'conditional_on_fixed_training': True, 'draws_released': False},
            'endpoints': endpoints, 'primary_macro': macro,
            'eligible_for_separate_unified_refit_review': gate,
            'unified_model_promoted': False, 'clinical_validation_established': False,
            'patient_level_output_emitted': False}
