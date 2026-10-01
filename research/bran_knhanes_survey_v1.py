"""Local-only survey statistics following KNHANES guide PDF page 43.

Pure arrays, no file access or logging. This is NOT a disclosure/release API.
An admitting caller must pass the complete positive-weight survey design and
a separate analysis-domain mask; dropping out-of-domain PSUs is incorrect.
Fit uncertainty is not captured by these fixed-prediction design intervals.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import t

GUIDE_SHA256 = '921818c62267bd2949dd08e7d0143ef8cd30eb3086722e696481aa162ba42ce3'


def require(condition):
    if not condition:
        raise ValueError('survey_contract_failed')


@dataclass(frozen=True, repr=False)
class SurveyEstimate:
    mean: float
    variance: float
    ci95: tuple
    degrees_freedom: int

    def __repr__(self):
        return '<PrivateSurveyEstimate>'

    def __reduce__(self):
        raise TypeError('private_survey_result_not_serializable')


def numeric(a, n=None):
    require(type(a) is np.ndarray and a.ndim == 1 and a.dtype.kind in 'fiu')
    if n is not None:
        require(len(a) == n)


def estimate_mean(value, weight, stratum, psu, domain):
    """Taylor-linearized weighted domain mean, with-replacement PSU variance.

    Stratum and PSU codes must be already namespaced by source year. They are
    private numeric grouping codes, never outputs. All strata need >=2 PSUs;
    no silent singleton-stratum repair or survey redesign is permitted.
    """
    numeric(value)
    n = len(value)
    require(n > 0)
    for a in (weight, stratum, psu):
        numeric(a, n)
        require(np.isfinite(a).all())
    require(stratum.dtype.kind in 'iu' and psu.dtype.kind in 'iu')
    require(type(domain) is np.ndarray and domain.dtype == np.dtype(bool)
            and domain.shape == (n,) and domain.any())
    require((weight > 0).all() and np.isfinite(value[domain]).all())
    # Normalize first to avoid overflow from arbitrary survey-weight scales.
    w = weight.astype(float) / np.max(weight)
    denominator = np.sum(w[domain])
    require(np.isfinite(denominator) and denominator > 0)
    mean = float(np.sum((w[domain] / denominator) * value[domain]))
    residual = np.zeros(n, dtype=float)
    residual[domain] = (w[domain] / denominator) * (value[domain] - mean)
    require(np.isfinite(residual).all() and np.isfinite(mean))
    variance = 0.0
    degrees = 0
    for s in np.unique(stratum):
        selected = stratum == s
        codes, inverse = np.unique(psu[selected], return_inverse=True)
        m = len(codes)
        require(m >= 2)
        totals = np.bincount(inverse, weights=residual[selected], minlength=m)
        variance += m / (m - 1) * float(np.sum((totals - totals.mean()) ** 2))
        degrees += m - 1
    require(degrees > 0 and np.isfinite(variance) and variance >= 0)
    margin = float(t.ppf(.975, degrees) * np.sqrt(variance))
    require(np.isfinite(margin))
    return SurveyEstimate(mean, variance, (mean - margin, mean + margin), degrees)


def estimate_mae_difference(target, candidate, reference, weight, stratum, psu, domain):
    """Paired candidate-minus-reference MAE difference on identical support."""
    numeric(target)
    for a in (candidate, reference):
        numeric(a, len(target))
    require(type(domain) is np.ndarray and domain.dtype == np.dtype(bool)
            and domain.shape == target.shape)
    for a in (target, candidate, reference):
        require(np.isfinite(a[domain]).all())
    difference = np.zeros(len(target))
    difference[domain] = (np.abs(candidate[domain] - target[domain])
                          - np.abs(reference[domain] - target[domain]))
    return estimate_mean(difference, weight, stratum, psu, domain)
