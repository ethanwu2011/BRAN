"""Aggregate-only paired screening inference for frozen Patient Atlas V6.2."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from patient_atlas_disease_universal import BreadthEndpoint, _family_score


NONINFERIORITY_MARGIN = 0.001


def summarize_v6_2_vs_concat(
    *,
    v6_2_both_losses: np.ndarray,
    tuned_concat_losses: np.ndarray,
    endpoints: Sequence[BreadthEndpoint],
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    v6 = np.asarray(v6_2_both_losses, dtype=np.float64)
    concat = np.asarray(tuned_concat_losses, dtype=np.float64)
    if (
        v6.shape != concat.shape
        or v6.ndim != 2
        or v6.shape[1] != len(endpoints)
        or len(v6) == 0
        or not np.array_equal(np.isfinite(v6), np.isfinite(concat))
    ):
        raise ValueError("V6.2 and concat loss matrices must be patient-paired")
    if bootstrap_samples < 100 or not 0.5 < confidence_level < 1.0:
        raise ValueError("V6.2 screening bootstrap differs")
    v6_score, v6_endpoint, v6_family = _family_score(v6, endpoints)
    concat_score, concat_endpoint, concat_family = _family_score(concat, endpoints)
    point = v6_score - concat_score
    generator = np.random.default_rng(int(bootstrap_seed))
    draws = np.empty(bootstrap_samples, dtype=np.float64)
    for iteration in range(bootstrap_samples):
        indices = generator.integers(0, len(v6), size=len(v6))
        draws[iteration] = (
            _family_score(v6[indices], endpoints)[0]
            - _family_score(concat[indices], endpoints)[0]
        )
    alpha = 1.0 - confidence_level
    lower = float(np.quantile(draws, alpha))
    upper = float(np.quantile(draws, confidence_level))
    return {
        "primary_estimand": "V6.2-both minus tuned-concat equal-organ-family mean cross-fitted log loss",
        "negative_contrast_favors": "v6_2_both",
        "positive_contrast_favors": "tuned_concat",
        "v6_2_both_score": v6_score,
        "tuned_concat_score": concat_score,
        "v6_2_both_minus_tuned_concat": point,
        "paired_two_sided_confidence_interval": [
            float(np.quantile(draws, alpha / 2.0)),
            float(np.quantile(draws, 1.0 - alpha / 2.0)),
        ],
        "paired_one_sided_lower_confidence_bound": lower,
        "paired_one_sided_upper_confidence_bound": upper,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "noninferiority_passed": upper < NONINFERIORITY_MARGIN,
        "point_estimate_not_worse_than_concat": point <= 0.0,
        "locked_concat_target_recovered": (
            upper < NONINFERIORITY_MARGIN and point <= 0.0
        ),
        "v6_2_superiority_passed": upper < 0.0,
        "tuned_concat_superiority_passed": lower > 0.0,
        "per_organ_family_v6_2_minus_concat": {
            family: v6_family[family] - concat_family[family]
            for family in sorted(v6_family)
        },
        "per_endpoint_v6_2_minus_concat": {
            endpoint: v6_endpoint[endpoint] - concat_endpoint[endpoint]
            for endpoint in sorted(v6_endpoint)
        },
        "bootstrap": {
            "unit": "patient",
            "paired_resampling": True,
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "interval": "percentile",
        },
        "contains_patient_losses_or_bootstrap_draws": False,
    }


__all__ = ["NONINFERIORITY_MARGIN", "summarize_v6_2_vs_concat"]
