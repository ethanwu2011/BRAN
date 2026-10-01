"""Local-only, discovery-fit structure kernel for 192-dimensional BRAN states.

This is an unsupervised technical discovery primitive, not an efficacy, stability, or
novel-subtype claim.  Bootstrap replication and independent characterization require
a future protocol.  It accepts no clinical outcomes, CGM, ECG, or test arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


STATE_WIDTH = 192
MIN_DISCOVERY_ROWS = 80
MIN_VALIDATION_ROWS = 40
MIN_GROUP_ROWS = 20
K_VALUES = (1, 2, 3, 4)
RANDOM_STATE = 93501

STATUS_SUPPORTED = "supported"
STATUS_UNSUPPORTED_INSUFFICIENT_ROWS = "unsupported_insufficient_rows"
STATUS_UNSUPPORTED_NOT_CONVERGED = "unsupported_not_converged"
STATUS_UNSUPPORTED_GROUP_SUPPORT = "unsupported_group_support"


class StructureInputError(ValueError):
    pass


@dataclass(repr=False)
class _Candidate:
    k: int
    status: str
    mean_log_density: float | None = None


@dataclass(repr=False)
class StructureResult:
    """Local result; fitted objects and assignments are intentionally not exported."""

    status: str
    _scaler: StandardScaler | None
    _pca: PCA | None
    _mixture: GaussianMixture | None
    _candidates: tuple[_Candidate, ...]
    _selected_k: int | None

    def predict(self, new_states: Any) -> np.ndarray:
        """Return local cluster labels for finite N×192 state vectors only."""
        if self.status != STATUS_SUPPORTED or self._scaler is None or self._pca is None or self._mixture is None:
            raise RuntimeError("structure model is unsupported")
        states = _states(new_states, "new_states")
        return self._mixture.predict(self._pca.transform(self._scaler.transform(states)))


def fit_structure(discovery_states: Any, validation_states: Any) -> StructureResult:
    """Fit scaler/PCA/GMM on discovery only, then select with held-out density."""
    discovery = _states(discovery_states, "discovery_states")
    validation = _states(validation_states, "validation_states")
    if len(discovery) < MIN_DISCOVERY_ROWS or len(validation) < MIN_VALIDATION_ROWS:
        return StructureResult(STATUS_UNSUPPORTED_INSUFFICIENT_ROWS, None, None, None, (), None)

    scaler = StandardScaler()
    discovery_scaled = scaler.fit_transform(discovery)
    validation_scaled = scaler.transform(validation)
    pca = PCA(n_components=8, whiten=True, svd_solver="full")
    discovery_reduced = pca.fit_transform(discovery_scaled)
    validation_reduced = pca.transform(validation_scaled)
    candidates: list[_Candidate] = []
    mixtures: dict[int, GaussianMixture] = {}
    for k in K_VALUES:
        mixture = GaussianMixture(
            n_components=k, covariance_type="diag", reg_covar=1e-4, n_init=5,
            max_iter=500, random_state=RANDOM_STATE,
        ).fit(discovery_reduced)
        if not mixture.converged_:
            candidates.append(_Candidate(k, STATUS_UNSUPPORTED_NOT_CONVERGED))
            continue
        discovery_groups = np.bincount(mixture.predict(discovery_reduced), minlength=k)
        validation_groups = np.bincount(mixture.predict(validation_reduced), minlength=k)
        if np.any(discovery_groups < MIN_GROUP_ROWS) or np.any(validation_groups < MIN_GROUP_ROWS):
            candidates.append(_Candidate(k, STATUS_UNSUPPORTED_GROUP_SUPPORT))
            continue
        mean = float(np.mean(mixture.score_samples(validation_reduced)))
        candidates.append(_Candidate(k, STATUS_SUPPORTED, mean))
        mixtures[k] = mixture
    supported = [candidate for candidate in candidates if candidate.status == STATUS_SUPPORTED]
    if not supported:
        return StructureResult(STATUS_UNSUPPORTED_GROUP_SUPPORT, scaler, pca, None, tuple(candidates), None)
    best = max(supported, key=lambda candidate: candidate.mean_log_density)
    best_mixture = mixtures[best.k]
    scores = best_mixture.score_samples(validation_reduced)
    one_se = float(np.std(scores, ddof=1) / math.sqrt(len(validation)))
    selected = min(
        (candidate for candidate in supported if candidate.mean_log_density >= best.mean_log_density - one_se),
        key=lambda candidate: candidate.k,
    )
    return StructureResult(STATUS_SUPPORTED, scaler, pca, mixtures[selected.k], tuple(candidates), selected.k)


def aggregate_diagnostics(result: Any) -> dict[str, Any] | None:
    """Emit a closed aggregate schema only after a supported selection exists."""
    if not isinstance(result, StructureResult) or result.status != STATUS_SUPPORTED or result._selected_k not in K_VALUES:
        return None
    candidates: dict[str, dict[str, Any]] = {}
    for candidate in result._candidates:
        if candidate.status == STATUS_SUPPORTED:
            if type(candidate.mean_log_density) is not float or not math.isfinite(candidate.mean_log_density):
                return None
            candidates[str(candidate.k)] = {"status": STATUS_SUPPORTED, "validation_mean_log_density": candidate.mean_log_density}
        elif candidate.status in {STATUS_UNSUPPORTED_NOT_CONVERGED, STATUS_UNSUPPORTED_GROUP_SUPPORT} and candidate.mean_log_density is None:
            candidates[str(candidate.k)] = {"status": candidate.status}
        else:
            return None
    diagnostics = {"status": STATUS_SUPPORTED, "selected_k": result._selected_k, "candidates": candidates}
    return diagnostics if validate_aggregate_diagnostics(diagnostics) else None


def validate_aggregate_diagnostics(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"status", "selected_k", "candidates"}:
        return False
    if value["status"] != STATUS_SUPPORTED or type(value["selected_k"]) is not int or value["selected_k"] not in K_VALUES:
        return False
    candidates = value["candidates"]
    if not isinstance(candidates, dict) or set(candidates) != {str(k) for k in K_VALUES}:
        return False
    selected_supported = False
    for k in K_VALUES:
        candidate = candidates[str(k)]
        if not isinstance(candidate, dict) or "status" not in candidate:
            return False
        if candidate["status"] == STATUS_SUPPORTED:
            if set(candidate) != {"status", "validation_mean_log_density"}:
                return False
            score = candidate["validation_mean_log_density"]
            if type(score) not in (int, float) or isinstance(score, bool) or not math.isfinite(score):
                return False
            selected_supported = selected_supported or k == value["selected_k"]
        elif candidate["status"] in {STATUS_UNSUPPORTED_NOT_CONVERGED, STATUS_UNSUPPORTED_GROUP_SUPPORT}:
            if set(candidate) != {"status"}:
                return False
        else:
            return False
    return selected_supported


def _states(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != STATE_WIDTH:
        raise StructureInputError(f"{name} must be N×{STATE_WIDTH}")
    if array.dtype.kind not in "fiu" or array.dtype.kind == "b":
        raise StructureInputError(f"{name} must be numeric")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise StructureInputError(f"{name} must be finite")
    return array
