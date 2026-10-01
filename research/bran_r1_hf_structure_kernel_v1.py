"""Array-only, outcome-blind R1 heart-failure structure kernel.

This is deliberately a new protocol surface.  It neither loads data nor
accepts outcomes, identifiers, frame artifacts, or clinical readout designs.
The caller is responsible for authenticating those private inputs before it
provides already-aligned 192-coordinate state arrays.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


ERROR = "bran_r1_hf_structure_kernel_contract_failed"
STATE_WIDTH = 192
K_VALUES = (1, 2, 3, 4, 5, 6)
MIN_DISCOVERY_ROWS = 80
MIN_SELECTION_ROWS = 40
MIN_REPLICATION_ROWS = 40
MIN_GROUP_ROWS = 20
RANDOM_STATE = 20260919
BOOTSTRAPS = 20
MIN_CONVERGED_BOOTSTRAPS = 18
ARI_MEDIAN_MINIMUM = 0.8
ARI_Q025_MINIMUM = 0.6


class StructureKernelError(ValueError):
    """Private-array input failure, reported with a source-free message."""


def _require(value: bool) -> None:
    if not value:
        raise StructureKernelError(ERROR)


def _states(value: Any) -> np.ndarray:
    array = np.asarray(value)
    _require(array.ndim == 2 and array.shape[1] == STATE_WIDTH)
    _require(array.dtype.kind in "fiu" and array.dtype.kind != "b")
    array = np.asarray(array, dtype=np.float64)
    _require(np.isfinite(array).all())
    return array


def _components(discovery: np.ndarray) -> int:
    # This is intentionally based on the actual discovery fit sample.  A
    # bootstrap is itself a discovery refit, so it recomputes this bound.
    nonzero = int(np.count_nonzero(np.var(discovery, axis=0) > 0.0))
    return min(20, len(discovery) - 1, nonzero)


def _fit_transform(discovery: np.ndarray) -> tuple[StandardScaler, PCA, np.ndarray] | None:
    count = _components(discovery)
    if count < 1:
        return None
    scaler = StandardScaler()
    scaled = scaler.fit_transform(discovery)
    pca = PCA(n_components=count, whiten=True, svd_solver="full")
    reduced = pca.fit_transform(scaled)
    if not np.isfinite(reduced).all():
        return None
    return scaler, pca, reduced


def _fit_mixture(reduced: np.ndarray, k: int) -> GaussianMixture | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mixture = GaussianMixture(
            n_components=k, covariance_type="diag", reg_covar=1e-4,
            n_init=10, max_iter=500, random_state=RANDOM_STATE,
        ).fit(reduced)
    # Only sklearn's explicit non-convergence is an analysis status.  A fit
    # exception is an execution failure and must fail closed at the boundary.
    if type(mixture.converged_) is not bool:
        raise TypeError("invalid mixture convergence state")
    return mixture if mixture.converged_ else None


def _candidate_report(candidates: dict[int, tuple[GaussianMixture | None, float | None]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for k in K_VALUES:
        mixture, bic = candidates[k]
        if mixture is None or bic is None:
            result[str(k)] = {"status": "not_converged"}
        else:
            result[str(k)] = {"status": "converged", "discovery_bic": float(bic)}
    return result


def _quantiles(values: list[float]) -> dict[str, float]:
    q = np.quantile(np.asarray(values, dtype=np.float64), (.025, .5, .975))
    return {"q025": float(q[0]), "q500": float(q[1]), "q975": float(q[2])}


def _labels(value: Any, length: int, k: int) -> np.ndarray:
    labels = np.asarray(value)
    _require(type(k) is int and k in K_VALUES and labels.shape == (length,))
    _require(labels.dtype.kind in "iu" and labels.dtype.kind != "b")
    _require(np.all((labels >= 0) & (labels < k)))
    return labels.astype(np.int64, copy=False)


def _support(mixture: GaussianMixture, reduced_roles: tuple[np.ndarray, np.ndarray, np.ndarray], k: int) -> bool:
    return all(np.all(np.bincount(_labels(mixture.predict(values), len(values), k), minlength=k) >= MIN_GROUP_ROWS)
               for values in reduced_roles)


def _stability(discovery: np.ndarray, selection: np.ndarray, replication: np.ndarray,
               locked: GaussianMixture, locked_scaler: StandardScaler, locked_pca: PCA, k: int) -> dict[str, object]:
    locked_selection = _labels(locked.predict(locked_pca.transform(locked_scaler.transform(selection))), len(selection), k)
    locked_replication = _labels(locked.predict(locked_pca.transform(locked_scaler.transform(replication))), len(replication), k)
    rng = np.random.default_rng(RANDOM_STATE)
    selection_ari: list[float] = []
    replication_ari: list[float] = []
    failed = 0
    for _ in range(BOOTSTRAPS):
        rows = rng.integers(0, len(discovery), size=len(discovery))
        try:
            fitted = _fit_transform(discovery[rows])
            if fitted is None:
                failed += 1
                continue
            scaler, pca, reduced = fitted
            mixture = _fit_mixture(reduced, k)
            if mixture is None:
                failed += 1
                continue
            selection_labels = _labels(mixture.predict(pca.transform(scaler.transform(selection))), len(selection), k)
            replication_labels = _labels(mixture.predict(pca.transform(scaler.transform(replication))), len(replication), k)
            a = float(adjusted_rand_score(locked_selection, selection_labels))
            b = float(adjusted_rand_score(locked_replication, replication_labels))
            if not (math.isfinite(a) and math.isfinite(b)):
                raise FloatingPointError
        except (np.linalg.LinAlgError, FloatingPointError):
            failed += 1
            continue
        selection_ari.append(a)
        replication_ari.append(b)
    converged = len(selection_ari)
    available = converged >= MIN_CONVERGED_BOOTSTRAPS
    return {
        "requested_bootstraps": BOOTSTRAPS,
        "converged_bootstraps": converged,
        "failed_bootstraps": failed,
        "minimum_converged_bootstraps": MIN_CONVERGED_BOOTSTRAPS,
        "available": available,
        "selection_ari": _quantiles(selection_ari) if available else None,
        "replication_ari": _quantiles(replication_ari) if available else None,
    }


def _stability_gate(stability: dict[str, object]) -> bool:
    if not stability["available"] or stability["converged_bootstraps"] < MIN_CONVERGED_BOOTSTRAPS:
        return False
    try:
        return all(stability[name]["q500"] >= ARI_MEDIAN_MINIMUM
                   and stability[name]["q025"] >= ARI_Q025_MINIMUM
                   for name in ("selection_ari", "replication_ari"))
    except Exception:
        return False


@dataclass(frozen=True, repr=False)
class HFStructureResult:
    """Private fitted transform/mixture plus a closed aggregate receipt."""

    status: str
    selected_k: int | None
    _scaler: StandardScaler | None
    _pca: PCA | None
    _mixture: GaussianMixture | None
    _aggregate: dict[str, object]

    @property
    def aggregate(self) -> dict[str, object]:
        return deepcopy(self._aggregate)

    def predict(self, new192: Any) -> np.ndarray:
        if self.status not in {"supported", "no_discrete_groups"}:
            raise StructureKernelError(ERROR)
        _require(self._scaler is not None and self._pca is not None and self._mixture is not None)
        values = _states(new192)
        try:
            labels = self._mixture.predict(self._pca.transform(self._scaler.transform(values)))
        except Exception:
            raise StructureKernelError(ERROR) from None
        _require(labels.shape == (len(values),) and labels.dtype.kind in "iu")
        return np.asarray(labels, dtype=np.int64)


def _result(status: str, selected_k: int | None, scaler: StandardScaler | None,
            pca: PCA | None, mixture: GaussianMixture | None, aggregate: dict[str, object]) -> HFStructureResult:
    _require(validate_aggregate(aggregate))
    return HFStructureResult(status, selected_k, scaler, pca, mixture, deepcopy(aggregate))


def fit_structure(discovery_states: Any, selection_states: Any, replication_states: Any) -> HFStructureResult:
    """Fit discovery-only BIC structure and locked-K stability diagnostics.

    Selection and replication arrays are never passed into fitting, PCA, BIC,
    or K selection.  They are used only after K is frozen for support and ARI.
    """
    try:
        discovery, selection, replication = (_states(value) for value in
                                             (discovery_states, selection_states, replication_states))
        if len(discovery) < MIN_DISCOVERY_ROWS or len(selection) < MIN_SELECTION_ROWS or len(replication) < MIN_REPLICATION_ROWS:
            return _result("unsupported_insufficient_rows", None, None, None, None,
                           {"status": "unsupported_insufficient_rows"})
        fitted = _fit_transform(discovery)
        if fitted is None:
            return _result("unsupported_degenerate_input", None, None, None, None,
                           {"status": "unsupported_degenerate_input"})
        scaler, pca, reduced_discovery = fitted
        candidates: dict[int, tuple[GaussianMixture | None, float | None]] = {}
        for k in K_VALUES:
            mixture = _fit_mixture(reduced_discovery, k)
            bic: float | None = None
            if mixture is not None:
                score = float(mixture.bic(reduced_discovery))
                if math.isfinite(score):
                    bic = score
                else:
                    mixture = None
            candidates[k] = (mixture, bic)
        report = _candidate_report(candidates)
        usable = [(k, pair[0], pair[1]) for k, pair in candidates.items()
                  if pair[0] is not None and pair[1] is not None]
        if not usable:
            return _result("unsupported_not_converged", None, None, None, None,
                           {"status": "unsupported_not_converged", "candidates": report})
        # Iteration order is ascending K; strict comparison keeps the smaller
        # K under exact equal discovery BIC.
        selected_k, mixture, best_bic = usable[0]
        for k, candidate, bic in usable[1:]:
            if bic < best_bic:
                selected_k, mixture, best_bic = k, candidate, bic
        _require(mixture is not None)
        reduced_roles = tuple(pca.transform(scaler.transform(values)) for values in (discovery, selection, replication))
        if not _support(mixture, reduced_roles, selected_k):
            return _result("unsupported_group_support", selected_k, scaler, pca, mixture,
                           {"status": "unsupported_group_support", "selected_k": selected_k,
                            "candidates": report, "group_support_at_least_20": False})
        if selected_k == 1:
            return _result("no_discrete_groups", selected_k, scaler, pca, mixture,
                           {"status": "no_discrete_groups", "selected_k": 1,
                            "candidates": report, "group_support_at_least_20": True,
                            "stability_gate": False})
        stability = _stability(discovery, selection, replication, mixture, scaler, pca, selected_k)
        aggregate = {"status": "supported", "selected_k": selected_k, "candidates": report,
                     "group_support_at_least_20": True, "stability": stability,
                     "stability_gate": _stability_gate(stability)}
        return _result("supported", selected_k, scaler, pca, mixture, aggregate)
    except StructureKernelError:
        raise
    except Exception:
        raise StructureKernelError(ERROR) from None


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(float(value))


def _valid_candidates(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {str(k) for k in K_VALUES}:
        return False
    for k in K_VALUES:
        item = value[str(k)]
        if not isinstance(item, dict) or item.get("status") not in {"converged", "not_converged"}:
            return False
        if item["status"] == "converged":
            if set(item) != {"status", "discovery_bic"} or not _finite_number(item["discovery_bic"]):
                return False
        elif set(item) != {"status"}:
            return False
    return True


def _bic_selected(candidates: dict[str, dict[str, object]]) -> int | None:
    """Return the frozen lowest-BIC choice, retaining the smaller K on ties."""
    usable = [(k, float(candidates[str(k)]["discovery_bic"])) for k in K_VALUES
              if candidates[str(k)]["status"] == "converged"]
    return min(usable, key=lambda item: (item[1], item[0]))[0] if usable else None


def _valid_stability(value: object) -> bool:
    keys = {"requested_bootstraps", "converged_bootstraps", "failed_bootstraps",
            "minimum_converged_bootstraps", "available", "selection_ari", "replication_ari"}
    if not isinstance(value, dict) or set(value) != keys:
        return False
    if (type(value["requested_bootstraps"]) is not int or type(value["minimum_converged_bootstraps"]) is not int
            or value["requested_bootstraps"] != BOOTSTRAPS or value["minimum_converged_bootstraps"] != MIN_CONVERGED_BOOTSTRAPS
            or type(value["converged_bootstraps"]) is not int or type(value["failed_bootstraps"]) is not int
            or type(value["available"]) is not bool or value["converged_bootstraps"] < 0
            or value["failed_bootstraps"] < 0 or value["converged_bootstraps"] + value["failed_bootstraps"] != BOOTSTRAPS
            or value["available"] != (value["converged_bootstraps"] >= MIN_CONVERGED_BOOTSTRAPS)):
        return False
    for key in ("selection_ari", "replication_ari"):
        q = value[key]
        if not value["available"]:
            if q is not None:
                return False
            continue
        if not isinstance(q, dict) or set(q) != {"q025", "q500", "q975"}:
            return False
        if not all(_finite_number(q[name]) and -1 <= q[name] <= 1 for name in q) or not q["q025"] <= q["q500"] <= q["q975"]:
            return False
    return True


def validate_aggregate(value: object) -> bool:
    """Validate the closed aggregate schema without accepting private payloads."""
    if not isinstance(value, dict) or not isinstance(value.get("status"), str):
        return False
    status = value["status"]
    if status in {"unsupported_insufficient_rows", "unsupported_degenerate_input"}:
        return set(value) == {"status"}
    if status == "unsupported_not_converged":
        return (set(value) == {"status", "candidates"} and _valid_candidates(value["candidates"])
                and _bic_selected(value["candidates"]) is None)
    if status == "unsupported_group_support":
        return (set(value) == {"status", "selected_k", "candidates", "group_support_at_least_20"}
                and type(value["selected_k"]) is int and value["selected_k"] in K_VALUES
                and _valid_candidates(value["candidates"]) and value["candidates"][str(value["selected_k"])]["status"] == "converged"
                and value["selected_k"] == _bic_selected(value["candidates"])
                and value["group_support_at_least_20"] is False)
    if status == "no_discrete_groups":
        return (set(value) == {"status", "selected_k", "candidates", "group_support_at_least_20", "stability_gate"}
                and type(value["selected_k"]) is int and value["selected_k"] == 1 and _valid_candidates(value["candidates"])
                and value["candidates"]["1"]["status"] == "converged"
                and value["selected_k"] == _bic_selected(value["candidates"])
                and value["group_support_at_least_20"] is True and value["stability_gate"] is False)
    if status != "supported":
        return False
    keys = {"status", "selected_k", "candidates", "group_support_at_least_20", "stability", "stability_gate"}
    if (set(value) != keys or type(value["selected_k"]) is not int or value["selected_k"] not in K_VALUES[1:]
            or not _valid_candidates(value["candidates"]) or value["candidates"][str(value["selected_k"])]["status"] != "converged"
            or value["selected_k"] != _bic_selected(value["candidates"])
            or value["group_support_at_least_20"] is not True or type(value["stability_gate"]) is not bool
            or not _valid_stability(value["stability"])):
        return False
    return value["stability_gate"] is _stability_gate(value["stability"])
