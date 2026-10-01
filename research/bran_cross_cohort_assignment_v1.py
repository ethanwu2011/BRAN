"""In-memory transport for an already locked 192-coordinate cluster rule.

This module deliberately has no source loaders, fitting calls, outcome inputs,
or artifact I/O. Trusted fitted objects are serialized in memory only to hash
their integrity. It is a technical assignment guard, not evidence of stable,
validated, or new subtypes, nor of safe transport to a new domain.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import pickle
import re
from typing import Any

import numpy as np


ERROR = "bran_cross_cohort_assignment_contract_failed"
STATE_WIDTH = 192
MIN_CALIBRATION_ROWS = 80
ALPHA_DENSITY = 0.01
MIN_MAX_COMPONENT_POSTERIOR = 0.8
RECONSTRUCTION_RESIDUAL_QUANTILE = 0.99
OOD_LABEL = -1


class CrossCohortAssignmentError(ValueError):
    """A source-free failure for an invalid transport assignment request."""


def _fail() -> None:
    raise CrossCohortAssignmentError(ERROR) from None


def _sha256(value: Any) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _fail()
    return value


@dataclass(frozen=True, repr=False)
class AssignmentBinding:
    """Exact identifiers required for calibration and every target call.

    Hash equality binds declared policy material.  It cannot authenticate the
    physical no-retina route or participant-disjoint source roles; a lifecycle
    must establish those facts before this primitive is called.
    """

    model_frame_sha256: str
    canonical_schema_units_sha256: str
    normalization_sha256: str
    clinical_only_input_policy_sha256: str

    def __post_init__(self) -> None:
        for value in (
            self.model_frame_sha256,
            self.canonical_schema_units_sha256,
            self.normalization_sha256,
            self.clinical_only_input_policy_sha256,
        ):
            _sha256(value)


def _binding(value: Any) -> AssignmentBinding:
    if type(value) is not AssignmentBinding:
        _fail()
    # Recheck so a malicious object.__setattr__ mutation cannot bypass the
    # construction-time validation of a frozen dataclass.
    AssignmentBinding(
        value.model_frame_sha256,
        value.canonical_schema_units_sha256,
        value.normalization_sha256,
        value.clinical_only_input_policy_sha256,
    )
    return value


def _states(value: Any, *, minimum_rows: int) -> np.ndarray:
    if type(value) is not np.ndarray:
        _fail()
    array = np.asarray(value)
    if (array.ndim != 2 or array.shape[1:] != (STATE_WIDTH,) or array.shape[0] < minimum_rows
            or array.dtype.kind not in "fiu" or array.dtype.kind == "b"):
        _fail()
    try:
        result = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        _fail()
    if not bool(np.isfinite(result).all()):
        _fail()
    return result.copy()


def _digest(*objects: Any) -> str:
    try:
        return hashlib.sha256(pickle.dumps(objects, protocol=5)).hexdigest()
    except Exception:
        _fail()


def _k(gmm: Any) -> int:
    value = getattr(gmm, "n_components", None)
    if type(value) is not int or value < 2:
        _fail()
    return value


def _project(scaler: Any, pca: Any, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        transformed = np.asarray(scaler.transform(states.copy()), dtype=np.float64)
        reduced = np.asarray(pca.transform(transformed.copy()), dtype=np.float64)
        reconstructed = np.asarray(pca.inverse_transform(reduced.copy()), dtype=np.float64)
        residual = np.linalg.norm(transformed - reconstructed, axis=1)
    except Exception:
        _fail()
    if (transformed.shape != states.shape or not bool(np.isfinite(transformed).all())
            or reduced.ndim != 2 or reduced.shape[0] != len(states) or reduced.shape[1] < 1
            or reconstructed.shape != transformed.shape or residual.shape != (len(states),)
            or not bool(np.isfinite(reduced).all()) or not bool(np.isfinite(residual).all())
            or bool(np.any(residual < 0.0))):
        _fail()
    return reduced, residual


def _scores(gmm: Any, reduced: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        density = np.asarray(gmm.score_samples(reduced.copy()), dtype=np.float64)
        posterior = np.asarray(gmm.predict_proba(reduced.copy()), dtype=np.float64)
    except Exception:
        _fail()
    if (density.shape != (len(reduced),) or posterior.shape != (len(reduced), k)
            or not bool(np.isfinite(density).all()) or not bool(np.isfinite(posterior).all())
            or bool(np.any(posterior < 0.0)) or bool(np.any(posterior > 1.0))
            or not bool(np.allclose(posterior.sum(axis=1), 1.0, rtol=0.0, atol=1e-8))):
        _fail()
    return density, posterior


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, repr=False)
class LockedCrossCohortAssignment:
    """Private frozen rule and source-only technical calibration metadata."""

    _scaler: Any = field(repr=False)
    _pca: Any = field(repr=False)
    _gmm: Any = field(repr=False)
    _binding: AssignmentBinding = field(repr=False)
    _density_floor: float = field(repr=False)
    _reconstruction_residual_ceiling: float = field(repr=False)
    _alpha_density: float = field(repr=False)
    _min_max_component_posterior: float = field(repr=False)
    _reconstruction_residual_quantile: float = field(repr=False)
    _integrity_sha256: str = field(repr=False)


@dataclass(frozen=True, repr=False)
class AssignmentResult:
    """Private technical-diagnostic result, never a public aggregate.

    Density and reconstruction-residual filters are source-calibrated
    technical diagnostics only.  This object contains no stable, validated, or
    novel-subtype claim and no target-domain safety guarantee.
    """

    labels: np.ndarray = field(repr=False)
    accepted: np.ndarray = field(repr=False)
    density_supported: np.ndarray = field(repr=False)
    confidence_supported: np.ndarray = field(repr=False)
    residual_supported: np.ndarray = field(repr=False)
    max_posterior: np.ndarray = field(repr=False)
    gmm_posterior_is_disease_probability: bool = False
    source_density_gate_validates_target_domain_safety: bool = False


def calibrate_locked_assignment(
    scaler: Any,
    pca: Any,
    gmm: Any,
    source_calibration_states: Any,
    *,
    binding: AssignmentBinding,
    source_calibration_participant_disjoint_from_discovery: bool,
) -> LockedCrossCohortAssignment:
    """Freeze an existing rule and obtain its source-only lower density floor.

    No transformer or mixture is fit here.  The caller's explicit attestation
    records claimed participant disjointness; the lifecycle must authenticate
    that claim before calling this in-memory primitive.
    """
    try:
        checked_binding = _binding(binding)
        if source_calibration_participant_disjoint_from_discovery is not True:
            _fail()
        calibration = _states(source_calibration_states, minimum_rows=MIN_CALIBRATION_ROWS)
        copied_scaler, copied_pca, copied_gmm = deepcopy(scaler), deepcopy(pca), deepcopy(gmm)
        k = _k(copied_gmm)
        before = _digest(copied_scaler, copied_pca, copied_gmm, checked_binding)
        reduced, residual = _project(copied_scaler, copied_pca, calibration)
        density, _ = _scores(copied_gmm, reduced, k)
        if _digest(copied_scaler, copied_pca, copied_gmm, checked_binding) != before:
            _fail()
        # "higher" selects the conservative observed order statistic at the
        # lower-tail alpha, rather than relaxing the floor below that sample.
        alpha_density = 0.01
        minimum_posterior = 0.8
        residual_quantile = 0.99
        floor = float(np.quantile(density, alpha_density, method="higher"))
        residual_ceiling = float(np.quantile(residual, residual_quantile, method="higher"))
        if not np.isfinite(floor) or not np.isfinite(residual_ceiling) or residual_ceiling < 0.0:
            _fail()
        integrity = _digest(copied_scaler, copied_pca, copied_gmm, checked_binding, floor, residual_ceiling,
                            alpha_density, minimum_posterior, residual_quantile)
        return LockedCrossCohortAssignment(copied_scaler, copied_pca, copied_gmm, checked_binding, floor,
                                           residual_ceiling, alpha_density, minimum_posterior,
                                           residual_quantile, integrity)
    except CrossCohortAssignmentError:
        raise
    except Exception:
        _fail()


def assign_locked(
    locked: LockedCrossCohortAssignment,
    target_states: Any,
    *,
    binding: AssignmentBinding,
    **forbidden: Any,
) -> AssignmentResult:
    """Apply a frozen rule; target callers cannot refit or alter thresholds."""
    try:
        if forbidden:
            _fail()
        if type(locked) is not LockedCrossCohortAssignment or _binding(binding) != locked._binding:
            _fail()
        if (locked._alpha_density != 0.01 or locked._min_max_component_posterior != 0.8
                or locked._reconstruction_residual_quantile != 0.99):
            _fail()
        if _digest(locked._scaler, locked._pca, locked._gmm, locked._binding, locked._density_floor,
                   locked._reconstruction_residual_ceiling, locked._alpha_density,
                   locked._min_max_component_posterior, locked._reconstruction_residual_quantile) != locked._integrity_sha256:
            _fail()
        target = _states(target_states, minimum_rows=1)
        k = _k(locked._gmm)
        reduced, residual = _project(locked._scaler, locked._pca, target)
        density, posterior = _scores(locked._gmm, reduced, k)
        confidence = posterior.max(axis=1)
        density_ok = density >= locked._density_floor
        confidence_ok = confidence >= locked._min_max_component_posterior
        residual_ok = residual <= locked._reconstruction_residual_ceiling
        accepted = density_ok & confidence_ok & residual_ok
        labels = np.argmax(posterior, axis=1).astype(np.int64, copy=False)
        labels = labels.copy(); labels[~accepted] = OOD_LABEL
        if _digest(locked._scaler, locked._pca, locked._gmm, locked._binding, locked._density_floor,
                   locked._reconstruction_residual_ceiling, locked._alpha_density,
                   locked._min_max_component_posterior, locked._reconstruction_residual_quantile) != locked._integrity_sha256:
            _fail()
        return AssignmentResult(
            _readonly(labels), _readonly(accepted), _readonly(density_ok), _readonly(confidence_ok),
            _readonly(residual_ok), _readonly(confidence.astype(np.float64, copy=False)),
        )
    except CrossCohortAssignmentError:
        raise
    except Exception:
        _fail()
