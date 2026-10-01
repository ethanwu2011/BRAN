"""Local fixed-encoder disease-structure evaluation without outcome analysis.

The only supervised-looking input is a source-defined boolean T2D membership
flag.  It is applied after unsupervised inference solely to select state rows
for discovery, validation, and replication structure diagnostics.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

import bran_disease_structure_192_v1 as structure
import bran_disease_structure_diagnostics_v1 as diagnostics


STATE_DIMENSION = 192
FIT_FOLDS = (0, 1, 2)
VALIDATION_FOLD = 3
REPLICATION_FOLD = 4
ENCODER_STEPS = 1500
ENCODER_SEED = 1701


class DiseaseStructureEvaluationError(ValueError):
    """Raised before a local structure evaluation can violate split contracts."""


def _require(ok: bool) -> None:
    if not ok:
        raise DiseaseStructureEvaluationError("disease_structure_contract_failed")


def _inference(model, clinical, clinical_mask, retina, retina_mask, age) -> np.ndarray:
    """Direct 192-coordinate encoder inference; age is never appended."""
    import torch
    with torch.no_grad():
        state = model.encode(
            torch.tensor(clinical, dtype=torch.float32), torch.tensor(clinical_mask),
            torch.tensor(retina[:, None], dtype=torch.float32), torch.tensor(retina_mask[:, None]),
            torch.tensor(age, dtype=torch.float32),
        ).mean.numpy()
    _require(state.shape == (len(clinical), STATE_DIMENSION) and np.isfinite(state).all())
    return state


def _closed_structure(result) -> dict:
    aggregate = structure.aggregate_diagnostics(result)
    if aggregate is not None:
        return aggregate
    _require(isinstance(result, structure.StructureResult) and isinstance(result.status, str))
    return {"status": result.status}


def _closed_stability(result) -> dict:
    aggregate = diagnostics.aggregate_report(result)
    if aggregate is not None:
        return aggregate
    _require(isinstance(result, diagnostics.StabilityResult) and isinstance(result.status, str))
    return {"status": result.status}


def evaluate(
    c0: np.ndarray,
    cm0: np.ndarray,
    eligible: np.ndarray,
    r0: np.ndarray,
    rm: np.ndarray,
    names: tuple[str, ...] | list[str],
    ages: np.ndarray,
    outer: np.ndarray,
    patient_ids: tuple[str, ...] | list[str],
    t2d_membership: np.ndarray,
    *,
    steps: int = ENCODER_STEPS,
    progress: Callable[[str], None] | None = None,
    trainer=None,
    infer=None,
    structure_fitter=None,
    diagnostic=None,
) -> dict:
    """Fit one discovery-only encoder then emit count-free structure aggregates.

    Hooks are for synthetic tests only.  Default fitting is the unchanged V2
    encoder schedule, and no outcome arrays, CGM, ECG, or manual K selection
    enter this function.
    """
    import run_bran_anchor_ablation_v2 as paired
    from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
    from run_bran_frozen_cbc_readout_v1 import _encoder_hash

    trainer = paired._train if trainer is None else trainer
    infer = _inference if infer is None else infer
    structure_fitter = structure.fit_structure if structure_fitter is None else structure_fitter
    diagnostic = diagnostics.diagnose if diagnostic is None else diagnostic
    c0, cm0, r0, rm, ages, outer = [np.asarray(value) for value in (c0, cm0, r0, rm, ages, outer)]
    eligibility = np.asarray(eligible)
    n = len(c0)
    _require(c0.shape == cm0.shape == eligibility.shape == (n, 59) and r0.shape == (n, 384))
    _require(cm0.dtype == np.dtype(bool) and eligibility.dtype == np.dtype(bool) and rm.dtype == np.dtype(bool))
    _require(rm.shape == ages.shape == outer.shape == (n,) and outer.dtype.kind in "iu" and set(outer.tolist()) == set(range(5)))
    _require(len(names) == len(set(names)) == 59 and all(type(name) is str and name for name in names))
    ids = tuple(patient_ids)
    _require(len(ids) == len(set(ids)) == n and all(type(value) is str and value for value in ids))
    membership = np.asarray(t2d_membership)
    _require(membership.dtype == np.dtype(bool) and membership.shape == (n,))
    _require(np.isfinite(c0[cm0]).all() and np.isfinite(r0[rm[:, None].repeat(r0.shape[1], axis=1)]).all() and np.isfinite(ages).all())
    discovery_rows = np.flatnonzero(np.isin(outer, FIT_FOLDS))
    _require(len(discovery_rows) >= 10 and len(np.flatnonzero(outer == VALIDATION_FOLD)) >= 10 and len(np.flatnonzero(outer == REPLICATION_FOLD)) >= 10)
    # Work on a local eligibility copy; all diagnosis-history columns are off.
    local_eligibility = eligibility.copy()
    local_eligibility[:, 48:] = False
    _require(not local_eligibility[:, 48:].any())
    transform = paired.base.FoldTransform(c0, cm0, local_eligibility, r0, rm, ages, discovery_rows)
    clinical, clinical_mask, retina, age = transform.apply(c0, cm0, local_eligibility, r0, rm, ages)
    if progress:
        progress("training")
    model = trainer(BRANClinicalAnchorV2, clinical, clinical_mask, retina, rm, age, discovery_rows, ENCODER_SEED, steps=steps)
    model.eval()
    before = _encoder_hash(model)
    states = np.asarray(infer(model, clinical, clinical_mask, retina, rm, age), dtype=np.float64)
    _require(states.shape == (n, STATE_DIMENSION) and np.isfinite(states).all())
    # Membership is intentionally not passed to transforms, encoder, or trainer.
    discovery = states[(outer < 3) & membership]
    validation = states[(outer == VALIDATION_FOLD) & membership]
    replication = states[(outer == REPLICATION_FOLD) & membership]
    if progress:
        progress("structure")
    fitted = structure_fitter(discovery, validation)
    # The diagnostic receives the already selected locked model, so replication
    # cannot trigger another candidate/K search.
    stability = diagnostic(discovery, validation, replication, locked_model=fitted)
    _require(_encoder_hash(model) == before)
    result = {
        "structure": _closed_structure(fitted),
        "diagnostics": _closed_stability(stability),
        "encoder_fits": 1,
        "state_dimension": STATE_DIMENSION,
        "encoder_unchanged_after_diagnostics": True,
        "patient_arrays_serialized": False,
        "default_promotion": False,
        "novel_subtype_claimed": False,
        "clinical_efficacy_claimed": False,
    }
    validate_export(result)
    return result


def validate_export(value: object) -> None:
    """Ensure the evaluator emits closed aggregates and no row-level payloads."""
    _require(isinstance(value, dict) and set(value) == {
        "structure", "diagnostics", "encoder_fits", "state_dimension", "encoder_unchanged_after_diagnostics",
        "patient_arrays_serialized", "default_promotion", "novel_subtype_claimed", "clinical_efficacy_claimed",
    })
    _require(type(value["encoder_fits"]) is int and value["encoder_fits"] == 1 and type(value["state_dimension"]) is int and value["state_dimension"] == STATE_DIMENSION)
    for key in ("encoder_unchanged_after_diagnostics", "patient_arrays_serialized", "default_promotion", "novel_subtype_claimed", "clinical_efficacy_claimed"):
        _require(type(value[key]) is bool)
    _require(value["encoder_unchanged_after_diagnostics"] and not value["patient_arrays_serialized"] and not value["default_promotion"] and not value["novel_subtype_claimed"] and not value["clinical_efficacy_claimed"])
    _validate_structure_and_diagnostics(value["structure"], value["diagnostics"])
    _require(not _contains_array(value))


def validate_result(value: object) -> bool:
    """Boolean driver hook that swallows only this module's fixed contract error."""
    try:
        validate_export(value)
    except DiseaseStructureEvaluationError:
        return False
    return True


def _validate_structure_and_diagnostics(structure_report: object, diagnostic_report: object) -> None:
    _require(isinstance(structure_report, dict) and isinstance(diagnostic_report, dict))
    structure_supported = structure.validate_aggregate_diagnostics(structure_report)
    allowed_structure_unsupported = {
        structure.STATUS_UNSUPPORTED_INSUFFICIENT_ROWS,
        structure.STATUS_UNSUPPORTED_NOT_CONVERGED,
        structure.STATUS_UNSUPPORTED_GROUP_SUPPORT,
    }
    if not structure_supported:
        _require(set(structure_report) == {"status"} and structure_report.get("status") in allowed_structure_unsupported)
    diagnostic_supported = diagnostics.validate_report(diagnostic_report)
    if not diagnostic_supported:
        _require(diagnostic_report == {"status": diagnostics.STATUS_UNSUPPORTED})
    if not structure_supported:
        _require(not diagnostic_supported or diagnostic_report.get("status") not in {diagnostics.STATUS_SUPPORTED, diagnostics.STATUS_NO_DISCRETE_GROUPS, "unsupported_replication_support"})
        return
    selected_k = structure_report["selected_k"]
    if not diagnostic_supported:
        return
    status = diagnostic_report["status"]
    if status == diagnostics.STATUS_SUPPORTED:
        _require(diagnostic_report["selected_k"] == selected_k)
    elif status == diagnostics.STATUS_NO_DISCRETE_GROUPS:
        _require(selected_k == 1 and diagnostic_report == {"status": diagnostics.STATUS_NO_DISCRETE_GROUPS, "selected_k": 1})
    elif status == "unsupported_replication_support":
        _require(selected_k == diagnostic_report["selected_k"])
    else:
        _require(False)


def _contains_array(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, dict):
        return any(_contains_array(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_array(item) for item in value)
    return False
