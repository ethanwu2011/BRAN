"""R7 S4 clinical panel with authenticated, state-independent raw reuse.

The input frame is deliberately the S1 frame: only its 192-coordinate state
may differ.  Raw clinical structure is therefore reused only after the old S1
receipt and private fitted objects replay against the current raw design.  R7
state structure and every fixed outcome readout are newly fitted.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np
from threadpoolctl import threadpool_limits

import bran_mimic_clinical_outcomes_v3 as outcome
import bran_multisource_clinical_panel_s1 as s1
import bran_r1_hf_structure_kernel_v1 as structure


ERROR = "bran_r7_clinical_s4_failed"
# S4 changes only the private state and its binding.  Its released inner
# receipt is intentionally the frozen S1 schema; the runner supplies the
# explicit S4 outer protocol and provenance envelope.
SCHEMA = "bran-multisource-clinical-family-s1"
FAMILIES = s1.FAMILIES
SOURCES = s1.SOURCES
FLAGS = s1.FLAGS
# The only S1 design statistics that are invariant when the state changes.
RAW_STAT_KEYS = (
    "lab_medians", "age_mean", "age_scale", "raw_scaler_mean", "raw_scaler_scale",
)
PrivateResult = s1.PrivateResult


def require(ok: bool) -> None:
    if not ok:
        raise ValueError(ERROR) from None


def check_frame(frame: object) -> None:
    try:
        s1.check_frame(frame)
    except Exception:
        raise ValueError(ERROR) from None


def build(frame: dict[str, np.ndarray], rows: np.ndarray):
    return s1.build(frame, rows)


def _equal(value: object, expected: object) -> bool:
    return np.array_equal(np.asarray(value), np.asarray(expected), equal_nan=True)


def _raw_stats_match(current: dict[str, object], historical: object) -> bool:
    # Presence of state normalization fields is authenticated by the S1
    # receipt, but their values must not be compared across different states.
    return (type(historical) is dict and set(historical) == set(current)
            and all(_equal(current[key], historical[key]) for key in RAW_STAT_KEYS))


def _binding(family: str) -> str:
    return hashlib.sha256(("S4|" + family + "|raw49_source2_pad8_r7state192").encode()).hexdigest()


def _active(fit: object, report: dict[str, object]) -> bool:
    return bool(getattr(fit, "status", None) == "supported" and report.get("stability_gate") is True)


def _component_result(component: object) -> dict[str, object]:
    require(type(component) is dict and set(component) == {
        "result", "objects_sha256", "object_replay_exact", "patient_level_output_emitted"}
            and type(component["objects_sha256"]) is str and len(component["objects_sha256"]) == 64
            and component["object_replay_exact"] is True and component["patient_level_output_emitted"] is False
            and type(component["result"]) is dict)
    return component["result"]


def _reference(frame: dict[str, np.ndarray], rows: np.ndarray, family: str,
               component: object, objects: object):
    """Authenticate raw-only S1 reuse without inspecting R7 state equivalence."""
    component = _component_result(component)
    try:
        s1.validate_report(component)
    except Exception:
        raise ValueError(ERROR) from None
    require(component.get("status") == "evaluated" and component.get("family") == family)
    require(type(objects) is dict and set(objects) == {"rows", "fits", "outcome", "binding", "design_stats"})
    old_rows = objects["rows"]
    require(type(old_rows) is np.ndarray and old_rows.dtype == np.dtype(np.int64)
            and np.array_equal(old_rows, rows))
    require(type(objects["fits"]) is dict and set(objects["fits"]) == {"bran", "raw"})
    raw_fit = objects["fits"]["raw"]
    raw_report = component["branches"]["raw"]
    require(type(raw_report) is dict and structure.validate_aggregate(raw_report.get("structure")))
    require(getattr(raw_fit, "aggregate", None) == raw_report["structure"])
    require(raw_report.get("group_arms_admitted") is _active(raw_fit, raw_report["structure"]))
    expected_binding = hashlib.sha256(("S1|" + family + "|raw49_source2_pad8_state192").encode()).hexdigest()
    require(objects["binding"] == expected_binding)
    x = build(frame, rows)
    require(_raw_stats_match(x.stats, objects["design_stats"]))
    return x, raw_fit, deepcopy(raw_report)


def _unsupported_reference(component: object, objects: object, family: str) -> None:
    """Short cohorts still require an authenticated matching S1 closure."""
    result = _component_result(component)
    try:
        s1.validate_report(result)
    except Exception:
        raise ValueError(ERROR) from None
    require(result.get("family") == family and result.get("status") == "unsupported_cohort_roles"
            and type(objects) is dict and not objects)


def _replay_raw(frame: dict[str, np.ndarray], rows: np.ndarray, x, raw_fit: object,
                raw_report: dict[str, object], historical: dict[str, object]) -> np.ndarray | None:
    """Replay only raw-invariant labels and readout arms from the S1 receipt."""
    active = _active(raw_fit, raw_report["structure"])
    labels = raw_fit.predict(x.raw_padded192) if active else None
    if active:
        require(type(labels) is np.ndarray and labels.dtype == np.dtype(np.int64)
                and labels.shape == (len(rows),))
        source, roles = frame["source"][rows], frame["roles"][rows]
        profiles = {SOURCES[index]: s1.previous.profiles(
            frame["values"][rows][(roles == 2) & (source == index)],
            frame["observed"][rows][(roles == 2) & (source == index)],
            labels[(roles == 2) & (source == index)],
            frame["outcome"][rows][(roles == 2) & (source == index)], raw_fit.selected_k,
        ) for index in range(2)}
        require(profiles == raw_report["source_test_profiles"])
        require(s1.nuisance_check(labels, raw_fit.selected_k, x.nuisance_design, roles, source)
                == raw_report["nuisance"])
    else:
        require(raw_report["source_test_profiles"] == {name: {"status": "not_run_structure_gate_failed"} for name in SOURCES}
                and raw_report["nuisance"] == {"status": "not_run_structure_gate_failed"})

    sink = historical["outcome"]
    require(type(sink) is dict and sink.get("schema") == outcome.PRIVATE_SCHEMA
            and sink.get("design_binding") == historical["binding"] and type(sink.get("arms")) is dict)
    known_test = (frame["roles"][rows] == 2) & (frame["outcome"][rows] >= 0)
    designs = {"context": x.context59[known_test]}
    if labels is not None:
        designs["context_raw_groups"] = np.column_stack((
            x.context59[known_test], outcome._one_hot(labels[known_test], raw_fit.selected_k)))
    # ``context_state`` and every BRAN arm intentionally stay out: S1 used a
    # different state, so their stored V5 predictions are not comparable.
    for arm in ("context", "context_raw_groups"):
        if arm not in sink["arms"]:
            continue
        saved = sink["arms"][arm]
        require(designs[arm].shape[1] == saved["feature_width"])
        probability = outcome._calibrated_probabilities(
            saved["model"].decision_function(designs[arm]), saved["calibration_offset"])
        require(np.array_equal(probability, saved["test_predictions"]))
    return labels


def _branch(frame: dict[str, np.ndarray], rows: np.ndarray, x, labels: np.ndarray | None,
            fit: object) -> dict[str, object]:
    report = getattr(fit, "aggregate", None)
    require(type(report) is dict and structure.validate_aggregate(report))
    active = _active(fit, report)
    result = {"structure": report, "group_arms_admitted": active}
    source, roles = frame["source"][rows], frame["roles"][rows]
    if active:
        require(labels is not None and labels.shape == (len(rows),))
        result["source_test_profiles"] = {SOURCES[index]: s1.previous.profiles(
            frame["values"][rows][(roles == 2) & (source == index)],
            frame["observed"][rows][(roles == 2) & (source == index)],
            labels[(roles == 2) & (source == index)],
            frame["outcome"][rows][(roles == 2) & (source == index)], fit.selected_k,
        ) for index in range(2)}
        result["nuisance"] = s1.nuisance_check(labels, fit.selected_k, x.nuisance_design, roles, source)
    else:
        result["source_test_profiles"] = {name: {"status": "not_run_structure_gate_failed"} for name in SOURCES}
        result["nuisance"] = {"status": "not_run_structure_gate_failed"}
    return result


def _fit_family(frame: dict[str, np.ndarray], family: str, reference_component: object,
                reference_objects: object, progress):
    check_frame(frame)
    require(family in FAMILIES)
    rows = np.flatnonzero(frame["membership"][:, FAMILIES.index(family)])
    roles = frame["roles"][rows]
    base = {"schema": SCHEMA, "family": family, "support": s1.support(frame, rows),
            "source_names": list(SOURCES), "outcomes_used_for_group_fitting": False, **FLAGS}
    if any(np.sum(roles == role) < minimum for role, minimum in enumerate((80, 40, 40))):
        _unsupported_reference(reference_component, reference_objects, family)
        return PrivateResult({**base, "status": "unsupported_cohort_roles"}, {})
    x, raw_fit, raw_report = _reference(frame, rows, family, reference_component, reference_objects)
    raw_labels = _replay_raw(frame, rows, x, raw_fit, raw_report, reference_objects)
    if progress:
        progress("bran_structure")
    bran_fit = structure.fit_structure(*(frame["state"][rows][roles == role] for role in range(3)))
    bran_labels = (bran_fit.predict(frame["state"][rows])
                   if _active(bran_fit, bran_fit.aggregate) else None)
    bran_report = _branch(frame, rows, x, bran_labels, bran_fit)
    # The raw report is copied only after the checks above have replayed it.
    branches = {"bran": bran_report, "raw": raw_report}
    if progress:
        progress("outcome_utility")
    sink: dict[str, object] = {}
    binding = _binding(family)
    utility = outcome.evaluate(
        x.context59, x.state_scaled, frame["person_group"][rows], roles, frame["outcome"][rows],
        bran_groups=bran_labels,
        bran_k=(bran_fit.selected_k if bran_report["group_arms_admitted"] else None),
        raw_groups=raw_labels, raw_k=(raw_fit.selected_k if raw_labels is not None else None),
        private_sink=sink, design_binding=binding,
    )
    keys = ("bran_groups_minus_context", "bran_groups_minus_raw_groups", "state_bran_groups_minus_state")
    incremental = branches["bran"]["group_arms_admitted"] and all(
        utility.get("contrasts", {}).get(key, {}).get("utility_gate") is True for key in keys)
    aggregate = {**base, "status": "evaluated", "branches": branches, "outcome_utility": utility,
                 "three_adjusted_increment_checks_passed": bool(incremental),
                 "interpretation": "internal_two_source_candidate_heterogeneity_not_validated_subtypes"}
    s1.validate_report(aggregate)
    return PrivateResult(aggregate, {"rows": rows, "fits": {"bran": bran_fit, "raw": raw_fit},
                                     "outcome": sink, "binding": binding, "design_stats": x.stats})


def fit_family(frame: dict[str, np.ndarray], family: str, reference_component: object,
               reference_objects: object, progress=None) -> PrivateResult:
    try:
        with threadpool_limits(limits=2):
            return _fit_family(frame, family, reference_component, reference_objects, progress)
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


validate_report = s1.validate_report
replay = s1.replay


__all__ = ["ERROR", "FAMILIES", "FLAGS", "PrivateResult", "RAW_STAT_KEYS", "SCHEMA", "SOURCES",
           "build", "check_frame", "fit_family", "replay", "validate_report"]
