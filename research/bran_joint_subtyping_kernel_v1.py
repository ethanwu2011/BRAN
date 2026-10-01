"""Array-only, aggregate-only evaluation of pre-specified joint subtyping cohorts."""
from __future__ import annotations

import numpy as np

import bran_disease_structure_192_v1 as structure
import bran_disease_structure_diagnostics_v1 as diagnostics
import bran_disease_structure_evaluation_v1 as old
import bran_fixed_subgroup_evaluation_v1 as fixed
import bran_other_disease_panel_evaluation_v1 as panel
import bran_multigroup_utility_kernel_v1 as utility


CODES = (*panel.CODES, "mhterm_dm2")
_INVALID = "joint subtyping kernel inputs invalid"
_CONTRACT = "joint subtyping kernel result invalid"


def _require(value: bool, message: str = _INVALID) -> None:
    if not value:
        raise ValueError(message)


def _support(member, folds):
    return {
        "discovery_at_least_80": bool(np.sum(member & (folds < 3)) >= 80),
        "validation_at_least_40": bool(np.sum(member & (folds == 3)) >= 40),
        "replication_at_least_40": bool(np.sum(member & (folds == 4)) >= 40),
    }


def _profile_values(clinical, mask, names, profile_age, cgm, cgm_observed):
    """Profiles use original-unit clinical inputs, never normalized designs."""
    values = {"age_years": profile_age, "manifest_average_cgm_glucose_mg_dl": cgm}
    masks = {"age_years": np.ones(len(clinical), dtype=bool),
             "manifest_average_cgm_glucose_mg_dl": cgm_observed}
    for field in panel.FIELDS[1:-1]:
        _require(field in names)
        index = names.index(field)
        _require(index < 48)
        values[field] = clinical[:, index]
        masks[field] = mask[:, index]
    return values, masks


def _known_condition_controls(labels, observed):
    clean = np.zeros_like(labels, dtype=float)
    clean[observed] = labels[observed]
    return np.column_stack((clean, observed.astype(float)))


def _validate_inputs(states, clinical, clinical_mask, retinal_mask, normalized_age, names, site_ids,
                     outer_folds, memberships, labels, label_observed, cgm, cgm_observed,
                     normalized_clinical, profile_age):
    arrays = [np.asarray(value) for value in (states, clinical, clinical_mask, retinal_mask,
              normalized_age, outer_folds, labels, label_observed, cgm, cgm_observed,
              normalized_clinical, profile_age)]
    (states, clinical, clinical_mask, retinal_mask, normalized_age, outer, labels, label_observed,
     cgm, cgm_observed, normalized_clinical, profile_age) = arrays
    n = len(states)
    _require(states.shape == (n, 192) and states.dtype.kind in "fiu" and np.isfinite(states).all())
    _require(clinical.shape == normalized_clinical.shape == clinical_mask.shape == (n, 59))
    _require(clinical_mask.dtype == np.dtype(bool) and retinal_mask.shape == normalized_age.shape == outer.shape == cgm.shape == cgm_observed.shape == profile_age.shape == (n,))
    _require(retinal_mask.dtype == cgm_observed.dtype == np.dtype(bool) and outer.dtype.kind in "iu" and set(outer.tolist()) == set(range(5)))
    _require(np.isfinite(clinical[clinical_mask]).all() and np.isfinite(normalized_clinical[clinical_mask]).all() and np.isfinite(normalized_age).all() and np.isfinite(profile_age).all())
    _require(labels.shape == label_observed.shape == (n, 26) and label_observed.dtype == np.dtype(bool))
    _require(np.all(~label_observed | (np.isfinite(labels) & ((labels == 0) | (labels == 1)))))
    _require(np.isfinite(cgm[cgm_observed]).all() and np.all(cgm[cgm_observed] > 0))
    names = tuple(names); sites = tuple(site_ids)
    _require(len(names) == len(set(names)) == 59 and all(type(name) is str and name for name in names))
    _require(len(sites) == n and all(type(site) is str and site for site in sites))
    _require(isinstance(memberships, dict) and set(memberships) == set(CODES))
    membership = {code: np.asarray(memberships[code]) for code in CODES}
    _require(all(value.shape == (n,) and value.dtype == np.dtype(bool) for value in membership.values()))
    return states.astype(float, copy=False), clinical, clinical_mask, retinal_mask, normalized_age, names, sites, outer, membership, labels, label_observed, cgm, cgm_observed, normalized_clinical, profile_age


def evaluate(states, clinical59, clinical_mask, retinal_mask, normalized_age, names, site_ids,
             outer_folds, memberships, clinical_labels, label_observed, cgmmean, cgm_observed,
             *, normalized_clinical, profile_age, structure_fitter=None, diagnostic=None,
             utility_evaluator=None):
    """Evaluate fixed cohorts without exporting assignments, states, IDs, or counts.

    ``clinical59`` and ``profile_age`` must be original-unit values for the
    descriptive profiles; ``normalized_clinical`` and ``normalized_age`` are
    used only for fixed utility designs.
    """
    (states, clinical, clinical_mask, retinal_mask, normalized_age, names, sites, outer, membership,
     labels, label_observed, cgm, cgm_observed, normalized_clinical, profile_age) = _validate_inputs(
        states, clinical59, clinical_mask, retinal_mask, normalized_age, names, site_ids, outer_folds,
        memberships, clinical_labels, label_observed, cgmmean, cgm_observed, normalized_clinical, profile_age
    )
    fitter = structure.fit_structure if structure_fitter is None else structure_fitter
    diagnose = diagnostics.diagnose if diagnostic is None else diagnostic
    run_utility = utility.fit_and_evaluate if utility_evaluator is None else utility_evaluator
    values, masks = _profile_values(clinical, clinical_mask, names, profile_age, cgm, cgm_observed)
    design_mask = clinical_mask.copy(); design_mask[:, 48:] = False
    result = {
        "schema_version": "bran_joint_subtyping_kernel_v1", "candidate_codes": list(CODES),
        "state_dimension": 192, "supervised_representation": True, "novel_subtype_claimed": False,
        "clinical_utility_established": False, "patient_arrays_serialized": False, "candidates": {},
    }
    for code in CODES:
        member = membership[code]; support = _support(member, outer)
        item = {"support": support, "status": "unsupported_cohort_support"}
        if all(support.values()):
            discovery, validation, replication = (states[member & (outer < 3)], states[member & (outer == 3)], states[member & (outer == 4)])
            locked = fitter(discovery, validation)
            stability = diagnose(discovery, validation, replication, locked_model=locked)
            structure_report, diagnostic_report = old._closed_structure(locked), old._closed_stability(stability)
            old._validate_structure_and_diagnostics(structure_report, diagnostic_report)
            item = {"support": support, "status": "evaluated", "structure": structure_report,
                    "diagnostics": diagnostic_report, "profiles": {"status": "not_applicable_no_supported_groups"},
                    "cgm_utility": {"status": "not_applicable_no_supported_groups"}}
            k = structure_report.get("selected_k")
            if diagnostic_report.get("status") == diagnostics.STATUS_SUPPORTED and type(k) is int and k > 1:
                groups = locked.predict(states)
                profiles = panel.profiles(locked, states, member, outer, values, masks)
                fit = np.flatnonzero(member & (outer < 3))
                base_design = fixed.make_designs(normalized_clinical, design_mask, retinal_mask, normalized_age, names, sites, fit)["combined"]
                severity = np.column_stack((base_design, _known_condition_controls(labels, label_observed)))
                group_onehot = np.column_stack([(groups == group).astype(float) for group in range(1, k)])
                report = run_utility({"severity": severity, "severity_group": np.column_stack((severity, group_onehot))}, cgm, cgm_observed, outer, member)
                utility.validate_report(report)
                item.update(profiles=profiles, cgm_utility=report)
        result["candidates"][code] = item
    validate_result(result)
    return result


def validate_result(result):
    _require(isinstance(result, dict) and set(result) == {"schema_version", "candidate_codes", "state_dimension", "supervised_representation", "novel_subtype_claimed", "clinical_utility_established", "patient_arrays_serialized", "candidates"}, _CONTRACT)
    _require(result["schema_version"] == "bran_joint_subtyping_kernel_v1" and result["candidate_codes"] == list(CODES) and result["state_dimension"] == 192, _CONTRACT)
    _require(result["supervised_representation"] is True and all(result[key] is False for key in ("novel_subtype_claimed", "clinical_utility_established", "patient_arrays_serialized")), _CONTRACT)
    _require(isinstance(result["candidates"], dict) and set(result["candidates"]) == set(CODES) and not old._contains_array(result), _CONTRACT)
    for item in result["candidates"].values():
        _require(isinstance(item, dict) and set(item) >= {"support", "status"} and set(item["support"]) == {"discovery_at_least_80", "validation_at_least_40", "replication_at_least_40"} and all(type(flag) is bool for flag in item["support"].values()), _CONTRACT)
        if item["status"] == "unsupported_cohort_support":
            _require(set(item) == {"support", "status"} and not all(item["support"].values()), _CONTRACT); continue
        _require(set(item) == {"support", "status", "structure", "diagnostics", "profiles", "cgm_utility"} and item["status"] == "evaluated" and all(item["support"].values()), _CONTRACT)
        old._validate_structure_and_diagnostics(item["structure"], item["diagnostics"])
        profile = item["profiles"]
        selected_k = item["structure"].get("selected_k")
        released = item["diagnostics"].get("status") == diagnostics.STATUS_SUPPORTED and type(selected_k) is int and 2 <= selected_k <= 4
        if profile == {"status": "not_applicable_no_supported_groups"}:
            _require(not released and item["cgm_utility"] == {"status": "not_applicable_no_supported_groups"}, _CONTRACT)
        else:
            _require(released, _CONTRACT)
            panel._validate_profiles(profile, item["diagnostics"])
            utility.validate_report(item["cgm_utility"])
