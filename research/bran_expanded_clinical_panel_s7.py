"""Source-free R7 S7 panel for eight prospectively attached disease families."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib

import numpy as np
from threadpoolctl import threadpool_limits

import bran_mimic_clinical_outcomes_v3 as outcome
import bran_mimic_clinical_panel_v3 as previous
import bran_multisource_clinical_design_s1 as design
import bran_multisource_clinical_panel_s1 as s1_panel
import bran_r1_hf_structure_kernel_v1 as structure


ERROR = "bran_expanded_clinical_panel_s7_failed"
FRAME = "R7_attempt1_fold0"
FAMILIES = (
    "recorded_hypertension", "recorded_obesity", "recorded_ischemic_heart_disease",
    "recorded_atrial_fibrillation_flutter", "recorded_copd", "recorded_asthma",
    "recorded_rheumatoid_arthritis", "recorded_osteoarthritis",
)
SOURCES = ("mimic_iv_hospital_day1", "eicu_icu_day1")
FAMILYWISE_COMPARISONS = len(FAMILIES) * len(outcome.CONTRASTS)
OVERLAY_SCHEMA = "bran-expanded-clinical-s7-family40-uncertainty"
FLAGS = {"patient_level_output_emitted": False, "encoder_fitted": False,
         "novel_subtype_claim": False, "clinical_utility_established": False,
         "external_validation_established": False, "outcomes_used_for_group_fitting": False}


def _fail() -> None:
    raise ValueError(ERROR) from None


def require(value: bool) -> None:
    if not value:
        _fail()


def check_frame(frame: object) -> None:
    try:
        require(type(frame) is dict)
        n = len(frame["state"])
        spec = {"state": ((n, 192), np.float32), "values": ((n, 21), np.float64), "observed": ((n, 21), bool),
                "age_value": ((n,), np.float64), "age_lower": ((n,), np.float64), "age_upper": ((n,), np.float64),
                "age_kind": ((n,), np.int64), "source": ((n,), np.uint8), "roles": ((n,), np.uint8),
                "membership": ((n, len(FAMILIES)), bool), "outcome": ((n,), np.int8), "person_group": ((n,), np.int64)}
        require(set(frame) == set(spec))
        for name, (shape, dtype) in spec.items():
            value = frame[name]
            require(type(value) is np.ndarray and value.shape == shape and value.dtype == np.dtype(dtype))
        require(n > 0 and np.isfinite(frame["state"]).all() and frame["observed"].any(1).all()
                and np.isfinite(frame["values"][frame["observed"]]).all()
                and np.isin(frame["source"], (0, 1)).all() and np.isin(frame["roles"], (0, 1, 2)).all()
                and np.isin(frame["outcome"], (-1, 0, 1)).all() and np.unique(frame["person_group"]).size == n
                and (frame["person_group"] >= 0).all())
    except Exception:
        _fail()


def _binding(family: str) -> str:
    require(family in FAMILIES)
    return hashlib.sha256(("S7|R7_fold0|" + family + "|raw49_source2_pad8_state192").encode()).hexdigest()


def _support(frame: dict, family: str) -> dict:
    """Release disease-member cells only when their complements are also safe.

    The original source pools are already public to the S7 lifecycle.  A small
    non-member complement would therefore disclose a small member cell by
    subtraction, even though the member table itself is rounded.  Keep the
    analysis eligibility rule separate from this display-only disclosure gate.
    """
    member = frame["membership"][:, FAMILIES.index(family)]
    cells = [[int(np.sum(member & (frame["source"] == source) & (frame["roles"] == role)))
              for role in range(3)] for source in range(2)]
    complements = [[int(np.sum(~member & (frame["source"] == source) & (frame["roles"] == role)))
                    for role in range(3)] for source in range(2)]
    if min(value for table in (cells, complements) for row in table for value in row) < 20:
        return {"status": "suppressed_source_role_support"}
    return {"status": "released", "source_role_people_lower_bounds_20":
            [[value // 20 * 20 for value in row] for row in cells]}


def _overlay_closed(legacy: dict) -> dict:
    return {"schema": OVERLAY_SCHEMA, "status": legacy["status"], "multiplicity_comparisons": FAMILYWISE_COMPARISONS,
            "requested_draws": None, "accepted_auroc_draws": None,
            "contrasts": {name: {"status": "unavailable"} for name in outcome.CONTRASTS}}


def _interval(values: np.ndarray) -> list[float]:
    tail = float(outcome.POLICY["familywise_alpha"]) / FAMILYWISE_COMPARISONS / 2.0
    value = np.quantile(values, (tail, 1.0 - tail))
    return [float(value[0]), float(value[1])]


def _labels_for_test(frame: dict, rows: np.ndarray) -> np.ndarray:
    known = frame["outcome"][rows] >= 0
    test = (frame["roles"][rows] == 2) & known
    return frame["outcome"][rows][test].astype(np.float64)


def _replay_legacy_bootstrap(legacy: dict, sink: dict, labels: np.ndarray) -> tuple[dict, np.ndarray] | None:
    """Independently recreate the legacy 15-family result from saved predictions."""
    if legacy["status"] == "unsupported_outcome_support":
        require(set(sink.get("arms", {})) == set())
        return None
    if legacy["status"] == "unsupported_bootstrap":
        # V3 has already fitted and retained its arms when the paired
        # bootstrap fails. Verify that this remains a bootstrap failure rather
        # than turning retained predictions into a successful report.
        arms = sink.get("arms")
        require(type(arms) is dict and len(arms) > 0)
        predictions = {name: item["test_predictions"] for name, item in arms.items()}
        require(outcome._bootstrap(labels, predictions) is None)
        return None
    arms = sink.get("arms")
    require(type(arms) is dict)
    predictions = {name: item["test_predictions"] for name, item in arms.items()}
    bootstrap = outcome._bootstrap(labels, predictions)
    require(bootstrap is not None)
    draws, valid = bootstrap
    rebuilt_arms = {name: (outcome._available_arm(outcome._point_metrics(labels, predictions[name]), draws[name], valid)
                           if name in predictions else {"status": "unavailable"}) for name in outcome.ARM_NAMES}
    rebuilt_contrasts = {name: outcome._contrast_report(plus, minus, rebuilt_arms, draws, valid)
                         for name, (plus, minus) in outcome.CONTRASTS.items()}
    require(legacy["arms"] == rebuilt_arms and legacy["contrasts"] == rebuilt_contrasts
            and legacy["bootstrap"] == {"requested_draws": int(outcome.POLICY["bootstrap_draws"]),
                                         "accepted_auroc_draws": int(valid.sum())})
    return draws, valid


def _overlay(legacy: dict, sink: dict, labels: np.ndarray) -> dict:
    require(outcome.validate_report(legacy))
    replay = _replay_legacy_bootstrap(legacy, sink, labels)
    if replay is None:
        value = _overlay_closed(legacy)
        validate_overlay(value, legacy)
        return value
    draws, valid = replay
    contrasts = {}
    for name, (plus, minus) in outcome.CONTRASTS.items():
        old = legacy["contrasts"][name]
        if old["status"] != "available":
            contrasts[name] = {"status": "unavailable"}
            continue
        auc = draws[plus]["auroc"][valid] - draws[minus]["auroc"][valid]
        losses = {metric: draws[minus][metric][valid] - draws[plus][metric][valid] for metric in ("logloss", "brier")}
        contrasts[name] = {"status": "available", "auroc_difference": old["auroc_difference"],
            "logloss_improvement": old["logloss_improvement"], "brier_improvement": old["brier_improvement"],
            "auroc_adjusted_95ci": _interval(auc),
            "logloss_improvement_adjusted_95ci": _interval(losses["logloss"]),
            "brier_improvement_adjusted_95ci": _interval(losses["brier"]),
            "utility_gate": bool(_interval(losses["logloss"])[0] > 0.0 and old["auroc_difference"] >= 0.0)}
    value = {"schema": OVERLAY_SCHEMA, "status": legacy["status"], "multiplicity_comparisons": FAMILYWISE_COMPARISONS,
             "requested_draws": int(outcome.POLICY["bootstrap_draws"]), "accepted_auroc_draws": int(valid.sum()),
             "contrasts": contrasts}
    validate_overlay(value, legacy)
    return value


def validate_overlay(value: object, legacy: object) -> None:
    try:
        require(type(value) is dict and type(legacy) is dict and outcome.validate_report(legacy)
                and set(value) == {"schema", "status", "multiplicity_comparisons", "requested_draws", "accepted_auroc_draws", "contrasts"}
                and value["schema"] == OVERLAY_SCHEMA and value["status"] == legacy["status"]
                and value["multiplicity_comparisons"] == FAMILYWISE_COMPARISONS
                and type(value["contrasts"]) is dict and set(value["contrasts"]) == set(outcome.CONTRASTS))
        closed = legacy["status"] in {"unsupported_outcome_support", "unsupported_bootstrap"}
        require((value["requested_draws"] is None and value["accepted_auroc_draws"] is None) if closed else
                (value["requested_draws"] == int(outcome.POLICY["bootstrap_draws"])
                 and type(value["accepted_auroc_draws"]) is int
                 and int(outcome.POLICY["minimum_valid_auroc_draws"]) <= value["accepted_auroc_draws"] <= value["requested_draws"]))
        for name, item in value["contrasts"].items():
            old = legacy.get("contrasts", {}).get(name)
            available = old is not None and old["status"] == "available"
            if not available:
                require(item == {"status": "unavailable"})
                continue
            require(set(item) == {"status", "auroc_difference", "logloss_improvement", "brier_improvement",
                                  "auroc_adjusted_95ci", "logloss_improvement_adjusted_95ci",
                                  "brier_improvement_adjusted_95ci", "utility_gate"} and item["status"] == "available"
                    and all(item[key] == old[key] for key in ("auroc_difference", "logloss_improvement", "brier_improvement"))
                    and all(outcome._interval(item[key], -1.0, 1.0) for key in ("auroc_adjusted_95ci", "brier_improvement_adjusted_95ci"))
                    and outcome._interval(item["logloss_improvement_adjusted_95ci"], -100.0, 100.0)
                    and item["utility_gate"] is (item["logloss_improvement_adjusted_95ci"][0] > 0.0 and item["auroc_difference"] >= 0.0))
    except Exception:
        _fail()


@dataclass(repr=False)
class PrivateResult:
    aggregate: dict
    objects: dict
    def __repr__(self) -> str: return "<PrivateExpandedClinicalResult>"
    def __reduce__(self): raise TypeError(ERROR)


def _base(frame: dict, family: str, rows: np.ndarray) -> dict:
    return {"schema": "bran-expanded-clinical-family-s7", "family": family, "frame": FRAME,
            "support": _support(frame, family), "source_names": list(SOURCES),
            "outcomes_used_for_group_fitting": False, **FLAGS}


def fit_family(frame: dict, family: str, progress=None) -> PrivateResult:
    try:
        with threadpool_limits(limits=2):
            check_frame(frame); require(family in FAMILIES)
            rows = np.flatnonzero(frame["membership"][:, FAMILIES.index(family)]).astype(np.int64)
            base = _base(frame, family, rows)
            roles, source = frame["roles"][rows], frame["source"][rows]
            if any(np.sum(roles == role) < minimum for role, minimum in enumerate((80, 40, 40))):
                result = PrivateResult({**base, "status": "unsupported_cohort_roles"}, {})
                validate_report(result.aggregate); return result
            x = design.build(**{key: frame[key][rows] for key in
                                ("values", "observed", "age_value", "age_lower", "age_upper", "age_kind", "state", "roles", "source")})
            fits, groups, branches = {}, {}, {}
            for branch, states in (("bran", frame["state"][rows]), ("raw", x.raw_padded192)):
                if progress: progress(branch + "_structure")
                fit = structure.fit_structure(*(states[roles == role] for role in range(3)))
                report = fit.aggregate; require(structure.validate_aggregate(report))
                active = fit.status == "supported" and report.get("stability_gate") is True
                fits[branch], groups[branch] = fit, fit.predict(states) if active else None
                if active:
                    profiles = {SOURCES[item]: previous.profiles(frame["values"][rows][(roles == 2) & (source == item)],
                        frame["observed"][rows][(roles == 2) & (source == item)], groups[branch][(roles == 2) & (source == item)],
                        frame["outcome"][rows][(roles == 2) & (source == item)], fit.selected_k) for item in range(2)}
                    nuisance = s1_panel.nuisance_check(groups[branch], fit.selected_k, x.nuisance_design, roles, source)
                else:
                    profiles, nuisance = {name: {"status": "not_run_structure_gate_failed"} for name in SOURCES}, {"status": "not_run_structure_gate_failed"}
                branches[branch] = {"structure": report, "group_arms_admitted": active, "source_test_profiles": profiles, "nuisance": nuisance}
            if progress: progress("outcome_utility")
            binding = _binding(family); sink = {}
            legacy = outcome.evaluate(x.context59, x.state_scaled, frame["person_group"][rows], roles, frame["outcome"][rows],
                bran_groups=groups["bran"], bran_k=fits["bran"].selected_k if groups["bran"] is not None else None,
                raw_groups=groups["raw"], raw_k=fits["raw"].selected_k if groups["raw"] is not None else None,
                private_sink=sink, design_binding=binding)
            overlay = _overlay(legacy, sink, _labels_for_test(frame, rows))
            keys = ("bran_groups_minus_context", "bran_groups_minus_raw_groups", "state_bran_groups_minus_state")
            success = branches["bran"]["group_arms_admitted"] and all(overlay["contrasts"][key].get("utility_gate") is True for key in keys)
            aggregate = {**base, "status": "evaluated", "branches": branches, "legacy_recipe": legacy,
                         "family40_uncertainty": overlay, "three_family40_increment_checks_passed": bool(success),
                         "interpretation": "internal_two_source_candidate_heterogeneity_not_validated_subtypes"}
            validate_report(aggregate)
            return PrivateResult(aggregate, {"family": family, "frame": FRAME, "rows": rows, "fits": fits, "outcome": sink,
                                               "binding": binding, "design_stats": x.stats,
                                               "known_rows": rows[frame["outcome"][rows] >= 0].copy(), "legacy_recipe": copy.deepcopy(legacy),
                                               "family40_uncertainty": copy.deepcopy(overlay)})
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail()


def validate_report(value: object) -> None:
    try:
        base = {"schema", "family", "frame", "support", "source_names", "outcomes_used_for_group_fitting", "status", *FLAGS}
        require(type(value) is dict and value.get("schema") == "bran-expanded-clinical-family-s7"
                and value.get("family") in FAMILIES and value.get("frame") == FRAME and value.get("source_names") == list(SOURCES)
                and value.get("outcomes_used_for_group_fitting") is False and all(value.get(key) is flag for key, flag in FLAGS.items()))
        support = value["support"]
        if support.get("status") == "suppressed_source_role_support": require(support == {"status": "suppressed_source_role_support"})
        else:
            cells = support.get("source_role_people_lower_bounds_20")
            require(support.get("status") == "released" and set(support) == {"status", "source_role_people_lower_bounds_20"}
                    and type(cells) is list and len(cells) == 2
                    and all(type(row) is list and len(row) == 3
                            and all(type(item) is int and item >= 20 and item % 20 == 0 for item in row) for row in cells))
        if value["status"] == "unsupported_cohort_roles": require(set(value) == base); return
        require(value["status"] == "evaluated" and set(value) == base | {"branches", "legacy_recipe", "family40_uncertainty", "three_family40_increment_checks_passed", "interpretation"}
                and value["interpretation"] == "internal_two_source_candidate_heterogeneity_not_validated_subtypes")
        require(set(value["branches"]) == {"bran", "raw"})
        for branch in value["branches"].values():
            require(set(branch) == {"structure", "group_arms_admitted", "source_test_profiles", "nuisance"}
                    and structure.validate_aggregate(branch["structure"]) and set(branch["source_test_profiles"]) == set(SOURCES))
            active = branch["structure"]["status"] == "supported" and branch["structure"].get("stability_gate") is True
            require(branch["group_arms_admitted"] is active)
            for profile in branch["source_test_profiles"].values():
                if active: previous.validate_profiles(profile, branch["structure"]["selected_k"])
                else: require(profile == {"status": "not_run_structure_gate_failed"})
            s1_panel.validate_nuisance(branch["nuisance"])
            if not active: require(branch["nuisance"] == {"status": "not_run_structure_gate_failed"})
        legacy = value["legacy_recipe"]
        require(outcome.validate_report(legacy))
        if "arms" in legacy:
            for branch, names in (("bran", ("context_bran_groups", "context_state_bran_groups")),
                                  ("raw", ("context_raw_groups",))):
                for name in names:
                    require((legacy["arms"][name]["status"] == "available") is
                            value["branches"][branch]["group_arms_admitted"])
        validate_overlay(value["family40_uncertainty"], value["legacy_recipe"])
        keys = ("bran_groups_minus_context", "bran_groups_minus_raw_groups", "state_bran_groups_minus_state")
        expected = value["branches"]["bran"]["group_arms_admitted"] and all(value["family40_uncertainty"]["contrasts"][key].get("utility_gate") is True for key in keys)
        require(value["three_family40_increment_checks_passed"] is bool(expected))
    except Exception:
        _fail()


def replay(frame: dict, objects: object) -> dict:
    try:
        check_frame(frame); require(type(objects) is dict and set(objects) == {"family", "frame", "rows", "known_rows", "fits", "outcome", "binding", "design_stats", "legacy_recipe", "family40_uncertainty"}
                and objects["family"] in FAMILIES and objects["frame"] == FRAME and objects["binding"] == _binding(objects["family"]))
        rows = objects["rows"]; require(type(rows) is np.ndarray and rows.dtype == np.dtype(np.int64)
                and np.array_equal(rows, np.flatnonzero(frame["membership"][:, FAMILIES.index(objects["family"])])))
        known_rows = objects["known_rows"]
        require(type(known_rows) is np.ndarray and known_rows.dtype == np.dtype(np.int64)
                and np.array_equal(known_rows, rows[frame["outcome"][rows] >= 0]))
        require(type(objects["fits"]) is dict and set(objects["fits"]) == {"bran", "raw"})
        x = design.build(**{key: frame[key][rows] for key in
                            ("values", "observed", "age_value", "age_lower", "age_upper", "age_kind", "state", "roles", "source")})
        require(set(x.stats) == set(objects["design_stats"]) and all(np.array_equal(np.asarray(x.stats[key]), np.asarray(value), equal_nan=True)
                for key, value in objects["design_stats"].items()))
        labels, result = {}, {}
        for branch, states in (("bran", frame["state"][rows]), ("raw", x.raw_padded192)):
            fit = objects["fits"][branch]
            if fit._mixture is not None:
                # Some rejected structure fits retain a GMM, while their
                # public predict rejects the status. Replay the saved numeric
                # transform directly, as the S1 panel does.
                replayed = fit._mixture.predict(fit._pca.transform(fit._scaler.transform(states)))
                require(type(replayed) is np.ndarray and replayed.shape == (len(rows),)
                        and np.issubdtype(replayed.dtype, np.integer)
                        and np.all((replayed >= 0) & (replayed < fit.selected_k)))
                labels[branch] = replayed
                result[branch] = replayed
        sink = objects["outcome"]
        require(type(sink) is dict and sink.get("schema") == outcome.PRIVATE_SCHEMA
                and sink.get("design_binding") == objects["binding"] and type(sink.get("arms")) is dict)
        test = (frame["roles"][rows] == 2) & (frame["outcome"][rows] >= 0)
        if sink["arms"]:
            designs = {"context": x.context59[test], "context_state": np.column_stack((x.context59[test], x.state_scaled[test]))}
            for branch, names in (("bran", ("context_bran_groups", "context_state_bran_groups")), ("raw", ("context_raw_groups",))):
                if names[0] not in sink["arms"]: continue
                one = outcome._one_hot(labels[branch][test], objects["fits"][branch].selected_k)
                designs[names[0]] = np.column_stack((x.context59[test], one))
                if branch == "bran": designs[names[1]] = np.column_stack((x.context59[test], x.state_scaled[test], one))
            require(set(designs) == set(sink["arms"]))
            for arm, saved in sink["arms"].items():
                require(type(saved) is dict and set(saved) == {"model", "calibration_offset", "feature_width", "test_predictions"}
                        and type(saved["feature_width"]) is int and saved["feature_width"] == designs[arm].shape[1]
                        and getattr(saved["model"], "n_features_in_", saved["feature_width"]) == saved["feature_width"]
                        and type(saved["test_predictions"]) is np.ndarray and saved["test_predictions"].dtype == np.dtype(np.float64)
                        and saved["test_predictions"].shape == (int(test.sum()),)
                        and type(saved["calibration_offset"]) is float and np.isfinite(saved["calibration_offset"]))
                probability = outcome._calibrated_probabilities(saved["model"].decision_function(designs[arm]), saved["calibration_offset"])
                require(np.array_equal(probability, saved["test_predictions"])); result["outcome_" + arm] = probability
        overlay = _overlay(objects["legacy_recipe"], sink, _labels_for_test(frame, rows))
        require(overlay == objects["family40_uncertainty"])
        return result
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail()


__all__ = ["ERROR", "FAMILIES", "FAMILYWISE_COMPARISONS", "FLAGS", "FRAME", "OVERLAY_SCHEMA", "PrivateResult",
           "SOURCES", "check_frame", "fit_family", "replay", "validate_overlay", "validate_report"]
