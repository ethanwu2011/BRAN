"""Exclusive V5-M-state replacement for the frozen information-matched FM readout.

No source arrays, states, predictions, cache payloads, or checkpoints leave the
quiet local process.  The old V2 adapter is used only for its authenticated FM
cache/context components; it is never asked to construct a V1 BRAN state.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import bran_information_matched_fm_v2 as bench
import run_bran_information_matched_fm_v2 as old
import run_bran_v5_cbc_uncertainty as source
import run_bran_context_preservation_v5 as v5
from bran_multisource_batches_v2 import tensor
from bran_multisource_outcomes_v2 import original_age
from bran_multisource_profiles_v3 import _unchanged, _validate_provider
from bran_v5_state_routes import ROUTES, state_routes
from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha


ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "BRAN_V5_INFORMATION_MATCHED_FM_PLAN.md"
CODE = tuple(sorted(set(source.CODE) | {"bran_information_matched_fm_v2.py", "run_bran_information_matched_fm_v2.py",
        "run_bran_v5_information_matched_fm.py", "test_run_bran_v5_information_matched_fm.py",
        "BRAN_V5_INFORMATION_MATCHED_FM_PLAN.md", "bran_v5_state_routes.py", "test_bran_v5_state_routes.py",
        "bran_v5_residual_training.py", "run_bran_v5_cbc_uncertainty.py"}))
PARAMETERS = {
    "arms": list(bench.ARMS), "outer_folds": 5, "endpoints": 26,
    "readout": "prespecified_fixed_StandardScaler_L2_logistic_C1_lbfgs_max5000_default_class_policy",
    "bootstrap": {"draws": 1000, "seed": 91501, "minimum_valid": 900},
    "bran_representation": "frozen_v5_M_target_free_state_routes", "bran_replaced_arms": list(bench.BRAN_ARMS),
    "fm_raw_labrador": "existing_qualified_v2_cache_and_raw_designs", "encoder_training": False,
    "retinal_extraction": False, "historical_score_replay": False, "automatic_promotion": False,
}
FLAGS = {**bench.FLAGS, "candidate_promoted": False, "historical_score_equality_required": False,
         "v5_state_replayed_exactly": True, "v5_input_frame_matched": True}
_ERROR = "v5_information_matched_fm_contract_failed"


def require(ok: bool) -> None:
    if not ok:
        raise ValueError(_ERROR)


def paths(attempt: int) -> Path:
    require(type(attempt) is int and 1 <= attempt <= 99)
    return ROOT / f"BRAN_V5_INFORMATION_MATCHED_FM_ATTEMPT{attempt}"


def _digest(value: Any) -> str:
    return bench.canonical_sha256(value)


def _equal(left: Any, right: Any) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.flush(); os.fsync(handle.fileno())
    except Exception:
        raise


def _sha(path: Path) -> str:
    return sha(path)


def _code_hashes() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in CODE}


def _v5_checkpoint_hashes(components: Mapping[tuple[str, int], Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for fold in range(5):
        try:
            item = components[("M", fold)]
        except (KeyError, TypeError):
            require(False)
        require(item["binding"]["role"] == "M" and item["binding"]["fold"] == fold)
        value = item["checkpoint_sha256"]
        require(isinstance(value, str) and len(value) == 64)
        result[f"v5_m_fold{fold}"] = value
    return result


def decorate_binding(old_binding: Mapping[str, Any], v5_evidence: Mapping[str, Any],
                     components: Mapping[tuple[str, int], Mapping[str, Any]]) -> dict[str, Any]:
    """Append V5 provenance while retaining the V2 cache/source-byte receipt."""

    bench.validate_source_binding(old_binding)
    pins = _v5_checkpoint_hashes(components)
    value = copy.deepcopy(dict(old_binding))
    composite = _digest(pins)
    value["bran_checkpoints_sha256"] = composite
    for arm in bench.BRAN_ARMS:
        value["representation_sha256"][arm] = composite
        value["arm_provenance"][arm] = {
            "upstream_supervision": "authenticated seven-source V5-M outer-fold supervised encoder; fixed readout adds no encoder training",
            "training_exposure": "fold-specific frozen V5-M checkpoint and inherited transform; target-free route state",
        }
    value["source_files_sha256"].update(pins)
    value["parent_receipts_sha256"].update({"v5_source_context": _digest(v5_evidence),
                                             "v5_checkpoint_composite": composite})
    bench.validate_source_binding(value)
    return value


def protocol(binding: Mapping[str, Any], old_binding: Mapping[str, Any], v5_evidence: Mapping[str, Any]) -> dict[str, Any]:
    bench.validate_source_binding(binding); bench.validate_source_binding(old_binding)
    return {"schema": "bran-v5-information-matched-fm-protocol-v1", "status": "frozen_before_readouts",
            "parameters": PARAMETERS, "source_binding": dict(binding), "old_metadata_binding_sha256": _digest(old_binding),
            "v5_evidence_sha256": _digest(v5_evidence), "code_sha256": _code_hashes(), "privacy": FLAGS}


def validate_protocol(value: Mapping[str, Any], pin: str, out: Path) -> None:
    required = {"schema", "status", "parameters", "source_binding", "old_metadata_binding_sha256",
                "v5_evidence_sha256", "code_sha256", "privacy"}
    require(type(value) is dict and set(value) == required and value["schema"] == "bran-v5-information-matched-fm-protocol-v1"
            and value["status"] == "frozen_before_readouts" and value["parameters"] == PARAMETERS
            and value["privacy"] == FLAGS and value["code_sha256"] == _code_hashes() and _sha(out / "protocol.json") == pin)
    bench.validate_source_binding(value["source_binding"])


def validate_aggregate(value: Mapping[str, Any], endpoint_names: list[str]) -> None:
    expected = {"schema", "status", "protocol_sha256", "source_binding_sha256", "results"} | set(FLAGS)
    require(type(value) is dict and set(value) == expected and value["schema"] == "bran-v5-information-matched-fm-aggregate-v1"
            and value["status"] == "completed" and isinstance(value["protocol_sha256"], str) and len(value["protocol_sha256"]) == 64
            and isinstance(value["source_binding_sha256"], str) and len(value["source_binding_sha256"]) == 64
            and all(value[key] is flag for key, flag in FLAGS.items()))
    bench.validate_result(value["results"], endpoint_names)


def _aligned(old_parts: Mapping[str, Any], paired: Any, evidence: Mapping[str, Any], old_binding: Mapping[str, Any]) -> tuple:
    ctx, folds, c0, cm0, eligible, r0, rm, ages, names = old_parts["value"]
    require(_equal(folds, paired.folds) and _equal(c0, paired.c) and np.array_equal(cm0, paired.cm)
            and _equal(r0, paired.r) and np.array_equal(rm, paired.rm))
    paired_age = np.asarray(getattr(paired, "age_value"))
    require(_equal(ages, paired_age) and tuple(names) == tuple(paired.names) and tuple(old_binding["endpoint_names"]) == tuple(paired.endpoint_names))
    slots = np.zeros(59, dtype=bool); slots[list(paired.eligible_indices)] = True
    require(np.array_equal(slots, eligible[0]))
    labels = {name: np.asarray(ctx["labels_by_source"][name]) for name in old_binding["endpoint_names"]}
    observed = {name: np.asarray(ctx["observed_by_source"][name], bool) for name in old_binding["endpoint_names"]}
    for index, name in enumerate(old_binding["endpoint_names"]):
        require(_equal(labels[name], paired.labels[:, index]) and np.array_equal(observed[name], paired.labelmask[:, index]))
    receipt = paired.receipt
    require(receipt["outer_fold_sha256"] == old_binding["outer_fold_sha256"]
            and list(receipt["inner_fold_sha256"]) == old_binding["inner_fold_sha256"])
    identifiers = list(map(str, ctx["raw_cohort"].patient_ids))
    require(evidence["split_authentication"]["patient_order_sha256"] == source.digest(identifiers))
    return ctx, np.asarray(folds), c0, cm0, eligible, r0, np.asarray(rm, bool), np.asarray(ages), labels, observed


def _frame_match(c: np.ndarray, cm: np.ndarray, r: np.ndarray, rm: np.ndarray,
                 v5_c: np.ndarray, v5_cm: np.ndarray, v5_r: np.ndarray, v5_rm: np.ndarray) -> None:
    """The sole allowed bridge between V2 raw/FM and V5 state coordinates."""

    require(np.array_equal(cm, v5_cm) and np.array_equal(rm, v5_rm)
            and np.allclose(c, v5_c, rtol=1e-6, atol=1e-6)
            and np.allclose(r, v5_r, rtol=1e-6, atol=1e-6))


def build_live_adapter() -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any], Mapping[str, Any], callable, callable, callable]:
    """Return a V5-state design factory without invoking historical score replay."""

    old_binding, parts = old._live_components()
    paired, _roles, components, evidence = source.source_context()
    ctx, folds, c0, cm0, eligible, r0, rm, ages, labels, observed = _aligned(parts, paired, evidence, old_binding)
    binding = decorate_binding(old_binding, evidence, components)
    fm, old_source = parts["fm"], parts["source"]
    arrays = fm.load_fm_cache(parts["cache_binding"], expected_current_inputs_sha256=parts["current_input_sha256"],
                              expected_row_order_sha256=old_binding["row_order_sha256"],
                              expected_outer_fold_sha256=old_binding["outer_fold_sha256"],
                              expected_inner_fold_sha256=old_binding["inner_fold_sha256"])
    _, private = v5.paths("fit", 2)
    age = original_age(paired)
    states: dict[int, dict[str, np.ndarray]] = {}
    common_available = np.zeros(len(folds), dtype=bool)

    def fold_inputs(fold: int):
        train = np.flatnonzero(folds != fold)
        legacy_transform = old_source.base.FoldTransform(c0, cm0, eligible, r0, rm, ages, train)
        c, cm, r, age_value = legacy_transform.apply(c0, cm0, eligible, r0, rm, ages)
        item = components[("M", fold)]
        teacher, transform = v5.oldfit.load_checkpoint(private / f"fold{fold}_M.pt", item["checkpoint_sha256"], item["binding"])
        slots = tuple(paired.names.index(field) for field in __import__("bran_clinical_semantics_v1").CBC_FIELDS)
        before, thash, grads = _validate_provider(teacher, transform, fold, slots, paired.transforms[fold])
        vc, vcm = transform.clinical(paired.c, paired.cm)
        vr, vrm = transform.retinal(paired.r, paired.rm)
        _frame_match(c, cm, r, rm, vc, vcm, vr, vrm)
        routes = state_routes(teacher, tensor(vc), tensor(vcm, torch.bool), tensor(vr), tensor(vrm, torch.bool),
                              age, transform.age_mean, transform.age_scale)
        replay = state_routes(teacher, tensor(vc), tensor(vcm, torch.bool), tensor(vr), tensor(vrm, torch.bool),
                              age, transform.age_mean, transform.age_scale)
        require(set(routes.states) == set(ROUTES) == set(routes.available)
                and all(torch.equal(routes.states[key], replay.states[key]) and torch.equal(routes.available[key], replay.available[key]) for key in ROUTES)
                and _unchanged(before, thash, grads, teacher, transform))
        return train, c, cm, r, age_value, {key: routes.states[key].numpy() for key in ROUTES}, np.logical_and.reduce([routes.available[key].numpy() for key in ROUTES])

    legacy_by_fold: dict[int, tuple] = {}
    for fold in range(5):
        train, c, cm, r, age_value, route_states, available = fold_inputs(fold)
        legacy_by_fold[fold] = (train, c, cm, r, age_value)
        states[fold] = route_states
        common_available[folds == fold] = available[folds == fold]
    declared = fm._support_mask(rm, c0, cm0, eligible)
    common = declared & common_available
    matched_support = {name: fm.common_evaluation_mask(observed[name], common, labels[name]) for name in binding["endpoint_names"]}

    def design_factory(fold: int) -> Mapping[str, Any]:
        require(type(fold) is int and fold in range(5))
        _train, c, cm, r, age_value = legacy_by_fold[fold]
        designs = {"bran_combined_state": states[fold]["both"], "bran_clinical_state": states[fold]["clinical"],
                   "bran_retinal_state": states[fold]["retinal"], "blood_age": parts["blood_design"](c, cm, age_value),
                   "raw_clinical_age": np.c_[c, cm, age_value], "raw_clinical_retinal_age": np.c_[c, cm, r, rm, age_value]}
        for base in fm.ARMS:
            if base == "labrador":
                designs["labrador_clinical_age"] = np.c_[arrays[base], age_value]
                designs["labrador_clinical_retinal_age"] = np.c_[arrays[base], c, cm, r, rm, age_value]
            else:
                designs[base + "_age"] = np.c_[arrays[base], age_value]
                designs[base + "_clinical_age"] = np.c_[arrays[base], c, cm, age_value]
        require(set(designs) == set(bench.ARMS))
        return designs

    def reauthenticate() -> None:
        old._metadata_reauthenticate(old_binding)
        _paired, _r, _c, fresh = source.source_context()
        require(_digest(fresh) == _digest(evidence))

    return binding, {"folds": folds, "labels": labels, "observed": observed, "matched_support": matched_support,
                     "inner_folds": parts["inner"], "design_factory": design_factory, "input_sha256": binding["shared_input_sha256"],
                     "outer_fold_sha256": binding["outer_fold_sha256"], "row_order_sha256": binding["row_order_sha256"]}, old_binding, evidence, reauthenticate, old_source.ev.paired_counts, old_source.base


def run(attempt: int) -> dict[str, Any]:
    """Create one exclusive attempt.  This is never called by tests on private data."""
    out = paths(attempt)
    with quiet():
        with LOCK.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(not out.exists() and not out.is_symlink())
            out.mkdir(mode=0o700); owned = True; state = {"phase": "authentication", "fold": None, "arm": None}
            try:
                binding, context, old_binding, evidence, reauthenticate, counts_fn, base = build_live_adapter()
                frozen = protocol(binding, old_binding, evidence)
                _write_new(out / "protocol.json", frozen); pin = _sha(out / "protocol.json")
                validate_protocol(frozen, pin, out); require(frozen["v5_evidence_sha256"] == _digest(evidence)); reauthenticate()
                def progress(fold, arm, completed, budget):
                    state.update(phase="fixed_readouts", fold=fold, arm=arm)
                    temp = out / "progress.next.json"; _write_new(temp, {"phase": "fixed_readouts", "fold": fold, "arm": arm,
                        "completed_readout_calls": completed, "readout_call_budget": budget, "patient_level_output_emitted": False}); os.replace(temp, out / "progress.json")
                from sklearn.exceptions import ConvergenceWarning
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                import warnings
                def fit(x_train, y_train, observed_train, x_test):
                    mask = np.asarray(observed_train, bool); y = np.asarray(y_train)
                    require(mask.sum() >= 20 and set(np.unique(y[mask])) == {0, 1})
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., penalty="l2", solver="lbfgs", max_iter=5000))
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning); model.fit(np.asarray(x_train)[mask], y[mask])
                    require(not any(issubclass(item.category, ConvergenceWarning) for item in caught))
                    return model.predict_proba(x_test)[:, 1]
                with threadpool_limits(limits=2):
                    predictions = bench.fit_all_readouts(context, binding, fit, progress)
                    counts = counts_fn(context["folds"], draws=1000, seed=91501)
                    result = bench.summarize(predictions, context["labels"], context["matched_support"], context["folds"], counts,
                                             binding["endpoint_names"], auc_fn=base.fold_weighted_auc, bootstrap_auc_fn=base._weighted_auc_draws)
                bench.validate_result(result, binding["endpoint_names"]); reauthenticate()
                validate_protocol(frozen, pin, out)
                aggregate = {"schema": "bran-v5-information-matched-fm-aggregate-v1", "status": "completed", "protocol_sha256": pin,
                             "source_binding_sha256": _digest(binding), "results": result, **FLAGS}
                validate_aggregate(aggregate, binding["endpoint_names"])
                _write_new(out / "aggregate.json", aggregate)
                _write_new(out / "manifest.json", {"protocol_sha256": pin, "aggregate_sha256": _sha(out / "aggregate.json"), **FLAGS})
                _write_new(out / "completed.json", {"status": "authenticated", "protocol_sha256": pin,
                           "aggregate_sha256": _sha(out / "aggregate.json"), "candidate_promoted": False, "patient_level_output_emitted": False})
                return aggregate
            except Exception as exc:
                if owned and not (out / "completed.json").exists() and not (out / "failure.json").exists():
                    # Static code locations only. Never serialize exception text,
                    # locals, array values, patient paths, or source payloads.
                    trace = exc.__traceback__
                    site = {"module": "unclassified", "line": None}
                    while trace is not None:
                        candidate = Path(trace.tb_frame.f_code.co_filename)
                        if candidate.parent == ROOT and candidate.suffix == ".py":
                            site = {"module": candidate.name, "line": trace.tb_lineno}
                        trace = trace.tb_next
                    _write_new(out / "failure.json", {"schema": "bran-v5-information-matched-fm-failure-v1", "status": "technical_failure",
                               "phase": state["phase"], "fold": state["fold"], "arm": state["arm"], "code_site": site,
                               "patient_level_output_emitted": False})
                raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--attempt", required=True, type=int); args = parser.parse_args(argv)
    try:
        run(args.attempt); print(json.dumps({"status": "completed", "patient_level_output_emitted": False})); return 0
    except Exception:
        print(json.dumps({"status": "not_completed", "patient_level_output_emitted": False})); return 1


if __name__ == "__main__":
    raise SystemExit(main())
