"""Private, unchanged historical-reference prediction assembly for V2 gates.

The caller must hold the local FD-quiet boundary and legacy lock.  This module
uses the original historical transform and fixed reference helpers only: it
does not train, write, expose arrays, or construct a V2 transform.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Optional

import numpy as np

import bran_multisource_mask_contract_v1 as contract
import bran_multisource_mask_experiment_v1 as experiment
import bran_multisource_reference_roles_v2 as reference_roles
import run_bran_retinal_input_bridge_v3 as bridge


_INVALID = "historical reference prediction assembly invalid"
_ROUTES = ("both", "clinical", "retinal")
_RAW = ("raw_clinical", "raw_retinal", "raw_concat", "late_average")
_COMPLETION = ("initial", "continued", "raw_clinical", "raw_concat")


@dataclass(frozen=True, repr=False)
class HistoricalReferencePredictionsV2:
    """Private in-memory reference arrays; never serialize this object."""

    screening: dict
    completion: dict
    target: np.ndarray
    observed: dict
    groups: dict
    labels: dict
    labelmask: dict
    missingness: dict
    replay_receipt: Mapping[str, Any]


def _invalid() -> None:
    raise ValueError(_INVALID)


def _require(value: bool) -> None:
    if not value:
        _invalid()


def _equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, right, equal_nan=True))


def _endpoint_names(paired: Any, context: reference_roles.HistoricalReferenceContextV2) -> tuple[str, ...]:
    names = tuple(getattr(paired, "endpoint_names", ()))
    try:
        historical = tuple(context.protocol["native_source"]["source"]["endpoint_names"])
    except (KeyError, TypeError):
        _invalid()
    _require(len(names) == len(set(names)) == 26 and names == historical)
    return names


def _paired_age(paired: Any) -> np.ndarray:
    value = getattr(paired, "age_value", None)
    _require(isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype.kind == "f")
    return value


def _receipt_digest(value: Any) -> str:
    if isinstance(value, str) and len(value) == 64:
        return value
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError):
        _invalid()
    return hashlib.sha256(encoded).hexdigest()


def _validate_retinal_binding(context: reference_roles.HistoricalReferenceContextV2, receipt: Mapping[str, Any]) -> None:
    """Authenticate V2 pooled features through the V3 bridge metadata only."""

    try:
        expected = context.protocol["retinal_bridge"]
        binding = receipt["retinal_binding"]
        path = bridge.PROTOCOL
        first = bridge.sha(path)
        _require(first == expected["protocol_sha256"] and path.is_file() and not path.is_symlink())
        metadata = json.loads(path.read_text())
        _require(bridge.sha(path) == first)
        _require(metadata["retinal_protocol_sha256"] == binding["protocol_sha256"]
                 and metadata["retinal_audit_sha256"] == binding["audit_sha256"]
                 and metadata["source_policy_sha256"] == binding["source_policy_sha256"])
    except (AttributeError, KeyError, TypeError, OSError, ValueError, json.JSONDecodeError):
        _invalid()


def _align(paired: Any, source: Any, names: tuple[str, ...], context: reference_roles.HistoricalReferenceContextV2):
    """Load the legacy context once and reject every row/frame mismatch."""

    try:
        ctx, folds, c0, cm0, eligible, r0, rm, ages, registry = source.io.load_context()
    except (AttributeError, TypeError, ValueError, OSError, RuntimeError):
        _invalid()
    required = (c0, cm0, eligible, r0, rm, ages, folds)
    _require(all(isinstance(item, np.ndarray) for item in required)
             and c0.ndim == 2 and c0.shape[1] == 59 and cm0.shape == c0.shape and cm0.dtype == bool
             and eligible.shape == c0.shape and eligible.dtype == bool
             and np.array_equal(eligible, np.broadcast_to(eligible[0], eligible.shape))
             and r0.shape == (len(c0), 384) and rm.shape == (len(c0),) and rm.dtype == bool
             and ages.shape == (len(c0),) and folds.shape == (len(c0),)
             and tuple(registry) == tuple(getattr(paired, "names", ())))
    for name in ("c", "cm", "r", "rm", "folds", "labels", "labelmask"):
        _require(isinstance(getattr(paired, name, None), np.ndarray))
    _require(_equal(paired.c, c0) and np.array_equal(paired.cm, cm0)
             and np.array_equal(paired.rm, rm)
             and np.array_equal(paired.folds, folds) and _equal(_paired_age(paired), ages))
    if hasattr(paired, "eligible_indices"):
        slots = np.zeros(59, dtype=bool)
        try:
            slots[list(paired.eligible_indices)] = True
        except (TypeError, IndexError):
            _invalid()
        _require(np.array_equal(slots, eligible[0]))
    receipt = getattr(paired, "receipt", None)
    try:
        auth = context.protocol["native_source"]["source"]["authentication"]
        outer, inner = receipt["outer_fold_sha256"], receipt["inner_fold_sha256"]
        source_digest = receipt["source_receipt_sha256"]
        source_receipt = source.io.source_receipt()
    except (KeyError, TypeError):
        _invalid()
    _require(outer == auth["outer_fold_sha256"] and tuple(inner) == tuple(auth["inner_fold_sha256"])
             and source_digest == _receipt_digest(source_receipt)
             and np.allclose(r0[rm], paired.r[rm], **bridge.PARAMETERS["equivalence"])
             and set(np.unique(folds)) == set(range(5)))
    _validate_retinal_binding(context, receipt)
    labels = {name: ctx["labels_by_source"][name] for name in names}
    labelmask = {name: np.asarray(ctx["observed_by_source"][name], dtype=bool) for name in names}
    for index, name in enumerate(names):
        _require(labels[name].shape == (len(c0),) and labelmask[name].shape == (len(c0),)
                 and _equal(np.asarray(paired.labels[:, index]), np.asarray(labels[name]))
                 and np.array_equal(np.asarray(paired.labelmask[:, index], dtype=bool), labelmask[name]))
    return ctx, folds, c0, cm0, eligible, r0, rm, ages, tuple(registry), labels, labelmask


def _progress(callback: Optional[Any], phase: str, fold: Optional[int] = None) -> None:
    if callback is not None:
        value = {"phase": phase}
        if fold is not None:
            value["fold"] = fold
        callback(value)


def _same_aggregate(actual: Any, expected: Any) -> bool:
    """Compare frozen legacy summaries without relying on MappingProxy equality."""

    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and set(actual) == set(expected) and all(
            _same_aggregate(actual[key], expected[key]) for key in expected)
    if isinstance(expected, (tuple, list)):
        return isinstance(actual, (tuple, list)) and len(actual) == len(expected) and all(
            _same_aggregate(left, right) for left, right in zip(actual, expected))
    return actual == expected


def _verify_replay(context: reference_roles.HistoricalReferenceContextV2, screen: dict, completion: dict,
                   target: np.ndarray, observed: dict, groups: dict, labels: dict, labelmask: dict,
                   folds: np.ndarray, names: tuple[str, ...], source: Any) -> tuple[dict, dict]:
    """Check original published points without inventing the historical student."""
    try:
        expected_screen = context.result["screening"]["endpoints"]
        expected_completion = context.result["completion"]
        expected_no_retina = context.result["completion_no_retina"]
    except (KeyError, TypeError):
        _invalid()
    masks = experiment.old.screen_masks(screen, labelmask, names, labels, folds)
    arms = tuple("initial_" + route for route in _ROUTES) + tuple("continued_" + route for route in _ROUTES) + _RAW
    points = {arm: [] for arm in arms}
    for endpoint in names:
        mask = masks[endpoint]
        expected = expected_screen[endpoint]
        supported = all(np.count_nonzero(mask & (labels[endpoint] == value)) >= 20 for value in (0, 1))
        _require(expected.get("status") == ("supported" if supported else "unsupported"))
        if not supported:
            continue
        for arm in arms:
            _require(np.isfinite(screen[endpoint][arm][mask]).all())
            point = source.base.fold_weighted_auc(labels[endpoint], screen[endpoint][arm], mask, folds)
            _require(np.isfinite(point) and abs(float(point) - float(expected["arms"][arm]["auroc"])) <= 1e-10)
            points[arm].append(float(point))
    complete = all(len(values) == len(names) for values in points.values())
    _require(context.result["screening"]["complete_panel"] is complete)
    macro = context.result["screening"]["macro"]
    if complete:
        _require(isinstance(macro, Mapping))
        for arm in arms:
            _require(abs(float(np.mean(points[arm])) - float(macro["arms"][arm]["auroc"])) <= 1e-10)
    else:
        _require(macro is None)
    for pattern in contract.EVALPATTERNS:
        expected = expected_no_retina[pattern.replace("_no_retina", "_hidden")] if pattern.endswith("_no_retina") else expected_completion[pattern]
        safe_groups = experiment.old.safe_tail_groups(observed[pattern], groups[pattern])
        for index, field in enumerate(experiment.metrics.CBC_FIELDS):
            for group, cell in expected[field].items():
                valid = observed[pattern][:, index] & safe_groups[group][:, index] & np.isfinite(target[:, index])
                supported = np.count_nonzero(valid) >= 20
                _require(cell.get("status") == ("supported" if supported else "unsupported"))
                if not supported:
                    continue
                for arm in _COMPLETION:
                    error = completion[pattern][arm][valid, index] - target[valid, index]
                    values = cell["arms"][arm]
                    _require(np.isfinite(error).all()
                             and abs(float(np.abs(error).mean()) - float(values["mae"])) <= 1e-10
                             and abs(float(np.square(error).mean()) - float(values["mse"])) <= 1e-10)
    return {}, {}


def build(paired, reference_context: reference_roles.HistoricalReferenceContextV2, progress=None) -> HistoricalReferencePredictionsV2:
    """Assemble immutable V1 initial/continued/raw reference predictions in memory."""

    import torch
    torch.set_num_threads(2)  # Match the original frozen replay runtime.
    _require(isinstance(reference_context, reference_roles.HistoricalReferenceContextV2))
    source = experiment.origin.native.source
    names = _endpoint_names(paired, reference_context)
    ctx, folds, c0, cm0, eligible, r0, rm, ages, registry, labels, labelmask = _align(
        paired, source, names, reference_context)
    _progress(progress, "reference_loading")
    slots = tuple(registry.index(field) for field in experiment.metrics.CBC_FIELDS)
    _require(len(set(slots)) == 9 and all(0 <= item < 48 for item in slots))
    target, base_observed = c0[:, slots].copy(), (cm0 & eligible)[:, slots].copy()
    screening = {name: {"initial_" + route: np.full(len(folds), np.nan) for route in _ROUTES} for name in names}
    for values in screening.values():
        values.update({"continued_" + route: np.full(len(folds), np.nan) for route in _ROUTES})
        values.update({key: np.full(len(folds), np.nan) for key in _RAW})
    completion = {pattern: {key: np.full((len(folds), 9), np.nan) for key in _COMPLETION}
                  for pattern in contract.EVALPATTERNS}
    observed = {pattern: base_observed.copy() for pattern in contract.EVALPATTERNS}
    groups = {pattern: {key: np.zeros((len(folds), 9), dtype=bool) for key in experiment.metrics.GROUPS}
              for pattern in contract.EVALPATTERNS}
    stress = {version: {pattern: np.full((len(folds), len(names)), np.nan) for pattern in experiment.masking.PATTERNS}
              for version in ("initial", "continued")}
    auth = reference_context.protocol["native_source"]["source"]["authentication"]
    for fold in range(5):
        _progress(progress, "reference_inference", fold)
        tr, te = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        _require(len(te) > 0)
        transform = source.base.FoldTransform(c0, cm0, eligible, r0, rm, ages, tr)
        c, cm, r, age = transform.apply(c0, cm0, eligible, r0, rm, ages)
        models = reference_roles.fold_models(reference_context, fold, transform)
        _require(set(models) == {"initial", "continued"})
        inner, identity = source.base._inner_context(ctx, tr, fold)
        _require(identity == auth["inner_fold_sha256"][fold])
        full_inner = np.full(len(folds), -1, dtype=int)
        full_inner[tr] = inner
        native = {role: experiment.origin.native.kernel.predict_native(model, c, cm, r, rm, age)
                  for role, model in models.items()}
        from bran_matched_screening_kernel_v1 import fit_predict, late_fusion_average
        raw = {"raw_clinical": np.c_[c, cm, age], "raw_retinal": np.c_[r, rm, age],
               "raw_concat": np.c_[c, cm, r, rm, age]}
        for endpoint_index, endpoint in enumerate(names):
            for role in ("initial", "continued"):
                for route in _ROUTES:
                    screening[endpoint][role + "_" + route][te] = native[role][route][te, endpoint_index]
            for arm, values in raw.items():
                screening[endpoint][arm][te], _ = fit_predict(values, labels[endpoint], labelmask[endpoint], tr, te,
                                                               full_inner, family="extra_trees", seed=92381 + fold)
            screening[endpoint]["late_average"][te] = late_fusion_average(
                screening[endpoint]["raw_clinical"][te], screening[endpoint]["raw_retinal"][te])
        for pattern in contract.EVALPATTERNS:
            visible_r, visible_rm, base_pattern = experiment.cbc_context(r, rm, pattern)
            for role, model in models.items():
                values, support = experiment.cbc_prediction(model, c, cm, r, rm, age, slots, transform, pattern)
                if role == "initial":
                    reference_support = support.copy()
                    reference_support_full = support.copy()
                else:
                    _require(np.array_equal(support, reference_support))
                completion[pattern][role][te] = values[te]
                observed[pattern][te] &= support[te]
            for index in range(9):
                hidden, hidden_mask = source.mask_inputs(c, cm, slots, base_pattern, slots[index])
                valid = base_observed[:, index] & reference_support_full[:, index]
                for arm, values in {"raw_clinical": np.c_[hidden, hidden_mask, age],
                                    "raw_concat": np.c_[hidden, hidden_mask, visible_r, visible_rm, age]}.items():
                    completion[pattern][arm][te, index] = source.ev.fixed_cbc_probe(
                        values, target[:, index], valid, tr, te)
                training_target = target[tr, index][valid[tr]]
                _require(len(training_target) >= 20)
                groups[pattern]["overall"][te, index] = True
                for name, mask in experiment.origin.previous.tail_masks(training_target, target[te, index]).items():
                    _require(name in groups[pattern] and mask.shape == (len(te),))
                    groups[pattern][name][te, index] = mask
        for role, model in models.items():
            for pattern in experiment.masking.PATTERNS:
                hidden = experiment.masking.remove_inputs(c, cm, r, rm, slots, pattern)
                experiment.masking.assert_no_input_leak(hidden, cm, rm, slots, pattern)
                values = experiment.origin.native.kernel.predict_native(
                    model, hidden.clinical[te], hidden.clinical_mask[te], hidden.retinal[te],
                    hidden.retinal_mask[te], age[te])["both"]
                _require(np.array_equal(np.isfinite(values).all(axis=1), hidden.available[te]))
                stress[role][pattern][te] = values
    _progress(progress, "reference_replay")
    screen_summary, completion_summary = _verify_replay(reference_context, screening, completion, target, observed,
                                                         groups, labels, labelmask, folds, names, source)
    _progress(progress, "reference_bootstrap")
    label_array = np.column_stack([labels[name] for name in names])
    labelmask_array = np.column_stack([labelmask[name] for name in names]).astype(bool)
    missingness = {role: experiment.old.stress_metrics.summarize(stress[role], label_array, labelmask_array, folds, names,
                                                                   source.ev.paired_counts(folds, draws=1000, seed=91501))
                   for role in ("initial", "continued")}
    _require(all(_same_aggregate(missingness[role], reference_context.result["missingness"][role])
                 for role in ("initial", "continued")))
    receipt = {"schema": "bran-multisource-reference-predictions-v2",
               "historical_protocol_sha256": reference_context.protocol_pin,
               "historical_audit_sha256": reference_context.audit_pin,
               "replayed_initial_and_continued": True,
               "raw_controls_replayed": True,
               "historical_point_metrics_replayed": True,
               "candidate_training_performed": False,
               "patient_level_output_emitted": False}
    return HistoricalReferencePredictionsV2(screening, completion, target, observed, groups, labels, labelmask,
                                            missingness, receipt)
