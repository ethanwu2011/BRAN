"""Closed, aggregate-only admission audit for the R7 training lifecycle.

This sidecar deliberately delegates the original authentication first.  It
then checks the bounded receipt surface that the original runner leaves open:
closed JSON schemas and file sets, the fit-to-pilot dependency, source/fold
pins, and private-checkpoint containment.  It never loads a checkpoint or a
source reader and never returns anything other than the original runner's
authenticated tuple.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import run_bran_robust_clinical_r7 as legacy
from audit_bran_source_pattern_v6 import read as _read_json
from bran_multisource_protocol_v2 import digest, validate_receipts
from bran_research_state_io_v1 import canonical_registry
from run_bran_multisource_fit_v2 import TRAINING


ROOT = Path(__file__).resolve().parent
ERROR = "robust_clinical_r7_v2_audit_failed"

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_RECEIPT_KEYS = frozenset(("protocol_sha256", "aggregate_sha256", "terminal_sha256"))
_SOURCE_BINDING_KEYS = frozenset((
    "source_roles", "artifact_pins", "reference_protocol_sha256",
    "reference_audit_sha256", "outer_fold_sha256", "inner_fold_sha256",
    "transform_sha256", "paired_retinal_input", "normalization",
    "protected_sources_used", "patient_level_output_emitted",
))
_PROTOCOL_KEYS = frozenset((
    "schema", "stage", "status", "parameters", "evaluation", "code_sha256",
    "source_binding", "baseline_record_sha256", "parent_checkpoints",
    "control_receipt", "pilot_dependency", "patient_level_output_emitted",
    "candidate_promoted",
))
_AGGREGATE_KEYS = frozenset((
    "schema", "status", "stage", "component_sha256",
    "total_training_loop_seconds", "projected_candidate_fit_seconds",
    "pilot_models_discarded", "control_retrained", "matched_all_streams",
    "patient_level_output_emitted", "candidate_promoted",
    "scientific_goal_achieved",
))
_TERMINAL_KEYS = frozenset((
    "status", "protocol_sha256", "aggregate_sha256",
    "patient_level_output_emitted",
))
_PROGRESS_KEYS = frozenset((
    "phase", "fold", "role", "updates_completed", "pid",
    "patient_level_output_emitted",
))
_COMPONENT_BASE_KEYS = frozenset((
    "role", "fold", "updates_completed", "runtime_seconds",
    "algorithm_update_counters", "paired_input_digest",
    "paired_completion_mask_digest", "bridge_mask_digest",
    "source_values_digest", "source_availability_digest",
    "patient_level_output_emitted", "archived_control_streams_matched",
))
_COMPONENT_FIT_KEYS = _COMPONENT_BASE_KEYS | frozenset((
    "binding", "checkpoint_sha256", "checkpoint_reload_exact", "exposure",
))
_BINDING_KEYS = frozenset((
    "protocol_sha256", "fold", "role", "training_recipe",
    "initial_checkpoint_sha256", "archived_control_checkpoint_sha256",
    "transform_sha256", "model_config_sha256", "clinical_field_order_sha256",
    "outer_fold_sha256", "inner_fold_sha256",
))
_TRACE_KEYS = (
    "paired_input_digest", "paired_completion_mask_digest", "bridge_mask_digest",
    "source_values_digest", "source_availability_digest",
)
_COMMON_PUBLIC_FILES = frozenset((
    "protocol.json", "aggregate.json", "completed.json", "progress.json",
))
_RETINAL_EXPOSURE_SOURCES = frozenset(("brset",))


def _fail() -> None:
    raise ValueError(ERROR) from None


def _require(condition: bool) -> None:
    if not condition:
        _fail()


def _is_hash(value: object) -> bool:
    return type(value) is str and _HASH.fullmatch(value) is not None


def _is_number(value: object, *, nonnegative: bool = False) -> bool:
    if type(value) not in (int, float) or isinstance(value, bool):
        return False
    result = math.isfinite(float(value))
    return result and (not nonnegative or float(value) >= 0.0)


def _file_sha(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1)
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _read(path: Path) -> dict:
    try:
        value = _read_json(path)
    except Exception:
        _fail()
    _require(type(value) is dict)
    return value


def _check_hash_map(value: object) -> None:
    _require(type(value) is dict and bool(value))
    _require(all(type(name) is str and Path(name).name == name and _is_hash(pin)
                 for name, pin in value.items()))


def _check_receipt(value: object) -> None:
    _require(type(value) is dict and set(value) == _RECEIPT_KEYS)
    _require(all(_is_hash(value[key]) for key in _RECEIPT_KEYS))


def _check_source_binding(value: object) -> None:
    """Validate the row-free source receipt emitted by BoundSourcesV3.

    ``validate_receipts`` is the existing closed validator for ``source_roles``;
    the remaining fields are the fixed receipt surface in
    ``BoundSourcesV3.receipt``.  Artifact-pin contents are intentionally not
    reinterpreted here because their producer owns that nested schema.
    """

    _require(type(value) is dict and set(value) == _SOURCE_BINDING_KEYS)
    try:
        validate_receipts(value["source_roles"])
    except Exception:
        _fail()
    _require(type(value["artifact_pins"]) is dict)
    for key in ("reference_protocol_sha256", "reference_audit_sha256",
                "outer_fold_sha256"):
        _require(_is_hash(value[key]))
    for key in ("inner_fold_sha256", "transform_sha256"):
        sequence = value[key]
        _require(type(sequence) in (list, tuple) and len(sequence) == 5
                 and all(_is_hash(item) for item in sequence))
    _require(value["paired_retinal_input"] ==
             "retained_native_original_bytes_in_authenticated_equivalent_V3_frame")
    _require(value["normalization"] == "retained_native_fold_values_not_refit_V2_values")
    _require(value["protected_sources_used"] is False
             and value["patient_level_output_emitted"] is False)


def _field_order_sha256(source_binding: dict) -> str:
    # The current source receipt predates an explicit field-order member.  The
    # runner binds ``digest(sources.paired.names)``; the immutable row-free
    # registry is the local source of that exact 59-name order.
    advertised = source_binding.get("clinical_field_order_sha256")
    if advertised is not None:
        _require(_is_hash(advertised))
        return advertised
    names, _ = canonical_registry()
    return digest(names)


def _lower_bound(value: object) -> None:
    _require(value is None or (
        type(value) is int and not isinstance(value, bool) and value >= 20 and value % 20 == 0))


def _check_exposure(value: object) -> None:
    """Check the exact coarsened exposure object emitted by ``_exposure``."""

    _require(type(value) is dict and set(value) == {
        "per_source", "global_unique_people", "cross_source_identity_resolved",
        "fold_exposures_must_not_be_summed",
    })
    _require(value["global_unique_people"] is None
             and value["cross_source_identity_resolved"] is False
             and value["fold_exposures_must_not_be_summed"] is True)
    per_source = value["per_source"]
    _require(type(per_source) is dict and set(per_source) == set(TRAINING))
    for source, item in per_source.items():
        _require(type(item) is dict)
        if source == "aireadi":
            _require(set(item) == {
                "paired_people_lower_bound_20",
                "unique_observed_clinical_measurements_lower_bound_20",
                "pooled_retinal_inputs_lower_bound_20",
                "pooled_vectors_not_counted_as_individual_images",
            })
            _lower_bound(item["paired_people_lower_bound_20"])
            _lower_bound(item["unique_observed_clinical_measurements_lower_bound_20"])
            _lower_bound(item["pooled_retinal_inputs_lower_bound_20"])
            _require(item["pooled_vectors_not_counted_as_individual_images"] is True)
            continue
        if source in _RETINAL_EXPOSURE_SOURCES:
            _require(set(item) == {
                "source_local_people_lower_bound_20", "unique_examples_lower_bound_20",
                "unique_retinal_images_lower_bound_20",
            })
            people = item["source_local_people_lower_bound_20"]
            _lower_bound(people)
            _lower_bound(item["unique_examples_lower_bound_20"])
            _lower_bound(item["unique_retinal_images_lower_bound_20"])
        else:
            _require(set(item) == {
                "source_local_people_lower_bound_20", "unique_examples_lower_bound_20",
                "unique_observed_measurements_lower_bound_20",
            })
            people = item["source_local_people_lower_bound_20"]
            _lower_bound(people)
            _lower_bound(item["unique_examples_lower_bound_20"])
            _lower_bound(item["unique_observed_measurements_lower_bound_20"])
        if people is None:
            _require(item["unique_examples_lower_bound_20"] is None)
        else:
            _require(item["unique_examples_lower_bound_20"] is not None)


def _check_directory(out: Path, private: Path, stage: str, component_names: set[str]) -> None:
    _require(out.is_dir() and not out.is_symlink())
    expected = set(_COMMON_PUBLIC_FILES) | component_names
    _require({item.name for item in out.iterdir()} == expected)
    for item in out.iterdir():
        _require(item.is_file() and not item.is_symlink() and item.stat().st_nlink == 1)
    if stage == "pilot":
        _require(not private.exists() and not private.is_symlink())
        return
    _require(private.is_dir() and not private.is_symlink()
             and private.stat().st_mode & 0o777 == 0o700)
    checkpoint_names = {f"fold{fold}_R.pt" for fold in range(5)}
    _require({item.name for item in private.iterdir()} == checkpoint_names)
    for item in private.iterdir():
        _require(item.is_file() and not item.is_symlink() and item.stat().st_nlink == 1
                 and item.stat().st_mode & 0o777 == 0o600)


def _check_progress(out: Path) -> None:
    value = _read(out / "progress.json")
    _require(set(value) == _PROGRESS_KEYS and value["phase"] == "completed"
             and value["fold"] is None and value["role"] is None
             and value["updates_completed"] == 0
             and type(value["pid"]) is int and not isinstance(value["pid"], bool)
             and value["pid"] > 0 and value["patient_level_output_emitted"] is False)


def _check_protocol(value: dict, stage: str) -> None:
    _require(set(value) == _PROTOCOL_KEYS
             and value["schema"] == "bran-robust-clinical-r7-protocol"
             and value["stage"] == stage and value["status"] == "frozen_before_training"
             and value["parameters"] == legacy.PARAMETERS
             and value["evaluation"] == legacy.EVALUATION)
    _check_hash_map(value["code_sha256"])
    _check_source_binding(value["source_binding"])
    _require(_is_hash(value["baseline_record_sha256"]))
    _check_receipt(value["control_receipt"])
    _require(value["patient_level_output_emitted"] is False
             and value["candidate_promoted"] is False)
    if stage == "pilot":
        _require(value["pilot_dependency"] is None)
    else:
        _check_receipt(value["pilot_dependency"])


def _check_aggregate(value: dict, stage: str, component_names: set[str]) -> None:
    _require(set(value) == _AGGREGATE_KEYS
             and value["schema"] == "bran-robust-clinical-r7-training"
             and value["status"] == "completed" and value["stage"] == stage
             and type(value["component_sha256"]) is dict
             and set(value["component_sha256"]) == component_names
             and all(_is_hash(pin) for pin in value["component_sha256"].values())
             and _is_number(value["total_training_loop_seconds"], nonnegative=True)
             and value["pilot_models_discarded"] is (stage == "pilot")
             and value["control_retrained"] is False
             and value["matched_all_streams"] is True
             and value["patient_level_output_emitted"] is False
             and value["candidate_promoted"] is False
             and value["scientific_goal_achieved"] is False)
    projected = value["projected_candidate_fit_seconds"]
    if stage == "pilot":
        _require(_is_number(projected, nonnegative=True))
    else:
        _require(projected is None)


def _check_terminal(value: dict, out: Path) -> None:
    _require(set(value) == _TERMINAL_KEYS
             and value["status"] == "authenticated_completed"
             and value["patient_level_output_emitted"] is False
             and value["protocol_sha256"] == _file_sha(out / "protocol.json")
             and value["aggregate_sha256"] == _file_sha(out / "aggregate.json"))


def _check_component(value: dict, name: str, stage: str, protocol: dict,
                     protocol_path: Path, private: Path) -> None:
    expected_keys = _COMPONENT_BASE_KEYS if stage == "pilot" else _COMPONENT_FIT_KEYS
    _require(set(value) == expected_keys)
    fold = value["fold"]
    role = value["role"]
    _require(type(fold) is int and not isinstance(fold, bool)
             and fold in ((0,) if stage == "pilot" else range(5)))
    _require(role in (("C", "R") if stage == "pilot" else ("R",)))
    _require(name == f"fold{fold}_{role}.json")
    budget = 100 if stage == "pilot" else 3000
    _require(value["updates_completed"] == budget
             and _is_number(value["runtime_seconds"], nonnegative=True)
             and value["patient_level_output_emitted"] is False
             and value["archived_control_streams_matched"] is True)
    try:
        legacy.control.check_counters(value["algorithm_update_counters"], "C", budget)
    except Exception:
        _fail()
    for key in _TRACE_KEYS:
        _require(_is_hash(value[key]))
    if stage == "pilot":
        return
    binding = value["binding"]
    _require(type(binding) is dict and set(binding) == _BINDING_KEYS
             and binding["protocol_sha256"] == _file_sha(protocol_path)
             and binding["fold"] == fold and binding["role"] == "R"
             and binding["training_recipe"] == "robust_clinical_r7"
             and _is_hash(binding["initial_checkpoint_sha256"])
             and _is_hash(binding["archived_control_checkpoint_sha256"])
             and _is_hash(binding["transform_sha256"])
             and _is_hash(binding["model_config_sha256"])
             and _is_hash(binding["clinical_field_order_sha256"])
             and _is_hash(binding["outer_fold_sha256"])
             and _is_hash(binding["inner_fold_sha256"])
             and binding["outer_fold_sha256"] == protocol["source_binding"]["outer_fold_sha256"]
             and binding["inner_fold_sha256"] == protocol["source_binding"]["inner_fold_sha256"][fold]
             and binding["transform_sha256"] == protocol["source_binding"]["transform_sha256"][fold]
             and binding["clinical_field_order_sha256"] ==
             _field_order_sha256(protocol["source_binding"])
             and value["checkpoint_reload_exact"] is True
             and _is_hash(value["checkpoint_sha256"])
             and value["checkpoint_sha256"] == _file_sha(private / f"fold{fold}_R.pt"))
    _check_exposure(value["exposure"])


def _validate(stage: str, attempt: int, result: tuple) -> None:
    _require(type(result) is tuple and len(result) == 4)
    protocol, aggregate, items, receipt = result
    _require(type(protocol) is dict and type(aggregate) is dict and type(items) is dict
             and type(receipt) is dict)
    _check_protocol(protocol, stage)
    out, private = legacy.paths(stage, attempt)
    folds = (0,) if stage == "pilot" else tuple(range(5))
    roles = ("C", "R") if stage == "pilot" else ("R",)
    component_names = {f"fold{fold}_{role}.json" for fold in folds for role in roles}
    _check_directory(out, private, stage, component_names)
    _check_progress(out)
    _check_aggregate(aggregate, stage, component_names)
    terminal = _read(out / "completed.json")
    _check_terminal(terminal, out)
    _require(receipt == {
        "protocol_sha256": terminal["protocol_sha256"],
        "aggregate_sha256": terminal["aggregate_sha256"],
        "terminal_sha256": _file_sha(out / "completed.json"),
    })
    protocol_path = out / "protocol.json"
    for name in sorted(component_names):
        component = _read(out / name)
        _check_component(component, name, stage, protocol, protocol_path, private)
        _require(_file_sha(out / name) == aggregate["component_sha256"][name])
        _require(items.get((component["role"], component["fold"])) == component
                 or not items)
    if stage == "fit":
        pilot = legacy.authenticate("pilot", attempt)
        _require(type(pilot) is tuple and len(pilot) == 4 and type(pilot[3]) is dict)
        _require(protocol["pilot_dependency"] == pilot[3])


def authenticate(stage: str, attempt: int):
    """Run legacy authentication first, then apply the closed R7 v2 audit."""

    try:
        result = legacy.authenticate(stage, attempt)
        _validate(stage, attempt, result)
        return result
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


__all__ = ["ERROR", "authenticate"]
