"""Exclusive, authenticated fixed-C1 frozen-FM V2 adapter.

The adapter is executable only in an explicitly authorized local terminal.  It
never replays historical AUROCs, trains an encoder, or writes patient-level
material.  All source/context/model handling is descriptor-quiet and held by
the existing shared heavy lock.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np
from threadpoolctl import threadpool_limits

import bran_information_matched_fm_v2 as bench

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "BRAN_INFORMATION_MATCHED_FM_PROTOCOL_V2.json"
OUT = ROOT / "BRAN_INFORMATION_MATCHED_FM_V2"
AUDIT = ROOT / "BRAN_INFORMATION_MATCHED_FM_AUDIT_V2"
LOCK = Path(bench.LOCK_PATH)
CODE = ("bran_information_matched_fm_v2.py", "run_bran_information_matched_fm_v2.py",
        "test_bran_information_matched_fm_v2.py", "BRAN_INFORMATION_MATCHED_FM_DESIGN_V2.md")
PARAMETERS = {
    "arms": list(bench.ARMS), "primary_comparisons": list(bench.PRIMARY_COMPARATORS),
    "outer_folds": 5, "authenticated_inner_folds": 5, "endpoints": 26,
    "readout": "prespecified_fixed_StandardScaler_L2_logistic_C1_lbfgs_max5000_default_class_policy",
    "inner_folds": "authenticated_original_assignments_provenance_only_no_V2_selection",
    "bootstrap": {"draws": 1000, "seed": 91501, "minimum_valid": 900},
    "encoder_training": False, "retinal_extraction": False, "pixel_input": False,
    "external_source_scores": False, "native_head_readout_budget_equivalence_claim": False,
    "historical_auroc_equality_required": False,
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_write(path: Path, value: Mapping[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise bench.ContractError("exclusive_artifact_already_exists") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.flush(); os.fsync(handle.fileno())
    except Exception:
        raise


def _exclusive_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise bench.ContractError("exclusive_output_exists") from exc
    if path.is_symlink() or not path.is_dir() or (path.stat().st_mode & 0o777) != 0o700:
        raise bench.ContractError("exclusive_directory_invalid")


@contextmanager
def _lifecycle_lock():
    """One non-reentrant process lock for every V2 lifecycle operation."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(LOCK, flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


@contextmanager
def fd_quiet():
    """Keep local array/model-library output off inherited descriptors."""
    saved_out, saved_err = os.dup(1), os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1); os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved_out, 1); os.dup2(saved_err, 2)
        os.close(null); os.close(saved_out); os.close(saved_err)


def prepare(source_binding: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze source bytes, code, runtime-independent policy before scores."""
    with _lifecycle_lock():
        return _prepare_unlocked(source_binding)


def _prepare_unlocked(source_binding: Mapping[str, Any]) -> dict[str, Any]:
    bench.validate_source_binding(source_binding)
    if PROTOCOL.exists() or OUT.exists() or AUDIT.exists():
        raise bench.ContractError("exclusive_study_artifact_exists")
    protocol = {
        "schema": "bran-information-matched-fm-protocol-v2", "status": "frozen_before_execution",
        "parameters": PARAMETERS, "source_binding": dict(source_binding),
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "privacy": dict(bench.FLAGS),
    }
    _exact_write(PROTOCOL, protocol)
    validate_protocol(protocol, sha(PROTOCOL))
    return protocol


def validate_protocol(protocol: Mapping[str, Any], pin: str) -> None:
    required = {"schema", "status", "parameters", "source_binding", "code_sha256", "privacy"}
    if type(protocol) is not dict or set(protocol) != required or protocol["schema"] != "bran-information-matched-fm-protocol-v2" \
            or protocol["status"] != "frozen_before_execution" or protocol["parameters"] != PARAMETERS \
            or protocol["privacy"] != bench.FLAGS or not isinstance(pin, str) or len(pin) != 64 or sha(PROTOCOL) != pin:
        raise bench.ContractError("protocol_binding_invalid")
    bench.validate_source_binding(protocol["source_binding"])
    if protocol["code_sha256"] != {name: sha(ROOT / name) for name in CODE}:
        raise bench.ContractError("code_closure_changed")


def execute(protocol: Mapping[str, Any], pin: str, context_loader: Callable[[], Mapping[str, Any]],
            fit_predict_fixed: Callable[..., Any], paired_counts: Callable[[Any, int, int], Any],
            *, auc_fn: Callable[..., float] = bench.fold_weighted_auc,
            bootstrap_auc_fn: Callable[..., Any] = bench.bootstrap_fold_weighted_auc,
            reauthenticate: Callable[[], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Run only caller-authenticated inputs; no representation is trained or extracted."""
    with fd_quiet():
        with _lifecycle_lock(), threadpool_limits(limits=2):
            return _execute_unlocked(protocol, pin, context_loader, fit_predict_fixed, paired_counts,
                                     auc_fn=auc_fn, bootstrap_auc_fn=bootstrap_auc_fn, reauthenticate=reauthenticate)


def _execute_unlocked(protocol: Mapping[str, Any], pin: str, context_loader: Callable[[], Mapping[str, Any]],
                   fit_predict_fixed: Callable[..., Any], paired_counts: Callable[[Any, int, int], Any],
                   *, auc_fn: Callable[..., float], bootstrap_auc_fn: Callable[..., Any],
                   reauthenticate: Callable[[], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Implementation entered only through ``execute``'s descriptor silence."""
    validate_protocol(protocol, pin)
    if OUT.exists(): raise bench.ContractError("exclusive_output_exists")
    state = {"phase": "authentication", "fold": None, "arm": None, "completed_readout_calls": 0,
             "readout_call_budget": 5 * len(bench.ARMS) * 26, "started": time.monotonic()}
    owned = False
    try:
        _exclusive_directory(OUT)
        owned = True
        def progress(fold: int, arm: str, completed: int, budget: int) -> None:
            state.update(phase="fixed_readouts", fold=fold, arm=arm, completed_readout_calls=completed,
                         readout_call_budget=budget)
            payload = {"phase": state["phase"], "fold": fold, "arm": arm,
                       "completed_readout_calls": completed, "readout_call_budget": budget,
                       "elapsed_seconds": round(time.monotonic() - state["started"], 1), **bench.FLAGS}
            temp = OUT / ".progress.tmp"
            _exact_write(temp, payload); os.replace(temp, OUT / "progress.json")
        if reauthenticate is not None:
            require_binding = reauthenticate()
            if bench.canonical_sha256(require_binding) != bench.canonical_sha256(protocol["source_binding"]):
                raise bench.ContractError("source_bytes_changed_before_run")
        context = context_loader()
        names = bench.validate_runtime_context(context, protocol["source_binding"])
        state["phase"] = "fixed_readouts"
        predictions = bench.fit_all_readouts(context, protocol["source_binding"], fit_predict_fixed, progress)
        state["phase"] = "paired_bootstrap"
        counts = paired_counts(context["folds"], bench.BOOTSTRAP_DRAWS, bench.BOOTSTRAP_SEED)
        result = bench.summarize(predictions, context["labels"], context["matched_support"], context["folds"], counts, names,
                                 auc_fn=auc_fn, bootstrap_auc_fn=bootstrap_auc_fn)
        aggregate = {"schema": "bran-information-matched-fm-aggregate-v2", "status": "completed",
                     "protocol_sha256": pin, "source_binding_sha256": bench.canonical_sha256(protocol["source_binding"]),
                     "results": result, **bench.FLAGS}
        bench.validate_result(result, names)
        # Reauthenticate the held frozen source immediately before any
        # completion artifact can be published.
        if reauthenticate is not None and bench.canonical_sha256(reauthenticate()) != bench.canonical_sha256(protocol["source_binding"]):
            raise bench.ContractError("source_bytes_changed_after_computation")
        _exact_write(OUT / "aggregate.json", aggregate)
        _exact_write(OUT / "manifest.json", {"protocol_sha256": pin, "aggregate_sha256": sha(OUT / "aggregate.json"),
                                               "source_binding_sha256": aggregate["source_binding_sha256"], **bench.FLAGS})
        return aggregate
    except Exception:
        if owned:
            _write_closed_failure(state)
        raise


def _remove_owned(path: Path) -> None:
    """Remove only a regular, private artifact created in this fresh directory."""
    if path.exists() or path.is_symlink():
        stat = os.lstat(path)
        if path.is_symlink() or not path.is_file() or stat.st_nlink != 1:
            raise bench.ContractError("terminal_artifact_cleanup_invalid")
        path.unlink()


def _write_closed_failure(state: Mapping[str, Any]) -> None:
    if not OUT.is_dir() or (OUT / "failure.json").exists():
        return
    # A half-written completion must never look successful.  These are known
    # fresh targets created solely by this invocation.
    for name in ("aggregate.json", "manifest.json", ".progress.tmp"):
        _remove_owned(OUT / name)
    _exact_write(OUT / "failure.json", {"schema": "bran-information-matched-fm-failure-v2", "status": "execution_failed",
                                          "phase": state["phase"], "fold": state["fold"], "arm": state["arm"],
                                          "completed_readout_calls": state["completed_readout_calls"], **bench.FLAGS})


def audit(protocol: Mapping[str, Any], pin: str, *, reauthenticate: Callable[[], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Authenticate only terminal artifacts and frozen source-byte/code pins."""
    with fd_quiet():
        with _lifecycle_lock():
            return _audit_unlocked(protocol, pin, reauthenticate=reauthenticate)


def _audit_unlocked(protocol: Mapping[str, Any], pin: str, *, reauthenticate: Callable[[], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    validate_protocol(protocol, pin)
    if reauthenticate is not None and bench.canonical_sha256(reauthenticate()) != bench.canonical_sha256(protocol["source_binding"]):
        raise bench.ContractError("source_bytes_changed_before_audit")
    if not OUT.is_dir():
        raise bench.ContractError("audit_output_missing")
    names = {path.name for path in OUT.iterdir()}
    if names in ({"failure.json"}, {"failure.json", "progress.json"}):
        failure = json.loads((OUT / "failure.json").read_text(encoding="utf-8"))
        expected_failure = {"schema", "status", "phase", "fold", "arm", "completed_readout_calls"} | set(bench.FLAGS)
        if set(failure) != expected_failure or failure["schema"] != "bran-information-matched-fm-failure-v2" \
                or failure["status"] != "execution_failed" or failure["phase"] not in {"authentication", "fixed_readouts", "paired_bootstrap"} \
                or failure["fold"] not in {None, 0, 1, 2, 3, 4} or failure["arm"] not in {None, *bench.ARMS} \
                or not isinstance(failure["completed_readout_calls"], int) or not all(failure[k] is v for k, v in bench.FLAGS.items()):
            raise bench.ContractError("failure_terminal_invalid")
        return {"schema": "bran-information-matched-fm-audit-v2", "status": "authenticated_execution_failure",
                "failure_sha256": sha(OUT / "failure.json"), **bench.FLAGS}
    if names != {"aggregate.json", "manifest.json", "progress.json"}:
        raise bench.ContractError("audit_terminal_not_exclusive")
    aggregate = json.loads((OUT / "aggregate.json").read_text(encoding="utf-8"))
    expected = {"schema", "status", "protocol_sha256", "source_binding_sha256", "results"} | set(bench.FLAGS)
    if set(aggregate) != expected or aggregate["schema"] != "bran-information-matched-fm-aggregate-v2" or aggregate["status"] != "completed" \
            or aggregate["protocol_sha256"] != pin or aggregate["source_binding_sha256"] != bench.canonical_sha256(protocol["source_binding"]):
        raise bench.ContractError("aggregate_binding_invalid")
    if not all(aggregate[key] is value for key, value in bench.FLAGS.items()):
        raise bench.ContractError("aggregate_privacy_flags_invalid")
    bench.validate_result(aggregate["results"], protocol["source_binding"]["endpoint_names"])
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    if manifest != {"protocol_sha256": pin, "aggregate_sha256": sha(OUT / "aggregate.json"),
                    "source_binding_sha256": aggregate["source_binding_sha256"], **bench.FLAGS}:
        raise bench.ContractError("manifest_binding_invalid")
    if reauthenticate is not None and bench.canonical_sha256(reauthenticate()) != bench.canonical_sha256(protocol["source_binding"]):
        raise bench.ContractError("source_bytes_changed_after_audit")
    return {"schema": "bran-information-matched-fm-audit-v2", "status": "authenticated",
            "protocol_sha256": pin, "aggregate_sha256": sha(OUT / "aggregate.json"),
            "source_bytes_authenticated_independently": True,
            "historical_score_equality_required": False, **bench.FLAGS}


def audit_terminal(protocol: Mapping[str, Any], pin: str, *, reauthenticate: Callable[[], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Write one separate, exclusive audit receipt after an authenticated terminal."""
    with fd_quiet():
        with _lifecycle_lock():
            return _audit_terminal_unlocked(protocol, pin, reauthenticate=reauthenticate)


def _audit_terminal_unlocked(protocol: Mapping[str, Any], pin: str, *, reauthenticate: Callable[[], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    if AUDIT.exists():
        raise bench.ContractError("exclusive_audit_artifact_exists")
    receipt = _audit_unlocked(protocol, pin, reauthenticate=reauthenticate)
    _exclusive_directory(AUDIT)
    _exact_write(AUDIT / "audit.json", receipt)
    return receipt


def _metadata_components() -> tuple[dict[str, Any], dict[str, Any]]:
    """Recheck byte-addressable prerequisites without reloading private rows."""
    import run_bran_named_fm_cache_v1 as cache
    import run_bran_blood_learning_curve_v1 as blood

    safe, fm, retained = cache.safe, cache.fm, cache.retained
    cache_pin_before = safe.sha(cache.PROTOCOL)
    cache_record = cache.verify(cache_pin_before)  # verify already authenticates load_protocol.
    cache_pin_after = safe.sha(cache.PROTOCOL)
    bench.require(cache_pin_before == cache_pin_after, "cache_protocol_changed_during_verify")
    cache_protocol = cache.read(cache.PROTOCOL)
    bench.require(safe.sha(cache.PROTOCOL) == cache_pin_before, "cache_protocol_changed_during_read")
    description = cache_protocol["source"]
    source = retained.native.source
    source_receipt = source.io.source_receipt()
    cache_binding = fm._cache_binding(fm.DEFAULT_CACHE_MANIFEST)
    expected = description["learning_source"]
    bench.require(cache_binding["outer_fold_sha256"] == expected["outer_fold_sha256"]
                  and list(cache_binding["inner_fold_sha256"]) == list(expected["inner_fold_sha256"]),
                  "cache_fold_binding_invalid")
    checkpoint_hashes = description["retained"]["native_source"]["checkpoint_sha256"]
    for fold in range(5):
        key = "fold" + str(fold)
        bench.require(safe.sha(source.PRIVATE / (key + ".pt")) == checkpoint_hashes[key],
                      "retained_checkpoint_bytes_invalid")
    # Bind every project-local import reachable from this adapter, the cache
    # receipt, the retained state path, and the raw blood-control helper.
    seeds = set(CODE) | set(cache.CODE) | set(cache_protocol["code_sha256"]) | set(getattr(blood, "CODE", ()))
    seeds.add("run_bran_blood_learning_curve_v1.py")
    names = cache.original.closure(ROOT, seeds)
    source_code = source.io.code_closure(names)
    names |= set(source_code)
    code_hashes = {name: safe.sha(ROOT / name) for name in sorted(names)}
    bench.require(all(code_hashes.get(name) == digest for name, digest in source_code.items()),
                  "source_code_closure_changed")
    source_auth = description["retained"]["native_source"]["source"]["authentication"]
    canonical = dict(source_auth["canonical_source_hashes"])
    # Receipts alone do not prove that the current canonical source bytes
    # remain unchanged. Hash them without parsing any patient records.
    from patient_atlas_v6_2_expanded_endpoint_atlas_support_materializer import authenticate_canonical_dataset_sources
    source_protocol = source.io.joint.old_protocol()
    actual_canonical = authenticate_canonical_dataset_sources(Path(source_protocol["data_roots"]["dataset_root"]))
    bench.require(dict(actual_canonical) == canonical, "canonical_source_bytes_changed")
    parent_receipts = {"cache_protocol": cache_pin_before,
                       "cache_result": safe.sha(cache.OUT / "result.json"),
                       "cache_success": safe.sha(cache.OUT / "success.json"),
                       "cache_manifest": cache_binding["manifest_sha256"],
                       "cache_current_inputs": cache_binding["current_inputs_sha256"],
                       "cache_row_order": cache_binding["row_order_sha256"],
                       "source_io_receipt": bench.canonical_sha256(source_receipt),
                       **{str(name): str(digest) for name, digest in source_receipt["pins"].items()}}
    return {"cache": cache, "fm": fm, "retained": retained, "source": source, "description": description,
            "cache_binding": cache_binding, "cache_record": cache_record, "blood_design": blood.blood_design,
            "expected": expected, "checkpoint_hashes": checkpoint_hashes, "code_hashes": code_hashes,
            "canonical_source": canonical, "parent_receipts": parent_receipts}, {}


def _live_components() -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate source bytes through established local code, without score replay.

    This intentionally consumes only the historical cache/protocol identity and
    its frozen bytes.  It never invokes any historical AUROC comparison or
    tolerance gate; those gates remain owned by their historical runners.
    """
    parts, _ = _metadata_components()
    cache, fm, retained, source = parts["cache"], parts["fm"], parts["retained"], parts["source"]
    safe, description, cache_binding, expected = cache.safe, parts["description"], parts["cache_binding"], parts["expected"]
    value = source.io.load_context()
    ctx, folds, c0, cm0, eligible, r0, rm, ages, names = value
    endpoints, row_ids = fm._validate_context({**description, "cache": cache_binding}, source, *value)
    inner = fm._inner_assignments(description, source, ctx, folds)
    actual_outer = np.asarray(ctx["outer_assignment"], int)
    bench.require(np.array_equal(np.asarray(folds, int), actual_outer), "returned_outer_assignment_changed")
    from patient_atlas_v6_2_expanded_endpoint_evaluation import EXACT_OUTER_FOLD_HASH
    bench.require(expected["outer_fold_sha256"] == EXACT_OUTER_FOLD_HASH
                  and dict(ctx["source_hashes"]) == parts["canonical_source"], "actual_outer_source_binding_invalid")
    current = fm.current_inputs_sha256(c0, cm0, eligible, r0, rm, ages, names)
    require = bench.require
    require(current == cache_binding["current_inputs_sha256"] and fm.row_order_sha256(row_ids) == cache_binding["row_order_sha256"],
            "current_input_binding_invalid")
    checkpoint_hashes = parts["checkpoint_hashes"]
    def array_sha(value: Any) -> str:
        array = np.ascontiguousarray(np.asarray(value))
        digest = hashlib.sha256(); digest.update(str(array.dtype).encode()); digest.update(repr(array.shape).encode()); digest.update(array.tobytes())
        return digest.hexdigest()
    clinical_values_sha, clinical_masks_sha = array_sha(c0), array_sha(cm0)
    eligible_sha, age_sha = array_sha(eligible), array_sha(ages)
    shared_input_sha = bench.matched_input_digest(clinical_values_sha, clinical_masks_sha, eligible_sha, age_sha)
    # The Labrador cache was built from the released ordered clinical names
    # with only canonical positions 0..37 eligible; bind that contract without
    # exposing the names or any row values.
    labrador_input_sha = bench.canonical_sha256({"ordered_feature_names_sha256": array_sha(np.asarray(names)),
                                                 "eligible_positions": "0_through_37", "cache_current_inputs_sha256": current})
    representation = {arm: bench.canonical_sha256(checkpoint_hashes) for arm in bench.BRAN_ARMS}
    for arm in bench.FM_AGE_ARMS + bench.FM_CLINICAL_AGE_ARMS:
        base = arm.removesuffix("_clinical_age").removesuffix("_age")
        representation[arm] = cache_binding["embedding_files"][base]["sha256"]
    for arm in bench.LABRADOR_ARMS:
        representation[arm] = cache_binding["embedding_files"]["labrador"]["sha256"]
    for arm in bench.CONTROL_ARMS:
        representation[arm] = current
    files = {"cache_protocol": safe.sha(cache.PROTOCOL), "cache_result": safe.sha(cache.OUT / "result.json"),
             "cache_manifest": cache_binding["manifest_sha256"], **{key: value for key, value in checkpoint_hashes.items()}}
    files["runtime"] = bench.canonical_sha256(source.io.runtime())
    for arm in fm.ARMS:
        files["embedding_" + arm] = cache_binding["embedding_files"][arm]["sha256"]
        files["artifact_" + arm] = cache_binding["artifact_files"][arm]["sha256"]
    provenance = {arm: {"upstream_supervision": "retained BRAN outer-fold supervised encoder; inner-validation label exposure is not undone by fixed-probe refitting",
                        "training_exposure": "outer-fold checkpoint; coordinate frame is fold-specific"} for arm in bench.BRAN_ARMS}
    for arm in bench.FM_AGE_ARMS + bench.FM_CLINICAL_AGE_ARMS:
        provenance[arm] = {"upstream_supervision": "qualified frozen upstream representation; not matched to BRAN supervision",
                           "training_exposure": "existing frozen cache; no V2 encoder fitting or extraction"}
    for arm in bench.LABRADOR_ARMS:
        provenance[arm] = {"upstream_supervision": "qualified frozen clinical/blood Labrador representation; slots below 38 only",
                           "training_exposure": "existing frozen cache; no V2 encoder fitting or extraction"}
    for arm in bench.CONTROL_ARMS:
        provenance[arm] = {"upstream_supervision": "no encoder; fixed raw-control representation",
                           "training_exposure": "V2 fixed C=1 readout only"}
    binding = {"schema": "bran-information-matched-fm-source-binding-v2", "endpoint_names": list(endpoints),
               "outer_fold_sha256": expected["outer_fold_sha256"], "inner_fold_sha256": list(expected["inner_fold_sha256"]),
               "row_order_sha256": fm.row_order_sha256(row_ids), "shared_clinical_values_sha256": clinical_values_sha,
               "shared_clinical_masks_sha256": clinical_masks_sha, "shared_eligible_masks_sha256": eligible_sha, "shared_age_sha256": age_sha,
               "shared_input_sha256": shared_input_sha, "labrador_input_sha256": labrador_input_sha,
               "bran_checkpoints_sha256": bench.canonical_sha256(checkpoint_hashes),
               "fm_cache_manifest_sha256": cache_binding["manifest_sha256"], "representation_sha256": representation,
               "arm_input_sha256": {arm: shared_input_sha for arm in bench.ARMS}, "arm_provenance": provenance,
               "source_files_sha256": files, "transitive_code_sha256": parts["code_hashes"],
               "canonical_source_sha256": parts["canonical_source"], "parent_receipts_sha256": parts["parent_receipts"],
               "all_source_bytes_authenticated": True, "retinal_encoders_frozen": True,
               "retinal_extraction_performed": False, "historical_score_equality_required": False}
    bench.validate_source_binding(binding)
    # Context loading itself is within the held lock, but still prove that all
    # byte-addressable prerequisites remained pinned across that load.
    _metadata_reauthenticate(binding)
    return binding, {"cache": cache, "fm": fm, "retained": retained, "source": source, "description": description,
                     "value": value, "context": ctx, "inner": inner, "cache_binding": cache_binding,
                     "cache_record": parts["cache_record"], "blood_design": parts["blood_design"], "current_input_sha256": current}


def _metadata_reauthenticate(binding: Mapping[str, Any]) -> Mapping[str, Any]:
    """Prove the frozen held-context receipt still names identical bytes."""
    fresh, _ = _metadata_components()
    files = {"cache_protocol": fresh["cache"].safe.sha(fresh["cache"].PROTOCOL),
             "cache_result": fresh["cache"].safe.sha(fresh["cache"].OUT / "result.json"),
             "cache_manifest": fresh["cache_binding"]["manifest_sha256"],
             "runtime": bench.canonical_sha256(fresh["source"].io.runtime()),
             **{key: value for key, value in fresh["checkpoint_hashes"].items()}}
    for arm in fresh["fm"].ARMS:
        files["embedding_" + arm] = fresh["cache_binding"]["embedding_files"][arm]["sha256"]
        files["artifact_" + arm] = fresh["cache_binding"]["artifact_files"][arm]["sha256"]
    bench.require(files == binding["source_files_sha256"]
                  and fresh["code_hashes"] == binding["transitive_code_sha256"]
                  and fresh["canonical_source"] == binding["canonical_source_sha256"]
                  and fresh["parent_receipts"] == binding["parent_receipts_sha256"],
                  "source_bytes_changed_during_held_context")
    return binding


def live_adapter() -> tuple[dict[str, Any], Callable[[], Mapping[str, Any]], Callable[..., Any], Callable[[Any, int, int], Any], Callable[..., float], Callable[..., Any], Callable[[], Mapping[str, Any]]]:
    """Return the executable V2 adapter using only authenticated local contracts."""
    with fd_quiet():
        with _lifecycle_lock():
            return _live_adapter_unlocked()


def _live_adapter_unlocked() -> tuple[dict[str, Any], Callable[[], Mapping[str, Any]], Callable[..., Any], Callable[[Any, int, int], Any], Callable[..., float], Callable[..., Any], Callable[[], Mapping[str, Any]]]:
    """Build the adapter while the caller already owns the shared lifecycle lock."""
    import torch
    torch.set_num_threads(2)
    binding, parts = _live_components()
    fm, retained, source, blood_design = parts["fm"], parts["retained"], parts["source"], parts["blood_design"]
    native = retained.native
    ctx, folds, c0, cm0, eligible, r0, rm, ages, names = parts["value"]
    endpoints, inner, cache_binding = tuple(binding["endpoint_names"]), parts["inner"], parts["cache_binding"]
    arrays = fm.load_fm_cache(cache_binding, expected_current_inputs_sha256=parts["current_input_sha256"],
                              expected_row_order_sha256=binding["row_order_sha256"],
                              expected_outer_fold_sha256=binding["outer_fold_sha256"],
                              expected_inner_fold_sha256=binding["inner_fold_sha256"])
    labels = {name: np.asarray(ctx["labels_by_source"][name]) for name in endpoints}
    observed = {name: np.asarray(ctx["observed_by_source"][name], bool) for name in endpoints}
    def fold_inputs(fold: int):
        train = np.flatnonzero(folds != fold)
        transform = source.base.FoldTransform(c0, cm0, eligible, r0, rm, ages, train)
        c, cm, r, age = transform.apply(c0, cm0, eligible, r0, rm, ages)
        # Ineligible/masked payload is physically erased before every retained
        # encode and every raw/clinical concatenation.  The original fold-only
        # normalizers remain those authenticated by retained.load_initial.
        visible = cm & eligible & np.isfinite(c)
        retinal_visible = np.asarray(rm, bool)[:, None]
        return train, transform, np.where(visible, c, 0.), visible, np.where(retinal_visible, r, 0.), age

    route_available = np.zeros(len(folds), bool)
    for fold in range(5):
        train, transform, c, cm, r, age = fold_inputs(fold)
        model = retained.load_initial(fold, transform, parts["description"]["retained"])
        native_scores = native.kernel.predict_native(model, c, cm, r, rm, age)
        bench.require(set(native_scores) == {"both", "clinical", "retinal"}, "native_route_schema_invalid")
        test = folds == fold
        route_available[test] = np.logical_and.reduce([np.isfinite(np.asarray(native_scores[route])[test]).all(axis=1)
                                                        for route in ("both", "clinical", "retinal")])
    common = fm._support_mask(rm, c0, cm0, eligible) & route_available
    matched_support = {name: fm.common_evaluation_mask(observed[name], common, labels[name]) for name in endpoints}

    def design_factory(fold: int) -> Mapping[str, Any]:
        bench.require(type(fold) is int and 0 <= fold < 5, "outer_fold_design_request_invalid")
        train, transform, c, cm, r, age = fold_inputs(fold)
        model = retained.load_initial(fold, transform, parts["description"]["retained"])
        routes = source.lineage._state_routes(model, c, cm, r, rm, age)
        bench.require(set(routes) == {"both", "clinical", "retinal"} and all(np.asarray(value).shape == (len(folds), 192) for value in routes.values()),
                      "retained_state_route_invalid")
        designs = {"bran_combined_state": routes["both"],
                   "bran_clinical_state": routes["clinical"],
                   "bran_retinal_state": routes["retinal"],
                   "blood_age": blood_design(c, cm, age),
                   "raw_clinical_age": np.c_[c, cm, age],
                   "raw_clinical_retinal_age": np.c_[c, cm, r, rm, age]}
        for base in fm.ARMS:
            if base == "labrador":
                designs["labrador_clinical_age"] = np.c_[arrays[base], age]
                designs["labrador_clinical_retinal_age"] = np.c_[arrays[base], c, cm, r, rm, age]
            else:
                designs[base + "_age"] = np.c_[arrays[base], age]
                designs[base + "_clinical_age"] = np.c_[arrays[base], c, cm, age]
        bench.require(set(designs) == set(bench.ARMS), "fold_design_arm_set_invalid")
        return designs

    def reauthenticate() -> Mapping[str, Any]:
        return _metadata_reauthenticate(binding)

    def context_loader() -> Mapping[str, Any]:
        # Metadata authentication is sufficient here: the private context was
        # already authenticated against its cache input digest and canonical
        # source receipt before it was held in this closure.
        reauthenticate()
        return {"folds": np.asarray(folds), "labels": labels, "observed": observed, "matched_support": matched_support,
                "inner_folds": inner, "design_factory": design_factory, "input_sha256": binding["shared_input_sha256"],
                "outer_fold_sha256": binding["outer_fold_sha256"], "row_order_sha256": binding["row_order_sha256"]}

    def fit_predict_fixed(x_train, y_train, observed_train, x_test):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        import warnings
        mask = np.asarray(observed_train, bool); y = np.asarray(y_train)
        bench.require(mask.shape == y.shape == (len(x_train),) and mask.sum() >= 20
                      and set(np.unique(y[mask])) == {0, 1}, "fixed_probe_training_support_invalid")
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1., penalty="l2", solver="lbfgs", max_iter=5000))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(np.asarray(x_train)[mask], y[mask])
        bench.require(not any(issubclass(item.category, ConvergenceWarning) for item in caught), "fixed_probe_nonconvergence")
        return model.predict_proba(x_test)[:, 1]

    def counts(value, draws, seed):
        bench.require(draws == bench.BOOTSTRAP_DRAWS and seed == bench.BOOTSTRAP_SEED, "bootstrap_configuration_changed")
        return source.ev.paired_counts(value, draws=draws, seed=seed)

    return binding, context_loader, fit_predict_fixed, counts, source.base.fold_weighted_auc, source.base._weighted_auc_draws, reauthenticate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--audit", action="store_true")
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args(argv)
    operation = "prepare" if args.prepare else "run" if args.run else "audit"
    ok = False; detail = "execution_failed"
    with fd_quiet():
        try:
            with _lifecycle_lock(), threadpool_limits(limits=2):
                if args.prepare:
                    binding, *_ = _live_adapter_unlocked()
                    _prepare_unlocked(binding); detail = "protocol_prepared"
                else:
                    if not isinstance(args.protocol_sha256, str):
                        raise bench.ContractError("protocol_sha256_required")
                    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8")); validate_protocol(protocol, args.protocol_sha256)
                    if operation == "run":
                        binding, loader, fit, counts, auc, draws, reauth = _live_adapter_unlocked()
                        _execute_unlocked(protocol, args.protocol_sha256, loader, fit, counts, auc_fn=auc, bootstrap_auc_fn=draws, reauthenticate=reauth)
                        detail = "completed"
                    else:
                        _audit_terminal_unlocked(protocol, args.protocol_sha256,
                                                 reauthenticate=lambda: _metadata_reauthenticate(protocol["source_binding"]))
                        detail = "audited"
            ok = True
        except Exception as exc:
            detail = "failed_closed_" + type(exc).__name__
    print(json.dumps({"operation": operation, "status": detail, **bench.FLAGS}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
