"""Exclusive local combined multisource/missingness comparison and fixed-model audit."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from bran_joint_lab_cache_v1 import valid_sha
from run_bran_source_linkage_audit_v1 import sha, exclusive_json
import bran_multisource_mask_contract_v1 as contract
import bran_multisource_mask_experiment_v1 as experiment
import bran_multisource_mask_training_v1 as training
import run_bran_native_rehearsal_v1 as external_origin
from bran_native_rehearsal_batches_v1 import prepare_sources
from bran_supervised_mask_tasks_v1 import SCREEN_PATTERNS
import run_bran_retinal_input_bridge_v3 as bridge

origin = experiment.origin
ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "BRAN_MULTISOURCE_MASK_PROTOCOL_V1.json"
OUT = ROOT / "BRAN_MULTISOURCE_MASK_V1"
AUDIT = ROOT / "BRAN_MULTISOURCE_MASK_AUDIT_V1"
PRIVATE = ROOT / "private_artifacts/bran_multisource_mask_v1"
LOCK = Path("/private/tmp/bran_retinal_extraction_v1.lock")
PARAMETERS = {key: copy.deepcopy(value) for key, value in experiment.old.PARAMETERS.items()
              if key not in ("external_weights", "external_tasks", "student_definition", "seed_base")}
PARAMETERS.update(seed_base=95001, task_mask_seed_offset=200000,
    combined_weights={"continued": 0., "student": .5}, external_seed_offset=100000,
    student_definition="combined masked screening, masked CBC and source-balanced external CBC loss; no distillation",
    extra_screen_cycle=list(SCREEN_PATTERNS),
    external_tasks="alternate wholeCBC/partialCBC; source then person then episode balanced",
    extra_cbc_cycle=["partial_cbc", "whole_cbc_no_retina"],
    cbc_evaluation_patterns=list(contract.EVALPATTERNS),
    no_retina_priority_gate="supported Hb/platelet/WBC overall MAE point-nonworse versus initial and control in both patterns",
    retinal_inputs="unchanged historical native inputs after authenticated V3 bridge equivalence",
    historical_production_proven=False,
    uncertainty={"alpha": .1, "minimum_calibration": 20,
        "split_salt": experiment.splitting.SALT, "calibration_fraction": .5,
        "split_scope": "outcome-blind halves within each checkpoint's own heldout fold",
        "patterns": list(contract.EVALPATTERNS), "versions": list(experiment.metrics.VERSIONS),
        "bootstrap_draws": 1000, "bootstrap_seed": 94701,
        "bootstrap_scope": "scoring halves within outer fold; fixed fitted models and radii",
        "support": "same calibrated scoring scope for all three native models",
        "clinical_coverage_guarantee": False, "changes_advancement_gates": False},
    audit="reload fixed candidate checkpoints and replay all aggregates; raw reference heads refit")
CODE = (
    "run_bran_multisource_mask_v1.py", "test_run_bran_multisource_mask_v1.py",
    "bran_multisource_mask_contract_v1.py", "test_bran_multisource_mask_contract_v1.py",
    "bran_multisource_mask_experiment_v1.py", "test_bran_multisource_mask_experiment_v1.py",
    "bran_multisource_mask_training_v1.py", "test_bran_multisource_mask_training_v1.py",
    "BRAN_MULTISOURCE_MASK_DESIGN_V1.md",
    "run_bran_supervised_missingness_v1.py", "test_run_bran_supervised_missingness_v1.py",
    "bran_supervised_mask_experiment_v1.py", "test_bran_supervised_mask_experiment_v1.py",
    "bran_supervised_mask_contract_v1.py", "test_bran_supervised_mask_contract_v1.py",
    "bran_supervised_mask_training_v1.py", "test_bran_supervised_mask_training_v1.py",
    "bran_supervised_mask_tasks_v1.py", "test_bran_supervised_mask_tasks_v1.py",
    "test_bran_supervised_mask_completion_inputs_v1.py", "test_bran_supervised_mask_bridge_receipt_v1.py",
    "bran_supervised_mask_uncertainty_v1.py", "test_bran_supervised_mask_uncertainty_v1.py",
    "bran_native_cbc_calibration_metrics_v1.py", "test_bran_native_cbc_calibration_metrics_v1.py",
    "bran_native_calibration_split_v1.py", "test_bran_native_calibration_split_v1.py",
    "run_bran_native_rehearsal_v1.py", "test_run_bran_native_rehearsal_v1.py",
    "bran_native_rehearsal_kernel_v1.py", "test_bran_native_rehearsal_kernel_v1.py",
    "bran_missingness_stress_v1.py", "bran_missingness_stress_metrics_v1.py",
    "test_bran_missingness_stress_v1.py", "test_bran_missingness_stress_metrics_v1.py",
    "BRAN_SUPERVISED_MISSINGNESS_DESIGN_V1.md",
)
PHASES = ("authentication", "baseline_replay", "external_loading", "external_batch_preparation",
          "adapt_continued", "adapt_student", "checkpoint_reload",
          "frozen_inference_initial", "frozen_inference_continued", "frozen_inference_student",
          "aggregate_bootstrap", "calibration_scoring", "audit_replay", "publication")


def require(ok):
    if not ok:
        raise ValueError("multisource_mask_runner_contract_failed")


def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def regular(path, mode=None):
    require(path.is_file() and not path.is_symlink())
    if mode is not None:
        require(path.stat().st_mode & 0o777 == mode)


def terminal_inventory(directory, names):
    require(directory.is_dir() and not directory.is_symlink())
    require({path.name for path in directory.iterdir()} == set(names) | {"progress.json"})
    for name in (*names, "progress.json"):
        regular(directory / name)
    require(same(json.loads((directory / "progress.json").read_text()),
                 {"phase": "publication", "fold": None, "step": None,
                  "patient_level_output_emitted": False}))


def authenticate_bridge(protocol_pin, audit_pin):
    require(valid_sha(protocol_pin) and valid_sha(audit_pin))
    for directory in (bridge.OUT, bridge.AUDIT):
        require(directory.is_dir() and not directory.is_symlink() and not (directory / "failure.json").exists())
    for path in (bridge.PROTOCOL, bridge.OUT / "aggregate.json", bridge.OUT / "aggregate.manifest.json",
                 bridge.AUDIT / "audit.json", bridge.AUDIT / "audit.manifest.json"):
        regular(path)
    require(sha(bridge.PROTOCOL) == protocol_pin and sha(bridge.AUDIT / "audit.json") == audit_pin)
    bp = json.loads(bridge.PROTOCOL.read_text())
    bridge.validate_protocol(bp)
    a = json.loads((bridge.OUT / "aggregate.json").read_text())
    bridge.validate_result(a, bp)
    ap = sha(bridge.OUT / "aggregate.json")
    require(json.loads((bridge.OUT / "aggregate.manifest.json").read_text()) ==
            {"protocol_sha256": protocol_pin, "artifact_sha256": ap})
    expected = {"schema": "bran-retinal-input-bridge-audit-v3", "status": "authenticated",
        "protocol_sha256": protocol_pin, "aggregate_sha256": ap,
        "retinal_audit_sha256": bp["retinal_audit_sha256"], "fold_authentication": a["fold_authentication"],
        "all_aggregates_replayed": True, "patient_level_output_emitted": False, "model_promoted": False}
    require(same(json.loads((bridge.AUDIT / "audit.json").read_text()), expected))
    require(json.loads((bridge.AUDIT / "audit.manifest.json").read_text()) ==
            {"protocol_sha256": protocol_pin, "artifact_sha256": audit_pin})
    require(set(a["equivalence"]) == {"pooled_input", "screening", "whole_cbc", "replacement_review_eligible"}
            and all(value is True for value in a["equivalence"].values()))
    require(sha(bridge.PROTOCOL) == protocol_pin and sha(bridge.AUDIT / "audit.json") == audit_pin
            and sha(bridge.OUT / "aggregate.json") == ap)
    require(not any((d / "failure.json").exists() for d in (bridge.OUT, bridge.AUDIT)))
    return {"protocol_sha256": protocol_pin, "audit_sha256": audit_pin, "aggregate_sha256": ap,
            "equivalence": a["equivalence"]}, bp


def prepare(bridge_protocol_pin, bridge_audit_pin):
    receipt, bp = authenticate_bridge(bridge_protocol_pin, bridge_audit_pin)
    base = origin.prepare()
    require(same(base["native_source"], bp["native_source"]))
    external = external_origin.prepare()
    require(same(base["native_source"], external["native_source"]))
    require(base["native_aggregate_sha256"] == external["native_aggregate_sha256"]
            and base["native_audit_sha256"] == external["native_audit_sha256"]
            and base["raw_reference_sha256"] == external["raw_reference_sha256"]
            and same(base["runtime"], external["runtime"]))
    source_receipt = {key: external[key] for key in ("qualification_protocol_sha256",
        "qualification_audit_sha256", "sources", "joint_registry_indices")}
    closure = {**base["code_sha256"], **bp["code_sha256"], **external["code_sha256"],
               **{name: sha(ROOT / name) for name in CODE}}
    return {"schema": "bran-multisource-mask-protocol-v1", "status": "frozen_before_execution",
        "parameters": PARAMETERS, "native_source": base["native_source"],
        "native_aggregate_sha256": base["native_aggregate_sha256"],
        "native_audit_sha256": base["native_audit_sha256"], "raw_reference_sha256": base["raw_reference_sha256"],
        "retinal_bridge": receipt, "external_source": source_receipt,
        "code_sha256": closure, "runtime": base["runtime"]}


def validate_protocol(p):
    require(type(p) is dict and same(p, prepare(p["retinal_bridge"]["protocol_sha256"], p["retinal_bridge"]["audit_sha256"])))


def progress(destination, state, phase, fold=None, step=None):
    require(phase in PHASES)
    require(fold is None or (type(fold) is int and 0 <= fold < 5))
    require(step is None or (type(step) is int and 0 < step <= 1500))
    state["phase"] = phase
    value = {"phase": phase, "fold": fold, "step": step, "patient_level_output_emitted": False}
    temp = destination / "progress.tmp"
    temp.write_text(json.dumps(value) + "\n")
    os.replace(temp, destination / "progress.json")


def original_bundle(p, fold):
    import torch
    path = origin.native.source.PRIVATE / ("fold" + str(fold) + ".pt")
    regular(path, 0o600)
    pin = p["native_source"]["checkpoint_sha256"]["fold" + str(fold)]
    require(sha(path) == pin)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    require(sha(path) == pin)
    return bundle


def read_checkpoint(p, pin, fold, checkpoint_pin):
    import torch
    path = PRIVATE / ("fold" + str(fold) + ".pt")
    require(PRIVATE.is_dir() and not PRIVATE.is_symlink() and PRIVATE.stat().st_mode & 0o777 == 0o700)
    regular(path, 0o600)
    require(valid_sha(checkpoint_pin) and sha(path) == checkpoint_pin)
    value = torch.load(path, map_location="cpu", weights_only=False)
    contract.validate_bundle(value, original_bundle(p, fold), p, pin, fold)
    require(sha(path) == checkpoint_pin)
    return value


def calibration_metadata(pin, checkpoints, split_auth):
    require(valid_sha(pin) and type(checkpoints) is dict
            and set(checkpoints) == {"fold" + str(f) for f in range(5)}
            and all(valid_sha(value) for value in checkpoints.values()))
    require(type(split_auth) is dict and set(split_auth) == {"patient_order_sha256", "roles_sha256"}
            and all(valid_sha(value) for value in split_auth.values()))
    return {"protocol_sha256": pin, "checkpoint_sha256": checkpoints,
        "split_authentication": split_auth, "radius_axes": ["pattern", "fold", "version", "cbc_field"],
        "patterns": list(contract.EVALPATTERNS), "versions": list(experiment.metrics.VERSIONS),
        "cbc_fields": list(contract.CBC_FIELDS), "parameters": PARAMETERS["uncertainty"]}


def read_calibration(pin, checkpoints, split_auth, calibration_pin):
    path = PRIVATE / "calibration.npz"
    require(PRIVATE.is_dir() and not PRIVATE.is_symlink() and PRIVATE.stat().st_mode & 0o777 == 0o700)
    regular(path, 0o600)
    require(valid_sha(calibration_pin) and sha(path) == calibration_pin)
    with np.load(path, allow_pickle=False) as bundle:
        require(set(bundle.files) == {"radii", "metadata_json"})
        radii = bundle["radii"]
        radii.setflags(write=False)
        experiment.uncertainty.validate_radii(radii)
        metadata = bundle["metadata_json"]
        require(metadata.dtype == np.uint8 and metadata.ndim == 1)
        require(same(json.loads(metadata.tobytes()), calibration_metadata(pin, checkpoints, split_auth)))
    require(sha(path) == calibration_pin)
    radii.setflags(write=False)
    return radii


def compute(p, pin, *, destination, state, refit, expected_checkpoints=None, expected_calibration=None):
    import torch
    require(type(refit) is bool)
    checkpoints = {}
    calibration_pin = None
    external_pool = None

    def notify(phase, fold=None, step=None):
        progress(destination, state, phase, fold, step)

    def baseline_ready(value):
        require(same(value, experiment.BASELINE))
        if refit:
            exclusive_json(OUT / "baseline_replay.json", value)
        else:
            require(same(json.loads((OUT / "baseline_replay.json").read_text()), value))

    def provider(fold, transform, initial, arrays, names):
        nonlocal external_pool
        key = "fold" + str(fold)
        path = PRIVATE / (key + ".pt")
        models = {}
        if refit:
            require(type(names) is tuple and len(names) == len(set(names)) == 59
                    and all(type(name) is str and name for name in names))
            indices = p["external_source"]["joint_registry_indices"]
            require(all(type(j) is int and 0 <= j < 48 and names[j] == field
                        for field, j in indices.items()))
            require(tuple(names.index(field) for field in contract.CBC_FIELDS) == arrays[-1])
            if external_pool is None:
                notify("external_loading", fold)
                external_pool = {source: external_origin.qualification.load_one(source,
                    p["external_source"]["sources"][source]) for source in external_origin.qualification.TASKS}
            notify("external_batch_preparation", fold)
            prepared = prepare_sources(external_pool, names, transform.clinical_median,
                transform.clinical_iqr, transform.age_mean, transform.age_scale)
            for version, weight in PARAMETERS["combined_weights"].items():
                notify("adapt_" + version, fold)
                models[version] = training.adapt(initial, *arrays, prepared, seed=95001 + fold,
                    combined_weight=weight, steps=1500, batch_size=96,
                    progress=lambda step: notify("adapt_" + version, fold, step))
            del prepared
            bundle = {version: model.state_dict() for version, model in models.items()}
            bundle.update({name: getattr(transform, name) for name in origin.NORMALIZERS})
            bundle.update(protocol_sha256=pin, initial_checkpoint_sha256=p["native_source"]["checkpoint_sha256"][key],
                fold=fold, endpoint_names=p["native_source"]["source"]["endpoint_names"], cbc_fields=list(contract.CBC_FIELDS))
            contract.validate_bundle(bundle, original_bundle(p, fold), p, pin, fold)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                torch.save(bundle, handle)
            checkpoints[key] = sha(path)
        else:
            checkpoints[key] = expected_checkpoints[key]
        notify("checkpoint_reload", fold)
        loaded = read_checkpoint(p, pin, fold, checkpoints[key])
        reloaded = {}
        for version in ("continued", "student"):
            reloaded[version] = copy.deepcopy(initial)
            reloaded[version].load_state_dict(loaded[version], strict=True)
            reloaded[version].eval()
            if refit:
                require(all(torch.equal(models[version].state_dict()[k], v) for k, v in loaded[version].items()))
            else:
                models[version] = copy.deepcopy(reloaded[version])
        return models, reloaded

    def calibration_ready(radii, split_auth):
        nonlocal calibration_pin
        require(calibration_pin is None)
        experiment.uncertainty.validate_radii(radii)
        metadata = calibration_metadata(pin, checkpoints, split_auth)
        path = PRIVATE / "calibration.npz"
        if refit:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(handle, radii=radii,
                    metadata_json=np.frombuffer(json.dumps(metadata, sort_keys=True, allow_nan=False).encode(), np.uint8))
            calibration_pin = sha(path)
        else:
            calibration_pin = expected_calibration
        replay = read_calibration(pin, checkpoints, split_auth, calibration_pin)
        require(np.array_equal(replay, radii, equal_nan=True))

    value = experiment.evaluate(p, pin, provider=provider, baseline_ready=baseline_ready,
                               calibration_ready=calibration_ready, progress=notify)
    require(set(checkpoints) == {"fold" + str(i) for i in range(5)})
    require(valid_sha(calibration_pin))
    for fold in range(5):
        read_checkpoint(p, pin, fold, checkpoints["fold" + str(fold)])
    read_calibration(pin, checkpoints, value["calibrated_completion"]["split_authentication"], calibration_pin)
    return value, checkpoints, calibration_pin


def authenticate_terminal(p, pin):
    require(valid_sha(pin) and sha(PROTOCOL) == pin and same(json.loads(PROTOCOL.read_text()), p))
    validate_protocol(p)
    require(OUT.is_dir() and not OUT.is_symlink() and not (OUT / "failure.json").exists())
    terminal_inventory(OUT, ("aggregate.json", "manifest.json", "baseline_replay.json"))
    for name in ("aggregate.json", "manifest.json", "baseline_replay.json"):
        regular(OUT / name)
    a = json.loads((OUT / "aggregate.json").read_text())
    contract.validate_result(a, p, pin)
    manifest_pin = sha(OUT / "manifest.json")
    m = json.loads((OUT / "manifest.json").read_text())
    contract.validate_manifest(m, pin, sha(OUT / "aggregate.json"), m["checkpoint_sha256"],
                               sha(OUT / "baseline_replay.json"), m["calibration_sha256"])
    require(same(json.loads((OUT / "baseline_replay.json").read_text()), experiment.BASELINE))
    require(PRIVATE.is_dir() and not PRIVATE.is_symlink() and PRIVATE.stat().st_mode & 0o777 == 0o700)
    require({path.name for path in PRIVATE.iterdir()} == {"fold" + str(fold) + ".pt" for fold in range(5)} | {"calibration.npz"})
    for fold in range(5):
        read_checkpoint(p, pin, fold, m["checkpoint_sha256"]["fold" + str(fold)])
    read_calibration(pin, m["checkpoint_sha256"], a["calibrated_completion"]["split_authentication"], m["calibration_sha256"])
    require(sha(OUT / "aggregate.json") == m["aggregate_sha256"] and sha(PROTOCOL) == pin
            and sha(OUT / "manifest.json") == manifest_pin
            and not (OUT / "failure.json").exists())
    return a, m


def audit(p, pin, state):
    a, m = authenticate_terminal(p, pin)
    mp = sha(OUT / "manifest.json")
    progress(AUDIT, state, "audit_replay")
    replay, cp, cal_pin = compute(p, pin, destination=AUDIT, state=state, refit=False,
                         expected_checkpoints=m["checkpoint_sha256"], expected_calibration=m["calibration_sha256"])
    require(same(a, replay) and cp == m["checkpoint_sha256"] and cal_pin == m["calibration_sha256"])
    later, later_m = authenticate_terminal(p, pin)
    require(same(a, later) and same(m, later_m) and sha(OUT / "manifest.json") == mp)
    value = {"schema": "bran-multisource-mask-audit-v1", "status": "authenticated",
        "protocol_sha256": pin, "aggregate_sha256": m["aggregate_sha256"], "manifest_sha256": mp,
        "checkpoint_sha256": cp, "fold_authentication": a["fold_authentication"],
        "calibration_sha256": cal_pin, "calibration_replayed": True,
        "all_aggregates_replayed": True, "checkpoint_predictions_replayed": True,
        "candidate_training_repeated": False, "raw_reference_heads_refit": True, "patient_level_output_emitted": False}
    contract.validate_audit(value, p, pin, m["aggregate_sha256"], mp, cp, cal_pin)
    return value


def authenticate_audit(p, pin, audit_pin):
    require(valid_sha(audit_pin) and AUDIT.is_dir() and not AUDIT.is_symlink()
            and not (AUDIT / "failure.json").exists())
    terminal_inventory(AUDIT, ("audit.json", "audit.manifest.json"))
    regular(AUDIT / "audit.json")
    regular(AUDIT / "audit.manifest.json")
    require(sha(AUDIT / "audit.json") == audit_pin)
    a, m = authenticate_terminal(p, pin)
    value = json.loads((AUDIT / "audit.json").read_text())
    contract.validate_audit(value, p, pin, m["aggregate_sha256"], sha(OUT / "manifest.json"),
                           m["checkpoint_sha256"], m["calibration_sha256"])
    require(json.loads((AUDIT / "audit.manifest.json").read_text()) ==
            {"protocol_sha256": pin, "artifact_sha256": audit_pin})
    require(sha(AUDIT / "audit.json") == audit_pin and not (AUDIT / "failure.json").exists())
    return a


def main(argv=None):
    parser = argparse.ArgumentParser()
    operations = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "audit"):
        operations.add_argument("--" + name, action="store_true")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--retinal-bridge-protocol-sha256")
    parser.add_argument("--retinal-bridge-audit-sha256")
    args = parser.parse_args(argv)
    destination = AUDIT if args.audit else OUT
    state, owned, ok = {"phase": "authentication"}, False, False
    start = time.monotonic()
    with _quiet():
        try:
            with open(LOCK, "a") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.prepare:
                    require(not any(path.exists() or path.is_symlink() for path in (PROTOCOL, OUT, AUDIT, PRIVATE)))
                    exclusive_json(PROTOCOL, prepare(args.retinal_bridge_protocol_sha256, args.retinal_bridge_audit_sha256))
                else:
                    regular(PROTOCOL)
                    require(valid_sha(args.protocol_sha256) and sha(PROTOCOL) == args.protocol_sha256)
                    p = json.loads(PROTOCOL.read_text())
                    validate_protocol(p)
                    if args.run:
                        require(not any(path.exists() or path.is_symlink() for path in (OUT, AUDIT, PRIVATE)))
                    destination.mkdir()
                    owned = True
                    if args.run:
                        PRIVATE.mkdir(mode=0o700)
                        a, cp, cal_pin = compute(p, args.protocol_sha256, destination=OUT, state=state, refit=True)
                        contract.validate_result(a, p, args.protocol_sha256)
                    else:
                        a = audit(p, args.protocol_sha256, state)
                    validate_protocol(p)
                    require(sha(PROTOCOL) == args.protocol_sha256 and same(json.loads(PROTOCOL.read_text()), p))
                    progress(destination, state, "publication")
                    require(not (destination / "failure.json").exists())
                    if args.run:
                        for fold in range(5):
                            read_checkpoint(p, args.protocol_sha256, fold, cp["fold" + str(fold)])
                        require(same(json.loads((OUT / "baseline_replay.json").read_text()), experiment.BASELINE))
                        read_calibration(args.protocol_sha256, cp, a["calibrated_completion"]["split_authentication"], cal_pin)
                        contract.validate_result(a, p, args.protocol_sha256)
                        exclusive_json(OUT / "aggregate.json", a)
                        m = {"protocol_sha256": args.protocol_sha256, "aggregate_sha256": sha(OUT / "aggregate.json"),
                            "checkpoint_sha256": cp, "baseline_replay_sha256": sha(OUT / "baseline_replay.json"),
                            "calibration_sha256": cal_pin,
                            "elapsed_seconds": round(time.monotonic() - start, 1), "patient_level_output_emitted": False}
                        contract.validate_manifest(m, args.protocol_sha256, m["aggregate_sha256"], cp, m["baseline_replay_sha256"], cal_pin)
                        exclusive_json(OUT / "manifest.json", m)
                    else:
                        terminal, manifest = authenticate_terminal(p, args.protocol_sha256)
                        contract.validate_audit(a, p, args.protocol_sha256, manifest["aggregate_sha256"],
                            sha(OUT / "manifest.json"), manifest["checkpoint_sha256"], manifest["calibration_sha256"])
                        exclusive_json(AUDIT / "audit.json", a)
                        exclusive_json(AUDIT / "audit.manifest.json", {"protocol_sha256": args.protocol_sha256,
                            "artifact_sha256": sha(AUDIT / "audit.json")})
            ok = True
        except Exception:
            if owned:
                try:
                    success = (destination / ("audit.manifest.json" if args.audit else "manifest.json")).exists()
                    if not success:
                        exclusive_json(destination / "failure.json", {"status": "execution_failed",
                            "phase": state["phase"] if state["phase"] in PHASES else "authentication",
                            "patient_level_output_emitted": False})
                except Exception:
                    pass
    print(json.dumps({"status": "completed" if ok else "execution_failed", "patient_level_output_emitted": False}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
