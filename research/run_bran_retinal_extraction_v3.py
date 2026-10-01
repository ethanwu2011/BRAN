"""Fresh retinal extraction with early local-GPU startup; no scientific changes.

Every real operation is FD-quiet and holds the shared heavy-job lock. Failed V1
and V2 artifacts are authenticated, never overwritten or used as feature inputs.
"""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path

import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import exclusive_json, sha
from run_bran_cbc_reference_preflight_v1 import publish
import run_bran_retinal_extraction_v1 as old
import run_bran_retinal_extraction_v2 as previous

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "BRAN_RETINAL_EXTRACTION_PROTOCOL_V3.json"
OUT = ROOT / "BRAN_RETINAL_EXTRACTION_V3"
AUDIT = ROOT / "BRAN_RETINAL_EXTRACTION_AUDIT_V3"
PRIVATE = ROOT / "private_artifacts/bran_retinal_extraction_v3"
LOCK = Path("/private/tmp/bran_retinal_extraction_v1.lock")
PREVIOUS_PIN = "4fe9af017eb9b6f9ad372e59aeca9c8f2f20f45655ff32cfc697efd401272173"
FAILURE_PIN = "ed268c95c2464d23d11c84f33e2dec67deda8dfce32bfcbf749d22d2b89503f2"
PREVIOUS_INVENTORY_PIN = "120336c23828a42d557a460f0bf2ba61e5587544d0d847417184e6df91833983"
PARAMETERS = dict(old.PARAMETERS)
TECHNICAL = {**previous.TECHNICAL, "partial_v2_reuse": False,
    "gpu_startup_before_source_scan": True, "cpu_fallback": False,
    "failure_diagnostics": "closed_loader_stage_category_and_gpu_startup_stage",
    "startup_probe": "two zero RGB tensors; original frozen encoder; no training",
    "execution_context": "local MPS available; explicit GPU-enabled launch required",
    "all_operations_use_shared_heavy_lock": True}
FILES = ("run_bran_retinal_extraction_v3.py", "test_run_bran_retinal_extraction_v3.py",
         "BRAN_RETINAL_EXTRACTION_EXECUTION_DESIGN_V3.md")
FLAGS = {**previous.FLAGS, "gpu_preflight_passed": True}
STARTUP_STAGES = ("not_started", "gpu_availability", "model_startup", "synthetic_forward", "complete")
PHASES = ("protocol", "gpu_preflight", "source_authentication", "extraction", "audit", "publishing")


def require(ok):
    if not ok:
        raise ValueError("retinal_extraction_v3_contract_failed")


def require_protocol_pin(pin):
    require(type(pin) is str and len(pin) == 64 and sha(PROTOCOL) == pin)


def authenticate_previous():
    require(sha(previous.PROTOCOL) == PREVIOUS_PIN)
    p = json.loads(previous.PROTOCOL.read_text())
    previous.validate_protocol(p)
    require(sha(previous.OUT / "failure.json") == FAILURE_PIN)
    require(json.loads((previous.OUT / "failure.json").read_text()) == {
        "status": "execution_failed", "phase": "extraction", "category": "validation",
        "loader_failure": None, "patient_level_output_emitted": False})
    require(not previous.AUDIT.exists())
    require(not any((previous.OUT / name).exists() for name in
                    ("aggregate.json", "aggregate.manifest.json", "audit.json", "audit.manifest.json")))
    require(not (previous.PRIVATE / "features.npy").exists())
    require((previous.PRIVATE / "inventory.json").stat().st_mode & 0o777 == 0o600)
    require(sha(previous.PRIVATE / "inventory.json") == PREVIOUS_INVENTORY_PIN)
    return p


def prepare(audit_pin):
    prior = authenticate_previous()
    require(audit_pin == prior["preflight_audit_sha256"])
    # authenticate_previous reproduces V2 preparation, including the actual
    # conformance audit and original source/checkpoint/code/private hashes.
    return {"schema": "bran-retinal-extraction-protocol-v3", "status": "frozen_before_execution",
        "parameters": PARAMETERS, "technical_parameters": TECHNICAL,
        "origin_protocol": prior["origin_protocol"], "selection": prior["selection"],
        "preflight_protocol_sha256": prior["preflight_protocol_sha256"],
        "preflight_aggregate_sha256": prior["preflight_aggregate_sha256"],
        "preflight_audit_sha256": audit_pin, "v1_private_sha256": prior["v1_private_sha256"],
        "previous_protocol_sha256": PREVIOUS_PIN, "previous_failure_sha256": FAILURE_PIN,
        "previous_inventory_sha256": PREVIOUS_INVENTORY_PIN,
        "code_sha256": {**prior["code_sha256"], **{name: sha(ROOT / name) for name in FILES}}}


def validate_protocol(p, *, full=False):
    require(p == prepare(p["preflight_audit_sha256"]))
    if full:
        old.native.validate_protocol(p["origin_protocol"]["native_source"])
        require(old.selection()[3] == p["selection"])


def gpu_preflight(state):
    """Check GPU before source hashing; run only a synthetic startup probe.

    The caller keeps this same eval-mode tower for extraction/replay. Nothing is
    trained and nothing silently falls back to CPU. The model load retains V1's
    original checkpoint/inference contract checks.
    """
    state["startup_stage"] = "gpu_availability"
    import torch
    require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0")
    require(torch.backends.mps.is_built() and torch.backends.mps.is_available())
    state["startup_stage"] = "model_startup"
    _, tower = old.load_tower()
    state["startup_stage"] = "synthetic_forward"
    value = old.encoder(tower)(np.zeros((2, 3, 224, 224), np.float32))
    require(isinstance(value, np.ndarray) and value.shape == (2, 384)
            and value.dtype == np.float32 and np.isfinite(value).all()
            and np.all(np.linalg.norm(value, axis=1) > 0))
    torch.mps.synchronize()
    state["startup_stage"] = "complete"
    return tower


def progress(phase, done=None, total=None):
    require(phase in ("source_hashing", "frozen_image_inference", "output_validation", "completed"))
    value = {"status": "completed" if phase == "completed" else "running", "phase": phase}
    if done is not None:
        require(type(done) is int and type(total) is int and 0 <= done <= total)
        value.update(completed_batches=done, total_batches=total)
    temp = OUT / "progress.tmp"
    temp.write_text(json.dumps(value) + "\n")
    os.replace(temp, OUT / "progress.json")


def validate_result(a, p, pin):
    require(type(a) is dict and a.get("schema") == "bran-retinal-extraction-aggregate-v3"
            and a.get("gpu_preflight_passed") is True)
    prior = {key: value for key, value in a.items() if key != "gpu_preflight_passed"}
    prior["schema"] = "bran-retinal-extraction-aggregate-v2"
    previous.validate_result(prior, p, pin)


def historical_compatibility(records, old_rows):
    """Compare all fresh features; never reuse failed-run feature arrays."""
    current = historical = None
    try:
        current = np.load(PRIVATE / "features.npy", mmap_mode="r", allow_pickle=False)
        historical = np.load(old.CLINICAL / "data/local_emb/aireadi_emb_ours.npy",
                             mmap_mode="r", allow_pickle=False)
        return bool(all(np.allclose(current[i:i+256], historical[old_rows[i:i+256]],
                        **PARAMETERS["historical_compatibility"]) for i in range(0, len(records), 256)))
    finally:
        for array in (current, historical):
            if isinstance(array, np.memmap):
                array._mmap.close()


def run(p, pin, state, tower):
    require(state["startup_stage"] == "complete")
    records, old_rows, _, identity = old.selection()
    require(identity == p["selection"])
    original = json.loads((old.PRIVATE / "inventory.json").read_text())
    old.kernel.validate_inventory(original)
    require([{**row, "source_sha256": "0"*64} for row in original] == records)
    progress("source_hashing")
    for record, prior in zip(records, original):
        path = old.DATASET / record["relative_path"]
        require(path.resolve().is_relative_to(old.DATASET.resolve()))
        record["source_sha256"] = sha(path)
        require(record["source_sha256"] == prior["source_sha256"])
    old.private_json(PRIVATE / "inventory.json", records)
    progress("frozen_image_inference")

    def tick(done, total):
        if done % TECHNICAL["python_gc_every_batches"] == 0:
            gc.collect()
        if done % 100 == 0 or done == total:
            progress("frozen_image_inference", done, total)

    value = old.kernel.extract_batches(records, PRIVATE / "features.npy",
        previous.make_loader(old.DATASET, state), old.encoder(tower),
        batch_size=PARAMETERS["batch_size"], width=384, progress=tick)
    progress("output_validation")
    require(value == old.kernel.validate_output(PRIVATE / "features.npy", len(records), width=384))
    a = {"schema": "bran-retinal-extraction-aggregate-v3", "status": "completed",
        "protocol_sha256": pin, "selection": identity,
        "inventory_sha256": old.kernel.inventory_sha256(records),
        "inventory_file_sha256": sha(PRIVATE / "inventory.json"),
        "output_sha256": value["output_sha256"], "dimension": 384, "dtype": "float32",
        "historical_features_allclose": historical_compatibility(records, old_rows),
        "preflight_audit_sha256": p["preflight_audit_sha256"], **FLAGS}
    validate_result(a, p, pin)
    validate_protocol(p)
    require_protocol_pin(pin)
    return a


def authenticate_result(p, pin):
    require(not (OUT / "failure.json").exists())
    require(not any((OUT / name).exists() for name in ("audit.json", "audit.manifest.json")))
    a = json.loads((OUT / "aggregate.json").read_text())
    validate_result(a, p, pin)
    require(json.loads((OUT / "aggregate.manifest.json").read_text()) == {
        "protocol_sha256": pin, "artifact_sha256": sha(OUT / "aggregate.json")})
    return a


def audit(p, pin, state, tower):
    require(state["startup_stage"] == "complete")
    require(not (AUDIT / "failure.json").exists())
    a = authenticate_result(p, pin)
    records = json.loads((PRIVATE / "inventory.json").read_text())
    old.kernel.validate_inventory(records)
    require(PRIVATE.stat().st_mode & 0o777 == 0o700)
    require((PRIVATE / "inventory.json").stat().st_mode & 0o777 == 0o600)
    require(sha(PRIVATE / "inventory.json") == a["inventory_file_sha256"]
            and old.kernel.inventory_sha256(records) == a["inventory_sha256"])
    expected, old_rows, _, identity = old.selection()
    require(identity == p["selection"] and [{**row, "source_sha256": "0"*64} for row in records] == expected)
    require(records == json.loads((old.PRIVATE / "inventory.json").read_text()))
    for record in records:
        path = old.DATASET / record["relative_path"]
        require(path.resolve().is_relative_to(old.DATASET.resolve()) and sha(path) == record["source_sha256"])
    value = old.kernel.validate_output(PRIVATE / "features.npy", len(records), width=384)
    require(value["output_sha256"] == a["output_sha256"])
    current = np.load(PRIVATE / "features.npy", mmap_mode="r", allow_pickle=False)
    try:
        indices = np.linspace(0, len(records)-1, PARAMETERS["audit_replay_images"], dtype=int)
        load = previous.make_loader(old.DATASET, state)
        replay = old.encoder(tower)(np.stack([load(records[i]) for i in indices]))
        require(np.allclose(replay, current[indices], **PARAMETERS["historical_compatibility"]))
    finally:
        if isinstance(current, np.memmap):
            current._mmap.close()
    require(historical_compatibility(records, old_rows) == a["historical_features_allclose"])
    validate_protocol(p, full=True)
    require_protocol_pin(pin)
    require(sha(PRIVATE / "inventory.json") == a["inventory_file_sha256"])
    require(old.kernel.validate_output(PRIVATE / "features.npy", len(records), width=384) == value)
    require(authenticate_result(p, pin) == a)
    return {"schema": "bran-retinal-extraction-audit-v3", "status": "authenticated",
        "protocol_sha256": pin, "aggregate_sha256": sha(OUT / "aggregate.json"),
        "inventory_file_sha256": a["inventory_file_sha256"], "output_sha256": value["output_sha256"],
        "selection": identity, "preflight_audit_sha256": p["preflight_audit_sha256"],
        "gpu_preflight_passed": True, "independent_generator_replay_passed": True,
        "patient_level_output_emitted": False}


def main(argv=None):
    parser = argparse.ArgumentParser()
    op = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "audit"):
        op.add_argument("--" + name, action="store_true")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--conformance-audit-sha256")
    args = parser.parse_args(argv)
    dest = AUDIT if args.audit else OUT
    owned, ok, phase = False, False, "protocol"
    state = {"loader_failure": None, "startup_stage": "not_started"}
    with _quiet():
        try:
            with open(LOCK, "a") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.prepare:
                    require(not any(x.exists() for x in (PROTOCOL, OUT, AUDIT, PRIVATE)))
                else:
                    require_protocol_pin(args.protocol_sha256)
                    p = json.loads(PROTOCOL.read_text())
                    if args.run:
                        require(not any(x.exists() for x in (OUT, AUDIT, PRIVATE)))
                    dest.mkdir()
                    owned = True
                # Crucially before full metadata authentication, which itself
                # may hash source files, and before selection or image scanning.
                phase = "gpu_preflight"
                tower = gpu_preflight(state)
                phase = "source_authentication"
                if args.prepare:
                    p = prepare(args.conformance_audit_sha256)
                    exclusive_json(PROTOCOL, p)
                else:
                    validate_protocol(p, full=True)
                    if args.run:
                        PRIVATE.mkdir(mode=0o700)
                        phase = "extraction"
                        a = run(p, args.protocol_sha256, state, tower)
                        name = "aggregate.json"
                    else:
                        phase = "audit"
                        a = audit(p, args.protocol_sha256, state, tower)
                        name = "audit.json"
                    phase = "publishing"
                    require_protocol_pin(args.protocol_sha256)
                    require(json.loads(PROTOCOL.read_text()) == p)
                    require(not (dest / "failure.json").exists())
                    if args.run:
                        progress("completed")
                    publish(dest, name, a, args.protocol_sha256)
                del tower
            ok = True
        except Exception as error:
            try:
                if owned and not any((dest / name).exists() for name in ("aggregate.json", "audit.json")):
                    previous.validate_loader_failure(state["loader_failure"])
                    require(state["startup_stage"] in STARTUP_STAGES and phase in PHASES)
                    category = ("memory" if isinstance(error, MemoryError) else "io" if isinstance(error, OSError)
                                else "validation" if isinstance(error, ValueError) else "other")
                    exclusive_json(dest / "failure.json", {"status": "execution_failed", "phase": phase,
                        "category": category, "loader_failure": state["loader_failure"],
                        "startup_stage": state["startup_stage"], "patient_level_output_emitted": False})
            except Exception:
                pass  # Even failure-writer errors must never escape FD redaction.
    public = {"status": "completed" if ok else "execution_failed", "patient_level_output_emitted": False}
    if not ok:
        # Preparation or lock failures may not own a destination. Preserve
        # their closed stage in stdout too, without creating misleading receipts.
        public["failure_phase"] = phase if type(phase) is str and phase in PHASES else "protocol"
        stage = state.get("startup_stage")
        public["startup_stage"] = stage if type(stage) is str and stage in STARTUP_STAGES else "not_started"
    print(json.dumps(public))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
