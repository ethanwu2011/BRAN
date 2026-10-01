"""Versioned technical recovery; no training, V1 overwrite or partial salvage."""
import argparse
import errno
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
import run_bran_retinal_decode_conformance_v2 as conformance
import bran_retinal_decode_v2 as decoder

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT/"BRAN_RETINAL_EXTRACTION_PROTOCOL_V2.json"
OUT = ROOT/"BRAN_RETINAL_EXTRACTION_V2"
AUDIT = ROOT/"BRAN_RETINAL_EXTRACTION_AUDIT_V2"
PRIVATE = ROOT/"private_artifacts/bran_retinal_extraction_v2"
PARAMETERS = dict(old.PARAMETERS)
TECHNICAL = {"decoder": "resource_lifecycle_v2_exact_pixel_sample_conformance",
             "python_gc_every_batches": 100, "failure_diagnostics": "closed_loader_stage_and_category",
             "partial_v1_reuse": False, "automatic_retry": False}
FILES = ("run_bran_retinal_extraction_v2.py", "test_run_bran_retinal_extraction_v2.py",
         "BRAN_RETINAL_EXTRACTION_EXECUTION_DESIGN_V2.md")
FLAGS = {**old.FLAGS, "original_failure_cause_established": False,
         "v1_partial_features_reused": False, "scientific_parameters_changed": False}
PHASES = ("protocol", "extraction", "audit", "publishing")
LOAD_STAGES = ("source_read", "source_path", "decoder")
IO_REASONS = ("not_applicable", "permission", "missing", "io", "descriptor_limit", "memory", "timeout", "other_io")


def require(ok):
    if not ok:
        raise ValueError("retinal_extraction_v2_contract_failed")


def authenticate_conformance(audit_pin):
    require(type(audit_pin) is str and len(audit_pin) == 64)
    cp = json.loads(conformance.PROTOCOL.read_text()); conformance.validate_protocol(cp)
    pin = sha(conformance.PROTOCOL)
    require(not (conformance.OUT/"failure.json").exists() and not (conformance.AUDIT/"failure.json").exists())
    result = json.loads((conformance.OUT/"aggregate.json").read_text()); conformance.validate_result(result, cp)
    require(result["pixel_equivalence_passed"] is True)
    require(sha(conformance.AUDIT/"audit.json") == audit_pin)
    ah = sha(conformance.OUT/"aggregate.json")
    require(json.loads((conformance.OUT/"aggregate.manifest.json").read_text()) ==
            {"protocol_sha256": pin, "artifact_sha256": ah})
    require(json.loads((conformance.AUDIT/"audit.manifest.json").read_text()) ==
            {"protocol_sha256": pin, "artifact_sha256": audit_pin})
    require(json.loads((conformance.AUDIT/"audit.json").read_text()) == {
        "schema": "bran-retinal-decode-conformance-audit-v2", "status": "authenticated",
        "protocol_sha256": pin, "aggregate_sha256": ah,
        "independent_pixel_comparison_replayed": True, "patient_level_output_emitted": False})
    return cp, pin, ah


def prepare(audit_pin):
    cp, pin, ah = authenticate_conformance(audit_pin)
    return {"schema": "bran-retinal-extraction-protocol-v2", "status": "frozen_before_execution",
        "parameters": PARAMETERS, "technical_parameters": TECHNICAL,
        "origin_protocol": cp["origin_protocol"], "selection": cp["origin_protocol"]["selection"],
        "preflight_protocol_sha256": pin, "preflight_aggregate_sha256": ah,
        "preflight_audit_sha256": audit_pin, "v1_private_sha256": cp["old_private_sha256"],
        "code_sha256": {**cp["code_sha256"], **{name: sha(ROOT/name) for name in FILES}}}


def validate_protocol(p, *, full=False):
    require(p == prepare(p["preflight_audit_sha256"]))
    if full:
        old.native.validate_protocol(p["origin_protocol"]["native_source"])
        require(old.selection()[3] == p["selection"])


def validate_loader_failure(value):
    if value is None:
        return
    require(type(value) is dict and set(value) == {"stage", "category", "decoder", "io_reason"}
            and value["stage"] in LOAD_STAGES and value["category"] in decoder.CATEGORIES
            and value["io_reason"] in IO_REASONS)
    if value["decoder"] is not None:
        d = value["decoder"]
        require(type(d) is dict and set(d) == {"stage", "category"}
                and d["stage"] in decoder.STAGES and d["category"] in decoder.CATEGORIES)


def make_loader(root, state):
    def load(record):
        stage = "source_path"
        try:
            path = root/record["relative_path"]
            require(path.resolve().is_relative_to(root.resolve()))
            stage = "source_read"; data = path.read_bytes()
            stage = "decoder"
            return decoder.decode_bytes(data, record["source_sha256"])
        except Exception as error:
            category = "memory" if isinstance(error, MemoryError) else "io" if isinstance(error, OSError) else "validation" if isinstance(error, ValueError) else "other"
            detail = error.safe_metadata() if isinstance(error, decoder.SafeDecodeFailure) else None
            if detail is not None:
                category = detail["category"]
            # Read ONLY errno/type from a bounded local exception chain, never
            # args, filenames, str(error), or DICOM metadata. No errno inferred.
            reason = "not_applicable"; internal = error
            for _ in range(4):
                if isinstance(internal, OSError):
                    reason = {errno.EACCES: "permission", errno.EPERM: "permission", errno.ENOENT: "missing",
                              errno.EIO: "io", errno.EMFILE: "descriptor_limit", errno.ENFILE: "descriptor_limit",
                              errno.ENOMEM: "memory", errno.ETIMEDOUT: "timeout"}.get(internal.errno, "other_io")
                    break
                internal = getattr(internal, "__context__", None)
                if internal is None:
                    break
            value = {"stage": stage, "category": category, "decoder": detail, "io_reason": reason}
            validate_loader_failure(value); state["loader_failure"] = value
            raise ValueError("retinal_v2_loader_failed") from None
    return load


def progress(phase, done=None, total=None):
    require(phase in ("source_hashing", "frozen_image_inference", "output_validation", "completed"))
    value = {"status": "completed" if phase == "completed" else "running", "phase": phase}
    if done is not None:
        require(type(done) is int and type(total) is int and 0 <= done <= total)
        value.update(completed_batches=done, total_batches=total)
    temp = OUT/"progress.tmp"; temp.write_text(json.dumps(value)+"\n"); os.replace(temp, OUT/"progress.json")


def validate_result(a, p, pin):
    require(type(a) is dict and set(a) == {"schema", "status", "protocol_sha256", "selection",
        "inventory_sha256", "inventory_file_sha256", "output_sha256", "dimension", "dtype",
        "historical_features_allclose", "preflight_audit_sha256"} | set(FLAGS))
    require(a["schema"] == "bran-retinal-extraction-aggregate-v2"
            and a["preflight_audit_sha256"] == p["preflight_audit_sha256"])
    require(all(a[key] is value for key, value in FLAGS.items()))
    original = {key: value for key, value in a.items() if key not in (set(FLAGS)-set(old.FLAGS)) | {"preflight_audit_sha256"}}
    original["schema"] = "bran-retinal-extraction-aggregate-v1"
    old.validate_result(original, p, pin)


def run(p, pin, state):
    records, old_rows, _, identity = old.selection(); require(identity == p["selection"])
    prior_inventory = json.loads((old.PRIVATE/"inventory.json").read_text())
    old.kernel.validate_inventory(prior_inventory)
    require([{**row, "source_sha256": "0"*64} for row in prior_inventory] == records)
    progress("source_hashing")
    for record, original in zip(records, prior_inventory):
        path = old.DATASET/record["relative_path"]
        require(path.resolve().is_relative_to(old.DATASET.resolve()))
        record["source_sha256"] = sha(path)
        require(record["source_sha256"] == original["source_sha256"])
    old.private_json(PRIVATE/"inventory.json", records)
    artifact, tower = old.load_tower()
    progress("frozen_image_inference")
    def tick(done, total):
        if done % TECHNICAL["python_gc_every_batches"] == 0:
            gc.collect()
        if done % 100 == 0 or done == total:
            progress("frozen_image_inference", done, total)
    value = old.kernel.extract_batches(records, PRIVATE/"features.npy", make_loader(old.DATASET, state),
        old.encoder(tower), batch_size=PARAMETERS["batch_size"], width=384, progress=tick)
    progress("output_validation")
    require(value == old.kernel.validate_output(PRIVATE/"features.npy", len(records), width=384))
    historical = np.load(old.CLINICAL/"data/local_emb/aireadi_emb_ours.npy", mmap_mode="r", allow_pickle=False)
    current = np.load(PRIVATE/"features.npy", mmap_mode="r", allow_pickle=False)
    try:
        compatible = all(np.allclose(current[i:i+256], historical[old_rows[i:i+256]],
                                    **PARAMETERS["historical_compatibility"]) for i in range(0, len(records), 256))
    finally:
        for array in (historical, current):
            if isinstance(array, np.memmap): array._mmap.close()
    a = {"schema": "bran-retinal-extraction-aggregate-v2", "status": "completed", "protocol_sha256": pin,
        "selection": identity, "inventory_sha256": old.kernel.inventory_sha256(records),
        "inventory_file_sha256": sha(PRIVATE/"inventory.json"), "output_sha256": value["output_sha256"],
        "dimension": 384, "dtype": "float32", "historical_features_allclose": bool(compatible),
        "preflight_audit_sha256": p["preflight_audit_sha256"], **FLAGS}
    validate_result(a, p, pin); validate_protocol(p)
    return a


def audit(p, pin, state):
    require(not (OUT/"failure.json").exists())
    a = json.loads((OUT/"aggregate.json").read_text()); validate_result(a, p, pin)
    require(json.loads((OUT/"aggregate.manifest.json").read_text()) ==
            {"protocol_sha256": pin, "artifact_sha256": sha(OUT/"aggregate.json")})
    records = json.loads((PRIVATE/"inventory.json").read_text()); old.kernel.validate_inventory(records)
    require((PRIVATE/"inventory.json").stat().st_mode & 0o777 == 0o600)
    require(sha(PRIVATE/"inventory.json") == a["inventory_file_sha256"]
            and old.kernel.inventory_sha256(records) == a["inventory_sha256"])
    expected, old_rows, _, identity = old.selection()
    require(identity == p["selection"] and [{**row, "source_sha256": "0"*64} for row in records] == expected)
    original = json.loads((old.PRIVATE/"inventory.json").read_text()); require(records == original)
    for record in records:
        path = old.DATASET/record["relative_path"]
        require(path.resolve().is_relative_to(old.DATASET.resolve()) and sha(path) == record["source_sha256"])
    value = old.kernel.validate_output(PRIVATE/"features.npy", len(records), width=384)
    require(value["output_sha256"] == a["output_sha256"])
    current = np.load(PRIVATE/"features.npy", mmap_mode="r", allow_pickle=False)
    historical = np.load(old.CLINICAL/"data/local_emb/aireadi_emb_ours.npy", mmap_mode="r", allow_pickle=False)
    try:
        _, tower = old.load_tower(); load = make_loader(old.DATASET, state)
        indices = np.linspace(0, len(records)-1, PARAMETERS["audit_replay_images"], dtype=int)
        replay = old.encoder(tower)(np.stack([load(records[i]) for i in indices]))
        require(np.allclose(replay, current[indices], **PARAMETERS["historical_compatibility"]))
        compatible = all(np.allclose(current[i:i+256], historical[old_rows[i:i+256]],
            **PARAMETERS["historical_compatibility"]) for i in range(0, len(records), 256))
        require(bool(compatible) == a["historical_features_allclose"])
    finally:
        for array in (current, historical):
            if isinstance(array, np.memmap): array._mmap.close()
    validate_protocol(p, full=True)
    return {"schema": "bran-retinal-extraction-audit-v2", "status": "authenticated", "protocol_sha256": pin,
        "aggregate_sha256": sha(OUT/"aggregate.json"), "inventory_file_sha256": a["inventory_file_sha256"],
        "output_sha256": value["output_sha256"], "selection": identity,
        "preflight_audit_sha256": p["preflight_audit_sha256"],
        "independent_generator_replay_passed": True, "patient_level_output_emitted": False}


def main():
    parser = argparse.ArgumentParser(); op = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "audit"):
        op.add_argument("--"+name, action="store_true")
    parser.add_argument("--protocol-sha256"); parser.add_argument("--conformance-audit-sha256"); args = parser.parse_args()
    dest = AUDIT if args.audit else OUT; owned = False; ok = False; phase = "protocol"; state = {"loader_failure": None}
    with _quiet():
        try:
            if args.prepare:
                require(not OUT.exists() and not AUDIT.exists() and not PRIVATE.exists())
                exclusive_json(PROTOCOL, prepare(args.conformance_audit_sha256))
            else:
                require(args.protocol_sha256 and sha(PROTOCOL) == args.protocol_sha256)
                p = json.loads(PROTOCOL.read_text()); validate_protocol(p, full=True)
                dest.mkdir(); owned = True
                with open("/private/tmp/bran_retinal_extraction_v1.lock", "a") as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if args.run:
                        PRIVATE.mkdir(mode=0o700); phase = "extraction"
                        a = run(p, args.protocol_sha256, state); name = "aggregate.json"
                    else:
                        phase = "audit"; a = audit(p, args.protocol_sha256, state); name = "audit.json"
                    phase = "publishing"
                    if args.run: progress("completed")
                    # The exclusive terminal artifact, not progress, commits last.
                    publish(dest, name, a, args.protocol_sha256)
            ok = True
        except Exception as error:
            if owned and not any((dest/name).exists() for name in ("aggregate.json", "audit.json")):
                validate_loader_failure(state["loader_failure"])
                category = "memory" if isinstance(error, MemoryError) else "io" if isinstance(error, OSError) else "validation" if isinstance(error, ValueError) else "other"
                exclusive_json(dest/"failure.json", {"status": "execution_failed", "phase": phase,
                    "category": category, "loader_failure": state["loader_failure"], "patient_level_output_emitted": False})
    print(json.dumps({"status": "completed" if ok else "execution_failed", "patient_level_output_emitted": False}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
