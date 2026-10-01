"""Bounded local V1/V2 pixel-conformance gate; no encoder inference or restart."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import exclusive_json, sha
from run_bran_cbc_reference_preflight_v1 import publish
import run_bran_retinal_extraction_v1 as old
import bran_retinal_decode_v2 as new

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT/"BRAN_RETINAL_DECODE_CONFORMANCE_PROTOCOL_V2.json"
OUT = ROOT/"BRAN_RETINAL_DECODE_CONFORMANCE_V2"
AUDIT = ROOT/"BRAN_RETINAL_DECODE_CONFORMANCE_AUDIT_V2"
OLD_PIN = "e82f307d81f8f8ff3cce84ec130d1eab0512d0dc4cc9a217d5565db77fb8f3e1"
FAILURE_PIN = "0c74178158f66bf2b40063f4ec6460b4e975125dd055c0395f68b4bacaafeed1"
CODE = ("run_bran_retinal_decode_conformance_v2.py", "test_run_bran_retinal_decode_conformance_v2.py",
        "bran_retinal_decode_v2.py", "test_bran_retinal_decode_v2.py",
        "run_bran_cbc_reference_preflight_v1.py", "BRAN_RETINAL_TECHNICAL_RECOVERY_DESIGN_V2.md")
PARAMETERS = {"evenly_spaced_images": 48, "next_candidate_batch_images": 24,
              "comparison": "exact_array_equality", "image_size": 224,
              "selection": "union_of_fixed_evenly_spaced_indices_and_next_candidate_batch",
              "source_bytes": "same_hash_authenticated_bytes_for_both_decoders",
              "model_inference_steps": 0, "training_steps": 0}
FLAGS = {"patient_level_output_emitted": False, "pixel_or_metadata_values_emitted": False,
         "private_inputs_unchanged": True, "source_bytes_reauthenticated": True,
         "encoder_inference_performed": False, "extraction_restarted": False,
         "original_failure_cause_established": False, "clinical_benefit_claim": False,
         "incomplete_features_used_as_scientific_output": False}
PHASES = ("protocol", "selection", "prefix", "load_bytes", "old_decode", "new_decode", "compare", "publishing")


def require(ok):
    if not ok:
        raise ValueError("retinal_decode_conformance_contract_failed")


def prepare():
    require(sha(old.PROTOCOL) == OLD_PIN and sha(old.OUT/"failure.json") == FAILURE_PIN)
    require(not (old.OUT/"aggregate.json").exists() and not (old.AUDIT/"audit.json").exists())
    origin = json.loads(old.PROTOCOL.read_text()); old.validate_protocol(origin)
    paths = {"inventory": old.PRIVATE/"inventory.json", "partial_features": old.PRIVATE/"features.npy"}
    require(all(path.stat().st_mode & 0o777 == 0o600 for path in paths.values()))
    return {"schema": "bran-retinal-decode-conformance-protocol-v2", "status": "frozen_before_execution",
        "old_protocol_sha256": OLD_PIN, "old_failure_sha256": FAILURE_PIN, "origin_protocol": origin,
        "old_private_sha256": {key: sha(path) for key, path in paths.items()}, "parameters": PARAMETERS,
        "code_sha256": {**origin["code_sha256"], **{name: sha(ROOT/name) for name in CODE}}}


def validate_protocol(p):
    require(p == prepare())


def candidate_indices(features, rows):
    require(isinstance(features, np.ndarray) and features.shape == (rows, 384)
            and features.dtype == np.float32 and np.isfinite(features).all())
    written = np.abs(features).sum(1, dtype=np.float64) > 0
    missing = np.flatnonzero(~written)
    require(len(missing) > 0)
    start = int(missing[0])
    require(start % 24 == 0 and written[:start].all() and not written[start:].any())
    return np.unique(np.concatenate((np.linspace(0, rows-1, min(rows, 48), dtype=int),
                                    np.arange(start, min(start+24, rows))))).tolist()


def compare_records(records, indices, root, old_decode, new_decode, state):
    """No record/index enters the returned boolean or a generated message."""
    equal = True
    for index in indices:
        record = records[index]; path = root/record["relative_path"]
        state["phase"] = "load_bytes"
        require(path.resolve().is_relative_to(root.resolve()))
        data = path.read_bytes(); require(hashlib.sha256(data).hexdigest() == record["source_sha256"])
        state["phase"] = "old_decode"; before = old_decode(data, record["source_sha256"])
        state["phase"] = "new_decode"; after = new_decode(data, record["source_sha256"])
        state["phase"] = "compare"
        require(before.shape == after.shape == (3, 224, 224) and before.dtype == after.dtype == np.float32
                and np.isfinite(before).all() and np.isfinite(after).all())
        equal &= bool(np.array_equal(before, after))
        require(sha(path) == record["source_sha256"])
        del data, before, after
    return bool(equal)


def compute(p, state):
    state["phase"] = "selection"
    old.native.validate_protocol(p["origin_protocol"]["native_source"])
    expected, _, _, selection = old.selection()
    require(selection == p["origin_protocol"]["selection"])
    records = json.loads((old.PRIVATE/"inventory.json").read_text()); old.kernel.validate_inventory(records)
    require([{**row, "source_sha256": "0"*64} for row in records] == expected)
    state["phase"] = "prefix"
    features = np.load(old.PRIVATE/"features.npy", mmap_mode="r", allow_pickle=False)
    try:
        indices = candidate_indices(features, len(records))
    finally:
        if isinstance(features, np.memmap):
            features._mmap.close()
    equal = compare_records(records, indices, old.DATASET, old.decode_bytes, new.decode_bytes, state)
    validate_protocol(p)
    result = {"schema": "bran-retinal-decode-conformance-aggregate-v2", "status": "completed",
              "pixel_equivalence_passed": equal, "sampling_rule": PARAMETERS["selection"],
              "origin_selection": selection, "old_private_sha256": p["old_private_sha256"], **FLAGS}
    validate_result(result, p)
    return result


def validate_result(a, p):
    require(type(a) is dict and set(a) == {"schema", "status", "pixel_equivalence_passed", "sampling_rule",
            "origin_selection", "old_private_sha256"} | set(FLAGS))
    require(a["schema"] == "bran-retinal-decode-conformance-aggregate-v2" and a["status"] == "completed"
            and type(a["pixel_equivalence_passed"]) is bool and a["sampling_rule"] == PARAMETERS["selection"])
    require(a["origin_selection"] == p["origin_protocol"]["selection"]
            and a["old_private_sha256"] == p["old_private_sha256"])
    require(all(a[key] is value for key, value in FLAGS.items()))


def audit(p, pin, state):
    require(not (OUT/"failure.json").exists())
    raw = (OUT/"aggregate.json").read_bytes(); a = json.loads(raw); validate_result(a, p)
    require(json.loads((OUT/"aggregate.manifest.json").read_text()) ==
            {"protocol_sha256": pin, "artifact_sha256": hashlib.sha256(raw).hexdigest()})
    require(compute(p, state) == a and (OUT/"aggregate.json").read_bytes() == raw)
    validate_protocol(p)
    return {"schema": "bran-retinal-decode-conformance-audit-v2", "status": "authenticated",
            "protocol_sha256": pin, "aggregate_sha256": hashlib.sha256(raw).hexdigest(),
            "independent_pixel_comparison_replayed": True, "patient_level_output_emitted": False}


def safe_failure(error, phase):
    require(phase in PHASES)
    category = "memory" if isinstance(error, MemoryError) else "io" if isinstance(error, OSError) else "validation" if isinstance(error, ValueError) else "other"
    decoder = None
    if isinstance(error, new.SafeDecodeFailure):
        decoder = error.safe_metadata()
        require(type(decoder) is dict and set(decoder) == {"stage", "category"}
                and decoder["stage"] in new.STAGES and decoder["category"] in new.CATEGORIES)
    return {"status": "execution_failed", "phase": phase, "category": category,
            "decoder": decoder, "patient_level_output_emitted": False}


def main():
    parser = argparse.ArgumentParser(); op = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "audit"):
        op.add_argument("--"+name, action="store_true")
    parser.add_argument("--protocol-sha256"); args = parser.parse_args()
    dest = AUDIT if args.audit else OUT; owned = False; ok = False; state = {"phase": "protocol"}
    with _quiet():
        try:
            if args.prepare:
                require(not OUT.exists() and not AUDIT.exists()); exclusive_json(PROTOCOL, prepare())
            else:
                require(args.protocol_sha256 and sha(PROTOCOL) == args.protocol_sha256)
                p = json.loads(PROTOCOL.read_text()); validate_protocol(p); dest.mkdir(); owned = True
                if args.audit:
                    result = audit(p, args.protocol_sha256, state); name = "audit.json"
                else:
                    result = compute(p, state); name = "aggregate.json"
                state["phase"] = "publishing"; publish(dest, name, result, args.protocol_sha256)
            ok = True
        except Exception as error:
            if owned and not any((dest/name).exists() for name in ("aggregate.json", "audit.json")):
                exclusive_json(dest/"failure.json", safe_failure(error, state["phase"]))
    print(json.dumps({"status": "completed" if ok else "execution_failed", "patient_level_output_emitted": False}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
