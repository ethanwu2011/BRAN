"""Prospective retinal production receipt; every patient operation stays FD-quiet."""
import argparse
import fcntl
import glob
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import traceback

import numpy as np
from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import exclusive_json, sha
import bran_retinal_extraction_kernel_v1 as kernel
from bran_retinal_decode_v1 import decode_bytes
import run_bran_native_screening_v1 as native
from patient_atlas_eye_contracts import load_external_eye_tower

ROOT = Path(__file__).resolve().parent
CLINICAL = Path("/Users/ethanwu/clinical-world-model")
DATASET = Path("/Volumes/Extreme/AIREADI_raw/aireadi-container/849e8f67-8355-4ded-a48e-019320371848/dataset")
PROTOCOL = ROOT / "BRAN_RETINAL_EXTRACTION_PROTOCOL_V1.json"
OUT = ROOT / "BRAN_RETINAL_EXTRACTION_V1"
AUDIT = ROOT / "BRAN_RETINAL_EXTRACTION_AUDIT_V1"
PRIVATE = ROOT / "private_artifacts/bran_retinal_extraction_v1"
PARAMETERS = {"image_size": 224, "adaptation_image_size": 128, "batch_size": 24,
    "device": "mps", "dtype": "float32", "torch_threads": 2, "seed": 94601,
    "pooling": "mean normalized final-layer patch tokens after five prefix tokens",
    "normalization_mean": [.485, .456, .406], "normalization_std": [.229, .224, .225],
    "decode": "authenticated in-memory DICOM; single embedded frame; PIL RGB, no double YBR conversion; reduced JPEG/JPEG2000 decode",
    "resize": "PIL bicubic square224; no crop or augmentation",
    "selection": "all CFP images of exact1928 train/val participants in original sorted physical order",
    "historical_compatibility": {"rtol": .0001, "atol": .00001},
    "audit_replay_images": 24,
    "per_patient_pooling": "arithmetic mean over observed images; separate downstream operation",
    "training_steps": 0, "clinical_benefit_claim": False}
FILES = ("run_bran_retinal_extraction_v1.py", "test_run_bran_retinal_extraction_v1.py",
    "bran_retinal_extraction_kernel_v1.py", "test_bran_retinal_extraction_kernel_v1.py",
    "bran_retinal_decode_v1.py", "test_bran_retinal_decode_v1.py", "patient_atlas_eye_contracts.py",
    "patient_atlas_contracts.py", "BRAN_RETINAL_EXTRACTION_DESIGN_V1.md")
EXTERNAL = {"checkpoint": CLINICAL / "data/dinov3_fundus.pt",
    "cache_metadata": CLINICAL / "data/img_cache_128_meta.npy",
    "cache_builder": CLINICAL / "prep_image_cache.py", "adaptation": CLINICAL / "adapt_dinov3.py",
    "eye_contract": ROOT / "EXTERNAL_EYE_TOWER_CONTRACT.json",
    "source_policy": ROOT / "PATIENT_ATLAS_SOURCE_POLICY.json",
    "retinal_manifest": DATASET / "retinal_photography/manifest.tsv"}
FLAGS = {"patient_level_output_emitted": False, "official_test_images_encoded": False,
    "historical_production_proven": False, "clinical_benefit_claim": False, "model_promoted": False,
    "all_rows_authenticated_before_decode": True, "all_rows_written_once": True}


def require(ok):
    if not ok:
        raise ValueError("retinal_extraction_contract_failed")


def digest_json(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def private_json(path, x):
    data = json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())


def physical_paths(dataset):
    # Match the existing authenticated loader, including linked directories.
    return [Path(x) for x in sorted(glob.glob(str(dataset / "retinal_photography/cfp/**/*.dcm"), recursive=True))]


def selection():
    import pandas as pd
    from patient_atlas_real_data import _normalized_retinal_path
    ctx, folds, *_ = native.source.io.load_context()
    ids = list(map(str, ctx["raw_cohort"].patient_ids))
    require(len(ids) == len(set(ids)) == 1928 and set(ctx["raw_cohort"].split_labels) <= {"train", "val"})
    require(list(map(str, ctx["feature_cohort"].patient_ids)) == ids)
    participants = set(ids)
    m = pd.read_csv(EXTERNAL["retinal_manifest"], sep="\t", dtype=str,
                    usecols=["person_id", "filepath", "laterality"])
    m["rel"] = m["filepath"].map(_normalized_retinal_path)
    m = m[m["rel"].str.contains(r"(?:^|/)cfp(?:/|$)", regex=True, na=False)]
    require(m["rel"].is_unique)
    physical = physical_paths(DATASET)
    rels = [p.relative_to(DATASET).as_posix() for p in physical]
    require(len(rels) == len(set(rels)) == 50315 and set(rels) == set(m["rel"]))
    by_path = m.set_index("rel", verify_integrity=True)
    records, old_rows = [], []
    for old_row, rel in enumerate(rels):
        row = by_path.loc[rel]
        if str(row["person_id"]) not in participants:
            continue
        parts = Path(rel).parts
        laterality = str(row["laterality"])
        records.append({"row": len(records), "relative_path": rel, "person_id": str(row["person_id"]),
            "laterality": laterality if laterality in ("L", "R") else "unknown",
            "device": parts[parts.index("cfp") + 1], "source_sha256": "0" * 64})
        old_rows.append(old_row)
    kernel.validate_inventory(records); require(len(records) >= 24)
    identity = {"patient_order_sha256": digest_json(ids), "fold_order_sha256": digest_json(np.asarray(folds).tolist()),
                "selection_sha256": kernel.inventory_sha256(records), "rows": len(records), "people": 1928}
    return records, old_rows, ids, identity


def load_tower():
    import torch
    torch.set_num_threads(2); torch.manual_seed(PARAMETERS["seed"])
    require(torch.backends.mps.is_available())
    artifact = load_external_eye_tower(EXTERNAL["checkpoint"], EXTERNAL["eye_contract"],
        cache_metadata_path=EXTERNAL["cache_metadata"], cache_builder_path=EXTERNAL["cache_builder"],
        adaptation_script_path=EXTERNAL["adaptation"])
    inf = artifact.contract["inference"]
    require(inf["resolution_pixels"] == 224 and inf["normalization_mean"] == PARAMETERS["normalization_mean"]
        and inf["normalization_std"] == PARAMETERS["normalization_std"]
        and inf["embedding_rule"] == "mean of normalized final-layer patch tokens after all prefix tokens")
    tower = artifact.tower.to("mps", dtype=torch.float32).eval()
    require(tower.num_prefix_tokens == 5 and tower.embed_dim == 384)
    return artifact, tower


def encoder(tower):
    import torch
    def encode(x):
        with torch.inference_mode():
            z = tower.forward_features(torch.from_numpy(x).to("mps"))
            require(z.ndim == 3 and z.shape[1:] == (201, 384))
            return z[:, 5:].mean(1).float().cpu().numpy()
    return encode


def prepare():
    parent = native.prepare()
    _, _, _, selected = selection()
    artifact, tower = load_tower()
    probe = np.zeros((2, 3, 224, 224), np.float32)
    z = encoder(tower)(probe)
    require(z.shape == (2, 384) and z.dtype == np.float32 and np.isfinite(z).all()
            and np.all(np.linalg.norm(z, axis=1) > 0))
    del artifact, tower
    return {"schema": "bran-retinal-extraction-protocol-v1", "status": "frozen_before_execution",
        "parameters": PARAMETERS, "selection": selected, "native_source": parent,
        "external_sha256": {name: sha(path) for name, path in EXTERNAL.items()},
        "code_sha256": {**parent["code_sha256"], **{name: sha(ROOT / name) for name in FILES}},
        "runtime": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ("torch", "timm", "numpy", "pandas", "pydicom", "Pillow")}}}


def validate_protocol(p, *, full=False):
    require(set(p) == {"schema", "status", "parameters", "selection", "native_source", "external_sha256", "code_sha256", "runtime"})
    require(p["schema"] == "bran-retinal-extraction-protocol-v1" and p["status"] == "frozen_before_execution"
        and p["parameters"] == PARAMETERS)
    require(p["external_sha256"] == {name: sha(path) for name, path in EXTERNAL.items()})
    require(set(p["code_sha256"]) == set(p["native_source"]["code_sha256"]) | set(FILES))
    require(all(sha(ROOT / name) == value for name, value in p["code_sha256"].items()))
    require(p["runtime"] == {"python": platform.python_version(), **{name: importlib.metadata.version(name)
        for name in ("torch", "timm", "numpy", "pandas", "pydicom", "Pillow")}})
    if full:
        native.validate_protocol(p["native_source"])
        require(selection()[3] == p["selection"])


def progress(phase, completed_batches=None, total_batches=None):
    value = {"status": "running", "phase": phase}
    if completed_batches is not None:
        value.update(completed_batches=completed_batches, total_batches=total_batches)
    temp = OUT / "progress.tmp"
    temp.write_text(json.dumps(value) + "\n"); os.replace(temp, OUT / "progress.json")


def validate_result(a, p, pin):
    require(set(a) == {"schema", "status", "protocol_sha256", "selection", "inventory_sha256",
        "inventory_file_sha256", "output_sha256", "dimension", "dtype", "historical_features_allclose"} | set(FLAGS))
    require(a["schema"] == "bran-retinal-extraction-aggregate-v1" and a["status"] == "completed"
        and a["protocol_sha256"] == pin and a["selection"] == p["selection"]
        and a["dimension"] == 384 and a["dtype"] == "float32" and type(a["historical_features_allclose"]) is bool)
    for key in ("inventory_sha256", "inventory_file_sha256", "output_sha256"):
        require(type(a[key]) is str and len(a[key]) == 64 and all(c in "0123456789abcdef" for c in a[key]))
    require(all(a[k] is v for k, v in FLAGS.items()))


def run(p, pin):
    records, old_rows, ids, identity = selection(); require(identity == p["selection"])
    progress("source_hashing")
    for record in records:
        path = DATASET / record["relative_path"]
        require(path.resolve().is_relative_to(DATASET.resolve()))
        record["source_sha256"] = sha(path)
    private_json(PRIVATE / "inventory.json", records)
    artifact, tower = load_tower()
    def loader(record):
        return decode_bytes((DATASET / record["relative_path"]).read_bytes(), record["source_sha256"])
    progress("frozen_image_inference")
    def tick(done, total):
        if done % 100 == 0 or done == total:
            progress("frozen_image_inference", done, total)
    result = kernel.extract_batches(records, PRIVATE / "features.npy", loader, encoder(tower),
                                    batch_size=24, width=384, progress=tick)
    progress("output_validation")
    require(result == kernel.validate_output(PRIVATE / "features.npy", len(records), width=384))
    old = np.load(CLINICAL / "data/local_emb/aireadi_emb_ours.npy", mmap_mode="r", allow_pickle=False)
    new = np.load(PRIVATE / "features.npy", mmap_mode="r", allow_pickle=False)
    compatible = True
    for start in range(0, len(records), 256):
        compatible &= bool(np.allclose(new[start:start+256], old[old_rows[start:start+256]],
                                      **PARAMETERS["historical_compatibility"]))
    a = {"schema": "bran-retinal-extraction-aggregate-v1", "status": "completed", "protocol_sha256": pin,
        "selection": identity, "inventory_sha256": kernel.inventory_sha256(records),
        "inventory_file_sha256": sha(PRIVATE / "inventory.json"), "output_sha256": result["output_sha256"],
        "dimension": 384, "dtype": "float32", "historical_features_allclose": compatible, **FLAGS}
    validate_result(a, p, pin); validate_protocol(p)
    return a


def audit(p, pin):
    require(not (OUT / "failure.json").exists())
    a = json.loads((OUT / "aggregate.json").read_text()); validate_result(a, p, pin)
    records = json.loads((PRIVATE / "inventory.json").read_text()); kernel.validate_inventory(records)
    require((PRIVATE / "inventory.json").stat().st_mode & 0o777 == 0o600)
    require(sha(PRIVATE / "inventory.json") == a["inventory_file_sha256"]
        and kernel.inventory_sha256(records) == a["inventory_sha256"])
    expected, old_rows, _, identity = selection(); require(identity == p["selection"])
    require([{**x, "source_sha256": "0"*64} for x in records] == expected)
    for record in records:
        require(sha(DATASET / record["relative_path"]) == record["source_sha256"])
    value = kernel.validate_output(PRIVATE / "features.npy", len(records), width=384)
    require(value["output_sha256"] == a["output_sha256"])
    new = np.load(PRIVATE / "features.npy", mmap_mode="r", allow_pickle=False)
    artifact, tower = load_tower()
    indices = np.linspace(0, len(records) - 1, PARAMETERS["audit_replay_images"], dtype=int)
    replay = encoder(tower)(np.stack([decode_bytes((DATASET / records[i]["relative_path"]).read_bytes(),
                            records[i]["source_sha256"]) for i in indices]))
    require(np.allclose(replay, new[indices], **PARAMETERS["historical_compatibility"]))
    old = np.load(CLINICAL / "data/local_emb/aireadi_emb_ours.npy", mmap_mode="r", allow_pickle=False)
    compatible = all(np.allclose(new[i:i+256], old[old_rows[i:i+256]], **PARAMETERS["historical_compatibility"])
                     for i in range(0, len(records), 256))
    require(bool(compatible) == a["historical_features_allclose"])
    validate_protocol(p, full=True)
    return {"schema": "bran-retinal-extraction-audit-v1", "status": "authenticated", "protocol_sha256": pin,
        "aggregate_sha256": sha(OUT / "aggregate.json"), "inventory_file_sha256": a["inventory_file_sha256"],
        "output_sha256": value["output_sha256"], "selection": identity,
        "independent_generator_replay_passed": True, "patient_level_output_emitted": False}


def main():
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "audit"):
        group.add_argument("--" + name, action="store_true")
    parser.add_argument("--protocol-sha256"); args = parser.parse_args()
    ok, owned, phase = False, False, "protocol"
    dest = AUDIT if args.audit else OUT
    with _quiet():
        try:
            if args.prepare:
                require(not OUT.exists() and not PRIVATE.exists() and not AUDIT.exists())
                exclusive_json(PROTOCOL, prepare())
            else:
                require(args.protocol_sha256 is not None and sha(PROTOCOL) == args.protocol_sha256)
                p = json.loads(PROTOCOL.read_text()); validate_protocol(p, full=True)
                dest.mkdir(); owned = True
                if args.run:
                    PRIVATE.mkdir(mode=0o700)
                    with open("/private/tmp/bran_retinal_extraction_v1.lock", "a") as lock:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        phase = "extraction"; a = run(p, args.protocol_sha256)
                        exclusive_json(OUT / "aggregate.json", a)
                        native.source.base._atomic_completed(OUT / "progress.json")
                else:
                    phase = "audit"; exclusive_json(AUDIT / "audit.json", audit(p, args.protocol_sha256))
            ok = True
        except Exception as error:
            if owned:
                frames = [{"file": Path(x.filename).name, "line": x.lineno}
                    for x in traceback.extract_tb(error.__traceback__) if Path(x.filename).parent == ROOT and Path(x.filename).name in FILES]
                exclusive_json(dest / "failure.json", {"status": "execution_failed", "phase": phase,
                    "error_class": type(error).__name__ if type(error) in (ValueError, RuntimeError, TypeError, OSError, KeyError) else "other",
                    "code_frames": frames, "patient_level_output_emitted": False})
    print(json.dumps({"operation": "prepare" if args.prepare else "audit" if args.audit else "run",
                      "status": "completed" if ok else "execution_failed", "patient_level_output_emitted": False}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
