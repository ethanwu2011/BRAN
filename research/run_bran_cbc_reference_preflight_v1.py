"""Exclusive local source-reference preflight; no model fitting or efficacy claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_missingness_stress_metrics_v1 import safe_coverage
from run_bran_source_linkage_audit_v1 import exclusive_json, sha
import bran_cbc_source_reference_io_v1 as reading
import bran_clinical_utility_schema_v1 as location
import run_bran_native_cbc_calibration_v1 as calibration

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT/"BRAN_CBC_REFERENCE_PREFLIGHT_PROTOCOL_V1.json"
OUT = ROOT/"BRAN_CBC_REFERENCE_PREFLIGHT_V1"
AUDIT = ROOT/"BRAN_CBC_REFERENCE_PREFLIGHT_AUDIT_V1"
CALIBRATION_PIN = "acd5a6fa5415e691c0c8c56c02496b37811b1530ef8409462be6a6ef0e2e8edb"
CODE = ("run_bran_cbc_reference_preflight_v1.py", "test_run_bran_cbc_reference_preflight_v1.py",
        "bran_cbc_source_reference_io_v1.py", "test_bran_cbc_source_reference_io_v1.py",
        "bran_cbc_source_reference_v1.py", "test_bran_cbc_source_reference_v1.py",
        "bran_cbc_reference_band_v1.py", "test_bran_cbc_reference_band_v1.py",
        "bran_clinical_utility_schema_v1.py", "BRAN_CBC_SOURCE_REFERENCE_DESIGN_V1.md")
FLAGS = {"patient_level_output_emitted": False, "reference_endpoints_emitted": False,
         "clinical_diagnosis_claim": False, "model_fitting_performed": False,
         "prediction_metrics_computed": False, "official_test_used": False,
         "source_targets_replayed": True, "source_bytes_reauthenticated": True}


def require(ok):
    if not ok:raise ValueError("cbc_reference_preflight_contract_failed")


def prepare():
    require(sha(calibration.PROTOCOL)==CALIBRATION_PIN)
    old=json.loads(calibration.PROTOCOL.read_text());calibration.validate_protocol(old)
    require(sha(ROOT/location.SOURCE_PROTOCOL)==location.SOURCE_PIN)
    dataset=json.loads((ROOT/location.SOURCE_PROTOCOL).read_text())["data_roots"]["dataset_root"]
    policy_path=ROOT/"PATIENT_ATLAS_SOURCE_POLICY.json"
    policy=json.loads(policy_path.read_text())
    require(policy["schema_version"]=="patient-atlas-source-policy-v1"
            and policy["scope"]=="exploratory_train_validation_only")
    hashes={key:policy["source_hashes"][key] for key in reading.FILES}
    return {"schema":"bran-cbc-reference-preflight-protocol-v1","status":"frozen_before_execution",
        "calibration_protocol_sha256":CALIBRATION_PIN,"native_source":old["native_source"],
        "source_location_protocol_sha256":location.SOURCE_PIN,"dataset_root":dataset,
        "source_policy_sha256":sha(policy_path),"source_hashes":hashes,
        "code_sha256":{**old["code_sha256"],**{name:sha(ROOT/name) for name in CODE}},
        "scope":"exact1928_train_val_index_CBC_source_range_qualification_only",
        "release":"minimum20_with_complementary_suppression_no_ranges_or_abnormality_counts",
        "runtime":old["runtime"]}


def validate_protocol(p):require(p==prepare())


def validate_result(a,p):
    require(type(a)is dict and set(a)=={"schema","status","people","field_reference_coverage",
        "fold_authentication","source_policy_sha256","source_hashes","private_array_fingerprint"}|set(FLAGS))
    require(a["schema"]=="bran-cbc-reference-preflight-aggregate-v1" and a["status"]=="completed"
            and type(a["people"])is int and a["people"]==1928)
    require(all(a[key] is value for key,value in FLAGS.items()))
    require(a["fold_authentication"]==p["native_source"]["source"]["authentication"]
            and a["source_policy_sha256"]==p["source_policy_sha256"] and a["source_hashes"]==p["source_hashes"])
    require(type(a["field_reference_coverage"])is dict and set(a["field_reference_coverage"])==set(CBC_FIELDS))
    for coverage in a["field_reference_coverage"].values():calibration.check_coverage(coverage)
    require(type(a["private_array_fingerprint"])is str and len(a["private_array_fingerprint"])==64
            and all(c in "0123456789abcdef" for c in a["private_array_fingerprint"]))


def compute(p):
    source=calibration.origin.native.source
    ctx,folds,c0,cm0,eligible,r0,rm,ages,names=source.io.load_context()
    raw=ctx["raw_cohort"]; ids=tuple(map(str,raw.patient_ids))
    require(len(ids)==1928 and raw.source_policy_sha256==p["source_policy_sha256"])
    require(set(raw.split_labels)<= {"train","val"} and set(np.asarray(folds).tolist())==set(range(5)))
    for fold in range(5):
        _,identity=source.base._inner_context(ctx,np.flatnonzero(folds!=fold),fold)
        require(identity==p["native_source"]["source"]["authentication"]["inner_fold_sha256"][fold])
    slots=tuple(names.index(field) for field in CBC_FIELDS)
    targets=np.asarray(c0[:,slots]);observed=np.asarray((cm0&eligible)[:,slots],bool)
    lower,upper,valid=reading.read_references(p["dataset_root"],ids,targets,observed,p["source_hashes"])
    # One global fingerprint, never per-person hashes, IDs, values or ranges.
    digest=hashlib.sha256()
    for array in (lower,upper,valid):
        digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    result={"schema":"bran-cbc-reference-preflight-aggregate-v1","status":"completed","people":1928,
        "field_reference_coverage":{field:safe_coverage(valid[observed[:,j],j]) for j,field in enumerate(CBC_FIELDS)},
        "fold_authentication":p["native_source"]["source"]["authentication"],
        "source_policy_sha256":p["source_policy_sha256"],"source_hashes":p["source_hashes"],
        "private_array_fingerprint":digest.hexdigest(),**FLAGS}
    validate_result(result,p)
    return result


def audit(p,pin):
    require(not(OUT/"failure.json").exists())
    raw=(OUT/"aggregate.json").read_bytes();a=json.loads(raw);validate_result(a,p)
    manifest=json.loads((OUT/"aggregate.manifest.json").read_text())
    require(manifest=={"protocol_sha256":pin,"artifact_sha256":hashlib.sha256(raw).hexdigest()})
    require(compute(p)==a)
    require((OUT/"aggregate.json").read_bytes()==raw)
    validate_protocol(p)
    return {"schema":"bran-cbc-reference-preflight-audit-v1","status":"authenticated",
            "protocol_sha256":pin,"aggregate_sha256":hashlib.sha256(raw).hexdigest(),
            "source_and_private_array_fingerprint_replayed":True,"patient_level_output_emitted":False}


def publish(dest,name,result,pin):
    require(name in ("aggregate.json","audit.json"))
    stage=dest/(name+".staged")
    exclusive_json(stage,result)
    exclusive_json(dest/(Path(name).stem+".manifest.json"),
                   {"protocol_sha256":pin,"artifact_sha256":sha(stage)})
    # A successful exclusive terminal file is the final, atomic operation.
    os.link(stage,dest/name)


def main():
    parser=argparse.ArgumentParser();op=parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare","run","audit"):op.add_argument("--"+name,action="store_true")
    parser.add_argument("--protocol-sha256");args=parser.parse_args()
    owned=False;success=False;phase="protocol";dest=AUDIT if args.audit else OUT
    with _quiet():
        try:
            if args.prepare:
                require(not OUT.exists() and not AUDIT.exists())
                exclusive_json(PROTOCOL,prepare())
            else:
                require(args.protocol_sha256 and sha(PROTOCOL)==args.protocol_sha256)
                p=json.loads(PROTOCOL.read_text());validate_protocol(p)
                dest.mkdir();owned=True
                if args.audit:
                    phase="source_reference_replay"
                    result=audit(p,args.protocol_sha256);name="audit.json"
                else:
                    phase="source_reference_qualification"
                    result=compute(p);validate_protocol(p);name="aggregate.json"
                phase="publishing";publish(dest,name,result,args.protocol_sha256)
            success=True
        except Exception as error:
            if owned and not any((dest/name).exists() for name in ("aggregate.json","audit.json")):
                frames=[{"file":Path(x.filename).name,"line":x.lineno} for x in traceback.extract_tb(error.__traceback__)
                        if Path(x.filename).parent==ROOT and Path(x.filename).name in CODE]
                exclusive_json(dest/"failure.json",{"status":"execution_failed","phase":phase,
                    "error_class":type(error).__name__ if type(error) in (ValueError,TypeError,KeyError,RuntimeError,OSError) else "other",
                    "code_frames":frames,"patient_level_output_emitted":False})
    print(json.dumps({"status":"completed" if success else "execution_failed","patient_level_output_emitted":False}))
    return 0 if success else 1


if __name__=="__main__":raise SystemExit(main())
