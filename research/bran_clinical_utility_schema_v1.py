"""Header-only readiness for CBC references and independent ICU outcomes.

No patient records are parsed. This is NOT unit, outcome, linkage, population,
or full-source authentication. Only fixed requested column names are released.
Existing caches and all frozen scientific protocols remain unchanged.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

MAX_HEADER = 65536
ROOT = Path(__file__).resolve().parent
SOURCE_PROTOCOL = "BRAN_INDEPENDENT_PHENOTYPE_PREFLIGHT_PROTOCOL_V1.json"
SOURCE_PIN = "58d742a4c00219efce99b03f4ae8c64223afbe355ca4e0d0448274870d06e179"
OUT = ROOT / "BRAN_CLINICAL_UTILITY_SCHEMA_V1.json"

# Candidate fields from existing local readers / source-table schemas. Header
# presence does not validate values, population eligibility, or outcome timing.
AIREADI = {
    "measurement": ("clinical_data/measurement.csv", ",", (
        "person_id", "measurement_date", "measurement_datetime", "visit_occurrence_id",
        "measurement_source_value", "value_as_number", "unit_concept_id", "unit_source_value",
        "range_low", "range_high")),
    "visits": ("clinical_data/visit_occurrence.csv", ",", (
        "person_id", "visit_occurrence_id", "visit_start_date")),
    "participants": ("participants.tsv", "\t", (
        "person_id", "study_visit_date", "age", "sex", "gender", "pregnancy_status")),
    "person": ("clinical_data/person.csv", ",", (
        "person_id", "gender_concept_id", "gender_source_value", "year_of_birth")),
}
ICU = {
    "mimic_patients": ("/Users/ethanwu/mimiciv-3.1/hosp/patients.csv.gz", (
        "subject_id", "gender", "anchor_age", "anchor_year", "dod")),
    "mimic_admissions": ("/Users/ethanwu/mimiciv-3.1/hosp/admissions.csv.gz", (
        "subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag")),
    "mimic_labs": ("/Users/ethanwu/mimiciv-3.1/hosp/labevents.csv.gz", (
        "subject_id", "hadm_id", "itemid", "charttime", "storetime", "valuenum", "valueuom",
        "ref_range_lower", "ref_range_upper")),
    "mimic_diagnoses": ("/Users/ethanwu/mimiciv-3.1/hosp/diagnoses_icd.csv.gz", (
        "subject_id", "hadm_id", "seq_num", "icd_code", "icd_version", "diagnosis_time")),
    "eicu_patients": ("/Users/ethanwu/eicu-crd-2.0/patient.csv.gz", (
        "uniquepid", "patientunitstayid", "patienthealthsystemstayid", "age", "gender",
        "hospitaladmitoffset", "hospitaldischargeoffset", "hospitaldischargestatus",
        "unitdischargeoffset", "unitdischargestatus")),
    "eicu_diagnoses": ("/Users/ethanwu/eicu-crd-2.0/diagnosis.csv.gz", (
        "patientunitstayid", "diagnosisoffset", "diagnosisstring", "icd9code")),
}


def require(ok):
    if not ok:
        raise ValueError("clinical_utility_schema_contract_failed")


def signature(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def failed(status):
    return {"status": status, "patient_records_parsed": False}


def inspect_header(path, required_columns, delimiter=","):
    try:
        require(type(delimiter) is str and delimiter in (",", "\t"))
        require(isinstance(required_columns, (tuple, list)) and len(required_columns) > 0)
        require(all(type(x) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", x) for x in required_columns))
        require(len(set(x.casefold() for x in required_columns)) == len(required_columns))
        path = Path(path)
        if not path.is_file():
            return failed("not_present")
        before = signature(path.stat())
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rb") as handle:
            # A single bounded logical header read. Never iterate CSV records.
            header = handle.readline(MAX_HEADER + 1)
        if before != signature(path.stat()):
            return failed("source_changed")
        require(0 < len(header) < MAX_HEADER and header.endswith(b"\n"))
        line = header.decode("utf-8-sig").rstrip("\r\n")
        fields = next(csv.reader([line], delimiter=delimiter, strict=True))
        normalized = [x.strip().casefold() for x in fields]
        require(bool(normalized) and len(set(normalized)) == len(normalized))
        require(all(re.fullmatch(r"[a-z_][a-z0-9_]*", x) for x in normalized))
        return {"status": "header_checked",
                "columns_present": {x: x.casefold() in normalized for x in required_columns},
                "header_sha256": hashlib.sha256(header).hexdigest(),
                "patient_records_parsed": False}
    except Exception:
        # No exception text, source paths, unknown headers or record text.
        return failed("header_invalid_or_unreadable")


def expected_tables():
    return {**{"aireadi_"+k: v[2] for k,v in AIREADI.items()},
            **{k: v[1] for k,v in ICU.items()}}


FLAGS = {"patient_records_parsed": False, "predictions_or_states_read": False,
         "full_source_payload_authenticated": False, "reference_values_validated": False,
         "outcomes_or_linkage_validated": False, "training_or_evaluation_performed": False,
         "scientific_gates_changed": False}


def validate_report(report):
    require(type(report) is dict and set(report) == {"schema", "status", "source_protocol_sha256",
        "code_sha256", "tables"} | set(FLAGS))
    require(report["schema"] == "bran-clinical-utility-schema-v1" and report["status"] == "completed_headers_only")
    require(report["source_protocol_sha256"] == SOURCE_PIN)
    require(type(report["code_sha256"]) is dict and set(report["code_sha256"]) ==
            {"bran_clinical_utility_schema_v1.py", "test_bran_clinical_utility_schema_v1.py"})
    require(all(type(h) is str and re.fullmatch(r"[0-9a-f]{64}", h) for h in report["code_sha256"].values()))
    require(all(report[k] is v for k,v in FLAGS.items()))
    expected = expected_tables()
    require(type(report["tables"]) is dict and set(report["tables"]) == set(expected))
    for key, cell in report["tables"].items():
        require(type(cell) is dict and cell.get("patient_records_parsed") is False)
        if cell.get("status") == "header_checked":
            require(set(cell) == {"status", "columns_present", "header_sha256", "patient_records_parsed"})
            require(type(cell["columns_present"]) is dict and set(cell["columns_present"]) == set(expected[key]))
            require(all(type(v) is bool for v in cell["columns_present"].values()))
            require(type(cell["header_sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}",cell["header_sha256"]))
        else:
            require(set(cell) == {"status", "patient_records_parsed"}
                    and cell["status"] in ("not_present", "source_changed", "header_invalid_or_unreadable"))


def run(root=ROOT):
    # Caller must keep the complete operation inside a local FD-quiet boundary.
    root = Path(root)
    raw = (root / SOURCE_PROTOCOL).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SOURCE_PIN)
    dataset = Path(json.loads(raw)["data_roots"]["dataset_root"])
    require(dataset.is_absolute())
    report = {"schema": "bran-clinical-utility-schema-v1", "status": "completed_headers_only",
        "source_protocol_sha256": SOURCE_PIN,
        "code_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ("bran_clinical_utility_schema_v1.py", "test_bran_clinical_utility_schema_v1.py")},
        "tables": {**{"aireadi_"+key: inspect_header(dataset/rel, cols, sep)
            for key,(rel,sep,cols) in AIREADI.items()},
            **{key: inspect_header(path, cols) for key,(path,cols) in ICU.items()}}, **FLAGS}
    require(hashlib.sha256((root/SOURCE_PROTOCOL).read_bytes()).hexdigest() == SOURCE_PIN)
    require(all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
                for name,digest in report["code_sha256"].items()))
    validate_report(report)
    return report


def main():
    from bran_clinical_dictionary_binding_v1 import _quiet
    from run_bran_source_linkage_audit_v1 import exclusive_json
    try:
        with _quiet():
            require(not OUT.exists())
            report = run()
            exclusive_json(OUT, report)
        print(json.dumps({"status": "completed_headers_only", "artifact_sha256":
            hashlib.sha256(OUT.read_bytes()).hexdigest()}))
        return 0
    except Exception:
        print(json.dumps({"status": "header_readiness_failed_no_raw_error_emitted"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
