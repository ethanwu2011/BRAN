"""Local source-bound CBC reference input; call only under FD suppression.

This module does not fit models, evaluate predictions, or write patient arrays.
It reconstructs the existing index-visit CBC targets before associating source
laboratory ranges. It deliberately does not create sex-specific clinical rules.
"""
import csv
import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd

from bran_clinical_semantics_v1 import CBC_FIELDS
from patient_atlas_raw_audit import map_measurement_features
import bran_cbc_source_reference_v1 as references

FILES = {"participants_tsv": "participants.tsv", "measurement_csv": "clinical_data/measurement.csv",
         "visit_occurrence_csv": "clinical_data/visit_occurrence.csv"}
MEASUREMENT_COLUMNS = ("person_id", "measurement_date", "visit_occurrence_id", "measurement_source_value",
                       "value_as_number", "unit_source_value", "range_low", "range_high", "operator_concept_id")
MAX_HEADER = 65536


def require(ok):
    if not ok:
        raise ValueError("cbc_source_reference_io_contract_failed")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def authenticate_sources(dataset_root, expected_hashes):
    require(type(expected_hashes) is dict and set(expected_hashes) == set(FILES))
    require(all(type(x) is str and re.fullmatch(r"[0-9a-f]{64}", x) for x in expected_hashes.values()))
    root = Path(dataset_root)
    require(root.is_absolute())
    require(all(sha(root/rel) == expected_hashes[key] for key,rel in FILES.items()))


def require_header(path, columns, delimiter=","):
    """Permit exactly one empty leading export-index name; no duplicate names.

    All actual record parsing uses explicit named columns. An unnamed leading
    index is never interpreted as patient identity or as a measurement.
    """
    with Path(path).open("rb") as handle:
        raw = handle.readline(MAX_HEADER+1)
    require(0 < len(raw) < MAX_HEADER and raw.endswith(b"\n"))
    fields = next(csv.reader([raw.decode("utf-8-sig").rstrip("\r\n")], delimiter=delimiter, strict=True))
    if fields and fields[0] == "":
        fields = fields[1:]
    require(len(fields) > 0 and len(fields) == len({x.casefold() for x in fields}))
    require(all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", x) for x in fields))
    require(set(columns) <= set(fields))


def read_references(dataset_root, patient_ids, target_N9, observed_N9, expected_hashes):
    """Recreate source targets and ranges for the exact existing train/val cohort.

    Measurement-date OR linked visit-date index rule is inherited unchanged.
    Contradictory visit/person links fail; no row-order matching or target-based
    choice among duplicates. Nonzero/unparseable operator codes disqualify range
    interpretation rather than assuming an inequality is an exact observation.
    """
    root = Path(dataset_root)
    ids = tuple(patient_ids)
    require(bool(ids) and all(type(x) is str and x for x in ids) and len(ids) == len(set(ids)))
    authenticate_sources(root, expected_hashes)
    participant_columns = ("person_id", "study_visit_date", "recommended_split")
    visit_columns = ("person_id", "visit_occurrence_id", "visit_start_date")
    require_header(root/FILES["participants_tsv"], participant_columns, "\t")
    require_header(root/FILES["visit_occurrence_csv"], visit_columns)
    require_header(root/FILES["measurement_csv"], MEASUREMENT_COLUMNS)
    participants = pd.read_csv(root/FILES["participants_tsv"], sep="\t", usecols=participant_columns)
    require(participants["person_id"].notna().all() and participants["person_id"].is_unique)
    participants["person_id"] = participants["person_id"].astype(str)
    participants = participants.loc[participants["recommended_split"].isin(("train", "val"))].copy()
    require(set(participants["person_id"]) == set(ids) and len(participants) == len(ids))
    participants["study_visit_date"] = pd.to_datetime(participants["study_visit_date"], errors="coerce", format="mixed")
    require(participants["study_visit_date"].notna().all())
    visits = pd.read_csv(root/FILES["visit_occurrence_csv"], usecols=visit_columns)
    require(visits["visit_occurrence_id"].notna().all() and visits["visit_occurrence_id"].is_unique
            and visits["person_id"].notna().all())
    visits["person_id"] = visits["person_id"].astype(str)
    visits = visits.rename(columns={"person_id": "visit_person_id"})
    visits["visit_start_date"] = pd.to_datetime(visits["visit_start_date"], errors="coerce", format="mixed")
    allowed = set(ids)

    def selected_rows():
        # Source chunks stay local. Filter people and CBC fields BEFORE numeric
        # conversion or any reference assessment; official-test rows never enter
        # the returned reference assembler or model evaluation.
        with pd.read_csv(root/FILES["measurement_csv"], usecols=MEASUREMENT_COLUMNS,
                         low_memory=False, chunksize=50000) as chunks:
            for chunk in chunks:
                selected = chunk.loc[chunk["person_id"].astype(str).isin(allowed)].copy()
                selected["person_id"] = selected["person_id"].astype(str)
                selected["field"] = map_measurement_features(selected["measurement_source_value"], set(CBC_FIELDS))
                selected = selected.loc[selected["field"].notna()].copy()
                if selected.empty:
                    continue
                selected["measurement_date"] = pd.to_datetime(selected["measurement_date"], errors="coerce", format="mixed")
                selected = selected.merge(participants[["person_id", "study_visit_date"]], on="person_id", validate="many_to_one")
                selected = selected.merge(visits, on="visit_occurrence_id", how="left", validate="many_to_one")
                known_visit = selected["visit_person_id"].notna()
                require(selected.loc[known_visit, "visit_person_id"].eq(selected.loc[known_visit, "person_id"]).all())
                selected = selected.loc[selected["measurement_date"].eq(selected["study_visit_date"])
                                        | selected["visit_start_date"].eq(selected["study_visit_date"])].copy()
                for name in ("value_as_number", "range_low", "range_high"):
                    selected[name] = pd.to_numeric(selected[name], errors="coerce")
                raw_operator = selected["operator_concept_id"]
                operator = pd.to_numeric(raw_operator, errors="coerce")
                literal = raw_operator.isna() | operator.eq(0)
                # Unknown/nonliteral operator => no range qualification. It cannot
                # create a healthy label or modify the previously frozen target.
                selected.loc[~literal, "unit_source_value"] = None
                for row in selected.itertuples(index=False):
                    yield {name: getattr(row, name) for name in references.ROW_KEYS}

    rows = selected_rows()
    try:
        result = references.assemble(ids, target_N9, observed_N9, rows)
    finally:
        rows.close()
    authenticate_sources(root, expected_hashes)
    return result
