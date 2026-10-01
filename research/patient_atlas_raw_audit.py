"""Disclosure-safe audit of authoritative AI-READI inputs for Patient Atlas.

The audit reads patient-derived source tables locally but emits only hashes,
schemas, vocabularies, booleans, and small-cell-suppressed aggregate counts. It
does not read functional target values and never emits identifiers or rows.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "patient-atlas-raw-source-audit-v1"
SMALL_CELL_THRESHOLD = 10
REQUIRED_SPLITS = ("train", "val", "test")
ANALYTE_PREFIXES = ("import_", "lbscat_")
VITAL_PATTERN = re.compile(
    r"sysbp|diabp|pulse|weight|height|bmi|waist|hip|whr", re.IGNORECASE
)
HEMOGLOBIN_SOURCE = "lbscat_a1c"
CONDITION_SOURCES = {
    "hypertension": "mhoccur_hbp",
    "hyperlipidemia": "mhoccur_clsh",
    "cancer": "mhoccur_ca",
    "kidney": "mhoccur_rnl",
    "myocardial_inf": "mhoccur_mi",
    "stroke": "mhoccur_strk",
    "arthritis": "mhoccur_ra",
    "osteoporosis": "mhoccur_oa",
    "heart_failure": "mhoccur_cvdot",
    "chronic_lung": "mhoccur_plm",
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_count(value: int) -> int | str:
    value = int(value)
    if value == 0 or value >= SMALL_CELL_THRESHOLD:
        return value
    return f"<{SMALL_CELL_THRESHOLD}"


def _safe_counts(values: pd.Series) -> dict[str, int | str]:
    counts = values.astype(str).value_counts(dropna=False).sort_index()
    return {str(key): _safe_count(int(value)) for key, value in counts.items()}


def _normalized_vocab(values: Iterable[Any]) -> list[str]:
    result = {
        str(value).strip()
        for value in values
        if pd.notna(value) and str(value).strip() and str(value).strip().lower() != "nan"
    }
    return sorted(result)


def canonical_analyte_name(source_value: str) -> str:
    """Match the historical builder while making the Hb/HbA1c trap explicit."""

    token = str(source_value).split(",", 1)[0].strip()
    if token == HEMOGLOBIN_SOURCE:
        return "hemoglobin"
    return token.removeprefix("import_").removeprefix("lbscat_")


def canonical_vital_name(source_value: str) -> str:
    token = str(source_value).split(",", 1)[0].strip()
    name = "vit_" + token
    return re.sub(r"^vit_bp\d_", "vit_", name)


def map_measurement_features(
    source_values: pd.Series,
    allowed_continuous: set[str],
) -> pd.Series:
    """Return registry feature names or missing for non-input measurements."""

    source = source_values.fillna("").astype(str)
    mapped = pd.Series(pd.NA, index=source.index, dtype="string")
    analyte = source.str.startswith(ANALYTE_PREFIXES)
    if bool(analyte.any()):
        mapped.loc[analyte] = source.loc[analyte].map(canonical_analyte_name)
    vital = ~analyte & source.str.contains(VITAL_PATTERN, na=False)
    if bool(vital.any()):
        mapped.loc[vital] = source.loc[vital].map(canonical_vital_name)
    return mapped.where(mapped.isin(allowed_continuous))


def _load_feature_registry(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    value = json.loads(path.read_text())
    features = value.get("features")
    if not isinstance(features, list) or len(features) != 59:
        raise ValueError("feature registry must contain exactly 59 features")
    continuous = [feature for feature in features if feature.get("type") == "continuous"]
    if len(continuous) != 48:
        raise ValueError("feature registry must contain exactly 48 continuous fields")
    names = [str(feature.get("name")) for feature in continuous]
    if len(names) != len(set(names)):
        raise ValueError("continuous feature names are duplicated")
    return continuous, set(names)


def _measurement_audit(
    dataset_root: Path,
    participants: pd.DataFrame,
    continuous_features: Sequence[Mapping[str, Any]],
    allowed_continuous: set[str],
) -> dict[str, Any]:
    path = dataset_root / "clinical_data" / "measurement.csv"
    columns = (
        "person_id",
        "measurement_date",
        "visit_occurrence_id",
        "measurement_source_value",
        "value_as_number",
        "unit_concept_id",
        "unit_source_value",
        "range_low",
        "range_high",
    )
    measurements = pd.read_csv(path, usecols=columns, low_memory=False)
    visit_path = dataset_root / "clinical_data" / "visit_occurrence.csv"
    visits = pd.read_csv(
        visit_path,
        usecols=("visit_occurrence_id", "visit_start_date"),
        low_memory=False,
    )
    if not visits["visit_occurrence_id"].is_unique:
        raise ValueError("visit_occurrence_id must be unique")
    visits["visit_start_date"] = pd.to_datetime(
        visits["visit_start_date"], errors="coerce", format="mixed"
    )
    measurements["feature"] = map_measurement_features(
        measurements["measurement_source_value"], allowed_continuous
    )
    selected = measurements.loc[measurements["feature"].notna()].copy()
    selected["measurement_date"] = pd.to_datetime(
        selected["measurement_date"], errors="coerce", format="mixed"
    )
    selected = selected.merge(
        participants[["person_id", "study_visit_date"]],
        on="person_id",
        how="left",
        validate="many_to_one",
    )
    selected = selected.merge(
        visits,
        on="visit_occurrence_id",
        how="left",
        validate="many_to_one",
    )
    measurement_date_match = selected["measurement_date"].eq(
        selected["study_visit_date"]
    )
    visit_start_match = selected["visit_start_date"].eq(
        selected["study_visit_date"]
    )
    selected["at_index_visit"] = (
        selected["study_visit_date"].notna()
        & (measurement_date_match | visit_start_match)
    )
    selected["numeric"] = pd.to_numeric(selected["value_as_number"], errors="coerce")

    feature_reports: list[dict[str, Any]] = []
    for feature in continuous_features:
        name = str(feature["name"])
        block = str(feature["block"])
        rows = selected.loc[selected["feature"].eq(name)]
        observed = rows.loc[rows["numeric"].notna()]
        index_rows = observed.loc[observed["at_index_visit"]]
        replicate_sizes = index_rows.groupby("person_id", sort=False).size()
        reference_pairs = (
            rows[["range_low", "range_high"]]
            .dropna(how="all")
            .drop_duplicates()
        )
        unit_pairs = (
            rows[["unit_source_value", "unit_concept_id"]]
            .dropna(how="all")
            .drop_duplicates()
        )
        feature_reports.append(
            {
                "index": int(feature["index"]),
                "name": name,
                "block": block,
                "source_values": _normalized_vocab(rows["measurement_source_value"]),
                "unit_source_values": _normalized_vocab(rows["unit_source_value"]),
                "unit_concept_ids": _normalized_vocab(rows["unit_concept_id"]),
                "unit_pair_count": int(len(unit_pairs)),
                "reference_range_pair_count": int(len(reference_pairs)),
                "observed_rows": _safe_count(len(observed)),
                "observed_patients": _safe_count(observed["person_id"].nunique()),
                "index_visit_rows": _safe_count(len(index_rows)),
                "index_visit_patients": _safe_count(index_rows["person_id"].nunique()),
                "off_index_visit_rows": _safe_count(
                    int((observed["at_index_visit"] == False).sum())  # noqa: E712
                ),
                "patients_with_index_visit_replicates": _safe_count(
                    int((replicate_sizes > 1).sum())
                ),
                "single_unit_pair_at_index_visit": bool(
                    len(
                        index_rows[["unit_source_value", "unit_concept_id"]]
                        .dropna(how="all")
                        .drop_duplicates()
                    )
                    <= 1
                ),
            }
        )

    return {
        "file": path.name,
        "file_sha256": _sha256(path),
        "visit_file": visit_path.name,
        "visit_file_sha256": _sha256(visit_path),
        "index_visit_rule": "measurement_date equals participants.study_visit_date OR the linked visit_start_date equals participants.study_visit_date",
        "input_rows": _safe_count(len(measurements)),
        "selected_continuous_rows": _safe_count(len(selected)),
        "selected_rows_with_linked_visit": _safe_count(
            int(selected["visit_start_date"].notna().sum())
        ),
        "measurement_date_index_matches": _safe_count(
            int(measurement_date_match.sum())
        ),
        "linked_visit_start_index_matches": _safe_count(
            int(visit_start_match.sum())
        ),
        "either_index_match": _safe_count(int(selected["at_index_visit"].sum())),
        "present_feature_count": int(selected["feature"].nunique()),
        "features": feature_reports,
    }


def _condition_audit(
    dataset_root: Path,
    participants: pd.DataFrame,
) -> dict[str, Any]:
    path = dataset_root / "clinical_data" / "observation.csv"
    columns = (
        "person_id",
        "observation_date",
        "visit_occurrence_id",
        "observation_source_value",
        "value_as_number",
    )
    observations = pd.read_csv(path, usecols=columns, low_memory=False)
    visit_path = dataset_root / "clinical_data" / "visit_occurrence.csv"
    visits = pd.read_csv(
        visit_path,
        usecols=("visit_occurrence_id", "visit_start_date"),
        low_memory=False,
    )
    if not visits["visit_occurrence_id"].is_unique:
        raise ValueError("visit_occurrence_id must be unique")
    visits["visit_start_date"] = pd.to_datetime(
        visits["visit_start_date"], errors="coerce", format="mixed"
    )
    observations["source"] = (
        observations["observation_source_value"]
        .fillna("")
        .astype(str)
        .str.split(",", n=1)
        .str[0]
        .str.strip()
    )
    source_to_condition = {source: name for name, source in CONDITION_SOURCES.items()}
    selected = observations.loc[observations["source"].isin(source_to_condition)].copy()
    selected["condition"] = selected["source"].map(source_to_condition)
    selected["observation_date"] = pd.to_datetime(
        selected["observation_date"], errors="coerce", format="mixed"
    )
    selected = selected.merge(
        participants[["person_id", "study_visit_date"]],
        on="person_id",
        how="left",
        validate="many_to_one",
    )
    selected = selected.merge(
        visits,
        on="visit_occurrence_id",
        how="left",
        validate="many_to_one",
    )
    observation_date_match = selected["observation_date"].eq(
        selected["study_visit_date"]
    )
    visit_start_match = selected["visit_start_date"].eq(
        selected["study_visit_date"]
    )
    selected["at_index_visit"] = (
        selected["study_visit_date"].notna()
        & (observation_date_match | visit_start_match)
    )
    selected["numeric"] = pd.to_numeric(selected["value_as_number"], errors="coerce")
    selected["valid_binary"] = selected["numeric"].isin((0.0, 1.0))

    reports: list[dict[str, Any]] = []
    for condition, source in CONDITION_SOURCES.items():
        rows = selected.loc[selected["condition"].eq(condition)]
        index_rows = rows.loc[rows["at_index_visit"]]
        valid = index_rows.loc[index_rows["valid_binary"]]
        per_patient_sizes = valid.groupby("person_id", sort=False).size()
        per_patient_unique = valid.groupby("person_id", sort=False)["numeric"].nunique()
        reports.append(
            {
                "name": condition,
                "source_value": source,
                "observed_rows": _safe_count(len(rows)),
                "index_visit_rows": _safe_count(len(index_rows)),
                "index_visit_patients": _safe_count(index_rows["person_id"].nunique()),
                "valid_binary_rows": _safe_count(len(valid)),
                "invalid_or_sentinel_rows": _safe_count(
                    int((~index_rows["valid_binary"]).sum())
                ),
                "off_index_visit_rows": _safe_count(
                    int((rows["at_index_visit"] == False).sum())  # noqa: E712
                ),
                "patients_with_index_visit_replicates": _safe_count(
                    int((per_patient_sizes > 1).sum())
                ),
                "patients_with_conflicting_binary_values": _safe_count(
                    int((per_patient_unique > 1).sum())
                ),
            }
        )

    recruitment = participants["study_group"].astype(str)
    reports.insert(
        2,
        {
            "name": "diabetes",
            "source_value": "participants.study_group",
            "definition": "1 for oral-medication/non-insulin-controlled or insulin-dependent; 0 otherwise",
            "available_patients": _safe_count(int(participants["study_group"].notna().sum())),
            "study_group_counts": _safe_counts(recruitment),
            "visit_policy": "participant recruitment-group field aligned to the declared study visit",
        },
    )
    return {
        "file": path.name,
        "file_sha256": _sha256(path),
        "visit_file": visit_path.name,
        "visit_file_sha256": _sha256(visit_path),
        "index_visit_rule": "observation_date equals participants.study_visit_date OR the linked visit_start_date equals participants.study_visit_date",
        "input_rows": _safe_count(len(observations)),
        "selected_condition_rows": _safe_count(len(selected)),
        "selected_rows_with_linked_visit": _safe_count(
            int(selected["visit_start_date"].notna().sum())
        ),
        "observation_date_index_matches": _safe_count(
            int(observation_date_match.sum())
        ),
        "linked_visit_start_index_matches": _safe_count(
            int(visit_start_match.sum())
        ),
        "either_index_match": _safe_count(int(selected["at_index_visit"].sum())),
        "condition_count": len(reports),
        "conditions": reports,
    }


def _normalized_retinal_path(value: str) -> str:
    path = str(value).replace("\\", "/").lstrip("./")
    marker = "retinal_photography/"
    position = path.find(marker)
    return path[position:] if position >= 0 else path


def _retinal_audit(
    dataset_root: Path,
    participants: pd.DataFrame,
    embedding_path: Path,
) -> dict[str, Any]:
    manifest_path = dataset_root / "retinal_photography" / "manifest.tsv"
    manifest_columns = (
        "person_id",
        "manufacturer",
        "manufacturers_model_name",
        "laterality",
        "filepath",
    )
    manifest = pd.read_csv(manifest_path, sep="\t", usecols=manifest_columns)
    normalized = manifest["filepath"].astype(str).map(_normalized_retinal_path)
    cfp_mask = normalized.str.contains(r"(?:^|/)cfp(?:/|$)", regex=True, na=False)
    cfp = manifest.loc[cfp_mask].copy()
    cfp["normalized_path"] = normalized.loc[cfp_mask]

    # Reproduce encode_aireadi_ours.py exactly. pathlib.Path.glob traverses a
    # symlinked Triton subtree on the current volume and finds 2,310 additional
    # files; stdlib glob.glob is the frozen encoder enumeration rule.
    glob_pattern = str(dataset_root / "retinal_photography" / "cfp" / "**" / "*.dcm")
    glob_paths = [Path(value) for value in sorted(glob.glob(glob_pattern, recursive=True))]
    normalized_glob = [_normalized_retinal_path(str(path)) for path in glob_paths]
    manifest_basenames = cfp["normalized_path"].map(lambda value: Path(value).name)
    glob_basenames = [path.name for path in glob_paths]
    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError("eye embedding artifact must be a rank-2 NPY array")
    finite_rows = np.isfinite(embeddings).all(axis=1)
    nonzero_rows = np.abs(embeddings).sum(axis=1) > 0

    participant_ids = set(participants["person_id"].astype(str))
    cfp_ids = cfp["person_id"].astype(str)
    return {
        "manifest_file": manifest_path.name,
        "manifest_sha256": _sha256(manifest_path),
        "manifest_rows": _safe_count(len(manifest)),
        "manifest_cfp_rows": _safe_count(len(cfp)),
        "manifest_cfp_patients": _safe_count(cfp_ids.nunique()),
        "manifest_unknown_participant_rows": _safe_count(
            int((~cfp_ids.isin(participant_ids)).sum())
        ),
        "manifest_duplicate_cfp_paths": _safe_count(
            int(cfp["normalized_path"].duplicated().sum())
        ),
        "physical_cfp_files": _safe_count(len(glob_paths)),
        "physical_enumeration_rule": "sorted stdlib glob.glob('**/*.dcm', recursive=True), matching encode_aireadi_ours.py",
        "physical_paths_unique": len(normalized_glob) == len(set(normalized_glob)),
        "manifest_physical_path_sets_equal": set(cfp["normalized_path"])
        == set(normalized_glob),
        "manifest_cfp_basenames_unique": bool(manifest_basenames.is_unique),
        "physical_cfp_basenames_unique": len(glob_basenames)
        == len(set(glob_basenames)),
        "manifest_physical_basename_sets_equal": set(manifest_basenames)
        == set(glob_basenames),
        "embedding_file": embedding_path.name,
        "embedding_file_sha256": _sha256(embedding_path),
        "embedding_shape": [int(value) for value in embeddings.shape],
        "embedding_dtype": str(embeddings.dtype),
        "embedding_rows_match_sorted_glob": int(embeddings.shape[0])
        == len(glob_paths),
        "embedding_nonfinite_rows": _safe_count(int((~finite_rows).sum())),
        "embedding_zero_rows": _safe_count(int((~nonzero_rows).sum())),
        "laterality_counts": _safe_counts(cfp["laterality"]),
        "manufacturer_vocabulary": _normalized_vocab(cfp["manufacturer"]),
        "model_vocabulary": _normalized_vocab(cfp["manufacturers_model_name"]),
    }


def audit_raw_sources(
    *,
    dataset_root: str | Path,
    feature_registry_path: str | Path,
    embedding_path: str | Path,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    feature_registry_path = Path(feature_registry_path).resolve()
    embedding_path = Path(embedding_path).resolve()
    participants_path = dataset_root / "participants.tsv"
    participants_schema_path = dataset_root / "participants.json"
    required = (
        participants_path,
        participants_schema_path,
        dataset_root / "clinical_data" / "measurement.csv",
        dataset_root / "clinical_data" / "observation.csv",
        dataset_root / "clinical_data" / "visit_occurrence.csv",
        dataset_root / "retinal_photography" / "manifest.tsv",
        feature_registry_path,
        embedding_path,
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required source artifacts are missing: " + ", ".join(missing))

    continuous_features, allowed_continuous = _load_feature_registry(
        feature_registry_path
    )
    participant_columns = (
        "person_id",
        "clinical_site",
        "study_group",
        "age",
        "study_visit_date",
        "recommended_split",
    )
    participants = pd.read_csv(
        participants_path, sep="\t", usecols=participant_columns
    )
    participants["study_visit_date"] = pd.to_datetime(
        participants["study_visit_date"], errors="coerce"
    )
    split_values = set(participants["recommended_split"].dropna().astype(str))
    if split_values != set(REQUIRED_SPLITS):
        raise ValueError("recommended_split must contain exactly train, val, and test")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "privacy": {
            "patient_rows_emitted": False,
            "patient_identifiers_emitted": False,
            "functional_target_values_read": False,
            "small_cell_threshold": SMALL_CELL_THRESHOLD,
        },
        "participants": {
            "file_sha256": _sha256(participants_path),
            "schema_file_sha256": _sha256(participants_schema_path),
            "rows": _safe_count(len(participants)),
            "patient_ids_unique": bool(participants["person_id"].is_unique),
            "missing_age": _safe_count(int(participants["age"].isna().sum())),
            "missing_study_visit_date": _safe_count(
                int(participants["study_visit_date"].isna().sum())
            ),
            "split_counts": _safe_counts(participants["recommended_split"]),
            "site_counts": _safe_counts(participants["clinical_site"]),
        },
        "measurements": _measurement_audit(
            dataset_root,
            participants,
            continuous_features,
            allowed_continuous,
        ),
        "conditions": _condition_audit(dataset_root, participants),
        "retinal": _retinal_audit(dataset_root, participants, embedding_path),
        "unread_sources": {
            "functional_targets": "not read; official test outcomes remain sealed",
        },
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--feature-registry",
        type=Path,
        default=Path(__file__).resolve().with_name(
            "PATIENT_ATLAS_FEATURE_REGISTRY.json"
        ),
    )
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = audit_raw_sources(
        dataset_root=args.dataset_root,
        feature_registry_path=args.feature_registry,
        embedding_path=args.embedding,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(encoded)
        compact = {
            "schema_version": report["schema_version"],
            "report_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "participants": report["participants"]["rows"],
            "continuous_features_present": report["measurements"][
                "present_feature_count"
            ],
            "embedding_shape": report["retinal"]["embedding_shape"],
            "embedding_rows_match_sorted_glob": report["retinal"][
                "embedding_rows_match_sorted_glob"
            ],
            "manifest_physical_basename_sets_equal": report["retinal"][
                "manifest_physical_basename_sets_equal"
            ],
            "output": str(args.output),
        }
        print(json.dumps(compact, indent=2, sort_keys=True))
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "SMALL_CELL_THRESHOLD",
    "audit_raw_sources",
    "canonical_analyte_name",
    "canonical_vital_name",
    "map_measurement_features",
]
