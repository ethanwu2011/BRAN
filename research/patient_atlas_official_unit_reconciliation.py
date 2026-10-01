"""Reconcile Atlas clinical units against pinned official AI-READI v2 docs.

This prospective governance artifact does not rewrite the fitted registry or
any existing exploratory model.  It combines the row-free release audit with a
hash-pinned official data dictionary.  Fields with conflicting semantics are
fully masked in any future confirmatory refit; values are never silently
converted or relabeled.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "patient-atlas-official-unit-reconciliation-v1"
DOCS_COMMIT = "4e15c6c2e51e0615ac941c299b574b084e7a2334"
CLINICAL_LAB_JSON_SHA256 = (
    "a36b367f28afca7aa0495ad8ce59186277902ca3e966df8aa2e67e82ba23bedf"
)
PHYSICAL_ASSESSMENT_SHA256 = (
    "1b5882c0b9e8210169acfb70077e0fcf1b403c648e000dd539367567d036c9bb"
)
CLINICAL_LAB_RELATIVE = Path(
    "versioned_docs/version-2.0.0/static/json/clinicalLabData.json"
)
PHYSICAL_ASSESSMENT_RELATIVE = Path(
    "versioned_docs/version-2.0.0/dataset/clinical-data/physical-assessment.mdx"
)

LAB_DOC_NAMES = {
    "a_g_ratio": "A/G Ratio (calculated field)",
    "albumin": "Albumin",
    "alkaline_phosphatase": "Alkaline Phosphatase",
    "alt_got": "ALT (GPT)",
    "ast_got": "AST (GOT)",
    "bilirubin_total": "Bilirubin, Total",
    "bun": "BUN",
    "buncreatinineratio": "BUN/Creatinine ratio",
    "c_peptide": "C-Peptide",
    "calcium": "Calcium",
    "carbon_dioxide_total": "Carbon Dioxide, Total",
    "chloride": "Chloride",
    "creatinine": "Creatinine",
    "crp_hs": "CRP-HS",
    "globulin_total": "Globulin, total (calculated field)",
    "glucose": "Glucose",
    "hba1c": "HbA1c",
    "hdl_cholesterol": "HDL Cholesterol",
    "insulin": "Insulin",
    "ldl_cholesterol": "LDL Cholesterol (calculated field)",
    "nt_probnp": "NT-proBNP",
    "potassium": "Potassium",
    "protein_total": "Protein, total",
    "sodium": "Sodium",
    "total_cholesterol": "Total Cholesterol",
    "triglycerides": "Triglycerides",
    "troponin_t": "Troponin-T",
    "urine_albumin": "Urine Albumin",
    "urine_creatinine": "Urine creatinine",
}

CBC_SOURCE_UNITS = {
    "hct": "%",
    "hemoglobin": "g/dL",
    "mch": "pg",
    "mchc": "g/dL",
    "mcv": "fL",
    "plt": "10^3/uL",
    "rbc": "10^6/uL",
    "rdw": "%",
    "wbc": "10^3/uL",
}

VITAL_UNITS = {
    "vit_bmi_vsorres": "kg/m^2",
    "vit_diabp_vsorres": "mmHg",
    "vit_height_vsorres": "cm",
    "vit_hip_vsorres": "cm",
    "vit_pulse_vsorres": "bpm",
    "vit_pulse_vsorres_2": "bpm",
    "vit_sysbp_vsorres": "mmHg",
    "vit_waist_vsorres": "cm",
    "vit_weight_vsorres": "kg",
    "vit_whr_vsorres": "1",
}

PHYSICAL_DOC_REQUIRED_TEXT = (
    "Height is recorded to the nearest tenth of a centimeter (cm).",
    "Weight is recorded to the nearest tenth of a kilogram (kg).",
    "BMI(metric) = weight in kilograms",
    "Waist Circumference (cm)",
    "Hip Circumference (cm)",
    "Heart Rate (bpm)",
    "Systolic (mmHg)",
    "Diastolic (mmHg)",
    "Waist to Hip Ratio (WHR)",
)

EXPECTED_CONFLICTS = frozenset(
    {"c_peptide", "insulin", "calcium", "urine_albumin", "urine_creatinine"}
)
DIMENSIONLESS_LABS = frozenset({"a_g_ratio", "buncreatinineratio"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{path.name} must contain a JSON object or array")
    return value


def _source_label_has_unit(name: str, source_values: Sequence[str]) -> bool:
    joined = " ".join(map(str, source_values))
    expected_fragments = {
        "hct": "%",
        "hemoglobin": "g/dL",
        "mch": "MCH - pg",
        "mchc": "MCHC - g/dL",
        "mcv": "MCV - fL",
        "plt": "x10E3/µL",
        "rbc": "x10E6/µL",
        "rdw": "RDW - %",
        "wbc": "x10E3/µL",
        "vit_diabp_vsorres": "Diastolic (mmHg)",
        "vit_height_vsorres": "Height (cm)",
        "vit_hip_vsorres": "Hip Circumference (cm)",
        "vit_pulse_vsorres": "Heart Rate (bpm)",
        "vit_pulse_vsorres_2": "Heart Rate (bpm)",
        "vit_sysbp_vsorres": "Systolic (mmHg)",
        "vit_waist_vsorres": "Waist Circumference (cm)",
        "vit_weight_vsorres": "Weight (kilograms)",
        "vit_bmi_vsorres": "BMI",
        "vit_whr_vsorres": "Waist to Hip Ratio (WHR)",
    }
    fragment = expected_fragments.get(name)
    return isinstance(fragment, str) and fragment in joined


def build_official_unit_reconciliation(
    *,
    feature_registry_path: str | Path,
    base_contract_path: str | Path,
    raw_audit_path: str | Path,
    docs_root: str | Path,
) -> dict[str, Any]:
    """Build the 48-field prospective unit policy from row-free sources."""

    feature_path = Path(feature_registry_path).resolve()
    base_path = Path(base_contract_path).resolve()
    audit_path = Path(raw_audit_path).resolve()
    docs_root = Path(docs_root).resolve()
    lab_path = docs_root / CLINICAL_LAB_RELATIVE
    physical_path = docs_root / PHYSICAL_ASSESSMENT_RELATIVE
    if _sha256(lab_path) != CLINICAL_LAB_JSON_SHA256:
        raise ValueError("official v2 clinical-lab dictionary hash differs")
    if _sha256(physical_path) != PHYSICAL_ASSESSMENT_SHA256:
        raise ValueError("official v2 physical-assessment documentation hash differs")

    registry = _load_json(feature_path)
    base = _load_json(base_path)
    audit = _load_json(audit_path)
    lab_rows = _load_json(lab_path)
    if not isinstance(registry, dict) or not isinstance(base, dict) or not isinstance(audit, dict):
        raise ValueError("registry, base contract, and audit must be objects")
    if not isinstance(lab_rows, list):
        raise ValueError("official clinical-lab dictionary must be an array")
    features = [
        feature
        for feature in registry.get("features", ())
        if feature.get("type") == "continuous"
    ]
    base_fields = base.get("fields")
    audit_fields = audit.get("measurements", {}).get("features")
    if (
        registry.get("schema_version") != "patient-atlas-feature-registry-v1"
        or base.get("schema_version") != "patient-atlas-clinical-field-contract-v1"
        or audit.get("schema_version") != "patient-atlas-raw-source-audit-v1"
        or len(features) != 48
        or not isinstance(base_fields, list)
        or not isinstance(audit_fields, list)
        or len(base_fields) != 48
        or len(audit_fields) != 48
    ):
        raise ValueError("unit reconciliation inputs have unsupported schemas")
    names = [str(feature["name"]) for feature in features]
    if (
        [field.get("name") for field in base_fields] != names
        or [field.get("name") for field in audit_fields] != names
    ):
        raise ValueError("row-free unit sources differ from fitted feature order")

    lab_units = {
        str(row.get("Name")): str(row.get("Units", "")).strip()
        for row in lab_rows
        if isinstance(row, dict) and row.get("Name") in set(LAB_DOC_NAMES.values())
    }
    if set(lab_units) != set(LAB_DOC_NAMES.values()):
        raise ValueError("official lab dictionary lacks a registered analyte")
    physical_text = physical_path.read_text()
    if any(fragment not in physical_text for fragment in PHYSICAL_DOC_REQUIRED_TEXT):
        raise ValueError("official physical-assessment unit definitions changed")

    records: list[dict[str, Any]] = []
    for feature, base_field, audit_field in zip(features, base_fields, audit_fields):
        name = str(feature["name"])
        release_units = tuple(
            base_field.get("source_unit", {}).get("observed_nonempty_values", ())
        )
        source_values = tuple(map(str, audit_field.get("source_values", ())))
        official_unit: str | None = None
        canonical_unit: str | None = None
        status: str
        row_policy: str

        if name in LAB_DOC_NAMES:
            official_unit = lab_units[LAB_DOC_NAMES[name]] or None
            if name in DIMENSIONLESS_LABS:
                canonical_unit = "1"
                status = "authorized_dimensionless_calculated_ratio"
                row_policy = "accept finite rows from the exact registered ratio source code"
            elif name == "creatinine":
                if release_units != ("mg/dL",) or official_unit != "md/dL":
                    raise ValueError("expected creatinine documentation typo evidence changed")
                canonical_unit = "mg/dL"
                status = "authorized_release_tag_documentation_typo_flagged"
                row_policy = "require unit_source_value exactly mg/dL; otherwise mask the row"
            elif name in EXPECTED_CONFLICTS:
                status = "blocked_unit_conflict"
                row_policy = "fully mask this field in confirmatory refits until adjudicated"
            else:
                if not official_unit:
                    raise ValueError(f"official unit is blank for non-ratio field {name}")
                if release_units and release_units != (official_unit,):
                    raise ValueError(f"unexpected release/documentation conflict for {name}")
                canonical_unit = official_unit
                status = (
                    "authorized_official_documentation_and_release_agree"
                    if release_units
                    else "authorized_official_versioned_documentation"
                )
                row_policy = "accept finite rows from the exact registered source code in the documented unit"
        elif name in CBC_SOURCE_UNITS:
            if not _source_label_has_unit(name, source_values):
                raise ValueError(f"CBC source label no longer authenticates {name}")
            canonical_unit = CBC_SOURCE_UNITS[name]
            status = "authorized_exact_release_source_label"
            row_policy = "accept finite rows only from the exact unit-bearing registered source code"
        elif name in VITAL_UNITS:
            if not _source_label_has_unit(name, source_values):
                raise ValueError(f"vital source label no longer authenticates {name}")
            canonical_unit = VITAL_UNITS[name]
            status = "authorized_official_physical_assessment_documentation"
            row_policy = "accept finite rows only from the exact registered physical-assessment source code"
        else:
            raise ValueError(f"no unit reconciliation route for {name}")

        if name in EXPECTED_CONFLICTS:
            if official_unit is None or not release_units or release_units == (official_unit,):
                raise ValueError(f"expected unit conflict disappeared for {name}")
            conflict = {
                "release_unit_values": list(release_units),
                "official_documentation_unit": official_unit,
            }
        else:
            conflict = None
        records.append(
            {
                "index": int(feature["index"]),
                "name": name,
                "block": str(feature["block"]),
                "canonical_unit": canonical_unit,
                "canonical_unit_authorized": canonical_unit is not None,
                "evidence_status": status,
                "confirmatory_row_policy": row_policy,
                "official_documentation_unit": official_unit,
                "release_unit_values": list(release_units),
                "release_source_label_unit_consistent": _source_label_has_unit(
                    name, source_values
                ),
                "unit_conflict": conflict,
                "plausibility_range": None,
                "plausibility_range_authorized": False,
            }
        )

    conflict_names = {
        record["name"]
        for record in records
        if record["evidence_status"] == "blocked_unit_conflict"
    }
    if conflict_names != EXPECTED_CONFLICTS:
        raise ValueError("unit-conflict set differs from the frozen reconciliation")
    authorized = sum(record["canonical_unit_authorized"] for record in records)
    status_counts = Counter(record["evidence_status"] for record in records)
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "prospective confirmatory unit policy; existing exploratory artifacts unchanged",
        "status_summary": {
            "continuous_field_count": 48,
            "canonical_unit_authorized_count": int(authorized),
            "fully_masked_unit_conflict_count": len(conflict_names),
            "fully_masked_unit_conflict_fields": sorted(conflict_names),
            "all_fields_have_fail_closed_unit_policy": True,
            "unit_metadata_gate_resolved_by_masking_conflicts": True,
            "confirmatory_plausibility_range_authorized_count": 0,
            "plausibility_policy_still_unresolved": True,
            "future_confirmatory_refit_required": True,
            "existing_exploratory_model_changed": False,
            "evidence_status_counts": dict(sorted(status_counts.items())),
        },
        "fields": records,
        "source_bindings": {
            "feature_registry_sha256": _sha256(feature_path),
            "base_clinical_field_contract_sha256": _sha256(base_path),
            "raw_source_audit_sha256": _sha256(audit_path),
            "official_docs_repository": "https://github.com/AI-READI/ai-readi-docs",
            "official_docs_commit": DOCS_COMMIT,
            "clinical_lab_json_relative_path": str(CLINICAL_LAB_RELATIVE),
            "clinical_lab_json_sha256": CLINICAL_LAB_JSON_SHA256,
            "physical_assessment_relative_path": str(PHYSICAL_ASSESSMENT_RELATIVE),
            "physical_assessment_sha256": PHYSICAL_ASSESSMENT_SHA256,
        },
        "limitations": [
            "five conflicting fields are excluded rather than relabeled or converted",
            "no physiological plausibility or clipping ranges are authorized",
            "the policy applies only to a new prospective refit; it does not retroactively change existing vectors",
        ],
        "privacy": {
            "patient_rows_emitted": False,
            "patient_identifiers_emitted": False,
            "raw_values_emitted": False,
            "small_cells_emitted": False,
        },
    }
    return report


def validate_official_unit_reconciliation(
    report: Mapping[str, Any],
    *,
    expected_feature_registry_sha256: str | None = None,
) -> None:
    fields = report.get("fields")
    summary = report.get("status_summary", {})
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected official unit-reconciliation schema")
    if not isinstance(fields, list) or len(fields) != 48:
        raise ValueError("official unit reconciliation must contain 48 fields")
    if [field.get("index") for field in fields] != list(range(48)):
        raise ValueError("official unit field indices must equal 0..47")
    conflict_names = {
        field.get("name")
        for field in fields
        if field.get("evidence_status") == "blocked_unit_conflict"
    }
    authorized = sum(
        field.get("canonical_unit_authorized") is True for field in fields
    )
    if (
        conflict_names != EXPECTED_CONFLICTS
        or summary.get("canonical_unit_authorized_count") != authorized
        or authorized != 43
        or summary.get("fully_masked_unit_conflict_count") != 5
        or summary.get("fully_masked_unit_conflict_fields")
        != sorted(EXPECTED_CONFLICTS)
        or summary.get("all_fields_have_fail_closed_unit_policy") is not True
        or summary.get("unit_metadata_gate_resolved_by_masking_conflicts") is not True
        or summary.get("confirmatory_plausibility_range_authorized_count") != 0
        or summary.get("plausibility_policy_still_unresolved") is not True
        or summary.get("future_confirmatory_refit_required") is not True
        or summary.get("existing_exploratory_model_changed") is not False
    ):
        raise ValueError("official unit-reconciliation summary is inconsistent")
    for field in fields:
        authorized_field = field.get("canonical_unit_authorized") is True
        if authorized_field != isinstance(field.get("canonical_unit"), str):
            raise ValueError("canonical-unit authorization is inconsistent")
        if field.get("plausibility_range") is not None or field.get(
            "plausibility_range_authorized"
        ) is not False:
            raise ValueError("unit reconciliation may not invent plausibility ranges")
        if field.get("name") in EXPECTED_CONFLICTS and (
            field.get("canonical_unit") is not None
            or field.get("unit_conflict") is None
            or not str(field.get("confirmatory_row_policy", "")).startswith(
                "fully mask"
            )
        ):
            raise ValueError("conflicting field is not fail-closed")
    bindings = report.get("source_bindings", {})
    if (
        bindings.get("official_docs_commit") != DOCS_COMMIT
        or bindings.get("clinical_lab_json_sha256")
        != CLINICAL_LAB_JSON_SHA256
        or bindings.get("physical_assessment_sha256")
        != PHYSICAL_ASSESSMENT_SHA256
    ):
        raise ValueError("official documentation bindings changed")
    if expected_feature_registry_sha256 is not None and bindings.get(
        "feature_registry_sha256"
    ) != expected_feature_registry_sha256:
        raise ValueError("unit reconciliation binds a different fitted registry")
    privacy = report.get("privacy", {})
    if any(
        privacy.get(key) is not False
        for key in (
            "patient_rows_emitted",
            "patient_identifiers_emitted",
            "raw_values_emitted",
            "small_cells_emitted",
        )
    ):
        raise ValueError("official unit-reconciliation privacy contract failed")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-registry", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--raw-audit", type=Path, required=True)
    parser.add_argument("--docs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError("official unit-reconciliation output must be new")
    report = build_official_unit_reconciliation(
        feature_registry_path=args.feature_registry,
        base_contract_path=args.base_contract,
        raw_audit_path=args.raw_audit,
        docs_root=args.docs_root,
    )
    validate_official_unit_reconciliation(
        report,
        expected_feature_registry_sha256=_sha256(args.feature_registry.resolve()),
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "event": "patient_atlas_official_unit_reconciliation_completed",
                "authorized_unit_count": report["status_summary"][
                    "canonical_unit_authorized_count"
                ],
                "masked_conflict_count": report["status_summary"][
                    "fully_masked_unit_conflict_count"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "build_official_unit_reconciliation",
    "validate_official_unit_reconciliation",
]
