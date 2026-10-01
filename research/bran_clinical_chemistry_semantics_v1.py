"""Closed chemistry semantics and row-free MIMIC dictionary binding helpers.

This is a draft binding layer, not source admission or a training interface.
Unknown units and unmatched dictionary rows remain unapproved.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import math
from numbers import Real
import re
from types import MappingProxyType


CHEMISTRY_FIELDS = (
    "albumin", "alt_got", "ast_got", "bilirubin_total", "bun",
    "carbon_dioxide_total", "chloride", "creatinine", "glucose",
    "potassium", "protein_total", "sodium",
)
CANONICAL_UNITS = MappingProxyType({
    "albumin": "g/dL", "alt_got": "IU/L", "ast_got": "IU/L",
    "bilirubin_total": "mg/dL", "bun": "mg/dL", "carbon_dioxide_total": "mEq/L",
    "chloride": "mEq/L", "creatinine": "mg/dL", "glucose": "mg/dL",
    "potassium": "mEq/L", "protein_total": "g/dL", "sodium": "mEq/L",
})

# Exact MIMIC-IV d_labitems identity plus item IDs from MIT-LCP measurement
# concepts. The local dictionary must independently affirm every component.
MIMIC_CHEMISTRY_RULES = MappingProxyType({
    "50862": ("albumin", "Albumin"),
    "50861": ("alt_got", "Alanine Aminotransferase (ALT)"),
    "50878": ("ast_got", "Asparate Aminotransferase (AST)"),
    "50885": ("bilirubin_total", "Bilirubin, Total"),
    "51006": ("bun", "Urea Nitrogen"),
    "50882": ("carbon_dioxide_total", "Bicarbonate"),
    "50902": ("chloride", "Chloride"),
    "50912": ("creatinine", "Creatinine"),
    "50931": ("glucose", "Glucose"),
    "50971": ("potassium", "Potassium"),
    "50976": ("protein_total", "Protein, Total"),
    "50983": ("sodium", "Sodium"),
})
MIMIC_DICTIONARY_COLUMNS = ("itemid", "label", "fluid", "category")
MIMIC_PRIMARY_SOURCES = MappingProxyType({
    "chemistry": "https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/chemistry.sql",
    "enzyme": "https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/enzyme.sql",
})


@dataclass(frozen=True, repr=False)
class ChemistryObservation:
    value: float
    observed: bool
    status: str


def _unit_token(unit: object) -> str | None:
    if not isinstance(unit, str):
        return None
    return re.sub(r"\s+", "", unit.strip().replace("µ", "u").replace("μ", "u")).casefold()


# No cross-dimension/molar-mass conversion is authorized in this draft.
_UNIT_FACTORS = MappingProxyType({field: MappingProxyType({_unit_token(unit): 1.0}) for field, unit in CANONICAL_UNITS.items()})


def canonicalize_chemistry(field: str, value: object, unit: object) -> ChemistryObservation:
    """Convert only an exact approved original unit; never guess from magnitude."""
    if not isinstance(field, str) or field not in CANONICAL_UNITS:
        raise ValueError("unsupported canonical chemistry field")
    if isinstance(value, bool) or not isinstance(value, Real):
        return ChemistryObservation(math.nan, False, "invalid_numeric_value")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return ChemistryObservation(math.nan, False, "invalid_numeric_value")
    if not math.isfinite(number) or number < 0.0:
        return ChemistryObservation(math.nan, False, "invalid_numeric_value")
    factor = _UNIT_FACTORS[field].get(_unit_token(unit))
    if factor is None:
        return ChemistryObservation(math.nan, False, "unsupported_unit")
    converted = number * factor
    if not math.isfinite(converted):
        return ChemistryObservation(math.nan, False, "invalid_numeric_value")
    return ChemistryObservation(converted, True, "observed_converted")


def _code(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdigit()
        or not value
        or value[0] == "0"
    ):
        raise ValueError("invalid MIMIC dictionary itemid")
    return value


def bind_mimic_chemistry_dictionary(rows: Iterable[Mapping[str, object]]) -> dict:
    """Bind only exact code/name/Blood/Chemistry dictionary rows.

    The result intentionally stores codes, canonical fields, fixed multipliers,
    and aggregate reasons only—never arbitrary local dictionary text.
    """
    fields = {
        field: {"candidate_codes": [], "dictionary_bound_codes": [], "canonical_unit": CANONICAL_UNITS[field]}
        for field in CHEMISTRY_FIELDS
    }
    nonmatches = {"not_exact_blood_chemistry_identity": 0, "unrecognized_itemid": 0}
    approved: dict[str, str] = {}
    conversions: dict[str, dict] = {}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or any(column not in row for column in MIMIC_DICTIONARY_COLUMNS):
            raise ValueError("MIMIC dictionary row is malformed")
        itemid = _code(row["itemid"])
        if itemid in seen:
            raise ValueError("repeated MIMIC dictionary itemid")
        seen.add(itemid)
        rule = MIMIC_CHEMISTRY_RULES.get(itemid)
        if rule is None:
            nonmatches["unrecognized_itemid"] += 1
            continue
        field, expected_label = rule
        if row["label"] != expected_label or row["fluid"] != "Blood" or row["category"] != "Chemistry":
            nonmatches["not_exact_blood_chemistry_identity"] += 1
            continue
        fields[field]["candidate_codes"].append(itemid)
        fields[field]["dictionary_bound_codes"].append(itemid)
        approved[itemid] = field
        conversions[itemid] = {"canonical_unit": CANONICAL_UNITS[field], "multiplier": 1.0}
    for record in fields.values():
        record["candidate_codes"].sort(key=int)
        record["dictionary_bound_codes"].sort(key=int)
    result = {
        "fields": fields, "approved_code_to_field": approved, "approved_unit_conversions": conversions,
        "metadata_nonmatches": nonmatches, "training_ready": False, "patient_rows_read": False,
    }
    validate_binding_result(result)
    return result


def validate_binding_result(result: object) -> None:
    expected = {"fields", "approved_code_to_field", "approved_unit_conversions", "metadata_nonmatches", "training_ready", "patient_rows_read"}
    if not isinstance(result, Mapping) or set(result) != expected or result["training_ready"] is not False or result["patient_rows_read"] is not False:
        raise ValueError("chemistry binding result is malformed")
    fields, approved, conversions = result["fields"], result["approved_code_to_field"], result["approved_unit_conversions"]
    if not isinstance(fields, Mapping) or tuple(fields) != CHEMISTRY_FIELDS or not isinstance(approved, Mapping) or not isinstance(conversions, Mapping):
        raise ValueError("chemistry binding result is malformed")
    expected_approved = {}
    for field in CHEMISTRY_FIELDS:
        record = fields[field]
        if not isinstance(record, Mapping) or set(record) != {"candidate_codes", "dictionary_bound_codes", "canonical_unit"} or record["canonical_unit"] != CANONICAL_UNITS[field]:
            raise ValueError("chemistry binding result is malformed")
        candidates, bound = record["candidate_codes"], record["dictionary_bound_codes"]
        if not isinstance(candidates, list) or not isinstance(bound, list) or candidates != sorted(set(candidates), key=int) or bound != sorted(set(bound), key=int) or candidates != bound:
            raise ValueError("chemistry binding result is malformed")
        if any(
            not isinstance(code, str)
            or not code.isascii()
            or not code.isdigit()
            or code not in MIMIC_CHEMISTRY_RULES
            or MIMIC_CHEMISTRY_RULES[code][0] != field
            for code in bound
        ):
            raise ValueError("chemistry binding result is malformed")
        expected_approved.update({code: field for code in bound})
    if dict(approved) != expected_approved or set(conversions) != set(approved):
        raise ValueError("chemistry binding result is malformed")
    for code, field in approved.items():
        conversion = conversions[code]
        if not isinstance(conversion, Mapping) or conversion != {"canonical_unit": CANONICAL_UNITS[field], "multiplier": 1.0}:
            raise ValueError("chemistry binding result is malformed")
    nonmatches = result["metadata_nonmatches"]
    if (
        not isinstance(nonmatches, Mapping)
        or set(nonmatches) != {"not_exact_blood_chemistry_identity", "unrecognized_itemid"}
        or any(type(value) is not int or value < 0 for value in nonmatches.values())
    ):
        raise ValueError("chemistry binding result is malformed")


def dictionary_sha256(path) -> str:
    """Hash a non-patient dictionary file for an external receipt."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
