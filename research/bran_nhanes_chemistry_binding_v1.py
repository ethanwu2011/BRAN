"""Header-only NHANES BIOPRO chemistry binding, never an XPORT row reader.

This module recognizes only the published BIOPRO_D/E source columns.  It does
not link SEQN, read observations, establish eligibility, or authorize training.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import re
from types import MappingProxyType

from bran_clinical_chemistry_semantics_v1 import CANONICAL_UNITS, CHEMISTRY_FIELDS
from bran_nhanes_header_extension_v1 import HeaderOnlyXportReader


NHANES_BIOPRO_CODEBOOK_URLS = MappingProxyType({
    "D": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/BIOPRO_D.htm",
    "E": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BIOPRO_E.htm",
})


@dataclass(frozen=True, repr=False)
class _SourceRule:
    column: str
    source_unit: str
    multiplier: float | None
    deferred_reason: str | None = None


# The D/E codebooks specify these same columns and units.  Factors are
# individual analyte rules, not a general mmol/L-to-mEq/L conversion rule.
_COMMON_RULES = MappingProxyType({
    "albumin": _SourceRule("LBXSAL", "g/dL", 1.0),
    "alt_got": _SourceRule("LBXSATSI", "U/L", None, "U_per_L_not_authorized_as_IU_per_L"),
    "ast_got": _SourceRule("LBXSASSI", "U/L", None, "U_per_L_not_authorized_as_IU_per_L"),
    "bilirubin_total": _SourceRule("LBXSTB", "mg/dL", 1.0),
    "bun": _SourceRule("LBXSBU", "mg/dL", 1.0),
    # The codebooks label this Bicarbonate (mmol/L), while the methodology
    # identifies total CO2. Bicarbonate is monovalent: this is field-specific.
    "carbon_dioxide_total": _SourceRule("LBXSC3SI", "mmol/L", 1.0),
    "chloride": _SourceRule("LBXSCLSI", "mmol/L", 1.0),
    "creatinine": _SourceRule("LBXSCR", "mg/dL", 1.0),
    "glucose": _SourceRule("LBXSGL", "mg/dL", 1.0),
    "potassium": _SourceRule("LBXSKSI", "mmol/L", 1.0),
    "protein_total": _SourceRule("LBXSTP", "g/dL", 1.0),
    "sodium": _SourceRule("LBXSNASI", "mmol/L", 1.0),
})
NHANES_BIOPRO_RULES = MappingProxyType({
    cycle: _COMMON_RULES for cycle in NHANES_BIOPRO_CODEBOOK_URLS
})
DEFERRED_FIELDS = MappingProxyType({
    field: rule.deferred_reason
    for field, rule in _COMMON_RULES.items()
    if rule.deferred_reason is not None
})


@dataclass(frozen=True, repr=False)
class NHANESChemistryObservation:
    value: float
    observed: bool
    provenance: str


def _valid_cycle(cycle: object) -> str:
    if not isinstance(cycle, str) or cycle not in NHANES_BIOPRO_RULES:
        raise ValueError("unsupported NHANES BIOPRO cycle")
    return cycle


def _binding_payload() -> bytes:
    payload = {
        "codebook_urls": dict(NHANES_BIOPRO_CODEBOOK_URLS),
        "rules": {
            cycle: {
                field: (rule.column, rule.source_unit, rule.multiplier, rule.deferred_reason)
                for field, rule in rules.items()
            }
            for cycle, rules in NHANES_BIOPRO_RULES.items()
        },
        "canonical_units": dict(CANONICAL_UNITS),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def codebook_binding_sha256() -> str:
    """Return a deterministic digest of closed codebook metadata, not an XPORT hash."""
    return hashlib.sha256(_binding_payload()).hexdigest()


def _header_columns(path: object) -> frozenset[str]:
    try:
        with HeaderOnlyXportReader(path) as reader:
            columns = tuple(reader.columns)
    except Exception:
        raise ValueError("NHANES BIOPRO header inspection failed") from None
    if (
        not columns
        or len(set(columns)) != len(columns)
        or any(
            not isinstance(column, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column)
            for column in columns
        )
    ):
        raise ValueError("NHANES BIOPRO header inspection failed")
    return frozenset(columns)


def inspect_nhanes_biopro_header(cycle: str, path: object) -> dict:
    """Return a row-free receipt containing only whitelist presence flags."""
    selected_cycle = _valid_cycle(cycle)
    columns = _header_columns(path)
    rules = NHANES_BIOPRO_RULES[selected_cycle]
    present = {field: rule.column in columns for field, rule in rules.items()}
    accepted = {
        field: bool(present[field] and rule.multiplier is not None)
        for field, rule in rules.items()
    }
    return {
        "schema": "bran-nhanes-chemistry-binding-v1",
        "cycle": selected_cycle,
        "codebook_url": NHANES_BIOPRO_CODEBOOK_URLS[selected_cycle],
        "codebook_binding_sha256": codebook_binding_sha256(),
        "whitelist_columns_present": present,
        "accepted_field_flags": accepted,
        "canonical_units": {field: CANONICAL_UNITS[field] for field in CHEMISTRY_FIELDS},
        "patient_rows_read": False,
        "training_ready": False,
    }


def convert_nhanes_biopro_value(
    cycle: str, field: str, source_column: object, value: object, source_unit: object,
) -> NHANESChemistryObservation:
    """Convert a typed source value only for an exact published BIOPRO rule.

    A false ``observed`` flag always carries NaN.  It never substitutes a value
    or uses a magnitude to infer a unit or analyte.
    """
    selected_cycle = _valid_cycle(cycle)
    if not isinstance(field, str) or field not in CHEMISTRY_FIELDS:
        raise ValueError("unsupported canonical chemistry field")
    rule = NHANES_BIOPRO_RULES[selected_cycle][field]
    if rule.multiplier is None:
        return NHANESChemistryObservation(math.nan, False, "deferred_source_assay_unit")
    if source_column != rule.column:
        return NHANESChemistryObservation(math.nan, False, "invalid_source_provenance")
    if source_unit != rule.source_unit:
        return NHANESChemistryObservation(math.nan, False, "unsupported_source_unit")
    if isinstance(value, bool) or not isinstance(value, Real):
        return NHANESChemistryObservation(math.nan, False, "invalid_numeric_value")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return NHANESChemistryObservation(math.nan, False, "invalid_numeric_value")
    converted = number * rule.multiplier
    if not math.isfinite(number) or number < 0.0 or not math.isfinite(converted):
        return NHANESChemistryObservation(math.nan, False, "invalid_numeric_value")
    return NHANESChemistryObservation(converted, True, "observed_nhanes_biopro")
