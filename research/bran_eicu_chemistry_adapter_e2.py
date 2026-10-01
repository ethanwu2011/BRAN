"""Pure observed-only eICU chemistry projection for the E2 admission path."""

from __future__ import annotations

from collections.abc import Mapping
import math
from types import MappingProxyType

from bran_cbc_event_adapter_v1 import CanonicalCBCEvent, parse_numeric_measurement
from bran_clinical_chemistry_semantics_v1 import CHEMISTRY_FIELDS, canonicalize_chemistry
from bran_clinical_source_reader_v1 import EpisodeLinks, validate_event_link
from bran_eicu_cbc_adapter_v2 import CBC_FIELDS, relative_minutes
import bran_eicu_discovery_bridge_v1 as bridge


ERROR = "eicu_chemistry_adapter_e2_contract_failed"
LANDMARK = bridge.LANDMARK
FIELDS = CBC_FIELDS + CHEMISTRY_FIELDS

# These are the exact public eICU labName labels admitted by the mapping
# proposal.  Bedside glucose is intentionally not an alias for glucose, and
# protein_total remains unbound until a source-specific label is approved.
CHEMISTRY_NAMES = MappingProxyType({
    "albumin": "albumin",
    "ALT (SGPT)": "alt_got",
    "AST (SGOT)": "ast_got",
    "total bilirubin": "bilirubin_total",
    "BUN": "bun",
    "Total CO2": "carbon_dioxide_total",
    "chloride": "chloride",
    "creatinine": "creatinine",
    "glucose": "glucose",
    "potassium": "potassium",
    "sodium": "sodium",
})


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError(ERROR)


def _interface_is_noncontradictory(
    field: str,
    system_unit: object,
    interface_unit: object,
) -> bool:
    """Accept an unknown interface unit but never use it to rescue the system."""
    system = canonicalize_chemistry(field, 1.0, system_unit)
    if not system.observed or not math.isfinite(system.value):
        return False
    interface = canonicalize_chemistry(field, 1.0, interface_unit)
    if not interface.observed:
        return True
    return math.isfinite(interface.value) and interface.value == system.value


def observed_chemistry_before_landmark(
    row: object,
    links: EpisodeLinks,
) -> CanonicalCBCEvent | None:
    """Project one admitted chemistry row, or return None when it is unobserved."""
    try:
        _require(isinstance(row, Mapping) and set(row) == set(bridge.LAB_COLUMNS))
        _require(isinstance(links, EpisodeLinks))

        if row["labtypeid"] != "1":
            return None
        field = CHEMISTRY_NAMES.get(row["labname"])
        if field is None:
            return None

        specimen = parse_numeric_measurement(row["labresultoffset"])
        revision = parse_numeric_measurement(row["labresultrevisedoffset"])
        if (
            specimen is None
            or revision is None
            or not 0.0 <= specimen <= LANDMARK
            or not specimen <= revision <= LANDMARK
        ):
            return None

        person = validate_event_link("eicu", row, links)
        if person is None:
            return None
        episode = row["patientunitstayid"]
        if not isinstance(episode, str) or not episode or episode != episode.strip():
            return None

        if not _interface_is_noncontradictory(
            field,
            row["labmeasurenamesystem"],
            row["labmeasurenameinterface"],
        ):
            return None
        raw_value = parse_numeric_measurement(row["labresult"])
        if raw_value is None:
            return None
        observation = canonicalize_chemistry(
            field,
            raw_value,
            row["labmeasurenamesystem"],
        )
        if not observation.observed or not math.isfinite(observation.value):
            return None

        minutes = relative_minutes("eicu", offset=specimen)
        if not math.isfinite(minutes):
            return None
        return CanonicalCBCEvent(
            "eicu",
            episode,
            person,
            len(CBC_FIELDS) + CHEMISTRY_FIELDS.index(field),
            float(minutes),
            float(observation.value),
        )
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc) == ERROR:
            raise
        raise ValueError(ERROR) from None
