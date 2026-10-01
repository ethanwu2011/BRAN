"""Observed-only eICU CBC adapter with authoritative system-unit V3 conversion."""

from __future__ import annotations

import math
from collections.abc import Mapping

from bran_cbc_event_adapter_v1 import CanonicalCBCEvent, parse_numeric_measurement
from bran_cbc_mcl_units_v3 import canonicalize_cbc_v3
from bran_clinical_semantics_v1 import CBC_FIELDS, EICU_CBC_NAMES, relative_minutes
from bran_clinical_source_reader_v1 import EpisodeLinks, validate_event_link


EICU_CBC_COLUMNS = (
    "patientunitstayid", "labname", "labresult", "labtypeid",
    "labmeasurenamesystem", "labmeasurenameinterface", "labresultoffset",
)


def adapt_eicu_cbc_event_v2(row: object, links: EpisodeLinks) -> CanonicalCBCEvent | None:
    """Adapt one original hematology row; system unit is authoritative.

    The interface unit is used only to reject a recognized unequal canonical
    scale. An unknown interface never becomes a fallback and never invalidates
    an otherwise recognized system unit.
    """
    if not isinstance(row, Mapping) or set(row) != set(EICU_CBC_COLUMNS):
        raise ValueError("eICU CBC row is malformed")
    if not isinstance(links, EpisodeLinks):
        raise ValueError("links must be an EpisodeLinks instance")
    if row["labtypeid"] != "3":
        return None
    field = EICU_CBC_NAMES.get(row["labname"])
    if field is None:
        return None
    person = validate_event_link("eicu", row, links)
    if person is None:
        return None
    episode = row["patientunitstayid"]
    if not isinstance(episode, str) or episode != episode.strip() or not episode:
        return None
    value = parse_numeric_measurement(row["labresult"])
    offset = parse_numeric_measurement(row["labresultoffset"])
    if value is None or value <= 0.0 or offset is None:
        return None
    try:
        minutes = relative_minutes("eicu", offset=offset)
    except ValueError:
        return None
    if not math.isfinite(minutes):
        return None
    system_factor = canonicalize_cbc_v3(field, 1.0, row["labmeasurenamesystem"], provenance=1)
    if not system_factor.observed or not math.isfinite(system_factor.value):
        return None
    interface_factor = canonicalize_cbc_v3(field, 1.0, row["labmeasurenameinterface"], provenance=1)
    if interface_factor.observed and interface_factor.value != system_factor.value:
        return None
    observation = canonicalize_cbc_v3(field, value, row["labmeasurenamesystem"], provenance=1)
    if not observation.observed or not math.isfinite(observation.value) or observation.value <= 0.0:
        return None
    return CanonicalCBCEvent(
        "eicu", episode, person, CBC_FIELDS.index(field), float(minutes), float(observation.value)
    )
