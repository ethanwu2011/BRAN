"""Pure, observed-only joint CBC/chemistry snapshot selection.

This module has no source I/O, linkage, admission-time conversion, or source
dictionary lookup.  Callers must provide an authenticated, exact code mapping
and one admission-relative offset for each original measured event.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
import unicodedata

import numpy as np

from bran_cbc_event_adapter_v1 import parse_numeric_measurement
from bran_clinical_semantics_v1 import (
    CANONICAL_UNITS as CBC_CANONICAL_UNITS,
    CBC_FIELDS,
    canonicalize_cbc,
)
from bran_clinical_chemistry_semantics_v1 import (
    CANONICAL_UNITS as CHEMISTRY_CANONICAL_UNITS,
    CHEMISTRY_FIELDS,
    canonicalize_chemistry,
)


CLINICAL_SNAPSHOT_FIELDS = CBC_FIELDS + CHEMISTRY_FIELDS
_CBC_COUNT = len(CBC_FIELDS)
_FIELD_TO_INDEX = {field: index for index, field in enumerate(CLINICAL_SNAPSHOT_FIELDS)}
_FIELD_UNITS = {**CBC_CANONICAL_UNITS, **CHEMISTRY_CANONICAL_UNITS}
_DAY_MINUTES = 1440.0
_SNAPSHOT_MINUTES = 60.0


@dataclass(frozen=True, repr=False)
class CanonicalClinicalEvent:
    source: str
    episode_key: str
    person_key: str
    field_index: int
    offset_minutes: float
    canonical_value: float
    original_measured: int
    canonical_unit: str


@dataclass(frozen=True, repr=False)
class CBCchemistrySnapshot:
    values: np.ndarray
    observed: np.ndarray
    conflicts: np.ndarray
    anchor_minutes: float
    selected_offsets_minutes: np.ndarray


def _safe_text(value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if any(unicodedata.category(character) == "Cc" for character in value):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _validated_binding(mapping: object) -> Mapping[str, str]:
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("approved_code_to_field is invalid")
    for code, field in mapping.items():
        if not _safe_text(code) or not isinstance(field, str) or field not in CLINICAL_SNAPSHOT_FIELDS:
            raise ValueError("approved_code_to_field is invalid")
    return mapping


def _event_text(source: object, episode_key: object, person_key: object) -> tuple[str, str, str] | None:
    if not _safe_text(source):
        raise ValueError("event source is invalid")
    if not _safe_text(episode_key) or not _safe_text(person_key):
        raise ValueError("event identifiers are invalid")
    return source, episode_key, person_key


def convert_bound_event(
    source: str,
    episode_key: str,
    person_key: str,
    itemid: object,
    valuenum: object,
    valueuom: object,
    offset_minutes: object,
    provenance: object,
    approved_code_to_field: Mapping[str, str],
) -> CanonicalClinicalEvent | None:
    """Convert one original MIMIC-style measurement under a caller-bound code map.

    Unknown codes, non-original provenance, malformed values, units, and times
    become missing (``None``).  Invalid binding metadata and identifiers fail
    closed with static errors.  This function intentionally does not apply the
    admission-day window; selection owns that policy.
    """
    source, episode_key, person_key = _event_text(source, episode_key, person_key)
    mapping = _validated_binding(approved_code_to_field)
    if isinstance(provenance, bool) or not isinstance(provenance, Integral) or provenance != 1:
        return None
    if not _safe_text(itemid):
        return None
    field = mapping.get(itemid)
    if field is None:
        return None
    value = parse_numeric_measurement(valuenum)
    if value is None or not isinstance(offset_minutes, Real) or isinstance(offset_minutes, bool):
        return None
    try:
        offset = float(offset_minutes)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(offset):
        return None
    if field in CBC_FIELDS:
        observation = canonicalize_cbc(field, value, valueuom, provenance=1)
        # Preserve the frozen CBC event-layer zero exclusion.
        valid = observation.observed and observation.value > 0.0
    else:
        observation = canonicalize_chemistry(field, value, valueuom)
        valid = observation.observed
    if not valid or not math.isfinite(observation.value):
        return None
    return CanonicalClinicalEvent(
        source=source,
        episode_key=episode_key,
        person_key=person_key,
        field_index=_FIELD_TO_INDEX[field],
        offset_minutes=offset,
        canonical_value=float(observation.value),
        original_measured=1,
        canonical_unit=_FIELD_UNITS[field],
    )


def _readonly(values: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    copied = np.array(values, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


def _validated_event(event: object) -> CanonicalClinicalEvent:
    if not isinstance(event, CanonicalClinicalEvent):
        raise ValueError("event must be a CanonicalClinicalEvent")
    _event_text(event.source, event.episode_key, event.person_key)
    if (
        isinstance(event.field_index, bool)
        or not isinstance(event.field_index, Integral)
        or not 0 <= event.field_index < len(CLINICAL_SNAPSHOT_FIELDS)
        or isinstance(event.offset_minutes, bool)
        or not isinstance(event.offset_minutes, Real)
        or isinstance(event.canonical_value, bool)
        or not isinstance(event.canonical_value, Real)
        or isinstance(event.original_measured, bool)
        or not isinstance(event.original_measured, Integral)
        or event.original_measured != 1
        or event.canonical_unit != _FIELD_UNITS[CLINICAL_SNAPSHOT_FIELDS[int(event.field_index)]]
    ):
        raise ValueError("clinical event is malformed")
    try:
        offset, value = float(event.offset_minutes), float(event.canonical_value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("clinical event is malformed") from None
    if not math.isfinite(offset) or not math.isfinite(value) or value < 0.0:
        raise ValueError("clinical event is malformed")
    if event.field_index < _CBC_COUNT and value == 0.0:
        raise ValueError("clinical event is malformed")
    return event


def select_cbc_chemistry_snapshot(events: Iterable[CanonicalClinicalEvent]) -> CBCchemistrySnapshot:
    """Select one per-episode 21-field snapshot anchored by the first valid CBC.

    Only finite events in [0, 1440] are eligible.  Chemistry never anchors a
    panel.  Each field chooses its earliest value in the first-CBC hour; ties
    of equal values collapse while conflicting earliest ties remain missing.
    """
    try:
        checked = tuple(_validated_event(event) for event in events)
    except TypeError:
        raise ValueError("events must be an iterable") from None
    if checked:
        first = checked[0]
        if any(
            event.source != first.source
            or event.episode_key != first.episode_key
            or event.person_key != first.person_key
            for event in checked[1:]
        ):
            raise ValueError("events must describe one episode")

    values = np.full(len(CLINICAL_SNAPSHOT_FIELDS), np.nan, dtype=float)
    observed = np.zeros(len(CLINICAL_SNAPSHOT_FIELDS), dtype=bool)
    conflicts = np.zeros(len(CLINICAL_SNAPSHOT_FIELDS), dtype=bool)
    selected_offsets = np.full(len(CLINICAL_SNAPSHOT_FIELDS), np.nan, dtype=float)

    in_day = tuple(event for event in checked if 0.0 <= event.offset_minutes <= _DAY_MINUTES)
    cbc_offsets = tuple(event.offset_minutes for event in in_day if event.field_index < _CBC_COUNT)
    if not cbc_offsets:
        return CBCchemistrySnapshot(
            _readonly(values, float), _readonly(observed, bool), _readonly(conflicts, bool),
            float("nan"), _readonly(selected_offsets, float),
        )

    anchor = float(min(cbc_offsets))
    cutoff = min(anchor + _SNAPSHOT_MINUTES, _DAY_MINUTES)
    for field in range(len(CLINICAL_SNAPSHOT_FIELDS)):
        candidates = tuple(
            event for event in in_day
            if event.field_index == field and anchor <= event.offset_minutes <= cutoff
        )
        if not candidates:
            continue
        earliest = min(event.offset_minutes for event in candidates)
        tied_values = tuple(event.canonical_value for event in candidates if event.offset_minutes == earliest)
        if any(value != tied_values[0] for value in tied_values[1:]):
            conflicts[field] = True
            continue
        values[field] = tied_values[0]
        observed[field] = True
        selected_offsets[field] = earliest

    return CBCchemistrySnapshot(
        _readonly(values, float), _readonly(observed, bool), _readonly(conflicts, bool),
        anchor, _readonly(selected_offsets, float),
    )
