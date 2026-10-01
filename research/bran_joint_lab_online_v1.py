"""Two-pass, bounded online selection for joint CBC/chemistry snapshots.

The first pass retains only an admission-day CBC anchor per seen episode.  The
second pass retains only the earliest eligible value/conflict state per field.
No source I/O, linkage, dictionary discovery, or serialization occurs here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from types import MappingProxyType
import unicodedata

import numpy as np

from bran_cbc_event_adapter_v1 import parse_numeric_measurement
from bran_cbc_chemistry_snapshot_v1 import (
    CBCchemistrySnapshot,
    CLINICAL_SNAPSHOT_FIELDS,
    CanonicalClinicalEvent,
)
from bran_clinical_semantics_v1 import CANONICAL_UNITS as CBC_UNITS, CBC_FIELDS, canonicalize_cbc
from bran_clinical_chemistry_semantics_v1 import (
    CANONICAL_UNITS as CHEMISTRY_UNITS,
    canonicalize_chemistry,
)


_FIELD_UNITS = {**CBC_UNITS, **CHEMISTRY_UNITS}
_FIELD_TO_INDEX = {field: index for index, field in enumerate(CLINICAL_SNAPSHOT_FIELDS)}
_CBC_COUNT = len(CBC_FIELDS)
_DAY_MINUTES = 1440.0
_SNAPSHOT_MINUTES = 60.0


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


def _source(value: object) -> str:
    if not _safe_text(value):
        raise ValueError("event source is invalid")
    return value


@dataclass(frozen=True, repr=False)
class AuthenticatedCodeBinding:
    """Immutable source-bound code map, validated once before repeated events."""

    source: str
    code_to_field: Mapping[str, str]

    def __post_init__(self) -> None:
        source = _source(self.source)
        if not isinstance(self.code_to_field, Mapping) or not self.code_to_field:
            raise ValueError("approved_code_to_field is invalid")
        copied: dict[str, str] = {}
        for code, field in self.code_to_field.items():
            if not _safe_text(code) or not isinstance(field, str) or field not in CLINICAL_SNAPSHOT_FIELDS:
                raise ValueError("approved_code_to_field is invalid")
            copied[code] = field
        if len(copied) != len(self.code_to_field):
            raise ValueError("approved_code_to_field is invalid")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "code_to_field", MappingProxyType(copied))


def authenticate_code_binding(source: str, approved_code_to_field: Mapping[str, str]) -> AuthenticatedCodeBinding:
    """Authenticate a plain mapping once for re-use by repeated conversions."""
    return AuthenticatedCodeBinding(source, approved_code_to_field)


def _binding(source: str, binding: AuthenticatedCodeBinding | Mapping[str, str]) -> AuthenticatedCodeBinding:
    if isinstance(binding, AuthenticatedCodeBinding):
        if binding.source != source:
            raise ValueError("binding source does not match event source")
        return binding
    # Preserve validation for the plain-dict API, while callers that retain an
    # AuthenticatedCodeBinding avoid a map-wide validation for every row.
    return authenticate_code_binding(source, binding)


def _identifiers(source: object, episode_key: object, person_key: object) -> tuple[str, str, str]:
    source = _source(source)
    if not _safe_text(episode_key) or not _safe_text(person_key):
        raise ValueError("event identifiers are invalid")
    return source, episode_key, person_key


def convert_joint_bound_event(
    source: str,
    episode_key: str,
    person_key: str,
    itemid: object,
    valuenum: object,
    valueuom: object,
    offset_minutes: object,
    provenance: object,
    binding: AuthenticatedCodeBinding | Mapping[str, str],
) -> CanonicalClinicalEvent | None:
    """Convert one original measurement using a source-bound authenticated map.

    Non-original provenance, unknown code/unit, and nonfinite value/time are
    returned as missing. Plain mappings are validated on every call; pass an
    ``AuthenticatedCodeBinding`` to validate once before a streaming loop.
    """
    source, episode_key, person_key = _identifiers(source, episode_key, person_key)
    authenticated = _binding(source, binding)
    if isinstance(provenance, bool) or not isinstance(provenance, Integral) or provenance != 1:
        return None
    if not _safe_text(itemid):
        return None
    field = authenticated.code_to_field.get(itemid)
    if field is None:
        return None
    value = parse_numeric_measurement(valuenum)
    if value is None or isinstance(offset_minutes, bool) or not isinstance(offset_minutes, Real):
        return None
    try:
        offset = float(offset_minutes)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(offset):
        return None
    if field in CBC_FIELDS:
        observation = canonicalize_cbc(field, value, valueuom, provenance=1)
        valid = observation.observed and observation.value > 0.0
    else:
        observation = canonicalize_chemistry(field, value, valueuom)
        valid = observation.observed
    if not valid or not math.isfinite(observation.value):
        return None
    return CanonicalClinicalEvent(
        source, episode_key, person_key, _FIELD_TO_INDEX[field], offset,
        float(observation.value), 1, _FIELD_UNITS[field],
    )


def _checked_event(event: object) -> CanonicalClinicalEvent:
    if not isinstance(event, CanonicalClinicalEvent):
        raise ValueError("event must be a CanonicalClinicalEvent")
    _identifiers(event.source, event.episode_key, event.person_key)
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


@dataclass(frozen=True, repr=False)
class EpisodeAnchors:
    source: str
    episode_to_person: Mapping[str, str]
    anchor_minutes: Mapping[str, float]

    def __post_init__(self) -> None:
        source = _source(self.source)
        if not isinstance(self.episode_to_person, Mapping) or not isinstance(self.anchor_minutes, Mapping):
            raise ValueError("episode anchors are malformed")
        if set(self.episode_to_person) != set(self.anchor_minutes):
            raise ValueError("episode anchors are malformed")
        people: dict[str, str] = {}
        anchors: dict[str, float] = {}
        for episode, person in self.episode_to_person.items():
            if not _safe_text(episode) or not _safe_text(person):
                raise ValueError("episode anchors are malformed")
            value = self.anchor_minutes[episode]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("episode anchors are malformed")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                raise ValueError("episode anchors are malformed") from None
            if not (math.isnan(number) or 0.0 <= number <= _DAY_MINUTES):
                raise ValueError("episode anchors are malformed")
            people[episode] = person
            anchors[episode] = number
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "episode_to_person", MappingProxyType(people))
        object.__setattr__(self, "anchor_minutes", MappingProxyType(anchors))


class CBCAnchorAccumulator:
    """First pass: preserve all seen groups and retain the first eligible CBC."""

    def __init__(self, source: str):
        self._source = _source(source)
        self._people: dict[str, str] = {}
        self._anchors: dict[str, float] = {}

    def add(self, event: CanonicalClinicalEvent) -> None:
        event = _checked_event(event)
        if event.source != self._source:
            raise ValueError("event source does not match accumulator source")
        prior_person = self._people.get(event.episode_key)
        if prior_person is not None and prior_person != event.person_key:
            raise ValueError("episode maps to conflicting persons")
        if prior_person is None:
            self._people[event.episode_key] = event.person_key
            self._anchors[event.episode_key] = math.nan
        if (
            event.field_index < _CBC_COUNT
            and 0.0 <= event.offset_minutes <= _DAY_MINUTES
            and (math.isnan(self._anchors[event.episode_key]) or event.offset_minutes < self._anchors[event.episode_key])
        ):
            self._anchors[event.episode_key] = float(event.offset_minutes)

    def finalize(self) -> EpisodeAnchors:
        return EpisodeAnchors(self._source, self._people, self._anchors)


@dataclass(repr=False)
class _SelectionState:
    field_offsets: np.ndarray
    field_values: np.ndarray
    field_conflicts: np.ndarray


def _state() -> _SelectionState:
    return _SelectionState(
        np.full(len(CLINICAL_SNAPSHOT_FIELDS), np.nan, dtype=float),
        np.full(len(CLINICAL_SNAPSHOT_FIELDS), np.nan, dtype=float),
        np.zeros(len(CLINICAL_SNAPSHOT_FIELDS), dtype=bool),
    )


def _readonly(values: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    copied = np.array(values, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


class JointLabSecondPass:
    """Second pass: retain at most 21 candidate states for each anchored group."""

    def __init__(self, anchors: EpisodeAnchors):
        if not isinstance(anchors, EpisodeAnchors):
            raise ValueError("anchors must be an EpisodeAnchors instance")
        self._anchors = anchors
        self._states = {episode: _state() for episode in anchors.episode_to_person}

    def add(self, event: CanonicalClinicalEvent) -> None:
        event = _checked_event(event)
        if event.source != self._anchors.source:
            raise ValueError("event source does not match accumulator source")
        expected_person = self._anchors.episode_to_person.get(event.episode_key)
        if expected_person is None:
            raise ValueError("event episode is not in finalized anchors")
        if expected_person != event.person_key:
            raise ValueError("episode maps to conflicting persons")
        anchor = self._anchors.anchor_minutes[event.episode_key]
        if math.isnan(anchor) or not 0.0 <= event.offset_minutes <= _DAY_MINUTES:
            return
        if not anchor <= event.offset_minutes <= min(anchor + _SNAPSHOT_MINUTES, _DAY_MINUTES):
            return
        state = self._states[event.episode_key]
        field = int(event.field_index)
        prior_offset = state.field_offsets[field]
        if math.isnan(prior_offset) or event.offset_minutes < prior_offset:
            state.field_offsets[field] = event.offset_minutes
            state.field_values[field] = event.canonical_value
            state.field_conflicts[field] = False
        elif event.offset_minutes == prior_offset and event.canonical_value != state.field_values[field]:
            state.field_conflicts[field] = True

    def iterate_snapshots(self):
        for episode in sorted(self._anchors.episode_to_person):
            state = self._states[episode]
            values = np.full(len(CLINICAL_SNAPSHOT_FIELDS), np.nan, dtype=float)
            observed = np.zeros(len(CLINICAL_SNAPSHOT_FIELDS), dtype=bool)
            conflicts = np.array(state.field_conflicts, dtype=bool, copy=True)
            offsets = np.full(len(CLINICAL_SNAPSHOT_FIELDS), np.nan, dtype=float)
            selected = ~conflicts & ~np.isnan(state.field_offsets)
            values[selected] = state.field_values[selected]
            observed[selected] = True
            offsets[selected] = state.field_offsets[selected]
            yield (
                episode,
                self._anchors.episode_to_person[episode],
                CBCchemistrySnapshot(
                    _readonly(values, float), _readonly(observed, bool), _readonly(conflicts, bool),
                    self._anchors.anchor_minutes[episode], _readonly(offsets, float),
                ),
            )
