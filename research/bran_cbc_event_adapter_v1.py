"""Local adapters from approved raw CBC rows to bounded episode snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Integral, Real
import re
import unicodedata

import numpy as np

from bran_clinical_semantics_v1 import CBC_FIELDS, canonicalize_cbc, relative_minutes
from bran_clinical_snapshot_v1 import CBCSnapshot
from bran_clinical_source_reader_v1 import EpisodeLinks, validate_event_link


_SOURCES = frozenset({"mimic", "nwicu", "eicu", "sicdb"})
_NUMERIC_CSV = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
_DAY_MINUTES = 1440.0
_WINDOW_MINUTES = 60.0


@dataclass(frozen=True, repr=False)
class CanonicalCBCEvent:
    source: str
    episode_key: str
    person_key: str
    field_index: int
    offset_minutes: float
    canonical_value: float


def parse_numeric_measurement(value: object) -> float | None:
    """Parse a finite decimal/scientific CSV string without inequality coercion."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Real):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None
    if not isinstance(value, str) or _NUMERIC_CSV.fullmatch(value) is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _source(source: object) -> str:
    if not isinstance(source, str) or source not in _SOURCES:
        raise ValueError("source is unsupported for CBC events")
    return source


def _approved_code_map(mapping: object) -> Mapping[str, str]:
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("approved_code_to_field must be a nonempty mapping")
    for code, field in mapping.items():
        if not isinstance(code, str) or not code or not isinstance(field, str) or field not in CBC_FIELDS:
            raise ValueError("approved_code_to_field is invalid")
    return mapping


def _approved_unit_map(mapping: object) -> Mapping[str, str]:
    if not isinstance(mapping, Mapping):
        raise ValueError("approved_units is required for sicdb")
    for code, unit in mapping.items():
        if not isinstance(code, str) or not code or not isinstance(unit, str) or not unit:
            raise ValueError("approved_units is invalid")
    return mapping


def _row_value(row: object, key: str) -> object | None:
    if not isinstance(row, Mapping):
        raise ValueError("row must be a mapping")
    return row.get(key)


def _numeric_key(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return str(value) if value >= 0 else None
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        return None
    return value.lstrip("0") or "0"


def _opaque_key(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or any(unicodedata.category(character) == "Cc" for character in normalized):
        return None
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError:
        return None
    return normalized


def _naive_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or value != value.strip() or ("T" not in value and " " not in value):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed.tzinfo is None else None


def _episode_key(source: str, row: Mapping[str, object]) -> str | None:
    if source in {"mimic", "nwicu"}:
        return _numeric_key(row.get("hadm_id"))
    if source == "eicu":
        return _opaque_key(row.get("patientunitstayid"))
    return _numeric_key(row.get("CaseID"))


def _source_columns(source: str) -> tuple[str, str, str, str]:
    if source in {"mimic", "nwicu"}:
        return "itemid", "valuenum", "valueuom", "charttime"
    if source == "eicu":
        return "labname", "labresult", "labmeasurenamesystem", "labresultoffset"
    return "LaboratoryID", "LaboratoryValue", "", "Offset"


def adapt_cbc_event(
    source: str,
    row: Mapping[str, object],
    links: EpisodeLinks,
    approved_code_to_field: Mapping[str, str],
    admission_times: Mapping[str, object] | None = None,
    approved_units: Mapping[str, str] | None = None,
) -> CanonicalCBCEvent | None:
    """Adapt one approved raw source row; unknown or invalid observations are missing.

    The caller remains responsible for ensuring that rows are original numeric
    source observations rather than imputed or derived records.
    """
    source = _source(source)
    mapping = _approved_code_map(approved_code_to_field)
    units = None
    if source == "sicdb":
        units = _approved_unit_map(approved_units)
        if any(code not in units for code in mapping):
            raise ValueError("SICdb approved units are incomplete")
    if not isinstance(row, Mapping):
        raise ValueError("row must be a mapping")
    if not isinstance(links, EpisodeLinks):
        raise ValueError("links must be an EpisodeLinks instance")
    # This also enforces source binding before a row is admitted.
    person = validate_event_link(source, row, links)
    if person is None:
        return None
    episode = _episode_key(source, row)
    if episode is None:
        return None

    code_column, value_column, unit_column, time_column = _source_columns(source)
    code = _row_value(row, code_column)
    if not isinstance(code, str):
        return None
    field = mapping.get(code)
    if field is None:
        return None
    value = parse_numeric_measurement(_row_value(row, value_column))
    if value is None:
        return None

    if source in {"mimic", "nwicu"}:
        if admission_times is None:
            return None
        if not isinstance(admission_times, Mapping):
            raise ValueError("admission_times must be a mapping")
        lab_time = _naive_iso_datetime(_row_value(row, time_column))
        admission_time = _naive_iso_datetime(admission_times.get(episode))
        if lab_time is None or admission_time is None:
            return None
        try:
            offset_minutes = relative_minutes(
                source, lab_time=lab_time, admission_time=admission_time
            )
        except ValueError:
            return None
        unit = _row_value(row, unit_column)
    elif source == "eicu":
        offset = parse_numeric_measurement(_row_value(row, time_column))
        if offset is None:
            return None
        try:
            offset_minutes = relative_minutes(source, offset=offset)
        except ValueError:
            return None
        unit = _row_value(row, unit_column)
    else:
        assert units is not None
        unit = units.get(code)
        if unit is None:  # Defensive: completeness was checked before row linkage.
            raise ValueError("SICdb approved units are incomplete")
        offset = parse_numeric_measurement(_row_value(row, time_column))
        if offset is None:
            return None
        try:
            offset_minutes = relative_minutes(source, offset=offset)
        except ValueError:
            return None

    observation = canonicalize_cbc(field, value, unit, provenance=1)
    if (
        not observation.observed
        or not math.isfinite(observation.value)
        or observation.value <= 0.0
        or not math.isfinite(offset_minutes)
    ):
        return None
    return CanonicalCBCEvent(
        source=source,
        episode_key=episode,
        person_key=person,
        field_index=CBC_FIELDS.index(field),
        offset_minutes=float(offset_minutes),
        canonical_value=float(observation.value),
    )


@dataclass(repr=False)
class _EpisodeState:
    person_key: str
    anchor_minutes: float
    field_offsets: np.ndarray
    field_values: np.ndarray
    field_conflicts: np.ndarray


def _readonly(values: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    copied = np.array(values, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


class OnlineCBCPanels:
    """O(episodes * 9) online accumulator for admission-day CBC snapshots."""

    def __init__(self, source: str):
        self._source = _source(source)
        self._episodes: dict[str, _EpisodeState] = {}

    def add(self, event: CanonicalCBCEvent) -> None:
        if not isinstance(event, CanonicalCBCEvent):
            raise ValueError("event must be a CanonicalCBCEvent")
        if event.source != self._source:
            raise ValueError("event source does not match panel source")
        if (
            not isinstance(event.episode_key, str)
            or not event.episode_key
            or not isinstance(event.person_key, str)
            or not event.person_key
            or isinstance(event.field_index, bool)
            or not isinstance(event.field_index, Integral)
            or not 0 <= event.field_index < len(CBC_FIELDS)
            or not isinstance(event.offset_minutes, (int, float))
            or isinstance(event.offset_minutes, bool)
            or not isinstance(event.canonical_value, (int, float))
            or isinstance(event.canonical_value, bool)
        ):
            raise ValueError("CBC event is malformed")
        offset = float(event.offset_minutes)
        value = float(event.canonical_value)
        if not math.isfinite(offset) or not math.isfinite(value) or value <= 0.0:
            raise ValueError("CBC event is malformed")
        state = self._episodes.get(event.episode_key)
        if state is not None and state.person_key != event.person_key:
            raise ValueError("episode maps to conflicting persons")
        if not 0.0 <= offset <= _DAY_MINUTES:
            return
        if state is None:
            state = _EpisodeState(
                event.person_key,
                math.inf,
                np.full(len(CBC_FIELDS), np.nan, dtype=float),
                np.full(len(CBC_FIELDS), np.nan, dtype=float),
                np.zeros(len(CBC_FIELDS), dtype=bool),
            )
            self._episodes[event.episode_key] = state
        if offset < state.anchor_minutes:
            state.anchor_minutes = offset
        field = int(event.field_index)
        prior_offset = state.field_offsets[field]
        if math.isnan(prior_offset) or offset < prior_offset:
            state.field_offsets[field] = offset
            state.field_values[field] = value
            state.field_conflicts[field] = False
        elif offset == prior_offset and value != state.field_values[field]:
            state.field_conflicts[field] = True

    def iterate_snapshots(self):
        """Yield immutable ``(episode_key, person_key, CBCSnapshot)`` tuples."""
        for episode_key in sorted(self._episodes):
            state = self._episodes[episode_key]
            values = np.full(len(CBC_FIELDS), np.nan, dtype=float)
            observed = np.zeros(len(CBC_FIELDS), dtype=bool)
            conflicts = np.zeros(len(CBC_FIELDS), dtype=bool)
            selected_offsets = np.full(len(CBC_FIELDS), np.nan, dtype=float)
            if math.isfinite(state.anchor_minutes):
                cutoff = min(state.anchor_minutes + _WINDOW_MINUTES, _DAY_MINUTES)
                selected = state.field_offsets <= cutoff
                for field in np.flatnonzero(selected):
                    if state.field_conflicts[field]:
                        conflicts[field] = True
                    else:
                        values[field] = state.field_values[field]
                        observed[field] = True
                        selected_offsets[field] = state.field_offsets[field]
            yield (
                episode_key,
                state.person_key,
                CBCSnapshot(
                    _readonly(values, float),
                    _readonly(observed, bool),
                    _readonly(conflicts, bool),
                    float(state.anchor_minutes) if math.isfinite(state.anchor_minutes) else float("nan"),
                    _readonly(selected_offsets, float),
                ),
            )
