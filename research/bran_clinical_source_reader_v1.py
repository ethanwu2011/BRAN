"""Bounded local CSV projection and source-local episode linkage helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import csv
import gzip
import os
import re
from numbers import Integral
from types import MappingProxyType
import unicodedata


_HEADER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FIELD_SIZE_LIMIT = 1_000_000
_SUPPORTED_LINK_SOURCES = frozenset({"mimic", "nwicu", "eicu", "sicdb"})


@dataclass(frozen=True, repr=False)
class EpisodeLinks:
    """Immutable source-local episode mappings.

    ``episode_to_care_group`` is populated only when the source exposes a
    care-group identifier (currently eICU); it is otherwise ``None``.
    """

    source: str
    episode_to_person: Mapping[str, str]
    episode_to_care_group: Mapping[str, str] | None


def _validate_projection_columns(columns: object) -> tuple[str, ...]:
    if not isinstance(columns, tuple) or not columns:
        raise ValueError("columns must be a nonempty tuple of CSV identifiers")
    if any(not isinstance(column, str) or not _HEADER_PATTERN.fullmatch(column) for column in columns):
        raise ValueError("columns must be a nonempty tuple of CSV identifiers")
    if len(set(columns)) != len(columns):
        raise ValueError("columns must be a nonempty tuple of CSV identifiers")
    return columns


def _csv_operation_with_field_limit(operation):
    previous_limit = csv.field_size_limit(_FIELD_SIZE_LIMIT)
    try:
        return operation()
    finally:
        csv.field_size_limit(previous_limit)


def iter_projected_csv(path: str | os.PathLike[str], columns: tuple[str, ...], *, max_rows: int):
    """Yield a bounded projection from a local plain or gzip CSV file.

    The generator checks one record beyond ``max_rows`` and fails rather than
    silently treating a prefix as the complete input.
    """
    requested = _validate_projection_columns(columns)
    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")
    try:
        file_path = os.fspath(path)
    except TypeError:
        raise ValueError("path must be a filesystem path") from None
    if not isinstance(file_path, str):
        raise ValueError("path must be a filesystem path")

    opener = gzip.open if file_path.endswith(".gz") else open
    try:
        with opener(file_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, restkey="__bran_extra_cells__", restval=None, strict=True)
            headers = _csv_operation_with_field_limit(lambda: reader.fieldnames)
            if not headers:
                raise ValueError("CSV input is malformed")
            headers = list(headers)
            if headers[0].startswith("\ufeff"):
                headers[0] = headers[0][1:]
                reader.fieldnames = headers
            if (
                any(not isinstance(header, str) or not _HEADER_PATTERN.fullmatch(header) for header in headers)
                or len(set(headers)) != len(headers)
                or "__bran_extra_cells__" in headers
                or any(column not in headers for column in requested)
            ):
                raise ValueError("CSV input is malformed")
            row_count = 0
            while True:
                try:
                    row = _csv_operation_with_field_limit(lambda: next(reader))
                except StopIteration:
                    break
                row_count += 1
                if row_count > max_rows:
                    raise ValueError("CSV bounded-read limit exceeded")
                if row.get("__bran_extra_cells__") is not None or any(
                    row.get(column) is None for column in headers
                ):
                    raise ValueError("CSV input is malformed")
                yield {column: row[column] for column in requested}
    except (OSError, UnicodeError, EOFError):
        raise ValueError("CSV input could not be read") from None
    except csv.Error:
        raise ValueError("CSV input is malformed") from None


def _require_source(source: object) -> str:
    if not isinstance(source, str) or source not in _SUPPORTED_LINK_SOURCES:
        raise ValueError("source is unsupported for episode linkage")
    return source


def _iterable_rows(rows: object):
    if isinstance(rows, (str, bytes, Mapping)):
        raise ValueError("linkage rows must be iterable mappings")
    try:
        return iter(rows)
    except TypeError:
        raise ValueError("linkage rows must be iterable mappings") from None


def _row_value(row: object, key: str) -> object:
    if not isinstance(row, Mapping) or key not in row or row[key] is None:
        raise ValueError("linkage input is malformed")
    return row[key]


def _numeric_key(value: object) -> str:
    if isinstance(value, bool):
        raise ValueError("linkage input is malformed")
    if isinstance(value, Integral):
        if value < 0:
            raise ValueError("linkage input is malformed")
        return str(value)
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value):
        raise ValueError("linkage input is malformed")
    return value.lstrip("0") or "0"


def _opaque_key(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("linkage input is malformed")
    normalized = value.strip()
    if not normalized:
        raise ValueError("linkage input is malformed")
    if any(unicodedata.category(character) == "Cc" for character in normalized):
        raise ValueError("linkage input is malformed")
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("linkage input is malformed") from None
    return normalized


def _add_episode_link(episode_to_person: dict[str, str], episode: str, person: str) -> None:
    existing = episode_to_person.get(episode)
    if existing is None:
        episode_to_person[episode] = person
    elif existing == person:
        raise ValueError("duplicate episode link")
    else:
        raise ValueError("conflicting episode link")


def build_episode_links(source: str, person_rows: Iterable[Mapping[str, object]], episode_rows: Iterable[Mapping[str, object]]) -> EpisodeLinks:
    """Build immutable, source-local episode-to-person mappings.

    For MIMIC/NWICU, ``person_rows`` are patient rows and ``episode_rows`` are
    admission rows. For eICU and SICdb, linkage rows may be supplied in either
    iterable (or divided between them), but any repeated episode is rejected.
    """
    source = _require_source(source)
    people = _iterable_rows(person_rows)
    episodes = _iterable_rows(episode_rows)
    episode_to_person: dict[str, str] = {}
    care_groups: dict[str, str] = {}

    if source in {"mimic", "nwicu"}:
        masters: set[str] = set()
        for row in people:
            person = _numeric_key(_row_value(row, "subject_id"))
            if person in masters:
                raise ValueError("duplicate person master row")
            masters.add(person)
        for row in episodes:
            episode = _numeric_key(_row_value(row, "hadm_id"))
            person = _numeric_key(_row_value(row, "subject_id"))
            if person not in masters:
                raise ValueError("orphan episode link")
            _add_episode_link(episode_to_person, episode, person)
        return EpisodeLinks(source, MappingProxyType(dict(episode_to_person)), None)

    if source == "eicu":
        for rows in (people, episodes):
            for row in rows:
                person = _opaque_key(_row_value(row, "uniquepid"))
                episode = _opaque_key(_row_value(row, "patientunitstayid"))
                care_value = row.get("patienthealthsystemstayid")
                care_group = _opaque_key(care_value) if care_value not in (None, "") else ""
                _add_episode_link(episode_to_person, episode, person)
                if care_group:
                    care_groups[episode] = care_group
        optional_care_groups = MappingProxyType(dict(care_groups)) if care_groups else None
        return EpisodeLinks(source, MappingProxyType(dict(episode_to_person)), optional_care_groups)

    # SICdb has no separate person master in this helper; every case carries
    # both PatientID and CaseID and is therefore a direct episode linkage row.
    for rows in (people, episodes):
        for row in rows:
            person = _numeric_key(_row_value(row, "PatientID"))
            episode = _numeric_key(_row_value(row, "CaseID"))
            _add_episode_link(episode_to_person, episode, person)
    return EpisodeLinks(source, MappingProxyType(dict(episode_to_person)), None)


def _event_value(event: object, key: str) -> object | None:
    if not isinstance(event, Mapping) or key not in event or event[key] is None:
        return None
    return event[key]


def _event_numeric_key(event: object, key: str) -> str | None:
    value = _event_value(event, key)
    if value is None:
        return None
    try:
        return _numeric_key(value)
    except ValueError:
        return None


def _event_opaque_key(event: object, key: str) -> str | None:
    value = _event_value(event, key)
    if value is None:
        return None
    try:
        return _opaque_key(value)
    except ValueError:
        return None


def validate_event_link(source: str, event: Mapping[str, object], links: EpisodeLinks) -> str | None:
    """Return a linked person only when the event's source-local keys agree."""
    source = _require_source(source)
    if not isinstance(links, EpisodeLinks):
        raise ValueError("links must be an EpisodeLinks instance")
    if source != links.source:
        raise ValueError("source does not match EpisodeLinks")
    if source in {"mimic", "nwicu"}:
        episode = _event_numeric_key(event, "hadm_id")
        person = _event_numeric_key(event, "subject_id")
        if episode is None or person is None or links.episode_to_person.get(episode) != person:
            return None
        return person
    if source == "eicu":
        episode = _event_opaque_key(event, "patientunitstayid")
        return None if episode is None else links.episode_to_person.get(episode)
    episode = _event_numeric_key(event, "CaseID")
    return None if episode is None else links.episode_to_person.get(episode)
