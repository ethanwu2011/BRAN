"""Pure MIMIC-III metadata/projection adapter for a future original-unit cache.

It accepts only caller-supplied synthetic mappings, performs no I/O, and never
prints patient-level metadata.  This is not a MIMIC-IV adapter: identifiers are
explicitly release-namespaced before they leave this module.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
import math
from numbers import Integral
import re
from types import MappingProxyType

from bran_multisource_age_v2 import INTERVAL, UNKNOWN


_INVALID = "mimiciii metadata inputs invalid"
_COLUMNS = {
    "patients": ("SUBJECT_ID", "DOB"),
    "admissions": ("SUBJECT_ID", "HADM_ID", "ADMITTIME"),
    "dictionary": ("ITEMID", "LABEL", "FLUID", "CATEGORY"),
    "events": ("SUBJECT_ID", "HADM_ID", "ITEMID", "VALUENUM", "VALUEUOM", "CHARTTIME"),
}
_LOWER = {name: tuple(value.casefold() for value in columns) for name, columns in _COLUMNS.items()}


@dataclass(frozen=True, repr=False)
class MimicIIISourceIdentity:
    family: str = "mimic"
    release: str = "mimiciii"


@dataclass(frozen=True, repr=False)
class MimicIIIAge:
    """Original-unit age metadata; kind uses the V2 age codes.

    Calendar-derived normal ages retain an apparent numeric value and a
    completed-year interval.  A downstream typed-age bridge must use the
    interval fields for ``kind=INTERVAL`` rather than treating value as exact.
    """

    value: float
    lower: float
    upper: float
    kind: int
    adult_qualified: bool


@dataclass(frozen=True, repr=False)
class MimicIIIMetadata:
    source_identity: MimicIIISourceIdentity
    encounter_to_subject: Mapping[str, str]
    encounter_admittime: Mapping[str, datetime | None]
    encounter_age: Mapping[str, MimicIIIAge]


def _invalid() -> None:
    raise ValueError(_INVALID)


def mimiciii_required_columns(table: str, columns: object) -> tuple[str, ...]:
    """Validate an exact uppercase projection and return its lowercase schema."""

    if table not in _COLUMNS or not isinstance(columns, tuple) or len(columns) != len(_COLUMNS[table]):
        _invalid()
    if any(not isinstance(column, str) for column in columns) or len(set(columns)) != len(columns) or set(columns) != set(_COLUMNS[table]):
        _invalid()
    return _LOWER[table]


def _records(records: object):
    if isinstance(records, (str, bytes, Mapping)):
        _invalid()
    try:
        return iter(records)
    except TypeError:
        _invalid()


def normalize_mimiciii_projection(table: str, records: Iterable[Mapping[str, object]]):
    """Stream exact uppercase projected rows into the existing lowercase schema.

    Extra columns are rejected, so unrelated values cannot pass through the
    adapter.  Dictionary ``ITEMID`` values must be unique; event rows are not
    deduplicated because repeated laboratory measurements can be meaningful.
    """

    required = _COLUMNS.get(table)
    if required is None:
        _invalid()
    seen_dictionary: set[object] = set()
    for row in _records(records):
        if not isinstance(row, Mapping) or set(row) != set(required):
            _invalid()
        if table == "dictionary":
            itemid = _numeric_key(row["ITEMID"])
            if itemid in seen_dictionary:
                _invalid()
            seen_dictionary.add(itemid)
        yield {lower: row[upper] for upper, lower in zip(required, _LOWER[table])}


def _numeric_key(value: object) -> str:
    if isinstance(value, bool):
        _invalid()
    if isinstance(value, Integral):
        if value < 0:
            _invalid()
        return str(value)
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        _invalid()
    return value.lstrip("0") or "0"


def _subject_key(value: object) -> str:
    return "mimiciii:subject:" + _numeric_key(value)


def _encounter_key(value: object) -> str:
    return "mimiciii:hadm:" + _numeric_key(value)


def _datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is not None or not (1 <= parsed.year <= 9999):
        return None
    return parsed


def _unknown(*, adult: bool = False) -> MimicIIIAge:
    return MimicIIIAge(math.nan, math.nan, math.nan, UNKNOWN, adult)


def _documented_shift(dob: datetime | None, first_admission: datetime | None) -> bool:
    """Match MIMIC-III's documented DOB=first-admission-minus-300-years rule.

    This compares calendar dates, not elapsed seconds.  A February 29 admission
    whose target year has no February 29 has no exact counterpart and therefore
    does not establish the shift; it is never coerced to February 28 or March 1.
    """
    if dob is None or first_admission is None or first_admission.year < 301:
        return False
    target_year = first_admission.year - 300
    try:
        expected = date(target_year, first_admission.month, first_admission.day)
    except ValueError:
        return False
    return dob.date() == expected


def _calendar_age(dob: datetime | None, admitted: datetime | None, *, shifted: bool) -> MimicIIIAge:
    if dob is None or admitted is None or admitted < dob:
        return _unknown()
    if shifted:
        # The documented shift says only that the person was older than 89 at
        # some point; no numeric age or threshold at this encounter is inferred.
        return _unknown(adult=True)
    years = admitted.year - dob.year - ((admitted.month, admitted.day) < (dob.month, dob.day))
    if years < 0 or years > 120:
        return _unknown()
    lower = float(years)
    return MimicIIIAge(lower, lower, lower + 1.0, INTERVAL, years >= 18)


def build_mimiciii_metadata(patients: Iterable[Mapping[str, object]], admissions: Iterable[Mapping[str, object]]) -> MimicIIIMetadata:
    """Build source-local patient/admission metadata without choosing duplicates.

    A patient can have many distinct admissions.  Every patient master and every
    encounter key itself is unique; duplicate or conflicting links are errors.
    Invalid/missing dates produce an unknown age while retaining a valid
    encounter link for future observed-clinical handling.
    """

    people: dict[str, datetime | None] = {}
    for row in normalize_mimiciii_projection("patients", patients):
        subject = _subject_key(row["subject_id"])
        if subject in people:
            _invalid()
        people[subject] = _datetime(row["dob"])
    links: dict[str, str] = {}
    times: dict[str, datetime | None] = {}
    for row in normalize_mimiciii_projection("admissions", admissions):
        subject, encounter = _subject_key(row["subject_id"]), _encounter_key(row["hadm_id"])
        if subject not in people or encounter in links:
            _invalid()
        admitted = _datetime(row["admittime"])
        links[encounter], times[encounter] = subject, admitted
    first_valid: dict[str, datetime] = {}
    for encounter, subject in links.items():
        admitted = times[encounter]
        if admitted is not None and (subject not in first_valid or admitted < first_valid[subject]):
            first_valid[subject] = admitted
    shifted = {subject for subject, first in first_valid.items() if _documented_shift(people[subject], first)}
    ages = {encounter: _calendar_age(people[subject], admitted, shifted=subject in shifted)
            for encounter, subject in links.items() for admitted in (times[encounter],)}
    return MimicIIIMetadata(MimicIIISourceIdentity(), MappingProxyType(dict(links)),
                            MappingProxyType(dict(times)), MappingProxyType(dict(ages)))


def event_age(metadata: MimicIIIMetadata, event: Mapping[str, object]) -> MimicIIIAge:
    """Return linked age only when the event subject and admission agree exactly."""

    if not isinstance(metadata, MimicIIIMetadata):
        _invalid()
    normalized = list(normalize_mimiciii_projection("events", (event,)))[0]
    subject, encounter = _subject_key(normalized["subject_id"]), _encounter_key(normalized["hadm_id"])
    if metadata.encounter_to_subject.get(encounter) != subject:
        _invalid()
    return metadata.encounter_age[encounter]
