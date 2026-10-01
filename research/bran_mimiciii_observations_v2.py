"""Pure, two-pass MIMIC-III 21-field observation adapter.

The caller provides projected dictionary/event mappings and a private salt. No
paths, source files, serialization, printing, or model fitting are present.
MIMIC-III identifiers remain release-namespaced through the metadata adapter.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
import math
import re
from types import MappingProxyType

import numpy as np

from bran_clinical_chemistry_semantics_v1 import CHEMISTRY_FIELDS, MIMIC_CHEMISTRY_RULES
from bran_clinical_dictionary_binding_v1 import LABELS as CBC_LABELS
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_joint_lab_online_v1 import CBCAnchorAccumulator, JointLabSecondPass, authenticate_code_binding, convert_joint_bound_event
from bran_mimiciii_metadata_v2 import MimicIIIAge, MimicIIIMetadata, normalize_mimiciii_projection


ERROR = "mimiciii observation adapter rejected"
FIELDS = CBC_FIELDS + CHEMISTRY_FIELDS
_CODE = re.compile(r"[0-9]{1,12}")
_CHEMISTRY_LABELS = {
    field: frozenset(label.casefold() for _, (mapped, label) in MIMIC_CHEMISTRY_RULES.items() if mapped == field)
    for field in CHEMISTRY_FIELDS
}


def _fail() -> None:
    raise ValueError(ERROR) from None


def _code(value: object) -> str:
    if not isinstance(value, str) or _CODE.fullmatch(value) is None:
        _fail()
    return str(int(value))


def _text(value: object) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


@dataclass(frozen=True, repr=False)
class MimicIIIDictionaryBinding:
    code_to_field: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.code_to_field, Mapping) or not self.code_to_field:
            _fail()
        copied = dict(self.code_to_field)
        if (any(not isinstance(code, str) or not _CODE.fullmatch(code) or field not in FIELDS
                for code, field in copied.items()) or len(copied) != len(self.code_to_field)):
            _fail()
        object.__setattr__(self, "code_to_field", MappingProxyType(copied))


@dataclass(frozen=True, repr=False)
class MimicIIIObservationArrays:
    values: np.ndarray
    observed: np.ndarray
    provenance: np.ndarray
    age_value: np.ndarray
    age_lower: np.ndarray
    age_upper: np.ndarray
    age_kind: np.ndarray
    adult_qualified: np.ndarray
    person_group: np.ndarray
    encounter_group: np.ndarray
    split: np.ndarray
    person_weight: np.ndarray


def bind_mimiciii_dictionary(records: Iterable[Mapping[str, object]]) -> MimicIIIDictionaryBinding:
    """Bind exact MIMIC-III metadata labels without importing MIMIC-IV codes.

    A code must have exactly one field identity and each canonical field may have
    at most one approved code; multiple eligible codes are retained as an
    ambiguity rather than arbitrarily selected.
    """
    try:
        result: dict[str, str] = {}
        field_codes: dict[str, list[str]] = {field: [] for field in FIELDS}
        for row in normalize_mimiciii_projection("dictionary", records):
            code = _code(row["itemid"])
            label, fluid, category = _text(row["label"]), _text(row["fluid"]), _text(row["category"])
            fields = []
            if fluid == "blood" and category == "hematology":
                fields = [field for field in CBC_FIELDS if label in CBC_LABELS[field]]
            elif fluid == "blood" and category == "chemistry":
                fields = [field for field in CHEMISTRY_FIELDS if label in _CHEMISTRY_LABELS[field]]
            if len(fields) > 1 or code in result:
                _fail()
            if fields:
                result[code] = fields[0]
                field_codes[fields[0]].append(code)
        if not result or any(len(codes) > 1 for codes in field_codes.values()):
            _fail()
        return MimicIIIDictionaryBinding(result)
    except Exception:
        _fail()


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or value != value.strip() or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed.tzinfo is None else None


def _identifiers(row: Mapping[str, object]) -> tuple[str, str]:
    try:
        subject = "mimiciii:subject:" + _code(row["subject_id"])
        encounter = "mimiciii:hadm:" + _code(row["hadm_id"])
    except Exception:
        _fail()
    return subject, encounter


def _event(metadata: MimicIIIMetadata, authenticated_binding, raw: Mapping[str, object]):
    """Adapt one exact-schema row; source-link failures are row-level missingness."""
    normalized = list(normalize_mimiciii_projection("events", (raw,)))[0]
    try:
        person, encounter = _identifiers(normalized)
    except ValueError:
        return None
    # Do not manufacture a join for outpatient/missing HADM rows or mismatches.
    if metadata.encounter_to_subject.get(encounter) != person:
        return None
    age = metadata.encounter_age.get(encounter)
    if age is None:
        return None
    if not age.adult_qualified:
        return None
    admitted = metadata.encounter_admittime.get(encounter)
    stamped = _timestamp(normalized["charttime"])
    if admitted is None or stamped is None:
        return None
    offset = (stamped - admitted).total_seconds() / 60.0
    if not 0.0 <= offset <= 1440.0:
        return None
    try:
        code = _code(normalized["itemid"])
    except ValueError:
        return None
    return convert_joint_bound_event("mimiciii", encounter, person, code, normalized["valuenum"],
                                     normalized["valueuom"], offset, 1,
                                     authenticated_binding)


def _factory(factory: object):
    if not callable(factory):
        _fail()
    try:
        rows = factory()
        if isinstance(rows, (str, bytes, Mapping)):
            _fail()
        return iter(rows)
    except Exception:
        _fail()


def _readonly(value: np.ndarray, dtype) -> np.ndarray:
    copied = np.array(value, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


def _split(person: str, salt: bytes) -> int:
    value = hmac.new(salt, b"bran-multisource-v2\x00mimiciii\x00" + person.encode("utf-8"), hashlib.sha256).digest()
    bucket = int.from_bytes(value[:8], "big") % 10
    return 0 if bucket < 8 else (1 if bucket == 8 else 2)


def _pack(rows: list[tuple[str, str, np.ndarray, np.ndarray, MimicIIIAge]], salt: bytes) -> MimicIIIObservationArrays:
    if not rows or not isinstance(salt, bytes) or len(salt) != 32:
        _fail()
    rows.sort(key=lambda item: item[0])
    people = sorted({person for _, person, *_ in rows})
    encounters = sorted({encounter for encounter, *_ in rows})
    person_ids, encounter_ids = {person: i for i, person in enumerate(people)}, {encounter: i for i, encounter in enumerate(encounters)}
    values = np.stack([value for _, _, value, _, _ in rows]).astype(np.float64)
    observed = np.stack([mask for _, _, _, mask, _ in rows]).astype(bool)
    ages = [age for *_, age in rows]
    if values.shape != (len(rows), len(FIELDS)) or observed.shape != values.shape or np.any(observed[:, :9].sum(axis=1) < 2):
        _fail()
    values[~observed] = np.nan
    if (not np.isfinite(values[observed]).all() or np.any(values[:, :9][observed[:, :9]] <= 0)
            or not np.isnan(values[~observed]).all()):
        _fail()
    lower = np.array([age.lower for age in ages], dtype=np.float64)
    upper = np.array([age.upper for age in ages], dtype=np.float64)
    kind = np.array([age.kind for age in ages], dtype=np.int64)
    value = np.full(len(rows), np.nan, dtype=np.float64)  # interval/unknown contract: inactive value remains NaN.
    adult = np.array([age.adult_qualified for age in ages], dtype=bool)
    if not adult.all() or np.any((kind != 1) & (kind != 3)):
        _fail()
    if np.any(kind == 1):
        interval = kind == 1
        if not (np.isfinite(lower[interval]).all() and np.isfinite(upper[interval]).all() and np.all(lower[interval] >= 0) and np.all(upper[interval] >= lower[interval])):
            _fail()
    if np.any(kind == 3):
        unknown = kind == 3
        lower[unknown], upper[unknown] = np.nan, np.nan
    person_group = np.array([person_ids[person] for _, person, *_ in rows], dtype=np.int64)
    encounter_group = np.array([encounter_ids[encounter] for encounter, *_ in rows], dtype=np.int64)
    person_splits = np.array([_split(person, salt) for person in people], dtype=np.uint8)
    counts = np.bincount(person_group, minlength=len(people))
    return MimicIIIObservationArrays(_readonly(values, np.float64), _readonly(observed, bool),
                                     _readonly(observed.astype(np.uint8), np.uint8), _readonly(value, np.float64),
                                     _readonly(lower, np.float64), _readonly(upper, np.float64), _readonly(kind, np.int64),
                                     _readonly(adult, bool), _readonly(person_group, np.int64), _readonly(encounter_group, np.int64),
                                     _readonly(person_splits[person_group], np.uint8), _readonly(1.0 / counts[person_group], np.float64))


def build_mimiciii_observations(metadata: MimicIIIMetadata, binding: MimicIIIDictionaryBinding,
                                event_factory: Callable[[], Iterable[Mapping[str, object]]], salt: bytes) -> MimicIIIObservationArrays:
    """Run the fixed two-pass selection over a factory of uppercase projected rows."""
    try:
        if not isinstance(metadata, MimicIIIMetadata) or not isinstance(binding, MimicIIIDictionaryBinding):
            _fail()
        authenticated = authenticate_code_binding("mimiciii", binding.code_to_field)
        first = CBCAnchorAccumulator("mimiciii")
        for row in _factory(event_factory):
            converted = _event(metadata, authenticated, row)
            if converted is not None:
                first.add(converted)
        anchors = first.finalize()
        second = JointLabSecondPass(anchors)
        for row in _factory(event_factory):
            converted = _event(metadata, authenticated, row)
            if converted is not None:
                second.add(converted)
        selected = []
        for encounter, person, snapshot in second.iterate_snapshots():
            if snapshot.observed[:len(CBC_FIELDS)].sum() < 2:
                continue
            age = metadata.encounter_age.get(encounter)
            if age is None or not age.adult_qualified:
                _fail()
            selected.append((encounter, person, snapshot.values, snapshot.observed, age))
        return _pack(selected, salt)
    except Exception:
        _fail()


def validate_observations(value: object) -> None:
    """Fail-closed private-array schema check used by a future local runner."""
    try:
        if not isinstance(value, MimicIIIObservationArrays):
            _fail()
        arrays = (value.values, value.observed, value.provenance, value.age_value, value.age_lower,
                  value.age_upper, value.age_kind, value.adult_qualified, value.person_group,
                  value.encounter_group, value.split, value.person_weight)
        if any(not isinstance(item, np.ndarray) or item.flags.writeable for item in arrays):
            _fail()
        n = value.values.shape[0]
        if (n <= 0 or value.values.dtype != np.float64 or value.values.shape != (n, len(FIELDS))
                or value.observed.dtype != np.bool_ or value.observed.shape != value.values.shape
                or value.provenance.dtype != np.uint8 or not np.array_equal(value.provenance, value.observed.astype(np.uint8))
                or not np.isfinite(value.values[value.observed]).all() or not np.isnan(value.values[~value.observed]).all()
                or np.any(value.values[:, :9][value.observed[:, :9]] <= 0)
                or np.any(value.values[:, 9:][value.observed[:, 9:]] < 0)
                or np.any(value.observed[:, :9].sum(axis=1) < 2)):
            _fail()
        for array in (value.age_value, value.age_lower, value.age_upper):
            if array.dtype != np.float64 or array.shape != (n,):
                _fail()
        if (not np.isnan(value.age_value).all() or value.age_kind.dtype != np.int64 or value.age_kind.shape != (n,)
                or not np.isin(value.age_kind, (1, 3)).all() or value.adult_qualified.dtype != np.bool_
                or value.adult_qualified.shape != (n,) or not value.adult_qualified.all()):
            _fail()
        interval, unknown = value.age_kind == 1, value.age_kind == 3
        if (not (np.isfinite(value.age_lower[interval]).all() and np.isfinite(value.age_upper[interval]).all()
                 and np.all(value.age_lower[interval] >= 0) and np.all(value.age_upper[interval] >= value.age_lower[interval]))
                or not (np.isnan(value.age_lower[unknown]).all() and np.isnan(value.age_upper[unknown]).all())):
            _fail()
        for array in (value.person_group, value.encounter_group):
            if array.dtype != np.int64 or array.shape != (n,) or np.any(array < 0):
                _fail()
        if (value.split.dtype != np.uint8 or value.split.shape != (n,) or not np.isin(value.split, (0, 1, 2)).all()
                or value.person_weight.dtype != np.float64 or value.person_weight.shape != (n,)
                or not np.isfinite(value.person_weight).all() or np.any(value.person_weight <= 0)):
            _fail()
        people, counts = np.unique(value.person_group, return_counts=True)
        if not np.array_equal(people, np.arange(len(people), dtype=np.int64)):
            _fail()
        if not np.array_equal(value.person_weight, 1.0 / counts[value.person_group]):
            _fail()
        first = np.unique(value.person_group, return_index=True)[1]
        if not np.array_equal(value.split, value.split[first][value.person_group]):
            _fail()
        encounters = np.unique(value.encounter_group)
        if len(encounters) != n or not np.array_equal(encounters, np.arange(n, dtype=np.int64)):
            _fail()
    except Exception:
        _fail()
