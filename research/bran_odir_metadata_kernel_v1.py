"""Private, in-memory ODIR metadata and archive-member qualification.

This module never authenticates an archive, selects a duplicate member, or
asserts that a supplied identifier is an externally verified identity.  Its
records are deliberately private; ``validate_aggregate`` accepts only the
coarsened, closed public summary.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import re
from typing import Any


_ERROR = "odir metadata qualification failed"
_FIELDS = (
    "ID", "Patient Age", "Patient Sex", "Left-Fundus", "Right-Fundus",
    "N", "D", "G", "C", "A", "H", "M", "O",
)
_LABELS = _FIELDS[5:]
_MISSING = frozenset({"", "na", "nan", "unknown"})
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.(?:jpg|jpeg|png)", re.IGNORECASE)
_ID = re.compile(r"[0-9]+")
_COUNTS = (
    "input_rows", "grouped_patients", "repeated_rows",
    "metadata_conflict_patients", "invalid_metadata_patients",
    "shared_reference_patients", "adult_metadata_eligible_patients",
    "fully_unique_linked_adult_patients", "partly_linked_adult_patients",
    "adult_patients_with_ambiguous_links", "adult_patients_with_missing_links",
    "uniquely_linked_adult_eyes",
)
_FLAGS = {
    "training_admitted": False,
    "patient_identity_externally_authenticated": False,
    "cross_source_overlap_cleared": False,
    "patient_level_output_emitted": False,
}
_AGGREGATE_KEYS = frozenset({"schema", "status", "counts_rounded_down20", *_FLAGS})


def _fail() -> None:
    raise ValueError(_ERROR) from None


@dataclass(frozen=True, repr=False)
class EyeLink:
    """One private reference and every exact archive-member candidate."""

    reference: str | None
    member_candidates: tuple[str, ...]


@dataclass(frozen=True, repr=False)
class PatientRecord:
    """Private grouped ODIR metadata; never serialize this outside local scope."""

    patient_id: str
    age: float | None
    sex: str | None
    labels: tuple[int | None, ...]
    left: EyeLink
    right: EyeLink
    reasons: tuple[str, ...]
    input_row_count: int
    signature_count: int
    signature_variants: tuple[tuple[object, ...], ...]


@dataclass(frozen=True, repr=False)
class PrivateRecords:
    """Private return container.  ``records`` retains failures for audit."""

    records: tuple[PatientRecord, ...]


@dataclass(frozen=True, repr=False)
class _Parsed:
    patient_id: str
    age: float | None
    sex: str | None
    left: str | None
    right: str | None
    labels: tuple[int | None, ...]
    invalid: frozenset[str]

    @property
    def signature(self) -> tuple[object, ...]:
        return (self.age, self.sex, self.left, self.right, self.labels, tuple(sorted(self.invalid)))


def _missing(value: object) -> bool:
    return value is None or (type(value) is str and value.strip().lower() in _MISSING)


def _age(value: object) -> tuple[float | None, str | None]:
    if _missing(value):
        return None, None
    if isinstance(value, bool):
        return None, "invalid_age"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, "invalid_age"
    if not math.isfinite(parsed) or parsed < 0 or parsed > 130:
        return None, "invalid_age"
    return parsed, None


def _sex(value: object) -> tuple[str | None, str | None]:
    if _missing(value):
        return None, None
    if type(value) is not str:
        return None, "invalid_sex"
    normalized = value.strip().lower()
    if normalized in {"male", "m"}:
        return "male", None
    if normalized in {"female", "f"}:
        return "female", None
    return None, "invalid_sex"


def _label(value: object) -> tuple[int | None, str | None]:
    if _missing(value):
        return None, None
    if isinstance(value, bool):
        return None, "invalid_label"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, "invalid_label"
    if not math.isfinite(parsed) or parsed not in (0.0, 1.0):
        return None, "invalid_label"
    return int(parsed), None


def _reference(value: object, side: str) -> tuple[str | None, str | None]:
    if type(value) is not str or _NAME.fullmatch(value) is None:
        return None, "invalid_" + side + "_reference"
    return value, None


def _row(value: object) -> _Parsed:
    if type(value) is not dict or set(value) != set(_FIELDS):
        _fail()
    patient_id = value["ID"]
    if type(patient_id) is not str or _ID.fullmatch(patient_id) is None:
        _fail()
    age, age_error = _age(value["Patient Age"])
    sex, sex_error = _sex(value["Patient Sex"])
    left, left_error = _reference(value["Left-Fundus"], "left")
    right, right_error = _reference(value["Right-Fundus"], "right")
    labels, label_errors = [], set()
    for name in _LABELS:
        label, error = _label(value[name])
        labels.append(label)
        if error:
            label_errors.add(error)
    invalid = {error for error in (age_error, sex_error, left_error, right_error) if error}
    invalid.update(label_errors)
    if left is not None and left == right:
        invalid.add("same_eye_reference")
    return _Parsed(patient_id, age, sex, left, right, tuple(labels), frozenset(invalid))


def _members(value: object) -> dict[str, tuple[str, ...]]:
    if type(value) is not list or any(type(name) is not str or not name for name in value):
        _fail()
    if value != sorted(value) or len(set(value)) != len(value):
        _fail()
    result: dict[str, list[str]] = defaultdict(list)
    for name in value:
        if (name.startswith("/") or "\\" in name or "\x00" in name
                or any(part in {"", ".", ".."} for part in name.split("/"))):
            _fail()
        basename = name.rsplit("/", 1)[-1]
        if _NAME.fullmatch(basename) is None:
            _fail()
        result[basename].append(name)
    return {name: tuple(candidates) for name, candidates in result.items()}


def _coarse(count: int) -> int:
    return (count // 20) * 20


def _valid_count(value: object) -> bool:
    return type(value) is int and value >= 0 and value % 20 == 0


def validate_aggregate(value: object) -> None:
    """Accept only the public, closed, coarsened qualification aggregate."""
    if type(value) is not dict or set(value) != _AGGREGATE_KEYS:
        _fail()
    if value.get("schema") != "bran-odir-metadata-qualification-v1" or value.get("status") != "qualified":
        _fail()
    counts = value.get("counts_rounded_down20")
    if type(counts) is not dict or set(counts) != set(_COUNTS):
        _fail()
    if any(not _valid_count(counts[name]) for name in _COUNTS):
        _fail()
    if any(value.get(name) is not expected for name, expected in _FLAGS.items()):
        _fail()
    if counts["grouped_patients"] > counts["input_rows"] or counts["repeated_rows"] > counts["input_rows"]:
        _fail()
    patient_counts = (
        "metadata_conflict_patients", "invalid_metadata_patients", "shared_reference_patients",
        "adult_metadata_eligible_patients", "fully_unique_linked_adult_patients",
        "partly_linked_adult_patients", "adult_patients_with_ambiguous_links",
        "adult_patients_with_missing_links",
    )
    if any(counts[name] > counts["grouped_patients"] for name in patient_counts):
        _fail()
    eligible = counts["adult_metadata_eligible_patients"]
    if (counts["fully_unique_linked_adult_patients"] + counts["partly_linked_adult_patients"] > eligible
            or counts["adult_patients_with_ambiguous_links"] > eligible
            or counts["adult_patients_with_missing_links"] > eligible
            or counts["uniquely_linked_adult_eyes"] > 2 * eligible + 20):
        _fail()


def _signature_key(value: tuple[object, ...]) -> str:
    """Deterministic private ordering across None/numeric/string fields."""
    return repr(value)


def qualify(rows: list[dict[str, Any]], members: list[str]) -> tuple[PrivateRecords, dict[str, object]]:
    """Qualify caller-local ODIR metadata without selecting or exposing members.

    ``members`` must be a sorted, duplicate-free list of archive member names.
    Candidate lists remain private because directory-bearing member names can be
    identifying source details.  Counts in the aggregate are floored to 20.
    Consequently a public zero means only that fewer than twenty were observed,
    not that the underlying count was zero.
    """
    if type(rows) is not list or not rows:
        _fail()
    member_index = _members(members)
    grouped: dict[str, list[_Parsed]] = {}
    for item in rows:
        parsed = _row(item)
        grouped.setdefault(parsed.patient_id, []).append(parsed)

    ref_owners: dict[str, set[str]] = defaultdict(set)
    for patient_id, items in grouped.items():
        for item in items:
            for reference in (item.left, item.right):
                if reference is not None:
                    ref_owners[reference].add(patient_id)
    shared = {reference for reference, owners in ref_owners.items() if len(owners) > 1}

    counts = {name: 0 for name in _COUNTS}
    counts["input_rows"] = len(rows)
    counts["grouped_patients"] = len(grouped)
    counts["repeated_rows"] = len(rows) - len(grouped)
    records: list[PatientRecord] = []
    for patient_id in sorted(grouped, key=lambda value: (int(value), value)):
        items = grouped[patient_id]
        signatures = tuple(sorted({item.signature for item in items}, key=_signature_key))
        canonical = next(item for item in items if item.signature == signatures[0])
        reasons = set(canonical.invalid)
        if len(signatures) != 1:
            reasons.add("metadata_conflict")
            counts["metadata_conflict_patients"] += 1
        if any(item.invalid for item in items):
            reasons.add("invalid_metadata")
            counts["invalid_metadata_patients"] += 1
        references = {reference for item in items for reference in (item.left, item.right) if reference is not None}
        if references & shared:
            reasons.add("shared_reference")
            counts["shared_reference_patients"] += 1

        left_candidates = member_index.get(canonical.left, ()) if canonical.left is not None else ()
        right_candidates = member_index.get(canonical.right, ()) if canonical.right is not None else ()
        # Shared references are a linkage quarantine, not an alteration of the
        # patient's demographic/label metadata status.  Keep those axes
        # separate so the private audit can distinguish the two failures.
        metadata_adult = (len(signatures) == 1 and not any(item.invalid for item in items)
                          and canonical.age is not None and canonical.age >= 18)
        if canonical.age is None or (canonical.age is not None and canonical.age < 18):
            reasons.add("age_not_adult_or_unknown")
        if metadata_adult:
            counts["adult_metadata_eligible_patients"] += 1
            link_lengths = (len(left_candidates), len(right_candidates))
            ambiguous = any(length > 1 for length in link_lengths)
            missing = any(length == 0 for length in link_lengths)
            unique = sum(length == 1 for length in link_lengths)
            if ambiguous:
                reasons.add("ambiguous_member_link")
                counts["adult_patients_with_ambiguous_links"] += 1
            if missing:
                reasons.add("missing_member_link")
                counts["adult_patients_with_missing_links"] += 1
            if "shared_reference" not in reasons:
                if unique == 2 and not ambiguous and not missing:
                    counts["fully_unique_linked_adult_patients"] += 1
                    counts["uniquely_linked_adult_eyes"] += 2
                elif unique:
                    counts["partly_linked_adult_patients"] += 1
                    counts["uniquely_linked_adult_eyes"] += unique

        records.append(PatientRecord(
            patient_id=patient_id, age=canonical.age, sex=canonical.sex, labels=canonical.labels,
            left=EyeLink(canonical.left, left_candidates), right=EyeLink(canonical.right, right_candidates),
            reasons=tuple(sorted(reasons)), input_row_count=len(items), signature_count=len(signatures),
            signature_variants=signatures,
        ))
    aggregate: dict[str, object] = {
        "schema": "bran-odir-metadata-qualification-v1", "status": "qualified",
        "counts_rounded_down20": {name: _coarse(count) for name, count in counts.items()}, **_FLAGS,
    }
    validate_aggregate(aggregate)
    return PrivateRecords(tuple(records)), aggregate


__all__ = ["EyeLink", "PatientRecord", "PrivateRecords", "qualify", "validate_aggregate"]
