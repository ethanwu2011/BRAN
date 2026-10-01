"""Closed, metadata-only BRSET grouped-readiness assessment.

This module intentionally accepts only caller-provided row dictionaries.  It
does not discover files, inspect image content, link sources, or train a model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from types import MappingProxyType
import re
import struct
import unicodedata
from collections.abc import Mapping

import numpy as np


SOURCE = "brset"
SCHEMA = "bran_retinal_group_readiness_v1"
STATUS = "metadata_assessed"
ENDPOINTS = (
    "diabetic_retinopathy",
    "macular_edema",
    "scar",
    "nevus",
    "amd",
    "vascular_occlusion",
    "hypertensive_retinopathy",
    "drusens",
    "hemorrhage",
    "retinal_detachment",
    "myopic_fundus",
    "increased_cup_disc",
    "other",
)
_ROW_FIELDS = frozenset(("image_id", "patient_id", "patient_age", "exam_eye", *ENDPOINTS))
_IMAGE_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
_UNKNOWN = frozenset({"", "NA", "nan"})
_SPLIT_NAMESPACE = b"bran-retinal-group-readiness-v1\x00"
_SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
_SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True, repr=False)
class RetinalGroupReadiness:
    """Private per-image metadata assessment; do not publish this object."""

    image_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    ages: np.ndarray
    adult_eligible: np.ndarray
    split: np.ndarray
    labels: np.ndarray
    observed: np.ndarray
    safe_summary: Mapping[str, object]


def _readonly(values: object, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _invalid() -> ValueError:
    # One generic error prevents malformed private metadata from becoming a
    # disclosure channel through detailed validation failures.
    return ValueError("invalid retinal readiness metadata")


def _require_salt(salt: object) -> bytes:
    if not isinstance(salt, bytes) or len(salt) < 16:
        raise _invalid()
    return salt


def _assign_split(patient_id: str, *, salt: bytes) -> str:
    """Versioned BRSET-only deterministic 80/10/10 person assignment."""
    try:
        source_bytes = SOURCE.encode("utf-8")
        person_bytes = patient_id.encode("utf-8")
    except UnicodeEncodeError as error:
        raise _invalid() from error
    message = (
        _SPLIT_NAMESPACE
        + struct.pack(">I", len(source_bytes))
        + source_bytes
        + struct.pack(">I", len(person_bytes))
        + person_bytes
    )
    digest = hmac.new(salt, message, hashlib.sha256).digest()
    bucket = (int.from_bytes(digest, byteorder="big", signed=False) * 100) >> 256
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _validate_patient_id(value: str) -> None:
    if not value:
        raise _invalid()
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise _invalid() from error
    if any(
        character.isspace() or unicodedata.category(character) in {"Cc", "Cs"}
        for character in value
    ):
        raise _invalid()


def _parse_age(value: str) -> float:
    if value in _UNKNOWN:
        return float("nan")
    if value != value.strip():
        raise _invalid()
    try:
        result = float(value)
    except ValueError as error:
        raise _invalid() from error
    if not np.isfinite(result) or result < 0.0:
        raise _invalid()
    return result


def _parse_eye(value: str) -> None:
    if value not in {"", "1", "2"}:
        raise _invalid()


def _parse_binary(value: str) -> tuple[float, bool]:
    if value in _UNKNOWN:
        return float("nan"), False
    if value != value.strip():
        raise _invalid()
    try:
        numeric = float(value)
    except ValueError as error:
        raise _invalid() from error
    if not np.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise _invalid()
    return numeric, True


def _coarse(value: int) -> int | str:
    if value == 0:
        return 0
    if value < 20:
        return "<20"
    return (value // 20) * 20


def _freeze_mapping(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_mapping(item) for key, item in value.items()})
    return value


def _summary(
    patient_ids: tuple[str, ...], adult_eligible: np.ndarray, split: np.ndarray,
    labels: np.ndarray, observed: np.ndarray,
) -> Mapping[str, object]:
    person_to_rows: dict[str, list[int]] = {}
    for index, patient in enumerate(patient_ids):
        person_to_rows.setdefault(patient, []).append(index)
    unique_people = tuple(person_to_rows)
    total_people = len(unique_people)
    eligible_people = [patient for patient in unique_people if bool(adult_eligible[person_to_rows[patient][0]])]
    counts: dict[str, object] = {
        "total": {"images": _coarse(len(patient_ids)), "people": _coarse(total_people)},
        "eligible": {"images": _coarse(int(np.sum(adult_eligible))), "people": _coarse(len(eligible_people))},
        "splits": {},
    }
    support: dict[str, object] = {}
    for split_name in _SPLIT_NAMES:
        code = _SPLIT_CODES[split_name]
        split_rows = adult_eligible & (split == code)
        split_people = [
            patient for patient in eligible_people
            if int(split[person_to_rows[patient][0]]) == code
        ]
        counts["splits"][split_name] = {
            "images": _coarse(int(np.sum(split_rows))),
            "people": _coarse(len(split_people)),
        }
        endpoint_support: dict[str, object] = {}
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            positive = 0
            negative = 0
            for patient in split_people:
                rows = person_to_rows[patient]
                endpoint_observed = observed[rows, endpoint_index]
                endpoint_values = labels[rows, endpoint_index]
                if np.any(endpoint_observed & (endpoint_values == 1.0)):
                    positive += 1
                elif np.all(endpoint_observed & (endpoint_values == 0.0)):
                    negative += 1
            endpoint_support[endpoint] = {
                "supported": bool(positive >= 20 and negative >= 20),
                "positive_patients": _coarse(positive),
                "negative_patients": _coarse(negative),
            }
        support[split_name] = endpoint_support
    return _freeze_mapping({
        "schema": SCHEMA,
        "status": STATUS,
        "source": SOURCE,
        "grouping_consistent": True,
        "split_patient_disjoint": True,
        "image_content_verified": False,
        "cross_source_overlap_checked": False,
        "model_trained": False,
        "patient_level_output_emitted": False,
        "counts": counts,
        "label_support": support,
    })


def assess(rows: list[dict[str, str]], *, salt: bytes) -> RetinalGroupReadiness:
    """Assess closed BRSET metadata without content, overlap, or training claims."""
    secret = _require_salt(salt)
    if type(rows) is not list or not rows:
        raise _invalid()

    image_ids: list[str] = []
    patient_ids: list[str] = []
    ages: list[float] = []
    label_values: list[list[float]] = []
    label_observed: list[list[bool]] = []
    seen_images: set[str] = set()
    for row in rows:
        if type(row) is not dict or set(row) != _ROW_FIELDS:
            raise _invalid()
        if any(type(row[field]) is not str for field in _ROW_FIELDS):
            raise _invalid()
        image_id = row["image_id"]
        patient_id = row["patient_id"]
        if not image_id or _IMAGE_ID.fullmatch(image_id) is None or image_id in seen_images:
            raise _invalid()
        _validate_patient_id(patient_id)
        seen_images.add(image_id)
        age = _parse_age(row["patient_age"])
        _parse_eye(row["exam_eye"])
        parsed = [_parse_binary(row[endpoint]) for endpoint in ENDPOINTS]
        image_ids.append(image_id)
        patient_ids.append(patient_id)
        ages.append(age)
        label_values.append([value for value, _ in parsed])
        label_observed.append([present for _, present in parsed])

    n_rows = len(rows)
    ages_array = np.asarray(ages, dtype=np.float64)
    labels = np.asarray(label_values, dtype=np.float32).reshape(n_rows, len(ENDPOINTS))
    observed = np.asarray(label_observed, dtype=bool).reshape(n_rows, len(ENDPOINTS))
    adult = np.zeros(n_rows, dtype=bool)
    split = np.full(n_rows, -1, dtype=np.int8)
    patient_to_rows: dict[str, list[int]] = {}
    for index, patient_id in enumerate(patient_ids):
        patient_to_rows.setdefault(patient_id, []).append(index)
    for patient_id, indices in patient_to_rows.items():
        group_ages = ages_array[indices]
        if np.all(np.isfinite(group_ages)) and np.all(group_ages >= 18.0):
            assigned = _SPLIT_CODES[_assign_split(patient_id, salt=secret)]
            adult[indices] = True
            split[indices] = assigned

    image_tuple = tuple(image_ids)
    patient_tuple = tuple(patient_ids)
    return RetinalGroupReadiness(
        image_ids=image_tuple,
        patient_ids=patient_tuple,
        ages=_readonly(ages_array, dtype=np.float64),
        adult_eligible=_readonly(adult, dtype=bool),
        split=_readonly(split, dtype=np.int8),
        labels=_readonly(labels, dtype=np.float32),
        observed=_readonly(observed, dtype=bool),
        safe_summary=_summary(patient_tuple, adult, split, labels, observed),
    )


def _valid_coarse(value: object) -> bool:
    return (type(value) is int and value >= 0 and value % 20 == 0) or (
        type(value) is str and value == "<20"
    )


def validate_summary(summary: object) -> None:
    """Reject any non-public, non-coarsened, or schema-unknown summary value."""
    if not isinstance(summary, Mapping):
        raise ValueError("invalid public readiness summary")
    required = {
        "schema", "status", "source", "grouping_consistent", "split_patient_disjoint",
        "image_content_verified", "cross_source_overlap_checked", "model_trained",
        "patient_level_output_emitted", "counts", "label_support",
    }
    if set(summary) != required or summary.get("schema") != SCHEMA or summary.get("status") != STATUS or summary.get("source") != SOURCE:
        raise ValueError("invalid public readiness summary")
    for key, expected in {
        "grouping_consistent": True,
        "split_patient_disjoint": True,
        "image_content_verified": False,
        "cross_source_overlap_checked": False,
        "model_trained": False,
        "patient_level_output_emitted": False,
    }.items():
        if type(summary.get(key)) is not bool or summary[key] is not expected:
            raise ValueError("invalid public readiness summary")
    counts = summary["counts"]
    if not isinstance(counts, Mapping) or set(counts) != {"total", "eligible", "splits"}:
        raise ValueError("invalid public readiness summary")
    for key in ("total", "eligible"):
        group = counts[key]
        if not isinstance(group, Mapping) or set(group) != {"images", "people"} or not all(_valid_coarse(group[item]) for item in group):
            raise ValueError("invalid public readiness summary")
    splits = counts["splits"]
    if not isinstance(splits, Mapping) or set(splits) != set(_SPLIT_NAMES):
        raise ValueError("invalid public readiness summary")
    support = summary["label_support"]
    if not isinstance(support, Mapping) or set(support) != set(_SPLIT_NAMES):
        raise ValueError("invalid public readiness summary")
    for split_name in _SPLIT_NAMES:
        group = splits[split_name]
        if not isinstance(group, Mapping) or set(group) != {"images", "people"} or not all(_valid_coarse(group[item]) for item in group):
            raise ValueError("invalid public readiness summary")
        endpoint_support = support[split_name]
        if not isinstance(endpoint_support, Mapping) or set(endpoint_support) != set(ENDPOINTS):
            raise ValueError("invalid public readiness summary")
        for endpoint in ENDPOINTS:
            detail = endpoint_support[endpoint]
            if not isinstance(detail, Mapping) or set(detail) != {"supported", "positive_patients", "negative_patients"}:
                raise ValueError("invalid public readiness summary")
            if type(detail["supported"]) is not bool or not _valid_coarse(detail["positive_patients"]) or not _valid_coarse(detail["negative_patients"]):
                raise ValueError("invalid public readiness summary")
            if detail["supported"] is not (detail["positive_patients"] != "<20" and detail["negative_patients"] != "<20" and detail["positive_patients"] >= 20 and detail["negative_patients"] >= 20):
                raise ValueError("invalid public readiness summary")
