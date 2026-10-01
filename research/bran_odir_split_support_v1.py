"""Deterministic ODIR-local patient split and disclosure-safe label support."""
from __future__ import annotations

import hashlib

import numpy as np


_ERROR = "odir split support failed"
LABELS = ("N", "D", "G", "C", "A", "H", "M", "O")
SPLITS = ("train", "validation", "test")
_FLAGS = {
    "patient_level_output_emitted": False,
    "official_split_authenticated": False,
    "training_admitted": False,
    "systemic_disease_validation": False,
}
_KEYS = frozenset({"schema", "status", "split_counts_rounded_down20", "labels", *_FLAGS})


def _fail() -> None:
    raise ValueError(_ERROR) from None


def _coarse(value: int) -> int:
    return (value // 20) * 20


def _count(value: object) -> bool:
    return type(value) is int and value >= 0 and value % 20 == 0


def _split(patient_id: str) -> int:
    digest = hashlib.sha256(("bran-odir-source-split-v1\0" + patient_id).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") % 10000
    return 0 if value < 8000 else (1 if value < 9000 else 2)


def _validate_inputs(patient_ids: object, labels: object, eye_counts: object) -> tuple[list[str], np.ndarray, np.ndarray]:
    if (type(patient_ids) is not list or not patient_ids
            or any(type(value) is not str or not value for value in patient_ids)
            or len(set(patient_ids)) != len(patient_ids)):
        _fail()
    n_people = len(patient_ids)
    if type(labels) is not np.ndarray or labels.dtype != np.dtype(np.float64) or labels.shape != (n_people, 8):
        _fail()
    if not np.all(np.isnan(labels) | (labels == 0.0) | (labels == 1.0)):
        _fail()
    if (type(eye_counts) is not np.ndarray or eye_counts.dtype != np.dtype(np.int64)
            or eye_counts.shape != (n_people,) or not np.all((eye_counts == 1) | (eye_counts == 2))):
        _fail()
    return patient_ids, labels, eye_counts


def validate_aggregate(value: object) -> None:
    """Validate only the closed public aggregate, not a private split array."""
    if type(value) is not dict or set(value) != _KEYS:
        _fail()
    if value.get("schema") != "bran-odir-split-support-v1" or value.get("status") != "qualified":
        _fail()
    if any(value.get(key) is not expected for key, expected in _FLAGS.items()):
        _fail()
    split_counts = value.get("split_counts_rounded_down20")
    if type(split_counts) is not dict or set(split_counts) != set(SPLITS):
        _fail()
    for split in SPLITS:
        item = split_counts[split]
        if type(item) is not dict or set(item) != {"groups", "images"}:
            _fail()
        groups, images = item["groups"], item["images"]
        if not _count(groups) or not _count(images) or images < groups or images > 2 * groups + 20:
            _fail()
    labels = value.get("labels")
    if type(labels) is not dict or set(labels) != set(LABELS):
        _fail()
    for label in LABELS:
        item = labels[label]
        if type(item) is not dict or set(item) != {*SPLITS, "primary_evaluable"}:
            _fail()
        if type(item["primary_evaluable"]) is not bool:
            _fail()
        supported = []
        for split in SPLITS:
            support = item[split]
            if type(support) is not dict or set(support) != {"status", "positive", "negative"}:
                _fail()
            status, positive, negative = support["status"], support["positive"], support["negative"]
            groups = split_counts[split]["groups"]
            if status == "supported":
                if (not _count(positive) or not _count(negative) or positive < 20 or negative < 20
                        or positive + negative > groups):
                    _fail()
                supported.append(True)
            elif status == "unsupported":
                if positive is not None or negative is not None:
                    _fail()
                supported.append(False)
            else:
                _fail()
        if item["primary_evaluable"] is not all(supported):
            _fail()


def qualify(
    patient_ids: list[str], labels: np.ndarray, eye_counts: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return a private fixed split and a closed aggregate; never repartition."""
    patient_ids, labels, eye_counts = _validate_inputs(patient_ids, labels, eye_counts)
    split = np.asarray([_split(patient_id) for patient_id in patient_ids], dtype=np.uint8)
    split.setflags(write=False)
    split_counts: dict[str, dict[str, int]] = {}
    for index, name in enumerate(SPLITS):
        selected = split == index
        split_counts[name] = {
            "groups": _coarse(int(selected.sum())),
            "images": _coarse(int(eye_counts[selected].sum())),
        }
    aggregate_labels: dict[str, dict[str, object]] = {}
    for column, label in enumerate(LABELS):
        item: dict[str, object] = {}
        statuses = []
        for index, name in enumerate(SPLITS):
            values = labels[split == index, column]
            positive, negative = int((values == 1.0).sum()), int((values == 0.0).sum())
            if positive >= 20 and negative >= 20:
                item[name] = {"status": "supported", "positive": _coarse(positive), "negative": _coarse(negative)}
                statuses.append(True)
            else:
                item[name] = {"status": "unsupported", "positive": None, "negative": None}
                statuses.append(False)
        item["primary_evaluable"] = all(statuses)
        aggregate_labels[label] = item
    aggregate: dict[str, object] = {
        "schema": "bran-odir-split-support-v1", "status": "qualified",
        "split_counts_rounded_down20": split_counts, "labels": aggregate_labels, **_FLAGS,
    }
    validate_aggregate(aggregate)
    return split, aggregate


__all__ = ["LABELS", "SPLITS", "qualify", "validate_aggregate"]
