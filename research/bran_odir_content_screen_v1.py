"""Private, in-memory ODIR content-screen and quarantine policy.

This module accepts only caller-local hash arrays.  It neither authenticates
their provenance nor publishes an image, identifier, hash, or match index.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from bran_retinal_hash_match_v1 import PrivateHashIndex


_ERROR = "odir content screen failed"
_HASH_KEYS = frozenset({"decoded_sha256", "phash", "dhash"})
_REFERENCE_KEYS = frozenset({"ai_readi", "brset"})
_FLAG_KEYS = frozenset({
    "within_odir_direct", "ai_readi_direct", "brset_direct",
    "same_person_exact_duplicate", "quarantined", "retained_representative",
})
_COUNT_KEYS = (
    "input_images", "input_people", "within_odir_direct_images", "ai_readi_direct_images",
    "brset_direct_images", "same_person_exact_duplicate_images", "quarantined_images",
    "quarantined_people", "retained_images", "retained_people",
)
_FLAGS = {
    "training_admitted": False,
    "cross_study_identity_proven": False,
    "absence_of_all_duplicates_proven": False,
    "patient_level_output_emitted": False,
}
_SUMMARY_KEYS = frozenset({"schema", "status", "counts_rounded_down20", *_FLAGS})


def _fail() -> None:
    raise ValueError(_ERROR) from None


def _array(value: object, dtype: np.dtype, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != dtype or (shape is not None and value.shape != shape):
        _fail()
    return value


def _bundle(value: object, *, required_nonempty: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if type(value) is not dict or set(value) != _HASH_KEYS:
        _fail()
    decoded = _array(value["decoded_sha256"], np.dtype(np.uint8))
    if decoded.ndim != 2 or decoded.shape[1:] != (32,) or (required_nonempty and len(decoded) == 0):
        _fail()
    n_rows = len(decoded)
    phash = _array(value["phash"], np.dtype(np.uint64), (n_rows,))
    dhash = _array(value["dhash"], np.dtype(np.uint64), (n_rows,))
    return decoded, phash, dhash


class PrivateFlags(dict[str, np.ndarray]):
    """Private arrays with a representation that cannot expose their values."""

    def __repr__(self) -> str:
        return "<PrivateFlags private>"


def validate_flags(value: object) -> None:
    """Validate private flag shape/dtype without exposing values."""
    if not isinstance(value, dict) or set(value) != _FLAG_KEYS:
        _fail()
    arrays = list(value.values())
    if not arrays or any(type(item) is not np.ndarray or item.dtype != np.dtype(bool) or item.ndim != 1
                         for item in arrays):
        _fail()
    n_rows = len(arrays[0])
    if n_rows == 0 or any(item.shape != (n_rows,) for item in arrays):
        _fail()
    quarantined = value["quarantined"]
    duplicate = value["same_person_exact_duplicate"]
    retained = value["retained_representative"]
    if not np.array_equal(retained, ~quarantined & ~duplicate):
        _fail()
    for name in ("within_odir_direct", "ai_readi_direct", "brset_direct"):
        if np.any(value[name] & ~quarantined):
            _fail()


def _count_valid(value: object) -> bool:
    return type(value) is int and value >= 0 and value % 20 == 0


def validate_summary(value: object) -> None:
    """Validate the closed public summary, never a match-level result."""
    if type(value) is not dict or set(value) != _SUMMARY_KEYS:
        _fail()
    if value.get("schema") != "bran-odir-content-screen-v1" or value.get("status") != "screened":
        _fail()
    if any(value.get(key) is not expected for key, expected in _FLAGS.items()):
        _fail()
    counts = value.get("counts_rounded_down20")
    if type(counts) is not dict or set(counts) != set(_COUNT_KEYS):
        _fail()
    if any(not _count_valid(counts[key]) for key in _COUNT_KEYS):
        _fail()
    images, people = counts["input_images"], counts["input_people"]
    if people > images:
        _fail()
    image_counts = (
        "within_odir_direct_images", "ai_readi_direct_images", "brset_direct_images",
        "same_person_exact_duplicate_images", "quarantined_images", "retained_images",
    )
    if any(counts[key] > images for key in image_counts):
        _fail()
    if counts["quarantined_people"] > people or counts["retained_people"] > people:
        _fail()
    if counts["quarantined_people"] > counts["quarantined_images"]:
        _fail()
    if counts["retained_people"] > counts["retained_images"]:
        _fail()
    if counts["quarantined_images"] + counts["retained_images"] > images:
        _fail()
    if counts["quarantined_people"] + counts["retained_people"] > people:
        _fail()


def _coarse(value: int) -> int:
    return (value // 20) * 20


def _private_bool(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.bool_, copy=True)
    result.setflags(write=False)
    return result


def screen(
    patient_ids: list[str], odir: dict[str, np.ndarray], references: dict[str, dict],
) -> tuple[PrivateFlags, dict[str, object]]:
    """Screen private ODIR hash arrays under the fixed exact-or-radius-four rule."""
    if type(patient_ids) is not list or not patient_ids or any(type(value) is not str or not value for value in patient_ids):
        _fail()
    decoded, phash, dhash = _bundle(odir, required_nonempty=True)
    n_rows = len(decoded)
    if len(patient_ids) != n_rows:
        _fail()
    if type(references) is not dict or set(references) != _REFERENCE_KEYS:
        _fail()
    reference_arrays = {name: _bundle(references[name], required_nonempty=True) for name in _REFERENCE_KEYS}

    within = np.zeros(n_rows, dtype=np.bool_)
    ai_direct = np.zeros(n_rows, dtype=np.bool_)
    brset_direct = np.zeros(n_rows, dtype=np.bool_)
    duplicate = np.zeros(n_rows, dtype=np.bool_)
    quarantined_people: set[str] = set()

    own_index = PrivateHashIndex(decoded, phash, dhash)
    for index in range(n_rows):
        matches = own_index.match(bytes(decoded[index]), int(phash[index]), int(dhash[index]))
        for other in matches:
            if other == index:
                continue
            if patient_ids[index] != patient_ids[other]:
                within[index] = True
                within[other] = True
                quarantined_people.add(patient_ids[index])
                quarantined_people.add(patient_ids[other])

    digest_people: dict[tuple[str, bytes], list[int]] = defaultdict(list)
    for index, patient_id in enumerate(patient_ids):
        digest_people[(patient_id, bytes(decoded[index]))].append(index)
    for indexes in digest_people.values():
        if len(indexes) > 1:
            for index in indexes[1:]:
                duplicate[index] = True

    for source, (reference_decoded, reference_phash, reference_dhash) in reference_arrays.items():
        reference_index = PrivateHashIndex(reference_decoded, reference_phash, reference_dhash)
        direct = ai_direct if source == "ai_readi" else brset_direct
        for index in range(n_rows):
            if reference_index.match(bytes(decoded[index]), int(phash[index]), int(dhash[index])):
                direct[index] = True
                quarantined_people.add(patient_ids[index])

    quarantined = np.asarray([patient_id in quarantined_people for patient_id in patient_ids], dtype=np.bool_)
    retained = ~quarantined & ~duplicate
    flags = PrivateFlags({
        "within_odir_direct": _private_bool(within),
        "ai_readi_direct": _private_bool(ai_direct),
        "brset_direct": _private_bool(brset_direct),
        "same_person_exact_duplicate": _private_bool(duplicate),
        "quarantined": _private_bool(quarantined),
        "retained_representative": _private_bool(retained),
    })
    validate_flags(flags)
    retained_people = {patient_ids[index] for index in np.flatnonzero(retained)}
    counts = {
        "input_images": n_rows,
        "input_people": len(set(patient_ids)),
        "within_odir_direct_images": int(within.sum()),
        "ai_readi_direct_images": int(ai_direct.sum()),
        "brset_direct_images": int(brset_direct.sum()),
        "same_person_exact_duplicate_images": int(duplicate.sum()),
        "quarantined_images": int(quarantined.sum()),
        "quarantined_people": len(quarantined_people),
        "retained_images": int(retained.sum()),
        "retained_people": len(retained_people),
    }
    summary: dict[str, object] = {
        "schema": "bran-odir-content-screen-v1", "status": "screened",
        "counts_rounded_down20": {key: _coarse(value) for key, value in counts.items()}, **_FLAGS,
    }
    validate_summary(summary)
    return flags, summary


__all__ = ["PrivateFlags", "screen", "validate_flags", "validate_summary"]
