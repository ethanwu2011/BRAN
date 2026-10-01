"""Private source-row prior-admission disease membership assembly.

The assembler operates only on caller-projected MIMIC rows.  It constructs a
retrospective administrative phenotype from diagnoses attached to completed
prior admissions; it does not infer disease absence and it does not use
diagnosis sequence numbers, outcomes, or other clinical labels.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
import re
import unicodedata

import numpy as np

from bran_cbc_event_adapter_v1 import _naive_iso_datetime
from bran_clinical_source_reader_v1 import _numeric_key


FAMILIES = (
    "type2_or_unspecified_diabetes",
    "heart_failure",
    "chronic_kidney_disease",
)
_FAMILY_INDEX = {family: index for index, family in enumerate(FAMILIES)}
_TARGET_KEYS = {"person", "episode", "row_binding"}
_ADMISSION_KEYS = {"subject_id", "hadm_id", "admittime", "dischtime"}
_DIAGNOSIS_KEYS = {"subject_id", "hadm_id", "icd_code", "icd_version"}
_HEX = frozenset("0123456789abcdef")
_ICD9_NUMERIC = re.compile(r"[0-9]{3,5}")
_ICD9_V = re.compile(r"V[0-9]{2,4}")
_ICD9_E = re.compile(r"E[0-9]{3,4}")
_ICD10 = re.compile(r"[A-Z][0-9][A-Z0-9]{1,5}")


def _fail(message: str) -> None:
    # Never include row contents, identifiers, or arbitrary parser text in a
    # failure.  This keeps source/patient material out of exception strings.
    raise ValueError(message) from None


def canonical_icd(version: object, code: object) -> tuple[str, str] | None:
    """Canonicalize one ICD-9/10 code without assigning a disease family."""

    if type(version) is not str or version not in ("9", "10") or type(code) is not str:
        return None
    # Only ordinary outer spaces are trim-authorized.  Tabs/newlines and all
    # other control or whitespace characters remain invalid.
    token = code.strip(" ")
    if not token or not token.isascii():
        return None
    if any(character.isspace() or unicodedata.category(character).startswith("C") for character in token):
        return None
    token = token.upper()
    dots = token.count(".")
    if dots > 1:
        return None
    if dots == 1:
        dot_index = token.find(".")
        # Numeric/V ICD-9 and ICD-10 use the three-character stem; ICD-9
        # external-cause E codes use their four-character stem.
        expected_dot_index = 4 if version == "9" and token.startswith("E") else 3
        if dot_index != expected_dot_index or dot_index == len(token) - 1:
            return None
        token = token.replace(".", "", 1)
    if version == "9":
        valid = (
            _ICD9_NUMERIC.fullmatch(token) is not None
            or _ICD9_V.fullmatch(token) is not None
            or _ICD9_E.fullmatch(token) is not None
        )
    else:
        valid = _ICD10.fullmatch(token) is not None
    return (version, token) if valid else None


def _validate_approved_codes(approved_codes: object) -> dict[tuple[str, str], str]:
    if not isinstance(approved_codes, Mapping):
        _fail("invalid approved ICD mapping")
    try:
        if not approved_codes:
            _fail("invalid approved ICD mapping")
        items = list(approved_codes.items())
    except Exception:
        _fail("invalid approved ICD mapping")
    normalized: dict[tuple[str, str], str] = {}
    for key, family in items:
        if type(key) is not tuple or len(key) != 2:
            _fail("invalid approved ICD mapping")
        canonical = canonical_icd(key[0], key[1])
        if canonical is None or canonical != key:
            _fail("invalid approved ICD mapping")
        if type(family) is not str or family not in _FAMILY_INDEX:
            _fail("invalid approved ICD mapping")
        normalized[canonical] = family
    if set(normalized.values()) != set(FAMILIES):
        _fail("invalid approved ICD mapping")
    return normalized


def _close(iterator: object) -> None:
    try:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    except Exception:
        _fail("source iterator close failed")


def _iter_rows(rows: object, message: str) -> tuple[object, object]:
    if isinstance(rows, (str, bytes, bytearray, Mapping)):
        _fail(message)
    try:
        iterator = iter(rows)
    except Exception:
        _fail(message)
    return iterator, iterator


def _exact_row(row: object, keys: set[str], message: str) -> Mapping[str, object]:
    if not isinstance(row, Mapping):
        _fail(message)
    try:
        if set(row) != keys:
            _fail(message)
    except Exception:
        _fail(message)
    return row


def _canonical_id(value: object, *, require_string: bool = False) -> str:
    try:
        canonical = _numeric_key(value)
    except Exception:
        _fail("invalid source identifier")
    if require_string and (type(value) is not str or value != canonical):
        _fail("noncanonical target identifier")
    return canonical


def _binding(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or not value.isascii()
        or any(character not in _HEX for character in value)
    ):
        _fail("invalid target row binding")
    return value


def _materialize_admissions(admission_rows: object) -> dict[str, tuple[str, datetime | None, datetime | None]]:
    iterator, _ = _iter_rows(admission_rows, "admission rows must be iterable")
    admissions: dict[str, tuple[str, datetime | None, datetime | None]] = {}
    try:
        while True:
            try:
                row = next(iterator)
            except StopIteration:
                break
            row = _exact_row(row, _ADMISSION_KEYS, "invalid admission row")
            episode = _canonical_id(row["hadm_id"])
            person = _canonical_id(row["subject_id"])
            if episode in admissions:
                _fail("duplicate admission episode")
            admit = _naive_iso_datetime(row["admittime"])
            discharge = _naive_iso_datetime(row["dischtime"])
            admissions[episode] = (person, admit, discharge)
    except Exception:
        _fail("invalid admission rows")
    finally:
        _close(iterator)
    return admissions


def _validate_targets(target_rows: object) -> list[tuple[str, str, str]]:
    if type(target_rows) is not list or not target_rows:
        _fail("target rows must be a nonempty list")
    targets: list[tuple[str, str, str]] = []
    episodes: set[str] = set()
    bindings: set[str] = set()
    for row in target_rows:
        row = _exact_row(row, _TARGET_KEYS, "invalid target row")
        person = _canonical_id(row["person"], require_string=True)
        episode = _canonical_id(row["episode"], require_string=True)
        binding = _binding(row["row_binding"])
        if episode in episodes or binding in bindings:
            _fail("duplicate target row")
        episodes.add(episode)
        bindings.add(binding)
        targets.append((person, episode, binding))
    return targets


def _materialize_diagnoses(
    diagnosis_rows: object,
    admissions: Mapping[str, tuple[str, datetime | None, datetime | None]],
    approved_codes: Mapping[tuple[str, str], str],
) -> dict[str, frozenset[str]]:
    iterator, _ = _iter_rows(diagnosis_rows, "diagnosis rows must be iterable")
    by_episode: dict[str, set[str]] = defaultdict(set)
    try:
        while True:
            try:
                row = next(iterator)
            except StopIteration:
                break
            row = _exact_row(row, _DIAGNOSIS_KEYS, "invalid diagnosis row")
            episode = _canonical_id(row["hadm_id"])
            person = _canonical_id(row["subject_id"])
            master = admissions.get(episode)
            if master is None or master[0] != person:
                _fail("diagnosis row does not match admission")
            normalized = canonical_icd(row["icd_version"], row["icd_code"])
            if normalized is None:
                continue
            family = approved_codes.get(normalized)
            if family is not None:
                by_episode[episode].add(family)
    except Exception:
        _fail("invalid diagnosis rows")
    finally:
        _close(iterator)
    return {episode: frozenset(families) for episode, families in by_episode.items()}


def _assemble(
    targets: list[tuple[str, str, str]],
    admissions: Mapping[str, tuple[str, datetime | None, datetime | None]],
    diagnoses: Mapping[str, frozenset[str]],
) -> tuple[np.ndarray, np.ndarray]:
    target_admits: list[datetime] = []
    for person, episode, _ in targets:
        master = admissions.get(episode)
        if master is None or master[0] != person or master[1] is None:
            _fail("target admission is not authenticated")
        target_admits.append(master[1])

    # Retain only complete, chronologically coherent admission intervals.
    # Sorting by discharge permits one monotone per-person sweep across targets
    # rather than an all-pairs target/admission join.
    by_person: dict[str, list[tuple[datetime, datetime, str, frozenset[str]]]] = defaultdict(list)
    for episode, (person, admit, discharge) in admissions.items():
        if admit is None or discharge is None or discharge < admit:
            continue
        by_person[person].append((discharge, admit, episode, diagnoses.get(episode, frozenset())))
    for person in by_person:
        by_person[person].sort(key=lambda item: (item[0], item[1], item[2]))

    target_indices: dict[str, list[tuple[datetime, int]]] = defaultdict(list)
    for index, (person, _, _) in enumerate(targets):
        target_indices[person].append((target_admits[index], index))

    membership = np.zeros((len(targets), len(FAMILIES)), dtype=bool)
    has_prior = np.zeros(len(targets), dtype=bool)
    for person, grouped_targets in target_indices.items():
        grouped_targets.sort(key=lambda item: (item[0], item[1]))
        candidates = by_person.get(person, [])
        discharge_times = [candidate[0] for candidate in candidates]
        pointer = 0
        prior_families: set[str] = set()
        for target_admit, target_index in grouped_targets:
            cutoff = bisect_left(discharge_times, target_admit)
            while pointer < cutoff:
                prior_families.update(candidates[pointer][3])
                pointer += 1
            has_prior[target_index] = pointer > 0
            for family in prior_families:
                membership[target_index, _FAMILY_INDEX[family]] = True
    return membership, has_prior


def assemble_prior_membership(
    target_rows: list[Mapping[str, object]],
    admission_rows: Iterable[Mapping[str, object]],
    diagnosis_rows: Iterable[Mapping[str, object]],
    approved_codes: Mapping[tuple[str, str], str],
) -> dict[str, np.ndarray]:
    """Assemble prior-admission family membership in authenticated target order."""

    approved = _validate_approved_codes(approved_codes)
    targets = _validate_targets(target_rows)
    admissions = _materialize_admissions(admission_rows)
    diagnoses = _materialize_diagnoses(diagnosis_rows, admissions, approved)
    membership, has_prior = _assemble(targets, admissions, diagnoses)
    row_binding = np.array([binding for _, _, binding in targets], dtype="U64")
    result = {
        "row_binding": row_binding,
        "prior_membership": membership,
        "has_prior_admission": has_prior,
    }
    for array in result.values():
        array.setflags(write=False)
    return result
