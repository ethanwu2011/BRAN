"""Outcome-blind, PSU-grouped KNHANES 2022/2023 fold assignment.

This local array kernel has no source I/O, outcomes, weights, target support,
or model code.  It reserves five adaptation/evaluation partitions before any
readout sees hemoglobin.  Returned arrays are private/read-only and its receipt
contains counts only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any

import numpy as np


FOLDS = 5
YEARS = (2022, 2023)
SALT = "bran-knhanes-v5-psu-folds-2026-09-20"
_ERROR = "knhanes_grouped_folds_contract_failed"


def _require(ok: bool) -> None:
    if not ok:
        raise ValueError(_ERROR)


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value).copy()
    output.setflags(write=False)
    return output


@dataclass(frozen=True, repr=False)
class GroupedKNHANESFolds:
    """Private local grouping output; never serialize or print row assignments."""

    folds: np.ndarray
    psu_groups: np.ndarray
    stratum_groups: np.ndarray
    receipt: MappingProxyType

    def __repr__(self) -> str:
        return "<PrivateGroupedKNHANESFolds>"

    def __reduce__(self):
        raise TypeError(_ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(_ERROR)


def _vector(value: Any, rows: int | None = None) -> np.ndarray:
    _require(isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype.kind in "iufUS")
    if rows is not None:
        _require(value.shape == (rows,))
    _require(value.shape[0] > 0)
    return value


def _atom(value: Any) -> tuple[str, str]:
    """Typed, nonmissing scalar key; type tags avoid ``1``/``'1'`` collisions."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        _require(False)
    if isinstance(value, str):
        _require(bool(value.strip()))
        return ("str", value)
    if isinstance(value, int):
        return ("int", str(value))
    if isinstance(value, float):
        _require(math.isfinite(value))
        return ("float", value.hex())
    _require(False)
    raise AssertionError


def _digest(*parts: object) -> bytes:
    payload = "|".join(repr(part) for part in (SALT,) + parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


def assign(year: Any, person_id: Any, psu: Any, stratum: Any, household: Any) -> GroupedKNHANESFolds:
    """Assign five folds while keeping every namespaced PSU wholly together.

    ``year`` must contain only the admitted 2022/2023 releases.  The function
    intentionally has no target, target-mask, score, or weight argument.
    """

    try:
        years = _vector(year)
        rows = years.shape[0]
        people, psus, strata, households = (_vector(value, rows) for value in (person_id, psu, stratum, household))
        _require(years.dtype.kind in "iu" and set(int(item) for item in np.unique(years)) <= set(YEARS))
        _require(set(int(item) for item in np.unique(years)) == set(YEARS))
        person_keys: set[tuple[int, tuple[str, str]]] = set()
        household_psu: dict[tuple[int, tuple[str, str]], tuple[str, str]] = {}
        psu_stratum: dict[tuple[int, tuple[str, str]], tuple[str, str]] = {}
        psu_rows: dict[tuple[int, tuple[str, str]], list[int]] = {}
        psu_order: dict[tuple[int, tuple[str, str]], bytes] = {}
        stratum_psus: dict[tuple[int, tuple[str, str]], list[tuple[int, tuple[str, str]]]] = {}
        for index in range(rows):
            local_year = int(years[index])
            person_key, psu_key, stratum_key, household_key = (_atom(value[index]) for value in (people, psus, strata, households))
            full_person = (local_year, person_key)
            full_psu = (local_year, psu_key)
            full_stratum = (local_year, stratum_key)
            full_household = (local_year, household_key)
            _require(full_person not in person_keys)
            person_keys.add(full_person)
            if full_household in household_psu:
                _require(household_psu[full_household] == psu_key)
            else:
                household_psu[full_household] = psu_key
            if full_psu in psu_stratum:
                _require(psu_stratum[full_psu] == stratum_key)
            else:
                psu_stratum[full_psu] = stratum_key
                stratum_psus.setdefault(full_stratum, []).append(full_psu)
                psu_order[full_psu] = _digest("psu", local_year, stratum_key, psu_key)
            psu_rows.setdefault(full_psu, []).append(index)
        _require(len(psu_rows) >= FOLDS)
        # Stable salt-hashed stratum order plus a cumulative deterministic
        # offset makes the concatenated PSU sequence cover every fold whenever
        # at least five PSUs exist.  Within each stratum, PSUs stay hash-sorted
        # and round-robin; no outcome can affect this allocation.
        ordered_strata = sorted(stratum_psus, key=lambda key: _digest("stratum", key[0], key[1]))
        assigned: dict[tuple[int, tuple[str, str]], int] = {}
        cursor = 0
        for full_stratum in ordered_strata:
            members = sorted(stratum_psus[full_stratum], key=lambda key: psu_order[key])
            offset = cursor % FOLDS
            for rank, full_psu in enumerate(members):
                assigned[full_psu] = (offset + rank) % FOLDS
            cursor += len(members)
        folds = np.empty(rows, dtype=np.int64)
        group = np.empty(rows, dtype=np.int64)
        stratum_group = np.empty(rows, dtype=np.int64)
        stratum_ids = {key: index for index, key in enumerate(ordered_strata)}
        for group_index, full_psu in enumerate(sorted(psu_rows, key=lambda key: psu_order[key])):
            positions = np.asarray(psu_rows[full_psu], dtype=np.int64)
            folds[positions] = assigned[full_psu]
            group[positions] = group_index
            stratum_group[positions] = stratum_ids[(full_psu[0], psu_stratum[full_psu])]
        participant_counts = tuple(int(np.count_nonzero(folds == fold)) for fold in range(FOLDS))
        psu_counts = tuple(int(sum(value == fold for value in assigned.values())) for fold in range(FOLDS))
        _require(all(count > 0 for count in psu_counts))
        receipt = MappingProxyType({
            "schema": "bran-knhanes-grouped-folds-v1", "years": YEARS, "fold_count": FOLDS,
            "participant_count": int(rows), "psu_count": int(len(psu_rows)), "stratum_count": int(len(stratum_psus)),
            "household_count": int(len(household_psu)), "participants_per_fold": participant_counts,
            "psus_per_fold": psu_counts, "assignment_salt": SALT, "outcome_blind": True,
        })
        return GroupedKNHANESFolds(_readonly(folds), _readonly(group), _readonly(stratum_group), receipt)
    except (TypeError, ValueError, OverflowError, KeyError, IndexError):
        raise ValueError(_ERROR) from None


assign_grouped_folds = assign
