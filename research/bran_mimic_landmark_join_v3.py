"""Pure, source-free MIMIC landmark/outcome join for the V3 selector.

The caller supplies already authenticated linked-snapshot target rows and the
row-aligned administrative membership object.  This module joins admission
outcomes by both canonical person and episode keys, converts naive admission
times to epoch minutes, preserves a safe unknown endpoint when chronology proves
observation beyond 24 hours, and delegates index/disease/outcome selection to
the frozen clinical V3 contract.  It performs no source I/O, fitting, or
outcome inference beyond the documented chronology checks.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Integral

import numpy as np

from bran_cbc_event_adapter_v1 import _naive_iso_datetime
from bran_clinical_discovery_landmark_v3 import (
    FAMILIES,
    LANDMARK_MINUTES,
    OUTCOME_DEATH,
    OUTCOME_SURVIVING_DISCHARGE,
    OUTCOME_UNKNOWN,
    ClinicalDiscoveryLandmarkV3,
    select_landmark_partition,
)
from bran_mimic_landmark_source_v1 import (
    OUTCOME_COLUMNS,
    classify_landmark_outcome,
)
from bran_mimic_retrospective_membership_v2 import RetrospectiveMembershipV2


ERROR = "bran_mimic_landmark_join_v3_contract_failed"
_TARGET_COLUMNS = frozenset(("person", "episode", "row_binding"))
_EPOCH = datetime(1970, 1, 1)
_STATUS_BY_SOURCE = {
    "eligible_death": OUTCOME_DEATH,
    "eligible_survival": OUTCOME_SURVIVING_DISCHARGE,
}


def _fail() -> None:
    raise ValueError(ERROR) from None


def _require(condition: bool) -> None:
    if not condition:
        _fail()


def _canonical_id(value: object, *, strict: bool) -> str | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, Integral):
        integer = int(value)
        if integer < 0:
            return None
        result = str(integer)
        return result if len(result) <= 64 else None
    if not isinstance(value, str) or not value or not value.isascii() or not value.isdigit():
        return None
    if strict and len(value) > 1 and value[0] == "0":
        return None
    result = value.lstrip("0") or "0"
    return result if len(result) <= 64 else None


def _binding(value: object) -> str:
    _require(type(value) is str and len(value) == 64 and value.isascii())
    _require(all(character in "0123456789abcdef" for character in value))
    return value


def _readonly(value: object, dtype: np.dtype | str) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, repr=False)
class PrivateMimicLandmarkJoinV3:
    """Private row-aligned chronology, membership, and landmark selection.

    ``admission_time_minutes`` is the absolute naive-datetime epoch-minute
    chronology used for earliest-person selection.  ``landmark_end_minutes``
    and ``outcome_time_minutes`` are relative minutes from each admission,
    matching the source classifier and clinical V3 selector.  The nested
    selection contains only readonly private arrays and no person identifiers.
    """

    row_binding: np.ndarray
    admission_time_minutes: np.ndarray
    landmark_end_minutes: np.ndarray
    outcome_status: np.ndarray
    outcome_time_minutes: np.ndarray
    current_membership: np.ndarray
    current_admission_eligible: np.ndarray
    selection: ClinicalDiscoveryLandmarkV3

    def __repr__(self) -> str:
        return "<PrivateMimicLandmarkJoinV3>"

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)


def _materialize_targets(target_rows: object) -> list[tuple[str, str, str]]:
    _require(not isinstance(target_rows, (str, bytes, bytearray, Mapping)))
    try:
        iterator = iter(target_rows)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        _fail()
    rows: list[tuple[str, str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_episodes: set[str] = set()
    seen_bindings: set[str] = set()
    try:
        for row in iterator:
            _require(isinstance(row, Mapping) and set(row) == _TARGET_COLUMNS)
            person = _canonical_id(row["person"], strict=True)
            episode = _canonical_id(row["episode"], strict=True)
            _require(person is not None and episode is not None)
            binding = _binding(row["row_binding"])
            pair = (person, episode)
            _require(pair not in seen_pairs and episode not in seen_episodes)
            _require(binding not in seen_bindings)
            seen_pairs.add(pair)
            seen_episodes.add(episode)
            seen_bindings.add(binding)
            rows.append((person, episode, binding))
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    _require(bool(rows))
    return rows


def _validate_membership(
    membership: object,
    targets: list[tuple[str, str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require(type(membership) is RetrospectiveMembershipV2)
    rows = len(targets)
    bindings = membership.row_binding
    current = membership.current_membership
    eligible = membership.current_admission_eligible
    _require(isinstance(bindings, np.ndarray) and bindings.dtype == np.dtype("U64") and bindings.shape == (rows,))
    _require(isinstance(current, np.ndarray) and current.dtype == np.dtype(bool)
             and current.shape == (rows, len(FAMILIES)))
    _require(isinstance(eligible, np.ndarray) and eligible.dtype == np.dtype(bool) and eligible.shape == (rows,))
    _require(type(membership.membership_known_at_24h) is bool)
    _require(type(membership.membership_is_clinically_adjudicated) is bool)
    _require(type(membership.absence_of_qualifying_code_proves_health) is bool)
    _require(not membership.membership_known_at_24h)
    _require(not membership.membership_is_clinically_adjudicated)
    _require(not membership.absence_of_qualifying_code_proves_health)
    target_bindings = np.array([binding for _person, _episode, binding in targets], dtype="U64")
    _require(np.array_equal(bindings, target_bindings))
    for value in bindings:
        _binding(str(value))
    return (
        np.array(current, dtype=bool, copy=True),
        np.array(eligible, dtype=bool, copy=True),
        np.array(bindings, dtype="U64", copy=True),
    )


def _absolute_epoch_minutes(value: object) -> float:
    parsed = _naive_iso_datetime(value)
    if parsed is None:
        return math.nan
    return (parsed - _EPOCH).total_seconds() / 60.0


def _unknown_endpoint(row: Mapping[str, object]) -> float:
    """Return only an independently proven endpoint for an unknown status.

    A valid death after 24 hours is an observation endpoint even when the
    mortality flag is inconsistent.  A valid discharge may establish the same
    fact only when the death field is empty or explicitly places a death after
    that discharge.  A documented early death is returned at its early time so
    a late discharge can never manufacture landmark eligibility.
    """
    start = _naive_iso_datetime(row["admittime"])
    if start is None:
        return math.nan
    discharge = _naive_iso_datetime(row["dischtime"])
    if discharge is not None and discharge < start:
        return math.nan
    discharge_minutes = math.nan if discharge is None else (discharge - start).total_seconds() / 60.0
    death_raw = row["deathtime"]
    flag = row["hospital_expire_flag"]
    if death_raw == "":
        if flag not in ("0", "1") and math.isfinite(discharge_minutes) and discharge_minutes > LANDMARK_MINUTES:
            return discharge_minutes
        return math.nan

    death = _naive_iso_datetime(death_raw)
    if death is None or death < start:
        return math.nan
    death_minutes = (death - start).total_seconds() / 60.0
    if death_minutes <= LANDMARK_MINUTES:
        return death_minutes
    if discharge is None:
        return death_minutes
    if death <= discharge:
        return death_minutes
    if discharge_minutes > LANDMARK_MINUTES:
        return discharge_minutes
    return math.nan


def _materialize_outcomes(
    outcome_rows: object,
    targets: list[tuple[str, str, str]],
) -> dict[str, tuple[float, np.int8, float]]:
    _require(not isinstance(outcome_rows, (str, bytes, bytearray, Mapping)))
    target_by_episode = {episode: person for person, episode, _binding in targets}
    joined: dict[str, tuple[float, np.int8, float]] = {}
    seen_episodes: set[str] = set()
    try:
        iterator = iter(outcome_rows)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        _fail()
    try:
        for row in iterator:
            _require(isinstance(row, Mapping) and set(row) == set(OUTCOME_COLUMNS))
            person = _canonical_id(row["subject_id"], strict=False)
            episode = _canonical_id(row["hadm_id"], strict=False)
            _require(person is not None and episode is not None)
            _require(episode not in seen_episodes)
            seen_episodes.add(episode)
            if episode not in target_by_episode:
                continue
            _require(target_by_episode[episode] == person)
            classified = classify_landmark_outcome(row)
            start_absolute = _absolute_epoch_minutes(row["admittime"])
            _require(math.isfinite(start_absolute))
            status = _STATUS_BY_SOURCE.get(classified.status, OUTCOME_UNKNOWN)
            if status != OUTCOME_UNKNOWN:
                _require(classified.end_minutes is not None and math.isfinite(classified.end_minutes))
                endpoint = float(classified.end_minutes)
            else:
                endpoint = _unknown_endpoint(row)
            outcome_time = endpoint if math.isfinite(endpoint) else math.nan
            joined[episode] = (start_absolute, np.int8(status), outcome_time)
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    _require(set(joined) == set(target_by_episode))
    return joined


def assemble_landmark_join(
    target_rows: Iterable[Mapping[str, object]],
    admission_rows: Iterable[Mapping[str, object]],
    current_membership: RetrospectiveMembershipV2,
) -> PrivateMimicLandmarkJoinV3:
    """Join authenticated target rows to outcomes and run the frozen V3 selector."""
    try:
        targets = _materialize_targets(target_rows)
        membership, membership_eligible, row_binding = _validate_membership(current_membership, targets)
        outcomes = _materialize_outcomes(admission_rows, targets)
        persons = np.array([person for person, _episode, _binding in targets], dtype="U64")
        episodes = np.array([episode for _person, episode, _binding in targets], dtype="U64")
        admission_time = np.array([outcomes[episode][0] for episode in episodes], dtype=np.float64)
        status = np.array([outcomes[episode][1] for episode in episodes], dtype=np.int8)
        endpoint = np.array([outcomes[episode][2] for episode in episodes], dtype=np.float64)

        # Administrative date eligibility is an ascertainment gate, never a
        # disease-negative label.  An ineligible membership row cannot supply
        # a landmark endpoint, but its row remains aligned for private audit.
        endpoint[~membership_eligible] = np.nan
        status[~membership_eligible] = OUTCOME_UNKNOWN
        outcome_time = endpoint.copy()

        selection = select_landmark_partition(
            persons,
            episodes,
            admission_time,
            endpoint,
            status,
            outcome_time,
            membership,
        )
        return PrivateMimicLandmarkJoinV3(
            _readonly(row_binding, "U64"),
            _readonly(admission_time, np.float64),
            _readonly(endpoint, np.float64),
            _readonly(status, np.int8),
            _readonly(outcome_time, np.float64),
            _readonly(membership, bool),
            _readonly(membership_eligible, bool),
            selection,
        )
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


__all__ = ["ERROR", "PrivateMimicLandmarkJoinV3", "assemble_landmark_join"]
