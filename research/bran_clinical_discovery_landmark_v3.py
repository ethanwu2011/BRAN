"""Pure clinically-led landmark selection and person-level V3 partitions.

The caller supplies already-typed local arrays from an authenticated source
layer.  This module does not read source files, map diagnoses, inspect model
states, fit clusters, or analyze outcomes.  It selects one earliest eligible
landmark index admission per person using only landmark ascertainment and
admission chronology, then assigns that fixed index-person pool once to fixed
60/20/20 discovery, characterization-development, and test roles.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


ERROR = "bran_clinical_discovery_landmark_v3_contract_failed"
LANDMARK_MINUTES = 24.0 * 60.0
FAMILIES = (
    "type2_or_unspecified_diabetes",
    "heart_failure",
    "chronic_kidney_disease",
)
OUTCOME_UNKNOWN = np.int8(0)
OUTCOME_DEATH = np.int8(1)
OUTCOME_SURVIVING_DISCHARGE = np.int8(2)
LABEL_SURVIVAL = np.int8(0)
LABEL_DEATH = np.int8(1)
LABEL_UNKNOWN = np.int8(-1)
ROLE_DISCOVERY = np.uint8(0)
ROLE_CHARACTERIZATION_DEVELOPMENT = np.uint8(1)
ROLE_TEST = np.uint8(2)
ROLE_UNASSIGNED = np.uint8(255)

# This salt is used only inside the local role-order calculation.  Person
# identities and digests are never returned, serialized, or put in errors.
_ROLE_SALT = b"bran-clinical-discovery-landmark-v3-fixed-role-salt\0"


@dataclass(frozen=True, slots=True, repr=False)
class ClinicalDiscoveryLandmarkV3:
    """Private row-aligned selection, labels, and person-level role masks.

    ``selected`` and ``labels`` have one column per family in ``FAMILIES``.
    ``index_admission`` marks the one earliest landmark index per person.
    ``roles`` is assigned on those index rows before disease/outcome filtering;
    ``ROLE_UNASSIGNED`` marks every other row.  Unknown/indeterminate index
    outcomes remain unselected with ``LABEL_UNKNOWN``.  No person key or person
    hash is retained in this result.
    """

    selected: np.ndarray
    index_admission: np.ndarray
    roles: np.ndarray
    labels: np.ndarray
    landmark_eligible: np.ndarray

    def __repr__(self) -> str:
        return "<ClinicalDiscoveryLandmarkV3>"

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)


class _Invalid(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _Invalid


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _array(value: object, dtype: str | np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    _require(type(value) is np.ndarray and value.dtype == np.dtype(dtype) and value.shape == shape)
    return value


def _validate_inputs(
    person_key: object,
    episode_key: object,
    admission_time_minutes: object,
    landmark_end_minutes: object,
    outcome_status: object,
    outcome_time_minutes: object,
    disease_membership: object,
) -> int:
    _require(type(person_key) is np.ndarray and person_key.ndim == 1 and person_key.dtype == np.dtype("U64"))
    rows = person_key.shape[0]
    _array(episode_key, "U64", (rows,))
    _array(admission_time_minutes, np.float64, (rows,))
    _array(landmark_end_minutes, np.float64, (rows,))
    _array(outcome_status, np.int8, (rows,))
    _array(outcome_time_minutes, np.float64, (rows,))
    _array(disease_membership, bool, (rows, len(FAMILIES)))
    _require(all(isinstance(value, str) and bool(value) for value in person_key))
    _require(all(isinstance(value, str) and bool(value) for value in episode_key))
    _require(len(set(episode_key)) == rows)
    _require(np.isfinite(admission_time_minutes).all())
    _require(not np.isinf(landmark_end_minutes).any() and not np.isinf(outcome_time_minutes).any())
    _require(np.isin(outcome_status, (OUTCOME_UNKNOWN, OUTCOME_DEATH, OUTCOME_SURVIVING_DISCHARGE)).all())
    return rows


def _landmark_eligibility(landmark_end_minutes: np.ndarray) -> np.ndarray:
    """Require explicit observation beyond 24h; outcome is not used here."""

    finite_end = np.isfinite(landmark_end_minutes)
    return finite_end & (landmark_end_minutes > LANDMARK_MINUTES)


def _choose_index_rows(
    person_key: np.ndarray,
    episode_key: np.ndarray,
    admission_time_minutes: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    chosen: dict[str, tuple[tuple[float, str], int]] = {}
    for index in np.flatnonzero(eligible):
        person = str(person_key[index])
        order = (float(admission_time_minutes[index]), str(episode_key[index]))
        if person not in chosen or order < chosen[person][0]:
            chosen[person] = (order, int(index))
    mask = np.zeros(person_key.shape[0], dtype=bool)
    for _order, index in chosen.values():
        mask[index] = True
    return mask


def _known_outcomes(
    landmark_end_minutes: np.ndarray,
    outcome_status: np.ndarray,
    outcome_time_minutes: np.ndarray,
) -> np.ndarray:
    """Return only documented, post-landmark outcomes for primary cohorts."""

    status_known = np.isin(outcome_status, (OUTCOME_DEATH, OUTCOME_SURVIVING_DISCHARGE))
    finite_time = np.isfinite(outcome_time_minutes)
    post_landmark = outcome_time_minutes > LANDMARK_MINUTES
    # A known event must be the same explicit endpoint that established the
    # index landmark.  Any mismatch remains indeterminate and is not inferred.
    return status_known & finite_time & post_landmark & (outcome_time_minutes == landmark_end_minutes)


def _person_roles(persons: set[str]) -> dict[str, np.uint8]:
    ordered = sorted(
        persons,
        key=lambda person: (hashlib.sha256(_ROLE_SALT + person.encode("utf-8")).digest(), person),
    )
    count = len(ordered)
    discovery = (3 * count) // 5
    development = count // 5
    roles: dict[str, np.uint8] = {}
    for index, person in enumerate(ordered):
        if index < discovery:
            role = ROLE_DISCOVERY
        elif index < discovery + development:
            role = ROLE_CHARACTERIZATION_DEVELOPMENT
        else:
            role = ROLE_TEST
        roles[person] = role
    return roles


def select_landmark_partition(
    person_key: np.ndarray,
    episode_key: np.ndarray,
    admission_time_minutes: np.ndarray,
    landmark_end_minutes: np.ndarray,
    outcome_status: np.ndarray,
    outcome_time_minutes: np.ndarray,
    disease_membership: np.ndarray,
) -> ClinicalDiscoveryLandmarkV3:
    """Select one outcome-blind landmark index per person and assign roles.

    ``landmark_end_minutes`` is the caller-attested last observed/in-hospital
    or index-outcome time relative to admission.  An index candidate qualifies
    only when that time is strictly after 24 hours.  The earliest candidate is
    fixed before disease membership or outcome filtering.  Known outcome codes
    must have a finite matching event time after the landmark; unknown or
    indeterminate outcomes remain on the index pool but are excluded from the
    primary selected masks with label ``-1``.

    One row per person is chosen by earliest admission time, with episode-key
    lexicographic order as the exact tie-break.  Roles are assigned once to the
    complete index-person pool by salted hash order: floor ``3*n/5`` discovery,
    then floor ``n/5`` characterization-development, and the remainder test.
    No family or outcome balancing is performed.
    """
    try:
        rows = _validate_inputs(
            person_key, episode_key, admission_time_minutes, landmark_end_minutes,
            outcome_status, outcome_time_minutes, disease_membership,
        )
        eligible = _landmark_eligibility(landmark_end_minutes)
        index_admission = _choose_index_rows(
            person_key, episode_key, admission_time_minutes, eligible
        )
        known_outcome = _known_outcomes(landmark_end_minutes, outcome_status, outcome_time_minutes)
        selected = index_admission[:, None] & disease_membership & known_outcome[:, None]

        index_persons = {str(value) for value in person_key[index_admission]}
        role_by_person = _person_roles(index_persons)
        roles = np.full(rows, ROLE_UNASSIGNED, dtype=np.uint8)
        for index in np.flatnonzero(index_admission):
            roles[index] = role_by_person[str(person_key[index])]

        labels = np.full((rows, len(FAMILIES)), LABEL_UNKNOWN, dtype=np.int8)
        death_rows = selected & (outcome_status[:, None] == OUTCOME_DEATH)
        survival_rows = selected & (outcome_status[:, None] == OUTCOME_SURVIVING_DISCHARGE)
        labels[death_rows] = LABEL_DEATH
        labels[survival_rows] = LABEL_SURVIVAL

        _require(not np.any(index_admission & ~eligible))
        _require(not np.any(selected & ~index_admission[:, None]))
        _require(np.all(roles[index_admission] != ROLE_UNASSIGNED))
        return ClinicalDiscoveryLandmarkV3(
            _readonly(selected), _readonly(index_admission), _readonly(roles), _readonly(labels),
            _readonly(eligible)
        )
    except _Invalid:
        raise ValueError(ERROR) from None
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


__all__ = [
    "ERROR", "FAMILIES", "LANDMARK_MINUTES", "LABEL_DEATH", "LABEL_SURVIVAL", "LABEL_UNKNOWN",
    "OUTCOME_DEATH", "OUTCOME_SURVIVING_DISCHARGE", "OUTCOME_UNKNOWN", "ROLE_CHARACTERIZATION_DEVELOPMENT",
    "ROLE_DISCOVERY", "ROLE_TEST", "ROLE_UNASSIGNED", "ClinicalDiscoveryLandmarkV3",
    "select_landmark_partition",
]
