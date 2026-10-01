"""Index-admission, discharge-coded MIMIC disease membership assembly.

This pure in-memory adapter reuses the locked MIMIC ICD vocabulary semantics.
It creates neither a prospective label nor an adjudicated diagnosis: codes are
retrospective and absence of a qualifying code is not evidence of health.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import bran_mimic_prior_disease_v1 as prior


ERROR = "bran_mimic_retrospective_membership_v2_contract_failed"
FAMILIES = prior.FAMILIES


def _fail() -> None:
    raise ValueError(ERROR) from None


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, repr=False)
class RetrospectiveMembershipV2:
    """Private row-aligned administrative membership without clinical claims."""

    row_binding: np.ndarray = field(repr=False)
    current_membership: np.ndarray = field(repr=False)
    current_admission_eligible: np.ndarray = field(repr=False)
    membership_known_at_24h: bool = False
    membership_is_clinically_adjudicated: bool = False
    absence_of_qualifying_code_proves_health: bool = False


def assemble_current_membership(
    target_rows: Any,
    admission_rows: Any,
    diagnosis_rows: Any,
    approved_codes: Any,
) -> RetrospectiveMembershipV2:
    """Assemble index-admission codes in exact target-row binding order.

    A target admission must be present and person-consistent.  A missing,
    unparsable, reversed, or unfinished date interval makes that row ineligible
    rather than a negative member.  Only diagnoses attached to that same index
    admission can set a family member; prior and future admissions are ignored.
    """

    try:
        approved = prior._validate_approved_codes(approved_codes)
        targets = prior._validate_targets(target_rows)
        admissions = prior._materialize_admissions(admission_rows)
        diagnoses = prior._materialize_diagnoses(diagnosis_rows, admissions, approved)
        membership = np.zeros((len(targets), len(FAMILIES)), dtype=bool)
        eligible = np.zeros(len(targets), dtype=bool)
        for index, (person, episode, _binding) in enumerate(targets):
            admission = admissions.get(episode)
            # Unknown target IDs and cross-person links are source-linkage
            # errors.  Date quality is represented by the explicit eligibility
            # mask so it cannot become an implicit disease-negative label.
            if admission is None or admission[0] != person:
                _fail()
            admit, discharge = admission[1], admission[2]
            if admit is None or discharge is None or discharge < admit:
                continue
            eligible[index] = True
            for family in diagnoses.get(episode, frozenset()):
                membership[index, prior._FAMILY_INDEX[family]] = True
        row_binding = np.asarray([binding for _, _, binding in targets], dtype="U64")
        return RetrospectiveMembershipV2(
            _readonly(row_binding), _readonly(membership), _readonly(eligible)
        )
    except Exception:
        _fail()
