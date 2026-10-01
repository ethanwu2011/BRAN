"""Private, source-free frame assembly for the two-source S1 analysis.

The admission contracts and state artifacts are authenticated by their caller
before reaching this module.  This layer only validates their closed in-memory
schemas, translates source-specific age/outcome conventions, and concatenates
the two source-local frames.  It does not read source files, fit a model,
select a disease cohort, or deduplicate people across sources.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

import bran_clinical_discovery_landmark_v3 as discovery
import bran_mimic_selected_state_v3 as mimic_state_bridge
import bran_multisource_clinical_panel_s1 as panel
import run_bran_eicu_discovery_admission_e2 as eicu_admission
import run_bran_mimic_broad_admission_m1 as mimic_admission
from run_bran_eicu_v5_state_e3 import validate_arrays as validate_projection


ERROR = "multisource_clinical_inputs_s1_failed"
FRAME_KEYS = (
    "state",
    "values",
    "observed",
    "age_value",
    "age_lower",
    "age_upper",
    "age_kind",
    "source",
    "roles",
    "membership",
    "outcome",
    "person_group",
)
STATE_KEYS = ("parent_row", "state", "available", "clinical", "clinical_mask")
STATE_WIDTH = 192
CLINICAL_WIDTH = 59
PHYSIOLOGY_WIDTH = 21


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError(ERROR) from None


def _state_mapping(state: object) -> dict[str, object]:
    """Copy only the packed state fields from a mapping or private object."""
    try:
        if isinstance(state, Mapping):
            _require(set(state) == set(STATE_KEYS))
            return {key: state[key] for key in STATE_KEYS}
        # Supporting a private state wrapper keeps this adapter independent of
        # the lifecycle runner while still requiring its explicit row binding.
        return {key: getattr(state, key) for key in STATE_KEYS}
    except Exception:
        raise ValueError(ERROR) from None


def _validate_state(state: object, rows: int) -> dict[str, np.ndarray]:
    """Validate the common packed state contract without I/O or inference."""
    try:
        raw = _state_mapping(state)
        parent = raw["parent_row"]
        encoded = raw["state"]
        available = raw["available"]
        clinical = raw["clinical"]
        clinical_mask = raw["clinical_mask"]
        _require(type(parent) is np.ndarray and parent.dtype == np.dtype(np.int64)
                 and parent.shape == (rows,)
                 and np.array_equal(parent, np.arange(rows, dtype=np.int64)))
        _require(type(encoded) is np.ndarray and encoded.dtype == np.dtype(np.float32)
                 and encoded.shape == (rows, STATE_WIDTH) and np.isfinite(encoded).all())
        _require(type(available) is np.ndarray and available.dtype == np.dtype(bool)
                 and available.shape == (rows,))
        _require(type(clinical) is np.ndarray and clinical.dtype == np.dtype(np.float64)
                 and clinical.shape == (rows, CLINICAL_WIDTH) and np.isfinite(clinical).all())
        _require(type(clinical_mask) is np.ndarray and clinical_mask.dtype == np.dtype(bool)
                 and clinical_mask.shape == (rows, CLINICAL_WIDTH))
        _require((clinical[~clinical_mask] == 0).all()
                 and np.array_equal(available, clinical_mask.any(axis=1))
                 and (encoded[~available] == 0).all())
        return {
            "state": np.array(encoded, dtype=np.float32, copy=True),
            "available": np.array(available, dtype=bool, copy=True),
            "clinical": np.array(clinical, dtype=np.float64, copy=True),
            "clinical_mask": np.array(clinical_mask, dtype=bool, copy=True),
        }
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def _mimic_age(cohort: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    try:
        # M1's packed kinds are translated by the sealed V5 bridge: a derived
        # year is an interval, a top-coded year is right-censored, and missing
        # age remains unknown.  No scalar age is fabricated.
        return {
            key: np.array(value, copy=True)
            for key, value in mimic_state_bridge._typed_age_arrays(cohort).items()
        }
    except Exception:
        raise ValueError(ERROR) from None


def _eicu_age(cohort: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Translate E2 reported/unknown/censored age to the S1 typed-age codes."""
    try:
        triplet = cohort["age_triplet"]
        packed_kind = cohort["age_kind"]
        n = len(packed_kind)
        value = np.full(n, np.nan, dtype=np.float64)
        lower = np.full(n, np.nan, dtype=np.float64)
        upper = np.full(n, np.nan, dtype=np.float64)
        kind = np.full(n, 3, dtype=np.int64)  # S1 UNKNOWN
        reported = packed_kind == 1
        censored = packed_kind == 2
        value[reported] = triplet[reported, 0]
        kind[reported] = 0  # S1 REPORTED
        lower[censored] = triplet[censored, 1]
        kind[censored] = 2  # S1 RIGHT_CENSORED
        return {"age_value": value, "age_lower": lower,
                "age_upper": upper, "age_kind": kind}
    except Exception:
        raise ValueError(ERROR) from None


def _eicu_roles(person: np.ndarray) -> np.ndarray:
    """Assign E2 roles once, using source-prefixed local person keys only."""
    try:
        keys = {"eicu:" + str(value) for value in person.tolist()}
        _require(len(keys) == len(person))
        role_by_key = discovery._person_roles(keys)
        result = np.asarray([role_by_key["eicu:" + str(value)] for value in person.tolist()],
                            dtype=np.uint8)
        _require(result.shape == (len(person),) and np.isin(result, (0, 1, 2)).all())
        return result
    except Exception:
        raise ValueError(ERROR) from None


def _outcome_mimic(value: np.ndarray) -> np.ndarray:
    # M1 status: 0 unknown, 1 death, 2 surviving discharge.
    result = np.full(value.shape, -1, dtype=np.int8)
    result[value == 1] = 1
    result[value == 2] = 0
    return result


def _outcome_eicu(value: np.ndarray) -> np.ndarray:
    # E2 already carries -1 unknown, 0 survival, 1 death.
    result = np.array(value, dtype=np.int8, copy=True)
    _require(np.isin(result, (-1, 0, 1)).all())
    return result


def _copy_cohort(cohort: object, validator) -> dict[str, np.ndarray]:
    try:
        _require(isinstance(cohort, dict))
        validator(cohort)
        # Do not expose or retain caller-owned views.  The validator is called
        # before copying so it remains the sole source-schema authority.
        return {key: np.array(value, copy=True) for key, value in cohort.items()}
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def assemble(mimic_cohort: object, mimic_state: object,
             eicu_cohort: object, eicu_state: object) -> dict[str, np.ndarray]:
    """Return the exact private S1 frame from two already-authenticated pools.

    The function retains every supplied source-local row and never uses disease
    membership or outcomes to select rows or assign roles.  ``person_group``
    is a source-local ordinal namespace: equal identifiers in different source
    pools therefore remain distinct without making an identity claim.
    """
    try:
        mimic = _copy_cohort(mimic_cohort, mimic_admission.validate_arrays)
        eicu = _copy_cohort(eicu_cohort, eicu_admission.validate_arrays)
        n_m, n_e = len(mimic["person"]), len(eicu["person"])
        _require(n_m >= 1 and n_e >= 1)
        m_state = _validate_state(mimic_state, n_m)
        e_state = _validate_state(eicu_state, n_e)
        # Reject self-consistent but wrong original-unit/mask projections too.
        validate_projection(_state_mapping(mimic_state), mimic)
        validate_projection(_state_mapping(eicu_state), eicu)
        expected_families = ('type2_or_unspecified_diabetes', 'heart_failure', 'chronic_kidney_disease')
        _require(tuple(mimic_admission.FAMILIES) == expected_families
                 and tuple(eicu_admission.e1.dictionary.FAMILIES) == expected_families)

        m_age = _mimic_age(mimic)
        e_age = _eicu_age(eicu)
        e_roles = _eicu_roles(eicu["person"])

        # The M1 family order is dictionary-bound (diabetes, HF, CKD).  E2's
        # second and third columns are the already-qualified dictionary-bound
        # HF/CKD memberships; only diabetes is deliberately replaced by the
        # separate, type-blind APACHE indicator definition.
        m_membership = np.array(mimic["membership"], dtype=bool, copy=True)
        e_membership = np.array(eicu["membership"], dtype=bool, copy=True)
        e_membership[:, 0] = eicu["apache_diabetes_indicator"] == 1

        frame = {
            "state": np.concatenate((m_state["state"], e_state["state"]), axis=0),
            "values": np.concatenate((mimic["values"], eicu["values"]), axis=0),
            "observed": np.concatenate((mimic["observed"], eicu["observed"]), axis=0),
            "age_value": np.concatenate((m_age["age_value"], e_age["age_value"])),
            "age_lower": np.concatenate((m_age["age_lower"], e_age["age_lower"])),
            "age_upper": np.concatenate((m_age["age_upper"], e_age["age_upper"])),
            "age_kind": np.concatenate((m_age["age_kind"], e_age["age_kind"])),
            "source": np.concatenate((np.zeros(n_m, dtype=np.uint8),
                                       np.ones(n_e, dtype=np.uint8))),
            "roles": np.concatenate((np.array(mimic["roles"], dtype=np.uint8, copy=True),
                                      e_roles)),
            "membership": np.concatenate((m_membership, e_membership), axis=0),
            "outcome": np.concatenate((_outcome_mimic(mimic["outcome"]),
                                        _outcome_eicu(eicu["outcome"]))),
            # Ordinals are intentionally source-local namespaces, offset only
            # to remain unique in the pooled frame.  This is not a dedup key.
            "person_group": np.arange(n_m + n_e, dtype=np.int64),
        }
        _require(tuple(frame) == FRAME_KEYS)
        panel.check_frame(frame)
        return frame
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


__all__ = ["ERROR", "FRAME_KEYS", "STATE_KEYS", "assemble"]
