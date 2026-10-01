"""Apply the frozen prospective clinical mask without exposing patient rows."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from patient_atlas_preprocessing import hash_json, policy_mask_hash
from patient_atlas_real_data import ExploratoryRawCohort


POLICY_FILENAME = "PATIENT_ATLAS_CONTINUOUS_VALIDITY_POLICY_V1.json"
UNIT_FILENAME = "PATIENT_ATLAS_OFFICIAL_UNIT_RECONCILIATION_V1.json"
FEATURE_FILENAME = "PATIENT_ATLAS_FEATURE_REGISTRY.json"
CONTEXT_FILENAME = "PATIENT_ATLAS_CONTEXT_SCHEMA.json"
PREPROCESSING_FILENAME = "patient_atlas_preprocessing.py"
POLICY_SCHEMA = "patient-atlas-continuous-validity-policy-v1"
EXPECTED_FALSE_INDICES = (8, 9, 20, 35, 36)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_prospective_policy_mask(
    project_root: str | Path,
) -> tuple[np.ndarray, dict[str, str]]:
    """Return the authenticated 59-field mask and its row-free bindings."""

    root = Path(project_root).resolve()
    policy_path = root / POLICY_FILENAME
    unit_path = root / UNIT_FILENAME
    feature_path = root / FEATURE_FILENAME
    context_path = root / CONTEXT_FILENAME
    preprocessing_path = root / PREPROCESSING_FILENAME
    policy: Any = json.loads(policy_path.read_text())
    if not isinstance(policy, dict) or policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("prospective continuous-validity policy has an unsupported schema")
    mask = np.asarray(policy.get("policy_eligible_mask"))
    if mask.shape != (59,) or mask.dtype != np.bool_:
        raise TypeError("prospective clinical policy mask must be boolean with width 59")
    if tuple(np.flatnonzero(~mask)) != EXPECTED_FALSE_INDICES:
        raise ValueError("prospective policy must mask exactly the five unit conflicts")
    feature: Any = json.loads(feature_path.read_text())
    if not isinstance(feature, dict):
        raise TypeError("feature registry must be a JSON object")
    expected_mask_hash = policy_mask_hash(
        str(feature.get("ordered_columns_sha256")), mask
    )
    if policy.get("policy_eligible_mask_sha256") != expected_mask_hash:
        raise ValueError("prospective clinical policy mask hash is invalid")
    artifact_bindings = policy.get("artifact_bindings", {})
    unit_sha256 = _sha256(unit_path)
    authenticated = {
        "feature_registry": _sha256(feature_path),
        "context_schema": _sha256(context_path),
        "official_unit_reconciliation": unit_sha256,
        "preprocessing_code": _sha256(preprocessing_path),
    }
    for name, digest in authenticated.items():
        if artifact_bindings.get(name, {}).get("sha256") != digest:
            raise ValueError(f"prospective policy binds a different {name}")
    decision = policy.get("decision", {})
    if (
        decision.get("prospective_refit_required") is not True
        or decision.get("prospective_refit_completed") is not False
        or decision.get("official_test_accessed") is not False
    ):
        raise ValueError("prospective policy is not in the pre-refit sealed state")
    return mask, {
        "continuous_validity_policy": _sha256(policy_path),
        **authenticated,
    }


def apply_prospective_policy(
    cohort: ExploratoryRawCohort,
    *,
    project_root: str | Path,
) -> ExploratoryRawCohort:
    """Return an aligned cohort view with conflicts excluded before fold fitting."""

    if not bool(np.all(cohort.blood_eligible_mask)):
        raise ValueError("prospective policy must be applied once to the base cohort")
    mask, bindings = load_prospective_policy_mask(project_root)
    composite_policy_sha256 = hash_json(
        {
            "base_source_policy_sha256": cohort.source_policy_sha256,
            **bindings,
        }
    )
    return replace(
        cohort,
        blood_eligible_mask=mask,
        source_policy_sha256=composite_policy_sha256,
        source_hashes={**dict(cohort.source_hashes), **bindings},
    )


def align_cohort_to_preprocessor_policy(
    cohort: ExploratoryRawCohort,
    *,
    preprocessor: Any,
    project_root: str | Path,
) -> tuple[ExploratoryRawCohort, bool]:
    """Select the base or prospective view that exactly matches preprocessing."""

    expected = getattr(preprocessor, "source_policy_sha256", None)
    if expected == cohort.source_policy_sha256:
        return cohort, False
    prospective = apply_prospective_policy(cohort, project_root=project_root)
    if expected != prospective.source_policy_sha256:
        raise ValueError("cohort cannot be aligned to the preprocessor source policy")
    return prospective, True


__all__ = [
    "POLICY_FILENAME",
    "EXPECTED_FALSE_INDICES",
    "align_cohort_to_preprocessor_policy",
    "apply_prospective_policy",
    "load_prospective_policy_mask",
]
