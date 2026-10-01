"""Fail-closed validator for the capacity-preserving Patient Atlas v3."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "patient-atlas-v3-protocol-v1"
DEFAULT_PROTOCOL = "PATIENT_ATLAS_V3_PROTOCOL_V1.json"
V1_OUTCOME_FREE_THRESHOLD = 1.1207184791564941


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def validate_v3_protocol(
    project_root: str | Path,
    protocol_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Authenticate every result binding and the one permitted v3 candidate."""

    root = Path(project_root).resolve()
    path = root / DEFAULT_PROTOCOL if protocol_path is None else Path(protocol_path).resolve()
    path.relative_to(root)
    protocol = _mapping(json.loads(path.read_text()), "v3 protocol")
    prior = _mapping(protocol.get("prior_evidence"), "prior evidence")
    architecture = _mapping(protocol.get("architecture"), "architecture")
    support = _mapping(protocol.get("hard_support_contract"), "support contract")
    training = _mapping(
        protocol.get("preserved_training_contract"), "training contract"
    )
    boundary = _mapping(protocol.get("data_boundary"), "data boundary")
    acceptance = _mapping(
        protocol.get("outcome_free_acceptance"), "outcome-free acceptance"
    )
    failure = _mapping(protocol.get("failure_policy"), "failure policy")
    privacy = _mapping(protocol.get("privacy"), "privacy")

    if (
        protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("status")
        != "frozen_after_v2_rejection_before_v3_training"
    ):
        raise ValueError("v3 protocol status differs")
    expected_bindings = {
        "v2_decision": "PATIENT_ATLAS_V2_DEVELOPMENT_DECISION_V1.json",
        "v2_development_report": "PATIENT_ATLAS_V2_DEVELOPMENT_VALIDATION_ATTEMPT2.json",
        "v2_stage2_summary": "PATIENT_ATLAS_V2_STAGE2_ATTEMPT1.json",
        "v1_stage2_control": "PATIENT_ATLAS_EXPLORATORY_STAGE2_PROSPECTIVE_V1_ATTEMPT2.json",
        "v1_release": "PATIENT_ATLAS_RELEASE_EVIDENCE_V1.json",
        "v1_official_test_retirement": "PATIENT_ATLAS_OFFICIAL_TEST_RESULT_V1.json",
        "confirmatory_readout_freeze": "PATIENT_ATLAS_CONFIRMATORY_READOUT_FREEZE_V1.json",
    }
    if set(prior) != set(expected_bindings):
        raise ValueError("v3 prior-evidence binding set differs")
    for role, filename in expected_bindings.items():
        binding = _mapping(prior[role], role)
        if binding.get("file") != filename:
            raise ValueError(f"v3 {role} filename differs")
        artifact = (root / filename).resolve()
        artifact.relative_to(root)
        if not artifact.is_file() or _sha256(artifact) != binding.get("sha256"):
            raise ValueError(f"v3 {role} hash binding differs")
    if (
        prior["v1_official_test_retirement"].get(
            "official_test_rescoring_allowed"
        )
        is not False
        or not math.isclose(
            float(prior["v1_stage2_control"].get("outcome_free_balanced_proper_score", -1.0)),
            V1_OUTCOME_FREE_THRESHOLD,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("v3 v1 control or test retirement differs")

    paired = _mapping(architecture.get("paired_evidence_vector"), "paired vector")
    if (
        architecture.get("total_fused_latent_dimension") != 96
        or architecture.get("shared_dimension") != 32
        or architecture.get("eye_private_dimension") != 32
        or architecture.get("clinical_private_dimension") != 32
        or architecture.get("per_modality_permitted_factor_dimension") != 64
        or paired.get("name")
        != "capacity_preserving_paired_evidence_atlas_129"
        or paired.get("dimension") != 129
        or architecture.get("uncertainty_sidecar_dimension") != 128
        or architecture.get("fused_decoder_state_dimension") != 96
        or architecture.get("retinal_pixel_decoder")
        != "explicitly deferred sidecar"
    ):
        raise ValueError("v3 architecture differs")
    if (
        support.get("eye_evidence") != ["shared", "eye_private"]
        or support.get("clinical_evidence") != ["shared", "clinical_private"]
        or support.get("eye_decoder") != ["shared", "eye_private"]
        or support.get("clinical_decoder") != ["shared", "clinical_private"]
        or support.get("forbidden_cross_private_loadings_exactly_zero") is not True
        or support.get("missing_private_block_mean") != 0.0
        or support.get("missing_private_block_variance") != 1.0
        or support.get("availability_masks_appended_to_default_vector") is not False
    ):
        raise ValueError("v3 support or missingness contract differs")

    if (
        training.get("candidate_count") != 1
        or training.get("latent_partition_tuned") is not False
        or training.get("training_seed") != 20260826
        or training.get("beta") != 0.01
        or training.get("group_shrinkage_rate") != 1e-6
        or training.get("interaction_enabled") is not False
        or training.get("external_blood_anchor_enabled") is not False
        or training.get("clinical_unit_conflict_fields_masked") != 5
        or training.get("eligible_clinical_fields") != 54
    ):
        raise ValueError("v3 training contract differs")
    if (
        boundary.get("functional_outcomes_loaded_during_v3_stage2") is not False
        or boundary.get("official_validation_functional_outcomes")
        != "prohibited for v3 training, selection, and engineering promotion"
        or boundary.get("retired_v1_official_test") != "prohibited permanently"
        or boundary.get("new_untouched_cohort_required_before_any_v3_screening_claim")
        is not True
        or acceptance.get("lower_is_better") is not True
        or not math.isclose(
            float(acceptance.get("strict_threshold", -1.0)),
            V1_OUTCOME_FREE_THRESHOLD,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or failure.get("used_functional_validation_may_not_rescue_candidate")
        is not True
    ):
        raise ValueError("v3 data or acceptance boundary differs")
    if any(
        privacy.get(key) is not False
        for key in (
            "patient_rows_emitted",
            "patient_identifiers_emitted",
            "patient_vectors_emitted",
            "per_patient_predictions_emitted",
        )
    ):
        raise ValueError("v3 privacy contract differs")
    return protocol


__all__ = [
    "DEFAULT_PROTOCOL",
    "SCHEMA_VERSION",
    "V1_OUTCOME_FREE_THRESHOLD",
    "validate_v3_protocol",
]
