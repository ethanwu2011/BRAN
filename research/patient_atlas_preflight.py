"""Fail-closed manifest preflight for Patient Atlas experiments.

This module deliberately operates on contracts and schemas only. It never opens a
patient table, embedding matrix, target vector, or fold manifest. Its purpose is
to prevent a structurally valid synthetic build from being mistaken for a data-
ready or confirmatory experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FEATURE_SCHEMA = "patient-atlas-feature-registry-v1"
CONTEXT_SCHEMA = "patient-atlas-context-v1"
EYE_SCHEMA = "patient-atlas-eye-registry-v1"
TARGET_SCHEMA = "patient-atlas-primary-targets-v1"
BLOOD_CONTRACT_SCHEMA = "external-blood-tower-v1"
BLOOD_ARTIFACT_ID = "external_denoise30_v1"
EYE_CONTRACT_SCHEMA = "external-eye-tower-v1"
EYE_ARTIFACT_ID = "adapted_dinov3_s16_fundus_v1"
EXPECTED_FEATURE_COUNT = 59
EXPECTED_CONTINUOUS_COUNT = 48
EXPECTED_BINARY_COUNT = 11
EXPECTED_ANCHOR_ELIGIBLE_COUNT = 45
EXPECTED_TARGET_COUNT = 5
SOURCE_POLICY_SCHEMA = "patient-atlas-source-policy-v1"
RAW_AUDIT_SCHEMA = "patient-atlas-raw-source-audit-v1"
CLINICAL_FIELD_CONTRACT_SCHEMA = "patient-atlas-clinical-field-contract-v1"
COMPARISON_POLICY_SCHEMA = "patient-atlas-comparison-policy-v1"
OFFICIAL_UNIT_RECONCILIATION_SCHEMA = "patient-atlas-official-unit-reconciliation-v1"
EXPECTED_UNIT_CONFLICT_FIELDS = frozenset(
    {"c_peptide", "insulin", "calcium", "urine_albumin", "urine_creatinine"}
)
OFFICIAL_DOCS_COMMIT = "4e15c6c2e51e0615ac941c299b574b084e7a2334"
OFFICIAL_LAB_DOC_SHA256 = "a36b367f28afca7aa0495ad8ce59186277902ca3e966df8aa2e67e82ba23bedf"
OFFICIAL_PHYSICAL_DOC_SHA256 = "1b5882c0b9e8210169acfb70077e0fcf1b403c648e000dd539367567d036c9bb"
COMPONENT_POLICY_SCHEMA = "patient-atlas-component-policy-v1"
SOURCE_POLICY_SHA256 = "57f8499830c94503353e529e9229f79d9f3e9afdec620b69ac4113b268a816c3"
SELECTED_STAGE2_SUMMARY_SHA256 = "0bd36b642fd4c4a90469b9c64060cedd8b81fc4e6d87ba296cd50eb877e1dd39"
EXTERNAL_BLOOD_CONTRACT_SHA256 = "1742988192def381d6b409b58d1cd6b7acadc6b0e52964a75084c1270d690987"
CONTINUOUS_VALIDITY_POLICY_SCHEMA = "patient-atlas-continuous-validity-policy-v1"
OFFICIAL_UNIT_POLICY_SHA256 = "4d428667185a974116b0f13d525bcf86c9837ea86806604b26f6f0c41184e167"
PREPROCESSING_CODE_SHA256 = "31761a42f8c0781a22c42eceb9040f6be2b28c4b972672e1ad415d0ee27fe636"
EXPECTED_POLICY_MASK_SHA256 = "31babdb7eec88d3ab29c95f7b9a6cc95b9e37989d1438f9e2445a7ef6c7542af"
PROSPECTIVE_REFIT_ATTESTATION_SCHEMA = (
    "patient-atlas-prospective-refit-attestation-v1"
)


class PreflightStage(str, Enum):
    """Increasingly strict experiment boundaries."""

    SYNTHETIC = "synthetic"
    EXPLORATORY = "exploratory"
    REAL_REPRESENTATION = "real_representation"
    CONFIRMATORY = "confirmatory"


@dataclass(frozen=True)
class PreflightIssue:
    code: str
    manifest: str
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    stage: str
    ready: bool
    blocking_issues: tuple[PreflightIssue, ...]
    manifest_sha256: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "ready": self.ready,
            "blocking_issues": [asdict(issue) for issue in self.blocking_issues],
            "manifest_sha256": dict(sorted(self.manifest_sha256.items())),
        }


class DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError, DuplicateKeyError) as exc:
        raise ValueError(f"invalid manifest {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"manifest {path.name} must contain a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _issue(
    issues: list[PreflightIssue], code: str, manifest: str, detail: str
) -> None:
    issues.append(PreflightIssue(code=code, manifest=manifest, detail=detail))


def _validate_feature_registry(
    registry: Mapping[str, Any], contract: Mapping[str, Any], issues: list[PreflightIssue]
) -> None:
    name = "PATIENT_ATLAS_FEATURE_REGISTRY.json"
    if registry.get("schema_version") != FEATURE_SCHEMA:
        _issue(issues, "feature_schema_version", name, "unexpected schema version")

    features = registry.get("features")
    if not isinstance(features, list) or len(features) != EXPECTED_FEATURE_COUNT:
        _issue(issues, "feature_count", name, "exactly 59 ordered features are required")
        return

    if registry.get("feature_count") != EXPECTED_FEATURE_COUNT:
        _issue(issues, "declared_feature_count", name, "declared feature count must be 59")

    ordered_names: list[str] = []
    indices: list[int] = []
    continuous = 0
    binary = 0
    anchor_mask: list[bool] = []
    for feature in features:
        if not isinstance(feature, dict):
            _issue(issues, "feature_record_type", name, "every feature must be an object")
            return
        ordered_names.append(str(feature.get("name")))
        indices.append(feature.get("index"))
        feature_type = feature.get("type")
        continuous += int(feature_type == "continuous")
        binary += int(feature_type == "binary")
        anchor_mask.append(feature.get("anchor_eligible") is True)

    if indices != list(range(EXPECTED_FEATURE_COUNT)):
        _issue(issues, "feature_indices", name, "feature indices must equal 0..58 in order")
    if len(set(ordered_names)) != EXPECTED_FEATURE_COUNT:
        _issue(issues, "feature_names_unique", name, "feature names must be unique")
    if (continuous, binary) != (EXPECTED_CONTINUOUS_COUNT, EXPECTED_BINARY_COUNT):
        _issue(issues, "feature_types", name, "expected 48 continuous and 11 binary fields")

    computed_columns_hash = _canonical_json_sha256(ordered_names)
    if registry.get("ordered_columns_sha256") != computed_columns_hash:
        _issue(issues, "feature_columns_hash", name, "ordered feature-name hash is invalid")

    contract_names = contract.get("ordered_columns")
    contract_coverage = contract.get("anchor_pretraining_coverage", {})
    contract_mask = contract_coverage.get("anchor_eligible_mask")
    if contract_names != ordered_names:
        _issue(
            issues,
            "contract_feature_order",
            "EXTERNAL_BLOOD_TOWER_CONTRACT.json",
            "tower and raw-clinical feature orders differ",
        )
    if contract.get("ordered_columns_sha256") != computed_columns_hash:
        _issue(
            issues,
            "contract_columns_hash",
            "EXTERNAL_BLOOD_TOWER_CONTRACT.json",
            "tower ordered-column hash is invalid",
        )
    if contract_mask != anchor_mask:
        _issue(
            issues,
            "anchor_policy_mismatch",
            "EXTERNAL_BLOOD_TOWER_CONTRACT.json",
            "tower and feature-registry anchor masks differ",
        )
    if sum(anchor_mask) != EXPECTED_ANCHOR_ELIGIBLE_COUNT:
        _issue(issues, "anchor_count", name, "exactly 45 tower slots must be anchor eligible")


def _validate_clinical_field_contract(
    clinical: Mapping[str, Any],
    registry: Mapping[str, Any],
    registry_sha256: str,
    issues: list[PreflightIssue],
) -> None:
    name = "PATIENT_ATLAS_CLINICAL_FIELD_CONTRACT_V1.json"
    if clinical.get("schema_version") != CLINICAL_FIELD_CONTRACT_SCHEMA:
        _issue(issues, "clinical_field_contract_version", name, "unexpected contract version")
        return
    fields = clinical.get("fields")
    expected = [
        feature
        for feature in registry.get("features", ())
        if isinstance(feature, Mapping) and feature.get("type") == "continuous"
    ]
    if not isinstance(fields, list) or len(fields) != EXPECTED_CONTINUOUS_COUNT:
        _issue(issues, "clinical_field_contract_count", name, "exactly 48 fields are required")
        return
    if [field.get("index") for field in fields] != [field.get("index") for field in expected]:
        _issue(issues, "clinical_field_contract_indices", name, "field indices differ from the fitted registry")
    if [field.get("name") for field in fields] != [field.get("name") for field in expected]:
        _issue(issues, "clinical_field_contract_names", name, "field names differ from the fitted registry")
    bindings = clinical.get("source_bindings", {})
    if bindings.get("feature_registry_sha256") != registry_sha256:
        _issue(issues, "clinical_field_registry_binding", name, "contract binds a different fitted registry")
    for field, value in bindings.items():
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            _issue(issues, "clinical_field_source_binding", name, f"{field} is not a SHA-256 digest")
    summary = clinical.get("status_summary", {})
    if (
        summary.get("continuous_field_count") != EXPECTED_CONTINUOUS_COUNT
        or summary.get("visit_and_replicate_policy_resolved_count")
        != EXPECTED_CONTINUOUS_COUNT
        or summary.get("confirmatory_plausibility_range_authorized_count") != 0
        or summary.get("complete_unit_and_validity_contract") is not False
        or summary.get("real_training_ready") is not False
        or clinical.get("model_input_semantics_changed") is not False
    ):
        _issue(issues, "clinical_field_summary", name, "contract readiness summary overclaims or differs")
    identity_count = sum(
        field.get("conversion", {}).get("status")
        == "identity_authorized_within_this_release"
        for field in fields
    )
    if summary.get("canonical_identity_unit_authorized_count") != identity_count:
        _issue(issues, "clinical_field_unit_count", name, "authorized unit count differs from field records")
    allowed_statuses = {
        "authenticated_unique_source_unit",
        "incomplete_row_level_unit_coverage",
        "malformed_source_unit",
        "unavailable_in_release",
        "ambiguous_multiple_source_units",
    }
    for field in fields:
        if (
            field.get("source_unit", {}).get("status") not in allowed_statuses
            or field.get("visit_and_replicate_policy_status")
            != "resolved_and_fail_closed"
            or field.get("raw_plausibility_range") is not None
            or field.get("source_reference_interval", {}).get("endpoints_emitted")
            is not False
        ):
            _issue(issues, "clinical_field_record", name, "a field record is malformed or unsafe")
            break
    privacy = clinical.get("privacy", {})
    if any(
        privacy.get(field) is not False
        for field in (
            "patient_rows_emitted",
            "patient_identifiers_emitted",
            "raw_values_emitted",
            "range_endpoints_emitted",
        )
    ):
        _issue(issues, "clinical_field_privacy", name, "contract contains prohibited patient material")


def _validate_blood_contract(
    contract: Mapping[str, Any], issues: list[PreflightIssue]
) -> None:
    name = "EXTERNAL_BLOOD_TOWER_CONTRACT.json"
    if contract.get("schema_version") != BLOOD_CONTRACT_SCHEMA:
        _issue(issues, "blood_contract_version", name, "unexpected contract version")
    if contract.get("artifact_id") != BLOOD_ARTIFACT_ID:
        _issue(issues, "blood_artifact_id", name, "only external_denoise30_v1 is accepted")
    for field in ("file_sha256", "state_sha256", "ordered_columns_sha256"):
        value = contract.get(field)
        if not isinstance(value, str) or len(value) != 64:
            _issue(issues, f"blood_{field}", name, f"{field} must be a SHA-256 hex digest")
    if not isinstance(contract.get("state_hash_algorithm"), str):
        _issue(issues, "blood_hash_algorithm", name, "state hash algorithm must be explicit")
    provenance = contract.get("training_provenance", {})
    if provenance.get("paired_ai_readi_rows_used") != 0:
        _issue(issues, "blood_pretraining_leakage", name, "paired AI-READI rows must equal zero")


def _validate_context(context: Mapping[str, Any], issues: list[PreflightIssue]) -> None:
    name = "PATIENT_ATLAS_CONTEXT_SCHEMA.json"
    if context.get("schema_version") != CONTEXT_SCHEMA:
        _issue(issues, "context_schema_version", name, "unexpected context schema version")
    fields = context.get("fields")
    if not isinstance(fields, list) or [field.get("name") for field in fields] != ["age"]:
        _issue(issues, "context_fields", name, "v1 requires exactly the named age context field")
        return
    age = fields[0]
    if age.get("units") != "years" or age.get("required_for_confirmatory_v1") is not True:
        _issue(issues, "age_contract", name, "age must be observed in years for confirmatory v1")
    transform = age.get("outer_fold_transform", {})
    if transform.get("center") != "median" or transform.get("scale") != "IQR/1.349":
        _issue(issues, "age_transform", name, "age transform must be outer-fold median/IQR")
    expected_order = ["physiology_mean[0:64]", "standardized_age"]
    if context.get("default_vector_order") != expected_order:
        _issue(issues, "patient_vector_order", name, "default patient vector order changed")


def _validate_eye_registry(
    eye: Mapping[str, Any],
    eye_contract: Mapping[str, Any],
    issues: list[PreflightIssue],
) -> None:
    name = "PATIENT_ATLAS_EYE_REGISTRY.json"
    if eye.get("schema_version") != EYE_SCHEMA:
        _issue(issues, "eye_schema_version", name, "unexpected eye registry version")
    embedding = eye.get("embedding", {})
    if embedding.get("dimension") != 384 or embedding.get("tower") != "adapted DINOv3-S/16":
        _issue(issues, "eye_embedding_contract", name, "expected the frozen 384-d adapted DINO tower")
    row_rule = eye.get("row_alignment", {}).get("rule", "")
    if "embedding_row" not in row_rule or "before filtering" not in row_rule:
        _issue(issues, "eye_row_alignment", name, "physical embedding-row preservation is required")
    forbidden = set(eye.get("forbidden_model_inputs", []))
    required_forbidden = {"path", "pid", "ok_as_quality", "outer_fold_id", "target_label"}
    if not required_forbidden.issubset(forbidden):
        _issue(issues, "eye_forbidden_inputs", name, "observation/process identifiers must be forbidden")
    registry_contract = eye.get("tower_contract", {})
    contract_pairs = {
        "schema_version": EYE_CONTRACT_SCHEMA,
        "artifact_id": EYE_ARTIFACT_ID,
        "checkpoint_file_sha256": eye_contract.get("file_sha256"),
        "state_sha256": eye_contract.get("state_sha256"),
    }
    if any(registry_contract.get(key) != value for key, value in contract_pairs.items()):
        _issue(issues, "eye_registry_contract", name, "eye registry and tower contract differ")


def _validate_eye_contract(
    contract: Mapping[str, Any], issues: list[PreflightIssue]
) -> None:
    name = "EXTERNAL_EYE_TOWER_CONTRACT.json"
    if contract.get("schema_version") != EYE_CONTRACT_SCHEMA:
        _issue(issues, "eye_contract_version", name, "unexpected contract version")
    if contract.get("artifact_id") != EYE_ARTIFACT_ID:
        _issue(issues, "eye_artifact_id", name, "unexpected adapted eye artifact")
    for field in ("file_sha256", "state_sha256"):
        value = contract.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            _issue(issues, f"eye_{field}", name, f"{field} must be a lowercase SHA-256 digest")
    architecture = contract.get("architecture", {})
    inference = contract.get("inference", {})
    if architecture.get("embedding_width") != 384 or inference.get(
        "per_image_output_width"
    ) != 384:
        _issue(issues, "eye_contract_width", name, "eye tower must emit 384 dimensions")
    provenance = contract.get("source_cache_provenance", {})
    counts = provenance.get("source_counts", {})
    if (
        not isinstance(counts, dict)
        or any(not isinstance(value, int) or value < 0 for value in counts.values())
        or sum(counts.values()) != 115273
    ):
        _issue(issues, "eye_source_counts", name, "source counts must sum to 115,273")
    if provenance.get("ai_readi_source_tag_count") != 0:
        _issue(issues, "eye_pretraining_source_tag", name, "AI-READI source tags must equal zero")
    if provenance.get("per_file_identity_manifest_available") is not False:
        _issue(
            issues,
            "eye_overlap_limit",
            name,
            "historical per-file identity limitation must remain explicit",
        )


def _validate_targets(targets: Mapping[str, Any], issues: list[PreflightIssue]) -> None:
    name = "PATIENT_ATLAS_TARGET_MANIFEST.json"
    if targets.get("schema_version") != TARGET_SCHEMA:
        _issue(issues, "target_schema_version", name, "unexpected target manifest version")
    target_list = targets.get("targets")
    if not isinstance(target_list, list) or len(target_list) != EXPECTED_TARGET_COUNT:
        _issue(issues, "target_count", name, "exactly five primary non-input targets are required")
        return
    if len({target.get("id") for target in target_list}) != EXPECTED_TARGET_COUNT:
        _issue(issues, "target_ids", name, "target ids must be unique")
    if targets.get("representation_artifact") != "atlas-full":
        _issue(issues, "target_artifact", name, "primary targets require the atlas-full artifact")
    if targets.get("primary_estimand") != "family_balanced_outer_test_normalized_squared_loss":
        _issue(issues, "primary_estimand", name, "primary estimand changed")


def _validate_comparison_policy(
    policy: Mapping[str, Any],
    target_manifest_sha256: str,
    issues: list[PreflightIssue],
) -> None:
    """Authenticate the decision-only augmentation without rewriting old results."""

    name = "PATIENT_ATLAS_COMPARISON_POLICY_V1.json"
    if policy.get("schema_version") != COMPARISON_POLICY_SCHEMA:
        _issue(issues, "comparison_policy_version", name, "unexpected policy version")
        return
    if policy.get("status") != "frozen_before_official_test":
        _issue(issues, "comparison_policy_status", name, "policy must be frozen before test access")

    binding = policy.get("source_target_manifest", {})
    if (
        binding.get("file") != "PATIENT_ATLAS_TARGET_MANIFEST.json"
        or binding.get("sha256") != target_manifest_sha256
    ):
        _issue(issues, "comparison_target_binding", name, "policy binds a different target manifest")

    primary = policy.get("primary_claim", {})
    expected_contrasts = [
        "eye_atlas_plus_age minus both_atlas_plus_age",
        "blood_clinical_atlas_plus_age minus both_atlas_plus_age",
    ]
    if (
        primary.get("estimand") != "family_balanced_outer_test_normalized_squared_loss"
        or primary.get("contrasts") != expected_contrasts
        or primary.get("multiplicity") != "Holm one-sided family-wise alpha 0.05"
        or "both one-sided superiority nulls rejected" not in str(primary.get("decision_rule", ""))
    ):
        _issue(issues, "comparison_primary_claim", name, "paired-versus-single primary rule changed")

    concat = policy.get("concat_reference", {})
    if (
        concat.get("role") != "descriptive_reference_only"
        or concat.get("formal_superiority_claim") is not False
        or concat.get("formal_noninferiority_claim") is not False
        or concat.get("formal_equivalence_claim") is not False
        or concat.get("noninferiority_margin_required") is not False
        or concat.get("missing_margin_blocks_confirmatory_evaluation") is not False
    ):
        _issue(issues, "comparison_concat_role", name, "concat must remain descriptive-only")

    supersession = policy.get("supersession", {})
    if (
        supersession.get("target_definitions_changed") is not False
        or supersession.get("representation_changed") is not False
        or supersession.get("model_input_semantics_changed") is not False
    ):
        _issue(issues, "comparison_scope", name, "comparison policy may not change targets or representation")

    official = policy.get("official_test_policy", {})
    if (
        official.get("test_partition_accessed_during_freeze") is not False
        or official.get("score_once_after_all_remaining_release_gates_pass") is not True
        or official.get("no_post_test_retuning") is not True
    ):
        _issue(issues, "comparison_test_policy", name, "official-test seal policy changed")


def _validate_official_unit_reconciliation(
    policy: Mapping[str, Any],
    registry: Mapping[str, Any],
    registry_sha256: str,
    base_contract_sha256: str,
    issues: list[PreflightIssue],
) -> None:
    """Validate the prospective unit policy without changing fitted artifacts."""

    name = "PATIENT_ATLAS_OFFICIAL_UNIT_RECONCILIATION_V1.json"
    if policy.get("schema_version") != OFFICIAL_UNIT_RECONCILIATION_SCHEMA:
        _issue(issues, "official_unit_policy_version", name, "unexpected policy version")
        return
    fields = policy.get("fields")
    expected = [
        feature
        for feature in registry.get("features", ())
        if isinstance(feature, Mapping) and feature.get("type") == "continuous"
    ]
    if not isinstance(fields, list) or len(fields) != EXPECTED_CONTINUOUS_COUNT:
        _issue(issues, "official_unit_policy_count", name, "exactly 48 fields are required")
        return
    if [field.get("index") for field in fields] != [field.get("index") for field in expected]:
        _issue(issues, "official_unit_policy_indices", name, "field indices differ from the registry")
    if [field.get("name") for field in fields] != [field.get("name") for field in expected]:
        _issue(issues, "official_unit_policy_names", name, "field names differ from the registry")

    conflicts = {
        field.get("name")
        for field in fields
        if field.get("evidence_status") == "blocked_unit_conflict"
    }
    authorized = sum(field.get("canonical_unit_authorized") is True for field in fields)
    summary = policy.get("status_summary", {})
    if (
        conflicts != EXPECTED_UNIT_CONFLICT_FIELDS
        or authorized != 43
        or summary.get("canonical_unit_authorized_count") != 43
        or summary.get("fully_masked_unit_conflict_count") != 5
        or summary.get("fully_masked_unit_conflict_fields")
        != sorted(EXPECTED_UNIT_CONFLICT_FIELDS)
        or summary.get("all_fields_have_fail_closed_unit_policy") is not True
        or summary.get("unit_metadata_gate_resolved_by_masking_conflicts") is not True
        or summary.get("confirmatory_plausibility_range_authorized_count") != 0
        or summary.get("plausibility_policy_still_unresolved") is not True
        or summary.get("future_confirmatory_refit_required") is not True
        or summary.get("existing_exploratory_model_changed") is not False
    ):
        _issue(issues, "official_unit_policy_summary", name, "unit-policy summary changed")
    for field in fields:
        field_name = field.get("name")
        if field_name in EXPECTED_UNIT_CONFLICT_FIELDS:
            if (
                field.get("canonical_unit") is not None
                or field.get("canonical_unit_authorized") is not False
                or field.get("unit_conflict") is None
                or not str(field.get("confirmatory_row_policy", "")).startswith("fully mask")
            ):
                _issue(issues, "official_unit_conflict_not_masked", name, str(field_name))
        elif (
            not isinstance(field.get("canonical_unit"), str)
            or field.get("canonical_unit_authorized") is not True
        ):
            _issue(issues, "official_unit_authorization", name, str(field_name))
        if field.get("plausibility_range") is not None or field.get(
            "plausibility_range_authorized"
        ) is not False:
            _issue(issues, "official_unit_range_overclaim", name, str(field_name))

    bindings = policy.get("source_bindings", {})
    if (
        bindings.get("feature_registry_sha256") != registry_sha256
        or bindings.get("base_clinical_field_contract_sha256")
        != base_contract_sha256
    ):
        _issue(issues, "official_unit_local_binding", name, "policy binds different local contracts")
    if (
        bindings.get("official_docs_commit") != OFFICIAL_DOCS_COMMIT
        or bindings.get("clinical_lab_json_sha256") != OFFICIAL_LAB_DOC_SHA256
        or bindings.get("physical_assessment_sha256")
        != OFFICIAL_PHYSICAL_DOC_SHA256
    ):
        _issue(issues, "official_unit_documentation_binding", name, "official documentation binding changed")
    privacy = policy.get("privacy", {})
    if any(
        privacy.get(key) is not False
        for key in (
            "patient_rows_emitted",
            "patient_identifiers_emitted",
            "raw_values_emitted",
            "small_cells_emitted",
        )
    ):
        _issue(issues, "official_unit_privacy", name, "privacy contract failed")


def _validate_component_policy(
    policy: Mapping[str, Any], issues: list[PreflightIssue]
) -> None:
    name = "PATIENT_ATLAS_COMPONENT_POLICY_V1.json"
    if policy.get("schema_version") != COMPONENT_POLICY_SCHEMA:
        _issue(issues, "component_policy_version", name, "unexpected policy version")
        return
    if policy.get("status") != "frozen_before_official_test":
        _issue(issues, "component_policy_status", name, "component policy is not frozen")
    primary = policy.get("primary_model", {})
    if (
        primary.get("external_blood_anchor_enabled") is not False
        or primary.get("external_tower_output_used_as_primary_input") is not False
        or primary.get("clinical_view")
        != "typed 59-field raw clinical encoder with fold-fit preprocessing"
    ):
        _issue(issues, "component_primary_model", name, "selected primary components changed")
    normalization = policy.get("historical_normalization", {})
    if (
        normalization.get("status") != "unresolved"
        or normalization.get("blocks_selected_primary_model") is not False
        or normalization.get("blocks_external_blood_anchor_sensitivity") is not True
    ):
        _issue(issues, "component_normalization_scope", name, "normalization gate scope changed")
    scope = policy.get("scope", {})
    if (
        scope.get("selected_representation_changed") is not False
        or scope.get("existing_exploratory_vectors_changed") is not False
        or scope.get("future_confirmatory_refit_must_keep_anchor_disabled") is not True
        or scope.get("official_test_accessed") is not False
    ):
        _issue(issues, "component_policy_scope", name, "component policy scope changed")
    bindings = policy.get("artifact_bindings", {})
    expected = {
        "source_policy": SOURCE_POLICY_SHA256,
        "selected_stage2_summary": SELECTED_STAGE2_SUMMARY_SHA256,
        "external_blood_tower_contract": EXTERNAL_BLOOD_CONTRACT_SHA256,
    }
    for key, digest in expected.items():
        if bindings.get(key, {}).get("sha256") != digest:
            _issue(issues, "component_policy_binding", name, f"{key} binding changed")


def _validate_continuous_validity_policy(
    policy: Mapping[str, Any],
    registry: Mapping[str, Any],
    registry_sha256: str,
    context_sha256: str,
    unit_policy_sha256: str,
    issues: list[PreflightIssue],
) -> None:
    name = "PATIENT_ATLAS_CONTINUOUS_VALIDITY_POLICY_V1.json"
    if policy.get("schema_version") != CONTINUOUS_VALIDITY_POLICY_SCHEMA:
        _issue(issues, "continuous_validity_version", name, "unexpected policy version")
        return
    if policy.get("status") != "frozen_for_prospective_refit_before_official_test":
        _issue(issues, "continuous_validity_status", name, "policy is not frozen")
    mask = policy.get("policy_eligible_mask")
    expected_false = {8, 9, 20, 35, 36}
    if (
        not isinstance(mask, list)
        or len(mask) != EXPECTED_FEATURE_COUNT
        or any(type(value) is not bool for value in mask)
        or {index for index, value in enumerate(mask) if not value} != expected_false
    ):
        _issue(issues, "continuous_validity_mask", name, "policy mask must disable exactly five conflicts")
    else:
        computed_mask_hash = _canonical_json_sha256(
            {
                "ordered_features_hash": registry.get("ordered_columns_sha256"),
                "policy_eligible_mask": mask,
            }
        )
        if (
            policy.get("policy_eligible_mask_sha256") != computed_mask_hash
            or computed_mask_hash != EXPECTED_POLICY_MASK_SHA256
        ):
            _issue(issues, "continuous_validity_mask_hash", name, "policy mask hash is invalid")
    admission = policy.get("value_admission", {})
    if (
        admission.get("continuous_field_count") != EXPECTED_CONTINUOUS_COUNT
        or admission.get("unit_authorized_field_count") != 43
        or admission.get("fully_masked_unit_conflict_indices")
        != sorted(expected_false)
        or admission.get("fully_masked_unit_conflict_fields")
        != ["c_peptide", "calcium", "insulin", "urine_albumin", "urine_creatinine"]
        or admission.get("nonfinite_rule") != "missing"
        or admission.get("raw_physiological_clipping") is not False
        or admission.get("raw_value_imputation_before_encoder") is not False
    ):
        _issue(issues, "continuous_validity_admission", name, "value-admission policy changed")
    transform = policy.get("fold_fit_robust_transform", {})
    if (
        transform.get("fit_scope") != "outer representation-training patients only"
        or transform.get("standardized_clip") != [-10.0, 10.0]
        or transform.get("masked_value") != 0.0
        or transform.get("masked_observation_indicator") is not False
        or transform.get("parameters_and_fit_id_hash_serialized") is not True
    ):
        _issue(issues, "continuous_validity_transform", name, "robust transform policy changed")
    decision = policy.get("decision", {})
    if (
        decision.get("field_specific_plausibility_ranges_required_for_representation_training")
        is not False
        or decision.get("raw_clinical_validity_claim_authorized") is not False
        or decision.get("clinical_decision_support_claim_authorized") is not False
        or decision.get("prospective_refit_required") is not True
        or decision.get("prospective_refit_completed") is not False
        or decision.get("existing_exploratory_model_changed") is not False
        or decision.get("official_test_accessed") is not False
    ):
        _issue(issues, "continuous_validity_decision", name, "decision boundary changed")
    bindings = policy.get("artifact_bindings", {})
    if (
        bindings.get("feature_registry", {}).get("sha256") != registry_sha256
        or bindings.get("context_schema", {}).get("sha256") != context_sha256
        or bindings.get("official_unit_reconciliation", {}).get("sha256")
        != unit_policy_sha256
        or unit_policy_sha256 != OFFICIAL_UNIT_POLICY_SHA256
        or bindings.get("preprocessing_code", {}).get("sha256")
        != PREPROCESSING_CODE_SHA256
    ):
        _issue(issues, "continuous_validity_binding", name, "policy artifact binding changed")


def _real_data_readiness(
    root: Path,
    feature_registry: Mapping[str, Any],
    eye_registry: Mapping[str, Any],
    clinical_field_contract: Mapping[str, Any],
    official_unit_policy: Mapping[str, Any],
    component_policy: Mapping[str, Any],
    continuous_validity_policy: Mapping[str, Any],
    issues: list[PreflightIssue],
    digests: dict[str, str],
) -> None:
    feature_name = "PATIENT_ATLAS_FEATURE_REGISTRY.json"
    gaps = feature_registry.get("blocking_gaps")
    visit_policy_resolved = (
        clinical_field_contract.get("status_summary", {}).get(
            "visit_and_replicate_policy_resolved_count"
        )
        == EXPECTED_CONTINUOUS_COUNT
    )
    if feature_registry.get("real_training_ready") is not True:
        if isinstance(gaps, list) and gaps:
            for gap in gaps:
                if isinstance(gap, dict):
                    gap_id = str(gap.get("id", "unnamed"))
                    if gap_id == "visit_and_duplicate_policy" and visit_policy_resolved:
                        continue
                    if (
                        gap_id == "continuous_units_and_ranges"
                        and official_unit_policy.get("status_summary", {}).get(
                            "unit_metadata_gate_resolved_by_masking_conflicts"
                        )
                        is True
                        and continuous_validity_policy.get("decision", {}).get(
                            "field_specific_plausibility_ranges_required_for_representation_training"
                        )
                        is False
                    ):
                        continue
                    if (
                        gap_id == "historical_pretraining_normalization"
                        and component_policy.get("primary_model", {}).get(
                            "external_blood_anchor_enabled"
                        )
                        is False
                        and component_policy.get("historical_normalization", {}).get(
                            "blocks_selected_primary_model"
                        )
                        is False
                    ):
                        continue
                    detail = str(
                        gap.get("required_resolution", gap.get("detail", "unresolved"))
                    )
                    if gap_id == "continuous_units_and_ranges":
                        summary = official_unit_policy.get("status_summary", {})
                        detail = (
                            "official unit reconciliation authorizes "
                            f"{summary.get('canonical_unit_authorized_count', 0)}/48 fields "
                            "and fully masks five conflicts; 0/48 field-specific plausibility "
                            "ranges are authorized, so the validity policy remains incomplete"
                        )
                    _issue(
                        issues,
                        f"data_gap:{gap_id}",
                        feature_name,
                        detail,
                    )
        else:
            _issue(issues, "feature_registry_not_ready", feature_name, "real_training_ready is not true")

    # Current v1 safely assigns unknown/default laterality and quality. The only
    # eye-registry gap that blocks reuse of the adapted tower is unresolved
    # source-cohort overlap/provenance.
    tower_contract = eye_registry.get("tower_contract", {})
    supplement_passed = _validate_eye_provenance_supplement(root, issues, digests)
    if (
        tower_contract.get("per_file_identity_manifest_available") is not True
        and not supplement_passed
    ):
        _issue(
            issues,
            "data_gap:eye_pretraining_overlap",
            "PATIENT_ATLAS_EYE_REGISTRY.json",
            "source tags and counts are authenticated, but the historical adaptation cache lacks a per-file identity/content manifest",
        )


def _validate_eye_provenance_supplement(
    root: Path,
    issues: list[PreflightIssue],
    digests: dict[str, str],
) -> bool:
    """Accept a local aggregate overlap audit without rewriting legacy contracts."""

    name = "PATIENT_ATLAS_EYE_SOURCE_PROVENANCE_RECONSTRUCTION_V2.json"
    path = root / name
    if not path.is_file():
        return False
    try:
        report, digest = _load_json(path)
    except (OSError, ValueError) as exc:
        _issue(issues, "eye_provenance_supplement", name, str(exc))
        return False
    digests[name] = digest
    reconstruction = report.get("reconstruction", {})
    overlap = report.get("ai_readi_overlap_audit", {})
    gates = report.get("gates", {})
    bindings = report.get("bindings", {})
    privacy = report.get("privacy", {})
    manifest_name = bindings.get("manifest_file")
    if not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name:
        _issue(issues, "eye_provenance_supplement", name, "manifest name is unsafe")
        return False
    manifest_path = root / "exploratory_artifacts" / manifest_name
    try:
        manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        audit_digest = hashlib.sha256(
            (root / "patient_atlas_eye_source_provenance.py").read_bytes()
        ).hexdigest()
    except OSError as exc:
        _issue(issues, "eye_provenance_supplement", name, str(exc))
        return False
    digests[str(manifest_path.relative_to(root))] = manifest_digest
    base_valid = bool(
        report.get("schema_version")
        == "patient-atlas-eye-source-provenance-reconstruction-v1"
        and reconstruction.get("source_counts")
        == {"drunified": 92501, "brset": 16266, "odir": 4512, "jsiec": 1994}
        and reconstruction.get("total_files") == 115273
        and reconstruction.get("cache_rows_pixel_exact") == 115273
        and reconstruction.get("all_cache_rows_pixel_exact") is True
        and reconstruction.get("per_file_identity_manifest_available") is True
        and overlap.get("ai_readi_cfp_images") == 50315
        and overlap.get("any_exact_decoded_pixel_overlap") is False
        and overlap.get("patient_identifiers_compared") is False
        and overlap.get("subject_identity_overlap_proven") is False
        and gates.get("per_file_manifest_reconstruction_passed") is True
        and gates.get("exact_image_overlap_gate_passed") is True
        and gates.get("strongest_subject_identity_claim_authorized") is False
        and bindings.get("manifest_file_sha256") == manifest_digest
        and bindings.get("audit_code_sha256") == audit_digest
        and privacy.get("patient_rows_emitted") is False
        and privacy.get("patient_identifiers_emitted") is False
        and privacy.get("source_paths_emitted") is False
        and privacy.get("images_emitted") is False
        and privacy.get("ai_readi_hashes_emitted") is False
    )
    if not base_valid:
        _issue(
            issues,
            "eye_provenance_supplement",
            name,
            "aggregate reconstruction, exact-overlap, binding, or privacy gates failed",
        )
        return False

    perceptual_passed = bool(
        overlap.get("any_dual_hash_hamming_le_4_candidate") is False
        and gates.get("perceptual_overlap_screen_passed") is True
    )
    if perceptual_passed:
        return True

    suppressed_candidate_cell = bool(
        overlap.get("any_dual_hash_hamming_le_4_candidate") is True
        and overlap.get("dual_hash_hamming_le_4_candidate_count") == "<10"
        and overlap.get("perceptual_candidate_count_small_cell_suppressed") is True
        and overlap.get("minimum_candidate_mean_absolute_pixel_difference") is None
        and gates.get("perceptual_overlap_screen_passed") is False
    )
    if not suppressed_candidate_cell:
        _issue(
            issues,
            "eye_provenance_supplement",
            name,
            "perceptual screen did not pass and is not eligible for small-cell adjudication",
        )
        return False
    return _validate_eye_candidate_adjudication(
        root=root,
        issues=issues,
        digests=digests,
        primary_report_digest=digest,
        manifest_digest=manifest_digest,
        primary_decoder_digest=bindings.get("ai_readi_decoder_sha256"),
    )


def _validate_eye_candidate_adjudication(
    *,
    root: Path,
    issues: list[PreflightIssue],
    digests: dict[str, str],
    primary_report_digest: str,
    manifest_digest: str,
    primary_decoder_digest: Any,
) -> bool:
    """Validate fail-closed local geometric adjudication of a suppressed cell."""

    name = "PATIENT_ATLAS_EYE_CANDIDATE_ADJUDICATION_V1.json"
    path = root / name
    if not path.is_file():
        return False
    try:
        report, digest = _load_json(path)
        audit_digest = hashlib.sha256(
            (root / "patient_atlas_eye_candidate_adjudication.py").read_bytes()
        ).hexdigest()
    except (OSError, ValueError) as exc:
        _issue(issues, "eye_candidate_adjudication", name, str(exc))
        return False
    digests[name] = digest
    calibration = report.get("external_control_calibration", {})
    adjudication = report.get("ai_readi_candidate_adjudication", {})
    gates = report.get("gates", {})
    bindings = report.get("bindings", {})
    privacy = report.get("privacy", {})
    decision = report.get("decision_contract", {})
    valid = bool(
        report.get("schema_version")
        == "patient-atlas-eye-candidate-adjudication-v1"
        and report.get("scope")
        == "suppressed_perceptual_candidate_geometric_adjudication"
        and calibration.get("positive_control_count") == 40
        and calibration.get("synthetic_nonmatch_control_count") == 40
        and calibration.get("minimum_required_positive_sensitivity") == 0.9
        and calibration.get("positive_sensitivity_gate_passed") is True
        and calibration.get("synthetic_nonmatch_zero_false_positive_gate_passed")
        is True
        and calibration.get("calibration_passed") is True
        and calibration.get("control_scores_emitted") is False
        and adjudication.get("primary_candidate_set_reproduced") is True
        and adjudication.get("candidate_cell_small_cell_suppressed") is True
        and adjudication.get("candidate_count_emitted") is False
        and adjudication.get("candidate_scores_emitted") is False
        and adjudication.get("any_geometrically_confirmed_near_duplicate") is False
        and adjudication.get("geometric_adjudication_passed") is True
        and adjudication.get("subject_identity_overlap_proven") is False
        and gates.get("adjudicator_calibration_passed") is True
        and gates.get("no_confirmed_near_duplicate_gate_passed") is True
        and gates.get("image_level_overlap_provenance_gate_passed") is True
        and gates.get("strongest_subject_identity_claim_authorized") is False
        and decision.get("thresholds_frozen_before_ai_readi_candidate_adjudication")
        is True
        and decision.get("horizontal_reflection_checked") is True
        and bindings.get("primary_report_file_sha256") == primary_report_digest
        and bindings.get("manifest_file_sha256") == manifest_digest
        and bindings.get("ai_readi_decoder_sha256") == primary_decoder_digest
        and bindings.get("audit_code_sha256") == audit_digest
        and bindings.get("opencv_version") == "4.11.0"
        and bindings.get("decision_contract_sha256")
        == _canonical_json_sha256(decision)
        and privacy.get("patient_rows_emitted") is False
        and privacy.get("patient_identifiers_emitted") is False
        and privacy.get("source_paths_emitted") is False
        and privacy.get("images_emitted") is False
        and privacy.get("candidate_hashes_emitted") is False
        and privacy.get("candidate_pair_metrics_emitted") is False
        and privacy.get("small_cell_threshold") == 10
    )
    if not valid:
        _issue(
            issues,
            "eye_candidate_adjudication",
            name,
            "calibration, geometric adjudication, binding, or privacy gates failed",
        )
    return valid


def _confirmatory_readiness(
    comparison_policy: Mapping[str, Any],
    refit_attestation: Mapping[str, Any],
    issues: list[PreflightIssue],
) -> None:
    """Decision policy is already validated; concat has no confirmatory gate."""

    if comparison_policy.get("concat_reference", {}).get(
        "missing_margin_blocks_confirmatory_evaluation"
    ) is not False:
        _issue(
            issues,
            "comparison_policy_not_frozen",
            "PATIENT_ATLAS_COMPARISON_POLICY_V1.json",
            "descriptive concat must not introduce an undeclared confirmatory gate",
        )
    if refit_attestation.get("decision", {}).get(
        "prospective_refit_completed"
    ) is not True:
        _issue(
            issues,
            "prospective_refit_required",
            "PATIENT_ATLAS_CONTINUOUS_VALIDITY_POLICY_V1.json",
            "fit and freeze the masked 43-field prospective representation before official-test scoring",
        )


def _validate_prospective_refit_attestation(
    root: Path,
    attestation: Mapping[str, Any],
    *,
    stage: PreflightStage,
    continuous_policy_sha256: str,
    official_unit_sha256: str,
    comparison_policy_sha256: str,
    issues: list[PreflightIssue],
    digests: dict[str, str],
) -> None:
    name = "PATIENT_ATLAS_PROSPECTIVE_REFIT_ATTESTATION_V1.json"
    if attestation.get("schema_version") != PROSPECTIVE_REFIT_ATTESTATION_SCHEMA:
        _issue(issues, "prospective_refit_version", name, "unexpected attestation version")
        return
    if (
        attestation.get("status")
        != "masked_prospective_development_refit_complete_test_sealed"
    ):
        _issue(issues, "prospective_refit_status", name, "refit is not complete and sealed")
    bindings = attestation.get("artifact_bindings", {})
    expected_policy_bindings = {
        "continuous_validity_policy": continuous_policy_sha256,
        "official_unit_reconciliation": official_unit_sha256,
        "comparison_policy": comparison_policy_sha256,
    }
    for role, digest in expected_policy_bindings.items():
        if bindings.get(role, {}).get("sha256") != digest:
            _issue(issues, "prospective_refit_policy_binding", name, f"{role} hash differs")

    refit = attestation.get("refit", {})
    if (
        refit.get("policy_mask_hash") != EXPECTED_POLICY_MASK_SHA256
        or refit.get("eligible_clinical_fields") != 54
        or refit.get("masked_unit_conflict_indices") != [8, 9, 20, 35, 36]
        or refit.get("candidate_count") != 1
        or refit.get("beta") != 0.01
        or refit.get("group_shrinkage_rate") != 1e-6
        or not isinstance(refit.get("selected_step"), int)
        or not isinstance(refit.get("model_state_sha256"), str)
        or len(refit.get("model_state_sha256", "")) != 64
    ):
        _issue(issues, "prospective_refit_configuration", name, "refit configuration differs")
    superiority = attestation.get("development_superiority", {})
    if (
        superiority.get("primary_rule_passed") is not True
        or superiority.get("blood_minus_both_loss_difference", 0.0) <= 0.0
        or superiority.get("eye_minus_both_loss_difference", 0.0) <= 0.0
        or superiority.get("holm_family_wise_alpha") != 0.05
        or superiority.get("development_not_confirmatory") is not True
    ):
        _issue(issues, "prospective_refit_development_gate", name, "development gate failed")
    probabilistic = attestation.get("probabilistic_state", {})
    if (
        probabilistic.get("posterior_mean_dimension") != 64
        or probabilistic.get("posterior_log_variance_sidecar_dimension") != 64
        or probabilistic.get("paired_evidence_vector_dimension") != 129
        or probabilistic.get("observable_conformal_families_fitted") != 6
        or probabilistic.get("latent_variance_absolute_coverage_claim_allowed")
        is not False
    ):
        _issue(issues, "prospective_refit_probability", name, "probability contract differs")
    missingness = attestation.get("missingness", {})
    if (
        missingness.get("pattern_count") != 9
        or missingness.get("eye_uncertainty_monotone") is not True
        or missingness.get("blood_clinical_uncertainty_monotone") is not True
    ):
        _issue(issues, "prospective_refit_missingness", name, "missingness gate failed")
    interpretation = attestation.get("interpretability", {})
    if (
        interpretation.get("anchored_eye_concepts") != 45
        or interpretation.get("anchored_blood_clinical_concepts") != 54
        or interpretation.get("anchored_vector_dimension") != 100
        or interpretation.get("interpretation_sidecar_not_predictive_default")
        is not True
    ):
        _issue(issues, "prospective_refit_interpretability", name, "interpretability contract differs")
    decision = attestation.get("decision", {})
    if (
        decision.get("prospective_refit_completed") is not True
        or decision.get("official_test_accessed") is not False
        or decision.get("official_test_scoring_authorized") is not False
        or decision.get("clinical_benefit_claim_allowed") is not False
    ):
        _issue(issues, "prospective_refit_decision", name, "decision boundary differs")
    privacy = attestation.get("privacy", {})
    if any(
        privacy.get(field) is not False
        for field in (
            "patient_rows_emitted",
            "patient_identifiers_emitted",
            "patient_vectors_emitted",
            "per_patient_predictions_emitted",
        )
    ):
        _issue(issues, "prospective_refit_privacy", name, "attestation exposes patient material")

    if stage is not PreflightStage.CONFIRMATORY:
        return
    required_roles = (
        "preprocessor",
        "stage2_summary",
        "checkpoint",
        "paired_vector_schema",
        "development_validation",
        "missingness_audit",
        "factor_interpretability",
        "anchored_concept_audit",
        "anchored_concept_schema",
        "anchored_concept_development",
    )
    for role in required_roles:
        binding = bindings.get(role, {})
        relative = binding.get("file")
        expected = binding.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            _issue(issues, "prospective_refit_artifact_binding", name, f"{role} binding is missing")
            continue
        path = root / relative
        try:
            resolved = path.resolve()
            resolved.relative_to(root.resolve())
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            _issue(issues, "prospective_refit_artifact_unreadable", name, f"{role}: {exc}")
            continue
        digests[relative] = digest
        if digest != expected:
            _issue(issues, "prospective_refit_artifact_hash", name, f"{role} hash differs")


def _validate_exploratory_policy(
    root: Path,
    issues: list[PreflightIssue],
    digests: dict[str, str],
) -> None:
    """Authenticate the explicit nonconfirmatory source policy and row-free audit."""

    policy_name = "PATIENT_ATLAS_SOURCE_POLICY.json"
    try:
        policy, policy_digest = _load_json(root / policy_name)
    except (OSError, ValueError) as exc:
        _issue(issues, "exploratory_policy_unreadable", policy_name, str(exc))
        return
    digests[policy_name] = policy_digest
    if policy.get("schema_version") != SOURCE_POLICY_SCHEMA:
        _issue(issues, "exploratory_policy_version", policy_name, "unexpected schema version")
    if policy.get("scope") != "exploratory_train_validation_only":
        _issue(issues, "exploratory_scope", policy_name, "scope must be train/validation only")
    if policy.get("confirmatory_ready") is not False:
        _issue(issues, "exploratory_confirmatory_label", policy_name, "exploratory policy must remain nonconfirmatory")

    binding = policy.get("binding_audit", {})
    audit_name = binding.get("file")
    code_name = binding.get("audit_code")
    if not isinstance(audit_name, str) or not isinstance(code_name, str):
        _issue(issues, "exploratory_binding", policy_name, "audit file and code must be named")
        return
    try:
        audit, audit_digest = _load_json(root / audit_name)
        code_raw = (root / code_name).read_bytes()
    except (OSError, ValueError) as exc:
        _issue(issues, "exploratory_binding_unreadable", policy_name, str(exc))
        return
    code_digest = hashlib.sha256(code_raw).hexdigest()
    digests[audit_name] = audit_digest
    digests[code_name] = code_digest
    if binding.get("file_sha256") != audit_digest:
        _issue(issues, "exploratory_audit_hash", policy_name, "raw-source audit hash mismatch")
    if binding.get("audit_code_sha256") != code_digest:
        _issue(issues, "exploratory_audit_code_hash", policy_name, "raw-audit code hash mismatch")
    if audit.get("schema_version") != RAW_AUDIT_SCHEMA:
        _issue(issues, "exploratory_audit_version", audit_name, "unexpected audit schema")

    cohort = policy.get("cohort", {})
    if cohort.get("exploratory_allowed_splits") != ["train", "val"] or "sealed" not in str(
        cohort.get("test_target_policy", "")
    ):
        _issue(issues, "exploratory_test_seal", policy_name, "test outcomes must remain sealed")
    anchor = policy.get("blood_anchor", {})
    if (
        anchor.get("primary_exploratory_setting") != "disabled with an exact zero anchor"
        or anchor.get("sensitivity_allowed") is not False
    ):
        _issue(issues, "exploratory_anchor_disabled", policy_name, "uncontracted blood anchor must be disabled")

    source_hashes = policy.get("source_hashes", {})
    if not isinstance(source_hashes, Mapping) or not source_hashes or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in source_hashes.values()
    ):
        _issue(issues, "exploratory_source_hashes", policy_name, "source hashes must be complete SHA-256 digests")

    eye = policy.get("eye_alignment", {})
    eye_assertions = eye.get("manifest_join_assertions", {})
    row_proof = eye.get("historical_metadata_row_proof", {})
    if (
        eye.get("embedding_shape") != [50315, 384]
        or eye.get("embedding_nonfinite_rows") != 0
        or eye.get("embedding_zero_rows") != 0
        or eye_assertions.get("manifest_physical_basename_sets_equal") is not True
        or row_proof.get("metadata_row_order_matches_current_encoder_glob") is not True
    ):
        _issue(issues, "exploratory_eye_alignment", policy_name, "eye row proof is incomplete")

    privacy = policy.get("privacy", {})
    if any(
        privacy.get(field) is not False
        for field in (
            "patient_rows_in_artifact",
            "patient_identifiers_in_artifact",
            "per_patient_embeddings_or_predictions_in_artifact",
        )
    ):
        _issue(issues, "exploratory_policy_privacy", policy_name, "policy artifact contains prohibited patient material")
    audit_privacy = audit.get("privacy", {})
    if (
        audit_privacy.get("patient_rows_emitted") is not False
        or audit_privacy.get("patient_identifiers_emitted") is not False
        or audit_privacy.get("functional_target_values_read") is not False
    ):
        _issue(issues, "exploratory_audit_privacy", audit_name, "audit privacy contract failed")
    if (
        audit.get("measurements", {}).get("present_feature_count") != 48
        or audit.get("conditions", {}).get("condition_count") != 11
        or audit.get("retinal", {}).get("embedding_rows_match_sorted_glob") is not True
        or audit.get("retinal", {}).get("manifest_physical_basename_sets_equal") is not True
    ):
        _issue(issues, "exploratory_audit_coverage", audit_name, "raw-source coverage assertions failed")


def assess_preflight(
    root: str | Path,
    stage: PreflightStage | str = PreflightStage.SYNTHETIC,
) -> PreflightReport:
    """Validate schema consistency and readiness without opening patient data."""

    root = Path(root)
    stage = PreflightStage(stage)
    filenames = (
        "PATIENT_ATLAS_FEATURE_REGISTRY.json",
        "PATIENT_ATLAS_CONTEXT_SCHEMA.json",
        "PATIENT_ATLAS_EYE_REGISTRY.json",
        "PATIENT_ATLAS_TARGET_MANIFEST.json",
        "EXTERNAL_BLOOD_TOWER_CONTRACT.json",
        "EXTERNAL_EYE_TOWER_CONTRACT.json",
        "PATIENT_ATLAS_CLINICAL_FIELD_CONTRACT_V1.json",
        "PATIENT_ATLAS_COMPARISON_POLICY_V1.json",
        "PATIENT_ATLAS_OFFICIAL_UNIT_RECONCILIATION_V1.json",
        "PATIENT_ATLAS_COMPONENT_POLICY_V1.json",
        "PATIENT_ATLAS_CONTINUOUS_VALIDITY_POLICY_V1.json",
        "PATIENT_ATLAS_PROSPECTIVE_REFIT_ATTESTATION_V1.json",
    )
    documents: dict[str, dict[str, Any]] = {}
    digests: dict[str, str] = {}
    issues: list[PreflightIssue] = []
    for filename in filenames:
        try:
            document, digest = _load_json(root / filename)
        except (OSError, ValueError) as exc:
            _issue(issues, "manifest_unreadable", filename, str(exc))
            continue
        documents[filename] = document
        digests[filename] = digest

    if len(documents) != len(filenames):
        return PreflightReport(stage.value, False, tuple(issues), digests)

    features = documents[filenames[0]]
    context = documents[filenames[1]]
    eye = documents[filenames[2]]
    targets = documents[filenames[3]]
    contract = documents[filenames[4]]
    eye_contract = documents[filenames[5]]
    clinical_field_contract = documents[filenames[6]]
    comparison_policy = documents[filenames[7]]
    official_unit_policy = documents[filenames[8]]
    component_policy = documents[filenames[9]]
    continuous_validity_policy = documents[filenames[10]]
    prospective_refit_attestation = documents[filenames[11]]
    _validate_blood_contract(contract, issues)
    _validate_eye_contract(eye_contract, issues)
    _validate_feature_registry(features, contract, issues)
    _validate_context(context, issues)
    _validate_eye_registry(eye, eye_contract, issues)
    _validate_targets(targets, issues)
    _validate_comparison_policy(
        comparison_policy,
        digests[filenames[3]],
        issues,
    )
    _validate_official_unit_reconciliation(
        official_unit_policy,
        features,
        digests[filenames[0]],
        digests[filenames[6]],
        issues,
    )
    _validate_component_policy(component_policy, issues)
    _validate_continuous_validity_policy(
        continuous_validity_policy,
        features,
        digests[filenames[0]],
        digests[filenames[1]],
        digests[filenames[8]],
        issues,
    )
    _validate_clinical_field_contract(
        clinical_field_contract,
        features,
        digests[filenames[0]],
        issues,
    )
    _validate_prospective_refit_attestation(
        root,
        prospective_refit_attestation,
        stage=stage,
        continuous_policy_sha256=digests[filenames[10]],
        official_unit_sha256=digests[filenames[8]],
        comparison_policy_sha256=digests[filenames[7]],
        issues=issues,
        digests=digests,
    )

    if stage is PreflightStage.EXPLORATORY:
        _validate_exploratory_policy(root, issues, digests)
    if stage in (PreflightStage.REAL_REPRESENTATION, PreflightStage.CONFIRMATORY):
        _real_data_readiness(
            root,
            features,
            eye,
            clinical_field_contract,
            official_unit_policy,
            component_policy,
            continuous_validity_policy,
            issues,
            digests,
        )
    if stage is PreflightStage.CONFIRMATORY:
        _confirmatory_readiness(
            comparison_policy,
            prospective_refit_attestation,
            issues,
        )

    return PreflightReport(stage.value, not issues, tuple(issues), digests)


def require_preflight(
    root: str | Path,
    stage: PreflightStage | str,
) -> PreflightReport:
    """Return a valid report or raise before any real-data loader can be called."""

    report = assess_preflight(root, stage)
    if not report.ready:
        codes = ", ".join(issue.code for issue in report.blocking_issues)
        raise RuntimeError(f"Patient Atlas {report.stage} preflight failed: {codes}")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--stage",
        choices=[stage.value for stage in PreflightStage],
        default=PreflightStage.SYNTHETIC.value,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = assess_preflight(args.root, args.stage)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
