"""Row-free validation for the Patient Atlas V5 disease-utility registry.

This module deliberately inspects only JSON contracts and source-code hashes.  It does
not load cohort tables, labels, embeddings, predictions, or any patient-level artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "patient-atlas-v5-disease-utility-benchmark-registry-v1"
MANDATORY_COMPARATORS = {
    "patient_atlas_v5",
    "plain_concat",
    "dinov3_generic",
    "retfound_mae_nature_cfp",
    "retfound_green",
    "visionfm_fundus",
    "external_denoise30_blood_tower",
    "labrador",
    "raw_target_safe_clinical_xgboost",
    "raw_target_safe_clinical_elastic_net",
}
FOUNDATION_COMPARATOR_KINDS = {
    "general_vision_foundation_encoder",
    "retinal_foundation_encoder",
    "ophthalmic_foundation_encoder",
    "retinal_vision_language_foundation_encoder",
    "laboratory_foundation_encoder",
    "tabular_foundation_readout_sensitivity",
}
VALID_LANES = {
    "reference_test_hidden_case_finding",
    "standard_marker_increment",
    "universal_mask_breadth",
}
EXPECTED_READOUT_PENALTY_GRID = (
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
)


class RegistryValidationError(ValueError):
    """Raised when the benchmark registry violates a scientific contract."""


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RegistryValidationError(f"{path} must contain a JSON object")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_unique_ids(items: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for item in items:
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise RegistryValidationError(f"{label} item has an invalid id")
        if item_id in result:
            raise RegistryValidationError(f"duplicate {label} id: {item_id}")
        result[item_id] = item
    return result


def _string_set(values: Any, label: str) -> set[str]:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise RegistryValidationError(f"{label} must be a list of strings")
    if len(values) != len(set(values)):
        raise RegistryValidationError(f"{label} contains duplicates")
    return set(values)


def validate_registry_payload(
    registry: Mapping[str, Any],
    feature_registry: Mapping[str, Any],
    unit_reconciliation: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Validate scientific and schema invariants without touching patient data."""

    if registry.get("schema_version") != SCHEMA_VERSION:
        raise RegistryValidationError("unexpected disease-utility registry schema_version")

    scope = registry.get("scope", {})
    for forbidden_flag in (
        "longitudinal_claims_allowed",
        "external_validation_claims_allowed",
        "clinical_deployment_claims_allowed",
    ):
        if scope.get(forbidden_flag) is not False:
            raise RegistryValidationError(f"scope.{forbidden_flag} must be false")
    if scope.get("unit_of_analysis") != "one row per patient":
        raise RegistryValidationError("unit_of_analysis must be one row per patient")

    feature_items = feature_registry.get("features")
    if not isinstance(feature_items, list):
        raise RegistryValidationError("feature registry has no features list")
    feature_names = {item.get("name") for item in feature_items if isinstance(item, dict)}
    if None in feature_names or len(feature_names) != len(feature_items):
        raise RegistryValidationError("feature registry names are missing or duplicated")

    global_policy = registry.get("global_clinical_policy", {})
    global_exclusions = _string_set(
        global_policy.get("globally_excluded_fields"),
        "global clinical policy excluded fields",
    )
    unknown_global = global_exclusions - feature_names
    if unknown_global:
        raise RegistryValidationError(
            f"global clinical policy contains unknown fields: {sorted(unknown_global)}"
        )
    if global_policy.get("existing_vectors_retroactively_changed") is not False:
        raise RegistryValidationError("unit reconciliation may not rewrite existing vectors")
    if unit_reconciliation is not None:
        if (
            unit_reconciliation.get("schema_version")
            != "patient-atlas-official-unit-reconciliation-v1"
        ):
            raise RegistryValidationError("official unit reconciliation schema differs")
        blocked_by_unit = {
            str(field.get("name"))
            for field in unit_reconciliation.get("fields", [])
            if field.get("canonical_unit_authorized") is False
        }
        if global_exclusions != blocked_by_unit:
            raise RegistryValidationError(
                "global clinical policy is not the exact fail-closed unit-conflict set; "
                f"registry={sorted(global_exclusions)}, reconciliation={sorted(blocked_by_unit)}"
            )
        authorized_count = sum(
            field.get("canonical_unit_authorized") is True
            for field in unit_reconciliation.get("fields", [])
        )
        if global_policy.get("authorized_unit_field_count") != authorized_count:
            raise RegistryValidationError("authorized unit field count differs from reconciliation")

    lanes = _require_unique_ids(registry.get("evaluation_lanes", []), "evaluation lane")
    if set(lanes) != VALID_LANES:
        raise RegistryValidationError("evaluation lane set is not the frozen three-lane design")

    profiles = _require_unique_ids(registry.get("feature_exclusion_profiles", []), "exclusion profile")
    profile_fields: dict[str, set[str]] = {}
    for profile_id, profile in profiles.items():
        fields = _string_set(profile.get("fields"), f"profile {profile_id} fields")
        unknown = fields - feature_names
        if unknown:
            raise RegistryValidationError(
                f"profile {profile_id} contains unknown clinical fields: {sorted(unknown)}"
            )
        profile_fields[profile_id] = fields

    core = _require_unique_ids(registry.get("core_endpoints", []), "core endpoint")
    for endpoint_id, endpoint in core.items():
        lane_id = endpoint.get("primary_lane")
        if lane_id not in lanes:
            raise RegistryValidationError(f"core endpoint {endpoint_id} references an unknown lane")
        profile_id = endpoint.get("v5_exclusion_profile")
        if profile_id not in profile_fields:
            raise RegistryValidationError(f"core endpoint {endpoint_id} references an unknown profile")
        defining = _string_set(
            endpoint.get("defining_input_fields"), f"core endpoint {endpoint_id} defining fields"
        )
        unknown = defining - feature_names
        if unknown:
            raise RegistryValidationError(
                f"core endpoint {endpoint_id} has unknown defining fields: {sorted(unknown)}"
            )
        missing = defining - profile_fields[profile_id]
        if missing:
            raise RegistryValidationError(
                f"core endpoint {endpoint_id} leaks defining fields outside its V5 profile: {sorted(missing)}"
            )

        standard = endpoint.get("reference_or_standard_care", {})
        if lane_id == "reference_test_hidden_case_finding":
            if standard.get("role") != "target_or_reference_not_predictor":
                raise RegistryValidationError(
                    f"case-finding endpoint {endpoint_id} must treat the reference test as target, not predictor"
                )
        elif lane_id == "standard_marker_increment":
            if standard.get("role") != "separate_baseline_covariate":
                raise RegistryValidationError(
                    f"increment endpoint {endpoint_id} must declare a separate marker baseline"
                )
            marker_fields = _string_set(
                standard.get("marker_fields"), f"core endpoint {endpoint_id} marker fields"
            )
            unmasked_markers = marker_fields - profile_fields[profile_id]
            if unmasked_markers:
                raise RegistryValidationError(
                    f"increment endpoint {endpoint_id} leaves standard markers inside V5: {sorted(unmasked_markers)}"
                )
            caveat = endpoint.get("ascertainment_caveat")
            source_url = endpoint.get("ascertainment_source_url")
            if not isinstance(caveat, str) or "not" not in caveat.lower():
                raise RegistryValidationError(
                    f"increment endpoint {endpoint_id} lacks a claim-limiting ascertainment caveat"
                )
            if not isinstance(source_url, str) or not source_url.startswith("https://"):
                raise RegistryValidationError(
                    f"increment endpoint {endpoint_id} lacks an ascertainment source"
                )

    breadth = _require_unique_ids(registry.get("breadth_outcomes", []), "breadth outcome")
    if len(breadth) != 18:
        raise RegistryValidationError("the frozen breadth panel must contain exactly 18 outcomes")
    universal = profile_fields.get("circularity18_universal")
    if universal is None:
        raise RegistryValidationError("circularity18_universal profile is required")
    breadth_union: set[str] = set()
    for outcome_id, outcome in breadth.items():
        defining = _string_set(
            outcome.get("defining_input_fields"), f"breadth outcome {outcome_id} defining fields"
        )
        unknown = defining - feature_names
        if unknown:
            raise RegistryValidationError(
                f"breadth outcome {outcome_id} has unknown defining fields: {sorted(unknown)}"
            )
        breadth_union.update(defining)
    if breadth_union != universal:
        missing = breadth_union - universal
        extra = universal - breadth_union
        raise RegistryValidationError(
            "universal breadth profile is not the exact union of outcome exclusions; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    comparators = _require_unique_ids(registry.get("comparator_registry", []), "comparator")
    missing_comparators = MANDATORY_COMPARATORS - set(comparators)
    if missing_comparators:
        raise RegistryValidationError(
            f"mandatory comparator entries are absent: {sorted(missing_comparators)}"
        )
    for comparator_id in MANDATORY_COMPARATORS:
        if comparators[comparator_id].get("required") is not True:
            raise RegistryValidationError(f"mandatory comparator {comparator_id} is not required")
    for comparator_id, comparator in comparators.items():
        if comparator.get("kind") not in FOUNDATION_COMPARATOR_KINDS:
            continue
        weights = comparator.get("public_weights")
        if comparator.get("required") is True:
            if not isinstance(weights, str) or weights in {"unavailable", "unknown"}:
                raise RegistryValidationError(
                    f"required foundation comparator {comparator_id} lacks obtainable weights"
                )
            source_url = comparator.get("source_url")
            if not isinstance(source_url, str) or not source_url.startswith("https://"):
                raise RegistryValidationError(
                    f"required foundation comparator {comparator_id} lacks an HTTPS primary source"
                )

    protocol = registry.get("fair_comparison_protocol", {})
    split = protocol.get("split", {})
    if split.get("patient_level") is not True or split.get("same_folds_for_all_arms") is not True:
        raise RegistryValidationError("comparison protocol must use common patient-level folds")
    representation = protocol.get("representation_fitting", {})
    required_true = (
        "one_fold_specific_model_transforms_outer_train_and_outer_test",
        "outer_test_never_used_for_representation_fit",
        "outcome_defining_inputs_forbidden_from_encoder",
        "outcome_defining_inputs_forbidden_from_reconstruction_targets",
        "outcome_defining_inputs_physically_zeroed_before_frozen_tower_calls",
        "fold_latent_coordinates_never_pooled_across_models",
    )
    for field in required_true:
        if representation.get(field) is not True:
            raise RegistryValidationError(f"representation_fitting.{field} must be true")
    readouts = protocol.get("readouts", {})
    if int(readouts.get("inner_folds", 0)) != 5:
        raise RegistryValidationError("disease readout must use five inner folds")
    if tuple(float(value) for value in readouts.get("penalty_grid", ())) != (
        EXPECTED_READOUT_PENALTY_GRID
    ):
        raise RegistryValidationError("disease readout penalty grid differs from the freeze")
    if readouts.get("exact_tie_break") != "larger penalty":
        raise RegistryValidationError("disease readout tie break differs from the freeze")

    gate = registry.get("execution_gate", {})
    if gate.get("ready") is not False or not gate.get("blockers"):
        raise RegistryValidationError("execution gate must fail closed while blockers remain")
    privacy = registry.get("privacy", {})
    if privacy.get("patient_derived_processing") != "local_only":
        raise RegistryValidationError("patient-derived processing must remain local only")
    if privacy.get("minimum_disclosable_cell_count") != 10:
        raise RegistryValidationError("small-cell disclosure threshold must remain 10")

    return {
        "feature_count": len(feature_names),
        "exclusion_profile_count": len(profiles),
        "core_endpoint_count": len(core),
        "breadth_outcome_count": len(breadth),
        "comparator_count": len(comparators),
        "mandatory_comparator_count": len(MANDATORY_COMPARATORS),
    }


def _resolve_contract_path(base_dir: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else base_dir / path


def validate_registry_files(
    registry_path: str | Path,
    *,
    verify_historical_hashes: bool = True,
) -> dict[str, int]:
    """Load and validate the registry, its row-free contracts, and declared hashes."""

    registry_path = Path(registry_path).resolve()
    base_dir = registry_path.parent
    registry = _load_json(registry_path)

    source_contracts = _require_unique_ids(registry.get("source_contracts", []), "source contract")
    for contract_id, contract in source_contracts.items():
        contract_path = _resolve_contract_path(base_dir, str(contract.get("path", "")))
        if not contract_path.is_file():
            raise RegistryValidationError(f"source contract {contract_id} is missing: {contract_path}")
        if contract.get("hash_required") is True:
            expected = contract.get("sha256")
            actual = sha256_file(contract_path)
            if actual != expected:
                raise RegistryValidationError(
                    f"source contract {contract_id} hash mismatch: expected {expected}, got {actual}"
                )

    if verify_historical_hashes:
        historical = _require_unique_ids(
            registry.get("historical_row_free_sources", []), "historical source"
        )
        for source_id, source in historical.items():
            source_path = _resolve_contract_path(base_dir, str(source.get("path", "")))
            if not source_path.is_file():
                raise RegistryValidationError(f"historical source {source_id} is missing: {source_path}")
            expected = source.get("sha256")
            actual = sha256_file(source_path)
            if actual != expected:
                raise RegistryValidationError(
                    f"historical source {source_id} hash mismatch: expected {expected}, got {actual}"
                )

    feature_contract = source_contracts.get("feature_registry")
    if feature_contract is None:
        raise RegistryValidationError("feature_registry source contract is required")
    feature_registry = _load_json(
        _resolve_contract_path(base_dir, str(feature_contract.get("path", "")))
    )
    unit_contract = source_contracts.get("official_unit_reconciliation")
    if unit_contract is None:
        raise RegistryValidationError("official_unit_reconciliation source contract is required")
    unit_reconciliation = _load_json(
        _resolve_contract_path(base_dir, str(unit_contract.get("path", "")))
    )
    summary = validate_registry_payload(
        registry,
        feature_registry,
        unit_reconciliation,
    )
    summary["source_hash_count"] = len(source_contracts)
    summary["historical_hash_count"] = (
        len(registry.get("historical_row_free_sources", [])) if verify_historical_hashes else 0
    )
    return summary


__all__ = [
    "RegistryValidationError",
    "SCHEMA_VERSION",
    "sha256_file",
    "validate_registry_files",
    "validate_registry_payload",
]
