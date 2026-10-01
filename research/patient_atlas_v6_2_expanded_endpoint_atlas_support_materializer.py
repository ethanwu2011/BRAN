"""Authenticated local aggregate support materializer for the V6.2 atlas.

This module is a narrow local-data seam for the already frozen 32-endpoint
survey-condition precommit.  Production accepts one canonical dataset root,
authenticates the participants/observation/visit source files by their frozen
SHA-256 digests, derives the existing train/validation fold map, and streams
index-visit binary observation labels in memory.  A separate explicit row
sequence is retained only for synthetic kernel tests.

Only integer 0/1 observations are label evidence.  An absent participant /
source observation is missing and is excluded from that endpoint's observed
denominator.  Duplicate observations with the same value collapse; a
duplicate disagreement fails closed.  No model, checkpoint, embedding,
official-test, or public-mapping loader exists in this seam.  Support outcome
values are accessed locally only and are never serialized.

Participant identifiers, visit identifiers, rows, labels, and fold
assignments remain in local memory only.  The returned receipt contains only
authenticated artifact hashes, frozen endpoint metadata, and
small-cell-suppressed per-endpoint/per-fold support counts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from patient_atlas_v6_2_expanded_endpoint_atlas import (
    AggregateOnlyError,
    CLAIM_PROXIMITY_CLASSES,
    EXPECTED_CANDIDATE_SOURCE_CODES,
    FREE_TEXT_SOURCE_CODES,
    LANE_DE_NOVO,
    LANE_FULL_CONTEXT,
    MIN_CASES,
    MIN_CASES_PER_OUTER_FOLD,
    MIN_CONTROLS,
    MIN_CONTROLS_PER_OUTER_FOLD,
    OfficialTestRefusal,
    PUBLIC_CUSTOM_CODE_MASTER_COMMIT,
    PUBLIC_CUSTOM_CODE_MASTER_REPO_PATH,
    PUBLIC_CUSTOM_CODE_MASTER_REPO_URL,
    PUBLIC_CUSTOM_CODE_MASTER_SHA256,
    ProtocolError,
    RUNNER_SCHEMA_VERSION as FROZEN_ATLAS_RUNNER_SCHEMA_VERSION,
    SMALL_CELL_THRESHOLD,
    SOURCE_TO_DIRECT_V62_FLAG,
    SURVEY_BINARY_LABEL_ENCODING,
    SURVEY_LABEL_POLICY,
    SURVEY_MISSING_RESPONSE_POLICY,
    V62_CLINICAL_FLAG_DIRECT_FEATURE_EXCLUSIONS,
    V62_CLINICAL_FLAG_NAMES,
    V62_OUTER_FOLD_SHA256,
    _claim_reporting_metadata,
    assert_aggregate_only_payload,
    canonical_json_bytes,
    sha256_file,
    validate_protocol as validate_frozen_endpoint_protocol,
)


SUPPORT_MATERIALIZER_PROTOCOL_NAME = (
    "BARAS_V6_2_EXPANDED_ENDPOINT_ATLAS_SUPPORT_MATERIALIZER_PROTOCOL_V1.json"
)
SUPPORT_MATERIALIZER_PROTOCOL_SCHEMA_VERSION = (
    "baras-v6-2-expanded-endpoint-atlas-support-materializer-protocol-v1"
)
SUPPORT_MATERIALIZER_SCHEMA_VERSION = (
    "baras-v6-2-expanded-endpoint-atlas-support-materializer-v1"
)
SUPPORT_MATERIALIZER_RUNNER_SCHEMA_VERSION = (
    "baras-v6-2-expanded-endpoint-atlas-support-materializer-run-v1"
)

FROZEN_ENDPOINT_PROTOCOL_NAME = (
    "BARAS_V6_2_EXPANDED_ENDPOINT_ATLAS_PROTOCOL_V1.json"
)
FROZEN_ENDPOINT_PROTOCOL_SHA256 = (
    "c504bfd847a0ec2c8b610da2c69371112aec9814cdda1626d6876e968f514aa5"
)
FROZEN_PRECOMMIT_RECEIPT_NAME = (
    "BARAS_V6_2_EXPANDED_ENDPOINT_ATLAS_PRECOMMIT_V1_ATTEMPT1.json"
)
FROZEN_PRECOMMIT_RECEIPT_SHA256 = (
    "6b15330de433a40d44eac9e48f4cd8864a78fbd3519e113a1be3b0aa990efcf7"
)
OUTER_FOLD_COUNT = 5
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NULL_TEXT = frozenset({"", "nan", "none", "null"})
_OFFICIAL_SPLITS = frozenset({"test", "official_test", "official-test", "holdout"})
_ALLOWED_SPLITS = frozenset({"train", "val", "validation"})
_INPUT_ARTIFACT_KEYS = frozenset({"participants", "visits", "observations"})

# These are the only three patient-derived sources that the production seam
# authenticates and reads.  The digests are frozen from the row-free source
# audit; a caller cannot substitute an arbitrary artifact or an out-of-band
# digest.  Hashing happens before any CSV/TSV parser is opened.
CANONICAL_DATASET_SOURCES = MappingProxyType(
    {
        "participants_tsv": {
            "path": "participants.tsv",
            "sha256": "481130e61c26a7bd2fa05f6421f236c1d52ac4aebee87df1f07ba7e2e7819825",
        },
        "observation_csv": {
            "path": "clinical_data/observation.csv",
            "sha256": "4f85df31a90e830108b36191681ed96ca78cc534aae11c38caa01306c0a0f062",
        },
        "visit_occurrence_csv": {
            "path": "clinical_data/visit_occurrence.csv",
            "sha256": "ca0b3818cedfaeba0b8e83c64020be8f5a89fd991406acd7844a5a7fd3ec1653",
        },
    }
)

# The fold is not recomputed or tuned by this materializer.  These bindings
# identify the exact canonical cohort/fold API used by the existing V6.2
# screening path.  They are checked before any local target/support rows are
# consumed.  ``load_development_disease_targets`` is used only to reproduce
# the already-frozen fold map; the endpoint support labels are read by the
# explicit binary loader below.
CANONICAL_COHORT_FOLD_API = MappingProxyType(
    {
        "cohort_loader": {
            "module": "patient_atlas_real_data",
            "symbol": "load_exploratory_raw_cohort",
            "file": "patient_atlas_real_data.py",
            "sha256": "1c0f454ae1147c696481c55cc860c39428a2878881cd5646375af600244d0ce7",
        },
        "fold_policy_validator": {
            "module": "patient_atlas_disease_folds",
            "symbol": "validate_disease_fold_policy",
            "file": "patient_atlas_disease_folds.py",
            "sha256": "af9c43c39f3bae7fc2c5b6f60a382741ec36363f86e33510512e35eac8589a31",
        },
        "fold_map_builder": {
            "module": "patient_atlas_disease_folds",
            "symbol": "make_disease_fold_map",
            "file": "patient_atlas_disease_folds.py",
            "sha256": "af9c43c39f3bae7fc2c5b6f60a382741ec36363f86e33510512e35eac8589a31",
        },
        "fold_policy": {
            "file": "PATIENT_ATLAS_V5_DISEASE_FOLD_POLICY_V1.json",
            "sha256": "f889b48fb494ea26d4c8c0265517863ab743c3ca74d410689126b0136713793b",
        },
        "fold_target_loader": {
            "module": "patient_atlas_disease_targets",
            "symbol": "load_development_disease_targets",
            "file": "patient_atlas_disease_targets.py",
            "sha256": "9eac8aeacf74624f028873eca38716e590e526bdac6832ed8d5ef436fdf0440a",
        },
    }
)


class SupportMaterializerError(ValueError):
    """Base error for malformed or unauthenticated local support inputs."""


class ArtifactHashError(SupportMaterializerError):
    """Raised when an explicitly supplied artifact hash does not authenticate."""


class InputSchemaError(SupportMaterializerError):
    """Raised when local support input is outside the frozen row contract."""


class DuplicateDisagreementError(InputSchemaError):
    """Raised when duplicate metadata or observation rows disagree."""


class IndexVisitOnlyError(InputSchemaError):
    """Raised when an observation artifact contains a non-index visit."""


class FoldBindingError(SupportMaterializerError):
    """Raised when the exact frozen V6.2 outer-fold map is not supplied."""


class PrecommitBindingError(SupportMaterializerError):
    """Raised when the endpoint protocol or precommit receipt has drifted."""


class SupportMaterializerProtocolError(SupportMaterializerError):
    """Raised when the materializer protocol is not authenticated."""


@dataclass(frozen=True)
class _Participant:
    participant_id: str
    split: str
    outer_fold: int


@dataclass(frozen=True)
class _DatasetParticipant:
    """Minimal in-memory index metadata for one exploratory participant."""

    participant_id: str
    split: str
    outer_fold: int
    study_visit_date: date


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputSchemaError("JSON artifact contains duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise InputSchemaError(f"JSON artifact contains non-finite value: {value}")


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ArtifactHashError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise InputSchemaError(f"{label} must be an integer")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.strip() == str(value).strip():
        token = value.strip()
        if not re.fullmatch(r"-?[0-9]+", token):
            raise InputSchemaError(f"{label} must be an integer")
        number = int(token)
    else:
        raise InputSchemaError(f"{label} must be an integer")
    return number


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise InputSchemaError(f"{label} must be text")
    result = str(value).strip()
    if result.lower() in _NULL_TEXT:
        raise InputSchemaError(f"{label} is missing")
    return result


def _normalize_split(value: Any) -> str:
    token = _text(value, label="participant split").lower()
    if token in _OFFICIAL_SPLITS:
        raise OfficialTestRefusal("official test participants are refused")
    if token not in _ALLOWED_SPLITS:
        raise InputSchemaError("participants must be train/validation only")
    return "val" if token == "validation" else token


def _normalize_iso_date(value: Any, *, label: str) -> date:
    """Parse the date portion used by the canonical index-link rule."""

    token = _text(value, label=label)
    # The canonical OMOP extracts use ISO dates.  Accept an ISO timestamp only
    # by taking its date portion; never infer a date from a locale-dependent
    # string or from a missing value.
    candidate = token[:10]
    try:
        return date.fromisoformat(candidate)
    except ValueError as error:
        raise InputSchemaError(f"{label} must be an ISO date") from error


def _reject_input_hazards(value: Any, *, where: str) -> None:
    """Reject outcome/model/official-test material while allowing local IDs."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            token = str(key).strip().lower()
            if token in {
                "target_values",
                "outcome_values",
                "predictions",
                "model_scores",
                "model_performance",
                "losses",
                "embeddings",
                "coordinates",
                "labels",
                "fold_assignments",
                "official_test",
                "official_test_included",
                "is_official_test",
            }:
                if token in {"official_test", "official_test_included", "is_official_test"}:
                    if child not in (False, None, 0, "", "false", "False", "0"):
                        raise OfficialTestRefusal("official test input is refused")
                else:
                    raise InputSchemaError(f"{where} contains a forbidden field")
            if token in {"split", "cohort", "recommended_split"}:
                if str(child).strip().lower() in _OFFICIAL_SPLITS:
                    raise OfficialTestRefusal("official test input is refused")
            _reject_input_hazards(child, where=where)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_input_hazards(child, where=where)


def _exact_row_keys(row: Mapping[str, Any], required: frozenset[str], *, label: str) -> None:
    if set(row) != set(required):
        raise InputSchemaError(f"{label} fields do not match the frozen schema")


def _normalize_participants(rows: Sequence[Mapping[str, Any]]) -> tuple[_Participant, ...]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)) or not rows:
        raise InputSchemaError("participants must be a non-empty row sequence")
    _reject_input_hazards(rows, where="participants")
    required = frozenset({"participant_id", "split", "outer_fold"})
    result: list[_Participant] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise InputSchemaError("participant records must be objects")
        _exact_row_keys(row, required, label="participant")
        participant_id = _text(row["participant_id"], label="participant_id")
        if participant_id in seen:
            raise DuplicateDisagreementError("duplicate participant metadata is not allowed")
        seen.add(participant_id)
        split = _normalize_split(row["split"])
        outer_fold = _strict_integer(row["outer_fold"], label="outer_fold")
        if outer_fold < 0 or outer_fold >= OUTER_FOLD_COUNT:
            raise FoldBindingError("participant outer fold is outside the frozen five-fold map")
        result.append(_Participant(participant_id, split, outer_fold))
    if {item.split for item in result} - {"train", "val"}:
        raise InputSchemaError("participants must be train/validation only")
    return tuple(result)


def _normalize_visits(
    rows: Sequence[Mapping[str, Any]],
    participants: Sequence[_Participant],
) -> Mapping[str, str]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise InputSchemaError("visits must be a row sequence")
    _reject_input_hazards(rows, where="visits")
    required = frozenset({"participant_id", "visit_id", "is_index_visit"})
    participant_ids = {item.participant_id for item in participants}
    visit_roles: dict[tuple[str, str], bool] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise InputSchemaError("visit records must be objects")
        _exact_row_keys(row, required, label="visit")
        participant_id = _text(row["participant_id"], label="visit participant_id")
        if participant_id not in participant_ids:
            raise InputSchemaError("visit references an unknown participant")
        visit_id = _text(row["visit_id"], label="visit_id")
        is_index = row["is_index_visit"]
        if not isinstance(is_index, bool):
            raise InputSchemaError("is_index_visit must be an explicit boolean")
        key = (participant_id, visit_id)
        prior = visit_roles.get(key)
        if prior is not None and prior != is_index:
            raise DuplicateDisagreementError("duplicate visit index roles disagree")
        visit_roles[key] = is_index
    index_visits: dict[str, str] = {}
    for participant_id in participant_ids:
        candidate = [
            visit_id
            for (owner, visit_id), is_index in visit_roles.items()
            if owner == participant_id and is_index
        ]
        if len(candidate) != 1:
            raise InputSchemaError("each participant must have exactly one index visit")
        index_visits[participant_id] = candidate[0]
    return MappingProxyType(index_visits)


def _binary_value(value: Any) -> int:
    if isinstance(value, bool):
        raise InputSchemaError("observation values must be numeric 0 or 1, not boolean")
    if isinstance(value, int) and value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    raise InputSchemaError("observation values must be explicit numeric 0 or 1")


def _normalize_observations(
    rows: Sequence[Mapping[str, Any]],
    participants: Sequence[_Participant],
    index_visits: Mapping[str, str],
) -> Mapping[tuple[str, str], int]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise InputSchemaError("observations must be a row sequence")
    _reject_input_hazards(rows, where="observations")
    required = frozenset({"participant_id", "visit_id", "source_code", "value"})
    participant_ids = {item.participant_id for item in participants}
    expected_sources = set(EXPECTED_CANDIDATE_SOURCE_CODES)
    normalized: dict[tuple[str, str], int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise InputSchemaError("observation records must be objects")
        _exact_row_keys(row, required, label="observation")
        participant_id = _text(row["participant_id"], label="observation participant_id")
        if participant_id not in participant_ids:
            raise InputSchemaError("observation references an unknown participant")
        visit_id = _text(row["visit_id"], label="observation visit_id")
        if index_visits[participant_id] != visit_id:
            raise IndexVisitOnlyError("observation artifact contains a non-index visit")
        source_code = _text(row["source_code"], label="observation source_code")
        if source_code not in expected_sources:
            if source_code in FREE_TEXT_SOURCE_CODES:
                raise InputSchemaError("free-text condition sources are excluded")
            raise InputSchemaError("observation source is outside the frozen 32-endpoint set")
        value = _binary_value(row["value"])
        key = (participant_id, source_code)
        prior = normalized.get(key)
        if prior is not None and prior != value:
            raise DuplicateDisagreementError("duplicate observation values disagree")
        normalized[key] = value
    return MappingProxyType(normalized)


def _safe_hash_map(
    *,
    participants_sha256: Any,
    visits_sha256: Any,
    observations_sha256: Any,
) -> Mapping[str, str]:
    return MappingProxyType(
        {
            "participants_sha256": _require_sha256(
                participants_sha256, label="participants artifact"
            ),
            "visits_sha256": _require_sha256(visits_sha256, label="visits artifact"),
            "observations_sha256": _require_sha256(
                observations_sha256, label="observations artifact"
            ),
        }
    )


def _validate_frozen_receipt(root: Path) -> Mapping[str, Any]:
    protocol_path = root / FROZEN_ENDPOINT_PROTOCOL_NAME
    if not protocol_path.is_file() or sha256_file(protocol_path) != FROZEN_ENDPOINT_PROTOCOL_SHA256:
        raise PrecommitBindingError("frozen endpoint protocol hash differs")
    try:
        protocol, _ = validate_frozen_endpoint_protocol(
            project_root=root,
            protocol_path=protocol_path,
            verify_bindings=True,
        )
    except (ProtocolError, OSError, ValueError) as error:
        raise PrecommitBindingError("frozen endpoint protocol is not authenticated") from error
    if protocol.get("candidate_count") != len(EXPECTED_CANDIDATE_SOURCE_CODES):
        raise PrecommitBindingError("frozen endpoint protocol candidate count differs")
    receipt_path = root / FROZEN_PRECOMMIT_RECEIPT_NAME
    if not receipt_path.is_file() or sha256_file(receipt_path) != FROZEN_PRECOMMIT_RECEIPT_SHA256:
        raise PrecommitBindingError("frozen endpoint precommit receipt hash differs")
    try:
        receipt = json.loads(
            receipt_path.read_bytes(),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, InputSchemaError) as error:
        raise PrecommitBindingError("frozen endpoint precommit receipt is not valid JSON") from error
    if not isinstance(receipt, Mapping):
        raise PrecommitBindingError("frozen endpoint precommit receipt is not an object")
    try:
        assert_aggregate_only_payload(receipt)
    except (AggregateOnlyError, OfficialTestRefusal) as error:
        raise PrecommitBindingError("frozen endpoint precommit receipt is not aggregate-safe") from error
    if (
        receipt.get("schema_version") != FROZEN_ATLAS_RUNNER_SCHEMA_VERSION
        or receipt.get("status") != "aggregate_only_precommit_complete"
        or receipt.get("protocol_sha256") != FROZEN_ENDPOINT_PROTOCOL_SHA256
        or receipt.get("outer_fold_assignment_sha256") != V62_OUTER_FOLD_SHA256
        or receipt.get("candidate_count") != len(EXPECTED_CANDIDATE_SOURCE_CODES)
        or tuple(receipt.get("candidate_source_codes", ())) != EXPECTED_CANDIDATE_SOURCE_CODES
        or receipt.get("official_test_inputs_loaded") is not False
        or receipt.get("patient_rows_or_identifiers_emitted") is not False
        or receipt.get("outcome_values_accessed") is not False
    ):
        raise PrecommitBindingError("frozen endpoint receipt metadata differs")
    return MappingProxyType(
        {
            "protocol": protocol,
            "receipt": receipt,
            "protocol_sha256": FROZEN_ENDPOINT_PROTOCOL_SHA256,
            "receipt_sha256": FROZEN_PRECOMMIT_RECEIPT_SHA256,
        }
    )


def _expected_protocol_contract() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "schema_version": SUPPORT_MATERIALIZER_PROTOCOL_SCHEMA_VERSION,
            "status": "frozen_before_local_support_materialization",
            "precommit_binding": {
                "protocol_file": FROZEN_ENDPOINT_PROTOCOL_NAME,
                "protocol_sha256": FROZEN_ENDPOINT_PROTOCOL_SHA256,
                "receipt_file": FROZEN_PRECOMMIT_RECEIPT_NAME,
                "receipt_sha256": FROZEN_PRECOMMIT_RECEIPT_SHA256,
            },
            "fold_binding": {
                "outer_fold_assignment_sha256": V62_OUTER_FOLD_SHA256,
                "outer_fold_count": OUTER_FOLD_COUNT,
                "cohort_scope": "official_train_validation_only",
                "official_test_inputs_or_targets_allowed": False,
                "new_fold_assignment_or_rebalancing_allowed": False,
            },
            "candidate_contract": {
                "candidate_count": len(EXPECTED_CANDIDATE_SOURCE_CODES),
                "candidate_source_codes": list(EXPECTED_CANDIDATE_SOURCE_CODES),
                "free_text_sources_excluded": list(FREE_TEXT_SOURCE_CODES),
                "all_declared_candidates_retained": True,
            },
            "mapping_provenance": {
                "repo_url": PUBLIC_CUSTOM_CODE_MASTER_REPO_URL,
                "commit": PUBLIC_CUSTOM_CODE_MASTER_COMMIT,
                "repo_relative_path": PUBLIC_CUSTOM_CODE_MASTER_REPO_PATH,
                "sha256": PUBLIC_CUSTOM_CODE_MASTER_SHA256,
                "runtime_temp_checkout_required": False,
            },
            "lanes": {
                LANE_FULL_CONTEXT: {
                    "target_specific_rule": "erase_exact_target_source_and_direct_features",
                    "condition_history_policy": "retain_other_condition_history_flags",
                    "all_condition_history_flags_removed": False,
                },
                LANE_DE_NOVO: {
                    "target_specific_rule": "erase_exact_target_source_and_all_condition_history_flags",
                    "condition_history_policy": "erase_all_condition_history_flags",
                    "continuous_labs_and_vitals_retained": True,
                    "all_condition_history_flags_removed": True,
                },
            },
            "circularity_contract": {
                "support_labels_used_only_for_local_support_counts": True,
                "endpoint_values_used_as_model_features": False,
                "full_context_exact_target_and_direct_features_removed": True,
                "de_novo_all_condition_history_flags_removed": True,
                "continuous_labs_and_vitals_erased": False,
                "model_scoring_performed": False,
            },
            "claim_reporting": _claim_reporting_metadata(),
            "support_gates": {
                "minimum_cases": MIN_CASES,
                "minimum_controls": MIN_CONTROLS,
                "minimum_cases_per_outer_fold": MIN_CASES_PER_OUTER_FOLD,
                "minimum_controls_per_outer_fold": MIN_CONTROLS_PER_OUTER_FOLD,
                "small_cell_threshold": SMALL_CELL_THRESHOLD,
                "support_counts_denominator": "observed_explicit_binary_labels_only",
                "gate_fail_action": "retain_candidate_with_withheld_or_missing_or_excluded_status",
                "threshold_or_fold_retuning_allowed": False,
            },
            "survey_label_policy": dict(SURVEY_LABEL_POLICY),
            "duplicate_policy": {
                "same_key_same_binary_value": "collapse",
                "same_key_disagreement": "fail_closed",
                "missing_response_representation": "absence_of_observation_row",
                "non_binary_or_explicit_missing_value": "ignored_as_missing",
            },
            "canonical_sources": {
                key: dict(value) for key, value in CANONICAL_DATASET_SOURCES.items()
            },
            "canonical_cohort_fold_api": {
                key: dict(value) for key, value in CANONICAL_COHORT_FOLD_API.items()
            },
            "production_input_contract": {
                "dataset_root_argument": True,
                "project_root_argument": True,
                "clinical_project_root_argument": True,
                "source_hashes_authenticated_before_parse": True,
                "participants_required_fields": [
                    "person_id",
                    "study_visit_date",
                    "recommended_split",
                ],
                "visit_required_fields": [
                    "visit_occurrence_id",
                    "visit_start_date",
                ],
                "observation_required_fields": [
                    "person_id",
                    "observation_date",
                    "visit_occurrence_id",
                    "observation_source_value",
                    "value_as_number",
                ],
                "index_visit_rule": "observation_date_or_linked_visit_start_date_equals_study_visit_date",
                "accepted_value_domain": [0, 1],
                "non_binary_values": "ignored_as_missing",
                "row_sequence_json_or_npz_artifacts": "internal_kernel_tests_only",
            },
            "privacy": {
                "patient_derived_processing": "local_only",
                "patient_ids_visits_rows_labels_arrays_predictions_scores_or_fold_assignments_serialized": False,
                "aggregate_only_output": True,
                "official_test_inputs_or_targets_allowed": False,
                "outcome_values_accessed_locally": True,
                "outcome_values_serialized": False,
            },
            "model_scoring": {
                "models_loaded": False,
                "predictions_or_scores_loaded": False,
                "outcome_values_accessed_locally": True,
                "outcome_values_serialized": False,
            },
        }
    )


def _load_protocol(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, InputSchemaError) as error:
        raise SupportMaterializerProtocolError("support materializer protocol is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise SupportMaterializerProtocolError("support materializer protocol must be one object")
    return value


def _validate_canonical_api_bindings(root: Path, value: Mapping[str, Any]) -> None:
    """Verify the exact cohort/fold implementation before any dataset parse."""

    bindings = value.get("canonical_cohort_fold_api")
    expected = _expected_protocol_contract()["canonical_cohort_fold_api"]
    if bindings != expected or not isinstance(bindings, Mapping):
        raise SupportMaterializerProtocolError("canonical cohort/fold API contract differs")
    seen: set[Path] = set()
    for label, raw in bindings.items():
        if not isinstance(raw, Mapping):
            raise SupportMaterializerProtocolError(f"canonical API binding malformed: {label}")
        name = raw.get("file")
        digest = raw.get("sha256")
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise SupportMaterializerProtocolError(f"canonical API binding path unsafe: {label}")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise SupportMaterializerProtocolError(f"canonical API binding hash malformed: {label}")
        source = (root / name).resolve()
        if source in seen:
            continue
        seen.add(source)
        if not source.is_file() or sha256_file(source) != digest:
            raise SupportMaterializerProtocolError(f"canonical API binding differs: {label}")


def authenticate_canonical_dataset_sources(
    dataset_root: str | Path,
) -> Mapping[str, str]:
    """Hash the three canonical local sources before opening any row parser."""

    root = Path(dataset_root).resolve()
    observed: dict[str, str] = {}
    for label, contract in CANONICAL_DATASET_SOURCES.items():
        path = root / str(contract["path"])
        if not path.is_file():
            raise ArtifactHashError(f"canonical dataset source is missing: {label}")
        digest = sha256_file(path)
        expected = str(contract["sha256"])
        if digest != expected:
            raise ArtifactHashError(f"canonical dataset source hash differs: {label}")
        observed[label] = digest
    return MappingProxyType(observed)


def validate_support_materializer_protocol(
    project_root: str | Path | None = None,
    protocol_path: str | Path | None = None,
    *,
    verify_bindings: bool = True,
    verify_frozen_receipt: bool = True,
) -> tuple[Mapping[str, Any], str]:
    """Authenticate the local materializer contract and its frozen parent."""

    root = Path(__file__).resolve().parent if project_root is None else Path(project_root).resolve()
    canonical = (root / SUPPORT_MATERIALIZER_PROTOCOL_NAME).resolve()
    path = canonical if protocol_path is None else Path(protocol_path).resolve()
    if path != canonical:
        raise SupportMaterializerProtocolError("support materializer protocol must be canonical")
    value = _load_protocol(path)
    expected = _expected_protocol_contract()
    for key in (
        "schema_version",
        "status",
        "precommit_binding",
        "fold_binding",
        "candidate_contract",
        "mapping_provenance",
        "lanes",
        "circularity_contract",
        "claim_reporting",
        "support_gates",
        "survey_label_policy",
        "duplicate_policy",
        "canonical_sources",
        "canonical_cohort_fold_api",
        "production_input_contract",
        "privacy",
        "model_scoring",
    ):
        if value.get(key) != expected[key]:
            raise SupportMaterializerProtocolError(f"support materializer protocol section differs: {key}")
    if verify_bindings:
        bindings = value.get("bindings")
        expected_labels = {"kernel", "runner", "synthetic_tests"}
        if not isinstance(bindings, Mapping) or set(bindings) != expected_labels:
            raise SupportMaterializerProtocolError("support materializer code bindings differ")
        for label, raw in bindings.items():
            if not isinstance(raw, Mapping) or set(raw) != {"file", "sha256"}:
                raise SupportMaterializerProtocolError(f"support materializer binding malformed: {label}")
            name = str(raw.get("file", ""))
            digest = raw.get("sha256")
            if Path(name).is_absolute() or ".." in Path(name).parts or not _SHA256.fullmatch(str(digest)):
                raise SupportMaterializerProtocolError(f"support materializer binding unsafe: {label}")
            source = (root / name).resolve()
            if not source.is_file() or sha256_file(source) != digest:
                raise SupportMaterializerProtocolError(f"support materializer binding differs: {label}")
        _validate_canonical_api_bindings(root, value)
    protocol_sha256 = sha256_file(path)
    if verify_frozen_receipt:
        _validate_frozen_receipt(root)
    return value, protocol_sha256


def _counts_by_source(
    participants: Sequence[_Participant],
    observations: Mapping[tuple[str, str], int],
) -> Mapping[str, Mapping[str, Any]]:
    participant_by_id = {item.participant_id: item for item in participants}
    counts: dict[str, dict[str, Any]] = {
        source: {
            "cases": 0,
            "controls": 0,
            "cases_by_outer_fold": [0] * OUTER_FOLD_COUNT,
            "controls_by_outer_fold": [0] * OUTER_FOLD_COUNT,
            "coverage_status": "unavailable",
            "negative_label_status": "missing_or_excluded",
            "label_encoding": None,
        }
        for source in EXPECTED_CANDIDATE_SOURCE_CODES
    }
    observed_by_source: dict[str, int] = {source: 0 for source in EXPECTED_CANDIDATE_SOURCE_CODES}
    for (participant_id, source), value in observations.items():
        participant = participant_by_id[participant_id]
        row = counts[source]
        fold = participant.outer_fold
        observed_by_source[source] += 1
        if value == 1:
            row["cases"] += 1
            row["cases_by_outer_fold"][fold] += 1
        else:
            row["controls"] += 1
            row["controls_by_outer_fold"][fold] += 1
    participant_count = len(participants)
    for source, observed in observed_by_source.items():
        row = counts[source]
        if observed:
            row["negative_label_status"] = "validated_binary"
            row["label_encoding"] = SURVEY_BINARY_LABEL_ENCODING
            row["coverage_status"] = "complete" if observed == participant_count else "partial"
    return MappingProxyType(counts)


def _safe_count(value: int) -> int | str:
    """Small-cell suppress a count while preserving zero and gate-sized values."""

    number = int(value)
    if number == 0 or number >= SMALL_CELL_THRESHOLD:
        return number
    return f"<{SMALL_CELL_THRESHOLD}"


def _claim_class_for_source(source: str) -> str:
    metadata = _claim_reporting_metadata()
    source_fields = {
        "measurement_proximal_or_policy": "measurement_proximal_or_policy_sources",
        "metabolic_history_or_proximal": "metabolic_history_or_proximal_sources",
        "non_proximal_disease_screening": "non_proximal_disease_screening_sources",
    }
    for class_name, field in source_fields.items():
        if source in metadata.get(field, ()):
            return class_name
    raise SupportMaterializerError(f"source lacks frozen claim-proximity group: {source}")


def _summarize_materialized_support(
    raw_support: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Convert local counts to the aggregate-only support audit.

    This is intentionally separate from the outcome-free precommit summarizer:
    support materialization does inspect local binary outcome values, but never
    serializes those values or any row-level representation.
    """

    rows: dict[str, Any] = {}
    for source in EXPECTED_CANDIDATE_SOURCE_CODES:
        support = raw_support.get(source)
        if not isinstance(support, Mapping):
            raise InputSchemaError("materialized support source map is incomplete")
        cases = int(support["cases"])
        controls = int(support["controls"])
        case_folds = tuple(int(value) for value in support["cases_by_outer_fold"])
        control_folds = tuple(int(value) for value in support["controls_by_outer_fold"])
        if len(case_folds) != OUTER_FOLD_COUNT or len(control_folds) != OUTER_FOLD_COUNT:
            raise FoldBindingError("materialized support fold counts are malformed")
        binary = support.get("negative_label_status") == "validated_binary"
        gate = bool(
            binary
            and cases >= MIN_CASES
            and controls >= MIN_CONTROLS
            and min(case_folds) >= MIN_CASES_PER_OUTER_FOLD
            and min(control_folds) >= MIN_CONTROLS_PER_OUTER_FOLD
        )
        if not binary:
            status = "missing_or_withheld"
        elif gate:
            status = "eligible_under_frozen_support_gates"
        else:
            status = "withheld_under_frozen_support_gates"
        rows[source] = {
            "status": status,
            "claim_proximity_class": _claim_class_for_source(source),
            "support": {
                "cases": _safe_count(cases),
                "controls": _safe_count(controls),
                "cases_by_outer_fold": [_safe_count(value) for value in case_folds],
                "controls_by_outer_fold": [_safe_count(value) for value in control_folds],
                "coverage_status": str(support["coverage_status"]),
                "negative_label_status": str(support["negative_label_status"]),
                "label_encoding": support.get("label_encoding"),
                "missing_response_policy": SURVEY_MISSING_RESPONSE_POLICY,
            },
            "gates": {
                "minimum_cases": MIN_CASES,
                "minimum_controls": MIN_CONTROLS,
                "minimum_cases_per_outer_fold": MIN_CASES_PER_OUTER_FOLD,
                "minimum_controls_per_outer_fold": MIN_CONTROLS_PER_OUTER_FOLD,
                "whole_cohort_complete_coverage_required": False,
                "support_denominator": "observed_explicit_binary_labels_only",
                "selection_uses_model_performance": False,
            },
        }
    return MappingProxyType(
        {
            "status": "aggregate_support_audited_under_frozen_gates",
            "outcome_values_accessed_locally": True,
            "outcome_values_serialized": False,
            "candidate_registry_selection_uses_support": False,
            "evaluation_eligibility_uses_frozen_support": True,
            "all_declared_candidates_retained": True,
            "survey_label_policy": dict(SURVEY_LABEL_POLICY),
            "candidates": rows,
        }
    )


def _safe_materializer_output(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate output privacy and return ordinary JSON-safe containers."""

    def _jsonable(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [_jsonable(item) for item in value]
        return value

    ordinary = json.loads(canonical_json_bytes(_jsonable(payload)))
    assert_aggregate_only_payload(ordinary)
    if (
        ordinary.get("patient_labels_or_arrays_emitted") is not False
        or ordinary.get("fold_assignments_emitted") is not False
        or ordinary.get("models_scored") is not False
        or ordinary.get("predictions_or_scores_loaded") is not False
        or ordinary.get("outcome_values_accessed_locally") is not True
        or ordinary.get("outcome_values_serialized") is not False
    ):
        raise AggregateOnlyError("support materializer output lacks privacy attestations")
    encoded = json.dumps(ordinary, sort_keys=True, ensure_ascii=False)
    for token in (
        '"participant_id":',
        '"person_id":',
        '"visit_id":',
        '"value":',
        '"labels":',
        '"fold_assignments":',
        '"predictions":',
        '"model_scores":',
        '"outcome_values":',
        '"outcome_values_accessed":',
    ):
        if token in encoded:
            raise AggregateOnlyError("support materializer output contains local data")
    return ordinary


def _csv_fieldnames(reader: csv.DictReader[str], *, required: frozenset[str], label: str) -> None:
    fields = reader.fieldnames
    if fields is None or not required.issubset(set(fields)):
        raise InputSchemaError(f"canonical {label} source lacks required fields")


def _optional_iso_date(value: Any) -> date | None:
    token = "" if value is None else str(value).strip()
    if token.lower() in _NULL_TEXT:
        return None
    candidate = token[:10]
    try:
        return date.fromisoformat(candidate)
    except ValueError:
        return None


def _load_dataset_participants(
    dataset_root: Path,
    *,
    cohort: Any,
    fold_map: Any,
) -> tuple[_DatasetParticipant, ...]:
    """Read only cohort participant metadata needed for index linking."""

    cohort_ids = tuple(str(value) for value in cohort.patient_ids)
    if not cohort_ids or len(cohort_ids) != len(set(cohort_ids)):
        raise FoldBindingError("canonical cohort participant identities are malformed")
    cohort_splits = tuple(str(value) for value in cohort.split_labels)
    if len(cohort_splits) != len(cohort_ids):
        raise FoldBindingError("canonical cohort split metadata is misaligned")
    split_by_id = dict(zip(cohort_ids, cohort_splits))
    assignments = fold_map.as_mapping()
    if set(assignments) != set(cohort_ids):
        raise FoldBindingError("canonical fold map and cohort identities differ")
    found: dict[str, _DatasetParticipant] = {}
    path = dataset_root / str(CANONICAL_DATASET_SOURCES["participants_tsv"]["path"])
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            _csv_fieldnames(
                reader,
                required=frozenset({"person_id", "study_visit_date", "recommended_split"}),
                label="participants",
            )
            for row in reader:
                participant_id = str(row.get("person_id", "")).strip()
                if participant_id not in split_by_id:
                    continue
                if participant_id in found:
                    raise DuplicateDisagreementError("canonical participant identity is duplicated")
                split = _normalize_split(row.get("recommended_split"))
                if split != split_by_id[participant_id]:
                    raise FoldBindingError("canonical participant split differs from cohort")
                fold = _strict_integer(assignments[participant_id], label="outer_fold")
                if fold < 0 or fold >= OUTER_FOLD_COUNT:
                    raise FoldBindingError("canonical outer fold is outside five-fold map")
                found[participant_id] = _DatasetParticipant(
                    participant_id=participant_id,
                    split=split,
                    outer_fold=fold,
                    study_visit_date=_normalize_iso_date(
                        row.get("study_visit_date"),
                        label="study_visit_date",
                    ),
                )
    except OfficialTestRefusal:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise InputSchemaError("canonical participants source cannot be read") from error
    if set(found) != set(cohort_ids):
        raise InputSchemaError("canonical participants source lacks cohort identities")
    # Stable cohort order is useful only in memory; no ordering is emitted.
    return tuple(found[participant_id] for participant_id in cohort_ids)


def _load_dataset_visit_dates(dataset_root: Path) -> Mapping[str, date | None]:
    """Read canonical visit IDs and dates without materializing a row artifact."""

    path = dataset_root / str(CANONICAL_DATASET_SOURCES["visit_occurrence_csv"]["path"])
    result: dict[str, date | None] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _csv_fieldnames(
                reader,
                required=frozenset({"visit_occurrence_id", "visit_start_date"}),
                label="visit_occurrence",
            )
            for row in reader:
                visit_id = str(row.get("visit_occurrence_id", "")).strip()
                if not visit_id or visit_id.lower() in _NULL_TEXT:
                    raise InputSchemaError("canonical visit ID is missing")
                if visit_id in result:
                    raise DuplicateDisagreementError("canonical visit ID is duplicated")
                result[visit_id] = _optional_iso_date(row.get("visit_start_date"))
    except (OSError, UnicodeError, csv.Error) as error:
        raise InputSchemaError("canonical visit source cannot be read") from error
    return MappingProxyType(result)


def _production_binary_value(value: Any) -> int | None:
    """Return only an explicit finite numeric 0/1; other values are missing."""

    token = "" if value is None else str(value).strip()
    if token.lower() in _NULL_TEXT:
        return None
    try:
        number = float(token)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    if number == 0.0:
        return 0
    if number == 1.0:
        return 1
    return None


def _load_dataset_observations(
    dataset_root: Path,
    *,
    participants: Sequence[_DatasetParticipant],
    visit_dates: Mapping[str, date | None],
) -> Mapping[tuple[str, str], int]:
    """Stream candidate observation rows and retain only index-date 0/1 labels."""

    by_id = {item.participant_id: item for item in participants}
    expected_sources = set(EXPECTED_CANDIDATE_SOURCE_CODES)
    normalized: dict[tuple[str, str], int] = {}
    path = dataset_root / str(CANONICAL_DATASET_SOURCES["observation_csv"]["path"])
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _csv_fieldnames(
                reader,
                required=frozenset(
                    {
                        "person_id",
                        "observation_date",
                        "visit_occurrence_id",
                        "observation_source_value",
                        "value_as_number",
                    }
                ),
                label="observation",
            )
            for row in reader:
                participant_id = str(row.get("person_id", "")).strip()
                participant = by_id.get(participant_id)
                if participant is None:
                    continue
                raw_source = str(row.get("observation_source_value", "")).strip()
                source = raw_source.split(",", 1)[0].strip()
                if source not in expected_sources:
                    # This also excludes the two mapped free-text fields and
                    # generic/non-clinical survey fields without treating them
                    # as endpoint observations.
                    continue
                value = _production_binary_value(row.get("value_as_number"))
                if value is None:
                    # A missing/non-binary response is not a negative label.
                    continue
                observation_date = _optional_iso_date(row.get("observation_date"))
                visit_id = str(row.get("visit_occurrence_id", "")).strip()
                linked_visit_date = visit_dates.get(visit_id)
                if not (
                    observation_date == participant.study_visit_date
                    or linked_visit_date == participant.study_visit_date
                ):
                    continue
                key = (participant_id, source)
                prior = normalized.get(key)
                if prior is not None and prior != value:
                    raise DuplicateDisagreementError(
                        "canonical index observation values disagree"
                    )
                normalized[key] = value
    except DuplicateDisagreementError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise InputSchemaError("canonical observation source cannot be read") from error
    return MappingProxyType(normalized)


def _build_support_report(
    *,
    root: Path,
    protocol_sha256: str,
    participants: Sequence[_Participant],
    observations: Mapping[tuple[str, str], int],
    source_hashes: Mapping[str, str],
    source_hash_kind: str,
) -> Mapping[str, Any]:
    """Build the one aggregate-only report shared by production and tests."""

    raw_support = _counts_by_source(participants, observations)
    support_audit = _summarize_materialized_support(raw_support)
    frozen = _validate_frozen_receipt(root)
    receipt = frozen["receipt"]
    report: dict[str, Any] = {
        "schema_version": SUPPORT_MATERIALIZER_SCHEMA_VERSION,
        "status": "authenticated_local_support_materialization_complete",
        "protocol_sha256": FROZEN_ENDPOINT_PROTOCOL_SHA256,
        "support_materializer_protocol_sha256": protocol_sha256,
        "precommit_receipt_sha256": FROZEN_PRECOMMIT_RECEIPT_SHA256,
        "outer_fold_assignment_sha256": V62_OUTER_FOLD_SHA256,
        "candidate_count": len(EXPECTED_CANDIDATE_SOURCE_CODES),
        "candidate_source_codes": list(EXPECTED_CANDIDATE_SOURCE_CODES),
        "condition_history_flag_names": list(V62_CLINICAL_FLAG_NAMES),
        "direct_feature_exclusions": {
            key: list(value)
            for key, value in V62_CLINICAL_FLAG_DIRECT_FEATURE_EXCLUSIONS.items()
        },
        "source_to_direct_v6_2_flag": dict(SOURCE_TO_DIRECT_V62_FLAG),
        "mapping_provenance": {
            "repo_url": PUBLIC_CUSTOM_CODE_MASTER_REPO_URL,
            "commit": PUBLIC_CUSTOM_CODE_MASTER_COMMIT,
            "repo_relative_path": PUBLIC_CUSTOM_CODE_MASTER_REPO_PATH,
            "sha256": PUBLIC_CUSTOM_CODE_MASTER_SHA256,
            "runtime_temp_checkout_required": False,
        },
        "claim_reporting": receipt["claim_reporting"],
        "lanes": receipt["lanes"],
        "circularity_contract": dict(_expected_protocol_contract()["circularity_contract"]),
        "support_gates": dict(_expected_protocol_contract()["support_gates"]),
        "survey_label_policy": dict(SURVEY_LABEL_POLICY),
        "artifact_hashes": dict(source_hashes),
        "source_hashes": dict(source_hashes),
        "source_hash_kind": source_hash_kind,
        "support_audit": support_audit,
        "current_v6_2_tower_count": 2,
        "procedure_fourth_tower_added": False,
        "selection_uses_model_performance": False,
        "outcome_values_accessed_locally": True,
        "outcome_values_serialized": False,
        "patient_labels_or_arrays_emitted": False,
        "fold_assignments_emitted": False,
        "patient_rows_or_identifiers_emitted": False,
        "official_test_inputs_loaded": False,
        "models_scored": False,
        "predictions_or_scores_loaded": False,
        "runner": {
            "schema_version": SUPPORT_MATERIALIZER_RUNNER_SCHEMA_VERSION,
            "protocol_validated_before_local_input": True,
            "precommit_receipt_validated_before_local_input": True,
            "source_hashes_authenticated_before_parse": (
                source_hash_kind == "canonical_dataset_source_sha256"
            ),
            "row_level_processing_local_only": True,
            "index_visit_observations_only": True,
            "missing_response_remains_missing": True,
            "duplicate_disagreement_fails_closed": True,
            "no_model_scoring": True,
            "official_test_refused": True,
        },
        "provenance": {
            "frozen_endpoint_protocol_file": FROZEN_ENDPOINT_PROTOCOL_NAME,
            "frozen_precommit_receipt_file": FROZEN_PRECOMMIT_RECEIPT_NAME,
            "support_materializer_protocol_file": SUPPORT_MATERIALIZER_PROTOCOL_NAME,
            "cohort_scope": "official_train_validation_only",
            "source_hashes_authenticated": (
                source_hash_kind == "canonical_dataset_source_sha256"
            ),
            "source_hash_kind": source_hash_kind,
            "participant_count_or_local_rows_emitted": False,
        },
    }
    return _safe_materializer_output(report)


def materialize_local_support(
    *,
    participants: Sequence[Mapping[str, Any]],
    visits: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    participants_sha256: str,
    visits_sha256: str,
    observations_sha256: str,
    project_root: str | Path | None = None,
    protocol_path: str | Path | None = None,
    verify_bindings: bool = True,
) -> Mapping[str, Any]:
    """Internal/test-only row-sequence seam; production uses dataset_root.

    The explicit row contract is retained solely for synthetic kernel tests and
    never forms a production CLI or serialized patient-level artifact.
    """

    root = Path(__file__).resolve().parent if project_root is None else Path(project_root).resolve()
    _, protocol_sha256 = validate_support_materializer_protocol(
        project_root=root,
        protocol_path=protocol_path,
        verify_bindings=verify_bindings,
        verify_frozen_receipt=True,
    )
    artifact_hashes = _safe_hash_map(
        participants_sha256=participants_sha256,
        visits_sha256=visits_sha256,
        observations_sha256=observations_sha256,
    )
    normalized_participants = _normalize_participants(participants)
    index_visits = _normalize_visits(visits, normalized_participants)
    normalized_observations = _normalize_observations(
        observations,
        normalized_participants,
        index_visits,
    )
    return _build_support_report(
        root=root,
        protocol_sha256=protocol_sha256,
        participants=normalized_participants,
        observations=normalized_observations,
        source_hashes=artifact_hashes,
        source_hash_kind="internal_synthetic_row_fixture_sha256",
    )


def materialize_local_support_from_dataset(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    clinical_project_root: str | Path,
    protocol_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Authenticate and materialize support from canonical local dataset files.

    The source files are hashed before the canonical cohort/fold loader or any
    explicit observation parser is invoked.  Participant/visit/observation
    identifiers and binary labels exist only in local memory.  The official
    test split, model artifacts, predictions, and scores are never loaded.
    """

    root = Path(project_root).resolve()
    dataset = Path(dataset_root).resolve()
    clinical = Path(clinical_project_root).resolve()
    _, protocol_sha256 = validate_support_materializer_protocol(
        project_root=root,
        protocol_path=protocol_path,
        verify_bindings=True,
        verify_frozen_receipt=True,
    )
    # This is deliberately the first dataset operation: no CSV/TSV parser is
    # opened until all three concrete canonical source digests match.
    source_hashes = authenticate_canonical_dataset_sources(dataset)

    # Imports remain lazy so importing/testing this module cannot touch any
    # patient-derived source.  The APIs and their source hashes are frozen in
    # the materializer protocol and authenticated above.
    from patient_atlas_disease_folds import (
        make_disease_fold_map,
        validate_disease_fold_policy,
    )
    from patient_atlas_disease_targets import load_development_disease_targets
    from patient_atlas_real_data import load_exploratory_raw_cohort

    fold_policy, _ = validate_disease_fold_policy(root)
    cohort = load_exploratory_raw_cohort(
        project_root=root,
        dataset_root=dataset,
        clinical_project_root=clinical,
    )
    if set(str(value) for value in cohort.split_labels) - {"train", "val"}:
        raise OfficialTestRefusal("official test cohort entered support materialization")
    # Reproduce the exact existing V6.2 fold contract.  This call is only for
    # the frozen assignment; the endpoint labels below come from the explicit
    # source-aligned 0/1 observation loader, not from the closed endpoint target
    # implementation.
    fold_targets, _ = load_development_disease_targets(
        project_root=root,
        dataset_root=dataset,
        cohort=cohort,
    )
    outer_map = make_disease_fold_map(
        patient_ids=cohort.patient_ids,
        site_ids=cohort.site_ids,
        targets=fold_targets,
        policy=fold_policy,
    )
    if outer_map.assignment_sha256 != V62_OUTER_FOLD_SHA256:
        raise FoldBindingError("canonical V6.2 outer-fold assignment differs")
    dataset_participants = _load_dataset_participants(
        dataset,
        cohort=cohort,
        fold_map=outer_map,
    )
    visit_dates = _load_dataset_visit_dates(dataset)
    endpoint_observations = _load_dataset_observations(
        dataset,
        participants=dataset_participants,
        visit_dates=visit_dates,
    )
    participants = tuple(
        _Participant(item.participant_id, item.split, item.outer_fold)
        for item in dataset_participants
    )
    return _build_support_report(
        root=root,
        protocol_sha256=protocol_sha256,
        participants=participants,
        observations=endpoint_observations,
        source_hashes=source_hashes,
        source_hash_kind="canonical_dataset_source_sha256",
    )


def build_failure_payload(error: BaseException) -> dict[str, Any]:
    """Return a redacted aggregate-only failure marker."""

    return {
        "schema_version": SUPPORT_MATERIALIZER_RUNNER_SCHEMA_VERSION + "-failure-v1",
        "status": "failed_closed",
        "error_type": type(error).__name__,
        "details_emitted": False,
        "official_test_inputs_loaded": False,
        "patient_rows_or_identifiers_emitted": False,
        "patient_labels_or_arrays_emitted": False,
        "fold_assignments_emitted": False,
        "outcome_values_accessed_locally": "unknown_before_failed_closed_completion",
        "outcome_values_accessed_locally_may_have_occurred": True,
        "outcome_values_serialized": False,
        "models_scored": False,
        "predictions_or_scores_loaded": False,
        "retry_requires_preserving_this_artifact": True,
    }


__all__ = [
    "ArtifactHashError",
    "CANONICAL_COHORT_FOLD_API",
    "CANONICAL_DATASET_SOURCES",
    "DuplicateDisagreementError",
    "FROZEN_ENDPOINT_PROTOCOL_NAME",
    "FROZEN_ENDPOINT_PROTOCOL_SHA256",
    "FROZEN_PRECOMMIT_RECEIPT_NAME",
    "FROZEN_PRECOMMIT_RECEIPT_SHA256",
    "FoldBindingError",
    "IndexVisitOnlyError",
    "InputSchemaError",
    "PrecommitBindingError",
    "SUPPORT_MATERIALIZER_PROTOCOL_NAME",
    "SUPPORT_MATERIALIZER_PROTOCOL_SCHEMA_VERSION",
    "SUPPORT_MATERIALIZER_RUNNER_SCHEMA_VERSION",
    "SUPPORT_MATERIALIZER_SCHEMA_VERSION",
    "SupportMaterializerError",
    "SupportMaterializerProtocolError",
    "authenticate_canonical_dataset_sources",
    "build_failure_payload",
    "materialize_local_support",
    "materialize_local_support_from_dataset",
    "validate_support_materializer_protocol",
]
