"""Frozen, target-safe modality-partition evaluation for the V6.2 endpoint atlas.

This module is intentionally the row-free evaluation kernel.  It does not
discover endpoints or open a dataset/checkpoint; the companion runner binds
the actual V6.2 local cohort path for production, while the in-memory
:class:`EvaluationInput` seam is reserved for synthetic tests.  Patient arrays
and per-patient losses stay in process and are never included in the returned
report.

The public contract is deliberately stricter than a convenience evaluator:

* only explicit numeric 0/1 observations are accepted as labels;
* official-test partitions are refused before label access;
* target channels are physically zeroed before a representation factory sees
  them (``full_context`` removes the exact source and direct V6.2 flag;
  ``de_novo`` removes all 11 condition-history flags while retaining labs and
  vitals);
* one representation model/basis transforms outer-train and outer-test in
  each fold; and
* each of age-only, eye-only, clinical-only, and both gets a fresh nested
  train-only logistic readout.

The implementation reuses the existing V6.2 atlas metadata and grouped
logistic readout APIs selectively.  It never serializes bootstrap/multiplier
draws, endpoint loss vectors, labels, predictions, coefficients, or patient
identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from patient_atlas_disease_group_readout import (
    _binary_log_loss,
    _fit_grouped_logistic,
    _probability,
    evaluate_nested_grouped_logistic_readout,
)
from patient_atlas_disease_readout import DiseaseReadoutConfig
from patient_atlas_v6_2_expanded_endpoint_atlas import (
    EXPECTED_CANDIDATE_SOURCE_CODES,
    FREE_TEXT_SOURCE_CODES,
    LANE_DE_NOVO,
    LANE_FULL_CONTEXT,
    MIN_CASES,
    MIN_CASES_PER_OUTER_FOLD,
    MIN_CONTROLS,
    MIN_CONTROLS_PER_OUTER_FOLD,
    SMALL_CELL_THRESHOLD,
    SOURCE_TO_DIRECT_V62_FLAG,
    V62_CLINICAL_FLAG_NAMES,
    V62_OUTER_FOLD_SHA256,
    assert_aggregate_only_payload as assert_atlas_aggregate_only_payload,
    canonical_json_bytes,
    freeze_condition_candidates,
    sha256_file,
    target_specific_exclusions,
    validate_protocol as validate_atlas_protocol,
)


PROTOCOL_NAME = "BARAS_V6_2_EXPANDED_ENDPOINT_EVALUATION_V1.json"
PROTOCOL_SCHEMA_VERSION = "baras-v6-2-expanded-endpoint-evaluation-protocol-v1"
RUNNER_SCHEMA_VERSION = "baras-v6-2-expanded-endpoint-evaluation-run-v1"
EVALUATION_PREFIX = "BARAS_V6_2_EXPANDED_ENDPOINT_EVALUATION_V1"
TARGET_PREFIX = "patient_atlas_v6_2_expanded_endpoint_evaluation"

FROZEN_ATLAS_PROTOCOL_NAME = "BARAS_V6_2_EXPANDED_ENDPOINT_ATLAS_PROTOCOL_V1.json"
FROZEN_ATLAS_PROTOCOL_SHA256 = (
    "c504bfd847a0ec2c8b610da2c69371112aec9814cdda1626d6876e968f514aa5"
)
FROZEN_PRECOMMIT_RECEIPT_NAME = (
    "BARAS_V6_2_EXPANDED_ENDPOINT_ATLAS_PRECOMMIT_V1_ATTEMPT1.json"
)
FROZEN_PRECOMMIT_RECEIPT_SHA256 = (
    "6b15330de433a40d44eac9e48f4cd8864a78fbd3519e113a1be3b0aa990efcf7"
)
FROZEN_SUPPORT_RECEIPT_NAME = (
    "validation_results/BARAS_V6_2_EXPANDED_ENDPOINT_ATLAS_SUPPORT_V1_ATTEMPT1.json"
)
FROZEN_SUPPORT_RECEIPT_SHA256 = (
    "8a64c3689b1950eedda2f1d7dc942322f140f137a5cc515329b36488b5b75c3c"
)
EXPECTED_ELIGIBLE_SOURCE_CODES = (
    "mhoccur_mi",
    "mhoccur_cvdot",
    "mhoccur_strk",
    "mhoccur_clsh",
    "mhoccur_hbp",
    "mhoccur_ua",
    "mhoccur_ear",
    "mh_a1c",
    "mhterm_dm2",
    "mhterm_predm",
    "mhoccur_circ",
    "mhoccur_lbp",
    "mhoccur_cogn",
    "mhoccur_cns",
    "mhoccur_ra",
    "mhoccur_oa",
    "mhoccur_ca",
    "mhoccur_plm",
    "mhoccur_gi",
    "mhoccur_rnl",
    "mhoccur_obs",
    "mhoccur_glc",
    "mhoccur_amd",
    "mhoccur_crt",
    "mhoccur_ded",
    "mhoccur_fall",
)
EXPECTED_WITHHELD_SOURCE_CODES = tuple(
    source for source in EXPECTED_CANDIDATE_SOURCE_CODES
    if source not in EXPECTED_ELIGIBLE_SOURCE_CODES
)
# The authenticated support receipt carries a fixed, row-free claim-proximity
# audit.  Keep this contract explicit so a receipt cannot silently change the
# reporting strata while retaining the same 26 eligible endpoint names.
EXPECTED_ELIGIBLE_BY_CLAIM_PROXIMITY_CLASS = MappingProxyType(
    {
        "measurement_proximal_or_policy": 4,
        "metabolic_history_or_proximal": 2,
        "non_proximal_disease_screening": 20,
    }
)
OUTER_FOLD_COUNT = 5
EXACT_INNER_FOLD_ASSIGNMENT_SHA256 = (
    "2fdb6e82f29c6932a62fa1c347bdb569aa794c8991ad0f8c3fc5a8a17ae9cfe3",
    "f371bb5fb7ee88d5a51392941eac88f201776f573db6fab23d73e9365b807629",
    "8d7576ee85aef8a32bd182b3d6215f5091d89ca76ed37de4a686e3df7a005e10",
    "0407e4d06591ad8ab129c6556b2c369f6ca69f0bec9016994799d14e71200e40",
    "98b896c06367aeef74e1457d1f9aa22b6238cd87fbdb67d75166e99ad2cc3259",
)
CANONICAL_V6_2_EXECUTION_BINDINGS = MappingProxyType(
    {
        "screening_protocol": {
            "file": "PATIENT_ATLAS_V6_2_SCREENING_PROTOCOL_V1.json",
            "sha256": "a67aa99d95a2f77ecbd726c83a0b607d313a3e3b42676aa8e7a93e5d90cbaca9",
        },
        "cohort_loader": {
            "file": "patient_atlas_real_data.py",
            "sha256": "1c0f454ae1147c696481c55cc860c39428a2878881cd5646375af600244d0ce7",
        },
        "fold_policy_and_map": {
            "file": "patient_atlas_disease_folds.py",
            "sha256": "af9c43c39f3bae7fc2c5b6f60a382741ec36363f86e33510512e35eac8589a31",
        },
        "fold_target_loader": {
            "file": "patient_atlas_disease_targets.py",
            "sha256": "9eac8aeacf74624f028873eca38716e590e526bdac6832ed8d5ef436fdf0440a",
        },
        "prospective_policy": {
            "file": "patient_atlas_prospective_policy.py",
            "sha256": "05e8a928a04e92c4bdf8e59a63351dc29d24735e3618dda54608a5e175990553",
        },
        "training_feature_adapter": {
            "file": "patient_atlas_aireadi_internal_crossfit.py",
            "sha256": "f9e9a1d8097bd66bedb7d7323bb2cb0b517e9a898987bcb5651bb9c95cc38b30",
        },
        "v6_2_representation_factory": {
            "file": "patient_atlas_aireadi_v6_2_evaluation.py",
            "sha256": "e2a5e16ef70ad1dab8c332a2e2cac90b01946fa4124784cebaee0b9f250d1dcc",
        },
        "support_endpoint_loader": {
            "file": "patient_atlas_v6_2_expanded_endpoint_atlas_support_materializer.py",
            "sha256": "cd91c8db8354a5ea99bf6be523d0ed4a4ee424d1ded3ac56390056f7c5115303",
        },
        "screening_runner": {
            "file": "run_patient_atlas_v6_2_screening.py",
            "sha256": "c7ea70c93ebdb0766549b2ecd328d27ae9f019e53230f6c12c81f21c8ff8abfe",
        },
        "v5_training_protocol_validator": {
            "file": "patient_atlas_aireadi_v5_evaluation.py",
            "sha256": "c3b0987b57463fbaeeca77a6ea2ebc75610dff104675b94faa46b6b07ce9af8f",
        },
        "inner_fold_assignment": {
            "file": "patient_atlas_disease_universal.py",
            "sha256": "bd36b1c9a9229d08d9255f0ef6b539ccd93cc9532d6dd304079243685c857e10",
        },
        "phase_partition_and_demographic_transform": {
            "file": "eval_soft_patient_atlas.py",
            "sha256": "8173c985287b1aebbf1f981ae8352613e30338bbac5c826b218f8cebc82280bc",
        },
        "grouped_logistic_readout": {
            "file": "patient_atlas_disease_group_readout.py",
            "sha256": "c705081bb342c4d0d896f7d73e923310a5c0b866ad33f96cea18fc9c2604c41d",
        },
        "disease_readout_contract": {
            "file": "patient_atlas_disease_readout.py",
            "sha256": "7237ffcbf645719ac96a89c7b6ddf9734302e37fabaa45a9377a363bbced4bfc",
        },
    }
)

ROUTE_AGE_ONLY = "age_only"
ROUTE_EYE_ONLY = "eye_only"
ROUTE_CLINICAL_ONLY = "clinical_only"
ROUTE_BOTH = "both"
ROUTES = (ROUTE_AGE_ONLY, ROUTE_EYE_ONLY, ROUTE_CLINICAL_ONLY, ROUTE_BOTH)
MODALITY_ROUTES = (ROUTE_EYE_ONLY, ROUTE_CLINICAL_ONLY, ROUTE_BOTH)

# Positive values favour the subtracted (second-operand) route in each frozen
# formula: both for the first three and ``both_vs_age``, eye for
# ``eye_vs_clinical``.  These strings are estimand labels, never causal claims.
CONTRAST_NAMES = (
    "eye_added",
    "clinical_added",
    "both_vs_best_single",
    "eye_vs_clinical",
    "both_vs_age",
)
CONTRAST_FORMULAS = MappingProxyType(
    {
        "eye_added": "clinical_only-both",
        "clinical_added": "eye_only-both",
        "both_vs_best_single": "min(eye_only,clinical_only)-both",
        "eye_vs_clinical": "clinical_only-eye_only",
        "both_vs_age": "age_only-both",
    }
)

PRIMARY_METRIC = "proper_log_loss"
OPTIONAL_DESCRIPTIVE_METRIC = "auc_descriptive"
DEFAULT_PENALTY_GRID = (0.01, 0.1, 1.0, 10.0)
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260905
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_MAXIMUM_PENALTY_COMBINATIONS = 100
DEFAULT_PROBABILITY_CLIP = 1e-6
# Multipliers are generated in bounded row chunks.  The fixed chunk size is
# large enough to amortize the BLAS call while keeping the private KxN
# multiplication bounded for the production cohort; draws are never retained
# after the max-statistic values are updated.
MULTIPLIER_DRAW_CHUNK_SIZE = 128

# The only accepted parent fold digest.  It is repeated here so a caller
# cannot silently substitute a newly generated fold map.
EXACT_OUTER_FOLD_HASH = V62_OUTER_FOLD_SHA256

_SHA256_CHARS = frozenset("0123456789abcdef")
_OFFICIAL_SPLIT_TOKENS = frozenset(
    {"test", "official_test", "official-test", "holdout"}
)
_FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "patient_id",
        "person_id",
        "subject_id",
        "participant_id",
        "patient_ids",
        "rows",
        "records",
        "target_values",
        "outcome_values",
        "predictions",
        "model_scores",
        "losses",
        "per_patient_losses",
        "embeddings",
        "coordinates",
        "coefficients",
        "bootstrap_draws",
        "multiplier_draws",
        "fold_assignments",
    }
)
_FORBIDDEN_OUTPUT_KEYS = _FORBIDDEN_INPUT_KEYS | frozenset(
    {
        "labels",
        "probabilities",
        "draws",
        "patient_rows",
        "patient_identifiers",
    }
)


class ExpandedEndpointEvaluationError(ValueError):
    """Base error for a malformed or scientifically unsafe evaluation."""


class ProtocolBindingError(ExpandedEndpointEvaluationError):
    """Raised when the frozen parent or local code binding drifts."""


class OfficialTestRefusal(ExpandedEndpointEvaluationError):
    """Raised before any official-test label or feature is consumed."""


class DirectLabelLeakageError(ExpandedEndpointEvaluationError):
    """Raised when a target/direct feature survives or enters an unsafe seam."""


class FoldBasisError(ExpandedEndpointEvaluationError):
    """Raised when outer train/test transforms do not share one fold basis."""


class ImproperNestingError(ExpandedEndpointEvaluationError):
    """Raised when readout tuning is not wholly nested in outer training."""


class EndpointSupportDriftError(ExpandedEndpointEvaluationError):
    """Raised when support receipt and observed labels disagree."""


class PrivacyPayloadError(ExpandedEndpointEvaluationError):
    """Raised when a result or receipt contains row-level material."""


class ClassificationError(ExpandedEndpointEvaluationError):
    """Raised for malformed fail-closed partition evidence."""


class InputContractError(ExpandedEndpointEvaluationError):
    """Raised when the local in-memory evaluation input is malformed."""


class RepresentationFactory(Protocol):
    """Minimal fold-specific factory accepted by the local runner."""

    def fit(
        self,
        train_values: np.ndarray,
        *,
        feature_names: Sequence[str],
        feature_groups: Sequence[str],
        outer_fold: int,
        lane: str,
        target_source_code: str,
    ) -> Any: ...


@dataclass(frozen=True)
class EvaluationInput:
    """In-memory local evaluation contract.

    The fields are deliberately arrays rather than file paths and are a
    synthetic-test seam only.  Production execution is bound to the actual
    V6.2 cohort/fold/representation adapter in the companion runner.
    ``outer_fold_hash`` must be the authenticated V6.2 digest, not a digest
    computed from a new assignment.
    """

    feature_values: np.ndarray
    feature_names: tuple[str, ...]
    feature_groups: tuple[str, ...]
    labels_by_source: Mapping[str, np.ndarray]
    observed_by_source: Mapping[str, np.ndarray]
    outer_fold_ids: np.ndarray
    participant_splits: tuple[str, ...]
    outer_fold_hash: str = EXACT_OUTER_FOLD_HASH

    def __post_init__(self) -> None:
        values = np.asarray(self.feature_values)
        if values.ndim != 2 or values.shape[0] == 0:
            raise InputContractError("feature_values must be a non-empty 2-D matrix")
        if values.dtype.kind not in "biufc":
            raise InputContractError("feature_values must be numeric")
        if not np.isfinite(values).all():
            raise InputContractError("feature_values must be finite before preprocessing")
        names = tuple(str(item) for item in self.feature_names)
        groups = tuple(str(item) for item in self.feature_groups)
        if len(names) != values.shape[1] or not names or len(set(names)) != len(names):
            raise InputContractError("feature_names do not align to the matrix")
        if len(groups) != values.shape[1] or any(not item for item in groups):
            raise InputContractError("feature_groups do not align to the matrix")
        if not any(_is_age_group(item) for item in groups):
            raise InputContractError("an age feature group is required")
        for item in groups:
            if not _is_age_group(item) and not _is_eye_group(item) and not _is_clinical_group(item):
                raise InputContractError("feature groups must be age, eye, or clinical")
        fold_ids = np.asarray(self.outer_fold_ids)
        if fold_ids.shape != (values.shape[0],) or not np.issubdtype(fold_ids.dtype, np.integer):
            raise InputContractError("outer_fold_ids do not align to the feature matrix")
        if set(int(item) for item in np.unique(fold_ids)) - set(range(OUTER_FOLD_COUNT)):
            raise InputContractError("outer_fold_ids are outside the frozen five-fold map")
        if any(int((fold_ids == fold).sum()) == 0 for fold in range(OUTER_FOLD_COUNT)):
            raise InputContractError("every frozen outer fold must have test patients")
        splits = tuple(str(item).strip().lower() for item in self.participant_splits)
        if len(splits) != values.shape[0]:
            raise InputContractError("participant_splits do not align to the feature matrix")
        if any(item in _OFFICIAL_SPLIT_TOKENS for item in splits):
            raise OfficialTestRefusal("official-test partition is refused")
        if set(splits) - {"train", "val", "validation"}:
            raise InputContractError("only train/validation participants are accepted")
        if not _is_sha256(self.outer_fold_hash) or self.outer_fold_hash != EXACT_OUTER_FOLD_HASH:
            raise ProtocolBindingError("only the exact frozen V6.2 outer-fold hash is accepted")
        sources = set(EXPECTED_CANDIDATE_SOURCE_CODES)
        if set(self.labels_by_source) - sources or set(self.observed_by_source) - sources:
            raise InputContractError("labels contain an undeclared endpoint")
        if set(self.labels_by_source) != set(self.observed_by_source):
            raise InputContractError("labels and observed masks have different endpoint sets")
        for source in EXPECTED_CANDIDATE_SOURCE_CODES:
            labels = np.asarray(self.labels_by_source[source])
            observed = np.asarray(self.observed_by_source[source])
            if labels.shape != (values.shape[0],) or observed.shape != labels.shape:
                raise InputContractError("endpoint labels do not align to the feature matrix")
            if observed.dtype != np.bool_:
                raise InputContractError("observed masks must be boolean")
            _validate_explicit_binary_labels(labels, observed, source)

    @property
    def size(self) -> int:
        return int(np.asarray(self.feature_values).shape[0])


@dataclass(frozen=True)
class RepresentationCoordinates:
    """Output contract for one fold model transform."""

    values: np.ndarray
    feature_groups: tuple[str, ...]
    fold_id: int
    model_token: str
    basis_token: str
    outer_train_only: bool = True
    target_values_seen: bool = False
    outer_test_seen_during_fit: bool = False

    def __post_init__(self) -> None:
        matrix = np.asarray(self.values)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.isfinite(matrix).all():
            raise FoldBasisError("representation coordinates must be finite 2-D values")
        if len(self.feature_groups) != matrix.shape[1] or any(not item for item in self.feature_groups):
            raise FoldBasisError("representation coordinate groups do not align")
        if not isinstance(self.fold_id, (int, np.integer)) or isinstance(self.fold_id, bool):
            raise FoldBasisError("representation fold id is malformed")
        if not self.model_token or not self.basis_token:
            raise FoldBasisError("representation provenance tokens are required")
        if self.target_values_seen or self.outer_test_seen_during_fit or not self.outer_train_only:
            raise DirectLabelLeakageError("representation provenance records unsafe material")


@dataclass(frozen=True)
class ReadoutResult:
    """Private readout result; ``losses`` never crosses the aggregate boundary."""

    route: str
    selected_penalties: Mapping[str, float]
    inner_primary_loss: float
    test_losses: np.ndarray
    test_auc: float | None
    test_eligible_count: int
    test_event_count: int


@dataclass(frozen=True)
class EligibleSupport:
    """Authenticated aggregate eligibility metadata loaded for one run."""

    receipt: Mapping[str, Any]
    receipt_sha256: str | None
    eligible_sources: tuple[str, ...]
    support_by_source: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class InferenceResult:
    """Aggregate simultaneous inference; draw arrays remain private."""

    point: Mapping[str, float]
    lower: Mapping[str, float]
    upper: Mapping[str, float]
    simultaneous_passed: Mapping[str, bool]
    test_count: int
    bootstrap_samples: int
    bootstrap_seed: int
    confidence_level: float


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_CHARS


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _is_age_group(group: str) -> bool:
    token = str(group).strip().lower()
    return token in {"age", "demographics", "demographic", "age_evidence"}


def _is_eye_group(group: str) -> bool:
    token = str(group).strip().lower()
    return token.startswith("eye") or token in {"retinal", "retinal_evidence"}


def _is_clinical_group(group: str) -> bool:
    token = str(group).strip().lower()
    return token.startswith("clinical") or token in {"blood", "clinical_evidence", "blood_evidence"}


def _group_family(group: str) -> str:
    if _is_age_group(group):
        return "age"
    if _is_eye_group(group):
        return "eye"
    if _is_clinical_group(group):
        return "clinical"
    raise InputContractError("unknown modality feature group")


def _validate_explicit_binary_labels(labels: np.ndarray, observed: np.ndarray, source: str) -> None:
    if labels.ndim != 1 or observed.ndim != 1:
        raise InputContractError(f"{source} labels must be vectors")
    values = labels[observed]
    if values.size == 0:
        return
    if values.dtype.kind not in "iuf" or values.dtype.kind == "b":
        raise InputContractError(f"{source} labels must be numeric")
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise InputContractError(f"{source} labels must be explicit numeric 0/1")


def validate_development_partitions(participant_splits: Sequence[str]) -> None:
    """Refuse official-test membership before a caller accesses labels."""

    for value in participant_splits:
        token = str(value).strip().lower()
        if token in _OFFICIAL_SPLIT_TOKENS:
            raise OfficialTestRefusal("official-test partition is refused")
        if token not in {"train", "val", "validation"}:
            raise InputContractError("only train/validation partitions are allowed")


def _forbidden_payload_walk(value: Any, *, output: bool, where: str) -> None:
    keys = _FORBIDDEN_OUTPUT_KEYS if output else _FORBIDDEN_INPUT_KEYS
    if isinstance(value, Mapping):
        for key, child in value.items():
            token = str(key).strip().lower()
            if token in keys:
                raise PrivacyPayloadError(f"{where} contains a forbidden payload field")
            if token in {"official_test", "official_test_included", "is_official_test"}:
                if child not in (False, None, 0, "", "false", "False", "0"):
                    raise OfficialTestRefusal("official-test material is refused")
            if token in {"split", "cohort", "recommended_split"}:
                if str(child).strip().lower() in _OFFICIAL_SPLIT_TOKENS:
                    raise OfficialTestRefusal("official-test material is refused")
            _forbidden_payload_walk(child, output=output, where=where)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _forbidden_payload_walk(child, output=output, where=where)


def assert_aggregate_only_result(value: Any) -> None:
    """Reject patient-derived fields before a result is written."""

    if not isinstance(value, Mapping):
        raise PrivacyPayloadError("aggregate output must be a JSON object")
    _forbidden_payload_walk(value, output=True, where="output")
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    for token in ('"patient_id":', '"patient_ids":', '"labels":', '"losses":', '"draws":'):
        if token in encoded:
            raise PrivacyPayloadError("aggregate output contains a forbidden token")
    if value.get("official_test_inputs_loaded") is not False:
        raise OfficialTestRefusal("output does not attest official-test refusal")
    if value.get("patient_rows_or_identifiers_emitted") is not False:
        raise PrivacyPayloadError("output does not attest row-free emission")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def safe_count(value: int) -> int | str:
    count = int(value)
    if count < 0:
        raise PrivacyPayloadError("counts cannot be negative")
    return count if count == 0 or count >= SMALL_CELL_THRESHOLD else f"<{SMALL_CELL_THRESHOLD}"


def _safe_name(value: Any) -> str:
    name = str(value).strip()
    if not name or len(name) > 240:
        raise InputContractError("feature name is malformed")
    return name


def physically_erase_target_channels(
    feature_values: Any,
    feature_names: Sequence[str],
    *,
    target_source_code: str,
    lane: str,
    observed_mask: Any | None = None,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray | None]:
    """Erase direct channels on a copy before preprocessing/encoding.

    The exact survey source is removed whenever present.  ``full_context``
    additionally removes only the source's direct V6.2 flag; ``de_novo``
    removes all 11 flags.  Continuous laboratories/vitals are not matched by
    a broad substring and therefore remain in the de-novo lane.
    """

    source = str(target_source_code).strip()
    if source not in EXPECTED_CANDIDATE_SOURCE_CODES or source in FREE_TEXT_SOURCE_CODES:
        raise DirectLabelLeakageError("target source is outside the frozen candidate set")
    lane_name = str(lane).strip().lower()
    if lane_name not in (LANE_FULL_CONTEXT, LANE_DE_NOVO):
        raise DirectLabelLeakageError("lane must be full_context or de_novo")
    names = tuple(_safe_name(name) for name in feature_names)
    if len(names) == 0 or len(set(names)) != len(names):
        raise DirectLabelLeakageError("feature names must be non-empty and unique")
    matrix = np.asarray(feature_values)
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or matrix.dtype.kind not in "biufc":
        raise DirectLabelLeakageError("feature matrix does not align to feature names")
    if not np.isfinite(matrix).all():
        raise DirectLabelLeakageError("feature matrix must be finite before erasure")
    # A direct target-shaped field is unsafe even when the caller tries to
    # hide it under a common label field name.
    for name in names:
        if name.strip().lower() in {"target", "target_value", "target_values", "label", "labels", "outcome", "outcome_values"}:
            raise DirectLabelLeakageError("target/label feature field is forbidden")
    if lane_name == LANE_FULL_CONTEXT:
        excluded = set(target_specific_exclusions(source)["full_context_excluded_features"])
    else:
        excluded = set(target_specific_exclusions(source)["de_novo_excluded_features"])
    excluded_lower = {str(item).strip().lower() for item in excluded}
    keep = np.asarray([name.strip().lower() not in excluded_lower for name in names], dtype=bool)
    safe = np.asarray(matrix, dtype=np.float64).copy()
    safe[:, ~keep] = 0.0
    safe_mask: np.ndarray | None = None
    if observed_mask is not None:
        safe_mask = np.asarray(observed_mask)
        if safe_mask.shape != safe.shape or safe_mask.dtype != np.bool_:
            raise DirectLabelLeakageError("feature observed mask does not align")
        safe_mask = safe_mask.copy()
        safe_mask[:, ~keep] = False
    if np.any(safe[:, ~keep] != 0.0) or safe_mask is not None and np.any(safe_mask[:, ~keep]):
        raise DirectLabelLeakageError("direct target channels survived physical erasure")
    return safe, names, safe_mask


def assert_target_channels_erased(
    values: Any,
    feature_names: Sequence[str],
    *,
    target_source_code: str,
    lane: str,
    observed_mask: Any | None = None,
) -> None:
    """Fail closed if a forbidden target/direct feature remains nonzero."""

    source = str(target_source_code).strip()
    lane_name = str(lane).strip().lower()
    if source not in EXPECTED_CANDIDATE_SOURCE_CODES or source in FREE_TEXT_SOURCE_CODES:
        raise DirectLabelLeakageError("target source is outside the frozen candidate set")
    if lane_name not in (LANE_FULL_CONTEXT, LANE_DE_NOVO):
        raise DirectLabelLeakageError("lane must be full_context or de_novo")
    excluded = set(
        target_specific_exclusions(source)[
            "full_context_excluded_features" if lane_name == LANE_FULL_CONTEXT else "de_novo_excluded_features"
        ]
    )
    names = tuple(str(item) for item in feature_names)
    matrix = np.asarray(values)
    forbidden = np.asarray([name.strip().lower() in {str(item).strip().lower() for item in excluded} for name in names], dtype=bool)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise DirectLabelLeakageError("feature matrix does not align to feature names")
    if np.any(matrix[:, forbidden] != 0.0):
        raise DirectLabelLeakageError("forbidden feature survived erasure")
    if observed_mask is not None:
        mask = np.asarray(observed_mask)
        if mask.shape != matrix.shape or mask.dtype != np.bool_:
            raise DirectLabelLeakageError("feature observed mask does not align")
        if np.any(mask[:, forbidden]):
            raise DirectLabelLeakageError("forbidden feature observation mask survived erasure")


def _call_factory_fit(
    factory: Any,
    train_values: np.ndarray,
    *,
    feature_names: Sequence[str],
    feature_groups: Sequence[str],
    outer_fold: int,
    lane: str,
    target_source_code: str,
) -> Any:
    fit = getattr(factory, "fit", None)
    if not callable(fit):
        raise FoldBasisError("representation factory must expose fit")
    try:
        return fit(
            train_values,
            feature_names=feature_names,
            feature_groups=feature_groups,
            outer_fold=outer_fold,
            lane=lane,
            target_source_code=target_source_code,
        )
    except TypeError as error:
        # A narrower adapter may use a positional contract.  Do not silently
        # pass labels or test data through either form.
        try:
            return fit(
                train_values,
                feature_names=feature_names,
                outer_fold=outer_fold,
                lane=lane,
                target_source_code=target_source_code,
            )
        except TypeError:
            try:
                return fit(
                    train_values,
                    outer_fold=outer_fold,
                    lane=lane,
                    target_source_code=target_source_code,
                )
            except TypeError:
                try:
                    return fit(train_values, feature_names, feature_groups, outer_fold, lane, target_source_code)
                except TypeError:
                    raise FoldBasisError("representation factory fit signature differs") from error


def _call_model_transform(
    model: Any,
    values: np.ndarray,
    *,
    route: str,
    is_outer_test: bool,
) -> RepresentationCoordinates:
    transform = getattr(model, "transform", None)
    if not callable(transform):
        raise FoldBasisError("fitted representation must expose transform")
    try:
        raw = transform(values, route=route, is_outer_test=is_outer_test)
    except TypeError as first_error:
        try:
            raw = transform(values, route=route)
        except TypeError:
            try:
                raw = transform(values, route)
            except TypeError:
                raise FoldBasisError("representation transform signature differs") from first_error
    if isinstance(raw, RepresentationCoordinates):
        result = raw
    elif isinstance(raw, Mapping):
        try:
            result = RepresentationCoordinates(
                values=np.asarray(raw["values"]),
                feature_groups=tuple(str(item) for item in raw["feature_groups"]),
                fold_id=int(raw["fold_id"]),
                model_token=str(raw["model_token"]),
                basis_token=str(raw["basis_token"]),
                outer_train_only=bool(raw.get("outer_train_only", True)),
                target_values_seen=bool(raw.get("target_values_seen", False)),
                outer_test_seen_during_fit=bool(raw.get("outer_test_seen_during_fit", False)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise FoldBasisError("representation transform provenance is malformed") from error
    else:
        raise FoldBasisError("representation transform must return provenance-tagged coordinates")
    if result.values.shape[0] != len(values):
        raise FoldBasisError("representation transform changed patient count")
    return result


def _validate_fold_pair(
    train: RepresentationCoordinates,
    test: RepresentationCoordinates,
    *,
    outer_fold: int,
    route: str,
) -> None:
    if train.fold_id != outer_fold or test.fold_id != outer_fold:
        raise FoldBasisError("coordinates were produced by the wrong outer fold")
    if train.model_token != test.model_token or train.basis_token != test.basis_token:
        raise FoldBasisError("outer train/test coordinates do not share one fitted basis")
    if train.feature_groups != test.feature_groups:
        raise FoldBasisError("outer train/test coordinate groups differ")
    if train.values.shape[1] != test.values.shape[1]:
        raise FoldBasisError("outer train/test coordinate widths differ")
    if not route:
        raise FoldBasisError("route provenance is empty")


def _validate_outer_fit_provenance(model: Any, outer_fold: int) -> None:
    """Reject pooled or target-aware factories before transforms are used."""

    if hasattr(model, "outer_fold") and int(getattr(model, "outer_fold")) != outer_fold:
        raise FoldBasisError("representation model is not fold-specific")
    if hasattr(model, "fit_outer_fold") and int(getattr(model, "fit_outer_fold")) != outer_fold:
        raise FoldBasisError("representation was fit on the wrong outer fold")
    for attr in ("target_values_seen", "used_outer_test", "outer_test_seen_during_fit"):
        if bool(getattr(model, attr, False)):
            raise DirectLabelLeakageError("representation fit provenance records unsafe values")
    if hasattr(model, "fit_outer_train_only") and not bool(getattr(model, "fit_outer_train_only")):
        raise FoldBasisError("representation fit was not outer-train-only")
    if hasattr(model, "fit_fold_ids"):
        fold_ids = tuple(int(item) for item in getattr(model, "fit_fold_ids"))
        if outer_fold in fold_ids or set(fold_ids) - set(range(OUTER_FOLD_COUNT)):
            raise FoldBasisError("representation fit fold membership includes outer test")


def _route_indices(groups: Sequence[str], route: str) -> np.ndarray:
    families = np.asarray([_group_family(item) for item in groups], dtype=object)
    if route == ROUTE_AGE_ONLY:
        keep = families == "age"
    elif route == ROUTE_EYE_ONLY:
        keep = np.isin(families, ("eye", "age"))
    elif route == ROUTE_CLINICAL_ONLY:
        keep = np.isin(families, ("clinical", "age"))
    elif route == ROUTE_BOTH:
        keep = np.ones(len(groups), dtype=bool)
    else:
        raise InputContractError("unprespecified modality route")
    indices = np.flatnonzero(keep)
    if not len(indices) or not np.any(families[indices] == "age"):
        raise InputContractError("every modality route must retain age")
    if route == ROUTE_EYE_ONLY and not np.any(families[indices] == "eye"):
        raise InputContractError("eye-only route has no eye evidence")
    if route == ROUTE_CLINICAL_ONLY and not np.any(families[indices] == "clinical"):
        raise InputContractError("clinical-only route has no clinical evidence")
    if route == ROUTE_BOTH and not (np.any(families == "eye") and np.any(families == "clinical")):
        raise InputContractError("both route requires eye and clinical evidence")
    return indices


def route_design(values: Any, feature_groups: Sequence[str], route: str) -> tuple[np.ndarray, tuple[str, ...]]:
    """Select one route from a single fold's shared representation basis."""

    matrix = np.asarray(values)
    groups = tuple(str(item) for item in feature_groups)
    if matrix.ndim != 2 or matrix.shape[1] != len(groups) or not np.isfinite(matrix).all():
        raise FoldBasisError("route design values/groups do not align")
    indices = _route_indices(groups, route)
    selected = matrix[:, indices].copy()
    # Downstream grouped readout tuning has one stable public spelling per
    # modality.  Preserve the coordinate basis privately, but canonicalize
    # aliases such as ``eye_evidence``/``clinical_evidence`` at this route
    # boundary so age remains exactly the unpenalized ``age`` group.
    selected_groups = tuple(_group_family(groups[int(index)]) for index in indices)
    if selected.shape[1] != len(selected_groups):
        raise FoldBasisError("route design width differs")
    return selected, selected_groups


def _auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    sorted_scores = score[order]
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[position]:
            end += 1
        ranks[order[position:end]] = 0.5 * (position + end - 1) + 1.0
        position = end
    positives = float(np.sum(y == 1.0))
    negatives = float(np.sum(y == 0.0))
    return float((np.sum(ranks[y == 1.0]) - positives * (positives + 1.0) / 2.0) / (positives * negatives))


def _age_only_readout(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_eligible: np.ndarray,
    inner_fold_ids: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_eligible: np.ndarray,
    config: DiseaseReadoutConfig,
    route: str,
) -> ReadoutResult:
    """Age has no penalized group but still follows outer-train-only CV."""

    train_x = np.asarray(x_train, dtype=np.float64)
    test_x = np.asarray(x_test, dtype=np.float64)
    train_y = np.asarray(y_train, dtype=np.float64)
    test_y = np.asarray(y_test, dtype=np.float64)
    train_mask = np.asarray(train_eligible, dtype=bool)
    test_mask = np.asarray(test_eligible, dtype=bool)
    folds = np.asarray(inner_fold_ids)
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != 1 or test_x.shape[1] != 1:
        raise ImproperNestingError("age-only readout must have one age coordinate")
    if train_x.shape[0] != len(train_y) or test_x.shape[0] != len(test_y) or train_mask.shape != train_y.shape or test_mask.shape != test_y.shape or folds.shape != train_y.shape:
        raise ImproperNestingError("age-only readout arrays do not align")
    if not train_mask.any() or not test_mask.any() or set(np.unique(train_y[train_mask])) != {0.0, 1.0}:
        raise ImproperNestingError("age-only readout lacks a binary outer training set")
    populated = tuple(sorted(int(item) for item in np.unique(folds[train_mask])))
    if len(populated) < 2:
        raise ImproperNestingError("age-only readout requires at least two inner folds")
    fold_losses: list[float] = []
    fold_counts: list[int] = []
    for fold in populated:
        validation = train_mask & (folds == fold)
        training = train_mask & (folds != fold)
        if not validation.any() or set(np.unique(train_y[training])) != {0.0, 1.0}:
            raise ImproperNestingError("age-only inner training split lacks both classes")
        model = _fit_grouped_logistic(
            train_x[training], train_y[training], np.zeros(1, dtype=np.float64), config
        )
        probability = _probability(model, train_x[validation], config)
        fold_losses.append(_binary_log_loss(train_y[validation], probability))
        fold_counts.append(int(validation.sum()))
    final_model = _fit_grouped_logistic(
        train_x[train_mask], train_y[train_mask], np.zeros(1, dtype=np.float64), config
    )
    probability = _probability(final_model, test_x[test_mask], config)
    losses = np.full(len(test_y), np.nan, dtype=np.float64)
    losses[test_mask] = -(
        test_y[test_mask] * np.log(probability)
        + (1.0 - test_y[test_mask]) * np.log1p(-probability)
    )
    return ReadoutResult(
        route=route,
        selected_penalties=MappingProxyType({}),
        inner_primary_loss=float(np.average(fold_losses, weights=fold_counts)),
        test_losses=losses,
        test_auc=_auc(test_y[test_mask], probability),
        test_eligible_count=int(test_mask.sum()),
        test_event_count=int(test_y[test_mask].sum()),
    )


def nested_logistic_readout(
    *,
    route: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_eligible: np.ndarray,
    inner_fold_ids: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_eligible: np.ndarray,
    feature_groups: Sequence[str],
    penalty_grid: Sequence[float] = DEFAULT_PENALTY_GRID,
    config: DiseaseReadoutConfig | None = None,
    maximum_penalty_combinations: int = DEFAULT_MAXIMUM_PENALTY_COMBINATIONS,
) -> ReadoutResult:
    """Fit one fresh nested grouped-logistic readout for one route.

    Selection sees only outer-train labels.  The outer-test labels are used
    after the final fit solely to compute private held-out losses.
    """

    if route not in ROUTES:
        raise ImproperNestingError("route is not prespecified")
    raw_train_y = np.asarray(y_train)
    raw_test_y = np.asarray(y_test)
    train_x = np.asarray(x_train, dtype=np.float64)
    test_x = np.asarray(x_test, dtype=np.float64)
    train_y = np.asarray(y_train, dtype=np.float64)
    test_y = np.asarray(y_test, dtype=np.float64)
    train_mask = np.asarray(train_eligible, dtype=bool)
    test_mask = np.asarray(test_eligible, dtype=bool)
    folds = np.asarray(inner_fold_ids)
    groups = tuple(str(item) for item in feature_groups)
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != test_x.shape[1] or train_x.shape[1] != len(groups):
        raise ImproperNestingError("readout design/groups do not align")
    if train_x.shape[0] != len(train_y) or test_x.shape[0] != len(test_y) or train_mask.shape != train_y.shape or test_mask.shape != test_y.shape or folds.shape != train_y.shape:
        raise ImproperNestingError("readout arrays do not align")
    if not np.issubdtype(folds.dtype, np.integer) or set(int(item) for item in np.unique(folds)) - set(range(OUTER_FOLD_COUNT)):
        raise ImproperNestingError("inner fold ids are malformed")
    _validate_explicit_binary_labels(raw_train_y, train_mask, "outer-train")
    _validate_explicit_binary_labels(raw_test_y, test_mask, "outer-test")
    if not train_mask.any() or not test_mask.any() or set(np.unique(train_y[train_mask])) != {0.0, 1.0}:
        raise ImproperNestingError("readout requires both binary outer-training classes")
    if set(np.unique(test_y[test_mask])) - {0.0, 1.0}:
        raise ImproperNestingError("readout test labels are not binary")
    if route == ROUTE_AGE_ONLY:
        if len(groups) != 1 or not _is_age_group(groups[0]):
            raise ImproperNestingError("age-only design/group contract differs")
        return _age_only_readout(
            x_train=train_x,
            y_train=train_y,
            train_eligible=train_mask,
            inner_fold_ids=folds,
            x_test=test_x,
            y_test=test_y,
            test_eligible=test_mask,
            config=config or DiseaseReadoutConfig(probability_clip=DEFAULT_PROBABILITY_CLIP),
            route=route,
        )
    if route == ROUTE_EYE_ONLY and not any(_is_eye_group(item) for item in groups):
        raise ImproperNestingError("eye-only design lacks eye group")
    if route == ROUTE_CLINICAL_ONLY and not any(_is_clinical_group(item) for item in groups):
        raise ImproperNestingError("clinical-only design lacks clinical group")
    if route == ROUTE_BOTH and not (any(_is_eye_group(item) for item in groups) and any(_is_clinical_group(item) for item in groups)):
        raise ImproperNestingError("both design lacks both modality groups")
    if config is None:
        config = DiseaseReadoutConfig(
            penalty_grid=tuple(float(item) for item in penalty_grid),
            probability_clip=DEFAULT_PROBABILITY_CLIP,
        )
    else:
        config = DiseaseReadoutConfig(
            penalty_grid=tuple(float(item) for item in penalty_grid),
            logistic_max_iterations=config.logistic_max_iterations,
            logistic_tolerance=config.logistic_tolerance,
            probability_clip=config.probability_clip,
            minimum_disclosable_cell_count=config.minimum_disclosable_cell_count,
        )
    try:
        sink = np.empty(len(test_y), dtype=np.float64)
        result = evaluate_nested_grouped_logistic_readout(
            x_train=train_x,
            y_train=train_y,
            train_eligible=train_mask,
            inner_fold_ids=folds,
            x_test=test_x,
            y_test=test_y,
            test_eligible=test_mask,
            feature_groups=groups,
            unpenalized_groups=("age",),
            test_patient_id_hash="0" * 64,
            config=config,
            maximum_penalty_combinations=int(maximum_penalty_combinations),
            _private_test_primary_loss_out=sink,
        )
    except Exception as error:
        # The grouped evaluator is a nested primitive; expose a stable seam so
        # a caller cannot mistake a malformed nested fit for a valid score.
        if isinstance(error, (ImproperNestingError, DirectLabelLeakageError)):
            raise
        raise ImproperNestingError("nested grouped readout failed") from error
    if not np.array_equal(np.isfinite(sink), test_mask):
        raise ImproperNestingError("nested readout did not preserve test eligibility")
    metrics = result.metrics
    return ReadoutResult(
        route=route,
        selected_penalties=MappingProxyType(dict(result.selected_penalties)),
        inner_primary_loss=float(result.inner_primary_loss),
        test_losses=sink.copy(),
        test_auc=(None if metrics.get("auroc") is None else float(metrics["auroc"])),
        test_eligible_count=int(test_mask.sum()),
        test_event_count=int(test_y[test_mask].sum()),
    )


def make_inner_fold_ids(
    outer_fold_ids: Sequence[int],
    train_indices: Sequence[int],
    *,
    outer_fold: int,
) -> np.ndarray:
    """Create deterministic train-only inner folds without using test rows."""

    outer = np.asarray(outer_fold_ids)
    indices = np.asarray(train_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(outer)):
        raise ImproperNestingError("inner-fold train indices are malformed")
    if np.any(outer[indices] == outer_fold):
        raise ImproperNestingError("outer-test identity entered inner tuning")
    # A stable cyclic assignment is used only as a local fallback when the
    # caller has not supplied a precomputed V6.2 inner map.  It is never
    # serialized and never changes the authenticated outer fold assignment.
    order = np.asarray(indices, dtype=np.int64)
    values = np.empty(len(order), dtype=np.int64)
    for position, index in enumerate(order):
        values[position] = (int(index) * 2654435761 + int(outer_fold) * 17) % OUTER_FOLD_COUNT
    return values


def _support_record_from_receipt(receipt: Mapping[str, Any], source: str) -> Mapping[str, Any]:
    audit = receipt.get("support_audit")
    if not isinstance(audit, Mapping):
        raise EndpointSupportDriftError("support receipt lacks support_audit")
    candidates = audit.get("candidates")
    if not isinstance(candidates, Mapping) or source not in candidates:
        raise EndpointSupportDriftError("support receipt lacks a frozen candidate record")
    row = candidates[source]
    if not isinstance(row, Mapping):
        raise EndpointSupportDriftError("support receipt candidate record is malformed")
    return row


def _eligible_from_receipt(receipt: Mapping[str, Any]) -> tuple[tuple[str, ...], Mapping[str, Mapping[str, Any]]]:
    expected = tuple(EXPECTED_CANDIDATE_SOURCE_CODES)
    if receipt.get("candidate_count") != len(expected) or tuple(receipt.get("candidate_source_codes", ())) != expected:
        raise EndpointSupportDriftError("support receipt candidate set/order differs")
    if receipt.get("outer_fold_assignment_sha256") != EXACT_OUTER_FOLD_HASH:
        raise EndpointSupportDriftError("support receipt outer fold hash differs")
    if receipt.get("protocol_sha256") != FROZEN_ATLAS_PROTOCOL_SHA256:
        raise EndpointSupportDriftError("support receipt parent protocol differs")
    if receipt.get("precommit_receipt_sha256") != FROZEN_PRECOMMIT_RECEIPT_SHA256:
        raise EndpointSupportDriftError("support receipt parent precommit differs")
    schema = str(receipt.get("schema_version", ""))
    materializer_schema = "baras-v6-2-expanded-endpoint-atlas-support-materializer-v1"
    synthetic_schema = "baras-v6-2-expanded-endpoint-evaluation-eligible-support-receipt-v1"
    if schema not in {materializer_schema, synthetic_schema}:
        raise EndpointSupportDriftError("support receipt schema is not recognized")
    if receipt.get("selection_uses_model_performance") not in (False, None) or receipt.get("selection_uses_outcome_values") not in (False, None):
        raise EndpointSupportDriftError("support receipt selection provenance differs")
    audit = receipt.get("support_audit")
    audit_outcome_accessed = (
        audit.get("outcome_values_accessed")
        if isinstance(audit, Mapping) and "outcome_values_accessed" in audit
        else audit.get("outcome_values_accessed_locally")
        if isinstance(audit, Mapping)
        else None
    )
    audit_selection_uses_support = (
        audit.get("selection_uses_support")
        if isinstance(audit, Mapping) and "selection_uses_support" in audit
        else audit.get("candidate_registry_selection_uses_support")
        if isinstance(audit, Mapping)
        else None
    )
    expected_audit_outcome = True if schema == materializer_schema else False
    if (
        not isinstance(audit, Mapping)
        or audit_outcome_accessed is not expected_audit_outcome
        or audit_selection_uses_support is not False
        or audit.get("all_declared_candidates_retained") is not True
    ):
        raise EndpointSupportDriftError("support receipt audit provenance differs")
    eligible: list[str] = []
    support: dict[str, Mapping[str, Any]] = {}
    for source in expected:
        row = _support_record_from_receipt(receipt, source)
        status = row.get("status")
        if status == "eligible_under_frozen_support_gates":
            detail = row.get("support")
            if not isinstance(detail, Mapping):
                raise EndpointSupportDriftError("eligible endpoint lacks support counts")
            _validate_support_counts(detail, source)
            eligible.append(source)
            support[source] = detail
        elif status in {"withheld_under_frozen_support_gates", "missing_or_withheld", "support_not_supplied"}:
            support[source] = row.get("support", {}) if isinstance(row.get("support", {}), Mapping) else {}
        else:
            raise EndpointSupportDriftError("support receipt has an unknown eligibility status")
    if schema == materializer_schema and tuple(eligible) != EXPECTED_ELIGIBLE_SOURCE_CODES:
        raise EndpointSupportDriftError("authenticated support receipt eligible set differs")
    if schema == materializer_schema:
        metadata = frozen_candidate_metadata()
        observed_class_counts = {
            claim_class: sum(
                1
                for source in eligible
                if metadata[source]["claim_proximity_class"] == claim_class
            )
            for claim_class in EXPECTED_ELIGIBLE_BY_CLAIM_PROXIMITY_CLASS
        }
        if observed_class_counts != dict(EXPECTED_ELIGIBLE_BY_CLAIM_PROXIMITY_CLASS):
            raise EndpointSupportDriftError(
                "authenticated support receipt claim-proximity class counts differ"
            )
    return tuple(eligible), MappingProxyType(support)


def _validate_support_counts(detail: Mapping[str, Any], source: str) -> None:
    def integer(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise EndpointSupportDriftError(f"{source} support {label} is not an integer")
        if int(value) < 0:
            raise EndpointSupportDriftError(f"{source} support {label} is negative")
        return int(value)

    cases = integer(detail.get("cases"), "cases")
    controls = integer(detail.get("controls"), "controls")
    cases_by_fold = detail.get("cases_by_outer_fold")
    controls_by_fold = detail.get("controls_by_outer_fold")
    if not isinstance(cases_by_fold, (list, tuple)) or not isinstance(controls_by_fold, (list, tuple)) or len(cases_by_fold) != OUTER_FOLD_COUNT or len(controls_by_fold) != OUTER_FOLD_COUNT:
        raise EndpointSupportDriftError(f"{source} support fold counts are malformed")
    if sum(integer(item, "cases_by_outer_fold") for item in cases_by_fold) != cases or sum(integer(item, "controls_by_outer_fold") for item in controls_by_fold) != controls:
        raise EndpointSupportDriftError(f"{source} support totals do not match fold counts")
    if detail.get("negative_label_status") != "validated_binary" or detail.get("label_encoding") not in ("explicit_numeric_0_1_on_observed_rows", None):
        raise EndpointSupportDriftError(f"{source} support label policy differs")
    if cases < MIN_CASES or controls < MIN_CONTROLS or min(int(item) for item in cases_by_fold) < MIN_CASES_PER_OUTER_FOLD or min(int(item) for item in controls_by_fold) < MIN_CONTROLS_PER_OUTER_FOLD:
        raise EndpointSupportDriftError(f"{source} is marked eligible below frozen support gates")


def build_synthetic_support_receipt(
    labels_by_source: Mapping[str, np.ndarray],
    observed_by_source: Mapping[str, np.ndarray],
    outer_fold_ids: np.ndarray,
    *,
    eligible_sources: Iterable[str] | None = None,
) -> Mapping[str, Any]:
    """Build a row-free support receipt for synthetic tests/local adapters."""

    folds = np.asarray(outer_fold_ids)
    if folds.ndim != 1:
        raise EndpointSupportDriftError("synthetic support folds are malformed")
    desired = set(EXPECTED_CANDIDATE_SOURCE_CODES if eligible_sources is None else (str(item) for item in eligible_sources))
    if not desired <= set(EXPECTED_CANDIDATE_SOURCE_CODES):
        raise EndpointSupportDriftError("synthetic support names differ")
    candidates: dict[str, Any] = {}
    for source in EXPECTED_CANDIDATE_SOURCE_CODES:
        labels = np.asarray(labels_by_source[source])
        observed = np.asarray(observed_by_source[source], dtype=bool)
        if labels.shape != folds.shape or observed.shape != folds.shape:
            raise EndpointSupportDriftError("synthetic support arrays do not align")
        _validate_explicit_binary_labels(labels, observed, source)
        cases_by_fold = [int(np.sum(observed & (labels == 1) & (folds == fold))) for fold in range(OUTER_FOLD_COUNT)]
        controls_by_fold = [int(np.sum(observed & (labels == 0) & (folds == fold))) for fold in range(OUTER_FOLD_COUNT)]
        cases, controls = sum(cases_by_fold), sum(controls_by_fold)
        eligible = source in desired
        gate = cases >= MIN_CASES and controls >= MIN_CONTROLS and min(cases_by_fold) >= MIN_CASES_PER_OUTER_FOLD and min(controls_by_fold) >= MIN_CONTROLS_PER_OUTER_FOLD
        status = "eligible_under_frozen_support_gates" if eligible and gate else "withheld_under_frozen_support_gates"
        candidates[source] = {
            "status": status,
            "support": {
                "cases": cases,
                "controls": controls,
                "cases_by_outer_fold": cases_by_fold,
                "controls_by_outer_fold": controls_by_fold,
                "coverage_status": "complete" if bool(observed.all()) else "partial",
                "negative_label_status": "validated_binary" if bool(observed.any()) else "missing_or_excluded",
                "label_encoding": "explicit_numeric_0_1_on_observed_rows" if bool(observed.any()) else None,
            },
        }
    return {
        "schema_version": "baras-v6-2-expanded-endpoint-evaluation-eligible-support-receipt-v1",
        "status": "authenticated_local_support_materialization_complete",
        "protocol_sha256": FROZEN_ATLAS_PROTOCOL_SHA256,
        "precommit_receipt_sha256": FROZEN_PRECOMMIT_RECEIPT_SHA256,
        "outer_fold_assignment_sha256": EXACT_OUTER_FOLD_HASH,
        "candidate_count": len(EXPECTED_CANDIDATE_SOURCE_CODES),
        "candidate_source_codes": list(EXPECTED_CANDIDATE_SOURCE_CODES),
        "support_audit": {
            "status": "aggregate_support_audited_under_frozen_gates",
            "outcome_values_accessed": False,
            "selection_uses_support": False,
            "all_declared_candidates_retained": True,
            "candidates": candidates,
        },
        "official_test_inputs_loaded": False,
        "patient_rows_or_identifiers_emitted": False,
        "outcome_values_accessed": False,
        "models_scored": False,
        "predictions_or_scores_loaded": False,
    }


def load_eligible_support_receipt(
    source: str | Path | Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
) -> EligibleSupport:
    """Load a later support receipt without loading local rows or labels."""

    digest: str | None = None
    if isinstance(source, Mapping):
        receipt = _json_safe(source)
        if isinstance(receipt, Mapping):
            digest = canonical_sha256(receipt)
    else:
        path_value = Path(source)
        if not path_value.is_absolute() and project_root is not None:
            path_value = Path(project_root).resolve() / path_value
        path = path_value.resolve()
        try:
            if sha256_file(path) != FROZEN_SUPPORT_RECEIPT_SHA256:
                raise EndpointSupportDriftError("file-backed support receipt hash differs")
            receipt = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EndpointSupportDriftError("support receipt is not valid JSON") from error
        digest = sha256_file(path)
    if not isinstance(receipt, Mapping):
        raise EndpointSupportDriftError("support receipt must be one object")
    _forbidden_payload_walk(receipt, output=False, where="support receipt")
    schema = str(receipt.get("schema_version", ""))
    materializer_schema = "baras-v6-2-expanded-endpoint-atlas-support-materializer-v1"
    synthetic_schema = "baras-v6-2-expanded-endpoint-evaluation-eligible-support-receipt-v1"
    local_outcome_accessed = receipt.get(
        "outcome_values_accessed_locally",
        receipt.get("outcome_values_accessed"),
    )
    if schema == materializer_schema:
        outcome_contract_ok = (
            receipt.get("outcome_values_accessed_locally") is True
            and receipt.get("outcome_values_serialized") is False
        )
    elif schema == synthetic_schema:
        outcome_contract_ok = local_outcome_accessed is False
    else:
        outcome_contract_ok = False
    if (
        receipt.get("official_test_inputs_loaded") is not False
        or receipt.get("patient_rows_or_identifiers_emitted") is not False
        or not outcome_contract_ok
        or receipt.get("models_scored") not in (False, None)
        or receipt.get("predictions_or_scores_loaded") not in (False, None)
    ):
        raise OfficialTestRefusal("support receipt is not train-validation aggregate-only")
    eligible, support = _eligible_from_receipt(receipt)
    return EligibleSupport(
        receipt=MappingProxyType(dict(receipt)),
        receipt_sha256=digest,
        eligible_sources=eligible,
        support_by_source=support,
    )


def validate_support_against_observed(
    support: EligibleSupport,
    labels_by_source: Mapping[str, np.ndarray],
    observed_by_source: Mapping[str, np.ndarray],
    outer_fold_ids: np.ndarray,
) -> None:
    """Ensure later-loaded support is the same support used for this input."""

    folds = np.asarray(outer_fold_ids)
    for source in support.eligible_sources:
        labels = np.asarray(labels_by_source[source])
        observed = np.asarray(observed_by_source[source], dtype=bool)
        detail = support.support_by_source[source]
        cases_by_fold = [int(np.sum(observed & (labels == 1) & (folds == fold))) for fold in range(OUTER_FOLD_COUNT)]
        controls_by_fold = [int(np.sum(observed & (labels == 0) & (folds == fold))) for fold in range(OUTER_FOLD_COUNT)]
        if int(detail.get("cases", -1)) != sum(cases_by_fold) or int(detail.get("controls", -1)) != sum(controls_by_fold) or tuple(int(item) for item in detail.get("cases_by_outer_fold", ())) != tuple(cases_by_fold) or tuple(int(item) for item in detail.get("controls_by_outer_fold", ())) != tuple(controls_by_fold):
            raise EndpointSupportDriftError("eligible support receipt differs from observed labels")


def _centered_multiplier_inference(
    contrast_vectors: Mapping[str, np.ndarray],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> InferenceResult:
    """Compute patient-level centered multiplier max-stat intervals in memory."""

    if bootstrap_samples < 100 or not 0.5 < confidence_level < 1.0:
        raise ExpandedEndpointEvaluationError("multiplier inference settings are malformed")
    names = tuple(contrast_vectors)
    if not names:
        return InferenceResult({}, {}, {}, {}, 0, bootstrap_samples, bootstrap_seed, confidence_level)
    vectors: dict[str, np.ndarray] = {}
    point: dict[str, float] = {}
    se: dict[str, float] = {}
    n_max = 0
    for name, raw in contrast_vectors.items():
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 1 or not len(values):
            raise ExpandedEndpointEvaluationError("contrast vector is malformed")
        finite = values[np.isfinite(values)]
        if len(finite) < 2:
            raise ExpandedEndpointEvaluationError("contrast has too few paired patients")
        vectors[name] = values
        point[name] = float(np.mean(finite))
        centered = finite - point[name]
        se[name] = float(np.sqrt(np.sum(centered * centered) / (len(finite) * max(len(finite) - 1, 1))))
        n_max = len(values) if n_max == 0 else n_max
        if len(values) != n_max:
            raise ExpandedEndpointEvaluationError("simultaneous contrasts do not share one patient index")
    # Build one studentized patient-influence row per contrast.  Missing
    # endpoint observations contribute zero to that contrast while the
    # patient multiplier remains shared across every endpoint and group.
    # This is algebraically identical to the scalar implementation below,
    # but permits bounded matrix multiplication instead of a Python loop over
    # every draw/contrast/patient.
    influence = np.zeros((len(names), n_max), dtype=np.float64)
    for row, name in enumerate(names):
        values = vectors[name]
        finite = np.isfinite(values)
        denominator = se[name] if se[name] > 0.0 else 1.0
        influence[row, finite] = (values[finite] - point[name]) / (
            float(np.sum(finite)) * denominator
        )
    rng = np.random.default_rng(int(bootstrap_seed))
    max_statistics = np.empty(int(bootstrap_samples), dtype=np.float64)
    chunk_size = int(MULTIPLIER_DRAW_CHUNK_SIZE)
    for start in range(0, int(bootstrap_samples), chunk_size):
        stop = min(start + chunk_size, int(bootstrap_samples))
        global_multipliers = rng.standard_normal((stop - start, n_max))
        global_multipliers -= np.mean(global_multipliers, axis=1, keepdims=True)
        # (draws x patients) @ (patients x contrasts), then max over the
        # simultaneous family.  Only this chunk and max statistics persist.
        studentized = np.abs(global_multipliers @ influence.T)
        max_statistics[start:stop] = np.max(studentized, axis=1)
    critical = float(np.quantile(max_statistics, confidence_level))
    lower = {name: point[name] - critical * se[name] for name in names}
    upper = {name: point[name] + critical * se[name] for name in names}
    passed = {name: bool(lower[name] > 0.0) for name in names}
    return InferenceResult(
        point=MappingProxyType(point),
        lower=MappingProxyType(lower),
        upper=MappingProxyType(upper),
        simultaneous_passed=MappingProxyType(passed),
        test_count=len(names),
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_seed=int(bootstrap_seed),
        confidence_level=float(confidence_level),
    )


def _paired_contrast_vectors(
    losses: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    required = set(ROUTES)
    if set(losses) != required:
        raise ExpandedEndpointEvaluationError("all four route losses are required")
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in losses.items()}
    size = len(arrays[ROUTE_BOTH])
    if any(value.ndim != 1 or len(value) != size for value in arrays.values()):
        raise ExpandedEndpointEvaluationError("paired route losses do not share one patient index")
    valid = np.ones(size, dtype=bool)
    for value in arrays.values():
        valid &= np.isfinite(value)
    if valid.sum() < 2:
        raise ExpandedEndpointEvaluationError("paired route losses have too few patients")
    age, eye, clinical, both = (arrays[item][valid] for item in ROUTES)
    # Endpoint-level formulas are evaluated on patient-paired losses.  The
    # best-single branch is selected at the aggregate mean and then applied
    # to its paired patient losses for the centered multiplier statistic.
    eye_mean, clinical_mean = float(np.mean(eye)), float(np.mean(clinical))
    best = eye if eye_mean <= clinical_mean else clinical
    # Preserve the full patient index.  Endpoint-specific missingness must be
    # represented by NaN at the unavailable positions so the shared patient
    # multiplier remains aligned across endpoint and group contrasts.
    result = {
        contrast: np.full(size, np.nan, dtype=np.float64)
        for contrast in CONTRAST_NAMES
    }
    result["eye_added"][valid] = clinical - both
    result["clinical_added"][valid] = eye - both
    result["both_vs_best_single"][valid] = best - both
    result["eye_vs_clinical"][valid] = clinical - eye
    result["both_vs_age"][valid] = age - both
    return MappingProxyType(result)


def summarize_endpoint_routes(
    losses: Mapping[str, np.ndarray],
    *,
    auc_by_route: Mapping[str, float | None] | None = None,
    support_count: int,
) -> Mapping[str, Any]:
    """Return one aggregate endpoint score/contrast summary."""

    required = set(ROUTES)
    if set(losses) != required:
        raise ExpandedEndpointEvaluationError("endpoint summary requires all four routes")
    means: dict[str, float] = {}
    eligible_count: int | None = None
    for route in ROUTES:
        values = np.asarray(losses[route], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            raise ExpandedEndpointEvaluationError("endpoint route has no eligible loss")
        if eligible_count is None:
            eligible_count = len(finite)
        means[route] = float(np.mean(finite))
    contrasts = {
        "eye_added": means[ROUTE_CLINICAL_ONLY] - means[ROUTE_BOTH],
        "clinical_added": means[ROUTE_EYE_ONLY] - means[ROUTE_BOTH],
        "both_vs_best_single": min(means[ROUTE_EYE_ONLY], means[ROUTE_CLINICAL_ONLY]) - means[ROUTE_BOTH],
        "eye_vs_clinical": means[ROUTE_CLINICAL_ONLY] - means[ROUTE_EYE_ONLY],
        "both_vs_age": means[ROUTE_AGE_ONLY] - means[ROUTE_BOTH],
    }
    return {
        "metric": PRIMARY_METRIC,
        "route_log_loss": means,
        "paired_contrasts": contrasts,
        "support": safe_count(int(support_count)),
        "auc_descriptive": dict(auc_by_route or {}),
        "contains_patient_losses": False,
    }


def _partition_evidence(
    lower: Mapping[str, float],
    upper: Mapping[str, float],
) -> Mapping[str, Any]:
    required = set(CONTRAST_NAMES)
    if set(lower) != required or set(upper) != required:
        raise ClassificationError("partition evidence does not contain all frozen contrasts")
    complementary = lower["eye_added"] > 0.0 and lower["clinical_added"] > 0.0
    eye_dominant = lower["eye_vs_clinical"] > 0.0 and upper["both_vs_best_single"] <= 0.0
    clinical_dominant = upper["eye_vs_clinical"] < 0.0 and upper["both_vs_best_single"] <= 0.0
    if complementary:
        label = "complementary_both"
    elif eye_dominant:
        label = "retinal_dominant"
    elif clinical_dominant:
        label = "clinical_dominant"
    else:
        label = "unresolved"
    return {
        "label": label,
        "fail_closed": label == "unresolved",
        "evidence": {
            "both_beats_eye_only_simultaneously": bool(lower["clinical_added"] > 0.0),
            "both_beats_clinical_only_simultaneously": bool(lower["eye_added"] > 0.0),
            "eye_beats_clinical_simultaneously": bool(lower["eye_vs_clinical"] > 0.0),
            "clinical_beats_eye_simultaneously": bool(upper["eye_vs_clinical"] < 0.0),
            "both_beats_best_single_simultaneously": bool(lower["both_vs_best_single"] > 0.0),
            "both_increment_not_supported_for_dominance": bool(upper["both_vs_best_single"] <= 0.0),
        },
        "causal_language_authorized": False,
    }


def classify_modality_partition(
    *,
    lower_bounds: Mapping[str, float],
    upper_bounds: Mapping[str, float],
) -> Mapping[str, Any]:
    """Apply the frozen fail-closed partition labels."""

    return _partition_evidence(lower_bounds, upper_bounds)


def _aggregate_group_vectors(
    endpoint_losses: Mapping[str, Mapping[str, np.ndarray]],
    endpoint_sources: Sequence[str],
) -> Mapping[str, np.ndarray]:
    """Construct patient-level vectors for equal-endpoint aggregation.

    Each endpoint contributes its own paired mean with weight ``1/E``.  The
    finite observations are scaled by their endpoint denominator, so partial
    survey response patterns cannot silently turn the family point estimate
    into a patient-weighted endpoint estimate.
    """

    if not endpoint_sources:
        raise ExpandedEndpointEvaluationError("aggregate group has no endpoints")
    first = np.asarray(endpoint_losses[endpoint_sources[0]][ROUTE_BOTH], dtype=np.float64)
    if first.ndim != 1 or not len(first):
        raise ExpandedEndpointEvaluationError("aggregate group loss vectors are malformed")
    size = len(first)
    endpoint_count = float(len(endpoint_sources))
    endpoint_differences: dict[str, dict[str, np.ndarray]] = {}
    endpoint_means: dict[str, dict[str, float]] = {}
    for source in endpoint_sources:
        endpoint_differences[source] = dict(_paired_contrast_vectors(endpoint_losses[source]))
        endpoint_means[source] = {
            contrast: float(np.mean(vector[np.isfinite(vector)]))
            for contrast, vector in endpoint_differences[source].items()
        }
    # A group target is the equal-endpoint mean of endpoint targets.  The
    # patient pseudo-value adds each endpoint's centered influence with its own
    # observed-patient denominator.  This preserves the endpoint-specific
    # ``both_vs_best_single`` branch and remains valid when endpoints have
    # unequal missingness.
    vectors: dict[str, np.ndarray] = {}
    for contrast in CONTRAST_NAMES:
        group_point = float(
            np.mean([endpoint_means[source][contrast] for source in endpoint_sources])
        )
        pseudo = np.full(size, group_point, dtype=np.float64)
        for source in endpoint_sources:
            diff = endpoint_differences[source][contrast]
            finite = np.isfinite(diff)
            count = int(finite.sum())
            if count < 2:
                raise ExpandedEndpointEvaluationError(
                    "aggregate endpoint contrast has too few paired patients"
                )
            endpoint_point = endpoint_means[source][contrast]
            pseudo[finite] += (
                size
                / (endpoint_count * count)
                * (diff[finite] - endpoint_point)
            )
        vectors[contrast] = pseudo
    return MappingProxyType(vectors)


def infer_endpoint_and_group_contrasts(
    *,
    lane_endpoint_losses: Mapping[str, Mapping[str, np.ndarray]],
    endpoint_metadata: Mapping[str, Mapping[str, str]],
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> tuple[Mapping[str, Any], InferenceResult]:
    """Infer all endpoint and organ/claim aggregate contrasts together."""

    vectors: dict[str, np.ndarray] = {}
    for source in EXPECTED_CANDIDATE_SOURCE_CODES:
        if source not in lane_endpoint_losses:
            continue
        endpoint_vectors = _paired_contrast_vectors(lane_endpoint_losses[source])
        for contrast, vector in endpoint_vectors.items():
            vectors[f"endpoint::{source}::{contrast}"] = vector

    groupings = (
        ("organ_family", lambda meta: meta["organ_family"]),
        ("claim_proximity_class", lambda meta: meta["claim_proximity_class"]),
    )
    grouped_names: dict[str, dict[str, list[str]]] = {name: {} for name, _ in groupings}
    for source in lane_endpoint_losses:
        meta = endpoint_metadata[source]
        for grouping, keyer in groupings:
            key = str(keyer(meta))
            grouped_names[grouping].setdefault(key, []).append(source)
    for grouping, groups in grouped_names.items():
        for group, sources in groups.items():
            group_vectors = _aggregate_group_vectors(lane_endpoint_losses, sources)
            for contrast, vector in group_vectors.items():
                vectors[f"{grouping}::{group}::{contrast}"] = vector
    inference = _centered_multiplier_inference(
        vectors,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    return grouped_names, inference


def _inference_to_public(
    inference: InferenceResult,
    *,
    prefix: str,
) -> Mapping[str, Any]:
    rows: dict[str, Any] = {}
    for name in inference.point:
        if not name.startswith(prefix):
            continue
        rows[name] = {
            "point": inference.point[name],
            "simultaneous_lower_confidence_bound": inference.lower[name],
            "simultaneous_upper_confidence_bound": inference.upper[name],
            "simultaneous_positive": inference.simultaneous_passed[name],
        }
    return rows


def build_lane_aggregate_report(
    *,
    lane: str,
    endpoint_losses: Mapping[str, Mapping[str, np.ndarray]],
    endpoint_metadata: Mapping[str, Mapping[str, str]],
    support_by_source: Mapping[str, Mapping[str, Any]],
    auc_by_source: Mapping[str, Mapping[str, float | None]],
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> Mapping[str, Any]:
    """Convert private endpoint losses into aggregate-only lane output."""

    endpoint_rows: dict[str, Any] = {}
    for source, losses in endpoint_losses.items():
        detail = support_by_source[source]
        cases = int(detail.get("cases", 0))
        controls = int(detail.get("controls", 0))
        endpoint_rows[source] = dict(
            summarize_endpoint_routes(
                losses,
                auc_by_route=auc_by_source.get(source, {}),
                support_count=cases + controls,
            )
        )
        endpoint_rows[source]["organ_family"] = endpoint_metadata[source]["organ_family"]
        endpoint_rows[source]["claim_proximity_class"] = endpoint_metadata[source]["claim_proximity_class"]
        endpoint_rows[source]["label"] = endpoint_metadata[source]["label"]
        endpoint_rows[source]["support_cases"] = safe_count(cases)
        endpoint_rows[source]["support_controls"] = safe_count(controls)

    _, inference = infer_endpoint_and_group_contrasts(
        lane_endpoint_losses=endpoint_losses,
        endpoint_metadata=endpoint_metadata,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    for source in endpoint_rows:
        lower = {contrast: inference.lower[f"endpoint::{source}::{contrast}"] for contrast in CONTRAST_NAMES}
        upper = {contrast: inference.upper[f"endpoint::{source}::{contrast}"] for contrast in CONTRAST_NAMES}
        endpoint_rows[source]["simultaneous_inference"] = {
            "contrasts": {
                contrast: {
                    "point": inference.point[f"endpoint::{source}::{contrast}"],
                    "simultaneous_lower_confidence_bound": lower[contrast],
                    "simultaneous_upper_confidence_bound": upper[contrast],
                    "simultaneous_positive": inference.simultaneous_passed[f"endpoint::{source}::{contrast}"],
                }
                for contrast in CONTRAST_NAMES
            },
            "partition": classify_modality_partition(lower_bounds=lower, upper_bounds=upper),
        }
    group_rows: dict[str, Any] = {"organ_family": {}, "claim_proximity_class": {}}
    for name in inference.point:
        pieces = name.split("::", 2)
        if len(pieces) != 3 or pieces[0] not in group_rows:
            continue
        grouping, group, contrast = pieces
        row = group_rows[grouping].setdefault(group, {"contrasts": {}})
        row["contrasts"][contrast] = {
            "point": inference.point[name],
            "simultaneous_lower_confidence_bound": inference.lower[name],
            "simultaneous_upper_confidence_bound": inference.upper[name],
            "simultaneous_positive": inference.simultaneous_passed[name],
        }
    for grouping, groups in group_rows.items():
        for group, row in groups.items():
            lower = {contrast: row["contrasts"][contrast]["simultaneous_lower_confidence_bound"] for contrast in CONTRAST_NAMES}
            upper = {contrast: row["contrasts"][contrast]["simultaneous_upper_confidence_bound"] for contrast in CONTRAST_NAMES}
            row["partition"] = classify_modality_partition(lower_bounds=lower, upper_bounds=upper)
            row["endpoint_count"] = safe_count(sum(1 for meta in endpoint_metadata.values() if (meta["organ_family"] if grouping == "organ_family" else meta["claim_proximity_class"]) == group))
    return {
        "lane": lane,
        "primary_metric": PRIMARY_METRIC,
        "optional_metric": OPTIONAL_DESCRIPTIVE_METRIC,
        "endpoints": endpoint_rows,
        "aggregate_inference": group_rows,
        "multiplicity_control": {
            "method": "patient_level_centered_multiplier_max_statistic",
            "paired": True,
            "all_eligible_endpoints_and_group_contrasts_in_one_family": True,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "draws_serialized": False,
        },
        "causal_language_authorized": False,
    }


def frozen_candidate_metadata() -> Mapping[str, Mapping[str, str]]:
    """Return public row-free metadata for all fixed survey candidates."""

    return MappingProxyType(
        {
            item.source_code: {
                "target_id": item.target_id,
                "label": item.label,
                "organ_family": item.organ_family,
                "claim_proximity_class": item.claim_proximity_class,
                "source_mapping_status": item.source_mapping_status,
            }
            for item in freeze_condition_candidates()
        }
    )


def _load_strict_json(path: Path) -> Mapping[str, Any]:
    def strict(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolBindingError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=strict)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ProtocolBindingError) as error:
        raise ProtocolBindingError("JSON binding is invalid") from error
    if not isinstance(value, Mapping):
        raise ProtocolBindingError("JSON binding must be one object")
    return value


def _validate_parent_precommit(root: Path) -> None:
    protocol_path = (root / FROZEN_ATLAS_PROTOCOL_NAME).resolve()
    if not protocol_path.is_file() or sha256_file(protocol_path) != FROZEN_ATLAS_PROTOCOL_SHA256:
        raise ProtocolBindingError("atlas protocol hash differs")
    try:
        validate_atlas_protocol(root, protocol_path, verify_bindings=True)
    except Exception as error:
        raise ProtocolBindingError("atlas protocol is not authenticated") from error
    receipt_path = (root / FROZEN_PRECOMMIT_RECEIPT_NAME).resolve()
    if not receipt_path.is_file() or sha256_file(receipt_path) != FROZEN_PRECOMMIT_RECEIPT_SHA256:
        raise ProtocolBindingError("atlas precommit receipt hash differs")
    receipt = _load_strict_json(receipt_path)
    _forbidden_payload_walk(receipt, output=False, where="precommit receipt")
    if receipt.get("schema_version") != "baras-v6-2-expanded-endpoint-atlas-run-v1" or receipt.get("status") != "aggregate_only_precommit_complete" or receipt.get("protocol_sha256") != FROZEN_ATLAS_PROTOCOL_SHA256 or receipt.get("outer_fold_assignment_sha256") != EXACT_OUTER_FOLD_HASH or receipt.get("candidate_count") != len(EXPECTED_CANDIDATE_SOURCE_CODES) or tuple(receipt.get("candidate_source_codes", ())) != EXPECTED_CANDIDATE_SOURCE_CODES or receipt.get("official_test_inputs_loaded") is not False or receipt.get("patient_rows_or_identifiers_emitted") is not False or receipt.get("outcome_values_accessed") is not False:
        raise ProtocolBindingError("atlas precommit receipt metadata differs")


def validate_protocol(
    project_root: str | Path | None = None,
    protocol_path: str | Path | None = None,
    *,
    verify_bindings: bool = True,
) -> tuple[Mapping[str, Any], str]:
    """Authenticate this evaluation protocol and its immutable parents."""

    root = Path(__file__).resolve().parent if project_root is None else Path(project_root).resolve()
    canonical = (root / PROTOCOL_NAME).resolve()
    path = canonical if protocol_path is None else Path(protocol_path).resolve()
    if path != canonical:
        raise ProtocolBindingError("evaluation protocol must be canonical")
    protocol = _load_strict_json(path)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION or protocol.get("status") != "frozen_before_any_expanded_endpoint_evaluation_score":
        raise ProtocolBindingError("evaluation protocol is not frozen")
    if protocol.get("parent_bindings") != {
        "atlas_protocol": {"file": FROZEN_ATLAS_PROTOCOL_NAME, "sha256": FROZEN_ATLAS_PROTOCOL_SHA256},
        "precommit_receipt": {"file": FROZEN_PRECOMMIT_RECEIPT_NAME, "sha256": FROZEN_PRECOMMIT_RECEIPT_SHA256},
        "outer_fold_assignment_sha256": EXACT_OUTER_FOLD_HASH,
        "outer_fold_count": OUTER_FOLD_COUNT,
    }:
        raise ProtocolBindingError("evaluation parent bindings differ")
    if tuple(protocol.get("candidate_contract", {}).get("candidate_source_codes", ())) != EXPECTED_CANDIDATE_SOURCE_CODES or protocol.get("candidate_contract", {}).get("candidate_count") != len(EXPECTED_CANDIDATE_SOURCE_CODES) or tuple(protocol.get("candidate_contract", {}).get("free_text_sources_excluded", ())) != FREE_TEXT_SOURCE_CODES:
        raise ProtocolBindingError("evaluation candidate contract differs")
    if protocol.get("canonical_v6_2_execution") != dict(CANONICAL_V6_2_EXECUTION_BINDINGS):
        raise ProtocolBindingError("canonical V6.2 execution bindings differ")
    if tuple(protocol.get("condition_history_flag_names", ())) != tuple(V62_CLINICAL_FLAG_NAMES):
        raise ProtocolBindingError("evaluation condition-history flag set differs")
    if protocol.get("source_to_direct_v6_2_flag") != dict(SOURCE_TO_DIRECT_V62_FLAG):
        raise ProtocolBindingError("evaluation direct-feature mapping differs")
    if protocol.get("labels") != {
        "accepted_encoding": "explicit_numeric_0_1_on_observed_rows",
        "missing_response_policy": "missing_or_excluded_from_denominator",
        "positive_unlabelled_survey_status_allowed": False,
        "official_test_labels_allowed": False,
        "observed_numeric_only": True,
    }:
        raise ProtocolBindingError("evaluation label contract differs")
    required_lanes = {
        LANE_FULL_CONTEXT: {
            "exact_target_source_removed": True,
            "direct_history_flag_removed": True,
            "all_condition_history_flags_removed": False,
            "other_condition_history_flags_retained": True,
        },
        LANE_DE_NOVO: {
            "exact_target_source_removed": True,
            "all_condition_history_flags_removed": True,
            "condition_history_flag_count": len(V62_CLINICAL_FLAG_NAMES),
            "continuous_labs_and_vitals_retained": True,
        },
    }
    if protocol.get("lanes") != required_lanes:
        raise ProtocolBindingError("evaluation lane contract differs")
    if tuple(protocol.get("routes", {}).get("per_lane", ())) != ROUTES:
        raise ProtocolBindingError("evaluation route contract differs")
    if protocol.get("contrasts", {}).get("formulas") != dict(CONTRAST_FORMULAS):
        raise ProtocolBindingError("evaluation contrast formulas differ")
    readout = protocol.get("readout", {})
    if (
        tuple(readout.get("inner_assignment_hashes", ()))
        != EXACT_INNER_FOLD_ASSIGNMENT_SHA256
        or tuple(float(value) for value in readout.get("penalty_grid", ()))
        != DEFAULT_PENALTY_GRID
        or int(readout.get("maximum_penalty_combinations", -1))
        != DEFAULT_MAXIMUM_PENALTY_COMBINATIONS
    ):
        raise ProtocolBindingError("evaluation inner fold assignment hashes differ")
    representation = protocol.get("representation")
    if (
        not isinstance(representation, Mapping)
        or representation.get("same_fold_model_transforms_outer_train_and_outer_test")
        is not True
        or representation.get("target_channels_erased_before_endpoint_transform")
        is not True
        or representation.get("full_context_representation_training_may_use_condition_history")
        is not True
        or representation.get("full_context_claim_limit")
        != "Outer-train representation fitting may observe condition-history inputs corresponding to mapped survey targets; full_context is secondary/current-context evidence and is not target-history-free screening."
        or representation.get("full_context_direct_target_flag_absent_from_outer_train_and_test_transforms")
        is not True
        or representation.get("de_novo_all_11_condition_history_flags_erased_before_representation_fit")
        is not True
        or representation.get("de_novo_labs_and_vitals_retained_except_frozen_prospective_policy_fields")
        is not True
        or representation.get("cross_fold_latent_coordinates_pooled") is not False
        or representation.get("outer_test_used_for_representation_fit") is not False
        or representation.get("outer_test_used_for_readout_selection") is not False
        or int(representation.get("representation_fits_per_outer_fold", -1)) != 2
        or int(representation.get("representation_fits_total", -1)) != 10
    ):
        raise ProtocolBindingError("evaluation representation contract differs")
    inference = protocol.get("inference")
    if not isinstance(inference, Mapping) or inference.get("method") != "patient_level_centered_multiplier_max_statistic" or inference.get("draws_serialized") is not False or inference.get("bootstrap_unit") != "patient" or int(inference.get("draw_chunk_size", -1)) != MULTIPLIER_DRAW_CHUNK_SIZE:
        raise ProtocolBindingError("evaluation inference contract differs")
    support = protocol.get("support_receipt")
    if not isinstance(support, Mapping) or support.get("later_loadable") is not True or support.get("selection_uses_model_performance") is not False:
        raise ProtocolBindingError("evaluation support receipt contract differs")
    if support.get("file") != FROZEN_SUPPORT_RECEIPT_NAME or support.get("sha256") != FROZEN_SUPPORT_RECEIPT_SHA256 or support.get("expected_eligible_endpoint_count") != len(EXPECTED_ELIGIBLE_SOURCE_CODES) or support.get("expected_withheld_endpoint_count") != len(EXPECTED_WITHHELD_SOURCE_CODES) or tuple(support.get("expected_eligible_source_codes", ())) != EXPECTED_ELIGIBLE_SOURCE_CODES or support.get("expected_eligible_by_claim_proximity_class") != dict(EXPECTED_ELIGIBLE_BY_CLAIM_PROXIMITY_CLASS):
        raise ProtocolBindingError("evaluation support receipt binding differs")
    privacy = protocol.get("privacy")
    if not isinstance(privacy, Mapping) or privacy.get("aggregate_only_output") is not True or privacy.get("official_test_inputs_or_targets_allowed") is not False or privacy.get("small_cell_threshold") != SMALL_CELL_THRESHOLD or privacy.get("execution_status") != "precommit_only_no_real_data_run_launched":
        raise ProtocolBindingError("evaluation privacy contract differs")
    if verify_bindings:
        for label, raw in CANONICAL_V6_2_EXECUTION_BINDINGS.items():
            source = (root / raw["file"]).resolve()
            if not source.is_file() or sha256_file(source) != raw["sha256"]:
                raise ProtocolBindingError(f"canonical V6.2 execution binding differs: {label}")
        bindings = protocol.get("bindings")
        if not isinstance(bindings, Mapping) or set(bindings) != {"kernel", "runner", "synthetic_tests"}:
            raise ProtocolBindingError("evaluation code bindings differ")
        for label, raw in bindings.items():
            if not isinstance(raw, Mapping) or set(raw) != {"file", "sha256"} or not isinstance(raw["file"], str) or Path(raw["file"]).is_absolute() or ".." in Path(raw["file"]).parts or not _is_sha256(raw["sha256"]):
                raise ProtocolBindingError(f"evaluation binding is malformed: {label}")
            source = (root / raw["file"]).resolve()
            if not source.is_file() or sha256_file(source) != raw["sha256"]:
                raise ProtocolBindingError(f"evaluation binding differs: {label}")
    _validate_parent_precommit(root)
    return protocol, sha256_file(path)


def build_failure_payload(error: BaseException, *, protocol_sha256: str | None = None) -> Mapping[str, Any]:
    """Build a redacted exclusive failure marker."""

    return {
        "schema_version": RUNNER_SCHEMA_VERSION + "-failure-v1",
        "status": "failed_closed",
        "error_type": type(error).__name__,
        "details_emitted": False,
        "protocol_sha256": protocol_sha256,
        "atlas_protocol_sha256": FROZEN_ATLAS_PROTOCOL_SHA256,
        "precommit_receipt_sha256": FROZEN_PRECOMMIT_RECEIPT_SHA256,
        "outer_fold_assignment_sha256": EXACT_OUTER_FOLD_HASH,
        "official_test_inputs_loaded": False,
        "patient_rows_or_identifiers_emitted": False,
        "outcome_values_accessed_locally_may_have_occurred": True,
        "outcome_values_serialized": False,
        "model_scoring_may_have_occurred": True,
        "predictions_or_scores_loaded": False,
        "draws_serialized": False,
        "retry_requires_preserving_this_artifact": True,
    }


__all__ = [
    "CONTRAST_FORMULAS",
    "CONTRAST_NAMES",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "MULTIPLIER_DRAW_CHUNK_SIZE",
    "EVALUATION_PREFIX",
    "EXACT_OUTER_FOLD_HASH",
    "EXACT_INNER_FOLD_ASSIGNMENT_SHA256",
    "CANONICAL_V6_2_EXECUTION_BINDINGS",
    "EXPECTED_ELIGIBLE_SOURCE_CODES",
    "EXPECTED_ELIGIBLE_BY_CLAIM_PROXIMITY_CLASS",
    "EXPECTED_WITHHELD_SOURCE_CODES",
    "EligibleSupport",
    "EvaluationInput",
    "ExpandedEndpointEvaluationError",
    "FROZEN_ATLAS_PROTOCOL_SHA256",
    "FROZEN_PRECOMMIT_RECEIPT_SHA256",
    "FROZEN_SUPPORT_RECEIPT_NAME",
    "FROZEN_SUPPORT_RECEIPT_SHA256",
    "FoldBasisError",
    "InputContractError",
    "ImproperNestingError",
    "LANE_DE_NOVO",
    "LANE_FULL_CONTEXT",
    "OfficialTestRefusal",
    "PRIMARY_METRIC",
    "PROTOCOL_NAME",
    "ProtocolBindingError",
    "ROUTES",
    "ReadoutResult",
    "RepresentationCoordinates",
    "RepresentationFactory",
    "TARGET_PREFIX",
    "assert_aggregate_only_result",
    "assert_target_channels_erased",
    "build_failure_payload",
    "build_lane_aggregate_report",
    "build_synthetic_support_receipt",
    "canonical_sha256",
    "classify_modality_partition",
    "frozen_candidate_metadata",
    "infer_endpoint_and_group_contrasts",
    "load_eligible_support_receipt",
    "make_inner_fold_ids",
    "nested_logistic_readout",
    "physically_erase_target_channels",
    "route_design",
    "safe_count",
    "summarize_endpoint_routes",
    "validate_development_partitions",
    "validate_protocol",
    "validate_support_against_observed",
]
