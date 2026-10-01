"""Row-safe, outcome-free V6.2 expanded survey-condition endpoint atlas.

This module freezes the endpoint *definitions* before any endpoint values or
model scores are opened.  It is deliberately independent of the existing
patient loaders: mapping metadata is public and row-free, while aggregate
support (when supplied) is validated as an already de-identified audit.  No
function in this module accepts or emits patient rows, identifiers, dates,
fold assignments, target arrays, predictions, or model scores.

The 32 candidates are the source-aligned ``Custom_Code_Master`` fields whose
source code starts with ``mhoccur_`` or ``mhterm_``, or is exactly
``mh_a1c``.  The two mapped ``If yes, please specify:`` fields are excluded;
the historical ``mhoccur_fall`` endpoint is retained even though it is not in
the public mapping clone.

Two input lanes are intentionally separate:

``full_context``
    Retains the other condition history while erasing the exact target source
    field and any prespecified direct V6.2 feature/defining field.

``de_novo``
    Erases every one of the 11 V6.2 condition-history flags (and the exact
    target source when present), while retaining continuous laboratory and
    vital predictors.  This is a strict discovery lane, not a version of full
    context.

Procedures are only summarized as care-process exposures.  They are never
added to the condition endpoint registry or treated as a fourth V6.2 tower.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROTOCOL_NAME = "BARAS_V6_2_EXPANDED_ENDPOINT_ATLAS_PROTOCOL_V1.json"
PROTOCOL_SCHEMA_VERSION = "baras-v6-2-expanded-endpoint-atlas-protocol-v1"
SCHEMA_VERSION = "baras-v6-2-expanded-endpoint-atlas-precommit-v1"
RUNNER_SCHEMA_VERSION = "baras-v6-2-expanded-endpoint-atlas-run-v1"

V62_OUTER_FOLD_SHA256 = (
    "c10632ba7b94cbd5571b56ae6889c6c37ea8fe0c7e71f6c8ed9dda6ebb8a0f48"
)
OUTER_FOLD_COUNT = 5
SMALL_CELL_THRESHOLD = 10
MIN_CASES = 50
MIN_CONTROLS = 50
MIN_CASES_PER_OUTER_FOLD = 10
MIN_CONTROLS_PER_OUTER_FOLD = 10

# Survey labels are binary only when an aggregate producer has counted explicit
# numeric 0/1 observations.  A participant with no response is excluded from
# that observed-label denominator; incomplete whole-cohort response coverage is
# therefore not a reason to call an observed zero positive-unlabelled.
SURVEY_BINARY_LABEL_ENCODING = "explicit_numeric_0_1_on_observed_rows"
SURVEY_MISSING_RESPONSE_POLICY = "missing_or_excluded_from_denominator"
SURVEY_LABEL_POLICY = MappingProxyType(
    {
        "label_source": "source_aligned_survey_binary",
        "accepted_label_encoding": SURVEY_BINARY_LABEL_ENCODING,
        "support_counts_denominator": "observed_explicit_binary_labels_only",
        "whole_cohort_complete_coverage_required": False,
        "missing_response_policy": SURVEY_MISSING_RESPONSE_POLICY,
        "positive_unlabelled_status_allowed_for_survey_candidates": False,
        "positive_unlabelled_reserved_for": [
            "condition_occurrence_or_procedure_sources"
        ],
    }
)

MAPPING_SOURCE_PREFIXES = ("mhoccur_", "mhterm_")
MAPPING_SOURCE_EXACT = "mh_a1c"
PUBLIC_CUSTOM_CODE_MASTER_SHA256 = (
    "449f57793d5781cc352cce681d995b4723847e059621fa72bfbade7c3542387b"
)
# Durable provenance for the authenticated public mapping artifact.  The
# runner may receive a local authenticated copy explicitly; this contract does
# not require or canonize a temporary checkout path.
PUBLIC_CUSTOM_CODE_MASTER_REPO_URL = "https://github.com/AI-READI/DataElementMaps"
PUBLIC_CUSTOM_CODE_MASTER_COMMIT = "3a47c45be2af0529a1c6ac65cbe6c63e2d26b89f"
PUBLIC_CUSTOM_CODE_MASTER_REPO_PATH = "Custom_Concepts_Review/Custom_Code_Master.csv"

LANE_FULL_CONTEXT = "full_context"
LANE_DE_NOVO = "de_novo"
LANES = (LANE_FULL_CONTEXT, LANE_DE_NOVO)

# These are the exact 11 binary condition features in the frozen V6.2
# clinical input block.  The values are feature names, not outcome values.
V62_CLINICAL_FLAG_NAMES = (
    "hypertension",
    "hyperlipidemia",
    "diabetes",
    "cancer",
    "kidney",
    "myocardial_inf",
    "stroke",
    "arthritis",
    "osteoporosis",
    "heart_failure",
    "chronic_lung",
)

# A target is never allowed to enter the representation through its matching
# existing flag.  Keeping this map one-to-one makes the exclusion auditable
# and prevents an implementation from silently broadening or weakening it.
V62_CLINICAL_FLAG_DIRECT_FEATURE_EXCLUSIONS = MappingProxyType(
    {name: (name,) for name in V62_CLINICAL_FLAG_NAMES}
)

SOURCE_TO_DIRECT_V62_FLAG = MappingProxyType(
    {
        "mhoccur_hbp": "hypertension",
        "mhoccur_clsh": "hyperlipidemia",
        "mhterm_dm1": "diabetes",
        "mhterm_dm2": "diabetes",
        "mhterm_predm": "diabetes",
        "mh_a1c": "diabetes",
        "mhoccur_ca": "cancer",
        "mhoccur_rnl": "kidney",
        "mhoccur_mi": "myocardial_inf",
        "mhoccur_strk": "stroke",
        "mhoccur_ra": "arthritis",
        "mhoccur_oa": "osteoporosis",
        "mhoccur_cvdot": "heart_failure",
        "mhoccur_plm": "chronic_lung",
    }
)

# These source-aligned metadata classes constrain interpretation; they are not
# model lanes and never alter feature retention.  The first class is reported
# separately because an endpoint such as ``mh_a1c`` can be measurement-adjacent
# by construction.  Diabetes and pre-diabetes history are kept in a distinct
# metabolic-history/proximal class rather than being presented as de-novo
# disease discovery.  Unlisted retained candidates are non-proximal screening
# proxies.
CLAIM_PROXIMITY_MEASUREMENT_POLICY = "measurement_proximal_or_policy"
CLAIM_PROXIMITY_METABOLIC_HISTORY = "metabolic_history_or_proximal"
CLAIM_PROXIMITY_NON_PROXIMAL = "non_proximal_disease_screening"
CLAIM_PROXIMITY_CLASSES = (
    CLAIM_PROXIMITY_MEASUREMENT_POLICY,
    CLAIM_PROXIMITY_METABOLIC_HISTORY,
    CLAIM_PROXIMITY_NON_PROXIMAL,
)
SOURCE_TO_CLAIM_PROXIMITY = MappingProxyType(
    {
        "mh_a1c": CLAIM_PROXIMITY_MEASUREMENT_POLICY,
        "mhoccur_hbp": CLAIM_PROXIMITY_MEASUREMENT_POLICY,
        "mhoccur_clsh": CLAIM_PROXIMITY_MEASUREMENT_POLICY,
        "mhoccur_obs": CLAIM_PROXIMITY_MEASUREMENT_POLICY,
        "mhterm_dm1": CLAIM_PROXIMITY_METABOLIC_HISTORY,
        "mhterm_dm2": CLAIM_PROXIMITY_METABOLIC_HISTORY,
        "mhterm_predm": CLAIM_PROXIMITY_METABOLIC_HISTORY,
    }
)

# No continuous lab/vital is treated as an exact direct label for these
# source-aligned survey endpoints.  In particular, HbA1c, LDL, blood pressure,
# and kidney measurements remain available as predictors in both lanes.  The
# only exact V6.2 direct channels are the explicitly named binary flags above.
CANDIDATE_DEFINING_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType({})

# The source descriptions are copied from the public row-free mapping clone.
# They are labels for survey fields, not labels inferred from patient values.
_PUBLIC_CANDIDATE_ROWS: tuple[tuple[str, str, str | None, str], ...] = (
    ("mhterm_dm1", "Type I Diabetes", "201254", "metabolic"),
    ("mhoccur_mi", "Heart attack", "4329847", "cardiovascular"),
    (
        "mhoccur_cvdot",
        "Other heart issues (Examples: pacemaker, heart valve disease, open heart surgery)",
        "2005200627",
        "cardiovascular",
    ),
    ("mhoccur_strk", "Stroke", "2005200628", "neurologic"),
    ("mhoccur_clsh", "High blood cholesterol", "4159131", "metabolic"),
    ("mhoccur_hbp", "High blood pressure", "316866", "cardiovascular"),
    (
        "mhoccur_ua",
        "Urinary problems (Examples: urinary tract infections, incontinence, prostate problems)",
        "81902",
        "genitourinary",
    ),
    ("mhoccur_ear", "Hearing impairment", "439378", "auditory"),
    (
        "mhoccur_pdr",
        "Diabetic retinopathy (in one or both eyes)",
        "4174977",
        "ophthalmic",
    ),
    (
        "mh_a1c",
        "Elevated A1C levels (elevated blood sugars)",
        "2005200547",
        "metabolic",
    ),
    ("mhterm_dm2", "Type II Diabetes", "201826", "metabolic"),
    ("mhterm_predm", "Pre-diabetes", "37018196", "metabolic"),
    (
        "mhoccur_circ",
        "Circulation problems (Examples: arteriosclerosis, atherosclerosis, blood clots in lungs or leg veins)",
        "2005200015",
        "cardiovascular",
    ),
    ("mhoccur_lbp", "Low blood pressure", "317002", "cardiovascular"),
    ("mhoccur_pd", "Parkinson's disease", "381270", "neurologic"),
    (
        "mhoccur_ad",
        "Dementia (Examples: Alzheimer's Disease, vascular dementia, etc)",
        "4182210",
        "neurologic",
    ),
    (
        "mhoccur_cogn",
        'Mild cognitive impairment (known as "MCI"; mild but noticeable cognitive changes, may slow or interfere with daily activities but does not stop them)',
        "439795",
        "neurologic",
    ),
    ("mhoccur_ms", "Multiple sclerosis", "374919", "neurologic"),
    ("mhoccur_cns", "Other neurological conditions", "46271045", "neurologic"),
    ("mhoccur_ra", "Arthritis (joint pain)", "4291025", "musculoskeletal"),
    ("mhoccur_oa", "Osteoporosis", "80502", "musculoskeletal"),
    ("mhoccur_ca", "Cancer (any type)", "4194405", "oncology"),
    (
        "mhoccur_plm",
        "Chronic pulmonary (lung) problems (Examples: emphysema, asthma, tuberculosis, asbestosis)",
        "4186898",
        "pulmonary",
    ),
    (
        "mhoccur_gi",
        "Digestive problems (Examples: stomach ulcer, gastrointestinal problems, hiatal hernia)",
        "4201745",
        "gastrointestinal",
    ),
    ("mhoccur_rnl", "Kidney problems", "2005200017", "genitourinary"),
    ("mhoccur_obs", "Obesity", "433736", "metabolic"),
    ("mhoccur_glc", "Glaucoma (in one or both eyes)", "437541", "ophthalmic"),
    (
        "mhoccur_amd",
        "Age-related macular degeneration (AMD) (in one or both eyes)",
        "374028",
        "ophthalmic",
    ),
    ("mhoccur_crt", "Cataracts (in one or both eyes)", "4317977", "ophthalmic"),
    (
        "mhoccur_rvo",
        'Retinal vascular occlusion ("stroke in the eye or eye vessels" - in one or both eyes)',
        "440392",
        "ophthalmic",
    ),
    ("mhoccur_ded", "Dry eye (in one or both eyes)", "4036620", "ophthalmic"),
    ("mhoccur_fall", "Falls in prior 12 months", None, "geriatric_safety"),
)

EXPECTED_CANDIDATE_SOURCE_CODES = tuple(item[0] for item in _PUBLIC_CANDIDATE_ROWS)
FREE_TEXT_SOURCE_CODES = ("mhoccur_cnsot", "mhoccur_cnrot")


def _claim_reporting_metadata() -> dict[str, Any]:
    """Return the frozen, row-free claim-proximity reporting groups."""

    groups = {
        class_name: [
            source
            for source in EXPECTED_CANDIDATE_SOURCE_CODES
            if SOURCE_TO_CLAIM_PROXIMITY.get(source, CLAIM_PROXIMITY_NON_PROXIMAL)
            == class_name
        ]
        for class_name in CLAIM_PROXIMITY_CLASSES
    }
    return {
        "field": "claim_proximity_class",
        "classes": list(CLAIM_PROXIMITY_CLASSES),
        "measurement_proximal_or_policy_sources": groups[CLAIM_PROXIMITY_MEASUREMENT_POLICY],
        "metabolic_history_or_proximal_sources": groups[CLAIM_PROXIMITY_METABOLIC_HISTORY],
        "non_proximal_disease_screening_sources": groups[CLAIM_PROXIMITY_NON_PROXIMAL],
        "reported_separately_from_non_proximal_disease_screening": True,
        "interpretation_boundary": (
            "measurement-proximal or metabolic-history sources are not presented "
            "as novel disease discovery"
        ),
    }

_SAFE_CODE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_SAFE_LABEL_MAX = 240
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PAYLOAD_KEYS = frozenset(
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
        "model_performance",
        "losses",
        "embeddings",
        "coordinates",
        "fold_assignments",
    }
)


class ExpandedEndpointAtlasError(ValueError):
    """Base error for malformed or scientifically unsafe atlas inputs."""


class DirectLabelCircularityError(ExpandedEndpointAtlasError):
    """Raised when a target value or direct label channel is supplied."""


class OfficialTestRefusal(ExpandedEndpointAtlasError):
    """Raised before any aggregate support is consumed if official test data appears."""


class ProtocolError(ExpandedEndpointAtlasError):
    """Raised when the immutable endpoint-atlas protocol is not authenticated."""


class MappingIntegrityError(ExpandedEndpointAtlasError):
    """Raised when the public mapping clone or candidate declarations drift."""


class AggregateOnlyError(ExpandedEndpointAtlasError):
    """Raised when a purported aggregate input contains row-level material."""


class FoldBindingError(ExpandedEndpointAtlasError):
    """Raised when the frozen V6.2 outer-fold binding is changed or omitted."""


@dataclass(frozen=True)
class ConditionCandidate:
    """A public source-aligned binary survey-condition candidate."""

    source_code: str
    label: str
    organ_family: str
    mapped_concept_id: str | None = None
    source_mapping_status: str = "public_custom_code_master"

    def __post_init__(self) -> None:
        if not _SAFE_CODE.fullmatch(self.source_code):
            raise MappingIntegrityError("candidate source code is malformed")
        if not self.label or len(self.label) > _SAFE_LABEL_MAX:
            raise MappingIntegrityError("candidate source label is missing or too long")
        if not self.organ_family or not _SAFE_CODE.fullmatch(self.organ_family):
            raise MappingIntegrityError("candidate organ family is malformed")
        if self.mapped_concept_id is not None and not str(self.mapped_concept_id).strip():
            raise MappingIntegrityError("candidate mapped concept id is empty")

    @property
    def target_id(self) -> str:
        return f"survey_condition_{self.source_code}"

    @property
    def direct_feature_exclusions(self) -> tuple[str, ...]:
        flag = SOURCE_TO_DIRECT_V62_FLAG.get(self.source_code)
        fields = list(CANDIDATE_DEFINING_FIELDS.get(self.source_code, ()))
        if flag is not None:
            fields.insert(0, flag)
        return tuple(dict.fromkeys(fields))

    @property
    def claim_proximity_class(self) -> str:
        return SOURCE_TO_CLAIM_PROXIMITY.get(
            self.source_code,
            CLAIM_PROXIMITY_NON_PROXIMAL,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "source_code": self.source_code,
            "label": self.label,
            "organ_family": self.organ_family,
            "mapped_concept_id": self.mapped_concept_id,
            "source_mapping_status": self.source_mapping_status,
            "claim_proximity_class": self.claim_proximity_class,
            "measurement_proximal_or_policy": (
                self.claim_proximity_class == CLAIM_PROXIMITY_MEASUREMENT_POLICY
            ),
            "task": "binary",
            "outcome_type": "prevalent",
            "time_origin": "patient_index_date",
            "source_quality_label": "source_aligned_self_reported_survey_condition_proxy",
            "negative_label_rule": (
                "explicit_numeric_0_1_on_observed_survey_rows; missing responses "
                "are excluded_or_missing rather than positive-unlabelled"
            ),
            "defining_fields": list(CANDIDATE_DEFINING_FIELDS.get(self.source_code, ())),
            "direct_feature_exclusions": list(self.direct_feature_exclusions),
            "lanes": {
                LANE_FULL_CONTEXT: {
                    "exact_target_source_removed": True,
                    "condition_history_flags_removed": list(
                        SOURCE_TO_DIRECT_V62_FLAG.get(self.source_code, "")
                        and (SOURCE_TO_DIRECT_V62_FLAG[self.source_code],)
                        or ()
                    ),
                    "all_condition_history_flags_removed": False,
                },
                LANE_DE_NOVO: {
                    "exact_target_source_removed": True,
                    "condition_history_flags_removed": list(V62_CLINICAL_FLAG_NAMES),
                    "all_condition_history_flags_removed": True,
                },
            },
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def safe_count(value: int) -> int | str:
    count = int(value)
    if count < 0:
        raise AggregateOnlyError("aggregate counts cannot be negative")
    return count if count == 0 or count >= SMALL_CELL_THRESHOLD else f"<{SMALL_CELL_THRESHOLD}"


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: str | Path) -> tuple[Any, str]:
    source = Path(path)
    try:
        value = json.loads(source.read_bytes(), object_pairs_hook=_strict_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ProtocolError) as error:
        raise ProtocolError(f"invalid JSON artifact {source.name}") from error
    return value, sha256_file(source)


def _reject_forbidden_payload(value: Any, *, where: str = "payload") -> None:
    """Reject row/label material recursively without echoing offending values."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            token = str(key).strip().lower()
            if token in _FORBIDDEN_PAYLOAD_KEYS:
                raise AggregateOnlyError(f"{where} contains forbidden row-level key")
            if token in {"official_test", "official_test_included", "is_official_test"}:
                if child not in (False, None, 0, "", "false", "False", "0"):
                    raise OfficialTestRefusal("official test input is refused")
            if token in {"split", "recommended_split"}:
                if str(child).strip().lower() in {"test", "official_test", "official-test", "holdout"}:
                    raise OfficialTestRefusal("official test split is refused")
            _reject_forbidden_payload(child, where=where)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_forbidden_payload(child, where=where)


def assert_no_outcome_values(
    *,
    outcome_values: Any = None,
    target_values: Any = None,
    model_scores: Any = None,
    model_performance: Any = None,
) -> None:
    """Make outcome-value and score dependence impossible at the API boundary."""

    if any(item is not None for item in (outcome_values, target_values, model_scores, model_performance)):
        raise DirectLabelCircularityError(
            "expanded endpoint precommit cannot receive outcome values or model scores"
        )


def _fallback_candidates() -> tuple[ConditionCandidate, ...]:
    return tuple(
        ConditionCandidate(
            source_code=source,
            label=label,
            organ_family=family,
            mapped_concept_id=mapped,
            source_mapping_status=(
                "retained_existing_v6_2_source"
                if source == "mhoccur_fall"
                else "public_custom_code_master"
            ),
        )
        for source, label, mapped, family in _PUBLIC_CANDIDATE_ROWS
    )


def _mapping_row_is_candidate(row: Mapping[str, Any]) -> bool:
    source = str(row.get("SRC_CODE", "")).strip()
    return source.startswith(MAPPING_SOURCE_PREFIXES) or source == MAPPING_SOURCE_EXACT


def _is_free_text_row(row: Mapping[str, Any]) -> bool:
    source = str(row.get("SRC_CODE", "")).strip()
    description = str(row.get("SRC_CD_DESCRIPTION", "")).strip().lower()
    return source in FREE_TEXT_SOURCE_CODES or ("if yes" in description and "specif" in description)


def freeze_condition_candidates(
    mapping_rows: Iterable[Mapping[str, Any]] | None = None,
    *,
    include_legacy_fall: bool = True,
) -> tuple[ConditionCandidate, ...]:
    """Freeze source candidates from public mapping metadata only.

    Mapping rows are accepted only as metadata.  A row-level patient key or a
    value-bearing outcome key is rejected before candidate construction.
    Duplicate answer rows are collapsed by exact source code; the first
    field-level description is authoritative and labels are never inferred
    from observed data.
    """

    if mapping_rows is None:
        candidates = list(_fallback_candidates())
        if not include_legacy_fall:
            candidates = [item for item in candidates if item.source_code != "mhoccur_fall"]
        return tuple(candidates)

    first: dict[str, Mapping[str, Any]] = {}
    for row in mapping_rows:
        if not isinstance(row, Mapping):
            raise MappingIntegrityError("Custom_Code_Master rows must be objects")
        _reject_forbidden_payload(row, where="mapping")
        if not _mapping_row_is_candidate(row) or _is_free_text_row(row):
            continue
        source = str(row.get("SRC_CODE", "")).strip()
        if source not in first:
            first[source] = row

    fallback_by_source = {item.source_code: item for item in _fallback_candidates()}
    result: list[ConditionCandidate] = []
    # Preserve the public mapping order.  This is source alignment, not a
    # data-dependent ranking or selection.
    for source in EXPECTED_CANDIDATE_SOURCE_CODES:
        if source == "mhoccur_fall":
            continue
        row = first.get(source)
        if row is None:
            raise MappingIntegrityError(f"public mapping candidate is missing: {source}")
        reference = fallback_by_source[source]
        label = str(row.get("SRC_CD_DESCRIPTION", "")).strip()
        if not label or len(label) > _SAFE_LABEL_MAX:
            raise MappingIntegrityError(f"public mapping candidate label is invalid: {source}")
        mapped = str(row.get("MODIFIER", "")).strip() or str(row.get("TARGET_CONCEPT_ID", "")).strip() or None
        result.append(
            ConditionCandidate(
                source_code=source,
                label=label,
                organ_family=reference.organ_family,
                mapped_concept_id=mapped,
            )
        )
    if include_legacy_fall:
        result.append(fallback_by_source["mhoccur_fall"])
    expected = set(EXPECTED_CANDIDATE_SOURCE_CODES) if include_legacy_fall else set(EXPECTED_CANDIDATE_SOURCE_CODES) - {"mhoccur_fall"}
    if {item.source_code for item in result} != expected or len(result) != len(expected):
        raise MappingIntegrityError("candidate source set differs from the frozen atlas")
    return tuple(result)


def load_custom_code_master(
    path: str | Path,
    *,
    expected_sha256: str = PUBLIC_CUSTOM_CODE_MASTER_SHA256,
    allow_synthetic_mapping: bool = False,
) -> tuple[ConditionCandidate, ...]:
    """Read the public CSV mapping without retaining or emitting its rows."""

    source = Path(path).resolve()
    observed = sha256_file(source)
    if not _SHA256.fullmatch(str(expected_sha256)):
        raise MappingIntegrityError("mapping hash must be lowercase SHA-256")
    if not allow_synthetic_mapping and observed != expected_sha256:
        raise MappingIntegrityError("Custom_Code_Master exact bytes differ")
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise MappingIntegrityError("Custom_Code_Master has no header")
            required = {"SRC_CODE", "SRC_CD_DESCRIPTION", "TARGET_CONCEPT_ID", "MODIFIER"}
            if not required.issubset(set(reader.fieldnames)):
                raise MappingIntegrityError("Custom_Code_Master schema is incomplete")
            candidates = freeze_condition_candidates(reader)
    except OSError as error:
        raise MappingIntegrityError("Custom_Code_Master cannot be read") from error
    return candidates


def candidate_source_codes(candidates: Sequence[ConditionCandidate] | None = None) -> tuple[str, ...]:
    return tuple(item.source_code for item in (freeze_condition_candidates() if candidates is None else candidates))


def _validate_candidate_sequence(candidates: Sequence[ConditionCandidate]) -> tuple[ConditionCandidate, ...]:
    normalized = tuple(candidates)
    if not normalized or len({item.source_code for item in normalized}) != len(normalized):
        raise MappingIntegrityError("candidate sequence is empty or duplicated")
    expected = set(EXPECTED_CANDIDATE_SOURCE_CODES)
    if {item.source_code for item in normalized} != expected:
        raise MappingIntegrityError("candidate sequence does not contain all frozen candidates")
    if any(item.source_code in FREE_TEXT_SOURCE_CODES for item in normalized):
        raise MappingIntegrityError("free-text candidate was not excluded")
    return normalized


def target_specific_exclusions(source_code: str) -> Mapping[str, tuple[str, ...] | str]:
    """Return exact target-safe exclusions for one source candidate."""

    source = str(source_code).strip()
    candidates = {item.source_code: item for item in freeze_condition_candidates()}
    if source not in candidates:
        raise MappingIntegrityError("unknown endpoint source code")
    candidate = candidates[source]
    direct = candidate.direct_feature_exclusions
    return MappingProxyType(
        {
            "target_source_code": source,
            "direct_feature_exclusions": direct,
            "defining_fields": tuple(CANDIDATE_DEFINING_FIELDS.get(source, ())),
            "full_context_excluded_features": tuple(dict.fromkeys((source,) + direct)),
            "de_novo_excluded_features": tuple(
                dict.fromkeys(
                    (source,) + tuple(V62_CLINICAL_FLAG_NAMES)
                )
            ),
        }
    )


def lane_exclusion_plan(source_code: str) -> Mapping[str, Any]:
    exclusions = target_specific_exclusions(source_code)
    return MappingProxyType(
        {
            "target_source_code": exclusions["target_source_code"],
            "lanes": {
                LANE_FULL_CONTEXT: {
                    "excluded_features": exclusions["full_context_excluded_features"],
                    "condition_history_flags_removed": tuple(
                        flag for flag in V62_CLINICAL_FLAG_NAMES if flag in exclusions["direct_feature_exclusions"]
                    ),
                    "all_condition_history_flags_removed": False,
                },
                LANE_DE_NOVO: {
                    "excluded_features": exclusions["de_novo_excluded_features"],
                    "condition_history_flags_removed": tuple(V62_CLINICAL_FLAG_NAMES),
                    "all_condition_history_flags_removed": True,
                },
            },
        }
    )


def _validate_lane(lane: str) -> str:
    value = str(lane).strip().lower()
    if value not in LANES:
        raise DirectLabelCircularityError("lane must be full_context or de_novo")
    return value


def exclusion_mask(
    feature_names: Sequence[str],
    *,
    target_source_code: str,
    lane: str,
) -> tuple[bool, ...]:
    """Return a physically applicable keep-mask for named feature columns."""

    names = tuple(str(name) for name in feature_names)
    if not names or len(names) != len(set(names)):
        raise DirectLabelCircularityError("feature names must be nonempty and unique")
    profile = target_specific_exclusions(target_source_code)
    lane_name = _validate_lane(lane)
    excluded = set(
        profile["full_context_excluded_features"]
        if lane_name == LANE_FULL_CONTEXT
        else profile["de_novo_excluded_features"]
    )
    return tuple(name not in excluded for name in names)


def _zero_like(value: Any) -> Any:
    try:
        import numpy as np

        array = np.asarray(value)
        return np.zeros_like(array)
    except (TypeError, ValueError, ImportError):
        if isinstance(value, (list, tuple)):
            zero = [_zero_like(item) for item in value]
            return type(value)(zero)
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float, complex)):
            return type(value)(0)
        return 0


def apply_exclusion_profile(
    features: Mapping[str, Any],
    *,
    target_source_code: str,
    lane: str,
) -> Mapping[str, Any]:
    """Physically erase excluded named feature channels before preprocessing."""

    if not isinstance(features, Mapping):
        raise DirectLabelCircularityError("features must be a mapping")
    _reject_forbidden_payload(features, where="features")
    names = tuple(str(name) for name in features)
    keep = exclusion_mask(names, target_source_code=target_source_code, lane=lane)
    result = dict(features)
    for name, retain in zip(names, keep):
        if not retain:
            result[name] = _zero_like(features[name])
    return MappingProxyType(result)


def apply_exclusion_to_matrix(
    values: Any,
    feature_names: Sequence[str],
    *,
    target_source_code: str,
    lane: str,
    observed_mask: Any | None = None,
) -> tuple[Any, Any | None]:
    """Erase matrix columns and, when supplied, their observation masks."""

    _reject_forbidden_payload({"feature_names": tuple(feature_names)}, where="features")
    keep = exclusion_mask(feature_names, target_source_code=target_source_code, lane=lane)
    try:
        import numpy as np

        array = np.asarray(values).copy()
        if array.ndim != 2 or array.shape[1] != len(tuple(feature_names)):
            raise DirectLabelCircularityError("feature matrix does not align to feature names")
        array[:, np.asarray(keep, dtype=bool) == 0] = 0
        safe_mask = None
        if observed_mask is not None:
            safe_mask = np.asarray(observed_mask).copy()
            if safe_mask.shape != array.shape or safe_mask.dtype != np.bool_:
                raise DirectLabelCircularityError("observed mask does not align to feature matrix")
            safe_mask[:, np.asarray(keep, dtype=bool) == 0] = False
        return array, safe_mask
    except ImportError as error:  # pragma: no cover - numpy is a project dependency.
        raise DirectLabelCircularityError("matrix exclusion requires numpy") from error


def assert_exclusion_applied(
    values: Any,
    feature_names: Sequence[str],
    *,
    target_source_code: str,
    lane: str,
    observed_mask: Any | None = None,
) -> None:
    """Fail closed if a forbidden channel survives physical erasure."""

    keep = exclusion_mask(feature_names, target_source_code=target_source_code, lane=lane)
    try:
        import numpy as np

        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != len(tuple(feature_names)):
            raise DirectLabelCircularityError("feature matrix does not align to feature names")
        forbidden = np.asarray(keep, dtype=bool) == 0
        if np.any(array[:, forbidden] != 0):
            raise DirectLabelCircularityError("forbidden direct feature survived erasure")
        if observed_mask is not None:
            mask = np.asarray(observed_mask)
            if mask.shape != array.shape or mask.dtype != np.bool_:
                raise DirectLabelCircularityError("observed mask does not align")
            if np.any(mask[:, forbidden]):
                raise DirectLabelCircularityError("forbidden direct feature mask survived erasure")
    except ImportError as error:  # pragma: no cover
        raise DirectLabelCircularityError("matrix validation requires numpy") from error


def _support_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise AggregateOnlyError(f"{label} must be an integer count")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise AggregateOnlyError(f"{label} must be an integer count") from error
    if str(value).strip() != str(number):
        raise AggregateOnlyError(f"{label} must be an unsuppressed integer input count")
    if number < 0:
        raise AggregateOnlyError(f"{label} cannot be negative")
    return number


def _fold_counts(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise AggregateOnlyError(f"{label} must contain five outer-fold counts")
    counts = tuple(_support_int(item, label=label) for item in value)
    if len(counts) != OUTER_FOLD_COUNT:
        raise AggregateOnlyError(f"{label} must contain five outer-fold counts")
    return counts


def _support_record(raw: Any, *, source_code: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise AggregateOnlyError("support records must be objects")
    _reject_forbidden_payload(raw, where="aggregate support")
    cases = _support_int(raw.get("cases", 0), label=f"{source_code}.cases")
    controls = _support_int(raw.get("controls", 0), label=f"{source_code}.controls")
    cases_by_fold = _fold_counts(raw.get("cases_by_outer_fold", [0] * OUTER_FOLD_COUNT), label=f"{source_code}.cases_by_outer_fold")
    controls_by_fold = _fold_counts(raw.get("controls_by_outer_fold", [0] * OUTER_FOLD_COUNT), label=f"{source_code}.controls_by_outer_fold")
    if sum(cases_by_fold) != cases or sum(controls_by_fold) != controls:
        raise AggregateOnlyError(f"{source_code} fold support does not sum to total support")
    negative_status = str(raw.get("negative_label_status", "missing_or_excluded")).strip().lower()
    if "positive_unlabelled" in negative_status or "positive-unlabelled" in negative_status:
        raise AggregateOnlyError(
            f"{source_code} positive-unlabelled status is reserved for condition-occurrence or procedure sources"
        )
    if negative_status not in {"validated_binary", "missing_or_excluded", "unknown", "withheld"}:
        raise AggregateOnlyError(f"{source_code} negative label status is invalid")
    coverage = str(raw.get("coverage_status", "unavailable")).strip().lower()
    if coverage not in {"complete", "partial", "unavailable"}:
        raise AggregateOnlyError(f"{source_code} coverage status is invalid")
    label_encoding = str(raw.get("label_encoding", "")).strip().lower()
    if negative_status == "validated_binary":
        if coverage == "unavailable":
            raise AggregateOnlyError(f"{source_code} validated labels require observed coverage metadata")
        if label_encoding != SURVEY_BINARY_LABEL_ENCODING:
            raise AggregateOnlyError(
                f"{source_code} validated labels require explicit numeric 0/1 observation evidence"
            )
    return {
        "cases": cases,
        "controls": controls,
        "cases_by_outer_fold": cases_by_fold,
        "controls_by_outer_fold": controls_by_fold,
        "coverage_status": coverage,
        "negative_label_status": negative_status,
        "label_encoding": label_encoding or None,
        "missing_response_policy": SURVEY_MISSING_RESPONSE_POLICY,
    }


def summarize_aggregate_support(
    aggregate_support: Mapping[str, Any] | None,
    *,
    candidates: Sequence[ConditionCandidate] | None = None,
) -> Mapping[str, Any]:
    """Audit aggregate support under frozen gates without selecting endpoints."""

    frozen = _validate_candidate_sequence(
        tuple(freeze_condition_candidates() if candidates is None else candidates)
    )
    source_codes = tuple(item.source_code for item in frozen)
    if aggregate_support is None:
        return MappingProxyType(
            {
                "status": "aggregate_support_not_supplied",
                "outcome_values_accessed": False,
                "selection_uses_support": False,
                "all_declared_candidates_retained": True,
                "survey_label_policy": dict(SURVEY_LABEL_POLICY),
                "candidates": {
                    source: {
                        "status": "precommit_only_awaiting_aggregate_audit",
                        "support": {},
                    }
                    for source in source_codes
                },
            }
        )
    if not isinstance(aggregate_support, Mapping):
        raise AggregateOnlyError("aggregate support must be one object")
    _reject_forbidden_payload(aggregate_support, where="aggregate support")
    raw_map: Any = aggregate_support.get("conditions", aggregate_support.get("condition_support", aggregate_support))
    if not isinstance(raw_map, Mapping):
        raise AggregateOnlyError("aggregate support condition map is malformed")
    unknown = set(str(key) for key in raw_map) - set(source_codes)
    if unknown:
        raise AggregateOnlyError("aggregate support contains an unknown endpoint")
    rows: dict[str, Any] = {}
    for source in source_codes:
        if source not in raw_map:
            rows[source] = {
                "status": "support_not_supplied",
                "support": {},
            }
            continue
        support = _support_record(raw_map[source], source_code=source)
        gate = (
            support["negative_label_status"] == "validated_binary"
            and support["cases"] >= MIN_CASES
            and support["controls"] >= MIN_CONTROLS
            and min(support["cases_by_outer_fold"]) >= MIN_CASES_PER_OUTER_FOLD
            and min(support["controls_by_outer_fold"]) >= MIN_CONTROLS_PER_OUTER_FOLD
        )
        status = "eligible_under_frozen_support_gates" if gate else "withheld_under_frozen_support_gates"
        if support["negative_label_status"] != "validated_binary":
            status = "missing_or_withheld"
        rows[source] = {
            "status": status,
            "support": {
                "cases": safe_count(support["cases"]),
                "controls": safe_count(support["controls"]),
                "cases_by_outer_fold": [safe_count(item) for item in support["cases_by_outer_fold"]],
                "controls_by_outer_fold": [safe_count(item) for item in support["controls_by_outer_fold"]],
                "coverage_status": support["coverage_status"],
                "negative_label_status": support["negative_label_status"],
                "label_encoding": support["label_encoding"],
                "missing_response_policy": support["missing_response_policy"],
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
            "outcome_values_accessed": False,
            "selection_uses_support": False,
            "all_declared_candidates_retained": True,
            "survey_label_policy": dict(SURVEY_LABEL_POLICY),
            "candidates": rows,
        }
    )


def summarize_procedure_exposure_audit(
    procedure_audit: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Return a separate, aggregate-only care-process exposure audit."""

    if procedure_audit is None:
        return MappingProxyType(
            {
                "status": "procedure_audit_not_supplied",
                "target_semantics": "care_process_exposure",
                "condition_endpoint_count": 0,
                "fourth_current_tower_added": False,
            }
        )
    if not isinstance(procedure_audit, Mapping):
        raise AggregateOnlyError("procedure audit must be one aggregate object")
    _reject_forbidden_payload(procedure_audit, where="procedure audit")
    if procedure_audit.get("target_semantics") not in (None, "care_process_exposure"):
        raise AggregateOnlyError("procedures cannot be disease labels")
    result: dict[str, Any] = {
        "status": "procedure_exposure_audit",
        "target_semantics": "care_process_exposure",
        "condition_endpoint_count": 0,
        "fourth_current_tower_added": False,
    }
    exposures = procedure_audit.get("exposures", procedure_audit.get("procedures", {}))
    if not isinstance(exposures, Mapping):
        raise AggregateOnlyError("procedure exposure map is malformed")
    public: dict[str, Any] = {}
    for name, raw in exposures.items():
        safe_name = str(name).strip()
        if not safe_name or not _SAFE_CODE.fullmatch(safe_name):
            raise AggregateOnlyError("procedure exposure name is malformed")
        if not isinstance(raw, Mapping):
            raise AggregateOnlyError("procedure exposure records must be objects")
        count = _support_int(raw.get("patients", raw.get("count", 0)), label=f"procedure.{safe_name}")
        public[safe_name] = {
            "patient_count": safe_count(count),
            "exposure_only": True,
            "validated_disease_label": False,
        }
    result["exposures"] = public
    return MappingProxyType(result)


def build_target_registry(
    candidates: Sequence[ConditionCandidate] | None = None,
    *,
    aggregate_support: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    frozen = _validate_candidate_sequence(
        tuple(freeze_condition_candidates() if candidates is None else candidates)
    )
    audit = summarize_aggregate_support(aggregate_support, candidates=frozen)
    target_rows: list[dict[str, Any]] = []
    for item in frozen:
        row = item.to_dict()
        row["support_audit"] = audit["candidates"][item.source_code]
        row["target_semantics"] = "source_aligned_survey_condition_proxy"
        row["validated_disease_label"] = False
        row["recorded_code_occurrence_proxy"] = False
        target_rows.append(row)
    return MappingProxyType(
        {
            "schema_version": SCHEMA_VERSION + "-target-registry",
            "status": "frozen_before_endpoint_values",
            "registry_version": "baras-v6-2-expanded-survey-condition-target-registry-v1",
            "all_declared_candidates_retained": True,
            "candidate_count": len(target_rows),
            "selection_uses_model_performance": False,
            "outcome_values_accessed": False,
            "lanes": {
                LANE_FULL_CONTEXT: "exact_target_source_and_direct_features_removed",
                LANE_DE_NOVO: "all_condition_history_flags_and_candidate_sources_removed",
            },
            "targets": target_rows,
        }
    )


def build_endpoint_atlas(
    *,
    mapping_rows: Iterable[Mapping[str, Any]] | None = None,
    candidates: Sequence[ConditionCandidate] | None = None,
    aggregate_support: Mapping[str, Any] | None = None,
    procedure_audit: Mapping[str, Any] | None = None,
    outer_fold_sha256: str = V62_OUTER_FOLD_SHA256,
    mapping_sha256: str | None = None,
    outcome_values: Any = None,
    target_values: Any = None,
    model_scores: Any = None,
    model_performance: Any = None,
) -> Mapping[str, Any]:
    """Materialize row-free endpoint definitions and optional aggregate audit."""

    assert_no_outcome_values(
        outcome_values=outcome_values,
        target_values=target_values,
        model_scores=model_scores,
        model_performance=model_performance,
    )
    if str(outer_fold_sha256) != V62_OUTER_FOLD_SHA256:
        raise FoldBindingError("only the exact V6.2 outer-fold hash is accepted")
    if mapping_rows is not None and candidates is not None:
        raise MappingIntegrityError("provide mapping rows or frozen candidates, not both")
    frozen_candidates = (
        tuple(candidates)
        if candidates is not None
        else freeze_condition_candidates(mapping_rows)
    )
    _validate_candidate_sequence(frozen_candidates)
    registry = build_target_registry(frozen_candidates, aggregate_support=aggregate_support)
    procedure = summarize_procedure_exposure_audit(procedure_audit)
    source_mapping_sha256 = (
        str(mapping_sha256)
        if mapping_sha256 is not None
        else PUBLIC_CUSTOM_CODE_MASTER_SHA256
        if mapping_rows is None
        else "synthetic_mapping_rows_in_memory"
    )
    return MappingProxyType(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "precommit_frozen_before_endpoint_values",
            "outer_fold_assignment_sha256": V62_OUTER_FOLD_SHA256,
            "outer_fold_count": OUTER_FOLD_COUNT,
            "source_mapping_sha256": source_mapping_sha256,
            "source_prefixes": list(MAPPING_SOURCE_PREFIXES),
            "source_exact_inclusions": [MAPPING_SOURCE_EXACT],
            "free_text_sources_excluded": list(FREE_TEXT_SOURCE_CODES),
            "candidate_source_codes": list(candidate_source_codes(frozen_candidates)),
            "candidate_count": len(frozen_candidates),
            "condition_history_flag_names": list(V62_CLINICAL_FLAG_NAMES),
            "direct_feature_exclusions": {
                key: list(value) for key, value in V62_CLINICAL_FLAG_DIRECT_FEATURE_EXCLUSIONS.items()
            },
            "source_to_direct_v6_2_flag": dict(SOURCE_TO_DIRECT_V62_FLAG),
            "claim_reporting": _claim_reporting_metadata(),
            "survey_label_policy": dict(SURVEY_LABEL_POLICY),
            "target_registry": registry,
            "support_audit": summarize_aggregate_support(aggregate_support, candidates=frozen_candidates),
            "procedure_exposure_audit": procedure,
            "current_v6_2_tower_count": 2,
            "procedure_fourth_tower_added": False,
            "selection_uses_model_performance": False,
            "outcome_values_accessed": False,
            "patient_rows_or_identifiers_emitted": False,
            "official_test_inputs_loaded": False,
        }
    )


def _expected_protocol_candidates(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    raw = protocol.get("candidate_source_codes")
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ProtocolError("protocol candidate source list is malformed")
    return tuple(raw)


def validate_protocol(
    project_root: str | Path | None = None,
    protocol_path: str | Path | None = None,
    *,
    verify_bindings: bool = True,
) -> tuple[Mapping[str, Any], str]:
    """Authenticate the canonical protocol and frozen code/test bindings."""

    root = Path(__file__).resolve().parent if project_root is None else Path(project_root).resolve()
    path = (root / PROTOCOL_NAME) if protocol_path is None else Path(protocol_path).resolve()
    path = path.resolve()
    if path != (root / PROTOCOL_NAME).resolve():
        raise ProtocolError("expanded endpoint-atlas protocol must be canonical")
    value, digest = _load_json(path)
    if not isinstance(value, Mapping):
        raise ProtocolError("expanded endpoint-atlas protocol must be one object")
    if value.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ProtocolError("expanded endpoint-atlas protocol schema differs")
    if value.get("status") != "frozen_before_expanded_endpoint_values_or_scores":
        raise ProtocolError("expanded endpoint-atlas protocol is not frozen")
    if value.get("outer_fold_assignment_sha256") != V62_OUTER_FOLD_SHA256:
        raise FoldBindingError("protocol outer-fold hash differs")
    if value.get("outer_fold_count") != OUTER_FOLD_COUNT:
        raise FoldBindingError("protocol outer-fold count differs")
    source_mapping = value.get("source_mapping")
    if not isinstance(source_mapping, Mapping):
        raise ProtocolError("protocol source mapping section is missing")
    if {
        "artifact": source_mapping.get("artifact"),
        "public_repo_url": source_mapping.get("public_repo_url"),
        "public_repo_commit": source_mapping.get("public_repo_commit"),
        "repo_relative_path": source_mapping.get("repo_relative_path"),
        "sha256": source_mapping.get("sha256"),
        "candidate_selection_from_mapping_values": source_mapping.get(
            "candidate_selection_from_mapping_values"
        ),
    } != {
        "artifact": "Custom_Code_Master.csv",
        "public_repo_url": PUBLIC_CUSTOM_CODE_MASTER_REPO_URL,
        "public_repo_commit": PUBLIC_CUSTOM_CODE_MASTER_COMMIT,
        "repo_relative_path": PUBLIC_CUSTOM_CODE_MASTER_REPO_PATH,
        "sha256": PUBLIC_CUSTOM_CODE_MASTER_SHA256,
        "candidate_selection_from_mapping_values": False,
    }:
        raise ProtocolError("protocol public mapping provenance differs")
    if tuple(source_mapping.get("source_prefixes", ())) != MAPPING_SOURCE_PREFIXES:
        raise ProtocolError("protocol source prefixes differ")
    if tuple(source_mapping.get("source_exact_inclusions", ())) != (MAPPING_SOURCE_EXACT,):
        raise ProtocolError("protocol exact source inclusion differs")
    if tuple(source_mapping.get("free_text_sources_excluded", ())) != FREE_TEXT_SOURCE_CODES:
        raise ProtocolError("protocol free-text exclusion differs")
    if _expected_protocol_candidates(value) != EXPECTED_CANDIDATE_SOURCE_CODES:
        raise ProtocolError("protocol candidate source set/order differs")
    if value.get("candidate_count") != len(EXPECTED_CANDIDATE_SOURCE_CODES):
        raise ProtocolError("protocol candidate count differs")
    protocol_candidates = value.get("candidates")
    if not isinstance(protocol_candidates, list) or len(protocol_candidates) != len(EXPECTED_CANDIDATE_SOURCE_CODES):
        raise ProtocolError("protocol candidate declarations are missing")
    candidate_keys = (
        "source_code",
        "label",
        "organ_family",
        "mapped_concept_id",
        "source_mapping_status",
        "defining_fields",
        "direct_feature_exclusions",
        "claim_proximity_class",
        "measurement_proximal_or_policy",
    )
    for expected, raw in zip(_fallback_candidates(), protocol_candidates):
        if not isinstance(raw, Mapping):
            raise ProtocolError("protocol candidate declaration is malformed")
        expected_row = expected.to_dict()
        expected_contract = {key: expected_row[key] for key in candidate_keys}
        if dict(raw) != expected_contract:
            raise ProtocolError(
                f"protocol candidate declaration differs: {expected.source_code}"
            )
    if value.get("condition_history_flag_names") != list(V62_CLINICAL_FLAG_NAMES):
        raise ProtocolError("protocol condition-history flag set differs")
    if value.get("direct_feature_exclusions") != {
        key: list(item) for key, item in V62_CLINICAL_FLAG_DIRECT_FEATURE_EXCLUSIONS.items()
    }:
        raise ProtocolError("protocol direct-feature exclusions differ")
    if value.get("source_to_direct_v6_2_flag") != dict(SOURCE_TO_DIRECT_V62_FLAG):
        raise ProtocolError("protocol source-to-direct-flag mapping differs")
    if value.get("claim_reporting") != _claim_reporting_metadata():
        raise ProtocolError("protocol claim-proximity reporting classes differ")
    gates = value.get("support_gates")
    if gates != {
        "minimum_cases": MIN_CASES,
        "minimum_controls": MIN_CONTROLS,
        "minimum_cases_per_outer_fold": MIN_CASES_PER_OUTER_FOLD,
        "minimum_controls_per_outer_fold": MIN_CONTROLS_PER_OUTER_FOLD,
        "small_cell_threshold": SMALL_CELL_THRESHOLD,
        "gate_fail_action": "retain_candidate_with_withheld_or_missing_or_excluded_status",
        "threshold_or_fold_retuning_allowed": False,
    }:
        raise ProtocolError("protocol support gates differ")
    if value.get("survey_label_policy") != dict(SURVEY_LABEL_POLICY):
        raise ProtocolError("protocol survey label policy differs")
    lanes = value.get("lanes")
    if lanes != {
        LANE_FULL_CONTEXT: {
            "target_specific_rule": "erase_exact_target_source_and_direct_defining_features",
            "condition_history_policy": "retain_other_condition_history_flags",
            "all_condition_history_flags_removed": False,
        },
        LANE_DE_NOVO: {
            "target_specific_rule": "erase_exact_target_source_and_all_condition_history_flags",
            "condition_history_policy": "erase_all_condition_history_flags",
            "continuous_labs_and_vitals_retained": True,
            "all_condition_history_flags_removed": True,
        },
    }:
        raise ProtocolError("protocol lane separation differs")
    procedures = value.get("procedures")
    if procedures != {
        "status": "separate_care_process_exposure_audit",
        "disease_labels": False,
        "fourth_current_v6_2_tower": False,
        "condition_registry_inclusion": False,
    }:
        raise ProtocolError("protocol procedure policy differs")
    privacy = value.get("privacy")
    if privacy != {
        "patient_derived_processing": "local_only",
        "aggregate_only_outputs": True,
        "patient_rows_identifiers_dates_targets_predictions_scores_embeddings_or_folds_serialized": False,
        "official_test_inputs_or_targets_allowed": False,
        "target_values_accessed_during_precommit": False,
    }:
        raise ProtocolError("protocol privacy boundary differs")
    if value.get("selection") != {
        "candidate_selection_uses_model_performance": False,
        "candidate_selection_uses_outcome_values": False,
        "all_declared_candidates_retained": True,
        "reuse_existing_v6_2_outer_folds": True,
    }:
        raise ProtocolError("protocol selection boundary differs")
    if verify_bindings:
        bindings = value.get("bindings")
        expected_labels = {"kernel", "runner", "synthetic_tests"}
        if not isinstance(bindings, Mapping) or set(bindings) != expected_labels:
            raise ProtocolError("protocol code-binding labels differ")
        for label, raw in bindings.items():
            if not isinstance(raw, Mapping) or set(raw) != {"file", "sha256"}:
                raise ProtocolError(f"protocol code binding malformed: {label}")
            name = str(raw.get("file", ""))
            digest_value = raw.get("sha256")
            if Path(name).is_absolute() or ".." in Path(name).parts or not _SHA256.fullmatch(str(digest_value)):
                raise ProtocolError(f"protocol code binding unsafe: {label}")
            source = (root / name).resolve()
            if not source.is_file() or sha256_file(source) != digest_value:
                raise ProtocolError(f"protocol code binding differs: {label}")
    return value, digest


def validate_v6_2_expanded_endpoint_atlas_protocol(
    project_root: str | Path | None = None,
    protocol_path: str | Path | None = None,
    *,
    verify_bindings: bool = True,
) -> tuple[Mapping[str, Any], str]:
    return validate_protocol(project_root, protocol_path, verify_bindings=verify_bindings)


def assert_aggregate_only_payload(payload: Any) -> None:
    """Validate a receipt before it is written or returned to a caller."""

    _reject_forbidden_payload(payload, where="output")
    if not isinstance(payload, Mapping):
        raise AggregateOnlyError("output must be a JSON object")
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    if any(
        f'"{token}":' in encoded
        for token in ("person_id", "patient_id", "target_values", "outcome_values")
    ):
        raise AggregateOnlyError("aggregate output contains a forbidden identity or value token")
    if payload.get("official_test_inputs_loaded") is not False:
        raise OfficialTestRefusal("output does not attest official-test refusal")
    if payload.get("patient_rows_or_identifiers_emitted") is not False:
        raise AggregateOnlyError("output does not attest row-free emission")


def build_failure_payload(error: BaseException) -> dict[str, Any]:
    # Exception messages are intentionally omitted because malformed input
    # names could contain identifiers or dates.
    return {
        "schema_version": RUNNER_SCHEMA_VERSION + "-failure-v1",
        "status": "failed_closed",
        "error_type": type(error).__name__,
        "details_emitted": False,
        "official_test_inputs_loaded": False,
        "patient_rows_or_identifiers_emitted": False,
        "outcome_values_accessed": False,
        "model_scores_loaded": False,
        "retry_requires_preserving_this_artifact": True,
    }


# Compatibility aliases make the narrow seam discoverable without creating a
# second implementation surface.
materialize_precommit = build_endpoint_atlas
build_expanded_endpoint_atlas = build_endpoint_atlas
validate_expanded_endpoint_atlas_protocol = validate_protocol
