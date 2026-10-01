"""Local-only runner for the frozen V6.2 expanded-endpoint evaluation.

Production execution accepts explicit dataset and clinical-project roots and
binds the authenticated V6.2 cohort/fold/representation path.  The flat
``EvaluationInput``/standardizer seam is retained only for row-free synthetic
tests.  Every path validates the official-train/validation boundary before
labels are coerced and writes exactly one aggregate success or redacted
failure artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from eval_soft_patient_atlas import (
    ARM_BLOOD,
    ARM_BOTH,
    ARM_EYE,
    BASE_STRATUM,
    FeatureView,
    RepresentationFitRequest,
    _fit_demographic_transform,
    _partition_outer_train,
    _patient_hash,
    _patient_json_hash,
)
from patient_atlas_disease_readout import DiseaseReadoutConfig
from patient_atlas_v6_2_expanded_endpoint_atlas import (
    LANE_DE_NOVO,
    LANE_FULL_CONTEXT,
    EXPECTED_CANDIDATE_SOURCE_CODES,
    V62_CLINICAL_FLAG_NAMES,
    freeze_condition_candidates,
    target_specific_exclusions,
)
from patient_atlas_v6_2_expanded_endpoint_evaluation import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_MAXIMUM_PENALTY_COMBINATIONS,
    MULTIPLIER_DRAW_CHUNK_SIZE,
    EXACT_OUTER_FOLD_HASH,
    EXACT_INNER_FOLD_ASSIGNMENT_SHA256,
    FROZEN_ATLAS_PROTOCOL_SHA256,
    FROZEN_PRECOMMIT_RECEIPT_SHA256,
    FROZEN_SUPPORT_RECEIPT_NAME,
    FROZEN_SUPPORT_RECEIPT_SHA256,
    EXPECTED_ELIGIBLE_SOURCE_CODES,
    EXPECTED_ELIGIBLE_BY_CLAIM_PROXIMITY_CLASS,
    EXPECTED_WITHHELD_SOURCE_CODES,
    ROUTES,
    RUNNER_SCHEMA_VERSION,
    EligibleSupport,
    EvaluationInput,
    FoldBasisError,
    DirectLabelLeakageError,
    EndpointSupportDriftError,
    InputContractError,
    OfficialTestRefusal,
    ProtocolBindingError,
    RepresentationCoordinates,
    assert_aggregate_only_result,
    build_failure_payload,
    build_lane_aggregate_report,
    canonical_sha256,
    frozen_candidate_metadata,
    load_eligible_support_receipt,
    make_inner_fold_ids,
    nested_logistic_readout,
    physically_erase_target_channels,
    route_design,
    validate_development_partitions,
    validate_protocol,
    validate_support_against_observed,
    _call_factory_fit,
    _call_model_transform,
    _validate_fold_pair,
    _validate_outer_fit_provenance,
    _validate_explicit_binary_labels,
)


LANES = (LANE_FULL_CONTEXT, LANE_DE_NOVO)


class StandardizingFoldModel:
    """Small deterministic representation adapter for synthetic/local use.

    It fits center/scale on the supplied outer-training matrix only, then
    applies the same basis to every route's outer-train and outer-test rows.
    A real local adapter may provide a frozen V6.2 encoder with the same
    provenance-tagged transform contract.
    """

    def __init__(
        self,
        *,
        center: np.ndarray,
        scale: np.ndarray,
        feature_groups: tuple[str, ...],
        outer_fold: int,
        lane: str,
        target_source_code: str,
    ) -> None:
        self.center = np.asarray(center, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.feature_groups = tuple(feature_groups)
        self.outer_fold = int(outer_fold)
        self.fit_outer_fold = int(outer_fold)
        self.fit_outer_train_only = True
        self.fit_fold_ids = tuple(fold for fold in range(5) if fold != outer_fold)
        self.model_token = (
            f"v6-2-expanded-endpoint:{lane}:{target_source_code}:outer-{outer_fold}:standardized"
        )
        self.basis_token = self.model_token

    def transform(
        self,
        values: np.ndarray,
        *,
        route: str,
        is_outer_test: bool = False,
    ) -> RepresentationCoordinates:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_groups):
            raise FoldBasisError("standardizing model input width differs")
        encoded = (matrix - self.center[None, :]) / self.scale[None, :]
        return RepresentationCoordinates(
            values=encoded,
            feature_groups=self.feature_groups,
            fold_id=self.outer_fold,
            model_token=self.model_token,
            basis_token=self.basis_token,
            outer_train_only=True,
            target_values_seen=False,
            outer_test_seen_during_fit=False,
        )


class StandardizingRepresentationFactory:
    """Fold-specific numeric preprocessing with no target/test access."""

    def fit(
        self,
        train_values: np.ndarray,
        *,
        feature_names: Sequence[str],
        feature_groups: Sequence[str],
        outer_fold: int,
        lane: str,
        target_source_code: str,
    ) -> StandardizingFoldModel:
        matrix = np.asarray(train_values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(feature_names) or matrix.shape[1] != len(feature_groups):
            raise FoldBasisError("factory fit matrix and metadata differ")
        if not np.isfinite(matrix).all():
            raise FoldBasisError("factory fit matrix is not finite")
        center = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return StandardizingFoldModel(
            center=center,
            scale=scale,
            feature_groups=tuple(str(item) for item in feature_groups),
            outer_fold=int(outer_fold),
            lane=str(lane),
            target_source_code=str(target_source_code),
        )


def _coerce_evaluation_input(raw: EvaluationInput | Mapping[str, Any]) -> EvaluationInput:
    if isinstance(raw, EvaluationInput):
        return raw
    if not isinstance(raw, Mapping):
        raise InputContractError("local loader must return EvaluationInput or one mapping")
    required = {
        "feature_values",
        "feature_names",
        "feature_groups",
        "labels_by_source",
        "observed_by_source",
        "outer_fold_ids",
        "participant_splits",
    }
    if not required <= set(raw):
        raise InputContractError("local loader mapping is missing evaluation fields")
    return EvaluationInput(
        feature_values=np.asarray(raw["feature_values"]),
        feature_names=tuple(str(item) for item in raw["feature_names"]),
        feature_groups=tuple(str(item) for item in raw["feature_groups"]),
        labels_by_source=raw["labels_by_source"],
        observed_by_source=raw["observed_by_source"],
        outer_fold_ids=np.asarray(raw["outer_fold_ids"]),
        participant_splits=tuple(str(item) for item in raw["participant_splits"]),
        outer_fold_hash=str(raw.get("outer_fold_hash", EXACT_OUTER_FOLD_HASH)),
    )


def _readonly_local(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Copy one local patient array before handing it to a representation API."""

    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _subset_actual_feature_view(view: FeatureView, indices: Sequence[int]) -> FeatureView:
    """Subset an outcome-free FeatureView without exposing its identifiers."""

    positions = np.asarray(indices, dtype=np.int64)
    if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= view.size):
        raise FoldBasisError("actual feature-view subset is malformed")
    features = {
        str(name): _readonly_local(values[positions])
        for name, values in view.features.items()
    }
    return FeatureView(
        patient_ids=tuple(view.patient_ids[int(index)] for index in positions),
        site_ids=tuple(view.site_ids[int(index)] for index in positions),
        features=MappingProxyType(features),
        demographics=_readonly_local(view.demographics[positions], dtype=np.float64),
        demographic_mask=_readonly_local(view.demographic_mask[positions], dtype=bool),
    )


def _actual_base_feature_view(feature_cohort: Any) -> FeatureView:
    """Build the existing V6.2 outcome-free FeatureView in cohort order."""

    from patient_atlas_aireadi_internal_crossfit import _evaluation_features

    features = _evaluation_features(feature_cohort)
    return FeatureView(
        patient_ids=tuple(str(value) for value in feature_cohort.patient_ids),
        site_ids=tuple(str(value) for value in feature_cohort.site_ids),
        features=MappingProxyType(
            {
                str(name): _readonly_local(values)
                for name, values in features.items()
            }
        ),
        demographics=_readonly_local(
            np.asarray(feature_cohort.ages, dtype=np.float64)[:, None],
            dtype=np.float64,
        ),
        demographic_mask=_readonly_local(
            np.asarray(feature_cohort.age_observed_mask, dtype=bool)[:, None],
            dtype=bool,
        ),
    )


def _mask_actual_feature_view(
    view: FeatureView,
    *,
    feature_names: Sequence[str],
    target_source_code: str,
    lane: str,
) -> FeatureView:
    """Physically mask direct clinical channels before a V6.2 transform."""

    from patient_atlas_aireadi_internal_crossfit import (
        FEATURE_BLOOD_ELIGIBLE,
        FEATURE_BLOOD_OBSERVED,
        FEATURE_BLOOD_VALUES,
    )

    features = dict(view.features)
    if not {
        FEATURE_BLOOD_VALUES,
        FEATURE_BLOOD_OBSERVED,
        FEATURE_BLOOD_ELIGIBLE,
    } <= set(features):
        raise DirectLabelLeakageError("actual V6.2 feature view lacks clinical channels")
    clinical = np.asarray(features[FEATURE_BLOOD_VALUES])
    observed = np.asarray(features[FEATURE_BLOOD_OBSERVED])
    eligible = np.asarray(features[FEATURE_BLOOD_ELIGIBLE])
    if clinical.ndim != 2 or observed.shape != clinical.shape:
        raise DirectLabelLeakageError("actual clinical values and observation mask differ")
    if eligible.ndim == 1:
        eligible = np.broadcast_to(eligible[None, :], clinical.shape)
    if eligible.shape != clinical.shape or eligible.dtype != np.bool_:
        raise DirectLabelLeakageError("actual clinical policy mask differs")
    safe, _, safe_observed = physically_erase_target_channels(
        clinical,
        feature_names,
        target_source_code=target_source_code,
        lane=lane,
        observed_mask=np.asarray(observed, dtype=bool),
    )
    if safe_observed is None:
        raise DirectLabelLeakageError("actual clinical erasure did not return a mask")
    features[FEATURE_BLOOD_VALUES] = _readonly_local(safe, dtype=np.float64)
    features[FEATURE_BLOOD_OBSERVED] = _readonly_local(safe_observed, dtype=bool)
    # The V6.2 fold preprocessor binds the policy-eligibility mask exactly at
    # fit and runtime.  Target erasure is represented by physical zeroes and
    # an observed-mask drop; the policy mask itself is intentionally retained.
    features[FEATURE_BLOOD_ELIGIBLE] = _readonly_local(eligible, dtype=bool)
    return FeatureView(
        patient_ids=view.patient_ids,
        site_ids=view.site_ids,
        features=MappingProxyType(
            {str(name): _readonly_local(values) for name, values in features.items()}
        ),
        demographics=view.demographics,
        demographic_mask=view.demographic_mask,
    )


def _actual_exclusion_profile(
    feature_names: Sequence[str],
    *,
    target_source_code: str,
    lane: str,
) -> tuple[str, tuple[str, ...]]:
    """Return the target-mask profile used for safe route-design caching.

    ``de_novo`` has one profile for every endpoint.  ``full_context`` has one
    profile per direct-history flag (plus a shared profile for endpoints with
    no mapped flag).  The exact source field is included when it is present in
    the V6.2 clinical feature schema, so this key cannot merge unequal masks.
    """

    exclusions = target_specific_exclusions(target_source_code)
    excluded_key = (
        "full_context_excluded_features"
        if lane == LANE_FULL_CONTEXT
        else "de_novo_excluded_features"
    )
    excluded = {str(value).strip().lower() for value in exclusions[excluded_key]}
    present = tuple(
        str(name).strip().lower()
        for name in feature_names
        if str(name).strip().lower() in excluded
    )
    return str(lane), present


def _actual_representation_fit_phase_views(
    phase_views: Sequence[FeatureView],
    *,
    feature_names: Sequence[str],
    lane: str,
) -> tuple[FeatureView, ...]:
    """Apply the frozen representation-fit lane mask to all fit phases."""

    phases = tuple(phase_views)
    if len(phases) != 3:
        raise FoldBasisError("actual V6.2 representation fit requires three phases")
    if lane == LANE_FULL_CONTEXT:
        # Full context is intentionally history-informed during outer-train
        # representation fitting; direct target erasure occurs on every
        # endpoint transform, never by silently changing this fit contract.
        return phases
    if lane != LANE_DE_NOVO:
        raise FoldBasisError("unknown actual V6.2 evaluation lane")
    return tuple(
        _mask_actual_feature_view(
            view,
            feature_names=feature_names,
            target_source_code=EXPECTED_CANDIDATE_SOURCE_CODES[0],
            lane=LANE_DE_NOVO,
        )
        for view in phases
    )


def _clear_actual_representation_caches(fitted: Any) -> None:
    """Invalidate the existing transform cache before changing lane masks.

    The V6.2 fitted adapter keys its private feature cache by patient-ID set.
    Expanded-endpoint target masks intentionally keep that set fixed, so a
    stale cached batch would otherwise replay one endpoint's channels for the
    next endpoint.  Caches are local implementation state and are never
    serialized.
    """

    for name in ("_batch_cache", "_baseline_cache", "_atlas_cache"):
        cache = getattr(fitted, name, None)
        if isinstance(cache, dict):
            cache.clear()


def _actual_modality_groups(raw_groups: Sequence[str], route: str) -> tuple[str, ...]:
    """Map V6.2 latent blocks to the two grouped-readout modality families."""

    groups = tuple(str(value) for value in raw_groups)
    if not groups:
        raise FoldBasisError("actual V6.2 transform returned no penalty groups")
    if route == "eye_only":
        return ("eye",) * len(groups)
    if route == "clinical_only":
        return ("clinical",) * len(groups)
    if route != "both":
        raise FoldBasisError("actual modality group route is not prespecified")
    mapped: list[str] = []
    for group in groups:
        token = group.strip().lower()
        if token.startswith("eye"):
            mapped.append("eye")
        elif token.startswith("clinical") or token.startswith("blood"):
            mapped.append("clinical")
        else:
            raise FoldBasisError("actual V6.2 transform returned an unknown modality block")
    return tuple(mapped)


def _actual_transform_pair(
    fitted: Any,
    train_view: FeatureView,
    test_view: FeatureView,
    *,
    arm: str,
    outer_fold: int,
) -> tuple[Any, Any]:
    """Transform one availability arm with one fold-specific model/basis."""

    train_coordinates = fitted.transform(train_view, arm=arm, stratum=BASE_STRATUM)
    test_coordinates = fitted.transform(test_view, arm=arm, stratum=BASE_STRATUM)
    if (
        train_coordinates.fold_key != test_coordinates.fold_key
        or train_coordinates.model_token != test_coordinates.model_token
        or train_coordinates.basis_token != test_coordinates.basis_token
        or train_coordinates.values.shape[1] != test_coordinates.values.shape[1]
        or tuple(train_coordinates.penalty_groups)
        != tuple(test_coordinates.penalty_groups)
        or train_coordinates.fold_key != f"disease-universal-seed-1701-outer-{outer_fold}"
    ):
        raise FoldBasisError("actual V6.2 outer train/test transforms do not share one basis")
    if tuple(train_coordinates.patient_ids) != tuple(train_view.patient_ids) or tuple(test_coordinates.patient_ids) != tuple(test_view.patient_ids):
        raise FoldBasisError("actual V6.2 transform changed patient order")
    return train_coordinates, test_coordinates


def _actual_route_designs(
    fitted: Any,
    train_view: FeatureView,
    test_view: FeatureView,
    *,
    outer_fold: int,
) -> tuple[
    Mapping[str, np.ndarray],
    Mapping[str, np.ndarray],
    Mapping[str, tuple[str, ...]],
]:
    """Build fresh age/availability designs from the same fold representation."""

    demographic_transform = _fit_demographic_transform(train_view)
    age_train = demographic_transform.apply(
        np.asarray(train_view.demographics), np.asarray(train_view.demographic_mask)
    )
    age_test = demographic_transform.apply(
        np.asarray(test_view.demographics), np.asarray(test_view.demographic_mask)
    )
    designs_train: dict[str, np.ndarray] = {"age_only": np.asarray(age_train, dtype=np.float64)}
    designs_test: dict[str, np.ndarray] = {"age_only": np.asarray(age_test, dtype=np.float64)}
    groups: dict[str, tuple[str, ...]] = {"age_only": ("age",)}
    arms = {
        "eye_only": ARM_EYE,
        "clinical_only": ARM_BLOOD,
        "both": ARM_BOTH,
    }
    for route, arm in arms.items():
        train_coordinates, test_coordinates = _actual_transform_pair(
            fitted,
            train_view,
            test_view,
            arm=arm,
            outer_fold=outer_fold,
        )
        modality_groups = _actual_modality_groups(
            train_coordinates.penalty_groups,
            route,
        )
        if route == "eye_only" and not all(value == "eye" for value in modality_groups):
            raise FoldBasisError("eye availability re-encoding lost eye modality provenance")
        if route == "clinical_only" and not all(value == "clinical" for value in modality_groups):
            raise FoldBasisError("clinical availability re-encoding lost clinical provenance")
        train_values = np.concatenate(
            [np.asarray(train_coordinates.values, dtype=np.float64), age_train], axis=1
        )
        test_values = np.concatenate(
            [np.asarray(test_coordinates.values, dtype=np.float64), age_test], axis=1
        )
        designs_train[route] = train_values
        designs_test[route] = test_values
        groups[route] = modality_groups + ("age",)
    return MappingProxyType(designs_train), MappingProxyType(designs_test), MappingProxyType(groups)


def _new_loss_sink(size: int) -> np.ndarray:
    return np.full(int(size), np.nan, dtype=np.float64)


def _fit_endpoint_lane(
    *,
    data: EvaluationInput,
    source: str,
    lane: str,
    representation_factory: Any,
    readout_config: DiseaseReadoutConfig,
    penalty_grid: Sequence[float],
    maximum_penalty_combinations: int,
) -> tuple[Mapping[str, Mapping[str, np.ndarray]], Mapping[str, float | None], Mapping[str, Any]]:
    """Evaluate all four fresh route readouts for one target/lane."""

    folds = np.asarray(data.outer_fold_ids, dtype=np.int64)
    values = np.asarray(data.feature_values, dtype=np.float64)
    labels = np.asarray(data.labels_by_source[source], dtype=np.float64)
    observed = np.asarray(data.observed_by_source[source], dtype=bool)
    losses = {route: _new_loss_sink(data.size) for route in ROUTES}
    auc_values: dict[str, list[float]] = {route: [] for route in ROUTES}
    selected_penalties: dict[str, list[Mapping[str, float]]] = {route: [] for route in ROUTES}
    seen_model_tokens: set[str] = set()
    seen_basis_tokens: set[str] = set()

    for outer_fold in range(5):
        test_indices = np.flatnonzero(folds == outer_fold)
        train_indices = np.flatnonzero(folds != outer_fold)
        if not len(test_indices) or not len(train_indices):
            raise FoldBasisError("outer fold has an empty train or test partition")
        safe_train, names, _ = physically_erase_target_channels(
            values[train_indices],
            data.feature_names,
            target_source_code=source,
            lane=lane,
        )
        safe_test, _, _ = physically_erase_target_channels(
            values[test_indices],
            data.feature_names,
            target_source_code=source,
            lane=lane,
        )
        model = _call_factory_fit(
            representation_factory,
            safe_train,
            feature_names=names,
            feature_groups=data.feature_groups,
            outer_fold=outer_fold,
            lane=lane,
            target_source_code=source,
        )
        _validate_outer_fit_provenance(model, outer_fold)
        inner_fold_ids = make_inner_fold_ids(
            folds,
            train_indices,
            outer_fold=outer_fold,
        )
        train_y = labels[train_indices]
        test_y = labels[test_indices]
        train_eligible = observed[train_indices]
        test_eligible = observed[test_indices]
        for route in ROUTES:
            train_coordinates = _call_model_transform(
                model, safe_train, route=route, is_outer_test=False
            )
            test_coordinates = _call_model_transform(
                model, safe_test, route=route, is_outer_test=True
            )
            if route == ROUTES[0]:
                # A pooled representation often exposes one basis/model token
                # for every fold.  Require a genuinely fold-specific fit so a
                # hidden global standardizer cannot enter this evaluation.
                if test_coordinates.model_token in seen_model_tokens or test_coordinates.basis_token in seen_basis_tokens:
                    raise FoldBasisError("one representation basis was reused across outer folds")
                seen_model_tokens.add(test_coordinates.model_token)
                seen_basis_tokens.add(test_coordinates.basis_token)
            _validate_fold_pair(
                train_coordinates,
                test_coordinates,
                outer_fold=outer_fold,
                route=route,
            )
            train_design, route_groups = route_design(
                train_coordinates.values,
                train_coordinates.feature_groups,
                route,
            )
            test_design, test_groups = route_design(
                test_coordinates.values,
                test_coordinates.feature_groups,
                route,
            )
            if route_groups != test_groups:
                raise FoldBasisError("train/test route groups differ")
            result = nested_logistic_readout(
                route=route,
                x_train=train_design,
                y_train=train_y,
                train_eligible=train_eligible,
                inner_fold_ids=inner_fold_ids,
                x_test=test_design,
                y_test=test_y,
                test_eligible=test_eligible,
                feature_groups=route_groups,
                penalty_grid=penalty_grid,
                config=readout_config,
                maximum_penalty_combinations=maximum_penalty_combinations,
            )
            losses[route][test_indices] = result.test_losses
            if result.test_auc is not None:
                auc_values[route].append(float(result.test_auc))
            selected_penalties[route].append(dict(result.selected_penalties))
        for route in ROUTES:
            expected = observed[test_indices]
            if not np.array_equal(np.isfinite(losses[route][test_indices]), expected):
                raise FoldBasisError("route loss eligibility does not match observed labels")

    public_auc = {
        route: (None if not values else float(np.mean(values)))
        for route, values in auc_values.items()
    }
    public_tuning = {
        route: {
            "fresh_outer_fold_readouts": len(selected_penalties[route]),
            "inner_tuning_outer_test_labels_used": False,
        }
        for route in ROUTES
    }
    return losses, public_auc, public_tuning


def _load_actual_expanded_endpoint_labels(
    *,
    dataset_root: Path,
    cohort: Any,
    fold_map: Any,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    """Load source-aligned index-visit labels into local arrays only."""

    # The support materializer owns the authenticated CSV schema and index-date
    # rule.  Reuse its parser rather than introducing a second interpretation
    # of the source observation table.  These imports are deliberately lazy so
    # importing the runner never opens a patient-derived artifact.
    from patient_atlas_v6_2_expanded_endpoint_atlas_support_materializer import (
        _load_dataset_observations,
        _load_dataset_participants,
        _load_dataset_visit_dates,
    )

    participants = _load_dataset_participants(
        dataset_root,
        cohort=cohort,
        fold_map=fold_map,
    )
    visit_dates = _load_dataset_visit_dates(dataset_root)
    observations = _load_dataset_observations(
        dataset_root,
        participants=participants,
        visit_dates=visit_dates,
    )
    positions = {
        str(participant.participant_id): index
        for index, participant in enumerate(participants)
    }
    labels = {
        source: np.zeros(len(participants), dtype=np.float64)
        for source in EXPECTED_CANDIDATE_SOURCE_CODES
    }
    observed = {
        source: np.zeros(len(participants), dtype=bool)
        for source in EXPECTED_CANDIDATE_SOURCE_CODES
    }
    for (participant_id, source), value in observations.items():
        if source not in labels or participant_id not in positions:
            raise InputContractError("local endpoint observation escaped the cohort set")
        index = positions[participant_id]
        numeric = float(value)
        if numeric not in (0.0, 1.0):
            raise InputContractError("local endpoint labels are not explicit numeric 0/1")
        labels[source][index] = numeric
        observed[source][index] = True
    for source in EXPECTED_CANDIDATE_SOURCE_CODES:
        _validate_explicit_binary_labels(labels[source], observed[source], source)
    return MappingProxyType(labels), MappingProxyType(observed)


def _fit_actual_v6_2_representation_folds(
    *,
    root: Path,
    feature_cohort: Any,
    base_view: FeatureView,
    outer_assignment: np.ndarray,
    representation_protocol: Mapping[str, Any],
    lane: str,
) -> tuple[tuple[int, np.ndarray, np.ndarray, Any], ...]:
    """Fit one canonical V6.2 model per outer fold for one evaluation lane."""

    from patient_atlas_aireadi_internal_crossfit import _training_from_protocol
    from patient_atlas_aireadi_v6_2_evaluation import AIReadIV62FoldRepresentationFactory

    factory = AIReadIV62FoldRepresentationFactory(
        project_root=root,
        ordered_feature_names=tuple(feature_cohort.feature_names),
        source_policy_sha256=str(feature_cohort.source_policy_sha256),
        source_hashes=feature_cohort.source_hashes,
        base_training_seed=int(
            representation_protocol["representation"]["training"]["base_training_seed"]
        ),
        training=_training_from_protocol(representation_protocol),
    )
    fitted_folds: list[tuple[int, np.ndarray, np.ndarray, Any]] = []
    model_tokens: set[str] = set()
    for outer_fold in range(5):
        test_indices = np.flatnonzero(outer_assignment == outer_fold)
        train_indices = np.flatnonzero(outer_assignment != outer_fold)
        if not len(test_indices) or not len(train_indices):
            raise FoldBasisError("actual V6.2 outer fold has an empty partition")
        train_view = _subset_actual_feature_view(base_view, train_indices)
        fit_rel, validation_rel, calibration_rel = _partition_outer_train(
            train_view.patient_ids,
            train_view.site_ids,
            fractions=(0.7, 0.15, 0.15),
            salt=(
                "patient-atlas-v5-disease-representation-phases-v1:"
                f"disease-universal-seed-1701-outer-{outer_fold}"
            ),
        )
        phase_views = (
            _subset_actual_feature_view(train_view, fit_rel),
            _subset_actual_feature_view(train_view, validation_rel),
            _subset_actual_feature_view(train_view, calibration_rel),
        )
        phase_views = _actual_representation_fit_phase_views(
            phase_views,
            feature_names=feature_cohort.feature_names,
            lane=lane,
        )
        request = RepresentationFitRequest(
            fold_key=f"disease-universal-seed-1701-outer-{outer_fold}",
            fit=phase_views[0],
            validation=phase_views[1],
            calibration=phase_views[2],
            expected_outer_train_patient_hash=_patient_hash(train_view.patient_ids),
            expected_outer_test_patient_id_hash=_patient_json_hash(
                tuple(base_view.patient_ids[index] for index in test_indices)
            ),
            expected_outer_test_patient_count=len(test_indices),
        )
        fitted = factory.fit(request)
        provenance = getattr(fitted, "provenance", None)
        if provenance is None or provenance.fold_key != request.fold_key:
            raise FoldBasisError("actual V6.2 fitted representation provenance differs")
        token = str(provenance.model_token)
        if not token or token in model_tokens:
            raise FoldBasisError("actual V6.2 representation model was reused across folds")
        model_tokens.add(token)
        summary = factory.fold_summaries[-1]
        if (
            summary.get("retinal_recovery_gate_passed") is not True
            or summary.get("capacity_and_prior_reversion_gates_passed") is not True
        ):
            raise FoldBasisError("actual V6.2 representation gate failed before readout")
        fitted_folds.append((outer_fold, train_indices, test_indices, fitted))
    if len(fitted_folds) != 5:
        raise FoldBasisError("actual V6.2 evaluation requires five representation folds")
    return tuple(fitted_folds)


def _load_actual_v6_2_context(
    *,
    root: Path,
    dataset_root: str | Path,
    clinical_project_root: str | Path,
    support: EligibleSupport,
) -> Mapping[str, Any]:
    """Authenticate the old V6.2 cohort/fold path and load local labels."""

    # Import the established production path only after a caller explicitly
    # supplies dataset and clinical roots.
    from patient_atlas_disease_folds import make_disease_fold_map, validate_disease_fold_policy
    from patient_atlas_disease_targets import load_development_disease_targets
    from patient_atlas_aireadi_v5_evaluation import validate_v5_internal_evaluation_protocol
    from patient_atlas_real_data import load_exploratory_raw_cohort
    from patient_atlas_prospective_policy import apply_prospective_policy
    from run_patient_atlas_v6_2_screening import validate_v6_2_screening_protocol
    from patient_atlas_v6_2_expanded_endpoint_atlas_support_materializer import (
        authenticate_canonical_dataset_sources,
    )

    dataset = Path(dataset_root).resolve()
    clinical = Path(clinical_project_root).resolve()
    # This is deliberately the first operation on the supplied dataset.  It
    # hashes all three canonical sources before any cohort/target/endpoint
    # parser can open a patient-derived row.  The hashes must match the
    # authenticated support receipt loaded by the outer runner.
    source_hashes = authenticate_canonical_dataset_sources(dataset)
    receipt_hashes = support.receipt.get(
        "source_hashes", support.receipt.get("artifact_hashes")
    )
    if not isinstance(receipt_hashes, Mapping) or dict(receipt_hashes) != dict(source_hashes):
        raise EndpointSupportDriftError(
            "canonical dataset source hashes differ from the support receipt"
        )
    screening_protocol, screening_sha256, _, _ = validate_v6_2_screening_protocol(root)
    if screening_protocol["evaluation"]["outer_fold_assignment_sha256"] != EXACT_OUTER_FOLD_HASH:
        raise ProtocolBindingError("existing V6.2 screening fold hash differs")
    representation_protocol = validate_v5_internal_evaluation_protocol(
        root,
        root / screening_protocol["bindings"]["training_template_protocol"]["file"],
    )
    fold_policy, fold_policy_sha256 = validate_disease_fold_policy(root)
    raw_cohort = load_exploratory_raw_cohort(
        project_root=root,
        dataset_root=dataset,
        clinical_project_root=clinical,
    )
    if set(str(value).strip().lower() for value in raw_cohort.split_labels) - {"train", "val"}:
        raise OfficialTestRefusal("actual V6.2 cohort contains an official-test participant")
    fold_targets, _ = load_development_disease_targets(
        project_root=root,
        dataset_root=dataset,
        cohort=raw_cohort,
    )
    feature_cohort = apply_prospective_policy(raw_cohort, project_root=root)
    if tuple(feature_cohort.patient_ids) != tuple(raw_cohort.patient_ids):
        raise FoldBasisError("prospective feature policy changed patient order")
    outer_map = make_disease_fold_map(
        patient_ids=raw_cohort.patient_ids,
        site_ids=raw_cohort.site_ids,
        targets=fold_targets,
        policy=fold_policy,
    )
    if outer_map.assignment_sha256 != EXACT_OUTER_FOLD_HASH:
        raise ProtocolBindingError("actual V6.2 outer fold assignment differs")
    outer_assignment = outer_map.assignments_for(raw_cohort.patient_ids)
    labels, observed = _load_actual_expanded_endpoint_labels(
        dataset_root=dataset,
        cohort=raw_cohort,
        fold_map=outer_map,
    )
    base_view = _actual_base_feature_view(feature_cohort)
    return MappingProxyType(
        {
            "dataset_root": dataset,
            "clinical_project_root": clinical,
            "screening_protocol_sha256": screening_sha256,
            "representation_protocol": representation_protocol,
            "fold_policy": fold_policy,
            "fold_policy_sha256": fold_policy_sha256,
            "source_hashes": source_hashes,
            "raw_cohort": raw_cohort,
            "feature_cohort": feature_cohort,
            "fold_targets": fold_targets,
            "outer_assignment": outer_assignment,
            "labels_by_source": labels,
            "observed_by_source": observed,
            "base_view": base_view,
        }
    )


def _support_selection_public(
    support: EligibleSupport,
) -> Mapping[str, Any]:
    rows: dict[str, Any] = {}
    for source in EXPECTED_CANDIDATE_SOURCE_CODES:
        detail = support.support_by_source.get(source, {})
        status = "eligible_under_frozen_support_gates" if source in support.eligible_sources else str(
            support.receipt.get("support_audit", {})
            .get("candidates", {})
            .get(source, {})
            .get("status", "support_not_supplied")
        )
        row: dict[str, Any] = {
            "status": status,
            "evaluated": source in support.eligible_sources,
        }
        if isinstance(detail, Mapping):
            for key in ("cases", "controls"):
                if key in detail and isinstance(detail[key], (int, np.integer)):
                    value = int(detail[key])
                    row[key] = value if value == 0 or value >= 10 else "<10"
        rows[source] = row
    return rows


def _run_core(
    *,
    root: Path,
    data: EvaluationInput,
    support: EligibleSupport,
    representation_factory: Any,
    penalty_grid: Sequence[float],
    readout_config: DiseaseReadoutConfig,
    maximum_penalty_combinations: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
    protocol_sha256: str,
) -> Mapping[str, Any]:
    validate_development_partitions(data.participant_splits)
    validate_support_against_observed(
        support,
        data.labels_by_source,
        data.observed_by_source,
        data.outer_fold_ids,
    )
    metadata = frozen_candidate_metadata()
    lane_reports: dict[str, Any] = {}
    tuning_summary: dict[str, Any] = {}
    for lane in LANES:
        endpoint_losses: dict[str, Mapping[str, np.ndarray]] = {}
        endpoint_auc: dict[str, Mapping[str, float | None]] = {}
        endpoint_tuning: dict[str, Any] = {}
        endpoint_metadata = {
            source: metadata[source]
            for source in support.eligible_sources
        }
        for source in support.eligible_sources:
            losses, aucs, tuning = _fit_endpoint_lane(
                data=data,
                source=source,
                lane=lane,
                representation_factory=representation_factory,
                readout_config=readout_config,
                penalty_grid=penalty_grid,
                maximum_penalty_combinations=maximum_penalty_combinations,
            )
            endpoint_losses[source] = losses
            endpoint_auc[source] = aucs
            endpoint_tuning[source] = tuning
        lane_reports[lane] = build_lane_aggregate_report(
            lane=lane,
            endpoint_losses=endpoint_losses,
            endpoint_metadata=endpoint_metadata,
            support_by_source=support.support_by_source,
            auc_by_source=endpoint_auc,
            bootstrap_samples=bootstrap_samples,
            # The frozen protocol defines one multiplier seed for the entire
            # simultaneous family.  A lane-specific offset would silently
            # deviate from that precommit even though point estimates remain
            # unchanged.
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
        tuning_summary[lane] = endpoint_tuning

    report: dict[str, Any] = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "status": "completed_aggregate_only",
        "evaluation_prefix": "BARAS_V6_2_EXPANDED_ENDPOINT_EVALUATION_V1",
        "target_prefix": "patient_atlas_v6_2_expanded_endpoint_evaluation",
        "protocol_sha256": protocol_sha256,
        "atlas_protocol_sha256": FROZEN_ATLAS_PROTOCOL_SHA256,
        "precommit_receipt_sha256": FROZEN_PRECOMMIT_RECEIPT_SHA256,
        "outer_fold_assignment_sha256": EXACT_OUTER_FOLD_HASH,
        "outer_fold_count": 5,
        "candidate_count": len(EXPECTED_CANDIDATE_SOURCE_CODES),
        "candidate_source_codes": list(EXPECTED_CANDIDATE_SOURCE_CODES),
        "free_text_sources_excluded": ["mhoccur_cnsot", "mhoccur_cnrot"],
        "scope": {
            "patient_count": int(data.size),
            "declared_endpoint_count": len(EXPECTED_CANDIDATE_SOURCE_CODES),
            "scored_endpoint_count": len(support.eligible_sources),
            "withheld_endpoint_count": len(EXPECTED_WITHHELD_SOURCE_CODES),
            "support_class_counts": dict(EXPECTED_ELIGIBLE_BY_CLAIM_PROXIMITY_CLASS),
        },
        "support_receipt": {
            "later_loaded": True,
            "frozen_file": FROZEN_SUPPORT_RECEIPT_NAME,
            "frozen_file_sha256": FROZEN_SUPPORT_RECEIPT_SHA256,
            "receipt_sha256": support.receipt_sha256,
            "eligible_endpoint_count": len(support.eligible_sources),
            "frozen_eligible_endpoint_count": len(EXPECTED_ELIGIBLE_SOURCE_CODES),
            "frozen_eligible_by_claim_proximity_class": dict(
                EXPECTED_ELIGIBLE_BY_CLAIM_PROXIMITY_CLASS
            ),
            "selection_uses_model_performance": False,
            "selection_uses_outcome_values": False,
            "all_declared_candidates_retained": True,
            "endpoints": _support_selection_public(support),
        },
        "lanes": lane_reports,
        "readout": {
            "routes_per_lane": list(ROUTES),
            "fresh_readout_per_route_and_outer_fold": True,
            "nested_train_only_tuning": True,
            "age_unpenalized": True,
            "penalty_grid": [float(item) for item in penalty_grid],
            "maximum_penalty_combinations": int(maximum_penalty_combinations),
            "selected_penalties_or_models_serialized": False,
            "private_tuning_audit": tuning_summary,
        },
        "label_policy": {
            "encoding": "explicit_numeric_0_1_on_observed_rows",
            "missing_response_policy": "missing_or_excluded_from_denominator",
            "official_test_refused": True,
        },
        "representation": {
            "same_fold_model_transforms_outer_train_and_test": True,
            "target_channels_erased_before_preprocessing_or_encoding": True,
            "full_context_direct_history_flag_erased": True,
            "de_novo_all_11_condition_history_flags_erased": True,
            "de_novo_labs_and_vitals_retained": True,
            "cross_fold_bases_pooled": False,
            "execution_path": "synthetic_or_explicit_flat_fixture_only",
        },
        "inference": {
            "method": "patient_level_centered_multiplier_max_statistic",
            "paired": True,
            "draws_serialized": False,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "organ_family_and_claim_proximity_aggregates": True,
        },
        "procedures": {
            "included": False,
            "separate_care_process_exposure_audit": True,
            "disease_endpoint_tower_count": 2,
        },
        "official_test_inputs_loaded": False,
        "patient_rows_or_identifiers_emitted": False,
        "outcome_values_accessed_locally": True,
        "outcome_values_serialized": False,
        "models_scored": True,
        "predictions_or_scores_loaded": False,
        "causal_language_authorized": False,
    }
    assert_aggregate_only_result(report)
    return report


def _run_actual_v6_2_core(
    *,
    root: Path,
    dataset_root: str | Path,
    clinical_project_root: str | Path,
    support: EligibleSupport,
    penalty_grid: Sequence[float],
    maximum_penalty_combinations: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
    protocol_sha256: str,
) -> Mapping[str, Any]:
    """Execute the production local V6.2 path with no patient output.

    This path intentionally does not coerce into ``EvaluationInput``.  The
    established AI-READI cohort/FeatureView and V6.2 fold representation are
    the production contracts; the flat numeric fixture seam above remains
    available only for focused synthetic tests.
    """

    from patient_atlas_disease_universal import (
        make_inner_fold_ids as make_frozen_inner_fold_ids,
        subset_targets,
    )

    context = _load_actual_v6_2_context(
        root=root,
        dataset_root=dataset_root,
        clinical_project_root=clinical_project_root,
        support=support,
    )
    feature_cohort = context["feature_cohort"]
    base_view = context["base_view"]
    outer_assignment = np.asarray(context["outer_assignment"], dtype=np.int64)
    labels_by_source = context["labels_by_source"]
    observed_by_source = context["observed_by_source"]
    validate_development_partitions(tuple(context["raw_cohort"].split_labels))
    validate_support_against_observed(
        support,
        labels_by_source,
        observed_by_source,
        outer_assignment,
    )
    metadata = frozen_candidate_metadata()

    # Build each inner assignment once from the original frozen disease target
    # matrix restricted to the corresponding outer-training set.  The same
    # assignment is then reused across endpoints and both representation lanes.
    fold_contexts: list[Mapping[str, Any]] = []
    for outer_fold in range(5):
        test_indices = np.flatnonzero(outer_assignment == outer_fold)
        train_indices = np.flatnonzero(outer_assignment != outer_fold)
        if not len(test_indices) or not len(train_indices):
            raise FoldBasisError("actual V6.2 outer fold has an empty partition")
        train_view = _subset_actual_feature_view(base_view, train_indices)
        test_view = _subset_actual_feature_view(base_view, test_indices)
        inner_targets = subset_targets(
            context["fold_targets"],
            train_indices,
            train_view.patient_ids,
        )
        inner_folds, inner_hash = make_frozen_inner_fold_ids(
            outer_fold=outer_fold,
            patient_ids=train_view.patient_ids,
            site_ids=train_view.site_ids,
            targets=inner_targets,
            outer_fold_policy=context["fold_policy"],
        )
        if np.any(np.asarray(inner_folds) < 0) or len(inner_folds) != len(train_indices):
            raise FoldBasisError("actual V6.2 inner assignment is malformed")
        if str(inner_hash) != EXACT_INNER_FOLD_ASSIGNMENT_SHA256[outer_fold]:
            raise ProtocolBindingError("actual V6.2 inner fold assignment hash differs")
        fold_contexts.append(
            MappingProxyType(
                {
                    "outer_fold": outer_fold,
                    "train_indices": train_indices,
                    "test_indices": test_indices,
                    "train_view": train_view,
                    "test_view": test_view,
                    "inner_folds": np.asarray(inner_folds, dtype=np.int64),
                    "inner_hash": str(inner_hash),
                    "outer_test_patient_set_sha256": _patient_json_hash(
                        tuple(base_view.patient_ids[index] for index in test_indices)
                    ),
                }
            )
        )

    # The canonical V6.2 representation is fitted at most once per lane/fold,
    # then reused for every endpoint and route.  No endpoint labels enter this
    # fit.  Full context intentionally permits other histories during fit;
    # target-specific direct channels are erased before each held-out
    # transform.  De novo masks all 11 histories during fit and transforms.
    fitted_by_lane = {
        lane: _fit_actual_v6_2_representation_folds(
            root=root,
            feature_cohort=feature_cohort,
            base_view=base_view,
            outer_assignment=outer_assignment,
            representation_protocol=context["representation_protocol"],
            lane=lane,
        )
        for lane in LANES
    }
    outer_fold_records: list[Mapping[str, Any]] = []
    for fold in fold_contexts:
        outer_fold = int(fold["outer_fold"])
        train_indices = np.asarray(fold["train_indices"], dtype=np.int64)
        test_indices = np.asarray(fold["test_indices"], dtype=np.int64)
        lane_records: dict[str, Any] = {}
        for lane in LANES:
            fitted = fitted_by_lane[lane][outer_fold][3]
            provenance = getattr(fitted, "provenance", None)
            preprocessor = getattr(fitted, "preprocessor", None)
            model_token = "" if provenance is None else str(provenance.model_token)
            preprocessor_sha256 = "" if preprocessor is None else str(
                preprocessor.bundle_sha256
            )
            if (
                len(model_token) != 64
                or len(preprocessor_sha256) != 64
                or any(value not in "0123456789abcdef" for value in model_token)
                or any(
                    value not in "0123456789abcdef"
                    for value in preprocessor_sha256
                )
            ):
                raise FoldBasisError("actual V6.2 fold provenance hashes are malformed")
            lane_records[lane] = {
                "model_state_sha256": model_token,
                "preprocessor_bundle_sha256": preprocessor_sha256,
            }
        outer_fold_records.append(
            {
                "outer_fold": outer_fold,
                "outer_train_count": int(len(train_indices)),
                "outer_test_count": int(len(test_indices)),
                "outer_test_patient_set_sha256": str(
                    fold["outer_test_patient_set_sha256"]
                ),
                "inner_assignment_sha256": str(fold["inner_hash"]),
                "representation_fold_state": lane_records,
            }
        )
    readout_config = DiseaseReadoutConfig(
        penalty_grid=tuple(float(item) for item in penalty_grid),
        probability_clip=1e-6,
        minimum_disclosable_cell_count=10,
    )
    lane_reports: dict[str, Any] = {}
    tuning_summary: dict[str, Any] = {}
    for lane in LANES:
        endpoint_losses: dict[str, Mapping[str, np.ndarray]] = {}
        endpoint_auc: dict[str, Mapping[str, float | None]] = {}
        endpoint_tuning: dict[str, Any] = {}
        fitted_folds = fitted_by_lane[lane]
        # Route designs depend on the fitted outer-fold model and the exact
        # target-erasure profile, but not on the endpoint labels or readout.
        # Cache those private arrays per fold/profile so the canonical encoder
        # is not repeatedly run for the 26 endpoints.  Readout tuning remains
        # fresh for every endpoint/route/fold below.
        route_design_cache: dict[
            int,
            dict[
                tuple[str, tuple[str, ...]],
                tuple[
                    Mapping[str, np.ndarray],
                    Mapping[str, np.ndarray],
                    Mapping[str, tuple[str, ...]],
                ],
            ],
        ] = {outer_fold: {} for outer_fold in range(5)}
        for source in support.eligible_sources:
            losses = {route: _new_loss_sink(len(base_view.patient_ids)) for route in ROUTES}
            auc_values: dict[str, list[tuple[float, int]]] = {route: [] for route in ROUTES}
            selected_penalties: dict[str, list[Mapping[str, float]]] = {
                route: [] for route in ROUTES
            }
            for outer_fold, train_indices, test_indices, fitted in fitted_folds:
                fold = fold_contexts[outer_fold]
                train_view = fold["train_view"]
                test_view = fold["test_view"]
                profile = _actual_exclusion_profile(
                    feature_cohort.feature_names,
                    target_source_code=source,
                    lane=lane,
                )
                cached_designs = route_design_cache[outer_fold].get(profile)
                if cached_designs is None:
                    safe_train = _mask_actual_feature_view(
                        train_view,
                        feature_names=feature_cohort.feature_names,
                        target_source_code=source,
                        lane=lane,
                    )
                    safe_test = _mask_actual_feature_view(
                        test_view,
                        feature_names=feature_cohort.feature_names,
                        target_source_code=source,
                        lane=lane,
                    )
                    _clear_actual_representation_caches(fitted)
                    cached_designs = _actual_route_designs(
                        fitted,
                        safe_train,
                        safe_test,
                        outer_fold=outer_fold,
                    )
                    route_design_cache[outer_fold][profile] = cached_designs
                train_designs, test_designs, route_groups = cached_designs
                train_labels = np.asarray(labels_by_source[source])[train_indices]
                test_labels = np.asarray(labels_by_source[source])[test_indices]
                train_observed = np.asarray(observed_by_source[source], dtype=bool)[train_indices]
                test_observed = np.asarray(observed_by_source[source], dtype=bool)[test_indices]
                for route in ROUTES:
                    result = nested_logistic_readout(
                        route=route,
                        x_train=train_designs[route],
                        y_train=train_labels,
                        train_eligible=train_observed,
                        inner_fold_ids=fold["inner_folds"],
                        x_test=test_designs[route],
                        y_test=test_labels,
                        test_eligible=test_observed,
                        feature_groups=route_groups[route],
                        penalty_grid=penalty_grid,
                        config=readout_config,
                        maximum_penalty_combinations=maximum_penalty_combinations,
                    )
                    losses[route][test_indices] = result.test_losses
                    selected_penalties[route].append(dict(result.selected_penalties))
                    if result.test_auc is not None:
                        auc_values[route].append(
                            (float(result.test_auc), int(result.test_eligible_count))
                        )
                for route in ROUTES:
                    if not np.array_equal(
                        np.isfinite(losses[route][test_indices]), test_observed
                    ):
                        raise FoldBasisError(
                            "actual V6.2 route loss eligibility differs from observed labels"
                        )
            endpoint_losses[source] = losses
            endpoint_auc[source] = {
                route: (
                    None
                    if not values
                    else float(
                        np.average(
                            [item[0] for item in values],
                            weights=[item[1] for item in values],
                        )
                    )
                )
                for route, values in auc_values.items()
            }
            endpoint_tuning[source] = {
                route: {
                    "fresh_outer_fold_readouts": len(selected_penalties[route]),
                    "inner_tuning_outer_test_labels_used": False,
                }
                for route in ROUTES
            }
        lane_reports[lane] = build_lane_aggregate_report(
            lane=lane,
            endpoint_losses=endpoint_losses,
            endpoint_metadata={source: metadata[source] for source in support.eligible_sources},
            support_by_source=support.support_by_source,
            auc_by_source=endpoint_auc,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
        tuning_summary[lane] = endpoint_tuning

    inner_hashes = [str(item["inner_hash"]) for item in fold_contexts]
    report: dict[str, Any] = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "status": "completed_aggregate_only",
        "evaluation_prefix": "BARAS_V6_2_EXPANDED_ENDPOINT_EVALUATION_V1",
        "target_prefix": "patient_atlas_v6_2_expanded_endpoint_evaluation",
        "execution_path": "canonical_local_v6_2_dataset_and_clinical_project_adapter",
        "protocol_sha256": protocol_sha256,
        "atlas_protocol_sha256": FROZEN_ATLAS_PROTOCOL_SHA256,
        "precommit_receipt_sha256": FROZEN_PRECOMMIT_RECEIPT_SHA256,
        "outer_fold_assignment_sha256": EXACT_OUTER_FOLD_HASH,
        "source_hashes": {
            str(name): str(digest)
            for name, digest in context["source_hashes"].items()
        },
        "outer_fold_count": 5,
        "candidate_count": len(EXPECTED_CANDIDATE_SOURCE_CODES),
        "candidate_source_codes": list(EXPECTED_CANDIDATE_SOURCE_CODES),
        "free_text_sources_excluded": ["mhoccur_cnsot", "mhoccur_cnrot"],
        "scope": {
            "patient_count": int(base_view.size),
            "declared_endpoint_count": len(EXPECTED_CANDIDATE_SOURCE_CODES),
            "scored_endpoint_count": len(support.eligible_sources),
            "withheld_endpoint_count": len(EXPECTED_WITHHELD_SOURCE_CODES),
            "support_class_counts": dict(EXPECTED_ELIGIBLE_BY_CLAIM_PROXIMITY_CLASS),
        },
        "outer_folds": outer_fold_records,
        "support_receipt": {
            "later_loaded": True,
            "frozen_file": FROZEN_SUPPORT_RECEIPT_NAME,
            "frozen_file_sha256": FROZEN_SUPPORT_RECEIPT_SHA256,
            "receipt_sha256": support.receipt_sha256,
            "eligible_endpoint_count": len(support.eligible_sources),
            "frozen_eligible_endpoint_count": len(EXPECTED_ELIGIBLE_SOURCE_CODES),
            "frozen_eligible_by_claim_proximity_class": dict(
                EXPECTED_ELIGIBLE_BY_CLAIM_PROXIMITY_CLASS
            ),
            "selection_uses_model_performance": False,
            "selection_uses_outcome_values": False,
            "all_declared_candidates_retained": True,
            "endpoints": _support_selection_public(support),
        },
        "lanes": lane_reports,
        "readout": {
            "routes_per_lane": list(ROUTES),
            "fresh_readout_per_route_and_outer_fold": True,
            "nested_train_only_tuning": True,
            "age_unpenalized": True,
            "penalty_grid": [float(item) for item in penalty_grid],
            "maximum_penalty_combinations": int(maximum_penalty_combinations),
            "selected_penalties_or_models_serialized": False,
            "private_tuning_audit": tuning_summary,
        },
        "representation": {
            "canonical_v6_2_factory": "patient_atlas_aireadi_v6_2_evaluation.AIReadIV62FoldRepresentationFactory",
            "representation_fits_per_outer_fold": 2,
            "representation_fits_total": 10,
            "representation_fit_reused_across_endpoints_and_routes": True,
            "route_designs_cached_by_lane_fold_exclusion_profile": True,
            "full_context_condition_histories_enabled_during_outer_train_fit": True,
            "full_context_claim_limit": "Outer-train representation fitting may observe mapped condition-history inputs; full_context is secondary/current-context evidence, not target-history-free screening.",
            "full_context_target_direct_channel_erased_before_each_transform": True,
            "de_novo_all_11_condition_history_flags_erased_before_fit_and_transform": True,
            "de_novo_labs_and_vitals_retained_except_frozen_prospective_policy_fields": True,
            "same_fold_model_transforms_outer_train_and_test": True,
            "target_channels_erased_before_encoding_transform": True,
            "cross_fold_bases_pooled": False,
            "inner_assignment_hashes": inner_hashes,
            "outer_fold_state_records": outer_fold_records,
            "screening_protocol_sha256": context["screening_protocol_sha256"],
            "fold_policy_sha256": context["fold_policy_sha256"],
        },
        "label_policy": {
            "encoding": "explicit_numeric_0_1_on_observed_rows",
            "missing_response_policy": "missing_or_excluded_from_denominator",
            "official_test_refused": True,
            "labels_loaded_locally": True,
            "labels_or_patient_rows_serialized": False,
        },
        "inference": {
            "method": "patient_level_centered_multiplier_max_statistic",
            "paired": True,
            "draws_serialized": False,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "draw_chunk_size": MULTIPLIER_DRAW_CHUNK_SIZE,
            "organ_family_and_claim_proximity_aggregates": True,
        },
        "procedures": {
            "included": False,
            "separate_care_process_exposure_audit": True,
            "disease_endpoint_tower_count": 2,
        },
        "official_test_inputs_loaded": False,
        "patient_rows_or_identifiers_emitted": False,
        "outcome_values_accessed_locally": True,
        "outcome_values_serialized": False,
        "models_scored": True,
        "predictions_or_scores_loaded": False,
        "causal_language_authorized": False,
    }
    assert_aggregate_only_result(report)
    return report


def run_v6_2_expanded_endpoint_evaluation(
    *,
    project_root: str | Path,
    output_path: str | Path,
    failure_path: str | Path,
    dataset_root: str | Path | None = None,
    clinical_project_root: str | Path | None = None,
    data_loader: Callable[[], EvaluationInput | Mapping[str, Any]] | None = None,
    evaluation_input: EvaluationInput | Mapping[str, Any] | None = None,
    support_receipt: str | Path | Mapping[str, Any] | None = None,
    representation_factory: Any | None = None,
    penalty_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    maximum_penalty_combinations: int = DEFAULT_MAXIMUM_PENALTY_COMBINATIONS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> Mapping[str, Any]:
    """Run one safe aggregate evaluation through a local-only path.

    Production calls provide both ``dataset_root`` and
    ``clinical_project_root`` and use the authenticated V6.2 cohort/fold,
    endpoint loader, representation factory, and nested readout path.  The
    ``data_loader``/``evaluation_input`` arguments are a deliberately flat,
    synthetic fixture seam used by focused tests only; they are not a
    production adapter.
    """

    root = Path(project_root).resolve()
    output = Path(output_path).resolve()
    failure = Path(failure_path).resolve()
    if output == failure:
        raise ValueError("success and failure paths must differ")
    if output.exists() or failure.exists():
        raise FileExistsError("evaluation output paths must be new")
    protocol_sha256: str | None = None
    protocol: Mapping[str, Any] | None = None
    try:
        protocol, protocol_sha256 = validate_protocol(root)
        if support_receipt is None:
            raise InputContractError("a later-loadable eligible-support receipt is required")
        if data_loader is not None and evaluation_input is not None:
            raise InputContractError("provide data_loader or evaluation_input, not both")
        production_roots_supplied = dataset_root is not None or clinical_project_root is not None
        if production_roots_supplied:
            if isinstance(support_receipt, Mapping):
                raise InputContractError(
                    "production V6.2 requires the authenticated path-backed support receipt"
                )
            canonical_support = (root / FROZEN_SUPPORT_RECEIPT_NAME).resolve()
            supplied_support = Path(support_receipt).resolve()
            if supplied_support != canonical_support:
                raise InputContractError(
                    "production V6.2 support receipt must be the canonical frozen artifact"
                )
        support = load_eligible_support_receipt(support_receipt, project_root=root)
        if production_roots_supplied:
            if (
                support.receipt_sha256 != FROZEN_SUPPORT_RECEIPT_SHA256
                or support.receipt.get("schema_version")
                != "baras-v6-2-expanded-endpoint-atlas-support-materializer-v1"
                or tuple(support.eligible_sources) != EXPECTED_ELIGIBLE_SOURCE_CODES
            ):
                raise EndpointSupportDriftError(
                    "production V6.2 support provenance differs from the frozen 26-endpoint artifact"
                )
            if dataset_root is None or clinical_project_root is None:
                raise InputContractError(
                    "production V6.2 path requires both dataset_root and clinical_project_root"
                )
            if data_loader is not None or evaluation_input is not None:
                raise InputContractError(
                    "production V6.2 roots cannot be combined with synthetic input"
                )
            if representation_factory is not None:
                raise InputContractError(
                    "production V6.2 path binds its canonical representation factory"
                )
            if protocol is None:
                raise ProtocolBindingError("authenticated evaluation protocol is unavailable")
            frozen_readout = protocol["readout"]
            frozen_inference = protocol["inference"]
            if (
                tuple(float(item) for item in penalty_grid)
                != tuple(float(item) for item in frozen_readout["penalty_grid"])
                or int(maximum_penalty_combinations)
                != int(frozen_readout["maximum_penalty_combinations"])
                or int(bootstrap_samples)
                != int(frozen_inference["bootstrap_samples"])
                or int(bootstrap_seed) != int(frozen_inference["bootstrap_seed"])
                or float(confidence_level) != float(frozen_inference["confidence_level"])
            ):
                raise InputContractError(
                    "production V6.2 scientific settings differ from the authenticated protocol"
                )
            report = _run_actual_v6_2_core(
                root=root,
                dataset_root=dataset_root,
                clinical_project_root=clinical_project_root,
                support=support,
                penalty_grid=penalty_grid,
                maximum_penalty_combinations=maximum_penalty_combinations,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                confidence_level=confidence_level,
                protocol_sha256=protocol_sha256,
            )
        else:
            if evaluation_input is not None:
                raw = evaluation_input
            elif data_loader is not None:
                raw = data_loader()
            else:
                raise InputContractError(
                    "no local synthetic evaluation input or production V6.2 roots were supplied"
                )
            # Check partition metadata before coercion, which is where label
            # arrays are validated/accessed.  A well-behaved adapter should
            # perform the same check before opening its label source.
            if isinstance(raw, Mapping) and "participant_splits" in raw:
                validate_development_partitions(raw["participant_splits"])
                if raw.get("outer_fold_hash") != EXACT_OUTER_FOLD_HASH:
                    raise InputContractError(
                        "local input must bind the exact frozen outer-fold hash"
                    )
            elif isinstance(raw, EvaluationInput):
                validate_development_partitions(raw.participant_splits)
            data = _coerce_evaluation_input(raw)
            factory = representation_factory or StandardizingRepresentationFactory()
            config = DiseaseReadoutConfig(
                penalty_grid=tuple(float(item) for item in penalty_grid),
                probability_clip=1e-6,
                minimum_disclosable_cell_count=10,
            )
            report = _run_core(
                root=root,
                data=data,
                support=support,
                representation_factory=factory,
                penalty_grid=penalty_grid,
                readout_config=config,
                maximum_penalty_combinations=maximum_penalty_combinations,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                confidence_level=confidence_level,
                protocol_sha256=protocol_sha256,
            )
        output.write_text(json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        return report
    except Exception as error:
        failure_payload = build_failure_payload(error, protocol_sha256=protocol_sha256)
        failure.write_text(json.dumps(failure_payload, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local-only V6.2 expanded endpoint evaluation")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--support-receipt", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="authenticated local dataset root for the production V6.2 path",
    )
    parser.add_argument(
        "--clinical-project-root",
        type=Path,
        help="local clinical project root for the production V6.2 path",
    )
    parser.add_argument(
        "--notice",
        action="store_true",
        help="print the local-adapter requirement and exit without launching a run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.notice:
        print(
            "Production path requires --dataset-root and --clinical-project-root; "
            "flat inputs are synthetic-test only."
        )
        return 0
    if args.dataset_root is None or args.clinical_project_root is None:
        raise SystemExit(
            "--dataset-root and --clinical-project-root are required outside --notice"
        )
    try:
        run_v6_2_expanded_endpoint_evaluation(
            project_root=args.project_root,
            output_path=args.output,
            failure_path=args.failure,
            dataset_root=args.dataset_root,
            clinical_project_root=args.clinical_project_root,
            support_receipt=args.support_receipt,
        )
    except Exception as error:
        print(f"failed_closed: {type(error).__name__}")
        return 1
    return 0


__all__ = [
    "StandardizingFoldModel",
    "StandardizingRepresentationFactory",
    "run_v6_2_expanded_endpoint_evaluation",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
