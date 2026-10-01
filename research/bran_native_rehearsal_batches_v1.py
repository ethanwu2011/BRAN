"""Pure, source/person/episode-balanced native CBC rehearsal batches.

This module consumes already-adapted private arrays.  It performs no source
I/O or authentication, fits no normalization, uses no torch, and carries no
retinal data or labels.  The resulting masks are task eligibility gates for a
future learner, not a model fit or a guarantee that more data improves
performance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from bran_external_native_age_v1 import eligible_native_age
from bran_joint_lab_cache_v1 import FIELDS
from bran_joint_lab_pretraining_v1 import (
    HierarchicalEpisodeSampler,
    SourceEpisodePool,
    _validate_source_arrays,
)
from bran_joint_lab_task_contract_v1 import (
    CBC_WIDTH,
    JOINT_WIDTH,
    REGISTRY_WIDTH,
    build_task_masks,
    masked_registry_inputs,
    project_joint_labs_to_registry,
)


_SOURCES = frozenset({"mimic", "nhanes", "eicu"})
_MODES = frozenset({"partial_cbc", "whole_cbc"})
_SOURCE_FIELDS = frozenset(
    {
        "values",
        "observed",
        "provenance",
        "person_group",
        "split",
        "adult_qualified",
        "age_triplet",
        "age_kind",
        "scalar_age",
        "native_age_eligible",
        "partial_cbc_eligible",
        "whole_cbc_eligible",
    }
)


def _readonly(array: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    copied = np.array(array, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


def _validate_joint_labs(
    values: object, observed: object, provenance: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 2
        or values.shape[1] != JOINT_WIDTH
        or values.dtype.kind != "f"
        or not isinstance(observed, np.ndarray)
        or observed.shape != values.shape
        or observed.dtype != np.dtype(bool)
        or not isinstance(provenance, np.ndarray)
        or provenance.shape != values.shape
        or provenance.dtype.kind not in "iu"
        or values.shape[0] <= 0
    ):
        raise ValueError("native source laboratory arrays are invalid")
    if (
        (provenance.dtype.kind == "u" and np.any(provenance > 1))
        or (provenance.dtype.kind == "i" and np.any((provenance < 0) | (provenance > 1)))
        or not np.array_equal(provenance, observed.astype(provenance.dtype))
    ):
        raise ValueError("native source laboratory provenance is invalid")
    if np.any(~observed & ~np.isnan(values)):
        raise ValueError("missing native source values must be NaN")
    if np.any(observed & ~np.isfinite(values)):
        raise ValueError("observed native source values must be finite")
    if np.any(observed[:, :CBC_WIDTH] & (values[:, :CBC_WIDTH] <= 0.0)):
        raise ValueError("observed native CBC values must be positive")
    if np.any(observed[:, CBC_WIDTH:] & (values[:, CBC_WIDTH:] < 0.0)):
        raise ValueError("observed native chemistry values must be nonnegative")
    return (
        np.array(values, dtype=np.float64, copy=True),
        np.array(observed, dtype=np.bool_, copy=True),
        np.array(provenance, dtype=np.uint8, copy=True),
    )


def _validate_source_output(source: str, item: Mapping[str, object]) -> dict[str, np.ndarray]:
    if not isinstance(item, Mapping) or not _SOURCE_FIELDS.issubset(item):
        raise ValueError("native source adapter output is incomplete")

    values, observed, provenance = _validate_joint_labs(
        item["values"], item["observed"], item["provenance"]
    )
    n = values.shape[0]
    if source == "eicu":
        if (
            np.any(observed[:, CBC_WIDTH:])
            or np.any(provenance[:, CBC_WIDTH:])
            or np.any(~np.isnan(values[:, CBC_WIDTH:]))
        ):
            raise ValueError("eICU chemistry must remain unavailable")

    # Reuse the existing six-key source contract for group/split validation.
    _, _, _, groups, split = _validate_source_arrays(
        source,
        {
            "values": values,
            "observed": observed,
            "provenance": provenance,
            "person_group": item["person_group"],
            "split": item["split"],
            "adult_qualified": item["adult_qualified"],
        },
    )
    adult = item["adult_qualified"]
    if not isinstance(adult, np.ndarray) or adult.shape != (n,) or adult.dtype != np.dtype(bool):
        raise ValueError("native source adult mask is invalid")
    triplet = item["age_triplet"]
    kind = item["age_kind"]
    expected_age, expected_native = eligible_native_age(triplet, kind, adult)
    if (
        not isinstance(item["age_triplet"], np.ndarray)
        or item["age_triplet"].shape != (n, 3)
        or not isinstance(item["age_kind"], np.ndarray)
        or item["age_kind"].shape != (n,)
    ):
        raise ValueError("native source age arrays have invalid row count")
    native = item["native_age_eligible"]
    if not isinstance(native, np.ndarray) or native.shape != (n,) or native.dtype != np.dtype(bool):
        raise ValueError("native age eligibility mask is invalid")
    if not np.array_equal(native, expected_native):
        raise ValueError("native age eligibility disagrees with the age gate")
    scalar_age = item["scalar_age"]
    if (
        not isinstance(scalar_age, np.ndarray)
        or scalar_age.shape != (n,)
        or scalar_age.dtype.kind != "f"
        or np.any(np.isinf(scalar_age))
        or not np.array_equal(
            np.isnan(scalar_age), np.isnan(expected_age)
        )
        or not np.array_equal(
            np.where(np.isfinite(expected_age), scalar_age, 0.0),
            np.where(np.isfinite(expected_age), expected_age, 0.0),
        )
    ):
        raise ValueError("native scalar age disagrees with the age gate")

    cbc_count = observed[:, :CBC_WIDTH].sum(axis=1)
    expected_partial = (split == 0) & expected_native & (cbc_count >= 2)
    expected_whole = expected_partial & observed[:, CBC_WIDTH:].any(axis=1)
    partial = item["partial_cbc_eligible"]
    whole = item["whole_cbc_eligible"]
    if (
        not isinstance(partial, np.ndarray)
        or partial.shape != (n,)
        or partial.dtype != np.dtype(bool)
        or not isinstance(whole, np.ndarray)
        or whole.shape != (n,)
        or whole.dtype != np.dtype(bool)
        or not np.array_equal(partial, expected_partial)
        or not np.array_equal(whole, expected_whole)
    ):
        raise ValueError("native task eligibility disagrees with source arrays")

    return {
        "values": values,
        "observed": observed,
        "provenance": provenance,
        "person_group": groups,
        "split": split,
        "adult_qualified": np.array(adult, dtype=np.bool_, copy=True),
        "age_triplet": np.array(triplet, dtype=np.float64, copy=True),
        "age_kind": np.array(kind, dtype=np.uint8, copy=True),
        "scalar_age": np.array(expected_age, dtype=np.float64, copy=True),
        "native_age_eligible": np.array(expected_native, dtype=np.bool_, copy=True),
        "partial_cbc_eligible": np.array(expected_partial, dtype=np.bool_, copy=True),
        "whole_cbc_eligible": np.array(expected_whole, dtype=np.bool_, copy=True),
    }


def _validate_registry_inputs(
    registry_fields: tuple[str, ...], medians: np.ndarray, iqrs: np.ndarray
) -> tuple[int, ...]:
    if (
        not isinstance(registry_fields, tuple)
        or len(registry_fields) != REGISTRY_WIDTH
        or any(not isinstance(field, str) for field in registry_fields)
        or len(set(registry_fields)) != REGISTRY_WIDTH
        or any(field not in registry_fields for field in FIELDS)
    ):
        raise ValueError("registry schema is invalid")
    if any(registry_fields.index(field) >= 48 for field in FIELDS):
        raise ValueError("joint laboratory fields must occupy registry slots below 48")
    for scale, label, positive in (
        (medians, "medians", False),
        (iqrs, "iqrs", True),
    ):
        if (
            not isinstance(scale, np.ndarray)
            or scale.shape != (REGISTRY_WIDTH,)
            or scale.dtype.kind != "f"
            or not np.isfinite(scale).all()
            or (positive and np.any(scale <= 0.0))
        ):
            raise ValueError(label + " are invalid")
    return tuple(registry_fields.index(field) for field in FIELDS)


def _validate_age_scale(age_mean: object, age_scale: object) -> tuple[float, float]:
    if (
        isinstance(age_mean, bool)
        or not isinstance(age_mean, Real)
        or isinstance(age_scale, bool)
        or not isinstance(age_scale, Real)
    ):
        raise ValueError("age normalization parameters are invalid")
    mean = float(age_mean)
    scale = float(age_scale)
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("age normalization parameters are invalid")
    return mean, scale


@dataclass(frozen=True, repr=False)
class _Mode:
    sampler: HierarchicalEpisodeSampler
    projected_values: tuple[np.ndarray, ...]
    projected_masks: tuple[np.ndarray, ...]
    observed: tuple[np.ndarray, ...]
    normalized_age: tuple[np.ndarray, ...]
    registry_slots: tuple[int, ...]
    registry_fields: tuple[str, ...]


@dataclass(frozen=True, repr=False)
class _Prepared:
    whole: _Mode
    partial: _Mode


def _make_mode(
    mode: str,
    sources: tuple[tuple[str, dict[str, np.ndarray]], ...],
    registry_fields: tuple[str, ...],
    medians: np.ndarray,
    iqrs: np.ndarray,
    age_mean: float,
    age_scale: float,
) -> _Mode:
    pools: list[SourceEpisodePool] = []
    projected_values: list[np.ndarray] = []
    projected_masks: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    normalized_age: list[np.ndarray] = []
    registry_slots = tuple(registry_fields.index(field) for field in FIELDS)

    for source, item in sources:
        indices = np.flatnonzero(
            item["whole_cbc_eligible"]
            if mode == "whole"
            else item["partial_cbc_eligible"]
        )
        if indices.size == 0:
            continue
        source_values, source_mask = project_joint_labs_to_registry(
            item["values"][indices],
            item["observed"][indices],
            item["provenance"][indices],
            registry_fields,
            medians,
            iqrs,
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            age = (item["scalar_age"][indices] - age_mean) / age_scale
        if not np.isfinite(age).all():
            raise ValueError("age normalization produced nonfinite values")
        pools.append(
            SourceEpisodePool(
                source,
                _readonly(item["person_group"][indices], np.int64),
                _readonly(item["split"][indices], np.int64),
                _readonly(item["adult_qualified"][indices], np.bool_),
                _readonly(item["observed"][indices, :CBC_WIDTH], np.bool_),
            )
        )
        projected_values.append(_readonly(source_values, np.float64))
        projected_masks.append(_readonly(source_mask, np.bool_))
        observed.append(_readonly(item["observed"][indices], np.bool_))
        normalized_age.append(_readonly(age, np.float64))

    if not pools:
        raise ValueError("native rehearsal mode has no eligible source episodes")
    return _Mode(
        HierarchicalEpisodeSampler(pools),
        tuple(projected_values),
        tuple(projected_masks),
        tuple(observed),
        tuple(normalized_age),
        registry_slots,
        registry_fields,
    )


def prepare_sources(
    sources: Mapping[str, Mapping[str, np.ndarray]],
    registry_fields: tuple[str, ...],
    medians: np.ndarray,
    iqrs: np.ndarray,
    age_mean: float,
    age_scale: float,
) -> _Prepared:
    """Prepare opaque whole/partial native-CBC rehearsal pools.

    ``sources`` must contain outputs of the pure native source adapter.  Each
    mode samples uniformly by source, then source-local person, then eligible
    episode.  Normalization parameters and age scaling are caller-supplied;
    no statistics are fitted here.
    """

    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("native source mapping is invalid")
    if any(not isinstance(source, str) or source not in _SOURCES for source in sources):
        raise ValueError("native source mapping is invalid")
    validated_sources: list[tuple[str, dict[str, np.ndarray]]] = []
    for source, item in sources.items():
        validated_sources.append((source, _validate_source_output(source, item)))
    slots = _validate_registry_inputs(registry_fields, medians, iqrs)
    mean, scale = _validate_age_scale(age_mean, age_scale)
    del slots  # _make_mode recomputes the immutable field-to-slot mapping.
    source_tuple = tuple(validated_sources)
    return _Prepared(
        whole=_make_mode(
            "whole", source_tuple, registry_fields, medians, iqrs, mean, scale
        ),
        partial=_make_mode(
            "partial", source_tuple, registry_fields, medians, iqrs, mean, scale
        ),
    )


def sample_batch(
    prepared: _Prepared, mode: str, batch_size: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Sample one balanced native-CBC batch with hidden CBC targets removed."""

    if not isinstance(prepared, _Prepared):
        raise ValueError("prepared rehearsal pools are invalid")
    if not isinstance(mode, str) or mode not in _MODES:
        raise ValueError("rehearsal mode is invalid")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
        raise ValueError("batch size is invalid")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng is invalid")
    mode_data = prepared.whole if mode == "whole_cbc" else prepared.partial
    size = int(batch_size)

    batch_observed = np.empty((size, JOINT_WIDTH), dtype=np.bool_)
    batch_age = np.empty(size, dtype=np.float64)
    sampled_episodes = []

    for row in range(size):
        sampled = mode_data.sampler.sample(rng)
        source_index = sampled.source_index
        episode_index = sampled.episode_index
        sampled_episodes.append(sampled)
        batch_observed[row] = mode_data.observed[source_index][episode_index]
        batch_age[row] = mode_data.normalized_age[source_index][episode_index]

    projected_values = np.stack(
        [
            mode_data.projected_values[sampled.source_index][sampled.episode_index]
            for sampled in sampled_episodes
        ],
        axis=0,
    ).astype(np.float64, copy=True)
    projected_mask = np.stack(
        [
            mode_data.projected_masks[sampled.source_index][sampled.episode_index]
            for sampled in sampled_episodes
        ],
        axis=0,
    ).astype(np.bool_, copy=True)

    visible, hidden = build_task_masks(batch_observed, mode, rng)
    clinical, clinical_mask = masked_registry_inputs(
        projected_values, projected_mask, visible, hidden, mode_data.registry_fields
    )
    target_mask = np.array(hidden[:, :CBC_WIDTH], dtype=np.bool_, copy=True)
    target_cbc = np.zeros((size, CBC_WIDTH), dtype=np.float64)
    projected_cbc = projected_values[:, mode_data.registry_slots[:CBC_WIDTH]]
    target_cbc[target_mask] = projected_cbc[target_mask]

    return {
        "clinical": _readonly(clinical, np.float64),
        "clinical_mask": _readonly(clinical_mask, np.bool_),
        "age": _readonly(batch_age, np.float64),
        "target_cbc": _readonly(target_cbc, np.float64),
        "target_mask": _readonly(target_mask, np.bool_),
    }


__all__ = ["prepare_sources", "sample_batch"]
