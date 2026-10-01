"""Aggregate-only coverage preflight for prespecified BRAN CGM/ECG metadata.

This module deliberately accepts only local identifiers, fold assignments, and boolean
``is finite`` metadata masks.  It never receives measurement values, applies no
clinical thresholds, and returns neither identifiers nor masks.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

try:  # NumPy is optional at import time; its scalar bool is accepted when available.
    from numpy import bool_ as _numpy_bool
except ImportError:  # pragma: no cover - covered by environments without NumPy.
    _numpy_bool = None


CELL_FLOOR = 10
MODALITIES = ("cgm", "ecg")
FIELDS = {
    "cgm": ("mean_glucose", "record_count", "duration_days"),
    "ecg": ("rate", "pr", "qrsd", "qt", "qtc"),
}

# These are the complete set of statuses this function can emit.
STATUS_SUPPORTED = "supported"
STATUS_SUPPRESSED_PANEL_SMALL_CELL = "suppressed_panel_small_cell"
STATUS_SUPPRESSED_FIELD_SMALL_CELL = "suppressed_field_small_cell"
SAFE_STATUSES = frozenset(
    {
        STATUS_SUPPORTED,
        STATUS_SUPPRESSED_PANEL_SMALL_CELL,
        STATUS_SUPPRESSED_FIELD_SMALL_CELL,
    }
)


def summarize_coverage(
    cohort_ids: Any,
    outer: Any,
    source_ids: Mapping[str, Any],
    finite_measurement_masks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize prespecified source coverage without returning row-level material.

    ``outer`` must assign every canonical cohort member to each of folds 0 through 4.
    Source IDs not in the cohort are validated for format and duplicate-free identity,
    but ignored for all counting.  A modality is suppressed when any nonzero matched or
    unmatched fold cell is below ``CELL_FLOOR``.  Otherwise, a field is independently
    suppressed when any nonzero canonical observed/missing *or matched-source
    observed/missing* fold cell is below that floor.  The matched-source cells are a
    disclosure gate only and are never emitted.

    The returned dictionary has a fixed schema: ``cell_floor`` and ``modalities``;
    modality/field keys are always canonical.  Suppressed counts are ``None`` so that
    complements cannot reconstruct a small cell.
    """
    canonical_ids = _validate_ids(cohort_ids, "cohort_ids")
    fold_assignments = _validate_outer(outer, len(canonical_ids))
    _validate_top_level_inputs(source_ids, finite_measurement_masks)

    cohort_index = {identifier: index for index, identifier in enumerate(canonical_ids)}
    result_modalities: dict[str, Any] = {}

    for modality in MODALITIES:
        modality_source_ids = _validate_ids(source_ids[modality], f"source_ids[{modality!r}]")
        masks = finite_measurement_masks[modality]
        _validate_field_mapping(masks, modality)
        validated_masks = {
            field: _validate_boolean_mask(
                masks[field], len(modality_source_ids), f"finite_measurement_masks[{modality!r}][{field!r}]"
            )
            for field in FIELDS[modality]
        }

        # Extras are intentionally not counted.  Duplicate source IDs fail before this
        # selection, so each canonical cohort member has at most one source position.
        source_position_by_cohort_index = {
            cohort_index[identifier]: source_index
            for source_index, identifier in enumerate(modality_source_ids)
            if identifier in cohort_index
        }
        coverage_counts = _coverage_counts(
            len(canonical_ids), fold_assignments, source_position_by_cohort_index
        )

        if _has_nonzero_small_cell(coverage_counts["folds"]):
            result_modalities[modality] = _suppressed_panel(modality)
            continue

        field_results: dict[str, Any] = {}
        for field in FIELDS[modality]:
            observed_by_cohort_index = {
                cohort_index: validated_masks[field][source_index]
                for cohort_index, source_index in source_position_by_cohort_index.items()
            }
            field_counts = _canonical_field_counts(
                len(canonical_ids), fold_assignments, observed_by_cohort_index
            )
            matched_field_counts = _matched_field_counts(
                fold_assignments, observed_by_cohort_index
            )
            # Canonical missingness can include unmatched people.  Checking only that
            # cell would permit a small matched-source missing complement to be derived
            # from published modality coverage and field observed counts.
            if _has_nonzero_small_cell(field_counts["folds"]) or _has_nonzero_small_cell(
                matched_field_counts
            ):
                field_results[field] = {
                    "status": STATUS_SUPPRESSED_FIELD_SMALL_CELL,
                    "counts": None,
                }
            else:
                field_results[field] = {"status": STATUS_SUPPORTED, "counts": field_counts}

        result_modalities[modality] = {
            "status": STATUS_SUPPORTED,
            "coverage": coverage_counts,
            "fields": field_results,
        }

    return {"cell_floor": CELL_FLOOR, "modalities": result_modalities}


def _validate_top_level_inputs(source_ids: Any, masks: Any) -> None:
    if not isinstance(source_ids, Mapping):
        raise TypeError("source_ids must be a mapping")
    if not isinstance(masks, Mapping):
        raise TypeError("finite_measurement_masks must be a mapping")
    _require_exact_keys(source_ids, MODALITIES, "source_ids")
    _require_exact_keys(masks, MODALITIES, "finite_measurement_masks")


def _validate_field_mapping(masks: Any, modality: str) -> None:
    if not isinstance(masks, Mapping):
        raise TypeError(f"finite_measurement_masks[{modality!r}] must be a mapping")
    _require_exact_keys(masks, FIELDS[modality], f"finite_measurement_masks[{modality!r}]")


def _require_exact_keys(mapping: Mapping[Any, Any], allowed: tuple[str, ...], name: str) -> None:
    actual = set(mapping.keys())
    expected = set(allowed)
    if actual != expected:
        unknown = actual - expected
        missing = expected - actual
        if unknown:
            raise ValueError(f"{name} contains unknown keys")
        raise ValueError(f"{name} is missing required keys")


def _validate_ids(values: Any, name: str) -> tuple[str, ...]:
    sequence = _as_flat_sequence(values, name)
    validated: list[str] = []
    seen: set[str] = set()
    for value in sequence:
        if not isinstance(value, str) or not value:
            raise TypeError(f"{name} must contain nonempty string IDs")
        if value in seen:
            raise ValueError(f"{name} contains duplicate IDs")
        seen.add(value)
        validated.append(value)
    return tuple(validated)


def _validate_outer(values: Any, expected_length: int) -> tuple[int, ...]:
    sequence = _as_flat_sequence(values, "outer")
    if len(sequence) != expected_length:
        raise ValueError("outer length must match cohort_ids")
    folds: list[int] = []
    for value in sequence:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("outer must contain integer fold assignments")
        fold = int(value)
        if fold not in range(5):
            raise ValueError("outer fold assignments must be in 0..4")
        folds.append(fold)
    if set(folds) != set(range(5)):
        raise ValueError("outer must contain each of the exact five folds 0..4")
    return tuple(folds)


def _validate_boolean_mask(values: Any, expected_length: int, name: str) -> tuple[bool, ...]:
    sequence = _as_flat_sequence(values, name)
    if len(sequence) != expected_length:
        raise ValueError(f"{name} length must match its modality source IDs")
    validated: list[bool] = []
    for value in sequence:
        if not _is_boolean(value):
            raise TypeError(f"{name} must contain boolean values")
        validated.append(bool(value))
    return tuple(validated)


def _as_flat_sequence(values: Any, name: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a one-dimensional sequence, not a string")
    if isinstance(values, (Mapping, set, frozenset)):
        raise TypeError(f"{name} must be an ordered one-dimensional sequence")
    try:
        sequence = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be a one-dimensional sequence") from error
    if any(isinstance(value, (list, tuple, dict, set)) for value in sequence):
        raise TypeError(f"{name} must be one-dimensional")
    return sequence


def _is_boolean(value: Any) -> bool:
    # Exact scalar types prevent a lookalike class called ``bool_`` from being accepted.
    return type(value) is bool or (_numpy_bool is not None and type(value) is _numpy_bool)


def _coverage_counts(
    cohort_size: int,
    folds: tuple[int, ...],
    source_position_by_cohort_index: Mapping[int, int],
) -> dict[str, Any]:
    overall = {"matched": len(source_position_by_cohort_index), "unmatched": 0}
    overall["unmatched"] = cohort_size - overall["matched"]
    per_fold = [{"matched": 0, "unmatched": 0} for _ in range(5)]
    for cohort_index, fold in enumerate(folds):
        key = "matched" if cohort_index in source_position_by_cohort_index else "unmatched"
        per_fold[fold][key] += 1
    return {"overall": overall, "folds": tuple(per_fold)}


def _canonical_field_counts(
    cohort_size: int,
    folds: tuple[int, ...],
    observed_by_cohort_index: Mapping[int, bool],
) -> dict[str, Any]:
    overall = {"observed": sum(observed_by_cohort_index.values()), "missing": 0}
    overall["missing"] = cohort_size - overall["observed"]
    per_fold = [{"observed": 0, "missing": 0} for _ in range(5)]
    for cohort_index, fold in enumerate(folds):
        key = "observed" if observed_by_cohort_index.get(cohort_index, False) else "missing"
        per_fold[fold][key] += 1
    return {"overall": overall, "folds": tuple(per_fold)}


def _matched_field_counts(
    folds: tuple[int, ...], observed_by_cohort_index: Mapping[int, bool]
) -> tuple[dict[str, int], ...]:
    """Return internal matched-source cells used only for disclosure gating."""
    per_fold = [{"observed": 0, "missing": 0} for _ in range(5)]
    for cohort_index, observed in observed_by_cohort_index.items():
        key = "observed" if observed else "missing"
        per_fold[folds[cohort_index]][key] += 1
    return tuple(per_fold)


def _has_nonzero_small_cell(per_fold_counts: tuple[dict[str, int], ...]) -> bool:
    return any(0 < count < CELL_FLOOR for cells in per_fold_counts for count in cells.values())


def _suppressed_panel(modality: str) -> dict[str, Any]:
    return {
        "status": STATUS_SUPPRESSED_PANEL_SMALL_CELL,
        "coverage": None,
        "fields": {
            field: {"status": STATUS_SUPPRESSED_PANEL_SMALL_CELL, "counts": None}
            for field in FIELDS[modality]
        },
    }
