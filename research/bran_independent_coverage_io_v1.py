"""Strict local manifest adapter and aggregate-only coverage-output validator.

This module is intentionally narrow: ``parse_manifest`` returns only canonical
source identifiers and finite-value masks, and ``validate_coverage`` either returns
an already aggregate-only coverage summary unchanged or rejects it with ``None``.
Neither function returns measurements, free text, paths, or participant rows.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from typing import Any


CELL_FLOOR = 10
MAX_SOURCE_ROWS = 10_000
MAX_HEADER_COLUMNS = 64
MAX_LINE_LENGTH = 1_048_576

MODALITY_FILES = {
    "cgm": "wearable_blood_glucose/manifest.tsv",
    "ecg": "cardiac_ecg/manifest.tsv",
}
SOURCE_COLUMNS = {
    "cgm": {
        "mean_glucose": "average_glucose_level_mg_dl",
        "record_count": "glucose_level_record_count",
        "duration_days": "glucose_sensor_sampling_duration_days",
    },
    "ecg": {"rate": "Rate", "pr": "PR", "qrsd": "QRSD", "qt": "QT", "qtc": "QTc"},
}
MODALITIES = tuple(MODALITY_FILES)
STATUS_SUPPORTED = "supported"
STATUS_SUPPRESSED_PANEL_SMALL_CELL = "suppressed_panel_small_cell"
STATUS_SUPPRESSED_FIELD_SMALL_CELL = "suppressed_field_small_cell"


def parse_manifest(handle: Any, modality: str) -> tuple[tuple[str, ...], dict[str, tuple[bool, ...]]]:
    """Parse a supplied TSV text handle into IDs and finite-number masks only.

    Required headers must appear exactly once.  Extra *named* columns are ignored;
    they are never copied into the return value.  The returned identifiers and masks
    are strictly local caller state and must never be sent to a hosted service or
    serialized.  Any malformed identifier causes the complete parse to fail,
    including one for a person outside the canonical cohort.
    """
    if modality not in MODALITY_FILES:
        raise ValueError("unsupported modality")
    if isinstance(handle, (str, bytes)) or not hasattr(handle, "__iter__"):
        raise TypeError("handle must be an iterable text handle")

    fields = SOURCE_COLUMNS[modality]
    required = ("person_id", *fields.values())
    try:
        reader = csv.DictReader(_bounded_text_lines(handle), delimiter="\t", strict=True)
        header = reader.fieldnames
        if header is None or not header or len(header) > MAX_HEADER_COLUMNS:
            raise ValueError("invalid manifest header")
        if any(header.count(column) != 1 for column in required):
            raise ValueError("manifest required columns must occur exactly once")

        source_ids: list[str] = []
        seen: set[str] = set()
        masks: dict[str, list[bool]] = {field: [] for field in fields}
        for row_count, row in enumerate(reader, start=1):
            if row_count > MAX_SOURCE_ROWS:
                raise ValueError("manifest exceeds source row limit")
            # Values beyond the named header are structurally malformed, not ignored
            # source columns.  Named extra headers remain safely ignored.
            if None in row:
                raise ValueError("manifest row has too many columns")
            identifier = _canonical_id(row.get("person_id"))
            if identifier in seen:
                raise ValueError("manifest contains duplicate IDs")
            seen.add(identifier)
            source_ids.append(identifier)
            for field, column in fields.items():
                masks[field].append(_is_finite_numeric(row.get(column)))
    except (csv.Error, UnicodeError) as error:
        raise ValueError("malformed manifest") from error

    return tuple(source_ids), {field: tuple(values) for field, values in masks.items()}


def validate_coverage(result: Any, cohort_size: Any, fold_sizes: Any) -> Any | None:
    """Validate the closed aggregate schema emitted by ``summarize_coverage``.

    This is a disclosure/schema check, not a claim about clinical validity.  A valid
    summary is returned unchanged; invalid inputs return ``None`` and never expose a
    partial reconstruction.
    """
    if not _is_nonnegative_int(cohort_size):
        return None
    if not isinstance(fold_sizes, (tuple, list)) or len(fold_sizes) != 5:
        return None
    if any(not _is_nonnegative_int(size) for size in fold_sizes) or sum(fold_sizes) != cohort_size:
        return None
    if not _exact_keys(result, ("cell_floor", "modalities")):
        return None
    if result["cell_floor"] != CELL_FLOOR or type(result["cell_floor"]) is not int:
        return None
    modalities = result["modalities"]
    if not _exact_keys(modalities, MODALITIES):
        return None

    for modality in MODALITIES:
        if not _validate_panel(modalities[modality], modality, cohort_size, fold_sizes):
            return None
    return result


def _bounded_text_lines(handle: Any):
    for line in handle:
        if not isinstance(line, str):
            raise TypeError("handle must yield text lines")
        if len(line) > MAX_LINE_LENGTH:
            raise ValueError("manifest line exceeds length limit")
        yield line


def _canonical_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("manifest has a missing ID")
    identifier = value.strip()
    if not identifier or not identifier.isascii() or not identifier.isdecimal():
        raise ValueError("manifest ID must be an ASCII decimal string")
    if len(identifier) > 1 and identifier[0] == "0":
        raise ValueError("manifest ID has a leading zero")
    return identifier


def _is_finite_numeric(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return math.isfinite(float(value.strip()))
    except (TypeError, ValueError, OverflowError):
        return False


def _exact_keys(value: Any, expected: tuple[str, ...]) -> bool:
    return isinstance(value, dict) and set(value) == set(expected) and len(value) == len(expected)


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _valid_safe_cell(value: Any) -> bool:
    return _is_nonnegative_int(value) and (value == 0 or value >= CELL_FLOOR)


def _validate_panel(panel: Any, modality: str, cohort_size: int, fold_sizes: tuple[int, ...] | list[int]) -> bool:
    if not _exact_keys(panel, ("status", "coverage", "fields")):
        return False
    if not _exact_keys(panel["fields"], tuple(SOURCE_COLUMNS[modality])):
        return False
    status = panel["status"]
    if status == STATUS_SUPPRESSED_PANEL_SMALL_CELL:
        return panel["coverage"] is None and all(
            _exact_keys(field, ("status", "counts"))
            and field["status"] == STATUS_SUPPRESSED_PANEL_SMALL_CELL
            and field["counts"] is None
            for field in panel["fields"].values()
        )
    if status != STATUS_SUPPORTED or not _validate_coverage_counts(panel["coverage"], cohort_size, fold_sizes):
        return False
    matched = _counts_by_name(panel["coverage"], "matched")
    for field in panel["fields"].values():
        if not _validate_field(field, cohort_size, fold_sizes, matched):
            return False
    return True


def _validate_coverage_counts(value: Any, cohort_size: int, fold_sizes: tuple[int, ...] | list[int]) -> bool:
    if not _exact_keys(value, ("overall", "folds")) or not _exact_keys(value["overall"], ("matched", "unmatched")):
        return False
    folds = value["folds"]
    if not isinstance(folds, (tuple, list)) or len(folds) != 5:
        return False
    cells = [value["overall"], *folds]
    if any(not _exact_keys(cell, ("matched", "unmatched")) for cell in cells):
        return False
    if any(not _valid_safe_cell(number) for cell in cells for number in cell.values()):
        return False
    if any(cell["matched"] + cell["unmatched"] != fold_sizes[index] for index, cell in enumerate(folds)):
        return False
    return (
        value["overall"]["matched"] == sum(cell["matched"] for cell in folds)
        and value["overall"]["unmatched"] == sum(cell["unmatched"] for cell in folds)
        and value["overall"]["matched"] + value["overall"]["unmatched"] == cohort_size
    )


def _validate_field(field: Any, cohort_size: int, fold_sizes: tuple[int, ...] | list[int], matched: tuple[int, ...]) -> bool:
    if not _exact_keys(field, ("status", "counts")):
        return False
    if field["status"] == STATUS_SUPPRESSED_FIELD_SMALL_CELL:
        return field["counts"] is None
    if field["status"] != STATUS_SUPPORTED:
        return False
    counts = field["counts"]
    if not _exact_keys(counts, ("overall", "folds")) or not _exact_keys(counts["overall"], ("observed", "missing")):
        return False
    folds = counts["folds"]
    if not isinstance(folds, (tuple, list)) or len(folds) != 5:
        return False
    cells = [counts["overall"], *folds]
    if any(not _exact_keys(cell, ("observed", "missing")) for cell in cells):
        return False
    if any(not _valid_safe_cell(number) for cell in cells for number in cell.values()):
        return False
    if any(cell["observed"] + cell["missing"] != fold_sizes[index] for index, cell in enumerate(folds)):
        return False
    observed = _counts_by_name(counts, "observed")
    if any(observed[index] > matched[index] for index in range(6)):
        return False
    # The withheld within-source missing complements are derivable from published
    # coverage and field observations, so they must satisfy the same cell floor.
    if any(not _valid_safe_cell(matched[index] - observed[index]) for index in range(6)):
        return False
    return (
        counts["overall"]["observed"] == sum(cell["observed"] for cell in folds)
        and counts["overall"]["missing"] == sum(cell["missing"] for cell in folds)
        and counts["overall"]["observed"] + counts["overall"]["missing"] == cohort_size
    )


def _counts_by_name(counts: dict[str, Any], name: str) -> tuple[int, ...]:
    return (counts["overall"][name], *(fold[name] for fold in counts["folds"]))
