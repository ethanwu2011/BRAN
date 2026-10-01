"""Attempt2 local manifest adapter: collapse only duplicate-identical finite masks.

No values are aggregated, averaged, selected, or returned.  Repeated IDs keep their
first local position only when every prespecified finite-presence boolean agrees.
"""
from __future__ import annotations

import csv
from typing import Any

from bran_independent_coverage_io_v1 import (
    MAX_HEADER_COLUMNS, MAX_SOURCE_ROWS, MODALITY_FILES, SOURCE_COLUMNS,
    _bounded_text_lines, _canonical_id, _is_finite_numeric,
)


DUPLICATE_POLICY = "collapse_repeated_id_only_when_all_finite_presence_masks_identical_preserve_first_order"


def parse_manifest(handle: Any, modality: str) -> tuple[tuple[str, ...], dict[str, tuple[bool, ...]]]:
    """Return local first-order IDs and masks, rejecting inconsistent repeats."""
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
        masks: dict[str, list[bool]] = {field: [] for field in fields}
        first_mask_by_id: dict[str, tuple[bool, ...]] = {}
        for row_count, row in enumerate(reader, start=1):
            if row_count > MAX_SOURCE_ROWS or None in row:
                raise ValueError("malformed manifest row")
            identifier = _canonical_id(row.get("person_id"))
            mask = tuple(_is_finite_numeric(row.get(column)) for column in fields.values())
            previous = first_mask_by_id.get(identifier)
            if previous is not None:
                if previous != mask:
                    raise ValueError("repeated ID finite masks conflict")
                continue
            first_mask_by_id[identifier] = mask
            source_ids.append(identifier)
            for field, finite in zip(fields, mask, strict=True):
                masks[field].append(finite)
    except (csv.Error, UnicodeError) as error:
        raise ValueError("malformed manifest") from error
    return tuple(source_ids), {field: tuple(values) for field, values in masks.items()}
