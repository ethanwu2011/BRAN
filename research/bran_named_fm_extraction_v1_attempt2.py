"""Attempt-2 canonical-row wrapper for the frozen local FM adapter.

Only present retinal rows are passed to the attempt-1 adapter.  Their returned
embeddings are scattered back into the caller's original canonical patient
row-space, where patients with no selected retinal image retain an exact zero
vector.  This module performs no enumeration, persistence, or outcome access.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

import bran_named_fm_extraction_v1 as _attempt1


FMExtractionError = _attempt1.FMExtractionError
extract_labrador = _attempt1.extract_labrador


def _validated_rows(
    paths: Sequence[str | Path], patient_rows: np.ndarray, patient_count: int
) -> np.ndarray:
    """Validate enough to compact safely; attempt 1 remains the path guard."""

    try:
        path_count = len(paths)
    except Exception:
        raise FMExtractionError("retinal paths are malformed") from None
    rows = np.asarray(patient_rows)
    if (
        rows.ndim != 1
        or len(rows) != path_count
        or not np.issubdtype(rows.dtype, np.integer)
        or rows.dtype == np.dtype(bool)
        or isinstance(patient_count, bool)
        or not isinstance(patient_count, (int, np.integer))
        or int(patient_count) < 1
    ):
        raise FMExtractionError("retinal patient rows are malformed")
    if len(rows) == 0:
        raise FMExtractionError("retinal paths are malformed")
    if bool((rows < 0).any()) or bool((rows >= int(patient_count)).any()):
        raise FMExtractionError("retinal patient rows are malformed")
    return np.ascontiguousarray(rows, dtype=np.int64)


def extract_retinal(
    paths: Sequence[str | Path],
    patient_rows: np.ndarray,
    patient_count: int,
    *,
    variant: str,
    artifact: Mapping[str, Any],
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> np.ndarray:
    """Return canonical-row retinal embeddings with exact-zero absent rows."""

    rows = _validated_rows(paths, patient_rows, patient_count)
    present_rows = np.unique(rows)
    compact_rows = np.searchsorted(present_rows, rows).astype(np.int64, copy=False)
    present_embedding = _attempt1.extract_retinal(
        paths,
        compact_rows,
        int(len(present_rows)),
        variant=variant,
        artifact=artifact,
        progress_callback=progress_callback,
    )
    values = np.asarray(present_embedding, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(present_rows):
        raise FMExtractionError("retinal FM embedding shape differs")
    output = np.zeros((int(patient_count), values.shape[1]), dtype=np.float32)
    output[present_rows] = values
    return output


__all__ = ["FMExtractionError", "extract_labrador", "extract_retinal"]
