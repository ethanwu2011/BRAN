"""Pure INSPIRE-to-BRAN original-unit input assembly.

This module is intentionally only an array contract.  It accepts a caller
selected, fixed-width INSPIRE laboratory matrix and typed original-year age
bounds.  It does not read source files, authenticate source metadata, infer a
source or assay, use timing, derive a cohort, normalize values, or invoke a
model.

The adapter keeps the BRAN 59-slot clinical registry width.  The only slots it
can populate are the eighteen fixed laboratory fields in :data:`SOURCE_FIELDS`.
Unobserved payload is represented as ``NaN`` plus a false mask; the payload of
a false-masked input cell is never copied into the result.  A true observation
must already be finite and sign-valid (the upstream source adapter erases bad
latest readings), otherwise this boundary fails generically.  Ages are
retained as original-year intervals or explicit unknowns and are deliberately
not part of ``eligible``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from bran_knhanes_input_kernel_v1 import (
    ADMITTED_CANONICAL_INDICES,
    CANONICAL_INDEX,
    CANONICAL_NAMES,
)
from bran_multisource_age_v2 import AgeBatch, INTERVAL, UNKNOWN


ERROR = "inspire_input_adapter_contract_failed"
CLINICAL_WIDTH = 59
RETINAL_VIEWS = 2
RETINAL_FEATURE_DIM = 384


# This order is part of the public positional-array contract.  Do not derive
# it from a source dictionary or accept an arbitrary caller ordering.
SOURCE_FIELDS = (
    "albumin",
    "alp",
    "alt",
    "ast",
    "bun",
    "chloride",
    "creatinine",
    "crp",
    "glucose",
    "hb",
    "hba1c",
    "hct",
    "platelet",
    "potassium",
    "sodium",
    "total_bilirubin",
    "total_protein",
    "wbc",
)


@dataclass(frozen=True, slots=True, repr=False)
class SourceFieldSpec:
    """A closed, value-free source-to-canonical conversion specification."""

    canonical_name: str
    source_unit: str
    canonical_unit: str
    conversion_factor: float

    def __repr__(self) -> str:
        return "<InspireSourceFieldSpec>"


_FIELD_ROWS = (
    ("albumin", "albumin", "g/dL", "g/dL", 1.0),
    ("alp", "alkaline_phosphatase", "IU/L", "IU/L", 1.0),
    ("alt", "alt_got", "IU/L", "IU/L", 1.0),
    ("ast", "ast_got", "IU/L", "IU/L", 1.0),
    ("bun", "bun", "mg/dL", "mg/dL", 1.0),
    ("chloride", "chloride", "mmol/L", "mEq/L", 1.0),
    ("creatinine", "creatinine", "mg/dL", "mg/dL", 1.0),
    # The canonical registry calls this fixed slot ``crp_hs``.  Its requested
    # canonical unit is mg/L even though the legacy chemistry table predates
    # this field; no unit is guessed from an observation value.
    ("crp", "crp_hs", "mg/dL", "mg/L", 10.0),
    ("glucose", "glucose", "mg/dL", "mg/dL", 1.0),
    ("hb", "hemoglobin", "g/dL", "g/dL", 1.0),
    ("hba1c", "hba1c", "%", "%", 1.0),
    ("hct", "hct", "%", "%", 1.0),
    ("platelet", "plt", "/nL", "10^3/uL", 1.0),
    ("potassium", "potassium", "mmol/L", "mEq/L", 1.0),
    ("sodium", "sodium", "mmol/L", "mEq/L", 1.0),
    ("total_bilirubin", "bilirubin_total", "mg/dL", "mg/dL", 1.0),
    ("total_protein", "protein_total", "g/dL", "g/dL", 1.0),
    ("wbc", "wbc", "/nL", "10^3/uL", 1.0),
)


def _field_specs() -> dict[str, SourceFieldSpec]:
    return {
        source: SourceFieldSpec(canonical, source_unit, canonical_unit, factor)
        for source, canonical, source_unit, canonical_unit, factor in _FIELD_ROWS
    }


_FIELD_SPEC_DICT = _field_specs()
SOURCE_FIELD_SPECS = MappingProxyType(_FIELD_SPEC_DICT)

# ``SOURCE_TO_CANONICAL`` is deliberately a mapping of exactly the eighteen
# source fields and nothing else.
SOURCE_TO_CANONICAL = MappingProxyType({
    source: spec.canonical_name for source, spec in SOURCE_FIELD_SPECS.items()
})
SOURCE_UNITS = tuple(SOURCE_FIELD_SPECS[field].source_unit for field in SOURCE_FIELDS)
CANONICAL_UNITS = MappingProxyType({
    spec.canonical_name: spec.canonical_unit
    for spec in SOURCE_FIELD_SPECS.values()
})
SOURCE_UNIT_FACTORS = MappingProxyType({
    source: spec.conversion_factor for source, spec in SOURCE_FIELD_SPECS.items()
})


# Canonical registry checks are row-free and happen at import time.  The CRP
# slot is present in the fixed registry at index 13.  The legacy chemistry
# semantics module predates three requested fields; their authorized units are
# pinned by the row-free reconciliation receipt below.  The caller must
# authenticate that receipt before real inference; this adapter performs no
# file I/O or receipt validation.
CRP_CANONICAL_FIELD = "crp_hs"
CRP_CANONICAL_INDEX = CANONICAL_INDEX.get(CRP_CANONICAL_FIELD)
CRP_CANONICAL_CODE_MATCH = (
    CRP_CANONICAL_INDEX is not None
    and CANONICAL_NAMES[CRP_CANONICAL_INDEX] == CRP_CANONICAL_FIELD
)
CRP_CANONICAL_UNIT = "mg/L"
CRP_CANONICAL_UNIT_REGISTRY_PRESENT = False
CRP_CANONICAL_UNIT_REGISTRY_MISMATCH = False
CRP_SCHEMA_FLAG = "caller_must_authenticate_pinned_unit_receipt_before_inference"

# The checked-in canonical chemistry semantics do not provide unit evidence for
# these three slots.  Keep that local gap visible while pinning the separately
# authenticated, row-free official reconciliation artifact.  The adapter does
# not infer or authenticate units from a patient value.
CANONICAL_UNIT_EVIDENCE_MISSING_FIELDS = frozenset((
    "crp_hs", "alkaline_phosphatase", "hba1c",
))
UNIT_EVIDENCE_SCHEMA_FLAG = "caller_must_authenticate_pinned_unit_receipt_before_inference"
OFFICIAL_UNIT_RECONCILIATION_SHA256 = (
    "4d428667185a974116b0f13d525bcf86c9837ea86806604b26f6f0c41184e167"
)


# No caller-supplied field can target any other registry slot.  Keeping this
# as a frozenset also makes the mask-subset check independent of source data.
_ALLOWED_CANONICAL_INDICES = frozenset(
    CANONICAL_INDEX[spec.canonical_name] for spec in SOURCE_FIELD_SPECS.values()
)
ALLOWED_CANONICAL_FIELDS = frozenset(SOURCE_TO_CANONICAL.values())
DISABLED_SOURCE_FIELDS = frozenset((
    "calcium",
    "bicarbonate",
    "blood_gas_bicarbonate",
    "troponin",
    "derived_cbc",
    "vitals",
    "height",
    "bmi",
    "diagnosis",
    "context",
))
_ZERO_ALLOWED_SOURCE_FIELDS = frozenset(("alp", "alt", "ast", "crp"))


class _Invalid(Exception):
    """Internal sentinel collapsed to one public, value-free error."""


def _require(condition: bool) -> None:
    if not condition:
        raise _Invalid


def _numeric_array(value: Any, *, ndim: int) -> np.ndarray:
    _require(type(value) is np.ndarray)
    _require(value.ndim == ndim)
    _require(value.dtype.kind in "fiu")
    _require(value.dtype.kind != "b")
    return value


def _bool_array(value: Any, *, ndim: int) -> np.ndarray:
    _require(type(value) is np.ndarray)
    _require(value.ndim == ndim and value.dtype == np.dtype(bool))
    return value


def _kind_array(value: Any, n_rows: int) -> np.ndarray:
    _require(type(value) is np.ndarray)
    _require(value.ndim == 1 and value.shape == (n_rows,))
    _require(value.dtype.kind in "iu" and value.dtype.kind != "b")
    return value


def _validated_source_fields(source_fields: Any) -> tuple[str, ...]:
    _require(isinstance(source_fields, (tuple, list)))
    _require(tuple(source_fields) == SOURCE_FIELDS)
    return SOURCE_FIELDS


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


class _SafeAgeBatch(AgeBatch):
    """Fresh typed age view whose repr and pickle never expose age values."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<INSPIREAgeBatch>"

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)


class InspireInputBatch:
    """Readonly fixed-width INSPIRE inputs.

    The object stores only copies of caller arrays.  ``ages`` is a fresh typed
    :class:`~bran_multisource_age_v2.AgeBatch` on each access so mutating a
    caller-owned view cannot alter this result.  IDs, outcomes, timestamps,
    provenance, and source metadata are intentionally absent.
    """

    __slots__ = (
        "_clinical_values",
        "_clinical_mask",
        "_retinal_features",
        "_retinal_mask",
        "_eligible",
        "_age_lower",
        "_age_upper",
        "_age_kind",
    )

    def __init__(
        self,
        clinical_values: np.ndarray,
        clinical_mask: np.ndarray,
        retinal_features: np.ndarray,
        retinal_mask: np.ndarray,
        eligible: np.ndarray,
        age_lower: np.ndarray,
        age_upper: np.ndarray,
        age_kind: np.ndarray,
    ) -> None:
        self._clinical_values = _readonly_copy(clinical_values)
        self._clinical_mask = _readonly_copy(clinical_mask)
        self._retinal_features = _readonly_copy(retinal_features)
        self._retinal_mask = _readonly_copy(retinal_mask)
        self._eligible = _readonly_copy(eligible)
        self._age_lower = _readonly_copy(age_lower)
        self._age_upper = _readonly_copy(age_upper)
        self._age_kind = _readonly_copy(age_kind)

    def __repr__(self) -> str:
        return "<INSPIREInputBatch>"

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)

    @property
    def clinical_values(self) -> np.ndarray:
        return self._clinical_values

    @property
    def clinical_mask(self) -> np.ndarray:
        return self._clinical_mask

    @property
    def retinal_features(self) -> np.ndarray:
        return self._retinal_features

    @property
    def retinal_mask(self) -> np.ndarray:
        return self._retinal_mask

    @property
    def eligible(self) -> np.ndarray:
        return self._eligible

    @property
    def ages(self) -> AgeBatch:
        # Construct from fresh copies.  The stored NumPy arrays remain private
        # and read-only; the returned tensors are caller-owned and disposable.
        lower = torch.from_numpy(np.array(self._age_lower, dtype=np.float64, copy=True))
        upper = torch.from_numpy(np.array(self._age_upper, dtype=np.float64, copy=True))
        kind = torch.from_numpy(np.array(self._age_kind, dtype=np.int64, copy=True))
        value = torch.full((self._age_kind.shape[0],), float("nan"), dtype=torch.float64)
        return _SafeAgeBatch(value, lower, upper, kind)


def _validate_age_arrays(
    age_lower: Any,
    age_upper: Any,
    age_kind: Any,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = _numeric_array(age_lower, ndim=1)
    upper = _numeric_array(age_upper, ndim=1)
    kind = _kind_array(age_kind, n_rows)
    _require(lower.shape == (n_rows,) and upper.shape == (n_rows,))

    # Convert only after shape and dtype validation.  Unknown rows are
    # deliberately not inspected: their arbitrary payload is erased below.
    with np.errstate(over="ignore", invalid="ignore"):
        lower64 = lower.astype(np.float64, copy=True)
        upper64 = upper.astype(np.float64, copy=True)
    kind64 = kind.astype(np.int64, copy=True)
    _require(np.all((kind64 == INTERVAL) | (kind64 == UNKNOWN)))

    interval = kind64 == INTERVAL
    if np.any(interval):
        interval_lower = lower64[interval]
        interval_upper = upper64[interval]
        _require(bool(np.all(np.isfinite(interval_lower))))
        _require(bool(np.all(np.isfinite(interval_upper))))
        _require(bool(np.all(interval_lower >= 0.0)))
        _require(bool(np.all(interval_upper >= 0.0)))
        _require(bool(np.all(interval_lower <= interval_upper)))

    # Unknown age payload is always erased, including finite, infinite, and
    # huge values.  This branch intentionally does not validate unknown rows.
    unknown = kind64 == UNKNOWN
    lower64[unknown] = np.nan
    upper64[unknown] = np.nan
    return lower64, upper64, kind64


def verify59slotmasksubset(clinical_mask: Any) -> bool:
    """Verify that a mask has BRAN's width and only the eighteen allowed slots.

    Invalid input raises the same generic contract error as the main adapter;
    no array values are included in the exception.
    """

    try:
        _require(CLINICAL_WIDTH == 59 and len(CANONICAL_NAMES) == 59)
        _require(_ALLOWED_CANONICAL_INDICES.issubset(ADMITTED_CANONICAL_INDICES))
        mask = _bool_array(clinical_mask, ndim=2)
        _require(mask.shape[1] == CLINICAL_WIDTH)
        if mask.shape[0]:
            forbidden = np.ones(CLINICAL_WIDTH, dtype=bool)
            forbidden[list(_ALLOWED_CANONICAL_INDICES)] = False
            _require(not bool(mask[:, forbidden].any()))
        return True
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def _prepare(
    values: Any,
    mask: Any,
    age_lower: Any,
    age_upper: Any,
    age_kind: Any,
    source_fields: Any,
) -> InspireInputBatch:
    _require(CLINICAL_WIDTH == 59 and len(CANONICAL_NAMES) == 59)
    _require(_ALLOWED_CANONICAL_INDICES.issubset(ADMITTED_CANONICAL_INDICES))
    _validated_source_fields(source_fields)
    source_values = _numeric_array(values, ndim=2)
    observed = _bool_array(mask, ndim=2)
    _require(source_values.shape[1] == len(SOURCE_FIELDS))
    _require(observed.shape == source_values.shape)
    n_rows = source_values.shape[0]
    lower, upper, kind = _validate_age_arrays(age_lower, age_upper, age_kind, n_rows)

    # Copy before selecting rows so caller mutations after return cannot alter
    # the result.  Values hidden by a false mask are never converted or copied
    # into a clinical slot.
    with np.errstate(over="ignore", invalid="ignore"):
        values64 = source_values.astype(np.float64, copy=True)
    clinical_values = np.full((n_rows, CLINICAL_WIDTH), np.nan, dtype=np.float64)
    clinical_mask = np.zeros((n_rows, CLINICAL_WIDTH), dtype=bool)

    for column, source_field in enumerate(SOURCE_FIELDS):
        rows = np.flatnonzero(observed[:, column])
        if rows.size == 0:
            continue
        # Only observed rows are touched.  This is both a missing-payload
        # invariant and a simple guard against accidental metadata leakage.
        source_observed = values64[rows, column]
        with np.errstate(over="ignore", invalid="ignore"):
            valid = np.isfinite(source_observed)
            if source_field in _ZERO_ALLOWED_SOURCE_FIELDS:
                valid &= source_observed >= 0.0
            else:
                valid &= source_observed > 0.0
            factor = SOURCE_FIELD_SPECS[source_field].conversion_factor
            converted = source_observed * factor
            valid &= np.isfinite(converted)
        # The upstream source adapter is responsible for erasing invalid
        # latest readings.  A true observation mask reaching this boundary is
        # therefore an invariant: any nonfinite, sign-invalid, or overflowing
        # value fails closed instead of being silently reclassified here.
        _require(bool(np.all(valid)))
        output_rows = rows[valid]
        canonical_name = SOURCE_FIELD_SPECS[source_field].canonical_name
        canonical_index = CANONICAL_INDEX[canonical_name]
        clinical_values[output_rows, canonical_index] = converted[valid]
        clinical_mask[output_rows, canonical_index] = True

    # Retinal inputs are intentionally absent for this laboratory-only source.
    retinal_features = np.zeros(
        (n_rows, RETINAL_VIEWS, RETINAL_FEATURE_DIM), dtype=np.float64
    )
    retinal_mask = np.zeros((n_rows, RETINAL_VIEWS), dtype=bool)
    eligible = np.any(clinical_mask[:, tuple(sorted(_ALLOWED_CANONICAL_INDICES))], axis=1)

    # Keep this invariant close to construction: future mapping edits cannot
    # silently open a disabled slot.
    _require(verify59slotmasksubset(clinical_mask))
    _require(not bool(clinical_mask[:, [CANONICAL_INDEX["calcium"], CANONICAL_INDEX["carbon_dioxide_total"], CANONICAL_INDEX["troponin_t"]]].any()))

    return InspireInputBatch(
        clinical_values,
        clinical_mask,
        retinal_features,
        retinal_mask,
        eligible,
        lower,
        upper,
        kind,
    )


def prepare_inputs(
    values: Any = None,
    mask: Any = None,
    age_lower: Any = None,
    age_upper: Any = None,
    age_kind: Any = None,
    source_fields: Any = SOURCE_FIELDS,
) -> InspireInputBatch:
    """Prepare fixed-width INSPIRE laboratory inputs.

    Parameters are positional arrays by design: ``values`` and ``mask`` have
    shape ``(n, 18)`` in :data:`SOURCE_FIELDS` order; age bounds and kind have
    shape ``(n,)``.  ``kind`` may contain only the integer constants
    ``INTERVAL`` and ``UNKNOWN`` from :mod:`bran_multisource_age_v2`.

    Any non-array metadata (including IDs, outcomes, timing, or provenance) is
    outside this interface and fails with the generic adapter error.
    """

    try:
        return _prepare(values, mask, age_lower, age_upper, age_kind, source_fields)
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


__all__ = [
    "ALLOWED_CANONICAL_FIELDS",
    "CANONICAL_UNITS",
    "CLINICAL_WIDTH",
    "CRP_CANONICAL_CODE_MATCH",
    "CRP_CANONICAL_FIELD",
    "CRP_CANONICAL_INDEX",
    "CRP_CANONICAL_UNIT",
    "CRP_CANONICAL_UNIT_REGISTRY_MISMATCH",
    "CRP_CANONICAL_UNIT_REGISTRY_PRESENT",
    "CRP_SCHEMA_FLAG",
    "CANONICAL_UNIT_EVIDENCE_MISSING_FIELDS",
    "DISABLED_SOURCE_FIELDS",
    "ERROR",
    "InspireInputBatch",
    "RETINAL_FEATURE_DIM",
    "RETINAL_VIEWS",
    "OFFICIAL_UNIT_RECONCILIATION_SHA256",
    "SOURCE_FIELDS",
    "SOURCE_FIELD_SPECS",
    "SOURCE_TO_CANONICAL",
    "SOURCE_UNITS",
    "SOURCE_UNIT_FACTORS",
    "SourceFieldSpec",
    "UNIT_EVIDENCE_SCHEMA_FLAG",
    "prepare_inputs",
    "verify59slotmasksubset",
]
