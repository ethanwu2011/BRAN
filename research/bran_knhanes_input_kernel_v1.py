"""Pure, array-only KNHANES-to-BRAN candidate input preparation.

This module is deliberately a small contract boundary.  It does not read a
source file, identify a source assay, infer units, derive features, fit a
model, or admit a cohort.  A caller supplies an authenticated (or future
authenticated) column crosswalk as :class:`ColumnRule` objects and supplies
the row-level observation masks.

The output keeps the fixed 59-column clinical width expected by BRAN.  Missing
clinical values are NaN with a false mask; the portable model API sanitizes
false-masked values itself, while callers must subset ``eligible`` rows first.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real
import re
from types import MappingProxyType

import numpy as np

from bran_clinical_semantics_v1 import CANONICAL_UNITS, CBC_FIELDS, UNIT_FACTORS


# This is an embedded, row-free copy of the registry's ordered names.  It is
# intentionally not loaded from disk: the kernel has no source/registry I/O.
CANONICAL_NAMES = (
    'a_g_ratio', 'albumin', 'alkaline_phosphatase', 'alt_got', 'ast_got',
    'bilirubin_total', 'bun', 'buncreatinineratio', 'c_peptide', 'calcium',
    'carbon_dioxide_total', 'chloride', 'creatinine', 'crp_hs',
    'globulin_total', 'glucose', 'hba1c', 'hct', 'hdl_cholesterol',
    'hemoglobin', 'insulin', 'ldl_cholesterol', 'mch', 'mchc', 'mcv',
    'nt_probnp', 'plt', 'potassium', 'protein_total', 'rbc', 'rdw',
    'sodium', 'total_cholesterol', 'triglycerides', 'troponin_t',
    'urine_albumin', 'urine_creatinine', 'wbc', 'vit_bmi_vsorres',
    'vit_diabp_vsorres', 'vit_height_vsorres', 'vit_hip_vsorres',
    'vit_pulse_vsorres', 'vit_pulse_vsorres_2', 'vit_sysbp_vsorres',
    'vit_waist_vsorres', 'vit_weight_vsorres', 'vit_whr_vsorres',
    'hypertension', 'hyperlipidemia', 'diabetes', 'cancer', 'kidney',
    'myocardial_inf', 'stroke', 'arthritis', 'osteoporosis', 'heart_failure',
    'chronic_lung',
)

# Convenient public aliases for callers that prefer a feature-oriented name.
CANONICAL_FEATURE_NAMES = CANONICAL_NAMES
CANONICAL_INDEX = MappingProxyType({name: index for index, name in enumerate(CANONICAL_NAMES)})

# The candidate kernel accepts only the first 48 registry positions, with the
# five explicitly disabled positions removed.  The eleven history/binary
# positions are never writable by a ColumnRule.
DISABLED_CANONICAL_INDICES = frozenset((8, 9, 20, 35, 36))
ADMITTED_CANONICAL_INDICES = frozenset(
    index for index in range(48) if index not in DISABLED_CANONICAL_INDICES
)
ADMITTED_CANONICAL_NAMES = frozenset(CANONICAL_NAMES[index] for index in ADMITTED_CANONICAL_INDICES)
_CBC_CANONICAL_INDICES = frozenset(CANONICAL_INDEX[field] for field in CBC_FIELDS)
_WHOLE_CBC_INDICES = _CBC_CANONICAL_INDICES
_SINGLE_HB_INDICES = frozenset((CANONICAL_INDEX['hemoglobin'],))
_HASH_RE = re.compile(r'^[0-9a-f]{64}$')
_ERROR = 'input_kernel_contract_failed'


@dataclass(frozen=True, slots=True, repr=False)
class ColumnRule:
    """One caller-supplied raw-column to fixed canonical-slot conversion.

    ``conversion_offset`` is retained only as an explicit contract field and
    must be exactly zero.  It is last and optional so the nine core fields can
    also be supplied positionally.  The two SHA-256 values are evidence pins,
    not proof of provenance: their syntax is checked here, while ownership and
    authentication remain caller responsibilities.
    """

    source_column: str
    canonical_name: str
    canonical_index: int
    source_unit: str
    target_unit: str
    conversion_factor: float
    missing_codes: tuple
    source_guide_sha256: str
    model_unit_evidence_sha256: str
    conversion_offset: float = 0.0


@dataclass(frozen=True, slots=True, repr=False)
class PreparedInputs:
    """Readonly fixed-width candidate inputs.

    The representation intentionally omits every field value.  Pickling is
    denied as a precaution against accidentally serializing candidate data.
    """

    clinical_values: np.ndarray
    clinical_mask: np.ndarray
    retinal_features: np.ndarray
    retinal_mask: np.ndarray
    ages: np.ndarray
    eligible: np.ndarray

    def __repr__(self):
        return '<BRANKNHANESInputResult>'

    def __reduce__(self):
        raise TypeError(_ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(_ERROR)


# Descriptive aliases keep the result type easy to discover without creating
# alternate representations or exposing any additional data.
BRANKNHANESInputResult = PreparedInputs
InputKernelResult = PreparedInputs


class _ContractFailure(Exception):
    """Internal sentinel collapsed to one public error message."""


def _require(condition):
    if not condition:
        raise _ContractFailure


def _finite_real(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _numeric_array(value, *, ndim):
    """Return only a plain real numeric ndarray of the requested rank."""
    _require(type(value) is np.ndarray and value.ndim == ndim)
    # Kinds f/i/u cover real floats, signed integers, and unsigned integers.
    # In particular this rejects bool, complex, object/string, datetime64 and
    # timedelta64 without relying on broad ``np.number`` subclass behavior.
    _require(value.dtype.kind in 'fiu')
    return value


def _unit_token(unit):
    # Formatting normalization is limited to the existing CBC unit dictionary;
    # no magnitude, assay, specimen, or source provenance is inferred.
    return re.sub(r'\s+', '', unit.strip().replace('µ', 'u').replace('μ', 'u')).lower()


def _validate_sha(value):
    _require(isinstance(value, str) and _HASH_RE.fullmatch(value) is not None)


def _validate_rule(rule, source_columns, seen_sources, seen_targets):
    _require(type(rule) is ColumnRule)
    _require(isinstance(rule.source_column, str) and bool(rule.source_column))
    _require(rule.source_column in source_columns)
    _require(rule.source_column not in seen_sources)

    _require(isinstance(rule.canonical_name, str))
    _require(rule.canonical_name in CANONICAL_INDEX)
    _require(isinstance(rule.canonical_index, Integral) and not isinstance(rule.canonical_index, (bool, np.bool_)))
    canonical_index = int(rule.canonical_index)
    _require(canonical_index == CANONICAL_INDEX[rule.canonical_name])
    _require(canonical_index in ADMITTED_CANONICAL_INDICES)
    _require(canonical_index not in seen_targets)

    _require(isinstance(rule.source_unit, str) and bool(rule.source_unit.strip()))
    _require(isinstance(rule.target_unit, str) and bool(rule.target_unit.strip()))

    factor = _finite_real(rule.conversion_factor)
    _require(factor is not None and factor > 0.0)
    offset = _finite_real(rule.conversion_offset)
    _require(offset is not None and offset == 0.0)

    _require(type(rule.missing_codes) is tuple)
    missing_codes = []
    for code in rule.missing_codes:
        number = _finite_real(code)
        _require(number is not None)
        missing_codes.append(number)

    _validate_sha(rule.source_guide_sha256)
    _validate_sha(rule.model_unit_evidence_sha256)

    # CBC target units and conversion factors are closed by the existing pure
    # helper tables.  Non-CBC units intentionally remain opaque strings: only
    # their explicit spelling and evidence pins are checked above.
    if rule.canonical_name in CBC_FIELDS:
        _require(rule.target_unit == CANONICAL_UNITS[rule.canonical_name])
        source_key = _unit_token(rule.source_unit)
        _require(source_key in UNIT_FACTORS[rule.canonical_name])
        expected = float(UNIT_FACTORS[rule.canonical_name][source_key])
        _require(math.isfinite(expected) and factor == expected)

    seen_sources.add(rule.source_column)
    seen_targets.add(canonical_index)
    return canonical_index, factor, tuple(missing_codes)


def _validate_inputs(raw, source_columns, exact_observed, imputed, ages, rules,
                     prohibited_canonical_names):
    _numeric_array(raw, ndim=2)
    _numeric_array(ages, ndim=1)
    _require(type(exact_observed) is np.ndarray and exact_observed.dtype == np.dtype(bool))
    _require(type(imputed) is np.ndarray and imputed.dtype == np.dtype(bool))
    _require(exact_observed.ndim == 2 and imputed.ndim == 2)

    n_rows, n_columns = raw.shape
    _require(exact_observed.shape == raw.shape and imputed.shape == raw.shape)
    _require(ages.shape == (n_rows,))

    _require(isinstance(source_columns, (tuple, list)))
    _require(len(source_columns) == n_columns)
    _require(all(isinstance(name, str) and bool(name) for name in source_columns))
    _require(len(set(source_columns)) == len(source_columns))
    source_columns = tuple(source_columns)

    _require(type(rules) is tuple)
    _require(type(prohibited_canonical_names) is tuple)
    prohibited_indices = set()
    for name in prohibited_canonical_names:
        _require(isinstance(name, str) and name in CANONICAL_INDEX)
        _require(name not in {CANONICAL_NAMES[index] for index in prohibited_indices})
        prohibited_indices.add(CANONICAL_INDEX[name])

    seen_sources = set()
    seen_targets = set()
    validated_rules = []
    for rule in rules:
        validated_rules.append(_validate_rule(rule, source_columns, seen_sources, seen_targets))

    return source_columns, validated_rules, prohibited_indices


def _readonly_copy(array):
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _prepare_inputs(raw, source_columns, exact_observed, imputed, ages, rules,
                    mask_pattern, prohibited_canonical_names):
    _require(mask_pattern in ('whole_cbc', 'single_hb', 'available'))
    source_columns, validated_rules, prohibited_indices = _validate_inputs(
        raw, source_columns, exact_observed, imputed, ages, rules,
        prohibited_canonical_names,
    )

    n_rows = raw.shape[0]
    clinical_values = np.full((n_rows, len(CANONICAL_NAMES)), np.nan, dtype=np.float64)
    clinical_mask = np.zeros((n_rows, len(CANONICAL_NAMES)), dtype=bool)
    retinal_features = np.zeros((n_rows, 384), dtype=np.float64)
    retinal_mask = np.zeros(n_rows, dtype=bool)

    wiped_indices = set(prohibited_indices)
    if mask_pattern == 'whole_cbc':
        wiped_indices.update(_WHOLE_CBC_INDICES)
    elif mask_pattern == 'single_hb':
        wiped_indices.update(_SINGLE_HB_INDICES)

    source_index = {name: index for index, name in enumerate(source_columns)}
    for rule, (canonical_index, factor, missing_codes) in zip(rules, validated_rules):
        # Do this before touching the source column.  In particular, changing a
        # hidden target, imputed value, or censored value cannot affect output.
        if canonical_index in wiped_indices:
            continue

        source_index_for_rule = source_index[rule.source_column]
        observed_rows = np.flatnonzero(
            exact_observed[:, source_index_for_rule] &
            ~imputed[:, source_index_for_rule]
        )
        # Never cast or scale censored, imputed, or otherwise non-exact cells.
        # This also avoids touching a hidden value before the mask decision.
        if observed_rows.size == 0:
            continue
        with np.errstate(over='ignore', invalid='ignore'):
            source_values = raw[observed_rows, source_index_for_rule].astype(
                np.float64, copy=True
            )
        with np.errstate(over='ignore', invalid='ignore'):
            valid = np.isfinite(source_values)
            for missing_code in missing_codes:
                valid &= source_values != missing_code
            if rule.canonical_name in CBC_FIELDS:
                valid &= source_values >= 0.0
            usable_values = source_values[valid]
            converted = usable_values * factor + float(rule.conversion_offset)
            valid_converted = np.isfinite(converted)

        if np.any(valid_converted):
            usable_rows = observed_rows[valid]
            output_rows = usable_rows[valid_converted]
            clinical_values[output_rows, canonical_index] = converted[valid_converted]
            clinical_mask[output_rows, canonical_index] = True

    with np.errstate(over='ignore', invalid='ignore'):
        ages_out = ages.astype(np.float64, copy=True)
    age_valid = np.isfinite(ages_out) & (ages_out >= 0.0)
    ages_out[~age_valid] = np.nan
    eligible = age_valid & np.any(clinical_mask[:, tuple(sorted(ADMITTED_CANONICAL_INDICES))], axis=1)

    return PreparedInputs(
        _readonly_copy(clinical_values),
        _readonly_copy(clinical_mask),
        _readonly_copy(retinal_features),
        _readonly_copy(retinal_mask),
        _readonly_copy(ages_out),
        _readonly_copy(eligible),
    )


def prepare_inputs(raw, source_columns, exact_observed, imputed, ages, rules,
                   *, mask_pattern='whole_cbc', prohibited_canonical_names=()):
    """Prepare fixed-width candidate inputs from already-parsed arrays.

    All contract failures intentionally collapse to ``ValueError`` with the
    same value-free message.  This function does not authenticate either
    evidence hash's provenance; the future owner/caller must do that before
    passing rules here.
    """
    try:
        return _prepare_inputs(
            raw, source_columns, exact_observed, imputed, ages, rules,
            mask_pattern, prohibited_canonical_names,
        )
    except _ContractFailure:
        raise ValueError(_ERROR) from None
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        # Keep malformed ndarray subclasses and conversion edge cases from
        # leaking source values or implementation details through exceptions.
        raise ValueError(_ERROR) from None


__all__ = [
    'ADMITTED_CANONICAL_INDICES',
    'ADMITTED_CANONICAL_NAMES',
    'BRANKNHANESInputResult',
    'CANONICAL_FEATURE_NAMES',
    'CANONICAL_INDEX',
    'CANONICAL_NAMES',
    'ColumnRule',
    'DISABLED_CANONICAL_INDICES',
    'InputKernelResult',
    'PreparedInputs',
    'prepare_inputs',
]
