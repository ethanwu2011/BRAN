"""Local-only CBC/age/time adapters; no I/O, fitting, or patient-facing output.

These functions implement prospective preprocessing, not source admission. A
caller must separately bind source files/dictionaries, linkage and cohort roles.
Unknown units/assays remain missing. No normalization is learned here.
"""
from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real, Integral
import re
from types import MappingProxyType

CBC_FIELDS = ('hct', 'hemoglobin', 'mch', 'mchc', 'mcv', 'plt', 'rbc', 'rdw', 'wbc')
SOURCES = frozenset(('mimic', 'eicu', 'nwicu', 'sicdb', 'nhanes', 'zigong'))
CANONICAL_UNITS = MappingProxyType(dict(zip(
    CBC_FIELDS, ('%', 'g/dL', 'pg', 'g/dL', 'fL', '10^3/uL', '10^6/uL', '%', '10^3/uL'))))
MIMIC_CBC_CODES = MappingProxyType(dict(zip(
    ('51221', '51222', '51248', '51249', '51250', '51265', '51279', '51277', '51301'), CBC_FIELDS)))
NHANES_CBC_CODES = MappingProxyType(dict(zip(
    ('LBXHCT', 'LBXHGB', 'LBXMCHSI', 'LBXMC', 'LBXMCVSI', 'LBXPLTSI', 'LBXRBCSI', 'LBXRDW', 'LBXWBCSI'), CBC_FIELDS)))
EICU_CBC_NAMES = MappingProxyType(dict(zip(
    ('Hct', 'Hgb', 'MCH', 'MCHC', 'MCV', 'platelets x 1000', 'RBC', 'RDW', 'WBC x 1000'), CBC_FIELDS)))


def _unit_token(unit):
    if not isinstance(unit, str):
        return None
    # Formatting only: never infer magnitude, assay, or specimen from a value.
    return re.sub(r'\s+', '', unit.strip().replace('µ', 'u').replace('μ', 'u')).lower()


_FACTORS = {
    'hct': {'%': 1., 'percent': 1., 'l/l': 100.},
    'hemoglobin': {'g/dl': 1., 'g/l': .1},
    'mch': {'pg': 1.},
    'mchc': {'g/dl': 1., 'g/l': .1},
    'mcv': {'fl': 1.},
    'plt': {'10^3/ul': 1., 'k/ul': 1., '10^9/l': 1., '1000cells/ul': 1., 'cells/ul': .001},
    'rbc': {'10^6/ul': 1., 'm/ul': 1., '10^12/l': 1., 'millioncells/ul': 1., 'cells/ul': .000001},
    # RDW-CV (%) is deliberately not interchangeable with RDW-SD (fL).
    'rdw': {'%': 1., 'percent': 1.},
    'wbc': {'10^3/ul': 1., 'k/ul': 1., '10^9/l': 1., '1000cells/ul': 1., 'cells/ul': .001},
}
UNIT_FACTORS = MappingProxyType({k: MappingProxyType(v) for k, v in _FACTORS.items()})
del _FACTORS


@dataclass(frozen=True, repr=False)
class CanonicalObservation:
    value: float
    observed: bool
    status: str


def _unobserved(status):
    return CanonicalObservation(math.nan, False, status)


def canonicalize_cbc(field, value, unit, *, provenance):
    """Convert a known CBC analyte; provenance 0=missing, 1=observed, 2=imputed.

    No upper plausibility cutoff or winsorization is invented. Zero values are
    retained at this unit-conversion layer; any clinical exclusion is a separate
    frozen rule. Numeric input must already be parsed without inequality coercion.
    """
    if not isinstance(field, str) or field not in CBC_FIELDS:
        raise ValueError('unsupported canonical CBC field')
    if isinstance(provenance, bool) or not isinstance(provenance, Integral) or provenance not in (0, 1, 2):
        raise ValueError('unsupported observation provenance')
    if provenance != 1:
        return _unobserved('missing' if provenance == 0 else 'imputed_excluded')
    if isinstance(value, bool) or not isinstance(value, Real):
        return _unobserved('invalid_numeric_value')
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return _unobserved('invalid_numeric_value')
    if not math.isfinite(number) or number < 0:
        return _unobserved('invalid_numeric_value')
    factor = UNIT_FACTORS[field].get(_unit_token(unit))
    if factor is None:
        return _unobserved('unsupported_unit')
    converted = number * factor
    if not math.isfinite(converted):
        return _unobserved('invalid_numeric_value')
    return CanonicalObservation(converted, True, 'observed_converted')


def source_cbc_field(source, code, *, cycle=None):
    """Closed draft mappings, NOT evidence that the local dictionary was checked.

    eICU names require source/hospital assay validation before source admission.
    NWICU and SICdb require independently authenticated dictionary mappings;
    MIMIC item identifiers must never be transplanted into those sources.
    """
    if source not in SOURCES:
        raise ValueError('unsupported source')
    if not isinstance(code, str):
        return None
    if source == 'nhanes':
        if cycle not in ('D', 'E'):
            raise ValueError('unsupported NHANES cycle')
        return NHANES_CBC_CODES.get(code)
    if source == 'mimic':
        return MIMIC_CBC_CODES.get(code)
    if source == 'eicu':
        return EICU_CBC_NAMES.get(code)
    return None


def nhanes_cbc_observation(cycle, code, value, *, provenance):
    """Cycle-specific codebook unit (XPORT has no per-observation unit column)."""
    field = source_cbc_field('nhanes', code, cycle=cycle)
    if field is None:
        return _unobserved('unsupported_assay')
    return canonicalize_cbc(field, value, CANONICAL_UNITS[field], provenance=provenance)


@dataclass(frozen=True, repr=False)
class AgeObservation:
    # Bounds describe source precision, not confidence/credible intervals.
    reported_years: float
    lower_years: float
    upper_years: float
    kind: str
    reference: str


def _missing_age(kind='missing_or_invalid', reference='unknown'):
    return AgeObservation(math.nan, math.nan, math.nan, kind, reference)


def _integer(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        if re.fullmatch(r'[0-9]{1,4}', value.strip()) is None:
            return None
        return int(value.strip())
    if not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


def decode_age(source, value, *, cycle=None, anchor_year=None, admission_year=None):
    """Retain source censoring/coarsening; never impute an exact chronological age.

    NHANES uses screening age, NOT exact age at blood collection. MIMIC uses an
    admission-year approximation, with conservative +/-1-year bounds. SICdb's
    rounded 90 is both coarsened and capped, so its upper bound remains open.
    NWICU's specific age definition remains unresolved; schema similarity alone
    cannot license reusing the MIMIC rule.
    """
    if source not in SOURCES:
        raise ValueError('unsupported source')
    if source in ('nwicu', 'zigong'):
        return _missing_age('source_age_unresolved')
    if source == 'nhanes' and cycle not in ('D', 'E'):
        raise ValueError('unsupported NHANES cycle')
    if source == 'eicu' and isinstance(value, str) and re.fullmatch(r'>\s*89', value.strip()):
        return AgeObservation(math.nan, 90., math.inf, 'topcoded', 'unit_admission')
    age = _integer(value)
    if age is None or age < 0:
        return _missing_age()
    if source == 'nhanes':
        cap = 85 if cycle == 'D' else 80
        if age > cap:
            return _missing_age()
        return AgeObservation(float(age), float(age), math.inf if age == cap else float(age + 1),
                              'topcoded' if age == cap else 'reported_year', 'household_screening')
    if source == 'eicu':
        if age > 89:
            return _missing_age()
        return AgeObservation(float(age), float(age), float(age + 1), 'reported_year', 'unit_admission')
    if source == 'sicdb':
        if age > 90:
            return _missing_age()
        return AgeObservation(float(age), float(max(0, age - 5)), math.inf if age == 90 else float(age + 5),
                              'rounded_topcoded' if age == 90 else 'rounded', 'metavision_admission')
    anchor, admission = _integer(anchor_year), _integer(admission_year)
    if anchor is None or admission is None or not (1 <= anchor <= 9999 and 1 <= admission <= 9999):
        return _missing_age()
    if age > 91 or age == 90:
        return _missing_age()
    delta = admission - anchor
    if age == 91:
        # 91 is a code, not an age. Apply elapsed years only to the lower bound.
        return AgeObservation(math.nan, float(max(0, 89 + delta)), math.inf, 'topcoded', 'hospital_admission')
    estimate = age + delta
    if estimate < 0:
        return _missing_age()
    return AgeObservation(float(estimate), float(max(0, estimate - 1)), float(estimate + 1),
                          'year_derived', 'hospital_admission')


def current_v2_age_scalar(age):
    """Conservative adult bridge, not an automatic admission rule for a new model.

    The existing scalar-only encoder has no interval/censoring flag. Do not feed
    capped/rounded ages through it as exact values. A future candidate may use
    these observations once its coarsened/missing-age policy is explicitly set.
    """
    if not isinstance(age, AgeObservation):
        raise ValueError('unsupported age object')
    if age.kind not in ('reported_year', 'year_derived') or not math.isfinite(age.reported_years) or age.lower_years < 18:
        return None
    return age.reported_years


def relative_minutes(source, *, offset=None, lab_time=None, admission_time=None):
    """Return source-relative offset without changing origin or clipping negatives.

    MIMIC/NWICU hospital admission; eICU ICU-unit admission; SICdb initial
    MetaVision admission (can include preceding surgery). These are not the same
    clinical event. NHANES uses a survey examination panel, not ICU timestamps.
    """
    if source not in SOURCES:
        raise ValueError('unsupported source')
    if source in ('mimic', 'nwicu'):
        if offset is not None or not isinstance(lab_time, datetime) or not isinstance(admission_time, datetime):
            raise ValueError('invalid timestamp input')
        try:
            number = (lab_time - admission_time).total_seconds() / 60.
        except (TypeError, ValueError, OverflowError):
            raise ValueError('invalid timestamp input') from None
    elif source in ('eicu', 'sicdb'):
        if lab_time is not None or admission_time is not None or isinstance(offset, bool) or not isinstance(offset, Real):
            raise ValueError('invalid offset input')
        try:
            number = float(offset) / (60. if source == 'sicdb' else 1.)
        except (ValueError, TypeError, OverflowError):
            raise ValueError('invalid offset input') from None
    else:
        raise ValueError('source has no approved offset conversion')
    return number if math.isfinite(number) else math.nan
