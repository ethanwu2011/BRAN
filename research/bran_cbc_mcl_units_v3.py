"""Exact microliter display aliases, independent of measurement magnitude.

This adds no assay or training admission. Frozen V1/V2 remain unchanged.
HL7's approved Results-with-translation-unit example explicitly translates
THOUS/MCL to UCUM 10*3/uL (thousand per microliter). `mcL` below is a
case-sensitive display spelling, not a general UCUM parser.
"""
import math
from types import MappingProxyType

from bran_cbc_ucum_aliases_v2 import _token, canonicalize_cbc_v2
from bran_clinical_semantics_v1 import CanonicalObservation

PRIMARY_SOURCE = 'https://cdasearch.hl7.org/examples/view/Results/Results%20with%20translation%20unit'
DISPLAY_CELLS_PER_UL = MappingProxyType({
    'K/mcL': 1000., 'x10^3/mcL': 1000., '10^3/mcL': 1000.,
    'x10(3)/mcL': 1000.,
    'M/mcL': 1000000., 'x10^6/mcL': 1000000., '10^6/mcL': 1000000.,
    'x10(6)/mcL': 1000000.,
})
_CANONICAL_CELLS_PER_UL = MappingProxyType({'plt': 1000., 'wbc': 1000., 'rbc': 1000000.})


def canonicalize_cbc_v3(field, value, unit, *, provenance=1):
    """Preserve missing/imputed/invalid inputs; never use a value to infer units."""
    original = canonicalize_cbc_v2(field, value, unit, provenance=provenance)
    if original.status != 'unsupported_unit' or field not in _CANONICAL_CELLS_PER_UL:
        return original
    numerator = DISPLAY_CELLS_PER_UL.get(_token(unit))
    if numerator is None:
        return original
    converted = float(value) * (numerator / _CANONICAL_CELLS_PER_UL[field])
    if not math.isfinite(converted):
        return CanonicalObservation(math.nan, False, 'invalid_numeric_value')
    return CanonicalObservation(converted, True, 'observed_converted')
