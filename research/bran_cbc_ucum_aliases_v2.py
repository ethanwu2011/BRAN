"""Explicit count-concentration spellings; no source admission or unit guessing.

The frozen V1 unit table is unchanged. These additional exact expressions use
UCUM powers/volume identities, independently of patient measurement magnitude.
"""
from types import MappingProxyType
import re

from bran_clinical_semantics_v1 import CBC_FIELDS, canonicalize_cbc

PRIMARY_SOURCES = (
    'https://ucum.org/ucum',
    'https://ucum.org/docs/common-units',
)


def _token(unit):
    if not isinstance(unit, str): return None
    # Legacy display spelling normalization, not a general UCUM parser.
    return re.sub(r'\s+', '', unit.replace('µ','u').replace('μ','u'))


_COUNT3 = {
    '10*3/uL': '10^3/uL', '10*3/mm3': '10^3/uL', '10*3/mm^3': '10^3/uL',
    '10^3/mm3': '10^3/uL', '10^3/mm^3': '10^3/uL', '10*9/L': '10^9/L',
    '10*6/L': 'cells/uL', '10*3/mL': 'cells/uL',
}
_COUNT6 = {
    '10*6/uL': '10^6/uL', '10*6/mm3': '10^6/uL', '10*6/mm^3': '10^6/uL',
    '10^6/mm3': '10^6/uL', '10^6/mm^3': '10^6/uL', '10*12/L': '10^12/L',
    '10*6/L': 'cells/uL', '10*3/mL': 'cells/uL',
}
ALIASES = MappingProxyType({field: MappingProxyType(dict(_COUNT6 if field == 'rbc' else _COUNT3) if field in ('rbc','plt','wbc') else {}) for field in CBC_FIELDS})


def canonicalize_cbc_v2(field, value, unit, *, provenance=1):
    original = canonicalize_cbc(field, value, unit, provenance=provenance)
    if original.observed: return original
    mapped = ALIASES[field].get(_token(unit))
    if mapped is None: return original
    return canonicalize_cbc(field, value, mapped, provenance=provenance)
