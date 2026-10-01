"""Local-only XPORT projection and cycle-specific NHANES person linkage.

Unlike the earlier header reader, this is a real observation reader. Never call
on real files outside a reviewed runner with OS stdout/stderr suppression.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
import re
from types import MappingProxyType

from bran_clinical_semantics_v1 import decode_age


@dataclass(frozen=True, repr=False)
class NHANESCycleLinks:
    cycle: str
    person_age: Mapping


def _cycle(cycle):
    if cycle not in ('D', 'E'):
        raise ValueError('unsupported NHANES cycle')
    return 4 if cycle == 'D' else 5


def nhanes_numeric_key(value):
    # XPORT stores numeric identifiers as doubles. Admit integer-valued doubles
    # only in the exact-integer range; never truncate decimals or parse an ID
    # through floating point when it was originally a string.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('invalid NHANES numeric identifier')
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError('invalid NHANES numeric identifier') from None
    if not math.isfinite(number) or not number.is_integer() or number <= 0 or number > 2**53 - 1:
        raise ValueError('invalid NHANES numeric identifier')
    return str(int(number))


def iter_xport_projection(path, columns, *, max_rows):
    """Project caller-specified columns from bounded local XPORT chunks."""
    if not isinstance(columns, tuple) or not columns or any(not isinstance(c,str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',c) for c in columns) or len(set(columns)) != len(columns):
        raise ValueError('invalid XPORT projection')
    if type(max_rows) is not int or max_rows <= 0:
        raise ValueError('invalid XPORT row bound')
    try:
        import pandas as pd
        with pd.read_sas(path, format='xport', iterator=True, chunksize=512) as reader:
            count = 0
            for frame in reader:
                if not frame.columns.is_unique or not set(columns).issubset(frame.columns):
                    raise ValueError('invalid XPORT schema')
                # Do not yield full frames, source labels, indices or other fields.
                for values in frame.loc[:,list(columns)].itertuples(index=False, name=None):
                    count += 1
                    if count > max_rows:
                        raise ValueError('XPORT bounded-read limit exceeded')
                    yield dict(zip(columns, values))
    except Exception:
        raise ValueError('XPORT read or validation failed') from None


def link_nhanes_cycle(cycle, demographic_rows, cbc_rows):
    """Require unique one-to-one SEQN linkage within an authenticated file cycle.

    Some demographic participants have no CBC row; they are not manufactured
    into examinations. A CBC row is not evidence of any finite CBC analyte.
    """
    release = _cycle(cycle)
    demographics = {}
    for row in demographic_rows:
        if not isinstance(row, Mapping) or not {'SEQN','SDDSRVYR','RIDAGEYR'}.issubset(row):
            raise ValueError('invalid NHANES demographic linkage')
        key = nhanes_numeric_key(row['SEQN'])
        if key in demographics:
            raise ValueError('duplicate NHANES demographic person')
        if nhanes_numeric_key(row['SDDSRVYR']) != str(release):
            raise ValueError('NHANES cycle mismatch')
        demographics[key] = decode_age('nhanes', row['RIDAGEYR'], cycle=cycle)
    linked = {}
    for row in cbc_rows:
        if not isinstance(row, Mapping) or 'SEQN' not in row:
            raise ValueError('invalid NHANES CBC linkage')
        key = nhanes_numeric_key(row['SEQN'])
        if key in linked:
            raise ValueError('duplicate NHANES CBC person')
        if key not in demographics:
            raise ValueError('orphan NHANES CBC person')
        linked[key] = demographics[key]
    return NHANESCycleLinks(cycle, MappingProxyType(linked))
