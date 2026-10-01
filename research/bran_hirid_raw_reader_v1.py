"""Quiet-caller-only HiRID raw observation reader and published quality filter.

No source is admitted by this module. A caller must establish study approval,
source completeness/exposure and schema provenance before real-data access.
Only raw Parquet observations and reference general-table CSV are accepted;
merged/imputed stages cannot satisfy this interface. Nothing is printed.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType

from bran_hirid_v5_observation_kernel import Event, INPUT_CANDIDATES, TARGET_IDS, select_episode

ERROR = 'hirid_raw_reader_v1_contract_failed'
SCHEMA_PDF_SHA256 = '0ccbc6f55c970ce8f2d43ef4ed598790e58b59d6c6ac654f28263b7bfb7644c0'
COLUMNS = ('patientid', 'datetime', 'entertime', 'status', 'stringvalue', 'type', 'value', 'variableid')
GENERAL_COLUMNS = ('patientid', 'admissiontime', 'sex', 'age', 'discharge_status')
# Published raw-schema status bits. Unknown bits fail closed for that assay.
KNOWN_STATUS = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 1024
INVALID_STATUS = 1 | 2 | 32 | 64 | 128
ADMITTED_IDS = frozenset((*TARGET_IDS, *INPUT_CANDIDATES))


def require(ok):
    if not ok:
        raise ValueError(ERROR) from None


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _path(path, digest):
    require(type(path) is Path or isinstance(path, Path))
    require(path.is_file() and not path.is_symlink() and type(digest) is str
            and len(digest) == 64 and set(digest) <= set('0123456789abcdef'))
    require(_hash_file(path) == digest)


def _time(value):
    if isinstance(value, datetime):
        parsed = value
    elif type(value) is str:
        parsed = datetime.fromisoformat(value)
    else:
        raise ValueError(ERROR)
    require(parsed.tzinfo is None)
    return parsed


def _id(value):
    if isinstance(value, Integral) and not isinstance(value, bool):
        result = int(value)
    elif type(value) is str and value.isascii() and value.isdigit():
        result = int(value)
    else:
        raise ValueError(ERROR)
    require(result >= 0)
    return result


@dataclass(frozen=True, slots=True, repr=False)
class PrivateEpisodeSelections:
    """Admission-local handles and selections; never a public report."""
    admission_ids: tuple
    selections: tuple

    def __repr__(self):
        return '<PrivateHiRIDRawSelections>'

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)


class PrivateAdmissions(Mapping):
    """Read-only local join handles whose repr cannot disclose identities."""
    __slots__ = ('_values',)

    def __init__(self, values):
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return '<PrivateHiRIDAdmissions>'

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)


def read_admissions(path, expected_sha256):
    """Read only admission identity/time; ignore age, sex and discharge values."""
    try:
        _path(path, expected_sha256)
        values = {}
        with path.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames))
                    and set(reader.fieldnames) == set(GENERAL_COLUMNS))
            for i, row in enumerate(reader):
                require(i < 1000000 and None not in row)
                key = _id(row['patientid'])
                require(key not in values)
                values[key] = _time(row['admissiontime'])
        _path(path, expected_sha256)
        return PrivateAdmissions(values)
    except Exception:
        raise ValueError(ERROR) from None


def qualified_event(row, admission):
    """Return (observed event, entry hours) or None; no inferred validity.

    F and C are the documented final/corrected laboratory result types. A
    correction conflicting with another result remains a conflict in the
    unchanged selector; it does not overwrite a target or select a nicer value.
    Preliminary, invalidated, not-measured, censored, unrecognized-status and
    nonnumeric/string-valued assays are not original-precision truth.
    """
    try:
        require(isinstance(row, Mapping) and set(row) == set(COLUMNS))
        variable = _id(row['variableid'])
        if variable not in ADMITTED_IDS:
            return None
        status = row['status']
        if not isinstance(status, Integral) or isinstance(status, bool) or status < 0:
            return None
        if status & ~KNOWN_STATUS or status & INVALID_STATUS:
            return None
        if row['type'] not in ('F', 'C') or row['stringvalue'] not in (None, ''):
            return None
        value = row['value']
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            return None
        try:
            origin = _time(admission)
            observed = (_time(row['datetime']) - origin).total_seconds() / 3600.
            entered = (_time(row['entertime']) - origin).total_seconds() / 3600.
        except (ValueError, TypeError, OverflowError):
            return None
        if not 0 <= observed <= entered or observed > 24:
            return None
        return Event(variable, observed, float(value), True), entered
    except Exception:
        raise ValueError(ERROR) from None


def select_partition_rows(rows, admissions):
    """Read a complete raw partition into private, admission-grouped selections.

    The target is retrospective assay estimation, anchored on measurement time.
    Target finalization may occur later; predictors must have entered the
    database by the target measurement time, not merely carry an earlier date.
    The caller verifies partition completeness/nonoverlap before cohort use.
    """
    try:
        require(isinstance(admissions, Mapping))
        grouped = {}
        for i, row in enumerate(rows):
            require(i < 100000000 and isinstance(row, Mapping) and set(row) == set(COLUMNS))
            key = _id(row['patientid'])
            require(key in admissions)
            grouped.setdefault(key, [])
            value = qualified_event(row, admissions[key])
            if value is not None:
                grouped[key].append(value)
        ids, selections = [], []
        for key in sorted(grouped):
            candidates = grouped[key]
            targets = tuple(event for event, _ in candidates if event.variable_id in TARGET_IDS)
            anchor = select_episode(targets)
            if anchor.status in ('abstain_no_target', 'target_conflict'):
                selected = anchor
            else:
                inputs = tuple(event for event, entered in candidates
                    if event.variable_id in INPUT_CANDIDATES and entered <= anchor.time_hours)
                selected = select_episode(targets + inputs)
            ids.append(key)
            selections.append(selected)
        return PrivateEpisodeSelections(tuple(ids), tuple(selections))
    except Exception:
        raise ValueError(ERROR) from None


def read_partition(path, expected_sha256, admissions):
    """Read one authenticated raw Parquet file, never patient-valued metadata."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        _path(path, expected_sha256)
        reader = pq.ParquetFile(path)
        schema = reader.schema_arrow
        require(len(schema.names) == len(set(schema.names)) and set(schema.names) == set(COLUMNS))
        for name in ('patientid', 'variableid', 'status'):
            require(pa.types.is_integer(schema.field(name).type))
        for name in ('datetime', 'entertime'):
            kind = schema.field(name).type
            require(pa.types.is_timestamp(kind) and kind.tz is None)
        require(pa.types.is_floating(schema.field('value').type))
        for name in ('stringvalue', 'type'):
            require(pa.types.is_string(schema.field(name).type) or pa.types.is_large_string(schema.field(name).type))
        def rows():
            for batch in reader.iter_batches(batch_size=8192, columns=list(COLUMNS), use_threads=False):
                yield from batch.to_pylist()
        result = select_partition_rows(rows(), admissions)
        _path(path, expected_sha256)
        return result
    except Exception:
        raise ValueError(ERROR) from None
