"""Arrow-first, quiet HiRID raw-partition reader.

This preserves the raw-reader contract while avoiding Python materialization of
raw rows whose variable IDs cannot affect the observation selector.
"""
from collections.abc import Mapping
from pathlib import Path

from bran_hirid_raw_reader_v1 import (
    ADMITTED_IDS,
    COLUMNS,
    ERROR,
    PrivateEpisodeSelections,
    _hash_file,
    require,
    select_partition_rows,
)
from bran_hirid_v5_observation_kernel import select_episode


MAX_ROWS = 100000000


def _path(path, digest):
    require(type(path) is Path or isinstance(path, Path))
    require(path.is_file() and not path.is_symlink() and type(digest) is str
            and len(digest) == 64 and set(digest) <= set('0123456789abcdef'))
    require(_hash_file(path) == digest)


def _schema(schema, pa):
    require(len(schema.names) == len(set(schema.names)) and set(schema.names) == set(COLUMNS))
    for name in ('patientid', 'variableid', 'status'):
        require(pa.types.is_integer(schema.field(name).type))
    for name in ('datetime', 'entertime'):
        kind = schema.field(name).type
        require(pa.types.is_timestamp(kind) and kind.tz is None)
    require(pa.types.is_floating(schema.field('value').type))
    for name in ('stringvalue', 'type'):
        require(pa.types.is_string(schema.field(name).type) or pa.types.is_large_string(schema.field(name).type))


def _all(values, pc):
    """Require every boolean Arrow value to be true without row conversion."""
    result = pc.all(values).as_py()
    return bool(result)


def read_partition(path, expected_sha256, admissions):
    """Read one authenticated partition with Arrow ID checks and filtering.

    Patient and variable IDs are checked for every raw row.  Only rows whose
    variable ID is admitted by the existing selector are converted to Python;
    membership is collected separately so an admission with no useful assay is
    retained as the selector's normal no-target denominator.
    """
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        require(isinstance(admissions, Mapping))
        _path(path, expected_sha256)
        reader = pq.ParquetFile(path)
        _schema(reader.schema_arrow, pa)

        # Admission identities are checked once after complete partition
        # membership has been collected.  This avoids rebuilding Arrow's
        # membership hash table for every raw batch.
        allowed_variables = pa.array(tuple(ADMITTED_IDS))
        membership = set()
        row_count = 0
        def rows():
            nonlocal row_count
            for batch in reader.iter_batches(batch_size=8192, columns=list(COLUMNS), use_threads=False):
                row_count += batch.num_rows
                require(row_count <= MAX_ROWS)
                patient_ids = batch.column(batch.schema.get_field_index('patientid'))
                variable_ids = batch.column(batch.schema.get_field_index('variableid'))
                require(not bool(pc.any(pc.is_null(patient_ids)).as_py()))
                require(not bool(pc.any(pc.is_null(variable_ids)).as_py()))
                require(_all(pc.greater_equal(patient_ids, 0), pc))
                require(_all(pc.greater_equal(variable_ids, 0), pc))
                # This is deliberately ID-only conversion: it records partition
                # membership even where filtering leaves no assay row to select.
                membership.update(int(value) for value in pc.unique(patient_ids).to_pylist())
                selector_ids = pc.is_in(variable_ids, value_set=allowed_variables)
                if bool(pc.any(selector_ids).as_py()):
                    yield from pc.filter(batch, selector_ids).to_pylist()

        selected = select_partition_rows(rows(), admissions)
        require(membership.issubset(admissions))
        by_id = dict(zip(selected.admission_ids, selected.selections, strict=True))
        ids = tuple(sorted(membership))
        result = PrivateEpisodeSelections(
            ids,
            tuple(by_id[identifier] if identifier in by_id else select_episode(()) for identifier in ids),
        )
        _path(path, expected_sha256)
        return result
    except Exception:
        raise ValueError(ERROR) from None
