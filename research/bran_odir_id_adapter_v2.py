"""Lossless ID serialization repair and private ungrouped-row quarantine."""
from dataclasses import asdict, dataclass
import re

import bran_odir_metadata_kernel_v1 as k

FIELDS = ('ID', 'Patient Age', 'Patient Sex', 'Left-Fundus', 'Right-Fundus', *tuple('NDGCAHMO'))
ID = re.compile(r'([0-9]+)(?:\.0+)?')
MISSING = frozenset(('', 'na', 'nan', 'unknown'))
COUNT_KEYS = ('source_rows', 'identified_rows', 'lossless_decimal_id_rows', 'ungroupable_missing_id_rows')


def require(value):
    if not value: raise ValueError('ODIR ID adapter rejected input') from None


@dataclass(frozen=True, repr=False)
class PreparedRows:
    identified: tuple[dict, ...]
    source_row_bindings: tuple[dict, ...]
    ungroupable: tuple[dict, ...]


def prepare_rows(rows):
    require(type(rows) is list and rows)
    identified, bindings, ungroupable = [], [], []
    decimal = 0
    for index, row in enumerate(rows):
        require(type(row) is dict and set(row) == set(FIELDS))
        value = row['ID']; require(type(value) is str)
        if value.strip().lower() in MISSING:
            ungroupable.append({'source_row_index': index, 'reason': 'missing_patient_id', 'metadata': dict(row)})
            continue
        match = ID.fullmatch(value)
        require(match is not None)
        canonical = match.group(1)  # No float conversion or leading-zero removal.
        copy = dict(row); copy['ID'] = canonical
        identified.append(copy)
        bindings.append({'source_row_index': index, 'original_id_serialization': value, 'patient_id': canonical})
        decimal += value != canonical
    counts = {'source_rows': len(rows), 'identified_rows': len(identified),
              'lossless_decimal_id_rows': decimal, 'ungroupable_missing_id_rows': len(ungroupable)}
    summary = {'schema': 'bran-odir-id-adapter-v2',
               'counts_rounded_down20': {name: 20*(n//20) for name, n in counts.items()},
               'synthetic_patient_ids_created': False, 'missing_id_rows_used_for_training': False,
               'patient_level_output_emitted': False}
    validate_summary(summary)
    return PreparedRows(tuple(identified), tuple(bindings), tuple(ungroupable)), summary


def validate_summary(value):
    require(type(value) is dict and set(value) == {'schema', 'counts_rounded_down20',
            'synthetic_patient_ids_created', 'missing_id_rows_used_for_training', 'patient_level_output_emitted'})
    require(value['schema'] == 'bran-odir-id-adapter-v2')
    counts = value['counts_rounded_down20']
    require(type(counts) is dict and set(counts) == set(COUNT_KEYS)
            and all(type(n) is int and n >= 0 and n % 20 == 0 for n in counts.values()))
    require(counts['source_rows'] - counts['identified_rows'] - counts['ungroupable_missing_id_rows'] in (0, 20)
            and counts['lossless_decimal_id_rows'] <= counts['identified_rows'])
    require(all(value[name] is False for name in ('synthetic_patient_ids_created',
            'missing_id_rows_used_for_training', 'patient_level_output_emitted')))


def qualify(rows, members):
    prepared, ingestion = prepare_rows(rows)
    require(prepared.identified)
    records, metadata = k.qualify(list(prepared.identified), members)
    private = {'grouped': asdict(records), 'source_row_bindings': prepared.source_row_bindings,
               'ungroupable_rows': prepared.ungroupable}
    public = {'schema': 'bran-odir-qualification-v2', 'id_ingestion': ingestion,
              'identified_record_qualification': metadata, 'training_admitted': False,
              'patient_level_output_emitted': False}
    validate_aggregate(public)
    return private, public


def validate_aggregate(value):
    require(type(value) is dict and set(value) == {'schema', 'id_ingestion',
            'identified_record_qualification', 'training_admitted', 'patient_level_output_emitted'})
    require(value['schema'] == 'bran-odir-qualification-v2'
            and value['training_admitted'] is False and value['patient_level_output_emitted'] is False)
    validate_summary(value['id_ingestion']); k.validate_aggregate(value['identified_record_qualification'])
    require(value['id_ingestion']['counts_rounded_down20']['identified_rows'] ==
            value['identified_record_qualification']['counts_rounded_down20']['input_rows'])
