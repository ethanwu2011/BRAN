"""Closed CBC matching of non-patient dictionaries, not patient observations.

The runner reads only three explicitly named dictionary files. It never opens
patients, admissions, laboratory events, images, embeddings or normalized caches.
Dictionary counts are vocabulary counts, never clinical support counts.
"""
from contextlib import contextmanager
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from bran_clinical_semantics_v1 import CBC_FIELDS, MIMIC_CBC_CODES, canonicalize_cbc

ROOT = Path(__file__).resolve().parent
OUT = 'BRAN_LOCAL_CBC_DICTIONARY_BINDINGS_V1'
SOURCE_PATHS = {
    'mimic': '/Users/ethanwu/mimiciv-3.1/hosp/d_labitems.csv.gz',
    'nwicu': '/Users/ethanwu/nwicu-northwestern-icu-0.1.0/data/nw_hosp/d_labitems.csv.gz',
    'sicdb': '/Users/ethanwu/sicdb-1.0.8/d_references.csv.gz',
}
# Exact names only; no substring, regex assay inference, LOINC guessing, or
# transplantation of one institution's numeric code into another dictionary.
LABELS = {
    'hct': frozenset(('hematocrit', 'hct')),
    'hemoglobin': frozenset(('hemoglobin', 'hgb')),
    'mch': frozenset(('mch', 'mean corpuscular hemoglobin')),
    'mchc': frozenset(('mchc', 'mean corpuscular hemoglobin concentration')),
    'mcv': frozenset(('mcv', 'mean corpuscular volume')),
    'plt': frozenset(('platelet count', 'platelets')),
    'rbc': frozenset(('red blood cells', 'red blood cell count', 'rbc')),
    'rdw': frozenset(('rdw', 'red cell distribution width', 'rdw-cv')),
    'wbc': frozenset(('white blood cells', 'white blood cell count', 'wbc')),
}


def _closed_text(value):
    return value.strip().casefold() if isinstance(value, str) else ''


def _code(value):
    if not isinstance(value, str) or re.fullmatch(r'[0-9]{1,12}', value) is None:
        return None
    return str(int(value))


def match_dictionary(source, rows):
    """Return only canonical names, numeric vocabulary codes and fixed statuses.

    MIMIC: fixed official CBC codes plus local Blood/Hematology/name checks.
    NWICU: independently match full names and Blood/Hematology metadata.
    SICdb: unit-compatible vocabulary candidates only; laboratory type/specimen
    remains unresolved and no candidate is cleared for observation extraction.
    """
    if source not in SOURCE_PATHS:
        raise ValueError('unsupported dictionary source')
    result = {f: {'candidate_codes': [], 'dictionary_bound_codes': []} for f in CBC_FIELDS}
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('invalid dictionary record')
        code = _code(row.get('ReferenceGlobalID' if source == 'sicdb' else 'itemid'))
        if code is None or code in seen:
            raise ValueError('invalid or repeated dictionary code')
        seen.add(code)
        label = _closed_text(row.get('ReferenceValue' if source == 'sicdb' else 'label'))
        if source == 'mimic':
            field = MIMIC_CBC_CODES.get(code)
            fields = [field] if field is not None and label in LABELS[field] else []
        else:
            fields = [field for field in CBC_FIELDS if label in LABELS[field]]
        for field in fields:
            if source == 'sicdb':
                if _closed_text(row.get('ReferenceName')) != 'laboratory':
                    continue
                # Synthetic sentinel tests unit compatibility; no lab value read.
                if not canonicalize_cbc(field, 1., row.get('ReferenceUnit'), provenance=1).observed:
                    continue
                result[field]['candidate_codes'].append(code)
            else:
                result[field]['candidate_codes'].append(code)
                if _closed_text(row.get('fluid')) == 'blood' and _closed_text(row.get('category')) == 'hematology':
                    result[field]['dictionary_bound_codes'].append(code)
    for item in result.values():
        for key in ('candidate_codes', 'dictionary_bound_codes'):
            item[key].sort(key=int)
    return {
        'status': 'dictionary_checked',
        'fields': result,
        'canonical_fields_with_candidates': sum(bool(item['candidate_codes']) for item in result.values()),
        'canonical_fields_dictionary_bound': sum(bool(item['dictionary_bound_codes']) for item in result.values()),
        'observation_units_validated': False,
        'lab_type_or_specimen_pending': source == 'sicdb',
        'patient_rows_read': False,
        'source_training_ready': False,
    }


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect_dictionary(source, path):
    """Bounded dictionary-only read with pre/post hash authentication."""
    path = Path(path)
    if not path.is_file():
        return {'status': 'dictionary_absent', 'patient_rows_read': False, 'source_training_ready': False}
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('dictionary byte limit exceeded')
    before = _sha(path)
    required = ('ReferenceGlobalID', 'ReferenceValue', 'ReferenceUnit') if source == 'sicdb' else ('itemid', 'label', 'fluid')
    optional = ('ReferenceName',) if source == 'sicdb' else ('category',)
    opener = gzip.open if path.suffix == '.gz' else open
    old_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(65536)
        with opener(path, 'rt', encoding='utf-8-sig', newline='') as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader)
            if len(set(header)) != len(header) or any(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', h) is None for h in header):
                raise ValueError('invalid dictionary header')
            if not set(required).issubset(header):
                raise ValueError('dictionary columns absent')
            indices = {name: header.index(name) for name in required + optional if name in header}
            def projected():
                for number, values in enumerate(reader, 1):
                    if number > 200000:
                        raise ValueError('dictionary row limit exceeded')
                    if len(values) != len(header):
                        raise ValueError('malformed dictionary row')
                    yield {name: values[index] for name, index in indices.items()}
            result = match_dictionary(source, projected())
    except Exception:
        raise ValueError('dictionary read or validation failed') from None
    finally:
        csv.field_size_limit(old_limit)
    if _sha(path) != before:
        raise ValueError('dictionary changed during inspection')
    result['dictionary_sha256'] = before
    result['expected_metadata_columns_present'] = {name: name in header for name in required + optional}
    return result


@contextmanager
def _quiet():
    sys.stdout.flush(); sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1); os.dup2(null, 2)
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved_out, 1); os.dup2(saved_err, 2)
        os.close(null); os.close(saved_out); os.close(saved_err)


def build():
    results = {}
    for source, path in SOURCE_PATHS.items():
        try:
            results[source] = inspect_dictionary(source, path)
        except Exception:
            results[source] = {'status': 'dictionary_validation_failed', 'patient_rows_read': False, 'source_training_ready': False}
    results['eicu'] = {'status': 'no_separate_dictionary_bound', 'patient_rows_read': False, 'source_training_ready': False}
    results['nhanes'] = {'status': 'codebook_units_in_adapter_contract', 'patient_rows_read': False, 'source_training_ready': False}
    results['zigong'] = {'status': 'original_source_unresolved', 'patient_rows_read': False, 'source_training_ready': False}
    return {
        'schema_version': 'bran-local-cbc-dictionary-bindings-v1',
        'scope': 'non-patient vocabulary metadata only; no observation eligibility or efficacy',
        'sources': results,
        'source_files': dict(SOURCE_PATHS),
        'code_sha256': {name: _sha(ROOT / name) for name in ('bran_clinical_dictionary_binding_v1.py', 'bran_clinical_semantics_v1.py')},
        'adapter_contract_sha256': _sha(ROOT / 'BRAN_CLINICAL_ADAPTER_CONTRACT_V1.json'),
        'patient_rows_read': False,
        'arbitrary_source_text_emitted': False,
        'training_started': False,
    }


def main():
    out = ROOT / OUT
    # Exclusive directory: never overwrite a previous result or failure receipt.
    try:
        out.mkdir()
    except Exception:
        print(json.dumps({'status': 'output_unavailable_or_exists'}))
        return 1
    success = False
    with _quiet():
        try:
            payload = build()
            with (out / 'bindings.json').open('x') as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            manifest = {'schema_version': 'bran-dictionary-binding-manifest-v1', 'bindings_sha256': _sha(out / 'bindings.json'),
                        'patient_rows_read': False, 'training_started': False}
            with (out / 'manifest.json').open('x') as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            success = True
        except Exception:
            with (out / 'failure.json').open('x') as handle:
                json.dump({'status': 'dictionary_runner_failed', 'training_started': False}, handle)
    print(json.dumps({'status': 'dictionary_binding_completed' if success else 'dictionary_runner_failed', 'patient_rows_read': False}))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
