"""Read SAS column metadata only; no row reads, extraction, joining or scoring."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import zipfile

import pandas as pd
from pandas.io.sas.sas7bdat import SAS7BDATReader
import organize_knhanes_downloads_v1 as intake

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/Users/ethanwu/Downloads/KNHANES')
OUT = ROOT / 'BRAN_KNHANES_METADATA_V1'
TABLE = re.compile(r'hn(?P<year>\d{2})_(?P<module>all|eye)\.sas7bdat', re.I)
IDENTIFIER = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,127}')
CBC_NAMES = {'hct': 'he_hct', 'hemoglobin': 'he_hb', 'mch': 'he_mch', 'mchc': 'he_mchc',
             'mcv': 'he_mcv', 'plt': 'he_bplt', 'rbc': 'he_rbc', 'rdw': 'he_rdw', 'wbc': 'he_wbc'}
FLAGS = {'patient_rows_decoded': False, 'patient_counts_emitted': False, 'archives_extracted_to_disk': False,
         'data_merged': False, 'model_scored': False, 'model_fitted': False,
         'source_qualified_for_evaluation': False, 'units_confirmed': False, 'permission_confirmed': False}
require = intake.require


@contextmanager
def quiet():
    copies = [os.dup(1), os.dup(2)]; sink = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(sink, 1); os.dup2(sink, 2); yield
    finally:
        os.dup2(copies[0], 1); os.dup2(copies[1], 2)
        for fd in copies + [sink]: os.close(fd)


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def decode(value):
    if isinstance(value, bytes):
        for encoding in ('utf-8', 'cp949'):
            try: return value.decode(encoding)
            except UnicodeDecodeError: pass
        raise ValueError('header_encoding_unresolved')
    require(type(value) is str)
    return value


def member_identity(name):
    path = PurePosixPath(name.replace('\\', '/'))
    require(not path.is_absolute() and '..' not in path.parts and ':' not in name)
    match = TABLE.fullmatch(path.name); require(match is not None)
    year = 2000 + int(match['year']); require(2008 <= year <= 2023)
    return year, match['module'].lower()


def read_metadata(stream, reader_factory=SAS7BDATReader):
    reader = reader_factory(stream, convert_header_text=False)
    try:
        require(reader._current_row_in_file_index == 0)
        columns = []
        for column in reader.columns:
            name = decode(column.name); label = decode(column.label)
            require(IDENTIFIER.fullmatch(name) is not None and len(label) <= 2048)
            kind = decode(column.ctype)
            require(kind in ('d', 's'))
            columns.append({'name': name, 'label': label, 'type': 'numeric' if kind == 'd' else 'text'})
        require(0 < len(columns) <= 5000 and len({c['name'].lower() for c in columns}) == len(columns))
        require(reader._current_row_in_file_index == 0)
        return columns
    finally: reader.close()


def authenticated_files():
    require(SOURCE.is_dir() and not SOURCE.is_symlink())
    with (SOURCE / 'inventory.json').open() as f: inventory = json.load(f)
    with (SOURCE / 'move_receipt.json').open() as f: receipt = json.load(f)
    require(receipt['status'] == 'completed' and receipt['whole_file_hashes_verified'] is True)
    require(intake.digest(SOURCE / 'inventory.json') == receipt['inventory_sha256'])
    require(intake.digest(SOURCE / 'README.md') == receipt['readme_sha256'])
    require(inventory['schema'] == 'knhanes-original-download-inventory-v1' and len(inventory['files']) == 35)
    records = inventory['files']; require(len({r['filename'] for r in records}) == len(records))
    for r in records:
        require(set(r) == {'filename', 'destination', 'bytes', 'sha256'})
        require(str(intake.target(r['filename'])) == r['destination'])
        path = SOURCE / r['destination']; require(path.resolve().is_relative_to(SOURCE.resolve()))
        require(intake.metadata(path)[2] == r['bytes'] and intake.digest(path) == r['sha256'])
    return records, receipt['inventory_sha256']


def profiles(tables):
    grouped = defaultdict(list)
    for t in tables: grouped[(t['year'], t['module'])].append(t)
    out = []
    for (year, module), variants in sorted(grouped.items()):
        schemas = {v['schema_sha256'] for v in variants}
        require(len(schemas) == 1)  # Different versions must not be silently merged.
        columns = variants[0]['columns']; names = {c['name'].lower() for c in columns}
        out.append({'year': year, 'module': module, 'file_variant_count': len(variants),
            'column_count': len(columns), 'schema_sha256': variants[0]['schema_sha256'],
            'candidate_cbc_columns': {f: n for f, n in CBC_NAMES.items() if n in names},
            'has_id_column': 'id' in names, 'has_age_column': 'age' in names,
            'survey_design_columns': [c['name'] for c in columns if c['name'].lower().startswith(('wt_', 'kstrata', 'psu'))]})
    return out


def compute():
    records, pin = authenticated_files(); tables = []; repeats = []; seen = {}
    for r in records:
        if r['sha256'] in seen:
            repeats.append({'file': r['destination'], 'identical_to': seen[r['sha256']]}); continue
        seen[r['sha256']] = r['destination']; path = SOURCE / r['destination']
        sources = []
        if path.suffix.lower() == '.sas7bdat':
            y, m = member_identity(path.name)
            columns = read_metadata(path)
            sources.append({'year': y, 'module': m, 'member': None, 'columns': columns})
        else:
            with zipfile.ZipFile(path) as archive:
                candidates = [i for i in archive.infolist() if i.filename.lower().endswith('.sas7bdat') and not i.is_dir()]
                require(1 <= len(candidates) <= 5)
                for info in candidates:
                    y, m = member_identity(info.filename)
                    require(0 < info.file_size <= 2 * 1024**3 and info.compress_size > 0
                            and info.file_size / info.compress_size < 2500 and not info.flag_bits & 1)
                    with archive.open(info) as stream: columns = read_metadata(stream)
                    sources.append({'year': y, 'module': m, 'member': info.filename, 'columns': columns})
        for t in sources:
            require(str(t['year']) == Path(r['destination']).parts[1])
            tables.append({'source_file': r['destination'], 'source_sha256': r['sha256'],
                           'schema_sha256': json_sha(t['columns']), **t})
        require(intake.digest(path) == r['sha256'])
    rows = profiles(tables)
    return {'schema': 'bran-knhanes-metadata-v1', 'status': 'completed_metadata_only',
            'source_inventory_sha256': pin, 'source_file_count': len(records),
            'byte_identical_archive_duplicates_preserved': repeats,
            'tables': tables, 'year_module_profiles': rows, 'pandas_version': pd.__version__, **FLAGS}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument('--audit', action='store_true'); args = parser.parse_args(argv)
    ok = False; output = None
    with quiet():
        try:
            if args.audit:
                require({p.name for p in OUT.iterdir()} == {'metadata.json', 'manifest.json'})
                manifest = json.loads((OUT / 'manifest.json').read_text())
                require(intake.digest(OUT / 'metadata.json') == manifest['metadata_sha256'])
                require(all(intake.digest(ROOT / n) == h for n, h in manifest['code_sha256'].items()))
                _, pin = authenticated_files(); require(pin == manifest['source_inventory_sha256'])
                value = json.loads((OUT / 'metadata.json').read_text())
                require(value['year_module_profiles'] == profiles(value['tables']))
                require(all(value[k] is v for k, v in FLAGS.items()))
            else:
                require(not OUT.exists()); value = compute(); OUT.mkdir(mode=0o700)
                intake.write_new(OUT / 'metadata.json', value)
                intake.write_new(OUT / 'manifest.json', {'source_inventory_sha256': value['source_inventory_sha256'],
                    'metadata_sha256': intake.digest(OUT / 'metadata.json'), 'code_sha256': {n: intake.digest(ROOT / n) for n in
                    ('audit_knhanes_metadata_v1.py', 'test_audit_knhanes_metadata_v1.py', 'organize_knhanes_downloads_v1.py')}})
            rows = value['year_module_profiles']
            output = {'status': 'completed_metadata_only', 'year_module_profiles': rows,
                      'duplicate_archives_preserved': len(value['byte_identical_archive_duplicates_preserved']), **FLAGS}
            ok = True
        except Exception: pass
    print(json.dumps(output if ok else {'status': 'failed_closed', **FLAGS})); return 0 if ok else 1


if __name__ == '__main__': raise SystemExit(main())
