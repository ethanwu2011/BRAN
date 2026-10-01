"""Bind existing 2022/2023 archive bytes to documented, row-free schemas.

Reads archived file bytes only for hashing; never opens/decompresses a SAS
member, loads a model, or counts participants. No source admission is implied.
"""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/Users/ethanwu/Downloads/KNHANES')
OUT = ROOT/'BRAN_KNHANES_PHASE9_BINDING_V1'
META = 'BRAN_KNHANES_METADATA_V1/metadata.json'
META_PIN = '21ab3dcb68e69d96a92867e95102c40ce887bf8c9b4ba8f34a7c4abaf002cd22'
GUIDE = 'public_documentation/knhanes/KNHANES_9th_2022-2024_raw_data_user_guide.pdf'
GUIDE_PIN = '921818c62267bd2949dd08e7d0143ef8cd30eb3086722e696481aa162ba42ce3'
UNITS = 'PATIENT_ATLAS_OFFICIAL_UNIT_RECONCILIATION_V1.json'
UNITS_PIN = '4d428667185a974116b0f13d525bcf86c9837ea86806604b26f6f0c41184e167'
MAPPINGS = {
    'HE_glu': ('glucose', 'mg/dL', 188),
    'HE_HbA1c': ('hba1c', '%', 188),
    'HE_chol': ('total_cholesterol', 'mg/dL', 188),
    'HE_HDL_st2': ('hdl_cholesterol', 'mg/dL', 188),
    'HE_LDL_drct': ('ldl_cholesterol', 'mg/dL', 188),
    'HE_TG': ('triglycerides', 'mg/dL', 188),
    'HE_BUN': ('bun', 'mg/dL', 189),
    'HE_crea': ('creatinine', 'mg/dL', 189),
    'HE_ast': ('ast_got', 'IU/L', 188),
    'HE_alt': ('alt_got', 'IU/L', 188),
    'HE_hsCRP': ('crp_hs', 'mg/L', 189),
    'HE_ht': ('vit_height_vsorres', 'cm', 187),
    'HE_wt': ('vit_weight_vsorres', 'kg', 187),
    'HE_wc': ('vit_waist_vsorres', 'cm', 187),
    'HE_BMI': ('vit_bmi_vsorres', 'kg/m^2', 187),
    'HE_sbp': ('vit_sysbp_vsorres', 'mmHg', 185),
    'HE_dbp': ('vit_diabp_vsorres', 'mmHg', 185),
}
COMPANIONS = {'HE_alt': 'HE_alt_etc', 'HE_hsCRP': 'HE_hsCRP_etc'}


def require(ok):
    if not ok: raise ValueError('phase9_binding_failed')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bound_path(root, relative, expected):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts)
    path = root/relative
    require(root.is_dir() and not root.is_symlink() and path.is_file()
            and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()))
    require(sha(path) == expected)
    return path


def describe_columns(columns, model_fields):
    names = {c['name'].casefold(): c for c in columns}
    require(len(names) == len(columns))
    for field in ('year', 'age', 'sex', 'HE_prg', 'HE_HB', 'wt_itvex', 'kstrata'):
        require(field.casefold() in names and names[field.casefold()]['type'] == 'numeric')
    for field in ('ID', 'ID_fam', 'psu'):
        require(field.casefold() in names and names[field.casefold()]['type'] == 'text')
    lookup = {f['name']: f for f in model_fields}
    out = []
    for key, (canonical, unit, page) in MAPPINGS.items():
        if key.casefold() not in names:
            continue
        column = names[key.casefold()]
        require(column['type'] == 'numeric')
        model = lookup[canonical]
        require(model['canonical_unit_authorized'] is True and model['canonical_unit'] == unit)
        companion = COMPANIONS.get(key)
        found = None if companion is None else names.get(companion.casefold())
        if found is not None: require(found['type'] == 'text')
        out.append({'source_column': column['name'], 'canonical_name': canonical,
            'canonical_index': model['index'], 'source_unit': unit, 'target_unit': unit,
            'conversion_factor': 1., 'guide_pdf_page': page,
            'companion_column': None if found is None else found['name'],
            'usable_if_row_rules_pass': companion is None or found is not None,
            'provenance': 'released_measurement_summary' if key in ('HE_BMI','HE_sbp','HE_dbp')
                          else 'released_numeric_measurement'})
    return out


def main():
    ok = False; result = None
    try:
        require(not OUT.exists() and not OUT.is_symlink())
        for name, pin in ((META,META_PIN),(GUIDE,GUIDE_PIN),(UNITS,UNITS_PIN)):
            bound_path(ROOT, name, pin)
        meta = json.loads((ROOT/META).read_text())
        require(meta['schema'] == 'bran-knhanes-metadata-v1'
                and meta['status'] == 'completed_metadata_only'
                and meta['patient_rows_decoded'] is False)
        model = json.loads((ROOT/UNITS).read_text())['fields']
        sources = []
        for year in (2022, 2023):
            rows = [r for r in meta['tables'] if r['year'] == year and r['module'] == 'all']
            require(len(rows) == 1)
            row = rows[0]
            require(hashlib.sha256(json.dumps(row['columns'], sort_keys=True,
                    ensure_ascii=False).encode()).hexdigest() == row['schema_sha256'])
            path = bound_path(SOURCE, row['source_file'], row['source_sha256'])
            require(path.suffix.lower() == '.zip' and row['member'] == f'hn{year%100}_all.sas7bdat')
            with zipfile.ZipFile(path) as archive:
                found = [i for i in archive.infolist() if i.filename == row['member']]
                require(len(found) == 1 and not found[0].is_dir() and not found[0].flag_bits & 1)
                require(0 < found[0].file_size <= 2*1024**3)
                # No archive.open(), extraction, or SAS member read occurs.
            sources.append({k: row[k] for k in ('year','source_file','source_sha256','member','schema_sha256')}
                | {'mapping_candidates': describe_columns(row['columns'], model)})
        result = {'schema': 'bran-knhanes-phase9-binding-v1', 'status': 'bytes_and_schemas_bound',
            'sources': sources, 'metadata_sha256': META_PIN, 'guide_sha256': GUIDE_PIN,
            'model_units_sha256': UNITS_PIN, 'patient_rows_decoded': False,
            'patient_counts_emitted': False, 'model_scored': False, 'permission_confirmed': False,
            'source_qualified_for_evaluation': False}
        OUT.mkdir()
        with (OUT/'binding.json').open('x') as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        with (OUT/'manifest.json').open('x') as handle:
            json.dump({'binding_sha256': sha(OUT/'binding.json'),
                'code_sha256': sha(Path(__file__)),
                'test_sha256': sha(ROOT/'test_build_bran_knhanes_phase9_binding_v1.py'),
                'patient_rows_decoded': False}, handle, indent=2, sort_keys=True)
        ok = True
    except Exception:
        pass
    print(json.dumps({'status': 'bytes_and_schemas_bound' if ok else 'failed_closed',
        'source_file_count': 2 if ok else None, 'patient_rows_decoded': False,
        'model_scored': False, 'source_qualified_for_evaluation': False}))
    return 0 if ok else 1


if __name__ == '__main__': raise SystemExit(main())
