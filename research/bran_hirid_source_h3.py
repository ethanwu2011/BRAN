"""Approved quiet-caller-only source authentication; never logs source contents."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

from bran_hirid_raw_reader_v1 import _hash_file, SCHEMA_PDF_SHA256

ERROR = 'hirid_h3_source_authentication_failed'
METADATA_PIN = '667f518ddc6792e3e56b6fd619d972a41daf82394bd0a28bf0f18f10929179fc'
CHECKSUM_PIN = '61abaa6174ef885fea91ccbb4a95d18b7ce9aa6a3bec0c7b4271003f3286873b'
DICTIONARY_PIN = '7f3095dbfe15512c8dd313ce83a6613e18762f0e364b7fbcd0ed177ff9de5f8f'
REFERENCE_PIN = '33670c9eac871e607174a28b9a943c2e2db0f0b885c01eddd9b1ddce39a1f4e5'
SOURCE = Path('/Volumes/Extreme/hirid-1.1.1')
PARTITIONS = Path('ext/observation_tables/parquet')
ARCHIVE = Path('raw_stage/observation_tables_parquet.tar.gz')


def require(ok):
    if not ok:
        raise ValueError(ERROR) from None


def regular(path):
    require(path.is_file() and not path.is_symlink())
    require(not any(p.is_symlink() for p in path.parents))


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def stream_hash(handle):
    h = hashlib.sha256()
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def checksums(path):
    regular(path)
    require(_hash_file(path) == CHECKSUM_PIN)
    result = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})\s+\*?(.+)', line)
        require(match is not None)
        digest, name = match.groups()
        name = str(PurePosixPath(name))
        require(not name.startswith('/') and '..' not in PurePosixPath(name).parts
                and name not in result)
        result[name] = digest
    return result


def authenticate_reference(root, published):
    reference = root / 'reference_data.tar.gz'
    dictionary = root / 'ref/hirid_variable_reference.csv'
    general = root / 'ref/general_table.csv'
    schema = root / 'schemata.pdf'
    for path in (reference, dictionary, general, schema):
        regular(path)
    require(published['reference_data.tar.gz'] == REFERENCE_PIN
            and _hash_file(reference) == REFERENCE_PIN
            and _hash_file(dictionary) == DICTIONARY_PIN
            and _hash_file(schema) == SCHEMA_PDF_SHA256)
    desired = {'general_table.csv': general, 'hirid_variable_reference.csv': dictionary}
    pins = {}
    with tarfile.open(reference, 'r:gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name).name
            if name not in desired:
                continue
            require(member.isfile() and name not in pins)
            handle = archive.extractfile(member)
            require(handle is not None)
            with handle:
                digest = stream_hash(handle)
            require(digest == _hash_file(desired[name]))
            pins[name] = digest
    require(set(pins) == set(desired) and _hash_file(reference) == REFERENCE_PIN)
    return {'reference_archive_sha256': REFERENCE_PIN,
            'general_table_sha256': pins['general_table.csv'],
            'dictionary_sha256': DICTIONARY_PIN, 'schema_sha256': SCHEMA_PDF_SHA256}


def authenticate_source(root=SOURCE, progress=lambda phase, n, total: None):
    """Full archive/extracted-byte equivalence. No patient rows are decoded."""
    try:
        require(root.is_dir() and not root.is_symlink())
        published = checksums(root / 'SHA256SUMS.txt')
        progress('reference_authentication', 0, 0)
        reference = authenticate_reference(root, published)
        path = root / ARCHIVE
        regular(path)
        archive_pin = published[str(ARCHIVE)]
        progress('archive_authentication', 0, 0)
        require(_hash_file(path) == archive_pin)
        folder = root / PARTITIONS
        require(folder.is_dir() and not folder.is_symlink())
        files = {p.name: p for p in folder.iterdir() if p.suffix == '.parquet'}
        require(bool(files) and all(re.fullmatch(r'[A-Za-z0-9_.-]+\.parquet', n) for n in files))
        for file in files.values():
            regular(file)
        pins, auxiliary, seen_members = {}, {}, set()
        with tarfile.open(path, 'r|gz', bufsize=1024*1024) as archive:
            for member in archive:
                if member.isdir():
                    continue
                member_path = PurePosixPath(member.name)
                normalized = str(member_path)
                require(member.isfile() and not member_path.is_absolute()
                        and '..' not in member_path.parts and normalized not in seen_members)
                seen_members.add(normalized)
                name = member_path.name
                # The release archive contains non-Parquet auxiliary material.
                # Hash it, without decoding or using it as a data source. It is
                # not an extra observation partition. Unknown Parquet files
                # still fail; the complete Parquet inventory stays exact.
                is_partition = name.lower().endswith('.parquet')
                if is_partition:
                    require(name in files and name not in pins
                            and member.size == files[name].stat().st_size)
                handle = archive.extractfile(member)
                require(handle is not None)
                with handle:
                    digest = stream_hash(handle)
                if not is_partition:
                    auxiliary[normalized] = {'sha256': digest, 'bytes': member.size}
                    continue
                require(digest == _hash_file(files[name]))
                pins[name] = digest
                if len(pins) % 25 == 0 or len(pins) == len(files):
                    progress('partition_bytes_authentication', len(pins), len(files))
        require(set(pins) == set(files) and len(set(pins.values())) == len(pins)
                and _hash_file(path) == archive_pin)
        return {'schema': 'bran-hirid-source-bytes-h3', 'source': 'HiRID', 'version': '1.1.1',
                'checksum_manifest_sha256': CHECKSUM_PIN, **reference,
                'raw_archive_sha256': archive_pin,
                'partitions': dict(sorted(pins.items())),
                'auxiliary_archive_file_count': len(auxiliary),
                'auxiliary_archive_inventory_sha256': content_hash(auxiliary),
                'auxiliary_members_used_as_data': False,
                'partition_inventory_sha256': hashlib.sha256(
                    '\n'.join(sorted(pins.values())).encode('ascii')).hexdigest(),
                'raw_only': True, 'complete_archive_matched': True,
                'patient_level_output_emitted': False}
    except Exception:
        raise ValueError(ERROR) from None


def recheck_source(receipt, root=SOURCE):
    """Check full extracted bytes before final acceptance; no raw row output."""
    try:
        published = checksums(root / 'SHA256SUMS.txt')
        require(published[str(ARCHIVE)] == receipt['raw_archive_sha256'])
        require(authenticate_reference(root, published) == {
            k: receipt[k] for k in ('reference_archive_sha256', 'general_table_sha256',
                                  'dictionary_sha256', 'schema_sha256')})
        files = {p.name: p for p in (root / PARTITIONS).iterdir() if p.suffix == '.parquet'}
        require(set(files) == set(receipt['partitions']))
        for name, pin in receipt['partitions'].items():
            regular(files[name])
            require(_hash_file(files[name]) == pin)
        return True
    except Exception:
        raise ValueError(ERROR) from None
