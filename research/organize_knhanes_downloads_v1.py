"""Organize explicitly scoped local downloads; filenames and whole-file hashes only."""
import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path

DOWNLOADS = Path('/Users/ethanwu/Downloads')
DESTINATION = DOWNLOADS / 'KNHANES'
DOCUMENT = Path(__file__).with_name('KNHANES_DOWNLOADS_README_2026-09-12.md')
NAME = re.compile(r'^hn(?P<year>\d{2})_(?:all(?:\(sas\))?|eye)(?: copy| \(\d+\))?\.(?P<ext>sas7bdat|zip)$', re.I)


def require(value):
    if not value: raise ValueError('knhanes_organization_failed_closed')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def target(name):
    m = NAME.fullmatch(name)
    require(m is not None and Path(name).name == name)
    year = int(m['year']); require(8 <= year <= 23)
    category = 'archives' if m['ext'].lower() == 'zip' else 'sas_tables'
    return Path(category) / str(2000 + year) / name


def metadata(path):
    require(not path.is_symlink())
    s = path.stat(); require(stat.S_ISREG(s.st_mode) and s.st_size > 0)
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def prepare(source, destination):
    require(source.is_dir() and source.resolve() == source and destination.parent == source)
    require(not destination.exists() and not destination.is_symlink())
    records = []
    for path in sorted(source.iterdir()):
        if NAME.fullmatch(path.name) is None: continue
        before = metadata(path); sha = digest(path); require(metadata(path) == before)
        records.append({'filename': path.name, 'destination': str(target(path.name)),
                        'bytes': before[2], 'sha256': sha, '_identity': before})
    require(bool(records))
    return records


def write_new(path, value):
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), 'w') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())


def organize(source=DOWNLOADS, destination=DESTINATION, *, move=False, document=DOCUMENT):
    records = prepare(source, destination)
    summary = {'file_count': len(records), 'total_bytes': sum(r['bytes'] for r in records),
               'years': sorted({str(target(r['filename']).parts[1]) for r in records}),
               'destination': str(destination), 'patient_records_opened': False,
               'archives_extracted': False, 'training_started': False}
    if not move: return {'status': 'planned', **summary}
    require(document.is_file() and not document.is_symlink())
    # Complete the read-only preflight before moving any file. Existing folders
    # are never merged or overwritten by this one-time operation.
    for r in records: require(metadata(source / r['filename']) == r['_identity'])
    destination.mkdir(mode=0o700)
    safe_records = [{k: v for k, v in r.items() if k != '_identity'} for r in records]
    write_new(destination / 'inventory.json', {'schema': 'knhanes-original-download-inventory-v1',
        'source_directory': str(source), 'files': safe_records, **summary})
    with document.open('rb') as src, (destination / 'README.md').open('xb') as dst:
        dst.write(src.read())
    moved = []
    try:
        for r in records:
            src = source / r['filename']; dst = destination / r['destination']
            require(metadata(src) == r['_identity'])
            dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            require(not dst.exists() and not dst.is_symlink())
            # Same-volume exclusive link prevents overwrite; source is unlinked
            # only after the destination's bytes and identity are verified.
            os.link(src, dst, follow_symlinks=False)
            require(metadata(src) == r['_identity'] and metadata(dst) == r['_identity'])
            require(digest(dst) == r['sha256'])
            src.unlink(); moved.append(r['filename'])
        require(all(not (source / r['filename']).exists() for r in records))
        write_new(destination / 'move_receipt.json', {'status': 'completed',
            'inventory_sha256': digest(destination / 'inventory.json'),
            'readme_sha256': digest(destination / 'README.md'),
            'whole_file_hashes_verified': True, 'duplicates_deleted': False, **summary})
    except Exception:
        write_new(destination / 'move_failure.json', {'status': 'partial_move_preserved',
            'moved_filenames': moved, 'patient_records_opened': False})
        raise ValueError('knhanes_organization_failed_closed') from None
    return {'status': 'completed', 'whole_file_hashes_verified': True, 'duplicates_deleted': False, **summary}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--move', action='store_true'); args = parser.parse_args()
    try: result = organize(move=args.move)
    except Exception:
        print(json.dumps({'status': 'failed_closed', 'patient_records_opened': False})); return 1
    print(json.dumps(result)); return 0


if __name__ == '__main__': raise SystemExit(main())
