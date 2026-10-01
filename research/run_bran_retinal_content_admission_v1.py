"""Current-image/overlap qualification. Every patient operation is FD-quiet."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import glob
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import os

import numpy as np
from PIL import Image
import patient_atlas_eye_source_provenance as historical
import run_bran_retinal_group_readiness_v1 as r
import run_bran_retinal_manifest_linkage_diagnostic_v1 as linkage
from bran_retinal_hash_match_v1 import PrivateHashIndex
from bran_retinal_content_decode_v1 import decode_dicom128, decode_jpeg

ROOT = r.ROOT
BRSET = Path('/Users/ethanwu/brazilian-ophthalmological-1.0.1/fundus_photos')
AI = Path('/Volumes/Extreme/AIREADI_raw/aireadi-container/849e8f67-8355-4ded-a48e-019320371848/dataset')
AI_MANIFEST = AI / 'retinal_photography/manifest.tsv'
CLINICAL = Path('/Users/ethanwu/clinical-world-model')
PROTOCOL = ROOT / 'BRAN_RETINAL_CONTENT_ADMISSION_PROTOCOL_V1.json'
OUT = ROOT / 'BRAN_RETINAL_CONTENT_ADMISSION_V1'
AUDIT = ROOT / 'BRAN_RETINAL_CONTENT_ADMISSION_AUDIT_V1'
PRIVATE = ROOT / 'private_artifacts/bran_retinal_content_admission_v1'
LOCK = r.LOCK
N_AI = 50315
FILES = ('run_bran_retinal_content_admission_v1.py', 'test_run_bran_retinal_content_admission_v1.py',
         'bran_retinal_hash_match_v1.py', 'test_bran_retinal_hash_match_v1.py',
         'bran_retinal_content_decode_v1.py', 'test_bran_retinal_content_decode_v1.py',
         'BRAN_RETINAL_CONTENT_ADMISSION_DESIGN_V1.md',
         'run_bran_retinal_manifest_linkage_diagnostic_v1.py', 'patient_atlas_eye_source_provenance.py')
SCHEMA = 'bran-retinal-content-admission-v1'
PHASES = ('authentication', 'inventory', 'pixel_preflight', 'brset_content',
          'ai_readi_content', 'quarantine', 'publication', 'audit_raw_bytes', 'audit_pixels')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def coarse(n):
    return linkage.coarse(n)


def upstream():
    r.authenticate_audit(linkage.PROTOCOL_PIN, linkage.AUDIT_PIN)
    r.require(r.sha(linkage.REPORT) == linkage.REPORT_PIN and r.sha(linkage.MANIFEST) == linkage.MANIFEST_PIN)
    old = json.loads(linkage.REPORT.read_text())
    r.require(old['bindings']['audit_code_sha256'] == r.sha(ROOT / 'patient_atlas_eye_source_provenance.py'))
    r.require(old['bindings']['ai_readi_decoder_sha256'] == r.sha(CLINICAL / 'encode_aireadi.py'))
    r.require(r.sha(linkage.OUTPUT) == 'bbde3c842540f993701e0595ad31580cbb39ed2d15a4d73460a1d184397b2992')
    return {'metadata_protocol_sha256': linkage.PROTOCOL_PIN, 'metadata_audit_sha256': linkage.AUDIT_PIN,
            'historical_manifest_sha256': linkage.MANIFEST_PIN, 'historical_report_sha256': linkage.REPORT_PIN,
            'historical_decoder_sha256': r.sha(CLINICAL / 'encode_aireadi.py'),
            'ai_source_manifest_sha256': r.sha(AI_MANIFEST)}


def stat_record(path, root):
    r.require(path.is_file() and path.resolve().is_relative_to(root.resolve()))
    s = path.stat()
    r.require(s.st_size > 0)
    return {'relative_path': path.relative_to(root).as_posix(), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns}


def ai_inventory():
    paths = sorted(Path(x) for x in glob.glob(str(AI / 'retinal_photography/cfp/**/*.dcm'), recursive=True))
    r.require(len(paths) == N_AI)
    rows = [stat_record(path, AI) for path in paths]
    names = [x['relative_path'] for x in rows]
    r.require(len(set(names)) == N_AI)
    declared = []
    with AI_MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t', strict=True)
        r.require(reader.fieldnames is not None and 'filepath' in reader.fieldnames)
        for row in reader:
            path = row['filepath'].replace('\\', '/').lstrip('./')
            marker = path.find('retinal_photography/')
            if marker >= 0:
                path = path[marker:]
            if path.startswith('retinal_photography/cfp/'):
                declared.append(path)
    r.require(len(declared) == len(set(declared)) == N_AI and set(declared) == set(names))
    return rows


def private_json(path, value):
    with path.open('x') as f:
        os.fchmod(f.fileno(), 0o600)
        json.dump(value, f, sort_keys=True, separators=(',', ':'), allow_nan=False)


def private_npz(path, arrays):
    with path.open('xb') as f:
        os.fchmod(f.fileno(), 0o600)
        np.savez_compressed(f, **arrays)


def runtime():
    return {**r.runtime(), **{name: importlib.metadata.version(name) for name in ('Pillow', 'pydicom', 'scipy')}}


def expected_protocol():
    return {'schema': SCHEMA, 'upstream': upstream(),
        'code_sha256': {name: r.sha(ROOT / name) for name in FILES}, 'runtime': runtime(),
        'ai_inventory_sha256': r.sha(PRIVATE / 'ai_inventory.json'),
        'parameters': {'ai_images': N_AI, 'hash_image_size': 128, 'new_retinal_image_size': 224,
            'hamming_radius_each_hash': 4, 'decode_workers': 4, 'window': 128, 'replay_images_per_source': 32,
            'quarantine_whole_person': True, 'disease_labels_used_to_quarantine': False,
            'patient_level_output_permitted': False, 'training_permitted': False}}


def load_protocol(pin):
    r.regular(PROTOCOL)
    r.require(type(pin) is str and len(pin) == 64 and r.sha(PROTOCOL) == pin)
    r.regular(PRIVATE / 'ai_inventory.json', private=True)
    p = json.loads(PROTOCOL.read_text())
    r.require(r.equal(p, expected_protocol()))
    return p


def prepare(state):
    r.absent(PROTOCOL, OUT, AUDIT, PRIVATE)
    upstream()
    state['phase'] = 'inventory'
    records = ai_inventory()
    PRIVATE.mkdir(mode=0o700)
    private_json(PRIVATE / 'ai_inventory.json', records)
    r.write_json(PROTOCOL, expected_protocol())
    return r.sha(PROTOCOL)


def selected_metadata():
    with np.load(r.PRIVATE / 'assignment.npz', allow_pickle=False) as data:
        m = {key: data[key] for key in r.ARRAYS}
    with np.load(linkage.MANIFEST, allow_pickle=False) as data:
        names = data['logical_relative_path_sha256']
        source = data['source_code']
        rows = np.flatnonzero(source == 1)
        by_name = {bytes(names[i]): int(i) for i in rows}
        selected = np.flatnonzero(m['adult_eligible'])
        old_rows = np.asarray([by_name[hashlib.sha256(('brset/' + str(m['image_ids'][i]) + '.jpg').encode()).digest()]
                              for i in selected], dtype=np.int64)
        ref = {key: data[key][old_rows] for key in ('raw_file_sha256', 'file_size', 'decoded_128_rgb_sha256',
                                                   'perceptual_hash64', 'difference_hash64')}
    return m, selected, ref


def stable_bytes(record, root):
    path = root / record['relative_path']
    r.require(stat_record(path, root) == record)
    data = path.read_bytes()
    r.require(len(data) == record['size'] and stat_record(path, root) == record)
    return data


def hash_pixels(pixels):
    r.require(type(pixels) is np.ndarray and pixels.dtype == np.uint8 and pixels.shape == (128, 128, 3))
    ph, dh = historical._perceptual_hashes(pixels)
    return np.frombuffer(hashlib.sha256(pixels.tobytes()).digest(), dtype=np.uint8).copy(), np.uint64(ph), np.uint64(dh)


def sampled(n):
    return np.unique(np.linspace(0, n - 1, min(32, n), dtype=np.int64)).tolist()


def preflight(ai_rows):
    old_decoder, _ = historical._load_ai_decode(CLINICAL)
    for i in sampled(len(ai_rows)):
        data = stable_bytes(ai_rows[i], AI)
        actual = decode_dicom128(data, digest(data))
        with io.BytesIO(data) as buffer:
            legacy = old_decoder(buffer, 128)
            try:
                with legacy.resize((128, 128), Image.Resampling.BICUBIC) as resized:
                    expected = np.asarray(resized, dtype=np.uint8).copy()
            finally:
                legacy.close()
        r.require(np.array_equal(actual, expected))


def progress(state, phase, done=0, total=0):
    r.require(phase in PHASES)
    state['phase'] = phase
    value = {'phase': phase, 'completed_work_coarsened': coarse(done), 'total_work_coarsened': coarse(total)}
    destination = state.get('owned')
    if destination is not None:
        tmp = destination / 'progress.tmp'
        r.absent(tmp)
        r.write_json(tmp, value)
        os.replace(tmp, destination / 'progress.json')


def validate_progress(value, terminal=False):
    r.require(type(value) is dict and set(value) == {'phase', 'completed_work_coarsened', 'total_work_coarsened'})
    r.require(type(value['phase']) is str and value['phase'] in PHASES)
    r.require(all(r.kernel._valid_coarse(value[k]) for k in ('completed_work_coarsened', 'total_work_coarsened')))
    if terminal:
        r.require(value == {'phase': 'publication', 'completed_work_coarsened': 0, 'total_work_coarsened': 0})


def map_rows(function, n, state, phase):
    values = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, n, 128):
            values.extend(pool.map(function, range(start, min(start + 128, n))))
            progress(state, phase, len(values), n)
    return values


def brset_content(m, selected, ref, state):
    def one(i):
        path = BRSET / (str(m['image_ids'][selected[i]]) + '.jpg')
        record = stat_record(path, BRSET)
        data = stable_bytes(record, BRSET)
        expected = bytes(ref['raw_file_sha256'][i]).hex()
        r.require(digest(data) == expected and len(data) == int(ref['file_size'][i]))
        dh, ph, diff = hash_pixels(decode_jpeg(data, expected, size=128))
        r.require(np.array_equal(dh, ref['decoded_128_rgb_sha256'][i]) and ph == ref['perceptual_hash64'][i]
                  and diff == ref['difference_hash64'][i])
        pixels224 = decode_jpeg(data, expected, size=224)
        return record, dh, ph, diff, np.frombuffer(hashlib.sha256(pixels224.tobytes()).digest(), dtype=np.uint8).copy()
    values = map_rows(one, len(selected), state, 'brset_content')
    private_json(PRIVATE / 'brset_inventory.json', [v[0] for v in values])
    return {'source_rows': selected, 'raw_sha256': ref['raw_file_sha256'].copy(),
            'decoded_sha256': np.stack([v[1] for v in values]),
            'phash': np.asarray([v[2] for v in values], dtype=np.uint64),
            'dhash': np.asarray([v[3] for v in values], dtype=np.uint64),
            'decoded224_sha256': np.stack([v[4] for v in values])}


def ai_content(records, state):
    def one(i):
        data = stable_bytes(records[i], AI)
        raw = np.frombuffer(hashlib.sha256(data).digest(), dtype=np.uint8).copy()
        dh, ph, diff = hash_pixels(decode_dicom128(data, bytes(raw).hex()))
        return raw, dh, ph, diff
    values = map_rows(one, len(records), state, 'ai_readi_content')
    return {'raw_sha256': np.stack([v[0] for v in values]), 'decoded_sha256': np.stack([v[1] for v in values]),
            'phash': np.asarray([v[2] for v in values], dtype=np.uint64),
            'dhash': np.asarray([v[3] for v in values], dtype=np.uint64)}


def screen(m, b, a):
    rows = b['source_rows']
    index = PrivateHashIndex(b['decoded_sha256'], b['phash'], b['dhash'])
    within = np.zeros(len(rows), bool)
    ai_match = np.zeros(len(rows), bool)
    for i in range(len(rows)):
        matches = index.match(bytes(b['decoded_sha256'][i]), int(b['phash'][i]), int(b['dhash'][i]))
        for j in matches:
            if m['patient_ids'][rows[i]] != m['patient_ids'][rows[j]]:
                within[i] = within[j] = True
    for i in range(len(a['phash'])):
        matches = index.match(bytes(a['decoded_sha256'][i]), int(a['phash'][i]), int(a['dhash'][i]))
        if matches:
            ai_match[list(matches)] = True
    people_within = set(m['patient_ids'][rows[within]].tolist())
    people_ai = set(m['patient_ids'][rows[ai_match]].tolist())
    quarantine = m['adult_eligible'] & np.isin(m['patient_ids'], list(people_within | people_ai))
    admitted = m['adult_eligible'] & ~quarantine
    return {'within_brset_direct': within, 'ai_readi_direct': ai_match,
            'quarantined': quarantine, 'admitted': admitted}


def result_value(p, pin, m, b, flags):
    s = r.kernel._summary(tuple(m['patient_ids'].tolist()), flags['admitted'], m['split'], m['labels'], m['observed'])
    r.kernel.validate_summary(s)
    return {'schema': SCHEMA, 'status': 'content_assessed', 'protocol_sha256': pin,
        'upstream': p['upstream'], 'current_brset_raw_and_128_hashes_match_historical': True,
        'current_brset_224_decode_passed': True, 'all_ai_source_files_scanned': True,
        'historical_decoder_preflight_passed': True, 'original_patient_split_unchanged': True,
        'quarantine_policy': 'whole_person_exact_or_dual_hash_radius4_label_blind',
        'flagged_counts_coarsened': {
            'within_brset_direct_images': coarse(int(flags['within_brset_direct'].sum())),
            'ai_readi_direct_images': coarse(int(flags['ai_readi_direct'].sum())),
            'quarantined_images': coarse(int(flags['quarantined'].sum())),
            'quarantined_people': coarse(len(set(m['patient_ids'][flags['quarantined']].tolist())))},
        'post_quarantine_counts': r.plain(s['counts']), 'post_quarantine_label_support': r.plain(s['label_support']),
        'cross_study_person_identity_proven': False, 'absence_of_all_duplicates_proven': False,
        'historical_all_source_perceptual_warning_preserved': True,
        'model_training_performed': False, 'training_authorized_by_this_audit': False,
        'patient_level_output_emitted': False}


def bundle_files():
    return ('ai_inventory.json', 'brset_inventory.json', 'brset_hashes.npz', 'ai_hashes.npz', 'admission.npz')


def manifest(pin):
    return {'schema': SCHEMA, 'protocol_sha256': pin, 'aggregate_sha256': r.sha(OUT / 'aggregate.json'),
            'private_sha256': {name: r.sha(PRIVATE / name) for name in bundle_files()},
            'patient_level_output_emitted': False}


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def validate_arrays(m, selected, ref, b, a, flags):
    nb, nm = len(selected), len(m['adult_eligible'])
    def bundle(value, spec):
        r.require(type(value) is dict and set(value) == set(spec))
        for name, (dtype, shape) in spec.items():
            arr = value[name]
            r.require(type(arr) is np.ndarray and arr.dtype == np.dtype(dtype) and arr.shape == shape)
    bundle(b, {'source_rows': (np.int64, (nb,)), 'raw_sha256': (np.uint8, (nb, 32)),
        'decoded_sha256': (np.uint8, (nb, 32)), 'phash': (np.uint64, (nb,)),
        'dhash': (np.uint64, (nb,)), 'decoded224_sha256': (np.uint8, (nb, 32))})
    bundle(a, {'raw_sha256': (np.uint8, (N_AI, 32)), 'decoded_sha256': (np.uint8, (N_AI, 32)),
        'phash': (np.uint64, (N_AI,)), 'dhash': (np.uint64, (N_AI,))})
    bundle(flags, {'within_brset_direct': (bool, (nb,)), 'ai_readi_direct': (bool, (nb,)),
        'quarantined': (bool, (nm,)), 'admitted': (bool, (nm,))})
    r.require(np.array_equal(b['source_rows'], selected))
    for new, old in (('raw_sha256', 'raw_file_sha256'), ('decoded_sha256', 'decoded_128_rgb_sha256'),
                     ('phash', 'perceptual_hash64'), ('dhash', 'difference_hash64')):
        r.require(np.array_equal(b[new], ref[old]))
    r.require(not np.any(flags['quarantined'] & ~m['adult_eligible']))
    r.require(np.array_equal(flags['admitted'], m['adult_eligible'] & ~flags['quarantined']))


def brset_stats_unchanged():
    records = json.loads((PRIVATE / 'brset_inventory.json').read_text())
    r.require(all(stat_record(BRSET / record['relative_path'], BRSET) == record for record in records))


def authenticate(pin, replay=False):
    p = load_protocol(pin)
    r.require(not (AUDIT / 'failure.json').exists())
    r.inventory(OUT, ('aggregate.json', 'manifest.json', 'progress.json'))
    validate_progress(json.loads((OUT / 'progress.json').read_text()), terminal=True)
    r.inventory(PRIVATE, bundle_files(), private=True)
    r.require(r.equal(json.loads((OUT / 'manifest.json').read_text()), manifest(pin)))
    m, selected, ref = selected_metadata()
    b = load_npz(PRIVATE / 'brset_hashes.npz'); a = load_npz(PRIVATE / 'ai_hashes.npz')
    flags = load_npz(PRIVATE / 'admission.npz')
    validate_arrays(m, selected, ref, b, a, flags)
    if replay:
        expected = screen(m, b, a)
        r.require(set(flags) == set(expected) and all(np.array_equal(flags[k], expected[k]) for k in expected))
    v = result_value(p, pin, m, b, flags)
    r.require(r.equal(json.loads((OUT / 'aggregate.json').read_text()), v))
    return p, m, b, a, flags


def run(pin, state):
    p = load_protocol(pin)
    r.inventory(PRIVATE, ('ai_inventory.json',), private=True)
    r.absent(OUT, AUDIT)
    OUT.mkdir(); state['owned'] = OUT
    records = json.loads((PRIVATE / 'ai_inventory.json').read_text())
    r.require(r.equal(records, ai_inventory()))
    progress(state, 'pixel_preflight')
    preflight(records)
    m, selected, ref = selected_metadata()
    b = brset_content(m, selected, ref, state)
    private_npz(PRIVATE / 'brset_hashes.npz', b)
    a = ai_content(records, state)
    private_npz(PRIVATE / 'ai_hashes.npz', a)
    progress(state, 'quarantine')
    flags = screen(m, b, a)
    validate_arrays(m, selected, ref, b, a, flags)
    private_npz(PRIVATE / 'admission.npz', flags)
    r.require(r.equal(records, ai_inventory()))
    brset_stats_unchanged()
    load_protocol(pin)
    progress(state, 'publication')
    r.write_json(OUT / 'aggregate.json', result_value(p, pin, m, b, flags))
    r.write_json(OUT / 'manifest.json', manifest(pin))
    authenticate(pin)


def audit(pin, state):
    p, m, b, a, flags = authenticate(pin, replay=True)
    r.absent(AUDIT)
    AUDIT.mkdir(); state['owned'] = AUDIT
    ai_rows = json.loads((PRIVATE / 'ai_inventory.json').read_text())
    brset_rows = json.loads((PRIVATE / 'brset_inventory.json').read_text())
    r.require(r.equal(ai_rows, ai_inventory()))
    for records, root, hashes in ((brset_rows, BRSET, b), (ai_rows, AI, a)):
        pixel_rows = frozenset(sampled(len(records)))
        def one(i):
            data = stable_bytes(records[i], root)
            r.require(digest(data) == bytes(hashes['raw_sha256'][i]).hex())
            if i in pixel_rows:
                pix = decode_dicom128(data, digest(data)) if root == AI else decode_jpeg(data, digest(data), size=128)
                dh, ph, diff = hash_pixels(pix)
                r.require(np.array_equal(dh, hashes['decoded_sha256'][i]) and ph == hashes['phash'][i] and diff == hashes['dhash'][i])
                if root == BRSET:
                    pix224 = decode_jpeg(data, digest(data), size=224)
                    r.require(hashlib.sha256(pix224.tobytes()).digest() == bytes(hashes['decoded224_sha256'][i]))
            return True
        map_rows(one, len(records), state, 'audit_raw_bytes')
    r.require(r.equal(ai_rows, ai_inventory()))
    brset_stats_unchanged()
    authenticate(pin, replay=True)
    progress(state, 'publication')
    value = audit_value(pin)
    r.write_json(AUDIT / 'audit.json', value)
    r.write_json(AUDIT / 'manifest.json', {'protocol_sha256': pin, 'audit_sha256': r.sha(AUDIT / 'audit.json')})
    authenticate_audit(pin, r.sha(AUDIT / 'audit.json'))


def audit_value(pin):
    return {'schema': SCHEMA, 'status': 'authenticated', 'protocol_sha256': pin,
        'terminal_manifest_sha256': r.sha(OUT / 'manifest.json'), 'all_current_raw_bytes_reauthenticated': True,
        'deterministic_pixel_replay_passed': True, 'all_match_flags_and_admission_replayed': True,
        'patient_split_regenerated': False, 'model_training_performed': False, 'patient_level_output_emitted': False}


def authenticate_audit(pin, audit_pin):
    authenticate(pin)
    r.inventory(AUDIT, ('audit.json', 'manifest.json', 'progress.json'))
    validate_progress(json.loads((AUDIT / 'progress.json').read_text()), terminal=True)
    r.require(r.sha(AUDIT / 'audit.json') == audit_pin and r.equal(json.loads((AUDIT / 'audit.json').read_text()), audit_value(pin)))
    r.require(r.equal(json.loads((AUDIT / 'manifest.json').read_text()), {'protocol_sha256': pin, 'audit_sha256': audit_pin}))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'audit', 'verify'))
    parser.add_argument('--protocol-sha256'); parser.add_argument('--audit-sha256')
    args = parser.parse_args(argv)
    state = {'phase': 'authentication', 'owned': None}
    answer = {'status': 'failed', 'patient_level_output_emitted': False}
    with r.quiet():
        try:
            with LOCK.open('a') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                pin = args.protocol_sha256
                if args.action == 'prepare': pin = prepare(state)
                elif args.action == 'run': run(pin, state)
                elif args.action == 'audit': audit(pin, state)
                elif args.audit_sha256: authenticate_audit(pin, args.audit_sha256)
                else: authenticate(pin, replay=True)
                answer = {'status': 'complete', 'action': args.action, 'protocol_sha256': pin, 'patient_level_output_emitted': False}
                if args.action != 'prepare': answer['aggregate_sha256'] = r.sha(OUT / 'aggregate.json')
                if args.action == 'audit' or args.audit_sha256: answer['audit_sha256'] = r.sha(AUDIT / 'audit.json')
        except Exception:
            answer['phase'] = state['phase']
            if state['owned'] is not None:
                try: r.write_json(state['owned'] / 'failure.json', answer)
                except Exception: pass
    print(json.dumps(answer, sort_keys=True))
    return int(answer['status'] != 'complete')


if __name__ == '__main__':
    raise SystemExit(main())
