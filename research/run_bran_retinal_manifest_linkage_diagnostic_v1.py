"""Read-only private linkage computation; publishes only a closed diagnostic.

No images, model inference or training. Existing per-file hashes remain private.
This links metadata to an authenticated historical manifest, not live file bytes.
"""
from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import run_bran_retinal_group_readiness_v1 as r

PROTOCOL_PIN = '7f934a7938851804b695e8529e51ecf66152e915d48e035f45b3296e596715e7'
AUDIT_PIN = '4ea37b283be55def66a9093dc08a6cdf8f7fab92964b647c095ca93f4b51323a'
REPORT = r.ROOT / 'PATIENT_ATLAS_EYE_SOURCE_PROVENANCE_RECONSTRUCTION_V2.json'
REPORT_PIN = '319cda7b26dc4df3b0c5256fd50b55c837f4cd93bb45020c71f129f969f8636b'
MANIFEST = r.ROOT / 'exploratory_artifacts/patient_atlas_external_eye_file_manifest_v1.npz'
MANIFEST_PIN = '36683903b5dcf1330fd74ed0cf536daad4ba9c54df624e1a2c9cdbb9e0ac3dd2'
OUTPUT = r.ROOT / 'BRAN_RETINAL_MANIFEST_LINKAGE_DIAGNOSTIC_V1.json'


def coarse(n):
    return 0 if n == 0 else '<20' if n < 20 else int(n // 20 * 20)


def crossing_count(hashes, splits):
    """Count selected image rows in a digest group spanning distinct splits."""
    groups = defaultdict(list)
    for i, value in enumerate(hashes):
        groups[bytes(value)].append(i)
    return sum(len(indices) for indices in groups.values()
               if len({int(splits[i]) for i in indices}) > 1)


def assess():
    r.authenticate_audit(PROTOCOL_PIN, AUDIT_PIN)
    r.require(r.sha(REPORT) == REPORT_PIN and r.sha(MANIFEST) == MANIFEST_PIN)
    old = json.loads(REPORT.read_text())
    r.require(old['bindings']['manifest_file_sha256'] == MANIFEST_PIN)
    r.require(old['bindings']['audit_code_sha256'] == r.sha(r.ROOT / 'patient_atlas_eye_source_provenance.py'))
    with np.load(MANIFEST, allow_pickle=False) as data:
        logical = data['logical_relative_path_sha256']
        raw = data['raw_file_sha256']
        decoded = data['decoded_128_rgb_sha256']
        source = data['source_code']
        for value in (logical, raw, decoded):
            r.require(value.dtype == np.uint8 and value.shape == (115273, 32))
        r.require(source.dtype == np.uint8 and source.shape == (115273,))
        r.require(np.array_equal(source, np.repeat(np.arange(4, dtype=np.uint8), [92501, 16266, 4512, 1994])))
    indexes = np.flatnonzero(source == 1)
    lookup = {bytes(logical[i]): int(i) for i in indexes}
    r.require(len(lookup) == len(indexes))
    with np.load(r.PRIVATE / 'assignment.npz', allow_pickle=False) as data:
        names = data['image_ids']
        split = data['split']
        eligible = data['adult_eligible']
    r.require(names.dtype.kind == 'U' and eligible.dtype == bool and split.dtype == np.int8)
    r.require(names.shape == split.shape == eligible.shape and len(names) == len(indexes))
    rows = np.asarray([lookup[hashlib.sha256(('brset/' + str(name) + '.jpg').encode()).digest()]
                       for name in names], dtype=np.int64)
    r.require(len(set(rows.tolist())) == len(indexes))
    selected = rows[eligible]
    checks = {}
    for name, array in [('raw', raw), ('decoded_128', decoded)]:
        crossings = crossing_count(array[selected], split[eligible])
        other = {bytes(value) for value in array[source != 1]}
        overlaps = sum(bytes(value) in other for value in array[selected])
        checks[name] = {
            'cross_partition_duplicate_images_coarsened': coarse(crossings),
            'cross_partition_duplicates_present': bool(crossings),
            'other_external_source_overlap_images_coarsened': coarse(overlaps),
            'other_external_source_overlap_present': bool(overlaps)}
    r.authenticate_audit(PROTOCOL_PIN, AUDIT_PIN)
    r.require(r.sha(REPORT) == REPORT_PIN and r.sha(MANIFEST) == MANIFEST_PIN)
    return {
        'schema': 'bran-retinal-manifest-linkage-diagnostic-v1', 'status': 'authenticated',
        'metadata_protocol_sha256': PROTOCOL_PIN, 'metadata_audit_sha256': AUDIT_PIN,
        'historical_report_sha256': REPORT_PIN, 'historical_manifest_sha256': MANIFEST_PIN,
        'diagnostic_code_sha256': r.sha(Path(__file__)),
        'all_metadata_images_linked_bijectively_to_brset_manifest': True,
        'eligible_images_coarsened': coarse(len(selected)), 'historical_digest_checks': checks,
        'historical_all_source_ai_readi_perceptual_screen_passed':
            bool(old['gates']['perceptual_overlap_screen_passed']),
        'live_image_bytes_rechecked': False, 'new_image_decoding_performed': False,
        'new_near_duplicate_screen_performed': False, 'cross_source_people_linked': False,
        'training_authorized_by_this_diagnostic': False, 'patient_level_output_emitted': False}


def main():
    result = {'status': 'failed', 'patient_level_output_emitted': False}
    with r.quiet():
        try:
            with r.LOCK.open('a') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                r.absent(OUTPUT)
                result = assess()
                r.write_json(OUTPUT, result)
        except Exception:
            result = {'status': 'failed', 'patient_level_output_emitted': False}
    # All values are fixed statuses, coarsened counters or authenticated hashes.
    print(json.dumps(result, sort_keys=True))
    return int(result['status'] != 'authenticated')


if __name__ == '__main__':
    raise SystemExit(main())
