"""FD-quiet local metadata contract preflight; no images, fitting or model scores.

Reuses authenticated BRSET/ODIR qualification artifacts. This is not an ODIR
reuse authorization, real-data training protocol, or source-performance test.
"""
import fcntl
import json
from pathlib import Path

import numpy as np

import run_bran_retinal_supervised_adaptation_v1 as brset
import run_bran_odir_split_qualification_v1 as odir
from bran_retinal_multisource_sampling_v1 import BatchPlanner
from bran_retinal_multisource_source_contract_v1 import brset_spec, odir_spec

r = brset.r
ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / 'BRAN_RETINAL_MULTISOURCE_SOURCE_BINDING_V1.json'
ODIR_SPLIT_PIN = 'cdfac2ad66cb3b3fca2df6ef5ca68a0c0a194522ad2839c933722c0d3a5951e3'
ODIR_SPLIT_AUDIT = 'cc4c5f85f04693d14ea910325bf5f7cbea35bd692e5281832e79800d17605d44'
CLOSURE = (
    Path(__file__).name, 'bran_retinal_multisource_sampling_v1.py',
    'bran_retinal_multisource_source_contract_v1.py',
    'test_bran_retinal_multisource_sampling_v1.py',
    'test_bran_retinal_multisource_source_contract_v1.py',
    'bran_retinal_adaptation_evaluation_v1.py',
    'run_bran_retinal_supervised_adaptation_v1.py',
    'run_bran_odir_split_qualification_v1.py',
    'run_bran_odir_content_qualification_v1.py',
    'run_bran_retinal_group_readiness_v1.py',
)


def identities():
    return {'code_sha256': {name: r.sha(ROOT / name) for name in CLOSURE},
            'upstream_manifest_sha256': {
                'brset_content': r.sha(brset.a.OUT / 'manifest.json'),
                'brset_content_audit': r.sha(brset.a.AUDIT / 'manifest.json'),
                'odir_content': r.sha(odir.content.OUT / 'manifest.json'),
                'odir_content_audit': r.sha(odir.content.AUDIT / 'manifest.json'),
                'odir_split': r.sha(odir.OUT / 'manifest.json'),
                'odir_split_audit': r.sha(odir.AUDIT / 'manifest.json')}}


def load_qualified_specs():
    """Private return values; entry requires caller-held lock and FD silence."""
    odir.authenticate_audit(ODIR_SPLIT_PIN, ODIR_SPLIT_AUDIT)
    # The ODIR audit authenticates its content+metadata and BRSET reference chain,
    # preserving the separate original runtimes. No monkey-patched runtime checks.
    source = brset.load_source()
    b, bw, _ = brset_spec(source['arrays'])
    inputs = brset.load_arrays(odir.PRIVATE / 'inputs.npz')
    split = brset.load_arrays(odir.PRIVATE / 'assignment.npz')['split']
    selection = json.loads((odir.content.PRIVATE / 'selection.json').read_text())
    retained = brset.load_arrays(odir.content.PRIVATE / 'flags.npz')['retained_representative']
    o, ow, _ = odir_spec(inputs, split, selection, retained)
    return {'brset': b, 'odir': o}, {'brset': bw, 'odir': ow}


def validate_plans(specs, weights):
    """Check actual metadata joins and deterministic train-only selections locally."""
    first = BatchPlanner(specs, steps=2, batch_size=16)
    second = BatchPlanner(specs, steps=2, batch_size=16)
    for a, b in zip(first, second):
        for name, width in (('brset', 13), ('odir', 8)):
            p, q = a[name], specs[name]
            r.require(all(np.array_equal(p[key], b[name][key], equal_nan=True) for key in p))
            n = len(p['image_indices'])
            r.require(p['labels'].shape == p['observed'].shape == (16, width)
                      and p['group_index'].shape == (n,) and p['patch_mask'].shape == (n, 196)
                      and np.all(p['patch_mask'].sum(1) == 147))
            groups = q['image_groups'][p['image_indices']]
            r.require(np.all(q['group_split'][groups] == 0))
            for group_index in range(16):
                image_rows = p['image_indices'][p['group_index'] == group_index]
                actual_groups = q['image_groups'][image_rows]
                r.require(len(np.unique(actual_groups)) == 1)
                if name == 'brset':
                    r.require(len(image_rows) == 1)
                    label_index = image_rows[0]
                else:
                    label_index = actual_groups[0]
                    r.require(np.array_equal(image_rows, np.flatnonzero(q['image_groups'] == label_index)))
                known = q['observed'][label_index] & q['usable']
                r.require(np.array_equal(known, p['observed'][group_index])
                          and np.array_equal(q['labels'][label_index, known], p['labels'][group_index, known])
                          and np.all(np.isnan(p['labels'][group_index, ~known])))
            r.require(weights[name].shape == (width,) and np.isfinite(weights[name]).all()
                      and np.all((weights[name] >= 1) & (weights[name] <= 10)))
    r.require(first.receipt() == second.receipt() and first.receipt()['completed_steps'] == 2)


def main():
    result = {'schema': 'bran-retinal-multisource-source-binding-v1', 'status': 'failed',
              'patient_processing_local_only': True, 'patient_level_output_emitted': False,
              'images_read': False, 'real_data_training_run': False, 'training_admitted': False}
    phase = 'exclusive_destination'
    created = False
    with r.quiet():
        try:
            r.absent(DESTINATION)
            phase = 'exclusive_compute_lock'
            with brset.LOCK.open('a') as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                phase = 'source_authentication'
                before = identities()
                specs, weights = load_qualified_specs()
                phase = 'patient_label_and_sampling_contract'
                validate_plans(specs, weights)
                phase = 'closure_binding'
                r.require(before == identities())
                result.update(status='passed', source_patient_label_mapping=True,
                    one_two_eye_occurrences_preserved=True, matched_source_stream=True,
                    all_sampled_groups_training_only=True, unknown_labels_preserved=True,
                    class_weights_train_only=True, source_approval_still_required=True, **before)
        except Exception:
            result['failure_phase'] = phase
        if not DESTINATION.exists() and not DESTINATION.is_symlink():
            r.write_json(DESTINATION, result)
            created = True
    print(json.dumps({'status': result['status'], 'phase': phase,
                      'exclusive_result_created': created, 'patient_level_output_emitted': False}, sort_keys=True))
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
