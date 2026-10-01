"""Exclusive local re-extraction of frozen comparators; no model training.

The previous benchmark intentionally discarded its representations. This new
protocol permits local private caching, without changing that prior protocol.
All actual work holds the shared lock and silences both output descriptors.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path

import numpy as np

import bran_named_fm_experiment_v1_attempt2 as original
import bran_named_fm_extraction_v1_attempt2 as extraction
import bran_named_fm_cache_storage_v1 as storage
import run_bran_fm_learning_curve_v1 as fm
import run_bran_raw_teacher_distillation_v1 as retained
import run_bran_retinal_group_readiness_v1 as safe

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'BRAN_NAMED_FM_CACHE_PROTOCOL_V1.json'
OUT = ROOT / 'BRAN_NAMED_FM_CACHE_V1'
PRIVATE = fm.DEFAULT_CACHE_MANIFEST.parent
LOCK = Path('/private/tmp/bran_retinal_extraction_v1.lock')
CODE = ('run_bran_named_fm_cache_v1.py', 'test_run_bran_named_fm_cache_v1.py',
        'bran_named_fm_cache_storage_v1.py', 'test_bran_named_fm_cache_storage_v1.py',
        'BRAN_NAMED_FM_CACHE_EXECUTION_V1.md')
ERROR = 'frozen FM cache execution rejected'


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def read(path):
    return json.loads(Path(path).read_text())


def write_private(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())


def describe():
    named = original.validate_protocol(ROOT, check_external=True)
    learning = fm._learning_binding(ROOT)
    native = retained.prepare()
    report = read(ROOT / original.PATHS['success'])
    summary = original.validate_report(report, named)
    approved = approved_reference(named, report, summary, learning)
    require(approved['endpoint_names'] == learning['endpoint_names']
            == native['native_source']['source']['endpoint_names'])
    auth = named['authentication']
    require(auth['outer_fold_sha256'] == learning['outer_fold_sha256']
            and auth['inner_fold_sha256'] == learning['inner_fold_sha256'])
    require(safe.sha(ROOT / original.PROTOCOL) == fm.NAMED_PROTOCOL_SHA
            and safe.sha(ROOT / original.PATHS['success']) == fm.NAMED_SUCCESS_SHA)
    return {'named_protocol': named, 'named_protocol_sha256': fm.NAMED_PROTOCOL_SHA,
            'named_success_sha256': fm.NAMED_SUCCESS_SHA, 'approved_reference': approved,
            'learning_source': learning, 'retained': native,
            'image_selection_sha256': report['selection']['ordered_source_sha256']}


def approved_reference(named, report, summary, learning):
    """Read endpoint identities from the protocol, not the count-only report scope."""
    names = learning['endpoint_names']
    endpoints = report['result']['endpoint_results']
    require(len(names) == len(set(names)) == fm.ENDPOINT_COUNT
            and set(names) == set(endpoints) == set(named['scope']['eligible_source_codes']))
    values = {arm: {name: float(endpoints[name]['arms'][arm]['auroc']) for name in names}
              for arm in fm.ARMS}
    macro = {arm: float(summary['macro_auroc'][arm]) for arm in fm.ARMS}
    for arm in fm.ARMS:
        require(all(np.isfinite(x) and 0 <= x <= 1 for x in values[arm].values())
                and np.isfinite(macro[arm])
                and abs(np.mean(list(values[arm].values())) - macro[arm]) < 1e-10)
    return {'protocol_sha256': fm.NAMED_PROTOCOL_SHA, 'success_sha256': fm.NAMED_SUCCESS_SHA,
            'endpoint_names': list(names), 'endpoint_auroc': values, 'macro_auroc': macro}


def private_inputs(description):
    """Load unchanged native inputs and authenticate original selected images."""
    from run_patient_atlas_v5_foundation_comparator import _enumerate_selected_cfp
    source = retained.native.source
    context = source.io.load_context()
    ctx, folds, c, cm, eligible, retina, present, age, names = context
    ids = list(ctx['raw_cohort'].patient_ids)
    require(len(ids) == len(set(ids)) == fm.PATIENT_COUNT
            and ids == list(ctx['feature_cohort'].patient_ids)
            and set(ctx['raw_cohort'].split_labels) <= {'train', 'val'})
    binding = {'current_inputs_sha256': fm.current_inputs_sha256(c, cm, eligible, retina, present, age, names),
               'row_order_sha256': fm.row_order_sha256(ids),
               'outer_fold_sha256': description['learning_source']['outer_fold_sha256'],
               'inner_fold_sha256': description['learning_source']['inner_fold_sha256']}
    fm._validate_context({**description, 'cache': binding}, source, *context)
    fm._inner_assignments(description, source, ctx, folds)
    dataset = Path(description['named_protocol']['data_roots']['dataset_root'])
    paths, rows = _enumerate_selected_cfp(dataset_root=dataset, patient_ids=ids)
    count = np.bincount(rows, minlength=fm.PATIENT_COUNT)
    require(np.array_equal(count, np.asarray(ctx['feature_cohort'].eye_observed_mask, bool).sum(1))
            and np.array_equal(count > 0, present))
    signature, stats = original.image_signature(paths, rows, dataset)
    require(signature == description['image_selection_sha256'])
    binding['image_selection_sha256'] = signature
    return context, paths, rows, stats, binding


def code_hashes(description):
    names = set(CODE) | set(fm.CODE) | set(description['named_protocol']['expected_hashes'])
    names |= set(description['retained']['code_sha256'])
    # Recursive project-local imports cover local IO/authentication helpers too.
    names = original.closure(ROOT, names)
    return {name: safe.sha(ROOT / name) for name in sorted(names)}


def value(description, binding):
    return {'schema': 'bran-named-fm-cache-protocol-v1', 'status': 'frozen_before_execution',
            'source': description, 'inputs_binding': binding, 'code_sha256': code_hashes(description),
            'encoder_training': False, 'head_training': False, 'patient_level_output_permitted': False,
            'private_cache_permitted': True, 'automatic_promotion': False,
            'retinal_variants': ['retfound_green', 'visionfm_last4', 'dinov3_generic'],
            'clinical_variants': ['labrador'], 'cpu_threads': 2, 'retinal_batch_size': 16}


def prepare():
    safe.absent(PROTOCOL, OUT, PRIVATE)
    description = describe()
    *_, binding = private_inputs(description)
    require(safe.equal(description, describe()))
    p = value(description, binding)
    safe.absent(PROTOCOL, OUT, PRIVATE)
    write_private(PROTOCOL, p)
    return safe.sha(PROTOCOL)


def load_protocol(pin):
    require(type(pin) is str and len(pin) == 64)
    safe.regular(PROTOCOL, private=True)
    require(os.lstat(PROTOCOL).st_nlink == 1 and safe.sha(PROTOCOL) == pin)
    p = read(PROTOCOL)
    description = describe()
    require(safe.equal(p, value(description, p['inputs_binding'])))
    require(safe.sha(PROTOCOL) == pin)
    return p


def progress(state, phase, variant=None):
    require(phase in ('authentication', 'context', 'extraction', 'verification', 'publication'))
    require(variant is None or variant in fm.ARMS)
    state['phase'] = phase
    if state['owned']:
        temp = OUT / 'progress.tmp'
        safe.absent(temp)
        safe.write_json(temp, {'phase': phase, 'variant': variant, 'patient_level_output_emitted': False})
        os.replace(temp, OUT / 'progress.json')


def run(pin, state):
    p = load_protocol(pin)
    safe.absent(OUT, PRIVATE)
    OUT.mkdir(); state['owned'] = True
    progress(state, 'context')
    context, paths, rows, stats, binding = private_inputs(p['source'])
    require(safe.equal(binding, p['inputs_binding']))
    _, _, c, cm, eligible, _, _, _, names = context
    PRIVATE.parent.mkdir(exist_ok=True)
    writer = storage.create(PRIVATE)
    def unchanged():
        require([(Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in paths] == stats)
    for variant in fm.ARMS:
        progress(state, 'extraction', variant)
        unchanged()
        artifact = p['source']['named_protocol']['artifacts'][variant]
        if variant == 'labrador':
            array = extraction.extract_labrador(c, cm, eligible, names, artifact=artifact, project_root=ROOT)
        else:
            array = extraction.extract_retinal(paths, rows, fm.PATIENT_COUNT,
                                               variant=variant, artifact=artifact)
        unchanged()
        writer.write(variant, array)
        del array
    progress(state, 'verification')
    # Content, not only mtime, is authenticated again after all extraction.
    *_, after = private_inputs(p['source'])
    require(safe.equal(after, binding) and safe.equal(load_protocol(pin), p))
    manifest_pin = writer.finish(**{k: binding[k] for k in (
        'current_inputs_sha256', 'row_order_sha256', 'outer_fold_sha256', 'inner_fold_sha256')},
        artifact_path=ROOT / original.PROTOCOL, artifact_sha256=fm.NAMED_PROTOCOL_SHA)
    require(storage.authenticate(PRIVATE, manifest_pin) is True)
    progress(state, 'publication')
    record = {'schema': 'bran-named-fm-cache-result-v1', 'status': 'completed',
              'protocol_sha256': pin, 'cache_manifest_sha256': manifest_pin,
              'patient_count_verified': True, 'outer_and_five_inner_folds_verified': True,
              'historical_selected_images_verified': True, 'variants_completed': list(fm.ARMS),
              'encoder_training': False, 'head_training': False,
              'clinical_efficacy_established': False, 'patient_level_output_emitted': False}
    safe.write_json(OUT / 'result.json', record)
    safe.write_json(OUT / 'success.json', {'protocol_sha256': pin, 'result_sha256': safe.sha(OUT / 'result.json')})
    state['owned'] = False


def verify(pin):
    load_protocol(pin)
    safe.inventory(OUT, ('result.json', 'success.json', 'progress.json'))
    record = read(OUT / 'result.json')
    require(set(record) == {'schema', 'status', 'protocol_sha256', 'cache_manifest_sha256',
        'patient_count_verified', 'outer_and_five_inner_folds_verified', 'historical_selected_images_verified',
        'variants_completed', 'encoder_training', 'head_training', 'clinical_efficacy_established',
        'patient_level_output_emitted'})
    require(record['schema'] == 'bran-named-fm-cache-result-v1' and record['status'] == 'completed'
            and record['protocol_sha256'] == pin and record['variants_completed'] == list(fm.ARMS))
    for key in ('patient_count_verified', 'outer_and_five_inner_folds_verified', 'historical_selected_images_verified'):
        require(record[key] is True)
    for key in ('encoder_training', 'head_training', 'clinical_efficacy_established', 'patient_level_output_emitted'):
        require(record[key] is False)
    require(safe.equal(read(OUT / 'success.json'), {'protocol_sha256': pin,
                                                  'result_sha256': safe.sha(OUT / 'result.json')}))
    require(safe.equal(read(OUT / 'progress.json'), {'phase': 'publication', 'variant': None,
                                                   'patient_level_output_emitted': False}))
    require(storage.authenticate(PRIVATE, record['cache_manifest_sha256']) is True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'verify'))
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args(argv)
    state = {'owned': False, 'phase': 'authentication'}
    answer = {'status': 'failed', 'patient_level_output_emitted': False}
    with safe.quiet():
        try:
            with LOCK.open('a') as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                pin = args.protocol_sha256
                if args.action == 'prepare': pin = prepare()
                elif args.action == 'run': run(pin, state)
                else: verify(pin)
                answer.update(status='completed', action=args.action, protocol_sha256=pin)
        except Exception:
            answer['phase'] = state['phase']
            if state['owned']:
                try:
                    safe.write_json(OUT / 'failure.json', answer)
                except Exception:
                    pass
    print(json.dumps(answer, sort_keys=True))
    return int(answer['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
