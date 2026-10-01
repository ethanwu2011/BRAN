"""Prospective local unified refit: prepare, execute, verify and replay audit.

No action on import. The CLI holds one shared nonblocking compute lock and
silences both file descriptors. All patient material stays in private memory
or mode600 artifacts. Completed negative evidence is never overwritten.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import torch

import bran_agefree_unified_context_v1 as source
import bran_retinal_unified_refit_artifacts_v1 as storage
import bran_agefree_unified_references_v1 as references
import bran_agefree_unified_reference_checks_v1 as checks
import bran_agefree_unified_oof_v1 as oof
import bran_agefree_unified_metrics_v1 as metrics
import bran_agefree_unified_structure_v1 as structure

jobs = oof.jobs
r = source.clinical.safe
CODE_ROOT = Path(__file__).resolve().parent
ROOT = CODE_ROOT
PROTOCOL = ROOT / 'BRAN_AGEFREE_UNIFIED_PROTOCOL_V1.json'
OUT = ROOT / 'BRAN_AGEFREE_UNIFIED_V1'
AUDIT = ROOT / 'BRAN_AGEFREE_UNIFIED_AUDIT_V1'
PRIVATE = ROOT / 'private_artifacts/bran_agefree_unified_v1'
LOCK = Path('/private/tmp/bran_retinal_extraction_v1.lock')
ERROR = 'unified refit execution rejected'
PHASES = ('authentication', 'reference_replay', 'fit_outer', 'fit_structure', 'evaluation', 'publication', 'audit')
PARTS = ('training', 'jobs', 'inference', 'oof', 'metrics', 'structure',
         'references', 'reference_checks', 'source', 'context')
CODE = tuple('bran_agefree_unified_' + name + '_v1.py' for name in PARTS) + tuple(
    'test_bran_agefree_unified_' + name + '_v1.py' for name in PARTS) + (
    'run_bran_agefree_unified_v1.py', 'test_run_bran_agefree_unified_v1.py',
    'BRAN_AGEFREE_UNIFIED_EXECUTION_V1.md', 'BRAN_AGEFREE_UNIFIED_DESIGN_V1.md',
    'bran_agefree_source_sampling_v1.py', 'test_bran_agefree_source_sampling_v1.py',
    'bran_retinal_unified_refit_artifacts_v1.py', 'test_bran_retinal_unified_refit_artifacts_v1.py',
    'bran_retinal_unified_refit_membership_v1.py', 'test_bran_retinal_unified_refit_membership_v1.py',
    'bran_retinal_unified_refit_source_v1.py',
    'bran_missingness_stress_v1.py', 'bran_missingness_stress_metrics_v1.py',
    'bran_multisource_mask_contract_v1.py', 'bran_supervised_mask_uncertainty_v1.py',
    'bran_agefree_platelet_evaluation_v1.py', 'test_bran_agefree_platelet_evaluation_v1.py',
    'BRAN_AGEFREE_PLATELET_EVALUATION_V1.md',
    'bran_agefree_available_inference_v1.py', 'test_bran_agefree_available_inference_v1.py',
    'BRAN_AGEFREE_AVAILABLE_INFERENCE_V1.md', 'bran_research_state_io_v1.py',
    'PATIENT_ATLAS_FEATURE_REGISTRY.json',
    'bran_native_calibration_split_v1.py', 'bran_native_cbc_calibration_metrics_v1.py',
    'bran_native_cbc_decoders_v1.py', 'bran_native_screening_kernel_v1.py',
    'run_bran_native_rehearsal_v1.py', 'bran_retinal_refit_inputs_v2.py',
    'bran_clinical_semantics_v1.py', 'bran_disease_structure_192_v1.py',
    'bran_disease_structure_diagnostics_v1.py', 'bran_disease_structure_evaluation_v1.py',
    'bran_other_disease_panel_evaluation_v1.py', 'bran_multigroup_utility_kernel_v1.py')
RESULT_FILES = ('results.json', 'manifest.json', 'success.json', 'progress.json')
AUDIT_FILES = ('audit.json', 'success.json', 'progress.json')


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def read_json(path):
    return json.loads(Path(path).read_text())


def write_private_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())


def parameters():
    return {'external_steps': 3000, 'paired_steps': 1500, 'joint_steps': 1500, 'batch_size': 96, 'cpu_threads': 2,
        'jobs': list(jobs.membership.JOBS), 'paired_people': source.PEOPLE,
        'recorded_conditions': 26, 'state_width': 192, 'replay_atol': 1e-10,
        'role_mapping': metrics.ROLE_MAP, 'source_admission_required': True,
        'retained_reference_replay_before_refit': True, 'readout_refit_during_audit': True,
        'representation_refit_during_audit': False, 'resume_supported': False,
        'automatic_promotion': False, 'official_test_used': False}


def _code(description):
    result = {}
    for key in ('retained', 'foundation', 'blood', 'structure', 'foundation_cache'):
        for name, value in description[key]['code_sha256'].items():
            require(name not in result or result[name] == value)
            require(r.sha(CODE_ROOT / name) == value)
            result[name] = value
    for name in tuple(dict.fromkeys(CODE + tuple(source.clinical.three.CODE) + tuple(source.clinical.nwicu.CODE))):
        value = r.sha(CODE_ROOT / name)
        require(name not in result or result[name] == value)
        result[name] = value
    return result


def _value(description, private_hashes):
    return {'schema': 'bran-agefree-unified-protocol-v1', 'status': 'frozen_before_execution',
        'source': description, 'source_sha256': oof.private_digest(description),
        'prepared_private_sha256': private_hashes, 'parameters': parameters(),
        'code_sha256': _code(description), 'runtime': source.retained.native.source.io.runtime(),
        'patient_level_output_permitted': False}


def prepare():
    r.absent(PROTOCOL, OUT, AUDIT, PRIVATE)
    description = source.describe()
    inputs = source.load(description, oof.private_digest(description))
    r.absent(PROTOCOL, OUT, AUDIT, PRIVATE)
    PRIVATE.parent.mkdir(mode=0o700, exist_ok=True)
    store = storage.PrivateArtifacts.create(PRIVATE)
    store.write_json('membership.json', inputs.plan)
    store.write_json('inputs_binding.json', inputs.binding())
    expected = store.hashes()
    require(set(expected) == set(storage.PREPARE))
    # Write protocol only after all private preparation and complete closure.
    value = _value(description, expected)
    write_private_json(PROTOCOL, value)
    return r.sha(PROTOCOL)


def load_protocol(pin):
    require(jobs._sha(pin))
    r.regular(PROTOCOL, private=True)
    require(os.lstat(PROTOCOL).st_nlink == 1 and r.sha(PROTOCOL) == pin)
    p = read_json(PROTOCOL)
    require(type(p) is dict and set(p) == {'schema', 'status', 'source', 'source_sha256',
        'prepared_private_sha256', 'parameters', 'code_sha256', 'runtime', 'patient_level_output_permitted'}
        and type(p['prepared_private_sha256']) is dict and set(p['prepared_private_sha256']) == set(storage.PREPARE))
    description = source.describe()
    require(r.equal(p, _value(description, p['prepared_private_sha256'])) and r.sha(PROTOCOL) == pin)
    return p


def _inputs(p, hashes):
    storage.authenticate(PRIVATE, hashes)
    plan = storage.load_json(PRIVATE, 'membership.json', hashes)
    binding = storage.load_json(PRIVATE, 'inputs_binding.json', hashes)
    require(all(hashes[name] == p['prepared_private_sha256'][name] for name in storage.PREPARE))
    inputs = source.load(p['source'], p['source_sha256'])
    require(r.equal(inputs.plan, plan) and r.equal(inputs.binding(), binding))
    return inputs


def progress(state, phase, fold=None):
    require(phase in PHASES and (fold is None or type(fold) is int and 0 <= fold < 5))
    state['phase'] = phase
    value = {'phase': phase, 'fold': fold, 'patient_level_output_emitted': False}
    if state.get('owned') is not None:
        temporary = state['owned'] / 'progress.tmp'
        r.absent(temporary)
        r.write_json(temporary, value)
        os.replace(temporary, state['owned'] / 'progress.json')


def _checker(inputs, p, pin):
    args = inputs.context(pin)
    binding = oof.context_binding(**args)
    expected = p['source']['expected_reference']
    return lambda ref: checks.check(ref, expected=expected,
        expected_sha256=oof.private_digest(expected), expected_binding=binding,
        labels=inputs.labels, label_observed=inputs.label_observed,
        clinical59=inputs.data['clinical'], observed=inputs.data['observed'], eligible=inputs.data['eligible'],
        registry_names=inputs.registry_names, folds=inputs.data['folds'], endpoint_names=inputs.endpoint_names)


def _references(inputs, p, pin):
    args = inputs.context(pin)
    return references.build(**args, labels=inputs.labels, label_observed=inputs.label_observed,
        expected_labels_sha256=inputs.binding()['labels_sha256'], foundation_features=inputs.foundation_features,
        expected_foundation_features_sha256=inputs.binding()['foundation_sha256'],
        initial_provider=lambda fold, transform: source.retained.load_initial(fold, transform, p['source']['retained']),
        reference_checker=_checker(inputs, p, pin))


def _same(a, b, atol):
    """Fixed audit numeric tolerance; identity, support and decisions stay exact."""
    if type(a) is np.ndarray or type(b) is np.ndarray:
        return type(a) is type(b) is np.ndarray and a.dtype == b.dtype and a.shape == b.shape and bool(
            np.allclose(a, b, rtol=0, atol=atol, equal_nan=True))
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return set(a) == set(b) and all(_same(a[k], b[k], atol) for k in a)
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(_same(x, y, atol) for x, y in zip(a, b))
    if type(a) is float:
        return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= atol)
    return a == b


def _evaluate(inputs, p, pin, ref, provider, structure_job):
    args = inputs.context(pin)
    budget = {key: p['parameters'][key] for key in ('external_steps', 'paired_steps', 'joint_steps', 'batch_size')}
    collected = oof.collect(**args, provider=provider, **budget)
    evaluated = metrics.evaluate(collected, ref, **args,
        expected_reference_sha256=oof.private_digest(ref), labels=inputs.labels,
        label_observed=inputs.label_observed, expected_labels_sha256=inputs.binding()['labels_sha256'])
    structured = structure.evaluate(structure_job(), **inputs.structure_arguments())
    return {'schema': 'bran-agefree-unified-results-v1', 'status': 'completed',
        'protocol_sha256': pin, 'source_sha256': p['source_sha256'], 'state_width': 192,
        'paired_people': p['parameters']['paired_people'], 'recorded_conditions': 26,
        'oof': evaluated.aggregate, 'structure': structured,
        'retained_reference_replay_passed': True, 'source_admission_authenticated': True,
        'model_selection': 'requires_review_retained_bran_unchanged',
        'patient_level_output_emitted': False, 'automatic_promotion': False}, evaluated.calibration_radii


def validate_result(value, p, pin):
    require(type(value) is dict and set(value) == {'schema', 'status', 'protocol_sha256', 'source_sha256',
        'state_width', 'paired_people', 'recorded_conditions', 'oof', 'structure',
        'retained_reference_replay_passed', 'source_admission_authenticated', 'model_selection',
        'patient_level_output_emitted', 'automatic_promotion'}
        and value['schema'] == 'bran-agefree-unified-results-v1' and value['status'] == 'completed'
        and value['protocol_sha256'] == pin and value['source_sha256'] == p['source_sha256']
        and value['model_selection'] == 'requires_review_retained_bran_unchanged'
        and value['retained_reference_replay_passed'] is True and value['source_admission_authenticated'] is True
        and value['patient_level_output_emitted'] is False and value['automatic_promotion'] is False)
    for key, expected in (('state_width', 192), ('paired_people', p['parameters']['paired_people']), ('recorded_conditions', 26)):
        require(type(value[key]) is int and value[key] == expected)
    metrics.validate_result(value['oof'], tuple(p['source']['retained']['native_source']['source']['endpoint_names']))
    structure.validate_result(value['structure'])


def run(pin, state):
    p = load_protocol(pin)
    r.absent(OUT, AUDIT)
    inputs = _inputs(p, p['prepared_private_sha256'])
    store = storage.PrivateArtifacts.attach_prepared(PRIVATE, p['prepared_private_sha256'])
    OUT.mkdir(); state['owned'] = OUT
    torch.set_num_threads(p['parameters']['cpu_threads'])
    progress(state, 'reference_replay')
    reference = _references(inputs, p, pin)
    store.write_torch('references.pt', reference)
    external_sources = source.clinical.load(p['source']['external_clinical'])
    args = inputs.context(pin)
    budget = {key: p['parameters'][key] for key in ('external_steps', 'paired_steps', 'joint_steps', 'batch_size')}

    def fit(job):
        progress(state, 'fit_structure' if job == 'structure' else 'fit_outer',
                 None if job == 'structure' else int(job[-1]))
        fitted = jobs.fit_job(**args, external_sources=external_sources, labels=inputs.labels, label_observed=inputs.label_observed, job=job, **budget)
        store.write_torch(job + '.pt', fitted.bundle)
        bundle = storage.load_torch(PRIVATE, job + '.pt', store.hashes())
        return jobs.restore_job(bundle, **args, job=job, **budget)

    result, radii = _evaluate(inputs, p, pin, reference, lambda f: fit('outer' + str(f)), lambda: fit('structure'))
    progress(state, 'evaluation')
    validate_result(result, p, pin)
    store.write_npz('radii.npz', radii)
    hashes = store.seal(store.hashes())
    require(set(hashes) == set(storage.INVENTORY))
    _inputs(p, hashes)  # Same input bytes/identities after all fitting/scoring.
    require(r.equal(load_protocol(pin), p))
    progress(state, 'publication')
    r.write_json(OUT / 'results.json', result)
    manifest = {'protocol_sha256': pin, 'results_sha256': r.sha(OUT / 'results.json'), 'private_sha256': hashes}
    r.write_json(OUT / 'manifest.json', manifest)
    r.write_json(OUT / 'success.json', {'protocol_sha256': pin, 'manifest_sha256': r.sha(OUT / 'manifest.json')})
    state['owned'] = None


def payload(pin):
    p = load_protocol(pin)
    r.inventory(OUT, RESULT_FILES)
    require(not (AUDIT / 'failure.json').exists())
    m = read_json(OUT / 'manifest.json')
    require(type(m) is dict and set(m) == {'protocol_sha256', 'results_sha256', 'private_sha256'}
            and m['protocol_sha256'] == pin and type(m['private_sha256']) is dict
            and set(m['private_sha256']) == set(storage.INVENTORY)
            and m['results_sha256'] == r.sha(OUT / 'results.json'))
    require(r.equal(read_json(OUT / 'success.json'), {
        'protocol_sha256': pin, 'manifest_sha256': r.sha(OUT / 'manifest.json')})
        and r.equal(read_json(OUT / 'progress.json'), {
            'phase': 'publication', 'fold': None, 'patient_level_output_emitted': False}))
    storage.authenticate(PRIVATE, m['private_sha256'])
    result = read_json(OUT / 'results.json')
    validate_result(result, p, pin)
    return p, m, result


def audit(pin, state):
    p, manifest, expected = payload(pin)
    r.absent(AUDIT)
    AUDIT.mkdir(); state['owned'] = AUDIT
    # Audit is normally a fresh process: do not inherit an ambient thread count
    # or rely on run() having configured this interpreter first.
    torch.set_num_threads(p['parameters']['cpu_threads'])
    progress(state, 'audit')
    hashes = manifest['private_sha256']; inputs = _inputs(p, hashes)
    stored = storage.load_torch(PRIVATE, 'references.pt', hashes)
    require(_checker(inputs, p, pin)(stored) is True)
    actual = _references(inputs, p, pin)  # Fixed readouts only; no encoder refit.
    require(_same(stored, actual, p['parameters']['replay_atol']))
    args = inputs.context(pin)
    budget = {key: p['parameters'][key] for key in ('external_steps', 'paired_steps', 'joint_steps', 'batch_size')}

    def restore(job):
        return jobs.restore_job(storage.load_torch(PRIVATE, job + '.pt', hashes), **args, job=job, **budget)

    result, radii = _evaluate(inputs, p, pin, actual, lambda f: restore('outer' + str(f)), lambda: restore('structure'))
    validate_result(result, p, pin)
    require(_same(result, expected, p['parameters']['replay_atol'])
            and _same(radii, storage.load_npz(PRIVATE, 'radii.npz', hashes), p['parameters']['replay_atol']))
    _inputs(p, hashes)
    require(r.equal(load_protocol(pin), p))
    progress(state, 'publication')
    record = {'schema': 'bran-agefree-unified-audit-v1', 'status': 'authenticated',
        'protocol_sha256': pin, 'manifest_sha256': r.sha(OUT / 'manifest.json'),
        'results_sha256': manifest['results_sha256'], 'reference_readouts_replayed': True,
        'all_six_job_inference_replayed': True, 'aggregate_and_calibration_replayed': True,
        'representation_retrained': False, 'patient_level_output_emitted': False, 'automatic_promotion': False}
    r.write_json(AUDIT / 'audit.json', record)
    r.write_json(AUDIT / 'success.json', {'protocol_sha256': pin, 'audit_sha256': r.sha(AUDIT / 'audit.json')})
    state['owned'] = None


def authenticate_audit(pin, audit_pin):
    _p, m, _value = payload(pin)
    r.inventory(AUDIT, AUDIT_FILES)
    require(r.sha(AUDIT / 'audit.json') == audit_pin)
    record = read_json(AUDIT / 'audit.json')
    expected = {'schema': 'bran-agefree-unified-audit-v1', 'status': 'authenticated',
        'protocol_sha256': pin, 'manifest_sha256': r.sha(OUT / 'manifest.json'),
        'results_sha256': m['results_sha256'], 'reference_readouts_replayed': True,
        'all_six_job_inference_replayed': True, 'aggregate_and_calibration_replayed': True,
        'representation_retrained': False, 'patient_level_output_emitted': False, 'automatic_promotion': False}
    require(r.equal(record, expected) and r.equal(read_json(AUDIT / 'success.json'), {
        'protocol_sha256': pin, 'audit_sha256': audit_pin}) and r.equal(read_json(AUDIT / 'progress.json'), {
            'phase': 'publication', 'fold': None, 'patient_level_output_emitted': False}))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'verify', 'audit'))
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--audit-sha256')
    args = parser.parse_args(argv)
    state = {'phase': 'authentication', 'owned': None}
    answer = {'status': 'failed', 'patient_level_output_emitted': False}
    with r.quiet():
        try:
            with LOCK.open('a') as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                pin = args.protocol_sha256
                if args.action == 'prepare':
                    pin = prepare()
                elif args.action == 'run':
                    run(pin, state)
                elif args.action == 'audit':
                    audit(pin, state)
                elif args.audit_sha256:
                    authenticate_audit(pin, args.audit_sha256)
                else:
                    payload(pin)
                answer = {'status': 'complete', 'action': args.action, 'protocol_sha256': pin,
                          'patient_level_output_emitted': False}
                if args.action != 'prepare':
                    answer['results_sha256'] = r.sha(OUT / 'results.json')
                if args.action == 'audit' or args.audit_sha256:
                    answer['audit_sha256'] = r.sha(AUDIT / 'audit.json')
        except Exception:
            answer['phase'] = state['phase']
            if state['owned'] is not None:
                try:
                    r.write_json(state['owned'] / 'failure.json', answer)
                except Exception:
                    pass
    print(json.dumps(answer, sort_keys=True))
    return int(answer['status'] != 'complete')


if __name__ == '__main__':
    raise SystemExit(main())
