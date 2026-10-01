"""Prospective local-only source-addition experiment; never grants source access.

Prepare freezes code/runtime/admission/preflights before source/model loading.
Run persists private audit checkpoints and matched streams. Verify replays all
fixed readouts; audit additionally rechecks pixels and a fixed inference sample.
No action on import, no resume, no automatic promotion of unified BRAN.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import fcntl
import json
import math
import os
from pathlib import Path
import random
import re
from urllib.parse import urlsplit

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run_bran_retinal_supervised_adaptation_v1 as ref
import bran_retinal_multisource_artifacts_v1 as artifacts
import bran_retinal_multisource_evaluation_v1 as evaluation
import bran_retinal_multisource_feature_export_v1 as export
import bran_retinal_multisource_readers_v1 as readers
import bran_retinal_multisource_training_v1 as training
from bran_retinal_multisource_sampling_v1 import BatchPlanner
from bran_retinal_multisource_patient_kernel_v1 import RetinalMultisourcePatientKernel

r = ref.r
ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'BRAN_RETINAL_MULTISOURCE_ADAPTATION_PROTOCOL_V1.json'
OUT = ROOT / 'BRAN_RETINAL_MULTISOURCE_ADAPTATION_V1'
AUDIT = ROOT / 'BRAN_RETINAL_MULTISOURCE_ADAPTATION_AUDIT_V1'
PRIVATE = ROOT / 'private_artifacts/bran_retinal_multisource_adaptation_v1'
ADMISSION = ROOT / 'BRAN_ODIR_RESEARCH_USE_ADMISSION_V1.json'
EVIDENCE = ROOT / 'source_admissions'
LOCK = ref.LOCK
SCHEMA = 'bran-retinal-multisource-adaptation-v1'
ARMS = ('source_control', 'multisource_candidate')
PHASES = ('authentication', 'export', 'training', 'evaluation', 'publication', 'audit')
ERROR = 'retinal multisource execution validation failed'
FILES = (
    'run_bran_retinal_multisource_adaptation_v1.py',
    'test_run_bran_retinal_multisource_adaptation_v1.py',
    'BRAN_RETINAL_MULTISOURCE_EXECUTION_DESIGN_V1.md',
    'BRAN_RETINAL_MULTISOURCE_PATIENT_DESIGN_V1.md',
    'BRAN_RETINAL_MULTISOURCE_EVALUATION_DESIGN_V1.md',
    'bran_retinal_multisource_artifacts_v1.py', 'test_bran_retinal_multisource_artifacts_v1.py',
    'bran_retinal_multisource_feature_export_v1.py', 'test_bran_retinal_multisource_feature_export_v1.py',
    'bran_retinal_multisource_evaluation_v1.py', 'test_bran_retinal_multisource_evaluation_v1.py',
    'bran_retinal_multisource_odir_evaluation_v1.py', 'test_bran_retinal_multisource_odir_evaluation_v1.py',
    'bran_retinal_multisource_training_v1.py', 'test_bran_retinal_multisource_training_v1.py',
    'bran_retinal_multisource_patient_kernel_v1.py', 'test_bran_retinal_multisource_patient_kernel_v1.py',
    'bran_retinal_multisource_sampling_v1.py', 'test_bran_retinal_multisource_sampling_v1.py',
    'bran_retinal_multisource_source_contract_v1.py', 'test_bran_retinal_multisource_source_contract_v1.py',
    'bran_retinal_multisource_readers_v1.py', 'test_bran_retinal_multisource_readers_v1.py',
    'bran_retinal_adaptation_training_v1.py', 'bran_retinal_adaptation_evaluation_v1.py',
    'bran_retinal_supervised_adaptation_kernel_v1.py',
    'run_bran_retinal_supervised_adaptation_v1.py', 'run_bran_retinal_group_readiness_v1.py',
)
PREFLIGHTS = {
    'BRAN_RETINAL_MULTISOURCE_SYNTHETIC_GPU_V1.json': '6eafaa8a7a714b57b121c5de3135497e8be0025750a1be74b7de38b0d19c4543',
    'BRAN_RETINAL_MULTISOURCE_SOURCE_BINDING_V1.json': '0c58d68abf817a2f16922c2644cc2148ede54f70260c111bddde2ab402427ba1',
    'BRAN_RETINAL_MULTISOURCE_PIXEL_BINDING_V1.json': '466fa5e67930d2b5ac6e712c87bdcb34cd4c1c63a42accf7aab579b44ea8afd9',
}


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def read_json(path):
    r.regular(path)
    return json.loads(Path(path).read_text())


def digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def admission(pin):
    """Authenticate a separately reviewed attestation, not infer legal permission.

    This module cannot create the admission/evidence files. A reviewer must first
    establish original-source identity and the local copy's permitted use.
    """
    require(digest(pin))
    value = read_json(ADMISSION)
    require(r.sha(ADMISSION) == pin and type(value) is dict and set(value) == {
        'schema', 'decision', 'scope', 'reviewer_role', 'review_date', 'source_url',
        'source_identity_reviewed', 'research_use_reviewed', 'evidence_filename',
        'evidence_sha256', 'qualified_metadata_audit_sha256', 'qualified_content_audit_sha256',
        'qualified_split_audit_sha256', 'patient_level_output_permitted', 'redistribution_authorized'})
    require(value['schema'] == 'bran-odir-research-use-admission-v1'
            and value['decision'] == 'admitted_for_local_research'
            and value['scope'] == 'retinal_representation_training_and_local_evaluation'
            and value['reviewer_role'] in ('principal_investigator', 'data_custodian', 'documented_source_review')
            and type(value['review_date']) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value['review_date'])
            and value['source_identity_reviewed'] is True and value['research_use_reviewed'] is True
            and value['patient_level_output_permitted'] is False and value['redistribution_authorized'] is False)
    date.fromisoformat(value['review_date'])
    require(type(value['source_url']) is str)
    url = urlsplit(value['source_url'])
    require(url.scheme == 'https' and url.hostname and not url.username and not url.password
            and not url.query and not url.fragment)
    filename = value['evidence_filename']
    require(type(filename) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.md', filename)
            and EVIDENCE.is_dir() and not EVIDENCE.is_symlink())
    evidence = EVIDENCE / filename
    r.regular(evidence)
    require(digest(value['evidence_sha256']) and r.sha(evidence) == value['evidence_sha256'])
    require(value['qualified_metadata_audit_sha256'] == '5f519e1304d0ef35aff9276ca88e6f2bc7516928bb8a9e5d5a6e082098bd56bf'
            and value['qualified_content_audit_sha256'] == '458e81841509463214413a783d44fff03bf3aa36bae3bb5cae67d9af386af84e'
            and value['qualified_split_audit_sha256'] == 'cc4c5f85f04693d14ea910325bf5f7cbea35bd692e5281832e79800d17605d44')
    return {'admission_sha256': pin, 'reviewed_evidence_sha256': value['evidence_sha256']}


def validate_preflight(name, value):
    require(name in PREFLIGHTS and type(value) is dict
            and value['schema'] == name.removesuffix('.json').lower().replace('_', '-')
            and value['status'] == 'passed' and value['training_admitted'] is False)
    if name == 'BRAN_RETINAL_MULTISOURCE_SYNTHETIC_GPU_V1.json':
        # Original synthetic-only receipt uses this field, not the real-data
        # preflights' patient_level_output_emitted flag. Preserve its schema.
        require(value['patient_data_used'] is False and value['real_data_training_run'] is False
                and value['device'] == 'mps' and value['feature_parity'] is True
                and value['finite_forward_backward_optimizer_ema'] is True
                and value['mps_rng_recorded'] is True and value['synthetic_arms_passed'] == [0, 1]
                and value['one_and_two_eye_groups_checked'] is True
                and value['full_batch_images'] == {'brset': 16, 'odir': 32}
                and value['base_sha256'] == ref.MODEL_SHA256 and value['torch_version'] == torch.__version__)
    else:
        require(value['patient_level_output_emitted'] is False and value['patient_processing_local_only'] is True)
        if name == 'BRAN_RETINAL_MULTISOURCE_SOURCE_BINDING_V1.json':
            require(value['images_read'] is False and value['real_data_training_run'] is False)
        else:
            require(value['model_fitting_run'] is False and value['patient_images_saved_or_displayed'] is False
                    and value['raw_and_decoded_bytes_authenticated'] is True
                    and value['matched_pixel_batches_exact'] is True
                    and value['fixed_normalization_and_flip_verified'] is True and value['archive_closed'] is True)


def preflights():
    for name, pin in PREFLIGHTS.items():
        value = read_json(ROOT / name)
        require(r.sha(ROOT / name) == pin)
        validate_preflight(name, value)
        require(all(r.sha(ROOT / filename) == saved for filename, saved in value['code_sha256'].items()))
    readers.authenticate_binding()
    authenticate_base()
    return dict(PREFLIGHTS)


def authenticate_base():
    # Hugging Face snapshots normally symlink to their immutable blob cache.
    # Accept the resolved regular blob ONLY with the already pinned SHA-256.
    target = ref.MODEL_FILE.resolve(strict=True)
    r.regular(target)
    require(r.sha(target) == ref.MODEL_SHA256 and r.sha(ref.MODEL_FILE) == ref.MODEL_SHA256)


def parameters():
    return {'steps_per_arm': 512, 'batch_size': 16, 'seed': 74191,
            'stream_seeds': {'brset': 74192, 'odir': 74193}, 'export_batch_size': 16,
            'source_weights': {'source_control': 0, 'multisource_candidate': 1},
            'image_size': 224, 'patches': 196, 'masked_patches': 147, 'flip_probability': .5,
            'lr': 1e-4, 'weight_decay': .04, 'betas': [.9, .95], 'clip_norm': 1., 'ema': .996,
            'bootstrap_draws': 2000, 'brset_bootstrap_seed': 74195, 'odir_bootstrap_seed': 74194,
            'inference_images_per_source_arm': 32, 'inference_atol': 1e-5, 'inference_rtol': 1e-5,
            'device': 'mps', 'resume_supported': False, 'unified_model_promotion_permitted': False}


def expected_protocol(admission_pin):
    # Admission must fail BEFORE loading any sources, checkpoint or outputs.
    reviewed = admission(admission_pin)
    return {'schema': SCHEMA, 'admission': reviewed, 'preflights': preflights(),
            'code_sha256': {name: r.sha(ROOT / name) for name in FILES},
            'runtime': ref.runtime(), 'parameters': parameters(),
            'base': {'model': ref.MODEL, 'sha256': ref.MODEL_SHA256,
                     'prefix_tokens': 5, 'embedding_dim': 384, 'download_permitted': False},
            'patient_level_output_permitted': False}


def prepare(admission_pin):
    value = expected_protocol(admission_pin)
    r.absent(PROTOCOL, OUT, AUDIT, PRIVATE)
    r.write_json(PROTOCOL, value)
    return r.sha(PROTOCOL)


def load_protocol(pin):
    require(digest(pin))
    value = read_json(PROTOCOL)
    require(r.sha(PROTOCOL) == pin)
    require(r.equal(value, expected_protocol(value['admission']['admission_sha256'])))
    return value


def progress(state, phase, arm=None, completed=0):
    require(phase in PHASES and (arm is None or arm in ARMS)
            and type(completed) is int and 0 <= completed <= parameters()['steps_per_arm'])
    state['phase'] = phase
    if state.get('owned') is not None:
        temporary = state['owned'] / 'progress.tmp'
        r.absent(temporary)
        r.write_json(temporary, {'phase': phase, 'arm': arm, 'completed_steps': completed,
                                 'patient_level_output_emitted': False})
        os.replace(temporary, state['owned'] / 'progress.json')


def evaluation_inputs(sources):
    b = sources._brset['arrays']
    o = sources.private_specs['odir']
    # Do not persist patient IDs, source paths or per-image fingerprints here.
    return {**{'brset_' + k: np.asarray(b[k]).copy() for k in ('groups', 'split', 'labels', 'observed', 'ages')},
            **{'odir_' + k: np.asarray(o[k]).copy() for k in ('image_groups', 'group_split', 'labels', 'observed', 'usable')}}


def score(inputs, by_arm):
    b = {k.removeprefix('brset_'): v for k, v in inputs.items() if k.startswith('brset_')}
    o = {k.removeprefix('odir_'): v for k, v in inputs.items() if k.startswith('odir_')}
    with threadpool_limits(limits=1):
        return evaluation.evaluate(b, o, export.arrange_arms(by_arm), draws=parameters()['bootstrap_draws'])


def planner(sources):
    p = parameters()
    return BatchPlanner(sources.private_specs, steps=p['steps_per_arm'], batch_size=p['batch_size'], seeds=p['stream_seeds'])


def validate_receipt(receipt, arm):
    p = parameters()
    require(type(receipt) is dict and set(receipt) == {'schema', 'resume_supported', 'training_config',
        'kernel_state', 'optimizer_state', 'scheduler', 'rng', 'completed_step', 'source_forwards_per_step'})
    require(receipt['schema'] == 'bran-retinal-multisource-training-receipt-v1'
            and receipt['resume_supported'] is False and receipt['completed_step'] == p['steps_per_arm']
            and receipt['source_forwards_per_step'] == {'brset': 1, 'odir': 1})
    config = {'source_weight': float(p['source_weights'][arm]), 'source_heads': {'brset': 13, 'odir': 8},
              'lr': p['lr'], 'weight_decay': p['weight_decay'], 'ema': p['ema'], 'seed': p['seed'],
              'steps': p['steps_per_arm'], 'optimizer': {'name': 'AdamW', 'betas': tuple(p['betas'])}}
    require(ref.private_equal(receipt['training_config'], config))
    require(ref.private_equal(receipt['scheduler'], {'name': 'linear_warmup_cosine', 'warmup_fraction': .1,
        'warmup_steps': max(1, math.ceil(p['steps_per_arm'] * .1)), 'min_ratio': .1,
        'step': p['steps_per_arm'], 'steps': p['steps_per_arm'],
        'last_lr': training.old.learning_rate_at_step(p['steps_per_arm'] - 1, steps=p['steps_per_arm'], lr=p['lr'])}))
    require(type(receipt['kernel_state']) is dict and receipt['kernel_state']
            and all(isinstance(v, torch.Tensor) and torch.isfinite(v).all().item() for v in receipt['kernel_state'].values()))
    require(set(receipt['rng']) == {'python', 'numpy', 'torch_cpu', 'torch_mps'}
            and isinstance(receipt['rng']['torch_cpu'], torch.Tensor)
            and (p['device'] != 'mps' or isinstance(receipt['rng']['torch_mps'], torch.Tensor)))
    # Validate recorded states using isolated generators, never changing replay RNG.
    random.Random().setstate(receipt['rng']['python'])
    np.random.RandomState().set_state(receipt['rng']['numpy'])
    torch.Generator(device='cpu').set_state(receipt['rng']['torch_cpu'])
    mps_rng = receipt['rng']['torch_mps']
    require(mps_rng is None or (isinstance(mps_rng, torch.Tensor) and mps_rng.dtype == torch.uint8
                               and mps_rng.ndim == 1 and mps_rng.numel() > 0))
    require(set(receipt['optimizer_state']) == {'state', 'param_groups'} and receipt['optimizer_state']['state'])
    require(all(torch.isfinite(v).all().item() for entry in receipt['optimizer_state']['state'].values()
                for v in entry.values() if isinstance(v, torch.Tensor)))
    groups = receipt['optimizer_state']['param_groups']
    require(type(groups) is list and len(groups) == 1
            and groups[0]['lr'] == receipt['scheduler']['last_lr']
            and groups[0]['betas'] == tuple(p['betas']) and groups[0]['weight_decay'] == p['weight_decay'])


def execute(state):
    p = parameters()
    if p['device'] == 'mps':
        ref.device()  # No silent CPU fallback on a real run.
    bundle = artifacts.PrivateBundle.create(PRIVATE)
    with readers.QualifiedSources() as sources:
        inputs = evaluation_inputs(sources)
        bundle.write_npz('evaluation_inputs.npz', inputs)
        base = ref.load_base()
        progress(state, 'export')
        features = {'base': export.export_arm(base, sources, device=p['device'], batch_size=p['export_batch_size'])}
        bundle.write_npz('base_features.npz', features['base'])
        streams = {}
        for arm in ARMS:
            progress(state, 'training', arm)
            stream = planner(sources)

            def checkpoint(receipt, arm=arm):
                bundle.checkpoint(arm, receipt)
                progress(state, 'training', arm, receipt['completed_step'])

            trained, receipt = training.train_arm(base, (sources.materialize(item) for item in stream),
                source_weight=p['source_weights'][arm],
                positive_weights={k: torch.from_numpy(v.copy()) for k, v in sources.private_weights.items()},
                steps=p['steps_per_arm'], seed=p['seed'], device=p['device'], checkpoint_callback=checkpoint)
            validate_receipt(receipt, arm)
            streams[arm] = stream.receipt()
            progress(state, 'export', arm)
            features[arm] = export.export_arm(trained, sources, device=p['device'], batch_size=p['export_batch_size'])
            bundle.write_npz(arm + '_features.npz', features[arm])
            del trained, receipt
            if p['device'] == 'mps':
                torch.mps.empty_cache()
        require(streams[ARMS[0]] == streams[ARMS[1]])
        bundle.write_json('streams.json', streams)
        progress(state, 'evaluation')
        summary, readouts = score(inputs, features)
        bundle.write_torch('readouts.pt', readouts)
    return summary, bundle.seal(p['steps_per_arm'])


def replay():
    """Caller authenticates every private file BEFORE deserializing own receipts."""
    saved = ref.load_arrays(PRIVATE / 'evaluation_inputs.npz')
    streams = read_json(PRIVATE / 'streams.json')
    require(type(streams) is dict and set(streams) == set(ARMS))
    with readers.QualifiedSources() as sources:
        require(ref.private_equal(saved, evaluation_inputs(sources)))
        stream = planner(sources)
        for _ in stream:
            pass
        require(all(streams[arm] == stream.receipt() for arm in ARMS))
    features = {arm: ref.load_arrays(PRIVATE / (arm + '_features.npz')) for arm in export.ARMS}
    summary, readouts = score(saved, features)
    stored = torch.load(PRIVATE / 'readouts.pt', map_location='cpu', weights_only=False)
    require(ref.private_equal(readouts, stored))
    for arm in ARMS:
        validate_receipt(torch.load(PRIVATE / (arm + '.pt'), map_location='cpu', weights_only=False), arm)
    return summary


def result_value(pin, summary):
    evaluation.validate_summary(summary)
    return {'schema': SCHEMA, 'status': 'comparison_completed', 'protocol_sha256': pin,
            'evaluation': summary, 'model_training_performed': True, 'unified_model_promoted': False,
            'clinical_validation_established': False, 'patient_level_output_emitted': False}


def marker(pin):
    return {'schema': SCHEMA, 'status': 'completed', 'protocol_sha256': pin,
            'manifest_sha256': r.sha(OUT / 'manifest.json'), 'patient_level_output_emitted': False}


def payload(pin, *, terminal, replay_results):
    load_protocol(pin)
    require(not (AUDIT / 'failure.json').exists())
    r.inventory(OUT, ('results.json', 'manifest.json', 'progress.json', *(('success.json',) if terminal else ())))
    require(read_json(OUT / 'progress.json') == {'phase': 'publication', 'arm': None,
                                               'completed_steps': 0, 'patient_level_output_emitted': False})
    manifest = read_json(OUT / 'manifest.json')
    require(set(manifest) == {'schema', 'protocol_sha256', 'results_sha256', 'private_sha256', 'patient_level_output_emitted'}
            and manifest['schema'] == SCHEMA and manifest['protocol_sha256'] == pin
            and manifest['patient_level_output_emitted'] is False and manifest['results_sha256'] == r.sha(OUT / 'results.json'))
    artifacts.authenticate(PRIVATE, manifest['private_sha256'])
    stored = read_json(OUT / 'results.json')
    summary = replay() if replay_results else stored['evaluation']
    require(r.equal(stored, result_value(pin, summary)))
    if terminal:
        require(read_json(OUT / 'success.json') == marker(pin))
    return summary


def run(pin, state):
    load_protocol(pin)
    r.absent(OUT, AUDIT, PRIVATE)
    OUT.mkdir()
    state['owned'] = OUT
    summary, private_manifest = execute(state)
    progress(state, 'publication')
    r.write_json(OUT / 'results.json', result_value(pin, summary))
    r.write_json(OUT / 'manifest.json', {'schema': SCHEMA, 'protocol_sha256': pin,
        'results_sha256': r.sha(OUT / 'results.json'), 'private_sha256': private_manifest,
        'patient_level_output_emitted': False})
    payload(pin, terminal=False, replay_results=True)
    r.write_json(OUT / 'success.json', marker(pin))  # Exclusive terminal only after replay passes.
    state['owned'] = None


class _Subset:
    def __init__(self, sources, indices):
        self.sources, self.indices = sources, indices
        self.private_specs = {k: {'image_groups': np.arange(len(v))} for k, v in indices.items()}

    def read(self, source, index):
        return self.sources.read(source, int(self.indices[source][index]))


def inference_audit():
    p = parameters()
    with readers.QualifiedSources() as sources:
        for source in ('brset', 'odir'):
            def check(index):
                sources.read(source, index)  # Every raw and decoded image is reauthenticated.
                return True
            with ThreadPoolExecutor(max_workers=4) as pool:
                require(all(pool.map(check, range(len(sources.private_specs[source]['image_groups'])))))
        indices = {}
        for source in ('brset', 'odir'):
            n = len(sources.private_specs[source]['image_groups'])
            require(n >= p['inference_images_per_source_arm'])
            indices[source] = np.linspace(0, n - 1, p['inference_images_per_source_arm'], dtype=np.int64)
        subset, base = _Subset(sources, indices), ref.load_base()
        for arm in export.ARMS:
            if arm == 'base':
                model = base
            else:
                receipt = torch.load(PRIVATE / (arm + '.pt'), map_location='cpu', weights_only=False)
                validate_receipt(receipt, arm)
                model = RetinalMultisourcePatientKernel(base)
                model.load_state_dict(receipt['kernel_state'], strict=True)
                model.eval()
            actual = export.export_arm(model, subset, device=p['device'], batch_size=p['export_batch_size'])
            saved = ref.load_arrays(PRIVATE / (arm + '_features.npz'))
            require(all(np.allclose(actual[k], saved[k][indices[k]], atol=p['inference_atol'], rtol=p['inference_rtol'])
                        for k in actual))
            del model


def audit_value(pin):
    return {'schema': SCHEMA, 'status': 'authenticated', 'protocol_sha256': pin,
            'terminal_manifest_sha256': r.sha(OUT / 'manifest.json'),
            'all_admitted_raw_and_decoded_pixels_reauthenticated': True,
            'inference_images_per_source_arm': parameters()['inference_images_per_source_arm'],
            'all_inference_replayed': False, 'full_readout_and_aggregate_replay': True,
            'matched_training_stream_replayed': True, 'encoder_training_repeated': False,
            'unified_model_promoted': False, 'patient_level_output_emitted': False}


def audit(pin, state):
    payload(pin, terminal=True, replay_results=True)
    r.absent(AUDIT)
    AUDIT.mkdir()
    state['owned'] = AUDIT
    progress(state, 'audit')
    inference_audit()
    payload(pin, terminal=True, replay_results=False)  # Closure/private hashes still unchanged.
    progress(state, 'publication')
    r.write_json(AUDIT / 'audit.json', audit_value(pin))
    r.write_json(AUDIT / 'success.json', {'audit_sha256': r.sha(AUDIT / 'audit.json'), 'protocol_sha256': pin})
    state['owned'] = None


def authenticate_audit(pin, audit_pin):
    payload(pin, terminal=True, replay_results=True)
    r.inventory(AUDIT, ('audit.json', 'success.json', 'progress.json'))
    require(read_json(AUDIT / 'progress.json') == {'phase': 'publication', 'arm': None,
        'completed_steps': 0, 'patient_level_output_emitted': False})
    require(digest(audit_pin) and r.sha(AUDIT / 'audit.json') == audit_pin
            and read_json(AUDIT / 'audit.json') == audit_value(pin)
            and read_json(AUDIT / 'success.json') == {'audit_sha256': audit_pin, 'protocol_sha256': pin})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'verify', 'audit'))
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--admission-sha256')
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
                    pin = prepare(args.admission_sha256)
                elif args.action == 'run':
                    run(pin, state)
                elif args.action == 'audit':
                    audit(pin, state)
                elif args.audit_sha256:
                    authenticate_audit(pin, args.audit_sha256)
                else:
                    payload(pin, terminal=True, replay_results=True)
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
