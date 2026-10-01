"""Closed prospective lifecycle for the retinal supervised-adaptation comparison.

The command never discovers data before entering ``r.quiet`` and never emits a
patient-level value.  Source admission, model construction, and evaluation are
intentionally factored into patchable local helpers for synthetic lifecycle
coverage; no action is launched by importing this file.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import bran_retinal_adaptation_evaluation_v1 as e
from bran_retinal_adaptation_training_v1 import train_arm
from bran_retinal_supervised_adaptation_kernel_v1 import RetinalSupervisedAdaptationKernel

import run_bran_retinal_content_admission_v1 as a
import run_bran_retinal_group_readiness_v1 as r

ROOT = r.ROOT
PROTOCOL = ROOT / "BRAN_RETINAL_SUPERVISED_ADAPTATION_PROTOCOL_V1.json"
OUT = ROOT / "BRAN_RETINAL_SUPERVISED_ADAPTATION_V1"
AUDIT = ROOT / "BRAN_RETINAL_SUPERVISED_ADAPTATION_AUDIT_V1"
PRIVATE = ROOT / "private_artifacts/bran_retinal_supervised_adaptation_v1"
LOCK = r.LOCK
ADMISSION_PROTOCOL_SHA256 = "ee76c14b6b59ad090772508feafa9bf5e19a4258ee3d47e404d51e5d873ad7e6"
MODEL = "vit_small_patch16_dinov3.lvd1689m"
MODEL_FILE = Path("/Users/ethanwu/.cache/huggingface/hub/models--timm--vit_small_patch16_dinov3.lvd1689m/snapshots/3bf4720a82ec2066db88137180ff1f83a675cef0/model.safetensors")
MODEL_SHA256 = "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
FILES = (
    "run_bran_retinal_supervised_adaptation_v1.py", "test_run_bran_retinal_supervised_adaptation_v1.py",
    "bran_retinal_supervised_adaptation_kernel_v1.py", "bran_retinal_adaptation_training_v1.py",
    "bran_retinal_adaptation_evaluation_v1.py", "BRAN_RETINAL_SUPERVISED_ADAPTATION_DESIGN_V1.md",
    "test_bran_retinal_supervised_adaptation_kernel_v1.py", "test_bran_retinal_architecture_binding_v1.py",
    "test_bran_retinal_adaptation_training_v1.py", "test_bran_retinal_adaptation_evaluation_v1.py",
    "check_bran_retinal_adaptation_synthetic_gpu_v1.py",
)
SCHEMA = "bran-retinal-supervised-adaptation-v1"
PRIVATE_FILES = ("upstream.json", "grouping.npz", "readouts.pt", "stream.json",
                 "base_features.npy", "masked_control_features.npy", "supervised_candidate_features.npy",
                 "masked_control.pt", "supervised_candidate.pt")
TERMINAL = ("results.json", "manifest.json", "progress.json")
PHASES = frozenset(("authentication", "training", "export", "publication", "audit"))


def require(value):
    if not value:
        raise ValueError("retinal supervised adaptation validation failed")


def runtime():
    packages = ("torch", "timm", "scikit-learn", "numpy", "Pillow", "safetensors", "scipy", "threadpoolctl")
    return {**r.runtime(), **{name: importlib.metadata.version(name) for name in packages}}


def write_private_json(path, value):
    with Path(path).open("x") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)


def progress(state, phase, *, arm=None, steps=0):
    require(phase in PHASES)
    require(arm is None or arm in ('masked_control', 'supervised_candidate'))
    require(type(steps) is int and 0 <= steps <= parameters()['steps_per_arm'])
    state["phase"] = phase
    if state.get("owned") is not None:
        tmp = state["owned"] / "progress.tmp"
        r.absent(tmp)
        r.write_json(tmp, {"phase": phase, 'arm': arm, 'completed_steps': steps,
                           "patient_level_output_emitted": False})
        os.replace(tmp, state["owned"] / "progress.json")


def validate_progress(value, *, terminal):
    require(type(value) is dict and set(value) == {"phase", 'arm', 'completed_steps', "patient_level_output_emitted"})
    require(type(value["phase"]) is str and value["phase"] in PHASES and value["patient_level_output_emitted"] is False)
    require(value['arm'] is None or value['arm'] in ('masked_control', 'supervised_candidate'))
    require(type(value['completed_steps']) is int and 0 <= value['completed_steps'] <= parameters()['steps_per_arm'])
    if terminal:
        require(value == {"phase": "publication", 'arm': None, 'completed_steps': 0,
                          "patient_level_output_emitted": False})


def upstream(admission_audit_sha256):
    require(type(admission_audit_sha256) is str and len(admission_audit_sha256) == 64)
    a.authenticate_audit(ADMISSION_PROTOCOL_SHA256, admission_audit_sha256)
    require(a.OUT.is_dir() and a.AUDIT.is_dir())
    return {
        "admission_protocol_sha256": ADMISSION_PROTOCOL_SHA256,
        "admission_audit_sha256": admission_audit_sha256,
        "admission_terminal_manifest_sha256": r.sha(a.OUT / "manifest.json"),
        "admission_audit_manifest_sha256": r.sha(a.AUDIT / "manifest.json"),
    }


def parameters():
    return {"steps_per_arm": 512, "batch_size": 16, "train_seed": 73191, "stream_seed": 73192,
            "summary_seed": 73193, "image_size": 224, "patches": 196, "masked_patches": 147,
            "label_weights": {"masked_control": 0, "supervised_candidate": 1},
            "lr": .0001, "weight_decay": .04, "ema": .996, "bootstrap_draws": 2000,
            "inference_replay_atol": 1e-5, "inference_replay_rtol": 1e-5,
            "patient_level_output_permitted": False, "training_authorized_by_protocol": True,
            "resume_supported": False}


def expected_protocol(up):
    return {"schema": SCHEMA, "upstream": up, "code_sha256": {name: r.sha(ROOT / name) for name in FILES},
            "runtime": runtime(), "base": {"model": MODEL, "path_sha256": MODEL_SHA256,
            "prefix_tokens": 5, "embedding_dim": 384, "pretrained_download": False}, "parameters": parameters(),
            'selection': selection_fingerprint(), 'synthetic_gpu_check_sha256': gpu_smoke_binding()}


def gpu_smoke_binding():
    path = ROOT / 'BRAN_RETINAL_ADAPTATION_SYNTHETIC_GPU_V1.json'
    r.regular(path)
    value = json.loads(path.read_text())
    require(value['schema'] == 'bran-retinal-adaptation-synthetic-gpu-v1'
            and value['status'] == 'passed' and value['patient_data_used'] is False
            and value['retinal_training_experiment_run'] is False and value['device'] == 'mps'
            and value['feature_parity'] is True and value['finite_forward_backward_optimizer_ema'] is True
            and value['mps_rng_recorded'] is True and value['base_sha256'] == MODEL_SHA256
            and value['torch_version'] == torch.__version__)
    require(value['code_sha256'] == {name: r.sha(ROOT / name) for name in (
        'check_bran_retinal_adaptation_synthetic_gpu_v1.py',
        'bran_retinal_supervised_adaptation_kernel_v1.py', 'bran_retinal_adaptation_training_v1.py')})
    return r.sha(path)


def selection_fingerprint():
    source = load_source()['arrays']
    digests = {}
    for name, values in source.items():
        digest = hashlib.sha256(str(values.dtype).encode() + str(values.shape).encode())
        digest.update(np.ascontiguousarray(values).tobytes())
        digests[name] = digest.hexdigest()
    return {'private_array_sha256': digests, 'images_coarsened': e.coarse(len(source['split'])),
            'patients_coarsened': e.coarse(len(np.unique(source['groups']))),
            'patient_split_counts_coarsened': {part: e.coarse(len(np.unique(source['groups'][source['split'] == code])))
                                               for code, part in enumerate(('train', 'validation', 'test'))}}


def load_protocol(pin):
    r.regular(PROTOCOL)
    require(type(pin) is str and len(pin) == 64 and r.sha(PROTOCOL) == pin)
    r.regular(PRIVATE / "upstream.json", private=True)
    protocol = json.loads(PROTOCOL.read_text())
    up = protocol.get("upstream", {})
    require(r.equal(protocol, expected_protocol(up)))
    # Reauthenticate rather than trusting an earlier admission progress file.
    require(r.equal(upstream(up["admission_audit_sha256"]), up))
    require(r.equal(json.loads((PRIVATE / 'upstream.json').read_text()), up))
    require(MODEL_FILE.is_file() and r.sha(MODEL_FILE) == MODEL_SHA256)
    return protocol


def prepare(admission_audit_sha256, state):
    r.absent(PROTOCOL, OUT, AUDIT, PRIVATE)
    up = upstream(admission_audit_sha256)
    require(MODEL_FILE.is_file() and r.sha(MODEL_FILE) == MODEL_SHA256)
    source = load_source()
    label_policy(source)
    protocol = expected_protocol(up)
    progress(state, "authentication")
    PRIVATE.mkdir(mode=0o700)
    write_private_json(PRIVATE / "upstream.json", up)
    r.write_json(PROTOCOL, protocol)
    return r.sha(PROTOCOL)


def device():
    require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0'
            and torch.backends.mps.is_built() and torch.backends.mps.is_available())
    return 'mps'


def load_base():
    import timm
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    torch.manual_seed(parameters()['train_seed'])
    require(r.sha(MODEL_FILE) == MODEL_SHA256)
    base = timm.create_model(MODEL, pretrained=False, num_classes=0)
    base.load_state_dict(load_file(str(MODEL_FILE), device='cpu'), strict=True)
    require(base.embed_dim == 384 and base.num_prefix_tokens == 5
            and base.patch_drop is None and not base.rope.aug_active)
    return base.eval()


def load_source():
    """Read only previously admitted rows; no new eligibility or source pairing."""
    _, m, b, _, flags = a.authenticate(ADMISSION_PROTOCOL_SHA256)
    keep = np.flatnonzero(flags['admitted'][b['source_rows']])
    rows = b['source_rows'][keep]
    require(len(rows) > 0)
    records = json.loads((a.PRIVATE / 'brset_inventory.json').read_text())
    _, groups = np.unique(m['patient_ids'][rows], return_inverse=True)
    arrays = {'source_rows': rows.copy(), 'groups': groups.astype(np.int64),
              'split': m['split'][rows].copy(), 'labels': m['labels'][rows].copy(),
              'observed': m['observed'][rows].copy(), 'ages': m['ages'][rows].copy(),
              'raw_sha256': b['raw_sha256'][keep].copy(),
              'decoded224_sha256': b['decoded224_sha256'][keep].copy()}
    selected_records = [records[i] for i in keep]
    require(all(record['relative_path'] == str(m['image_ids'][row]) + '.jpg'
                for record, row in zip(selected_records, rows)))
    return {'arrays': arrays, 'records': selected_records}


def grouped(source, features):
    q = source['arrays']
    return e.group_patients(q['groups'], q['split'], q['labels'], q['observed'], q['ages'], features)


def label_policy(source):
    q = source['arrays']
    dummy = {arm: np.zeros((len(q['split']), 1)) for arm in e.ARMS}
    patients = grouped(source, dummy)
    primary = [e.ENDPOINTS.index(name) for name in e.PRIMARY]
    require(all(np.all(e.support(patients['labels'], patients['observed'], patients['split'], code)[primary] >= e.MIN_CLASS)
                for code in (0, 1, 2)))
    return e.training_label_policy(q['labels'], q['observed'], q['split'], patients)


def read_pixels(source, index):
    q = source['arrays']
    raw = a.stable_bytes(source['records'][int(index)], a.BRSET)
    expected = bytes(q['raw_sha256'][int(index)]).hex()
    require(hashlib.sha256(raw).hexdigest() == expected)
    pixels = a.decode_jpeg(raw, expected, size=224)
    require(hashlib.sha256(pixels.tobytes()).digest() == bytes(q['decoded224_sha256'][int(index)]))
    return pixels


def pixel_batch(source, indices, flips=None):
    with ThreadPoolExecutor(max_workers=4) as pool:
        pixels = np.stack(list(pool.map(lambda i: read_pixels(source, i), indices)))
    if flips is not None:
        pixels[flips] = pixels[flips, :, ::-1, :]
    images = pixels.astype(np.float32) / np.float32(255)
    images = (images - np.asarray([.485, .456, .406], np.float32)) / np.asarray([.229, .224, .225], np.float32)
    return torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2)))


class BatchStream:
    def __init__(self, source, usable):
        self.source, self.usable = source, usable
        self.digest = hashlib.sha256()
        self.seen = set()
        self.seen_people = set()
        self.steps = 0

    def __iter__(self):
        p, q = parameters(), self.source['arrays']
        train = np.flatnonzero(q['split'] == 0)
        people = np.unique(q['groups'][train])
        by_person = [train[q['groups'][train] == person] for person in people]
        rng = np.random.default_rng(p['stream_seed'])
        for _ in range(p['steps_per_arm']):
            sampled_people = rng.integers(0, len(people), p['batch_size'])
            indices = np.asarray([rng.choice(by_person[i]) for i in sampled_people], np.int64)
            flips = rng.random(len(indices)) < .5
            mask = np.zeros((len(indices), p['patches']), bool)
            for row in mask:
                row[rng.choice(p['patches'], p['masked_patches'], replace=False)] = True
            known = q['observed'][indices] & self.usable[None, :]
            labels = np.where(known, q['labels'][indices], np.nan).astype(np.float32)
            for value in (q['source_rows'][indices], flips, mask, known):
                self.digest.update(np.ascontiguousarray(value).tobytes())
            self.seen.update(indices.tolist()); self.seen_people.update(q['groups'][indices].tolist())
            self.steps += 1
            yield pixel_batch(self.source, indices, flips), torch.from_numpy(mask), torch.from_numpy(labels), torch.from_numpy(known)

    def receipt(self):
        return {'sha256': self.digest.hexdigest(), 'steps': self.steps,
                'images_sampled_coarsened': e.coarse(len(self.seen)),
                'patients_sampled_coarsened': e.coarse(len(self.seen_people))}


def save_private(path, value, kind):
    with Path(path).open('xb') as handle:
        os.fchmod(handle.fileno(), 0o600)
        if kind == 'torch': torch.save(value, handle)
        elif kind == 'npz': np.savez_compressed(handle, **value)
        elif kind == 'npy': np.save(handle, value, allow_pickle=False)
        else: raise ValueError('invalid private serialization')


def load_arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def export_features(encoder, source, *, indices=None):
    selected = np.arange(len(source['records'])) if indices is None else np.asarray(indices)
    target = device()
    encoder = encoder.to(target).eval()
    features = []
    for start in range(0, len(selected), parameters()['batch_size']):
        images = pixel_batch(source, selected[start:start + parameters()['batch_size']]).to(target)
        with torch.inference_mode():
            if isinstance(encoder, RetinalSupervisedAdaptationKernel):
                z = encoder.encode_student(images)
            else:
                tokens = encoder.forward_features(images)
                require(tokens.ndim == 3 and tokens.shape[1:] == (201, 384))
                z = tokens[:, 5:].mean(1)
            result = z.float().cpu().numpy()
        require(result.shape == (len(images), 384) and np.all(np.isfinite(result)))
        features.append(result)
    encoder.to('cpu')
    return np.concatenate(features)


def evaluations(patients):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        readouts = {name: e.fit_readouts(patients, include_age=age)
                    for name, age in (('primary', False), ('secondary', True))}
        summaries = {name: e.summarize(patients, value, draws=parameters()['bootstrap_draws'],
                                       seed=parameters()['summary_seed']) for name, value in readouts.items()}
    return readouts, summaries


def execute_comparison(protocol, state):
    target = device()
    source = load_source()
    weights, usable = label_policy(source)
    save_private(PRIVATE / 'grouping.npz', source['arrays'], 'npz')
    base = load_base()
    progress(state, 'export')
    features = {'base': export_features(base, source)}
    save_private(PRIVATE / 'base_features.npy', features['base'], 'npy')
    streams = {}
    for arm, coefficient in parameters()['label_weights'].items():
        progress(state, 'training', arm=arm)
        stream = BatchStream(source, usable)
        model, receipt = train_arm(base, stream, label_weight=coefficient,
            positive_weight=torch.from_numpy(weights), steps=parameters()['steps_per_arm'],
            lr=parameters()['lr'], weight_decay=parameters()['weight_decay'], ema=parameters()['ema'],
            seed=parameters()['train_seed'], device=target,
            checkpoint_callback=lambda saved, arm=arm: progress(state, 'training', arm=arm,
                                                                 steps=saved['completed_step']))
        save_private(PRIVATE / (arm + '.pt'), receipt, 'torch')
        streams[arm] = stream.receipt()
        require(stream.steps == parameters()['steps_per_arm'])
        progress(state, 'export')
        features[arm] = export_features(model, source)
        save_private(PRIVATE / (arm + '_features.npy'), features[arm], 'npy')
        del model, receipt
    require(streams['masked_control'] == streams['supervised_candidate'])
    write_private_json(PRIVATE / 'stream.json', streams)
    readouts, summaries = evaluations(grouped(source, features))
    save_private(PRIVATE / 'readouts.pt', readouts, 'torch')
    return summaries


def result_value(protocol, pin, replay):
    require(type(replay) is dict and set(replay) == {"primary", "secondary"})
    return {"schema": SCHEMA, "status": "prospective_comparison_completed", "protocol_sha256": pin,
            "upstream": protocol["upstream"], "primary": replay["primary"], "secondary": replay["secondary"],
            "retained_unified_embedding_promoted": False, "patient_level_output_emitted": False,
            "model_training_performed": True, "training_authorized_by_this_result": False}


def manifest(pin):
    return {"schema": SCHEMA, "protocol_sha256": pin, "results_sha256": r.sha(OUT / "results.json"),
            "private_sha256": {name: r.sha(PRIVATE / name) for name in PRIVATE_FILES},
            "patient_level_output_emitted": False}


def replay_private(protocol):
    """Full fixed-readout/aggregate replay, never encoder retraining."""
    source = load_source()
    saved = load_arrays(PRIVATE / 'grouping.npz')
    require(set(saved) == set(source['arrays']) and all(np.array_equal(saved[k], v, equal_nan=True)
                                                      for k, v in source['arrays'].items()))
    streams = json.loads((PRIVATE / 'stream.json').read_text())
    require(set(streams) == set(parameters()['label_weights'])
            and streams['masked_control'] == streams['supervised_candidate']
            and streams['masked_control']['steps'] == parameters()['steps_per_arm'])
    features = {arm: np.load(PRIVATE / (arm + '_features.npy'), allow_pickle=False) for arm in e.ARMS}
    readouts, summaries = evaluations(grouped(source, features))
    expected = torch.load(PRIVATE / 'readouts.pt', map_location='cpu', weights_only=False)
    require(private_equal(readouts, expected))
    for arm in parameters()['label_weights']:
        load_receipt(arm)
    return summaries


def private_equal(left, right):
    if type(left) is np.ndarray:
        return type(right) is np.ndarray and left.dtype == right.dtype and np.array_equal(left, right, equal_nan=True)
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if type(left) is dict:
        return type(right) is dict and set(left) == set(right) and all(private_equal(left[k], right[k]) for k in left)
    if type(left) in (tuple, list):
        return type(left) is type(right) and len(left) == len(right) and all(private_equal(x, y) for x, y in zip(left, right))
    return type(left) is type(right) and left == right


def load_receipt(arm):
    receipt = torch.load(PRIVATE / (arm + '.pt'), map_location='cpu', weights_only=False)
    p = parameters()
    require(receipt['schema'] == 'bran-retinal-adaptation-training-receipt-v1'
            and receipt['resume_supported'] is False and receipt['completed_step'] == p['steps_per_arm'])
    expected = {'label_weight': float(p['label_weights'][arm]), 'lr': p['lr'],
                'weight_decay': p['weight_decay'], 'ema': p['ema'], 'seed': p['train_seed'],
                'steps': p['steps_per_arm'], 'optimizer': {'name': 'AdamW', 'betas': (.9, .95)}}
    require(private_equal(receipt['training_config'], expected))
    require(set(receipt['rng']) == {'python', 'numpy', 'torch_cpu', 'torch_mps'}
            and receipt['scheduler']['step'] == p['steps_per_arm'] and receipt['optimizer_state']['state'])
    require(all(torch.isfinite(x).all().item() for x in receipt['kernel_state'].values()))
    return receipt


def audit_sources_and_inference(protocol):
    source = load_source()
    def check_raw(i):
        raw = a.stable_bytes(source['records'][i], a.BRSET)
        require(hashlib.sha256(raw).digest() == bytes(source['arrays']['raw_sha256'][i]))
        return True
    with ThreadPoolExecutor(max_workers=4) as pool:
        require(all(pool.map(check_raw, range(len(source['records'])))))
    indices = np.unique(np.linspace(0, len(source['records']) - 1, min(32, len(source['records'])), dtype=np.int64))
    base = load_base()
    for arm in e.ARMS:
        if arm == 'base': encoder = base
        else:
            encoder = RetinalSupervisedAdaptationKernel(base)
            encoder.load_state_dict(load_receipt(arm)['kernel_state'], strict=True)
        actual = export_features(encoder, source, indices=indices)
        stored = np.load(PRIVATE / (arm + '_features.npy'), allow_pickle=False)[indices]
        require(np.allclose(actual, stored, atol=parameters()['inference_replay_atol'],
                            rtol=parameters()['inference_replay_rtol']))
        del encoder


def authenticate(pin, *, replay=False):
    protocol = load_protocol(pin)
    require(not (AUDIT / "failure.json").exists())
    r.inventory(OUT, TERMINAL)
    validate_progress(json.loads((OUT / "progress.json").read_text()), terminal=True)
    r.inventory(PRIVATE, PRIVATE_FILES, private=True)
    require(r.equal(json.loads((OUT / "manifest.json").read_text()), manifest(pin)))
    saved = json.loads((OUT / "results.json").read_text())
    value = replay_private(protocol) if replay else {key: saved[key] for key in ('primary', 'secondary')}
    require(r.equal(json.loads((OUT / "results.json").read_text()), result_value(protocol, pin, value)))
    return protocol, value


def run(pin, state):
    protocol = load_protocol(pin)
    r.inventory(PRIVATE, ("upstream.json",), private=True)
    r.absent(OUT, AUDIT)
    OUT.mkdir(); state["owned"] = OUT
    progress(state, "training")
    replay = execute_comparison(protocol, state)
    require(type(replay) is dict and set(replay) == {"primary", "secondary"})
    progress(state, "publication")
    r.write_json(OUT / "results.json", result_value(protocol, pin, replay))
    r.write_json(OUT / "manifest.json", manifest(pin))
    authenticate(pin, replay=True)


def audit(pin, state):
    protocol, replay = authenticate(pin, replay=True)
    r.absent(AUDIT)
    AUDIT.mkdir(); state["owned"] = AUDIT
    progress(state, "audit")
    audit_sources_and_inference(protocol)
    replay = replay_private(protocol)
    require(r.equal(json.loads((OUT / "results.json").read_text()), result_value(protocol, pin, replay)))
    progress(state, "publication")
    r.write_json(AUDIT / "audit.json", audit_value(pin))
    r.write_json(AUDIT / "manifest.json", {"protocol_sha256": pin, "audit_sha256": r.sha(AUDIT / "audit.json")})
    authenticate_audit(pin, r.sha(AUDIT / "audit.json"))


def authenticate_audit(pin, audit_pin):
    authenticate(pin, replay=True)
    r.inventory(AUDIT, ("audit.json", "manifest.json", "progress.json"))
    validate_progress(json.loads((AUDIT / 'progress.json').read_text()), terminal=True)
    require(r.sha(AUDIT / "audit.json") == audit_pin)
    require(r.equal(json.loads((AUDIT / 'audit.json').read_text()), audit_value(pin)))
    require(r.equal(json.loads((AUDIT / "manifest.json").read_text()), {"protocol_sha256": pin, "audit_sha256": audit_pin}))


def audit_value(pin):
    return {'schema': SCHEMA, 'status': 'authenticated', 'protocol_sha256': pin,
            'terminal_manifest_sha256': r.sha(OUT / 'manifest.json'),
            'training_repeated': False, 'all_admitted_raw_bytes_reauthenticated': True,
            'partial_pixel_and_inference_replay': True, 'replay_images_per_arm': 32,
            'all_inference_replayed': False, 'full_readout_and_aggregate_replay': True,
            'patient_level_output_emitted': False}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "verify", "audit"))
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--admission-audit-sha256")
    parser.add_argument("--audit-sha256")
    args = parser.parse_args(argv)
    state, answer = {"phase": "authentication", "owned": None}, {"status": "failed", "patient_level_output_emitted": False}
    with r.quiet():
        try:
            with LOCK.open("a") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                pin = args.protocol_sha256
                if args.action == "prepare": pin = prepare(args.admission_audit_sha256, state)
                elif args.action == "run": run(pin, state)
                elif args.action == "audit": audit(pin, state)
                elif args.audit_sha256: authenticate_audit(pin, args.audit_sha256)
                else: authenticate(pin, replay=True)
                answer = {"status": "complete", "action": args.action, "protocol_sha256": pin, "patient_level_output_emitted": False}
                if args.action != "prepare": answer["results_sha256"] = r.sha(OUT / "results.json")
                if args.action == "audit" or args.audit_sha256: answer["audit_sha256"] = r.sha(AUDIT / "audit.json")
        except Exception:
            answer["phase"] = state["phase"]
            if state["owned"] is not None:
                try: r.write_json(state["owned"] / "failure.json", answer)
                except Exception: pass
    print(json.dumps(answer, sort_keys=True))
    return int(answer["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
