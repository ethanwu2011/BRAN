"""Locked, local and FD-quiet H4 adaptation; H3 artifacts are read-only."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from bran_clinical_dictionary_binding_v1 import _quiet
import run_bran_hirid_h3 as h3
import bran_hirid_source_h3 as source

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'BRAN_HIRID_H4_ADAPTATION_PROTOCOL_2026-09-23.md'
H3_PROTOCOL = 'cab732d6466dfa041cce707d8318437c724d01828ed77f9b7fd595a2cdaabbbd'
H3_AGGREGATE = '352a39236086507a39797f5f0759428ee512f5b88c180af3f97f93c59832d52c'
H3_REPORT = '1377723c9e801f24937ed15dbf82c2495983b3c9b09668d4393d853cca6c5eff'
FIELDS = ('potassium', 'sodium', 'chloride', 'creatinine', 'bilirubin_total', 'albumin', 'glucose')
CODE = ('run_bran_hirid_h4.py', 'bran_hirid_state_h4.py', 'bran_hirid_adaptation_h4.py',
        'report_bran_hirid_h4.py', 'run_bran_hirid_h4_attempt1.sh')
PHASES = ('authentication', 'source_authentication', 'source_selection', 'manifest_preflight',
          'state_inference', 'checkpoint_replay', 'readout_fitting_and_bootstrap',
          'aggregate_replay', 'post_authentication', 'completed', 'unsupported', 'failed')
ERROR = 'bran_hirid_h4_execution_failed'


def require(ok):
    if not ok:
        raise ValueError(ERROR) from None


def digest(path):
    return source._hash_file(path)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def paths(attempt):
    require(type(attempt) is int and 1 <= attempt <= 99)
    return ROOT / f'BRAN_HIRID_H4_ATTEMPT{attempt}'


def role_for_identifier(identifier):
    # Local-only, outcome-blind assignment. Never export individual IDs/digests.
    require(type(identifier) in (str, int) and bool(str(identifier)))
    block = hashlib.sha256(('bran-hirid-h4-v1:' + str(identifier)).encode()).digest()
    bucket = int.from_bytes(block[:8], 'big') % 10
    return 0 if bucket < 6 else (1 if bucket < 8 else 2)


def authenticate():
    import sklearn
    import scipy
    parent = h3.paths(2)
    require(digest(parent / 'protocol.json') == H3_PROTOCOL
            and digest(parent / 'aggregate.json') == H3_AGGREGATE
            and not (parent / 'failed.json').exists())
    protocol = h3.read(parent / 'protocol.json')
    done = h3.read(parent / 'completed.json')
    require(done['status'] == 'completed' and done['protocol_sha256'] == H3_PROTOCOL
            and done['aggregate_sha256'] == H3_AGGREGATE
            and done['patient_level_output_emitted'] is False)
    h3.validate_aggregate(h3.read(parent / 'aggregate.json'))
    report = ROOT / 'BRAN_HIRID_H3_REPORT_ATTEMPT2_2026-09-23'
    require(digest(report / 'manifest.json') == H3_REPORT)
    receipt = h3.read(report / 'manifest.json')['audit']
    require(receipt['independent_source_inference_aggregate_replay'] is True
            and receipt['protocol_sha256'] == H3_PROTOCOL and receipt['aggregate_sha256'] == H3_AGGREGATE
            and receipt['patient_level_output_emitted'] is False)
    model = h3.authenticate_model()
    require(model == protocol['launch']['model'] and h3.approval() == protocol['launch']['approval_sha256'])
    binding = {'parent_protocol_sha256': H3_PROTOCOL, 'parent_aggregate_sha256': H3_AGGREGATE,
        'parent_report_manifest_sha256': H3_REPORT, 'model': model,
        'approval_sha256': h3.approval(), 'plan_sha256': digest(PLAN),
        'code_sha256': {name: digest(ROOT / name) for name in CODE},
        'environment': {'sklearn': sklearn.__version__, 'scipy': scipy.__version__},
        'patient_level_output_emitted': False}
    return binding, protocol['source']


def load_source(receipt, binding, callback):
    from bran_hirid_raw_reader_v1 import read_admissions
    callback('source_authentication')
    source.recheck_source(receipt)
    callback('source_selection')
    selections = h3.select_source(receipt, binding['approval_sha256'], callback)
    admissions = read_admissions(source.SOURCE / 'ref/general_table.csv', receipt['general_table_sha256'])
    # Pinned h3.select_source returns selections in sorted(admissions) order,
    # including no-observation admissions. This is not a partition-order join.
    # Both reads authenticate the identical general-table bytes.
    identifiers = sorted(admissions)
    require(len(identifiers) == len(selections))
    roles = np.asarray([role_for_identifier(key) for key in identifiers], dtype=np.int8)
    # One whole-population binding only, not patient-level hashes or assignments.
    split_pin = hashlib.sha256(canonical([(str(key), int(role)) for key, role in zip(identifiers, roles)])).hexdigest()
    return selections, roles, split_pin


def extract_private(selections, binding):
    import torch
    import run_bran_context_preservation_v5 as v5
    from bran_hirid_v5_input_adapter_v1 import prepare_inputs
    from bran_hirid_state_h4 import extract
    from bran_knhanes_input_kernel_v1 import CANONICAL_INDEX
    from bran_hirid_hb_evaluation_h2 import _selection_statuses
    torch.set_num_threads(2)
    model_binding = binding['model']
    model, transform = v5.oldfit.load_checkpoint(ROOT / model_binding['checkpoint_path'],
        model_binding['checkpoint_sha256'], model_binding['binding'])
    batch = prepare_inputs(selections)
    state = extract(batch, model, transform, model_binding['binding']['transform_sha256'])
    statuses, truth = _selection_statuses(selections)
    require(np.array_equal(state.available, statuses == 'ready'))
    columns = [CANONICAL_INDEX[name] for name in FIELDS]
    raw_mask = batch.clinical_mask[:, columns]
    require(np.array_equal(raw_mask.any(axis=1), statuses == 'ready'))
    raw_values = np.where(raw_mask, batch.clinical_values[:, columns], np.nan)
    return state.states, raw_values, raw_mask, state.native_hb, truth, statuses


def calculate(selections, roles, binding, callback):
    from bran_hirid_adaptation_h4 import evaluate, validate_result
    callback('state_inference')
    arrays = extract_private(selections, binding)
    callback('checkpoint_replay')
    other = extract_private(selections, binding)
    for a, b in zip(arrays, other):
        require(np.array_equal(a, b, equal_nan=True) if a.dtype.kind == 'f' else np.array_equal(a, b))
    callback('readout_fitting_and_bootstrap')
    result = evaluate(*arrays, roles)
    validate_result(result)
    callback('aggregate_replay')
    require(result == evaluate(*other, roles))
    return result


def progress(out, state, phase, n=0, total=0):
    require(phase in PHASES and type(n) is int and type(total) is int and 0 <= n <= total)
    state['phase'] = phase
    value = {'phase': phase, 'partitions_completed': n, 'partitions_total': total,
        'elapsed_seconds': round(time.monotonic() - state['start'], 1), 'patient_level_output_emitted': False}
    temporary = out / 'progress.next.json'
    h3.write(temporary, value)
    os.replace(temporary, out / 'progress.json')


def validate_aggregate(value):
    from bran_hirid_adaptation_h4 import validate_result
    require(type(value) is dict and set(value) == {'schema', 'protocol_sha256', 'report',
        'checkpoint_replay_equal', 'aggregate_replay_equal', 'encoder_updated', 'patient_level_output_emitted'})
    require(value['schema'] == 'bran-hirid-h4-aggregate-v1'
        and value['checkpoint_replay_equal'] is True and value['aggregate_replay_equal'] is True
        and value['encoder_updated'] is False and value['patient_level_output_emitted'] is False)
    pin = value['protocol_sha256']
    require(type(pin) is str and len(pin) == 64 and set(pin) <= set('0123456789abcdef'))
    validate_result(value['report'])


def run(attempt):
    out = paths(attempt)
    require(not out.exists() and not out.is_symlink())
    state = {'phase': 'authentication', 'start': time.monotonic()}
    with h3.LOCK.open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'not_started_shared_lock_busy', 'patient_level_output_emitted': False}
        out.mkdir(mode=0o700)
        try:
            with _quiet():
                callback = lambda phase, n=0, total=0: progress(out, state, phase, n, total)
                callback('authentication')
                binding, receipt = authenticate()
                h3.write(out / 'launch.json', binding)
                selections, roles, split_pin = load_source(receipt, binding, callback)
                protocol = {'schema': 'bran-hirid-h4-protocol-v1', 'launch': binding,
                    'source_receipt_sha256': source.content_hash(receipt),
                    'admission_role_binding_sha256': split_pin, 'frozen_before_fitting': True}
                h3.write(out / 'protocol.json', protocol)
                protocol_pin = digest(out / 'protocol.json')
                report = calculate(selections, roles, binding, callback)
                callback('post_authentication')
                require(authenticate() == (binding, receipt))
                source.recheck_source(receipt)
                result = {'schema': 'bran-hirid-h4-aggregate-v1', 'protocol_sha256': protocol_pin,
                    'report': report, 'checkpoint_replay_equal': True, 'aggregate_replay_equal': True,
                    'encoder_updated': False, 'patient_level_output_emitted': False}
                validate_aggregate(result)
                require(digest(out / 'protocol.json') == protocol_pin)
                h3.write(out / 'aggregate.json', result)
                status = 'unsupported' if report['status'] == 'unsupported' else 'completed'
                callback(status)
                h3.write(out / (status + '.json'), {'status': status, 'protocol_sha256': protocol_pin,
                    'aggregate_sha256': digest(out / 'aggregate.json'),
                    'elapsed_seconds': round(time.monotonic()-state['start'], 1),
                    'patient_level_output_emitted': False})
            return {'status': status + '_pending_independent_audit', 'patient_level_output_emitted': False}
        except BaseException:
            h3.write(out / 'failed.json', {'status': 'technical_failure', 'phase': state['phase'],
                'error_code': ERROR, 'elapsed_seconds': round(time.monotonic()-state['start'], 1),
                'patient_level_output_emitted': False})
            return {'status': 'technical_failure', 'phase': state['phase'], 'patient_level_output_emitted': False}


def _audit(attempt):
    with h3.LOCK.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with _quiet():
            out = paths(attempt)
            terminal = [name for name in ('completed.json', 'unsupported.json', 'failed.json') if (out/name).exists()]
            require(len(terminal) == 1 and terminal[0] != 'failed.json')
            require({p.name for p in out.iterdir()} == {'launch.json','protocol.json','aggregate.json','progress.json',terminal[0]})
            require(all(p.is_file() and not p.is_symlink() for p in out.iterdir()))
            done, protocol, result = (h3.read(out/name) for name in (terminal[0], 'protocol.json', 'aggregate.json'))
            pin = digest(out/'protocol.json')
            require(done['protocol_sha256'] == pin == result['protocol_sha256']
                and done['aggregate_sha256'] == digest(out/'aggregate.json')
                and done['patient_level_output_emitted'] is False
                and done['status'] + '.json' == terminal[0]
                and done['status'] == result['report']['status']
                and protocol['frozen_before_fitting'] is True)
            validate_aggregate(result)
            binding, receipt = authenticate()
            require(binding == protocol['launch'] == h3.read(out/'launch.json')
                and source.content_hash(receipt) == protocol['source_receipt_sha256'])
            selections, roles, split_pin = load_source(receipt, binding, lambda *args: None)
            require(split_pin == protocol['admission_role_binding_sha256'])
            require(calculate(selections, roles, binding, lambda *args: None) == result['report'])
            require(authenticate() == (binding, receipt))
            return {'status': 'aggregate_terminal_authenticated', 'protocol_sha256': pin,
                'aggregate_sha256': done['aggregate_sha256'], 'independent_source_fit_calibration_evaluation_replay': True,
                'patient_level_output_emitted': False}


def audit(attempt):
    try:
        return _audit(attempt)
    except BaseException:
        raise ValueError(ERROR) from None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    try:
        result = run(args.attempt)
    except BaseException:
        result = {'status': 'execution_not_started_or_failed', 'patient_level_output_emitted': False}
    print(json.dumps(result))
    return 0 if result['status'] in ('completed_pending_independent_audit','unsupported_pending_independent_audit') else 1


if __name__ == '__main__':
    raise SystemExit(main())
