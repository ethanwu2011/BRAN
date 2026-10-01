"""One locked, FD-quiet HiRID direct-transport run with exclusive terminals."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import time

from bran_clinical_dictionary_binding_v1 import _quiet
import bran_hirid_source_h3 as source

ROOT = Path(__file__).resolve().parent
LOCK = Path('/private/tmp/bran_retinal_extraction_v1.lock')
APPROVAL = ROOT / 'BRAN_HIRID_APPROVAL_2026-09-23.json'
PLAN = ROOT / 'BRAN_HIRID_H3_EXECUTION_PROTOCOL.md'
BASE = ROOT / 'BRAN_RESEARCH_BASELINE_V5_2026-09-19.json'
CHECKPOINT_PIN = 'faa2f6e0dd88ef4e7d827eb30f90fa0ca4b1b03cf43789fef229f672ddb9b8ba'
ERROR = 'bran_hirid_h3_execution_failed'
PHASES = ('authentication', 'reference_authentication', 'archive_authentication',
          'partition_bytes_authentication', 'source_selection', 'manifest_preflight',
          'native_inference', 'checkpoint_replay', 'aggregate_bootstrap',
          'post_authentication', 'completed')
CODE = ('run_bran_hirid_h3.py', 'run_bran_hirid_h3_attempt1.sh',
        'run_bran_hirid_h3_attempt2.sh', 'report_bran_hirid_h3.py',
        'BRAN_HIRID_H3_TECHNICAL_CORRECTION_2026-09-23.md',
        'bran_hirid_source_h3.py', 'bran_hirid_partition_reader_h3.py',
        'bran_hirid_raw_reader_v1.py', 'bran_hirid_v5_observation_kernel.py',
        'bran_hirid_manifest_preflight_h1.py', 'bran_hirid_v5_input_adapter_v1.py',
        'bran_hirid_v5_inference_v1.py', 'bran_hirid_hb_evaluation_h2.py',
        'bran_clinical_dictionary_binding_v1.py', 'bran_knhanes_input_kernel_v1.py',
        'bran_clinical_semantics_v1.py', 'run_bran_v5_cbc_uncertainty.py',
        'audit_bran_context_v5.py', 'BRAN_HIRID_H2_EVALUATION_CONTRACT.md')


def require(ok):
    if not ok:
        raise ValueError(ERROR) from None


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def paths(attempt):
    require(type(attempt) is int and 1 <= attempt <= 99)
    return ROOT / f'BRAN_HIRID_H3_ATTEMPT{attempt}'


def approval():
    source.regular(APPROVAL)
    record = read(APPROVAL)
    require(record['schema'] == 'bran-hirid-study-approval-attestation-v1'
            and record['source'] == 'HiRID' and record['version'] == '1.1.1'
            and record['status'] == 'approved'
            and record['hosted_patient_level_output_permitted'] is False)
    return source._hash_file(APPROVAL)


def authenticate_model():
    import numpy as np
    import torch
    import run_bran_v5_cbc_uncertainty as v5
    base, fit, _, components = v5.small_authentication()
    selected = base['fold_checkpoints'][0]
    require(selected['fold'] == 0 and selected['checkpoint_sha256'] == CHECKPOINT_PIN)
    roles = fit['source_binding']['source_roles']
    require(fit['source_binding']['protected_sources_used'] is False
            and roles['hirid']['disposition'] != 'training')
    code = {name: source._hash_file(ROOT / name)
            for name in sorted(set(CODE) | set(fit['code_sha256']))}
    require(all(code[name] == pin for name, pin in fit['code_sha256'].items()))
    return {'baseline_sha256': source._hash_file(BASE),
            'checkpoint_path': selected['checkpoint_path'], 'checkpoint_sha256': CHECKPOINT_PIN,
            'component_sha256': selected['receipt_sha256'],
            'binding': components[('M', 0)]['binding'], 'code_sha256': code,
            'hirid_encoder_exposed': False,
            'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                            'torch': torch.__version__}}


def model_inference(selections, model_receipt):
    import numpy as np
    import torch
    import run_bran_context_preservation_v5 as v5
    from bran_hirid_v5_input_adapter_v1 import prepare_inputs
    from bran_hirid_v5_inference_v1 import infer
    from bran_knhanes_input_kernel_v1 import CANONICAL_INDEX
    torch.set_num_threads(2)
    model, transform = v5.oldfit.load_checkpoint(ROOT / model_receipt['checkpoint_path'],
        model_receipt['checkpoint_sha256'], model_receipt['binding'])
    # This is the observed AI-READI training-fold Hb median, not a HiRID estimate.
    median = float(transform.clinical_median[CANONICAL_INDEX['hemoglobin']])
    require(np.isfinite(median) and median > 0)
    result = infer(prepare_inputs(selections), model, transform,
                   model_receipt['binding']['transform_sha256'])
    return result, median


def select_source(receipt, approval_pin, progress=lambda phase, n, total: None):
    from bran_hirid_raw_reader_v1 import read_admissions, COLUMNS
    from bran_hirid_partition_reader_h3 import read_partition
    from bran_hirid_v5_observation_kernel import select_episode
    import bran_hirid_manifest_preflight_h1 as preflight
    require(approval() == approval_pin)
    admissions = read_admissions(source.SOURCE / 'ref/general_table.csv',
                                 receipt['general_table_sha256'])
    require(len(admissions) > 0)
    by_admission, entries = {}, []
    total = len(receipt['partitions'])
    for number, (name, pin) in enumerate(receipt['partitions'].items(), 1):
        selected = read_partition(source.SOURCE / source.PARTITIONS / name, pin, admissions)
        require(not set(selected.admission_ids).intersection(by_admission))
        by_admission.update(zip(selected.admission_ids, selected.selections))
        entries.append((pin, selected.admission_ids))
        if number % 10 == 0 or number == total:
            progress('source_selection', number, total)
    progress('manifest_preflight', 0, 0)
    manifest = preflight.qualify_manifest(source='HiRID', version='1.1.1',
        source_sha256=source.content_hash(receipt),
        reference_sha256=receipt['reference_archive_sha256'],
        schema_sha256=receipt['schema_sha256'], raw_header=COLUMNS,
        expected_partition_hashes=tuple(receipt['partitions'].values()),
        raw_partition_inventory_sha256=receipt['partition_inventory_sha256'],
        partition_entries=tuple(entries), repeat_person_linkage='absent')
    granted = preflight.AuthenticatedStudyApproval('HiRID', '1.1.1', 'approved', approval_pin)
    preflight.authorize_reader(manifest, granted, lambda authenticated_manifest: None)
    require(set(by_admission).issubset(admissions))
    # Preserve general-table admissions that have no observations at all.
    return tuple(by_admission.get(key, select_episode(())) for key in sorted(admissions))


def progress(out, state, phase, number=0, total=0):
    require(phase in PHASES and type(number) is int and type(total) is int
            and 0 <= number <= total)
    state['phase'] = phase
    value = {'phase': phase, 'partitions_completed': number, 'partitions_total': total,
             'elapsed_seconds': round(time.monotonic() - state['start'], 1),
             'patient_level_output_emitted': False}
    temporary = out / 'progress.next.json'
    write(temporary, value)
    os.replace(temporary, out / 'progress.json')


def validate_aggregate(value):
    import bran_hirid_hb_evaluation_h2 as metrics
    require(type(value) is dict and set(value) == {'schema', 'report',
        'checkpoint_replay_equal', 'aggregate_replay_equal', 'protocol_sha256',
        'source_manifest_authenticated', 'encoder_updated', 'external_model_selection',
        'patient_level_output_emitted', 'clinical_use'})
    require(value['schema'] == 'bran-hirid-h3-aggregate-v1')
    for key in ('checkpoint_replay_equal', 'aggregate_replay_equal', 'source_manifest_authenticated'):
        require(value[key] is True)
    for key in ('encoder_updated', 'external_model_selection', 'patient_level_output_emitted', 'clinical_use'):
        require(value[key] is False)
    require(type(value['protocol_sha256']) is str and len(value['protocol_sha256']) == 64
            and set(value['protocol_sha256']) <= set('0123456789abcdef'))
    metrics.validate_result(value['report'])


def evaluate(selections, bindings, callback):
    import numpy as np
    import bran_hirid_hb_evaluation_h2 as metrics
    callback('native_inference', 0, 0)
    predictions, median = model_inference(selections, bindings)
    callback('checkpoint_replay', 0, 0)
    replay, replay_median = model_inference(selections, bindings)
    require(median == replay_median and np.array_equal(predictions.available, replay.available)
            and np.array_equal(predictions.native_hemoglobin, replay.native_hemoglobin, equal_nan=True))
    callback('aggregate_bootstrap', 0, 0)
    result = metrics.summarize(selections, predictions, median)
    require(result == metrics.summarize(selections, replay, replay_median))
    metrics.validate_result(result)
    return result


def run(attempt):
    out = paths(attempt)
    require(not out.exists() and not out.is_symlink())
    state = {'phase': 'authentication', 'start': time.monotonic()}
    with LOCK.open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'not_started_shared_lock_busy', 'patient_level_output_emitted': False}
        out.mkdir(mode=0o700)
        try:
            with _quiet():
                callback = lambda phase, n=0, total=0: progress(out, state, phase, n, total)
                callback('authentication')
                approval_pin = approval()
                require(source._hash_file(ROOT / 'BRAN_PROTECTED_METADATA_V1/metadata.json')
                        == source.METADATA_PIN)
                model = authenticate_model()
                launch = {'schema': 'bran-hirid-h3-launch-v1', 'approval_sha256': approval_pin,
                          'plan_sha256': source._hash_file(PLAN), 'model': model,
                          'patient_level_output_emitted': False}
                write(out / 'launch.json', launch)
                receipt = source.authenticate_source(progress=callback)
                protocol = {'schema': 'bran-hirid-h3-protocol-v1', 'launch': launch,
                            'source': receipt, 'frozen_before_inference': True}
                write(out / 'protocol.json', protocol)
                pin = source._hash_file(out / 'protocol.json')
                callback('source_selection')
                selections = select_source(receipt, approval_pin, callback)
                report = evaluate(selections, model, callback)
                callback('post_authentication')
                require(authenticate_model() == model and approval() == approval_pin
                        and source._hash_file(PLAN) == launch['plan_sha256']
                        and source._hash_file(out / 'protocol.json') == pin)
                source.recheck_source(receipt)
                result = {'schema': 'bran-hirid-h3-aggregate-v1', 'report': report,
                          'checkpoint_replay_equal': True, 'aggregate_replay_equal': True,
                          'source_manifest_authenticated': True, 'protocol_sha256': pin,
                          'encoder_updated': False, 'external_model_selection': False,
                          'patient_level_output_emitted': False, 'clinical_use': False}
                validate_aggregate(result)
                write(out / 'aggregate.json', result)
                callback('completed')
                write(out / 'completed.json', {'status': 'completed', 'protocol_sha256': pin,
                    'aggregate_sha256': source._hash_file(out / 'aggregate.json'),
                    'elapsed_seconds': round(time.monotonic() - state['start'], 1),
                    'patient_level_output_emitted': False})
            return {'status': 'completed_pending_independent_audit', 'patient_level_output_emitted': False}
        except BaseException:
            # Never interpolate exception strings, patient values or native diagnostics.
            if not (out / 'completed.json').exists():
                write(out / 'failed.json', {'status': 'technical_failure', 'phase': state['phase'],
                    'error_code': ERROR, 'elapsed_seconds': round(time.monotonic() - state['start'], 1),
                    'patient_level_output_emitted': False})
            return {'status': 'technical_failure', 'phase': state['phase'],
                    'patient_level_output_emitted': False}


def audit(attempt, replay=True):
    with LOCK.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with _quiet():
            out = paths(attempt)
            require(out.is_dir() and not out.is_symlink()
                    and (out / 'completed.json').is_file() and not (out / 'failed.json').exists())
            require({p.name for p in out.iterdir()} == {'launch.json', 'protocol.json',
                                                       'aggregate.json', 'completed.json', 'progress.json'})
            done, protocol, result = (read(out / n) for n in ('completed.json', 'protocol.json', 'aggregate.json'))
            pin = source._hash_file(out / 'protocol.json')
            require(done['status'] == 'completed' and done['protocol_sha256'] == pin
                    and done['aggregate_sha256'] == source._hash_file(out / 'aggregate.json')
                    and result['protocol_sha256'] == pin and done['patient_level_output_emitted'] is False
                    and read(out / 'launch.json') == protocol['launch']
                    and protocol['frozen_before_inference'] is True)
            validate_aggregate(result)
            launch = protocol['launch']
            require(approval() == launch['approval_sha256']
                    and source._hash_file(PLAN) == launch['plan_sha256']
                    and authenticate_model() == launch['model'])
            source.recheck_source(protocol['source'])
            if replay:
                selections = select_source(protocol['source'], launch['approval_sha256'])
                report = evaluate(selections, launch['model'], lambda *args: None)
                require(report == result['report'])
            return {'status': 'aggregate_terminal_authenticated', 'protocol_sha256': pin,
                    'aggregate_sha256': done['aggregate_sha256'],
                    'independent_source_inference_aggregate_replay': replay,
                    'patient_level_output_emitted': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    try:
        result = audit(args.attempt) if args.audit else run(args.attempt)
    except BaseException:
        result = {'status': 'not_started_or_audit_failed', 'patient_level_output_emitted': False}
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] in ('completed_pending_independent_audit',
                                     'aggregate_terminal_authenticated') else 1


if __name__ == '__main__':
    raise SystemExit(main())
