"""Fresh V3 authenticated-retinal no-fit bridge; no extraction or training."""
import argparse
import fcntl
import json
from pathlib import Path

import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import exclusive_json, sha
from run_bran_cbc_reference_preflight_v1 import publish
import run_bran_retinal_input_bridge_v1 as legacy
import run_bran_retinal_extraction_v3 as extraction
import bran_authenticated_retinal_input_v3 as inputs
from bran_clinical_semantics_v1 import CBC_FIELDS

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'BRAN_RETINAL_INPUT_BRIDGE_PROTOCOL_V3.json'
OUT = ROOT / 'BRAN_RETINAL_INPUT_BRIDGE_V3'
AUDIT = ROOT / 'BRAN_RETINAL_INPUT_BRIDGE_AUDIT_V3'
LOCK = Path('/private/tmp/bran_retinal_extraction_v1.lock')
PARAMETERS = dict(legacy.PARAMETERS)
FLAGS = dict(legacy.FLAGS)
origin, kernel, metrics, require = legacy.origin, legacy.kernel, legacy.metrics, legacy.require
FILES = (
    'run_bran_retinal_input_bridge_v3.py', 'test_run_bran_retinal_input_bridge_v3.py',
    'bran_authenticated_retinal_input_v3.py', 'test_bran_authenticated_retinal_input_v3.py',
    'bran_authenticated_retinal_input_v2.py', 'test_bran_authenticated_retinal_input_v2.py',
    'bran_authenticated_retinal_input_v1.py', 'run_bran_retinal_input_bridge_v1.py',
    'run_bran_retinal_input_bridge_v2.py', 'run_bran_retinal_extraction_v3.py',
    'run_bran_retinal_extraction_v2.py', 'run_bran_retinal_extraction_v1.py',
    'run_bran_cbc_reference_preflight_v1.py', 'run_bran_source_linkage_audit_v1.py',
    'bran_clinical_dictionary_binding_v1.py',
    'BRAN_RETINAL_INPUT_BRIDGE_DESIGN_V3.md',
)
PHASES = ('protocol', 'source_authentication', 'bridge_comparison', 'audit', 'publishing')


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _private_file(path):
    return path.is_file() and not path.is_symlink() and path.stat().st_mode & 0o777 == 0o600


def _authenticate_retinal(protocol_pin, audit_pin):
    require(_digest(protocol_pin) and _digest(audit_pin))
    require(sha(extraction.PROTOCOL) == protocol_pin)
    require(not any((directory / 'failure.json').exists() for directory in (extraction.OUT, extraction.AUDIT, extraction.PRIVATE)))
    ep = json.loads(extraction.PROTOCOL.read_text()); extraction.validate_protocol(ep)
    aggregate_path = extraction.OUT / 'aggregate.json'; audit_path = extraction.AUDIT / 'audit.json'
    aggregate = json.loads(aggregate_path.read_text()); extraction.validate_result(aggregate, ep, protocol_pin)
    require(sha(audit_path) == audit_pin)
    require(json.loads((extraction.OUT / 'aggregate.manifest.json').read_text()) ==
            {'protocol_sha256': protocol_pin, 'artifact_sha256': sha(aggregate_path)})
    require(json.loads((extraction.AUDIT / 'audit.manifest.json').read_text()) ==
            {'protocol_sha256': protocol_pin, 'artifact_sha256': audit_pin})
    audit = json.loads(audit_path.read_text())
    require(audit == {
        'schema': 'bran-retinal-extraction-audit-v3', 'status': 'authenticated',
        'protocol_sha256': protocol_pin, 'aggregate_sha256': sha(aggregate_path),
        'inventory_file_sha256': aggregate['inventory_file_sha256'], 'output_sha256': aggregate['output_sha256'],
        'selection': ep['selection'], 'preflight_audit_sha256': ep['preflight_audit_sha256'],
        'gpu_preflight_passed': True, 'independent_generator_replay_passed': True,
        'patient_level_output_emitted': False,
    })
    require(aggregate.get('gpu_preflight_passed') is True)
    for key, expected in (('gpu_preflight_passed', True), ('independent_generator_replay_passed', True),
                          ('patient_level_output_emitted', False)):
        require(type(audit.get(key)) is bool and audit[key] is expected)
    require(extraction.PRIVATE.is_dir() and not extraction.PRIVATE.is_symlink()
            and extraction.PRIVATE.stat().st_mode & 0o777 == 0o700)
    for name, digest in (('features.npy', audit['output_sha256']), ('inventory.json', audit['inventory_file_sha256'])):
        path = extraction.PRIVATE / name
        require(_private_file(path) and sha(path) == digest)
    require(not any((directory / 'failure.json').exists() for directory in (extraction.OUT, extraction.AUDIT, extraction.PRIVATE)))
    return ep, aggregate, audit


def prepare(retinal_protocol_pin, retinal_audit_pin):
    ep, _, audit = _authenticate_retinal(retinal_protocol_pin, retinal_audit_pin)
    base = origin.prepare()
    closure = {**base['code_sha256'], **ep['code_sha256']}
    closure.update({name: sha(ROOT / name) for name in set(legacy.FILES) | set(FILES)})
    return {
        'schema': 'bran-retinal-input-bridge-protocol-v3', 'status': 'frozen_before_execution',
        'parameters': PARAMETERS, 'native_source': base['native_source'],
        'native_aggregate_sha256': base['native_aggregate_sha256'], 'native_audit_sha256': base['native_audit_sha256'],
        'retinal_protocol_sha256': retinal_protocol_pin, 'retinal_audit_sha256': retinal_audit_pin,
        'retinal_aggregate_sha256': audit['aggregate_sha256'], 'retinal_output_sha256': audit['output_sha256'],
        'retinal_selection': audit['selection'],
        'source_policy_sha256': ep['origin_protocol']['external_sha256']['source_policy'],
        'code_sha256': closure, 'runtime': base['runtime'],
    }


def validate_protocol(p):
    require(type(p) is dict and p == prepare(p['retinal_protocol_sha256'], p['retinal_audit_sha256']))


def validate_result(a, p):
    require(type(a) is dict and a.get('schema') == 'bran-retinal-input-bridge-aggregate-v3')
    legacy.validate_result({**a, 'schema': 'bran-retinal-input-bridge-aggregate-v1'}, p)


def compute(p, notify=None):
    import torch
    torch.set_num_threads(2); source = origin.native.source
    ctx, folds, c0, cm0, eligible, r0, rm, ages, names = source.io.load_context()
    ids = list(map(str, ctx['raw_cohort'].patient_ids)); require(len(ids) == 1928)
    rnew, rmnew = inputs.load_pooled_features(protocol_pin=p['retinal_protocol_sha256'], audit_pin=p['retinal_audit_sha256'],
        patient_ids=ids, folds=folds, source_policy_sha256=p['source_policy_sha256'])
    require(np.array_equal(rm, rmnew)); endpoints = p['native_source']['source']['endpoint_names']
    screen = {arm: np.full((len(folds), 26), np.nan) for arm in metrics.ARMS}
    blood = {value: np.full((len(folds), 9), np.nan) for value in kernel.VERSIONS}
    support = np.zeros((len(folds), 9), bool); slots = tuple(names.index(field) for field in CBC_FIELDS)
    for fold in range(5):
        if notify: notify('fixed_checkpoint_bridge', fold)
        fit, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        _, identity = source.base._inner_context(ctx, fit, fold)
        require(identity == p['native_source']['source']['authentication']['inner_fold_sha256'][fold])
        transform = source.base.FoldTransform(c0, cm0, eligible, r0, rm, ages, fit)
        model = origin.load_initial(fold, transform, p)
        screens, cbc, available = kernel.infer(model, transform, c0[test], cm0[test], eligible[test], r0[test],
            rnew[test], rm[test], rmnew[test], ages[test], names, batch_size=256)
        for value in kernel.VERSIONS:
            for route in kernel.ROUTES: screen[value + '_' + route][test] = screens[value][route]
            blood[value][test] = cbc[value]
        support[test] = available
    if notify: notify('reference_replay')
    labels = np.column_stack([ctx['labels_by_source'][name] for name in endpoints])
    observed = np.column_stack([ctx['observed_by_source'][name] for name in endpoints]).astype(bool)
    baseline = json.loads((origin.native.OUT / 'aggregate.json').read_text())['results']['endpoints']
    common = np.logical_and.reduce([np.isfinite(screen['historical_' + route]).all(1) for route in kernel.ROUTES])
    for j, name in enumerate(endpoints):
        for route in kernel.ROUTES:
            require(abs(source.base.fold_weighted_auc(labels[:, j], screen['historical_' + route][:, j], observed[:, j] & common, folds)
                    - baseline[name]['arms']['native_' + route]['auroc']) <= PARAMETERS['historical_screen_replay_atol'])
    bobs, truth = (cm0 & eligible)[:, slots] & support, c0[:, slots]
    reference = json.loads((source.OUT / 'aggregate.json').read_text())['retained_cbc_head_whole_panel']
    for j, field in enumerate(CBC_FIELDS):
        error = blood['historical'][bobs[:, j], j] - truth[bobs[:, j], j]
        require(np.isclose(np.abs(error).mean(), reference[field]['mae'], atol=PARAMETERS['historical_cbc_replay_atol'], rtol=PARAMETERS['historical_cbc_replay_rtol']))
        require(np.isclose(np.square(error).mean(), reference[field]['mse'], atol=PARAMETERS['historical_cbc_replay_atol'], rtol=PARAMETERS['historical_cbc_replay_rtol']))
    if notify: notify('paired_aggregate')
    counts = source.ev.paired_counts(folds, draws=1000, seed=94801)
    results = metrics.summarize(screen, labels, observed, folds, endpoints, counts, truth, blood, bobs)
    equivalence = {'pooled_input': bool(np.allclose(r0[rm], rnew[rm], **PARAMETERS['equivalence'])),
        'screening': bool(all(np.allclose(screen['historical_' + route], screen['authenticated_' + route], equal_nan=True, **PARAMETERS['equivalence']) for route in kernel.ROUTES)),
        'whole_cbc': bool(np.allclose(blood['historical'], blood['authenticated'], equal_nan=True, **PARAMETERS['equivalence']))}
    equivalence['replacement_review_eligible'] = all(equivalence.values())
    value = {'schema': 'bran-retinal-input-bridge-aggregate-v3', 'status': 'completed', 'results': results,
        'equivalence': equivalence, 'fold_authentication': p['native_source']['source']['authentication'],
        'retinal_selection': p['retinal_selection'], **FLAGS}
    validate_result(value, p); return value


def audit(p, pin):
    require(not (OUT / 'failure.json').exists())
    aggregate_path, manifest_path = OUT / 'aggregate.json', OUT / 'aggregate.manifest.json'
    raw, manifest_raw = aggregate_path.read_bytes(), manifest_path.read_bytes()
    aggregate = json.loads(raw); validate_result(aggregate, p)
    require(json.loads(manifest_raw) == {'protocol_sha256': pin, 'artifact_sha256': sha(aggregate_path)})
    require(compute(p) == aggregate and (OUT / 'aggregate.json').read_bytes() == raw)
    validate_protocol(p)
    require(not (OUT / 'failure.json').exists() and aggregate_path.read_bytes() == raw and manifest_path.read_bytes() == manifest_raw)
    return {'schema': 'bran-retinal-input-bridge-audit-v3', 'status': 'authenticated', 'protocol_sha256': pin,
        'aggregate_sha256': sha(OUT / 'aggregate.json'), 'retinal_audit_sha256': p['retinal_audit_sha256'],
        'fold_authentication': aggregate['fold_authentication'], 'all_aggregates_replayed': True,
        'patient_level_output_emitted': False, 'model_promoted': False}


def main(argv=None):
    parser = argparse.ArgumentParser(); operations = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'run', 'audit'): operations.add_argument('--' + name, action='store_true')
    parser.add_argument('--protocol-sha256'); parser.add_argument('--retinal-protocol-sha256'); parser.add_argument('--retinal-audit-sha256')
    args = parser.parse_args(argv); dest = AUDIT if args.audit else OUT; owned = ok = False; phase = 'protocol'
    with _quiet():
        try:
            with open(LOCK, 'a') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.prepare:
                    require(not any(path.exists() for path in (PROTOCOL, OUT, AUDIT)))
                    value = prepare(args.retinal_protocol_sha256, args.retinal_audit_sha256)
                    exclusive_json(PROTOCOL, value)
                else:
                    require(_digest(args.protocol_sha256) and sha(PROTOCOL) == args.protocol_sha256)
                    p = json.loads(PROTOCOL.read_text()); phase = 'source_authentication'; validate_protocol(p)
                    if args.run: require(not AUDIT.exists())
                    require(not (dest / 'failure.json').exists()); dest.mkdir(); owned = True
                    if args.run:
                        phase = 'bridge_comparison'
                        value = compute(p, lambda step, fold=None: origin.native.source.base._atomic_progress(OUT / 'progress.json', step, fold))
                        name = 'aggregate.json'
                    else:
                        phase = 'audit'; value = audit(p, args.protocol_sha256); name = 'audit.json'
                    validate_protocol(p); require(sha(PROTOCOL) == args.protocol_sha256 and json.loads(PROTOCOL.read_text()) == p)
                    phase = 'publishing'; require(not (dest / 'failure.json').exists()); publish(dest, name, value, args.protocol_sha256)
            ok = True
        except Exception:
            try:
                if owned and not any((dest / name).exists() for name in ('aggregate.json', 'audit.json')):
                    require(phase in PHASES)
                    exclusive_json(dest / 'failure.json', {'status': 'execution_failed', 'phase': phase,
                        'patient_level_output_emitted': False, 'model_promoted': False})
            except Exception:
                pass
    print(json.dumps({'status': 'completed' if ok else 'execution_failed', 'patient_level_output_emitted': False}))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
