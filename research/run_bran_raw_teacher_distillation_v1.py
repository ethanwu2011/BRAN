"""One local native-state distillation experiment; only closed terminal outputs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import sha, exclusive_json
import run_bran_native_screening_v1 as native
import run_bran_modality_union_v1 as previous
import bran_distillation_metrics_v1 as metrics
from bran_distillation_metrics_v1 import require

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'BRAN_RAW_TEACHER_DISTILLATION_PROTOCOL_V1.json'
OUT = ROOT / 'BRAN_RAW_TEACHER_DISTILLATION_V1'
AUDIT = ROOT / 'BRAN_RAW_TEACHER_DISTILLATION_AUDIT_V1'
PRIVATE = ROOT / 'private_artifacts/bran_raw_teacher_distillation_v1'
NATIVE_PIN = '2465e7d0d142d4919579aa7ad34849aaa64ce602e3f54a75ad1662f1605b28bc'
NATIVE_AGG = '9e87ddfe86f935da804ed0cb0e3cea845159a0a3a262c2c68f36be4c2fe1f002'
NORMALIZERS = ('clinical_median', 'clinical_iqr', 'retinal_mean', 'retinal_scale', 'age_mean', 'age_scale')
PARAMETERS = {
    'state_width': 192, 'native_heads_retained': True, 'outer_folds': 5, 'inner_folds': 5,
    'steps_per_arm_fold': 1500, 'batch_size': 96, 'seed_base': 96091,
    'learning_rate': .0001, 'weight_decay': .0001, 'gradient_clip_norm': 5.,
    'unchanged_common_losses': {'disease': 1., 'whole_cbc': .5, 'clinical_private_preservation': .1},
    'distillation_weights': {'continued': 0., 'student': 1.}, 'distillation_temperature': 1.,
    'teacher': 'raw clinical and retinal ExtraTrees128 depth6 leaf10 maxfeatures0.7 threads2',
    'teacher_seed_base': 96301, 'teacher_scope': 'crossfitted labels within outer train; fixed outer-training preprocessing',
    'teacher_routes': 'clinical,retinal,equal_probability_average_for_both',
    'teacher_validity': 'observed training endpoint and route physiology present',
    'inference_side_inputs': False, 'raw_replay_seed_base': 92381,
    'screening_scope': 'observed endpoint and every initial native route nonabstaining; all arms identical rows',
    'screening_evaluation': 'retained native heads; no posthoc head fitting for learned states',
    'cbc_evaluation': 'retained native CBC heads; raw standardized Ridge1 references; all9values_and_flags_hidden',
    'bootstrap_draws': 1000, 'bootstrap_seed': 91501, 'minimum_release_support': 20,
    'minimum_valid_draws': 900, 'tail_quantiles': [.1, .9], 'equal_tail_quantiles': 'suppress_low_middle_high',
    'baseline_replay_tolerance': 1e-10, 'automatic_promotion': False,
    'new_subtype_claim': False, 'clinical_use': False, 'official_test_used': False,
    'adaptive_development': True, 'patient_level_output_permitted': False,
}
CODE = ('run_bran_raw_teacher_distillation_v1.py', 'test_run_bran_raw_teacher_distillation_v1.py',
        'bran_raw_teacher_distillation_v1.py', 'test_bran_raw_teacher_distillation_v1.py',
        'bran_distillation_metrics_v1.py', 'test_bran_distillation_metrics_v1.py',
        'BRAN_RAW_TEACHER_DISTILLATION_DESIGN_V1.md', 'bran_matched_screening_kernel_v1.py',
        'bran_screening_joint_kernel_v1.py', 'bran_clinical_preservation_loss_draft_v1.py',
        'run_bran_modality_union_v1.py')


def prepare():
    require(sha(native.PROTOCOL) == NATIVE_PIN, 'native_protocol_changed')
    p = json.loads(native.PROTOCOL.read_text()); native.validate_protocol(p)
    require(sha(native.OUT / 'aggregate.json') == NATIVE_AGG and not (native.OUT / 'failure.json').exists())
    old = json.loads((native.OUT / 'aggregate.json').read_text()); native.validate_result(old, p)
    a_dir = ROOT / 'BRAN_NATIVE_SCREENING_AUDIT_V1'
    a = json.loads((a_dir / 'audit.json').read_text()); m = json.loads((a_dir / 'manifest.json').read_text())
    require(not (a_dir / 'failure.json').exists() and a['status'] == 'authenticated' and a['aggregate_sha256'] == NATIVE_AGG)
    require(a['protocol_sha256'] == NATIVE_PIN and m['audit_sha256'] == sha(a_dir / 'audit.json'))
    require(a['patient_level_output_emitted'] is False and a['candidate_promoted'] is False)
    previous.baseline()
    return {'schema': 'bran-raw-teacher-distillation-protocol-v1', 'status': 'frozen_before_execution',
        'parameters': PARAMETERS, 'native_protocol_sha256': NATIVE_PIN, 'native_aggregate_sha256': NATIVE_AGG,
        'native_audit_sha256': sha(a_dir / 'audit.json'), 'native_source': p,
        'raw_reference_sha256': previous.BASELINE_PINS,
        'code_sha256': {**p['code_sha256'], **native.source.io.code_closure(CODE)}, 'runtime': native.source.io.runtime()}


def validate_protocol(p): require(p == prepare(), 'distillation_protocol_changed')


def load_initial(fold, transform, p):
    import torch
    source = native.source; path = source.PRIVATE / ('fold' + str(fold) + '.pt')
    require(sha(path) == p['native_source']['checkpoint_sha256']['fold' + str(fold)] and path.stat().st_mode & 0o777 == 0o600)
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    model = native.auth.validate_joint_bundle(bundle, json.loads(source.PROTOCOL.read_text()), fold)
    model.load_state_dict(bundle['candidate'], strict=True); model.eval()
    for key in NORMALIZERS: require(np.array_equal(bundle[key], getattr(transform, key)), 'initial_normalizer_changed')
    require(sha(path) == p['native_source']['checkpoint_sha256']['fold' + str(fold)])
    return model


def native_cbc(model, c, cm, r, rm, age, slots, transform):
    import torch
    hidden, hm = native.source.mask_inputs(c, cm, slots, 'whole_cbc_hidden', slots[0])
    require(not hidden[:, slots].any() and not hm[:, slots].any(), 'CBC_erasure_failed')
    z = native.source.lineage._state_routes(model, hidden, hm, r, rm, age)['both']
    with torch.no_grad(): value = model.cbc_joint_head(torch.tensor(z, dtype=torch.float32)).numpy()
    return value * transform.clinical_iqr[list(slots)] + transform.clinical_median[list(slots)]


def common_masks(pred, observed, endpoints):
    result = {}
    for e in endpoints:
        valid = np.asarray(observed[e], bool).copy()
        for route in metrics.ROUTES: valid &= np.isfinite(pred[e]['initial_' + route])
        result[e] = valid
    return result


def validate_result(a, p):
    keys = {'schema', 'status', 'paired_people', 'recorded_conditions', 'screening', 'completion', 'decisions',
            'baseline_replay', 'reload_predictions_equal', 'state_width', 'patient_level_output_emitted',
            'official_test_used', 'adaptive_development', 'automatic_promotion', 'new_subtype_claim', 'clinical_use'}
    require(set(a) == keys and a['schema'] == 'bran-raw-teacher-distillation-aggregate-v1' and a['status'] == 'completed')
    require(type(a['paired_people']) is int and a['paired_people'] == 1928 and type(a['recorded_conditions']) is int and a['recorded_conditions'] == 26)
    require(type(a['state_width']) is int and a['state_width'] == 192 and a['reload_predictions_equal'] is True and a['adaptive_development'] is True)
    require(all(a[k] is False for k in ('patient_level_output_emitted', 'official_test_used', 'automatic_promotion', 'new_subtype_claim', 'clinical_use')))
    require(a['baseline_replay'] == {'native_all26_three_routes': True, 'raw_all26_four_arms': True, 'native_CBC_all9': True, 'tolerance': 1e-10})
    metrics.validate(a['screening'], a['completion'], a['decisions'], p['native_source']['source']['endpoint_names'])


def validate_bundle(b, original, p, fold):
    import torch
    keys = set(NORMALIZERS) | {'student', 'continued', 'protocol_sha256', 'initial_checkpoint_sha256', 'fold', 'endpoint_names', 'cbc_fields'}
    require(set(b) == keys and b['protocol_sha256'] == sha(PROTOCOL) and type(b['fold']) is int and b['fold'] == fold)
    require(b['initial_checkpoint_sha256'] == p['native_source']['checkpoint_sha256']['fold' + str(fold)])
    require(b['endpoint_names'] == p['native_source']['source']['endpoint_names'] and b['cbc_fields'] == list(metrics.CBC_FIELDS))
    for key in NORMALIZERS: require(np.array_equal(b[key], original[key]), 'saved_normalizer_changed')
    expected = original['candidate']
    for version in ('continued', 'student'):
        state = b[version]; require(set(state) == set(expected))
        for key, value in state.items():
            require(isinstance(value, torch.Tensor) and value.shape == expected[key].shape and value.dtype == expected[key].dtype and torch.isfinite(value).all())


def run(p):
    import torch
    import bran_raw_teacher_distillation_v1 as training
    from bran_matched_screening_kernel_v1 import fit_predict, late_fusion_average
    source = native.source; torch.set_num_threads(2)
    ctx, folds, c0, cm0, eligible, r0, rm, ages, names = source.io.load_context()
    require(len(folds) == 1928 and set(np.unique(folds)) == set(range(5)))
    endpoints = p['native_source']['source']['endpoint_names']; slots = tuple(names.index(f) for f in metrics.CBC_FIELDS)
    labels = np.column_stack([ctx['labels_by_source'][e] for e in endpoints]); lm = np.column_stack([ctx['observed_by_source'][e] for e in endpoints]).astype(bool)
    pred = {e: {a: np.full(len(folds), np.nan) for a in metrics.S_ARMS} for e in endpoints}
    target = c0[:, slots].copy(); observed = (cm0 & eligible)[:, slots].copy(); score_observed = observed.copy()
    cpred = {a: np.full(target.shape, np.nan) for a in metrics.C_ARMS}
    groups = {g: np.zeros(target.shape, bool) for g in metrics.GROUPS}; cache = []
    old_raw = previous.baseline(); old_native = json.loads((native.OUT / 'aggregate.json').read_text())
    old_cbc = json.loads((source.OUT / 'aggregate.json').read_text())['retained_cbc_head_whole_panel']
    for fold in range(5):
        source.base._atomic_progress(OUT / 'progress.json', 'initial_native_and_raw_replay', fold)
        tr, te = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        transform = source.base.FoldTransform(c0, cm0, eligible, r0, rm, ages, tr)
        c, cm, r, age = transform.apply(c0, cm0, eligible, r0, rm, ages)
        initial = load_initial(fold, transform, p)
        inner, identity = source.base._inner_context(ctx, tr, fold)
        require(identity == p['native_source']['source']['authentication']['inner_fold_sha256'][fold])
        full_inner = np.full(len(folds), -1, int); full_inner[tr] = inner
        initial_pred = native.kernel.predict_native(initial, c, cm, r, rm, age)
        raw = {'raw_clinical': np.c_[c, cm, age], 'raw_retinal': np.c_[r, rm, age], 'raw_concat': np.c_[c, cm, r, rm, age]}
        for j, e in enumerate(endpoints):
            for route in metrics.ROUTES: pred[e]['initial_' + route][te] = initial_pred[route][te, j]
            for arm in raw:
                pred[e][arm][te], _ = fit_predict(raw[arm], labels[:, j], lm[:, j], tr, te, full_inner, family='extra_trees', seed=92381 + fold)
            pred[e]['late_average'][te] = late_fusion_average(pred[e]['raw_clinical'][te], pred[e]['raw_retinal'][te])
        cpred['initial'][te] = native_cbc(initial, c, cm, r, rm, age, slots, transform)[te]
        hidden, hm = source.mask_inputs(c, cm, slots, 'whole_cbc_hidden', slots[0]); physiology = hm.any(1) | rm
        score_observed[te] &= physiology[te, None]
        for j in range(9):
            fit_obs = observed[:, j] & physiology
            for arm, x in {'raw_clinical': np.c_[hidden, hm, age], 'raw_concat': np.c_[hidden, hm, r, rm, age]}.items():
                cpred[arm][te, j] = source.ev.fixed_cbc_probe(x, target[:, j], fit_obs, tr, te)
            train_y = target[tr, j][fit_obs[tr]]; require(len(train_y) >= 20)
            groups['overall'][te, j] = True
            for group, mask in previous.tail_masks(train_y, target[te, j]).items(): groups[group][te, j] = mask
        cache.append((tr, te, full_inner, transform, initial, c, cm, r, age, raw, initial_pred))
    masks = common_masks(pred, ctx['observed_by_source'], endpoints)
    for e in endpoints:
        for route in metrics.ROUTES:
            point = source.base.fold_weighted_auc(ctx['labels_by_source'][e], pred[e]['initial_' + route], masks[e], folds)
            require(abs(point - old_native['results']['endpoints'][e]['arms']['native_' + route]['auroc']) <= 1e-10, 'native_replay_failed')
        for arm in ('raw_clinical', 'raw_retinal', 'raw_concat', 'late_average'):
            point = source.base.fold_weighted_auc(ctx['labels_by_source'][e], pred[e][arm], ctx['observed_by_source'][e], folds)
            require(abs(point - old_raw['endpoints'][e]['arms'][arm]['auroc']) <= 1e-10, 'raw_replay_failed')
    for j, field in enumerate(metrics.CBC_FIELDS):
        v = score_observed[:, j]; err = cpred['initial'][v, j] - target[v, j]
        require(old_cbc[field]['status'] == 'complete' and np.isclose(np.abs(err).mean(), old_cbc[field]['mae'], rtol=1e-10, atol=1e-10) and
                np.isclose(np.square(err).mean(), old_cbc[field]['mse'], rtol=1e-10, atol=1e-10), 'native_CBC_replay_failed')
    exclusive_json(OUT / 'baseline_replay.json', {'status': 'passed', 'native_all26_three_routes': True,
        'raw_all26_four_arms': True, 'native_CBC_all9': True, 'patient_level_output_emitted': False})
    checkpoints = {}
    for fold, (tr, te, full_inner, transform, initial, c, cm, r, age, raw, initial_pred) in enumerate(cache):
        source.base._atomic_progress(OUT / 'progress.json', 'crossfitted_raw_teachers', fold)
        teachers = training.crossfit_teachers(raw['raw_clinical'], raw['raw_retinal'], labels, lm, tr, full_inner, seed=96301 + 10 * fold)
        models = {}
        for version, weight in (('continued', 0.), ('student', 1.)):
            source.base._atomic_progress(OUT / 'progress.json', 'adapt_' + version, fold)
            model = training.adapt(initial, c, cm, r, rm, age, labels, lm, tr, slots, teachers,
                                   seed=96091 + fold, steps=1500, batch_size=96, distill_weight=weight)
            model.eval(); models[version] = model
            pp = native.kernel.predict_native(model, c, cm, r, rm, age)
            for route in metrics.ROUTES:
                require(np.array_equal(np.isnan(pp[route]), np.isnan(initial_pred[route])), 'route_abstention_changed')
                for j, e in enumerate(endpoints): pred[e][version + '_' + route][te] = pp[route][te, j]
            cpred[version][te] = native_cbc(model, c, cm, r, rm, age, slots, transform)[te]
        path = PRIVATE / ('fold' + str(fold) + '.pt')
        bundle = {version: model.state_dict() for version, model in models.items()}
        bundle.update({k: getattr(transform, k) for k in NORMALIZERS})
        bundle.update(protocol_sha256=sha(PROTOCOL), initial_checkpoint_sha256=p['native_source']['checkpoint_sha256']['fold' + str(fold)],
                      fold=fold, endpoint_names=endpoints, cbc_fields=list(metrics.CBC_FIELDS))
        with path.open('xb') as handle: torch.save(bundle, handle)
        os.chmod(path, 0o600); checkpoints['fold' + str(fold)] = sha(path)
        reloaded = torch.load(path, map_location='cpu', weights_only=False)
        original = torch.load(source.PRIVATE / ('fold' + str(fold) + '.pt'), map_location='cpu', weights_only=False)
        validate_bundle(reloaded, original, p, fold)
        for version, model in models.items():
            for key, value in model.state_dict().items(): require(torch.equal(value, reloaded[version][key]))
            model.load_state_dict(reloaded[version], strict=True); model.eval()
            pp = native.kernel.predict_native(model, c, cm, r, rm, age)
            for route in metrics.ROUTES:
                for j, e in enumerate(endpoints): require(np.array_equal(pp[route][te, j], pred[e][version + '_' + route][te], equal_nan=True))
            require(np.array_equal(native_cbc(model, c, cm, r, rm, age, slots, transform)[te], cpred[version][te], equal_nan=True))
    source.base._atomic_progress(OUT / 'progress.json', 'paired_aggregate_uncertainty')
    for e in endpoints:
        for arm in metrics.S_ARMS: pred[e][arm][~masks[e]] = np.nan
    counts = source.ev.paired_counts(folds, draws=1000, seed=91501)
    screen = metrics.screening(pred, ctx['labels_by_source'], masks, folds, endpoints, counts)
    cbc = metrics.completion(target, score_observed, cpred, groups, counts)
    a = {'schema': 'bran-raw-teacher-distillation-aggregate-v1', 'status': 'completed', 'paired_people': 1928,
        'recorded_conditions': 26, 'screening': screen, 'completion': cbc, 'decisions': metrics.decisions(screen, cbc),
        'baseline_replay': {'native_all26_three_routes': True, 'raw_all26_four_arms': True, 'native_CBC_all9': True, 'tolerance': 1e-10},
        'reload_predictions_equal': True, 'state_width': 192, 'patient_level_output_emitted': False,
        'official_test_used': False, 'adaptive_development': True, 'automatic_promotion': False, 'new_subtype_claim': False, 'clinical_use': False}
    return a, checkpoints


def audit(p, pin):
    import torch
    from patient_atlas_v6_2_expanded_endpoint_evaluation import EXACT_OUTER_FOLD_HASH, EXACT_INNER_FOLD_ASSIGNMENT_SHA256
    require(not (OUT / 'failure.json').exists() and (OUT / 'aggregate.json').is_file() and (OUT / 'manifest.json').is_file())
    a = json.loads((OUT / 'aggregate.json').read_text()); validate_result(a, p)
    m = json.loads((OUT / 'manifest.json').read_text())
    require(set(m) == {'protocol_sha256', 'aggregate_sha256', 'checkpoint_sha256', 'baseline_replay_sha256', 'elapsed_seconds', 'patient_level_output_emitted'})
    require(m['protocol_sha256'] == pin and m['aggregate_sha256'] == sha(OUT / 'aggregate.json') and m['patient_level_output_emitted'] is False)
    require(metrics.finite(m['elapsed_seconds']) and m['elapsed_seconds'] >= 0 and set(m['checkpoint_sha256']) == {'fold' + str(f) for f in range(5)})
    require(m['baseline_replay_sha256'] == sha(OUT / 'baseline_replay.json'))
    require(json.loads((OUT / 'baseline_replay.json').read_text()) == {'status': 'passed', 'native_all26_three_routes': True, 'raw_all26_four_arms': True, 'native_CBC_all9': True, 'patient_level_output_emitted': False})
    for fold in range(5):
        name = 'fold' + str(fold); path = PRIVATE / (name + '.pt'); src = native.source.PRIVATE / (name + '.pt')
        require(sha(path) == m['checkpoint_sha256'][name] and path.stat().st_mode & 0o777 == 0o600)
        require(sha(src) == p['native_source']['checkpoint_sha256'][name])
        validate_bundle(torch.load(path, map_location='cpu', weights_only=False), torch.load(src, map_location='cpu', weights_only=False), p, fold)
        require(sha(path) == m['checkpoint_sha256'][name])
    auth = p['native_source']['source']['authentication']
    require(auth['outer_fold_sha256'] == EXACT_OUTER_FOLD_HASH and auth['inner_fold_sha256'] == list(EXACT_INNER_FOLD_ASSIGNMENT_SHA256))
    return {'schema': 'bran-raw-teacher-distillation-audit-v1', 'status': 'authenticated', 'protocol_sha256': pin,
        'aggregate_sha256': sha(OUT / 'aggregate.json'), 'manifest_sha256': sha(OUT / 'manifest.json'),
        'checkpoint_sha256': m['checkpoint_sha256'], 'outer_fold_sha256': auth['outer_fold_sha256'],
        'inner_fold_sha256': auth['inner_fold_sha256'], 'paired_people': 1928, 'recorded_conditions': 26,
        'baseline_replay_passed': True, 'private_checkpoint_contract_passed': True,
        'patient_level_output_emitted': False, 'automatic_promotion': False}


def main():
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'run', 'audit'): group.add_argument('--' + name, action='store_true')
    parser.add_argument('--protocol-sha256'); args = parser.parse_args()
    os.umask(0o077); ok = False; owned = False; phase = 'protocol'; start = time.monotonic()
    target = AUDIT if args.audit else OUT
    with _quiet():
        try:
            if args.prepare:
                require(not OUT.exists() and not AUDIT.exists() and not PRIVATE.exists()); exclusive_json(PROTOCOL, prepare()); ok = True
            else:
                require(args.protocol_sha256 and sha(PROTOCOL) == args.protocol_sha256)
                p = json.loads(PROTOCOL.read_text()); validate_protocol(p); target.mkdir(); owned = True
                if args.run:
                    PRIVATE.mkdir(mode=0o700)
                    with open('/private/tmp/bran_raw_teacher_distillation_v1.lock', 'a') as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); phase = 'experiment'
                        a, checkpoints = run(p); phase = 'terminal_validation'
                        validate_protocol(p); validate_result(a, p); exclusive_json(OUT / 'aggregate.json', a)
                        exclusive_json(OUT / 'manifest.json', {'protocol_sha256': args.protocol_sha256,
                            'aggregate_sha256': sha(OUT / 'aggregate.json'), 'checkpoint_sha256': checkpoints,
                            'baseline_replay_sha256': sha(OUT / 'baseline_replay.json'), 'elapsed_seconds': round(time.monotonic() - start, 1),
                            'patient_level_output_emitted': False})
                        native.source.base._atomic_completed(OUT / 'progress.json')
                else:
                    phase = 'audit'; exclusive_json(AUDIT / 'audit.json', audit(p, args.protocol_sha256))
                ok = True
        except Exception as error:
            if owned:
                exclusive_json(target / 'failure.json', {'status': 'execution_failed', 'phase': phase,
                    'error_class': type(error).__name__ if type(error) in (ValueError, TypeError, RuntimeError, OSError, KeyError) else 'other_execution_error',
                    'patient_level_output_emitted': False})
    print(json.dumps({'operation': 'prepare' if args.prepare else ('audit' if args.audit else 'run'),
                      'status': 'completed' if ok else 'failed', 'patient_level_output_emitted': False}))
    return 0 if ok else 1


if __name__ == '__main__': raise SystemExit(main())
