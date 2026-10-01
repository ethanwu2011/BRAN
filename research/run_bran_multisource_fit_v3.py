"""Exclusive local V3 C/M fits; immutable I and protected cohorts stay untouched.

No evaluation-based choices. Final checkpoint per arm/fold is retained privately,
reloaded exactly, and tied to source, fold, code, normalizer and protocol hashes.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import torch

from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json
from run_bran_multisource_pilot_v3 import CODE as PILOT_CODE, authenticate_preflight
from bran_multisource_binding_v3 import load_bound_sources, bind_fold
from bran_multisource_fit_v3 import fit_one_v3
from bran_multisource_fit_v2 import _transform_payload, _restore_transform, _exposure
from bran_multisource_batches_v2 import transform_hash
from bran_multisource_anchored_model_v3 import BRANMultisourceAnchoredModelV3
from bran_multisource_protocol_v2 import digest

ROOT = Path(__file__).resolve().parent
PILOT = ROOT/'BRAN_MULTISOURCE_PILOT_V3_ATTEMPT1'
ROLES = ('C', 'M')
CODE = tuple(sorted(set(PILOT_CODE) | {'run_bran_multisource_fit_v3.py',
    'bran_multisource_fit_v3.py', 'bran_multisource_fit_v2.py',
    'bran_multisource_mask_contract_v1.py', 'bran_multisource_advancement_v2.py'}))
PARAMETERS = {
    'folds': 5, 'roles': list(ROLES), 'updates_per_fold_role': 3000,
    'seed_base': 95101, 'paired_seed_offset': 202, 'source_seed_offset': 404,
    'paired_batch': 96, 'source_batch': 128, 'learning_rate': 5e-5,
    'weight_decay': 1e-4, 'gradient_clip': 5., 'clinical_encoder': 'inherited_mlp',
    'retinal_feature_backbone_frozen': True, 'state_width': 192,
    'paired_loss': {'generative': 1., 'screening': 1., 'cbc': .5, 'state_preservation': .1},
    'source_loss': {'ramp_to': .1, 'ramp_steps': 300, 'generative': 1., 'cbc': .5},
    'visible_reconstruction': .1, 'retained_visible_clinical_anchor': .1,
    'kl_max': .001, 'kl_warmup_steps': 300,
    'reported_age_probabilities': {'reported': .8, 'unknown': .1, 'five_year_interval': .1},
    'state_preservation': 'reported_age_paired_same_visible_inputs_full192_train_only_std_floor1',
    'source_labels_permitted': False, 'checkpoint_selection': 'fixed_final_update_only',
    'scientific_retries': 'none_automatic', 'technical_retry': 'identified_cause_new_attempt_only',
}
EVALUATION = {
    'roles': {'I': 'unchanged_native_per_fold', 'C': 'paired_only_continuation',
              'M': 'same_continuation_plus_incomplete_sources'},
    'native_screening': 'locked_26_recorded_conditions_macro_and_each_AUROC',
    'native_completion': 'all_nine_CBC_six_existing_erasure_contexts',
    'stress': 'existing_field_and_modality_patterns_plus_new_age_profiles',
    'historical_advancement': 'unchanged_gates_for_each_candidate_vs_historical_initial_continued_and_raw',
    'matched_advancement': 'same_gates_with_continued_role_replaced_by_C_and_student_by_M',
    'eligible_for_promotion': 'M_only_and_both_historical_and_matched_advancement_must_pass',
    'bootstrap': {'draws': 1000, 'seed': 91501, 'minimum_valid': 900,
                  'unit': 'matched_participant_within_outer_fold_fixed_fits'},
    'multiple_required_gates': 'intersection_union_all_required_no_selective_gate_reporting',
    'endpoint_discovery': 'descriptive_unless_prespecified_familywise_or_FDR_adjustment',
    'clinical_low_hb': 'Hb_below12_g_dL_separate_from_original_training_quantile_gate',
    'protected_external_sources': 'never_encoder_selection_or_loss_tuning',
    'subtypes': 'not_established_by_training_or_screening_success',
    'selection_uncertainty': 'fixed_fit_bootstrap_does_not_cover_all_development_selection',
}


def require(ok):
    if not ok: raise ValueError('multisource_fit_v3_failed')


def code_hashes():
    return {name: sha(ROOT/name) for name in CODE}


def paths(attempt):
    require(type(attempt) is int and 1 <= attempt <= 99)
    return (ROOT/f'BRAN_MULTISOURCE_FIT_V3_ATTEMPT{attempt}',
            ROOT/'private_artifacts'/f'bran_multisource_fit_v3_attempt{attempt}')


def authenticate_pilot():
    require(not (PILOT/'failure.json').exists())
    m = json.loads((PILOT/'manifest.json').read_text())
    a = json.loads((PILOT/'aggregate.json').read_text())
    require(sha(PILOT/'aggregate.json') == m['aggregate_sha256']
        and sha(PILOT/'protocol.json') == m['protocol_sha256']
        and all(sha(ROOT/name) == pin for name, pin in m['code_sha256'].items())
        and a['status'] == 'pilot_passed' and a['patient_level_output_emitted'] is False
        and a['pilot_models_discarded'] is True and a['heldout_scoring_performed'] is False
        and a['paired_batches_equivalent'] is True and a['paired_mask_streams_equivalent'] is True
        and a['source_generative_supported'] is True and a['source_cbc_supported'] is True)
    authenticate_preflight()
    return sha(PILOT/'manifest.json')


def progress(out, phase, fold=None, role=None, updates=0):
    require(phase in ('source_binding', 'training', 'checkpoint_replay', 'post_fit_authentication',
                      'fits_completed_pending_evaluation'))
    require(fold is None or type(fold) is int and fold in range(5))
    require(role is None or role in ROLES)
    require(type(updates) is int and 0 <= updates <= 3000)
    value = {'phase': phase, 'fold': fold, 'role': role, 'updates_completed': updates,
             'pid': os.getpid(), 'patient_level_output_emitted': False}
    tmp = out/'progress.next.json'
    write_json(tmp, value)
    os.replace(tmp, out/'progress.json')


def save_checkpoint(path, result, bound, binding):
    require(not path.exists() and not path.is_symlink())
    payload = {'schema': 'bran-multisource-continuation-checkpoint-v3',
        'config': result['model'].export_config(), 'binding': binding, 'updates': 3000,
        'state_dict': result['model'].state_dict(), 'optimizer_state': result['optimizer'].state_dict(),
        'input_transform': _transform_payload(bound.transform)}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(path)


def load_checkpoint(path, expected_sha, binding):
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
        and path.stat().st_mode & 0o777 == 0o600 and sha(path) == expected_sha)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(set(payload) == {'schema', 'config', 'binding', 'updates', 'state_dict',
                            'optimizer_state', 'input_transform'}
        and payload['schema'] == 'bran-multisource-continuation-checkpoint-v3'
        and payload['binding'] == binding and payload['updates'] == 3000
        and binding['role'] in ROLES and binding['fold'] in range(5))
    model = BRANMultisourceAnchoredModelV3.from_config(payload['config'])
    require(digest(model.export_config()) == binding['model_config_sha256'])
    require(all(not t.is_floating_point() or bool(torch.isfinite(t).all())
                for t in payload['state_dict'].values()))
    model.load_state_dict(payload['state_dict'], strict=True)
    transform = _restore_transform(payload['input_transform'], binding['transform_sha256'])
    require(transform.heldout_fold == binding['fold'] and sha(path) == expected_sha)
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model, transform


def execute(out, private, state):
    state['phase'] = 'source_binding'
    progress(out, state['phase'])
    code, pilot_pin = code_hashes(), authenticate_pilot()
    sources = load_bound_sources()
    source_receipt = sources.receipt()
    protocol = {'schema': 'bran-multisource-continuation-protocol-v3',
        'status': 'frozen_before_any_full_fit', 'parameters': PARAMETERS, 'evaluation': EVALUATION,
        'code_sha256': code, 'source_binding': source_receipt, 'pilot_manifest_sha256': pilot_pin,
        'plan_sha256': sha(ROOT/'BRAN_MULTISOURCE_CONTINUATION_V3_PLAN.md'),
        'evaluation_code_must_be_frozen_before_scoring': True,
        'patient_level_output_emitted': False}
    write_json(out/'protocol.json', protocol)
    protocol_pin = sha(out/'protocol.json')
    private.mkdir(mode=0o700)
    components = {}
    torch.set_num_threads(2)
    for fold in range(5):
        bound = bind_fold(sources, fold)
        traces = []
        for role in ROLES:
            require(code_hashes() == code and sha(out/'protocol.json') == protocol_pin)
            state['phase'] = 'training'
            progress(out, state['phase'], fold, role)
            result = fit_one_v3(bound.model, bound.teacher, bound.paired_factory, bound.source_factory,
                bound.state_scale, bound.transform.age_mean, bound.transform.age_scale,
                bound.seed, bound.cbc_indices, role,
                progress=lambda updates: progress(out, 'training', fold, role, updates))
            binding = {'protocol_sha256': protocol_pin, 'fold': fold, 'role': role,
                'initial_checkpoint_sha256': sources.context.protocol['native_source']['checkpoint_sha256'][f'fold{fold}'],
                'transform_sha256': transform_hash(bound.transform),
                'model_config_sha256': digest(result['model'].export_config()),
                'clinical_field_order_sha256': digest(sources.paired.names),
                'outer_fold_sha256': source_receipt['outer_fold_sha256'],
                'inner_fold_sha256': source_receipt['inner_fold_sha256'][fold]}
            name = f'fold{fold}_{role}'
            state['phase'] = 'checkpoint_replay'
            progress(out, state['phase'], fold, role, 3000)
            checkpoint_pin = save_checkpoint(private/(name+'.pt'), result, bound, binding)
            restored, transform = load_checkpoint(private/(name+'.pt'), checkpoint_pin, binding)
            require(transform_hash(transform) == transform_hash(bound.transform)
                and all(torch.equal(value, restored.state_dict()[key])
                    for key, value in result['model'].state_dict().items()))
            exposure = _exposure(result['source_sampler'].private_sampler,
                result['paired_sampler'].private_sampler, sources.paired, sources.pools)
            if role == 'C':
                exposure['per_source'] = {'aireadi': exposure['per_source']['aireadi']}
                exposure['source_samples_evaluated_for_training_loss'] = False
            else:
                require(all(exposure['per_source'][p.source]['source_local_people_lower_bound_20']
                            is not None for p in sources.pools))
            item = {'status': 'fit_completed_pending_evaluation', 'binding': binding,
                'checkpoint_sha256': checkpoint_pin, 'checkpoint_reload_exact': True,
                'updates_completed': result['updates'], 'runtime_seconds': result['elapsed_seconds'],
                'source_loss_supported': result['source_gradient_supported'], 'exposure': exposure,
                'paired_input_digest': result['paired_input_digest'],
                'paired_mask_digest': result['paired_completion_mask_digest'],
                'candidate_promoted': False, 'patient_level_output_emitted': False}
            write_json(out/(name+'.json'), item)
            components[name+'.json'] = sha(out/(name+'.json'))
            traces.append((item['paired_input_digest'], item['paired_mask_digest']))
            del result, restored
        require(traces[0] == traces[1])
    state['phase'] = 'post_fit_authentication'
    progress(out, state['phase'])
    require(code_hashes() == code and authenticate_pilot() == pilot_pin
        and load_bound_sources().receipt() == source_receipt)
    aggregate = {'schema': 'bran-multisource-continuation-fits-v3',
        'status': 'fits_completed_pending_evaluation', 'component_sha256': components,
        'protocol_sha256': protocol_pin, 'ten_final_checkpoints_reloaded': True,
        'paired_role_inputs_and_masks_identical': True, 'five_folds_completed': True,
        'all_seven_qualified_sources_sampled_by_M': True,
        'historical_models_unchanged': True, 'protected_external_sources_used': False,
        'candidate_promoted': False, 'scientific_goal_achieved': False,
        'patient_level_output_emitted': False}
    write_json(out/'aggregate.json', aggregate)
    write_json(out/'manifest.json', {'aggregate_sha256': sha(out/'aggregate.json'),
        'protocol_sha256': protocol_pin, 'component_sha256': components,
        'patient_level_output_emitted': False})
    progress(out, 'fits_completed_pending_evaluation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    out, private = paths(args.attempt)
    require(not out.exists() and not out.is_symlink() and not private.exists() and not private.is_symlink())
    state, ok = {'phase': 'lock_acquisition'}, False
    with quiet():
        try:
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                out.mkdir()
                execute(out, private, state)
                ok = True
        except Exception:
            if out.is_dir() and not (out/'aggregate.json').exists():
                write_json(out/'failure.json', {'status': 'technical_fit_failure',
                    'phase': state['phase'], 'completed_components_preserved': True,
                    'patient_level_output_emitted': False, 'candidate_promoted': False})
    print(json.dumps({'status': 'fits_completed_pending_evaluation' if ok else 'fit_not_completed',
        'phase': state['phase'], 'patient_level_output_emitted': False, 'candidate_promoted': False}))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
