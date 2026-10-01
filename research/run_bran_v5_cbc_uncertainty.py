"""One quiet, locked V5 calibration diagnostic; no network fitting or selection."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np

from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json
from bran_multisource_protocol_v2 import digest
import bran_native_calibration_split_v1 as splitting
import bran_multisource_binding_v3 as binding
import run_bran_context_preservation_v5 as v5
import audit_bran_context_v5 as terminal_audit
import run_bran_multisource_calibration_v2 as legacy
import bran_v5_cbc_uncertainty as evaluation

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT/'BRAN_RESEARCH_BASELINE_V5_2026-09-19.json'
PLAN = ROOT/'BRAN_V5_CBC_UNCERTAINTY_PLAN.md'
CODE = tuple(sorted(set(v5.CODE) | set(legacy.CODE) | {
    'audit_bran_context_v5.py', 'bran_v5_cbc_uncertainty.py',
    'run_bran_v5_cbc_uncertainty.py', 'test_bran_v5_cbc_uncertainty.py'}))
PARAMETERS = {
    'roles': ['C', 'M'], 'patterns': list(evaluation.PATTERNS), 'interval_level': .9,
    'minimum_calibration': 20, 'calibration_fraction': .5, 'split_salt': splitting.SALT,
    'split_scope': 'outcome_blind_halves_within_each_checkpoint_heldout_fold',
    'bootstrap_draws': 1000, 'bootstrap_seed': 94701, 'minimum_valid_draws': 900,
    'low_hb_stratum': 'observed Hb <12 g/dL; research stratum, not universal clinical definition',
    'coverage_small_complement_rule': 'withhold when hits or misses are 1 through 19',
    'internal_metric_slot_adapter': {'C': 'mlp', 'M': 'token'},
    'adapter_is_not_architecture_label': True, 'network_training_steps': 0,
    'parameter_selection': False, 'protected_external_data_used': False,
    'historical_gates_changed': False, 'clinical_use': False,
    'patient_level_output_permitted': False,
}


def require(ok):
    if not ok:
        raise ValueError('v5_uncertainty_runner_contract_failed')


def paths(attempt):
    require(type(attempt) is int and 1 <= attempt <= 99)
    return ROOT/f'BRAN_V5_CBC_UNCERTAINTY_ATTEMPT{attempt}'


def code_hashes():
    return {name: sha(ROOT/name) for name in CODE}


def small_authentication():
    base = json.loads(BASELINE.read_text())
    require(base['role'] == 'M' and base['status'] == 'user_selected_lead_research_candidate')
    require(base['historical_candidate_promoted'] is False and base['historical_promotion_eligible'] is False)
    for prefix in ('fit_protocol', 'fit_aggregate', 'evaluation_aggregate', 'native_profiles'):
        require(sha(ROOT/base[prefix+'_path']) == base[prefix+'_sha256'])
    audit = terminal_audit.audit(2)
    require(audit['status'] == 'authenticated' and audit['promotion_eligible'] is False)
    fit_out, private = v5.paths('fit', 2)
    fp, fa, ft = v5.auth_terminal(fit_out)
    components = {}
    require([r['fold'] for r in base['fold_checkpoints']] == list(range(5)))
    for fold in range(5):
        for role in evaluation.ROLES:
            name = f'fold{fold}_{role}.json'
            require(sha(fit_out/name) == fa['component_sha256'][name])
            item = json.loads((fit_out/name).read_text())
            require(sha(private/f'fold{fold}_{role}.pt') == item['checkpoint_sha256'])
            components[(role, fold)] = item
        row = base['fold_checkpoints'][fold]
        require(row['checkpoint_path'] == str((private/f'fold{fold}_M.pt').relative_to(ROOT)))
        require(row['checkpoint_sha256'] == components[('M', fold)]['checkpoint_sha256'])
        require(row['receipt_sha256'] == fa['component_sha256'][f'fold{fold}_M.json'])
    return base, fp, fa, components


def source_context():
    base, fp, fa, components = small_authentication()
    sources = binding.load_bound_sources()
    require(fp['source_binding'] == sources.receipt())
    paired = sources.paired
    source = binding.roles.old.origin.native.source
    ctx, folds = v5.references._align(paired, source, tuple(paired.endpoint_names), sources.context)[:2]
    require(np.array_equal(folds, paired.folds))
    ids = list(map(str, ctx['raw_cohort'].patient_ids))
    roles = splitting.make_roles(ids, folds)
    split_auth = {'patient_order_sha256': digest(ids), 'roles_sha256': digest(roles.tolist())}
    require(split_auth == dict(sources.context.result['calibrated_completion']['split_authentication']))
    evidence = {'baseline_record_sha256': sha(BASELINE), 'plan_sha256': sha(PLAN),
                'fit_aggregate_sha256': base['fit_aggregate_sha256'],
                'evaluation_aggregate_sha256': base['evaluation_aggregate_sha256'],
                'component_sha256': fa['component_sha256'], 'source_binding': sources.receipt(),
                'split_authentication': split_auth}
    return paired, roles, components, evidence


def protocol(evidence):
    return {'schema': 'bran-v5-cbc-uncertainty-protocol-v1', 'status': 'frozen_before_inference',
            'parameters': PARAMETERS, 'code_sha256': code_hashes(), 'source_binding': evidence}


def progress(out, state, event):
    require(set(event) <= {'phase', 'role', 'fold'})
    require(event['phase'] in ('source_authentication', 'inference', 'checkpoint_replay',
                               'interval_fitting', 'aggregate_bootstrap', 'post_authentication'))
    require('role' not in event or event['role'] in evaluation.ROLES)
    require('fold' not in event or type(event['fold']) is int and event['fold'] in range(5))
    state['phase'] = event['phase']
    temp = out/'progress.next.json'
    write_json(temp, {**event, 'pid': os.getpid(), 'patient_level_output_emitted': False})
    os.replace(temp, out/'progress.json')


def run(attempt, state):
    start = time.monotonic()
    out = paths(attempt)
    require(not out.exists() and not out.is_symlink())
    out.mkdir(); state['owned'] = out
    progress(out, state, {'phase': 'source_authentication'})
    paired, roles, components, evidence = source_context()
    frozen = protocol(evidence)
    write_json(out/'protocol.json', frozen)
    pin = sha(out/'protocol.json')
    _, private = v5.paths('fit', 2)

    def provider(role, fold):
        require(role in evaluation.ROLES and type(fold) is int and fold in range(5))
        item = components[(role, fold)]
        require(item['binding']['role'] == role and item['binding']['fold'] == fold)
        return v5.oldfit.load_checkpoint(private/f'fold{fold}_{role}.pt',
                                         item['checkpoint_sha256'], item['binding'])

    result = evaluation.evaluate(paired, roles, provider, progress=lambda e: progress(out, state, e))
    evaluation.validate_patterns(result['patterns'])
    progress(out, state, {'phase': 'post_authentication'})
    # Revalidate every source receipt and immutable candidate binding after work.
    _, _, _, replay_evidence = source_context()
    require(replay_evidence == evidence and protocol(evidence) == frozen)
    require(sha(out/'protocol.json') == pin)
    result.update(protocol_sha256=pin, source_binding=evidence)
    write_json(out/'aggregate.json', result)
    manifest = {'protocol_sha256': pin, 'aggregate_sha256': sha(out/'aggregate.json'),
                'patient_level_output_emitted': False}
    write_json(out/'manifest.json', manifest)
    audit_record = {'status': 'authenticated', **manifest,
                    'all_ten_checkpoint_bindings_authenticated': True,
                    'calibration_split_authenticated': True, 'inference_replayed': True,
                    'model_parameters_unchanged': True,
                    'aggregate_schema_validated': True,
                    'independent_full_aggregate_recomputation': False,
                    'historical_gates_changed': False, 'candidate_promoted': False}
    write_json(out/'audit.json', audit_record)
    write_json(out/'completed.json', {'status': 'completed', **manifest,
               'audit_sha256': sha(out/'audit.json'), 'elapsed_seconds': time.monotonic()-start,
               'candidate_promoted': False, 'scientific_goal_achieved': False})
    state['owned'] = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    state = {'owned': None, 'phase': 'authentication'}
    ok = False
    with quiet():
        try:
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                run(args.attempt, state)
                ok = True
        except Exception:
            if state['owned'] is not None and not (state['owned']/'completed.json').exists():
                try:
                    write_json(state['owned']/'failure.json', {'status': 'technical_failure',
                               'phase': state['phase'], 'patient_level_output_emitted': False,
                               'candidate_promoted': False})
                except Exception:
                    pass
    print(json.dumps({'status': 'completed' if ok else 'not_completed', 'phase': state['phase'],
                      'patient_level_output_emitted': False, 'candidate_promoted': False}))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
