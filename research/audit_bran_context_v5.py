"""Row-free terminal audit for the V5 context-preservation queue.

This authenticates receipts and hashes only. It never deserializes checkpoints,
opens patient arrays, or recomputes patient-level aggregates.
"""
import argparse
import json
import re
from pathlib import Path

import bran_context_diagnostic_v5 as context
import run_bran_context_preservation_v5 as run


_SOURCES = ('brset', 'eicu', 'mimiciii', 'mimiciv', 'nhanes_exposed', 'nwicu')
_CONTEXTS = ('single_target_hidden', 'whole_cbc_hidden', 'red_cell_hidden',
             'single_target_no_retina', 'whole_cbc_no_retina', 'red_cell_no_retina')

def require(ok):
    if not ok:
        raise ValueError('context_v5_audit_failed')


def _sha(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None)


def _binding(value):
    require(value['protected_sources_used'] is False and value['patient_level_output_emitted'] is False)
    _sha(value['outer_fold_sha256'])
    require(type(value['inner_fold_sha256']) is list and len(value['inner_fold_sha256']) == 5)
    for item in value['inner_fold_sha256']:
        _sha(item)


def _counters(value, role, updates):
    require(set(value) == set(run.COUNTERS) and role in run.ROLES)
    require(all(type(item) is int and 0 <= item <= updates for item in value.values()))
    require(value['cap_contract_checks'] == updates and value['preservation_supported_updates'] > 0)
    if role == 'C':
        require(value['cap_applied_updates'] == 0
                and value['source_generative_nonzero_updates'] == 0
                and value['source_cbc_nonzero_updates'] == 0)
    else:
        require(value['source_generative_nonzero_updates'] > 0
                and value['source_cbc_nonzero_updates'] > 0)


def _private_checkpoint(private, name, expected):
    path = private / name
    require(not private.is_symlink() and private.stat().st_mode & 0o777 == 0o700
            and not path.is_symlink() and path.stat().st_nlink == 1
            and path.stat().st_mode & 0o777 == 0o600)
    require(run.sha(path) == expected)


def _validate_context_summary(summary):
    require(type(summary) is dict and set(summary) == set(_SOURCES))
    for source in _SOURCES:
        require(type(summary[source]) is dict and set(summary[source]) == set(_CONTEXTS))
        for name in _CONTEXTS:
            context.validate(summary[source][name])


def _coarse(value):
    require(type(value) is int and value >= 20 and value % 20 == 0)


def _exposure(value):
    require(set(value) == {'per_source', 'global_unique_people', 'cross_source_identity_resolved',
                           'fold_exposures_must_not_be_summed'})
    require(value['global_unique_people'] is None and value['cross_source_identity_resolved'] is False
            and value['fold_exposures_must_not_be_summed'] is True)
    per = value['per_source']
    require(set(per) == {*_SOURCES, 'aireadi'})
    for name in _SOURCES:
        item = per[name]
        require({'source_local_people_lower_bound_20', 'unique_examples_lower_bound_20'} <= set(item))
        _coarse(item['source_local_people_lower_bound_20'])
        _coarse(item['unique_examples_lower_bound_20'])
        remaining = set(item) - {'source_local_people_lower_bound_20', 'unique_examples_lower_bound_20'}
        require(remaining in ({'unique_observed_measurements_lower_bound_20'},
                              {'unique_retinal_images_lower_bound_20'}))
        _coarse(item[next(iter(remaining))])
    require(set(per['aireadi']) == {'paired_people_lower_bound_20',
                                     'unique_observed_clinical_measurements_lower_bound_20',
                                     'pooled_retinal_inputs_lower_bound_20',
                                     'pooled_vectors_not_counted_as_individual_images'})
    _coarse(per['aireadi']['paired_people_lower_bound_20'])
    _coarse(per['aireadi']['unique_observed_clinical_measurements_lower_bound_20'])
    _coarse(per['aireadi']['pooled_retinal_inputs_lower_bound_20'])
    require(per['aireadi']['pooled_vectors_not_counted_as_individual_images'] is True)


def audit(attempt):
    stages = {stage: run.auth_terminal(run.paths(stage, attempt)[0])
              for stage in ('diagnostic', 'pilot', 'fit', 'evaluate')}
    dp, da, dt = stages['diagnostic']; pp, pa, pt = stages['pilot']
    fp, fa, ft = stages['fit']; ep, ea, et = stages['evaluate']
    require((dp['stage'], pp['stage'], fp['stage'], ep['stage']) ==
            ('diagnostic', 'pilot', 'fit', 'evaluate'))
    require(dp['source_binding'] == pp['source_binding'] == fp['source_binding'] == ep['source_binding'])
    binding = fp['source_binding']; _binding(binding)
    require(dp['parameters'] == pp['parameters'] == fp['parameters'] == ep['parameters'] == run.PARAMETERS)
    require(dp['code_sha256'] == pp['code_sha256'] == fp['code_sha256'] == ep['code_sha256'] == run.code_hashes())
    _validate_context_summary(da['summary'])
    require(da['status'] == 'completed' and da['training_only'] is True and da['training_updates'] == 0
            and da['five_fold_initial_models_unchanged'] is True
            and da['heldout_scoring_performed'] is False and da['protected_sources_used'] is False
            and da['patient_level_output_emitted'] is False)

    diagnostic_out, _ = run.paths('diagnostic', attempt)
    pilot_out, _ = run.paths('pilot', attempt)
    fit_out, private = run.paths('fit', attempt)
    eval_out, _ = run.paths('evaluate', attempt)
    require(pp['dependencies']['diagnostic_terminal_sha256'] == run.sha(diagnostic_out / 'completed.json'))
    require(fp['dependencies']['pilot_terminal_sha256'] == run.sha(pilot_out / 'completed.json'))
    require(ep['dependencies']['fit_terminal_sha256'] == run.sha(fit_out / 'completed.json'))
    require(dp['dependencies']['V3_fit'] == pp['dependencies']['V3_fit']
            == fp['dependencies']['V3_fit'] == ep['dependencies']['V3_fit'])
    old_receipt = fp['dependencies']['V3_fit']
    oldout, oldprivate = run.oldfit.paths(1)
    require(not (oldout / 'failure.json').exists())
    require(run.sha(oldout / 'protocol.json') == old_receipt['protocol_sha256']
            and run.sha(oldout / 'aggregate.json') == old_receipt['aggregate_sha256']
            and run.sha(oldout / 'manifest.json') == old_receipt['manifest_sha256'])

    require(pa['status'] == 'pilot_passed' and pa['pilot_models_discarded'] is True
            and pa['heldout_scoring_performed'] is False)
    require(set(pa['arms']) == set(run.ROLES))
    require(pa['arms']['C']['updates'] == pa['arms']['M']['updates'] == 100
            and pa['arms']['C']['digests'] == pa['arms']['M']['digests'])
    for role, item in pa['arms'].items():
        _counters(item['algorithm_update_counters'], role, 100)

    names = {f'fold{fold}_{role}.json' for fold in range(5) for role in run.ROLES}
    require(fa['status'] == 'fits_completed_pending_evaluation'
            and fa['ten_final_checkpoints_reloaded'] is True
            and fa['paired_inputs_masks_and_bridge_match_C_M'] is True
            and fa['all_seven_qualified_sources_sampled'] is True
            and fa['source_exposure_in_C_is_masks_only'] is True
            and fa['historical_models_unchanged'] is True
            and fa['protected_sources_used'] is False)
    require(set(fa['component_sha256']) == names)
    updates = 0
    for fold in range(5):
        old_name = f'fold{fold}_C.json'
        require(run.sha(oldout / old_name) == old_receipt['component_sha256'][old_name])
        control = json.loads((oldout / old_name).read_text())
        traces = None
        for role in run.ROLES:
            name = f'fold{fold}_{role}.json'
            require(run.sha(fit_out / name) == fa['component_sha256'][name])
            item = json.loads((fit_out / name).read_text())
            b = item['binding']
            require(b['protocol_sha256'] == ft['protocol_sha256'] and b['fold'] == fold and b['role'] == role
                    and b['training_recipe'] == 'context_preservation_v5'
                    and b['outer_fold_sha256'] == binding['outer_fold_sha256']
                    and b['inner_fold_sha256'] == binding['inner_fold_sha256'][fold]
                    and b['transform_sha256'] == control['binding']['transform_sha256']
                    and b['initial_checkpoint_sha256'] == control['binding']['initial_checkpoint_sha256'])
            require(item['checkpoint_reload_exact'] is True and item['updates_completed'] == 3000
                    and item['source_loss_supported'] is (role == 'M')
                    and item['paired_input_digest'] == control['paired_input_digest']
                    and item['paired_completion_mask_digest'] == control['paired_mask_digest'])
            trace = {key: item[key] for key in ('paired_input_digest', 'paired_completion_mask_digest', 'bridge_mask_digest')}
            require(traces is None or traces == trace); traces = trace
            _counters(item['algorithm_update_counters'], role, 3000)
            require(item['source_exposure_semantics'] ==
                    ('availability_masks_only' if role == 'C' else 'measurement_values_and_masks'))
            _exposure(item['exposure'])
            require(item['candidate_promoted'] is False and item['patient_level_output_emitted'] is False)
            _private_checkpoint(private, f'fold{fold}_{role}.pt', item['checkpoint_sha256'])
            updates += item['updates_completed']

    require(ea['status'] == 'completed' and ea['roles'] == {**run.ROLES, 'I': 'unchanged_initial'})
    require(run.sha(eval_out / 'native_profiles.json') == ea['native_profiles_sha256'])
    native = json.loads((eval_out / 'native_profiles.json').read_text())
    require(native['reload_predictions_equal'] is True and native['all_empty_physiology_abstained'] is True
            and native['patient_level_output_emitted'] is False and native['candidate_promoted'] is False)
    run.oldeval.validate_result(ea['comparison'])
    require(ea['promotion_eligible'] is ea['comparison']['promotion_eligible']
            and ea['candidate_promoted'] is False and ea['protected_sources_used'] is False
            and ea['historical_gate_definitions_changed'] is False
            and ea['independent_candidate_aggregate_recomputation'] is False
            and ea['scientific_goal_achieved'] is False)
    require(updates == 30000)
    return {'schema': 'bran-context-v5-terminal-audit', 'status': 'authenticated',
            'diagnostic_aggregate_sha256': dt['aggregate_sha256'],
            'evaluation_aggregate_sha256': et['aggregate_sha256'],
            'fit_aggregate_sha256': ft['aggregate_sha256'], 'full_training_updates': updates,
            'five_outer_inner_fold_bindings_authenticated': True,
            'paired_and_bridge_trace_identity_authenticated': True,
            'all_ten_candidate_checkpoint_hashes_authenticated': True,
            'context_aggregate_closed_schema_authenticated': True,
            'gate_decisions_independently_recomputed_from_aggregate_metrics': True,
            'candidate_predictions_replayed_by_evaluation_runner': True,
            'independent_candidate_aggregate_recomputation': False,
            'promotion_eligible': ea['promotion_eligible'], 'candidate_promoted': False,
            'patient_level_output_emitted': False,
            'audit_code_sha256': run.sha(Path(__file__).resolve())}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.attempt)
        out, _ = run.paths('evaluate', args.attempt)
        run.write_json(out / 'audit.json', result)
        print(json.dumps(result))
    except Exception:
        print('{"status":"audit_failed","patient_level_output_emitted":false}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
