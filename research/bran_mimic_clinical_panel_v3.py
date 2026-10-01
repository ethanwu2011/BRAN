"""Private, outcome-blind clinical structure and fixed outcome controls.

No source access or encoder fitting. The lifecycle authenticates the V3 frame,
row bindings and source urgency context. This component never emits row-level
values, group assignments, predictions or small cells.
"""
from dataclasses import dataclass
import hashlib
import math
import numpy as np
import bran_mimic_clinical_design_v3 as design
import bran_r1_hf_structure_kernel_v1 as structure
import bran_cross_cohort_assignment_v1 as assignment
from bran_clinical_discovery_landmark_v3 import FAMILIES
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_clinical_chemistry_semantics_v1 import CHEMISTRY_FIELDS

ERROR = 'bran_mimic_clinical_panel_v3_contract_failed'
LAB_FIELDS = tuple(CBC_FIELDS) + tuple(CHEMISTRY_FIELDS)
ROLES = ('discovery', 'characterization_development', 'test')
FLAGS = {'patient_level_output_emitted': False, 'novel_subtype_claim': False,
         'external_validation_claim': False, 'encoder_updated': False,
         'clinical_utility_established': False, 'clinical_use': False}


def require(ok):
    if not ok:
        raise ValueError(ERROR) from None


def check_frame(frame, urgency):
    n = len(frame['roles'])
    required = {
        'source_row': ((n,), np.int64), 'row_binding': ((n,), 'U64'),
        'person_group': ((n,), np.int64), 'roles': ((n,), np.uint8),
        'selected': ((n, 3), bool), 'labels': ((n, 3), np.int8),
        'current_membership': ((n, 3), bool), 'values': ((n, 21), np.float64),
        'observed': ((n, 21), bool), 'age_triplet': ((n, 3), np.float64),
        'age_kind': ((n,), np.uint8), 'state': ((n, 192), np.float32),
        'available': ((n,), bool), 'clinical': ((n, 59), np.float64),
        'clinical_mask': ((n, 59), bool),
    }
    require(set(frame) == set(required) and len(LAB_FIELDS) == 21 and n >= 20)
    for key, (shape, dtype) in required.items():
        a = frame[key]
        require(type(a) is np.ndarray and a.shape == shape and a.dtype == np.dtype(dtype))
    require(np.unique(frame['person_group']).size == n and np.unique(frame['row_binding']).size == n
            and np.unique(frame['source_row']).size == n and np.isin(frame['roles'], (0, 1, 2)).all()
            and np.all(frame['person_group'] >= 0) and np.all(frame['source_row'] >= 0)
            and all(len(v) == 64 and set(v) <= set('0123456789abcdef') for v in frame['row_binding'])
            and np.isin(frame['labels'], (-1, 0, 1)).all() and np.isfinite(frame['state']).all()
            and np.isfinite(frame['values'][frame['observed']]).all()
            and not np.any(frame['selected'] & ~frame['current_membership'])
            and np.array_equal(frame['selected'], frame['labels'] >= 0))
    require(type(urgency) is np.ndarray and urgency.shape == (n, 10) and urgency.dtype == np.float64
            and np.isin(urgency, (0., 1.)).all() and np.all(urgency.sum(axis=1) == 1))


def _coverage(selected, accepted):
    n, yes = int(selected), int(accepted)
    require(0 <= yes <= n)
    if min(yes, n-yes) < 20:
        return {'status': 'suppressed_complement_support'}
    return {'status': 'released', 'eligible_lower_bound': (n//20)*20,
            'retained_lower_bound': (yes//20)*20, 'retained_fraction': float(yes/n)}


def _wilson(events, n):
    p, z = events/n, 1.959963984540054
    denominator = 1+z*z/n
    middle = (p+z*z/(2*n))/denominator
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0., middle-radius), min(1., middle+radius)]


def profiles(values, observed, labels, outcome, k):
    """Known-outcome test population; whole risk table has >=20 of each class.

    Numeric lab profiles are descriptive, not independent subgroup validation.
    A field is suppressed for the entire table unless >=20 observed values are
    available in every group. Neither missing-value counts nor small cells are
    released. Risk intervals are Wilson, not bootstrap model-performance CIs.
    """
    known = outcome >= 0
    counts = [(int(np.sum(known & (labels == g) & (outcome == 1))),
               int(np.sum(known & (labels == g) & (outcome == 0)))) for g in range(k)]
    if not all(min(pair) >= 20 for pair in counts):
        return {'status': 'suppressed_outcome_complement_support'}
    field_ok = [all(np.sum(known & (labels == g) & observed[:, j]) >= 20 for g in range(k))
                for j in range(21)]
    groups = []
    for g, (deaths, survivors) in enumerate(counts):
        n = deaths+survivors; cells = {}
        for j, name in enumerate(LAB_FIELDS):
            if not field_ok[j]:
                cells[name] = {'status': 'suppressed_observed_support'}
                continue
            x = values[known & (labels == g) & observed[:, j], j]
            q = np.quantile(x, (.25, .5, .75))
            cells[name] = {'status': 'released', 'q25': float(q[0]), 'median': float(q[1]), 'q75': float(q[2])}
        groups.append({'group': g, 'known_outcome_people_lower_bound': (n//20)*20,
            'unadjusted_hospital_death_rate': float(deaths/n), 'wilson95': _wilson(deaths, n), 'labs': cells})
    return {'status': 'released', 'groups': groups,
            'labs_are_clustering_inputs_not_independent_validation': True}


def validate_coverage(value):
    if value == {'status': 'suppressed_complement_support'}:
        return
    require(type(value) is dict and set(value) == {'status', 'eligible_lower_bound', 'retained_lower_bound', 'retained_fraction'}
            and value['status'] == 'released')
    n, y, f = (value[k] for k in ('eligible_lower_bound', 'retained_lower_bound', 'retained_fraction'))
    require(type(n) is int and type(y) is int and n >= 40 and 20 <= y <= n-20 and n % 20 == y % 20 == 0
            and type(f) is float and math.isfinite(f) and 0 < f < 1)


def validate_profiles(value, k):
    if value == {'status': 'suppressed_outcome_complement_support'}:
        return
    require(type(value) is dict and set(value) == {'status', 'groups', 'labs_are_clustering_inputs_not_independent_validation'}
            and value['status'] == 'released' and value['labs_are_clustering_inputs_not_independent_validation'] is True
            and type(value['groups']) is list and len(value['groups']) == k)
    for i, group in enumerate(value['groups']):
        require(type(group) is dict and set(group) == {'group', 'known_outcome_people_lower_bound',
            'unadjusted_hospital_death_rate', 'wilson95', 'labs'} and group['group'] == i)
        n = group['known_outcome_people_lower_bound']; rate = group['unadjusted_hospital_death_rate']; ci = group['wilson95']
        require(type(n) is int and n >= 40 and n % 20 == 0 and type(rate) is float and 0 < rate < 1
                and type(ci) is list and len(ci) == 2 and all(type(v) is float and math.isfinite(v) for v in ci)
                and 0 <= ci[0] <= rate <= ci[1] <= 1
                and type(group['labs']) is dict and set(group['labs']) == set(LAB_FIELDS))
        for cell in group['labs'].values():
            if cell == {'status': 'suppressed_observed_support'}:
                continue
            require(type(cell) is dict and set(cell) == {'status', 'q25', 'median', 'q75'} and cell['status'] == 'released'
                    and all(type(cell[x]) is float and math.isfinite(cell[x]) for x in ('q25', 'median', 'q75'))
                    and cell['q25'] <= cell['median'] <= cell['q75'])
    # Field suppression is whole-table, never just its small group.
    for name in LAB_FIELDS:
        require(len({group['labs'][name]['status'] for group in value['groups']}) == 1)


def confidence_sensitivity(sink, test_labels, retained):
    """Same fitted arms on the same retained test people; never a primary gate."""
    import bran_mimic_clinical_outcomes_v3 as outcome
    require(type(retained) is np.ndarray and retained.dtype == np.dtype(bool)
            and retained.shape == test_labels.shape and np.isin(test_labels, (0, 1)).all())
    # Complementary suppression covers each class in both retained/excluded
    # people; no small outcome cell can be recovered from the primary table.
    if min(int(np.sum(mask & (test_labels == label)))
           for mask in (retained, ~retained) for label in (0, 1)) < 20:
        return {'status': 'suppressed_outcome_complement_support'}
    y = test_labels[retained].astype(np.float64)
    predictions = {arm: saved['test_predictions'][retained] for arm, saved in sink['arms'].items()}
    bootstrap = outcome._bootstrap(y, predictions)
    if bootstrap is None:
        return {'status': 'unsupported_bootstrap'}
    draws, valid = bootstrap
    arms = {arm: outcome._available_arm(outcome._point_metrics(y, prediction), draws[arm], valid)
            for arm, prediction in predictions.items()}
    complete_arms = {arm: arms.get(arm, {'status': 'unavailable'}) for arm in outcome.ARM_NAMES}
    contrasts = {}
    for name, (plus, minus) in outcome.CONTRASTS.items():
        contrast = outcome._contrast_report(plus, minus, complete_arms, draws, valid)
        # Same fixed comparisons and intervals, but no sensitivity can promote
        # a primary-failing group or support a new selected-subset claim.
        if contrast['status'] == 'available':
            contrast.pop('utility_gate')
        contrasts[name] = contrast
    return {'status': 'released', 'coverage': _coverage(len(retained), int(retained.sum())),
        'arms': complete_arms, 'contrasts': contrasts, 'accepted_draws': int(valid.sum()),
        'same_fitted_models': True, 'primary_gate_eligible': False,
        'patient_level_output_emitted': False}


def validate_sensitivity(value):
    import bran_mimic_clinical_outcomes_v3 as outcome
    closed = ('suppressed_outcome_complement_support', 'unsupported_bootstrap',
              'not_run_structure_gate_failed', 'unavailable_calibration_support',
              'not_run_outcome_support')
    if type(value) is dict and value.get('status') in closed:
        require(set(value) == {'status'}); return
    require(type(value) is dict and set(value) == {'status', 'coverage', 'arms', 'contrasts',
        'accepted_draws', 'same_fitted_models', 'primary_gate_eligible', 'patient_level_output_emitted'}
        and value['status'] == 'released' and value['same_fitted_models'] is True
        and value['primary_gate_eligible'] is False and value['patient_level_output_emitted'] is False)
    validate_coverage(value['coverage'])
    require(value['coverage']['status'] == 'released')
    # Reuse the exact primary metric/contrast validator, reconstructing the
    # derived gate only internally. It is not released as a sensitivity claim.
    contrasts = {}
    for name, cell in value['contrasts'].items():
        require('utility_gate' not in cell)
        contrasts[name] = dict(cell)
        if cell['status'] == 'available':
            contrasts[name]['utility_gate'] = bool(cell['logloss_improvement_adjusted_95ci'][0] > 0
                                                  and cell['auroc_difference'] >= 0)
    all_available = all(arm['status'] == 'available' for arm in value['arms'].values())
    reconstructed = {'schema': outcome.SCHEMA, 'status': 'ok' if all_available else 'group_branch_unavailable',
        'policy': dict(outcome.POLICY), 'arms': value['arms'], 'contrasts': contrasts,
        'bootstrap': {'requested_draws': outcome.POLICY['bootstrap_draws'],
                      'accepted_auroc_draws': value['accepted_draws']},
        'patient_level_output_emitted': False, 'clinical_utility_established': False,
        'novel_subtype_claim': False, 'external_validation_claim': False}
    require(outcome.validate_report(reconstructed))


@dataclass(repr=False)
class PrivateClinicalPanelV3:
    aggregate: dict
    objects: dict

    def __repr__(self):
        return '<PrivateClinicalPanelV3>'

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)


def fit_family(frame, urgency, family, binding, progress=None):
    import bran_mimic_clinical_outcomes_v3 as outcome
    try:
        check_frame(frame, urgency)
        require(family in FAMILIES and type(binding) is assignment.AssignmentBinding)
        j = FAMILIES.index(family)
        # Unknown outcomes remain in structure. No target value participates in
        # this selection, normalization, GMM fit, K choice or stability test.
        selected = frame['current_membership'][:, j]
        keep = selected & frame['available']
        rows = np.flatnonzero(keep); split = frame['roles'][rows]
        role_indices = {name: np.flatnonzero(split == role).astype(np.int64)
                        for role, name in enumerate(ROLES)}
        counts = [len(role_indices[n]) for n in ROLES]
        base = {'family': family, 'status': 'evaluated', 'coverage': _coverage(int(selected.sum()), len(rows)),
                'known_outcomes_used_for_clustering': False, **FLAGS}
        if any(n < threshold for n, threshold in zip(counts, (80, 40, 40))):
            return PrivateClinicalPanelV3({**base, 'status': 'unsupported_cohort_roles'}, {})
        x = design.fit_transform(frame['values'][rows], frame['observed'][rows],
            frame['age_triplet'][rows], frame['age_kind'][rows], frame['state'][rows], role_indices)
        fitted, groups, models, reports = {}, {}, {}, {}
        for branch, states in (('bran', frame['state'][rows]), ('raw', x.raw_clustering_states)):
            if progress: progress(f'{branch}_structure')
            fit = structure.fit_structure(*(states[role_indices[r]] for r in ROLES))
            report = fit.aggregate
            require(structure.validate_aggregate(report))
            fitted[branch] = fit
            stable = report.get('stability_gate') is True
            active = fit.status == 'supported' and stable
            groups[branch] = fit.predict(states) if active else None
            models[branch] = fit
            reports[branch] = {'structure': report, 'group_arms_admitted': active}
            if active:
                test = split == 2
                reports[branch]['profiles'] = profiles(frame['values'][rows][test], frame['observed'][rows][test],
                    groups[branch][test], frame['labels'][rows, j][test], fit.selected_k)
                # Existing confidence calibration; no held-out label or test
                # observation influences its thresholds.
                if len(role_indices[ROLES[1]]) >= assignment.MIN_CALIBRATION_ROWS:
                    branch_binding = assignment.AssignmentBinding(
                        binding.model_frame_sha256, binding.canonical_schema_units_sha256,
                        binding.normalization_sha256, hashlib.sha256(
                            f'{binding.clinical_only_input_policy_sha256}|V3|{family}|{branch}'.encode()).hexdigest())
                    locked = assignment.calibrate_locked_assignment(fit._scaler, fit._pca, fit._mixture,
                        states[role_indices[ROLES[1]]], binding=branch_binding,
                        source_calibration_participant_disjoint_from_discovery=True)
                    accepted = assignment.assign_locked(locked, states, binding=branch_binding).accepted
                    # The retained guard is intentionally in-memory-only: its
                    # pickle-byte digest is not a portable serialization format.
                    # Persist the fitted rule, exact thresholds and binding;
                    # replay rebuilds the guard on the SAME development rows.
                    models[branch+'_assignment_thresholds'] = {
                        'density_floor': locked._density_floor,
                        'residual_ceiling': locked._reconstruction_residual_ceiling}
                    models[branch+'_binding'] = branch_binding
                    reports[branch]['confidence_coverage'] = _coverage(int(test.sum()), int(accepted[test].sum()))
                    models[branch+'_accepted'] = accepted
                else:
                    reports[branch]['confidence_coverage'] = {'status': 'unavailable_calibration_support'}
            else:
                reports[branch]['profiles'] = {'status': 'not_run_structure_gate_failed'}
                reports[branch]['confidence_coverage'] = {'status': 'not_run_structure_gate_failed'}
        if progress: progress('outcome_utility')
        context = np.column_stack((x.raw_scaled, urgency[rows]))
        outcome_objects = {}
        outcome_binding = hashlib.sha256(
            f'{binding.model_frame_sha256}|{binding.normalization_sha256}|{family}|raw49-urgency10-state192'.encode()).hexdigest()
        utility = outcome.evaluate(context, x.state_design, frame['person_group'][rows], split,
            frame['labels'][rows, j], bran_groups=groups['bran'],
            bran_k=fitted['bran'].selected_k if groups['bran'] is not None else None,
            raw_groups=groups['raw'], raw_k=fitted['raw'].selected_k if groups['raw'] is not None else None,
            private_sink=outcome_objects, design_binding=outcome_binding)
        known_test = (split == 2) & (frame['labels'][rows, j] >= 0)
        for branch in ('bran', 'raw'):
            if not reports[branch]['group_arms_admitted']:
                sensitivity = {'status': 'not_run_structure_gate_failed'}
            elif branch+'_accepted' not in models:
                sensitivity = {'status': 'unavailable_calibration_support'}
            elif utility['status'] not in ('ok', 'group_branch_unavailable'):
                sensitivity = {'status': 'not_run_outcome_support'}
            else:
                sensitivity = confidence_sensitivity(outcome_objects, frame['labels'][rows, j][known_test],
                    models[branch+'_accepted'][known_test])
            validate_sensitivity(sensitivity)
            reports[branch]['confidence_outcome_sensitivity'] = sensitivity
        models.update(source_rows=rows, family=family, outcome_objects=outcome_objects,
            outcome_binding=outcome_binding, raw_design_statistics={
            key: getattr(x, key) for key in ('lab_medians', 'age_mean', 'age_scale',
                'raw_scaler_mean', 'raw_scaler_scale', 'state_scaler_mean', 'state_scaler_scale')})
        result = {**base, 'branches': reports, 'outcome_utility': utility,
            'primary_assignment': 'hard_GMM_includes_uncertain_members',
            'confidence_analysis': 'locked_sensitivity_not_primary_cohort_selection'}
        return PrivateClinicalPanelV3(result, models)
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def validate_report(value):
    import bran_mimic_clinical_outcomes_v3 as outcome
    base = {'family', 'status', 'coverage', 'known_outcomes_used_for_clustering', *FLAGS}
    require(type(value) is dict and value.get('family') in FAMILIES and
            value.get('known_outcomes_used_for_clustering') is False and
            all(value.get(k) is v for k, v in FLAGS.items()))
    validate_coverage(value['coverage'])
    if value['status'] == 'unsupported_cohort_roles':
        require(set(value) == base)
        return
    require(value['status'] == 'evaluated' and set(value) == base | {
        'branches', 'outcome_utility', 'primary_assignment', 'confidence_analysis'}
        and value['primary_assignment'] == 'hard_GMM_includes_uncertain_members'
        and value['confidence_analysis'] == 'locked_sensitivity_not_primary_cohort_selection')
    require(type(value['branches']) is dict and set(value['branches']) == {'bran', 'raw'})
    for report in value['branches'].values():
        require(type(report) is dict and set(report) == {'structure', 'group_arms_admitted', 'profiles',
                'confidence_coverage', 'confidence_outcome_sensitivity'}
                and structure.validate_aggregate(report['structure']))
        active = report['structure'].get('stability_gate') is True and report['structure']['status'] == 'supported'
        require(report['group_arms_admitted'] is active)
        if active:
            validate_profiles(report['profiles'], report['structure']['selected_k'])
            if report['confidence_coverage'] != {'status': 'unavailable_calibration_support'}:
                validate_coverage(report['confidence_coverage'])
        else:
            require(report['profiles'] == {'status': 'not_run_structure_gate_failed'}
                    and report['confidence_coverage'] == {'status': 'not_run_structure_gate_failed'}
                    and report['confidence_outcome_sensitivity'] == {'status': 'not_run_structure_gate_failed'})
        validate_sensitivity(report['confidence_outcome_sensitivity'])
        if report['confidence_outcome_sensitivity']['status'] == 'released':
            require(active and report['confidence_coverage']['status'] != 'unavailable_calibration_support')
            for arm, cell in report['confidence_outcome_sensitivity']['arms'].items():
                require(cell['status'] == value['outcome_utility']['arms'][arm]['status'])
    require(outcome.validate_report(value['outcome_utility']))
    if 'arms' in value['outcome_utility']:
        arms = value['outcome_utility']['arms']
        for branch, names in (
            ('bran', ('context_bran_groups', 'context_state_bran_groups')),
            ('raw', ('context_raw_groups',)),
        ):
            for name in names:
                require((arms[name]['status'] == 'available') is
                        value['branches'][branch]['group_arms_admitted'])


def replay_objects(frame, objects, urgency=None):
    """Private labels/probabilities before/after reload; no GMM/readout refit."""
    import bran_mimic_clinical_outcomes_v3 as outcome
    rows = objects['source_rows']; split = frame['roles'][rows]
    roles = {name: np.flatnonzero(split == i).astype(np.int64) for i, name in enumerate(ROLES)}
    x = design.fit_transform(frame['values'][rows], frame['observed'][rows], frame['age_triplet'][rows],
        frame['age_kind'][rows], frame['state'][rows], roles)
    for key, stored in objects['raw_design_statistics'].items():
        require(np.array_equal(np.asarray(stored), np.asarray(getattr(x, key))))
    result = {}
    for branch, states in (('bran', frame['state'][rows]), ('raw', x.raw_clustering_states)):
        fit = objects[branch]
        if fit._mixture is not None:
            result[branch] = fit._mixture.predict(fit._pca.transform(fit._scaler.transform(states)))
        if branch+'_assignment_thresholds' in objects:
            locked = assignment.calibrate_locked_assignment(fit._scaler, fit._pca, fit._mixture,
                states[roles[ROLES[1]]], binding=objects[branch+'_binding'],
                source_calibration_participant_disjoint_from_discovery=True)
            require(objects[branch+'_assignment_thresholds'] == {
                'density_floor': locked._density_floor,
                'residual_ceiling': locked._reconstruction_residual_ceiling})
            result[branch+'_accepted'] = assignment.assign_locked(locked, states,
                binding=objects[branch+'_binding']).accepted
            require(np.array_equal(result[branch+'_accepted'], objects[branch+'_accepted']))
    sink = objects['outcome_objects']
    require(sink.get('design_binding') == objects['outcome_binding'])
    if sink['arms']:
        require(urgency is not None)
        labels = frame['labels'][rows, FAMILIES.index(objects['family'])]
        test = (split == 2) & (labels >= 0)
        context = np.column_stack((x.raw_scaled[test], urgency[rows][test]))
        state = x.state_design[test]
        designs = {'context': context, 'context_state': np.column_stack((context, state))}
        for branch, names in (('bran', ('context_bran_groups', 'context_state_bran_groups')),
                               ('raw', ('context_raw_groups',))):
            if names[0] not in sink['arms']:
                continue
            groups = objects[branch].predict(frame['state'][rows] if branch == 'bran' else x.raw_clustering_states)
            one = outcome._one_hot(groups[test], objects[branch].selected_k)
            designs[names[0]] = np.column_stack((context, one))
            if branch == 'bran':
                designs[names[1]] = np.column_stack((context, state, one))
        require(set(designs) == set(sink['arms']))
        for arm, saved in sink['arms'].items():
            require(designs[arm].shape[1] == saved['feature_width'])
            probability = outcome._calibrated_probabilities(
                saved['model'].decision_function(designs[arm]), saved['calibration_offset'])
            require(np.array_equal(probability, saved['test_predictions']))
            result['outcome_'+arm] = probability
    return result
