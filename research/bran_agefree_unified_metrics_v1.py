"""Closed aggregate evaluation of authenticated private matched-refit outputs.

No I/O, fitting, or automatic selection. Raw reference arrays must come from
the frozen, independently authenticated reference replay, not from this module.
All radii/inputs/draws remain private. Existing advancement gates are unchanged.
"""
from dataclasses import dataclass
import json

import numpy as np

import bran_agefree_unified_oof_v1 as oof
import bran_distillation_metrics_v1 as metrics
import bran_missingness_stress_metrics_v1 as stress_metrics
import bran_multisource_mask_contract_v1 as gate
import bran_supervised_mask_uncertainty_v1 as uncertainty
import bran_agefree_platelet_evaluation_v1 as platelet
import run_bran_native_rehearsal_v1 as old

ERROR = 'age-free refit aggregate contract rejected'
ROLE_MAP = {'initial': 'retained_bran', 'continued': 'three_source_clinical_transfer',
            'student': 'four_source_clinical_transfer'}
EXTERNAL = ('blood_age', 'retfound_green', 'visionfm_last4', 'dinov3_generic', 'labrador')
REFERENCE_SCREEN = tuple('initial_' + route for route in oof.ROUTES) + (
    'raw_clinical', 'raw_retinal', 'raw_concat', 'late_average') + EXTERNAL
ATLAS_ARMS = ('candidate_both', 'candidate_clinical', 'candidate_retinal', 'retained_both') + EXTERNAL
ATLAS_PAIRS = {**{'candidate_minus_' + ref: ('candidate_both', ref) for ref in ATLAS_ARMS[1:]},
               'retinal_minus_clinical': ('candidate_retinal', 'candidate_clinical')}
FLAGS = {'patient_level_output_emitted': False, 'automatic_promotion': False,
         'externally_validated': False, 'novel_subtype_claimed': False,
         'retinal_coordinates_changed': False, 'fixed_fit_bootstrap': True,
         'original_nature_retfound_included': False, 'longitudinal_ehr_fm_included': False}


def require(value):
    if not value:
        raise ValueError(ERROR) from None


@dataclass(frozen=True, repr=False)
class PrivateEvaluation:
    aggregate: dict
    calibration_radii: dict


def _labels(labels, observed, n):
    require(type(labels) is np.ndarray and labels.dtype.kind == 'f' and labels.shape == (n, 26)
            and type(observed) is np.ndarray and observed.dtype == bool and observed.shape == labels.shape
            and np.isfinite(labels[observed]).all() and np.isin(labels[observed], (0, 1)).all())


def _reference(value, expected_sha256, binding, collected, n):
    require(type(expected_sha256) is str and len(expected_sha256) == 64
            and oof.private_digest(value) == expected_sha256
            and type(value) is dict and set(value) == {
                'binding', 'screening', 'completion', 'completion_available', 'stress'}
            and value['binding'] == binding
            and type(value['screening']) is dict and set(value['screening']) == set(REFERENCE_SCREEN)
            and type(value['completion']) is dict and set(value['completion']) == set(oof.COMPLETION)
            and type(value['completion_available']) is dict and set(value['completion_available']) == set(oof.COMPLETION)
            and type(value['stress']) is dict and set(value['stress']) == set(oof.masking.PATTERNS))
    for name, prediction in value['screening'].items():
        oof._prediction(prediction, (n, 26), np.isfinite(prediction).all(1)[:, None], probability=True)
        if name.startswith('initial_'):
            require(np.array_equal(np.isnan(prediction), np.isnan(collected.screening['control'][name[8:]])))
    for pattern, predictions in value['completion'].items():
        support = value['completion_available'][pattern]
        require(type(support) is np.ndarray and support.dtype == bool and support.shape == (n, 9)
                and np.array_equal(support, collected.completion_available[pattern])
                and type(predictions) is dict and set(predictions) == {
                    'initial_native', 'initial_generative', 'raw_clinical', 'raw_concat'})
        for prediction in predictions.values():
            oof._prediction(prediction, (n, 9), support)
    for pattern, prediction in value['stress'].items():
        oof._prediction(prediction, (n, 26), np.isfinite(prediction).all(1)[:, None], probability=True)
        require(np.array_equal(np.isnan(prediction), np.isnan(collected.stress['control'][pattern])))
    require(np.array_equal(value['stress']['available'], value['screening']['initial_both'], equal_nan=True))


def _atlas_cell(points, draws):
    return {'status': 'supported',
        'arms': {arm: {'auroc': float(points[arm]), 'ci95': metrics.interval(draws[arm])} for arm in ATLAS_ARMS},
        'contrasts': {name: {'delta': float(points[a] - points[b]),
                            'ci95': metrics.interval(draws[a] - draws[b])}
                      for name, (a, b) in ATLAS_PAIRS.items()}}


def _atlas(predictions, labels, masks, folds, names, counts):
    points, draws, cells = {}, {}, {}
    for j, name in enumerate(names):
        valid, y = masks[name], labels[:, j]
        if min(np.sum(valid & (y == 0)), np.sum(valid & (y == 1))) < 20:
            cells[name] = {'status': 'unsupported'}
            continue
        require(all(np.isfinite(predictions[arm][valid, j]).all() for arm in ATLAS_ARMS))
        p = {arm: metrics.base.fold_weighted_auc(y, predictions[arm][:, j], valid, folds) for arm in ATLAS_ARMS}
        d = {arm: metrics.base._weighted_auc_draws(y, predictions[arm][:, j], valid, folds, counts)
             for arm in ATLAS_ARMS}
        required_draws = (*d.values(), *(d[a] - d[b] for a, b in ATLAS_PAIRS.values()))
        if any(np.isfinite(x).sum() < 900 for x in required_draws):
            cells[name] = {'status': 'unsupported'}
            continue
        points[name], draws[name] = p, d
        cells[name] = _atlas_cell(p, d)
    complete = len(points) == len(names)
    macro = None
    if complete:
        p = {arm: np.mean([x[arm] for x in points.values()]) for arm in ATLAS_ARMS}
        d = {arm: np.mean([x[arm] for x in draws.values()], axis=0) for arm in ATLAS_ARMS}
        if all(np.isfinite(x).sum() >= 900 for x in (*d.values(), *(d[a] - d[b] for a, b in ATLAS_PAIRS.values()))):
            macro = _atlas_cell(p, d)
    return {'endpoints': cells, 'macro': macro}


def _groups(target, observed, folds):
    """Training-only quantiles; raw tails stay private until suppression."""
    groups = {key: np.zeros(target.shape, bool) for key in metrics.GROUPS}
    groups['overall'][:] = True
    for fold in range(5):
        te, tr = folds == fold, folds != fold
        for j in range(9):
            train_y = target[tr & observed[:, j], j]
            if len(train_y) < 20:
                continue
            for name, mask in old.origin.previous.tail_masks(train_y, target[te, j]).items():
                groups[name][te, j] = mask
    return groups


def evaluate(collected, reference, *, expected_reference_sha256, labels, label_observed,
             expected_labels_sha256, data, plan, plan_sha256, inner_folds, registry_names,
             endpoint_names, protocol_sha256, source_descriptor_sha256):
    """All OOF routes, both CBC heads, fixed missingness and split calibration.

    Expected private hashes must be supplied by authenticated execution context.
    Deriving them from unchecked inputs is not provenance. This layer cannot
    authenticate files or conclude that a reference replay actually occurred.
    """
    try:
        args = dict(data=data, plan=plan, plan_sha256=plan_sha256, inner_folds=inner_folds,
                    registry_names=registry_names, endpoint_names=endpoint_names, protocol_sha256=protocol_sha256,
                    source_descriptor_sha256=source_descriptor_sha256)
        binding = oof.context_binding(**args); n = len(data['folds']); folds = data['folds']
        oof.jobs.membership.authenticate(plan, plan_sha256, patient_ids=data['patient_ids'], folds=folds, inner_folds=inner_folds)
        require(type(endpoint_names) is tuple and len(set(endpoint_names)) == len(endpoint_names) == 26
                and all(type(x) is str and x for x in endpoint_names)
                and type(registry_names) is tuple and len(set(registry_names)) == len(registry_names) == 59)
        _labels(labels, label_observed, n)
        require(oof.private_digest({'labels': labels, 'observed': label_observed}) == expected_labels_sha256)
        oof.validate_private(collected, expected_binding=binding, n=n)
        _reference(reference, expected_reference_sha256, binding, collected, n)
        names = endpoint_names
        target_slots = tuple(registry_names.index(x) for x in metrics.CBC_FIELDS)
        target = data['clinical'][:, target_slots].copy()
        truth_mask = (data['observed'] & data['eligible'])[:, target_slots].copy()
        require(np.isfinite(target[truth_mask]).all())
        target[~truth_mask] = np.nan
        prediction = {name: {} for name in names}
        for j, name in enumerate(names):
            for arm in metrics.S_ARMS:
                if arm.startswith('continued_'):
                    values = collected.screening['control'][arm[len('continued_'):]]
                elif arm.startswith('student_'):
                    values = collected.screening['candidate'][arm[len('student_'):]]
                else:
                    values = reference['screening'][arm]
                prediction[name][arm] = values[:, j].copy()
        y = {name: labels[:, j] for j, name in enumerate(names)}
        lm = {name: label_observed[:, j] for j, name in enumerate(names)}
        masks = old.screen_masks(prediction, lm, names, y, folds)
        # Keep the predeclared primary evaluation population. Missing external
        # reference predictions do not silently shrink it to favor a model.
        for j, name in enumerate(names):
            require(all(np.isfinite(reference['screening'][arm][masks[name], j]).all() for arm in EXTERNAL))
        counts = old.origin.native.source.ev.paired_counts(folds, draws=1000, seed=91501)
        require(counts.shape == (1000, n))
        for fold in range(5):
            require(np.all(counts[:, folds == fold].sum(1) == np.sum(folds == fold)))
        screen = metrics.screening(prediction, y, masks, folds, names, counts)
        atlas_predictions = {'candidate_' + route: collected.screening['candidate'][route] for route in oof.ROUTES}
        atlas_predictions.update({'retained_both': reference['screening']['initial_both'],
                                  **{arm: reference['screening'][arm] for arm in EXTERNAL}})
        atlas = _atlas(atlas_predictions, labels, masks, folds, names, counts)
        completion, calibrated, radii, abnormal = {}, {}, {}, {}
        cobs = {p: truth_mask & collected.completion_available[p] for p in oof.COMPLETION}
        groups = {p: _groups(target, cobs[p], folds) for p in oof.COMPLETION}
        roles = np.asarray(plan['calibration_roles'], dtype=np.int64)
        score_counts = old.origin.native.source.ev.paired_counts(folds[roles == 1], draws=1000, seed=94701)
        for head in oof.HEADS:
            cpred = {p: {'initial': reference['completion'][p]['initial_' + head],
                'continued': collected.completion['control'][p][head],
                'student': collected.completion['candidate'][p][head],
                'raw_clinical': reference['completion'][p]['raw_clinical'],
                'raw_concat': reference['completion'][p]['raw_concat']} for p in oof.COMPLETION}
            completion[head] = {p: metrics.completion(target, cobs[p], cpred[p],
                old.safe_tail_groups(cobs[p], groups[p]), counts) for p in oof.COMPLETION}
            calibrated[head], radii[head] = uncertainty.evaluate(target, cobs,
                {p: {v: cpred[p][v] for v in metrics.VERSIONS} for p in oof.COMPLETION},
                {p: {g: groups[p][g] for g in ('low', 'middle', 'high')} for p in oof.COMPLETION},
                folds, roles, score_counts)
            abnormal[head] = platelet.evaluate(target, cobs,
                {p: {v: cpred[p][v] for v in metrics.VERSIONS} for p in oof.COMPLETION},
                folds, roles, score_counts, radii[head], adult_mask_N=data['age'] >= 18.)
        stress = {'initial': reference['stress'], 'continued': collected.stress['control'],
                  'student': collected.stress['candidate']}
        missingness = {version: stress_metrics.summarize(value, labels, label_observed, folds, names, counts)
                       for version, value in stress.items()}
        native = completion['native']
        decisions = gate.decisions(screen, {p: native[p] for p in gate.PATTERNS}, missingness,
            {'single_target_hidden': native['single_target_no_retina'], 'whole_cbc_hidden': native['whole_cbc_no_retina']})
        result = {'schema': 'bran-agefree-unified-oof-aggregate-v1', 'role_mapping': dict(ROLE_MAP),
            'screening': screen, 'screening_atlas': atlas, 'completion_by_head': completion,
            'missingness': missingness, 'calibrated_completion_by_head': calibrated,
            'clinical_abnormality_by_head': abnormal,
            'decisions': decisions, 'state_width': 192, 'recorded_conditions': 26,
            'readout_budgets_and_information_matched_across_all_comparators': False,
            'blood_draw_replacement_claimed': False, **FLAGS}
        validate_result(result, names)
        require(oof.context_binding(**args) == binding
                and oof.private_digest(reference) == expected_reference_sha256
                and oof.private_digest({'labels': labels, 'observed': label_observed}) == expected_labels_sha256)
        oof.validate_private(collected, expected_binding=binding, n=n)
        return PrivateEvaluation(result, radii)
    except Exception:
        raise ValueError(ERROR) from None


def _validate_atlas_cell(cell):
    if cell == {'status': 'unsupported'}:
        return
    require(type(cell) is dict and set(cell) == {'status', 'arms', 'contrasts'} and cell['status'] == 'supported'
            and type(cell['arms']) is dict and set(cell['arms']) == set(ATLAS_ARMS)
            and type(cell['contrasts']) is dict and set(cell['contrasts']) == set(ATLAS_PAIRS))
    for arm in cell['arms'].values():
        require(type(arm) is dict and set(arm) == {'auroc', 'ci95'} and metrics.finite(arm['auroc']) and 0 <= arm['auroc'] <= 1)
        ci = arm['ci95']
        require(type(ci) is list and len(ci) == 2 and all(metrics.finite(x) for x in ci) and 0 <= ci[0] <= ci[1] <= 1)
    for name, (a, b) in ATLAS_PAIRS.items():
        contrast = cell['contrasts'][name]
        require(type(contrast) is dict and set(contrast) == {'delta', 'ci95'} and metrics.finite(contrast['delta'])
                and abs(contrast['delta'] - (cell['arms'][a]['auroc'] - cell['arms'][b]['auroc'])) < 1e-10)
        ci = contrast['ci95']
        require(type(ci) is list and len(ci) == 2 and all(metrics.finite(x) for x in ci) and -1 <= ci[0] <= ci[1] <= 1)


def validate_result(result, names):
    """Closed publication-safe schema; private membership hashes/radii forbidden."""
    try:
        require(type(result) is dict and set(result) == {
            'schema', 'role_mapping', 'screening', 'screening_atlas', 'completion_by_head',
            'missingness', 'calibrated_completion_by_head', 'clinical_abnormality_by_head',
            'decisions', 'state_width', 'recorded_conditions',
            'readout_budgets_and_information_matched_across_all_comparators', 'blood_draw_replacement_claimed', *FLAGS}
            and result['schema'] == 'bran-agefree-unified-oof-aggregate-v1'
            and result['role_mapping'] == ROLE_MAP
            and type(result['state_width']) is int and result['state_width'] == 192
            and type(result['recorded_conditions']) is int and result['recorded_conditions'] == 26
            and all(result[k] is v for k, v in FLAGS.items())
            and result['readout_budgets_and_information_matched_across_all_comparators'] is False
            and result['blood_draw_replacement_claimed'] is False)
        json.dumps(result, allow_nan=False)
        require(type(result['completion_by_head']) is dict and set(result['completion_by_head']) == set(oof.HEADS)
                and type(result['calibrated_completion_by_head']) is dict
                and set(result['calibrated_completion_by_head']) == set(oof.HEADS)
                and type(result['clinical_abnormality_by_head']) is dict
                and set(result['clinical_abnormality_by_head']) == set(oof.HEADS))
        for head in oof.HEADS:
            require(type(result['completion_by_head'][head]) is dict
                    and set(result['completion_by_head'][head]) == set(oof.COMPLETION))
            for report in result['completion_by_head'][head].values():
                metrics.validate(result['screening'], report, metrics.decisions(result['screening'], report), names)
            uncertainty.validate_result(result['calibrated_completion_by_head'][head])
            platelet.validate_result(result['clinical_abnormality_by_head'][head])
        require(type(result['missingness']) is dict and set(result['missingness']) == set(metrics.VERSIONS))
        for report in result['missingness'].values():
            stress_metrics.validate_result(report, names)
        native = result['completion_by_head']['native']
        expected = gate.decisions(result['screening'], {p: native[p] for p in gate.PATTERNS}, result['missingness'],
            {'single_target_hidden': native['single_target_no_retina'], 'whole_cbc_hidden': native['whole_cbc_no_retina']})
        require(json.dumps(result['decisions'], sort_keys=True) == json.dumps(expected, sort_keys=True))
        atlas = result['screening_atlas']
        require(type(atlas) is dict and set(atlas) == {'endpoints', 'macro'}
                and type(atlas['endpoints']) is dict and set(atlas['endpoints']) == set(names))
        for cell in atlas['endpoints'].values():
            _validate_atlas_cell(cell)
        if atlas['macro'] is not None:
            require(all(cell['status'] == 'supported' for cell in atlas['endpoints'].values()))
            _validate_atlas_cell(atlas['macro'])
            require(atlas['macro']['status'] == 'supported')
            for arm in ATLAS_ARMS:
                require(abs(atlas['macro']['arms'][arm]['auroc'] - np.mean([
                    cell['arms'][arm]['auroc'] for cell in atlas['endpoints'].values()])) < 1e-10)
        return True
    except Exception:
        raise ValueError(ERROR) from None
