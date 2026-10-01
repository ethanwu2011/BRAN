"""Private original-coordinate reference replay for matched age-free refits."""
from __future__ import annotations

import numpy as np
import torch

import bran_matched_screening_kernel_v1 as matched
import bran_missingness_stress_v1 as masking
import bran_named_fm_auc_v1 as fm
import bran_native_cbc_decoders_v1 as cbc
import bran_native_screening_kernel_v1 as native
import bran_agefree_unified_jobs_v1 as jobs
import bran_agefree_unified_metrics_v1 as metrics
import bran_agefree_unified_oof_v1 as oof
import bran_agefree_unified_training_v1 as training
import run_bran_blood_learning_curve_v1 as blood
import bran_multisource_mask_experiment_v1 as source


ERROR = 'age-free unified reference rejected'


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def _snapshot(model):
    training._validate_model(model); require(not model.training)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    require(state and all(value.device.type == 'cpu' and bool(torch.isfinite(value).all())
                          for value in state.values()))
    return state


def _unchanged(model, state):
    require(not model.training and set(model.state_dict()) == set(state)
            and all(value.device.type == 'cpu' and bool(torch.isfinite(value).all())
                    and torch.equal(value, state[name]) for name, value in model.state_dict().items()))


def _inputs(data, labels, observed, expected_labels, features, expected_features):
    n = len(data['folds'])
    require(type(labels) is np.ndarray and labels.dtype.kind == 'f' and labels.shape == (n, 26)
            and type(observed) is np.ndarray and observed.dtype == bool and observed.shape == labels.shape
            and np.isfinite(labels[observed]).all() and np.isin(labels[observed], (0, 1)).all()
            and oof.private_digest({'labels': labels, 'observed': observed}) == expected_labels
            and type(features) is dict and set(features) == set(fm.WIDTHS)
            and oof.private_digest(features) == expected_features)
    for name, width in fm.WIDTHS.items():
        value = features[name]
        require(type(value) is np.ndarray and value.dtype.kind == 'f' and value.shape == (n, width)
                and np.isfinite(value).all())
    return n


def _cbc(model, c, cm, r, rm, age, names, transform, pattern):
    base = 'single_target_hidden' if pattern.startswith('single_target') else 'whole_cbc_hidden'
    noeye = pattern.endswith('no_retina')
    rr, rrm, route = (np.zeros_like(r), np.zeros_like(rm), 'clinical') if noeye else (r, rm, 'both')
    values, support = cbc.infer(model, c, cm, rr, rrm, age, names,
                                transform.clinical_median, transform.clinical_iqr,
                                pattern=base, route=route)
    return values, support


def build(*, data, plan, plan_sha256, inner_folds, registry_names, endpoint_names,
          protocol_sha256, source_descriptor_sha256, labels, label_observed, expected_labels_sha256,
          foundation_features, expected_foundation_features_sha256,
          initial_provider, reference_checker):
    """Produce the exact private reference schema consumed by refit metrics."""
    try:
        args = dict(data=data, plan=plan, plan_sha256=plan_sha256, inner_folds=inner_folds,
                    registry_names=registry_names, endpoint_names=endpoint_names, protocol_sha256=protocol_sha256,
                    source_descriptor_sha256=source_descriptor_sha256)
        binding = oof.context_binding(**args); n = _inputs(data, labels, label_observed,
            expected_labels_sha256, foundation_features, expected_foundation_features_sha256)
        require(type(initial_provider) is not type(None) and callable(initial_provider)
                and callable(reference_checker))
        jobs.membership.authenticate(plan, plan_sha256, patient_ids=data['patient_ids'],
                                     folds=data['folds'], inner_folds=inner_folds)
        screen = {name: np.full((n, 26), np.nan) for name in metrics.REFERENCE_SCREEN}
        completion = {p: {name: np.full((n, 9), np.nan) for name in
                          ('initial_native', 'initial_generative', 'raw_clinical', 'raw_concat')}
                      for p in oof.COMPLETION}
        support = {p: np.zeros((n, 9), bool) for p in oof.COMPLETION}
        stress = {p: np.full((n, 26), np.nan) for p in masking.PATTERNS}
        seen = np.zeros(n, bool); slots = tuple(registry_names.index(x) for x in jobs.CBC_FIELDS)
        for fold in range(5):
            selected, prepared, _, _ = jobs.context(data, plan, plan_sha256, 'outer' + str(fold),
                                                   inner_folds, registry_names, endpoint_names)
            tr, te = np.asarray(selected['fit_indices'], dtype=np.int64), np.flatnonzero(data['folds'] == fold)
            require(np.array_equal(te, np.flatnonzero(data['folds'] == fold)) and not seen[te].any())
            x = prepared.paired
            c, cm, r, rm, age = x.clinical, x.clinical_mask, x.retinal, x.retinal_present, x.age
            normalizers = oof.private_digest(jobs.normalizers(prepared.transform))
            model = initial_provider(fold, prepared.transform)
            require(oof.context_binding(**args) == binding
                    and oof.private_digest({'labels': labels, 'observed': label_observed}) == expected_labels_sha256
                    and oof.private_digest(foundation_features) == expected_foundation_features_sha256
                    and oof.private_digest(jobs.normalizers(prepared.transform)) == normalizers)
            state = _snapshot(model)
            initial = native.predict_native(model, c, cm, r, rm, age)
            for route in oof.ROUTES: screen['initial_' + route][te] = initial[route][te]
            raw = {'raw_clinical': np.c_[c, cm, age], 'raw_retinal': np.c_[r, rm, age],
                   'raw_concat': np.c_[c, cm, r, rm, age]}
            inner = np.asarray(inner_folds[fold], dtype=np.int64)
            for j in range(26):
                for name, values in raw.items():
                    screen[name][te, j], _ = matched.fit_predict(values, labels[:, j], label_observed[:, j],
                        tr, te, inner, 'extra_trees', 92381 + fold)
                screen['late_average'][te, j] = matched.late_fusion_average(
                    screen['raw_clinical'][te, j], screen['raw_retinal'][te, j])
            xblood = blood.blood_design(c, cm, age)
            for j in range(26):
                screen['blood_age'][te, j], _ = blood.fit_predict(xblood, labels[:, j], label_observed[:, j],
                    tr, te, inner[tr])
            for variant, embedding in foundation_features.items():
                predicted, _ = fm.fit_probe_fold(embedding, data['age'], tr, te, inner[tr],
                    {endpoint_names[j]: labels[:, j] for j in range(26)},
                    {endpoint_names[j]: label_observed[:, j] for j in range(26)},
                    variant=variant, source_codes=endpoint_names)
                for j, name in enumerate(endpoint_names): screen[variant][te, j] = predicted[name]
            target = data['clinical'][:, slots]; observed_cbc = (data['observed'] & data['eligible'])[:, slots]
            for pattern in oof.COMPLETION:
                values, valid = _cbc(model, c, cm, r, rm, age, registry_names, prepared.transform, pattern)
                support[pattern][te] = valid[te]
                for head, name in (('native', 'initial_native'), ('generative', 'initial_generative')):
                    completion[pattern][name][te] = values[head][te]
                base = 'single_target_hidden' if pattern.startswith('single_target') else 'whole_cbc_hidden'
                noeye = pattern.endswith('no_retina')
                visible_r, visible_rm = (np.zeros_like(r), np.zeros_like(rm)) if noeye else (r, rm)
                for j in range(9):
                    hc, hm = source.origin.native.source.mask_inputs(c, cm, slots, base, slots[j])
                    valid_j = observed_cbc[:, j] & valid[:, j]
                    for name, design in {'raw_clinical': np.c_[hc, hm, age],
                                         'raw_concat': np.c_[hc, hm, visible_r, visible_rm, age]}.items():
                        output = source.origin.native.source.ev.fixed_cbc_probe(design, target[:, j], valid_j, tr, te)
                        completion[pattern][name][te, j] = output
            with torch.inference_mode():
                for pattern in masking.PATTERNS:
                    masked = masking.remove_inputs(c, cm, r, rm, slots, pattern)
                    masking.assert_no_input_leak(masked, cm, rm, slots, pattern)
                    stress[pattern][te] = native.predict_native(model, masked.clinical[te], masked.clinical_mask[te],
                        masked.retinal[te], masked.retinal_mask[te], age[te])['both']
            _unchanged(model, state)
            require(oof.private_digest(jobs.normalizers(prepared.transform)) == normalizers)
            seen[te] = True
        require(seen.all() and oof.context_binding(**args) == binding
                and oof.private_digest({'labels': labels, 'observed': label_observed}) == expected_labels_sha256
                and oof.private_digest(foundation_features) == expected_foundation_features_sha256)
        for p in oof.COMPLETION:
            for values in completion[p].values(): values[~support[p]] = np.nan
        reference = {'binding': binding, 'screening': screen, 'completion': completion,
                     'completion_available': support, 'stress': stress}
        # Reuse the exact closed downstream reference validator without needing
        # a candidate OOF payload: its shape/availability invariants are checked
        # locally below before the caller's retained-reference canary callback.
        for value in screen.values(): oof._prediction(value, (n, 26), np.isfinite(value).all(1)[:, None], probability=True)
        for p in oof.COMPLETION:
            for value in completion[p].values(): oof._prediction(value, (n, 9), support[p])
        for value in stress.values(): oof._prediction(value, (n, 26), np.isfinite(value).all(1)[:, None], probability=True)
        require(np.array_equal(stress['available'], screen['initial_both'], equal_nan=True))
        oof._seal(reference)
        reference_hash = oof.private_digest(reference)
        require(reference_checker(reference) is True and oof.private_digest(reference) == reference_hash)
        require(oof.context_binding(**args) == binding
                and oof.private_digest({'labels': labels, 'observed': label_observed}) == expected_labels_sha256
                and oof.private_digest(foundation_features) == expected_foundation_features_sha256)
        return reference
    except Exception:
        raise ValueError(ERROR) from None
