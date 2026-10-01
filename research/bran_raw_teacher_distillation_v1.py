"""Array-only, fixed-budget raw-teacher distillation for retained BRAN heads.

This diagnostic kernel neither reads nor writes files.  The caller supplies
outer-training-only, cross-fitted teacher probabilities; no teacher output is
accepted outside an observed, route-available outer-training cell.  The result
is a caller-owned model clone, not a promoted or calibrated predictor.
"""
from __future__ import annotations

import copy

import numpy as np


_INVALID = "raw teacher distillation inputs invalid"
_FAILURE = "raw teacher distillation training failed"
_ROUTES = ("both", "clinical", "retinal")
_ROUTE_CYCLE = ("both", "both", "both", "clinical", "retinal")


def _invalid() -> None:
    raise ValueError(_INVALID)


def _array(value: object, *, shape: tuple[int, ...] | None = None,
           boolean: bool = False) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        _invalid()
    if boolean:
        if value.dtype != np.dtype(bool):
            _invalid()
    elif value.dtype.kind not in "iuf":
        _invalid()
    if shape is not None and value.shape != shape:
        _invalid()
    return value


def _indices(value: object, n: int) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.ndim != 1
            or value.dtype.kind not in "iu" or value.dtype.kind == "b"):
        _invalid()
    if (len(value) == 0 or np.any(value < 0) or np.any(value >= n)
            or len(np.unique(value)) != len(value)):
        _invalid()
    return value.astype(np.intp, copy=False)


def _clean_for_encode(torch, values, mask):
    """Erase unavailable payloads without evaluating multiplication by NaN."""
    return torch.where(mask.bool(), values, torch.zeros_like(values))


def _whole_cbc_removed(values: np.ndarray, mask: np.ndarray,
                       cbc_indices: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    removed_values = values.copy()
    removed_mask = mask.copy()
    removed_values[:, cbc_indices] = 0.0
    removed_mask[:, cbc_indices] = False
    return removed_values, removed_mask


def _positive_weights(labels: np.ndarray, mask: np.ndarray, train: np.ndarray) -> np.ndarray:
    observed = mask[train]
    positive = ((labels[train] == 1) & observed).sum(axis=0)
    negative = ((labels[train] == 0) & observed).sum(axis=0)
    weights = np.ones(26, dtype=np.float32)
    has_positive = positive > 0
    weights[has_positive] = np.clip(negative[has_positive] / positive[has_positive], 1.0, 10.0)
    return weights


def _disease_loss(torch, logits, targets, mask, positive_weight):
    import torch.nn.functional as functional

    valid = mask.bool() & torch.isfinite(targets)
    clean = torch.where(valid, targets, torch.zeros_like(targets))
    item = functional.binary_cross_entropy_with_logits(
        logits, clean, reduction="none", pos_weight=positive_weight
    )
    endpoint_mean = (item * valid.to(item.dtype)).sum(dim=0) / valid.sum(dim=0).clamp_min(1)
    return endpoint_mean.mean()


def _cbc_loss(torch, prediction, targets, target_mask, eligible_rows):
    import torch.nn.functional as functional

    valid = target_mask.bool() & torch.isfinite(targets)
    clean = torch.where(valid, targets, torch.zeros_like(targets))
    item = functional.smooth_l1_loss(prediction, clean, reduction="none")
    row_count = valid.sum(dim=1)
    supported = eligible_rows.bool() & (row_count > 0)
    per_row = (item * valid.to(item.dtype)).sum(dim=1) / row_count.clamp_min(1)
    return per_row[supported].sum() / supported.sum().clamp_min(1)


def _soft_teacher_loss(torch, logits, probabilities, valid):
    import torch.nn.functional as functional

    clean = torch.where(valid, probabilities.detach(), torch.zeros_like(probabilities))
    item = functional.binary_cross_entropy_with_logits(logits, clean, reduction="none")
    return (item * valid.to(item.dtype)).sum() / valid.sum().clamp_min(1)


def _validate_model(initial_model):
    import torch

    if not isinstance(initial_model, torch.nn.Module):
        _invalid()
    config = getattr(initial_model, "config", None)
    screening = getattr(initial_model, "screening_joint_head", None)
    cbc = getattr(initial_model, "cbc_joint_head", None)
    if (getattr(config, "state_dim", None) != 192
            or getattr(config, "clinical_dim", None) != 59
            or getattr(config, "retinal_feature_dim", None) != 384
            or not callable(getattr(initial_model, "encode", None))
            or not callable(getattr(initial_model, "objective", None))
            or not isinstance(screening, torch.nn.Linear)
            or not isinstance(cbc, torch.nn.Linear)
            or screening.bias is None or cbc.bias is None
            or tuple(screening.weight.shape) != (26, 192)
            or tuple(screening.bias.shape) != (26,)
            or tuple(cbc.weight.shape) != (9, 192)
            or tuple(cbc.bias.shape) != (9,)
            or not torch.isfinite(screening.weight).all()
            or not torch.isfinite(screening.bias).all()
            or not torch.isfinite(cbc.weight).all()
            or not torch.isfinite(cbc.bias).all()):
        _invalid()
    parameters = list(initial_model.parameters())
    if not parameters or any(parameter.device.type != "cpu" for parameter in parameters):
        _invalid()


def _teacher_validity(clinical_mask: np.ndarray, retinal_mask: np.ndarray,
                      label_mask: np.ndarray, train: np.ndarray) -> dict[str, np.ndarray]:
    in_train = np.zeros(len(clinical_mask), dtype=bool)
    in_train[train] = True
    clinical = clinical_mask.any(axis=1)
    physiology = {
        "both": clinical | retinal_mask,
        "clinical": clinical,
        "retinal": retinal_mask,
    }
    return {route: in_train[:, None] & label_mask & physiology[route][:, None] for route in _ROUTES}


def _validate_teachers(teacher_probabilities, clinical_mask, retinal_mask, label_mask, train):
    if not isinstance(teacher_probabilities, dict) or set(teacher_probabilities) != set(_ROUTES):
        _invalid()
    n = len(clinical_mask)
    valid = _teacher_validity(clinical_mask, retinal_mask, label_mask, train)
    result: dict[str, np.ndarray] = {}
    for route in _ROUTES:
        values = _array(teacher_probabilities[route], shape=(n, 26))
        # Only outer-training, observed, route-available teacher cells belong
        # to this fit.  Do not validate or inspect held-out teacher payloads.
        fit_values = values[valid[route]]
        if (not np.isfinite(fit_values).all()
                or np.any((fit_values < 0.0) | (fit_values > 1.0))):
            _invalid()
        result[route] = values.astype(np.float32, copy=False)
    return result


def _validate_adapt(initial_model, clinical, clinical_mask, retinal, retinal_mask, age,
                    labels, label_mask, train_indices, cbc_indices, teacher_probabilities,
                    seed, steps, batch_size, distill_weight):
    _validate_model(initial_model)
    n = clinical.shape[0] if isinstance(clinical, np.ndarray) and clinical.ndim == 2 else -1
    clinical = _array(clinical, shape=(n, 59))
    clinical_mask = _array(clinical_mask, shape=(n, 59), boolean=True)
    retinal = _array(retinal, shape=(n, 384))
    retinal_mask = _array(retinal_mask, shape=(n,), boolean=True)
    age = _array(age, shape=(n,))
    labels = _array(labels, shape=(n, 26))
    label_mask = _array(label_mask, shape=(n, 26), boolean=True)
    if (n <= 0 or not np.isfinite(age).all()
            or not np.isfinite(clinical[clinical_mask]).all()
            or not np.isfinite(retinal[retinal_mask]).all()):
        _invalid()
    train = _indices(train_indices, n)
    if np.any(label_mask[train] & (~np.isfinite(labels[train]) | ((labels[train] != 0) & (labels[train] != 1)))):
        _invalid()
    if (not isinstance(cbc_indices, tuple) or len(cbc_indices) != 9
            or any(isinstance(index, bool) or not isinstance(index, (int, np.integer)) for index in cbc_indices)
            or len(set(cbc_indices)) != 9 or any(index < 0 or index >= 48 for index in cbc_indices)):
        _invalid()
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0
            or isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps <= 0
            or isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0
            or isinstance(distill_weight, (bool, np.bool_)) or not isinstance(distill_weight, (int, float, np.number))
            or not np.isfinite(distill_weight) or distill_weight < 0):
        _invalid()
    teachers = _validate_teachers(teacher_probabilities, clinical_mask, retinal_mask, label_mask, train)
    return (clinical, clinical_mask, retinal, retinal_mask, age, labels, label_mask, train,
            tuple(int(index) for index in cbc_indices), teachers, int(seed), int(steps),
            int(batch_size), float(distill_weight))


def adapt(initial_model, clinical59, clinical_mask, retinal384, retinal_mask, age,
          labels, label_mask, train_indices, cbc_indices, teacher_probabilities,
          seed, steps=1500, batch_size=96, distill_weight=1.0):
    """Return a final fixed-budget clone with optional route-matched distillation.

    Existing retained heads are copied unchanged at initialization.  All prior
    candidate terms (disease=1, whole-CBC=.5, private preservation=.1) run for
    every call; only the unweighted detached soft-target BCE is multiplied by
    ``distill_weight``.  A zero weight is therefore the matched continuation.
    """
    import torch
    import run_bran_overnight_diagnostic_v1 as base
    from bran_clinical_preservation_loss_draft_v1 import clinical_preservation_loss

    (clinical, clinical_mask, retinal, retinal_mask, age, labels, label_mask, train,
     cbc_indices, teachers, seed, steps, batch_size, distill_weight) = _validate_adapt(
        initial_model, clinical59, clinical_mask, retinal384, retinal_mask, age,
        labels, label_mask, train_indices, cbc_indices, teacher_probabilities,
        seed, steps, batch_size, distill_weight,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        torch.set_num_threads(2)
        model = copy.deepcopy(initial_model)
        teacher = copy.deepcopy(initial_model)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

        target_ct = torch.tensor(clinical, dtype=torch.float32)
        cmt = torch.tensor(clinical_mask, dtype=torch.bool)
        ct = _clean_for_encode(torch, target_ct, cmt)
        target_rt = torch.tensor(retinal[:, None, :], dtype=torch.float32)
        rmt = torch.tensor(retinal_mask[:, None], dtype=torch.bool)
        rt = _clean_for_encode(torch, target_rt, rmt[..., None])
        at = torch.tensor(age, dtype=torch.float32)
        yt = torch.tensor(labels, dtype=torch.float32)
        ymt = torch.tensor(label_mask, dtype=torch.bool)
        teacher_t = {route: torch.tensor(values, dtype=torch.float32) for route, values in teachers.items()}
        positive_weight = torch.tensor(_positive_weights(labels, label_mask, train), dtype=torch.float32)
        rng = np.random.default_rng(seed)

        with torch.no_grad():
            full_teacher = teacher.encode(ct[train], cmt[train], torch.zeros_like(rt[train]),
                                          torch.zeros_like(rmt[train]), at[train])
            teacher_scale = full_teacher.mean[:, 128:192].std(dim=0, unbiased=False).clamp_min(1.0)

        for step in range(steps):
            index = rng.choice(train, batch_size, replace=len(train) < batch_size)
            visible_clinical, visible_retinal = base.masked_route(rng, clinical_mask[index], retinal_mask[index])
            vc = torch.tensor(visible_clinical, dtype=torch.bool)
            vr = torch.tensor(visible_retinal[:, None], dtype=torch.bool)
            student_clinical = _clean_for_encode(torch, ct[index], vc)
            student_retinal = _clean_for_encode(torch, rt[index], vr[..., None])
            state = model.encode(student_clinical, vc, student_retinal, vr, at[index])
            generative = model.objective(
                state, at[index], target_ct[index], cmt[index], vc,
                target_rt[index, 0], rmt[index].expand(-1, 384), vr.expand(-1, 384),
                kl_weight=0.001 * min(1.0, (step + 1) / 300), visible_weight=0.1,
                clinical_eligible_mask=cmt[index],
            )["loss"]

            route = _ROUTE_CYCLE[step % len(_ROUTE_CYCLE)]
            full_cmask = cmt[index] if route != "retinal" else torch.zeros_like(cmt[index])
            full_rmask = rmt[index] if route != "clinical" else torch.zeros_like(rmt[index])
            disease_state = model.encode(
                _clean_for_encode(torch, ct[index], full_cmask), full_cmask,
                _clean_for_encode(torch, rt[index], full_rmask[..., None]), full_rmask, at[index],
            )
            disease = _disease_loss(
                torch, model.screening_joint_head(disease_state.mean), yt[index], ymt[index], positive_weight
            )
            route_physiology = full_cmask.any(dim=1) | full_rmask.reshape(len(full_rmask), -1).any(dim=1)
            route_valid = (ymt[index] & route_physiology[:, None]
                           & torch.isfinite(teacher_t[route][index]))
            distillation = _soft_teacher_loss(
                torch, model.screening_joint_head(disease_state.mean), teacher_t[route][index], route_valid
            )

            cbc_values, cbc_mask = _whole_cbc_removed(clinical[index], clinical_mask[index], cbc_indices)
            cbc_values_t = torch.tensor(cbc_values, dtype=torch.float32)
            cbc_mask_t = torch.tensor(cbc_mask, dtype=torch.bool)
            cbc_state = model.encode(
                _clean_for_encode(torch, cbc_values_t, cbc_mask_t), cbc_mask_t,
                _clean_for_encode(torch, rt[index], rmt[index, ..., None]), rmt[index], at[index],
            )
            completion = _cbc_loss(
                torch, model.cbc_joint_head(cbc_state.mean), target_ct[index][:, cbc_indices],
                cmt[index][:, cbc_indices],
                (cbc_state.clinical_available | cbc_state.retinal_available) & ~cbc_state.abstain,
            )
            with torch.no_grad():
                teacher_state = teacher.encode(student_clinical, vc, torch.zeros_like(rt[index]),
                                               torch.zeros_like(rmt[index]), at[index])
            preservation = clinical_preservation_loss(
                state.mean, teacher_state.mean, teacher_scale, teacher_state.clinical_available
            )
            loss = generative + disease + 0.5 * completion + 0.1 * preservation + distill_weight * distillation
            if not torch.isfinite(loss):
                raise ValueError(_FAILURE)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError(_FAILURE)
    return model


def _validate_crossfit(x_clinical, x_retinal, labels, label_mask, train_indices,
                       inner_fold_ids, seed):
    clinical = _array(x_clinical)
    retinal = _array(x_retinal)
    if clinical.ndim != 2 or retinal.ndim != 2 or clinical.shape[1] == 0 or retinal.shape[1] == 0:
        _invalid()
    n = clinical.shape[0]
    labels = _array(labels, shape=(n, 26))
    label_mask = _array(label_mask, shape=(n, 26), boolean=True)
    if retinal.shape[0] != n or not np.isfinite(clinical).all() or not np.isfinite(retinal).all():
        _invalid()
    train = _indices(train_indices, n)
    if np.any(label_mask[train] & (~np.isfinite(labels[train]) | ((labels[train] != 0) & (labels[train] != 1)))):
        _invalid()
    folds = np.asarray(inner_fold_ids)
    if (folds.shape != (n,) or folds.dtype.kind not in "iu" or folds.dtype.kind == "b"
            or set(np.unique(folds[train])) != {0, 1, 2, 3, 4}
            or np.any(folds[~np.isin(np.arange(n), train)] != -1)):
        _invalid()
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        _invalid()
    return clinical, retinal, labels, label_mask, train, folds.astype(np.intp, copy=False), int(seed)


def _crossfit_route(features, labels, label_mask, train, folds, seed, fit_fn):
    n = len(features)
    output = np.full((n, 26), np.nan, dtype=float)
    for endpoint in range(26):
        for fold in range(5):
            held = train[folds[train] == fold]
            inner_train = train[folds[train] != fold]
            # Blank every non-fitting label before the shared fixed recipe sees it.
            local_y = np.zeros(n, dtype=float)
            local_observed = np.zeros(n, dtype=bool)
            local_y[inner_train] = labels[inner_train, endpoint]
            local_observed[inner_train] = label_mask[inner_train, endpoint]
            try:
                probability, _ = fit_fn(
                    features, local_y, local_observed, inner_train, held, folds,
                    "extra_trees", seed + fold,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(_FAILURE) from error
            probability = np.asarray(probability, dtype=float)
            if (probability.shape != (len(held),) or not np.isfinite(probability).all()
                    or np.any((probability < 0.0) | (probability > 1.0))):
                raise ValueError(_FAILURE)
            observed_held = label_mask[held, endpoint]
            output[held[observed_held], endpoint] = probability[observed_held]
    return output


def crossfit_teachers(x_clinical, x_retinal, labels, label_mask, train_indices,
                      inner_fold_ids, seed, fit_fn=None):
    """Cross-fit fixed ExtraTrees teachers strictly inside the supplied outer train.

    ``x_clinical`` and ``x_retinal`` must already be finite, outer-train-normalized
    raw designs.  Returned arrays are OOF only: every non-observed or non-training
    cell is NaN.  The optional ``fit_fn`` exists solely for synthetic tests and
    must have the positional contract of ``matched.fit_predict``.
    """
    if fit_fn is None:
        from bran_matched_screening_kernel_v1 import fit_predict as fit_fn
    if not callable(fit_fn):
        _invalid()
    clinical, retinal, labels, label_mask, train, folds, seed = _validate_crossfit(
        x_clinical, x_retinal, labels, label_mask, train_indices, inner_fold_ids, seed
    )
    clinical_output = _crossfit_route(clinical, labels, label_mask, train, folds, seed, fit_fn)
    retinal_output = _crossfit_route(retinal, labels, label_mask, train, folds, seed, fit_fn)
    both = 0.5 * (clinical_output + retinal_output)
    if np.any(~np.isfinite(both[~np.isnan(clinical_output)])):
        raise ValueError(_FAILURE)
    return {"both": both, "clinical": clinical_output, "retinal": retinal_output}
