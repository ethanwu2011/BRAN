"""Array-only, paired fixed-budget adaptation for BRAN screening heads.

The caller owns every array and the returned model.  This module performs no
I/O and does not inspect labels outside ``train_indices`` when fitting.
"""
from __future__ import annotations

import copy

import numpy as np


_INVALID = "screening joint kernel inputs invalid"
_FAILURE = "screening joint kernel training failed"
_ROUTE_CYCLE = ("both", "both", "both", "clinical", "retinal")


def _invalid() -> None:
    raise ValueError(_INVALID)


def _array(value: object, *, shape: tuple[int, ...] | None = None, boolean: bool = False) -> np.ndarray:
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


def _whole_cbc_removed(values: np.ndarray, mask: np.ndarray,
                       cbc_indices: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Erase all CBC values and indicators before a completion encode."""
    removed_values = values.copy()
    removed_mask = mask.copy()
    removed_values[:, cbc_indices] = 0.0
    removed_mask[:, cbc_indices] = False
    return removed_values, removed_mask


def _validate(initial_model, clinical, clinical_mask, retinal, retinal_mask, age,
              labels, label_mask, train_indices, cbc_indices, seed, steps, batch_size, candidate):
    import torch

    if not isinstance(initial_model, torch.nn.Module):
        _invalid()
    config = getattr(initial_model, "config", None)
    if (getattr(config, "state_dim", None) != 192 or getattr(config, "clinical_dim", None) != 59
            or getattr(config, "retinal_feature_dim", None) != 384
            or not callable(getattr(initial_model, "encode", None))
            or not callable(getattr(initial_model, "objective", None))):
        _invalid()
    parameters = list(initial_model.parameters())
    if not parameters or any(parameter.device.type != "cpu" for parameter in parameters):
        _invalid()
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
            or not np.isfinite(retinal[retinal_mask]).all()
            or np.any(label_mask & (~np.isfinite(labels) | ((labels != 0) & (labels != 1))))):
        _invalid()
    if not isinstance(train_indices, np.ndarray) or train_indices.ndim != 1 or train_indices.dtype.kind not in "iu":
        _invalid()
    if (len(train_indices) == 0 or np.any(train_indices < 0) or np.any(train_indices >= n)
            or len(np.unique(train_indices)) != len(train_indices)):
        _invalid()
    if (not isinstance(cbc_indices, tuple) or len(cbc_indices) != 9
            or any(isinstance(index, bool) or not isinstance(index, (int, np.integer)) for index in cbc_indices)
            or len(set(cbc_indices)) != 9 or any(index < 0 or index >= 48 for index in cbc_indices)):
        _invalid()
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0
            or isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps <= 0
            or isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0
            or not isinstance(candidate, (bool, np.bool_))):
        _invalid()
    return (clinical, clinical_mask, retinal, retinal_mask, age, labels, label_mask,
            train_indices.astype(np.intp, copy=False), tuple(int(index) for index in cbc_indices),
            int(seed), int(steps), int(batch_size), bool(candidate))


def _positive_weights(labels: np.ndarray, mask: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Endpoint weights use only observed labels on the outer training rows."""
    observed = mask[train]
    positive = ((labels[train] == 1) & observed).sum(axis=0)
    negative = ((labels[train] == 0) & observed).sum(axis=0)
    weights = np.ones(labels.shape[1], dtype=np.float32)
    present = positive > 0
    weights[present] = np.clip(negative[present] / positive[present], 1.0, 10.0)
    return weights


def _disease_loss(torch, logits, targets, mask, positive_weight):
    import torch.nn.functional as functional

    valid = mask.bool() & torch.isfinite(targets)
    clean = torch.zeros_like(targets)
    clean = clean.masked_scatter(valid, targets.masked_select(valid))
    raw = functional.binary_cross_entropy_with_logits(
        logits, clean, reduction="none", pos_weight=positive_weight
    )
    endpoint_mean = (raw * valid.to(raw.dtype)).sum(dim=0) / valid.sum(dim=0).clamp_min(1)
    return endpoint_mean.mean()


def _cbc_loss(torch, prediction, targets, target_mask, eligible_rows):
    import torch.nn.functional as functional

    valid = target_mask.bool() & torch.isfinite(targets)
    clean = torch.zeros_like(targets)
    clean = clean.masked_scatter(valid, targets.masked_select(valid))
    item = functional.smooth_l1_loss(prediction, clean, reduction="none")
    per_row_count = valid.sum(dim=1)
    supported = eligible_rows.bool() & (per_row_count > 0)
    per_row = (item * valid.to(item.dtype)).sum(dim=1) / per_row_count.clamp_min(1)
    return per_row[supported].sum() / supported.sum().clamp_min(1)


def _clean_for_encode(torch, values, mask):
    """Zero masked payloads before every encode without evaluating NaN * 0."""
    return torch.where(mask.bool(), values, torch.zeros_like(values))


def adapt(initial_model, clinical59, clinical_mask, retinal384, retinal_mask, age,
          labels, label_mask, train_indices, cbc_indices, seed, steps=1500,
          batch_size=96, candidate=False):
    """Clone and adapt a V2 model for equal-budget control or candidate training.

    The retained heads read only the 192-wide posterior mean.  Candidate-only
    coefficients are disease=1.0, CBC=0.5, and clinical preservation=0.1;
    control evaluates the same paths at coefficient zero.
    """
    import torch
    import run_bran_overnight_diagnostic_v1 as base
    from bran_clinical_preservation_loss_draft_v1 import clinical_preservation_loss

    (clinical, clinical_mask, retinal, retinal_mask, age, labels, label_mask,
     train, cbc_indices, seed, steps, batch_size, candidate) = _validate(
        initial_model, clinical59, clinical_mask, retinal384, retinal_mask, age,
        labels, label_mask, train_indices, cbc_indices, seed, steps, batch_size, candidate
    )
    # Forking makes a control and candidate call from the same initial model and
    # seed begin with identical added heads and stochastic objective draws.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        torch.set_num_threads(2)
        model = copy.deepcopy(initial_model)
        teacher = copy.deepcopy(initial_model)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        model.screening_joint_head = torch.nn.Linear(192, 26)
        model.cbc_joint_head = torch.nn.Linear(192, 9)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

        # Targets remain separate.  Every encoder receives an explicitly
        # cleaned tensor and its corresponding availability flags.
        target_ct = torch.tensor(clinical, dtype=torch.float32)
        cmt = torch.tensor(clinical_mask, dtype=torch.bool)
        ct = _clean_for_encode(torch, target_ct, cmt)
        target_rt = torch.tensor(retinal[:, None, :], dtype=torch.float32)
        rmt = torch.tensor(retinal_mask[:, None], dtype=torch.bool)
        rt = _clean_for_encode(torch, target_rt, rmt[..., None])
        at = torch.tensor(age, dtype=torch.float32)
        yt = torch.tensor(labels, dtype=torch.float32)
        ymt = torch.tensor(label_mask, dtype=torch.bool)
        positive_weight = torch.tensor(_positive_weights(labels, label_mask, train), dtype=torch.float32)
        rng = np.random.default_rng(seed)

        # Training-only full-clinical teacher scale; no teacher operation carries gradients.
        with torch.no_grad():
            full_teacher = teacher.encode(ct[train], cmt[train], torch.zeros_like(rt[train]),
                                          torch.zeros_like(rmt[train]), at[train])
            teacher_scale = full_teacher.mean[:, 128:192].std(dim=0, unbiased=False).clamp_min(1.0)

        for step in range(steps):
            index = rng.choice(train, batch_size, replace=len(train) < batch_size)
            visible_clinical, visible_retinal_1d = base.masked_route(
                rng, clinical_mask[index], retinal_mask[index]
            )
            vc = torch.tensor(visible_clinical, dtype=torch.bool)
            vr = torch.tensor(visible_retinal_1d[:, None], dtype=torch.bool)
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
                _clean_for_encode(torch, rt[index], full_rmask[..., None]), full_rmask, at[index]
            )
            disease = _disease_loss(
                torch, model.screening_joint_head(disease_state.mean), yt[index], ymt[index], positive_weight
            )

            # Completion hides every CBC value and indicator while preserving
            # all non-CBC clinical context and any retinal evidence.
            cbc_values, cbc_mask = _whole_cbc_removed(clinical[index], clinical_mask[index], cbc_indices)
            cbc_values_t = torch.tensor(cbc_values, dtype=torch.float32)
            cbc_mask_t = torch.tensor(cbc_mask, dtype=torch.bool)
            cbc_state = model.encode(
                _clean_for_encode(torch, cbc_values_t, cbc_mask_t), cbc_mask_t,
                _clean_for_encode(torch, rt[index], rmt[index, ..., None]), rmt[index], at[index]
            )
            cbc_targets = target_ct[index][:, cbc_indices]
            cbc_target_mask = cmt[index][:, cbc_indices]
            completion = _cbc_loss(
                torch, model.cbc_joint_head(cbc_state.mean), cbc_targets, cbc_target_mask,
                (cbc_state.clinical_available | cbc_state.retinal_available) & ~cbc_state.abstain,
            )

            with torch.no_grad():
                teacher_state = teacher.encode(student_clinical, vc, torch.zeros_like(rt[index]),
                                               torch.zeros_like(rmt[index]), at[index])
            preservation = clinical_preservation_loss(
                state.mean, teacher_state.mean, teacher_scale, teacher_state.clinical_available
            )
            extra = disease + 0.5 * completion + 0.1 * preservation
            loss = generative + (extra if candidate else 0.0 * extra)
            if not torch.isfinite(loss):
                raise ValueError(_FAILURE)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError(_FAILURE)
    return model
