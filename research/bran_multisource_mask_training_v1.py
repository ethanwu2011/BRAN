"""Array-only paired continuation with external CBC rehearsal and masked tasks.

The caller owns source qualification and all I/O.  Both coefficient arms execute
the same paired, masked-task, and source-balanced external forwards; only the
combined loss coefficient differs.
"""
from __future__ import annotations

import copy

import numpy as np

import bran_native_rehearsal_kernel_v1 as native
import bran_raw_teacher_distillation_v1 as common
from bran_clinical_preservation_loss_draft_v1 import clinical_preservation_loss
from bran_native_rehearsal_batches_v1 import sample_batch
from bran_supervised_mask_tasks_v1 import build_masks
from bran_supervised_mask_training_v1 import masked_supervised_losses


ERROR = "bran_multisource_mask_training_v1_failed"


def _fail():
    raise ValueError(ERROR) from None


def _validated(initial, clinical, cm, retinal, rm, age, labels, lm, train, slots,
               seed, combined_weight, steps, batch_size, progress):
    try:
        selected = native.validate(initial, clinical, cm, retinal, rm, age, labels, lm,
                                   train, slots, seed, steps, batch_size, combined_weight)
    except Exception:
        _fail()
    if progress is not None and not callable(progress):
        _fail()
    if type(combined_weight) not in (int, float) or combined_weight not in (0, .5):
        _fail()
    return selected


def adapt(initial, clinical, cm, retinal, rm, age, labels, lm, train, slots, prepared, *,
          seed, combined_weight, steps=1500, batch_size=96, progress=None):
    """Return a finite eval-mode native clone after a fixed paired continuation.

    ``prepared`` remains opaque to this kernel and is validated by
    ``sample_batch``.  Its external source task never supplies retinal or disease
    targets.  The task/external torch forwards run under a restored RNG state so
    their stochasticity cannot perturb the retained paired training stream.
    """
    import torch
    import run_bran_overnight_diagnostic_v1 as base

    train = _validated(initial, clinical, cm, retinal, rm, age, labels, lm, train, slots,
                       seed, combined_weight, steps, batch_size, progress)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model, teacher = copy.deepcopy(initial), copy.deepcopy(initial)
        model.train(); teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
        target_c = torch.tensor(clinical, dtype=torch.float32)
        cmt = torch.tensor(cm, dtype=torch.bool)
        ct = common._clean_for_encode(torch, target_c, cmt)
        target_r = torch.tensor(retinal[:, None, :], dtype=torch.float32)
        rmt = torch.tensor(rm[:, None], dtype=torch.bool)
        rt = common._clean_for_encode(torch, target_r, rmt[..., None])
        at = torch.tensor(age, dtype=torch.float32)
        yt, ymt = torch.tensor(labels, dtype=torch.float32), torch.tensor(lm, dtype=torch.bool)
        physiology_labels = lm & (cm.any(axis=1) | rm)[:, None]
        pos = torch.tensor(common._positive_weights(labels, physiology_labels, train), dtype=torch.float32)
        paired_rng = np.random.default_rng(seed)
        mask_rng = np.random.default_rng(seed + 200000)
        external_rng = np.random.default_rng(seed + 100000)
        with torch.no_grad():
            reference = teacher.encode(ct[train], cmt[train], torch.zeros_like(rt[train]),
                                       torch.zeros_like(rmt[train]), at[train])
            scale = reference.mean[:, 128:192].std(dim=0, unbiased=False).clamp_min(1.)
        for step in range(steps):
            index = paired_rng.choice(train, batch_size, replace=len(train) < batch_size)
            vc0, vr0 = base.masked_route(paired_rng, cm[index], rm[index])
            vc, vr = torch.tensor(vc0, dtype=torch.bool), torch.tensor(vr0[:, None], dtype=torch.bool)
            sc = common._clean_for_encode(torch, ct[index], vc)
            sr = common._clean_for_encode(torch, rt[index], vr[..., None])
            state = model.encode(sc, vc, sr, vr, at[index])
            generative = model.objective(state, at[index], target_c[index], cmt[index], vc,
                target_r[index, 0], rmt[index].expand(-1, 384), vr.expand(-1, 384),
                kl_weight=.001 * min(1., (step + 1) / 300), visible_weight=.1,
                clinical_eligible_mask=cmt[index])["loss"]
            route = common._ROUTE_CYCLE[step % len(common._ROUTE_CYCLE)]
            dc = cmt[index] if route != "retinal" else torch.zeros_like(cmt[index])
            dr = rmt[index] if route != "clinical" else torch.zeros_like(rmt[index])
            disease_state = model.encode(common._clean_for_encode(torch, ct[index], dc), dc,
                common._clean_for_encode(torch, rt[index], dr[..., None]), dr, at[index])
            disease = common._disease_loss(torch, model.screening_joint_head(disease_state.mean),
                yt[index], ymt[index] & ~disease_state.abstain[:, None], pos)
            hc, hm = common._whole_cbc_removed(clinical[index], cm[index], slots)
            hct, hmt = torch.tensor(hc, dtype=torch.float32), torch.tensor(hm, dtype=torch.bool)
            cbc_state = model.encode(common._clean_for_encode(torch, hct, hmt), hmt,
                common._clean_for_encode(torch, rt[index], rmt[index, ..., None]), rmt[index], at[index])
            completion = common._cbc_loss(torch, model.cbc_joint_head(cbc_state.mean),
                target_c[index][:, slots], cmt[index][:, slots], ~cbc_state.abstain)
            with torch.no_grad():
                reference = teacher.encode(sc, vc, torch.zeros_like(rt[index]),
                                           torch.zeros_like(rmt[index]), at[index])
            preservation = clinical_preservation_loss(state.mean, reference.mean, scale,
                                                       reference.clinical_available)
            masks = build_masks(cm[index], rm[index], slots, step=step, rng=mask_rng)
            mode = "whole_cbc" if step % 2 == 0 else "partial_cbc"
            batch = sample_batch(prepared, mode, batch_size, external_rng)
            with torch.random.fork_rng(devices=[]):
                masked_screen, masked_cbc = masked_supervised_losses(
                    model, target_c[index], cmt[index], target_r[index], rmt[index], at[index],
                    yt[index], ymt[index], pos, slots, masks,
                )
                external_cbc = native.external_loss(model, batch, slots)
            loss = (generative + disease + .5 * completion + .1 * preservation
                    + combined_weight * (masked_screen + masked_cbc + external_cbc))
            if not bool(torch.isfinite(loss)):
                _fail()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5., error_if_nonfinite=True)
            optimizer.step()
            if progress is not None and ((step + 1) % 100 == 0 or step + 1 == steps):
                progress(step + 1)
        if not all(bool(value.isfinite().all()) for value in model.state_dict().values()):
            _fail()
        model.eval()
    return model
