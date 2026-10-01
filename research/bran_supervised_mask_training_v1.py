"""Array-only continuation with prospective supervised missingness tasks.

No source access, checkpoint I/O, fitting protocol, or external rehearsal lives
here.  This kernel only clones a supplied native model and performs a bounded
local continuation on caller-supplied arrays.
"""
from __future__ import annotations

import copy

import numpy as np

import bran_native_rehearsal_kernel_v1 as native
import bran_raw_teacher_distillation_v1 as common
from bran_supervised_mask_tasks_v1 import SCREEN_PATTERNS, SupervisedMasks, build_masks


ERROR = "bran_supervised_mask_training_v1_failed"


def _fail():
    raise ValueError(ERROR) from None


def _require(ok):
    if not ok:
        _fail()


def _validated(initial, clinical, cm, retinal, rm, age, labels, lm, train, slots,
               seed, mask_weight, steps, batch_size, progress):
    try:
        selected = native.validate(initial, clinical, cm, retinal, rm, age, labels, lm,
                                   train, slots, seed, steps, batch_size, mask_weight)
    except Exception:
        _fail()
    _require(progress is None or callable(progress))
    # ``native.validate`` intentionally owns the canonical array/model contract.
    # This extra exact-type check makes the paired coefficient unambiguous.
    _require(type(mask_weight) in (int, float) and mask_weight in (0, .5))
    return selected


def _mask_arrays(masks, n, cm, rm, slots):
    """Reject forged/leaky masks before any hidden payload reaches an encoder."""
    _require(type(masks) is SupervisedMasks)
    fields = (
        (masks.screen_clinical_mask, (n, 59)), (masks.screen_retinal_mask, (n,)),
        (masks.screen_available, (n,)), (masks.cbc_clinical_mask, (n, 59)),
        (masks.cbc_retinal_mask, (n,)), (masks.cbc_target_mask, (n, 9)),
        (masks.cbc_available, (n,)),
    )
    for value, shape in fields:
        _require(type(value) is np.ndarray and value.dtype == np.dtype(bool) and value.shape == shape)
    _require(type(masks.screen_pattern) is str and type(masks.cbc_pattern) is str
             and masks.screen_pattern in SCREEN_PATTERNS
             and masks.cbc_pattern in ("partial_cbc", "whole_cbc_no_retina"))
    _require(np.all(masks.screen_clinical_mask <= cm) and np.all(masks.screen_retinal_mask <= rm))
    _require(np.array_equal(masks.screen_available,
                            masks.screen_clinical_mask.any(axis=1) | masks.screen_retinal_mask))
    non_cbc = np.ones(59, dtype=bool); non_cbc[list(slots)] = False
    if masks.screen_pattern.startswith("clinical_drop"):
        _require(np.array_equal(masks.screen_retinal_mask, rm))
    elif masks.screen_pattern == "whole_cbc_hidden":
        _require(not masks.screen_clinical_mask[:, slots].any()
                 and np.array_equal(masks.screen_clinical_mask[:, non_cbc], cm[:, non_cbc])
                 and np.array_equal(masks.screen_retinal_mask, rm))
    elif masks.screen_pattern == "whole_cbc_no_retina":
        _require(not masks.screen_clinical_mask[:, slots].any()
                 and np.array_equal(masks.screen_clinical_mask[:, non_cbc], cm[:, non_cbc])
                 and not masks.screen_retinal_mask.any())
    else:  # all_clinical_hidden
        _require(not masks.screen_clinical_mask.any() and np.array_equal(masks.screen_retinal_mask, rm))
    _require(np.all(masks.cbc_clinical_mask <= cm) and np.all(masks.cbc_retinal_mask <= rm))
    original_cbc = cm[:, slots]
    _require(np.all(masks.cbc_target_mask <= original_cbc)
             and not np.any(masks.cbc_target_mask & masks.cbc_clinical_mask[:, slots]))
    if masks.cbc_pattern == "whole_cbc_no_retina":
        _require(np.array_equal(masks.cbc_target_mask, original_cbc)
                 and not masks.cbc_clinical_mask[:, slots].any()
                 and np.array_equal(masks.cbc_clinical_mask[:, non_cbc], cm[:, non_cbc])
                 and not masks.cbc_retinal_mask.any())
    else:
        _require(np.array_equal(masks.cbc_retinal_mask, rm)
                 and np.array_equal(masks.cbc_clinical_mask[:, non_cbc], cm[:, non_cbc])
                 and np.array_equal(masks.cbc_clinical_mask[:, slots], original_cbc & ~masks.cbc_target_mask))
        count = original_cbc.sum(axis=1)
        target_count = masks.cbc_target_mask.sum(axis=1)
        _require(np.all(target_count[count < 2] == 0)
                 and np.all((target_count[count >= 2] >= 1) & (target_count[count >= 2] < count[count >= 2])))
    _require(np.array_equal(masks.cbc_available,
                            (masks.cbc_clinical_mask.any(axis=1) | masks.cbc_retinal_mask)
                            & masks.cbc_target_mask.any(axis=1)))


def masked_supervised_losses(model, c, cm, r, rm, age, labels, lm, pos, slots, masks):
    """Return screening and CBC masked-task losses from a validated mask plan.

    ``c`` and ``r`` are original payload tensors.  They may contain arbitrary
    values behind false supplied masks: every encoder input is erased with
    ``torch.where`` before encoding.  CBC targets remain the original observed
    values selected by ``masks.cbc_target_mask``.
    """
    import torch

    _require(isinstance(c, torch.Tensor) and isinstance(cm, torch.Tensor)
             and isinstance(r, torch.Tensor) and isinstance(rm, torch.Tensor)
             and isinstance(age, torch.Tensor) and isinstance(labels, torch.Tensor)
             and isinstance(lm, torch.Tensor) and isinstance(pos, torch.Tensor))
    n = c.shape[0] if c.ndim == 2 else -1
    _require(n > 0 and c.shape == (n, 59) and cm.shape == (n, 59)
             and r.shape == (n, 1, 384) and rm.shape == (n, 1)
             and age.shape == (n,) and labels.shape == (n, 26) and lm.shape == (n, 26)
             and pos.shape == (26,) and cm.dtype == torch.bool and rm.dtype == torch.bool
             and lm.dtype == torch.bool)
    _require(type(slots) is tuple and len(slots) == 9 and len(set(slots)) == 9
             and all(type(slot) is int and 0 <= slot < 48 for slot in slots))
    cm_np, rm_np = cm.detach().cpu().numpy(), rm.detach().cpu().numpy()[:, 0]
    _mask_arrays(masks, n, cm_np, rm_np, slots)
    device = c.device
    sc = torch.tensor(masks.screen_clinical_mask, dtype=torch.bool, device=device)
    sr = torch.tensor(masks.screen_retinal_mask[:, None], dtype=torch.bool, device=device)
    cc = torch.tensor(masks.cbc_clinical_mask, dtype=torch.bool, device=device)
    cr = torch.tensor(masks.cbc_retinal_mask[:, None], dtype=torch.bool, device=device)
    target = torch.tensor(masks.cbc_target_mask, dtype=torch.bool, device=device)
    _require(bool(torch.isfinite(age).all())
             and bool(torch.isfinite(labels[lm]).all())
             and bool(((labels[lm] == 0) | (labels[lm] == 1)).all())
             and bool(torch.isfinite(c[:, slots][target]).all()))
    # Non-finite data are allowed behind a false supplied/task mask; every
    # payload that can reach either extra encoder must be finite.
    _require(bool(torch.isfinite(c[sc]).all()) and bool(torch.isfinite(c[cc]).all())
             and bool(torch.isfinite(r[sr[..., 0]]).all()) and bool(torch.isfinite(r[cr[..., 0]]).all()))
    screen_state = model.encode(common._clean_for_encode(torch, c, sc), sc,
                                common._clean_for_encode(torch, r, sr[..., None]), sr, age)
    cbc_state = model.encode(common._clean_for_encode(torch, c, cc), cc,
                             common._clean_for_encode(torch, r, cr[..., None]), cr, age)
    screen_available = torch.tensor(masks.screen_available, dtype=torch.bool, device=device)
    _require(bool(torch.equal(~screen_state.abstain, screen_available)))
    cbc_physiology = torch.tensor(masks.cbc_clinical_mask.any(axis=1) | masks.cbc_retinal_mask,
                                  dtype=torch.bool, device=device)
    _require(bool(torch.equal(~cbc_state.abstain, cbc_physiology)))
    screen_loss = common._disease_loss(
        torch, model.screening_joint_head(screen_state.mean), labels,
        lm & ~screen_state.abstain[:, None], pos,
    )
    cbc_available = torch.tensor(masks.cbc_available, dtype=torch.bool, device=device)
    cbc_loss = common._cbc_loss(torch, model.cbc_joint_head(cbc_state.mean), c[:, slots],
                                target, cbc_available & ~cbc_state.abstain)
    _require(bool(torch.isfinite(screen_loss)) and bool(torch.isfinite(cbc_loss)))
    return screen_loss, cbc_loss


def adapt(initial, clinical, cm, retinal, rm, age, labels, lm, train, slots, *,
          seed, mask_weight, steps=1500, batch_size=96, progress=None):
    """Return an eval-mode finite native clone after fixed paired continuation.

    Both coefficient arms execute the identical paired sampling, generative
    masks, and two extra masked forwards.  ``mask_weight`` is restricted to 0
    (control) or .5 (candidate); no external sample or loss path is present.
    """
    import torch
    import run_bran_overnight_diagnostic_v1 as base
    from bran_clinical_preservation_loss_draft_v1 import clinical_preservation_loss

    train = _validated(initial, clinical, cm, retinal, rm, age, labels, lm, train, slots,
                       seed, mask_weight, steps, batch_size, progress)
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
        paired_rng, mask_rng = np.random.default_rng(seed), np.random.default_rng(seed + 200000)
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
                                     common._clean_for_encode(torch, rt[index], rmt[index, ..., None]),
                                     rmt[index], at[index])
            completion = common._cbc_loss(torch, model.cbc_joint_head(cbc_state.mean),
                target_c[index][:, slots], cmt[index][:, slots], ~cbc_state.abstain)
            with torch.no_grad():
                reference = teacher.encode(sc, vc, torch.zeros_like(rt[index]),
                                           torch.zeros_like(rmt[index]), at[index])
            preservation = clinical_preservation_loss(state.mean, reference.mean, scale,
                                                       reference.clinical_available)
            masks = build_masks(cm[index], rm[index], slots, step=step, rng=mask_rng)
            # Extra task forwards must not perturb the native paired stream
            # (for example through a stochastic layer); their gradients remain
            # attached while this nested RNG state is restored afterwards.
            with torch.random.fork_rng(devices=[]):
                masked_screen, masked_cbc = masked_supervised_losses(
                    model, target_c[index], cmt[index], target_r[index], rmt[index], at[index],
                    yt[index], ymt[index], pos, slots, masks,
                )
            loss = generative + disease + .5 * completion + .1 * preservation + mask_weight * (masked_screen + masked_cbc)
            _require(bool(torch.isfinite(loss)))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5., error_if_nonfinite=True)
            optimizer.step()
            if progress is not None and ((step + 1) % 100 == 0 or step + 1 == steps):
                progress(step + 1)
        _require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()))
        model.eval()
    return model
