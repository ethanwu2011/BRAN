"""Fixed-budget native-head continuation with optional external CBC rehearsal.

Array-only CPU training. No I/O, source admission, selection or clinical claim.
Both arms retain the initial encoder and native heads; only the external loss
coefficient differs. External data never supply retinal or disease targets.
"""
import copy

import numpy as np

import bran_raw_teacher_distillation_v1 as common


def require(ok):
    if not ok:
        raise ValueError("native_rehearsal_contract_failed")


def validate(initial, clinical, cm, retinal, rm, age, labels, lm, train, slots,
             seed, steps, batch_size, external_weight):
    common._validate_model(initial)
    n = len(clinical)
    for value, shape, boolean in ((clinical, (n, 59), False), (cm, (n, 59), True),
            (retinal, (n, 384), False), (rm, (n,), True), (age, (n,), False),
            (labels, (n, 26), False), (lm, (n, 26), True)):
        common._array(value, shape=shape, boolean=boolean)
    train = common._indices(train, n)
    require(np.isfinite(clinical[cm]).all() and np.isfinite(retinal[rm]).all() and np.isfinite(age).all())
    require(np.all(~lm[train] | (np.isfinite(labels[train]) & ((labels[train] == 0) | (labels[train] == 1)))))
    require(isinstance(slots, tuple) and len(slots) == len(set(slots)) == 9
            and all(type(j) is int and 0 <= j < 48 for j in slots))
    require(all(type(v) is int and v > 0 for v in (seed, steps, batch_size)))
    require(type(external_weight) in (int, float) and external_weight in (0., .5))
    return train


def external_loss(model, batch, slots):
    """Native 192→9 head on already-erased, chemistry/partial-CBC inputs."""
    import torch
    keys = {"clinical", "clinical_mask", "age", "target_cbc", "target_mask"}
    require(type(batch) is dict and set(batch) == keys)
    n = len(batch["age"])
    for key, shape, boolean in (("clinical", (n, 59), False), ("clinical_mask", (n, 59), True),
            ("age", (n,), False), ("target_cbc", (n, 9), False), ("target_mask", (n, 9), True)):
        common._array(batch[key], shape=shape, boolean=boolean)
    require(n > 0 and np.isfinite(batch["clinical"]).all() and np.isfinite(batch["age"]).all()
            and np.isfinite(batch["target_cbc"]).all())
    require(isinstance(slots, tuple) and len(slots) == len(set(slots)) == 9
            and all(type(j) is int and 0 <= j < 48 for j in slots))
    require(not np.any(batch["clinical_mask"][:, slots] & batch["target_mask"])
            and not np.any(batch["clinical"][:, slots][batch["target_mask"]]))
    require(not np.any(batch["clinical"][~batch["clinical_mask"]])
            and batch["clinical_mask"].any(1).all() and batch["target_mask"].any(1).all())
    c = torch.tensor(batch["clinical"], dtype=torch.float32)
    cm = torch.tensor(batch["clinical_mask"], dtype=torch.bool)
    age = torch.tensor(batch["age"], dtype=torch.float32)
    state = model.encode(c, cm, torch.zeros((n, 1, 384)), torch.zeros((n, 1), dtype=torch.bool), age)
    require(state.mean.shape == (n, 192) and bool(torch.isfinite(state.mean).all())
            and not bool(state.abstain.any()))
    return common._cbc_loss(torch, model.cbc_joint_head(state.mean),
        torch.tensor(batch["target_cbc"], dtype=torch.float32),
        torch.tensor(batch["target_mask"], dtype=torch.bool), ~state.abstain)


def adapt(initial, clinical, cm, retinal, rm, age, labels, lm, train, slots,
          prepared, *, seed, external_weight, steps=1500, batch_size=96, progress=None):
    import torch
    import run_bran_overnight_diagnostic_v1 as base
    from bran_clinical_preservation_loss_draft_v1 import clinical_preservation_loss
    from bran_native_rehearsal_batches_v1 import sample_batch

    train = validate(initial, clinical, cm, retinal, rm, age, labels, lm, train, slots,
                     seed, steps, batch_size, external_weight)
    require(progress is None or callable(progress))
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
        physiology_labels = lm & (cm.any(1) | rm)[:, None]
        pos = torch.tensor(common._positive_weights(labels, physiology_labels, train), dtype=torch.float32)
        rng, external_rng = np.random.default_rng(seed), np.random.default_rng(seed + 100000)
        with torch.no_grad():
            reference = teacher.encode(ct[train], cmt[train], torch.zeros_like(rt[train]),
                                       torch.zeros_like(rmt[train]), at[train])
            scale = reference.mean[:, 128:192].std(dim=0, unbiased=False).clamp_min(1.)
        for step in range(steps):
            index = rng.choice(train, batch_size, replace=len(train) < batch_size)
            vc0, vr0 = base.masked_route(rng, cm[index], rm[index])
            vc, vr = torch.tensor(vc0), torch.tensor(vr0[:, None])
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
            # Shared correction in both arms: no disease loss from an abstained
            # all-empty physiology state (age alone is not a supported route).
            disease = common._disease_loss(torch, model.screening_joint_head(disease_state.mean),
                yt[index], ymt[index] & ~disease_state.abstain[:, None], pos)
            hc, hm = common._whole_cbc_removed(clinical[index], cm[index], slots)
            hct, hmt = torch.tensor(hc, dtype=torch.float32), torch.tensor(hm)
            cbc_state = model.encode(common._clean_for_encode(torch, hct, hmt), hmt,
                                     rt[index], rmt[index], at[index])
            completion = common._cbc_loss(torch, model.cbc_joint_head(cbc_state.mean),
                target_c[index][:, slots], cmt[index][:, slots], ~cbc_state.abstain)
            with torch.no_grad():
                reference = teacher.encode(sc, vc, torch.zeros_like(rt[index]),
                                           torch.zeros_like(rmt[index]), at[index])
            preservation = clinical_preservation_loss(state.mean, reference.mean, scale, reference.clinical_available)
            mode = "whole_cbc" if step % 2 == 0 else "partial_cbc"
            batch = sample_batch(prepared, mode, batch_size, external_rng)
            rehearsal = external_loss(model, batch, slots)
            loss = generative + disease + .5 * completion + .1 * preservation + external_weight * rehearsal
            require(bool(torch.isfinite(loss)))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5., error_if_nonfinite=True)
            optimizer.step()
            if progress is not None and ((step + 1) % 100 == 0 or step + 1 == steps):
                progress(step + 1)
        require(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()))
        model.eval()
    return model
