"""Array-only synthetic BRAN V2 training step; deliberately no I/O or loaders."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Optional

import torch
from torch import Tensor

from bran_multisource_age_v2 import AgeBatch, augment_reported_age, normalize_age, validate_age
from bran_multisource_model_v2 import BRANMultisourceModelV2
from bran_multisource_protocol_v2 import PARAMETERS
from bran_screening_joint_kernel_v1 import _cbc_loss, _disease_loss


_INVALID = "multisource training inputs invalid"
_RATES = (0.25, 0.50, 0.75)


@dataclass(frozen=True, repr=False)
class MaterializedBatch:
    """Already-recipient-normalized tensors and original-year typed ages only."""

    c: Tensor                 # [B, 59]
    cm: Tensor                # [B, 59] bool
    r: Tensor                 # [B, images, 384]
    rm: Tensor                # [B, images] bool
    age: AgeBatch
    labels: Optional[Tensor] = None       # [B, 26]
    labelmask: Optional[Tensor] = None    # [B, 26] bool


def _invalid() -> None:
    raise ValueError(_INVALID)


def _validate_batch(batch: MaterializedBatch) -> None:
    if not isinstance(batch, MaterializedBatch):
        _invalid()
    if (not isinstance(batch.c, Tensor) or not batch.c.is_floating_point() or batch.c.ndim != 2 or batch.c.shape[1] != 59
            or not isinstance(batch.cm, Tensor) or batch.cm.dtype != torch.bool or batch.cm.shape != batch.c.shape
            or not isinstance(batch.r, Tensor) or not batch.r.is_floating_point() or batch.r.ndim != 3 or batch.r.shape[0] != batch.c.shape[0] or batch.r.shape[2] != 384
            or not isinstance(batch.rm, Tensor) or batch.rm.dtype != torch.bool or batch.rm.shape != batch.r.shape[:2]):
        _invalid()
    if any(x.device != batch.c.device for x in (batch.cm, batch.r, batch.rm, batch.age.value, batch.age.lower, batch.age.upper, batch.age.kind)):
        _invalid()
    validate_age(batch.age)
    if batch.age.value.shape != (batch.c.shape[0],):
        _invalid()
    if not torch.isfinite(batch.c[batch.cm]).all() or not torch.isfinite(batch.r[batch.rm]).all():
        _invalid()
    if (batch.labels is None) != (batch.labelmask is None):
        _invalid()
    if batch.labels is not None:
        if (not isinstance(batch.labels, Tensor) or not batch.labels.is_floating_point() or batch.labels.shape != (batch.c.shape[0], 26)
                or not isinstance(batch.labelmask, Tensor) or batch.labelmask.dtype != torch.bool or batch.labelmask.shape != batch.labels.shape
                or batch.labels.device != batch.c.device or batch.labelmask.device != batch.c.device):
            _invalid()
        observed = batch.labelmask
        if not torch.isfinite(batch.labels[observed]).all() or not torch.all((batch.labels[observed] == 0) | (batch.labels[observed] == 1)):
            _invalid()


def _has_observed_labels(batch: MaterializedBatch) -> bool:
    return batch.labelmask is not None and bool(batch.labelmask.any())


def _generator(device: torch.device, seed: int, step: int, salt: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed((int(seed) + 1000003 * int(step) + int(salt)) % (2**63 - 1))
    return generator


def _age7(batch: MaterializedBatch, mean: float, scale: float, generator: torch.Generator) -> Tensor:
    return normalize_age(augment_reported_age(batch.age, generator), mean, scale)


def _clinical_visible(mask: Tensor, generator: torch.Generator, step: int) -> Tensor:
    """Fixed 25/50/75% field hiding cycle; inactive payloads stay untouched."""
    hidden_rate = _RATES[step % len(_RATES)]
    return mask & (torch.rand(mask.shape, device=mask.device, generator=generator) >= hidden_rate)


def _retinal_target(batch: MaterializedBatch) -> tuple[Tensor, Tensor]:
    valid = batch.rm & torch.isfinite(batch.r).all(dim=-1)
    clean = torch.zeros_like(batch.r)
    expanded = valid[..., None].expand_as(batch.r)
    clean = clean.masked_scatter(expanded, batch.r.masked_select(expanded))
    target = clean.sum(dim=1) / valid.sum(dim=1).clamp_min(1)[:, None]
    mask = valid.any(dim=1)[:, None].expand(-1, 384)
    return target, mask


def _rows(batch: MaterializedBatch, rows: Tensor) -> MaterializedBatch:
    labels = None if batch.labels is None else batch.labels[rows]
    labelmask = None if batch.labelmask is None else batch.labelmask[rows]
    return MaterializedBatch(batch.c[rows], batch.cm[rows], batch.r[rows], batch.rm[rows],
                             AgeBatch(batch.age.value[rows], batch.age.lower[rows], batch.age.upper[rows], batch.age.kind[rows]), labels, labelmask)


def _zero(model: BRANMultisourceModelV2) -> Tensor:
    return next(model.parameters()).sum() * 0.0


def _masked_digest(*masks: Tensor) -> str:
    digest = hashlib.sha256()
    for mask in masks:
        digest.update(str(tuple(mask.shape)).encode())
        digest.update(mask.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy().tobytes())
    return digest.hexdigest()


def _generative(model: BRANMultisourceModelV2, batch: MaterializedBatch, age7: Tensor,
                visible_c: Tensor, visible_r: Tensor, step: int) -> tuple[Tensor, int]:
    """Filter and re-encode, so an abstained state is never conditionally scored."""
    with torch.no_grad():
        preliminary = model.encode(batch.c, visible_c, batch.r, visible_r, age7)
    keep = ~preliminary.abstain
    if not bool(keep.any()):
        return _zero(model), 0
    selected = _rows(batch, keep)
    age_selected = age7[keep]
    state = model.encode(selected.c, visible_c[keep], selected.r, visible_r[keep], age_selected)
    target_r, target_rm = _retinal_target(selected)
    result = model.objective(state, age_selected, selected.c, selected.cm, visible_c[keep], target_r, target_rm,
                             visible_r[keep].any(dim=1)[:, None].expand(-1, 384),
                             kl_weight=PARAMETERS["kl_max"] * min(1.0, (step + 1) / PARAMETERS["kl_warmup_steps"]),
                             visible_weight=PARAMETERS["visible_weight"])
    return result["loss"], int(keep.sum())


def _route_masks(batch: MaterializedBatch, step: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    route = PARAMETERS["screening_routes"][step % len(PARAMETERS["screening_routes"])]
    clinical = _clinical_visible(batch.cm, generator, step)
    retinal = batch.rm.clone()
    if route == "clinical":
        retinal.zero_()
    elif route == "retinal":
        clinical.zero_()
    return clinical, retinal


def _completion_masks(batch: MaterializedBatch, cbc_indices: tuple[int, ...], step: int) -> tuple[Tensor, Tensor]:
    pattern = PARAMETERS["completion_patterns"][step % len(PARAMETERS["completion_patterns"])]
    hidden = []
    if pattern.startswith("single_target"):
        hidden = [cbc_indices[(step // len(PARAMETERS["completion_patterns"])) % len(cbc_indices)]]
    elif pattern.startswith("whole_cbc"):
        hidden = list(cbc_indices)
    else:  # red-cell analytes in the fixed V2 CBC target ordering.
        hidden = [cbc_indices[i] for i in (0, 1, 2, 3, 4, 6)]
    clinical = batch.cm.clone()
    clinical[:, hidden] = False
    retinal = batch.rm.clone()
    if pattern.endswith("no_retina"):
        retinal.zero_()
    return clinical, retinal


def _screening_loss(model: BRANMultisourceModelV2, batch: MaterializedBatch, age7: Tensor,
                    visible_c: Tensor, visible_r: Tensor, positive_weight: Tensor) -> tuple[Tensor, int]:
    if batch.labels is None or batch.labelmask is None:
        _invalid()
    with torch.no_grad():
        provisional = model.encode(batch.c, visible_c, batch.r, visible_r, age7)
    keep = ~provisional.abstain & batch.labelmask.any(dim=1)
    if not bool(keep.any()):
        return _zero(model), 0
    selected = _rows(batch, keep)
    state = model.encode(selected.c, visible_c[keep], selected.r, visible_r[keep], age7[keep])
    return _disease_loss(torch, model.screening_joint_head(state.mean), selected.labels, selected.labelmask, positive_weight), int(keep.sum())


def _cbc_completion_loss(model: BRANMultisourceModelV2, batch: MaterializedBatch, age7: Tensor,
                         cbc_indices: tuple[int, ...], step: int) -> tuple[Tensor, int, Tensor, Tensor, Tensor]:
    visible_c, visible_r = _completion_masks(batch, cbc_indices, step)
    # Partial patterns preserve the other CBC values and flags as context.  The
    # all-nine erase helper remains available for callers requesting whole CBC.
    erased_mask = visible_c
    erased_c = torch.where(erased_mask, batch.c, torch.zeros_like(batch.c))
    with torch.no_grad():
        provisional = model.encode(erased_c, erased_mask, batch.r, visible_r, age7)
    keep = ~provisional.abstain
    if not bool(keep.any()):
        return _zero(model), 0, erased_mask, visible_r, torch.zeros((0, 9), device=batch.c.device, dtype=torch.bool)
    state = model.encode(erased_c[keep], erased_mask[keep], batch.r[keep], visible_r[keep], age7[keep])
    targets = batch.c[keep][:, cbc_indices]
    # Only fields explicitly removed from the completion input are targets.
    target_mask = batch.cm[keep][:, cbc_indices] & ~erased_mask[keep][:, cbc_indices]
    loss = _cbc_loss(torch, model.cbc_joint_head(state.mean), targets, target_mask,
                     (state.clinical_available | state.retinal_available) & ~state.abstain)
    return loss, int(target_mask.any(dim=1).sum()), erased_mask, visible_r, target_mask


def _require_unpaired(batch: MaterializedBatch, model: BRANMultisourceModelV2) -> None:
    has_clinical = batch.cm[:, model.eligible_indices].any(dim=1)
    if bool((has_clinical & batch.rm.any(dim=1)).any()):
        _invalid()


def _valid_optimizer(model: BRANMultisourceModelV2, optimizer: torch.optim.Optimizer) -> bool:
    model_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    return (model_ids == optimizer_ids and all(float(group.get("lr", float("nan"))) == PARAMETERS["learning_rate"]
                                                and float(group.get("weight_decay", float("nan"))) == PARAMETERS["weight_decay"]
                                                for group in optimizer.param_groups))


def train_step(model: BRANMultisourceModelV2, optimizer: torch.optim.Optimizer, stage: str,
               batch: MaterializedBatch, rehearsal_batch: Optional[MaterializedBatch], step: int,
               age_mean: float, age_scale: float, seed: int, cbc_indices, positive_weight: Optional[Tensor] = None) -> dict[str, object]:
    """One deterministic local training update; returns only scalar losses/audits.

    A uses an unlabeled unpaired batch. B/C add one unlabeled external rehearsal
    batch at 0.5. C alone adds the native screening and CBC completion losses.
    The returned dictionary is private in-memory diagnostics, never a batch-loss
    logging format for real data.
    """
    if (not isinstance(model, BRANMultisourceModelV2) or not isinstance(optimizer, torch.optim.AdamW)
            or stage not in ("A", "B", "C") or isinstance(step, bool) or not isinstance(step, int) or step < 0
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        _invalid()
    cbc = tuple(cbc_indices)
    if cbc != model.cbc_indices:
        _invalid()
    _validate_batch(batch)
    if batch.c.device.type != "cpu" or any(parameter.device.type != "cpu" for parameter in model.parameters()) or not _valid_optimizer(model, optimizer):
        _invalid()
    if stage == "A":
        if rehearsal_batch is not None or _has_observed_labels(batch):
            _invalid()
        _require_unpaired(batch, model)
    else:
        if rehearsal_batch is None:
            _invalid()
        _validate_batch(rehearsal_batch)
        if rehearsal_batch.c.device.type != "cpu" or _has_observed_labels(rehearsal_batch):
            _invalid()
        _require_unpaired(rehearsal_batch, model)
    if stage == "C":
        if batch.labels is None or batch.labelmask is None or not isinstance(positive_weight, Tensor) or positive_weight.shape != (26,) or not positive_weight.is_floating_point() or not torch.isfinite(positive_weight).all() or bool((positive_weight <= 0).any()):
            _invalid()
    elif positive_weight is not None:
        _invalid()

    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        main_age = _age7(batch, age_mean, age_scale, _generator(batch.c.device, seed, step, 11))
        main_mask_gen = _generator(batch.c.device, seed, step, 23)
        if stage == "A":
            visible_c = _clinical_visible(batch.cm, main_mask_gen, step)
            visible_r = batch.rm.clone()  # retinal-only unpaired loss is always visible reconstruction.
        else:
            visible_c, visible_r = _route_masks(batch, step, main_mask_gen)
        torch.manual_seed((seed + 1000003 * step + 101) % (2**63 - 1))
        generative, generative_rows = _generative(model, batch, main_age, visible_c, visible_r, step)
        rehearsal = _zero(model)
        rehearsal_rows = 0
        rehearsal_c = rehearsal_r = None
        if rehearsal_batch is not None:
            rehearse_age = _age7(rehearsal_batch, age_mean, age_scale, _generator(rehearsal_batch.c.device, seed, step, 31))
            rehearse_gen = _generator(rehearsal_batch.c.device, seed, step, 37)
            rehearsal_c = _clinical_visible(rehearsal_batch.cm, rehearse_gen, step)
            rehearsal_r = rehearsal_batch.rm.clone()  # same unpaired visible-retina treatment as A.
            torch.manual_seed((seed + 1000003 * step + 103) % (2**63 - 1))
            rehearsal, rehearsal_rows = _generative(model, rehearsal_batch, rehearse_age, rehearsal_c, rehearsal_r, step)
        screening = _zero(model)
        cbc_loss = _zero(model)
        screening_rows = cbc_rows = 0
        completion_c = completion_r = None
        if stage == "C":
            torch.manual_seed((seed + 1000003 * step + 107) % (2**63 - 1))
            screening, screening_rows = _screening_loss(model, batch, main_age, visible_c, visible_r, positive_weight)
            torch.manual_seed((seed + 1000003 * step + 109) % (2**63 - 1))
            cbc_loss, cbc_rows, completion_c, completion_r, _ = _cbc_completion_loss(model, batch, main_age, cbc, step)
        total = generative + PARAMETERS["rehearsal_weight"] * rehearsal
        if stage == "C":
            total = total + PARAMETERS["screening_weight"] * screening + PARAMETERS["cbc_weight"] * cbc_loss
        supported = bool(generative_rows or rehearsal_rows or screening_rows or cbc_rows)
        if not bool(torch.isfinite(total)):
            _invalid()
        if supported:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PARAMETERS["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()

    hashes = {"main": _masked_digest(visible_c, visible_r)}
    if rehearsal_c is not None:
        hashes["rehearsal"] = _masked_digest(rehearsal_c, rehearsal_r)
    if completion_c is not None:
        hashes["completion"] = _masked_digest(completion_c, completion_r)
    return {"stage": stage, "step": step, "loss": float(total.detach()), "generative_loss": float(generative.detach()),
            "rehearsal_loss": float(rehearsal.detach()), "screening_loss": float(screening.detach()), "cbc_loss": float(cbc_loss.detach()),
            "generative_supervised": bool(generative_rows), "rehearsal_supervised": bool(rehearsal_rows),
            "screening_supervised": bool(screening_rows), "cbc_supervised": bool(cbc_rows), "optimizer_updated": supported,
            "mask_hashes": hashes}
