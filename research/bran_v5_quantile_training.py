"""Array-only training and target-erased inference for the V5 quantile head."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from bran_multisource_age_v2 import AgeBatch
from bran_multisource_inference_v2 import _pattern_positions
from bran_multisource_anchored_model_v3 import BRANMultisourceAnchoredModelV3
from bran_v5_quantile_cbc import QuantileCBC
from bran_v5_residual_training import (
    PATTERNS,
    _fold_ids_ok,
    _input_ok,
    _same_tensors,
    _single_position,
    _snapshot,
    encode_erased_state,
)


_ERROR = "v5_quantile_training_contract_failed"


def _require(ok: bool) -> None:
    if not ok:
        raise ValueError(_ERROR)


@dataclass(frozen=True, repr=False)
class QuantilePredictions:
    q05: Tensor
    median: Tensor
    q95: Tensor
    scoringmask: Tensor
    abstained: Tensor


@dataclass(frozen=True, repr=False)
class QuantileTrainingReceipt:
    attempted_updates: int
    optimizer_updates: int
    empty_updates: int
    elapsed_seconds: float
    encoder_unchanged: bool
    baseline_head_unchanged: bool
    quantile_parameters_changed: bool
    sampling_schedule_sha256: str


def _quantile_ok(teacher: BRANMultisourceAnchoredModelV3, head: QuantileCBC) -> None:
    _require(isinstance(head, QuantileCBC))
    _require(_same_tensors(_snapshot(head.baseline), _snapshot(teacher.cbc_joint_head)))
    _require(not head.baseline.training and all(not parameter.requires_grad for parameter in head.baseline.parameters()))


def predict_quantiles(teacher: BRANMultisourceAnchoredModelV3, head: QuantileCBC,
                      clinical: Tensor, cm: Tensor, retinal: Tensor, rm: Tensor,
                      age: AgeBatch, age_mean: float, age_scale: float, pattern: str,
                      cbc_indices: Iterable[int], batch_size: int = 256) -> QuantilePredictions:
    """Infer all requested target quantiles after fresh context-specific erasure."""

    try:
        cbc = _input_ok(teacher, clinical, cm, retinal, rm, age, age_mean, age_scale, cbc_indices, batch_size)
        _require(pattern in PATTERNS)
        _quantile_ok(teacher, head)
        selected = _pattern_positions(pattern)
        targets = cm[:, cbc]
        outputs = [torch.full((len(clinical), 9), float("nan"), device=clinical.device, dtype=clinical.dtype) for _ in range(3)]
        abstained = torch.ones((len(clinical), 9), device=clinical.device, dtype=torch.bool)
        with torch.no_grad():
            if pattern.startswith("single_target"):
                for position in selected:
                    state, row_abstained = encode_erased_state(teacher, clinical, cm, retinal, rm, age, age_mean, age_scale,
                                                               pattern, cbc, target_position=position, batch_size=batch_size)
                    values = head(state)
                    present = ~row_abstained
                    for output, value in zip(outputs, values):
                        output[present, position] = value[present, position]
                    abstained[:, position] = row_abstained
            else:
                state, row_abstained = encode_erased_state(teacher, clinical, cm, retinal, rm, age, age_mean, age_scale,
                                                           pattern, cbc, batch_size=batch_size)
                values = head(state)
                present = ~row_abstained
                for position in selected:
                    for output, value in zip(outputs, values):
                        output[present, position] = value[present, position]
                    abstained[:, position] = row_abstained
        selector = torch.zeros(9, dtype=torch.bool, device=clinical.device)
        selector[list(selected)] = True
        targetmask = targets & selector[None, :]
        q05, median, q95 = [torch.where(targetmask, output, torch.full_like(output, float("nan"))).detach().clone() for output in outputs]
        scoringmask = (targetmask & ~abstained).detach().clone()
        _require(bool(torch.isfinite(q05[scoringmask]).all()) and bool(torch.isfinite(median[scoringmask]).all())
                 and bool(torch.isfinite(q95[scoringmask]).all()) and bool((q05[scoringmask] < median[scoringmask]).all())
                 and bool((median[scoringmask] < q95[scoringmask]).all()))
        return QuantilePredictions(q05, median, q95, scoringmask, abstained.detach().clone())
    except ValueError:
        _require(False)
    raise AssertionError("unreachable")


def train_quantile(teacher: BRANMultisourceAnchoredModelV3, clinical: Tensor, cm: Tensor,
                   retinal: Tensor, rm: Tensor, age: AgeBatch, age_mean: float, age_scale: float,
                   cbc_indices: Iterable[int], pool_fold_ids: Tensor, heldout_fold: int,
                   updates: int = 1500, batch_size: int = 96) -> tuple[QuantileCBC, QuantileTrainingReceipt]:
    """Fit the disposable quantile attachment on the non-held-out outer pool only."""

    _require(type(updates) is int and updates >= 1 and type(batch_size) is int and batch_size >= 1)
    try:
        cbc = _input_ok(teacher, clinical, cm, retinal, rm, age, age_mean, age_scale, cbc_indices, batch_size)
        _fold_ids_ok(pool_fold_ids, len(clinical), heldout_fold)
    except ValueError:
        _require(False)
    before_teacher, before_native = _snapshot(teacher), _snapshot(teacher.cbc_joint_head)
    mode = teacher.training
    grads = tuple(parameter.requires_grad for parameter in teacher.parameters())
    seed = 98301 + heldout_fold
    head = QuantileCBC(teacher.cbc_joint_head, seed)
    before_quantiles = _snapshot(head.quantiles)
    optimizer = torch.optim.AdamW(head.quantiles.parameters(), lr=1e-4, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    schedule = hashlib.sha256()
    schedule.update(f"v5-quantile|{seed}|{updates}|{batch_size}|{len(clinical)}".encode("ascii"))
    cycle = {pattern: 0 for pattern in PATTERNS if pattern.startswith("single_target")}
    cbc_tensor = torch.tensor(cbc, device=clinical.device, dtype=torch.long)
    optimized = empty = 0
    started = time.perf_counter()
    for update in range(updates):
        pattern = PATTERNS[update % len(PATTERNS)]
        position: int | None = None
        if pattern.startswith("single_target"):
            position = cycle[pattern] % 9
            cycle[pattern] += 1
        rows = torch.randint(len(clinical), (batch_size,), generator=generator, device="cpu")
        schedule.update(pattern.encode("ascii")); schedule.update(bytes((255 if position is None else position,)))
        schedule.update(rows.numpy().tobytes())
        batch_age = AgeBatch(age.value[rows], age.lower[rows], age.upper[rows], age.kind[rows])
        state, abstained = encode_erased_state(teacher, clinical[rows], cm[rows], retinal[rows], rm[rows], batch_age,
                                               age_mean, age_scale, pattern, cbc, target_position=position, batch_size=batch_size)
        positions = _single_position(pattern, position)
        selector = torch.zeros(9, dtype=torch.bool, device=clinical.device); selector[list(positions)] = True
        target = clinical[rows].index_select(1, cbc_tensor)
        observed = cm[rows].index_select(1, cbc_tensor) & selector[None, :] & ~abstained[:, None]
        head.train(); optimizer.zero_grad(set_to_none=True)
        loss = head.objective(state, target, observed)
        if loss is None:
            empty += 1
            continue
        loss.backward()
        _require(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in head.quantiles.parameters()))
        torch.nn.utils.clip_grad_norm_(head.quantiles.parameters(), 5.0)
        optimizer.step()
        _require(all(bool(torch.isfinite(parameter).all()) for parameter in head.quantiles.parameters()))
        optimized += 1
    head.eval()
    encoder_unchanged = (_same_tensors(before_teacher, _snapshot(teacher)) and teacher.training == mode
                         and tuple(parameter.requires_grad for parameter in teacher.parameters()) == grads)
    baseline_unchanged = (_same_tensors(before_native, _snapshot(teacher.cbc_joint_head))
                          and _same_tensors(before_native, _snapshot(head.baseline)))
    changed = not _same_tensors(before_quantiles, _snapshot(head.quantiles))
    _require(encoder_unchanged and baseline_unchanged)
    return head, QuantileTrainingReceipt(updates, optimized, empty, float(time.perf_counter() - started),
                                         encoder_unchanged, baseline_unchanged, changed, schedule.hexdigest())


quantile_completion_predictions = predict_quantiles
