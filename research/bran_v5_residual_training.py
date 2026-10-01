"""Array-only V5 CBC residual-head training kernel.

This module deliberately has no checkpoint, source, serialization, or output
I/O.  It accepts an already authenticated, anchored MLP V5 model plus an
explicit proper-training tensor pool.  The teacher encoder and its native
``nn.Linear`` CBC head are never optimized or replaced.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from bran_multisource_age_v2 import AgeBatch, normalize_age
from bran_multisource_inference_v2 import (
    CompletionPredictionsV2,
    _clean_for_encode,
    _indices,
    _masked_completion_inputs,
    _model_ok,
    _pattern_no_retina,
    _pattern_positions,
    _validate_structure,
    _validate_visible_values,
)
from bran_multisource_anchored_model_v3 import BRANMultisourceAnchoredModelV3
from bran_v5_residual_cbc import ResidualCBC


_ERROR = "v5_residual_training_contract_failed"
PATTERNS = (
    "single_target_hidden",
    "whole_cbc_hidden",
    "red_cell_hidden",
    "single_target_no_retina",
    "whole_cbc_no_retina",
    "red_cell_no_retina",
)
RED_CELL_POSITIONS = (0, 1, 2, 3, 4, 6)


def _require(ok: bool) -> None:
    if not ok:
        raise ValueError(_ERROR)


def _same_tensors(left: dict[str, Tensor], right: dict[str, Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def _snapshot(module: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _teacher_ok(teacher: BRANMultisourceAnchoredModelV3) -> None:
    """Validate the fixed, native-head teacher shape used by the V5 protocol."""

    _model_ok(teacher)
    _require(isinstance(teacher, BRANMultisourceAnchoredModelV3))
    # The experiment is intentionally only for the authenticated anchored MLP
    # arm; accepting the token arm would test a different representation.
    _require(getattr(teacher, "arm", None) == "mlp")
    _require(not teacher.training and all(not parameter.requires_grad for parameter in teacher.parameters()))
    _require(isinstance(teacher.cbc_joint_head, nn.Linear))
    _require(teacher.cbc_joint_head.in_features == 192 and teacher.cbc_joint_head.out_features == 9)
    _require(teacher.cbc_joint_head.weight.device.type == "cpu"
             and teacher.cbc_joint_head.weight.dtype == torch.float32)


def _input_ok(teacher: BRANMultisourceAnchoredModelV3, clinical: Tensor, cm: Tensor,
              retinal: Tensor, rm: Tensor, age: AgeBatch, age_mean: float,
              age_scale: float, cbc_indices: Iterable[int], batch_size: int) -> tuple[int, ...]:
    """Use the native inference validators before any erased-state encode."""

    _teacher_ok(teacher)
    _validate_structure(teacher, clinical, cm, retinal, rm, age, batch_size)
    _require(clinical.shape[0] >= 1)
    _require(clinical.device.type == "cpu" and clinical.dtype == torch.float32)
    _require(retinal.device == clinical.device and retinal.dtype == torch.float32)
    _require(cm.device == clinical.device and rm.device == clinical.device)
    # normalize_age has the authoritative scalar and typed-age validation.
    normalized = normalize_age(age, age_mean, age_scale)
    _require(normalized.device == clinical.device and normalized.dtype == torch.float32)
    return _indices(cbc_indices, teacher)


def _single_position(pattern: str, target_position: int | None) -> tuple[int, ...]:
    _require(pattern in PATTERNS)
    if pattern.startswith("single_target"):
        _require(type(target_position) is int and 0 <= target_position < 9)
        return (target_position,)
    _require(target_position is None)
    return _pattern_positions(pattern)


def encode_erased_state(teacher: BRANMultisourceAnchoredModelV3, clinical: Tensor, cm: Tensor,
                        retinal: Tensor, rm: Tensor, age: AgeBatch, age_mean: float,
                        age_scale: float, pattern: str, cbc_indices: Iterable[int],
                        target_position: int | None = None, batch_size: int = 256) -> tuple[Tensor, Tensor]:
    """Return detached ``(posterior_mean, abstained)`` after exact CBC erasure.

    ``target_position`` is required for a single-target pattern, making it
    impossible to accidentally encode the answer or collapse that context into
    a whole-CBC-hidden encode.  The returned state is finite (abstained rows
    are the model's fixed zero state) and is not persisted by this module.
    """

    cbc = _input_ok(teacher, clinical, cm, retinal, rm, age, age_mean, age_scale,
                    cbc_indices, batch_size)
    positions = _single_position(pattern, target_position)
    erased, visible = _masked_completion_inputs(clinical, cm, cbc, positions)
    encode_rm = torch.zeros_like(rm) if _pattern_no_retina(pattern) else rm
    # Validate only what remains visible.  Thus erased target payloads can be
    # arbitrary/non-finite, but visible physiology still fails closed.
    _validate_visible_values(teacher, erased, visible, retinal, encode_rm)
    age7 = normalize_age(age, age_mean, age_scale)
    states: list[Tensor] = []
    abstentions: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, clinical.shape[0], batch_size):
            stop = min(start + batch_size, clinical.shape[0])
            clean_clinical, clean_retinal = _clean_for_encode(
                teacher, erased[start:stop], visible[start:stop], retinal[start:stop], encode_rm[start:stop]
            )
            posterior = teacher.encode(clean_clinical, visible[start:stop], clean_retinal,
                                       encode_rm[start:stop, None], age7[start:stop])
            _require(posterior.mean.shape == (stop - start, 192)
                     and posterior.abstain.shape == (stop - start,)
                     and posterior.abstain.dtype == torch.bool)
            _require(bool(torch.isfinite(posterior.mean).all()))
            states.append(posterior.mean.detach().clone())
            abstentions.append(posterior.abstain.detach().clone())
    return torch.cat(states, dim=0), torch.cat(abstentions, dim=0)


def _residual_ok(teacher: BRANMultisourceAnchoredModelV3, residual: ResidualCBC) -> None:
    _require(isinstance(residual, ResidualCBC))
    _require(isinstance(residual.baseline, nn.Linear))
    # A residual head is bound to exactly this native frozen base.  It can never
    # be silently reused with a different V5 checkpoint.
    _require(_same_tensors(_snapshot(residual.baseline), _snapshot(teacher.cbc_joint_head)))


def residual_completion_predictions(teacher: BRANMultisourceAnchoredModelV3, residual: ResidualCBC,
                                    clinical: Tensor, cm: Tensor, retinal: Tensor, rm: Tensor,
                                    age: AgeBatch, age_mean: float, age_scale: float,
                                    pattern: str, cbc_indices: Iterable[int],
                                    batch_size: int = 256) -> CompletionPredictionsV2:
    """Complete CBC targets with the bound residual head after fresh erasure."""

    cbc = _input_ok(teacher, clinical, cm, retinal, rm, age, age_mean, age_scale,
                    cbc_indices, batch_size)
    _require(pattern in PATTERNS)
    _residual_ok(teacher, residual)
    selected = _pattern_positions(pattern)
    targets = cm[:, cbc]
    predictions = torch.full((clinical.shape[0], 9), float("nan"), device=clinical.device, dtype=clinical.dtype)
    abstained = torch.ones((clinical.shape[0], 9), device=clinical.device, dtype=torch.bool)
    # No gradients or teacher-head calls are possible in completion inference.
    with torch.no_grad():
        if pattern.startswith("single_target"):
            for position in selected:
                state, row_abstained = encode_erased_state(
                    teacher, clinical, cm, retinal, rm, age, age_mean, age_scale, pattern,
                    cbc, target_position=position, batch_size=batch_size
                )
                present = ~row_abstained
                values = residual(state)
                predictions[present, position] = values[present, position]
                abstained[:, position] = row_abstained
        else:
            state, row_abstained = encode_erased_state(
                teacher, clinical, cm, retinal, rm, age, age_mean, age_scale, pattern,
                cbc, batch_size=batch_size
            )
            present = ~row_abstained
            values = residual(state)
            for position in selected:
                predictions[present, position] = values[present, position]
                abstained[:, position] = row_abstained
    selector = torch.zeros(9, dtype=torch.bool, device=clinical.device)
    selector[list(selected)] = True
    targetmask = targets & selector[None, :]
    predictions = torch.where(targetmask, predictions, torch.full_like(predictions, float("nan")))
    return CompletionPredictionsV2(predictions.detach().clone(), targetmask.detach().clone(),
                                   (targetmask & ~abstained).detach().clone(), abstained.detach().clone())


# A concise alias for evaluation code that calls this residual completion
# inference rather than the frozen native completion helper.
residual_completion_inference = residual_completion_predictions


@dataclass(frozen=True, repr=False)
class TrainingReceipt:
    """Closed aggregate-only record; deliberately contains no examples or outputs."""

    attempted_updates: int
    optimizer_updates: int
    empty_updates: int
    elapsed_seconds: float
    encoder_unchanged: bool
    baseline_head_unchanged: bool
    residual_parameters_changed: bool
    sampling_schedule_sha256: str


def _fold_ids_ok(pool_fold_ids: Tensor, rows: int, heldout_fold: int) -> None:
    _require(isinstance(pool_fold_ids, Tensor) and pool_fold_ids.ndim == 1
             and pool_fold_ids.shape[0] == rows and pool_fold_ids.device.type == "cpu")
    _require(pool_fold_ids.dtype == torch.int64)
    _require(type(heldout_fold) is int and 0 <= heldout_fold <= 4)
    _require(bool(((pool_fold_ids >= 0) & (pool_fold_ids <= 4)).all()))
    # The caller supplies the outcome-blind pool membership.  The only allowed
    # membership rule here is that no held-out-fold row may enter the sampler.
    _require(not bool((pool_fold_ids == heldout_fold).any()))


def train_residual(teacher: BRANMultisourceAnchoredModelV3, clinical: Tensor, cm: Tensor,
                   retinal: Tensor, rm: Tensor, age: AgeBatch, age_mean: float,
                   age_scale: float, cbc_indices: Iterable[int], pool_fold_ids: Tensor,
                   heldout_fold: int, updates: int = 1500, batch_size: int = 96) -> tuple[ResidualCBC, TrainingReceipt]:
    """Fit only a disposable, state-only residual CBC head on a proper-training pool.

    Rows are sampled uniformly with replacement from the supplied non-held-out
    pool using seed ``96101 + heldout_fold``.  Context and target-position
    scheduling is fixed before optimization and never inspects outcomes.
    """

    _require(type(updates) is int and updates >= 1)
    _require(type(batch_size) is int and batch_size >= 1)
    cbc = _input_ok(teacher, clinical, cm, retinal, rm, age, age_mean, age_scale,
                    cbc_indices, batch_size)
    _fold_ids_ok(pool_fold_ids, clinical.shape[0], heldout_fold)
    before_teacher = _snapshot(teacher)
    before_native = _snapshot(teacher.cbc_joint_head)
    before_teacher_mode = teacher.training
    before_teacher_grad = tuple(parameter.requires_grad for parameter in teacher.parameters())
    seed = 96101 + heldout_fold
    head = ResidualCBC(teacher.cbc_joint_head, seed)
    before_residual = _snapshot(head.residual)
    optimizer = torch.optim.AdamW(head.residual.parameters(), lr=1e-4, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    schedule = hashlib.sha256()
    schedule.update(f"v5-residual|{seed}|{updates}|{batch_size}|{clinical.shape[0]}".encode("ascii"))
    single_cycles = {pattern: 0 for pattern in PATTERNS if pattern.startswith("single_target")}
    cbc_tensor = torch.tensor(cbc, dtype=torch.long, device=clinical.device)
    optimizer_updates = 0
    empty_updates = 0
    started = time.perf_counter()
    for update in range(updates):
        pattern = PATTERNS[update % len(PATTERNS)]
        target_position: int | None = None
        if pattern.startswith("single_target"):
            target_position = single_cycles[pattern] % 9
            single_cycles[pattern] += 1
        rows = torch.randint(clinical.shape[0], (batch_size,), generator=generator, device="cpu")
        schedule.update(pattern.encode("ascii"))
        schedule.update(bytes((255 if target_position is None else target_position,)))
        schedule.update(rows.numpy().tobytes())
        state, abstained = encode_erased_state(
            teacher, clinical[rows], cm[rows], retinal[rows], rm[rows],
            AgeBatch(age.value[rows], age.lower[rows], age.upper[rows], age.kind[rows]),
            age_mean, age_scale, pattern, cbc, target_position=target_position, batch_size=batch_size
        )
        positions = _single_position(pattern, target_position)
        selector = torch.zeros(9, dtype=torch.bool, device=clinical.device)
        selector[list(positions)] = True
        target = clinical[rows].index_select(1, cbc_tensor)
        observed_erased = cm[rows].index_select(1, cbc_tensor) & selector[None, :] & ~abstained[:, None]
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = head.objective(state, target, observed_erased)
        if loss is None:
            empty_updates += 1
            continue  # Includes no AdamW decay when there is no valid support.
        loss.backward()
        _require(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                     for parameter in head.residual.parameters()))
        torch.nn.utils.clip_grad_norm_(head.residual.parameters(), 5.0)
        _require(all(bool(torch.isfinite(parameter).all()) for parameter in head.residual.parameters()))
        optimizer.step()
        _require(all(bool(torch.isfinite(parameter).all()) for parameter in head.residual.parameters()))
        optimizer_updates += 1
    elapsed = time.perf_counter() - started
    head.eval()
    encoder_unchanged = (_same_tensors(before_teacher, _snapshot(teacher))
                         and teacher.training == before_teacher_mode
                         and tuple(parameter.requires_grad for parameter in teacher.parameters()) == before_teacher_grad)
    baseline_head_unchanged = (_same_tensors(before_native, _snapshot(teacher.cbc_joint_head))
                               and _same_tensors(before_native, _snapshot(head.baseline)))
    residual_parameters_changed = not _same_tensors(before_residual, _snapshot(head.residual))
    _require(encoder_unchanged and baseline_head_unchanged)
    return head, TrainingReceipt(updates, optimizer_updates, empty_updates, float(elapsed),
                                 encoder_unchanged, baseline_head_unchanged,
                                 residual_parameters_changed, schedule.hexdigest())
