"""Private, array-only V3 pilot lifecycle core; no loaders, files, or logging."""
from __future__ import annotations

import copy
import hashlib
import time
from typing import Callable

import torch
from torch import Tensor

from bran_multisource_continuation_v3 import train_step_v3
from bran_multisource_model_v2 import BRANMultisourceModelV2
from bran_multisource_training_v2 import MaterializedBatch


_INVALID = "multisource pilot inputs invalid"
_STEPS = 100
_LR = 5e-5
_WEIGHT_DECAY = 1e-4


def _invalid() -> None:
    raise ValueError(_INVALID)


def _seed(seed: int, step: int, salt: int) -> int:
    return (seed + 1000003 * step + salt) % (2**63 - 1)


def _digest_tensor(digest: "hashlib._Hash", value: Tensor) -> None:
    digest.update(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.detach().to(device="cpu").contiguous().numpy().tobytes())


def _batch_digest(batch: MaterializedBatch, positive_weight: Tensor) -> str:
    """Private equality audit only; no batch values are returned or logged."""
    digest = hashlib.sha256()
    for value in (batch.c, batch.cm, batch.r, batch.rm, batch.age.value,
                  batch.age.lower, batch.age.upper, batch.age.kind, positive_weight):
        _digest_tensor(digest, value)
    if batch.labels is None:
        digest.update(b"labels:none")
    else:
        _digest_tensor(digest, batch.labels)
        _digest_tensor(digest, batch.labelmask)
    return digest.hexdigest()


def _valid_sampler(value: object, paired: bool) -> bool:
    return (callable(getattr(value, "sample", None))
            and (not paired or isinstance(getattr(value, "positive_weights", None), Tensor)))


def _fresh(factory: object, seed: int, salt: int) -> object:
    if not callable(factory):
        _invalid()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_seed(seed, 0, salt))
        sampler = factory()
    return sampler


def _sample(sampler: object, seed: int, step: int, salt: int) -> object:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_seed(seed, step, salt))
        return sampler.sample()


def _validate_base(model: object, teacher: object, steps: object, age_mean: object,
                   age_scale: object, seed: object, cbc_indices: object,
                   state_scale: object) -> tuple[int, ...]:
    if (not isinstance(model, BRANMultisourceModelV2)
            or not isinstance(teacher, BRANMultisourceModelV2) or model is teacher
            or isinstance(steps, bool) or not isinstance(steps, int) or steps != _STEPS
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        _invalid()
    try:
        mean, scale = float(age_mean), float(age_scale)
        cbc = tuple(cbc_indices)
    except (TypeError, ValueError, OverflowError):
        _invalid()
    if (not torch.isfinite(torch.tensor(mean)) or not torch.isfinite(torch.tensor(scale)) or scale <= 0
            or model.arm != "mlp" or teacher.arm != "mlp" or model.config != teacher.config
            or model.eligible_indices != teacher.eligible_indices or model.cbc_indices != teacher.cbc_indices
            or cbc != model.cbc_indices or teacher.training
            or any(parameter.requires_grad for parameter in teacher.parameters())
            or any(parameter.device.type != "cpu" for parameter in model.parameters())
            or any(parameter.device.type != "cpu" for parameter in teacher.parameters())):
        _invalid()
    if (not isinstance(state_scale, Tensor) or not state_scale.is_floating_point()
            or state_scale.shape != (192,) or state_scale.device.type != "cpu"
            or state_scale.dtype != next(model.parameters()).dtype or state_scale.requires_grad
            or not torch.isfinite(state_scale).all() or bool((state_scale < 1).any())):
        _invalid()
    return cbc


def _unchanged(before: dict[str, Tensor], module: BRANMultisourceModelV2) -> bool:
    current = module.state_dict()
    return before.keys() == current.keys() and all(torch.equal(before[name], current[name]) for name in before)


def run_pilot(
    model: BRANMultisourceModelV2,
    teacher: BRANMultisourceModelV2,
    paired_sampler: Callable[[], object],
    source_sampler: Callable[[], object],
    *,
    steps: int,
    age_mean: float,
    age_scale: float,
    seed: int,
    cbc_indices,
    state_scale: Tensor,
) -> dict[str, object]:
    """Run fixed 100-update C then M pilots on disposable deep copies.

    Sampler arguments are zero-argument factories.  Their returned paired
    sampler exposes ``sample()`` and a tensor ``positive_weights``; source
    samplers expose ``sample()``.  Returned models remain local to the caller
    and are intentionally neither saved nor selected here.
    """
    try:
        cbc = _validate_base(model, teacher, steps, age_mean, age_scale, seed,
                             cbc_indices, state_scale)
        baseline_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        teacher_before = {name: value.detach().clone() for name, value in teacher.state_dict().items()}
        baseline_mode, teacher_mode = model.training, teacher.training
        baseline_grad = tuple(parameter.requires_grad for parameter in model.parameters())
        teacher_grad = tuple(parameter.requires_grad for parameter in teacher.parameters())

        control = copy.deepcopy(model)
        candidate = copy.deepcopy(model)
        for arm in (control, candidate):
            arm.train()
            for parameter in arm.parameters():
                parameter.requires_grad_(True)
        control_optimizer = torch.optim.AdamW(control.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY)
        candidate_optimizer = torch.optim.AdamW(candidate.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY)

        control_paired = _fresh(paired_sampler, seed, 11)
        candidate_paired = _fresh(paired_sampler, seed, 11)
        control_source = _fresh(source_sampler, seed, 17)
        candidate_source = _fresh(source_sampler, seed, 17)
        if not all((_valid_sampler(control_paired, True), _valid_sampler(candidate_paired, True),
                    _valid_sampler(control_source, False), _valid_sampler(candidate_source, False))):
            _invalid()

        started = time.perf_counter()
        control_updates = candidate_updates = 0
        source_generative_supported = source_cbc_supported = False
        paired_batches_equal = paired_masks_equal = True
        with torch.random.fork_rng(devices=[]):
            for step in range(_STEPS):
                c_pair = _sample(control_paired, seed, step, 31)
                m_pair = _sample(candidate_paired, seed, step, 31)
                c_source = _sample(control_source, seed, step, 37)
                m_source = _sample(candidate_source, seed, step, 37)
                c_weight, m_weight = control_paired.positive_weights, candidate_paired.positive_weights
                if not isinstance(c_pair, MaterializedBatch) or not isinstance(m_pair, MaterializedBatch):
                    _invalid()
                paired_batches_equal = paired_batches_equal and _batch_digest(c_pair, c_weight) == _batch_digest(m_pair, m_weight)
                c_result = train_step_v3(control, teacher, control_optimizer, c_pair, c_source, step,
                                         age_mean, age_scale, seed, cbc, c_weight, state_scale,
                                         source_enabled=False)
                m_result = train_step_v3(candidate, teacher, candidate_optimizer, m_pair, m_source, step,
                                         age_mean, age_scale, seed, cbc, m_weight, state_scale,
                                         source_enabled=True)
                paired_masks_equal = paired_masks_equal and (
                    c_result["mask_hashes"].get("paired") == m_result["mask_hashes"].get("paired")
                    and c_result["mask_hashes"].get("paired_completion") == m_result["mask_hashes"].get("paired_completion"))
                control_updates += int(bool(c_result["optimizer_updated"]))
                candidate_updates += int(bool(m_result["optimizer_updated"]))
                source_generative_supported = source_generative_supported or bool(c_result.get("source_generative_supervised", False)) or bool(m_result["source_generative_supervised"])
                source_cbc_supported = source_cbc_supported or bool(c_result.get("source_cbc_supervised", False)) or bool(m_result["source_cbc_supervised"])
        elapsed = time.perf_counter() - started
        if (control_updates == 0 or candidate_updates == 0
                or not _unchanged(baseline_before, model) or not _unchanged(teacher_before, teacher)
                or model.training != baseline_mode or teacher.training != teacher_mode
                or tuple(parameter.requires_grad for parameter in model.parameters()) != baseline_grad
                or tuple(parameter.requires_grad for parameter in teacher.parameters()) != teacher_grad):
            _invalid()
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        _invalid()

    return {
        "elapsed_seconds": float(elapsed), "control_updates": control_updates,
        "candidate_updates": candidate_updates,
        "source_generative_supported": source_generative_supported,
        "source_cbc_supported": source_cbc_supported,
        "paired_batches_equivalent": paired_batches_equal,
        "paired_mask_streams_equivalent": paired_masks_equal,
        "control_model": control, "multisource_model": candidate,
    }
