"""Private array-only V3 full-fit lifecycle; no checkpoints, loaders, or I/O."""
from __future__ import annotations

import copy
import hashlib
import time
from typing import Callable, Optional

import torch
from torch import Tensor

from bran_multisource_continuation_v3 import train_step_v3
from bran_multisource_model_v2 import BRANMultisourceModelV2
from bran_multisource_training_v2 import MaterializedBatch


_INVALID = "multisource fit inputs invalid"
_STEPS = 3000
_PROGRESS_INTERVAL = 300
_LR = 5e-5
_WEIGHT_DECAY = 1e-4


def _invalid() -> None:
    raise ValueError(_INVALID)


def _seed(seed: int, step: int, salt: int) -> int:
    return (seed + 1000003 * step + salt) % (2**63 - 1)


def _tensor_digest(digest: "hashlib._Hash", value: Tensor) -> None:
    digest.update(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.detach().to(device="cpu").contiguous().numpy().tobytes())


def _add_paired_input(digest: "hashlib._Hash", batch: MaterializedBatch,
                      positive_weight: Tensor) -> None:
    for value in (batch.c, batch.cm, batch.r, batch.rm, batch.age.value,
                  batch.age.lower, batch.age.upper, batch.age.kind, positive_weight):
        _tensor_digest(digest, value)
    if batch.labels is None:
        digest.update(b"labels:none")
    else:
        _tensor_digest(digest, batch.labels)
        _tensor_digest(digest, batch.labelmask)


def _fresh(factory: object, seed: int, salt: int) -> object:
    if not callable(factory):
        _invalid()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_seed(seed, 0, salt))
        return factory()


def _sample(sampler: object, seed: int, step: int, salt: int) -> object:
    if not callable(getattr(sampler, "sample", None)):
        _invalid()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_seed(seed, step, salt))
        return sampler.sample()


def _validate(initial: object, teacher: object, paired_factory: object,
              source_factory: object, state_scale: object, age_mean: object,
              age_scale: object, seed: object, cbc_indices: object, role: object,
              progress: object) -> tuple[int, ...]:
    if (not isinstance(initial, BRANMultisourceModelV2)
            or not isinstance(teacher, BRANMultisourceModelV2) or initial is teacher
            or role not in ("C", "M") or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or not callable(paired_factory) or not callable(source_factory)
            or (progress is not None and not callable(progress))):
        _invalid()
    try:
        mean, scale = float(age_mean), float(age_scale)
        cbc = tuple(cbc_indices)
    except (TypeError, ValueError, OverflowError):
        _invalid()
    if (not torch.isfinite(torch.tensor(mean)) or not torch.isfinite(torch.tensor(scale)) or scale <= 0
            or initial.arm != "mlp" or teacher.arm != "mlp" or initial.config != teacher.config
            or initial.eligible_indices != teacher.eligible_indices or initial.cbc_indices != teacher.cbc_indices
            or cbc != initial.cbc_indices or teacher.training
            or any(parameter.requires_grad for parameter in teacher.parameters())
            or any(parameter.device.type != "cpu" for parameter in initial.parameters())
            or any(parameter.device.type != "cpu" for parameter in teacher.parameters())):
        _invalid()
    if (not isinstance(state_scale, Tensor) or not state_scale.is_floating_point()
            or state_scale.shape != (192,) or state_scale.device.type != "cpu"
            or state_scale.dtype != next(initial.parameters()).dtype or state_scale.requires_grad
            or not torch.isfinite(state_scale).all() or bool((state_scale < 1).any())):
        _invalid()
    return cbc


def _unchanged(before: dict[str, Tensor], module: BRANMultisourceModelV2) -> bool:
    current = module.state_dict()
    return before.keys() == current.keys() and all(torch.equal(before[name], current[name]) for name in before)


def fit_one_v3(
    initial: BRANMultisourceModelV2,
    teacher: BRANMultisourceModelV2,
    paired_sampler: Callable[[], object],
    source_sampler: Callable[[], object],
    state_scale: Tensor,
    age_mean: float,
    age_scale: float,
    seed: int,
    cbc_indices,
    role: str,
    progress: Optional[Callable[[int], None]] = None,
) -> dict[str, object]:
    """Fit one disposable C or M candidate for the fixed private 3,000 updates.

    Factories are zero-argument and must return new samplers with ``sample()``;
    the paired sampler also supplies a tensor ``positive_weights``.  No sampler,
    input, target, state, or gradient is emitted from this function.
    """
    try:
        cbc = _validate(initial, teacher, paired_sampler, source_sampler, state_scale,
                        age_mean, age_scale, seed, cbc_indices, role, progress)
        initial_before = {name: value.detach().clone() for name, value in initial.state_dict().items()}
        teacher_before = {name: value.detach().clone() for name, value in teacher.state_dict().items()}
        initial_mode, teacher_mode = initial.training, teacher.training
        initial_grad = tuple(parameter.requires_grad for parameter in initial.parameters())
        teacher_grad = tuple(parameter.requires_grad for parameter in teacher.parameters())

        candidate = copy.deepcopy(initial)
        candidate.train()
        for parameter in candidate.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY)
        paired = _fresh(paired_sampler, seed, 11)
        source = _fresh(source_sampler, seed, 17)
        if (not callable(getattr(paired, "sample", None)) or not callable(getattr(source, "sample", None))
                or not isinstance(getattr(paired, "positive_weights", None), Tensor)):
            _invalid()

        input_trace, mask_trace = hashlib.sha256(), hashlib.sha256()
        updates = 0
        source_supported = False
        started = time.perf_counter()
        with torch.random.fork_rng(devices=[]):
            for step in range(_STEPS):
                pair_batch = _sample(paired, seed, step, 31)
                source_batch = _sample(source, seed, step, 37)
                if not isinstance(pair_batch, MaterializedBatch):
                    _invalid()
                positive_weight = paired.positive_weights
                _add_paired_input(input_trace, pair_batch, positive_weight)
                result = train_step_v3(candidate, teacher, optimizer, pair_batch, source_batch,
                                       step, age_mean, age_scale, seed, cbc, positive_weight,
                                       state_scale, source_enabled=role == "M")
                if not bool(result["optimizer_updated"]):
                    _invalid()
                updates += 1
                source_step = bool(result["source_generative_supervised"]) or bool(result["source_cbc_supervised"])
                source_supported = source_supported or source_step
                hashes = result["mask_hashes"]
                mask_trace.update(str(hashes.get("paired", "")).encode())
                mask_trace.update(str(hashes.get("paired_completion", "")).encode())
                if (step + 1) % _PROGRESS_INTERVAL == 0 and progress is not None:
                    progress(step + 1)
        elapsed = time.perf_counter() - started
        if (updates != _STEPS or source_supported != (role == "M")
                or not _unchanged(initial_before, initial) or not _unchanged(teacher_before, teacher)
                or initial.training != initial_mode or teacher.training != teacher_mode
                or tuple(parameter.requires_grad for parameter in initial.parameters()) != initial_grad
                or tuple(parameter.requires_grad for parameter in teacher.parameters()) != teacher_grad):
            _invalid()
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        _invalid()

    return {
        "model": candidate, "optimizer": optimizer, "paired_sampler": paired,
        "source_sampler": source, "elapsed_seconds": float(elapsed), "updates": updates,
        "source_gradient_supported": source_supported,
        "paired_input_digest": input_trace.hexdigest(),
        "paired_completion_mask_digest": mask_trace.hexdigest(),
    }
