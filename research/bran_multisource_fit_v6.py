"""Private V6 C/S continuation lifecycle from an authenticated V5 parent."""
from __future__ import annotations

import copy
import hashlib
import time
from typing import Callable, Optional

import torch
from torch import Tensor

from bran_multisource_continuation_v6 import train_step_v6
from bran_multisource_fit_v3 import _add_paired_input, _fresh, _sample, _tensor_digest, _unchanged, _validate as _validate_v3
from bran_multisource_model_v2 import BRANMultisourceModelV2
from bran_multisource_training_v2 import MaterializedBatch


_INVALID = "multisource fit v6 inputs invalid"
_STEPS = 3000
_PROGRESS_INTERVAL = 300
_LR = 5e-5
_WEIGHT_DECAY = 1e-4


def _invalid() -> None:
    raise ValueError(_INVALID)


def fit_one_v6(initial: BRANMultisourceModelV2, teacher: BRANMultisourceModelV2,
               paired_sampler: Callable[[], object], source_sampler: Callable[[], object],
               state_scale: Tensor, age_mean: float, age_scale: float, seed: int,
               cbc_indices, role: str, progress: Optional[Callable[[int], None]] = None, *, updates: int = 3000) -> dict[str, object]:
    """Fit C=ordinary V5-M continuation or S=additional source-pattern supervision."""
    try:
        if role not in ("C", "S") or type(updates) is not int or updates not in (100, 3000):
            _invalid()
        budget = updates
        cbc = _validate_v3(initial, teacher, paired_sampler, source_sampler, state_scale,
                           age_mean, age_scale, seed, cbc_indices, "M", progress)
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
        paired, source = _fresh(paired_sampler, seed, 11), _fresh(source_sampler, seed, 17)
        if (not callable(getattr(paired, "sample", None)) or not callable(getattr(source, "sample", None))
                or not isinstance(getattr(paired, "positive_weights", None), Tensor)):
            _invalid()

        input_trace, mask_trace, bridge_trace = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        source_values, source_masks = hashlib.sha256(), hashlib.sha256()
        bridge_screen_updates = bridge_cbc_updates = 0
        updates = cap_applied = cap_checks = nonzero_source_gen = nonzero_source_cbc = preservation_updates = 0
        source_supported = False
        started = time.perf_counter()
        with torch.random.fork_rng(devices=[]):
            for step in range(budget):
                pair_batch, source_batch = _sample(paired, seed, step, 31), _sample(source, seed, step, 37)
                if not isinstance(pair_batch, MaterializedBatch):
                    _invalid()
                positive_weight = paired.positive_weights
                _add_paired_input(input_trace, pair_batch, positive_weight)
                for value in (source_batch.c, source_batch.r, source_batch.age.value,
                              source_batch.age.lower, source_batch.age.upper, source_batch.age.kind):
                    _tensor_digest(source_values, value)
                for value in (source_batch.cm, source_batch.rm):
                    _tensor_digest(source_masks, value)
                result = train_step_v6(candidate, teacher, optimizer, pair_batch, source_batch,
                                       step, age_mean, age_scale, seed, cbc, positive_weight,
                                       state_scale, bridge_enabled=role == "S")
                if not bool(result["optimizer_updated"]):
                    _invalid()
                updates += 1
                bridge_screen_updates += int(result["bridge_screening_supervised"])
                bridge_cbc_updates += int(result["bridge_cbc_supervised"])
                source_step = bool(result["source_generative_supervised"]) or bool(result["source_cbc_supervised"])
                source_supported = source_supported or source_step
                cap_applied += int(bool(result["source_generative_cap_applied"]))
                cap_checks += int(bool(result["cap_contract_satisfied"]))
                nonzero_source_gen += int(bool(result["source_generative_gradient_nonzero"]))
                nonzero_source_cbc += int(bool(result["source_cbc_gradient_nonzero"]))
                preservation_updates += int(bool(result["prediction_preservation_supported"]))
                hashes = result["mask_hashes"]
                mask_trace.update(str(hashes.get("paired", "")).encode())
                mask_trace.update(str(hashes.get("paired_completion", "")).encode())
                bridge_trace.update(str(result["bridge_mask_digest"]).encode())
                if ((step + 1) % _PROGRESS_INTERVAL == 0 or step + 1 == budget) and progress is not None:
                    progress(step + 1)
        elapsed = time.perf_counter() - started
        if (updates != budget or not source_supported
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
        "bridge_mask_digest": bridge_trace.hexdigest(),
        "source_values_digest": source_values.hexdigest(),
        "source_availability_digest": source_masks.hexdigest(),
        "algorithm_update_counters": {
            "cap_applied_updates": cap_applied,
            "cap_contract_checks": cap_checks,
            "source_generative_nonzero_updates": nonzero_source_gen,
            "source_cbc_nonzero_updates": nonzero_source_cbc,
            "preservation_supported_updates": preservation_updates,
            "bridge_screen_supervised_updates": bridge_screen_updates,
            "bridge_cbc_supervised_updates": bridge_cbc_updates,
        },
    }

