"""V6 continuation: unchanged V5 recipe plus optional true-label bridge losses."""
from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from bran_multisource_continuation_v3 import _CLIP, _seed, _state_preservation
from bran_multisource_continuation_v4 import (
    _cap_alpha, _fail as _v4_fail, _gradients, _norm64, _set_merged_grads,
    _validate as _validate_v4,
)
from bran_multisource_model_v2 import BRANMultisourceModelV2
from bran_multisource_preservation_v5 import preservation
from bran_source_pattern_supervision_v6 import supervise
from bran_multisource_training_v2 import (
    MaterializedBatch, _age7, _cbc_completion_loss, _clinical_visible,
    _generator, _generative, _masked_digest, _route_masks, _screening_loss,
    _zero,
)


_INVALID = "multisource continuation v6 inputs invalid"


def _fail() -> None:
    raise ValueError(_INVALID)


def train_step_v6(
    model: BRANMultisourceModelV2, teacher: BRANMultisourceModelV2,
    optimizer: torch.optim.Optimizer, paired_batch: MaterializedBatch,
    source_batch: MaterializedBatch, step: int, age_mean: float, age_scale: float,
    seed: int, cbc_indices: Iterable[int], positive_weight: Tensor, state_scale: Tensor,
    *, bridge_enabled: bool,
) -> dict[str, object]:
    """Both arms consume source values; only S adds genuine paired bridge labels."""
    try:
        if type(bridge_enabled) is not bool:
            _fail()
        source_enabled = True
        cbc, state_scale = _validate_v4(model, teacher, optimizer, paired_batch, source_batch,
                                        step, age_mean, age_scale, seed, cbc_indices,
                                        positive_weight, state_scale, source_enabled)
        with torch.random.fork_rng(devices=[]):
            paired_age = _age7(paired_batch, age_mean, age_scale,
                               _generator(paired_batch.c.device, seed, step, 11))
            paired_mask_generator = _generator(paired_batch.c.device, seed, step, 23)
            visible_c, visible_r = _route_masks(paired_batch, step, paired_mask_generator)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(_seed(seed, step, 101))
                generative, generative_rows = _generative(model, paired_batch, paired_age,
                                                           visible_c, visible_r, step)
            screening, screening_rows = _screening_loss(model, paired_batch, paired_age,
                                                         visible_c, visible_r, positive_weight)
            completion, completion_rows, completion_c, completion_r, _ = _cbc_completion_loss(
                model, paired_batch, paired_age, cbc, step)
            state_preservation, state_preservation_rows = _state_preservation(
                model, teacher, paired_batch, paired_age, visible_c, visible_r, state_scale)
            prediction = preservation(model, teacher, paired_batch, source_batch, paired_age,
                                      visible_c, visible_r, step)
            prediction_loss = prediction["loss"]
            bridge = {'screening': _zero(model), 'cbc': _zero(model),
                      'screening_rows': 0, 'cbc_rows': 0}
            if bridge_enabled:
                bridge = supervise(model, paired_batch, source_batch, paired_age,
                                   visible_c, visible_r, step, positive_weight)
            bridge_loss = 0.5 * bridge['screening'] + 0.25 * bridge['cbc']
            paired_native = screening + 0.5 * completion + bridge_loss
            paired_total = (generative + screening + 0.5 * completion
                            + 0.1 * state_preservation + prediction_loss + bridge_loss)

            source_generative = _zero(model)
            source_completion = _zero(model)
            source_generative_rows = source_completion_rows = 0
            source_c = source_r = None
            source_weight = 0.0
            if source_enabled:
                source_age = _age7(source_batch, age_mean, age_scale,
                                   _generator(source_batch.c.device, seed, step, 31))
                source_mask_generator = _generator(source_batch.c.device, seed, step, 37)
                source_c = _clinical_visible(source_batch.cm, source_mask_generator, step)
                source_r = source_batch.rm.clone()
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(_seed(seed, step, 103))
                    source_generative, source_generative_rows = _generative(
                        model, source_batch, source_age, source_c, source_r, step)
                source_completion, source_completion_rows, _, _, _ = _cbc_completion_loss(
                    model, source_batch, source_age, cbc, step)
                source_weight = 0.1 * min(1.0, (step + 1) / 300.0)
            source_total = source_generative + 0.5 * source_completion
            total = paired_total + source_weight * source_total
            supported = bool(generative_rows or screening_rows or completion_rows
                             or state_preservation_rows or prediction["supported"]
                             or source_generative_rows or source_completion_rows)
            if not bool(torch.isfinite(total)):
                _fail()
            cap_applied = False
            cap_contract = True
            source_gen_nonzero = False
            source_cbc_nonzero = False
            if supported:
                optimizer.zero_grad(set_to_none=True)
                if not source_enabled:
                    total.backward()
                else:
                    parameters = tuple(model.parameters())
                    grad_pair = _gradients(paired_total, parameters, retain_graph=True)
                    grad_native = _gradients(paired_native, parameters, retain_graph=True)
                    grad_source_gen = _gradients(source_weight * source_generative, parameters, retain_graph=True)
                    grad_source_cbc = _gradients(source_weight * 0.5 * source_completion,
                                                 parameters, retain_graph=False)
                    alpha, cap_applied, cap_contract, source_gen_nonzero = _cap_alpha(
                        grad_native, grad_source_gen)
                    source_cbc_nonzero = _norm64(grad_source_cbc) > 0.0
                    _set_merged_grads(parameters, grad_pair, grad_source_gen, grad_source_cbc, alpha)
                torch.nn.utils.clip_grad_norm_(model.parameters(), _CLIP, error_if_nonfinite=True)
                optimizer.step()
    except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
        _fail()

    hashes = {"paired": _masked_digest(visible_c, visible_r),
              "paired_completion": _masked_digest(completion_c, completion_r)}
    if source_c is not None:
        hashes["source"] = _masked_digest(source_c, source_r)
    return {
        "step": step, "loss": float(total.detach()), "paired_loss": float(paired_total.detach()),
        "generative_loss": float(generative.detach()), "screening_loss": float(screening.detach()),
        "cbc_loss": float(completion.detach()), "state_preservation_loss": float(state_preservation.detach()),
        "source_loss": float(source_total.detach()),
        "source_generative_loss": float(source_generative.detach()),
        "source_cbc_loss": float(source_completion.detach()), "source_weight": source_weight,
        "generative_supervised": bool(generative_rows), "screening_supervised": bool(screening_rows),
        "cbc_supervised": bool(completion_rows),
        "state_preservation_supervised": bool(state_preservation_rows),
        "source_generative_supervised": bool(source_generative_rows),
        "source_cbc_supervised": bool(source_completion_rows),
        "source_enabled": source_enabled, "optimizer_updated": supported, "mask_hashes": hashes,
        "source_generative_cap_applied": cap_applied,
        "cap_contract_satisfied": cap_contract,
        "source_generative_gradient_nonzero": source_gen_nonzero,
        "source_cbc_gradient_nonzero": source_cbc_nonzero,
        "prediction_preservation_loss": float(prediction_loss.detach()),
        "prediction_preservation_supported": bool(prediction["supported"]),
        "bridge_mask_digest": prediction["bridge_mask_digest"],
        "bridge_enabled": bridge_enabled,
        "bridge_screening_supervised": bool(bridge["screening_rows"]),
        "bridge_cbc_supervised": bool(bridge["cbc_rows"]),
        "bridge_loss": float(bridge_loss.detach()),
    }

