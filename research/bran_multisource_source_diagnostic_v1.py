"""Private no-update source-gradient diagnostic for anchored BRAN V3.

All values returned here are scalar, per-batch private diagnostics.  This module
does no I/O, source admission, optimization, release decision, or promotion.
"""
from __future__ import annotations

from numbers import Real

import torch
from torch import Tensor

from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_gradient_geometry_v1 import gradient_summary
from bran_multisource_anchored_model_v3 import BRANMultisourceAnchoredModelV3
from bran_multisource_training_v2 import (
    MaterializedBatch, _age7, _cbc_completion_loss, _clinical_visible, _completion_masks,
    _generator, _generative, _require_unpaired, _route_masks, _screening_loss,
    _validate_batch, _zero,
)


_INVALID = "multisource source diagnostic inputs invalid"
_MAX_STEP = 2999


def _invalid() -> None:
    raise ValueError(_INVALID)


def _seed(seed: int, step: int, salt: int) -> int:
    return (seed + 1000003 * step + salt) % (2**63 - 1)


def _same_state(before: dict[str, Tensor], model: torch.nn.Module) -> bool:
    """Strictly verify this diagnostic did not alter persistent module state."""
    after = model.state_dict()
    return (before.keys() == after.keys()
            and all(torch.equal(value, after[name]) for name, value in before.items()))


def _finite_scalar(value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _invalid()
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or (positive and result <= 0):
        _invalid()
    return result


def _groups(model: BRANMultisourceAnchoredModelV3) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    encoder_prefixes = (
        "clinical_encoder.", "clinical_residual.", "retinal_projection.",
        "retinal_pool_gate.", "retinal_pool.", "retinal_residual.",
        "shared_posterior.", "retinal_prior.", "clinical_prior.",
        "retinal_delta.", "clinical_delta.",
    )
    encoder = []
    native = []
    for name, parameter in model.named_parameters():
        if name in ("retinal_prior_logvar", "clinical_prior_logvar") or name.startswith(encoder_prefixes):
            encoder.append(parameter)
        elif name.startswith("screening_joint_head.") or name.startswith("cbc_joint_head."):
            native.append(parameter)
    if not encoder or not native:
        _invalid()
    return {"encoder": tuple(encoder), "native_heads": tuple(native)}


def _hb_diagnostic(model: BRANMultisourceAnchoredModelV3, batch: MaterializedBatch,
                   age7: Tensor, step: int, hb_median: float, hb_iqr: float,
                   minimum_support: int) -> tuple[Tensor | None, Tensor | None, dict[str, object], dict[str, object]]:
    """Return hidden-Hb MAE losses and scalar original-unit diagnostics only."""
    try:
        hb_position = CBC_FIELDS.index("hemoglobin")
        hb_slot = model.cbc_indices[hb_position]
    except (ValueError, IndexError):
        _invalid()
    visible_c, visible_r = _completion_masks(batch, model.cbc_indices, step)
    erased_c = torch.where(visible_c, batch.c, torch.zeros_like(batch.c))
    with torch.no_grad():
        preliminary = model.encode(erased_c, visible_c, batch.r, visible_r, age7)
    keep = ~preliminary.abstain
    hidden_hb = batch.cm[:, hb_slot] & ~visible_c[:, hb_slot] & keep
    count = int(hidden_hb.sum())
    if count == 0:
        unsupported = {"supported": False, "count": 0, "abs_error_sum": None, "signed_error_sum": None}
        return None, None, unsupported, {**unsupported, "minimum_support_met": False}
    state = model.encode(erased_c[keep], visible_c[keep], batch.r[keep], visible_r[keep], age7[keep])
    prediction = model.cbc_joint_head(state.mean)[:, hb_position] * hb_iqr + hb_median
    target = batch.c[keep, hb_slot] * hb_iqr + hb_median
    retained_hidden = hidden_hb[keep]
    signed = prediction[retained_hidden] - target[retained_hidden]
    loss = signed.abs().mean()
    summary = {"supported": True, "count": count,
               "abs_error_sum": float(signed.detach().abs().sum()),
               "signed_error_sum": float(signed.detach().sum())}
    low = target[retained_hidden] < 12.0
    low_count = int(low.sum())
    if low_count < minimum_support:
        low_summary = {"supported": False, "count": low_count,
                       "abs_error_sum": None, "signed_error_sum": None,
                       "minimum_support_met": False}
        return loss, None, summary, low_summary
    low_signed = signed[low]
    low_summary = {"supported": True, "count": low_count,
                   "abs_error_sum": float(low_signed.detach().abs().sum()),
                   "signed_error_sum": float(low_signed.detach().sum()),
                   "minimum_support_met": True}
    return loss, low_signed.abs().mean(), summary, low_summary


def _validate(model: object, paired: object, source: object, step: object, seed: object,
              age_mean: object, age_scale: object, positive_weight: object,
              hb_median: object, hb_iqr: object, minimum_support: object) -> tuple[float, float, float, float, int]:
    if (not isinstance(model, BRANMultisourceAnchoredModelV3) or model.arm != "mlp" or model.training
            or any(not parameter.requires_grad for parameter in model.parameters())
            or isinstance(step, bool) or not isinstance(step, int) or step < 0 or step > _MAX_STEP
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or isinstance(minimum_support, bool) or not isinstance(minimum_support, int) or minimum_support < 20):
        _invalid()
    mean = _finite_scalar(age_mean)
    scale = _finite_scalar(age_scale, positive=True)
    median = _finite_scalar(hb_median)
    iqr = _finite_scalar(hb_iqr, positive=True)
    _validate_batch(paired)
    _validate_batch(source)
    dtype = next(model.parameters()).dtype
    if (paired.c.device.type != "cpu" or source.c.device.type != "cpu"
            or paired.labels is None or paired.labelmask is None
            or source.labels is not None or source.labelmask is not None
            or any(value.dtype != dtype for value in (
                paired.c, paired.r, paired.age.value, paired.age.lower, paired.age.upper,
                source.c, source.r, source.age.value, source.age.lower, source.age.upper))
            or not isinstance(positive_weight, Tensor) or not positive_weight.is_floating_point()
            or positive_weight.shape != (26,) or positive_weight.device != paired.c.device
            or not torch.isfinite(positive_weight).all() or bool((positive_weight <= 0).any())):
        _invalid()
    _require_unpaired(source, model)
    return mean, scale, median, iqr, minimum_support


def diagnostic(model: BRANMultisourceAnchoredModelV3, paired_batch: MaterializedBatch,
               source_batch: MaterializedBatch, *, step: int, seed: int, age_mean: float,
               age_scale: float, positive_weight: Tensor, hb_median: float, hb_iqr: float,
               minimum_support: int = 20) -> dict[str, object]:
    """Compute private source-vs-paired gradient geometry without updating model state."""
    try:
        mean, scale, median, iqr, minimum = _validate(
            model, paired_batch, source_batch, step, seed, age_mean, age_scale,
            positive_weight, hb_median, hb_iqr, minimum_support)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        before_mode = model.training
        before_grad = tuple(parameter.requires_grad for parameter in model.parameters())
        before_rng = torch.random.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            paired_age = _age7(paired_batch, mean, scale, _generator(paired_batch.c.device, seed, step, 11))
            paired_mask_generator = _generator(paired_batch.c.device, seed, step, 23)
            visible_c, visible_r = _route_masks(paired_batch, step, paired_mask_generator)
            screening, screening_rows = _screening_loss(model, paired_batch, paired_age,
                                                         visible_c, visible_r, positive_weight)
            paired_cbc, paired_cbc_rows, _, _, _ = _cbc_completion_loss(
                model, paired_batch, paired_age, model.cbc_indices, step)
            paired_hb, paired_low_hb, paired_hb_summary, paired_low_hb_summary = _hb_diagnostic(
                model, paired_batch, paired_age, step, median, iqr, minimum)

            source_age = _age7(source_batch, mean, scale, _generator(source_batch.c.device, seed, step, 31))
            source_mask_generator = _generator(source_batch.c.device, seed, step, 37)
            source_c = _clinical_visible(source_batch.cm, source_mask_generator, step)
            source_r = source_batch.rm.clone()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(_seed(seed, step, 103))
                source_generative, source_generative_rows = _generative(
                    model, source_batch, source_age, source_c, source_r, step)
            source_cbc, source_cbc_rows, _, _, _ = _cbc_completion_loss(
                model, source_batch, source_age, model.cbc_indices, step)
            _, _, source_hb_summary, source_low_hb_summary = _hb_diagnostic(
                model, source_batch, source_age, step, median, iqr, minimum)

            losses: dict[str, Tensor] = {}
            # Geometry only reports tasks with the caller's minimum support;
            # lower-support scalar summaries remain private in their Hb fields.
            if screening_rows >= minimum:
                losses["paired_screening"] = screening
            if paired_cbc_rows >= minimum:
                losses["paired_cbc"] = paired_cbc
            if paired_hb is not None and paired_hb_summary["count"] >= minimum:
                losses["paired_hb"] = paired_hb
            if paired_low_hb is not None:
                losses["paired_low_hb"] = paired_low_hb
            if source_generative_rows >= minimum:
                losses["source_generative"] = source_generative
            if source_cbc_rows >= minimum:
                losses["source_cbc"] = source_cbc
            geometry = (gradient_summary(losses, _groups(model), {name: 1.0 for name in losses})
                        if losses else None)
        if (not _same_state(before, model) or model.training != before_mode
                or tuple(parameter.requires_grad for parameter in model.parameters()) != before_grad
                or not torch.equal(before_rng, torch.random.get_rng_state())):
            _invalid()
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        _invalid()

    return {
        "step": step,
        "support": {"paired_screening": int(screening_rows), "paired_cbc": int(paired_cbc_rows),
                    "paired_hb": int(paired_hb_summary["count"]),
                    "paired_low_hb": int(paired_low_hb_summary["count"]),
                    "source_generative": int(source_generative_rows), "source_cbc": int(source_cbc_rows)},
        "geometry": ({"loss_values": {}, "groups": {}} if geometry is None else geometry),
        "hb": {"paired_hb": paired_hb_summary, "paired_low_hb": paired_low_hb_summary,
               "source_hb": source_hb_summary, "source_low_hb": source_low_hb_summary},
        "core_decides_release": False, "patient_level_output_emitted": False,
    }
