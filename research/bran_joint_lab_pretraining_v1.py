"""Array-only chemistry-conditioned CBC warm-start helpers.

This is a prospective pretraining utility only.  It neither reads source data
nor trains endpoint, retinal, disease, or demographic components.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from bran_external_cbc_pretraining_v1 import (
    HierarchicalEpisodeSampler,
    SourceEpisodePool,
    export_clinical_warm_start,
    make_cbc_pretrain_head,
    make_cbc_pretrain_optimizer,
)
from bran_joint_lab_task_contract_v1 import (
    CBC_WIDTH,
    JOINT_WIDTH,
    REGISTRY_WIDTH,
    build_task_masks,
    masked_registry_inputs,
    project_joint_labs_to_registry,
)


_SOURCES = frozenset({"mimic", "eicu", "nwicu", "sicdb", "nhanes", "zigong"})


def _torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("PyTorch is required for joint laboratory pretraining") from error
    return torch, functional


def _readonly(array: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    copied = np.array(array, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


def _validate_source_arrays(source: object, item: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate one caller-owned source without retaining raw identifiers."""
    if not isinstance(source, str) or source not in _SOURCES or not isinstance(item, Mapping):
        raise ValueError("private joint source arrays are invalid")
    required = {"values", "observed", "provenance", "person_group", "split", "adult_qualified"}
    if set(item) != required:
        raise ValueError("private joint source arrays are invalid")
    values, observed, provenance = item["values"], item["observed"], item["provenance"]
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 2
        or values.shape[1] != JOINT_WIDTH
        or values.dtype.kind != "f"
        or not isinstance(observed, np.ndarray)
        or observed.shape != values.shape
        or observed.dtype != np.dtype(bool)
        or not isinstance(provenance, np.ndarray)
        or provenance.shape != values.shape
        or provenance.dtype.kind not in "iu"
    ):
        raise ValueError("private joint source arrays are invalid")
    n = values.shape[0]
    if n <= 0:
        raise ValueError("private joint source arrays are invalid")
    groups, split, adult = item["person_group"], item["split"], item["adult_qualified"]
    if (
        not isinstance(groups, np.ndarray)
        or groups.shape != (n,)
        or groups.dtype.kind not in "iu"
        or not isinstance(split, np.ndarray)
        or split.shape != (n,)
        or split.dtype.kind not in "iu"
        or not isinstance(adult, np.ndarray)
        or adult.shape != (n,)
        or adult.dtype != np.dtype(bool)
    ):
        raise ValueError("private joint source arrays are invalid")
    if (
        (groups.dtype.kind == "u" and np.any(groups > np.iinfo(np.int64).max))
        or (split.dtype.kind == "u" and np.any(split > np.iinfo(np.int64).max))
        or np.any(groups < 0)
        or np.any(split < 0)
        or np.any(split > 2)
    ):
        raise ValueError("private joint source arrays are invalid")
    normalized_groups = np.array(groups, dtype=np.int64, copy=True)
    normalized_split = np.array(split, dtype=np.int64, copy=True)
    order = np.argsort(normalized_groups, kind="stable")
    if np.any(
        (normalized_groups[order][1:] == normalized_groups[order][:-1])
        & (normalized_split[order][1:] != normalized_split[order][:-1])
    ):
        raise ValueError("person group has inconsistent split assignment")
    return values, observed, provenance, normalized_groups, normalized_split


@dataclass(frozen=True, repr=False)
class _ModeSource:
    projected_values: np.ndarray
    projected_mask: np.ndarray
    observed: np.ndarray


@dataclass(frozen=True, repr=False)
class _ModeSampler:
    sampler: HierarchicalEpisodeSampler
    sources: tuple[_ModeSource, ...]


def _prepare_mode(
    source_name: str,
    projected_values: np.ndarray,
    projected_mask: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
    split: np.ndarray,
    adult: np.ndarray,
    mode: str,
) -> tuple[SourceEpisodePool, _ModeSource] | None:
    cbc_count = observed[:, :CBC_WIDTH].sum(axis=1)
    if mode == "partial_cbc":
        task_eligible = cbc_count >= 2
    else:
        task_eligible = (cbc_count >= 2) & observed[:, CBC_WIDTH:].any(axis=1)
    eligible = task_eligible & (split == 0) & adult
    if not np.any(eligible):
        return None
    indices = np.flatnonzero(eligible)
    mode_observed = _readonly(observed[indices], bool)
    pool = SourceEpisodePool(
        source_name,
        _readonly(groups[indices], np.int64),
        np.zeros(len(indices), dtype=np.int64),
        np.ones(len(indices), dtype=bool),
        mode_observed[:, :CBC_WIDTH],
    )
    return pool, _ModeSource(
        _readonly(projected_values[indices], np.float64),
        _readonly(projected_mask[indices], bool),
        mode_observed,
    )


def _prepare_samplers(
    private_sources: object,
    registry_fields: tuple[str, ...],
    medians: np.ndarray,
    iqrs: np.ndarray,
) -> tuple[_ModeSampler, _ModeSampler]:
    if not isinstance(private_sources, Mapping) or not private_sources:
        raise ValueError("private joint source arrays are invalid")
    seen: set[str] = set()
    mode_pools: dict[str, list[SourceEpisodePool]] = {"whole_cbc": [], "partial_cbc": []}
    mode_sources: dict[str, list[_ModeSource]] = {"whole_cbc": [], "partial_cbc": []}
    for source_name, item in private_sources.items():
        if source_name in seen:
            raise ValueError("private joint source arrays are invalid")
        seen.add(source_name)
        values, observed, provenance, groups, split = _validate_source_arrays(source_name, item)
        adult = item["adult_qualified"]
        projected_values, projected_mask = project_joint_labs_to_registry(
            values, observed, provenance, registry_fields, medians, iqrs
        )
        for mode in ("whole_cbc", "partial_cbc"):
            prepared = _prepare_mode(
                source_name, projected_values, projected_mask, observed, groups, split, adult, mode
            )
            if prepared is not None:
                pool, mode_source = prepared
                mode_pools[mode].append(pool)
                mode_sources[mode].append(mode_source)
    if not mode_pools["whole_cbc"] or not mode_pools["partial_cbc"]:
        raise ValueError("joint task has no eligible source episodes")
    return (
        _ModeSampler(HierarchicalEpisodeSampler(mode_pools["whole_cbc"]), tuple(mode_sources["whole_cbc"])),
        _ModeSampler(HierarchicalEpisodeSampler(mode_pools["partial_cbc"]), tuple(mode_sources["partial_cbc"])),
    )


def _sample_batch(prepared: _ModeSampler, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.empty((batch_size, REGISTRY_WIDTH), dtype=np.float64)
    masks = np.empty((batch_size, REGISTRY_WIDTH), dtype=bool)
    observed = np.empty((batch_size, JOINT_WIDTH), dtype=bool)
    for row in range(batch_size):
        sample = prepared.sampler.sample(rng)
        source = prepared.sources[sample.source_index]
        values[row] = source.projected_values[sample.episode_index]
        masks[row] = source.projected_mask[sample.episode_index]
        observed[row] = source.observed[sample.episode_index]
    return values, masks, observed


def _registry_slots(registry_fields: tuple[str, ...]) -> tuple[int, ...]:
    if not isinstance(registry_fields, tuple) or len(registry_fields) != REGISTRY_WIDTH:
        raise ValueError("registry schema is invalid")
    from bran_joint_lab_cache_v1 import FIELDS
    if any(field not in registry_fields for field in FIELDS):
        raise ValueError("registry schema is invalid")
    slots = tuple(registry_fields.index(field) for field in FIELDS)
    if any(slot >= 48 for slot in slots):
        raise ValueError("registry schema is invalid")
    return slots


def _row_mean_smooth_l1(prediction, target, mask, functional):
    torch, _ = _torch()
    if prediction.shape != target.shape or mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("joint loss tensors are invalid")
    counts = mask.sum(dim=1)
    if torch.any(counts < 1):
        raise ValueError("joint loss mask is empty")
    raw = functional.smooth_l1_loss(
        torch.where(mask, prediction, torch.zeros_like(prediction)),
        torch.where(mask, target, torch.zeros_like(target)),
        reduction="none",
    )
    return ((raw * mask.to(raw.dtype)).sum(dim=1) / counts.to(raw.dtype)).mean()


def joint_pretrain_step(
    model,
    head,
    optimizer,
    projected_values: np.ndarray,
    projected_mask: np.ndarray,
    visible: np.ndarray,
    hidden: np.ndarray,
    registry_fields: tuple[str, ...],
    *,
    mode: str,
):
    """Update clinical encoder/residual and temporary head for one joint task batch."""
    torch, functional = _torch()
    if mode not in {"whole_cbc", "partial_cbc"}:
        raise ValueError("joint task mode is invalid")
    slots = _registry_slots(registry_fields)
    input_values, input_mask = masked_registry_inputs(
        projected_values, projected_mask, visible, hidden, registry_fields
    )
    observed_cbc_array = projected_mask[:, slots[:CBC_WIDTH]]
    observed_chemistry_array = projected_mask[:, slots[CBC_WIDTH:]]
    if np.any(observed_cbc_array.sum(axis=1) < 2):
        raise ValueError("joint batch has insufficient observed CBC")
    if mode == "whole_cbc" and np.any(~observed_chemistry_array.any(axis=1)):
        raise ValueError("whole CBC batch lacks observed chemistry")
    if (
        not hasattr(model, "config")
        or not hasattr(model, "clinical_encoder")
        or not hasattr(model, "clinical_residual")
        or model.config.clinical_dim != REGISTRY_WIDTH
        or model.config.hidden_dim != 128
    ):
        raise ValueError("model lacks compatible clinical modules")
    if (
        not isinstance(head, torch.nn.Linear)
        or head.weight.shape != (CBC_WIDTH, 128)
        or head.weight.dtype != next(model.clinical_encoder.parameters()).dtype
        or head.weight.device != next(model.clinical_encoder.parameters()).device
    ):
        raise ValueError("joint pretrain head is invalid")
    expected_parameters = list(model.clinical_encoder.parameters()) + list(model.clinical_residual.parameters()) + list(head.parameters())
    expected_ids = {id(parameter) for parameter in expected_parameters}
    actual_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    actual_ids = {id(parameter) for parameter in actual_parameters}
    if len(actual_parameters) != len(expected_parameters) or actual_ids != expected_ids:
        raise ValueError("optimizer must contain only joint warm-start parameters")
    other_parameters = [parameter for parameter in model.parameters() if id(parameter) not in expected_ids]
    before_expected = [parameter.detach().clone() for parameter in expected_parameters]
    before_other = [parameter.detach().clone() for parameter in other_parameters]
    dtype = next(model.clinical_encoder.parameters()).dtype
    device = next(model.clinical_encoder.parameters()).device
    values_tensor = torch.tensor(input_values, dtype=dtype, device=device)
    mask_tensor = torch.tensor(input_mask, dtype=torch.bool, device=device)
    target_tensor = torch.tensor(projected_values[:, slots[:CBC_WIDTH]], dtype=dtype, device=device)
    visible_tensor = torch.tensor(visible[:, :CBC_WIDTH], dtype=torch.bool, device=device)
    hidden_tensor = torch.tensor(hidden[:, :CBC_WIDTH], dtype=torch.bool, device=device)
    observed_cbc = torch.tensor(observed_cbc_array, dtype=torch.bool, device=device)
    if mode == "whole_cbc":
        if torch.any(visible_tensor) or not torch.equal(hidden_tensor, observed_cbc):
            raise ValueError("whole CBC task masks are invalid")
    elif torch.any(visible_tensor.sum(dim=1) < 1) or torch.any(hidden_tensor.sum(dim=1) < 1):
        raise ValueError("partial CBC task masks are invalid")
    age_disabled = torch.zeros((values_tensor.shape[0], 1), dtype=dtype, device=device)
    encoded = model.clinical_encoder(torch.cat([values_tensor, mask_tensor.to(dtype), age_disabled], dim=1))
    prediction = head(encoded + model.clinical_residual(values_tensor))
    hidden_loss = _row_mean_smooth_l1(prediction, target_tensor, hidden_tensor, functional)
    if mode == "whole_cbc":
        visible_loss = torch.zeros((), dtype=dtype, device=device)
        loss = hidden_loss
    else:
        visible_loss = _row_mean_smooth_l1(prediction, target_tensor, visible_tensor, functional)
        loss = hidden_loss + 0.1 * visible_loss
    if not torch.isfinite(loss):
        raise ValueError("joint pretrain loss is nonfinite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in expected_parameters):
        raise ValueError("joint pretrain gradient is nonfinite")
    if any(parameter.grad is not None for parameter in other_parameters):
        raise ValueError("joint pretrain touched nonclinical parameters")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in expected_parameters):
        with torch.no_grad():
            for parameter, prior in zip(expected_parameters, before_expected):
                parameter.copy_(prior)
        raise ValueError("joint pretrain update is nonfinite")
    if any(not torch.equal(parameter, prior) for parameter, prior in zip(other_parameters, before_other)):
        raise ValueError("joint pretrain modified nonclinical parameters")
    return {
        "loss": loss.detach(),
        "hidden_loss": hidden_loss.detach(),
        "visible_loss": visible_loss.detach(),
    }


def external_joint_warm_start(
    private_sources: Mapping[str, Mapping[str, np.ndarray]],
    registry_fields: tuple[str, ...],
    fold_medians: np.ndarray,
    fold_iqrs: np.ndarray,
    *,
    seed: int,
    steps: int = 3000,
    batch_size: int = 96,
) -> dict[str, object]:
    """Return only clinical warm-start weights after an equal whole/partial schedule."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 1701:
        raise ValueError("seed is invalid")
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps <= 0 or steps % 2:
        raise ValueError("steps must be positive and even")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
        raise ValueError("batch_size is invalid")
    whole, partial = _prepare_samplers(private_sources, registry_fields, fold_medians, fold_iqrs)
    torch, _ = _torch()
    torch.set_num_threads(2)
    torch.manual_seed(int(seed))
    from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
    from bran_patient_state_prototype_v1 import PatientStateConfig
    model = BRANClinicalAnchorV2(PatientStateConfig())
    head = make_cbc_pretrain_head(model)
    optimizer = make_cbc_pretrain_optimizer(model, head, learning_rate=0.001)
    rng = np.random.default_rng(19001 + (int(seed) - 1701))
    for step in range(int(steps)):
        mode = "whole_cbc" if step % 2 == 0 else "partial_cbc"
        prepared = whole if mode == "whole_cbc" else partial
        batch_values, batch_mask, batch_observed = _sample_batch(prepared, int(batch_size), rng)
        visible, hidden = build_task_masks(batch_observed, mode, rng)
        joint_pretrain_step(
            model, head, optimizer, batch_values, batch_mask, visible, hidden, registry_fields, mode=mode
        )
    return export_clinical_warm_start(model)
