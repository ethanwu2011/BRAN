"""Array-only external CBC warm-start and paired V2 training kernel.

This module accepts caller-owned private arrays only. It does not discover
sources, read files, fit endpoints, or make clinical/scientific decisions.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from bran_cbc_event_adapter_v1 import CBC_FIELDS
from bran_external_cbc_pretraining_v1 import (
    HierarchicalEpisodeSampler, SourceEpisodePool, cbc_pretrain_step,
    export_clinical_warm_start, make_cbc_pretrain_head, make_cbc_pretrain_optimizer,
    mask_observed_cbc_rows, project_cbc_to_registry, transfer_clinical_warm_start,
)


def _torch():
    import torch
    return torch


def _source_arrays(private_sources: object) -> tuple[tuple[str, ...], tuple[Mapping[str, np.ndarray], ...], HierarchicalEpisodeSampler]:
    if not isinstance(private_sources, Mapping) or not private_sources:
        raise ValueError("private source arrays are invalid")
    names: list[str] = []
    arrays: list[Mapping[str, np.ndarray]] = []
    pools: list[SourceEpisodePool] = []
    required = {"values", "observed", "person_group", "split", "adult_qualified"}
    for source, item in private_sources.items():
        if not isinstance(source, str) or not isinstance(item, Mapping) or set(item) != required:
            raise ValueError("private source arrays are invalid")
        values, observed = item["values"], item["observed"]
        if not isinstance(values, np.ndarray) or values.ndim != 2 or values.shape[1] != len(CBC_FIELDS) or values.dtype.kind not in "iuf":
            raise ValueError("private source arrays are invalid")
        if not isinstance(observed, np.ndarray) or observed.shape != values.shape or observed.dtype != np.dtype(bool):
            raise ValueError("private source arrays are invalid")
        n = values.shape[0]
        if any(not isinstance(item[key], np.ndarray) or item[key].shape != (n,) for key in ("person_group", "split", "adult_qualified")):
            raise ValueError("private source arrays are invalid")
        pools.append(SourceEpisodePool(source, item["person_group"], item["split"], item["adult_qualified"], observed))
        names.append(source); arrays.append(item)
    return tuple(names), tuple(arrays), HierarchicalEpisodeSampler(pools)


def external_warm_start(
    private_sources: Mapping[str, Mapping[str, np.ndarray]],
    registry_fields: tuple[str, ...],
    fold_medians: np.ndarray,
    fold_iqrs: np.ndarray,
    *,
    seed: int,
    steps: int = 3000,
    batch_size: int = 96,
) -> dict[str, object]:
    """Train only external CBC encoder/residual weights from private arrays."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1701:
        raise ValueError("seed is invalid")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps is invalid")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size is invalid")
    names, arrays, sampler = _source_arrays(private_sources)
    torch = _torch()
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
    from bran_patient_state_prototype_v1 import PatientStateConfig
    model = BRANClinicalAnchorV2(PatientStateConfig())
    head = make_cbc_pretrain_head(model)
    optimizer = make_cbc_pretrain_optimizer(model, head, learning_rate=0.001)
    slots = tuple(registry_fields.index(field) for field in CBC_FIELDS) if isinstance(registry_fields, tuple) and all(field in registry_fields for field in CBC_FIELDS) else ()
    rng = np.random.default_rng(19001 + (seed - 1701))
    for _ in range(steps):
        selected = [sampler.sample(rng) for _ in range(batch_size)]
        batch_values = np.empty((batch_size, len(CBC_FIELDS)), dtype=np.float64)
        batch_observed = np.empty((batch_size, len(CBC_FIELDS)), dtype=bool)
        for row, sample in enumerate(selected):
            source_values = arrays[sample.source_index]["values"]
            source_observed = arrays[sample.source_index]["observed"]
            batch_values[row] = source_values[sample.episode_index]
            batch_observed[row] = source_observed[sample.episode_index]
        projected, projected_observed = project_cbc_to_registry(
            batch_values, batch_observed, registry_fields, fold_medians, fold_iqrs
        )
        visible, hidden = mask_observed_cbc_rows(batch_observed, rng)
        cbc_pretrain_step(
            model, head, optimizer, torch.tensor(projected, dtype=torch.float32),
            torch.tensor(projected_observed, dtype=torch.bool), torch.tensor(visible, dtype=torch.bool),
            torch.tensor(hidden, dtype=torch.bool), slots,
        )
    return export_clinical_warm_start(model)


def paired_train(
    clinical: np.ndarray,
    clinical_mask: np.ndarray,
    retinal: np.ndarray,
    retinal_mask: np.ndarray,
    age: np.ndarray,
    train_indices: np.ndarray,
    *,
    seed: int,
    steps: int = 1500,
    batch_size: int = 96,
    warm_start: Mapping[str, object] | None = None,
):
    """Current V2 paired recipe, optionally with external clinical warm-start."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0 or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("paired training configuration is invalid")
    torch = _torch()
    import run_bran_overnight_diagnostic_v1 as base
    from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
    from bran_patient_state_prototype_v1 import PatientStateConfig
    if not all(isinstance(value, np.ndarray) for value in (clinical, clinical_mask, retinal, retinal_mask, age, train_indices)):
        raise ValueError("paired training arrays are invalid")
    if clinical.ndim != 2 or clinical.shape[1] != 59 or clinical_mask.shape != clinical.shape or clinical_mask.dtype != np.dtype(bool):
        raise ValueError("paired training arrays are invalid")
    n = clinical.shape[0]
    if retinal.ndim != 2 or retinal.shape[0] != n or retinal.shape[1] != 384 or retinal_mask.shape != (n,) or retinal_mask.dtype != np.dtype(bool) or age.shape != (n,) or train_indices.ndim != 1 or train_indices.dtype.kind not in "iu" or not len(train_indices) or np.any(train_indices >= n) or np.any(train_indices<0):
        raise ValueError("paired training arrays are invalid")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BRANClinicalAnchorV2(PatientStateConfig())
    if warm_start is not None:
        transfer_clinical_warm_start(model, warm_start)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    ct = torch.tensor(clinical, dtype=torch.float32)
    cmt = torch.tensor(clinical_mask, dtype=torch.bool)
    rt = torch.tensor(retinal[:, None], dtype=torch.float32)
    rmt = torch.tensor(retinal_mask[:, None], dtype=torch.bool)
    at = torch.tensor(age, dtype=torch.float32)
    for step in range(steps):
        index = rng.choice(train_indices, batch_size, replace=len(train_indices) < batch_size)
        visible_clinical, visible_retinal_1d = base.masked_route(rng, clinical_mask[index], retinal_mask[index])
        visible_retinal = torch.tensor(visible_retinal_1d[:, None], dtype=torch.bool)
        vc = torch.tensor(visible_clinical, dtype=torch.bool)
        state = model.encode(ct[index] * vc, vc, rt[index] * visible_retinal[..., None], visible_retinal, at[index])
        loss = model.objective(
            state, at[index], ct[index], cmt[index], vc,
            rt[index, 0], rmt[index].expand(-1, retinal.shape[1]), visible_retinal.expand(-1, retinal.shape[1]),
            kl_weight=0.001 * min(1.0, (step + 1) / 300), visible_weight=0.1,
            clinical_eligible_mask=cmt[index],
        )["loss"]
        if not torch.isfinite(loss):
            raise ValueError("paired training produced nonfinite loss")
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("paired training produced nonfinite parameters")
    return model
