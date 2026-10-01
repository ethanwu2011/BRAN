"""Array-only external CBC warm-start utilities for BRANClinicalAnchorV2.

No source ingestion, fitting schedule, cohort admission, or scientific claims are
implemented here. Callers supply already-private arrays and control all use.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from bran_clinical_snapshot_v1 import CBC_FIELDS


REGISTRY_WIDTH = 59
_CANONICAL_SOURCES = frozenset({"mimic", "eicu", "nwicu", "sicdb", "nhanes", "zigong"})


def _boolean_array(value: object, shape: tuple[int, ...], error: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.dtype(bool):
        raise ValueError(error)
    return value


def project_cbc_to_registry(
    values: np.ndarray,
    observed: np.ndarray,
    registry_fields: tuple[str, ...],
    medians: np.ndarray,
    iqrs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Place canonical CBC values into a 59-slot registry without imputing them."""
    if not isinstance(values, np.ndarray) or values.ndim != 2 or values.shape[1] != len(CBC_FIELDS):
        raise ValueError("CBC values must have shape [rows, 9]")
    mask = _boolean_array(observed, values.shape, "CBC observed mask shape or dtype is invalid")
    if not isinstance(registry_fields, tuple) or len(registry_fields) != REGISTRY_WIDTH:
        raise ValueError("registry schema must be a 59-field tuple")
    if any(not isinstance(name, str) for name in registry_fields) or len(set(registry_fields)) != REGISTRY_WIDTH:
        raise ValueError("registry schema must contain unique field names")
    if any(field not in registry_fields for field in CBC_FIELDS):
        raise ValueError("registry schema is missing a CBC field")
    if any(registry_fields.index(field)>=48 for field in CBC_FIELDS):
        raise ValueError("CBC fields must occupy continuous registry slots")
    for scale, label in ((medians, "medians"), (iqrs, "iqrs")):
        if not isinstance(scale, np.ndarray) or scale.shape != (REGISTRY_WIDTH,) or scale.dtype.kind not in "iuf":
            raise ValueError(label + " must be a finite 59-vector")
        if not np.isfinite(scale).all() or (label == "iqrs" and np.any(scale <= 0)):
            raise ValueError(label + " must be a finite 59-vector")
    if values.dtype.kind not in "iuf":
        raise ValueError("CBC values must be numeric")
    input_values = values.astype(np.float64, copy=False)
    if np.any(mask & ~np.isfinite(input_values)):
        raise ValueError("observed CBC values must be finite")
    projected = np.zeros((values.shape[0], REGISTRY_WIDTH), dtype=np.float64)
    projected_mask = np.zeros((values.shape[0], REGISTRY_WIDTH), dtype=bool)
    for field_index, field in enumerate(CBC_FIELDS):
        slot = registry_fields.index(field)
        active = mask[:, field_index]
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = (input_values[active, field_index] - medians[slot]) / iqrs[slot]
        if not np.isfinite(normalized).all():
            raise ValueError("CBC normalization produced nonfinite values")
        projected[active, slot] = normalized
        projected_mask[:, slot] = active
    return projected, projected_mask


def mask_observed_cbc_rows(observed: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Split each >=2-observed CBC row into nonempty visible and hidden subsets."""
    if not isinstance(observed, np.ndarray) or observed.ndim != 2 or observed.shape[1] != len(CBC_FIELDS) or observed.dtype != np.dtype(bool):
        raise ValueError("CBC observed mask must have shape [rows, 9] and boolean dtype")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a NumPy Generator")
    if np.any(observed.sum(axis=1) < 2):
        raise ValueError("each CBC row must have at least two observed fields")
    visible = observed.copy()
    hidden = np.zeros_like(observed)
    for row in range(observed.shape[0]):
        candidates = np.flatnonzero(observed[row])
        hidden_count = int(rng.integers(1, len(candidates)))
        selected = rng.choice(candidates, size=hidden_count, replace=False)
        visible[row, selected] = False
        hidden[row, selected] = True
    return visible, hidden


@dataclass(frozen=True, repr=False)
class SourceEpisodePool:
    source: str
    person_group: np.ndarray
    split: np.ndarray
    adult_qualified: np.ndarray
    observed: np.ndarray


@dataclass(frozen=True, repr=False)
class SampledEpisode:
    source_index: int
    person_group: int
    episode_index: int


@dataclass(frozen=True, repr=False)
class _PreparedSource:
    source: str
    person_groups: np.ndarray
    eligible_episode_indices: tuple[np.ndarray, ...]


class HierarchicalEpisodeSampler:
    """Uniform source, then person, then eligible episode sampler without concatenation."""

    def __init__(self, pools: Iterable[SourceEpisodePool]):
        pool_tuple = tuple(pools)
        if not pool_tuple:
            raise ValueError("at least one source pool is required")
        source_names: set[str] = set()
        normalized: list[_PreparedSource] = []
        for pool in pool_tuple:
            if not isinstance(pool, SourceEpisodePool) or not isinstance(pool.source, str) or pool.source not in _CANONICAL_SOURCES or pool.source in source_names:
                raise ValueError("source pools are invalid")
            source_names.add(pool.source)
            normalized.append(self._validate_pool(pool))
        self._pools = tuple(normalized)

    @staticmethod
    def _validate_pool(pool: SourceEpisodePool) -> _PreparedSource:
        n = len(pool.person_group) if isinstance(pool.person_group, np.ndarray) and pool.person_group.ndim == 1 else -1
        if n <= 0 or not isinstance(pool.person_group, np.ndarray) or pool.person_group.dtype.kind not in "iu":
            raise ValueError("source pool arrays are invalid")
        if pool.person_group.dtype.kind == "u" and np.any(pool.person_group > np.iinfo(np.int64).max):
            raise ValueError("source pool arrays are invalid")
        if np.any(pool.person_group < 0):
            raise ValueError("source pool arrays are invalid")
        if not isinstance(pool.split, np.ndarray) or pool.split.shape != (n,) or pool.split.dtype.kind not in "iu":
            raise ValueError("source pool arrays are invalid")
        if pool.split.dtype.kind == "u" and np.any(pool.split > np.iinfo(np.int64).max):
            raise ValueError("source pool arrays are invalid")
        if np.any(pool.split < 0) or np.any(pool.split > 2):
            raise ValueError("source pool arrays are invalid")
        if not isinstance(pool.adult_qualified, np.ndarray) or pool.adult_qualified.shape != (n,) or pool.adult_qualified.dtype != np.dtype(bool):
            raise ValueError("source pool arrays are invalid")
        if not isinstance(pool.observed, np.ndarray) or pool.observed.shape != (n, len(CBC_FIELDS)) or pool.observed.dtype != np.dtype(bool):
            raise ValueError("source pool arrays are invalid")
        groups = np.array(pool.person_group, dtype=np.int64, copy=True)
        split = np.array(pool.split, dtype=np.int64, copy=True)
        order=np.argsort(groups,kind='stable')
        ordered_groups,ordered_split=groups[order],split[order]
        if np.any((ordered_groups[1:]==ordered_groups[:-1]) & (ordered_split[1:]!=ordered_split[:-1])):
            raise ValueError("person group has inconsistent split assignment")
        eligible = (split == 0) & pool.adult_qualified & (pool.observed.sum(axis=1) >= 2)
        if not np.any(eligible):
            raise ValueError("source pool has no eligible episodes")
        eligible_order=order[eligible[order]]
        eligible_groups=groups[eligible_order]
        boundaries=np.r_[0,np.flatnonzero(np.diff(eligible_groups))+1,len(eligible_order)]
        people=eligible_groups[boundaries[:-1]].copy()
        # Sort/group once in O(N log N), not one full-array scan per person.
        episodes=tuple(eligible_order[a:b].copy() for a,b in zip(boundaries[:-1],boundaries[1:]))
        people.setflags(write=False)
        for indices in episodes: indices.setflags(write=False)
        return _PreparedSource(pool.source, people, episodes)

    def sample(self, rng: np.random.Generator) -> SampledEpisode:
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a NumPy Generator")
        source_index = int(rng.integers(len(self._pools)))
        pool = self._pools[source_index]
        person_index = int(rng.integers(len(pool.person_groups)))
        person = int(pool.person_groups[person_index])
        episodes = pool.eligible_episode_indices[person_index]
        episode = int(episodes[int(rng.integers(len(episodes)))])
        return SampledEpisode(source_index, person, episode)


def _torch():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("PyTorch is required for encoder pretraining") from error
    return torch, nn, functional


def make_cbc_pretrain_head(model):
    """Make the temporary 128-to-9 readout; it is never exported."""
    torch, nn, _ = _torch()
    if not hasattr(model, "config") or not hasattr(model, "clinical_encoder") or not hasattr(model, "clinical_residual"):
        raise ValueError("model lacks compatible clinical modules")
    if model.config.clinical_dim != REGISTRY_WIDTH or model.config.hidden_dim != 128:
        raise ValueError("model is not compatible with the 59-slot CBC warm start")
    parameter = next(model.clinical_encoder.parameters(), None)
    if parameter is None:
        raise ValueError("model lacks compatible clinical modules")
    return nn.Linear(128, len(CBC_FIELDS), device=parameter.device, dtype=parameter.dtype)


def make_cbc_pretrain_optimizer(model, head, *, learning_rate: float):
    torch, _, _ = _torch()
    if not isinstance(learning_rate, float) or not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    params = list(model.clinical_encoder.parameters()) + list(model.clinical_residual.parameters()) + list(head.parameters())
    return torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0)


def _torch_masks(values, observed, visible, hidden, cbc_slots):
    torch, _, _ = _torch()
    if not isinstance(values, torch.Tensor) or values.ndim != 2 or values.shape[1] != REGISTRY_WIDTH:
        raise ValueError("registry values must have shape [batch, 59]")
    if any(not isinstance(mask, torch.Tensor) for mask in (observed, visible, hidden)):
        raise ValueError("CBC masks must be tensors")
    if observed.shape != values.shape or visible.shape != (values.shape[0], len(CBC_FIELDS)) or hidden.shape != visible.shape:
        raise ValueError("CBC mask shape mismatch")
    if observed.dtype != torch.bool or visible.dtype != torch.bool or hidden.dtype != torch.bool:
        raise ValueError("CBC masks must be boolean")
    if not isinstance(cbc_slots, tuple) or len(cbc_slots) != len(CBC_FIELDS) or len(set(cbc_slots)) != len(cbc_slots):
        raise ValueError("CBC slots are invalid")
    if any(isinstance(slot, bool) or not isinstance(slot, Integral) or not 0 <= slot < 48 for slot in cbc_slots):
        raise ValueError("CBC slots are invalid")
    slot_tensor = torch.tensor(cbc_slots, device=values.device)
    target_observed = observed.index_select(1, slot_tensor)
    if not torch.equal(target_observed, visible | hidden) or torch.any(visible & hidden) or torch.any(visible.sum(1) < 1) or torch.any(hidden.sum(1) < 1):
        raise ValueError("CBC visible and hidden masks are invalid")
    if torch.any(target_observed & ~torch.isfinite(values.index_select(1, slot_tensor))):
        raise ValueError("observed CBC targets must be finite")
    non_cbc = torch.ones(REGISTRY_WIDTH, dtype=torch.bool, device=values.device)
    non_cbc[slot_tensor] = False
    if torch.any(observed[:, non_cbc]):
        raise ValueError("CBC pretraining does not accept observed non-CBC fields")
    return slot_tensor


def _row_masked_smooth_l1(prediction, target, mask, functional):
    """Mean selected-field loss within each row, then mean rows equally."""
    torch,_,_=_torch()
    per_field = functional.smooth_l1_loss(torch.where(mask,prediction,torch.zeros_like(prediction)),
        torch.where(mask,target,torch.zeros_like(target)),reduction="none")
    counts = mask.sum(dim=1)
    if np.any(counts.detach().cpu().numpy() < 1):
        raise ValueError("CBC loss mask is empty")
    return ((per_field * mask.to(per_field.dtype)).sum(dim=1) / counts.to(per_field.dtype)).mean()


def cbc_pretrain_step(model, head, optimizer, values, observed, visible, hidden, cbc_slots: tuple[int, ...]):
    """One age-free masked-CBC update limited to clinical encoder/residual and head."""
    torch, _, functional = _torch()
    slot_tensor = _torch_masks(values, observed, visible, hidden, cbc_slots)
    if not isinstance(head, torch.nn.Linear) or head.weight.shape != (len(CBC_FIELDS), 128):
        raise ValueError("CBC pretrain head is invalid")
    expected = {id(p) for p in list(model.clinical_encoder.parameters()) + list(model.clinical_residual.parameters()) + list(head.parameters())}
    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if actual != expected:
        raise ValueError("optimizer must contain only CBC warm-start parameters")
    visible_registry = observed.clone()
    visible_registry[:, slot_tensor] = visible
    clean = torch.where(visible_registry & torch.isfinite(values), values, torch.zeros_like(values))
    age_disabled = torch.zeros((values.shape[0], 1), device=values.device, dtype=values.dtype)
    h = model.clinical_encoder(torch.cat([clean, visible_registry.to(values.dtype), age_disabled], dim=1))
    h = h + model.clinical_residual(clean)
    prediction = head(h)
    target = values.index_select(1, slot_tensor)
    hidden_loss = _row_masked_smooth_l1(prediction, target, hidden, functional)
    visible_loss = _row_masked_smooth_l1(prediction, target, visible, functional)
    loss = hidden_loss + 0.1 * visible_loss
    if not torch.isfinite(loss):
        raise ValueError("CBC pretrain loss is nonfinite")
    parameters = list(model.clinical_encoder.parameters()) + list(model.clinical_residual.parameters()) + list(head.parameters())
    before_update = [parameter.detach().clone() for parameter in parameters]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in parameters):
        raise ValueError("CBC pretrain gradient is nonfinite")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in parameters):
        with torch.no_grad():
            for parameter, previous in zip(parameters, before_update):
                parameter.copy_(previous)
        raise ValueError("CBC pretrain update is nonfinite")
    return {"loss": loss.detach(), "hidden_loss": hidden_loss.detach(), "visible_loss": visible_loss.detach()}


def export_clinical_warm_start(model) -> dict[str, object]:
    """Export only compatible clinical modules; the age column is handled on transfer."""
    torch, _, _ = _torch()
    selected = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith("clinical_encoder.") or name.startswith("clinical_residual.")
    }
    if not selected or any(not isinstance(value, torch.Tensor) for value in selected.values()):
        raise ValueError("model lacks compatible clinical modules")
    return selected


def transfer_clinical_warm_start(recipient, warm_start: Mapping[str, object]) -> None:
    """Copy only clinical modules, retaining recipient's age-input encoder column."""
    torch, _, _ = _torch()
    if not isinstance(warm_start, Mapping):
        raise ValueError("warm start is invalid")
    target = {
        name: value for name, value in recipient.state_dict().items()
        if name.startswith("clinical_encoder.") or name.startswith("clinical_residual.")
    }
    if set(warm_start) != set(target):
        raise ValueError("warm start keys do not match recipient")
    validated: list[tuple[str, object, object]] = []
    for name, destination in target.items():
        source = warm_start[name]
        if not isinstance(source, torch.Tensor) or source.shape != destination.shape or source.dtype != destination.dtype:
            raise ValueError("warm start shapes do not match recipient")
        if not torch.isfinite(source).all():
            raise ValueError("warm start contains nonfinite values")
        if name == "clinical_encoder.0.weight" and source.shape[1] < 2:
            raise ValueError("warm start shapes do not match recipient")
        validated.append((name, destination, source))
    with torch.no_grad():
        state = recipient.state_dict()
        for name, destination, source in validated:
            if name == "clinical_encoder.0.weight":
                state[name][:, :-1].copy_(source[:, :-1])
            else:
                state[name].copy_(source)
