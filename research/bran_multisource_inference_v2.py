"""Pure, native-head inference for the synthetic BRAN multisource V2 model.

This module intentionally has no source, checkpoint, serialization, or output
I/O.  It accepts already-standardized physiology only; typed ages remain in
original units until the supplied fold transform is applied locally.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn

from bran_multisource_age_v2 import AgeBatch, normalize_age, validate_age
from bran_multisource_model_v2 import BRANMultisourceModelV2


_INVALID = "multisource inference inputs invalid"
_ROUTES = ("both", "clinical", "retinal")
_PATTERNS = (
    "single_target_hidden",
    "whole_cbc_hidden",
    "red_cell_hidden",
    "single_target_no_retina",
    "whole_cbc_no_retina",
    "red_cell_no_retina",
)
_RED_CELL_POSITIONS = (0, 1, 2, 3, 4, 6)


@dataclass(frozen=True, repr=False)
class NativeInferenceV2:
    """Detached native-head outputs; abstained rows are represented as NaN."""

    screening_probability: Tensor  # [N, 26]
    cbc_standardized: Tensor       # [N, 9]
    abstained: Tensor              # [N] bool


@dataclass(frozen=True, repr=False)
class CompletionPredictionsV2:
    """Detached completion predictions with support masks, never posterior state."""

    cbc_standardized: Tensor       # [N, 9], NaN outside completion targets
    targetmask: Tensor             # [N, 9], original observed-and-erased fields
    scoring_target_mask: Tensor    # [N, 9], target_mask excluding abstained encodes
    abstained: Tensor              # [N, 9], target-specific where required

    @property
    def target_mask(self) -> Tensor:
        """Spelled-out compatibility accessor for the scoring-support mask."""

        return self.targetmask

    @property
    def available(self) -> Tensor:
        """Target-specific availability; callers should still use targetmask."""

        return ~self.abstained


# The concise name is provided for outcome evaluators; retain the descriptive
# original class spelling for callers that import it directly.
NativePredictions = NativeInferenceV2


def _invalid() -> None:
    raise ValueError(_INVALID)


def _indices(cbc_indices: Iterable[int], model: BRANMultisourceModelV2) -> tuple[int, ...]:
    try:
        indices = tuple(cbc_indices)
    except TypeError:
        _invalid()
    if (len(indices) != 9 or len(set(indices)) != 9
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 or x >= 48 for x in indices)
            or indices != model.cbc_indices):
        _invalid()
    return indices


def _model_ok(model: BRANMultisourceModelV2) -> None:
    if not isinstance(model, BRANMultisourceModelV2):
        _invalid()
    config = getattr(model, "config", None)
    if (config is None or config.state_dim != 192 or config.clinical_dim != 59
            or config.retinal_feature_dim != 384 or config.hidden_dim != 128
            or not isinstance(model.screening_joint_head, nn.Linear)
            or not isinstance(model.cbc_joint_head, nn.Linear)
            or model.screening_joint_head.in_features != 192 or model.screening_joint_head.out_features != 26
            or model.cbc_joint_head.in_features != 192 or model.cbc_joint_head.out_features != 9
            or len(model.eligible_indices) != 43 or len(set(model.eligible_indices)) != 43
            or len(model.cbc_indices) != 9 or not set(model.cbc_indices).issubset(model.eligible_indices)):
        _invalid()
    try:
        parameters = tuple(model.parameters())
    except (AttributeError, TypeError):
        _invalid()
    if not parameters or any(not torch.isfinite(parameter).all() for parameter in parameters):
        _invalid()


def _validate_structure(model: BRANMultisourceModelV2, clinical: Tensor, cm: Tensor,
                        retinal: Tensor, rm: Tensor, age: AgeBatch, batch_size: int) -> None:
    _model_ok(model)
    if (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
            or not isinstance(clinical, Tensor) or not clinical.is_floating_point()
            or clinical.ndim != 2 or clinical.shape[1] != 59
            or not isinstance(cm, Tensor) or cm.dtype != torch.bool or cm.shape != clinical.shape
            or not isinstance(retinal, Tensor) or not retinal.is_floating_point()
            or retinal.shape != (clinical.shape[0], 384)
            or not isinstance(rm, Tensor) or rm.dtype != torch.bool or rm.shape != (clinical.shape[0],)):
        _invalid()
    if any(item.device != clinical.device for item in (cm, retinal, rm, age.value, age.lower, age.upper, age.kind)):
        _invalid()
    validate_age(age)
    if age.value.shape != (clinical.shape[0],):
        _invalid()
    try:
        parameter_device = next(model.parameters()).device
    except StopIteration:
        _invalid()
    if parameter_device != clinical.device:
        _invalid()
    # Value validity is deliberately separate: routes and completion contexts
    # may make an originally marked field inaccessible before encoding.


def _validate_visible_values(model: BRANMultisourceModelV2, clinical: Tensor, cm: Tensor,
                             retinal: Tensor, rm: Tensor) -> None:
    # V2 permanently disables every binary and ineligible clinical field. Their
    # payload cannot affect inference and is intentionally not inspected.
    eligible = model.eligible_slots.to(device=clinical.device)[None, :]
    effective_clinical = cm & eligible
    if not torch.isfinite(clinical[effective_clinical]).all() or not torch.isfinite(retinal[rm]).all():
        _invalid()


def _route_masks(cm: Tensor, rm: Tensor, route: str) -> tuple[Tensor, Tensor]:
    if route not in _ROUTES:
        _invalid()
    clinical_mask, retinal_mask = cm.clone(), rm.clone()
    if route == "clinical":
        retinal_mask.zero_()
    elif route == "retinal":
        clinical_mask.zero_()
    return clinical_mask, retinal_mask


def _clean_for_encode(model: BRANMultisourceModelV2, clinical: Tensor, cm: Tensor,
                      retinal: Tensor, rm: Tensor) -> tuple[Tensor, Tensor]:
    """Remove inaccessible payloads before dispatch, without mutating callers."""

    eligible = model.eligible_slots.to(device=clinical.device)[None, :]
    effective = cm & eligible
    clean_clinical = torch.where(effective, clinical, torch.zeros_like(clinical))
    clean_retinal = torch.where(rm[:, None], retinal, torch.zeros_like(retinal))
    return clean_clinical, clean_retinal[:, None, :]


def _infer_normalized(model: BRANMultisourceModelV2, clinical: Tensor, cm: Tensor,
                      retinal: Tensor, rm: Tensor, age7: Tensor, batch_size: int) -> NativeInferenceV2:
    """Run a normalized age batch through only the two native state-only heads."""

    total = clinical.shape[0]
    screen_chunks: list[Tensor] = []
    cbc_chunks: list[Tensor] = []
    abstained_chunks: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, total, batch_size):
            stop = min(total, start + batch_size)
            c, r = _clean_for_encode(model, clinical[start:stop], cm[start:stop], retinal[start:stop], rm[start:stop])
            state = model.encode(c, cm[start:stop], r, rm[start:stop, None], age7[start:stop])
            if (state.mean.shape != (stop - start, 192) or state.abstain.shape != (stop - start,)
                    or state.abstain.dtype != torch.bool or not torch.isfinite(state.mean[~state.abstain]).all()):
                _invalid()
            screening = torch.sigmoid(model.screening_joint_head(state.mean))
            cbc = model.cbc_joint_head(state.mean)
            active = ~state.abstain
            if (not torch.isfinite(screening[active]).all() or not torch.isfinite(cbc[active]).all()
                    or (screening[active] < 0).any() or (screening[active] > 1).any()):
                _invalid()
            nan_screen = torch.full_like(screening, float("nan"))
            nan_cbc = torch.full_like(cbc, float("nan"))
            screen_chunks.append(torch.where(active[:, None], screening, nan_screen))
            cbc_chunks.append(torch.where(active[:, None], cbc, nan_cbc))
            abstained_chunks.append(state.abstain)
    return NativeInferenceV2(torch.cat(screen_chunks, dim=0).detach().clone(),
                             torch.cat(cbc_chunks, dim=0).detach().clone(),
                             torch.cat(abstained_chunks, dim=0).detach().clone())


def infer_native(model: BRANMultisourceModelV2, clinical: Tensor, cm: Tensor, retinal: Tensor, rm: Tensor,
                 age: AgeBatch, age_mean: float, age_scale: float, batch_size: int = 256,
                 *, route: str = "both") -> NativeInferenceV2:
    """Infer native screening/CBC heads using posterior means only.

    ``clinical`` and ``retinal`` are already recipient-fold standardized.  Age
    is still the typed, original-unit :class:`AgeBatch`; no augmentation occurs
    during inference.  Inaccessible values are zeroed before model dispatch.
    """

    _validate_structure(model, clinical, cm, retinal, rm, age, batch_size)
    clinical_mask, retinal_mask = _route_masks(cm, rm, route)
    _validate_visible_values(model, clinical, clinical_mask, retinal, retinal_mask)
    age7 = normalize_age(age, age_mean, age_scale)
    return _infer_normalized(model, clinical, clinical_mask, retinal, retinal_mask, age7, batch_size)


def infer_routes(model: BRANMultisourceModelV2, clinical: Tensor, cm: Tensor, retinal: Tensor, rm: Tensor,
                 age: AgeBatch, age_mean: float, age_scale: float, batch_size: int = 256) -> Mapping[str, NativeInferenceV2]:
    """Return the three deterministic mask-routed native predictions."""

    return MappingProxyType({route: infer_native(model, clinical, cm, retinal, rm, age, age_mean, age_scale,
                                                  batch_size, route=route)
                             for route in _ROUTES})


def route_predictions(model: BRANMultisourceModelV2, clinical: Tensor, cm: Tensor, retinal: Tensor, rm: Tensor,
                      age: AgeBatch, age_mean: float, age_scale: float,
                      batch_size: int = 256) -> Mapping[str, NativePredictions]:
    """Named route API: immutable mapping for ``both``, ``clinical``, ``retinal``."""

    return infer_routes(model, clinical, cm, retinal, rm, age, age_mean, age_scale, batch_size)


def _pattern_positions(pattern: str) -> tuple[int, ...]:
    if pattern not in _PATTERNS:
        _invalid()
    if pattern.startswith("single_target") or pattern.startswith("whole_cbc"):
        return tuple(range(9))
    return _RED_CELL_POSITIONS


def _pattern_no_retina(pattern: str) -> bool:
    return pattern.endswith("_no_retina")


def _masked_completion_inputs(clinical: Tensor, cm: Tensor, cbc: tuple[int, ...], positions: tuple[int, ...]) -> tuple[Tensor, Tensor]:
    """Erase selected CBC values and flags, preserving all other context."""

    visible = cm.clone()
    target_columns = torch.tensor([cbc[position] for position in positions], device=clinical.device, dtype=torch.long)
    visible.index_fill_(1, target_columns, False)
    erased = torch.where(visible, clinical, torch.zeros_like(clinical))
    return erased, visible


def completion_predictions(model: BRANMultisourceModelV2, clinical: Tensor, cm: Tensor,
                           retinal: Tensor, rm: Tensor, age: AgeBatch, age_mean: float, age_scale: float,
                           pattern: str, cbc_indices: Iterable[int], batch_size: int = 256) -> CompletionPredictionsV2:
    """Predict observed CBC fields after pattern-specific value-and-flag erasure.

    Single-target contexts run one encode per CBC target, so every prediction
    is conditioned on the other eight CBC observations.  Whole-CBC and red-cell
    contexts use one shared encode.  Only original observed-and-erased targets
    are exposed as scoring support; all non-target output cells are NaN.
    """

    _validate_structure(model, clinical, cm, retinal, rm, age, batch_size)
    cbc = _indices(cbc_indices, model)
    selected = _pattern_positions(pattern)
    no_retina = _pattern_no_retina(pattern)
    targets = cm[:, cbc]
    predictions = torch.full((clinical.shape[0], 9), float("nan"), device=clinical.device, dtype=clinical.dtype)
    abstained = torch.ones((clinical.shape[0], 9), device=clinical.device, dtype=torch.bool)

    if pattern.startswith("single_target"):
        # A target-specific encode is essential: hiding all nine would silently
        # turn the single-target context into whole-CBC completion.
        for position in selected:
            erased, visible = _masked_completion_inputs(clinical, cm, cbc, (position,))
            result = infer_native(model, erased, visible, retinal, torch.zeros_like(rm) if no_retina else rm,
                                  age, age_mean, age_scale, batch_size)
            present = ~result.abstained
            predictions[present, position] = result.cbc_standardized[present, position]
            abstained[:, position] = result.abstained
    else:
        erased, visible = _masked_completion_inputs(clinical, cm, cbc, selected)
        result = infer_native(model, erased, visible, retinal, torch.zeros_like(rm) if no_retina else rm,
                              age, age_mean, age_scale, batch_size)
        present = ~result.abstained
        for position in selected:
            predictions[present, position] = result.cbc_standardized[present, position]
            abstained[:, position] = result.abstained

    selector = torch.zeros((9,), dtype=torch.bool, device=clinical.device)
    selector[list(selected)] = True
    target_mask = targets & selector[None, :]
    predictions = torch.where(target_mask, predictions, torch.full_like(predictions, float("nan")))
    scoring_target_mask = target_mask & ~abstained
    return CompletionPredictionsV2(predictions.detach().clone(), target_mask.detach().clone(),
                                   scoring_target_mask.detach().clone(), abstained.detach().clone())
