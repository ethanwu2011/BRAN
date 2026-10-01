"""Prospective-only masked/observed-label retinal adaptation step kernel.

This module has no sampler, optimizer, source I/O, schedule, evaluation, or
promotion policy.  Callers supply already-authorized in-memory tensors and own
all randomness and experiment receipts.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


_LABELS = 13


def _invalid() -> ValueError:
    return ValueError("invalid retinal adaptation input")


def _require_finite_tensor(value: object, *, ndim: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise _invalid()
    if ndim is not None and value.ndim != ndim:
        raise _invalid()
    if not torch.isfinite(value).all().item():
        raise _invalid()
    return value


def _label_weight(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) not in (0.0, 1.0):
        raise _invalid()
    return float(value)


class RetinalSupervisedAdaptationKernel(nn.Module):
    """Copied student/EMA teacher encoder with masked and observed-label losses.

    ``encode_student`` returns the mean of final-norm patch tokens, excluding
    the encoder's declared prefix tokens.  A real binding must provide the
    intended 384-dimensional, five-prefix encoder; tiny synthetic encoders are
    supported to test the token/RoPE contract only.
    """

    def __init__(self, base_encoder: nn.Module) -> None:
        super().__init__()
        self._validate_architecture(base_encoder)
        self.student = copy.deepcopy(base_encoder)
        self.teacher = copy.deepcopy(base_encoder)
        self.embedding_dim = int(self.student.embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # This is the common masked-token predictor from the historical loop;
        # it exists and is trainable in both matched comparison arms.
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.label_head = nn.Linear(self.embedding_dim, _LABELS)
        self._freeze_encoder_for_adaptation()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

    @staticmethod
    def _validate_architecture(encoder: object) -> None:
        if not isinstance(encoder, nn.Module):
            raise _invalid()
        if type(getattr(encoder, "embed_dim", None)) is not int or encoder.embed_dim <= 0:
            raise _invalid()
        if not callable(getattr(encoder, "patch_embed", None)) or not callable(getattr(encoder, "_pos_embed", None)):
            raise _invalid()
        if not callable(getattr(encoder, "norm", None)):
            raise _invalid()
        blocks = getattr(encoder, "blocks", None)
        try:
            if blocks is None or len(blocks) < 2:
                raise _invalid()
        except TypeError:
            raise _invalid() from None
        prefix = getattr(encoder, "num_prefix_tokens", None)
        if type(prefix) is not int or prefix < 0:
            raise _invalid()
        norm_pre = getattr(encoder, "norm_pre", None)
        if norm_pre is not None and not callable(norm_pre):
            raise _invalid()

    def _freeze_encoder_for_adaptation(self) -> None:
        for parameter in self.student.parameters():
            parameter.requires_grad_(False)
        for block in self.student.blocks[-2:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in self.student.norm.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # ``nn.Module.train`` would otherwise flip the EMA teacher to training.
        self.teacher.eval()
        return self

    def _forward_tokens(
        self,
        encoder: nn.Module,
        images: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        patches = encoder.patch_embed(images)
        if not isinstance(patches, torch.Tensor) or patches.ndim not in (3, 4):
            raise _invalid()
        if patches.shape[0] != images.shape[0] or patches.shape[-1] != self.embedding_dim:
            raise _invalid()
        if patches.ndim == 4:
            batch, height, width, _ = patches.shape
            patch_count = height * width
        else:
            batch, patch_count, _ = patches.shape
        if patch_count <= 0:
            raise _invalid()
        if patch_mask is not None:
            if (
                not isinstance(patch_mask, torch.Tensor)
                or patch_mask.dtype is not torch.bool
                or patch_mask.ndim != 2
                or patch_mask.shape != (batch, patch_count)
                or patch_mask.device != patches.device
            ):
                raise _invalid()
            mask_shape = (batch, height, width, 1) if patches.ndim == 4 else (batch, patch_count, 1)
            mask = patch_mask.reshape(mask_shape)
            token = self.mask_token.to(device=patches.device, dtype=patches.dtype).expand_as(patches)
            patches = torch.where(mask, token, patches)
        positioned = encoder._pos_embed(patches)
        if isinstance(positioned, tuple):
            if len(positioned) != 2 or not isinstance(positioned[0], torch.Tensor):
                raise _invalid()
            tokens, rope = positioned
        elif isinstance(positioned, torch.Tensor):
            tokens, rope = positioned, None
        else:
            raise _invalid()
        if getattr(encoder, "norm_pre", None) is not None:
            tokens = encoder.norm_pre(tokens)
        mixed = bool(getattr(encoder, "rope_mixed", False)) and rope is not None
        for index, block in enumerate(encoder.blocks):
            try:
                if mixed:
                    tokens = block(tokens, rope=rope[index])
                elif rope is not None:
                    tokens = block(tokens, rope=rope)
                else:
                    tokens = block(tokens)
            except (IndexError, KeyError, TypeError, AttributeError):
                raise _invalid() from None
        tokens = encoder.norm(tokens)
        prefixes = int(encoder.num_prefix_tokens)
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.ndim != 3
            or tokens.shape != (batch, prefixes + patch_count, self.embedding_dim)
        ):
            raise _invalid()
        return tokens, patch_count

    def _patch_tokens(self, tokens: torch.Tensor, patch_count: int) -> torch.Tensor:
        prefixes = int(self.student.num_prefix_tokens)
        patches = tokens[:, prefixes:]
        if patches.shape != (tokens.shape[0], patch_count, self.embedding_dim):
            raise _invalid()
        return patches

    def encode_student(self, images: torch.Tensor) -> torch.Tensor:
        """Return final-norm mean patch embeddings for caller-owned downstream use."""
        checked = _require_finite_tensor(images, ndim=4)
        tokens, patch_count = self._forward_tokens(self.student, checked)
        return self._patch_tokens(tokens, patch_count).mean(dim=1)

    def _validate_objective_inputs(
        self,
        images: object,
        patch_mask: object,
        labels: object,
        observed: object,
        positive_weight: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        checked_images = _require_finite_tensor(images, ndim=4)
        if checked_images.shape[0] <= 0:
            raise _invalid()
        if not isinstance(patch_mask, torch.Tensor) or patch_mask.dtype is not torch.bool or patch_mask.ndim != 2:
            raise _invalid()
        if patch_mask.shape[0] != checked_images.shape[0] or not patch_mask.any().item():
            raise _invalid()
        if not isinstance(labels, torch.Tensor) or not labels.is_floating_point() or labels.shape != (checked_images.shape[0], _LABELS):
            raise _invalid()
        if not isinstance(observed, torch.Tensor) or observed.dtype is not torch.bool or observed.shape != labels.shape:
            raise _invalid()
        if labels.device != checked_images.device or observed.device != checked_images.device or patch_mask.device != checked_images.device:
            raise _invalid()
        finite = torch.isfinite(labels)
        if (observed & ~finite).any().item():
            raise _invalid()
        if (observed & ~((labels == 0.0) | (labels == 1.0))).any().item():
            raise _invalid()
        if not isinstance(positive_weight, torch.Tensor) or not positive_weight.is_floating_point() or positive_weight.shape != (_LABELS,):
            raise _invalid()
        if positive_weight.device != checked_images.device or not torch.isfinite(positive_weight).all().item() or not (positive_weight > 0).all().item():
            raise _invalid()
        return checked_images, patch_mask, labels, observed, positive_weight

    def objective(
        self,
        images: torch.Tensor,
        patch_mask: torch.Tensor,
        labels: torch.Tensor,
        observed: torch.Tensor,
        *,
        label_weight: float | int,
        positive_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one local scalar loss; caller owns optimizer, RNG, and reporting."""
        weight = _label_weight(label_weight)
        images, patch_mask, labels, observed, positive_weight = self._validate_objective_inputs(
            images, patch_mask, labels, observed, positive_weight
        )
        with torch.no_grad():
            teacher_tokens, patch_count = self._forward_tokens(self.teacher, images)
            teacher_target = F.layer_norm(
                self._patch_tokens(teacher_tokens, patch_count), (self.embedding_dim,)
            )
        masked_tokens, masked_count = self._forward_tokens(self.student, images, patch_mask)
        if masked_count != patch_count:
            raise _invalid()
        prediction = self.predictor(self._patch_tokens(masked_tokens, patch_count))
        selected_prediction = prediction[patch_mask]
        selected_target = teacher_target[patch_mask]
        if selected_prediction.numel() == 0:
            raise _invalid()
        masked_loss = F.smooth_l1_loss(selected_prediction, selected_target, reduction="sum")
        masked_loss = masked_loss / selected_prediction.numel()

        # Execute the full student path even at coefficient zero, preserving the
        # forward structure of the matched control arm.
        full_tokens, full_count = self._forward_tokens(self.student, images)
        if full_count != patch_count:
            raise _invalid()
        pooled = self._patch_tokens(full_tokens, patch_count).mean(dim=1)
        logits = self.label_head(pooled)
        safe_labels = torch.where(observed, labels, torch.zeros_like(labels))
        element_loss = F.binary_cross_entropy_with_logits(
            logits, safe_labels, pos_weight=positive_weight, reduction="none"
        )
        denominator = observed.sum()
        if denominator.item() == 0:
            label_loss = logits.sum() * 0.0
        else:
            label_loss = (element_loss * observed.to(element_loss.dtype)).sum() / denominator
        return masked_loss + weight * label_loss

    @torch.no_grad()
    def update_teacher(self, coefficient: float | int) -> None:
        """Apply EMA to copied encoder parameters after a caller optimizer step."""
        if type(coefficient) not in (int, float) or not math.isfinite(float(coefficient)) or not 0.0 <= float(coefficient) <= 1.0:
            raise _invalid()
        for teacher_parameter, student_parameter in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_parameter.mul_(float(coefficient)).add_(student_parameter.detach(), alpha=1.0 - float(coefficient))
        self.teacher.eval()
