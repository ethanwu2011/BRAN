"""Private in-memory grouped retinal adaptation objective for BRSET and ODIR."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from bran_retinal_supervised_adaptation_kernel_v1 import RetinalSupervisedAdaptationKernel, _invalid, _require_finite_tensor


_WIDTHS = {"brset": 13, "odir": 8}


class RetinalMultisourcePatientKernel(RetinalSupervisedAdaptationKernel):
    """Copied encoder/EMA teacher with source-specific grouped label heads.

    This is a step objective only.  It accepts caller-authorized tensors and
    does not sample, read files, optimize parameters, or decide any comparison.
    """

    def __init__(self, base_encoder: nn.Module) -> None:
        super().__init__(base_encoder)
        self.brset_label_head = self.label_head
        del self.label_head
        self.odir_label_head = nn.Linear(self.embedding_dim, _WIDTHS["odir"])

    @staticmethod
    def _mean_by_group(values: torch.Tensor, group_index: torch.Tensor, groups: int) -> torch.Tensor:
        totals = torch.zeros((groups, *values.shape[1:]), device=values.device, dtype=values.dtype)
        totals.index_add_(0, group_index, values)
        counts = torch.zeros((groups,), device=values.device, dtype=values.dtype)
        counts.index_add_(0, group_index, torch.ones((values.shape[0],), device=values.device, dtype=values.dtype))
        return totals / counts.reshape(groups, *([1] * (values.ndim - 1)))

    def _validate_grouped_inputs(
        self,
        source: object,
        images: object,
        patch_mask: object,
        group_index: object,
        labels: object,
        observed: object,
        positive_weight: object,
    ) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if type(source) is not str or source not in _WIDTHS:
            raise _invalid()
        checked_images = _require_finite_tensor(images, ndim=4)
        n_images = checked_images.shape[0]
        if n_images <= 0 or checked_images.shape[1] != 3 or checked_images.shape[2] <= 0 or checked_images.shape[3] <= 0:
            raise _invalid()
        if (not isinstance(group_index, torch.Tensor) or group_index.dtype is not torch.int64
                or group_index.ndim != 1 or group_index.shape != (n_images,)
                or group_index.device != checked_images.device):
            raise _invalid()
        if group_index.numel() == 0 or (group_index < 0).any().item():
            raise _invalid()
        groups = int(group_index.max().item()) + 1
        # Bound the allocation before constructing any group-sized tensor.  The
        # following count uses index_add, supported by the intended accelerator
        # paths, rather than CPU-only grouping helpers.
        if groups > n_images:
            raise _invalid()
        group_counts = torch.zeros((groups,), device=group_index.device, dtype=torch.float32)
        group_counts.index_add_(0, group_index, torch.ones((n_images,), device=group_index.device, dtype=torch.float32))
        if (group_counts == 0).any().item():
            raise _invalid()
        if source == "brset":
            if groups != n_images or not torch.equal(group_index, torch.arange(n_images, device=group_index.device)):
                raise _invalid()
        elif (group_counts < 1).any().item() or (group_counts > 2).any().item():
            raise _invalid()
        width = _WIDTHS[source]
        if (not isinstance(labels, torch.Tensor) or not labels.is_floating_point() or labels.shape != (groups, width)
                or labels.device != checked_images.device):
            raise _invalid()
        if (not isinstance(observed, torch.Tensor) or observed.dtype is not torch.bool or observed.shape != labels.shape
                or observed.device != checked_images.device):
            raise _invalid()
        finite = torch.isfinite(labels)
        if (observed & ~finite).any().item() or (observed & ~((labels == 0.0) | (labels == 1.0))).any().item():
            raise _invalid()
        if (not isinstance(positive_weight, torch.Tensor) or not positive_weight.is_floating_point()
                or positive_weight.shape != (width,) or positive_weight.device != checked_images.device
                or not torch.isfinite(positive_weight).all().item() or not (positive_weight > 0).all().item()):
            raise _invalid()
        if (not isinstance(patch_mask, torch.Tensor) or patch_mask.dtype is not torch.bool or patch_mask.ndim != 2
                or patch_mask.shape[0] != n_images or patch_mask.device != checked_images.device):
            raise _invalid()
        return source, checked_images, patch_mask, group_index, labels, observed, groups

    def grouped_objective(
        self,
        source: str,
        images: torch.Tensor,
        patch_mask: torch.Tensor,
        group_index: torch.Tensor,
        labels: torch.Tensor,
        observed: torch.Tensor,
        *,
        positive_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Return patient-balanced masked plus observed-label loss for one source."""
        source, images, patch_mask, group_index, labels, observed, groups = self._validate_grouped_inputs(
            source, images, patch_mask, group_index, labels, observed, positive_weight
        )
        with torch.no_grad():
            teacher_tokens, patch_count = self._forward_tokens(self.teacher, images)
            teacher_target = F.layer_norm(self._patch_tokens(teacher_tokens, patch_count), (self.embedding_dim,))
        if patch_mask.shape != (images.shape[0], patch_count) or not patch_mask.any(dim=1).all().item():
            raise _invalid()
        masked_tokens, masked_count = self._forward_tokens(self.student, images, patch_mask)
        if masked_count != patch_count:
            raise _invalid()
        prediction = self.predictor(self._patch_tokens(masked_tokens, patch_count))
        token_losses = F.smooth_l1_loss(prediction, teacher_target, reduction="none").mean(dim=-1)
        image_masked = (token_losses * patch_mask.to(token_losses.dtype)).sum(dim=1) / patch_mask.sum(dim=1)
        masked_loss = self._mean_by_group(image_masked, group_index, groups).mean()

        full_tokens, full_count = self._forward_tokens(self.student, images)
        if full_count != patch_count:
            raise _invalid()
        image_features = self._patch_tokens(full_tokens, patch_count).mean(dim=1)
        patient_features = self._mean_by_group(image_features, group_index, groups)
        head = self.brset_label_head if source == "brset" else self.odir_label_head
        logits = head(patient_features)
        safe_labels = torch.where(observed, labels, torch.zeros_like(labels))
        element_loss = F.binary_cross_entropy_with_logits(logits, safe_labels, pos_weight=positive_weight, reduction="none")
        denominator = observed.sum()
        supervised_loss = logits.sum() * 0.0 if denominator.item() == 0 else (
            (element_loss * observed.to(element_loss.dtype)).sum() / denominator
        )
        return masked_loss + supervised_loss


__all__ = ["RetinalMultisourcePatientKernel"]
