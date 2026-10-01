"""Retinally recoverable probabilistic Patient Atlas V6.2 kernel."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from soft_patient_atlas import PatientAtlasConfig, PatientStateOutput
from soft_patient_atlas_v5 import RetinalCapacityExpandedSoftPatientAtlas, V5_PARTITION


V6_2_RECOVERY_DIM = 64
V6_2_RECOVERY_WEIGHT = 0.05


class RetinallyRecoverableSoftPatientAtlas(
    RetinalCapacityExpandedSoftPatientAtlas
):
    """Exact V5 state plus an outcome-free linear-recoverability training head."""

    def __init__(
        self,
        config: PatientAtlasConfig,
        blood_tower: nn.Module,
        *,
        blood_anchor_eligible_mask: torch.Tensor,
        retinal_recovery_components: torch.Tensor,
    ) -> None:
        super().__init__(
            config,
            blood_tower,
            blood_anchor_eligible_mask=blood_anchor_eligible_mask,
        )
        components = torch.as_tensor(retinal_recovery_components).detach().clone()
        expected = (config.eye_dim, V6_2_RECOVERY_DIM)
        if components.shape != expected or not components.is_floating_point():
            raise TypeError(
                f"retinal_recovery_components must be floating with shape {expected}"
            )
        if not bool(torch.isfinite(components).all()):
            raise ValueError("retinal_recovery_components contain non-finite values")
        components = components.to(dtype=torch.float32)
        identity = torch.eye(V6_2_RECOVERY_DIM, dtype=components.dtype)
        if not torch.allclose(
            components.T @ components, identity, atol=1e-5, rtol=1e-5
        ):
            raise ValueError(
                "retinal_recovery_components must have orthonormal columns"
            )
        self.register_buffer("retinal_recovery_components", components)
        self.retinal_recovery_head = nn.Linear(
            V5_PARTITION.eye_private_dim, V6_2_RECOVERY_DIM, bias=False
        )
        nn.init.orthogonal_(self.retinal_recovery_head.weight)

    def retinal_recovery_prediction(
        self, output: PatientStateOutput
    ) -> torch.Tensor:
        eye_state = self._state_from_factor(
            output.eye_evidence,
            min_log_variance=self.config.min_log_variance,
            max_log_variance=self.config.max_log_variance,
        )
        return self.retinal_recovery_head(
            eye_state.mean[:, V5_PARTITION.eye_private_slice]
        )

    def retinal_recovery_loss(
        self,
        output: PatientStateOutput,
        *,
        target_eye_embeddings: torch.Tensor,
        target_eye_mask: torch.Tensor,
    ) -> torch.Tensor:
        if target_eye_embeddings.ndim != 3:
            raise ValueError("retinal recovery eye targets must be rank three")
        if target_eye_mask.shape != target_eye_embeddings.shape[:2]:
            raise ValueError("retinal recovery eye mask differs")
        safe = torch.where(
            target_eye_mask[..., None],
            target_eye_embeddings,
            torch.zeros_like(target_eye_embeddings),
        )
        count = target_eye_mask.sum(dim=1, keepdim=True)
        mean = safe.sum(dim=1) / count.clamp(min=1).to(
            dtype=target_eye_embeddings.dtype
        )
        target = mean @ self.retinal_recovery_components.to(
            dtype=target_eye_embeddings.dtype
        )
        prediction = self.retinal_recovery_prediction(output)
        eligible = output.eye_evidence.present & target_eye_mask.any(dim=1)
        if not bool(eligible.any()):
            return prediction.sum() * 0.0
        return (prediction[eligible] - target[eligible]).square().mean()

    def variational_loss(
        self,
        output: PatientStateOutput,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        base = super().variational_loss(output, **kwargs)
        recovery = self.retinal_recovery_loss(
            output,
            target_eye_embeddings=kwargs["target_eye_embeddings"],
            target_eye_mask=kwargs["target_eye_mask"],
        )
        result = dict(base)
        result["total"] = base["total"] + V6_2_RECOVERY_WEIGHT * recovery
        result["retinal_recovery"] = recovery
        return result


__all__ = [
    "RetinallyRecoverableSoftPatientAtlas",
    "V6_2_RECOVERY_DIM",
    "V6_2_RECOVERY_WEIGHT",
]
