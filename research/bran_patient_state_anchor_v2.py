"""Fresh V2 linear clinical-recoverability ablation; V1 remains untouched.

This is not a pretrained blood anchor.  It is a train-only linear auxiliary
that asks the advertised state to preserve already-visible, eligible,
outer-fold-normalized continuous clinical coordinates.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from bran_patient_state_prototype_v1 import (
    BRANPatientStatePrototypeV1,
    PatientStateConfig,
    PosteriorState,
    _masked_finite,
)


CLINICAL_ANCHOR_WEIGHT = 0.1
V2_EXTENSION_NAME = "visible_eligible_clinical_linear_recoverability_v2"


class BRANClinicalAnchorV2(BRANPatientStatePrototypeV1):
    """V1 plus a fixed-weight, visible-only linear clinical preservation head.

    The head reads only shared and clinical-private posterior-mean coordinates.
    It is not part of the exported state or any downstream decoder/readout.
    """

    def __init__(self, config: PatientStateConfig = PatientStateConfig()) -> None:
        super().__init__(config)
        c = config
        # Do not advance the caller's RNG sequence: under an equal seed V1
        # base parameters and all subsequent V1 random draws are identical.
        with torch.random.fork_rng(devices=[]):
            self.clinical_linear_anchor = nn.Linear(
                c.shared_dim + c.clinical_private_dim, c.clinical_continuous_dim
            )

    def export_config(self) -> Dict[str, object]:
        """Explicitly distinguish this fresh extension from V1's config schema."""
        return {
            "base_patient_state_config": self.config.to_dict(),
            "extension": {
                "name": V2_EXTENSION_NAME,
                "weight": CLINICAL_ANCHOR_WEIGHT,
                "target": "visible_eligible_normalized_continuous_clinical_fields_only",
                "head": "linear(shared_plus_clinical_private_to_48)",
                "pretrained": False,
                "downstream_raw_feature_bypass": False,
            },
        }

    @classmethod
    def from_config(cls, value: Dict[str, object]) -> "BRANClinicalAnchorV2":
        extension = value.get("extension", {})
        if extension.get("name") != V2_EXTENSION_NAME or extension.get("weight") != CLINICAL_ANCHOR_WEIGHT:
            raise ValueError("not a BRAN clinical-anchor V2 configuration")
        base = value.get("base_patient_state_config")
        if not isinstance(base, dict):
            raise ValueError("missing base_patient_state_config")
        return cls(PatientStateConfig.from_dict(base))

    def _anchor_prediction(self, state: PosteriorState) -> Tensor:
        c = self.config
        shared = state.mean[:, :c.shared_dim]
        clinical_start = c.shared_dim + c.retinal_private_dim
        clinical_private = state.mean[:, clinical_start:]
        return self.clinical_linear_anchor(torch.cat([shared, clinical_private], dim=-1))

    def objective(
        self,
        state: PosteriorState,
        age: Tensor,
        target_clinical_values: Tensor,
        target_clinical_mask: Tensor,
        visible_clinical_mask: Tensor,
        target_retinal_features: Optional[Tensor] = None,
        target_retinal_mask: Optional[Tensor] = None,
        visible_retinal_mask: Optional[Tensor] = None,
        disease_target: Optional[Tensor] = None,
        disease_mask: Optional[Tensor] = None,
        kl_weight: float = 0.01,
        disease_weight: float = 0.0,
        visible_weight: float = 0.0,
        clinical_eligible_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Return the untouched V1 objective plus visible clinical anchor MSE."""
        base = super().objective(
            state, age, target_clinical_values, target_clinical_mask,
            visible_clinical_mask, target_retinal_features, target_retinal_mask,
            visible_retinal_mask, disease_target, disease_mask, kl_weight,
            disease_weight, visible_weight,
        )
        c = self.config
        if target_clinical_values.shape[-1] != c.clinical_dim:
            raise ValueError("target clinical dimension mismatch")
        target = target_clinical_values[:, :c.clinical_continuous_dim]
        visible = (target_clinical_mask[:, :c.clinical_continuous_dim].bool()
                   & visible_clinical_mask[:, :c.clinical_continuous_dim].bool())
        if clinical_eligible_mask is not None:
            if clinical_eligible_mask.shape != target_clinical_values.shape:
                raise ValueError("clinical_eligible_mask shape mismatch")
            visible = visible & clinical_eligible_mask[:, :c.clinical_continuous_dim].bool()
        clean, valid = _masked_finite(target, visible)
        prediction = self._anchor_prediction(state)
        anchor_mse = ((prediction - clean).square() * valid.to(prediction.dtype)).sum() / valid.sum().clamp_min(1)
        base["clinical_anchor_visible_mse"] = anchor_mse
        base["loss"] = base["loss"] + CLINICAL_ANCHOR_WEIGHT * anchor_mse
        return base
