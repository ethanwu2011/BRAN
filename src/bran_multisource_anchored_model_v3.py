"""Typed-age extension preserving the retained visible clinical anchor.

No I/O. This is the same 192-state MLP with the retained auxiliary head;
the head is not an input bypass or a downstream prediction attachment.
"""
import torch
from torch import nn

from bran_multisource_model_v2 import BRANMultisourceModelV2
from bran_patient_state_anchor_v2 import CLINICAL_ANCHOR_WEIGHT, V2_EXTENSION_NAME
from bran_patient_state_prototype_v1 import _masked_finite


class BRANMultisourceAnchoredModelV3(BRANMultisourceModelV2):
    def __init__(self, arm, eligible_indices, cbc_indices, seed=95101):
        if arm != 'mlp':
            raise ValueError('anchored_multisource_model_invalid')
        super().__init__(arm, eligible_indices, cbc_indices, seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.clinical_linear_anchor = nn.Linear(128, 48)

    def _anchor_prediction(self, state):
        return self.clinical_linear_anchor(torch.cat([state.mean[:, :64], state.mean[:, 128:]], -1))

    def objective(self, state, age, target_clinical_values, target_clinical_mask,
                  visible_clinical_mask, target_retinal_features=None,
                  target_retinal_mask=None, visible_retinal_mask=None,
                  disease_target=None, disease_mask=None, kl_weight=.01,
                  disease_weight=0., visible_weight=0., clinical_eligible_mask=None):
        base = super().objective(state, age, target_clinical_values, target_clinical_mask,
            visible_clinical_mask, target_retinal_features, target_retinal_mask,
            visible_retinal_mask, disease_target, disease_mask, kl_weight,
            disease_weight, visible_weight, clinical_eligible_mask)
        visible = target_clinical_mask[:, :48] & visible_clinical_mask[:, :48]
        visible = visible & self.eligible_slots[None, :48]
        if clinical_eligible_mask is not None:
            visible = visible & clinical_eligible_mask[:, :48]
        clean, valid = _masked_finite(target_clinical_values[:, :48], visible)
        prediction = self._anchor_prediction(state)
        mse = ((prediction-clean).square()*valid.to(prediction.dtype)).sum()/valid.sum().clamp_min(1)
        base['clinical_anchor_visible_mse'] = mse
        base['loss'] = base['loss'] + CLINICAL_ANCHOR_WEIGHT*mse
        return base

    def export_config(self):
        return {**super().export_config(), 'version': 3,
                'retained_extension': {'name': V2_EXTENSION_NAME, 'weight': CLINICAL_ANCHOR_WEIGHT}}

    @classmethod
    def from_config(cls, values):
        if not isinstance(values, dict):
            raise ValueError('anchored_multisource_model_invalid')
        try:
            model = cls(values['arm'], values['eligible_indices'], values['cbc_indices'])
        except (KeyError, TypeError):
            raise ValueError('anchored_multisource_model_invalid') from None
        if values != model.export_config():
            raise ValueError('anchored_multisource_model_invalid')
        return model
