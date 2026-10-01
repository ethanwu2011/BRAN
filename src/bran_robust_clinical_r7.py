"""No-I/O, input-only clinical compression; original targets and heads unchanged."""
import torch
from bran_multisource_anchored_model_v3 import BRANMultisourceAnchoredModelV3

ERROR = 'robust_clinical_r7_contract_failed'
INPUT_MAP = {'name': 'scaled_asinh', 'scale': 3.0,
             'location': 'sanitized_clinical_mlp_and_residual_inputs',
             'targets_changed': False, 'age_changed': False, 'masks_changed': False}


class BRANRobustClinicalR7(BRANMultisourceAnchoredModelV3):
    def _clinical_hidden(self, clean, valid, age):
        # Base encode has already applied eligibility, finite-value and mask
        # sanitization. Apply the same map before BOTH inherited numerical paths.
        compressed = 3.0 * torch.asinh(clean / 3.0)
        return super()._clinical_hidden(compressed, valid, age)

    def export_config(self):
        return {**super().export_config(), 'version': 7,
                'clinical_encoder_input_map': dict(INPUT_MAP)}

    @classmethod
    def from_config(cls, values):
        try:
            if type(values) is not dict:
                raise ValueError(ERROR)
            model = cls(values['arm'], values['eligible_indices'], values['cbc_indices'])
            if values != model.export_config():
                raise ValueError(ERROR)
            return model
        except (KeyError, TypeError, ValueError):
            raise ValueError(ERROR) from None

    @classmethod
    def from_parent(cls, parent):
        if type(parent) is not BRANMultisourceAnchoredModelV3:
            raise ValueError(ERROR)
        model = cls(parent.arm, parent.eligible_indices, parent.cbc_indices)
        model.load_state_dict(parent.state_dict(), strict=True)
        model.train(parent.training)
        for target, source in zip(model.parameters(), parent.parameters()):
            target.requires_grad_(source.requires_grad)
        if not all(torch.equal(v, model.state_dict()[k]) for k, v in parent.state_dict().items()):
            raise ValueError(ERROR)
        return model
