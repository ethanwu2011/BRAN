"""Audited multisource retinal candidate binding for a separate BRAN refit."""
import copy
import re

import numpy as np
import torch

import bran_retinal_multisource_evaluation_v1 as evaluation
import run_bran_retinal_multisource_adaptation_v1 as run
from bran_retinal_multisource_patient_kernel_v1 import RetinalMultisourcePatientKernel


_ERROR = 'retinal multisource candidate binding rejected'


def require(value):
    if not value:
        raise ValueError(_ERROR)


def pin(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None)


def validate_contract(value):
    require(type(value) is dict and set(value) == {
        'schema', 'scope', 'protocol_sha256', 'audit_sha256', 'terminal_manifest_sha256', 'checkpoint_sha256',
        'source_admission_sha256', 'source_review_evidence_sha256', 'source_gate', 'input', 'embedding',
        'requires_unified_refit', 'old_bran_coordinate_compatibility_established', 'unified_model_promoted',
        'clinical_validation_established', 'patient_level_output_emitted'})
    require(value['schema'] == 'bran-multisource-retinal-encoder-binding-v1'
            and value['scope'] == 'retinal_encoder_for_separate_refit'
            and value['source_gate'] == 'brset_image_only_primary')
    for key in ('protocol_sha256', 'audit_sha256', 'terminal_manifest_sha256', 'checkpoint_sha256',
                'source_admission_sha256', 'source_review_evidence_sha256'):
        pin(value[key])
    require(value['input'] == {'layout': 'NCHW', 'resolution': 224, 'channels': 3, 'dtype': 'float32',
                               'normalization_mean': [.485, .456, .406], 'normalization_std': [.229, .224, .225]})
    require(value['embedding'] == {'width': 384, 'dtype': 'float32', 'prefix_tokens_excluded': 5,
                                   'pooling': 'mean_of_learned_final_norm_patch_tokens', 'additional_normalization': None})
    require(value['requires_unified_refit'] is True and all(value[key] is False for key in (
        'old_bran_coordinate_compatibility_established', 'unified_model_promoted',
        'clinical_validation_established', 'patient_level_output_emitted')))


def require_encoder_identity(contract, expected_checkpoint_sha256):
    """Check exact candidate bytes; this does not prove BRAN coordinate compatibility."""
    validate_contract(contract)
    pin(expected_checkpoint_sha256)
    require(contract['checkpoint_sha256'] == expected_checkpoint_sha256)


class AuditedRetinalCandidate:
    """Private frozen candidate encoder; no source data, grouping, or refit occurs here."""

    def __init__(self, model, contract, device):
        validate_contract(contract)
        self._contract = copy.deepcopy(contract)
        self._device = device
        self._model = model.to(device).eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    def __repr__(self):
        return '<AuditedRetinalCandidate private>'

    @property
    def contract(self):
        return copy.deepcopy(self._contract)

    def encode(self, images):
        """Encode at most 16 already normalized NCHW RGB images to private 384-D features."""
        with run.r.quiet():
            try:
                require(not self._model.training and all(not p.requires_grad for p in self._model.parameters()))
                require(type(images) is np.ndarray and images.dtype == np.float32 and images.ndim == 4
                        and images.shape[1:] == (3, 224, 224) and 0 < len(images) <= 16 and np.all(np.isfinite(images)))
                mean = np.asarray((.485, .456, .406), np.float32).reshape(1, 3, 1, 1)
                std = np.asarray((.229, .224, .225), np.float32).reshape(1, 3, 1, 1)
                rgb = images * std + mean
                require(np.all(rgb >= -1e-5) and np.all(rgb <= 1.00001))
                with torch.inference_mode():
                    encoded = self._model.encode_student(torch.from_numpy(np.ascontiguousarray(images)).to(self._device))
                    value = encoded.float().cpu().numpy()
                require(value.shape == (len(images), 384) and value.dtype == np.float32 and np.all(np.isfinite(value)))
                return value
            except Exception:
                raise ValueError(_ERROR) from None


def _positive_brset_gate(summary):
    require(type(summary) is dict)
    primary = summary.get('brset', {}).get('image_only_primary') if type(summary.get('brset')) is dict else None
    require(summary.get('eligible_for_separate_unified_refit_review') is True
            and type(primary) is dict
            and primary.get('analysis') == 'image_only_primary'
            and primary.get('eligible_for_separate_unified_refit_review') is True)


def authenticate_contract(protocol_sha256, audit_sha256):
    """Authenticate source-stage evidence without constructing an inference model.

    The original audit still replays its private readouts/receipts locally.
    This is not a replay-free shortcut or permission to bypass the source gate.
    """
    with run.r.quiet():
        try:
            pin(protocol_sha256)
            pin(audit_sha256)
            # authenticate_audit replays readouts and therefore deserializes
            # receipts.  Establish the stored terminal BRSET gate first.
            summary = run.payload(protocol_sha256, terminal=True, replay_results=False)
            evaluation.validate_summary(summary)
            _positive_brset_gate(summary)  # ODIR or the age secondary cannot substitute.
            run.authenticate_audit(protocol_sha256, audit_sha256)
            # Recheck terminal evidence after the full audit, without granting
            # a replay-free audit bypass.
            summary = run.payload(protocol_sha256, terminal=True, replay_results=False)
            evaluation.validate_summary(summary)
            _positive_brset_gate(summary)
            protocol = run.load_protocol(protocol_sha256)
            admission = protocol.get('admission') if type(protocol) is dict else None
            require(type(admission) is dict and set(admission) == {'admission_sha256', 'reviewed_evidence_sha256'})
            pin(admission['admission_sha256'])
            pin(admission['reviewed_evidence_sha256'])
            manifest = run.read_json(run.OUT / 'manifest.json')
            require(type(manifest) is dict and type(manifest.get('private_sha256')) is dict)
            checkpoint = run.PRIVATE / 'multisource_candidate.pt'
            expected_checkpoint = manifest['private_sha256'].get('multisource_candidate.pt')
            pin(expected_checkpoint)
            require(run.r.sha(checkpoint) == expected_checkpoint)
            contract = {
                'schema': 'bran-multisource-retinal-encoder-binding-v1',
                'scope': 'retinal_encoder_for_separate_refit',
                'protocol_sha256': protocol_sha256, 'audit_sha256': audit_sha256,
                'terminal_manifest_sha256': run.r.sha(run.OUT / 'manifest.json'),
                'checkpoint_sha256': expected_checkpoint,
                'source_admission_sha256': admission['admission_sha256'],
                'source_review_evidence_sha256': admission['reviewed_evidence_sha256'],
                'source_gate': 'brset_image_only_primary',
                'input': {'layout': 'NCHW', 'resolution': 224, 'channels': 3, 'dtype': 'float32',
                          'normalization_mean': [.485, .456, .406], 'normalization_std': [.229, .224, .225]},
                'embedding': {'width': 384, 'dtype': 'float32', 'prefix_tokens_excluded': 5,
                              'pooling': 'mean_of_learned_final_norm_patch_tokens', 'additional_normalization': None},
                'requires_unified_refit': True, 'old_bran_coordinate_compatibility_established': False,
                'unified_model_promoted': False, 'clinical_validation_established': False,
                'patient_level_output_emitted': False,
            }
            validate_contract(contract)
            return contract
        except Exception:
            raise ValueError(_ERROR) from None


def load_candidate(protocol_sha256, audit_sha256):
    """Load only after an authenticated positive BRSET image-only source-stage result."""
    with run.r.quiet():
        try:
            contract = authenticate_contract(protocol_sha256, audit_sha256)
            checkpoint = run.PRIVATE / 'multisource_candidate.pt'
            require(run.r.sha(checkpoint) == contract['checkpoint_sha256'])
            receipt = torch.load(checkpoint, map_location='cpu', weights_only=False)
            run.validate_receipt(receipt, 'multisource_candidate')
            model = RetinalMultisourcePatientKernel(run.ref.load_base())
            model.load_state_dict(receipt['kernel_state'], strict=True)
            require(run.r.sha(checkpoint) == contract['checkpoint_sha256'])
            return AuditedRetinalCandidate(model, contract, run.ref.device())
        except Exception:
            raise ValueError(_ERROR) from None
