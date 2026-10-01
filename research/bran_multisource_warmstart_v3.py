"""Source-only V1-to-V2 MLP warm start for synthetic BRAN experiments.

This module deliberately accepts a live V1 module rather than a checkpoint.  It
does no filesystem, network, data, logging, or training I/O.  Only the exact
default V1 architecture with its two attached native heads is supported.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from bran_multisource_model_v2 import BRANMultisourceModelV2
from bran_multisource_anchored_model_v3 import BRANMultisourceAnchoredModelV3
from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
from bran_patient_state_prototype_v1 import BRANPatientStatePrototypeV1, PatientStateConfig


_INVALID = "multisource warmstart invalid"
_SOURCE_HEADS = ("screening_joint_head", "cbc_joint_head")
_ANCHOR_KEYS = frozenset({'clinical_linear_anchor.weight', 'clinical_linear_anchor.bias'})

# The default V1 state dict, excluding the two heads that this initializer
# requires callers to attach.  Keeping this explicit makes new/unknown V1
# state (including a stray buffer) fail closed instead of being ignored.
_V1_KEYS = frozenset({
    "retinal_prior_logvar", "clinical_prior_logvar", "continuous_logscale",
    "retinal_logscale", "clinical_residual_factor",
    "clinical_encoder.0.weight", "clinical_encoder.0.bias",
    "clinical_encoder.2.weight", "clinical_encoder.2.bias",
    "clinical_residual.weight", "image_encoder.0.weight",
    "image_encoder.0.bias", "image_encoder.2.weight", "image_encoder.2.bias",
    "image_gate.weight", "image_gate.bias", "retinal_encoder.0.weight",
    "retinal_encoder.0.bias", "retinal_encoder.2.weight",
    "retinal_encoder.2.bias", "retinal_residual.weight",
    "shared_posterior.0.weight", "shared_posterior.0.bias",
    "shared_posterior.2.weight", "shared_posterior.2.bias",
    "retinal_prior.weight", "clinical_prior.weight", "retinal_delta.0.weight",
    "retinal_delta.0.bias", "retinal_delta.2.weight", "retinal_delta.2.bias",
    "clinical_delta.0.weight", "clinical_delta.0.bias",
    "clinical_delta.2.weight", "clinical_delta.2.bias",
    "continuous_decoder.0.weight", "continuous_decoder.0.bias",
    "continuous_decoder.2.weight", "continuous_decoder.2.bias",
    "binary_decoder.0.weight", "binary_decoder.0.bias",
    "binary_decoder.2.weight", "binary_decoder.2.bias",
    "retinal_decoder.0.weight", "retinal_decoder.0.bias",
    "retinal_decoder.2.weight", "retinal_decoder.2.bias",
    "disease_head.weight", "disease_head.bias",
})

_RENAMED_PREFIXES = {
    "image_encoder.": "retinal_projection.",
    "image_gate.": "retinal_pool_gate.",
    "retinal_encoder.": "retinal_pool.",
}
_AGE_EXPANDED_WEIGHTS = frozenset({
    "clinical_encoder.0.weight", "image_encoder.0.weight",
    "retinal_encoder.0.weight", "continuous_decoder.0.weight",
    "binary_decoder.0.weight", "retinal_decoder.0.weight",
})


def _fail() -> None:
    raise ValueError(_INVALID)


def _indices(values: Iterable[int], count: int) -> tuple[int, ...]:
    try:
        result = tuple(values)
        unique = set(result)
    except (TypeError, ValueError):
        _fail()
    if (len(result) != count or len(unique) != len(result)
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 or value >= 48 for value in result)):
        _fail()
    return result


def _target_name(source_name: str) -> str:
    for old, new in _RENAMED_PREFIXES.items():
        if source_name.startswith(old):
            return new + source_name[len(old):]
    return source_name


def _source_parameter(initial: nn.Module, name: str) -> nn.Parameter:
    parameter = dict(initial.named_parameters()).get(name)
    if parameter is None:
        _fail()
    return parameter


def _validate_source(initial: object) -> tuple[dict[str, Tensor], torch.device, torch.dtype]:
    if type(initial) not in (BRANPatientStatePrototypeV1, BRANClinicalAnchorV2):
        _fail()
    if initial.config != PatientStateConfig():
        _fail()
    if (not isinstance(getattr(initial, "screening_joint_head", None), nn.Linear)
            or not isinstance(getattr(initial, "cbc_joint_head", None), nn.Linear)):
        _fail()
    c = initial.config
    screening, cbc = initial.screening_joint_head, initial.cbc_joint_head
    if (screening.in_features != c.state_dim or screening.out_features != 26
            or cbc.in_features != c.state_dim or cbc.out_features != 9):
        _fail()

    state = initial.state_dict()
    expected = _V1_KEYS | {f"{head}.{part}" for head in _SOURCE_HEADS for part in ("weight", "bias")}
    if type(initial) is BRANClinicalAnchorV2:
        expected = expected | _ANCHOR_KEYS
    if set(state) != expected:
        _fail()
    first: Tensor | None = None
    for value in state.values():
        if not isinstance(value, Tensor) or not value.is_floating_point() or not torch.isfinite(value).all():
            _fail()
        if first is None:
            first = value
        elif value.device != first.device or value.dtype != first.dtype:
            _fail()
    if first is None or first.device.type != "cpu":
        # The V2 constructor's private RNG guard is CPU-only; reject rather
        # than risk altering a caller's CUDA RNG state during construction.
        _fail()
    return state, first.device, first.dtype


def _copy_age_extended(source: Tensor, target: Tensor) -> None:
    """Copy an old trailing scalar-age column into age7's reported value."""
    if (source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]
            or target.shape[1] != source.shape[1] + 6):
        _fail()
    # The common prefix retains every non-age column exactly.  Age7 columns
    # are [reported value, interval lower, interval upper, four kind flags].
    target.zero_()
    target[:, :source.shape[1] - 1].copy_(source[:, :-1])
    target[:, source.shape[1] - 1].copy_(source[:, -1])


def initialize_from_native(
    initial: BRANPatientStatePrototypeV1,
    eligible_indices: Iterable[int],
    cbc_indices: Iterable[int],
) -> BRANMultisourceModelV2:
    """Return an exact CPU V2 MLP extension of an attached-head default V1.

    A normalized *reported* age must be supplied to the returned model as
    ``age7[:, 0]`` plus its reported one-hot flag.  Interval/censored/unknown
    age columns begin at zero, intentionally creating a zero-initialized age
    extension rather than inventing learned values.  The returned candidate
    preserves each source parameter's ``requires_grad`` flag; callers that
    want to train a frozen historical candidate must enable that explicitly.
    Only CPU sources are accepted so construction cannot touch CUDA RNG state.
    """
    eligible = _indices(eligible_indices, 43)
    cbc = _indices(cbc_indices, 9)
    if not set(cbc).issubset(eligible):
        _fail()
    source_state, device, dtype = _validate_source(initial)

    # BRANMultisourceModelV2 uses torch.random.fork_rng for its constructor;
    # all created learnable tensors below are overwritten, never retained as
    # random warm-start values.
    try:
        cls = BRANMultisourceAnchoredModelV3 if type(initial) is BRANClinicalAnchorV2 else BRANMultisourceModelV2
        target = cls("mlp", eligible, cbc).to(device=device, dtype=dtype)
    except (TypeError, ValueError, RuntimeError):
        _fail()
    target_state = target.state_dict()
    expected_target = {_target_name(name) for name in _V1_KEYS | {
        f"{head}.{part}" for head in _SOURCE_HEADS for part in ("weight", "bias")
    }} | {"eligible_slots"}
    if type(initial) is BRANClinicalAnchorV2:
        expected_target = expected_target | _ANCHOR_KEYS
    if set(target_state) != expected_target:
        _fail()

    source_parameters = dict(initial.named_parameters())
    target_parameters = dict(target.named_parameters())
    try:
        with torch.no_grad():
            for source_name, source_value in source_state.items():
                target_name = _target_name(source_name)
                target_value = target_state[target_name]
                if source_name in _AGE_EXPANDED_WEIGHTS:
                    _copy_age_extended(source_value, target_value)
                else:
                    if source_value.shape != target_value.shape:
                        _fail()
                    target_value.copy_(source_value)
                # Parameter grad policy is state-like metadata: preserve it
                # while leaving the source parameter and its .grad untouched.
                target_parameters[target_name].requires_grad_(source_parameters[source_name].requires_grad)
    except (KeyError, RuntimeError):
        _fail()
    if not torch.equal(target.eligible_slots.cpu(), torch.tensor(
            [index in set(eligible) for index in range(48)] + [False] * 11,
            dtype=torch.bool)):
        _fail()
    target.train(initial.training)
    return target
