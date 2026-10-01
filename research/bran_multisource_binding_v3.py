"""Private local binding of qualified sources to unchanged native fold frames.

Caller MUST own the heavy-job lock and FD-quiet boundary. No serialization or
logging. Protected external sources are never admitted by this loader.
"""
import copy
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import torch

import bran_multisource_reference_roles_v2 as roles
import bran_multisource_reference_predictions_v2 as references
from run_bran_multisource_fit_v2 import load_sources, source_closure
from bran_multisource_age_v2 import AgeBatch, normalize_age
from bran_multisource_batches_v2 import PairedBatchesV2, UnpairedBatchesV2, tensor, transform_hash
from bran_multisource_native_transform_v3 import inherit_native_transform
from bran_multisource_warmstart_v3 import initialize_from_native
from bran_multisource_fit_v2 import _TrackedChoice
from bran_clinical_semantics_v1 import CBC_FIELDS

PROTOCOL_PIN = 'ce793a9f902b27b6f5c09f48c8b659ee98c5c5d251d61436014e505e1642d5da'
AUDIT_PIN = '4191cdb3b6965ac65d6c0eb43d6613079668b0065178801713953304def0020c'


def require(ok):
    if not ok:
        raise ValueError('multisource_native_binding_invalid')


@dataclass(repr=False)
class BoundSourcesV3:
    paired: object
    pools: tuple
    context: object
    native_transforms: tuple
    source_evidence: dict
    artifact_pins: dict

    def receipt(self):
        return {'source_roles': source_closure(self.source_evidence),
                'artifact_pins': self.artifact_pins,
                'reference_protocol_sha256': PROTOCOL_PIN, 'reference_audit_sha256': AUDIT_PIN,
                'outer_fold_sha256': self.paired.receipt['outer_fold_sha256'],
                'inner_fold_sha256': self.paired.receipt['inner_fold_sha256'],
                'transform_sha256': [transform_hash(t) for t in self.paired.transforms],
                'paired_retinal_input': 'retained_native_original_bytes_in_authenticated_equivalent_V3_frame',
                'normalization': 'retained_native_fold_values_not_refit_V2_values',
                'protected_sources_used': False, 'patient_level_output_emitted': False}


def load_bound_sources():
    sources = load_sources()
    context = roles.authenticate_references(PROTOCOL_PIN, AUDIT_PIN)
    native = roles.old.origin.native.source
    names = references._endpoint_names(sources.paired, context)
    _, folds, c, cm, eligible, r, rm, ages, _, _, _ = references._align(
        sources.paired, native, names, context)
    originals, mapped = [], []
    for fold in range(5):
        tr = np.flatnonzero(folds != fold)
        original = native.base.FoldTransform(c, cm, eligible, r, rm, ages, tr)
        originals.append(original)
        mapped.append(inherit_native_transform(original, sources.paired.eligible_indices, folds, fold))
    # Deliberately preserve the native paired feature bytes. V3 source features
    # have already passed the unchanged historical retinal equivalence contract.
    paired = replace(sources.paired, r=r, transforms=tuple(mapped))
    return BoundSourcesV3(paired, sources.pools, context, tuple(originals),
                          sources.source_evidence, sources.artifact_pins)


def bind_fold(sources, fold):
    require(type(fold) is int and fold in range(5))
    paired, transform = sources.paired, sources.paired.transforms[fold]
    initial = roles.old.origin.load_initial(fold, sources.native_transforms[fold],
                                            roles._thaw(sources.context.protocol))
    cbc = tuple(paired.names.index(name) for name in CBC_FIELDS)
    model = initialize_from_native(initial, paired.eligible_indices, cbc)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    teacher = copy.deepcopy(model)
    missing = np.full(len(paired.age_value), np.nan)
    age = AgeBatch(tensor(paired.age_value), tensor(missing), tensor(missing),
                   tensor(paired.age_kind, torch.long))
    seed = 95101+fold

    def paired_factory():
        sampler = PairedBatchesV2(paired.c, paired.cm, paired.r, paired.rm, age,
            paired.labels, paired.labelmask, paired.folds, transform, seed+202)
        sampler.rng = _TrackedChoice(sampler.rng)
        return SimpleNamespace(sample=lambda: sampler.sample(96, supervised=True),
            positive_weights=sampler.positive_weights(), private_sampler=sampler)

    def source_factory():
        sampler = UnpairedBatchesV2(sources.pools, transform, paired.names, seed+404)
        return SimpleNamespace(sample=lambda: sampler.sample(128), private_sampler=sampler)

    # Scale uses proper-training physiology only; neither targets nor heldout
    # people determine this fixed normalization of the preservation objective.
    c, cm = transform.clinical(paired.c, paired.cm)
    r, rm = transform.retinal(paired.r, paired.rm)
    tr = np.flatnonzero(paired.folds != fold)
    states = []
    with torch.no_grad():
        for start in range(0, len(tr), 256):
            idx = tr[start:start+256]
            part_age = AgeBatch(*(getattr(age, key)[idx] for key in ('value', 'lower', 'upper', 'kind')))
            state = teacher.encode(tensor(c[idx]), tensor(cm[idx], torch.bool),
                tensor(r[idx, None]), tensor(rm[idx, None], torch.bool),
                normalize_age(part_age, transform.age_mean, transform.age_scale))
            states.append(state.mean[~state.abstain])
    joined = torch.cat(states)
    require(len(joined) >= 20)
    scale = joined.std(dim=0, correction=0).clamp_min(1).detach()
    require(bool(torch.isfinite(scale).all()))
    return SimpleNamespace(model=model, teacher=teacher, transform=transform,
        paired_factory=paired_factory, source_factory=source_factory,
        state_scale=scale, cbc_indices=cbc, seed=seed)
