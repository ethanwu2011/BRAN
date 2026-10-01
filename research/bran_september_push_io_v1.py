"""Private local checkpoint/context access. Call only inside the FD-quiet boundary."""
import json
from pathlib import Path
import numpy as np
from run_bran_source_linkage_audit_v1 import sha
import run_bran_joint_lab_comparison_v1 as joint

ROOT = Path(__file__).resolve().parent
PINS = {
    'BRAN_JOINT_LAB_COMPARISON_PROTOCOL_V1.json': '8a1d8e483478a636d607c439e51eaf265eb26b240df353e45e865bc850877297',
    'BRAN_JOINT_LAB_COMPARISON_V1/aggregate.json': 'e3f1abd7d3f76805ebc82b7d0c5ab4222e95a550f9b7d7f15750069602644370',
    'BRAN_JOINT_LAB_COMPARISON_AUDIT_V1/audit.json': '9d4fff543f136f5fd26dce051a88f1652475cc144315fe28cff0274c79e62434',
}


def source_receipt():
    for name, digest in PINS.items():
        if sha(ROOT/name) != digest:
            raise ValueError('prior_identity_changed')
    if (joint.OUT/'failure.json').exists():
        raise ValueError('prior_terminal_conflict')
    p = json.loads(joint.PROTOCOL.read_text())
    joint.validate_protocol(p)
    m = json.loads((joint.OUT/'manifest.json').read_text())
    a = json.loads((ROOT/'BRAN_JOINT_LAB_COMPARISON_AUDIT_V1/audit.json').read_text())
    if m['aggregate_sha256'] != PINS['BRAN_JOINT_LAB_COMPARISON_V1/aggregate.json'] or m['checkpoint_sha256'] != a['checkpoint_sha256']:
        raise ValueError('prior_checkpoint_binding_changed')
    for f in range(5):
        path = joint.PRIVATE/('fold'+str(f)+'.pt')
        if sha(path) != m['checkpoint_sha256']['fold'+str(f)] or path.stat().st_mode & 0o777 != 0o600:
            raise ValueError('prior_checkpoint_changed')
    return {'pins': PINS, 'manifest_sha256': sha(joint.OUT/'manifest.json'),
            'checkpoint_sha256': m['checkpoint_sha256'], 'authentication': p['paired_authentication'],
            'endpoint_names': p['endpoint_names']}


def load_context():
    old = joint.old_protocol()
    ctx, folds = joint.prior.context(old)
    c0, cm0, eligible, r0, rm, names = joint.prior.lineage._actual_arrays(ROOT, ctx)
    eligible[:,48:] = False
    if tuple(np.flatnonzero(eligible[0,:48])) != joint.prior.lineage.ELIGIBLE_CONTINUOUS_INDICES or not np.array_equal(eligible, np.broadcast_to(eligible[0], eligible.shape)):
        raise ValueError('clinical_eligibility_changed')
    return ctx, folds, c0, cm0, eligible, r0, rm, np.asarray(ctx['raw_cohort'].ages), tuple(names)


def load_control(fold, transform, receipt):
    import torch
    from bran_patient_state_anchor_v2 import BRANClinicalAnchorV2
    from audit_bran_external_cbc_comparison_v1 import validate_checkpoint
    path = joint.PRIVATE/('fold'+str(fold)+'.pt')
    expected = receipt['checkpoint_sha256']['fold'+str(fold)]
    if sha(path) != expected or path.stat().st_mode & 0o777 != 0o600:
        raise ValueError('checkpoint_identity_failed')
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    model = BRANClinicalAnchorV2()
    validate_checkpoint(bundle, model.state_dict(), PINS['BRAN_JOINT_LAB_COMPARISON_PROTOCOL_V1.json'])
    for key in ('clinical_median','clinical_iqr','retinal_mean','retinal_scale','age_mean','age_scale'):
        if not np.array_equal(bundle[key], getattr(transform,key)):
            raise ValueError('outer_train_normalizer_replay_failed')
    model.load_state_dict(bundle['control'], strict=True)
    model.eval()
    if sha(path) != expected:
        raise ValueError('checkpoint_changed_during_load')
    return model


def code_closure(extra):
    old = joint.old_protocol()
    p = json.loads(joint.PROTOCOL.read_text())
    names = set(old['code_sha256']) | set(p['code_sha256']) | set(extra)
    return {name: sha(ROOT/name) for name in sorted(names)}


def runtime():
    return joint.runtime()
