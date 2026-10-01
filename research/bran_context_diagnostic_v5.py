"""Private training-only availability experiment; only pooled cells may be released."""
import math
import numpy as np
import torch

from bran_multisource_age_v2 import normalize_age
from bran_multisource_training_v2 import _completion_masks
from bran_multisource_preservation_v5 import bridge_masks

VIEWS = ('paired_original', 'paired_source_pattern', 'source_observed')
GROUPS = ('overall', 'low_hb_research')


def require(ok):
    if not ok: raise ValueError('context_diagnostic_contract_failed')


def cell(error, mask):
    """Private sufficient statistics, never exported or logged directly."""
    require(bool(torch.isfinite(error[mask]).all()))
    return {'count': int(mask.sum()), 'absolute': float(error[mask].abs().sum()),
            'signed': float(error[mask].sum())}


@torch.no_grad()
def diagnostic(model, paired, source, transform, step):
    require(not model.training)
    cbc = model.cbc_indices
    hb_position = 1  # Frozen CBC order: hct, hemoglobin, ...
    hb = cbc[hb_position]
    median, iqr = float(transform.clinical_median[hb]), float(transform.clinical_iqr[hb])

    def predict(batch, visible_c, visible_r):
        state = model.encode(torch.where(visible_c, batch.c, 0), visible_c,
            torch.where(visible_r[..., None], batch.r, 0), visible_r,
            normalize_age(batch.age, transform.age_mean, transform.age_scale))
        prediction = model.cbc_joint_head(state.mean)[:, hb_position]*iqr+median
        truth = batch.c[:, hb]*iqr+median
        keep = batch.cm[:, hb] & ~visible_c[:, hb] & ~state.abstain
        return prediction-truth, truth, keep

    pc, pr = _completion_masks(paired, cbc, step)
    bc, br = bridge_masks(paired, source, step)
    sc, sr = _completion_masks(source, cbc, step)
    original = predict(paired, pc, pr)
    bridge = predict(paired, pc & bc, pr & br)
    actual = predict(source, sc, sr)
    # Both paired views use identical participants for fair error comparison.
    common = original[2] & bridge[2]
    result = {}
    for name, (error, truth, keep) in zip(VIEWS, (original, bridge, actual)):
        if name != 'source_observed': keep = common
        low = keep & (truth < 12.)
        not_low = keep & (truth >= 12.)
        result[name] = {'overall': cell(error, keep), 'low_hb_research': cell(error, low),
                        'low_release_supported': int(low.sum()) >= 20 and int(not_low.sum()) >= 20}
    # Suppressed support is never released as exact counts or percentages.
    result['paired_coverage_supported'] = int(common.sum()) >= 20
    return result


def aggregate(records):
    require(type(records) is list and len(records) == 20)
    out = {}
    for view in VIEWS:
        out[view] = {}
        for group in GROUPS:
            cells = [r[view][group] for r in records
                     if r[view][group]['count'] >= 20
                     and (group == 'overall' or r[view]['low_release_supported'])]
            if len(cells) != 20:
                out[view][group] = {'status': 'withheld'}
            else:
                n = sum(c['count'] for c in cells)
                out[view][group] = {'status': 'released',
                    'mae_g_dl': sum(c['absolute'] for c in cells)/n,
                    'bias_g_dl': sum(c['signed'] for c in cells)/n}
    out['matched_paired_support'] = ('supported_all_batches' if all(
        r['paired_coverage_supported'] for r in records) else 'insufficient_for_pooled_release')
    validate(out)
    return out


def validate(value):
    require(set(value) == {*VIEWS, 'matched_paired_support'})
    require(value['matched_paired_support'] in ('supported_all_batches', 'insufficient_for_pooled_release'))
    for view in VIEWS:
        require(set(value[view]) == set(GROUPS))
        for c in value[view].values():
            if c == {'status': 'withheld'}: continue
            require(set(c) == {'status', 'mae_g_dl', 'bias_g_dl'} and c['status'] == 'released')
            require(all(type(c[k]) in (int, float) and math.isfinite(c[k]) for k in ('mae_g_dl', 'bias_g_dl')))
            require(c['mae_g_dl'] >= abs(c['bias_g_dl'])-1e-8)


def execute(sources, report):
    from bran_multisource_binding_v3 import bind_fold
    from run_bran_multisource_source_diagnostic_v1 import SOURCES, CONTEXTS, group_index, source_batch, paired_batch
    pools = {p.source: p for p in sources.pools}
    require(set(pools) == set(SOURCES))
    grouping = {name: group_index(p) for name, p in pools.items()}
    records = {(s, c): [] for s in SOURCES for c in CONTEXTS}
    for fold in range(5):
        report(fold)
        bound = bind_fold(sources, fold)
        model = bound.model.eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng_before = torch.get_rng_state().clone()
        paired_rng = np.random.default_rng(96101+fold)
        source_rng = {s: np.random.default_rng(96101+fold+1000*(j+1)) for j, s in enumerate(SOURCES)}
        paired_sampler = bound.paired_factory().private_sampler
        for repeat in range(4):
            for pattern, context in enumerate(CONTEXTS):
                paired = paired_batch(paired_sampler, paired_rng)
                for name in SOURCES:
                    source = source_batch(pools[name], bound.transform, sources.paired.names,
                        source_rng[name], grouping=grouping[name])
                    records[(name, context)].append(diagnostic(model, paired, source,
                        bound.transform, 330+54*repeat+pattern))
        require(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
        require(torch.equal(rng_before, torch.get_rng_state()))
        require(all(p.grad is None for p in model.parameters()))
    return {s: {c: aggregate(records[(s, c)]) for c in CONTEXTS} for s in SOURCES}
