"""Local-only frozen V5 inference and U1 application evaluation. No file I/O."""
import numpy as np
import torch

from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_multisource_age_v2 import normalize_age
from bran_multisource_batches_v2 import tensor
from bran_multisource_inference_v2 import infer_native
from bran_multisource_outcomes_v2 import original_age, subset_age
from bran_multisource_profiles_v3 import _validate_provider, _unchanged
from bran_external_cbc_evaluation_v1 import paired_counts
import bran_v5_completion_utility_readout_u1 as readout
import bran_v5_completion_utility_metrics_u1 as metrics


def require(ok):
    if not ok:
        raise ValueError('completion_utility_u1_evaluation_contract_failed') from None


def equal_local(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_local(left[k], right[k]) for k in left)
    if isinstance(left, np.ndarray):
        return left.shape == right.shape and np.allclose(left, right, rtol=0, atol=1e-10, equal_nan=True)
    return left == right


def evaluate(paired, provider, progress):
    folds = paired.folds
    n = len(folds)
    require(n >= 100 and set(np.unique(folds)) == set(range(5)))
    require(len(paired.endpoint_names) == 26 and 'mh_a1c' in paired.endpoint_names)
    names = tuple(x for x in paired.endpoint_names if x != 'mh_a1c')
    columns = [paired.endpoint_names.index(x) for x in names]
    slots = tuple(paired.names.index(x) for x in CBC_FIELDS)
    hidden, mask = readout.erase_cbc(paired.c, paired.cm, slots, paired.eligible_indices)
    require(not mask[:, slots].any() and np.all(hidden[:, slots] == 0))
    age = original_age(paired)
    estimates = {route: np.full((n, 9), np.nan) for route in ('clinical', 'both')}
    age7_by_fold = {}
    for fold in range(5):
        rows = np.flatnonzero(folds == fold)
        require(len(rows) > 0)
        progress('native_inference', fold)
        model, t = provider(fold)
        before, pin, grad = _validate_provider(model, t, fold, slots, paired.transforms[fold])
        c, cm = t.clinical(hidden, mask)
        r, rm = t.retinal(paired.r, paired.rm)
        require(not cm[:, slots].any())
        age7_by_fold[fold] = normalize_age(age, t.age_mean, t.age_scale).numpy()
        args = (tensor(c[rows]), tensor(cm[rows], torch.bool), tensor(r[rows]),
                tensor(rm[rows], torch.bool), subset_age(age, rows), t.age_mean, t.age_scale)
        for route in estimates:
            item = infer_native(model, *args, route=route)
            estimates[route][rows] = (item.cbc_standardized.numpy() * t.clinical_iqr[list(slots)]
                                      + t.clinical_median[list(slots)])
        require(_unchanged(before, pin, grad, model, t))
        progress('checkpoint_replay', fold)
        reloaded, rt = provider(fold)
        require(reloaded is not model)
        rb, rp, rg = _validate_provider(reloaded, rt, fold, slots, paired.transforms[fold])
        # Independently rebuild transformed inputs rather than reusing encoded state.
        rc, rcm = rt.clinical(hidden, mask)
        rr, rrm = rt.retinal(paired.r, paired.rm)
        rargs = (tensor(rc[rows]), tensor(rcm[rows], torch.bool), tensor(rr[rows]),
                 tensor(rrm[rows], torch.bool), subset_age(age, rows), rt.age_mean, rt.age_scale)
        for route in estimates:
            replay = infer_native(reloaded, *rargs, route=route)
            predicted = replay.cbc_standardized.numpy() * rt.clinical_iqr[list(slots)] + rt.clinical_median[list(slots)]
            require(np.allclose(predicted, estimates[route][rows], rtol=0, atol=1e-7, equal_nan=True))
        require(_unchanged(rb, rp, rg, reloaded, rt))
    kwargs = dict(c=paired.c, cm=paired.cm, age7_by_fold=age7_by_fold, folds=folds,
        labels=paired.labels[:, columns], labelmask=paired.labelmask[:, columns], cbc_slots=slots,
        eligible_indices=paired.eligible_indices, native_no_retina=estimates['clinical'],
        native_with_retina=estimates['both'])
    progress('fixed_readouts')
    result = readout.fit_predict(**kwargs)
    progress('readout_replay')
    replay = readout.fit_predict(**kwargs)
    require(equal_local(result, replay))
    del replay
    progress('aggregate_bootstrap')
    counts = paired_counts(folds, draws=1000, seed=99221)
    aggregate = metrics.summarize(result, paired.c[:, slots], kwargs['labels'], kwargs['labelmask'], folds, names, counts)
    metrics.validate_aggregate(aggregate, names)
    return aggregate
