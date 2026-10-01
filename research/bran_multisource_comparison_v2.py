"""Private candidate inference and immutable-reference advancement aggregation.

No I/O, source loading, fitting, tuning or promotion. Caller authenticates all
sources/frames/references and keeps this computation within its local quiet job.
"""
import numpy as np
import torch

import bran_multisource_advancement_v2 as gates
import bran_multisource_outcomes_v2 as profile
from bran_multisource_batches_v2 import tensor, transform_hash
from bran_multisource_inference_v2 import infer_native, route_predictions, completion_predictions
from bran_external_cbc_evaluation_v1 import paired_counts
from bran_missingness_stress_metrics_v1 import summarize
import bran_missingness_stress_v1 as masking


def require(ok):
    if not ok: raise ValueError('multisource_comparison_failed')


def evaluate(paired, reference, provider, *, progress=None):
    torch.set_num_threads(2)
    n = len(paired.folds); names = tuple(paired.endpoint_names)
    require(tuple(reference.screening) == names and len(names) == 26)
    slots = tuple(paired.names.index(field) for field in gates.old.metrics.CBC_FIELDS)
    require(reference.target.shape == (n,9)
            and np.array_equal(reference.target,paired.c[:,slots],equal_nan=True))
    for j, name in enumerate(names):
        require(np.array_equal(reference.labels[name],paired.labels[:,j],equal_nan=True)
                and np.array_equal(reference.labelmask[name],paired.labelmask[:,j]))
    require(reference.replay_receipt['historical_point_metrics_replayed'] is True
            and reference.replay_receipt['patient_level_output_emitted'] is False)
    screen = {endpoint:{key:value.copy() for key,value in data.items()}
              for endpoint,data in reference.screening.items()}
    for data in screen.values():
        for arm in gates.ARMS:
            for route in gates.ROUTES: data[arm+'_'+route] = np.full(n,np.nan)
    completion = {pattern:{key:value.copy() for key,value in data.items()}
                  for pattern,data in reference.completion.items()}
    require(set(completion) == set(gates.contract.EVALPATTERNS))
    for data in completion.values():
        for arm in gates.ARMS: data[arm] = np.full((n,9),np.nan)
    stress = {arm:{pattern:np.full((n,26),np.nan) for pattern in masking.PATTERNS} for arm in gates.ARMS}
    age = profile.original_age(paired)
    for fold in range(5):
        rows = np.flatnonzero(paired.folds == fold)
        local_age = profile.subset_age(age,rows)
        for arm in gates.ARMS:
            if progress: progress({'phase':'candidate_inference','arm':arm,'fold':fold})
            model, transform = provider(arm,fold)
            require(model.arm == arm and model.cbc_indices == slots and transform.heldout_fold == fold
                    and transform_hash(transform) == transform_hash(paired.transforms[fold]))
            c,cm = transform.clinical(paired.c,paired.cm)
            r,rm = transform.retinal(paired.r,paired.rm)
            args = (tensor(c[rows]),tensor(cm[rows],torch.bool),tensor(r[rows]),tensor(rm[rows],torch.bool),
                    local_age,transform.age_mean,transform.age_scale)
            predictions = route_predictions(model,*args)
            for route,item in predictions.items():
                values = item.screening_probability.numpy()
                for j,endpoint in enumerate(names):
                    require(np.array_equal(np.isfinite(values[:,j]),np.isfinite(screen[endpoint]['initial_'+route][rows])))
                    screen[endpoint][arm+'_'+route][rows] = values[:,j]
            for pattern in completion:
                item = completion_predictions(model,*args,pattern,slots)
                require(np.array_equal(item.scoring_target_mask.numpy(),reference.observed[pattern][rows]))
                completion[pattern][arm][rows] = item.cbc_standardized.numpy()*transform.clinical_iqr[list(slots)]+transform.clinical_median[list(slots)]
            for pattern in masking.PATTERNS:
                hidden = masking.remove_inputs(c,cm,r,rm,slots,pattern)
                masking.assert_no_input_leak(hidden,cm,rm,slots,pattern)
                item = infer_native(model,tensor(hidden.clinical[rows]),tensor(hidden.clinical_mask[rows],torch.bool),
                                    tensor(hidden.retinal[rows]),tensor(hidden.retinal_mask[rows],torch.bool),
                                    local_age,transform.age_mean,transform.age_scale)
                require(np.array_equal(~item.abstained.numpy(),hidden.available[rows]))
                stress[arm][pattern][rows] = item.screening_probability.numpy()
            del model
    counts = paired_counts(paired.folds,draws=1000,seed=91501)
    missingness = dict(reference.missingness)
    for arm in gates.ARMS:
        missingness[arm] = summarize(stress[arm],paired.labels,paired.labelmask,paired.folds,names,counts)
    results = {}
    for arm in gates.ARMS:
        if progress: progress({'phase':'advancement_bootstrap','arm':arm})
        results[arm] = gates.evaluate(arm,screen,completion,missingness,reference.target,reference.observed,
                            reference.groups,reference.labels,reference.labelmask,paired.folds,names,counts)
    return {'schema':'bran-multisource-advancement-comparison-v2',
            'status':'historical_comparison_completed_pending_audit',
            'candidates':results, 'historical_replay':dict(reference.replay_receipt),
            'eligibility':{arm:results[arm]['decisions']['advancement_supported'] for arm in gates.ARMS},
            'historical_gate_definitions_changed':False,'named_fm_benchmark_complete':False,
            'external_validation_complete':False,'subtyping_established':False,
            'candidate_promoted':False,'scientific_goal_achieved':False,'patient_level_output_emitted':False}
