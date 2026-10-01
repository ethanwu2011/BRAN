"""R7-specific erased-CBC inference; unchanged U1 numerical/readout recipe."""
from __future__ import annotations
import numpy as np
import torch
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_multisource_age_v2 import normalize_age
from bran_multisource_batches_v2 import tensor
from bran_multisource_inference_v2 import infer_native
from bran_multisource_outcomes_v2 import original_age, subset_age
from bran_external_cbc_evaluation_v1 import paired_counts
import bran_r7_fixed_state_p1 as p1
import bran_v5_completion_utility_readout_u1 as readout
import bran_v5_completion_utility_metrics_u1 as metrics
from bran_v5_completion_utility_u1 import equal_local

ERROR='r7_completion_utility_u2_contract_failed'
FLAGS={'encoder_updated':False,'readout_selection':False,'patient_level_output_emitted':False,
       'historical_gates_changed':False,'blood_draw_replacement_established':False}


def require(ok):
    if not ok:raise ValueError(ERROR) from None


def evaluate(paired,provider,progress):
    folds=paired.folds;n=len(folds)
    p1.validate_evaluation_inputs(paired,provider)
    require('mh_a1c' in paired.endpoint_names)
    names=tuple(x for x in paired.endpoint_names if x!='mh_a1c')
    require(len(names)==25)
    columns=[paired.endpoint_names.index(x) for x in names]
    slots=tuple(paired.names.index(x) for x in CBC_FIELDS)
    hidden,mask=readout.erase_cbc(paired.c,paired.cm,slots,paired.eligible_indices)
    require(not mask[:,slots].any() and np.all(hidden[:,slots]==0))
    age=original_age(paired)
    estimates={route:np.full((n,9),np.nan) for route in ('clinical','both')}
    age7={}
    for fold in range(5):
        rows=np.flatnonzero(folds==fold)
        require(len(rows)>0)
        progress('native_inference',fold)
        model,t=provider(fold)
        before,pin,grad=p1._validate_r7(model,t,fold,slots,paired.transforms[fold])
        c,cm=t.clinical(hidden,mask);r,rm=t.retinal(paired.r,paired.rm)
        require(not cm[:,slots].any())
        age7[fold]=normalize_age(age,t.age_mean,t.age_scale).numpy()
        args=(tensor(c[rows]),tensor(cm[rows],torch.bool),tensor(r[rows]),
              tensor(rm[rows],torch.bool),subset_age(age,rows),t.age_mean,t.age_scale)
        for route in estimates:
            item=infer_native(model,*args,route=route)
            estimates[route][rows]=(item.cbc_standardized.numpy()*t.clinical_iqr[list(slots)]+
                                    t.clinical_median[list(slots)])
        require(p1._role_unchanged(before,pin,grad,model,t))
        progress('checkpoint_replay',fold)
        reloaded,rt=provider(fold)
        require(reloaded is not model)
        rb,rp,rg=p1._validate_r7(reloaded,rt,fold,slots,paired.transforms[fold])
        rc,rcm=rt.clinical(hidden,mask);rr,rrm=rt.retinal(paired.r,paired.rm)
        rargs=(tensor(rc[rows]),tensor(rcm[rows],torch.bool),tensor(rr[rows]),
               tensor(rrm[rows],torch.bool),subset_age(age,rows),rt.age_mean,rt.age_scale)
        for route in estimates:
            replay=infer_native(reloaded,*rargs,route=route)
            predicted=replay.cbc_standardized.numpy()*rt.clinical_iqr[list(slots)]+rt.clinical_median[list(slots)]
            require(np.array_equal(predicted,estimates[route][rows],equal_nan=True))
        require(p1._role_unchanged(rb,rp,rg,reloaded,rt))
    kwargs=dict(c=paired.c,cm=paired.cm,age7_by_fold=age7,folds=folds,
        labels=paired.labels[:,columns],labelmask=paired.labelmask[:,columns],cbc_slots=slots,
        eligible_indices=paired.eligible_indices,native_no_retina=estimates['clinical'],
        native_with_retina=estimates['both'])
    progress('fixed_readouts',None)
    result=readout.fit_predict(**kwargs)
    progress('readout_replay',None)
    replay=readout.fit_predict(**kwargs)
    require(equal_local(result,replay))
    progress('aggregate_bootstrap',None)
    counts=paired_counts(folds,draws=1000,seed=99221)
    report=metrics.summarize(result,paired.c[:,slots],kwargs['labels'],kwargs['labelmask'],folds,names,counts)
    metrics.validate_aggregate(report,names)
    aggregate={'schema':'bran-r7-completion-utility-u2-aggregate-v1','encoder_version':'R7',
        'numerical_recipe':'unchanged_U1_historical_schema_not_model_identity',
        'report':report,'all_five_R7_inference_replayed':True,'readout_fit_replayed':True,**FLAGS}
    validate_aggregate(aggregate,names)
    return aggregate


def validate_aggregate(a,names):
    require(type(a) is dict and set(a)=={'schema','encoder_version','numerical_recipe','report',
        'all_five_R7_inference_replayed','readout_fit_replayed',*FLAGS})
    require(a['schema']=='bran-r7-completion-utility-u2-aggregate-v1' and a['encoder_version']=='R7'
        and a['numerical_recipe']=='unchanged_U1_historical_schema_not_model_identity'
        and a['all_five_R7_inference_replayed'] is True and a['readout_fit_replayed'] is True
        and all(a[k] is v for k,v in FLAGS.items()))
    metrics.validate_aggregate(a['report'],names)
