"""Local Q1 evaluation. Arrays remain private; only closed safe aggregates leave."""
import math
import numpy as np
import torch
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_multisource_batches_v2 import tensor
from bran_multisource_calibration_metrics_v2 import _validate_split
from bran_multisource_inference_v2 import completion_predictions
from bran_multisource_outcomes_v2 import original_age, subset_age
from bran_multisource_profiles_v3 import _validate_provider, _unchanged
from bran_v5_residual_training import PATTERNS

WHOLE=('whole_cbc_hidden','whole_cbc_no_retina')
METHODS=('native','quantile')
METRICS=('mae','bias','rmse','interval_score','mean_width','coverage')
DRAWS,SEED,MIN_VALID=1000,98351,900
FLAGS={'patient_level_output_emitted':False,'encoder_parameters_changed':False,
       'native_heads_changed':False,'protected_external_data_used':False,
       'candidate_promoted':False,'clinical_use_established':False,
       'fixed_fit_development_intervals':True,'state_only_attachment':True}

def require(ok):
    if not ok: raise ValueError('q1_evaluation_contract_failed')

def calibrate(target,native,lower,upper,support,folds,roles):
    """Fit only role0; output interval bounds only for role1. No pooled fallback."""
    _validate_split(folds,roles)
    require(target.shape==native.shape==lower.shape==upper.shape==support.shape==(len(folds),9)
            and support.dtype==bool)
    require(all(np.isfinite(a[support]).all() for a in (target,native,lower,upper))
            and (lower[support]<=upper[support]).all())
    out={method:np.full((len(folds),9,2),np.nan) for method in METHODS}
    for fold in range(5):
        for j in range(9):
            cal=(folds==fold)&(roles==0)&support[:,j]
            score=(folds==fold)&(roles==1)&support[:,j]
            n=int(cal.sum())
            if n<20: continue
            k=int(math.ceil((n+1)*.9))
            require(1<=k<=n)
            radius=np.sort(np.abs(target[cal,j]-native[cal,j]))[k-1]
            correction=max(0.,float(np.sort(np.maximum(lower[cal,j]-target[cal,j],
                                                        target[cal,j]-upper[cal,j]))[k-1]))
            out['native'][score,j,0]=native[score,j]-radius
            out['native'][score,j,1]=native[score,j]+radius
            out['quantile'][score,j,0]=lower[score,j]-correction
            out['quantile'][score,j,1]=upper[score,j]+correction
    return out

def counts_for(folds,roles):
    _validate_split(folds,roles)
    rng=np.random.default_rng(SEED); counts=np.zeros((DRAWS,len(folds)))
    for fold in range(5):
        rows=np.flatnonzero((folds==fold)&(roles==1))
        for draw in range(DRAWS):
            counts[draw,rows]=rng.multinomial(len(rows),np.full(len(rows),1/len(rows)))
    return counts

def metric(estimate,draws):
    valid=draws[np.isfinite(draws)]
    if len(valid)<MIN_VALID: return {'status':'unsupported'}
    return {'status':'supported','estimate':float(estimate),
            'ci95':[float(x) for x in np.quantile(valid,[.025,.975])]}

def mean(values,rows,counts):
    require(values.shape==rows.shape and rows.dtype==bool)
    x=values[rows]; require(len(x)>=20 and np.isfinite(x).all())
    w=counts[:,rows]; den=w.sum(axis=1)
    draws=np.divide(w@x,den,out=np.full(DRAWS,np.nan),where=den>0)
    return metric(x.mean(),draws),draws

def interval_score(y,bounds):
    lo,hi=bounds[:,0],bounds[:,1]
    return hi-lo+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0)

def group(target,predictions,bounds,rows,counts):
    if int(rows.sum())<20: return {'status':'unsupported'}
    arms={}; samples={}
    for method in METHODS:
        error=predictions[method]-target
        hit=(target>=bounds[method][:,0])&(target<=bounds[method][:,1])
        values={'mae':np.abs(error),'bias':error,'rmse':error**2,
                'interval_score':interval_score(target,bounds[method]),
                'mean_width':bounds[method][:,1]-bounds[method][:,0],'coverage':hit.astype(float)}
        arms[method]={}; samples[method]={}
        for name in METRICS:
            if name=='coverage':
                good=int(hit[rows].sum()); n=int(rows.sum())
                if 0<good<20 or 0<n-good<20:
                    arms[method][name]={'status':'withheld'}; continue
            value,draws=mean(values[name],rows,counts)
            if name=='rmse':
                draws=np.sqrt(draws); value=metric(np.sqrt(values[name][rows].mean()),draws)
            arms[method][name]=value; samples[method][name]=draws
    differences={}
    for name in METRICS:
        if any(arms[m][name]['status']!='supported' for m in METHODS):
            differences[name]={'status':'withheld'}; continue
        differences[name]=metric(arms['quantile'][name]['estimate']-arms['native'][name]['estimate'],
                                 samples['quantile'][name]-samples['native'][name])
    return {'status':'supported','arms':arms,'quantile_minus_native':differences}

def summarize(target,predictions,lowers,uppers,support,folds,roles,iqrs):
    require(target.shape==(len(folds),9) and iqrs.shape==(5,9)
            and np.isfinite(iqrs).all() and (iqrs>0).all())
    require(set(predictions)==set(lowers)==set(uppers)==set(support)==set(PATTERNS))
    counts=counts_for(folds,roles); fields={}; primary_points=[]; primary_draws=[]
    for pattern in PATTERNS:
        require(set(predictions[pattern])==set(METHODS))
        pred=predictions[pattern]; obs=support[pattern]
        require(all(x.shape==target.shape for x in pred.values()) and obs.dtype==bool)
        require(all(np.isfinite(x[obs]).all() for x in pred.values()))
        require((lowers[pattern][obs]<=pred['quantile'][obs]).all()
                and (pred['quantile'][obs]<=uppers[pattern][obs]).all())
        bounds=calibrate(target,pred['native'],lowers[pattern],uppers[pattern],obs,folds,roles)
        matched=np.isfinite(bounds['native']).all(axis=2)&np.isfinite(bounds['quantile']).all(axis=2)&obs
        require(not matched[roles==0].any())
        fields[pattern]={}
        for j,name in enumerate(CBC_FIELDS):
            rows=matched[:,j]
            localp={m:pred[m][:,j] for m in METHODS}; localb={m:bounds[m][:,j,:] for m in METHODS}
            cell={'overall':group(target[:,j],localp,localb,rows,counts)}
            if name=='hemoglobin':
                low=rows&(target[:,j]<12); other=rows&~low
                cell['low_hb_research_stratum']=(group(target[:,j],localp,localb,low,counts)
                    if int(low.sum())>=20 and (int(other.sum())==0 or int(other.sum())>=20)
                    else {'status':'withheld'})
            fields[pattern][name]=cell
            if pattern in WHOLE and int(rows.sum())>=20 and set(np.unique(folds[rows]))==set(range(5)):
                diff=(interval_score(target[:,j],localb['quantile'])-
                      interval_score(target[:,j],localb['native']))/iqrs[folds,j]
                value,draws=mean(diff,rows,counts)
                if value['status']=='supported':
                    primary_points.append(value['estimate']); primary_draws.append(draws)
    primary={'status':'unsupported'}
    if len(primary_points)==18:
        primary=metric(np.mean(primary_points),np.mean(primary_draws,axis=0))
        if primary['status']=='supported':
            coverage=[fields[p]['hemoglobin']['overall']['arms']['quantile']['coverage'] for p in WHOLE]
            guard=all(x['status']=='supported' and x['estimate']>=.85 for x in coverage)
            primary.update({'all_18_cells_supported':True,'hb_coverage_guard_passed':guard,
                            'eligible_limited_distribution_improvement':bool(primary['ci95'][1]<0 and guard)})
    result={'schema':'bran-v5-quantile-evaluation-v1','fields':fields,'primary':primary,
            'bootstrap_draws':DRAWS,'bootstrap_seed':SEED,'minimum_valid_draws':MIN_VALID,
            'interval_level':.9,**FLAGS}
    validate_result(result); return result

def valid_metric(x,signed=False,coverage=False):
    require(type(x)==dict and x.get('status') in ('supported','unsupported','withheld'))
    if x['status']!='supported': require(set(x)=={'status'}); return
    require(set(x)=={'status','estimate','ci95'} and type(x['ci95'])==list and len(x['ci95'])==2)
    require(all(type(t) in (int,float) and math.isfinite(t) for t in [x['estimate']]+x['ci95'])
            and x['ci95'][0]<=x['ci95'][1])
    if not signed: require(x['estimate']>=0 and x['ci95'][0]>=0)
    if coverage: require(x['estimate']<=1 and x['ci95'][1]<=1)

def validate_result(x):
    require(type(x)==dict and set(x)=={'schema','fields','primary','bootstrap_draws','bootstrap_seed',
                                    'minimum_valid_draws','interval_level'}|set(FLAGS))
    require(x['schema']=='bran-v5-quantile-evaluation-v1' and x['bootstrap_draws']==DRAWS
            and x['bootstrap_seed']==SEED and x['minimum_valid_draws']==MIN_VALID and x['interval_level']==.9
            and all(x[k] is v for k,v in FLAGS.items()) and set(x['fields'])==set(PATTERNS))
    for p,fields in x['fields'].items():
        require(set(fields)==set(CBC_FIELDS))
        for name,cell in fields.items():
            require(set(cell)==({'overall','low_hb_research_stratum'} if name=='hemoglobin' else {'overall'}))
            for g in cell.values():
                require(type(g)==dict and g.get('status') in ('supported','unsupported','withheld'))
                if g['status']!='supported': require(set(g)=={'status'}); continue
                require(set(g)=={'status','arms','quantile_minus_native'} and set(g['arms'])==set(METHODS)
                        and set(g['quantile_minus_native'])==set(METRICS))
                for arm in g['arms'].values():
                    require(set(arm)==set(METRICS))
                    for k,v in arm.items(): valid_metric(v,signed=k=='bias',coverage=k=='coverage')
                for v in g['quantile_minus_native'].values(): valid_metric(v,signed=True)
    p=x['primary']; require(type(p)==dict and p.get('status') in ('supported','unsupported'))
    if p['status']=='unsupported': require(set(p)=={'status'}); return
    require(set(p)=={'status','estimate','ci95','all_18_cells_supported','hb_coverage_guard_passed',
                    'eligible_limited_distribution_improvement'} and p['all_18_cells_supported'] is True)
    valid_metric({k:p[k] for k in ('status','estimate','ci95')},signed=True)
    require(all(x['fields'][pattern][field]['overall']['status']=='supported'
                for pattern in WHOLE for field in CBC_FIELDS))
    cover=[x['fields'][pattern]['hemoglobin']['overall']['arms']['quantile']['coverage'] for pattern in WHOLE]
    guard=all(v['status']=='supported' and v['estimate']>=.85 for v in cover)
    require(p['hb_coverage_guard_passed'] is guard
            and p['eligible_limited_distribution_improvement'] is bool(p['ci95'][1]<0 and guard))

def evaluate(paired,roles,provider,progress=None):
    from bran_v5_quantile_training import predict_quantiles
    torch.set_num_threads(1); _validate_split(paired.folds,roles)
    slots=tuple(paired.names.index(f) for f in CBC_FIELDS); n=len(paired.folds)
    target=paired.c[:,slots].copy(); iqrs=np.empty((5,9)); age=original_age(paired)
    predictions={p:{m:np.full((n,9),np.nan) for m in METHODS} for p in PATTERNS}
    lowers={p:np.full((n,9),np.nan) for p in PATTERNS}; uppers={p:a.copy() for p,a in lowers.items()}
    support={p:np.zeros((n,9),bool) for p in PATTERNS}; seen=[]
    for fold in range(5):
        if progress: progress('inference',fold)
        teacher,transform,head=provider(fold)
        require(not any(teacher is t for t in seen)); seen.append(teacher)
        before,thash,grads=_validate_provider(teacher,transform,fold,slots,paired.transforms[fold])
        headbefore={k:v.detach().clone() for k,v in head.state_dict().items()}
        rows=np.flatnonzero(paired.folds==fold); c,cm=transform.clinical(paired.c,paired.cm)
        r,rm=transform.retinal(paired.r,paired.rm)
        args=(tensor(c[rows]),tensor(cm[rows],torch.bool),tensor(r[rows]),tensor(rm[rows],torch.bool),
              subset_age(age,rows),transform.age_mean,transform.age_scale)
        scale=transform.clinical_iqr[list(slots)]; center=transform.clinical_median[list(slots)]
        iqrs[fold]=scale; cached={}
        for pattern in PATTERNS:
            q=predict_quantiles(teacher,head,*args,pattern,slots)
            ref=completion_predictions(teacher,*args,pattern,slots)
            mask=ref.scoring_target_mask.numpy()
            require(torch.equal(q.scoringmask,ref.scoring_target_mask)
                    and torch.equal(q.abstained,ref.abstained))
            support[pattern][rows]=mask
            for method,arr in (('native',ref.cbc_standardized),('quantile',q.median)):
                predictions[pattern][method][rows]=np.where(mask,arr.numpy()*scale+center,np.nan)
            lowers[pattern][rows]=np.where(mask,q.q05.numpy()*scale+center,np.nan)
            uppers[pattern][rows]=np.where(mask,q.q95.numpy()*scale+center,np.nan)
            cached[pattern]=(q.q05.clone(),q.median.clone(),q.q95.clone(),q.scoringmask.clone(),q.abstained.clone())
        require(_unchanged(before,thash,grads,teacher,transform)
                and all(torch.equal(v,head.state_dict()[k]) for k,v in headbefore.items()))
        if progress: progress('checkpoint_replay',fold)
        restored,rt,rhead=provider(fold)
        require(restored is not teacher and rhead is not head)
        rb,rh,rg=_validate_provider(restored,rt,fold,slots,paired.transforms[fold])
        require(rh==thash and all(torch.equal(before[k],rb[k]) for k in before))
        require(all(torch.equal(v,rhead.state_dict()[k]) for k,v in headbefore.items()))
        for pattern in PATTERNS:
            q=predict_quantiles(restored,rhead,*args,pattern,slots)
            for actual,expected in zip((q.q05,q.median,q.q95,q.scoringmask,q.abstained),cached[pattern]):
                require(np.array_equal(actual.numpy(),expected.numpy(),equal_nan=True))
        require(_unchanged(rb,rh,rg,restored,rt))
    if progress: progress('aggregate_bootstrap',None)
    result=summarize(target,predictions,lowers,uppers,support,paired.folds,roles,iqrs)
    if progress: progress('aggregate_replay',None)
    require(result==summarize(target,predictions,lowers,uppers,support,paired.folds,roles,iqrs))
    return result
