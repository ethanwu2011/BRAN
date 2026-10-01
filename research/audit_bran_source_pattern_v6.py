"""Aggregate-only terminal admission for V6 reporting; no arrays/models loaded."""
import hashlib
import json
import math
from pathlib import Path

import bran_source_pattern_metrics_v6 as gate
import bran_source_pattern_evaluation_v6 as evaluation

ERROR = 'source_pattern_v6_aggregate_audit_failed'
ROOT = Path(__file__).resolve().parent


def require(ok):
    if not ok: raise ValueError(ERROR) from None


def sha(path):
    require(path.is_file() and not path.is_symlink())
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):
    require(path.stat().st_size<=5000000 and not path.is_symlink())
    def unique(items):
        d={}
        for k,v in items:
            require(k not in d);d[k]=v
        return d
    value=json.loads(path.read_text(),object_pairs_hook=unique,
        parse_constant=lambda _:require(False))
    require(type(value)is dict)
    return value


def number(value,lower=-math.inf,upper=math.inf):
    require(type(value) in (int,float) and math.isfinite(value) and lower<=value<=upper)


def interval(value,lower,upper):
    require(type(value)is list and len(value)==2)
    for x in value:number(x,lower,upper)
    require(value[0]<=value[1])


def cell(value,metric,*,extras=False,roles=gate.ROLES,contrasts=gate.CONTRASTS):
    if value=={'status':'unsupported'}:return
    require(type(value)is dict and set(value)=={'status','arms','contrasts'} and value['status']=='supported')
    require(set(value['arms'])==set(roles) and set(value['contrasts'])==set(contrasts))
    maximum=1 if metric=='auroc' else math.inf
    for item in value['arms'].values():
        require(set(item)==({metric,'ci95','bias','rmse'} if extras else {metric,'ci95'}))
        number(item[metric],0,maximum);interval(item['ci95'],0,maximum)
        if extras:
            number(item['bias']);number(item['rmse'],0)
            require(abs(item['bias'])<=item['mae']+1e-9 and item['mae']<=item['rmse']+1e-9)
    for key,(left,right) in contrasts.items():
        item=value['contrasts'][key]
        require(set(item)=={'delta','ci95'})
        number(item['delta'],-maximum,maximum);interval(item['ci95'],-maximum,maximum)
        require(abs(item['delta']-(value['arms'][left][metric]-value['arms'][right][metric]))<1e-9)


def coverage(value):
    if value=={'status':'withheld'}:return
    require(type(value)is dict and set(value)=={'status','supported','total'} and value['status']=='released')
    n,k=value['total'],value['supported']
    require(type(n)is int and type(k)is int and n>=20 and 0<=k<=n
        and (k==0 or k>=20) and (n-k==0 or n-k>=20))


def screen(value,names,*,roles=gate.ROLES,contrasts=gate.CONTRASTS):
    require(type(value)is dict and set(value)=={'complete_26_panel','macro','endpoints',
        'prediction_coverage','matched_population_coverage'} and type(value['complete_26_panel'])is bool)
    require(set(value['endpoints'])==set(names) and set(value['prediction_coverage'])==set(roles))
    for item in value['endpoints'].values():cell(item,'auroc',roles=roles,contrasts=contrasts)
    complete=all(v['status']=='supported' for v in value['endpoints'].values())
    require(value['complete_26_panel']==complete)
    if complete:
        cell(value['macro'],'auroc',roles=roles,contrasts=contrasts)
        if value['macro']['status']=='supported':
            for r in roles:
                mean=sum(v['arms'][r]['auroc'] for v in value['endpoints'].values())/26
                require(abs(mean-value['macro']['arms'][r]['auroc'])<1e-9)
    else:require(value['macro'] is None)
    for item in value['prediction_coverage'].values():coverage(item)
    coverage(value['matched_population_coverage'])


def validate_native(value):
    evaluation.validate_result(value)
    names=tuple(value['screening']['both']['endpoints'])
    require(len(names)==26 and all(type(n)is str and n and len(n)<100 and '\n' not in n for n in names))
    for panels in (value['screening'],value['stress_profiles'],value['age_profiles']):
        for item in panels.values():screen(item,names)
    stress=value['stress_primary']
    require(set(stress)=={'status','complete_26_panel','summary','patterns','coverage','endpoint_count'}
        and type(stress['complete_26_panel'])is bool and type(stress['endpoint_count'])is int
        and 0<=stress['endpoint_count']<=26 and set(stress['coverage'])==set(gate.PATTERNS))
    for item in stress['coverage'].values():
        require(set(item)=={'arms','matched'} and set(item['arms'])==set(gate.ROLES))
        for v in item['arms'].values():coverage(v)
        coverage(item['matched'])
    if stress['complete_26_panel']:
        require(stress['endpoint_count']==26 and set(stress['patterns'])==set(gate.PATTERNS))
        cell(stress['summary'],'auroc')
        require(stress['status']==stress['summary']['status'])
        for v in stress['patterns'].values():cell(v,'auroc')
        if stress['status']=='supported':
            for r in gate.ROLES:
                require(abs(stress['summary']['arms'][r]['auroc']-
                    sum(v['arms'][r]['auroc'] for v in stress['patterns'].values())/5)<1e-9)
    else:
        require(stress['status']=='unsupported' and stress['endpoint_count']<26
                and stress['summary'] is None and stress['patterns']=={})
    for panel in value['completion'].values():
        require(set(panel)==set(gate.CBC_FIELDS))
        for field,v in panel.items():
            require(set(v)=={'prediction_coverage_among_observed','groups'})
            coverage(v['prediction_coverage_among_observed'])
            expected={'overall','hb_below_12_research_stratum'} if field=='hemoglobin' else {'overall'}
            require(set(v['groups'])==expected)
            for item in v['groups'].values():cell(item,'mae',extras=True)
    require(set(value['historical_low_hb'])=={'whole_cbc_hidden','whole_cbc_no_retina'})
    for v in value['historical_low_hb'].values():cell(v,'mae')
    checks=gate.protected_completion_checks(value['completion'],value['historical_low_hb'])
    require(value['decision_pending_authentication']==gate.decide(stress,value['screening']['both'],checks,False))
    return checks


def authenticate_native(attempt=1,fit_attempt=1):
    """Only sealed public aggregate/protocol/component bytes, never checkpoints."""
    try:
        import run_bran_source_pattern_v6 as training
        import run_bran_source_pattern_evaluation_v6 as runner
        require(type(attempt)is int and 1<=attempt<=99 and type(fit_attempt)is int and 1<=fit_attempt<=99)
        out=ROOT/f'BRAN_SOURCE_PATTERN_V6_NATIVE_EVALUATE_ATTEMPT{attempt}'
        fit=ROOT/f'BRAN_SOURCE_PATTERN_V6_FIT_ATTEMPT{fit_attempt}'
        for directory in (out,fit):
            require(directory.is_dir() and not directory.is_symlink() and not (directory/'failure.json').exists())
        p,a,t=(read(out/n) for n in ('protocol.json','aggregate.json','completed.json'))
        fp,fa,ft=(read(fit/n) for n in ('protocol.json','aggregate.json','completed.json'))
        require(fp['parameters']==training.PARAMETERS and fp['evaluation']==training.EVALUATION
            and p['parameters']==training.EVALUATION and fp['stage']=='fit'
            and fp['code_sha256']==training.code_hashes() and p['code_sha256']==runner.code_hashes())
        for directory,protocol,aggregate,terminal in ((out,p,a,t),(fit,fp,fa,ft)):
            require(terminal['status']=='authenticated_completed' and terminal['patient_level_output_emitted']is False
                and terminal['protocol_sha256']==sha(directory/'protocol.json')
                and terminal['aggregate_sha256']==sha(directory/'aggregate.json')
                and aggregate['patient_level_output_emitted']is False and aggregate['candidate_promoted']is False)
            for name,pin in protocol['code_sha256'].items():
                require(type(name)is str and Path(name).name==name and sha(ROOT/name)==pin)
        require(p['schema']=='bran-source-pattern-v6-evaluation-protocol' and p['stage']=='native'
            and p['status']=='frozen_before_evaluation' and p['protected_sources_used']is False
            and p['automatic_promotion']is False and p['source_binding']==fp['source_binding'])
        require(p['fit_receipt']=={'protocol_sha256':ft['protocol_sha256'],
            'aggregate_sha256':ft['aggregate_sha256'],'terminal_sha256':sha(fit/'completed.json')})
        require(set(fa['component_sha256'])=={f'fold{f}_{r}.json' for f in range(5) for r in ('C','S')})
        for f in range(5):
            traces=[]
            for r in ('C','S'):
                name=f'fold{f}_{r}.json';require(sha(fit/name)==fa['component_sha256'][name])
                item=read(fit/name)
                require(item['fold']==f and item['role']==r and item['updates_completed']==3000
                    and item['checkpoint_reload_exact']is True and item['patient_level_output_emitted']is False)
                binding=item['binding']
                require(binding['fold']==f and binding['role']==r and binding['protocol_sha256']==ft['protocol_sha256']
                    and binding['outer_fold_sha256']==fp['source_binding']['outer_fold_sha256']
                    and binding['inner_fold_sha256']==fp['source_binding']['inner_fold_sha256'][f])
                traces.append({k:item[k] for k in ('paired_input_digest','paired_completion_mask_digest',
                    'bridge_mask_digest','source_values_digest','source_availability_digest')})
            require(traces[0]==traces[1])
        require(set(a)=={'schema','status','stage','result','research_lead_decision','historical_role_adapter',
            'historical_gate_changed','candidate_promoted','scientific_goal_achieved','patient_level_output_emitted'})
        require(a['schema']=='bran-source-pattern-v6-evaluation-terminal' and a['status']=='completed'
            and a['stage']=='native' and a['historical_gate_changed']is False
            and a['scientific_goal_achieved']is False and a['historical_role_adapter']is None)
        checks=validate_native(a['result'])
        require(a['research_lead_decision']==gate.decide(a['result']['stress_primary'],
            a['result']['screening']['both'],checks,True))
        return a,{'protocol_sha256':t['protocol_sha256'],'aggregate_sha256':t['aggregate_sha256'],
            'fit_protocol_sha256':ft['protocol_sha256'],'fit_aggregate_sha256':ft['aggregate_sha256'],
            'status':'aggregate_terminal_authenticated','checkpoint_bytes_reloaded_here':False,
            'source_rows_read':False,'patient_level_output_emitted':False}
    except Exception:
        raise ValueError(ERROR) from None


def public_terminal(directory):
    """Authenticate bounded public artifacts and exact currently pinned code."""
    require(directory.is_dir() and not directory.is_symlink() and not (directory/'failure.json').exists())
    p,a,t=(read(directory/n) for n in ('protocol.json','aggregate.json','completed.json'))
    require(set(t)=={'status','protocol_sha256','aggregate_sha256','patient_level_output_emitted'}
        and t['status']=='authenticated_completed' and t['patient_level_output_emitted']is False
        and p['patient_level_output_emitted']is False and a['patient_level_output_emitted']is False
        and t['protocol_sha256']==sha(directory/'protocol.json')
        and t['aggregate_sha256']==sha(directory/'aggregate.json'))
    for name,pin in p['code_sha256'].items():
        require(type(name)is str and Path(name).name==name and sha(ROOT/name)==pin)
    return p,a,t


def authenticate_queue(attempt=1):
    """Admit native, unchanged historical gate and sealed C/M readout together."""
    native,receipt=authenticate_native(attempt,attempt)
    np=read(ROOT/f'BRAN_SOURCE_PATTERN_V6_NATIVE_EVALUATE_ATTEMPT{attempt}'/'protocol.json')
    hp,h,ht=public_terminal(ROOT/f'BRAN_SOURCE_PATTERN_V6_HISTORICAL_EVALUATE_ATTEMPT{attempt}')
    import run_bran_multisource_evaluation_v3 as legacy
    import bran_source_value_readout_v6 as readout
    import run_bran_source_pattern_evaluation_v6 as runner
    import run_bran_v5_cbc_uncertainty as parent
    require(hp['schema']=='bran-source-pattern-v6-evaluation-protocol' and hp['stage']=='historical'
        and hp['status']=='frozen_before_evaluation' and hp['parameters']==np['parameters']
        and hp['code_sha256']==runner.code_hashes() and hp['fit_receipt']==np['fit_receipt']
        and hp['source_binding']==np['source_binding'] and hp['protected_sources_used']is False
        and hp['automatic_promotion']is False and set(h)==set(native)
        and h['schema']==native['schema'] and h['stage']=='historical' and h['status']=='completed'
        and h['historical_role_adapter']=={'C':'V6_control','M':'V6_source_pattern_candidate'}
        and h['historical_gate_changed']is False and h['candidate_promoted']is False
        and h['scientific_goal_achieved']is False and h['research_lead_decision']is None)
    legacy.validate_result(h['result'])
    cp,c,ct=public_terminal(ROOT/f'BRAN_SOURCE_VALUE_READOUT_V6_ATTEMPT{attempt}')
    code=set(parent.CODE)|{'bran_source_value_readout_v6.py','run_bran_source_value_readout_v6.py',
        'test_bran_source_value_readout_v6.py','bran_v5_state_routes.py','bran_v5_residual_training.py'}
    require(cp['schema']=='bran-v5-source-value-readout-protocol' and cp['status']=='frozen_before_readouts'
        and cp['parameters']==readout.PARAMETERS and cp['source_binding']==np['source_binding']
        and cp['code_sha256']=={n:sha(ROOT/n) for n in sorted(code)}
        and cp['baseline_record_sha256']==np['baseline_record_sha256']==sha(parent.BASELINE))
    require(set(c)=={'schema','parameters','routes','all_state_inference_replayed','encoder_updated',
        'candidate_promoted','protected_sources_used','patient_level_output_emitted'}
        and c['schema']=='bran-v5-source-value-fixed-readout' and c['parameters']==readout.PARAMETERS
        and c['all_state_inference_replayed']is True and c['encoder_updated']is False
        and c['candidate_promoted']is False and c['protected_sources_used']is False
        and set(c['routes'])=={'both','clinical','retinal'})
    names=tuple(native['result']['screening']['both']['endpoints'])
    for panel in c['routes'].values():screen(panel,names,roles=('C','M'),contrasts={'M_minus_C':('M','C')})
    return {'status':'queue_aggregate_terminals_authenticated','native_receipt':receipt,
        'research_lead_decision':native['research_lead_decision'],
        'historical_promotion_eligible':h['result']['promotion_eligible'],
        'historical_receipt':ht,'source_value_receipt':ct,
        'source_value_macro':{route:panel['macro'] for route,panel in c['routes'].items()},
        'source_value_scope':'source_values_beyond_availability_exposure_not_paired_only_comparison',
        'scientific_goal_achieved':False,'patient_level_output_emitted':False}
