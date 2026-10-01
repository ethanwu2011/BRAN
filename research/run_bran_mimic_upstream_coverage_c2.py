"""One source-authenticated, FD-quiet upstream coverage scan. No model fitting."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json
from diagnose_bran_agefree_reference_failure_v1 import safe_trace

ROOT = Path(__file__).resolve().parent
DESIGN = ROOT/'BRAN_MIMIC_UPSTREAM_COVERAGE_C2_DESIGN.md'
OWN_CODE = ('bran_mimic_upstream_coverage_c2.py', 'test_bran_mimic_upstream_coverage_c2.py',
            'run_bran_mimic_upstream_coverage_c2.py', 'test_run_bran_mimic_upstream_coverage_c2.py',
            'run_bran_mimic_upstream_coverage_c2_attempt1.sh',
            'diagnose_bran_agefree_reference_failure_v1.py', 'run_bran_multisource_retinal_features_v2.py')
PHASES = ('authentication', 'source_hashes', 'metadata', 'lab_scan', 'gate_replay', 'post_authentication', 'completed')


def require(ok):
    if not ok: raise ValueError('mimic_upstream_coverage_c2_contract_failed') from None


def paths(attempt):
    require(type(attempt) is int and 1 <= attempt <= 99)
    return ROOT/f'BRAN_MIMIC_UPSTREAM_COVERAGE_C2_ATTEMPT{attempt}'


def prior(*, with_map=False):
    """Authenticate only the source/map evidence used here, never outcome arrays."""
    import run_bran_mimic_joint_labs_v1 as parent
    import run_bran_mimic_landmark_linkage_v1 as linked
    p = json.loads(parent.PROTOCOL.read_text()); parent.validate_protocol(p)
    lp = json.loads(linked.PROTOCOL.read_text()); lpin = sha(linked.PROTOCOL)
    require(lp['parent_protocol_sha256'] == sha(parent.PROTOCOL))
    require(set(lp['source_sha256']) == set(linked.INPUTS))
    for name, pin in lp['code_sha256'].items(): require(sha(ROOT/name) == pin)
    require(all(lp['source_sha256'][key] == p['source_files']['mimic_'+key]['sha256'] for key in linked.INPUTS))
    require(not (linked.OUT/'failure.json').exists() and not (linked.AUDIT/'failure.json').exists())
    a = json.loads((linked.OUT/'aggregate.json').read_text()); linked.validate_result(a)
    apin = sha(linked.OUT/'aggregate.json')
    require(json.loads((linked.OUT/'aggregate.manifest.json').read_text()) ==
            {'protocol_sha256':lpin,'artifact_sha256':apin})
    audit = json.loads((linked.AUDIT/'audit.json').read_text()); linked.validate_audit(audit,lpin,apin)
    require(json.loads((linked.AUDIT/'audit.manifest.json').read_text()) ==
            {'protocol_sha256':lpin,'artifact_sha256':sha(linked.AUDIT/'audit.json')})
    private_map = linked.PRIVATE/'row_map.json'
    require(private_map.is_file() and not private_map.is_symlink()
            and private_map.stat().st_mode & 0o777 == 0o600)
    map_pin = a['private_sha256']['row_map.json']; require(sha(private_map) == map_pin)
    pairs = None
    if with_map:
        value = json.loads(private_map.read_text())
        require(sha(private_map) == map_pin and set(value) == {'schema','source_sha256','rows'})
        require(value['schema'] == 'bran-mimic-linked-snapshot-map-v1'
                and value['source_sha256'] == lp['source_sha256'])
        require(all(set(r) == {'person','episode','row_binding'} for r in value['rows']))
        pairs = {(r['person'],r['episode']) for r in value['rows']}
        require(len(pairs) == len(value['rows']))
    pins = {'parent_protocol_sha256':sha(parent.PROTOCOL),'landmark_protocol_sha256':lpin,
        'landmark_aggregate_sha256':apin,'landmark_audit_sha256':sha(linked.AUDIT/'audit.json'),
        'legacy_gate_map_sha256':map_pin,'source_sha256':lp['source_sha256']}
    return pins, pairs


def code_hashes():
    import run_bran_mimic_landmark_linkage_v1 as linked
    return {name:sha(ROOT/name) for name in sorted(set(linked.CODE)|set(OWN_CODE))}


def source_hashes(pins):
    import run_bran_mimic_landmark_linkage_v1 as linked
    require(set(pins['source_sha256']) == set(linked.INPUTS))
    for key, path in linked.INPUTS.items(): require(sha(path) == pins['source_sha256'][key])


def validate_protocol(p):
    require(type(p) is dict and set(p) == {'schema','status','sources','design_sha256','code_sha256',
        'max_metadata_rows','max_lab_rows','lab_passes','no_model_inference_or_fitting',
        'clinical_outcomes_read','patient_level_output_permitted'})
    require(p['schema']=='bran-mimic-upstream-coverage-c2-protocol' and p['status']=='frozen_before_scan'
        and p['max_metadata_rows']==2000000 and p['max_lab_rows']==200000000 and p['lab_passes']==1
        and p['no_model_inference_or_fitting'] is True and p['clinical_outcomes_read'] is False
        and p['patient_level_output_permitted'] is False)
    require(p['code_sha256']==code_hashes() and p['design_sha256']==sha(DESIGN) and p['sources']==prior()[0])


def validate_terminal(done, protocol_pin, aggregate_pin):
    require(type(done) is dict and set(done)=={'status','protocol_sha256','aggregate_sha256','elapsed_seconds',
        'old_gate_episode_membership_reproduced_exactly','full_source_scan_completed',
        'patient_level_output_emitted','scientific_goal_achieved'})
    require(done['status']=='authenticated_completed' and done['protocol_sha256']==protocol_pin
        and done['aggregate_sha256']==aggregate_pin and type(done['elapsed_seconds']) in (float,int)
        and math.isfinite(done['elapsed_seconds']) and done['elapsed_seconds']>=0
        and done['old_gate_episode_membership_reproduced_exactly'] is True
        and done['full_source_scan_completed'] is True and done['patient_level_output_emitted'] is False
        and done['scientific_goal_achieved'] is False)


def run(attempt, state):
    import bran_mimic_landmark_source_v1 as source
    import run_bran_mimic_landmark_linkage_v1 as linked
    import run_bran_mimic_joint_labs_v1 as parent
    import bran_mimic_upstream_coverage_c2 as coverage
    from bran_clinical_source_reader_v1 import iter_projected_csv
    from bran_joint_lab_online_v1 import authenticate_code_binding
    out=paths(attempt); require(not out.exists() and not out.is_symlink()); out.mkdir();state['out']=out
    start=time.monotonic()
    def progress(phase, bucket=None):
        require(phase in PHASES and (bucket is None or type(bucket) is int and bucket>=0 and bucket%1000000==0))
        state['phase']=phase
        write_json(out/'progress.next.json',{'phase':phase,'pid':os.getpid(),'elapsed_seconds':time.monotonic()-start,
            'processed_lab_rows_lower_bound_million':bucket,'patient_level_output_emitted':False})
        os.replace(out/'progress.next.json',out/'progress.json')
    progress('authentication'); pins, expected_pairs=prior(with_map=True)
    mapping=parent.mapping()
    protocol={'schema':'bran-mimic-upstream-coverage-c2-protocol','status':'frozen_before_scan',
        'sources':pins,'design_sha256':sha(DESIGN),'code_sha256':code_hashes(),
        'max_metadata_rows':2000000,'max_lab_rows':200000000,'lab_passes':1,
        'no_model_inference_or_fitting':True,'clinical_outcomes_read':False,'patient_level_output_permitted':False}
    validate_protocol(protocol)
    write_json(out/'protocol.json',protocol);pin=sha(out/'protocol.json')
    progress('source_hashes');source_hashes(pins)
    progress('metadata')
    def rows(key,columns):
        return iter_projected_csv(linked.INPUTS[key],columns,max_rows=200000000 if key=='labs' else 2000000)
    metadata=source.predictor_metadata(rows('patients',source.PATIENT_COLUMNS),rows('admissions',source.ADMISSION_COLUMNS))
    accumulator=coverage.CoverageAccumulator(metadata)
    def labs():
        for n,row in enumerate(rows('labs',source.LAB_COLUMNS),1):
            if n%1000000==0:progress('lab_scan',n)
            yield row
    progress('lab_scan',0)
    stream=source.available_events(labs(),metadata,authenticate_code_binding('mimic',mapping))
    try:
        for item in stream:accumulator.add(item)
    finally:stream.close()
    progress('gate_replay');require(accumulator.private_gate_pairs()==expected_pairs)
    aggregate=accumulator.finish();coverage.validate_result(aggregate)
    progress('post_authentication');source_hashes(pins)
    require(prior()[0]==pins and code_hashes()==protocol['code_sha256']
            and sha(DESIGN)==protocol['design_sha256'] and sha(out/'protocol.json')==pin)
    require(not (out/'failure.json').exists())
    write_json(out/'aggregate.json',aggregate)
    progress('completed')
    write_json(out/'completed.json',{'status':'authenticated_completed','protocol_sha256':pin,
        'aggregate_sha256':sha(out/'aggregate.json'),'elapsed_seconds':time.monotonic()-start,
        'old_gate_episode_membership_reproduced_exactly':True,'full_source_scan_completed':True,
        'patient_level_output_emitted':False,'scientific_goal_achieved':False})


def authenticate(attempt):
    import bran_mimic_upstream_coverage_c2 as coverage
    out=paths(attempt)
    require(out.is_dir() and not out.is_symlink() and not (out/'failure.json').exists())
    for name in ('protocol.json','aggregate.json','completed.json'):require((out/name).is_file() and not (out/name).is_symlink())
    p=json.loads((out/'protocol.json').read_text());a=json.loads((out/'aggregate.json').read_text())
    done=json.loads((out/'completed.json').read_text())
    validate_terminal(done,sha(out/'protocol.json'),sha(out/'aggregate.json'))
    validate_protocol(p)
    source_hashes(p['sources']);coverage.validate_result(a)
    return out,a,{'status':'authenticated','protocol_sha256':done['protocol_sha256'],
                 'aggregate_sha256':done['aggregate_sha256'],'patient_level_output_emitted':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,required=True);p.add_argument('--audit-only',action='store_true')
    args=p.parse_args();state={'out':None,'phase':'authentication'};ok=False
    with quiet():
        try:
            with LOCK.open('a+b') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if args.audit_only:receipt=authenticate(args.attempt)[2]
                else:run(args.attempt,state)
                ok=True
        except Exception as exc:
            if state['out'] is not None and not (state['out']/'completed.json').exists():
                import run_bran_mimic_landmark_linkage_v1 as linked
                write_json(state['out']/'failure.json',{'status':'technical_failure','phase':state['phase'],
                    'safe_exception_chain':safe_trace(exc,ROOT,set(linked.CODE)|set(OWN_CODE)),
                    'patient_level_output_emitted':False})
    result=receipt if ok and args.audit_only else {'status':'completed' if ok else 'not_completed',
        'phase':state['phase'],'patient_level_output_emitted':False}
    print(json.dumps(result,sort_keys=True));return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
