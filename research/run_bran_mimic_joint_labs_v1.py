"""Two-pass original MIMIC CBC+chemistry cache; no training or efficacy claim."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_clinical_chemistry_semantics_v1 import CHEMISTRY_FIELDS, CANONICAL_UNITS, bind_mimic_chemistry_dictionary
from bran_clinical_source_reader_v1 import iter_projected_csv, validate_event_link, _numeric_key
from bran_cbc_event_adapter_v1 import _naive_iso_datetime
from bran_joint_lab_online_v1 import authenticate_code_binding, convert_joint_bound_event, CBCAnchorAccumulator, JointLabSecondPass
from bran_nhanes_joint_labs_v1 import JointRecord
from bran_joint_lab_cache_v1 import pack_records, safe_payload, validate_aggregate, valid_sha
from run_bran_source_linkage_audit_v1 import sha, exclusive_json
import run_bran_observed_cbc_pool_v1 as old

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'BRAN_MIMIC_JOINT_LABS_PROTOCOL_V1.json'
PUBLIC = ROOT / 'BRAN_MIMIC_JOINT_LABS_V1'
PRIVATE = ROOT / 'private_artifacts' / 'bran_mimic_joint_labs_v1'
SALT = old.PRIVATE / 'split_salt.bin'
PREVIOUS_SHA = '397292c03fa89e496ec3295b19ca9200b6d93e3d80717251d07f0003748880f6'
CHEMISTRY_BINDING = ROOT / 'BRAN_CHEMISTRY_SOURCE_BINDINGS_DRAFT_V1.json'
CHEMISTRY_SHA = '29e6a59845288c799db334ed08137f242e932e1c1668f836addca17964d4f371'
INPUTS = {k: old.INPUTS[k] for k in ('mimic_patients', 'mimic_admissions', 'mimic_labs')}
INPUTS['mimic_dictionary'] = '/Users/ethanwu/mimiciv-3.1/hosp/d_labitems.csv.gz'
CODE = tuple(dict.fromkeys(old.CODE + (
    'run_bran_mimic_joint_labs_v1.py', 'bran_clinical_chemistry_semantics_v1.py', 'bran_cbc_chemistry_snapshot_v1.py',
    'bran_joint_lab_online_v1.py', 'bran_joint_lab_cache_v1.py', 'bran_nhanes_joint_labs_v1.py',
    'bran_nhanes_chemistry_binding_v1.py', 'bran_nhanes_chemistry_observation_v2.py', 'bran_nhanes_header_extension_v1.py',
    'bran_six_source_schema_preflight_v1.py', 'test_bran_cbc_chemistry_snapshot_v1.py', 'test_bran_joint_lab_online_v1.py',
    'test_bran_joint_lab_cache_v1.py', 'test_run_bran_mimic_joint_labs_v1.py',
)))
POLICY = {
    'max_rows_per_metadata_table': 2000000, 'max_rows_per_lab_pass': 200000000,
    'minimum_observed_CBC_fields': 2,
    'source': 'original MIMIC-IV3.1 Blood/Hematology CBC and exact Blood/Chemistry laboratory codes',
    'passes': 'first full lab pass fixes earliest valid CBC in admission minutes[0,1440]; second full pass selects all21fields within[anchor,min(anchor+60,1440)]',
    'ties': 'earliest per-field; identical earliest ties collapse, conflicting earliest ties remain missing, no later replacement',
    'join': 'source subject_id and hadm_id must agree with authenticated patients/admissions; no cache-row joins',
    'truth': 'original measured numeric observations only, CBC>0 chemistry>=0; exact compatible units, no magnitude-based units or upper clipping',
    'age': 'source anchor_age/year at admission; preserve lower/upper/kind including censoring; adult lower bound>=18',
    'split': 'reuse original source-person HMAC split salt80/10/10; repeated admissions grouped and inverse-person-episode weighting stored',
    'clinical_limit': 'admission-hour context is not proven same draw, pretreatment or outpatient physiology; no diagnostic labels derived here',
    'chemistry_only': 'not admitted without a valid CBC anchor and at least2observed CBC fields',
    'retinal_data_read': False, 'training_permitted': False, 'patient_level_output_permitted': False,
    'count_release': 'lower-bound multiples of20; smaller counts suppressed; not exact totals',
}
PHASES = ('source_authentication', 'metadata', 'CBC_anchor_scan', 'joint_context_scan', 'observation_pack', 'private_cache_write', 'post_scan_authentication', 'aggregate_commit', 'completed')


def runtime():
    return {'python': sys.version.split()[0], 'numpy': np.__version__}


def mapping():
    if sha(CHEMISTRY_BINDING) != CHEMISTRY_SHA: raise ValueError('chemistry binding changed')
    draft = json.loads(CHEMISTRY_BINDING.read_text())
    if sha(INPUTS['mimic_dictionary']) != draft['dictionary']['sha256']: raise ValueError('dictionary changed')
    bound = bind_mimic_chemistry_dictionary(iter_projected_csv(INPUTS['mimic_dictionary'], ('itemid','label','fluid','category'), max_rows=POLICY['max_rows_per_metadata_table']))
    expected = {v['itemid']: f for f,v in draft['approved_fields'].items()}
    if bound['approved_code_to_field'] != expected or set(expected.values()) != set(CHEMISTRY_FIELDS): raise ValueError('chemistry identity mismatch')
    for field, entry in draft['approved_fields'].items():
        if entry['canonical_unit'] != CANONICAL_UNITS[field] or entry['multiplier'] != 1.: raise ValueError('chemistry unit mismatch')
    cbc = old.dictionary_mapping('mimic')
    if set(cbc.values()) != set(CBC_FIELDS) or set(cbc) & set(expected): raise ValueError('CBC identity mismatch')
    return {**cbc, **expected}


def prepare():
    if sha(old.PROTOCOL) != PREVIOUS_SHA: raise ValueError('previous protocol changed')
    prior = json.loads(old.PROTOCOL.read_text()); old.validate_protocol(prior)
    for k in INPUTS:
        if k != 'mimic_dictionary' and sha(INPUTS[k]) != prior['source_files'][k]['sha256']: raise ValueError('previous source changed')
    mapping()
    return {'schema': 'bran-mimic-joint-labs-protocol-v1', 'policy': POLICY, 'previous_protocol_sha256': PREVIOUS_SHA,
        'chemistry_binding_sha256': CHEMISTRY_SHA, 'code_sha256': {n: sha(ROOT/n) for n in CODE},
        'source_files': {k: {'path':v, 'sha256':sha(v)} for k,v in INPUTS.items()},
        'split_salt_sha256': sha(SALT), 'runtime': runtime()}


def validate_protocol(p):
    if set(p) != {'schema','policy','previous_protocol_sha256','chemistry_binding_sha256','code_sha256','source_files','split_salt_sha256','runtime'} or p['schema'] != 'bran-mimic-joint-labs-protocol-v1' or p['policy'] != POLICY or p['runtime'] != runtime(): raise ValueError('protocol mismatch')
    if p['previous_protocol_sha256'] != PREVIOUS_SHA or sha(old.PROTOCOL) != PREVIOUS_SHA or p['chemistry_binding_sha256'] != CHEMISTRY_SHA or sha(CHEMISTRY_BINDING) != CHEMISTRY_SHA: raise ValueError('prior binding mismatch')
    old.validate_protocol(json.loads(old.PROTOCOL.read_text()))
    if set(p['code_sha256']) != set(CODE) or set(p['source_files']) != set(INPUTS): raise ValueError('binding keys mismatch')
    for name,h in p['code_sha256'].items():
        if sha(ROOT/name) != h: raise ValueError('code changed')
    for k,spec in p['source_files'].items():
        if set(spec) != {'path','sha256'} or spec['path'] != INPUTS[k] or not valid_sha(spec['sha256']): raise ValueError('source binding mismatch')
    if p['split_salt_sha256'] != sha(SALT): raise ValueError('split salt changed')


def progress(phase):
    if phase not in PHASES: raise ValueError('invalid phase')
    payload = {'schema':'bran-mimic-joint-labs-progress-v1', 'phase':phase, 'pid':os.getpid(), 'patient_level_output_emitted':False, 'training_started':False}
    tmp = PUBLIC/'progress.tmp'
    tmp.write_text(json.dumps(payload, sort_keys=True))
    os.replace(tmp, PUBLIC/'progress.json')


def events(rows, links, admissions, binding, *, anchors=None):
    """Source-local projection; callers pass original bound lab rows only."""
    for row in rows:
        if row['itemid'] not in binding.code_to_field: continue
        person = validate_event_link('mimic', row, links)
        if person is None: continue
        episode = _numeric_key(row['hadm_id'])
        if anchors is not None and episode not in anchors.episode_to_person: continue
        adm = admissions.get(episode)
        stamp = _naive_iso_datetime(row['charttime'])
        if adm is None or stamp is None: continue
        offset = (stamp-adm).total_seconds()/60.
        if not 0 <= offset <= 1440: continue
        if anchors is not None:
            anchor = anchors.anchor_minutes[episode]
            if not math.isfinite(anchor) or not anchor <= offset <= min(anchor+60,1440): continue
        item = convert_joint_bound_event('mimic', episode, person, row['itemid'], row['valuenum'], row['valueuom'], offset, 1, binding)
        if item is not None: yield item


def records():
    progress('metadata')
    links, raw_admissions, ages = old.metadata('mimic')
    admissions = {key: _naive_iso_datetime(value) for key,value in raw_admissions.items()}
    del raw_admissions
    maps = mapping()
    cbc = authenticate_code_binding('mimic', {code:field for code,field in maps.items() if field in CBC_FIELDS})
    all_labs = authenticate_code_binding('mimic', maps)
    columns = ('subject_id','hadm_id','itemid','valuenum','valueuom','charttime')
    def rows():
        return iter_projected_csv(INPUTS['mimic_labs'], columns, max_rows=POLICY['max_rows_per_lab_pass'])
    progress('CBC_anchor_scan')
    first = CBCAnchorAccumulator('mimic')
    for item in events(rows(), links, admissions, cbc): first.add(item)
    anchors = first.finalize()
    del first
    progress('joint_context_scan')
    second = JointLabSecondPass(anchors)
    for item in events(rows(), links, admissions, all_labs, anchors=anchors): second.add(item)
    progress('observation_pack')
    for episode, person, snap in second.iterate_snapshots():
        if snap.observed[:9].sum() >= 2:
            yield JointRecord(person, snap.values, snap.observed, np.zeros(21, np.uint8), ages[episode], '')


def run(p):
    validate_protocol(p)
    PUBLIC.mkdir()
    phase = 'source_authentication'
    try:
        progress(phase)
        for k,path in INPUTS.items():
            if sha(path) != p['source_files'][k]['sha256']: raise ValueError('input changed')
        PRIVATE.mkdir(parents=True, mode=0o700); os.chmod(PRIVATE,0o700)
        phase = 'observation_pack'
        arrays, summary = pack_records('mimic', records(), SALT.read_bytes())
        phase = 'private_cache_write'; progress(phase)
        cache = PRIVATE/'observations.npz'
        fd = os.open(cache, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd,'wb') as handle: np.savez_compressed(handle, **arrays)
        phase = 'post_scan_authentication'; progress(phase)
        for k,path in INPUTS.items():
            if sha(path) != p['source_files'][k]['sha256']: raise ValueError('input changed during scan')
        validate_protocol(p)
        phase = 'aggregate_commit'; progress(phase)
        payload = safe_payload('mimic', summary, sha(cache)); validate_aggregate(payload)
        exclusive_json(PUBLIC/'aggregate.json', payload)
        exclusive_json(PUBLIC/'manifest.json', {'protocol_sha256':sha(PROTOCOL), 'aggregate_sha256':sha(PUBLIC/'aggregate.json'),
            'split_salt_sha256':sha(SALT), 'patient_level_output_emitted':False, 'training_started':False})
        progress('completed')
        return True
    except Exception:
        exclusive_json(PUBLIC/'failure.json', {'status':'joint_lab_observation_scan_failed', 'phase':phase, 'patient_level_output_emitted':False, 'training_started':False})
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-protocol', action='store_true')
    parser.add_argument('--run', action='store_true')
    args=parser.parse_args()
    if args.prepare_protocol == args.run: parser.error('choose exactly one operation')
    ok=False
    with _quiet():
        try:
            if args.prepare_protocol: exclusive_json(PROTOCOL,prepare()); ok=True
            else: ok=run(json.loads(PROTOCOL.read_text()))
        except Exception: pass
    print(json.dumps({'status':('protocol_prepared' if args.prepare_protocol else 'joint_labs_completed') if ok else 'joint_labs_failed', 'patient_level_output_emitted':False, 'training_started':False}))
    return 0 if ok else 1


if __name__ == '__main__': raise SystemExit(main())
