"""Private 21-field observation cache and deeply closed aggregate receipt."""
import math
import numpy as np

from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_clinical_chemistry_semantics_v1 import CHEMISTRY_FIELDS
from bran_clinical_snapshot_v1 import assign_person_split

FIELDS = CBC_FIELDS + CHEMISTRY_FIELDS
SOURCES = ('nhanes', 'mimic')
AGE_KINDS = ('missing_or_invalid', 'source_age_unresolved', 'reported_year', 'year_derived', 'topcoded', 'rounded', 'rounded_topcoded')
COUNT_KEYS = ('snapshots', 'source_local_people', 'adult_qualified_snapshots', 'CBC_with_chemistry_snapshots', 'adult_CBC_with_chemistry_snapshots')
SPLITS = {'train': 0, 'validation': 1, 'test': 2}


def coarse_count(n):
    if type(n) is not int or n < 0: raise ValueError('invalid count')
    return n // 20 * 20 if n >= 20 else None


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def pack_records(source, records, salt):
    if source not in SOURCES or not isinstance(salt, bytes) or len(salt) != 32:
        raise ValueError('invalid cache configuration')
    records = list(records)
    if not records: raise ValueError('no usable observations')
    people = sorted({r.person for r in records})
    indices = {p: i for i, p in enumerate(people)}
    groups = np.array([indices[r.person] for r in records], np.int64)
    if source == 'nhanes' and len(people) != len(records):
        raise ValueError('duplicate cross-cycle NHANES identity')
    values = np.stack([r.values for r in records]).astype(np.float64)
    masks = np.stack([r.observed for r in records])
    calibration = np.stack([r.calibration for r in records])
    shape = (len(records), len(FIELDS))
    if values.shape != shape or masks.shape != shape or masks.dtype != np.bool_ or calibration.shape != shape:
        raise ValueError('invalid cache arrays')
    if calibration.dtype.kind not in 'iu' or np.any((calibration < 0) | (calibration > 2)):
        raise ValueError('invalid calibration flags')
    if np.any(masks & (~np.isfinite(values) | (values < 0))) or np.any(masks[:, :9] & (values[:, :9] <= 0)) or np.any(masks[:, :9].sum(1) < 2):
        raise ValueError('invalid observed values')
    permitted = np.zeros(shape, bool)
    if source == 'nhanes': permitted[:, FIELDS.index('creatinine')] = True
    if np.any((calibration > 0) & (~permitted | ~masks)):
        raise ValueError('invalid calibration support')
    values[~masks] = np.nan
    ages = np.array([[r.age.reported_years, r.age.lower_years, r.age.upper_years] for r in records], np.float64)
    age_kind = np.array([AGE_KINDS.index(r.age.kind) for r in records], np.uint8)
    adult = np.array([math.isfinite(r.age.lower_years) and r.age.lower_years >= 18 and r.age.kind != 'source_age_unresolved' for r in records], bool)
    chem = masks[:, 9:].any(1)
    psplits = np.array([SPLITS[assign_person_split(source, p, salt=salt)] for p in people], np.uint8)
    counts = np.bincount(groups, minlength=len(people))
    cycle = np.array([('D', 'E').index(r.cycle) if source == 'nhanes' else -1 for r in records], np.int8)
    if source == 'nhanes':
        j = FIELDS.index('creatinine')
        if np.any(masks[:, j] & (calibration[:, j] != cycle + 1)):
            raise ValueError('calibration cycle mismatch')
    arrays = {'values': values, 'observed': masks, 'provenance': masks.astype(np.uint8), 'calibration_code': calibration.astype(np.uint8),
              'age_triplet': ages, 'age_kind': age_kind, 'adult_qualified': adult, 'person_group': groups,
              'split': psplits[groups], 'person_weight': 1. / counts[groups], 'cycle_group': cycle}
    summary = dict(zip(COUNT_KEYS, map(coarse_count, (len(records), len(people), int(adult.sum()), int(chem.sum()), int((adult & chem).sum())))))
    summary['field_observed_snapshots'] = {f: coarse_count(int(masks[:, j].sum())) for j, f in enumerate(FIELDS)}
    summary['adult_field_observed_snapshots'] = {f: coarse_count(int((masks[:, j] & adult).sum())) for j, f in enumerate(FIELDS)}
    summary['split_snapshots'] = {name: coarse_count(int((arrays['split'] == value).sum())) for name, value in SPLITS.items()}
    return arrays, summary


def safe_payload(source, summary, cache_hash):
    return {'schema': 'bran-joint-lab-cache-v1', 'source': source, 'status': 'observation_cache_materialized',
            'counts_lower_bounds_20': summary, 'private_cache_sha256': cache_hash,
            'patient_level_output_emitted': False, 'observations_processed_locally': True,
            'training_started': False, 'training_ready': False, 'model_benefit_established': False,
            'same_physical_draw_proven': False, 'cross_source_identity_resolved': False,
            'population_inference_permitted': False, 'clinical_diagnosis_permitted': False}


def validate_aggregate(p):
    expected = set(safe_payload('nhanes', {}, 'a' * 64))
    if not isinstance(p, dict) or set(p) != expected or p['schema'] != 'bran-joint-lab-cache-v1' or p['source'] not in SOURCES or p['status'] != 'observation_cache_materialized' or not valid_sha(p['private_cache_sha256']):
        raise ValueError('invalid aggregate schema')
    for key in ('patient_level_output_emitted', 'training_started', 'training_ready', 'model_benefit_established', 'same_physical_draw_proven', 'cross_source_identity_resolved', 'population_inference_permitted', 'clinical_diagnosis_permitted'):
        if p[key] is not False: raise ValueError('invalid privacy or claim flag')
    if p['observations_processed_locally'] is not True: raise ValueError('invalid local processing flag')
    c = p['counts_lower_bounds_20']
    if not isinstance(c, dict) or set(c) != set(COUNT_KEYS) | {'field_observed_snapshots', 'adult_field_observed_snapshots', 'split_snapshots'}:
        raise ValueError('invalid aggregate count keys')
    for key in ('field_observed_snapshots', 'adult_field_observed_snapshots'):
        if not isinstance(c[key], dict) or set(c[key]) != set(FIELDS): raise ValueError('invalid field count keys')
    if not isinstance(c['split_snapshots'], dict) or set(c['split_snapshots']) != set(SPLITS): raise ValueError('invalid split count keys')
    counts = [c[k] for k in COUNT_KEYS] + list(c['field_observed_snapshots'].values()) + list(c['adult_field_observed_snapshots'].values()) + list(c['split_snapshots'].values())
    if any(v is not None and (type(v) is not int or v < 20 or v % 20) for v in counts):
        raise ValueError('invalid coarsened count')
