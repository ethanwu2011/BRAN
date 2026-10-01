"""Local-only manifest-average CGM target adapter; never serialize returned arrays."""
from dataclasses import dataclass
import csv
import math
import numpy as np
from bran_independent_coverage_io_v1 import _bounded_text_lines, _canonical_id, MAX_SOURCE_ROWS, MAX_HEADER_COLUMNS

COLUMNS=('person_id','average_glucose_level_mg_dl','glucose_level_record_count','glucose_sensor_sampling_duration_days')
TARGET_CONTRACT={'target':'manifest_average_cgm_glucose','unit':'mg/dL','source':'wearable_blood_glucose/manifest.tsv',
    'required_columns':list(COLUMNS),'validity':'positive_finite_mean_and_duration_positive_integer_count',
    'standardized_window_claimed':False,'clinical_wear_quality_claimed':False,'raw_time_series_loaded':False}

class CGMTargetInputError(ValueError): pass

@dataclass(repr=False)
class LocalTargets:
    ids:tuple[str,...]
    values:np.ndarray
    observed:np.ndarray

def _number(value):
    if not isinstance(value,str): return None
    try: number=float(value.strip())
    except (ValueError,OverflowError): return None
    return number if math.isfinite(number) else None

def parse_manifest(handle):
    if isinstance(handle,(str,bytes)) or not hasattr(handle,'__iter__'): raise CGMTargetInputError('invalid_handle')
    ids=[]; values=[]; observed=[]; seen=set()
    try:
        reader=csv.DictReader(_bounded_text_lines(handle),delimiter='\t',strict=True)
        header=reader.fieldnames
        if not header or len(header)>MAX_HEADER_COLUMNS or any(header.count(c)!=1 for c in COLUMNS): raise CGMTargetInputError('invalid_columns')
        for row_number,row in enumerate(reader,1):
            if row_number>MAX_SOURCE_ROWS or None in row: raise CGMTargetInputError('invalid_shape')
            identifier=_canonical_id(row.get('person_id'))
            if identifier in seen: raise CGMTargetInputError('duplicate_id')
            seen.add(identifier)
            mean,count,duration=(_number(row.get(c)) for c in COLUMNS[1:])
            valid=(mean is not None and mean>0 and count is not None and count>0 and count.is_integer() and duration is not None and duration>0)
            ids.append(identifier); values.append(mean if valid else 0.0); observed.append(bool(valid))
    except (ValueError,TypeError,UnicodeError,csv.Error) as error:
        raise CGMTargetInputError('target_manifest_invalid_contents_withheld') from error
    return LocalTargets(tuple(ids),np.asarray(values,float),np.asarray(observed,bool))

def align(targets,canonical_ids):
    ids=tuple(canonical_ids)
    if not isinstance(targets,LocalTargets) or not ids or len(ids)!=len(set(ids)) or any(type(i)is not str or not i for i in ids): raise CGMTargetInputError('invalid_identity_contract')
    if targets.values.shape!=(len(targets.ids),) or targets.observed.shape!=(len(targets.ids),) or targets.observed.dtype!=np.dtype(bool) or not np.isfinite(targets.values).all() or len(set(targets.ids))!=len(targets.ids): raise CGMTargetInputError('invalid_target_arrays')
    if any(type(i)is not str or not i for i in targets.ids) or np.any(targets.values[targets.observed]<=0): raise CGMTargetInputError('invalid_observed_targets')
    lookup={identifier:index for index,identifier in enumerate(targets.ids)}
    y=np.zeros(len(ids)); observed=np.zeros(len(ids),bool)
    for index,identifier in enumerate(ids):
        position=lookup.get(identifier)
        if position is not None and targets.observed[position]: y[index]=targets.values[position]; observed[index]=True
    return y,observed
