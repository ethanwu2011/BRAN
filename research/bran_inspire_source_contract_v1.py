"""Fixed source-local INSPIRE selection. No I/O, model, print or data admission.

Call only inside an FD-quiet local runner on authenticated source columns.
Synthetic fixtures may exercise the same iterator interfaces. The earlier
generic preop helper remains unchanged: this task resolves hospitalization-
bound death/discharge equality prospectively and enforces admission chronology.
"""
from dataclasses import dataclass
import hashlib
import math
import re
import numpy as np

ERROR='bran_inspire_source_contract_v1_failed'
SOURCE_FIELDS=('albumin','alp','alt','ast','bun','chloride','creatinine','crp','glucose',
    'hb','hba1c','hct','platelet','potassium','sodium','total_bilirubin','total_protein','wbc')
SOURCE_UNITS=('g/dL','IU/L','IU/L','IU/L','mg/dL','mmol/L','mg/dL','mg/dL','mg/dL',
    'g/dL','%','%','/nL','mmol/L','mmol/L','mg/dL','g/dL','/nL')
LOOKBACK_MINUTES=30*24*60
ANCHOR_GUARD_MINUTES=5
ROLE_SALT=b'BRAN_INSPIRE_V1_98621:'
OP_COLUMNS=('op_id','subject_id','hadm_id','age','sex','asa','emop','department',
    'orin_time','opstart_time','admission_time','discharge_time','anstart_time','inhosp_death_time')
LAB_COLUMNS=('subject_id','chart_time','item_name','value')
ZERO_ALLOWED=frozenset(('alp','alt','ast','crp'))
NAN=float('nan')


def _require(condition):
    if not condition:
        raise ValueError(ERROR)


def _number(value):
    # Source format failures remain missing locally; raw payload is never
    # interpolated into an error. A later invalid assay erases an earlier value.
    if isinstance(value,(bool,np.bool_)):
        return NAN
    try:
        number=float(value)
        return number if math.isfinite(number) else NAN
    except (ValueError,TypeError,OverflowError):
        return NAN


def _key(value):
    _require(type(value) is str and re.fullmatch(r'[0-9]{1,24}',value) is not None)
    return value


def _absent(value):
    return value is None or (type(value) is str and value.strip().lower() in ('','nan','na','null'))


def role_for_person(key):
    digest=hashlib.sha256(ROLE_SALT+_key(key).encode('ascii')).digest()
    value=int.from_bytes(digest[:8],'big'); total=1<<64
    return 0 if value < total*3//5 else (1 if value < total*4//5 else 2)


def _signature(row):
    return tuple(None if not math.isfinite(v:=_number(row[k])) else v
                 for k in ('admission_time','discharge_time','inhosp_death_time'))


def _readonly(value,dtype):
    out=np.array(value,dtype=dtype,copy=True);out.setflags(write=False);return out


@dataclass(frozen=True,slots=True,repr=False)
class PrivateInspireFrame:
    person: np.ndarray
    operation: np.ndarray
    role: np.ndarray
    anchor_minutes: np.ndarray
    chronology_eligible: np.ndarray
    outcome: np.ndarray
    age_lower: np.ndarray
    age_upper: np.ndarray
    age_kind: np.ndarray
    context: np.ndarray
    values: np.ndarray
    observed: np.ndarray

    def __repr__(self):
        return '<PrivateInspireFrame>'

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self,protocol):
        raise TypeError(ERROR)


def select_operations(rows):
    """Select first chronology before labels/predictors; outputs remain private."""
    try:
        first={}; seen_ops=set(); unknown_chronology=set(); hadm_owner={}
        admission_signatures={}; contradictory_admissions=set()
        for row in rows:
            _require(type(row) is dict and set(OP_COLUMNS).issubset(row))
            person=_key(row['subject_id']);op=_key(row['op_id']);hadm=_key(row['hadm_id'])
            _require(op not in seen_ops);seen_ops.add(op)
            _require(hadm not in hadm_owner or hadm_owner[hadm]==person);hadm_owner[hadm]=person
            signature=_signature(row)
            if hadm in admission_signatures and admission_signatures[hadm]!=signature:
                contradictory_admissions.add(hadm)
            admission_signatures[hadm]=signature
            start=_number(row['opstart_time'])
            if not math.isfinite(start):unknown_chronology.add(person)
            order=(start if math.isfinite(start) else math.inf,int(op))
            if person not in first or order < first[person][0]:
                # Copy only the fixed source contract columns.
                first[person]=(order,{k:row[k] for k in OP_COLUMNS})
        _require(bool(first))
        people=sorted(first);n=len(people)
        anchors=np.full(n,NAN);chronology=np.zeros(n,bool);outcome=np.full(n,-1,np.int8)
        lower=np.full(n,NAN);upper=np.full(n,NAN);kind=np.full(n,3,np.int8)
        contexts=[];operations=[]
        for i,person in enumerate(people):
            row=first[person][1];operations.append(row['op_id'])
            age=_number(row['age'])
            if math.isfinite(age) and age>=0 and age%5==0:
                lower[i]=max(0.,age-2.5);upper[i]=age+2.4;kind[i]=1
            sex=row['sex'] if row['sex'] in ('M','F') else 'unknown'
            asa=_number(row['asa']); emergency=_number(row['emop'])
            asa_text=str(int(asa)) if math.isfinite(asa) and asa in (1,2,3,4,5,6) else 'unknown'
            emergency_text=str(int(emergency)) if emergency in (0,1) else 'unknown'
            department=row['department']
            _require(type(department) is str and len(department)<=64)
            department=department.strip() or 'unknown'
            contexts.append((sex,asa_text,emergency_text,department))
            adm=_number(row['admission_time']);anchor=_number(row['orin_time'])
            start=_number(row['opstart_time']);anstart=_number(row['anstart_time'])
            valid=(person not in unknown_chronology and row['hadm_id'] not in contradictory_admissions
                and all(math.isfinite(x) for x in (adm,anchor,start)) and adm<=anchor<=start
                and (_absent(row['anstart_time']) or (math.isfinite(anstart) and anchor<=anstart<=start))
                and asa_text!='6')
            if not valid:continue
            chronology[i]=True;anchors[i]=anchor
            discharge=_number(row['discharge_time']);death=_number(row['inhosp_death_time'])
            if not math.isfinite(discharge) or discharge<=start:continue
            if math.isfinite(death):
                if start<death<=discharge:outcome[i]=1
            else:
                # Only documented source missing tokens imply absent death.
                # Malformed nonnumeric payload is unknown, not a negative.
                value=row['inhosp_death_time']
                if _absent(value):
                    outcome[i]=0
        return PrivateInspireFrame(_readonly(people,'U24'),_readonly(operations,'U24'),
            _readonly([role_for_person(p) for p in people],np.int8),_readonly(anchors,np.float64),
            _readonly(chronology,bool),_readonly(outcome,np.int8),_readonly(lower,np.float64),
            _readonly(upper,np.float64),_readonly(kind,np.int8),_readonly(contexts,'U64'),
            _readonly(np.full((n,len(SOURCE_FIELDS)),NAN),np.float64),_readonly(np.zeros((n,len(SOURCE_FIELDS))),bool))
    except (MemoryError,KeyboardInterrupt,SystemExit):raise
    except Exception:raise ValueError(ERROR) from None


def fill_labs(frame,rows):
    """One-pass latest pre-anchor observations with same-time conflict erasure."""
    try:
        _require(type(frame) is PrivateInspireFrame)
        person_index={p:i for i,p in enumerate(frame.person)}
        fields={f:j for j,f in enumerate(SOURCE_FIELDS)}
        n=len(frame.person);m=len(SOURCE_FIELDS)
        time=np.full((n,m),-np.inf);value=np.full((n,m),NAN);conflict=np.zeros((n,m),bool)
        for row in rows:
            _require(type(row) is dict and set(LAB_COLUMNS).issubset(row))
            field=row['item_name']
            if field not in fields:continue
            person=row['subject_id'];i=person_index.get(person)
            if i is None or not frame.chronology_eligible[i]:continue
            t=_number(row['chart_time']);anchor=frame.anchor_minutes[i]
            if not math.isfinite(t) or not anchor-LOOKBACK_MINUTES<=t<=anchor-ANCHOR_GUARD_MINUTES:continue
            j=fields[field];v=_number(row['value'])
            if math.isfinite(v) and (v<0 or (v==0 and field not in ZERO_ALLOWED)):v=NAN
            if t>time[i,j]:
                time[i,j]=t;value[i,j]=v;conflict[i,j]=False
            elif t==time[i,j] and not (v==value[i,j] or (math.isnan(v) and math.isnan(value[i,j]))):
                conflict[i,j]=True
        observed=np.isfinite(value)&~conflict
        value[~observed]=NAN
        return PrivateInspireFrame(frame.person,frame.operation,frame.role,frame.anchor_minutes,
            frame.chronology_eligible,frame.outcome,frame.age_lower,frame.age_upper,frame.age_kind,
            frame.context,_readonly(value,np.float64),_readonly(observed,bool))
    except (MemoryError,KeyboardInterrupt,SystemExit):raise
    except Exception:raise ValueError(ERROR) from None


def safe_support(frame):
    """Closed admission facts; suppress the entire detailed flow if any cell is small."""
    _require(type(frame) is PrivateInspireFrame)
    eligible=frame.chronology_eligible & (frame.outcome>=0) & frame.observed.any(axis=1)
    cells=[];passes=[]
    for role,min_n in ((0,200),(1,100),(2,100)):
        mask=frame.role==role
        # Disjoint flow categories prevent complement disclosure.
        counts=[int((mask&~frame.chronology_eligible).sum()),
            int((mask&frame.chronology_eligible&(frame.outcome<0)).sum()),
            int((mask&frame.chronology_eligible&(frame.outcome>=0)&~frame.observed.any(axis=1)).sum()),
            int((mask&eligible&(frame.outcome==0)).sum()),int((mask&eligible&(frame.outcome==1)).sum())]
        cells.append(counts)
        passes.append(counts[-1]>=20 and counts[-2]>=20 and sum(counts[-2:])>=min_n)
    report={'schema':'bran-inspire-source-support-v1','role_support_met':all(passes),
        'source_task':'external_adaptation_not_fitted','performance_evaluated':False,
        'patient_level_output_emitted':False,'clinical_use':False}
    if all(x>=20 for row in cells for x in row):
        report['flow']={'status':'released_lower_bounds','columns':['chronology_ineligible','unknown_outcome',
            'empty_physiology','eligible_surviving_discharge','eligible_hospital_death'],
            'roles':['fit','calibration','test'],'lower_bounds':[[x//20*20 for x in row] for row in cells]}
    else:
        report['flow']={'status':'suppressed_complement_support'}
    return report
