"""Pure original-code membership; no source I/O, fitting or patient output."""
import math
import numpy as np
from bran_mimic_prior_disease_v1 import canonical_icd
from bran_eicu_discovery_bridge_v1 import parse_numeric_measurement

ERROR='expanded_membership_s7_failed'
FAMILIES=('recorded_hypertension','recorded_obesity','recorded_ischemic_heart_disease',
    'recorded_atrial_fibrillation_flutter','recorded_copd','recorded_asthma',
    'recorded_rheumatoid_arthritis','recorded_osteoarthritis')

def require(ok):
    if not ok:raise ValueError(ERROR) from None

def family_for_code(version,code):
    require(canonical_icd(version,code)==(version,code))
    if version=='9':
        if code[:3] in ('401','402','403','404','405'):return FAMILIES[0]
        if code in ('27800','27801','27803'):return FAMILIES[1]
        if code[:3] in ('410','411','412','413','414'):return FAMILIES[2]
        if code in ('42731','42732'):return FAMILIES[3]
        if code[:3] in ('491','492','496'):return FAMILIES[4]
        if code.startswith('493') and code!='49381':return FAMILIES[5]
        if code in ('7140','7141','7142','71481'):return FAMILIES[6]
        if code.startswith('715'):return FAMILIES[7]
    if version=='10':
        if code[:3] in ('I10','I11','I12','I13','I15'):return FAMILIES[0]
        if code.startswith('E66') and code not in ('E66','E663'):return FAMILIES[1]
        if code[:3] in ('I20','I21','I22','I23','I24','I25'):return FAMILIES[2]
        if code.startswith('I48'):return FAMILIES[3]
        if code[:3] in ('J41','J42','J43','J44'):return FAMILIES[4]
        if code.startswith('J45') and code!='J45990':return FAMILIES[5]
        if code.startswith(('M05','M060','M068','M069')):return FAMILIES[6]
        if code[:3] in ('M15','M16','M17','M18','M19'):return FAMILIES[7]
    return None

def title_agrees(family,title):
    require(type(title) is str and title.strip() and not any(ord(c)<32 for c in title))
    text=title.casefold()
    words=(('hypertens',),('obesity',),('myocardial','infarction','coronary','angina','ischemic','atherosclero','aneurysm of heart','dressler'),
        ('atrial fibrillation','atrial flutter'),('bronchitis','emphysem','chronic airway obstruction','chronic obstructive'),
        ('asthma',),('rheumatoid','felty'),('osteoarth','(osteo)arthritis',"heberden's nodes","bouchard's nodes",'secondary multiple arthritis'))
    return any(word in text for word in words[FAMILIES.index(family)])

def bind_dictionary(rows):
    result={};seen=set()
    for row in rows:
        require(set(row)=={'icd_version','icd_code','long_title'})
        pair=canonical_icd(row['icd_version'],row['icd_code'])
        require(pair is not None and pair not in seen);seen.add(pair)
        family=family_for_code(*pair)
        if family is not None:
            require(title_agrees(family,row['long_title']));result[pair]=family
    require(all(any(v==version and f==family for (v,_),f in result.items())
        for version in ('9','10') for family in FAMILIES))
    return result

def _index(cohort):
    people=cohort['person'];episodes=cohort['episode']
    require(type(people) is np.ndarray and people.ndim==1 and people.dtype.kind=='U'
        and type(episodes) is np.ndarray and episodes.shape==people.shape and episodes.dtype.kind=='U'
        and len(set(people))==len(people) and len(set(episodes))==len(episodes))
    return {str(e):(i,str(p)) for i,(p,e) in enumerate(zip(people,episodes))}

def mimic_membership(cohort,rows,mapping,families=FAMILIES):
    index=_index(cohort);result=np.zeros((len(index),len(families)),bool)
    columns={f:i for i,f in enumerate(families)}
    require(all(f in columns for f in mapping.values()))
    for row in rows:
        require(set(row)=={'subject_id','hadm_id','icd_version','icd_code'})
        target=index.get(row['hadm_id'])
        if target is None:continue
        i,person=target;require(row['subject_id']==person)
        pair=canonical_icd(row['icd_version'],row['icd_code'])
        if pair in mapping:result[i,columns[mapping[pair]]]=True
    return result

def eicu_membership(cohort,episodes,rows,mapping,families=FAMILIES,*,icd9_only=True):
    index=_index(cohort);require(set(episodes)==set(index));columns={f:i for i,f in enumerate(families)}
    require(all(f in columns for f in mapping.values()))
    result=np.zeros((len(index),len(families)),bool)
    for key,(i,person) in index.items():
        e=episodes[key]
        require(e.episode==key and e.person==person and e.hospital_stay==cohort['hospital_stay'][i]
            and e.hospital==cohort['hospital'][i] and e.outcome==cohort['outcome'][i])
    for row in rows:
        require(set(row)=={'patientunitstayid','diagnosisoffset','icd9code'})
        target=index.get(row['patientunitstayid'])
        if target is None:continue
        i,_=target;e=episodes[row['patientunitstayid']]
        offset=parse_numeric_measurement(row['diagnosisoffset'])
        if offset is None or not math.isfinite(e.unit_end) or not 0<=offset<=e.unit_end:continue
        payload=row['icd9code'];require(type(payload) is str)
        if not payload or len(payload)>512:continue
        tokens=payload.split(',')
        if len(tokens)>16:continue
        for token in tokens:
            for version in (('9',) if icd9_only else ('9','10')):
                pair=canonical_icd(version,token.strip(' '))
                if pair in mapping:result[i,columns[mapping[pair]]]=True
    return result

def support(membership,roles,sources):
    n=len(roles)
    require(membership.shape==(n,len(FAMILIES)) and membership.dtype==bool
        and roles.shape==sources.shape==(n,) and np.isin(roles,(0,1,2)).all()
        and np.isin(sources,(0,1)).all())
    output={}
    for j,family in enumerate(FAMILIES):
        cells=[[int(np.sum(membership[:,j]&(sources==s)&(roles==r))) for r in range(3)] for s in range(2)]
        complements=[[int(np.sum(~membership[:,j]&(sources==s)&(roles==r))) for r in range(3)] for s in range(2)]
        released=min(v for group in (cells,complements) for row in group for v in row)>=20
        output[family]={'status':'released' if released else 'suppressed',
            'source_role_lower_bounds_20':[[v//20*20 for v in row] for row in cells] if released else None}
    return output
