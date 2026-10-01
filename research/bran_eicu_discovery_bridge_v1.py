"""Pure, private eICU landmark bridge; no I/O, fitting or result publication.

This new cohort permits one observed CBC field, not the old two-field gate.
Chemistry/vitals may be added only by independently qualified adapters. Disease
codes are candidate retrospective clinician-list memberships, not adjudication.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import hashlib
import math
from types import MappingProxyType

from bran_cbc_event_adapter_v1 import parse_numeric_measurement
from bran_clinical_semantics_v1 import decode_age, EICU_CBC_NAMES
from bran_eicu_cbc_adapter_v2 import EICU_CBC_COLUMNS, adapt_eicu_cbc_event_v2
from bran_mimic_prior_disease_v1 import canonical_icd, _validate_approved_codes

ERROR = 'eicu_discovery_bridge_v1_contract_failed'
LANDMARK = 1440.
PATIENT_COLUMNS = ('uniquepid', 'patientunitstayid', 'patienthealthsystemstayid',
    'age', 'hospitalid', 'hospitaladmitoffset', 'unitdischargeoffset',
    'hospitaldischargeoffset', 'hospitaldischargestatus')
LAB_COLUMNS = (*EICU_CBC_COLUMNS, 'labresultrevisedoffset')
DIAGNOSIS_COLUMNS = ('patientunitstayid', 'diagnosisoffset', 'icd9code')


def require(ok):
    if not ok:
        raise ValueError(ERROR) from None


def key(value):
    require(type(value) is str and 0 < len(value) <= 128 and value == value.strip()
            and value.isascii() and not any(c.isspace() or ord(c) < 33 for c in value))
    return value


class DiseaseVocabulary:
    """Validate a nonpatient, source-authenticated code dictionary once."""
    __slots__ = ('mapping',)

    def __init__(self, approved_codes):
        try:
            self.mapping = MappingProxyType(_validate_approved_codes(approved_codes))
        except Exception:
            require(False)

    def __repr__(self):
        return '<QualifiedDiseaseVocabulary>'


@dataclass(frozen=True, repr=False)
class Episode:
    person: str
    episode: str
    hospital_stay: str
    hospital: str
    age: object
    hospital_admit_offset: float
    unit_end: float
    hospital_end: float
    outcome: int  # -1 unknown, 0 documented alive, 1 documented expired

    def __repr__(self):
        return '<PrivateEicuEpisode>'


def parse_patient(row):
    """Retain unknown age/outcomes; malformed linkage is a hard safe failure."""
    try:
        require(isinstance(row, Mapping) and set(row) == set(PATIENT_COLUMNS))
        keys = [key(row[n]) for n in ('uniquepid', 'patientunitstayid',
                'patienthealthsystemstayid', 'hospitalid')]
        times = [parse_numeric_measurement(row[n]) for n in
                 ('hospitaladmitoffset', 'unitdischargeoffset', 'hospitaldischargeoffset')]
        times = [math.nan if n is None else n for n in times]
        outcome = {'Alive': 0, 'Expired': 1}.get(row['hospitaldischargestatus'], -1)
        return Episode(*keys, decode_age('eicu', row['age']), *times, outcome)
    except Exception:
        require(False)


def observed_cbc_before_landmark(row, links):
    """Reject late/unknown revisions as well as post-landmark specimen times."""
    try:
        require(isinstance(row, Mapping) and set(row) == set(LAB_COLUMNS))
        if row['labtypeid'] != '3' or row['labname'] not in EICU_CBC_NAMES:
            return None
        revision = parse_numeric_measurement(row['labresultrevisedoffset'])
        specimen = parse_numeric_measurement(row['labresultoffset'])
        if (revision is None or specimen is None or not 0 <= specimen <= LANDMARK
                or not specimen <= revision <= LANDMARK):
            return None
        return adapt_eicu_cbc_event_v2({k: row[k] for k in EICU_CBC_COLUMNS}, links)
    except Exception:
        require(False)


def eligible_at_landmark(episode, observed_fields):
    """Outcome-blind eligibility; age alone never qualifies physiology."""
    require(type(episode) is Episode and type(observed_fields) is int and observed_fields >= 0)
    if not observed_fields or not all(math.isfinite(n) for n in
        (episode.hospital_admit_offset, episode.unit_end, episode.hospital_end)):
        return False
    # Known minors are excluded; unknown age remains explicit, not invented.
    if math.isfinite(episode.age.upper_years) and episode.age.upper_years <= 18:
        return False
    return (episode.hospital_admit_offset <= 0
            and episode.unit_end > LANDMARK
            and episode.hospital_end >= episode.unit_end)


def select_one_per_person(episodes, observed_counts):
    """Outcome/diagnosis-blind deterministic selection, NOT first-ever admission.

    No chronological ordering across hospital stays is assumed. Select an
    eligible hospital stay by fixed hash; inside it choose the earliest eligible
    ICU stay from the admission offset. Never order ICU stays by numeric ID.
    """
    try:
        require(type(observed_counts) is dict)
        chosen = {}
        seen = set()
        hospital_people = {}
        for episode in episodes:
            require(type(episode) is Episode and episode.episode not in seen)
            seen.add(episode.episode)
            care_key = (episode.hospital, episode.hospital_stay)
            require(care_key not in hospital_people or hospital_people[care_key] == episode.person)
            hospital_people[care_key] = episode.person
            count = observed_counts.get(episode.episode, 0)
            if not eligible_at_landmark(episode, count):
                continue
            care_hash = hashlib.sha256(('bran-eicu-index-v1\0' + episode.person + '\0'
                + episode.hospital + '\0' + episode.hospital_stay).encode()).digest()
            # Offset is hospital admission relative to ICU admission. Larger
            # (less negative) offsets denote earlier ICU stays within a stay.
            rank = (care_hash, -episode.hospital_admit_offset, episode.episode)
            previous = chosen.get(episode.person)
            if previous is None or rank < previous[0]:
                chosen[episode.person] = (rank, episode)
        return tuple(chosen[p][1] for p in sorted(chosen))
    except Exception:
        require(False)


def recorded_membership(row, vocabulary, episode):
    """Match canonical codes only; do not inspect free-text diagnosis strings.

    eICU's nominal ICD9 field can carry comma-separated ICD9/ICD10 codes. A
    caller-supplied authenticated vocabulary defines eligible disease codes.
    Whole ICU-stay entries imply retrospective, not at-landmark membership.
    """
    try:
        require(isinstance(row, Mapping) and set(row) == set(DIAGNOSIS_COLUMNS))
        require(type(episode) is Episode and row['patientunitstayid'] == episode.episode)
        require(type(vocabulary) is DiseaseVocabulary)
        approved = vocabulary.mapping
        offset = parse_numeric_measurement(row['diagnosisoffset'])
        if offset is None or not math.isfinite(episode.unit_end) or not 0 <= offset <= episode.unit_end:
            return frozenset()
        payload = row['icd9code']
        require(type(payload) is str)
        if not payload or len(payload) > 512:
            return frozenset()
        tokens = payload.split(',')
        if len(tokens) > 16:
            return frozenset()
        result = set()
        for token in tokens:
            token = token.strip(' ')
            # Numeric and V/E-prefixed ICD9 versus qualified ICD10 are parsed
            # independently. Only explicitly approved codes can set membership.
            for version in ('9', '10'):
                code = canonical_icd(version, token)
                if code in approved:
                    result.add(approved[code])
        return frozenset(result)
    except Exception:
        require(False)
