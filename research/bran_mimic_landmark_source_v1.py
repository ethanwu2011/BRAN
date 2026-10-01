"""Local-only MIMIC source adapter for a new, explicitly timed clinical cache.

No source I/O, model fitting, or printing occurs here. Caller-owned iterables
must come from authenticated original sources under OS-level output suppression.
This does not retrofit identity onto an old cache or assert clinical efficacy.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

from bran_cbc_event_adapter_v1 import _naive_iso_datetime
from bran_clinical_semantics_v1 import CBC_FIELDS, decode_age
from bran_clinical_source_reader_v1 import (
    build_episode_links, validate_event_link, _numeric_key,
)
from bran_joint_lab_online_v1 import (
    CBCAnchorAccumulator, JointLabSecondPass, authenticate_code_binding,
    convert_joint_bound_event,
)
from bran_mimic_linked_snapshot_contract_v1 import (
    LANDMARK_MINUTES, LinkedSnapshot, validate_linked_snapshot_pack,
)

PATIENT_COLUMNS = ('subject_id', 'anchor_age', 'anchor_year')
ADMISSION_COLUMNS = ('subject_id', 'hadm_id', 'admittime')
LAB_COLUMNS = ('subject_id', 'hadm_id', 'itemid', 'valuenum', 'valueuom',
               'charttime', 'storetime')
OUTCOME_COLUMNS = ADMISSION_COLUMNS + ('dischtime', 'deathtime', 'hospital_expire_flag')
OUTCOME_STATUSES = (
    'eligible_death', 'eligible_survival', 'invalid_admission_time',
    'invalid_discharge_time', 'invalid_expire_flag', 'missing_death_time',
    'invalid_death_time', 'inconsistent_death_status', 'death_outside_admission',
    'discharged_before_or_at_landmark', 'death_before_or_at_landmark',
)


def _require(ok):
    if not ok:
        raise ValueError('mimic_landmark_source_contract_failed')


def _row(row, columns):
    _require(isinstance(row, Mapping) and set(row) == set(columns))
    return row


def _close(iterator):
    method = getattr(iterator, 'close', None)
    if method is not None:
        method()


@dataclass(frozen=True, repr=False)
class PredictorMetadata:
    links: object
    admissions: Mapping
    ages: Mapping


@dataclass(frozen=True, repr=False)
class AvailableEvent:
    event: object
    available_minutes: float


@dataclass(frozen=True, repr=False)
class LandmarkOutcome:
    status: str
    label: int | None
    end_minutes: float | None


def predictor_metadata(patient_rows, admission_rows):
    """Construct predictor metadata without accepting any outcome columns."""
    patients, episodes = [], []
    for rows, columns, target in (
        (patient_rows, PATIENT_COLUMNS, patients),
        (admission_rows, ADMISSION_COLUMNS, episodes),
    ):
        iterator = iter(rows)
        try:
            for row in iterator:
                target.append(dict(_row(row, columns)))
        finally:
            _close(iterator)
    links = build_episode_links('mimic', patients, episodes)
    masters = {_numeric_key(row['subject_id']): row for row in patients}
    admissions, ages = {}, {}
    for row in episodes:
        episode, person = _numeric_key(row['hadm_id']), _numeric_key(row['subject_id'])
        start = _naive_iso_datetime(row['admittime'])
        admissions[episode] = start
        age = masters[person]
        ages[episode] = decode_age('mimic', age['anchor_age'],
            anchor_year=age['anchor_year'], admission_year=start.year if start else None)
    return PredictorMetadata(links, MappingProxyType(admissions), MappingProxyType(ages))


def available_events(rows, metadata, binding, *, anchors=None):
    """Use storetime, not charttime, to prove availability by admission+24h.

    Unknown/missing/late availability remains unavailable. No inferred hadm_id,
    time-only join, outcome filter, or value-dependent upper clipping is used.
    """
    _require(isinstance(metadata, PredictorMetadata) and binding.source == 'mimic')
    iterator = iter(rows)
    try:
        for row in iterator:
            _row(row, LAB_COLUMNS)
            if row['itemid'] not in binding.code_to_field:
                continue
            person = validate_event_link('mimic', row, metadata.links)
            if person is None:
                continue
            episode = _numeric_key(row['hadm_id'])
            start = metadata.admissions.get(episode)
            chart, stored = _naive_iso_datetime(row['charttime']), _naive_iso_datetime(row['storetime'])
            if start is None or chart is None or stored is None:
                continue
            offset = (chart - start).total_seconds() / 60.
            available = (stored - start).total_seconds() / 60.
            if not 0. <= offset <= available <= LANDMARK_MINUTES:
                continue
            if anchors is not None:
                anchor = anchors.anchor_minutes.get(episode, math.nan)
                if not math.isfinite(anchor) or not anchor <= offset <= min(anchor + 60., LANDMARK_MINUTES):
                    continue
            event = convert_joint_bound_event('mimic', episode, person,
                row['itemid'], row['valuenum'], row['valueuom'], offset, 1, binding)
            if event is not None:
                yield AvailableEvent(event, available)
    finally:
        _close(iterator)


def linked_snapshots(rows_factory, metadata, code_to_field):
    """Two full source passes; preserve original source identity and timestamps.

    Earliest-time conflicting values stay missing, never replaced by a later
    result. Identical earliest-time duplicates use the latest eligible storetime
    conservatively. The 60-minute context is not claimed to be a single draw.
    """
    _require(callable(rows_factory) and isinstance(metadata, PredictorMetadata))
    all_labs = authenticate_code_binding('mimic', code_to_field)
    cbc = authenticate_code_binding('mimic',
        {code: field for code, field in code_to_field.items() if field in CBC_FIELDS})
    first = CBCAnchorAccumulator('mimic')
    stream = available_events(rows_factory(), metadata, cbc)
    try:
        for item in stream:
            first.add(item.event)
    finally:
        stream.close()
    anchors = first.finalize()
    second = JointLabSecondPass(anchors)
    selected = {}
    stream = available_events(rows_factory(), metadata, all_labs, anchors=anchors)
    try:
        for item in stream:
            event = item.event
            second.add(event)
            if event.episode_key not in selected:
                selected[event.episode_key] = (np.full(21, np.nan), np.full(21, np.nan))
            times, availability = selected[event.episode_key]
            field = event.field_index
            if math.isnan(times[field]) or event.offset_minutes < times[field]:
                times[field], availability[field] = event.offset_minutes, item.available_minutes
            elif event.offset_minutes == times[field]:
                availability[field] = max(availability[field], item.available_minutes)
    finally:
        stream.close()
    for episode, person, snapshot in second.iterate_snapshots():
        if int(snapshot.observed[:9].sum()) < 2:
            continue
        times, availability = selected[episode]
        _require(np.array_equal(times[snapshot.observed], snapshot.selected_offsets_minutes[snapshot.observed]))
        available = availability.copy()
        available[~snapshot.observed] = np.nan
        yield LinkedSnapshot(person, episode, snapshot.values, snapshot.observed,
            snapshot.conflicts, snapshot.anchor_minutes, snapshot.selected_offsets_minutes,
            available, metadata.ages[episode])


def classify_landmark_outcome(row):
    """Recorded index-hospital death after24h, NOT fixed-horizon mortality.

    Alive discharge is the competing end of this admission, not proof of later
    survival. Ambiguous timing never becomes a negative mortality label.
    """
    _row(row, OUTCOME_COLUMNS)
    start = _naive_iso_datetime(row['admittime'])
    if start is None:
        return LandmarkOutcome('invalid_admission_time', None, None)
    stop = _naive_iso_datetime(row['dischtime'])
    if stop is None or stop < start:
        return LandmarkOutcome('invalid_discharge_time', None, None)
    flag = row['hospital_expire_flag']
    if type(flag) is not str or flag not in ('0', '1'):
        return LandmarkOutcome('invalid_expire_flag', None, None)
    end = (stop - start).total_seconds() / 60.
    death_raw = row['deathtime']
    if flag == '0':
        if death_raw != '':
            return LandmarkOutcome('inconsistent_death_status', None, None)
        if end <= LANDMARK_MINUTES:
            return LandmarkOutcome('discharged_before_or_at_landmark', None, None)
        return LandmarkOutcome('eligible_survival', 0, end)
    if death_raw == '':
        return LandmarkOutcome('missing_death_time', None, None)
    death = _naive_iso_datetime(death_raw)
    if death is None:
        return LandmarkOutcome('invalid_death_time', None, None)
    if not start <= death <= stop:
        return LandmarkOutcome('death_outside_admission', None, None)
    event_time = (death - start).total_seconds() / 60.
    if event_time <= LANDMARK_MINUTES:
        return LandmarkOutcome('death_before_or_at_landmark', None, None)
    return LandmarkOutcome('eligible_death', 1, event_time)


def join_landmark_outcomes(arrays, private_map, salt, source_sha256, outcome_rows):
    """Authenticate each snapshot, then join outcomes by episode AND person.

    Return separate private arrays. Mortality/eligibility cannot alter predictors.
    Source pins and the caller's before/after source hash checks authenticate the
    admission rows; this pure function cannot attest source files independently.
    """
    validate_linked_snapshot_pack(arrays, private_map, salt, source_sha256)
    targets = {row['episode']: row for row in private_map['rows']}
    joined, seen = {}, set()
    iterator = iter(outcome_rows)
    try:
        for row in iterator:
            _row(row, OUTCOME_COLUMNS)
            episode, person = _numeric_key(row['hadm_id']), _numeric_key(row['subject_id'])
            _require(episode not in seen)
            seen.add(episode)
            if episode not in targets:
                continue
            _require(targets[episode]['person'] == person)
            joined[episode] = classify_landmark_outcome(row)
    finally:
        _close(iterator)
    _require(set(joined) == set(targets))
    ordered = [joined[row['episode']] for row in private_map['rows']]
    result = {
        'row_binding': arrays['row_binding'].copy(),
        'status': np.array([OUTCOME_STATUSES.index(x.status) for x in ordered], np.uint8),
        'label': np.array([-1 if x.label is None else x.label for x in ordered], np.int8),
        'end_minutes': np.array([np.nan if x.end_minutes is None else x.end_minutes for x in ordered]),
    }
    for array in result.values():
        array.setflags(write=False)
    return result
