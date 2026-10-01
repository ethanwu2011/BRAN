"""Local aggregate-only C2 coverage census for MIMIC day-one physiology.

This module receives only already-authenticated ``AvailableEvent`` objects.
It performs neither source I/O nor outcome handling and never exposes event,
episode, or person state in its public result.  ``private_gate_pairs`` is the
sole deliberately private hand-off for comparison with an authenticated linked
snapshot map in the same local privacy boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
import math
from numbers import Real

from bran_cbc_chemistry_snapshot_v1 import CLINICAL_SNAPSHOT_FIELDS
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_clinical_source_reader_v1 import EpisodeLinks
from bran_joint_lab_online_v1 import _checked_event
from bran_mimic_landmark_source_v1 import AvailableEvent, PredictorMetadata


SCHEMA = "bran-mimic-upstream-coverage-c2"
STATUS = "completed_aggregate_only"
SCOPE = "mimic_source_local_valid_admissions_no_age_or_outcome_restriction"
INTERPRETATION = "potential_day_one_measurement_coverage_not_new_cohort_validation_or_utility"
_MIN_SUPPORT = 20
_DAY_MINUTES = 1440.0
_GATE_MINUTES = 60.0
_FIELD_COUNT = len(CLINICAL_SNAPSHOT_FIELDS)
_CBC_COUNT = len(CBC_FIELDS)
_PATTERNS = (
    "any_day_one_physiology",
    "any_cbc",
    "any_chemistry",
    "chemistry_with_fewer_than_two_cbc",
    "no_cbc_but_chemistry",
    "at_least_two_cbc_anywhere_day_one",
    "existing_two_cbc_60_minute_gate",
    "broad_physiology_without_existing_gate",
    "no_broad_physiology",
    "person_level_expansion",
)


def _fail() -> None:
    raise ValueError("bran_mimic_upstream_coverage_c2_contract_failed") from None


def _plain_text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _coarse(count: int) -> int:
    if type(count) is not int or count < 0:
        _fail()
    if count == 0:
        return 0
    if count < _MIN_SUPPORT:
        _fail()
    return (count // _MIN_SUPPORT) * _MIN_SUPPORT


def _pair_cell(positive: int, negative: int) -> dict[str, object]:
    """Return a disclosure-safe person count and complement, or withhold both."""

    if type(positive) is not int or type(negative) is not int or positive < 0 or negative < 0:
        _fail()
    if (positive and positive < _MIN_SUPPORT) or (negative and negative < _MIN_SUPPORT):
        return {"status": "withheld"}
    return {
        "status": "released",
        "positive_people_lower_bound_20": _coarse(positive),
        "complement_people_lower_bound_20": _coarse(negative),
    }


def _validate_metadata(metadata: object) -> dict[str, str]:
    """Close the denominator to validly linked MIMIC admissions only."""

    if type(metadata) is not PredictorMetadata or set(vars(metadata)) != {"links", "admissions", "ages"}:
        _fail()
    links = metadata.links
    if (
        type(links) is not EpisodeLinks
        or set(vars(links)) != {"source", "episode_to_person", "episode_to_care_group"}
        or links.source != "mimic"
        or links.episode_to_care_group is not None
        or not isinstance(links.episode_to_person, Mapping)
        or not isinstance(metadata.admissions, Mapping)
        or not isinstance(metadata.ages, Mapping)
    ):
        _fail()
    try:
        episodes = set(links.episode_to_person)
        if episodes != set(metadata.admissions) or episodes != set(metadata.ages):
            _fail()
    except (TypeError, ValueError):
        _fail()

    valid: dict[str, str] = {}
    for episode, person in links.episode_to_person.items():
        if not _plain_text(episode) or not _plain_text(person):
            _fail()
        admission = metadata.admissions[episode]
        # predictor_metadata represents failed admission parsing by ``None``;
        # these remain outside the source-local denominator.
        if admission is None:
            continue
        if not isinstance(admission, datetime):
            _fail()
        valid[episode] = person
    return valid


def _checked_available_item(item: object, valid_admissions: Mapping[str, str]):
    """Validate the source adapter's available-event contract without source I/O."""

    if type(item) is not AvailableEvent or set(vars(item)) != {"event", "available_minutes"}:
        _fail()
    event = _checked_event(item.event)
    if event.source != "mimic" or event.episode_key not in valid_admissions:
        _fail()
    if valid_admissions[event.episode_key] != event.person_key:
        _fail()
    available = item.available_minutes
    if isinstance(available, bool) or not isinstance(available, Real):
        _fail()
    try:
        available_float = float(available)
    except (TypeError, ValueError, OverflowError):
        _fail()
    if (
        not math.isfinite(available_float)
        or not 0.0 <= float(event.offset_minutes) <= available_float <= _DAY_MINUTES
        or not 0 <= int(event.field_index) < _FIELD_COUNT
    ):
        _fail()
    return event


def _is_public_scalar(value: object) -> bool:
    return type(value) in (str, int, bool)


def _validate_pair_cell(value: object) -> None:
    if not isinstance(value, dict):
        _fail()
    if value == {"status": "withheld"}:
        return
    if set(value) != {
        "status",
        "positive_people_lower_bound_20",
        "complement_people_lower_bound_20",
    } or value["status"] != "released":
        _fail()
    for key in ("positive_people_lower_bound_20", "complement_people_lower_bound_20"):
        number = value[key]
        if type(number) is not int or number < 0 or (number and (number < _MIN_SUPPORT or number % _MIN_SUPPORT)):
            _fail()


def validate_result(result: object) -> None:
    """Reject tampered, non-aggregate, or non-disclosure-safe public output."""

    if not isinstance(result, dict) or set(result) != {
        "schema",
        "status",
        "scope",
        "interpretation",
        "population",
        "patterns",
        "category_semantics",
        "patient_level_output_emitted",
        "arrays_or_ids_emitted",
    }:
        _fail()
    if (
        result["schema"] != SCHEMA
        or result["status"] != STATUS
        or result["scope"] != SCOPE
        or result["interpretation"] != INTERPRETATION
        or result["patient_level_output_emitted"] is not False
        or result["arrays_or_ids_emitted"] is not False
    ):
        _fail()
    population = result["population"]
    if not isinstance(population, dict):
        _fail()
    if population == {"status": "withheld"}:
        pass
    elif set(population) == {
        "status",
        "source_local_people_lower_bound_20",
        "valid_admissions_lower_bound_20",
    } and population["status"] == "released":
        for key in ("source_local_people_lower_bound_20", "valid_admissions_lower_bound_20"):
            number = population[key]
            if type(number) is not int or number < _MIN_SUPPORT or number % _MIN_SUPPORT:
                _fail()
    else:
        _fail()
    patterns = result["patterns"]
    if not isinstance(patterns, dict) or set(patterns) != set(_PATTERNS):
        _fail()
    for cell in patterns.values():
        _validate_pair_cell(cell)
    semantics = result["category_semantics"]
    if semantics != {
        "unit": "distinct_source_local_people_with_any_valid_admission",
        "patterns_overlap_across_admissions": True,
        "person_level_expansion_excludes_any_existing_gate_admission": True,
        "no_outcomes_or_age_filters": True,
    }:
        _fail()
    # The fixed schema above contains only scalar leaves.  This final check
    # keeps future edits from quietly releasing a nested array or identifier.
    def visit(value: object) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif not _is_public_scalar(value):
            _fail()
    visit(result)


class CoverageAccumulator:
    """Accumulate earliest valid event states and emit a closed public census.

    Per-episode state remains private and local.  Events may arrive in any
    order: a field's earliest time wins, equal earliest values collapse, and a
    conflicting equal-time value permanently makes that field unavailable.
    """

    def __init__(self, metadata: PredictorMetadata):
        self._valid_admissions = _validate_metadata(metadata)
        self._states: dict[str, dict[int, tuple[float, float, bool]]] = {}
        self._finished = False
        self._result: dict[str, object] | None = None

    def __repr__(self) -> str:
        return "<CoverageAccumulator aggregate-only>"

    __str__ = __repr__

    def add(self, item: AvailableEvent) -> None:
        if self._finished:
            _fail()
        event = _checked_available_item(item, self._valid_admissions)
        fields = self._states.setdefault(event.episode_key, {})
        field = int(event.field_index)
        candidate = (float(event.offset_minutes), float(event.canonical_value), False)
        prior = fields.get(field)
        if prior is None or candidate[0] < prior[0]:
            fields[field] = candidate
        elif candidate[0] == prior[0] and candidate[1] != prior[1]:
            # Do not replace the original time; later rows can never repair a
            # conflict at that earliest time.
            fields[field] = (prior[0], prior[1], True)

    @staticmethod
    def _episode_patterns(fields: Mapping[int, tuple[float, float, bool]]) -> dict[str, bool]:
        usable_cbc = [field for field in range(_CBC_COUNT) if field in fields and not fields[field][2]]
        usable_chemistry = [
            field for field in range(_CBC_COUNT, _FIELD_COUNT) if field in fields and not fields[field][2]
        ]
        all_cbc_times = [fields[field][0] for field in range(_CBC_COUNT) if field in fields]
        gate = False
        if all_cbc_times:
            anchor = min(all_cbc_times)
            gate = sum(fields[field][0] <= anchor + _GATE_MINUTES for field in usable_cbc) >= 2
        broad = bool(usable_cbc or usable_chemistry)
        any_cbc = bool(usable_cbc)
        any_chemistry = bool(usable_chemistry)
        cbc_count = len(usable_cbc)
        return {
            "any_day_one_physiology": broad,
            "any_cbc": any_cbc,
            "any_chemistry": any_chemistry,
            "chemistry_with_fewer_than_two_cbc": any_chemistry and cbc_count < 2,
            "no_cbc_but_chemistry": any_chemistry and not any_cbc,
            "at_least_two_cbc_anywhere_day_one": cbc_count >= 2,
            "existing_two_cbc_60_minute_gate": gate,
            "broad_physiology_without_existing_gate": broad and not gate,
        }

    def _person_patterns(self) -> tuple[dict[str, set[str]], set[tuple[str, str]]]:
        person_flags: dict[str, set[str]] = {
            person: set() for person in set(self._valid_admissions.values())
        }
        gate_pairs: set[tuple[str, str]] = set()
        for episode, person in self._valid_admissions.items():
            episode_flags = self._episode_patterns(self._states.get(episode, {}))
            positive = person_flags[person]
            for name, enabled in episode_flags.items():
                if enabled:
                    positive.add(name)
            if episode_flags["existing_two_cbc_60_minute_gate"]:
                gate_pairs.add((person, episode))

        for positive in person_flags.values():
            if "any_day_one_physiology" not in positive:
                positive.add("no_broad_physiology")
            if (
                "any_day_one_physiology" in positive
                and "existing_two_cbc_60_minute_gate" not in positive
            ):
                positive.add("person_level_expansion")
        return person_flags, gate_pairs

    def private_gate_pairs(self) -> set[tuple[str, str]]:
        """Return local-only ``(person, episode)`` gate pairs for authenticated comparison."""

        _, pairs = self._person_patterns()
        return set(pairs)

    def finish(self) -> dict[str, object]:
        """Close the accumulator and return a fresh disclosure-safe aggregate."""

        if self._result is not None:
            return deepcopy(self._result)
        person_flags, _ = self._person_patterns()
        total_people = len(person_flags)
        valid_admissions = len(self._valid_admissions)
        if total_people < _MIN_SUPPORT:
            population: dict[str, object] = {"status": "withheld"}
        else:
            population = {
                "status": "released",
                "source_local_people_lower_bound_20": _coarse(total_people),
                "valid_admissions_lower_bound_20": _coarse(valid_admissions),
            }
        patterns = {
            name: _pair_cell(
                sum(name in positive for positive in person_flags.values()),
                sum(name not in positive for positive in person_flags.values()),
            )
            for name in _PATTERNS
        }
        result: dict[str, object] = {
            "schema": SCHEMA,
            "status": STATUS,
            "scope": SCOPE,
            "interpretation": INTERPRETATION,
            "population": population,
            "patterns": patterns,
            "category_semantics": {
                "unit": "distinct_source_local_people_with_any_valid_admission",
                "patterns_overlap_across_admissions": True,
                "person_level_expansion_excludes_any_existing_gate_admission": True,
                "no_outcomes_or_age_filters": True,
            },
            "patient_level_output_emitted": False,
            "arrays_or_ids_emitted": False,
        }
        validate_result(result)
        self._finished = True
        self._result = deepcopy(result)
        return deepcopy(result)
