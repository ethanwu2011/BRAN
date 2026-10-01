"""Pure candidate HiRID observed-assay extraction; no I/O, admission, or scoring."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_knhanes_input_kernel_v1 import CANONICAL_NAMES


ERROR = "hirid_v5_observation_contract_failed"
TARGET_IDS = (20000900, 24000836)  # Candidate routine-blood Hb channels; candidate policy only.
EXCLUDED_ARTERIAL_HB = 24000548
INPUT_CANDIDATES = MappingProxyType({
    20000500: ("potassium", 1.0), 20000400: ("sodium", 1.0),
    24000439: ("chloride", 1.0), 20000600: ("creatinine", 1.0 / 88.4),
    20004300: ("bilirubin_total", 1.0 / 17.1), 24000605: ("albumin", 1.0 / 10.0),
    20005110: ("glucose", 18.0182),
})
STATUS = frozenset(("ready", "abstain_no_target", "abstain_no_physiology", "target_conflict"))


@dataclass(frozen=True, slots=True, repr=False)
class Event:
    variable_id: int
    time_hours: float
    value: float
    observed: bool


@dataclass(frozen=True, slots=True, repr=False)
class EpisodeSelection:
    target_hb: float
    time_hours: float
    canonical_labs: Mapping[str, float]
    status: str

    def __repr__(self) -> str:
        return "<HiRIDV5ObservationResult>"

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)


class _Invalid(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _Invalid


def _number(value: object) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _observed(event: object) -> tuple[int, float, float | None] | None:
    _require(type(event) is Event)
    _require(isinstance(event.observed, (bool, np.bool_)))
    if not bool(event.observed):
        # Deliberately do not parse a missing-event id, time, or payload.
        return None
    _require(isinstance(event.variable_id, Integral) and not isinstance(event.variable_id, (bool, np.bool_)))
    identifier = int(event.variable_id)
    time = _number(event.time_hours)
    _require(time is not None)
    return identifier, time, _number(event.value)


def _selection(target: float = math.nan, time: float = math.nan, labs: Mapping[str, float] | None = None,
               status: str = "abstain_no_target") -> EpisodeSelection:
    _require(status in STATUS)
    frozen = MappingProxyType(dict(labs or {}))
    return EpisodeSelection(float(target), float(time), frozen, status)


def select_episode(events: Iterable[Event]) -> EpisodeSelection:
    """Select one outcome-blind anchor and its causal non-CBC lab prefix.

    Identifier/unit policies in this candidate-only kernel remain subject to an
    independent metadata review before any source admission or model use.
    """
    try:
        observed = []
        for event in events:
            parsed = _observed(event)
            if parsed is not None:
                observed.append(parsed)
        targets: dict[float, dict[int, list[float]]] = {}
        for identifier, time, value in observed:
            if identifier in TARGET_IDS and 0.0 <= time <= 24.0 and value is not None and value > 0.0:
                targets.setdefault(time, {}).setdefault(identifier, []).append(value)
        if not targets:
            return _selection()
        anchor = min(targets)
        channels = targets[anchor]
        # Both admitted routine-Hb channels represent the same target/unit.
        # Never prefer one channel when contemporaneous observed truth conflicts.
        values = [value for samples in channels.values() for value in samples]
        if len(set(values)) != 1:
            return _selection(time=anchor, status="target_conflict")
        target = values[0] * .1
        _require(math.isfinite(target) and target > 0.0)
        candidates: dict[str, dict[float, list[float]]] = {}
        for identifier, time, value in observed:
            policy = INPUT_CANDIDATES.get(identifier)
            if policy is None or value is None or not (0.0 <= time <= anchor and anchor - time <= 6.0):
                continue
            field, factor = policy
            converted = value * factor
            if math.isfinite(converted):
                candidates.setdefault(field, {}).setdefault(time, []).append(converted)
        labs: dict[str, float] = {}
        for field, samples in candidates.items():
            latest = max(samples)
            values = samples[latest]
            # Conflicted latest samples remove that field; no earlier fallback.
            if len(set(values)) == 1:
                labs[field] = values[0]
        _require(set(labs).issubset(set(CANONICAL_NAMES)) and not set(labs).intersection(CBC_FIELDS))
        return _selection(target, anchor, labs, "ready" if labs else "abstain_no_physiology")
    except _Invalid:
        raise ValueError(ERROR) from None
    except Exception:
        raise ValueError(ERROR) from None
