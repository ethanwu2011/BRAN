"""Pure broad MIMIC day-one snapshot accumulator for the M1 hand-off.

The accumulator receives only validated AvailableEvent objects. It does not
read source rows, apply outcomes or labels, deduplicate people, or publish
counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bran_clinical_semantics_v1 import AgeObservation, CBC_FIELDS
from bran_cbc_chemistry_snapshot_v1 import CLINICAL_SNAPSHOT_FIELDS
from bran_mimic_landmark_source_v1 import AvailableEvent, PredictorMetadata
from bran_mimic_upstream_coverage_c2 import (
    _checked_available_item,
    _plain_text,
    _validate_metadata,
)


ERROR = "mimic_broad_snapshot_m1_contract_failed"
FIELD_COUNT = len(CLINICAL_SNAPSHOT_FIELDS)
CBC_COUNT = len(CBC_FIELDS)
DAY_MINUTES = 1440.0


def _fail() -> None:
    raise ValueError(ERROR) from None


def _require(condition: bool) -> None:
    if not condition:
        _fail()


def _readonly(value: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    copied = np.array(value, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, repr=False)
class BroadSnapshot:
    """Private one-admission broad snapshot with no serialization boundary."""

    person: str
    episode: str
    values: np.ndarray
    observed: np.ndarray
    specimen_minutes: np.ndarray
    available_minutes: np.ndarray
    age: AgeObservation

    def __post_init__(self) -> None:
        try:
            _require(_plain_text(self.person) and _plain_text(self.episode))
            _require(type(self.age) is AgeObservation)
            values = np.asarray(self.values)
            observed = np.asarray(self.observed)
            specimen = np.asarray(self.specimen_minutes)
            available = np.asarray(self.available_minutes)
            _require(
                type(values) is np.ndarray
                and values.shape == (FIELD_COUNT,)
                and values.dtype == np.float64
                and type(observed) is np.ndarray
                and observed.shape == (FIELD_COUNT,)
                and observed.dtype == np.bool_
                and type(specimen) is np.ndarray
                and specimen.shape == (FIELD_COUNT,)
                and specimen.dtype == np.float64
                and type(available) is np.ndarray
                and available.shape == (FIELD_COUNT,)
                and available.dtype == np.float64
            )
            _require(np.isfinite(values).all() and (values[~observed] == 0).all())
            _require((values[observed] >= 0).all())
            _require((values[:CBC_COUNT][observed[:CBC_COUNT]] > 0).all())
            _require(np.isnan(specimen[~observed]).all())
            _require(np.isnan(available[~observed]).all())
            _require(
                np.isfinite(specimen[observed]).all()
                and (specimen[observed] >= 0).all()
                and (specimen[observed] <= DAY_MINUTES).all()
                and np.isfinite(available[observed]).all()
                and (specimen[observed] <= available[observed]).all()
                and (available[observed] <= DAY_MINUTES).all()
            )
            object.__setattr__(self, "values", _readonly(values, np.float64))
            object.__setattr__(self, "observed", _readonly(observed, np.bool_))
            object.__setattr__(self, "specimen_minutes", _readonly(specimen, np.float64))
            object.__setattr__(self, "available_minutes", _readonly(available, np.float64))
        except Exception:
            _fail()

    def __repr__(self) -> str:
        return "<PrivateMimicBroadSnapshot>"

    def __reduce__(self):
        raise TypeError("mimic_broad_snapshot_m1_serialization_forbidden")


PrivateBroadSnapshot = BroadSnapshot


class BroadSnapshotAccumulator:
    """Retain the latest valid specimen for each field in each admission."""

    def __init__(self, metadata: PredictorMetadata):
        try:
            valid_admissions = _validate_metadata(metadata)
            _require(type(metadata) is PredictorMetadata)
            ages = {}
            for episode in valid_admissions:
                age = metadata.ages[episode]
                _require(type(age) is AgeObservation)
                ages[episode] = age
            self._valid_admissions = dict(valid_admissions)
            self._ages = ages
            self._states: dict[str, dict[int, tuple[float, float, float, bool]]] = {}
        except Exception:
            _fail()

    def __repr__(self) -> str:
        return "<BroadSnapshotAccumulator private>"

    __str__ = __repr__

    def add(self, item: AvailableEvent) -> None:
        """Add one validated event; invalid linkage/time fails closed."""
        try:
            _require(type(item) is AvailableEvent)
            event = _checked_available_item(item, self._valid_admissions)
            available = float(item.available_minutes)
            field = int(event.field_index)
            specimen = float(event.offset_minutes)
            value = float(event.canonical_value)
            episode = event.episode_key
            fields = self._states.setdefault(episode, {})
            prior = fields.get(field)
            if prior is None or specimen > prior[0]:
                fields[field] = (specimen, value, available, False)
            elif specimen == prior[0]:
                if value != prior[1]:
                    fields[field] = (
                        prior[0],
                        prior[1],
                        max(prior[2], available),
                        True,
                    )
                elif available > prior[2]:
                    fields[field] = (prior[0], prior[1], available, prior[3])
        except Exception:
            _fail()

    def private_snapshots(self):
        """Yield one private snapshot for each admission with an observed field."""
        try:
            for episode in sorted(self._states):
                fields = self._states[episode]
                values = np.zeros(FIELD_COUNT, dtype=np.float64)
                observed = np.zeros(FIELD_COUNT, dtype=np.bool_)
                specimen = np.full(FIELD_COUNT, np.nan, dtype=np.float64)
                available = np.full(FIELD_COUNT, np.nan, dtype=np.float64)
                for field, (field_time, value, field_available, conflict) in fields.items():
                    if conflict:
                        continue
                    values[field] = value
                    observed[field] = True
                    specimen[field] = field_time
                    available[field] = field_available
                if not observed.any():
                    continue
                yield BroadSnapshot(
                    self._valid_admissions[episode],
                    episode,
                    values,
                    observed,
                    specimen,
                    available,
                    self._ages[episode],
                )
        except Exception:
            _fail()
