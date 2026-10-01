"""Private deterministic two-source patient-balanced batch planning only."""
from __future__ import annotations

import hashlib

import numpy as np


_ERROR = "retinal multisource sampling failed"
_SOURCES = ("brset", "odir")
_FIELDS = ("image_indices", "flips", "patch_mask", "group_index", "labels", "observed")
_WIDTHS = {"brset": 13, "odir": 8}


def _fail() -> None:
    raise ValueError(_ERROR) from None


def _array(value: object, dtype: np.dtype, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != dtype or (shape is not None and value.shape != shape):
        _fail()
    return value


def _copy(value: np.ndarray) -> np.ndarray:
    copied = np.array(value, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _source(value: object, name: str) -> dict[str, np.ndarray]:
    if type(value) is not dict or set(value) != {"image_groups", "group_split", "labels", "observed", "usable"}:
        _fail()
    groups = _array(value["image_groups"], np.dtype(np.int64))
    split = _array(value["group_split"], np.dtype(np.uint8))
    if groups.ndim != 1 or split.ndim != 1 or len(groups) == 0 or len(split) == 0:
        _fail()
    if (groups < 0).any() or int(groups.max()) >= len(split):
        _fail()
    represented = np.zeros((len(split),), dtype=np.bool_)
    represented[groups] = True
    if not represented.all() or not np.isin(split, np.asarray([0, 1, 2], dtype=np.uint8)).all() or not (split == 0).any():
        _fail()
    group_sizes = np.zeros((len(split),), dtype=np.int64)
    np.add.at(group_sizes, groups, 1)
    if name == "brset":
        if (group_sizes < 1).any():
            _fail()
    elif (group_sizes < 1).any() or (group_sizes > 2).any():
        _fail()
    width = _WIDTHS[name]
    labels = value["labels"]
    expected_rows = len(groups) if name == "brset" else len(split)
    if type(labels) is not np.ndarray or labels.dtype.kind != "f" or labels.shape != (expected_rows, width):
        _fail()
    observed = _array(value["observed"], np.dtype(bool), labels.shape)
    usable = _array(value["usable"], np.dtype(bool), (width,))
    finite = np.isfinite(labels)
    if np.any(observed & (~finite | ((labels != 0.0) & (labels != 1.0)))):
        _fail()
    return {"image_groups": _copy(groups), "group_split": _copy(split), "labels": _copy(labels),
            "observed": _copy(observed), "usable": _copy(usable)}


def _config(value: object, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        _fail()
    return value


def _seed(value: object) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0 or value >= 2 ** 64:
        _fail()
    return value


class BatchPlanner:
    """Single-use private two-source stream; it never loads or transforms images."""

    def __init__(self, sources: dict[str, dict[str, np.ndarray]], *, steps: int = 512, batch_size: int = 16,
                 seeds: dict[str, int] = {"brset": 74192, "odir": 74193}) -> None:
        if type(sources) is not dict or set(sources) != set(_SOURCES):
            _fail()
        if type(seeds) is not dict or set(seeds) != set(_SOURCES):
            _fail()
        self._sources = {name: _source(sources[name], name) for name in _SOURCES}
        self._steps = _config(steps, "steps")
        self._batch_size = _config(batch_size, "batch_size")
        self._rng = {name: np.random.default_rng(_seed(seeds[name])) for name in _SOURCES}
        self._training_groups = {
            name: np.flatnonzero(value["group_split"] == 0).astype(np.int64, copy=False)
            for name, value in self._sources.items()
        }
        self._images = {
            name: tuple(np.flatnonzero(value["image_groups"] == group).astype(np.int64, copy=False)
                        for group in range(len(value["group_split"])))
            for name, value in self._sources.items()
        }
        self._used = False
        self._completed = 0
        self._digest = hashlib.sha256()

    def __repr__(self) -> str:
        return "<BatchPlanner private>"

    def _update_array(self, value: np.ndarray) -> None:
        self._digest.update(value.dtype.str.encode("ascii"))
        self._digest.update(len(value.shape).to_bytes(2, "big"))
        for axis in value.shape:
            self._digest.update(int(axis).to_bytes(8, "big", signed=False))
        canonical = np.ascontiguousarray(value)
        if canonical.dtype.kind == "f" and np.isnan(canonical).any():
            canonical = canonical.copy()
            canonical[np.isnan(canonical)] = np.nan
        self._digest.update(canonical.tobytes())

    def _record(self, step: int, batch: dict[str, dict[str, np.ndarray]]) -> None:
        self._digest.update(b"step")
        self._digest.update(step.to_bytes(8, "big", signed=False))
        for source in _SOURCES:
            self._digest.update(source.encode("ascii"))
            for field in _FIELDS:
                self._digest.update(field.encode("ascii"))
                self._update_array(batch[source][field])

    def _batch(self, source: str) -> dict[str, np.ndarray]:
        values = self._sources[source]
        rng = self._rng[source]
        sampled_groups = rng.choice(self._training_groups[source], size=self._batch_size, replace=True)
        image_indices: list[int] = []
        occurrence: list[int] = []
        label_rows: list[int] = []
        for group_occurrence, group in enumerate(sampled_groups):
            candidates = self._images[source][int(group)]
            if source == "brset":
                selected = int(rng.choice(candidates))
                image_indices.append(selected)
                occurrence.append(group_occurrence)
                label_rows.append(selected)
            else:
                label_rows.append(int(group))
                for selected in candidates:
                    image_indices.append(int(selected))
                    occurrence.append(group_occurrence)
        indices = np.asarray(image_indices, dtype=np.int64)
        group_index = np.asarray(occurrence, dtype=np.int64)
        labels = np.asarray(values["labels"][np.asarray(label_rows, dtype=np.int64)], dtype=np.float32).copy()
        observed = np.asarray(values["observed"][np.asarray(label_rows, dtype=np.int64)], dtype=np.bool_).copy()
        observed &= values["usable"][None, :]
        labels[~observed] = np.nan
        flips = rng.random(len(indices)) < 0.5
        masks = np.zeros((len(indices), 196), dtype=np.bool_)
        for row in range(len(indices)):
            masks[row, rng.choice(196, size=147, replace=False)] = True
        output = {
            "image_indices": np.ascontiguousarray(indices, dtype=np.int64),
            "flips": np.ascontiguousarray(flips, dtype=np.bool_),
            "patch_mask": np.ascontiguousarray(masks, dtype=np.bool_),
            "group_index": np.ascontiguousarray(group_index, dtype=np.int64),
            "labels": np.ascontiguousarray(labels, dtype=np.float32),
            "observed": np.ascontiguousarray(observed, dtype=np.bool_),
        }
        return output

    def __iter__(self):
        if self._used:
            _fail()
        self._used = True
        for step in range(self._steps):
            batch = {source: self._batch(source) for source in _SOURCES}
            self._record(step, batch)
            self._completed += 1
            yield batch

    def receipt(self) -> dict[str, object]:
        return {"schema": "bran-retinal-multisource-sampling-v1", "completed_steps": self._completed,
                "stream_sha256": self._digest.hexdigest()}


__all__ = ["BatchPlanner"]
