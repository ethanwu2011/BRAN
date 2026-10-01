"""Fail-closed private persistence for matched retinal multisource artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import copy

import numpy as np
import torch


_ERROR = 'private retinal multisource artifact persistence failed'
_FINAL_NAMES = frozenset((
    'evaluation_inputs.npz', 'base_features.npz', 'source_control_features.npz',
    'multisource_candidate_features.npz', 'readouts.pt', 'streams.json',
))
_NPZ_NAMES = frozenset(name for name in _FINAL_NAMES if name.endswith('.npz'))
_TORCH_NAMES = frozenset(('readouts.pt',))
_JSON_NAMES = frozenset(('streams.json',))
_CHECKPOINTS = {'source_control': 'source_control.pt', 'multisource_candidate': 'multisource_candidate.pt'}
_INVENTORY = frozenset((*_FINAL_NAMES, *_CHECKPOINTS.values()))


def _fail():
    raise ValueError(_ERROR) from None


def _require(value):
    if not value:
        _fail()


def _mode(value):
    return stat.S_IMODE(value.st_mode)


def _directory(path):
    try:
        value = os.lstat(path)
    except OSError:
        _fail()
    _require(stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode) and _mode(value) == 0o700)
    return value


def _regular(path):
    try:
        value = os.lstat(path)
    except OSError:
        _fail()
    _require(stat.S_ISREG(value.st_mode) and not stat.S_ISLNK(value.st_mode) and _mode(value) == 0o600
             and value.st_nlink == 1)
    return value


def _sha256(path):
    digest = hashlib.sha256()
    try:
        with open(path, 'rb') as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError:
        _fail()
    return digest.hexdigest()


def _safe_name(name, allowed):
    _require(type(name) is str and name in allowed and Path(name).name == name and '/' not in name and '\\' not in name)


def _receipt(arm, receipt):
    _require(type(arm) is str and arm in _CHECKPOINTS and type(receipt) is dict
             and receipt.get('schema') == 'bran-retinal-multisource-training-receipt-v1')
    config = receipt.get('training_config')
    completed = receipt.get('completed_step')
    _require(type(config) is dict and type(completed) is int and not isinstance(completed, bool) and completed > 0)
    steps, weight = config.get('steps'), config.get('source_weight')
    _require(type(steps) is int and not isinstance(steps, bool) and steps > 0 and completed <= steps
             and type(weight) in (int, float) and not isinstance(weight, bool)
             and float(weight) == (0.0 if arm == 'source_control' else 1.0))
    return completed, steps


class PrivateBundle:
    """A single private bundle. Its representation never exposes its path or data."""

    def __init__(self, path, directory_stat):
        self._path = Path(path)
        self._directory_stat = (directory_stat.st_dev, directory_stat.st_ino, _mode(directory_stat))
        self._sealed = False
        self._finals = {}
        self._checkpoints = {}

    def __repr__(self):
        return '<PrivateBundle private>'

    @classmethod
    def create(cls, path):
        try:
            target = Path(path)
            parent = target.parent
            parent_stat = os.lstat(parent)
            _require(stat.S_ISDIR(parent_stat.st_mode) and not stat.S_ISLNK(parent_stat.st_mode)
                     and not os.path.lexists(target))
            os.mkdir(target, mode=0o700)
            os.chmod(target, 0o700)
            created = _directory(target)
            return cls(target, created)
        except Exception:
            _fail()

    def _ready(self):
        _require(not self._sealed)
        current = _directory(self._path)
        _require((current.st_dev, current.st_ino, _mode(current)) == self._directory_stat)

    def _target(self, name):
        target = self._path / name
        _require(not os.path.lexists(target))
        return target

    def _temporary(self, name):
        return self._path / f'.{name}.{secrets.token_hex(16)}.tmp'

    def _publish_new(self, name, writer):
        self._ready()
        target = self._target(name)
        temporary = self._temporary(name)
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'wb') as handle:
                writer(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            _regular(temporary)
            # link() is atomic and refuses a concurrent destination; unlike
            # replace(), it cannot overwrite a pre-existing final artifact.
            os.link(temporary, target)
            os.unlink(temporary)
            stored = _regular(target)
            self._finals[name] = {
                'stat': (stored.st_dev, stored.st_ino, stored.st_size, _mode(stored)), 'sha256': _sha256(target),
            }
            self._sync_directory()
        except Exception:
            _fail()

    def _sync_directory(self):
        try:
            descriptor = os.open(self._path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            _fail()

    def write_npz(self, name, arrays):
        _safe_name(name, _NPZ_NAMES)
        _require(type(arrays) is dict and len(arrays) > 0 and all(type(key) is str and type(value) is np.ndarray
                 and value.dtype.kind in 'biufc' for key, value in arrays.items()))
        try:
            self._publish_new(name, lambda handle: np.savez_compressed(handle, **arrays))
        except Exception:
            _fail()

    def write_torch(self, name, value):
        _safe_name(name, _TORCH_NAMES)
        try:
            self._publish_new(name, lambda handle: torch.save(value, handle))
        except Exception:
            _fail()

    def write_json(self, name, value):
        _safe_name(name, _JSON_NAMES)

        def writer(handle):
            encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
            handle.write(encoded)

        try:
            self._publish_new(name, writer)
        except Exception:
            _fail()

    def checkpoint(self, arm, receipt):
        completed, steps = _receipt(arm, receipt)
        self._ready()
        name = _CHECKPOINTS[arm]
        target = self._path / name
        previous = self._checkpoints.get(arm)
        try:
            configuration = copy.deepcopy(receipt['training_config'])
        except Exception:
            _fail()
        if previous is None:
            _require(not os.path.lexists(target))
        else:
            _require(completed > previous['completed'])
            _require(configuration == previous['configuration'])
            current = _regular(target)
            _require((current.st_dev, current.st_ino, current.st_size, _mode(current)) == previous['stat']
                     and _sha256(target) == previous['sha256'])
        temporary = self._temporary(name)
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'wb') as handle:
                torch.save(receipt, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            _regular(temporary)
            if previous is None:
                os.link(temporary, target)
                os.unlink(temporary)
            else:
                os.replace(temporary, target)
            stored = _regular(target)
            self._checkpoints[arm] = {
                'completed': completed, 'steps': steps,
                'configuration': configuration,
                'stat': (stored.st_dev, stored.st_ino, stored.st_size, _mode(stored)), 'sha256': _sha256(target),
            }
            self._sync_directory()
        except Exception:
            _fail()

    def seal(self, expected_steps):
        _require(type(expected_steps) is int and not isinstance(expected_steps, bool) and expected_steps > 0)
        self._ready()
        _require(set(self._finals) == _FINAL_NAMES)
        for name, entry in self._finals.items():
            current = _regular(self._path / name)
            _require((current.st_dev, current.st_ino, current.st_size, _mode(current)) == entry['stat']
                     and _sha256(self._path / name) == entry['sha256'])
        _require(set(self._checkpoints) == set(_CHECKPOINTS))
        for arm, entry in self._checkpoints.items():
            _require(entry['completed'] == expected_steps and entry['steps'] == expected_steps)
            current = _regular(self._path / _CHECKPOINTS[arm])
            _require((current.st_dev, current.st_ino, current.st_size, _mode(current)) == entry['stat']
                     and _sha256(self._path / _CHECKPOINTS[arm]) == entry['sha256'])
        try:
            names = set(os.listdir(self._path))
        except OSError:
            _fail()
        _require(names == _INVENTORY)
        manifest = {}
        for name in sorted(_INVENTORY):
            _regular(self._path / name)
            manifest[name] = _sha256(self._path / name)
        self._sealed = True
        return manifest


def authenticate(path, manifest):
    """Read-only inventory/hash authentication; never deserialize private files."""
    try:
        target = Path(path)
        _directory(target)
        _require(type(manifest) is dict and set(manifest) == _INVENTORY
                 and all(type(value) is str and len(value) == 64
                         and all(character in '0123456789abcdef' for character in value)
                         for value in manifest.values()))
        _require(set(os.listdir(target)) == _INVENTORY)
        for name in _INVENTORY:
            artifact = target / name
            _regular(artifact)
            _require(_sha256(artifact) == manifest[name])
    except Exception:
        _fail()
