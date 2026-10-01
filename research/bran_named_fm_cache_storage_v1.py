"""Private persistence for a fresh, already-extracted named-FM cache.

This module only writes and authenticates local arrays supplied by its caller.
It neither authorizes a source nor performs extraction, encoder training, or
readout fitting.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import stat

import numpy as np

import run_bran_fm_learning_curve_v1 as fm


ERROR = 'named fm cache storage rejected'
MANIFEST = 'manifest.json'


def _fail():
    raise ValueError(ERROR) from None


def _require(value):
    if not value:
        _fail()


def _mode(value):
    return stat.S_IMODE(value.st_mode)


def _hex(value):
    return type(value) is str and len(value) == 64 and all(item in '0123456789abcdef' for item in value)


def _directory(path, *, private):
    value = os.lstat(path)
    _require(stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode))
    if private:
        _require(_mode(value) == 0o700)
    return value


def _regular(path, *, private=True):
    value = os.lstat(path)
    _require(stat.S_ISREG(value.st_mode) and not stat.S_ISLNK(value.st_mode) and value.st_nlink == 1)
    if private:
        _require(_mode(value) == 0o600)
    return value


def _sha(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        digest = hashlib.sha256()
        while True:
            value = os.read(fd, 1 << 20)
            if not value:
                return digest.hexdigest()
            digest.update(value)
    finally:
        os.close(fd)


def _bytes(path, expected):
    """Read only the same private inode whose whole-file hash was checked."""
    before = _regular(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        _require((opened.st_dev, opened.st_ino, opened.st_size) ==
                 (before.st_dev, before.st_ino, before.st_size))
        chunks = []
        while True:
            value = os.read(fd, 1 << 20)
            if not value:
                break
            chunks.append(value)
        result = b''.join(chunks)
        after = os.fstat(fd)
        current = _regular(path)
        _require((after.st_dev, after.st_ino, after.st_size) ==
                 (before.st_dev, before.st_ino, before.st_size) ==
                 (current.st_dev, current.st_ino, current.st_size)
                 and hashlib.sha256(result).hexdigest() == expected)
        return result
    finally:
        os.close(fd)


def _array(variant, value):
    _require(type(variant) is str and variant in fm.ARMS
             and type(value) is np.ndarray and value.dtype == np.dtype('float32')
             and value.shape == (fm.PATIENT_COUNT, fm.WIDTHS[variant])
             and bool(np.isfinite(value).all()))


def _exact_manifest(value):
    try:
        fm._validate_cache_manifest(value)
        _require(value['embedding_files'] == {
            variant: {'path': variant + '.npy', 'sha256': value['embedding_files'][variant]['sha256'],
                      'dtype': 'float32', 'shape': [fm.PATIENT_COUNT, fm.WIDTHS[variant]],
                      'array_key': None}
            for variant in fm.ARMS
        })
        artifact = value['artifact_files'][fm.ARMS[0]]
        _require(all(value['artifact_files'][variant] == artifact for variant in fm.ARMS)
                 and Path(artifact['path']).is_absolute())
        return artifact
    except Exception:
        _fail()


class Writer:
    """Single-use private writer.  Its representation never discloses a path."""
    def __init__(self, path, entry):
        self._path = Path(path)
        self._entry = (entry.st_dev, entry.st_ino)
        self._hashes = {}
        self._finished = False

    def __repr__(self):
        return '<NamedFMCacheWriter private>'

    def _ready(self):
        _require(not self._finished)
        entry = _directory(self._path, private=True)
        _require((entry.st_dev, entry.st_ino) == self._entry)

    def _publish(self, name, write):
        try:
            self._ready()
            target = self._path / name
            _require(not os.path.lexists(target))
            temporary = self._path / ('.' + name + '.' + secrets.token_hex(12) + '.tmp')
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as handle:
                write(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            _regular(temporary)
            self._ready()
            os.link(temporary, target)
            os.unlink(temporary)
            _regular(target)
            self._ready()
            parent_fd = os.open(self._path, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except Exception:
            _fail()

    def write(self, variant, array):
        """Persist exactly one ordered, finite float32 embedding matrix."""
        try:
            self._ready()
            _require(type(variant) is str and len(self._hashes) < len(fm.ARMS)
                     and variant == fm.ARMS[len(self._hashes)])
            _array(variant, array)
            name = variant + '.npy'
            self._publish(name, lambda handle: np.save(handle, array, allow_pickle=False))
            digest = _sha(self._path / name)
            _require(_hex(digest))
            self._hashes[variant] = digest
        except Exception:
            _fail()

    def finish(self, *, current_inputs_sha256, row_order_sha256, outer_fold_sha256,
               inner_fold_sha256, artifact_path, artifact_sha256):
        """Write the manifest last and return its whole-file SHA-256 only."""
        try:
            self._ready()
            _require(tuple(self._hashes) == tuple(fm.ARMS)
                     and all(_hex(value) for value in (current_inputs_sha256, row_order_sha256,
                                                        outer_fold_sha256, artifact_sha256))
                     and type(inner_fold_sha256) is list and len(inner_fold_sha256) == fm.INNER_FOLD_COUNT
                     and all(_hex(value) for value in inner_fold_sha256))
            artifact = Path(artifact_path)
            _require(artifact.is_absolute())
            _regular(artifact, private=False)
            _require(_sha(artifact) == artifact_sha256)
            artifact_spec = {'path': str(artifact.resolve()), 'sha256': artifact_sha256}
            value = {
                'schema': fm.CACHE_SCHEMA, 'status': 'frozen_local_only', 'variant_order': list(fm.ARMS),
                'patient_count': fm.PATIENT_COUNT, 'dimensions': dict(fm.WIDTHS),
                'embedding_files': {variant: {'path': variant + '.npy', 'sha256': self._hashes[variant],
                                               'dtype': 'float32',
                                               'shape': [fm.PATIENT_COUNT, fm.WIDTHS[variant]],
                                               'array_key': None} for variant in fm.ARMS},
                'artifact_files': {variant: dict(artifact_spec) for variant in fm.ARMS},
                'current_inputs_sha256': current_inputs_sha256, 'row_order_sha256': row_order_sha256,
                'outer_fold_sha256': outer_fold_sha256, 'inner_fold_sha256': list(inner_fold_sha256),
                'patient_level_output_emitted': False, 'encoder_training': False,
            }
            _exact_manifest(value)
            self._publish(MANIFEST, lambda handle: handle.write(
                json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')))
            self._finished = True
            digest = _sha(self._path / MANIFEST)
            _require(authenticate(self._path, digest) is True)
            return digest
        except Exception:
            _fail()


def create(root: Path):
    """Create a fresh mode-0700 child directory under an existing parent."""
    try:
        path = Path(root)
        _directory(path.parent, private=False)
        _require(not os.path.lexists(path))
        os.mkdir(path, 0o700)
        os.chmod(path, 0o700)
        return Writer(path, _directory(path, private=True))
    except Exception:
        _fail()


def authenticate(root: Path, manifest_sha256: str):
    """Verify exact private inventory, pins, byte hashes, and array contracts."""
    try:
        path = Path(root)
        entry = _directory(path, private=True)
        _require(_hex(manifest_sha256) and set(os.listdir(path)) ==
                 {*(variant + '.npy' for variant in fm.ARMS), MANIFEST})
        manifest_path = path / MANIFEST
        before_manifest = _regular(manifest_path)
        _require(_sha(manifest_path) == manifest_sha256)
        value = json.loads(_bytes(manifest_path, manifest_sha256).decode('utf-8'))
        artifact = _exact_manifest(value)
        _regular(Path(artifact['path']), private=False)
        _require(_sha(Path(artifact['path'])) == artifact['sha256'])
        for variant in fm.ARMS:
            spec = value['embedding_files'][variant]
            array_path = path / spec['path']
            before = _regular(array_path)
            payload = _bytes(array_path, spec['sha256'])
            loaded = np.load(io.BytesIO(payload), allow_pickle=False)
            try:
                _array(variant, loaded)
            finally:
                if hasattr(loaded, 'close'):
                    loaded.close()
            after = _regular(array_path)
            _require((before.st_dev, before.st_ino, before.st_size) ==
                     (after.st_dev, after.st_ino, after.st_size) and _sha(array_path) == spec['sha256'])
        # Reuse the consuming runner's closed manifest binding only after the
        # private byte/inode checks above; it returns no embedding arrays.
        fm._cache_binding(manifest_path)
        after_manifest = _regular(manifest_path)
        now = _directory(path, private=True)
        _require((before_manifest.st_dev, before_manifest.st_ino, before_manifest.st_size) ==
                 (after_manifest.st_dev, after_manifest.st_ino, after_manifest.st_size)
                 and _sha(manifest_path) == manifest_sha256
                 and (entry.st_dev, entry.st_ino) == (now.st_dev, now.st_ino))
        return True
    except Exception:
        _fail()
