"""Bounded local ZIP reading and authenticated JPEG/PNG content fingerprints.

No images are persisted. Callers must keep all returned bytes/arrays private.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import threading
import zipfile

import numpy as np
from PIL import Image

from bran_retinal_content_decode_v1 import decode_jpeg
import run_bran_retinal_content_admission_v1 as reference

LIMIT = 64 * 1024 * 1024
MAX_PIXELS = 40_000_000
ERROR = 'odir content decoding failed'


def require(condition):
    if not condition:
        raise ValueError(ERROR) from None


def pixels(data, size):
    """Same JPEG transform as reference fingerprints; explicit PNG fallback."""
    try:
        require(type(data) is bytes and 0 < len(data) <= LIMIT)
        require(type(size) is int and size in (128, 224))
        with Image.open(io.BytesIO(data)) as opened:
            require(0 < opened.width * opened.height <= MAX_PIXELS)
            require(opened.format in ('JPEG', 'PNG') and getattr(opened, 'n_frames', 1) == 1)
            if opened.format == 'JPEG':
                return decode_jpeg(data, hashlib.sha256(data).hexdigest(), size=size)
            with opened.convert('RGB') as rgb:
                with rgb.resize((size, size), Image.Resampling.BICUBIC) as resized:
                    return np.array(resized, dtype=np.uint8, copy=True, order='C')
    except Exception:
        raise ValueError(ERROR) from None


def fingerprint(data):
    decoded, ph, dh = reference.hash_pixels(pixels(data, 128))
    return {
        'raw_sha256': np.frombuffer(hashlib.sha256(data).digest(), dtype=np.uint8).copy(),
        'decoded_sha256': decoded, 'phash': ph, 'dhash': dh,
        'decoded224_sha256': np.frombuffer(hashlib.sha256(pixels(data, 224).tobytes()).digest(),
                                         dtype=np.uint8).copy(),
    }


class PrivateArchive:
    """One ZIP handle per worker, closed after joined threads; ordered bounded map."""

    def __repr__(self):
        return '<PrivateArchive private>'

    def __init__(self, path):
        self.path = Path(path)
        self.local = threading.local()
        self.handles = []
        self.mutex = threading.Lock()
        self.before = None

    def stamp(self):
        s = self.path.stat()
        return s.st_size, s.st_mtime_ns, s.st_ino

    def __enter__(self):
        require(self.path.is_file() and not self.path.is_symlink())
        self.before = self.stamp()
        return self

    def read(self, member):
        try:
            require(self.before is not None and self.stamp() == self.before)
            if not hasattr(self.local, 'archive'):
                self.local.archive = zipfile.ZipFile(self.path)
                with self.mutex:
                    self.handles.append(self.local.archive)
            archive = self.local.archive
            info = archive.getinfo(member)
            require(not info.is_dir() and 0 < info.file_size <= LIMIT and not info.flag_bits & 1)
            with archive.open(info) as stream:
                data = stream.read(LIMIT + 1)
            require(len(data) == info.file_size and self.stamp() == self.before)
            return data
        except Exception:
            raise ValueError(ERROR) from None

    def map(self, function, members, progress):
        values = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(0, len(members), 128):
                def one(index):
                    return function(index, self.read(members[index]))
                values.extend(pool.map(one, range(start, min(start + 128, len(members)))))
                progress(len(values), len(members))
        return values

    def __exit__(self, kind, value, trace):
        try:
            for handle in self.handles:
                handle.close()
        finally:
            if kind is None:
                require(self.stamp() == self.before)


def stack_fingerprints(rows):
    require(type(rows) is list and len(rows) > 0)
    keys = {'raw_sha256', 'decoded_sha256', 'phash', 'dhash', 'decoded224_sha256'}
    require(all(type(row) is dict and set(row) == keys for row in rows))
    return {key: np.stack([row[key] for row in rows]) for key in sorted(keys)}


def validate_hashes(value, n):
    require(type(value) is dict and set(value) == {
        'raw_sha256', 'decoded_sha256', 'phash', 'dhash', 'decoded224_sha256'})
    require(type(n) is int and n > 0)
    for key, array in value.items():
        short = key in ('phash', 'dhash')
        require(type(array) is np.ndarray and array.dtype == np.dtype(np.uint64 if short else np.uint8)
                and array.shape == ((n,) if short else (n, 32)))
