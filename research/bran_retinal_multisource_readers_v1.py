"""Local qualified-image readers; permission and scientific training are separate.

Caller must hold the common compute lock and keep FD output quiet. Returned
metadata, bytes, images and batches are private and must never be displayed.
No source is discovered, matched or admitted for training by this module.
"""
import hashlib
import json
from pathlib import Path
import threading
import zipfile

import numpy as np

import check_bran_retinal_multisource_source_binding_v1 as binding
from bran_retinal_multisource_source_contract_v1 import PixelBridge, brset_spec, odir_spec

brset, odir, r = binding.brset, binding.odir, binding.r
ERROR = 'multisource retinal reader failed'
BINDING_SHA256 = '0c58d68abf817a2f16922c2644cc2148ede54f70260c111bddde2ab402427ba1'


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def authenticate_binding():
    r.regular(binding.DESTINATION)
    require(r.sha(binding.DESTINATION) == BINDING_SHA256)
    value = json.loads(binding.DESTINATION.read_text())
    require(value['status'] == 'passed' and value['training_admitted'] is False
            and value['patient_level_output_emitted'] is False and value['images_read'] is False
            and value['real_data_training_run'] is False)
    current = binding.identities()
    require(all(value[key] == current[key] for key in current))
    return current


class StableArchive:
    """One bounded ZIP handle across batches, with serialized byte reads.

Pixel decoding runs outside the lock. Repeated short-lived pixel executors do
not accumulate one open ZIP handle per thread or per batch.
"""

    def __repr__(self):
        return '<StableArchive private>'

    def __init__(self, path):
        self._path = Path(path)
        self._lock = threading.RLock()
        self._archive = None
        self._before = None
        self._used = False

    def _stamp(self):
        require(self._path.is_file() and not self._path.is_symlink())
        st = self._path.stat()
        return st.st_size, st.st_mtime_ns, st.st_ino

    def __enter__(self):
        try:
            with self._lock:
                require(not self._used)
                self._used = True
                self._before = self._stamp()
                self._archive = zipfile.ZipFile(self._path)
                require(self._stamp() == self._before)
            return self
        except Exception:
            if self._archive is not None:
                self._archive.close()
                self._archive = None
            raise ValueError(ERROR) from None

    def read(self, member):
        try:
            with self._lock:
                require(self._archive is not None and self._stamp() == self._before)
                require(type(member) is str and len(member) > 0)
                info = self._archive.getinfo(member)
                require(not info.is_dir() and not info.flag_bits & 1
                        and 0 < info.file_size <= odir.content.decode.LIMIT)
                with self._archive.open(info) as stream:
                    data = stream.read(odir.content.decode.LIMIT + 1)
                require(len(data) == info.file_size and self._stamp() == self._before)
                return data
        except Exception:
            raise ValueError(ERROR) from None

    def __exit__(self, kind, value, trace):
        try:
            with self._lock:
                require(self._archive is not None)
                try:
                    if kind is None:
                        require(self._stamp() == self._before)
                finally:
                    self._archive.close()
                    self._archive = None
        except Exception:
            raise ValueError(ERROR) from None


def checked_odir_pixels(data, raw_digest, pixel_digest):
    try:
        require(type(data) is bytes and type(raw_digest) is bytes and len(raw_digest) == 32
                and type(pixel_digest) is bytes and len(pixel_digest) == 32)
        require(hashlib.sha256(data).digest() == raw_digest)
        pixels = odir.content.decode.pixels(data, 224)
        require(hashlib.sha256(pixels.tobytes()).digest() == pixel_digest)
        return pixels
    except Exception:
        raise ValueError(ERROR) from None


class QualifiedSources:
    """Context-bound, authenticated local sources; all returned arrays private."""

    def __repr__(self):
        return '<QualifiedSources private; not training admission>'

    def __init__(self):
        self._active = False
        self._used = False
        self._archive = None

    def __enter__(self):
        try:
            require(not self._used)
            self._used = True
            self._identity = authenticate_binding()
            self.private_specs, self.private_weights = binding.load_qualified_specs()
            self._brset = brset.load_source()
            b, bw, _ = brset_spec(self._brset['arrays'])
            inputs = brset.load_arrays(odir.PRIVATE / 'inputs.npz')
            split = brset.load_arrays(odir.PRIVATE / 'assignment.npz')['split']
            selection = json.loads((odir.content.PRIVATE / 'selection.json').read_text())
            flags = brset.load_arrays(odir.content.PRIVATE / 'flags.npz')
            o, ow, rows = odir_spec(inputs, split, selection, flags['retained_representative'])
            for name, spec, weights in (('brset', b, bw), ('odir', o, ow)):
                require(all(np.array_equal(spec[key], self.private_specs[name][key], equal_nan=True) for key in spec)
                        and np.array_equal(weights, self.private_weights[name]))
            self._members = [selection[index]['member'] for index in rows]
            fingerprints = brset.load_arrays(odir.content.PRIVATE / 'fingerprints.npz')
            self._raw = fingerprints['raw_sha256'][rows].copy()
            self._pixels = fingerprints['decoded224_sha256'][rows].copy()
            locator = json.loads((odir.content.PRIVATE / 'source_locator.json').read_text())
            self._archive = StableArchive(locator['archive'])
            self._archive.__enter__()
            require(authenticate_binding() == self._identity)
            self._active = True
            return self
        except Exception:
            if self._archive is not None and self._archive._archive is not None:
                self._archive.__exit__(ValueError, None, None)
            raise ValueError(ERROR) from None

    def read(self, source, index):
        try:
            require(self._active and type(source) is str and source in ('brset', 'odir')
                    and type(index) is int and 0 <= index < len(self.private_specs[source]['image_groups']))
            if source == 'brset':
                return brset.read_pixels(self._brset, index)
            data = self._archive.read(self._members[index])
            return checked_odir_pixels(data, bytes(self._raw[index]), bytes(self._pixels[index]))
        except Exception:
            raise ValueError(ERROR) from None

    def materialize(self, plan):
        require(self._active)
        return PixelBridge({name: lambda index, name=name: self.read(name, index)
                            for name in ('brset', 'odir')}).materialize(plan)

    def __exit__(self, kind, value, trace):
        try:
            require(self._active)
            self._active = False
            try:
                self._archive.__exit__(kind, value, trace)
            finally:
                require(authenticate_binding() == self._identity)
        except Exception:
            raise ValueError(ERROR) from None
