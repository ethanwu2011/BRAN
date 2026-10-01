"""Private, source-bound AI-READI adapter for a future paired retinal extraction."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType

import numpy as np

import run_bran_retinal_extraction_v2 as decoder_runner
import run_bran_retinal_extraction_v3 as extraction
import run_bran_retinal_input_bridge_v3 as bridge


PROTOCOL_SHA256 = 'c147f5df7abf4c5f2decb1996e06fa2d620a55ff25eaf9658b9d0b76556e9e95'
AUDIT_SHA256 = '88b61f0a721b037d823634a02aa759573b016781d221d2cd29dff8819361d1fc'
_ERROR = 'multisource paired source rejected'


def _fail():
    raise ValueError(_ERROR) from None


def _require(value):
    if not value:
        _fail()


def _sha(path):
    return extraction.sha(path)


def _selection_identity(expected, ids, folds):
    return {
        'patient_order_sha256': extraction.old.digest_json(ids),
        'fold_order_sha256': extraction.old.digest_json(folds.tolist()),
        'selection_sha256': extraction.old.kernel.inventory_sha256(expected),
        'rows': len(expected), 'people': 1928,
    }


def _regular(path, links):
    value = os.lstat(path)
    _require(stat.S_ISREG(value.st_mode) and not stat.S_ISLNK(value.st_mode)
             and value.st_nlink == links)
    return value


def _authenticate():
    try:
        protocol, aggregate, audit = bridge._authenticate_retinal(PROTOCOL_SHA256, AUDIT_SHA256)
        private = extraction.PRIVATE
        _require(private.is_dir() and not private.is_symlink() and stat.S_IMODE(os.lstat(private).st_mode) == 0o700)
        _require(set(path.name for path in private.iterdir()) == {'inventory.json', 'features.npy'})
        inventory = private / 'inventory.json'
        _require(inventory.is_file() and not inventory.is_symlink() and stat.S_IMODE(os.lstat(inventory).st_mode) == 0o600
                 and os.lstat(inventory).st_nlink == 1)
        _require(_sha(inventory) == aggregate['inventory_file_sha256'] == audit['inventory_file_sha256'])
        # The original successful V3 publication retains an immutable staged
        # receipt hard-linked to its terminal name.  Both are required.
        for directory, terminal, names in (
            (extraction.OUT, 'aggregate.json',
             {'aggregate.json', 'aggregate.json.staged', 'aggregate.manifest.json', 'progress.json'}),
            (extraction.AUDIT, 'audit.json',
             {'audit.json', 'audit.json.staged', 'audit.manifest.json'}),
        ):
            _require(directory.is_dir() and not directory.is_symlink()
                     and set(path.name for path in directory.iterdir()) == names)
            staged = _regular(directory / (terminal + '.staged'), 2)
            published = _regular(directory / terminal, 2)
            _require((staged.st_dev, staged.st_ino) == (published.st_dev, published.st_ino))
            for name in names - {terminal, terminal + '.staged'}:
                _regular(directory / name, 1)
        policy = protocol['origin_protocol']['external_sha256']['source_policy']
        _require(all(type(value) is str and len(value) == 64 for value in (policy, audit['aggregate_sha256'])))
        binding = {
            'schema': 'bran-multisource-paired-source-binding-v1',
            'protocol_sha256': PROTOCOL_SHA256, 'audit_sha256': AUDIT_SHA256,
            'aggregate_sha256': audit['aggregate_sha256'], 'inventory_sha256': aggregate['inventory_file_sha256'],
            'source_policy_sha256': policy,
        }
        return binding, protocol
    except Exception:
        _fail()


def authenticate_binding():
    """Return only fixed receipt identities; no record, image, or feature data."""
    return _authenticate()[0]


class QualifiedPairedSource:
    """Single-use private original-cohort context; it never invents retinal availability."""

    def __init__(self):
        self._used = False
        self._entered = False

    def __repr__(self):
        return '<QualifiedPairedSource private>'

    def __enter__(self):
        _require(not self._used and not self._entered)
        try:
            self.binding, protocol = _authenticate()
            expected, old_rows, selected_ids, identity = extraction.old.selection()
            context, folds, *_ = bridge.origin.native.source.io.load_context()
            ids = list(map(str, context['raw_cohort'].patient_ids))
            _require(len(ids) == len(set(ids)) == 1928 and ids == list(map(str, selected_ids))
                     and set(context['raw_cohort'].split_labels) <= {'train', 'val'}
                     and list(map(str, context['feature_cohort'].patient_ids)) == ids)
            fold_values = np.asarray(folds)
            _require(fold_values.ndim == 1 and len(fold_values) == len(ids)
                     and np.all(np.isin(fold_values, (0, 1, 2, 3, 4))))
            _require(identity == protocol['selection'] == _selection_identity(expected, ids, fold_values))
            records = json.loads((extraction.PRIVATE / 'inventory.json').read_text())
            extraction.old.kernel.validate_inventory(records)
            _require([{**record, 'source_sha256': '0' * 64} for record in records] == expected
                     and len(old_rows) == len(records))
            self.private_records = tuple(MappingProxyType(copy.deepcopy(record)) for record in records)
            self.private_patient_ids = tuple(ids)
            self.private_folds = np.asarray(fold_values, dtype=np.int64).copy()
            self.private_folds.setflags(write=False)
            represented_people = {record['person_id'] for record in records}
            present = np.asarray([person in represented_people for person in ids], dtype=bool)
            present.setflags(write=False)
            self.private_retinal_present = present
            self.private_selection = tuple(int(row) for row in old_rows)
            self._entered = True
            return self
        except Exception:
            self._used = True
            _fail()

    def __exit__(self, *unused):
        try:
            self.reauthenticate_context()
        finally:
            self._entered = False
            self._used = True

    def _record(self, record):
        _require(self._entered and type(record) is dict and type(record.get('row')) is int)
        row = record['row']
        _require(0 <= row < len(self.private_records) and record == self.private_records[row]
                 and record['source_sha256'] != '0' * 64)
        return record

    def read(self, record):
        """Read one revalidated original record as normalized float32 CHW224 pixels."""
        try:
            record = self._record(record)
            root = extraction.old.DATASET.resolve()
            path = root / record['relative_path']
            _require(path.resolve().is_relative_to(root))
            state = {'loader_failure': None}
            image = decoder_runner.make_loader(root, state)(record)
            _require(type(image) is np.ndarray and image.shape == (3, 224, 224) and image.dtype == np.float32
                     and np.all(np.isfinite(image)) and state['loader_failure'] is None)
            return np.ascontiguousarray(image)
        except Exception:
            _fail()

    def reauthenticate_context(self):
        """Recheck fixed V3 binding plus original selection/cohort identity once."""
        try:
            _require(self._entered)
            binding, protocol = _authenticate()
            _require(binding == self.binding)
            expected, old_rows, selected_ids, identity = extraction.old.selection()
            context, folds, *_ = bridge.origin.native.source.io.load_context()
            ids, fold_values = list(map(str, context['raw_cohort'].patient_ids)), np.asarray(folds)
            _require(ids == list(self.private_patient_ids) == list(map(str, selected_ids))
                     and np.array_equal(fold_values, self.private_folds)
                     and set(context['raw_cohort'].split_labels) <= {'train', 'val'}
                     and list(map(str, context['feature_cohort'].patient_ids)) == ids
                     and identity == protocol['selection'] == _selection_identity(expected, ids, fold_values)
                     and [{**dict(record), 'source_sha256': '0' * 64} for record in self.private_records] == expected
                     and tuple(int(row) for row in old_rows) == self.private_selection)
            return True
        except Exception:
            _fail()

    def reauthenticate_all_bytes(self):
        """Bounded four-worker byte reauthentication; no pixels or identities are returned."""
        try:
            _require(self._entered)
            root = extraction.old.DATASET.resolve()

            def check(record):
                path = root / record['relative_path']
                _require(path.resolve().is_relative_to(root) and _sha(path) == record['source_sha256'])
                return True

            with ThreadPoolExecutor(max_workers=4) as pool:
                _require(all(pool.map(check, self.private_records)))
            self.reauthenticate_context()
            return True
        except Exception:
            _fail()
