"""Bounded, synthetic-testable kernel for a new retinal feature extraction.

This module deliberately contains no source authentication, protocol receipt,
or privacy aggregation.  Its inputs are caller-owned inventory records and
caller-supplied loader/encoder functions.  Successful return values contain
only closed scalar metadata; image and feature arrays never leave this module.
"""

from __future__ import annotations

import hashlib
import json
import ntpath
import os
import re
import stat
from typing import Any, Callable, Mapping, NoReturn

import numpy as np


_INVENTORY_KEYS = frozenset(
    {
        "row",
        "relative_path",
        "person_id",
        "laterality",
        "device",
        "source_sha256",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_SIZE = 224
_IMAGE_CHANNELS = 3
_DEFAULT_BATCH_SIZE = 24
_DEFAULT_WIDTH = 384
_READ_CHUNK_BYTES = 1024 * 1024

_ERR_INVENTORY = "invalid retinal inventory"
_ERR_EXTRACTION = "retinal extraction failed"
_ERR_EXISTS = "retinal extraction output already exists"
_ERR_OUTPUT = "invalid retinal extraction output"


def _inventory_error() -> NoReturn:
    raise ValueError(_ERR_INVENTORY)


def _extraction_error() -> NoReturn:
    raise ValueError(_ERR_EXTRACTION)


def _output_error() -> NoReturn:
    raise ValueError(_ERR_OUTPUT)


def _is_exact_int(value: Any) -> bool:
    return type(value) is int


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value:
        return False
    try:
        drive, _ = ntpath.splitdrive(value)
        if drive or os.path.isabs(value) or ntpath.isabs(value):
            return False
    except Exception:
        return False
    components = re.split(r"[\\/]", value)
    if any(component == ".." for component in components):
        return False
    if any(component == "" for component in components):
        return False
    return True


def validate_inventory(records: Any) -> None:
    """Validate a closed, row-ordered inventory without exposing record values."""

    if not isinstance(records, list) or not records:
        _inventory_error()

    paths: list[str] = []
    seen_paths: set[str] = set()
    normalized_paths: set[str] = set()
    for expected_row, record in enumerate(records):
        if not isinstance(record, dict):
            _inventory_error()
        try:
            if frozenset(record.keys()) != _INVENTORY_KEYS:
                _inventory_error()
            row = record["row"]
            relative_path = record["relative_path"]
            person_id = record["person_id"]
            laterality = record["laterality"]
            device = record["device"]
            source_sha256 = record["source_sha256"]
        except Exception:
            _inventory_error()

        if not _is_exact_int(row) or row != expected_row:
            _inventory_error()
        if not _safe_relative_path(relative_path):
            _inventory_error()
        if relative_path in seen_paths:
            _inventory_error()
        # Treat slash and backslash aliases as the same path for safety.
        normalized = relative_path.replace("\\", "/")
        if normalized in normalized_paths:
            _inventory_error()
        seen_paths.add(relative_path)
        normalized_paths.add(normalized)
        paths.append(relative_path)

        if not isinstance(person_id, str) or not person_id:
            _inventory_error()
        if not isinstance(laterality, str) or laterality not in {"L", "R", "unknown"}:
            _inventory_error()
        if not isinstance(device, str) or not device:
            _inventory_error()
        if not isinstance(source_sha256, str) or _SHA256_RE.fullmatch(source_sha256) is None:
            _inventory_error()

    if paths != sorted(paths):
        _inventory_error()


def inventory_sha256(records: Any) -> str:
    """Return the digest of the canonical JSON representation of a valid inventory."""

    validate_inventory(records)
    try:
        payload = json.dumps(
            records,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    except Exception:
        _inventory_error()


def _coerce_path(path: Any, error: Callable[[], None]) -> str | bytes:
    try:
        value = os.fspath(path)
    except Exception:
        error()
    if not isinstance(value, (str, bytes)) or not value:
        error()
    return value


def _validate_positive_int(value: Any, error: Callable[[], None]) -> int:
    if not _is_exact_int(value) or value <= 0:
        error()
    return int(value)


def _validate_loader_image(image: Any) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        _extraction_error()
    if image.shape != (_IMAGE_CHANNELS, _IMAGE_SIZE, _IMAGE_SIZE):
        _extraction_error()
    if image.dtype != np.dtype(np.float32):
        _extraction_error()
    try:
        if not bool(np.isfinite(image).all()):
            _extraction_error()
    except Exception:
        _extraction_error()
    return image


def _validate_encoded_batch(encoded: Any, rows: int, width: int) -> np.ndarray:
    if not isinstance(encoded, np.ndarray):
        _extraction_error()
    if encoded.shape != (rows, width):
        _extraction_error()
    if encoded.dtype != np.dtype(np.float32):
        _extraction_error()
    try:
        if not bool(np.isfinite(encoded).all()):
            _extraction_error()
        # Every produced row must carry a nonzero feature vector.
        if bool(np.any(np.sum(np.abs(encoded), axis=1, dtype=np.float64) == 0.0)):
            _extraction_error()
    except Exception:
        _extraction_error()
    return encoded


def _sha256_file(path: str | bytes) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while True:
                block = handle.read(_READ_CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
    except Exception:
        _output_error()
    return digest.hexdigest()


def _metadata(rows: int, width: int, digest: str) -> dict[str, Any]:
    return {
        "rows": int(rows),
        "dimension": int(width),
        "dtype": "float32",
        "output_sha256": digest,
    }


def validate_output(
    path: Any,
    expected_rows: Any,
    width: Any = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """Validate a feature NPY and return only closed scalar metadata."""

    if not _is_exact_int(expected_rows) or expected_rows <= 0:
        _output_error()
    if not _is_exact_int(width) or width <= 0:
        _output_error()
    output_path = _coerce_path(path, _output_error)

    mapped: np.memmap | None = None
    try:
        file_stat = os.stat(output_path)
        if not stat.S_ISREG(file_stat.st_mode):
            _output_error()
        if stat.S_IMODE(file_stat.st_mode) != 0o600:
            _output_error()

        # mmap_mode plus allow_pickle=False accepts only a numeric NPY/NPZ
        # representation; shape and dtype checks below reject anything but the
        # exact feature matrix emitted by this kernel.
        loaded = np.load(output_path, mmap_mode="r", allow_pickle=False)
        if not isinstance(loaded, np.memmap):
            _output_error()
        mapped = loaded
        if mapped.shape != (expected_rows, width):
            _output_error()
        if mapped.dtype != np.dtype(np.float32):
            _output_error()
        if not bool(mapped.flags.c_contiguous):
            _output_error()

        data_offset = int(mapped.offset)
        expected_size = data_offset + expected_rows * width * np.dtype(np.float32).itemsize
        if int(file_stat.st_size) != expected_size:
            _output_error()

        rows_per_chunk = max(
            1,
            min(
                expected_rows,
                _READ_CHUNK_BYTES // max(1, width * np.dtype(np.float32).itemsize),
            ),
        )
        for start in range(0, expected_rows, rows_per_chunk):
            stop = min(expected_rows, start + rows_per_chunk)
            block = np.asarray(mapped[start:stop])
            if not bool(np.isfinite(block).all()):
                _output_error()
            if bool(np.any(np.sum(np.abs(block), axis=1, dtype=np.float64) == 0.0)):
                _output_error()
    except ValueError as exc:
        if str(exc) == _ERR_OUTPUT:
            raise
        _output_error()
    except Exception:
        _output_error()
    finally:
        if mapped is not None:
            try:
                mapped.flush()
            except Exception:
                pass
            mapped = None

    return _metadata(expected_rows, width, _sha256_file(output_path))


def extract_batches(
    records: Any,
    output_path: Any,
    loader: Callable[[Mapping[str, Any]], np.ndarray],
    encoder: Callable[[np.ndarray], np.ndarray],
    *,
    batch_size: Any = _DEFAULT_BATCH_SIZE,
    width: Any = _DEFAULT_WIDTH,
    progress: Callable[[int, int], Any] | None = None,
) -> dict[str, Any]:
    """Extract finite float32 features into a new mode-0600 NPY memmap.

    The caller owns source authentication and any protocol receipt.  Even
    scalar or per-batch values returned by this function are private to the
    caller; only a separately supported pooled aggregate may be released.
    This kernel is not a release or publication mechanism.
    """

    validate_inventory(records)
    batch_size_int = _validate_positive_int(batch_size, _extraction_error)
    width_int = _validate_positive_int(width, _extraction_error)
    if not callable(loader) or not callable(encoder):
        _extraction_error()
    if progress is not None and not callable(progress):
        _extraction_error()
    output_path_value = _coerce_path(output_path, _extraction_error)

    total_rows = len(records)
    total_batches = (total_rows + batch_size_int - 1) // batch_size_int
    file_descriptor: int | None = None
    handle: Any = None
    mapped: np.memmap | None = None

    try:
        open_flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            open_flags |= os.O_CLOEXEC
        try:
            file_descriptor = os.open(output_path_value, open_flags, 0o600)
        except FileExistsError:
            raise ValueError(_ERR_EXISTS)
        except Exception:
            raise ValueError(_ERR_EXTRACTION)

        os.fchmod(file_descriptor, 0o600)
        handle = os.fdopen(file_descriptor, "w+b", closefd=True)
        file_descriptor = None
        header = {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
            "fortran_order": False,
            "shape": (total_rows, width_int),
        }
        np.lib.format.write_array_header_1_0(handle, header)
        data_offset = int(handle.tell())
        data_bytes = total_rows * width_int * np.dtype(np.float32).itemsize
        handle.truncate(data_offset + data_bytes)
        handle.flush()
        mapped = np.memmap(
            output_path_value,
            dtype=np.float32,
            mode="r+",
            offset=data_offset,
            shape=(total_rows, width_int),
            order="C",
        )

        for batch_index, start in enumerate(range(0, total_rows, batch_size_int), start=1):
            stop = min(total_rows, start + batch_size_int)
            batch_images: list[np.ndarray] = []
            for record in records[start:stop]:
                try:
                    image = loader(dict(record))
                except Exception:
                    _extraction_error()
                batch_images.append(_validate_loader_image(image))

            try:
                stacked = np.stack(batch_images, axis=0)
            except Exception:
                _extraction_error()
            try:
                encoded = encoder(stacked)
            except Exception:
                _extraction_error()
            encoded_array = _validate_encoded_batch(encoded, stop - start, width_int)
            mapped[start:stop] = encoded_array

            if progress is not None:
                try:
                    progress(int(batch_index), int(total_batches))
                except Exception:
                    _extraction_error()

        mapped.flush()
        handle.flush()
        os.fsync(handle.fileno())
    except ValueError as exc:
        if str(exc) in {_ERR_EXISTS, _ERR_EXTRACTION}:
            raise
        raise ValueError(_ERR_EXTRACTION)
    except Exception:
        raise ValueError(_ERR_EXTRACTION)
    finally:
        if mapped is not None:
            try:
                mapped.flush()
            except Exception:
                pass
            mapped = None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except Exception:
                pass

    try:
        return validate_output(output_path_value, total_rows, width_int)
    except Exception:
        _extraction_error()
