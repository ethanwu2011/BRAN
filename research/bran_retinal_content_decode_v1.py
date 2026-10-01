"""Private, authenticated RGB pixel decoding for content hashing only.

Returned pixels are not model inputs, embeddings, or a recovery of normalized
arrays.  This module performs no source I/O or image persistence.
"""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import io
import re

import numpy as np
import pydicom
from pydicom.encaps import generate_frames
from PIL import Image


class ContentDecodeError(ValueError):
    """Fixed, disclosure-safe decode failure without input-derived details."""

    def __init__(self) -> None:
        super().__init__("retinal_content_decode_failed")

    def __repr__(self) -> str:
        return "ContentDecodeError()"


def _fail() -> None:
    raise ContentDecodeError() from None


def _validate_identity(data: object, expected_sha256: object, size: object) -> bytes:
    try:
        if (
            type(data) is not bytes
            or type(expected_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
            or type(size) is not int
            or size not in (128, 224)
            or hashlib.sha256(data).hexdigest() != expected_sha256
        ):
            _fail()
        return data
    except ContentDecodeError:
        raise
    except Exception:
        _fail()


def _safe_close(resource: object) -> None:
    try:
        resource.close()
    except Exception:
        _fail()


def _register_close(stack: ExitStack, resource: object, registered: set[int]) -> object:
    if resource is not None and id(resource) not in registered:
        registered.add(id(resource))
        stack.callback(_safe_close, resource)
    return resource


def _pixels(image: Image.Image, size: int, stack: ExitStack, registered: set[int]) -> np.ndarray:
    resized = image.resize((size, size), Image.Resampling.BICUBIC)
    _register_close(stack, resized, registered)
    result = np.array(resized, dtype=np.uint8, copy=True, order="C")
    if result.shape != (size, size, 3) or result.dtype != np.dtype(np.uint8):
        _fail()
    return np.ascontiguousarray(result)


def decode_jpeg(data: bytes, expected_sha256: str, size: int = 128) -> np.ndarray:
    """Decode authenticated JPEG bytes to contiguous uint8 RGB pixels."""
    payload = _validate_identity(data, expected_sha256, size)
    try:
        with ExitStack() as stack:
            registered: set[int] = set()
            encoded = io.BytesIO(payload)
            stack.callback(_safe_close, encoded)
            opened = Image.open(encoded)
            _register_close(stack, opened, registered)
            if opened.format != "JPEG":
                _fail()
            opened.draft("RGB", (size, size))
            converted = opened.convert("RGB")
            _register_close(stack, converted, registered)
            return _pixels(converted, size, stack, registered)
    except ContentDecodeError:
        raise
    except Exception:
        _fail()


def decode_dicom128(data: bytes, expected_sha256: str) -> np.ndarray:
    """Decode authenticated single-frame DICOM bytes to 128px uint8 RGB pixels."""
    payload = _validate_identity(data, expected_sha256, 128)
    try:
        with ExitStack() as stack:
            registered: set[int] = set()
            source = io.BytesIO(payload)
            stack.callback(_safe_close, source)
            dataset = pydicom.dcmread(source)
            if int(getattr(dataset, "NumberOfFrames", 1)) != 1:
                _fail()
            if bool(dataset.file_meta.TransferSyntaxUID.is_encapsulated):
                frames = generate_frames(dataset.PixelData, number_of_frames=1)
                if getattr(frames, "close", None) is not None:
                    stack.callback(_safe_close, frames)
                frame = next(frames)
                if next(frames, None) is not None:
                    _fail()
                encoded = io.BytesIO(frame)
                stack.callback(_safe_close, encoded)
                opened = Image.open(encoded)
                _register_close(stack, opened, registered)
                if opened.format == "JPEG2000":
                    reduction = 0
                    while min(opened.size) / 2 ** (reduction + 1) >= 128:
                        reduction += 1
                    opened.reduce = reduction
                    opened.load()
                else:
                    # The embedded JPEG decoder yields RGB pixels.  Applying a
                    # second DICOM YBR conversion would change the frozen recipe.
                    opened.draft("RGB", (128, 128))
                converted = opened.convert("RGB")
                _register_close(stack, converted, registered)
                return _pixels(converted, 128, stack, registered)
            array = dataset.pixel_array
            source_image = Image.fromarray(
                array if array.ndim == 3 else np.stack([array] * 3, axis=-1)
            )
            _register_close(stack, source_image, registered)
            converted = source_image.convert("RGB")
            _register_close(stack, converted, registered)
            return _pixels(converted, 128, stack, registered)
    except ContentDecodeError:
        raise
    except Exception:
        _fail()
