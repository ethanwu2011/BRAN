"""Authenticated, in-memory DICOM decoding with explicit resource cleanup.

This module intentionally keeps the V1 pixel operations unchanged.  The V2
surface adds fixed, disclosure-safe failure metadata and closes every PIL and
``BytesIO`` intermediate on both successful and failed decodes.
"""

from contextlib import ExitStack
import hashlib
import io

import numpy as np
import pydicom
from pydicom.encaps import generate_frames
from PIL import Image


MEAN = np.array([.485, .456, .406], np.float32)[:, None, None]
STD = np.array([.229, .224, .225], np.float32)[:, None, None]

# These sets are intentionally closed.  They are the only values that may be
# returned in SafeDecodeFailure.safe_metadata().
STAGES = frozenset(("identity", "dicom", "frame", "pixel", "rgb", "resize", "normalize", "cleanup"))
CATEGORIES = frozenset(("validation", "memory", "io", "decoder", "other"))

__all__ = [
    "CATEGORIES",
    "MEAN",
    "STD",
    "STAGES",
    "SafeDecodeFailure",
    "decode_bytes",
]


class SafeDecodeFailure(ValueError):
    """A decode failure whose public fields contain only fixed safe values."""

    __slots__ = ("_stage", "_category")

    def __init__(self, stage, category):
        if stage not in STAGES or category not in CATEGORIES:
            raise ValueError("retinal_decode_failure_metadata_invalid")
        self._stage = stage
        self._category = category
        # Do not include input data or the underlying exception in the public
        # message.  Stage and category are available through safe_metadata().
        super().__init__("retinal_decode_failed")

    def safe_metadata(self):
        """Return a fresh, closed two-key dictionary of safe failure data."""

        return {"stage": self._stage, "category": self._category}

    def __repr__(self):
        # Keep repr deterministic and limited to the fixed allowlists.
        return "SafeDecodeFailure(stage={!r}, category={!r})".format(
            self._stage, self._category
        )


def _category_for(exc, fallback):
    """Map an internal exception to a fixed category without inspecting text."""

    if isinstance(exc, MemoryError):
        return "memory"
    if isinstance(exc, (OSError, EOFError, io.UnsupportedOperation)):
        return "io"
    return fallback


def _raise_failure(stage, category):
    # The explicit ``from None`` is part of the public-safety boundary: the
    # underlying exception and its text are not exposed as a chained cause.
    raise SafeDecodeFailure(stage, category) from None


def _safe_close(resource):
    """Close one intermediate while preventing cleanup text from escaping."""

    try:
        resource.close()
    except Exception as exc:
        # ExitStack still attempts the remaining callbacks. A cleanup failure
        # must not silently return a successful decode or expose library text.
        _raise_failure("cleanup", _category_for(exc, "other"))


def _register_close(stack, resource, registered):
    """Register a resource once, preserving cleanup for aliased PIL images."""

    if resource is None:
        return resource
    key = id(resource)
    if key not in registered:
        registered.add(key)
        stack.callback(_safe_close, resource)
    return resource


def _decode_impl(data, size, stack):
    """Decode after identity validation, with stage-local safe error mapping."""

    registered = set()

    try:
        source = io.BytesIO(data)
        stack.callback(_safe_close, source)
        ds = pydicom.dcmread(source)
    except SafeDecodeFailure:
        raise
    except Exception as exc:
        _raise_failure("dicom", _category_for(exc, "decoder"))

    try:
        number_of_frames = int(getattr(ds, "NumberOfFrames", 1))
    except Exception as exc:
        _raise_failure("frame", _category_for(exc, "validation"))
    if number_of_frames != 1:
        _raise_failure("frame", "validation")

    try:
        is_encapsulated = bool(ds.file_meta.TransferSyntaxUID.is_encapsulated)
    except Exception as exc:
        _raise_failure("dicom", _category_for(exc, "validation"))

    if is_encapsulated:
        try:
            frames = generate_frames(ds.PixelData, number_of_frames=1)
            close_frames = getattr(frames, "close", None)
            if close_frames is not None:
                stack.callback(_safe_close, frames)
            frame = next(frames)
            if next(frames, None) is not None:
                _raise_failure("frame", "validation")
        except SafeDecodeFailure:
            raise
        except Exception as exc:
            _raise_failure("frame", _category_for(exc, "decoder"))

        try:
            encoded = io.BytesIO(frame)
            stack.callback(_safe_close, encoded)
            opened = Image.open(encoded)
            _register_close(stack, opened, registered)
        except Exception as exc:
            _raise_failure("rgb", _category_for(exc, "decoder"))

        try:
            if opened.format == "JPEG2000":
                reduction = 0
                while min(opened.size) / 2 ** (reduction + 1) >= size:
                    reduction += 1
                # Match V1's JPEG2000 reduction hint exactly.
                opened.reduce = reduction
                opened.load()
            else:
                # Match V1's JPEG draft hint exactly.  The embedded decoder
                # already returns RGB; do not apply DICOM YBR conversion.
                opened.draft("RGB", (size, size))
            converted = opened.convert("RGB")
            _register_close(stack, converted, registered)
            image = converted
        except Exception as exc:
            _raise_failure("rgb", _category_for(exc, "decoder"))
    else:
        try:
            array = ds.pixel_array
        except Exception as exc:
            _raise_failure("pixel", _category_for(exc, "decoder"))

        try:
            source_image = Image.fromarray(
                array if array.ndim == 3 else np.stack([array] * 3, -1)
            )
            _register_close(stack, source_image, registered)
            converted = source_image.convert("RGB")
            _register_close(stack, converted, registered)
            image = converted
        except Exception as exc:
            _raise_failure("rgb", _category_for(exc, "decoder"))

    try:
        resized = image.resize((size, size), Image.Resampling.BICUBIC)
        _register_close(stack, resized, registered)
    except Exception as exc:
        _raise_failure("resize", _category_for(exc, "decoder"))

    try:
        # Keep V1's dtype, channel order, resize result, and normalization
        # operations byte-for-byte/element-for-element equivalent.
        x = np.asarray(resized, np.float32).transpose(2, 0, 1) / np.float32(255)
        result = (x - MEAN) / STD
        if (
            result.shape != (3, size, size)
            or result.dtype != np.float32
            or not np.isfinite(result).all()
        ):
            _raise_failure("normalize", "other")
        return np.ascontiguousarray(result)
    except SafeDecodeFailure:
        raise
    except Exception as exc:
        _raise_failure("normalize", _category_for(exc, "other"))


def decode_bytes(data: bytes, expected_sha256: str, size=224):
    """Decode one authenticated DICOM image into a contiguous float32 tensor."""

    try:
        if (
            type(data) is not bytes
            or hashlib.sha256(data).hexdigest() != expected_sha256
            or size != 224
        ):
            _raise_failure("identity", "validation")
    except SafeDecodeFailure:
        raise
    except MemoryError:
        _raise_failure("identity", "memory")
    except Exception:
        _raise_failure("identity", "validation")

    try:
        with ExitStack() as stack:
            return _decode_impl(data, size, stack)
    except SafeDecodeFailure:
        raise
    except MemoryError:
        _raise_failure("normalize", "memory")
    except Exception:
        _raise_failure("normalize", "other")
