"""Authenticated in-memory DICOM decoding. Never emits pixels or metadata."""
import hashlib
import io
import math
import numpy as np
import pydicom
from pydicom.encaps import generate_frames
from PIL import Image

MEAN = np.array([.485, .456, .406], np.float32)[:, None, None]
STD = np.array([.229, .224, .225], np.float32)[:, None, None]


def decode_bytes(data, expected_sha256, size=224):
    if type(data) is not bytes or hashlib.sha256(data).hexdigest() != expected_sha256 or size != 224:
        raise ValueError("retinal_decode_identity_failed")
    ds = pydicom.dcmread(io.BytesIO(data))
    if int(getattr(ds, "NumberOfFrames", 1)) != 1:
        raise ValueError("retinal_multiframe_not_supported")
    if ds.file_meta.TransferSyntaxUID.is_encapsulated:
        frames = iter(generate_frames(ds.PixelData, number_of_frames=1))
        frame = next(frames)
        if next(frames, None) is not None:
            raise ValueError("retinal_multiframe_not_supported")
        im = Image.open(io.BytesIO(frame))
        if im.format == "JPEG2000":
            reduction = 0
            while min(im.size) / 2 ** (reduction + 1) >= size:
                reduction += 1
            im.reduce = reduction
            im.load()
        else:
            im.draft("RGB", (size, size))
        # Embedded JPEG/JPEG2000 decoder already returns RGB; do not apply a
        # second conversion based on DICOM's declared YBR photometric label.
        im = im.convert("RGB")
    else:
        a = ds.pixel_array
        im = Image.fromarray(a if a.ndim == 3 else np.stack([a] * 3, -1)).convert("RGB")
    im = im.resize((size, size), Image.Resampling.BICUBIC)
    x = np.asarray(im, np.float32).transpose(2, 0, 1) / np.float32(255)
    result = (x - MEAN) / STD
    if result.shape != (3, size, size) or result.dtype != np.float32 or not np.isfinite(result).all():
        raise ValueError("retinal_decode_output_failed")
    return np.ascontiguousarray(result)
