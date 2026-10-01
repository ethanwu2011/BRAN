"""Local-only, serial extraction for the frozen retinal and Labrador comparators.

This module intentionally has no cohort enumeration, outcome, persistence, or
reporting interface.  Its caller owns the authenticated canonical ordering of
the supplied retinal paths and patient rows, and must keep file-descriptor
silencing outside this adapter.
"""

from __future__ import annotations

import io
from collections.abc import Mapping as ABCMapping
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import patient_atlas_foundation_adapters as _foundation
import patient_atlas_retfound_green as _retfound_green
import patient_atlas_visionfm as _visionfm


BATCH_SIZE = 16
CPU_DEVICE = "cpu"
TORCH_CPU_THREADS = 2
LABRADOR_BLOOD_FEATURE_STOP = 38

RETINAL_VARIANTS: Mapping[str, Mapping[str, Any]] = {
    "retfound_green": {
        "input_size": 392,
        "embedding_dimension": 384,
        "interpolation": "bilinear_antialias",
        "artifact_keys": ("checkpoint", "checkpoint_sha256"),
    },
    "visionfm_last4": {
        "input_size": 224,
        "embedding_dimension": 3072,
        "interpolation": "bilinear_antialias",
        "artifact_keys": ("tensor_checkpoint", "architecture_file"),
    },
    "dinov3_generic": {
        "input_size": 256,
        "embedding_dimension": 384,
        "interpolation": "bicubic",
        "artifact_keys": ("checkpoint", "checkpoint_sha256"),
    },
}
LABRADOR_ARTIFACT_KEYS = (
    "model_root",
    "saved_model_sha256",
    "variables_data_sha256",
    "variables_index_sha256",
    "codebook",
    "codebook_sha256",
    "ecdf",
    "ecdf_sha256",
    "tensorflow_python",
)


class FMExtractionError(ValueError):
    """Raised without exposing patient-derived paths, rows, or values."""


def _fail(message: str) -> None:
    raise FMExtractionError(message)


def _configure_cpu_runtime() -> None:
    try:
        import torch

        torch.set_num_threads(TORCH_CPU_THREADS)
    except Exception:
        _fail("local FM CPU runtime is unavailable")


def _require_artifact_keys(
    artifact: Mapping[str, Any], required_keys: Sequence[str]
) -> None:
    if not isinstance(artifact, Mapping):
        _fail("FM artifact is malformed")
    for key in required_keys:
        value = artifact.get(key)
        if not isinstance(value, str) or not value:
            _fail("FM artifact is malformed")


def _retinal_inputs(
    paths: Sequence[str | Path], patient_rows: np.ndarray, patient_count: int
) -> tuple[tuple[Path, ...], np.ndarray]:
    if isinstance(paths, (str, bytes, ABCMapping, set, frozenset)):
        _fail("retinal paths are malformed")
    try:
        raw_paths = tuple(paths)
        canonical_paths = tuple(
            Path(path).expanduser().resolve() for path in raw_paths
        )
    except Exception:
        _fail("retinal paths are malformed")
    if not canonical_paths or len(set(canonical_paths)) != len(canonical_paths):
        _fail("retinal paths are malformed")
    rows = np.asarray(patient_rows)
    if (
        rows.ndim != 1
        or len(rows) != len(canonical_paths)
        or not np.issubdtype(rows.dtype, np.integer)
        or rows.dtype == np.dtype(bool)
        or isinstance(patient_count, bool)
        or not isinstance(patient_count, (int, np.integer))
        or int(patient_count) < 1
    ):
        _fail("retinal patient rows are malformed")
    count = int(patient_count)
    if bool((rows < 0).any()) or bool((rows >= count).any()):
        _fail("retinal patient rows are malformed")
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    if np.unique(rows).size != count:
        _fail("retinal patient rows omit a patient")
    return canonical_paths, rows


def _decode_cfp(path: Path, *, input_size: int, interpolation: str) -> np.ndarray:
    """Match the existing CFP decoder, but perform exactly one serial decode."""

    import pydicom
    from pydicom.encaps import generate_pixel_data_frame
    from PIL import Image

    dataset = pydicom.dcmread(str(path))
    if not dataset.file_meta.TransferSyntaxUID.is_encapsulated:
        pixels = dataset.pixel_array
        image = Image.fromarray(
            pixels if pixels.ndim == 3 else np.stack([pixels] * 3, axis=-1)
        ).convert("RGB")
    else:
        image = Image.open(io.BytesIO(next(generate_pixel_data_frame(dataset.PixelData))))
        if image.format == "JPEG2000":
            reduction = 0
            while min(image.size) / 2 ** (reduction + 1) >= input_size:
                reduction += 1
            image.reduce = reduction
            image.load()
        else:
            image.draft("RGB", (input_size, input_size))
        image = image.convert("RGB")
    if interpolation == "bilinear_antialias":
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as functional_v2

        resized = functional_v2.resize(
            image,
            [input_size, input_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    else:
        resized = image.resize(
            (input_size, input_size), resample=Image.Resampling.BICUBIC
        )
    values = np.asarray(resized, dtype=np.uint8)
    if values.shape != (input_size, input_size, 3):
        _fail("decoded CFP pixels are malformed")
    return values


def _load_retinal_model(variant: str, artifact: Mapping[str, Any]) -> Any:
    if variant == "retfound_green":
        return _retfound_green.load_model(
            checkpoint_path=artifact["checkpoint"],
            checkpoint_sha256=artifact["checkpoint_sha256"],
            device=CPU_DEVICE,
        )
    if variant == "visionfm_last4":
        return _visionfm.load_model(
            checkpoint_path=artifact["tensor_checkpoint"],
            architecture_path=artifact["architecture_file"],
            device=CPU_DEVICE,
        )
    return _foundation.load_dinov3_model(
        checkpoint_path=artifact["checkpoint"],
        checkpoint_sha256=artifact["checkpoint_sha256"],
        device=CPU_DEVICE,
    )


def _encode_retinal_pixels(variant: str, model: Any, pixels: np.ndarray) -> np.ndarray:
    if variant == "retfound_green":
        return _retfound_green.encode_pixels(model, pixels, device=CPU_DEVICE)
    if variant == "visionfm_last4":
        return _visionfm.encode_pixels(model, pixels, device=CPU_DEVICE)
    return _foundation.encode_dinov3_pixels(model, pixels, device=CPU_DEVICE)


def _checked_retinal_embeddings(
    embedded: Any, *, batch_count: int, embedding_dimension: int
) -> np.ndarray:
    values = np.asarray(embedded, dtype=np.float32)
    if values.shape != (batch_count, embedding_dimension):
        _fail("retinal FM embedding shape differs")
    if not np.isfinite(values).all():
        _fail("retinal FM embedding is non-finite")
    if bool(np.all(values == 0.0, axis=1).any()):
        _fail("retinal FM embedding is zero")
    return values


def extract_retinal(
    paths: Sequence[str | Path],
    patient_rows: np.ndarray,
    patient_count: int,
    *,
    variant: str,
    artifact: Mapping[str, Any],
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> np.ndarray:
    """Return one local-only, unweighted image-mean embedding per patient.

    The caller must supply the same already authenticated, canonically ordered
    paths and row map for every retinal variant.  ``progress_callback`` receives
    only ``(variant, completed_batch_count, total_batch_count)``.
    """

    spec = RETINAL_VARIANTS.get(variant)
    if spec is None:
        _fail("retinal FM variant is invalid")
    _require_artifact_keys(artifact, spec["artifact_keys"])
    selected_paths, rows = _retinal_inputs(paths, patient_rows, patient_count)
    _configure_cpu_runtime()
    try:
        model = _load_retinal_model(variant, artifact)
    except Exception:
        _fail("retinal FM model load failed")
    totals = np.zeros((int(patient_count), int(spec["embedding_dimension"])), dtype=np.float64)
    counts = np.zeros(int(patient_count), dtype=np.int64)
    total_batches = (len(selected_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    for start in range(0, len(selected_paths), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(selected_paths))
        try:
            pixels = np.stack(
                [
                    _decode_cfp(
                        path,
                        input_size=int(spec["input_size"]),
                        interpolation=str(spec["interpolation"]),
                    )
                    for path in selected_paths[start:stop]
                ],
                axis=0,
            )
        except FMExtractionError:
            raise
        except Exception:
            _fail("selected CFP decoding failed")
        try:
            embedded = _checked_retinal_embeddings(
                _encode_retinal_pixels(variant, model, pixels),
                batch_count=stop - start,
                embedding_dimension=int(spec["embedding_dimension"]),
            )
        except FMExtractionError:
            raise
        except Exception:
            _fail("retinal FM encoding failed")
        batch_rows = rows[start:stop]
        np.add.at(totals, batch_rows, embedded.astype(np.float64))
        np.add.at(counts, batch_rows, 1)
        if progress_callback is not None:
            try:
                progress_callback(
                    variant,
                    int(start // BATCH_SIZE + 1),
                    int(total_batches),
                )
            except Exception:
                _fail("retinal FM progress callback failed")
    if bool((counts == 0).any()):
        _fail("retinal patient rows omit a patient")
    pooled = (totals / counts[:, None]).astype(np.float32)
    if not np.isfinite(pooled).all():
        _fail("retinal patient pooling is non-finite")
    if bool(np.all(pooled == 0.0, axis=1).any()):
        _fail("retinal patient pooling is zero")
    return pooled


def _labrador_inputs(
    c: np.ndarray,
    cm: np.ndarray,
    elig: np.ndarray,
    names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    values = np.asarray(c)
    observed = np.asarray(cm)
    eligible = np.asarray(elig)
    try:
        ordered_names = tuple(names)
    except Exception:
        _fail("Labrador inputs are malformed")
    if (
        values.ndim != 2
        or observed.shape != values.shape
        or observed.dtype != np.bool_
        or eligible.dtype != np.bool_
        or eligible.ndim not in (1, 2)
        or (eligible.ndim == 1 and eligible.shape != (values.shape[1],))
        or (eligible.ndim == 2 and eligible.shape != values.shape)
        or len(ordered_names) != values.shape[1]
        or any(not isinstance(name, str) for name in ordered_names)
        or len(set(ordered_names)) != len(ordered_names)
    ):
        _fail("Labrador inputs are malformed")
    if eligible.ndim == 1:
        eligible = np.broadcast_to(eligible[None, :], values.shape)
    blood_only = np.arange(values.shape[1]) < LABRADOR_BLOOD_FEATURE_STOP
    eligible = np.ascontiguousarray(eligible & blood_only[None, :], dtype=bool)
    return values, observed, eligible, ordered_names


def _checked_labrador_embeddings(embedded: Any, patient_count: int) -> np.ndarray:
    values = np.asarray(embedded, dtype=np.float32)
    if values.shape != (patient_count, _foundation.LABRADOR_EMBEDDING_DIMENSION):
        _fail("Labrador embedding shape differs")
    if not np.isfinite(values).all():
        _fail("Labrador embedding is non-finite")
    return values


def extract_labrador(
    c: np.ndarray,
    cm: np.ndarray,
    elig: np.ndarray,
    names: Sequence[str],
    *,
    artifact: Mapping[str, Any],
    project_root: str | Path,
) -> np.ndarray:
    """Build the canonical blood-only Labrador input and return local embeddings."""

    _require_artifact_keys(artifact, LABRADOR_ARTIFACT_KEYS)
    values, observed, eligible, ordered_names = _labrador_inputs(c, cm, elig, names)
    try:
        inputs = _foundation.build_labrador_inputs(
            clinical_values=values,
            clinical_observed_mask=observed,
            clinical_eligible_mask=eligible,
            ordered_feature_names=ordered_names,
            codebook_path=artifact["codebook"],
            codebook_sha256=artifact["codebook_sha256"],
            ecdf_path=artifact["ecdf"],
            ecdf_sha256=artifact["ecdf_sha256"],
            max_length=_foundation.LABRADOR_MAX_LENGTH,
        )
        embedded = _foundation.encode_labrador_in_memory(
            inputs,
            python_executable=artifact["tensorflow_python"],
            worker_path=Path(project_root) / "patient_atlas_labrador_worker.py",
            model_root=artifact["model_root"],
            saved_model_sha256=artifact["saved_model_sha256"],
            variables_data_sha256=artifact["variables_data_sha256"],
            variables_index_sha256=artifact["variables_index_sha256"],
        )
    except FMExtractionError:
        raise
    except Exception:
        _fail("Labrador local extraction failed")
    return _checked_labrador_embeddings(embedded, len(values))


__all__ = [
    "BATCH_SIZE",
    "CPU_DEVICE",
    "FMExtractionError",
    "LABRADOR_ARTIFACT_KEYS",
    "LABRADOR_BLOOD_FEATURE_STOP",
    "RETINAL_VARIANTS",
    "TORCH_CPU_THREADS",
    "extract_labrador",
    "extract_retinal",
]
