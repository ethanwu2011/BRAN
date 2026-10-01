"""Frozen, local-only adapters for mandatory public Patient Atlas comparators.

The helpers in this module deliberately separate three concerns:

* deterministic DINOv3 pixel preprocessing and patient-level mean pooling;
* target-safe Labrador tokenisation using its released MIMIC-IV eCDFs; and
* an in-memory subprocess bridge to Labrador's isolated TensorFlow runtime.

No helper accepts outcomes or patient identifiers, and no patient-derived
embedding is written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import io
import os
from pathlib import Path
import subprocess
from typing import Iterable, Sequence

import numpy as np


DINO_MODEL_ID = "vit_small_patch16_dinov3.lvd1689m"
DINO_INPUT_SIZE = 256
DINO_EMBEDDING_DIMENSION = 384
DINO_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
DINO_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
LABRADOR_EMBEDDING_DIMENSION = 1024
LABRADOR_MAX_LENGTH = 64
LABRADOR_MAX_TOKEN = 530

# Exact, manually reviewed MIMIC item IDs used by the released Labrador
# codebook. Item IDs, not labels, disambiguate Chemistry/Hematology labs from
# blood-gas and urine rows sharing the same label. Tuple order is the
# deterministic left-pack order.
LABRADOR_ANALYTE_MAP: tuple[tuple[str, str, int], ...] = (
    ("hemoglobin", "Hemoglobin", 51222),
    ("hct", "Hematocrit", 51221),
    ("creatinine", "Creatinine", 50912),
    ("plt", "Platelet Count", 51265),
    ("wbc", "White Blood Cells", 51301),
    ("bun", "Urea Nitrogen", 51006),
    ("mchc", "MCHC", 51249),
    ("rbc", "Red Blood Cells", 51279),
    ("mcv", "MCV", 51250),
    ("mch", "MCH", 51248),
    ("rdw", "RDW", 51277),
    ("potassium", "Potassium", 50971),
    ("sodium", "Sodium", 50983),
    ("chloride", "Chloride", 50902),
    ("carbon_dioxide_total", "Bicarbonate", 50882),
    ("glucose", "Glucose", 50931),
    ("calcium", "Calcium, Total", 50893),
    ("alt_got", "Alanine Aminotransferase (ALT)", 50861),
    ("ast_got", "Asparate Aminotransferase (AST)", 50878),
    ("bilirubin_total", "Bilirubin, Total", 50885),
    ("alkaline_phosphatase", "Alkaline Phosphatase", 50863),
    ("albumin", "Albumin", 50862),
    ("protein_total", "Protein, Total", 50976),
    ("globulin_total", "Globulin", 50930),
    ("total_cholesterol", "Cholesterol, Total", 50907),
    ("hdl_cholesterol", "Cholesterol, HDL", 50904),
    ("ldl_cholesterol", "Cholesterol, LDL, Calculated", 50905),
    ("triglycerides", "Triglycerides", 51000),
    ("hba1c", "% Hemoglobin A1c", 50852),
    ("troponin_t", "Troponin T", 51003),
    ("nt_probnp", "NTproBNP", 50963),
    ("crp_hs", "C-Reactive Protein", 50889),
    ("urine_albumin", "Albumin, Urine", 51069),
    ("urine_creatinine", "Creatinine, Urine", 51082),
)


class FoundationAdapterError(ValueError):
    """Raised when a frozen comparator cannot be used without ambiguity."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def require_artifact(path: str | Path, expected_sha256: str) -> Path:
    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise FoundationAdapterError("required comparator artifact is absent")
    if (
        len(expected_sha256) != 64
        or any(value not in "0123456789abcdef" for value in expected_sha256)
        or sha256_file(artifact) != expected_sha256
    ):
        raise FoundationAdapterError("comparator artifact hash differs")
    return artifact


def normalize_dinov3_pixels(images: np.ndarray) -> np.ndarray:
    """Apply the frozen timm DINOv3 deterministic preprocessing to RGB pixels."""

    values = np.asarray(images)
    if (
        values.ndim != 4
        or values.shape[1:] != (DINO_INPUT_SIZE, DINO_INPUT_SIZE, 3)
        or values.dtype != np.uint8
    ):
        raise FoundationAdapterError(
            "DINOv3 pixels must be uint8 [images,256,256,3] RGB"
        )
    normalized = values.astype(np.float32) / np.float32(255.0)
    normalized = (normalized - DINO_MEAN[None, None, None]) / DINO_STD[
        None, None, None
    ]
    normalized = np.ascontiguousarray(normalized.transpose(0, 3, 1, 2))
    if not np.isfinite(normalized).all():
        raise FoundationAdapterError("DINOv3 preprocessing produced non-finite pixels")
    return normalized


def load_dinov3_model(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    device: str = "cpu",
):
    """Load the generic DINOv3 encoder from one authenticated local checkpoint."""

    # Authenticate the resolved bytes, but retain the user-facing .safetensors
    # symlink for timm's format dispatch. Passing the extensionless resolved HF
    # blob makes timm misclassify it as a pickle checkpoint.
    require_artifact(checkpoint_path, checkpoint_sha256)
    artifact = Path(checkpoint_path).expanduser().absolute()
    try:
        import timm
        import torch
    except ImportError as error:  # pragma: no cover - runtime-specific
        raise FoundationAdapterError("DINOv3 runtime dependencies are unavailable") from error
    if device not in {"cpu", "mps", "cuda"}:
        raise FoundationAdapterError("DINOv3 device is invalid")
    previous_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        model = timm.create_model(
            DINO_MODEL_ID,
            pretrained=False,
            num_classes=0,
            img_size=DINO_INPUT_SIZE,
            checkpoint_path=str(artifact),
        )
    finally:
        if previous_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_offline
    if int(getattr(model, "num_features", -1)) != DINO_EMBEDDING_DIMENSION:
        raise FoundationAdapterError("DINOv3 embedding dimension differs")
    model = model.to(torch.device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode_dinov3_pixels(model, images: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    """Encode one already-resized batch without augmentations or gradient state."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime-specific
        raise FoundationAdapterError("PyTorch is unavailable") from error
    batch = torch.from_numpy(normalize_dinov3_pixels(images)).to(torch.device(device))
    with torch.inference_mode():
        encoded = model(batch)
    if isinstance(encoded, (tuple, list)):
        encoded = encoded[0]
    values = np.asarray(encoded.detach().cpu().numpy(), dtype=np.float32)
    if values.shape != (len(images), DINO_EMBEDDING_DIMENSION):
        raise FoundationAdapterError("DINOv3 returned an unexpected shape")
    if not np.isfinite(values).all():
        raise FoundationAdapterError("DINOv3 returned non-finite embeddings")
    return values


def pool_patient_image_embeddings(
    image_embeddings: np.ndarray,
    patient_rows: np.ndarray,
    *,
    patient_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unweighted patient mean over images, with explicit absent-patient state."""

    values = np.asarray(image_embeddings, dtype=np.float32)
    rows = np.asarray(patient_rows)
    if values.ndim != 2 or values.shape[0] != len(rows) or values.shape[1] < 1:
        raise FoundationAdapterError("image embeddings are malformed")
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise FoundationAdapterError("patient row mapping is malformed")
    if patient_count < 1 or bool((rows < 0).any()) or bool((rows >= patient_count).any()):
        raise FoundationAdapterError("patient row mapping is out of bounds")
    if not np.isfinite(values).all():
        raise FoundationAdapterError("image embeddings contain non-finite values")
    totals = np.zeros((patient_count, values.shape[1]), dtype=np.float64)
    counts = np.zeros(patient_count, dtype=np.int64)
    np.add.at(totals, rows, values.astype(np.float64))
    np.add.at(counts, rows, 1)
    present = counts > 0
    pooled = np.zeros_like(totals, dtype=np.float32)
    pooled[present] = (totals[present] / counts[present, None]).astype(np.float32)
    return pooled, present, counts


@dataclass(frozen=True)
class LabradorInputs:
    categorical: np.ndarray
    continuous: np.ndarray
    mapped_features: tuple[str, ...]


def _load_labrador_codebook(path: Path) -> dict[int, tuple[int, str]]:
    output: dict[int, tuple[int, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {
            "itemid",
            "frequency_rank",
            "label",
        }.issubset(reader.fieldnames):
            raise FoundationAdapterError("Labrador codebook schema differs")
        for row in reader:
            itemid = int(row["itemid"])
            if itemid in output:
                raise FoundationAdapterError("Labrador codebook item IDs are not unique")
            output[itemid] = (int(row["frequency_rank"]), str(row["label"]).strip())
    return output


def build_labrador_inputs(
    *,
    clinical_values: np.ndarray,
    clinical_observed_mask: np.ndarray,
    clinical_eligible_mask: np.ndarray,
    ordered_feature_names: Sequence[str],
    codebook_path: str | Path,
    codebook_sha256: str,
    ecdf_path: str | Path,
    ecdf_sha256: str,
    max_length: int = LABRADOR_MAX_LENGTH,
) -> LabradorInputs:
    """Map policy-visible raw labs to Labrador tokens and MIMIC eCDF values."""

    values = np.asarray(clinical_values, dtype=np.float64)
    observed = np.asarray(clinical_observed_mask)
    eligible = np.asarray(clinical_eligible_mask)
    names = tuple(str(value) for value in ordered_feature_names)
    if values.ndim != 2 or observed.shape != values.shape:
        raise FoundationAdapterError("clinical Labrador arrays are malformed")
    if observed.dtype != np.bool_:
        raise FoundationAdapterError("clinical observed mask must be boolean")
    if eligible.ndim == 1:
        eligible = np.broadcast_to(eligible[None], values.shape)
    if eligible.shape != values.shape or eligible.dtype != np.bool_:
        raise FoundationAdapterError("clinical eligible mask is malformed")
    if len(names) != values.shape[1] or len(set(names)) != len(names):
        raise FoundationAdapterError("ordered clinical feature names differ")
    if max_length < 1 or max_length > LABRADOR_MAX_LENGTH:
        raise FoundationAdapterError("Labrador maximum sequence length differs")
    visible = observed & eligible
    if not np.isfinite(values[visible]).all():
        raise FoundationAdapterError("visible Labrador values are non-finite")

    codebook = require_artifact(codebook_path, codebook_sha256)
    ecdf_artifact = require_artifact(ecdf_path, ecdf_sha256)
    rows = _load_labrador_codebook(codebook)
    feature_index = {name: index for index, name in enumerate(names)}
    specifications: list[tuple[str, int, int, int]] = []
    with np.load(ecdf_artifact, allow_pickle=False) as ecdfs:
        for feature, label, itemid in LABRADOR_ANALYTE_MAP:
            if feature not in feature_index or itemid not in rows:
                continue
            token, observed_label = rows[itemid]
            if observed_label != label:
                raise FoundationAdapterError("Labrador item label differs from frozen map")
            if not 1 <= token <= LABRADOR_MAX_TOKEN:
                raise FoundationAdapterError("Labrador token is outside the embedding table")
            if f"{itemid}_x" not in ecdfs or f"{itemid}_y" not in ecdfs:
                continue
            specifications.append((feature, feature_index[feature], token, itemid))
        if not specifications:
            raise FoundationAdapterError("no policy-compatible Labrador analytes map")

        categorical = np.zeros((len(values), max_length), dtype=np.int32)
        continuous = np.zeros((len(values), max_length), dtype=np.float32)
        for patient in range(len(values)):
            position = 0
            for _feature, column, token, itemid in specifications:
                if position >= max_length:
                    break
                if not visible[patient, column]:
                    continue
                xs = np.asarray(ecdfs[f"{itemid}_x"])
                ys = np.asarray(ecdfs[f"{itemid}_y"])
                if (
                    xs.ndim != 1
                    or ys.shape != xs.shape
                    or len(xs) < 1
                    or np.isnan(xs).any()
                    or np.isposinf(xs).any()
                    or not np.isfinite(ys).all()
                    or bool((np.diff(xs) < 0).any())
                ):
                    raise FoundationAdapterError("Labrador eCDF is malformed")
                index = int(
                    np.clip(
                        np.searchsorted(xs, values[patient, column]),
                        0,
                        len(ys) - 1,
                    )
                )
                categorical[patient, position] = token
                continuous[patient, position] = np.float32(ys[index])
                position += 1
    return LabradorInputs(
        categorical=categorical,
        continuous=continuous,
        mapped_features=tuple(value[0] for value in specifications),
    )


def encode_labrador_in_memory(
    inputs: LabradorInputs,
    *,
    python_executable: str | Path,
    worker_path: str | Path,
    model_root: str | Path,
    saved_model_sha256: str,
    variables_data_sha256: str,
    variables_index_sha256: str,
    timeout_seconds: int = 900,
) -> np.ndarray:
    """Encode Labrador through a binary pipe; never serialize patient embeddings."""

    categorical = np.asarray(inputs.categorical)
    continuous = np.asarray(inputs.continuous)
    if (
        categorical.ndim != 2
        or categorical.shape != continuous.shape
        or categorical.dtype != np.int32
        or continuous.dtype != np.float32
        or categorical.shape[1] != LABRADOR_MAX_LENGTH
        or bool((categorical < 0).any())
        or bool((categorical > LABRADOR_MAX_TOKEN).any())
        or not np.isfinite(continuous).all()
    ):
        raise FoundationAdapterError("Labrador subprocess inputs are malformed")
    # Preserve a virtual-environment interpreter symlink. Resolving it to the
    # base executable silently drops that environment's installed packages.
    python_path = Path(python_executable).expanduser().absolute()
    worker = Path(worker_path).resolve()
    root = Path(model_root).resolve()
    if not python_path.is_file() or not worker.is_file() or not root.is_dir():
        raise FoundationAdapterError("Labrador runtime path is absent")
    buffer = io.BytesIO()
    np.savez(buffer, categorical=categorical, continuous=continuous)
    command = [
        str(python_path),
        str(worker),
        "--model-root",
        str(root),
        "--saved-model-sha256",
        saved_model_sha256,
        "--variables-data-sha256",
        variables_data_sha256,
        "--variables-index-sha256",
        variables_index_sha256,
    ]
    environment = dict(os.environ)
    environment["TF_CPP_MIN_LOG_LEVEL"] = "3"
    environment["TF_USE_LEGACY_KERAS"] = "1"
    try:
        completed = subprocess.run(
            command,
            input=buffer.getvalue(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise FoundationAdapterError("Labrador local worker did not complete") from error
    if completed.returncode != 0:
        # Deliberately do not relay worker stderr: fail closed if a dependency ever
        # includes patient-derived content in an exception.
        raise FoundationAdapterError("Labrador local worker failed")
    if len(completed.stdout) > max(1 << 20, len(categorical) * 5000):
        raise FoundationAdapterError("Labrador worker output exceeds its bound")
    try:
        embedded = np.load(io.BytesIO(completed.stdout), allow_pickle=False)
    except Exception as error:
        raise FoundationAdapterError("Labrador worker output is not a safe NPY array") from error
    values = np.asarray(embedded, dtype=np.float32)
    if values.shape != (len(categorical), LABRADOR_EMBEDDING_DIMENSION):
        raise FoundationAdapterError("Labrador embedding shape differs")
    if not np.isfinite(values).all():
        raise FoundationAdapterError("Labrador embedding contains non-finite values")
    return values


__all__ = [
    "DINO_EMBEDDING_DIMENSION",
    "DINO_INPUT_SIZE",
    "DINO_MEAN",
    "DINO_MODEL_ID",
    "DINO_STD",
    "FoundationAdapterError",
    "LABRADOR_ANALYTE_MAP",
    "LABRADOR_EMBEDDING_DIMENSION",
    "LABRADOR_MAX_LENGTH",
    "LabradorInputs",
    "build_labrador_inputs",
    "encode_dinov3_pixels",
    "encode_labrador_in_memory",
    "load_dinov3_model",
    "normalize_dinov3_pixels",
    "pool_patient_image_embeddings",
    "require_artifact",
    "sha256_file",
]
