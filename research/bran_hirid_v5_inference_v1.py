"""Pure frozen-model HiRID V5 native Hb inference bridge.

The caller supplies an already-authenticated adapter batch, fold transform, and
already-loaded frozen BRAN model.  This module performs no source or
checkpoint I/O, fitting, calibration, uncertainty estimation, or output
emission.  Only the native Hb head is inverse-transformed and retained in the
local private result.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np
import torch

from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_hirid_v5_input_adapter_v1 import HiRIDV5InputBatch
from bran_knhanes_input_kernel_v1 import (
    ADMITTED_CANONICAL_INDICES,
    CANONICAL_INDEX,
    CANONICAL_NAMES,
)
from bran_multisource_age_v2 import UNKNOWN, validate_age
from bran_multisource_batches_v2 import tensor, transform_hash
from bran_multisource_data_v2 import FoldTransformV2
from bran_multisource_inference_v2 import _model_ok, infer_native
from bran_multisource_model_v2 import BRANMultisourceModelV2


ERROR = "hirid_v5_inference_contract_failed"
CLINICAL_WIDTH = 59
RETINAL_FEATURE_DIM = 384
STATE_DIM = 192
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CBC_INDICES = tuple(CANONICAL_INDEX[field] for field in CBC_FIELDS)
_HB_POSITION = CBC_FIELDS.index("hemoglobin")
_EXPECTED_ELIGIBLE = np.zeros(CLINICAL_WIDTH, dtype=bool)
_EXPECTED_ELIGIBLE[list(ADMITTED_CANONICAL_INDICES)] = True


@dataclass(frozen=True, slots=True, repr=False)
class PrivateHiRIDV5Hb:
    """Local original-unit Hb values and availability; abstentions stay NaN."""

    native_hemoglobin: np.ndarray
    available: np.ndarray

    def __repr__(self) -> str:
        return "<PrivateHiRIDV5Hb>"

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)

    @property
    def abstained(self) -> np.ndarray:
        result = np.logical_not(self.available)
        result.setflags(write=False)
        return result


class _Invalid(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _Invalid


def _finite_scalar(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise _Invalid
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise _Invalid from None
    if not math.isfinite(number):
        raise _Invalid
    return number


def _valid_hash(value: object) -> None:
    _require(type(value) is str and _HASH_RE.fullmatch(value) is not None)


def _valid_float_array(value: object, shape: tuple[int, ...]) -> np.ndarray:
    _require(type(value) is np.ndarray and value.shape == shape and value.dtype.kind == "f")
    _require(np.isfinite(value).all())
    return value


def _validate_model(model: object) -> None:
    _require(isinstance(model, BRANMultisourceModelV2))
    _model_ok(model)
    _require(model.training is False)
    parameters = tuple(model.parameters())
    _require(bool(parameters))
    _require(all(not parameter.requires_grad for parameter in parameters))
    _require(all(parameter.device.type == "cpu" and parameter.dtype == torch.float32
                 for parameter in parameters))
    _require(all(buffer.device.type == "cpu" for buffer in model.buffers()))
    _require(tuple(model.cbc_indices) == _CBC_INDICES)
    _require(np.array_equal(model.eligible_slots.detach().cpu().numpy(), _EXPECTED_ELIGIBLE))
    _require(model.cbc_joint_head.in_features == STATE_DIM)
    _require(model.cbc_joint_head.out_features == len(CBC_FIELDS))


def _validate_transform(transform: object, model: BRANMultisourceModelV2,
                        expected_transform_sha256: object) -> None:
    _require(type(transform) is FoldTransformV2)
    _valid_hash(expected_transform_sha256)
    _valid_float_array(transform.clinical_median, (CLINICAL_WIDTH,))
    _valid_float_array(transform.clinical_iqr, (CLINICAL_WIDTH,))
    _valid_float_array(transform.retinal_mean, (RETINAL_FEATURE_DIM,))
    _valid_float_array(transform.retinal_scale, (RETINAL_FEATURE_DIM,))
    _require((transform.clinical_iqr > 0.0).all())
    _require((transform.retinal_scale > 0.0).all())
    _finite_scalar(transform.age_mean)
    _require(_finite_scalar(transform.age_scale) > 0.0)
    _require(type(transform.eligible) is np.ndarray and transform.eligible.shape == (CLINICAL_WIDTH,)
             and transform.eligible.dtype == np.bool_)
    _require(np.array_equal(transform.eligible, _EXPECTED_ELIGIBLE))
    _require(np.array_equal(model.eligible_slots.detach().cpu().numpy(), transform.eligible))
    _require(type(transform.heldout_fold) is int and not isinstance(transform.heldout_fold, bool)
             and transform.heldout_fold == 0)
    _valid_hash(transform.fold_identity_sha256)
    _valid_hash(transform.training_indices_sha256)
    _require(transform_hash(transform) == expected_transform_sha256)


def _validate_batch(batch: object) -> tuple[int, HiRIDV5InputBatch]:
    _require(type(batch) is HiRIDV5InputBatch)
    clinical_values = batch.clinical_values
    clinical_mask = batch.clinical_mask
    retinal_features = batch.retinal_features
    retinal_mask = batch.retinal_mask
    clinical_eligible = batch.clinical_eligible
    _require(type(clinical_values) is np.ndarray and clinical_values.ndim == 2
             and clinical_values.shape[1] == CLINICAL_WIDTH and clinical_values.dtype.kind == "f")
    n_rows = clinical_values.shape[0]
    _require(type(clinical_mask) is np.ndarray and clinical_mask.dtype == np.bool_
             and clinical_mask.shape == clinical_values.shape)
    _require(type(retinal_features) is np.ndarray and retinal_features.shape == (n_rows, RETINAL_FEATURE_DIM)
             and retinal_features.dtype.kind == "f")
    _require(type(retinal_mask) is np.ndarray and retinal_mask.dtype == np.bool_
             and retinal_mask.shape == (n_rows,) and not retinal_mask.any())
    _require(type(clinical_eligible) is np.ndarray and clinical_eligible.dtype == np.bool_
             and clinical_eligible.shape == (n_rows,))

    allowed = np.zeros(CLINICAL_WIDTH, dtype=bool)
    allowed[list(CANONICAL_INDEX[field] for field in (
        "potassium", "sodium", "chloride", "creatinine", "bilirubin_total", "albumin", "glucose"
    ))] = True
    _require(not clinical_mask[:, ~allowed].any())
    _require(np.isfinite(clinical_values[clinical_mask]).all())
    _require(np.array_equal(clinical_eligible, np.any(clinical_mask[:, allowed], axis=1)))

    age = batch.ages
    validate_age(age)
    _require(age.value.shape == (n_rows,) and age.value.device.type == "cpu"
             and age.lower.device.type == "cpu" and age.upper.device.type == "cpu"
             and age.kind.device.type == "cpu")
    _require(age.value.dtype == torch.float32 and age.lower.dtype == torch.float32
             and age.upper.dtype == torch.float32 and age.kind.dtype == torch.long)
    _require(bool(torch.all(age.kind == UNKNOWN)))
    _require(bool(torch.isnan(age.value).all()) and bool(torch.isnan(age.lower).all())
             and bool(torch.isnan(age.upper).all()))
    return n_rows, batch


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def infer(batch: HiRIDV5InputBatch, model: BRANMultisourceModelV2,
          transform: FoldTransformV2, expected_transform_sha256: str,
          *, batch_size: int = 256) -> PrivateHiRIDV5Hb:
    """Return local original-unit native Hb predictions with abstention.

    The caller remains responsible for authenticating the transform and
    checkpoint before passing them here.  Hidden clinical and retinal payloads
    are zeroed before the fold transform; the selector target and anchor are
    never read.  No uncertainty or calibration is inferred.
    """
    try:
        _validate_model(model)
        _validate_transform(transform, model, expected_transform_sha256)
        n_rows, batch = _validate_batch(batch)
        _require(type(batch_size) is int and not isinstance(batch_size, bool) and batch_size >= 1)

        # Only observed clinical cells are allowed to reach the normalizer.
        # Hidden target/CBC payloads may be NaN or arbitrary and are replaced
        # before any transform arithmetic.
        clinical_values = np.where(batch.clinical_mask, batch.clinical_values, 0.0)
        clinical_values = np.asarray(clinical_values, dtype=np.float64)
        clinical, clinical_mask = transform.clinical(clinical_values, batch.clinical_mask)

        # HiRID is clinical-only.  Retinal payload is zero/false even if a
        # malformed caller supplied hidden values in the adapter container.
        retinal_values = np.zeros((n_rows, RETINAL_FEATURE_DIM), dtype=np.float64)
        retinal_observed = np.zeros(n_rows, dtype=np.bool_)
        retinal, retinal_mask = transform.retinal(retinal_values, retinal_observed)

        native = np.full(n_rows, np.nan, dtype=np.float64)
        available = np.zeros(n_rows, dtype=np.bool_)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if n_rows:
            result = infer_native(
                model,
                tensor(clinical),
                tensor(clinical_mask, torch.bool),
                tensor(retinal),
                tensor(retinal_mask, torch.bool),
                batch.ages,
                transform.age_mean,
                transform.age_scale,
                batch_size,
            )
            standardized = result.cbc_standardized.detach().cpu().numpy()
            model_available = (~result.abstained.detach().cpu().numpy().astype(bool))
            available = model_available & batch.clinical_eligible
            _require(standardized.shape == (n_rows, len(CBC_FIELDS)))
            hb = standardized[:, _HB_POSITION]
            _require(np.isfinite(hb[available]).all())
            hb_slot = CANONICAL_INDEX["hemoglobin"]
            native[available] = (hb[available] * transform.clinical_iqr[hb_slot]
                                 + transform.clinical_median[hb_slot])
            _require(np.isfinite(native[available]).all())

        _require(not model.training and all(not parameter.requires_grad for parameter in model.parameters()))
        _require(all(torch.equal(value, model.state_dict()[name]) for name, value in before.items()))
        _require(transform_hash(transform) == expected_transform_sha256)
        return PrivateHiRIDV5Hb(_readonly_copy(native), _readonly_copy(available))
    except _Invalid:
        raise ValueError(ERROR) from None
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


__all__ = ["ERROR", "PrivateHiRIDV5Hb", "infer"]
