"""Strict R7 adapter for the frozen Q1 quantile numerical kernels.

The Q1 training and evaluation modules contain the numerical recipe.  This
module only supplies the R7 identity boundary: an exact
``BRANRobustClinicalR7`` teacher is required, and evaluation providers are
validated against the fixed R7 fold transform before the unchanged numerical
kernel is called.  No source, checkpoint, or artifact I/O belongs here.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import torch

import bran_r7_fixed_state_p1 as p1
from bran_robust_clinical_r7 import BRANRobustClinicalR7
import bran_v5_quantile_evaluation as _numerical_evaluation
import bran_v5_quantile_training as _numerical_training
from bran_v5_quantile_cbc import QuantileCBC


ERROR = "r7_quantile_q4_contract_failed"
SCHEMA = "bran-r7-quantile-q4-evaluation-v1"
NUMERICAL_SCHEMA = "bran-v5-quantile-evaluation-v1"
NUMERICAL_IMPLEMENTATION_IDENTITY = NUMERICAL_SCHEMA
ENCODER_IDENTITY = "BRANRobustClinicalR7"
ENCODER_VERSION = 7

# Keep the frozen Q1 schedule and contexts visible at this boundary.  These
# names are aliases only; no recipe, architecture, or optimizer is changed.
PATTERNS = _numerical_training.PATTERNS
WHOLE = _numerical_evaluation.WHOLE


def _fail() -> None:
    raise ValueError(ERROR) from None


def _require(condition: bool) -> None:
    if not condition:
        _fail()


def _validate_r7_teacher(teacher: object) -> None:
    """Require the exact frozen R7 class, not merely a compatible base class."""

    _require(type(teacher) is BRANRobustClinicalR7)
    _require(getattr(teacher, "arm", None) == "mlp")
    _require(not teacher.training)
    _require(all(not parameter.requires_grad for parameter in teacher.parameters()))


def _validate_r7_quantile_pair(teacher: object, head: object) -> None:
    _validate_r7_teacher(teacher)
    _require(isinstance(head, QuantileCBC))
    # The generic numerical kernel owns the exact native-head equality check;
    # invoke its closed validator here so the adapter cannot widen that bind.
    try:
        _numerical_training._quantile_ok(teacher, head)
    except Exception:
        _fail()


def _slots_and_transforms(paired: object) -> tuple[tuple[int, ...], tuple[object, ...]]:
    try:
        names = tuple(getattr(paired, "names"))
        slots = tuple(names.index(field) for field in _numerical_evaluation.CBC_FIELDS)
        transforms = tuple(getattr(paired, "transforms"))
    except Exception:
        _fail()
    _require(len(slots) == 9 and len(set(slots)) == 9 and len(transforms) == 5)
    return slots, transforms


def train_quantile(teacher: BRANRobustClinicalR7, clinical: torch.Tensor,
                   cm: torch.Tensor, retinal: torch.Tensor, rm: torch.Tensor,
                   age: Any, age_mean: float, age_scale: float,
                   cbc_indices: Iterable[int], pool_fold_ids: torch.Tensor,
                   heldout_fold: int, updates: int = 1500,
                   batch_size: int = 96):
    """Fit only the unchanged Q1 attachment on an R7 frozen teacher."""

    _validate_r7_teacher(teacher)
    return _numerical_training.train_quantile(
        teacher, clinical, cm, retinal, rm, age, age_mean, age_scale,
        cbc_indices, pool_fold_ids, heldout_fold, updates, batch_size,
    )


def predict_quantiles(teacher: BRANRobustClinicalR7, head: QuantileCBC,
                      clinical: torch.Tensor, cm: torch.Tensor,
                      retinal: torch.Tensor, rm: torch.Tensor, age: Any,
                      age_mean: float, age_scale: float, pattern: str,
                      cbc_indices: Iterable[int], batch_size: int = 256):
    """Run target-erased Q1 inference with an exact R7 teacher identity."""

    _validate_r7_quantile_pair(teacher, head)
    return _numerical_training.predict_quantiles(
        teacher, head, clinical, cm, retinal, rm, age, age_mean,
        age_scale, pattern, cbc_indices, batch_size,
    )


def _provider_for_r7(paired: object, provider: Callable[[int], tuple[object, object, object]]):
    """Wrap a provider so every inference/reload call is R7-bound."""

    slots, inherited = _slots_and_transforms(paired)

    def wrapped(fold: int):
        try:
            value = provider(fold)
            if not isinstance(value, tuple) or len(value) != 3:
                _fail()
            teacher, transform, head = value
            # This is the authoritative R7 fold/transform validator from P1.
            p1._validate_r7(teacher, transform, fold, slots, inherited[fold])
            _validate_r7_quantile_pair(teacher, head)
            return teacher, transform, head
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail()
        raise AssertionError("unreachable")

    return wrapped


def _wrap_result(numerical: dict[str, object]) -> dict[str, object]:
    """Close the numerical result under an explicit R7-Q4 identity."""

    try:
        _numerical_evaluation.validate_result(numerical)
    except Exception:
        _fail()
    result = {
        "schema": SCHEMA,
        "numerical_result_schema": numerical["schema"],
        "numerical_implementation_identity": NUMERICAL_IMPLEMENTATION_IDENTITY,
        "numerical_schema_is_encoder_identity": False,
        "encoder_identity": ENCODER_IDENTITY,
        "encoder_version": ENCODER_VERSION,
        "r7_provider_validation": True,
        "result": numerical,
        "encoder_updated": False,
        "native_heads_changed": False,
        "protected_external_data_used": False,
        "candidate_promoted": False,
        "clinical_use_established": False,
        "patient_level_output_emitted": False,
    }
    validate_result(result)
    return result


def validate_result(value: object) -> None:
    """Validate the closed aggregate-only R7-Q4 wrapper schema."""

    try:
        _require(isinstance(value, dict))
        expected = {
            "schema", "numerical_result_schema", "numerical_implementation_identity",
            "numerical_schema_is_encoder_identity", "encoder_identity", "encoder_version",
            "r7_provider_validation", "result", "encoder_updated",
            "native_heads_changed", "protected_external_data_used", "candidate_promoted",
            "clinical_use_established", "patient_level_output_emitted",
        }
        _require(set(value) == expected)
        _require(value["schema"] == SCHEMA)
        _require(value["numerical_result_schema"] == NUMERICAL_SCHEMA)
        _require(value["numerical_implementation_identity"] == NUMERICAL_IMPLEMENTATION_IDENTITY)
        _require(value["numerical_schema_is_encoder_identity"] is False)
        _require(value["encoder_identity"] == ENCODER_IDENTITY and value["encoder_version"] == ENCODER_VERSION)
        _require(value["r7_provider_validation"] is True)
        _numerical_evaluation.validate_result(value["result"])
        for key in ("encoder_updated", "native_heads_changed", "protected_external_data_used",
                    "candidate_promoted", "clinical_use_established", "patient_level_output_emitted"):
            _require(value[key] is False)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail()


def evaluate(paired: object, roles: object, provider: Callable[[int], tuple[object, object, object]],
             progress: Callable[..., object] | None = None) -> dict[str, object]:
    """Evaluate Q1 numerics through a strict R7 provider boundary."""

    _require(callable(provider) and (progress is None or callable(progress)))
    wrapped = _provider_for_r7(paired, provider)
    numerical = _numerical_evaluation.evaluate(paired, roles, wrapped, progress)
    return _wrap_result(numerical)


__all__ = [
    "ENCODER_IDENTITY", "ENCODER_VERSION", "ERROR", "NUMERICAL_SCHEMA", "PATTERNS",
    "SCHEMA", "WHOLE", "evaluate", "predict_quantiles", "train_quantile",
    "validate_result",
]
