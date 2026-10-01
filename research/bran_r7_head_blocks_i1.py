"""Pure, aggregate-only native R7 screening-head block summaries.

This module deliberately does not load models, checkpoints, source tables, or
patient-level artifacts.  It centers the native 192-dimensional state on the
available training people, decomposes the fixed 26-output screening head into
its three algebraically unfolded head blocks, and summarizes held-out block
contributions with a fixed within-fold participant bootstrap.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


ERROR = "bran_r7_head_blocks_i1_contract_failed"
SCHEMA = "bran-r7-head-blocks-i1"
BLOCKS = ("shared_derived", "retinal_innovation", "clinical_innovation")
_BLOCK_SLICES = {
    "shared_derived": (0, 64),
    "retinal_innovation": (64, 128),
    "clinical_innovation": (128, 192),
}
STATE_WIDTH = 192
N_ENDPOINTS = 26
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 92571
MINIMUM_VALID_DRAWS = 900
MINIMUM_CLASS_SUPPORT = 20
SHARE_DEFINITION = (
    "normalized mean absolute centered block contribution; not explained variance"
)

PARAMETERS = {
    "state_width": STATE_WIDTH,
    "screening_outputs": N_ENDPOINTS,
    "blocks": {name: list(_BLOCK_SLICES[name]) for name in BLOCKS},
    "block_semantics": {
        "shared_derived": "shared state with private loading effects unfolded into it",
        "retinal_innovation": "retinal private innovation after shared loading removal",
        "clinical_innovation": "clinical private innovation after shared loading removal",
    },
    "centering": "available_train_only",
    "bootstrap_draws": BOOTSTRAP_DRAWS,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "minimum_valid_draws": MINIMUM_VALID_DRAWS,
    "minimum_class_support": MINIMUM_CLASS_SUPPORT,
    "confidence_interval": "marginal_95_percentile",
    "gap_definition": (
        "within_fold_positive_minus_negative_logit_gap_weighted_by_valid_participants"
    ),
    "share_definition": SHARE_DEFINITION,
}

FLAGS = {
    "patient_level_output_emitted": False,
    "arrays_or_bootstrap_draws_emitted": False,
    "causal_attribution_claim": False,
    "ablation_or_performance_search": False,
    "p_values_or_multiplicity_claim": False,
    "novel_subtype_claim": False,
}


@dataclass(frozen=True, slots=True, repr=False)
class BlockDecomposition:
    """Private numerical result from :func:`decompose`.

    The arrays are intentionally not part of any public aggregate.  They are
    returned read-only so a caller cannot accidentally mutate the transformed
    decomposition before assembling a held-out summary.
    """

    blocks: np.ndarray
    reference_logit: np.ndarray

    def __repr__(self) -> str:
        return "<BlockDecomposition private numerical result>"

    @property
    def centered_block_logits(self) -> np.ndarray:
        """Compatibility alias for the private centered block array."""

        return self.blocks

    @property
    def train_reference_logits(self) -> np.ndarray:
        """Compatibility alias for the private reference logits."""

        return self.reference_logit


@dataclass(frozen=True, slots=True, repr=False)
class HeadUnfolding:
    """Private exact algebraic unfolding of the shared-loading state."""

    states: np.ndarray
    weight: np.ndarray

    def __repr__(self) -> str:
        return "<HeadUnfolding private numerical result>"

    @property
    def transformed_states(self) -> np.ndarray:
        return self.states

    @property
    def transformed_weight(self) -> np.ndarray:
        return self.weight

    def __iter__(self):
        yield self.states
        yield self.weight


def _fail() -> None:
    raise ValueError(ERROR)


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _require_float_array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype.kind not in "f":
        _fail()
    if shape is not None and value.shape != shape:
        _fail()
    return value


def _require_numeric_array(value: Any) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype.kind not in "iuf":
        _fail()
    return value


def _require_bool_vector(value: Any, n: int) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != np.dtype(bool) or value.shape != (n,):
        _fail()
    return value


def _validate_decompose_inputs(
    states: Any,
    available: Any,
    train: Any,
    weight: Any,
    bias: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = _require_float_array(states)
    if states.ndim != 2 or states.shape[1] != STATE_WIDTH or states.shape[0] < 1:
        _fail()
    n = states.shape[0]
    available = _require_bool_vector(available, n)
    train = _require_bool_vector(train, n)
    weight = _require_float_array(weight, (N_ENDPOINTS, STATE_WIDTH))
    bias = _require_float_array(bias, (N_ENDPOINTS,))
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        _fail()
    if not np.isfinite(states[available]).all():
        _fail()
    train_available = available & train
    if int(train_available.sum()) < MINIMUM_CLASS_SUPPORT:
        _fail()
    return (
        states.astype(np.float64, copy=False),
        available,
        train_available,
        weight.astype(np.float64, copy=False),
        bias.astype(np.float64, copy=False),
    )


def unfold(
    states: np.ndarray,
    weight: np.ndarray,
    retinal_loading: np.ndarray,
    clinical_loading: np.ndarray,
) -> HeadUnfolding:
    """Unfold shared loading from the two private state blocks exactly.

    If ``r = A_retina s + r_innovation`` and ``c = A_clinical s +
    c_innovation``, the returned coordinates are ``[s, r_innovation,
    c_innovation]`` and the returned screening-head weights preserve the exact
    native linear logit.  This is algebraic re-expression only; no blocks are
    zeroed or refit.
    """

    try:
        states = _require_float_array(states)
        if states.ndim != 2 or states.shape[1] != STATE_WIDTH or states.shape[0] < 1:
            _fail()
        weight = _require_float_array(weight, (N_ENDPOINTS, STATE_WIDTH))
        retinal_loading = _require_float_array(retinal_loading, (64, 64))
        clinical_loading = _require_float_array(clinical_loading, (64, 64))
        if not (
            np.isfinite(states).all()
            and np.isfinite(weight).all()
            and np.isfinite(retinal_loading).all()
            and np.isfinite(clinical_loading).all()
        ):
            _fail()
        states = states.astype(np.float64, copy=False)
        weight = weight.astype(np.float64, copy=False)
        retinal_loading = retinal_loading.astype(np.float64, copy=False)
        clinical_loading = clinical_loading.astype(np.float64, copy=False)
        shared = states[:, 0:64]
        retinal_innovation = states[:, 64:128] - shared @ retinal_loading.T
        clinical_innovation = states[:, 128:192] - shared @ clinical_loading.T
        transformed_states = np.concatenate(
            (shared, retinal_innovation, clinical_innovation), axis=1
        )
        shared_weight = (
            weight[:, 0:64]
            + weight[:, 64:128] @ retinal_loading
            + weight[:, 128:192] @ clinical_loading
        )
        transformed_weight = np.concatenate(
            (shared_weight, weight[:, 64:128], weight[:, 128:192]), axis=1
        )
        native = states @ weight.T
        reconstructed = transformed_states @ transformed_weight.T
        if not np.allclose(native, reconstructed, rtol=1e-10, atol=1e-10):
            _fail()
        return HeadUnfolding(_readonly(transformed_states), _readonly(transformed_weight))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def decompose(
    states: np.ndarray,
    available: np.ndarray,
    train: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> BlockDecomposition:
    """Center available states on available training people and split logits.

    ``available`` is the state-availability gate.  ``train`` only selects the
    reference population and never changes which rows receive a decomposition.
    Inactive rows are represented by NaNs in the returned block array.
    """

    try:
        states, available, train_available, weight, bias = _validate_decompose_inputs(
            states, available, train, weight, bias
        )
        center = states[train_available].mean(axis=0, dtype=np.float64)
        centered = np.full(states.shape, np.nan, dtype=np.float64)
        centered[available] = states[available] - center

        reference = center @ weight.T + bias
        block_logits = np.full(
            (states.shape[0], N_ENDPOINTS, len(BLOCKS)), np.nan, dtype=np.float64
        )
        for block_index, name in enumerate(BLOCKS):
            lower, upper = _BLOCK_SLICES[name]
            block_logits[available, :, block_index] = (
                centered[available, lower:upper] @ weight[:, lower:upper].T
            )

        native = states[available] @ weight.T + bias
        reconstructed = reference + block_logits[available].sum(axis=2)
        if not np.allclose(native, reconstructed, rtol=1e-10, atol=1e-10):
            _fail()
        return BlockDecomposition(_readonly(block_logits), _readonly(reference))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def _paired_counts(folds: np.ndarray) -> np.ndarray:
    """Return fixed-seed within-fold participant multiplicities privately."""

    counts = np.zeros((BOOTSTRAP_DRAWS, len(folds)), dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for fold in range(5):
        indices = np.flatnonzero(folds == fold)
        if len(indices) == 0:
            _fail()
        for draw in range(BOOTSTRAP_DRAWS):
            sampled = rng.choice(len(indices), len(indices), replace=True)
            counts[draw, indices] = np.bincount(sampled, minlength=len(indices))
    return counts


def _validate_summary_inputs(
    blocks: Any,
    labels: Any,
    observed: Any,
    available: Any,
    folds: Any,
    endpoint_names: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    blocks = _require_float_array(blocks)
    if blocks.ndim != 3 or blocks.shape[1:] != (N_ENDPOINTS, len(BLOCKS)) or blocks.shape[0] < 1:
        _fail()
    n = blocks.shape[0]
    labels = _require_numeric_array(labels)
    if labels.shape != (n, N_ENDPOINTS):
        _fail()
    if type(observed) is not np.ndarray or observed.dtype != np.dtype(bool):
        _fail()
    if observed.shape != (n, N_ENDPOINTS):
        _fail()
    available = _require_bool_vector(available, n)
    if type(folds) is not np.ndarray or folds.ndim != 1 or folds.shape != (n,):
        _fail()
    if folds.dtype.kind not in "iu" or set(np.unique(folds).tolist()) != set(range(5)):
        _fail()
    if not np.isnan(blocks[~available]).all() or not np.isfinite(blocks[available]).all():
        _fail()
    if not np.isfinite(labels[observed]).all():
        _fail()
    if not np.isin(labels[observed], (0.0, 1.0)).all():
        _fail()
    if not isinstance(endpoint_names, (list, tuple)) or len(endpoint_names) != N_ENDPOINTS:
        _fail()
    if any(type(name) is not str or not name.strip() for name in endpoint_names):
        _fail()
    names = tuple(endpoint_names)
    if len(set(names)) != N_ENDPOINTS:
        _fail()
    return (
        blocks.astype(np.float64, copy=False),
        labels.astype(np.float64, copy=False),
        observed,
        available,
        folds.astype(np.int64, copy=False),
        names,
    )


def _endpoint_valid_rows(
    labels: np.ndarray,
    observed: np.ndarray,
    available: np.ndarray,
    folds: np.ndarray,
    endpoint: int,
) -> np.ndarray | None:
    raw = available & observed
    raw &= np.isfinite(labels[:, endpoint])
    raw &= (labels[:, endpoint] == 0.0) | (labels[:, endpoint] == 1.0)
    valid = raw.copy()
    for fold in range(5):
        fold_rows = raw & (folds == fold)
        has_positive = bool(np.any(fold_rows & (labels[:, endpoint] == 1.0)))
        has_negative = bool(np.any(fold_rows & (labels[:, endpoint] == 0.0)))
        if not (has_positive and has_negative):
            valid[folds == fold] = False
    positives = int(np.sum(valid & (labels[:, endpoint] == 1.0)))
    negatives = int(np.sum(valid & (labels[:, endpoint] == 0.0)))
    if positives < MINIMUM_CLASS_SUPPORT or negatives < MINIMUM_CLASS_SUPPORT:
        return None
    return valid


def _gap_point_and_draws(
    endpoint_blocks: np.ndarray,
    endpoint_labels: np.ndarray,
    valid: np.ndarray,
    folds: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    point_numerator = np.zeros(3, dtype=np.float64)
    point_denominator = 0.0
    for fold in range(5):
        rows = valid & (folds == fold)
        positive = rows & (endpoint_labels == 1.0)
        negative = rows & (endpoint_labels == 0.0)
        n_rows = int(rows.sum())
        if n_rows == 0 or not positive.any() or not negative.any():
            continue
        gap = endpoint_blocks[positive].mean(axis=0) - endpoint_blocks[negative].mean(axis=0)
        point_numerator += gap * float(n_rows)
        point_denominator += float(n_rows)
    if point_denominator <= 0.0:
        _fail()
    point = point_numerator / point_denominator

    eligible_folds = tuple(
        fold for fold in range(5) if np.any(valid & (folds == fold))
    )
    draws = np.full((BOOTSTRAP_DRAWS, 3), np.nan, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        numerator = np.zeros(3, dtype=np.float64)
        denominator = 0.0
        draw_invalid = False
        for fold in eligible_folds:
            rows = valid & (folds == fold)
            positive = rows & (endpoint_labels == 1.0)
            negative = rows & (endpoint_labels == 0.0)
            sampled_weights = counts[draw]
            positive_weight = float(sampled_weights[positive].sum())
            negative_weight = float(sampled_weights[negative].sum())
            fold_weight = float(sampled_weights[rows].sum())
            if positive_weight <= 0.0 or negative_weight <= 0.0 or fold_weight <= 0.0:
                draw_invalid = True
                break
            positive_mean = (
                endpoint_blocks[positive] * sampled_weights[positive, None]
            ).sum(axis=0) / positive_weight
            negative_mean = (
                endpoint_blocks[negative] * sampled_weights[negative, None]
            ).sum(axis=0) / negative_weight
            numerator += (positive_mean - negative_mean) * fold_weight
            denominator += fold_weight
        if not draw_invalid and denominator > 0.0:
            draws[draw] = numerator / denominator
    return point, draws


def _supported_endpoint(
    endpoint_blocks: np.ndarray,
    endpoint_labels: np.ndarray,
    valid: np.ndarray,
    folds: np.ndarray,
    counts: np.ndarray,
) -> dict[str, Any] | None:
    mean_absolute = np.mean(np.abs(endpoint_blocks[valid]), axis=0, dtype=np.float64)
    denominator = float(mean_absolute.sum())
    point, draw_values = _gap_point_and_draws(
        endpoint_blocks, endpoint_labels, valid, folds, counts
    )
    metrics: dict[str, Any] = {}
    for block_index, block_name in enumerate(BLOCKS):
        finite = draw_values[:, block_index][np.isfinite(draw_values[:, block_index])]
        if len(finite) < MINIMUM_VALID_DRAWS:
            return None
        if denominator > 0.0:
            share: float | None = float(mean_absolute[block_index] / denominator)
        else:
            share = None
        ci = np.percentile(finite, [2.5, 97.5]).astype(np.float64)
        metrics[block_name] = {
            "mean_absolute_contribution": float(mean_absolute[block_index]),
            "share_of_total_mean_absolute": share,
            "signed_positive_minus_negative_logit_gap": {
                "estimate": float(point[block_index]),
                "ci95": [float(ci[0]), float(ci[1])],
            },
        }
    return {
        "status": "supported",
        "share_definition": SHARE_DEFINITION,
        "block_metrics": metrics,
    }


def summarize(
    blocks: np.ndarray,
    labels: np.ndarray,
    observed: np.ndarray,
    available: np.ndarray,
    folds: np.ndarray,
    endpoint_names: Sequence[str],
) -> dict[str, Any]:
    """Return disclosure-safe aggregate block contributions for 26 endpoints."""

    try:
        blocks, labels, observed, available, folds, names = _validate_summary_inputs(
            blocks, labels, observed, available, folds, endpoint_names
        )
        counts = _paired_counts(folds)
        endpoint_results: dict[str, Any] = {}
        for endpoint_index, name in enumerate(names):
            valid = _endpoint_valid_rows(
                labels,
                observed[:, endpoint_index],
                available,
                folds,
                endpoint_index,
            )
            if valid is None:
                endpoint_results[name] = {"status": "unsupported"}
                continue
            cell = _supported_endpoint(
                blocks[:, endpoint_index, :],
                labels[:, endpoint_index],
                valid,
                folds,
                counts,
            )
            endpoint_results[name] = cell if cell is not None else {"status": "unsupported"}
        public_parameters = copy.deepcopy(PARAMETERS)
        result = {
            "schema": SCHEMA,
            "status": "completed",
            "parameters": public_parameters,
            "endpoint_names": list(names),
            "endpoints": endpoint_results,
            "flags": {key: value for key, value in FLAGS.items()},
        }
        validate_result(result)
        return result
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def _is_plain_number(value: Any) -> bool:
    return type(value) is float and np.isfinite(value)


def _reject_arrays(value: Any) -> None:
    if isinstance(value, np.ndarray):
        _fail()
    if isinstance(value, np.generic):
        _fail()
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            _fail()
        items = value.values()
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return
    for item in items:
        _reject_arrays(item)


def _validate_metric(metric: Any) -> None:
    if type(metric) is not dict or set(metric) != {
        "mean_absolute_contribution",
        "share_of_total_mean_absolute",
        "signed_positive_minus_negative_logit_gap",
    }:
        _fail()
    if (
        not _is_plain_number(metric["mean_absolute_contribution"])
        or metric["mean_absolute_contribution"] < 0.0
    ):
        _fail()
    share = metric["share_of_total_mean_absolute"]
    if share is not None and (not _is_plain_number(share) or not 0.0 <= share <= 1.0):
        _fail()
    gap = metric["signed_positive_minus_negative_logit_gap"]
    if type(gap) is not dict or set(gap) != {"estimate", "ci95"}:
        _fail()
    if not _is_plain_number(gap["estimate"]):
        _fail()
    ci = gap["ci95"]
    if type(ci) is not list or len(ci) != 2 or not all(_is_plain_number(x) for x in ci):
        _fail()
    if ci[0] > ci[1]:
        _fail()


def _validate_parameters(parameters: Any) -> None:
    if type(parameters) is not dict or parameters != PARAMETERS:
        _fail()


def validate_result(result: Any) -> bool:
    """Validate the closed aggregate schema; raise one closed error on failure."""

    try:
        _reject_arrays(result)
        if type(result) is not dict or set(result) != {
            "schema",
            "status",
            "parameters",
            "endpoint_names",
            "endpoints",
            "flags",
        }:
            _fail()
        if result["schema"] != SCHEMA or result["status"] != "completed":
            _fail()
        _validate_parameters(result["parameters"])
        names = result["endpoint_names"]
        if type(names) is not list or len(names) != N_ENDPOINTS:
            _fail()
        if any(type(name) is not str or not name.strip() for name in names):
            _fail()
        if tuple(names) != tuple(dict.fromkeys(names)):
            _fail()
        endpoints = result["endpoints"]
        if type(endpoints) is not dict or set(endpoints) != set(names):
            _fail()
        for name in names:
            cell = endpoints[name]
            if type(cell) is not dict or "status" not in cell:
                _fail()
            if cell["status"] == "unsupported":
                if set(cell) != {"status"}:
                    _fail()
                continue
            if cell["status"] != "supported" or set(cell) != {
                "status",
                "share_definition",
                "block_metrics",
            }:
                _fail()
            if cell["share_definition"] != SHARE_DEFINITION:
                _fail()
            metrics = cell["block_metrics"]
            if type(metrics) is not dict or set(metrics) != set(BLOCKS):
                _fail()
            for block_name in BLOCKS:
                _validate_metric(metrics[block_name])
            shares = [metrics[name]["share_of_total_mean_absolute"] for name in BLOCKS]
            means = [metrics[name]["mean_absolute_contribution"] for name in BLOCKS]
            denominator = sum(means)
            if all(share is None for share in shares):
                if denominator != 0.0:
                    _fail()
            elif denominator == 0.0 or any(share is None for share in shares):
                _fail()
            elif any(
                not np.isclose(share, mean / denominator, rtol=0.0, atol=1e-12)
                for share, mean in zip(shares, means)
            ) or not np.isclose(sum(shares), 1.0, rtol=0.0, atol=1e-12):
                _fail()
        flags = result["flags"]
        if type(flags) is not dict or flags != FLAGS:
            _fail()
        return True
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


validate = validate_result


__all__ = [
    "BLOCKS",
    "BlockDecomposition",
    "ERROR",
    "FLAGS",
    "HeadUnfolding",
    "PARAMETERS",
    "SCHEMA",
    "decompose",
    "summarize",
    "unfold",
    "validate",
    "validate_result",
]
