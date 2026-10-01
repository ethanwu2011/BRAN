"""Detached-state conditional CBC quantile attachment; no source or checkpoint I/O."""
from __future__ import annotations

import copy

import torch
from torch import Tensor, nn


_ERROR = "v5_quantile_cbc_contract_failed"
_QUANTILES = (0.05, 0.50, 0.95)


def _require(ok: bool) -> None:
    if not ok:
        raise ValueError(_ERROR)


class QuantileCBC(nn.Module):
    """Frozen native CBC center plus a zero-initialized 5/50/95 attachment."""

    def __init__(self, baseline: nn.Linear, seed: int) -> None:
        super().__init__()
        _require(isinstance(baseline, nn.Linear) and baseline.in_features == 192 and baseline.out_features == 9
                 and baseline.bias is not None and type(seed) is int and seed >= 0)
        _require(baseline.weight.device.type == "cpu" and baseline.weight.dtype == torch.float32)
        self.baseline = copy.deepcopy(baseline).eval()
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.quantiles = nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 27))
            nn.init.zeros_(self.quantiles[-1].weight)
            nn.init.zeros_(self.quantiles[-1].bias)

    def train(self, mode: bool = True) -> "QuantileCBC":
        super().train(mode)
        self.baseline.eval()
        return self

    @staticmethod
    def _state(state: Tensor) -> Tensor:
        _require(isinstance(state, Tensor) and state.shape[1:] == (192,) and state.ndim == 2
                 and state.dtype == torch.float32 and state.device.type == "cpu" and bool(torch.isfinite(state).all()))
        return state.detach()

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return noncrossing standardized ``(q05, median, q95)`` tensors."""

        z = self._state(state)
        values = self.quantiles(z)
        correction, lower, upper = values.split(9, dim=1)
        median = self.baseline(z) + correction
        q05 = median - torch.nn.functional.softplus(lower) - 1e-6
        q95 = median + torch.nn.functional.softplus(upper) + 1e-6
        _require(bool(torch.isfinite(q05).all()) and bool(torch.isfinite(median).all()) and bool(torch.isfinite(q95).all())
                 and bool((q05 < median).all()) and bool((median < q95).all()))
        return q05, median, q95

    def objective(self, state: Tensor, target: Tensor, erased_observed_mask: Tensor) -> Tensor | None:
        """Equal-supported-field, equal-quantile pinball loss on erased targets."""

        z = self._state(state)
        _require(isinstance(target, Tensor) and target.shape == (len(z), 9) and target.dtype == torch.float32
                 and target.device == z.device and isinstance(erased_observed_mask, Tensor)
                 and erased_observed_mask.dtype == torch.bool and erased_observed_mask.shape == target.shape
                 and erased_observed_mask.device == z.device and bool(torch.isfinite(target[erased_observed_mask]).all()))
        q05, median, q95 = self(z)
        mask = erased_observed_mask
        supported = mask.sum(dim=0) > 0
        if not bool(supported.any()):
            return None
        clean = torch.where(mask, target.detach(), torch.zeros_like(target))
        losses = []
        for prediction, quantile in zip((q05, median, q95), _QUANTILES):
            error = clean - prediction
            pinball = torch.maximum(quantile * error, (quantile - 1.0) * error)
            per_field = torch.where(mask, pinball, torch.zeros_like(pinball)).sum(dim=0) / mask.sum(dim=0).clamp_min(1)
            losses.append(per_field)
        loss = torch.stack(losses, dim=0).mean(dim=0)[supported].mean()
        _require(bool(torch.isfinite(loss)))
        return loss
