"""State-only residual CBC head. Synthetic-tested kernel; no source I/O."""
from __future__ import annotations
import copy
import torch
from torch import nn

_ERROR = 'v5_residual_cbc_contract_failed'


def require(ok):
    if not ok: raise ValueError(_ERROR)


class ResidualCBC(nn.Module):
    """Original V5 CBC head plus zero-initialized nonlinear residual.

    Detachment at this boundary prevents head training from changing the state
    encoder. No target value, availability flag, age or outcome is an input.
    """
    def __init__(self, baseline: nn.Linear, seed: int):
        super().__init__()
        require(isinstance(baseline, nn.Linear) and baseline.in_features == 192
                and baseline.out_features == 9 and baseline.bias is not None)
        require(type(seed) is int and seed >= 0)
        require(baseline.weight.device.type == 'cpu' and baseline.weight.dtype == torch.float32)
        self.baseline = copy.deepcopy(baseline).eval()
        for p in self.baseline.parameters(): p.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.residual = nn.Sequential(nn.Linear(192,128), nn.GELU(), nn.Linear(128,9))
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        return self

    def _state(self, state):
        require(isinstance(state, torch.Tensor) and state.ndim == 2 and state.shape[1] == 192
                and state.dtype == torch.float32 and state.device.type == 'cpu'
                and bool(torch.isfinite(state).all()))
        return state.detach()

    def forward(self, state):
        z = self._state(state)
        return self.baseline(z) + self.residual(z)

    def objective(self, state, target, erased_observed_mask):
        z = self._state(state)
        require(isinstance(target, torch.Tensor) and target.shape == (len(z),9)
                and target.dtype == torch.float32 and target.device == z.device)
        mask = erased_observed_mask
        require(isinstance(mask, torch.Tensor) and mask.shape == target.shape
                and mask.dtype == torch.bool and mask.device == z.device)
        require(bool(torch.isfinite(target[mask]).all()))
        correction = self.residual(z)
        prediction = self.baseline(z) + correction
        # Never allow an inaccessible NaN or target payload into the objective.
        clean_target = torch.where(mask, target.detach(), torch.zeros_like(target))
        errors = torch.where(mask, (prediction-clean_target).abs(), torch.zeros_like(prediction))
        changes = torch.where(mask, correction.square(), torch.zeros_like(correction))
        counts = mask.sum(dim=0)
        supported = counts > 0
        require(bool(torch.isfinite(prediction).all()))
        if not bool(supported.any()):
            return None  # Caller must skip optimizer.step (including AdamW decay).
        per_field = (errors.sum(0) + .1*changes.sum(0)) / counts.clamp_min(1)
        result = per_field[supported].mean()
        require(bool(torch.isfinite(result)))
        return result
