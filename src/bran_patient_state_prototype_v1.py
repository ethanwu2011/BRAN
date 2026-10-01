"""Compact, synthetic-only BRAN patient-state prototype (V1).

This is a research scaffold, not a clinical model.  It uses a hierarchical
Gaussian posterior: q(s) is diagonal and private blocks are conditionally
Gaussian given the same sampled s.  A low-rank term is used only in the
continuous clinical observation residual for coherent panel sampling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PatientStateConfig:
    """All dimensions are deliberately small; default state width is 192."""

    clinical_continuous_dim: int = 48
    clinical_binary_dim: int = 11
    retinal_feature_dim: int = 384
    shared_dim: int = 64
    retinal_private_dim: int = 64
    clinical_private_dim: int = 64
    hidden_dim: int = 128
    clinical_covariance_rank: int = 3
    disease_outputs: int = 1
    min_logvar: float = -8.0
    max_logvar: float = 4.0

    @property
    def state_dim(self) -> int:
        return self.shared_dim + self.retinal_private_dim + self.clinical_private_dim

    @property
    def clinical_dim(self) -> int:
        return self.clinical_continuous_dim + self.clinical_binary_dim

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, object]) -> "PatientStateConfig":
        return cls(**values)


@dataclass
class PosteriorState:
    """The sole physiological state interface used by all heads and decoders."""

    mean: Tensor
    logvar: Tensor
    retinal_available: Tensor
    clinical_available: Tensor
    abstain: Tensor
    retinal_delta_mean: Tensor
    retinal_cond_logvar: Tensor
    clinical_delta_mean: Tensor
    clinical_cond_logvar: Tensor
    retinal_loading: Tensor
    clinical_loading: Tensor

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Sample the shared block once, then private blocks conditional on it."""
        sdim = self.retinal_loading.shape[1]
        s_mean, s_lv = self.mean[..., :sdim], self.logvar[..., :sdim]
        shared = s_mean + torch.randn(sample_shape + s_mean.shape, device=s_mean.device, dtype=s_mean.dtype) * torch.exp(0.5 * s_lv)
        r_mean = torch.einsum("...s,rs->...r", shared, self.retinal_loading) + self.retinal_delta_mean
        c_mean = torch.einsum("...s,cs->...c", shared, self.clinical_loading) + self.clinical_delta_mean
        retinal = r_mean + torch.randn_like(r_mean) * torch.exp(0.5 * self.retinal_cond_logvar)
        clinical = c_mean + torch.randn_like(c_mean) * torch.exp(0.5 * self.clinical_cond_logvar)
        return torch.cat([shared, retinal, clinical], dim=-1)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim))


def _masked_finite(values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Copy only valid elements; hidden NaNs/sentinels never enter a network."""
    valid = mask.bool() & torch.isfinite(values)
    clean = torch.zeros_like(values)
    # masked_select/scatter avoids doing arithmetic with masked values.
    clean = clean.masked_scatter(valid, values.masked_select(valid))
    return clean, valid


class BRANPatientStatePrototypeV1(nn.Module):
    """Small set-pooled hierarchical latent model with typed observation heads.

    Retinal inputs are frozen per-image embeddings, never raw pixels.  The
    gated mean/dispersion pool is a deliberate V1 compact substitute for a
    transformer.  Its shared posterior is diagonal but private blocks are
    conditionally Gaussian on the common shared draw; it is not a calibrated
    physiological posterior or a biological compartment model.
    """

    def __init__(self, config: PatientStateConfig = PatientStateConfig()) -> None:
        super().__init__()
        if config.state_dim > 384:
            raise ValueError("V1 state_dim must be <= 384")
        self.config = config
        c = config
        self.clinical_encoder = _mlp(2 * c.clinical_dim + 1, c.hidden_dim, c.hidden_dim)
        self.clinical_residual = nn.Linear(c.clinical_dim, c.hidden_dim, bias=False)
        self.image_encoder = _mlp(c.retinal_feature_dim + 1, c.hidden_dim, c.hidden_dim)
        self.image_gate = nn.Linear(c.hidden_dim, 1)
        self.retinal_encoder = _mlp(2 * c.hidden_dim + 1, c.hidden_dim, c.hidden_dim)
        self.retinal_residual = nn.Linear(c.retinal_feature_dim, c.hidden_dim, bias=False)
        self.shared_posterior = _mlp(2 * c.hidden_dim + 2, c.hidden_dim, 2 * c.shared_dim)
        # Conditional p(u_m|s)=N(A_m s, diag(exp(prior_logvar))).
        # No bias makes all-empty physiology the actual joint hierarchical prior.
        self.retinal_prior = nn.Linear(c.shared_dim, c.retinal_private_dim, bias=False)
        self.clinical_prior = nn.Linear(c.shared_dim, c.clinical_private_dim, bias=False)
        self.retinal_prior_logvar = nn.Parameter(torch.zeros(c.retinal_private_dim))
        self.clinical_prior_logvar = nn.Parameter(torch.zeros(c.clinical_private_dim))
        self.retinal_delta = _mlp(c.hidden_dim, c.hidden_dim, 2 * c.retinal_private_dim)
        self.clinical_delta = _mlp(c.hidden_dim, c.hidden_dim, 2 * c.clinical_private_dim)

        self.continuous_decoder = _mlp(c.state_dim + 1, c.hidden_dim, c.clinical_continuous_dim)
        self.continuous_logscale = nn.Parameter(torch.zeros(c.clinical_continuous_dim))
        self.binary_decoder = _mlp(c.state_dim + 1, c.hidden_dim, c.clinical_binary_dim)
        self.retinal_decoder = _mlp(c.state_dim + 1, c.hidden_dim, c.retinal_feature_dim)
        self.retinal_logscale = nn.Parameter(torch.zeros(c.retinal_feature_dim))
        self.clinical_residual_factor = nn.Parameter(
            torch.randn(c.clinical_continuous_dim, c.clinical_covariance_rank) * 1e-3)
        self.disease_head = nn.Linear(c.state_dim, c.disease_outputs)

    def export_config(self) -> Dict[str, object]:
        """JSON-serializable architecture configuration (not weights)."""
        return self.config.to_dict()

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "BRANPatientStatePrototypeV1":
        return cls(PatientStateConfig.from_dict(config))

    def _age(self, age: Tensor, batch: int) -> Tensor:
        if age.ndim == 1:
            age = age[:, None]
        if age.shape != (batch, 1):
            raise ValueError("age must have shape [batch] or [batch, 1]")
        if not torch.isfinite(age).all():
            raise ValueError("age is explicit required context and must be finite")
        return age

    def forward(
        self,
        clinical_values: Tensor,
        clinical_observed_mask: Tensor,
        retinal_embeddings: Tensor,
        retinal_visible_mask: Tensor,
        age: Tensor,
        clinical_eligible_mask: Optional[Tensor] = None,
    ) -> PosteriorState:
        """Infer q(s,u_retina,u_clinical | visible evidence).

        Masked values are copied to zeros before any learned operation.  The
        mask itself remains an input so missingness is represented explicitly.
        """
        c = self.config
        if clinical_values.ndim != 2 or clinical_values.shape[1] != c.clinical_dim:
            raise ValueError(f"clinical_values must be [batch, {c.clinical_dim}]")
        if clinical_observed_mask.shape != clinical_values.shape:
            raise ValueError("clinical_observed_mask shape mismatch")
        b = clinical_values.shape[0]
        if retinal_embeddings.ndim != 3 or retinal_embeddings.shape[0] != b or retinal_embeddings.shape[2] != c.retinal_feature_dim:
            raise ValueError(f"retinal_embeddings must be [batch, images, {c.retinal_feature_dim}]")
        if retinal_visible_mask.shape != retinal_embeddings.shape[:2]:
            raise ValueError("retinal_visible_mask shape mismatch")
        age = self._age(age, b)
        if clinical_eligible_mask is None:
            clinical_eligible_mask = torch.ones_like(clinical_observed_mask, dtype=torch.bool)
        if clinical_eligible_mask.shape != clinical_values.shape:
            raise ValueError("clinical_eligible_mask shape mismatch")

        clinical_clean, clinical_valid = _masked_finite(
            clinical_values, clinical_observed_mask.bool() & clinical_eligible_mask.bool())
        clinical_any = clinical_valid.any(dim=-1)
        clinical_input = torch.cat([clinical_clean, clinical_valid.to(clinical_clean.dtype), age], dim=-1)
        h_clinical = self.clinical_encoder(clinical_input) + self.clinical_residual(clinical_clean)
        h_clinical = h_clinical * clinical_any[:, None].to(h_clinical.dtype)

        retinal_valid = retinal_visible_mask.bool() & torch.isfinite(retinal_embeddings).all(dim=-1)
        retinal_clean = torch.zeros_like(retinal_embeddings)
        # Select complete per-image rows only, so invisible corrupt embeddings are untouched.
        expanded = retinal_valid[..., None].expand_as(retinal_embeddings)
        retinal_clean = retinal_clean.masked_scatter(expanded, retinal_embeddings.masked_select(expanded))
        retinal_any = retinal_valid.any(dim=-1)
        image_h = self.image_encoder(torch.cat([retinal_clean, age[:, None, :].expand(-1, retinal_clean.shape[1], -1)], dim=-1))
        logits = self.image_gate(image_h).squeeze(-1).masked_fill(~retinal_valid, -1e9)
        weights = torch.softmax(logits, dim=-1) * retinal_valid.to(image_h.dtype)
        mean = (weights[..., None] * image_h).sum(dim=1)
        simple_mean = (retinal_clean * retinal_valid[..., None].to(retinal_clean.dtype)).sum(dim=1) / retinal_valid.sum(dim=1).clamp_min(1)[:, None]
        # Dispersion is permutation invariant and informative only when images exist.
        disp = ((image_h - mean[:, None, :]).square() * retinal_valid[..., None].to(image_h.dtype)).sum(dim=1)
        disp = disp / retinal_valid.sum(dim=1).clamp_min(1)[:, None]
        h_retinal = self.retinal_encoder(torch.cat([mean, disp, age], dim=-1))
        # Direct residual preservation enters the fixed posterior state via h_retinal.
        h_retinal = h_retinal + self.retinal_residual(simple_mean)
        h_retinal = h_retinal * retinal_any[:, None].to(h_retinal.dtype)

        shared_raw = self.shared_posterior(torch.cat([
            h_retinal, h_clinical, retinal_any[:, None].to(age.dtype), clinical_any[:, None].to(age.dtype)
        ], dim=-1))
        s_mu, s_logvar = shared_raw.chunk(2, dim=-1)
        s_logvar = s_logvar.clamp(c.min_logvar, c.max_logvar)
        any_physiology = retinal_any | clinical_any
        s_mu = torch.where(any_physiology[:, None], s_mu, torch.zeros_like(s_mu))
        s_logvar = torch.where(any_physiology[:, None], s_logvar, torch.zeros_like(s_logvar))

        r_prior_mu, cl_prior_mu = self.retinal_prior(s_mu), self.clinical_prior(s_mu)
        r_prior_lv = self.retinal_prior_logvar[None, :].expand_as(r_prior_mu).clamp(c.min_logvar, c.max_logvar)
        cl_prior_lv = self.clinical_prior_logvar[None, :].expand_as(cl_prior_mu).clamp(c.min_logvar, c.max_logvar)
        r_delta_mu, r_delta_lv = self.retinal_delta(h_retinal).chunk(2, dim=-1)
        cl_delta_mu, cl_delta_lv = self.clinical_delta(h_clinical).chunk(2, dim=-1)
        r_delta_mu = torch.where(retinal_any[:, None], r_delta_mu, torch.zeros_like(r_delta_mu))
        cl_delta_mu = torch.where(clinical_any[:, None], cl_delta_mu, torch.zeros_like(cl_delta_mu))
        r_mu = r_prior_mu + r_delta_mu
        cl_mu = cl_prior_mu + cl_delta_mu
        r_lv = torch.where(retinal_any[:, None], (r_prior_lv + r_delta_lv).clamp(c.min_logvar, c.max_logvar), r_prior_lv)
        cl_lv = torch.where(clinical_any[:, None], (cl_prior_lv + cl_delta_lv).clamp(c.min_logvar, c.max_logvar), cl_prior_lv)
        # Exported sidecar is each block's marginal variance, including shared
        # uncertainty propagated through A_m.  The conditional tensors remain
        # in the state only to enable coherent reparameterized samples.
        r_marginal_lv = torch.log(r_lv.exp() + s_logvar.exp() @ self.retinal_prior.weight.square().T)
        cl_marginal_lv = torch.log(cl_lv.exp() + s_logvar.exp() @ self.clinical_prior.weight.square().T)
        return PosteriorState(torch.cat([s_mu, r_mu, cl_mu], dim=-1), torch.cat([s_logvar, r_marginal_lv, cl_marginal_lv], dim=-1),
                              retinal_any, clinical_any, ~any_physiology, r_delta_mu, r_lv, cl_delta_mu, cl_lv,
                              self.retinal_prior.weight, self.clinical_prior.weight)

    encode = forward

    def _decoder_input(self, state_vector: Tensor, age: Tensor) -> Tensor:
        return torch.cat([state_vector, self._age(age, state_vector.shape[0])], dim=-1)

    def decode(self, state: PosteriorState, age: Tensor, state_vector: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """All prediction heads consume exactly the exported physiological state."""
        vector = state.mean if state_vector is None else state_vector
        x = self._decoder_input(vector, age)
        return {"continuous_mean": self.continuous_decoder(x),
                "continuous_logscale": self.continuous_logscale[None, :].expand(x.shape[0], -1),
                "binary_logits": self.binary_decoder(x),
                "retinal_mean": self.retinal_decoder(x),
                "retinal_logscale": self.retinal_logscale[None, :].expand(x.shape[0], -1),
                "disease_logits": self.disease_head(state.mean)}

    def predictive_mean(self, state: PosteriorState, age: Tensor) -> Dict[str, Tensor]:
        """Plug-in decoder locations at posterior mean, not a true posterior expectation."""
        out = self.decode(state, age)
        return {"continuous": out["continuous_mean"], "binary_probability": torch.sigmoid(out["binary_logits"]),
                "retinal_features": out["retinal_mean"]}

    def kl_to_conditional_priors(self, state: PosteriorState) -> Tensor:
        c = self.config
        s_mu, s_lv = state.mean[:, :c.shared_dim], state.logvar[:, :c.shared_dim]
        r_plv = self.retinal_prior_logvar[None, :].expand_as(state.retinal_cond_logvar).clamp(c.min_logvar, c.max_logvar)
        cl_plv = self.clinical_prior_logvar[None, :].expand_as(state.clinical_cond_logvar).clamp(c.min_logvar, c.max_logvar)
        def kl(mu: Tensor, lv: Tensor, pm: Tensor, plv: Tensor) -> Tensor:
            return 0.5 * (plv - lv + (lv.exp() + (mu - pm).square()) / plv.exp() - 1).sum(dim=-1)
        # Given s, matching A_m*s terms cancel exactly, yielding an analytic
        # conditional KL without plugging a sampled or mean s into the prior.
        return (kl(s_mu, s_lv, torch.zeros_like(s_mu), torch.zeros_like(s_lv))
                + kl(state.retinal_delta_mean, state.retinal_cond_logvar, torch.zeros_like(state.retinal_delta_mean), r_plv)
                + kl(state.clinical_delta_mean, state.clinical_cond_logvar, torch.zeros_like(state.clinical_delta_mean), cl_plv))

    @staticmethod
    def _gaussian_nll(value: Tensor, mean: Tensor, logscale: Tensor, mask: Tensor) -> Tensor:
        clean, valid = _masked_finite(value, mask)
        nll = 0.5 * (((clean - mean) / logscale.exp()).square() + 2 * logscale + 1.8378770664093453)
        denom = valid.sum().clamp_min(1)
        return (nll * valid.to(nll.dtype)).sum() / denom

    def _clinical_gaussian_nll(self, value: Tensor, mean: Tensor, mask: Tensor) -> Tensor:
        """Masked Gaussian panel likelihood with diag + low-rank covariance.

        A batched Woodbury evaluation accommodates a different eligible field
        subset per row and explicitly trains the residual factor used by
        ``sample_clinical``.  It is algebraically the same Gaussian likelihood
        as a selected-field Cholesky factorization, but is much cheaper for a
        48-field, rank-3 CPU prototype.
        """
        clean, valid = _masked_finite(value, mask)
        selected = valid.to(value.dtype)
        residual = (clean - mean) * selected
        inv_diag = selected / self.continuous_logscale.mul(2).exp()[None, :]
        factor = self.clinical_residual_factor[None, :, :] * selected[..., None]
        # I + F' D^-1 F is only rank x rank (default 3 x 3).
        inner = torch.eye(self.config.clinical_covariance_rank, device=value.device, dtype=value.dtype)[None]
        inner = inner + torch.einsum("bfr,bf,bfs->brs", factor, inv_diag, factor)
        chol = torch.linalg.cholesky(inner)
        v = torch.einsum("bfr,bf,bf->br", factor, inv_diag, residual)
        solved = torch.cholesky_solve(v[..., None], chol).squeeze(-1)
        quadratic = (residual.square() * inv_diag).sum(-1) - (v * solved).sum(-1)
        logdet = (selected * self.continuous_logscale.mul(2)[None, :]).sum(-1)
        logdet = logdet + 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        count = selected.sum(-1)
        nll = 0.5 * (quadratic + logdet + count * 1.8378770664093453)
        return nll.sum() / count.sum().clamp_min(1)

    def objective(
        self, state: PosteriorState, age: Tensor, target_clinical_values: Tensor,
        target_clinical_mask: Tensor, visible_clinical_mask: Tensor,
        target_retinal_features: Optional[Tensor] = None, target_retinal_mask: Optional[Tensor] = None,
        visible_retinal_mask: Optional[Tensor] = None, disease_target: Optional[Tensor] = None,
        disease_mask: Optional[Tensor] = None, kl_weight: float = 0.01, disease_weight: float = 0.0,
        visible_weight: float = 0.0,
    ) -> Dict[str, Tensor]:
        """Hidden-target and visible reconstruction are scored separately."""
        c, out = self.config, self.decode(state, age, state.rsample())
        if target_clinical_values.shape[-1] != c.clinical_dim:
            raise ValueError("target clinical dimension mismatch")
        cont, binary = target_clinical_values.split([c.clinical_continuous_dim, c.clinical_binary_dim], dim=-1)
        tm_cont, tm_bin = target_clinical_mask.bool().split([c.clinical_continuous_dim, c.clinical_binary_dim], dim=-1)
        vm_cont, vm_bin = visible_clinical_mask.bool().split([c.clinical_continuous_dim, c.clinical_binary_dim], dim=-1)
        hidden_cont = tm_cont & ~vm_cont
        hidden_bin = tm_bin & ~vm_bin
        hidden = self._clinical_gaussian_nll(cont, out["continuous_mean"], hidden_cont)
        bclean, bvalid = _masked_finite(binary, hidden_bin)
        hidden = hidden + (F.binary_cross_entropy_with_logits(out["binary_logits"], bclean, reduction="none") * bvalid.to(bclean.dtype)).sum() / bvalid.sum().clamp_min(1)
        visible = self._clinical_gaussian_nll(cont, out["continuous_mean"], tm_cont & vm_cont)
        bclean, bvalid = _masked_finite(binary, tm_bin & vm_bin)
        visible = visible + (F.binary_cross_entropy_with_logits(out["binary_logits"], bclean, reduction="none") * bvalid.to(bclean.dtype)).sum() / bvalid.sum().clamp_min(1)
        retinal = torch.zeros((), device=state.mean.device, dtype=state.mean.dtype)
        retinal_visible = torch.zeros_like(retinal)
        if target_retinal_features is not None:
            if target_retinal_mask is None or visible_retinal_mask is None:
                raise ValueError("retinal target and visible masks are both required")
            retinal = self._gaussian_nll(target_retinal_features, out["retinal_mean"], out["retinal_logscale"], target_retinal_mask.bool() & ~visible_retinal_mask.bool())
            retinal_visible = self._gaussian_nll(target_retinal_features, out["retinal_mean"], out["retinal_logscale"], target_retinal_mask.bool() & visible_retinal_mask.bool())
        disease = torch.zeros_like(retinal)
        if disease_target is not None:
            if disease_mask is None:
                disease_mask = torch.ones_like(disease_target, dtype=torch.bool)
            dclean, dvalid = _masked_finite(disease_target, disease_mask)
            disease = (F.binary_cross_entropy_with_logits(out["disease_logits"], dclean, reduction="none") * dvalid.to(dclean.dtype)).sum() / dvalid.sum().clamp_min(1)
        kl = self.kl_to_conditional_priors(state).mean()
        total = hidden + visible_weight * (visible + retinal_visible) + retinal + kl_weight * kl + disease_weight * disease
        return {"loss": total, "hidden_target_nll": hidden, "visible_reconstruction_nll": visible,
                "retinal_hidden_nll": retinal, "retinal_visible_reconstruction_nll": retinal_visible, "kl": kl, "disease_loss": disease}

    def hidden_only_objective(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Convenience route: hidden targets plus KL, with disease supervision off."""
        kwargs["visible_weight"] = 0.0
        kwargs["disease_weight"] = 0.0
        return self.objective(*args, **kwargs)

    @torch.no_grad()
    def sample_clinical(self, state: PosteriorState, age: Tensor, samples: int = 1) -> Dict[str, Tensor]:
        """Joint draws: one state draw and one low-rank residual draw per panel."""
        if samples < 1:
            raise ValueError("samples must be positive")
        z = state.rsample(torch.Size([samples]))  # [S, B, D], shared across every field
        s, b, d = z.shape
        age_s = self._age(age, b)[None].expand(s, -1, -1)
        flat = torch.cat([z, age_s], dim=-1).reshape(s * b, d + 1)
        cont_mean = self.continuous_decoder(flat).reshape(s, b, -1)
        binary_logits = self.binary_decoder(flat).reshape(s, b, -1)
        eps_factor = torch.randn(s, b, self.config.clinical_covariance_rank, device=z.device, dtype=z.dtype)
        correlated = torch.einsum("sbr,fr->sbf", eps_factor, self.clinical_residual_factor)
        independent = torch.randn_like(cont_mean) * self.continuous_logscale.exp()[None, None, :]
        continuous = cont_mean + correlated + independent
        binary = torch.bernoulli(torch.sigmoid(binary_logits))
        return {"continuous": continuous, "binary": binary, "continuous_mean": cont_mean,
                "binary_probability": torch.sigmoid(binary_logits)}
