"""Synthetic-only BRAN multisource V2 model; it performs no data or training I/O."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch import Tensor, nn

from bran_multisource_age_v2 import AGE_FEATURE_DIM, validate_normalized_age
from bran_patient_state_prototype_v1 import (
    BRANPatientStatePrototypeV1,
    PatientStateConfig,
    PosteriorState,
    _masked_finite,
    _mlp,
)


_INVALID = "multisource model inputs invalid"


def _indices(values: Iterable[int], *, limit: int, count: Optional[int] = None) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError:
        raise ValueError(_INVALID) from None
    if (count is not None and len(result) != count) or len(set(result)) != len(result):
        raise ValueError(_INVALID)
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 or x >= limit for x in result):
        raise ValueError(_INVALID)
    return result


def erase_cbc_for_completion(values: Tensor, observed_mask: Tensor, cbc_indices: Iterable[int]) -> tuple[Tensor, Tensor]:
    """Erase all nine CBC values *and* flags before a completion encode."""

    indices = _indices(cbc_indices, limit=48, count=9)
    if (not isinstance(values, Tensor) or not values.is_floating_point() or not isinstance(observed_mask, Tensor)
            or observed_mask.dtype != torch.bool or values.ndim != 2 or values.shape[1] != 59 or observed_mask.shape != values.shape):
        raise ValueError(_INVALID)
    out_values, out_mask = values.clone(), observed_mask.bool().clone()
    target = torch.tensor(indices, device=values.device, dtype=torch.long)
    out_values.index_fill_(1, target, 0.0)
    out_mask.index_fill_(1, target, False)
    return out_values, out_mask


class BRANMultisourceModelV2(BRANPatientStatePrototypeV1):
    """V2 candidate with typed seven-feature age context and two encoder arms.

    ``arm='mlp'`` uses the compact clinical MLP; ``arm='token'`` uses field,
    value, and missingness tokens followed by two deterministic transformer
    layers.  Both retain the same state-only native screening/CBC heads.
    """

    def __init__(self, arm: str, eligible_indices: Iterable[int], cbc_indices: Iterable[int], seed: int = 94101) -> None:
        nn.Module.__init__(self)
        if arm not in ("mlp", "token") or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(_INVALID)
        self.arm = arm
        self.eligible_indices = _indices(eligible_indices, limit=48, count=43)
        self.cbc_indices = _indices(cbc_indices, limit=48, count=9)
        if not set(self.cbc_indices).issubset(self.eligible_indices):
            raise ValueError(_INVALID)
        self.config = PatientStateConfig()
        c = self.config
        eligibility = torch.zeros(c.clinical_dim, dtype=torch.bool)
        eligibility[list(self.eligible_indices)] = True
        self.register_buffer("eligible_slots", eligibility, persistent=True)

        # A forked fixed seed makes every shared component byte-identical across
        # arms, independent of arm-specific allocation and caller RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.retinal_projection = _mlp(c.retinal_feature_dim + AGE_FEATURE_DIM, c.hidden_dim, c.hidden_dim)
            self.retinal_pool_gate = nn.Linear(c.hidden_dim, 1)
            self.retinal_pool = _mlp(2 * c.hidden_dim + AGE_FEATURE_DIM, c.hidden_dim, c.hidden_dim)
            self.retinal_residual = nn.Linear(c.retinal_feature_dim, c.hidden_dim, bias=False)
            self.shared_posterior = _mlp(2 * c.hidden_dim + 2, c.hidden_dim, 2 * c.shared_dim)
            self.retinal_prior = nn.Linear(c.shared_dim, c.retinal_private_dim, bias=False)
            self.clinical_prior = nn.Linear(c.shared_dim, c.clinical_private_dim, bias=False)
            self.retinal_prior_logvar = nn.Parameter(torch.zeros(c.retinal_private_dim))
            self.clinical_prior_logvar = nn.Parameter(torch.zeros(c.clinical_private_dim))
            self.retinal_delta = _mlp(c.hidden_dim, c.hidden_dim, 2 * c.retinal_private_dim)
            self.clinical_delta = _mlp(c.hidden_dim, c.hidden_dim, 2 * c.clinical_private_dim)
            self.continuous_decoder = _mlp(c.state_dim + AGE_FEATURE_DIM, c.hidden_dim, c.clinical_continuous_dim)
            self.continuous_logscale = nn.Parameter(torch.zeros(c.clinical_continuous_dim))
            self.binary_decoder = _mlp(c.state_dim + AGE_FEATURE_DIM, c.hidden_dim, c.clinical_binary_dim)
            self.retinal_decoder = _mlp(c.state_dim + AGE_FEATURE_DIM, c.hidden_dim, c.retinal_feature_dim)
            self.retinal_logscale = nn.Parameter(torch.zeros(c.retinal_feature_dim))
            self.clinical_residual_factor = nn.Parameter(torch.randn(c.clinical_continuous_dim, c.clinical_covariance_rank) * 1e-3)
            self.disease_head = nn.Linear(c.state_dim, c.disease_outputs)
            self.screening_joint_head = nn.Linear(c.state_dim, 26)
            self.cbc_joint_head = nn.Linear(c.state_dim, 9)
            if arm == "mlp":
                self.clinical_encoder = _mlp(2 * c.clinical_dim + AGE_FEATURE_DIM, c.hidden_dim, c.hidden_dim)
                self.clinical_residual = nn.Linear(c.clinical_dim, c.hidden_dim, bias=False)
            else:
                self.clinical_field_embedding = nn.Embedding(c.clinical_dim, c.hidden_dim)
                self.clinical_value_projection = nn.Linear(1, c.hidden_dim, bias=False)
                self.clinical_missing_embedding = nn.Embedding(2, c.hidden_dim)
                self.age_token_projection = nn.Linear(AGE_FEATURE_DIM, c.hidden_dim)
                layer = nn.TransformerEncoderLayer(c.hidden_dim, 4, dim_feedforward=4 * c.hidden_dim,
                                                   dropout=0.0, batch_first=True, activation="gelu")
                self.clinical_token_encoder = nn.TransformerEncoder(layer, num_layers=2)

    def _age(self, age: Tensor, batch: int) -> Tensor:
        if not isinstance(age, Tensor) or age.shape != (batch, AGE_FEATURE_DIM):
            raise ValueError(_INVALID)
        return validate_normalized_age(age)

    def _clinical_hidden(self, clean: Tensor, valid: Tensor, age: Tensor) -> Tensor:
        any_clinical = valid.any(dim=-1)
        if self.arm == "mlp":
            hidden = self.clinical_encoder(torch.cat([clean, valid.to(clean.dtype), age], dim=-1)) + self.clinical_residual(clean)
        else:
            b, fields = clean.shape
            field_ids = torch.arange(fields, device=clean.device)
            tokens = (self.clinical_field_embedding(field_ids)[None].expand(b, -1, -1)
                      + self.clinical_value_projection(clean[..., None])
                      + self.clinical_missing_embedding(valid.long()))
            age_token = self.age_token_projection(age)[:, None, :]
            encoded = self.clinical_token_encoder(torch.cat([tokens, age_token], dim=1))
            hidden = encoded.mean(dim=1)  # fixed token count keeps this pool stable.
        return hidden * any_clinical[:, None].to(hidden.dtype)

    def encode(self, clinical_values: Tensor, clinical_observed_mask: Tensor, retinal_embeddings: Tensor,
               retinal_visible_mask: Tensor, age7: Tensor, clinical_eligible_mask: Optional[Tensor] = None) -> PosteriorState:
        c = self.config
        if not isinstance(clinical_values, Tensor) or not clinical_values.is_floating_point() or clinical_values.ndim != 2 or clinical_values.shape[1] != c.clinical_dim:
            raise ValueError(_INVALID)
        if not isinstance(clinical_observed_mask, Tensor) or clinical_observed_mask.dtype != torch.bool or clinical_observed_mask.shape != clinical_values.shape:
            raise ValueError(_INVALID)
        b = clinical_values.shape[0]
        if (not isinstance(retinal_embeddings, Tensor) or not retinal_embeddings.is_floating_point() or retinal_embeddings.ndim != 3
                or retinal_embeddings.shape[0] != b or retinal_embeddings.shape[2] != c.retinal_feature_dim
                or not isinstance(retinal_visible_mask, Tensor) or retinal_visible_mask.dtype != torch.bool or retinal_visible_mask.shape != retinal_embeddings.shape[:2]):
            raise ValueError(_INVALID)
        age = self._age(age7, b)
        if age.device != clinical_values.device:
            raise ValueError(_INVALID)
        if clinical_eligible_mask is not None and (not isinstance(clinical_eligible_mask, Tensor) or clinical_eligible_mask.dtype != torch.bool or clinical_eligible_mask.device != clinical_values.device):
            raise ValueError(_INVALID)
        caller_eligible = torch.ones_like(clinical_observed_mask, dtype=torch.bool) if clinical_eligible_mask is None else clinical_eligible_mask
        if caller_eligible.shape != clinical_values.shape:
            raise ValueError(_INVALID)
        # Static eligibility always applies; binary 48:59 is permanently disabled.
        effective = clinical_observed_mask.bool() & caller_eligible & self.eligible_slots[None, :]
        clinical_clean, clinical_valid = _masked_finite(clinical_values, effective)
        clinical_any = clinical_valid.any(dim=-1)
        h_clinical = self._clinical_hidden(clinical_clean, clinical_valid, age)

        retinal_valid = retinal_visible_mask.bool() & torch.isfinite(retinal_embeddings).all(dim=-1)
        retinal_clean = torch.zeros_like(retinal_embeddings)
        expanded = retinal_valid[..., None].expand_as(retinal_embeddings)
        retinal_clean = retinal_clean.masked_scatter(expanded, retinal_embeddings.masked_select(expanded))
        retinal_any = retinal_valid.any(dim=-1)
        image_h = self.retinal_projection(torch.cat([retinal_clean, age[:, None, :].expand(-1, retinal_clean.shape[1], -1)], dim=-1))
        logits = self.retinal_pool_gate(image_h).squeeze(-1).masked_fill(~retinal_valid, -1e9)
        weights = torch.softmax(logits, dim=-1) * retinal_valid.to(image_h.dtype)
        mean = (weights[..., None] * image_h).sum(dim=1)
        simple_mean = (retinal_clean * retinal_valid[..., None].to(retinal_clean.dtype)).sum(dim=1) / retinal_valid.sum(dim=1).clamp_min(1)[:, None]
        dispersion = ((image_h - mean[:, None, :]).square() * retinal_valid[..., None].to(image_h.dtype)).sum(dim=1)
        dispersion = dispersion / retinal_valid.sum(dim=1).clamp_min(1)[:, None]
        h_retinal = self.retinal_pool(torch.cat([mean, dispersion, age], dim=-1)) + self.retinal_residual(simple_mean)
        h_retinal = h_retinal * retinal_any[:, None].to(h_retinal.dtype)

        shared_raw = self.shared_posterior(torch.cat([h_retinal, h_clinical, retinal_any[:, None].to(age.dtype), clinical_any[:, None].to(age.dtype)], dim=-1))
        s_mu, s_logvar = shared_raw.chunk(2, dim=-1)
        s_logvar = s_logvar.clamp(c.min_logvar, c.max_logvar)
        any_physiology = retinal_any | clinical_any
        s_mu, s_logvar = torch.where(any_physiology[:, None], s_mu, torch.zeros_like(s_mu)), torch.where(any_physiology[:, None], s_logvar, torch.zeros_like(s_logvar))
        r_prior_mu, cl_prior_mu = self.retinal_prior(s_mu), self.clinical_prior(s_mu)
        r_prior_lv = self.retinal_prior_logvar[None].expand_as(r_prior_mu).clamp(c.min_logvar, c.max_logvar)
        cl_prior_lv = self.clinical_prior_logvar[None].expand_as(cl_prior_mu).clamp(c.min_logvar, c.max_logvar)
        r_delta_mu, r_delta_lv = self.retinal_delta(h_retinal).chunk(2, dim=-1)
        cl_delta_mu, cl_delta_lv = self.clinical_delta(h_clinical).chunk(2, dim=-1)
        r_delta_mu, cl_delta_mu = torch.where(retinal_any[:, None], r_delta_mu, torch.zeros_like(r_delta_mu)), torch.where(clinical_any[:, None], cl_delta_mu, torch.zeros_like(cl_delta_mu))
        r_lv = torch.where(retinal_any[:, None], (r_prior_lv + r_delta_lv).clamp(c.min_logvar, c.max_logvar), r_prior_lv)
        cl_lv = torch.where(clinical_any[:, None], (cl_prior_lv + cl_delta_lv).clamp(c.min_logvar, c.max_logvar), cl_prior_lv)
        r_marginal_lv = torch.log(r_lv.exp() + s_logvar.exp() @ self.retinal_prior.weight.square().T)
        cl_marginal_lv = torch.log(cl_lv.exp() + s_logvar.exp() @ self.clinical_prior.weight.square().T)
        return PosteriorState(torch.cat([s_mu, r_prior_mu + r_delta_mu, cl_prior_mu + cl_delta_mu], dim=-1),
                              torch.cat([s_logvar, r_marginal_lv, cl_marginal_lv], dim=-1), retinal_any, clinical_any, ~any_physiology,
                              r_delta_mu, r_lv, cl_delta_mu, cl_lv, self.retinal_prior.weight, self.clinical_prior.weight)

    def forward(self, *args, **kwargs) -> PosteriorState:
        return self.encode(*args, **kwargs)

    def objective(self, state: PosteriorState, age: Tensor, target_clinical_values: Tensor, target_clinical_mask: Tensor,
                  visible_clinical_mask: Tensor, target_retinal_features: Optional[Tensor] = None,
                  target_retinal_mask: Optional[Tensor] = None, visible_retinal_mask: Optional[Tensor] = None,
                  disease_target: Optional[Tensor] = None, disease_mask: Optional[Tensor] = None,
                  kl_weight: float = 0.01, disease_weight: float = 0.0, visible_weight: float = 0.0,
                  clinical_eligible_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """V1 objective with permanent eligibility and binary-slot exclusion."""
        c = self.config
        if (not isinstance(target_clinical_values, Tensor) or not target_clinical_values.is_floating_point()
                or not isinstance(target_clinical_mask, Tensor) or target_clinical_mask.dtype != torch.bool
                or not isinstance(visible_clinical_mask, Tensor) or visible_clinical_mask.dtype != torch.bool
                or target_clinical_values.shape[-1] != c.clinical_dim or target_clinical_mask.shape != target_clinical_values.shape or visible_clinical_mask.shape != target_clinical_values.shape):
            raise ValueError(_INVALID)
        if clinical_eligible_mask is not None and (not isinstance(clinical_eligible_mask, Tensor) or clinical_eligible_mask.dtype != torch.bool):
            raise ValueError(_INVALID)
        caller = torch.ones_like(target_clinical_mask, dtype=torch.bool) if clinical_eligible_mask is None else clinical_eligible_mask
        if caller.shape != target_clinical_mask.shape:
            raise ValueError(_INVALID)
        eligible = caller & self.eligible_slots[None, :]
        target_mask, visible_mask = target_clinical_mask.bool() & eligible, visible_clinical_mask.bool() & eligible
        # Eligible slots only live in 0:48; all binary heads are intentionally unscored.
        out = self.decode(state, age, state.rsample())
        cont, _ = target_clinical_values.split([48, 11], dim=-1)
        tm, vm = target_mask[:, :48], visible_mask[:, :48]
        hidden = self._clinical_gaussian_nll(cont, out["continuous_mean"], tm & ~vm)
        visible = self._clinical_gaussian_nll(cont, out["continuous_mean"], tm & vm)
        zero = torch.zeros((), device=state.mean.device, dtype=state.mean.dtype)
        retinal, retinal_visible = zero, zero
        if target_retinal_features is not None:
            if (not isinstance(target_retinal_features, Tensor) or not target_retinal_features.is_floating_point()
                    or target_retinal_features.shape != (state.mean.shape[0], c.retinal_feature_dim)
                    or not isinstance(target_retinal_mask, Tensor) or target_retinal_mask.dtype != torch.bool or target_retinal_mask.shape != target_retinal_features.shape
                    or not isinstance(visible_retinal_mask, Tensor) or visible_retinal_mask.dtype != torch.bool or visible_retinal_mask.shape != target_retinal_features.shape):
                raise ValueError(_INVALID)
            retinal = self._gaussian_nll(target_retinal_features, out["retinal_mean"], out["retinal_logscale"], target_retinal_mask.bool() & ~visible_retinal_mask.bool())
            retinal_visible = self._gaussian_nll(target_retinal_features, out["retinal_mean"], out["retinal_logscale"], target_retinal_mask.bool() & visible_retinal_mask.bool())
        disease = zero
        if disease_target is not None:
            if (not isinstance(disease_target, Tensor) or not disease_target.is_floating_point()
                    or disease_target.shape != (state.mean.shape[0], c.disease_outputs)
                    or (disease_mask is not None and (not isinstance(disease_mask, Tensor) or disease_mask.dtype != torch.bool or disease_mask.shape != disease_target.shape))):
                raise ValueError(_INVALID)
            mask = torch.ones_like(disease_target, dtype=torch.bool) if disease_mask is None else disease_mask
            clean, valid = _masked_finite(disease_target, mask)
            disease = (torch.nn.functional.binary_cross_entropy_with_logits(out["disease_logits"], clean, reduction="none") * valid.to(clean.dtype)).sum() / valid.sum().clamp_min(1)
        kl = self.kl_to_conditional_priors(state).mean()
        total = hidden + visible_weight * (visible + retinal_visible) + retinal + kl_weight * kl + disease_weight * disease
        return {"loss": total, "hidden_target_nll": hidden, "visible_reconstruction_nll": visible,
                "retinal_hidden_nll": retinal, "retinal_visible_reconstruction_nll": retinal_visible, "kl": kl, "disease_loss": disease}

    @torch.no_grad()
    def sample_clinical(self, state: PosteriorState, age: Tensor, samples: int = 1) -> Dict[str, Tensor]:
        """V2-compatible joint panel draws with the seven-feature age context."""
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError(_INVALID)
        z = state.rsample(torch.Size([samples]))
        count, batch, width = z.shape
        age_s = self._age(age, batch)[None].expand(count, -1, -1)
        flat = torch.cat([z, age_s], dim=-1).reshape(count * batch, width + AGE_FEATURE_DIM)
        cont_mean = self.continuous_decoder(flat).reshape(count, batch, -1)
        binary_logits = self.binary_decoder(flat).reshape(count, batch, -1)
        eps_factor = torch.randn(count, batch, self.config.clinical_covariance_rank, device=z.device, dtype=z.dtype)
        correlated = torch.einsum("sbr,fr->sbf", eps_factor, self.clinical_residual_factor)
        continuous = cont_mean + correlated + torch.randn_like(cont_mean) * self.continuous_logscale.exp()[None, None, :]
        binary = torch.bernoulli(torch.sigmoid(binary_logits))
        return {"continuous": continuous, "binary": binary, "continuous_mean": cont_mean,
                "binary_probability": torch.sigmoid(binary_logits)}

    def export_config(self) -> Dict[str, object]:
        return {"version": 2, "patient_state_config": self.config.to_dict(), "arm": self.arm,
                "eligible_indices": list(self.eligible_indices), "cbc_indices": list(self.cbc_indices),
                "age_contract": {"version": 2, "feature_dim": AGE_FEATURE_DIM, "kinds": ["reported", "interval", "right_censored", "unknown"]}}

    @classmethod
    def from_config(cls, values: Dict[str, object]) -> "BRANMultisourceModelV2":
        expected_age = {"version": 2, "feature_dim": AGE_FEATURE_DIM, "kinds": ["reported", "interval", "right_censored", "unknown"]}
        if (not isinstance(values, dict) or set(values) != {"version", "patient_state_config", "arm", "eligible_indices", "cbc_indices", "age_contract"}
                or values.get("version") != 2 or values.get("patient_state_config") != PatientStateConfig().to_dict()
                or values.get("age_contract") != expected_age):
            raise ValueError(_INVALID)
        return cls(values.get("arm"), values.get("eligible_indices"), values.get("cbc_indices"))


def make_model_pair(eligible_indices: Iterable[int], cbc_indices: Iterable[int], seed: int = 94101) -> Dict[str, BRANMultisourceModelV2]:
    """Construct MLP/token arms whose common components have identical bytes."""
    eligible, cbc = tuple(eligible_indices), tuple(cbc_indices)
    return {"mlp": BRANMultisourceModelV2("mlp", eligible, cbc, seed),
            "token": BRANMultisourceModelV2("token", eligible, cbc, seed)}
