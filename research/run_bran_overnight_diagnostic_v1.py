"""Offline-only exploratory BRAN overnight diagnostic.

Imports are row-free.  The canonical AI-READI loader is imported only after
the frozen protocol, code hashes, output paths, lock, support receipt and
source hashes have all been validated.  This is a new descriptive diagnostic,
not a modification of any historical V6.2 protocol or a clinical result.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from contextlib import contextmanager

import numpy as np

PROTOCOL_SCHEMA = "bran-overnight-diagnostic-v1"
ARMS = ("age", "original_retinal", "original_clinical", "original_concat",
        "prototype_retinal", "prototype_clinical", "prototype_both")
CBC_FIELDS = ("hct", "hemoglobin", "rbc", "mcv", "mch", "mchc", "rdw", "platelet", "wbc")
PARAMETERS = {"outer_folds": 5, "inner_folds": 5, "state_dim": 192, "hidden_dim": 128,
              "steps": 1500, "batch_size": 96, "learning_rate": 0.0003,
              "weight_decay": 0.0001, "kl_weight": 0.001, "kl_ramp_steps": 300,
              "visible_reconstruction_weight": 0.1,
              "posterior_predictive_draws": 64, "coverage_level": 0.9,
              "seed": 1701, "bootstrap_samples": 1000, "bootstrap_seed": 91501,
              "route_probabilities": {"both": 0.4, "eye_only": 0.3, "clinical_only": 0.3},
              "keep_range": [0.5, 1.0], "penalty_grid": [0.01, 0.1, 1.0, 10.0]}


class DiagnosticError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise DiagnosticError("protocol_invalid")
    return value


def validate_protocol(root: str | Path, protocol_path: str | Path) -> Mapping[str, Any]:
    """Validate static frozen settings before any patient-data import."""
    root, path = Path(root).resolve(), Path(protocol_path).resolve()
    p = _load_json(path)
    if p.get("schema_version") != PROTOCOL_SCHEMA or p.get("status") != "frozen_before_execution":
        raise DiagnosticError("protocol_not_frozen")
    if p.get("parameters") != PARAMETERS:
        raise DiagnosticError("protocol_parameters_differ")
    hashes = p.get("expected_hashes")
    required = {"run_bran_overnight_diagnostic_v1.py", "bran_patient_state_prototype_v1.py"}
    if not isinstance(hashes, Mapping) or not required <= set(hashes):
        raise DiagnosticError("protocol_hashes_missing")
    for name, expected in hashes.items():
        candidate = (root / str(name)).resolve()
        if candidate.parent != root or not isinstance(expected, str) or len(expected) != 64 or not candidate.is_file():
            raise DiagnosticError("protocol_hash_binding_invalid")
        if _sha256(candidate) != expected:
            raise DiagnosticError("protocol_hash_mismatch")
    paths = p.get("paths")
    if not isinstance(paths, Mapping) or set(paths) != {"output", "failure", "progress", "lock"}:
        raise DiagnosticError("protocol_paths_missing")
    p["_bound_paths"] = {key: (root / str(value)).resolve() for key, value in paths.items()}
    return p


@contextmanager
def _quiet_sensitive_block():
    """FD-level silence covers loaders, training libraries, and exceptions."""
    sys.stdout.flush(); sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1); os.dup2(null, 2)
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved_out, 1); os.dup2(saved_err, 2)
        os.close(null); os.close(saved_out); os.close(saved_err)


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(encoded)


def _atomic_progress(path: Path, phase: str, fold: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"status": "running", "phase": phase}
    if fold is not None:
        payload["fold"] = int(fold)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_completed(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps({"status": "completed", "phase": "completed"}, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as e:
        raise DiagnosticError("active_or_stale_lock") from e
    os.write(fd, str(os.getpid()).encode("ascii"))
    return fd


def _safe_zero(values: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masked NaNs/sentinels are excluded before numerical transforms."""
    valid = np.asarray(observed, bool) & np.isfinite(values)
    clean = np.zeros_like(values, dtype=np.float64)
    clean[valid] = np.asarray(values, dtype=np.float64)[valid]
    return clean, valid


class FoldTransform:
    def __init__(self, clinical: np.ndarray, observed: np.ndarray, eligible: np.ndarray,
                 retinal: np.ndarray, retinal_observed: np.ndarray, age: np.ndarray, train: np.ndarray):
        clean, valid = _safe_zero(clinical, observed & eligible)
        self.clinical_median = np.array([np.median(clean[train, j][valid[train, j]]) if valid[train, j].any() else 0.0
                                         for j in range(clinical.shape[1])])
        q1 = np.array([np.percentile(clean[train, j][valid[train, j]], 25) if valid[train, j].any() else 0.0
                       for j in range(clinical.shape[1])])
        q3 = np.array([np.percentile(clean[train, j][valid[train, j]], 75) if valid[train, j].any() else 1.0
                       for j in range(clinical.shape[1])])
        self.clinical_iqr = np.maximum(q3 - q1, 1e-6)
        rclean, rvalid = _safe_zero(retinal, np.broadcast_to(retinal_observed[:, None], retinal.shape))
        denom = rvalid[train].sum(axis=0).clip(1)
        self.retinal_mean = rclean[train].sum(axis=0) / denom
        self.retinal_scale = np.sqrt(((rclean[train] - self.retinal_mean) ** 2 * rvalid[train]).sum(axis=0) / denom)
        self.retinal_scale = np.maximum(self.retinal_scale, 1e-6)
        self.age_mean, self.age_scale = float(np.mean(age[train])), max(float(np.std(age[train])), 1e-6)

    def apply(self, clinical: np.ndarray, observed: np.ndarray, eligible: np.ndarray,
              retinal: np.ndarray, retinal_observed: np.ndarray, age: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        clean, valid = _safe_zero(clinical, observed & eligible)
        c = (clean - self.clinical_median) / self.clinical_iqr
        c[~valid] = 0.0
        rclean, rvalid = _safe_zero(retinal, np.broadcast_to(retinal_observed[:, None], retinal.shape))
        r = (rclean - self.retinal_mean) / self.retinal_scale
        r[~rvalid] = 0.0
        return c, valid, r, (age - self.age_mean) / self.age_scale

    def inverse_continuous(self, standardized: np.ndarray) -> np.ndarray:
        return standardized * self.clinical_iqr[:standardized.shape[-1]] + self.clinical_median[:standardized.shape[-1]]


def masked_route(rng: np.random.Generator, clinical_mask: np.ndarray, eye_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Whole-view dropout plus independent observed-item keep rates."""
    n = len(clinical_mask)
    route = rng.choice(3, size=n, p=[0.4, 0.3, 0.3])
    keep_c = rng.uniform(0.5, 1.0, size=(n, 1))
    keep_e = rng.uniform(0.5, 1.0, size=n)
    c = clinical_mask & (rng.random(clinical_mask.shape) < keep_c) & (route[:, None] != 1)
    e = eye_mask & (rng.random(n) < keep_e) & (route != 2)
    return c, e


def _torch_train(c: np.ndarray, cm: np.ndarray, r: np.ndarray, rm: np.ndarray, age: np.ndarray, train: np.ndarray, seed: int, *, steps: int = 1500):
    """Fresh fixed-step self-supervised fold model; no endpoint labels enter."""
    import torch
    from bran_patient_state_prototype_v1 import BRANPatientStatePrototypeV1, PatientStateConfig
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BRANPatientStatePrototypeV1(PatientStateConfig())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    ct, cmt = torch.tensor(c, dtype=torch.float32), torch.tensor(cm, dtype=torch.bool)
    rt, rmt, at = torch.tensor(r[:, None], dtype=torch.float32), torch.tensor(rm[:, None], dtype=torch.bool), torch.tensor(age, dtype=torch.float32)
    for step in range(steps):
        ix = rng.choice(train, size=96, replace=len(train) < 96)
        visible_c, visible_r = masked_route(rng, cm[ix], rm[ix])
        vc, vr = torch.tensor(visible_c), torch.tensor(visible_r[:, None])
        state = model(ct[ix] * vc, vc, rt[ix] * vr[..., None], vr, at[ix])
        loss = model.objective(state, at[ix], ct[ix], cmt[ix], vc,
                                           rt[ix, 0], rmt[ix].expand(-1, r.shape[1]), vr.expand(-1, r.shape[1]),
                                           kl_weight=0.001 * min(1.0, (step + 1) / 300), visible_weight=0.1)["loss"]
        if not torch.isfinite(loss):
            raise DiagnosticError("nonfinite_training")
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def encode_arms(model: Any, c: np.ndarray, cm: np.ndarray, r: np.ndarray, rm: np.ndarray, age: np.ndarray) -> Mapping[str, np.ndarray]:
    import torch
    with torch.no_grad():
        ct, rt, at = torch.tensor(c, dtype=torch.float32), torch.tensor(r[:, None], dtype=torch.float32), torch.tensor(age, dtype=torch.float32)
        def state(use_c: bool, use_r: bool) -> np.ndarray:
            s = model.encode(ct if use_c else torch.zeros_like(ct), torch.tensor(cm if use_c else np.zeros_like(cm)),
                             rt if use_r else torch.zeros_like(rt), torch.tensor(rm[:, None] if use_r else np.zeros((len(rm), 1), bool)), at)
            return s.mean.numpy()
        prototype_retinal = state(False, True)
        prototype_clinical = state(True, False)
        prototype_both = state(True, True)
    age2 = age[:, None]
    return {"age": age2, "original_retinal": np.c_[r, age2], "original_clinical": np.c_[c, age2],
            "original_concat": np.c_[r, c, age2], "prototype_retinal": np.c_[prototype_retinal, age2],
            "prototype_clinical": np.c_[prototype_clinical, age2], "prototype_both": np.c_[prototype_both, age2]}


def _fit_predict_nested(x: np.ndarray, y: np.ndarray, observed: np.ndarray, inner: np.ndarray, test_x: np.ndarray,
                        *, diagnostics: dict[str, int] | None = None) -> tuple[np.ndarray, float]:
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    warnings.filterwarnings("ignore", message="'n_jobs' has no effect", category=FutureWarning)
    def fit_checked(xfit: np.ndarray, yfit: np.ndarray, C: float):
        m = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(C=C, solver="lbfgs", max_iter=500, n_jobs=1))])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning); m.fit(xfit, yfit)
        return m, not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    choices = []
    for C in PARAMETERS["penalty_grid"]:
        aucs, losses, weights, converged = [], [], [], True
        for fold in range(5):
            fit, val = observed & (inner != fold), observed & (inner == fold)
            if fit.sum() < 2 or val.sum() < 2 or len(np.unique(y[fit])) < 2 or len(np.unique(y[val])) < 2: continue
            m, ok = fit_checked(x[fit], y[fit], C)
            if not ok: converged = False; break
            p = m.predict_proba(x[val])[:, 1]
            aucs.append(roc_auc_score(y[val], p)); losses.append(log_loss(y[val], p, labels=[0, 1])); weights.append(val.sum())
        if not converged or len(weights) != 5:
            if diagnostics is not None:
                key = "candidates_rejected_nonconvergence" if not converged else "candidates_rejected_incomplete_inner_support"
                diagnostics[key] = diagnostics.get(key, 0) + 1
            continue
        choices.append((-float(np.average(aucs, weights=weights)), float(np.average(losses, weights=weights)), float(C)))
    if not choices: raise DiagnosticError("no_complete_converged_head_candidate")
    _, _, C = min(choices)  # AUROC, then log loss, then stronger penalty (smaller C).
    if observed.sum() < 2 or len(np.unique(y[observed])) < 2: return np.full(len(test_x), np.nan), C
    m, ok = fit_checked(x[observed], y[observed], C)
    if not ok: raise DiagnosticError("selected_logistic_max_iter_exhausted")
    if diagnostics is not None:
        diagnostics["completed_head_refits"] = diagnostics.get("completed_head_refits", 0) + 1
    return m.predict_proba(test_x)[:, 1], C


def fold_weighted_auc(y: np.ndarray, p: np.ndarray, observed: np.ndarray, folds: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    values, weights = [], []
    for fold in range(5):
        m = observed & (folds == fold)
        if m.sum() and len(np.unique(y[m])) == 2:
            values.append(roc_auc_score(y[m], p[m])); weights.append(m.sum())
    return float(np.average(values, weights=weights)) if weights else float("nan")


def fold_weighted_logloss(y: np.ndarray, p: np.ndarray, observed: np.ndarray, folds: np.ndarray) -> float:
    values, weights = [], []
    for fold in range(5):
        m = observed & (folds == fold) & np.isfinite(p)
        if m.any():
            q = np.clip(p[m], 1e-6, 1 - 1e-6); values.append(float(np.mean(-(y[m] * np.log(q) + (1-y[m]) * np.log1p(-q))))) ; weights.append(m.sum())
    return float(np.average(values, weights=weights)) if weights else float("nan")


def _weighted_auc_draws(y: np.ndarray, p: np.ndarray, observed: np.ndarray, folds: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Vectorized weighted rank/U statistic with exact 0.5 tie handling."""
    draws = counts.shape[0]; numer, denom = np.zeros(draws), np.zeros(draws)
    for fold in range(5):
        ix = np.flatnonzero(observed & (folds == fold) & np.isfinite(p))
        if not len(ix): continue
        order = ix[np.argsort(p[ix], kind="mergesort")]; w = counts[:, order]
        pos, neg = w * (y[order][None] == 1), w * (y[order][None] == 0)
        lower_neg = np.zeros(draws); u = np.zeros(draws); start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and p[order[stop]] == p[order[start]]: stop += 1
            gp, gn = pos[:, start:stop].sum(1), neg[:, start:stop].sum(1)
            u += gp * (lower_neg + 0.5 * gn); lower_neg += gn; start = stop
        npos, nneg = pos.sum(1), neg.sum(1); ok = (npos > 0) & (nneg > 0)
        auc = np.full(draws, np.nan); auc[ok] = u[ok] / (npos[ok] * nneg[ok])
        nobs = (npos + nneg); good = np.isfinite(auc); numer[good] += auc[good] * nobs[good]; denom[good] += nobs[good]
    result = np.full(draws, np.nan); good = denom > 0; result[good] = numer[good] / denom[good]
    return result


def bootstrap_fold_weighted(y: np.ndarray, predictions: Mapping[str, np.ndarray], observed: np.ndarray, folds: np.ndarray,
                            draws: int = 1000, seed: int = 91501) -> Mapping[str, Any]:
    """Patient bootstrap fixed OOF predictions; no refits and no draw release."""
    rng, point = np.random.default_rng(seed), {a: fold_weighted_auc(y, p, observed, folds) for a, p in predictions.items()}
    counts = np.zeros((draws, len(y)), dtype=np.int16)
    for fold in range(5):
        ix = np.flatnonzero(folds == fold)
        for d in range(draws): counts[d, ix] = np.bincount(rng.choice(len(ix), len(ix), replace=True), minlength=len(ix))
    samples = {a: _weighted_auc_draws(y, p, observed, folds, counts) for a, p in predictions.items()}
    result = {a: {"status": "scored" if np.isfinite(point[a]) else "unscorable", "estimate": (float(point[a]) if np.isfinite(point[a]) else None),
                  "fold_weighted_logloss": (float(fold_weighted_logloss(y, predictions[a], observed, folds)) if np.isfinite(point[a]) else None),
                  "ci95": ([float(np.nanpercentile(samples[a], 2.5)), float(np.nanpercentile(samples[a], 97.5))] if np.isfinite(point[a]) else None)} for a in predictions}
    if "prototype_both" in predictions:
        result["paired_delta_prototype_both_minus_single"] = {a: {"estimate": (float(point["prototype_both"] - point[a]) if np.isfinite(point["prototype_both"]) and np.isfinite(point[a]) else None),
            "ci95": [float(np.nanpercentile(np.asarray(samples["prototype_both"]) - np.asarray(samples[a]), 2.5)),
                     float(np.nanpercentile(np.asarray(samples["prototype_both"]) - np.asarray(samples[a]), 97.5))]}
            for a in ("prototype_retinal", "prototype_clinical")}
    return result


def _inner_context(context: Mapping[str, Any], train: np.ndarray, fold: int) -> tuple[np.ndarray, str]:
    from patient_atlas_disease_universal import make_inner_fold_ids, subset_targets
    raw = context["raw_cohort"]
    targets = subset_targets(context["fold_targets"], train, tuple(raw.patient_ids[i] for i in train))
    return make_inner_fold_ids(outer_fold=fold, patient_ids=tuple(raw.patient_ids[i] for i in train),
                               site_ids=tuple(raw.site_ids[i] for i in train), targets=targets,
                               outer_fold_policy=context["fold_policy"])


def _cbc_completion_fold(model: Any, c: np.ndarray, cm: np.ndarray, r: np.ndarray, rm: np.ndarray, age: np.ndarray,
                         names: Sequence[str], train: np.ndarray, test: np.ndarray, *, seed: int) -> Mapping[str, Mapping[str, float]]:
    """All-nine-CBC hidden completion: decoder, state bridge, and Ridge reference.

    Errors are normalized-space MAE because this compact runner deliberately
    does not assert authoritative clinical units.  No values/predictions leave
    this function.
    """
    from sklearn.linear_model import Ridge
    cbc = np.asarray([tuple(names).index(name) for name in CBC_FIELDS], dtype=int)
    hidden_c, hidden_m = c.copy(), cm.copy(); hidden_c[:, cbc] = 0.0; hidden_m[:, cbc] = False
    hidden_states = encode_arms(model, hidden_c, hidden_m, r, rm, age)["prototype_both"]  # fixed state plus explicit age
    import torch
    with torch.no_grad():
        st = model.encode(torch.tensor(hidden_c, dtype=torch.float32), torch.tensor(hidden_m), torch.tensor(r[:, None], dtype=torch.float32), torch.tensor(rm[:, None]), torch.tensor(age, dtype=torch.float32))
        bare = model.predictive_mean(st, torch.tensor(age, dtype=torch.float32))["continuous"].numpy()
        torch.manual_seed(seed)
        samples = model.sample_clinical(st, torch.tensor(age, dtype=torch.float32), samples=64)
        draws = samples["continuous"].numpy()
        # Integrate zero-mean observation noise analytically for the point
        # estimate; retain it in predictive intervals.
        mc_mean = samples["continuous_mean"].numpy().mean(axis=0)
        lo, hi = np.quantile(draws, 0.05, axis=0), np.quantile(draws, 0.95, axis=0)
    keep = np.ones(c.shape[1], bool); keep[cbc] = False
    reference_x = np.c_[hidden_c[:, keep], r, age]
    result: dict[str, Mapping[str, float]] = {}
    for field, j in zip(CBC_FIELDS, cbc):
        fit, score = cm[train, j], cm[test, j]
        if int(score.sum()) < 10 or int(fit.sum()) < 2:
            result[field] = {"status": "suppressed_small_test_support"}; continue
        bridge = Ridge(alpha=1.0).fit(hidden_states[train][fit], c[train, j][fit])
        reference = Ridge(alpha=1.0).fit(reference_x[train][fit], c[train, j][fit])
        age_baseline = Ridge(alpha=1.0).fit(age[train][fit, None], c[train, j][fit])
        target = c[test, j][score]
        def scores(pred: np.ndarray, prefix: str) -> dict[str, float]:
            residual = pred[score] - target
            return {prefix + "_mae": float(np.mean(np.abs(residual))), prefix + "_mse": float(np.mean(residual ** 2))}
        result[field] = {"status": "scored_normalized_units_only", "test_support": int(score.sum()),
                         "outer_train_target_variance": float(np.var(c[train, j][fit])),
                         **scores(bare[test, j], "bare_decoder"), **scores(bridge.predict(hidden_states[test]), "state_ridge_bridge"),
                         **scores(mc_mean[test, j], "posterior_predictive_mc_mean"),
                         "posterior_predictive_coverage90": float(np.mean((target >= lo[test, j][score]) & (target <= hi[test, j][score]))),
                         "posterior_predictive_interval_width90": float(np.mean(hi[test, j][score] - lo[test, j][score])),
                         **scores(reference.predict(reference_x[test]), "ridge_reference"), **scores(age_baseline.predict(age[test, None]), "age_ridge"),
                         "median_baseline_mae": float(np.mean(np.abs(target))), "median_baseline_mse": float(np.mean(target ** 2))}
    return result


def _summarize_completion(folds: Sequence[Mapping[str, Mapping[str, float]]]) -> Mapping[str, Any]:
    """Suppress a CBC field globally if any outer test fold has <10 support."""
    result: dict[str, Any] = {}
    for field in CBC_FIELDS:
        rows = [fold[field] for fold in folds]
        if any(row.get("status") != "scored_normalized_units_only" for row in rows):
            result[field] = {"status": "suppressed_any_outer_fold_below_10"}; continue
        keys = [key for key in rows[0] if key not in {"status", "test_support"}]
        result[field] = {"status": "scored_normalized_units_only", **{key: float(np.mean([row[key] for row in rows])) for key in keys}}
    return result


def _actual_arrays(root: Path, context: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Extract no identifiers; patient arrays remain local and in-memory."""
    cohort, raw = context["feature_cohort"], context["raw_cohort"]
    from patient_atlas_real_data import _load_registry
    names, types = _load_registry(root)
    if tuple(types) != ("continuous",) * 48 + ("binary",) * 11 or not set(CBC_FIELDS) <= set(names[:48]):
        raise DiagnosticError("clinical_schema_or_cbc_names_differ")
    if len(raw.patient_ids) != 1928 or set(raw.split_labels) - {"train", "val"}:
        raise DiagnosticError("cohort_not_exact_train_validation_1928")
    eye, eye_m = np.asarray(cohort.eye_embeddings), np.asarray(cohort.eye_observed_mask, bool)
    eye_m = eye_m & np.isfinite(eye).all(axis=2)
    eye_clean = np.zeros_like(eye)
    eye_clean[eye_m] = eye[eye_m]
    mean = np.zeros((len(eye), eye.shape[2]), dtype=np.float64); present = eye_m.any(1)
    mean[present] = eye_clean[present].sum(1) / eye_m[present].sum(1, keepdims=True)
    return np.asarray(cohort.blood_values), np.asarray(cohort.blood_observed_mask, bool), np.asarray(cohort.blood_eligible_mask, bool)[None, :].repeat(len(eye), 0), mean, present, names


def run_bran_overnight_diagnostic_v1(*, project_root: str | Path, protocol_path: str | Path, output_path: str | Path,
                                     failure_path: str | Path, progress_path: str | Path, lock_path: str | Path,
                                     dataset_root: str | Path, clinical_project_root: str | Path) -> Mapping[str, Any]:
    """Execute the single local-only diagnostic; returns only aggregate-safe output."""
    root = Path(project_root).resolve(); output, failure, progress, lock = map(lambda x: Path(x).resolve(), (output_path, failure_path, progress_path, lock_path))
    if output == failure or output.exists() or failure.exists(): raise DiagnosticError("exclusive_output_required")
    fd = None; phase = "protocol"; locked = False; quiet = None
    try:
        protocol = validate_protocol(root, protocol_path)
        bound = protocol["_bound_paths"]
        if {"output": output, "failure": failure, "progress": progress, "lock": lock} != bound:
            raise DiagnosticError("runtime_paths_differ_from_frozen_protocol")
        if len(set(bound.values())) != 4 or any(path.parent != output.parent for path in bound.values()):
            raise DiagnosticError("protocol_paths_overlap_or_escape_output_directory")
        fd = _acquire_lock(lock); locked = True; _atomic_progress(progress, "validated")
        quiet = _quiet_sensitive_block(); quiet.__enter__()
        # Delayed production imports: no loader has run before this point.
        from patient_atlas_v6_2_expanded_endpoint_evaluation import (FROZEN_SUPPORT_RECEIPT_NAME, EXACT_OUTER_FOLD_HASH, EXACT_INNER_FOLD_ASSIGNMENT_SHA256,
                                                                       load_eligible_support_receipt, validate_support_against_observed)
        from run_patient_atlas_v6_2_expanded_endpoint_evaluation import _load_actual_v6_2_context
        phase = "support"; support = load_eligible_support_receipt(root / FROZEN_SUPPORT_RECEIPT_NAME, project_root=root)
        phase = "context"; context = _load_actual_v6_2_context(root=root, dataset_root=dataset_root, clinical_project_root=clinical_project_root, support=support)
        folds = np.asarray(context["outer_assignment"], int); labels, observed = context["labels_by_source"], context["observed_by_source"]
        validate_support_against_observed(support, labels, observed, folds)
        c0, cm0, elig, r0, rm0, names = _actual_arrays(root, context)
        # De novo contract: every condition-history input and target is gone;
        # five prospectively ineligible continuous slots remain ineligible.
        elig[:, 48:] = False
        all_oof = {s: {a: np.full(len(folds), np.nan) for a in ARMS} for s in support.eligible_sources}
        if len(support.eligible_sources) != 26:
            raise DiagnosticError("supported_endpoint_count_not_26")
        inner_contexts = []
        for fold in range(5):
            train = np.flatnonzero(folds != fold)
            inner, inner_hash = _inner_context(context, train, fold)
            if inner_hash != EXACT_INNER_FOLD_ASSIGNMENT_SHA256[fold]: raise DiagnosticError("inner_fold_hash_mismatch")
            inner_contexts.append((inner, inner_hash))
        inner_hashes, completion_folds = [item[1] for item in inner_contexts], []
        head_diagnostics = {"candidates_rejected_nonconvergence": 0, "candidates_rejected_incomplete_inner_support": 0, "completed_head_refits": 0}
        phase = "folds"
        for fold in range(5):
            started = time.monotonic(); _atomic_progress(progress, "training", fold)
            train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            inner, inner_hash = inner_contexts[fold]
            transform = FoldTransform(c0, cm0, elig, r0, rm0, np.asarray(context["raw_cohort"].ages), train)
            c, cm, r, age = transform.apply(c0, cm0, elig, r0, rm0, np.asarray(context["raw_cohort"].ages))
            model = _torch_train(c, cm, r, rm0, age, train, 1701 + fold); arms = encode_arms(model, c, cm, r, rm0, age)
            _atomic_progress(progress, "completion", fold)
            completion_folds.append(_cbc_completion_fold(model, c, cm, r, rm0, age, names, train, test, seed=91501 + fold))
            _atomic_progress(progress, "screening", fold)
            for source in support.eligible_sources:
                y, m = np.asarray(labels[source]), np.asarray(observed[source], bool)
                for arm in ARMS:
                    p, _ = _fit_predict_nested(arms[arm][train], y[train], m[train], inner, arms[arm][test], diagnostics=head_diagnostics)
                    all_oof[source][arm][test] = p
            _atomic_progress(progress, "fold_complete", fold)
        phase = "aggregate"; _atomic_progress(progress, "aggregate"); endpoint = {}
        for source in support.eligible_sources:
            y, m = np.asarray(labels[source]), np.asarray(observed[source], bool)
            endpoint[source] = bootstrap_fold_weighted(y, all_oof[source], m, folds)
        if any(endpoint[s][a]["status"] != "scored" for s in support.eligible_sources for a in ARMS):
            raise DiagnosticError("missing_or_nonfinite_screening_score")
        macro = {a: float(np.mean([endpoint[s][a]["estimate"] for s in support.eligible_sources])) for a in ARMS}
        report = {"schema_version": PROTOCOL_SCHEMA, "status": "completed_aggregate_only", "exploratory": True,
                  "scope": {"patient_count": 1928, "endpoint_count": len(support.eligible_sources), "official_test_loaded": False},
                  "fold_hashes": {"outer": EXACT_OUTER_FOLD_HASH, "inner": inner_hashes},
                  "protocol_sha256": _sha256(Path(protocol_path).resolve()), "code_hashes": dict(protocol["expected_hashes"]), "configuration": PARAMETERS,
                  "support_receipt_sha256": support.receipt_sha256,
                  "canonical_source_hashes": dict(context.get("source_hashes", {})),
                  "representation": {"retinal_pooling": "visible_image_mean_384_one_token", "posterior": "hierarchical_gaussian_diagonal_shared_conditionally_gaussian_private_no_lowrank_posterior", "clinical_likelihood": "gaussian_diag_plus_lowrank", "disease_supervision": False, "visible_reconstruction_auxiliary_weight": 0.1},
                  "screening": {"endpoint_results": endpoint, "macro_endpoint_mean_auc_descriptive": macro, "head_diagnostics": head_diagnostics,
                                "ci": "marginal_95_percent_patient_bootstrap_fixed_oof_no_refits"},
                  "completion": {"scenario": "all_nine_cbc_hidden_before_encoding", "fields": list(CBC_FIELDS), "metric": "normalized_mae_and_mse_no_authoritative_units_claimed", "results": _summarize_completion(completion_folds),
                                 "methods": ["bare_prototype_plugin_decoder_location", "posterior_predictive_mc_mean_and_90pct_interval_64_joint_draws_not_calibrated", "prototype_state_plus_age_ridge_bridge", "nonCBC_age_retina_ridge_reference"]},
                  "patient_rows_or_ids_serialized": False, "predictions_embeddings_models_or_draws_serialized": False}
        _exclusive_json(output, report); _atomic_completed(progress); return report
    except Exception as e:
        # Source positions only: never messages, frame locals, or row values.
        frames = []
        tb = e.__traceback__
        while tb is not None:
            filename = Path(tb.tb_frame.f_code.co_filename).resolve()
            if filename.parent == root and filename.name in locals().get("protocol", {}).get("expected_hashes", {}):
                frames.append({"file": filename.name, "line": int(tb.tb_lineno)})
            tb = tb.tb_next
        safe = {"schema_version": PROTOCOL_SCHEMA, "status": "failed", "phase": phase, "error_class": type(e).__name__,
                "code_frames": frames[-8:], "exception_text_serialized": False, "patient_rows_or_ids_serialized": False}
        if locked and not output.exists() and not failure.exists(): _exclusive_json(failure, safe)
        return safe
    finally:
        if quiet is not None:
            quiet.__exit__(None, None, None)
        if fd is not None:
            os.close(fd)
            try: lock.unlink()
            except FileNotFoundError: pass
