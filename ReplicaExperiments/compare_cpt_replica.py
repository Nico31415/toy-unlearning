#!/usr/bin/env python3
"""
Replica curves for unlearning + adversarial relearning, comparing c_PT values.

Stage 2: gen_err_forget(α_UL), gen_err_retain(α_UL)  – one curve per c_PT
Stage 3: mse_relearn(α_RL)                            – one curve per c_PT

Quantities computed here (NOTE the two distinct conventions):

  Per-active-feature errors (averaged over a single group, NOT weighted by ρ):
      err_F  = E[(β̂_UL,F  - β*_PT,F)²]   over active forget features  (Stage 2)
      err_R  = E[(β̂_UL,R  - β*_PT,R)²]   over active retain features  (Stage 2)
      err_RL_F = E[(β̂_RL,F - β*_PT,F)²]  over active forget features  (Stage 3)

  Overall MSE over ALL D coordinates (theory eq. 88, ≈ ρ_F·err_RL_F):
      mse_RL = (1/D) Σ_i (β̂_RL,i - β*_PT,i)²

  stage3_oracle_curve() returns BOTH (mse_RL, err_RL_F).  The Stage-3 figure
  plots mse_RL (overall, theory eq. 88); unlearning_relearning_replica.py plots
  the per-feature err_RL_F, so compare the two via mse_RL ≈ ρ_F·err_RL_F.

Per-active-feature reference values:
    Perfect forgetting  → err_F     = var_nz = 1/ρ_PT  (≈ 10)
    Perfect relearning  → err_RL_F ≈ σ₀²               (dangerous)
    Failed  relearning  → err_RL_F  = var_nz            (safe)
    (overall mse_RL is ≈ ρ_F× smaller throughout)

Physics:
    Small c_PT  (k = 4c² ≈ 0, L1-like, "dangerous"):
        α_c_RL ≈ ρ_FT ≪ 1 → adversary relearns with tiny data
    Large c_PT  (k ≫ 0, L2-like, "safe"):
        α_c_RL ≈ 1         → adversary needs many samples to relearn

================================================================================
METHOD: ORACLE CALIBRATION  (replica_derivation.pdf sec. 8.3 / 9.4 / 11)
================================================================================
This script solves each stage with the *oracle-gp* method: γ* (= gp) is fixed
externally by a calibration condition (Ψ_UL(γ*) = α·v_nz for Stage 2, eq. 64;
Ψ_RL(γ*) = (1-α)·v_nz for Stage 3, eq. 86) and ONLY s² is iterated (eqs. 65, 87).

IMPORTANT (theory warning box, sec. 8.3):  this is a heuristic interpolation
rule, NOT the Onsager self-consistency condition γ = α·E[Σ_qk].  The resulting
curves are a parametric family indexed by γ*, not true GAMP/replica fixed points.
The companion script unlearning_relearning_replica.py instead iterates the FULL
PMAP (both s² and gp), so the two scripts are NOT expected to coincide exactly.

Stage 3 here also uses the DETERMINISTIC-CENTRE approximation (sec. 9.1 warning):
the Bregman centre is set to its mean (0_F, β*_R, 0) rather than the random β̂_UL,
which is exact only in the α_UL → ∞ (perfect-unlearning) limit.  For the
principled finite-α_UL nested treatment (sec. 9.2, eqs. 69–83) see
nested_cascade_replica.py.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

_EPS_K = 1e-30


# ──────────────────────────────────────────────────────────────────────────────
# Effective geometry after pretraining + layer rescaling
#
# From zero_func.pdf Theorem 2 (eq 38-39), balanced-pathway case
# (w^+ = v^+, w^- = v^- at pretrain init → pretrained λ_e invariant = 0):
#
#   c_PT,i (pretrained invariant) = sqrt(c_PT² - λ_PT²)  [same for all coords]
#   k_i^FT = (α²_in + α²_out)² · (c_PT² - λ_PT²)
#           + (α²_in - α²_out)² · β̂²_PT,i
#
# Special cases:
#   α_in = α_out = 1, λ_PT = 0  →  k = 4 c_PT²   (current default)
#   α_in = α_out = a            →  k = 4a⁴(c_PT²-λ_PT²)  [β-independent]
#   α_in ≠ α_out                →  k depends on β̂_PT,i  [per-coord]
# ──────────────────────────────────────────────────────────────────────────────

def compute_k_eff(
    beta_pt: np.ndarray,
    c_PT: float,
    lambda_PT: float = 0.0,
    alpha_in: float = 1.0,
    alpha_out: float = 1.0,
) -> np.ndarray:
    """Per-coordinate k_i^FT after pretraining (c_PT, lambda_PT) + Mode-B rescaling."""
    ain2 = alpha_in ** 2
    aout2 = alpha_out ** 2
    A = (ain2 + aout2) ** 2 * (c_PT ** 2 - lambda_PT ** 2)
    B = (ain2 - aout2) ** 2
    return A + B * np.asarray(beta_pt, dtype=float) ** 2


def compute_c_ft_reset(
    beta_pt: np.ndarray,
    c_PT: float,
    lambda_PT: float = 0.0,
    gamma_reinit: float = 0.0,
    input_scale: float = 1.0,
) -> np.ndarray:
    """Mode-A reset geometry: per-coord c_ft after readout reinit-to-γ + input averaging.

    Cosyne mapping (= compute_c_ft_from_pt in fixed_lambda_all.py at input_scale=1),
    generalised with a single input-layer scale s (output is reinitialised to γ, NOT
    scaled — so there is no α_out):

        c_ft_i = s²·(λ_PT + c_PT)·(1 + sqrt(1 + (β_PT,i/c_PT)²)) + γ_reinit²/2

    The s²·(λ+c)(1+√(1+(β/c)²)) term is the (arithmetic-)averaged input contribution
    w̄² ∝ w⁺²+w⁻², which is β-DEPENDENT (it grows with pathway imbalance, hence with
    |β_PT|), scaled by the input multiplier s².  The +γ²/2 term is the readout reinit
    to constant γ.  k_i = 4·c_ft_i².  (At λ_PT=0 this is still β-dependent via the
    √(1+(β/c)²) factor — averaging does NOT remove the pretrained dependence.)
    """
    c = float(c_PT)
    if c <= 0.0:
        raise ValueError(f"c_PT must be > 0 for the reset geometry, got {c}")
    beta_pt = np.asarray(beta_pt, dtype=float)
    ratio_sq = (beta_pt / c) ** 2
    s2 = float(input_scale) ** 2
    return s2 * (lambda_PT + c) * (1.0 + np.sqrt(1.0 + ratio_sq)) + 0.5 * float(gamma_reinit) ** 2


def compute_ft_geometry(
    beta_pt: np.ndarray,
    c_PT: float,
    lambda_PT: float = 0.0,
    alpha_in: float = 1.0,
    alpha_out: float = 1.0,
    gamma_reinit: float | None = None,
    average_inputs: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Fine-tuning Bregman geometry: returns (k_all, center, reset).

    Two regimes (zero_func.tex):

      * Mode B only — rescaling, NO reset (gamma_reinit is None and not average_inputs):
            center_i = s·β_PT,i,  s = alpha_in·alpha_out      (Thm 2 warm-start center)
            k_i      = compute_k_eff(β_PT, c_PT, λ_PT, α_in, α_out)

      * Mode A — RESET (reinit readout to γ AND average inputs):
            center_i = 0   (reset ⇒ predictor starts at zero: averaging makes
                            w⁺=w⁻, so reinit gives γw̄ − γw̄ = 0)
            k_i      = 4·compute_c_ft_reset(β_PT, c_PT, λ_PT, γ, input_scale=s)²
            The single input-scale s = alpha_in (the output is reinitialised to γ,
            NOT scaled, so alpha_out plays no role and is ignored in this branch).
            k stays β-DEPENDENT (arithmetic averaging keeps w⁺²+w⁻², which grows
            with |β_PT|); averaging does not flatten the pretrained dependence.

    Mode A pairs the two operations; requesting only one raises (single-operation
    geometry is not specified in the local theory).
    """
    beta_pt = np.asarray(beta_pt, dtype=float)
    reinit = gamma_reinit is not None
    if reinit != bool(average_inputs):
        raise NotImplementedError(
            "Mode A requires BOTH readout reinit (gamma_reinit set) and input "
            "averaging (average_inputs=True); the single-operation geometry is "
            "not specified in zero_func.tex."
        )

    if reinit and average_inputs:
        s_in = alpha_in  # single input-layer scale; output is reinit to γ (not scaled)
        if abs(alpha_out - 1.0) > 1e-12:
            warnings.warn(
                f"Mode A reset: the output layer is reinitialised to γ (not scaled), "
                f"so alpha_out={alpha_out} is ignored.  Use alpha_in as the single "
                f"input-scale s.",
                RuntimeWarning, stacklevel=2,
            )
        c_ft = compute_c_ft_reset(beta_pt, c_PT, lambda_PT, gamma_reinit, input_scale=s_in)
        k_all = 4.0 * c_ft ** 2
        center = np.zeros_like(beta_pt)
        return k_all, center, True

    # Mode B (rescaling only, no reset)
    k_all = compute_k_eff(beta_pt, c_PT, lambda_PT, alpha_in, alpha_out)
    center = (alpha_in * alpha_out) * beta_pt
    return k_all, center, False


# ──────────────────────────────────────────────────────────────────────────────
# Bregman proximal operator — supports scalar OR per-coord array k
# ──────────────────────────────────────────────────────────────────────────────

def _prox_qk_newton(v: np.ndarray, lam: float, k,
                    n_iter: int = 80, tol: float = 1e-12) -> np.ndarray:
    """Hybrid Newton+bisection prox_{lam*q_k}(v), safe for per-coord k arrays."""
    lam   = float(max(lam, 1e-14))
    k     = np.maximum(np.asarray(k, dtype=float), _EPS_K)
    coeff = lam / 2.0
    # Bracket: root of F(x) = x + coeff*arcsinh(2x/√k) - v lies in [min(v,0), max(v,0)]
    lo = np.minimum(v, 0.0)
    hi = np.maximum(v, 0.0)
    x  = 0.5 * (lo + hi)
    for _ in range(n_iter):
        t    = 2.0 * x / np.sqrt(k)
        r    = x + coeff * np.arcsinh(t) - v
        drdx = 1.0 + 2.0 * coeff / np.sqrt(k + 4.0 * x * x)  # avoids t² overflow
        x_new = x - r / drdx
        neg = r < 0
        lo  = np.where(neg, x, lo)
        hi  = np.where(~neg, x, hi)
        bad = (x_new < lo) | (x_new > hi) | ~np.isfinite(x_new)
        x   = np.where(bad, 0.5 * (lo + hi), x_new)
        if float(np.max(np.abs(r))) < tol:
            break
    return x


def sigma2_qk_local(xstar, lam: float, k) -> np.ndarray:
    """Local variance σ²_qk = lam / (1 + lam·q''_k(x)) = lam·√(k+4x²)/(√(k+4x²)+lam).

    Equivalent to lam·dx̂/dz at the prox solution; in (0, lam].  k scalar or array.
    """
    lam = float(max(lam, 1e-14))
    sq  = np.sqrt(np.maximum(np.asarray(k, dtype=float), _EPS_K)
                  + 4.0 * np.asarray(xstar, dtype=float) ** 2)
    return lam * sq / (sq + lam)


def bregman_prox(z: np.ndarray, lam: float, k,
                 beta0: np.ndarray) -> np.ndarray:
    """prox_{λ D_{q_k}(·, β₀)}(z).  k may be scalar or per-coord array."""
    lam  = float(max(lam, 1e-14))
    k    = np.maximum(np.asarray(k, dtype=float), _EPS_K)
    sqk  = np.sqrt(k)
    shift = lam * 0.5 * np.arcsinh(2.0 * np.asarray(beta0, dtype=float) / sqk)
    return _prox_qk_newton(np.asarray(z, dtype=float) + shift, lam, k)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: oracle-gp SE
#
# Physical insight: at D→∞ with exact forget labels y_F=0, the Bregman
# minimiser gives err_F = α·var_nz (linear in α, independent of k).
# We find the unique gp(α,k) such that
#     E_{β~N(0,var_nz)}[(prox_{gp·q_k(·-β)}(0) − β)²] = α·var_nz
# and then evaluate the full SE at that fixed gp.
# ──────────────────────────────────────────────────────────────────────────────

def _oracle_gp(alpha: float, betas: np.ndarray, k,
               var_nz: float, s: float = 1.0) -> float | None:
    """Return gp s.t. E[(bregman_prox(0,gp,k,s·β)−β)²] = α·var_nz.

    betas: pre-sampled forget-group β*_PT values (shape n_mc)
    k: per-coord array or scalar (computed from beta_PT via compute_k_eff)
    s: rescaling factor s = alpha_in * alpha_out; center = s * betas
    """
    target = alpha * var_nz
    center = s * betas

    def err_at(gp: float) -> float:
        return float(np.mean((bregman_prox(np.zeros_like(betas), gp, k, center) - betas) ** 2))

    # err_at(gp) is decreasing in gp: e_lo ≈ Ψ_UL(0⁺) = v_nz (or (s-1)²v_nz for
    # s≠1 at the gp→∞ end) and e_hi ≈ Ψ_UL(∞).  If the target α·v_nz lies
    # outside [e_hi, e_lo] the calibration is infeasible (see Lemma 8.1); this
    # happens for α≥1, or for s≠1 at small α.  Return None → caller emits NaN.
    e_lo = err_at(1e-8)
    e_hi = err_at(1e4)
    if target >= e_lo or target <= e_hi:
        return None
    return float(brentq(lambda g: err_at(g) - target, 1e-8, 1e4,
                        xtol=1e-5, rtol=1e-5))


def _oracle_gp_rl(alpha_rl: float, betas: np.ndarray, k,
                   var_nz: float) -> float:
    """Return gp s.t. E[(bregman_prox(β*_PT,gp,k,0)−β*_PT)²] = (1−α_rl)·var_nz.

    betas: pre-sampled forget-group values; k: scalar or per-coord array.
    err_at is INCREASING in gp:
      gp→0: prox→β*_PT → err→0 (adversary recovers)
      gp→∞: prox→0    → err→var_nz (adversary fails)
    """
    target = (1.0 - alpha_rl) * var_nz

    def err_at(gp: float) -> float:
        xhat = bregman_prox(betas, gp, k, np.zeros_like(betas))
        return float(np.mean((xhat - betas) ** 2))

    e_lo = err_at(1e-8)
    e_hi = err_at(1e4)
    if target <= e_lo:
        return 1e-8
    if target >= e_hi:
        return 1e4
    return float(brentq(lambda g: err_at(g) - target, 1e-8, 1e4,
                        xtol=1e-5, rtol=1e-5))


def stage3_oracle_curve(
    alphas: np.ndarray,
    rho_pt: float,
    p_forget: float,
    sigma0_sq: float,
    c_PT: float,
    lambda_PT: float = 0.0,
    alpha_in: float = 1.0,
    alpha_out: float = 1.0,
    n_mc: int = 20_000,
    seed: int = 42,
    max_se_iters: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stage 3 oracle-gp SE curve.

    Supports per-coord k via (c_PT, lambda_PT, alpha_in, alpha_out).
    Center after perfect unlearning: 0 for F coords, β*_PT for R coords.

    Returns
    -------
    mse_RL    : overall MSE (1/D)Σ_i(β̂_RL,i - β*_PT,i)² over ALL coords
                (theory eq. 88); this is what the Stage-3 figure plots.
    err_RL_F  : per-active-forget-feature error E[(β̂_RL,F - β*_PT,F)²]
                (theory eq. 89 inner expectation; mse_RL ≈ ρ_F·err_RL_F).
                Matches the quantity plotted by unlearning_relearning_replica.py.
    """
    var_nz = 1.0 / rho_pt
    rho_ft = rho_pt * p_forget
    rho_rt = rho_pt * (1.0 - p_forget)
    s = alpha_in * alpha_out

    rng  = np.random.default_rng(seed)
    n_f  = int(round(n_mc * rho_ft))
    n_r  = int(round(n_mc * rho_rt))
    n_0  = n_mc - n_f - n_r

    bF = rng.normal(0.0, math.sqrt(var_nz), n_f)
    bR = rng.normal(0.0, math.sqrt(var_nz), n_r)

    # Stage 3: target = β*_PT, center = β̂_UL (≈0 for F, ≈β*_PT for R)
    beta_eff = np.concatenate([bF, bR, np.zeros(n_0)])
    beta_ctr = np.concatenate([np.zeros(n_f), bR, np.zeros(n_0)])

    # Per-coord k based on β*_PT values (the relearning geometry)
    k_all = compute_k_eff(beta_eff, c_PT, lambda_PT, alpha_in, alpha_out)
    k_F   = k_all[:n_f]

    mse_RL_out   = np.zeros(len(alphas))   # overall MSE over all D coords (eq. 88)
    err_RL_F_out = np.zeros(len(alphas))   # per-active-forget-feature error (eq. 89)

    for i, alpha in enumerate(alphas):
        if alpha >= 1.0:
            # For α≥1: oracle target (1-α)·var_nz ≤ 0 is unphysical;
            # use gp→0 (near-identity prox, adversary relearns).
            # With gp→0 the prox ≈ identity, so MSE ≈ s² and SE1
            # s² = (σ₀² + MSE)/α gives the noise-floor FP s²* = σ₀²/(α-1)
            # for α > 1.  At exactly α = 1 there is no finite fixed point
            # (the equation reduces to σ₀² = 0); the small but nonzero
            # gp = 1e-8 provides a touch of shrinkage so the iteration
            # still settles numerically.
            gp = 1e-8
        else:
            # Oracle uses forget-group betas to find gp targeting (1-α)·var_nz
            gp = _oracle_gp_rl(alpha, bF, k_F, var_nz)
        if gp is None:
            mse_RL_out[i]   = float("nan")
            err_RL_F_out[i] = float("nan")
            continue

        s2 = sigma0_sq
        v  = rng.standard_normal(n_mc)
        for _ in range(max_se_iters):
            z    = beta_eff + math.sqrt(max(s2, 1e-20)) * v
            xhat = bregman_prox(z, gp, k_all, beta_ctr)
            mse  = float(np.mean((xhat - beta_eff) ** 2))
            s2_new = (sigma0_sq + mse) / alpha
            if abs(s2_new - s2) < 1e-6 * s2:
                break
            s2 = 0.9 * s2 + 0.1 * s2_new
            s2 = max(s2, sigma0_sq / alpha)

        v    = rng.standard_normal(n_mc)
        z    = beta_eff + math.sqrt(max(s2, 1e-20)) * v
        xhat = bregman_prox(z, gp, k_all, beta_ctr)
        mse_RL_out[i]   = float(np.mean((xhat - beta_eff) ** 2))          # overall (eq. 88)
        err_RL_F_out[i] = float(np.mean((xhat[:n_f] - bF) ** 2))          # per-forget-feature (eq. 89)

    return mse_RL_out, err_RL_F_out


def stage2_oracle_curve(
    alphas: np.ndarray,
    rho_pt: float,
    p_forget: float,
    sigma0_sq: float,
    c_PT: float,
    lambda_PT: float = 0.0,
    alpha_in: float = 1.0,
    alpha_out: float = 1.0,
    t_forget: float = 0.0,
    n_mc: int = 20_000,
    seed: int = 42,
    max_se_iters: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stage 2 oracle-gp SE curve with per-coord k.

    Supports any (c_PT, lambda_PT, alpha_in, alpha_out) combination.
    Center = s·β*_PT  (rescaled pretraining weights, s = alpha_in·alpha_out).
    Target = t_forget·β*_PT_F for forget group, β*_PT_R for retain group.

    Returns (err_F, err_R) arrays over alphas.
    """
    var_nz = 1.0 / rho_pt
    rho_ft = rho_pt * p_forget
    rho_rt = rho_pt * (1.0 - p_forget)
    s = alpha_in * alpha_out

    # Calibration feasibility (theory Lemma 8.1 / sec. 11.4.1): the Stage-2
    # oracle condition Ψ_UL(γ*) = α·v_nz has a guaranteed solution only for
    # s = 1.  For s ≠ 1, Ψ_UL(∞) = (s-1)²·v_nz > 0, so the target α·v_nz can
    # fall below the achievable range and _oracle_gp() returns None → NaN curve.
    if abs(s - 1.0) > 1e-12:
        warnings.warn(
            f"stage2_oracle_curve called with s = alpha_in*alpha_out = {s:.4g} "
            f"!= 1 (alpha_in={alpha_in}, alpha_out={alpha_out}). The Stage-2 "
            f"oracle calibration is only guaranteed feasible for s = 1 "
            f"(Lemma 8.1); expect NaNs at small alpha where alpha*v_nz < "
            f"(s-1)^2*v_nz = {(s - 1.0) ** 2 * var_nz:.4g}.",
            RuntimeWarning,
            stacklevel=2,
        )

    rng  = np.random.default_rng(seed)
    n_f  = int(round(n_mc * rho_ft))
    n_r  = int(round(n_mc * rho_rt))
    n_0  = n_mc - n_f - n_r

    bF = rng.normal(0.0, math.sqrt(var_nz), n_f)
    bR = rng.normal(0.0, math.sqrt(var_nz), n_r)

    # Stage 2: target = [t_forget·bF, bR, 0], center = s·[bF, bR, 0]
    beta_pt  = np.concatenate([bF, bR, np.zeros(n_0)])
    beta_eff = np.concatenate([t_forget * bF, bR, np.zeros(n_0)])
    beta_ctr = s * beta_pt

    # Per-coord k from β*_PT values (the unlearning geometry after rescaling)
    k_all = compute_k_eff(beta_pt, c_PT, lambda_PT, alpha_in, alpha_out)
    k_F   = k_all[:n_f]

    err_F_out   = np.zeros(len(alphas))
    err_R_out   = np.zeros(len(alphas))
    mse_ul_out  = np.zeros(len(alphas))   # overall MSE vs unlearning target β_eff_UL

    for i, alpha in enumerate(alphas):
        # Oracle: find gp targeting α·var_nz on forget-group coords (center = s·β*_F)
        gp = _oracle_gp(alpha, bF, k_F, var_nz, s=s)
        if gp is None:
            err_F_out[i] = float("nan")
            err_R_out[i] = float("nan")
            mse_ul_out[i] = float("nan")
            continue

        s2 = sigma0_sq
        v  = rng.standard_normal(n_mc)
        for _ in range(max_se_iters):
            z    = beta_eff + math.sqrt(max(s2, 1e-20)) * v
            xhat = bregman_prox(z, gp, k_all, beta_ctr)
            mse  = float(np.mean((xhat - beta_eff) ** 2))
            s2_new = (sigma0_sq + mse) / alpha
            if abs(s2_new - s2) < 1e-6 * s2:
                break
            s2 = 0.9 * s2 + 0.1 * s2_new
            s2 = max(s2, sigma0_sq / alpha)

        v    = rng.standard_normal(n_mc)
        z    = beta_eff + math.sqrt(max(s2, 1e-20)) * v
        xhat = bregman_prox(z, gp, k_all, beta_ctr)
        err_F_out[i]  = float(np.mean((xhat[:n_f] - bF) ** 2))
        err_R_out[i]  = float(np.mean((xhat[n_f:n_f + n_r] - bR) ** 2))
        mse_ul_out[i] = float(np.mean((xhat - beta_eff) ** 2))  # MSE vs β_eff_UL (decreases with α)

    return err_F_out, err_R_out, mse_ul_out


def stage2_pmap_curve(
    alphas: np.ndarray,
    rho_pt: float,
    p_forget: float,
    sigma0_sq: float,
    c_PT: float,
    lambda_PT: float = 0.0,
    alpha_in: float = 1.0,
    alpha_out: float = 1.0,
    t_forget: float = 0.0,
    n_mc: int = 40_000,
    seed: int = 42,
    gamma_ext: float = 0.0,
    gamma_reinit: float | None = None,
    average_inputs: bool = False,
    max_se_iters: int = 2000,
    damp: float = 0.25,
    tol_fp: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 2 FULL PMAP — iterates BOTH s² and gp to self-consistency.

    Uses the D/N (beta = 1/alpha) additive convention, identical to the
    Stage-1 solver solve_rspmap_qk_one() in fixed_lambda_all.py:

        beta    = D/N = 1/alpha
        s²_new  = sigma0_sq + beta * MSE                   (floor sigma0_sq)
        gp_new  = gamma_ext + beta * mean(sigma2_i)        (floor gamma_ext)
        sigma2_i = gp / (1 + gp * q''_i),  q''_i = 1/sqrt(k_i + 4 x_i²)

    NOTE: gp scales as beta=1/alpha (NOT alpha).  As alpha->0 this drives
    gp->inf so the prox collapses onto the Bregman center (no data => stay at
    the prior), which is the physically correct branch and keeps s² bounded.
    Using alpha*mean_sigma2 instead lets gp->0, the prox -> identity,
    MSE -> s², and s²_new = MSE/alpha diverges (NaN).

    Supports per-coord k (alpha_in != alpha_out).  Returns (err_F, err_R,
    mse_ul) over alphas.
    """
    var_nz = 1.0 / rho_pt
    rho_ft = rho_pt * p_forget
    rho_rt = rho_pt * (1.0 - p_forget)

    rng  = np.random.default_rng(seed)
    n_f  = int(round(n_mc * rho_ft))
    n_r  = int(round(n_mc * rho_rt))
    n_0  = n_mc - n_f - n_r

    bF = rng.normal(0.0, math.sqrt(var_nz), n_f)
    bR = rng.normal(0.0, math.sqrt(var_nz), n_r)
    v  = rng.standard_normal(n_mc)

    beta_pt  = np.concatenate([bF, bR, np.zeros(n_0)])
    beta_eff = np.concatenate([t_forget * bF, bR, np.zeros(n_0)])

    # Geometry + Bregman center: Mode B (rescaling, center = s*beta_PT) or
    # Mode A reset (reinit+average, center = 0).  See compute_ft_geometry.
    k_all, beta_ctr, _reset = compute_ft_geometry(
        beta_pt, c_PT, lambda_PT, alpha_in, alpha_out,
        gamma_reinit=gamma_reinit, average_inputs=average_inputs)

    err_F_out  = np.zeros(len(alphas))
    err_R_out  = np.zeros(len(alphas))
    mse_ul_out = np.zeros(len(alphas))

    S2_CAP = 1e14   # hard ceiling: prevents sqrt(s2)*v overflow if a stray iterate runs

    # Sweep ascending in alpha so the first (smallest-alpha) solve sits in the
    # robust "stay near the center" basin; warm-start each subsequent alpha.
    prev_state = None
    for i, alpha in enumerate(alphas):
        beta = 1.0 / alpha

        if prev_state is None:
            # Non-trivial warm start: s² from the prior (alpha=0) MSE so the
            # iteration avoids the s²=0 collapse; gp large (= prior local
            # variance * beta) so the prox starts strongly shrinking.
            prior_mse = float(np.mean((beta_eff - beta_ctr) ** 2))
            s2 = sigma0_sq + beta * prior_mse
            sig2_ctr = sigma2_qk_local(beta_ctr, 1.0, k_all)   # in (0,1]
            gp = max(gamma_ext + beta * float(np.mean(sig2_ctr)), 1e-14)
        else:
            s2, gp = prev_state
            s2 = max(s2, sigma0_sq)
            gp = max(gp, gamma_ext, 1e-14)

        for _ in range(max_se_iters):
            s2_eff = min(max(s2, 1e-20), S2_CAP)
            z    = beta_eff + math.sqrt(s2_eff) * v
            xhat = bregman_prox(z, gp, k_all, beta_ctr)

            mse = float(np.mean((xhat - beta_eff) ** 2))
            mean_sigma2 = float(np.mean(sigma2_qk_local(xhat, gp, k_all)))

            s2_new = sigma0_sq + beta * mse
            gp_new = gamma_ext + beta * mean_sigma2

            if max(abs(s2_new - s2), abs(gp_new - gp)) < tol_fp:
                s2, gp = s2_new, gp_new
                break

            s2 = (1.0 - damp) * s2 + damp * s2_new
            gp = (1.0 - damp) * gp + damp * gp_new
            s2 = min(max(s2, sigma0_sq, 1e-20), S2_CAP)
            gp = max(gp, gamma_ext, 1e-14)

        s2_eff = min(max(s2, 1e-20), S2_CAP)
        z    = beta_eff + math.sqrt(s2_eff) * v
        xhat = bregman_prox(z, gp, k_all, beta_ctr)
        err_F_out[i]  = float(np.mean((xhat[:n_f] - bF) ** 2))
        err_R_out[i]  = float(np.mean((xhat[n_f:n_f+n_r] - bR) ** 2))
        mse_ul_out[i] = float(np.mean((xhat - beta_eff) ** 2))
        prev_state = (s2, gp)

    return err_F_out, err_R_out, mse_ul_out


def stage3_pmap_curve(
    alphas: np.ndarray,
    rho_pt: float,
    p_forget: float,
    sigma0_sq: float,
    c_PT: float,
    lambda_PT: float = 0.0,
    alpha_in: float = 1.0,
    alpha_out: float = 1.0,
    n_mc: int = 40_000,
    seed: int = 42,
    gamma_ext: float = 0.0,
    gamma_reinit: float | None = None,
    average_inputs: bool = False,
    max_se_iters: int = 2000,
    damp: float = 0.25,
    tol_fp: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stage 3 (adversarial relearning) FULL PMAP — iterates BOTH s² and gp.

    Uses the corrected (1/alpha) self-consistency convention (same as
    stage2_pmap_curve; see project memory project_pmap_gp_convention):

        beta    = D/N = 1/alpha_RL
        s²_new  = sigma0_sq + beta * MSE
        gp_new  = gamma_ext + beta * mean(sigma2_qk)

    Deterministic-centre approximation (matches stage3_oracle_curve so the two
    are directly comparable): the Bregman centre is the MEAN unlearned model
    β̂_UL ≈ (0_F, β*_R, 0) rather than the random Stage-2 output.  For the
    principled finite-alpha_UL nested treatment see nested_cascade_replica.py.

        target = β*_PT     = (β_F, β_R, 0)   (what the adversary recovers)
        centre = β̂_UL      ≈ (0_F, β_R, 0)   (perfectly-unlearned model)

    Physics: at alpha_RL→0 the estimate sits at the centre, so x̂_F≈0 and
    err_RL_F≈var_nz (relearning FAILED, safe).  As alpha_RL grows the data pull
    x̂_F→β*_PT,F and err_RL_F→0 (relearning SUCCEEDS, dangerous).  Small c_PT
    (L1-like) relearns at tiny alpha_RL; large c_PT (L2-like) needs alpha_RL≈1.

    Returns
    -------
    mse_RL    : overall MSE (1/D)Σ_i(x̂_i - β*_PT,i)² over ALL coords (eq. 88)
    err_RL_F  : per-active-forget-feature error E[(x̂_F - β*_PT,F)²] (eq. 89);
                mse_RL ≈ ρ_F·err_RL_F.  Matches unlearning_relearning_replica.py.
    """
    var_nz = 1.0 / rho_pt
    rho_ft = rho_pt * p_forget
    rho_rt = rho_pt * (1.0 - p_forget)

    rng  = np.random.default_rng(seed)
    n_f  = int(round(n_mc * rho_ft))
    n_r  = int(round(n_mc * rho_rt))
    n_0  = n_mc - n_f - n_r

    bF = rng.normal(0.0, math.sqrt(var_nz), n_f)
    bR = rng.normal(0.0, math.sqrt(var_nz), n_r)
    v  = rng.standard_normal(n_mc)

    beta_eff = np.concatenate([bF, bR, np.zeros(n_0)])            # target = β*_PT
    beta_ctr = np.concatenate([np.zeros(n_f), bR, np.zeros(n_0)]) # centre = β̂_UL

    # Per-coord k from β*_PT (the relearning geometry), CONSERVED from UL start —
    # so it reflects whatever Mode A/B was applied after pretraining.  The Stage-3
    # Bregman centre is the unlearned model β̂_UL (not the geometry helper's center).
    k_all, _ctr_unused, _reset = compute_ft_geometry(
        beta_eff, c_PT, lambda_PT, alpha_in, alpha_out,
        gamma_reinit=gamma_reinit, average_inputs=average_inputs)

    mse_RL_out   = np.zeros(len(alphas))
    err_RL_F_out = np.zeros(len(alphas))

    S2_CAP = 1e14

    prev_state = None
    for i, alpha in enumerate(alphas):
        beta = 1.0 / alpha

        if prev_state is None:
            # Warm start in the robust "stay at the centre" basin (small alpha_RL).
            prior_mse = float(np.mean((beta_eff - beta_ctr) ** 2))
            s2 = sigma0_sq + beta * prior_mse
            sig2_ctr = sigma2_qk_local(beta_ctr, 1.0, k_all)
            gp = max(gamma_ext + beta * float(np.mean(sig2_ctr)), 1e-14)
        else:
            s2, gp = prev_state
            s2 = max(s2, sigma0_sq)
            gp = max(gp, gamma_ext, 1e-14)

        for _ in range(max_se_iters):
            s2_eff = min(max(s2, 1e-20), S2_CAP)
            z    = beta_eff + math.sqrt(s2_eff) * v
            xhat = bregman_prox(z, gp, k_all, beta_ctr)

            mse = float(np.mean((xhat - beta_eff) ** 2))
            mean_sigma2 = float(np.mean(sigma2_qk_local(xhat, gp, k_all)))

            s2_new = sigma0_sq + beta * mse
            gp_new = gamma_ext + beta * mean_sigma2

            if max(abs(s2_new - s2), abs(gp_new - gp)) < tol_fp:
                s2, gp = s2_new, gp_new
                break

            s2 = (1.0 - damp) * s2 + damp * s2_new
            gp = (1.0 - damp) * gp + damp * gp_new
            s2 = min(max(s2, sigma0_sq, 1e-20), S2_CAP)
            gp = max(gp, gamma_ext, 1e-14)

        s2_eff = min(max(s2, 1e-20), S2_CAP)
        z    = beta_eff + math.sqrt(s2_eff) * v
        xhat = bregman_prox(z, gp, k_all, beta_ctr)
        mse_RL_out[i]   = float(np.mean((xhat - beta_eff) ** 2))
        err_RL_F_out[i] = float(np.mean((xhat[:n_f] - bF) ** 2))
        prev_state = (s2, gp)

    return mse_RL_out, err_RL_F_out


# ──────────────────────────────────────────────────────────────────────────────
# MC sampling
# ──────────────────────────────────────────────────────────────────────────────

def make_mc_samples(
    rho_pt: float,
    p_forget: float,
    n_mc: int,
    seed: int,
    t_forget: float = 0.0,
    t_retain: float = 1.0,
):
    rng = np.random.default_rng(seed)
    sigma_nz = math.sqrt(1.0 / rho_pt)

    n_F = int(round(n_mc * rho_pt * p_forget))
    n_R = int(round(n_mc * rho_pt * (1.0 - p_forget)))
    n_0 = n_mc - n_F - n_R

    beta_F = rng.normal(0.0, sigma_nz, n_F)
    beta_R = rng.normal(0.0, sigma_nz, n_R)
    beta_0 = np.zeros(n_0)

    beta_pt = np.concatenate([beta_F, beta_R, beta_0])
    eff_ul  = np.concatenate([t_forget * beta_F, t_retain * beta_R, beta_0])
    eff_rl  = beta_pt.copy()   # adversary wants to recover β*_PT

    group = np.concatenate([
        np.ones(n_F, dtype=int),
        2 * np.ones(n_R, dtype=int),
        np.zeros(n_0, dtype=int),
    ])

    v_ul = rng.standard_normal(n_mc)
    v_rl = rng.standard_normal(n_mc)

    return beta_pt, eff_ul, eff_rl, v_ul, v_rl, group


# ──────────────────────────────────────────────────────────────────────────────
# Main comparison
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── configuration ────────────────────────────────────────────────────────
    rho_pt    = 0.1
    p_forget  = 0.5           # forget fraction of active dims
    rho_ft    = rho_pt * p_forget   # = 0.05
    var_nz    = 1.0 / rho_pt        # = 10 (variance of active coefficients)
    sigma0_sq = 0.01
    n_mc      = 80_000
    seed      = 2024

    # α sweeps — extend past 1 to see k-dependent behaviour
    alpha_ul  = np.concatenate([np.linspace(0.02, 0.98, 30),
                                 np.linspace(1.02, 3.0, 20)])
    alpha_rl  = np.concatenate([np.linspace(0.02, 0.98, 30),
                                 np.linspace(1.02, 3.0, 20)])

    c_pt_list = [1e-3, 1e-1, 1.0]

    fp_kw = dict(max_iters=800, tol=1e-10, damp=0.25, gamma_ext=1e-9)

    out_dir = Path(__file__).parent / "compare_cpt_figures"
    out_dir.mkdir(exist_ok=True)

    # ── draw one set of MC samples (same β*_PT for all c_PT) ─────────────────
    beta_pt, eff_ul, eff_rl, v_ul, v_rl, group = make_mc_samples(
        rho_pt, p_forget, n_mc, seed
    )
    mask_F = group == 1   # active forget features
    mask_R = group == 2   # active retain features

    print(f"MC: n_mc={n_mc}, n_F={mask_F.sum()}, n_R={mask_R.sum()}")
    print(f"var_nz={var_nz:.2f}, rho_ft={rho_ft:.3f}, sigma0_sq={sigma0_sq}")

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2: Unlearning  (oracle-gp SE)
    # ─────────────────────────────────────────────────────────────────────────
    alpha_ul_s2 = alpha_ul[alpha_ul <= 1.0]   # restrict to α ≤ 1

    stage2 = {}   # c_pt → (err_F, err_R)
    for c_pt in c_pt_list:
        k_scalar = 4.0 * c_pt ** 2   # default: lambda_PT=0, alpha_in=alpha_out=1
        print(f"\n=== Stage 2: c_PT={c_pt}  k={k_scalar:.2e} ===")

        err_F, err_R, mse_ul = stage2_oracle_curve(
            alpha_ul_s2, rho_pt, p_forget, sigma0_sq,
            c_PT=c_pt, lambda_PT=0.0, alpha_in=1.0, alpha_out=1.0,
            n_mc=40_000, seed=seed,
        )
        stage2[c_pt] = (err_F, err_R, mse_ul)

        for i in [0, 5, 10, 20, len(alpha_ul_s2) - 1]:
            if i < len(alpha_ul_s2):
                print(f"  α={alpha_ul_s2[i]:.3f}  err_F={err_F[i]:.4f}  err_R={err_R[i]:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3: Relearning from fully-unlearned model
    #   center_RL,F = β̂_UL,F ≈ β_eff_UL,F = 0  (perfect forgetting)
    #   target_RL   = β*_PT  (adversary wants to recover pretrained teacher)
    # ─────────────────────────────────────────────────────────────────────────
    # We use the analytically known UL endpoint:
    #   β̂_UL,F ≈ 0 (forget erased), β̂_UL,R ≈ β*_PT,R (retain preserved)
    center_rl_analytic = np.concatenate([
        np.zeros(mask_F.sum()),
        beta_pt[mask_R],
        np.zeros((group == 0).sum()),
    ])

    stage3 = {}   # c_pt → mse array (overall MSE vs β*_PT)
    for c_pt in c_pt_list:
        k_scalar = 4.0 * c_pt ** 2
        print(f"\n=== Stage 3: c_PT={c_pt}  k={k_scalar:.2e} ===")

        mse_rl, err_rl_f = stage3_oracle_curve(
            alpha_rl, rho_pt, p_forget, sigma0_sq,
            c_PT=c_pt, lambda_PT=0.0, alpha_in=1.0, alpha_out=1.0,
            n_mc=40_000, seed=seed,
        )
        stage3[c_pt] = mse_rl   # overall MSE (eq. 88) — plotted in Figure 2

        for i in [0, 5, 10, 20, 29, 35, 45]:
            if i < len(alpha_rl):
                print(f"  α={alpha_rl[i]:.3f}  mse={mse_rl[i]:.4f}  err_F={err_rl_f[i]:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Extra sweeps: lambda_PT and (alpha_in, alpha_out) at fixed c_PT=0.1
    # ─────────────────────────────────────────────────────────────────────────
    c_fixed = 0.1

    # Sweep 1: vary lambda_PT, equal scaling
    lambda_configs = [
        dict(label=r"$\lambda_{\rm PT}=0$",      lambda_PT=0.0,    alpha_in=1.0, alpha_out=1.0),
        dict(label=r"$\lambda_{\rm PT}=0.5c$",   lambda_PT=0.05,   alpha_in=1.0, alpha_out=1.0),
        dict(label=r"$\lambda_{\rm PT}=0.95c$",  lambda_PT=0.095,  alpha_in=1.0, alpha_out=1.0),
    ]

    # Sweep 2: vary (alpha_in, alpha_out), lambda_PT=0
    scale_configs = [
        dict(label=r"$\alpha_{\rm in}=\alpha_{\rm out}=1$  ($s=1$)",   alpha_in=1.0, alpha_out=1.0),
        dict(label=r"$\alpha_{\rm in}=\alpha_{\rm out}=2$  ($s=4$)",   alpha_in=2.0, alpha_out=2.0),
        dict(label=r"$\alpha_{\rm in}=\alpha_{\rm out}=0.5$ ($s=0.25$)", alpha_in=0.5, alpha_out=0.5),
        dict(label=r"$\alpha_{\rm in}=2,\,\alpha_{\rm out}=0.5$ ($s=1$, unequal)", alpha_in=2.0, alpha_out=0.5),
    ]

    # Extra sweeps show oracle region only (α ≤ 1); PMAP with heterogeneous k is
    # unreliable and slow, so we cap at α = 1.
    alpha_sweep = alpha_rl[alpha_rl <= 1.0]

    def _run_stage3(lam, ain, aout):
        # Figures 3 & 4 plot the overall MSE (eq. 88); discard the per-feature err.
        mse_rl, _ = stage3_oracle_curve(
            alpha_sweep, rho_pt, p_forget, sigma0_sq,
            c_PT=c_fixed, lambda_PT=lam, alpha_in=ain, alpha_out=aout,
            n_mc=20_000, seed=seed, max_se_iters=300,
        )
        return mse_rl

    print(f"\n=== Lambda sweep (c_PT={c_fixed}) ===")
    lam_results = []
    for cfg in lambda_configs:
        print(f"  {cfg['label']} ...", flush=True)
        lam_results.append(_run_stage3(cfg["lambda_PT"], cfg["alpha_in"], cfg["alpha_out"]))

    print(f"\n=== Scale sweep (c_PT={c_fixed}, lambda_PT=0) ===")
    scale_results = []
    for cfg in scale_configs:
        print(f"  {cfg['label']} ...", flush=True)
        scale_results.append(_run_stage3(0.0, cfg["alpha_in"], cfg["alpha_out"]))

    # ─────────────────────────────────────────────────────────────────────────
    # Figures
    # ─────────────────────────────────────────────────────────────────────────
    cmap   = plt.cm.plasma
    colors = [cmap(v) for v in [0.15, 0.50, 0.85]]

    # --- Figure 1: Stage 2 unlearning ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        fr"Stage 2 — Unlearning   "
        fr"($\rho_{{\rm PT}}={rho_pt}$, $p_f={p_forget}$, "
        fr"$\sigma_0^2={sigma0_sq}$)",
        fontsize=11
    )

    for col, c_pt in zip(colors, c_pt_list):
        err_F, err_R, mse_ul = stage2[c_pt]
        lbl = fr"$c_{{\rm PT}}={c_pt:.0e}$"
        ax1.plot(alpha_ul_s2, err_F, '-o', ms=2.5, color=col, label=lbl)
        ax2.plot(alpha_ul_s2, err_R, '-o', ms=2.5, color=col, label=lbl)

    ax1.axhline(var_nz, color='k', ls='--', lw=0.8, label=f'$1/\\rho_{{PT}}={var_nz}$ (perfect forget)')
    ax1.set_xlabel(r'$\alpha_{\rm UL} = N_{\rm UL}/D$')
    ax1.set_ylabel(r'$\mathcal{E}_F$ (per active forget feature)')
    ax1.set_title('Forget group generalization error')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.5, var_nz + 1])

    ax2.axhline(0, color='k', ls='--', lw=0.8, label='Perfect retain ($=0$)')
    ax2.set_xlabel(r'$\alpha_{\rm UL} = N_{\rm UL}/D$')
    ax2.set_ylabel(r'$\mathcal{E}_R$ (per active retain feature)')
    ax2.set_title('Retain group generalization error')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = out_dir / "stage2_unlearning.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {p}")

    # --- Figure 2: Stage 3 relearning ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        fr"Stage 3 — Adversarial Relearning   "
        fr"($\rho_{{\rm PT}}={rho_pt}$, $p_f={p_forget}$)",
        fontsize=11
    )

    for col, c_pt in zip(colors, c_pt_list):
        lbl = fr"$c_{{\rm PT}}={c_pt:.0e}$"
        ax.plot(alpha_rl, stage3[c_pt], '-o', ms=2.5, color=col, label=lbl)

    ax.set_xlabel(r'$\alpha_{\rm RL} = N_{\rm RL}/D$')
    ax.set_ylabel(r'$\frac{1}{D}\|\hat\beta - \beta^*\|^2$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "stage3_relearning.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")


    # --- Figure 3: Stage 3 — lambda_PT sweep ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        fr"Stage 3 — Effect of $\lambda_{{\rm PT}}$   "
        fr"($c_{{\rm PT}}={c_fixed}$, $\alpha_{{\rm in}}=\alpha_{{\rm out}}=1$)",
        fontsize=11,
    )
    lam_colors = [plt.cm.viridis(v) for v in [0.15, 0.50, 0.85]]
    for col, cfg, res in zip(lam_colors, lambda_configs, lam_results):
        ax.plot(alpha_sweep, res, '-o', ms=2.5, color=col, label=cfg["label"])
    ax.set_xlabel(r'$\alpha_{\rm RL} = N_{\rm RL}/D$')
    ax.set_ylabel(r'$\frac{1}{D}\|\hat\beta - \beta^*\|^2$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "stage3_lambda_sweep.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")

    # --- Figure 4: Stage 3 — (alpha_in, alpha_out) sweep ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        fr"Stage 3 — Effect of layer rescaling   "
        fr"($c_{{\rm PT}}={c_fixed}$, $\lambda_{{\rm PT}}=0$)",
        fontsize=11,
    )
    scale_colors = [plt.cm.cool(v) for v in [0.1, 0.4, 0.7, 0.95]]
    for col, cfg, res in zip(scale_colors, scale_configs, scale_results):
        ax.plot(alpha_sweep, res, '-o', ms=2.5, color=col, label=cfg["label"])
    ax.set_xlabel(r'$\alpha_{\rm RL} = N_{\rm RL}/D$')
    ax.set_ylabel(r'$\frac{1}{D}\|\hat\beta - \beta^*\|^2$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "stage3_scale_sweep.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
