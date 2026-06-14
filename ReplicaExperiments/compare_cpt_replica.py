#!/usr/bin/env python3
"""
Replica curves for unlearning + adversarial relearning, comparing c_PT values.

Stage 2: gen_err_forget(α_UL), gen_err_retain(α_UL)  – one curve per c_PT
Stage 3: gen_err_relearn(α_RL)                         – one curve per c_PT

Key quantities (per-active-feature generalization error):
    err_F  = E[(β̂_UL,F  - β*_PT,F)²]   over active forget features
    err_RL = E[(β̂_RL,F  - β*_PT,F)²]   over active forget features

Perfect forgetting  → err_F = var_nz = 1/ρ_PT  (≈ 10)
Perfect relearning  → err_RL ≈ σ₀²             (dangerous)
Failed  relearning  → err_RL = var_nz           (safe)

Physics:
    Small c_PT  (k = 4c² ≈ 0, L1-like, "dangerous"):
        α_c_RL ≈ ρ_FT ≪ 1 → adversary relearns with tiny data
    Large c_PT  (k ≫ 0, L2-like, "safe"):
        α_c_RL ≈ 1         → adversary needs many samples to relearn
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent))
from fixed_lambda_all import prox_qk_safeguarded, sigma2_qk

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


def sigma2_qk_arr(xstar: np.ndarray, lam: float, k: np.ndarray) -> np.ndarray:
    """q_k'' curvature term; k may be a per-coord array."""
    lam = float(max(lam, 1e-14))
    k = np.maximum(np.asarray(k, dtype=float), _EPS_K)
    qpp = 1.0 / np.sqrt(k + 4.0 * np.asarray(xstar, dtype=float) ** 2)
    return 1.0 / (1.0 / lam + qpp)


# ──────────────────────────────────────────────────────────────────────────────
# Bregman proximal operator — supports scalar OR per-coord array k
# ──────────────────────────────────────────────────────────────────────────────

def _prox_qk_newton(v: np.ndarray, lam: float, k,
                    n_iter: int = 80) -> np.ndarray:
    lam   = float(max(lam, 1e-14))
    k     = np.maximum(np.asarray(k, dtype=float), _EPS_K)
    sqk   = np.sqrt(k)
    coeff = lam / 2.0
    x = v / (1.0 + 2.0 * coeff / sqk)
    for _ in range(n_iter):
        t    = 2.0 * x / sqk
        r    = x + coeff * np.arcsinh(t) - v
        drdx = 1.0 + coeff * 2.0 / sqk / np.sqrt(1.0 + t ** 2)
        x    = x - r / drdx
    return x


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
               var_nz: float) -> float | None:
    """Return gp s.t. E[(bregman_prox(0,gp,k,β)−β)²] = α·var_nz.

    betas: pre-sampled forget-group values (shape n_mc)
    k: per-coord array or scalar (computed from beta_PT via compute_k_eff)
    """
    target = alpha * var_nz

    def err_at(gp: float) -> float:
        return float(np.mean((bregman_prox(np.zeros_like(betas), gp, k, betas) - betas) ** 2))

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
) -> np.ndarray:
    """
    Stage 3 oracle-gp SE curve for α ≤ 1.

    Supports per-coord k via (c_PT, lambda_PT, alpha_in, alpha_out).
    Center after perfect unlearning: 0 for F coords, β*_PT for R coords.
    Returns err_RL_F array over alphas.
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

    err_RL_out = np.zeros(len(alphas))

    for i, alpha in enumerate(alphas):
        # Oracle uses forget-group betas to find gp targeting (1-α)·var_nz
        gp = _oracle_gp_rl(alpha, bF, k_F, var_nz)

        s2 = sigma0_sq
        v  = rng.standard_normal(n_mc)
        for _ in range(max_se_iters):
            z    = beta_eff + math.sqrt(max(s2, 1e-20)) * v
            xhat = bregman_prox(z, gp, k_all, beta_ctr)
            mse  = float(np.mean((xhat - beta_eff) ** 2))
            s2_new = sigma0_sq + alpha * mse
            if abs(s2_new - s2) < 1e-6 * s2:
                break
            s2 = 0.9 * s2 + 0.1 * s2_new
            s2 = max(s2, sigma0_sq)

        v    = rng.standard_normal(n_mc)
        z    = beta_eff + math.sqrt(max(s2, 1e-20)) * v
        xhat = bregman_prox(z, gp, k_all, beta_ctr)
        err_RL_out[i] = float(np.mean((xhat[:n_f] - bF) ** 2))

    return err_RL_out


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

    err_F_out = np.zeros(len(alphas))
    err_R_out = np.zeros(len(alphas))

    for i, alpha in enumerate(alphas):
        # Oracle: find gp targeting α·var_nz on forget-group coords
        gp = _oracle_gp(alpha, bF, k_F, var_nz)
        if gp is None:
            err_F_out[i] = float("nan")
            err_R_out[i] = float("nan")
            continue

        s2 = sigma0_sq
        v  = rng.standard_normal(n_mc)
        for _ in range(max_se_iters):
            z    = beta_eff + math.sqrt(max(s2, 1e-20)) * v
            xhat = bregman_prox(z, gp, k_all, beta_ctr)
            mse  = float(np.mean((xhat - beta_eff) ** 2))
            s2_new = sigma0_sq + alpha * mse
            if abs(s2_new - s2) < 1e-6 * s2:
                break
            s2 = 0.9 * s2 + 0.1 * s2_new
            s2 = max(s2, sigma0_sq)

        v    = rng.standard_normal(n_mc)
        z    = beta_eff + math.sqrt(max(s2, 1e-20)) * v
        xhat = bregman_prox(z, gp, k_all, beta_ctr)
        err_F_out[i] = float(np.mean((xhat[:n_f] - bF) ** 2))
        err_R_out[i] = float(np.mean((xhat[n_f:n_f + n_r] - bR) ** 2))

    return err_F_out, err_R_out


# ──────────────────────────────────────────────────────────────────────────────
# RS-PMAP fixed point
# ──────────────────────────────────────────────────────────────────────────────

def solve_fp(
    alpha: float,
    target_mc: np.ndarray,
    center_mc: np.ndarray,
    v_mc: np.ndarray,
    k: float,
    sigma0_sq: float,
    gamma_ext: float = 1e-9,
    max_iters: int = 800,
    tol: float = 1e-10,
    damp: float = 0.25,
    init_state=None,
):
    """
    Self-consistent RS-PMAP fixed point with Bregman center.

    Returns (xhat, mse_target, s2, gp)
    where mse_target = E[(target - xhat)²].
    """
    alpha = float(alpha)
    k = np.maximum(np.asarray(k, dtype=float), 1e-30)

    if init_state is None:
        prior_mse = float(np.mean((target_mc - center_mc) ** 2))
        s2 = sigma0_sq + alpha * prior_mse
        s2 = max(s2, 1e-6)
        mean_sig2 = float(np.mean(sigma2_qk_arr(center_mc, 1.0, k)))
        gp = gamma_ext + alpha * mean_sig2
        gp = max(gp, 1e-14)
    else:
        s2 = max(float(init_state[0]), sigma0_sq, 1e-15)
        gp = max(float(init_state[1]), gamma_ext, 1e-14)

    for _ in range(max_iters):
        z = target_mc + math.sqrt(max(s2, 1e-15)) * v_mc
        xhat = bregman_prox(z, gp, k, center_mc)

        mse = float(np.mean((target_mc - xhat) ** 2))
        ms2 = float(np.mean(sigma2_qk_arr(xhat, gp, k)))

        s2_new = float(sigma0_sq + alpha * mse)
        gp_new = float(gamma_ext + alpha * ms2)

        if max(abs(s2_new - s2), abs(gp_new - gp)) < tol:
            return xhat, mse, s2_new, gp_new

        s2 = (1 - damp) * s2 + damp * s2_new
        gp = (1 - damp) * gp + damp * gp_new
        s2 = max(s2, sigma0_sq, 1e-15)
        gp = max(gp, gamma_ext, 1e-14)

    z = target_mc + math.sqrt(max(s2, 1e-15)) * v_mc
    xhat = bregman_prox(z, gp, k, center_mc)
    mse = float(np.mean((target_mc - xhat) ** 2))
    return xhat, mse, s2, gp


def solve_curve(
    alphas: np.ndarray,
    target_mc: np.ndarray,
    center_mc: np.ndarray,
    v_mc: np.ndarray,
    k: float,
    sigma0_sq: float,
    mask_eval: np.ndarray,
    beta_ref_mc: np.ndarray,
    pick: str = "min",
    **fp_kw,
):
    """
    Compute gen_err = E[(xhat - beta_ref)²] on mask_eval, for each alpha.
    Uses forward + backward sweeps.

    pick='min': lower target-MSE branch  (trivial FP, xhat→target)
    pick='max': higher target-MSE branch (nontrivial FP, xhat→center)

    Stage 2 (unlearning) uses pick='max': at small α the data is scarce, so the
    Bregman minimizer stays close to the center (β*_PT) rather than jumping to
    the target.  The max-MSE branch tracks this physically correct nontrivial FP.
    Stage 3 (relearning) uses pick='min'.
    """
    n = len(alphas)

    # Forward sweep
    state = None
    xhat_f = np.zeros((n, len(target_mc)))
    for i, a in enumerate(alphas):
        xhat, _, s2, gp = solve_fp(a, target_mc, center_mc, v_mc, k,
                                    sigma0_sq, init_state=state, **fp_kw)
        state = (s2, gp)
        xhat_f[i] = xhat

    # Backward sweep
    state = None
    xhat_b_rev = np.zeros((n, len(target_mc)))
    for j, a in enumerate(alphas[::-1]):
        xhat, _, s2, gp = solve_fp(a, target_mc, center_mc, v_mc, k,
                                    sigma0_sq, init_state=state, **fp_kw)
        state = (s2, gp)
        xhat_b_rev[j] = xhat
    xhat_b = xhat_b_rev[::-1]

    # Pointwise: pick branch according to 'pick'
    gen_err = np.zeros(n)
    for i in range(n):
        mse_f = float(np.mean((target_mc - xhat_f[i]) ** 2))
        mse_bwd = float(np.mean((target_mc - xhat_b[i]) ** 2))
        if pick == "max":
            xhat = xhat_b[i] if mse_bwd > mse_f else xhat_f[i]
        else:  # "min"
            xhat = xhat_b[i] if mse_bwd < mse_f else xhat_f[i]
        if mask_eval.any():
            gen_err[i] = float(np.mean((xhat[mask_eval] - beta_ref_mc[mask_eval]) ** 2))

    return gen_err


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

        err_F, err_R = stage2_oracle_curve(
            alpha_ul_s2, rho_pt, p_forget, sigma0_sq,
            c_PT=c_pt, lambda_PT=0.0, alpha_in=1.0, alpha_out=1.0,
            n_mc=40_000, seed=seed,
        )
        stage2[c_pt] = (err_F, err_R)

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

    # split α_RL sweep: oracle for ≤1, PMAP for >1
    alpha_rl_s3 = alpha_rl[alpha_rl <= 1.0]
    alpha_rl_hi = alpha_rl[alpha_rl > 1.0]

    stage3 = {}   # c_pt → err_RL (full alpha_rl array)
    for c_pt in c_pt_list:
        k_scalar = 4.0 * c_pt ** 2
        print(f"\n=== Stage 3: c_PT={c_pt}  k={k_scalar:.2e} ===")

        # Per-coord k for the PMAP region (uses beta_pt MC samples)
        k_mc = compute_k_eff(beta_pt, c_pt, lambda_PT=0.0,
                             alpha_in=1.0, alpha_out=1.0)

        # oracle region α ≤ 1
        err_RL_lo = stage3_oracle_curve(
            alpha_rl_s3, rho_pt, p_forget, sigma0_sq,
            c_PT=c_pt, lambda_PT=0.0, alpha_in=1.0, alpha_out=1.0,
            n_mc=40_000, seed=seed,
        )

        # PMAP region α > 1 (trivial FP unstable here)
        if len(alpha_rl_hi) > 0:
            err_RL_hi = solve_curve(
                alpha_rl_hi, eff_rl, center_rl_analytic, v_rl, k_mc, sigma0_sq,
                mask_F, beta_pt, **fp_kw
            )
        else:
            err_RL_hi = np.array([])

        err_RL = np.concatenate([err_RL_lo, err_RL_hi])
        stage3[c_pt] = err_RL

        for i in [0, 5, 10, 20, 29, 35, 45]:
            if i < len(alpha_rl):
                print(f"  α={alpha_rl[i]:.3f}  err_RL_F={err_RL[i]:.4f}")

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
        return stage3_oracle_curve(
            alpha_sweep, rho_pt, p_forget, sigma0_sq,
            c_PT=c_fixed, lambda_PT=lam, alpha_in=ain, alpha_out=aout,
            n_mc=20_000, seed=seed, max_se_iters=300,
        )

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
        err_F, err_R = stage2[c_pt]
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

    ax.axhline(var_nz, color='k', ls='--', lw=0.8,
               label=fr'Signal variance $1/\rho_{{PT}}={var_nz}$ (adversary fails)')
    ax.axhline(sigma0_sq, color='gray', ls=':', lw=0.8,
               label=fr'$\sigma_0^2={sigma0_sq}$ (perfect relearning)')

    ax.set_xlabel(r'$\alpha_{\rm RL} = N_{\rm RL}/D$')
    ax.set_ylabel(r'$\mathcal{E}_{\rm RL,F}$ (per active forget feature)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, var_nz + 1])
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
    ax.axhline(var_nz, color='k', ls='--', lw=0.8,
               label=fr'$1/\rho_{{PT}}={var_nz}$ (adversary fails)')
    ax.axhline(sigma0_sq, color='gray', ls=':', lw=0.8,
               label=fr'$\sigma_0^2={sigma0_sq}$ (perfect relearning)')
    ax.set_xlabel(r'$\alpha_{\rm RL} = N_{\rm RL}/D$')
    ax.set_ylabel(r'$\mathcal{E}_{\rm RL,F}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, var_nz + 1])
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
    ax.axhline(var_nz, color='k', ls='--', lw=0.8,
               label=fr'$1/\rho_{{PT}}={var_nz}$ (adversary fails)')
    ax.axhline(sigma0_sq, color='gray', ls=':', lw=0.8,
               label=fr'$\sigma_0^2={sigma0_sq}$ (perfect relearning)')
    ax.set_xlabel(r'$\alpha_{\rm RL} = N_{\rm RL}/D$')
    ax.set_ylabel(r'$\mathcal{E}_{\rm RL,F}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, var_nz + 1])
    plt.tight_layout()
    p = out_dir / "stage3_scale_sweep.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
