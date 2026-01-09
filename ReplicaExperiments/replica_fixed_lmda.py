#!/usr/bin/env python3
"""
Replica-style generalization MSE curves vs measurement ratio beta = n/m
for a Bernoulli–Gaussian (BG) prior under Gaussian measurements with AWGN.

This script plots (NO tuning across beta), for multiple fixed gamma values:
  1) Ridge (fixed gamma in GAMMAS_RIDGE) using RS-PMAP coupled fixed point
  2) LASSO (fixed gamma in GAMMAS_LASSO) using RS-PMAP coupled fixed point
  3) Optimal MMSE (BG posterior mean) using standard SE fixed point

Key correction vs the broken ridge version:
  - Ridge must be treated inside the same RS-PMAP coupled fixed point as LASSO:
        sigma_eff^2 = sigma0^2 + beta * mse
        gamma_p     = gamma + beta * E[sigma^2_local]
    with denoiser parameter lambda = gamma_p (NOT just gamma), where for ridge:
        xhat = z / (1 + lambda)
        sigma^2_local = lambda/(1+lambda)
    Omitting gamma_p coupling can make ridge blow up around beta ~ 1.

Output:
  - fixed_lambda_ridge_lasso_mmse.png
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.special import erfc
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# USER PARAMETERS (edit these)
# =============================================================================

# Fixed regularization parameters (NO tuning across beta)
# The script will plot one curve per value.
# Decade grid from 1e-5 up to 10 (inclusive), plus 2 and 5.
GAMMAS_COMMON: Tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0, 5.0, 10.0)
GAMMAS_RIDGE: Tuple[float, ...] = GAMMAS_COMMON   # ridge gamma in RS-PMAP (user knob)
GAMMAS_LASSO: Tuple[float, ...] = GAMMAS_COMMON   # lasso gamma in RS-PMAP (user knob)

# Data model parameters
RHO: float = 0.1               # sparsity of BG prior
SNR0_DB: float = 10.0          # true SNR in dB -> sets true noise variance sigma0^2

# Sweep measurement ratios beta = n/m
BETA_MIN: float = 0.5
BETA_MAX: float = 3.0
BETA_POINTS: int = 51

# Monte Carlo samples for replica expectations (only used for MMSE now)
# Ridge and LASSO use analytical/closed-form solutions.
MC_SAMPLES: int = 50_000

# Fixed point iteration controls
MAX_FP_ITERS: int = 600
TOL_FP: float = 1e-9
DAMP: float = 0.35

# Robust-solver fallback controls
# (Used when plain fixed-point iteration doesn't converge)
BROYDEN_MAX_ITERS: int = 200
BROYDEN_TOL: float = 1e-8

# Debug controls (optional)
# Run only a single curve quickly by setting env vars:
#   DEBUG_SINGLE=1 DEBUG_MODE=ridge DEBUG_GAMMA=1e-5 python replica_fixed_lmda.py
DEBUG_SINGLE: bool = os.getenv("DEBUG_SINGLE", "0").strip() == "1"
DEBUG_MODE: str = os.getenv("DEBUG_MODE", "ridge").strip().lower()   # "ridge" or "lasso"
# Backward compatible: accept DEBUG_LAMBDA too.
DEBUG_GAMMA: float = float(os.getenv("DEBUG_GAMMA", os.getenv("DEBUG_LAMBDA", "1e-5")))

# Deterministic seed
SEED: int = 12345

# Output image
# Save next to this script (independent of current working directory)
OUT_PNG: str = str(Path(__file__).with_name("fixed_lambda_ridge_lasso_mmse.png"))

# Plot controls
PLOT_MMSE: bool = True


# =============================================================================
# Configuration and validation
# =============================================================================
@dataclass(frozen=True)
class Config:
    seed: int
    rho: float
    var_nonzero: float         # chosen so Var(X)=1
    snr0_db: float
    sigma0_2: float            # true noise variance
    betas: np.ndarray
    mc_samples: int
    max_fp_iters: int
    tol_fp: float
    damp: float


def build_config() -> Config:
    if not (0.0 < RHO < 1.0):
        raise ValueError("RHO must be in (0, 1)")
    if any(g < 0.0 for g in GAMMAS_RIDGE) or any(g < 0.0 for g in GAMMAS_LASSO):
        raise ValueError("All entries in GAMMAS_RIDGE and GAMMAS_LASSO must be >= 0")
    if not (0.0 < BETA_MIN < BETA_MAX):
        raise ValueError("Require 0 < BETA_MIN < BETA_MAX")
    if BETA_POINTS < 2:
        raise ValueError("BETA_POINTS must be >= 2")
    if MC_SAMPLES <= 0:
        raise ValueError("MC_SAMPLES must be > 0")
    if MAX_FP_ITERS <= 0:
        raise ValueError("MAX_FP_ITERS must be > 0")
    if TOL_FP <= 0:
        raise ValueError("TOL_FP must be > 0")
    if not (0.0 < DAMP <= 1.0):
        raise ValueError("DAMP must be in (0, 1]")

    # BG prior with Var(X)=1 => rho * var_nonzero = 1
    var_nonzero = 1.0 / RHO

    # Noise: SNR0 dB with Var(X)=1 => sigma0^2 = 1/SNR
    snr0 = 10.0 ** (SNR0_DB / 10.0)
    sigma0_2 = 1.0 / snr0

    betas = np.linspace(BETA_MIN, BETA_MAX, BETA_POINTS)
    return Config(
        seed=SEED,
        rho=RHO,
        var_nonzero=var_nonzero,
        snr0_db=SNR0_DB,
        sigma0_2=sigma0_2,
        betas=betas,
        mc_samples=MC_SAMPLES,
        max_fp_iters=MAX_FP_ITERS,
        tol_fp=TOL_FP,
        damp=DAMP,
    )


# =============================================================================
# Utilities
# =============================================================================
def to_db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, 1e-15))


def soft_threshold(z: np.ndarray, t: float) -> np.ndarray:
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


# =============================================================================
# Analytical LASSO expectations for Bernoulli-Gaussian prior
# =============================================================================
def lasso_analytical_expectations(
    sigma_eff2: float,
    gamma_p: float,
    rho: float,
    var_nonzero: float,
) -> Tuple[float, float]:
    """
    Compute E[(x - xhat)^2] and E[sigma^2_local] = lambda * P(|z| > lambda)
    ANALYTICALLY for soft-threshold denoiser with Bernoulli-Gaussian prior.

    x ~ BG(rho, var_nonzero):
      - P(x=0) = 1 - rho
      - P(x ~ N(0, var_nonzero)) = rho

    z = x + sqrt(sigma_eff2) * v,  v ~ N(0,1)
    xhat = sign(z) * max(|z| - lambda, 0)   where lambda = gamma_p

    Returns: (mse, mean_sigma2) where mean_sigma2 = lambda * P(|z| > lambda)
    """
    sigma_eff2 = float(max(sigma_eff2, 1e-15))
    lam = float(max(gamma_p, 1e-15))
    tau = math.sqrt(sigma_eff2)  # std of noise component

    # Variance of z for each component
    # Component 0: x=0, so z ~ N(0, sigma_eff2)
    var_z0 = sigma_eff2
    tau_z0 = tau

    # Component 1: x ~ N(0, var_nonzero), so z ~ N(0, var_nonzero + sigma_eff2)
    var_z1 = var_nonzero + sigma_eff2
    tau_z1 = math.sqrt(var_z1)

    # P(|z| > lambda) for each component (using symmetry)
    # P(|N(0, var)| > lam) = 2 * P(N(0, var) > lam) = 2 * Phi_c(lam / sqrt(var)) = erfc(lam / (sqrt(2) * sqrt(var)))
    p_active_0 = float(erfc(lam / (math.sqrt(2.0) * tau_z0)))  # = 2 * Phi_c(lam/tau_z0)
    p_active_1 = float(erfc(lam / (math.sqrt(2.0) * tau_z1)))  # = 2 * Phi_c(lam/tau_z1)

    # Overall P(|z| > lambda)
    p_active = (1.0 - rho) * p_active_0 + rho * p_active_1
    mean_sigma2 = lam * p_active

    # MSE = E[(x - xhat)^2]
    # = (1 - rho) * E[xhat^2 | x=0] + rho * E[(x - xhat)^2 | x ~ N(0, var_nonzero)]

    # For component 0 (x=0):
    # E[xhat^2 | x=0] = E[(sign(z)*max(|z|-lam,0))^2] = E[(max(|z|-lam,0))^2]
    # For z ~ N(0, tau^2), by symmetry:
    # = 2 * E[(z - lam)^2 * 1_{z > lam}]
    # = 2 * tau^2 * [Phi_c(t)*(1 + t^2) - t*phi(t)]  where t = lam/tau
    t0 = lam / tau_z0
    phi_t0 = float(norm.pdf(t0))
    Phi_c_t0 = float(norm.sf(t0))  # = 1 - Phi(t0)
    mse_comp0 = 2.0 * var_z0 * (Phi_c_t0 * (1.0 + t0 * t0) - t0 * phi_t0)

    # For component 1 (x ~ N(0, var_nonzero)):
    # z = x + w where x ~ N(0, var_nonzero), w ~ N(0, sigma_eff2)
    # z ~ N(0, var_z1)
    # xhat = sign(z) * max(|z| - lam, 0)
    # E[(x - xhat)^2] = E[x^2] - 2*E[x*xhat] + E[xhat^2]
    #                 = var_nonzero - 2*E[x*xhat] + E[xhat^2]
    #
    # E[xhat^2] same formula as above but with var_z1:
    t1 = lam / tau_z1
    phi_t1 = float(norm.pdf(t1))
    Phi_c_t1 = float(norm.sf(t1))
    E_xhat2_comp1 = 2.0 * var_z1 * (Phi_c_t1 * (1.0 + t1 * t1) - t1 * phi_t1)

    # E[x * xhat] for x ~ N(0, var_nonzero), z = x + w, xhat = sign(z)*max(|z|-lam, 0)
    # By Stein's lemma and properties of soft-threshold:
    # E[x * xhat] = Cov(x, z) * E[xhat'(z)] where xhat'(z) = 1_{|z| > lam}
    # Cov(x, z) = Cov(x, x + w) = Var(x) = var_nonzero
    # E[xhat'(z)] = P(|z| > lam) = p_active_1
    # So: E[x * xhat] = var_nonzero * p_active_1
    E_x_xhat_comp1 = var_nonzero * p_active_1

    mse_comp1 = var_nonzero - 2.0 * E_x_xhat_comp1 + E_xhat2_comp1

    # Total MSE
    mse = (1.0 - rho) * mse_comp0 + rho * mse_comp1
    mse = float(max(mse, 0.0))  # ensure non-negative

    return mse, mean_sigma2


def solve_lasso_analytical(
    beta: float,
    gamma: float,
    cfg: Config,
    init_sigma_eff2: Optional[float] = None,
    init_gamma_p: Optional[float] = None,
    max_iters: int = 2000,
    tol: float = 1e-10,
    damp: float = 0.3,
) -> Tuple[float, float, float, bool, int]:
    """
    Solve LASSO RS-PMAP fixed point using ANALYTICAL expectations.
    No Monte Carlo noise -> smooth, correct curves.

    Fixed point equations:
        sigma_eff2 = sigma0^2 + beta * MSE(sigma_eff2, gamma_p)
        gamma_p = gamma + beta * mean_sigma2(sigma_eff2, gamma_p)
    """
    beta = float(beta)
    gamma = float(gamma)
    if beta <= 0:
        raise ValueError("beta must be positive")
    if gamma < 0:
        raise ValueError("gamma must be >= 0")

    sigma_eff2 = float(cfg.sigma0_2 if init_sigma_eff2 is None else max(init_sigma_eff2, cfg.sigma0_2))
    gamma_p = float(max(gamma, 1e-14) if init_gamma_p is None else max(init_gamma_p, gamma, 1e-14))

    converged = False
    iters = 0

    for it in range(max_iters):
        iters = it + 1

        mse, mean_sigma2 = lasso_analytical_expectations(sigma_eff2, gamma_p, cfg.rho, cfg.var_nonzero)

        sigma_new = cfg.sigma0_2 + beta * mse
        gamma_p_new = gamma + beta * mean_sigma2

        # Enforce physical bounds
        sigma_new = float(max(sigma_new, cfg.sigma0_2))
        gamma_p_new = float(max(gamma_p_new, gamma, 1e-14))

        # Check convergence
        delta = max(abs(sigma_new - sigma_eff2), abs(gamma_p_new - gamma_p))
        if delta < tol:
            sigma_eff2, gamma_p = sigma_new, gamma_p_new
            converged = True
            break

        # Damped update
        sigma_eff2 = (1.0 - damp) * sigma_eff2 + damp * sigma_new
        gamma_p = (1.0 - damp) * gamma_p + damp * gamma_p_new

        # Enforce bounds again
        sigma_eff2 = float(max(sigma_eff2, cfg.sigma0_2))
        gamma_p = float(max(gamma_p, gamma, 1e-14))

    # Final MSE at converged state
    mse, _ = lasso_analytical_expectations(sigma_eff2, gamma_p, cfg.rho, cfg.var_nonzero)

    return mse, sigma_eff2, gamma_p, converged, iters


def sample_bg(n: int, rng: np.random.Generator, rho: float, var_nonzero: float) -> np.ndarray:
    active = rng.random(n) < rho
    x = np.zeros(n, dtype=float)
    if active.any():
        x[active] = rng.normal(0.0, math.sqrt(var_nonzero), int(active.sum()))
    return x


# =============================================================================
# BG posterior mean (MMSE benchmark)
# =============================================================================
def bg_posterior_mean(z: np.ndarray, tau2: float, rho: float, var_nonzero: float) -> np.ndarray:
    """
    Posterior mean E[X|Z=z] for BG prior and Z = X + N(0,tau2).
    """
    tau2 = float(max(tau2, 1e-15))
    c = ((1.0 - rho) / rho) * math.sqrt((var_nonzero + tau2) / tau2)
    expo = -0.5 * z**2 * (1.0 / tau2 - 1.0 / (var_nonzero + tau2))
    expo = np.clip(expo, -700.0, 700.0)  # avoid overflow in exp
    pi = 1.0 / (1.0 + c * np.exp(expo))
    shrink = var_nonzero / (var_nonzero + tau2)
    return pi * shrink * z


# =============================================================================
# RS-PMAP coupled fixed point for Ridge and LASSO (fixed external gamma)
# =============================================================================
def rspmap_map(
    beta: float,
    gamma: float,
    mode: str,
    x_mc: np.ndarray,
    v_mc: np.ndarray,
    cfg: Config,
    sigma_eff2: float,
    gamma_p: float,
) -> Tuple[float, float, float]:
    """
    One evaluation of the coupled RS-PMAP map:
      (sigma_eff2, gamma_p) -> (sigma_new, gamma_p_new), plus the MSE at that state.
    """
    sigma_eff2 = float(max(sigma_eff2, 1e-15))
    gamma_p = float(max(gamma_p, 1e-14))
    z = x_mc + math.sqrt(sigma_eff2) * v_mc
    lam = gamma_p

    if mode == "lasso":
        xhat = soft_threshold(z, lam)
        p_act = float(np.mean(np.abs(z) > lam))
        mean_sigma2 = lam * p_act
    elif mode == "ridge":
        # Ridge has a linear denoiser, and with Var(X)=1 we can compute the MSE in closed form:
        #   xhat = inv * z, inv=1/(1+lam), z = x + sqrt(sigma_eff2) v
        #   mse = E[(x - inv(x+sqrt(s2)v))^2] = (1-inv)^2 Var(x) + inv^2 sigma_eff2
        #       = (lam/(1+lam))^2 + inv^2 sigma_eff2
        inv = 1.0 / (1.0 + lam)
        shrink = lam * inv  # = lam/(1+lam)
        mean_sigma2 = shrink
        mse = (shrink ** 2) + (inv ** 2) * sigma_eff2
        sigma_new = float(cfg.sigma0_2 + beta * mse)
        gamma_p_new = float(gamma + beta * mean_sigma2)
        return float(mse), sigma_new, gamma_p_new
    else:
        raise ValueError("mode must be 'lasso' or 'ridge'")

    mse = float(np.mean((x_mc - xhat) ** 2))
    sigma_new = float(cfg.sigma0_2 + beta * mse)
    gamma_p_new = float(gamma + beta * mean_sigma2)
    return mse, sigma_new, gamma_p_new


def solve_ridge_closed_form(beta: float, gamma: float, cfg: Config) -> Tuple[float, float, float]:
    """
    Closed-form solution for Ridge RS-PMAP fixed point when Var(X)=1.

    For ridge, mean_sigma2 = lam/(1+lam) with lam=gamma_p, and gamma_p satisfies:
        gamma_p = gamma + beta * gamma_p/(1+gamma_p)
    which reduces to the quadratic:
        gamma_p^2 + (1 - gamma - beta) gamma_p - gamma = 0
    Take the nonnegative root.

    Then sigma_eff2 is solved from:
        sigma_eff2 = sigma0^2 + beta * mse
        mse = (lam/(1+lam))^2 + (1/(1+lam))^2 * sigma_eff2
    """
    beta = float(beta)
    gamma = float(gamma)
    if beta <= 0:
        raise ValueError("beta must be positive")
    if gamma < 0:
        raise ValueError("gamma must be >= 0")

    a = 1.0 - gamma - beta
    disc = a * a + 4.0 * gamma
    gp = 0.5 * (-a + math.sqrt(disc))
    gp = float(max(gp, gamma, 0.0))

    inv = 1.0 / (1.0 + gp)
    shrink = gp * inv
    denom = 1.0 - beta * (inv ** 2)
    # denom can get very small near a transition; keep it away from 0 for numerical stability
    denom = float(np.sign(denom) * max(abs(denom), 1e-15))
    sigma_eff2 = float((cfg.sigma0_2 + beta * (shrink ** 2)) / denom)
    sigma_eff2 = float(max(sigma_eff2, cfg.sigma0_2))

    mse = float((shrink ** 2) + (inv ** 2) * sigma_eff2)
    return mse, sigma_eff2, gp


def solve_rspmap(
    beta: float,
    gamma: float,
    mode: str,
    x_mc: np.ndarray,
    v_mc: np.ndarray,
    cfg: Config,
    init_sigma_eff2: Optional[float] = None,
    init_gamma_p: Optional[float] = None,
) -> Tuple[float, float, float, bool, int]:
    """
    Coupled RS-PMAP fixed point:
        sigma_eff^2 = sigma0^2 + beta * mse
        gamma_p     = gamma + beta * E[sigma^2_local]
    where the denoiser uses lambda = gamma_p.

    modes:
      - "lasso": xhat = soft_threshold(z, lambda), mean_sigma2 = lambda * P(|z|>lambda)
      - "ridge": xhat = z/(1+lambda), mean_sigma2 = lambda/(1+lambda)
    """
    if mode not in ("lasso", "ridge"):
        raise ValueError("mode must be 'lasso' or 'ridge'")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if gamma < 0:
        raise ValueError("gamma must be >= 0")

    sigma_eff2 = float(cfg.sigma0_2 if init_sigma_eff2 is None else max(init_sigma_eff2, 1e-15))
    gamma_p = float(max(gamma, 1e-14) if init_gamma_p is None else max(init_gamma_p, 1e-14))

    converged = False
    iters = 0
    # We measure convergence by the *true* fixed-point residual:
    #   r = [sigma_new - sigma_eff2, gamma_p_new - gamma_p]
    # This is more reliable than checking successive iterates when damping is used.
    for it in range(cfg.max_fp_iters):
        iters = it + 1
        _, sigma_new, gamma_p_new = rspmap_map(beta, gamma, mode, x_mc, v_mc, cfg, sigma_eff2, gamma_p)

        if not (math.isfinite(sigma_new) and math.isfinite(gamma_p_new)):
            raise FloatingPointError(
                f"Non-finite fixed point iterate: sigma_new={sigma_new}, gamma_p_new={gamma_p_new}. "
                f"Try larger damping or check parameters."
            )

        r0 = abs(sigma_new - sigma_eff2)
        r1 = abs(gamma_p_new - gamma_p)
        if max(r0, r1) < cfg.tol_fp:
            sigma_eff2, gamma_p = sigma_new, gamma_p_new
            converged = True
            break

        sigma_eff2 = (1.0 - cfg.damp) * sigma_eff2 + cfg.damp * sigma_new
        gamma_p = (1.0 - cfg.damp) * gamma_p + cfg.damp * gamma_p_new

        # keep gamma_p away from 0 to avoid pathological numerical behavior
        gamma_p = max(gamma_p, 1e-14)

    # Final MSE evaluation at converged state
    mse, _, _ = rspmap_map(beta, gamma, mode, x_mc, v_mc, cfg, sigma_eff2, gamma_p)
    return mse, sigma_eff2, gamma_p, converged, iters


def solve_rspmap_broyden(
    beta: float,
    gamma: float,
    mode: str,
    x_mc: np.ndarray,
    v_mc: np.ndarray,
    cfg: Config,
    init_sigma_eff2: Optional[float] = None,
    init_gamma_p: Optional[float] = None,
    max_iters: int = BROYDEN_MAX_ITERS,
    tol: float = BROYDEN_TOL,
) -> Tuple[float, float, float, bool, int]:
    """
    Solve the RS-PMAP coupled fixed point by finding a root of
        r(sigma_eff2, gamma_p) = [sigma_new - sigma_eff2, gamma_p_new - gamma_p] = 0
    using a damped Broyden method (inverse-Jacobian update).

    This is used as a robustness fallback when the plain fixed-point iteration
    fails to converge (common near phase transitions and for very small gamma).
    """
    if mode not in ("lasso", "ridge"):
        raise ValueError("mode must be 'lasso' or 'ridge'")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if gamma < 0:
        raise ValueError("gamma must be >= 0")

    s2_min = float(cfg.sigma0_2)          # sigma_eff^2 = sigma0^2 + beta*mse >= sigma0^2
    gp_min = float(max(gamma, 1e-14))     # gamma_p = gamma + beta*E[sigma_local^2] >= gamma

    def eval_map(s2: float, gp: float) -> Tuple[float, float, float]:
        s2 = float(max(s2, s2_min))
        gp = float(max(gp, gp_min))
        mse, sigma_new, gp_new = rspmap_map(beta, gamma, mode, x_mc, v_mc, cfg, s2, gp)
        sigma_new = float(max(sigma_new, s2_min))
        gp_new = float(max(gp_new, gp_min))
        return mse, sigma_new, gp_new

    # Initial guess
    s2 = float(cfg.sigma0_2 if init_sigma_eff2 is None else max(init_sigma_eff2, s2_min))
    gp = float(gp_min if init_gamma_p is None else max(init_gamma_p, gp_min))

    mse, s2_new, gp_new = eval_map(s2, gp)
    r = np.array([s2_new - s2, gp_new - gp], dtype=float)
    rnorm = float(np.linalg.norm(r))

    # Inverse-Jacobian initial guess: small diagonal step scaling for stability
    B = np.diag([0.5, 0.5]).astype(float)

    best = (rnorm, mse, s2, gp, s2_new, gp_new)
    converged = rnorm < tol
    iters = 0

    for k in range(max_iters):
        iters = k + 1
        if converged:
            break

        step = -B @ r
        # Limit extreme steps (helps avoid exploding into bad regions)
        step[0] = float(np.clip(step[0], -0.9 * (s2 - s2_min), 10.0))
        step[1] = float(np.clip(step[1], -0.9 * (gp - gp_min), 10.0))

        # Backtracking line search on residual norm
        t = 1.0
        accepted = False
        for _ in range(10):
            s2_try = max(s2 + t * step[0], s2_min)
            gp_try = max(gp + t * step[1], gp_min)
            mse_try, s2n_try, gpn_try = eval_map(s2_try, gp_try)
            r_try = np.array([s2n_try - s2_try, gpn_try - gp_try], dtype=float)
            rnorm_try = float(np.linalg.norm(r_try))
            if rnorm_try <= rnorm or t <= 0.03125:
                accepted = True
                break
            t *= 0.5

        if not accepted:
            # Fall back to a tiny fixed-point step
            s2_try = max((1.0 - 0.1) * s2 + 0.1 * s2_new, s2_min)
            gp_try = max((1.0 - 0.1) * gp + 0.1 * gp_new, gp_min)
            mse_try, s2n_try, gpn_try = eval_map(s2_try, gp_try)
            r_try = np.array([s2n_try - s2_try, gpn_try - gp_try], dtype=float)
            rnorm_try = float(np.linalg.norm(r_try))

        # Track best iterate even if not converged
        if rnorm_try < best[0]:
            best = (rnorm_try, mse_try, s2_try, gp_try, s2n_try, gpn_try)

        # Broyden inverse update
        s_vec = np.array([s2_try - s2, gp_try - gp], dtype=float)
        y_vec = r_try - r
        By = B @ y_vec
        denom = float(s_vec @ By)
        if abs(denom) > 1e-14:
            u = s_vec - By
            vT = s_vec @ B
            B = B + np.outer(u, vT) / denom

        # Accept step
        s2, gp = float(s2_try), float(gp_try)
        mse, s2_new, gp_new = float(mse_try), float(s2n_try), float(gpn_try)
        r, rnorm = r_try, rnorm_try

        # Convergence check: absolute residual on the fixed point equations
        converged = rnorm < tol

    # Return the best point we found (if we didn't converge)
    if not converged:
        _, mse, s2, gp, _, _ = best

    # Recompute MSE at returned state (consistent with solve_rspmap)
    mse, _, _ = rspmap_map(beta, gamma, mode, x_mc, v_mc, cfg, s2, gp)
    return mse, float(s2), float(gp), bool(converged), int(iters)


# =============================================================================
# MMSE state-evolution fixed point
# =============================================================================
def mmse_curve_replica(betas: np.ndarray, x_mc: np.ndarray, v_mc: np.ndarray, cfg: Config) -> np.ndarray:
    out = np.zeros_like(betas, dtype=float)
    tau2 = float(cfg.sigma0_2)  # warm start across betas

    for i, beta in enumerate(betas):
        for _ in range(cfg.max_fp_iters):
            z = x_mc + math.sqrt(max(tau2, 1e-15)) * v_mc
            xhat = bg_posterior_mean(z, tau2, cfg.rho, cfg.var_nonzero)
            mmse = float(np.mean((x_mc - xhat) ** 2))
            tau2_new = cfg.sigma0_2 + float(beta) * mmse

            if abs(tau2_new - tau2) < cfg.tol_fp:
                tau2 = tau2_new
                break
            tau2 = (1.0 - cfg.damp) * tau2 + cfg.damp * tau2_new

        z = x_mc + math.sqrt(max(tau2, 1e-15)) * v_mc
        out[i] = float(np.mean((x_mc - bg_posterior_mean(z, tau2, cfg.rho, cfg.var_nonzero)) ** 2))
    return out


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    cfg = build_config()
    rng = np.random.default_rng(cfg.seed)

    ridge_gammas: Tuple[float, ...] = tuple(float(x) for x in GAMMAS_RIDGE)
    lasso_gammas: Tuple[float, ...] = tuple(float(x) for x in GAMMAS_LASSO)
    plot_mmse = PLOT_MMSE
    if DEBUG_SINGLE:
        if DEBUG_MODE not in ("ridge", "lasso"):
            raise ValueError("DEBUG_MODE must be 'ridge' or 'lasso'")
        if DEBUG_MODE == "ridge":
            ridge_gammas = (float(DEBUG_GAMMA),)
            lasso_gammas = ()
        else:
            ridge_gammas = ()
            lasso_gammas = (float(DEBUG_GAMMA),)
        plot_mmse = False

    print("=== Starting ===")
    print(f"beta grid: [{cfg.betas.min():.2f}, {cfg.betas.max():.2f}] with {cfg.betas.size} points")
    print(f"rho={cfg.rho:.3g}, Var(nonzero)={cfg.var_nonzero:.3g} => Var(X)=1")
    print(f"SNR0={cfg.snr0_db:.2f} dB => sigma0^2={cfg.sigma0_2:.6g}")
    print(f"Fixed GAMMAS_RIDGE={ridge_gammas}")
    print(f"Fixed GAMMAS_LASSO={lasso_gammas}")
    print(f"MC samples={cfg.mc_samples}")
    if DEBUG_SINGLE:
        print(f"DEBUG_SINGLE=1 (mode={DEBUG_MODE}, gamma={DEBUG_GAMMA:g})")
    print(f"PLOT_MMSE={plot_mmse}\n")

    # Monte Carlo samples (shared)
    t0 = time.time()
    x_mc = sample_bg(cfg.mc_samples, rng, cfg.rho, cfg.var_nonzero)
    v_mc = rng.normal(size=cfg.mc_samples)
    print(f"[Init] drew MC samples in {time.time() - t0:.2f}s\n")

    def solve_rspmap_with_retries(
        beta: float,
        gamma: float,
        mode: str,
        init_state: Optional[Tuple[float, float]],
    ) -> Tuple[float, float, float, bool, int]:
        """
        Try a small set of (damp, init) strategies to avoid locking onto
        unstable/high-error fixed points near phase transitions (often around beta~1).
        """
        # Ridge can be solved in closed form (no iteration, no MC noise)
        if mode == "ridge":
            mse, s2, gp = solve_ridge_closed_form(beta, gamma, cfg)
            return mse, s2, gp, True, 1

        # Try a couple of fast fixed-point attempts first
        fp_candidates: list[Tuple[Optional[Tuple[float, float]], float, int]] = []
        fp_candidates.append((init_state, cfg.damp, cfg.max_fp_iters))
        fp_candidates.append((None, cfg.damp, cfg.max_fp_iters))
        fp_candidates.append((init_state, min(cfg.damp, 0.2), max(150, cfg.max_fp_iters)))

        best: Optional[Tuple[float, float, float, bool, int]] = None

        def consider(mse: float, s2: float, gp: float, ok: bool, iters: int) -> None:
            nonlocal best
            if best is None:
                best = (mse, s2, gp, ok, iters)
                return
            best_mse, _, _, best_ok, _ = best
            if (ok and not best_ok) or (ok == best_ok and mse < best_mse):
                best = (mse, s2, gp, ok, iters)

        for st, damp, itmax in fp_candidates:
            cfg_try = Config(
                seed=cfg.seed,
                rho=cfg.rho,
                var_nonzero=cfg.var_nonzero,
                snr0_db=cfg.snr0_db,
                sigma0_2=cfg.sigma0_2,
                betas=cfg.betas,
                mc_samples=cfg.mc_samples,
                max_fp_iters=itmax,
                tol_fp=cfg.tol_fp,
                damp=damp,
            )
            mse, s2, gp, ok, iters = solve_rspmap(
                float(beta), float(gamma), mode, x_mc, v_mc, cfg_try,
                init_sigma_eff2=None if st is None else st[0],
                init_gamma_p=None if st is None else st[1],
            )
            consider(mse, s2, gp, ok, iters)
            if ok:
                # Good enough; avoid extra work.
                return mse, s2, gp, ok, iters

        # If fixed-point didn't converge, fall back to Broyden from a few inits.
        broyden_inits: list[Optional[Tuple[float, float]]] = [init_state, None]
        for st in broyden_inits:
            mse, s2, gp, ok, iters = solve_rspmap_broyden(
                float(beta), float(gamma), mode, x_mc, v_mc, cfg,
                init_sigma_eff2=None if st is None else st[0],
                init_gamma_p=None if st is None else st[1],
                max_iters=BROYDEN_MAX_ITERS,
                tol=BROYDEN_TOL,
            )
            consider(mse, s2, gp, ok, iters)
            if ok:
                return mse, s2, gp, ok, iters

        assert best is not None
        return best

    def rspmap_curve_forward(mode: str, gamma: float, betas: np.ndarray) -> np.ndarray:
        """Forward continuation over beta."""
        mse_out = np.zeros_like(betas, dtype=float)
        prev_state: Optional[Tuple[float, float]] = None
        for i, beta in enumerate(betas):
            mse, s2, gp, ok, _ = solve_rspmap_with_retries(float(beta), float(gamma), mode, prev_state)
            mse_out[i] = mse
            prev_state = (s2, gp)
        return mse_out

    def rspmap_curve_backward(mode: str, gamma: float, betas: np.ndarray) -> np.ndarray:
        """Backward continuation over beta (high to low)."""
        mse_out = np.zeros_like(betas, dtype=float)
        prev_state: Optional[Tuple[float, float]] = None
        for j, beta in enumerate(betas[::-1]):
            mse, s2, gp, ok, _ = solve_rspmap_with_retries(float(beta), float(gamma), mode, prev_state)
            mse_out[j] = mse
            prev_state = (s2, gp)
        return mse_out[::-1]

    def smooth_spikes(mse: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """
        Detect and interpolate over obvious spikes (discontinuities).
        A spike is where mse[i] >> neighbors.
        """
        mse = mse.copy()
        n = len(mse)
        for i in range(1, n - 1):
            left, mid, right = mse[i - 1], mse[i], mse[i + 1]
            local_ref = 0.5 * (left + right)
            if mid > threshold * local_ref and mid > threshold * left and mid > threshold * right:
                # Spike detected; interpolate
                mse[i] = local_ref
        return mse

    def lasso_curve_analytical(gamma: float) -> np.ndarray:
        """
        Compute LASSO curve using ANALYTICAL expectations (no Monte Carlo noise).
        Uses forward continuation with warm-starting.
        """
        print(f"[Replica] computing LASSO (analytical, fixed gamma={gamma:g}) ...")
        t1 = time.time()
        betas = cfg.betas

        mse_out = np.zeros_like(betas, dtype=float)
        prev_sigma_eff2: Optional[float] = None
        prev_gamma_p: Optional[float] = None

        for i, beta in enumerate(betas):
            mse, s2, gp, ok, iters = solve_lasso_analytical(
                float(beta), float(gamma), cfg,
                init_sigma_eff2=prev_sigma_eff2,
                init_gamma_p=prev_gamma_p,
            )
            mse_out[i] = mse
            prev_sigma_eff2, prev_gamma_p = s2, gp

            if not ok and (i == 0 or i == betas.size - 1 or (i + 1) % max(1, betas.size // 8) == 0):
                print(f"  [warn] not converged at beta={beta:.2f} (iters={iters})")

        # Progress print
        for i, beta in enumerate(betas):
            if i == 0 or (i + 1) % max(1, betas.size // 8) == 0 or i == betas.size - 1:
                print(f"  beta {i+1:3d}/{betas.size}={beta:.2f} mse={mse_out[i]:.3e}")

        print(f"[Replica] LASSO (gamma={gamma:g}) done in {time.time() - t1:.2f}s\n")
        return mse_out

    def rspmap_curve_for_lambda(mode: str, gamma: float) -> np.ndarray:
        """
        Robust curve computation:
        - Ridge: use closed-form
        - LASSO: use analytical solver (no Monte Carlo noise)
        """
        if mode == "ridge":
            print(f"[Replica] computing RIDGE (RS-PMAP, fixed gamma={gamma:g}) ...")
            t1 = time.time()
            betas = cfg.betas
            mse_out = rspmap_curve_forward(mode, gamma, betas)
            # Progress print
            for i, beta in enumerate(betas):
                if i == 0 or (i + 1) % max(1, betas.size // 8) == 0 or i == betas.size - 1:
                    print(f"  beta {i+1:3d}/{betas.size}={beta:.2f} mse={mse_out[i]:.3e}")
            print(f"[Replica] RIDGE (gamma={gamma:g}) done in {time.time() - t1:.2f}s\n")
            return mse_out
        else:
            # LASSO: use analytical solver
            return lasso_curve_analytical(gamma)

    # Curves (RS-PMAP)
    mse_ridge_by_gamma = {float(g): rspmap_curve_for_lambda("ridge", float(g)) for g in ridge_gammas}
    mse_lasso_by_gamma = {float(g): rspmap_curve_for_lambda("lasso", float(g)) for g in lasso_gammas}

    # MMSE curve (optional)
    mse_mmse = None
    if plot_mmse:
        print("[Replica] computing Optimal MMSE ...")
        t0 = time.time()
        mse_mmse = mmse_curve_replica(cfg.betas, x_mc, v_mc, cfg)
        print(f"[Replica] MMSE done in {time.time() - t0:.2f}s\n")

    # Quick sanity prints showing sensitivity to lambdas
    print("[Sanity] first/last MSE values:")
    for gam, mse_r in mse_ridge_by_gamma.items():
        print(
            f"  Ridge  (γ={gam:g}): beta={cfg.betas[0]:.2f} mse={mse_r[0]:.4g} | "
            f"beta={cfg.betas[-1]:.2f} mse={mse_r[-1]:.4g}"
        )
    for gam, mse_l in mse_lasso_by_gamma.items():
        print(
            f"  LASSO  (γ={gam:g}): beta={cfg.betas[0]:.2f} mse={mse_l[0]:.4g} | "
            f"beta={cfg.betas[-1]:.2f} mse={mse_l[-1]:.4g}"
        )
    if mse_mmse is not None:
        print(f"  MMSE : beta={cfg.betas[0]:.2f} mse={mse_mmse[0]:.4g} | beta={cfg.betas[-1]:.2f} mse={mse_mmse[-1]:.4g}")
    print()

    # Plot
    print("[Plot] saving figure ...")

    has_ridge = len(mse_ridge_by_gamma) > 0
    has_lasso = len(mse_lasso_by_gamma) > 0

    if has_ridge and has_lasso:
        fig, (ax_ridge, ax_lasso) = plt.subplots(1, 2, figsize=(14.0, 5.4), sharey=True)
    elif has_lasso:
        fig, ax_lasso = plt.subplots(1, 1, figsize=(8.0, 5.4))
        ax_ridge = None
    elif has_ridge:
        fig, ax_ridge = plt.subplots(1, 1, figsize=(8.0, 5.4))
        ax_lasso = None
    else:
        print("[Plot] No curves to plot!")
        return

    # Color gradients: same hue per method, varying lightness across gammas.
    blues = plt.get_cmap("Blues")
    greens = plt.get_cmap("Greens")

    if has_ridge and ax_ridge is not None:
        ridge_gams = sorted(mse_ridge_by_gamma.keys())
        ridge_cs = [greens(x) for x in np.linspace(0.40, 0.85, max(1, len(ridge_gams)))]
        for gam, c in zip(ridge_gams, ridge_cs):
            ax_ridge.plot(cfg.betas, to_db(mse_ridge_by_gamma[gam]), linewidth=2.2, color=c,
                          label=f"$\\gamma$={gam:g}")
        if mse_mmse is not None:
            ax_ridge.plot(cfg.betas, to_db(mse_mmse), linewidth=2.4, linestyle="-.", color="black", label="MMSE")
        ax_ridge.grid(True, linestyle=":", linewidth=1.0)
        ax_ridge.set_xlabel(r"Measurement ratio $\beta = n/m$", fontsize=16)
        ax_ridge.set_ylabel("Mean squared error (dB)", fontsize=16)
        ax_ridge.set_xlim(float(cfg.betas.min()), float(cfg.betas.max()))
        ax_ridge.set_ylim(-17.5, 15.0)
        ax_ridge.set_title("Ridge", fontsize=16)
        leg_r = ax_ridge.legend(loc="upper left", fontsize=10, frameon=True, title="Fixed $\\gamma$")
        leg_r.get_frame().set_edgecolor("black")
        leg_r.get_frame().set_linewidth(1.0)

    if has_lasso and ax_lasso is not None:
        lasso_gams = sorted(mse_lasso_by_gamma.keys())
        lasso_cs = [blues(x) for x in np.linspace(0.40, 0.85, max(1, len(lasso_gams)))]
        for gam, c in zip(lasso_gams, lasso_cs):
            ax_lasso.plot(cfg.betas, to_db(mse_lasso_by_gamma[gam]), linewidth=2.2, color=c,
                          label=f"$\\gamma$={gam:g}")
        if mse_mmse is not None:
            ax_lasso.plot(cfg.betas, to_db(mse_mmse), linewidth=2.4, linestyle="-.", color="black", label="MMSE")
        ax_lasso.grid(True, linestyle=":", linewidth=1.0)
        ax_lasso.set_xlabel(r"Measurement ratio $\beta = n/m$", fontsize=16)
        if not has_ridge:
            ax_lasso.set_ylabel("Mean squared error (dB)", fontsize=16)
        ax_lasso.set_xlim(float(cfg.betas.min()), float(cfg.betas.max()))
        ax_lasso.set_ylim(-17.5, 15.0)
        ax_lasso.set_title("LASSO", fontsize=16)
        leg_l = ax_lasso.legend(loc="upper left", fontsize=10, frameon=True, title="Fixed $\\gamma$")
        leg_l.get_frame().set_edgecolor("black")
        leg_l.get_frame().set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"[Plot] Saved: {OUT_PNG}")
    print("=== Done ===")


if __name__ == "__main__":
    main()