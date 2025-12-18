#!/usr/bin/env python3
"""
Replica-style generalization MSE curves vs measurement ratio beta = n/m
for a Bernoulli–Gaussian (BG) prior under Gaussian measurements with AWGN.

This script plots (NO tuning across beta):
  1) Ridge (fixed gamma = LAMBDA_RIDGE) using RS-PMAP coupled fixed point
  2) LASSO (fixed gamma = LAMBDA_LASSO) using RS-PMAP coupled fixed point
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
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# USER PARAMETERS (edit these)
# =============================================================================

# Fixed regularization parameters (NO tuning across beta)
LAMBDA_RIDGE: float = 1e-1     # ridge "gamma" in RS-PMAP (user knob)
LAMBDA_LASSO: float = 5e-2     # lasso "gamma" in RS-PMAP (user knob)

# Data model parameters
RHO: float = 0.1               # sparsity of BG prior
SNR0_DB: float = 10.0          # true SNR in dB -> sets true noise variance sigma0^2

# Sweep measurement ratios beta = n/m
BETA_MIN: float = 0.5
BETA_MAX: float = 3.0
BETA_POINTS: int = 51

# Monte Carlo samples for replica expectations (increase for smoother curves)
MC_SAMPLES: int = 40_000

# Fixed point iteration controls
MAX_FP_ITERS: int = 250
TOL_FP: float = 1e-9
DAMP: float = 0.35

# Deterministic seed
SEED: int = 12345

# Output image
OUT_PNG: str = "fixed_lambda_ridge_lasso_mmse.png"

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
    if LAMBDA_RIDGE < 0.0 or LAMBDA_LASSO < 0.0:
        raise ValueError("LAMBDA_RIDGE and LAMBDA_LASSO must be >= 0")
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
def solve_rspmap(
    beta: float,
    gamma: float,
    mode: str,
    x_mc: np.ndarray,
    v_mc: np.ndarray,
    cfg: Config,
    init_sigma_eff2: Optional[float] = None,
    init_gamma_p: Optional[float] = None,
) -> Tuple[float, float, float]:
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

    for _ in range(cfg.max_fp_iters):
        z = x_mc + math.sqrt(sigma_eff2) * v_mc
        lam = gamma_p

        if mode == "lasso":
            xhat = soft_threshold(z, lam)
            p_act = float(np.mean(np.abs(z) > lam))
            mean_sigma2 = lam * p_act

        else:  # ridge
            inv = 1.0 / (1.0 + lam)
            xhat = inv * z
            mean_sigma2 = lam * inv  # = lam/(1+lam)

        mse = float(np.mean((x_mc - xhat) ** 2))
        sigma_new = cfg.sigma0_2 + beta * mse
        gamma_p_new = gamma + beta * mean_sigma2

        if not (math.isfinite(sigma_new) and math.isfinite(gamma_p_new)):
            raise FloatingPointError(
                f"Non-finite fixed point iterate: sigma_new={sigma_new}, gamma_p_new={gamma_p_new}. "
                f"Try larger damping or check parameters."
            )

        if max(abs(sigma_new - sigma_eff2), abs(gamma_p_new - gamma_p)) < cfg.tol_fp:
            sigma_eff2, gamma_p = sigma_new, gamma_p_new
            break

        sigma_eff2 = (1.0 - cfg.damp) * sigma_eff2 + cfg.damp * sigma_new
        gamma_p = (1.0 - cfg.damp) * gamma_p + cfg.damp * gamma_p_new

        # keep gamma_p away from 0 to avoid pathological numerical behavior
        gamma_p = max(gamma_p, 1e-14)

    # Final MSE evaluation at converged state
    z = x_mc + math.sqrt(max(sigma_eff2, 1e-15)) * v_mc
    lam = gamma_p
    if mode == "lasso":
        xhat = soft_threshold(z, lam)
    else:
        xhat = z / (1.0 + lam)

    mse = float(np.mean((x_mc - xhat) ** 2))
    return mse, sigma_eff2, gamma_p


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

    print("=== Starting ===")
    print(f"beta grid: [{cfg.betas.min():.2f}, {cfg.betas.max():.2f}] with {cfg.betas.size} points")
    print(f"rho={cfg.rho:.3g}, Var(nonzero)={cfg.var_nonzero:.3g} => Var(X)=1")
    print(f"SNR0={cfg.snr0_db:.2f} dB => sigma0^2={cfg.sigma0_2:.6g}")
    print(f"Fixed LAMBDA_RIDGE={LAMBDA_RIDGE:.6g}")
    print(f"Fixed LAMBDA_LASSO={LAMBDA_LASSO:.6g}")
    print(f"MC samples={cfg.mc_samples}")
    print(f"PLOT_MMSE={PLOT_MMSE}\n")

    # Monte Carlo samples (shared)
    t0 = time.time()
    x_mc = sample_bg(cfg.mc_samples, rng, cfg.rho, cfg.var_nonzero)
    v_mc = rng.normal(size=cfg.mc_samples)
    print(f"[Init] drew MC samples in {time.time() - t0:.2f}s\n")

    # Ridge curve (RS-PMAP)
    print("[Replica] computing Ridge (RS-PMAP, fixed gamma) ...")
    t0 = time.time()
    mse_ridge = np.zeros_like(cfg.betas, dtype=float)
    prev_state_ridge: Optional[Tuple[float, float]] = None
    for i, beta in enumerate(cfg.betas):
        mse, s2, gp = solve_rspmap(
            float(beta), LAMBDA_RIDGE, "ridge", x_mc, v_mc, cfg,
            init_sigma_eff2=None if prev_state_ridge is None else prev_state_ridge[0],
            init_gamma_p=None if prev_state_ridge is None else prev_state_ridge[1],
        )
        mse_ridge[i] = mse
        prev_state_ridge = (s2, gp)

        if i == 0 or (i + 1) % max(1, cfg.betas.size // 8) == 0 or i == cfg.betas.size - 1:
            print(f"  beta {i+1:3d}/{cfg.betas.size}={beta:.2f} mse={mse:.3e} (sigma_eff2={s2:.3e}, gamma_p={gp:.3e})")
    print(f"[Replica] Ridge done in {time.time() - t0:.2f}s\n")

    # LASSO curve (RS-PMAP)
    print("[Replica] computing LASSO (RS-PMAP, fixed gamma) ...")
    t0 = time.time()
    mse_lasso = np.zeros_like(cfg.betas, dtype=float)
    prev_state_lasso: Optional[Tuple[float, float]] = None
    for i, beta in enumerate(cfg.betas):
        mse, s2, gp = solve_rspmap(
            float(beta), LAMBDA_LASSO, "lasso", x_mc, v_mc, cfg,
            init_sigma_eff2=None if prev_state_lasso is None else prev_state_lasso[0],
            init_gamma_p=None if prev_state_lasso is None else prev_state_lasso[1],
        )
        mse_lasso[i] = mse
        prev_state_lasso = (s2, gp)

        if i == 0 or (i + 1) % max(1, cfg.betas.size // 8) == 0 or i == cfg.betas.size - 1:
            print(f"  beta {i+1:3d}/{cfg.betas.size}={beta:.2f} mse={mse:.3e} (sigma_eff2={s2:.3e}, gamma_p={gp:.3e})")
    print(f"[Replica] LASSO done in {time.time() - t0:.2f}s\n")

    # MMSE curve (optional)
    mse_mmse = None
    if PLOT_MMSE:
        print("[Replica] computing Optimal MMSE ...")
        t0 = time.time()
        mse_mmse = mmse_curve_replica(cfg.betas, x_mc, v_mc, cfg)
        print(f"[Replica] MMSE done in {time.time() - t0:.2f}s\n")

    # Quick sanity prints showing sensitivity to lambdas
    print("[Sanity] first/last MSE values:")
    print(f"  Ridge: beta={cfg.betas[0]:.2f} mse={mse_ridge[0]:.4g} | beta={cfg.betas[-1]:.2f} mse={mse_ridge[-1]:.4g}")
    print(f"  LASSO: beta={cfg.betas[0]:.2f} mse={mse_lasso[0]:.4g} | beta={cfg.betas[-1]:.2f} mse={mse_lasso[-1]:.4g}")
    if mse_mmse is not None:
        print(f"  MMSE : beta={cfg.betas[0]:.2f} mse={mse_mmse[0]:.4g} | beta={cfg.betas[-1]:.2f} mse={mse_mmse[-1]:.4g}")
    print()

    # Plot
    print("[Plot] saving figure ...")
    plt.figure(figsize=(8.0, 5.4))

    plt.plot(cfg.betas, to_db(mse_ridge), linewidth=3.0,
             label=f"Ridge (fixed)\n$\\lambda$={LAMBDA_RIDGE:g}")
    plt.plot(cfg.betas, to_db(mse_lasso), linewidth=3.0,
             label=f"LASSO (fixed)\n$\\lambda$={LAMBDA_LASSO:g}")
    if mse_mmse is not None:
        plt.plot(cfg.betas, to_db(mse_mmse), linewidth=3.0, linestyle="-.",
                 label="Optimal MMSE")

    plt.grid(True, linestyle=":", linewidth=1.0)
    plt.xlabel(r"Measurement ratio $\beta = n/m$", fontsize=18)
    plt.ylabel("Mean squared error (dB)", fontsize=18)
    plt.xlim(float(cfg.betas.min()), float(cfg.betas.max()))

    # Choose y-limits from data (robust-ish)
    ys = [to_db(mse_ridge), to_db(mse_lasso)]
    if mse_mmse is not None:
        ys.append(to_db(mse_mmse))
    y_all = np.concatenate(ys)
    lo = float(np.quantile(y_all, 0.01) - 2.0)
    hi = float(np.quantile(y_all, 0.99) + 2.0)
    plt.ylim(lo, hi)

    leg = plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=12, frameon=True)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"[Plot] Saved: {OUT_PNG}")
    print("=== Done ===")


if __name__ == "__main__":
    main()