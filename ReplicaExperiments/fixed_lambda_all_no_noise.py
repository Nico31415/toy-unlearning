#!/usr/bin/env python3
"""
Overlay check (NOISELESS, FAST): Ridge vs q_k(big k) and LASSO vs q_k(small k)

Same as your noiseless script, but with faster settings (fewer betas, fewer MC
samples, looser FP tolerance, fewer iterations). Good for a quick sanity check.

Output:
  overlay_q_vs_ridge_lasso_SIGMA0_0_FAST.png
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.special import erfc
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

# Only these labeled gammas (requested)
GAMMAS: Tuple[float, ...] = (10.0, 0.1, 0.001, 1e-5)

# q params
K_Q_BIG: float = 200.0
K_Q_SMALL: float = 1e-5

# BG prior
RHO: float = 0.1

# -------------------------
# NOISELESS OVERRIDE
# -------------------------
SIGMA0_2: float = 0.0  # force sigma_0^2 = 0 (noiseless)
# -------------------------

# beta sweep (FASTER: fewer points)
BETA_MIN: float = 0.5
BETA_MAX: float = 3.0
BETA_POINTS: int = 21   # was 51

# -------------------------
# FAST SETTINGS
# -------------------------
MC_SAMPLES_Q: int = 12_000   # was 80_000
MAX_FP_ITERS: int = 220      # was 900
TOL_FP: float = 3e-8         # was 1e-10
DAMP: float = 0.35           # a bit larger for faster settling (can be less stable near beta~1)
# -------------------------

SEED: int = 12345
OUT_PNG: str = str(Path(__file__).with_name("overlay_q_vs_ridge_lasso_SIGMA0_0_FAST.png"))


# =============================================================================
# Config
# =============================================================================
@dataclass(frozen=True)
class Config:
    rho: float
    var_nonzero: float
    sigma0_2: float
    betas: np.ndarray
    max_fp_iters: int
    tol_fp: float
    damp: float


def build_config() -> Config:
    if not (0.0 < RHO < 1.0):
        raise ValueError("RHO must be in (0, 1)")
    if not (0.0 <= SIGMA0_2):
        raise ValueError("SIGMA0_2 must be >= 0")
    if not (0.0 < BETA_MIN < BETA_MAX):
        raise ValueError("Require 0 < BETA_MIN < BETA_MAX")
    if BETA_POINTS < 2:
        raise ValueError("BETA_POINTS must be >= 2")
    if not (0.0 < DAMP <= 1.0):
        raise ValueError("DAMP must be in (0, 1]")
    if MC_SAMPLES_Q <= 0 or MAX_FP_ITERS <= 0 or TOL_FP <= 0:
        raise ValueError("MC_SAMPLES_Q, MAX_FP_ITERS, TOL_FP must be positive")
    if K_Q_BIG <= 0 or K_Q_SMALL <= 0:
        raise ValueError("k values must be > 0")

    var_nonzero = 1.0 / RHO
    betas = np.linspace(BETA_MIN, BETA_MAX, BETA_POINTS)
    return Config(
        rho=RHO,
        var_nonzero=var_nonzero,
        sigma0_2=float(SIGMA0_2),
        betas=betas,
        max_fp_iters=MAX_FP_ITERS,
        tol_fp=TOL_FP,
        damp=DAMP,
    )


# =============================================================================
# Utilities
# =============================================================================
def to_db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, 1e-15))


def sample_bg(n: int, rng: np.random.Generator, rho: float, var_nonzero: float) -> np.ndarray:
    active = rng.random(n) < rho
    x = np.zeros(n, dtype=float)
    if active.any():
        x[active] = rng.normal(0.0, math.sqrt(var_nonzero), int(active.sum()))
    return x


# =============================================================================
# Ridge: closed-form RS-PMAP
# =============================================================================
def solve_ridge_closed_form(beta: float, gamma: float, cfg: Config) -> float:
    beta = float(beta)
    gamma = float(gamma)

    a = 1.0 - gamma - beta
    disc = a * a + 4.0 * gamma
    gp = 0.5 * (-a + math.sqrt(disc))
    gp = float(max(gp, gamma, 0.0))

    inv = 1.0 / (1.0 + gp)
    shrink = gp * inv

    denom = 1.0 - beta * (inv ** 2)
    if abs(denom) < 1e-15:
        denom = 1e-15 if denom >= 0 else -1e-15

    sigma_eff2 = float((cfg.sigma0_2 + beta * (shrink ** 2)) / denom)
    sigma_eff2 = float(max(sigma_eff2, 0.0))

    mse = float((shrink ** 2) + (inv ** 2) * sigma_eff2)
    return mse


# =============================================================================
# LASSO: analytical RS-PMAP
# =============================================================================
def lasso_analytical_expectations(
    sigma_eff2: float,
    gamma_p: float,
    rho: float,
    var_nonzero: float,
) -> Tuple[float, float]:
    # Keep things numerically stable even when noiseless makes s2 very small.
    sigma_eff2 = float(max(sigma_eff2, 1e-15))
    lam = float(max(gamma_p, 1e-15))

    var_z0 = sigma_eff2
    tau_z0 = math.sqrt(var_z0)

    var_z1 = var_nonzero + sigma_eff2
    tau_z1 = math.sqrt(var_z1)

    p_active_0 = float(erfc(lam / (math.sqrt(2.0) * tau_z0)))
    p_active_1 = float(erfc(lam / (math.sqrt(2.0) * tau_z1)))
    p_active = (1.0 - rho) * p_active_0 + rho * p_active_1
    mean_sigma2 = lam * p_active

    t0 = lam / tau_z0
    phi_t0 = float(norm.pdf(t0))
    Phi_c_t0 = float(norm.sf(t0))
    mse0 = 2.0 * var_z0 * (Phi_c_t0 * (1.0 + t0 * t0) - t0 * phi_t0)

    t1 = lam / tau_z1
    phi_t1 = float(norm.pdf(t1))
    Phi_c_t1 = float(norm.sf(t1))
    Exhat2 = 2.0 * var_z1 * (Phi_c_t1 * (1.0 + t1 * t1) - t1 * phi_t1)

    Exxhat = var_nonzero * p_active_1
    mse1 = var_nonzero - 2.0 * Exxhat + Exhat2

    mse = (1.0 - rho) * mse0 + rho * mse1
    mse = float(max(mse, 0.0))
    return mse, mean_sigma2


def solve_lasso_analytical(
    beta: float,
    gamma: float,
    cfg: Config,
    init_s2: Optional[float] = None,
    init_gp: Optional[float] = None,
    max_iters: int = 1200,  # was 6000
    tol: float = 3e-10,     # was 1e-12
    damp: float = 0.35,     # faster
) -> Tuple[float, float, float]:
    beta = float(beta)
    gamma = float(gamma)

    s2 = float(cfg.sigma0_2 if init_s2 is None else max(init_s2, cfg.sigma0_2))
    gp = float(max(gamma, 1e-14) if init_gp is None else max(init_gp, gamma, 1e-14))

    for _ in range(max_iters):
        mse, mean_sigma2 = lasso_analytical_expectations(s2, gp, cfg.rho, cfg.var_nonzero)

        s2_new = float(max(cfg.sigma0_2, cfg.sigma0_2 + beta * mse))
        gp_new = float(max(gamma, gamma + beta * mean_sigma2, 1e-14))

        if max(abs(s2_new - s2), abs(gp_new - gp)) < tol:
            s2, gp = s2_new, gp_new
            mse_final, _ = lasso_analytical_expectations(s2, gp, cfg.rho, cfg.var_nonzero)
            return float(mse_final), float(s2), float(gp)

        s2 = (1.0 - damp) * s2 + damp * s2_new
        gp = (1.0 - damp) * gp + damp * gp_new
        s2 = float(max(s2, cfg.sigma0_2))
        gp = float(max(gp, gamma, 1e-14))

    mse_final, _ = lasso_analytical_expectations(s2, gp, cfg.rho, cfg.var_nonzero)
    return float(mse_final), float(s2), float(gp)


# =============================================================================
# q_k regulariser: prox + local variance (MC)
# =============================================================================
def prox_qk_safeguarded(
    z: np.ndarray, lam: float, k: float, tol: float = 1e-12, max_iters: int = 90  # was 140
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    lam = float(lam)
    k = float(k)
    if lam <= 0.0:
        return z.copy()

    sk = math.sqrt(k)
    lo = np.minimum(z, 0.0)
    hi = np.maximum(z, 0.0)
    x = 0.5 * (lo + hi)

    for _ in range(max_iters):
        Fx = x - z + 0.5 * lam * np.arcsinh(2.0 * x / sk)
        if float(np.max(np.abs(Fx))) < tol:
            break

        Fpx = 1.0 + lam / np.sqrt(k + 4.0 * x * x)
        x_new = x - Fx / Fpx

        neg = Fx < 0
        lo = np.where(neg, x, lo)
        hi = np.where(~neg, x, hi)

        bad = (x_new < lo) | (x_new > hi) | (~np.isfinite(x_new))
        x = np.where(bad, 0.5 * (lo + hi), x_new)

    return x


def sigma2_qk(xstar: np.ndarray, lam: float, k: float) -> np.ndarray:
    lam = float(max(lam, 1e-14))
    qpp = 1.0 / np.sqrt(float(k) + 4.0 * xstar * xstar)
    return 1.0 / (1.0 / lam + qpp)


def solve_rspmap_qk_one(
    beta: float,
    gamma_ext: float,
    k_q: float,
    x_mc: np.ndarray,
    v_mc: np.ndarray,
    cfg: Config,
    init_state: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float]:
    beta = float(beta)
    gamma_ext = float(gamma_ext)
    k_q = float(k_q)

    s2 = float(cfg.sigma0_2 if init_state is None else max(init_state[0], cfg.sigma0_2))
    gp = float(max(gamma_ext, 1e-14) if init_state is None else max(init_state[1], gamma_ext, 1e-14))

    for _ in range(cfg.max_fp_iters):
        z = x_mc + math.sqrt(max(s2, 0.0)) * v_mc
        xhat = prox_qk_safeguarded(z, gp, k_q)
        mse = float(np.mean((x_mc - xhat) ** 2))
        mean_sigma2 = float(np.mean(sigma2_qk(xhat, gp, k_q)))

        s2_new = float(cfg.sigma0_2 + beta * mse)
        gp_new = float(gamma_ext + beta * mean_sigma2)

        if max(abs(s2_new - s2), abs(gp_new - gp)) < cfg.tol_fp:
            return float(mse), float(s2_new), float(gp_new)

        s2 = (1.0 - cfg.damp) * s2 + cfg.damp * s2_new
        gp = (1.0 - cfg.damp) * gp + cfg.damp * gp_new
        s2 = float(max(s2, cfg.sigma0_2))
        gp = float(max(gp, gamma_ext, 1e-14))

    z = x_mc + math.sqrt(max(s2, 0.0)) * v_mc
    xhat = prox_qk_safeguarded(z, gp, k_q)
    mse = float(np.mean((x_mc - xhat) ** 2))
    return float(mse), float(s2), float(gp)


def solve_rspmap_qk_curve_best_of_forward_backward(
    betas: np.ndarray,
    gamma_ext: float,
    k_q: float,
    x_mc: np.ndarray,
    v_mc: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    # forward continuation
    f = np.zeros_like(betas, dtype=float)
    state = None
    for i, b in enumerate(betas):
        mse, s2, gp = solve_rspmap_qk_one(b, gamma_ext, k_q, x_mc, v_mc, cfg, init_state=state)
        f[i] = mse
        state = (s2, gp)

    # backward continuation
    bwd = np.zeros_like(betas, dtype=float)
    state = None
    for j, b in enumerate(betas[::-1]):
        mse, s2, gp = solve_rspmap_qk_one(b, gamma_ext, k_q, x_mc, v_mc, cfg, init_state=state)
        bwd[j] = mse
        state = (s2, gp)
    bwd = bwd[::-1]

    return np.minimum(f, bwd)


# =============================================================================
# Gamma scaling
# =============================================================================
def gamma_ext_for_q_big(gamma_label: float, k_big: float) -> float:
    return float(gamma_label) * math.sqrt(k_big)


def gamma_ext_for_q_small(gamma_label: float, k_small: float) -> float:
    if not (0.0 < k_small < 1.0):
        return float(gamma_label)
    return float(gamma_label) * (4.0 / math.log(1.0 / k_small))


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    cfg = build_config()
    rng = np.random.default_rng(SEED)
    betas = cfg.betas

    print("=== NOISELESS FAST overlay plot ===")
    print(f"gammas(label) = {GAMMAS}")
    print(f"k_big={K_Q_BIG:g}, k_small={K_Q_SMALL:g}")
    print(f"sigma0^2 = {cfg.sigma0_2:g} (FORCED)")
    print(f"BETA_POINTS={len(betas)} | MC_SAMPLES_Q={MC_SAMPLES_Q} | MAX_FP_ITERS={cfg.max_fp_iters} | TOL_FP={cfg.tol_fp:g} | DAMP={cfg.damp:g}")
    print()

    xq = sample_bg(MC_SAMPLES_Q, rng, cfg.rho, cfg.var_nonzero)
    vq = rng.normal(size=MC_SAMPLES_Q)

    ridge: Dict[float, np.ndarray] = {}
    lasso: Dict[float, np.ndarray] = {}
    q_big: Dict[float, np.ndarray] = {}
    q_small: Dict[float, np.ndarray] = {}

    for g in GAMMAS:
        ridge[g] = np.array([solve_ridge_closed_form(b, g, cfg) for b in betas], dtype=float)

        arr = np.zeros_like(betas)
        prev_s2, prev_gp = None, None
        for i, b in enumerate(betas):
            mse, s2, gp = solve_lasso_analytical(b, g, cfg, prev_s2, prev_gp)
            arr[i] = mse
            prev_s2, prev_gp = s2, gp
        lasso[g] = arr

    for g_label in GAMMAS:
        g_ext = gamma_ext_for_q_big(g_label, K_Q_BIG)
        q_big[g_label] = solve_rspmap_qk_curve_best_of_forward_backward(betas, g_ext, K_Q_BIG, xq, vq, cfg)

        g_ext = gamma_ext_for_q_small(g_label, K_Q_SMALL)
        q_small[g_label] = solve_rspmap_qk_curve_best_of_forward_backward(betas, g_ext, K_Q_SMALL, xq, vq, cfg)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.7, 5.8), sharey=True)

    cmap = plt.get_cmap("viridis")
    gam_sorted = sorted(GAMMAS, reverse=True)
    cols = [cmap(x) for x in np.linspace(0.15, 0.85, len(gam_sorted))]
    color_of = {g: c for g, c in zip(gam_sorted, cols)}

    for g in gam_sorted:
        c = color_of[g]
        ax_left.plot(betas, to_db(ridge[g]), linewidth=2.2, color=c, linestyle="-",
                     label=rf"$\gamma$={g:g}  (ridge)")
        ax_left.plot(betas, to_db(q_big[g]), linewidth=2.2, color=c, linestyle="--",
                     label=rf"$\gamma$={g:g}  ($q_k$, $k$={K_Q_BIG:g})")

    ax_left.set_title(rf"Ridge (solid) vs $q_k$ (dashed),  $k$={K_Q_BIG:g}", fontsize=13)
    ax_left.set_xlabel(r"Measurement ratio $\beta=n/m$", fontsize=12)
    ax_left.set_ylabel("MSE (dB)", fontsize=12)
    ax_left.grid(True, linestyle=":", linewidth=1.0)

    for g in gam_sorted:
        c = color_of[g]
        ax_right.plot(betas, to_db(lasso[g]), linewidth=2.2, color=c, linestyle="-",
                      label=rf"$\gamma$={g:g}  (lasso)")
        ax_right.plot(betas, to_db(q_small[g]), linewidth=2.2, color=c, linestyle="--",
                      label=rf"$\gamma$={g:g}  ($q_k$, $k$={K_Q_SMALL:g})")

    ax_right.set_title(rf"LASSO (solid) vs $q_k$ (dashed),  $k$={K_Q_SMALL:g}", fontsize=13)
    ax_right.set_xlabel(r"Measurement ratio $\beta=n/m$", fontsize=12)
    ax_right.grid(True, linestyle=":", linewidth=1.0)

    ax_left.set_xlim(float(betas.min()), float(betas.max()))
    ax_left.set_ylim(-18.0, 15.0)

    ax_left.legend(loc="upper left", fontsize=8.3, frameon=True, ncol=1)
    ax_right.legend(loc="upper left", fontsize=8.3, frameon=True, ncol=1)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")  # lower dpi for speed
    plt.close(fig)

    print(f"[Saved] {OUT_PNG}")
    print("=== Done ===")


if __name__ == "__main__":
    main()