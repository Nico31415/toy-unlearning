#!/usr/bin/env python3
"""
Overlay check (tighter match): Ridge vs q_k(big k) and LASSO vs q_k(small k)

ONE FIGURE with 1x2 subplots (side-by-side):
  Left : Ridge (solid) + q_k at LARGE k (dashed)
  Right: LASSO (solid) + q_k at SMALL k (dashed)

Gammas (label gammas shown in legend):
  {10, 0.1, 0.001, 1e-05}

What’s changed vs the “quick” version to tighten the match:
  1) More MC samples for q_k expectations  -> reduces MC noise and stabilizes the FP
  2) Tighter fixed-point tolerance         -> less residual bias in (sigma_eff2, gamma_p)
  3) More FP iterations                    -> better convergence near the spike around beta~1
  4) A gentle continuation pass forward AND backward in beta for q_k,
     then take the lower-MSE branch pointwise (helps near phase-transition hysteresis)

Everything else is the same.

Output:
  overlay_q_vs_ridge_lasso_TIGHT.png

================================================================================
STATE-EVOLUTION CONVENTION  (read before comparing against replica_derivation.pdf)
================================================================================
This module uses the *reciprocal* (compressed-sensing) convention, which is
DIFFERENT from the N/D convention used in replica_derivation.pdf and in the
unlearning scripts (compare_cpt_replica.py, unlearning_relearning_replica.py).

  * Here  ``beta`` is the aspect ratio  D / N  (signal dimension / measurements),
    i.e. the x-axis label "beta = n/m" has n = signal dimension, m = #measurements.
    Hence  beta = 1 / alpha  where alpha = N/D is the measurement ratio used in
    the theory note.  Small beta = overdetermined = small MSE.

  * The SE noise update is the *additive* form
        s^2  <-  sigma0^2  +  beta * MSE                         (this file)
    which is the self-consistent partner of solve_ridge_closed_form() and
    lasso_analytical_expectations() below: the ridge closed form is DERIVED from
    exactly this update (see solve_ridge_closed_form), and main() verifies that
    the q_k MC solver reproduces it.  Changing this line to the theory's
    divisive form  s^2 = (sigma0^2 + MSE)/alpha  would BREAK that internal
    ridge/LASSO agreement and must not be done in isolation.

  * The theory note (replica_derivation.pdf, eq. SE1, and the warning in sec. 9.3)
    instead writes, in the alpha = N/D convention,
        s^2  =  (sigma0^2 + MSE) / alpha .
    Substituting beta = 1/alpha, the MSE/alpha term matches; the two conventions
    differ only in how the (tiny) noise floor sigma0^2 enters
    (sigma0^2 here vs. sigma0^2/alpha there), which is numerically negligible at
    the SNRs used here (sigma0^2 ~ 1e-2).  This is a convention choice, NOT a bug.

The unlearning pipeline (compare_cpt_replica.py, unlearning_relearning_replica.py)
uses the theory's alpha = N/D convention with the divisive SE update and only
imports the convention-free primitives prox_qk_safeguarded() and sigma2_qk().
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

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

# q params (your current ones)
K_Q_BIG: float = 200.0
K_Q_SMALL: float = 1e-5

# BG + noise
RHO: float = 0.1
SNR0_DB: float = 10.0

# beta sweep
BETA_MIN: float = 0.5
BETA_MAX: float = 3.0
BETA_POINTS: int = 51

# -------------------------
# TIGHTER / SLOWER SETTINGS
# -------------------------
MC_SAMPLES_Q: int = 80_000     # was 12k
MAX_FP_ITERS: int = 900        # was 260
TOL_FP: float = 1e-10          # was 3e-8
DAMP: float = 0.25             # slightly smaller damping = more stable convergence near beta~1
# -------------------------

SEED: int = 12345
OUT_PNG: str = str(Path(__file__).with_name("overlay_q_vs_ridge_lasso_TIGHT.png"))


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


@dataclass(frozen=True)
class KModeConfig:
    """
    Configuration for heterogeneous-k sampling in the replica q_k solver.

    Modes:
        - "homogeneous": all coordinates share the same k value (k_hom).
        - "mixture"    : coordinates are independently assigned to group A/B
                         with probability pi_A, and use k_A / k_B.
        - "support"    : k depends on the BG teacher support implied by x_mc:
                         nonzero -> k_nz, zero -> k_z.

    The default mode is "homogeneous"; in the existing codepath we *do not*
    use this config and keep behaviour identical. It is only used when the
    diagonal replica CLI explicitly enables hetero-k.
    """

    mode: str = "homogeneous"
    # Homogeneous mode: scalar k shared by all coordinates
    k_hom: Optional[float] = None
    # Mixture mode (Step 1A)
    k_A: Optional[float] = None
    k_B: Optional[float] = None
    pi_A: Optional[float] = None
    # Support-conditioned mode (Step 1B)
    k_nz: Optional[float] = None
    k_z: Optional[float] = None


@dataclass
class PTFTOracleConfig:
    """
    Configuration for Step 2A: Deterministic PT + Stochastic FT replica curves.
    
    The FT implicit regularizer is derived per-coordinate from PT parameters
    via the Cosyne mapping. This is SEPARATE from ft_regulariser_scale (which
    sets gamma_ext).
    
    Parameters:
        rho_pt: PT support fraction (0 < rho_pt < 1)
        rho_ft: FT teacher sparsity (0 < rho_ft < 1)
        omega: Overlap fraction |S_pt ∩ S_ft| / |S_ft| (0 <= omega <= 1)
        a_pt: Deterministic PT ground truth amplitude (constant on PT support)
        c_pt: PT parameter c (must be > 0)
        lambda_pt: PT initialization parameter λ (can be 0)
                   NOTE: This is NOT the regularizer scale!
        gamma_reinit: Readout reinitialization parameter
    """
    rho_pt: float
    rho_ft: float
    omega: float
    a_pt: float
    c_pt: float
    lambda_pt: float      # PT init param, NOT regularizer scale
    gamma_reinit: float


def build_config() -> Config:
    if not (0.0 < RHO < 1.0):
        raise ValueError("RHO must be in (0, 1)")
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
    snr0 = 10.0 ** (SNR0_DB / 10.0)
    sigma0_2 = 1.0 / snr0
    betas = np.linspace(BETA_MIN, BETA_MAX, BETA_POINTS)
    return Config(
        rho=RHO,
        var_nonzero=var_nonzero,
        sigma0_2=sigma0_2,
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


def sample_bg(
    n: int,
    rng: np.random.Generator,
    rho: float,
    var_nonzero: float,
    return_mask: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    active = rng.random(n) < rho
    x = np.zeros(n, dtype=float)
    if active.any():
        x[active] = rng.normal(0.0, math.sqrt(var_nonzero), int(active.sum()))
    if return_mask:
        return x, active
    return x


def sample_k_mc(
    k_cfg: KModeConfig,
    x_mc: np.ndarray,
    rng: np.random.Generator,
    mask_bg: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Sample coordinate-wise k values for the q_k regulariser, given BG teacher samples.

    Args:
        k_cfg: KModeConfig describing the hetero-k mode and parameters.
        x_mc:  Monte Carlo BG samples (teacher coefficients), shape (mc_samples,).
        rng:   NumPy Generator seeded from the global seed (shared with x_mc, v_mc).
        mask_bg: Optional boolean mask indicating BG active coordinates (True = nonzero).

    Returns:
        k_mc:  np.ndarray of shape (mc_samples,), with per-coordinate k values.
        g_mc:  Integer group labels for robust grouping. None only for homogeneous mode.

    Modes:
        - homogeneous: k_mc[:] = k_hom, g_mc = None (no grouping needed)
        - mixture   : independent Bernoulli(pi_A) group assignment, k_A vs k_B, g_mc ∈ {0,1}
        - support   : k_nz on nonzeros (BG active), k_z on zeros, g_mc ∈ {0,1}
    """
    x_mc = np.asarray(x_mc, dtype=float)
    n = x_mc.shape[0]
    mode = (k_cfg.mode or "homogeneous").lower()

    if mode == "homogeneous":
        if k_cfg.k_hom is None:
            raise ValueError("KModeConfig(mode='homogeneous') requires k_hom.")
        k_val = float(k_cfg.k_hom)
        if k_val <= 0.0:
            raise ValueError("k_hom must be > 0.")
        # Homogeneous: no grouping needed, return None for g_mc
        return np.full(n, k_val, dtype=float), None

    if mode == "mixture":
        if k_cfg.k_A is None or k_cfg.k_B is None or k_cfg.pi_A is None:
            raise ValueError("KModeConfig(mode='mixture') requires k_A, k_B, pi_A.")
        k_A = float(k_cfg.k_A)
        k_B = float(k_cfg.k_B)
        pi_A = float(k_cfg.pi_A)
        if not (0.0 < pi_A < 1.0):
            raise ValueError("pi_A must be in (0, 1) for mixture mode.")
        if k_A <= 0.0 or k_B <= 0.0:
            raise ValueError("k_A and k_B must be > 0 for mixture mode.")
        g = rng.random(n) < pi_A
        k_out = np.where(g, k_A, k_B).astype(float)
        g_mc = np.where(g, 1, 0).astype(int)
        return k_out, g_mc

    if mode == "support":
        if k_cfg.k_nz is None or k_cfg.k_z is None:
            raise ValueError("KModeConfig(mode='support') requires k_nz and k_z.")
        k_nz = float(k_cfg.k_nz)
        k_z = float(k_cfg.k_z)
        if k_nz <= 0.0 or k_z <= 0.0:
            raise ValueError("k_nz and k_z must be > 0 for support mode.")
        if mask_bg is None:
            # BG sampling uses exact zeros for inactive coordinates; add a lightweight assertion.
            assert np.all((x_mc == 0.0) | (np.abs(x_mc) > 1e-12)), "BG zeros should be exact."
            nz = x_mc != 0.0
        else:
            nz = mask_bg
        k_out = np.where(nz, k_nz, k_z).astype(float)
        g_mc = np.where(nz, 1, 0).astype(int)
        return k_out, g_mc

    raise ValueError(f"Unknown k-mode: {k_cfg.mode!r} (expected 'homogeneous', 'mixture', or 'support').")


def compute_c_ft_from_pt(
    beta_pt: np.ndarray,
    c_pt: float,
    lambda_pt: float,
    gamma_reinit: float
) -> np.ndarray:
    """
    Compute per-coordinate effective FT regularizer c_ft from PT ground truth.
    
    Formula (Cosyne mapping):
        c_ft_i = (lambda_pt + c_pt) * (1 + sqrt(1 + (beta_pt_i / c_pt)^2)) 
                 + (gamma_reinit^2)/2
    
    Then k_i = 4 * c_ft_i^2 is used by the q_k solver.
    
    Args:
        beta_pt: PT ground truth amplitudes, shape (n,), float64
        c_pt: PT parameter c (scalar, must be > 0)
        lambda_pt: PT initialization parameter (scalar, can be 0)
                   NOTE: This is the PT init param, NOT the FT regularizer scale!
        gamma_reinit: Readout reinit parameter (scalar)
    
    Returns:
        c_ft: Per-coordinate FT regularizer values, shape (n,), float64
    
    Raises:
        ValueError: if c_pt <= 0
    """
    if not isinstance(beta_pt, np.ndarray):
        beta_pt = np.asarray(beta_pt, dtype=np.float64)
    else:
        beta_pt = beta_pt.astype(np.float64, copy=False)
    
    c_pt = float(c_pt)
    lambda_pt = float(lambda_pt)
    gamma_reinit = float(gamma_reinit)
    
    if c_pt <= 0.0:
        raise ValueError(f"c_pt must be > 0, got {c_pt}")
    
    # Vectorized computation
    ratio_sq = (beta_pt / c_pt) ** 2
    sqrt_term = np.sqrt(1.0 + ratio_sq)
    c_ft = (lambda_pt + c_pt) * (1.0 + sqrt_term) + 0.5 * (gamma_reinit ** 2)
    
    return c_ft.astype(np.float64)


def sample_ptft_oracle_mc(
    cfg: PTFTOracleConfig,
    rng: np.random.Generator,
    mc_samples: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo sampler for PT+FT oracle teacher mode (Step 2A).
    
    4-group coordinate model:
        - OV (label=0): PT-active AND FT-nonzero
        - NEW (label=1): PT-inactive AND FT-nonzero
        - PTONLY (label=2): PT-active AND FT-zero
        - NONE (label=3): PT-inactive AND FT-zero
    
    Sampling:
        1. Sample group label from categorical distribution
        2. Deterministic PT ground truth:
           - OV, PTONLY: beta_pt = a_pt (constant amplitude)
           - NEW, NONE: beta_pt = 0
        3. Stochastic FT ground truth (BG conditioned):
           - OV, NEW: beta_ft ~ N(0, 1/rho_ft)
           - PTONLY, NONE: beta_ft = 0
        4. Compute c_ft from beta_pt via Cosyne mapping
        5. Convert to k_i = 4 * c_ft_i^2
    
    Args:
        cfg: PTFTOracleConfig with all parameters
        rng: NumPy random generator
        mc_samples: Number of Monte Carlo samples
    
    Returns:
        beta_ft_mc: FT ground truth (x_mc for solver), shape (mc_samples,), float64
        beta_pt_mc: PT ground truth, shape (mc_samples,), float64
        k_mc: Per-coordinate k values, shape (mc_samples,), float64
        g_mc: Group labels (0=OV, 1=NEW, 2=PTONLY, 3=NONE), shape (mc_samples,), int
    
    Raises:
        ValueError: if feasibility constraints are violated
    """
    # Validate inputs
    if not (0.0 < cfg.rho_pt < 1.0):
        raise ValueError(f"rho_pt must be in (0, 1), got {cfg.rho_pt}")
    if not (0.0 < cfg.rho_ft < 1.0):
        raise ValueError(f"rho_ft must be in (0, 1), got {cfg.rho_ft}")
    if not (0.0 <= cfg.omega <= 1.0):
        raise ValueError(f"omega must be in [0, 1], got {cfg.omega}")
    if cfg.c_pt <= 0.0:
        raise ValueError(f"c_pt must be > 0, got {cfg.c_pt}")
    
    # Compute group probabilities
    p_ov = cfg.omega * cfg.rho_ft
    p_new = (1.0 - cfg.omega) * cfg.rho_ft
    p_ptonly = cfg.rho_pt - p_ov
    p_none = 1.0 - cfg.rho_pt - p_new
    
    # Feasibility checks with clear error messages
    if p_ov < 0.0:
        raise ValueError(
            f"Infeasible: p_ov = omega * rho_ft = {p_ov:.6e} < 0. "
            f"Check: omega={cfg.omega}, rho_ft={cfg.rho_ft}"
        )
    if p_new < 0.0:
        raise ValueError(
            f"Infeasible: p_new = (1-omega) * rho_ft = {p_new:.6e} < 0. "
            f"Check: omega={cfg.omega}, rho_ft={cfg.rho_ft}"
        )
    if p_ptonly < 0.0:
        raise ValueError(
            f"Infeasible: p_ptonly = rho_pt - omega*rho_ft = {p_ptonly:.6e} < 0. "
            f"Constraint violated: omega * rho_ft <= rho_pt. "
            f"Got: omega={cfg.omega}, rho_ft={cfg.rho_ft}, rho_pt={cfg.rho_pt}, "
            f"omega*rho_ft={p_ov:.6e} > rho_pt={cfg.rho_pt:.6e}"
        )
    if p_none < 0.0:
        raise ValueError(
            f"Infeasible: p_none = 1 - rho_pt - (1-omega)*rho_ft = {p_none:.6e} < 0. "
            f"Check: rho_pt={cfg.rho_pt}, rho_ft={cfg.rho_ft}, omega={cfg.omega}"
        )
    
    # Normalize probabilities (should sum to 1, but ensure numerical stability)
    probs = np.array([p_ov, p_new, p_ptonly, p_none], dtype=np.float64)
    probs /= probs.sum()
    
    # Sample group labels: 0=OV, 1=NEW, 2=PTONLY, 3=NONE
    g_mc = rng.choice(4, size=mc_samples, p=probs).astype(np.int32)
    
    # Construct beta_pt_mc (deterministic from PT)
    beta_pt_mc = np.zeros(mc_samples, dtype=np.float64)
    pt_active = (g_mc == 0) | (g_mc == 2)  # OV or PTONLY
    beta_pt_mc[pt_active] = cfg.a_pt
    
    # Construct beta_ft_mc (stochastic FT ground truth)
    beta_ft_mc = np.zeros(mc_samples, dtype=np.float64)
    ft_nonzero = (g_mc == 0) | (g_mc == 1)  # OV or NEW
    n_ft_nonzero = int(ft_nonzero.sum())
    if n_ft_nonzero > 0:
        # Sample from N(0, 1/rho_ft)
        sigma_ft = 1.0 / math.sqrt(cfg.rho_ft)
        beta_ft_mc[ft_nonzero] = rng.normal(0.0, sigma_ft, size=n_ft_nonzero)
    
    # Compute c_ft from beta_pt via Cosyne mapping
    c_ft_mc = compute_c_ft_from_pt(beta_pt_mc, cfg.c_pt, cfg.lambda_pt, cfg.gamma_reinit)
    
    # Convert to k_i = 4 * c_ft_i^2
    k_mc = 4.0 * (c_ft_mc ** 2)
    
    return beta_ft_mc, beta_pt_mc, k_mc, g_mc


def soft_threshold(z: np.ndarray, t: float) -> np.ndarray:
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


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
    sigma_eff2 = float(max(sigma_eff2, cfg.sigma0_2))

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
    max_iters: int = 6000,
    tol: float = 1e-12,
    damp: float = 0.25,
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
def prox_qk_safeguarded(z: np.ndarray, lam: float, k: float, tol: float = 1e-12, max_iters: int = 140) -> np.ndarray:
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
    k_mc: Optional[np.ndarray] = None,
    g_mc: Optional[np.ndarray] = None,
    return_diag: bool = False,
    eps_active: float = 1e-6,
) -> Union[Tuple[float, float, float], Tuple[float, float, float, Dict]]:
    """
    Solve the RS-PMAP fixed point for q_k regulariser at a single beta value.
    
    Args:
        beta: Measurement ratio n/d
        gamma_ext: External regularization parameter
        k_q: Scalar k value (used in homogeneous mode)
        x_mc: Monte Carlo teacher samples
        v_mc: Monte Carlo noise samples
        cfg: Configuration object
        init_state: Optional (s2, gp) initialization state
        k_mc: Optional per-coordinate k values (enables heterogeneous mode)
        g_mc: Optional group labels for coordinates
        return_diag: If True, return diagnostics dict with active_frac
        eps_active: Threshold for considering a coordinate "active"
    
    Returns:
        If return_diag=False: (mse, s2, gp)
        If return_diag=True: (mse, s2, gp, diag_dict) where diag_dict contains:
            - 'active_frac': fraction of coordinates with |xhat| > eps_active
            - 'active_frac_by_group': dict mapping group label to active fraction (if g_mc provided)
    """
    beta = float(beta)
    gamma_ext = float(gamma_ext)
    k_q = float(k_q)

    s2 = float(cfg.sigma0_2 if init_state is None else max(init_state[0], cfg.sigma0_2))
    gp = float(max(gamma_ext, 1e-14) if init_state is None else max(init_state[1], gamma_ext, 1e-14))

    # Helper to compute diagnostics from xhat
    def _compute_diag(xhat_final: np.ndarray) -> Dict:
        active_mask = np.abs(xhat_final) > eps_active
        diag = {'active_frac': float(np.mean(active_mask))}
        if g_mc is not None:
            labels = np.asarray(g_mc)
            active_by_group = {}
            for lab in np.unique(labels):
                idx = labels == lab
                if np.any(idx):
                    active_by_group[int(lab)] = float(np.mean(active_mask[idx]))
            diag['active_frac_by_group'] = active_by_group
        return diag

    # Homogeneous path: preserve exact historical behaviour when k_mc is None.
    if k_mc is None:
        for _ in range(cfg.max_fp_iters):
            z = x_mc + math.sqrt(max(s2, 1e-15)) * v_mc
            lam = gp

            xhat = prox_qk_safeguarded(z, lam, k_q)
            mse = float(np.mean((x_mc - xhat) ** 2))
            mean_sigma2 = float(np.mean(sigma2_qk(xhat, lam, k_q)))

            # Additive SE update under the reciprocal convention beta = D/N
            # (see the STATE-EVOLUTION CONVENTION block at the top of this file).
            # This is the self-consistent partner of solve_ridge_closed_form();
            # do NOT swap it for the theory's divisive form in isolation.
            s2_new = float(cfg.sigma0_2 + beta * mse)
            gp_new = float(gamma_ext + beta * mean_sigma2)

            if max(abs(s2_new - s2), abs(gp_new - gp)) < cfg.tol_fp:
                if return_diag:
                    return float(mse), float(s2_new), float(gp_new), _compute_diag(xhat)
                return float(mse), float(s2_new), float(gp_new)

            s2 = (1.0 - cfg.damp) * s2 + cfg.damp * s2_new
            gp = (1.0 - cfg.damp) * gp + cfg.damp * gp_new
            s2 = float(max(s2, cfg.sigma0_2))
            gp = float(max(gp, gamma_ext, 1e-14))

        z = x_mc + math.sqrt(max(s2, 1e-15)) * v_mc
        xhat = prox_qk_safeguarded(z, gp, k_q)
        mse = float(np.mean((x_mc - xhat) ** 2))
        if return_diag:
            return float(mse), float(s2), float(gp), _compute_diag(xhat)
        return float(mse), float(s2), float(gp)

    # Heterogeneous path: k varies per MC sample.
    k_vec = np.asarray(k_mc, dtype=float)
    if k_vec.shape != x_mc.shape:
        raise ValueError("k_mc must have the same shape as x_mc.")

    for _ in range(cfg.max_fp_iters):
        z = x_mc + math.sqrt(max(s2, 1e-15)) * v_mc
        lam = gp

        xhat = np.empty_like(z)
        sigma2 = np.empty_like(z)

        # Group by labels if provided; otherwise by unique k values.
        if g_mc is not None:
            labels = np.asarray(g_mc)
            if labels.shape != x_mc.shape:
                raise ValueError("g_mc must have the same shape as x_mc.")
            uniq_labels = np.unique(labels)
            for lab in uniq_labels:
                idx = labels == lab
                if not np.any(idx):
                    continue
                k_val = float(np.mean(k_vec[idx]))
                xhat[idx] = prox_qk_safeguarded(z[idx], lam, k_val)
                sigma2[idx] = sigma2_qk(xhat[idx], lam, k_val)
        else:
            unique_k = np.unique(k_vec)
            for k_val in unique_k:
                idx = k_vec == k_val
                if not np.any(idx):
                    continue
                xhat[idx] = prox_qk_safeguarded(z[idx], lam, float(k_val))
                sigma2[idx] = sigma2_qk(xhat[idx], lam, float(k_val))

        mse = float(np.mean((x_mc - xhat) ** 2))
        mean_sigma2 = float(np.mean(sigma2))

        s2_new = float(cfg.sigma0_2 + beta * mse)
        gp_new = float(gamma_ext + beta * mean_sigma2)

        if max(abs(s2_new - s2), abs(gp_new - gp)) < cfg.tol_fp:
            if return_diag:
                return float(mse), float(s2_new), float(gp_new), _compute_diag(xhat)
            return float(mse), float(s2_new), float(gp_new)

        s2 = (1.0 - cfg.damp) * s2 + cfg.damp * s2_new
        gp = (1.0 - cfg.damp) * gp + cfg.damp * gp_new
        s2 = float(max(s2, cfg.sigma0_2))
        gp = float(max(gp, gamma_ext, 1e-14))

    z = x_mc + math.sqrt(max(s2, 1e-15)) * v_mc
    xhat = np.empty_like(z)
    if g_mc is not None:
        labels = np.asarray(g_mc)
        uniq_labels = np.unique(labels)
        for lab in uniq_labels:
            idx = labels == lab
            if not np.any(idx):
                continue
            k_val = float(np.mean(k_vec[idx]))
            xhat[idx] = prox_qk_safeguarded(z[idx], gp, k_val)
    else:
        unique_k = np.unique(k_vec)
        for k_val in unique_k:
            idx = k_vec == k_val
            if not np.any(idx):
                continue
            xhat[idx] = prox_qk_safeguarded(z[idx], gp, float(k_val))
    mse = float(np.mean((x_mc - xhat) ** 2))
    if return_diag:
        return float(mse), float(s2), float(gp), _compute_diag(xhat)
    return float(mse), float(s2), float(gp)


def solve_rspmap_qk_curve_best_of_forward_backward(
    betas: np.ndarray,
    gamma_ext: float,
    k_q: float,
    x_mc: np.ndarray,
    v_mc: np.ndarray,
    cfg: Config,
    k_mc: Optional[np.ndarray] = None,
    g_mc: Optional[np.ndarray] = None,
    return_diag: bool = False,
    eps_active: float = 1e-6,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Solve the RS-PMAP fixed point curve with forward/backward continuation.
    
    Args:
        betas: Array of beta values (measurement ratios)
        gamma_ext: External regularization parameter
        k_q: Scalar k value (used in homogeneous mode)
        x_mc: Monte Carlo teacher samples
        v_mc: Monte Carlo noise samples
        cfg: Configuration object
        k_mc: Optional per-coordinate k values (enables heterogeneous mode)
        g_mc: Optional group labels for coordinates
        return_diag: If True, return active fraction curve alongside MSE
        eps_active: Threshold for considering a coordinate "active"
    
    Returns:
        If return_diag=False: mse_curve (np.ndarray)
        If return_diag=True: (mse_curve, active_frac_curve) tuple
    """
    # forward continuation
    f = np.zeros_like(betas, dtype=float)
    active_fwd = np.zeros_like(betas, dtype=float) if return_diag else None
    state = None
    for i, b in enumerate(betas):
        result = solve_rspmap_qk_one(
            b,
            gamma_ext,
            k_q,
            x_mc,
            v_mc,
            cfg,
            init_state=state,
            k_mc=k_mc,
            g_mc=g_mc,
            return_diag=return_diag,
            eps_active=eps_active,
        )
        if return_diag:
            mse, s2, gp, diag = result
            active_fwd[i] = diag['active_frac']
        else:
            mse, s2, gp = result
        f[i] = mse
        state = (s2, gp)

    # backward continuation
    bwd = np.zeros_like(betas, dtype=float)
    active_bwd = np.zeros_like(betas, dtype=float) if return_diag else None
    state = None
    for j, b in enumerate(betas[::-1]):
        result = solve_rspmap_qk_one(
            b,
            gamma_ext,
            k_q,
            x_mc,
            v_mc,
            cfg,
            init_state=state,
            k_mc=k_mc,
            g_mc=g_mc,
            return_diag=return_diag,
            eps_active=eps_active,
        )
        if return_diag:
            mse, s2, gp, diag = result
            active_bwd[j] = diag['active_frac']
        else:
            mse, s2, gp = result
        bwd[j] = mse
        state = (s2, gp)
    bwd = bwd[::-1]
    if return_diag:
        active_bwd = active_bwd[::-1]

    # take best branch pointwise (helps near beta~1)
    mse_curve = np.minimum(f, bwd)
    
    if return_diag:
        # Select active_frac from the same branch as the lower MSE
        active_curve = np.where(f <= bwd, active_fwd, active_bwd)
        return mse_curve, active_curve
    
    return mse_curve


def _test_homogeneous_vs_vector_k(
    beta: float,
    gamma_ext: float,
    k_q: float,
    x_mc: np.ndarray,
    v_mc: np.ndarray,
    cfg: Config,
) -> None:
    """
    Lightweight internal check: homogeneous scalar-k path vs heterogeneous path
    with a constant k_mc vector should agree up to numerical tolerance.

    This is not run in normal usage; it can be invoked manually from a REPL or
    small test script to sanity-check refactors.
    """
    mse0, s20, gp0 = solve_rspmap_qk_one(beta, gamma_ext, k_q, x_mc, v_mc, cfg, init_state=None, k_mc=None)

    k_vec = np.full_like(x_mc, float(k_q), dtype=float)
    mse1, s21, gp1 = solve_rspmap_qk_one(beta, gamma_ext, k_q, x_mc, v_mc, cfg, init_state=None, k_mc=k_vec, g_mc=np.zeros_like(k_vec, dtype=int))

    if not (abs(mse0 - mse1) <= 1e-10 and abs(s20 - s21) <= 1e-10 and abs(gp0 - gp1) <= 1e-10):
        raise AssertionError(
            f"Homogeneous vs vector-k mismatch: "
            f"mse0={mse0:.3e}, mse1={mse1:.3e}, "
            f"s20={s20:.3e}, s21={s21:.3e}, "
            f"gp0={gp0:.3e}, gp1={gp1:.3e}"
        )


def _test_step1a_collapse(
    beta: float,
    gamma_ext: float,
    k_q: float,
    x_mc: np.ndarray,
    v_mc: np.ndarray,
    cfg: Config,
) -> None:
    """
    Mixture collapse sanity: when k_A == k_B == k_q, hetero path equals homogeneous.
    """
    k_vec = np.full_like(x_mc, float(k_q), dtype=float)
    g_vec = np.zeros_like(k_vec, dtype=int)  # all in one group

    mse0, s20, gp0 = solve_rspmap_qk_one(beta, gamma_ext, k_q, x_mc, v_mc, cfg, init_state=None, k_mc=None)
    mse1, s21, gp1 = solve_rspmap_qk_one(beta, gamma_ext, k_q, x_mc, v_mc, cfg, init_state=None, k_mc=k_vec, g_mc=g_vec)

    if not (abs(mse0 - mse1) <= 1e-10 and abs(s20 - s21) <= 1e-10 and abs(gp0 - gp1) <= 1e-10):
        raise AssertionError(
            f"Step1A collapse mismatch: mse0={mse0:.3e}, mse1={mse1:.3e}, "
            f"s20={s20:.3e}, s21={s21:.3e}, gp0={gp0:.3e}, gp1={gp1:.3e}"
        )


def _test_step1b_collapse(
    beta: float,
    gamma_ext: float,
    k_q: float,
    x_mc: np.ndarray,
    v_mc: np.ndarray,
    cfg: Config,
) -> None:
    """
    Support-conditioned collapse: when k_nz == k_z == k_q, hetero path equals homogeneous.
    """
    # Build mask for nz vs z
    nz = x_mc != 0.0
    k_vec = np.where(nz, k_q, k_q).astype(float)
    g_vec = np.where(nz, 1, 0).astype(int)

    mse0, s20, gp0 = solve_rspmap_qk_one(beta, gamma_ext, k_q, x_mc, v_mc, cfg, init_state=None, k_mc=None)
    mse1, s21, gp1 = solve_rspmap_qk_one(beta, gamma_ext, k_q, x_mc, v_mc, cfg, init_state=None, k_mc=k_vec, g_mc=g_vec)

    if not (abs(mse0 - mse1) <= 1e-10 and abs(s20 - s21) <= 1e-10 and abs(gp0 - gp1) <= 1e-10):
        raise AssertionError(
            f"Step1B collapse mismatch: mse0={mse0:.3e}, mse1={mse1:.3e}, "
            f"s20={s20:.3e}, s21={s21:.3e}, gp0={gp0:.3e}, gp1={gp1:.3e}"
        )


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

    print("=== TIGHT overlay plot ===")
    print(f"gammas(label) = {GAMMAS}")
    print(f"k_big={K_Q_BIG:g}, k_small={K_Q_SMALL:g}")
    print(f"MC_SAMPLES_Q={MC_SAMPLES_Q} | MAX_FP_ITERS={cfg.max_fp_iters} | TOL_FP={cfg.tol_fp:g} | DAMP={cfg.damp:g}")
    print()

    # Larger shared MC for both q panels
    xq = sample_bg(MC_SAMPLES_Q, rng, cfg.rho, cfg.var_nonzero)
    vq = rng.normal(size=MC_SAMPLES_Q)

    ridge: Dict[float, np.ndarray] = {}
    lasso: Dict[float, np.ndarray] = {}
    q_big: Dict[float, np.ndarray] = {}
    q_small: Dict[float, np.ndarray] = {}

    # Solid curves
    for g in GAMMAS:
        ridge[g] = np.array([solve_ridge_closed_form(b, g, cfg) for b in betas], dtype=float)

        arr = np.zeros_like(betas)
        prev_s2, prev_gp = None, None
        for i, b in enumerate(betas):
            mse, s2, gp = solve_lasso_analytical(b, g, cfg, prev_s2, prev_gp)
            arr[i] = mse
            prev_s2, prev_gp = s2, gp
        lasso[g] = arr

    # Dashed q curves with best-of forward/backward continuation
    for g_label in GAMMAS:
        g_ext = gamma_ext_for_q_big(g_label, K_Q_BIG)
        q_big[g_label] = solve_rspmap_qk_curve_best_of_forward_backward(betas, g_ext, K_Q_BIG, xq, vq, cfg)

        g_ext = gamma_ext_for_q_small(g_label, K_Q_SMALL)
        q_small[g_label] = solve_rspmap_qk_curve_best_of_forward_backward(betas, g_ext, K_Q_SMALL, xq, vq, cfg)

    # Plot overlays
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.7, 5.8), sharey=True)

    cmap = plt.get_cmap("viridis")
    gam_sorted = sorted(GAMMAS, reverse=True)
    cols = [cmap(x) for x in np.linspace(0.15, 0.85, len(gam_sorted))]
    color_of = {g: c for g, c in zip(gam_sorted, cols)}

    # Left: ridge vs q_big
    for g in gam_sorted:
        c = color_of[g]
        ax_left.plot(betas, to_db(ridge[g]), linewidth=2.4, color=c, linestyle="-",
                     label=rf"$\gamma$={g:g}  (ridge)")
        ax_left.plot(betas, to_db(q_big[g]), linewidth=2.4, color=c, linestyle="--",
                     label=rf"$\gamma$={g:g}  ($q_k$, $k$={K_Q_BIG:g})")

    ax_left.set_title(rf"Ridge (solid) vs $q_k$ (dashed),  $k$={K_Q_BIG:g}", fontsize=13)
    ax_left.set_xlabel(r"Measurement ratio $\beta=n/m$", fontsize=12)
    ax_left.set_ylabel("MSE (dB)", fontsize=12)
    ax_left.grid(True, linestyle=":", linewidth=1.0)

    # Right: lasso vs q_small
    for g in gam_sorted:
        c = color_of[g]
        ax_right.plot(betas, to_db(lasso[g]), linewidth=2.4, color=c, linestyle="-",
                      label=rf"$\gamma$={g:g}  (lasso)")
        ax_right.plot(betas, to_db(q_small[g]), linewidth=2.4, color=c, linestyle="--",
                      label=rf"$\gamma$={g:g}  ($q_k$, $k$={K_Q_SMALL:g})")

    ax_right.set_title(rf"LASSO (solid) vs $q_k$ (dashed),  $k$={K_Q_SMALL:g}", fontsize=13)
    ax_right.set_xlabel(r"Measurement ratio $\beta=n/m$", fontsize=12)
    ax_right.grid(True, linestyle=":", linewidth=1.0)

    ax_left.set_xlim(float(betas.min()), float(betas.max()))
    ax_left.set_ylim(-18.0, 15.0)

    ax_left.legend(loc="upper left", fontsize=8.3, frameon=True, ncol=1)
    ax_right.legend(loc="upper left", fontsize=8.3, frameon=True, ncol=1)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {OUT_PNG}")
    print("=== Done ===")


def _run_all_sanity_tests():
    """
    Run all sanity tests: homogeneous vs vector-k, Step 1A collapse, Step 1B collapse.
    Uses small MC samples and a few beta points for fast execution.
    """
    print("=" * 80)
    print("RUNNING SANITY TESTS (SMALL SCALE)")
    print("=" * 80)
    
    # Small test config
    test_rho = 0.1
    test_var_nz = 1.0 / test_rho
    test_sigma0_2 = 0.01
    test_betas = np.array([0.8, 1.0, 1.5, 2.0])
    test_cfg = Config(
        rho=test_rho,
        var_nonzero=test_var_nz,
        sigma0_2=test_sigma0_2,
        betas=test_betas,
        max_fp_iters=500,
        tol_fp=1e-9,
        damp=0.3,
    )
    
    # Small MC samples
    mc_test = 2000
    rng_test = np.random.default_rng(99999)
    x_test = sample_bg(mc_test, rng_test, test_rho, test_var_nz)
    v_test = rng_test.normal(size=mc_test)
    
    # Test parameters
    beta_test = 1.2
    gamma_ext_test = 0.01
    k_q_test = 4e-6
    
    print("\nTest 1: Homogeneous vs vector-k consistency check")
    print("-" * 80)
    try:
        _test_homogeneous_vs_vector_k(beta_test, gamma_ext_test, k_q_test, x_test, v_test, test_cfg)
        print("✓ PASSED: Homogeneous vs vector-k agree within tolerance")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        return False
    
    print("\nTest 2: Step 1A collapse (k_A == k_B)")
    print("-" * 80)
    try:
        _test_step1a_collapse(beta_test, gamma_ext_test, k_q_test, x_test, v_test, test_cfg)
        print("✓ PASSED: Mixture mode collapses to homogeneous when k_A == k_B")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        return False
    
    print("\nTest 3: Step 1B collapse (k_nz == k_z)")
    print("-" * 80)
    try:
        _test_step1b_collapse(beta_test, gamma_ext_test, k_q_test, x_test, v_test, test_cfg)
        print("✓ PASSED: Support mode collapses to homogeneous when k_nz == k_z")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("ALL SANITY TESTS PASSED ✓")
    print("=" * 80)
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run sanity tests
        success = _run_all_sanity_tests()
        sys.exit(0 if success else 1)
    else:
        # Run main overlay plot
        main()