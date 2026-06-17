#!/usr/bin/env python3
"""
Replica curves for the unlearning + relearning pipeline.

Setting
-------
- No readout reinitialization; no weight averaging after pretraining.
- Initial network function = pretrained predictor:  β₀ = β̂_PT ≠ 0.
- α_input = α_output = 1  (no layer rescaling in this script).
- Perfect pretraining (oracle):  β̂_PT = β*_PT  (α_PT ≥ 1).

Geometry (no rescaling, complex init, ρ = 1):
    k_i = 4 c_PT²   (uniform across all coordinates, λ_PT irrelevant)

Implicit bias (Theorem 1, Nonzero paper):
    Stage 2 (Unlearning):
        β̂_UL = arg min D_{Q_k}(β, β₀)   s.t.  X^T β = X^T β_eff_UL

    Stage 3 (Relearning):
        β̂_RL = arg min D_{Q_k}(β, β̂_UL)  s.t.  X^T β = X^T β_eff_RL

Key identity (Bregman proximal):
    prox_{λ D_{q_k}(·, β₀)}(z) = prox_{λ q_k}(z + λ · q'_k(β₀))
    where q'_k(x) = (1/2) arcsinh(2x/√k)

Generative model
----------------
D dimensions, partitioned into three groups:
  - F  (forget active):  fraction π_F = ρ_PT · p_forget
  - R  (retain active):  fraction π_R = ρ_PT · (1 - p_forget)
  - 0  (inactive):       fraction π_0 = 1 - ρ_PT

Active coordinate teacher:  β*_PT,i ~ N(0, 1/ρ_PT)  → unit E[β*²] = 1.

Unlearning target per group:
  - β_eff_UL,F = t_forget · β*_PT,F     (e.g. t_forget = 0: zero out forget features)
  - β_eff_UL,R = t_retain · β*_PT,R     (e.g. t_retain = 1: preserve retain features)
  - β_eff_UL,0 = 0

Relearning target (adversary recovers forget features, maintains retain):
  - β_eff_RL,F = β*_PT,F
  - β_eff_RL,R = β*_PT,R
  - β_eff_RL,0 = 0

Outputs
-------
Stage 2:  gen_err_forget(α_UL),  gen_err_retain(α_UL)
    = E[(β̂_UL,F - β*_PT,F)²],  E[(β̂_UL,R - β*_PT,R)²]

Stage 3:  gen_err_relearn(α_UL, α_RL)
    = E[(β̂_RL,F - β*_PT,F)²]  (how much of the forget feature the adversary recovers)

================================================================================
METHOD: FULL PMAP  (replica_derivation.pdf sec. 8.2 / 6.3)
================================================================================
Unlike compare_cpt_replica.py (which fixes γ* by an oracle calibration), this
script iterates the FULL PMAP state evolution — BOTH s² (SE1) and gp (SE2,
Onsager) self-consistently in solve_bregman_fp().  Consequences:

  * The two scripts use non-equivalent methods and will NOT coincide exactly;
    compare_cpt is a parametric γ*-family, this one is the GAMP/replica FP.

  * Multiple fixed points exist (theory Remark 6.1).  We resolve them with a
    forward+backward sweep and a per-α branch pick (Stage 2: higher target-MSE
    "nontrivial" FP; Stage 3: lower target-MSE "successful-relearn" FP).  NOTE:
    this max/min pick is a numerical HEURISTIC and departs from the theory's
    prescribed selection rule, which is the prior-MSE warm start of Remark 6.1
    (s₀² = (σ₀² + E[(β₀-β_eff)²])/α) followed by a single damped iteration.
    The warm start is implemented in solve_bregman_fp(); the extra sweep/pick
    layered on top in run_unlearning_replica()/run_relearning_replica() is the
    heuristic part.

  * The Stage-3 centre is the actual Stage-2 endpoint β̂_UL (a single MC sample
    path), which captures finite-α_UL randomness better than compare_cpt's
    deterministic-centre oracle but is still NOT the principled nested
    expectation of sec. 9.2 — see nested_cascade_replica.py for that.

SE convention: alpha = N/D with the divisive update s² = (σ₀² + MSE)/α (theory
eq. SE1), matching replica_derivation.pdf — NOT the reciprocal β=D/N convention
of fixed_lambda_all.py.
"""

from __future__ import annotations

import math
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- import shared proximal primitives from existing replica module ----------
sys.path.insert(0, str(Path(__file__).parent))
from fixed_lambda_all import prox_qk_safeguarded, sigma2_qk


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class UnlearningConfig:
    # PT initialisation (determines k and Bregman center)
    c_pt: float = 1e-2          # absolute scale; k = 4 c_pt²

    # Teacher / task
    rho_pt: float = 0.1          # PT sparsity (fraction of active dimensions)
    p_forget: float = 0.5        # fraction of active dims that are forget

    # Unlearning targets
    t_forget: float = 0.0        # β_eff_UL,F = t_forget · β*_PT,F  (0 = zero-out)
    t_retain: float = 1.0        # β_eff_UL,R = t_retain · β*_PT,R  (1 = keep)

    # Data scale sweeps
    alpha_ul: np.ndarray = field(
        default_factory=lambda: np.linspace(0.02, 1.0, 40))
    alpha_rl: np.ndarray = field(
        default_factory=lambda: np.linspace(0.02, 1.0, 40))

    # Fixed-point numerics
    mc_samples: int = 60_000
    max_fp_iters: int = 600
    tol_fp: float = 1e-10
    damp: float = 0.25
    sigma0_sq: float = 0.01      # small noise floor (avoids trivial FP; matches Anguita SNR~20dB)
    gamma_ext: float = 1e-9      # small external reg (approaches 0)

    # Reproducibility
    seed: int = 2024


# =============================================================================
# Bregman proximal operator
# =============================================================================

def bregman_prox(z: np.ndarray, lam: float, k: float,
                 beta0: np.ndarray) -> np.ndarray:
    """
    Proximal operator of the Bregman divergence D_{q_k}(·, β₀):

        prox_{λ D_{q_k}(·,β₀)}(z)
            = arg min_β { (z-β)²/(2λ) + q_k(β) - q'_k(β₀)·β }
            = prox_{λ q_k}(z + λ · q'_k(β₀))

    with  q'_k(x) = (1/2) arcsinh(2x/√k).
    """
    shift = lam * 0.5 * np.arcsinh(2.0 * beta0 / math.sqrt(k))
    return prox_qk_safeguarded(z + shift, lam, k)


# =============================================================================
# RS-PMAP fixed point with Bregman center
# =============================================================================

def solve_bregman_fp(
    alpha: float,
    target_mc: np.ndarray,
    center_mc: np.ndarray,
    v_mc: np.ndarray,
    k: float,
    cfg: UnlearningConfig,
    init_state: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, float, float, float]:
    """
    RS-PMAP fixed point for the Bregman-centered q_k regulariser.

    The implicit bias is
        β̂ = arg min D_{Q_k}(β, β₀)   s.t.   X^T β = X^T β_eff

    In the scalar Gaussian-channel replica this becomes:
        observation:  z = β_eff + √s² · v,      v ~ N(0,1)
        estimate:     x̂ = prox_{gp · q_k}(z + gp · q'_k(β₀))
    with self-consistency:
        s²_new  = σ₀² + α · E[(β_eff - x̂)²]
        gp_new  = γ_ext + α · E[σ²_{q_k}(x̂, gp, k)]

    Warm-start strategy:
        If no init_state is given, we initialise s2 from the prior MSE
        E[(β_eff − β₀)²] (the MSE at α = 0, before any data is seen).
        This avoids the trivial fixed point s2=0 that exists for σ₀²=0.

    Parameters
    ----------
    alpha      : data scale  N/D
    target_mc  : MC samples of β_eff (unlearning / relearning target), shape (n,)
    center_mc  : MC samples of β₀   (Bregman center), shape (n,)
    v_mc       : iid N(0,1) samples, shape (n,)
    k          : geometry parameter  (= 4 c_PT²)
    cfg        : UnlearningConfig
    init_state : optional (s2, gp) warm start

    Returns
    -------
    xhat   : converged denoised estimates, shape (n,)
    mse    : E[(β_eff - x̂)²]  (target-recovery MSE, drives fixed point)
    s2     : converged effective noise variance
    gp     : converged effective regularisation
    """
    alpha = float(alpha)

    if init_state is None:
        # Warm-start from the α=0 limit:
        #   At α=0 the estimator has no data → stays at the Bregman center β₀.
        #   So the initial MSE against β_eff is E[(β₀ - β_eff)²].
        #   This gives a nonzero s2 that steers the iteration toward the
        #   physically correct (nontrivial) fixed point.
        #
        # s2 = (σ₀² + E[(β₀ - β_eff)²]) / α   (prior-MSE warm start)
        # gp = α · mean σ²_{q_k}(β₀, 1, k)   (local variance at the center)
        #
        # We clamp s2 to at least 1e-6 so the first iteration introduces
        # enough noise to avoid the trivially-zero collapse.
        prior_mse = float(np.mean((target_mc - center_mc) ** 2))
        s2 = (cfg.sigma0_sq + prior_mse) / alpha
        s2 = max(s2, 1e-6)

        # Estimate gp from mean local variance evaluated at the center β₀.
        # sigma2_qk(x, lam=1, k) = 1/(1 + q''_k(x)) which is in (0,1].
        mean_sig2_at_center = float(np.mean(sigma2_qk(center_mc, 1.0, k)))
        gp = cfg.gamma_ext + alpha * mean_sig2_at_center
        gp = max(gp, 1e-14)
    else:
        s2 = float(max(init_state[0], cfg.sigma0_sq))
        gp = float(max(init_state[1], cfg.gamma_ext))
        s2 = max(s2, 1e-15)
        gp = max(gp, 1e-14)

    for _ in range(cfg.max_fp_iters):
        z = target_mc + math.sqrt(s2) * v_mc
        xhat = bregman_prox(z, gp, k, center_mc)

        mse = float(np.mean((target_mc - xhat) ** 2))
        mean_sigma2 = float(np.mean(sigma2_qk(xhat, gp, k)))

        s2_new = float((cfg.sigma0_sq + mse) / alpha)
        gp_new = float(cfg.gamma_ext + alpha * mean_sigma2)

        if max(abs(s2_new - s2), abs(gp_new - gp)) < cfg.tol_fp:
            return xhat, mse, s2_new, gp_new

        s2 = (1.0 - cfg.damp) * s2 + cfg.damp * s2_new
        gp = (1.0 - cfg.damp) * gp + cfg.damp * gp_new
        s2 = max(s2, cfg.sigma0_sq / alpha, 1e-15)
        gp = max(gp, cfg.gamma_ext, 1e-14)

    # Return best estimate after max iters
    z = target_mc + math.sqrt(s2) * v_mc
    xhat = bregman_prox(z, gp, k, center_mc)
    mse = float(np.mean((target_mc - xhat) ** 2))
    return xhat, mse, s2, gp


# =============================================================================
# MC sample generation
# =============================================================================

def sample_mc(cfg: UnlearningConfig, seed: int = None):
    """
    Draw MC samples for all coordinate groups.

    Returns
    -------
    beta_pt_mc      : pretrained teacher β*_PT per MC sample    (n_mc,)
    beta_eff_ul_mc  : unlearning target β_eff_UL per MC sample  (n_mc,)
    beta_eff_rl_mc  : relearning target β_eff_RL per MC sample  (n_mc,)
    v_ul            : iid N(0,1) for Stage-2 fixed point         (n_mc,)
    v_rl            : iid N(0,1) for Stage-3 fixed point         (n_mc,)
    group           : integer group label per sample  0=inactive, 1=forget, 2=retain
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    n = cfg.mc_samples

    # Group sizes
    n_F = int(round(n * cfg.rho_pt * cfg.p_forget))
    n_R = int(round(n * cfg.rho_pt * (1.0 - cfg.p_forget)))
    n_0 = n - n_F - n_R

    sigma_nz = math.sqrt(1.0 / cfg.rho_pt)   # std of active coordinates

    # Sample β*_PT for each group
    beta_F = rng.normal(0.0, sigma_nz, n_F)
    beta_R = rng.normal(0.0, sigma_nz, n_R)
    beta_0 = np.zeros(n_0)

    beta_pt_mc = np.concatenate([beta_F, beta_R, beta_0])

    # Unlearning targets
    eff_ul = np.concatenate([
        cfg.t_forget * beta_F,   # forget group: push toward t_forget * β*_PT
        cfg.t_retain * beta_R,   # retain group: keep at t_retain * β*_PT
        beta_0,
    ])

    # Relearning targets (adversary recovers the full pretrained teacher)
    eff_rl = beta_pt_mc.copy()

    # Group labels
    group = np.concatenate([
        np.ones(n_F, dtype=int),    # 1 = forget
        2 * np.ones(n_R, dtype=int),# 2 = retain
        np.zeros(n_0, dtype=int),   # 0 = inactive
    ])

    v_ul = rng.standard_normal(n)
    v_rl = rng.standard_normal(n)

    return beta_pt_mc, eff_ul, eff_rl, v_ul, v_rl, group


# =============================================================================
# Stage 2: Unlearning replica curves
# =============================================================================

def run_unlearning_replica(cfg: UnlearningConfig):
    """
    Compute Stage-2 unlearning generalization curves.

    Returns
    -------
    alpha_ul         : data scale array                              (n_alpha,)
    gen_err_forget   : E[(β̂_UL,F - β*_PT,F)²] per α_UL             (n_alpha,)
    gen_err_retain   : E[(β̂_UL,R - β*_PT,R)²] per α_UL             (n_alpha,)
    xhat_ul_all      : denoised UL estimates per α_UL  (n_alpha, n_mc)
                       (used as Stage-3 centers)
    """
    beta_pt, eff_ul, _, v_ul, _, group = sample_mc(cfg)

    k = 4.0 * cfg.c_pt ** 2
    center = beta_pt      # oracle: Bregman center = pretrained teacher
    mask_F = group == 1
    mask_R = group == 2

    n_alpha = len(cfg.alpha_ul)
    gen_err_forget = np.zeros(n_alpha)
    gen_err_retain = np.zeros(n_alpha)
    xhat_ul_all = np.zeros((n_alpha, cfg.mc_samples))

    # Forward sweep (small → large alpha)
    state_fwd = None
    xhat_fwd = np.zeros((n_alpha, cfg.mc_samples))
    err_F_fwd = np.zeros(n_alpha)
    err_R_fwd = np.zeros(n_alpha)
    for i, alpha in enumerate(cfg.alpha_ul):
        xhat, mse, s2, gp = solve_bregman_fp(
            alpha, eff_ul, center, v_ul, k, cfg, init_state=state_fwd
        )
        state_fwd = (s2, gp)
        xhat_fwd[i] = xhat
        if mask_F.any():
            err_F_fwd[i] = float(np.mean((xhat[mask_F] - beta_pt[mask_F]) ** 2))
        if mask_R.any():
            err_R_fwd[i] = float(np.mean((xhat[mask_R] - beta_pt[mask_R]) ** 2))

    # Backward sweep (large → small alpha): warm-start from largest alpha downward
    state_bwd = None
    xhat_bwd_rev = np.zeros((n_alpha, cfg.mc_samples))
    err_F_bwd_rev = np.zeros(n_alpha)
    err_R_bwd_rev = np.zeros(n_alpha)
    for j, alpha in enumerate(cfg.alpha_ul[::-1]):
        xhat, mse, s2, gp = solve_bregman_fp(
            alpha, eff_ul, center, v_ul, k, cfg, init_state=state_bwd
        )
        state_bwd = (s2, gp)
        xhat_bwd_rev[j] = xhat
        if mask_F.any():
            err_F_bwd_rev[j] = float(np.mean((xhat[mask_F] - beta_pt[mask_F]) ** 2))
        if mask_R.any():
            err_R_bwd_rev[j] = float(np.mean((xhat[mask_R] - beta_pt[mask_R]) ** 2))

    # Reverse the backward sweep arrays to align with alpha_ul order
    xhat_bwd = xhat_bwd_rev[::-1]
    err_F_bwd = err_F_bwd_rev[::-1]
    err_R_bwd = err_R_bwd_rev[::-1]

    # Select the physically correct branch at each alpha:
    # The correct solution has LOWER err_F for small alpha (Bregman center holds β̂ near β₀)
    # and HIGHER err_F for large alpha (data forces β̂ toward target=0, successful forgetting).
    # For the forget group: the correct MSE against target (eff_ul) is HIGHER for small alpha
    # (since β̂ stays near β₀ ≠ target).  We therefore pick the branch with higher
    # target-MSE, which corresponds to the nontrivial (physical) fixed point.
    # Concretely: compute target-MSE for both branches; pick the larger one at each alpha.
    mse_fwd = np.array([
        float(np.mean((eff_ul - xhat_fwd[i]) ** 2))
        for i in range(n_alpha)
    ])
    mse_bwd = np.array([
        float(np.mean((eff_ul - xhat_bwd[i]) ** 2))
        for i in range(n_alpha)
    ])
    use_bwd = mse_bwd > mse_fwd   # nontrivial FP has higher target-MSE

    for i in range(n_alpha):
        if use_bwd[i]:
            xhat_ul_all[i] = xhat_bwd[i]
            gen_err_forget[i] = err_F_bwd[i]
            gen_err_retain[i] = err_R_bwd[i]
        else:
            xhat_ul_all[i] = xhat_fwd[i]
            gen_err_forget[i] = err_F_fwd[i]
            gen_err_retain[i] = err_R_fwd[i]

    for i, alpha in enumerate(cfg.alpha_ul):
        branch = "bwd" if use_bwd[i] else "fwd"
        print(f"  UL α={alpha:.3f}  [{branch}]"
              f"  err_F={gen_err_forget[i]:.4f}  err_R={gen_err_retain[i]:.4f}"
              f"  mse_fwd={mse_fwd[i]:.4e}  mse_bwd={mse_bwd[i]:.4e}")

    return cfg.alpha_ul, gen_err_forget, gen_err_retain, xhat_ul_all


# =============================================================================
# Stage 3: Relearning replica curves
# =============================================================================

def run_relearning_replica(
    cfg: UnlearningConfig,
    xhat_ul_all: np.ndarray,
    alpha_ul_idx: int = -1,
):
    """
    Compute Stage-3 relearning generalization curves, for a fixed unlearning
    data scale (selected by alpha_ul_idx, default = last = most unlearned).

    The Bregman center for relearning is the Stage-2 endpoint β̂_UL.
    The geometry k^RL = k = 4 c_PT²  (conserved through unlearning GF).

    Parameters
    ----------
    xhat_ul_all   : Stage-2 denoised estimates, shape (n_alpha_ul, n_mc)
    alpha_ul_idx  : which UL data scale to use as starting point

    Returns
    -------
    alpha_rl         : data scale array                              (n_alpha,)
    gen_err_relearn  : E[(β̂_RL,F - β*_PT,F)²] per α_RL             (n_alpha,)
    gen_err_retain   : E[(β̂_RL,R - β*_PT,R)²] per α_RL  (retain maintained)
    """
    beta_pt, _, eff_rl, _, v_rl, group = sample_mc(cfg)

    k = 4.0 * cfg.c_pt ** 2
    center_rl = xhat_ul_all[alpha_ul_idx]   # β₀^RL = β̂_UL
    mask_F = group == 1
    mask_R = group == 2

    n_alpha = len(cfg.alpha_rl)
    gen_err_relearn = np.zeros(n_alpha)
    gen_err_retain = np.zeros(n_alpha)

    # Forward sweep
    state_fwd = None
    xhat_fwd_rl = np.zeros((n_alpha, cfg.mc_samples))
    for i, alpha in enumerate(cfg.alpha_rl):
        xhat, mse, s2, gp = solve_bregman_fp(
            alpha, eff_rl, center_rl, v_rl, k, cfg, init_state=state_fwd
        )
        state_fwd = (s2, gp)
        xhat_fwd_rl[i] = xhat

    # Backward sweep — start from trivial init (s2=sigma0_sq) so it finds
    # the low-s2 FP (adversary relearns) where it exists.
    state_bwd = (max(cfg.sigma0_sq, 1e-15), max(cfg.gamma_ext, 1e-14))
    xhat_bwd_rev_rl = np.zeros((n_alpha, cfg.mc_samples))
    for j, alpha in enumerate(cfg.alpha_rl[::-1]):
        xhat, mse, s2, gp = solve_bregman_fp(
            alpha, eff_rl, center_rl, v_rl, k, cfg, init_state=state_bwd
        )
        state_bwd = (s2, gp)
        xhat_bwd_rev_rl[j] = xhat
    xhat_bwd_rl = xhat_bwd_rev_rl[::-1]

    # Pick branch with LOWER target-MSE (adversary picks best relearning strategy)
    mse_fwd_rl = np.array([float(np.mean((eff_rl - xhat_fwd_rl[i]) ** 2)) for i in range(n_alpha)])
    mse_bwd_rl = np.array([float(np.mean((eff_rl - xhat_bwd_rl[i]) ** 2)) for i in range(n_alpha)])
    use_bwd_rl = mse_bwd_rl < mse_fwd_rl  # min-MSE branch

    for i in range(n_alpha):
        xhat = xhat_bwd_rl[i] if use_bwd_rl[i] else xhat_fwd_rl[i]
        if mask_F.any():
            gen_err_relearn[i] = float(np.mean((xhat[mask_F] - beta_pt[mask_F]) ** 2))
        if mask_R.any():
            gen_err_retain[i] = float(np.mean((xhat[mask_R] - beta_pt[mask_R]) ** 2))
        branch = "bwd" if use_bwd_rl[i] else "fwd"
        print(f"  RL α={cfg.alpha_rl[i]:.3f}  [{branch}]"
              f"  err_F={gen_err_relearn[i]:.4f}  err_R={gen_err_retain[i]:.4f}")

    return cfg.alpha_rl, gen_err_relearn, gen_err_retain


# =============================================================================
# Full pipeline: sweep over multiple UL scales + plot
# =============================================================================

def run_full_pipeline(cfg: UnlearningConfig, out_dir: Path = None):
    """
    Run Stage 2 (unlearning) and Stage 3 (relearning from each UL endpoint).
    Produce two figures:
      1. UL curves: forget error + retain error vs α_UL
      2. RL curves: relearning (forget recovery) error vs α_RL,
         one curve per selected α_UL value
    """
    out_dir = out_dir or Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    k = 4.0 * cfg.c_pt ** 2
    print(f"\n=== Unlearning Replica  |  c_PT={cfg.c_pt}  k={k:.3e}  "
          f"ρ_PT={cfg.rho_pt}  p_forget={cfg.p_forget} ===")
    print(f"    t_forget={cfg.t_forget}  t_retain={cfg.t_retain}\n")

    # -------------------------------------------------------------------
    # Stage 2
    # -------------------------------------------------------------------
    print("--- Stage 2: Unlearning ---")
    alpha_ul, err_F, err_R, xhat_ul_all = run_unlearning_replica(cfg)

    # -------------------------------------------------------------------
    # Stage 3 — relearn from a few UL checkpoints
    # -------------------------------------------------------------------
    # Pick a few UL data scales to start relearning from
    ul_idxs = [
        int(len(alpha_ul) * frac) - 1
        for frac in [0.25, 0.5, 0.75, 1.0]
    ]
    ul_idxs = [max(0, min(i, len(alpha_ul) - 1)) for i in ul_idxs]

    rl_curves = {}
    for idx in ul_idxs:
        a_ul_val = float(alpha_ul[idx])
        print(f"\n--- Stage 3: Relearning from α_UL={a_ul_val:.3f} ---")
        alpha_rl, err_RL, _ = run_relearning_replica(cfg, xhat_ul_all, alpha_ul_idx=idx)
        rl_curves[a_ul_val] = err_RL

    # -------------------------------------------------------------------
    # Figure 1: Unlearning curves
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f"Unlearning  |  $c_{{\\rm PT}}={cfg.c_pt}$,  "
        f"$\\rho_{{\\rm PT}}={cfg.rho_pt}$,  "
        f"$t_f={cfg.t_forget}$,  $t_r={cfg.t_retain}$",
        fontsize=11
    )

    ax = axes[0]
    ax.plot(alpha_ul, err_F, 'b-o', ms=3, label='Forget error')
    ax.set_xlabel(r'$\alpha_{\rm UL} = N_{\rm UL}/D$')
    ax.set_ylabel(r'$\mathcal{E}$  (gen. error)')
    ax.set_title('Stage 2: Forget group')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(alpha_ul, err_R, 'g-o', ms=3, label='Retain error')
    ax.set_xlabel(r'$\alpha_{\rm UL} = N_{\rm UL}/D$')
    ax.set_title('Stage 2: Retain group')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ul_path = out_dir / "unlearning_replica_curves.png"
    fig.savefig(ul_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {ul_path}")

    # -------------------------------------------------------------------
    # Figure 2: Relearning curves
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(
        f"Relearning (forget recovery)  |  $c_{{\\rm PT}}={cfg.c_pt}$,  "
        f"$\\rho_{{\\rm PT}}={cfg.rho_pt}$",
        fontsize=10
    )
    cmap = plt.cm.plasma
    colors = [cmap(v) for v in np.linspace(0.2, 0.85, len(rl_curves))]
    for color, (a_ul_val, err_RL) in zip(colors, sorted(rl_curves.items())):
        ax.plot(alpha_rl, err_RL, '-o', ms=3, color=color,
                label=f'$\\alpha_{{\\rm UL}}={a_ul_val:.2f}$')

    # Baseline: relearn from pretrained checkpoint (no unlearning), β̂_UL = β*_PT
    # In this case center = β*_PT, target = β*_PT, so error → 0 quickly
    ax.set_xlabel(r'$\alpha_{\rm RL} = N_{\rm RL}/D$')
    ax.set_ylabel(r'$\mathcal{E}_{\rm forget}$  (relearning error)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    rl_path = out_dir / "relearning_replica_curves.png"
    fig.savefig(rl_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {rl_path}")

    return {
        'alpha_ul': alpha_ul,
        'gen_err_forget': err_F,
        'gen_err_retain': err_R,
        'alpha_rl': alpha_rl,
        'rl_curves': rl_curves,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    # Default config: small c_PT (rich/dangerous regime) vs large c_PT (safe regime)
    for c_pt in [1e-3, 1e-1, 1.0]:
        cfg = UnlearningConfig(
            c_pt=c_pt,
            rho_pt=0.1,
            p_forget=0.5,
            t_forget=0.0,    # zero out forget features
            t_retain=1.0,    # preserve retain features
            alpha_ul=np.linspace(0.02, 0.8, 35),
            alpha_rl=np.linspace(0.02, 0.8, 35),
            mc_samples=60_000,
            gamma_ext=1e-9,
        )
        out_dir = Path(__file__).parent / f"unlearning_replica_cpt{c_pt}"
        run_full_pipeline(cfg, out_dir=out_dir)


if __name__ == "__main__":
    main()
