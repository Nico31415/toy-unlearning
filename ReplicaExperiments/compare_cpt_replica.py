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

sys.path.insert(0, str(Path(__file__).parent))
from fixed_lambda_all import prox_qk_safeguarded, sigma2_qk


# ──────────────────────────────────────────────────────────────────────────────
# Bregman proximal operator
# ──────────────────────────────────────────────────────────────────────────────

def bregman_prox(z: np.ndarray, lam: float, k: float,
                 beta0: np.ndarray) -> np.ndarray:
    """prox_{λ D_{q_k}(·, β₀)}(z) = prox_{λ q_k}(z + λ q'_k(β₀))"""
    shift = lam * 0.5 * np.arcsinh(2.0 * beta0 / math.sqrt(max(k, 1e-30)))
    return prox_qk_safeguarded(z + shift, lam, k)


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
    k = max(float(k), 1e-30)

    if init_state is None:
        prior_mse = float(np.mean((target_mc - center_mc) ** 2))
        s2 = sigma0_sq + alpha * prior_mse
        s2 = max(s2, 1e-6)
        mean_sig2 = float(np.mean(sigma2_qk(center_mc, 1.0, k)))
        gp = gamma_ext + alpha * mean_sig2
        gp = max(gp, 1e-14)
    else:
        s2 = max(float(init_state[0]), sigma0_sq, 1e-15)
        gp = max(float(init_state[1]), gamma_ext, 1e-14)

    for _ in range(max_iters):
        z = target_mc + math.sqrt(max(s2, 1e-15)) * v_mc
        xhat = bregman_prox(z, gp, k, center_mc)

        mse = float(np.mean((target_mc - xhat) ** 2))
        ms2 = float(np.mean(sigma2_qk(xhat, gp, k)))

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
    **fp_kw,
):
    """
    Compute gen_err = E[(xhat - beta_ref)²] on mask_eval, for each alpha.
    Uses forward + backward sweeps; picks lower mse_target (physical FP).
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

    # Pointwise: pick branch with LOWER target-MSE (minimise free energy proxy)
    gen_err = np.zeros(n)
    for i in range(n):
        mse_f = float(np.mean((target_mc - xhat_f[i]) ** 2))
        mse_bwd = float(np.mean((target_mc - xhat_b[i]) ** 2))
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
    # Stage 2: Unlearning
    # ─────────────────────────────────────────────────────────────────────────
    stage2 = {}   # c_pt → (err_F, err_R)
    for c_pt in c_pt_list:
        k = 4.0 * c_pt ** 2
        print(f"\n=== Stage 2: c_PT={c_pt}  k={k:.2e} ===")

        # For Stage 2: target = β_eff_UL, center = β*_PT (oracle)
        # gen_err_F = E[(β̂_UL_F - β*_PT_F)²] on mask_F
        err_F = solve_curve(
            alpha_ul, eff_ul, beta_pt, v_ul, k, sigma0_sq,
            mask_F, beta_pt, **fp_kw
        )
        err_R = solve_curve(
            alpha_ul, eff_ul, beta_pt, v_ul, k, sigma0_sq,
            mask_R, beta_pt, **fp_kw
        )
        stage2[c_pt] = (err_F, err_R)

        for i in [0, 5, 10, 20, 29, 35, 45]:
            if i < len(alpha_ul):
                print(f"  α={alpha_ul[i]:.3f}  err_F={err_F[i]:.4f}  err_R={err_R[i]:.4f}")

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

    stage3 = {}   # c_pt → err_RL
    for c_pt in c_pt_list:
        k = 4.0 * c_pt ** 2
        print(f"\n=== Stage 3: c_PT={c_pt}  k={k:.2e} ===")

        err_RL = solve_curve(
            alpha_rl, eff_rl, center_rl_analytic, v_rl, k, sigma0_sq,
            mask_F, beta_pt, **fp_kw
        )
        stage3[c_pt] = err_RL

        for i in [0, 5, 10, 20, 29, 35, 45]:
            if i < len(alpha_rl):
                print(f"  α={alpha_rl[i]:.3f}  err_RL_F={err_RL[i]:.4f}")

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
        k = 4 * c_pt**2
        err_F, err_R = stage2[c_pt]
        lbl = fr"$c_{{\rm PT}}={c_pt:.0e}$ ($k={k:.1e}$)"
        ax1.plot(alpha_ul, err_F, '-o', ms=2.5, color=col, label=lbl)
        ax2.plot(alpha_ul, err_R, '-o', ms=2.5, color=col, label=lbl)

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
        k = 4 * c_pt**2
        lbl = fr"$c_{{\rm PT}}={c_pt:.0e}$ ($k={k:.1e}$)"
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


if __name__ == "__main__":
    main()
