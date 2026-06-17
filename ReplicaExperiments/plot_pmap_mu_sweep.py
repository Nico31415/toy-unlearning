#!/usr/bin/env python3
"""Plot PMAP curves under different effective loss weights.

This script treats (mu_f, mu_r) as quadratic effective-teacher weights:

    t_f = mu_f / (mu_f + mu_r),    t_r = mu_r / (mu_f + mu_r).

For Stage 2 this interpolates between standard unlearning
(mu_f, mu_r)=(0,1), i.e. target [0_F, beta_R, 0], and softer targets.
For Stage 3 it interpolates the adversary's effective target in the same way.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare_cpt_replica import bregman_prox, compute_ft_geometry, sigma2_qk_local


def _solve_pmap_curve(
    alphas: np.ndarray,
    beta_eff: np.ndarray,
    beta_ctr: np.ndarray,
    k_all: np.ndarray,
    v: np.ndarray,
    sigma0_sq: float,
    gamma_ext: float = 0.0,
    max_se_iters: int = 1000,
    damp: float = 0.25,
    tol_fp: float = 1e-10,
) -> np.ndarray:
    """Full PMAP fixed-point curve with 1/alpha convention."""
    xhat_out = np.zeros((len(alphas), beta_eff.size))
    prev_state = None
    s2_cap = 1e14

    for i, alpha in enumerate(alphas):
        beta = 1.0 / alpha
        if prev_state is None:
            prior_mse = float(np.mean((beta_eff - beta_ctr) ** 2))
            s2 = sigma0_sq + beta * prior_mse
            sig2_ctr = sigma2_qk_local(beta_ctr, 1.0, k_all)
            gp = max(gamma_ext + beta * float(np.mean(sig2_ctr)), 1e-14)
        else:
            s2, gp = prev_state
            s2 = max(s2, sigma0_sq, 1e-20)
            gp = max(gp, gamma_ext, 1e-14)

        for _ in range(max_se_iters):
            s2_eff = min(max(s2, 1e-20), s2_cap)
            z = beta_eff + math.sqrt(s2_eff) * v
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
            s2 = min(max(s2, sigma0_sq, 1e-20), s2_cap)
            gp = max(gp, gamma_ext, 1e-14)

        s2_eff = min(max(s2, 1e-20), s2_cap)
        z = beta_eff + math.sqrt(s2_eff) * v
        xhat_out[i] = bregman_prox(z, gp, k_all, beta_ctr)
        prev_state = (s2, gp)

    return xhat_out


def _target_multipliers(mu_f: float, mu_r: float) -> tuple[float, float]:
    denom = mu_f + mu_r
    if denom <= 0:
        raise ValueError("mu_f + mu_r must be positive")
    return mu_f / denom, mu_r / denom


def main() -> None:
    rho_pt = 0.1
    p_forget = 0.5
    sigma0_sq = 1e-6
    c_pt = 0.2
    lambda_pt = 0.0
    alpha_in = 1.0
    alpha_out = 1.0
    n_mc = 20_000
    seed = 7

    mu_pairs = [
        (0.0, 1.0),
        (0.25, 1.0),
        (1.0, 1.0),
        (1.0, 0.25),
        (1.0, 0.0),
    ]
    alphas_ul = np.linspace(0.02, 1.0, 8)
    alphas_rl = np.linspace(0.02, 1.0, 8)

    rng = np.random.default_rng(seed)
    var_nz = 1.0 / rho_pt
    n_f = int(round(n_mc * rho_pt * p_forget))
    n_r = int(round(n_mc * rho_pt * (1.0 - p_forget)))
    n_0 = n_mc - n_f - n_r

    beta_f = rng.normal(0.0, math.sqrt(var_nz), n_f)
    beta_r = rng.normal(0.0, math.sqrt(var_nz), n_r)
    beta_0 = np.zeros(n_0)
    beta_pt = np.concatenate([beta_f, beta_r, beta_0])
    v_ul = rng.standard_normal(n_mc)
    v_rl = rng.standard_normal(n_mc)

    k_all, beta_ctr_ul, _ = compute_ft_geometry(
        beta_pt,
        c_pt,
        lambda_pt,
        alpha_in,
        alpha_out,
        gamma_reinit=None,
        average_inputs=False,
    )

    out_dir = Path(__file__).parent / "compare_cpt_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_ul, ax_ul = plt.subplots(1, 1, figsize=(7, 4.8))
    fig_rl, ax_rl = plt.subplots(1, 1, figsize=(7, 4.8))

    for mu_f, mu_r in mu_pairs:
        t_f, t_r = _target_multipliers(mu_f, mu_r)
        label = rf"$\mu_f={mu_f:g},\,\mu_r={mu_r:g}$"
        print(f"Running mu_f={mu_f:g}, mu_r={mu_r:g}  (t_f={t_f:.3f}, t_r={t_r:.3f})", flush=True)

        beta_eff_ul = np.concatenate([t_f * beta_f, t_r * beta_r, beta_0])
        xhat_ul = _solve_pmap_curve(
            alphas_ul, beta_eff_ul, beta_ctr_ul, k_all, v_ul, sigma0_sq
        )

        mse_truth_ul = np.mean((xhat_ul - beta_pt[None, :]) ** 2, axis=1)
        ax_ul.plot(alphas_ul, mse_truth_ul, lw=2, marker="o", ms=4, label=label)

        beta_eff_rl = np.concatenate([t_f * beta_f, t_r * beta_r, beta_0])
        beta_ctr_rl = np.concatenate([np.zeros(n_f), beta_r, beta_0])
        xhat_rl = _solve_pmap_curve(
            alphas_rl, beta_eff_rl, beta_ctr_rl, k_all, v_rl, sigma0_sq
        )

        mse_truth_rl = np.mean((xhat_rl - beta_pt[None, :]) ** 2, axis=1)
        ax_rl.plot(alphas_rl, mse_truth_rl, lw=2, marker="o", ms=4, label=label)

    ax_ul.set_title("Stage 2: overall MSE to ground truth")
    ax_ul.set_xlabel(r"$\alpha_{\rm UL}$")
    ax_ul.set_ylabel(r"$\mathrm{MSE}_{\rm UL} = D^{-1}\|\hat\beta_{\rm UL}-\beta^*_{\rm PT}\|_2^2$")
    ax_ul.grid(True, alpha=0.3)
    ax_ul.legend(fontsize=8)
    fig_ul.suptitle(
        rf"Stage 2 PMAP under effective loss weights "
        rf"($c_{{PT}}={c_pt}$, $\rho={rho_pt}$, $p_f={p_forget}$)"
    )
    fig_ul.tight_layout()
    ul_path = out_dir / "stage2_pmap_mu_sweep.png"
    fig_ul.savefig(ul_path, dpi=170)
    plt.close(fig_ul)

    ax_rl.axhline(rho_pt * var_nz, color="0.5", ls=":", lw=1, label="zero predictor")
    ax_rl.set_title("Stage 3: overall MSE to ground truth")
    ax_rl.set_xlabel(r"$\alpha_{\rm RL}$")
    ax_rl.set_ylabel(r"$\mathrm{MSE}_{\rm RL} = D^{-1}\|\hat\beta_{\rm RL}-\beta^*_{\rm PT}\|_2^2$")
    ax_rl.grid(True, alpha=0.3)
    ax_rl.legend(fontsize=8)
    fig_rl.suptitle(
        rf"Stage 3 PMAP under effective loss weights "
        rf"($c_{{PT}}={c_pt}$, deterministic unlearned center)"
    )
    fig_rl.tight_layout()
    rl_path = out_dir / "stage3_pmap_mu_sweep.png"
    fig_rl.savefig(rl_path, dpi=170)
    plt.close(fig_rl)

    print(f"Saved {ul_path}")
    print(f"Saved {rl_path}")


if __name__ == "__main__":
    main()
