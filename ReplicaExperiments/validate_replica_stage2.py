#!/usr/bin/env python3
"""
Validate oracle SE (Stage 2) against direct Bregman-regularized regression.

Metric: finetuning validation MSE = (1/D)||beta_hat - eff_ul||^2
where eff_ul = [0_F, beta_PT_R, 0_inactive] is the unlearning target.

For each alpha, we binary-search for gp_reg such that mse_exp = mse_SE.
This shows the oracle SE correctly characterises the Pareto frontier.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ReplicaExperiments"))

from compare_cpt_replica import (
    _oracle_gp, bregman_prox, compute_k_eff,
)

# ── parameters ────────────────────────────────────────────────────────────────
RHO_PT    = 0.1
P_FORGET  = 0.5
C_PT      = 0.2
LAMBDA_PT = 0.1
D         = 500
SEEDS     = list(range(10))
ALPHA_LIST  = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
ALPHA_IN    = 0.7
ALPHA_OUT   = 1.5

# Warm-start guesses for binary search (will be loose but bracket-expanded if needed)
_GP_REG_INIT = {0.05: 30.0, 0.1: 25.0, 0.2: 10.0, 0.3: 5.0, 0.5: 1.0, 0.7: 0.2}


def sample_bg(D, rho, seed):
    torch.manual_seed(seed)
    mask = (torch.rand(D) < rho).float()
    vals = torch.randn(D) / math.sqrt(rho)
    return mask * vals


def bregman_div(beta, beta0, k):
    # k can be a scalar float or a 1-D tensor of per-coordinate values
    if not isinstance(k, torch.Tensor):
        k = torch.tensor(k, dtype=beta.dtype)
    sq = lambda x: torch.sqrt(k + 4.0 * x ** 2)
    return (sq(beta) / 4 - sq(beta0) / 4 - beta0 / sq(beta0) * (beta - beta0)).sum()


def run_one(seed, alpha, gp_reg):
    beta_PT = sample_bg(D, RHO_PT, seed)

    active = torch.where(beta_PT != 0)[0]
    n_f = max(1, int(round(P_FORGET * len(active))))
    torch.manual_seed(seed + 1000)
    perm   = torch.randperm(len(active))
    mask_f = active[perm[:n_f]]

    eff_ul = beta_PT.clone()
    eff_ul[mask_f] = 0.0

    N = max(1, int(round(alpha * D)))
    torch.manual_seed(seed + 200 + int(alpha * 1000))
    X = torch.randn(N, D) / math.sqrt(D)
    y = X @ eff_ul

    # Per-coordinate k_eff (depends on beta_PT and alpha_in/alpha_out)
    k_np  = compute_k_eff(beta_PT.numpy(), C_PT, LAMBDA_PT, ALPHA_IN, ALPHA_OUT)
    k_tor = torch.tensor(k_np, dtype=torch.float64)
    beta_PT_d = beta_PT.double()

    beta = beta_PT_d.clone().detach().requires_grad_(True)
    opt  = torch.optim.LBFGS([beta], lr=0.5, max_iter=300,
                              line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (gp_reg * bregman_div(beta, beta_PT_d, k_tor)
                + (1.0 / N) * ((X.double() @ beta - y.double()) ** 2).sum())
        loss.backward()
        return loss

    for _ in range(5):
        opt.step(closure)

    beta_hat = beta.detach()
    return ((beta_hat - eff_ul.double()) ** 2).mean().item()


def oracle_se(alpha, n_mc=80_000, seed=2024, max_se_iters=2000):
    """Oracle SE prediction for mse_ft = E[(xhat - eff_ul)^2]."""
    var_nz = 1.0 / RHO_PT
    rng  = np.random.default_rng(seed)
    n_f  = int(round(n_mc * RHO_PT * P_FORGET))
    n_r  = int(round(n_mc * RHO_PT * (1 - P_FORGET)))
    n_0  = n_mc - n_f - n_r

    bF = rng.normal(0.0, math.sqrt(var_nz), n_f)
    bR = rng.normal(0.0, math.sqrt(var_nz), n_r)

    beta_eff = np.concatenate([np.zeros(n_f), bR, np.zeros(n_0)])
    beta_ctr = np.concatenate([bF, bR, np.zeros(n_0)])
    k_all    = compute_k_eff(np.concatenate([bF, bR, np.zeros(n_0)]), C_PT, LAMBDA_PT, ALPHA_IN, ALPHA_OUT)
    k_F      = k_all[:n_f]

    gp = _oracle_gp(alpha, bF, k_F, var_nz)
    s2 = 0.0
    v  = rng.standard_normal(n_mc)
    for _ in range(max_se_iters):
        z    = beta_eff + math.sqrt(max(s2, 1e-20)) * v
        xhat = bregman_prox(z, gp, k_all, beta_ctr)
        mse  = float(np.mean((xhat - beta_eff) ** 2))
        s2_new = alpha * mse
        if abs(s2_new - s2) < 1e-6 * max(s2, 1e-20):
            break
        s2 = 0.9 * s2 + 0.1 * s2_new

    v    = rng.standard_normal(n_mc)
    z    = beta_eff + math.sqrt(max(s2, 1e-20)) * v
    xhat = bregman_prox(z, gp, k_all, beta_ctr)
    return float(np.mean((xhat - beta_eff) ** 2))


def _mean_mse(alpha, gp_reg, seeds):
    return float(np.mean([run_one(s, alpha, gp_reg) for s in seeds]))


def binary_search_gp_reg(alpha, mse_target, n_bs_seeds=5, n_iters=30, tol=1e-3):
    """
    Find gp_reg* such that mean mse_exp(gp_reg*, seeds) ≈ mse_target.

    Higher gp_reg → beta stays near beta_PT (far from eff_ul) → higher mse_ft.
    So we search in the direction: increase gp_reg if mse < target.
    """
    seeds = list(range(n_bs_seeds))
    lo = _GP_REG_INIT.get(alpha, 1.0) / 4.0
    hi = _GP_REG_INIT.get(alpha, 1.0) * 4.0

    # Expand bracket if needed
    for _ in range(10):
        if _mean_mse(alpha, lo, seeds) > mse_target:
            lo /= 4.0
        else:
            break
    for _ in range(10):
        if _mean_mse(alpha, hi, seeds) < mse_target:
            hi *= 4.0
        else:
            break

    for it in range(n_iters):
        mid = math.sqrt(lo * hi)
        mse_mid = _mean_mse(alpha, mid, seeds)
        if mse_mid < mse_target:
            lo = mid   # need higher gp_reg
        else:
            hi = mid   # need lower gp_reg
        if hi / lo < 1.0 + tol:
            break
        print(f"    bs iter {it:2d}: lo={lo:.3f} hi={hi:.3f} mid={mid:.3f} mse={mse_mid:.4f} target={mse_target:.4f}", flush=True)

    return math.sqrt(lo * hi)


def oracle_finetune_mse_curve(alphas, n_mc=80_000, seed=2024):
    """Dense oracle SE curve for plotting."""
    return np.array([oracle_se(a, n_mc=n_mc, seed=seed) for a in alphas])


def main():
    out_dir = Path(__file__).parent / "compare_cpt_figures"
    out_dir.mkdir(exist_ok=True)

    print("Computing oracle SE for each alpha ...", flush=True)
    mse_se = {}
    for alpha in ALPHA_LIST:
        mse_se[alpha] = oracle_se(alpha)
        print(f"  alpha={alpha:.2f}: mse_SE={mse_se[alpha]:.4f}", flush=True)

    print("\nBinary-searching gp_reg* per alpha ...", flush=True)
    gp_reg_star = {}
    for alpha in ALPHA_LIST:
        print(f"  alpha={alpha:.2f} (target mse={mse_se[alpha]:.4f}) ...", flush=True)
        gp_reg_star[alpha] = binary_search_gp_reg(alpha, mse_se[alpha])
        print(f"  -> gp_reg*={gp_reg_star[alpha]:.4f}", flush=True)

    print("\nRunning main experiment (10 seeds per alpha) ...", flush=True)
    results = {}
    for alpha in ALPHA_LIST:
        for seed in SEEDS:
            mse_ft = run_one(seed, alpha, gp_reg_star[alpha])
            results[(alpha, seed)] = mse_ft
        vals = [results[(alpha, s)] for s in SEEDS]
        print(f"  alpha={alpha:.2f}: mean={np.mean(vals):.4f}  SE={mse_se[alpha]:.4f}  "
              f"ratio={np.mean(vals)/mse_se[alpha]:.3f}", flush=True)

    print("\nComputing dense oracle SE curve ...", flush=True)
    alpha_grid = np.linspace(0.02, 0.80, 50)
    mse_ft_grid = oracle_finetune_mse_curve(alpha_grid)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(alpha_grid, mse_ft_grid, '-', color='C0', lw=2,
            label="Oracle SE (theory)")

    exp_means = [np.mean([results[(a, s)] for s in SEEDS]) for a in ALPHA_LIST]
    exp_stds  = [np.std([results[(a, s)] for s in SEEDS]) for a in ALPHA_LIST]
    ax.errorbar(ALPHA_LIST, exp_means, yerr=exp_stds,
                fmt='o', color='C1', ms=7, capsize=4,
                label=f"Bregman reg. (mean±std, {len(SEEDS)} seeds, $D={D}$)")

    ax.set_xlabel(r"$\alpha = N_{\rm UL}/D$", fontsize=12)
    ax.set_ylabel(r"$\frac{1}{D}\|\hat\beta - \beta_{\rm eff}\|^2$", fontsize=12)
    ax.set_title(
        fr"Oracle SE validation: finetuning MSE vs $\alpha$  "
        fr"($c_{{PT}}={C_PT}$, $\alpha_{{in}}={ALPHA_IN}$, $\alpha_{{out}}={ALPHA_OUT}$, "
        fr"$\rho_{{PT}}={RHO_PT}$, $p_F={P_FORGET}$)",
        fontsize=9,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.005)

    plt.tight_layout()
    p = out_dir / f"validation_stage2_cpt{C_PT}_lpt{LAMBDA_PT}_ain{ALPHA_IN}_aout{ALPHA_OUT}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {p}")

    # Print summary table
    print("\nalpha   mse_SE   gp_reg*  mse_exp   ratio")
    for alpha in ALPHA_LIST:
        vals = [results[(alpha, s)] for s in SEEDS]
        m = np.mean(vals)
        print(f"{alpha:6.2f}  {mse_se[alpha]:.4f}  {gp_reg_star[alpha]:7.3f}  {m:.4f}  {m/mse_se[alpha]:.3f}")


if __name__ == "__main__":
    main()
