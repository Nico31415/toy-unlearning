#!/usr/bin/env python3
"""
Verify replica curves by directly computing the Bregman minimizer
(what DLN gradient flow converges to) via scipy, then comparing gen errors.

For each (alpha, c_pt) we solve:
    β̂ = argmin D_{q_k}(β, β₀)   s.t.  X β ≈ y
using a penalty formulation (small sigma0_sq → constrained limit).

Theory comparison:
    Stage 2 (UL, center=β*_PT, target=β_eff_UL):
        - expect gen_err_F → var_nz=10 for all alpha (perfect forgetting)
        - expect gen_err_R → 0         (retain preserved)
    Stage 3 (RL, center=β̂_UL≈0, target=β*_PT):
        - large k (c_pt=1.0): gen_err_F ≈ max(0,1-α)*var_nz  (Ridge formula)
        - small k (c_pt=1e-3): gen_err_F ≈ var_nz for α<α_c, 0 for α>α_c  (L1 CS)
"""

from __future__ import annotations
import math, sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from fixed_lambda_all import prox_qk_safeguarded, sigma2_qk

# ──────────────────────────────────────────────────────────────────────────────
# q_k potential and Bregman divergence  (from bregman.py inline)
# ──────────────────────────────────────────────────────────────────────────────

_EPS_K = 1e-30

def q_k(z, k):
    k = max(float(k), _EPS_K)
    sk = math.sqrt(k)
    t = 2.0 * np.asarray(z) / sk
    return (sk / 4.0) * (1.0 - np.sqrt(1.0 + t*t) + t * np.arcsinh(t))

def q_k_prime(z, k):
    k = max(float(k), _EPS_K)
    return 0.5 * np.arcsinh(2.0 * np.asarray(z) / math.sqrt(k))

def bregman_D(z, b, k):
    return q_k(z, k) - q_k(b, k) - q_k_prime(b, k) * (np.asarray(z) - np.asarray(b))

# ──────────────────────────────────────────────────────────────────────────────
# Direct Bregman minimizer via L-BFGS-B
# ──────────────────────────────────────────────────────────────────────────────

def bregman_minimizer(
    X: np.ndarray,      # (N, D)
    y: np.ndarray,      # (N,)
    beta0: np.ndarray,  # (D,) Bregman center
    k: float,
    sigma0_sq: float = 1e-5,  # small noise; approaches constrained problem as → 0
) -> np.ndarray:
    """
    Solve  min_{β}  D_{q_k}(β, β₀) + (1/(2*sigma0_sq*alpha)) * ||y - Xβ||²

    Returns β̂ at convergence.
    """
    D = X.shape[1]
    N = X.shape[0]
    reg_scale = 1.0 / (2.0 * sigma0_sq * N)

    def obj_and_grad(beta):
        # Bregman divergence sum
        bd   = bregman_D(beta, beta0, k).sum()
        # gradient of sum D_{q_k}(β, β₀) = sum q_k'(β) - q_k'(β₀)
        g_bd = q_k_prime(beta, k) - q_k_prime(beta0, k)
        # MSE term
        resid = X @ beta - y
        mse_val = reg_scale * np.dot(resid, resid)
        g_mse   = 2.0 * reg_scale * (X.T @ resid)
        return float(bd + mse_val), g_bd + g_mse

    beta_init = beta0.copy()
    result = minimize(
        obj_and_grad, beta_init, method='L-BFGS-B', jac=True,
        options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    return result.x


# ──────────────────────────────────────────────────────────────────────────────
# MC sampling
# ──────────────────────────────────────────────────────────────────────────────

def make_problem(D, rho_pt, p_forget, alpha, seed, sigma0_sq=1e-5):
    rng = np.random.default_rng(seed)
    sigma_nz = math.sqrt(1.0 / rho_pt)
    n_active  = int(round(D * rho_pt))
    n_F = int(round(n_active * p_forget))
    n_R = n_active - n_F

    beta_star = np.zeros(D)
    aidx = rng.choice(D, n_active, replace=False)
    beta_star[aidx] = rng.normal(0, sigma_nz, n_active)
    forget_idx = aidx[:n_F]
    retain_idx = aidx[n_F:]

    beta_eff_ul = np.zeros(D)
    beta_eff_ul[retain_idx] = beta_star[retain_idx]

    N = max(1, int(round(alpha * D)))
    X = rng.standard_normal((N, D)) / math.sqrt(D)

    return beta_star, beta_eff_ul, forget_idx, retain_idx, X


# ──────────────────────────────────────────────────────────────────────────────
# Replica reference (single-alpha, forward iteration from prior-MSE init)
# ──────────────────────────────────────────────────────────────────────────────

def _prox_qk_newton(v, lam, k, n_iter=80):
    lam = float(max(lam, 1e-14)); k = float(max(k, _EPS_K))
    sqk = math.sqrt(k); coeff = lam / 2.0
    v = np.asarray(v, dtype=float); x = v / (1.0 + 2.0 * coeff / sqk)
    for _ in range(n_iter):
        t = 2.0 * x / sqk; r = x + coeff * np.arcsinh(t) - v
        drdx = 1.0 + coeff * 2.0 / sqk / np.sqrt(1.0 + t**2); x = x - r / drdx
    return x


def bregman_prox_np(z, lam, k, beta0):
    lam = float(max(lam, 1e-14)); k = float(max(k, _EPS_K))
    shift = lam * 0.5 * np.arcsinh(2.0 * np.asarray(beta0) / math.sqrt(k))
    return _prox_qk_newton(np.asarray(z, dtype=float) + shift, lam, k)


def _oracle_gp(alpha, k, var_nz, n_mc=20_000, seed=0):
    """Find gp s.t. E[(bregman_prox(0,gp,k,β)−β)²] = α·var_nz (DECREASING in gp)."""
    rng = np.random.default_rng(seed)
    betas = rng.normal(0.0, math.sqrt(var_nz), n_mc)
    target = alpha * var_nz
    def err_at(gp):
        return float(np.mean((bregman_prox_np(np.zeros(n_mc), gp, k, betas) - betas)**2))
    e_lo = err_at(1e-8); e_hi = err_at(1e4)
    if target >= e_lo or target <= e_hi:
        return None
    return float(brentq(lambda g: err_at(g) - target, 1e-8, 1e4, xtol=1e-5, rtol=1e-5))


def stage2_oracle_curve_v(alphas, beta_pt_mc, eff_ul_mc, mask_F_mc, n_mc,
                           rho_pt, p_forget, sigma0_sq, k, seed=42):
    """Oracle-gp SE for Stage 2; returns err_F array over alphas (α ≤ 1 only)."""
    var_nz = 1.0 / rho_pt
    rng = np.random.default_rng(seed)
    errs = np.full(len(alphas), float('nan'))
    for i, alpha in enumerate(alphas):
        if alpha > 1.0:
            continue
        gp = _oracle_gp(alpha, k, var_nz, n_mc=n_mc, seed=seed + i + 1)
        if gp is None:
            continue
        s2 = sigma0_sq
        v  = rng.standard_normal(n_mc)
        for _ in range(2000):
            z    = eff_ul_mc + math.sqrt(max(s2, 1e-20)) * v
            xhat = bregman_prox_np(z, gp, k, beta_pt_mc)
            mse  = float(np.mean((xhat - eff_ul_mc)**2))
            s2n  = sigma0_sq + alpha * mse
            if abs(s2n - s2) < 1e-12: break
            s2 = 0.9 * s2 + 0.1 * s2n; s2 = max(s2, sigma0_sq)
        v    = rng.standard_normal(n_mc)
        z    = eff_ul_mc + math.sqrt(max(s2, 1e-20)) * v
        xhat = bregman_prox_np(z, gp, k, beta_pt_mc)
        errs[i] = float(np.mean((xhat[mask_F_mc] - beta_pt_mc[mask_F_mc])**2))
    return errs


def replica_sweep(alphas, target_mc, center_mc, v_mc, k, sigma0_sq=0.001, gamma_ext=1e-9):
    """
    Forward + backward sweep returning (xhat_fwd, xhat_bwd) arrays of shape (n_alpha, n_mc).
    """
    def _run_one(alpha, s2i, gpi):
        s2_, gp_ = s2i, gpi
        for _ in range(1000):
            z    = target_mc + math.sqrt(max(s2_, 1e-15)) * v_mc
            xhat = bregman_prox_np(z, gp_, k, center_mc)
            mse  = float(np.mean((target_mc - xhat)**2))
            ms2  = float(np.mean(sigma2_qk(xhat, gp_, k)))
            s2n  = float(sigma0_sq + alpha * mse)
            gpn  = float(gamma_ext + alpha * ms2)
            if max(abs(s2n-s2_), abs(gpn-gp_)) < 1e-10: break
            s2_ = (1-.25)*s2_ + .25*s2n;  s2_ = max(s2_, sigma0_sq, 1e-15)
            gp_ = (1-.25)*gp_ + .25*gpn;  gp_ = max(gp_, gamma_ext, 1e-14)
        z    = target_mc + math.sqrt(max(s2_, 1e-15)) * v_mc
        xhat = bregman_prox_np(z, gp_, k, center_mc)
        return xhat, s2_, gp_

    n = len(alphas)
    prior_mse = float(np.mean((target_mc - center_mc)**2))

    # Forward sweep: start from small (s2, gp) at smallest alpha
    xhat_fwd = []
    s2, gp = max(sigma0_sq + alphas[0]*prior_mse, 1e-6), max(gamma_ext, 1e-14)
    for a in alphas:
        xhat, s2, gp = _run_one(a, s2, gp)
        xhat_fwd.append(xhat)

    # Backward sweep: start from large (s2, gp) at largest alpha, sweep down
    xhat_bwd_rev = []
    s2 = max(sigma0_sq + alphas[-1]*prior_mse, 1e-6)
    gp = max(gamma_ext + alphas[-1]*float(np.mean(sigma2_qk(center_mc, 1.0, k))), 1e-4)
    for a in alphas[::-1]:
        xhat, s2, gp = _run_one(a, s2, gp)
        xhat_bwd_rev.append(xhat)
    xhat_bwd = list(reversed(xhat_bwd_rev))

    return xhat_fwd, xhat_bwd


def replica_curve(alphas, target_mc, center_mc, v_mc, k, mask_F, beta_ref,
                  sigma0_sq=0.001, pick='max'):
    """
    pick='max': higher target-MSE branch (nontrivial FP, xhat→center) — correct for Stage 2
    pick='min': lower  target-MSE branch (trivial   FP, xhat→target) — for Stage 3
    """
    xhat_fwd, xhat_bwd = replica_sweep(alphas, target_mc, center_mc, v_mc, k, sigma0_sq)
    errs = []
    for xf, xb in zip(xhat_fwd, xhat_bwd):
        mf = float(np.mean((target_mc - xf)**2))
        mb = float(np.mean((target_mc - xb)**2))
        if pick == 'max':
            xhat = xb if mb > mf else xf
        else:
            xhat = xb if mb < mf else xf
        errs.append(float(np.mean((xhat[mask_F] - beta_ref[mask_F])**2)))
    return np.array(errs)


# ──────────────────────────────────────────────────────────────────────────────
# Analytical baselines
# ──────────────────────────────────────────────────────────────────────────────

def ridge_relearn_theory(alpha: float, var_nz: float, rho_ft: float) -> float:
    """Minimum-L2-norm relearning: gen_err_F = max(0,1-α)*var_nz for α<1, else 0."""
    if alpha >= 1.0:
        return 0.0
    return (1.0 - alpha) * var_nz


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────────────

def main():
    D        = 200
    rho_pt   = 0.1
    p_forget = 0.5
    rho_ft   = rho_pt * p_forget
    var_nz   = 1.0 / rho_pt    # = 10
    sigma0_sq_bregman = 1e-5   # very small → approaches constrained problem
    n_trials = 5
    seed_base = 0

    alpha_vals = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.3, 1.5, 2.0, 3.0])

    # MC replica samples
    rng_rep = np.random.default_rng(9999)
    n_mc = 50_000
    sigma_nz = math.sqrt(var_nz)
    n_F_mc = int(round(n_mc * rho_ft))
    n_R_mc = int(round(n_mc * rho_pt * (1 - p_forget)))
    n_0_mc = n_mc - n_F_mc - n_R_mc
    beta_F_mc = rng_rep.normal(0, sigma_nz, n_F_mc)
    beta_R_mc = rng_rep.normal(0, sigma_nz, n_R_mc)
    beta_pt_mc = np.concatenate([beta_F_mc, beta_R_mc, np.zeros(n_0_mc)])
    eff_ul_mc  = np.concatenate([np.zeros(n_F_mc), beta_R_mc, np.zeros(n_0_mc)])
    eff_rl_mc  = beta_pt_mc.copy()
    center_rl_mc = np.concatenate([np.zeros(n_F_mc), beta_R_mc, np.zeros(n_0_mc)])
    mask_F_mc = np.arange(n_mc) < n_F_mc
    v_ul_mc = rng_rep.standard_normal(n_mc)
    v_rl_mc = rng_rep.standard_normal(n_mc)

    c_pt_list = [1e-3, 1.0]
    colors = {1e-3: 'C0', 1.0: 'C3'}

    out_dir = Path(__file__).parent / "verify_bregman_figures"
    out_dir.mkdir(exist_ok=True)

    # ─── Stage 2: Unlearning ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 2: Unlearning (Bregman minimizer from β*_PT toward 0_F)")
    print("="*60)

    stage2_bm = {}    # c_pt → (means, stds) per alpha
    stage2_rep = {}   # c_pt → replica curve

    for c_pt in c_pt_list:
        k = 4.0 * c_pt**2
        print(f"\n-- c_PT={c_pt:.0e}  k={k:.2e} --")

        # Replica curve for Stage 2 — oracle-gp SE (physical FP, α ≤ 1)
        rep = stage2_oracle_curve_v(
            alpha_vals, beta_pt_mc, eff_ul_mc, mask_F_mc, n_mc,
            rho_pt, p_forget, sigma0_sq=0.001, k=k, seed=9999,
        )
        stage2_rep[c_pt] = rep

        # Bregman minimizer experiments
        means_F, stds_F = [], []
        for alpha in alpha_vals:
            errs_F = []
            for trial in range(n_trials):
                beta_star, beta_eff_ul, forget_idx, retain_idx, X_ul = make_problem(
                    D, rho_pt, p_forget, alpha, seed=trial*1000+1,
                    sigma0_sq=sigma0_sq_bregman
                )
                y_ul = X_ul @ beta_eff_ul
                beta_hat_ul = bregman_minimizer(X_ul, y_ul, beta_star, k,
                                                sigma0_sq=sigma0_sq_bregman)

                # Check convergence
                train_res = float(np.mean((X_ul @ beta_hat_ul - y_ul)**2))
                eF = float(np.mean((beta_hat_ul[forget_idx] - beta_star[forget_idx])**2))
                errs_F.append(eF)
                print(f"  α={alpha:.2f}  trial={trial}  err_F={eF:.3f}  train_res={train_res:.2e}")

            means_F.append(float(np.mean(errs_F)))
            stds_F.append(float(np.std(errs_F)))

        stage2_bm[c_pt] = (np.array(means_F), np.array(stds_F))

    # ─── Stage 3: Relearning ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 3: Relearning (Bregman minimizer from β̂_UL≈0 toward β*_PT)")
    print("="*60)

    stage3_bm  = {}
    stage3_rep = {}

    alpha_ul_fixed = 2.0   # use well-unlearned model as Stage-3 start

    for c_pt in c_pt_list:
        k = 4.0 * c_pt**2
        print(f"\n-- c_PT={c_pt:.0e}  k={k:.2e} --")

        # Replica curve for Stage 3
        rep = replica_curve(
            alpha_vals, eff_rl_mc, center_rl_mc, v_rl_mc, k,
            mask_F_mc, beta_pt_mc, sigma0_sq=0.001
        )
        stage3_rep[c_pt] = rep

        # Bregman minimizer experiments
        means_F, stds_F = [], []
        for alpha_rl in alpha_vals:
            errs_F = []
            for trial in range(n_trials):
                beta_star, beta_eff_ul, forget_idx, retain_idx, _ = make_problem(
                    D, rho_pt, p_forget, alpha_ul_fixed, seed=trial*1000+1
                )
                # Stage 2: compute β̂_UL (use large alpha_ul = 2 → β̂_UL ≈ β_eff_UL)
                X_ul = np.random.default_rng(trial*1000+100).standard_normal((
                    max(1, int(alpha_ul_fixed*D)), D)) / math.sqrt(D)
                y_ul  = X_ul @ beta_eff_ul
                beta_ul = bregman_minimizer(X_ul, y_ul, beta_star, k,
                                            sigma0_sq=sigma0_sq_bregman)

                # Stage 3: adversary relearns from β̂_UL
                N_rl = max(1, int(round(alpha_rl * D)))
                rng_rl = np.random.default_rng(trial*1000+200)
                X_rl   = rng_rl.standard_normal((N_rl, D)) / math.sqrt(D)
                y_rl   = X_rl @ beta_star
                beta_rl = bregman_minimizer(X_rl, y_rl, beta_ul, k,
                                            sigma0_sq=sigma0_sq_bregman)

                train_res = float(np.mean((X_rl @ beta_rl - y_rl)**2))
                eF = float(np.mean((beta_rl[forget_idx] - beta_star[forget_idx])**2))
                errs_F.append(eF)
                print(f"  α_RL={alpha_rl:.2f}  trial={trial}  err_F={eF:.3f}  train_res={train_res:.2e}")

            means_F.append(float(np.mean(errs_F)))
            stds_F.append(float(np.std(errs_F)))

        stage3_bm[c_pt] = (np.array(means_F), np.array(stds_F))

    # ─── Figures ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        f"Bregman minimizer vs Replica (D={D}, ρ_PT={rho_pt}, p_f={p_forget})",
        fontsize=11
    )

    # Analytical baselines for Stage 3
    alpha_fine = np.linspace(0.02, 3.0, 200)
    ridge_curve = np.array([ridge_relearn_theory(a, var_nz, rho_ft) for a in alpha_fine])

    for ax, stage, bm_data, rep_data, xlabel, title, add_ridge in [
        (axes[0], "Stage 2 UL", stage2_bm, stage2_rep,
         r"$\alpha_{\rm UL}$", "Forget error after unlearning", False),
        (axes[1], "Stage 3 RL", stage3_bm, stage3_rep,
         r"$\alpha_{\rm RL}$", "Forget error after adversarial relearning", True),
    ]:
        for c_pt in c_pt_list:
            col = colors[c_pt]
            m, s = bm_data[c_pt]
            rep  = rep_data[c_pt]
            lbl  = fr"$c_{{PT}}={c_pt:.0e}$"

            ax.plot(alpha_vals, rep, '-', color=col, lw=1.5,
                    label=f"Replica {lbl}")
            ax.errorbar(alpha_vals, m, yerr=s, fmt='o', color=col, ms=5,
                        capsize=3, label=f"Bregman {lbl}")

        if add_ridge:
            ax.plot(alpha_fine, ridge_curve, 'k--', lw=1.2,
                    label=r"Ridge theory: $(1-\alpha)\cdot v_{\rm nz}$")

        ax.axhline(var_nz, color='gray', ls=':', lw=0.8,
                   label=f"Signal var {var_nz}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\mathcal{E}_F$ (per active forget feature)")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.3, var_nz + 0.5])

    plt.tight_layout()
    p = out_dir / "bregman_vs_replica.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
