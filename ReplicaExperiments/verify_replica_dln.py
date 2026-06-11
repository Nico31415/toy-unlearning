#!/usr/bin/env python3
"""
Verify replica curves against actual diagonal linear network (DLN) training.

Pipeline:
    Stage 1 (PT):  Oracle — β̂_PT = β*_PT  (skip training, assume α_PT → ∞)
    Stage 2 (UL):  DLN GF from β̂_PT, data X_UL, targets y = β_eff_UL · x
    Stage 3 (RL):  DLN GF from β̂_UL, data X_RL, targets y = β*_PT · x

The DLN implicit bias (Gradient Flow, balanced init) gives:
    β̂ = argmin D_{q_k}(β, β₀)   s.t.  X^T β = X^T β_eff   [interpolating solution]

Comparison: replica curves (from compare_cpt_replica.py logic) vs DLN gen errors.

Dimensions:
    D    = 500   (feature dimension)
    α_UL = N_UL/D sweep
    α_RL = N_RL/D sweep

c_PT values compared: 1e-3 (dangerous/L1) and 1.0 (safe/L2)
"""

from __future__ import annotations
import math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from fixed_lambda_all import prox_qk_safeguarded, sigma2_qk


# ──────────────────────────────────────────────────────────────────────────────
# Diagonal linear network
# ──────────────────────────────────────────────────────────────────────────────

class DLN:
    """Diagonal linear network β = w⁺∘v⁺ - w⁻∘v⁻, gradient-flow trained."""

    def __init__(self, D: int, c_pt: float, beta_init: np.ndarray):
        """
        Initialise at the balanced point around beta_init:
            w⁺_i = v⁺_i = sqrt(c_pt² + |β̂_PT_i|/2 + β̂_PT_i/2)  (active pathway)
            w⁻_i = v⁻_i = sqrt(c_pt² + |β̂_PT_i|/2 - β̂_PT_i/2)  (negative pathway)

        This ensures β(0) = β̂_PT and all four parameters are real and equal in
        magnitude per pathway (balanced gauge).

        For oracle PT: β_init = β*_PT.
        """
        b = beta_init.astype(np.float64)
        half_abs = 0.5 * np.abs(b)
        # w⁺·v⁺ - w⁻·v⁻ = b, balanced: w⁺=v⁺=sqrt(c²+half_abs+b/2), w⁻=v⁻=sqrt(c²+half_abs-b/2)
        self.wp = np.sqrt(c_pt**2 + half_abs + 0.5 * b)
        self.vp = self.wp.copy()
        self.wm = np.sqrt(c_pt**2 + half_abs - 0.5 * b)
        self.vm = self.wm.copy()

    def beta(self) -> np.ndarray:
        return self.wp * self.vp - self.wm * self.vm

    def grad_step(self, X: np.ndarray, y: np.ndarray, lr: float):
        """Full-batch GD step on MSE = (1/N)||y - X β||²."""
        N = X.shape[0]
        beta = self.beta()
        resid = X @ beta - y                    # (N,)
        dL_dbeta = (2.0 / N) * (X.T @ resid)   # (D,)

        # Chain rule: β_i = wp_i·vp_i - wm_i·vm_i
        self.wp -= lr * dL_dbeta * self.vp
        self.vp -= lr * dL_dbeta * self.wp.copy()  # use original wp
        self.wm -= lr * (-dL_dbeta) * self.vm
        self.vm -= lr * (-dL_dbeta) * self.wm.copy()

    def grad_step_correct(self, X: np.ndarray, y: np.ndarray, lr: float):
        """Full-batch GD step — correct simultaneous update."""
        N = X.shape[0]
        beta = self.beta()
        resid = X @ beta - y
        g = (2.0 / N) * (X.T @ resid)           # gradient w.r.t. β

        wp0, vp0, wm0, vm0 = self.wp.copy(), self.vp.copy(), self.wm.copy(), self.vm.copy()
        self.wp = wp0 - lr * g * vp0
        self.vp = vp0 - lr * g * wp0
        self.wm = wm0 - lr * (-g) * vm0
        self.vm = vm0 - lr * (-g) * wm0


def train_dln(
    D: int,
    c_pt: float,
    beta_init: np.ndarray,
    X: np.ndarray,          # (N, D) training data
    y_target: np.ndarray,   # (N,)   training targets = X @ beta_eff
    lr: float = 0.05,
    max_steps: int = 80_000,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Train a DLN via full-batch GD from beta_init on (X, y_target).
    Returns β at convergence.
    """
    net = DLN(D, c_pt, beta_init)
    N = X.shape[0]
    prev_loss = np.inf

    for step in range(max_steps):
        beta = net.beta()
        pred = X @ beta
        loss = float(np.mean((pred - y_target) ** 2))

        if loss < tol:
            break
        if step > 0 and abs(loss - prev_loss) < 1e-15 and loss > tol:
            # Stuck — halve lr
            lr *= 0.5
            if lr < 1e-12:
                break

        prev_loss = loss
        net.grad_step_correct(X, y_target, lr)

    return net.beta()


# ──────────────────────────────────────────────────────────────────────────────
# Replica fixed-point (minimal re-implementation, same as compare_cpt_replica)
# ──────────────────────────────────────────────────────────────────────────────

def bregman_prox(z: np.ndarray, lam: float, k: float, beta0: np.ndarray) -> np.ndarray:
    shift = lam * 0.5 * np.arcsinh(2.0 * beta0 / math.sqrt(max(k, 1e-30)))
    return prox_qk_safeguarded(z + shift, lam, k)


def replica_gen_err(
    alpha: float,
    target_mc: np.ndarray,
    center_mc: np.ndarray,
    v_mc: np.ndarray,
    k: float,
    mask: np.ndarray,
    beta_ref: np.ndarray,
    sigma0_sq: float = 0.01,
    gamma_ext: float = 1e-9,
    max_iters: int = 800,
    tol: float = 1e-10,
    damp: float = 0.25,
):
    """Single-alpha forward-backward replica curve point (returns gen_err on mask)."""
    def _solve(init_s2, init_gp):
        s2, gp = init_s2, init_gp
        for _ in range(max_iters):
            z = target_mc + math.sqrt(max(s2, 1e-15)) * v_mc
            xhat = bregman_prox(z, gp, k, center_mc)
            mse  = float(np.mean((target_mc - xhat) ** 2))
            ms2  = float(np.mean(sigma2_qk(xhat, gp, k)))
            s2n  = float(sigma0_sq + alpha * mse)
            gpn  = float(gamma_ext + alpha * ms2)
            if max(abs(s2n - s2), abs(gpn - gp)) < tol:
                return xhat, mse, s2n, gpn
            s2 = (1 - damp) * s2 + damp * s2n
            gp = (1 - damp) * gp + damp * gpn
            s2 = max(s2, sigma0_sq, 1e-15)
            gp = max(gp, gamma_ext, 1e-14)
        z = target_mc + math.sqrt(max(s2, 1e-15)) * v_mc
        xhat = bregman_prox(z, gp, k, center_mc)
        return xhat, float(np.mean((target_mc - xhat) ** 2)), s2, gp

    prior_mse = float(np.mean((target_mc - center_mc) ** 2))
    s2_0 = max(sigma0_sq + alpha * prior_mse, 1e-6)
    gp_0 = max(gamma_ext + alpha * float(np.mean(sigma2_qk(center_mc, 1.0, k))), 1e-14)

    xf, mf, s2f, gpf = _solve(s2_0, gp_0)
    xb, mb, s2b, gpb = _solve(s2_0, gp_0)  # same init (just one solution needed)

    # forward-backward: pick lower target-mse (physical)
    xhat = xf if mf <= mb else xb
    if mask.any():
        return float(np.mean((xhat[mask] - beta_ref[mask]) ** 2))
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Experiment
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(
    D: int = 500,
    rho_pt: float = 0.1,
    p_forget: float = 0.5,
    c_pt: float = 1.0,
    alpha_ul: float = 0.5,
    alpha_rl: float = 1.0,
    seed: int = 42,
    lr_ul: float = 0.02,
    lr_rl: float = 0.02,
    max_steps: int = 60_000,
    tol: float = 5e-8,
):
    """
    Run one instance of the 3-stage pipeline.
    Returns (gen_err_F_ul, gen_err_R_ul, gen_err_F_rl).
    """
    rng = np.random.default_rng(seed)

    # ── Teacher β*_PT ─────────────────────────────────────────────────────────
    n_active = int(round(D * rho_pt))
    n_F      = int(round(D * rho_pt * p_forget))
    n_R      = n_active - n_F
    sigma_nz = math.sqrt(1.0 / rho_pt)

    beta_star = np.zeros(D)
    active_idx = rng.choice(D, n_active, replace=False)
    beta_star[active_idx] = rng.normal(0, sigma_nz, n_active)

    forget_idx  = active_idx[:n_F]
    retain_idx  = active_idx[n_F:]

    # Unlearning target: zero forget, keep retain, zero inactive
    beta_eff_ul = np.zeros(D)
    beta_eff_ul[retain_idx] = beta_star[retain_idx]

    # ── Oracle pretraining: β̂_PT = β*_PT ────────────────────────────────────
    beta_pt = beta_star.copy()

    # ── Stage 2: Unlearning ───────────────────────────────────────────────────
    N_ul = max(1, int(round(alpha_ul * D)))
    X_ul = rng.standard_normal((N_ul, D)) / math.sqrt(D)
    y_ul = X_ul @ beta_eff_ul                # targets = X · β_eff_UL

    beta_ul = train_dln(D, c_pt, beta_pt, X_ul, y_ul, lr=lr_ul,
                        max_steps=max_steps, tol=tol)

    # Measure residual training loss
    train_loss_ul = float(np.mean((X_ul @ beta_ul - y_ul) ** 2))

    # Gen errors
    gen_err_F_ul = float(np.mean((beta_ul[forget_idx] - beta_star[forget_idx]) ** 2))
    gen_err_R_ul = float(np.mean((beta_ul[retain_idx] - beta_star[retain_idx]) ** 2))

    # ── Stage 3: Adversarial relearning ───────────────────────────────────────
    N_rl = max(1, int(round(alpha_rl * D)))
    X_rl = rng.standard_normal((N_rl, D)) / math.sqrt(D)
    y_rl = X_rl @ beta_star                  # adversary targets β*_PT on forget

    beta_rl = train_dln(D, c_pt, beta_ul, X_rl, y_rl, lr=lr_rl,
                        max_steps=max_steps, tol=tol)

    train_loss_rl = float(np.mean((X_rl @ beta_rl - y_rl) ** 2))
    gen_err_F_rl = float(np.mean((beta_rl[forget_idx] - beta_star[forget_idx]) ** 2))

    return gen_err_F_ul, gen_err_R_ul, gen_err_F_rl, train_loss_ul, train_loss_rl


def run_sweep(
    D: int = 500,
    rho_pt: float = 0.1,
    p_forget: float = 0.5,
    c_pt: float = 1.0,
    alpha_values: np.ndarray = None,
    n_trials: int = 3,
    mode: str = "ul",     # "ul" or "rl"
    alpha_ul_fixed: float = 2.0,  # for rl mode: fixed α_UL
    **kw
):
    """
    Sweep α (for Stage 2 or Stage 3) over multiple trials; return mean ± std of gen_err_F.
    """
    if alpha_values is None:
        alpha_values = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])

    means_F, stds_F = [], []

    for alpha in alpha_values:
        errs_F = []
        for trial in range(n_trials):
            if mode == "ul":
                eF, eR, _, tl_ul, _ = run_experiment(
                    D=D, rho_pt=rho_pt, p_forget=p_forget, c_pt=c_pt,
                    alpha_ul=alpha, alpha_rl=alpha_ul_fixed, seed=trial*1000+1, **kw
                )
                errs_F.append(eF)
                print(f"    c_PT={c_pt}  α_UL={alpha:.2f}  trial={trial}"
                      f"  err_F={eF:.4f}  train_loss={tl_ul:.2e}")
            else:
                eF, eR, eRL, tl_ul, tl_rl = run_experiment(
                    D=D, rho_pt=rho_pt, p_forget=p_forget, c_pt=c_pt,
                    alpha_ul=alpha_ul_fixed, alpha_rl=alpha, seed=trial*1000+1, **kw
                )
                errs_F.append(eRL)
                print(f"    c_PT={c_pt}  α_RL={alpha:.2f}  trial={trial}"
                      f"  err_RL_F={eRL:.4f}  train_loss_rl={tl_rl:.2e}")

        means_F.append(float(np.mean(errs_F)))
        stds_F.append(float(np.std(errs_F)))

    return np.array(means_F), np.array(stds_F)


# ──────────────────────────────────────────────────────────────────────────────
# Replica reference curves
# ──────────────────────────────────────────────────────────────────────────────

def replica_sweep(
    alphas: np.ndarray,
    c_pt: float,
    mode: str,          # "ul" or "rl"
    rho_pt: float = 0.1,
    p_forget: float = 0.5,
    n_mc: int = 60_000,
    seed: int = 2024,
    sigma0_sq: float = 0.001,  # small for noiseless DLN comparison
):
    rng = np.random.default_rng(seed)
    k   = 4.0 * c_pt ** 2
    sigma_nz = math.sqrt(1.0 / rho_pt)

    n_F = int(round(n_mc * rho_pt * p_forget))
    n_R = int(round(n_mc * rho_pt * (1 - p_forget)))
    n_0 = n_mc - n_F - n_R

    beta_F = rng.normal(0, sigma_nz, n_F)
    beta_R = rng.normal(0, sigma_nz, n_R)
    beta_pt = np.concatenate([beta_F, beta_R, np.zeros(n_0)])
    mask_F = np.arange(n_mc) < n_F

    if mode == "ul":
        eff_ul = np.concatenate([np.zeros(n_F), beta_R, np.zeros(n_0)])
        v = rng.standard_normal(n_mc)
        center = beta_pt
        target = eff_ul
        beta_ref = beta_pt
    else:
        center = np.concatenate([np.zeros(n_F), beta_R, np.zeros(n_0)])
        target = beta_pt
        v = rng.standard_normal(n_mc)
        beta_ref = beta_pt

    def solve_one(alpha, init_s2=None, init_gp=None):
        gamma_ext = 1e-9
        damp      = 0.25
        if init_s2 is None:
            prior_mse = float(np.mean((target - center) ** 2))
            s2 = max(sigma0_sq + alpha * prior_mse, 1e-6)
            gp = max(gamma_ext + alpha * float(np.mean(sigma2_qk(center, 1.0, k))), 1e-14)
        else:
            s2, gp = max(init_s2, sigma0_sq, 1e-15), max(init_gp, gamma_ext, 1e-14)

        for _ in range(800):
            z    = target + math.sqrt(max(s2, 1e-15)) * v
            xhat = bregman_prox(z, gp, k, center)
            mse  = float(np.mean((target - xhat) ** 2))
            ms2  = float(np.mean(sigma2_qk(xhat, gp, k)))
            s2n  = float(sigma0_sq + alpha * mse)
            gpn  = float(gamma_ext + alpha * ms2)
            if max(abs(s2n - s2), abs(gpn - gp)) < 1e-10:
                break
            s2 = (1 - damp)*s2 + damp*s2n;  s2 = max(s2, sigma0_sq, 1e-15)
            gp = (1 - damp)*gp + damp*gpn;  gp = max(gp, gamma_ext, 1e-14)

        z    = target + math.sqrt(max(s2, 1e-15)) * v
        xhat = bregman_prox(z, gp, k, center)
        ge   = float(np.mean((xhat[mask_F] - beta_ref[mask_F]) ** 2))
        return ge, s2, gp

    # forward-backward sweeps
    errs_f, errs_b = [], []
    state = None
    for a in alphas:
        ge, s2, gp = solve_one(a, *(state if state else (None, None)))
        state = (s2, gp)
        errs_f.append(ge)

    state = None
    for a in alphas[::-1]:
        ge, s2, gp = solve_one(a, *(state if state else (None, None)))
        state = (s2, gp)
        errs_b.append(ge)
    errs_b = errs_b[::-1]

    # Minimum (physical FP)
    return np.minimum(errs_f, errs_b)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    D        = 500
    rho_pt   = 0.1
    p_forget = 0.5
    n_trials = 4

    alpha_ul_vals = np.array([0.1, 0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 3.0])
    alpha_rl_vals = np.array([0.1, 0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 3.0])
    alpha_fine    = np.concatenate([np.linspace(0.05, 0.95, 25),
                                    np.linspace(1.05, 3.0, 15)])

    c_pt_list = [1e-3, 1.0]

    out_dir = Path(__file__).parent / "verify_dln_figures"
    out_dir.mkdir(exist_ok=True)

    # ─── Stage 2 sweep ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 2: Unlearning sweep (α_UL)")
    print("="*60)

    stage2_dln = {}
    stage2_rep = {}

    for c_pt in c_pt_list:
        print(f"\n--- c_PT={c_pt} ---")
        m, s = run_sweep(D=D, rho_pt=rho_pt, p_forget=p_forget, c_pt=c_pt,
                         alpha_values=alpha_ul_vals, n_trials=n_trials,
                         mode="ul", alpha_ul_fixed=2.0)
        stage2_dln[c_pt] = (m, s)

        print(f"  Computing replica curve...")
        rep = replica_sweep(alpha_fine, c_pt, "ul", rho_pt, p_forget)
        stage2_rep[c_pt] = rep

    # ─── Stage 3 sweep ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 3: Relearning sweep (α_RL), fixed α_UL=2.0")
    print("="*60)

    stage3_dln = {}
    stage3_rep = {}

    for c_pt in c_pt_list:
        print(f"\n--- c_PT={c_pt} ---")
        m, s = run_sweep(D=D, rho_pt=rho_pt, p_forget=p_forget, c_pt=c_pt,
                         alpha_values=alpha_rl_vals, n_trials=n_trials,
                         mode="rl", alpha_ul_fixed=2.0)
        stage3_dln[c_pt] = (m, s)

        print(f"  Computing replica curve...")
        rep = replica_sweep(alpha_fine, c_pt, "rl", rho_pt, p_forget)
        stage3_rep[c_pt] = rep

    # ─── Figures ─────────────────────────────────────────────────────────────
    colors = {1e-3: 'C0', 1.0: 'C3'}
    var_nz = 1.0 / rho_pt   # = 10

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"DLN verification  (D={D}, ρ_PT={rho_pt}, p_f={p_forget}, {n_trials} trials)",
        fontsize=11
    )

    for ax, stage, dln_data, rep_data, alphas_dln, xlabel, title in [
        (axes[0], "Stage 2 Unlearning", stage2_dln, stage2_rep,
         alpha_ul_vals, r"$\alpha_{\rm UL} = N_{\rm UL}/D$", "Forget gen error after unlearning"),
        (axes[1], "Stage 3 Relearning", stage3_dln, stage3_rep,
         alpha_rl_vals, r"$\alpha_{\rm RL} = N_{\rm RL}/D$", "Forget gen error after relearning"),
    ]:
        for c_pt in c_pt_list:
            col = colors[c_pt]
            m, s = dln_data[c_pt]
            rep  = rep_data[c_pt]

            # Replica curve
            ax.plot(alpha_fine, rep, '-', color=col, lw=1.5,
                    label=fr"Replica $c_{{PT}}={c_pt:.0e}$")
            # DLN points
            ax.errorbar(alphas_dln, m, yerr=s, fmt='o', color=col, ms=5,
                        capsize=3, label=fr"DLN $c_{{PT}}={c_pt:.0e}$")

        ax.axhline(var_nz, color='k', ls='--', lw=0.8,
                   label=f"Signal variance {var_nz}")
        ax.axhline(0, color='gray', ls=':', lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\mathcal{E}_F$ (per active forget feature)")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.5, var_nz + 1])

    plt.tight_layout()
    p = out_dir / "replica_vs_dln.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
