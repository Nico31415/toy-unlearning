"""
Plot empirical forgetting metrics vs replica theory predictions (Theorem 2).

Produces one figure per regime (II, III, IV) with 3 panels:
  1. FT generalisation error (p_FT): theory vs empirical MSE on FT task
  2. Total forgetting (F): theory vs empirical ||β̂_FT - β*_PT||² / D
  3. PT-only forgetting (F^(g2)): theory vs empirical on PT-only coordinates

Run from repo root:
  python experiments/diagonal/replica/plot_forgetting_theory_vs_emp.py
"""
import sys, os
sys.path.insert(0, "experiments/diagonal/replica")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path

import ptft_replica_qk as rep

# -------------------------
# Configuration
# -------------------------
RESULTS_DIR = Path("results/forgetting")
OUT_DIR      = RESULTS_DIR
SEEDS        = list(range(5))          # which seeds were run locally
MC           = 50_000
THEORY_ALPHAS = np.linspace(0.01, 0.5, 21)

REGIMES = [
    ("regime_I",    dict(c_pt=1e-3, lambda_pt=+0.99e-3,   gamma_reinit=0.0),  "Regime I (super-lazy)\nc=1e-3, λ=+0.99c, γ=0"),
    ("regime_II",   dict(c_pt=1e-3, lambda_pt=0.0,        gamma_reinit=0.0),  "Regime II (lazy)\nc=1e-3, λ=0, γ=0"),
    ("regime_IV",   dict(c_pt=1e-3, lambda_pt=-0.99e-3,   gamma_reinit=0.0),  "Regime IV (rich)\nc=1e-3, λ=-0.99c, γ=0"),
    ("regime_III",  dict(c_pt=1e-3, lambda_pt=0.0,        gamma_reinit=10.0), "Regime III (lazy, PT-indep)\nc=1e-3, λ=0, γ=10"),
    ("regime_new",  dict(c_pt=1e-3, lambda_pt=-0.999e-3,  gamma_reinit=0.0),  "Rich+ (≈Regime I)\nc=1e-3, λ=-0.999c, γ=0"),
    ("regime_largec", dict(c_pt=2.0,  lambda_pt=0.0,      gamma_reinit=0.0),  "Large c (lazy, PT-indep)\nc=2, λ=0, γ=0"),
]


# -------------------------
# Load empirical data
# -------------------------

def load_empirical(regime_name: str, seeds=SEEDS):
    """
    For each (seed, alpha) folder, load model.pt, beta_pt.pt, beta_ft.pt
    and compute per-group MSE metrics.
    Returns a dict of alpha -> dict of metrics (averaged over seeds).
    """
    regime_dir = RESULTS_DIR / regime_name
    if not regime_dir.exists():
        return {}

    records = []
    for folder in sorted(regime_dir.glob("seed*_alpha*")):
        parts = folder.name.split("_alpha")
        seed = int(parts[0].replace("seed", ""))
        alpha = float(parts[1])
        if seed not in seeds:
            continue

        model_path  = folder / "model.pt"
        bpt_path    = folder / "beta_pt.pt"
        bft_path    = folder / "beta_ft.pt"

        if not (model_path.exists() and bpt_path.exists() and bft_path.exists()):
            continue

        state = torch.load(model_path, map_location="cpu", weights_only=True)
        beta_pt = torch.load(bpt_path, map_location="cpu", weights_only=True).numpy()
        beta_ft = torch.load(bft_path, map_location="cpu", weights_only=True).numpy()

        # Extract β̂_FT from model state_dict
        # Keys vary: 'w_pos'/'v_pos'/'v_neg'/'w_neg' or 'wp'/'vp'/'wm'/'vm' etc.
        keys = list(state.keys())
        if "w_pos" in keys:
            wp = state["w_pos"].numpy(); wm = state["w_neg"].numpy()
            vp = state["v_pos"].numpy(); vm = state["v_neg"].numpy()
            beta_hat = wp * vp - wm * vm
        elif "wp" in keys:
            wp = state["wp"].numpy(); wm = state["wm"].numpy()
            vp = state["vp"].numpy(); vm = state["vm"].numpy()
            beta_hat = wp * vp - wm * vm
        elif "beta" in keys:
            beta_hat = state["beta"].numpy()
        else:
            print(f"  [WARN] Cannot reconstruct beta from {folder.name}: keys={keys[:8]}")
            continue

        D = len(beta_pt)
        rho_pt = 0.1
        rho_ft = 0.1
        omega  = 0.5

        # Identify groups
        pt_active = beta_pt != 0
        ft_active = beta_ft != 0
        g_overlap = pt_active & ft_active   # g=0
        g_new     = (~pt_active) & ft_active # g=1
        g_ptonly  = pt_active & (~ft_active) # g=2

        # Generalisation error (FT task): E[(β̂ - β*_FT)^2]
        mse_total = float(np.mean((beta_hat - beta_ft) ** 2))

        # Forgetting (PT task): E[(β̂ - β*_PT)^2]
        fgt_total  = float(np.mean((beta_hat - beta_pt) ** 2))

        # Per-group forgetting: conditional mean E[(β̂-β*_PT)^2 | g=j]
        # Theory (_forgetting closure) computes conditional mean, so we match that.
        # frac_j = |group j| / D; dividing by it converts the D-normalised sum to group mean.
        frac_ov  = float(g_overlap.mean()) or 1.0
        frac_new = float(g_new.mean())     or 1.0
        frac_pt  = float(g_ptonly.mean())  or 1.0
        fgt_overlap = float(np.mean((beta_hat - beta_pt) ** 2 * g_overlap)) / frac_ov
        fgt_new     = float(np.mean((beta_hat - beta_pt) ** 2 * g_new))     / frac_new
        fgt_ptonly  = float(np.mean((beta_hat - beta_pt) ** 2 * g_ptonly))  / frac_pt

        # Per-group MSE (contribution weighted by pi_j)
        mse_overlap = float(np.mean((beta_hat - beta_ft) ** 2 * g_overlap))
        mse_new     = float(np.mean((beta_hat - beta_ft) ** 2 * g_new))
        mse_ptonly  = float(np.mean((beta_hat - beta_ft) ** 2 * g_ptonly))

        records.append({
            "seed": seed, "alpha": alpha,
            "mse_total": mse_total,
            "fgt_total": fgt_total,
            "fgt_overlap": fgt_overlap,
            "fgt_new": fgt_new,
            "fgt_ptonly": fgt_ptonly,
            "mse_overlap": mse_overlap,
            "mse_new": mse_new,
            "mse_ptonly": mse_ptonly,
        })

    if not records:
        return {}

    df = pd.DataFrame(records)
    grouped = df.groupby("alpha").mean(numeric_only=True).reset_index()
    return grouped


# -------------------------
# Theory (replica predictions) with disk cache
# -------------------------

CACHE_DIR = RESULTS_DIR / "theory_cache"
CACHE_DIR.mkdir(exist_ok=True)

def _cache_key(regime_kw: dict, alphas) -> str:
    import hashlib, json
    # MC intentionally excluded — changing precision doesn't invalidate cached curves
    payload = json.dumps({**regime_kw, "alphas": list(np.round(alphas, 6))}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()

def compute_theory(regime_kw: dict, alphas=THEORY_ALPHAS):
    cache_file = CACHE_DIR / (_cache_key(regime_kw, alphas) + ".npz")
    if cache_file.exists():
        print(f"  (loading from cache: {cache_file.name})")
        d = np.load(cache_file, allow_pickle=True)
        return {k: d[k] for k in d.files}

    out = rep.ptft_forgetting_curve(
        rho_pt=0.1, rho_ft=0.1, omega=0.5,
        alphas=alphas, mc=MC, seed=0,
        a_pt=1.0,
        **regime_kw
    )
    # Save only the plain array keys (skip nested dicts like _curve, _reliability)
    saveable = {k: v for k, v in out.items() if isinstance(v, np.ndarray) or np.isscalar(v)}
    np.savez(cache_file, **saveable)
    return out


# -------------------------
# Plotting
# -------------------------

fig, axes = plt.subplots(5, 6, figsize=(26, 17))
fig.suptitle("Forgetting replica theory vs empirical  (ρ_PT=ρ_FT=0.1, ω=0.5)", fontsize=13)

PANEL_TITLES  = [
    "FT generalisation error  p_FT",
    "Total forgetting  F",
    "PT-only forgetting  F^(ptonly)  [g=2]",
    "Overlap forgetting  F^(overlap)  [g=0]",
    "New-FT forgetting  F^(new)  [g=1]",
]
COLORS        = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

for col, (regime_name, kw, label) in enumerate(REGIMES):
    print(f"Computing theory for {regime_name}...")
    theory = compute_theory(kw)

    print(f"Loading empirical data for {regime_name}...")
    emp = load_empirical(regime_name)

    for row, (metric_t, metric_e, title) in enumerate([
        ("mse",                  "mse_total",   PANEL_TITLES[0]),
        ("forgetting",           "fgt_total",   PANEL_TITLES[1]),
        ("forgetting_g2_ptonly", "fgt_ptonly",  PANEL_TITLES[2]),
        ("forgetting_g0_overlap","fgt_overlap", PANEL_TITLES[3]),
        ("forgetting_g1_new",    "fgt_new",     PANEL_TITLES[4]),
    ]):
        ax = axes[row, col]

        # Theory line
        ax.plot(theory["alpha"], theory[metric_t],
                color=COLORS[col], lw=2, label="Theory")

        # Empirical scatter (mean over seeds)
        if not isinstance(emp, dict) and len(emp) > 0 and metric_e in emp.columns:
            ax.scatter(emp["alpha"], emp[metric_e],
                       color=COLORS[col], s=30, zorder=5, label="Empirical (mean)")
            # Error bars from forgetting_results.csv if available
        else:
            ax.text(0.5, 0.5, "No empirical data", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=9)

        if row == 0:
            ax.set_title(label, fontsize=10)
        if col == 0:
            ax.set_ylabel(title, fontsize=9)
        ax.set_xlabel("α_FT", fontsize=9)
        ax.set_xlim(0, 0.52)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "forgetting_theory_vs_emp.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.close()
