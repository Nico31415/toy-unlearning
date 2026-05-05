"""
Sanity-check plot for the forgetting framework.

Three FT teacher conventions, all with ω=0.5, ρ_PT=ρ_FT=0.1, a_PT=1:

  "aligned_overlap"  β*_FT[g=0] = +a_pt = +1
  "zero_overlap"     β*_FT[g=0] =  0
  "opposite_overlap" β*_FT[g=0] = -a_pt = -1

Analytic predictions at α → ∞:
  F^overlap:  aligned→0,  zero→1,  opposite→4
  F^ptonly:   all three → 1

Run from repo root:
  python experiments/diagonal/replica/plot_sanity_check.py
"""
import sys
sys.path.insert(0, "experiments/diagonal/replica")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import hashlib, json
import torch

import ptft_replica_qk as rep

# -------------------------
# Config
# -------------------------
THEORY_DIR   = Path("results/forgetting")
EMP_DIR      = Path("results/sanity_check")
OUT_DIR      = THEORY_DIR
MC           = 50_000
ALPHAS       = np.linspace(0.01, 0.8, 21)

CACHE_DIR = THEORY_DIR / "theory_cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

RHO   = 0.1
OMEGA = 0.5
A_PT  = 1.0
SEEDS = list(range(5))

CONVENTIONS = [
    ("aligned_overlap",  "#1f77b4", "-",  "aligned  β*_FT[g0]=+1"),
    ("zero_overlap",     "#ff7f0e", "--", "zero     β*_FT[g0]=0"),
    ("opposite_overlap", "#d62728", ":",  "opposite β*_FT[g0]=−1"),
]

ASYMPTOTES = {
    "forgetting_g0_overlap": {"aligned_overlap": 0.0, "zero_overlap": 1.0, "opposite_overlap": 4.0},
    "forgetting_g2_ptonly":  {"aligned_overlap": 1.0, "zero_overlap": 1.0, "opposite_overlap": 1.0},
}

# -------------------------
# Theory cache
# -------------------------

def _cache_key(kw: dict) -> str:
    return hashlib.md5(json.dumps(kw, sort_keys=True).encode()).hexdigest()


def compute_theory(teacher_norm: str, omega: float = OMEGA,
                   c_pt: float = 1e-3, lambda_pt: float = 0.0, gamma_reinit: float = 0.0):
    key_dict = dict(
        rho_pt=RHO, rho_ft=RHO, omega=omega, a_pt=A_PT,
        c_pt=c_pt, lambda_pt=lambda_pt, gamma_reinit=gamma_reinit,
        ft_teacher_norm=teacher_norm,
        alphas=list(np.round(ALPHAS, 6)),
    )
    cache_file = CACHE_DIR / (_cache_key(key_dict) + ".npz")
    if cache_file.exists():
        print(f"  (cache hit: {teacher_norm}, ω={omega})")
        d = np.load(cache_file, allow_pickle=True)
        return {k: d[k] for k in d.files}
    print(f"  Computing (teacher={teacher_norm}, ω={omega}, c_pt={c_pt})...", flush=True)
    out = rep.ptft_forgetting_curve(
        rho_pt=RHO, rho_ft=RHO, omega=omega,
        alphas=ALPHAS, mc=MC, seed=0,
        a_pt=A_PT, c_pt=c_pt, lambda_pt=lambda_pt, gamma_reinit=gamma_reinit,
        ft_teacher_norm=teacher_norm,
    )
    saveable = {k: v for k, v in out.items() if isinstance(v, np.ndarray) or np.isscalar(v)}
    np.savez(cache_file, **saveable)
    return out


# -------------------------
# Empirical loader
# -------------------------

def load_empirical(regime_name: str, teacher_norm: str, seeds=SEEDS):
    """Load saved model weights and compute per-group forgetting metrics."""
    base = EMP_DIR / regime_name / teacher_norm
    if not base.exists():
        return None

    records = []
    for folder in sorted(base.glob("seed*_alpha*")):
        parts = folder.name.split("_alpha")
        seed  = int(parts[0].replace("seed", ""))
        alpha = float(parts[1])
        if seed not in seeds:
            continue

        model_path = folder / "model.pt"
        bpt_path   = folder / "beta_pt.pt"
        bft_path   = folder / "beta_ft.pt"
        if not (model_path.exists() and bpt_path.exists() and bft_path.exists()):
            continue

        state   = torch.load(model_path, map_location="cpu", weights_only=True)
        beta_pt = torch.load(bpt_path,   map_location="cpu", weights_only=True).numpy()
        beta_ft = torch.load(bft_path,   map_location="cpu", weights_only=True).numpy()

        keys = list(state.keys())
        if "w_pos" in keys:
            beta_hat = state["w_pos"].numpy() * state["v_pos"].numpy() \
                     - state["w_neg"].numpy() * state["v_neg"].numpy()
        elif "wp" in keys:
            beta_hat = state["wp"].numpy() * state["vp"].numpy() \
                     - state["wm"].numpy() * state["vm"].numpy()
        elif "beta" in keys:
            beta_hat = state["beta"].numpy()
        else:
            continue

        # Use saved support masks when available — needed for zero_overlap where
        # beta_ft[g=0]=0 makes ft_active unreliable for group identification.
        spt_path = folder / "support_pt.pt"
        sft_path = folder / "support_ft.pt"
        if spt_path.exists() and sft_path.exists():
            pt_active = torch.load(spt_path, map_location="cpu", weights_only=True).numpy().astype(bool)
            ft_active = torch.load(sft_path, map_location="cpu", weights_only=True).numpy().astype(bool)
        else:
            pt_active = beta_pt != 0
            ft_active = beta_ft != 0
        g_overlap = pt_active & ft_active
        g_new     = (~pt_active) & ft_active
        g_ptonly  = pt_active & (~ft_active)

        mse_total = float(np.mean((beta_hat - beta_ft) ** 2))
        fgt_total = float(np.mean((beta_hat - beta_pt) ** 2))

        frac_ov  = float(g_overlap.mean()) or 1.0
        frac_new = float(g_new.mean())     or 1.0
        frac_pt  = float(g_ptonly.mean())  or 1.0

        fgt_overlap = float(np.mean((beta_hat - beta_pt) ** 2 * g_overlap)) / frac_ov
        fgt_new     = float(np.mean((beta_hat - beta_pt) ** 2 * g_new))     / frac_new
        fgt_ptonly  = float(np.mean((beta_hat - beta_pt) ** 2 * g_ptonly))  / frac_pt

        # Magnitude forgetting: (|β̂| - |β*_PT|)² — sign-agnostic, zero if magnitude preserved
        fgt_mag_overlap = float(np.mean((np.abs(beta_hat) - np.abs(beta_pt)) ** 2 * g_overlap)) / frac_ov
        fgt_mag_ptonly  = float(np.mean((np.abs(beta_hat) - np.abs(beta_pt)) ** 2 * g_ptonly))  / frac_pt

        records.append(dict(seed=seed, alpha=alpha,
                            mse_total=mse_total, fgt_total=fgt_total,
                            fgt_overlap=fgt_overlap, fgt_new=fgt_new, fgt_ptonly=fgt_ptonly,
                            fgt_mag_overlap=fgt_mag_overlap, fgt_mag_ptonly=fgt_mag_ptonly))

    if not records:
        return None
    df = pd.DataFrame(records).groupby("alpha").mean(numeric_only=True).reset_index()
    return df


EMP_METRIC_MAP = {
    "mse":                   "mse_total",
    "forgetting":            "fgt_total",
    "forgetting_g2_ptonly":  "fgt_ptonly",
    "forgetting_g0_overlap": "fgt_overlap",
    None:                    None,   # placeholder; handled per-row below
}
EMP_MAG_METRICS = {
    "F_mag^overlap [g=0]  (|β̂|−|β*|)²  → 0 / 1 / 0": "fgt_mag_overlap",
    "F_mag^ptonly  [g=2]  (|β̂|−|β*|)²  → 1 for all":  "fgt_mag_ptonly",
}


# =========================================================
# Figure 1: three teacher conventions (ω=0.5, Regime II)
# =========================================================
REGIME_CONFIGS = [
    ("regime_II",  dict(gamma_reinit=0.0,  c_pt=1e-3, lambda_pt=0.0),      "Regime II  (lazy, c=1e-3, λ=0, γ=0)"),
    ("regime_III", dict(gamma_reinit=10.0, c_pt=1e-3, lambda_pt=0.0),      "Regime III (PT-indep, c=1e-3, λ=0, γ=10)"),
    ("regime_IV",  dict(gamma_reinit=0.0,  c_pt=1e-3, lambda_pt=-0.95e-3), "Regime IV  (rich, c=1e-3, λ=−0.95c, γ=0)"),
]

print("=== Figure 1: three teacher conventions (Regime II) ===")

ROWS_1 = [
    ("mse",                   "p_FT   (gen. error on FT task)"),
    ("forgetting",            "F      (total forgetting)"),
    ("forgetting_g2_ptonly",  "F^ptonly  [g=2]   → 1 for all"),
    ("forgetting_g0_overlap", "F^overlap [g=0]   → 0 / 1 / 4  (signed)"),
    (None,                    "F_mag^overlap [g=0]  (|β̂|−|β*|)²  → 0 / 1 / 0"),
    (None,                    "F_mag^ptonly  [g=2]  (|β̂|−|β*|)²  → 1 for all"),
]

theories = {}
emp_data  = {}
for norm, color, ls, label in CONVENTIONS:
    print(f"\n[{norm}]")
    theories[norm] = compute_theory(norm)
    emp_data[norm] = load_empirical("regime_II", norm)

fig1, axes1 = plt.subplots(len(ROWS_1), 1, figsize=(8, 4 * len(ROWS_1)))
fig1.suptitle(
    "Sanity check: three FT teacher conventions  —  Regime II (lazy, c=1e-3, λ=0)\n"
    "ρ_PT=ρ_FT=0.1, ω=0.5, a_PT=1   |   lines = theory,  dots = empirical (mean over seeds)\n"
    "Asymptotes at α→∞:  F^ptonly→1 (all three),  F^overlap→ 0 / 1 / 4",
    fontsize=10,
)

MAG_ASYMPTOTES_OVERLAP = {"aligned_overlap": 0.0, "zero_overlap": 1.0, "opposite_overlap": 0.0}
MAG_ASYMPTOTES_PTONLY  = {"aligned_overlap": 1.0, "zero_overlap": 1.0, "opposite_overlap": 1.0}

for row_i, (metric, row_title) in enumerate(ROWS_1):
    ax = axes1[row_i]
    emp_col_mag = EMP_MAG_METRICS.get(row_title)

    for norm, color, ls, label in CONVENTIONS:
        if metric is not None:
            # Theory curve exists
            t = theories[norm]
            ax.plot(t["alpha"], t[metric], color=color, ls=ls, lw=2.5, label=f"Theory {label}")
            emp = emp_data[norm]
            emp_col = EMP_METRIC_MAP.get(metric)
            if emp is not None and emp_col and emp_col in emp.columns:
                ax.scatter(emp["alpha"], emp[emp_col], color=color, s=40, zorder=5,
                           marker="o", label=f"Emp. {label}")
        else:
            # Magnitude rows: empirical only (no replica theory yet)
            emp = emp_data[norm]
            if emp is not None and emp_col_mag and emp_col_mag in emp.columns:
                ax.scatter(emp["alpha"], emp[emp_col_mag], color=color, s=40, zorder=5,
                           marker="s", label=f"Emp. {label}")

    if metric in ASYMPTOTES:
        for norm, color, ls, _ in CONVENTIONS:
            ax.axhline(ASYMPTOTES[metric][norm], color=color, lw=0.8, ls=":", alpha=0.4)
    elif emp_col_mag == "fgt_mag_overlap":
        for norm, color, ls, _ in CONVENTIONS:
            ax.axhline(MAG_ASYMPTOTES_OVERLAP[norm], color=color, lw=0.8, ls=":", alpha=0.4)
    elif emp_col_mag == "fgt_mag_ptonly":
        for norm, color, ls, _ in CONVENTIONS:
            ax.axhline(MAG_ASYMPTOTES_PTONLY[norm], color=color, lw=0.8, ls=":", alpha=0.4)

    ax.set_ylabel(row_title, fontsize=9)
    ax.set_xlabel("α_FT", fontsize=9)
    ax.set_xlim(0, ALPHAS[-1] * 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out1 = OUT_DIR / "sanity_check_teacher_conventions.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out1}")
plt.close()


# =========================================================
# Figure 2: same plot for Regime III
# =========================================================
print("\n=== Figure 2: three teacher conventions (Regime III) ===")

theories_III = {}
emp_data_III = {}
for norm, color, ls, label in CONVENTIONS:
    print(f"\n[{norm}]")
    theories_III[norm] = compute_theory(norm, gamma_reinit=10.0)
    emp_data_III[norm] = load_empirical("regime_III", norm)

fig2, axes2 = plt.subplots(len(ROWS_1), 1, figsize=(8, 4 * len(ROWS_1)))
fig2.suptitle(
    "Sanity check: three FT teacher conventions  —  Regime III (PT-indep, c=1e-3, γ=10)\n"
    "ρ_PT=ρ_FT=0.1, ω=0.5, a_PT=1   |   lines = theory,  dots = empirical (mean over seeds)\n"
    "Asymptotes at α→∞:  F^ptonly→1 (all three),  F^overlap→ 0 / 1 / 4",
    fontsize=10,
)

for row_i, (metric, row_title) in enumerate(ROWS_1):
    ax = axes2[row_i]
    emp_col_mag = EMP_MAG_METRICS.get(row_title)

    for norm, color, ls, label in CONVENTIONS:
        if metric is not None:
            t = theories_III[norm]
            ax.plot(t["alpha"], t[metric], color=color, ls=ls, lw=2.5, label=f"Theory {label}")
            emp = emp_data_III[norm]
            emp_col = EMP_METRIC_MAP.get(metric)
            if emp is not None and emp_col and emp_col in emp.columns:
                ax.scatter(emp["alpha"], emp[emp_col], color=color, s=40, zorder=5,
                           marker="o", label=f"Emp. {label}")
        else:
            emp = emp_data_III[norm]
            if emp is not None and emp_col_mag and emp_col_mag in emp.columns:
                ax.scatter(emp["alpha"], emp[emp_col_mag], color=color, s=40, zorder=5,
                           marker="s", label=f"Emp. {label}")

    if metric in ASYMPTOTES:
        for norm, color, ls, _ in CONVENTIONS:
            ax.axhline(ASYMPTOTES[metric][norm], color=color, lw=0.8, ls=":", alpha=0.4)
    elif emp_col_mag == "fgt_mag_overlap":
        for norm, color, ls, _ in CONVENTIONS:
            ax.axhline(MAG_ASYMPTOTES_OVERLAP[norm], color=color, lw=0.8, ls=":", alpha=0.4)
    elif emp_col_mag == "fgt_mag_ptonly":
        for norm, color, ls, _ in CONVENTIONS:
            ax.axhline(MAG_ASYMPTOTES_PTONLY[norm], color=color, lw=0.8, ls=":", alpha=0.4)

    ax.set_ylabel(row_title, fontsize=9)
    ax.set_xlabel("α_FT", fontsize=9)
    ax.set_xlim(0, ALPHAS[-1] * 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out2 = OUT_DIR / "sanity_check_teacher_conventions_regimeIII.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close()


# =========================================================
# Figure 3: F ≡ p_FT identity check (ω=1, aligned, Regime II)
# =========================================================
print("\n=== Figure 3: F ≡ p_FT identity check (ω=1, aligned) ===")
t_omega1 = compute_theory("aligned_overlap", omega=1.0)

fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4))
fig3.suptitle(
    "Consistency check: ω=1, aligned_overlap  →  β*_FT = β*_PT everywhere\n"
    "F = E[(β̂−β*_PT)²] and p_FT = E[(β̂−β*_FT)²] must be IDENTICAL.  Gap = bug.",
    fontsize=10,
)

ax = axes3[0]
ax.plot(t_omega1["alpha"], t_omega1["mse"],        color="navy",   lw=3,   label="p_FT")
ax.plot(t_omega1["alpha"], t_omega1["forgetting"], color="tomato", lw=2, ls="--", label="F")
ax.set_title("F vs p_FT  (should overlap exactly)", fontsize=10)
ax.set_xlabel("α_FT"); ax.set_ylabel("value"); ax.legend(); ax.grid(True, alpha=0.3)

ax2 = axes3[1]
gap = t_omega1["forgetting"] - t_omega1["mse"]
ax2.plot(t_omega1["alpha"], gap, color="purple", lw=2)
ax2.axhline(0, color="k", lw=1, ls="--")
ax2.set_title("F − p_FT  (should be ≡ 0)", fontsize=10)
ax2.set_xlabel("α_FT"); ax2.set_ylabel("F − p_FT"); ax2.grid(True, alpha=0.3)

plt.tight_layout()
out3 = OUT_DIR / "sanity_check_identity.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close()

print("\nAll done. Open with:")
print(f"  open {out1}")
print(f"  open {out2}")
print(f"  open {out3}")
