"""
Empirical-only sanity-check plot: three regimes × three FT teacher conventions.

No theory curves — just empirical means across seeds, with dots connected.

Layout: 4 rows (metrics) × 3 cols (regimes)

Run from repo root:
  python experiments/diagonal/replica/plot_emp_only.py
"""
import sys
sys.path.insert(0, "experiments/diagonal/replica")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# -------------------------
# Config
# -------------------------
EMP_DIR = Path("results/sanity_check")
OUT_DIR = Path("results/forgetting")
SEEDS   = list(range(5))

CONVENTIONS = [
    ("aligned_overlap",  "#1f77b4", "o-",  "aligned  β*_FT[g0]=+1"),
    ("zero_overlap",     "#ff7f0e", "s-",  "zero     β*_FT[g0]=0"),
    ("opposite_overlap", "#d62728", "^-",  "opposite β*_FT[g0]=−1"),
]

REGIMES = [
    ("regime_II",  "Regime II\n(lazy, c=1e-3, λ=0, γ=0)"),
    ("regime_III", "Regime III\n(PT-indep, c=1e-3, λ=0, γ=10)"),
    ("regime_IV",  "Regime IV\n(rich, c=1e-3, λ=−0.95c, γ=0)"),
]

ROWS = [
    ("mse_total",       "p_FT   (gen. error on FT task)"),
    ("fgt_ptonly",      "F^ptonly  [g=2]   (forget PT-only)"),
    ("fgt_overlap",     "F^overlap [g=0]   (retain shared)"),
    ("fgt_mag_overlap", "F_mag^overlap  (|β̂|−|β*_PT|)²  [g=0]"),
]

ASYMPTOTES = {
    "fgt_ptonly":      {"aligned_overlap": 1.0, "zero_overlap": 1.0, "opposite_overlap": 1.0},
    "fgt_overlap":     {"aligned_overlap": 0.0, "zero_overlap": 1.0, "opposite_overlap": 4.0},
    "fgt_mag_overlap": {"aligned_overlap": 0.0, "zero_overlap": 1.0, "opposite_overlap": 0.0},
}


# -------------------------
# Empirical loader
# -------------------------

def load_empirical(regime_name: str, teacher_norm: str, seeds=SEEDS):
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

        spt_path = folder / "support_pt.pt"
        sft_path = folder / "support_ft.pt"
        if spt_path.exists() and sft_path.exists():
            pt_active = torch.load(spt_path, map_location="cpu", weights_only=True).numpy().astype(bool)
            ft_active = torch.load(sft_path, map_location="cpu", weights_only=True).numpy().astype(bool)
        else:
            pt_active = beta_pt != 0
            ft_active = beta_ft != 0

        g_overlap = pt_active & ft_active
        g_ptonly  = pt_active & (~ft_active)

        frac_ov = float(g_overlap.mean()) or 1.0
        frac_pt = float(g_ptonly.mean())  or 1.0

        mse_total       = float(np.mean((beta_hat - beta_ft) ** 2))
        fgt_ptonly      = float(np.mean((beta_hat - beta_pt) ** 2 * g_ptonly))  / frac_pt
        fgt_overlap     = float(np.mean((beta_hat - beta_pt) ** 2 * g_overlap)) / frac_ov
        fgt_mag_overlap = float(np.mean((np.abs(beta_hat) - np.abs(beta_pt)) ** 2 * g_overlap)) / frac_ov

        records.append(dict(seed=seed, alpha=alpha,
                            mse_total=mse_total,
                            fgt_ptonly=fgt_ptonly,
                            fgt_overlap=fgt_overlap,
                            fgt_mag_overlap=fgt_mag_overlap))

    if not records:
        return None
    df = pd.DataFrame(records).groupby("alpha").mean(numeric_only=True).reset_index()
    return df


# -------------------------
# Load all data
# -------------------------
print("Loading empirical data...")
emp = {}
for rname, _ in REGIMES:
    emp[rname] = {}
    for norm, _, _, _ in CONVENTIONS:
        emp[rname][norm] = load_empirical(rname, norm)
        n = 0 if emp[rname][norm] is None else len(emp[rname][norm])
        print(f"  {rname}/{norm}: {n} alpha points")


# -------------------------
# Combined figure: 4 rows × 3 cols
# -------------------------
fig, axes = plt.subplots(len(ROWS), len(REGIMES),
                         figsize=(5 * len(REGIMES), 4 * len(ROWS)),
                         sharey="row")

fig.suptitle(
    "Empirical sanity check — three regimes × three FT teacher conventions\n"
    "ρ_PT=ρ_FT=0.1, ω=0.5, a_PT=1  |  mean over seeds (dots connected)\n"
    "Asymptotes at α→∞:  F^ptonly→1 (all three),  F^overlap→ 0 / 1 / 4,  F_mag^overlap→ 0 / 1 / 0",
    fontsize=11,
)

for col, (rname, rtitle) in enumerate(REGIMES):
    for row, (col_name, row_title) in enumerate(ROWS):
        ax = axes[row][col]

        any_data = False
        for norm, color, fmt, label in CONVENTIONS:
            df = emp[rname].get(norm)
            if df is not None and col_name in df.columns and len(df) > 0:
                ax.plot(df["alpha"], df[col_name],
                        fmt, color=color, lw=1.8, ms=5, label=label)
                any_data = True

        # Asymptote dashed lines
        if col_name in ASYMPTOTES:
            for norm, color, _, _ in CONVENTIONS:
                ax.axhline(ASYMPTOTES[col_name][norm], color=color,
                           lw=0.8, ls=":", alpha=0.35)

        if row == 0:
            ax.set_title(rtitle, fontsize=10, fontweight="bold")
        if col == 0:
            ax.set_ylabel(row_title, fontsize=9)
        ax.set_xlabel("α_FT", fontsize=8)
        ax.grid(True, alpha=0.3)

        if not any_data:
            ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=10)
        elif row == 0 and col == 0:
            ax.legend(fontsize=7, loc="upper left")

# Single shared legend at bottom
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(CONVENTIONS),
           fontsize=9, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.03, 1, 1])
out = OUT_DIR / "sanity_check_emp_only.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close()
print(f"open {out}")
