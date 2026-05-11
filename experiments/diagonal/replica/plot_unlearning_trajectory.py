"""
Machine unlearning trajectory plot.

Single 2D plot:
  x-axis: F^overlap [g=0]  — want LOW (← left is better)
  y-axis: F^ptonly  [g=2]  — want HIGH (↑ up is better)

Each regime traces a parametric curve as α_FT increases, but only the
"useful" portion is shown: α values where p_FT ≤ 0.1 × p_FT(α→0).
(Before this threshold the FT model hasn't learned the task yet so
forgetting is irrelevant.)

Curve colour = α value (shared colourmap + colourbar) → encodes sample
efficiency directly: curves that enter the useful range at darker colours
(lower α) are more sample-efficient.

A filled marker + α annotation marks the threshold crossing for each regime.
The ideal point (0, 1) is marked with a star.

Run from repo root:
  python experiments/diagonal/replica/plot_unlearning_trajectory.py
"""
import sys
sys.path.insert(0, "experiments/diagonal/replica")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
import matplotlib.colorbar
from pathlib import Path
import hashlib, json

import ptft_replica_qk as rep

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/forgetting")
CACHE_DIR   = RESULTS_DIR / "theory_cache"
OUT_DIR     = RESULTS_DIR
MC          = 50_000
ALPHAS      = np.linspace(0.01, 0.5, 21)
RHO = 0.1; OMEGA = 0.5; A_PT = 1.0
C_PT_DEFAULT = 1e-3

P_FT_THRESHOLD_FRAC = 0.10   # "useful" once p_FT ≤ 10% of initial value

REGIMES = [
    # (label, c_pt, lambda_pt, gamma_reinit, color, marker)
    ("Regime II\n(lazy, λ=0)",          1e-3,  0.0,       0.0,  "#1f77b4", "o"),
    ("Regime III\n(PT-indep, γ=10)",    1e-3,  0.0,      10.0,  "#d62728", "s"),
    ("Regime IV\n(λ=−0.95c)",           1e-3, -0.95e-3,   0.0,  "#2ca02c", "^"),
    ("Regime IV\n(λ=−0.99c)",           1e-3, -0.99e-3,   0.0,  "#ff7f0e", "D"),
    ("Large c\n(c=2, λ=0)",             2.0,   0.0,       0.0,  "#9467bd", "P"),
]

# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_key(kw):
    return hashlib.md5(json.dumps(kw, sort_keys=True).encode()).hexdigest()

def load_or_compute(c_pt, lambda_pt, gamma_reinit, alphas=ALPHAS):
    kw = dict(c_pt=c_pt, lambda_pt=lambda_pt, gamma_reinit=gamma_reinit,
              ft_teacher_norm="aligned_overlap",
              alphas=list(np.round(alphas, 6)))
    cache_file = CACHE_DIR / (_cache_key(kw) + ".npz")
    if cache_file.exists():
        d = np.load(cache_file, allow_pickle=True)
        return {k: d[k] for k in d.files}
    print(f"  Computing c={c_pt}, λ={lambda_pt}, γ={gamma_reinit}...", flush=True)
    out = rep.ptft_forgetting_curve(
        rho_pt=RHO, rho_ft=RHO, omega=OMEGA, alphas=alphas,
        mc=MC, seed=0, a_pt=A_PT,
        c_pt=c_pt, lambda_pt=lambda_pt, gamma_reinit=gamma_reinit,
        ft_teacher_norm="aligned_overlap",
    )
    saveable = {k: v for k, v in out.items() if isinstance(v, np.ndarray) or np.isscalar(v)}
    np.savez(cache_file, **saveable)
    return out

# ── Load all theory curves ─────────────────────────────────────────────────────
print("Loading theory curves (aligned_overlap teacher)...")
data = {}
for label, c_pt, lambda_pt, gamma_reinit, color, marker in REGIMES:
    short = label.replace("\n", " ")
    print(f"  {short}")
    data[label] = load_or_compute(c_pt, lambda_pt, gamma_reinit)

# ── Build plot ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

# Shared colourmap for α values across all regimes
alpha_min_global = ALPHAS[0]
alpha_max_global = ALPHAS[-1]
cmap   = cm.viridis
norm   = matplotlib.colors.Normalize(vmin=alpha_min_global, vmax=alpha_max_global)

for label, c_pt, lambda_pt, gamma_reinit, color, marker in REGIMES:
    t       = data[label]
    alphas  = t["alpha"]
    p_ft    = t["mse"]
    f_ov    = t["forgetting_g0_overlap"]
    f_pt    = t["forgetting_g2_ptonly"]

    # Threshold: p_FT ≤ 10% of the smallest-α value (proxy for α→0)
    p_ft_initial = p_ft[0]
    threshold    = P_FT_THRESHOLD_FRAC * p_ft_initial
    useful_mask  = p_ft <= threshold

    if not useful_mask.any():
        print(f"  Warning: {label.replace(chr(10),' ')} never reaches p_FT threshold in α=[{alphas[0]:.3f},{alphas[-1]:.3f}]")
        # Plot full curve faded
        ax.plot(f_ov, f_pt, color=color, lw=1, ls="--", alpha=0.25)
        continue

    # Split into pre-threshold (faded dashed) and post-threshold (coloured solid)
    first_useful_idx = int(np.argmax(useful_mask))

    # --- pre-threshold (faded) ---
    pre_end = first_useful_idx + 1          # include the crossing point
    ax.plot(f_ov[:pre_end], f_pt[:pre_end],
            color=color, lw=1.2, ls="--", alpha=0.25)

    # --- post-threshold: coloured by α ---
    x_seg = f_ov[first_useful_idx:]
    y_seg = f_pt[first_useful_idx:]
    a_seg = alphas[first_useful_idx:]

    # Build line segments coloured by α
    points  = np.array([x_seg, y_seg]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_colors = cmap(norm(0.5 * (a_seg[:-1] + a_seg[1:])))
    lc = LineCollection(segs, colors=seg_colors, lw=2.5, zorder=3)
    ax.add_collection(lc)

    # Threshold-crossing marker
    xm, ym, am = f_ov[first_useful_idx], f_pt[first_useful_idx], alphas[first_useful_idx]
    ax.scatter([xm], [ym], color=color, s=100, marker=marker,
               zorder=5, edgecolors="k", linewidths=0.8)
    ax.annotate(f"α={am:.2f}", xy=(xm, ym),
                xytext=(6, 4), textcoords="offset points",
                fontsize=8, color=color, fontweight="bold")

    # End-of-curve dot (asymptote)
    ax.scatter([f_ov[-1]], [f_pt[-1]], color=color, s=40, marker=marker,
               zorder=4, alpha=0.6)

    # Legend proxy (invisible line with the right color)
    ax.plot([], [], color=color, lw=2.5, ls="-",
            marker=marker, ms=7, label=label.replace("\n", "  "))

# Ideal point
ax.scatter([0], [1], marker="*", s=350, color="gold", edgecolors="k",
           linewidths=0.8, zorder=6, label="Ideal (0, 1)")
ax.annotate("ideal", xy=(0, 1), xytext=(6, -10),
            textcoords="offset points", fontsize=8, color="goldenrod")

# Colourbar for α
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.85)
cbar.set_label("α_FT (FT sample ratio)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Threshold annotation
ax.text(0.98, 0.03,
        f"Solid = p_FT ≤ {int(P_FT_THRESHOLD_FRAC*100)}% of initial\n"
        f"Markers = threshold crossing (α)\nFaded dashed = model not yet useful",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
        color="gray",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.8))

ax.set_xlabel("F^overlap [g=0]   (shared feature forgetting)  ← want LOW", fontsize=10)
ax.set_ylabel("F^ptonly [g=2]   (PT-only forgetting)  ↑ want HIGH", fontsize=10)
ax.set_title(
    "Machine unlearning quality — aligned_overlap teacher\n"
    "ρ_PT=ρ_FT=0.1, ω=0.5, a_PT=1  |  theory (replica saddle point)",
    fontsize=10,
)
ax.set_xlim(-0.02, None)
ax.set_ylim(-0.02, 1.08)
ax.axvline(0, color="k", lw=0.6, ls=":", alpha=0.3)
ax.axhline(1, color="k", lw=0.6, ls=":", alpha=0.3)
ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.25)

plt.tight_layout()
out = OUT_DIR / "unlearning_trajectory.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close()
print(f"open {out}")
