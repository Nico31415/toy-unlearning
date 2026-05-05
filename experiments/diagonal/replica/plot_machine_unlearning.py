"""
Machine unlearning: selective forgetting via aligned vs independent FT teacher.

This plot asks: can fine-tuning selectively forget PT-only features (F^ptonly↑)
while KEEPING shared features (F^overlap→0)?

Two FT teacher conventions:
  - "independent" (ft_teacher_norm="unit_total_var"):  β*_FT[g=0] ~ N(0,σ)
        → F^overlap → 1/ρ_FT + 1 = 11 at α→∞  (model replaces shared features)
  - "aligned_overlap" (ft_teacher_norm="aligned_overlap"): β*_FT[g=0] = a_pt
        → F^overlap → 0 at α→∞               (model preserves shared features)
        → F^ptonly  → a_pt² = 1               (model forgets PT-only features)

Panels (rows):
  1.  FT generalisation error  p_FT
  2.  Total forgetting  F
  3.  PT-only forgetting  F^ptonly  [g=2]  ← want HIGH for unlearning
  4.  Overlap forgetting  F^overlap [g=0]  ← want LOW  for unlearning

Columns: one per regime.

Run from repo root:
  python experiments/diagonal/replica/plot_machine_unlearning.py
"""
import sys
sys.path.insert(0, "experiments/diagonal/replica")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import hashlib, json

import ptft_replica_qk as rep

# -------------------------
# Config
# -------------------------
RESULTS_DIR  = Path("results/forgetting")
OUT_DIR      = RESULTS_DIR
MC           = 50_000
THEORY_ALPHAS = np.linspace(0.01, 0.5, 21)   # match existing cache grid

REGIMES = [
    ("regime_II",   dict(c_pt=1e-3, lambda_pt=0.0,       gamma_reinit=0.0),  "Regime II (lazy)\nc=1e-3, λ=0, γ=0"),
    ("regime_IV",   dict(c_pt=1e-3, lambda_pt=-0.99e-3,  gamma_reinit=0.0),  "Regime IV (rich)\nc=1e-3, λ=-0.99c, γ=0"),
    ("regime_III",  dict(c_pt=1e-3, lambda_pt=0.0,       gamma_reinit=10.0), "Regime III (lazy, PT-indep)\nc=1e-3, λ=0, γ=10"),
    ("regime_largec",dict(c_pt=2.0, lambda_pt=0.0,       gamma_reinit=0.0),  "Large c\nc=2, λ=0, γ=0"),
]

CACHE_DIR = RESULTS_DIR / "theory_cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# -------------------------
# Caching
# -------------------------

def _cache_key(regime_kw: dict, alphas, teacher_norm: str) -> str:
    d = {**regime_kw, "alphas": list(np.round(alphas, 6))}
    # unit_total_var is the historical default — its caches were written without
    # ft_teacher_norm in the key (by plot_forgetting_theory_vs_emp.py).  Keep the
    # same key so we reuse those files rather than recompute.
    if teacher_norm != "unit_total_var":
        d["ft_teacher_norm"] = teacher_norm
    payload = json.dumps(d, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def compute_theory(regime_kw: dict, teacher_norm: str, alphas=THEORY_ALPHAS):
    cache_file = CACHE_DIR / (_cache_key(regime_kw, alphas, teacher_norm) + ".npz")
    if cache_file.exists():
        print(f"  (cache hit: {cache_file.name[:12]}... teacher={teacher_norm})")
        d = np.load(cache_file, allow_pickle=True)
        return {k: d[k] for k in d.files}

    print(f"  Computing theory (teacher={teacher_norm})...", flush=True)
    out = rep.ptft_forgetting_curve(
        rho_pt=0.1, rho_ft=0.1, omega=0.5,
        alphas=alphas, mc=MC, seed=0,
        a_pt=1.0,
        ft_teacher_norm=teacher_norm,
        **regime_kw,
    )
    saveable = {k: v for k, v in out.items() if isinstance(v, np.ndarray) or np.isscalar(v)}
    np.savez(cache_file, **saveable)
    return out


# -------------------------
# Plot
# -------------------------

ROWS = [
    ("mse",                   "FT generalisation error  p_FT"),
    ("forgetting",            "Total forgetting  F"),
    ("forgetting_g2_ptonly",  "PT-only forgetting  F^ptonly  [g=2]  ← want HIGH"),
    ("forgetting_g0_overlap", "Overlap forgetting  F^overlap [g=0]  ← want LOW"),
]
N_ROWS = len(ROWS)
N_COLS = len(REGIMES)

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(5 * N_COLS, 4 * N_ROWS), sharey="row")
fig.suptitle(
    "Machine unlearning: aligned vs independent FT teacher  (ρ_PT=ρ_FT=0.1, ω=0.5, a_PT=1)\n"
    "Aligned (solid): β*_FT[g=0]=a_pt  →  F^overlap→0 (keep shared),  F^ptonly→1 (forget PT-only)\n"
    "Independent (dashed): β*_FT[g=0]~N(0,σ) →  F^overlap→11 (replaces everything)",
    fontsize=11,
)

COLOR_ALIGNED     = "#1f77b4"  # blue
COLOR_INDEPENDENT = "#d62728"  # red

for col, (regime_name, kw, label) in enumerate(REGIMES):
    print(f"\n=== {regime_name} ===")

    print(f"  [aligned_overlap]")
    t_aligned = compute_theory(kw, "aligned_overlap")
    print(f"  [unit_total_var (independent)]")
    t_indep   = compute_theory(kw, "unit_total_var")

    for row, (metric_key, row_title) in enumerate(ROWS):
        ax = axes[row, col]

        ax.plot(t_aligned["alpha"], t_aligned[metric_key],
                color=COLOR_ALIGNED, lw=2.5, label="Aligned (unlearning)")
        ax.plot(t_indep["alpha"],   t_indep[metric_key],
                color=COLOR_INDEPENDENT, lw=2.5, ls="--", label="Independent (forgetting)")

        if row == 0:
            ax.set_title(label, fontsize=10)
        if col == 0:
            ax.set_ylabel(row_title, fontsize=9)
        ax.set_xlabel("α_FT", fontsize=9)
        ax.set_xlim(0, 0.52)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Annotate asymptotes on overlap row
        if "overlap" in metric_key:
            rho_ft = 0.1
            a_pt   = 1.0
            asym_indep   = 1.0 / rho_ft + a_pt**2   # = 11
            asym_aligned = 0.0
            ax.axhline(asym_indep,   color=COLOR_INDEPENDENT, lw=1, ls=":", alpha=0.6,
                       label=f"asymptote ≈ {asym_indep:.0f}")
            ax.axhline(asym_aligned, color=COLOR_ALIGNED,     lw=1, ls=":", alpha=0.6,
                       label=f"asymptote = 0")

plt.tight_layout()
out_path = OUT_DIR / "machine_unlearning_aligned_vs_indep.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.close()
