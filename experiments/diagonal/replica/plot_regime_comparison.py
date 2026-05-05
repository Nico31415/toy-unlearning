"""
Quick regime comparison plot using cached theory curves (no new solves).

Shows F^ptonly, F^overlap, p_FT for all regimes with aligned_overlap teacher.
Uses whatever is already in the theory cache.

Run from repo root:
  python experiments/diagonal/replica/plot_regime_comparison.py
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

RESULTS_DIR = Path("results/forgetting")
CACHE_DIR   = RESULTS_DIR / "theory_cache"
OUT_DIR     = RESULTS_DIR
MC          = 50_000
ALPHAS      = np.linspace(0.01, 0.5, 21)   # existing cache grid
RHO = 0.1; OMEGA = 0.5; A_PT = 1.0

def _cache_key(kw):
    return hashlib.md5(json.dumps(kw, sort_keys=True).encode()).hexdigest()

def _make_cache_key(c_pt, lambda_pt, gamma_reinit, teacher_norm, alphas):
    """Match the key convention used by plot_machine_unlearning.py."""
    d = dict(c_pt=c_pt, lambda_pt=lambda_pt, gamma_reinit=gamma_reinit,
             alphas=list(np.round(alphas, 6)))
    if teacher_norm != "unit_total_var":
        d["ft_teacher_norm"] = teacher_norm
    return _cache_key(d)

def load_or_compute(label, c_pt, lambda_pt, gamma_reinit, teacher_norm, alphas=ALPHAS):
    cache_file = CACHE_DIR / (_make_cache_key(c_pt, lambda_pt, gamma_reinit, teacher_norm, alphas) + ".npz")
    if cache_file.exists():
        print(f"  (cache hit: {label.replace(chr(10),' ')})")
        d = np.load(cache_file, allow_pickle=True)
        return {k: d[k] for k in d.files}
    print(f"  Computing {label} (teacher={teacher_norm})...", flush=True)
    out = rep.ptft_forgetting_curve(
        rho_pt=RHO, rho_ft=RHO, omega=OMEGA, alphas=alphas,
        mc=MC, seed=0, a_pt=A_PT,
        c_pt=c_pt, lambda_pt=lambda_pt, gamma_reinit=gamma_reinit,
        ft_teacher_norm=teacher_norm,
    )
    saveable = {k: v for k, v in out.items() if isinstance(v, np.ndarray) or np.isscalar(v)}
    np.savez(cache_file, **saveable)
    return out

REGIMES = [
    ("Regime II\n(lazy, λ=0)",          1e-3,  0.0,      0.0,  "#1f77b4", "-"),
    ("Regime III\n(PT-indep, γ=10)",     1e-3,  0.0,     10.0,  "#d62728", "-"),
    ("Regime IV\n(rich, λ=−0.95c)",      1e-3, -0.95e-3,  0.0,  "#2ca02c", "-"),
    ("Regime IV\n(rich, λ=−0.99c)",      1e-3, -0.99e-3,  0.0,  "#ff7f0e", "--"),
    ("Large c\n(c=2, λ=0)",              2.0,   0.0,      0.0,  "#9467bd", "-"),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(
    "Regime comparison — aligned teacher (β*_FT[g=0]=β*_PT)  |  ρ_PT=ρ_FT=0.1, ω=0.5\n"
    "Claims: Regime III fastest overall; Regime IV most selective; intermediate λ best trade-off",
    fontsize=11,
)

metrics = [
    ("forgetting_g2_ptonly",  "F^ptonly [g=2]\n← want HIGH (forget PT-only)"),
    ("forgetting_g0_overlap", "F^overlap [g=0]\n← want LOW (keep shared)"),
    ("mse",                   "p_FT (gen. error)\n← want LOW (learn FT task)"),
    (None,                    "Selectivity  F^ptonly / F^overlap\n← want HIGH"),
]

data = {}
for label, c_pt, lambda_pt, gamma_reinit, color, ls in REGIMES:
    print(f"Loading {label.replace(chr(10),' ')}  aligned...")
    data[label] = load_or_compute(label, c_pt, lambda_pt, gamma_reinit, "aligned_overlap")

for ax, (metric, title) in zip(axes, metrics):
    for label, c_pt, lambda_pt, gamma_reinit, color, ls in REGIMES:
        t = data[label]
        if metric is None:
            ptonly  = t["forgetting_g2_ptonly"]
            overlap = np.maximum(t["forgetting_g0_overlap"], 1e-6)
            y = ptonly / overlap
        else:
            y = t[metric]
        ax.plot(t["alpha"], y, color=color, ls=ls, lw=2.5,
                label=label.replace("\n", " "))

    if metric == "forgetting_g2_ptonly":
        ax.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.4)
    elif metric == "forgetting_g0_overlap":
        ax.axhline(0.0, color="k", lw=0.8, ls=":", alpha=0.4)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("α_FT", fontsize=9)
    ax.set_xlim(0, ALPHAS[-1] * 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = OUT_DIR / "regime_comparison_aligned.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close()
print(f"open {out}")
