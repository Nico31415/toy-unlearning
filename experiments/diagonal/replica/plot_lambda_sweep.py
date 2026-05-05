"""
Lambda sweep: theory curves for machine unlearning across λ_pt values.

Sweeps λ_pt / c_pt ∈ {0, -0.1, -0.3, -0.5, -0.7, -0.9, -0.99} with aligned_overlap teacher.

For each λ, plots:
  - F^ptonly  [g=2]: how fast PT-only features are forgotten
  - F^overlap [g=0]: how well shared features are preserved
  - p_FT:           FT generalisation error

The "optimal" λ for unlearning maximises F^ptonly while keeping F^overlap ≈ 0.

Run from repo root:
  python experiments/diagonal/replica/plot_lambda_sweep.py
"""
import sys
sys.path.insert(0, "experiments/diagonal/replica")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import hashlib, json

import ptft_replica_qk as rep

# -------------------------
# Config
# -------------------------
RESULTS_DIR   = Path("results/forgetting")
OUT_DIR       = RESULTS_DIR
MC            = 50_000
ALPHAS        = np.linspace(0.01, 0.8, 31)
C_PT          = 1e-3

LAMBDA_FRACS  = [0.0, -0.1, -0.3, -0.5, -0.7, -0.9, -0.95, -0.99]  # λ / c_pt
CACHE_DIR = RESULTS_DIR / "theory_cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

RHO  = 0.1
OMEGA = 0.5
A_PT  = 1.0

# -------------------------
# Cache
# -------------------------

def _cache_key(kw: dict) -> str:
    return hashlib.md5(json.dumps(kw, sort_keys=True).encode()).hexdigest()

def compute_theory(lambda_pt: float):
    kw = dict(rho_pt=RHO, rho_ft=RHO, omega=OMEGA, a_pt=A_PT,
              c_pt=C_PT, lambda_pt=lambda_pt, gamma_reinit=0.0,
              ft_teacher_norm="aligned_overlap",
              alphas=list(np.round(ALPHAS, 6)))
    cache_file = CACHE_DIR / (_cache_key(kw) + ".npz")
    if cache_file.exists():
        print(f"  (cache: λ/c={lambda_pt/C_PT:.2f})")
        d = np.load(cache_file, allow_pickle=True)
        return {k: d[k] for k in d.files}
    print(f"  Computing λ/c={lambda_pt/C_PT:.2f}...", flush=True)
    out = rep.ptft_forgetting_curve(
        rho_pt=RHO, rho_ft=RHO, omega=OMEGA,
        alphas=ALPHAS, mc=MC, seed=0, a_pt=A_PT,
        c_pt=C_PT, lambda_pt=lambda_pt, gamma_reinit=0.0,
        ft_teacher_norm="aligned_overlap",
    )
    saveable = {k: v for k, v in out.items() if isinstance(v, np.ndarray) or np.isscalar(v)}
    np.savez(cache_file, **saveable)
    return out


# -------------------------
# Plot
# -------------------------
colors = cm.plasma(np.linspace(0.1, 0.9, len(LAMBDA_FRACS)))

ROWS = [
    ("forgetting_g2_ptonly",  "F^ptonly [g=2]  ← want HIGH & fast"),
    ("forgetting_g0_overlap", "F^overlap [g=0]  ← want LOW"),
    ("mse",                   "p_FT  (gen. error on FT task)"),
]

fig, axes = plt.subplots(1, len(ROWS), figsize=(5 * len(ROWS), 5), sharey=False)
fig.suptitle(
    "λ sweep: machine unlearning (aligned teacher, ρ_PT=ρ_FT=0.1, ω=0.5, c=1e-3)\n"
    "λ/c from 0 (lazy) → −0.99 (rich).  Optimal = high F^ptonly + low F^overlap",
    fontsize=11,
)

theories = {}
for lam_frac in LAMBDA_FRACS:
    lam = lam_frac * C_PT
    print(f"λ/c = {lam_frac}")
    theories[lam_frac] = compute_theory(lam)

for ax, (metric, title) in zip(axes, ROWS):
    for i, lam_frac in enumerate(LAMBDA_FRACS):
        t = theories[lam_frac]
        lbl = f"λ/c = {lam_frac:+.2f}"
        ax.plot(t["alpha"], t[metric], color=colors[i], lw=2, label=lbl)

    # Asymptotes
    if "ptonly" in metric:
        ax.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.4, label="asymptote=1")
    elif "overlap" in metric:
        ax.axhline(0.0, color="k", lw=0.8, ls=":", alpha=0.4, label="asymptote=0")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("α_FT", fontsize=9)
    ax.set_xlim(0, ALPHAS[-1] * 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# Bonus: F^ptonly / F^overlap ratio plot (selectivity)
fig2, ax2 = plt.subplots(figsize=(6, 4))
fig2.suptitle("Unlearning selectivity: F^ptonly / max(F^overlap, 1e-6)\n"
              "Higher = more selective forgetting per sample", fontsize=10)
for i, lam_frac in enumerate(LAMBDA_FRACS):
    t = theories[lam_frac]
    ratio = t["forgetting_g2_ptonly"] / np.maximum(t["forgetting_g0_overlap"], 1e-6)
    ax2.plot(t["alpha"], ratio, color=colors[i], lw=2, label=f"λ/c={lam_frac:+.2f}")
ax2.set_xlabel("α_FT"); ax2.set_ylabel("F^ptonly / F^overlap")
ax2.set_xlim(0, ALPHAS[-1]*1.05)
ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

plt.figure(fig.number)
plt.tight_layout()
out1 = OUT_DIR / "lambda_sweep_unlearning.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out1}")

plt.figure(fig2.number)
plt.tight_layout()
out2 = OUT_DIR / "lambda_sweep_selectivity.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close("all")
