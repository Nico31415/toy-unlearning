# --- Quick-and-dirty replica PT+FT imperfect pretraining sanity check ---
# Reduced settings (mc=5k, 20 alpha_ft pts, tol=1e-4, max_iters=200) for fast iteration.
# Sweeps both alpha_pt (panel 1) and sigma0_pt (panel 2) so both imperfect PT axes can be checked.
# Saves CSV before plotting; prints fwd/bwd gap table after each curve.
# Figure is 2x2: rows = omega in {1.0, 0.0}, cols = alpha_pt sweep / sigma0_pt sweep.
# In the alpha_pt panels, alpha_pt=1.0 is computed twice (imperfect code + oracle code) to verify match.

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ptft_replica_imperfect_pt as rip
import ptft_replica_qk as rq

# --- Reduced settings ---
alphas_ft   = np.unique(np.concatenate([np.linspace(0.01, 0.5, 20), [0.2, 0.3]]))
MC          = 5_000
TOL         = 1e-4
MAX_ITERS   = 200
DAMP        = 0.25
SEED        = 0

# --- Fixed params ---
RHO_PT       = 0.10
RHO_FT       = 0.10
C_PT         = 1e-3
LAMBDA_PT    = 0.0
GAMMA_REINIT = 0.0
A_PT         = 1.0
GAMMA_EXT    = 1e-6
SIGMA0_2     = 0.0

OMEGA_LIST     = [1.0, 0.0]
ALPHA_PT_LIST  = [0.01, 0.2, 0.5, 1.0]
SIGMA0_PT_LIST = [0.0, 1.0, 10.0]

# ---------------------------------------------------------------------------

def _run_imperfect(alpha_pt, sigma0_pt, omega, label):
    print(f"\n=== {label} (imperfect) ===", flush=True)
    curve, reliability, info = rip.ptft_qk_curve_imperfect_pt(
        rho_pt=RHO_PT, rho_ft=RHO_FT, omega=omega,
        alpha_pt=alpha_pt, sigma0_pt=sigma0_pt,
        gamma_ext=GAMMA_EXT, sigma0_2=SIGMA0_2,
        alphas=alphas_ft, mc=MC, seed=SEED,
        a_pt=A_PT, c_pt=C_PT, lambda_pt=LAMBDA_PT, gamma_reinit=GAMMA_REINIT,
        tol=TOL, max_iters=MAX_ITERS, damp=DAMP,
    )
    _print_table(curve, reliability)
    n = len(curve["alpha"])
    return pd.DataFrame({
        "label":          [label] * n,
        "solver":         ["imperfect"] * n,
        "omega":          np.full(n, omega),
        "alpha_pt":       np.full(n, alpha_pt),
        "sigma0_pt":      np.full(n, sigma0_pt),
        "alpha":          curve["alpha"],
        "mse_best":       curve["mse_best"],
        "mse_fwd":        curve["mse_fwd"],
        "mse_bwd":        curve["mse_bwd"],
        "diff_db":        curve["diff_db"],
        "fp_residual":    curve["fp_residual"],
        "reliability_db": reliability["score_db"],
        "s2_pt":          info.get("s2_pt", float("nan")),
        "gp_pt":          info.get("gp_pt", float("nan")),
    })


def _run_oracle(omega, label):
    print(f"\n=== {label} (oracle) ===", flush=True)
    curve, reliability, _ = rq.ptft_qk_curve(
        rho_pt=RHO_PT, rho_ft=RHO_FT, omega=omega,
        gamma_ext=GAMMA_EXT, sigma0_2=SIGMA0_2,
        alphas=alphas_ft, mc=MC, seed=SEED,
        a_pt=A_PT, c_pt=C_PT, lambda_pt=LAMBDA_PT, gamma_reinit=GAMMA_REINIT,
        tol=TOL, max_iters=MAX_ITERS, damp=DAMP,
    )
    _print_table(curve, reliability)
    n = len(curve["alpha"])
    return pd.DataFrame({
        "label":          [label] * n,
        "solver":         ["oracle"] * n,
        "omega":          np.full(n, omega),
        "alpha_pt":       np.full(n, 1.0),
        "sigma0_pt":      np.full(n, 0.0),
        "alpha":          curve["alpha"],
        "mse_best":       curve["mse_best"],
        "mse_fwd":        curve["mse_fwd"],
        "mse_bwd":        curve["mse_bwd"],
        "diff_db":        curve["diff_db"],
        "fp_residual":    curve["fp_residual"],
        "reliability_db": reliability["score_db"],
        "s2_pt":          float("nan"),
        "gp_pt":          float("nan"),
    })


def _print_table(curve, reliability):
    print(f"  {'alpha_ft':>8}  {'mse_best':>9}  {'mse_fwd':>9}  {'mse_bwd':>9}  {'diff_db':>8}")
    for a, mb, mf, mr, dd in zip(
        curve["alpha"], curve["mse_best"], curve["mse_fwd"],
        curve["mse_bwd"], curve["diff_db"]
    ):
        print(f"  {a:8.3f}  {mb:9.5f}  {mf:9.5f}  {mr:9.5f}  {dd:8.3f}")
    print(f"  reliability_score_db = {reliability['score_db']:.2f}", flush=True)


# ---------------------------------------------------------------------------

records = []

for omega in OMEGA_LIST:
    # Panel 1: alpha_pt sweep (imperfect) + oracle alpha_pt=1 for comparison
    for alpha_pt in ALPHA_PT_LIST:
        records.append(_run_imperfect(alpha_pt, 0.0, omega, label=f"w{omega}_apt{alpha_pt}"))
    records.append(_run_oracle(omega, label=f"w{omega}_oracle"))

    # Panel 2: sigma0_pt sweep (alpha_pt=1, imperfect code)
    for sigma0_pt in SIGMA0_PT_LIST:
        records.append(_run_imperfect(1.0, sigma0_pt, omega, label=f"w{omega}_s0{sigma0_pt}"))

df = pd.concat(records, ignore_index=True)
out_csv = Path(__file__).parent / "replica_quick.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved {len(df)} rows → {out_csv}")

# --- Plot (2 rows x 2 cols) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
fig.suptitle(
    f"Quick replica check  (ρ_PT={RHO_PT}, ρ_FT={RHO_FT}, c_PT={C_PT}, λ_PT={LAMBDA_PT}, γ_FT={GAMMA_REINIT}, mc={MC}, tol={TOL})",
    fontsize=9,
)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
DIFF_DB_MASK = 5.0  # hide points where fwd/bwd gap exceeds this

def _masked(sub):
    s = sub[sub["diff_db"] <= DIFF_DB_MASK]
    return s["alpha"], s["mse_best"]

for row, omega in enumerate(OMEGA_LIST):
    sub_omega = df[df["omega"] == omega]

    # --- Col 0: alpha_pt sweep ---
    ax = axes[row, 0]
    for i, alpha_pt in enumerate(ALPHA_PT_LIST):
        sub = sub_omega[sub_omega["label"] == f"w{omega}_apt{alpha_pt}"]
        x, y = _masked(sub)
        ax.plot(x, y, color=colors[i], label=f"α_PT={alpha_pt} (imperfect)")
    sub_or = sub_omega[sub_omega["label"] == f"w{omega}_oracle"]
    x, y = _masked(sub_or)
    ax.plot(x, y, "k--", lw=1.5, label="α_PT=1 (oracle)")
    ax.set_xlabel("α_FT")
    ax.set_ylabel("MSE")
    ax.set_title(f"ω={omega}  — effect of α_PT  (σ²₀,PT=0)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Col 1: sigma0_pt sweep ---
    ax = axes[row, 1]
    for i, sigma0_pt in enumerate(SIGMA0_PT_LIST):
        sub = sub_omega[sub_omega["label"] == f"w{omega}_s0{sigma0_pt}"]
        x, y = _masked(sub)
        ax.plot(x, y, color=colors[i], label=f"σ²₀,PT={sigma0_pt}")
    x, y = _masked(sub_or)
    ax.plot(x, y, "k--", lw=1.5, label="oracle")
    ax.set_xlabel("α_FT")
    ax.set_title(f"ω={omega}  — effect of σ²₀,PT  (α_PT=1)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_png = Path(__file__).parent / "replica_quick.png"
fig.savefig(out_png, dpi=150)
print(f"Saved figure → {out_png}")
