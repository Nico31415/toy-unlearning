#!/usr/bin/env python3
"""
Sanity-check script for the noisy-pretraining-labels extension.

Runs four curves:
  A: oracle            (alpha_pt=1, sigma0_pt=0)       -- baseline
  B: tiny noise        (alpha_pt=1, sigma0_pt=1e-6)    -- should ≈ A
  C: huge noise        (alpha_pt=1, sigma0_pt=100.0)   -- PT info destroyed
  D: single-task ref   single_task_qk_curve(c=2e-3)   -- no PT signal at all

Plots all four on one figure (linear axes, alpha_ft capped at 0.5).

Expected:
  - A and B overlap closely (tiny noise ≈ oracle).
  - C and D are close (huge PT noise ≈ no PT information).
"""
from __future__ import annotations

import sys
import os
import math

import numpy as np

# ── make sure the replica directory is on the path ──────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ptft_replica_imperfect_pt as rip
from ptft_replica_qk import single_task_qk_curve

# ── shared parameters ────────────────────────────────────────────────────────
COMMON = dict(
    rho_pt=0.10,
    rho_ft=0.10,
    omega=1.0,
    c_pt=1e-3,
    lambda_pt=0.0,
    gamma_reinit=0.0,
    a_pt=1.0,
    gamma_ext=1e-6,
    sigma0_2=0.0,
    mc=20_000,
    seed=0,
    alphas=np.linspace(0.01, 0.50, 60),
    tol=1e-6,
    max_iters=900,
    damp=0.25,
)

# ── curve A: oracle (alpha_pt=1, noiseless) ──────────────────────────────────
print("Running curve A: oracle (alpha_pt=1, sigma0_pt=0) …", flush=True)
curve_A, _, info_A = rip.ptft_qk_curve_imperfect_pt(alpha_pt=1.0, sigma0_pt=0.0, **COMMON)
print(f"  oracle shortcut used: {info_A['oracle']}")
np.save(os.path.join(_HERE, "_sanity_mse_A.npy"), curve_A["mse_best"])
np.save(os.path.join(_HERE, "_sanity_alpha.npy"), curve_A["alpha"])

# ── curve B: tiny noise ───────────────────────────────────────────────────────
print("Running curve B: tiny noise (alpha_pt=1, sigma0_pt=1e-6) …", flush=True)
curve_B, _, info_B = rip.ptft_qk_curve_imperfect_pt(alpha_pt=1.0, sigma0_pt=1e-6, **COMMON)
print(f"  oracle shortcut used: {info_B['oracle']},  tau_PT={info_B['s2_pt']:.3e}")
np.save(os.path.join(_HERE, "_sanity_mse_B.npy"), curve_B["mse_best"])

# ── curve C: high noise ───────────────────────────────────────────────────────
print("Running curve C: huge noise (alpha_pt=1, sigma0_pt=100) …", flush=True)
curve_C, _, info_C = rip.ptft_qk_curve_imperfect_pt(alpha_pt=1.0, sigma0_pt=10.0, **COMMON)
print(f"  oracle shortcut used: {info_C['oracle']},  tau_PT={info_C['s2_pt']:.3e}")

# Explain the single-task reference value:
#   Psi(0) with c_pt=1e-3, lambda_pt=0, gamma_reinit=0:
#     c_ft  = (0 + 1e-3) * (1 + sqrt(1+0)) + 0.5*0 = 2e-3
#     k     = 4 * c_ft^2 = 4 * (2e-3)^2 = 1.6e-5
#   single_task_qk_curve uses k = 4*c^2, so c = 2e-3
C_SINGLE = 2e-3
print(f"Running curve D: single-task reference (c={C_SINGLE}) …", flush=True)
curve_D, _, _ = single_task_qk_curve(
    rho=0.10,
    c=C_SINGLE,
    gamma_ext=1e-6,
    sigma0_2=0.0,
    alphas=COMMON["alphas"],
    mc=COMMON["mc"],
    seed=COMMON["seed"],
    tol=COMMON["tol"],
    max_iters=COMMON["max_iters"],
    damp=COMMON["damp"],
)

# ── quick numerical check ────────────────────────────────────────────────────
alpha_arr = curve_A["alpha"]
mse_A = curve_A["mse_best"]
mse_B = curve_B["mse_best"]
mse_C = curve_C["mse_best"]
mse_D = curve_D["mse_best"]

max_diff_AB = float(np.max(np.abs(mse_A - mse_B)))
print(f"\nMax |MSE_A - MSE_B| = {max_diff_AB:.2e}  (should be tiny)")

# ── save to disk ──────────────────────────────────────────────────────────────
import pandas as pd

rows = []
for label, mse, sigma0_pt_val in [
    ("oracle",      mse_A, 0.0),
    ("tiny_noise",  mse_B, 1e-6),
    ("huge_noise",  mse_C, 100.0),
    ("single_task", mse_D, None),
]:
    for i, a in enumerate(alpha_arr):
        rows.append({"curve": label, "sigma0_pt": sigma0_pt_val, "alpha": a, "mse": mse[i]})

df = pd.DataFrame(rows)
out_csv = os.path.join(_HERE, "check_noisy_pt_sanity.csv")
df.to_csv(out_csv, index=False)
print(f"Data saved to {out_csv}")

# ── plot ─────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(alpha_arr, mse_A, color="#1f77b4", lw=2,      label=r"A: oracle ($\sigma^2_{PT}=0$)")
    ax.plot(alpha_arr, mse_B, color="#1f77b4", lw=2, ls="--", label=r"B: tiny noise ($\sigma^2_{PT}=10^{-6}$)")
    ax.plot(alpha_arr, mse_C, color="#d62728", lw=2,      label=r"C: high noise ($\sigma^2_{PT}=10$)")
    ax.plot(alpha_arr, mse_D, color="#d62728", lw=2, ls="--", label=r"D: single-task ref ($c=2\times10^{-3}$)")

    ax.set_xlabel(r"$\alpha_{FT}$", fontsize=13)
    ax.set_ylabel("Generalisation MSE", fontsize=13)
    ax.set_title(
        "Noisy PT labels sanity check\n"
        r"($\alpha_{PT}=1$, $\omega=1$, $\rho=0.1$, $c_{PT}=10^{-3}$)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.01, 0.50)

    out_png = os.path.join(_HERE, "check_noisy_pt_sanity.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\nFigure saved to {out_png}")

except Exception as exc:
    print(f"\nPlotting skipped ({exc})")

print("\nDone.")
