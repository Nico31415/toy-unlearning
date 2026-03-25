# --- Replica PT+FT imperfect pretraining (standalone script) ---
# Generates replica curves for alpha_pt in {0.01, 0.5, 1.0} in the full-overlap,
# sparsity=0.1 setting (lambda_pt=0, c_pt=1e-3, gamma_reinit=0).
# Mirrors the structure of compute_replica_curves.py.

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ptft_replica_imperfect_pt as rip

# --- Parameters ---
ALPHA_PT_LIST = [0.01, 0.5, 1.0]
alphas_ft = np.linspace(0.01, 2.0, 80)

# Fixed setting
RHO_PT       = 0.10
RHO_FT       = 0.10
OMEGA        = 1.0   # full overlap
C_PT         = 1e-3
LAMBDA_PT    = 0.0
GAMMA_REINIT = 0.0
A_PT         = 1.0
SIGMA0_PT    = 0.0
GAMMA_EXT    = 1e-6
SIGMA0_2     = 0.0
MC           = 80_000
SEED         = 0
TOL          = 1e-6
MAX_ITERS    = 900
DAMP         = 0.25

# --- Run ---
records = []
for alpha_pt in ALPHA_PT_LIST:
    print(f"Running alpha_pt={alpha_pt}...", flush=True)
    curve, reliability, info = rip.ptft_qk_curve_imperfect_pt(
        rho_pt=RHO_PT, rho_ft=RHO_FT, omega=OMEGA,
        alpha_pt=alpha_pt,
        sigma0_pt=SIGMA0_PT,
        gamma_ext=GAMMA_EXT, sigma0_2=SIGMA0_2,
        alphas=alphas_ft,
        mc=MC, seed=SEED,
        a_pt=A_PT, c_pt=C_PT, lambda_pt=LAMBDA_PT, gamma_reinit=GAMMA_REINIT,
        tol=TOL, max_iters=MAX_ITERS, damp=DAMP,
    )
    n = len(curve["alpha"])
    df_run = pd.DataFrame({
        "alpha_pt":             np.full(n, alpha_pt),
        "alpha":                curve["alpha"],
        "mse_best":             curve["mse_best"],
        "mse_fwd":              curve["mse_fwd"],
        "mse_bwd":              curve["mse_bwd"],
        "diff_db":              curve["diff_db"],
        "fp_residual":          curve["fp_residual"],
        "mse_se":               curve["mse_se"],
        "mse_rel_se":           curve["mse_rel_se"],
        "mse_se_db":            curve["mse_se_db"],
        # metadata
        "rho_pt":               RHO_PT,
        "rho_ft":               RHO_FT,
        "omega":                OMEGA,
        "c_pt":                 C_PT,
        "lambda_pt":            LAMBDA_PT,
        "gamma_reinit":         GAMMA_REINIT,
        "reliability_score_db": reliability["score_db"],
        # PT diagnostics
        "s2_pt":                info.get("s2_pt", float("nan")),
        "gp_pt":                info.get("gp_pt", float("nan")),
        "pt_oracle":            info.get("oracle", True),
    })
    records.append(df_run)

df = pd.concat(records, ignore_index=True)
out = Path(__file__).parent / "replica_imperfect_pt_omega1_rho0p1.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")
