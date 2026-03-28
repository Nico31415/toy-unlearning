"""
Re-run sigma0_pt=0.5 experiments with alpha_pt=2.0 (n_train_pt = 2*d = 2000).

At alpha_pt=1 the system is at the Marchenko-Pastur critical point: X is
nearly singular, interpolating solution X^{-1}y has pt_param_mse -> inf.

At alpha_pt=2 the system is well-conditioned (eigenvalues of X^TX bounded away
from 0), and the OLS/sparse estimator achieves pt_param_mse -> sigma0^2 = 0.25,
matching the AWGN bypass formula. The theory curve is unchanged (bypass uses
s2_pt = sigma0_pt^2 for any alpha_pt >= 1).

Replaces sigma0_pt=0.5 rows in emp_imperfect_pt_quick.csv.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DIAG_DIR = _HERE.parent
_REPO_ROOT = _HERE.parents[2]
for _p in (_REPO_ROOT, _DIAG_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
from compute_emp_imperfect_pt import run_one

ALPHA_FT_LIST = [0.05, 0.1, 0.15]
N_SEEDS = 5
OUT_CSV = _HERE / "emp_imperfect_pt_quick.csv"

all_rows = []
for omega in [1.0, 0.0]:
    for seed in range(N_SEEDS):
        for alpha_ft in ALPHA_FT_LIST:
            print(f"  noisy sigma0=0.5 alpha_pt=2, omega={omega}, alpha_ft={alpha_ft}, seed={seed}", flush=True)
            row = run_one(
                pt_mode="noisy",
                alpha_ft=alpha_ft,
                alpha_pt=2.0,          # n_train_pt = 2*d = 2000
                sigma0_pt=0.5,
                seed=seed,
                omega=omega,
                stop_grad_norm=1e-3,   # gradient-norm stopping (valid at alpha_pt=2)
                stop_pred_mse=100.0,   # open the pred_mse gate
            )
            print(f"    pt_param_mse={row['pt_param_mse']:.4e}  "
                  f"ft_param_mse={row['ft_param_mse']:.4e}  "
                  f"pt_stop={row['pt_stop']}  pt_epoch={row['pt_epoch']}  "
                  f"wall={row['wall_s']:.1f}s", flush=True)
            all_rows.append(row)

new_df = pd.DataFrame(all_rows)

# Replace sigma0_pt=0.5 rows in CSV
existing = pd.read_csv(OUT_CSV)
keep = existing[~(existing["sigma0_pt"].round(2) == 0.5)]
combined = pd.concat([keep, new_df], ignore_index=True)
combined.to_csv(OUT_CSV, index=False)
print(f"\nReplaced {len(new_df)} sigma0=0.5 rows -> {OUT_CSV}  (total {len(combined)})")
print(new_df[["omega", "alpha_ft_req", "seed", "pt_param_mse", "pt_stop", "pt_epoch", "ft_param_mse"]].to_string(index=False))
print(f"\nMean pt_param_mse (omega=1): {new_df[new_df['omega']==1.0]['pt_param_mse'].mean():.4f}")
print(f"Mean pt_param_mse (omega=0): {new_df[new_df['omega']==0.0]['pt_param_mse'].mean():.4f}")
print(f"Expected sigma0^2 = {0.5**2:.4f}")
