"""
Re-run sigma0_pt=0.5 simulations with corrected PT stopping criterion.

Previous runs used stop_grad_norm=1e-3 which fired at epoch ~46k while the
network was still mid-interpolation (train_pred_mse=0.027, not near 0).

Fix: use pt_stop_pred_mse=0.001 — stops when train_pred_mse < 0.001,
i.e. the network has nearly interpolated the noisy training data.
This corresponds to the interpolating solution that the AWGN bypass assumes.

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
            print(f"  noisy sigma0=0.5, omega={omega}, alpha_ft={alpha_ft}, seed={seed}", flush=True)
            row = run_one(
                pt_mode="noisy",
                alpha_ft=alpha_ft,
                sigma0_pt=0.5,
                seed=seed,
                omega=omega,
                pt_stop_pred_mse=0.001,   # near-interpolation stopping
            )
            print(f"    pt_param_mse={row['pt_param_mse']:.4e}  "
                  f"ft_param_mse={row['ft_param_mse']:.4e}  "
                  f"pt_stop={row['pt_stop']}  pt_epoch={row['pt_epoch']}  "
                  f"wall={row['wall_s']:.1f}s", flush=True)
            all_rows.append(row)

new_df = pd.DataFrame(all_rows)

# Load existing CSV and replace sigma0_pt=0.5 rows
existing = pd.read_csv(OUT_CSV)
keep = existing[~(existing["sigma0_pt"].round(2) == 0.5)]
combined = pd.concat([keep, new_df], ignore_index=True)
combined.to_csv(OUT_CSV, index=False)
print(f"\nReplaced {len(new_df)} sigma0=0.5 rows in {OUT_CSV}  (total {len(combined)})")
print(new_df[["omega", "alpha_ft_req", "seed", "pt_param_mse", "pt_stop", "pt_epoch", "ft_param_mse"]].to_string(index=False))
print(f"\nMean pt_param_mse (omega=1): {new_df[new_df['omega']==1.0]['pt_param_mse'].mean():.4f}")
print(f"Mean pt_param_mse (omega=0): {new_df[new_df['omega']==0.0]['pt_param_mse'].mean():.4f}")
print(f"Expected sigma0^2 = 0.25")
