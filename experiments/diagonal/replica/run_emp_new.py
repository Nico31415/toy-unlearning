"""
Run missing empirical experiments and append to emp_imperfect_pt_quick.csv:
  - sigma0=0.01, alpha_pt=0.95, seeds 0-9, omegas {1.0, 0.0}  (60 runs)
  - sigma0=0.5,  alpha_pt=0.95, seeds 5-9, omegas {1.0, 0.0}  (30 runs)
"""
import sys
from pathlib import Path
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[2]))

from compute_emp_imperfect_pt import run_one

ALPHA_FT_LIST = [0.05, 0.1, 0.15]
OUT_CSV = _HERE / "emp_imperfect_pt_quick.csv"

rows = []

# ── sigma0=0.01, alpha_pt=0.95, seeds 0-9 ─────────────────────────────────
for omega in [1.0, 0.0]:
    for seed in range(10):
        for alpha_ft in ALPHA_FT_LIST:
            label = f"s0=0.01 apt=0.95 omega={omega} aft={alpha_ft} seed={seed}"
            print(label, flush=True)
            row = run_one(
                pt_mode="noisy",
                alpha_ft=alpha_ft,
                alpha_pt=0.95,
                sigma0_pt=0.01,
                seed=seed,
                omega=omega,
                # Standard stopping — network interpolates noisy labels (sigma0=0.01 << noise floor)
                # threshold=1e-4 triggers when train_pred_mse < 1e-4
            )
            print(f"  pt_mse={row['pt_param_mse']:.4e} ft_mse={row['ft_param_mse']:.4e} "
                  f"pt_stop={row['pt_stop']} ft_stop={row['ft_stop']}", flush=True)
            rows.append(row)
            # Write incrementally so partial results are available
            existing = pd.read_csv(OUT_CSV)
            combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            combined.to_csv(OUT_CSV, index=False)

# ── sigma0=0.5, alpha_pt=0.95, seeds 5-9 ──────────────────────────────────
for omega in [1.0, 0.0]:
    for seed in range(5, 10):
        for alpha_ft in ALPHA_FT_LIST:
            label = f"s0=0.5 apt=0.95 omega={omega} aft={alpha_ft} seed={seed}"
            print(label, flush=True)
            row = run_one(
                pt_mode="noisy",
                alpha_ft=alpha_ft,
                alpha_pt=0.95,
                sigma0_pt=0.5,
                seed=seed,
                omega=omega,
                # Grad-norm stopping (pred_mse gate open): PT interpolates noisy labels,
                # gradient converges. Same parameters as seeds 0-4.
                stop_grad_norm=1e-3,
                stop_pred_mse=100.0,
            )
            print(f"  pt_mse={row['pt_param_mse']:.4e} ft_mse={row['ft_param_mse']:.4e} "
                  f"pt_stop={row['pt_stop']} ft_stop={row['ft_stop']}", flush=True)
            rows.append(row)
            # Write incrementally
            existing = pd.read_csv(OUT_CSV)
            combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            combined.to_csv(OUT_CSV, index=False)

print(f"\nDone. Final CSV: {OUT_CSV}")
df_final = pd.read_csv(OUT_CSV)
print(df_final[["pt_mode", "sigma0_pt", "alpha_pt", "omega"]].value_counts().to_string())
