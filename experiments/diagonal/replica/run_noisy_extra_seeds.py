"""
Run extra seeds (3-9) for the noisy PT case only (alpha_pt=1, sigma0_pt=0.01),
both omega=1 and omega=0, then append to emp_imperfect_pt_quick.csv.
"""
import sys
from pathlib import Path

_HERE      = Path(__file__).resolve().parent
_DIAG_DIR  = _HERE.parent
_REPO_ROOT = _HERE.parents[2]
for _p in (_REPO_ROOT, _DIAG_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
from compute_emp_imperfect_pt import sweep

EXTRA_SEEDS  = list(range(3, 10))   # seeds 3..9
ALPHA_FT_LIST = [0.05, 0.1, 0.15]
OUT_CSV = _HERE / "emp_imperfect_pt_quick.csv"

all_rows = []
for omega in [1.0, 0.0]:
    for seed in EXTRA_SEEDS:
        for alpha_ft in ALPHA_FT_LIST:
            from compute_emp_imperfect_pt import run_one
            print(f"  noisy, omega={omega}, alpha_ft={alpha_ft}, seed={seed}", flush=True)
            row = run_one(pt_mode="noisy", alpha_ft=alpha_ft,
                          sigma0_pt=0.01, seed=seed, omega=omega)
            print(f"    ft_param_mse={row['ft_param_mse']:.4e}  wall={row['wall_s']:.1f}s",
                  flush=True)
            all_rows.append(row)

import pandas as pd
existing = pd.read_csv(OUT_CSV)
new_df   = pd.DataFrame(all_rows)
combined = pd.concat([existing, new_df], ignore_index=True)
combined.to_csv(OUT_CSV, index=False)
print(f"\nAppended {len(new_df)} rows → {OUT_CSV}  (total {len(combined)})")
