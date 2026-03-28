"""
SLURM-array worker for imperfect-PT empirical experiments: varying alpha_pt (noiseless).

Each task handles one (config, alpha_pt, alpha_ft, seed) combination.
All tasks append to a single shared CSV (with file locking for parallel safety).

Grid (2772 tasks total):
  configs:  9 PT configs (two setups, see below)
  alpha_pt: [0.2, 0.5]  (noiseless / underdetermined)
  alpha_ft: 11 points in [0.01, 0.5]
  seeds:    0..13

Setup 1: rho_pt=0.1, rho_ft=0.1, omega in {0, 1}, lambda_pt sweep
Setup 2: rho_pt=0.1, rho_ft=0.01, omega=1, lambda_pt sweep
Both with c_pt=1e-3, gamma_reinit=0.

Example launch (SLURM):
  sbatch --array=0-2771 submit_emp_imperfect_pt_alpha_pt_sweep.sh

Or locally in parallel (e.g. with GNU parallel):
  seq 0 2771 | parallel -j 8 python compute_emp_imperfect_pt_alpha_pt_sweep_worker.py --task-id {}
"""

import argparse
import fcntl
import itertools
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- sys.path setup (same as compute_emp_imperfect_pt.py) ---
_HERE      = Path(__file__).resolve().parent
_DIAG_DIR  = _HERE.parent
_REPO_ROOT = _HERE.parents[2]
for _p in (_REPO_ROOT, _DIAG_DIR, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ptft_replica_imperfect_pt as rep


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

_C_PT = 1e-3

# 9 unique PT configs
CONFIGS = [
    # Setup 1: rho_pt=0.1, rho_ft=0.1, omega in {0, 1}, lambda_pt sweep
    dict(rho_pt=0.1, rho_ft=0.1,  omega=0.0, c_pt=_C_PT, lambda_pt=-0.99 * _C_PT, gamma_reinit=0.0),
    dict(rho_pt=0.1, rho_ft=0.1,  omega=0.0, c_pt=_C_PT, lambda_pt=0.0,            gamma_reinit=0.0),
    dict(rho_pt=0.1, rho_ft=0.1,  omega=0.0, c_pt=_C_PT, lambda_pt= 0.99 * _C_PT, gamma_reinit=0.0),
    dict(rho_pt=0.1, rho_ft=0.1,  omega=1.0, c_pt=_C_PT, lambda_pt=-0.99 * _C_PT, gamma_reinit=0.0),
    dict(rho_pt=0.1, rho_ft=0.1,  omega=1.0, c_pt=_C_PT, lambda_pt=0.0,            gamma_reinit=0.0),
    dict(rho_pt=0.1, rho_ft=0.1,  omega=1.0, c_pt=_C_PT, lambda_pt= 0.99 * _C_PT, gamma_reinit=0.0),
    # Setup 2: rho_pt=0.1, rho_ft=0.01, omega=1, lambda_pt sweep
    dict(rho_pt=0.1, rho_ft=0.01, omega=1.0, c_pt=_C_PT, lambda_pt=-0.99 * _C_PT, gamma_reinit=0.0),
    dict(rho_pt=0.1, rho_ft=0.01, omega=1.0, c_pt=_C_PT, lambda_pt=0.0,            gamma_reinit=0.0),
    dict(rho_pt=0.1, rho_ft=0.01, omega=1.0, c_pt=_C_PT, lambda_pt= 0.99 * _C_PT, gamma_reinit=0.0),
]

ALPHA_PT_LIST = [0.01]
ALPHA_FT_LIST = np.linspace(0.01, 0.5, 11).tolist()
SEEDS         = list(range(14))

# Full cartesian product: config × alpha_pt × alpha_ft × seed
PARAM_GRID = list(itertools.product(CONFIGS, ALPHA_PT_LIST, ALPHA_FT_LIST, SEEDS))


def _print_info():
    print(f"Total tasks: {len(PARAM_GRID)}")
    print(f"  configs:   {len(CONFIGS)}")
    print(f"  alpha_pt:  {ALPHA_PT_LIST}")
    print(f"  alpha_ft:  {len(ALPHA_FT_LIST)} pts in [{ALPHA_FT_LIST[0]:.3f}, {ALPHA_FT_LIST[-1]:.3f}]")
    print(f"  seeds:     0..{SEEDS[-1]}")
    for i, (cfg, a_pt, a_ft, seed) in enumerate(PARAM_GRID[:5]):
        print(f"  task {i:4d}: rho_ft={cfg['rho_ft']}, omega={cfg['omega']}, "
              f"lambda_pt={cfg['lambda_pt']:.5g}, alpha_pt={a_pt}, "
              f"alpha_ft={a_ft:.4f}, seed={seed}")
    print("  ...")


def _safe_append_row(csv_path: Path, row: dict, key_cols: list) -> None:
    """Append row to shared CSV under a file lock; skip if key already exists."""
    csv_path = Path(csv_path)
    lock_path = Path(str(csv_path) + ".lock")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(60):
        try:
            with open(lock_path, "w") as lock_f:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                    except Exception:
                        df = pd.DataFrame()
                else:
                    df = pd.DataFrame()

                if not df.empty:
                    mask = np.ones(len(df), dtype=bool)
                    for k in key_cols:
                        if k in df.columns:
                            mask &= (df[k].astype(str) == str(row.get(k)))
                    if bool(mask.any()):
                        print("  Row already exists, skipping.", flush=True)
                        return

                new_row_df = pd.DataFrame([row])
                all_cols = sorted(set(df.columns.tolist() + new_row_df.columns.tolist()))
                out = pd.concat(
                    [df.reindex(columns=all_cols), new_row_df.reindex(columns=all_cols)],
                    ignore_index=True,
                )
                out.to_csv(csv_path, index=False)
                return
        except Exception:
            if attempt == 59:
                raise
            time.sleep(0.1 * (2 ** min(attempt, 6)) + random.uniform(0, 0.05))


def main():
    parser = argparse.ArgumentParser(
        description="Imperfect-PT empirical worker: varying alpha_pt (noiseless)"
    )
    parser.add_argument("--task-id",    type=int,  default=None, help="SLURM array task ID")
    parser.add_argument("--output-dir", type=str,
                        default="results/emp_imperfect_pt_alpha_pt_sweep",
                        help="Directory for the shared output CSV")
    parser.add_argument("--info",       action="store_true", help="Print grid info and exit")
    # Override default training knobs if needed
    parser.add_argument("--inp-dim",    type=int,   default=1000)
    parser.add_argument("--n-test",     type=int,   default=10_000)
    parser.add_argument("--lr",         type=float, default=0.5)
    parser.add_argument("--epochs",     type=int,   default=5_000_000)
    parser.add_argument("--threshold",  type=float, default=1e-12)
    args = parser.parse_args()

    if args.info:
        _print_info()
        return

    if args.task_id is None:
        parser.error("--task-id is required (use --info to see grid size)")

    if args.task_id >= len(PARAM_GRID):
        print(f"Error: task-id {args.task_id} out of range (max {len(PARAM_GRID)-1})")
        return

    config, alpha_pt, alpha_ft, seed = PARAM_GRID[args.task_id]

    print(
        f"Task {args.task_id}/{len(PARAM_GRID)-1}: "
        f"rho_ft={config['rho_ft']}, omega={config['omega']}, "
        f"lambda_pt={config['lambda_pt']:.5g}, "
        f"alpha_pt={alpha_pt}, alpha_ft={alpha_ft:.4f}, seed={seed}",
        flush=True,
    )

    row = emp.run_one(
        pt_mode="underdetermined",
        alpha_ft=alpha_ft,
        alpha_pt=alpha_pt,
        sigma0_pt=0.0,
        seed=seed,
        inp_dim=args.inp_dim,
        n_test=args.n_test,
        lr=args.lr,
        epochs=args.epochs,
        threshold=args.threshold,
        **config,
    )

    print(
        f"  ft_param_mse={row['ft_param_mse']:.4e}  "
        f"pt_param_mse={row['pt_param_mse']:.4e}  "
        f"wall={row['wall_s']:.1f}s",
        flush=True,
    )

    out_dir = Path(args.output_dir)
    csv_path = out_dir / "emp_imperfect_pt_alpha_pt_sweep.csv"
    key_cols = [
        "rho_pt", "rho_ft", "omega", "c_pt", "lambda_pt", "gamma_reinit",
        "alpha_pt", "alpha_ft", "seed",
    ]
    _safe_append_row(csv_path, row, key_cols)
    print(f"  Appended → {csv_path}", flush=True)


if __name__ == "__main__":
    main()
