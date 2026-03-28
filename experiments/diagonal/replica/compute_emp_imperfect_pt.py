#!/usr/bin/env python3
"""
SLURM-array worker for imperfect-PT replica experiments: varying alpha_pt.

Each task handles one (config, alpha_pt) combination and computes the full
replica curve over alpha_ft.

All tasks append to a single shared CSV (with file locking for parallel safety).

Grid (27 tasks total):
  configs:  9 PT configs (two setups, see below)
  alpha_pt: [0.01, 0.2, 0.5]

Setup 1: rho_pt=0.1, rho_ft=0.1, omega in {0, 1}, lambda_pt sweep
Setup 2: rho_pt=0.1, rho_ft=0.01, omega=1, lambda_pt sweep
Both with c_pt=1e-3, gamma_reinit=0.

Example launch (SLURM):
  sbatch --array=0-26 submit_replica_imperfect_pt_alpha_pt_sweep.sh
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

# --- sys.path setup ---
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

ALPHA_PT_LIST = [0.01, 0.2, 0.5]

# Full cartesian product: config × alpha_pt
PARAM_GRID = list(itertools.product(CONFIGS, ALPHA_PT_LIST))


def _print_info():
    print(f"Total tasks: {len(PARAM_GRID)}")
    print(f"  configs:   {len(CONFIGS)}")
    print(f"  alpha_pt:  {ALPHA_PT_LIST}")
    for i, (cfg, a_pt) in enumerate(PARAM_GRID[:8]):
        print(
            f"  task {i:3d}: rho_ft={cfg['rho_ft']}, omega={cfg['omega']}, "
            f"lambda_pt={cfg['lambda_pt']:.5g}, alpha_pt={a_pt}"
        )
    print("  ...")


def _safe_append_df(csv_path: Path, new_df: pd.DataFrame, key_cols: list) -> None:
    """Append rows to shared CSV under a file lock; skip rows whose key already exists."""
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

                if df.empty:
                    out = new_df.copy()
                else:
                    df_key = df[key_cols].astype(str).agg("|".join, axis=1) if all(k in df.columns for k in key_cols) else pd.Series([], dtype=str)
                    new_key = new_df[key_cols].astype(str).agg("|".join, axis=1)
                    keep_mask = ~new_key.isin(set(df_key.tolist()))
                    n_skip = int((~keep_mask).sum())
                    if n_skip > 0:
                        print(f"  Skipping {n_skip} existing rows.", flush=True)
                    out = pd.concat([df, new_df.loc[keep_mask]], ignore_index=True)

                out.to_csv(csv_path, index=False)
                return

        except Exception:
            if attempt == 59:
                raise
            time.sleep(0.1 * (2 ** min(attempt, 6)) + random.uniform(0, 0.05))


def _coerce_curve_to_df(curve, config, alpha_pt):
    """
    Convert the output of ptft_qk_curve_imperfect_pt(...) into a DataFrame.

    Expected common cases:
      1) pandas.DataFrame already
      2) dict of arrays/lists
    """
    if isinstance(curve, pd.DataFrame):
        df = curve.copy()
    elif isinstance(curve, dict):
        df = pd.DataFrame(curve)
    else:
        raise TypeError(f"Unsupported curve return type: {type(curve)}")

    # Add metadata columns if missing
    df["rho_pt"] = config["rho_pt"]
    df["rho_ft"] = config["rho_ft"]
    df["omega"] = config["omega"]
    df["c_pt"] = config["c_pt"]
    df["lambda_pt"] = config["lambda_pt"]
    df["gamma_reinit"] = config["gamma_reinit"]
    df["alpha_pt"] = alpha_pt

    # Some scripts use different names; normalize if possible
    rename_map = {}
    if "alpha" in df.columns and "alpha_ft" not in df.columns:
        rename_map["alpha"] = "alpha_ft"
    if "mse_ft" in df.columns and "ft_param_mse" not in df.columns:
        rename_map["mse_ft"] = "ft_param_mse"
    if "mse_pt" in df.columns and "pt_param_mse" not in df.columns:
        rename_map["mse_pt"] = "pt_param_mse"
    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Imperfect-PT replica worker: varying alpha_pt"
    )
    parser.add_argument("--task-id", type=int, default=None, help="SLURM array task ID")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/replica_imperfect_pt_alpha_pt_sweep",
        help="Directory for the shared output CSV",
    )
    parser.add_argument("--info", action="store_true", help="Print grid info and exit")
    args = parser.parse_args()

    if args.info:
        _print_info()
        return

    if args.task_id is None:
        parser.error("--task-id is required (use --info to see grid size)")

    if args.task_id >= len(PARAM_GRID):
        print(f"Error: task-id {args.task_id} out of range (max {len(PARAM_GRID)-1})")
        return

    config, alpha_pt = PARAM_GRID[args.task_id]

    print(
        f"Task {args.task_id}/{len(PARAM_GRID)-1}: "
        f"rho_ft={config['rho_ft']}, omega={config['omega']}, "
        f"lambda_pt={config['lambda_pt']:.5g}, alpha_pt={alpha_pt}",
        flush=True,
    )

    curve = rep.ptft_qk_curve_imperfect_pt(
        alpha_pt=alpha_pt,
        sigma0_pt=0.0,
        **config,
    )

    df = _coerce_curve_to_df(curve, config, alpha_pt)

    if "alpha_ft" in df.columns:
        print(
            f"  Computed curve with {len(df)} points "
            f"(alpha_ft from {df['alpha_ft'].min():.4f} to {df['alpha_ft'].max():.4f})",
            flush=True,
        )
    else:
        print(f"  Computed curve with {len(df)} rows", flush=True)

    out_dir = Path(args.output_dir)
    csv_path = out_dir / "replica_imperfect_pt_alpha_pt_sweep.csv"
    key_cols = [
        "rho_pt", "rho_ft", "omega", "c_pt", "lambda_pt", "gamma_reinit",
        "alpha_pt", "alpha_ft",
    ]
    _safe_append_df(csv_path, df, key_cols)
    print(f"  Appended → {csv_path}", flush=True)


if __name__ == "__main__":
    main()