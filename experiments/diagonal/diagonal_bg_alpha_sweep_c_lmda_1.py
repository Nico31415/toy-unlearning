#!/usr/bin/env python3
"""
Diagonal network BG alpha sweep (single task) varying c and lmda.

Requested experiment:
  - c in {0.001, 0.5}
  - lmda in {-0.95*c, 0, +0.95*c}
  - alpha in {0.05, 0.1, ..., 1.0}
  - 2 seeds per (alpha, c, lmda)

This script is designed for SLURM array execution:
  python experiments/diagonal/diagonal_bg_alpha_sweep_c_lmda_1.py <array_id>

It runs:
  experiments/diagonal/diagonal_network_pretrain_bg.py

and appends a single row per run to:
  experiment_results_bg_alpha_sweep_c_lmda.csv
"""

import argparse
import json
import os
import random
import time
import fcntl
import sys

import pandas as pd

sys.path.append("")
from functions.array_training import ArgparseArray


def safe_csv_append(
    csv_path: str,
    new_row_data: dict,
    max_retries: int = 50,
    base_delay: float = 0.2,
) -> bool:
    """Safely append a row to a CSV file with file locking."""
    lock_file_path = f"{csv_path}.lock"
    for attempt in range(max_retries):
        try:
            with open(lock_file_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                if os.path.exists(csv_path):
                    try:
                        existing_df = pd.read_csv(csv_path)
                    except Exception:
                        existing_df = pd.DataFrame()
                else:
                    existing_df = pd.DataFrame()

                new_df = pd.DataFrame([new_row_data])
                all_columns = sorted(set(list(existing_df.columns) + list(new_df.columns)))
                existing_df = existing_df.reindex(columns=all_columns)
                new_df = new_df.reindex(columns=all_columns)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined.to_csv(csv_path, index=False)
                return True
        except (BlockingIOError, OSError):
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                time.sleep(delay)
                continue
            return False
        except Exception:
            return False
    return False


# Alpha values: 0.05, 0.1, ..., 1.0
ALPHA_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
INP_DIM = 1000
N_TRAIN_VALUES = [int(alpha * INP_DIM) for alpha in ALPHA_VALUES]

# Requested sweep:
C_VALUES = [0.001, 0.5]
LMDA_FRACS = [-0.95, 0.0, 0.95]  # lmda = lmda_frac * c


argparse_array = ArgparseArray(
    n_train=N_TRAIN_VALUES,
    seed=[0, 1],
    inp_dim=[INP_DIM],
    rho=[0.04],
    c=C_VALUES,
    aux_lmda_frac=LMDA_FRACS,
    lmda=(lambda c, lmda_frac, **kwargs: f"{c * lmda_frac:.12g}"),
    lr=[0.5],
    threshold=[1e-12],
    epochs=[5000000],
    n_test=[10000],
    test_every_n_epochs=[200],
    scaling=[1.0],
    init_method=["complex"],
    no_tuning=[True],
    save_folder=(
        # NOTE: callable args are evaluated from array args only; lmda itself is a callable arg.
        # So we compute lmda on the fly from (c, lmda_frac) here to avoid dependency ordering.
        lambda n_train, seed, rho, c, lmda_frac, lr, inp_dim, **kwargs: (
            "results/diagonal/bg_experiments_c_lmda_sweep/"
            f"alpha={n_train/inp_dim:.6f}--n_train={n_train}--seed={seed}--"
            f"rho={rho:.6f}--c={c:.6f}--lmda={float(c * lmda_frac):.12g}--lr={lr:.1f}/"
        )
    ),
)


def extract_and_save_results(save_folder: str, alpha: float, n_train: int, seed: int, c: float, lmda: float) -> bool:
    meta_path = os.path.join(save_folder, "results_meta.json")
    if not os.path.exists(meta_path):
        print(f"WARNING: results_meta.json not found at {meta_path}")
        return False

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)

        row = {
            "alpha": float(alpha),
            "n_train": int(n_train),
            "seed": int(seed),
            "rho": 0.04,
            "c": float(c),
            "lmda": float(lmda),
            "test_pred_mse": meta.get("final_test_pred_mse", None),
            "param_mse": meta.get("final_param_mse", None),
            "train_pred_mse": meta.get("final_train_pred_mse", None),
            "final_epoch": meta.get("final_epoch", None),
            "stop_reason": meta.get("stop_reason", None),
            "final_grad_norm": meta.get("final_grad_norm", None),
            "final_beta_update_rate": meta.get("final_beta_update_rate", None),
            "save_folder": save_folder,
        }

        csv_path = os.path.abspath("experiment_results_bg_alpha_sweep_c_lmda.csv")
        success = safe_csv_append(csv_path, row)
        if success:
            print(f"Results appended to {csv_path}")
        else:
            print(f"ERROR: Failed to append results to {csv_path}")
        return success
    except Exception as e:
        print(f"ERROR: Failed to extract results from {meta_path}: {e}")
        return False


def main(args):
    import sys as _sys

    resolved_args = argparse_array.get_args(args.array_id)

    print("=" * 80)
    print("BG Alpha Sweep (varying c and lmda) - Experiment Parameters:")
    print("=" * 80)
    for key in sorted(resolved_args.keys()):
        if not key.startswith("aux_"):
            print(f"  {key}: {resolved_args[key]}")
    print("=" * 80)

    n_train = resolved_args["n_train"]
    seed = resolved_args["seed"]
    inp_dim = resolved_args["inp_dim"]
    c = float(resolved_args["c"])
    lmda = float(resolved_args["lmda"])
    save_folder = resolved_args["save_folder"]
    alpha = n_train / inp_dim

    print("\nRunning training script...")
    argparse_array.call_script(
        "experiments/diagonal/diagonal_network_pretrain_bg.py",
        args.array_id,
        python_cmd=_sys.executable,
    )

    print("\nExtracting results and saving to CSV...")
    extract_and_save_results(save_folder, alpha, n_train, seed, c, lmda)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("array_id", type=int, help="SLURM array task ID")
    main(parser.parse_args())


