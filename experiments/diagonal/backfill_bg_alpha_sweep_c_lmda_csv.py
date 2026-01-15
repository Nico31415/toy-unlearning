#!/usr/bin/env python3
"""
Backfill missing rows for experiment_results_bg_alpha_sweep_c_lmda.csv.

Why this exists:
  The main sweep uses a locked CSV append. Under high concurrency, a small
  number of tasks can fail to acquire the lock within the retry budget and
  therefore skip appending, even though their results_meta.json exists.

This script:
  - Loads the existing CSV
  - Computes the expected (alpha, seed, c, lmda) grid (same as the sweep)
  - Finds missing combos
  - For each missing combo, locates results_meta.json in the corresponding save_folder
  - Appends the missing row(s) into the CSV

Usage:
  python experiments/diagonal/backfill_bg_alpha_sweep_c_lmda_csv.py
"""

import json
import os
import random
import time
import fcntl
import sys
from pathlib import Path

import pandas as pd

# Import the sweep ArgparseArray so we use exactly the same mapping.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from experiments.diagonal.diagonal_bg_alpha_sweep_c_lmda_1 import argparse_array


def safe_csv_append(csv_path: str, new_row_data: dict, max_retries: int = 200, base_delay: float = 0.2) -> bool:
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
            delay = base_delay * (2 ** min(attempt, 8)) + random.uniform(0, 0.1)
            time.sleep(delay)
            continue
        except Exception as e:
            print(f"ERROR in safe_csv_append: {e}")
            return False
    return False


def expected_grid():
    alpha_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    seeds = [0, 1]
    cs = [0.001, 0.5]
    fracs = [-0.95, 0.0, 0.95]
    expected = set()
    for a in alpha_values:
        for s in seeds:
            for c in cs:
                for f in fracs:
                    expected.add((float(a), int(s), float(c), float(c * f)))
    return expected


def observed_grid(df: pd.DataFrame):
    def _round(v):
        # match the rounding used in the earlier audit (tolerant to float text formatting)
        return round(float(v), 12)

    obs = set()
    for r in df.itertuples(index=False):
        obs.add((float(r.alpha), int(r.seed), float(r.c), _round(r.lmda)))
    return obs


def find_array_id_for(target):
    a_target, s_target, c_target, lmda_target = target
    for array_id in range(132):
        args = argparse_array.get_args(array_id)
        alpha = float(args["n_train"]) / float(args["inp_dim"])
        seed = int(args["seed"])
        c = float(args["c"])
        lmda = round(float(args["lmda"]), 12)
        if (float(alpha), seed, float(c), lmda) == (a_target, s_target, c_target, round(lmda_target, 12)):
            return array_id, args
    return None, None


def build_row_from_meta(save_folder: str, alpha: float, n_train: int, seed: int, c: float, lmda: float):
    meta_path = os.path.join(save_folder, "results_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return {
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


def main():
    csv_path = os.path.abspath("experiment_results_bg_alpha_sweep_c_lmda.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    exp = expected_grid()
    obs = observed_grid(df)
    missing = sorted(exp - obs)

    print(f"CSV: {csv_path}")
    print(f"Expected combos: {len(exp)}")
    print(f"Observed combos: {len(obs)} (rows={len(df)})")
    print(f"Missing combos:  {len(missing)}")

    if not missing:
        print("Nothing to backfill.")
        return

    for t in missing:
        array_id, args = find_array_id_for(t)
        if array_id is None:
            print(f"Could not map missing combo to array_id: {t}")
            continue

        save_folder = args["save_folder"]
        alpha = float(args["n_train"]) / float(args["inp_dim"])
        n_train = int(args["n_train"])
        seed = int(args["seed"])
        c = float(args["c"])
        lmda = float(args["lmda"])

        row = build_row_from_meta(save_folder, alpha, n_train, seed, c, lmda)
        ok = safe_csv_append(csv_path, row)
        print(f"Backfill array_id={array_id} combo={t} -> {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()


