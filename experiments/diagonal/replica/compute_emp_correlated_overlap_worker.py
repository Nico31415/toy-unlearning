"""
Empirical PT+FT sweep for partially correlated overlap teachers.

Teacher convention on overlap:
  beta_FT = q * beta_PT + sqrt(1 - q^2) * noise

Runs three regimes with Regime IV fixed to lambda_pt = -0.99 * c_pt.
"""
import argparse
import fcntl
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import ptft_empirical_finetune_df as emp


REGIMES = [
    ("regime_II", 1e-3, 0.0, 0.0),
    ("regime_III", 1e-3, 0.0, 10.0),
    ("regime_IV", 1e-3, -0.99e-3, 0.0),
]

QS = [0.25, 0.50, 0.75]
ALPHAS = list(np.linspace(0.01, 0.8, 11))
SEEDS = list(range(5))


def _q_name(q: float) -> str:
    return f"correlated_overlap_q{q:.2f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="results/sanity_check_correlated_overlap_q_sweep")
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    param_combinations = [
        (regime_name, c_pt, lambda_pt, gamma_reinit, q, seed, alpha)
        for (regime_name, c_pt, lambda_pt, gamma_reinit), q, seed, alpha
        in itertools.product(REGIMES, QS, SEEDS, ALPHAS)
    ]

    total_tasks = len(param_combinations)
    if args.list:
        print(f"total_tasks={total_tasks}")
        print(f"array_range=0-{total_tasks - 1}")
        return

    task_id = int(args.task_id)
    if task_id < 0 or task_id >= total_tasks:
        raise SystemExit(f"ERROR: task_id {task_id} out of range 0..{total_tasks - 1}")

    regime_name, c_pt, lambda_pt, gamma_reinit, q, seed, alpha = param_combinations[task_id]
    teacher_norm = _q_name(q)

    print(f"Total tasks: {total_tasks}")
    print(f"Running task {task_id}/{total_tasks - 1}")
    print(
        f"regime={regime_name} | teacher={teacher_norm} | seed={seed} | "
        f"alpha={alpha:.4f} | omega={args.omega:g} | lambda_pt={lambda_pt:g}"
    )

    save_folder = str(
        Path(args.output_dir) / regime_name / teacher_norm / f"seed{seed}_alpha{alpha:.4f}"
    )

    row = emp.run_one(
        setting="ptft",
        seed=int(seed),
        inp_dim=5000,
        alpha=float(alpha),
        n_test=10_000,
        c_pt=float(c_pt),
        lambda_pt=float(lambda_pt),
        gamma_reinit=float(gamma_reinit),
        rho_pt=0.1,
        a_pt=1.0,
        rho_ft=0.1,
        omega=float(args.omega),
        ft_teacher_norm=teacher_norm,
        lr=0.5,
        epochs=5_000_000,
        test_every_n_epochs=5_000,
        log_every_n_epochs=50_000,
        no_tuning=True,
        threshold=1e-4,
        stop_pred_mse=None,
        stop_beta_rate=0.0,
        stop_grad_norm=0.0,
        lr_decay=1.0,
        lr_decay_interval=2000,
        save_folder=save_folder,
    )
    row["regime"] = regime_name
    row["teacher_norm"] = teacher_norm
    row["overlap_q"] = float(q)
    row["experiment"] = "correlated_overlap_q_sweep"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_csv = out_dir / "sanity_check_results.csv"
    lock_path = master_csv.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        header = not master_csv.exists()
        pd.DataFrame([row]).to_csv(master_csv, mode="a", header=header, index=False)
        fcntl.flock(lf, fcntl.LOCK_UN)

    print(f"Saved weights to: {save_folder}")
    print(f"Appended row to:  {master_csv}")


if __name__ == "__main__":
    main()
