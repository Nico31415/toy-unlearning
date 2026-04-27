import argparse
import itertools
import numpy as np
import pandas as pd
import fcntl
from pathlib import Path

import ptft_empirical_finetune_df as emp


"""
FORGETTING EXPERIMENT:
Saves beta_pt.pt, beta_ft.pt, model.pt (+ beta0.npy, df.feather) for each run
so that per-group forgetting metrics can be computed post-hoc.

Two regimes:
  - regime_II:  lambda_pt=0.0,      c_pt=1e-3, gamma_reinit=0  (lazy, PT-dependent)
  - regime_IV:  lambda_pt=-0.99e-3, c_pt=1e-3, gamma_reinit=0  (rich, PT-dependent)

Fixed: rho_pt=rho_ft=0.1, omega=0.5, inp_dim=5000, a_pt=1.0
"""

REGIMES = [
    ("regime_II", 1e-3,  0.0,     0.0),   # (name, c_pt, lambda_pt, gamma_reinit)
    ("regime_IV", 1e-3, -0.99e-3, 0.0),
]

ALPHAS = list(np.linspace(0.01, 0.5, 11))
SEEDS  = list(range(10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id",    type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="results/forgetting")
    args = parser.parse_args()

    param_combinations = [
        (regime_name, c_pt, lambda_pt, gamma_reinit, seed, alpha)
        for (regime_name, c_pt, lambda_pt, gamma_reinit), seed, alpha
        in itertools.product(REGIMES, SEEDS, ALPHAS)
    ]

    total_tasks = len(param_combinations)
    if args.task_id < 0 or args.task_id >= total_tasks:
        print(f"Error: task_id {args.task_id} out of range (0..{total_tasks-1})")
        return

    regime_name, c_pt, lambda_pt, gamma_reinit, seed, alpha = param_combinations[args.task_id]

    print(f"Total tasks: {total_tasks}")
    print(f"Running task {args.task_id}/{total_tasks-1}")
    print(f"regime={regime_name} | c_pt={c_pt}, lambda_pt={lambda_pt}, "
          f"gamma_reinit={gamma_reinit}, seed={seed}, alpha={alpha:.4f}")

    save_folder = str(
        Path(args.output_dir) / regime_name / f"seed{seed}_alpha{alpha:.4f}"
    )

    row = emp.run_one(
        setting="ptft",
        seed=seed,
        inp_dim=5000,
        alpha=alpha,
        n_test=10_000,
        c_pt=c_pt,
        lambda_pt=lambda_pt,
        gamma_reinit=gamma_reinit,
        rho_pt=0.1,
        a_pt=1.0,
        rho_ft=0.1,
        omega=0.5,
        ft_teacher_norm="unit_total_var",
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
    row["experiment"] = "forgetting"

    # Append row to master CSV under a file lock
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_csv = out_dir / "forgetting_results.csv"
    df = pd.DataFrame([row])
    lock_path = master_csv.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        header = not master_csv.exists()
        df.to_csv(master_csv, mode="a", header=header, index=False)
        fcntl.flock(lf, fcntl.LOCK_UN)

    print(f"Saved weights to: {save_folder}")
    print(f"Appended row to: {master_csv}")


if __name__ == "__main__":
    main()
