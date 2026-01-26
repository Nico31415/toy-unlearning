import argparse
import itertools
import numpy as np
import pandas as pd
import ptft_empirical_finetune_df as emp
from pathlib import Path
import os


# c = 1e-3, lmda_pt = 0.0, gamma = 0.0, w = 0.0 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True, help="Slurm array task ID (0-59)")
    parser.add_argument("--output-dir", type=str, default="results/emp_ptft_parallel", help="Directory to save results")
    args = parser.parse_args()

    # Define the same grid as in compute_emp_curves.py
    # alphas = np.linspace(0.01, 0.5, 11)
    # alphas = np.array([0.01 , 0.059, 0.108])
    alphas = np.linspace(0.2, 0.5, 8)
    seeds = [i for i in range(6, 20)]
    lmdas = [0.0]
    cs = [1e-3]
    omegas = [0.0]
    gamma_reinits = [0.0]
    rho_pts = [0.10]
    rho_fts = [0.10]

    # Generate the Cartesian product of parameters including alpha
    param_combinations = list(itertools.product(
        rho_pts, rho_fts, omegas, cs, lmdas, gamma_reinits, seeds, alphas
    ))

    if args.task_id >= len(param_combinations):
        print(f"Error: Task ID {args.task_id} is out of range (max {len(param_combinations)-1})")
        return

    # Select the specific combination for this task
    rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit, seed, alpha = param_combinations[args.task_id]

    print(f"Running Task {args.task_id}:")
    print(f" omega={omega}, lambda_pt={lambda_pt}, gamma_reinit={gamma_reinit}, alpha={alpha}")

    # Run the empirical computation for this single combination and single alpha
    df = emp.build_ptft_finetune_curves_dataframe(
        rho_pt=[rho_pt],
        rho_ft=[rho_ft],
        omega=[omega],
        a_pt=1.0,
        c_pt=[c_pt],
        lambda_pt=[lambda_pt],
        gamma_reinit=[gamma_reinit],
        alphas=[alpha],
        inp_dim=5000,
        n_test=10_000,
        seeds=[seed],
        lr=0.9,
        epochs=5_000_000,
        test_every_n_epochs=5000,
        log_every_n_epochs=50000,
        no_tuning=True,
        threshold=1e-5,
    )

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"TESTHIGHLR0.9emp_ptft_omega{omega}_lambda{lambda_pt}_reinit{gamma_reinit}_alpha{alpha:.4f}_seed{seed}.csv"
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
