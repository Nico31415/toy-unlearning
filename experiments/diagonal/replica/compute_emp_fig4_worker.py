import argparse
import itertools
import numpy as np
import pandas as pd
import ptft_empirical_finetune_df as emp
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True, help="Slurm array task ID")
    parser.add_argument("--output-dir", type=str, default="results/emp_fig4_new", help="Directory to save results")
    args = parser.parse_args()

    # Define the grid for Figure 4 missing data (Regime I)
    # alphas = np.linspace(0.01, 0.5, 11)
    alphas = np.linspace(0.01, 0.5, 11)
    seeds = list(range(10)) # 10 seeds for better statistics
    c_pt = 1e-3
    lambda_pt = -0.999 * c_pt
    gamma_reinit = 0.0
    
    # Combinations we need for Regime I:
    # 1. w=0, rf=0.1
    # 2. w=1, rf=0.1
    # 3. w=1, rf=0.01
    curve_configs = [
        (0.0, 0.1),
        (1.0, 0.1),
        (1.0, 0.01)
    ]

    # Generate the Cartesian product: (omega, rho_ft) x seed x alpha
    param_combinations = list(itertools.product(
        curve_configs, seeds, alphas
    ))

    if args.task_id >= len(param_combinations):
        print(f"Error: Task ID {args.task_id} is out of range (max {len(param_combinations)-1})")
        return

    # Select the specific combination for this task
    (omega, rho_ft), seed, alpha = param_combinations[args.task_id]

    print(f"Running Task {args.task_id}:")
    print(f" omega={omega}, rho_ft={rho_ft}, lambda_pt={lambda_pt}, alpha={alpha}, seed={seed}")

    # Run the empirical computation
    df = emp.build_ptft_finetune_curves_dataframe(
        rho_pt=[0.10],
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
        lr=0.05,
        epochs=5_000_000,
        test_every_n_epochs=5000,
        log_every_n_epochs=50000,
        no_tuning=True,
        threshold=1e-5,
    )

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"emp_fig4_omega{omega}_rhoft{rho_ft}_lambda{lambda_pt}_alpha{alpha:.4f}_seed{seed}.csv"
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
