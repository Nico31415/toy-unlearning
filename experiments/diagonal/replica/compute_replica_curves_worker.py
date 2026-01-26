import argparse
import itertools
import numpy as np
import pandas as pd
import ptft_replica_qk as rq
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True, help="Slurm array task ID (0-11)")
    parser.add_argument("--output-dir", type=str, default="results/replica_ptft_parallel", help="Directory to save results")
    args = parser.parse_args()

    # Define the same grid as in compute_replica_curves.py
    alphas = np.linspace(0.01, 0.5, 40)
    seeds = [0]
    lmdas = [-0.00099, 0.0, 0.00099]
    cs = [1e-3]
    omegas = [0.0, 1.0]
    gamma_reinits = [0.0, 1.0]
    rho_pts = [0.10]
    rho_fts = [0.10]

    # Generate the Cartesian product of parameters
    # The order here should be consistent
    param_combinations = list(itertools.product(
        rho_pts, rho_fts, omegas, cs, lmdas, gamma_reinits, seeds
    ))

    if args.task_id >= len(param_combinations):
        print(f"Error: Task ID {args.task_id} is out of range (max {len(param_combinations)-1})")
        return

    # Select the specific combination for this task
    rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit, seed = param_combinations[args.task_id]

    print(f"Running Task {args.task_id}:")
    print(f"  omega={omega}, lambda_pt={lambda_pt}, gamma_reinit={gamma_reinit}")

    # Run the replica computation for this single combination
    df = rq.build_ptft_curves_dataframe(
        rho_pt=[rho_pt],
        rho_ft=[rho_ft],
        omega=[omega],
        c_pt=[c_pt],
        lambda_pt=[lambda_pt],
        gamma_reinit=[gamma_reinit],
        a_pt=1.0,
        alphas=alphas,
        mc=80_000,
        seed=[seed],
        gamma_ext=1e-6,
        tol=1e-6,
        max_iters=900,
        damp=0.25,
    )

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"replica_ptft_omega{omega}_lambda{lambda_pt}_reinit{gamma_reinit}.csv"
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
