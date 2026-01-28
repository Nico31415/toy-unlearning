import argparse
import itertools
import numpy as np
import pandas as pd
import ptft_replica_qk as rq
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True, help="Slurm array task ID")
    parser.add_argument("--output-dir", type=str, default="results/replica_fig4_new", help="Directory to save results")
    args = parser.parse_args()

    # Define the grid for Figure 4 missing data
    alphas = np.linspace(0.01, 0.5, 40)
    seeds = [0]
    c_pt = 1e-3
    
    # We need lambda_pt = -0.999 * c_pt for Regime I
    # And we want omega=0, rho_ft=0.1; omega=1, rho_ft=0.1; omega=1, rho_ft=0.01
    lmdas = [-0.999 * c_pt]
    gamma_reinits = [0.0]
    omegas = [0.0, 1.0]
    rho_pts = [0.10]
    rho_fts = [0.1, 0.01]

    # Generate the Cartesian product of parameters
    param_combinations = list(itertools.product(
        rho_pts, rho_fts, omegas, [c_pt], lmdas, gamma_reinits, seeds
    ))
    
    # Filter to only the 3 curves we need:
    # 1. w=0, rf=0.1
    # 2. w=1, rf=0.1
    # 3. w=1, rf=0.01
    valid_combinations = []
    for p in param_combinations:
        rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit, seed = p
        if (omega == 0.0 and np.isclose(rho_ft, 0.1)) or \
           (omega == 1.0 and np.isclose(rho_ft, 0.1)) or \
           (omega == 1.0 and np.isclose(rho_ft, 0.01)):
            valid_combinations.append(p)

    if args.task_id >= len(valid_combinations):
        print(f"Error: Task ID {args.task_id} is out of range (max {len(valid_combinations)-1})")
        return

    # Select the specific combination for this task
    rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit, seed = valid_combinations[args.task_id]

    print(f"Running Task {args.task_id}:")
    print(f"  omega={omega}, rho_ft={rho_ft}, lambda_pt={lambda_pt}, gamma_reinit={gamma_reinit}")

    # Run the replica computation
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
    
    filename = f"replica_fig4_omega{omega}_rhoft{rho_ft}_lambda{lambda_pt}_reinit{gamma_reinit}.csv"
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
