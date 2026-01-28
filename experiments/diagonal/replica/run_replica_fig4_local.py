import itertools
import numpy as np
import pandas as pd
import ptft_replica_qk as rq
from pathlib import Path

def run_local():
    # Standard alpha grid for Figure 4
    alphas = np.linspace(0.01, 0.5, 40)
    seeds = [0]
    c_pt = 1e-3
    
    # Target parameters for Regime I (yellow) as requested:
    # lambda_pt = -c_pt, gamma_reinit = 1e-6
    lambda_pt_val = -c_pt
    gamma_reinit_val = 1e-6
    
    omegas = [0.0, 1.0]
    rho_pts = [0.10]
    rho_fts = [0.1, 0.01]

    # Generate the 3 specific combinations needed for the figure
    valid_combinations = [
        (0.10, 0.1,  0.0, c_pt, lambda_pt_val, gamma_reinit_val, 0), # no overlap, rho_ft=0.1
        (0.10, 0.1,  1.0, c_pt, lambda_pt_val, gamma_reinit_val, 0), # full overlap, rho_ft=0.1
        (0.10, 0.01, 1.0, c_pt, lambda_pt_val, gamma_reinit_val, 0), # full overlap, rho_ft=0.01
    ]

    all_dfs = []
    for i, p in enumerate(valid_combinations):
        rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit, seed = p
        print(f"Running combination {i+1}/{len(valid_combinations)}:")
        print(f"  omega={omega}, rho_ft={rho_ft}, lambda={lambda_pt}, gamma={gamma_reinit}")
        
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
        all_dfs.append(df)
    
    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        out_path = Path("results/replica_fig4_local.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
        print(f"\nDone! Results saved to: {out_path}")
        print("You can now merge this into your main replica CSV.")

if __name__ == "__main__":
    run_local()
