import itertools
import numpy as np
import pandas as pd
import ptft_replica_qk as rq
from pathlib import Path
import argparse

def run_local(
    *,
    c_pt: float = 1e-3,
    lambda_ratio: float = -0.99,
    allow_lambda_eq_minus_c: bool = False,
    out_csv: str = "results/replica_fig4_local.csv",
    alpha_min: float = 0.01,
    alpha_max: float = 0.5,
    n_alphas: int = 40,
    mc: int = 80_000,
    max_iters: int = 900,
    tol: float = 1e-6,
    damp: float = 0.25,
):
    # Standard alpha grid for Figure 4
    alphas = np.linspace(float(alpha_min), float(alpha_max), int(n_alphas))
    seeds = [0]
    
    # NOTE:
    # The exact setting lambda_pt = -c_pt can be extremely slow / numerically fragile.
    # By default we instead use a nearby value lambda_pt = lambda_ratio * c_pt (default -0.99*c_pt).
    if (not allow_lambda_eq_minus_c) and np.isclose(float(lambda_ratio), -1.0, atol=0.0, rtol=0.0):
        raise ValueError("Refusing to run with lambda_ratio=-1. Set --allow_lambda_eq_minus_c to override.")

    lambda_pt_val = float(lambda_ratio) * float(c_pt)
    if (not allow_lambda_eq_minus_c) and np.isclose(lambda_pt_val, -float(c_pt), atol=1e-15, rtol=0.0):
        # Guard against floating-point values that effectively equal -c_pt.
        lambda_pt_val = -0.99 * float(c_pt)

    gamma_reinit_val = 1e-6
    print(f"Using: c_pt={c_pt:.6g}, lambda_pt={lambda_pt_val:.6g} (lambda+c={lambda_pt_val + c_pt:.3g}), gamma_reinit={gamma_reinit_val:.3g}")
    print(f"Grid: n_alphas={len(alphas)}, alpha in [{alphas.min():.3g}, {alphas.max():.3g}], mc={int(mc)}")
    
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
            mc=int(mc),
            seed=[seed],
            gamma_ext=1e-6,
            tol=float(tol),
            max_iters=int(max_iters),
            damp=float(damp),
        )
        all_dfs.append(df)
    
    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
        print(f"\nDone! Results saved to: {out_path}")
        print("You can now merge this into your main replica CSV.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--c_pt", type=float, default=1e-3)
    p.add_argument("--lambda_ratio", type=float, default=-0.99, help="Use lambda_pt = lambda_ratio * c_pt")
    p.add_argument("--allow_lambda_eq_minus_c", action="store_true", help="Allow lambda_ratio=-1 (slow/fragile).")
    p.add_argument("--out_csv", type=str, default="results/replica_fig4_local.csv")
    p.add_argument("--alpha_min", type=float, default=0.01)
    p.add_argument("--alpha_max", type=float, default=0.5)
    p.add_argument("--n_alphas", type=int, default=40)
    p.add_argument("--mc", type=int, default=80_000)
    p.add_argument("--max_iters", type=int, default=900)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--damp", type=float, default=0.25)
    args = p.parse_args()
    run_local(
        c_pt=float(args.c_pt),
        lambda_ratio=float(args.lambda_ratio),
        allow_lambda_eq_minus_c=bool(args.allow_lambda_eq_minus_c),
        out_csv=str(args.out_csv),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        n_alphas=int(args.n_alphas),
        mc=int(args.mc),
        max_iters=int(args.max_iters),
        tol=float(args.tol),
        damp=float(args.damp),
    )
