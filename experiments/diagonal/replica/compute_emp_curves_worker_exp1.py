import argparse
import itertools
import numpy as np
import ptft_empirical_finetune_df as emp
from pathlib import Path


"""
EXPERIMENT 1:
DO WE BENEFIT FROM EXISTING FEATURES?
Differentiate between pretraining dependence/independence.

We run:
  - ONE shared baseline (omega=0.5, c_pt=1e-3, lambda_pt=0, gamma_reinit=0)
  - PLUS four "vary one thing at a time" sweeps, EXCLUDING the baseline value
    to avoid duplicate runs.

Fixed: rho_pt = rho_ft = 0.1
Each task = one (sweep_name, params, seed, alpha)
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True, help="Slurm array task ID (0..N-1)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/emp_ptft_parallel",
        help="Directory to save results",
    )
    args = parser.parse_args()

    # Fixed sparsity (PT + FT)
    rho_pt = 0.10
    rho_ft = 0.10

    # Grids (keep as-is or reduce to shrink total jobs)
    alphas = np.linspace(0.01, 0.5, 11)   # 11
    seeds = list(range(6, 20))            # 14

    # Baseline parameters
    OMEGA_BASE = 0.5
    C_BASE = 1e-3
    LAMBDA_BASE = 0.0
    GAMMA_BASE = 0.0

    # Sweep values (baseline values excluded to avoid duplicates)
    omegas = [0.0, 1.0]                         # excluded 0.5
    cs = [1e-6, 1.0]                            # excluded 1e-3

    # "vary lambda_pt" with c_pt fixed to 1e-3, gamma_reinit=0, omega fixed
    # excluded 0.0
    lambda_pts = [-1.0e-3, -0.99e-3, 0.99e-3]

    gamma_reinits = [1.0, 10.0]                 # excluded 0.0

    # Build combinations:
    # (sweep_name, rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit, seed, alpha)
    param_combinations = []

    # 0) baseline (run exactly once per seed/alpha)
    param_combinations += [
        ("baseline", rho_pt, rho_ft, OMEGA_BASE, C_BASE, LAMBDA_BASE, GAMMA_BASE, seed, alpha)
        for seed, alpha in itertools.product(seeds, alphas)
    ]

    # A) vary omega; fixed c_pt, lambda_pt=0, gamma_reinit=0
    param_combinations += [
        ("sweep_omega", rho_pt, rho_ft, omega, C_BASE, LAMBDA_BASE, GAMMA_BASE, seed, alpha)
        for omega, seed, alpha in itertools.product(omegas, seeds, alphas)
    ]

    # B) vary c_pt; fixed lambda_pt=0, gamma_reinit=0, omega fixed
    param_combinations += [
        ("sweep_c", rho_pt, rho_ft, OMEGA_BASE, c_pt, LAMBDA_BASE, GAMMA_BASE, seed, alpha)
        for c_pt, seed, alpha in itertools.product(cs, seeds, alphas)
    ]

    # C) vary lambda_pt; fixed c_pt=1e-3, gamma_reinit=0, omega fixed
    param_combinations += [
        ("sweep_lambda", rho_pt, rho_ft, OMEGA_BASE, C_BASE, lambda_pt, GAMMA_BASE, seed, alpha)
        for lambda_pt, seed, alpha in itertools.product(lambda_pts, seeds, alphas)
    ]

    # D) vary gamma_reinit; fixed c_pt=1e-3, lambda_pt=0, omega fixed
    param_combinations += [
        ("sweep_gamma", rho_pt, rho_ft, OMEGA_BASE, C_BASE, LAMBDA_BASE, gamma_reinit, seed, alpha)
        for gamma_reinit, seed, alpha in itertools.product(gamma_reinits, seeds, alphas)
    ]

    # Optional: sanity check no duplicates (ignores sweep_name on purpose)
    # If you *want* to allow same param tuple under different labels, remove this.
    key = lambda t: t[1:]  # drop sweep_name
    n_unique = len({key(t) for t in param_combinations})
    if n_unique != len(param_combinations):
        # If this triggers, you still have duplicates somewhere.
        raise RuntimeError(f"Duplicate parameter tuples found: {len(param_combinations) - n_unique}")

    total_tasks = len(param_combinations)

    if args.task_id < 0 or args.task_id >= total_tasks:
        print(f"Error: Task ID {args.task_id} is out of range (0..{total_tasks-1})")
        print(f"Total tasks = {total_tasks}")
        return

    sweep_name, rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit, seed, alpha = param_combinations[args.task_id]

    print(f"Total tasks: {total_tasks}")
    print(f"Running Task {args.task_id}/{total_tasks-1}")
    print(
        f"sweep={sweep_name} | rho_pt={rho_pt}, rho_ft={rho_ft}, "
        f"omega={omega}, c_pt={c_pt}, lambda_pt={lambda_pt}, gamma_reinit={gamma_reinit}, "
        f"seed={seed}, alpha={alpha}"
    )

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
        lr=0.5,
        epochs=5_000_000,
        test_every_n_epochs=5000,
        log_every_n_epochs=50000,
        no_tuning=True,
        threshold=1e-4,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"EXPERIMENT1_{sweep_name}"
        f"_omega{omega}"
        f"_c{c_pt}"
        f"_lambda{lambda_pt}"
        f"_reinit{gamma_reinit}"
        f"_alpha{alpha:.4f}"
        f"_seed{seed}.csv"
    )
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()