import argparse
import itertools
import numpy as np
import ptft_empirical_finetune_df as emp
from pathlib import Path


"""
EXPERIMENT 3:
CAN WE GET INTO THE NESTED FEATURE REGIME?
Show rich vs lazy learning on PRETRAINED features.

Required runs:
- omega in {1, 0}
- rho_pt = 0.1
- rho_ft in {0.01, 0.04}

For each (omega, rho_ft):
  - baseline (c_pt=1e-3, lambda_pt=0, gamma_reinit=0)  [needed reference]
  - vary c_pt, fixed lambda_pt=0, gamma_reinit=0
  - vary lambda_pt, fixed c_pt=1e-3, gamma_reinit=0
  - vary gamma_reinit, fixed c_pt=1e-3, lambda_pt=0

MIN array IDs: run one baseline per (omega, rho_ft, seed, alpha), and exclude baseline values from sweeps.
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

    # Fixed / required settings
    rho_pt = 0.10
    rho_fts = [0.01, 0.04]
    omegas = [1.0, 0.0]

    # Grids (keep as-is or reduce to shrink total jobs)
    alphas = np.linspace(0.01, 0.5, 11)  # 11
    seeds = list(range(6, 20))           # 14

    # Baseline values
    C_BASE = 1e-3
    LAMBDA_BASE = 0.0
    GAMMA_BASE = 0.0

    # Sweep values (exclude baseline values to avoid duplicates)
    cs = [1e-6, 1.0]  # baseline 1e-3 excluded

    # Sweep lambda_pt directly with c fixed at 1e-3
    lambda_pts = [-1.0e-3, -0.99e-3, 0.99e-3]  # baseline 0 excluded

    gamma_reinits = [1.0, 10.0]  # baseline 0 excluded

    # Build combinations:
    # (sweep_name, rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit, seed, alpha)
    param_combinations = []

    # 0) baseline per (omega, rho_ft)
    param_combinations += [
        ("baseline", rho_pt, rho_ft, omega, C_BASE, LAMBDA_BASE, GAMMA_BASE, seed, alpha)
        for omega, rho_ft, seed, alpha in itertools.product(omegas, rho_fts, seeds, alphas)
    ]

    # 1) vary c_pt; fixed lambda_pt=0, gamma=0
    param_combinations += [
        ("sweep_c", rho_pt, rho_ft, omega, c_pt, LAMBDA_BASE, GAMMA_BASE, seed, alpha)
        for omega, rho_ft, c_pt, seed, alpha in itertools.product(omegas, rho_fts, cs, seeds, alphas)
    ]

    # 2) vary lambda_pt; fixed c_pt=1e-3, gamma=0
    param_combinations += [
        ("sweep_lambda", rho_pt, rho_ft, omega, C_BASE, lambda_pt, GAMMA_BASE, seed, alpha)
        for omega, rho_ft, lambda_pt, seed, alpha in itertools.product(omegas, rho_fts, lambda_pts, seeds, alphas)
    ]

    # 3) vary gamma_reinit; fixed c_pt=1e-3, lambda_pt=0
    param_combinations += [
        ("sweep_gamma", rho_pt, rho_ft, omega, C_BASE, LAMBDA_BASE, gamma_reinit, seed, alpha)
        for omega, rho_ft, gamma_reinit, seed, alpha in itertools.product(omegas, rho_fts, gamma_reinits, seeds, alphas)
    ]

    # Sanity check: ensure no duplicated parameter tuples (ignore sweep_name)
    key = lambda t: t[1:]  # drop sweep_name
    n_unique = len({key(t) for t in param_combinations})
    if n_unique != len(param_combinations):
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
        f"sweep={sweep_name} | rho_pt={rho_pt}, rho_ft={rho_ft}, omega={omega}, "
        f"c_pt={c_pt}, lambda_pt={lambda_pt}, gamma_reinit={gamma_reinit}, "
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
        f"EXPERIMENT3_{sweep_name}"
        f"_omega{omega}"
        f"_rhoft{rho_ft}"
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