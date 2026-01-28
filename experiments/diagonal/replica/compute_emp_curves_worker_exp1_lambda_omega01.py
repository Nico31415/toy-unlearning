import argparse
import itertools
from pathlib import Path

import numpy as np

import ptft_empirical_finetune_df as emp

"""
EXPERIMENT 1 (EMPIRICAL) — LAMBDA SWEEP @ OMEGA ∈ {0, 1}

Your original Exp1 runs sweep_lambda only at omega=0.5 (baseline overlap).
This worker adds ONLY the missing sweep_lambda runs at omega=0 and omega=1.

Fixed:
  - rho_pt = rho_ft = 0.1
  - c_pt = 1e-3
  - gamma_reinit = 0

Sweep:
  - omega ∈ {0.0, 1.0}
  - lambda_pt ∈ {-1e-3, -0.99e-3, 0.99e-3}  (baseline 0 excluded)

Each task = one (omega, lambda_pt, seed, alpha)
Total tasks = 2 * 3 * 14 * 11 = 924
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True, help="Slurm array task ID (0..N-1)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/emp_ptft_parallel",
        help="Directory to save results",
    )
    args = parser.parse_args()

    rho_pt = 0.10
    rho_ft = 0.10

    alphas = np.linspace(0.01, 0.5, 11)  # 11
    seeds = list(range(6, 20))  # 14

    omegas = [0.0, 1.0]

    C_BASE = 1e-3
    GAMMA_BASE = 0.0

    lambda_pts = [-1.0e-3, -0.99e-3, 0.99e-3]

    # Build combinations:
    # (rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit, seed, alpha)
    param_combinations: list[tuple[float, float, float, float, float, float, int, float]] = [
        (rho_pt, rho_ft, omega, C_BASE, lambda_pt, GAMMA_BASE, seed, float(alpha))
        for omega, lambda_pt, seed, alpha in itertools.product(omegas, lambda_pts, seeds, alphas)
    ]

    total_tasks = len(param_combinations)
    if args.task_id < 0 or args.task_id >= total_tasks:
        print(f"Error: Task ID {args.task_id} is out of range (0..{total_tasks-1})")
        print(f"Total tasks = {total_tasks}")
        return

    rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit, seed, alpha = param_combinations[args.task_id]

    print(f"Total tasks: {total_tasks}")
    print(f"Running Task {args.task_id}/{total_tasks-1}")
    print(
        f"sweep=sweep_lambda | rho_pt={rho_pt}, rho_ft={rho_ft}, "
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
        "EXPERIMENT1_sweep_lambda"
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

