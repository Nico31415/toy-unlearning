import argparse
import itertools
import numpy as np
import ptft_empirical_finetune_df as emp
from pathlib import Path


"""
EXPERIMENT 4:
SINGLE TASK LEARNING (SLT)

Goals:
- Compare to other curves
- Show lambda_pt is irrelevant for SLT
- Show influence of c_pt on SLT

Required grid:
- rho_pt in {0.9, 0.04, 0.01, 0.1}
- c_pt   in {1e-6, 1e-3, 1}
- lambda_pt in {0, -c_pt, -0.99*c_pt, 0.99*c_pt}

Each task = one (rho_pt, c_pt, lambda_pt, seed, alpha)

This runs emp.build_single_task_curves_dataframe (single-task version).
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True, help="Slurm array task ID (0..N-1)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/emp_slt_parallel",
        help="Directory to save results",
    )
    args = parser.parse_args()

    # Grids (keep as-is or reduce to shrink total jobs)
    alphas = np.linspace(0.01, 0.5, 11)  # 11
    seeds = list(range(6, 20))           # 14

    # Required experiment grid
    rho_pts = [0.9, 0.04, 0.01, 0.1]
    cs = [1e-6, 1e-3, 1.0]
    lmda_multipliers = [0.0, -1.0, -0.99, 0.99]  # lambda_pt = mult * c_pt

    # Build combinations:
    # (rho_pt, c_pt, lambda_pt, seed, alpha)
    param_combinations = [
        (rho_pt, c_pt, lmda_mult * c_pt, seed, alpha)
        for rho_pt, c_pt, lmda_mult, seed, alpha in itertools.product(
            rho_pts, cs, lmda_multipliers, seeds, alphas
        )
    ]

    total_tasks = len(param_combinations)

    if args.task_id < 0 or args.task_id >= total_tasks:
        print(f"Error: Task ID {args.task_id} is out of range (0..{total_tasks-1})")
        print(f"Total tasks = {total_tasks}")
        return

    rho_pt, c_pt, lambda_pt, seed, alpha = param_combinations[args.task_id]

    print(f"Total tasks: {total_tasks}")
    print(f"Running Task {args.task_id}/{total_tasks-1}")
    print(
        f"rho_pt={rho_pt}, c_pt={c_pt}, lambda_pt={lambda_pt}, seed={seed}, alpha={alpha}"
    )

    # Single-task empirical computation
    df = emp.build_single_task_curves_dataframe(
        rho_pt=[rho_pt],
        a_pt=1.0,
        c_pt=[c_pt],
        lambda_pt=[lambda_pt],
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

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"EXPERIMENT4_slt"
        f"_rhop{rho_pt}"
        f"_c{c_pt}"
        f"_lambda{lambda_pt}"
        f"_alpha{alpha:.4f}"
        f"_seed{seed}.csv"
    )
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()