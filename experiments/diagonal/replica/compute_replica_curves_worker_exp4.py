import argparse
import itertools
from pathlib import Path

import numpy as np

import ptft_replica_qk as rq

"""
EXPERIMENT 4 (REPLICA / THEORY):
SINGLE TASK LEARNING (SLT)

Mirror the parameter combinations used by compute_emp_curves_worker_exp4.py, but
run the replica fixed-point solver for single-task learning and compute a full 
alpha curve per task.

Required grid:
- rho_pt in {0.9, 0.04, 0.01, 0.1}
- c_pt   in {1e-6, 1e-3, 1}
- lambda_pt in {0, -c_pt, -0.99*c_pt, 0.99*c_pt} (though lambda_pt is irrelevant for SLT)

Supports optional alpha chunking to increase parallelism.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True, help="Slurm array task ID (0..N-1)")
    parser.add_argument(
        "--n-alpha-chunks",
        type=int,
        default=1,
        help="Split the alpha grid into this many chunks to increase parallelism.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected parameter combo/chunk and exit without computing.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/replica_slt_parallel_exp4",
        help="Directory to save results",
    )
    args = parser.parse_args()

    # Required experiment grid (match empirical Exp4)
    rho_pts = [0.9, 0.04, 0.01, 0.1]
    cs = [1e-6, 1e-3, 1.0]
    # lambda_pt = mult * c_pt. 
    # Note: for SLT, lambda_pt doesn't affect the result, but we keep it for consistency.
    lmda_multipliers = [0.0, -1.0, -0.99, 0.99]

    # Requested: dense alpha sweep; only one seed
    alphas_full = np.linspace(0.01, 0.5, 80)
    seed = 0

    # Build combinations:
    # (rho_pt, c_pt, lambda_pt, seed)
    param_combinations = [
        (rho_pt, c_pt, lmda_mult * c_pt, seed)
        for rho_pt, c_pt, lmda_mult in itertools.product(
            rho_pts, cs, lmda_multipliers
        )
    ]

    n_alpha_chunks = int(args.n_alpha_chunks)
    if n_alpha_chunks <= 0:
        raise ValueError("--n-alpha-chunks must be >= 1")

    total_tasks = len(param_combinations) * n_alpha_chunks
    if args.task_id < 0 or args.task_id >= total_tasks:
        print(f"Error: Task ID {args.task_id} is out of range (0..{total_tasks-1})")
        print(f"Total tasks = {total_tasks}")
        return

    param_combo_idx = int(args.task_id) // n_alpha_chunks
    alpha_chunk_idx = int(args.task_id) % n_alpha_chunks

    rho_pt, c_pt, lambda_pt, seed = param_combinations[param_combo_idx]

    # Chunk the alpha grid (contiguous slices)
    alpha_chunks = np.array_split(alphas_full, n_alpha_chunks)
    alphas = alpha_chunks[alpha_chunk_idx]

    print(f"Total tasks: {total_tasks}")
    print(f"Running Task {args.task_id}/{total_tasks-1}")
    print(
        f"rho_pt={rho_pt}, c_pt={c_pt}, lambda_pt={lambda_pt}, seed={seed}, "
        f"alpha_chunk={alpha_chunk_idx+1}/{n_alpha_chunks}, n_alpha_chunk={len(alphas)}"
    )

    if args.dry_run:
        print("DRY RUN: exiting before computation.")
        return

    # Single-task replica computation
    # Note: ptft_replica_qk.build_single_task_curves_dataframe uses 'rho' and 'c'
    df = rq.build_single_task_curves_dataframe(
        rho=[rho_pt],
        c=[c_pt],
        alphas=alphas,
        mc=80_000,
        seed=[seed],
        gamma_ext=1e-6,
        tol=1e-6,
        max_iters=900,
        damp=0.25,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_min = float(alphas[0])
    alpha_max = float(alphas[-1])
    filename = (
        f"EXPERIMENT4_REPLICA_slt"
        f"_rhop{rho_pt}"
        f"_c{c_pt}"
        f"_lambda{lambda_pt}"
        f"_seed{seed}.csv"
    )
    if n_alpha_chunks > 1:
        filename = filename.replace(
            ".csv",
            f"_alphachunk{alpha_chunk_idx:02d}of{n_alpha_chunks:02d}_alpharange{alpha_min:.4f}-{alpha_max:.4f}.csv",
        )
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
