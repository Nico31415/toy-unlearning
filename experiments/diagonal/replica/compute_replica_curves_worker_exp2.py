import argparse
import itertools
from pathlib import Path

import numpy as np

import ptft_replica_qk as rq

"""
EXPERIMENT 2 (REPLICA / THEORY):
CAN WE LEARN NEW FEATURES?

Mirror the parameter combinations used by compute_emp_curves_worker_exp2.py, but
run the replica fixed-point solver and compute a full alpha curve per task.

Supports optional alpha chunking to increase parallelism:
  - n_alpha_chunks = 1 (default): each task computes the full 80-point alpha curve
  - n_alpha_chunks > 1: split the alpha grid into chunks, and each task computes
    only its assigned chunk.

Task mapping when chunking:
  task_id in [0, n_param_combos*n_alpha_chunks)
  param_combo_idx = task_id // n_alpha_chunks
  alpha_chunk_idx = task_id %  n_alpha_chunks
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
        default="results/replica_ptft_parallel_exp2",
        help="Directory to save results",
    )
    args = parser.parse_args()

    # Fixed + shared (match empirical Exp2)
    rho_pt = 0.10
    rho_fts = [0.10, 0.90]
    omega = 0.0

    # Requested: dense alpha sweep; only one seed
    alphas_full = np.linspace(0.01, 0.5, 80)
    seed = 0

    # Baseline values (used whenever that parameter is not being swept)
    C_BASE = 1e-3
    LAMBDA_BASE = 0.0
    GAMMA_BASE = 0.0

    # Sweep values (exclude the baseline value to avoid duplicate runs)
    cs = [1e-6, 1.0]  # baseline 1e-3 excluded
    lambda_pts = [-0.999e-3, -0.99e-3, 0.99e-3]  # baseline 0 excluded
    gamma_reinits = [1.0, 10.0]  # baseline 0 excluded

    # Build combinations:
    # (sweep_name, rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit, seed)
    param_combinations: list[tuple[str, float, float, float, float, float, float, int]] = []

    # 0) baseline per rho_ft
    param_combinations += [
        ("baseline", rho_pt, rho_ft, omega, C_BASE, LAMBDA_BASE, GAMMA_BASE, seed)
        for rho_ft in rho_fts
    ]

    # 1) vary c_pt, fixed lambda_pt=0, gamma=0 (for each rho_ft)
    param_combinations += [
        ("sweep_c", rho_pt, rho_ft, omega, c_pt, LAMBDA_BASE, GAMMA_BASE, seed)
        for rho_ft, c_pt in itertools.product(rho_fts, cs)
    ]

    # 2) vary lambda_pt, fixed c_pt=1e-3, gamma=0 (for each rho_ft)
    param_combinations += [
        ("sweep_lambda", rho_pt, rho_ft, omega, C_BASE, lambda_pt, GAMMA_BASE, seed)
        for rho_ft, lambda_pt in itertools.product(rho_fts, lambda_pts)
    ]

    # 3) vary gamma_reinit, fixed c_pt=1e-3, lambda_pt=0 (for each rho_ft)
    param_combinations += [
        ("sweep_gamma", rho_pt, rho_ft, omega, C_BASE, LAMBDA_BASE, gamma_reinit, seed)
        for rho_ft, gamma_reinit in itertools.product(rho_fts, gamma_reinits)
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

    sweep_name, rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit, seed = param_combinations[param_combo_idx]

    # Chunk the alpha grid (contiguous slices)
    alpha_chunks = np.array_split(alphas_full, n_alpha_chunks)
    alphas = alpha_chunks[alpha_chunk_idx]

    print(f"Total tasks: {total_tasks}")
    print(f"Running Task {args.task_id}/{total_tasks-1}")
    print(
        f"sweep={sweep_name} | rho_pt={rho_pt}, rho_ft={rho_ft}, omega={omega}, "
        f"c_pt={c_pt}, lambda_pt={lambda_pt}, gamma_reinit={gamma_reinit}, seed={seed}, "
        f"alpha_chunk={alpha_chunk_idx+1}/{n_alpha_chunks}, n_alpha_chunk={len(alphas)}"
    )

    if args.dry_run:
        print("DRY RUN: exiting before computation.")
        return

    df = rq.build_ptft_curves_dataframe(
        rho_pt=[rho_pt],
        rho_ft=[rho_ft],
        omega=[omega],
        a_pt=1.0,
        c_pt=[c_pt],
        lambda_pt=[lambda_pt],
        gamma_reinit=[gamma_reinit],
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
        f"EXPERIMENT2_REPLICA_{sweep_name}"
        f"_rhoft{rho_ft}"
        f"_omega{omega}"
        f"_c{c_pt}"
        f"_lambda{lambda_pt}"
        f"_reinit{gamma_reinit}"
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

