import argparse
import itertools
from pathlib import Path

import numpy as np

import ptft_replica_qk as rq

"""
EXPERIMENT 1 (REPLICA) — OMEGA EXTENSION

Adds missing replica runs so that sweep_c / sweep_lambda / sweep_gamma are also
computed for omega ∈ {0, 1} (not just omega=0.5).

This worker runs ONLY the omega-extension combinations:
  - sweep_c:      omega ∈ {0,1}, c_pt ∈ {1e-6, 1.0}, lambda_pt=0, gamma_reinit=0
  - sweep_lambda: omega ∈ {0,1}, c_pt=1e-3, lambda_pt ∈ {-1e-3, -0.99e-3, 0.99e-3}, gamma_reinit=0
  - sweep_gamma:  omega ∈ {0,1}, c_pt=1e-3, lambda_pt=0, gamma_reinit ∈ {1,10}

Fixed: rho_pt=rho_ft=0.1
One seed (0). Computes a dense 80-point alpha curve, optionally chunked.
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
        default="results/replica_ptft_parallel_exp1",
        help="Directory to save results",
    )
    args = parser.parse_args()

    rho_pt = 0.10
    rho_ft = 0.10
    seed = 0

    alphas_full = np.linspace(0.01, 0.5, 80)
    n_alpha_chunks = int(args.n_alpha_chunks)
    if n_alpha_chunks <= 0:
        raise ValueError("--n-alpha-chunks must be >= 1")

    omegas = [0.0, 1.0]

    C_BASE = 1e-3
    LAMBDA_BASE = 0.0
    GAMMA_BASE = 0.0

    cs = [1e-6, 1.0]
    lambda_pts = [-1.0e-3, -0.99e-3, 0.99e-3]
    gamma_reinits = [1.0, 10.0]

    # (sweep_name, rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit, seed)
    param_combinations: list[tuple[str, float, float, float, float, float, float, int]] = []

    param_combinations += [
        ("sweep_c", rho_pt, rho_ft, omega, c_pt, LAMBDA_BASE, GAMMA_BASE, seed)
        for omega, c_pt in itertools.product(omegas, cs)
    ]
    param_combinations += [
        ("sweep_lambda", rho_pt, rho_ft, omega, C_BASE, lambda_pt, GAMMA_BASE, seed)
        for omega, lambda_pt in itertools.product(omegas, lambda_pts)
    ]
    param_combinations += [
        ("sweep_gamma", rho_pt, rho_ft, omega, C_BASE, LAMBDA_BASE, gamma_reinit, seed)
        for omega, gamma_reinit in itertools.product(omegas, gamma_reinits)
    ]

    total_tasks = len(param_combinations) * n_alpha_chunks
    if args.task_id < 0 or args.task_id >= total_tasks:
        print(f"Error: Task ID {args.task_id} is out of range (0..{total_tasks-1})")
        print(f"Total tasks = {total_tasks}")
        return

    param_combo_idx = int(args.task_id) // n_alpha_chunks
    alpha_chunk_idx = int(args.task_id) % n_alpha_chunks

    sweep_name, rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit, seed = param_combinations[param_combo_idx]

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
        f"EXPERIMENT1_REPLICA_{sweep_name}"
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

