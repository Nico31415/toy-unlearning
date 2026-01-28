import argparse
import itertools
from pathlib import Path

import numpy as np

import ptft_empirical_finetune_df as emp

"""
EXPERIMENT 1 (EMPIRICAL) — OMEGA EXTENSION

Your original Exp1 implementation runs "vary one thing at a time" sweeps with omega fixed to 0.5
for sweep_c / sweep_lambda / sweep_gamma.

This worker adds the *missing* runs so that these sweeps are also performed at omega ∈ {0, 1}.

What this worker runs (ONLY these; it does NOT re-run baseline or sweep_omega):
  - sweep_c:      omega ∈ {0,1}, c_pt ∈ {1e-6, 1.0}, lambda_pt=0, gamma_reinit=0
  - sweep_lambda: omega ∈ {0,1}, c_pt=1e-3, lambda_pt ∈ {-1e-3, -0.99e-3, 0.99e-3}, gamma_reinit=0
  - sweep_gamma:  omega ∈ {0,1}, c_pt=1e-3, lambda_pt=0, gamma_reinit ∈ {1,10}

Fixed: rho_pt=rho_ft=0.1
Each task = one (sweep_name, params, seed, alpha)
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

    # Baseline parameter values (used when not being swept)
    C_BASE = 1e-3
    LAMBDA_BASE = 0.0
    GAMMA_BASE = 0.0

    # Sweep values (baseline excluded)
    cs = [1e-6, 1.0]
    lambda_pts = [-1.0e-3, -0.99e-3, 0.99e-3]
    gamma_reinits = [1.0, 10.0]

    # Build combinations:
    # (sweep_name, rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit, seed, alpha)
    param_combinations: list[tuple[str, float, float, float, float, float, float, int, float]] = []

    # B-ext) vary c_pt at omega ∈ {0,1}
    param_combinations += [
        ("sweep_c", rho_pt, rho_ft, omega, c_pt, LAMBDA_BASE, GAMMA_BASE, seed, float(alpha))
        for omega, c_pt, seed, alpha in itertools.product(omegas, cs, seeds, alphas)
    ]

    # C-ext) vary lambda_pt at omega ∈ {0,1}
    param_combinations += [
        ("sweep_lambda", rho_pt, rho_ft, omega, C_BASE, lambda_pt, GAMMA_BASE, seed, float(alpha))
        for omega, lambda_pt, seed, alpha in itertools.product(omegas, lambda_pts, seeds, alphas)
    ]

    # D-ext) vary gamma_reinit at omega ∈ {0,1}
    param_combinations += [
        ("sweep_gamma", rho_pt, rho_ft, omega, C_BASE, LAMBDA_BASE, gamma_reinit, seed, float(alpha))
        for omega, gamma_reinit, seed, alpha in itertools.product(omegas, gamma_reinits, seeds, alphas)
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

