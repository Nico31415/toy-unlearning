import argparse
import itertools
import numpy as np
import ptft_empirical_finetune_df as emp
from pathlib import Path


"""
EXPERIMENT 2:
CAN WE LEARN NEW FEATURES?
Differentiate between rich vs lazy learning in NEW features.

Required runs (omega fixed to 0 throughout):
1) Vary rho_ft in {0.1, 0.9} with rho_pt=0.1  (i.e., baseline for each rho_ft)
2) Vary c_pt,   with lambda_pt=0, gamma_reinit=0
3) Vary lambda_pt, with c_pt=1e-3, gamma_reinit=0
4) Vary gamma_reinit, with c_pt=1e-3, lambda_pt=0

MIN array IDs: run one baseline per (rho_ft, seed, alpha) and exclude baseline values from sweeps.
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

    # Fixed + shared
    rho_pt = 0.10
    rho_fts = [0.10, 0.90]
    omega = 0.0

    # Keep your existing grids (reduce if you want fewer jobs)
    alphas = np.linspace(0.01, 0.5, 11)  # 11
    seeds = list(range(6, 20))           # 14

    # Baseline values (used whenever that parameter is not being swept)
    C_BASE = 1e-3
    LAMBDA_BASE = 0.0
    GAMMA_BASE = 0.0

    # Sweep values (exclude the baseline value to avoid duplicate runs)
    cs = [1e-6, 1.0]  # baseline 1e-3 excluded

    # Sweep lambda_pt directly; fixed c_pt=1e-3, gamma=0
    # (matches your prior style: values around 0 at scale of c=1e-3)
    lambda_pts = [-0.999e-3, 0.99e-3]  # baseline 0 excluded

    gamma_reinits = [1.0, 10.0]  # baseline 0 excluded

    # Build combinations:
    # (sweep_name, rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit, seed, alpha)
    param_combinations = []

    # 2) vary lambda_pt, fixed c_pt=1e-3, gamma=1e-6 (for each rho_ft)
    param_combinations += [
        ("sweep_lambda_gamma_ext", rho_pt, rho_ft, omega, C_BASE, lambda_pt, 1e-6, seed, alpha)
        for rho_ft, lambda_pt, seed, alpha in itertools.product(rho_fts, [-1.0e-3], seeds, alphas)
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
        f"EXPERIMENT2_{sweep_name}"
        f"_rhoft{rho_ft}"
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