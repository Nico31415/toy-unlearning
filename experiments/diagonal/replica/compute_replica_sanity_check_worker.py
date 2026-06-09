from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

import ptft_replica_qk as rq
from unlearning_experiment_utils import REPLICA_SANITY_ALPHAS, REPLICA_SANITY_TEACHERS, regimes_with_iv_lambda


def _omega_tag(omega: float) -> str:
    return f"{float(omega):g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="results/replica_sanity_check_omega05")
    parser.add_argument("--n-alpha-chunks", type=int, default=5)
    parser.add_argument("--mc", type=int, default=80_000)
    parser.add_argument("--regime-iv-lambda-mult", type=float, default=-0.95)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    regimes = regimes_with_iv_lambda(args.regime_iv_lambda_mult)
    chunk_ids = list(range(int(args.n_alpha_chunks)))
    combos = [
        (regime_name, c_pt, lambda_pt, gamma_reinit, teacher_norm, chunk_id)
        for (regime_name, c_pt, lambda_pt, gamma_reinit), teacher_norm, chunk_id
        in itertools.product(regimes, REPLICA_SANITY_TEACHERS, chunk_ids)
    ]

    total_tasks = len(combos)
    if args.list:
        print(f"total_tasks={total_tasks}")
        print(f"array_range=0-{total_tasks - 1}")
        return

    task_id = int(args.task_id)
    if task_id < 0 or task_id >= total_tasks:
        raise SystemExit(f"ERROR: task-id {task_id} out of range 0..{total_tasks - 1}")

    regime_name, c_pt, lambda_pt, gamma_reinit, teacher_norm, chunk_id = combos[task_id]
    alpha_chunks = np.array_split(np.asarray(REPLICA_SANITY_ALPHAS, dtype=float), int(args.n_alpha_chunks))
    alphas = alpha_chunks[int(chunk_id)]
    if alphas.size == 0:
        raise SystemExit(f"ERROR: alpha chunk {chunk_id} is empty")

    print(f"Total tasks: {total_tasks}")
    print(f"Running task {task_id}/{total_tasks - 1}")
    print(
        f"regime={regime_name} | teacher={teacher_norm} | chunk={chunk_id}/{args.n_alpha_chunks - 1} | "
        f"alpha=[{alphas[0]:.4f}, {alphas[-1]:.4f}] | omega={args.omega:g} | mc={args.mc}"
    )

    df = rq.build_ptft_curves_dataframe(
        rho_pt=[0.1],
        rho_ft=[0.1],
        omega=[float(args.omega)],
        c_pt=[float(c_pt)],
        lambda_pt=[float(lambda_pt)],
        gamma_reinit=[float(gamma_reinit)],
        a_pt=1.0,
        alphas=alphas,
        mc=int(args.mc),
        seed=[0],
        gamma_ext=1e-6,
        tol=1e-6,
        max_iters=900,
        damp=0.25,
        ft_teacher_norm=teacher_norm,
    )
    df["regime"] = regime_name
    df["teacher_norm"] = teacher_norm
    df["alpha_chunk"] = int(chunk_id)
    df["n_alpha_chunks"] = int(args.n_alpha_chunks)
    df["regime_iv_lambda_mult"] = float(args.regime_iv_lambda_mult)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"REPLICA_SANITY_omega{_omega_tag(args.omega)}_{regime_name}_{teacher_norm}_"
        f"chunk{chunk_id:02d}_of_{int(args.n_alpha_chunks):02d}.csv"
    )
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
