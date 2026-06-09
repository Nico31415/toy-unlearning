from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

from unlearning_experiment_utils import (
    INP_DIM,
    N_TEST,
    POST_SELECTIONS,
    RECOVERY_ALPHAS,
    RECOVERY_TARGETS,
    append_csv_locked,
    beta_error_rows,
    load_run_artifacts,
    make_recovery_data,
    q_name,
    select_post_run,
    target_beta,
)


METHODS = ["regime_II", "regime_IV", "scratch"]
TEACHERS = [q_name(q) for q in (0.25, 0.50, 0.75)]
SEEDS = [0, 1, 2]


def fit_readout_closed_form(net, x_train: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
    """Fit only final readout weights for fixed post-FT input weights."""
    d = int(x_train.shape[1])
    v_pos = net.v_pos.detach().cpu().to(torch.float64).numpy()
    v_neg = net.v_neg.detach().cpu().to(torch.float64).numpy()
    x_np = x_train.detach().cpu().to(torch.float64).numpy()
    y_np = y_train.detach().cpu().to(torch.float64).numpy()
    features = np.concatenate([x_np * v_pos[None, :], -x_np * v_neg[None, :]], axis=1)
    theta, *_ = np.linalg.lstsq(features, y_np, rcond=None)
    w_pos = theta[:d]
    w_neg = theta[d:]
    beta_rec = v_pos * w_pos - v_neg * w_neg
    return torch.from_numpy(beta_rec).to(torch.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="results/readout_recovery_correlated_overlap")
    parser.add_argument("--q-sweep-dir", type=str, default="results/sanity_check_correlated_overlap_q_sweep")
    parser.add_argument("--scratch-dir", type=str, default="results/sanity_check_correlated_overlap_scratch")
    parser.add_argument("--first-valid-threshold", type=float, default=0.003)
    parser.add_argument("--matched-alpha", type=float, default=0.563)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    combos = list(itertools.product(METHODS, TEACHERS, SEEDS, POST_SELECTIONS, RECOVERY_TARGETS))
    total_tasks = len(combos)
    if args.list:
        print(f"total_tasks={total_tasks}")
        print(f"array_range=0-{total_tasks - 1}")
        return

    task_id = int(args.task_id)
    if task_id < 0 or task_id >= total_tasks:
        raise SystemExit(f"ERROR: task-id {task_id} out of range 0..{total_tasks - 1}")

    method, teacher_norm, seed, post_selection, target = combos[task_id]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = select_post_run(
        method=method,
        teacher_norm=teacher_norm,
        seed=int(seed),
        post_selection=post_selection,
        base_q_sweep=Path(args.q_sweep_dir),
        base_scratch=Path(args.scratch_dir),
        first_valid_threshold=float(args.first_valid_threshold),
        matched_alpha=float(args.matched_alpha),
    )
    art = load_run_artifacts(selection.run_dir)
    beta_target = target_beta(art["beta_pt"], art["support_pt"], art["support_ft"], target)

    print(f"Total tasks: {total_tasks}")
    print(f"Running task {task_id}/{total_tasks - 1}")
    print(
        f"method={method} | teacher={teacher_norm} | seed={seed} | post={post_selection} "
        f"(alpha={selection.post_alpha:.4f}, p_FT={selection.post_p_FT:.6g}) | target={target}"
    )

    rows = []
    for i, recovery_alpha in enumerate(RECOVERY_ALPHAS):
        n_train = max(1, int(round(float(recovery_alpha) * INP_DIM)))
        x_train, y_train, x_test, y_test = make_recovery_data(
            beta_target,
            n_train=n_train,
            n_test=N_TEST,
            seed=10_000_000 + task_id * 101 + i,
            inp_dim=INP_DIM,
        )
        beta_rec = fit_readout_closed_form(art["net"], x_train, y_train)
        pred_test = x_test @ beta_rec
        row = {
            "task_id": task_id,
            "method": method,
            "teacher_norm": teacher_norm,
            "seed": int(seed),
            "post_selection": post_selection,
            "post_alpha": selection.post_alpha,
            "post_p_FT": selection.post_p_FT,
            "post_run_dir": str(selection.run_dir),
            "target": target,
            "recovery_alpha": float(recovery_alpha),
            "n_recovery": int(n_train),
            "test_pred_mse": float(torch.mean((pred_test - y_test) ** 2).item()),
            **beta_error_rows(beta_rec, beta_target, art["beta_pt"], art["support_pt"], art["support_ft"]),
        }
        rows.append(row)

    out_name = (
        f"readout_recovery_{method}_{teacher_norm}_seed{seed}_{post_selection}_{target}.csv"
    )
    out_path = out_dir / out_name
    torch.cuda.empty_cache()
    import pandas as pd

    pd.DataFrame(rows).to_csv(out_path, index=False)
    append_csv_locked(out_dir / "readout_recovery_results.csv", rows)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
