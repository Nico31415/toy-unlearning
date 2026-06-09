from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import torch
import torch.nn.functional as F

from unlearning_experiment_utils import (
    GD_VARIANTS,
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


def _parse_variants(value: str) -> List[str]:
    if value.strip().lower() in {"all", "*"}:
        return list(GD_VARIANTS)
    variants = [v.strip() for v in value.split(",") if v.strip()]
    unknown = sorted(set(variants) - set(GD_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Valid variants: {GD_VARIANTS}")
    return variants


def _prepare_net(post_net: DiagonalNet, variant: str) -> DiagonalNet:
    from diagonal_network_pretrain_bg import DiagonalNet

    net = copy.deepcopy(post_net)
    net.train()
    if variant.endswith("_reinit_w"):
        fresh = DiagonalNet(INP_DIM, scaling=1.0, lmda=0.0, c=1e-3, c_vec=None, init_method="complex")
        with torch.no_grad():
            net.w_pos.copy_(fresh.w_pos)
            net.w_neg.copy_(fresh.w_neg)
    readout_only = variant.startswith("readout_only")
    for p in (net.v_pos, net.v_neg):
        p.requires_grad_(not readout_only)
    for p in (net.w_pos, net.w_neg):
        p.requires_grad_(True)
    return net


def _train_recovery(
    net: DiagonalNet,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    beta_target: torch.Tensor,
    *,
    lr: float,
    epochs: int,
    test_every_n_epochs: int,
    threshold: float,
) -> dict:
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=float(lr))
    beta_target = beta_target.to(torch.float64)
    last = {}
    stop_reason = "max_epochs"
    final_epoch = int(epochs) - 1

    for epoch in range(int(epochs)):
        opt.zero_grad()
        loss = F.mse_loss(net(x_train), y_train)
        loss.backward()
        grad_norm = math.sqrt(sum(float(p.grad.norm().item()) ** 2 for p in params if p.grad is not None))
        opt.step()

        if epoch % int(test_every_n_epochs) == 0 or epoch == int(epochs) - 1:
            with torch.no_grad():
                beta = net.beta().detach().cpu().to(torch.float64)
                train_pred_mse = float(F.mse_loss(net(x_train), y_train).item())
                test_pred_mse = float(F.mse_loss(net(x_test), y_test).item())
                target_mse = float(torch.mean((beta - beta_target.cpu()) ** 2).item())
                last = {
                    "train_pred_mse": train_pred_mse,
                    "test_pred_mse": test_pred_mse,
                    "target_mse": target_mse,
                    "grad_norm": float(grad_norm),
                }
            if train_pred_mse < float(threshold):
                stop_reason = "train_pred_mse"
                final_epoch = int(epoch)
                break

    if not last:
        with torch.no_grad():
            beta = net.beta().detach().cpu().to(torch.float64)
            last = {
                "train_pred_mse": float(F.mse_loss(net(x_train), y_train).item()),
                "test_pred_mse": float(F.mse_loss(net(x_test), y_test).item()),
                "target_mse": float(torch.mean((beta - beta_target.cpu()) ** 2).item()),
                "grad_norm": float("nan"),
            }
    return {**last, "stop_reason": stop_reason, "final_epoch": final_epoch}


def build_combos(variants: Iterable[str], recovery_alphas: Iterable[float]):
    return list(
        itertools.product(
            list(variants),
            METHODS,
            TEACHERS,
            SEEDS,
            POST_SELECTIONS,
            RECOVERY_TARGETS,
            list(recovery_alphas),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="results/gd_recovery_correlated_overlap")
    parser.add_argument("--q-sweep-dir", type=str, default="results/sanity_check_correlated_overlap_q_sweep")
    parser.add_argument("--scratch-dir", type=str, default="results/sanity_check_correlated_overlap_scratch")
    parser.add_argument("--variants", type=str, default="all")
    parser.add_argument("--recovery-alphas-json", type=str, default=None)
    parser.add_argument("--first-valid-threshold", type=float, default=0.003)
    parser.add_argument("--matched-alpha", type=float, default=0.563)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5_000_000)
    parser.add_argument("--test-every-n-epochs", type=int, default=5_000)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    variants = _parse_variants(args.variants)
    recovery_alphas = RECOVERY_ALPHAS if args.recovery_alphas_json is None else json.loads(args.recovery_alphas_json)
    combos = build_combos(variants, recovery_alphas)
    total_tasks = len(combos)
    if args.list:
        print(f"total_tasks={total_tasks}")
        print(f"array_range=0-{total_tasks - 1}")
        return

    task_id = int(args.task_id)
    if task_id < 0 or task_id >= total_tasks:
        raise SystemExit(f"ERROR: task-id {task_id} out of range 0..{total_tasks - 1}")

    from diagonal_network_pretrain_bg import make_deterministic

    variant, method, teacher_norm, seed, post_selection, target, recovery_alpha = combos[task_id]
    make_deterministic(20_000_000 + task_id, use_gpu=False)
    torch.set_default_dtype(torch.float64)

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
    n_train = max(1, int(round(float(recovery_alpha) * INP_DIM)))
    x_train, y_train, x_test, y_test = make_recovery_data(
        beta_target,
        n_train=n_train,
        n_test=N_TEST,
        seed=30_000_000 + task_id,
        inp_dim=INP_DIM,
    )
    net = _prepare_net(art["net"], variant)

    print(f"Total tasks: {total_tasks}")
    print(f"Running task {task_id}/{total_tasks - 1}")
    print(
        f"variant={variant} | method={method} | teacher={teacher_norm} | seed={seed} | "
        f"post={post_selection} alpha={selection.post_alpha:.4f} | target={target} | "
        f"recovery_alpha={recovery_alpha:g}"
    )

    train_row = _train_recovery(
        net,
        x_train,
        y_train,
        x_test,
        y_test,
        beta_target,
        lr=float(args.lr),
        epochs=int(args.epochs),
        test_every_n_epochs=int(args.test_every_n_epochs),
        threshold=float(args.threshold),
    )
    with torch.no_grad():
        beta_rec = net.beta().detach().cpu().to(torch.float64)

    row = {
        "task_id": task_id,
        "variant": variant,
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
        "lr": float(args.lr),
        "epochs": int(args.epochs),
        "test_every_n_epochs": int(args.test_every_n_epochs),
        "threshold": float(args.threshold),
        **train_row,
        **beta_error_rows(beta_rec, beta_target, art["beta_pt"], art["support_pt"], art["support_ft"]),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = (
        f"gd_recovery_{variant}_{method}_{teacher_norm}_seed{seed}_"
        f"{post_selection}_{target}_alpha{float(recovery_alpha):.4f}.csv"
    )
    pd.DataFrame([row]).to_csv(out_dir / safe_name, index=False)
    append_csv_locked(out_dir / "gd_recovery_results.csv", row)
    print(f"Saved: {out_dir / safe_name}")


if __name__ == "__main__":
    main()
