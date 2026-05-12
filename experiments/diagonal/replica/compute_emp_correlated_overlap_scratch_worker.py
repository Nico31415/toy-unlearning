"""
Scratch-FT baseline for correlated-overlap teachers.

This trains a fresh diagonal network directly on the FT task, using the same
PT/FT teacher pair as the PT+FT experiments only so that forgetting metrics can
still be measured relative to beta_PT.
"""
import argparse
import fcntl
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
import ptft_empirical_finetune_df as emp
from diagonal_network_pretrain_bg import DiagonalNet, make_deterministic, train


QS = [0.25, 0.50, 0.75]
ALPHAS = list(np.linspace(0.01, 0.8, 11))
SEEDS = list(range(5))


def _q_name(q: float) -> str:
    return f"correlated_overlap_q{q:.2f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="results/sanity_check_correlated_overlap_scratch")
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    combos = [(q, seed, alpha) for q, seed, alpha in itertools.product(QS, SEEDS, ALPHAS)]
    total_tasks = len(combos)
    if args.list:
        print(f"total_tasks={total_tasks}")
        print(f"array_range=0-{total_tasks - 1}")
        return

    task_id = int(args.task_id)
    if task_id < 0 or task_id >= total_tasks:
        raise SystemExit(f"ERROR: task-id {task_id} out of range 0..{total_tasks - 1}")

    q, seed, alpha = combos[task_id]
    teacher_norm = _q_name(q)
    out_dir = Path(args.output_dir)
    save_folder = out_dir / "scratch" / teacher_norm / f"seed{seed}_alpha{alpha:.4f}"

    print(f"Total tasks: {total_tasks}")
    print(f"Running task {task_id}/{total_tasks - 1}")
    print(f"scratch | teacher={teacher_norm} | seed={seed} | alpha={alpha:.4f} | omega={args.omega:g}")

    t0 = time.time()
    seed = int(seed)
    inp_dim = 5000
    rho_pt = 0.1
    rho_ft = 0.1
    a_pt = 1.0
    c_init = 1e-3
    lambda_init = 0.0
    n_test = 10_000
    n_train = max(1, int(round(float(alpha) * inp_dim)))
    alpha_eff = n_train / inp_dim

    make_deterministic(seed, use_gpu=False)
    torch.set_default_dtype(torch.float64)

    gen_pt = torch.Generator(device="cpu").manual_seed(seed + 0)
    gen_ft = torch.Generator(device="cpu").manual_seed(seed + 1)
    gen_train_x = torch.Generator(device="cpu").manual_seed(seed + 2 + 10_000 * n_train)
    gen_test_x = torch.Generator(device="cpu").manual_seed(seed + 3)

    beta_pt, support_pt = emp.sample_pt_teacher_deterministic(inp_dim, rho_pt, a_pt, gen_pt)
    beta_ft, support_ft = emp.sample_ft_teacher_with_overlap(
        inp_dim, rho_ft, float(args.omega), support_pt, gen_ft,
        ft_teacher_norm=teacher_norm, beta_pt=beta_pt,
    )

    x_train = torch.randn(n_train, inp_dim, generator=gen_train_x) / math.sqrt(inp_dim)
    x_test = torch.randn(n_test, inp_dim, generator=gen_test_x) / math.sqrt(inp_dim)
    y_train = x_train @ beta_ft
    y_test = x_test @ beta_ft

    save_folder.mkdir(parents=True, exist_ok=True)
    net = DiagonalNet(inp_dim, scaling=1.0, lmda=lambda_init, c=c_init, c_vec=None, init_method="complex")

    with torch.no_grad():
        beta0 = net.beta().detach().cpu().numpy()
    beta0_norm = float(np.linalg.norm(beta0))

    df, net, norm_df, stop_reason, final_epoch = train(
        net,
        (x_train, y_train),
        (x_test, y_test),
        beta_ft,
        test_every_n_epochs=5_000,
        log_every_n_epochs=50_000,
        lr=0.5,
        epochs=5_000_000,
        lr_tuning=False,
        threshold=1e-4,
        stop_pred_mse=None,
        stop_beta_rate=0.0,
        stop_grad_norm=0.0,
        lr_decay=1.0,
        lr_decay_interval=2000,
        save_folder=str(save_folder),
    )

    df_test = df[df["split"] == "test"].sort_values("epoch")
    last = df_test.iloc[-1].to_dict()
    with torch.no_grad():
        beta_hat = net.beta().detach().cpu()

    ov = int((support_pt & support_ft).sum().item())
    new = int((~support_pt & support_ft).sum().item())
    ptonly = int((support_pt & ~support_ft).sum().item())
    none = int((~support_pt & ~support_ft).sum().item())
    active_frac = float((beta_hat.abs() > 1e-6).double().mean().item())

    torch.save(beta_pt, save_folder / "beta_pt.pt")
    torch.save(beta_ft, save_folder / "beta_ft.pt")
    torch.save(net.state_dict(), save_folder / "model.pt")
    torch.save(support_pt, save_folder / "support_pt.pt")
    torch.save(support_ft, save_folder / "support_ft.pt")
    df.to_feather(save_folder / "df.feather")
    norm_df.to_feather(save_folder / "norm_df.feather")
    np.save(save_folder / "beta0.npy", beta0)

    row = {
        "status": "ok",
        "setting": "scratch_ft",
        "seed": seed,
        "inp_dim": inp_dim,
        "alpha_requested": float(alpha),
        "n_train": int(n_train),
        "alpha": float(alpha_eff),
        "n_test": n_test,
        "c_pt": c_init,
        "lambda_pt": lambda_init,
        "gamma_reinit": 0.0,
        "rho_pt": rho_pt,
        "a_pt": a_pt,
        "rho_ft": rho_ft,
        "omega": float(args.omega),
        "ft_teacher_norm": teacher_norm,
        "save_folder": str(save_folder),
        "lr": 0.5,
        "epochs": 5_000_000,
        "test_every_n_epochs": 5_000,
        "log_every_n_epochs": 50_000,
        "no_tuning": True,
        "threshold": 1e-4,
        "stop_pred_mse": None,
        "stop_beta_rate": 0.0,
        "stop_grad_norm": 0.0,
        "lr_decay": 1.0,
        "lr_decay_interval": 2000,
        "stop_reason": stop_reason,
        "final_epoch": int(final_epoch),
        "final_test_pred_mse": float(last["pred_mse"]),
        "final_param_mse": float(last["param_mse"]),
        "final_grad_norm": float(last["grad_norm"]),
        "final_beta_update_rate": float(last["beta_update_rate"]),
        "final_beta_l1": float(last["beta_l1"]),
        "final_beta_l2": float(last["beta_l2"]),
        "final_active_frac": active_frac,
        "beta0_l2": beta0_norm,
        "empirical_omega": float(ov / max(1, int(support_ft.sum().item()))),
        "n_ov": ov,
        "n_new": new,
        "n_ptonly": ptonly,
        "n_none": none,
        "wall_s": float(time.time() - t0),
        "regime": "scratch",
        "teacher_norm": teacher_norm,
        "overlap_q": float(q),
        "experiment": "correlated_overlap_scratch",
    }

    with open(save_folder / "config.json", "w") as f:
        json.dump({**row, "alpha": float(alpha_eff)}, f, indent=2)

    out_dir.mkdir(parents=True, exist_ok=True)
    master_csv = out_dir / "scratch_results.csv"
    lock_path = master_csv.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        header = not master_csv.exists()
        pd.DataFrame([row]).to_csv(master_csv, mode="a", header=header, index=False)
        fcntl.flock(lf, fcntl.LOCK_UN)

    print(f"Saved weights to: {save_folder}")
    print(f"Appended row to:  {master_csv}")


if __name__ == "__main__":
    main()
