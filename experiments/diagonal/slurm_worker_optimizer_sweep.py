#!/usr/bin/env python3
"""
SLURM worker: optimizer comparison for SETUP 2 ONLY, now sweeping lambda_pt.

This worker runs only setup2 conditions:
    - omega fixed at 1
    - rho_ft in {0.1, 0.01}
    - lambda_pt in {-0.99*C_PT, 0.0, 0.99*C_PT}

Each SLURM array task corresponds to exactly one:
    (alpha_ft, seed, condition)

Within each task, the 3 optimizers are run sequentially.

Total tasks:
    11 alpha_ft × 14 seeds × 6 conditions = 924 tasks

Condition count:
    2 rho_ft values × 3 lambda_pt values = 6 setup2 conditions

Suggested SLURM array:
    --array=0-923
"""

import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import math
import sys
import json
from pathlib import Path

sys.path.append('')

import torch
import torch.nn.functional as F
import numpy as np

from experiments.diagonal.diagonal_network_pretrain_bg import (
    DiagonalNet, train, make_deterministic,
)
from ReplicaExperiments.fixed_lambda_all import compute_c_ft_from_pt
from experiments.diagonal.diagonal_ptft_oracle import (
    sample_pt_teacher, sample_ft_teacher_with_overlap,
)

# ── Fixed parameters ──────────────────────────────────────────────────────────
INP_DIM         = 1000
RHO_PT          = 0.10
A_PT            = 1.0
C_PT            = 1e-3
GAMMA_FT        = 0.0
N_TEST          = 10000
EPOCHS          = 5_000_000
THRESHOLD       = 1e-4

ALPHA_FT_VALUES = list(np.linspace(0, 0.5, 11))   # 0.0, 0.05, ..., 0.5
N_SEEDS         = 14

# ── Experimental conditions ───────────────────────────────────────────────────
# SETUP 1 intentionally omitted/commented out because data already exists.
#
# SETUP1_CONDITIONS = [
#     dict(setup='setup1', omega=omega, lambda_pt=lam, rho_ft=0.04)
#     for omega in [0, 1]
#     for lam in [-0.99 * C_PT, 0.0, 0.99 * C_PT]
# ]

# SETUP 2 ONLY: now sweep rho_ft × lambda_pt, with omega fixed to 1
SETUP2_CONDITIONS = [
    dict(setup='setup2', omega=1, lambda_pt=lam, rho_ft=rho_ft)
    for rho_ft in [0.1, 0.01]
    for lam in [-0.99 * C_PT, 0.0, 0.99 * C_PT]
]

ALL_CONDITIONS = SETUP2_CONDITIONS   # 6 conditions total

# ── Optimizers ────────────────────────────────────────────────────────────────
OPTIMIZERS = [
    dict(
        name='sgd_bs32',
        optimizer_type='sgd',
        lr=0.5,
        weight_decay=0.0,
        l2sp_lambda=0.0,
        batch_size=32,
    ),
    dict(
        name='adam_wd1e4',
        optimizer_type='adam',
        lr=1e-3,
        weight_decay=1e-4,
        l2sp_lambda=0.0,
        batch_size=None,
    ),
    dict(
        name='adam_l2sp1e4',
        optimizer_type='adam',
        lr=1e-3,
        weight_decay=0.0,
        l2sp_lambda=1e-4,
        batch_size=None,
    ),
]


def decode_task_id(task_id: int):
    """
    Decode task_id into (alpha_ft_idx, seed, condition_id).

    Ordering:
        alpha_ft varies fastest,
        then seed,
        then condition.

    task_id = condition_id * (N_SEEDS * N_ALPHA) + seed * N_ALPHA + alpha_ft_idx
    """
    n_alpha = len(ALPHA_FT_VALUES)
    n_seeds = N_SEEDS
    n_conditions = len(ALL_CONDITIONS)

    total_tasks = n_alpha * n_seeds * n_conditions

    if task_id < 0 or task_id >= total_tasks:
        raise ValueError(f"task_id {task_id} out of range for total_tasks={total_tasks}")

    alpha_ft_idx = task_id % n_alpha
    seed = (task_id // n_alpha) % n_seeds
    condition_id = task_id // (n_alpha * n_seeds)

    return alpha_ft_idx, seed, condition_id, total_tasks


def run_single(alpha_ft, seed, condition, optimizer_cfg, save_root):
    """Run one (alpha_ft, seed, condition, optimizer) combination."""
    make_deterministic(seed)
    torch.set_default_dtype(torch.float64)

    omega = condition["omega"]
    lambda_pt = condition["lambda_pt"]
    rho_ft = condition["rho_ft"]
    setup = condition["setup"]

    n_train = max(1, int(round(alpha_ft * INP_DIM)))

    gen_pt = torch.Generator().manual_seed(seed * 10 + 0)
    gen_ft = torch.Generator().manual_seed(seed * 10 + 1)
    gen_train_x = torch.Generator().manual_seed(seed * 10 + 2)
    gen_test_x = torch.Generator().manual_seed(seed * 10 + 3)

    # Teachers
    beta_pt, support_pt = sample_pt_teacher(INP_DIM, RHO_PT, A_PT, gen_pt)
    beta_ft, _ = sample_ft_teacher_with_overlap(
        INP_DIM, rho_ft, omega, support_pt, gen_ft
    )

    # Oracle FT init
    c_ft = compute_c_ft_from_pt(beta_pt.numpy(), C_PT, lambda_pt, GAMMA_FT)

    # Data
    x_train = torch.randn(n_train, INP_DIM, generator=gen_train_x) / math.sqrt(INP_DIM)
    x_test = torch.randn(N_TEST, INP_DIM, generator=gen_test_x) / math.sqrt(INP_DIM)
    y_train = x_train @ beta_ft
    y_test = x_test @ beta_ft

    # Model
    net = DiagonalNet(
        INP_DIM,
        scaling=1.0,
        lmda=0.0,
        c=C_PT,
        c_vec=c_ft,
        init_method="complex",
    )

    # Batch size: min(32, n_train) for SGD
    bs = optimizer_cfg["batch_size"]
    eff_batch = None if bs is None else min(int(bs), n_train)
    momentum = 0.9 if optimizer_cfg["optimizer_type"] == "sgd" else 0.0

    # Folder name encodes all hyperparams
    cond_str = (
        f"{setup}"
        f"--omega={omega}"
        f"--lpt={lambda_pt:.4f}"
        f"--rft={rho_ft}"
        f"--opt={optimizer_cfg['name']}"
        f"--aft={alpha_ft:.4f}"
        f"--seed={seed}"
    )
    run_folder = Path(save_root) / cond_str
    run_folder.mkdir(parents=True, exist_ok=True)

    _, net_out, _, stop_reason, final_epoch = train(
        net,
        (x_train, y_train),
        (x_test, y_test),
        beta_ft,
        lr=optimizer_cfg["lr"],
        momentum=momentum,
        epochs=EPOCHS,
        threshold=THRESHOLD,
        lr_tuning=True,
        optimizer_type=optimizer_cfg["optimizer_type"],
        weight_decay=optimizer_cfg["weight_decay"],
        l2sp_lambda=optimizer_cfg["l2sp_lambda"],
        batch_size=eff_batch,
        save_folder=str(run_folder),
    )

    with torch.no_grad():
        final_test_mse = F.mse_loss(net_out(x_test), y_test).item()

    result = dict(
        setup=setup,
        omega=omega,
        lambda_pt=lambda_pt,
        rho_ft=rho_ft,
        optimizer=optimizer_cfg["name"],
        alpha_ft=alpha_ft,
        n_train=n_train,
        seed=seed,
        final_test_mse=final_test_mse,
        stop_reason=stop_reason,
        final_epoch=final_epoch,
    )

    with open(run_folder / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_id",
        type=int,
        required=True,
        help="SLURM_ARRAY_TASK_ID encoding (alpha_ft_idx, seed, condition_id)",
    )
    parser.add_argument(
        "--n_tasks",
        type=int,
        default=len(ALPHA_FT_VALUES) * N_SEEDS * len(ALL_CONDITIONS),
        help="Total number of array tasks (default: 924)",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        required=True,
        help="Root directory for all output folders",
    )
    args = parser.parse_args()

    n_alpha = len(ALPHA_FT_VALUES)
    n_conditions = len(ALL_CONDITIONS)
    expected_n_tasks = n_alpha * N_SEEDS * n_conditions

    if args.n_tasks != expected_n_tasks:
        print(
            f"[WARN] --n_tasks={args.n_tasks}, but expected {expected_n_tasks} "
            f"from {n_alpha} alpha values × {N_SEEDS} seeds × {n_conditions} conditions."
        )

    try:
        alpha_ft_idx, seed, condition_id, total_tasks = decode_task_id(args.task_id)
    except ValueError as e:
        print(str(e))
        return

    alpha_ft = ALPHA_FT_VALUES[alpha_ft_idx]
    condition = ALL_CONDITIONS[condition_id]

    Path(args.save_root).mkdir(parents=True, exist_ok=True)

    print()
    print(f"Task {args.task_id}/{total_tasks - 1}")
    print(
        f"Decoded as: alpha_ft_idx={alpha_ft_idx}, "
        f"alpha_ft={alpha_ft:.4f}, seed={seed}, condition_id={condition_id}"
    )
    print(
        f"Condition: setup={condition['setup']}, omega={condition['omega']}, "
        f"lambda_pt={condition['lambda_pt']:.6f}, rho_ft={condition['rho_ft']}"
    )
    print(f"Running {len(OPTIMIZERS)} optimizers for this condition")
    print()

    results = []
    for opt_cfg in OPTIMIZERS:
        r = run_single(alpha_ft, seed, condition, opt_cfg, args.save_root)
        results.append(r)
        print(
            f"  [{r['setup']} omega={r['omega']} lpt={r['lambda_pt']:.4f} "
            f"rft={r['rho_ft']} {r['optimizer']:>14}] "
            f"alpha_ft={alpha_ft:.3f} seed={seed}  "
            f"test_mse={r['final_test_mse']:.6f}  ({r['stop_reason']})"
        )

    print()
    print(f"Done. Results saved under {args.save_root}")


if __name__ == "__main__":
    main()