#!/usr/bin/env python3
"""
SLURM worker: optimizer comparison (SGD, Adam+wd, Adam+L2SP) for diagonal networks.

Each SLURM array task is identified by a (alpha_ft_idx, seed) pair encoded in
SLURM_ARRAY_TASK_ID.  Within each task all experimental conditions are run
sequentially:

  Setup 1 — vary omega and lambda_pt
      omega      in {0, 1}
      lambda_pt  in {-0.99*C_PT, 0, 0.99*C_PT}

  Setup 2 — vary rho_ft  (omega=1 fixed)
      rho_ft     in {0.1, 0.01}

  Optimizers (3)
      SGD          lr=0.5, bs=min(32, n_train), momentum=0.9
      Adam + wd    lr=1e-3, weight_decay=1e-4
      Adam + L2SP  lr=1e-3, l2sp_lambda=1e-4

Usage (local test):
    python experiments/diagonal/slurm_worker_optimizer_sweep.py \
        --task_id 0 --n_tasks 55 --save_root /tmp/slurm_test

SLURM submission example (see submit_optimizer_sweep.sh):
    sbatch --array=0-54 submit_optimizer_sweep.sh
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
INP_DIM      = 1000
RHO_PT       = 0.10
RHO_FT_DEFAULT = 0.04
A_PT         = 1.0
C_PT         = 1e-3
GAMMA_FT     = 0.0
N_TEST       = 10000
EPOCHS       = 5_000_000
THRESHOLD    = 1e-4       # as specified

# alpha_ft: 11 evenly spaced points in [0, 0.5]
ALPHA_FT_VALUES = list(np.linspace(0, 0.5, 11))   # 0.0, 0.05, ..., 0.5
N_SEEDS          = 5                                # seeds 0..4

# ── Experimental conditions ───────────────────────────────────────────────────
# Setup 1: vary omega × lambda_pt
SETUP1_CONDITIONS = [
    dict(setup='setup1', omega=omega, lambda_pt=lam, rho_ft=RHO_FT_DEFAULT)
    for omega in [0, 1]
    for lam in [-0.99 * C_PT, 0.0, 0.99 * C_PT]
]
# Setup 2: vary rho_ft (omega=1 fixed)
SETUP2_CONDITIONS = [
    dict(setup='setup2', omega=1, lambda_pt=0.0, rho_ft=rho_ft)
    for rho_ft in [0.1, 0.01]
]
ALL_CONDITIONS = SETUP1_CONDITIONS + SETUP2_CONDITIONS   # 8 conditions total

# ── Optimizers ────────────────────────────────────────────────────────────────
OPTIMIZERS = [
    dict(name='sgd_bs32',    optimizer_type='sgd',  lr=0.5,  weight_decay=0.0,  l2sp_lambda=0.0,  batch_size=32),
    dict(name='adam_wd1e4',  optimizer_type='adam', lr=1e-3, weight_decay=1e-4, l2sp_lambda=0.0,  batch_size=None),
    dict(name='adam_l2sp1e4',optimizer_type='adam', lr=1e-3, weight_decay=0.0,  l2sp_lambda=1e-4, batch_size=None),
]


def run_single(alpha_ft, seed, condition, optimizer_cfg, save_root):
    """Run one (alpha_ft, seed, condition, optimizer) combination."""
    make_deterministic(seed)
    torch.set_default_dtype(torch.float64)

    omega     = condition['omega']
    lambda_pt = condition['lambda_pt']
    rho_ft    = condition['rho_ft']
    setup     = condition['setup']

    n_train = max(1, int(round(alpha_ft * INP_DIM)))

    gen_pt      = torch.Generator().manual_seed(seed * 10 + 0)
    gen_ft      = torch.Generator().manual_seed(seed * 10 + 1)
    gen_train_x = torch.Generator().manual_seed(seed * 10 + 2)
    gen_test_x  = torch.Generator().manual_seed(seed * 10 + 3)

    # Teachers
    beta_pt, support_pt = sample_pt_teacher(INP_DIM, RHO_PT, A_PT, gen_pt)
    beta_ft, _ = sample_ft_teacher_with_overlap(INP_DIM, rho_ft, omega, support_pt, gen_ft)

    # Oracle FT init
    c_ft = compute_c_ft_from_pt(beta_pt.numpy(), C_PT, lambda_pt, GAMMA_FT)

    # Data
    x_train = torch.randn(n_train, INP_DIM, generator=gen_train_x) / math.sqrt(INP_DIM)
    x_test  = torch.randn(N_TEST,  INP_DIM, generator=gen_test_x)  / math.sqrt(INP_DIM)
    y_train = x_train @ beta_ft
    y_test  = x_test  @ beta_ft

    # Model
    net = DiagonalNet(INP_DIM, scaling=1.0, lmda=0.0, c=C_PT, c_vec=c_ft, init_method='complex')

    # Batch size: min(32, n_train) for SGD
    bs = optimizer_cfg['batch_size']
    eff_batch = None if bs is None else min(int(bs), n_train)
    momentum  = 0.9 if optimizer_cfg['optimizer_type'] == 'sgd' else 0.0

    # Folder name encodes all hyperparams
    cond_str = (f"{setup}--omega={omega}--lpt={lambda_pt:.4f}"
                f"--rft={rho_ft}--opt={optimizer_cfg['name']}"
                f"--aft={alpha_ft:.4f}--seed={seed}")
    run_folder = Path(save_root) / cond_str
    run_folder.mkdir(parents=True, exist_ok=True)

    _, net_out, _, stop_reason, final_epoch = train(
        net,
        (x_train, y_train),
        (x_test,  y_test),
        beta_ft,
        lr=optimizer_cfg['lr'],
        momentum=momentum,
        epochs=EPOCHS,
        threshold=THRESHOLD,
        lr_tuning=True,
        optimizer_type=optimizer_cfg['optimizer_type'],
        weight_decay=optimizer_cfg['weight_decay'],
        l2sp_lambda=optimizer_cfg['l2sp_lambda'],
        batch_size=eff_batch,
        save_folder=str(run_folder),
    )

    with torch.no_grad():
        final_test_mse = F.mse_loss(net_out(x_test), y_test).item()

    result = dict(
        setup=setup, omega=omega, lambda_pt=lambda_pt, rho_ft=rho_ft,
        optimizer=optimizer_cfg['name'], alpha_ft=alpha_ft, n_train=n_train,
        seed=seed, final_test_mse=final_test_mse,
        stop_reason=stop_reason, final_epoch=final_epoch,
    )
    with open(run_folder / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True,
                        help='SLURM_ARRAY_TASK_ID (encodes alpha_ft_idx × seed)')
    parser.add_argument('--n_tasks', type=int,
                        default=len(ALPHA_FT_VALUES) * N_SEEDS,
                        help='Total number of array tasks (default: 55)')
    parser.add_argument('--save_root', type=str, required=True,
                        help='Root directory for all output folders')
    args = parser.parse_args()

    # Decode task_id → (alpha_ft_idx, seed)
    n_alpha = len(ALPHA_FT_VALUES)
    alpha_ft_idx = args.task_id % n_alpha
    seed         = args.task_id // n_alpha

    if seed >= N_SEEDS:
        print(f"task_id {args.task_id} out of range — nothing to do.")
        return

    alpha_ft = ALPHA_FT_VALUES[alpha_ft_idx]
    Path(args.save_root).mkdir(parents=True, exist_ok=True)

    print(f"\nTask {args.task_id}: alpha_ft={alpha_ft:.4f}, seed={seed}")
    print(f"Running {len(ALL_CONDITIONS)} conditions × {len(OPTIMIZERS)} optimizers "
          f"= {len(ALL_CONDITIONS) * len(OPTIMIZERS)} runs\n")

    results = []
    for condition in ALL_CONDITIONS:
        for opt_cfg in OPTIMIZERS:
            r = run_single(alpha_ft, seed, condition, opt_cfg, args.save_root)
            results.append(r)
            print(f"  [{r['setup']} omega={r['omega']} lpt={r['lambda_pt']:.4f} "
                  f"rft={r['rho_ft']} {r['optimizer']:>14}] "
                  f"alpha_ft={alpha_ft:.3f} seed={seed}  "
                  f"test_mse={r['final_test_mse']:.6f}  ({r['stop_reason']})")

    print(f"\nDone. Results saved under {args.save_root}")


if __name__ == '__main__':
    main()
