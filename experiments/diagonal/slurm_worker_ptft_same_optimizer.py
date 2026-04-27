#!/usr/bin/env python3
"""
SLURM worker: full PT+FT pipeline with the same optimizer for both phases.

Unlike slurm_worker_optimizer_sweep.py (which uses an oracle FT init),
this worker actually trains the PT network (alpha_pt=1.0) and uses the
learned per-coordinate c values to initialise the FT network directly:

    c_ft_i = w_pos_i * w_neg_i + v_pos_i * v_neg_i   (from trained PT net)

The same optimizer is used for both PT and FT, except that for the
Adam+L2SP condition the PT phase uses Adam+wd=1e-4 (L2SP is not meaningful
for PT where there are no pretrained reference weights).

Each SLURM array task is identified by a (alpha_ft_idx, seed) pair encoded
in SLURM_ARRAY_TASK_ID.  Within each task all experimental conditions are
run sequentially:

  Setup 1 — vary omega × lambda_pt  (rho_ft fixed)
      omega      in {0, 1}
      lambda_pt  in {-0.99*C_PT, 0, 0.99*C_PT}    → 6 conditions

  Setup 2 — vary rho_ft × lambda_pt  (omega=1 fixed)
      rho_ft     in {0.1, 0.01}
      lambda_pt  in {-0.99*C_PT, 0, 0.99*C_PT}    → 6 conditions

  Optimizers (3):
      SGD          lr=0.5, bs=min(32, n_train), momentum=0.9
      Adam + wd    lr=1e-3, weight_decay=1e-4
      Adam + L2SP  lr=1e-3, l2sp_lambda=1e-4  (FT only; PT uses Adam+wd)

Usage (local test):
    python experiments/diagonal/slurm_worker_ptft_same_optimizer.py \
        --task_id 0 --n_tasks 55 --save_root /tmp/ptft_test

SLURM submission (see submit_ptft_same_optimizer.sh):
    sbatch --array=0-54 submit_ptft_same_optimizer.sh
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
from experiments.diagonal.diagonal_ptft_oracle import (
    sample_pt_teacher, sample_ft_teacher_with_overlap,
)

# ── Fixed parameters ───────────────────────────────────────────────────────────
INP_DIM        = 1000
RHO_PT         = 0.10
RHO_FT_DEFAULT = 0.04
A_PT           = 1.0
C_PT           = 1e-3
GAMMA_FT       = 0.0
ALPHA_PT       = 1.0          # n_pt = 1000 = d
N_TEST         = 10000
EPOCHS         = 5_000_000
THRESHOLD      = 1e-4

# alpha_ft: 11 evenly spaced points in [0, 0.5]
ALPHA_FT_VALUES = list(np.linspace(0, 0.5, 11))
N_SEEDS         = 5

# ── Experimental conditions ────────────────────────────────────────────────────
# Setup 1: vary omega × lambda_pt  (rho_ft fixed at default)
SETUP1_CONDITIONS = [
    dict(omega=omega, lambda_pt=lam, rho_ft=RHO_FT_DEFAULT)
    for omega in [0, 1]
    for lam in [-0.99 * C_PT, 0.0, 0.99 * C_PT]
]
# Setup 2: vary rho_ft × lambda_pt  (omega=1 fixed)
SETUP2_CONDITIONS = [
    dict(omega=1, lambda_pt=lam, rho_ft=rho_ft)
    for rho_ft in [0.1, 0.01]
    for lam in [-0.99 * C_PT, 0.0, 0.99 * C_PT]
]
ALL_CONDITIONS = SETUP1_CONDITIONS + SETUP2_CONDITIONS   # 12 conditions total

# ── Optimizers ─────────────────────────────────────────────────────────────────
# Each entry defines the FT config.  pt_weight_decay overrides weight_decay for PT.
OPTIMIZERS = [
    dict(name='sgd_bs32',
         optimizer_type='sgd',  lr=0.5,  weight_decay=0.0,  l2sp_lambda=0.0,  batch_size=32,
         pt_weight_decay=0.0),
    dict(name='adam_wd1e4',
         optimizer_type='adam', lr=1e-3, weight_decay=1e-4, l2sp_lambda=0.0,  batch_size=None,
         pt_weight_decay=1e-4),
    dict(name='adam_l2sp1e4',
         optimizer_type='adam', lr=1e-3, weight_decay=0.0,  l2sp_lambda=1e-4, batch_size=None,
         pt_weight_decay=1e-4),   # PT uses Adam+wd=1e-4 (L2SP not meaningful for PT)
]


def run_single(alpha_ft, seed, condition, optimizer_cfg, save_root):
    """Run one (alpha_ft, seed, condition, optimizer) PT+FT pair."""
    make_deterministic(seed)
    torch.set_default_dtype(torch.float64)

    omega     = condition['omega']
    lambda_pt = condition['lambda_pt']
    rho_ft    = condition['rho_ft']

    n_pt    = max(1, int(round(ALPHA_PT * INP_DIM)))
    n_ft    = max(1, int(round(alpha_ft * INP_DIM)))

    # Separate generators for reproducibility
    gen_teacher_pt  = torch.Generator().manual_seed(seed * 10 + 0)
    gen_teacher_ft  = torch.Generator().manual_seed(seed * 10 + 1)
    gen_pt_train_x  = torch.Generator().manual_seed(seed * 10 + 2)
    gen_pt_test_x   = torch.Generator().manual_seed(seed * 10 + 3)
    gen_ft_train_x  = torch.Generator().manual_seed(seed * 10 + 4)
    gen_ft_test_x   = torch.Generator().manual_seed(seed * 10 + 5)

    # ── Teachers ───────────────────────────────────────────────────────────────
    beta_pt, support_pt = sample_pt_teacher(INP_DIM, RHO_PT, A_PT, gen_teacher_pt)
    beta_ft, _ = sample_ft_teacher_with_overlap(INP_DIM, rho_ft, omega, support_pt, gen_teacher_ft)

    # ── PT data ────────────────────────────────────────────────────────────────
    x_pt_train = torch.randn(n_pt, INP_DIM, generator=gen_pt_train_x) / math.sqrt(INP_DIM)
    x_pt_test  = torch.randn(N_TEST, INP_DIM, generator=gen_pt_test_x)  / math.sqrt(INP_DIM)
    y_pt_train = x_pt_train @ beta_pt
    y_pt_test  = x_pt_test  @ beta_pt

    # ── PT training ────────────────────────────────────────────────────────────
    pt_bs       = optimizer_cfg['batch_size']
    pt_eff_batch = None if pt_bs is None else min(int(pt_bs), n_pt)
    pt_momentum  = 0.9 if optimizer_cfg['optimizer_type'] == 'sgd' else 0.0

    pt_net = DiagonalNet(INP_DIM, scaling=1.0, lmda=lambda_pt, c=C_PT, init_method='complex')

    _, pt_net, _, pt_stop_reason, pt_final_epoch = train(
        pt_net,
        (x_pt_train, y_pt_train),
        (x_pt_test,  y_pt_test),
        beta_pt,
        lr=optimizer_cfg['lr'],
        momentum=pt_momentum,
        epochs=EPOCHS,
        threshold=THRESHOLD,
        lr_tuning=True,
        optimizer_type=optimizer_cfg['optimizer_type'],
        weight_decay=optimizer_cfg['pt_weight_decay'],  # wd for PT (no L2SP)
        l2sp_lambda=0.0,
        batch_size=pt_eff_batch,
        save_folder=None,   # skip per-epoch saves for PT to keep disk usage low
    )

    with torch.no_grad():
        pt_param_mse = F.mse_loss(pt_net.beta(), beta_pt).item()
        # Extract per-coordinate c values from trained PT network
        c_ft = (pt_net.w_pos * pt_net.w_neg + pt_net.v_pos * pt_net.v_neg).numpy()

    # ── FT data ────────────────────────────────────────────────────────────────
    x_ft_train = torch.randn(n_ft, INP_DIM, generator=gen_ft_train_x) / math.sqrt(INP_DIM)
    x_ft_test  = torch.randn(N_TEST, INP_DIM, generator=gen_ft_test_x)  / math.sqrt(INP_DIM)
    y_ft_train = x_ft_train @ beta_ft
    y_ft_test  = x_ft_test  @ beta_ft

    # ── FT init from learned PT c values ──────────────────────────────────────
    ft_net = DiagonalNet(INP_DIM, scaling=1.0, lmda=0.0, c=C_PT, c_vec=c_ft, init_method='complex')

    ft_bs        = optimizer_cfg['batch_size']
    ft_eff_batch = None if ft_bs is None else min(int(ft_bs), n_ft)
    ft_momentum  = 0.9 if optimizer_cfg['optimizer_type'] == 'sgd' else 0.0

    cond_str = (f"omega={omega}--lpt={lambda_pt:.6f}--rft={rho_ft}"
                f"--opt={optimizer_cfg['name']}"
                f"--apt={ALPHA_PT}--aft={alpha_ft:.4f}--seed={seed}")
    run_folder = Path(save_root) / cond_str
    run_folder.mkdir(parents=True, exist_ok=True)

    _, ft_net, _, ft_stop_reason, ft_final_epoch = train(
        ft_net,
        (x_ft_train, y_ft_train),
        (x_ft_test,  y_ft_test),
        beta_ft,
        lr=optimizer_cfg['lr'],
        momentum=ft_momentum,
        epochs=EPOCHS,
        threshold=THRESHOLD,
        lr_tuning=True,
        optimizer_type=optimizer_cfg['optimizer_type'],
        weight_decay=optimizer_cfg['weight_decay'],
        l2sp_lambda=optimizer_cfg['l2sp_lambda'],
        batch_size=ft_eff_batch,
        save_folder=str(run_folder),
    )

    with torch.no_grad():
        final_test_mse = F.mse_loss(ft_net(x_ft_test), y_ft_test).item()

    result = dict(
        omega=omega, lambda_pt=lambda_pt, rho_ft=rho_ft,
        c_pt=C_PT, gamma_ft=GAMMA_FT, alpha_pt=ALPHA_PT,
        optimizer=optimizer_cfg['name'], alpha_ft=alpha_ft, n_train=n_ft,
        seed=seed,
        final_test_mse=final_test_mse,
        ft_stop_reason=ft_stop_reason, ft_final_epoch=ft_final_epoch,
        pt_param_mse=pt_param_mse,
        pt_stop_reason=pt_stop_reason, pt_final_epoch=pt_final_epoch,
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

    n_alpha      = len(ALPHA_FT_VALUES)
    alpha_ft_idx = args.task_id % n_alpha
    seed         = args.task_id // n_alpha

    if seed >= N_SEEDS:
        print(f"task_id {args.task_id} out of range — nothing to do.")
        return

    alpha_ft = ALPHA_FT_VALUES[alpha_ft_idx]
    Path(args.save_root).mkdir(parents=True, exist_ok=True)

    print(f"\nTask {args.task_id}: alpha_ft={alpha_ft:.4f}, seed={seed}")
    print(f"Running {len(ALL_CONDITIONS)} conditions × {len(OPTIMIZERS)} optimizers "
          f"= {len(ALL_CONDITIONS) * len(OPTIMIZERS)} PT+FT pairs\n")

    for condition in ALL_CONDITIONS:
        for opt_cfg in OPTIMIZERS:
            r = run_single(alpha_ft, seed, condition, opt_cfg, args.save_root)
            print(f"  [omega={r['omega']} lpt={r['lambda_pt']:.6f} "
                  f"rft={r['rho_ft']} {r['optimizer']:>14}] "
                  f"alpha_ft={alpha_ft:.3f} seed={seed}  "
                  f"test_mse={r['final_test_mse']:.6f}  "
                  f"pt_param_mse={r['pt_param_mse']:.6f}  "
                  f"({r['ft_stop_reason']})")

    print(f"\nDone. Results saved under {args.save_root}")


if __name__ == '__main__':
    main()
