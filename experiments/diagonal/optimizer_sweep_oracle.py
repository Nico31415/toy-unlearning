#!/usr/bin/env python3
"""
Preliminary comparison: full_batch vs sgd vs adam (multiple lrs)
in the oracle FT setting (full overlap, omega=1).

Usage:
    python experiments/diagonal/optimizer_sweep_oracle.py \
        --save_folder /tmp/optimizer_sweep --seeds 3
"""

import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import argparse
import math
import sys
from pathlib import Path
sys.path.append('')

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from experiments.diagonal.diagonal_network_pretrain_bg import (
    DiagonalNet, train, make_deterministic, get_parameters_vectorized,
)
from ReplicaExperiments.fixed_lambda_all import compute_c_ft_from_pt
from experiments.diagonal.diagonal_ptft_oracle import (
    sample_pt_teacher, sample_ft_teacher_with_overlap,
)


# ── Fixed experiment parameters ──────────────────────────────────────────────
INP_DIM    = 1000
RHO_PT     = 0.10
RHO_FT     = 0.04
A_PT       = 1.0
C_PT       = 0.001
LAMBDA_PT  = 0.0
GAMMA_REINIT = 0.0
OMEGA      = 1.0          # full overlap
N_TEST     = 10000
EPOCHS     = 5_000_000
THRESHOLD  = 1e-12

ALPHA_VALUES_DEFAULT = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
ADAM_LRS_DEFAULT     = [1e-4, 1e-3, 1e-2]
ADAM_WDS_DEFAULT     = [1e-4, 1e-3, 1e-2, 0.1]
SGD_MOMENTUM         = 0.9


def run_single(alpha, optimizer_type, lr, seed, save_folder, epochs=EPOCHS, threshold=THRESHOLD,
               weight_decay=0.0, l2sp_lambda=0.0, batch_size=None):
    make_deterministic(seed)
    torch.set_default_dtype(torch.float64)

    n_train = max(1, int(round(alpha * INP_DIM)))

    gen_pt       = torch.Generator().manual_seed(seed + 0)
    gen_ft       = torch.Generator().manual_seed(seed + 1)
    gen_train_x  = torch.Generator().manual_seed(seed + 2)
    gen_test_x   = torch.Generator().manual_seed(seed + 3)

    beta_pt, support_pt = sample_pt_teacher(INP_DIM, RHO_PT, A_PT, gen_pt)
    beta_ft, _ = sample_ft_teacher_with_overlap(
        INP_DIM, RHO_FT, OMEGA, support_pt, gen_ft
    )

    c_ft = compute_c_ft_from_pt(beta_pt.numpy(), C_PT, LAMBDA_PT, GAMMA_REINIT)

    x_train = torch.randn(n_train, INP_DIM, generator=gen_train_x) / math.sqrt(INP_DIM)
    x_test  = torch.randn(N_TEST,   INP_DIM, generator=gen_test_x)  / math.sqrt(INP_DIM)
    y_train = x_train @ beta_ft
    y_test  = x_test  @ beta_ft

    net = DiagonalNet(INP_DIM, scaling=1.0, lmda=0.0, c=C_PT, c_vec=c_ft, init_method='complex')
    momentum = SGD_MOMENTUM if optimizer_type == 'sgd' else 0.0

    # Effective batch size: min(batch_size, n_train) — None means full batch
    eff_batch = None if batch_size is None else min(int(batch_size), n_train)

    if l2sp_lambda > 0.0:
        label = f"adam_l2sp={l2sp_lambda}"
    elif weight_decay > 0.0:
        label = f"adam_wd={weight_decay}"
    elif optimizer_type == 'sgd' and eff_batch is not None and eff_batch < n_train:
        label = f"sgd_bs={eff_batch}"
    elif optimizer_type == 'adam':
        label = f"adam"
    else:
        label = optimizer_type

    run_folder = os.path.join(
        save_folder, f"opt={label}--alpha={alpha}--seed={seed}"
    )
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    df, net, norm_df, stop_reason, final_epoch = train(
        net,
        (x_train, y_train),
        (x_test, y_test),
        beta_ft,
        lr=lr,
        momentum=momentum,
        epochs=epochs,
        threshold=threshold,
        lr_tuning=True,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
        l2sp_lambda=l2sp_lambda,
        batch_size=eff_batch,
        save_folder=run_folder,
    )

    with torch.no_grad():
        final_test_mse = F.mse_loss(net(x_test), y_test).item()

    return {
        'label':          label,
        'optimizer':      optimizer_type,
        'lr':             lr,
        'weight_decay':   weight_decay,
        'l2sp_lambda':    l2sp_lambda,
        'batch_size':     eff_batch,
        'alpha':          alpha,
        'n_train':        n_train,
        'seed':           seed,
        'final_test_mse': final_test_mse,
        'stop_reason':    stop_reason,
        'final_epoch':    final_epoch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--alphas',    type=float, nargs='+', default=ALPHA_VALUES_DEFAULT)
    parser.add_argument('--optimizers', type=str, nargs='+', default=['full_batch', 'sgd', 'adam'])
    parser.add_argument('--adam_lrs', type=float, nargs='+', default=ADAM_LRS_DEFAULT)
    parser.add_argument('--adam_wds', type=float, nargs='+', default=None,
                        help='If set, run Adam only with lr=1e-3 at these weight_decay values')
    parser.add_argument('--adam_l2sp', type=float, nargs='+', default=None,
                        help='If set, run Adam only with lr=1e-3 at these l2sp_lambda values')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Mini-batch size for SGD (None = full batch). Effective size is min(batch_size, n_train).')
    parser.add_argument('--clean_sweep', action='store_true',
                        help='Run the 4-condition clean sweep: SGD bs=32, Adam, Adam+wd=1e-4, Adam+L2SP=1e-4')
    parser.add_argument('--seeds',    type=int, default=3)
    parser.add_argument('--epochs',   type=int, default=EPOCHS)
    parser.add_argument('--threshold',type=float, default=THRESHOLD)
    args = parser.parse_args()

    Path(args.save_folder).mkdir(parents=True, exist_ok=True)

    # Build list of (optimizer_type, lr, weight_decay, l2sp_lambda, batch_size) conditions
    if args.clean_sweep:
        # 4-condition comparison: SGD bs=32, Adam, Adam+wd, Adam+L2SP
        conditions = [
            ('sgd',  0.5,  0.0,  0.0,  32),
            ('adam', 1e-3, 0.0,  0.0,  None),
            ('adam', 1e-3, 1e-4, 0.0,  None),
            ('adam', 1e-3, 0.0,  1e-4, None),
        ]
    elif args.adam_l2sp is not None:
        conditions = [('adam', 1e-3, 0.0, lam, None) for lam in args.adam_l2sp]
    elif args.adam_wds is not None:
        conditions = [('adam', 1e-3, wd, 0.0, None) for wd in args.adam_wds]
    else:
        base = []
        if 'full_batch' in args.optimizers:
            base.append(('full_batch', 0.5, 0.0, 0.0, None))
        if 'sgd' in args.optimizers:
            base.append(('sgd', 0.5, 0.0, 0.0, args.batch_size))
        if 'adam' in args.optimizers:
            base += [('adam', lr, 0.0, 0.0, None) for lr in args.adam_lrs]
        conditions = base

    seeds        = list(range(args.seeds))
    alpha_values = args.alphas
    total        = len(conditions) * len(alpha_values) * len(seeds)

    print(f"\nRunning {total} conditions "
          f"({len(conditions)} optimizer/lr combos × {len(alpha_values)} alphas × {len(seeds)} seeds)\n")
    print(f"{'label':>25}  {'alpha':>6}  {'seed':>4}  {'test_mse':>12}  stop_reason")
    print("-" * 70)

    rows = []
    for opt, lr, wd, l2sp, bs in conditions:
        for alpha in alpha_values:
            for seed in seeds:
                row = run_single(
                    alpha, opt, lr, seed,
                    args.save_folder,
                    epochs=args.epochs,
                    threshold=args.threshold,
                    weight_decay=wd,
                    l2sp_lambda=l2sp,
                    batch_size=bs,
                )
                rows.append(row)
                print(f"{row['label']:>25}  {row['alpha']:>6.2f}  {row['seed']:>4d}  "
                      f"{row['final_test_mse']:>12.6f}  {row['stop_reason']}")

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(args.save_folder, 'optimizer_comparison.csv')
    results_df.to_csv(csv_path, index=False)

    print("\n\n── Summary (mean ± std over seeds) ──────────────────────────")
    print(f"{'label':>20}  {'alpha':>6}  {'mean_mse':>12}  {'std_mse':>10}")
    print("-" * 55)
    summary = results_df.groupby(['label', 'alpha'])['final_test_mse'].agg(['mean', 'std'])
    for (label, alpha), row in summary.iterrows():
        print(f"{label:>20}  {alpha:>6.2f}  {row['mean']:>12.6f}  {row['std']:>10.6f}")

    print(f"\nFull results saved to {csv_path}")


if __name__ == '__main__':
    main()
