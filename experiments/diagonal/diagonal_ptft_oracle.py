#!/usr/bin/env python3
"""
Step 3: Empirical PT+FT Oracle Experiment

Implements the PT→FT oracle mechanism empirically:
1. Constructs deterministic PT teacher beta_pt with support S_pt, amplitude a_pt
2. Constructs stochastic FT teacher beta_ft (BG) with controlled overlap omega
3. Computes per-coordinate FT init c_ft from Cosyne formula:
   c_ft_i = (lambda_pt + c_pt) * (1 + sqrt(1 + (beta_pt_i/c_pt)^2)) + gamma_reinit^2/2
4. Initializes diagonal net with per-coordinate c_ft
5. Trains on FT data and records param MSE vs alpha

The oracle version skips PT training and directly uses beta_pt for the mapping.
"""

import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

from copy import deepcopy
import argparse
import math
import sys
from pathlib import Path
sys.path.append('')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import random

# Import DiagonalNet from existing script (to reuse the heterogeneous init)
from experiments.diagonal.diagonal_network_pretrain_bg import (
    DiagonalNet, train, l1_norm, l2_norm, make_deterministic,
    get_parameters_vectorized,
)

# Import canonical Cosyne PT->FT mapping (single source of truth)
from ReplicaExperiments.fixed_lambda_all import compute_c_ft_from_pt


def sample_pt_teacher(inp_dim, rho_pt, a_pt, generator):
    """
    Sample deterministic PT teacher with support fraction rho_pt and amplitude a_pt.
    
    Args:
        inp_dim: Input dimension
        rho_pt: PT support fraction
        a_pt: Amplitude on PT support (constant)
        generator: Torch generator for reproducibility
    
    Returns:
        beta_pt: Teacher vector of shape (inp_dim,)
        support_pt: Boolean mask indicating PT support
    """
    n_active = int(round(rho_pt * inp_dim))
    
    # Random permutation to select support
    perm = torch.randperm(inp_dim, generator=generator)
    support_indices = perm[:n_active]
    
    support_pt = torch.zeros(inp_dim, dtype=torch.bool)
    support_pt[support_indices] = True
    
    # Deterministic amplitude on support
    beta_pt = torch.zeros(inp_dim, dtype=torch.float64)
    beta_pt[support_pt] = a_pt
    
    return beta_pt, support_pt


def sample_ft_teacher_with_overlap(inp_dim, rho_ft, omega, support_pt, generator):
    """
    Sample stochastic FT teacher (BG) with controlled overlap with PT support.
    
    The overlap fraction omega = |S_pt ∩ S_ft| / |S_ft| controls how much
    the FT support overlaps with the PT support.
    
    Args:
        inp_dim: Input dimension
        rho_ft: FT teacher sparsity
        omega: Overlap fraction (0 = no overlap, 1 = full overlap)
        support_pt: Boolean mask for PT support
        generator: Torch generator for reproducibility
    
    Returns:
        beta_ft: FT teacher vector of shape (inp_dim,)
        support_ft: Boolean mask for FT support
    """
    n_ft_active = int(round(rho_ft * inp_dim))
    n_overlap = int(round(omega * n_ft_active))
    n_new = n_ft_active - n_overlap
    
    # Get PT support indices
    pt_indices = torch.where(support_pt)[0]
    non_pt_indices = torch.where(~support_pt)[0]
    
    # Validate feasibility
    if n_overlap > len(pt_indices):
        raise ValueError(
            f"Cannot achieve omega={omega}: need {n_overlap} overlap indices "
            f"but PT support has only {len(pt_indices)} indices"
        )
    if n_new > len(non_pt_indices):
        raise ValueError(
            f"Cannot achieve omega={omega}: need {n_new} new indices "
            f"but only {len(non_pt_indices)} non-PT indices available"
        )
    
    # Sample overlap indices from PT support
    perm_pt = torch.randperm(len(pt_indices), generator=generator)
    overlap_indices = pt_indices[perm_pt[:n_overlap]]
    
    # Sample new indices from non-PT coordinates
    perm_non_pt = torch.randperm(len(non_pt_indices), generator=generator)
    new_indices = non_pt_indices[perm_non_pt[:n_new]]
    
    # Build FT support
    support_ft = torch.zeros(inp_dim, dtype=torch.bool)
    support_ft[overlap_indices] = True
    support_ft[new_indices] = True
    
    # Sample BG values on support: N(0, 1/rho_ft)
    beta_ft = torch.zeros(inp_dim, dtype=torch.float64)
    n_ft_nonzero = support_ft.sum().item()
    gaussian_vals = torch.randn(n_ft_nonzero, generator=generator, dtype=torch.float64) / math.sqrt(rho_ft)
    beta_ft[support_ft] = gaussian_vals
    
    return beta_ft, support_ft


def main(args):
    print("\n" + "="*80)
    print("STEP 3: PT+FT ORACLE EXPERIMENT")
    print("="*80 + "\n")
    sys.stdout.flush()
    
    make_deterministic(args.seed, use_gpu=False)
    torch.set_default_dtype(torch.float64)
    
    print("="*80)
    print("EXPERIMENT SETTINGS")
    print("="*80)
    print(f"Seed: {args.seed}")
    print(f"Input dimension: {args.inp_dim}")
    print(f"PT support fraction (rho_pt): {args.rho_pt}")
    print(f"FT sparsity (rho_ft): {args.rho_ft}")
    print(f"Overlap (omega): {args.omega}")
    print(f"PT amplitude (a_pt): {args.a_pt}")
    print(f"PT c parameter (c_pt): {args.c_pt}")
    print(f"PT lambda (lambda_pt): {args.lambda_pt}")
    print(f"Gamma reinit: {args.gamma_reinit}")
    print(f"Training samples (FT): {args.n_train}")
    print(f"Test samples: {args.n_test}")
    print(f"Learning rate: {args.lr}")
    print(f"Max epochs: {args.epochs}")
    print(f"Save folder: {args.save_folder}")
    print("="*80)
    sys.stdout.flush()
    
    # Generators for reproducibility
    gen_pt = torch.Generator(device='cpu').manual_seed(args.seed + 0)
    gen_ft = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    gen_train_x = torch.Generator(device='cpu').manual_seed(args.seed + 2)
    gen_test_x = torch.Generator(device='cpu').manual_seed(args.seed + 3)
    
    Path(args.save_folder).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Sample PT teacher (deterministic, oracle)
    print("\nSampling PT teacher (oracle)...")
    beta_pt, support_pt = sample_pt_teacher(args.inp_dim, args.rho_pt, args.a_pt, gen_pt)
    n_pt_active = support_pt.sum().item()
    print(f"  PT support size: {n_pt_active} / {args.inp_dim} = {n_pt_active/args.inp_dim:.4f}")
    
    # Step 2: Sample FT teacher with controlled overlap
    print("\nSampling FT teacher with overlap...")
    beta_ft, support_ft = sample_ft_teacher_with_overlap(
        args.inp_dim, args.rho_ft, args.omega, support_pt, gen_ft
    )
    n_ft_active = support_ft.sum().item()
    n_overlap = (support_pt & support_ft).sum().item()
    empirical_omega = n_overlap / n_ft_active if n_ft_active > 0 else 0
    print(f"  FT support size: {n_ft_active} / {args.inp_dim} = {n_ft_active/args.inp_dim:.4f}")
    print(f"  Overlap: {n_overlap} / {n_ft_active} = {empirical_omega:.4f} (target: {args.omega})")
    
    # Analyze coordinate groups
    ov_mask = support_pt & support_ft
    new_mask = ~support_pt & support_ft
    ptonly_mask = support_pt & ~support_ft
    none_mask = ~support_pt & ~support_ft
    
    print(f"\n  Coordinate groups:")
    print(f"    OV (both):     {ov_mask.sum().item()}")
    print(f"    NEW (FT only): {new_mask.sum().item()}")
    print(f"    PTONLY:        {ptonly_mask.sum().item()}")
    print(f"    NONE:          {none_mask.sum().item()}")
    
    # Step 3: Compute per-coordinate c_ft from Cosyne formula (using canonical mapping)
    print("\nComputing c_ft from Cosyne formula...")
    c_ft = compute_c_ft_from_pt(
        beta_pt.numpy(),
        args.c_pt,
        args.lambda_pt,
        args.gamma_reinit
    )
    
    # Analyze c_ft distribution by group
    print(f"\n  c_ft statistics by group:")
    print(f"    OV:     mean={c_ft[ov_mask.numpy()].mean():.6f}, range=[{c_ft[ov_mask.numpy()].min():.6f}, {c_ft[ov_mask.numpy()].max():.6f}]" if ov_mask.sum() > 0 else "    OV:     N/A")
    print(f"    NEW:    mean={c_ft[new_mask.numpy()].mean():.6f} (beta_pt=0 -> baseline c_ft)" if new_mask.sum() > 0 else "    NEW:    N/A")
    print(f"    PTONLY: mean={c_ft[ptonly_mask.numpy()].mean():.6f}" if ptonly_mask.sum() > 0 else "    PTONLY: N/A")
    print(f"    NONE:   mean={c_ft[none_mask.numpy()].mean():.6f}" if none_mask.sum() > 0 else "    NONE:   N/A")
    
    # Step 4: Sample FT training/test data
    print("\nSampling FT data...")
    x_train = torch.randn(args.n_train, args.inp_dim, generator=gen_train_x) / math.sqrt(args.n_train)
    x_test = torch.randn(args.n_test, args.inp_dim, generator=gen_test_x) / math.sqrt(args.n_test)
    
    y_train = x_train @ beta_ft
    y_test = x_test @ beta_ft
    
    # Step 5: Initialize diagonal net with per-coordinate c_ft
    print("\nInitializing DiagonalNet with per-coordinate c_ft...")
    net = DiagonalNet(
        args.inp_dim, 
        scaling=1.0, 
        lmda=0.0, 
        c=args.c_pt,  # fallback (unused)
        c_vec=c_ft,   # per-coordinate init
        init_method='complex'
    )
    
    # Step 6: Train on FT data
    print("\nTraining on FT data...")
    df, net, norm_df, stop_reason, final_epoch = train(
        net,
        (x_train, y_train),
        (x_test, y_test),
        beta_ft,  # FT teacher is the ground truth
        test_every_n_epochs=args.test_every_n_epochs,
        lr=args.lr,
        epochs=args.epochs,
        lr_tuning=(not args.no_tuning),
        threshold=args.threshold,
        stop_pred_mse=args.stop_pred_mse,
        stop_beta_rate=args.stop_beta_rate,
        stop_grad_norm=args.stop_grad_norm,
        lr_decay=args.lr_decay,
        lr_decay_interval=args.lr_decay_interval,
        save_folder=args.save_folder
    )
    
    # Save results
    df.to_feather(os.path.join(args.save_folder, 'df.feather'))
    norm_df.to_feather(os.path.join(args.save_folder, 'norm_df.feather'))
    torch.save(beta_pt, os.path.join(args.save_folder, 'beta_pt.pt'))
    torch.save(beta_ft, os.path.join(args.save_folder, 'beta_ft.pt'))
    torch.save(net.state_dict(), os.path.join(args.save_folder, 'model.pt'))
    np.save(os.path.join(args.save_folder, 'c_ft.npy'), c_ft)
    
    # Save group masks
    np.save(os.path.join(args.save_folder, 'mask_ov.npy'), ov_mask.numpy())
    np.save(os.path.join(args.save_folder, 'mask_new.npy'), new_mask.numpy())
    np.save(os.path.join(args.save_folder, 'mask_ptonly.npy'), ptonly_mask.numpy())
    np.save(os.path.join(args.save_folder, 'mask_none.npy'), none_mask.numpy())
    
    # Save group diagnostics CSV
    group_stats = []
    for name, mask in [('OV', ov_mask), ('NEW', new_mask), ('PTONLY', ptonly_mask), ('NONE', none_mask)]:
        count = int(mask.sum().item())
        frac = count / args.inp_dim
        mean_c_ft = float(c_ft[mask.numpy()].mean()) if count > 0 else np.nan
        group_stats.append({
            'group': name,
            'count': count,
            'fraction': frac,
            'mean_c_ft': mean_c_ft
        })
    group_df = pd.DataFrame(group_stats)
    group_csv_path = os.path.join(args.save_folder, 'group_diagnostics.csv')
    group_df.to_csv(group_csv_path, index=False)
    print(f"Group diagnostics saved to {group_csv_path}")
    
    # Save config
    config = {
        'seed': args.seed,
        'inp_dim': args.inp_dim,
        'n_train': args.n_train,
        'n_test': args.n_test,
        'rho_pt': args.rho_pt,
        'rho_ft': args.rho_ft,
        'omega': args.omega,
        'empirical_omega': empirical_omega,
        'a_pt': args.a_pt,
        'c_pt': args.c_pt,
        'lambda_pt': args.lambda_pt,
        'gamma_reinit': args.gamma_reinit,
        'n_pt_active': n_pt_active,
        'n_ft_active': n_ft_active,
        'n_overlap': n_overlap,
        'n_ov': int(ov_mask.sum().item()),
        'n_new': int(new_mask.sum().item()),
        'n_ptonly': int(ptonly_mask.sum().item()),
        'n_none': int(none_mask.sum().item()),
    }
    
    config_path = os.path.join(args.save_folder, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nResults saved to {args.save_folder}")


def get_parser():
    parser = argparse.ArgumentParser(description="PT+FT Oracle Experiment (Step 3)")
    
    # Basic params
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--inp_dim', type=int, default=1000)
    parser.add_argument('--n_train', type=int, default=200)
    parser.add_argument('--n_test', type=int, default=10000)
    
    # PT+FT oracle params
    parser.add_argument('--rho_pt', type=float, required=True, 
                        help='PT support fraction')
    parser.add_argument('--rho_ft', type=float, required=True,
                        help='FT teacher sparsity (BG)')
    parser.add_argument('--omega', type=float, required=True,
                        help='Overlap fraction |S_pt ∩ S_ft| / |S_ft|')
    parser.add_argument('--a_pt', type=float, default=1.0,
                        help='PT ground truth amplitude')
    parser.add_argument('--c_pt', type=float, default=0.001,
                        help='PT parameter c')
    parser.add_argument('--lambda_pt', type=float, default=0.0,
                        help='PT initialization lambda (NOT regularizer)')
    parser.add_argument('--gamma_reinit', type=float, default=0.0,
                        help='Readout reinitialization parameter')
    
    # Training params
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=5000000)
    parser.add_argument('--threshold', type=float, default=1e-12)
    parser.add_argument('--stop_pred_mse', type=float, default=None)
    parser.add_argument('--stop_beta_rate', type=float, default=0.0)
    parser.add_argument('--stop_grad_norm', type=float, default=0.0)
    parser.add_argument('--lr_decay', type=float, default=1.0)
    parser.add_argument('--lr_decay_interval', type=int, default=2000)
    parser.add_argument('--test_every_n_epochs', type=int, default=200)
    parser.add_argument('--no_tuning', action='store_true')
    
    return parser


if __name__ == '__main__':
    print("Starting PT+FT Oracle experiment...", flush=True)
    parser = get_parser()
    args = parser.parse_args()
    main(args)


