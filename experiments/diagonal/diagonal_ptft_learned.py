#!/usr/bin/env python3
"""
Step 4: Empirical PT+FT with Learned PT

Implements the full PT→FT pipeline:
1. Train PT diagonal net on PT data to learn weights
2. Extract per-coordinate PT statistics (beta_hat_pt) from trained model
3. Apply Cosyne mapping to get predicted c_ft_i / k_i
4. Initialize FT diagonal net with predicted c_ft
5. Train on FT data and compare with oracle prediction

This is the final validation step that tests whether the Cosyne mapping
predicts FT performance from actual PT training.
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

from experiments.diagonal.diagonal_network_pretrain_bg import (
    DiagonalNet, train, make_deterministic, get_parameters_vectorized,
)
from experiments.diagonal.diagonal_ptft_oracle import (
    sample_pt_teacher, sample_ft_teacher_with_overlap, compute_c_ft_cosyne,
)


def train_pt_phase(
    inp_dim: int,
    n_train_pt: int,
    n_test: int,
    beta_pt: torch.Tensor,
    c_pt: float,
    lmda: float,
    lr: float,
    epochs: int,
    threshold: float,
    save_folder: str,
    gen_train: torch.Generator,
    gen_test: torch.Generator,
):
    """
    Train PT diagonal network on PT data.
    
    Returns:
        beta_hat_pt: Learned PT weights (numpy array)
        pt_model: Trained PT model
        pt_df: Training metrics DataFrame
    """
    print("\n" + "="*40)
    print("PT TRAINING PHASE")
    print("="*40)
    
    # Sample PT data
    x_train = torch.randn(n_train_pt, inp_dim, generator=gen_train) / math.sqrt(n_train_pt)
    x_test = torch.randn(n_test, inp_dim, generator=gen_test) / math.sqrt(n_test)
    
    y_train = x_train @ beta_pt
    y_test = x_test @ beta_pt
    
    # Initialize PT network with homogeneous c_pt
    pt_net = DiagonalNet(inp_dim, scaling=1.0, lmda=lmda, c=c_pt, init_method='complex')
    
    # Create PT save folder
    pt_save_folder = os.path.join(save_folder, 'pt_phase')
    Path(pt_save_folder).mkdir(parents=True, exist_ok=True)
    
    # Train PT
    pt_df, pt_net, pt_norm_df, pt_stop_reason, pt_final_epoch = train(
        pt_net,
        (x_train, y_train),
        (x_test, y_test),
        beta_pt,
        test_every_n_epochs=200,
        lr=lr,
        epochs=epochs,
        lr_tuning=True,
        threshold=threshold,
        save_folder=pt_save_folder,
    )
    
    # Extract learned beta_hat_pt
    with torch.no_grad():
        beta_hat_pt = pt_net.beta().numpy()
    
    # Save PT results
    pt_df.to_feather(os.path.join(pt_save_folder, 'df.feather'))
    torch.save(pt_net.state_dict(), os.path.join(pt_save_folder, 'model.pt'))
    np.save(os.path.join(pt_save_folder, 'beta_hat_pt.npy'), beta_hat_pt)
    
    # Compute PT metrics
    pt_param_mse = F.mse_loss(torch.from_numpy(beta_hat_pt), beta_pt).item()
    print(f"\nPT Phase completed:")
    print(f"  PT param MSE: {pt_param_mse:.6e}")
    print(f"  Stop reason: {pt_stop_reason}")
    print(f"  Final epoch: {pt_final_epoch}")
    
    return beta_hat_pt, pt_net, pt_df, pt_param_mse


def main(args):
    print("\n" + "="*80)
    print("STEP 4: PT+FT WITH LEARNED PT")
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
    print(f"PT training samples (n_train_pt): {args.n_train_pt}")
    print(f"FT sparsity (rho_ft): {args.rho_ft}")
    print(f"FT training samples (n_train_ft): {args.n_train_ft}")
    print(f"Overlap (omega): {args.omega}")
    print(f"PT amplitude (a_pt): {args.a_pt}")
    print(f"PT c parameter (c_pt): {args.c_pt}")
    print(f"PT lambda (lambda_pt): {args.lambda_pt}")
    print(f"Gamma reinit: {args.gamma_reinit}")
    print(f"Test samples: {args.n_test}")
    print(f"Save folder: {args.save_folder}")
    print("="*80)
    sys.stdout.flush()
    
    # Generators for reproducibility
    gen_pt_teacher = torch.Generator(device='cpu').manual_seed(args.seed + 0)
    gen_ft_teacher = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    gen_pt_train_x = torch.Generator(device='cpu').manual_seed(args.seed + 2)
    gen_pt_test_x = torch.Generator(device='cpu').manual_seed(args.seed + 3)
    gen_ft_train_x = torch.Generator(device='cpu').manual_seed(args.seed + 4)
    gen_ft_test_x = torch.Generator(device='cpu').manual_seed(args.seed + 5)
    
    Path(args.save_folder).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Sample PT teacher (deterministic)
    print("\nSampling PT teacher...")
    beta_pt, support_pt = sample_pt_teacher(args.inp_dim, args.rho_pt, args.a_pt, gen_pt_teacher)
    n_pt_active = support_pt.sum().item()
    print(f"  PT support size: {n_pt_active} / {args.inp_dim} = {n_pt_active/args.inp_dim:.4f}")
    
    # Step 2: Sample FT teacher with controlled overlap
    print("\nSampling FT teacher with overlap...")
    beta_ft, support_ft = sample_ft_teacher_with_overlap(
        args.inp_dim, args.rho_ft, args.omega, support_pt, gen_ft_teacher
    )
    n_ft_active = support_ft.sum().item()
    n_overlap = (support_pt & support_ft).sum().item()
    empirical_omega = n_overlap / n_ft_active if n_ft_active > 0 else 0
    print(f"  FT support size: {n_ft_active} / {args.inp_dim} = {n_ft_active/args.inp_dim:.4f}")
    print(f"  Overlap: {n_overlap} / {n_ft_active} = {empirical_omega:.4f} (target: {args.omega})")
    
    # Step 3: Train PT phase
    beta_hat_pt, pt_net, pt_df, pt_param_mse = train_pt_phase(
        args.inp_dim,
        args.n_train_pt,
        args.n_test,
        beta_pt,
        args.c_pt,
        args.lambda_pt,
        args.lr_pt,
        args.epochs_pt,
        args.threshold,
        args.save_folder,
        gen_pt_train_x,
        gen_pt_test_x,
    )
    
    # Step 4: Compute c_ft from LEARNED beta_hat_pt (not oracle beta_pt)
    print("\n" + "="*40)
    print("COMPUTING FT INIT FROM LEARNED PT")
    print("="*40)
    
    # Option A: Use learned beta_hat_pt
    c_ft_learned = compute_c_ft_cosyne(beta_hat_pt, args.c_pt, args.lambda_pt, args.gamma_reinit)
    print(f"\nUsing LEARNED beta_hat_pt for c_ft:")
    print(f"  c_ft range: [{c_ft_learned.min():.6f}, {c_ft_learned.max():.6f}]")
    
    # Option B: For comparison, also compute oracle c_ft
    c_ft_oracle = compute_c_ft_cosyne(beta_pt.numpy(), args.c_pt, args.lambda_pt, args.gamma_reinit)
    print(f"\nOracle c_ft (from true beta_pt):")
    print(f"  c_ft range: [{c_ft_oracle.min():.6f}, {c_ft_oracle.max():.6f}]")
    
    # Compare learned vs oracle c_ft
    c_ft_diff = np.abs(c_ft_learned - c_ft_oracle).mean()
    c_ft_corr = np.corrcoef(c_ft_learned, c_ft_oracle)[0, 1]
    print(f"\nLearned vs Oracle c_ft:")
    print(f"  Mean abs diff: {c_ft_diff:.6e}")
    print(f"  Correlation: {c_ft_corr:.6f}")
    
    # Step 5: Train FT phase with LEARNED c_ft
    print("\n" + "="*40)
    print("FT TRAINING PHASE (LEARNED INIT)")
    print("="*40)
    
    # Sample FT data
    x_ft_train = torch.randn(args.n_train_ft, args.inp_dim, generator=gen_ft_train_x) / math.sqrt(args.n_train_ft)
    x_ft_test = torch.randn(args.n_test, args.inp_dim, generator=gen_ft_test_x) / math.sqrt(args.n_test)
    
    y_ft_train = x_ft_train @ beta_ft
    y_ft_test = x_ft_test @ beta_ft
    
    # Initialize FT network with learned c_ft
    ft_net_learned = DiagonalNet(
        args.inp_dim, 
        scaling=1.0, 
        lmda=0.0, 
        c=args.c_pt,
        c_vec=c_ft_learned,
        init_method='complex'
    )
    
    ft_save_folder_learned = os.path.join(args.save_folder, 'ft_phase_learned')
    Path(ft_save_folder_learned).mkdir(parents=True, exist_ok=True)
    
    # Train FT with learned init
    ft_df_learned, ft_net_learned, _, ft_stop_reason_learned, ft_final_epoch_learned = train(
        ft_net_learned,
        (x_ft_train, y_ft_train),
        (x_ft_test, y_ft_test),
        beta_ft,
        test_every_n_epochs=args.test_every_n_epochs,
        lr=args.lr_ft,
        epochs=args.epochs_ft,
        lr_tuning=(not args.no_tuning),
        threshold=args.threshold,
        save_folder=ft_save_folder_learned,
    )
    
    # Get final FT metrics
    ft_test = ft_df_learned[(ft_df_learned['split'] == 'test') & 
                            (ft_df_learned['epoch'] == ft_df_learned['epoch'].max())]
    ft_param_mse_learned = ft_test['param_mse'].values[0] if len(ft_test) > 0 else np.nan
    
    # Save FT learned results
    ft_df_learned.to_feather(os.path.join(ft_save_folder_learned, 'df.feather'))
    torch.save(ft_net_learned.state_dict(), os.path.join(ft_save_folder_learned, 'model.pt'))
    np.save(os.path.join(ft_save_folder_learned, 'c_ft_learned.npy'), c_ft_learned)
    
    print(f"\nFT (learned init) completed:")
    print(f"  FT param MSE: {ft_param_mse_learned:.6e}")
    
    # Step 6: For comparison, also train FT with oracle c_ft
    print("\n" + "="*40)
    print("FT TRAINING PHASE (ORACLE INIT)")
    print("="*40)
    
    # Reset generator for fair comparison
    gen_ft_train_x_oracle = torch.Generator(device='cpu').manual_seed(args.seed + 4)
    gen_ft_test_x_oracle = torch.Generator(device='cpu').manual_seed(args.seed + 5)
    
    x_ft_train_oracle = torch.randn(args.n_train_ft, args.inp_dim, generator=gen_ft_train_x_oracle) / math.sqrt(args.n_train_ft)
    x_ft_test_oracle = torch.randn(args.n_test, args.inp_dim, generator=gen_ft_test_x_oracle) / math.sqrt(args.n_test)
    
    y_ft_train_oracle = x_ft_train_oracle @ beta_ft
    y_ft_test_oracle = x_ft_test_oracle @ beta_ft
    
    ft_net_oracle = DiagonalNet(
        args.inp_dim, 
        scaling=1.0, 
        lmda=0.0, 
        c=args.c_pt,
        c_vec=c_ft_oracle,
        init_method='complex'
    )
    
    ft_save_folder_oracle = os.path.join(args.save_folder, 'ft_phase_oracle')
    Path(ft_save_folder_oracle).mkdir(parents=True, exist_ok=True)
    
    ft_df_oracle, ft_net_oracle, _, ft_stop_reason_oracle, ft_final_epoch_oracle = train(
        ft_net_oracle,
        (x_ft_train_oracle, y_ft_train_oracle),
        (x_ft_test_oracle, y_ft_test_oracle),
        beta_ft,
        test_every_n_epochs=args.test_every_n_epochs,
        lr=args.lr_ft,
        epochs=args.epochs_ft,
        lr_tuning=(not args.no_tuning),
        threshold=args.threshold,
        save_folder=ft_save_folder_oracle,
    )
    
    ft_test_oracle = ft_df_oracle[(ft_df_oracle['split'] == 'test') & 
                                   (ft_df_oracle['epoch'] == ft_df_oracle['epoch'].max())]
    ft_param_mse_oracle = ft_test_oracle['param_mse'].values[0] if len(ft_test_oracle) > 0 else np.nan
    
    # Save FT oracle results
    ft_df_oracle.to_feather(os.path.join(ft_save_folder_oracle, 'df.feather'))
    torch.save(ft_net_oracle.state_dict(), os.path.join(ft_save_folder_oracle, 'model.pt'))
    np.save(os.path.join(ft_save_folder_oracle, 'c_ft_oracle.npy'), c_ft_oracle)
    
    print(f"\nFT (oracle init) completed:")
    print(f"  FT param MSE: {ft_param_mse_oracle:.6e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"PT param MSE:           {pt_param_mse:.6e}")
    print(f"FT param MSE (learned): {ft_param_mse_learned:.6e}")
    print(f"FT param MSE (oracle):  {ft_param_mse_oracle:.6e}")
    print(f"Learned/Oracle ratio:   {ft_param_mse_learned / ft_param_mse_oracle:.4f}" if ft_param_mse_oracle > 0 else "N/A")
    print(f"c_ft correlation:       {c_ft_corr:.6f}")
    print("="*80)
    
    # Save final config and summary
    config = {
        'seed': args.seed,
        'inp_dim': args.inp_dim,
        'n_train_pt': args.n_train_pt,
        'n_train_ft': args.n_train_ft,
        'n_test': args.n_test,
        'rho_pt': args.rho_pt,
        'rho_ft': args.rho_ft,
        'omega': args.omega,
        'empirical_omega': empirical_omega,
        'a_pt': args.a_pt,
        'c_pt': args.c_pt,
        'lambda_pt': args.lambda_pt,
        'gamma_reinit': args.gamma_reinit,
        'pt_param_mse': float(pt_param_mse),
        'ft_param_mse_learned': float(ft_param_mse_learned),
        'ft_param_mse_oracle': float(ft_param_mse_oracle),
        'c_ft_correlation': float(c_ft_corr),
        'c_ft_mean_diff': float(c_ft_diff),
    }
    
    config_path = os.path.join(args.save_folder, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save teachers
    torch.save(beta_pt, os.path.join(args.save_folder, 'beta_pt.pt'))
    torch.save(beta_ft, os.path.join(args.save_folder, 'beta_ft.pt'))
    
    print(f"\nResults saved to {args.save_folder}")


def get_parser():
    parser = argparse.ArgumentParser(description="PT+FT with Learned PT (Step 4)")
    
    # Basic params
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--inp_dim', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=10000)
    
    # PT params
    parser.add_argument('--rho_pt', type=float, required=True)
    parser.add_argument('--n_train_pt', type=int, required=True,
                        help='Number of PT training samples')
    parser.add_argument('--a_pt', type=float, default=1.0)
    parser.add_argument('--c_pt', type=float, default=0.001)
    parser.add_argument('--lambda_pt', type=float, default=0.0)
    parser.add_argument('--lr_pt', type=float, default=0.5)
    parser.add_argument('--epochs_pt', type=int, default=5000000)
    
    # FT params
    parser.add_argument('--rho_ft', type=float, required=True)
    parser.add_argument('--n_train_ft', type=int, required=True,
                        help='Number of FT training samples')
    parser.add_argument('--omega', type=float, required=True)
    parser.add_argument('--gamma_reinit', type=float, default=0.0)
    parser.add_argument('--lr_ft', type=float, default=0.5)
    parser.add_argument('--epochs_ft', type=int, default=5000000)
    
    # Common training params
    parser.add_argument('--threshold', type=float, default=1e-12)
    parser.add_argument('--test_every_n_epochs', type=int, default=200)
    parser.add_argument('--no_tuning', action='store_true')
    
    return parser


if __name__ == '__main__':
    print("Starting PT+FT with Learned PT experiment...", flush=True)
    parser = get_parser()
    args = parser.parse_args()
    main(args)


