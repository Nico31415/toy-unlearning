#!/usr/bin/env python3
"""
Run replica computations for experiments with empty replica_cache.
"""

import sys
from pathlib import Path
import os
import re
import math

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent.parent))

# Add ReplicaExperiments to path
sys.path.insert(0, str(THIS_DIR.parent.parent / "ReplicaExperiments"))

import numpy as np
import pandas as pd

from fixed_lambda_all import (
    Config,
    solve_rspmap_qk_curve_best_of_forward_backward,
)

# Import PTFTOracleConfig and sampler
from plot_replica_q_bg import PTFTOracleConfig, sample_ptft_oracle_mc

EXP_BASE = Path("figures/panel_experiments")

# Fixed params
RHO_PT = 0.1
FT_REGULARISER_SCALE = 1e-6
A_PT = 1.0

# Replica params
ALPHA_MIN = 0.02
ALPHA_MAX = 1.0
ALPHA_POINTS = 50
MC_SAMPLES = 10000


def build_config(
    rho: float,
    sigma0_2: float,
    beta_min: float,
    beta_max: float,
    beta_points: int,
    max_fp_iters: int = 900,
    tol_fp: float = 1e-10,
    damp: float = 0.25,
) -> Config:
    """Build Config for replica solver."""
    var_nonzero = 1.0 / rho
    betas = np.linspace(beta_min, beta_max, beta_points)
    
    return Config(
        rho=rho,
        var_nonzero=var_nonzero,
        sigma0_2=sigma0_2,
        betas=betas,
        max_fp_iters=max_fp_iters,
        tol_fp=tol_fp,
        damp=damp,
    )


def find_missing_replica_dirs():
    """Find all experiment directories with empty replica_cache."""
    missing = []
    
    for exp_dir in sorted(EXP_BASE.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        # Find the ptft_oracle subdirectory
        subdirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith('ptft_oracle')]
        if not subdirs:
            continue
        
        subdir = subdirs[0]
        cache_dir = subdir / "replica_cache"
        
        # Check if cache is empty
        if not cache_dir.exists() or len(list(cache_dir.glob("*.csv"))) == 0:
            missing.append((exp_dir, subdir))
    
    return missing


def parse_params_from_dirname(dirname: str) -> dict:
    """Parse parameters from directory name."""
    # Format: rpt=0.1__rft=0.1__om=0.5__cpt=0.001__lpt=0__gam=0
    params = {}
    for part in dirname.split("__"):
        if "=" in part:
            key, val = part.split("=", 1)
            params[key] = float(val)
    return params


def compute_replica_curve(rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit,
                          ft_regulariser_scale, alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX,
                          alpha_points=ALPHA_POINTS, mc_samples=MC_SAMPLES, seed=42):
    """Compute replica curve for given parameters."""
    
    # Create PTFT oracle config
    ptft_cfg = PTFTOracleConfig(
        rho_pt=rho_pt,
        rho_ft=rho_ft,
        omega=omega,
        a_pt=A_PT,
        c_pt=c_pt,
        lambda_pt=lambda_pt,
        gamma_reinit=gamma_reinit,
    )
    
    # Sample MC
    rng = np.random.default_rng(seed)
    beta_ft_mc, beta_pt_mc, k_mc, g_mc = sample_ptft_oracle_mc(ptft_cfg, rng, mc_samples)
    x_mc = beta_ft_mc
    v_mc = rng.normal(size=mc_samples)
    
    # Build solver config
    sigma0_2 = 0.0  # No output noise
    beta_min = 1.0 / alpha_max
    beta_max = 1.0 / alpha_min
    
    cfg = build_config(
        rho=rho_ft,
        sigma0_2=sigma0_2,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_points=alpha_points,
    )
    
    # For ptft_oracle: gamma_ext = ft_regulariser_scale directly (no k-scaling)
    gamma_ext = float(ft_regulariser_scale)
    
    # Alpha and beta ranges
    alpha_range = np.linspace(alpha_min, alpha_max, alpha_points)
    alpha_reversed = alpha_range[::-1]
    beta_range = 1.0 / alpha_reversed
    
    # Solve using hetero-k path (k_mc contains per-coordinate k values)
    k_q = 1.0  # Placeholder, not used when k_mc is provided
    mse_beta = solve_rspmap_qk_curve_best_of_forward_backward(
        beta_range, gamma_ext, k_q, x_mc, v_mc, cfg, k_mc=k_mc, g_mc=g_mc
    )
    
    # Reverse to get MSE in alpha order
    mse_alpha = mse_beta[::-1]
    
    # Build dataframe
    df = pd.DataFrame({
        'alpha': alpha_range,
        'mse': mse_alpha,
    })
    
    return df


def run_single_config(exp_dir: Path, subdir: Path, params: dict):
    """Run replica computation for a single config."""
    cache_dir = subdir / "replica_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    rho_ft = params['rft']
    omega = params['om']
    c_pt = params['cpt']
    lambda_pt = params['lpt']
    gamma_reinit = params['gam']
    
    print(f"  Computing replica: rft={rho_ft}, om={omega}, cpt={c_pt}, lpt={lambda_pt}, gam={gamma_reinit}")
    
    try:
        df = compute_replica_curve(
            rho_pt=RHO_PT,
            rho_ft=rho_ft,
            omega=omega,
            c_pt=c_pt,
            lambda_pt=lambda_pt,
            gamma_reinit=gamma_reinit,
            ft_regulariser_scale=FT_REGULARISER_SCALE,
        )
        
        # Save to cache
        cache_file = cache_dir / f"replica_rft={rho_ft}_om={omega}.csv"
        df.to_csv(cache_file, index=False)
        print(f"  ✓ Saved to {cache_file}")
        print(f"    MSE range: [{df['mse'].min():.2e}, {df['mse'].max():.2e}]")
        return True
    except Exception as e:
        import traceback
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', type=int, default=None,
                        help='SLURM array task ID (0-indexed)')
    parser.add_argument('--list', action='store_true',
                        help='List missing configs and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be run without running')
    args = parser.parse_args()
    
    missing = find_missing_replica_dirs()
    print(f"Found {len(missing)} experiments with empty replica_cache")
    
    if args.list:
        for i, (exp_dir, subdir) in enumerate(missing):
            params = parse_params_from_dirname(exp_dir.name)
            print(f"{i}: {exp_dir.name}")
        return
    
    if args.task_id is not None:
        # SLURM array mode
        if args.task_id >= len(missing):
            print(f"Task ID {args.task_id} >= {len(missing)} missing configs, skipping")
            return
        
        exp_dir, subdir = missing[args.task_id]
        params = parse_params_from_dirname(exp_dir.name)
        print(f"Task {args.task_id}: {exp_dir.name}")
        
        if args.dry_run:
            print(f"  Would compute replica for: {params}")
            return
        
        success = run_single_config(exp_dir, subdir, params)
        if success:
            print(f"✓ Completed task {args.task_id}")
        else:
            print(f"✗ Failed task {args.task_id}")
            sys.exit(1)
    else:
        # Local mode: run all
        successes = 0
        failures = 0
        for i, (exp_dir, subdir) in enumerate(missing):
            params = parse_params_from_dirname(exp_dir.name)
            print(f"\n[{i+1}/{len(missing)}] {exp_dir.name}")
            
            if args.dry_run:
                print(f"  Would compute replica for: {params}")
                continue
            
            if run_single_config(exp_dir, subdir, params):
                successes += 1
            else:
                failures += 1
        
        print(f"\n✓ Completed: {successes} succeeded, {failures} failed")


if __name__ == '__main__':
    main()
