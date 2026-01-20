#!/usr/bin/env python3
"""
Generate replica theory curves for diagonal network q-function implicit bias.

Computes theoretical generalization error curves using replica theory for the
q function (implicit bias from diagonal network initialization) with very small
regularization λ, and overlays them on empirical curves.

IMPORTANT NOTES:
- Replica solver outputs PARAMETER MSE (not prediction MSE)
- Empirical comparison must use param_mse (not test_pred_mse)
- Mapping from diagonal-net init parameter c to q-regularizer k uses: k = (2c)²
  This is for complex init with λ = 0: w_pos = v_pos = sqrt(c/2) => sqrt(k) = 2c => k = (2c)²
- Optional pred-MSE overlay uses d / n_test conversion factor

IMPORTANT PARAMETER DISTINCTION:
- ft_regulariser_scale (CLI arg): FT regularization strength (what was lambda_small)
- lambda_pt (PT+FT oracle only): PT initialization parameter (affects initial weights)
  These are DIFFERENT concepts and should not be confused!

SEMANTICS BY MODE:
- Homogeneous mode (default):
  * c determines k_q = (2c)²
  * gamma_ext is scaled using k_q (preserves legacy behavior)
  * Cache filenames remain unchanged from original implementation
  
- Hetero-k modes (mixture/support):
  * k_mc (per-coordinate k values) determines the regularizer parameters
  * gamma_ext = ft_regulariser_scale (NO k_q scaling, independent of c)
  * c parameter is IGNORED for gamma_ext computation
  * Cache filenames include hetero-k parameters for disambiguation

Example usages:

Homogeneous (default):
    python scripts/diagonal/plot_replica_q_bg.py \\
      --rho 0.04 --c_values 0.001 0.5 --ft_regulariser_scale 1e-6

Step 1A (mixture over k, independent of teacher support):
    python scripts/diagonal/plot_replica_q_bg.py \\
      --rho 0.04 --c_values 0.001 0.5 --ft_regulariser_scale 1e-6 \\
      --k_mode mixture --k_A 4e-6 --k_B 1.0 --pi_A 0.5

Step 1B (k correlated with teacher support):
    python scripts/diagonal/plot_replica_q_bg.py \\
      --rho 0.04 --c_values 0.001 0.5 --ft_regulariser_scale 1e-6 \\
      --k_mode support --k_nz 4e-6 --k_z 1.0

Step 2A (PT+FT oracle with overlap):
    python scripts/diagonal/plot_replica_q_bg.py \\
      --teacher_mode ptft_oracle \\
      --rho_pt 0.10 --rho_ft 0.04 --omega 0.5 \\
      --a_pt 1.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \\
      --ft_regulariser_scale 1e-6 \\
      --mc_samples 50000 --seed 12345 \\
      --debug_groups

BREAKING CHANGE (cache invalidation):
    This version renamed --lambda_small to --ft_regulariser_scale.
    Old cached curves with "lambda=" in filename will not be found.
    Delete old cache directory to regenerate with new naming.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import replica theory functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ReplicaExperiments.fixed_lambda_all import (
    Config,
    KModeConfig,
    PTFTOracleConfig,
    gamma_ext_for_q_small,
    gamma_ext_for_q_big,
    solve_rspmap_qk_curve_best_of_forward_backward,
    sample_bg,
    sample_k_mc,
    compute_c_ft_from_pt,
    sample_ptft_oracle_mc,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def to_db(x: np.ndarray) -> np.ndarray:
    """Convert MSE to dB: 10*log10(mse + 1e-15)"""
    return 10.0 * np.log10(np.maximum(x, 1e-15))


def gamma_ext_for_hetero_k(ft_regulariser_scale: float) -> float:
    """
    For heterogeneous-k runs, keep gamma_ext tied only to ft_regulariser_scale to avoid
    coupling the global strength to c-derived k_q.
    """
    return float(ft_regulariser_scale)


# =============================================================================
# PT+FT OVERLAP SWEEP DEFINITIONS (TASK B & C)
# =============================================================================

# Fixed PT scale parameters (per Global Rules)
PTFT_FIXED_C_PT = 0.001
PTFT_FIXED_LAMBDA_PT = 0.0
PTFT_FIXED_GAMMA_REINIT = 0.0
PTFT_FIXED_A_PT = 1.0

# Omega values to sweep
PTFT_OMEGA_LIST = [0.0, 0.5, 1.0]

# Sparsity regime definitions
PTFT_REGIMES: Dict[str, Dict] = {
    "sparse_sparse": {
        "name": "Sparse PT → Sparse FT",
        "rho_pt": 0.1,
        "rho_ft": 0.1,
        "omega_list": PTFT_OMEGA_LIST,
        "description": "Full overlap possible",
    },
    "sparse_verysparse": {
        "name": "Sparse PT → Very Sparse FT",
        "rho_pt": 0.1,
        "rho_ft": 0.04,
        "omega_list": PTFT_OMEGA_LIST,
        "description": "PT support contains FT support when omega=1",
    },
    "sparse_dense": {
        "name": "Sparse PT → Dense FT",
        "rho_pt": 0.1,
        "rho_ft": 0.99,
        "omega_list": [0.0],  # No overlap (rho_ft≈1 forces omega≈0)
        "description": "No overlap (rho_ft=0.99 forces omega=0)",
    },
    "dense_sparse": {
        "name": "Dense PT → Sparse FT",
        "rho_pt": 0.99,
        "rho_ft": 0.1,
        "omega_list": [0.1],  # Max overlap = rho_ft
        "description": "Max overlap (omega=rho_ft)",
    },
}


def check_ptft_feasibility(rho_pt: float, rho_ft: float, omega: float) -> bool:
    """
    Check if PT+FT parameters satisfy feasibility constraints.
    
    Constraints:
        - omega * rho_ft <= rho_pt  (p_ptonly >= 0)
        - rho_pt + (1-omega) * rho_ft <= 1.0  (p_none >= 0)
    
    Returns:
        True if feasible, False otherwise
    """
    p_ov = omega * rho_ft
    p_new = (1.0 - omega) * rho_ft
    p_ptonly = rho_pt - p_ov
    p_none = 1.0 - rho_pt - p_new
    
    return (p_ov >= -1e-10 and p_new >= -1e-10 and 
            p_ptonly >= -1e-10 and p_none >= -1e-10)


def run_ptft_overlap_sweep(
    regime_name: str,
    args,
    output_dir: str,
) -> Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Run PT+FT overlap sweep for a single regime.
    
    Args:
        regime_name: Name of the regime (key in PTFT_REGIMES)
        args: Parsed CLI arguments
        output_dir: Output directory for plots
    
    Returns:
        Dictionary mapping omega -> (alpha_vals, mse_vals, active_frac_vals)
    """
    regime = PTFT_REGIMES[regime_name]
    rho_pt = regime["rho_pt"]
    rho_ft = regime["rho_ft"]
    omega_list = regime["omega_list"]
    
    print(f"\n{'='*80}")
    print(f"REGIME: {regime['name']}")
    print(f"  rho_pt={rho_pt}, rho_ft={rho_ft}")
    print(f"  omega values: {omega_list}")
    print(f"  Description: {regime['description']}")
    print(f"{'='*80}")
    
    # Build config for this regime (use rho_ft for the BG variance scaling)
    sigma0_2 = 0.0  # Noiseless
    beta_min = 1.0 / args.alpha_max
    beta_max = 1.0 / args.alpha_min
    
    cfg = build_config(
        rho=rho_ft,  # Use rho_ft for variance scaling
        sigma0_2=sigma0_2,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_points=args.alpha_points,
        max_fp_iters=args.max_fp_iters,
        tol_fp=args.tol_fp,
        damp=args.damp,
    )
    
    alpha_range = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    results = {}
    
    # Cache directory
    cache_dir = os.path.join(output_dir, "replica_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    for omega in omega_list:
        # Check feasibility
        if not check_ptft_feasibility(rho_pt, rho_ft, omega):
            print(f"\n  [omega={omega}] SKIPPED: infeasible combination")
            continue
        
        print(f"\n  [omega={omega}] Computing curve...")
        
        # Create cache filename
        cache_filename = (
            f"ptft_sweep--regime={regime_name}--om={omega:.4f}--"
            f"rpt={rho_pt:.4f}--rft={rho_ft:.4f}--"
            f"cpt={PTFT_FIXED_C_PT:.4f}--lpt={PTFT_FIXED_LAMBDA_PT:.2f}--"
            f"gam={PTFT_FIXED_GAMMA_REINIT:.2f}--apt={PTFT_FIXED_A_PT:.2f}--"
            f"alpha_min={args.alpha_min:.4f}--alpha_max={args.alpha_max:.4f}--"
            f"alpha_pts={args.alpha_points}--mc={args.mc_samples}--seed={args.seed}.csv"
        )
        cache_path = os.path.join(cache_dir, cache_filename)
        
        # Try to load from cache
        if os.path.exists(cache_path):
            print(f"    Loading from cache...")
            df_cache = pd.read_csv(cache_path)
            alpha_vals = df_cache["alpha"].values
            mse_vals = df_cache["mse"].values
            active_vals = df_cache["active_frac"].values
            results[omega] = (alpha_vals, mse_vals, active_vals)
            print(f"    MSE range: [{mse_vals.min():.2e}, {mse_vals.max():.2e}]")
            print(f"    Active frac range: [{active_vals.min():.4f}, {active_vals.max():.4f}]")
            continue
        
        # Create ptft_oracle config
        ptft_cfg = PTFTOracleConfig(
            rho_pt=rho_pt,
            rho_ft=rho_ft,
            omega=omega,
            a_pt=PTFT_FIXED_A_PT,
            c_pt=PTFT_FIXED_C_PT,
            lambda_pt=PTFT_FIXED_LAMBDA_PT,
            gamma_reinit=PTFT_FIXED_GAMMA_REINIT,
        )
        
        # Sample MC
        rng = np.random.default_rng(args.seed)
        beta_ft_mc, beta_pt_mc, k_mc, g_mc = sample_ptft_oracle_mc(ptft_cfg, rng, args.mc_samples)
        x_mc = beta_ft_mc
        v_mc = rng.normal(size=args.mc_samples)
        
        # Debug group statistics
        if args.debug_groups:
            group_names = ["OV", "NEW", "PTONLY", "NONE"]
            print(f"    Group fractions:")
            for i, name in enumerate(group_names):
                frac = float(np.mean(g_mc == i))
                print(f"      {name}: {frac:.4f}")
        
        # Compute gamma_ext (no k-scaling for ptft_oracle)
        gamma_ext = float(args.ft_regulariser_scale)
        
        # Convert alpha to beta (beta = 1/alpha, increasing order)
        alpha_reversed = alpha_range[::-1]
        beta_range = 1.0 / alpha_reversed
        
        # Compute replica curve with diagnostics
        k_q = (2.0 * PTFT_FIXED_C_PT) ** 2  # Dummy, not used in hetero-k path
        mse_beta, active_beta = solve_rspmap_qk_curve_best_of_forward_backward(
            beta_range, gamma_ext, k_q, x_mc, v_mc, cfg,
            k_mc=k_mc, g_mc=g_mc,
            return_diag=True, eps_active=1e-6,
        )
        
        # Reverse to match alpha_range order
        mse_vals = mse_beta[::-1]
        active_vals = active_beta[::-1]
        alpha_vals = alpha_range
        
        results[omega] = (alpha_vals, mse_vals, active_vals)
        print(f"    MSE range: [{mse_vals.min():.2e}, {mse_vals.max():.2e}]")
        print(f"    Active frac range: [{active_vals.min():.4f}, {active_vals.max():.4f}]")
        
        # Save to cache
        df_cache = pd.DataFrame({
            "alpha": alpha_vals,
            "mse": mse_vals,
            "active_frac": active_vals,
        })
        df_cache.to_csv(cache_path, index=False)
        print(f"    Saved to cache: {cache_path}")
    
    return results


def plot_ptft_mse_sweep(
    results: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    regime_name: str,
    output_dir: str,
):
    """
    Plot MSE vs alpha for a PT+FT overlap sweep.
    
    Args:
        results: Dict mapping omega -> (alpha_vals, mse_vals, active_frac_vals)
        regime_name: Name of the regime
        output_dir: Output directory for plots
    """
    regime = PTFT_REGIMES[regime_name]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color map
    omegas = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(omegas)))
    
    for omega, color in zip(omegas, colors):
        alpha_vals, mse_vals, _ = results[omega]
        mse_db = to_db(mse_vals)
        ax.plot(
            alpha_vals, mse_db,
            "-", linewidth=2.5, color=color,
            label=f"ω = {omega:.2f}"
        )
    
    ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=14)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=14)
    ax.set_title(
        f"PT+FT Overlap Sweep: {regime['name']}\n"
        f"$\\rho_{{pt}}={regime['rho_pt']}$, $\\rho_{{ft}}={regime['rho_ft']}$, "
        f"$c_{{pt}}={PTFT_FIXED_C_PT}$, $\\lambda_{{pt}}={PTFT_FIXED_LAMBDA_PT}$",
        fontsize=13
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')
    
    fig.tight_layout()
    
    # Save
    png_path = os.path.join(output_dir, f"ptft_{regime_name}_overlap_sweep.png")
    pdf_path = os.path.join(output_dir, f"ptft_{regime_name}_overlap_sweep.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  MSE plot saved: {png_path}")


def plot_ptft_active_fraction(
    results: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    regime_name: str,
    output_dir: str,
):
    """
    Plot active fraction vs alpha for a PT+FT overlap sweep.
    
    Args:
        results: Dict mapping omega -> (alpha_vals, mse_vals, active_frac_vals)
        regime_name: Name of the regime
        output_dir: Output directory for plots
    """
    regime = PTFT_REGIMES[regime_name]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color map
    omegas = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(omegas)))
    
    for omega, color in zip(omegas, colors):
        alpha_vals, _, active_vals = results[omega]
        ax.plot(
            alpha_vals, active_vals,
            "-", linewidth=2.5, color=color,
            label=f"ω = {omega:.2f}"
        )
    
    ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=14)
    ax.set_ylabel("Active Fraction", fontsize=14)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"Active Fraction: {regime['name']}\n"
        f"$\\rho_{{pt}}={regime['rho_pt']}$, $\\rho_{{ft}}={regime['rho_ft']}$, "
        f"$c_{{pt}}={PTFT_FIXED_C_PT}$",
        fontsize=13
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')
    
    fig.tight_layout()
    
    # Save
    png_path = os.path.join(output_dir, f"ptft_{regime_name}_active_fraction.png")
    pdf_path = os.path.join(output_dir, f"ptft_{regime_name}_active_fraction.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  Active fraction plot saved: {png_path}")


def plot_ptft_ft_sparsity_comparison(args, output_dir: str) -> None:
    """
    Plot MSE vs alpha for varying FT sparsity with fixed PT sparsity.
    
    Fixed: rho_pt = 0.1
    Varying: rho_ft in [0.1, 0.05, 0.01]
    Compare: omega=1.0 (solid) vs omega=0.0 (dashed)
    """
    print("\n" + "=" * 80)
    print("FT SPARSITY COMPARISON PLOT")
    print("  Fixed rho_pt = 0.1")
    print("  Varying rho_ft = [0.1, 0.05, 0.01]")
    print("  Comparing omega = 1.0 (solid) vs omega = 0.0 (dashed)")
    print("=" * 80)
    
    rho_pt = 0.1
    rho_ft_list = [0.1, 0.05, 0.01]
    omega_list = [1.0, 0.0]
    
    # Cache directory
    cache_dir = os.path.join(output_dir, "replica_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    alpha_range = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    
    # Collect results: {(rho_ft, omega): (alpha, mse)}
    results = {}
    
    for rho_ft in rho_ft_list:
        # Build config for this rho_ft
        sigma0_2 = 0.0
        beta_min = 1.0 / args.alpha_max
        beta_max = 1.0 / args.alpha_min
        
        cfg = build_config(
            rho=rho_ft,
            sigma0_2=sigma0_2,
            beta_min=beta_min,
            beta_max=beta_max,
            beta_points=args.alpha_points,
            max_fp_iters=args.max_fp_iters,
            tol_fp=args.tol_fp,
            damp=args.damp,
        )
        
        for omega in omega_list:
            # Check feasibility
            if not check_ptft_feasibility(rho_pt, rho_ft, omega):
                print(f"  [rho_ft={rho_ft}, omega={omega}] SKIPPED: infeasible")
                continue
            
            print(f"\n  [rho_ft={rho_ft}, omega={omega}] Computing...")
            
            # Cache filename
            cache_filename = (
                f"ptft_ftsparsity--rpt={rho_pt:.4f}--rft={rho_ft:.4f}--om={omega:.4f}--"
                f"cpt={PTFT_FIXED_C_PT:.4f}--alpha_pts={args.alpha_points}--"
                f"mc={args.mc_samples}--seed={args.seed}.csv"
            )
            cache_path = os.path.join(cache_dir, cache_filename)
            
            if os.path.exists(cache_path):
                print(f"    Loading from cache...")
                df = pd.read_csv(cache_path)
                results[(rho_ft, omega)] = (df["alpha"].values, df["mse"].values)
                print(f"    MSE range: [{df['mse'].min():.2e}, {df['mse'].max():.2e}]")
                continue
            
            # Sample and compute
            ptft_cfg = PTFTOracleConfig(
                rho_pt=rho_pt,
                rho_ft=rho_ft,
                omega=omega,
                a_pt=PTFT_FIXED_A_PT,
                c_pt=PTFT_FIXED_C_PT,
                lambda_pt=PTFT_FIXED_LAMBDA_PT,
                gamma_reinit=PTFT_FIXED_GAMMA_REINIT,
            )
            
            rng = np.random.default_rng(args.seed)
            beta_ft_mc, _, k_mc, g_mc = sample_ptft_oracle_mc(ptft_cfg, rng, args.mc_samples)
            x_mc = beta_ft_mc
            v_mc = rng.normal(size=args.mc_samples)
            
            gamma_ext = float(args.ft_regulariser_scale)
            alpha_reversed = alpha_range[::-1]
            beta_range = 1.0 / alpha_reversed
            k_q = (2.0 * PTFT_FIXED_C_PT) ** 2
            
            mse_beta = solve_rspmap_qk_curve_best_of_forward_backward(
                beta_range, gamma_ext, k_q, x_mc, v_mc, cfg,
                k_mc=k_mc, g_mc=g_mc,
                return_diag=False,
            )
            mse_vals = mse_beta[::-1]
            
            results[(rho_ft, omega)] = (alpha_range, mse_vals)
            print(f"    MSE range: [{mse_vals.min():.2e}, {mse_vals.max():.2e}]")
            
            # Save cache
            df = pd.DataFrame({"alpha": alpha_range, "mse": mse_vals})
            df.to_csv(cache_path, index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rho_ft_list)))
    
    for rho_ft, color in zip(rho_ft_list, colors):
        # omega=1.0 solid
        if (rho_ft, 1.0) in results:
            alpha, mse = results[(rho_ft, 1.0)]
            ax.plot(alpha, to_db(mse), "-", linewidth=2.5, color=color,
                    label=f"ρ_ft={rho_ft}, ω=1.0")
        
        # omega=0.0 dashed
        if (rho_ft, 0.0) in results:
            alpha, mse = results[(rho_ft, 0.0)]
            ax.plot(alpha, to_db(mse), "--", linewidth=2.5, color=color,
                    label=f"ρ_ft={rho_ft}, ω=0.0")
    
    ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=14)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=14)
    ax.set_title(
        f"FT Sparsity Comparison (ρ_pt={rho_pt} fixed)\n"
        f"Solid: ω=1.0 (full overlap), Dashed: ω=0.0 (no overlap)",
        fontsize=13
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    
    fig.tight_layout()
    
    png_path = os.path.join(output_dir, "ptft_ft_sparsity_comparison.png")
    pdf_path = os.path.join(output_dir, "ptft_ft_sparsity_comparison.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\n  Saved: {png_path}")


def plot_ptft_dense_regime(args, output_dir: str) -> None:
    """
    Plot MSE vs alpha for dense FT regime.
    
    Fixed: rho_pt = 0.5, rho_ft = 0.5
    Varying: omega in [0.0, 0.5, 1.0]
    """
    print("\n" + "=" * 80)
    print("DENSE FT REGIME PLOT")
    print("  rho_pt = 0.5, rho_ft = 0.5")
    print("  omega = [0.0, 0.5, 1.0]")
    print("=" * 80)
    
    rho_pt = 0.5
    rho_ft = 0.5
    omega_list = [0.0, 0.5, 1.0]
    
    # Cache directory
    cache_dir = os.path.join(output_dir, "replica_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    alpha_range = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    
    # Build config
    sigma0_2 = 0.0
    beta_min = 1.0 / args.alpha_max
    beta_max = 1.0 / args.alpha_min
    
    cfg = build_config(
        rho=rho_ft,
        sigma0_2=sigma0_2,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_points=args.alpha_points,
        max_fp_iters=args.max_fp_iters,
        tol_fp=args.tol_fp,
        damp=args.damp,
    )
    
    results = {}
    
    for omega in omega_list:
        if not check_ptft_feasibility(rho_pt, rho_ft, omega):
            print(f"  [omega={omega}] SKIPPED: infeasible")
            continue
        
        print(f"\n  [omega={omega}] Computing...")
        
        cache_filename = (
            f"ptft_dense--rpt={rho_pt:.4f}--rft={rho_ft:.4f}--om={omega:.4f}--"
            f"cpt={PTFT_FIXED_C_PT:.4f}--alpha_pts={args.alpha_points}--"
            f"mc={args.mc_samples}--seed={args.seed}.csv"
        )
        cache_path = os.path.join(cache_dir, cache_filename)
        
        if os.path.exists(cache_path):
            print(f"    Loading from cache...")
            df = pd.read_csv(cache_path)
            results[omega] = (df["alpha"].values, df["mse"].values)
            print(f"    MSE range: [{df['mse'].min():.2e}, {df['mse'].max():.2e}]")
            continue
        
        ptft_cfg = PTFTOracleConfig(
            rho_pt=rho_pt,
            rho_ft=rho_ft,
            omega=omega,
            a_pt=PTFT_FIXED_A_PT,
            c_pt=PTFT_FIXED_C_PT,
            lambda_pt=PTFT_FIXED_LAMBDA_PT,
            gamma_reinit=PTFT_FIXED_GAMMA_REINIT,
        )
        
        rng = np.random.default_rng(args.seed)
        beta_ft_mc, _, k_mc, g_mc = sample_ptft_oracle_mc(ptft_cfg, rng, args.mc_samples)
        x_mc = beta_ft_mc
        v_mc = rng.normal(size=args.mc_samples)
        
        gamma_ext = float(args.ft_regulariser_scale)
        alpha_reversed = alpha_range[::-1]
        beta_range = 1.0 / alpha_reversed
        k_q = (2.0 * PTFT_FIXED_C_PT) ** 2
        
        mse_beta = solve_rspmap_qk_curve_best_of_forward_backward(
            beta_range, gamma_ext, k_q, x_mc, v_mc, cfg,
            k_mc=k_mc, g_mc=g_mc,
            return_diag=False,
        )
        mse_vals = mse_beta[::-1]
        
        results[omega] = (alpha_range, mse_vals)
        print(f"    MSE range: [{mse_vals.min():.2e}, {mse_vals.max():.2e}]")
        
        df = pd.DataFrame({"alpha": alpha_range, "mse": mse_vals})
        df.to_csv(cache_path, index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(omega_list)))
    
    for omega, color in zip(omega_list, colors):
        if omega in results:
            alpha, mse = results[omega]
            ax.plot(alpha, to_db(mse), "-", linewidth=2.5, color=color,
                    label=f"ω = {omega:.1f}")
    
    ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=14)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=14)
    ax.set_title(
        f"Dense Regime: ρ_pt={rho_pt}, ρ_ft={rho_ft}\n"
        f"Overlap sweep",
        fontsize=13
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')
    
    fig.tight_layout()
    
    png_path = os.path.join(output_dir, "ptft_dense_regime.png")
    pdf_path = os.path.join(output_dir, "ptft_dense_regime.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\n  Saved: {png_path}")


def plot_ptft_vs_baseline(args, output_dir: str) -> None:
    """
    Compare PT+FT oracle against no-transfer BG baseline.
    
    Shows whether PT actually helps or hurts compared to training from scratch.
    
    Plots:
    - BG baseline (rho=0.04, no PT)
    - PT+FT with omega=1.0 (full overlap)
    - PT+FT with omega=0.0 (no overlap)
    """
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON PLOT")
    print("  Comparing PT+FT oracle vs no-transfer BG baseline")
    print("  BG baseline: rho=0.04 (standard sparse recovery)")
    print("  PT+FT: rho_pt=0.1, rho_ft=0.04, omega in [0.0, 1.0]")
    print("=" * 80)
    
    rho_ft = 0.04  # FT sparsity (matches BG baseline)
    rho_pt = 0.1   # PT sparsity
    
    cache_dir = os.path.join(output_dir, "replica_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    alpha_range = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    
    # Build config
    sigma0_2 = 0.0
    beta_min = 1.0 / args.alpha_max
    beta_max = 1.0 / args.alpha_min
    
    cfg = build_config(
        rho=rho_ft,
        sigma0_2=sigma0_2,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_points=args.alpha_points,
        max_fp_iters=args.max_fp_iters,
        tol_fp=args.tol_fp,
        damp=args.damp,
    )
    
    results = {}
    
    # --- BG Baseline (no PT) ---
    print(f"\n  [BG Baseline] Computing...")
    
    cache_filename = (
        f"bg_baseline--rho={rho_ft:.4f}--"
        f"alpha_pts={args.alpha_points}--mc={args.mc_samples}--seed={args.seed}.csv"
    )
    cache_path = os.path.join(cache_dir, cache_filename)
    
    if os.path.exists(cache_path):
        print(f"    Loading from cache...")
        df = pd.read_csv(cache_path)
        results["baseline"] = (df["alpha"].values, df["mse"].values)
        print(f"    MSE range: [{df['mse'].min():.2e}, {df['mse'].max():.2e}]")
    else:
        # Sample standard BG (no PT)
        rng = np.random.default_rng(args.seed)
        x_mc = sample_bg(args.mc_samples, rng, rho_ft, cfg.var_nonzero)
        v_mc = rng.normal(size=args.mc_samples)
        
        # Use homogeneous k from c_pt (same implicit bias structure, but no PT information)
        k_q = (2.0 * PTFT_FIXED_C_PT) ** 2
        gamma_ext = float(args.ft_regulariser_scale)
        
        alpha_reversed = alpha_range[::-1]
        beta_range = 1.0 / alpha_reversed
        
        mse_beta = solve_rspmap_qk_curve_best_of_forward_backward(
            beta_range, gamma_ext, k_q, x_mc, v_mc, cfg,
            return_diag=False,
        )
        mse_vals = mse_beta[::-1]
        
        results["baseline"] = (alpha_range, mse_vals)
        print(f"    MSE range: [{mse_vals.min():.2e}, {mse_vals.max():.2e}]")
        
        df = pd.DataFrame({"alpha": alpha_range, "mse": mse_vals})
        df.to_csv(cache_path, index=False)
    
    # --- PT+FT with omega=1.0 and omega=0.0 ---
    for omega in [1.0, 0.0]:
        print(f"\n  [PT+FT omega={omega}] Computing...")
        
        cache_filename = (
            f"ptft_baseline_cmp--rpt={rho_pt:.4f}--rft={rho_ft:.4f}--om={omega:.4f}--"
            f"cpt={PTFT_FIXED_C_PT:.4f}--alpha_pts={args.alpha_points}--"
            f"mc={args.mc_samples}--seed={args.seed}.csv"
        )
        cache_path = os.path.join(cache_dir, cache_filename)
        
        if os.path.exists(cache_path):
            print(f"    Loading from cache...")
            df = pd.read_csv(cache_path)
            results[f"omega={omega}"] = (df["alpha"].values, df["mse"].values)
            print(f"    MSE range: [{df['mse'].min():.2e}, {df['mse'].max():.2e}]")
            continue
        
        ptft_cfg = PTFTOracleConfig(
            rho_pt=rho_pt,
            rho_ft=rho_ft,
            omega=omega,
            a_pt=PTFT_FIXED_A_PT,
            c_pt=PTFT_FIXED_C_PT,
            lambda_pt=PTFT_FIXED_LAMBDA_PT,
            gamma_reinit=PTFT_FIXED_GAMMA_REINIT,
        )
        
        rng = np.random.default_rng(args.seed)
        beta_ft_mc, _, k_mc, g_mc = sample_ptft_oracle_mc(ptft_cfg, rng, args.mc_samples)
        x_mc = beta_ft_mc
        v_mc = rng.normal(size=args.mc_samples)
        
        gamma_ext = float(args.ft_regulariser_scale)
        alpha_reversed = alpha_range[::-1]
        beta_range = 1.0 / alpha_reversed
        k_q = (2.0 * PTFT_FIXED_C_PT) ** 2
        
        mse_beta = solve_rspmap_qk_curve_best_of_forward_backward(
            beta_range, gamma_ext, k_q, x_mc, v_mc, cfg,
            k_mc=k_mc, g_mc=g_mc,
            return_diag=False,
        )
        mse_vals = mse_beta[::-1]
        
        results[f"omega={omega}"] = (alpha_range, mse_vals)
        print(f"    MSE range: [{mse_vals.min():.2e}, {mse_vals.max():.2e}]")
        
        df = pd.DataFrame({"alpha": alpha_range, "mse": mse_vals})
        df.to_csv(cache_path, index=False)
    
    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # BG Baseline - black dashed
    if "baseline" in results:
        alpha, mse = results["baseline"]
        ax.plot(alpha, to_db(mse), "k--", linewidth=2.5,
                label=f"BG Baseline (ρ={rho_ft}, no PT)")
    
    # PT+FT omega=1.0 - green solid
    if "omega=1.0" in results:
        alpha, mse = results["omega=1.0"]
        ax.plot(alpha, to_db(mse), "-", linewidth=2.5, color="green",
                label=f"PT+FT ω=1.0 (full overlap)")
    
    # PT+FT omega=0.0 - red solid
    if "omega=0.0" in results:
        alpha, mse = results["omega=0.0"]
        ax.plot(alpha, to_db(mse), "-", linewidth=2.5, color="red",
                label=f"PT+FT ω=0.0 (no overlap)")
    
    ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=14)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=14)
    ax.set_title(
        f"Baseline Comparison: Does PT Help?\n"
        f"ρ_pt={rho_pt}, ρ_ft={rho_ft}",
        fontsize=13
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')
    
    fig.tight_layout()
    
    png_path = os.path.join(output_dir, "ptft_vs_baseline.png")
    pdf_path = os.path.join(output_dir, "ptft_vs_baseline.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\n  Saved: {png_path}")


def run_ptft_overlap_sweep_all(args) -> None:
    """
    Run the complete PT+FT overlap sweep for all selected regimes.
    
    This is the main entry point when --ptft_overlap_sweep is specified.
    """
    print("=" * 80)
    print("PT+FT OVERLAP SWEEP MODE")
    print("=" * 80)
    print(f"Fixed PT parameters (Global Rules):")
    print(f"  c_pt = {PTFT_FIXED_C_PT}")
    print(f"  lambda_pt = {PTFT_FIXED_LAMBDA_PT}")
    print(f"  gamma_reinit = {PTFT_FIXED_GAMMA_REINIT}")
    print(f"  a_pt = {PTFT_FIXED_A_PT}")
    print(f"\nFT regulariser scale: {args.ft_regulariser_scale:.6e}")
    print(f"MC samples: {args.mc_samples}")
    print(f"Seed: {args.seed}")
    
    # Warn if c_values was specified
    if args.c_values != [0.001, 0.5]:
        print("\nWARNING: --c_values is ignored in ptft_overlap_sweep mode")
    
    # Determine which regimes to run
    if args.ptft_regime == "all":
        regime_list = list(PTFT_REGIMES.keys())
    else:
        regime_list = [args.ptft_regime]
    
    print(f"\nRegimes to process: {regime_list}")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, "replica_ptft")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process each regime
    for regime_name in regime_list:
        # Run sweep
        results = run_ptft_overlap_sweep(regime_name, args, output_dir)
        
        if not results:
            print(f"\n  WARNING: No valid results for regime {regime_name}")
            continue
        
        # Generate MSE plot only (active fraction not useful at small regularization)
        print(f"\n  Generating plots for {regime_name}...")
        plot_ptft_mse_sweep(results, regime_name, output_dir)
    
    # Generate additional comparison plots
    plot_ptft_ft_sparsity_comparison(args, output_dir)
    plot_ptft_dense_regime(args, output_dir)
    plot_ptft_vs_baseline(args, output_dir)
    
    print("\n" + "=" * 80)
    print("PT+FT OVERLAP SWEEP COMPLETE")
    print("=" * 80)


def build_config(
    rho: float,
    sigma0_2: float,
    beta_min: float,
    beta_max: float,
    beta_points: int,
    max_fp_iters: int,
    tol_fp: float,
    damp: float,
) -> Config:
    """Build Config for replica theory computation."""
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0, 1)")
    if not (0.0 <= sigma0_2):
        raise ValueError("sigma0_2 must be >= 0")
    if not (0.0 < beta_min < beta_max):
        raise ValueError("Require 0 < beta_min < beta_max")
    if beta_points < 2:
        raise ValueError("beta_points must be >= 2")
    if not (0.0 < damp <= 1.0):
        raise ValueError("damp must be in (0, 1]")
    
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


def compute_replica_curve(
    c: float,
    ft_regulariser_scale: float,
    alpha_range: np.ndarray,
    cfg: Config,
    x_mc: np.ndarray,
    v_mc: np.ndarray,
    k_mc: Optional[np.ndarray] = None,
    g_mc: Optional[np.ndarray] = None,
    k_mode: str = "homogeneous",
    teacher_mode: str = "bg",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute replica theory curve for given c value.
    
    Returns:
        alpha_values: array of alpha values
        mse_values: array of MSE values in alpha space
    """
    # Compute k_q from c (only used in homogeneous mode)
    # Correct mapping for complex diagonal-net init with lmda = 0:
    # w_pos = v_pos = sqrt(c/2)  =>  sqrt(k) = 2c  =>  k = (2c)^2
    k_q = (2.0 * c) ** 2
    
    # Compute gamma_ext based on mode
    k_mode_clean = (k_mode or "homogeneous").lower()
    if teacher_mode == "ptft_oracle":
        # For ptft_oracle: heterogeneity is in k_mc, so gamma_ext = ft_regulariser_scale directly
        gamma_ext = float(ft_regulariser_scale)
        print(f"  [ptft_oracle] ft_reg={ft_regulariser_scale:.6e}, gamma_ext={gamma_ext:.6e} (no k-scaling)")
    elif k_mode_clean == "homogeneous":
        # Homogeneous mode: scale gamma_ext using k_q (preserves old behavior)
        if k_q < 1.0:
            gamma_ext = gamma_ext_for_q_small(ft_regulariser_scale, k_q)
        else:
            gamma_ext = gamma_ext_for_q_big(ft_regulariser_scale, k_q)
        print(f"  c={c:.6f}, sqrt_k={math.sqrt(k_q):.6e}, k_q={k_q:.6e}, ft_reg={ft_regulariser_scale:.6e}, gamma_ext={gamma_ext:.6e}")
    else:
        # Hetero-k mode: gamma_ext independent of c, tied only to ft_regulariser_scale
        gamma_ext = gamma_ext_for_hetero_k(ft_regulariser_scale)
        print(f"  [hetero-k mode={k_mode_clean}] c={c:.6f} (ignored for gamma_ext), ft_reg={ft_regulariser_scale:.6e}, gamma_ext={gamma_ext:.6e}")
    
    # Convert alpha to beta (beta = 1/alpha)
    # beta increases as alpha decreases, so we need to reverse alpha_range
    # to get beta_range in increasing order (as required by the solver)
    alpha_reversed = alpha_range[::-1]  # [1.0, ..., 0.008] (decreasing)
    beta_range = 1.0 / alpha_reversed  # [1.0, ..., 125.0] (increasing)
    
    # Compute replica curve in beta space (beta_range is increasing)
    # Use hetero-k path when: explicit hetero-k mode OR ptft_oracle (which has per-coord k)
    use_hetero_k = (k_mode_clean != "homogeneous") or (teacher_mode == "ptft_oracle")
    
    if not use_hetero_k:
        # Homogeneous path: use scalar k_q, no k_mc
        mse_beta = solve_rspmap_qk_curve_best_of_forward_backward(
            beta_range, gamma_ext, k_q, x_mc, v_mc, cfg
        )
    else:
        # Hetero-k path: pass k_mc and g_mc, k_q is ignored by solver
        mse_beta = solve_rspmap_qk_curve_best_of_forward_backward(
            beta_range, gamma_ext, k_q, x_mc, v_mc, cfg, k_mc=k_mc, g_mc=g_mc
        )
    
    # mse_beta[i] corresponds to beta_range[i], which corresponds to alpha_reversed[i]
    # To match with original alpha_range (increasing), we need to reverse mse_beta
    mse_alpha = mse_beta[::-1]
    
    # Verify: alpha_range[0] = 0.008 should pair with beta = 125.0
    # alpha_reversed[-1] = 0.008, beta_range[-1] = 125.0, mse_beta[-1] should be used
    # After reversal: mse_alpha[0] = mse_beta[-1], which pairs with alpha_range[0] = 0.008 ✓
    
    return alpha_range, mse_alpha


def load_empirical_results(csv_path: str) -> pd.DataFrame:
    """Load aggregated empirical results."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Empirical results not found: {csv_path}")
    return pd.read_csv(csv_path)


def _nearest_row(df: pd.DataFrame, target_alpha: float) -> pd.Series:
    """Find row with alpha closest to target_alpha."""
    idx = (df["alpha"] - target_alpha).abs().idxmin()
    return df.loc[idx]


def plot_overlay(
    empirical_dfs: dict,
    replica_curves: dict,
    output_dir: str,
    rho: float,
    ft_regulariser_scale: float,
    plot_pred_mse_overlay: bool = False,
    inp_dim: int = 1000,
    n_test: int = 10000,
):
    """
    Create overlay plot of empirical and replica theory curves.
    
    Args:
        empirical_dfs: dict mapping c values to DataFrames with empirical results
        replica_curves: dict mapping c values to (alpha, mse) tuples
        output_dir: output directory for plots
        rho: sparsity parameter
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot empirical curves
    colors_emp = {'0.001': 'blue', '0.5': 'green'}
    colors_replica = {'0.001': 'red', '0.5': 'orange'}
    
    for c_str, df in empirical_dfs.items():
        c_val = float(c_str)
        df = df.sort_values("alpha")
        
        # Filter out rows with NaN param_mse values
        valid_mask = df["param_mse_mean"].notna() & df["param_mse_median"].notna()
        df_valid = df[valid_mask].copy()
        
        if len(df_valid) == 0:
            print(f"  WARNING: No valid data for c={c_val:.3f}, skipping empirical plot")
            continue
        
        # Convert to dB
        df_valid["param_mse_mean_db"] = to_db(df_valid["param_mse_mean"].values)
        df_valid["param_mse_median_db"] = to_db(df_valid["param_mse_median"].values)
        df_valid["param_mse_q25_db"] = to_db(df_valid["param_mse_q25"].values)
        df_valid["param_mse_q75_db"] = to_db(df_valid["param_mse_q75"].values)
        
        # Plot mean (solid)
        ax.plot(
            df_valid["alpha"],
            df_valid["param_mse_mean_db"],
            "o-",
            label=f"Empirical param-mse (c={c_val:.3f}, mean)",
            linewidth=2,
            markersize=5,
            color=colors_emp[c_str],
            alpha=0.8,
        )
        
        # Plot median (dashed)
        ax.plot(
            df_valid["alpha"],
            df_valid["param_mse_median_db"],
            "s--",
            label=f"Empirical param-mse (c={c_val:.3f}, median)",
            linewidth=2,
            markersize=4,
            color=colors_emp[c_str],
            alpha=0.6,
        )
        
        # Fill IQR (only where both q25 and q75 are valid)
        q_valid_mask = df_valid["param_mse_q25"].notna() & df_valid["param_mse_q75"].notna()
        if q_valid_mask.sum() > 0:
            ax.fill_between(
                df_valid.loc[q_valid_mask, "alpha"],
                df_valid.loc[q_valid_mask, "param_mse_q25_db"],
                df_valid.loc[q_valid_mask, "param_mse_q75_db"],
                alpha=0.15,
                color=colors_emp[c_str],
            )
    
    # Plot replica theory curves (parameter MSE)
    for c_str, (alpha_vals, mse_vals) in replica_curves.items():
        c_val = float(c_str)
        mse_db = to_db(mse_vals)
        
        ax.plot(
            alpha_vals,
            mse_db,
            "-",
            label=f"Replica q (c={c_val:.3f})",
            linewidth=2.5,
            color=colors_replica[c_str],
            alpha=0.9,
        )
        
        # Optional: overlay converted prediction-MSE replica curve
        if plot_pred_mse_overlay:
            scale = inp_dim / n_test
            pred_mse_replica = scale * mse_vals
            pred_mse_replica_db = to_db(pred_mse_replica)
            
            ax.plot(
                alpha_vals,
                pred_mse_replica_db,
                "--",
                label=f"Replica q (pred-mse converted, c={c_val:.3f})",
                linewidth=2.0,
                color=colors_replica[c_str],
                alpha=0.7,
            )
    
    ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=14)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=14)
    ax.set_title(
        f"Generalization Curves: Empirical vs Replica Theory\n"
        f"Bernoulli-Gaussian ($\\rho={rho:.3f}$), Noiseless",
        fontsize=14
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best', ncol=2)
    
    fig.tight_layout()
    
    # Save plots with ft_regulariser_scale in filename
    ft_reg_tag = f"{ft_regulariser_scale:.0e}".replace("+", "")
    png_path = os.path.join(output_dir, f"replica_overlay_ft_reg={ft_reg_tag}.png")
    pdf_path = os.path.join(output_dir, f"replica_overlay_ft_reg={ft_reg_tag}.pdf")
    
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nOverlay plot saved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate replica theory curves for diagonal network q-function"
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.04,
        help="Sparsity parameter for Bernoulli-Gaussian (default: 0.04)",
    )
    parser.add_argument(
        "--ft_regulariser_scale",
        type=float,
        default=1e-6,
        help="FT regularization strength (default: 1e-6)",
    )
    parser.add_argument(
        "--c_values",
        type=float,
        nargs="+",
        default=[0.001, 0.5],
        help="C values to compute curves for (default: 0.001 0.5)",
    )
    parser.add_argument(
        "--alpha_min",
        type=float,
        default=0.008,
        help="Minimum alpha value (default: 0.008)",
    )
    parser.add_argument(
        "--alpha_max",
        type=float,
        default=1.0,
        help="Maximum alpha value (default: 1.0)",
    )
    parser.add_argument(
        "--alpha_points",
        type=int,
        default=100,
        help="Number of alpha points (default: 100)",
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=50000,
        help="Monte Carlo samples (default: 50000)",
    )
    parser.add_argument(
        "--max_fp_iters",
        type=int,
        default=900,
        help="Max fixed point iterations (default: 900)",
    )
    parser.add_argument(
        "--tol_fp",
        type=float,
        default=1e-10,
        help="Fixed point tolerance (default: 1e-10)",
    )
    parser.add_argument(
        "--damp",
        type=float,
        default=0.25,
        help="Damping factor (default: 0.25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed (default: 12345)",
    )
    parser.add_argument(
        "--empirical_dir",
        type=str,
        default="figures/diagonal/bg_generalization",
        help="Directory with empirical results CSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures/diagonal/bg_generalization",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--plot_pred_mse_overlay",
        action="store_true",
        help="Optionally overlay converted prediction-MSE replica curve",
    )
    parser.add_argument(
        "--inp_dim",
        type=int,
        default=1000,
        help="Input dimension for pred-MSE conversion (default: 1000)",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=10000,
        help="Number of test samples for pred-MSE conversion (default: 10000)",
    )
    parser.add_argument(
        "--k_mode",
        type=str,
        choices=["homogeneous", "mixture", "support"],
        default="homogeneous",
        help="Mode for coordinate-wise k: homogeneous (default), mixture, or support",
    )
    parser.add_argument(
        "--k_A",
        type=float,
        default=None,
        help="Group-A k value for mixture mode (ignored otherwise)",
    )
    parser.add_argument(
        "--k_B",
        type=float,
        default=None,
        help="Group-B k value for mixture mode (ignored otherwise)",
    )
    parser.add_argument(
        "--pi_A",
        type=float,
        default=None,
        help="Probability a coordinate belongs to group A in mixture mode (ignored otherwise)",
    )
    parser.add_argument(
        "--k_nz",
        type=float,
        default=None,
        help="k value for nonzero teacher coordinates in support mode (ignored otherwise)",
    )
    parser.add_argument(
        "--k_z",
        type=float,
        default=None,
        help="k value for zero teacher coordinates in support mode (ignored otherwise)",
    )
    parser.add_argument(
        "--debug_groups",
        action="store_true",
        help="Enable extra diagnostics for heterogeneous-k and ptft_oracle group statistics",
    )
    
    # Teacher mode
    parser.add_argument(
        "--teacher_mode",
        type=str,
        choices=["bg", "ptft_oracle"],
        default="bg",
        help="Teacher mode: bg (Bernoulli-Gaussian, default) or ptft_oracle (Step 2A)",
    )
    
    # PT+FT Oracle parameters (Step 2A)
    parser.add_argument(
        "--rho_pt",
        type=float,
        default=None,
        help="PT support fraction (required for ptft_oracle)",
    )
    parser.add_argument(
        "--rho_ft",
        type=float,
        default=None,
        help="FT teacher sparsity (required for ptft_oracle)",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=None,
        help="Overlap fraction |S_pt ∩ S_ft| / |S_ft| (required for ptft_oracle)",
    )
    parser.add_argument(
        "--a_pt",
        type=float,
        default=None,
        help="Deterministic PT ground truth amplitude (required for ptft_oracle)",
    )
    parser.add_argument(
        "--c_pt",
        type=float,
        default=None,
        help="PT parameter c (required for ptft_oracle)",
    )
    parser.add_argument(
        "--lambda_pt",
        type=float,
        default=None,
        help="PT initialization parameter λ (NOT regularizer scale! required for ptft_oracle)",
    )
    parser.add_argument(
        "--gamma_reinit",
        type=float,
        default=None,
        help="Readout reinitialization parameter (required for ptft_oracle)",
    )
    
    # PT+FT Overlap Sweep mode (Tasks B & C)
    parser.add_argument(
        "--ptft_overlap_sweep",
        action="store_true",
        help="Enable PT+FT overlap sweep mode: generates MSE and active fraction plots "
             "for multiple sparsity regimes with varying omega values",
    )
    parser.add_argument(
        "--ptft_regime",
        type=str,
        choices=["sparse_sparse", "sparse_verysparse", "sparse_dense", "dense_sparse", "all"],
        default="all",
        help="Sparsity regime for ptft_overlap_sweep: "
             "sparse_sparse (rho_pt=rho_ft=0.1), "
             "sparse_verysparse (rho_pt=0.1, rho_ft=0.04), "
             "sparse_dense (rho_pt=0.1, rho_ft=1.0), "
             "dense_sparse (rho_pt=1.0, rho_ft=0.1), "
             "or 'all' to generate all regimes (default: all)",
    )
    
    args = parser.parse_args()
    
    # Handle PT+FT overlap sweep mode (Tasks B & C)
    if args.ptft_overlap_sweep:
        run_ptft_overlap_sweep_all(args)
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build configuration
    sigma0_2 = 0.0  # Noiseless
    beta_min = 1.0 / args.alpha_max  # beta = 1/alpha
    beta_max = 1.0 / args.alpha_min
    
    cfg = build_config(
        rho=args.rho,
        sigma0_2=sigma0_2,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_points=args.alpha_points,
        max_fp_iters=args.max_fp_iters,
        tol_fp=args.tol_fp,
        damp=args.damp,
    )
    
    print("="*80)
    print("REPLICA THEORY Q-FUNCTION CURVES")
    print("="*80)
    print(f"rho = {args.rho}")
    print(f"sigma0^2 = {sigma0_2} (noiseless)")
    print(f"ft_regulariser_scale = {args.ft_regulariser_scale:.6e}")
    print(f"c values = {args.c_values}")
    print(f"alpha range: [{args.alpha_min:.4f}, {args.alpha_max:.4f}] ({args.alpha_points} points)")
    print(f"beta range: [{beta_min:.4f}, {beta_max:.4f}]")
    print(f"MC samples = {args.mc_samples}")
    print("="*80)

    # Validate hetero-k arguments without changing homogeneous defaults.
    k_mode = (args.k_mode or "homogeneous").lower()
    if k_mode == "mixture":
        if args.k_A is None or args.k_B is None or args.pi_A is None:
            raise ValueError("k_mode='mixture' requires --k_A, --k_B, and --pi_A.")
        if not (0.0 < args.pi_A < 1.0):
            raise ValueError("--pi_A must be in (0, 1) for k_mode='mixture'.")
        if args.k_A <= 0.0 or args.k_B <= 0.0:
            raise ValueError("--k_A and --k_B must be > 0 for k_mode='mixture'.")
    elif k_mode == "support":
        if args.k_nz is None or args.k_z is None:
            raise ValueError("k_mode='support' requires --k_nz and --k_z.")
        if args.k_nz <= 0.0 or args.k_z <= 0.0:
            raise ValueError("--k_nz and --k_z must be > 0 for k_mode='support'.")
    
    # Generate Monte Carlo samples
    print("\nGenerating Monte Carlo samples...")
    rng = np.random.default_rng(args.seed)
    
    teacher_mode = (args.teacher_mode or "bg").lower()
    
    if teacher_mode == "bg":
        # Existing BG path - UNCHANGED
        # For support mode, we need the BG mask
        if k_mode == "support":
            x_mc, mask_bg = sample_bg(args.mc_samples, rng, args.rho, cfg.var_nonzero, return_mask=True)
        else:
            x_mc = sample_bg(args.mc_samples, rng, args.rho, cfg.var_nonzero, return_mask=False)
            mask_bg = None
        v_mc = rng.normal(size=args.mc_samples)
        print(f"Generated {args.mc_samples} MC samples")
        
        # Sample heterogeneous k values once per run when enabled.
        k_mc: Optional[np.ndarray] = None
        g_mc: Optional[np.ndarray] = None
        if k_mode != "homogeneous":
            if k_mode == "mixture":
                k_cfg = KModeConfig(
                    mode="mixture",
                    k_A=args.k_A,
                    k_B=args.k_B,
                    pi_A=args.pi_A,
                )
            elif k_mode == "support":
                k_cfg = KModeConfig(
                    mode="support",
                    k_nz=args.k_nz,
                    k_z=args.k_z,
                )
            else:
                raise ValueError(f"Unexpected k_mode: {k_mode}")
            k_mc, g_mc = sample_k_mc(k_cfg, x_mc, rng, mask_bg=mask_bg)
            if args.debug_groups:
                if k_mode == "mixture":
                    frac_A = float(np.mean(g_mc == 1)) if g_mc is not None else float(np.mean(k_mc == float(args.k_A)))
                    print(f"[debug_groups] mixture mode: empirical pi_A ≈ {frac_A:.6f}")
                elif k_mode == "support":
                    frac_nz = float(np.mean(g_mc == 1)) if g_mc is not None else float(np.mean(x_mc != 0.0))
                    print(f"[debug_groups] support mode: empirical rho_nz ≈ {frac_nz:.6f}")
    
    elif teacher_mode == "ptft_oracle":
        # Validate required args
        required_args = ["rho_pt", "rho_ft", "omega", "a_pt", "c_pt", "lambda_pt", "gamma_reinit"]
        missing = [name for name in required_args if getattr(args, name) is None]
        if missing:
            raise ValueError(
                f"ptft_oracle mode requires: {', '.join('--' + m for m in required_args)}. "
                f"Missing: {', '.join('--' + m for m in missing)}"
            )
        
        # Warn if c_values is non-default (it's ignored in this mode)
        if args.c_values != [0.001, 0.5]:
            print("WARNING: --c_values is ignored in ptft_oracle mode")
        
        # Create config
        ptft_cfg = PTFTOracleConfig(
            rho_pt=args.rho_pt,
            rho_ft=args.rho_ft,
            omega=args.omega,
            a_pt=args.a_pt,
            c_pt=args.c_pt,
            lambda_pt=args.lambda_pt,
            gamma_reinit=args.gamma_reinit,
        )
        
        # Sample MC (this validates feasibility and raises clear errors)
        beta_ft_mc, beta_pt_mc, k_mc, g_mc = sample_ptft_oracle_mc(ptft_cfg, rng, args.mc_samples)
        
        # FT ground truth is the teacher for the solver
        x_mc = beta_ft_mc
        v_mc = rng.normal(size=args.mc_samples)
        
        print(f"Generated {args.mc_samples} PT+FT oracle MC samples")
        
        # Debug groups
        if args.debug_groups:
            group_names = ["OV", "NEW", "PTONLY", "NONE"]
            print(f"\n[debug_groups] ptft_oracle mode:")
            print(f"  Theoretical probabilities:")
            p_ov = args.omega * args.rho_ft
            p_new = (1.0 - args.omega) * args.rho_ft
            p_ptonly = args.rho_pt - p_ov
            p_none = 1.0 - args.rho_pt - p_new
            for i, (name, p) in enumerate(zip(group_names, [p_ov, p_new, p_ptonly, p_none])):
                print(f"    {i} ({name:7s}): {p:.6f}")
            
            print(f"  Empirical fractions:")
            for i, name in enumerate(group_names):
                frac = float(np.mean(g_mc == i))
                print(f"    {i} ({name:7s}): {frac:.6f}")
            
            # Empirical overlap check
            ft_nonzero = (g_mc == 0) | (g_mc == 1)
            if ft_nonzero.sum() > 0:
                ov_frac = float(np.mean(g_mc[ft_nonzero] == 0))
                print(f"  Overlap among FT-nonzero: {ov_frac:.6f} (target: {args.omega:.6f})")
    
    else:
        raise ValueError(f"Unknown teacher_mode: {teacher_mode}")
    
    # Create alpha range
    alpha_range = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    
    # Check for cached replica curves
    replica_cache_dir = os.path.join(args.output_dir, "replica_cache")
    Path(replica_cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Compute replica curves for each c value (with caching)
    print("\nComputing replica theory curves...")
    replica_curves = {}
    replica_metadata = {}  # Store k_q and gamma_ext for each c
    for c in args.c_values:
        c_str = str(c)
        # Compute k_q from c (only used in homogeneous mode)
        k_q = (2.0 * c) ** 2
        
        # Compute gamma_ext based on mode (consistent with compute_replica_curve logic)
        k_mode_clean = (k_mode or "homogeneous").lower()
        if teacher_mode == "ptft_oracle":
            # For ptft_oracle: gamma_ext = ft_regulariser_scale directly (no k-scaling)
            gamma_ext = float(args.ft_regulariser_scale)
        elif k_mode_clean == "homogeneous":
            # Homogeneous mode: scale gamma_ext using k_q
            if k_q < 1.0:
                gamma_ext = gamma_ext_for_q_small(args.ft_regulariser_scale, k_q)
            else:
                gamma_ext = gamma_ext_for_q_big(args.ft_regulariser_scale, k_q)
        else:
            # Hetero-k modes: gamma_ext = ft_regulariser_scale directly
            gamma_ext = float(args.ft_regulariser_scale)
        
        replica_metadata[c_str] = {"k_q": k_q, "gamma_ext": gamma_ext}
        
        # For ptft_oracle mode, c is irrelevant (k is derived from PT params)
        if teacher_mode == "ptft_oracle":
            c_for_cache = "IGNORED"
        else:
            c_for_cache = f"{c:.6f}"
        
        # Create cache filename based on parameters (extend with hetero-k when enabled)
        cache_filename = (
            f"replica_curve_teacher={teacher_mode}--"
            f"rho={args.rho:.6f}--c={c_for_cache}--"
            f"ft_reg={args.ft_regulariser_scale:.6e}--alpha_min={args.alpha_min:.4f}--"
            f"alpha_max={args.alpha_max:.4f}--alpha_points={args.alpha_points}--"
            f"mc_samples={args.mc_samples}--seed={args.seed}"
        )
        # Add mode-specific suffixes
        if teacher_mode == "ptft_oracle":
            cache_filename += (
                f"--rpt={args.rho_pt:.4f}--rft={args.rho_ft:.4f}"
                f"--om={args.omega:.4f}--apt={args.a_pt:.2f}"
                f"--cpt={args.c_pt:.4f}--lpt={args.lambda_pt:.2f}"
                f"--gam={args.gamma_reinit:.2f}"
            )
        elif k_mode == "mixture":
            cache_filename += (
                f"--k_mode=mixture--k_A={args.k_A:.6e}"
                f"--k_B={args.k_B:.6e}--pi_A={args.pi_A:.6e}"
            )
        elif k_mode == "support":
            cache_filename += (
                f"--k_mode=support--k_nz={args.k_nz:.6e}"
                f"--k_z={args.k_z:.6e}"
            )
        cache_filename += ".csv"
        cache_path = os.path.join(replica_cache_dir, cache_filename)
        
        # Try to load from cache
        if os.path.exists(cache_path):
            print(f"\nLoading cached replica curve for c = {c:.6f}...")
            try:
                df_cache = pd.read_csv(cache_path)
                alpha_vals = df_cache["alpha"].values
                mse_vals = df_cache["mse"].values
                replica_curves[c_str] = (alpha_vals, mse_vals)
                print(f"  Loaded from cache. MSE range: [{mse_vals.min():.6e}, {mse_vals.max():.6e}]")
            except Exception as e:
                print(f"  WARNING: Failed to load cache ({e}), recomputing...")
                # Compute if cache load failed
                print(f"\nProcessing c = {c:.6f}...")
                alpha_vals, mse_vals = compute_replica_curve(
                    c,
                    args.ft_regulariser_scale,
                    alpha_range,
                    cfg,
                    x_mc,
                    v_mc,
                    k_mc=k_mc,
                    g_mc=g_mc,
                    k_mode=k_mode,
                    teacher_mode=teacher_mode,
                )
                replica_curves[c_str] = (alpha_vals, mse_vals)
                print(f"  Done. MSE range: [{mse_vals.min():.6e}, {mse_vals.max():.6e}]")
                
                # Save to cache
                df_cache = pd.DataFrame({"alpha": alpha_vals, "mse": mse_vals})
                df_cache.to_csv(cache_path, index=False)
                print(f"  Saved to cache: {cache_path}")
        else:
            # Compute if not cached
            print(f"\nProcessing c = {c:.6f}...")
            alpha_vals, mse_vals = compute_replica_curve(
                c,
                args.ft_regulariser_scale,
                alpha_range,
                cfg,
                x_mc,
                v_mc,
                k_mc=k_mc,
                g_mc=g_mc,
                k_mode=k_mode,
                teacher_mode=teacher_mode,
            )
            replica_curves[c_str] = (alpha_vals, mse_vals)
            print(f"  Done. MSE range: [{mse_vals.min():.6e}, {mse_vals.max():.6e}]")
            
            # Save to cache
            df_cache = pd.DataFrame({"alpha": alpha_vals, "mse": mse_vals})
            df_cache.to_csv(cache_path, index=False)
            print(f"  Saved to cache: {cache_path}")
        
        # Save .npz file for quantitative comparison
        ft_reg_tag = f"{args.ft_regulariser_scale:.0e}".replace("+", "")
        npz_filename = f"replica_curve_ft_reg={ft_reg_tag}--c={c:.6f}"
        if teacher_mode == "ptft_oracle":
            npz_filename += (
                f"--teacher=ptft--rpt={args.rho_pt:.4f}--rft={args.rho_ft:.4f}"
                f"--om={args.omega:.4f}--apt={args.a_pt:.2f}"
                f"--cpt={args.c_pt:.4f}--lpt={args.lambda_pt:.2f}"
                f"--gam={args.gamma_reinit:.2f}"
            )
        elif k_mode == "mixture":
            npz_filename += (
                f"--k_mode=mixture--k_A={args.k_A:.6e}"
                f"--k_B={args.k_B:.6e}--pi_A={args.pi_A:.6e}"
            )
        elif k_mode == "support":
            npz_filename += (
                f"--k_mode=support--k_nz={args.k_nz:.6e}"
                f"--k_z={args.k_z:.6e}"
            )
        npz_filename += ".npz"
        npz_path = os.path.join(args.output_dir, npz_filename)
        np.savez_compressed(
            npz_path,
            alpha_vals=alpha_vals,
            mse_vals=mse_vals,
            ft_regulariser_scale=args.ft_regulariser_scale,
            gamma_ext=gamma_ext,
            k_q=k_q,
        )
        print(f"  Saved .npz array: {npz_path}")
    
    # Load empirical results
    print("\nLoading empirical results...")
    empirical_dfs = {}
    for c in args.c_values:
        c_str = str(c)
        # Try SUCCESS.csv first, then regular CSV, then old naming convention
        csv_path_success = os.path.join(
            args.empirical_dir,
            f"aggregated_results_rho={args.rho:.6f}--c={c:.6f}--SUCCESS.csv"
        )
        csv_path1 = os.path.join(
            args.empirical_dir,
            f"aggregated_results_rho={args.rho:.6f}--c={c:.6f}.csv"
        )
        csv_path2 = os.path.join(
            args.empirical_dir,
            f"aggregated_results_rho={args.rho:.6f}.csv"
        )
        
        csv_path_all = os.path.join(
            args.empirical_dir,
            f"aggregated_results_rho={args.rho:.6f}--c={c:.6f}--ALL.csv"
        )
        
        if os.path.exists(csv_path_success):
            df = load_empirical_results(csv_path_success)
            # Check if SUCCESS.csv has valid data (not all NaN)
            if df["param_mse_mean"].notna().sum() == 0:
                print(f"  SUCCESS.csv for c={c:.6f} is empty, trying ALL.csv...")
                # Fall through to try ALL.csv or regular CSV
            else:
                empirical_dfs[c_str] = df
                print(f"  Loaded empirical results for c={c:.6f} from {csv_path_success} (SUCCESS)")
                # Skip to debug prints
                print(f"    [Empirical sanity c={c_str}] alpha range: [{df['alpha'].min():.6f}, {df['alpha'].max():.6f}]")
                
                row07 = _nearest_row(df, 0.7)
                row10 = _nearest_row(df, 1.0)
                
                for name, r in [("alpha~0.7", row07), ("alpha~1.0", row10)]:
                    if pd.notna(r["param_mse_mean"]) and r["param_mse_mean"] > 0:
                        ratio = r["test_pred_mse_mean"] / max(r["param_mse_mean"], 1e-30)
                        print(
                            f"    [Empirical sanity c={c_str}] {name}: "
                            f"param_mean={r['param_mse_mean']:.6e}, "
                            f"param_median={r['param_mse_median']:.6e}, "
                            f"pred_mean={r['test_pred_mse_mean']:.6e}, "
                            f"pred_median={r['test_pred_mse_median']:.6e}, "
                            f"pred/param(mean)={ratio:.6e}"
                        )
                continue
        
        # Try ALL.csv as fallback
        if os.path.exists(csv_path_all):
            df = load_empirical_results(csv_path_all)
            # Check if ALL.csv has valid data (not all NaN)
            if df["param_mse_mean"].notna().sum() == 0:
                print(f"  ALL.csv for c={c:.6f} is empty, trying regular CSV...")
                # Fall through to try regular CSV
            else:
                empirical_dfs[c_str] = df
                print(f"  Loaded empirical results for c={c:.6f} from {csv_path_all} (ALL)")
                
                # Debug prints
                print(f"    [Empirical sanity c={c_str}] alpha range: [{df['alpha'].min():.6f}, {df['alpha'].max():.6f}]")
                
                row07 = _nearest_row(df, 0.7)
                row10 = _nearest_row(df, 1.0)
                
                for name, r in [("alpha~0.7", row07), ("alpha~1.0", row10)]:
                    if pd.notna(r["param_mse_mean"]) and r["param_mse_mean"] > 0:
                        ratio = r["test_pred_mse_mean"] / max(r["param_mse_mean"], 1e-30)
                        print(
                            f"    [Empirical sanity c={c_str}] {name}: "
                            f"param_mean={r['param_mse_mean']:.6e}, "
                            f"param_median={r['param_mse_median']:.6e}, "
                            f"pred_mean={r['test_pred_mse_mean']:.6e}, "
                            f"pred_median={r['test_pred_mse_median']:.6e}, "
                            f"pred/param(mean)={ratio:.6e}"
                        )
                continue
        
        # Try regular CSV file
        if os.path.exists(csv_path1):
            df = load_empirical_results(csv_path1)
            empirical_dfs[c_str] = df
            print(f"  Loaded empirical results for c={c:.6f} from {csv_path1}")
            
            # Debug prints
            print(f"    [Empirical sanity c={c_str}] alpha range: [{df['alpha'].min():.6f}, {df['alpha'].max():.6f}]")
            
            row07 = _nearest_row(df, 0.7)
            row10 = _nearest_row(df, 1.0)
            
            for name, r in [("alpha~0.7", row07), ("alpha~1.0", row10)]:
                ratio = r["test_pred_mse_mean"] / max(r["param_mse_mean"], 1e-30)
                print(
                    f"    [Empirical sanity c={c_str}] {name}: "
                    f"param_mean={r['param_mse_mean']:.6e}, "
                    f"param_median={r['param_mse_median']:.6e}, "
                    f"pred_mean={r['test_pred_mse_mean']:.6e}, "
                    f"pred_median={r['test_pred_mse_median']:.6e}, "
                    f"pred/param(mean)={ratio:.6e}"
                )
        elif os.path.exists(csv_path2) and c == 0.001:
            # For c=0.001, might be in the file without c suffix
            df = load_empirical_results(csv_path2)
            empirical_dfs[c_str] = df
            print(f"  Loaded empirical results for c={c:.6f} from {csv_path2}")
            
            # Debug prints
            print(f"    [Empirical sanity c={c_str}] alpha range: [{df['alpha'].min():.6f}, {df['alpha'].max():.6f}]")
            
            row07 = _nearest_row(df, 0.7)
            row10 = _nearest_row(df, 1.0)
            
            for name, r in [("alpha~0.7", row07), ("alpha~1.0", row10)]:
                ratio = r["test_pred_mse_mean"] / max(r["param_mse_mean"], 1e-30)
                print(
                    f"    [Empirical sanity c={c_str}] {name}: "
                    f"param_mean={r['param_mse_mean']:.6e}, "
                    f"param_median={r['param_mse_median']:.6e}, "
                    f"pred_mean={r['test_pred_mse_mean']:.6e}, "
                    f"pred_median={r['test_pred_mse_median']:.6e}, "
                    f"pred/param(mean)={ratio:.6e}"
                )
        else:
            print(f"  WARNING: No empirical results found for c={c:.6f}")
    
    # Create overlay plot
    if empirical_dfs and replica_curves:
        print("\nCreating overlay plot...")
        plot_overlay(
            empirical_dfs, 
            replica_curves, 
            args.output_dir, 
            args.rho,
            args.ft_regulariser_scale,
            plot_pred_mse_overlay=args.plot_pred_mse_overlay,
            inp_dim=args.inp_dim,
            n_test=args.n_test,
        )
    else:
        print("\nWARNING: Cannot create overlay plot - missing data")
        if not empirical_dfs:
            print("  No empirical results loaded")
        if not replica_curves:
            print("  No replica curves computed")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

