#!/usr/bin/env python3
"""
Step 1 Validation: Overlay empirical mixture-k vs replica curves.

This script:
1. Aggregates empirical results from step1_mixture experiments
2. Generates replica curves with matching mixture-k parameters
3. Creates overlay plots for validation
4. Checks acceptance criteria

Usage:
    python scripts/diagonal/plot_step1_validation.py --output_dir figures/validation_step1
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import replica curve generation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ReplicaExperiments.fixed_lambda_all import (
    Config,
    KModeConfig,
    gamma_ext_for_q_small,
    gamma_ext_for_q_big,
    solve_rspmap_qk_curve_best_of_forward_backward,
    sample_bg,
    sample_k_mc,
)


def to_db(x: np.ndarray) -> np.ndarray:
    """Convert MSE to dB: 10*log10(mse + 1e-15)"""
    return 10.0 * np.log10(np.maximum(x, 1e-15))


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
    """Build Config for replica theory computation."""
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


def gamma_ext_for_hetero_k(ft_regulariser_scale: float) -> float:
    """For heterogeneous-k, gamma_ext = ft_regulariser_scale directly."""
    return float(ft_regulariser_scale)


def compute_replica_curve_mixture(
    k_A: float,
    k_B: float,
    pi_A: float,
    ft_regulariser_scale: float,
    alpha_range: np.ndarray,
    cfg: Config,
    x_mc: np.ndarray,
    v_mc: np.ndarray,
    rng: np.random.Generator,
) -> tuple:
    """Compute replica theory curve for mixture-k mode."""
    # Sample k_mc using mixture mode
    k_cfg = KModeConfig(
        mode="mixture",
        k_A=k_A,
        k_B=k_B,
        pi_A=pi_A,
    )
    k_mc, g_mc = sample_k_mc(k_cfg, x_mc, rng)
    
    # Gamma_ext for hetero-k
    gamma_ext = gamma_ext_for_hetero_k(ft_regulariser_scale)
    
    print(f"  [mixture] k_A={k_A:.6e}, k_B={k_B:.6e}, pi_A={pi_A:.4f}")
    print(f"  gamma_ext={gamma_ext:.6e}")
    
    # Convert alpha to beta
    alpha_reversed = alpha_range[::-1]
    beta_range = 1.0 / alpha_reversed
    
    # Use hetero-k path (k_q is ignored when k_mc is provided)
    k_q = 1.0  # Placeholder
    mse_beta = solve_rspmap_qk_curve_best_of_forward_backward(
        beta_range, gamma_ext, k_q, x_mc, v_mc, cfg, k_mc=k_mc, g_mc=g_mc
    )
    
    mse_alpha = mse_beta[::-1]
    return alpha_range, mse_alpha


def aggregate_empirical_results(csv_path: str) -> pd.DataFrame:
    """Load and aggregate empirical results by (pi_A, alpha)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Aggregate by (pi_A, alpha)
    agg = df.groupby(['pi_A', 'alpha']).agg({
        'param_mse': ['mean', 'median', 'std', 'count',
                      lambda x: np.percentile(x, 25),
                      lambda x: np.percentile(x, 75)],
        'test_pred_mse': ['mean', 'median'],
        'train_pred_mse': ['mean', 'median'],
    }).reset_index()
    
    # Flatten columns
    agg.columns = [
        'pi_A', 'alpha',
        'param_mse_mean', 'param_mse_median', 'param_mse_std', 'count',
        'param_mse_q25', 'param_mse_q75',
        'test_pred_mse_mean', 'test_pred_mse_median',
        'train_pred_mse_mean', 'train_pred_mse_median',
    ]
    
    return agg


def check_acceptance_criteria(agg_df: pd.DataFrame, replica_curves: dict) -> dict:
    """Check Step 1 acceptance criteria."""
    results = {
        "ordering_correct": False,
        "smooth_interpolation": False,
        "collapse_sanity": None,  # Requires c_A == c_B run (not in this sweep)
    }
    
    pi_A_values = sorted(agg_df['pi_A'].unique())
    
    if len(pi_A_values) < 2:
        print("WARNING: Need at least 2 pi_A values to check ordering")
        return results
    
    # Check ordering: larger pi_A should shift toward c_A behavior
    # At high alpha (near 1.0), c_A=0.001 (small k) should give lower MSE than c_B=0.5 (large k)
    # So larger pi_A (more c_A) should give lower MSE at high alpha
    
    ordering_checks = []
    for alpha in [0.7, 0.8, 0.9, 1.0]:
        mses_at_alpha = []
        for pi_A in pi_A_values:
            mask = (agg_df['pi_A'] == pi_A) & (np.abs(agg_df['alpha'] - alpha) < 0.01)
            if mask.sum() > 0:
                mse = agg_df.loc[mask, 'param_mse_mean'].values[0]
                mses_at_alpha.append((pi_A, mse))
        
        if len(mses_at_alpha) >= 2:
            # Check if MSE decreases as pi_A increases (more c_A)
            sorted_by_pi = sorted(mses_at_alpha, key=lambda x: x[0])
            mse_decreasing = all(
                sorted_by_pi[i][1] >= sorted_by_pi[i+1][1] 
                for i in range(len(sorted_by_pi)-1)
            )
            ordering_checks.append(mse_decreasing)
    
    results["ordering_correct"] = all(ordering_checks) if ordering_checks else False
    
    # Check smooth interpolation: std should be reasonable
    mean_std = agg_df['param_mse_std'].mean()
    mean_val = agg_df['param_mse_mean'].mean()
    cv = mean_std / mean_val if mean_val > 0 else float('inf')
    results["smooth_interpolation"] = cv < 0.5  # CV < 50%
    
    return results


def plot_step1_overlay(
    agg_df: pd.DataFrame,
    replica_curves: dict,
    output_dir: str,
    acceptance_results: dict,
):
    """Create overlay plot for Step 1 validation."""
    pi_A_values = sorted(agg_df['pi_A'].unique())
    
    # Color map for pi_A values
    colors = {0.1: ('blue', 'darkblue'), 0.5: ('green', 'darkgreen'), 0.9: ('red', 'darkred')}
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for pi_A in pi_A_values:
        emp_color, rep_color = colors.get(pi_A, ('gray', 'black'))
        
        # Empirical data
        mask = agg_df['pi_A'] == pi_A
        df_pi = agg_df[mask].sort_values('alpha')
        
        if len(df_pi) > 0:
            ax.plot(
                df_pi['alpha'],
                to_db(df_pi['param_mse_mean'].values),
                'o-',
                label=f'Empirical π_A={pi_A:.1f}',
                linewidth=2,
                markersize=5,
                color=emp_color,
            )
            
            # IQR fill
            if 'param_mse_q25' in df_pi.columns:
                ax.fill_between(
                    df_pi['alpha'],
                    to_db(df_pi['param_mse_q25'].values),
                    to_db(df_pi['param_mse_q75'].values),
                    alpha=0.15,
                    color=emp_color,
                )
        
        # Replica curve
        pi_str = str(pi_A)
        if pi_str in replica_curves:
            alpha_rep, mse_rep = replica_curves[pi_str]
            ax.plot(
                alpha_rep,
                to_db(mse_rep),
                '--',
                label=f'Replica π_A={pi_A:.1f}',
                linewidth=2.5,
                color=rep_color,
            )
    
    # Build status string
    status_parts = []
    if acceptance_results.get("ordering_correct"):
        status_parts.append("Ordering: ✓")
    else:
        status_parts.append("Ordering: ✗")
    if acceptance_results.get("smooth_interpolation"):
        status_parts.append("Smooth: ✓")
    else:
        status_parts.append("Smooth: ✗")
    status_str = ", ".join(status_parts)
    
    ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=12)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=12)
    ax.set_title(
        f"Step 1 Validation: Mixture-k Heterogeneous Init\n"
        f"c_A=0.001, c_B=0.5, ρ=0.04 | {status_str}",
        fontsize=11
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best', ncol=2)
    
    fig.tight_layout()
    
    png_path = os.path.join(output_dir, "step1_validation_overlay.png")
    pdf_path = os.path.join(output_dir, "step1_validation_overlay.pdf")
    
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nOverlay plot saved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 1 Validation: Mixture-k")
    parser.add_argument("--rho", type=float, default=0.04)
    parser.add_argument("--c_A", type=float, default=0.001)
    parser.add_argument("--c_B", type=float, default=0.5)
    parser.add_argument("--pi_A_values", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--ft_regulariser_scale", type=float, default=1e-6)
    parser.add_argument("--alpha_min", type=float, default=0.008)
    parser.add_argument("--alpha_max", type=float, default=1.0)
    parser.add_argument("--alpha_points", type=int, default=100)
    parser.add_argument("--mc_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--empirical_csv", type=str, 
                        default="experiment_results_step1_mixture.csv")
    parser.add_argument("--output_dir", type=str, 
                        default="figures/validation_step1")
    parser.add_argument("--skip_replica", action="store_true")
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "replica_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("STEP 1 VALIDATION: MIXTURE-K HETEROGENEOUS INIT")
    print("="*80)
    print(f"rho = {args.rho}")
    print(f"c_A = {args.c_A}, c_B = {args.c_B}")
    print(f"pi_A values = {args.pi_A_values}")
    print("="*80)
    
    # Compute k values from c
    k_A = (2.0 * args.c_A) ** 2
    k_B = (2.0 * args.c_B) ** 2
    print(f"k_A = (2*c_A)^2 = {k_A:.6e}")
    print(f"k_B = (2*c_B)^2 = {k_B:.6e}")
    
    # Build config
    sigma0_2 = 0.0
    beta_min = 1.0 / args.alpha_max
    beta_max = 1.0 / args.alpha_min
    
    cfg = build_config(
        rho=args.rho,
        sigma0_2=sigma0_2,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_points=args.alpha_points,
    )
    
    # Generate MC samples
    print("\nGenerating Monte Carlo samples...")
    rng = np.random.default_rng(args.seed)
    x_mc = sample_bg(args.mc_samples, rng, args.rho, cfg.var_nonzero)
    v_mc = rng.normal(size=args.mc_samples)
    
    alpha_range = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    
    # Compute replica curves for each pi_A
    print("\nComputing replica curves...")
    replica_curves = {}
    for pi_A in args.pi_A_values:
        pi_str = str(pi_A)
        cache_file = os.path.join(
            cache_dir,
            f"replica_mixture_pi_A={pi_A:.2f}_k_A={k_A:.6e}_k_B={k_B:.6e}.csv"
        )
        
        if args.skip_replica and os.path.exists(cache_file):
            print(f"\nLoading cached replica for pi_A = {pi_A:.2f}...")
            df_cache = pd.read_csv(cache_file)
            replica_curves[pi_str] = (df_cache["alpha"].values, df_cache["mse"].values)
        else:
            print(f"\nComputing replica for pi_A = {pi_A:.2f}...")
            # Use fresh RNG for each pi_A to ensure reproducibility
            rng_pi = np.random.default_rng(args.seed + int(pi_A * 1000))
            alpha_vals, mse_vals = compute_replica_curve_mixture(
                k_A, k_B, pi_A,
                args.ft_regulariser_scale,
                alpha_range, cfg, x_mc, v_mc, rng_pi
            )
            replica_curves[pi_str] = (alpha_vals, mse_vals)
            
            df_cache = pd.DataFrame({"alpha": alpha_vals, "mse": mse_vals})
            df_cache.to_csv(cache_file, index=False)
        
        alpha_vals, mse_vals = replica_curves[pi_str]
        print(f"  MSE range: [{mse_vals.min():.6e}, {mse_vals.max():.6e}]")
    
    # Load empirical results
    print("\nLoading empirical results...")
    if os.path.exists(args.empirical_csv):
        agg_df = aggregate_empirical_results(args.empirical_csv)
        print(f"  Loaded {len(agg_df)} aggregated rows")
        print(f"  pi_A values: {sorted(agg_df['pi_A'].unique())}")
    else:
        print(f"  WARNING: Empirical results not found at {args.empirical_csv}")
        print("  Run step1_mixture_sweep.sh first to generate empirical data")
        agg_df = pd.DataFrame()
    
    # Check acceptance criteria
    print("\n" + "="*80)
    print("ACCEPTANCE CRITERIA CHECK")
    print("="*80)
    
    if len(agg_df) > 0:
        acceptance_results = check_acceptance_criteria(agg_df, replica_curves)
        
        print(f"\n1. Ordering correct (larger pi_A -> more c_A behavior): "
              f"{'PASS' if acceptance_results['ordering_correct'] else 'FAIL'}")
        print(f"2. Smooth interpolation (low CV): "
              f"{'PASS' if acceptance_results['smooth_interpolation'] else 'FAIL'}")
        
        all_pass = acceptance_results['ordering_correct'] and acceptance_results['smooth_interpolation']
        
        print("\n" + "="*80)
        if all_pass:
            print("STEP 1 VALIDATION: ALL CRITERIA PASS ✓")
        else:
            print("STEP 1 VALIDATION: SOME CRITERIA FAIL ✗")
        print("="*80)
        
        # Create overlay plot
        print("\nCreating overlay plot...")
        plot_step1_overlay(agg_df, replica_curves, args.output_dir, acceptance_results)
        
        # Save results - convert numpy bools to Python bools for JSON serialization
        import json
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(x) for x in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            return obj
        
        results_path = os.path.join(args.output_dir, "acceptance_results.json")
        with open(results_path, "w") as f:
            json.dump(convert_to_serializable(acceptance_results), f, indent=2)
        
        return 0 if all_pass else 1
    else:
        print("\nNo empirical data available. Run experiments first.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


