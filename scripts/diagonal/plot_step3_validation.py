#!/usr/bin/env python3
"""
Step 3 Validation: Overlay empirical PT+FT Oracle vs replica curves.

This script:
1. Aggregates empirical results from step3_omega experiments
2. Generates replica curves with matching ptft_oracle parameters
3. Creates overlay plots for validation
4. Checks acceptance criteria

Usage:
    python scripts/diagonal/plot_step3_validation.py --output_dir figures/validation_step3
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
    PTFTOracleConfig,
    solve_rspmap_qk_curve_best_of_forward_backward,
    sample_ptft_oracle_mc,
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


def compute_replica_curve_ptft_oracle(
    omega: float,
    rho_pt: float,
    rho_ft: float,
    a_pt: float,
    c_pt: float,
    lambda_pt: float,
    gamma_reinit: float,
    ft_regulariser_scale: float,
    alpha_range: np.ndarray,
    cfg: Config,
    mc_samples: int,
    rng: np.random.Generator,
) -> tuple:
    """Compute replica theory curve for ptft_oracle mode."""
    # Create ptft oracle config
    ptft_cfg = PTFTOracleConfig(
        rho_pt=rho_pt,
        rho_ft=rho_ft,
        omega=omega,
        a_pt=a_pt,
        c_pt=c_pt,
        lambda_pt=lambda_pt,
        gamma_reinit=gamma_reinit,
    )
    
    # Sample MC
    beta_ft_mc, beta_pt_mc, k_mc, g_mc = sample_ptft_oracle_mc(ptft_cfg, rng, mc_samples)
    x_mc = beta_ft_mc
    v_mc = rng.normal(size=mc_samples)
    
    # Gamma_ext for ptft_oracle = ft_regulariser_scale directly
    gamma_ext = float(ft_regulariser_scale)
    
    print(f"  [ptft_oracle] omega={omega:.2f}, gamma_ext={gamma_ext:.6e}")
    
    # Convert alpha to beta
    alpha_reversed = alpha_range[::-1]
    beta_range = 1.0 / alpha_reversed
    
    # Use hetero-k path
    k_q = 1.0  # Placeholder
    mse_beta = solve_rspmap_qk_curve_best_of_forward_backward(
        beta_range, gamma_ext, k_q, x_mc, v_mc, cfg, k_mc=k_mc, g_mc=g_mc
    )
    
    mse_alpha = mse_beta[::-1]
    return alpha_range, mse_alpha


def aggregate_empirical_results(csv_path: str) -> pd.DataFrame:
    """Load and aggregate empirical results by (omega, alpha)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Aggregate by (omega, alpha)
    agg = df.groupby(['omega', 'alpha']).agg({
        'param_mse': ['mean', 'median', 'std', 'count',
                      lambda x: np.percentile(x, 25),
                      lambda x: np.percentile(x, 75)],
        'test_pred_mse': ['mean', 'median'],
        'train_pred_mse': ['mean', 'median'],
    }).reset_index()
    
    # Flatten columns
    agg.columns = [
        'omega', 'alpha',
        'param_mse_mean', 'param_mse_median', 'param_mse_std', 'count',
        'param_mse_q25', 'param_mse_q75',
        'test_pred_mse_mean', 'test_pred_mse_median',
        'train_pred_mse_mean', 'train_pred_mse_median',
    ]
    
    return agg


def check_acceptance_criteria(agg_df: pd.DataFrame, replica_curves: dict) -> dict:
    """Check Step 3 acceptance criteria."""
    results = {
        "monotonic_omega": False,
        "insensitive_to_c_values": True,  # Assumed true (oracle doesn't use c_values)
        "qualitative_match": False,
    }
    
    omega_values = sorted(agg_df['omega'].unique())
    
    if len(omega_values) < 2:
        print("WARNING: Need at least 2 omega values to check monotonicity")
        return results
    
    # Check monotonicity: higher omega should improve early-alpha performance
    monotonic_checks = []
    for alpha in [0.1, 0.15, 0.2, 0.25]:
        mses_at_alpha = []
        for omega in omega_values:
            mask = (agg_df['omega'] == omega) & (np.abs(agg_df['alpha'] - alpha) < 0.01)
            if mask.sum() > 0:
                mse = agg_df.loc[mask, 'param_mse_mean'].values[0]
                mses_at_alpha.append((omega, mse))
        
        if len(mses_at_alpha) >= 2:
            # MSE should decrease as omega increases
            sorted_by_omega = sorted(mses_at_alpha, key=lambda x: x[0])
            mse_decreasing = all(
                sorted_by_omega[i][1] >= sorted_by_omega[i+1][1]
                for i in range(len(sorted_by_omega)-1)
            )
            monotonic_checks.append(mse_decreasing)
    
    results["monotonic_omega"] = all(monotonic_checks) if monotonic_checks else False
    
    # Check qualitative match: compare curves shapes
    # Use correlation between empirical and replica at each omega
    correlations = []
    for omega in omega_values:
        omega_str = str(omega)
        if omega_str not in replica_curves:
            continue
        
        alpha_rep, mse_rep = replica_curves[omega_str]
        
        mask = agg_df['omega'] == omega
        df_omega = agg_df[mask].sort_values('alpha')
        
        if len(df_omega) < 3:
            continue
        
        emp_alpha = df_omega['alpha'].values
        emp_mse = df_omega['param_mse_mean'].values
        
        # Interpolate replica to empirical alpha values
        valid = np.isfinite(emp_mse) & (emp_mse > 0)
        if valid.sum() < 3:
            continue
        
        emp_log = np.log10(emp_mse[valid])
        rep_interp = np.interp(emp_alpha[valid], alpha_rep, mse_rep)
        rep_log = np.log10(rep_interp)
        
        corr = np.corrcoef(emp_log, rep_log)[0, 1]
        if np.isfinite(corr):
            correlations.append(corr)
    
    if correlations:
        mean_corr = np.mean(correlations)
        results["qualitative_match"] = mean_corr > 0.85
        results["mean_correlation"] = float(mean_corr)
    
    return results


def plot_step3_overlay(
    agg_df: pd.DataFrame,
    replica_curves: dict,
    output_dir: str,
    acceptance_results: dict,
    rho_pt: float | None = None,
    rho_ft: float | None = None,
    a_pt: float | None = None,
    c_pt: float | None = None,
    lambda_pt: float | None = None,
    gamma_reinit: float | None = None,
    ft_regulariser_scale: float | None = None,
):
    """Create overlay plot for Step 3 validation."""
    omega_values = sorted(agg_df['omega'].unique())
    
    # Color map for omega values
    cmap = plt.cm.viridis
    denom = max(1, (len(omega_values) - 1))
    colors = {omega: cmap(i / denom) for i, omega in enumerate(omega_values)}
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for omega in omega_values:
        color = colors[omega]
        
        # Empirical data
        mask = agg_df['omega'] == omega
        df_omega = agg_df[mask].sort_values('alpha')
        
        if len(df_omega) > 0:
            ax.plot(
                df_omega['alpha'],
                to_db(df_omega['param_mse_mean'].values),
                'o-',
                label=f'Emp ω={omega:.2f}',
                linewidth=2,
                markersize=4,
                color=color,
            )
        
        # Replica curve
        omega_str = str(omega)
        if omega_str in replica_curves:
            alpha_rep, mse_rep = replica_curves[omega_str]
            ax.plot(
                alpha_rep,
                to_db(mse_rep),
                '--',
                label=f'Rep ω={omega:.2f}',
                linewidth=2,
                color=color,
                alpha=0.7,
            )
    
    # Status string
    status_parts = []
    if acceptance_results.get("monotonic_omega"):
        status_parts.append("Monotonic: ✓")
    else:
        status_parts.append("Monotonic: ✗")
    if acceptance_results.get("qualitative_match"):
        corr = acceptance_results.get("mean_correlation", 0)
        status_parts.append(f"Match: ✓ (r={corr:.3f})")
    else:
        status_parts.append("Match: ✗")
    status_str = ", ".join(status_parts)
    
    ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=12)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=12)
    # Parameter line: keep old defaults if not provided (backwards compatible with validation usage)
    rho_pt_disp = 0.10 if rho_pt is None else float(rho_pt)
    rho_ft_disp = 0.04 if rho_ft is None else float(rho_ft)
    a_pt_disp = 1.0 if a_pt is None else float(a_pt)
    # These were previously omitted from the title; include when provided.
    param_parts = [
        f"ρ_pt={rho_pt_disp:.3g}",
        f"ρ_ft={rho_ft_disp:.3g}",
        f"a_pt={a_pt_disp:.3g}",
    ]
    if c_pt is not None:
        param_parts.append(f"c_pt={float(c_pt):.3g}")
    if lambda_pt is not None:
        param_parts.append(f"λ_pt={float(lambda_pt):.3g}")
    if gamma_reinit is not None:
        param_parts.append(f"γ={float(gamma_reinit):.3g}")
    if ft_regulariser_scale is not None:
        param_parts.append(f"ft_reg={float(ft_regulariser_scale):.1e}")
    param_line = ", ".join(param_parts)

    ax.set_title(
        f"PT+FT Oracle (Omega Sweep)\n"
        f"{param_line} | {status_str}",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best', ncol=2)
    
    fig.tight_layout()
    
    png_path = os.path.join(output_dir, "step3_validation_overlay.png")
    pdf_path = os.path.join(output_dir, "step3_validation_overlay.pdf")
    
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nOverlay plot saved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 3 Validation: PT+FT Oracle")
    parser.add_argument("--rho_pt", type=float, default=0.10)
    parser.add_argument("--rho_ft", type=float, default=0.04)
    parser.add_argument("--a_pt", type=float, default=1.0)
    parser.add_argument("--c_pt", type=float, default=0.001)
    parser.add_argument("--lambda_pt", type=float, default=0.0)
    parser.add_argument("--gamma_reinit", type=float, default=0.0)
    parser.add_argument("--omega_values", type=float, nargs="+", 
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--ft_regulariser_scale", type=float, default=1e-6)
    parser.add_argument("--alpha_min", type=float, default=0.008)
    parser.add_argument("--alpha_max", type=float, default=1.0)
    parser.add_argument("--alpha_points", type=int, default=100)
    parser.add_argument("--mc_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--empirical_csv", type=str, 
                        default="experiment_results_step3_omega.csv")
    parser.add_argument("--output_dir", type=str, 
                        default="figures/validation_step3")
    parser.add_argument("--skip_replica", action="store_true")
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "replica_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("STEP 3 VALIDATION: PT+FT ORACLE")
    print("="*80)
    print(f"rho_pt = {args.rho_pt}, rho_ft = {args.rho_ft}")
    print(f"a_pt = {args.a_pt}, c_pt = {args.c_pt}")
    print(f"lambda_pt = {args.lambda_pt}, gamma_reinit = {args.gamma_reinit}")
    print(f"omega values = {args.omega_values}")
    print("="*80)
    
    # Build config (use rho_ft for the solver)
    sigma0_2 = 0.0
    beta_min = 1.0 / args.alpha_max
    beta_max = 1.0 / args.alpha_min
    
    cfg = build_config(
        rho=args.rho_ft,  # FT sparsity for the solver
        sigma0_2=sigma0_2,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_points=args.alpha_points,
    )
    
    alpha_range = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    
    # Compute replica curves for each omega
    print("\nComputing replica curves...")
    replica_curves = {}
    for omega in args.omega_values:
        omega_str = str(omega)
        cache_file = os.path.join(
            cache_dir,
            f"replica_ptft_oracle_omega={omega:.2f}.csv"
        )
        
        if args.skip_replica and os.path.exists(cache_file):
            print(f"\nLoading cached replica for omega = {omega:.2f}...")
            df_cache = pd.read_csv(cache_file)
            replica_curves[omega_str] = (df_cache["alpha"].values, df_cache["mse"].values)
        else:
            print(f"\nComputing replica for omega = {omega:.2f}...")
            rng = np.random.default_rng(args.seed + int(omega * 1000))
            alpha_vals, mse_vals = compute_replica_curve_ptft_oracle(
                omega,
                args.rho_pt, args.rho_ft,
                args.a_pt, args.c_pt, args.lambda_pt, args.gamma_reinit,
                args.ft_regulariser_scale,
                alpha_range, cfg,
                args.mc_samples, rng
            )
            replica_curves[omega_str] = (alpha_vals, mse_vals)
            
            df_cache = pd.DataFrame({"alpha": alpha_vals, "mse": mse_vals})
            df_cache.to_csv(cache_file, index=False)
        
        alpha_vals, mse_vals = replica_curves[omega_str]
        print(f"  MSE range: [{mse_vals.min():.6e}, {mse_vals.max():.6e}]")
    
    # Load empirical results
    print("\nLoading empirical results...")
    if os.path.exists(args.empirical_csv):
        agg_df = aggregate_empirical_results(args.empirical_csv)
        print(f"  Loaded {len(agg_df)} aggregated rows")
        print(f"  Omega values: {sorted(agg_df['omega'].unique())}")
    else:
        print(f"  WARNING: Empirical results not found at {args.empirical_csv}")
        print("  Run step3_omega_sweep.sh first to generate empirical data")
        agg_df = pd.DataFrame()
    
    # Check acceptance criteria
    print("\n" + "="*80)
    print("ACCEPTANCE CRITERIA CHECK")
    print("="*80)
    
    if len(agg_df) > 0:
        acceptance_results = check_acceptance_criteria(agg_df, replica_curves)
        
        print(f"\n1. Omega sweep monotonic (higher omega -> lower MSE at low alpha): "
              f"{'PASS' if acceptance_results['monotonic_omega'] else 'FAIL'}")
        print(f"2. Qualitative match with replica: "
              f"{'PASS' if acceptance_results['qualitative_match'] else 'FAIL'}"
              f" (mean corr = {acceptance_results.get('mean_correlation', 'N/A'):.3f})" 
              if 'mean_correlation' in acceptance_results else "")
        
        all_pass = acceptance_results['monotonic_omega'] and acceptance_results['qualitative_match']
        
        print("\n" + "="*80)
        if all_pass:
            print("STEP 3 VALIDATION: ALL CRITERIA PASS ✓")
        else:
            print("STEP 3 VALIDATION: SOME CRITERIA FAIL ✗")
        print("="*80)
        
        # Create overlay plot
        print("\nCreating overlay plot...")
        plot_step3_overlay(agg_df, replica_curves, args.output_dir, acceptance_results)
        
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


