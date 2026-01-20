#!/usr/bin/env python3
"""
Step 4 Validation: Compare learned PT vs oracle PT for FT performance.

This script:
1. Loads empirical results from step4_learned experiments
2. Compares learned init vs oracle init FT performance
3. Creates comparison plots
4. Checks acceptance criteria

Usage:
    python scripts/diagonal/plot_step4_validation.py --output_dir figures/validation_step4
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


def to_db(x: np.ndarray) -> np.ndarray:
    """Convert MSE to dB: 10*log10(mse + 1e-15)"""
    return 10.0 * np.log10(np.maximum(x, 1e-15))


def aggregate_results(csv_path: str) -> pd.DataFrame:
    """Load and aggregate empirical results."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Aggregate by (omega, alpha_ft)
    agg = df.groupby(['omega', 'alpha_ft']).agg({
        'pt_param_mse': ['mean', 'median', 'std'],
        'ft_param_mse_learned': ['mean', 'median', 'std'],
        'ft_param_mse_oracle': ['mean', 'median', 'std'],
        'c_ft_correlation': ['mean', 'std'],
        'c_ft_mean_diff': ['mean', 'std'],
        'seed': 'count',
    }).reset_index()
    
    # Flatten columns
    agg.columns = [
        'omega', 'alpha_ft',
        'pt_param_mse_mean', 'pt_param_mse_median', 'pt_param_mse_std',
        'ft_learned_mean', 'ft_learned_median', 'ft_learned_std',
        'ft_oracle_mean', 'ft_oracle_median', 'ft_oracle_std',
        'c_ft_corr_mean', 'c_ft_corr_std',
        'c_ft_diff_mean', 'c_ft_diff_std',
        'count',
    ]
    
    return agg


def check_acceptance_criteria(agg_df: pd.DataFrame) -> dict:
    """Check Step 4 acceptance criteria."""
    results = {
        "learned_close_to_oracle": False,
        "c_ft_correlation_high": False,
        "pt_converged": False,
    }
    
    # 1. Check if learned FT is close to oracle FT
    # Compute ratio of learned/oracle MSE
    agg_df['ratio'] = agg_df['ft_learned_mean'] / agg_df['ft_oracle_mean']
    mean_ratio = agg_df['ratio'].mean()
    max_ratio = agg_df['ratio'].max()
    
    # Learned should be within 2x of oracle on average
    results["learned_close_to_oracle"] = mean_ratio < 2.0 and max_ratio < 5.0
    results["mean_ratio"] = float(mean_ratio)
    results["max_ratio"] = float(max_ratio)
    
    # 2. Check c_ft correlation
    mean_corr = agg_df['c_ft_corr_mean'].mean()
    results["c_ft_correlation_high"] = mean_corr > 0.9
    results["mean_c_ft_correlation"] = float(mean_corr)
    
    # 3. Check PT converged well
    mean_pt_mse = agg_df['pt_param_mse_mean'].mean()
    results["pt_converged"] = mean_pt_mse < 0.01  # PT param MSE should be low
    results["mean_pt_param_mse"] = float(mean_pt_mse)
    
    return results


def plot_step4_comparison(
    agg_df: pd.DataFrame,
    output_dir: str,
    acceptance_results: dict,
):
    """Create comparison plots for Step 4."""
    omega_values = sorted(agg_df['omega'].unique())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Learned vs Oracle FT param MSE
    ax1 = axes[0]
    cmap = plt.cm.viridis
    colors = {omega: cmap(i / (len(omega_values) - 1)) for i, omega in enumerate(omega_values)}
    
    for omega in omega_values:
        color = colors[omega]
        mask = agg_df['omega'] == omega
        df_omega = agg_df[mask].sort_values('alpha_ft')
        
        if len(df_omega) > 0:
            # Learned (solid)
            ax1.plot(
                df_omega['alpha_ft'],
                to_db(df_omega['ft_learned_mean'].values),
                'o-',
                label=f'Learned ω={omega:.2f}',
                linewidth=2,
                markersize=5,
                color=color,
            )
            
            # Oracle (dashed)
            ax1.plot(
                df_omega['alpha_ft'],
                to_db(df_omega['ft_oracle_mean'].values),
                's--',
                label=f'Oracle ω={omega:.2f}',
                linewidth=2,
                markersize=4,
                color=color,
                alpha=0.7,
            )
    
    ax1.set_xlabel(r"$\alpha_{FT} = n_{FT} / d$", fontsize=12)
    ax1.set_ylabel("FT Parameter MSE (dB)", fontsize=12)
    ax1.set_title("Learned vs Oracle FT Performance", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best', ncol=2)
    
    # Plot 2: Ratio of learned/oracle
    ax2 = axes[1]
    
    for omega in omega_values:
        color = colors[omega]
        mask = agg_df['omega'] == omega
        df_omega = agg_df[mask].sort_values('alpha_ft')
        
        if len(df_omega) > 0:
            ratio = df_omega['ft_learned_mean'] / df_omega['ft_oracle_mean']
            ax2.plot(
                df_omega['alpha_ft'],
                ratio.values,
                'o-',
                label=f'ω={omega:.2f}',
                linewidth=2,
                markersize=5,
                color=color,
            )
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Ratio=1')
    ax2.axhline(y=2.0, color='red', linestyle=':', alpha=0.5, label='Ratio=2')
    ax2.set_xlabel(r"$\alpha_{FT} = n_{FT} / d$", fontsize=12)
    ax2.set_ylabel("MSE Ratio (Learned / Oracle)", fontsize=12)
    ax2.set_title("How Close is Learned to Oracle?", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')
    
    # Build status string
    status_parts = []
    if acceptance_results.get("learned_close_to_oracle"):
        status_parts.append(f"Close: ✓ (ratio={acceptance_results.get('mean_ratio', 0):.2f})")
    else:
        status_parts.append("Close: ✗")
    if acceptance_results.get("c_ft_correlation_high"):
        status_parts.append(f"Corr: ✓ ({acceptance_results.get('mean_c_ft_correlation', 0):.3f})")
    else:
        status_parts.append("Corr: ✗")
    status_str = ", ".join(status_parts)
    
    fig.suptitle(
        f"Step 4 Validation: Learned PT vs Oracle | {status_str}",
        fontsize=13
    )
    
    fig.tight_layout()
    
    png_path = os.path.join(output_dir, "step4_validation_comparison.png")
    pdf_path = os.path.join(output_dir, "step4_validation_comparison.pdf")
    
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nComparison plot saved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 4 Validation: Learned PT")
    parser.add_argument("--empirical_csv", type=str, 
                        default="experiment_results_step4_learned.csv")
    parser.add_argument("--output_dir", type=str, 
                        default="figures/validation_step4")
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("STEP 4 VALIDATION: LEARNED PT VS ORACLE")
    print("="*80)
    
    # Load empirical results
    print("\nLoading empirical results...")
    if os.path.exists(args.empirical_csv):
        agg_df = aggregate_results(args.empirical_csv)
        print(f"  Loaded {len(agg_df)} aggregated rows")
        print(f"  Omega values: {sorted(agg_df['omega'].unique())}")
        print(f"  Alpha FT values: {sorted(agg_df['alpha_ft'].unique())}")
    else:
        print(f"  WARNING: Empirical results not found at {args.empirical_csv}")
        print("  Run step4_learned_sweep.sh first to generate empirical data")
        return 1
    
    # Check acceptance criteria
    print("\n" + "="*80)
    print("ACCEPTANCE CRITERIA CHECK")
    print("="*80)
    
    acceptance_results = check_acceptance_criteria(agg_df)
    
    print(f"\n1. Learned close to oracle (mean ratio < 2): "
          f"{'PASS' if acceptance_results['learned_close_to_oracle'] else 'FAIL'}"
          f" (mean={acceptance_results.get('mean_ratio', 'N/A'):.3f}, "
          f"max={acceptance_results.get('max_ratio', 'N/A'):.3f})")
    print(f"2. c_ft correlation high (> 0.9): "
          f"{'PASS' if acceptance_results['c_ft_correlation_high'] else 'FAIL'}"
          f" (mean={acceptance_results.get('mean_c_ft_correlation', 'N/A'):.4f})")
    print(f"3. PT converged (param MSE < 0.01): "
          f"{'PASS' if acceptance_results['pt_converged'] else 'FAIL'}"
          f" (mean={acceptance_results.get('mean_pt_param_mse', 'N/A'):.6e})")
    
    all_pass = (acceptance_results['learned_close_to_oracle'] and 
                acceptance_results['c_ft_correlation_high'] and
                acceptance_results['pt_converged'])
    
    print("\n" + "="*80)
    if all_pass:
        print("STEP 4 VALIDATION: ALL CRITERIA PASS ✓")
        print("\nThe Cosyne mapping successfully predicts FT performance from learned PT!")
    else:
        print("STEP 4 VALIDATION: SOME CRITERIA FAIL ✗")
        print("\nDiagnose: check PT convergence, c_ft mapping accuracy, or increase PT alpha.")
    print("="*80)
    
    # Create comparison plot
    print("\nCreating comparison plot...")
    plot_step4_comparison(agg_df, args.output_dir, acceptance_results)
    
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
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Omega':<8} {'Alpha_FT':<10} {'FT_Learned':<12} {'FT_Oracle':<12} {'Ratio':<8} {'c_ft_corr':<10}")
    print("-"*80)
    for _, row in agg_df.sort_values(['omega', 'alpha_ft']).iterrows():
        ratio = row['ft_learned_mean'] / row['ft_oracle_mean'] if row['ft_oracle_mean'] > 0 else float('inf')
        print(f"{row['omega']:<8.2f} {row['alpha_ft']:<10.4f} "
              f"{row['ft_learned_mean']:<12.6e} {row['ft_oracle_mean']:<12.6e} "
              f"{ratio:<8.3f} {row['c_ft_corr_mean']:<10.4f}")
    print("="*80)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())


