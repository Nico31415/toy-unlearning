#!/usr/bin/env python3
"""
Plot lr=0.5 empirical results with replica theory overlay, 
highlighting converged vs non-converged runs.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def to_db(mse_vals):
    """Convert MSE to dB: 10*log10(mse + 1e-15)"""
    return 10.0 * np.log10(np.maximum(mse_vals, 1e-15))

def load_replica_curve(cache_dir, c, rho=0.04, ft_regulariser_scale=1e-6, 
                       alpha_min=0.008, alpha_max=1.0, alpha_points=100,
                       mc_samples=50000, seed=12345):
    """Load replica curve from cache."""
    cache_filename = (
        f"replica_curve_teacher=bg--rho={rho:.6f}--c={c:.6f}--"
        f"ft_reg={ft_regulariser_scale:.6e}--alpha_min={alpha_min:.4f}--alpha_max={alpha_max:.4f}--"
        f"alpha_points={alpha_points}--mc_samples={mc_samples}--seed={seed}.csv"
    )
    cache_path = os.path.join(cache_dir, cache_filename)
    
    if not os.path.exists(cache_path):
        print(f"WARNING: Replica curve cache not found at {cache_path}")
        return None, None
    
    df = pd.read_csv(cache_path)
    return df["alpha"].values, df["mse"].values

def main():
    # Parameters
    csv_path = "experiment_results_bg_alpha_sweep.csv"
    output_dir = "figures/diagonal/bg_generalization"
    c = 0.001
    rho = 0.04
    ft_regulariser_scale = 1e-6
    lr_filter = 0.5
    
    # Load results
    print(f"Loading results from {csv_path}...")
    df_all = pd.read_csv(csv_path)
    df = df_all[df_all['save_folder'].str.contains(f'lr={lr_filter}')].copy()
    df_valid = df[df['param_mse'].notna()].copy()
    
    print(f"  Found {len(df_valid)} valid experiments with lr={lr_filter}")
    
    # Separate converged and non-converged
    df_converged = df_valid[df_valid['stop_reason'] == 'threshold'].copy()
    df_notconv = df_valid[df_valid['stop_reason'] != 'threshold'].copy()
    
    print(f"  Converged: {len(df_converged)}")
    print(f"  Not converged: {len(df_notconv)}")
    
    # Aggregate both groups
    def agg_func(df_sub):
        if len(df_sub) == 0:
            return pd.DataFrame()
        agg = df_sub.groupby('alpha').agg({
            'param_mse': ['mean', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75), 'count'],
        }).reset_index()
        agg.columns = ['alpha', 'param_mse_mean', 'param_mse_median', 'param_mse_q25', 'param_mse_q75', 'count']
        agg = agg.sort_values('alpha')
        agg['param_mse_mean_db'] = to_db(agg['param_mse_mean'].values)
        agg['param_mse_median_db'] = to_db(agg['param_mse_median'].values)
        return agg
    
    agg_conv = agg_func(df_converged)
    agg_notconv = agg_func(df_notconv)
    
    # Load replica curve
    print(f"\nLoading replica theory curve for c={c}...")
    cache_dir = os.path.join(output_dir, "replica_cache")
    replica_alpha, replica_mse = load_replica_curve(cache_dir, c, rho, ft_regulariser_scale)
    
    if replica_alpha is not None:
        replica_mse_db = to_db(replica_mse)
        print(f"  Loaded {len(replica_alpha)} points")
    else:
        replica_mse_db = None
    
    # Create plot with two subplots
    print("\nCreating plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # ============= TOP PLOT: All data =============
    ax = ax1
    
    # Plot converged runs
    if len(agg_conv) > 0:
        valid_mask = np.isfinite(agg_conv['param_mse_mean_db'])
        agg_conv_valid = agg_conv[valid_mask].copy()
        
        ax.plot(
            agg_conv_valid['alpha'],
            agg_conv_valid['param_mse_mean_db'],
            'o-',
            label='Converged (mean)',
            linewidth=2,
            markersize=8,
            color='blue',
            markeredgecolor='darkblue',
            markeredgewidth=1.5
        )
    
    # Plot non-converged runs
    if len(agg_notconv) > 0:
        valid_mask = np.isfinite(agg_notconv['param_mse_mean_db'])
        agg_notconv_valid = agg_notconv[valid_mask].copy()
        
        ax.plot(
            agg_notconv_valid['alpha'],
            agg_notconv_valid['param_mse_mean_db'],
            'x--',
            label='Not converged (mean)',
            linewidth=2,
            markersize=10,
            color='red',
            markeredgewidth=2
        )
    
    # Plot replica theory
    if replica_alpha is not None and replica_mse_db is not None:
        ax.plot(
            replica_alpha,
            replica_mse_db,
            '-',
            label=f'Replica theory (c={c:.3f})',
            linewidth=2.5,
            color='orange',
            alpha=0.9
        )
    
    ax.set_xlabel(r'$\alpha = n_{\mathrm{train}} / d$', fontsize=14)
    ax.set_ylabel('Parameter MSE (dB)', fontsize=14)
    ax.set_title(
        f'Empirical vs Replica Theory: Convergence Analysis (lr={lr_filter})\n'
        f'Bernoulli-Gaussian, $\\rho={rho:.3f}$, $c={c:.3f}$',
        fontsize=14
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    # ============= BOTTOM PLOT: Ratio analysis =============
    ax = ax2
    
    # Calculate ratio for converged runs
    if len(agg_conv) > 0 and replica_alpha is not None:
        ratios_conv = []
        alphas_conv = []
        for _, row in agg_conv.iterrows():
            alpha = row['alpha']
            emp_mse = row['param_mse_mean']
            # Find closest replica alpha
            idx = np.abs(replica_alpha - alpha).argmin()
            rep_mse = replica_mse[idx]
            ratio = emp_mse / rep_mse
            ratios_conv.append(ratio)
            alphas_conv.append(alpha)
        
        ax.plot(
            alphas_conv,
            ratios_conv,
            'o-',
            label='Converged runs',
            linewidth=2,
            markersize=8,
            color='blue',
            markeredgecolor='darkblue',
            markeredgewidth=1.5
        )
    
    # Calculate ratio for non-converged runs
    if len(agg_notconv) > 0 and replica_alpha is not None:
        ratios_notconv = []
        alphas_notconv = []
        for _, row in agg_notconv.iterrows():
            alpha = row['alpha']
            emp_mse = row['param_mse_mean']
            # Find closest replica alpha
            idx = np.abs(replica_alpha - alpha).argmin()
            rep_mse = replica_mse[idx]
            ratio = emp_mse / rep_mse
            ratios_notconv.append(ratio)
            alphas_notconv.append(alpha)
        
        ax.plot(
            alphas_notconv,
            ratios_notconv,
            'x--',
            label='Not converged runs',
            linewidth=2,
            markersize=10,
            color='red',
            markeredgewidth=2
        )
    
    # Reference line at ratio = 1
    ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Perfect match')
    
    ax.set_xlabel(r'$\alpha = n_{\mathrm{train}} / d$', fontsize=14)
    ax.set_ylabel('Empirical MSE / Replica MSE', fontsize=14)
    ax.set_title('Ratio of Empirical to Theoretical MSE', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.set_yscale('log')
    
    fig.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f'empirical_lr{lr_filter}_vs_replica_detailed.png')
    pdf_path = os.path.join(output_dir, f'empirical_lr{lr_filter}_vs_replica_detailed.pdf')
    
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nPlot saved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    if len(agg_conv) > 0 and replica_alpha is not None:
        print("\nCONVERGED RUNS:")
        print("Alpha | Empirical MSE | Replica MSE | Ratio")
        print("-"*60)
        for i, row in agg_conv.iterrows():
            alpha = row['alpha']
            emp_mse = row['param_mse_mean']
            idx = np.abs(replica_alpha - alpha).argmin()
            rep_mse = replica_mse[idx]
            ratio = emp_mse / rep_mse
            print(f"{alpha:.2f}  | {emp_mse:.6e}  | {rep_mse:.6e}  | {ratio:.2f}x")
    
    if len(agg_notconv) > 0 and replica_alpha is not None:
        print("\nNOT CONVERGED RUNS (hit max_epochs):")
        print("Alpha | Empirical MSE | Replica MSE | Ratio")
        print("-"*60)
        for i, row in agg_notconv.iterrows():
            alpha = row['alpha']
            emp_mse = row['param_mse_mean']
            idx = np.abs(replica_alpha - alpha).argmin()
            rep_mse = replica_mse[idx]
            ratio = emp_mse / rep_mse
            print(f"{alpha:.2f}  | {emp_mse:.6e}  | {rep_mse:.6e}  | {ratio:.2f}x")

if __name__ == '__main__':
    main()

