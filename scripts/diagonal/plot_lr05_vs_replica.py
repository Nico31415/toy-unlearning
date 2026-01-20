#!/usr/bin/env python3
"""
Plot lr=0.5 empirical results with replica theory overlay for c=0.001.
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
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache filename (match existing format)
    cache_filename = (
        f"replica_curve_teacher=bg--rho={rho:.6f}--c={c:.6f}--"
        f"ft_reg={ft_regulariser_scale:.6e}--alpha_min={alpha_min:.4f}--alpha_max={alpha_max:.4f}--"
        f"alpha_points={alpha_points}--mc_samples={mc_samples}--seed={seed}.csv"
    )
    cache_path = os.path.join(cache_dir, cache_filename)
    
    if not os.path.exists(cache_path):
        print(f"WARNING: Replica curve cache not found at {cache_path}")
        print(f"  You may need to run: python scripts/diagonal/plot_replica_q_bg.py --c_values {c}")
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
    lr_filter = 0.5  # Filter for lr=0.5
    
    # Load results
    print(f"Loading results from {csv_path}...")
    df_all = pd.read_csv(csv_path)
    print(f"  Loaded {len(df_all)} experiments")
    
    # Filter for lr=0.5 experiments
    print(f"\nFiltering for lr={lr_filter} experiments...")
    df = df_all[df_all['save_folder'].str.contains(f'lr={lr_filter}')].copy()
    print(f"  Found {len(df)} experiments with lr={lr_filter}")
    print(f"  Alpha range: [{df['alpha'].min():.2f}, {df['alpha'].max():.2f}]")
    
    if len(df) == 0:
        print("ERROR: No experiments found with lr=0.5")
        return
    
    # Remove rows with NaN param_mse
    df_valid_for_agg = df[df['param_mse'].notna()].copy()
    print(f"  Valid experiments (non-NaN param_mse): {len(df_valid_for_agg)}")
    
    # Aggregate by alpha
    print("\nAggregating results by alpha...")
    agg = df_valid_for_agg.groupby('alpha').agg({
        'param_mse': ['mean', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75), 'count'],
        'train_pred_mse': 'mean',
    }).reset_index()
    
    agg.columns = ['alpha', 'param_mse_mean', 'param_mse_median', 'param_mse_q25', 'param_mse_q75', 'count', 'train_pred_mse_mean']
    agg = agg.sort_values('alpha')
    
    # Convert to dB
    agg['param_mse_mean_db'] = to_db(agg['param_mse_mean'].values)
    agg['param_mse_median_db'] = to_db(agg['param_mse_median'].values)
    agg['param_mse_q25_db'] = to_db(agg['param_mse_q25'].values)
    agg['param_mse_q75_db'] = to_db(agg['param_mse_q75'].values)
    
    print(f"  Aggregated to {len(agg)} alpha values")
    
    # Load replica curve
    print(f"\nLoading replica theory curve for c={c}...")
    cache_dir = os.path.join(output_dir, "replica_cache")
    replica_alpha, replica_mse = load_replica_curve(cache_dir, c, rho, ft_regulariser_scale)
    
    if replica_alpha is not None:
        replica_mse_db = to_db(replica_mse)
        print(f"  Loaded {len(replica_alpha)} points")
        print(f"  Alpha range: [{replica_alpha.min():.3f}, {replica_alpha.max():.3f}]")
        print(f"  MSE range: [{replica_mse.min():.6e}, {replica_mse.max():.6e}]")
    else:
        replica_alpha = None
        replica_mse_db = None
    
    # Create plot
    print("\nCreating plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot empirical results
    valid_mask = (
        np.isfinite(agg['param_mse_mean_db']) &
        np.isfinite(agg['param_mse_median_db'])
    )
    agg_valid = agg[valid_mask].copy()
    
    if len(agg_valid) > 0:
        # Mean curve
        ax.plot(
            agg_valid['alpha'],
            agg_valid['param_mse_mean_db'],
            'o-',
            label='Empirical mean (lr=0.5)',
            linewidth=2,
            markersize=6,
            color='blue'
        )
        
        # Median curve
        ax.plot(
            agg_valid['alpha'],
            agg_valid['param_mse_median_db'],
            's--',
            label='Empirical median (lr=0.5)',
            linewidth=2,
            markersize=5,
            color='red',
            alpha=0.7
        )
        
        # IQR fill
        q_valid_mask = (
            np.isfinite(agg_valid['param_mse_q25_db']) &
            np.isfinite(agg_valid['param_mse_q75_db'])
        )
        if q_valid_mask.sum() > 0:
            ax.fill_between(
                agg_valid.loc[q_valid_mask, 'alpha'],
                agg_valid.loc[q_valid_mask, 'param_mse_q25_db'],
                agg_valid.loc[q_valid_mask, 'param_mse_q75_db'],
                alpha=0.2,
                color='blue',
                label='IQR (25-75%)'
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
    
    # Labels and title
    ax.set_xlabel(r'$\alpha = n_{\mathrm{train}} / d$', fontsize=14)
    ax.set_ylabel('Parameter MSE (dB)', fontsize=14)
    
    n_success = (df['train_pred_mse'] < 1e-10).sum() if 'train_pred_mse' in df.columns else 0
    ax.set_title(
        f'Empirical vs Replica Theory (lr={lr_filter}, c={c:.3f})\n'
        f'Bernoulli-Gaussian, $\\rho={rho:.3f}$, successes = {n_success}/{len(df)}',
        fontsize=14
    )
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    fig.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f'empirical_lr{lr_filter}_vs_replica_c{c}.png')
    pdf_path = os.path.join(output_dir, f'empirical_lr{lr_filter}_vs_replica_c{c}.pdf')
    
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nPlot saved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total experiments (lr={lr_filter}): {len(df)}")
    print(f"Alpha values: {sorted(df['alpha'].unique())}")
    print(f"Success rate (train_pred_mse < 1e-10): {n_success}/{len(df)} ({100*n_success/len(df):.1f}%)")
    print("\nAggregated results by alpha:")
    print(agg[['alpha', 'count', 'param_mse_mean', 'param_mse_median']].to_string(index=False))

if __name__ == '__main__':
    main()

