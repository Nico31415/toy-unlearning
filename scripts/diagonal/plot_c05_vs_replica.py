#!/usr/bin/env python3
"""
Compare c=0.5 empirical results with replica theory for c=0.5.
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
    csv_path = "experiment_results_bg_alpha_sweep_c05.csv"
    output_dir = "figures/diagonal/bg_generalization"
    c = 0.5
    rho = 0.04
    ft_regulariser_scale = 1e-6
    
    # Load c=0.5 results
    print(f"Loading c=0.5 results from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} experiments")
    print(f"  Alpha range: [{df['alpha'].min():.2f}, {df['alpha'].max():.2f}]")
    
    # Aggregate by alpha
    print("\nAggregating results by alpha...")
    agg = df.groupby('alpha').agg({
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
    else:
        replica_mse_db = None
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # ============= TOP PLOT: MSE curves =============
    ax = ax1
    
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
            label='Empirical mean (c=0.5)',
            linewidth=2,
            markersize=6,
            color='green'
        )
        
        # Median curve
        ax.plot(
            agg_valid['alpha'],
            agg_valid['param_mse_median_db'],
            's--',
            label='Empirical median (c=0.5)',
            linewidth=2,
            markersize=5,
            color='darkgreen',
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
                color='green',
                label='IQR (25-75%)'
            )
    
    # Plot replica theory
    if replica_alpha is not None and replica_mse_db is not None:
        ax.plot(
            replica_alpha,
            replica_mse_db,
            '-',
            label=f'Replica theory (c={c:.1f})',
            linewidth=2.5,
            color='orange',
            alpha=0.9
        )
    
    ax.set_xlabel(r'$\alpha = n_{\mathrm{train}} / d$', fontsize=13)
    ax.set_ylabel('Parameter MSE (dB)', fontsize=13)
    ax.set_title(
        f'c={c:.1f} Empirical vs Replica Theory\n'
        f'Bernoulli-Gaussian, $\\rho={rho:.3f}$',
        fontsize=14
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    # ============= BOTTOM PLOT: Ratio analysis =============
    ax = ax2
    
    if replica_alpha is not None:
        ratios = []
        alphas = []
        for _, row in agg.iterrows():
            alpha = row['alpha']
            emp_mse = row['param_mse_mean']
            # Find closest replica alpha
            idx = np.abs(replica_alpha - alpha).argmin()
            rep_mse = replica_mse[idx]
            ratio = emp_mse / rep_mse
            ratios.append(ratio)
            alphas.append(alpha)
        
        ax.plot(
            alphas,
            ratios,
            'o-',
            label='c=0.5',
            linewidth=2,
            markersize=8,
            color='green',
            markeredgecolor='darkgreen',
            markeredgewidth=1.5
        )
    
    # Reference line at ratio = 1
    ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Perfect match')
    
    ax.set_xlabel(r'$\alpha = n_{\mathrm{train}} / d$', fontsize=13)
    ax.set_ylabel('Empirical MSE / Replica MSE', fontsize=13)
    ax.set_title('Ratio of Empirical to Theoretical MSE (c=0.5)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.set_yscale('log')
    
    fig.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f'empirical_c{c}_vs_replica.png')
    pdf_path = os.path.join(output_dir, f'empirical_c{c}_vs_replica.pdf')
    
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nPlot saved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    
    # Print comparison
    print("\n" + "="*80)
    print("EMPIRICAL vs REPLICA COMPARISON (c=0.5)")
    print("="*80)
    
    if replica_alpha is not None:
        print("\nAlpha | Empirical MSE | Replica MSE | Ratio")
        print("-"*60)
        for _, row in agg.iterrows():
            alpha = row['alpha']
            emp_mse = row['param_mse_mean']
            idx = np.abs(replica_alpha - alpha).argmin()
            rep_mse = replica_mse[idx]
            ratio = emp_mse / rep_mse
            print(f"{alpha:.2f}  | {emp_mse:.6e}  | {rep_mse:.6e}  | {ratio:.2f}x")

if __name__ == '__main__':
    main()

