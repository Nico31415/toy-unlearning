#!/usr/bin/env python3
"""
Compare lr=0.5 and lr=5.0 empirical results with replica theory overlay.
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

def process_lr_data(df_all, lr_filter):
    """Process data for a specific learning rate."""
    df = df_all[df_all['save_folder'].str.contains(f'lr={lr_filter}')].copy()
    df_valid = df[df['param_mse'].notna()].copy()
    
    # Separate converged and non-converged
    df_converged = df_valid[df_valid['stop_reason'] == 'threshold'].copy()
    df_notconv = df_valid[df_valid['stop_reason'] != 'threshold'].copy()
    
    # Aggregate
    def agg_func(df_sub):
        if len(df_sub) == 0:
            return pd.DataFrame()
        agg = df_sub.groupby('alpha').agg({
            'param_mse': ['mean', 'median'],
        }).reset_index()
        agg.columns = ['alpha', 'param_mse_mean', 'param_mse_median']
        agg = agg.sort_values('alpha')
        agg['param_mse_mean_db'] = to_db(agg['param_mse_mean'].values)
        agg['param_mse_median_db'] = to_db(agg['param_mse_median'].values)
        return agg
    
    agg_conv = agg_func(df_converged)
    agg_notconv = agg_func(df_notconv)
    
    return {
        'df': df_valid,
        'df_converged': df_converged,
        'df_notconv': df_notconv,
        'agg_conv': agg_conv,
        'agg_notconv': agg_notconv,
    }

def main():
    # Parameters
    csv_path = "experiment_results_bg_alpha_sweep.csv"
    output_dir = "figures/diagonal/bg_generalization"
    c = 0.001
    rho = 0.04
    ft_regulariser_scale = 1e-6
    
    # Load results
    print(f"Loading results from {csv_path}...")
    df_all = pd.read_csv(csv_path)
    
    # Process both learning rates
    print("\nProcessing lr=0.5...")
    lr05_data = process_lr_data(df_all, 0.5)
    print(f"  Found {len(lr05_data['df'])} valid experiments")
    print(f"  Converged: {len(lr05_data['df_converged'])}, Not converged: {len(lr05_data['df_notconv'])}")
    
    print("\nProcessing lr=5.0...")
    lr50_data = process_lr_data(df_all, 5.0)
    print(f"  Found {len(lr50_data['df'])} valid experiments")
    print(f"  Converged: {len(lr50_data['df_converged'])}, Not converged: {len(lr50_data['df_notconv'])}")
    
    # Load replica curve
    print(f"\nLoading replica theory curve for c={c}...")
    cache_dir = os.path.join(output_dir, "replica_cache")
    replica_alpha, replica_mse = load_replica_curve(cache_dir, c, rho, ft_regulariser_scale)
    
    if replica_alpha is not None:
        replica_mse_db = to_db(replica_mse)
        print(f"  Loaded {len(replica_alpha)} points")
    else:
        replica_mse_db = None
    
    # Create plot with three subplots
    print("\nCreating plot...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))
    
    # ============= TOP PLOT: lr=0.5 vs Replica =============
    ax = ax1
    
    # lr=0.5 converged
    if len(lr05_data['agg_conv']) > 0:
        valid_mask = np.isfinite(lr05_data['agg_conv']['param_mse_mean_db'])
        agg_valid = lr05_data['agg_conv'][valid_mask].copy()
        ax.plot(
            agg_valid['alpha'],
            agg_valid['param_mse_mean_db'],
            'o-',
            label='lr=0.5 converged',
            linewidth=2,
            markersize=8,
            color='blue',
            markeredgecolor='darkblue',
            markeredgewidth=1.5
        )
    
    # lr=0.5 not converged
    if len(lr05_data['agg_notconv']) > 0:
        valid_mask = np.isfinite(lr05_data['agg_notconv']['param_mse_mean_db'])
        agg_valid = lr05_data['agg_notconv'][valid_mask].copy()
        ax.plot(
            agg_valid['alpha'],
            agg_valid['param_mse_mean_db'],
            'x--',
            label='lr=0.5 not converged',
            linewidth=2,
            markersize=10,
            color='cyan',
            markeredgewidth=2
        )
    
    # Replica theory
    if replica_alpha is not None and replica_mse_db is not None:
        ax.plot(
            replica_alpha,
            replica_mse_db,
            '-',
            label=f'Replica theory',
            linewidth=2.5,
            color='orange',
            alpha=0.9
        )
    
    ax.set_xlabel(r'$\alpha = n_{\mathrm{train}} / d$', fontsize=13)
    ax.set_ylabel('Parameter MSE (dB)', fontsize=13)
    ax.set_title(
        f'lr=0.5 vs Replica Theory (c={c:.3f}, $\\rho={rho:.3f}$)',
        fontsize=13
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    # ============= MIDDLE PLOT: lr=5.0 vs Replica =============
    ax = ax2
    
    # lr=5.0 converged
    if len(lr50_data['agg_conv']) > 0:
        valid_mask = np.isfinite(lr50_data['agg_conv']['param_mse_mean_db'])
        agg_valid = lr50_data['agg_conv'][valid_mask].copy()
        ax.plot(
            agg_valid['alpha'],
            agg_valid['param_mse_mean_db'],
            's-',
            label='lr=5.0 converged',
            linewidth=2,
            markersize=8,
            color='green',
            markeredgecolor='darkgreen',
            markeredgewidth=1.5
        )
    
    # lr=5.0 not converged
    if len(lr50_data['agg_notconv']) > 0:
        valid_mask = np.isfinite(lr50_data['agg_notconv']['param_mse_mean_db'])
        agg_valid = lr50_data['agg_notconv'][valid_mask].copy()
        ax.plot(
            agg_valid['alpha'],
            agg_valid['param_mse_mean_db'],
            '+--',
            label='lr=5.0 not converged',
            linewidth=2,
            markersize=12,
            color='lime',
            markeredgewidth=2
        )
    
    # Replica theory
    if replica_alpha is not None and replica_mse_db is not None:
        ax.plot(
            replica_alpha,
            replica_mse_db,
            '-',
            label=f'Replica theory',
            linewidth=2.5,
            color='orange',
            alpha=0.9
        )
    
    ax.set_xlabel(r'$\alpha = n_{\mathrm{train}} / d$', fontsize=13)
    ax.set_ylabel('Parameter MSE (dB)', fontsize=13)
    ax.set_title(
        f'lr=5.0 vs Replica Theory (c={c:.3f}, $\\rho={rho:.3f}$)',
        fontsize=13
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    # ============= BOTTOM PLOT: Ratio comparison =============
    ax = ax3
    
    # Calculate ratios for both learning rates
    def calc_ratios(agg_conv, agg_notconv, replica_alpha, replica_mse):
        ratios_conv, alphas_conv = [], []
        ratios_notconv, alphas_notconv = [], []
        
        if len(agg_conv) > 0:
            for _, row in agg_conv.iterrows():
                alpha = row['alpha']
                emp_mse = row['param_mse_mean']
                idx = np.abs(replica_alpha - alpha).argmin()
                rep_mse = replica_mse[idx]
                ratios_conv.append(emp_mse / rep_mse)
                alphas_conv.append(alpha)
        
        if len(agg_notconv) > 0:
            for _, row in agg_notconv.iterrows():
                alpha = row['alpha']
                emp_mse = row['param_mse_mean']
                idx = np.abs(replica_alpha - alpha).argmin()
                rep_mse = replica_mse[idx]
                ratios_notconv.append(emp_mse / rep_mse)
                alphas_notconv.append(alpha)
        
        return ratios_conv, alphas_conv, ratios_notconv, alphas_notconv
    
    if replica_alpha is not None:
        # lr=0.5
        r_conv, a_conv, r_notconv, a_notconv = calc_ratios(
            lr05_data['agg_conv'], lr05_data['agg_notconv'], 
            replica_alpha, replica_mse
        )
        if len(a_conv) > 0:
            ax.plot(a_conv, r_conv, 'o-', label='lr=0.5 converged', 
                   linewidth=2, markersize=8, color='blue',
                   markeredgecolor='darkblue', markeredgewidth=1.5)
        if len(a_notconv) > 0:
            ax.plot(a_notconv, r_notconv, 'x--', label='lr=0.5 not converged', 
                   linewidth=2, markersize=10, color='cyan', markeredgewidth=2)
        
        # lr=5.0
        r_conv, a_conv, r_notconv, a_notconv = calc_ratios(
            lr50_data['agg_conv'], lr50_data['agg_notconv'], 
            replica_alpha, replica_mse
        )
        if len(a_conv) > 0:
            ax.plot(a_conv, r_conv, 's-', label='lr=5.0 converged', 
                   linewidth=2, markersize=8, color='green',
                   markeredgecolor='darkgreen', markeredgewidth=1.5)
        if len(a_notconv) > 0:
            ax.plot(a_notconv, r_notconv, '+--', label='lr=5.0 not converged', 
                   linewidth=2, markersize=12, color='lime', markeredgewidth=2)
    
    # Reference line at ratio = 1
    ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Perfect match')
    
    ax.set_xlabel(r'$\alpha = n_{\mathrm{train}} / d$', fontsize=13)
    ax.set_ylabel('Empirical MSE / Replica MSE', fontsize=13)
    ax.set_title('Ratio Comparison: Different Learning Rates', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    ax.set_yscale('log')
    
    fig.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f'lr_comparison_vs_replica_c{c}.png')
    pdf_path = os.path.join(output_dir, f'lr_comparison_vs_replica_c{c}.pdf')
    
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nPlot saved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    
    # Print detailed comparison
    print("\n" + "="*80)
    print("DETAILED COMPARISON: lr=0.5 vs lr=5.0")
    print("="*80)
    
    if replica_alpha is not None:
        print("\nCONVERGED RUNS COMPARISON:")
        print("Alpha | lr=0.5 MSE    | lr=5.0 MSE    | Replica MSE   | Ratio 0.5 | Ratio 5.0")
        print("-"*80)
        
        # Get all alphas present in either dataset
        alphas_05 = set(lr05_data['agg_conv']['alpha'].values) if len(lr05_data['agg_conv']) > 0 else set()
        alphas_50 = set(lr50_data['agg_conv']['alpha'].values) if len(lr50_data['agg_conv']) > 0 else set()
        all_alphas = sorted(alphas_05 | alphas_50)
        
        for alpha in all_alphas:
            # Get lr=0.5 value
            mse_05 = None
            if len(lr05_data['agg_conv']) > 0:
                mask = lr05_data['agg_conv']['alpha'] == alpha
                if mask.any():
                    mse_05 = lr05_data['agg_conv'][mask]['param_mse_mean'].values[0]
            
            # Get lr=5.0 value
            mse_50 = None
            if len(lr50_data['agg_conv']) > 0:
                mask = lr50_data['agg_conv']['alpha'] == alpha
                if mask.any():
                    mse_50 = lr50_data['agg_conv'][mask]['param_mse_mean'].values[0]
            
            # Get replica value
            idx = np.abs(replica_alpha - alpha).argmin()
            mse_rep = replica_mse[idx]
            
            # Format output
            str_05 = f"{mse_05:.6e}" if mse_05 is not None else "    N/A     "
            str_50 = f"{mse_50:.6e}" if mse_50 is not None else "    N/A     "
            ratio_05 = f"{mse_05/mse_rep:.2f}x" if mse_05 is not None else " N/A "
            ratio_50 = f"{mse_50/mse_rep:.2f}x" if mse_50 is not None else " N/A "
            
            print(f"{alpha:.2f}  | {str_05} | {str_50} | {mse_rep:.6e} | {ratio_05:>9s} | {ratio_50:>9s}")
        
        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS (CONVERGED RUNS ONLY)")
        print("="*80)
        
        if len(lr05_data['df_converged']) > 0:
            n_conv_05 = len(lr05_data['df_converged'])
            n_total_05 = len(lr05_data['df'])
            print(f"lr=0.5: {n_conv_05}/{n_total_05} converged ({100*n_conv_05/n_total_05:.1f}%)")
        
        if len(lr50_data['df_converged']) > 0:
            n_conv_50 = len(lr50_data['df_converged'])
            n_total_50 = len(lr50_data['df'])
            print(f"lr=5.0: {n_conv_50}/{n_total_50} converged ({100*n_conv_50/n_total_50:.1f}%)")

if __name__ == '__main__':
    main()

