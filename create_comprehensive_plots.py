#!/usr/bin/env python3
"""
Comprehensive plotting script for the new experimental setup.

This script creates various plots for the diagonal network experiments with:
- 4 pretraining configurations: simple/complex init × lmda=[0, -c_init]
- 16 finetuning configurations: 2 init_methods × 2 active_dim_2 × 2 overlap_bool × 2 lmda values

Plots include:
1. Validation loss vs training set size (scaling laws)
2. Performance comparison across different configurations
3. Ground truth analysis plots
4. Learned vs ground truth comparisons
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def extract_final_val_mse(experiment_dir):
    """Extract the final validation MSE from an experiment directory."""
    df_file = os.path.join(experiment_dir, 'df.feather')
    
    if not os.path.exists(df_file):
        return None
    
    try:
        df = pd.read_feather(df_file)
        val_df = df[df['split'] == 'val']
        if len(val_df) > 0:
            return val_df['loss'].iloc[-1]
        else:
            return None
    except Exception as e:
        print(f"Error reading {df_file}: {e}")
        return None

def parse_experiment_params(dirname):
    """Parse experiment parameters from directory name."""
    # Extract parameters from directory name
    n_train2_match = re.search(r'n_train2=(\d+)', dirname)
    init_method_match = re.search(r'init_method=(\w+)', dirname)
    active_dim_2_match = re.search(r'active_dim_2=(\d+)', dirname)
    overlap_bool_match = re.search(r'overlap_bool=(\w+)', dirname)
    lmda_match = re.search(r'lmda=([\d\.\-e]+)', dirname)
    
    params = {}
    if n_train2_match:
        params['n_train2'] = int(n_train2_match.group(1))
    if init_method_match:
        params['init_method'] = init_method_match.group(1)
    if active_dim_2_match:
        params['active_dim_2'] = int(active_dim_2_match.group(1))
    if overlap_bool_match:
        params['overlap_bool'] = overlap_bool_match.group(1)
    if lmda_match:
        params['lmda'] = float(lmda_match.group(1))
    
    return params

def collect_finetuning_results():
    """Collect results from all finetuning experiments."""
    base_dir = "data/diagonal/sparse_overlap2"
    results = []
    
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} not found!")
        return pd.DataFrame()
    
    for dirname in os.listdir(base_dir):
        if 'init_method=' in dirname and 'n_train2=' in dirname:
            params = parse_experiment_params(dirname)
            if len(params) >= 4:  # Need at least init_method, active_dim_2, overlap_bool, lmda
                experiment_dir = os.path.join(base_dir, dirname)
                final_val_mse = extract_final_val_mse(experiment_dir)
                
                if final_val_mse is not None:
                    result = params.copy()
                    result['final_val_mse'] = final_val_mse
                    result['experiment_dir'] = experiment_dir
                    results.append(result)
                    print(f"Found: {params['init_method']}, active_dim_2={params['active_dim_2']}, "
                          f"overlap_bool={params['overlap_bool']}, lmda={params['lmda']}, "
                          f"val_mse={final_val_mse:.6f}")
    
    return pd.DataFrame(results)

def create_scaling_law_plots(df):
    """Create scaling law plots for different configurations."""
    if df.empty:
        print("No data available for scaling law plots")
        return
    
    # Filter for experiments with n_train2 data
    scaling_df = df[df['n_train2'].notna()].copy()
    if scaling_df.empty:
        print("No scaling data available")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Scaling Laws: Validation Loss vs Training Set Size', fontsize=16)
    
    # Plot 1: All configurations on one plot
    ax1 = axes[0, 0]
    for (init_method, active_dim_2, overlap_bool, lmda), group in scaling_df.groupby(['init_method', 'active_dim_2', 'overlap_bool', 'lmda']):
        label = f"{init_method}, dim2={active_dim_2}, overlap={overlap_bool}, λ={lmda}"
        ax1.loglog(group['n_train2'], group['final_val_mse'], 'o-', label=label, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Training Set Size (n_train2)')
    ax1.set_ylabel('Final Validation MSE')
    ax1.set_title('All Configurations')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: By initialization method
    ax2 = axes[0, 1]
    for init_method in scaling_df['init_method'].unique():
        method_data = scaling_df[scaling_df['init_method'] == init_method]
        # Average across other dimensions
        avg_data = method_data.groupby('n_train2')['final_val_mse'].mean()
        ax2.loglog(avg_data.index, avg_data.values, 'o-', label=f'{init_method} init', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Training Set Size (n_train2)')
    ax2.set_ylabel('Final Validation MSE (Averaged)')
    ax2.set_title('By Initialization Method')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: By overlap setting
    ax3 = axes[1, 0]
    for overlap_bool in scaling_df['overlap_bool'].unique():
        overlap_data = scaling_df[scaling_df['overlap_bool'] == overlap_bool]
        # Average across other dimensions
        avg_data = overlap_data.groupby('n_train2')['final_val_mse'].mean()
        ax3.loglog(avg_data.index, avg_data.values, 'o-', label=f'overlap={overlap_bool}', linewidth=2, markersize=8)
    
    ax3.set_xlabel('Training Set Size (n_train2)')
    ax3.set_ylabel('Final Validation MSE (Averaged)')
    ax3.set_title('By Overlap Setting')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: By active_dim_2
    ax4 = axes[1, 1]
    for active_dim_2 in scaling_df['active_dim_2'].unique():
        dim_data = scaling_df[scaling_df['active_dim_2'] == active_dim_2]
        # Average across other dimensions
        avg_data = dim_data.groupby('n_train2')['final_val_mse'].mean()
        ax4.loglog(avg_data.index, avg_data.values, 'o-', label=f'active_dim_2={active_dim_2}', linewidth=2, markersize=8)
    
    ax4.set_xlabel('Training Set Size (n_train2)')
    ax4.set_ylabel('Final Validation MSE (Averaged)')
    ax4.set_title('By Active Dimension 2')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, "comprehensive_scaling_laws.png")
    pdf_path = os.path.join(output_dir, "comprehensive_scaling_laws.pdf")
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"Scaling law plots saved to:")
    print(f"  {plot_path}")
    print(f"  {pdf_path}")
    
    plt.show()

def create_performance_comparison_plots(df):
    """Create performance comparison plots across different configurations."""
    if df.empty:
        print("No data available for performance comparison plots")
        return
    
    # Filter for fixed n_train2 experiments (if available)
    fixed_df = df[df['n_train2'] == 64].copy() if 'n_train2' in df.columns else df.copy()
    
    if fixed_df.empty:
        print("No fixed n_train2 data available for performance comparison")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Comparison Across Configurations', fontsize=16)
    
    # Plot 1: Heatmap of validation MSE by init_method and lmda
    ax1 = axes[0, 0]
    if len(fixed_df) > 1:
        pivot_data = fixed_df.groupby(['init_method', 'lmda'])['final_val_mse'].mean().unstack()
        sns.heatmap(pivot_data, annot=True, fmt='.6f', cmap='viridis', ax=ax1)
        ax1.set_title('Validation MSE by Init Method and Lambda')
        ax1.set_xlabel('Lambda (lmda)')
        ax1.set_ylabel('Initialization Method')
    
    # Plot 2: Box plot by initialization method
    ax2 = axes[0, 1]
    if len(fixed_df) > 1:
        sns.boxplot(data=fixed_df, x='init_method', y='final_val_mse', ax=ax2)
        ax2.set_title('Validation MSE Distribution by Init Method')
        ax2.set_ylabel('Final Validation MSE')
        ax2.set_xlabel('Initialization Method')
    
    # Plot 3: Box plot by overlap setting
    ax3 = axes[1, 0]
    if len(fixed_df) > 1:
        sns.boxplot(data=fixed_df, x='overlap_bool', y='final_val_mse', ax=ax3)
        ax3.set_title('Validation MSE Distribution by Overlap')
        ax3.set_ylabel('Final Validation MSE')
        ax3.set_xlabel('Overlap Boolean')
    
    # Plot 4: Box plot by active_dim_2
    ax4 = axes[1, 1]
    if len(fixed_df) > 1:
        sns.boxplot(data=fixed_df, x='active_dim_2', y='final_val_mse', ax=ax4)
        ax4.set_title('Validation MSE Distribution by Active Dim 2')
        ax4.set_ylabel('Final Validation MSE')
        ax4.set_xlabel('Active Dimension 2')
    
    plt.tight_layout()
    
    # Save plots
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, "performance_comparison.png")
    pdf_path = os.path.join(output_dir, "performance_comparison.pdf")
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"Performance comparison plots saved to:")
    print(f"  {plot_path}")
    print(f"  {pdf_path}")
    
    plt.show()

def create_summary_table(df):
    """Create a summary table of all results."""
    if df.empty:
        print("No data available for summary table")
        return
    
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    
    # Group by key parameters and show statistics
    if 'n_train2' in df.columns:
        print("\nBy Training Set Size:")
        for n_train2 in sorted(df['n_train2'].unique()):
            subset = df[df['n_train2'] == n_train2]
            print(f"  n_train2={n_train2}:")
            print(f"    Mean Val MSE: {subset['final_val_mse'].mean():.6f}")
            print(f"    Std Val MSE:  {subset['final_val_mse'].std():.6f}")
            print(f"    Min Val MSE:  {subset['final_val_mse'].min():.6f}")
            print(f"    Max Val MSE:  {subset['final_val_mse'].max():.6f}")
    
    print("\nBy Initialization Method:")
    for init_method in df['init_method'].unique():
        subset = df[df['init_method'] == init_method]
        print(f"  {init_method}:")
        print(f"    Mean Val MSE: {subset['final_val_mse'].mean():.6f}")
        print(f"    Std Val MSE:  {subset['final_val_mse'].std():.6f}")
        print(f"    Count:        {len(subset)}")
    
    print("\nBy Overlap Setting:")
    for overlap_bool in df['overlap_bool'].unique():
        subset = df[df['overlap_bool'] == overlap_bool]
        print(f"  overlap={overlap_bool}:")
        print(f"    Mean Val MSE: {subset['final_val_mse'].mean():.6f}")
        print(f"    Std Val MSE:  {subset['final_val_mse'].std():.6f}")
        print(f"    Count:        {len(subset)}")
    
    print("\nBy Active Dimension 2:")
    for active_dim_2 in df['active_dim_2'].unique():
        subset = df[df['active_dim_2'] == active_dim_2]
        print(f"  active_dim_2={active_dim_2}:")
        print(f"    Mean Val MSE: {subset['final_val_mse'].mean():.6f}")
        print(f"    Std Val MSE:  {subset['final_val_mse'].std():.6f}")
        print(f"    Count:        {len(subset)}")
    
    print("\nBy Lambda Value:")
    for lmda in df['lmda'].unique():
        subset = df[df['lmda'] == lmda]
        print(f"  λ={lmda}:")
        print(f"    Mean Val MSE: {subset['final_val_mse'].mean():.6f}")
        print(f"    Std Val MSE:  {subset['final_val_mse'].std():.6f}")
        print(f"    Count:        {len(subset)}")
    
    # Save detailed results to CSV
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "experimental_results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")

def main():
    """Main function to create all plots and summaries."""
    print("Collecting finetuning results...")
    df = collect_finetuning_results()
    
    if df.empty:
        print("No experimental results found!")
        print("Make sure to run the experiments first.")
        return
    
    print(f"\nCollected {len(df)} experimental results")
    
    # Create all plots
    print("\nCreating scaling law plots...")
    create_scaling_law_plots(df)
    
    print("\nCreating performance comparison plots...")
    create_performance_comparison_plots(df)
    
    print("\nCreating summary table...")
    create_summary_table(df)
    
    print("\nAll plots and summaries created successfully!")

if __name__ == "__main__":
    main()
