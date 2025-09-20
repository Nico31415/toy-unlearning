#!/usr/bin/env python3
"""
Script to create log-log plots showing validation loss vs n_train2 for both initialization methods.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_final_val_mse(experiment_dir):
    """Extract the final validation MSE from an experiment directory."""
    # Look for the df.feather file which contains the training history
    df_file = os.path.join(experiment_dir, 'df.feather')
    
    if not os.path.exists(df_file):
        print(f"Warning: {df_file} not found")
        return None
    
    try:
        df = pd.read_feather(df_file)
        # Get the final validation loss
        val_df = df[df['split'] == 'val']
        if len(val_df) > 0:
            final_val_mse = val_df['loss'].iloc[-1]
            return final_val_mse
        else:
            print(f"Warning: No validation data found in {df_file}")
            return None
    except Exception as e:
        print(f"Error reading {df_file}: {e}")
        return None

def parse_experiment_params(dirname):
    """Parse experiment parameters from directory name."""
    # Extract n_train2 and init_method from directory name
    n_train2_match = re.search(r'n_train2=(\d+)', dirname)
    init_method_match = re.search(r'init_method=(\w+)', dirname)
    
    if n_train2_match and init_method_match:
        n_train2 = int(n_train2_match.group(1))
        init_method = init_method_match.group(1)
        return n_train2, init_method
    else:
        return None, None

def main():
    # Base directory containing all experiments
    base_dir = "data/diagonal/sparse_overlap2"
    
    # Collect data from all experiments
    results = []
    
    for dirname in os.listdir(base_dir):
        if 'n_train2=' in dirname:
            n_train2, init_method = parse_experiment_params(dirname)
            if n_train2 is not None and init_method is not None:
                experiment_dir = os.path.join(base_dir, dirname)
                final_val_mse = extract_final_val_mse(experiment_dir)
                
                if final_val_mse is not None:
                    results.append({
                        'n_train2': n_train2,
                        'init_method': init_method,
                        'final_val_mse': final_val_mse
                    })
                    print(f"Found: {init_method}, n_train2={n_train2}, val_mse={final_val_mse:.6f}")
    
    if not results:
        print("No valid results found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values(['init_method', 'n_train2'])
    
    print(f"\nCollected {len(df)} results:")
    print(df)
    
    # Create log-log plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Both methods on same plot
    for init_method in df['init_method'].unique():
        method_data = df[df['init_method'] == init_method]
        ax1.loglog(method_data['n_train2'], method_data['final_val_mse'], 
                  'o-', label=f'{init_method} initialization', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Training Set Size (n_train2)', fontsize=12)
    ax1.set_ylabel('Final Validation MSE', fontsize=12)
    ax1.set_title('Validation Loss vs Training Set Size\n(Log-Log Scale)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Separate plots for each method
    for i, init_method in enumerate(df['init_method'].unique()):
        method_data = df[df['init_method'] == init_method]
        ax2.loglog(method_data['n_train2'], method_data['final_val_mse'], 
                  'o-', label=f'{init_method} initialization', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Training Set Size (n_train2)', fontsize=12)
    ax2.set_ylabel('Final Validation MSE', fontsize=12)
    ax2.set_title('Validation Loss vs Training Set Size\n(Separate Methods)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, "scaling_law_validation_loss.png")
    pdf_path = os.path.join(output_dir, "scaling_law_validation_loss.pdf")
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"\nPlots saved to:")
    print(f"  {plot_path}")
    print(f"  {pdf_path}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    for init_method in df['init_method'].unique():
        method_data = df[df['init_method'] == init_method]
        print(f"\n{init_method.upper()} Initialization:")
        for _, row in method_data.iterrows():
            print(f"  n_train2={row['n_train2']:4d}: val_mse={row['final_val_mse']:.6f}")
    
    plt.show()

if __name__ == "__main__":
    main()
