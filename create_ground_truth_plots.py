#!/usr/bin/env python3
"""
Ground truth analysis plotting script for the diagonal network experiments.

This script creates plots specifically focused on ground truth analysis:
1. Ground truth comparison plots (pretraining vs finetuning)
2. Learned vs ground truth comparison plots
3. Overlap analysis plots
4. Correlation analysis plots
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def find_ground_truth_plots():
    """Find all ground truth comparison plots in the results directories."""
    base_dir = "data/diagonal/sparse_overlap2"
    plot_files = []
    
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} not found!")
        return []
    
    for dirname in os.listdir(base_dir):
        if 'init_method=' in dirname:
            experiment_dir = os.path.join(base_dir, dirname)
            if os.path.isdir(experiment_dir):
                # Look for ground truth plots
                gt_comparison = os.path.join(experiment_dir, 'ground_truth_comparison.png')
                learned_vs_gt = os.path.join(experiment_dir, 'learned_vs_ground_truth.png')
                
                if os.path.exists(gt_comparison):
                    plot_files.append({
                        'type': 'ground_truth_comparison',
                        'path': gt_comparison,
                        'experiment_dir': experiment_dir,
                        'dirname': dirname
                    })
                
                if os.path.exists(learned_vs_gt):
                    plot_files.append({
                        'type': 'learned_vs_ground_truth',
                        'path': learned_vs_gt,
                        'experiment_dir': experiment_dir,
                        'dirname': dirname
                    })
    
    return plot_files

def parse_experiment_params(dirname):
    """Parse experiment parameters from directory name."""
    params = {}
    
    # Extract parameters using regex
    patterns = {
        'init_method': r'init_method=(\w+)',
        'active_dim_2': r'active_dim_2=(\d+)',
        'overlap_bool': r'overlap_bool=(\w+)',
        'lmda': r'lmda=([\d\.\-e]+)',
        'n_train2': r'n_train2=(\d+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, dirname)
        if match:
            if key in ['active_dim_2', 'n_train2']:
                params[key] = int(match.group(1))
            elif key == 'lmda':
                params[key] = float(match.group(1))
            else:
                params[key] = match.group(1)
    
    return params

def create_ground_truth_summary_plot():
    """Create a summary plot showing ground truth relationships across experiments."""
    plot_files = find_ground_truth_plots()
    
    if not plot_files:
        print("No ground truth plots found!")
        return
    
    # Parse all experiment parameters
    experiments = []
    for plot_file in plot_files:
        if plot_file['type'] == 'ground_truth_comparison':
            params = parse_experiment_params(plot_file['dirname'])
            params['plot_path'] = plot_file['path']
            experiments.append(params)
    
    if not experiments:
        print("No ground truth comparison plots found!")
        return
    
    # Create a grid of ground truth comparison plots
    n_experiments = len(experiments)
    n_cols = 4
    n_rows = (n_experiments + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    fig.suptitle('Ground Truth Comparison Across All Experiments', fontsize=16)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, exp in enumerate(experiments):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Load and display the plot
        if os.path.exists(exp['plot_path']):
            img = plt.imread(exp['plot_path'])
            ax.imshow(img)
            ax.axis('off')
            
            # Create title with experiment parameters
            title = f"Init: {exp.get('init_method', 'N/A')}\n"
            title += f"Dim2: {exp.get('active_dim_2', 'N/A')}\n"
            title += f"Overlap: {exp.get('overlap_bool', 'N/A')}\n"
            title += f"λ: {exp.get('lmda', 'N/A')}"
            ax.set_title(title, fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Plot not found', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_experiments, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save the summary plot
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, "ground_truth_summary.png")
    pdf_path = os.path.join(output_dir, "ground_truth_summary.pdf")
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"Ground truth summary plot saved to:")
    print(f"  {plot_path}")
    print(f"  {pdf_path}")
    
    plt.show()

def create_learned_vs_ground_truth_summary():
    """Create a summary plot showing learned vs ground truth comparisons."""
    plot_files = find_ground_truth_plots()
    
    # Filter for learned vs ground truth plots
    learned_plots = [p for p in plot_files if p['type'] == 'learned_vs_ground_truth']
    
    if not learned_plots:
        print("No learned vs ground truth plots found!")
        return
    
    # Parse all experiment parameters
    experiments = []
    for plot_file in learned_plots:
        params = parse_experiment_params(plot_file['dirname'])
        params['plot_path'] = plot_file['path']
        experiments.append(params)
    
    # Create a grid of learned vs ground truth plots
    n_experiments = len(experiments)
    n_cols = 4
    n_rows = (n_experiments + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    fig.suptitle('Learned vs Ground Truth Comparison Across All Experiments', fontsize=16)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, exp in enumerate(experiments):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Load and display the plot
        if os.path.exists(exp['plot_path']):
            img = plt.imread(exp['plot_path'])
            ax.imshow(img)
            ax.axis('off')
            
            # Create title with experiment parameters
            title = f"Init: {exp.get('init_method', 'N/A')}\n"
            title += f"Dim2: {exp.get('active_dim_2', 'N/A')}\n"
            title += f"Overlap: {exp.get('overlap_bool', 'N/A')}\n"
            title += f"λ: {exp.get('lmda', 'N/A')}"
            ax.set_title(title, fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Plot not found', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_experiments, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save the summary plot
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, "learned_vs_ground_truth_summary.png")
    pdf_path = os.path.join(output_dir, "learned_vs_ground_truth_summary.pdf")
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"Learned vs ground truth summary plot saved to:")
    print(f"  {plot_path}")
    print(f"  {pdf_path}")
    
    plt.show()

def create_experiment_summary_table():
    """Create a summary table of all experiments with their parameters."""
    plot_files = find_ground_truth_plots()
    
    if not plot_files:
        print("No experiments found!")
        return
    
    # Parse all experiment parameters
    experiments = []
    for plot_file in plot_files:
        if plot_file['type'] == 'ground_truth_comparison':
            params = parse_experiment_params(plot_file['dirname'])
            params['experiment_dir'] = plot_file['experiment_dir']
            experiments.append(params)
    
    if not experiments:
        print("No experiments with ground truth comparisons found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(experiments)
    
    # Sort by parameters for better readability
    sort_cols = ['init_method', 'active_dim_2', 'overlap_bool', 'lmda']
    available_sort_cols = [col for col in sort_cols if col in df.columns]
    if available_sort_cols:
        df = df.sort_values(available_sort_cols)
    
    print("\n" + "="*100)
    print("EXPERIMENT SUMMARY TABLE")
    print("="*100)
    print(df.to_string(index=False))
    
    # Save to CSV
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "experiment_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nExperiment summary saved to: {csv_path}")

def main():
    """Main function to create all ground truth plots and summaries."""
    print("Creating ground truth analysis plots...")
    
    print("\n1. Creating ground truth summary plot...")
    create_ground_truth_summary_plot()
    
    print("\n2. Creating learned vs ground truth summary...")
    create_learned_vs_ground_truth_summary()
    
    print("\n3. Creating experiment summary table...")
    create_experiment_summary_table()
    
    print("\nAll ground truth analysis plots created successfully!")

if __name__ == "__main__":
    main()
