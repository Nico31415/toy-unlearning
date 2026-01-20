#!/usr/bin/env python3
"""
Generate a combined 1x2 panel showing 4 key configurations:
1. negative λ_pt, small c_pt
2. positive λ_pt, small c_pt  
3. λ_pt=0, large c_pt
4. λ_pt=0, small c_pt, large γ

Left subplot: ω=1 fixed, varying ρ_ft ∈ {0.02, 0.04, 0.1}
Right subplot: ρ_ft=0.1 fixed, varying ω ∈ {0.0, 0.5, 1.0}
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Experiment base directory
EXP_BASE = Path("figures/panel_experiments")
OUTPUT_DIR = Path("figures/panels")

# Fixed params
RHO_PT = 0.1

# Subplot params
LEFT_RHO_FT = [0.02, 0.04, 0.1]
RIGHT_OMEGA = [0.0, 0.5, 1.0]

# The 4 configurations to compare
CONFIGS = [
    {'c_pt': 0.001, 'lambda_pt': -0.00095, 'gamma_reinit': 0.0, 'label': 'λ=-0.95c, c=0.001, γ=0', 'color': '#1f77b4'},
    {'c_pt': 0.001, 'lambda_pt': 0.00095, 'gamma_reinit': 0.0, 'label': 'λ=+0.95c, c=0.001, γ=0', 'color': '#ff7f0e'},
    {'c_pt': 1.0, 'lambda_pt': 0.0, 'gamma_reinit': 0.0, 'label': 'λ=0, c=1.0, γ=0', 'color': '#2ca02c'},
    {'c_pt': 0.001, 'lambda_pt': 0.0, 'gamma_reinit': 1.0, 'label': 'λ=0, c=0.001, γ=1', 'color': '#d62728'},
]


def to_dB(mse):
    """Convert MSE to dB: 10*log10(mse)."""
    mse = np.asarray(mse)
    return 10.0 * np.log10(np.maximum(mse, 1e-15))


def find_exp_dir(rho_ft, omega, c_pt, lambda_pt, gamma_reinit):
    """Find experiment directory matching parameters."""
    def fmt(x):
        return f"{float(x):.6g}".replace("+", "")
    
    dirname = (
        f"rpt={fmt(RHO_PT)}__rft={fmt(rho_ft)}__"
        f"om={fmt(omega)}__cpt={fmt(c_pt)}__"
        f"lpt={fmt(lambda_pt)}__gam={fmt(gamma_reinit)}"
    )
    exp_dir = EXP_BASE / dirname
    return exp_dir if exp_dir.exists() else None


def load_data(exp_dir: Path):
    """Load empirical and replica data from an experiment directory."""
    if exp_dir is None:
        return None, None
    
    subdirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith('ptft_oracle')]
    if not subdirs:
        return None, None
    subdir = subdirs[0]
    
    emp_csv = subdir / "empirical_results.csv"
    emp_df = pd.read_csv(emp_csv) if emp_csv.exists() else None
    
    replica_df = None
    cache_dir = subdir / "replica_cache"
    if cache_dir.exists():
        csv_files = list(cache_dir.glob("*.csv"))
        if csv_files:
            replica_df = pd.read_csv(csv_files[0])
    
    return emp_df, replica_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linestyles for different rho_ft / omega values
    linestyles = ['-', '--', ':']
    markers = ['o', 's', '^']
    
    # LEFT SUBPLOT: omega=1 fixed, varying rho_ft
    ax_left.set_title('ω = 1.0 (fixed), varying ρ_ft', fontsize=12)
    
    for cfg in CONFIGS:
        c_pt = cfg['c_pt']
        lambda_pt = cfg['lambda_pt']
        gamma_reinit = cfg['gamma_reinit']
        color = cfg['color']
        
        for j, rho_ft in enumerate(LEFT_RHO_FT):
            exp_dir = find_exp_dir(rho_ft=rho_ft, omega=1.0, c_pt=c_pt,
                                   lambda_pt=lambda_pt, gamma_reinit=gamma_reinit)
            emp_df, replica_df = load_data(exp_dir)
            
            # Plot replica curve
            if replica_df is not None:
                ax_left.plot(replica_df['alpha'], to_dB(replica_df['mse']),
                            color=color, linewidth=2, linestyle=linestyles[j], alpha=0.9)
            
            # Plot empirical points
            if emp_df is not None:
                df_agg = emp_df.groupby('alpha')['param_mse'].agg(['mean', 'std']).reset_index()
                ax_left.scatter(df_agg['alpha'], to_dB(df_agg['mean']), color=color,
                               marker=markers[j], s=40, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    ax_left.set_xlabel(r'$\alpha = n/d$', fontsize=12)
    ax_left.set_ylabel('Generalization Error (dB)', fontsize=12)
    ax_left.set_xlim(0, 1.05)
    ax_left.grid(True, alpha=0.3, linestyle='--')
    
    # RIGHT SUBPLOT: rho_ft=0.1 fixed, varying omega
    ax_right.set_title('ρ_ft = 0.1 (fixed), varying ω', fontsize=12)
    
    for cfg in CONFIGS:
        c_pt = cfg['c_pt']
        lambda_pt = cfg['lambda_pt']
        gamma_reinit = cfg['gamma_reinit']
        color = cfg['color']
        
        for j, omega in enumerate(RIGHT_OMEGA):
            exp_dir = find_exp_dir(rho_ft=0.1, omega=omega, c_pt=c_pt,
                                   lambda_pt=lambda_pt, gamma_reinit=gamma_reinit)
            emp_df, replica_df = load_data(exp_dir)
            
            # Plot replica curve
            if replica_df is not None:
                ax_right.plot(replica_df['alpha'], to_dB(replica_df['mse']),
                             color=color, linewidth=2, linestyle=linestyles[j], alpha=0.9)
            
            # Plot empirical points
            if emp_df is not None:
                df_agg = emp_df.groupby('alpha')['param_mse'].agg(['mean', 'std']).reset_index()
                ax_right.scatter(df_agg['alpha'], to_dB(df_agg['mean']), color=color,
                                marker=markers[j], s=40, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    ax_right.set_xlabel(r'$\alpha = n/d$', fontsize=12)
    ax_right.set_ylabel('Generalization Error (dB)', fontsize=12)
    ax_right.set_xlim(0, 1.05)
    ax_right.grid(True, alpha=0.3, linestyle='--')
    
    # Create legend
    # Config colors
    config_handles = [Line2D([0], [0], color=cfg['color'], linewidth=2, label=cfg['label']) 
                      for cfg in CONFIGS]
    
    # Linestyle legend (for rho_ft in left, omega in right)
    style_handles_left = [Line2D([0], [0], color='gray', linewidth=1.5, linestyle=ls, 
                                  label=f'ρ_ft={rft}') for ls, rft in zip(linestyles, LEFT_RHO_FT)]
    style_handles_right = [Line2D([0], [0], color='gray', linewidth=1.5, linestyle=ls,
                                   label=f'ω={om}') for ls, om in zip(linestyles, RIGHT_OMEGA)]
    
    # Add legends to each subplot
    ax_left.legend(handles=config_handles + style_handles_left, loc='upper right', fontsize=8, ncol=2)
    ax_right.legend(handles=config_handles + style_handles_right, loc='upper right', fontsize=8, ncol=2)
    
    fig.suptitle('PT+FT Oracle: Key Configuration Comparison\n(Lines = Replica, Markers = Empirical)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    fig.savefig(OUTPUT_DIR / 'combined_panel.png', dpi=150, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'combined_panel.pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved combined_panel.png/pdf to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()


