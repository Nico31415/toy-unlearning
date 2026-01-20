#!/usr/bin/env python3
"""
Generate two 1x2 panels for c_pt = gamma_reinit cases:
1. c_pt = gamma_reinit = 1.0, λ_pt ∈ {-0.95, 0, +0.95}
2. c_pt = gamma_reinit = 0.001, λ_pt ∈ {-0.00095, 0, +0.00095}
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


def plot_panel(c_pt, gamma_reinit, lambda_pts, output_name):
    """Generate a single 1x2 panel for given c_pt = gamma_reinit."""
    
    # Define configs for this panel
    configs = [
        {'lambda_pt': lambda_pts[0], 'label': r'$\lambda_{pt}=-0.95c_{pt}$', 'color': '#1f77b4'},
        {'lambda_pt': lambda_pts[2], 'label': r'$\lambda_{pt}=+0.95c_{pt}$', 'color': '#ff7f0e'},
        {'lambda_pt': lambda_pts[1], 'label': r'$\lambda_{pt}=0$', 'color': '#2ca02c'},
    ]
    
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linestyles for different rho_ft / omega values
    linestyles = ['-', '--', ':']
    markers = ['o', 's', '^']
    
    # LEFT SUBPLOT: omega=1 fixed, varying rho_ft
    ax_left.set_title('ω = 1.0 (fixed), varying ρ_ft', fontsize=12)
    
    for cfg in configs:
        lambda_pt = cfg['lambda_pt']
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
    
    for cfg in configs:
        lambda_pt = cfg['lambda_pt']
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
    config_handles = [Line2D([0], [0], color=cfg['color'], linewidth=2, label=cfg['label']) 
                      for cfg in configs]
    
    style_handles_left = [Line2D([0], [0], color='gray', linewidth=1.5, linestyle=ls, 
                                  label=f'ρ_ft={rft}') for ls, rft in zip(linestyles, LEFT_RHO_FT)]
    style_handles_right = [Line2D([0], [0], color='gray', linewidth=1.5, linestyle=ls,
                                   label=f'ω={om}') for ls, om in zip(linestyles, RIGHT_OMEGA)]
    
    ax_left.legend(handles=config_handles + style_handles_left, loc='upper right', fontsize=8, ncol=2)
    ax_right.legend(handles=config_handles + style_handles_right, loc='upper right', fontsize=8, ncol=2)
    
    fig.suptitle(f'PT+FT Oracle: c_pt = γ = {c_pt}\n(Lines = Replica, Markers = Empirical)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    fig.savefig(OUTPUT_DIR / f'{output_name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{output_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved {output_name}.png/pdf to {OUTPUT_DIR}/")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Panel 1: c_pt = gamma = 1.0
    plot_panel(
        c_pt=1.0, 
        gamma_reinit=1.0, 
        lambda_pts=[-0.95, 0.0, 0.95],
        output_name='cpt_eq_gamma_1'
    )
    
    # Panel 2: c_pt = gamma = 0.001
    plot_panel(
        c_pt=0.001, 
        gamma_reinit=0.001, 
        lambda_pts=[-0.00095, 0.0, 0.00095],
        output_name='cpt_eq_gamma_0001'
    )


if __name__ == '__main__':
    main()

