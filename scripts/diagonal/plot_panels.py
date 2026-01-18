#!/usr/bin/env python3
"""
Generate the 3 panel figures from completed experiments.
Each panel is a 3x2 grid (rows = varying parameter values, columns = rho_ft vs omega sweep).

Left column: omega=1 FIXED, varying rho_ft ∈ {0.02, 0.04, 0.1}
Right column: rho_ft=0.1 FIXED, varying omega ∈ {0.0, 0.5, 1.0}

Panel 1: Varying lambda_pt (rows: -0.95*c_pt, 0, +0.95*c_pt)
Panel 2: Varying c_pt (rows: 0.001, 0.01, 0.1, 0.5, 1.0) -> 5x2 grid
Panel 3: Varying gamma_reinit (rows: 0.001, 0.01, 0.1, 1.0) -> 4x2 grid
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
    """Convert MSE to dB: 10*log10(mse). Handles zeros/negatives."""
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


def plot_single_subplot(ax, rho_ft_values, omega_values, c_pt, lambda_pt, gamma_reinit, 
                        is_left_column, row_label):
    """Plot a single subplot (either rho_ft sweep or omega sweep)."""
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    markers = ['o', 's', '^']
    
    if is_left_column:
        # omega=1 fixed, varying rho_ft
        for j, rho_ft in enumerate(rho_ft_values):
            exp_dir = find_exp_dir(rho_ft=rho_ft, omega=1.0, c_pt=c_pt, 
                                   lambda_pt=lambda_pt, gamma_reinit=gamma_reinit)
            emp_df, replica_df = load_data(exp_dir)
            
            label = f'ρ_ft={rho_ft}'
            
            # Plot replica curve
            if replica_df is not None:
                replica_dB = to_dB(replica_df['mse'])
                ax.plot(replica_df['alpha'], replica_dB, color=colors[j], 
                       linewidth=2, alpha=0.9, label=label)
            
            # Plot empirical points
            if emp_df is not None:
                df_agg = emp_df.groupby('alpha')['param_mse'].agg(['mean', 'std']).reset_index()
                emp_dB = to_dB(df_agg['mean'])
                ax.scatter(df_agg['alpha'], emp_dB, color=colors[j], marker=markers[j],
                          s=50, alpha=0.8, edgecolors='black', linewidths=0.5, zorder=10)
        
        ax.set_title(f'{row_label}\nω=1.0 fixed', fontsize=10)
    else:
        # rho_ft=0.1 fixed, varying omega
        for j, omega in enumerate(omega_values):
            exp_dir = find_exp_dir(rho_ft=0.1, omega=omega, c_pt=c_pt,
                                   lambda_pt=lambda_pt, gamma_reinit=gamma_reinit)
            emp_df, replica_df = load_data(exp_dir)
            
            label = f'ω={omega}'
            
            # Plot replica curve
            if replica_df is not None:
                replica_dB = to_dB(replica_df['mse'])
                ax.plot(replica_df['alpha'], replica_dB, color=colors[j],
                       linewidth=2, alpha=0.9, label=label)
            
            # Plot empirical points
            if emp_df is not None:
                df_agg = emp_df.groupby('alpha')['param_mse'].agg(['mean', 'std']).reset_index()
                emp_dB = to_dB(df_agg['mean'])
                ax.scatter(df_agg['alpha'], emp_dB, color=colors[j], marker=markers[j],
                          s=50, alpha=0.8, edgecolors='black', linewidths=0.5, zorder=10)
        
        ax.set_title(f'{row_label}\nρ_ft=0.1 fixed', fontsize=10)
    
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=8)


def create_panel_figure(varying_param_name, varying_values, fixed_params, title):
    """Create a Nx2 grid panel figure."""
    n_rows = len(varying_values)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.5 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, var_val in enumerate(varying_values):
        params = fixed_params.copy()
        params[varying_param_name] = var_val
        
        # Row label
        if varying_param_name == 'lambda_pt':
            row_label = f'λ_pt = {var_val:.5f}'
        elif varying_param_name == 'c_pt':
            row_label = f'c_pt = {var_val}'
        else:
            row_label = f'γ = {var_val}'
        
        # Left column: omega=1, varying rho_ft
        plot_single_subplot(
            axes[i, 0], LEFT_RHO_FT, RIGHT_OMEGA,
            c_pt=params.get('c_pt', 0.001),
            lambda_pt=params.get('lambda_pt', 0.0),
            gamma_reinit=params.get('gamma_reinit', 0.0),
            is_left_column=True,
            row_label=row_label
        )
        
        # Right column: rho_ft=0.1, varying omega
        plot_single_subplot(
            axes[i, 1], LEFT_RHO_FT, RIGHT_OMEGA,
            c_pt=params.get('c_pt', 0.001),
            lambda_pt=params.get('lambda_pt', 0.0),
            gamma_reinit=params.get('gamma_reinit', 0.0),
            is_left_column=False,
            row_label=row_label
        )
        
        # Y-axis label only on left column
        axes[i, 0].set_ylabel('Gen. Error (dB)', fontsize=10)
    
    # X-axis labels only on bottom row
    axes[-1, 0].set_xlabel(r'$\alpha = n/d$', fontsize=12)
    axes[-1, 1].set_xlabel(r'$\alpha = n/d$', fontsize=12)
    
    fig.suptitle(title + '\n(Lines = Replica, Markers = Empirical)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Panel 1: Varying lambda_pt (c_pt=0.001, gamma=0)
    print("Creating Panel 1: Varying λ_pt...")
    c_pt_1 = 0.001
    lambda_pt_values = [-0.95 * c_pt_1, 0.0, 0.95 * c_pt_1]
    fig1 = create_panel_figure(
        varying_param_name='lambda_pt',
        varying_values=lambda_pt_values,
        fixed_params={'c_pt': c_pt_1, 'gamma_reinit': 0.0},
        title=f'Panel 1: Varying λ_pt (c_pt={c_pt_1}, γ=0, ρ_pt={RHO_PT})'
    )
    fig1.savefig(OUTPUT_DIR / 'panel1_varying_lambda_pt.png', dpi=150, bbox_inches='tight')
    fig1.savefig(OUTPUT_DIR / 'panel1_varying_lambda_pt.pdf', bbox_inches='tight')
    plt.close(fig1)
    print("  Saved panel1_varying_lambda_pt.png/pdf")
    
    # Panel 2: Varying c_pt (lambda_pt=0, gamma=0)
    print("Creating Panel 2: Varying c_pt...")
    c_pt_values = [0.001, 0.01, 0.1, 0.5, 1.0]
    fig2 = create_panel_figure(
        varying_param_name='c_pt',
        varying_values=c_pt_values,
        fixed_params={'lambda_pt': 0.0, 'gamma_reinit': 0.0},
        title=f'Panel 2: Varying c_pt (λ_pt=0, γ=0, ρ_pt={RHO_PT})'
    )
    fig2.savefig(OUTPUT_DIR / 'panel2_varying_c_pt.png', dpi=150, bbox_inches='tight')
    fig2.savefig(OUTPUT_DIR / 'panel2_varying_c_pt.pdf', bbox_inches='tight')
    plt.close(fig2)
    print("  Saved panel2_varying_c_pt.png/pdf")
    
    # Panel 3: Varying gamma_reinit (c_pt=0.001, lambda_pt=0)
    print("Creating Panel 3: Varying γ...")
    gamma_values = [0.001, 0.01, 0.1, 1.0]
    fig3 = create_panel_figure(
        varying_param_name='gamma_reinit',
        varying_values=gamma_values,
        fixed_params={'c_pt': 0.001, 'lambda_pt': 0.0},
        title=f'Panel 3: Varying γ_reinit (c_pt=0.001, λ_pt=0, ρ_pt={RHO_PT})'
    )
    fig3.savefig(OUTPUT_DIR / 'panel3_varying_gamma.png', dpi=150, bbox_inches='tight')
    fig3.savefig(OUTPUT_DIR / 'panel3_varying_gamma.pdf', bbox_inches='tight')
    plt.close(fig3)
    print("  Saved panel3_varying_gamma.png/pdf")
    
    print(f"\n✓ All panels saved to {OUTPUT_DIR}/")
    print("Y-axis: Generalization Error in dB = 10*log10(MSE)")


if __name__ == '__main__':
    main()
