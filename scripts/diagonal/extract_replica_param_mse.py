#!/usr/bin/env python3
"""
Extract and display replica theory curves for parameter MSE.

This script loads the cached replica curves for c=0.001 and c=0.5
and displays them as parameter MSE (which is what the replica solver outputs).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def to_db(x):
    """Convert MSE to dB: 10*log10(mse + 1e-15)"""
    return 10.0 * np.log10(np.maximum(x, 1e-15))

def load_replica_curve(cache_dir, c, rho=0.04, ft_regulariser_scale=1e-6, 
                       alpha_min=0.008, alpha_max=1.0, alpha_points=100,
                       mc_samples=50000, seed=12345):
    """Load replica curve from cache. Returns None, None if not found."""
    cache_filename = (
        f"replica_curve_teacher=bg--rho={rho:.6f}--c={c:.6f}--"
        f"ft_reg={ft_regulariser_scale:.6e}--alpha_min={alpha_min:.4f}--"
        f"alpha_max={alpha_max:.4f}--alpha_points={alpha_points}--"
        f"mc_samples={mc_samples}--seed={seed}.csv"
    )
    cache_path = os.path.join(cache_dir, cache_filename)
    
    if not os.path.exists(cache_path):
        return None, None
    
    df = pd.read_csv(cache_path)
    return df["alpha"].values, df["mse"].values

def main():
    # Parameters
    cache_dir = "figures/diagonal/bg_generalization/replica_cache"
    output_dir = "figures/diagonal/bg_generalization"
    c_values = [0.001, 0.5]
    rho_values = [0.01, 0.05, 0.1, 0.5, 0.8]
    ft_regulariser_scale = 1e-6
    
    # Load replica curves
    print("Loading replica theory curves for parameter MSE...")
    replica_curves = {}
    
    for c in c_values:
        replica_curves[c] = {}
        for rho in rho_values:
            alpha_vals, mse_vals = load_replica_curve(cache_dir, c, rho, ft_regulariser_scale)
            if alpha_vals is not None and mse_vals is not None:
                replica_curves[c][rho] = (alpha_vals, mse_vals)
                print(f"  c={c:.6f}, ρ={rho:.2f}: Loaded {len(alpha_vals)} points, MSE range: [{mse_vals.min():.6e}, {mse_vals.max():.6e}]")
            else:
                print(f"  WARNING: Replica curve not found for c={c:.6f}, ρ={rho:.2f}")
                print(f"    You may need to run: python scripts/diagonal/plot_replica_q_bg.py --c_values {c} --rho {rho} --ft_regulariser_scale {ft_regulariser_scale:.0e}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up color gradient: darker/more saturated for higher rho (denser)
    # c=0.001: shades of blue, c=0.5: shades of red/orange
    colormaps = {0.001: plt.cm.Blues, 0.5: plt.cm.Oranges}
    rho_normalizer = Normalize(vmin=min(rho_values), vmax=max(rho_values))
    
    for c in c_values:
        if not replica_curves[c]:
            continue
        
        # Sort rho values to ensure consistent plotting order
        available_rhos = sorted([rho for rho in rho_values if rho in replica_curves[c]])
        
        colormap = colormaps[c]
        
        for rho in available_rhos:
            alpha_vals, mse_vals = replica_curves[c][rho]
            mse_db = to_db(mse_vals)
            
            # Get color from colormap based on rho (normalized)
            # Use reversed colormap so darker = higher rho (denser)
            # Start from 0.3 to avoid very light colors, go to 0.95 for darker
            rho_norm = rho_normalizer(rho)
            # Map rho_norm from [0,1] to [0.3, 0.95] for better color range
            colormap_val = 0.3 + 0.65 * rho_norm
            base_color = colormap(colormap_val)
            
            # Use different line styles for different c values
            linestyle = '-' if c == 0.001 else '--'
            
            ax.plot(
                alpha_vals,
                mse_db,
                linestyle,
                label=f"Replica theory (c={c:.3f}, ρ={rho:.2f})",
                linewidth=2.5,
                color=base_color,
                alpha=0.9,
            )
    
    ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=14)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=14)
    ax.set_title(
        f"Replica Theory Curves: Parameter MSE\n"
        f"Bernoulli-Gaussian (varying $\\rho$), Noiseless, $\\ft_reg={ft_regulariser_scale:.0e}$",
        fontsize=14
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best', ncol=2)
    
    fig.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "replica_param_mse_curves_multi_rho.png")
    pdf_path = os.path.join(output_dir, "replica_param_mse_curves_multi_rho.pdf")
    
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nPlot saved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    
    # Print summary table
    print("\n" + "="*120)
    print("REPLICA THEORY PARAMETER MSE SUMMARY")
    print("="*120)
    
    # Sample at key alpha values
    alpha_sample = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Print tables for each c value
    for c in c_values:
        if not replica_curves[c]:
            continue
        
        print(f"\n{'='*120}")
        print(f"c = {c:.3f}")
        print(f"{'='*120}")
        
        # Create header with rho values
        header = f"{'Alpha':<10}"
        available_rhos = sorted([rho for rho in rho_values if rho in replica_curves[c]])
        for rho in available_rhos:
            header += f" {'ρ=' + str(rho):<15}"
        print(header)
        print("-"*120)
        
        for alpha in alpha_sample:
            row = f"{alpha:<10.4f}"
            for rho in available_rhos:
                alpha_vals, mse_vals = replica_curves[c][rho]
                idx = np.argmin(np.abs(alpha_vals - alpha))
                alpha_actual = alpha_vals[idx]
                mse = mse_vals[idx]
                mse_db = to_db(mse)
                row += f" {mse_db:<15.2f}"
            print(row)
    
    print("="*120)
    print("\nNote: These are PARAMETER MSE values (not prediction MSE).")
    print("The replica solver computes: MSE = mean((x_true - x_estimated)^2)")
    print("where x_true is the teacher signal and x_estimated is the student estimate.")
    print(f"\nColor gradient: Lighter colors = sparser (lower ρ), Darker colors = denser (higher ρ)")

if __name__ == "__main__":
    main()



