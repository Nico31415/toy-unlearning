#!/usr/bin/env python3
"""
Plot replica theory curves for parameter MSE for multiple c values.

This script loads cached replica curves for different c values (all with the same ft_regulariser_scale)
and plots them together on the same figure.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def to_db(x):
    """Convert MSE to dB: 10*log10(mse + 1e-15)"""
    return 10.0 * np.log10(np.maximum(x, 1e-15))

def load_replica_curve(cache_dir, c, rho=0.04, ft_regulariser_scale=1e-6, 
                       alpha_min=0.008, alpha_max=1.0, alpha_points=100,
                       mc_samples=50000, seed=12345):
    """Load replica curve from cache."""
    cache_filename = (
        f"replica_curve_rho={rho:.6f}--c={c:.6f}--"
        f"ft_reg={ft_regulariser_scale:.6e}--alpha_min={alpha_min:.4f}--"
        f"alpha_max={alpha_max:.4f}--alpha_points={alpha_points}--"
        f"mc_samples={mc_samples}--seed={seed}.csv"
    )
    cache_path = os.path.join(cache_dir, cache_filename)
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Replica curve not found: {cache_path}")
    
    df = pd.read_csv(cache_path)
    return df["alpha"].values, df["mse"].values

def main():
    # Parameters
    cache_dir = "figures/diagonal/bg_generalization/replica_cache"
    output_dir = "figures/diagonal/bg_generalization"
    c_values = [0.001, 0.01, 0.5]
    rho = 0.04
    ft_regulariser_scale = 1e-6
    
    # Load replica curves
    print("Loading replica theory curves for parameter MSE...")
    replica_curves = {}
    
    for c in c_values:
        try:
            alpha_vals, mse_vals = load_replica_curve(cache_dir, c, rho, ft_regulariser_scale)
            replica_curves[c] = (alpha_vals, mse_vals)
            print(f"  c={c:.6f}: Loaded {len(alpha_vals)} points, MSE range: [{mse_vals.min():.6e}, {mse_vals.max():.6e}]")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            print(f"  You may need to run: python scripts/diagonal/plot_replica_q_bg.py --c_values {c} --ft_regulariser_scale {ft_regulariser_scale:.0e}")
            continue
    
    if not replica_curves:
        print("No replica curves loaded. Exiting.")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {0.001: 'red', 0.01: 'blue', 0.5: 'orange'}
    linestyles = {0.001: '-', 0.01: '--', 0.5: '-.'}
    
    for c, (alpha_vals, mse_vals) in replica_curves.items():
        mse_db = to_db(mse_vals)
        
        ax.plot(
            alpha_vals,
            mse_db,
            linestyles[c],
            label=f"Replica theory (c={c:.3f}, param MSE)",
            linewidth=2.5,
            color=colors[c],
            alpha=0.9,
        )
    
    ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=14)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=14)
    ax.set_title(
        f"Replica Theory Curves: Parameter MSE\n"
        f"Bernoulli-Gaussian ($\\rho={rho:.3f}$), Noiseless, $\\ft_reg={ft_regulariser_scale:.0e}$",
        fontsize=14
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    
    fig.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    lambda_tag = f"{ft_regulariser_scale:.0e}".replace("+", "")
    png_path = os.path.join(output_dir, f"replica_param_mse_multi_c_ft_reg={lambda_tag}.png")
    pdf_path = os.path.join(output_dir, f"replica_param_mse_multi_c_ft_reg={lambda_tag}.pdf")
    
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nPlot saved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    
    # Print summary table
    print("\n" + "="*90)
    print("REPLICA THEORY PARAMETER MSE SUMMARY")
    print("="*90)
    header = f"{'Alpha':<10}"
    for c in c_values:
        header += f" {'c=' + str(c) + ' (dB)':<20}"
    print(header)
    print("-"*90)
    
    # Sample at key alpha values
    alpha_sample = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for alpha in alpha_sample:
        row = f"{alpha:<10.4f}"
        for c in c_values:
            if c in replica_curves:
                # Find closest alpha in curve
                idx = np.argmin(np.abs(replica_curves[c][0] - alpha))
                alpha_actual = replica_curves[c][0][idx]
                mse = replica_curves[c][1][idx]
                mse_db = to_db(mse)
                row += f" {mse_db:<20.2f}"
            else:
                row += f" {'N/A':<20}"
        print(row)
    
    print("="*90)
    print(f"\nNote: All curves use ft_regulariser_scale = {ft_regulariser_scale:.0e}")
    print("These are PARAMETER MSE values (not prediction MSE).")
    print("The replica solver computes: MSE = mean((x_true - x_estimated)^2)")
    print("where x_true is the teacher signal and x_estimated is the student estimate.")

if __name__ == "__main__":
    main()



