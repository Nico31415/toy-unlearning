#!/usr/bin/env python3
"""
Merge empirical curves for multiple c values into one figure.

NOTE: This script plots PARAMETER MSE (dB) to match replica theory outputs.
Replica theory computes parameter MSE, not test prediction MSE. CSV files should
be patched using scripts/diagonal/patch_aggregated_csvs_add_param_db.py to add
the required param_mse_*_db columns.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def to_db(mse_vals):
    """Convert MSE to dB."""
    return 10 * np.log10(mse_vals + 1e-15)

def plot_merged_curves(rho=0.04, c_values=[0.001, 0.5], output_dir='figures/diagonal/bg_generalization'):
    """Plot merged empirical and replica curves for multiple c values."""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Colors for different c values
    colors_emp = {0.001: 'blue', 0.5: 'green'}
    colors_replica = {0.001: 'orange', 0.5: 'red'}
    
    # Default parameters for replica curve
    ft_regulariser_scale = 1e-6
    alpha_min = 0.008
    alpha_max = 1.0
    alpha_points = 100
    mc_samples = 50000
    seed = 12345
    replica_cache_dir = os.path.join(output_dir, "replica_cache")
    
    # Load empirical results and replica curves for each c
    for c in c_values:
        c_str = f"{c:.6f}"
        
        # Load empirical results
        csv_path = os.path.join(output_dir, f"aggregated_results_rho={rho:.6f}--c={c:.6f}--ALL.csv")
        if not os.path.exists(csv_path):
            print(f"WARNING: Empirical results not found for c={c:.3f} at {csv_path}")
            continue
        
        df_emp = pd.read_csv(csv_path)
        print(f"\nLoaded empirical results for c={c:.3f}")
        print(f"  Alpha range: [{df_emp['alpha'].min():.4f}, {df_emp['alpha'].max():.4f}]")
        
        # Filter valid data (using param_mse columns)
        valid_mask = (
            df_emp["param_mse_mean"].notna() &
            df_emp["param_mse_median"].notna() &
            df_emp["param_mse_q25"].notna() &
            df_emp["param_mse_q75"].notna()
        )
        df_valid = df_emp[valid_mask].copy()
        df_valid = df_valid.sort_values("alpha")
        
        if len(df_valid) == 0:
            print(f"  WARNING: No valid data for c={c:.3f}, skipping")
            continue
        
        # Convert parameter MSE to dB (check if already exists from patched CSV)
        if "param_mse_mean_db" not in df_valid.columns:
            df_valid["param_mse_mean_db"] = to_db(df_valid["param_mse_mean"].values)
        if "param_mse_median_db" not in df_valid.columns:
            df_valid["param_mse_median_db"] = to_db(df_valid["param_mse_median"].values)
        if "param_mse_q25_db" not in df_valid.columns:
            df_valid["param_mse_q25_db"] = to_db(df_valid["param_mse_q25"].values)
        if "param_mse_q75_db" not in df_valid.columns:
            df_valid["param_mse_q75_db"] = to_db(df_valid["param_mse_q75"].values)
        
        # Plot empirical mean (using parameter MSE)
        ax.plot(
            df_valid["alpha"],
            df_valid["param_mse_mean_db"],
            "o-",
            label=f"Empirical (c={c:.3f}, mean)",
            linewidth=2,
            markersize=5,
            color=colors_emp[c],
            alpha=0.8,
        )
        
        # Plot empirical median (using parameter MSE)
        ax.plot(
            df_valid["alpha"],
            df_valid["param_mse_median_db"],
            "s--",
            label=f"Empirical (c={c:.3f}, median)",
            linewidth=2,
            markersize=4,
            color=colors_emp[c],
            alpha=0.6,
        )
        
        # Fill IQR (using parameter MSE)
        q_valid_mask = (
            df_valid["param_mse_q25_db"].notna() &
            df_valid["param_mse_q75_db"].notna()
        )
        if q_valid_mask.sum() > 0:
            ax.fill_between(
                df_valid.loc[q_valid_mask, "alpha"],
                df_valid.loc[q_valid_mask, "param_mse_q25_db"],
                df_valid.loc[q_valid_mask, "param_mse_q75_db"],
                alpha=0.15,
                color=colors_emp[c],
            )
        
        # Load replica curve
        cache_filename = (
            f"replica_curve_rho={rho:.6f}--c={c:.6f}--"
            f"lambda={ft_regulariser_scale:.6e}--alpha_min={alpha_min:.4f}--"
            f"alpha_max={alpha_max:.4f}--alpha_points={alpha_points}--"
            f"mc_samples={mc_samples}--seed={seed}.csv"
        )
        cache_path = os.path.join(replica_cache_dir, cache_filename)
        
        if os.path.exists(cache_path):
            try:
                df_replica = pd.read_csv(cache_path)
                replica_alpha = df_replica["alpha"].values
                replica_mse = df_replica["mse"].values
                replica_mse_db = to_db(replica_mse)
                
                # Plot replica theory curve
                ax.plot(
                    replica_alpha,
                    replica_mse_db,
                    "-",
                    label=f"Replica Theory (c={c:.3f})",
                    linewidth=2.5,
                    color=colors_replica[c],
                    alpha=0.9,
                )
                print(f"  Loaded replica curve for c={c:.3f}")
            except Exception as e:
                print(f"  WARNING: Failed to load replica curve for c={c:.3f}: {e}")
        else:
            print(f"  WARNING: Replica curve cache not found for c={c:.3f} at {cache_path}")
    
    ax.set_xlabel(r"$\alpha = n_{\text{train}} / d$", fontsize=14)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=14)
    ax.set_title(
        f"Generalization Curves: Empirical vs Replica Theory\n"
        f"Bernoulli-Gaussian, $\\rho={rho:.3f}$",
        fontsize=14
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    fig.tight_layout()
    
    # Save
    png_path = os.path.join(output_dir, f"generalization_curve_merged_rho={rho:.6f}.png")
    pdf_path = os.path.join(output_dir, f"generalization_curve_merged_rho={rho:.6f}.pdf")
    
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nMerged plot saved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, default=0.04)
    parser.add_argument("--c_values", type=float, nargs="+", default=[0.001, 0.5])
    parser.add_argument("--output_dir", type=str, default="figures/diagonal/bg_generalization")
    args = parser.parse_args()
    
    plot_merged_curves(args.rho, args.c_values, args.output_dir)

