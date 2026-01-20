#!/usr/bin/env python3
"""
Aggregator and plotting script for Bernoulli-Gaussian teacher-student diagonal network experiments.

Runs multiple trials across different n_train values (alpha = n_train / inp_dim) and seeds,
then aggregates results and produces generalization curves.

NOTE: This script plots PARAMETER MSE (dB) to match replica theory outputs.
Replica theory computes parameter MSE, not test prediction MSE. Older aggregated CSV files
can be patched using scripts/diagonal/patch_aggregated_csvs_add_param_db.py to add the
required param_mse_*_db columns without rerunning experiments.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Constants
TRAIN_MSE_SUCCESS_THRESHOLD = 1e-10
C_DEFAULT = 0.001
C_TOLERANCE = 1e-6
DEFAULT_N_TEST = 10000
DEFAULT_LR = 0.5
DEFAULT_EPOCHS = 200000
DEFAULT_THRESHOLD = 1e-5
DEFAULT_TEST_EVERY_N_EPOCHS = 200
DEFAULT_SCALING = 1.0
DEFAULT_LMDA = 0.0
DEFAULT_INIT_METHOD = "complex"
DB_EPSILON = 1e-15  # Small value to prevent log(0) in dB conversion

def run_experiment(seed, n_train, inp_dim, rho, save_folder, **kwargs):
    """Run a single experiment via subprocess."""
    # Get c value
    c_val = kwargs.get("c", C_DEFAULT)
    
    # Default settings
    default_lr = kwargs.get("lr", DEFAULT_LR)
    default_threshold = kwargs.get("threshold", DEFAULT_THRESHOLD)
    default_epochs = kwargs.get("epochs", DEFAULT_EPOCHS)
    
    cmd = [
        "conda", "run", "-n", "mtl_ft", "python",
        "experiments/diagonal/diagonal_network_pretrain_bg.py",
        "--seed", str(seed),
        "--n_train", str(n_train),
        "--n_test", str(kwargs.get("n_test", DEFAULT_N_TEST)),
        "--inp_dim", str(inp_dim),
        "--rho", str(rho),
        "--save_folder", save_folder,
        "--lr", str(default_lr),
        "--epochs", str(default_epochs),
        "--threshold", str(default_threshold),
        "--test_every_n_epochs", str(kwargs.get("test_every_n_epochs", DEFAULT_TEST_EVERY_N_EPOCHS)),
        "--scaling", str(kwargs.get("scaling", DEFAULT_SCALING)),
        "--lmda", str(kwargs.get("lmda", DEFAULT_LMDA)),
        "--c", str(c_val),
        "--init_method", kwargs.get("init_method", DEFAULT_INIT_METHOD),
    ]
    
    if kwargs.get("no_tuning", False):
        cmd.append("--no_tuning")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)  # Don't capture output, show it in real-time
    return result

def extract_final_metrics(df_path):
    """Extract final test pred_mse, param_mse, and train pred_mse from df.feather."""
    df = pd.read_feather(df_path)
    
    # Get test split results
    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        return None, None, None
    
    # Get final epoch (max epoch)
    final_epoch = test_df["epoch"].max()
    final_test = test_df[test_df["epoch"] == final_epoch].iloc[0]
    
    final_test_pred_mse = final_test["pred_mse"]
    final_param_mse = final_test["param_mse"]
    
    # Get train split results
    train_df = df[df["split"] == "train"].copy()
    if train_df.empty:
        final_train_pred_mse = None
    else:
        # Use row with same epoch if present, otherwise use train row with max epoch
        train_at_final_epoch = train_df[train_df["epoch"] == final_epoch]
        if not train_at_final_epoch.empty:
            final_train_pred_mse = train_at_final_epoch.iloc[0]["pred_mse"]
        else:
            final_train_pred_mse = train_df.loc[train_df["epoch"].idxmax()]["pred_mse"]
    
    return final_test_pred_mse, final_param_mse, final_train_pred_mse

def check_success(train_pred_mse, test_pred_mse, param_mse):
    """Check if experiment run was successful."""
    return (
        train_pred_mse is not None
        and train_pred_mse < TRAIN_MSE_SUCCESS_THRESHOLD
        and np.isfinite(train_pred_mse)
        and np.isfinite(test_pred_mse)
        and np.isfinite(param_mse)
    )

def extract_result_from_path(df_path, alpha, n_train, seed):
    """Extract result metrics from a single experiment path."""
    try:
        test_pred_mse, param_mse, train_pred_mse = extract_final_metrics(df_path)
        if test_pred_mse is not None:
            is_success = check_success(train_pred_mse, test_pred_mse, param_mse)
            return {
                "alpha": alpha,
                "n_train": n_train,
                "seed": seed,
                "test_pred_mse": test_pred_mse,
                "param_mse": param_mse,
                "train_pred_mse": train_pred_mse if train_pred_mse is not None else np.nan,
                "is_success": is_success,
            }
    except Exception as e:
        print(f"Error reading {df_path}: {e}")
    return None

def aggregate_results(base_dir, n_train_list, inp_dim, seeds, rho, c=C_DEFAULT):
    """Run experiments and aggregate results."""
    import sys
    results = []
    
    print("Starting experiment aggregation...", flush=True)
    print(f"Total experiments to run: {len(n_train_list)} alpha values × {len(seeds)} seeds = {len(n_train_list) * len(seeds)}", flush=True)
    print(f"Using c = {c}", flush=True)
    
    for n_train in n_train_list:
        alpha = n_train / inp_dim
        print(f"\n{'='*80}", flush=True)
        print(f"Processing alpha = {alpha:.4f} (n_train = {n_train}, inp_dim = {inp_dim})", flush=True)
        print(f"{'='*80}", flush=True)
        
        for seed in seeds:
            save_folder = os.path.join(
                base_dir,
                f"alpha={alpha:.6f}--n_train={n_train}--seed={seed}--rho={rho:.6f}--c={c:.6f}/"
            )
            
            # Check if already exists
            df_path = os.path.join(save_folder, "df.feather")
            if os.path.exists(df_path):
                print(f"  Seed {seed}: Results already exist, skipping...", flush=True)
            else:
                print(f"  Seed {seed}: Running experiment...", flush=True)
                try:
                    # For c=C_DEFAULT, run_experiment will automatically use special settings
                    # For other c values, use defaults
                    run_experiment(
                        seed=seed,
                        n_train=n_train,
                        inp_dim=inp_dim,
                        rho=rho,
                        save_folder=save_folder,
                        n_test=DEFAULT_N_TEST,
                        lr=DEFAULT_LR if abs(c - C_DEFAULT) >= C_TOLERANCE else None,
                        epochs=DEFAULT_EPOCHS,
                        threshold=DEFAULT_THRESHOLD if abs(c - C_DEFAULT) >= C_TOLERANCE else None,
                        test_every_n_epochs=DEFAULT_TEST_EVERY_N_EPOCHS,
                        scaling=DEFAULT_SCALING,
                        lmda=DEFAULT_LMDA,
                        c=c,
                        init_method=DEFAULT_INIT_METHOD,
                        no_tuning=True,  # Disable LR tuning to ensure stable convergence
                    )
                except subprocess.CalledProcessError as e:
                    print(f"  Seed {seed}: ERROR - {e}", flush=True)
                    if hasattr(e, 'stderr') and e.stderr:
                        print(f"  stderr: {e.stderr}", flush=True)
                    continue
            
            # Extract metrics
            result = extract_result_from_path(df_path, alpha, n_train, seed)
            if result is not None:
                results.append(result)
                train_mse_str = f"{result['train_pred_mse']:.6e}" if not np.isnan(result['train_pred_mse']) else "nan"
                print(f"  Seed {seed}: test_pred_mse = {result['test_pred_mse']:.6e}, param_mse = {result['param_mse']:.6e}, train_pred_mse = {train_mse_str}, success = {result['is_success']}", flush=True)
            else:
                print(f"  Seed {seed}: ERROR reading results", flush=True)
    
    return pd.DataFrame(results)

def compute_aggregates(df, success_only=False):
    """Compute mean, median, and quantiles for each alpha."""
    if success_only:
        df_work = df[df["is_success"] == True].copy()
    else:
        df_work = df.copy()
    
    # Compute success counts per alpha
    success_counts = df.groupby("alpha")["is_success"].sum()
    total_counts = df.groupby("alpha").size()
    
    # Compute aggregates
    agg_df = df_work.groupby("alpha").agg({
        "test_pred_mse": [
            "mean",
            "median",
            lambda x: np.percentile(x, 25),
            lambda x: np.percentile(x, 75),
            "count",
        ],
        "param_mse": [
            "mean",
            "median",
            lambda x: np.percentile(x, 25),
            lambda x: np.percentile(x, 75),
        ],
        "train_pred_mse": [
            "median",
        ],
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = [
        "alpha",
        "test_pred_mse_mean",
        "test_pred_mse_median",
        "test_pred_mse_q25",
        "test_pred_mse_q75",
        "test_pred_mse_count",
        "param_mse_mean",
        "param_mse_median",
        "param_mse_q25",
        "param_mse_q75",
        "train_pred_mse_median",
    ]
    
    # Add success counts
    agg_df["success_count"] = agg_df["alpha"].map(success_counts).fillna(0).astype(int)
    agg_df["total_count"] = agg_df["alpha"].map(total_counts).fillna(0).astype(int)
    
    # Check for alphas with no successes (only warn/fill NaN if success_only=True)
    for _, row in agg_df.iterrows():
        if row["success_count"] == 0:
            print(f"  WARNING: alpha={row['alpha']:.6f} has 0 successful runs out of {row['total_count']} total runs")
            # Only fill with NaN if we're filtering for success_only
            if success_only:
                agg_df.loc[agg_df["alpha"] == row["alpha"], ["test_pred_mse_mean", "test_pred_mse_median", 
                                                              "test_pred_mse_q25", "test_pred_mse_q75",
                                                              "param_mse_mean", "param_mse_median",
                                                              "param_mse_q25", "param_mse_q75",
                                                              "train_pred_mse_median"]] = np.nan
    
    # Also compute max train_pred_mse from all runs (for debugging)
    max_train_pred_mse_all = df.groupby("alpha")["train_pred_mse"].max()
    agg_df["max_train_pred_mse_all"] = agg_df["alpha"].map(max_train_pred_mse_all)
    
    return agg_df

def plot_generalization(agg_df, output_dir, rho, c=C_DEFAULT):
    """Plot generalization curves."""
    # Convert PARAMETER MSE to dB: 10*log10(mse + 1e-15)
    # Note: We use parameter MSE to match replica theory outputs
    agg_df = agg_df.copy()
    
    # Check if param_mse_*_db columns already exist (from patched CSV)
    if "param_mse_mean_db" not in agg_df.columns:
        agg_df["param_mse_mean_db"] = 10 * np.log10(agg_df["param_mse_mean"] + DB_EPSILON)
    if "param_mse_median_db" not in agg_df.columns:
        agg_df["param_mse_median_db"] = 10 * np.log10(agg_df["param_mse_median"] + DB_EPSILON)
    if "param_mse_q25_db" not in agg_df.columns:
        agg_df["param_mse_q25_db"] = 10 * np.log10(agg_df["param_mse_q25"] + DB_EPSILON)
    if "param_mse_q75_db" not in agg_df.columns:
        agg_df["param_mse_q75_db"] = 10 * np.log10(agg_df["param_mse_q75"] + DB_EPSILON)
    
    agg_df = agg_df.sort_values("alpha")
    
    # Filter out rows with NaN values for plotting (using param_mse columns)
    valid_mask = (
        np.isfinite(agg_df["param_mse_mean"]) &
        np.isfinite(agg_df["param_mse_median"]) &
        np.isfinite(agg_df["param_mse_q25"]) &
        np.isfinite(agg_df["param_mse_q75"])
    )
    agg_df_valid = agg_df[valid_mask].copy()
    
    # Compute total success info for title
    total_success = agg_df["success_count"].sum()
    total_runs = agg_df["total_count"].sum()
    
    # Try to load cached replica curve
    replica_alpha = None
    replica_mse_db = None
    replica_cache_dir = os.path.join(output_dir, "replica_cache")
    # Default parameters for replica curve
    lambda_small = 1e-6
    alpha_min = 0.008
    alpha_max = 1.0
    alpha_points = 100
    mc_samples = 50000
    seed = 12345
    
    cache_filename = (
        f"replica_curve_rho={rho:.6f}--c={c:.6f}--"
        f"lambda={lambda_small:.6e}--alpha_min={alpha_min:.4f}--"
        f"alpha_max={alpha_max:.4f}--alpha_points={alpha_points}--"
        f"mc_samples={mc_samples}--seed={seed}.csv"
    )
    cache_path = os.path.join(replica_cache_dir, cache_filename)
    
    if os.path.exists(cache_path):
        try:
            df_replica = pd.read_csv(cache_path)
            replica_alpha = df_replica["alpha"].values
            replica_mse = df_replica["mse"].values
            replica_mse_db = 10 * np.log10(replica_mse + DB_EPSILON)
            print(f"\nLoaded replica theory curve from cache: {cache_path}")
            print(f"  Alpha range: [{replica_alpha.min():.4f}, {replica_alpha.max():.4f}]")
            print(f"  MSE range: [{replica_mse.min():.6e}, {replica_mse.max():.6e}]")
        except Exception as e:
            print(f"\nWARNING: Failed to load replica curve from cache: {e}")
    else:
        print(f"\nWARNING: Replica curve cache not found at: {cache_path}")
        print("  Run plot_replica_q_bg.py first to generate the replica curve.")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if len(agg_df_valid) > 0:
        # Plot mean curve (solid) - using parameter MSE
        ax.plot(
            agg_df_valid["alpha"],
            agg_df_valid["param_mse_mean_db"],
            "o-",
            label="Mean",
            linewidth=2,
            markersize=6,
            color="blue",
        )
        
        # Plot median curve (dashed) - using parameter MSE
        ax.plot(
            agg_df_valid["alpha"],
            agg_df_valid["param_mse_median_db"],
            "s--",
            label="Median",
            linewidth=2,
            markersize=6,
            color="red",
            alpha=0.7,
        )
        
        # Fill between 25-75% quantiles - using parameter MSE
        q_valid_mask = (
            np.isfinite(agg_df_valid["param_mse_q25_db"]) &
            np.isfinite(agg_df_valid["param_mse_q75_db"])
        )
        if q_valid_mask.sum() > 0:
            ax.fill_between(
                agg_df_valid.loc[q_valid_mask, "alpha"],
                agg_df_valid.loc[q_valid_mask, "param_mse_q25_db"],
                agg_df_valid.loc[q_valid_mask, "param_mse_q75_db"],
                alpha=0.2,
                color="blue",
                label="IQR (25-75%)",
            )
    else:
        print("WARNING: No valid data points to plot!")
    
    # Overlay replica theory curve if available
    if replica_alpha is not None and replica_mse_db is not None:
        ax.plot(
            replica_alpha,
            replica_mse_db,
            "-",
            label="Replica Theory",
            linewidth=2.5,
            color="orange",
            alpha=0.9,
        )
    
    ax.set_xlabel(r"$\alpha = n_{\text{train}} / d$", fontsize=12)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=12)
    ax.set_title(
        f"Generalization Curve (Bernoulli-Gaussian, $\\rho={rho:.3f}$, $c={c:.3f}$)\n"
        f"successes = {total_success} / {total_runs}",
        fontsize=14
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    fig.tight_layout()
    
    # Save as PNG and PDF
    png_path = os.path.join(output_dir, f"generalization_curve_rho={rho:.6f}--c={c:.6f}.png")
    pdf_path = os.path.join(output_dir, f"generalization_curve_rho={rho:.6f}--c={c:.6f}.pdf")
    
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nPlot saved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")

def print_summary_table(agg_df):
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Alpha':<12} {'Mean Test MSE':<20} {'Median Test MSE':<20} {'Count':<8}")
    print("-"*80)
    
    for _, row in agg_df.sort_values("alpha").iterrows():
        print(
            f"{row['alpha']:<12.6f} "
            f"{row['test_pred_mse_mean']:<20.6e} "
            f"{row['test_pred_mse_median']:<20.6e} "
            f"{int(row['test_pred_mse_count']):<8}"
        )
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Run and aggregate Bernoulli-Gaussian teacher-student experiments"
    )
    parser.add_argument(
        "--inp_dim",
        type=int,
        default=1000,
        help="Input dimension (fixed)",
    )
    parser.add_argument(
        "--n_train_list",
        type=int,
        nargs="+",
        default=[200, 300, 400, 600, 800, 1024, 1500, 2000, 3000],
        help="List of n_train values to sweep",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="List of seeds to run",
    )
    parser.add_argument(
        "--rho",
        type=float,
        required=True,
        help="Sparsity parameter for Bernoulli-Gaussian teacher",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=C_DEFAULT,
        help=f"C parameter for complex initialization (default: {C_DEFAULT})",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory for experiment results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for plots and CSV",
    )
    parser.add_argument(
        "--skip_runs",
        action="store_true",
        help="Skip running experiments, only aggregate existing results",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.base_dir).mkdir(parents=True, exist_ok=True)
    
    # Run experiments and aggregate
    if not args.skip_runs:
        print("Running experiments...")
        results_df = aggregate_results(
            args.base_dir,
            args.n_train_list,
            args.inp_dim,
            args.seeds,
            args.rho,
            c=args.c,
        )
    else:
        print("Skipping runs, aggregating existing results...")
        # Load existing results using shared extraction logic
        results_list = []
        for n_train in args.n_train_list:
            alpha = n_train / args.inp_dim
            for seed in args.seeds:
                save_folder = os.path.join(
                    args.base_dir,
                    f"alpha={alpha:.6f}--n_train={n_train}--seed={seed}--rho={args.rho:.6f}--c={args.c:.6f}/"
                )
                df_path = os.path.join(save_folder, "df.feather")
                if os.path.exists(df_path):
                    result = extract_result_from_path(df_path, alpha, n_train, seed)
                    if result is not None:
                        results_list.append(result)
        results_df = pd.DataFrame(results_list)
    
    if results_df.empty:
        print("ERROR: No results found!")
        return
    
    # Compute aggregates (successful runs only)
    agg_df_success = compute_aggregates(results_df, success_only=True)
    
    # Compute aggregates (all runs)
    agg_df_all = compute_aggregates(results_df, success_only=False)
    
    # Print summary
    print_summary_table(agg_df_success)
    
    # Print per-alpha success table
    print("\n" + "="*80)
    print("SUCCESS SUMMARY TABLE")
    print("="*80)
    print(f"{'Alpha':<12} {'Success':<10} {'Total':<10} {'Median Train MSE (success)':<30} {'Max Train MSE (all)':<30}")
    print("-"*80)
    
    for _, row in agg_df_success.sort_values("alpha").iterrows():
        print(
            f"{row['alpha']:<12.6f} "
            f"{int(row['success_count']):<10} "
            f"{int(row['total_count']):<10} "
            f"{row['train_pred_mse_median']:<30.6e} "
            f"{row['max_train_pred_mse_all']:<30.6e}"
        )
    print("="*80)
    
    # Save CSVs
    csv_path_success = os.path.join(args.output_dir, f"aggregated_results_rho={args.rho:.6f}--c={args.c:.6f}--SUCCESS.csv")
    csv_path_all = os.path.join(args.output_dir, f"aggregated_results_rho={args.rho:.6f}--c={args.c:.6f}--ALL.csv")
    
    agg_df_success.to_csv(csv_path_success, index=False)
    agg_df_all.to_csv(csv_path_all, index=False)
    
    print(f"\nAggregated results (successful runs) saved to: {csv_path_success}")
    print(f"Aggregated results (all runs) saved to: {csv_path_all}")
    
    # Also save raw results
    raw_csv_path = os.path.join(args.output_dir, f"raw_results_rho={args.rho:.6f}--c={args.c:.6f}.csv")
    results_df.to_csv(raw_csv_path, index=False)
    print(f"Raw results saved to: {raw_csv_path}")
    
    # Plot (use all data, not just successful runs)
    plot_generalization(agg_df_all, args.output_dir, args.rho, c=args.c)
    
    print("\nDone!")

if __name__ == "__main__":
    main()

