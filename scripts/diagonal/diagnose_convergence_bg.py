#!/usr/bin/env python3
"""
Diagnostic script to test convergence for c=0.001 experiments.

Runs experiments with different learning rates and thresholds to diagnose
why the empirical curve plateaus at high alpha values.
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
import json

def run_experiment(seed, n_train, inp_dim, rho, c, save_folder, lr, threshold, epochs, test_every_n_epochs, stop_pred_mse=None, stop_beta_rate=0.0, stop_grad_norm=0.0, lr_decay=1.0, lr_decay_interval=2000, **kwargs):
    """Run a single experiment via subprocess."""
    cmd = [
        "conda", "run", "-n", "mtl_ft", "python",
        "experiments/diagonal/diagonal_network_pretrain_bg.py",
        "--seed", str(seed),
        "--n_train", str(n_train),
        "--n_test", str(10000),
        "--inp_dim", str(inp_dim),
        "--rho", str(rho),
        "--c", str(c),
        "--save_folder", save_folder,
        "--lr", str(lr),
        "--epochs", str(epochs),
        "--threshold", str(threshold),
        "--test_every_n_epochs", str(test_every_n_epochs),
        "--scaling", str(1.0),
        "--lmda", str(0.0),
        "--init_method", "complex",
        "--no_tuning",
    ]
    
    if stop_pred_mse is not None:
        cmd.extend(["--stop_pred_mse", str(stop_pred_mse)])
    if stop_beta_rate > 0:
        cmd.extend(["--stop_beta_rate", str(stop_beta_rate)])
    if stop_grad_norm > 0:
        cmd.extend(["--stop_grad_norm", str(stop_grad_norm)])
    if lr_decay < 1.0:
        cmd.extend(["--lr_decay", str(lr_decay), "--lr_decay_interval", str(lr_decay_interval)])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result

def extract_metrics(df_path, meta_path):
    """Extract final metrics from df.feather and results_meta.json."""
    # First try to load metadata (doesn't require pyarrow)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    
    # If we have metadata with all needed fields, use that (avoids pyarrow dependency)
    if meta and all(k in meta for k in ["final_train_pred_mse", "final_param_mse", "final_epoch", "stop_reason"]):
        result = {
            "final_train_pred_mse": meta["final_train_pred_mse"],
            "final_test_pred_mse": meta.get("final_test_pred_mse", np.nan),
            "final_param_mse": meta["final_param_mse"],
            "final_epoch": meta["final_epoch"],
            "stop_reason": meta["stop_reason"],
            "final_grad_norm": meta.get("final_grad_norm", np.nan),
            "final_beta_update_rate": meta.get("final_beta_update_rate", np.nan),
        }
        return result
    
    # Fallback: try to read feather file
    try:
        df = pd.read_feather(df_path)
    except Exception as e:
        # Fallback: try reading as CSV if feather fails
        if "pyarrow" in str(e).lower() or "arrow" in str(e).lower():
            print(f"  WARNING: pyarrow not available, trying CSV fallback...")
            csv_path = df_path.replace('.feather', '.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                # If we have partial metadata, use what we can
                if meta:
                    print(f"  Using partial metadata (pyarrow missing)...")
                    result = {
                        "final_train_pred_mse": meta.get("final_train_pred_mse", np.nan),
                        "final_test_pred_mse": meta.get("final_test_pred_mse", np.nan),
                        "final_param_mse": meta.get("final_param_mse", np.nan),
                        "final_epoch": meta.get("final_epoch", np.nan),
                        "stop_reason": meta.get("stop_reason", "unknown"),
                        "final_grad_norm": meta.get("final_grad_norm", np.nan),
                        "final_beta_update_rate": meta.get("final_beta_update_rate", np.nan),
                    }
                    return result
                else:
                    raise RuntimeError(f"Cannot read {df_path}: pyarrow missing. Install with: conda install -n mtl_ft pyarrow")
        else:
            raise
    
    # Get test split results
    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        return None
    
    # Get final epoch (max epoch)
    final_epoch = test_df["epoch"].max()
    final_test = test_df[test_df["epoch"] == final_epoch].iloc[0]
    
    # Get train split results
    train_df = df[df["split"] == "train"].copy()
    final_train = train_df[train_df["epoch"] == final_epoch].iloc[0] if not train_df.empty else None
    
    result = {
        "final_train_pred_mse": final_train["pred_mse"] if final_train is not None else np.nan,
        "final_test_pred_mse": final_test["pred_mse"],
        "final_param_mse": final_test["param_mse"],
        "final_epoch": int(final_epoch),
        "stop_reason": meta.get("stop_reason", "unknown"),
    }
    
    # Add diagnostic metrics if available
    if "grad_norm" in final_test:
        result["final_grad_norm"] = final_test["grad_norm"]
    else:
        result["final_grad_norm"] = np.nan
    
    if "beta_update_rate" in final_test:
        result["final_beta_update_rate"] = final_test["beta_update_rate"]
    else:
        result["final_beta_update_rate"] = np.nan
    
    return result

def run_diagnostics(base_dir, alphas, seeds, inp_dim, rho, c, lr_list, threshold_list, epochs, test_every_n_epochs, stop_pred_mse=None, stop_beta_rate=0.0, stop_grad_norm=0.0, lr_decay=1.0, lr_decay_interval=2000):
    """Run diagnostic experiments and collect results."""
    results = []
    
    print("="*80)
    print("DIAGNOSTIC EXPERIMENTS")
    print("="*80)
    print(f"Alphas: {alphas}")
    print(f"Seeds: {seeds}")
    print(f"LRs: {lr_list}")
    print(f"Thresholds: {threshold_list}")
    print(f"Total experiments: {len(alphas)} × {len(seeds)} × {len(lr_list)} × {len(threshold_list)} = {len(alphas) * len(seeds) * len(lr_list) * len(threshold_list)}")
    print("="*80)
    
    for alpha in alphas:
        n_train = int(alpha * inp_dim)
        print(f"\n{'='*80}")
        print(f"Alpha = {alpha:.4f} (n_train = {n_train})")
        print(f"{'='*80}")
        
        for seed in seeds:
            for lr in lr_list:
                for threshold in threshold_list:
                    
                    # Create save folder with all parameters
                    # Include stopping criteria in folder name if they're non-default
                    folder_suffix = ""
                    if stop_pred_mse is not None and abs(stop_pred_mse - threshold) > 1e-10:
                        folder_suffix += f"--stop_pred={stop_pred_mse:.6e}"
                    if stop_beta_rate > 0:
                        folder_suffix += f"--stop_beta={stop_beta_rate:.6e}"
                    if lr_decay < 1.0:
                        folder_suffix += f"--lrdecay={lr_decay:.3f}--lrdecayint={lr_decay_interval}"
                    
                    save_folder = os.path.join(
                        base_dir,
                        f"alpha={alpha:.6f}--n_train={n_train}--seed={seed}--rho={rho:.6f}--c={c:.6f}--lr={lr:.6f}--thr={threshold:.6e}{folder_suffix}/"
                    )
                    
                    df_path = os.path.join(save_folder, "df.feather")
                    meta_path = os.path.join(save_folder, "results_meta.json")
                    
                    # Check if already exists
                    if os.path.exists(df_path):
                        print(f"  Alpha={alpha:.4f}, seed={seed}, lr={lr:.6f}, thr={threshold:.6e}: Results exist, skipping...")
                    else:
                        print(f"  Alpha={alpha:.4f}, seed={seed}, lr={lr:.6f}, thr={threshold:.6e}: Running...")
                        try:
                            # Use stop_pred_mse if provided, otherwise use threshold
                            effective_stop_pred_mse = stop_pred_mse if stop_pred_mse is not None else threshold
                            run_experiment(
                                seed=seed,
                                n_train=n_train,
                                inp_dim=inp_dim,
                                rho=rho,
                                c=c,
                                save_folder=save_folder,
                                lr=lr,
                                threshold=threshold,
                                epochs=epochs,
                                test_every_n_epochs=test_every_n_epochs,
                                stop_pred_mse=effective_stop_pred_mse,
                                stop_beta_rate=stop_beta_rate,
                                stop_grad_norm=stop_grad_norm,
                                lr_decay=lr_decay,
                                lr_decay_interval=lr_decay_interval,
                            )
                        except subprocess.CalledProcessError as e:
                            print(f"  ERROR: {e}")
                            continue
                    
                    # Extract metrics
                    try:
                        metrics = extract_metrics(df_path, meta_path)
                        if metrics is not None:
                            metrics.update({
                                "alpha": alpha,
                                "seed": seed,
                                "lr": lr,
                                "threshold": threshold,
                            })
                            results.append(metrics)
                            print(f"    Final train_pred_mse = {metrics['final_train_pred_mse']:.6e}, param_mse = {metrics['final_param_mse']:.6e}, stop_reason = {metrics['stop_reason']}")
                    except Exception as e:
                        print(f"  ERROR reading results: {e}")
                        continue
    
    return pd.DataFrame(results)

def plot_diagnostics(df, output_dir, rho, c, base_dir="results/diagonal/bg_experiments", inp_dim=1000):
    """Create diagnostic plots for each alpha."""
    output_plot_dir = os.path.join(output_dir, "diagnostics_plots")
    Path(output_plot_dir).mkdir(parents=True, exist_ok=True)
    
    # Store for use in inner function
    plot_diagnostics._base_dir = base_dir
    plot_diagnostics._inp_dim = inp_dim
    
    alphas = sorted(df["alpha"].unique())
    
    for alpha in alphas:
        df_alpha = df[df["alpha"] == alpha].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Convergence Diagnostics: α = {alpha:.4f}, ρ = {rho:.3f}, c = {c:.3f}", fontsize=14)
        
        # Get unique (lr, threshold) combinations for colors
        df_alpha["combo"] = df_alpha.apply(lambda row: f"lr={row['lr']:.6f}, thr={row['threshold']:.6e}", axis=1)
        combos = sorted(df_alpha["combo"].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(combos)))
        combo_colors = dict(zip(combos, colors))
        
        # Load full training history for plotting
        # Need to get base_dir from args or use default
        base_dir = getattr(plot_diagnostics, '_base_dir', 'results/diagonal/bg_experiments')
        inp_dim = getattr(plot_diagnostics, '_inp_dim', 1000)
        
        for idx, row in df_alpha.iterrows():
            n_train = int(row['alpha'] * inp_dim)
            save_folder = os.path.join(
                base_dir,
                f"alpha={row['alpha']:.6f}--n_train={n_train}--seed={int(row['seed'])}--rho={rho:.6f}--c={c:.6f}--lr={row['lr']:.6f}--thr={row['threshold']:.6e}/"
            )
            df_path = os.path.join(save_folder, "df.feather")
            
            if not os.path.exists(df_path):
                continue
            
            try:
                df_full = pd.read_feather(df_path)
                df_test = df_full[df_full["split"] == "test"].copy()
                combo = row["combo"]
                color = combo_colors[combo]
                
                # Plot 1: param_mse vs epoch
                axes[0, 0].plot(df_test["epoch"], df_test["param_mse"], color=color, alpha=0.7, label=combo if idx == df_alpha.index[0] or combo not in [df_alpha.loc[i, "combo"] for i in df_alpha.index[:idx]] else "")
                axes[0, 0].set_xlabel("Epoch")
                axes[0, 0].set_ylabel("Parameter MSE")
                axes[0, 0].set_yscale("log")
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_title("Parameter MSE vs Epoch")
                
                # Plot 2: train_pred_mse vs epoch
                df_train = df_full[df_full["split"] == "train"].copy()
                axes[0, 1].plot(df_train["epoch"], df_train["pred_mse"], color=color, alpha=0.7, label=combo if idx == df_alpha.index[0] or combo not in [df_alpha.loc[i, "combo"] for i in df_alpha.index[:idx]] else "")
                axes[0, 1].set_xlabel("Epoch")
                axes[0, 1].set_ylabel("Train Prediction MSE")
                axes[0, 1].set_yscale("log")
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].set_title("Train Prediction MSE vs Epoch")
                
                # Plot 3: grad_norm vs epoch
                if "grad_norm" in df_test.columns:
                    axes[1, 0].plot(df_test["epoch"], df_test["grad_norm"], color=color, alpha=0.7, label=combo if idx == df_alpha.index[0] or combo not in [df_alpha.loc[i, "combo"] for i in df_alpha.index[:idx]] else "")
                    axes[1, 0].set_xlabel("Epoch")
                    axes[1, 0].set_ylabel("Gradient Norm")
                    axes[1, 0].set_yscale("log")
                    axes[1, 0].grid(True, alpha=0.3)
                    axes[1, 0].set_title("Gradient Norm vs Epoch")
                
                # Plot 4: beta_update_rate vs epoch
                if "beta_update_rate" in df_test.columns:
                    axes[1, 1].plot(df_test["epoch"], df_test["beta_update_rate"], color=color, alpha=0.7, label=combo if idx == df_alpha.index[0] or combo not in [df_alpha.loc[i, "combo"] for i in df_alpha.index[:idx]] else "")
                    axes[1, 1].set_xlabel("Epoch")
                    axes[1, 1].set_ylabel("Beta Update Rate")
                    axes[1, 1].set_yscale("log")
                    axes[1, 1].grid(True, alpha=0.3)
                    axes[1, 1].set_title("Beta Update Rate vs Epoch")
            except Exception as e:
                print(f"  WARNING: Failed to load data for plotting: {e}")
                continue
        
        # Add legend to first subplot only (to avoid clutter)
        axes[0, 0].legend(fontsize=8, loc='best', ncol=1)
        
        fig.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_plot_dir, f"diagnostics_alpha={alpha:.6f}--rho={rho:.6f}--c={c:.6f}.png")
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        
        print(f"  Saved diagnostic plot: {plot_path}")

def analyze_and_recommend(df):
    """Analyze results and provide recommendation."""
    print("\n" + "="*80)
    print("DIAGNOSTIC ANALYSIS")
    print("="*80)
    
    # Check stop reasons
    stop_reasons = df["stop_reason"].value_counts()
    print(f"\nStop reasons:")
    for reason, count in stop_reasons.items():
        print(f"  {reason}: {count}")
    
    # Check if max_epochs was hit frequently
    max_epochs_count = (df["stop_reason"] == "max_epochs").sum()
    if max_epochs_count > len(df) * 0.5:
        print(f"\n⚠️  WARNING: {max_epochs_count}/{len(df)} runs hit max_epochs")
        print("   → Plateau likely due to max epochs reached")
        return "Plateau due to max epochs"
    
    # Check final grad_norm
    if "final_grad_norm" in df.columns:
        high_grad_norm = (df["final_grad_norm"] > 1e-6).sum()
        if high_grad_norm > len(df) * 0.5:
            print(f"\n⚠️  WARNING: {high_grad_norm}/{len(df)} runs have high final grad_norm (>1e-6)")
            print("   → Training not converged (gradients still large)")
            return "Plateau due to lr discretization (gradients not small)"
    
    # Check beta_update_rate
    if "final_beta_update_rate" in df.columns:
        high_beta_rate = (df["final_beta_update_rate"] > 1e-8).sum()
        if high_beta_rate > len(df) * 0.5:
            print(f"\n⚠️  WARNING: {high_beta_rate}/{len(df)} runs have high final beta_update_rate (>1e-8)")
            print("   → Beta still moving (product parametrization slow convergence)")
            return "Plateau due to beta stagnation in product parametrization"
    
    # Check if threshold stopping happened but param_mse is still high
    threshold_stops = df[df["stop_reason"] == "threshold"]
    if len(threshold_stops) > 0:
        high_param_mse = (threshold_stops["final_param_mse"] > 1e-3).sum()
        if high_param_mse > len(threshold_stops) * 0.5:
            print(f"\n⚠️  WARNING: {high_param_mse}/{len(threshold_stops)} threshold-stopped runs have high param_mse (>1e-3)")
            print("   → Early stopping threshold too loose")
            return "Plateau due to early stopping (threshold too loose)"
    
    print("\n✓ No clear issue identified. Check plots for details.")
    return "Plateau cause unclear - check diagnostic plots"

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose convergence issues for c=0.001 experiments"
    )
    parser.add_argument("--rho", type=float, required=True, help="Sparsity parameter")
    parser.add_argument("--c", type=float, default=0.001, help="C parameter (default: 0.001)")
    parser.add_argument("--inp_dim", type=int, default=1000, help="Input dimension (default: 1000)")
    parser.add_argument("--n_test", type=int, default=10000, help="Test samples (default: 10000)")
    parser.add_argument("--alphas", type=float, nargs="+", required=True, help="Alpha values to test (e.g., 0.5 0.7 1.0)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds to run (default: 0 1 2)")
    parser.add_argument("--base_dir", type=str, default="results/diagonal/bg_experiments", help="Base directory for results")
    parser.add_argument("--lr_list", type=float, nargs="+", default=[0.5, 0.1, 0.02, 0.005], help="Learning rates to test")
    parser.add_argument("--threshold_list", type=float, nargs="+", default=[1e-5, 1e-8, 1e-12], help="Thresholds to test")
    parser.add_argument("--epochs", type=int, default=200000, help="Max epochs (default: 200000)")
    parser.add_argument("--test_every_n_epochs", type=int, default=200, help="Test every N epochs (default: 200)")
    parser.add_argument("--output_dir", type=str, default="figures/diagonal/bg_generalization", help="Output directory for plots and CSV")
    parser.add_argument("--stop_pred_mse", type=float, default=None, help="Stop when train_pred_mse < this value (default: None, uses threshold)")
    parser.add_argument("--stop_beta_rate", type=float, default=0.0, help="Stop beta rate threshold (default: 0.0, disabled)")
    parser.add_argument("--stop_grad_norm", type=float, default=0.0, help="Stop grad norm threshold (default: 0.0, disabled)")
    parser.add_argument("--lr_decay", type=float, default=1.0, help="LR decay factor (default: 1.0, no decay)")
    parser.add_argument("--lr_decay_interval", type=int, default=2000, help="LR decay interval (default: 2000)")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.base_dir).mkdir(parents=True, exist_ok=True)
    
    # Run diagnostics
    df = run_diagnostics(
        base_dir=args.base_dir,
        alphas=args.alphas,
        seeds=args.seeds,
        inp_dim=args.inp_dim,
        rho=args.rho,
        c=args.c,
        lr_list=args.lr_list,
        threshold_list=args.threshold_list,
        epochs=args.epochs,
        test_every_n_epochs=args.test_every_n_epochs,
        stop_pred_mse=args.stop_pred_mse,
        stop_beta_rate=args.stop_beta_rate,
        stop_grad_norm=args.stop_grad_norm,
        lr_decay=args.lr_decay,
        lr_decay_interval=args.lr_decay_interval,
    )
    
    if df.empty:
        print("ERROR: No results collected!")
        return
    
    # Save summary CSV
    csv_path = os.path.join(args.output_dir, f"diagnostics_rho={args.rho:.6f}--c={args.c:.6f}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved to: {csv_path}")
    
    # Create diagnostic plots
    print("\nCreating diagnostic plots...")
    plot_diagnostics(df, args.output_dir, args.rho, args.c, base_dir=args.base_dir, inp_dim=args.inp_dim)
    
    # Analyze and recommend
    recommendation = analyze_and_recommend(df)
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION: {recommendation}")
    print(f"{'='*80}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()

