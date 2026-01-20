#!/usr/bin/env python3
"""
Diagonal Net Post-hoc Debug Plots.

Diagnoses premature stopping, shows why error bars explode, and verifies
that filtering produces theory-matching curves using ONLY existing CSV + 
results_meta.json files (NO RERUNS).

Usage:
    python scripts/diagonal/posthoc_debug_plots.py
    python scripts/diagonal/posthoc_debug_plots.py --outdir figures/posthoc_debug
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# CLI Argument Parsing
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagonal Net Post-hoc Debug Plots"
    )
    parser.add_argument(
        "--csv_step1",
        type=str,
        default="experiment_results_step1_mixture.csv",
        help="Path to Step 1 CSV file"
    )
    parser.add_argument(
        "--csv_step2",
        type=str,
        default="experiment_results_step2_support.csv",
        help="Path to Step 2 CSV file"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="figures/posthoc_debug",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--test_every",
        type=int,
        default=200,
        help="Test frequency for premature stopping check"
    )
    parser.add_argument(
        "--beta_rate_cutoffs",
        type=str,
        default="1e-4,1e-5,1e-6,1e-7",
        help="Comma-separated beta rate cutoffs"
    )
    parser.add_argument(
        "--alpha_focus",
        type=str,
        default="0.05",
        help="Comma-separated alpha values to focus on"
    )
    parser.add_argument(
        "--alpha_tol",
        type=float,
        default=0.0005,
        help="Tolerance for alpha matching"
    )
    
    args = parser.parse_args()
    
    # Parse comma-separated floats
    args.beta_rate_cutoffs = [float(x) for x in args.beta_rate_cutoffs.split(",")]
    args.alpha_focus = [float(x) for x in args.alpha_focus.split(",")]
    
    return args


# ==============================================================================
# Data Loading
# ==============================================================================

def load_csv_with_meta(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and enrich with results_meta.json data.
    
    For each row, attempts to load save_folder/results_meta.json and adds:
    - final_beta_update_rate
    - final_grad_norm
    - final_train_pred_mse
    - final_test_pred_mse
    - final_param_mse
    - stop_reason
    - final_epoch
    """
    if not os.path.exists(csv_path):
        print(f"  WARNING: CSV not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows from {csv_path}")
    
    # Initialize meta columns with NaN
    meta_cols = [
        "meta_final_beta_update_rate",
        "meta_final_grad_norm",
        "meta_final_train_pred_mse",
        "meta_final_test_pred_mse",
        "meta_final_param_mse",
        "meta_stop_reason",
        "meta_final_epoch",
    ]
    for col in meta_cols:
        df[col] = np.nan
    df["meta_stop_reason"] = df["meta_stop_reason"].astype(object)
    
    # Load results_meta.json for each row
    loaded_count = 0
    for idx, row in df.iterrows():
        save_folder = row.get("save_folder", "")
        if pd.isna(save_folder) or not save_folder:
            continue
        
        meta_path = os.path.join(save_folder, "results_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                
                df.at[idx, "meta_final_beta_update_rate"] = meta.get("final_beta_update_rate", np.nan)
                df.at[idx, "meta_final_grad_norm"] = meta.get("final_grad_norm", np.nan)
                df.at[idx, "meta_final_train_pred_mse"] = meta.get("final_train_pred_mse", np.nan)
                df.at[idx, "meta_final_test_pred_mse"] = meta.get("final_test_pred_mse", np.nan)
                df.at[idx, "meta_final_param_mse"] = meta.get("final_param_mse", np.nan)
                df.at[idx, "meta_stop_reason"] = meta.get("stop_reason", np.nan)
                df.at[idx, "meta_final_epoch"] = meta.get("final_epoch", np.nan)
                loaded_count += 1
            except (json.JSONDecodeError, IOError):
                pass
    
    print(f"  Enriched {loaded_count}/{len(df)} rows with results_meta.json data")
    
    return df


# ==============================================================================
# Filter Definitions
# ==============================================================================

def create_filters(df: pd.DataFrame, test_every: int, beta_rate_cutoffs: list) -> dict:
    """
    Create boolean masks for filtering.
    
    Returns dict of filter_name -> boolean mask
    """
    filters = {}
    
    # ALL: no filtering
    filters["ALL"] = np.ones(len(df), dtype=bool)
    
    # Get stop_reason - prefer meta version if available, otherwise CSV
    stop_reason = df["meta_stop_reason"].fillna(df.get("stop_reason", ""))
    final_epoch = df["meta_final_epoch"].fillna(df.get("final_epoch", np.inf))
    
    # NO_PREMATURE_LEGACY: NOT (stop_reason == "loss_threshold_legacy" AND final_epoch < test_every)
    # Also handle "threshold" as legacy
    is_legacy = (stop_reason == "loss_threshold_legacy") | (stop_reason == "threshold")
    is_premature = final_epoch < test_every
    filters["NO_PREMATURE_LEGACY"] = ~(is_legacy & is_premature)
    
    # LONG_TIME_ONLY: stop_reason == "max_epochs" OR stop_reason starts with "train_pred_mse"
    is_max_epochs = stop_reason == "max_epochs"
    starts_with_train_pred = stop_reason.astype(str).str.startswith("train_pred_mse")
    filters["LONG_TIME_ONLY"] = is_max_epochs | starts_with_train_pred
    
    # FIXED_POINT_betaRate_<cutoff>: LONG_TIME_ONLY AND finite AND < cutoff
    beta_rate = df["meta_final_beta_update_rate"]
    is_finite = np.isfinite(beta_rate)
    
    for cutoff in beta_rate_cutoffs:
        filter_name = f"FIXED_POINT_betaRate_{cutoff:.0e}"
        filters[filter_name] = (
            filters["LONG_TIME_ONLY"] & 
            is_finite & 
            (beta_rate < cutoff)
        )
    
    # Ensure FIXED_POINT_betaRate_1e-6 exists
    if "FIXED_POINT_betaRate_1e-06" not in filters:
        cutoff = 1e-6
        filter_name = "FIXED_POINT_betaRate_1e-06"
        filters[filter_name] = (
            filters["LONG_TIME_ONLY"] & 
            is_finite & 
            (beta_rate < cutoff)
        )
    
    return filters


# ==============================================================================
# Utility Functions
# ==============================================================================

def to_db(x: np.ndarray) -> np.ndarray:
    """Convert MSE to dB: 10*log10(mse + 1e-15)"""
    return 10.0 * np.log10(np.maximum(x, 1e-15))


def save_figure(fig, outdir: str, basename: str):
    """Save figure as both PNG and PDF."""
    png_path = os.path.join(outdir, f"{basename}.png")
    pdf_path = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def get_stop_reason_order():
    """Fixed order for stop reason stacking."""
    return [
        "loss_threshold_legacy",
        "max_epochs",
        "train_pred_mse",
        "threshold",
        "other",
    ]


def categorize_stop_reason(reason: str) -> str:
    """Categorize stop reason into known buckets."""
    if pd.isna(reason) or reason == "":
        return "other"
    reason = str(reason)
    if reason == "loss_threshold_legacy":
        return "loss_threshold_legacy"
    if reason == "max_epochs":
        return "max_epochs"
    if reason.startswith("train_pred_mse"):
        return "train_pred_mse"
    if reason == "threshold":
        return "threshold"
    return "other"


# ==============================================================================
# P1: Stop Reason Composition
# ==============================================================================

def plot_p1_stop_reason(
    df: pd.DataFrame, 
    step_name: str,
    group_col: str,
    alpha_focus: list,
    alpha_tol: float,
    outdir: str
):
    """
    P1: Stacked bar chart of stop reason composition for low alpha.
    """
    if df is None or len(df) == 0:
        print(f"  Skipping P1 for {step_name}: no data")
        return
    
    # Get stop_reason
    stop_reason = df["meta_stop_reason"].fillna(df.get("stop_reason", ""))
    df = df.copy()
    df["_stop_category"] = stop_reason.apply(categorize_stop_reason)
    
    categories = get_stop_reason_order()
    # Colors for each category
    colors = {
        "loss_threshold_legacy": "#d62728",  # red
        "max_epochs": "#2ca02c",  # green
        "train_pred_mse": "#1f77b4",  # blue
        "threshold": "#ff7f0e",  # orange
        "other": "#7f7f7f",  # gray
    }
    
    for alpha in alpha_focus:
        # Filter to this alpha
        alpha_mask = np.abs(df["alpha"] - alpha) < alpha_tol
        df_alpha = df[alpha_mask]
        
        if len(df_alpha) == 0:
            print(f"  No rows for alpha={alpha} in {step_name}")
            continue
        
        # Get groups
        groups = sorted(df_alpha[group_col].unique())
        
        # Compute fractions
        fractions = {cat: [] for cat in categories}
        for group in groups:
            group_mask = df_alpha[group_col] == group
            group_df = df_alpha[group_mask]
            n_total = len(group_df)
            for cat in categories:
                n_cat = (group_df["_stop_category"] == cat).sum()
                fractions[cat].append(n_cat / n_total if n_total > 0 else 0)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(groups))
        width = 0.6
        bottom = np.zeros(len(groups))
        
        for cat in categories:
            if any(fractions[cat]):
                ax.bar(
                    x, fractions[cat], width, 
                    bottom=bottom, 
                    label=cat, 
                    color=colors.get(cat, "#7f7f7f")
                )
                bottom += np.array(fractions[cat])
        
        ax.set_xlabel(group_col, fontsize=12)
        ax.set_ylabel("Fraction of Runs", fontsize=12)
        ax.set_title(f"Stop Reason Composition - {step_name} (α={alpha})", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in groups], rotation=45, ha="right")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        
        fig.tight_layout()
        
        basename = f"P1_{step_name}_stopreason_alpha={alpha}"
        save_figure(fig, outdir, basename)


# ==============================================================================
# P2: Scatter Diagnostic (Step 1 only)
# ==============================================================================

def plot_p2_scatter(
    df: pd.DataFrame,
    alpha_focus: list,
    alpha_tol: float,
    beta_rate_cutoffs: list,
    outdir: str
):
    """
    P2: Scatter plot of final_beta_update_rate vs final_param_mse (Step 1 only).
    """
    if df is None or len(df) == 0:
        print("  Skipping P2: no data")
        return
    
    stop_reason = df["meta_stop_reason"].fillna(df.get("stop_reason", ""))
    df = df.copy()
    df["_stop_category"] = stop_reason.apply(categorize_stop_reason)
    
    categories = get_stop_reason_order()
    colors = {
        "loss_threshold_legacy": "#d62728",
        "max_epochs": "#2ca02c",
        "train_pred_mse": "#1f77b4",
        "threshold": "#ff7f0e",
        "other": "#7f7f7f",
    }
    
    for alpha in alpha_focus:
        alpha_mask = np.abs(df["alpha"] - alpha) < alpha_tol
        df_alpha = df[alpha_mask]
        
        if len(df_alpha) == 0:
            print(f"  No rows for alpha={alpha} in step1")
            continue
        
        # Get param_mse - prefer meta version if available
        param_mse = df_alpha["meta_final_param_mse"].fillna(df_alpha.get("param_mse", np.nan))
        beta_rate = df_alpha["meta_final_beta_update_rate"]
        
        # Filter to valid values
        valid = np.isfinite(beta_rate) & np.isfinite(param_mse) & (beta_rate > 0) & (param_mse > 0)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for cat in categories:
            cat_mask = (df_alpha["_stop_category"] == cat) & valid
            if cat_mask.any():
                ax.scatter(
                    beta_rate[cat_mask],
                    param_mse[cat_mask],
                    label=cat,
                    color=colors.get(cat, "#7f7f7f"),
                    alpha=0.7,
                    s=50,
                    edgecolors="white",
                    linewidths=0.5,
                )
        
        # Add vertical dashed lines for cutoffs
        for cutoff in sorted(beta_rate_cutoffs, reverse=True):
            ax.axvline(x=cutoff, linestyle="--", color="black", alpha=0.5, linewidth=1)
            ax.text(
                cutoff, ax.get_ylim()[1], f"{cutoff:.0e}", 
                rotation=90, va="top", ha="right", fontsize=8, alpha=0.7
            )
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Final Beta Update Rate (log scale)", fontsize=12)
        ax.set_ylabel("Final Param MSE (log scale)", fontsize=12)
        ax.set_title(f"P2: Beta Rate vs Param MSE - Step 1 (α={alpha})", fontsize=13)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        
        fig.tight_layout()
        
        basename = f"P2_step1_scatter_betaRate_vs_paramMSE_alpha={alpha}"
        save_figure(fig, outdir, basename)


# ==============================================================================
# P3 & P4: Curves Showing Error Bar Collapse
# ==============================================================================

def plot_curves_with_filters(
    df: pd.DataFrame,
    step_name: str,
    group_col: str,
    filters: dict,
    filter_names: list,
    outdir: str,
    plot_name: str
):
    """
    P3/P4: Plot curves showing median and IQR for each filter.
    """
    if df is None or len(df) == 0:
        print(f"  Skipping {plot_name} for {step_name}: no data")
        return
    
    groups = sorted(df[group_col].unique())
    n_groups = len(groups)
    
    if n_groups == 0:
        print(f"  Skipping {plot_name}: no groups")
        return
    
    # Colors for filters
    filter_colors = {
        "ALL": "#1f77b4",
        "NO_PREMATURE_LEGACY": "#ff7f0e",
        "LONG_TIME_ONLY": "#2ca02c",
        "FIXED_POINT_betaRate_1e-06": "#d62728",
    }
    
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 6), sharey=True)
    if n_groups == 1:
        axes = [axes]
    
    for ax_idx, group in enumerate(groups):
        ax = axes[ax_idx]
        group_mask = df[group_col] == group
        df_group = df[group_mask]
        
        alphas = sorted(df_group["alpha"].unique())
        
        for filter_name in filter_names:
            if filter_name not in filters:
                continue
            
            filter_mask = filters[filter_name][group_mask]
            df_filtered = df_group[filter_mask]
            
            if len(df_filtered) == 0:
                continue
            
            # Compute stats per alpha
            medians = []
            q25s = []
            q75s = []
            valid_alphas = []
            
            for alpha in alphas:
                alpha_mask = df_filtered["alpha"] == alpha
                df_a = df_filtered[alpha_mask]
                
                # Get param_mse - prefer meta version if available
                param_mse = df_a["meta_final_param_mse"].fillna(df_a.get("param_mse", np.nan))
                valid_mse = param_mse[np.isfinite(param_mse)]
                
                if len(valid_mse) > 0:
                    valid_alphas.append(alpha)
                    medians.append(np.median(valid_mse))
                    q25s.append(np.percentile(valid_mse, 25))
                    q75s.append(np.percentile(valid_mse, 75))
            
            if len(valid_alphas) == 0:
                continue
            
            valid_alphas = np.array(valid_alphas)
            medians = to_db(np.array(medians))
            q25s = to_db(np.array(q25s))
            q75s = to_db(np.array(q75s))
            
            color = filter_colors.get(filter_name, "#7f7f7f")
            
            ax.plot(valid_alphas, medians, "-o", label=filter_name, color=color, linewidth=2, markersize=4)
            ax.fill_between(valid_alphas, q25s, q75s, alpha=0.2, color=color)
        
        ax.set_xlabel(r"$\alpha = n / d$", fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel("Param MSE (dB)", fontsize=12)
        ax.set_title(f"{group_col}={group}", fontsize=11)
        ax.grid(True, alpha=0.3)
        if ax_idx == n_groups - 1:
            ax.legend(loc="best", fontsize=8)
    
    fig.suptitle(f"{plot_name}: Curves by Filter - {step_name}", fontsize=13)
    fig.tight_layout()
    
    basename = f"{plot_name}_{step_name}_curves_all_vs_filters"
    save_figure(fig, outdir, basename)


# ==============================================================================
# P5: Survival Counts Tables
# ==============================================================================

def save_survival_counts(
    df: pd.DataFrame,
    step_name: str,
    group_col: str,
    filters: dict,
    filter_names: list,
    outdir: str
):
    """
    P5: Save survival counts as CSV.
    """
    if df is None or len(df) == 0:
        print(f"  Skipping P5 for {step_name}: no data")
        return
    
    groups = sorted(df[group_col].unique())
    alphas = sorted(df["alpha"].unique())
    
    rows = []
    for group in groups:
        group_mask = df[group_col] == group
        for alpha in alphas:
            alpha_mask = df["alpha"] == alpha
            combined_mask = group_mask & alpha_mask
            
            row = {group_col: group, "alpha": alpha}
            for filter_name in filter_names:
                if filter_name in filters:
                    count = (combined_mask & filters[filter_name]).sum()
                    row[f"count_{filter_name}"] = count
            rows.append(row)
    
    counts_df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, f"P5_{step_name}_survival_counts.csv")
    counts_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()
    
    # Create output directory
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("DIAGONAL NET POST-HOC DEBUG PLOTS")
    print("=" * 80)
    print(f"\nOutput directory: {args.outdir}")
    print(f"Beta rate cutoffs: {args.beta_rate_cutoffs}")
    print(f"Alpha focus: {args.alpha_focus}")
    print(f"Alpha tolerance: {args.alpha_tol}")
    print(f"Test every: {args.test_every}")
    
    # Load data
    print("\n" + "-" * 40)
    print("LOADING DATA")
    print("-" * 40)
    
    df_step1 = load_csv_with_meta(args.csv_step1)
    df_step2 = load_csv_with_meta(args.csv_step2)
    
    # Create filters
    print("\n" + "-" * 40)
    print("CREATING FILTERS")
    print("-" * 40)
    
    filter_names = [
        "ALL",
        "NO_PREMATURE_LEGACY",
        "LONG_TIME_ONLY",
        "FIXED_POINT_betaRate_1e-06",
    ]
    
    if df_step1 is not None:
        filters_step1 = create_filters(df_step1, args.test_every, args.beta_rate_cutoffs)
        print("\nStep 1 filter survival counts:")
        for name in filter_names:
            if name in filters_step1:
                count = filters_step1[name].sum()
                print(f"  {name}: {count}/{len(df_step1)} rows")
    else:
        filters_step1 = {}
    
    if df_step2 is not None:
        filters_step2 = create_filters(df_step2, args.test_every, args.beta_rate_cutoffs)
        print("\nStep 2 filter survival counts:")
        for name in filter_names:
            if name in filters_step2:
                count = filters_step2[name].sum()
                print(f"  {name}: {count}/{len(df_step2)} rows")
    else:
        filters_step2 = {}
    
    # P1: Stop Reason Composition
    print("\n" + "-" * 40)
    print("P1: STOP REASON COMPOSITION")
    print("-" * 40)
    
    plot_p1_stop_reason(
        df_step1, "step1", "pi_A", 
        args.alpha_focus, args.alpha_tol, args.outdir
    )
    plot_p1_stop_reason(
        df_step2, "step2", "case_name",
        args.alpha_focus, args.alpha_tol, args.outdir
    )
    
    # P2: Scatter Diagnostic (Step 1 only)
    print("\n" + "-" * 40)
    print("P2: SCATTER DIAGNOSTIC (STEP 1)")
    print("-" * 40)
    
    plot_p2_scatter(
        df_step1, args.alpha_focus, args.alpha_tol,
        args.beta_rate_cutoffs, args.outdir
    )
    
    # P3: Curves for Step 1
    print("\n" + "-" * 40)
    print("P3: CURVES SHOWING ERROR BAR COLLAPSE (STEP 1)")
    print("-" * 40)
    
    plot_curves_with_filters(
        df_step1, "step1", "pi_A",
        filters_step1, filter_names, args.outdir, "P3"
    )
    
    # P4: Curves for Step 2
    print("\n" + "-" * 40)
    print("P4: CURVES SHOWING ERROR BAR COLLAPSE (STEP 2)")
    print("-" * 40)
    
    plot_curves_with_filters(
        df_step2, "step2", "case_name",
        filters_step2, filter_names, args.outdir, "P4"
    )
    
    # P5: Survival Counts Tables
    print("\n" + "-" * 40)
    print("P5: SURVIVAL COUNTS TABLES")
    print("-" * 40)
    
    save_survival_counts(
        df_step1, "step1", "pi_A",
        filters_step1, filter_names, args.outdir
    )
    save_survival_counts(
        df_step2, "step2", "case_name",
        filters_step2, filter_names, args.outdir
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"All outputs saved to: {args.outdir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())




