#!/usr/bin/env python3
"""
Phase 0.5 Sanity Checks: Verify fixed-point stopping fix works correctly.

Checks:
1. No legacy premature stops (stop_reason must NOT contain "loss_threshold_legacy")
2. Eval happened (eval_count >= 1 for every run)
3. Fixed-point correctness (for fixed_point stops, beta_rate <= threshold)
4. Max-epochs diagnosis (gap metrics for runs that hit max epochs)

Usage:
    python experiments/diagonal/phase0_5_sanity_checks.py
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("phase0_5_outputs")

# Mode A: Directory paths (full sweeps)
STEP_DIRS = {
    1: "results/diagonal/step1_mixture",
    2: "results/diagonal/step2_support",
    3: "results/diagonal/step3_omega",
}

# Mode A (Phase 0): Directory paths for Phase 0 smoke runs
PHASE0_STEP_DIRS = {
    1: "results/diagonal/phase0/step1",
    2: "results/diagonal/phase0/step2",
    3: "results/diagonal/phase0/step3",
}

# Mode A (Phase 1): Directory paths for Phase 1 full sweeps
PHASE1_STEP_DIRS = {
    1: "results/diagonal_phase1/step1_mixture",
    2: "results/diagonal_phase1/step2_support",
    3: "results/diagonal_phase1/step3_omega",
}

# Mode B: CSV paths (full sweeps)
STEP_CSVS = {
    1: "experiment_results_step1_mixture.csv",
    2: "experiment_results_step2_support.csv",
    3: "experiment_results_step3_omega.csv",
}

# Mode B (Phase 0): CSV paths for Phase 0 smoke runs  
PHASE0_STEP_CSVS = {
    1: "phase0_step1.csv",
    2: "phase0_step2.csv",
    3: "phase0_step3.csv",
}

# Mode B (Phase 1): CSV paths for Phase 1 full sweeps
PHASE1_STEP_CSVS = {
    1: "experiment_results_step1_mixture_phase1.csv",
    2: "experiment_results_step2_support_phase1.csv",
    3: "experiment_results_step3_omega_phase1.csv",
}

# Default stop_pred_mse (from training script default)
DEFAULT_STOP_PRED_MSE = 1e-10


# ============================================================================
# Path parsing helpers
# ============================================================================

def parse_step1_path(folder_name):
    """Parse step1 folder: pi_A=...--alpha=...--seed=..."""
    match = re.match(r"pi_A=([0-9.]+)--alpha=([0-9.]+)--seed=(\d+)", folder_name)
    if match:
        return {
            "pi_A": float(match.group(1)),
            "alpha": float(match.group(2)),
            "seed": int(match.group(3)),
            "group_key": float(match.group(1)),  # pi_A as group_key
        }
    return None


def parse_step2_path(folder_name):
    """Parse step2 folder: c_nz=...--c_z=...--alpha=...--seed=... or case=...--alpha=...--seed=..."""
    # Try full sweep format first: c_nz=...--c_z=...--alpha=...--seed=...
    match = re.match(r"c_nz=([0-9.]+)--c_z=([0-9.]+)--alpha=([0-9.]+)--seed=(\d+)", folder_name)
    if match:
        c_nz = float(match.group(1))
        c_z = float(match.group(2))
        case_name = "bad" if c_nz > c_z else "good"
        return {
            "c_nz": c_nz,
            "c_z": c_z,
            "case_name": case_name,
            "alpha": float(match.group(3)),
            "seed": int(match.group(4)),
            "group_key": case_name,
        }
    
    # Try Phase 0 format: case=...--alpha=...--seed=...
    match = re.match(r"case=(good|bad)--alpha=([0-9.]+)--seed=(\d+)", folder_name)
    if match:
        case_name = match.group(1)
        # Infer c_nz/c_z from case name
        if case_name == "bad":
            c_nz, c_z = 0.5, 0.001
        else:
            c_nz, c_z = 0.001, 0.5
        return {
            "c_nz": c_nz,
            "c_z": c_z,
            "case_name": case_name,
            "alpha": float(match.group(2)),
            "seed": int(match.group(3)),
            "group_key": case_name,
        }
    
    return None


def parse_step3_path(folder_name):
    """Parse step3 folder: omega=...--alpha=...--seed=..."""
    match = re.match(r"omega=([0-9.]+)--alpha=([0-9.]+)--seed=(\d+)", folder_name)
    if match:
        return {
            "omega": float(match.group(1)),
            "alpha": float(match.group(2)),
            "seed": int(match.group(3)),
            "group_key": float(match.group(1)),  # omega as group_key
        }
    return None


PARSERS = {
    1: parse_step1_path,
    2: parse_step2_path,
    3: parse_step3_path,
}


# ============================================================================
# Data loading
# ============================================================================

def load_mode_a(step, base_dir):
    """Load data from results_meta.json files in directory."""
    rows = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return None
    
    parser = PARSERS[step]
    
    for folder in base_path.iterdir():
        if not folder.is_dir():
            continue
        
        meta_path = folder / "results_meta.json"
        if not meta_path.exists():
            continue
        
        # Parse folder name
        parsed = parser(folder.name)
        if parsed is None:
            print(f"  Warning: Could not parse folder name: {folder.name}")
            continue
        
        # Load meta
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load {meta_path}: {e}")
            continue
        
        # Build row
        row = {
            "step": step,
            "alpha": parsed["alpha"],
            "seed": parsed["seed"],
            "group_key": parsed["group_key"],
            "stop_reason": meta.get("stop_reason"),
            "final_epoch": meta.get("final_epoch"),
            "eval_count": meta.get("eval_count"),
            "final_train_pred_mse": meta.get("final_train_pred_mse"),
            "final_test_pred_mse": meta.get("final_test_pred_mse"),
            "final_param_mse": meta.get("final_param_mse"),
            "final_beta_update_rate": meta.get("final_beta_update_rate"),
            "fixed_point_beta_rate": meta.get("fixed_point_beta_rate"),
            "fixed_point_consecutive_evals": meta.get("fixed_point_consecutive_evals"),
            "min_epochs_before_stop": meta.get("min_epochs_before_stop"),
            "legacy_loss_stop_disabled": meta.get("legacy_loss_stop_disabled"),
            "final_grad_norm": meta.get("final_grad_norm"),
        }
        
        # Add step-specific fields
        if step == 1:
            row["pi_A"] = parsed.get("pi_A")
        elif step == 2:
            row["case_name"] = parsed.get("case_name")
            row["c_nz"] = parsed.get("c_nz")
            row["c_z"] = parsed.get("c_z")
        elif step == 3:
            row["omega"] = parsed.get("omega")
        
        rows.append(row)
    
    if len(rows) == 0:
        return None
    
    return pd.DataFrame(rows)


def load_mode_b(step, csv_path):
    """Load data from CSV file."""
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Warning: Could not load {csv_path}: {e}")
        return None
    
    # Add step column
    df["step"] = step
    
    # Create group_key based on step
    if step == 1:
        if "pi_A" in df.columns:
            df["group_key"] = df["pi_A"]
    elif step == 2:
        if "case_name" in df.columns:
            df["group_key"] = df["case_name"]
        elif "c_nz" in df.columns and "c_z" in df.columns:
            df["case_name"] = df.apply(lambda r: "bad" if r["c_nz"] > r["c_z"] else "good", axis=1)
            df["group_key"] = df["case_name"]
    elif step == 3:
        if "omega" in df.columns:
            df["group_key"] = df["omega"]
    
    return df


def load_all_data(use_phase0=False, use_phase1=False):
    """Load data from Mode A (directories) or Mode B (CSVs).
    
    Args:
        use_phase0: If True, use Phase 0 directories/CSVs instead of full sweeps.
        use_phase1: If True, use Phase 1 directories/CSVs instead of full sweeps.
    """
    all_dfs = []
    mode_used = {}
    
    # Select which paths to use
    if use_phase1:
        step_dirs = PHASE1_STEP_DIRS
        step_csvs = PHASE1_STEP_CSVS
        source_name = "Phase 1"
    elif use_phase0:
        step_dirs = PHASE0_STEP_DIRS
        step_csvs = PHASE0_STEP_CSVS
        source_name = "Phase 0"
    else:
        step_dirs = STEP_DIRS
        step_csvs = STEP_CSVS
        source_name = "Full sweeps (old)"
    
    print(f"Loading from: {source_name}")
    
    for step in [1, 2, 3]:
        # Try Mode A first
        df = load_mode_a(step, step_dirs[step])
        if df is not None and len(df) > 0:
            mode_used[step] = "A (directories)"
            all_dfs.append(df)
            print(f"Step {step}: Loaded {len(df)} runs from Mode A (directories)")
            continue
        
        # Fall back to Mode B
        df = load_mode_b(step, step_csvs[step])
        if df is not None and len(df) > 0:
            mode_used[step] = "B (CSV)"
            all_dfs.append(df)
            print(f"Step {step}: Loaded {len(df)} runs from Mode B (CSV)")
            continue
        
        print(f"Step {step}: No data found")
        mode_used[step] = "None"
    
    if len(all_dfs) == 0:
        return None, mode_used
    
    # Combine all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    return df, mode_used


# ============================================================================
# Sanity checks
# ============================================================================

def compute_derived_columns(df):
    """Add derived check columns."""
    # 1. Legacy stop check
    # Distinguish between unguarded (bad) and guarded (ok) legacy stops
    df["is_legacy_stop_unguarded"] = df["stop_reason"].fillna("").apply(
        lambda x: "loss_threshold_legacy" in x and "guarded" not in x
    )
    df["is_legacy_stop_guarded"] = df["stop_reason"].fillna("").str.contains(
        "loss_threshold_legacy_guarded", case=False, regex=False
    )
    # For backward compat, is_legacy_stop = unguarded only (the bad kind)
    df["is_legacy_stop"] = df["is_legacy_stop_unguarded"]
    
    # 2. Missing eval check
    df["is_missing_eval"] = df["eval_count"].isna() | (df["eval_count"] < 1)
    
    # 3. Fixed-point correctness check
    def check_fixed_point(row):
        if row["stop_reason"] != "fixed_point":
            return np.nan
        if pd.isna(row["final_beta_update_rate"]) or pd.isna(row["fixed_point_beta_rate"]):
            return np.nan
        return row["final_beta_update_rate"] <= row["fixed_point_beta_rate"] + 1e-12
    
    df["is_fixed_point_ok"] = df.apply(check_fixed_point, axis=1)
    
    # 4. Gap metrics for max_epochs
    def compute_gap_to_beta_rate(row):
        if pd.isna(row["final_beta_update_rate"]) or pd.isna(row["fixed_point_beta_rate"]):
            return np.nan
        if row["fixed_point_beta_rate"] == 0:
            return np.nan
        return row["final_beta_update_rate"] / row["fixed_point_beta_rate"]
    
    df["gap_to_beta_rate"] = df.apply(compute_gap_to_beta_rate, axis=1)
    
    # Gap to pred MSE (using default threshold)
    def compute_gap_to_pred_mse(row):
        if pd.isna(row["final_train_pred_mse"]):
            return np.nan
        return row["final_train_pred_mse"] / DEFAULT_STOP_PRED_MSE
    
    df["gap_to_pred_mse"] = df.apply(compute_gap_to_pred_mse, axis=1)
    
    return df


# ============================================================================
# Report generation
# ============================================================================

def generate_terminal_report(df, mode_used, output_path):
    """Generate the terminal report."""
    lines = []
    lines.append("=" * 80)
    lines.append("PHASE 0.5 SANITY CHECK REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Data sources
    lines.append("DATA SOURCES:")
    for step, mode in mode_used.items():
        lines.append(f"  Step {step}: {mode}")
    lines.append("")
    
    # Total runs per step
    lines.append("-" * 40)
    lines.append("TOTAL RUNS PER STEP:")
    lines.append("-" * 40)
    for step in [1, 2, 3]:
        step_df = df[df["step"] == step]
        lines.append(f"  Step {step}: {len(step_df)} runs")
    lines.append(f"  Total: {len(df)} runs")
    lines.append("")
    
    # Stop reason counts per step
    lines.append("-" * 40)
    lines.append("STOP REASON COUNTS PER STEP:")
    lines.append("-" * 40)
    for step in [1, 2, 3]:
        step_df = df[df["step"] == step]
        if len(step_df) == 0:
            lines.append(f"  Step {step}: No data")
            continue
        lines.append(f"  Step {step}:")
        counts = step_df["stop_reason"].value_counts()
        for reason, count in counts.items():
            lines.append(f"    {reason}: {count}")
    lines.append("")
    
    # Check 1: Legacy stops
    lines.append("-" * 40)
    lines.append("CHECK 1: LEGACY PREMATURE STOPS")
    lines.append("-" * 40)
    
    # Unguarded legacy stops (bad - old code without safeguards)
    legacy_unguarded = df[df["is_legacy_stop_unguarded"] == True]
    if len(legacy_unguarded) == 0:
        lines.append("  ✓ PASS: No unguarded legacy stops (the bad kind)")
    else:
        lines.append(f"  ✗ FAIL: {len(legacy_unguarded)} UNGUARDED legacy stops (BAD)")
        for step in [1, 2, 3]:
            step_legacy = legacy_unguarded[legacy_unguarded["step"] == step]
            if len(step_legacy) > 0:
                lines.append(f"    Step {step}: {len(step_legacy)}")
    
    # Guarded legacy stops (ok - new code with safeguards verified)
    legacy_guarded = df[df["is_legacy_stop_guarded"] == True]
    if len(legacy_guarded) > 0:
        lines.append(f"  ℹ INFO: {len(legacy_guarded)} guarded legacy stops (OK - guards verified)")
        for step in [1, 2, 3]:
            step_legacy = legacy_guarded[legacy_guarded["step"] == step]
            if len(step_legacy) > 0:
                lines.append(f"    Step {step}: {len(step_legacy)}")
    lines.append("")
    
    # Check 2: Eval count
    lines.append("-" * 40)
    lines.append("CHECK 2: EVAL COUNT >= 1")
    lines.append("-" * 40)
    missing_eval = df[df["is_missing_eval"] == True]
    if len(missing_eval) == 0:
        lines.append("  ✓ PASS: All runs have eval_count >= 1")
    else:
        lines.append(f"  ✗ FAIL: {len(missing_eval)} runs with eval_count < 1")
        for step in [1, 2, 3]:
            step_missing = missing_eval[missing_eval["step"] == step]
            if len(step_missing) > 0:
                lines.append(f"    Step {step}: {len(step_missing)}")
    lines.append("")
    
    # Check 3: Fixed-point correctness
    lines.append("-" * 40)
    lines.append("CHECK 3: FIXED-POINT CORRECTNESS")
    lines.append("-" * 40)
    fixed_point_runs = df[df["stop_reason"] == "fixed_point"]
    if len(fixed_point_runs) == 0:
        lines.append("  (No fixed_point stops to check)")
    else:
        violations = fixed_point_runs[fixed_point_runs["is_fixed_point_ok"] == False]
        if len(violations) == 0:
            lines.append(f"  ✓ PASS: All {len(fixed_point_runs)} fixed_point stops are valid")
        else:
            lines.append(f"  ✗ FAIL: {len(violations)} / {len(fixed_point_runs)} fixed_point stops violated threshold")
    lines.append("")
    
    # Check 4: Max-epochs diagnosis
    lines.append("-" * 40)
    lines.append("CHECK 4: MAX-EPOCHS DIAGNOSIS")
    lines.append("-" * 40)
    max_epoch_runs = df[df["stop_reason"] == "max_epochs"]
    if len(max_epoch_runs) == 0:
        lines.append("  (No max_epochs stops)")
    else:
        lines.append(f"  Total max_epochs runs: {len(max_epoch_runs)}")
        lines.append("")
        
        # Gap to beta rate
        gap_beta = max_epoch_runs["gap_to_beta_rate"].dropna()
        if len(gap_beta) > 0:
            lines.append("  gap_to_beta_rate (final_beta_rate / threshold):")
            lines.append(f"    min:    {gap_beta.min():.4f}")
            lines.append(f"    median: {gap_beta.median():.4f}")
            lines.append(f"    max:    {gap_beta.max():.4f}")
        else:
            lines.append("  gap_to_beta_rate: N/A (no data)")
        lines.append("")
        
        # Final param MSE
        param_mse = max_epoch_runs["final_param_mse"].dropna()
        if len(param_mse) > 0:
            lines.append("  final_param_mse:")
            lines.append(f"    min:    {param_mse.min():.6e}")
            lines.append(f"    median: {param_mse.median():.6e}")
            lines.append(f"    max:    {param_mse.max():.6e}")
        else:
            lines.append("  final_param_mse: N/A (no data)")
    lines.append("")
    
    # Overall verdict
    lines.append("=" * 80)
    lines.append("OVERALL VERDICT")
    lines.append("=" * 80)
    
    all_pass = True
    if len(legacy_unguarded) > 0:
        all_pass = False
        lines.append("  ✗ Unguarded legacy stops detected (BAD)")
    if len(legacy_guarded) > 0:
        lines.append(f"  ℹ {len(legacy_guarded)} guarded legacy stops (OK)")
    if len(missing_eval) > 0:
        all_pass = False
        lines.append("  ✗ Missing evals detected")
    if len(fixed_point_runs) > 0:
        violations = fixed_point_runs[fixed_point_runs["is_fixed_point_ok"] == False]
        if len(violations) > 0:
            all_pass = False
            lines.append("  ✗ Fixed-point violations detected")
    
    if all_pass:
        lines.append("  ✓ ALL CHECKS PASSED")
    else:
        lines.append("  ✗ SOME CHECKS FAILED (see above)")
    
    lines.append("=" * 80)
    
    # Write to file
    report_text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report_text)
    
    # Also print to terminal
    print("\n" + report_text)
    
    return report_text


# ============================================================================
# Plots
# ============================================================================

def create_plots(df, output_dir):
    """Create diagnostic plots."""
    
    # Set style
    plt.style.use('default')
    
    # P0_5A: Stop reason bar by step
    fig, ax = plt.subplots(figsize=(10, 5))
    
    steps = [1, 2, 3]
    stop_reasons = df["stop_reason"].dropna().unique()
    x = np.arange(len(steps))
    width = 0.2
    
    colors = {
        "train_pred_mse": "#2ecc71",
        "fixed_point": "#3498db", 
        "max_epochs": "#e74c3c",
        "loss_threshold_legacy_guarded": "#f39c12",
        "loss_threshold_legacy": "#e67e22",
    }
    
    for i, reason in enumerate(sorted(stop_reasons)):
        counts = []
        for step in steps:
            step_df = df[df["step"] == step]
            counts.append(len(step_df[step_df["stop_reason"] == reason]))
        color = colors.get(reason, "#95a5a6")
        ax.bar(x + i * width, counts, width, label=reason, color=color)
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Count")
    ax.set_title("P0.5A: Stop Reason Distribution by Step")
    ax.set_xticks(x + width * (len(stop_reasons) - 1) / 2)
    ax.set_xticklabels([f"Step {s}" for s in steps])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "P0_5A_stop_reason_bar_by_step.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'P0_5A_stop_reason_bar_by_step.png'}")
    
    # P0_5B: Eval count histogram by step
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i, step in enumerate([1, 2, 3]):
        ax = axes[i]
        step_df = df[df["step"] == step]
        eval_counts = step_df["eval_count"].dropna()
        
        if len(eval_counts) > 0:
            ax.hist(eval_counts, bins=30, edgecolor="black", alpha=0.7)
            ax.axvline(x=1, color="red", linestyle="--", label="Threshold (1)")
        
        ax.set_xlabel("Eval Count")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Step {step}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle("P0.5B: Eval Count Distribution by Step")
    plt.tight_layout()
    plt.savefig(output_dir / "P0_5B_eval_count_hist_by_step.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'P0_5B_eval_count_hist_by_step.png'}")
    
    # P0_5C: Fixed-point beta rate scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fp_runs = df[df["stop_reason"] == "fixed_point"].copy()
    if len(fp_runs) > 0:
        for step in [1, 2, 3]:
            step_fp = fp_runs[fp_runs["step"] == step]
            if len(step_fp) > 0:
                ax.scatter(
                    step_fp["alpha"],
                    step_fp["final_beta_update_rate"],
                    label=f"Step {step}",
                    alpha=0.7,
                    s=50
                )
        
        # Add threshold line
        threshold = fp_runs["fixed_point_beta_rate"].iloc[0] if len(fp_runs) > 0 else 1e-6
        ax.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.0e})")
        
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Final Beta Update Rate")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No fixed_point stops", ha="center", va="center", transform=ax.transAxes)
    
    ax.set_title("P0.5C: Fixed-Point Stops - Beta Update Rate at Stop")
    plt.tight_layout()
    plt.savefig(output_dir / "P0_5C_fixed_point_beta_rate_scatter.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'P0_5C_fixed_point_beta_rate_scatter.png'}")
    
    # P0_5D: Max-epochs gap scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    max_runs = df[df["stop_reason"] == "max_epochs"].copy()
    
    # Gap to beta rate
    ax = axes[0]
    if len(max_runs) > 0:
        for step in [1, 2, 3]:
            step_max = max_runs[max_runs["step"] == step]
            if len(step_max) > 0:
                ax.scatter(
                    step_max["alpha"],
                    step_max["gap_to_beta_rate"],
                    label=f"Step {step}",
                    alpha=0.7,
                    s=50
                )
        ax.axhline(y=1, color="red", linestyle="--", label="Threshold (1.0)")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Gap to Beta Rate (final / threshold)")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No max_epochs stops", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Gap to Beta Rate Threshold")
    
    # Final param MSE
    ax = axes[1]
    if len(max_runs) > 0:
        for step in [1, 2, 3]:
            step_max = max_runs[max_runs["step"] == step]
            if len(step_max) > 0:
                ax.scatter(
                    step_max["alpha"],
                    step_max["final_param_mse"],
                    label=f"Step {step}",
                    alpha=0.7,
                    s=50
                )
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Final Param MSE")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No max_epochs stops", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Final Param MSE")
    
    plt.suptitle("P0.5D: Max-Epochs Runs Diagnosis")
    plt.tight_layout()
    plt.savefig(output_dir / "P0_5D_max_epochs_gap_scatter.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'P0_5D_max_epochs_gap_scatter.png'}")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 0.5 Sanity Checks")
    parser.add_argument("--phase0", action="store_true",
                        help="Check Phase 0 smoke run results instead of full sweeps")
    parser.add_argument("--phase1", action="store_true",
                        help="Check Phase 1 full sweep results (with updated stopping logic)")
    args = parser.parse_args()
    
    if args.phase0 and args.phase1:
        print("ERROR: Cannot specify both --phase0 and --phase1")
        return
    
    print("=" * 80)
    print("PHASE 0.5 SANITY CHECKS")
    print("=" * 80)
    print("")
    
    # Adjust output directory based on phase
    output_dir = OUTPUT_DIR
    if args.phase0:
        output_dir = Path("phase0_5_outputs_phase0")
    elif args.phase1:
        output_dir = Path("phase0_5_outputs_phase1")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print("")
    
    # Load data
    print("Loading data...")
    df, mode_used = load_all_data(use_phase0=args.phase0, use_phase1=args.phase1)
    
    if df is None or len(df) == 0:
        print("ERROR: No data found!")
        print("Expected directories or CSVs:")
        for step in [1, 2, 3]:
            print(f"  Step {step}: {STEP_DIRS[step]} or {STEP_CSVS[step]}")
        return
    
    print(f"\nTotal runs loaded: {len(df)}")
    print("")
    
    # Compute derived columns
    print("Computing sanity check columns...")
    df = compute_derived_columns(df)
    
    # Select and order columns for output
    output_columns = [
        "step", "group_key", "alpha", "seed",
        "stop_reason", "final_epoch", "eval_count",
        "final_train_pred_mse", "final_test_pred_mse", "final_param_mse",
        "final_beta_update_rate", "fixed_point_beta_rate",
        "fixed_point_consecutive_evals", "min_epochs_before_stop",
        "legacy_loss_stop_disabled",
        # Derived
        "is_legacy_stop", "is_missing_eval", "is_fixed_point_ok",
        "gap_to_beta_rate", "gap_to_pred_mse",
    ]
    
    # Keep only columns that exist
    output_columns = [c for c in output_columns if c in df.columns]
    
    # Save summary CSV
    csv_path = output_dir / "phase0_5_summary.csv"
    df[output_columns].to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    
    # Generate terminal report
    report_path = output_dir / "phase0_5_terminal_report.txt"
    generate_terminal_report(df, mode_used, report_path)
    print(f"\nSaved: {report_path}")
    
    # Create plots
    print("\nGenerating plots...")
    create_plots(df, output_dir)
    
    print("\n" + "=" * 80)
    print("PHASE 0.5 SANITY CHECKS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs in: {output_dir}/")
    print("  - phase0_5_summary.csv")
    print("  - phase0_5_terminal_report.txt")
    print("  - P0_5A_stop_reason_bar_by_step.png")
    print("  - P0_5B_eval_count_hist_by_step.png")
    print("  - P0_5C_fixed_point_beta_rate_scatter.png")
    print("  - P0_5D_max_epochs_gap_scatter.png")


if __name__ == "__main__":
    main()

