#!/usr/bin/env python3
"""
Patch existing aggregated CSV files to add parameter-MSE dB columns.

This script allows regeneration of plots using PARAMETER MSE instead of TEST PREDICTION MSE
without rerunning experiments. It adds param_mse_*_db columns to existing CSV files.
"""

import argparse
import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np


def patch_csv_file(csv_path, inp_dim=1000, n_test=10000, epsilon=1e-15, inplace=False):
    """
    Patch a single CSV file to add param_mse_*_db columns.
    
    Args:
        csv_path: Path to CSV file
        inp_dim: Input dimension (for diagnostic pred_mse_from_param columns)
        n_test: Number of test samples (for diagnostic pred_mse_from_param columns)
        epsilon: Small value to add before log10 to avoid log(0)
        inplace: If True, overwrite original file; else save as *_PATCHED.csv
    
    Returns:
        dict with status information
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"status": "error", "error": str(e), "columns_added": 0}
    
    required_cols = ["param_mse_mean", "param_mse_median", "param_mse_q25", "param_mse_q75"]
    
    # Check if required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {
            "status": "skipped",
            "reason": f"Missing required columns: {missing_cols}",
            "columns_added": 0
        }
    
    columns_added = 0
    
    # Add param_mse_*_db columns
    db_columns = [
        ("param_mse_mean", "param_mse_mean_db"),
        ("param_mse_median", "param_mse_median_db"),
        ("param_mse_q25", "param_mse_q25_db"),
        ("param_mse_q75", "param_mse_q75_db"),
    ]
    
    for source_col, target_col in db_columns:
        if target_col not in df.columns:
            df[target_col] = 10 * np.log10(df[source_col] + epsilon)
            columns_added += 1
    
    # Add optional diagnostic columns: pred_mse_from_param_*
    # These show what the test prediction MSE would be if computed from parameter MSE
    # Formula: pred_mse = (inp_dim / n_test) * param_mse
    diagnostic_cols = [
        ("param_mse_mean", "pred_mse_from_param_mean"),
        ("param_mse_median", "pred_mse_from_param_median"),
        ("param_mse_q25", "pred_mse_from_param_q25"),
        ("param_mse_q75", "pred_mse_from_param_q75"),
    ]
    
    for source_col, target_col in diagnostic_cols:
        if target_col not in df.columns:
            df[target_col] = (inp_dim / n_test) * df[source_col]
            columns_added += 1
    
    # Add dB versions of diagnostic columns
    diagnostic_db_cols = [
        ("pred_mse_from_param_mean", "pred_mse_from_param_mean_db"),
        ("pred_mse_from_param_median", "pred_mse_from_param_median_db"),
        ("pred_mse_from_param_q25", "pred_mse_from_param_q25_db"),
        ("pred_mse_from_param_q75", "pred_mse_from_param_q75_db"),
    ]
    
    for source_col, target_col in diagnostic_db_cols:
        if target_col not in df.columns and source_col in df.columns:
            df[target_col] = 10 * np.log10(df[source_col] + epsilon)
            columns_added += 1
    
    # Save file
    if inplace:
        output_path = csv_path
    else:
        # Add _PATCHED before .csv extension
        base_path = Path(csv_path)
        output_path = base_path.parent / f"{base_path.stem}_PATCHED{base_path.suffix}"
    
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        return {"status": "error", "error": f"Failed to save: {e}", "columns_added": columns_added}
    
    # Compute statistics for report
    if "param_mse_mean_db" in df.columns:
        valid_values = df["param_mse_mean_db"].dropna()
        if len(valid_values) > 0:
            min_val = valid_values.min()
            max_val = valid_values.max()
        else:
            min_val = max_val = np.nan
    else:
        min_val = max_val = np.nan
    
    return {
        "status": "success",
        "columns_added": columns_added,
        "output_path": str(output_path),
        "min_param_mse_mean_db": min_val,
        "max_param_mse_mean_db": max_val,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Patch aggregated CSV files to add parameter-MSE dB columns"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="figures/diagonal/bg_generalization",
        help="Directory containing CSV files to patch (default: figures/diagonal/bg_generalization)",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="aggregated_results_rho=*.csv",
        help="Glob pattern to match CSV files (default: aggregated_results_rho=*.csv)",
    )
    parser.add_argument(
        "--inp_dim",
        type=int,
        default=1000,
        help="Input dimension for diagnostic columns (default: 1000)",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=10000,
        help="Number of test samples for diagnostic columns (default: 10000)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-15,
        help="Small value added before log10 to avoid log(0) (default: 1e-15)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite original files instead of creating *_PATCHED.csv files",
    )
    
    args = parser.parse_args()
    
    # Find matching CSV files
    search_pattern = os.path.join(args.input_dir, args.glob)
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching pattern: {search_pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to process")
    print(f"Input directory: {args.input_dir}")
    print(f"Glob pattern: {args.glob}")
    print(f"Inplace mode: {args.inplace}")
    print()
    
    # Process each file
    for csv_path in sorted(csv_files):
        filename = os.path.basename(csv_path)
        result = patch_csv_file(
            csv_path,
            inp_dim=args.inp_dim,
            n_test=args.n_test,
            epsilon=args.epsilon,
            inplace=args.inplace,
        )
        
        # Print one-line report
        if result["status"] == "success":
            min_str = f"{result['min_param_mse_mean_db']:.2f}" if not np.isnan(result['min_param_mse_mean_db']) else "nan"
            max_str = f"{result['max_param_mse_mean_db']:.2f}" if not np.isnan(result['max_param_mse_mean_db']) else "nan"
            print(
                f"{filename}: added {result['columns_added']} columns, "
                f"param_mse_mean_db range: [{min_str}, {max_str}]"
            )
        elif result["status"] == "skipped":
            print(f"{filename}: SKIPPED - {result['reason']}")
        else:
            print(f"{filename}: ERROR - {result.get('error', 'Unknown error')}")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()


# Usage example:
#
# python scripts/diagonal/patch_aggregated_csvs_add_param_db.py \
#   --input_dir figures/diagonal/bg_generalization \
#   --glob "aggregated_results_rho=*.csv" \
#   --inp_dim 1000 \
#   --n_test 10000 \
#   --inplace







