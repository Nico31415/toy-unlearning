#!/usr/bin/env python3
"""
Diagnostic script to analyze a single empirical run's convergence behavior.

Checks train/test epoch alignment and prints detailed convergence metrics.
"""

import argparse
import os
import pandas as pd
import numpy as np


def analyze_run(run_dir):
    """Analyze a single run directory."""
    run_dir = run_dir.rstrip("/")
    df_path = os.path.join(run_dir, "df.feather")
    
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"df.feather not found in {run_dir}")
    
    print("=" * 80)
    print(f"ANALYZING RUN: {run_dir}")
    print("=" * 80)
    
    # Read dataframe
    df = pd.read_feather(df_path)
    
    # Split into train and test
    train_df = df[df["split"] == "train"].copy().sort_values("epoch")
    test_df = df[df["split"] == "test"].copy().sort_values("epoch")
    
    if train_df.empty:
        print("ERROR: No train split data found")
        return
    if test_df.empty:
        print("ERROR: No test split data found")
        return
    
    # Check epoch alignment
    train_epochs = set(train_df["epoch"].values)
    test_epochs = set(test_df["epoch"].values)
    common_epochs = train_epochs.intersection(test_epochs)
    
    final_train_epoch = train_df["epoch"].max()
    final_test_epoch = test_df["epoch"].max()
    
    print(f"\nEPOCH ALIGNMENT CHECK:")
    print(f"  Final train epoch: {final_train_epoch}")
    print(f"  Final test epoch: {final_test_epoch}")
    if final_train_epoch == final_test_epoch:
        print(f"  ✓ Train and test are aligned at final epoch {final_train_epoch}")
    else:
        print(f"  ⚠ WARNING: Train and test final epochs differ!")
        print(f"    Using epoch intersection for final metrics")
    
    print(f"  Common epochs: {len(common_epochs)} out of {len(train_epochs)} train epochs and {len(test_epochs)} test epochs")
    
    # Infer logging cadence
    if len(test_df) > 1:
        test_epoch_diffs = np.diff(sorted(test_df["epoch"].values))
        most_common_diff = np.bincount(test_epoch_diffs.astype(int)).argmax()
        print(f"  Inferred test_every_n_epochs: ~{most_common_diff} (from test epoch differences)")
    else:
        print(f"  Cannot infer test_every_n_epochs (only 1 test evaluation)")
    
    # Get final metrics at aligned epoch
    if final_train_epoch == final_test_epoch:
        final_epoch = final_train_epoch
        final_train_row = train_df[train_df["epoch"] == final_epoch].iloc[0]
        final_test_row = test_df[test_df["epoch"] == final_epoch].iloc[0]
    else:
        # Use intersection - find latest common epoch
        if common_epochs:
            final_epoch = max(common_epochs)
            final_train_row = train_df[train_df["epoch"] == final_epoch].iloc[0]
            final_test_row = test_df[test_df["epoch"] == final_epoch].iloc[0]
            print(f"  Using latest common epoch: {final_epoch}")
        else:
            print(f"  ERROR: No common epochs found!")
            return
    
    # Summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(f"  Final epoch: {final_epoch}")
    print(f"  Final train_pred_mse: {final_train_row['pred_mse']:.6e}")
    print(f"  Final test_pred_mse: {final_test_row['pred_mse']:.6e}")
    print(f"  Final param_mse: {final_test_row['param_mse']:.6e}")
    
    # Minimum train_pred_mse
    min_train_pred_mse_all = train_df["pred_mse"].min()
    min_train_pred_mse_idx = train_df["pred_mse"].idxmin()
    min_train_pred_mse_epoch = train_df.loc[min_train_pred_mse_idx, "epoch"]
    print(f"\n  Minimum train_pred_mse (overall): {min_train_pred_mse_all:.6e} at epoch {min_train_pred_mse_epoch}")
    
    # Minimum in last 20 train evaluations
    last_20_train = train_df.tail(20)
    if len(last_20_train) > 0:
        min_train_pred_mse_last20 = last_20_train["pred_mse"].min()
        min_train_pred_mse_last20_idx = last_20_train["pred_mse"].idxmin()
        min_train_pred_mse_last20_epoch = last_20_train.loc[min_train_pred_mse_last20_idx, "epoch"]
        print(f"  Minimum train_pred_mse (last 20): {min_train_pred_mse_last20:.6e} at epoch {min_train_pred_mse_last20_epoch}")
    else:
        print(f"  Minimum train_pred_mse (last 20): N/A (less than 20 train evaluations)")
    
    # Pred/param ratio
    pred_param_ratio = final_test_row['pred_mse'] / max(final_test_row['param_mse'], 1e-30)
    print(f"\n  Pred/param ratio at final epoch: {pred_param_ratio:.6e}")
    print(f"    (Expected ~ d/n_test if consistent, where d=inp_dim, n_test=test_size)")
    
    # Last 30 rows of train split
    print(f"\nLAST 30 ROWS - TRAIN SPLIT:")
    print(f"{'Epoch':>10} {'pred_mse':>15} {'param_mse':>15}")
    print("-" * 45)
    last_30_train = train_df.tail(30)
    for _, row in last_30_train.iterrows():
        print(f"{row['epoch']:>10} {row['pred_mse']:>15.6e} {row['param_mse']:>15.6e}")
    
    # Last 30 rows of test split
    print(f"\nLAST 30 ROWS - TEST SPLIT:")
    print(f"{'Epoch':>10} {'pred_mse':>15} {'param_mse':>15}")
    print("-" * 45)
    last_30_test = test_df.tail(30)
    for _, row in last_30_test.iterrows():
        print(f"{row['epoch']:>10} {row['pred_mse']:>15.6e} {row['param_mse']:>15.6e}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze convergence behavior of a single empirical run"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to run directory containing df.feather"
    )
    
    args = parser.parse_args()
    analyze_run(args.run_dir)


if __name__ == "__main__":
    main()




