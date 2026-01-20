#!/usr/bin/env python3
"""
Regenerate plots from existing aggregated CSV files.

This script loads aggregated CSV files and regenerates plots using the updated
plot_generalization function that uses parameter MSE.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import plot_generalization
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.diagonal.plot_generalization_bg import plot_generalization
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate plots from aggregated CSV files"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to aggregated CSV file (e.g., aggregated_results_rho=0.040000--c=0.001000--ALL.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same directory as CSV file)",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="Rho value (default: extract from filename)",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=None,
        help="C value (default: extract from filename)",
    )
    
    args = parser.parse_args()
    
    # Load CSV
    if not os.path.exists(args.csv_file):
        print(f"ERROR: CSV file not found: {args.csv_file}")
        return
    
    print(f"Loading CSV: {args.csv_file}")
    agg_df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(agg_df)} rows")
    
    # Extract rho and c from filename if not provided
    if args.rho is None or args.c is None:
        filename = os.path.basename(args.csv_file)
        # Try to extract from filename like "aggregated_results_rho=0.040000--c=0.001000--ALL.csv"
        import re
        rho_match = re.search(r'rho=([\d.]+)', filename)
        c_match = re.search(r'c=([\d.]+)', filename)
        
        if rho_match:
            args.rho = float(rho_match.group(1))
        if c_match:
            args.c = float(c_match.group(1))
    
    if args.rho is None or args.c is None:
        print("ERROR: Could not extract rho and c from filename. Please provide --rho and --c explicitly.")
        return
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.csv_file))
    
    print(f"Rho: {args.rho:.6f}")
    print(f"C: {args.c:.6f}")
    print(f"Output directory: {args.output_dir}")
    
    # Regenerate plot
    plot_generalization(agg_df, args.output_dir, args.rho, c=args.c)
    
    print("\nDone!")


if __name__ == "__main__":
    main()







