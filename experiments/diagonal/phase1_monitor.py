#!/usr/bin/env python3
"""
Phase 1 Monitor: Track progress of Phase 1 sweep jobs.

Usage:
    python experiments/diagonal/phase1_monitor.py [--watch]
    
Options:
    --watch     Continuously monitor until all jobs complete (poll every 30s)
"""

import os
import sys
import time
import argparse
from pathlib import Path

import pandas as pd

# Expected task counts (must match SLURM array ranges)
EXPECTED_TASKS = {
    1: 108,  # Step 1: 3 pi_A × 12 alpha × 3 seeds
    2: 72,   # Step 2: 2 cases × 12 alpha × 3 seeds
    3: 108,  # Step 3: 3 omega × 12 alpha × 3 seeds
}

# Result directories
RESULT_DIRS = {
    1: "results/diagonal_phase1/step1_mixture",
    2: "results/diagonal_phase1/step2_support",
    3: "results/diagonal_phase1/step3_omega",
}

# CSV files
CSV_FILES = {
    1: "experiment_results_step1_mixture_phase1.csv",
    2: "experiment_results_step2_support_phase1.csv",
    3: "experiment_results_step3_omega_phase1.csv",
}


def count_meta_files(result_dir):
    """Count results_meta.json files in directory."""
    path = Path(result_dir)
    if not path.exists():
        return 0
    count = 0
    for folder in path.iterdir():
        if folder.is_dir():
            meta_path = folder / "results_meta.json"
            if meta_path.exists():
                count += 1
    return count


def count_csv_rows(csv_path):
    """Count rows in CSV file (excluding header)."""
    if not os.path.exists(csv_path):
        return 0
    try:
        df = pd.read_csv(csv_path)
        return len(df)
    except Exception:
        return 0


def get_stop_reason_counts(csv_path):
    """Get stop_reason value counts from CSV."""
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        if 'stop_reason' in df.columns:
            return df['stop_reason'].value_counts().to_dict()
        return {}
    except Exception:
        return {}


def print_status():
    """Print current status of Phase 1 sweeps."""
    print("=" * 70)
    print("PHASE 1 SWEEP MONITOR")
    print("=" * 70)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_complete = True
    
    for step in [1, 2, 3]:
        expected = EXPECTED_TASKS[step]
        result_dir = RESULT_DIRS[step]
        csv_path = CSV_FILES[step]
        
        meta_count = count_meta_files(result_dir)
        csv_count = count_csv_rows(csv_path)
        stop_reasons = get_stop_reason_counts(csv_path)
        
        meta_pct = 100.0 * meta_count / expected
        csv_pct = 100.0 * csv_count / expected
        
        step_complete = (meta_count >= expected) and (csv_count >= expected)
        status = "✓ COMPLETE" if step_complete else "⏳ IN PROGRESS"
        
        if not step_complete:
            all_complete = False
        
        print(f"--- Step {step} {status} ---")
        print(f"  Results folder:  {result_dir}")
        print(f"  Meta files:      {meta_count} / {expected} ({meta_pct:.1f}%)")
        print(f"  CSV rows:        {csv_count} / {expected} ({csv_pct:.1f}%)")
        print(f"  CSV path:        {os.path.abspath(csv_path)}")
        
        if stop_reasons:
            print(f"  Stop reasons:")
            for reason, count in sorted(stop_reasons.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")
        print()
    
    print("=" * 70)
    
    if all_complete:
        print("✓ ALL PHASE 1 SWEEPS COMPLETE!")
        print()
        print("Phase 1 CSV file paths:")
        for step in [1, 2, 3]:
            print(f"  Step {step}: {os.path.abspath(CSV_FILES[step])}")
        print()
        print("Phase 1 result directories:")
        for step in [1, 2, 3]:
            print(f"  Step {step}: {os.path.abspath(RESULT_DIRS[step])}")
    else:
        print("⏳ Some sweeps still in progress...")
    
    print("=" * 70)
    
    return all_complete


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Sweep Monitor")
    parser.add_argument("--watch", action="store_true",
                        help="Continuously monitor until all jobs complete")
    parser.add_argument("--interval", type=int, default=30,
                        help="Poll interval in seconds (default: 30)")
    args = parser.parse_args()
    
    if args.watch:
        print("Starting continuous monitoring (Ctrl+C to stop)...")
        print()
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                all_complete = print_status()
                if all_complete:
                    print("\nMonitoring complete - all jobs finished!")
                    sys.exit(0)
                print(f"\nNext update in {args.interval} seconds...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            sys.exit(0)
    else:
        all_complete = print_status()
        sys.exit(0 if all_complete else 1)


if __name__ == "__main__":
    main()



