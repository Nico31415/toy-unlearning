#!/usr/bin/env python3
"""
Run empirical experiments for configurations missing empirical_results.csv.
"""

import sys
from pathlib import Path
import os
import json

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent.parent))

import pandas as pd

EXP_BASE = Path("figures/panel_experiments")

# Fixed params
RHO_PT = 0.1
FT_REGULARISER_SCALE = 1e-6
A_PT = 1.0
INP_DIM = 500
N_TEST = 2000

# Alpha values to sweep
ALPHA_VALUES = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def find_missing_empirical_dirs():
    """Find all experiment directories with missing/incomplete empirical_results.csv."""
    missing = []
    
    for exp_dir in sorted(EXP_BASE.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        # Find the ptft_oracle subdirectory
        subdirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith('ptft_oracle')]
        if not subdirs:
            continue
        
        subdir = subdirs[0]
        emp_csv = subdir / "empirical_results.csv"
        
        # Check if empirical results are missing or incomplete
        if not emp_csv.exists():
            missing.append((exp_dir, subdir))
        else:
            # Check if file has enough data (header + at least some results)
            with open(emp_csv) as f:
                lines = f.readlines()
            # Need at least header + 36 data rows (12 alphas * 3 seeds)
            if len(lines) < 20:
                missing.append((exp_dir, subdir))
            else:
                # Check if any non-NaN values in param_mse
                df = pd.read_csv(emp_csv)
                if df['param_mse'].isna().all():
                    missing.append((exp_dir, subdir))
    
    return missing


def parse_params_from_dirname(dirname: str) -> dict:
    """Parse parameters from directory name."""
    params = {}
    for part in dirname.split("__"):
        if "=" in part:
            key, val = part.split("=", 1)
            params[key] = float(val)
    return params


def run_single_config(exp_dir: Path, subdir: Path, params: dict, num_seeds: int = 3):
    """Run empirical experiment for a single config."""
    # Import here to avoid torch import issues when just listing
    from experiments.diagonal.diagonal_ptft_oracle import get_parser as ptft_get_parser
    from experiments.diagonal.diagonal_ptft_oracle import main as ptft_main
    
    rho_ft = params['rft']
    omega = params['om']
    c_pt = params['cpt']
    lambda_pt = params['lpt']
    gamma_reinit = params['gam']
    
    print(f"  Running empirical: rft={rho_ft}, om={omega}, cpt={c_pt}, lpt={lambda_pt}, gam={gamma_reinit}")
    
    emp_runs_dir = subdir / "empirical_runs"
    emp_runs_dir.mkdir(parents=True, exist_ok=True)
    
    parser = ptft_get_parser()
    results = []
    
    for seed in range(num_seeds):
        for alpha in ALPHA_VALUES:
            n_train = max(1, int(round(alpha * INP_DIM)))
            alpha_eff = n_train / float(INP_DIM)
            
            save_folder = emp_runs_dir / f"omega={omega:.3f}__alpha={alpha_eff:.6f}__seed={seed}"
            save_folder.mkdir(parents=True, exist_ok=True)
            
            meta_path = save_folder / "results_meta.json"
            
            try:
                # Build argv list
                argv = [
                    "--seed", str(seed),
                    "--save_folder", str(save_folder),
                    "--inp_dim", str(INP_DIM),
                    "--n_train", str(n_train),
                    "--n_test", str(N_TEST),
                    "--rho_pt", str(RHO_PT),
                    "--rho_ft", str(rho_ft),
                    "--omega", str(omega),
                    "--a_pt", str(A_PT),
                    "--c_pt", str(c_pt),
                    "--lambda_pt", str(lambda_pt),
                    "--gamma_reinit", str(gamma_reinit),
                    "--lr", "0.01",
                    "--epochs", "50000",
                    "--threshold", "1e-9",
                    "--test_every_n_epochs", "500",
                    "--no_tuning",
                ]
                
                args = parser.parse_args(argv)
                ptft_main(args)
                
                # Load results
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    results.append({
                        'seed': seed,
                        'alpha': alpha_eff,
                        'param_mse': meta.get('final_param_mse', float('nan')),
                        'test_loss': meta.get('final_test_pred_mse', float('nan')),
                        'final_train_loss': meta.get('final_train_pred_mse', float('nan')),
                    })
                    print(f"    seed={seed}, alpha={alpha:.2f}: MSE={meta.get('final_param_mse', float('nan')):.2e}")
                else:
                    results.append({
                        'seed': seed,
                        'alpha': alpha_eff,
                        'param_mse': float('nan'),
                        'test_loss': float('nan'),
                        'final_train_loss': float('nan'),
                    })
                    print(f"    seed={seed}, alpha={alpha:.2f}: No meta file")
                    
            except Exception as e:
                print(f"    seed={seed}, alpha={alpha:.2f}: FAILED - {e}")
                results.append({
                    'seed': seed,
                    'alpha': alpha_eff,
                    'param_mse': float('nan'),
                    'test_loss': float('nan'),
                    'final_train_loss': float('nan'),
                })
    
    # Save results
    df = pd.DataFrame(results)
    emp_csv = subdir / "empirical_results.csv"
    df.to_csv(emp_csv, index=False)
    
    # Count non-NaN results
    valid_count = df['param_mse'].notna().sum()
    print(f"  ✓ Saved {len(results)} results ({valid_count} valid) to {emp_csv}")
    
    return valid_count > 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', type=int, default=None,
                        help='SLURM array task ID (0-indexed)')
    parser.add_argument('--num-seeds', type=int, default=3)
    parser.add_argument('--list', action='store_true',
                        help='List missing configs and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be run without running')
    args = parser.parse_args()
    
    missing = find_missing_empirical_dirs()
    print(f"Found {len(missing)} experiments with missing/incomplete empirical data")
    
    if args.list:
        for i, (exp_dir, subdir) in enumerate(missing):
            print(f"{i}: {exp_dir.name}")
        return
    
    if args.task_id is not None:
        # SLURM array mode
        if args.task_id >= len(missing):
            print(f"Task ID {args.task_id} >= {len(missing)} missing configs, skipping")
            return
        
        exp_dir, subdir = missing[args.task_id]
        params = parse_params_from_dirname(exp_dir.name)
        print(f"Task {args.task_id}: {exp_dir.name}")
        
        if args.dry_run:
            print(f"  Would run empirical for: {params}")
            return
        
        success = run_single_config(exp_dir, subdir, params, num_seeds=args.num_seeds)
        if success:
            print(f"✓ Completed task {args.task_id}")
        else:
            print(f"✗ Failed task {args.task_id}")
            sys.exit(1)
    else:
        # Local mode: run all
        for i, (exp_dir, subdir) in enumerate(missing):
            params = parse_params_from_dirname(exp_dir.name)
            print(f"\n[{i+1}/{len(missing)}] {exp_dir.name}")
            
            if args.dry_run:
                print(f"  Would run empirical for: {params}")
                continue
            
            run_single_config(exp_dir, subdir, params, num_seeds=args.num_seeds)
        
        print(f"\n✓ All {len(missing)} configs completed")


if __name__ == '__main__':
    main()
