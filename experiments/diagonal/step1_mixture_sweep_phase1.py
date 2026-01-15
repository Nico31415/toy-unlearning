#!/usr/bin/env python3
"""
Step 1 Mixture Sweep (Phase 1): Run empirical experiments with mixture-k heterogeneous initialization.

PHASE 1 VERSION: Uses updated stopping logic, writes to new output locations.
- Results folder: results/diagonal_phase1/step1_mixture/
- CSV: experiment_results_step1_mixture_phase1.csv

Sweeps over pi_A values {0.1, 0.5, 0.9} with c_A=0.001, c_B=0.5 at rho=0.04.
Uses SLURM array jobs for parallelization.
"""

import argparse
import sys
import os
import json
import time
import random
import fcntl
from pathlib import Path

import pandas as pd

sys.path.append('')

from functions.array_training import ArgparseArray


def safe_csv_append(csv_path: str, new_row_data: dict, max_retries: int = 5, base_delay: float = 0.1) -> bool:
    """Safely append a row to CSV file with file locking."""
    lock_file_path = f"{csv_path}.lock"
    for attempt in range(max_retries):
        try:
            with open(lock_file_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if os.path.exists(csv_path):
                    try:
                        existing_df = pd.read_csv(csv_path)
                    except Exception:
                        existing_df = pd.DataFrame()
                else:
                    existing_df = pd.DataFrame()

                new_df = pd.DataFrame([new_row_data])
                all_columns = sorted(set(list(existing_df.columns) + list(new_df.columns)))
                existing_df = existing_df.reindex(columns=all_columns)
                new_df = new_df.reindex(columns=all_columns)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined.to_csv(csv_path, index=False)
                return True
        except (BlockingIOError, OSError):
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                time.sleep(delay)
                continue
            return False
        except Exception:
            return False
    return False


# Configuration
RHO = 0.04
C_A = 0.001
C_B = 0.5
PI_A_VALUES = [0.1, 0.5, 0.9]
INP_DIM = 1000

# Alpha values matching replica grid
alpha_values = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
n_train_values = [int(alpha * INP_DIM) for alpha in alpha_values]

# Seeds for each run
SEEDS = [0, 1, 2]

# ArgparseArray uses Cartesian product: 3 pi_A × 12 alpha × 3 seeds = 108 jobs
argparse_array = ArgparseArray(
    pi_A=PI_A_VALUES,
    n_train=n_train_values,
    seed=SEEDS,
    inp_dim=[INP_DIM],
    rho=[RHO],
    c=[C_A],  # Default c (used for homogeneous fallback)
    c_A=[C_A],
    c_B=[C_B],
    c_mode=['mixture'],
    lr=[0.5],
    threshold=[1e-12],
    epochs=[5000000],
    n_test=[10000],
    test_every_n_epochs=[200],
    scaling=[1.0],
    lmda=[0.0],
    init_method=['complex'],
    no_tuning=[True],
    
    # PHASE 1: Updated save folder prefix
    save_folder=(lambda n_train, seed, pi_A, **kwargs:
                 f"results/diagonal_phase1/step1_mixture/"
                 f"pi_A={pi_A:.2f}--alpha={n_train/INP_DIM:.6f}--seed={seed}/"),
)


def extract_and_save_results(save_folder, alpha, n_train, seed, pi_A):
    """Extract final metrics and append to CSV."""
    meta_path = os.path.join(save_folder, 'results_meta.json')
    
    if not os.path.exists(meta_path):
        print(f"WARNING: results_meta.json not found at {meta_path}")
        return False
    
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        row = {
            'alpha': alpha,
            'n_train': n_train,
            'seed': seed,
            'pi_A': pi_A,
            'c_A': C_A,
            'c_B': C_B,
            'rho': RHO,
            'test_pred_mse': meta.get('final_test_pred_mse', None),
            'param_mse': meta.get('final_param_mse', None),
            'train_pred_mse': meta.get('final_train_pred_mse', None),
            'final_epoch': meta.get('final_epoch', None),
            'stop_reason': meta.get('stop_reason', None),
            'save_folder': save_folder,
        }
        
        # PHASE 1: Updated CSV path
        csv_path = os.path.abspath('experiment_results_step1_mixture_phase1.csv')
        success = safe_csv_append(csv_path, row)
        
        if success:
            print(f"Results appended to {csv_path}")
            return True
        else:
            print(f"ERROR: Failed to append results to {csv_path}")
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to extract results: {e}")
        return False


def main(args):
    import sys as _sys
    
    resolved_args = argparse_array.get_args(args.array_id)
    
    print('='*80)
    print('Step 1 Mixture Sweep (PHASE 1) - Experiment Parameters:')
    print('='*80)
    for key in sorted(resolved_args.keys()):
        if not key.startswith('aux_'):
            print(f"  {key}: {resolved_args[key]}")
    print('='*80)
    
    n_train = resolved_args['n_train']
    inp_dim = resolved_args['inp_dim']
    seed = resolved_args['seed']
    pi_A = resolved_args['pi_A']
    save_folder = resolved_args['save_folder']
    
    alpha = n_train / inp_dim
    
    print(f"\nRunning training script...")
    argparse_array.call_script(
        'experiments/diagonal/diagonal_network_pretrain_bg.py',
        args.array_id,
        python_cmd=_sys.executable
    )
    
    print(f"\nExtracting results and saving to CSV...")
    success = extract_and_save_results(save_folder, alpha, n_train, seed, pi_A)
    
    if success:
        print(f"\n✓ Experiment {args.array_id} completed successfully")
    else:
        print(f"\n✗ Experiment {args.array_id} completed with warnings")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int, help='SLURM array task ID')
    
    # Print total number of jobs for reference
    total_jobs = len(PI_A_VALUES) * len(alpha_values) * len(SEEDS)
    print(f"[PHASE 1] Total jobs in this sweep: {total_jobs}")
    print(f"Pi_A values: {PI_A_VALUES}")
    print(f"Alpha values: {alpha_values}")
    print(f"Seeds: {SEEDS}")
    
    main(parser.parse_args())



