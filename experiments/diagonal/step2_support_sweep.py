#!/usr/bin/env python3
"""
Step 2 Support Sweep: Run empirical experiments with support-conditioned c initialization.

Runs two cases:
- Case A (good): c_nz=0.001, c_z=0.5 (small k on support, large off)
- Case B (bad): c_nz=0.5, c_z=0.001 (reversed)

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
INP_DIM = 1000

# Two cases: good assignment vs bad assignment
# Use auxiliary index to handle paired c_nz/c_z values
C_NZ_VALUES = [0.001, 0.5]    # Case 0 (good), Case 1 (bad)
C_Z_VALUES = [0.5, 0.001]     # Paired with above

# Alpha values matching replica grid
alpha_values = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
n_train_values = [int(alpha * INP_DIM) for alpha in alpha_values]

# Seeds for each run
SEEDS = [0, 1, 2]

# ArgparseArray uses Cartesian product: 2 cases × 13 alpha × 3 seeds = 78 jobs
argparse_array = ArgparseArray(
    aux_case_idx=[0, 1],  # Auxiliary index for case (not passed to script)
    n_train=n_train_values,
    seed=SEEDS,
    c_nz=(lambda case_idx, **kwargs: C_NZ_VALUES[case_idx]),
    c_z=(lambda case_idx, **kwargs: C_Z_VALUES[case_idx]),
    inp_dim=[INP_DIM],
    rho=[RHO],
    c=[0.001],  # Default c (not used in support mode)
    c_mode=['support'],
    lr=[0.5],
    threshold=[1e-12],
    epochs=[5000000],
    n_test=[10000],
    test_every_n_epochs=[200],
    scaling=[1.0],
    lmda=[0.0],
    init_method=['complex'],
    no_tuning=[True],
    
    # Save folder naming - use case_idx directly to avoid lambda dependency issue
    save_folder=(lambda n_train, seed, case_idx, **kwargs:
                 f"results/diagonal/step2_support/"
                 f"c_nz={C_NZ_VALUES[case_idx]:.6f}--c_z={C_Z_VALUES[case_idx]:.6f}--alpha={n_train/INP_DIM:.6f}--seed={seed}/"),
)


def get_case_name(c_nz, c_z):
    """Determine case name from c values."""
    if c_nz < c_z:
        return 'good'
    else:
        return 'bad'


def extract_and_save_results(save_folder, alpha, n_train, seed, c_nz, c_z):
    """Extract final metrics and append to CSV."""
    meta_path = os.path.join(save_folder, 'results_meta.json')
    
    if not os.path.exists(meta_path):
        print(f"WARNING: results_meta.json not found at {meta_path}")
        return False
    
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        case_name = get_case_name(c_nz, c_z)
        
        row = {
            'alpha': alpha,
            'n_train': n_train,
            'seed': seed,
            'case_name': case_name,
            'c_nz': c_nz,
            'c_z': c_z,
            'rho': RHO,
            'test_pred_mse': meta.get('final_test_pred_mse', None),
            'param_mse': meta.get('final_param_mse', None),
            'train_pred_mse': meta.get('final_train_pred_mse', None),
            'final_epoch': meta.get('final_epoch', None),
            'stop_reason': meta.get('stop_reason', None),
            'save_folder': save_folder,
        }
        
        csv_path = os.path.abspath('experiment_results_step2_support.csv')
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
    print('Step 2 Support Sweep - Experiment Parameters:')
    print('='*80)
    for key in sorted(resolved_args.keys()):
        if not key.startswith('aux_'):
            print(f"  {key}: {resolved_args[key]}")
    print('='*80)
    
    n_train = resolved_args['n_train']
    inp_dim = resolved_args['inp_dim']
    seed = resolved_args['seed']
    c_nz = resolved_args['c_nz']
    c_z = resolved_args['c_z']
    save_folder = resolved_args['save_folder']
    
    alpha = n_train / inp_dim
    case_name = get_case_name(c_nz, c_z)
    
    print(f"\nCase: {case_name} (c_nz={c_nz}, c_z={c_z})")
    
    print(f"\nRunning training script...")
    argparse_array.call_script(
        'experiments/diagonal/diagonal_network_pretrain_bg.py',
        args.array_id,
        python_cmd=_sys.executable
    )
    
    print(f"\nExtracting results and saving to CSV...")
    success = extract_and_save_results(save_folder, alpha, n_train, seed, c_nz, c_z)
    
    if success:
        print(f"\n✓ Experiment {args.array_id} completed successfully")
    else:
        print(f"\n✗ Experiment {args.array_id} completed with warnings")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int, help='SLURM array task ID')
    
    # Print total number of jobs for reference
    total_jobs = len(C_NZ_VALUES) * len(alpha_values) * len(SEEDS)
    print(f"Total jobs in this sweep: {total_jobs}")
    print(f"Cases: good (c_nz={C_NZ_VALUES[0]}, c_z={C_Z_VALUES[0]}), bad (c_nz={C_NZ_VALUES[1]}, c_z={C_Z_VALUES[1]})")
    print(f"Alpha values: {alpha_values}")
    print(f"Seeds: {SEEDS}")
    
    main(parser.parse_args())

