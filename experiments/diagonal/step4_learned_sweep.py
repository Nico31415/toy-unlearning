#!/usr/bin/env python3
"""
Step 4 Learned PT Sweep: Run empirical PT+FT experiments with learned PT.

Compares learned PT initialization vs oracle for various omega and alpha values.
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
RHO_PT = 0.10
RHO_FT = 0.04
A_PT = 1.0
C_PT = 0.001
LAMBDA_PT = 0.0
GAMMA_REINIT = 0.0
INP_DIM = 1000

# Omega values to test
OMEGA_VALUES = [0.0, 0.5, 1.0]

# Alpha values for FT (PT uses fixed high alpha for good learning)
alpha_ft_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
n_train_ft_values = [int(alpha * INP_DIM) for alpha in alpha_ft_values]

# PT uses high alpha (alpha_pt = 2.0) for good learning
ALPHA_PT = 2.0
N_TRAIN_PT = int(ALPHA_PT * INP_DIM)

# Seeds
SEEDS = [0, 1, 2]

# ArgparseArray uses Cartesian product: 3 omega × 6 alpha_ft × 3 seeds = 54 jobs
argparse_array = ArgparseArray(
    omega=OMEGA_VALUES,
    n_train_ft=n_train_ft_values,
    seed=SEEDS,
    n_train_pt=[N_TRAIN_PT],
    inp_dim=[INP_DIM],
    rho_pt=[RHO_PT],
    rho_ft=[RHO_FT],
    a_pt=[A_PT],
    c_pt=[C_PT],
    lambda_pt=[LAMBDA_PT],
    gamma_reinit=[GAMMA_REINIT],
    lr_pt=[0.5],
    lr_ft=[0.5],
    epochs_pt=[5000000],
    epochs_ft=[5000000],
    threshold=[1e-12],
    n_test=[10000],
    test_every_n_epochs=[200],
    no_tuning=[True],
    
    # Save folder naming
    save_folder=(lambda n_train_ft, seed, omega, **kwargs:
                 f"results/diagonal/step4_learned/"
                 f"omega={omega:.2f}--alpha_ft={n_train_ft/INP_DIM:.6f}--seed={seed}/"),
)


def extract_and_save_results(save_folder, alpha_ft, n_train_ft, seed, omega):
    """Extract final metrics and append to CSV."""
    config_path = os.path.join(save_folder, 'config.json')
    
    if not os.path.exists(config_path):
        print(f"WARNING: config.json not found at {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        row = {
            'alpha_ft': alpha_ft,
            'n_train_ft': n_train_ft,
            'alpha_pt': ALPHA_PT,
            'n_train_pt': N_TRAIN_PT,
            'seed': seed,
            'omega': omega,
            'empirical_omega': config.get('empirical_omega', omega),
            'rho_pt': RHO_PT,
            'rho_ft': RHO_FT,
            'a_pt': A_PT,
            'c_pt': C_PT,
            'lambda_pt': LAMBDA_PT,
            'gamma_reinit': GAMMA_REINIT,
            'pt_param_mse': config.get('pt_param_mse', None),
            'ft_param_mse_learned': config.get('ft_param_mse_learned', None),
            'ft_param_mse_oracle': config.get('ft_param_mse_oracle', None),
            'c_ft_correlation': config.get('c_ft_correlation', None),
            'c_ft_mean_diff': config.get('c_ft_mean_diff', None),
            'save_folder': save_folder,
        }
        
        csv_path = os.path.abspath('experiment_results_step4_learned.csv')
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
    print('Step 4 Learned PT Sweep - Experiment Parameters:')
    print('='*80)
    for key in sorted(resolved_args.keys()):
        if not key.startswith('aux_'):
            print(f"  {key}: {resolved_args[key]}")
    print('='*80)
    
    n_train_ft = resolved_args['n_train_ft']
    inp_dim = resolved_args['inp_dim']
    seed = resolved_args['seed']
    omega = resolved_args['omega']
    save_folder = resolved_args['save_folder']
    
    alpha_ft = n_train_ft / inp_dim
    
    print(f"\nOmega: {omega}, Alpha_FT: {alpha_ft:.4f}, Alpha_PT: {ALPHA_PT}")
    
    print(f"\nRunning PT+FT Learned script...")
    argparse_array.call_script(
        'experiments/diagonal/diagonal_ptft_learned.py',
        args.array_id,
        python_cmd=_sys.executable
    )
    
    print(f"\nExtracting results and saving to CSV...")
    success = extract_and_save_results(save_folder, alpha_ft, n_train_ft, seed, omega)
    
    if success:
        print(f"\n✓ Experiment {args.array_id} completed successfully")
    else:
        print(f"\n✗ Experiment {args.array_id} completed with warnings")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int, help='SLURM array task ID')
    
    # Print total number of jobs for reference
    total_jobs = len(OMEGA_VALUES) * len(alpha_ft_values) * len(SEEDS)
    print(f"Total jobs in this sweep: {total_jobs}")
    print(f"Omega values: {OMEGA_VALUES}")
    print(f"Alpha FT values: {alpha_ft_values}")
    print(f"Alpha PT: {ALPHA_PT}")
    print(f"Seeds: {SEEDS}")
    
    main(parser.parse_args())

