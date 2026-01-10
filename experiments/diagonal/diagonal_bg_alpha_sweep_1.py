#!/usr/bin/env python3
"""
Diagonal network BG alpha sweep with SLURM array job support.

Runs experiments across alpha values [0.05, 0.1, 0.2, ..., 1.0] with 2 seeds each.
Saves results to CSV for plotting with replica curves.
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

# Alpha values: 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
alpha_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
inp_dim = 1000
n_train_values = [int(alpha * inp_dim) for alpha in alpha_values]

argparse_array = ArgparseArray(
    n_train=n_train_values,  # n_train values corresponding to alpha values (must come before seed)
    seed=[0, 1],  # 2 seeds per alpha
    inp_dim=[inp_dim],
    rho=[0.04],
    c=[0.001],
    lr=[0.5],
    threshold=[1e-12],
    epochs=[5000000],
    n_test=[10000],
    test_every_n_epochs=[200],
    scaling=[1.0],
    lmda=[0.0],
    init_method=['complex'],
    no_tuning=[True],
    
    # Save folder naming (compute alpha from n_train and inp_dim)
    save_folder=(lambda n_train, seed, rho, c, lr, inp_dim, **kwargs:
                 f"results/diagonal/bg_experiments/"
                 f"alpha={n_train/inp_dim:.6f}--n_train={n_train}--seed={seed}--"
                 f"rho={rho:.6f}--c={c:.6f}--lr={lr:.1f}/"),
)


def extract_and_save_results(save_folder, alpha, n_train, seed):
    """Extract final metrics from results_meta.json and append to CSV."""
    meta_path = os.path.join(save_folder, 'results_meta.json')
    
    if not os.path.exists(meta_path):
        print(f"WARNING: results_meta.json not found at {meta_path}")
        return False
    
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Extract metrics
        row = {
            'alpha': alpha,
            'n_train': n_train,
            'seed': seed,
            'test_pred_mse': meta.get('final_test_pred_mse', None),
            'param_mse': meta.get('final_param_mse', None),
            'train_pred_mse': meta.get('final_train_pred_mse', None),
            'final_epoch': meta.get('final_epoch', None),
            'stop_reason': meta.get('stop_reason', None),
            'final_grad_norm': meta.get('final_grad_norm', None),
            'final_beta_update_rate': meta.get('final_beta_update_rate', None),
            'save_folder': save_folder,
        }
        
        # Append to CSV
        csv_path = os.path.abspath('experiment_results_bg_alpha_sweep.csv')
        success = safe_csv_append(csv_path, row)
        
        if success:
            print(f"Results appended to {csv_path}")
            return True
        else:
            print(f"ERROR: Failed to append results to {csv_path}")
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to extract results from {meta_path}: {e}")
        return False


def main(args):
    import sys as _sys
    
    # Get resolved arguments for this array_id
    resolved_args = argparse_array.get_args(args.array_id)
    
    print('='*80)
    print('BG Alpha Sweep - Experiment Parameters:')
    print('='*80)
    for key in sorted(resolved_args.keys()):
        if not key.startswith('aux_'):
            print(f"  {key}: {resolved_args[key]}")
    print('='*80)
    
    # Extract parameters for later use
    n_train = resolved_args['n_train']
    inp_dim = resolved_args['inp_dim']
    seed = resolved_args['seed']
    save_folder = resolved_args['save_folder']
    
    # Compute alpha from n_train and inp_dim
    alpha = n_train / inp_dim
    
    # Run the training script
    print(f"\nRunning training script...")
    argparse_array.call_script(
        'experiments/diagonal/diagonal_network_pretrain_bg.py',
        args.array_id,
        python_cmd=_sys.executable
    )
    
    # Extract and save results to CSV
    print(f"\nExtracting results and saving to CSV...")
    success = extract_and_save_results(save_folder, alpha, n_train, seed)
    
    if success:
        print(f"\n✓ Experiment {args.array_id} completed successfully")
    else:
        print(f"\n✗ Experiment {args.array_id} completed with warnings")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int, help='SLURM array task ID')
    main(parser.parse_args())

