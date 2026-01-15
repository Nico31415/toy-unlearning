#!/usr/bin/env python3
"""
Phase 0 Smoke Run: Targeted validation of fixed-point/legacy-stop fix.

This script runs a small, diagnostic set of experiments without using ArgparseArray.
It directly calls the training scripts (diagonal_network_pretrain_bg.py for step1/step2,
diagonal_ptft_oracle.py for step3) and appends results to CSV files.

Usage:
    python experiments/diagonal/phase0_smoke_run.py --phase step1 --task_id 0
    python experiments/diagonal/phase0_smoke_run.py --phase step1 --task_id 0 --dry_run
"""

import argparse
import os
import sys
import json
import time
import random
import fcntl
import subprocess
from pathlib import Path
from itertools import product

import pandas as pd

# ============================================================================
# Constants (matching main sweeps)
# ============================================================================
INP_DIM = 1000
RHO = 0.04  # Step1/Step2 sparsity

# Step1 mixture constants
C_A = 0.001
C_B = 0.5

# Step3 PT+FT constants
RHO_PT = 0.10
RHO_FT = 0.04
A_PT = 1.0
C_PT = 0.001
LAMBDA_PT = 0.0
GAMMA_REINIT = 0.0

# Training params (reduced epochs for smoke test)
TRAINING_PARAMS = {
    'lr': 0.5,
    'epochs': 200000,  # Reduced from 5M
    'threshold': 1e-12,
    'test_every_n_epochs': 200,
    'n_test': 10000,
}

# ============================================================================
# Phase 0 Config Lists
# ============================================================================

def build_step1_configs():
    """
    Step 1 (mixture): pi_A x alpha x seeds
    Total: 3 * 4 * 3 = 36 tasks
    """
    pi_A_values = [0.1, 0.5, 0.9]
    alpha_values = [0.05, 0.10, 0.20, 1.00]
    seeds = [0, 1, 2]
    
    configs = []
    for pi_A, alpha, seed in product(pi_A_values, alpha_values, seeds):
        n_train = int(round(alpha * INP_DIM))
        save_folder = f"results/diagonal/phase0/step1/pi_A={pi_A:.2f}--alpha={alpha:.6f}--seed={seed}/"
        configs.append({
            'pi_A': pi_A,
            'alpha': alpha,
            'n_train': n_train,
            'seed': seed,
            'save_folder': save_folder,
            'c_A': C_A,
            'c_B': C_B,
        })
    return configs


def build_step2_configs():
    """
    Step 2 (support): cases x alpha x seeds
    Cases: bad (c_nz=0.5, c_z=0.001), good (c_nz=0.001, c_z=0.5)
    Total: 2 * 4 * 3 = 24 tasks
    """
    cases = [
        {'name': 'bad', 'c_nz': 0.5, 'c_z': 0.001},
        {'name': 'good', 'c_nz': 0.001, 'c_z': 0.5},
    ]
    alpha_values = [0.05, 0.10, 0.20, 1.00]
    seeds = [0, 1, 2]
    
    configs = []
    for case, alpha, seed in product(cases, alpha_values, seeds):
        n_train = int(round(alpha * INP_DIM))
        save_folder = f"results/diagonal/phase0/step2/case={case['name']}--alpha={alpha:.6f}--seed={seed}/"
        configs.append({
            'case_name': case['name'],
            'c_nz': case['c_nz'],
            'c_z': case['c_z'],
            'alpha': alpha,
            'n_train': n_train,
            'seed': seed,
            'save_folder': save_folder,
        })
    return configs


def build_step3_configs():
    """
    Step 3 (omega): omega x alpha x seeds
    Total: 2 * 2 * 3 = 12 tasks
    """
    omega_values = [0.0, 1.0]
    alpha_values = [0.20, 1.00]
    seeds = [0, 1, 2]
    
    configs = []
    for omega, alpha, seed in product(omega_values, alpha_values, seeds):
        n_train = int(round(alpha * INP_DIM))
        save_folder = f"results/diagonal/phase0/step3/omega={omega:.2f}--alpha={alpha:.6f}--seed={seed}/"
        configs.append({
            'omega': omega,
            'alpha': alpha,
            'n_train': n_train,
            'seed': seed,
            'save_folder': save_folder,
            'rho_pt': RHO_PT,
            'rho_ft': RHO_FT,
            'a_pt': A_PT,
            'c_pt': C_PT,
            'lambda_pt': LAMBDA_PT,
            'gamma_reinit': GAMMA_REINIT,
        })
    return configs


# ============================================================================
# safe_csv_append (copied from sweep scripts to avoid import issues)
# ============================================================================

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
        except Exception as e:
            print(f"ERROR in safe_csv_append: {e}")
            return False
    return False


# ============================================================================
# Command builders
# ============================================================================

def build_step1_command(config, python_cmd):
    """Build command for step1 (mixture) using diagonal_network_pretrain_bg.py"""
    cmd = [
        python_cmd,
        'experiments/diagonal/diagonal_network_pretrain_bg.py',
        '--seed', str(config['seed']),
        '--save_folder', config['save_folder'],
        '--n_train', str(config['n_train']),
        '--n_test', str(TRAINING_PARAMS['n_test']),
        '--inp_dim', str(INP_DIM),
        '--rho', str(RHO),
        '--lr', str(TRAINING_PARAMS['lr']),
        '--epochs', str(TRAINING_PARAMS['epochs']),
        '--threshold', str(TRAINING_PARAMS['threshold']),
        '--test_every_n_epochs', str(TRAINING_PARAMS['test_every_n_epochs']),
        '--c', str(C_A),  # Default c
        '--c_mode', 'mixture',
        '--c_A', str(config['c_A']),
        '--c_B', str(config['c_B']),
        '--pi_A', str(config['pi_A']),
        '--init_method', 'complex',
        '--lmda', '0.0',
        '--scaling', '1.0',
        '--no_tuning',
    ]
    return cmd


def build_step2_command(config, python_cmd):
    """Build command for step2 (support) using diagonal_network_pretrain_bg.py"""
    cmd = [
        python_cmd,
        'experiments/diagonal/diagonal_network_pretrain_bg.py',
        '--seed', str(config['seed']),
        '--save_folder', config['save_folder'],
        '--n_train', str(config['n_train']),
        '--n_test', str(TRAINING_PARAMS['n_test']),
        '--inp_dim', str(INP_DIM),
        '--rho', str(RHO),
        '--lr', str(TRAINING_PARAMS['lr']),
        '--epochs', str(TRAINING_PARAMS['epochs']),
        '--threshold', str(TRAINING_PARAMS['threshold']),
        '--test_every_n_epochs', str(TRAINING_PARAMS['test_every_n_epochs']),
        '--c', str(0.001),  # Default c (unused)
        '--c_mode', 'support',
        '--c_nz', str(config['c_nz']),
        '--c_z', str(config['c_z']),
        '--init_method', 'complex',
        '--lmda', '0.0',
        '--scaling', '1.0',
        '--no_tuning',
    ]
    return cmd


def build_step3_command(config, python_cmd):
    """Build command for step3 (omega) using diagonal_ptft_oracle.py"""
    cmd = [
        python_cmd,
        'experiments/diagonal/diagonal_ptft_oracle.py',
        '--seed', str(config['seed']),
        '--save_folder', config['save_folder'],
        '--n_train', str(config['n_train']),
        '--n_test', str(TRAINING_PARAMS['n_test']),
        '--inp_dim', str(INP_DIM),
        '--rho_pt', str(config['rho_pt']),
        '--rho_ft', str(config['rho_ft']),
        '--omega', str(config['omega']),
        '--a_pt', str(config['a_pt']),
        '--c_pt', str(config['c_pt']),
        '--lambda_pt', str(config['lambda_pt']),
        '--gamma_reinit', str(config['gamma_reinit']),
        '--lr', str(TRAINING_PARAMS['lr']),
        '--epochs', str(TRAINING_PARAMS['epochs']),
        '--threshold', str(TRAINING_PARAMS['threshold']),
        '--test_every_n_epochs', str(TRAINING_PARAMS['test_every_n_epochs']),
        '--no_tuning',
    ]
    return cmd


# ============================================================================
# CSV output helpers
# ============================================================================

def extract_meta_fields(meta: dict) -> dict:
    """Extract fields from results_meta.json for CSV row."""
    return {
        'stop_reason': meta.get('stop_reason'),
        'final_epoch': meta.get('final_epoch'),
        'final_train_pred_mse': meta.get('final_train_pred_mse'),
        'final_test_pred_mse': meta.get('final_test_pred_mse'),
        'final_param_mse': meta.get('final_param_mse'),
        'final_grad_norm': meta.get('final_grad_norm'),
        'final_beta_update_rate': meta.get('final_beta_update_rate'),
        'eval_count': meta.get('eval_count'),
        'min_epochs_before_stop': meta.get('min_epochs_before_stop'),
        'fixed_point_beta_rate': meta.get('fixed_point_beta_rate'),
        'fixed_point_consecutive_evals': meta.get('fixed_point_consecutive_evals'),
        'legacy_loss_stop_disabled': meta.get('legacy_loss_stop_disabled'),
    }


def build_step1_row(config, meta):
    """Build CSV row for step1."""
    row = {
        'phase': 'step1',
        'alpha': config['alpha'],
        'n_train': config['n_train'],
        'seed': config['seed'],
        'save_folder': config['save_folder'],
        'pi_A': config['pi_A'],
        'c_A': config['c_A'],
        'c_B': config['c_B'],
    }
    if meta is not None:
        row.update(extract_meta_fields(meta))
    else:
        row['error'] = 'results_meta.json not found'
    return row


def build_step2_row(config, meta):
    """Build CSV row for step2."""
    row = {
        'phase': 'step2',
        'alpha': config['alpha'],
        'n_train': config['n_train'],
        'seed': config['seed'],
        'save_folder': config['save_folder'],
        'case_name': config['case_name'],
        'c_nz': config['c_nz'],
        'c_z': config['c_z'],
    }
    if meta is not None:
        row.update(extract_meta_fields(meta))
    else:
        row['error'] = 'results_meta.json not found'
    return row


def build_step3_row(config, meta):
    """Build CSV row for step3."""
    row = {
        'phase': 'step3',
        'alpha': config['alpha'],
        'n_train': config['n_train'],
        'seed': config['seed'],
        'save_folder': config['save_folder'],
        'omega': config['omega'],
        'rho_pt': config['rho_pt'],
        'rho_ft': config['rho_ft'],
        'a_pt': config['a_pt'],
        'c_pt': config['c_pt'],
        'lambda_pt': config['lambda_pt'],
        'gamma_reinit': config['gamma_reinit'],
    }
    if meta is not None:
        row.update(extract_meta_fields(meta))
    else:
        row['error'] = 'results_meta.json not found'
    return row


# ============================================================================
# Main execution
# ============================================================================

def run_experiment(phase, task_id, dry_run=False):
    """Run a single experiment based on phase and task_id."""
    # Get config list for this phase
    if phase == 'step1':
        configs = build_step1_configs()
        build_cmd = build_step1_command
        build_row = build_step1_row
        csv_filename = 'phase0_step1.csv'
    elif phase == 'step2':
        configs = build_step2_configs()
        build_cmd = build_step2_command
        build_row = build_step2_row
        csv_filename = 'phase0_step2.csv'
    elif phase == 'step3':
        configs = build_step3_configs()
        build_cmd = build_step3_command
        build_row = build_step3_row
        csv_filename = 'phase0_step3.csv'
    else:
        raise ValueError(f"Unknown phase: {phase}")
    
    # Validate task_id
    if task_id < 0 or task_id >= len(configs):
        raise ValueError(f"task_id {task_id} out of range [0, {len(configs)-1}] for phase {phase}")
    
    config = configs[task_id]
    python_cmd = sys.executable
    cmd = build_cmd(config, python_cmd)
    
    print("=" * 80)
    print(f"Phase 0 Smoke Run - {phase.upper()}")
    print("=" * 80)
    print(f"Task ID: {task_id} / {len(configs) - 1}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"Save folder: {config['save_folder']}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    
    if dry_run:
        print("\n[DRY RUN] Would execute the command above.")
        print(f"[DRY RUN] CSV would be written to: {csv_filename}")
        return
    
    # Create save folder
    Path(config['save_folder']).mkdir(parents=True, exist_ok=True)
    
    # Run subprocess
    print("\nRunning training script...")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n[WARNING] Training script exited with code {result.returncode}")
    
    # Read results_meta.json
    meta_path = os.path.join(config['save_folder'], 'results_meta.json')
    meta = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            print(f"\nLoaded results_meta.json from {meta_path}")
        except Exception as e:
            print(f"\n[ERROR] Failed to load results_meta.json: {e}")
    else:
        print(f"\n[WARNING] results_meta.json not found at {meta_path}")
    
    # Build CSV row
    row = build_row(config, meta)
    
    # Append to CSV (in repo root)
    csv_path = os.path.abspath(csv_filename)
    success = safe_csv_append(csv_path, row)
    
    if success:
        print(f"\n✓ Results appended to {csv_path}")
    else:
        print(f"\n✗ Failed to append results to {csv_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    if meta:
        print(f"Stop reason: {meta.get('stop_reason', 'N/A')}")
        print(f"Final epoch: {meta.get('final_epoch', 'N/A')}")
        print(f"Final train pred MSE: {meta.get('final_train_pred_mse', 'N/A')}")
        print(f"Final test pred MSE: {meta.get('final_test_pred_mse', 'N/A')}")
        print(f"Final param MSE: {meta.get('final_param_mse', 'N/A')}")
        print(f"Final beta update rate: {meta.get('final_beta_update_rate', 'N/A')}")
        print(f"Eval count: {meta.get('eval_count', 'N/A')}")
    else:
        print("No results available (results_meta.json not found)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 Smoke Run for fixed-point/legacy-stop validation"
    )
    parser.add_argument('--task_id', type=int, required=True,
                        help='Task ID (SLURM_ARRAY_TASK_ID)')
    parser.add_argument('--phase', type=str, required=True,
                        choices=['step1', 'step2', 'step3'],
                        help='Which phase to run')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print command and exit without running')
    
    args = parser.parse_args()
    
    # Print config summary
    print("\n" + "=" * 80)
    print("PHASE 0 SMOKE RUN CONFIG SUMMARY")
    print("=" * 80)
    
    step1_configs = build_step1_configs()
    step2_configs = build_step2_configs()
    step3_configs = build_step3_configs()
    
    print(f"Step1 (mixture):  {len(step1_configs)} tasks")
    print(f"  pi_A in [0.1, 0.5, 0.9]")
    print(f"  alpha in [0.05, 0.10, 0.20, 1.00]")
    print(f"  seeds in [0, 1, 2]")
    print()
    print(f"Step2 (support):  {len(step2_configs)} tasks")
    print(f"  cases: bad (c_nz=0.5, c_z=0.001), good (c_nz=0.001, c_z=0.5)")
    print(f"  alpha in [0.05, 0.10, 0.20, 1.00]")
    print(f"  seeds in [0, 1, 2]")
    print()
    print(f"Step3 (omega):    {len(step3_configs)} tasks")
    print(f"  omega in [0.0, 1.0]")
    print(f"  alpha in [0.20, 1.00]")
    print(f"  seeds in [0, 1, 2]")
    print("=" * 80 + "\n")
    
    run_experiment(args.phase, args.task_id, args.dry_run)


if __name__ == '__main__':
    main()




