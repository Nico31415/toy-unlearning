#!/usr/bin/env python3
"""
Generate replica theory curves for multiple rho values using SLURM array jobs.

Each array_id corresponds to a (c, rho) combination.
This script calls plot_replica_q_bg.py with the appropriate parameters.
"""

import argparse
import sys
import os

sys.path.append('')

from functions.array_training import ArgparseArray

# Parameter combinations
c_values = [0.001, 0.5]
rho_values = [0.01, 0.05, 0.1, 0.5, 0.8]

argparse_array = ArgparseArray(
    c=c_values,
    rho=rho_values,
    # Fixed parameters for replica curve generation
    lambda_small=[1e-6],
    alpha_min=[0.008],
    alpha_max=[1.0],
    alpha_points=[100],
    mc_samples=[50000],
    max_fp_iters=[900],
    tol_fp=[1e-10],
    damp=[0.25],
    seed=[12345],
    output_dir=["figures/diagonal/bg_generalization"],
)


def main(args):
    import sys as _sys
    
    # Get resolved arguments for this array_id
    resolved_args = argparse_array.get_args(args.array_id)
    
    print('='*80)
    print('Replica Curve Generation - Parameters:')
    print('='*80)
    for key in sorted(resolved_args.keys()):
        if not key.startswith('aux_'):
            print(f"  {key}: {resolved_args[key]}")
    print('='*80)
    
    # Extract key parameters
    c = resolved_args['c']
    rho = resolved_args['rho']
    lambda_small = resolved_args['lambda_small']
    alpha_min = resolved_args['alpha_min']
    alpha_max = resolved_args['alpha_max']
    alpha_points = resolved_args['alpha_points']
    mc_samples = resolved_args['mc_samples']
    max_fp_iters = resolved_args['max_fp_iters']
    tol_fp = resolved_args['tol_fp']
    damp = resolved_args['damp']
    seed = resolved_args['seed']
    output_dir = resolved_args['output_dir']
    
    # Build command to call plot_replica_q_bg.py
    # Note: plot_replica_q_bg.py accepts --c_values as a list, but we're calling it
    # with a single c value per array job for better parallelization
    script_path = 'scripts/diagonal/plot_replica_q_bg.py'
    
    cmd_args = [
        _sys.executable,
        script_path,
        '--rho', str(rho),
        '--lambda_small', str(lambda_small),
        '--c_values', str(c),  # Single c value per job
        '--alpha_min', str(alpha_min),
        '--alpha_max', str(alpha_max),
        '--alpha_points', str(alpha_points),
        '--mc_samples', str(mc_samples),
        '--max_fp_iters', str(max_fp_iters),
        '--tol_fp', str(tol_fp),
        '--damp', str(damp),
        '--seed', str(seed),
        '--output_dir', output_dir,
    ]
    
    print(f"\nRunning: {' '.join(cmd_args)}")
    print()
    
    # Run the script
    import subprocess
    result = subprocess.run(cmd_args, check=False)
    
    if result.returncode == 0:
        print(f"\n✓ Array ID {args.array_id} (c={c:.6f}, ρ={rho:.2f}) completed successfully")
    else:
        print(f"\n✗ Array ID {args.array_id} (c={c:.6f}, ρ={rho:.2f}) failed with return code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int, help='SLURM array task ID')
    main(parser.parse_args())

