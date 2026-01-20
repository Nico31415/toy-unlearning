#!/usr/bin/env python3
"""
Run ONLY empirical experiments for c_pt = gamma_reinit cases.
Skips already-completed runs and aggregates results at the end.

Two setups:
1. c_pt = gamma_reinit = 1.0, λ_pt ∈ {-0.95, 0, +0.95}
2. c_pt = gamma_reinit = 0.001, λ_pt ∈ {-0.00095, 0, +0.00095}
"""

import sys
from pathlib import Path
import os
import json

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent.parent))

import numpy as np
import pandas as pd

EXP_BASE = Path("figures/panel_experiments")

# Fixed params
RHO_PT = 0.1
A_PT = 1.0
FT_REGULARISER_SCALE = 1e-6
INP_DIM = 500
N_TEST = 2000

# Two setups: c_pt = gamma_reinit
SETUPS = [
    {'c_pt': 1.0, 'gamma_reinit': 1.0, 'lambda_pts': [-0.95, 0.0, 0.95]},
    {'c_pt': 0.001, 'gamma_reinit': 0.001, 'lambda_pts': [-0.00095, 0.0, 0.00095]},
]

# Subplot params - 5 unique (rho_ft, omega) combinations
SUBPLOT_CONFIGS = [
    (0.02, 1.0),
    (0.04, 1.0),
    (0.1, 1.0),
    (0.1, 0.5),
    (0.1, 0.0),
]

ALPHA_VALUES = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_SEEDS = 3


def generate_all_configs():
    """Generate all experiment configurations."""
    configs = []
    for setup in SETUPS:
        c_pt = setup['c_pt']
        gamma_reinit = setup['gamma_reinit']
        for lpt in setup['lambda_pts']:
            for rho_ft, omega in SUBPLOT_CONFIGS:
                configs.append({
                    'rho_pt': RHO_PT,
                    'rho_ft': rho_ft,
                    'omega': omega,
                    'c_pt': c_pt,
                    'lambda_pt': lpt,
                    'gamma_reinit': gamma_reinit,
                })
    return configs


def config_output_dir(cfg):
    """Generate output directory path for a config."""
    def fmt(x):
        return f"{float(x):.6g}".replace("+", "")
    return (
        f"{EXP_BASE}/"
        f"rpt={fmt(cfg['rho_pt'])}__rft={fmt(cfg['rho_ft'])}__"
        f"om={fmt(cfg['omega'])}__cpt={fmt(cfg['c_pt'])}__"
        f"lpt={fmt(cfg['lambda_pt'])}__gam={fmt(cfg['gamma_reinit'])}"
    )


def get_subdir(cfg):
    """Get the ptft_oracle subdirectory for a config."""
    out_dir = Path(config_output_dir(cfg))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    def fmt(x):
        return f"{float(x):.6g}".replace("+", "")
    subdir_name = (
        f"ptft_oracle__rpt={fmt(cfg['rho_pt'])}__rft={fmt(cfg['rho_ft'])}__"
        f"apt={fmt(A_PT)}__cpt={fmt(cfg['c_pt'])}__lpt={fmt(cfg['lambda_pt'])}__"
        f"gam={fmt(cfg['gamma_reinit'])}__ftreg={FT_REGULARISER_SCALE:.0e}"
    )
    subdir = out_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def run_single_empirical(cfg, seed, alpha):
    """Run a single empirical experiment if not already complete."""
    from experiments.diagonal.diagonal_ptft_oracle import get_parser as ptft_get_parser
    from experiments.diagonal.diagonal_ptft_oracle import main as ptft_main
    
    subdir = get_subdir(cfg)
    emp_runs_dir = subdir / "empirical_runs"
    emp_runs_dir.mkdir(parents=True, exist_ok=True)
    
    n_train = max(1, int(round(alpha * INP_DIM)))
    alpha_eff = n_train / float(INP_DIM)
    
    save_folder = emp_runs_dir / f"omega={cfg['omega']:.3f}__alpha={alpha_eff:.6f}__seed={seed}"
    save_folder.mkdir(parents=True, exist_ok=True)
    
    meta_path = save_folder / "results_meta.json"
    
    # Skip if already complete
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if 'final_param_mse' in meta and meta['final_param_mse'] is not None:
                return {'status': 'skipped', 'alpha': alpha_eff, 'seed': seed}
        except:
            pass
    
    parser = ptft_get_parser()
    
    try:
        argv = [
            "--seed", str(seed),
            "--save_folder", str(save_folder),
            "--inp_dim", str(INP_DIM),
            "--n_train", str(n_train),
            "--n_test", str(N_TEST),
            "--rho_pt", str(cfg['rho_pt']),
            "--rho_ft", str(cfg['rho_ft']),
            "--omega", str(cfg['omega']),
            "--a_pt", str(A_PT),
            "--c_pt", str(cfg['c_pt']),
            "--lambda_pt", str(cfg['lambda_pt']),
            "--gamma_reinit", str(cfg['gamma_reinit']),
            "--lr", "0.01",
            "--epochs", "50000",
            "--threshold", "1e-9",
            "--test_every_n_epochs", "500",
            "--no_tuning",
        ]
        
        args = parser.parse_args(argv)
        ptft_main(args)
        
        return {'status': 'completed', 'alpha': alpha_eff, 'seed': seed}
    except Exception as e:
        print(f"      seed={seed}, alpha={alpha:.2f}: FAILED - {e}")
        return {'status': 'failed', 'alpha': alpha_eff, 'seed': seed, 'error': str(e)}


def aggregate_results(cfg):
    """Aggregate all empirical results for a config into a CSV."""
    subdir = get_subdir(cfg)
    emp_runs_dir = subdir / "empirical_runs"
    
    results = []
    for run_dir in emp_runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "results_meta.json"
        if not meta_path.exists():
            continue
        
        try:
            meta = json.loads(meta_path.read_text())
            # Parse seed and alpha from directory name
            name = run_dir.name
            parts = name.split('__')
            omega_str = parts[0].replace('omega=', '')
            alpha_str = parts[1].replace('alpha=', '')
            seed_str = parts[2].replace('seed=', '')
            
            results.append({
                'seed': int(seed_str),
                'alpha': float(alpha_str),
                'param_mse': meta.get('final_param_mse', float('nan')),
                'test_loss': meta.get('final_test_pred_mse', float('nan')),
                'final_train_loss': meta.get('final_train_pred_mse', float('nan')),
            })
        except Exception as e:
            print(f"    Warning: failed to parse {run_dir.name}: {e}")
    
    if results:
        df = pd.DataFrame(results)
        emp_csv = subdir / "empirical_results.csv"
        df.to_csv(emp_csv, index=False)
        print(f"    Aggregated {len(results)} results to {emp_csv}")
        return len(results)
    return 0


def run_config_empirical(cfg):
    """Run all empirical experiments for a single config."""
    print(f"  Config: c={cfg['c_pt']}, g={cfg['gamma_reinit']}, lpt={cfg['lambda_pt']:.6f}, om={cfg['omega']}, rft={cfg['rho_ft']}")
    
    completed = 0
    skipped = 0
    failed = 0
    
    for seed in range(NUM_SEEDS):
        for alpha in ALPHA_VALUES:
            result = run_single_empirical(cfg, seed, alpha)
            if result['status'] == 'completed':
                completed += 1
            elif result['status'] == 'skipped':
                skipped += 1
            else:
                failed += 1
    
    print(f"    Completed: {completed}, Skipped: {skipped}, Failed: {failed}")
    
    # Aggregate results
    total = aggregate_results(cfg)
    return total


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', type=int, default=None, help='SLURM array task ID')
    parser.add_argument('--list', action='store_true', help='List all configs')
    args = parser.parse_args()
    
    all_configs = generate_all_configs()
    
    print(f"Total configs: {len(all_configs)}")
    
    if args.list:
        for i, cfg in enumerate(all_configs):
            print(f"{i}: c={cfg['c_pt']}, g={cfg['gamma_reinit']}, lpt={cfg['lambda_pt']:.6f}, om={cfg['omega']}, rft={cfg['rho_ft']}")
        return
    
    if args.task_id is not None:
        if args.task_id >= len(all_configs):
            print(f"Task ID {args.task_id} >= {len(all_configs)}, skipping")
            return
        
        cfg = all_configs[args.task_id]
        print(f"Task {args.task_id}:")
        run_config_empirical(cfg)
        print(f"✓ Completed task {args.task_id}")
    else:
        # Run all
        for i, cfg in enumerate(all_configs):
            print(f"\n[{i+1}/{len(all_configs)}]")
            run_config_empirical(cfg)
        print(f"\n✓ All done")


if __name__ == '__main__':
    main()
