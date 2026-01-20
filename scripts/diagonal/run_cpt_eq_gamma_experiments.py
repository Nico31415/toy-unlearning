#!/usr/bin/env python3
"""
Run experiments for c_pt = gamma_reinit cases.
Two setups:
1. c_pt = gamma_reinit = 1.0, λ_pt ∈ {-0.95, 0, +0.95}
2. c_pt = gamma_reinit = 0.001, λ_pt ∈ {-0.00095, 0, +0.00095}

For each config, 5 unique (rho_ft, omega) combinations.
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
    (0.02, 1.0),  # Left subplot
    (0.04, 1.0),  # Left subplot
    (0.1, 1.0),   # Shared
    (0.1, 0.5),   # Right subplot
    (0.1, 0.0),   # Right subplot
]

ALPHA_VALUES = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


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


def check_existing(cfg):
    """Check if config already has both replica and empirical data."""
    out_dir = Path(config_output_dir(cfg))
    if not out_dir.exists():
        return False, False
    
    subdirs = [d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith('ptft_oracle')]
    if not subdirs:
        return False, False
    
    subdir = subdirs[0]
    
    # Check replica
    cache_dir = subdir / "replica_cache"
    has_replica = cache_dir.exists() and len(list(cache_dir.glob("*.csv"))) > 0
    
    # Check empirical
    emp_csv = subdir / "empirical_results.csv"
    has_empirical = False
    if emp_csv.exists():
        df = pd.read_csv(emp_csv)
        has_empirical = df['param_mse'].notna().sum() > 10
    
    return has_replica, has_empirical


def run_replica(cfg):
    """Run replica computation for a config."""
    sys.path.insert(0, str(THIS_DIR.parent.parent / "ReplicaExperiments"))
    from fixed_lambda_all import Config, solve_rspmap_qk_curve_best_of_forward_backward
    from plot_replica_q_bg import PTFTOracleConfig, sample_ptft_oracle_mc
    
    out_dir = Path(config_output_dir(cfg))
    
    def fmt(x):
        return f"{float(x):.6g}".replace("+", "")
    subdir_name = (
        f"ptft_oracle__rpt={fmt(cfg['rho_pt'])}__rft={fmt(cfg['rho_ft'])}__"
        f"apt={fmt(A_PT)}__cpt={fmt(cfg['c_pt'])}__lpt={fmt(cfg['lambda_pt'])}__"
        f"gam={fmt(cfg['gamma_reinit'])}__ftreg={FT_REGULARISER_SCALE:.0e}"
    )
    subdir = out_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = subdir / "replica_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create PTFT oracle config
    ptft_cfg = PTFTOracleConfig(
        rho_pt=cfg['rho_pt'],
        rho_ft=cfg['rho_ft'],
        omega=cfg['omega'],
        a_pt=A_PT,
        c_pt=cfg['c_pt'],
        lambda_pt=cfg['lambda_pt'],
        gamma_reinit=cfg['gamma_reinit'],
    )
    
    # Sample MC
    mc_samples = 10000
    rng = np.random.default_rng(42)
    beta_ft_mc, beta_pt_mc, k_mc, g_mc = sample_ptft_oracle_mc(ptft_cfg, rng, mc_samples)
    x_mc = beta_ft_mc
    v_mc = rng.normal(size=mc_samples)
    
    # Build solver config
    alpha_min, alpha_max, alpha_points = 0.02, 1.0, 50
    beta_min = 1.0 / alpha_max
    beta_max = 1.0 / alpha_min
    var_nonzero = 1.0 / cfg['rho_ft']
    betas = np.linspace(beta_min, beta_max, alpha_points)
    
    solver_cfg = Config(
        rho=cfg['rho_ft'],
        var_nonzero=var_nonzero,
        sigma0_2=0.0,
        betas=betas,
        max_fp_iters=900,
        tol_fp=1e-10,
        damp=0.25,
    )
    
    # Solve
    gamma_ext = float(FT_REGULARISER_SCALE)
    alpha_range = np.linspace(alpha_min, alpha_max, alpha_points)
    alpha_reversed = alpha_range[::-1]
    beta_range = 1.0 / alpha_reversed
    
    k_q = 1.0  # Placeholder
    mse_beta = solve_rspmap_qk_curve_best_of_forward_backward(
        beta_range, gamma_ext, k_q, x_mc, v_mc, solver_cfg, k_mc=k_mc, g_mc=g_mc
    )
    mse_alpha = mse_beta[::-1]
    
    # Save
    df = pd.DataFrame({'alpha': alpha_range, 'mse': mse_alpha})
    cache_file = cache_dir / f"replica_rft={cfg['rho_ft']}_om={cfg['omega']}.csv"
    df.to_csv(cache_file, index=False)
    
    print(f"    Replica: MSE range [{mse_alpha.min():.2e}, {mse_alpha.max():.2e}]")
    return True


def run_empirical(cfg, num_seeds=3):
    """Run empirical experiments for a config."""
    from experiments.diagonal.diagonal_ptft_oracle import get_parser as ptft_get_parser
    from experiments.diagonal.diagonal_ptft_oracle import main as ptft_main
    
    out_dir = Path(config_output_dir(cfg))
    
    def fmt(x):
        return f"{float(x):.6g}".replace("+", "")
    subdir_name = (
        f"ptft_oracle__rpt={fmt(cfg['rho_pt'])}__rft={fmt(cfg['rho_ft'])}__"
        f"apt={fmt(A_PT)}__cpt={fmt(cfg['c_pt'])}__lpt={fmt(cfg['lambda_pt'])}__"
        f"gam={fmt(cfg['gamma_reinit'])}__ftreg={FT_REGULARISER_SCALE:.0e}"
    )
    subdir = out_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    
    emp_runs_dir = subdir / "empirical_runs"
    emp_runs_dir.mkdir(parents=True, exist_ok=True)
    
    parser = ptft_get_parser()
    results = []
    
    for seed in range(num_seeds):
        for alpha in ALPHA_VALUES:
            n_train = max(1, int(round(alpha * INP_DIM)))
            alpha_eff = n_train / float(INP_DIM)
            
            save_folder = emp_runs_dir / f"omega={cfg['omega']:.3f}__alpha={alpha_eff:.6f}__seed={seed}"
            save_folder.mkdir(parents=True, exist_ok=True)
            
            meta_path = save_folder / "results_meta.json"
            
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
                
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    results.append({
                        'seed': seed,
                        'alpha': alpha_eff,
                        'param_mse': meta.get('final_param_mse', float('nan')),
                        'test_loss': meta.get('final_test_pred_mse', float('nan')),
                        'final_train_loss': meta.get('final_train_pred_mse', float('nan')),
                    })
            except Exception as e:
                print(f"      seed={seed}, alpha={alpha:.2f}: FAILED - {e}")
                results.append({
                    'seed': seed,
                    'alpha': alpha_eff,
                    'param_mse': float('nan'),
                    'test_loss': float('nan'),
                    'final_train_loss': float('nan'),
                })
    
    df = pd.DataFrame(results)
    emp_csv = subdir / "empirical_results.csv"
    df.to_csv(emp_csv, index=False)
    
    valid = df['param_mse'].notna().sum()
    print(f"    Empirical: {valid}/{len(results)} valid")
    return valid > 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', type=int, default=None)
    parser.add_argument('--num-seeds', type=int, default=3)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--replica-only', action='store_true')
    parser.add_argument('--empirical-only', action='store_true')
    args = parser.parse_args()
    
    all_configs = generate_all_configs()
    
    # Filter to configs that need work
    configs_to_run = []
    for cfg in all_configs:
        has_replica, has_empirical = check_existing(cfg)
        needs_replica = not has_replica and not args.empirical_only
        needs_empirical = not has_empirical and not args.replica_only
        if needs_replica or needs_empirical:
            configs_to_run.append((cfg, needs_replica, needs_empirical))
    
    print(f"Total configs: {len(all_configs)}, Need work: {len(configs_to_run)}")
    
    if args.list:
        for i, (cfg, nr, ne) in enumerate(configs_to_run):
            status = []
            if nr: status.append("replica")
            if ne: status.append("empirical")
            print(f"{i}: c={cfg['c_pt']}, g={cfg['gamma_reinit']}, lpt={cfg['lambda_pt']:.6f}, om={cfg['omega']}, rft={cfg['rho_ft']} [{', '.join(status)}]")
        return
    
    if args.task_id is not None:
        if args.task_id >= len(configs_to_run):
            print(f"Task ID {args.task_id} >= {len(configs_to_run)}, skipping")
            return
        
        cfg, needs_replica, needs_empirical = configs_to_run[args.task_id]
        print(f"Task {args.task_id}: c={cfg['c_pt']}, g={cfg['gamma_reinit']}, lpt={cfg['lambda_pt']:.6f}, om={cfg['omega']}, rft={cfg['rho_ft']}")
        
        if needs_replica:
            print("  Running replica...")
            run_replica(cfg)
        
        if needs_empirical:
            print("  Running empirical...")
            run_empirical(cfg, num_seeds=args.num_seeds)
        
        print(f"✓ Completed task {args.task_id}")
    else:
        for i, (cfg, needs_replica, needs_empirical) in enumerate(configs_to_run):
            print(f"\n[{i+1}/{len(configs_to_run)}] c={cfg['c_pt']}, g={cfg['gamma_reinit']}, lpt={cfg['lambda_pt']:.6f}, om={cfg['omega']}, rft={cfg['rho_ft']}")
            
            if needs_replica:
                print("  Running replica...")
                run_replica(cfg)
            
            if needs_empirical:
                print("  Running empirical...")
                run_empirical(cfg, num_seeds=args.num_seeds)
        
        print(f"\n✓ All done")


if __name__ == '__main__':
    main()

