#!/usr/bin/env python3
"""
Run experiments for the 3-panel figure:
- Panel 1: Varying lambda_pt (c_pt=0.001 fixed, gamma_reinit=0)
- Panel 2: Varying c_pt (lambda_pt=0 fixed, gamma_reinit=0)
- Panel 3: Varying gamma_reinit (c_pt=0.001, lambda_pt=0 fixed)

Each panel has:
- Left subplot: omega=1 fixed, rho_ft ∈ {0.02, 0.04, 0.1}
- Right subplot: rho_ft=0.1 fixed, omega ∈ {0.0, 0.5, 1.0}
"""

import sys
from pathlib import Path
import os
import itertools

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent.parent))

from compare_ptft_replica_empirical import compare_ptft_replica_vs_empirical

# Fixed params
RHO_PT = 0.1
FT_REGULARISER_SCALE = 1e-6
A_PT = 1.0

# Subplot params
RHO_FT_LEFT = [0.02, 0.04, 0.1]  # Left subplot: varying rho_ft, omega=1
OMEGA_RIGHT = [0.0, 0.5, 1.0]    # Right subplot: varying omega, rho_ft=0.1

# Panel 1: Varying lambda_pt
PANEL1_C_PT = 0.001
PANEL1_GAMMA = 0.0
PANEL1_LAMBDA_PT = [-0.95 * PANEL1_C_PT, 0.0, 0.95 * PANEL1_C_PT]  # [-0.00095, 0, 0.00095]

# Panel 2: Varying c_pt
PANEL2_LAMBDA_PT = 0.0
PANEL2_GAMMA = 0.0
PANEL2_C_PT = [0.001, 0.01, 0.1, 0.5, 1.0]

# Panel 3: Varying gamma_reinit
PANEL3_C_PT = 0.001
PANEL3_LAMBDA_PT = 0.0
PANEL3_GAMMA = [0.001, 0.01, 0.1, 1.0]

OUTPUT_DIR_BASE = "figures/panel_experiments"


def generate_all_configs():
    """Generate all unique experiment configurations."""
    configs = []
    
    def add_config(c_pt, lambda_pt, gamma_reinit, omega, rho_ft, panel_id):
        cfg = {
            'rho_pt': RHO_PT,
            'rho_ft': rho_ft,
            'omega': omega,
            'c_pt': c_pt,
            'lambda_pt': lambda_pt,
            'gamma_reinit': gamma_reinit,
            'a_pt': A_PT,
            'ft_regulariser_scale': FT_REGULARISER_SCALE,
            'panel': panel_id,
        }
        configs.append(cfg)
    
    # Panel 1: Varying lambda_pt
    for lpt in PANEL1_LAMBDA_PT:
        # Left subplot: omega=1, varying rho_ft
        for rft in RHO_FT_LEFT:
            add_config(PANEL1_C_PT, lpt, PANEL1_GAMMA, 1.0, rft, 'panel1')
        # Right subplot: rho_ft=0.1, varying omega
        for om in OMEGA_RIGHT:
            if om != 1.0:  # omega=1, rho_ft=0.1 already added above
                add_config(PANEL1_C_PT, lpt, PANEL1_GAMMA, om, 0.1, 'panel1')
    
    # Panel 2: Varying c_pt
    for cpt in PANEL2_C_PT:
        # Left subplot: omega=1, varying rho_ft
        for rft in RHO_FT_LEFT:
            add_config(cpt, PANEL2_LAMBDA_PT, PANEL2_GAMMA, 1.0, rft, 'panel2')
        # Right subplot: rho_ft=0.1, varying omega
        for om in OMEGA_RIGHT:
            if om != 1.0:  # omega=1, rho_ft=0.1 already added above
                add_config(cpt, PANEL2_LAMBDA_PT, PANEL2_GAMMA, om, 0.1, 'panel2')
    
    # Panel 3: Varying gamma_reinit
    for gam in PANEL3_GAMMA:
        # Left subplot: omega=1, varying rho_ft
        for rft in RHO_FT_LEFT:
            add_config(PANEL3_C_PT, PANEL3_LAMBDA_PT, gam, 1.0, rft, 'panel3')
        # Right subplot: rho_ft=0.1, varying omega
        for om in OMEGA_RIGHT:
            if om != 1.0:  # omega=1, rho_ft=0.1 already added above
                add_config(PANEL3_C_PT, PANEL3_LAMBDA_PT, gam, om, 0.1, 'panel3')
    
    return configs


def deduplicate_configs(configs):
    """Remove duplicate configs based on experiment parameters (not panel ID)."""
    seen = set()
    unique = []
    for cfg in configs:
        key = (cfg['rho_pt'], cfg['rho_ft'], cfg['omega'], 
               cfg['c_pt'], cfg['lambda_pt'], cfg['gamma_reinit'])
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


def config_output_dir(cfg):
    """Generate output directory path for a config."""
    def fmt(x):
        return f"{float(x):.6g}".replace("+", "")
    return (
        f"{OUTPUT_DIR_BASE}/"
        f"rpt={fmt(cfg['rho_pt'])}__rft={fmt(cfg['rho_ft'])}__"
        f"om={fmt(cfg['omega'])}__cpt={fmt(cfg['c_pt'])}__"
        f"lpt={fmt(cfg['lambda_pt'])}__gam={fmt(cfg['gamma_reinit'])}"
    )


def run_single_config(cfg, num_seeds=3, skip_existing=True):
    """Run a single experiment configuration."""
    output_dir = config_output_dir(cfg)
    
    result = compare_ptft_replica_vs_empirical(
        rho_pt=cfg['rho_pt'],
        rho_ft=cfg['rho_ft'],
        omega=cfg['omega'],
        c_pt=cfg['c_pt'],
        lambda_pt=cfg['lambda_pt'],
        gamma_reinit=cfg['gamma_reinit'],
        a_pt=cfg['a_pt'],
        ft_regulariser_scale=cfg['ft_regulariser_scale'],
        num_seeds=num_seeds,
        output_dir=output_dir,
        skip_existing=skip_existing,
        run_empirical=True,
        run_replica=True,
        make_plot=True,
    )
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', type=int, default=None,
                        help='SLURM array task ID (0-indexed)')
    parser.add_argument('--num-seeds', type=int, default=3)
    parser.add_argument('--list-configs', action='store_true',
                        help='List all configs and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be run without running')
    args = parser.parse_args()
    
    # Generate and deduplicate configs
    all_configs = generate_all_configs()
    unique_configs = deduplicate_configs(all_configs)
    
    print(f"Total configs: {len(all_configs)}, Unique: {len(unique_configs)}")
    
    if args.list_configs:
        for i, cfg in enumerate(unique_configs):
            print(f"{i}: c_pt={cfg['c_pt']}, lpt={cfg['lambda_pt']:.6f}, "
                  f"gam={cfg['gamma_reinit']}, om={cfg['omega']}, rft={cfg['rho_ft']}")
        return
    
    if args.task_id is not None:
        # SLURM array mode: run single config
        if args.task_id >= len(unique_configs):
            print(f"Task ID {args.task_id} >= {len(unique_configs)} configs, skipping")
            return
        
        cfg = unique_configs[args.task_id]
        print(f"Running task {args.task_id}: {cfg}")
        
        if args.dry_run:
            print(f"  Would output to: {config_output_dir(cfg)}")
            return
        
        result = run_single_config(cfg, num_seeds=args.num_seeds)
        print(f"✓ Completed task {args.task_id}")
    else:
        # Local mode: run all configs sequentially
        for i, cfg in enumerate(unique_configs):
            print(f"\n[{i+1}/{len(unique_configs)}] Running: {cfg}")
            if args.dry_run:
                print(f"  Would output to: {config_output_dir(cfg)}")
                continue
            run_single_config(cfg, num_seeds=args.num_seeds)
        print(f"\n✓ All {len(unique_configs)} configs completed")


if __name__ == '__main__':
    main()



