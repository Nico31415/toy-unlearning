#!/usr/bin/env python3
"""
Run full PT+FT oracle validation grid: replica vs empirical diagonal network.

Grid:
  c_pt       ∈ {0.001, 0.5}
  lambda_pt  ∈ {0, -0.95*c_pt, +0.95*c_pt}  (avoid degenerate c_ft=0 case)
  omega      ∈ {0, 0.5, 1.0}
  rho_ft     ∈ {0.04, 0.1}
  gamma_reinit ∈ {0.0, 0.8}

Fixed:
  rho_pt = 0.10
  a_pt = 1.0
  ft_regulariser_scale = 1e-6

Total: 2 × 3 × 3 × 2 × 2 = 72 configurations
"""

import itertools
import sys
import os
from pathlib import Path

# Ensure we can import from the repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.diagonal.compare_ptft_replica_empirical import compare_ptft_replica_vs_empirical


# =============================================================================
# GRID DEFINITION
# =============================================================================

RHO_PT = 0.10
A_PT = 1.0
FT_REGULARISER_SCALE = 1e-6

C_PT_VALUES = [0.001, 0.5]
OMEGA_VALUES = [0.0, 0.5, 1.0]
RHO_FT_VALUES = [0.04, 0.1]
GAMMA_REINIT_VALUES = [0.0, 0.8]

# lambda_pt is relative to c_pt: {0, -0.95*c_pt, +0.95*c_pt}
# NOTE: Using 0.95 instead of 1.0 to avoid degenerate c_ft=0 case when gamma_reinit=0
LAMBDA_PT_MULTIPLIERS = [0.0, -0.95, +0.95]


# =============================================================================
# EXPERIMENT SETTINGS (adjust for speed vs accuracy)
# =============================================================================

INP_DIM = 500
N_TEST = 5000
NUM_SEEDS = 3
SEED_BASE = 0

ALPHA_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# Training
EPOCHS = 100_000
THRESHOLD = 1e-10
TEST_EVERY_N_EPOCHS = 500
LR = 0.5
NO_TUNING = True

# Replica
ALPHA_MIN = 0.02
ALPHA_MAX = 1.0
ALPHA_POINTS = 100
MC_SAMPLES = 30_000
REPLICA_SEED = 12345

OUTPUT_DIR_BASE = "figures/ptft_validation_grid_v2"


def _config_output_dir(cfg):
    """Generate unique output dir including omega."""
    def fmt(x):
        return f"{float(x):.6g}".replace("+", "")
    return (
        f"{OUTPUT_DIR_BASE}/"
        f"rpt={fmt(cfg['rho_pt'])}__rft={fmt(cfg['rho_ft'])}__"
        f"om={fmt(cfg['omega'])}__"
        f"cpt={fmt(cfg['c_pt'])}__lpt={fmt(cfg['lambda_pt'])}__"
        f"gam={fmt(cfg['gamma_reinit'])}"
    )


def generate_grid():
    """Generate all 72 parameter combinations."""
    configs = []
    for c_pt in C_PT_VALUES:
        for lmda_mult in LAMBDA_PT_MULTIPLIERS:
            lambda_pt = lmda_mult * c_pt
            for omega in OMEGA_VALUES:
                for rho_ft in RHO_FT_VALUES:
                    for gamma_reinit in GAMMA_REINIT_VALUES:
                        configs.append({
                            "rho_pt": RHO_PT,
                            "rho_ft": rho_ft,
                            "omega": omega,
                            "c_pt": c_pt,
                            "lambda_pt": lambda_pt,
                            "gamma_reinit": gamma_reinit,
                            "a_pt": A_PT,
                            "ft_regulariser_scale": FT_REGULARISER_SCALE,
                        })
    return configs


def check_feasibility(rho_pt, rho_ft, omega):
    """Check if (rho_pt, rho_ft, omega) is feasible."""
    # omega * rho_ft <= rho_pt
    if omega * rho_ft > rho_pt + 1e-9:
        return False
    # rho_pt + (1-omega) * rho_ft <= 1
    if rho_pt + (1.0 - omega) * rho_ft > 1.0 + 1e-9:
        return False
    return True


def run_single_config(cfg, idx, total):
    """Run a single configuration."""
    print("\n" + "=" * 80)
    print(f"CONFIG {idx + 1} / {total}")
    print("=" * 80)
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")
    print("=" * 80)

    # Feasibility check
    if not check_feasibility(cfg["rho_pt"], cfg["rho_ft"], cfg["omega"]):
        print("SKIPPED: infeasible parameter combination")
        return None

    # Each config gets its own output directory (including omega)
    output_dir = _config_output_dir(cfg)

    try:
        result = compare_ptft_replica_vs_empirical(
            rho_pt=cfg["rho_pt"],
            rho_ft=cfg["rho_ft"],
            omega=[cfg["omega"]],  # single omega per config
            c_pt=cfg["c_pt"],
            lambda_pt=cfg["lambda_pt"],
            gamma_reinit=cfg["gamma_reinit"],
            a_pt=cfg["a_pt"],
            ft_regulariser_scale=cfg["ft_regulariser_scale"],
            num_seeds=NUM_SEEDS,
            seed_base=SEED_BASE,
            replica_seed=REPLICA_SEED,
            inp_dim=INP_DIM,
            n_test=N_TEST,
            alpha_values=ALPHA_VALUES,
            lr=LR,
            epochs=EPOCHS,
            threshold=THRESHOLD,
            test_every_n_epochs=TEST_EVERY_N_EPOCHS,
            no_tuning=NO_TUNING,
            alpha_min=ALPHA_MIN,
            alpha_max=ALPHA_MAX,
            alpha_points=ALPHA_POINTS,
            mc_samples=MC_SAMPLES,
            output_dir=output_dir,
            run_empirical=True,
            run_replica=True,
            make_plot=True,
            skip_existing=True,
        )
        print(f"  ✓ Completed: {result.get('plot_paths', {}).get('png', 'N/A')}")
        return result
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run PT+FT validation grid")
    parser.add_argument("array_id", type=int, nargs="?", default=None,
                        help="SLURM array task ID (runs single config if provided)")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based, ignored if array_id given)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive, ignored if array_id given)")
    parser.add_argument("--dry_run", action="store_true", help="Just print configs, don't run")
    parser.add_argument("--list", action="store_true", help="List all configs and exit")
    args = parser.parse_args()

    configs = generate_grid()
    total = len(configs)

    # List mode: just print all configs
    if args.list:
        print(f"Total configurations: {total}")
        for i, cfg in enumerate(configs):
            feasible = check_feasibility(cfg["rho_pt"], cfg["rho_ft"], cfg["omega"])
            status = "✓" if feasible else "✗ INFEASIBLE"
            print(f"[{i:3d}] c_pt={cfg['c_pt']:.3f}, λ_pt={cfg['lambda_pt']:+.4f}, "
                  f"ω={cfg['omega']:.1f}, ρ_ft={cfg['rho_ft']:.2f}, γ={cfg['gamma_reinit']:.1f} {status}")
        return 0

    # SLURM array mode: run single config by array_id
    if args.array_id is not None:
        if args.array_id < 0 or args.array_id >= total:
            print(f"ERROR: array_id {args.array_id} out of range [0, {total - 1}]")
            return 1
        
        Path(OUTPUT_DIR_BASE).mkdir(parents=True, exist_ok=True)
        cfg = configs[args.array_id]
        res = run_single_config(cfg, args.array_id, total)
        return 0 if res is not None else 1

    # Batch mode: run range of configs
    print(f"Total configurations: {total}")

    end = args.end if args.end is not None else total
    configs_to_run = configs[args.start:end]

    print(f"Running configs {args.start} to {end - 1} ({len(configs_to_run)} total)")

    if args.dry_run:
        for i, cfg in enumerate(configs_to_run):
            idx = args.start + i
            feasible = check_feasibility(cfg["rho_pt"], cfg["rho_ft"], cfg["omega"])
            status = "✓" if feasible else "✗ INFEASIBLE"
            print(f"[{idx:3d}] c_pt={cfg['c_pt']:.3f}, λ_pt={cfg['lambda_pt']:+.4f}, "
                  f"ω={cfg['omega']:.1f}, ρ_ft={cfg['rho_ft']:.2f}, γ={cfg['gamma_reinit']:.1f} {status}")
        return 0

    Path(OUTPUT_DIR_BASE).mkdir(parents=True, exist_ok=True)

    results = []
    for i, cfg in enumerate(configs_to_run):
        idx = args.start + i
        res = run_single_config(cfg, idx, total)
        results.append({"config": cfg, "result": res})

    # Summary
    succeeded = sum(1 for r in results if r["result"] is not None)
    failed = len(results) - succeeded
    print("\n" + "=" * 80)
    print(f"SUMMARY: {succeeded} succeeded, {failed} failed/skipped out of {len(results)}")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

