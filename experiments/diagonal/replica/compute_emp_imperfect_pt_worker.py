"""
SLURM-array worker for imperfect-PT empirical experiments.

Each task handles one (pt_mode, alpha_pt_or_sigma0_pt, alpha_ft, seed) combination.

Grid (200 tasks total):
  cases:    ("underdetermined", alpha_pt=0.2) + ("noisy", sigma0_pt=0.01)
  alpha_ft: 20 points in [0.01, 0.5]
  seeds:    0..4

Example launch:
  sbatch --array=0-199 run_emp_imperfect_pt.sh

Or locally in parallel (e.g. with GNU parallel):
  seq 0 199 | parallel -j 8 python compute_emp_imperfect_pt_worker.py --task-id {} --output-dir results/emp_imperfect_pt
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

# --- sys.path setup (same as compute_emp_imperfect_pt.py) ---
_HERE      = Path(__file__).resolve().parent
_DIAG_DIR  = _HERE.parent
_REPO_ROOT = _HERE.parents[2]
for _p in (_REPO_ROOT, _DIAG_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import compute_emp_imperfect_pt as emp


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

# Each case: (pt_mode, alpha_pt, sigma0_pt)
CASES = [
    ("underdetermined", 0.2,  0.0),
    ("noisy",           1.0,  0.01),
]

ALPHA_FT_LIST = np.linspace(0.01, 0.5, 20).tolist()
SEEDS         = list(range(5))

# Full cartesian product: case × alpha_ft × seed
PARAM_GRID = list(itertools.product(CASES, ALPHA_FT_LIST, SEEDS))

# Print grid size if run with --info
def _print_info():
    print(f"Total tasks: {len(PARAM_GRID)}")
    print(f"  cases:     {CASES}")
    print(f"  alpha_ft:  {len(ALPHA_FT_LIST)} pts in [{ALPHA_FT_LIST[0]:.3f}, {ALPHA_FT_LIST[-1]:.3f}]")
    print(f"  seeds:     {SEEDS}")
    for i, ((mode, apt, s0), aft, seed) in enumerate(PARAM_GRID[:5]):
        print(f"  task {i:3d}: mode={mode}, alpha_pt={apt}, sigma0_pt={s0}, alpha_ft={aft:.4f}, seed={seed}")
    print("  ...")


def main():
    parser = argparse.ArgumentParser(description="Imperfect-PT empirical worker")
    parser.add_argument("--task-id",    type=int,  default=None, help="SLURM array task ID")
    parser.add_argument("--output-dir", type=str,  default="results/emp_imperfect_pt",
                        help="Directory for per-task CSVs")
    parser.add_argument("--info",       action="store_true", help="Print grid info and exit")
    # Override default training knobs if needed
    parser.add_argument("--inp-dim",    type=int,   default=1000)
    parser.add_argument("--n-test",     type=int,   default=10_000)
    parser.add_argument("--lr",         type=float, default=0.5)
    parser.add_argument("--epochs",     type=int,   default=5_000_000)
    parser.add_argument("--threshold",  type=float, default=1e-12)
    args = parser.parse_args()

    if args.info:
        _print_info()
        return

    if args.task_id is None:
        parser.error("--task-id is required (use --info to see grid size)")

    if args.task_id >= len(PARAM_GRID):
        print(f"Error: task-id {args.task_id} out of range (max {len(PARAM_GRID)-1})")
        return

    (pt_mode, alpha_pt, sigma0_pt), alpha_ft, seed = PARAM_GRID[args.task_id]

    print(f"Task {args.task_id}/{len(PARAM_GRID)-1}: "
          f"pt_mode={pt_mode}, alpha_pt={alpha_pt}, sigma0_pt={sigma0_pt}, "
          f"alpha_ft={alpha_ft:.4f}, seed={seed}", flush=True)

    row = emp.run_one(
        pt_mode=pt_mode,
        alpha_ft=alpha_ft,
        alpha_pt=alpha_pt,
        sigma0_pt=sigma0_pt,
        seed=seed,
        inp_dim=args.inp_dim,
        n_test=args.n_test,
        lr=args.lr,
        epochs=args.epochs,
        threshold=args.threshold,
    )

    print(f"  ft_param_mse={row['ft_param_mse']:.4e}  "
          f"pt_param_mse={row['pt_param_mse']:.4e}  "
          f"wall={row['wall_s']:.1f}s", flush=True)

    # Save per-task CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (f"emp_ipt_{pt_mode}_apt{alpha_pt}_s0{sigma0_pt}"
             f"_aft{alpha_ft:.4f}_seed{seed}.csv")
    import pandas as pd
    pd.DataFrame([row]).to_csv(out_dir / fname, index=False)
    print(f"  Saved → {out_dir / fname}", flush=True)


if __name__ == "__main__":
    main()
