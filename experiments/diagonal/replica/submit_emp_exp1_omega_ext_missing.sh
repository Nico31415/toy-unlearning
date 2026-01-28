#!/bin/bash
set -euo pipefail

# Driver script (NOT a sbatch script):
# - Computes which omega-extension Exp1 empirical tasks are missing
# - Submits only those task IDs using the sbatch script submit_emp_exp1_omega_ext.sh

REPO_ROOT="/home/na658/multi-task2"
OUT_DIR="$REPO_ROOT/results/emp_ptft_parallel"
SBATCH_SCRIPT="$REPO_ROOT/experiments/diagonal/replica/submit_emp_exp1_omega_ext.sh"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "ERROR: sbatch script not found: $SBATCH_SCRIPT"
  exit 1
fi

ARRAY_SPEC="$(
python - <<'PY'
from __future__ import annotations
from pathlib import Path
import itertools
import numpy as np

repo_root = Path("/home/na658/multi-task2")
out_dir = repo_root / "results/emp_ptft_parallel"

alphas = np.linspace(0.01, 0.5, 11)
seeds = list(range(6, 20))
omegas = [0.0, 1.0]

C_BASE = 1e-3
LAMBDA_BASE = 0.0
GAMMA_BASE = 0.0

cs = [1e-6, 1.0]
lambda_pts = [-1.0e-3, -0.99e-3, 0.99e-3]
gamma_reinits = [1.0, 10.0]

# IMPORTANT: must match compute_emp_curves_worker_exp1_omega_ext.py ordering exactly
param_combinations = []
param_combinations += [
    ("sweep_c", omega, c_pt, LAMBDA_BASE, GAMMA_BASE, seed, float(alpha))
    for omega, c_pt, seed, alpha in itertools.product(omegas, cs, seeds, alphas)
]
param_combinations += [
    ("sweep_lambda", omega, C_BASE, lambda_pt, GAMMA_BASE, seed, float(alpha))
    for omega, lambda_pt, seed, alpha in itertools.product(omegas, lambda_pts, seeds, alphas)
]
param_combinations += [
    ("sweep_gamma", omega, C_BASE, LAMBDA_BASE, gamma_reinit, seed, float(alpha))
    for omega, gamma_reinit, seed, alpha in itertools.product(omegas, gamma_reinits, seeds, alphas)
]

def fname(sweep: str, omega: float, c_pt: float, lambda_pt: float, gamma_reinit: float, seed: int, alpha: float) -> str:
    return (
        f"EXPERIMENT1_{sweep}"
        f"_omega{omega}"
        f"_c{c_pt}"
        f"_lambda{lambda_pt}"
        f"_reinit{gamma_reinit}"
        f"_alpha{alpha:.4f}"
        f"_seed{seed}.csv"
    )

missing_ids = []
for task_id, (sweep, omega, c_pt, lambda_pt, gamma_reinit, seed, alpha) in enumerate(param_combinations):
    f = out_dir / fname(sweep, omega, c_pt, lambda_pt, gamma_reinit, seed, alpha)
    if not f.exists():
        missing_ids.append(task_id)

if not missing_ids:
    print("")  # no array spec
    raise SystemExit(0)

# compress to SLURM --array spec (e.g. 0-10,12,15-20)
missing_ids.sort()
ranges = []
start = prev = missing_ids[0]
for x in missing_ids[1:]:
    if x == prev + 1:
        prev = x
        continue
    ranges.append((start, prev))
    start = prev = x
ranges.append((start, prev))

parts = []
for a, b in ranges:
    parts.append(str(a) if a == b else f"{a}-{b}")
print(",".join(parts))
PY
)"

if [[ -z "${ARRAY_SPEC}" ]]; then
  echo "No missing omega-extension Exp1 empirical tasks found. Nothing to submit."
  exit 0
fi

echo "Submitting missing Exp1 omega-extension empirical tasks:"
echo "  --array=${ARRAY_SPEC}"

sbatch --array="${ARRAY_SPEC}" "$SBATCH_SCRIPT"

