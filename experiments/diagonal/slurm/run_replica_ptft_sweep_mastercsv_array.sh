#!/bin/bash
#
# Replica PT+FT sweep (ptft_replica_qk.py) matching replica_ptft.ipynb parameter grids:
#   alphas: 11 values in [0.01, 0.5]
#   seeds: [0]
#   lambda_pt: [0.0, -0.00099, +0.00099]
#   c_pt: [1e-3]
#   omega: [1.0, 0.0]
#   gamma_reinit: [0.0, 1.0]
#
# Total tasks = 11 * 1 * 3 * 2 * 2 * 1 = 132 => array=0-131
#
# Each array task runs ONE (alpha, seed, lambda, omega, gamma_reinit, c_pt) combo and appends
# its one-row CSV output into a single master CSV under a file lock.
#
# Submit:
#   sbatch experiments/diagonal/slurm/run_replica_ptft_sweep_mastercsv_array.sh
#
# Optional overrides:
#   sbatch --export=RUN_NAME=mytag,PYTHON_BIN=/path/to/python experiments/diagonal/slurm/run_replica_ptft_sweep_mastercsv_array.sh
#

#SBATCH --job-name=replica_ptft
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --array=0-131
#SBATCH --output=logs/slurm/replica_ptft_%A_%a.out
#SBATCH --error=logs/slurm/replica_ptft_%A_%a.err
#
# If your cluster requires a partition, uncomment and set appropriately:
#SBATCH --partition=icelake

set -euo pipefail

echo "[SLURM] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting on $(hostname)"

cd /home/na658/multi-task2
mkdir -p logs/slurm
mkdir -p results/diagonal/replica_ptft

# Avoid oversubscription when launching many tasks in parallel.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Pick Python. Prefer $PYTHON_BIN if provided; otherwise use the repo's common conda env if present.
PY="${PYTHON_BIN:-/home/na658/.conda/envs/mtl_ft/bin/python}"
if [ ! -x "$PY" ]; then
  PY="${PYTHON_BIN:-python}"
fi
echo "[SLURM] Using Python: ${PY}"

# Grid (np.linspace(0.01, 0.5, 11))
ALPHAS=(0.01 0.059 0.108 0.157 0.206 0.255 0.304 0.353 0.402 0.451 0.5)
SEED=0

LAMBDAS=(0.0 -0.00099 0.00099)
OMEGAS=(1.0 0.0)
GAMMAS=(0.0 1.0)

# Fixed params
C_PT=0.001
RHO_PT=0.10
RHO_FT=0.10
A_PT=1.0

MC=80000
GAMMA_EXT=1e-6
TOL=1e-6
MAX_ITERS=900
DAMP=0.25

# Map array_id -> (alpha_idx, lambda_idx, omega_idx, gamma_idx) with seed fixed
tid="${SLURM_ARRAY_TASK_ID}"
alpha_idx=$(( tid % 11 ))
tid=$(( tid / 11 ))
lam_idx=$(( tid % 3 ))
tid=$(( tid / 3 ))
omega_idx=$(( tid % 2 ))
tid=$(( tid / 2 ))
gamma_idx=$(( tid % 2 ))

alpha="${ALPHAS[$alpha_idx]}"
lam="${LAMBDAS[$lam_idx]}"
omega="${OMEGAS[$omega_idx]}"
gamma_reinit="${GAMMAS[$gamma_idx]}"

RUN_NAME="${RUN_NAME:-replica_ptft_sweep}"
MASTER="results/diagonal/replica_ptft/${RUN_NAME}_master.csv"
LOCK="${MASTER}.lock"

echo "[task] alpha=${alpha} seed=${SEED} c_pt=${C_PT} lambda_pt=${lam} omega=${omega} gamma_reinit=${gamma_reinit}"
echo "[task] master=${MASTER}"

tmp_csv="$(mktemp "/tmp/replica_ptft_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.XXXXXX.csv")"

"$PY" - <<PY
import sys
from pathlib import Path
import numpy as np

# Ensure we can import the replica module when running from repo root under SLURM.
sys.path.insert(0, "/home/na658/multi-task2/experiments/diagonal/replica")

import ptft_replica_qk as rq

alpha = float("${alpha}")
seed = int("${SEED}")

df = rq.build_ptft_curves_dataframe(
    rho_pt=${RHO_PT},
    rho_ft=${RHO_FT},
    omega=${omega},
    c_pt=${C_PT},
    lambda_pt=${lam},
    gamma_reinit=${gamma_reinit},
    a_pt=${A_PT},
    alphas=np.array([alpha], dtype=float),
    mc=${MC},
    seed=[seed],
    gamma_ext=${GAMMA_EXT},
    tol=${TOL},
    max_iters=${MAX_ITERS},
    damp=${DAMP},
)

df.to_csv("${tmp_csv}", index=False)
print("wrote tmp", "${tmp_csv}", "rows", len(df))
PY

# Append under a file lock (safe for many parallel tasks).
# Note: rerunning the same sweep will append duplicates; if you want dedupe-on-key, ask and we can add it.
(
  flock -x 200
  if [ ! -f "${MASTER}" ]; then
    cp "${tmp_csv}" "${MASTER}"
  else
    # append without header
    tail -n +2 "${tmp_csv}" >> "${MASTER}"
  fi
) 200>"${LOCK}"

rm -f "${tmp_csv}"
echo "[task] appended OK"

