#!/bin/bash
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --array=0-21
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

set -euo pipefail

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID starting on $(hostname)"

# Ensure we run from project root and logs exist
cd /home/na658/multi-task2
mkdir -p logs

# Use absolute Python from conda env to avoid activation issues on compute nodes
PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] Expected Python not found at $PY" >&2
  which python || true
  python --version || true
  exit 1
fi

echo "[SLURM] Using Python: $PY"

# Run the Python script directly (CSV locking is handled in Python)
$PY /home/na658/multi-task2/experiments/diagonal/diagonal_bg_alpha_sweep_1.py $SLURM_ARRAY_TASK_ID

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID finished"

