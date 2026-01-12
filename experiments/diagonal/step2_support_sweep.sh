#!/bin/bash
#SBATCH --partition=icelake
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --array=0-71
#SBATCH --output=logs/step2_support_%A_%a.out
#SBATCH --error=logs/step2_support_%A_%a.err

# Total jobs: 2 cases × 12 alpha values × 3 seeds = 72 jobs (0-71)

set -euo pipefail

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID starting on $(hostname)"
echo "[SLURM] Step 2 Support Sweep"

# Ensure we run from project root and logs exist
cd /home/na658/multi-task2
mkdir -p logs
mkdir -p results/diagonal/step2_support
mkdir -p figures/validation_step2

# Use absolute Python from conda env
PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] Expected Python not found at $PY" >&2
  exit 1
fi

echo "[SLURM] Using Python: $PY"

# Run the sweep script
$PY experiments/diagonal/step2_support_sweep.py $SLURM_ARRAY_TASK_ID

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID finished"

