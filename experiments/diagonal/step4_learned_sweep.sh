#!/bin/bash
#SBATCH --partition=icelake
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --array=0-53
#SBATCH --output=logs/step4_learned_%A_%a.out
#SBATCH --error=logs/step4_learned_%A_%a.err

# Total jobs: 3 omega values × 6 alpha_ft values × 3 seeds = 54 jobs (0-53)
# Each job runs both PT and FT phases, so needs more time

set -euo pipefail

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID starting on $(hostname)"
echo "[SLURM] Step 4 Learned PT Sweep"

# Ensure we run from project root and logs exist
cd /home/na658/multi-task2
mkdir -p logs
mkdir -p results/diagonal/step4_learned
mkdir -p figures/validation_step4

# Use absolute Python from conda env
PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] Expected Python not found at $PY" >&2
  exit 1
fi

echo "[SLURM] Using Python: $PY"

# Run the sweep script
$PY experiments/diagonal/step4_learned_sweep.py $SLURM_ARRAY_TASK_ID

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID finished"

