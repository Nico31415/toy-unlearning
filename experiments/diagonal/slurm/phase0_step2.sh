#!/bin/bash
#SBATCH --partition=icelake
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/phase0_step2_%A_%a.out
#SBATCH --error=logs/phase0_step2_%A_%a.err

# Phase 0 Step 2 (support): 2 cases × 4 alpha × 3 seeds = 24 tasks (0-23)
# Targeted smoke-run for fixed-point/legacy-stop fix validation

set -euo pipefail

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID starting on $(hostname)"
echo "[SLURM] Phase 0 Step 2 Smoke Run"

# Ensure we run from project root and logs exist
cd /home/na658/multi-task2
mkdir -p logs
mkdir -p results/diagonal/phase0/step2

# Use absolute Python from conda env
PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] Expected Python not found at $PY" >&2
  exit 1
fi

echo "[SLURM] Using Python: $PY"

# Run the phase0 smoke run script
$PY experiments/diagonal/phase0_smoke_run.py --phase step2 --task_id $SLURM_ARRAY_TASK_ID

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID finished"



