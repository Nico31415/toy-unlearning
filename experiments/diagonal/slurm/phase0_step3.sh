#!/bin/bash
#SBATCH --partition=icelake
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/phase0_step3_%A_%a.out
#SBATCH --error=logs/phase0_step3_%A_%a.err

# Phase 0 Step 3 (omega): 2 omega × 2 alpha × 3 seeds = 12 tasks (0-11)
# Targeted smoke-run for fixed-point/legacy-stop fix validation

set -euo pipefail

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID starting on $(hostname)"
echo "[SLURM] Phase 0 Step 3 Smoke Run"

# Ensure we run from project root and logs exist
cd /home/na658/multi-task2
mkdir -p logs
mkdir -p results/diagonal/phase0/step3

# Use absolute Python from conda env
PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] Expected Python not found at $PY" >&2
  exit 1
fi

echo "[SLURM] Using Python: $PY"

# Run the phase0 smoke run script
$PY experiments/diagonal/phase0_smoke_run.py --phase step3 --task_id $SLURM_ARRAY_TASK_ID

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID finished"




