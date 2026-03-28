#!/bin/bash
#SBATCH --partition=icelake
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-1385
#SBATCH --output=logs/emp_imperfect_pt_alpha_pt_sweep_%A_%a.out
#SBATCH --error=logs/emp_imperfect_pt_alpha_pt_sweep_%A_%a.err

# Imperfect-PT empirical alpha_pt sweep
# Total tasks: 2772
#   9 configs × 2 alpha_pt × 11 alpha_ft × 14 seeds = 2772
# Each array task runs exactly one (config, alpha_pt, alpha_ft, seed) combo

set -euo pipefail

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID starting on $(hostname)"
echo "[SLURM] Imperfect-PT empirical alpha_pt sweep"

# Run from project root
cd /home/na658/multi-task2

# Create output folders
mkdir -p logs
mkdir -p results/emp_imperfect_pt_alpha_pt_sweep

# Use absolute Python from conda env
PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] Expected Python not found at $PY" >&2
  exit 1
fi

echo "[SLURM] Using Python: $PY"
echo "[SLURM] Working directory: $(pwd)"
echo "[SLURM] Output dir: results/emp_imperfect_pt_alpha_pt_sweep"

"$PY" experiments/diagonal/replica/compute_emp_imperfect_pt_alpha_pt_sweep_worker.py \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --output-dir results/emp_imperfect_pt_alpha_pt_sweep

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID finished"