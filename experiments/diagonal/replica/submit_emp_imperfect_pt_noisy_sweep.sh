#!/bin/bash
#SBATCH --partition=icelake
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-4157
#SBATCH --output=logs/emp_imperfect_pt_noisy_sweep_%A_%a.out
#SBATCH --error=logs/emp_imperfect_pt_noisy_sweep_%A_%a.err

# Imperfect-PT empirical noisy underdetermined sweep
# Fixed: pt_mode=noisy, alpha_pt=0.95
# Total tasks: 4158
#   9 configs × 3 sigma0_pt × 11 alpha_ft × 14 seeds = 4158

set -euo pipefail

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID starting on $(hostname)"
echo "[SLURM] Imperfect-PT empirical noisy underdetermined sweep"

cd /home/na658/multi-task2

mkdir -p logs
mkdir -p results/emp_imperfect_pt_noisy_sweep

PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] Expected Python not found at $PY" >&2
  exit 1
fi

echo "[SLURM] Using Python: $PY"
echo "[SLURM] Working directory: $(pwd)"
echo "[SLURM] Output dir: results/emp_imperfect_pt_noisy_sweep"

"$PY" experiments/diagonal/replica/compute_emp_imperfect_pt_noisy_sweep_worker.py \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --output-dir results/emp_imperfect_pt_noisy_sweep

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID finished"
