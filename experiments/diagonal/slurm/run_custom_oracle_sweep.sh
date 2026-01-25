#!/bin/bash
#SBATCH --partition=icelake
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --array=0-35
#SBATCH --output=logs/custom_oracle_sweep_%A_%a.out
#SBATCH --error=logs/custom_oracle_sweep_%A_%a.err

# Custom diagonal sweeps (oracle first).
# 1 SLURM task = 1 (parameter config × all alphas × all seeds)
#
# IMPORTANT:
# - This script does NOT submit itself; you will run: sbatch <this_file>
# - Array range assumes 36 configs (0-35). Use:
#     python experiments/diagonal/custom_oracle_sweep_driver.py --list
#   to confirm config indices before submitting.

set -euo pipefail

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID starting on $(hostname)"

# Run from repo root
cd /home/na658/multi-task2
mkdir -p logs
mkdir -p results/diagonal/custom_oracle_sweeps

# Use absolute Python from conda env (match repo conventions)
PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] Expected Python not found at $PY" >&2
  exit 1
fi
echo "[SLURM] Using Python: $PY"

# Choose a run name to avoid overwriting/ambiguity between batches.
# You can override at submission time like:
#   sbatch --export=RUN_NAME=jan22_oracle_sweep_v1 <script>
RUN_NAME="${RUN_NAME:-jan22_oracle_sweep_v1}"
echo "[SLURM] RUN_NAME=$RUN_NAME"

$PY experiments/diagonal/custom_oracle_sweep_driver.py "$SLURM_ARRAY_TASK_ID" --run_name "$RUN_NAME"

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID finished"

