#!/bin/bash
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --array=0-269
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

set -euo pipefail

echo "[SLURM] Job $SLURM_JOB_ID  task $SLURM_ARRAY_TASK_ID  host $(hostname)"

cd /home/na658/multi-task2
mkdir -p experiments/diagonal/logs

PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] Python not found at $PY" >&2
  exit 1
fi

$PY experiments/diagonal/run_unlearning_sweep.py --task_id $SLURM_ARRAY_TASK_ID

echo "[SLURM] task $SLURM_ARRAY_TASK_ID done"
