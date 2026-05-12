#!/bin/bash
#SBATCH --job-name=emp_corr_scratch
#SBATCH --partition=icelake
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-164
#SBATCH --chdir=/home/na658/multi-task2
#SBATCH --output=logs/emp_corr_scratch_%A_%a.out
#SBATCH --error=logs/emp_corr_scratch_%A_%a.err

set -euo pipefail

REPO_ROOT="/home/na658/multi-task2"
OUT_DIR="$REPO_ROOT/results/sanity_check_correlated_overlap_scratch"
LOG_DIR="$REPO_ROOT/logs"
PY="/home/na658/.conda/envs/mtl_ft/bin/python"
LOCK_DIR="$OUT_DIR/task_locks"
DONE_DIR="$OUT_DIR/task_done"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$LOCK_DIR" "$DONE_DIR"
cd "$REPO_ROOT"

echo "============================================"
echo "[SLURM] Job: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "[SLURM] Correlated overlap scratch-FT baseline"
echo "[SLURM] Host: $(hostname)"
echo "[SLURM] Start: $(date)"
echo "============================================"

if [ ! -x "$PY" ]; then
  echo "[SLURM] Expected Python not found at $PY" >&2
  exit 1
fi

task_id="$SLURM_ARRAY_TASK_ID"
done_file="$DONE_DIR/task_${task_id}.done"
lock_path="$LOCK_DIR/task_${task_id}.lock"

if [ -f "$done_file" ]; then
  echo "[SLURM] Task $task_id already marked done; exiting."
  exit 0
fi

if ! mkdir "$lock_path" 2>/dev/null; then
  echo "[SLURM] Task $task_id is already locked by another job; exiting."
  exit 0
fi
trap 'rmdir "$lock_path" 2>/dev/null || true' EXIT

echo "[SLURM] Using Python: $PY"

"$PY" experiments/diagonal/replica/compute_emp_correlated_overlap_scratch_worker.py \
  --task-id "$task_id" \
  --omega 0.5 \
  --output-dir "$OUT_DIR"

touch "$done_file"

echo "============================================"
echo "[SLURM] Done: $(date)"
echo "============================================"
