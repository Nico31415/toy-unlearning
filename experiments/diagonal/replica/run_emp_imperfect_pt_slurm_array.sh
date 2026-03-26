#!/usr/bin/env bash
set -euo pipefail

# SLURM array worker launcher for imperfect-PT empirical experiments.
#
# Typical usage (recommended: via submit helper):
#   experiments/diagonal/replica/submit_emp_imperfect_pt_slurm.sh -j 16
#
# Or directly (if you know the array size already):
#   sbatch --array=0-199 experiments/diagonal/replica/run_emp_imperfect_pt_slurm_array.sh

# ---------------- SLURM directives (edit to match your lab) ----------------
#SBATCH --job-name=emp_imperfect_pt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/emp_imperfect_pt_slurm/%A_%a.out
#SBATCH --error=logs/emp_imperfect_pt_slurm/%A_%a.err
# -------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKER="$SCRIPT_DIR/compute_emp_imperfect_pt_worker.py"

PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-results/emp_imperfect_pt}"

# Forward extra args to the worker by setting WORKER_ARGS env var, e.g.:
#   sbatch --export=ALL,WORKER_ARGS="--epochs 2000000 --lr 0.25" ...
WORKER_ARGS="${WORKER_ARGS:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "$TASK_ID" ]]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit as an array job." >&2
  exit 1
fi

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR" "logs/emp_imperfect_pt_slurm"

echo "Starting task $TASK_ID on host $(hostname)"
echo "Repo root:  $REPO_ROOT"
echo "Python:     $PYTHON"
echo "Worker:     $WORKER"
echo "Output dir: $OUTPUT_DIR"
if [[ -n "$WORKER_ARGS" ]]; then
  echo "Worker args: $WORKER_ARGS"
fi

# shellcheck disable=SC2086
exec "$PYTHON" "$WORKER" --task-id "$TASK_ID" --output-dir "$OUTPUT_DIR" $WORKER_ARGS

