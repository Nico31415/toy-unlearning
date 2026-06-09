#!/bin/bash
#SBATCH --job-name=emp_sanity_o1
#SBATCH --partition=icelake
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-494
#SBATCH --chdir=/home/na658/multi-task2
#SBATCH --output=logs/emp_sanity_o1_%A_%a.out
#SBATCH --error=logs/emp_sanity_o1_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/na658/multi-task2}"
PY="${PY:-/home/na658/.conda/envs/mtl_ft/bin/python}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/sanity_check_omega1}"

mkdir -p "$OUT_DIR" "$REPO_ROOT/logs"
cd "$REPO_ROOT"

"$PY" experiments/diagonal/replica/compute_emp_sanity_check_worker.py \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --omega 1.0 \
  --regime-iv-lambda-mult -0.95 \
  --output-dir "$OUT_DIR"
