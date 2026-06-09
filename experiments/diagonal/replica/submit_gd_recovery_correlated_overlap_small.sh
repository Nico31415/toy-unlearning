#!/bin/bash
#SBATCH --job-name=gd_recovery_sm
#SBATCH --partition=icelake
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-755
#SBATCH --chdir=/home/na658/multi-task2
#SBATCH --output=logs/gd_recovery_sm_%A_%a.out
#SBATCH --error=logs/gd_recovery_sm_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/na658/multi-task2}"
PY="${PY:-/home/na658/.conda/envs/mtl_ft/bin/python}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/gd_recovery_correlated_overlap_small}"

mkdir -p "$OUT_DIR" "$REPO_ROOT/logs"
cd "$REPO_ROOT"

"$PY" experiments/diagonal/replica/compute_gd_recovery_worker_small.py \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --output-dir "$OUT_DIR"
