#!/bin/bash
#SBATCH --job-name=fullnet_recovery
#SBATCH --partition=icelake
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-971
#SBATCH --chdir=/home/na658/multi-task2
#SBATCH --output=logs/fullnet_recovery_%A_%a.out
#SBATCH --error=logs/fullnet_recovery_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/na658/multi-task2}"
PY="${PY:-/home/na658/.conda/envs/mtl_ft/bin/python}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/full_network_recovery_correlated_overlap}"

mkdir -p "$OUT_DIR" "$REPO_ROOT/logs"
cd "$REPO_ROOT"

"$PY" experiments/diagonal/replica/compute_gd_recovery_worker.py \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --output-dir "$OUT_DIR" \
  --variants full_keep_w
