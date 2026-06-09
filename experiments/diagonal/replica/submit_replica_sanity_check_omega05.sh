#!/bin/bash
#SBATCH --job-name=rep_sanity_o05
#SBATCH --partition=icelake
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-29
#SBATCH --chdir=/home/na658/multi-task2
#SBATCH --output=logs/rep_sanity_o05_%A_%a.out
#SBATCH --error=logs/rep_sanity_o05_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/na658/multi-task2}"
PY="${PY:-/home/na658/.conda/envs/mtl_ft/bin/python}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/replica_sanity_check_omega05}"

mkdir -p "$OUT_DIR" "$REPO_ROOT/logs"
cd "$REPO_ROOT"

"$PY" experiments/diagonal/replica/compute_replica_sanity_check_worker.py \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --omega 0.5 \
  --output-dir "$OUT_DIR" \
  --n-alpha-chunks 5 \
  --mc 80000
