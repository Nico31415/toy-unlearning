#!/bin/bash
#SBATCH --job-name=corr_scratch_metrics
#SBATCH --partition=icelake
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --chdir=/home/na658/multi-task2
#SBATCH --output=logs/corr_scratch_metrics_%j.out
#SBATCH --error=logs/corr_scratch_metrics_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/na658/multi-task2}"
PY="${PY:-/home/na658/.conda/envs/mtl_ft/bin/python}"

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/results/forgetting"
cd "$REPO_ROOT"

"$PY" experiments/diagonal/replica/compute_correlated_overlap_with_scratch_metrics.py \
  --output-csv results/forgetting/correlated_overlap_with_scratch_metrics.csv
