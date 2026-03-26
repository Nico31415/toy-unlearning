#!/bin/bash
#SBATCH --job-name=diag_opt_sweep
#SBATCH --array=0-54                  # 11 alpha_ft × 5 seeds = 55 tasks (0..54)
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/diag_opt_%A_%a.out
#SBATCH --error=logs/diag_opt_%A_%a.err

# ── Edit these paths ──────────────────────────────────────────────────────────
REPO_ROOT="/path/to/multi-task2"
SAVE_ROOT="/path/to/results/optimizer_sweep"
CONDA_ENV="dgl-env"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$REPO_ROOT/logs"
mkdir -p "$SAVE_ROOT"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$REPO_ROOT"

python experiments/diagonal/slurm_worker_optimizer_sweep.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --n_tasks 55 \
    --save_root "$SAVE_ROOT"
