#!/bin/bash
#SBATCH --job-name=ptft_same_opt
#SBATCH --array=0-54
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/ptft_same_opt_%A_%a.out
#SBATCH --error=logs/ptft_same_opt_%A_%a.err

# ── Edit this path before submitting ──────────────────────────────────────────
SAVE_ROOT="/path/to/results/ptft_same_optimizer"
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p logs

source activate dgl-env   # or: conda activate dgl-env

python experiments/diagonal/slurm_worker_ptft_same_optimizer.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --n_tasks 55 \
    --save_root "$SAVE_ROOT"
