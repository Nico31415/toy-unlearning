#!/bin/bash
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --array=0-1539
#SBATCH --output=logs/emp_curves_parallel_%a.out
#SBATCH --error=logs/emp_curves_parallel_%a.err
#SBATCH --partition=icelake
#SBATCH --job-name=emp_curves





PY="/home/na658/.conda/envs/mtl_ft/bin/python"
REPO_ROOT="/home/na658/multi-task2"

mkdir -p "$REPO_ROOT/logs"
mkdir -p "$REPO_ROOT/results/emp_ptft_parallel"

cd "$REPO_ROOT"

# Add current directory to PYTHONPATH to ensure ptft_empirical_finetune_df is found
export PYTHONPATH="$REPO_ROOT/experiments/diagonal/replica:$PYTHONPATH"

echo "============================================"
echo "Empirical computation: Task ID $SLURM_ARRAY_TASK_ID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo "============================================"

$PY experiments/diagonal/replica/compute_emp_curves_worker_exp1.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --output-dir "results/emp_ptft_parallel"

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"
