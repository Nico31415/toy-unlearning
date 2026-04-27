#!/bin/bash
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --array=0-219
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=icelake
#SBATCH --job-name=emp_forgetting

PY="/home/na658/.conda/envs/mtl_ft/bin/python"
REPO_ROOT="/home/na658/multi-task2"

mkdir -p "$REPO_ROOT/logs"
mkdir -p "$REPO_ROOT/results/forgetting"

cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/experiments/diagonal/replica:$PYTHONPATH"

echo "============================================"
echo "Forgetting experiment: Task ID $SLURM_ARRAY_TASK_ID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo "============================================"

$PY experiments/diagonal/replica/compute_emp_forgetting_worker.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --output-dir "results/forgetting"

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"
