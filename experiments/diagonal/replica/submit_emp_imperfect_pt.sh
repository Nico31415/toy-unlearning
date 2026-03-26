#!/bin/bash
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --array=0-199
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=icelake
#SBATCH --job-name=emp_imperfect_pt

PY="/home/na658/.conda/envs/mtl_ft/bin/python"
REPO_ROOT="/home/na658/multi-task2"

mkdir -p "$REPO_ROOT/logs"
mkdir -p "$REPO_ROOT/results/emp_imperfect_pt"

cd "$REPO_ROOT"

# Add current directory to PYTHONPATH to ensure compute_emp_imperfect_pt is found
export PYTHONPATH="$REPO_ROOT/experiments/diagonal/replica:$PYTHONPATH"

echo "============================================"
echo "Empirical imperfect-PT computation: Task ID $SLURM_ARRAY_TASK_ID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo "============================================"

$PY experiments/diagonal/replica/compute_emp_imperfect_pt_worker.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --output-dir "results/emp_imperfect_pt"

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"

