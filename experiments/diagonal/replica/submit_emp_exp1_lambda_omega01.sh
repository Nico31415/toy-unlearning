#!/bin/bash
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --array=0-923
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=icelake
#SBATCH --job-name=emp_exp1_lam_om01

PY="/home/na658/.conda/envs/mtl_ft/bin/python"
REPO_ROOT="/home/na658/multi-task2"

mkdir -p "$REPO_ROOT/logs"
mkdir -p "$REPO_ROOT/results/emp_ptft_parallel"

cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/experiments/diagonal/replica:$PYTHONPATH"

echo "============================================"
echo "Empirical Exp1: sweep_lambda at omega in {0,1}: Task ID $SLURM_ARRAY_TASK_ID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo "============================================"

$PY experiments/diagonal/replica/compute_emp_curves_worker_exp1_lambda_omega01.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --output-dir "results/emp_ptft_parallel"

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"

