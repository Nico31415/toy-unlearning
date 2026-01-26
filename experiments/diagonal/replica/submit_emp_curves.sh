#!/bin/bash
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --array=0-200
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=icelake
#SBATCH --job-name=emp_curves

# Parallel empirical curves for:
# omega in [0.0, 1.0]
# lambda_pt in [-0.00099, 0, 0.00099]
# gamma_reinit in [0.0, 1.0]
# alpha in linspace(0.255, 0.5, 5)



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

$PY experiments/diagonal/replica/compute_emp_curves_worker_high_lr.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --output-dir "results/emp_ptft_parallel"

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"
