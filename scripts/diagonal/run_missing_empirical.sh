#!/bin/bash
#SBATCH --job-name=emp_fix
#SBATCH --output=logs/emp_fix_%a.out
#SBATCH --error=logs/emp_fix_%a.err
#SBATCH --array=0-23
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=icelake

# Create logs directory
mkdir -p logs

cd /home/na658/multi-task2

# Use absolute Python from conda env (more reliable than conda activate in SLURM)
PY="/home/na658/.conda/envs/mtl_ft/bin/python"

echo "Running task $SLURM_ARRAY_TASK_ID"
echo "Using Python: $PY"
$PY --version

$PY scripts/diagonal/run_missing_empirical.py --task-id $SLURM_ARRAY_TASK_ID --num-seeds 3

echo "Task $SLURM_ARRAY_TASK_ID completed with exit code $?"
