#!/bin/bash
#SBATCH --job-name=g1_lpt
#SBATCH --output=logs/g1_lpt_%a.out
#SBATCH --error=logs/g1_lpt_%a.err
#SBATCH --array=0-9
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=icelake

mkdir -p logs

cd /home/na658/multi-task2

# Use absolute Python path
PY="/home/na658/.conda/envs/mtl_ft/bin/python"

echo "Running task $SLURM_ARRAY_TASK_ID"
echo "Using Python: $PY"

$PY scripts/diagonal/run_gamma1_lambda_experiments.py --task-id $SLURM_ARRAY_TASK_ID --num-seeds 3

echo "Task $SLURM_ARRAY_TASK_ID completed with exit code $?"


