#!/bin/bash
#SBATCH --job-name=cpt_gam
#SBATCH --output=logs/cpt_gam_%a.out
#SBATCH --error=logs/cpt_gam_%a.err
#SBATCH --array=0-24
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

$PY scripts/diagonal/run_cpt_eq_gamma_experiments.py --task-id $SLURM_ARRAY_TASK_ID --num-seeds 3

echo "Task $SLURM_ARRAY_TASK_ID completed with exit code $?"

