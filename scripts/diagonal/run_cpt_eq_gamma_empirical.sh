#!/bin/bash
#SBATCH --job-name=cpt_gam_emp
#SBATCH --output=logs/cpt_gam_emp_%a.out
#SBATCH --error=logs/cpt_gam_emp_%a.err
#SBATCH --array=0-29
#SBATCH --time=02:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=icelake

mkdir -p logs

cd /home/na658/multi-task2

# Use absolute Python path
PY="/home/na658/.conda/envs/mtl_ft/bin/python"

echo "=== Task $SLURM_ARRAY_TASK_ID started at $(date) ==="
echo "Using Python: $PY"

$PY scripts/diagonal/run_cpt_eq_gamma_empirical.py --task-id $SLURM_ARRAY_TASK_ID

EXIT_CODE=$?
echo "=== Task $SLURM_ARRAY_TASK_ID finished at $(date) with exit code $EXIT_CODE ==="
