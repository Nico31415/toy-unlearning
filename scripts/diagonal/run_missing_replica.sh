#!/bin/bash
#SBATCH --job-name=replica_fix
#SBATCH --output=logs/replica_fix_%a.out
#SBATCH --error=logs/replica_fix_%a.err
#SBATCH --array=0-28
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=icelake

# Create logs directory
mkdir -p logs

cd /home/na658/multi-task2

# Activate environment if needed
source ~/.bashrc
conda activate multi-task 2>/dev/null || true

echo "Running task $SLURM_ARRAY_TASK_ID"
python scripts/diagonal/run_missing_replica.py --task-id $SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID completed with exit code $?"

