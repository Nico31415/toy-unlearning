#!/bin/bash
#SBATCH --job-name=rep_fig4
#SBATCH --output=logs/rep_fig4_%a.out
#SBATCH --error=logs/rep_fig4_%a.err
#SBATCH --array=0-2
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

python compute_replica_fig4_worker.py --task-id $SLURM_ARRAY_TASK_ID
