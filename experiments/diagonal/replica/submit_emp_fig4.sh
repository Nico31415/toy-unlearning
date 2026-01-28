#!/bin/bash
#SBATCH --job-name=emp_fig4
#SBATCH --output=logs/emp_fig4_%a.out
#SBATCH --error=logs/emp_fig4_%a.err
#SBATCH --array=0-329
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

python compute_emp_fig4_worker.py --task-id $SLURM_ARRAY_TASK_ID
