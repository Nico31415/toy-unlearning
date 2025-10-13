#!/bin/sh
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --array=0-126
#SBATCH --output=slurm/slurm-%A_%a.out

python experiments/diagonal/diagonal_network_pretrain_1.py $SLURM_ARRAY_TASK_ID
