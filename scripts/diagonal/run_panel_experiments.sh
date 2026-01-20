#!/bin/bash
#SBATCH --job-name=panel_exp
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-54
#SBATCH --output=logs/panel_exp_%A_%a.out
#SBATCH --error=logs/panel_exp_%A_%a.err

# Panel experiments for 3-panel figure
# 55 unique configurations

cd /home/na658/multi-task2
source ~/.bashrc
conda activate mtl_ft

echo "=== Panel Experiment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

python scripts/diagonal/run_panel_experiments.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --num-seeds 3

echo ""
echo "End time: $(date)"
echo "✓ Completed"



