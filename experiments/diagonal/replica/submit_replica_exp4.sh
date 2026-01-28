#!/bin/bash
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --array=0-239
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=icelake
#SBATCH --job-name=rep_exp4

PY="/home/na658/.conda/envs/mtl_ft/bin/python"
REPO_ROOT="/home/na658/multi-task2"

mkdir -p "$REPO_ROOT/logs"
mkdir -p "$REPO_ROOT/results/replica_slt_parallel_exp4"

cd "$REPO_ROOT"

# Add current directory to PYTHONPATH to ensure ptft_replica_qk is found
export PYTHONPATH="$REPO_ROOT/experiments/diagonal/replica:$PYTHONPATH"

echo "============================================"
echo "Replica (Exp4) computation: Task ID $SLURM_ARRAY_TASK_ID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo "============================================"

$PY experiments/diagonal/replica/compute_replica_curves_worker_exp4.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --n-alpha-chunks 5 \
    --output-dir "results/replica_slt_parallel_exp4"

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"
