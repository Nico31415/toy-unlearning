#!/bin/bash
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/replica_curves_parallel_%a.out
#SBATCH --error=logs/replica_curves_parallel_%a.err
#SBATCH --partition=icelake
#SBATCH --job-name=rep_curves

# Parallel replica curves for:
# omega in [0.0, 1.0]
# lambda_pt in [-0.00099, 0, 0.00099]
# gamma_reinit in [0.0, 1.0]

PY="/home/na658/.conda/envs/mtl_ft/bin/python"
REPO_ROOT="/home/na658/multi-task2"

mkdir -p "$REPO_ROOT/logs"
mkdir -p "$REPO_ROOT/results/replica_ptft_parallel"

cd "$REPO_ROOT"

# Add current directory to PYTHONPATH to ensure ptft_replica_qk is found
export PYTHONPATH="$REPO_ROOT/experiments/diagonal/replica:$PYTHONPATH"

echo "============================================"
echo "Replica computation: Task ID $SLURM_ARRAY_TASK_ID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo "============================================"

$PY experiments/diagonal/replica/compute_replica_curves_worker.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --output-dir "results/replica_ptft_parallel"

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"
