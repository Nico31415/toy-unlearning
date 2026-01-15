#!/bin/bash
#SBATCH --partition=icelake
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --array=0-107
#SBATCH --output=logs/step1_mixture_phase1_%A_%a.out
#SBATCH --error=logs/step1_mixture_phase1_%A_%a.err
#SBATCH --job-name=step1_ph1

# PHASE 1: Step 1 Mixture Sweep with updated stopping logic
# Total jobs: 3 pi_A values × 12 alpha values × 3 seeds = 108 jobs (0-107)

set -euo pipefail

echo "============================================================"
echo "[PHASE 1] Step 1 Mixture Sweep - Preflight Check"
echo "============================================================"
echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID starting on $(hostname)"
echo "[SLURM] Date: $(date)"

# Ensure we run from project root
cd /home/na658/multi-task2

# Create directories
mkdir -p logs
mkdir -p results/diagonal_phase1/step1_mixture
mkdir -p figures/validation_step1_phase1

# Use absolute Python from conda env
PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] ERROR: Expected Python not found at $PY" >&2
  exit 1
fi

echo "[SLURM] Using Python: $PY"
echo "[SLURM] Python version: $($PY --version)"

# Git commit hash (if repo is git)
if [ -d ".git" ]; then
  GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
  echo "[SLURM] Git commit: $GIT_HASH"
fi

echo "============================================================"
echo "[SLURM] Starting Phase 1 Step 1 task $SLURM_ARRAY_TASK_ID"
echo "============================================================"

# Run the Phase 1 sweep script
$PY experiments/diagonal/step1_mixture_sweep_phase1.py $SLURM_ARRAY_TASK_ID

echo "============================================================"
echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID finished"
echo "[SLURM] Date: $(date)"
echo "============================================================"



