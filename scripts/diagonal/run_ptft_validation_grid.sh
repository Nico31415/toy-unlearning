#!/bin/bash
#SBATCH --partition=icelake
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --array=0-71
#SBATCH --output=logs/ptft_validation_%A_%a.out
#SBATCH --error=logs/ptft_validation_%A_%a.err

# =============================================================================
# PT+FT Oracle Validation Grid: Replica vs Empirical Diagonal Network
# =============================================================================
#
# Grid (72 configurations):
#   c_pt       ∈ {0.001, 0.5}
#   lambda_pt  ∈ {0, -c_pt, +c_pt}
#   omega      ∈ {0, 0.5, 1.0}
#   rho_ft     ∈ {0.04, 0.1}
#   gamma_reinit ∈ {0.0, 0.8}
#
# Each job runs:
#   - 7 alpha values × 3 seeds = 21 empirical training runs
#   - 1 replica curve (100 alpha points, 30k MC samples)
#   - 1 overlay plot
#
# Total: 72 jobs × ~21 runs each = ~1512 empirical runs
# =============================================================================

set -euo pipefail

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID starting on $(hostname)"
echo "[SLURM] PT+FT Validation Grid"
echo "[SLURM] Time: $(date)"

# Ensure we run from project root
cd /home/na658/multi-task2
mkdir -p logs
mkdir -p figures/ptft_validation_grid

# Use absolute Python from conda env
PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] Expected Python not found at $PY" >&2
  exit 1
fi

echo "[SLURM] Using Python: $PY"
echo "[SLURM] Running config $SLURM_ARRAY_TASK_ID / 71"

# Run the grid script with this array task ID
$PY scripts/diagonal/run_ptft_validation_grid.py $SLURM_ARRAY_TASK_ID

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID finished at $(date)"






