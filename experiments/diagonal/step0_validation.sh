#!/bin/bash
#SBATCH --partition=icelake
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/step0_val_%j.out
#SBATCH --error=logs/step0_val_%j.err

set -euo pipefail

echo "[SLURM] Job $SLURM_JOB_ID starting on $(hostname)"
echo "[SLURM] Step 0 Validation: Golden Baseline"

# Ensure we run from project root and logs exist
cd /home/na658/multi-task2
mkdir -p logs
mkdir -p figures/validation_step0

# Use absolute Python from conda env to avoid activation issues on compute nodes
PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] Expected Python not found at $PY" >&2
  exit 1
fi

echo "[SLURM] Using Python: $PY"
echo "[SLURM] Starting Step 0 validation..."

# Run Step 0 validation script
$PY scripts/diagonal/plot_step0_validation.py \
  --rho 0.04 \
  --c_values 0.001 0.5 \
  --ft_regulariser_scale 1e-6 \
  --alpha_min 0.008 \
  --alpha_max 1.0 \
  --alpha_points 100 \
  --mc_samples 50000 \
  --seed 12345 \
  --empirical_dir figures/diagonal/bg_generalization \
  --output_dir figures/validation_step0

echo "[SLURM] Job $SLURM_JOB_ID finished"

