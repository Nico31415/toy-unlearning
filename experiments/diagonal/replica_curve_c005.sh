#!/bin/bash
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err

set -euo pipefail

echo "[SLURM] Job $SLURM_JOB_ID starting on $(hostname)"

# Ensure we run from project root and logs exist
cd /home/na658/multi-task2
mkdir -p logs

# Use absolute Python from conda env to avoid activation issues on compute nodes
PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] Expected Python not found at $PY" >&2
  which python || true
  python --version || true
  exit 1
fi

echo "[SLURM] Using Python: $PY"

# Generate replica curve for c=0.05, rho=0.04
echo "[SLURM] Generating replica theory curve for c=0.05, rho=0.04"
$PY scripts/diagonal/plot_replica_q_bg.py \
  --rho 0.04 \
  --lambda_small 1e-6 \
  --c_values 0.05 \
  --alpha_min 0.008 \
  --alpha_max 1.0 \
  --alpha_points 100 \
  --mc_samples 50000 \
  --max_fp_iters 900 \
  --tol_fp 1e-10 \
  --damp 0.25 \
  --seed 12345 \
  --output_dir figures/diagonal/bg_generalization

echo "[SLURM] Job $SLURM_JOB_ID finished"

