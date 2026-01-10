#!/bin/bash
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/step2a_fix_%A_%a.out
#SBATCH --error=logs/step2a_fix_%A_%a.err

set -euo pipefail

echo "[SLURM] Job $SLURM_JOB_ID, Array task $SLURM_ARRAY_TASK_ID starting on $(hostname)"

cd /home/na658/multi-task2
mkdir -p logs
mkdir -p figures/step2a_validation

PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] Expected Python not found at $PY" >&2
  exit 1
fi

echo "[SLURM] Using Python: $PY"

OUTPUT_DIR="figures/step2a_validation"

# Fixed tests:
# 0-1: Plot B (fixed): omega=0, vary ONLY a_pt (keep lambda_pt=0, gamma_reinit=0, c_pt=0.001)
# 2-4: Plot C (fixed): Compare BG baseline with ptft_oracle omega=0, a_pt=0
#      For matching k values when beta_pt=0: c_ft = 2*c_pt, k = 16*c_pt²
#      BG uses k = 4*c² = 4*(0.001)² = 4e-6
#      To match: 16*c_pt² = 4e-6 → c_pt² = 2.5e-7 → c_pt = 0.0005
#      But let's just use same c_pt=0.001 and compare shapes, not absolute values

case $SLURM_ARRAY_TASK_ID in
  # ===== Plot B (fixed): a_pt irrelevance when omega=0 =====
  # Keep lambda_pt=0, gamma_reinit=0, c_pt=0.001 fixed, only vary a_pt
  0)
    echo "[Plot B fixed] omega=0, a_pt=1.0 (baseline)"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 0.0 \
      --a_pt 1.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;
  1)
    echo "[Plot B fixed] omega=0, a_pt=10.0 (only a_pt changed)"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 0.0 \
      --a_pt 10.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;

  # ===== Plot C (fixed): BG baseline vs ptft_oracle with a_pt=0 =====
  # With a_pt=0, all coordinates have beta_pt=0, so k values only depend on c_pt
  2)
    echo "[Plot C fixed] BG baseline rho=0.04, c=0.001"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode bg \
      --rho 0.04 \
      --c_values 0.001 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;
  3)
    echo "[Plot C fixed] ptft_oracle omega=0, a_pt=0 (no PT signal)"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 0.0 \
      --a_pt 0.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;
  4)
    echo "[Plot C fixed] ptft_oracle omega=0, a_pt=0, rho_pt=0.04 (same as rho_ft)"
    # This makes PTONLY = 0%, so structure matches BG more closely
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.04 --rho_ft 0.04 --omega 0.0 \
      --a_pt 0.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;

  *)
    echo "[SLURM] Unknown array task ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

echo "[SLURM] Job $SLURM_JOB_ID, Array task $SLURM_ARRAY_TASK_ID finished"



